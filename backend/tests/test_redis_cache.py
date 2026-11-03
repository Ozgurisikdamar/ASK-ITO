"""Optional local Redis L2 cache: local-only policy, fallback, TTL, key
versioning and the embedding / retrieval / answer integrations.

Integration tests start a throwaway `redis-server` on a free loopback port and
are skipped when the binary is not installed. Fallback tests use a client that
always fails, so they need no server."""
import logging
import shutil
import socket
import subprocess
import time

import pytest

from app.config import settings
from app.llm.policy import RemoteEndpointError
from app.redis_cache import RedisCache, RedisUnavailable, cache, check_local_redis_url, hash_key


# -- local-only URL policy ------------------------------------------------------

@pytest.mark.parametrize("url", [
    "redis://localhost:6379/0", "redis://127.0.0.1:6379/1", "redis://[::1]:6379/0",
    "redis://:secret@127.0.0.1:6379/0", "unix:///var/run/redis.sock",
])
def test_loopback_redis_urls_are_accepted(url):
    check_local_redis_url(url)


@pytest.mark.parametrize("url", [
    "redis://8.8.8.8:6379/0", "redis://10.0.0.5:6379/0", "redis://192.168.1.2:6379",
    "rediss://my-cache.example-cloud.com:6380/0", "http://127.0.0.1:6379", "redis://0.0.0.0:6379",
])
def test_hosted_or_lan_redis_is_refused(url):
    with pytest.raises(RemoteEndpointError):  # a hostname resolving to a public address
        check_local_redis_url(url, resolver=_public)


def _public(host):
    return ["34.1.2.3"]


def test_allowed_container_host_must_resolve_to_a_private_address():
    check_local_redis_url("redis://redis:6379/0", ["redis"], resolver=lambda h: ["172.18.0.4"])
    with pytest.raises(RemoteEndpointError):
        check_local_redis_url("redis://redis:6379/0", ["redis"], resolver=lambda h: ["3.3.3.3"])
    with pytest.raises(RemoteEndpointError):  # not listed
        check_local_redis_url("redis://redis:6379/0", [], resolver=lambda h: ["172.18.0.4"])


# -- fallback -------------------------------------------------------------------

class _Broken:
    def get(self, *a, **k):
        raise ConnectionError("connection refused")

    set = ping = get

    def scan_iter(self, *a, **k):
        raise ConnectionError("connection refused")


@pytest.fixture
def broken(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", True)
    monkeypatch.setattr(settings, "redis_strict", False)
    c = RedisCache(client_factory=_Broken)
    return c


def test_disabled_cache_is_a_no_op(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", False)
    c = RedisCache(client_factory=lambda: pytest.fail("must not connect"))
    assert c.get_json("ret", "k") is None and c.set_json("ret", "k", {"a": 1}) is False


def test_unreachable_redis_falls_back_with_one_warning(broken, caplog):
    with caplog.at_level(logging.WARNING, logger="askito.redis"):
        assert broken.get_json("ret", "k") is None
        assert broken.set_json("ret", "k", {"a": 1}) is False  # circuit open: no retry
        assert broken.get_bytes("emb", "k") is None
    assert caplog.text.count("redis cache unavailable") == 1
    assert broken.status()["available"] is False
    assert "connection refused" not in (broken.status()["last_error"] or "")


def test_strict_mode_raises_instead_of_falling_back(broken, monkeypatch):
    monkeypatch.setattr(settings, "redis_strict", True)
    with pytest.raises(RedisUnavailable):
        broken.get_json("ret", "k")


def test_refused_url_disables_the_cache_without_crashing(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", True)
    monkeypatch.setattr(settings, "redis_strict", False)
    monkeypatch.setattr(settings, "redis_url", "redis://8.8.8.8:6379/0")
    c = RedisCache()
    assert c.get_json("ret", "k") is None and "RemoteEndpointError" in c.last_error


def test_retrieval_cache_works_when_redis_is_down(monkeypatch):
    from app.retrieval import pipeline

    monkeypatch.setattr(settings, "redis_enabled", True)
    monkeypatch.setattr(pipeline, "_cache", pipeline.OrderedDict())
    broken_cache = RedisCache(client_factory=_Broken)
    monkeypatch.setattr("app.redis_cache.cache", broken_cache)
    res = pipeline.PipelineResult("search", [(1, 0.5)], [], None, {}, ranked=[(1, 0.5)])
    pipeline.cache_put("q", "search", None, 5, res)
    hit = pipeline.cache_get("q", "search", None, 5)
    assert hit is not None and hit.cache_tier == "l1"


# -- real local redis-server ------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def redis_server():
    exe = shutil.which("redis-server")
    if exe is None:
        pytest.skip("redis-server not installed")
    pytest.importorskip("redis")
    port = _free_port()
    proc = subprocess.Popen([exe, "--port", str(port), "--bind", "127.0.0.1", "--save", "",
                             "--appendonly", "no"], stdout=subprocess.DEVNULL)
    import redis

    client = redis.Redis(port=port)
    for _ in range(50):
        try:
            client.ping()
            break
        except redis.ConnectionError:
            time.sleep(0.1)
    yield f"redis://127.0.0.1:{port}/0", client
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture
def live(redis_server, monkeypatch):
    url, client = redis_server
    client.flushdb()
    monkeypatch.setattr(settings, "redis_enabled", True)
    monkeypatch.setattr(settings, "redis_strict", False)
    monkeypatch.setattr(settings, "redis_url", url)
    cache.reset()
    yield client
    cache.reset()


def test_json_round_trip_with_ttl(live, monkeypatch):
    monkeypatch.setattr(settings, "redis_ttl_retrieval_s", 120)
    assert cache.set_json("ret", "abc", {"final": [[1, 0.5]]})
    assert cache.get_json("ret", "abc") == {"final": [[1, 0.5]]}
    ttl = live.ttl(cache.full_key("ret", "abc"))
    assert 0 < ttl <= 120
    assert cache.counters["ret"]["hits"] == 1 and cache.counters["ret"]["sets"] == 1


def test_oversized_values_are_not_stored(live, monkeypatch):
    monkeypatch.setattr(settings, "redis_max_value_bytes", 100)
    assert cache.set_bytes("ret", "big", b"x" * 101) is False
    assert cache.counters["ret"]["too_large"] == 1 and cache.get_bytes("ret", "big") is None


def test_clear_removes_only_this_apps_keys_of_one_kind(live):
    live.set("someone-else:key", b"1")
    cache.set_json("ret", "a", 1)
    cache.set_json("emb", "b", 2)
    assert cache.clear("ret") == 1
    assert cache.get_json("emb", "b") == 2 and live.get("someone-else:key") == b"1"


def test_concurrent_writers_do_not_corrupt_entries(live):
    import threading

    def work(i):
        for j in range(20):
            cache.set_json("tool", f"k{j}", {"v": j})
            assert cache.get_json("tool", f"k{j}") == {"v": j}

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert cache.counters["tool"]["errors"] == 0


def test_retrieval_l2_round_trip_and_invalidation(live, monkeypatch):
    from app.retrieval import pipeline

    monkeypatch.setattr(pipeline, "_cache", pipeline.OrderedDict())
    step = pipeline.RetrievalStep("bm25", 3, [7, 9], [1.0, 0.5])
    res = pipeline.PipelineResult("search", [(7, 1.0), (9, 0.5)], [step], None,
                                  {"bm25": 3}, ranked=[(7, 1.0), (9, 0.5)],
                                  counts={"final": 2}, embedding_source="model")
    pipeline.cache_put("Quantum computing", "search", None, 5, res)
    pipeline._cache.clear()  # a fresh process: only Redis has it
    hit = pipeline.cache_get("quantum   computing", "search", None, 5)
    assert hit is not None and hit.cache_tier == "l2" and hit.cached
    assert hit.final == [(7, 1.0), (9, 0.5)] and hit.trace[0].candidate_ids == [7, 9]
    assert pipeline.cache_get("quantum computing", "search", None, 5).cache_tier == "l1"
    # a rebuilt index (new index version) is a different key -> miss
    pipeline._cache.clear()
    monkeypatch.setattr(pipeline, "index_version", lambda: "rebuilt")
    assert pipeline.cache_get("quantum computing", "search", None, 5) is None


def test_query_embedding_l2(live, monkeypatch):
    from app.index import embed

    embed.embedding_cache_clear()
    vec = [0.1] * settings.embedding_dim
    embed.store_query_embedding("what is a qubit", vec)
    embed.embedding_cache_clear()  # drop L1
    got, tier = embed.cached_query_embedding("what  is a qubit")
    assert tier == "l2" and got == pytest.approx(vec, abs=1e-6)
    assert embed.cached_query_embedding("what is a qubit")[1] == "l1"
    # another embedding model or instruction never reuses the vector
    embed.embedding_cache_clear()
    monkeypatch.setattr(settings, "query_instruction", "other: ")
    assert embed.cached_query_embedding("what is a qubit") == (None, None)


def test_answer_front_cache_keeps_citations_and_checks_sources(live, monkeypatch):
    from app.retrieval import answer_cache

    monkeypatch.setattr(settings, "answer_cache_enabled", True)
    monkeypatch.setattr(answer_cache, "get_connection", _no_db)
    cites = [{"n": 1, "chunk_id": 11, "doc_title": "Qubit"}]
    answer_cache.put("what is a qubit", "overview", "all", "A qubit [1].", cites, [11, 12])
    hit = answer_cache.get("what is a qubit", "overview", "all", source_ids=[11, 12])
    assert hit["cache_kind"] == "redis" and hit["citations"] == cites
    assert hit["answer"] == "A qubit [1]."
    # different (or reordered) sources: [1] would point elsewhere -> miss
    assert answer_cache.get("what is a qubit", "overview", "all", source_ids=[12, 11]) is None


def test_answer_key_changes_with_model_quantization_and_prompt(monkeypatch):
    from app.llm import identity
    from app.retrieval import answer_cache

    base = answer_cache.make_key("q", "overview", "all")
    monkeypatch.setattr(identity, "model_tag", lambda role="answer": "ollama:m:Q8_0:abc")
    q8 = answer_cache.make_key("q", "overview", "all")
    monkeypatch.setattr(identity, "model_tag", lambda role="answer": "ollama:m:Q4_K_M:def")
    q4 = answer_cache.make_key("q", "overview", "all")
    monkeypatch.setattr(settings, "prompt_version", "v-next")
    q4_new_prompt = answer_cache.make_key("q", "overview", "all")
    assert len({base, q8, q4, q4_new_prompt}) == 4


def _no_db():
    raise RuntimeError("database not available in this test")


def test_hash_key_is_stable():
    assert hash_key(("a", 1)) == hash_key(("a", 1)) != hash_key(("a", 2))
