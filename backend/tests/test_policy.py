"""Strict local-only (loopback) LLM endpoint policy."""
import httpx
import pytest

from app.llm import client as llm
from app.llm.policy import RemoteEndpointError, ensure_local_url


def _resolver(mapping):
    def resolve(host):
        if host not in mapping:
            raise OSError("no such host")
        return mapping[host]
    return resolve


DNS = _resolver({
    "localhost": ["127.0.0.1", "::1"],
    "ollama": ["172.18.0.3"],
    "ollama6": ["fd00::12"],
    "evil": ["127.0.0.1", "93.184.216.34"],   # mixed loopback + public
    "public": ["93.184.216.34"],
    "lan-box": ["192.168.1.20"],
    "linklocal": ["169.254.10.10"],
})


@pytest.mark.parametrize("url", [
    "http://localhost:11434",
    "http://127.0.0.1:11434/v1",
    "https://127.1.2.3",
    "http://[::1]:11434",
    "http://[::ffff:127.0.0.1]:11434",
])
def test_loopback_endpoints_are_accepted(url):
    ensure_local_url(url, resolver=DNS)


@pytest.mark.parametrize("url", [
    "http://10.0.0.5:11434",            # RFC 1918
    "http://192.168.1.20:11434",
    "http://172.16.4.4:11434",
    "http://169.254.169.254/latest",    # link-local / cloud metadata
    "http://[fe80::1]:11434",
    "http://0.0.0.0:11434",             # wildcard
    "http://[::]:11434",
    "http://224.0.0.1",                 # multicast
    "http://8.8.8.8",                   # public
    "https://api.example.com/v1",
    "http://evil:11434",                # mixed DNS answer
    "http://public:11434",
    "http://lan-box:11434",             # private, not allow-listed
    "http://ollama:11434",              # docker name, not allow-listed
    "http://nowhere.invalid",           # unresolvable
])
def test_non_loopback_endpoints_are_refused(url):
    with pytest.raises(RemoteEndpointError):
        ensure_local_url(url, resolver=DNS)


@pytest.mark.parametrize("url", [
    "ftp://localhost/x",
    "file:///etc/passwd",
    "http://user:secret@localhost:11434",
    "http://localhost:99999",
    "http://",
    "not a url",
])
def test_malformed_or_unsafe_urls_are_refused(url):
    with pytest.raises(RemoteEndpointError):
        ensure_local_url(url, resolver=DNS)


def test_allowed_host_may_resolve_to_private_network_only():
    ensure_local_url("http://ollama:11434", allowed_hosts=["ollama"], resolver=DNS)
    ensure_local_url("http://OLLAMA:11434", allowed_hosts=["ollama"], resolver=DNS)
    ensure_local_url("http://ollama6:11434", allowed_hosts=["ollama6"], resolver=DNS)
    for host in ("public", "evil", "linklocal"):
        with pytest.raises(RemoteEndpointError):
            ensure_local_url(f"http://{host}:11434", allowed_hosts=[host], resolver=DNS)
    # the allow-list names hostnames, never IP literals
    with pytest.raises(RemoteEndpointError):
        ensure_local_url("http://192.168.1.20", allowed_hosts=["192.168.1.20"], resolver=DNS)
    # an allow-listed name does not widen the policy for other names
    with pytest.raises(RemoteEndpointError):
        ensure_local_url("http://lan-box:11434", allowed_hosts=["ollama"], resolver=DNS)


def test_make_client_and_judge_enforce_policy(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_local_only", True)
    monkeypatch.setattr(llm.settings, "llm_allowed_hosts", "")
    with pytest.raises(RemoteEndpointError):
        llm.make_client("openai", "http://192.168.1.20:8080/v1", "k", "m")
    with pytest.raises(RemoteEndpointError):
        llm.make_client("ollama", "http://8.8.8.8:11434", "")
    assert isinstance(llm.make_client("ollama", "http://127.0.0.1:11434", ""), llm.OllamaClient)

    from app.eval import ragas_eval

    monkeypatch.setattr(ragas_eval.settings, "llm_mode", "ollama")
    monkeypatch.setattr(ragas_eval.settings, "llm_base_url", "http://10.1.2.3:11434")
    with pytest.raises(RemoteEndpointError):
        ragas_eval.judge_client()


def test_policy_can_be_disabled_explicitly(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_local_only", False)
    assert isinstance(llm.make_client("openai", "http://8.8.8.8/v1", "k", "m"),
                      llm.OpenAICompatClient)


def test_requests_are_revalidated_and_redirects_not_followed(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_local_only", True)
    monkeypatch.setattr(llm.settings, "llm_allowed_hosts", "")
    c = llm.OllamaClient("http://127.0.0.1:11434", default_model="m")
    assert c._http.follow_redirects is False
    # a request to another host through the same client is refused before sending
    with pytest.raises(RemoteEndpointError):
        c._http.get("http://8.8.8.8:11434/api/tags")


def test_redirect_responses_are_returned_not_followed(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_local_only", True)
    hops = []

    def handler(req):
        hops.append(str(req.url))
        return httpx.Response(302, headers={"Location": "http://8.8.8.8/steal"})

    c = llm.OllamaClient("http://127.0.0.1:11434", default_model="m")
    c._http = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False,
                           event_hooks={"request": [llm._check_request_url]})
    with pytest.raises(llm.LLMError):
        c._post("http://127.0.0.1:11434/api/chat", {"model": "m"})
    assert hops == ["http://127.0.0.1:11434/api/chat"]
