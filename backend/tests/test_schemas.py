from app.agent.schemas import Analysis, Sufficiency, call_json, extract_json


def test_extract_json():
    assert extract_json('noise {"a": 1} tail') == {"a": 1}
    assert extract_json("no json") is None
    assert extract_json('[1,2,3]') is None  # not an object


class _LLM:
    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages, temperature=0.0, model=None, **kwargs):
        return self.replies.pop(0)

    def ping(self):
        return True, "fake"


def test_call_json_valid():
    llm = _LLM(['{"qtype": "multi_hop", "subquestions": ["a", "b"]}'])
    a = call_json(llm, "p", Analysis)
    assert a.qtype == "multi_hop" and a.subquestions == ["a", "b"]


def test_call_json_retries_then_fallback():
    llm = _LLM(["garbage", "still bad"])
    assert call_json(llm, "p", Sufficiency) is None  # 1 retry, both invalid


def test_call_json_second_attempt_succeeds():
    llm = _LLM(["garbage", '{"sufficient": true}'])
    s = call_json(llm, "p", Sufficiency)
    assert s is not None and s.sufficient is True
