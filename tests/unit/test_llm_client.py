"""Unit tests for the caching LLM client: keys, cache hits, retries, usage, and cost."""

import pytest

from meridian.common.config import get_models_config
from meridian.common.llm_client import (
    LLMClient,
    MemoryResponseCache,
    RetryableError,
    SqliteResponseCache,
    cache_key,
    strip_json_fences,
    strip_reasoning,
    usage_scope,
)
from tests.fakes import ScriptedTransport

pytestmark = pytest.mark.unit

MESSAGES = [{"role": "user", "content": "hello"}]


def test_cache_key_depends_on_model_messages_and_params():
    base = cache_key("m", MESSAGES, {"temperature": 0.0})
    assert base == cache_key("m", list(MESSAGES), {"temperature": 0.0})
    assert len(base) == 64
    assert base != cache_key("m2", MESSAGES, {"temperature": 0.0})
    assert base != cache_key("m", [{"role": "user", "content": "hi"}], {"temperature": 0.0})
    assert base != cache_key("m", MESSAGES, {"temperature": 0.5})


def test_cache_hit_skips_the_transport(tmp_path):
    transport = ScriptedTransport(lambda model, messages, params: "answer")
    client = LLMClient(get_models_config(), transport, SqliteResponseCache(tmp_path / "c.sqlite"))
    first = client.chat("router", MESSAGES)
    second = client.chat("router", MESSAGES)
    assert first.text == second.text == "answer"
    assert not first.o_cached
    assert second.o_cached
    assert len(transport.request_list) == 1
    client.chat("router", MESSAGES, o_use_cache=False)
    assert len(transport.request_list) == 2


def test_role_parameters_are_sent():
    transport = ScriptedTransport(lambda model, messages, params: "{}")
    client = LLMClient(get_models_config(), transport, MemoryResponseCache())
    client.chat("generator", MESSAGES, json_mode=True)
    model, _messages, params = transport.request_list[0]
    generator = get_models_config().role("generator")
    assert model == generator.model
    assert params["reasoning_effort"] == generator.reasoning_effort
    assert params["json_mode"] is True
    client.chat("judge", MESSAGES)
    assert "reasoning_effort" not in transport.request_list[1][2]


def test_retry_honors_retry_after_then_succeeds():
    attempt_list: list[int] = []

    def responder(model, messages, params):
        attempt_list.append(1)
        if len(attempt_list) < 3:
            raise RetryableError("429", retry_after_s=7.0)
        return "ok"

    slept: list[float] = []
    client = LLMClient(get_models_config(), ScriptedTransport(responder), None, sleep=slept.append)
    assert client.chat("router", MESSAGES).text == "ok"
    assert len(slept) == 2
    assert all(delay >= 7.0 for delay in slept)


def test_retries_are_bounded():
    def responder(model, messages, params):
        raise RetryableError("503")

    client = LLMClient(
        get_models_config(), ScriptedTransport(responder), None, sleep=lambda s: None
    )
    with pytest.raises(RetryableError):
        client.chat("router", MESSAGES)


def test_usage_scope_counts_calls_and_cost():
    client = LLMClient(
        get_models_config(), ScriptedTransport(lambda *_: "x"), MemoryResponseCache()
    )
    with usage_scope() as outer:
        client.chat("router", MESSAGES)
        with usage_scope() as inner:
            client.chat("generator", MESSAGES)
            client.chat("generator", MESSAGES)
    assert outer.calls == 3
    assert inner.calls == 2
    assert inner.uncached_calls == 1
    assert outer.calls_by_role() == {"router": 1, "generator": 2}
    generator_model = get_models_config().role("generator").model
    price = get_models_config().prices_usd_per_million_tokens[generator_model]
    expected = 2 * (100 * price.input + 20 * price.output) / 1e6
    assert inner.cost_usd(get_models_config()) == pytest.approx(expected)


def test_streaming_delivers_deltas_and_cached_text_once():
    client = LLMClient(
        get_models_config(), ScriptedTransport(lambda *_: "a b c"), MemoryResponseCache()
    )
    delta_list: list[str] = []
    client.chat("generator", MESSAGES, on_delta=delta_list.append)
    assert "".join(delta_list).strip() == "a b c"
    cached_delta_list: list[str] = []
    client.chat("generator", MESSAGES, on_delta=cached_delta_list.append)
    assert cached_delta_list == ["a b c"]


def test_chat_json_parses_fenced_objects_and_rejects_others():
    client = LLMClient(
        get_models_config(),
        ScriptedTransport(lambda *_: '```json\n{"question": "q"}\n```'),
        MemoryResponseCache(),
    )
    assert client.chat_json("router", MESSAGES) == {"question": "q"}
    assert strip_json_fences('{"a": 1}') == '{"a": 1}'
    bad = LLMClient(get_models_config(), ScriptedTransport(lambda *_: "[1]"), None)
    with pytest.raises(ValueError):
        bad.chat_json("router", MESSAGES)


def test_inline_reasoning_is_stripped_before_caching():
    reply = '<think>step one\nstep two</think>\n{"supported": true}'
    assert strip_reasoning(reply) == '{"supported": true}'
    cache = MemoryResponseCache()
    client = LLMClient(get_models_config(), ScriptedTransport(lambda *_: reply), cache)
    assert client.chat_json("judge", MESSAGES) == {"supported": True}
    assert client.chat("judge", MESSAGES).text == '{"supported": true}'
