from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

from local_llm_fpl_assistant import llm_client as client


def event(kind, **kwargs):
    return NS(type=kind, **kwargs)


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    factory = MagicMock()
    monkeypatch.setattr(client, "OpenAI", factory)
    instance = factory.return_value.__enter__.return_value
    stream = instance.responses.create.return_value.__enter__.return_value
    return factory, instance, stream


def run():
    return list(client.stream_chat_with_llm(
        system_prompt="Ground answers", user_prompt="Captain?", settings=client.AstraSettings()
    ))


def test_stream_preserves_text_and_ignores_other_events(api):
    factory, instance, stream = api
    stream.__iter__.return_value = iter([
        event("response.created"), event("response.output_text.delta", delta="Mid "),
        event("response.reasoning_summary_text.delta", delta="hidden"),
        event("response.output_text.delta", delta="A"), event("response.completed"),
    ])
    assert run() == ["Mid ", "A"]
    instance.responses.create.assert_called_once_with(
        model="gpt-6-astra", instructions="Ground answers", input="Captain?",
        reasoning={"effort": "low"}, store=False, stream=True,
    )
    factory.return_value.__exit__.assert_called_once()


@pytest.mark.parametrize("events,match", [
    ([event("response.completed")], "no answer text"),
    ([event("response.output_text.delta", delta="partial")], "ended before completion"),
    ([event("response.failed")], "failed"),
    ([event("response.incomplete")], "incomplete"),
    ([event("error", message="test-secret")], "failed"),
    ([event("response.refusal.delta", delta="refused")], "declined"),
])
def test_failed_streams_are_not_successful_answers(api, events, match):
    api[2].__iter__.return_value = iter(events)
    with pytest.raises(client.LLMError, match=match) as exc:
        run()
    assert "test-secret" not in str(exc.value)


def test_missing_key_fails_before_client_creation(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    factory = MagicMock()
    monkeypatch.setattr(client, "OpenAI", factory)
    with pytest.raises(client.LLMError, match="OPENAI_API_KEY"):
        run()
    factory.assert_not_called()


@pytest.mark.parametrize("effort", ["none", "minimal", "invalid"])
def test_invalid_reasoning_is_rejected(effort):
    with pytest.raises(ValueError, match="reasoning"):
        client.AstraSettings(reasoning_effort=effort)


@pytest.mark.parametrize("status,match", [
    (401, "authentication"), (403, "access"), (404, "access"),
    (429, "quota"), (500, "failed"),
])
def test_sdk_http_errors_are_sanitized(api, status, match):
    import httpx
    from openai import APIStatusError
    classes = {401: client.AuthenticationError, 403: client.PermissionDeniedError,
               404: client.NotFoundError, 429: client.RateLimitError}
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
    api[1].responses.create.side_effect = classes.get(status, APIStatusError)(
        "test-secret", response=response, body={"private": "test-secret"}
    )
    with pytest.raises(client.LLMError, match=match) as exc:
        run()
    assert "test-secret" not in str(exc.value)


@pytest.mark.parametrize("timeout", [False, True])
def test_connection_errors_are_sanitized(api, timeout):
    import httpx
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    error = client.APITimeoutError(request=request) if timeout else client.APIConnectionError(request=request)
    api[1].responses.create.side_effect = error
    with pytest.raises(client.LLMError, match="timed out" if timeout else "connect"):
        run()


def test_real_sdk_decodes_responses_stream(monkeypatch):
    import httpx
    import json
    from openai import OpenAI
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    received = []
    def handle(request):
        received.append(json.loads(request.content))
        events = [
            {"type": "response.output_text.delta", "delta": "Captain Mid A.", "item_id": "msg_1",
             "output_index": 0, "content_index": 0, "sequence_number": 1, "logprobs": []},
            {"type": "response.completed", "sequence_number": 2,
             "response": {"id": "resp_1", "status": "completed", "output": []}},
        ]
        body = "".join("data: " + json.dumps(item) + "\n\n" for item in events)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)
    def factory(**kwargs):
        return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    monkeypatch.setattr(client, "OpenAI", factory)
    assert run() == ["Captain Mid A."]
    assert received[0]["model"] == "gpt-6-astra"
    assert received[0]["store"] is False


def test_stream_transport_failure_is_sanitized(api):
    import httpx
    def broken_stream():
        yield event("response.output_text.delta", delta="partial")
        raise httpx.ReadError("test-secret")
    api[2].__iter__.side_effect = broken_stream
    with pytest.raises(client.LLMError, match="interrupted") as exc:
        run()
    assert "test-secret" not in str(exc.value)
