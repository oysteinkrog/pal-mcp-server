"""Tests for Gemini 4 requests: thinking level, continuation tokens and empty replies."""

from unittest.mock import MagicMock, patch

import pytest
from google.genai import types

from providers.gemini import GeminiModelProvider


def _http_reply(status=200, text="", finish="STOP", token=None, prompt=100, out=10, thoughts=5):
    reply = MagicMock()
    reply.status_code = status
    reply.text = f"status {status}"
    candidate = {"finishReason": finish, "content": {"parts": [{"text": "thinking", "thought": True}, {"text": text}]}}
    if token:
        candidate["continuationToken"] = token
    reply.json.return_value = {
        "candidates": [candidate],
        "usageMetadata": {"promptTokenCount": prompt, "candidatesTokenCount": out, "thoughtsTokenCount": thoughts},
    }
    return reply


def _call_gemini4(replies, thinking_mode="medium"):
    provider = GeminiModelProvider("test-key")
    http = MagicMock()
    http.post.side_effect = replies
    client_cls = MagicMock()
    client_cls.return_value.__enter__.return_value = http
    with patch("providers.gemini.httpx.Client", client_cls), patch("providers.gemini.time.sleep"):
        result = provider.generate_content(prompt="hi", model_name="argon", thinking_mode=thinking_mode)
    return result, [c.kwargs["json"] for c in http.post.call_args_list]


@pytest.mark.parametrize(
    "mode,level",
    [("minimal", "LOW"), ("low", "LOW"), ("medium", "MEDIUM"), ("high", "MEDIUM"), ("max", "MEDIUM")],
)
def test_gemini4_sends_thinking_level(mode, level):
    _, bodies = _call_gemini4([_http_reply(text="ok")], thinking_mode=mode)
    config = bodies[0]["generationConfig"]
    assert config["thinkingConfig"] == {"thinkingLevel": level}
    assert config["maxOutputTokens"] == GeminiModelProvider.GEMINI4_MAX_OUTPUT_TOKENS


def test_gemini4_follows_continuation_tokens():
    replies = [
        _http_reply(text="", finish="CONTINUATION", token="t1", out=0, thoughts=32768),
        _http_reply(text="part one ", finish="CONTINUATION", token="t2", out=20000, thoughts=0),
        _http_reply(text="part two", finish="STOP", out=300, thoughts=0),
    ]
    result, bodies = _call_gemini4(replies)
    assert result.content == "part one part two"
    assert "continuationToken" not in bodies[0]
    assert [b.get("continuationToken") for b in bodies[1:]] == ["t1", "t2"]
    assert all(b["contents"] == bodies[0]["contents"] for b in bodies)
    assert result.usage["input_tokens"] == 100
    assert result.usage["output_tokens"] == 32768 + 20000 + 300
    assert result.metadata["continuation_requests"] == 3
    assert result.metadata["finish_reason"] == "STOP"


def test_gemini4_retries_busy_continuation_without_restarting():
    replies = [
        _http_reply(text="", finish="CONTINUATION", token="t1"),
        _http_reply(status=503),
        _http_reply(text="done", finish="STOP"),
    ]
    result, bodies = _call_gemini4(replies)
    assert result.content == "done"
    assert [b.get("continuationToken") for b in bodies] == [None, "t1", "t1"]


def test_gemini4_empty_reply_raises_clear_error():
    with pytest.raises(RuntimeError, match="returned no text") as exc:
        _call_gemini4([_http_reply(text="", finish="MAX_TOKENS")])
    assert "after 1 attempt:" in str(exc.value)


def test_older_gemini_still_uses_sdk_with_budget():
    provider = GeminiModelProvider("test-key")
    sdk_reply = MagicMock()
    sdk_reply.text = "ok"
    sdk_reply.candidates = [MagicMock(finish_reason=types.FinishReason.STOP, safety_ratings=[])]
    sdk_reply.usage_metadata = None
    client = MagicMock()
    client.models.generate_content.return_value = sdk_reply
    with patch.object(GeminiModelProvider, "client", client):
        provider.generate_content(prompt="hi", model_name="pro", thinking_mode="high")
    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.thinking_config.thinking_budget == int(32768 * GeminiModelProvider.THINKING_BUDGETS["high"])
