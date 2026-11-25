"""Regression tests for attached files under /var/tmp, file budgets and Vertex 429 retries.

Seen 2026-11-25: chat with gemini-4-neon dropped files under /var/tmp without saying so,
refused a 1.4 MB file against a 268k token limit on a 1M-token model, and gave up on the
first Vertex 429 RESOURCE_EXHAUSTED.
"""

from pathlib import Path

import pytest

from providers.gemini import GeminiModelProvider
from utils.file_utils import check_total_file_size, read_file_content
from utils.security_config import is_dangerous_path


def test_var_tmp_files_are_allowed_but_rest_of_var_is_not():
    assert not is_dangerous_path(Path("/var/tmp/claude/x/brief.md"))
    assert is_dangerous_path(Path("/var/tmp"))
    assert is_dangerous_path(Path("/var/log/syslog"))
    assert is_dangerous_path(Path("/var/tmpfoo/x"))
    assert is_dangerous_path(Path("/etc/passwd"))


def test_vertex_throttling_429_is_retried():
    provider = GeminiModelProvider(api_key="test")
    throttled = RuntimeError(
        'HTTP 429: {"error": {"code": 429, "message": "Resource exhausted. Please try again later.", '
        '"status": "RESOURCE_EXHAUSTED"}}'
    )
    assert provider._is_error_retryable(throttled)


def test_hard_size_errors_are_not_retried():
    provider = GeminiModelProvider(api_key="test")
    assert not provider._is_error_retryable(RuntimeError("429 RESOURCE_EXHAUSTED: request too large"))
    assert not provider._is_error_retryable(RuntimeError("429 quota exceeded: requests per day"))


def test_one_megabyte_file_is_read_whole(tmp_path):
    f = tmp_path / "big.md"
    f.write_text("x" * 1_200_000)
    content, _ = read_file_content(str(f))
    assert "File too large" not in content


def test_neon_accepts_a_1_4_mb_file(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from utils import model_context

    # Neon's real context window, without depending on which providers this run registered.
    monkeypatch.setattr(
        model_context.ModelContext,
        "capabilities",
        property(lambda self: SimpleNamespace(context_window=1_048_576)),
    )
    f = tmp_path / "slice.md"
    f.write_text("y" * 1_425_179)
    assert check_total_file_size([str(f)], "gemini-4-neon") is None


def test_missing_attached_file_fails_before_the_model_call(tmp_path):
    from tools.chat import ChatRequest, ChatTool

    tool = ChatTool()
    req = ChatRequest(
        prompt="review",
        absolute_file_paths=[str(tmp_path / "missing.md")],
        working_directory_absolute_path=str(tmp_path),
        model="gemini-4-neon",
    )
    error = tool._validate_file_paths(req)
    assert error and "not found" in error


@pytest.mark.parametrize("path", ["/etc/passwd", "/var/log/syslog"])
def test_refused_attached_file_fails_loudly(path, tmp_path):
    from tools.chat import ChatRequest, ChatTool

    req = ChatRequest(
        prompt="review",
        absolute_file_paths=[path],
        working_directory_absolute_path=str(tmp_path),
        model="gemini-4-neon",
    )
    error = ChatTool()._validate_file_paths(req)
    assert error and "refused" in error
