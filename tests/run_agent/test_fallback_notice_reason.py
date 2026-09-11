"""The "⚠️ Model fallback" notice must name the real cause of the switch.

Four ``_try_activate_fallback()`` call sites in ``agent/conversation_loop.py``
used to pass no reason, so the user-visible notice said "provider failure"
even when the loop knew the cause (safety refusal, content-filter stream
termination, non-retryable HTTP error, retry exhaustion).  Each test drives the real loop through one of those branches
with the real ``try_activate_fallback`` and checks the recorded notice.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_constants import FINISH_REASON_LENGTH, PARTIAL_STREAM_STUB_ID
from run_agent import AIAgent
from tests.run_agent.test_run_agent import _mock_assistant_msg, _mock_response

FALLBACK = {"provider": "zai", "model": "glm-4.7"}
RESOLVE = "agent.auxiliary_client.resolve_provider_client"


def _fallback_client(recovery):
    """Client handed out by the fallback resolver; answers the retried call."""
    client = MagicMock()
    client.base_url = "https://api.z.ai/api/paas/v4"
    client.api_key = "fb-key"
    client.chat.completions.create.return_value = recovery
    return client


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            model="primary/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=FALLBACK,
        )
        a.client = MagicMock()
        a.provider = "openrouter"
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.compression_enabled = False
        a.save_trajectories = False
        return a


def _run(agent, recovery):
    """Run one turn; return (result, lifecycle notices emitted on success).

    A successful recovery drains ``_pending_fallback_notice`` through
    ``_emit_pending_fallback_notice`` -> ``_emit_status``, so the notice is
    observed there rather than on the attribute.
    """
    emitted = []
    with (
        patch(RESOLVE, return_value=(_fallback_client(recovery), "glm-4.7")),
        patch("time.sleep"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_emit_status", side_effect=emitted.append),
    ):
        return agent.run_conversation("hello"), emitted


def _notice(primary_provider, reason_text):
    return (
        f"⚠️ Model fallback: primary/model via {primary_provider} unavailable "
        f"({reason_text}); using glm-4.7 via zai."
    )


class _HttpError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code
        self.response = SimpleNamespace(headers={})
        self.body = {"error": {"message": message}}


def test_content_filter_refusal_names_content_policy(agent):
    agent.client.chat.completions.create.return_value = _mock_response(
        content="", finish_reason="content_filter"
    )
    result, notices = _run(agent, _mock_response(content="fallback answer"))

    assert result["final_response"] == "fallback answer"
    assert _notice("openrouter", "content policy blocked the request") in notices


def test_content_filter_terminated_stream_names_content_policy(agent):
    agent.client.chat.completions.create.return_value = SimpleNamespace(
        id=PARTIAL_STREAM_STUB_ID,
        model="primary/model",
        choices=[SimpleNamespace(
            index=0,
            message=_mock_assistant_msg(content="Writing the file..."),
            finish_reason=FINISH_REASON_LENGTH,
        )],
        usage=None,
        _dropped_tool_names=["write_file"],
        _content_filter_terminated=True,
    )
    result, notices = _run(agent, _mock_response(content="fallback answer"))

    assert result["final_response"].endswith("fallback answer")
    assert _notice("openrouter", "content policy blocked the request") in notices


def test_non_retryable_http_error_names_classified_reason(agent):
    agent.client.chat.completions.create.side_effect = _HttpError(
        "This content was flagged for possible cybersecurity risk.", 400
    )
    result, notices = _run(agent, _mock_response(content="fallback answer"))

    assert result["final_response"] == "fallback answer"
    assert _notice("openrouter", "content policy blocked the request") in notices


def test_retry_exhaustion_names_classified_reason(agent):
    agent._api_max_retries = 2
    agent.client.chat.completions.create.side_effect = _HttpError(
        "Internal server error", 500
    )
    with patch.object(agent, "_try_recover_primary_transport", return_value=False):
        result, notices = _run(agent, _mock_response(content="fallback answer"))

    assert result["final_response"] == "fallback answer"
    assert _notice("openrouter", "provider server error") in notices
