"""Tests gateway fallback before agent initialization."""

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.session import SessionSource


class _CapturingAgent:
    last_init = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []

    def run_conversation(self, user_message: str, conversation_history=None, task_id=None):
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
        }


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = {
        "provider": "ollama",
        "model": "qwen3.5:9b",
        "base_url": "http://localhost:11434/v1",
    }
    runner._running_agents = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._session_db = None
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._resolve_turn_agent_config = lambda _msg, model, runtime_kwargs: {
        "model": model,
        "runtime": runtime_kwargs,
    }
    return runner


def _make_source():
    return SessionSource(
        platform=Platform.LOCAL,
        chat_id="cli",
        chat_name="CLI",
        chat_type="dm",
        user_id="user-1",
    )


@pytest.mark.asyncio
async def test_run_agent_uses_fallback_model_when_primary_runtime_resolution_fails(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(gateway_run, "_env_path", hermes_home / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda: "openai/gpt-5")

    def _raise_primary_auth_error():
        raise RuntimeError("Codex token refresh failed with status 401")

    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", _raise_primary_auth_error)

    import hermes_cli.runtime_provider as runtime_provider

    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **kwargs: {
            "provider": kwargs["requested"],
            "api_mode": "chat_completions",
            "base_url": kwargs.get("explicit_base_url"),
            "api_key": kwargs.get("explicit_api_key"),
            "source": "explicit",
        },
    )

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    _CapturingAgent.last_init = None
    runner = _make_runner()

    result = await runner._run_agent(
        message="ping",
        context_prompt="",
        history=[],
        source=_make_source(),
        session_id="session-1",
        session_key="agent:main:local:dm",
    )

    assert result["final_response"] == "ok"
    assert _CapturingAgent.last_init is not None
    assert _CapturingAgent.last_init["model"] == "qwen3.5:9b"
    assert _CapturingAgent.last_init["provider"] == "ollama"
    assert _CapturingAgent.last_init["base_url"] == "http://localhost:11434/v1"
