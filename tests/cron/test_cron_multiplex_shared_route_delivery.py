"""Regression tests for #101113 — a credentialless satellite profile under
``gateway.profile_routes`` delivers cron output through the PRIMARY adapter
for exactly the targets the primary routes to it, and fails closed otherwise.

The multiplex ticker hands such a profile a ``SharedRouteAdapters`` view over
the primary adapter map; ``_deliver_result`` resolves a transport from it per
target using the same ``ProfileRoute.matches`` predicate as inbound routing.
"""
import asyncio
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest
import yaml

from cron.scheduler import (
    SharedRouteAdapters,
    _deliver_result,
    _primary_profile_routes_for_current_home,
)
from gateway.config import Platform, PlatformConfig
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

PRIMARY_YAML = {
    "gateway": {
        "multiplex_profiles": True,
        "profile_routes": [
            {"name": "fit", "platform": "discord", "chat_id": "1543065293755256852", "profile": "fitness"},
            {"name": "fit-thread", "platform": "discord", "chat_id": "155", "thread_id": "42", "profile": "fitness"},
            {"name": "off", "platform": "discord", "chat_id": "999", "profile": "fitness", "enabled": False},
            {"name": "other", "platform": "discord", "chat_id": "777", "profile": "other"},
        ],
    }
}


def _job(chat_id: str) -> dict:
    return {"id": "a7ae1520356c", "name": "brief", "deliver": f"discord:{chat_id}"}


def _run(
    job,
    adapters,
    *,
    native_configured=True,
    loop_running=True,
    future_timeout=None,
    platform_config=None,
):
    """Drive ``_deliver_result`` with a live loop and a real DeliveryRouter."""
    loop = MagicMock()
    loop.is_running.return_value = loop_running

    def fake_run_coro(coro, _loop):
        if future_timeout == "pending":
            coro.close()
            future = MagicMock()
            future.result.side_effect = TimeoutError
            future.done.return_value = False
            future.cancel.return_value = True
            return future
        if future_timeout == "completed":
            coro.close()
            future = Future()
            future.set_exception(TimeoutError("adapter timeout"))
            return future
        if future_timeout == "completed_success_after_timeout":
            coro.close()
            future = MagicMock()
            future.done.return_value = True
            future.result.side_effect = [
                TimeoutError(),
                {"success": True, "message_id": "m1"},
            ]
            return future
        future = Future()
        future.set_result(asyncio.run(coro))
        return future

    standalone = []

    async def _fake_send_to_platform(platform, pconfig, chat_id, text, **kwargs):
        standalone.append(chat_id)
        return {"success": False, "error": "DISCORD_BOT_TOKEN is not set"}

    config = MagicMock()
    config.platforms = (
        {Platform.DISCORD: platform_config or PlatformConfig(enabled=True)}
        if native_configured
        else {}
    )
    config.get_home_channel = lambda p: None
    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("tools.send_message_tool._send_to_platform", _fake_send_to_platform), \
         patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
        error = _deliver_result(job, "hello", adapters=adapters, loop=loop)
    return error, standalone


def _primary_adapter(outcome="success"):
    adapter = MagicMock()
    adapter.sent = []

    async def send(chat_id, content, metadata=None):
        adapter.sent.append(chat_id)
        if outcome == "exception":
            raise RuntimeError("primary adapter failed")
        if outcome == "failure":
            return {"success": False, "error": "primary adapter rejected send"}
        return {"success": True, "message_id": "m1"}

    adapter.send = send
    return adapter


def test_satellite_routes_exact_target_through_primary_adapter(tmp_path, monkeypatch):
    root = tmp_path / "root"
    fitness_home = root / "profiles" / "fitness"
    fitness_home.mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump(PRIMARY_YAML), encoding="utf-8")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    primary = _primary_adapter()

    token = set_hermes_home_override(str(fitness_home))
    try:
        shared = SharedRouteAdapters(
            {Platform.DISCORD: primary}, _primary_profile_routes_for_current_home()
        )
        # exact enabled route → primary adapter sends, no standalone attempt
        error, standalone = _run(_job("1543065293755256852"), shared)
        assert error is None, error
        assert primary.sent == ["1543065293755256852"]
        assert standalone == []

        # unmatched chat, disabled route, route for another profile → fail
        # closed before any primary or standalone transport can send.
        for chat in ("424242", "999", "777"):
            primary.sent.clear()
            error, standalone = _run(_job(chat), shared)
            assert error is not None and "outside the authorized scope" in error
            assert primary.sent == []
            assert standalone == []
    finally:
        reset_hermes_home_override(token)


def test_satellite_route_does_not_require_native_platform_config(tmp_path, monkeypatch):
    """A credentialless profile has no native platform block by design."""
    root = tmp_path / "root"
    fitness_home = root / "profiles" / "fitness"
    fitness_home.mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump(PRIMARY_YAML), encoding="utf-8")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    primary = _primary_adapter()

    token = set_hermes_home_override(str(fitness_home))
    try:
        shared = SharedRouteAdapters(
            {Platform.DISCORD: primary}, _primary_profile_routes_for_current_home()
        )
        error, standalone = _run(
            _job("1543065293755256852"),
            shared,
            native_configured=False,
        )
        assert error is None, error
        assert primary.sent == ["1543065293755256852"]
        assert standalone == []
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize(
    ("outcome", "loop_running", "future_timeout"),
    [
        ("success", False, None),
        ("failure", True, None),
        ("exception", True, None),
        ("success", True, "pending"),
        ("success", True, "completed"),
    ],
)
def test_satellite_route_never_falls_back_to_standalone(
    tmp_path,
    monkeypatch,
    outcome,
    loop_running,
    future_timeout,
):
    root = tmp_path / "root"
    fitness_home = root / "profiles" / "fitness"
    fitness_home.mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump(PRIMARY_YAML), encoding="utf-8")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    primary = _primary_adapter(outcome)

    token = set_hermes_home_override(str(fitness_home))
    try:
        shared = SharedRouteAdapters(
            {Platform.DISCORD: primary}, _primary_profile_routes_for_current_home()
        )
        error, standalone = _run(
            _job("1543065293755256852"),
            shared,
            native_configured=False,
            loop_running=loop_running,
            future_timeout=future_timeout,
        )
        if future_timeout == "pending":
            assert error is None
        else:
            assert error is not None
        assert standalone == []
    finally:
        reset_hermes_home_override(token)


def test_thread_scoped_route_cannot_flatten_into_parent_channel(tmp_path, monkeypatch):
    root = tmp_path / "root"
    fitness_home = root / "profiles" / "fitness"
    fitness_home.mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump(PRIMARY_YAML), encoding="utf-8")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    primary = _primary_adapter()
    primary.supports_inchannel_continuable = True
    job = {
        "id": "a7ae1520356c",
        "name": "brief",
        "deliver": "origin",
        "origin": {
            "platform": "discord",
            "chat_id": "155",
            "thread_id": "42",
        },
    }

    token = set_hermes_home_override(str(fitness_home))
    try:
        shared = SharedRouteAdapters(
            {Platform.DISCORD: primary}, _primary_profile_routes_for_current_home()
        )
        error, standalone = _run(
            job,
            shared,
            platform_config=PlatformConfig(
                enabled=True,
                extra={"cron_continuable_surface": "in_channel"},
            ),
        )
        assert error is not None and "outside the authorized scope" in error
        assert primary.sent == []
        assert standalone == []
    finally:
        reset_hermes_home_override(token)


def test_shared_view_is_falsy_without_routes_or_primary_adapters():
    assert not SharedRouteAdapters({}, [])
    assert SharedRouteAdapters({Platform.DISCORD: object()}, []).get(Platform.DISCORD) is None


def test_live_adapter_timeout_never_retries_through_standalone():
    """A timed-out live send is indeterminate, so retrying may duplicate it."""
    adapter = _primary_adapter()

    error, standalone = _run(
        _job("1543065293755256852"),
        {Platform.DISCORD: adapter},
        future_timeout="pending",
    )

    assert error is None
    assert standalone == []


def test_live_adapter_success_at_timeout_boundary_is_not_retried():
    adapter = _primary_adapter()

    error, standalone = _run(
        _job("1543065293755256852"),
        {Platform.DISCORD: adapter},
        future_timeout="completed_success_after_timeout",
    )

    assert error is None
    assert standalone == []
