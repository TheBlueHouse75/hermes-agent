"""Profile credentials stay isolated unless both sides grant a shared OAuth authority."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest


def _make_auth_store(pool: dict | None = None, providers: dict | None = None) -> dict:
    store: dict = {"version": 1}
    if pool is not None:
        store["credential_pool"] = pool
    if providers is not None:
        store["providers"] = providers
    return store


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """Global root + an active named profile under Path.home()/.hermes/profiles/coder.

    * Path.home() -> tmp_path
    * Global root -> tmp_path/.hermes            (has its own auth.json fixture)
    * Profile     -> tmp_path/.hermes/profiles/coder   (active, HERMES_HOME points here)
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    global_root = tmp_path / ".hermes"
    global_root.mkdir()
    profile_dir = global_root / "profiles" / "coder"
    profile_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_dir))
    return {"global": global_root, "profile": profile_dir}


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _shared_credentials_config(policy: dict) -> dict:
    return {"auth": {"shared_credentials": policy}}


def _grant_shared_providers(profile_env: dict[str, Path], *provider_ids: str) -> None:
    providers = list(provider_ids)
    _write(
        profile_env["profile"] / "config.yaml",
        _shared_credentials_config({"authority": "default", "providers": providers}),
    )
    _write(
        profile_env["global"] / "config.yaml",
        _shared_credentials_config({"consumers": {"coder": providers}}),
    )


_ROOT_STORE = _make_auth_store(
    providers={
        "nous": {"access_token": "nous-root", "refresh_token": "rt-root"},
        "xai-oauth": {"auth_mode": "oauth_pkce",
                      "tokens": {"access_token": "xai-root-access", "refresh_token": "xai-root-refresh"}},
    },
    pool={
        "openrouter": [{"id": "glob-1", "label": "root", "auth_type": "api_key", "priority": 0,
                        "source": "manual", "access_token": "sk-root"}],
        "openai-codex": [{"id": "glob-codex", "auth_type": "oauth", "priority": 0,
                          "source": "manual:device_code", "access_token": "root-codex-access",
                          "refresh_token": "root-codex-refresh"}],
    },
)


@pytest.mark.parametrize("malformed_access_token", [{}, [], 123])
@pytest.mark.parametrize("credential_shape", ["singleton", "pool"])
def test_provider_credential_probe_rejects_malformed_access_tokens(
    profile_env, credential_shape, malformed_access_token,
):
    from hermes_cli.auth_shared_authority import has_provider_credentials

    provider_id = "openai-codex"
    if credential_shape == "singleton":
        store = _make_auth_store(
            providers={provider_id: {"tokens": {"access_token": malformed_access_token}}}
        )
    else:
        store = _make_auth_store(
            pool={provider_id: [{"access_token": malformed_access_token}]}
        )
    _write(profile_env["profile"] / "auth.json", store)

    assert has_provider_credentials(provider_id) is False


def test_shared_auth_authority_requires_bilateral_provider_grant(profile_env):
    from hermes_cli.auth_shared_authority import resolve_auth_store_authority
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    local_store = profile_env["profile"] / "auth.json"
    root_store = profile_env["global"] / "auth.json"

    # No consumer request, including an owner-only grant, preserves isolation.
    assert resolve_auth_store_authority("openai-codex").store_path == local_store
    _write(
        profile_env["global"] / "config.yaml",
        _shared_credentials_config({"consumers": {"coder": ["openai-codex"]}}),
    )
    assert resolve_auth_store_authority("openai-codex").store_path == local_store

    # A bilateral grant shares only the exact requested providers.
    _grant_shared_providers(profile_env, "openai-codex", "xai-oauth")
    for provider_id in ("openai-codex", "xai-oauth"):
        authority = resolve_auth_store_authority(provider_id)
        assert authority.store_path == root_store
        assert authority.profile_name == "default"
        assert authority.shared is True

    local = resolve_auth_store_authority("openrouter")
    assert local.store_path == local_store
    assert local.profile_name == "coder"
    assert local.shared is False

    # The default profile never consumes its own grants as shared authority.
    token = set_hermes_home_override(profile_env["global"])
    try:
        default = resolve_auth_store_authority("openai-codex")
        assert default.store_path == root_store
        assert default.profile_name == "default"
        assert default.shared is False
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize(
    ("consumer_policy", "owner_policy"),
    [
        ({"authority": "default", "providers": ["openai-codex"]}, None),
        ({"authority": "default", "providers": "openai-codex"}, None),
        ({"authority": "root", "providers": ["openai-codex"]}, None),
        ({"authority": "default", "providers": ["unknown-oauth"]}, None),
        (
            {"authority": "default", "providers": ["openai-codex"]},
            {"consumers": []},
        ),
        (
            {"authority": "default", "providers": ["openai-codex"]},
            {"consumers": {"coder": ["unknown-oauth"]}},
        ),
        (
            {"authority": "default", "providers": ["openai-codex"]},
            {"consumers": {"coder": ["xai-oauth"]}},
        ),
    ],
)
def test_shared_auth_authority_fails_closed_on_invalid_or_denied_policy(
    profile_env, consumer_policy, owner_policy
):
    from hermes_cli.auth_shared_authority import (
        SharedCredentialAuthorityError,
        resolve_auth_store_authority,
    )

    _write(
        profile_env["profile"] / "config.yaml",
        _shared_credentials_config(consumer_policy),
    )
    if owner_policy is not None:
        _write(
            profile_env["global"] / "config.yaml",
            _shared_credentials_config(owner_policy),
        )

    with pytest.raises(SharedCredentialAuthorityError):
        resolve_auth_store_authority("openai-codex")


def test_shared_provider_slices_use_one_authority_store(profile_env):
    import hermes_cli.auth as auth

    root_file = profile_env["global"] / "auth.json"
    local_file = profile_env["profile"] / "auth.json"
    _write(
        profile_env["profile"] / "config.yaml",
        _shared_credentials_config(
            {"authority": "default", "providers": ["openai-codex", "xai-oauth"]}
        ),
    )
    _write(
        profile_env["global"] / "config.yaml",
        _shared_credentials_config(
            {"consumers": {"coder": ["openai-codex", "xai-oauth"]}}
        ),
    )
    root_store = json.loads(json.dumps(_ROOT_STORE))
    root_store.update(
        active_provider="xai-oauth",
        suppressed_sources={"openai-codex": ["device_code"]},
    )
    root_store["providers"]["openai-codex"] = {
        "tokens": {"access_token": "root-access", "refresh_token": "root-refresh"}
    }
    _write(root_file, root_store)
    local_store = _make_auth_store(
        providers={
            "openai-codex": {"tokens": {"access_token": "local-shadow"}},
            "openrouter": {"access_token": "local-openrouter"},
        },
        pool={
            "openai-codex": [{"id": "local-shadow", "access_token": "local-shadow"}],
            "openrouter": [{"id": "local-openrouter", "access_token": "sk-local"}],
        },
    )
    local_store.update(
        active_provider="openai-codex",
        suppressed_sources={"openai-codex": ["local-only"], "openrouter": ["env:OPENROUTER_API_KEY"]},
    )
    _write(local_file, local_store)

    assert auth.get_provider_auth_state("openai-codex")["tokens"]["access_token"] == "root-access"
    assert auth.get_provider_auth_state("openrouter")["access_token"] == "local-openrouter"
    assert [row["id"] for row in auth.read_credential_pool("openai-codex")] == ["glob-codex"]
    assert {
        provider: [row["id"] for row in rows]
        for provider, rows in auth.read_credential_pool().items()
    } == {
        "openai-codex": ["glob-codex"],
        "openrouter": ["local-openrouter"],
    }
    assert auth.is_source_suppressed("openai-codex", "device_code") is True
    assert auth.is_source_suppressed("openrouter", "env:OPENROUTER_API_KEY") is True
    with auth._provider_state_transaction("openai-codex") as (store, state, source_path):
        assert source_path == root_file
        state = dict(state)
        state["tokens"] = {"access_token": "rotated", "refresh_token": "rotated-refresh"}
        auth._save_provider_state_to_source(store, "openai-codex", state, source_path)

    auth.write_credential_pool(
        "openai-codex",
        [{"id": "root-new", "auth_type": "oauth", "source": "device_code",
          "access_token": "root-new-access"}],
        removed_ids={"glob-codex"},
    )
    auth.suppress_credential_source("xai-oauth", "device_code")
    assert auth.unsuppress_credential_source("openai-codex", "device_code") is True

    persisted_root = json.loads(root_file.read_text())
    persisted_local = json.loads(local_file.read_text())
    assert persisted_root["providers"]["openai-codex"]["tokens"]["access_token"] == "rotated"
    assert [row["id"] for row in persisted_root["credential_pool"]["openai-codex"]] == ["root-new"]
    assert persisted_root["suppressed_sources"] == {"xai-oauth": ["device_code"]}
    assert persisted_root["active_provider"] == "xai-oauth"
    assert persisted_local == local_store
    assert (profile_env["global"] / "auth.lock").exists()
    assert not (profile_env["profile"] / "auth.lock").exists()

    assert auth.clear_provider_auth("openai-codex") is True
    persisted_root = json.loads(root_file.read_text())
    persisted_local = json.loads(local_file.read_text())
    assert "openai-codex" not in persisted_root["providers"]
    assert "openai-codex" not in persisted_root["credential_pool"]
    assert persisted_root["active_provider"] == "xai-oauth"
    assert persisted_local["active_provider"] is None
    assert "openai-codex" in persisted_local["providers"]
    assert "openai-codex" in persisted_local["credential_pool"]


def test_shared_auth_remove_clears_authority_singleton_only(profile_env):
    from types import SimpleNamespace

    from hermes_cli.auth_commands import auth_remove_command

    _grant_shared_providers(profile_env, "xai-oauth")
    root_file = profile_env["global"] / "auth.json"
    _write(root_file, _make_auth_store(
        providers={
            "xai-oauth": {
                "tokens": {"access_token": "root-access", "refresh_token": "root-refresh"},
            },
        },
        pool={
            "xai-oauth": [
                {"id": "singleton", "label": "shared", "auth_type": "oauth", "priority": 0,
                 "source": "device_code", "access_token": "root-access",
                 "refresh_token": "root-refresh"},
                {"id": "manual", "label": "other", "auth_type": "oauth", "priority": 1,
                 "source": "manual:device_code", "access_token": "other-access",
                 "refresh_token": "other-refresh"},
            ],
        },
    ))

    auth_remove_command(SimpleNamespace(provider="xai-oauth", target="singleton"))

    persisted = json.loads(root_file.read_text())
    assert "xai-oauth" not in persisted["providers"]
    assert [row["id"] for row in persisted["credential_pool"]["xai-oauth"]] == ["manual"]


def test_shared_auth_add_unsuppresses_authority_marker(profile_env):
    from hermes_cli.auth_commands import _unsuppress_provider_sources

    _grant_shared_providers(profile_env, "openai-codex")
    root_file = profile_env["global"] / "auth.json"
    local_file = profile_env["profile"] / "auth.json"
    _write(root_file, {
        **_make_auth_store(providers={}),
        "suppressed_sources": {"openai-codex": ["device_code"]},
    })
    _write(local_file, {
        **_make_auth_store(providers={}),
        "suppressed_sources": {"openai-codex": ["local-shadow"]},
    })

    _unsuppress_provider_sources("openai-codex")

    assert "suppressed_sources" not in json.loads(root_file.read_text())
    assert json.loads(local_file.read_text())["suppressed_sources"] == {
        "openai-codex": ["local-shadow"],
    }


@pytest.mark.parametrize(
    ("provider_id", "status_getter", "entry_status"),
    [
        ("xai-oauth", "get_xai_oauth_auth_status", {}),
        (
            "openai-codex",
            "get_codex_auth_status",
            {
                "last_status": "exhausted",
                "last_error_code": 429,
                "last_error_reason": "rate_limit",
                "last_error_reset_at": 4_102_444_800,
            },
        ),
    ],
)
def test_shared_oauth_status_reports_authority_store(
    profile_env, provider_id, status_getter, entry_status,
):
    import hermes_cli.auth as auth

    _grant_shared_providers(profile_env, provider_id)
    root_file = profile_env["global"] / "auth.json"
    _write(root_file, _make_auth_store(
        providers=({provider_id: {"tokens": {
            "access_token": "singleton-access", "refresh_token": "singleton-refresh",
        }}} if provider_id == "xai-oauth" else {}),
        pool={
            provider_id: [{
                "id": "shared", "label": "shared", "auth_type": "oauth", "priority": 0,
                "source": "device_code", "access_token": "root-access",
                "refresh_token": "root-refresh", **entry_status,
            }],
        },
    ))

    before = root_file.read_bytes()
    before_mtime = root_file.stat().st_mtime_ns
    status = getattr(auth, status_getter)()

    assert status["logged_in"] is True
    assert status["auth_store"] == str(root_file)
    assert root_file.read_bytes() == before
    assert root_file.stat().st_mtime_ns == before_mtime
    if provider_id == "openai-codex":
        assert status["rate_limited"] is True


def test_corrupt_shared_store_fails_closed_without_local_fallback(profile_env):
    import hermes_cli.auth as auth

    root_file = profile_env["global"] / "auth.json"
    local_file = profile_env["profile"] / "auth.json"
    _grant_shared_providers(profile_env, "openai-codex")
    corrupt = b"{not valid json"
    root_file.write_bytes(corrupt)
    local_store = _make_auth_store(
        providers={"openai-codex": {"tokens": {"access_token": "local-shadow"}}},
        pool={"openai-codex": [{"id": "local-shadow", "access_token": "local-shadow"}]},
    )
    _write(local_file, local_store)
    local_before = local_file.read_bytes()

    with pytest.raises(auth.AuthError, match="Shared credential store is invalid"):
        auth.get_provider_auth_state("openai-codex")
    with pytest.raises(auth.AuthError, match="Shared credential store is invalid"):
        auth.write_credential_pool("openai-codex", [])

    assert root_file.read_bytes() == corrupt
    assert local_file.read_bytes() == local_before
    assert (profile_env["global"] / "auth.lock").exists()
    assert not (profile_env["profile"] / "auth.lock").exists()


def test_named_profile_never_reads_the_root_store(profile_env):
    """Provider state, pool slices and the whole-pool read all stop at the profile's own auth.json;
    the profile's own rows are what it sees, and a root-only provider resolves to nothing."""
    from hermes_cli.auth import get_provider_auth_state, read_credential_pool, resolve_codex_runtime_credentials
    from hermes_cli.auth_xai import _read_xai_oauth_tokens
    from hermes_cli.auth import AuthError

    _write(profile_env["global"] / "auth.json", _ROOT_STORE)
    _write(profile_env["profile"] / "auth.json", _make_auth_store(
        providers={},
        pool={"openrouter": [{"id": "prof-1", "auth_type": "api_key", "priority": 0,
                              "source": "manual", "access_token": "sk-profile"}]},
    ))

    assert get_provider_auth_state("nous") is None
    assert read_credential_pool("openai-codex") == []
    assert [e["id"] for e in read_credential_pool("openrouter")] == ["prof-1"]
    assert set(read_credential_pool()) == {"openrouter"}
    with pytest.raises(AuthError):
        _read_xai_oauth_tokens()
    with pytest.raises(AuthError):
        resolve_codex_runtime_credentials(refresh_if_expiring=False)

    # Control: a home that owns the store (single-profile layout) resolves its credentials.
    own = profile_env["profile"].parent / "owner"
    own.mkdir()
    _write(own / "auth.json", _ROOT_STORE)
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(own)
    try:
        assert get_provider_auth_state("nous")["access_token"] == "nous-root"
        assert read_credential_pool("openrouter")[0]["id"] == "glob-1"
    finally:
        reset_hermes_home_override(token)


def test_profile_refresh_never_writes_through_to_the_root_store(profile_env):
    """A token save inside a profile lands in the profile's auth.json; the root store is
    byte-identical afterwards (xAI OAuth save, Codex save, pool write, cooldown clear)."""
    from hermes_cli.auth import write_credential_pool
    from hermes_cli.auth_codex import _save_codex_tokens, clear_codex_pool_quota_cooldowns
    from hermes_cli.auth_xai import _save_xai_oauth_tokens

    root_file = profile_env["global"] / "auth.json"
    root_store = json.loads(json.dumps(_ROOT_STORE))
    root_store["credential_pool"]["openai-codex"][0].update(
        last_status="exhausted", last_error_reason="rate_limit", last_error_reset_at=4_102_444_800)
    _write(root_file, root_store)
    _write(profile_env["profile"] / "auth.json", _make_auth_store(providers={}, pool={}))
    before = root_file.read_bytes()

    _save_xai_oauth_tokens({"access_token": "xai-new", "refresh_token": "xai-new-rt"}, set_active=False)
    _save_codex_tokens({"access_token": "codex-new", "refresh_token": "codex-new-rt"})
    write_credential_pool("openrouter", [{"id": "prof-new", "auth_type": "api_key", "priority": 0,
                                          "source": "manual", "access_token": "sk-profile-new"}])
    assert clear_codex_pool_quota_cooldowns() == 0  # the root's exhausted row is not ours to clear

    assert root_file.read_bytes() == before
    profile_store = json.loads((profile_env["profile"] / "auth.json").read_text())
    assert profile_store["providers"]["xai-oauth"]["tokens"]["refresh_token"] == "xai-new-rt"
    assert profile_store["providers"]["openai-codex"]["tokens"]["refresh_token"] == "codex-new-rt"
    assert [e["id"] for e in profile_store["credential_pool"]["openrouter"]] == ["prof-new"]


@pytest.mark.parametrize(
    ("provider_id", "resolver_name", "refresh_name"),
    [
        ("openai-codex", "resolve_codex_runtime_credentials", "refresh_codex_oauth_pure"),
        ("xai-oauth", "resolve_xai_oauth_runtime_credentials", "refresh_xai_oauth_pure"),
    ],
)
def test_shared_authority_runtime_refresh_rotates_root_tokens_only(
    profile_env, monkeypatch, provider_id, resolver_name, refresh_name,
):
    """A bilateral grant rotates the authority pair under its lock, never the profile shadow."""
    import hermes_cli.auth as auth

    root_file = profile_env["global"] / "auth.json"
    local_file = profile_env["profile"] / "auth.json"
    _grant_shared_providers(profile_env, provider_id)
    root_store = _make_auth_store(providers={
        provider_id: {
            "auth_mode": "oauth_device_code",
            "tokens": {"access_token": "root-access", "refresh_token": "root-refresh"},
            **({"discovery": {"token_endpoint": "https://auth.x.ai/oauth/token"}}
               if provider_id == "xai-oauth" else {}),
        },
    })
    local_store = _make_auth_store(providers={
        provider_id: {"tokens": {"access_token": "profile-access", "refresh_token": "profile-refresh"}},
    })
    _write(root_file, root_store)
    _write(local_file, local_store)
    local_before = local_file.read_bytes()
    refresh_calls: list[tuple[str, str]] = []
    save_lock_depths: list[tuple[int, int]] = []
    save_auth_store = auth._save_auth_store

    def _save_while_locked(store, *, target_path=None):
        if target_path == root_file:
            save_lock_depths.append((
                getattr(auth._auth_lock_holder_for(root_file), "depth", 0),
                getattr(auth._auth_lock_holder_for(local_file), "depth", 0),
            ))
        return save_auth_store(store, target_path=target_path)

    def _refresh(access_token, refresh_token, **_kwargs):
        refresh_calls.append((access_token, refresh_token))
        return {
            "access_token": "rotated-access", "refresh_token": "rotated-refresh",
            "last_refresh": "2026-09-21T00:00:00Z",
        }

    monkeypatch.setattr(auth, refresh_name, _refresh)
    if provider_id == "openai-codex":
        import hermes_cli.auth_codex as auth_codex
        monkeypatch.setattr(auth_codex, refresh_name, _refresh)
    monkeypatch.setattr(auth, "_save_auth_store", _save_while_locked)

    resolved = getattr(auth, resolver_name)(force_refresh=True)

    assert refresh_calls == [("root-access", "root-refresh")]
    assert resolved["api_key"] == "rotated-access"
    persisted_root = json.loads(root_file.read_text())
    assert persisted_root["providers"][provider_id]["tokens"].get("access_token") == "rotated-access"
    assert persisted_root["providers"][provider_id]["tokens"].get("refresh_token") == "rotated-refresh"
    assert persisted_root.get("active_provider") is None
    assert local_file.read_bytes() == local_before
    assert save_lock_depths
    assert all(root_depth > 0 and profile_depth == 0 for root_depth, profile_depth in save_lock_depths)


@pytest.mark.parametrize("provider_id", ["openai-codex", "xai-oauth"])
def test_shared_named_login_keeps_authority_selection_and_sets_local_choice(profile_env, provider_id):
    """A shared credential save is authority data; only the initiating login picks locally."""
    from hermes_cli.auth_codex import _save_codex_tokens
    from hermes_cli.auth_xai import _save_xai_oauth_tokens

    _grant_shared_providers(profile_env, provider_id)
    root_file = profile_env["global"] / "auth.json"
    local_file = profile_env["profile"] / "auth.json"
    _write(root_file, {"version": 1, "active_provider": "anthropic", "providers": {}})
    _write(local_file, {"version": 1, "active_provider": "openrouter", "providers": {}})

    if provider_id == "openai-codex":
        _save_codex_tokens({"access_token": "new-access", "refresh_token": "new-refresh"})
    else:
        _save_xai_oauth_tokens({"access_token": "new-access", "refresh_token": "new-refresh"})

    root = json.loads(root_file.read_text())
    local = json.loads(local_file.read_text())
    assert root["active_provider"] == "anthropic"
    assert local["active_provider"] == provider_id


def test_malformed_or_missing_root_store_never_breaks_a_profile_read(profile_env):
    from hermes_cli.auth import read_credential_pool

    (profile_env["global"] / "auth.json").write_text("{not valid json")
    _write(profile_env["profile"] / "auth.json", _make_auth_store(pool={
        "openrouter": [{"id": "prof-1", "auth_type": "api_key", "priority": 0,
                        "source": "manual", "access_token": "sk-profile"}]}))
    assert read_credential_pool("openrouter")[0]["id"] == "prof-1"


def test_auth_lock_reentrancy_is_scoped_after_profile_context_switch(profile_env):
    """Changing profile context cannot inherit another store's lock depth."""
    import hermes_cli.auth as auth
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    profile_b = profile_env["global"] / "profiles" / "reviewer"
    profile_b.mkdir(parents=True)
    profile_b_lock = profile_b / "auth.lock"

    with auth._auth_store_lock():
        holder_a = auth._auth_lock_holder_for(profile_env["profile"] / "auth.json")
        assert getattr(holder_a, "depth", 0) == 1

        token = set_hermes_home_override(profile_b)
        try:
            holder_b = auth._auth_lock_holder_for(profile_b / "auth.json")
            assert holder_b is not holder_a
            assert getattr(holder_b, "depth", 0) == 0
            assert not profile_b_lock.exists()

            with auth._auth_store_lock():
                assert profile_b_lock.exists()
                assert getattr(holder_b, "depth", 0) == 1
        finally:
            reset_hermes_home_override(token)

    assert getattr(holder_a, "depth", 0) == 0


# ---------------------------------------------------------------------------
# write_credential_pool — stale-snapshot cooldown merge
# ---------------------------------------------------------------------------


@pytest.fixture()
def classic_env(tmp_path, monkeypatch):
    """Classic single-root layout (HERMES_HOME != ~/.hermes, no profiles)."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    hermes_home = tmp_path / "classic"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def _pool_entry(**overrides) -> dict:
    entry = {
        "id": "cred-x",
        "label": "key-x",
        "auth_type": "api_key",
        "priority": 0,
        "source": "manual",
        "access_token": "sk-x",
    }
    entry.update(overrides)
    return entry


def test_write_pool_never_merges_cooldown_onto_reauthed_entry(classic_env):
    """A token change means re-auth: the old cooldown must never carry over.

    A fresh login intentionally clears the entry's status; resurrecting the
    stale cooldown onto the new credentials would bench a just-authorized key.
    """
    from hermes_cli.auth import write_credential_pool

    _write(classic_env / "auth.json", _make_auth_store(pool={
        "openrouter": [_pool_entry(
            access_token="sk-old",
            last_status="exhausted",
            last_status_at=time.time() - 60,  # newer AND unexpired
            last_error_code=429,
        )],
    }))

    # Same entry id, freshly re-authed with a new token and cleared status.
    write_credential_pool("openrouter", [_pool_entry(access_token="sk-new")])

    data = json.loads((classic_env / "auth.json").read_text())
    persisted = data["credential_pool"]["openrouter"][0]
    assert persisted["access_token"] == "sk-new"
    assert persisted.get("last_status") != "exhausted"
    assert persisted.get("last_error_code") is None
