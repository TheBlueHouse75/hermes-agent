"""Resolve the one auth store a provider may use for the active profile."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from hermes_cli.config import InvalidUserConfigError
from hermes_cli.config_effective import load_user_config_effective
from hermes_constants import get_default_hermes_root, get_hermes_home, profile_name_for_home


_SHAREABLE_PROVIDER_IDS = frozenset({"openai-codex", "xai-oauth"})
_INVALID_POLICY_MESSAGE = "Shared credential authority configuration is invalid."
_UNAVAILABLE_AUTHORITY_MESSAGE = "Shared credential authority is unavailable."
_DENIED_AUTHORITY_MESSAGE = "Shared credential authority access is not granted."


class SharedCredentialAuthorityError(InvalidUserConfigError):
    """Raised when an explicitly requested shared credential store is unsafe to use."""


@dataclass(frozen=True)
class AuthStoreAuthority:
    store_path: Path
    profile_name: str
    shared: bool


def _load_config(path: Path) -> dict[str, Any]:
    try:
        return load_user_config_effective(path, fail_closed=True)
    except Exception:
        raise SharedCredentialAuthorityError(_INVALID_POLICY_MESSAGE) from None


def _shared_credentials(config: dict[str, Any]) -> Any:
    auth = config.get("auth")
    if not isinstance(auth, dict) or "shared_credentials" not in auth:
        return None
    return auth["shared_credentials"]


def _provider_list(value: Any) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(provider_id, str) or provider_id not in _SHAREABLE_PROVIDER_IDS
        for provider_id in value
    ):
        raise SharedCredentialAuthorityError(_INVALID_POLICY_MESSAGE)
    return value


def _consumer_request(policy: Any) -> list[str]:
    if not isinstance(policy, dict) or set(policy) != {"authority", "providers"}:
        raise SharedCredentialAuthorityError(_INVALID_POLICY_MESSAGE)
    if policy["authority"] != "default":
        raise SharedCredentialAuthorityError(_INVALID_POLICY_MESSAGE)
    return _provider_list(policy["providers"])


def _authority_grants(policy: Any) -> dict[str, list[str]]:
    if not isinstance(policy, dict) or set(policy) != {"consumers"}:
        raise SharedCredentialAuthorityError(_INVALID_POLICY_MESSAGE)
    consumers = policy["consumers"]
    if not isinstance(consumers, dict) or any(
        not isinstance(profile_name, str) or not profile_name
        for profile_name in consumers
    ):
        raise SharedCredentialAuthorityError(_INVALID_POLICY_MESSAGE)
    return {profile_name: _provider_list(provider_ids) for profile_name, provider_ids in consumers.items()}


def resolve_auth_store_authority(provider_id: str) -> AuthStoreAuthority:
    """Return the local store unless the active named profile has a bilateral grant."""
    active_home = get_hermes_home()
    profile_name = profile_name_for_home(active_home) or "default"
    local = AuthStoreAuthority(active_home / "auth.json", profile_name, False)
    if profile_name == "default":
        return local

    consumer_policy = _shared_credentials(_load_config(active_home / "config.yaml"))
    if consumer_policy is None:
        return local
    requested_providers = _consumer_request(consumer_policy)
    if provider_id not in requested_providers:
        return local

    try:
        authority_root = get_default_hermes_root()
    except Exception:
        raise SharedCredentialAuthorityError(_UNAVAILABLE_AUTHORITY_MESSAGE) from None
    if not authority_root.is_dir():
        raise SharedCredentialAuthorityError(_UNAVAILABLE_AUTHORITY_MESSAGE)
    authority_policy = _shared_credentials(_load_config(authority_root / "config.yaml"))
    grants = _authority_grants(authority_policy)
    if provider_id not in grants.get(profile_name, []):
        raise SharedCredentialAuthorityError(_DENIED_AUTHORITY_MESSAGE)
    return AuthStoreAuthority(authority_root / "auth.json", "default", True)


@contextmanager
def resolved_auth_store_transaction(
    provider_id: str, timeout_seconds: Optional[float] = None,
) -> Iterator[Tuple[Dict[str, Any], Path]]:
    """Lock and load the auth store that is authoritative for *provider_id*.

    The auth helpers are imported lazily because this policy module resolves the
    target for ``hermes_cli.auth`` itself.  Callers save with the yielded path so
    shared authorities remain explicit write targets.
    """
    from hermes_cli.auth import _auth_store_lock, _load_auth_store

    authority = resolve_auth_store_authority(provider_id)
    lock_kwargs: Dict[str, Any] = {"target_path": authority.store_path}
    if timeout_seconds is not None:
        lock_kwargs["timeout_seconds"] = timeout_seconds
    with _auth_store_lock(**lock_kwargs):
        yield (
            _load_auth_store(authority.store_path, fail_closed=authority.shared),
            authority.store_path,
        )


def _load_auth_store_maybe_locked(
    lock: bool, *, provider_id: str = "openai-codex",
    preserve_corrupt_store: bool = True,
) -> Dict[str, Any]:
    """Load one provider's resolved store, locking it unless the caller already holds it."""
    from hermes_cli.auth import _auth_store_lock, _load_auth_store

    authority = resolve_auth_store_authority(provider_id)
    if lock:
        with _auth_store_lock(target_path=authority.store_path):
            return _load_auth_store(
                authority.store_path, fail_closed=authority.shared,
                preserve_corrupt_store=preserve_corrupt_store,
            )
    return _load_auth_store(
        authority.store_path, fail_closed=authority.shared,
        preserve_corrupt_store=preserve_corrupt_store,
    )


def _refresh_payload_access_token(
    response: Any, *, provider: str, invalid_json: Tuple[str, str],
    invalid_response: Optional[Tuple[str, str]], missing_access: Tuple[str, str],
    relogin_required: bool = True, invalid_json_relogin: Optional[bool] = None,
    strict_str: bool = True,
) -> Tuple[Dict[str, Any], str]:
    """Parse a token-refresh response while preserving the caller's error copy."""
    from hermes_cli.auth_constants import AuthError

    def _err(message: str, code: str, relogin: bool = relogin_required) -> AuthError:
        return AuthError(message, provider=provider, code=code, relogin_required=relogin)

    try:
        payload = response.json()
    except Exception as exc:
        relogin = relogin_required if invalid_json_relogin is None else invalid_json_relogin
        raise _err(invalid_json[0].format(exc=exc), invalid_json[1], relogin) from exc
    if not isinstance(payload, dict):
        if invalid_response is not None:
            raise _err(*invalid_response)
        payload = {}
    access = payload.get("access_token")
    if strict_str:
        access = access.strip() if isinstance(access, str) else ""
    else:
        access = str(access or "").strip()
    if not access:
        raise _err(*missing_access)
    return payload, access


def has_provider_credentials(provider_id: str) -> bool:
    """Return whether the resolved local authority store has an access token.

    Availability probes must remain read-only and fail closed: do not refresh OAuth
    credentials here, and do not let a malformed or unavailable store abort a scan.
    """
    try:
        auth_path = resolve_auth_store_authority(provider_id).store_path
        store = json.loads(auth_path.read_text(encoding="utf-8-sig"))
        providers = store.get("providers") if isinstance(store, dict) else None
        provider = providers.get(provider_id) if isinstance(providers, dict) else None
        tokens = provider.get("tokens") if isinstance(provider, dict) else None
        access_token = tokens.get("access_token") if isinstance(tokens, dict) else None
        if isinstance(access_token, str) and access_token.strip():
            return True
        credential_pool = store.get("credential_pool") if isinstance(store, dict) else None
        entries = credential_pool.get(provider_id) if isinstance(credential_pool, dict) else None
        return isinstance(entries, list) and any(
            isinstance(entry, dict)
            and isinstance(entry.get("access_token"), str)
            and entry["access_token"].strip()
            for entry in entries
        )
    except Exception:
        return False
