"""Internal metadata attached to durable conversation messages."""

from __future__ import annotations

from time import time as wall_time
from typing import Any, MutableMapping, Optional, TypeVar


# These fields describe Hermes' durable record, not provider-visible message
# content. They must not influence context-pressure decisions, and every path
# that relays a stored message dict as-is (the Chat Completions transport, the
# summary path) must call ``strip_persistence_only_fields`` first: strict
# providers (Mistral 422 ``extra_forbidden``, Groq) reject any key outside the
# schema. The Anthropic/Bedrock/Codex transports rebuild wire messages from
# scratch, so nothing leaks there without this call.
PERSISTENCE_ONLY_MESSAGE_FIELDS = frozenset(
    {
        "timestamp",
        "platform_message_id",
        "message_id",
        "display_kind",
        "display_metadata",
        "observed",
    }
)


def strip_persistence_only_fields(message: MutableMapping[str, Any]) -> None:
    """Drop the persistence-only keys in place, right before a message goes on the wire."""
    for field in PERSISTENCE_ONLY_MESSAGE_FIELDS:
        message.pop(field, None)

_Message = TypeVar("_Message", bound=MutableMapping[str, Any])


def stamp_message_timestamp(
    message: _Message,
    *,
    timestamp: Optional[float] = None,
) -> _Message:
    """Attach a creation timestamp without replacing source-provided time.

    Gateway adapters can supply the platform event time. All other callers use
    the local wall clock at the point the message enters the live transcript.
    Returning the same mapping keeps the helper convenient at append sites.
    """
    if message.get("timestamp") is None:
        message["timestamp"] = wall_time() if timestamp is None else timestamp
    return message


def append_message(
    messages: list[Any],
    message: _Message,
    *,
    timestamp: Optional[float] = None,
) -> _Message:
    """Stamp and append one live transcript message."""
    stamp_message_timestamp(message, timestamp=timestamp)
    messages.append(message)
    return message
