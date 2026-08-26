"""Small, dependency-free helpers for server-sent event payloads."""

from __future__ import annotations

import json

from edgemind.application.agent import AgentEvent


def encode_sse_event(
    event: AgentEvent,
    ensure_ascii: bool,
) -> str:
    """Serialize one transport-neutral application event as SSE data."""
    payload: dict[str, object] = {
        "status": event.status,
        "content": event.content,
    }
    if event.attachments:
        payload["attachments"] = event.attachments
    serialized = json.dumps(payload, ensure_ascii=ensure_ascii)
    return f"data: {serialized}\n\n"
