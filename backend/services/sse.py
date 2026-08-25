"""Small, dependency-free helpers for server-sent event payloads."""

from __future__ import annotations

import json


def sse_event(
    status: str,
    content: str,
    ensure_ascii: bool,
    *,
    attachments: list[dict[str, str]] | None = None,
) -> str:
    """Serialize one application event using the SSE data-field format."""
    payload: dict[str, object] = {"status": status, "content": content}
    if attachments:
        payload["attachments"] = attachments
    serialized = json.dumps(payload, ensure_ascii=ensure_ascii)
    return f"data: {serialized}\n\n"
