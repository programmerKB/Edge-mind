"""Deterministic intent rules that force data-grounded forecast use cases."""

from __future__ import annotations

import re


# Device identifiers must contain at least one ASCII letter. This prevents the
# phrase "預測 30 分鐘" from accidentally treating ``30`` as a device ID.
DEVICE_ID = r"(?=[A-Za-z0-9._-]*[A-Za-z])[A-Za-z0-9][A-Za-z0-9._-]*"


def temperature_forecast_arguments(message: str) -> dict[str, str] | None:
    """Extract device arguments when a temperature forecast is requested."""
    compact = re.sub(r"\s+", " ", message).strip()
    if "溫度" not in compact or not any(
        keyword in compact
        for keyword in ("預測", "推論", "30 分鐘", "30分鐘", "未來")
    ):
        return None

    training_match = re.search(
        rf"(?:使用|用)\s*(?P<device>{DEVICE_ID})\s*"
        r"(?:訓練(?:的)?)?\s*模型",
        compact,
        flags=re.IGNORECASE,
    )
    inference_match = re.search(
        rf"(?:推論|預測)\s*(?:設備|馬達)?\s*(?P<device>{DEVICE_ID})",
        compact,
        flags=re.IGNORECASE,
    )
    if inference_match is None:
        inference_match = re.search(
            rf"(?:設備|馬達)\s*(?P<device>{DEVICE_ID}).*?"
            r"(?:30\s*分鐘|未來).*?溫度",
            compact,
            flags=re.IGNORECASE,
        )

    training_motor_id = (
        training_match.group("device") if training_match else None
    )
    motor_id = (
        inference_match.group("device") if inference_match else training_motor_id
    )
    if motor_id is None:
        return None

    arguments = {"motor_id": motor_id}
    if training_motor_id and training_motor_id != motor_id:
        arguments["training_motor_id"] = training_motor_id
    return arguments
