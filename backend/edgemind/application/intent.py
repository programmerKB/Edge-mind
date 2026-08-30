"""Deterministic intent rules that force data-grounded forecast use cases."""

from __future__ import annotations

import re
from typing import Any


# Device identifiers must contain at least one ASCII letter. This prevents the
# phrase "預測 30 分鐘" from accidentally treating ``30`` as a device ID.
DEVICE_ID = r"(?=[A-Za-z0-9._-]*[A-Za-z])[A-Za-z0-9][A-Za-z0-9._-]*"
MODEL_IDENTIFIERS = {
    "ridge",
    "lstm",
    "tcn",
    "dlinear",
    "patchtst",
}


def _device_arguments(compact: str) -> dict[str, Any] | None:
    """Extract source and target device IDs without mistaking model names."""
    training_match = re.search(
        rf"(?:使用|用)\s*(?:設備|馬達)?\s*(?P<device>{DEVICE_ID})\s*"
        r"(?:設備|馬達)?\s*(?:的)?(?:資料)?\s*訓練",
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
            r"(?:30\s*分鐘|未來|軌跡|風險)",
            compact,
            flags=re.IGNORECASE,
        )
    training_motor_id = training_match.group("device") if training_match else None
    if training_motor_id and training_motor_id.lower() in MODEL_IDENTIFIERS:
        training_motor_id = None
    motor_id = inference_match.group("device") if inference_match else training_motor_id
    if motor_id is None:
        return None
    arguments = {"motor_id": motor_id}
    if training_motor_id and training_motor_id != motor_id:
        arguments["training_motor_id"] = training_motor_id
    return arguments


def temperature_trajectory_arguments(message: str) -> dict[str, Any] | None:
    """Route trajectory, overheat-risk, and warning-lead-time requests."""
    compact = re.sub(r"\s+", " ", message).strip()
    trajectory_keywords = (
        "軌跡",
        "多時域",
        "多步",
        "過熱",
        "風險",
        "預警",
        "越過門檻",
        "升溫率",
        "time-to-threshold",
        "multi-horizon",
        "+5",
        "5 到 30",
        "5至30",
        "5～30",
    )
    if not any(keyword.lower() in compact.lower() for keyword in trajectory_keywords):
        return None
    arguments = _device_arguments(compact)
    if arguments is None:
        return None
    model_aliases = (
        (r"patchtst", "patchtst"),
        (r"dlinear", "dlinear"),
        (r"\blstm\b", "lstm"),
        (r"\btcn\b", "tcn"),
        (r"ridge\s*\+?\s*(?:history|historical|trend)", "ridge_history_trend"),
        (r"direct\s*ridge", "ridge_direct"),
    )
    for pattern, model_name in model_aliases:
        if re.search(pattern, compact, flags=re.IGNORECASE):
            arguments["model_name"] = model_name
            break
    threshold_match = re.search(
        r"(?P<value>-?\d+(?:\.\d+)?)\s*(?:°\s*)?[cC]\s*"
        r"(?:的?\s*)?(?:警戒(?:值|溫度)?|門檻|threshold)",
        compact,
        flags=re.IGNORECASE,
    ) or re.search(
        r"(?:警戒(?:值|溫度)?|門檻|threshold)\s*(?:為|是|=|:)?\s*"
        r"(?P<value>-?\d+(?:\.\d+)?)\s*(?:°\s*)?[cC]?",
        compact,
        flags=re.IGNORECASE,
    )
    if threshold_match:
        arguments["threshold_c"] = float(threshold_match.group("value"))

    range_match = re.search(
        r"(?P<start>\d+)\s*(?:到|至|～|~|-)\s*(?P<end>\d+)\s*分鐘",
        compact,
    )
    if range_match:
        start = int(range_match.group("start"))
        end = int(range_match.group("end"))
        if 0 < start <= end <= 60 and start % 5 == 0 and end % 5 == 0:
            arguments["horizons_minutes"] = list(range(start, end + 1, 5))
        else:
            # Preserve an explicitly invalid range so the domain contract
            # returns a clear error instead of silently using default horizons.
            arguments["horizons_minutes"] = [start, end]
    return arguments


def temperature_forecast_arguments(message: str) -> dict[str, str] | None:
    """Extract device arguments when a temperature forecast is requested."""
    compact = re.sub(r"\s+", " ", message).strip()
    if "溫度" not in compact or not any(
        keyword in compact
        for keyword in ("預測", "推論", "30 分鐘", "30分鐘", "未來")
    ):
        return None

    return _device_arguments(compact)
