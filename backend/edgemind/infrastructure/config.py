"""Typed, environment-backed application configuration.

Keeping environment parsing in one module makes invalid values fail during
startup instead of surfacing later in an API request.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _boolean(name: str, default: bool) -> bool:
    """Parse common truthy values while preserving an explicit default."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _positive_float(name: str, default: float) -> float:
    """Read a positive float and raise a useful startup error when invalid."""
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} 必須是數字，目前值為 {raw!r}") from error
    if value <= 0:
        raise ValueError(f"{name} 必須大於 0，目前值為 {value}")
    return value


def _float(name: str, default: float) -> float:
    """Read any finite deployment float with an actionable error."""
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except ValueError as error:
        raise ValueError(f"{name} 必須是數字，目前值為 {raw!r}") from error


def _origins() -> tuple[str, ...]:
    """Normalize a comma-separated CORS allow-list once at startup."""
    return tuple(
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "*").split(",")
        if origin.strip()
    )


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable runtime configuration consumed through dependency imports."""

    database_url: str
    model_id: str
    agent_response_timeout_seconds: float
    seed_demo_data: bool
    cors_origins: tuple[str, ...]
    report_root: Path
    report_timezone_offset_hours: float
    anomaly_temperature_threshold: float


settings = Settings(
    database_url=os.getenv(
        "DATABASE_URL",
        "postgresql://agent_user:agent_pass@db:5432/motor_monitor_db",
    ),
    model_id=os.getenv("GEMINI_MODEL_ID", "gemini-3.5-flash-lite"),
    agent_response_timeout_seconds=_positive_float(
        "AGENT_RESPONSE_TIMEOUT_SECONDS",
        60.0,
    ),
    seed_demo_data=_boolean("SEED_DEMO_DATA", True),
    cors_origins=_origins(),
    report_root=Path(
        os.getenv(
            "INFERENCE_OUTPUT_DIR",
            Path(__file__).resolve().parents[2] / "outputs",
        )
    ),
    report_timezone_offset_hours=_float(
        "REPORT_TIMEZONE_OFFSET_HOURS",
        8.0,
    ),
    anomaly_temperature_threshold=_float(
        "ANOMALY_TEMPERATURE_THRESHOLD",
        35.0,
    ),
)


# The instruction is code-owned because it is part of application behavior,
# while deployment-specific values remain in ``Settings`` above.
AGENT_SYSTEM_INSTRUCTION = (
    "你是一個專業的工業馬達與邊緣設備診斷助手。請根據數據回答問題，"
    "查詢狀態時呼叫狀態工具；詢問單一 30 分鐘溫度時呼叫既有預測工具；"
    "詢問溫度軌跡、多時域、過熱風險或預警時間時，務必呼叫多時域軌跡工具。"
    "若使用者要求用 A 設備訓練的模型推論 B 設備，請傳入 motor_id=B、"
    "training_motor_id=A。回答請使用繁體中文，並給出具體的維護建議。"
)
