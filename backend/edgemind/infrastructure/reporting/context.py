"""Explicit runtime context shared by filesystem reporting adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta, timezone
from pathlib import Path
import time
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from edgemind.infrastructure.config import Settings


@dataclass(frozen=True, slots=True)
class ReportContext:
    """Immutable paths and thresholds injected into report functions."""

    root: Path
    timezone: timezone
    anomaly_temperature_threshold: float
    process_started_at: float = field(default_factory=time.monotonic)

    @classmethod
    def from_settings(cls, settings: Settings) -> "ReportContext":
        """Build reporting context once at the composition boundary."""
        return cls(
            root=settings.report_root,
            timezone=timezone(
                timedelta(hours=settings.report_timezone_offset_hours)
            ),
            anomaly_temperature_threshold=(
                settings.anomaly_temperature_threshold
            ),
        )
