"""Filesystem and evaluation settings shared by report modules."""

from datetime import timedelta, timezone
import os
from pathlib import Path
import time


# ``parents[1]`` resolves to the backend root after moving this code into the
# ``reports`` package, preserving the historical ``backend/outputs`` default.
REPORT_ROOT = Path(
    os.getenv(
        "INFERENCE_OUTPUT_DIR",
        Path(__file__).resolve().parents[1] / "outputs",
    )
)
REPORT_TIMEZONE = timezone(
    timedelta(hours=float(os.getenv("REPORT_TIMEZONE_OFFSET_HOURS", "8")))
)
ANOMALY_TEMPERATURE_THRESHOLD = float(
    os.getenv("ANOMALY_TEMPERATURE_THRESHOLD", "35.0")
)
PROCESS_STARTED_AT = time.monotonic()
