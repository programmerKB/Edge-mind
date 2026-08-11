"""Deterministic training and inference data for the demo workflow."""

from datetime import datetime, timedelta, timezone
import math


DEMO_TRAINING_MOTOR_ID = "DEMO-1"
DEMO_INFERENCE_MOTOR_ID = "DEMO-2"
# Backwards-compatible name used by existing callers and tests.
DEMO_MOTOR_ID = DEMO_TRAINING_MOTOR_ID
DEMO_READING_COUNT = 36
DEMO_INFERENCE_READING_COUNT = 36
DEMO_INTERVAL_MINUTES = 5


def build_demo_readings(reference_time: datetime | None = None) -> list[dict]:
    """Return enough five-minute readings to train a 30-minute model."""
    end_time = reference_time or datetime.now(timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)
    end_time = end_time.replace(second=0, microsecond=0)
    start_time = end_time - timedelta(
        minutes=(DEMO_READING_COUNT - 1) * DEMO_INTERVAL_MINUTES
    )

    readings = []
    for index in range(DEMO_READING_COUNT):
        accel_x = 0.08 * math.sin(index / 2.7)
        accel_y = 0.07 * math.cos(index / 3.1)
        accel_z = 1.0 + 0.025 * math.sin(index / 4.3)
        humidity = 58.0 - index * 0.12 + 1.8 * math.cos(index / 5.0)
        temperature = (
            31.5
            + index * 0.11
            + 0.35 * math.sin(index / 3.8)
            + 0.8 * abs(accel_x)
        )
        readings.append(
            {
                "motor_id": DEMO_MOTOR_ID,
                "temperature": round(temperature, 4),
                "humidity": round(humidity, 4),
                "accel_x": round(accel_x, 6),
                "accel_y": round(accel_y, 6),
                "accel_z": round(accel_z, 6),
                "vibration": round(
                    math.sqrt(accel_x**2 + accel_y**2 + accel_z**2),
                    6,
                ),
                "status": "demo",
                "recorded_at": start_time
                + timedelta(minutes=index * DEMO_INTERVAL_MINUTES),
            }
        )
    return readings


def build_demo_inference_readings(
    reference_time: datetime | None = None,
) -> list[dict]:
    """Return unseen, in-distribution readings reserved for inference.

    The data covers the same operating envelope as DEMO-1 but uses different
    coefficients. Earlier rows provide known 30-minute outcomes for evaluation;
    the latest row remains the live inference input. DEMO-2 is never used to fit
    the model in the demo workflow.
    """
    end_time = reference_time or datetime.now(timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)
    end_time = end_time.replace(second=0, microsecond=0)
    start_time = end_time - timedelta(
        minutes=(DEMO_INFERENCE_READING_COUNT - 1) * DEMO_INTERVAL_MINUTES
    )

    readings = []
    for index in range(DEMO_INFERENCE_READING_COUNT):
        phase = index
        accel_x = 0.075 * math.sin(phase / 2.7) + 0.004
        accel_y = 0.065 * math.cos(phase / 3.1) - 0.003
        accel_z = 1.0 + 0.022 * math.sin(phase / 4.3)
        humidity = 57.6 - phase * 0.115 + 1.65 * math.cos(phase / 5.0)
        temperature = (
            31.65
            + phase * 0.108
            + 0.32 * math.sin(phase / 3.8)
            + 0.75 * abs(accel_x)
        )
        readings.append(
            {
                "motor_id": DEMO_INFERENCE_MOTOR_ID,
                "temperature": round(temperature, 4),
                "humidity": round(humidity, 4),
                "accel_x": round(accel_x, 6),
                "accel_y": round(accel_y, 6),
                "accel_z": round(accel_z, 6),
                "vibration": round(
                    math.sqrt(accel_x**2 + accel_y**2 + accel_z**2),
                    6,
                ),
                "status": "demo-inference",
                "recorded_at": start_time
                + timedelta(minutes=index * DEMO_INTERVAL_MINUTES),
            }
        )
    return readings
