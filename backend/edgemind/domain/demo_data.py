"""Deterministic domain fixtures for the demonstration workflow."""

from datetime import datetime, timedelta, timezone
import math
import random


DEMO_TRAINING_MOTOR_ID = "DEMO-1"
DEMO_INFERENCE_MOTOR_ID = "DEMO-2"
# Backwards-compatible name used by existing callers and tests.
DEMO_MOTOR_ID = DEMO_TRAINING_MOTOR_ID
# Seven complete days are the minimum exploratory volume advertised by the
# research workbench (7 × 24 × 12 five-minute readings).
DEMO_READING_COUNT = 2_016
DEMO_INFERENCE_READING_COUNT = 2_016
DEMO_INTERVAL_MINUTES = 5
DEMO_TRAINING_STATUS_PREFIX = "demo-v3-training"
DEMO_INFERENCE_STATUS_PREFIX = "demo-v3-inference"


def _utc_grid_floor(value: datetime) -> datetime:
    """Floor a reference time to the demo sampling grid in UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        value = value.replace(tzinfo=timezone.utc)
    utc_value = value.astimezone(timezone.utc)
    interval_seconds = DEMO_INTERVAL_MINUTES * 60
    aligned_timestamp = (
        int(utc_value.timestamp()) // interval_seconds * interval_seconds
    )
    return datetime.fromtimestamp(aligned_timestamp, tz=timezone.utc)


def _build_demo_series(
    *,
    motor_id: str,
    count: int,
    reference_time: datetime | None,
    seed: int,
    ambient_offset: float,
    thermal_gain: float,
    cooling_rate: float,
    phase_shift: float,
    status_prefix: str,
) -> list[dict]:
    """Generate deterministic nonstationary operating histories for UI demos."""
    # Research datasets require exact UTC sampling-grid timestamps.  Merely
    # removing seconds leaves a series generated at (for example) :03/:08 on
    # the wrong phase and causes every row to be rejected by the strict
    # sequence builder.
    end_time = _utc_grid_floor(reference_time or datetime.now(timezone.utc))
    start_time = end_time - timedelta(minutes=(count - 1) * DEMO_INTERVAL_MINUTES)
    samples_per_day = 24 * 60 // DEMO_INTERVAL_MINUTES
    rng = random.Random(seed)
    temperature = 25.0 + ambient_offset
    readings: list[dict] = []
    for index in range(count):
        day_fraction = (index % samples_per_day) / samples_per_day
        hour = day_fraction * 24
        if hour < 6:
            load = 0.20
        elif hour < 8:
            load = 0.48
        elif hour < 16:
            load = 0.70
        elif hour < 20:
            load = 0.91
        else:
            load = 0.36
        load = min(1.0, max(0.05, load + rng.gauss(0.0, 0.028)))
        wear_ratio = (index % (samples_per_day * 4)) / (samples_per_day * 4 - 1)
        fault_position = index % (samples_per_day * 3)
        fault_active = samples_per_day * 2 + 170 <= fault_position < samples_per_day * 2 + 210
        if fault_active:
            load = min(1.0, load + 0.16)
        ambient = (
            24.8
            + ambient_offset
            + 2.3 * math.sin(2 * math.pi * day_fraction - 1.1 + phase_shift)
            + 0.35 * math.sin(2 * math.pi * index / (samples_per_day * 7))
        )
        vibration_amplitude = 0.018 + 0.060 * load + 0.032 * wear_ratio
        if fault_active:
            vibration_amplitude += 0.024
        accel_x = vibration_amplitude * math.sin(index * 0.71) + rng.gauss(0, 0.004)
        accel_y = vibration_amplitude * math.cos(index * 0.53) + rng.gauss(0, 0.004)
        accel_z = 1.0 + 0.45 * vibration_amplitude * math.sin(index * 0.37) + rng.gauss(0, 0.002)
        vibration = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        thermal_target = (
            ambient
            + thermal_gain * load
            + 3.4 * wear_ratio
            + (3.0 if fault_active else 0.0)
            + 8.0 * max(0.0, vibration - 1.0)
        )
        temperature += cooling_rate * (thermal_target - temperature) + rng.gauss(0, 0.055)
        humidity = max(
            20.0,
            min(
                95.0,
                63.0
                - 0.60 * (temperature - ambient)
                + 5.0 * math.cos(2 * math.pi * day_fraction + 0.4)
                + rng.gauss(0.0, 0.5),
            ),
        )
        state = "warning" if temperature >= 35.0 else "normal"
        readings.append(
            {
                "motor_id": motor_id,
                "temperature": round(temperature, 4),
                "humidity": round(humidity, 4),
                "accel_x": round(accel_x, 6),
                "accel_y": round(accel_y, 6),
                "accel_z": round(accel_z, 6),
                "vibration": round(vibration, 6),
                "status": f"{status_prefix}-{state}",
                "recorded_at": start_time + timedelta(minutes=index * DEMO_INTERVAL_MINUTES),
            }
        )
    return readings


def build_demo_readings(reference_time: datetime | None = None) -> list[dict]:
    """Return a seven-day multi-regime training history."""
    return _build_demo_series(
        motor_id=DEMO_MOTOR_ID,
        count=DEMO_READING_COUNT,
        reference_time=reference_time,
        seed=421,
        ambient_offset=0.0,
        thermal_gain=13.2,
        cooling_rate=0.074,
        phase_shift=0.0,
        status_prefix=DEMO_TRAINING_STATUS_PREFIX,
    )


def build_demo_inference_readings(
    reference_time: datetime | None = None,
) -> list[dict]:
    """Return unseen, in-distribution readings reserved for inference.

    The data covers the same operating envelope as DEMO-1 but uses different
    coefficients. Earlier rows provide known 30-minute outcomes for evaluation;
    the latest row remains the live inference input. DEMO-2 is never used to fit
    the model in the demo workflow.
    """
    return _build_demo_series(
        motor_id=DEMO_INFERENCE_MOTOR_ID,
        count=DEMO_INFERENCE_READING_COUNT,
        reference_time=reference_time,
        seed=842,
        ambient_offset=0.8,
        thermal_gain=13.9,
        cooling_rate=0.067,
        phase_shift=0.45,
        status_prefix=DEMO_INFERENCE_STATUS_PREFIX,
    )
