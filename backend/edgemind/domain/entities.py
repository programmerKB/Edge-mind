"""Framework-independent entities shared by domain and application layers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class SensorReading:
    """One complete sensor snapshot without ORM or API framework behavior."""

    motor_id: str
    temperature: float
    humidity: float
    accel_x: float
    accel_y: float
    accel_z: float
    vibration: float
    status: str
    recorded_at: datetime | None = None
    id: int | None = None
