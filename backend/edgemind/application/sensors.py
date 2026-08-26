"""Sensor ingestion and latest-status application use cases."""

from __future__ import annotations

from datetime import datetime
import math

from edgemind.application.ports import UnitOfWork
from edgemind.domain.entities import SensorReading


class SensorValidationError(ValueError):
    """Raised when an incoming reading violates domain-level constraints."""


class SensorService:
    """Validate sensor data and coordinate its persistence."""

    def create(
        self,
        uow: UnitOfWork,
        *,
        motor_id: str,
        temperature: float,
        humidity: float,
        accel_x: float,
        accel_y: float,
        accel_z: float,
        status: str,
        recorded_at: datetime | None,
    ) -> SensorReading:
        """Validate, derive vibration magnitude, and persist one reading."""
        values = (temperature, humidity, accel_x, accel_y, accel_z)
        if not all(math.isfinite(value) for value in values):
            raise SensorValidationError("所有感測值都必須是有限數值")
        reading = SensorReading(
            motor_id=motor_id,
            temperature=temperature,
            humidity=humidity,
            accel_x=accel_x,
            accel_y=accel_y,
            accel_z=accel_z,
            vibration=math.sqrt(accel_x**2 + accel_y**2 + accel_z**2),
            status=status,
            recorded_at=recorded_at,
        )
        stored = uow.sensors.add(reading)
        uow.commit()
        return stored

    def latest_status(self, uow: UnitOfWork, motor_id: str) -> dict:
        """Return the latest sensor snapshot or a stable error payload."""
        record = uow.sensors.latest_reading(motor_id)
        if record is None:
            return {"error": f"資料庫中找不到馬達 {motor_id} 的感測器資料"}
        return {
            "motor_id": record.motor_id,
            "temperature": record.temperature,
            "humidity": record.humidity,
            "accel_x": record.accel_x,
            "accel_y": record.accel_y,
            "accel_z": record.accel_z,
            "vibration": record.vibration,
            "status": record.status,
            "recorded_at": (
                record.recorded_at.isoformat() if record.recorded_at else None
            ),
        }
