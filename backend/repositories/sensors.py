"""Queries for motor sensor readings."""

from sqlalchemy.orm import Session

from models import MotorSensorData


def list_readings(db: Session, motor_id: str) -> list[MotorSensorData]:
    """Return one device's complete history in chronological order."""
    return (
        db.query(MotorSensorData)
        .filter(MotorSensorData.motor_id == motor_id)
        .order_by(MotorSensorData.recorded_at.asc())
        .all()
    )


def latest_reading(db: Session, motor_id: str) -> MotorSensorData | None:
    """Return only the newest row for status queries that need no history."""
    return (
        db.query(MotorSensorData)
        .filter(MotorSensorData.motor_id == motor_id)
        .order_by(MotorSensorData.recorded_at.desc())
        .first()
    )
