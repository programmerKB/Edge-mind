"""SQLAlchemy implementation of the application sensor repository port."""

from sqlalchemy import and_, case, func
from sqlalchemy.orm import Session

from edgemind.domain.entities import SensorReading
from edgemind.infrastructure.persistence.models import MotorSensorData


class SqlAlchemySensorRepository:
    """Persist sensor entities while preventing ORM leakage to inner layers."""

    def __init__(self, session: Session):
        """Bind repository operations to the unit-of-work session."""
        self._session = session

    def list_readings(self, motor_id: str) -> list[SensorReading]:
        """Return one device's complete history in chronological order."""
        rows = (
            self._session.query(MotorSensorData)
            .filter(MotorSensorData.motor_id == motor_id)
            .order_by(MotorSensorData.recorded_at.asc())
            .all()
        )
        return [_to_entity(row) for row in rows]

    def latest_reading(self, motor_id: str) -> SensorReading | None:
        """Return only the newest row for a status query."""
        row = (
            self._session.query(MotorSensorData)
            .filter(MotorSensorData.motor_id == motor_id)
            .order_by(MotorSensorData.recorded_at.desc())
            .first()
        )
        return _to_entity(row) if row is not None else None

    def add(self, reading: SensorReading) -> SensorReading:
        """Insert one entity and refresh generated ID and timestamp fields."""
        row = MotorSensorData(
            motor_id=reading.motor_id,
            temperature=reading.temperature,
            humidity=reading.humidity,
            accel_x=reading.accel_x,
            accel_y=reading.accel_y,
            accel_z=reading.accel_z,
            vibration=reading.vibration,
            status=reading.status,
        )
        if reading.recorded_at is not None:
            row.recorded_at = reading.recorded_at
        self._session.add(row)
        self._session.flush()
        self._session.refresh(row)
        return _to_entity(row)

    def list_device_summaries(self) -> list[dict]:
        """Aggregate total and complete feature rows for every motor."""
        complete = and_(
            MotorSensorData.temperature.is_not(None),
            MotorSensorData.humidity.is_not(None),
            MotorSensorData.accel_x.is_not(None),
            MotorSensorData.accel_y.is_not(None),
            MotorSensorData.accel_z.is_not(None),
            MotorSensorData.recorded_at.is_not(None),
        )
        rows = (
            self._session.query(
                MotorSensorData.motor_id,
                func.count(MotorSensorData.id),
                func.sum(case((complete, 1), else_=0)),
                func.min(MotorSensorData.recorded_at),
                func.max(MotorSensorData.recorded_at),
            )
            .filter(MotorSensorData.motor_id.is_not(None))
            .group_by(MotorSensorData.motor_id)
            .order_by(MotorSensorData.motor_id.asc())
            .all()
        )
        return [
            {
                "motor_id": motor_id,
                "row_count": int(row_count or 0),
                "complete_row_count": int(complete_count or 0),
                "first_recorded_at": first_recorded_at.isoformat()
                if first_recorded_at
                else None,
                "last_recorded_at": last_recorded_at.isoformat()
                if last_recorded_at
                else None,
            }
            for (
                motor_id,
                row_count,
                complete_count,
                first_recorded_at,
                last_recorded_at,
            ) in rows
        ]


def _to_entity(row: MotorSensorData) -> SensorReading:
    """Map a persistence row to a framework-independent domain entity."""
    return SensorReading(
        id=row.id,
        motor_id=row.motor_id,
        temperature=row.temperature,
        humidity=row.humidity,
        accel_x=row.accel_x,
        accel_y=row.accel_y,
        accel_z=row.accel_z,
        vibration=row.vibration,
        status=row.status,
        recorded_at=row.recorded_at,
    )
