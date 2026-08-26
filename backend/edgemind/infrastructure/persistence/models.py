"""SQLAlchemy persistence models.

Models describe storage only; querying and orchestration live in repositories
and services so HTTP handlers do not accumulate database rules.
"""

from sqlalchemy import Column, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.sql import func
from edgemind.infrastructure.persistence.database import Base


class MotorSensorData(Base):
    """One timestamped environmental feature vector from an edge device."""

    __tablename__ = "motor_sensor_data"
    __table_args__ = (
        Index("ix_motor_sensor_motor_recorded_at", "motor_id", "recorded_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    motor_id = Column(String, index=True)
    temperature = Column(Float)
    humidity = Column(Float)
    accel_x = Column(Float)
    accel_y = Column(Float)
    accel_z = Column(Float)
    # Keep the legacy aggregate vibration field for deployed edge clients.
    vibration = Column(Float)
    status = Column(String)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())


class TemperatureForecastModel(Base):
    """Latest persisted 30-minute forecast model for one training device."""

    __tablename__ = "temperature_forecast_models"

    id = Column(Integer, primary_key=True, index=True)
    motor_id = Column(String, unique=True, index=True, nullable=False)
    model_json = Column(Text, nullable=False)
    sample_count = Column(Integer, nullable=False)
    mae = Column(Float)
    rmse = Column(Float)
    trained_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
