from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from database import Base

class MotorSensorData(Base):
    __tablename__ = "motor_sensor_data"

    id = Column(Integer, primary_key=True, index=True)
    motor_id = Column(String, index=True)
    temperature = Column(Float)
    vibration = Column(Float)
    status = Column(String)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())