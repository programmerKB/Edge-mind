from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.sql import func
from database import Base

class MotorSensorData(Base):
    __tablename__ = "motor_sensor_data"

    id = Column(Integer, primary_key=True, index=True)
    motor_id = Column(String, index=True)
    temperature = Column(Float)
    humidity = Column(Float)
    accel_x = Column(Float)
    accel_y = Column(Float)
    accel_z = Column(Float)
    # 保留舊欄位，讓既有設備與 API 仍可繼續使用。
    vibration = Column(Float)
    status = Column(String)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())


class TemperatureForecastModel(Base):
    """每台設備最近一次訓練完成的 30 分鐘溫度預測模型。"""

    __tablename__ = "temperature_forecast_models"

    id = Column(Integer, primary_key=True, index=True)
    motor_id = Column(String, unique=True, index=True, nullable=False)
    model_json = Column(Text, nullable=False)
    sample_count = Column(Integer, nullable=False)
    mae = Column(Float)
    rmse = Column(Float)
    trained_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
