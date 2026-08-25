"""Pydantic request contracts shared by API routers."""

from datetime import datetime

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Validated user input accepted by both chat streaming endpoints."""

    message: str = Field(min_length=1, max_length=10_000)


class SensorReadingRequest(BaseModel):
    """Complete five-feature sensor payload used by the forecast model."""

    motor_id: str = Field(min_length=1, max_length=100)
    temperature: float
    humidity: float = Field(ge=0, le=100)
    accel_x: float
    accel_y: float
    accel_z: float
    recorded_at: datetime | None = None
    status: str = Field(default="normal", max_length=50)
