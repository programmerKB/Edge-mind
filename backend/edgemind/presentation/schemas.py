"""Pydantic request contracts owned by the HTTP presentation layer."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Validated user input accepted by both chat streaming endpoints."""

    message: str = Field(min_length=1, max_length=10_000)
    inference_model: Literal["ridge_direct", "ridge_history"] = "ridge_direct"


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


class RidgeExperimentRequest(BaseModel):
    """One fixed dual-Ridge experiment selection."""

    training_motor_id: str = Field(min_length=1, max_length=100)
    evaluation_motor_id: str | None = Field(default=None, max_length=100)


class RidgeForecastRequest(BaseModel):
    """Apply one Ridge variant to a device's newest history window."""

    motor_id: str = Field(min_length=1, max_length=100)
    training_motor_id: str = Field(min_length=1, max_length=100)
    model_name: Literal["ridge_direct", "ridge_history"]
