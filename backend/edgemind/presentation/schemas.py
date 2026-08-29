"""Pydantic request contracts owned by the HTTP presentation layer."""

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


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


class ResearchExperimentRequest(BaseModel):
    """Validated, reproducible configuration for one synchronous experiment."""

    model_config = ConfigDict(extra="forbid")

    training_motor_id: str = Field(min_length=1, max_length=100)
    evaluation_motor_id: str | None = Field(default=None, max_length=100)
    history_minutes: int = Field(default=60, ge=10, le=24 * 60)
    horizons_minutes: list[Annotated[int, Field(ge=5, le=60)]] = Field(
        default_factory=lambda: [5, 10, 15, 20, 25, 30],
        min_length=1,
        max_length=12,
    )
    threshold_c: float = Field(default=35.0, ge=20, le=120, allow_inf_nan=False)
    model_names: list[str] = Field(
        default_factory=lambda: [
            "persistence",
            "ridge_direct",
            "ridge_history_trend",
            "xgboost",
            "gru",
            "lstm",
            "tcn",
            "dlinear",
            "transformer",
            "patchtst",
        ],
        min_length=1,
        max_length=10,
    )
    include_ablations: bool = True
    sampling_minutes: int = Field(default=5, ge=1, le=60)
    train_fraction: float = Field(default=0.60, gt=0, lt=1)
    validation_fraction: float = Field(default=0.20, gt=0, lt=1)
    gap_steps: int | None = Field(default=None, ge=0, le=288)
    walk_forward_folds: int = Field(default=3, ge=1, le=10)
    random_seed: int = Field(default=42, ge=0, le=2_147_483_647)


class ResearchForecastRequest(BaseModel):
    """Configuration for a current six-point trajectory and risk forecast."""

    model_config = ConfigDict(extra="forbid")

    motor_id: str = Field(min_length=1, max_length=100)
    training_motor_id: str | None = Field(default=None, max_length=100)
    model_name: str = Field(default="ridge_history_trend", min_length=1, max_length=50)
    history_minutes: int = Field(default=60, ge=10, le=24 * 60)
    horizons_minutes: list[Annotated[int, Field(ge=5, le=60)]] = Field(
        default_factory=lambda: [5, 10, 15, 20, 25, 30],
        min_length=1,
        max_length=12,
    )
    threshold_c: float = Field(default=35.0, ge=20, le=120, allow_inf_nan=False)
    sampling_minutes: int = Field(default=5, ge=1, le=60)
    random_seed: int = Field(default=42, ge=0, le=2_147_483_647)
