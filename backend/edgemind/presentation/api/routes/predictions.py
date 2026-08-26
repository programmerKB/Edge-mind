"""REST endpoints for model training and 30-minute inference."""

import time
from typing import Callable

from fastapi import APIRouter, HTTPException, status

from edgemind.application.forecasts import ForecastService
from edgemind.application.ports import UnitOfWork
from edgemind.domain.forecasting import ForecastError


def create_router(
    forecasts: ForecastService,
    uow_factory: Callable[[], UnitOfWork],
) -> APIRouter:
    """Create forecast routes bound to application-layer dependencies."""
    router = APIRouter(prefix="/api/predictions", tags=["predictions"])

    @router.post("/train/{motor_id}")
    def train_temperature_forecast(motor_id: str):
        """Train and persist a per-device model from historical readings."""
        try:
            with uow_factory() as uow:
                payload = forecasts.train(uow, motor_id)
            metrics = payload["validation_metrics"]
            return {
                "motor_id": motor_id,
                "feature_names": payload["feature_names"],
                "forecast_horizon_minutes": payload["horizon_minutes"],
                "sample_count": payload["sample_count"],
                "validation_sample_count": payload["validation_sample_count"],
                "validation_mae": round(payload["mae"], 4),
                "validation_rmse": round(payload["rmse"], 4),
                "validation_r2": (
                    round(metrics["r2_score"], 4)
                    if metrics["r2_score"] is not None
                    else None
                ),
                "validation_mape_percent": (
                    round(metrics["mape_percent"], 4)
                    if metrics["mape_percent"] is not None
                    else None
                ),
                "training_duration_ms": round(
                    payload["training_duration_ms"],
                    6,
                ),
                "trained_at": payload["trained_at"],
            }
        except ForecastError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"模型訓練失敗：{error}",
            ) from error

    @router.get("/temperature/{motor_id}")
    def get_30_minute_temperature_forecast(
        motor_id: str,
        auto_train: bool = True,
        training_motor_id: str | None = None,
    ):
        """Predict with the same device model or a specified training source."""
        started = time.perf_counter()
        try:
            with uow_factory() as uow:
                result = forecasts.forecast(
                    uow,
                    motor_id,
                    auto_train=auto_train,
                    training_motor_id=training_motor_id,
                )
            result["api_handler_duration_ms"] = round(
                (time.perf_counter() - started) * 1000,
                6,
            )
            return result
        except ForecastError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"溫度預測失敗：{error}",
            ) from error

    return router
