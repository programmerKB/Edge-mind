"""REST endpoints for model training and 30-minute inference."""

import time

from fastapi import APIRouter, Depends, HTTPException, status

from core.database import get_db
from forecast import ForecastError, forecast_temperature, train_and_save_model
from services.forecast_delivery import finalize_forecast_result


router = APIRouter(prefix="/api/predictions", tags=["predictions"])


@router.post("/train/{motor_id}")
def train_temperature_forecast(
    motor_id: str,
    db=Depends(get_db),
):
    """Train and persist a per-device model using historical readings."""
    try:
        payload = train_and_save_model(db, motor_id)
        return {
            "motor_id": motor_id,
            "feature_names": payload["feature_names"],
            "forecast_horizon_minutes": payload["horizon_minutes"],
            "sample_count": payload["sample_count"],
            "validation_sample_count": payload["validation_sample_count"],
            "validation_mae": round(payload["mae"], 4),
            "validation_rmse": round(payload["rmse"], 4),
            "validation_r2": (
                round(payload["validation_metrics"]["r2_score"], 4)
                if payload["validation_metrics"]["r2_score"] is not None
                else None
            ),
            "validation_mape_percent": (
                round(payload["validation_metrics"]["mape_percent"], 4)
                if payload["validation_metrics"]["mape_percent"] is not None
                else None
            ),
            "training_duration_ms": round(payload["training_duration_ms"], 6),
            "trained_at": payload["trained_at"],
        }
    except ForecastError as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
    except Exception as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"模型訓練失敗：{error}",
        ) from error


@router.get("/temperature/{motor_id}")
def get_30_minute_temperature_forecast(
    motor_id: str,
    auto_train: bool = True,
    training_motor_id: str | None = None,
    db=Depends(get_db),
):
    """Predict using the same device's model or an explicit training source."""
    request_started = time.perf_counter()
    try:
        result = forecast_temperature(
            db,
            motor_id,
            auto_train=auto_train,
            training_motor_id=training_motor_id,
        )
        api_duration_ms = (time.perf_counter() - request_started) * 1000
        result["api_handler_duration_ms"] = round(api_duration_ms, 6)
        return finalize_forecast_result(result)
    except ForecastError as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
    except Exception as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"溫度預測失敗：{error}",
        ) from error
