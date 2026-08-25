"""Synchronous database tools exposed to the Gemini function-calling layer."""

from __future__ import annotations

import json
import time

from core.database import SessionLocal
from forecast import ForecastError, forecast_temperature
from repositories.sensors import latest_reading
from services.forecast_delivery import finalize_forecast_result


def get_motor_status(motor_id: str) -> str:
    """Return the latest complete sensor snapshot for a motor.

    Args:
        motor_id: Device identifier such as ``M1`` or ``STM32-Node-1``.
    """
    db = SessionLocal()
    try:
        record = latest_reading(db, motor_id)
        if record is None:
            return json.dumps(
                {"error": f"資料庫中找不到馬達 {motor_id} 的感測器資料"},
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "motor_id": record.motor_id,
                "temperature": record.temperature,
                "humidity": record.humidity,
                "accel_x": record.accel_x,
                "accel_y": record.accel_y,
                "accel_z": record.accel_z,
                "vibration": record.vibration,
                "status": record.status,
                "recorded_at": (
                    record.recorded_at.isoformat()
                    if record.recorded_at
                    else None
                ),
            },
            ensure_ascii=False,
        )
    except Exception as error:
        return json.dumps({"error": str(error)}, ensure_ascii=False)
    finally:
        db.close()


def get_temperature_forecast(
    motor_id: str,
    training_motor_id: str | None = None,
) -> str:
    """Predict a motor's temperature 30 minutes ahead and expose its charts.

    Args:
        motor_id: Device whose latest sensor vector is used for inference.
        training_motor_id: Optional device that supplies the persisted model.
    """
    db = SessionLocal()
    request_started = time.perf_counter()
    try:
        result = forecast_temperature(
            db,
            motor_id,
            auto_train=True,
            training_motor_id=training_motor_id,
        )
        result["tool_duration_ms"] = round(
            (time.perf_counter() - request_started) * 1000,
            6,
        )
        finalize_forecast_result(result)
        return json.dumps(result, ensure_ascii=False)
    except ForecastError as error:
        db.rollback()
        return json.dumps({"error": str(error)}, ensure_ascii=False)
    except Exception as error:
        db.rollback()
        return json.dumps(
            {"error": f"預測失敗：{error}"},
            ensure_ascii=False,
        )
    finally:
        db.close()
