"""Write endpoint for complete edge-sensor feature vectors."""

from typing import Callable

from fastapi import APIRouter, HTTPException, status

from edgemind.application.ports import UnitOfWork
from edgemind.application.sensors import SensorService, SensorValidationError
from edgemind.presentation.schemas import SensorReadingRequest


def create_router(
    sensors: SensorService,
    uow_factory: Callable[[], UnitOfWork],
) -> APIRouter:
    """Create ingestion routes bound to the sensor application service."""
    router = APIRouter(prefix="/api/sensor-readings", tags=["sensors"])

    @router.post("", status_code=status.HTTP_201_CREATED)
    def create_sensor_reading(request: SensorReadingRequest):
        """Validate and store one complete edge-sensor vector."""
        try:
            with uow_factory() as uow:
                record = sensors.create(
                    uow,
                    motor_id=request.motor_id,
                    temperature=request.temperature,
                    humidity=request.humidity,
                    accel_x=request.accel_x,
                    accel_y=request.accel_y,
                    accel_z=request.accel_z,
                    status=request.status,
                    recorded_at=request.recorded_at,
                )
        except SensorValidationError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"感測資料寫入失敗：{error}",
            ) from error
        return {
            "id": record.id,
            "motor_id": record.motor_id,
            "temperature": record.temperature,
            "humidity": record.humidity,
            "accel_x": record.accel_x,
            "accel_y": record.accel_y,
            "accel_z": record.accel_z,
            "recorded_at": (
                record.recorded_at.isoformat()
                if record.recorded_at
                else None
            ),
        }

    return router
