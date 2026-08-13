import math

from fastapi import APIRouter, Depends, HTTPException, status

from database import get_db
from models import MotorSensorData
from schemas import SensorReadingRequest


router = APIRouter(prefix="/api/sensor-readings", tags=["sensors"])


@router.post("", status_code=status.HTTP_201_CREATED)
def create_sensor_reading(
    request: SensorReadingRequest,
    db=Depends(get_db),
):
    """Receive one complete feature vector from an edge sensor."""
    values = (
        request.temperature,
        request.humidity,
        request.accel_x,
        request.accel_y,
        request.accel_z,
    )
    if not all(math.isfinite(value) for value in values):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="所有感測值都必須是有限數值",
        )

    record = MotorSensorData(
        motor_id=request.motor_id,
        temperature=request.temperature,
        humidity=request.humidity,
        accel_x=request.accel_x,
        accel_y=request.accel_y,
        accel_z=request.accel_z,
        vibration=math.sqrt(
            request.accel_x ** 2
            + request.accel_y ** 2
            + request.accel_z ** 2
        ),
        status=request.status,
    )
    if request.recorded_at is not None:
        record.recorded_at = request.recorded_at
    try:
        db.add(record)
        db.commit()
        db.refresh(record)
    except Exception as error:
        db.rollback()
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
