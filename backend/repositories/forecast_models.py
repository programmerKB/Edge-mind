"""Persistence helpers for serialized temperature forecast models."""

from datetime import datetime, timezone
import json

from sqlalchemy.orm import Session

from models import TemperatureForecastModel


def get_model(
    db: Session,
    motor_id: str,
) -> TemperatureForecastModel | None:
    """Fetch the latest persisted model for a device."""
    return (
        db.query(TemperatureForecastModel)
        .filter(TemperatureForecastModel.motor_id == motor_id)
        .first()
    )


def save_model(db: Session, motor_id: str, payload: dict) -> None:
    """Insert or update one model and commit it as a single transaction."""
    stored = get_model(db, motor_id)
    if stored is None:
        stored = TemperatureForecastModel(
            motor_id=motor_id,
            model_json="{}",
            sample_count=0,
        )
        db.add(stored)
    stored.model_json = json.dumps(payload, ensure_ascii=False)
    stored.sample_count = payload["sample_count"]
    stored.mae = payload["mae"]
    stored.rmse = payload["rmse"]
    stored.trained_at = datetime.now(timezone.utc)
    db.commit()
