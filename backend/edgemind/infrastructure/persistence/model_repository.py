"""SQLAlchemy implementation of the forecast-model repository port."""

from datetime import datetime, timezone
import json

from sqlalchemy.orm import Session

from edgemind.infrastructure.persistence.models import TemperatureForecastModel


class SqlAlchemyForecastModelRepository:
    """Hide JSON and ORM persistence details from forecast use cases."""

    def __init__(self, session: Session):
        """Bind repository operations to the unit-of-work session."""
        self._session = session

    def _get_row(self, motor_id: str) -> TemperatureForecastModel | None:
        """Fetch one storage row without exposing it outside this adapter."""
        return (
            self._session.query(TemperatureForecastModel)
            .filter(TemperatureForecastModel.motor_id == motor_id)
            .first()
        )

    def get_payload(self, motor_id: str) -> dict | None:
        """Return a deserialized model or ``None`` when it has not been trained."""
        row = self._get_row(motor_id)
        return json.loads(row.model_json) if row is not None else None

    def save_payload(self, motor_id: str, payload: dict) -> None:
        """Stage an insert or update for the unit of work to commit."""
        row = self._get_row(motor_id)
        if row is None:
            row = TemperatureForecastModel(
                motor_id=motor_id,
                model_json="{}",
                sample_count=0,
            )
            self._session.add(row)
        row.model_json = json.dumps(payload, ensure_ascii=False)
        row.sample_count = payload["sample_count"]
        row.mae = payload["mae"]
        row.rmse = payload["rmse"]
        row.trained_at = datetime.now(timezone.utc)
