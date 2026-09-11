"""Dependency-inversion ports owned by the application layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Sequence

from edgemind.domain.entities import SensorReading
from edgemind.domain.ridge_experiments import RidgeParameters


class SensorRepository(Protocol):
    """Persistence operations required by sensor use cases."""

    def list_readings(self, motor_id: str) -> list[SensorReading]:
        """Return one device history in chronological order."""
        ...

    def latest_reading(self, motor_id: str) -> SensorReading | None:
        """Return the newest reading for one device."""
        ...

    def add(self, reading: SensorReading) -> SensorReading:
        """Stage one new sensor reading and return generated fields."""
        ...

    def list_device_summaries(self) -> list[dict]:
        """Return aggregate data availability for each device."""
        ...


class ForecastModelRepository(Protocol):
    """Persistence operations for serialized forecast model payloads."""

    def get_payload(self, motor_id: str) -> dict | None:
        """Return one deserialized model payload when present."""
        ...

    def save_payload(self, motor_id: str, payload: dict) -> None:
        """Stage one serialized model insert or update."""
        ...


class UnitOfWork(Protocol):
    """Transaction boundary shared by one application use case."""

    sensors: SensorRepository
    forecast_models: ForecastModelRepository

    def __enter__(self) -> "UnitOfWork":
        """Open the transaction and its repositories."""
        ...

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Rollback failed work and close transaction resources."""
        ...

    def commit(self) -> None:
        """Commit all repository changes atomically."""
        ...

    def rollback(self) -> None:
        """Discard all uncommitted repository changes."""
        ...


class ForecastReportGateway(Protocol):
    """Filesystem reporting operations required after numerical inference."""

    def create_inference_report(
        self,
        *,
        training_records: Sequence[SensorReading],
        inference_records: Sequence[SensorReading],
        model_payload: dict,
        inference_motor_id: str,
        training_motor_id: str,
        latest_prediction: float,
        generated_at: Any,
        model_inference_duration_ms: float,
        process_cpu_time_ms: float,
    ) -> dict:
        """Generate CSV, SVG, and metadata for one inference run."""
        ...

    def finalize_forecast_result(self, result: dict) -> dict:
        """Attach performance aggregation and public chart metadata."""
        ...


class ReportQueryGateway(Protocol):
    """Read-only access used by report presentation endpoints."""

    def performance_summary(self) -> dict:
        """Return statistics aggregated across finalized runs."""
        ...

    def resolve_chart_artifact(self, relative_path: str) -> Path | None:
        """Resolve one safe public chart path when it exists."""
        ...


class RidgeForecastReportGateway(Protocol):
    """Publish live Ridge forecast reports."""

    def create_ridge_inference_report(
        self,
        *,
        training_records: Sequence[SensorReading],
        inference_records: Sequence[SensorReading],
        parameters: RidgeParameters,
        forecast: dict,
        training_duration_ms: float,
        model_inference_duration_ms: float,
        process_cpu_time_ms: float,
    ) -> dict:
        """Write CSV, SVG, and metadata using the selected live Ridge model."""
        ...

    def finalize_forecast_result(self, result: dict) -> dict:
        """Attach performance aggregation and public chart metadata."""
        ...


class AgentModelGateway(Protocol):
    """Text-model behavior needed by the Agent orchestration use case."""

    async def decide(self, message: str) -> "ModelReply":
        """Return a direct response or structured tool calls."""
        ...

    async def summarize(self, message: str, tool_results: list[dict]) -> str:
        """Create a grounded answer from completed tool payloads."""
        ...

    async def close(self) -> None:
        """Release model-client resources."""
        ...


class DiagnosticTools(Protocol):
    """Allow-listed diagnostic operations callable by the Agent."""

    def execute(self, name: str, arguments: dict) -> dict:
        """Execute an allow-listed tool and return its payload."""
        ...
