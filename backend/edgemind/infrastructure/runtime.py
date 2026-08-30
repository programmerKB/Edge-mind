"""Application startup, demo seeding, and shutdown resource management."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from edgemind.application.agent import AgentService
from edgemind.domain.demo_data import (
    DEMO_INFERENCE_MOTOR_ID,
    DEMO_INFERENCE_READING_COUNT,
    DEMO_INFERENCE_STATUS_PREFIX,
    DEMO_READING_COUNT,
    DEMO_TRAINING_MOTOR_ID,
    DEMO_TRAINING_STATUS_PREFIX,
    build_demo_inference_readings,
    build_demo_readings,
)
from edgemind.infrastructure.config import settings
from edgemind.infrastructure.persistence.database import Base, SessionLocal, engine
from edgemind.infrastructure.persistence.migrations import migrate_sensor_columns
from edgemind.infrastructure.persistence.models import MotorSensorData
from edgemind.infrastructure.persistence.sensor_repository import (
    SqlAlchemySensorRepository,
)
from edgemind.infrastructure.reporting.gateway import FilesystemReportGateway


BASELINE_DEVICES = (
    {
        "motor_id": "M1",
        "temperature": 88.5,
        "humidity": 68.0,
        "accel_x": 0.12,
        "accel_y": 0.08,
        "accel_z": 1.03,
        "vibration": 1.5,
        "status": "warning",
    },
    {
        "motor_id": "M2",
        "temperature": 42.0,
        "humidity": 51.0,
        "accel_x": 0.02,
        "accel_y": 0.01,
        "accel_z": 0.99,
        "vibration": 0.2,
        "status": "normal",
    },
)


def _seed_baseline_devices(db) -> None:
    """Insert the two status-query examples only into a completely empty table."""
    if db.query(MotorSensorData.id).first() is None:
        db.add_all(MotorSensorData(**row) for row in BASELINE_DEVICES)


def _seed_demo_history(db) -> None:
    """Create deterministic training and inference histories idempotently."""
    training_query = db.query(MotorSensorData).filter(
        MotorSensorData.motor_id == DEMO_TRAINING_MOTOR_ID,
        MotorSensorData.status.like("demo%"),
    )
    training_current = training_query.filter(
        MotorSensorData.status.like(f"{DEMO_TRAINING_STATUS_PREFIX}-%")
    ).count()
    training_needs_refresh = (
        training_current != DEMO_READING_COUNT
        or training_query.count() != DEMO_READING_COUNT
    )

    inference_query = db.query(MotorSensorData).filter(
        MotorSensorData.motor_id == DEMO_INFERENCE_MOTOR_ID,
        MotorSensorData.status.like("demo%"),
    )
    inference_current = inference_query.filter(
        MotorSensorData.status.like(f"{DEMO_INFERENCE_STATUS_PREFIX}-%")
    ).count()
    inference_needs_refresh = (
        inference_current != DEMO_INFERENCE_READING_COUNT
        or inference_query.count() != DEMO_INFERENCE_READING_COUNT
    )
    if not training_needs_refresh and not inference_needs_refresh:
        return

    # Treat the two generated devices as one versioned release.  Refreshing
    # only one side at a later origin can make training labels extend beyond
    # the inference origin and invalidate cross-device forecasting.
    reference_time = datetime.now(timezone.utc)
    training_query.delete(synchronize_session=False)
    inference_query.delete(synchronize_session=False)
    db.add_all(
        MotorSensorData(**reading)
        for reading in build_demo_readings(reference_time)
    )
    db.add_all(
        MotorSensorData(**reading)
        for reading in build_demo_inference_readings(reference_time)
    )


def initialize_database(reports: FilesystemReportGateway) -> None:
    """Create schema, apply small migrations, and optionally refresh demo data."""
    Base.metadata.create_all(bind=engine)
    migrate_sensor_columns(engine)
    db = SessionLocal()
    try:
        _seed_baseline_devices(db)
        if settings.seed_demo_data:
            _seed_demo_history(db)
        db.commit()
        if settings.seed_demo_data:
            sensors = SqlAlchemySensorRepository(db)
            reports.export_demo_datasets(
                sensors.list_readings(DEMO_TRAINING_MOTOR_ID),
                sensors.list_readings(DEMO_INFERENCE_MOTOR_ID),
            )
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_lifespan(
    agent: AgentService,
    reports: FilesystemReportGateway,
):
    """Create FastAPI lifespan wiring without importing FastAPI in this layer."""

    @asynccontextmanager
    async def application_lifespan(_app):
        """Own database schema, Agent gateway, and connection-pool resources."""
        try:
            initialize_database(reports)
            yield
        finally:
            await agent.close()
            engine.dispose()

    return application_lifespan
