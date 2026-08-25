"""Application startup and shutdown resource management."""

from __future__ import annotations

from contextlib import asynccontextmanager

from core.config import settings
from core.database import Base, SessionLocal, engine
from migrations import migrate_sensor_columns
from models import MotorSensorData
from reports.io import export_demo_datasets
from seed_data import (
    DEMO_INFERENCE_MOTOR_ID,
    DEMO_INFERENCE_READING_COUNT,
    DEMO_TRAINING_MOTOR_ID,
    build_demo_inference_readings,
    build_demo_readings,
)
from services.agent_service import close_agent_client, initialize_agent_client


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
    training_exists = (
        db.query(MotorSensorData.id)
        .filter(MotorSensorData.motor_id == DEMO_TRAINING_MOTOR_ID)
        .first()
        is not None
    )
    if not training_exists:
        db.add_all(
            MotorSensorData(**reading) for reading in build_demo_readings()
        )

    inference_query = db.query(MotorSensorData).filter(
        MotorSensorData.motor_id == DEMO_INFERENCE_MOTOR_ID,
        MotorSensorData.status == "demo-inference",
    )
    if inference_query.count() != DEMO_INFERENCE_READING_COUNT:
        # Replace only generated demo rows; real rows using the same motor ID are
        # deliberately left untouched.
        inference_query.delete(synchronize_session=False)
        db.add_all(
            MotorSensorData(**reading)
            for reading in build_demo_inference_readings()
        )


def initialize_database() -> None:
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
            export_demo_datasets(db)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@asynccontextmanager
async def application_lifespan(_app):
    """Own long-lived Agent, database, and connection-pool resources."""
    try:
        initialize_agent_client()
        initialize_database()
        yield
    finally:
        await close_agent_client()
        engine.dispose()
