"""Application startup and shutdown resource management."""

from __future__ import annotations

from contextlib import asynccontextmanager

from database import Base, SessionLocal, engine
from migrations import migrate_sensor_columns
from models import MotorSensorData
from reporting import export_demo_datasets
from seed_data import (
    DEMO_INFERENCE_MOTOR_ID,
    DEMO_INFERENCE_READING_COUNT,
    DEMO_TRAINING_MOTOR_ID,
    build_demo_inference_readings,
    build_demo_readings,
)
from services.agent_service import close_agent_client, initialize_agent_client
from settings import SEED_DEMO_DATA


def initialize_database() -> None:
    """Create schema, migrate columns, and optionally refresh demo data."""
    Base.metadata.create_all(bind=engine)
    migrate_sensor_columns(engine)
    db = SessionLocal()
    try:
        if db.query(MotorSensorData).count() == 0:
            db.add_all(
                [
                    MotorSensorData(
                        motor_id="M1",
                        temperature=88.5,
                        humidity=68.0,
                        accel_x=0.12,
                        accel_y=0.08,
                        accel_z=1.03,
                        vibration=1.5,
                        status="warning",
                    ),
                    MotorSensorData(
                        motor_id="M2",
                        temperature=42.0,
                        humidity=51.0,
                        accel_x=0.02,
                        accel_y=0.01,
                        accel_z=0.99,
                        vibration=0.2,
                        status="normal",
                    ),
                ]
            )

        if SEED_DEMO_DATA:
            demo_datasets = (
                (DEMO_TRAINING_MOTOR_ID, build_demo_readings),
                (DEMO_INFERENCE_MOTOR_ID, build_demo_inference_readings),
            )
            for demo_motor_id, build_readings in demo_datasets:
                demo_query = db.query(MotorSensorData).filter(
                    MotorSensorData.motor_id == demo_motor_id
                )
                demo_exists = demo_query.first()
                if (
                    demo_motor_id == DEMO_INFERENCE_MOTOR_ID
                    and demo_query.filter(
                        MotorSensorData.status == "demo-inference"
                    ).count()
                    != DEMO_INFERENCE_READING_COUNT
                ):
                    demo_query.filter(
                        MotorSensorData.status == "demo-inference"
                    ).delete(synchronize_session=False)
                    demo_exists = None
                if demo_exists is None:
                    db.add_all(
                        MotorSensorData(**reading)
                        for reading in build_readings()
                    )
        db.commit()
        if SEED_DEMO_DATA:
            export_demo_datasets(db)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@asynccontextmanager
async def application_lifespan(_app):
    try:
        initialize_agent_client()
        initialize_database()
        yield
    finally:
        await close_agent_client()
        engine.dispose()
