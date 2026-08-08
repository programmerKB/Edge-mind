from datetime import datetime, timedelta, timezone
import math
from types import SimpleNamespace
import unittest

from forecasting import (
    FEATURE_NAMES,
    ForecastError,
    TrainingExample,
    build_training_examples,
    forecast_temperature,
    predict_with_payload,
    train_and_save_model,
    train_temperature_model,
)
try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from database import Base
    from models import MotorSensorData

    HAS_SQLALCHEMY = True
except ModuleNotFoundError:
    HAS_SQLALCHEMY = False



class ForecastingTests(unittest.TestCase):
    def test_builds_targets_within_30_minute_tolerance(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        records = [
            SimpleNamespace(
                recorded_at=start + timedelta(minutes=5 * index),
                temperature=25 + index * 0.2,
                humidity=50 + index % 3,
                accel_x=math.sin(index),
                accel_y=math.cos(index),
                accel_z=1 + index * 0.001,
            )
            for index in range(20)
        ]

        examples = build_training_examples(records)

        self.assertEqual(len(examples), 15)
        self.assertTrue(
            all(
                timedelta(minutes=25)
                <= example.target_time - example.source_time
                <= timedelta(minutes=35)
                for example in examples
            )
        )
        self.assertEqual(examples[0].target_temperature, records[6].temperature)

    def test_ridge_model_uses_all_five_features(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        examples = []
        for index in range(40):
            features = (
                20 + index * 0.2,
                45 + (index * 7) % 13,
                math.sin(index * 0.6),
                math.cos(index * 0.4),
                (index % 5) - 2,
            )
            target = (
                4
                + 0.7 * features[0]
                + 0.05 * features[1]
                + 0.8 * features[2]
                - 0.4 * features[3]
                + 0.3 * features[4]
            )
            examples.append(
                TrainingExample(
                    source_time=start + timedelta(minutes=5 * index),
                    target_time=start + timedelta(minutes=5 * index + 30),
                    features=features,
                    target_temperature=target,
                )
            )

        payload = train_temperature_model(examples)
        prediction = predict_with_payload(payload, examples[-1].features)

        self.assertEqual(payload["feature_names"], list(FEATURE_NAMES))
        self.assertLess(payload["mae"], 0.05)
        self.assertAlmostEqual(
            prediction,
            examples[-1].target_temperature,
            delta=0.05,
        )

    def test_rejects_insufficient_history(self):
        with self.assertRaises(ForecastError):
            train_temperature_model([])

    @unittest.skipUnless(HAS_SQLALCHEMY, "SQLAlchemy is not installed")
    def test_database_training_and_prediction_round_trip(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session = sessionmaker(bind=engine)()
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        try:
            for index in range(30):
                session.add(
                    MotorSensorData(
                        motor_id="TEST-1",
                        temperature=30 + index * 0.1,
                        humidity=55 + index % 4,
                        accel_x=math.sin(index) * 0.1,
                        accel_y=math.cos(index) * 0.1,
                        accel_z=1 + index * 0.001,
                        vibration=1,
                        status="normal",
                        recorded_at=start + timedelta(minutes=5 * index),
                    )
                )
            session.commit()

            trained = train_and_save_model(session, "TEST-1")
            result = forecast_temperature(
                session,
                "TEST-1",
                auto_train=False,
            )

            self.assertEqual(trained["sample_count"], 24)
            self.assertEqual(result["forecast_horizon_minutes"], 30)
            self.assertEqual(
                set(result["features"]),
                set(FEATURE_NAMES),
            )
            self.assertTrue(math.isfinite(result["predicted_temperature"]))
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()
