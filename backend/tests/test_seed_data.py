from datetime import datetime, timedelta, timezone
import math
from types import SimpleNamespace
import unittest

from forecasting import MIN_TRAINING_SAMPLES, build_training_examples
from seed_data import (
    DEMO_INFERENCE_MOTOR_ID,
    DEMO_INFERENCE_READING_COUNT,
    DEMO_INTERVAL_MINUTES,
    DEMO_MOTOR_ID,
    DEMO_READING_COUNT,
    build_demo_inference_readings,
    build_demo_readings,
)


class DemoSeedTests(unittest.TestCase):
    def test_demo_history_is_complete_and_trainable(self):
        reference_time = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)
        readings = build_demo_readings(reference_time)

        self.assertEqual(len(readings), DEMO_READING_COUNT)
        self.assertTrue(
            all(reading["motor_id"] == DEMO_MOTOR_ID for reading in readings)
        )
        self.assertEqual(readings[-1]["recorded_at"], reference_time)
        self.assertTrue(
            all(
                current["recorded_at"] - previous["recorded_at"]
                == timedelta(minutes=DEMO_INTERVAL_MINUTES)
                for previous, current in zip(readings, readings[1:])
            )
        )

        numeric_fields = (
            "temperature",
            "humidity",
            "accel_x",
            "accel_y",
            "accel_z",
        )
        self.assertTrue(
            all(
                math.isfinite(reading[field])
                for reading in readings
                for field in numeric_fields
            )
        )

        records = [SimpleNamespace(**reading) for reading in readings]
        examples = build_training_examples(records)
        self.assertGreaterEqual(len(examples), MIN_TRAINING_SAMPLES)
        self.assertEqual(len(examples), 31)

    def test_demo_2_has_enough_history_for_truth_evaluation(self):
        reference_time = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)
        readings = build_demo_inference_readings(reference_time)

        self.assertEqual(len(readings), DEMO_INFERENCE_READING_COUNT)
        self.assertTrue(
            all(
                reading["motor_id"] == DEMO_INFERENCE_MOTOR_ID
                for reading in readings
            )
        )
        self.assertEqual(readings[-1]["recorded_at"], reference_time)
        records = [SimpleNamespace(**reading) for reading in readings]
        self.assertGreaterEqual(
            len(build_training_examples(records)),
            MIN_TRAINING_SAMPLES,
        )


if __name__ == "__main__":
    unittest.main()
