"""Tests for the compact Direct Ridge versus history-aware Ridge workflow."""

from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from edgemind.domain.ridge_experiments import (
    DIRECT_MODEL,
    HISTORY_FEATURE_NAMES,
    HISTORY_MODEL,
    build_experiment_examples,
    chronological_split,
    run_ridge_experiment,
)
from edgemind.infrastructure.reporting.context import ReportContext
from edgemind.infrastructure.reporting.ridge_experiments import (
    get_ridge_experiment,
    save_ridge_experiment,
)


def sensor_history(count: int = 360) -> list[SimpleNamespace]:
    """Create a signal whose direction is visible only in its history."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        SimpleNamespace(
            recorded_at=start + timedelta(minutes=5 * index),
            temperature=30 + 4 * math.sin(index * 2 * math.pi / 48),
            humidity=50.0,
            accel_x=0.0,
            accel_y=0.0,
            accel_z=1.0,
        )
        for index in range(count)
    ]


class RidgeExperimentTests(unittest.TestCase):
    """Verify common sampling, chronological isolation, and report output."""

    def test_builds_exact_common_samples_and_history_features(self):
        records = sensor_history(120)
        examples = build_experiment_examples(records)

        self.assertEqual(len(examples), 103)
        self.assertEqual(len(examples[0].direct_features), 5)
        self.assertEqual(
            len(examples[0].history_features),
            len(HISTORY_FEATURE_NAMES),
        )
        self.assertEqual(len(HISTORY_FEATURE_NAMES), 35)
        self.assertEqual(
            examples[0].target_time - examples[0].origin_time,
            timedelta(minutes=30),
        )

    def test_chronological_split_purges_overlapping_horizons(self):
        examples = build_experiment_examples(sensor_history(240))
        split = chronological_split(examples)

        self.assertLess(
            split["training"][-1].target_time,
            split["validation"][0].origin_time,
        )
        self.assertLess(
            split["validation"][-1].target_time,
            split["testing"][0].origin_time,
        )
        self.assertEqual(
            len(examples) - sum(len(group) for group in split.values()),
            12,
        )

    def test_history_model_improves_directional_temperature_signal(self):
        result = run_ridge_experiment(sensor_history())
        by_name = {model["model_name"]: model for model in result["models"]}

        self.assertEqual(result["comparison"]["winner"], HISTORY_MODEL)
        self.assertLess(
            by_name[HISTORY_MODEL]["test_metrics"]["mae"],
            by_name[DIRECT_MODEL]["test_metrics"]["mae"],
        )
        self.assertGreater(
            result["comparison"]["test_mae_improvement_percent"],
            20,
        )

    def test_report_uses_datetime_directory_and_round_trips(self):
        result = run_ridge_experiment(sensor_history(180))
        result.update(
            {
                "experiment_id": "a" * 32,
                "created_at": "2026-09-01T01:02:03.456789+00:00",
                "training_motor_id": "TRAIN-1",
                "evaluation_motor_id": None,
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            context = ReportContext(
                root=Path(temporary),
                timezone=timezone(timedelta(hours=8)),
                anomaly_temperature_threshold=35.0,
            )
            saved = save_ridge_experiment(context, result)
            directory = Path(temporary) / saved["artifacts"]["run_directory"]

            self.assertIn("ridge_experiments/2026-09-01/09-02-03", directory.as_posix())
            self.assertTrue((directory / "result.json").is_file())
            self.assertTrue((directory / "model_comparison.csv").is_file())
            self.assertTrue((directory / "test_predictions.csv").is_file())
            restored = get_ridge_experiment(context, "a" * 32)
            self.assertEqual(restored["experiment_id"], "a" * 32)
            self.assertNotIn("predictions", restored)


if __name__ == "__main__":
    unittest.main()
