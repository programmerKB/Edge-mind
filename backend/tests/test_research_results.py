"""Artifact tests for completed forecasting research experiments."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from edgemind.infrastructure.reporting.research_results import (
    get_research_experiment,
    get_research_forecast,
    save_research_experiment,
    save_research_forecast,
)


class ResearchResultTests(unittest.TestCase):
    """Ensure experiment outputs remain reproducible and traversal-safe."""

    def test_saves_json_and_flat_metric_tables(self):
        experiment_id = "1" * 32
        result = {
            "experiment_id": experiment_id,
            "status": "completed",
            "models": {
                "persistence": {
                    "status": "completed",
                    "test": {
                        "overall": {"mae": 0.5},
                        "by_horizon": {
                            "5": {"mae": 0.2},
                            "30": {"mae": 0.8},
                        },
                        "prediction_records": [
                            {
                                "sample_id": "sample-1",
                                "device_id": "MOTOR-A",
                                "origin_time": "2026-01-01T00:00:00+00:00",
                                "target_time": "2026-01-01T00:05:00+00:00",
                                "horizon_minutes": 5,
                                "actual": 31.0,
                                "predicted": 31.5,
                                "error": 0.5,
                            }
                        ],
                        "risk": {
                            "event_evaluation": {
                                "status": "available",
                                "metrics": {"event_recall": 1.0},
                                "events": {"records": []},
                                "warnings": {"records": []},
                                "matches": [],
                            }
                        },
                    },
                }
            },
            "dataset": {
                "split_manifest_records": [
                    {
                        "sample_id": "sample-1",
                        "device_id": "MOTOR-A",
                        "origin_time": "2026-01-01T00:00:00+00:00",
                        "history_start_time": "2025-12-31T23:05:00+00:00",
                        "history_end_time": "2026-01-01T00:00:00+00:00",
                        "first_target_time": "2026-01-01T00:05:00+00:00",
                        "last_target_time": "2026-01-01T00:30:00+00:00",
                        "split": "test",
                    }
                ]
            },
            "ablations": {
                "results": {
                    "temperature_only": {"test_mae": 0.7},
                }
            },
            "statistical_comparisons": [
                {
                    "comparison": "ridge_vs_persistence",
                    "status": "available",
                    "candidate_minus_reference": -0.1,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            saved = save_research_experiment(result, Path(temporary))

            artifacts = saved["artifacts"]
            self.assertTrue(Path(artifacts["result_json"]).is_file())
            self.assertTrue(Path(artifacts["model_comparison_csv"]).is_file())
            self.assertTrue(Path(artifacts["horizon_metrics_csv"]).is_file())
            self.assertTrue(Path(artifacts["feature_ablations_csv"]).is_file())
            self.assertTrue(Path(artifacts["predictions_csv"]).is_file())
            self.assertTrue(Path(artifacts["split_manifest_csv"]).is_file())
            self.assertTrue(Path(artifacts["risk_events_csv"]).is_file())
            self.assertTrue(
                Path(artifacts["statistical_comparisons_csv"]).is_file()
            )
            self.assertTrue(
                all(len(value) == 64 for value in artifacts["sha256"].values())
            )
            loaded = get_research_experiment(experiment_id, Path(temporary))
            self.assertEqual(loaded["experiment_id"], experiment_id)
            self.assertNotIn(
                "prediction_records",
                loaded["models"]["persistence"]["test"],
            )
            self.assertNotIn("split_manifest_records", loaded["dataset"])
            with Path(artifacts["horizon_metrics_csv"]).open(
                encoding="utf-8-sig", newline=""
            ) as source:
                rows = list(csv.DictReader(source))
            self.assertEqual({row["horizon_minutes"] for row in rows}, {"5", "30"})
            with Path(artifacts["predictions_csv"]).open(
                encoding="utf-8-sig", newline=""
            ) as source:
                prediction_rows = list(csv.DictReader(source))
            self.assertEqual(prediction_rows[0]["model"], "persistence")
            self.assertEqual(prediction_rows[0]["split"], "test")
            self.assertEqual(prediction_rows[0]["sample_id"], "sample-1")
            with Path(artifacts["split_manifest_csv"]).open(
                encoding="utf-8-sig", newline=""
            ) as source:
                split_rows = list(csv.DictReader(source))
            self.assertEqual(split_rows[0]["split"], "test")
            json.loads(Path(artifacts["result_json"]).read_text(encoding="utf-8"))

    def test_rejects_invalid_identifiers_and_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                save_research_experiment({"experiment_id": "../escape"}, root)
            self.assertIsNone(get_research_experiment("../escape", root))

    def test_saves_live_forecast_separately_with_pending_truth(self):
        forecast_id = "a" * 32
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            saved = save_research_forecast(
                {
                    "forecast_id": forecast_id,
                    "truth_status": "pending",
                    "trajectory": [
                        {
                            "horizon_minutes": 5,
                            "predicted_temperature_c": 32.1,
                            "truth_status": "pending",
                        }
                    ],
                },
                root,
            )

            self.assertIn("research_forecasts", saved["artifacts"]["run_directory"])
            self.assertTrue(Path(saved["artifacts"]["result_json"]).is_file())
            loaded = get_research_forecast(forecast_id, root)
            self.assertEqual(loaded["truth_status"], "pending")
            self.assertIsNone(get_research_forecast("../escape", root))


if __name__ == "__main__":
    unittest.main()
