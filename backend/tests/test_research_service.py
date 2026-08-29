"""Application tests for the research experiment lifecycle."""

from datetime import datetime, timedelta, timezone
import math
import unittest

from edgemind.application.research import ResearchService, _reading_hash
from edgemind.domain.entities import SensorReading
from edgemind.domain.research import ResearchError, build_default_registry


def readings(motor_id: str, status: str = "collected") -> list[SensorReading]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        SensorReading(
            motor_id=motor_id,
            recorded_at=start + timedelta(minutes=5 * index),
            temperature=25 + 0.04 * index + 0.3 * math.sin(index / 5),
            humidity=60 - 0.03 * index + math.cos(index / 7),
            accel_x=0.08 * math.sin(index / 3),
            accel_y=0.07 * math.cos(index / 4),
            accel_z=1 + 0.02 * math.sin(index / 6),
            vibration=1.0,
            status=status,
        )
        for index in range(100)
    ]


class FakeSensors:
    def __init__(self, datasets):
        self.datasets = datasets

    def list_readings(self, motor_id):
        return list(self.datasets.get(motor_id, ()))

    def list_device_summaries(self):
        return [
            {
                "motor_id": motor_id,
                "reading_count": len(rows),
                "complete_reading_count": len(rows),
                "feature_completeness_percent": 100.0,
                "first_recorded_at": rows[0].recorded_at.isoformat(),
                "last_recorded_at": rows[-1].recorded_at.isoformat(),
            }
            for motor_id, rows in self.datasets.items()
        ]


class FakeUnitOfWork:
    def __init__(self, datasets):
        self.sensors = FakeSensors(datasets)


class FakeResearchReports:
    def __init__(self):
        self.results = {}
        self.forecasts = {}

    def save_research_experiment(self, result):
        saved = {**result, "artifacts": {"result_json": "/tmp/result.json"}}
        self.results[result["experiment_id"]] = saved
        return saved

    def get_research_experiment(self, experiment_id):
        return self.results.get(experiment_id)

    def save_research_forecast(self, result):
        saved = {**result, "artifacts": {"result_json": "/tmp/forecast.json"}}
        self.forecasts[result["forecast_id"]] = saved
        return saved

    def get_research_forecast(self, forecast_id):
        return self.forecasts.get(forecast_id)


class ResearchServiceTests(unittest.TestCase):
    def setUp(self):
        self.datasets = {"REAL-A": readings("REAL-A"), "REAL-B": readings("REAL-B")}
        self.reports = FakeResearchReports()
        self.service = ResearchService(self.reports, build_default_registry())
        self.uow = FakeUnitOfWork(self.datasets)

    def test_configuration_marks_small_datasets_as_pipeline_only(self):
        config = self.service.configuration(self.uow)

        self.assertEqual(config["defaults"]["history_minutes"], 60)
        self.assertEqual(config["defaults"]["gap_steps"], 6)
        self.assertTrue(config["motors"][0]["pipeline_eligible"])
        self.assertEqual(
            config["motors"][0]["pipeline_eligibility_scope"],
            "default_60min_history_30min_horizon_count_precheck",
        )
        self.assertTrue(config["motors"][0]["runtime_quality_audit_required"])
        self.assertFalse(config["motors"][0]["study_eligible"])
        self.assertEqual(config["constraints"]["minimum_pipeline_readings"], 36)
        self.assertIn(
            "sequence_count",
            config["constraints"]["minimum_pipeline_readings_formula"],
        )
        statuses = {item["name"]: item["status"] for item in config["models"]}
        self.assertEqual(statuses["ridge_direct"], "available")
        self.assertEqual(statuses["tcn"], "unavailable")
        self.assertEqual(
            set(config["defaults"]["model_names"]),
            {
                "ridge_direct", "ridge_history_trend", "dlinear",
                "lstm", "tcn", "patchtst",
            },
        )

    def test_runs_cross_device_experiment_and_persists_fingerprints(self):
        result = self.service.run(
            self.uow,
            training_motor_id="REAL-A",
            evaluation_motor_id="REAL-B",
            model_names=("ridge_direct", "ridge_history_trend"),
            include_ablations=False,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["experiment_id"]), 32)
        self.assertEqual(set(result["models"]), {
            "ridge_direct", "ridge_history_trend"
        })
        self.assertTrue(
            result["models"]["ridge_history_trend"]["cross_device"]["is_cross_device"]
        )
        self.assertFalse(result["dataset_provenance"]["research_claims_allowed"])
        self.assertEqual(
            result["dataset_provenance"]["provenance_status"],
            "unverified",
        )
        self.assertTrue(result["dataset_provenance"]["provenance_audit_required"])
        self.assertEqual(len(result["dataset_provenance"]["training_sha256"]), 64)
        self.assertIs(self.service.get(result["experiment_id"]), result)

    def test_any_known_synthetic_row_blocks_research_claims(self):
        mixed = readings("MIXED")
        mixed[0] = readings("MIXED", status="synthetic-normal")[0]
        uow = FakeUnitOfWork({"MIXED": mixed})

        result = self.service.run(
            uow,
            training_motor_id="MIXED",
            evaluation_motor_id=None,
            model_names=("ridge_direct",),
            include_ablations=False,
        )

        self.assertTrue(result["dataset_provenance"]["synthetic_or_demo_detected"])
        self.assertFalse(result["dataset_provenance"]["research_claims_allowed"])
        self.assertEqual(
            result["dataset_provenance"]["provenance_status"],
            "synthetic_or_demo",
        )
        self.assertNotEqual(_reading_hash(readings("MIXED")), _reading_hash(mixed))

    def test_rejects_removed_research_models(self):
        with self.assertRaisesRegex(ResearchError, "unsupported model_names"):
            self.service.run(
                self.uow,
                training_motor_id="REAL-A",
                evaluation_motor_id=None,
                model_names=("xgboost",),
                include_ablations=False,
            )

    def test_forecasts_latest_pending_trajectory_without_future_truth(self):
        result = self.service.forecast_trajectory(
            self.uow,
            motor_id="REAL-B",
            training_motor_id="REAL-A",
            model_name="ridge_history_trend",
        )

        self.assertEqual(result["truth_status"], "pending")
        self.assertEqual(result["source"]["anchor_time"], self.datasets["REAL-B"][-1].recorded_at.isoformat())
        self.assertAlmostEqual(
            result["source"]["current_temperature_c"],
            self.datasets["REAL-B"][-1].temperature,
        )
        self.assertEqual(len(result["trajectory"]), 6)
        self.assertTrue(
            all(point["truth_status"] == "pending" for point in result["trajectory"])
        )
        historical = result["historical_evaluation"]
        self.assertEqual(historical["status"], "completed")
        self.assertEqual(
            historical["locked_test"]["truth_status"],
            "observed",
        )
        self.assertGreater(
            historical["locked_test"]["overall"]["sample_count"],
            0,
        )
        self.assertIsNotNone(
            historical["locked_test"]["overall"]["mae"]
        )
        self.assertFalse(historical["locked_test_used_for_selection"])
        self.assertIn(result["risk"]["risk_level"], {"low", "medium", "high"})
        self.assertTrue(
            result["leakage_audit"][
                "training_labels_do_not_follow_forecast_origin"
            ]
        )
        self.assertFalse(result["leakage_audit"]["future_truth_used_for_prediction"])
        self.assertIs(self.service.get_forecast(result["forecast_id"]), result)

    def test_single_horizon_backtest_keeps_a_visible_historical_series(self):
        result = self.service.forecast_trajectory(
            self.uow,
            motor_id="REAL-B",
            training_motor_id="REAL-A",
            model_name="ridge_direct",
            horizons_minutes=(30,),
        )

        chart_series = result["historical_evaluation"]["locked_test"][
            "chart_series"
        ]
        self.assertEqual(chart_series["horizon_minutes"], 30)
        self.assertGreater(chart_series["point_count"], 1)
        self.assertLessEqual(chart_series["point_count"], 60)
        self.assertEqual(
            chart_series["point_count"],
            len(chart_series["points"]),
        )

    def test_rejects_out_of_contract_safety_threshold(self):
        with self.assertRaisesRegex(ResearchError, "between 20 and 120"):
            self.service.forecast_trajectory(
                self.uow,
                motor_id="REAL-B",
                threshold_c=0,
            )


if __name__ == "__main__":
    unittest.main()
