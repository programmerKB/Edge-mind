from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import reporting
from forecasting import (
    build_training_examples,
    predict_with_payload,
    train_temperature_model,
)
from seed_data import build_demo_inference_readings, build_demo_readings


class ReportingTests(unittest.TestCase):
    def test_creates_categorized_csv_charts_and_summary(self):
        reference_time = datetime(2026, 8, 11, 8, 30, tzinfo=timezone.utc)
        training = [
            SimpleNamespace(**row)
            for row in build_demo_readings(reference_time)
        ]
        inference = [
            SimpleNamespace(**row)
            for row in build_demo_inference_readings(reference_time)
        ]
        payload = train_temperature_model(build_training_examples(training))
        latest_features = tuple(
            getattr(inference[-1], name)
            for name in payload["feature_names"]
        )
        latest_prediction = predict_with_payload(payload, latest_features)

        original_root = reporting.REPORT_ROOT
        try:
            with tempfile.TemporaryDirectory() as temporary:
                reporting.REPORT_ROOT = Path(temporary)
                report = reporting.create_inference_report(
                    training_records=training,
                    inference_records=inference,
                    model_payload=payload,
                    inference_motor_id="DEMO-2",
                    training_motor_id="DEMO-1",
                    latest_prediction=latest_prediction,
                    generated_at=reference_time,
                    model_inference_duration_ms=0.1,
                    process_cpu_time_ms=0.1,
                )

                files = report["files"]
                self.assertEqual(report["completed_evaluation_samples"], 31)
                self.assertTrue(Path(files["predictions_csv"]).is_file())
                self.assertTrue(Path(files["baseline_csv"]).is_file())
                self.assertTrue(Path(files["anomaly_csv"]).is_file())
                self.assertTrue(Path(files["system_csv"]).is_file())
                self.assertTrue(Path(files["summary_json"]).is_file())
                self.assertEqual(len(files["charts"]), 7)
                self.assertTrue(
                    all(Path(chart).is_file() for chart in files["charts"])
                )
                self.assertIn("r2_score", report["metrics"])
                self.assertIn(
                    "specificity",
                    report["anomaly_classification"]["Five-feature Ridge"],
                )
                self.assertIn(
                    "roc_auc",
                    report["anomaly_classification"]["Five-feature Ridge"],
                )

                aggregate = reporting.update_api_duration(
                    files["system_csv"],
                    12.5,
                )
                self.assertEqual(aggregate["run_count"], 1)
                self.assertTrue(Path(aggregate["files"]["history_csv"]).is_file())
                self.assertTrue(Path(aggregate["files"]["summary_csv"]).is_file())
                summary = json.loads(
                    Path(aggregate["files"]["summary_json"]).read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(
                    summary["metrics"]["api_handler_duration_ms"]["median"],
                    12.5,
                )

                # Finalizing the same run is idempotent and does not inflate N.
                repeated = reporting.update_api_duration(
                    files["system_csv"],
                    14.0,
                )
                self.assertEqual(repeated["run_count"], 1)

                second_report = reporting.create_inference_report(
                    training_records=training,
                    inference_records=inference,
                    model_payload=payload,
                    inference_motor_id="DEMO-2",
                    training_motor_id="DEMO-1",
                    latest_prediction=latest_prediction,
                    generated_at=reference_time + timedelta(seconds=1),
                    model_inference_duration_ms=0.2,
                    process_cpu_time_ms=0.2,
                )
                combined = reporting.update_api_duration(
                    second_report["files"]["system_csv"],
                    26.0,
                )
                self.assertEqual(combined["run_count"], 2)
                self.assertEqual(
                    combined["metrics"]["api_handler_duration_ms"]["median"],
                    20.0,
                )
        finally:
            reporting.REPORT_ROOT = original_root


if __name__ == "__main__":
    unittest.main()
