"""Integration-style tests for generated CSV, SVG, and summary artifacts."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from reports import config as report_config
from reports.artifacts import build_chart_attachments, resolve_chart_artifact
from reports.inference import create_inference_report
from reports.performance import finalize_performance_report
from services.sse import sse_event
from forecast import (
    build_training_examples,
    predict_with_payload,
    train_temperature_model,
)
from seed_data import build_demo_inference_readings, build_demo_readings


class ReportingTests(unittest.TestCase):
    """Verify report contents, artifact safety, and aggregate idempotence."""

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

        original_root = report_config.REPORT_ROOT
        try:
            with tempfile.TemporaryDirectory() as temporary:
                report_config.REPORT_ROOT = Path(temporary)
                report = create_inference_report(
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
                attachments = build_chart_attachments(files["charts"])
                self.assertEqual(len(attachments), 7)
                self.assertIn("04_baseline_mae.svg", attachments[0]["url"])
                self.assertIn("05_anomaly_f1.svg", attachments[1]["url"])
                relative_chart = attachments[0]["url"].removeprefix(
                    "/api/report-artifacts/"
                )
                self.assertEqual(
                    resolve_chart_artifact(relative_chart),
                    Path(files["charts"][3]).resolve(),
                )
                self.assertIsNone(
                    resolve_chart_artifact("../../metadata/run_summary.json")
                )
                event_payload = json.loads(
                    sse_event(
                        "success",
                        "模型評估完成",
                        False,
                        attachments=attachments,
                    ).removeprefix("data: ")
                )
                self.assertEqual(event_payload["status"], "success")
                self.assertEqual(len(event_payload["attachments"]), 7)
                self.assertIn("r2_score", report["metrics"])
                self.assertIn(
                    "specificity",
                    report["anomaly_classification"]["Five-feature Ridge"],
                )
                self.assertIn(
                    "roc_auc",
                    report["anomaly_classification"]["Five-feature Ridge"],
                )

                system_csv = Path(files["system_csv"])
                system_header = system_csv.read_text(
                    encoding="utf-8-sig"
                ).splitlines()[0]
                self.assertNotIn("報表產生時間_ms", system_header)
                self.assertNotIn("API處理時間_ms", system_header)

                system_chart = Path(files["charts"][-1]).read_text(
                    encoding="utf-8"
                )
                self.assertIn("Training", system_chart)
                self.assertIn("Inference", system_chart)
                self.assertNotIn("Report", system_chart)
                self.assertNotIn("API", system_chart)

                aggregate = finalize_performance_report(
                    files["system_csv"]
                )
                self.assertEqual(aggregate["run_count"], 1)
                self.assertTrue(Path(aggregate["files"]["history_csv"]).is_file())
                self.assertTrue(Path(aggregate["files"]["summary_csv"]).is_file())
                summary = json.loads(
                    Path(aggregate["files"]["summary_json"]).read_text(
                        encoding="utf-8"
                    )
                )
                self.assertIn("model_training_duration_ms", summary["metrics"])
                self.assertIn("model_inference_duration_ms", summary["metrics"])
                self.assertNotIn("report_generation_duration_ms", summary["metrics"])
                self.assertNotIn("api_handler_duration_ms", summary["metrics"])

                # Finalizing the same run is idempotent and does not inflate N.
                repeated = finalize_performance_report(
                    files["system_csv"]
                )
                self.assertEqual(repeated["run_count"], 1)

                second_report = create_inference_report(
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
                combined = finalize_performance_report(
                    second_report["files"]["system_csv"]
                )
                self.assertEqual(combined["run_count"], 2)
                self.assertAlmostEqual(
                    combined["metrics"]["model_inference_duration_ms"]["median"],
                    0.15,
                )
        finally:
            report_config.REPORT_ROOT = original_root


if __name__ == "__main__":
    unittest.main()
