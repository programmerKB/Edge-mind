"""Integration-style tests for generated CSV, SVG, and summary artifacts."""

from datetime import datetime, timedelta, timezone
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from edgemind.application.agent import AgentEvent, TokenUsage
from edgemind.domain.forecasting import (
    build_training_examples,
    predict_with_payload,
    train_temperature_model,
)
from edgemind.domain.demo_data import (
    build_demo_inference_readings,
    build_demo_readings,
)
from edgemind.infrastructure.reporting.artifacts import (
    build_chart_attachments,
    resolve_chart_artifact,
)
from edgemind.infrastructure.reporting.context import ReportContext
from edgemind.infrastructure.reporting.inference import create_inference_report
from edgemind.infrastructure.reporting.performance import (
    finalize_performance_report,
    record_gemini_token_usage,
)
from edgemind.presentation.sse import encode_sse_event


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

        with tempfile.TemporaryDirectory() as temporary:
            report_root = Path(temporary)
            context = ReportContext(
                root=report_root,
                timezone=timezone.utc,
                anomaly_temperature_threshold=35.0,
            )
            report = create_inference_report(
                context=context,
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
            self.assertEqual(
                report["completed_evaluation_samples"],
                len(inference) - 5,
            )
            self.assertTrue(Path(files["predictions_csv"]).is_file())
            self.assertTrue(Path(files["baseline_csv"]).is_file())
            self.assertTrue(Path(files["anomaly_csv"]).is_file())
            self.assertTrue(Path(files["system_csv"]).is_file())
            self.assertTrue(Path(files["summary_json"]).is_file())
            self.assertEqual(len(files["charts"]), 7)
            self.assertTrue(
                all(Path(chart).is_file() for chart in files["charts"])
            )
            attachments = build_chart_attachments(
                files["charts"],
                report_root,
            )
            self.assertEqual(len(attachments), 7)
            self.assertIn("04_baseline_mae.svg", attachments[0]["url"])
            self.assertIn("05_anomaly_f1.svg", attachments[1]["url"])
            relative_chart = attachments[0]["url"].removeprefix(
                "/api/report-artifacts/"
            )
            self.assertEqual(
                resolve_chart_artifact(relative_chart, report_root),
                Path(files["charts"][3]).resolve(),
            )
            self.assertIsNone(
                resolve_chart_artifact(
                    "../../metadata/run_summary.json",
                    report_root,
                )
            )
            event_payload = json.loads(
                encode_sse_event(
                    AgentEvent(
                        "success",
                        "模型評估完成",
                        attachments,
                        TokenUsage(
                            prompt_tokens=100,
                            output_tokens=25,
                            total_tokens=125,
                            model_calls=1,
                        ),
                    ),
                    False,
                ).removeprefix("data: ")
            )
            self.assertEqual(event_payload["status"], "success")
            self.assertEqual(len(event_payload["attachments"]), 7)
            self.assertEqual(event_payload["token_usage"]["prompt_tokens"], 100)
            self.assertEqual(event_payload["token_usage"]["output_tokens"], 25)
            self.assertEqual(event_payload["token_usage"]["total_tokens"], 125)
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
            self.assertIn("Gemini輸入Token", system_header)
            self.assertIn("Gemini總Token", system_header)

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

            with_usage = record_gemini_token_usage(
                files["system_csv"],
                {
                    "prompt_tokens": 500,
                    "output_tokens": 60,
                    "thought_tokens": 20,
                    "cached_tokens": 100,
                    "tool_prompt_tokens": 5,
                    "total_tokens": 585,
                    "model_calls": 2,
                },
            )
            with Path(files["system_csv"]).open(
                encoding="utf-8-sig",
                newline="",
            ) as stream:
                system_usage = next(csv.DictReader(stream))
            self.assertEqual(system_usage["Gemini輸入Token"], "500")
            self.assertEqual(system_usage["Gemini輸出Token"], "60")
            self.assertEqual(system_usage["Gemini總Token"], "585")
            self.assertEqual(with_usage["run_count"], 1)
            self.assertEqual(
                with_usage["metrics"]["gemini_total_tokens"]["mean"],
                585.0,
            )

            second_report = create_inference_report(
                context=context,
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


if __name__ == "__main__":
    unittest.main()
