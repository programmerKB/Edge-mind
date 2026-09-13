"""Exercise Ridge forecast files, public images, and chat events together."""

from contextlib import contextmanager
import csv
from datetime import timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from xml.etree import ElementTree

from fastapi import FastAPI
from fastapi.testclient import TestClient

from edgemind.application.agent import AgentService, ModelReply, TokenUsage
from edgemind.application.diagnostics import DiagnosticToolService
from edgemind.application.forecasts import ForecastService
from edgemind.application.ridge_forecasts import RidgeForecastService
from edgemind.application.sensors import SensorService
from edgemind.domain.ridge_experiments import DIRECT_MODEL, HISTORY_MODEL
from edgemind.infrastructure.reporting.context import ReportContext
from edgemind.infrastructure.reporting.gateway import FilesystemReportGateway
from edgemind.presentation.api.router import create_api_router
from test_ridge_experiments import sensor_history


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class _SummaryModel:
    def __init__(self):
        self.error = False

    async def decide(self, message):
        raise AssertionError("Forecast intent should route directly to the tool")

    async def summarize(self, message, tool_results):
        if self.error:
            raise RuntimeError("summary unavailable")
        return ModelReply(
            text="已完成溫度預測與報表輸出。",
            token_usage=TokenUsage(
                prompt_tokens=300,
                output_tokens=40,
                thought_tokens=10,
                total_tokens=350,
                model_calls=1,
            ),
        )


class RidgeForecastReportingTests(unittest.TestCase):
    """Verify real reporting behind HTTP with isolated sensor and file storage."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.records = {"DEMO-1": sensor_history(180), "DEMO-2": sensor_history(120)}
        self.model_payloads = {}

        @contextmanager
        def uow_factory():
            yield SimpleNamespace(
                sensors=SimpleNamespace(list_readings=lambda motor: self.records.get(motor, [])),
                forecast_models=SimpleNamespace(
                    get_payload=self.model_payloads.get,
                    save_payload=self.model_payloads.__setitem__,
                ),
                commit=lambda: None,
            )

        reports = FilesystemReportGateway(ReportContext(
            root=self.root, timezone=timezone.utc, anomaly_temperature_threshold=33.0,
        ))
        self.uow_factory = uow_factory
        self.ridge_forecasts = RidgeForecastService(reports)
        forecasts = ForecastService(reports)
        sensors = SensorService()
        tools = DiagnosticToolService(uow_factory, forecasts, self.ridge_forecasts, sensors)
        self.model = _SummaryModel()
        app = FastAPI()
        app.include_router(create_api_router(SimpleNamespace(
            uow_factory=uow_factory,
            forecasts=forecasts,
            sensors=sensors,
            reports=reports,
            agent=AgentService(self.model, tools, reports),
        )))
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def forecast(self, model_name=HISTORY_MODEL):
        with self.uow_factory() as uow:
            return self.ridge_forecasts.forecast(
                uow,
                motor_id="DEMO-2",
                training_motor_id="DEMO-1",
                model_name=model_name,
            )

    def test_existing_rest_training_and_forecast_still_publish_charts(self):
        training = self.client.post("/api/predictions/train/DEMO-1")
        self.assertEqual(training.status_code, 200, training.text)
        response = self.client.get(
            "/api/predictions/temperature/DEMO-2",
            params={"training_motor_id": "DEMO-1", "auto_train": "false"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["motor_id"], "DEMO-2")
        self.assertEqual(result["training_motor_id"], "DEMO-1")
        self.assert_public_charts(result["attachments"])

    def assert_public_charts(self, attachments):
        self.assertEqual(len(attachments), 7)
        self.assertEqual(len({item["url"] for item in attachments}), 7)
        for attachment in attachments:
            response = self.client.get(attachment["url"])
            self.assertEqual(response.status_code, 200)
            self.assertIn("image/svg+xml", response.headers["content-type"])
            self.assertEqual(ElementTree.fromstring(response.text).tag, "{http://www.w3.org/2000/svg}svg")
            self.assertNotIn(str(self.root), attachment["url"])

    def test_both_ridge_variants_write_real_predictions_and_accessible_charts(self):
        errors = {}
        for model_name, feature_count in ((DIRECT_MODEL, 5), (HISTORY_MODEL, 35)):
            with self.subTest(model=model_name):
                result = self.forecast(model_name)
                files = result["artifacts"]
                run_directory = Path(files["run_directory"])
                self.assertTrue(run_directory.is_relative_to(self.root / "inference_runs"))
                self.assertEqual(len(list((run_directory / "csv").glob("*.csv"))), 5)
                self.assertEqual(len(list((run_directory / "charts").glob("*.svg"))), 7)
                self.assertEqual((self.root / "latest_run.txt").read_text(), str(run_directory))
                self.assert_public_charts(result["attachments"])

                rows = read_csv(files["predictions_csv"])
                self.assertEqual(len(rows), 104)  # 120 - 11 history - 6 horizon + 1 pending
                self.assertEqual({row["模型"] for row in rows}, {result["model_label"]})
                pending = rows[-1]
                self.assertEqual(pending["資料狀態"], "即時推論_等待真值")
                self.assertEqual(pending["30分鐘後實際溫度_C"], "")
                self.assertEqual(pending["絕對誤差_C"], "")
                self.assertEqual(pending["來源資料時間"], result["source_recorded_at"])
                self.assertEqual(pending["預測目標時間"], result["target_time"])
                self.assertAlmostEqual(float(pending["預測溫度_C"]), result["predicted_temperature"], delta=0.000501)

                # Recompute error from exported values: the pending forecast must
                # not enter metrics, and History must not reuse Direct predictions.
                errors[model_name] = sum(
                    abs(float(row["預測溫度_C"]) - float(row["30分鐘後實際溫度_C"]))
                    for row in rows[:-1]
                ) / 103
                self.assertAlmostEqual(result["evaluation"]["mae"], errors[model_name], places=5)
                summary = json.loads(Path(files["summary_json"]).read_text())
                self.assertEqual(summary["model_name"], model_name)
                self.assertEqual(summary["model"]["feature_count"], feature_count)
                self.assertEqual(summary["model"]["test_metrics"], result["test_metrics"])
                self.assertEqual(summary["model"]["validation_metrics"], result["validation_metrics"])
                self.assertEqual(summary["evaluation_scope"], "historical_backtest")
                self.assertIn(result["model_label"], summary["regression_baselines"])
                self.assertNotIn("Five-feature Ridge", summary["regression_baselines"])
                self.assertIn(result["model_label"], Path(files["charts"][5]).read_text())
                self.assertGreater(summary["model"]["training_duration_ms"], 0)
                self.assertIn(result["model_label"], {row["方法"] for row in read_csv(files["baseline_csv"])})
                self.assertIn(result["model_label"], {row["方法"] for row in read_csv(files["anomaly_csv"])})

        self.assertLess(errors[HISTORY_MODEL], errors[DIRECT_MODEL] * 0.5)
        self.assertEqual(result["performance_summary"]["run_count"], 2)

    def test_forecast_without_completed_truth_still_outputs_report(self):
        self.records["DEMO-2"] = sensor_history(12)
        result = self.forecast()
        self.assertEqual(result["evaluation"]["completed_samples"], 0)
        self.assertIsNone(result["evaluation"]["mae"])
        rows = read_csv(result["artifacts"]["predictions_csv"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["30分鐘後實際溫度_C"], "")
        self.assert_public_charts(result["attachments"])
        self.assertIn("No completed observations yet", Path(result["artifacts"]["charts"][0]).read_text())

    def test_report_uses_actual_prediction_origin_when_newest_reading_is_incomplete(self):
        self.records["DEMO-2"][-1].temperature = None
        result = self.forecast()
        expected = self.records["DEMO-2"][-2].recorded_at.isoformat()
        self.assertEqual(result["source_recorded_at"], expected)
        rows = read_csv(result["artifacts"]["predictions_csv"])
        self.assertEqual(rows[-1]["來源資料時間"], expected)

    def test_chat_sends_images_for_both_models_even_if_summary_fails(self):
        for model_name, summary_error in ((DIRECT_MODEL, False), (HISTORY_MODEL, False), (HISTORY_MODEL, True)):
            with self.subTest(model=model_name, summary_error=summary_error):
                self.model.error = summary_error
                response = self.client.post("/api/chat_utf8", json={
                    "message": "請用 DEMO-1 模型預測 DEMO-2 在 30 分鐘後的溫度",
                    "inference_model": model_name,
                })
                self.assertEqual(response.status_code, 200)
                events = [json.loads(line.removeprefix("data: ")) for line in response.text.splitlines() if line.startswith("data: ")]
                artifact_events = [event for event in events if event["status"] == "artifacts"]
                self.assertEqual(len(artifact_events), 1, events)
                self.assert_public_charts(artifact_events[0]["attachments"])
                self.assertEqual(events[-1]["status"], "error" if summary_error else "success")
                system_csv = Path(
                    (self.root / "latest_run.txt").read_text()
                ) / "csv" / "system_performance.csv"
                system_row = read_csv(system_csv)[0]
                if summary_error:
                    self.assertEqual(system_row["Gemini總Token"], "")
                else:
                    self.assertEqual(system_row["Gemini輸入Token"], "300")
                    self.assertEqual(system_row["Gemini輸出Token"], "40")
                    self.assertEqual(system_row["Gemini思考Token"], "10")
                    self.assertEqual(system_row["Gemini總Token"], "350")
                if model_name == HISTORY_MODEL:
                    self.assertIn("不等同鎖定測試集成績", artifact_events[0]["content"])


if __name__ == "__main__":
    unittest.main()
