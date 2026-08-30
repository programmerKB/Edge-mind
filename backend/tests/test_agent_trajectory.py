"""Agent routing tests for current multi-horizon temperature risk forecasts."""

import unittest

from edgemind.application.agent import AgentService
from edgemind.application.diagnostics import DiagnosticToolService


class FakeModel:
    async def decide(self, _message):
        raise AssertionError("deterministic trajectory intent should skip model routing")

    async def summarize(self, _message, tool_results):
        return f"grounded:{tool_results[0]['name']}"

    async def close(self):
        return None


class FakeTools:
    def __init__(self):
        self.calls = []

    def execute(self, name, arguments):
        self.calls.append((name, arguments))
        return {
            "motor_id": arguments["motor_id"],
            "model": {"name": arguments.get("model_name", "ridge_direct")},
            "source": {"current_temperature_c": 31.5},
            "trajectory": [{"horizon_minutes": 5, "predicted_temperature_c": 32.0}],
            "risk": {
                "risk_level": "low",
                "max_temperature_c": 32.0,
                "threshold_c": 35.0,
            },
            "historical_evaluation": {
                "status": "completed",
                "split": {"test": {"sample_count": 10}},
                "locked_test": {
                    "truth_status": "observed",
                    "overall": {"mae": 0.2, "rmse": 0.3},
                },
            },
            "attachments": [
                {
                    "title": "預測溫度軌跡",
                    "alt": "預測溫度軌跡圖表",
                    "url": "/api/report-artifacts/chart.svg",
                    "media_type": "image/svg+xml",
                }
            ],
        }


class FailingTools:
    def execute(self, _name, _arguments):
        return {"error": "感測資料未對齊 UTC 格點"}


class FakeUnitOfWork:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


class FakeResearchService:
    def __init__(self):
        self.arguments = None

    def forecast_trajectory(self, _uow, **arguments):
        self.arguments = arguments
        return {
            "forecast_id": "f" * 32,
            "motor_id": arguments["motor_id"],
            "truth_status": "pending",
            "trajectory": [],
        }


class AgentTrajectoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_routes_risk_prompt_to_trajectory_tool(self):
        tools = FakeTools()
        agent = AgentService(FakeModel(), tools)

        events = [
            event
            async for event in agent.stream(
                "請使用 DEMO-1 訓練的模型，預測 DEMO-2 未來溫度軌跡與過熱風險"
            )
        ]

        self.assertEqual(
            tools.calls,
            [
                (
                    "get_temperature_trajectory_forecast",
                    {"motor_id": "DEMO-2", "training_motor_id": "DEMO-1"},
                )
            ],
        )
        self.assertEqual(events[-1].status, "success")
        self.assertIn("完整預測流程已完成", events[-1].content)
        self.assertIn("MAE", events[-1].content)
        self.assertEqual(len(events[-1].attachments), 1)
        self.assertNotIn("artifacts", {event.status for event in events})

    async def test_diagnostic_tool_executes_research_forecast_use_case(self):
        research = FakeResearchService()
        tools = DiagnosticToolService(
            FakeUnitOfWork,
            forecasts=object(),
            research=research,
            sensors=object(),
        )

        payload = tools.execute(
            "get_temperature_trajectory_forecast",
            {
                "motor_id": "DEMO-2",
                "training_motor_id": "DEMO-1",
                "model_name": "ridge_history_trend",
                "threshold_c": 40.0,
                "horizons_minutes": list(range(5, 61, 5)),
            },
        )

        self.assertEqual(payload["truth_status"], "pending")
        self.assertGreaterEqual(payload["tool_duration_ms"], 0)
        self.assertEqual(
            research.arguments,
            {
                "motor_id": "DEMO-2",
                "training_motor_id": "DEMO-1",
                "model_name": "ridge_history_trend",
                "threshold_c": 40.0,
                "horizons_minutes": list(range(5, 61, 5)),
            },
        )

    async def test_tool_failure_is_returned_as_error_without_generated_summary(self):
        agent = AgentService(FakeModel(), FailingTools())

        events = [
            event
            async for event in agent.stream(
                "請預測馬達 M1 未來 5 到 30 分鐘的溫度軌跡"
            )
        ]

        self.assertEqual(events[-1].status, "error")
        self.assertEqual(events[-1].content, "感測資料未對齊 UTC 格點")
        self.assertNotIn("success", {event.status for event in events})

    async def test_agent_preserves_custom_risk_contract(self):
        tools = FakeTools()
        agent = AgentService(FakeModel(), tools)

        async for _event in agent.stream(
            "用 40°C 警戒值預測馬達 M1 未來 5 到 60 分鐘風險"
        ):
            pass

        self.assertEqual(
            tools.calls,
            [
                (
                    "get_temperature_trajectory_forecast",
                    {
                        "motor_id": "M1",
                        "threshold_c": 40.0,
                        "horizons_minutes": list(range(5, 61, 5)),
                    },
                )
            ],
        )

    async def test_ui_model_selection_overrides_model_named_in_prompt(self):
        tools = FakeTools()
        agent = AgentService(FakeModel(), tools)

        async for _event in agent.stream(
            "請用 TCN 預測馬達 M1 未來 5 到 30 分鐘的溫度軌跡",
            model_name="lstm",
        ):
            pass

        self.assertEqual(
            tools.calls[0],
            (
                "get_temperature_trajectory_forecast",
                {
                    "motor_id": "M1",
                    "model_name": "lstm",
                    "horizons_minutes": [5, 10, 15, 20, 25, 30],
                },
            ),
        )

    async def test_selected_model_routes_legacy_forecast_to_modelled_trajectory(self):
        tools = FakeTools()
        agent = AgentService(FakeModel(), tools)

        async for _event in agent.stream(
            "請使用 DEMO-1 訓練的模型，推論設備 DEMO-2 在 30 分鐘後的溫度",
            model_name="patchtst",
        ):
            pass

        self.assertEqual(
            tools.calls[0],
            (
                "get_temperature_trajectory_forecast",
                {
                    "motor_id": "DEMO-2",
                    "training_motor_id": "DEMO-1",
                    "model_name": "patchtst",
                    "horizons_minutes": [30],
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
