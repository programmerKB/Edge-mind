"""Tests that the chat UI's Ridge selection controls tool execution."""

from edgemind.application.agent import (
    AgentService,
    ModelReply,
    TokenUsage,
    ToolCall,
)
from edgemind.domain.ridge_experiments import DIRECT_MODEL, HISTORY_MODEL
import unittest


class _FakeModel:
    def __init__(
        self,
        reply: ModelReply | None = None,
        summary_usage: TokenUsage | None = None,
    ):
        self.reply = reply or ModelReply(text="不需要工具")
        self.summary_usage = summary_usage
        self.decide_calls = 0
        self.tool_results = None

    async def decide(self, message: str) -> ModelReply:
        self.decide_calls += 1
        return self.reply

    async def summarize(self, message: str, tool_results: list[dict]) -> ModelReply:
        self.tool_results = tool_results
        return ModelReply(
            text="已根據指定模型完成診斷。",
            token_usage=self.summary_usage,
        )

    async def close(self) -> None:
        return None


class _CapturingTools:
    def __init__(self):
        self.calls = []

    def execute(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        return {
            "motor_id": arguments["motor_id"],
            "model_name": arguments["model_name"],
            "model_label": (
                "Ridge + History"
                if arguments["model_name"] == HISTORY_MODEL
                else "Direct Ridge"
            ),
        }


class _CapturingUsageGateway:
    def __init__(self):
        self.records = []

    def record_token_usage(self, tool_results, usage):
        self.records.append((tool_results, usage))


class _UnavailableSummaryModel(_FakeModel):
    async def summarize(self, message: str, tool_results: list[dict]) -> ModelReply:
        self.tool_results = tool_results
        raise RuntimeError("模型服務目前忙碌")


class _ForecastResultTools:
    def __init__(self, *, error: str | None = None):
        self.error = error
        self.calls = []

    def execute(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        if self.error:
            return {"error": self.error}
        return {
            "motor_id": arguments["motor_id"],
            "training_motor_id": arguments.get(
                "training_motor_id", arguments["motor_id"]
            ),
            "model_name": arguments["model_name"],
            "model_label": "Direct Ridge",
            "current_temperature": 31.2,
            "predicted_temperature": 33.6,
            "predicted_change": 2.4,
            "forecast_horizon_minutes": 30,
            "evaluation": {"mae": 0.72, "rmse": 0.94},
        }


class _StatusResultTools:
    def __init__(self):
        self.calls = []

    def execute(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        return {
            "motor_id": arguments["motor_id"],
            "temperature": 31.2,
            "humidity": 55.4,
            "accel_x": 0.12,
            "accel_y": 0.08,
            "accel_z": 0.98,
            "vibration": 0.99,
            "status": "normal",
        }


class AgentModelSelectionTests(unittest.IsolatedAsyncioTestCase):
    """Keep the user-selected inference method authoritative."""

    async def test_selection_is_added_to_deterministic_forecast(self):
        model = _FakeModel()
        tools = _CapturingTools()
        agent = AgentService(model, tools)

        events = [
            event
            async for event in agent.stream(
                "請用 DEMO-1 模型預測 DEMO-2 在 30 分鐘後的溫度",
                HISTORY_MODEL,
            )
        ]

        self.assertEqual(model.decide_calls, 0)
        self.assertEqual(
            tools.calls,
            [
                (
                    "get_temperature_forecast",
                    {
                        "motor_id": "DEMO-2",
                        "training_motor_id": "DEMO-1",
                        "model_name": HISTORY_MODEL,
                    },
                )
            ],
        )
        self.assertTrue(any("Ridge + History" in event.content for event in events))

    async def test_selection_overrides_model_generated_argument(self):
        model = _FakeModel(
            ModelReply(
                tool_calls=(
                    ToolCall(
                        "get_temperature_forecast",
                        {"motor_id": "DEMO-2", "model_name": HISTORY_MODEL},
                    ),
                )
            )
        )
        tools = _CapturingTools()
        agent = AgentService(model, tools)

        async for _ in agent.stream("分析 DEMO-2 的未來風險", DIRECT_MODEL):
            pass

        self.assertEqual(model.decide_calls, 1)
        self.assertEqual(tools.calls[0][1]["model_name"], DIRECT_MODEL)
        self.assertEqual(
            model.tool_results[0]["result"]["model_name"],
            DIRECT_MODEL,
        )

    async def test_success_event_accumulates_all_gemini_token_usage(self):
        model = _FakeModel(
            ModelReply(
                tool_calls=(
                    ToolCall(
                        "get_temperature_forecast",
                        {"motor_id": "DEMO-2"},
                    ),
                ),
                token_usage=TokenUsage(
                    prompt_tokens=80,
                    output_tokens=10,
                    total_tokens=90,
                    model_calls=1,
                ),
            ),
            summary_usage=TokenUsage(
                prompt_tokens=320,
                output_tokens=45,
                thought_tokens=20,
                total_tokens=385,
                model_calls=1,
            ),
        )
        usage_gateway = _CapturingUsageGateway()
        agent = AgentService(model, _CapturingTools(), usage_gateway)

        events = [
            event
            async for event in agent.stream(
                "分析 DEMO-2 的未來風險",
                DIRECT_MODEL,
            )
        ]

        usage = events[-1].token_usage
        self.assertIsNotNone(usage)
        self.assertEqual(usage.prompt_tokens, 400)
        self.assertEqual(usage.output_tokens, 55)
        self.assertEqual(usage.thought_tokens, 20)
        self.assertEqual(usage.total_tokens, 475)
        self.assertEqual(usage.model_calls, 2)
        self.assertEqual(len(usage_gateway.records), 1)
        self.assertIs(usage_gateway.records[0][1], usage)

    async def test_tool_result_survives_unavailable_model_summary(self):
        agent = AgentService(
            _UnavailableSummaryModel(),
            _ForecastResultTools(),
        )

        events = [
            event
            async for event in agent.stream(
                "請預測馬達 M1 在 30 分鐘後的溫度",
                DIRECT_MODEL,
            )
        ]

        self.assertEqual(events[-1].status, "success")
        self.assertIn("模型摘要服務暫時無法使用", events[-1].content)
        self.assertIn("33.6°C", events[-1].content)
        self.assertIn("Direct Ridge", events[-1].content)
        self.assertIn("訓練設備：M1", events[-1].content)
        self.assertFalse(any(event.status == "error" for event in events))

    async def test_tool_error_survives_unavailable_model_summary(self):
        agent = AgentService(
            _UnavailableSummaryModel(),
            _ForecastResultTools(error="設備資料不足"),
        )

        events = [
            event
            async for event in agent.stream(
                "請預測馬達 M1 在 30 分鐘後的溫度",
                HISTORY_MODEL,
            )
        ]

        self.assertEqual(events[-1].status, "success")
        self.assertIn("設備資料不足", events[-1].content)

    async def test_explicit_status_survives_model_outage(self):
        model = _UnavailableSummaryModel()
        tools = _StatusResultTools()
        agent = AgentService(model, tools)

        events = [
            event
            async for event in agent.stream("請查詢馬達 M1 現在的溫度與狀態")
        ]

        self.assertEqual(model.decide_calls, 0)
        self.assertEqual(tools.calls, [("get_motor_status", {"motor_id": "M1"})])
        self.assertEqual(events[-1].status, "success")
        self.assertIn("溫度：31.2°C", events[-1].content)
        self.assertIn("工具判定狀態：normal", events[-1].content)


if __name__ == "__main__":
    unittest.main()
