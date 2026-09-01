"""Tests that the chat UI's Ridge selection controls tool execution."""

from edgemind.application.agent import AgentService, ModelReply, ToolCall
from edgemind.domain.ridge_experiments import DIRECT_MODEL, HISTORY_MODEL
import unittest


class _FakeModel:
    def __init__(self, reply: ModelReply | None = None):
        self.reply = reply or ModelReply(text="不需要工具")
        self.decide_calls = 0
        self.tool_results = None

    async def decide(self, message: str) -> ModelReply:
        self.decide_calls += 1
        return self.reply

    async def summarize(self, message: str, tool_results: list[dict]) -> str:
        self.tool_results = tool_results
        return "已根據指定模型完成診斷。"

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


if __name__ == "__main__":
    unittest.main()
