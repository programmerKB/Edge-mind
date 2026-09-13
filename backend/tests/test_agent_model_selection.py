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


if __name__ == "__main__":
    unittest.main()
