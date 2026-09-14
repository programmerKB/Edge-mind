"""Trajectory-capture tests for the Agent benchmark runner."""

import unittest

from agent_eval.dataset import load_cases
from agent_eval.runner import run_case
from edgemind.application.agent import ModelReply


class _FakeModel:
    def __init__(self):
        self.decide_calls = 0

    async def decide(self, message: str) -> ModelReply:
        self.decide_calls += 1
        return ModelReply(text="不需要工具")

    async def summarize(self, message: str, tool_results: list[dict]) -> ModelReply:
        result = tool_results[0]["result"]
        return ModelReply(
            text=(
                f"{result['motor_id']} 的 {result['model_label']} 預測為 "
                f"{result['predicted_temperature']}°C。"
            )
        )

    async def close(self) -> None:
        return None


class AgentEvalRunnerTests(unittest.IsolatedAsyncioTestCase):
    """Capture route, exact calls, observations, and final answer."""

    async def test_deterministic_case_records_complete_trajectory(self):
        case = next(
            case
            for case in load_cases()
            if case["case_id"] == "F-SAME-001"
        )
        model = _FakeModel()

        trace = await run_case(model, case, 1)

        self.assertEqual(trace["actual_route"], "deterministic")
        self.assertEqual(trace["actual_tool_calls"], case["expected_tool_calls"])
        self.assertEqual(trace["event_statuses"][-1], "success")
        self.assertEqual(model.decide_calls, 0)


if __name__ == "__main__":
    unittest.main()
