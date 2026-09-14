"""Tests for separating service capacity from Agent behavior."""

import unittest

from agent_eval.audit import build_service_audit
from agent_eval.dataset import load_cases
from agent_eval.fixtures import tool_result


class AgentEvalAuditTests(unittest.TestCase):
    def test_busy_model_runs_are_reported_but_not_conditionally_scored(self):
        case = next(case for case in load_cases() if case["case_id"] == "S-001")
        expected_call = case["expected_tool_calls"][0]
        good = {
            "case_id": case["case_id"],
            "run_index": 1,
            "actual_route": "model",
            "actual_tool_calls": [expected_call],
            "tool_results": [
                tool_result(case["fixture_id"], expected_call["name"], expected_call["arguments"])
            ],
            "final_answer": "M1 溫度 31.2°C，目前正常。",
            "runtime_error": None,
        }
        busy = {
            "case_id": case["case_id"],
            "run_index": 2,
            "actual_route": "model",
            "actual_tool_calls": [],
            "tool_results": [],
            "final_answer": "Agent 執行失敗：模型服務目前忙碌，請稍後再試。",
            "runtime_error": "Agent 執行失敗：模型服務目前忙碌，請稍後再試。",
        }

        audit = build_service_audit([case], [good, busy])

        self.assertEqual(audit["model_service_failures"], 1)
        self.assertEqual(audit["eligible_agent_runs"], 1)
        self.assertEqual(
            audit["conditional_agent_metrics"]["task_success_rate"], 1.0
        )
        self.assertNotIn("reliability", audit["conditional_agent_metrics"])


if __name__ == "__main__":
    unittest.main()
