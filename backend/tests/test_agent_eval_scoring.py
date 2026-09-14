"""Unit tests for Agent trajectory and answer-grounding metrics."""

import unittest

from agent_eval.dataset import load_cases
from agent_eval.fixtures import tool_result
from agent_eval.scoring import aggregate_scores, score_run


class AgentEvalScoringTests(unittest.TestCase):
    """Score semantics rather than byte-identical natural-language answers."""

    @classmethod
    def setUpClass(cls):
        cls.case = next(case for case in load_cases() if case["case_id"] == "S-001")
        cls.expected_call = cls.case["expected_tool_calls"][0]
        cls.result = tool_result(
            cls.case["fixture_id"],
            cls.expected_call["name"],
            cls.expected_call["arguments"],
        )

    def trace(self, answer: str, run_index: int = 1) -> dict:
        return {
            "case_id": self.case["case_id"],
            "run_index": run_index,
            "actual_route": "model",
            "actual_tool_calls": [self.expected_call],
            "tool_results": [self.result],
            "final_answer": answer,
            "latency_ms": 12.0,
            "token_usage": {"total_tokens": 20},
        }

    def test_complete_grounded_paraphrase_passes(self):
        scored = score_run(self.case, self.trace("M1 目前為正常，溫度 31.2°C。"))

        self.assertTrue(scored["task_success"])
        self.assertEqual(scored["key_fact_recall"], 1.0)
        self.assertEqual(scored["grounded_claim_precision"], 1.0)

    def test_unseen_numeric_claim_is_a_hallucination(self):
        scored = score_run(
            self.case,
            self.trace("M1 目前為正常，溫度 31.2°C；明天一定會升到 88.8°C。"),
        )

        self.assertFalse(scored["task_success"])
        self.assertTrue(scored["hallucination"])
        self.assertEqual(scored["ungrounded_numbers"], [88.8])

    def test_five_repeated_passes_report_pass_five(self):
        traces = [
            self.trace("M1 目前為正常，溫度 31.2°C。", run_index)
            for run_index in range(1, 6)
        ]

        report = aggregate_scores([self.case], traces)

        self.assertEqual(report["need_tool"]["f1"], 1.0)
        self.assertEqual(report["argument_field_accuracy"]["motor_id"], 1.0)
        self.assertEqual(report["reliability"]["pass^5"], 1.0)
        self.assertEqual(report["task_success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
