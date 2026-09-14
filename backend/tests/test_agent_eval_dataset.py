"""Structural regression tests for the versioned Agent benchmark."""

import unittest

from agent_eval.dataset import load_cases, validate_cases
from agent_eval.fixtures import tool_result


class AgentEvalDatasetTests(unittest.TestCase):
    """Keep published counts, split isolation, and fixtures reproducible."""

    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()

    def test_dataset_matches_published_design(self):
        stats = validate_cases(self.cases)

        self.assertEqual(stats["total_cases"], 180)
        self.assertEqual(
            stats["split_counts"],
            {"development": 60, "validation": 30, "locked_test": 90},
        )
        self.assertEqual(
            stats["forecast_model_counts"],
            {"ridge_direct": 40, "ridge_history": 40},
        )
        self.assertEqual(stats["template_family_leaks"], 0)

    def test_every_expected_call_has_a_controlled_result(self):
        for case in self.cases:
            with self.subTest(case_id=case["case_id"]):
                for expected_call in case["expected_tool_calls"]:
                    result = tool_result(
                        case["fixture_id"],
                        expected_call["name"],
                        expected_call["arguments"],
                    )
                    self.assertIsInstance(result, dict)
                    self.assertTrue(result)

    def test_no_case_uses_exact_answer_string_matching(self):
        for case in self.cases:
            with self.subTest(case_id=case["case_id"]):
                self.assertNotIn("expected_answer", case)
                self.assertIsInstance(case["required_answer_facts"], list)


if __name__ == "__main__":
    unittest.main()
