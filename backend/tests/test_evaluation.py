"""Unit tests for regression, classification, and descriptive metrics."""

import unittest

from edgemind.domain.evaluation import (
    binary_score_metrics,
    classification_metrics,
    descriptive_statistics,
    regression_metrics,
)


class EvaluationTests(unittest.TestCase):
    """Verify metric values and invalid-input behavior."""

    def test_descriptive_statistics_include_tail_percentiles(self):
        metrics = descriptive_statistics([1, 2, 3, 4, 100])

        self.assertEqual(metrics["count"], 5)
        self.assertEqual(metrics["median"], 3)
        self.assertAlmostEqual(metrics["mean"], 22)
        self.assertGreater(metrics["p95"], metrics["median"])
        self.assertEqual(metrics["max"], 100)

    def test_regression_metrics_cover_error_and_goodness_of_fit(self):
        metrics = regression_metrics([2, 4, 6], [1, 4, 7])

        self.assertEqual(metrics["sample_count"], 3)
        self.assertAlmostEqual(metrics["mae"], 2 / 3)
        self.assertAlmostEqual(metrics["median_absolute_error"], 1)
        self.assertIsNotNone(metrics["r2_score"])

    def test_classification_metrics_include_confusion_matrix(self):
        metrics = classification_metrics(
            [True, True, False, False],
            [True, False, True, False],
            lead_minutes=30,
        )

        self.assertEqual(
            (metrics["TP"], metrics["FP"], metrics["TN"], metrics["FN"]),
            (1, 1, 1, 1),
        )
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["specificity"], 0.5)
        self.assertEqual(metrics["f1_score"], 0.5)
        self.assertEqual(metrics["lead_time_minutes"], 30)

    def test_binary_score_metrics_reward_perfect_ranking(self):
        metrics = binary_score_metrics(
            [0.9, 0.8, 0.2, 0.1],
            [True, True, False, False],
        )

        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertAlmostEqual(metrics["pr_auc"], 1.0)
        self.assertAlmostEqual(metrics["average_precision"], 1.0)

    def test_metrics_reject_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            regression_metrics([1], [1, 2])


if __name__ == "__main__":
    unittest.main()
