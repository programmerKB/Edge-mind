"""Tests for event-level warnings and formal paired statistics."""

from datetime import datetime, timedelta, timezone
import unittest

from edgemind.domain.formal_analysis import (
    FormalAnalysisError,
    TemperatureObservation,
    WarningSignal,
    evaluate_event_warnings,
    holm_correction,
    paired_block_bootstrap,
)


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def observations(
    temperatures: list[float], *, device_id: str = "MOTOR-A"
) -> list[TemperatureObservation]:
    return [
        TemperatureObservation(
            device_id=device_id,
            recorded_at=START + timedelta(minutes=5 * index),
            temperature_c=value,
        )
        for index, value in enumerate(temperatures)
    ]


def warning(index: int, *, device_id: str = "MOTOR-A") -> WarningSignal:
    return WarningSignal(
        device_id=device_id,
        issued_at=START + timedelta(minutes=5 * index),
    )


class EventWarningTests(unittest.TestCase):
    def test_merges_events_and_matches_only_one_warning_episode(self):
        values = [30.0] * 30
        # Two hot runs with one ten-minute below-threshold gap form one event.
        values[10:12] = [36.0, 37.0]
        values[14:16] = [36.5, 36.0]
        result = evaluate_event_warnings(
            observations(values),
            [warning(5), warning(6), warning(8)],
            threshold_c=35.0,
            event_merge_gap_minutes=10,
            warning_lookback_minutes=30,
            warning_cooldown_minutes=5,
        )

        self.assertEqual(result["events"]["detected_count"], 1)
        self.assertEqual(result["events"]["eligible_count"], 1)
        self.assertEqual(result["events"]["matched_count"], 1)
        # 5/6 are one episode; index 8 is a second episode.  One event can be
        # claimed once, so the second episode remains a false alarm.
        self.assertEqual(result["warnings"]["early_warning_episode_count"], 2)
        self.assertEqual(result["warnings"]["matched_episode_count"], 1)
        self.assertEqual(result["warnings"]["false_alarm_count"], 1)
        self.assertAlmostEqual(result["metrics"]["event_precision"], 0.5)
        self.assertAlmostEqual(result["metrics"]["event_recall"], 1.0)
        self.assertEqual(result["metrics"]["lead_time_minutes"]["values"], [25.0])
        # Thirty observed five-minute slots = 1/9.6 day.
        self.assertAlmostEqual(
            result["metrics"]["false_alarms_per_observed_day"], 9.6
        )

    def test_excludes_ongoing_and_boundary_censored_evidence(self):
        values = [36.0, 36.0] + [30.0] * 26 + [36.0, 36.0]
        result = evaluate_event_warnings(
            observations(values),
            [warning(0), warning(24), warning(27)],
            threshold_c=35.0,
            warning_lookback_minutes=30,
        )

        self.assertEqual(result["events"]["detected_count"], 2)
        self.assertEqual(result["events"]["eligible_count"], 0)
        self.assertEqual(result["events"]["boundary_censored_count"], 2)
        self.assertEqual(result["warnings"]["ongoing_origin_signal_count"], 1)
        # Both remaining warnings could precede the right-censored event or do
        # not have complete future observation, so neither becomes an FP.
        self.assertEqual(result["warnings"]["evaluated_episode_count"], 0)
        self.assertEqual(result["warnings"]["boundary_censored_episode_count"], 2)
        self.assertIsNone(result["metrics"]["event_precision"])
        self.assertIsNone(result["metrics"]["event_recall"])

    def test_rejects_warning_without_an_observed_origin(self):
        with self.assertRaisesRegex(FormalAnalysisError, "observed device/time"):
            evaluate_event_warnings(
                observations([30.0] * 12),
                [WarningSignal("MOTOR-A", START + timedelta(minutes=3))],
                threshold_c=35.0,
            )


class FormalStatisticsTests(unittest.TestCase):
    def test_paired_bootstrap_is_seeded_and_preserves_pairing(self):
        reference = {"day-1": 2.0, "day-2": 4.0, "day-3": 6.0}
        candidate = {"day-1": 1.0, "day-2": 3.0, "day-3": 5.0}

        first = paired_block_bootstrap(
            reference, candidate, n_resamples=500, seed=29
        )
        second = paired_block_bootstrap(
            dict(reversed(list(reference.items()))),
            dict(reversed(list(candidate.items()))),
            n_resamples=500,
            seed=29,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["status"], "available")
        self.assertEqual(first["candidate_minus_reference"], -1.0)
        self.assertAlmostEqual(
            first["candidate_minus_reference_ci"]["lower"], -1.0
        )
        self.assertAlmostEqual(
            first["candidate_minus_reference_ci"]["upper"], -1.0
        )
        self.assertGreater(first["relative_improvement_percent"], 0)

    def test_paired_bootstrap_reports_insufficient_blocks(self):
        result = paired_block_bootstrap(
            {"only-day": 2.0}, {"only-day": 1.5}, n_resamples=10
        )

        self.assertEqual(result["status"], "insufficient_blocks")
        self.assertIsNone(result["candidate_minus_reference_ci"])
        with self.assertRaisesRegex(FormalAnalysisError, "identical block IDs"):
            paired_block_bootstrap({"a": 1.0}, {"b": 1.0})

    def test_holm_correction_is_step_down_and_keeps_unavailable_values(self):
        result = holm_correction(
            {"m1": 0.01, "m2": 0.04, "m3": 0.03, "not-run": None}
        )

        self.assertEqual(result["status"], "available")
        self.assertAlmostEqual(result["results"]["m1"]["adjusted_p_value"], 0.03)
        self.assertTrue(result["results"]["m1"]["reject_null"])
        self.assertAlmostEqual(result["results"]["m3"]["adjusted_p_value"], 0.06)
        self.assertFalse(result["results"]["m2"]["reject_null"])
        self.assertEqual(result["results"]["not-run"]["status"], "unavailable")
        self.assertEqual(
            holm_correction({"not-run": None})["status"],
            "insufficient_comparisons",
        )


if __name__ == "__main__":
    unittest.main()
