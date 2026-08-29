"""Tests for the reproducible, explicitly synthetic research data generator."""

from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_research_dataset.py"
SPEC = importlib.util.spec_from_file_location("generate_research_dataset", SCRIPT)
assert SPEC and SPEC.loader
GENERATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GENERATOR
SPEC.loader.exec_module(GENERATOR)


class ResearchDatasetGeneratorTests(unittest.TestCase):
    """Keep the smoke dataset aligned with the study's sequence contract."""

    def test_generates_reproducible_regular_multi_regime_device_histories(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        profile = GENERATOR.PROFILES[0]
        first = GENERATOR.generate_device_rows(
            profile,
            start=start,
            days=2,
            interval_minutes=5,
            seed=42,
        )
        second = GENERATOR.generate_device_rows(
            profile,
            start=start,
            days=2,
            interval_minutes=5,
            seed=42,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2 * 288)
        self.assertEqual(first[0]["recorded_at"], start.isoformat())
        self.assertEqual(first[1]["recorded_at"], "2026-01-01T00:05:00+00:00")
        self.assertTrue(any(row["temperature"] >= 35 for row in first))
        self.assertTrue(all(row["is_synthetic"] for row in first))

    def test_manifest_prevents_synthetic_results_becoming_research_claims(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = GENERATOR.generate_dataset(
                Path(temporary),
                days=2,
                interval_minutes=5,
                seed=7,
            )

            self.assertFalse(manifest["research_claims_allowed"])
            self.assertEqual(
                manifest["sequence_contract"]["horizons_minutes"],
                [5, 10, 15, 20, 25, 30],
            )
            self.assertEqual(manifest["sequence_contract"]["gap_steps"], 6)
            expected_devices = {profile.motor_id for profile in GENERATOR.PROFILES}
            self.assertEqual(set(manifest["row_counts"]), expected_devices)
            self.assertEqual(set(manifest["file_sha256"]), expected_devices)
            self.assertTrue(
                all(len(value) == 64 for value in manifest["file_sha256"].values())
            )
            self.assertEqual(len(manifest["dataset_content_sha256"]), 64)
            self.assertTrue(
                all(not Path(value).is_absolute() for value in manifest["files"].values())
            )
            self.assertTrue(Path(manifest["manifest"]).is_file())

    def test_release_hash_is_reproducible_and_overwrite_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = GENERATOR.generate_dataset(root / "first", days=2, seed=11)
            second = GENERATOR.generate_dataset(root / "second", days=2, seed=11)

            self.assertEqual(
                first["dataset_content_sha256"],
                second["dataset_content_sha256"],
            )
            with self.assertRaises(FileExistsError):
                GENERATOR.generate_dataset(root / "first", days=2, seed=11)
            replaced = GENERATOR.generate_dataset(
                root / "first",
                days=2,
                seed=11,
                overwrite=True,
            )
            self.assertEqual(
                first["dataset_content_sha256"],
                replaced["dataset_content_sha256"],
            )

    def test_rejects_cadence_outside_the_five_minute_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "interval_minutes=5"):
                GENERATOR.generate_dataset(
                    Path(temporary),
                    days=2,
                    interval_minutes=10,
                )

    def test_rejects_ambiguous_or_off_grid_release_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "timezone"):
                GENERATOR.generate_dataset(
                    root / "naive",
                    days=2,
                    start=datetime(2026, 1, 1),
                )
            with self.assertRaisesRegex(ValueError, "UTC 5-minute grid"):
                GENERATOR.generate_dataset(
                    root / "off-grid",
                    days=2,
                    start=datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc),
                )
            with self.assertRaisesRegex(ValueError, "timezone"):
                GENERATOR._parse_start("2026-01-01T00:00:00")


if __name__ == "__main__":
    unittest.main()
