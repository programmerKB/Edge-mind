"""Validation tests for research CSV ingestion before any database mutation."""

import csv
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "import_research_dataset.py"
SPEC = importlib.util.spec_from_file_location("import_research_dataset", SCRIPT)
assert SPEC and SPEC.loader
IMPORTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = IMPORTER
SPEC.loader.exec_module(IMPORTER)


FIELDS = (
    "motor_id", "recorded_at", "temperature", "humidity",
    "accel_x", "accel_y", "accel_z", "is_synthetic",
)


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def row(timestamp: str, *, synthetic: str = "false") -> dict:
    return {
        "motor_id": "RESEARCH-A",
        "recorded_at": timestamp,
        "temperature": "31.2",
        "humidity": "58.0",
        "accel_x": "0.1",
        "accel_y": "0.2",
        "accel_z": "1.0",
        "is_synthetic": synthetic,
    }


class ResearchDatasetImporterTests(unittest.TestCase):
    def test_normalizes_timezone_and_preserves_synthetic_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "research.csv"
            write_rows(path, [
                row("2026-01-01T08:00:00+08:00", synthetic="true"),
                row("2026-01-01T08:05:00+08:00", synthetic="true"),
            ])

            rows = IMPORTER.read_and_validate([path], 5)

            self.assertEqual(
                rows[0]["recorded_at"],
                datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
            self.assertEqual(rows[0]["status"], "synthetic-research-import")

    def test_rejects_duplicate_or_irregular_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            duplicate = Path(temporary) / "duplicate.csv"
            write_rows(duplicate, [
                row("2026-01-01T00:00:00Z"),
                row("2026-01-01T00:00:00Z"),
            ])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                IMPORTER.read_and_validate([duplicate], 5)

            irregular = Path(temporary) / "irregular.csv"
            write_rows(irregular, [
                row("2026-01-01T00:00:00Z"),
                # Both timestamps are on the UTC five-minute grid, but the
                # missing intermediate sample must still fail cadence checks.
                row("2026-01-01T00:10:00Z"),
            ])
            with self.assertRaisesRegex(ValueError, "irregular cadence"):
                IMPORTER.read_and_validate([irregular], 5)

    def test_rejects_naive_or_off_grid_timestamps(self):
        with tempfile.TemporaryDirectory() as temporary:
            naive = Path(temporary) / "naive.csv"
            write_rows(naive, [row("2026-01-01T00:00:00")])
            with self.assertRaisesRegex(ValueError, "ISO-8601"):
                IMPORTER.read_and_validate([naive], 5)

            off_grid = Path(temporary) / "off-grid.csv"
            write_rows(off_grid, [
                row("2026-01-01T00:02:00Z"),
                row("2026-01-01T00:07:00Z"),
            ])
            with self.assertRaisesRegex(ValueError, "off the UTC"):
                IMPORTER.read_and_validate([off_grid], 5)


if __name__ == "__main__":
    unittest.main()
