"""Safety-boundary tests for the live benchmark command."""

from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest

from agent_eval.cli import main


class AgentEvalCliTests(unittest.TestCase):
    def test_high_volume_run_is_rejected_before_model_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            error_output = io.StringIO()
            with redirect_stderr(error_output):
                status = main(
                    [
                        "live",
                        "--split",
                        "validation",
                        "--repetitions",
                        "4",
                        "--traces",
                        str(root / "traces.jsonl"),
                        "--report",
                        str(root / "report.json"),
                    ]
                )

            self.assertEqual(status, 2)
            self.assertIn("refusing 120 case runs", error_output.getvalue())
            self.assertFalse((root / "traces.jsonl").exists())
            self.assertFalse((root / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
