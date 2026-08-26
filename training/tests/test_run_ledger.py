import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))

from run_ledger import GpuRun  # noqa: E402


class GpuRunBudgetTests(unittest.TestCase):
    def test_runtime_reservation_is_measured_and_persisted(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "ledger.jsonl"
            with patch("run_ledger.time.time", side_effect=[100.0, 3_700.0, 3_700.0]):
                with GpuRun(
                    ledger,
                    "recovery",
                    1,
                    ["train"],
                    maximum_gpu_hours=0.5,
                ) as run:
                    self.assertTrue(run.budget_exhausted())
            record = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertEqual(record["gpu_hours"], 1.0)
            self.assertEqual(record["maximum_gpu_hours"], 0.5)

    def test_runtime_reservation_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "maximum_gpu_hours"):
            GpuRun(Path("ledger"), "recovery", 1, ["train"], maximum_gpu_hours=0)


if __name__ == "__main__":
    unittest.main()
