from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))

from select_teacher_throughput_probe import select  # noqa: E402
from run_teacher_throughput_probe import summarize  # noqa: E402


class TeacherThroughputProbeTests(unittest.TestCase):
    def test_selection_is_hash_stable_and_context_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            materialized = root / "train.jsonl"
            tokens = root / "tokens.jsonl"
            lengths = [100, 200, 3_000, 4_000, 10_000, 20_000, 40_000, 60_000, 70_000, 100_000]
            with materialized.open("w", encoding="utf-8") as sample_output, tokens.open(
                "w", encoding="utf-8"
            ) as token_output:
                for index, length in enumerate(lengths):
                    sample_id = f"sample-{index}"
                    sample_output.write(json.dumps({"id": sample_id}) + "\n")
                    token_output.write(
                        json.dumps({"id": sample_id, "sequence_tokens": length}) + "\n"
                    )
            quotas = {
                "up_to_2k": 1,
                "2k_to_8k": 1,
                "8k_to_32k": 1,
                "32k_to_64k": 1,
                "64k_to_128k": 1,
            }
            with patch("select_teacher_throughput_probe.QUOTAS", quotas):
                first = select(materialized, tokens, "seed")
                second = select(materialized, tokens, "seed")
            self.assertEqual(first, second)
            self.assertEqual(len(first[0]), 5)
            selected_tokens = [json.loads(line)["sequence_tokens"] for line in first[1]]
            self.assertEqual(
                sum(value <= 2_048 for value in selected_tokens), 1
            )
            self.assertEqual(
                sum(2_048 < value <= 8_192 for value in selected_tokens), 1
            )
            self.assertEqual(
                sum(8_192 < value <= 32_768 for value in selected_tokens), 1
            )
            self.assertEqual(
                sum(32_768 < value <= 65_536 for value in selected_tokens), 1
            )
            self.assertEqual(sum(value > 65_536 for value in selected_tokens), 1)

    def test_selection_fails_when_a_context_bucket_is_short(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            materialized = root / "train.jsonl"
            tokens = root / "tokens.jsonl"
            materialized.write_text('{"id":"only"}\n', encoding="utf-8")
            tokens.write_text(
                '{"id":"only","sequence_tokens":100}\n', encoding="utf-8"
            )
            quotas = {
                "up_to_2k": 1,
                "2k_to_8k": 1,
                "8k_to_32k": 0,
                "32k_to_64k": 0,
                "64k_to_128k": 0,
            }
            with patch("select_teacher_throughput_probe.QUOTAS", quotas):
                with self.assertRaisesRegex(ValueError, "cannot fill probe quotas"):
                    select(materialized, tokens, "seed")

    def test_probe_summary_uses_verified_cache_and_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "probe-batch-000-v1"
            cache.mkdir()
            (cache / "run.json").write_text(
                json.dumps(
                    {
                        "cuda_memory": [
                            {"index": 0, "peak_allocated_bytes": 10, "peak_reserved_bytes": 12},
                            {"index": 1, "peak_allocated_bytes": 20, "peak_reserved_bytes": 24},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            verification = root / "probe-batch-000-v1-verification-v1.json"
            verification.write_text(
                json.dumps({"status": "passed", "samples": 10_000, "artifact_bytes": 50_000}),
                encoding="utf-8",
            )
            ledger = root / "ledger.jsonl"
            ledger.write_text(
                json.dumps(
                    {
                        "stage": "teacher-cache",
                        "command": ["cache_teacher.py", "--output", str(cache)],
                        "success": True,
                        "elapsed_seconds": 100.0,
                        "gpu_hours": 2 / 36,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = summarize(
                {"records": 10_000, "sequence_tokens": 1_000_000},
                {
                    "summary": {"samples": 10_000},
                    "batches": [{"batch_index": 0, "samples": 10_000}],
                },
                root,
                "probe",
                ledger,
            )
            self.assertEqual(result["samples_per_second"], 100.0)
            self.assertEqual(result["sequence_tokens_per_second"], 10_000.0)
            self.assertEqual(result["projected_million"]["artifact_bytes"], 5_000_000)
            self.assertEqual(result["peak_reserved_bytes_by_logical_device"], [12, 24])


if __name__ == "__main__":
    unittest.main()
