from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))

from build_manifest import canonical_text, recovery_payload  # noqa: E402
from materialize_prompts import load_manifests  # noqa: E402
from select_manifest import select  # noqa: E402


class DatasetPipelineTests(unittest.TestCase):
    def test_hash_covers_reference_answer(self) -> None:
        first = {"input": "2+2?", "output": "4"}
        second = {"input": "2+2?", "output": "5"}
        self.assertNotEqual(canonical_text(first), canonical_text(second))

    def test_paired_sample_becomes_conversation(self) -> None:
        self.assertEqual(
            recovery_payload({"instruction": "hello", "response": "world"}),
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ]
            },
        )

    def test_quarantined_manifest_requires_explicit_opt_in(self) -> None:
        payload = canonical_text({"prompt": "secret"})
        record = {
            "id": "a" * 64,
            "source_repo": "example/data",
            "source_revision": "revision",
            "subset": "default",
            "split": "train",
            "source_id": "1",
            "prompt_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "release_eligible": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "quarantined sample"):
                load_manifests([manifest], allow_quarantined=False)
            groups = load_manifests([manifest], allow_quarantined=True)
            self.assertEqual(sum(map(len, groups.values())), 1)

    def test_selection_is_balanced_and_order_independent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for category in ("code", "math"):
                path = Path(directory) / f"{category}.jsonl"
                records = [
                    {"id": hashlib.sha256(f"{category}-{index}".encode()).hexdigest(), "category": category}
                    for index in range(10)
                ]
                path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
                paths.append(path)
            first = select(paths, per_manifest=3, seed="fixed")
            second = select(paths, per_manifest=3, seed="fixed")
            self.assertEqual(first, second)
            self.assertEqual([record["category"] for record in first].count("code"), 3)
            self.assertEqual([record["category"] for record in first].count("math"), 3)


if __name__ == "__main__":
    unittest.main()
