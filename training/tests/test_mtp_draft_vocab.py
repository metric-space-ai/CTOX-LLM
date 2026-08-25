from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory


TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))

from build_mtp_draft_vocab import (  # noqa: E402
    coverage_ppm,
    select_tokens,
    validate_teacher_cache_set,
)


class MtpDraftVocabularyTests(unittest.TestCase):
    def test_selection_balances_common_code_domain_and_language_tokens(self) -> None:
        selected = select_tokens(
            overall=Counter({0: 1000, 1: 100, 2: 10, 3: 1}),
            code=Counter({2: 100, 0: 1}),
            domains={"common": Counter({0: 100}), "rare": Counter({3: 10})},
            languages={"en": Counter({0: 100}), "de": Counter({1: 10})},
            token_count=4,
            required_ids=[],
        )
        self.assertEqual(selected, [0, 1, 2, 3])

    def test_selection_is_canonical_and_keeps_required_ids(self) -> None:
        selected = select_tokens(
            overall=Counter({9: 4, 3: 4, 5: 1}),
            code=Counter({3: 1}),
            domains={"software": Counter({9: 1})},
            languages={"en": Counter({9: 1})},
            token_count=3,
            required_ids=[7],
        )
        self.assertEqual(selected, sorted(selected))
        self.assertIn(7, selected)

    def test_coverage_uses_token_frequency_not_unique_ids(self) -> None:
        self.assertEqual(coverage_ppm(Counter({1: 9, 2: 1}), {1}), 900_000)

    def test_impossible_selection_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "distinct output"):
            select_tokens(
                overall=Counter({1: 1}),
                code=Counter({1: 1}),
                domains={"software": Counter({1: 1})},
                languages={"en": Counter({1: 1})},
                token_count=2,
                required_ids=[],
            )

    def test_teacher_cache_set_must_bind_exact_rehashed_mtp_input(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cache-set.json"
            document = {
                "format": "ctox.teacher-cache-set.v1",
                "samples": 2,
                "expected_input": {"sha256": "a" * 64, "records": 2},
                "all_artifacts_rehashed": True,
                "settings": {"mtp_targets": True},
            }
            path.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(
                validate_teacher_cache_set(path, "a" * 64, 2)["samples"], 2
            )
            document["settings"]["mtp_targets"] = False
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact rehashed"):
                validate_teacher_cache_set(path, "a" * 64, 2)


if __name__ == "__main__":
    unittest.main()
