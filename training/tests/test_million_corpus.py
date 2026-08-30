from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))

from audit_million_corpus import audit, validate_policy  # noqa: E402
from build_manifest import canonical_text  # noqa: E402
from build_million_corpus_evidence import (  # noqa: E402
    collect_partition,
    context_bucket,
    initialize_uniqueness,
    primary_group,
    validate_groups,
)


class MillionCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = json.loads((TRAINING / "MILLION_RECOVERY_POLICY.json").read_text())
        validate_policy(cls.policy)

    @staticmethod
    def partition(records: int) -> dict:
        bindings = {
            "materialized_sha256": "1" * 64,
            "domain_tags_sha256": "2" * 64,
            "service_tags_sha256": "3" * 64,
            "token_plan_sha256": "4" * 64,
            "provenance_manifest_sha256": "5" * 64,
            "semantic_dedup_sha256": "6" * 64,
        }
        return {
            "records": records,
            "unique_ids": records,
            "unique_payloads": records,
            "unique_semantic_clusters": records,
            **bindings,
            "content_root_sha256": "7" * 64,
            "binding_paths": {
                name: f"/evidence/{name}.json" for name in bindings
            },
            "primary_mix": {
                "general_dialogue_knowledge": records * 25 // 100,
                "coding_software": records * 20 // 100,
                "agentic_tools": records * 15 // 100,
                "mathematics_stem": records * 15 // 100,
                "professional": records * 10 // 100,
                "language_humanities_creative": records * 10 // 100,
                "long_context": records * 5 // 100,
            },
            "context_mix": {
                "up_to_2k": records * 650 // 1000,
                "2k_to_8k": records * 250 // 1000,
                "8k_to_32k": records * 80 // 1000,
                "32k_to_64k": records * 15 // 1000,
                "64k_to_128k": records * 5 // 1000,
            },
            "languages": {
                "en": records * 45 // 100,
                "de": records * 10 // 100,
                "zho": records * 8 // 100,
                "spa": records * 6 // 100,
                "fra": records * 5 // 100,
                "por": records * 4 // 100,
                "jpn": records * 4 // 100,
                "kor": records * 3 // 100,
                "arb": records * 3 // 100,
                "rus": records * 3 // 100,
                "hin": records * 3 // 100,
                "ita": records * 6 // 100,
            },
        }

    def evidence(self) -> dict:
        return {
            "format": "ctox.recovery-million-corpus-evidence.v1",
            "partitions": {
                "train": self.partition(1_000_000),
                "calibration": self.partition(50_000),
                "held_out": self.partition(50_000),
            },
            "hard_gates": {name: 0 for name in self.policy["hard_gates"]},
        }

    def test_exact_policy_mix_passes(self) -> None:
        report = audit(self.policy, self.evidence())
        self.assertEqual(report["status"], "passed")
        self.assertTrue(
            all(partition["status"] == "passed" for partition in report["partitions"].values())
        )

    def test_training_below_one_million_fails(self) -> None:
        evidence = self.evidence()
        evidence["partitions"]["train"] = self.partition(999_000)
        report = audit(self.policy, evidence)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(
            report["partitions"]["train"]["cardinality_gaps"]["records"]["required"],
            1_000_000,
        )

    def test_duplicate_or_cross_partition_semantics_fail(self) -> None:
        evidence = self.evidence()
        evidence["partitions"]["train"]["unique_semantic_clusters"] -= 1
        evidence["hard_gates"]["cross_partition_semantic_cluster_overlap"] = 1
        report = audit(self.policy, evidence)
        self.assertEqual(report["status"], "failed")
        self.assertIn(
            "unique_semantic_clusters",
            report["partitions"]["train"]["cardinality_gaps"],
        )
        self.assertIn("cross_partition_semantic_cluster_overlap", report["hard_gate_gaps"])

    def test_language_shortfall_fails_even_when_total_is_large_enough(self) -> None:
        evidence = self.evidence()
        train = evidence["partitions"]["train"]
        train["languages"]["de"] -= 1
        train["languages"]["en"] += 1
        report = audit(self.policy, evidence)
        self.assertEqual(report["status"], "failed")
        self.assertIn("de", report["partitions"]["train"]["language_gaps"])
        self.assertIn("en", report["partitions"]["train"]["language_gaps"])

    def test_unresolved_license_record_fails(self) -> None:
        evidence = self.evidence()
        evidence["hard_gates"]["unresolved_license_records"] = 1
        report = audit(self.policy, evidence)
        self.assertEqual(report["status"], "failed")
        self.assertIn("unresolved_license_records", report["hard_gate_gaps"])

    def test_context_buckets_close_every_policy_boundary(self) -> None:
        self.assertEqual(context_bucket(2_048), "up_to_2k")
        self.assertEqual(context_bucket(2_049), "2k_to_8k")
        self.assertEqual(context_bucket(8_193), "8k_to_32k")
        self.assertEqual(context_bucket(32_769), "32k_to_64k")
        self.assertEqual(context_bucket(65_537), "64k_to_128k")
        with self.assertRaises(ValueError):
            context_bucket(131_073)

    def test_primary_service_modes_override_domain_group(self) -> None:
        groups = json.loads((TRAINING / "MILLION_PRIMARY_GROUPS.json").read_text())
        validate_groups(groups)
        self.assertEqual(
            primary_group(
                "software_development",
                {"long_context_retrieval", "coding_debugging"},
                groups,
            ),
            "long_context",
        )

    def test_evidence_builder_binds_ordered_sidecars_and_semantic_contract(self) -> None:
        groups = json.loads((TRAINING / "MILLION_PRIMARY_GROUPS.json").read_text())
        record = {
            "id": "sample-a",
            "language": "de",
            "messages": [
                {"role": "user", "content": "Erkläre den Test."},
                {"role": "assistant", "content": "Gern."},
            ],
        }
        import hashlib

        record["prompt_sha256"] = hashlib.sha256(
            canonical_text(record).encode("utf-8")
        ).hexdigest()
        rows = {
            "materialized": record,
            "domain_tags": {"id": "sample-a", "primary_label": "software_development"},
            "service_tags": {"id": "sample-a", "labels": ["coding_debugging"]},
            "token_counts": {"id": "sample-a", "sequence_tokens": 512},
            "provenance": {
                "id": "sample-a",
                "source_repo": "example/recovery",
                "source_revision": "a" * 40,
                "source_id": "row-1",
                "prompt_sha256": record["prompt_sha256"],
                "license": "apache-2.0",
                "release_eligible": True,
            },
            "semantic_dedup": {
                "id": "sample-a",
                "cluster_id": "cluster-a",
                "embedding_model": "example/multilingual",
                "embedding_revision": "b" * 40,
                "distance_threshold": 0.08,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for name, row in rows.items():
                path = root / f"{name}.jsonl"
                path.write_text(json.dumps(row, sort_keys=True) + "\n")
                paths[name] = path
            database = sqlite3.connect(":memory:")
            initialize_uniqueness(database)
            hard_gates = Counter()
            semantic_contract = {}
            report = collect_partition(
                "train", paths, groups, database, hard_gates, semantic_contract
            )
            database.close()
        self.assertEqual(report["records"], 1)
        self.assertEqual(report["unique_semantic_clusters"], 1)
        self.assertEqual(report["primary_mix"], {"coding_software": 1})
        self.assertEqual(report["context_mix"], {"up_to_2k": 1})
        self.assertEqual(hard_gates, {})
        self.assertEqual(semantic_contract["distance_threshold"], 0.08)


if __name__ == "__main__":
    unittest.main()
