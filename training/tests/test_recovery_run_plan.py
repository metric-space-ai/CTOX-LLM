from __future__ import annotations

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))

from build_recovery_run_plan import (  # noqa: E402
    MODEL,
    REQUIRED_LOSSES,
    safetensors_metadata,
    validate_activation_assignment,
    validate_smoke,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")


def write_metadata(path: Path, metadata: dict[str, str]) -> None:
    header = json.dumps({"__metadata__": metadata}, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header)


class RecoveryRunPlanTests(unittest.TestCase):
    def activation_fixture(self, root: Path, sample_ids: list[str]) -> tuple[Path, ...]:
        revision = "1" * 40
        provenance = "2" * 64
        source_plan = "3" * 64
        stats = root / "stats.safetensors"
        write_metadata(
            stats,
            {
                "format": "ctox.activation-diagonal.v1",
                "revision": revision,
                "local_model_provenance_sha256": provenance,
                "quant_plan_sha256": source_plan,
                "sample_ids": json.dumps(sample_ids, separators=(",", ":")),
                "samples": str(len(sample_ids)),
                "unobserved_tensors": "[]",
            },
        )
        sensitivity = root / "sensitivity.json"
        write_json(
            sensitivity,
            {
                "format": "ctox.q2q4.sensitivity.v1",
                "model": MODEL,
                "revision": revision,
                "local_model_provenance_sha256": provenance,
                "quant_plan_sha256": source_plan,
                "activation_stats_sha256": sha256(stats),
                "candidates": [{"name": "matrix.weight", "observed": True}],
            },
        )
        assignment = root / "assignment.json"
        write_json(
            assignment,
            {
                "format": "ctox.q2q4.assignment.v2",
                "plan_sha256": source_plan,
                "sensitivity_sha256": sha256(sensitivity),
                "budget_bytes": 100,
                "bytes_used": 100,
            },
        )
        plan = root / "plan.json"
        write_json(
            plan,
            {
                "format": "ctox.q2q4.quant-plan.v2",
                "model": MODEL,
                "revision": revision,
                "local_model_provenance_sha256": provenance,
                "mtp": "resident",
                "vision": "separate",
                "fits_fold_limit": True,
                "total_bytes": 100,
                "assignment": {"sha256": sha256(assignment)},
                "tensors": [
                    {
                        "name": "matrix.weight",
                        "dtype": "q2_b64",
                    }
                ],
            },
        )
        return stats, sensitivity, assignment, plan, Path(revision), Path(provenance)

    def test_reads_bounded_safetensors_metadata_without_torch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.safetensors"
            write_metadata(path, {"format": "example", "samples": "2"})
            self.assertEqual(
                safetensors_metadata(path), {"format": "example", "samples": "2"}
            )

    def test_activation_assignment_requires_complete_training_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = ["a", "b"]
            stats, sensitivity, assignment, plan, revision, provenance = (
                self.activation_fixture(root, ids)
            )
            document, evidence = validate_activation_assignment(
                stats,
                sensitivity,
                assignment,
                plan,
                revision.name,
                provenance.name,
                set(ids),
            )
            self.assertEqual(document["total_bytes"], 100)
            self.assertEqual(evidence["activation_samples"], 2)
            self.assertEqual(evidence["sensitivity_candidates"], 1)

            with self.assertRaisesRegex(ValueError, "complete training cohort"):
                validate_activation_assignment(
                    stats,
                    sensitivity,
                    assignment,
                    plan,
                    revision.name,
                    provenance.name,
                    {"a", "b", "c"},
                )

    def test_activation_assignment_rejects_unobserved_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats, sensitivity, assignment, plan, revision, provenance = (
                self.activation_fixture(root, ["a"])
            )
            document = json.loads(sensitivity.read_text())
            document["candidates"][0]["observed"] = False
            write_json(sensitivity, document)
            assignment_document = json.loads(assignment.read_text())
            assignment_document["sensitivity_sha256"] = sha256(sensitivity)
            write_json(assignment, assignment_document)
            plan_document = json.loads(plan.read_text())
            plan_document["assignment"]["sha256"] = sha256(assignment)
            write_json(plan, plan_document)
            with self.assertRaisesRegex(ValueError, "unobserved matrices"):
                validate_activation_assignment(
                    stats,
                    sensitivity,
                    assignment,
                    plan,
                    revision.name,
                    provenance.name,
                    {"a"},
                )

    def test_smoke_must_match_chunked_full_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.json"
            artifact = "a" * 64
            cache = "b" * 64
            provenance = "c" * 64
            report = {
                "format": "ctox.recovery.training-run.v1",
                "status": "bounded_run_complete",
                "model": MODEL,
                "revision": "d" * 40,
                "local_model_provenance_sha256": provenance,
                "artifact_sha256": artifact,
                "teacher_cache_set_sha256": cache,
                "max_optimizer_steps": 1,
                "prefill_chunk_tokens": 512,
                "oversize_policy": "fail",
                "gradient_checkpointing": False,
                "fixed_logical_qcodes": True,
                "fanout_s_in_policy": "independent",
                "cursor": {"optimizer_steps": 1, "samples_seen": 1},
                "skipped_oversize_samples": [],
                "recent_mean_losses": {name: 0.1 for name in REQUIRED_LOSSES},
                "base_graph": {"layers": 64},
                "mtp_graph": {"layers": 1},
            }
            write_json(path, report)
            smoke = validate_smoke(
                path, artifact, cache, "d" * 40, provenance, 512
            )
            self.assertEqual(smoke["optimizer_steps"], 1)

            report["gradient_checkpointing"] = True
            write_json(path, report)
            with self.assertRaisesRegex(ValueError, "gradient_checkpointing"):
                validate_smoke(path, artifact, cache, "d" * 40, provenance, 512)


if __name__ == "__main__":
    unittest.main()
