from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))

from build_release_preparation_plan import (  # noqa: E402
    build,
    parse_args,
    validate_source_snapshot,
)
from run_recovery_execution_plan import validate_stages  # noqa: E402


SCRIPTS = (
    "build_teacher_cache_set.py",
    "run_teacher_batches.py",
    "run_activation_batches.py",
    "finalize_activation_assignment.py",
    "build_quant_plan.py",
    "fit_recovery_scales.py",
    "pack_checkpoint.py",
    "run_bound_recovery_smoke.py",
    "train_recovery.py",
)


def write_batch_plan(path: Path, batches: int, samples: int) -> None:
    path.write_text(
        json.dumps(
            {
                "batches": [{"batch_index": index} for index in range(batches)],
                "summary": {"batches": batches, "samples": samples},
            }
        ),
        encoding="utf-8",
    )


def write_jsonl(path: Path, prefix: str, count: int) -> None:
    with path.open("w", encoding="utf-8") as output:
        for index in range(count):
            output.write(json.dumps({"id": f"{prefix}-{index:04d}"}) + "\n")


class ReleasePreparationPlanTests(unittest.TestCase):
    def test_cli_parser_exposes_every_build_input(self) -> None:
        values = {
            "python": "/python",
            "source-root": "/source",
            "source-commit": "a" * 40,
            "data-root": "/data",
            "model-source": "/model",
            "revision": "b" * 40,
            "local-model-provenance": "/provenance.json",
            "teacher-provenance-sha256": "c" * 64,
            "train-input": "/train.jsonl",
            "train-missing-batch-plan": "/missing-plan.json",
            "train-missing-prefix": "missing",
            "evaluation-input": "/evaluation.jsonl",
            "evaluation-batch-plan": "/evaluation-plan.json",
            "evaluation-prefix": "evaluation",
            "activation-batch-plan": "/activation-plan.json",
            "activation-prefix": "activation",
            "base-quant-plan": "/quant-plan.json",
            "ledger": "/ledger.jsonl",
            "hf-home": "/hf",
            "smoke-sample-id": "sample-id",
            "output": "/output.json",
        }
        argv = ["build_release_preparation_plan.py"]
        for option, value in values.items():
            argv.extend([f"--{option}", value])
        argv.extend(["--train-existing-verification", "/verification.json"])
        with patch.object(sys, "argv", argv):
            args = parse_args()
        for option in values:
            self.assertTrue(hasattr(args, option.replace("-", "_")), option)
        self.assertEqual(args.train_existing_verification, [Path("/verification.json")])

    def fixture(self, root: Path) -> argparse.Namespace:
        source = root / "source"
        training = source / "training"
        training.mkdir(parents=True)
        for name in SCRIPTS:
            (training / name).write_text(f"# {name}\n", encoding="utf-8")

        data = root / "data"
        for name in (
            "teacher-cache",
            "plans",
            "materialized",
            "provenance",
            "activation-stats",
            "sensitivity",
            "assignments",
            "recovery",
            "packs",
        ):
            (data / name).mkdir(parents=True)
        model = root / "model"
        hf_home = root / "hf"
        model.mkdir()
        hf_home.mkdir()

        train = data / "materialized/train.jsonl"
        evaluation = data / "materialized/evaluation.jsonl"
        write_jsonl(train, "train", 2_328)
        write_jsonl(evaluation, "evaluation", 642)
        missing_plan = data / "plans/missing.json"
        evaluation_plan = data / "plans/evaluation.json"
        activation_plan = data / "plans/activation.json"
        write_batch_plan(missing_plan, 2, 1_735)
        write_batch_plan(evaluation_plan, 2, 642)
        write_batch_plan(activation_plan, 2, 2_328)
        base_plan = data / "plans/base.json"
        provenance = data / "provenance/model.json"
        ledger = data / "ledger.jsonl"
        base_plan.write_text("{}\n", encoding="utf-8")
        provenance.write_text("{}\n", encoding="utf-8")
        ledger.write_text("", encoding="utf-8")

        existing = []
        for index in range(5):
            path = data / f"teacher-cache/existing-{index}.json"
            path.write_text(f"{{\"index\":{index}}}\n", encoding="utf-8")
            existing.append(path)
        for index in range(2):
            path = data / f"teacher-cache/missing-batch-{index:03d}-v1-verification-v1.json"
            path.write_text(f"{{\"index\":{index}}}\n", encoding="utf-8")

        for path in sorted(source.rglob("*"), reverse=True):
            path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
        source.chmod(stat.S_IMODE(source.stat().st_mode) & ~0o222)
        return argparse.Namespace(
            python=Path(sys.executable),
            source_root=source,
            source_commit="a" * 40,
            data_root=data,
            model_source=model,
            revision="b" * 40,
            local_model_provenance=provenance,
            teacher_provenance_sha256="c" * 64,
            train_input=train,
            train_existing_verification=existing,
            train_missing_batch_plan=missing_plan,
            train_missing_prefix="missing",
            evaluation_input=evaluation,
            evaluation_batch_plan=evaluation_plan,
            evaluation_prefix="evaluation",
            activation_batch_plan=activation_plan,
            activation_prefix="activation",
            base_quant_plan=base_plan,
            ledger=ledger,
            hf_home=hf_home,
            smoke_sample_id="train-0001",
            output=root / "plan.json",
        )

    def test_plan_is_hash_bound_serial_and_never_uses_gpu_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            document = build(args)
            stages = validate_stages(document)
            self.assertEqual(len(stages), 9)
            self.assertEqual(document["cohorts"]["training_samples"], 2_328)
            self.assertEqual(document["cohorts"]["evaluation_samples"], 642)
            self.assertEqual(document["source_snapshot"]["commit"], "a" * 40)
            self.assertEqual(document["source_snapshot"]["files"], len(SCRIPTS))
            for stage in stages:
                physical = stage["environment"].get("CUDA_VISIBLE_DEVICES", "")
                self.assertNotIn("0", physical.split(","))
                self.assertNotIn("--gradient-checkpointing", stage["argv"])
            train_cache = stages[0]["argv"]
            self.assertEqual(train_cache.count("--bound-verification"), 5)
            self.assertIn("--bound-batch-group", train_cache)
            self.assertNotIn("--verification", train_cache)
            self.assertNotIn("--batch-group", train_cache)
            self.assertEqual(stages[-1]["resume_policy"], "none")

    def test_source_snapshot_must_be_read_only_and_symlink_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            source = args.source_root
            source.chmod(0o755)
            with self.assertRaisesRegex(ValueError, "writable"):
                validate_source_snapshot(source, "a" * 40)


if __name__ == "__main__":
    unittest.main()
