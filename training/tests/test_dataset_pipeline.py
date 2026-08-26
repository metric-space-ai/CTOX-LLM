from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from collections import Counter
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))

from build_manifest import (  # noqa: E402
    SOURCES,
    canonical_text,
    category_for,
    language_for,
    license_ids,
    recovery_payload,
    source_id_for,
    source_uses_raw_jsonl,
)
from build_teacher_cache_set import batch_group, main as build_teacher_cache_set_main  # noqa: E402
from audit_corpus import percentile  # noqa: E402
from audit_service_coverage import service_coverage_report  # noqa: E402
from build_quant_plan import (  # noqa: E402
    CONTAINER_MANIFEST_RESERVE,
    FOLD_PACKAGE_LIMIT,
    FOLD_RESIDENT_LIMIT,
    validate_assignment_source,
)
from cache_teacher import (  # noqa: E402
    mtp_target_positions,
    position_sets,
    recover_unindexed_tail,
    resume_prefix,
    save_sample_atomic,
    validate_local_model_provenance,
)
from classify_domains import (  # noqa: E402
    classification_text,
    deterministic_labels,
    deterministic_primary_label,
    quota_gaps,
    validate_rubric,
)
from classify_service_modes import (  # noqa: E402
    deterministic_modes,
    validate_rubric as validate_service_rubric,
)
from apply_primary_overrides import apply_overrides  # noqa: E402
from audit_selection_coverage import coverage_report, validate_language_rubric  # noqa: E402
from collect_activation_stats import (  # noqa: E402
    checkpoint_weight_name,
    prefill_ranges,
    quantized_source_names,
    save_file_atomic,
)
from ctox_artifact import CtoxArtifact, ENDIAN_MARKER, HEADER, MAGIC  # noqa: E402
from recovery_io import atomic_json, prepare_output_transaction  # noqa: E402
from materialize_prompts import load_local_materialized, load_manifests  # noqa: E402
from merge_manifests import merge  # noqa: E402
from merge_domain_tags import merge_ordered_tags  # noqa: E402
from merge_activation_stats import merged_metadata, source_runtime_profiles  # noqa: E402
from mtp_teacher import mtp_checkpoint_weight_name, mtp_parameter_mapping  # noqa: E402
from optimize_q4_budget import (  # noqa: E402
    initial_selections,
    layout_bytes,
    mixed_tensor_bytes,
    optimized_selections,
    validate_sensitivity_contract,
)
from plan_teacher_cache import sample_tensor_bytes  # noqa: E402
from plan_teacher_batches import batches as plan_teacher_batches  # noqa: E402
from plan_activation_batches import activation_batches  # noqa: E402
from prompt_format import normalize_content, normalize_messages, normalize_tool_call  # noqa: E402
from filter_recovery_cohort import filter_records  # noqa: E402
from generate_long_context import generated_record  # noqa: E402
from fit_recovery_scales import quant_dtype_ranges, validate_recovery_inputs  # noqa: E402
from evaluate_recovery import (  # noqa: E402
    MetricAggregate,
    logical_qcode_root,
    require_exact_ids,
)
from compare_recovery_evaluations import compare_reports  # noqa: E402
from fanout_recovery import (  # noqa: E402
    INDEPENDENT_POLICY,
    QWEN38_FANOUT_POLICY,
    fanout_group_sha256,
    qwen38_fanout_groups,
    tie_fanout_s_in,
)
from recovery_training_state import (  # noqa: E402
    normalize_accumulated_gradients,
    recovery_training_status,
    resolve_sample_indices,
    training_order,
)

try:  # Optional local training dependency; exercised in the pinned GPU venv.
    import torch  # noqa: E402
    from end_to_end_recovery import unique_scale_parameters  # noqa: E402
    from recovery_modules import (  # noqa: E402
        end_to_end_recovery_loss,
        normalized_hidden_loss,
        normalized_hidden_loss_contribution,
        sparse_teacher_kl,
        streamed_sparse_target_losses,
        supervised_mtp_token_loss,
        supervised_next_token_loss,
    )
except ModuleNotFoundError:
    torch = None
    unique_scale_parameters = None
from pack_checkpoint import validate_recovery_source  # noqa: E402
from packed_recovery_ops import (  # noqa: E402
    packed_linear,
    packed_recovery_embedding_class,
    packed_recovery_linear_class,
)
from packed_recovery_model import PackedRecoveryRegistry  # noqa: E402
from packed_student_model import (  # noqa: E402
    artifact_to_runtime_name,
    runtime_to_artifact_name,
    set_parameter,
    set_submodule,
)
from score_quant_sensitivity import (  # noqa: E402
    fixed_q4,
    quantized_entries,
    row_group_document,
    validate_stats_bindings,
)
from select_manifest import select  # noqa: E402
from select_activation_calibration import (  # noqa: E402
    load_token_counts,
    select_calibration,
    sequence_bucket,
)
from select_teacher_smoke import select_ids as select_teacher_smoke_ids  # noqa: E402
from select_uncached_teacher_records import select_missing  # noqa: E402
from select_primary_domain_supplement import select_supplement  # noqa: E402
from select_service_supplement import (  # noqa: E402
    select_supplement as select_service_supplement,
)
from select_coverage_supplement import (  # noqa: E402
    assigned_tag,
    candidate_coverage,
    primary_assignment,
    select_joint_supplement,
)
from split_manifests import split  # noqa: E402
from teacher_runtime import FLA_KERNEL_REVISION, weight_max_memory  # noqa: E402
from teacher_cache_dataset import VerifiedTeacherCache  # noqa: E402
from verify_vendor_manifest import verify  # noqa: E402
from verify_local_model import root_digest  # noqa: E402
from verify_teacher_cache import expected_tensor_specs  # noqa: E402
from verify_activation_stats import (  # noqa: E402
    expected_batch as expected_activation_batch,
    expected_keys as expected_activation_keys,
)
from run_teacher_batches import (  # noqa: E402
    cache_environment,
    completed_batch_matches,
    gpu_weight_memory_for_batch,
)
from run_activation_batches import (  # noqa: E402
    completed_batch_matches as completed_activation_batch_matches,
)
from finalize_activation_assignment import verified_artifacts  # noqa: E402


class DatasetPipelineTests(unittest.TestCase):
    @staticmethod
    def qwen38_fanout_weight_names() -> set[str]:
        names: set[str] = set()
        for layer in range(64):
            prefix = f"model.language_model.layers.{layer}"
            names.update(
                {
                    f"{prefix}.mlp.gate_proj.weight",
                    f"{prefix}.mlp.up_proj.weight",
                }
            )
            if (layer + 1) % 4 == 0:
                names.update(
                    f"{prefix}.self_attn.{projection}.weight"
                    for projection in ("q_proj", "k_proj", "v_proj")
                )
            else:
                names.update(
                    f"{prefix}.linear_attn.{projection}.weight"
                    for projection in (
                        "in_proj_qkv",
                        "in_proj_z",
                        "in_proj_a",
                        "in_proj_b",
                    )
                )
        names.update(
            f"mtp.layers.0.self_attn.{projection}.weight"
            for projection in ("q_proj", "k_proj", "v_proj")
        )
        names.update(
            f"mtp.layers.0.mlp.{projection}.weight"
            for projection in ("gate_proj", "up_proj")
        )
        return names

    @staticmethod
    def recovery_record(sample_id: str, prompt: str, answer: str) -> dict:
        record = {
            "id": sample_id,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
        }
        record["prompt_sha256"] = hashlib.sha256(
            canonical_text(record).encode("utf-8")
        ).hexdigest()
        return record

    def test_activation_calibration_sequence_buckets_are_closed(self) -> None:
        self.assertEqual(sequence_bucket(4_096), "up_to_4k")
        self.assertEqual(sequence_bucket(4_097), "4k_16k")
        self.assertEqual(sequence_bucket(16_384), "4k_16k")
        self.assertEqual(sequence_bucket(16_385), "16k_32k")
        self.assertEqual(sequence_bucket(32_769), "32k_64k")
        self.assertEqual(sequence_bucket(65_537), "64k_96k")
        self.assertEqual(sequence_bucket(98_305), "over_96k")

    def test_activation_calibration_selection_is_deterministic_and_closes_quotas(
        self,
    ) -> None:
        records = [
            {"id": "a", "language": "de", "category": "code"},
            {"id": "b", "language": "en", "category": "agentic"},
            {"id": "c", "language": "de", "category": "chat"},
            {"id": "d", "language": "en", "category": "math"},
            {"id": "e", "language": "fr", "category": "long_context"},
        ]
        domains = {
            sample_id: {"primary_label": domain}
            for sample_id, domain in zip("abcde", ("software", "tools", "software", "math", "tools"))
        }
        services = {
            sample_id: {"labels": labels}
            for sample_id, labels in zip(
                "abcde",
                (("implementation",), ("tool_calling",), ("implementation",), ("reasoning",), ("reasoning",)),
            )
        }
        tokens = {"a": 2_000, "b": 5_000, "c": 18_000, "d": 34_000, "e": 100_000}
        requirements = {
            ("domain", "software"): 1,
            ("domain", "tools"): 1,
            ("language", "de"): 1,
            ("language", "en"): 1,
            ("language", "fr"): 1,
            ("service", "implementation"): 1,
            ("service", "tool_calling"): 1,
            ("service", "reasoning"): 1,
            ("category", "long_context"): 1,
            ("length", "over_96k"): 1,
        }
        first, counts = select_calibration(
            records, domains, services, tokens, requirements, 5, 131_072
        )
        second, _ = select_calibration(
            records, domains, services, tokens, requirements, 5, 131_072
        )
        self.assertEqual(first, second)
        self.assertEqual(set(first), set("abcde"))
        for feature, minimum in requirements.items():
            self.assertGreaterEqual(counts[feature], minimum)

    def test_activation_calibration_rejects_mismatched_ids_and_impossible_quota(
        self,
    ) -> None:
        records = [{"id": "a", "language": "en", "category": "chat"}]
        domains = {"a": {"primary_label": "general"}}
        services = {"a": {"labels": ["conversation"]}}
        with self.assertRaisesRegex(ValueError, "token counts differ"):
            select_calibration(
                records, domains, services, {}, {("domain", "general"): 1}, 1, 4_096
            )
        with self.assertRaisesRegex(ValueError, "cannot satisfy"):
            select_calibration(
                records,
                domains,
                services,
                {"a": 1_000},
                {("language", "de"): 1},
                1,
                4_096,
            )

    def test_activation_calibration_token_plans_merge_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            first.write_text(json.dumps({"samples": [{"id": "a", "sequence_tokens": 7}]}))
            second.write_text(json.dumps({"samples": [{"id": "b", "sequence_tokens": 11}]}))
            self.assertEqual(load_token_counts([first, second], {"a", "b"}), {"a": 7, "b": 11})
            second.write_text(
                json.dumps(
                    {
                        "samples": [
                            {"id": "a", "sequence_tokens": 8},
                            {"id": "b", "sequence_tokens": 11},
                        ]
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "disagree"):
                load_token_counts([first, second], {"a", "b"})

    def test_activation_batches_are_contiguous_and_token_bounded(self) -> None:
        records = [{"id": sample_id} for sample_id in "abcde"]
        token_counts = {"a": 20, "b": 30, "c": 60, "d": 10, "e": 10}
        planned = activation_batches(records, token_counts, 3, 70, 64)
        self.assertEqual(
            planned,
            [
                {
                    "batch_index": 0,
                    "start_sample": 0,
                    "samples": 2,
                    "sequence_tokens": 50,
                    "maximum_sample_tokens": 30,
                    "first_id": "a",
                    "last_id": "b",
                },
                {
                    "batch_index": 1,
                    "start_sample": 2,
                    "samples": 2,
                    "sequence_tokens": 70,
                    "maximum_sample_tokens": 60,
                    "first_id": "c",
                    "last_id": "d",
                },
                {
                    "batch_index": 2,
                    "start_sample": 4,
                    "samples": 1,
                    "sequence_tokens": 10,
                    "maximum_sample_tokens": 10,
                    "first_id": "e",
                    "last_id": "e",
                },
            ],
        )
        with self.assertRaisesRegex(ValueError, "sequence limit"):
            activation_batches(records, token_counts, 3, 70, 59)

    def test_activation_statistics_save_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stats.safetensors"

            def save_file(_tensors, path, metadata):
                Path(path).write_bytes(json.dumps(metadata).encode())

            save_file_atomic(save_file, {"tensor": object()}, output, {"format": "test"})
            self.assertEqual(json.loads(output.read_text()), {"format": "test"})
            self.assertFalse((output.parent / f".{output.name}.tmp").exists())

    def test_activation_statistics_failed_save_leaves_no_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stats.safetensors"

            def fail(_tensors, path, metadata):
                Path(path).write_bytes(b"partial")
                raise RuntimeError(metadata["reason"])

            with self.assertRaisesRegex(RuntimeError, "injected"):
                save_file_atomic(fail, {}, output, {"reason": "injected"})
            self.assertFalse(output.exists())
            self.assertFalse((output.parent / f".{output.name}.tmp").exists())

    def test_activation_verifier_derives_the_complete_tensor_contract(self) -> None:
        entries = {
            "model.language_model.embed_tokens.weight": {},
            "lm_head.weight": {},
            "model.language_model.layers.0.mlp.gate_proj.weight": {},
        }
        self.assertEqual(
            expected_activation_keys(entries),
            {
                "model.language_model.embed_tokens.weight.row_count",
                "lm_head.weight.input_mean_sq",
                "lm_head.weight.token_count",
                "model.language_model.layers.0.mlp.gate_proj.weight.input_mean_sq",
                "model.language_model.layers.0.mlp.gate_proj.weight.output_mean_sq",
                "model.language_model.layers.0.mlp.gate_proj.weight.token_count",
            },
        )

    def test_activation_verifier_binds_batch_boundaries_to_input_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            source.write_text(
                "".join(json.dumps({"id": sample_id}) + "\n" for sample_id in "abc")
            )
            batch_plan = root / "batches.json"
            batch_plan.write_text(
                json.dumps(
                    {
                        "format": "ctox.activation-batch-plan.v1",
                        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "batches": [
                            {
                                "batch_index": 0,
                                "start_sample": 1,
                                "samples": 2,
                                "first_id": "b",
                                "last_id": "c",
                            }
                        ],
                    }
                )
            )
            batch, ids = expected_activation_batch(batch_plan, source, 0)
            self.assertEqual(batch["samples"], 2)
            self.assertEqual(ids, ["b", "c"])
            source.write_text(json.dumps({"id": "changed"}) + "\n")
            with self.assertRaisesRegex(ValueError, "hash"):
                expected_activation_batch(batch_plan, source, 0)

    def test_activation_runner_skip_requires_current_artifact_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "stats.safetensors"
            artifact.write_bytes(b"sealed")
            batch = {"batch_index": 2, "samples": 3, "sequence_tokens": 5}
            verification = {
                "status": "passed",
                "artifact_sha256": hashlib.sha256(b"sealed").hexdigest(),
                "batch_plan_sha256": "batches",
                "quant_plan_sha256": "quant",
                "model": "model",
                "revision": "revision",
                "local_model_provenance_sha256": "provenance",
                **batch,
            }
            self.assertTrue(
                completed_activation_batch_matches(
                    artifact,
                    verification,
                    batch,
                    "batches",
                    "quant",
                    "model",
                    "revision",
                    "provenance",
                )
            )
            artifact.write_bytes(b"changed")
            self.assertFalse(
                completed_activation_batch_matches(
                    artifact,
                    verification,
                    batch,
                    "batches",
                    "quant",
                    "model",
                    "revision",
                    "provenance",
                )
            )

    def test_sensitivity_rejects_cross_identity_activation_statistics(self) -> None:
        metadata = {
            "format": "ctox.activation-diagonal.v1",
            "quant_plan_sha256": "plan",
            "local_model_provenance_sha256": "teacher",
        }
        validate_stats_bindings(metadata, "plan", "teacher")
        with self.assertRaisesRegex(ValueError, "quant plan"):
            validate_stats_bindings(metadata, "other", "teacher")
        with self.assertRaisesRegex(ValueError, "BF16 provenance"):
            validate_stats_bindings(metadata, "plan", "other")

    def test_q4_optimizer_requires_complete_identity_bound_sensitivity(self) -> None:
        plan = {"model": "qwen", "revision": "revision"}
        sensitivity = {
            "format": "ctox.q2q4.sensitivity.v1",
            "model": "qwen",
            "revision": "revision",
            "quant_plan_sha256": "plan",
            "candidates": [{"name": "weight", "observed": True}],
        }
        validate_sensitivity_contract(sensitivity, plan, "plan")
        sensitivity["candidates"][0]["observed"] = False
        with self.assertRaisesRegex(ValueError, "unobserved"):
            validate_sensitivity_contract(sensitivity, plan, "plan")
        sensitivity["candidates"][0]["observed"] = True
        with self.assertRaisesRegex(ValueError, "bound"):
            validate_sensitivity_contract(sensitivity, plan, "other")

    def test_activation_finalizer_requires_every_verified_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "release-batch-000-v1.safetensors"
            artifact.write_bytes(b"sealed")
            batch = {"batch_index": 0, "samples": 2, "sequence_tokens": 3}
            verification = {
                "status": "passed",
                "artifact_sha256": hashlib.sha256(b"sealed").hexdigest(),
                "batch_plan_sha256": "batches",
                "quant_plan_sha256": "plan",
                "model": "model",
                "revision": "revision",
                "local_model_provenance_sha256": "teacher",
                **batch,
            }
            (root / "release-batch-000-v1-verification-v1.json").write_text(
                json.dumps(verification)
            )
            self.assertEqual(
                verified_artifacts(
                    root,
                    "release",
                    [batch],
                    "batches",
                    "plan",
                    "model",
                    "revision",
                    "teacher",
                ),
                [artifact],
            )
            artifact.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "does not match"):
                verified_artifacts(
                    root,
                    "release",
                    [batch],
                    "batches",
                    "plan",
                    "model",
                    "revision",
                    "teacher",
                )

    def test_quant_plan_rebuild_requires_the_exact_assignment_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source-plan.json"
            source.write_bytes(b"source")
            assignment = {
                "plan_sha256": hashlib.sha256(b"source").hexdigest(),
                "budget_bytes": FOLD_RESIDENT_LIMIT,
                "bytes_used": FOLD_RESIDENT_LIMIT - 1,
            }
            validate_assignment_source(assignment, source)
            with self.assertRaisesRegex(ValueError, "source quant plan"):
                validate_assignment_source(
                    {**assignment, "plan_sha256": "wrong"}, source
                )
            with self.assertRaisesRegex(ValueError, "requires"):
                validate_assignment_source(assignment, None)
            with self.assertRaisesRegex(ValueError, "resident budget"):
                validate_assignment_source(
                    {**assignment, "budget_bytes": FOLD_RESIDENT_LIMIT + 1},
                    source,
                )

    def test_recovery_fit_binds_plan_stats_and_bf16_identity(self) -> None:
        plan = {
            "revision": "revision",
            "local_model_provenance_sha256": "teacher",
        }
        metadata = {
            "format": "ctox.activation-diagonal.v1",
            "revision": "revision",
            "local_model_provenance_sha256": "teacher",
        }
        validate_recovery_inputs(plan, metadata, "revision", "teacher")
        with self.assertRaisesRegex(ValueError, "plan does not match"):
            validate_recovery_inputs(plan, metadata, "revision", "other")
        with self.assertRaisesRegex(ValueError, "statistics revision"):
            validate_recovery_inputs(
                plan,
                {**metadata, "revision": "other"},
                "revision",
                "teacher",
            )

    def test_teacher_cache_batch_group_binds_plan_and_contiguous_verifications(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "plan.json"
            encoded = json.dumps(
                {
                    "summary": {"batches": 2, "samples": 3},
                    "batches": [
                        {"batch_index": 0},
                        {"batch_index": 1},
                    ],
                },
                sort_keys=True,
            ).encode()
            plan.write_bytes(encoded)
            paths, record, samples = batch_group(plan, root, "teacher")
            self.assertEqual(samples, 3)
            self.assertEqual(record["batches"], 2)
            self.assertEqual(record["samples"], 3)
            self.assertEqual(record["batch_plan_sha256"], hashlib.sha256(encoded).hexdigest())
            self.assertEqual(
                [path.name for path in paths],
                [
                    "teacher-batch-000-v1-verification-v1.json",
                    "teacher-batch-001-v1-verification-v1.json",
                ],
            )

            plan.write_text(
                json.dumps(
                    {
                        "summary": {"batches": 2, "samples": 3},
                        "batches": [
                            {"batch_index": 0},
                            {"batch_index": 2},
                        ],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "not contiguous"):
                batch_group(plan, root, "teacher")

    def test_teacher_cache_set_combines_reused_verification_and_new_batch_group(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            revision = "revision"
            provenance = "a" * 64

            def verification(path: Path, sample_id: str, cache: Path) -> None:
                path.write_text(
                    json.dumps(
                        {
                            "format": "ctox.teacher-cache-verification.v1",
                            "status": "passed",
                            "teacher_revision": revision,
                            "teacher_provenance_sha256": provenance,
                            "hidden_layers": [0],
                            "hidden_size": 8,
                            "top_k": 4,
                            "mtp_targets": True,
                            "cache": str(cache),
                            "samples": 1,
                            "artifact_bytes": 3,
                            "artifact_root_sha256": "b" * 64,
                            "artifacts": [
                                {
                                    "id": sample_id,
                                    "file": f"{sample_id}.safetensors",
                                    "bytes": 3,
                                    "sha256": "c" * 64,
                                }
                            ],
                        }
                    )
                )

            reused = root / "reused.json"
            verification(reused, "1" * 64, root / "old-cache")
            reused_sha256 = hashlib.sha256(reused.read_bytes()).hexdigest()
            plan = root / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "summary": {"batches": 1, "samples": 1},
                        "batches": [{"batch_index": 0}],
                    }
                )
            )
            grouped = root / "new-batch-000-v1-verification-v1.json"
            verification(grouped, "2" * 64, root / "new-cache")
            expected = root / "expected.jsonl"
            expected.write_text(
                "\n".join(
                    json.dumps({"id": sample_id})
                    for sample_id in ("1" * 64, "2" * 64)
                )
                + "\n"
            )
            output = root / "set.json"
            unbound_arguments = [
                "build_teacher_cache_set.py",
                "--verification",
                str(reused),
                "--batch-group",
                str(plan),
                str(root),
                "new",
                "--expected-input",
                str(expected),
                "--teacher-revision",
                revision,
                "--teacher-provenance-sha256",
                provenance,
                "--output",
                str(output),
                "--skip-artifact-rehash",
            ]
            with patch.object(sys, "argv", unbound_arguments):
                with self.assertRaisesRegex(SystemExit, "requires bound"):
                    build_teacher_cache_set_main()

            arguments = [
                "build_teacher_cache_set.py",
                "--bound-verification",
                str(reused),
                reused_sha256,
                "--batch-group",
                str(plan),
                str(root),
                "new",
                "--expected-input",
                str(expected),
                "--teacher-revision",
                revision,
                "--teacher-provenance-sha256",
                provenance,
                "--output",
                str(output),
                "--skip-artifact-rehash",
            ]
            with patch.object(sys, "argv", arguments):
                build_teacher_cache_set_main()
            document = json.loads(output.read_text())
            self.assertEqual(document["samples"], 2)
            self.assertEqual(document["expected_input"]["records"], 2)
            self.assertEqual(document["batch_groups"][0]["samples"], 1)
            self.assertEqual(
                document["bound_verifications"],
                [
                    {
                        "path": str(reused.resolve()),
                        "bytes": reused.stat().st_size,
                        "sha256": reused_sha256,
                    }
                ],
            )

            reused.write_text(reused.read_text() + "\n")
            changed_output = root / "changed-set.json"
            changed_arguments = [
                "build_teacher_cache_set.py",
                "--bound-verification",
                str(reused),
                reused_sha256,
                "--batch-group",
                str(plan),
                str(root),
                "new",
                "--expected-input",
                str(expected),
                "--teacher-revision",
                revision,
                "--teacher-provenance-sha256",
                provenance,
                "--output",
                str(changed_output),
                "--skip-artifact-rehash",
            ]
            with patch.object(sys, "argv", changed_arguments):
                with self.assertRaisesRegex(SystemExit, "bound verification changed"):
                    build_teacher_cache_set_main()

    def test_committed_general_purpose_evidence_covers_every_declared_domain(
        self,
    ) -> None:
        domain_rubric = json.loads((TRAINING / "DOMAIN_RUBRIC.json").read_text())
        language_rubric = json.loads((TRAINING / "LANGUAGE_RUBRIC.json").read_text())
        evidence = json.loads(
            (
                TRAINING.parent / "models/qwen38_27b/docs/DOMAIN_COVERAGE_V2.json"
            ).read_text()
        )
        expected_domains = set(domain_rubric["domains"])
        expected_languages = set(language_rubric["languages"])
        for partition in ("train", "evaluation"):
            observed = evidence[partition]
            primary = observed["primary_domains"]
            self.assertEqual(set(primary), expected_domains)
            self.assertEqual(sum(primary.values()), observed["records"])
            self.assertEqual(set(observed["languages"]), expected_languages)
            for domain, policy in domain_rubric["domains"].items():
                minimum = policy.get(
                    f"minimum_primary_{partition}",
                    domain_rubric["policy"][f"minimum_primary_{partition}"],
                )
                self.assertGreaterEqual(primary[domain], minimum, domain)
        family_gate = evidence["multilingual_joint_gate"]
        expected_families = set(domain_rubric["policy"]["required_families"])
        self.assertEqual(
            set(family_gate["train_non_english_family_counts"]), expected_families
        )
        self.assertEqual(
            set(family_gate["evaluation_non_english_family_counts"]),
            expected_families,
        )
        for family, minima in language_rubric[
            "aggregate_non_english_family_minima"
        ].items():
            self.assertGreaterEqual(
                family_gate["train_non_english_family_counts"][family],
                minima["train"],
            )
            self.assertGreaterEqual(
                family_gate["evaluation_non_english_family_counts"][family],
                minima["evaluation"],
            )

    def test_cohort_filter_rejects_empty_conditioning_and_duplicate_payloads(
        self,
    ) -> None:
        valid = self.recovery_record("valid", "Explain it", "A useful answer")
        duplicate = dict(valid, id="duplicate")
        empty = self.recovery_record("empty", "", "An unconditioned answer")
        tags = [{"id": record["id"]} for record in (valid, duplicate, empty)]
        kept, kept_tags, removed = filter_records([valid, duplicate, empty], tags)
        self.assertEqual([record["id"] for record in kept], ["valid"])
        self.assertEqual([tag["id"] for tag in kept_tags], ["valid"])
        self.assertEqual(
            removed,
            Counter({"duplicate_payload": 1, "empty_conditioning": 1}),
        )

    def test_cohort_filter_rejects_cross_partition_payload(self) -> None:
        record = self.recovery_record("evaluation", "Prompt", "Answer")
        digest = record["prompt_sha256"]
        kept, kept_tags, removed = filter_records(
            [record], [{"id": "evaluation"}], {digest}
        )
        self.assertEqual(kept, [])
        self.assertEqual(kept_tags, [])
        self.assertEqual(removed, Counter({"cross_partition_payload": 1}))

    class FakeRecoveryTensor:
        def __init__(self, shape, dtype="torch.float16"):
            self.shape = shape
            self.dtype = dtype

    class FakeRecovery:
        def __init__(self, metadata, tensors):
            self._metadata = metadata
            self._tensors = tensors

        def metadata(self):
            return self._metadata

        def keys(self):
            return self._tensors.keys()

        def get_tensor(self, name):
            return self._tensors[name]

    class WordTokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            return "\n".join(
                f"{message['role']}: {message['content']}" for message in messages
            )

        def __call__(self, text, **_kwargs):
            return type(
                "Encoded", (), {"input_ids": text.replace("\n", " \n ").split()}
            )()

    class FakePackedArtifact:
        def __init__(self, torch_module):
            self.torch = torch_module
            self.tensors = {
                "layer.weight": {"dtype": "q2_b64", "shape": [2, 64]},
                "layer.weight.s_in": {"dtype": "f16", "shape": [64]},
                "layer.weight.s_out": {"dtype": "f16", "shape": [2]},
            }

        def decode_float_tensor(self, name, _torch, device):
            return self.torch.ones(tuple(self.tensors[name]["shape"]), device=device)

    def test_corpus_percentiles_are_deterministic(self) -> None:
        self.assertEqual(percentile([9, 1, 5, 3], 0.0), 1)
        self.assertEqual(percentile([9, 1, 5, 3], 0.5), 5)
        self.assertEqual(percentile([9, 1, 5, 3], 1.0), 9)

    def test_recovery_order_and_completion_status_fail_closed(self) -> None:
        self.assertEqual(training_order(9, 2, 38), training_order(9, 2, 38))
        self.assertNotEqual(training_order(9, 2, 38), training_order(9, 3, 38))
        selected = resolve_sample_indices(
            ["gamma", "alpha", "beta"],
            ["beta", "alpha"],
        )
        self.assertEqual(selected, (1, 2))
        restricted = training_order(3, 2, 38, selected)
        self.assertEqual(set(restricted), {1, 2})
        self.assertEqual(len(restricted), 2)
        with self.assertRaisesRegex(ValueError, "absent"):
            resolve_sample_indices(["alpha"], ["missing"])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            training_order(3, 0, 38, [1, 1])
        self.assertEqual(
            recovery_training_status(False, None, 0),
            "complete",
        )
        self.assertEqual(
            recovery_training_status(False, 10, 0),
            "subset_run_complete",
        )
        self.assertEqual(
            recovery_training_status(False, None, 1),
            "partial_coverage",
        )
        self.assertEqual(
            recovery_training_status(True, None, 0),
            "bounded_run_complete",
        )

    def test_heldout_metric_aggregation_tracks_sample_and_target_means(self) -> None:
        aggregate = MetricAggregate()
        base = {
            "sequence_tokens": 10,
            "logit_targets": 2,
            "hidden_targets": 1,
            "mtp_targets": 2,
            "mtp_hidden_targets": 1,
            "losses": {
                "kl": 1.0,
                "ce": 2.0,
                "hidden": 3.0,
                "mtp_kl": 4.0,
                "mtp_ce": 5.0,
                "mtp_hidden": 6.0,
            },
        }
        aggregate.add(base)
        changed = json.loads(json.dumps(base))
        changed["logit_targets"] = 6
        changed["mtp_targets"] = 6
        changed["losses"]["kl"] = 3.0
        aggregate.add(changed)
        report = aggregate.report()
        self.assertEqual(report["sample_mean_losses"]["kl"], 2.0)
        self.assertEqual(report["target_weighted_mean_losses"]["kl"], 2.5)
        self.assertEqual(report["target_counts"]["kl"], 8)

    def test_logical_qcode_root_ignores_recovery_scales_but_binds_codes(self) -> None:
        manifest = {
            "tensors": [
                {
                    "name": "matrix.weight",
                    "dtype": "q2_b64",
                    "shape": [2, 64],
                    "sha256": "1" * 64,
                },
                {
                    "name": "matrix.weight.s_in",
                    "dtype": "f16",
                    "shape": [64],
                    "sha256": "2" * 64,
                },
            ]
        }
        expected = logical_qcode_root(manifest)
        manifest["tensors"][1]["sha256"] = "3" * 64
        self.assertEqual(logical_qcode_root(manifest), expected)
        manifest["tensors"][0]["sha256"] = "4" * 64
        self.assertNotEqual(logical_qcode_root(manifest), expected)

    def test_heldout_comparison_requires_same_codes_and_thirty_percent_closure(
        self,
    ) -> None:
        losses = {
            "kl": 1.0,
            "ce": 2.0,
            "hidden": 1.0,
            "mtp_kl": 1.0,
            "mtp_ce": 2.0,
            "mtp_hidden": 1.0,
        }
        aggregate = {
            "records": 1,
            "sequence_tokens": 10,
            "sample_mean_losses": losses,
            "target_weighted_mean_losses": losses,
            "target_counts": {name: 1 for name in losses},
        }
        identity = {
            "format": "ctox.recovery.heldout-evaluation.v1",
            "status": "complete",
            "model": "Qwen/Qwen3.8-27B",
            "revision": "r",
            "local_model_provenance_sha256": "p",
            "logical_qcode_root_sha256": "q",
            "teacher_cache_set_sha256": "c",
            "teacher_artifact_root_sha256": "t",
            "materialized_sha256": "m",
            "domain_tags_sha256": "d",
            "service_tags_sha256": "s",
            "prefill_chunk_tokens": 512,
            "compute_dtype": "bfloat16",
            "artifact_sha256": "a",
            "samples": [{"id": "sample"}],
            "aggregates": {
                "overall": aggregate,
                "groups": {"category": {"chat": aggregate}},
            },
        }
        direct = json.loads(json.dumps(identity))
        recovered = json.loads(json.dumps(identity))
        recovered["artifact_sha256"] = "b"
        for target in (
            recovered["aggregates"]["overall"],
            recovered["aggregates"]["groups"]["category"]["chat"],
        ):
            for name in ("kl", "hidden", "mtp_kl", "mtp_hidden"):
                target["target_weighted_mean_losses"][name] *= 0.6
        comparison = compare_reports(direct, recovered, 0.30)
        self.assertEqual(comparison["status"], "passed")
        recovered["logical_qcode_root_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "logical_qcode"):
            compare_reports(direct, recovered, 0.30)

    def test_evaluation_sidecars_require_exact_ids(self) -> None:
        require_exact_ids({"a", "b"}, {"a": {}, "b": {}}, "tags")
        with self.assertRaisesRegex(ValueError, "missing"):
            require_exact_ids({"a", "b"}, {"a": {}, "c": {}}, "tags")

    def test_partial_accumulation_is_rescaled_to_the_exact_mean(self) -> None:
        if torch is None:
            self.skipTest("torch is not installed in the host-only test environment")
        parameter = torch.nn.Parameter(torch.tensor([2.0]))
        ((parameter.square().sum()) / 4).backward()
        ((parameter.square().sum()) / 4).backward()
        factor = normalize_accumulated_gradients([parameter], 2, 4)
        self.assertEqual(factor, 2.0)
        self.assertTrue(torch.allclose(parameter.grad, torch.tensor([4.0])))

    def test_teacher_cache_resume_accepts_only_exact_source_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text(
                "".join(
                    json.dumps({"id": sample_id}) + "\n"
                    for sample_id in ("a", "b", "c")
                )
            )
            cache = root / "cache"
            cache.mkdir()
            (cache / "run.json").write_text(
                json.dumps({"start_sample": 0, "selected_samples": 3})
            )
            (cache / "index.jsonl").write_text(
                "".join(
                    json.dumps({"id": sample_id, "file": f"{sample_id}.safetensors"})
                    + "\n"
                    for sample_id in ("a", "b")
                )
            )
            for sample_id in ("a", "b"):
                (cache / f"{sample_id}.safetensors").write_bytes(b"sealed")
            run, entries = resume_prefix(cache, source, 0, 3)
            self.assertEqual(run["selected_samples"], 3)
            self.assertEqual([entry["id"] for entry in entries], ["a", "b"])
            (cache / "unindexed.safetensors").write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "unindexed"):
                resume_prefix(cache, source, 0, 3)

    def test_teacher_cache_recovers_exactly_one_fsynced_canonical_tail(self) -> None:
        class FakeSafeTensor:
            def __init__(self, path: Path) -> None:
                self.path = path

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                return None

            def metadata(self) -> dict[str, str]:
                return json.loads(self.path.read_text())

            def keys(self) -> list[str]:
                return ["topk_logits"]

        def fake_safe_open(path: Path, **_kwargs) -> FakeSafeTensor:
            return FakeSafeTensor(path)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text(
                "".join(
                    json.dumps(
                        {
                            "id": sample_id,
                            "prompt_sha256": f"sha-{sample_id}",
                        }
                    )
                    + "\n"
                    for sample_id in ("a", "b")
                )
            )
            cache = root / "cache"
            cache.mkdir()
            (cache / "index.jsonl").write_text(
                json.dumps({"id": "a", "file": "a.safetensors"}) + "\n"
            )
            (cache / "a.safetensors").write_bytes(b"indexed")
            (cache / "b.safetensors").write_text(
                json.dumps(
                    {
                        "sample_id": "b",
                        "source_payload_sha256": "sha-b",
                        "sequence_tokens": "32",
                        "logit_target_count": "8",
                        "hidden_target_count": "2",
                        "mtp_target_count": "4",
                        "mtp_hidden_target_count": "1",
                    }
                )
            )
            self.assertEqual(
                recover_unindexed_tail(cache, source, 0, 2, fake_safe_open),
                1,
            )
            entries = [
                json.loads(line)
                for line in (cache / "index.jsonl").read_text().splitlines()
            ]
            self.assertEqual([entry["id"] for entry in entries], ["a", "b"])
            self.assertEqual(entries[-1]["source_line"], 2)
            self.assertEqual(entries[-1]["mtp_targets"], 4)
            self.assertEqual(
                recover_unindexed_tail(cache, source, 0, 2, fake_safe_open),
                0,
            )

    def test_teacher_cache_recovery_rejects_temporary_or_noncanonical_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text(
                json.dumps({"id": "a", "prompt_sha256": "sha-a"}) + "\n"
            )
            cache = root / "cache"
            cache.mkdir()
            (cache / "index.jsonl").write_text("")
            temporary = cache / ".a.safetensors.tmp"
            temporary.write_bytes(b"partial")
            with self.assertRaisesRegex(ValueError, "temporary"):
                recover_unindexed_tail(cache, source, 0, 1, lambda *_a, **_k: None)
            temporary.unlink()
            (cache / "wrong.safetensors").write_bytes(b"complete")
            with self.assertRaisesRegex(ValueError, "next source sample"):
                recover_unindexed_tail(cache, source, 0, 1, lambda *_a, **_k: None)

    def test_teacher_cache_sample_save_is_atomic_and_refuses_overwrite(self) -> None:
        writes: list[tuple[Path, dict[str, str]]] = []

        def fake_save_file(_tensors, path: Path, metadata: dict[str, str]) -> None:
            writes.append((path, metadata))
            path.write_bytes(b"complete")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_sample_atomic(
                fake_save_file,
                {"tensor": object()},
                output,
                "a.safetensors",
                {"sample_id": "a"},
            )
            self.assertEqual((output / "a.safetensors").read_bytes(), b"complete")
            self.assertFalse((output / ".a.safetensors.tmp").exists())
            self.assertEqual(writes[0][0].name, ".a.safetensors.tmp")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                save_sample_atomic(
                    fake_save_file,
                    {},
                    output,
                    "a.safetensors",
                    {"sample_id": "a"},
                )

    def test_domain_classifier_preserves_hard_capability_signals(self) -> None:
        record = {
            "category": "code",
            "tools": [{"type": "function"}],
            "messages": [
                {"role": "user", "content": "Debug this program"},
                {"role": "assistant", "content": "```rust\nfn main() {}\n```"},
            ],
        }
        self.assertEqual(
            deterministic_labels(record, record["messages"][-1]["content"]),
            {
                "agentic_tools_workflows",
                "data_analysis_statistics_structured",
                "software_development",
            },
        )
        self.assertIn("user: Debug this program", classification_text(record))
        self.assertEqual(deterministic_primary_label(record), "agentic_tools_workflows")
        self.assertEqual(
            deterministic_primary_label({"category": "code", "split": "code"}),
            "software_development",
        )
        self.assertEqual(
            deterministic_primary_label({"category": "math", "split": "stem"}),
            None,
        )

    def test_source_primary_override_preserves_classifier_score_evidence(self) -> None:
        records = [{"id": "code", "category": "code", "split": "code"}]
        tags = [
            {
                "id": "code",
                "labels": ["politics_civics_institutions"],
                "primary_label": "politics_civics_institutions",
                "scores": {
                    "politics_civics_institutions": 0.91,
                    "software_development": 0.63,
                },
            }
        ]
        corrected, counts = apply_overrides(records, tags)
        self.assertEqual(corrected[0]["primary_label"], "software_development")
        self.assertEqual(
            corrected[0]["classifier_primary_label"],
            "politics_civics_institutions",
        )
        self.assertEqual(corrected[0]["primary_confidence"], 1.0)
        self.assertIn("software_development", corrected[0]["labels"])
        self.assertEqual(counts, Counter({"software_development": 1}))

    def test_language_for_canonicalizes_common_iso_639_3_codes(self) -> None:
        self.assertEqual(language_for({"language_code": "eng"}, "und"), "en")
        self.assertEqual(language_for({"language": "deu"}, "und"), "de")
        self.assertEqual(language_for({}, "eng"), "en")

    def test_release_domain_rubric_covers_every_required_family(self) -> None:
        rubric = json.loads((TRAINING / "DOMAIN_RUBRIC.json").read_text())
        language_rubric = json.loads((TRAINING / "LANGUAGE_RUBRIC.json").read_text())
        validate_rubric(rubric)
        validate_language_rubric(language_rubric, rubric)
        self.assertEqual(rubric["format"], "ctox.recovery-domain-rubric.v2")
        self.assertGreaterEqual(len(rubric["domains"]), 36)
        self.assertEqual(
            {domain["family"] for domain in rubric["domains"].values()},
            set(rubric["policy"]["required_families"]),
        )

    def test_service_mode_rubric_is_joint_with_domains_and_languages(self) -> None:
        domain_rubric = json.loads((TRAINING / "DOMAIN_RUBRIC.json").read_text())
        language_rubric = json.loads((TRAINING / "LANGUAGE_RUBRIC.json").read_text())
        service_rubric = json.loads((TRAINING / "SERVICE_MODE_RUBRIC.json").read_text())
        validate_service_rubric(service_rubric, domain_rubric, language_rubric)
        self.assertGreaterEqual(len(service_rubric["modes"]), 14)
        self.assertEqual(
            set(service_rubric["language_minimum_distinct_modes"]),
            set(language_rubric["languages"]),
        )

    def test_deterministic_service_modes_preserve_hard_record_facts(self) -> None:
        record = {
            "category": "agentic",
            "split": "code",
            "tools": [{"type": "function"}],
            "messages": [
                {"role": "user", "content": "Calculate and call the tool"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"name": "calculate"}],
                },
                {"role": "tool", "content": "4"},
                {"role": "assistant", "content": '{"answer": 4}'},
            ],
        }
        self.assertEqual(
            deterministic_modes(record),
            {
                "coding_debugging",
                "multi_turn_dialogue",
                "structured_output_constraints",
                "tool_calling_agentic",
            },
        )

    def test_service_matrix_fails_closed_on_domain_and_pair_gaps(self) -> None:
        domain_rubric = {
            "policy": {
                "required_families": ["work"],
                "minimum_confidence": 0.7,
                "minimum_primary_train": 1,
                "minimum_primary_evaluation": 1,
            },
            "domains": {
                "software": {
                    "family": "work",
                    "description": "software",
                    "minimum_train": 1,
                    "minimum_evaluation": 1,
                },
                "writing": {
                    "family": "work",
                    "description": "writing",
                    "minimum_train": 1,
                    "minimum_evaluation": 1,
                },
            },
        }
        language_rubric = {"languages": {"en": {}}}
        service_rubric = {
            "format": "ctox.recovery-service-mode-rubric.v1",
            "policy": {
                "minimum_confidence": 0.7,
                "minimum_distinct_modes_per_domain_train": 2,
                "minimum_distinct_modes_per_domain_evaluation": 1,
                "minimum_distinct_modes_per_family_train": 2,
                "minimum_distinct_modes_per_family_evaluation": 1,
                "all_declared_domains_required": True,
            },
            "modes": {
                "code": {
                    "description": "code",
                    "minimum_train": 1,
                    "minimum_evaluation": 1,
                },
                "explain": {
                    "description": "explain",
                    "minimum_train": 1,
                    "minimum_evaluation": 1,
                },
            },
            "language_minimum_distinct_modes": {"en": {"train": 2, "evaluation": 1}},
            "required_domain_mode_pairs": {
                "software": {"code": {"train": 1, "evaluation": 1}}
            },
        }
        records = [{"id": "a", "language": "en"}]
        report = service_coverage_report(
            records,
            {"a": {"primary_label": "software"}},
            {"a": {"labels": ["explain"]}},
            domain_rubric,
            language_rubric,
            service_rubric,
            "train",
        )
        self.assertEqual(report["status"], "supplement_required")
        self.assertIn("writing", report["domain_presence_gaps"])
        self.assertIn("software", report["domain_mode_diversity_gaps"])
        self.assertEqual(
            report["required_domain_mode_pair_gaps"]["software"]["code"],
            {"observed": 0, "required": 1},
        )

    def test_service_supplement_closes_only_real_matrix_gaps(self) -> None:
        domain_rubric = {
            "policy": {
                "required_families": ["work"],
                "minimum_confidence": 0.7,
                "minimum_primary_train": 1,
                "minimum_primary_evaluation": 1,
            },
            "domains": {
                "software": {
                    "family": "work",
                    "description": "software",
                    "minimum_train": 1,
                    "minimum_evaluation": 1,
                }
            },
        }
        service_rubric = {
            "format": "ctox.recovery-service-mode-rubric.v1",
            "policy": {
                "minimum_confidence": 0.7,
                "minimum_distinct_modes_per_domain_train": 2,
                "minimum_distinct_modes_per_domain_evaluation": 1,
                "minimum_distinct_modes_per_family_train": 2,
                "minimum_distinct_modes_per_family_evaluation": 1,
                "all_declared_domains_required": True,
            },
            "modes": {
                "code": {
                    "description": "code",
                    "minimum_train": 2,
                    "minimum_evaluation": 1,
                },
                "explain": {
                    "description": "explain",
                    "minimum_train": 1,
                    "minimum_evaluation": 1,
                },
            },
            "language_minimum_distinct_modes": {"en": {"train": 2, "evaluation": 1}},
            "required_domain_mode_pairs": {
                "software": {"code": {"train": 2, "evaluation": 1}}
            },
        }
        baseline = [{"id": "base", "language": "en"}]
        language_rubric = {"languages": {"en": {}}}
        baseline_domains = {"base": {"primary_label": "software"}}
        baseline_services = {"base": {"labels": ["code"]}}
        candidates = [
            {"id": "expensive", "language": "en"},
            {"id": "mixed", "language": "en"},
        ]
        candidate_domains = {
            sample_id: {"primary_label": "software"}
            for sample_id in ("expensive", "mixed")
        }
        candidate_services = {
            "expensive": {"labels": ["explain"]},
            "mixed": {"labels": ["code", "explain"]},
        }
        selected, evidence = select_service_supplement(
            baseline,
            baseline_domains,
            baseline_services,
            candidates,
            candidate_domains,
            candidate_services,
            {"expensive": 1, "mixed": 20},
            domain_rubric,
            language_rubric,
            service_rubric,
            "train",
        )
        self.assertEqual([record["id"] for record in selected], ["mixed"])
        self.assertTrue(
            all(not values for values in evidence["remaining_requirements"].values())
        )

    def test_primary_quota_can_be_stricter_for_major_domains(self) -> None:
        rubric = {
            "policy": {
                "minimum_primary_train": 1,
            },
            "domains": {
                "general": {
                    "minimum_train": 1,
                    "minimum_primary_train": 3,
                },
                "niche": {"minimum_train": 1},
            },
        }
        gaps, primary_gaps = quota_gaps(
            Counter({"general": 5, "niche": 5}),
            Counter({"general": 2, "niche": 1}),
            rubric,
            "train",
        )
        self.assertEqual(gaps, {})
        self.assertEqual(primary_gaps, {"general": {"observed": 2, "required": 3}})

    def test_joint_language_gate_rejects_translation_only_coverage(self) -> None:
        domain_rubric = {
            "policy": {
                "required_families": ["language", "software"],
                "minimum_confidence": 0.7,
                "minimum_primary_train": 1,
                "minimum_primary_evaluation": 1,
            },
            "domains": {
                "translation": {
                    "family": "language",
                    "description": "translation",
                    "minimum_train": 1,
                    "minimum_evaluation": 1,
                },
                "code": {
                    "family": "software",
                    "description": "software",
                    "minimum_train": 1,
                    "minimum_evaluation": 1,
                },
            },
        }
        language_rubric = {
            "translation_domain": "translation",
            "languages": {
                "en": {
                    "minimum_train": 1,
                    "minimum_evaluation": 1,
                    "minimum_primary_domains_train": 1,
                    "minimum_primary_domains_evaluation": 1,
                    "minimum_non_translation_train": 1,
                    "minimum_non_translation_evaluation": 1,
                },
                "de": {
                    "minimum_train": 1,
                    "minimum_evaluation": 1,
                    "minimum_primary_domains_train": 1,
                    "minimum_primary_domains_evaluation": 1,
                    "minimum_non_translation_train": 1,
                    "minimum_non_translation_evaluation": 1,
                },
            },
            "aggregate_non_english_family_minima": {
                "language": {"train": 1, "evaluation": 1},
                "software": {"train": 1, "evaluation": 1},
            },
        }
        records = [{"id": "a", "language": "en"}, {"id": "b", "language": "de"}]
        tags = {
            "a": {"primary_label": "code"},
            "b": {"primary_label": "translation"},
        }
        report = coverage_report(records, tags, domain_rubric, language_rubric, "train")
        self.assertEqual(report["status"], "supplement_required")
        self.assertEqual(report["non_translation_gaps"]["de"]["observed"], 0)
        self.assertEqual(
            report["aggregate_non_english_family_gaps"]["software"]["observed"],
            0,
        )

    def test_joint_supplement_closes_domain_and_multilingual_gaps(self) -> None:
        domain_rubric = {
            "policy": {
                "required_families": ["language", "software"],
                "minimum_confidence": 0.7,
                "minimum_primary_train": 1,
                "minimum_primary_evaluation": 1,
            },
            "domains": {
                "translation": {
                    "family": "language",
                    "description": "translation",
                    "minimum_train": 1,
                    "minimum_evaluation": 1,
                },
                "code": {
                    "family": "software",
                    "description": "software",
                    "minimum_train": 1,
                    "minimum_evaluation": 1,
                },
            },
        }
        language_rubric = {
            "translation_domain": "translation",
            "languages": {
                language: {
                    "minimum_train": 1,
                    "minimum_evaluation": 1,
                    "minimum_primary_domains_train": 1,
                    "minimum_primary_domains_evaluation": 1,
                    "minimum_non_translation_train": 1,
                    "minimum_non_translation_evaluation": 1,
                }
                for language in ("en", "de")
            },
            "aggregate_non_english_family_minima": {
                "language": {"train": 1, "evaluation": 1},
                "software": {"train": 1, "evaluation": 1},
            },
        }
        baseline = [
            {"id": "base-en", "language": "en"},
            {"id": "base-de", "language": "de"},
        ]
        baseline_tags = {
            sample_id: {
                "primary_label": "translation",
                "labels": ["translation"],
                "scores": {"translation": 0.95},
            }
            for sample_id in ("base-en", "base-de")
        }
        candidates = [
            {"id": "code-en", "language": "en"},
            {"id": "code-de", "language": "de"},
        ]
        candidate_tags = {
            sample_id: {
                "id": sample_id,
                "primary_label": "code",
                "labels": ["code"],
                "scores": {"code": 0.9},
            }
            for sample_id in ("code-en", "code-de")
        }
        selected, selected_tags, evidence = select_joint_supplement(
            baseline,
            baseline_tags,
            candidates,
            candidate_tags,
            {"code-en": 10, "code-de": 12},
            domain_rubric,
            language_rubric,
            "train",
            domain_margin=0,
            minimum_confidence=0.8,
            primary_tie_tolerance=0.02,
        )
        self.assertEqual({record["id"] for record in selected}, {"code-en", "code-de"})
        self.assertEqual({tag["id"] for tag in selected_tags}, {"code-en", "code-de"})
        self.assertTrue(
            all(not values for values in evidence["remaining_requirements"].values())
        )

    def test_near_tie_domain_assignment_is_explicit_and_bounded(self) -> None:
        tag = {
            "id": "bio",
            "primary_label": "medicine",
            "scores": {"medicine": 0.91, "biology": 0.90, "chemistry": 0.70},
        }
        requirements = {"domain_primary": Counter({"biology": 2})}
        assignment = primary_assignment(tag, requirements, 0.7, 0.02)
        self.assertEqual(assignment, "biology")
        effective = assigned_tag(tag, assignment)
        self.assertEqual(effective["primary_source"], "near_tie_coverage_assignment")
        self.assertAlmostEqual(effective["primary_margin_from_classifier_max"], 0.01)
        self.assertIsNone(primary_assignment(tag, requirements, 0.7, 0.005))

    def test_unassigned_candidate_keeps_original_primary_for_language_gate(
        self,
    ) -> None:
        requirements = {
            name: Counter()
            for name in (
                "domain_label",
                "domain_primary",
                "language",
                "non_translation",
                "language_diversity",
                "non_english_family",
            )
        }
        requirements["non_english_family"]["science"] = 1
        coverage = candidate_coverage(
            {"language": "de"},
            {"primary_label": "biology", "labels": ["biology"]},
            None,
            requirements,
            {"de": {"biology"}},
            {"domains": {"biology": {"family": "science"}}},
            {"translation_domain": "translation"},
        )
        self.assertEqual(coverage, [("non_english_family", "science")])

    def test_domain_gate_requires_clear_primary_examples_not_only_multilabel_hits(
        self,
    ) -> None:
        rubric = {
            "policy": {"minimum_primary_train": 2},
            "domains": {
                "a": {"minimum_train": 2},
                "b": {"minimum_train": 2},
            },
        }
        gaps, primary_gaps = quota_gaps(
            Counter({"a": 3, "b": 3}),
            Counter({"a": 3, "b": 1}),
            rubric,
            "train",
        )
        self.assertEqual(gaps, {})
        self.assertEqual(primary_gaps, {"b": {"observed": 1, "required": 2}})

    def test_primary_supplement_closes_each_gap_with_margin_and_confidence(
        self,
    ) -> None:
        records = [{"id": value} for value in ("a", "b", "c", "d", "e")]
        tags = {
            "a": {"primary_label": "safety", "scores": {"safety": 0.95}},
            "b": {"primary_label": "safety", "scores": {"safety": 0.90}},
            "c": {"primary_label": "safety", "scores": {"safety": 0.79}},
            "d": {"primary_label": "social", "scores": {"social": 0.99}},
            "e": {"primary_label": "social", "scores": {"social": 0.90}},
        }
        selected, domain_samples = select_supplement(
            records,
            tags,
            {
                "safety": {"observed": 1, "required": 2},
                "social": {"observed": 0, "required": 1},
            },
            {"a": 20, "b": 10, "c": 1, "d": 10, "e": 5},
            margin=1,
            minimum_confidence=0.8,
        )
        self.assertEqual({record["id"] for record in selected}, {"a", "b", "d", "e"})
        self.assertEqual(domain_samples["safety"], ["a", "b"])

    def test_hash_covers_reference_answer(self) -> None:
        first = {"input": "2+2?", "output": "4"}
        second = {"input": "2+2?", "output": "5"}
        self.assertNotEqual(canonical_text(first), canonical_text(second))

    def test_domain_tag_shards_follow_materialized_order(self) -> None:
        records = [{"id": "b"}, {"id": "a"}, {"id": "c"}]
        shards = [
            [{"id": "a", "primary_label": "x"}, {"id": "c", "primary_label": "y"}],
            [{"id": "b", "primary_label": "z"}],
        ]
        self.assertEqual(
            [tag["id"] for tag in merge_ordered_tags(records, shards)],
            ["b", "a", "c"],
        )
        with self.assertRaisesRegex(ValueError, "domain tag a"):
            merge_ordered_tags(records, shards + [[{"id": "a"}]])

    def test_hash_covers_tool_schema(self) -> None:
        first = {
            "messages": [{"role": "user", "content": "Check Berlin"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }
        second = json.loads(json.dumps(first))
        second["tools"][0]["function"]["name"] = "forecast"
        self.assertNotEqual(canonical_text(first), canonical_text(second))
        self.assertEqual(recovery_payload(first)["tools"], first["tools"])

    def test_agentic_source_defaults_are_pinned(self) -> None:
        source = SOURCES["nemotron-sft-agentic-v2"]
        self.assertEqual(len(source["reviewed_revision"]), 40)
        self.assertEqual(
            source["default_splits"],
            ("interactive_agent", "search", "tool_calling"),
        )
        self.assertIn("cc-by-4.0", source["allowed_licenses"])
        self.assertTrue(source["raw_jsonl"])
        self.assertTrue(source_uses_raw_jsonl(source["repo"]))
        self.assertFalse(source_uses_raw_jsonl(SOURCES["nemotron-v1"]["repo"]))

    def test_aya_source_and_language_code_are_pinned(self) -> None:
        source = SOURCES["aya"]
        self.assertEqual(source["repo"], "CohereLabs/aya_dataset")
        self.assertEqual(len(source["reviewed_revision"]), 40)
        self.assertEqual(source["default_subsets"], ("default",))
        self.assertEqual(source["allowed_licenses"], ("apache-2.0",))
        self.assertEqual(
            language_for({"language": "French", "language_code": "FRA"}, "und"),
            "fra",
        )
        self.assertEqual(language_for({}, "und"), "und")
        self.assertEqual(language_for({"language": "und"}, "en"), "en")
        self.assertEqual(
            recovery_payload({"inputs": "Bonjour", "targets": "Salut"}),
            {
                "messages": [
                    {"role": "user", "content": "Bonjour"},
                    {"role": "assistant", "content": "Salut"},
                ]
            },
        )

    def test_ultrachat_source_is_pinned_and_release_eligible(self) -> None:
        source = SOURCES["ultrachat"]
        self.assertEqual(source["repo"], "HuggingFaceH4/ultrachat_200k")
        self.assertEqual(
            source["reviewed_revision"],
            "8049631c405ae6576f93f445c6b8166f76f5505a",
        )
        self.assertEqual(source["allowed_licenses"], ("mit",))
        self.assertTrue(source["release_eligible"])

    def test_nested_metadata_supplies_stable_source_id(self) -> None:
        row = {"metadata": {"uuid": "stable-row-id"}}
        self.assertEqual(source_id_for(row, 17), "stable-row-id")
        encoded = {"metadata": json.dumps({"uuid": "encoded-row-id"})}
        self.assertEqual(source_id_for(encoded, 17), "encoded-row-id")
        self.assertEqual(
            source_id_for({"id": "sample-id", "source_id": "source-coordinate"}, 17),
            "source-coordinate",
        )

    def test_license_ids_normalize_and_drop_placeholders(self) -> None:
        self.assertEqual(
            license_ids(["CC BY 4.0", "apache-2.0", "dataset-card", "CC BY 4.0"]),
            ["apache-2.0", "cc-by-4.0"],
        )

    def test_long_context_generator_is_sized_and_requires_two_retrieval_positions(
        self,
    ) -> None:
        manifest, materialized = generated_record(
            self.WordTokenizer(),
            seed="calibration-v1",
            target_tokens=4096,
            language="de",
            sample_index=1,
            tolerance=128,
            source_revision="a" * 40,
            split="calibration",
        )
        self.assertGreaterEqual(manifest["rendered_tokens"], 4096 - 128)
        self.assertLessEqual(manifest["rendered_tokens"], 4096)
        self.assertEqual(manifest["release_eligible"], True)
        positions = manifest["marker_normalized_positions"]
        self.assertEqual(len(positions), 2)
        self.assertGreater(abs(positions[0] - positions[1]), 0.25)
        self.assertEqual(
            hashlib.sha256(canonical_text(materialized).encode()).hexdigest(),
            manifest["prompt_sha256"],
        )

    def test_sparse_teacher_positions_cover_answer_needles_and_sequence(self) -> None:
        logits, hidden = position_sets(
            sequence_length=1000,
            target_mode="assistant",
            assistant_prefix_tokens=950,
            marker_offsets=[100, 800],
            marker_window=2,
            uniform_hidden_positions=5,
        )
        self.assertEqual(logits[0], 949)
        self.assertEqual(logits[-1], 998)
        self.assertTrue({98, 99, 100, 101, 102}.issubset(hidden))
        self.assertTrue({798, 799, 800, 801, 802}.issubset(hidden))
        self.assertTrue(set(logits).issubset(hidden))
        self.assertIn(0, hidden)
        self.assertIn(999, hidden)

    def test_assistant_hidden_targets_are_bounded_without_dropping_logits(self) -> None:
        logits, hidden = position_sets(
            sequence_length=1000,
            target_mode="assistant",
            assistant_prefix_tokens=100,
            marker_offsets=[500],
            marker_window=2,
            uniform_hidden_positions=5,
            assistant_hidden_positions=8,
        )
        self.assertEqual(len(logits), 900)
        self.assertLessEqual(len(hidden), 8 + 5 + 5)
        self.assertEqual(logits[0], 99)
        self.assertEqual(logits[-1], 998)
        self.assertIn(500, hidden)

    def test_teacher_all_mode_preserves_full_sequence_targets(self) -> None:
        logits, hidden = position_sets(8, "all", None, [], 0, 0)
        self.assertEqual(logits, list(range(8)))
        self.assertEqual(hidden, list(range(8)))

    def test_mtp_targets_exclude_only_unverifiable_final_draft(self) -> None:
        self.assertEqual(mtp_target_positions(8, [2, 3, 4, 5, 6]), [2, 3, 4, 5])
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            mtp_target_positions(8, [3, 2])
        with self.assertRaisesRegex(ValueError, "outside"):
            mtp_target_positions(8, [7])

    def test_teacher_cache_plan_counts_every_persisted_tensor_family(self) -> None:
        values = sample_tensor_bytes(
            sequence_tokens=100,
            logit_targets=20,
            hidden_targets=10,
            mtp_targets=19,
            mtp_hidden_targets=7,
            hidden_size=8,
            hidden_layer_count=5,
            top_k=4,
        )
        self.assertEqual(values["input_ids"], 400)
        self.assertEqual(values["attention_mask"], 100)
        self.assertEqual(values["topk_indices"], 320)
        self.assertEqual(values["topk_logprobs"], 160)
        self.assertEqual(values["hidden_layers"], 800)
        self.assertEqual(values["mtp_hidden"], 112)
        self.assertEqual(values["mtp_topk_indices"], 304)
        self.assertEqual(values["mtp_topk_logprobs"], 152)
        self.assertEqual(values["mtp_residual_probability"], 76)
        self.assertEqual(
            set(values),
            {
                "input_ids",
                "attention_mask",
                "logit_positions",
                "hidden_positions",
                "topk_indices",
                "topk_logprobs",
                "residual_probability",
                "hidden_layers",
                "mtp_positions",
                "mtp_hidden_positions",
                "mtp_hidden",
                "mtp_topk_indices",
                "mtp_topk_logprobs",
                "mtp_residual_probability",
            },
        )

    def test_teacher_cache_verifier_contract_includes_mtp_and_hidden_layers(
        self,
    ) -> None:
        specs = expected_tensor_specs(100, 20, 10, 19, 7, 64, 5120, [0, 63], True)
        self.assertEqual(specs["input_ids"], ("I32", [1, 100]))
        self.assertEqual(specs["topk_logprobs"], ("BF16", [1, 20, 64]))
        self.assertEqual(specs["hidden_63"], ("BF16", [1, 10, 5120]))
        self.assertEqual(specs["mtp_hidden"], ("BF16", [1, 7, 5120]))
        self.assertEqual(specs["mtp_topk_indices"], ("I32", [1, 19, 64]))

    def test_teacher_smoke_selects_every_domain_and_language_deterministically(
        self,
    ) -> None:
        records = [
            {"id": "a", "language": "de"},
            {"id": "b", "language": "en"},
            {"id": "c", "language": "ja"},
        ]
        tags = {
            "a": {"primary_label": "everyday"},
            "b": {"primary_label": "software"},
            "c": {"primary_label": "software"},
        }
        plans = {
            "a": {"sequence_tokens": 100, "projected_file_bytes": 200},
            "b": {"sequence_tokens": 90, "projected_file_bytes": 180},
            "c": {"sequence_tokens": 80, "projected_file_bytes": 160},
        }
        selected, domains, languages = select_teacher_smoke_ids(
            records,
            tags,
            plans,
            ["everyday", "software"],
            ["de", "en", "ja"],
            128,
        )
        self.assertEqual(selected, {"a", "b", "c"})
        self.assertEqual(domains, {"everyday": "a", "software": "c"})
        self.assertEqual(languages, {"de": "a", "en": "b", "ja": "c"})

    def test_teacher_batches_are_contiguous_and_bound_all_resources(self) -> None:
        samples = [
            {
                "id": str(index),
                "source_line": index,
                "sequence_tokens": tokens,
                "projected_file_bytes": tokens * 10,
            }
            for index, tokens in enumerate([40, 50, 60, 30], 1)
        ]
        result = plan_teacher_batches(samples, 3, 100, 1000)
        self.assertEqual([batch["start_sample"] for batch in result], [0, 2])
        self.assertEqual([batch["samples"] for batch in result], [2, 2])
        self.assertEqual([batch["sequence_tokens"] for batch in result], [90, 90])
        self.assertEqual(sum(batch["projected_cache_bytes"] for batch in result), 1800)

    def test_teacher_batch_runner_skips_only_matching_verified_work(self) -> None:
        batch = {"start_sample": 128, "samples": 32}
        run = {
            "teacher_revision": "r",
            "local_model_provenance_sha256": "p",
            "start_sample": 128,
            "selected_samples": 32,
            "written_samples": 32,
        }
        verification = {
            "status": "passed",
            "teacher_revision": "r",
            "teacher_provenance_sha256": "p",
            "samples": 32,
        }
        self.assertTrue(completed_batch_matches(run, verification, batch, "r", "p"))
        verification["samples"] = 31
        self.assertFalse(completed_batch_matches(run, verification, batch, "r", "p"))

    def test_long_context_batches_select_the_declared_lower_weight_tier(self) -> None:
        self.assertEqual(gpu_weight_memory_for_batch(16, 14, 65_536, 65_535), 16)
        self.assertEqual(gpu_weight_memory_for_batch(16, 14, 65_536, 65_536), 14)
        self.assertEqual(gpu_weight_memory_for_batch(16, None, 65_536, 131_072), 16)
        for values in [
            (0, 14, 65_536, 131_072),
            (16, 17, 65_536, 131_072),
            (16, 14, 0, 131_072),
            (16, 14, 65_536, 0),
        ]:
            with self.assertRaises(ValueError):
                gpu_weight_memory_for_batch(*values)

    def test_sparse_teacher_losses_preserve_topk_and_residual_semantics(self) -> None:
        if torch is None:
            self.skipTest("torch is not installed in the host-only test environment")

        probabilities = torch.tensor([[[0.5, 0.3, 0.2]]])
        logits = probabilities.log()
        indices = torch.tensor([[[0, 1]]], dtype=torch.int32)
        teacher_logprobs = probabilities[:, :, :2].log().to(torch.bfloat16)
        residual = torch.tensor([[0.2]], dtype=torch.float32)
        self.assertLess(
            float(sparse_teacher_kl(logits, indices, teacher_logprobs, residual)),
            2e-3,
        )
        changed = torch.tensor([[[0.2, 0.3, 0.5]]]).log()
        self.assertGreater(
            float(sparse_teacher_kl(changed, indices, teacher_logprobs, residual)),
            0.1,
        )

    def test_supervised_and_hidden_recovery_losses_use_recorded_positions(self) -> None:
        if torch is None:
            self.skipTest("torch is not installed in the host-only test environment")

        input_ids = torch.tensor([[4, 2, 1]])
        logits = torch.full((1, 2, 5), -10.0)
        logits[0, 0, 2] = 10.0
        logits[0, 1, 1] = 10.0
        self.assertLess(
            float(supervised_next_token_loss(logits, input_ids, torch.tensor([0, 1]))),
            1e-6,
        )
        hidden = torch.randn(1, 3, 8)
        self.assertLess(float(normalized_hidden_loss(hidden, hidden)), 1e-6)
        mtp_logits = torch.full((1, 1, 5), -10.0)
        mtp_logits[0, 0, 1] = 10.0
        self.assertLess(
            float(supervised_mtp_token_loss(mtp_logits, input_ids, torch.tensor([0]))),
            1e-6,
        )

    def test_chunked_hidden_contributions_match_full_loss_and_gradient(self) -> None:
        if torch is None:
            self.skipTest("torch is not installed in the host-only test environment")
        torch.manual_seed(38)
        teacher = torch.randn(1, 7, 5)
        full_student = torch.randn(1, 7, 5, requires_grad=True)
        full_loss = normalized_hidden_loss(full_student, teacher)
        full_loss.backward()
        expected_gradient = full_student.grad.clone()

        chunked_student = full_student.detach().clone().requires_grad_(True)
        teacher_square_sum = teacher.float().square().sum()
        contributions = []
        for start, stop in ((0, 3), (3, 5), (5, 7)):
            contribution = normalized_hidden_loss_contribution(
                chunked_student[:, start:stop],
                teacher[:, start:stop],
                teacher_square_sum,
                total_vectors=7,
            )
            contribution.backward()
            contributions.append(contribution.detach())
        self.assertTrue(
            torch.allclose(torch.stack(contributions).sum(), full_loss.detach(), atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(chunked_student.grad, expected_gradient, atol=1e-6)
        )

    def test_end_to_end_recovery_objective_includes_every_base_and_mtp_family(
        self,
    ) -> None:
        if torch is None:
            self.skipTest("torch is not installed in the host-only test environment")
        probabilities = torch.tensor([[[0.5, 0.3, 0.2]]])
        teacher = {
            "input_ids": torch.tensor([[2, 0, 1]]),
            "logit_positions": torch.tensor([0]),
            "topk_indices": torch.tensor([[[0, 1]]]),
            "topk_logprobs": probabilities[:, :, :2].log().to(torch.bfloat16),
            "residual_probability": torch.tensor([[0.2]]),
            "hidden_0": torch.ones(1, 1, 4),
            "mtp_positions": torch.tensor([0]),
            "mtp_topk_indices": torch.tensor([[[0, 1]]]),
            "mtp_topk_logprobs": probabilities[:, :, :2].log().to(torch.bfloat16),
            "mtp_residual_probability": torch.tensor([[0.2]]),
            "mtp_hidden": torch.ones(1, 1, 4),
        }
        student = {
            "logits": probabilities.log(),
            "hidden_0": teacher["hidden_0"].clone(),
            "mtp_logits": probabilities.log(),
            "mtp_hidden": teacher["mtp_hidden"].clone(),
        }
        total, losses = end_to_end_recovery_loss(student, teacher, [0])
        self.assertEqual(
            set(losses), {"kl", "ce", "hidden", "mtp_kl", "mtp_ce", "mtp_hidden"}
        )
        self.assertTrue(torch.isfinite(total))

    def test_streamed_sparse_projection_matches_dense_losses_and_gradients(
        self,
    ) -> None:
        if torch is None:
            self.skipTest("torch is not installed in the host-only test environment")
        torch.manual_seed(7)
        hidden = torch.randn(1, 5, 4, requires_grad=True)
        head = torch.nn.Linear(4, 7, bias=False)
        input_ids = torch.tensor([[0, 2, 4, 1, 3, 5]])
        positions = torch.tensor([0, 1, 2, 3, 4])
        with torch.no_grad():
            teacher_logits = head(hidden.detach())
            teacher_logprobs = teacher_logits.log_softmax(dim=-1)
            values, indices = teacher_logprobs.topk(3, dim=-1)
            residual = 1.0 - values.exp().sum(dim=-1)
        streamed_kl, streamed_ce = streamed_sparse_target_losses(
            head,
            hidden,
            indices,
            values,
            residual,
            input_ids,
            positions,
            target_offset=1,
            chunk_size=2,
        )
        dense_logits = head(hidden)
        dense_kl = sparse_teacher_kl(dense_logits, indices, values, residual)
        dense_ce = supervised_next_token_loss(dense_logits, input_ids, positions)
        self.assertTrue(torch.allclose(streamed_kl, dense_kl, atol=1e-6))
        self.assertTrue(torch.allclose(streamed_ce, dense_ce, atol=1e-6))
        (streamed_kl + streamed_ce).backward()
        streamed_gradient = hidden.grad.clone()
        hidden.grad = None
        (dense_kl + dense_ce).backward()
        self.assertTrue(torch.allclose(streamed_gradient, hidden.grad, atol=1e-6))

    def test_verified_teacher_cache_rejects_duplicate_or_changed_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            cache.mkdir()
            sample_id = "a" * 64
            artifact = cache / f"{sample_id}.safetensors"
            artifact.write_bytes(b"fixed")
            item = {
                "id": sample_id,
                "file": artifact.name,
                "bytes": 5,
                "sha256": hashlib.sha256(b"fixed").hexdigest(),
            }
            verification = root / "verification.json"
            verification.write_text(
                json.dumps(
                    {
                        "format": "ctox.teacher-cache-verification.v1",
                        "status": "passed",
                        "teacher_revision": "r",
                        "teacher_provenance_sha256": "p",
                        "hidden_layers": [0, 15, 31, 47, 63],
                        "hidden_size": 5120,
                        "top_k": 64,
                        "mtp_targets": True,
                        "cache": str(cache),
                        "samples": 1,
                        "artifact_bytes": 5,
                        "artifact_root_sha256": "z",
                        "artifacts": [item],
                    }
                ),
                encoding="utf-8",
            )
            dataset = VerifiedTeacherCache([verification], "r", "p")
            self.assertEqual(dataset.verified_artifact_path(0), artifact.resolve())
            self.assertEqual(dataset.manifest()["samples"], 1)
            manifest = dataset.manifest()
            manifest_path = root / "set.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            loaded = VerifiedTeacherCache.from_manifest(manifest_path, manifest_sha256)
            self.assertEqual(
                loaded.manifest()["artifact_root_sha256"],
                manifest["artifact_root_sha256"],
            )
            mismatched = root / "mismatched.json"
            mismatched_document = json.loads(verification.read_text())
            mismatched_document["top_k"] = 32
            mismatched.write_text(json.dumps(mismatched_document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "settings differ"):
                VerifiedTeacherCache([verification, mismatched], "r", "p")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                VerifiedTeacherCache([verification, verification], "r", "p")
            artifact.write_bytes(b"other")
            with self.assertRaisesRegex(ValueError, "content"):
                dataset.verified_artifact_path(0)

    def test_uncached_teacher_selection_is_exact_and_rejects_extras(self) -> None:
        records = [
            {"id": "a", "category": "chat"},
            {"id": "b", "category": "code"},
            {"id": "c", "category": "math"},
        ]
        reused, missing = select_missing(records, {"a", "c"})
        self.assertEqual([record["id"] for record in reused], ["a", "c"])
        self.assertEqual([record["id"] for record in missing], ["b"])
        with self.assertRaisesRegex(ValueError, "outside final cohort"):
            select_missing(records, {"z"})

    def test_teacher_batch_runner_binds_an_explicit_cache_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "hf"
            cache.mkdir()
            inherited = {"HF_HOME": "/stale/cache", "UNCHANGED": "yes"}
            environment = cache_environment(inherited, cache)
            self.assertEqual(environment["HF_HOME"], str(cache.resolve()))
            self.assertEqual(environment["HF_HUB_CACHE"], str(cache.resolve() / "hub"))
            self.assertEqual(environment["UNCHANGED"], "yes")
            self.assertEqual(inherited["HF_HOME"], "/stale/cache")
            with self.assertRaisesRegex(ValueError, "not a directory"):
                cache_environment(inherited, cache / "missing")

    def test_python_ctox_reader_matches_native_header_and_tensor_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tiny.ctoxq"
            payload = b"\x00\x01\x02\x03"
            manifest = {
                "format": "ctox.q2q4.v1",
                "model": "test",
                "revision": "revision",
                "alignment": 64,
                "target": "canonical-b64",
                "tensors": [
                    {
                        "name": "scale",
                        "dtype": "f16",
                        "shape": [2],
                        "offset": 0,
                        "length": 4,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
            manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode()
            data_offset = (HEADER.size + len(manifest_bytes) + 63) & ~63
            path.write_bytes(
                HEADER.pack(
                    MAGIC, 1, ENDIAN_MARKER, len(manifest_bytes), data_offset, 1, 64
                )
                + manifest_bytes
                + b"\0" * (data_offset - HEADER.size - len(manifest_bytes))
                + payload
            )
            with CtoxArtifact(path, verify_tensors=True) as artifact:
                view = artifact.tensor_bytes("scale")
                self.assertEqual(bytes(view), payload)
                view.release()
                values = (
                    artifact.decode_float_tensor("scale", torch, "cpu")
                    if torch
                    else None
                )
                if values is not None:
                    self.assertEqual(tuple(values.shape), (2,))

    def test_python_ctox_reader_decodes_canonical_q2_codes(self) -> None:
        if torch is None:
            self.skipTest("torch is not installed in the host-only test environment")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "q2.ctoxq"
            packed_code = bytes([0 | (1 << 2) | (2 << 4) | (3 << 6)])
            payload = (
                torch.tensor([1.0], dtype=torch.float16)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
            payload += packed_code * 16
            manifest = {
                "format": "ctox.q2q4.v1",
                "model": "test",
                "revision": "revision",
                "alignment": 64,
                "target": "canonical-b64",
                "tensors": [
                    {
                        "name": "weight",
                        "dtype": "q2_b64",
                        "shape": [1, 64],
                        "offset": 0,
                        "length": 18,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
            manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode()
            data_offset = (HEADER.size + len(manifest_bytes) + 63) & ~63
            path.write_bytes(
                HEADER.pack(
                    MAGIC, 1, ENDIAN_MARKER, len(manifest_bytes), data_offset, 1, 64
                )
                + manifest_bytes
                + b"\0" * (data_offset - HEADER.size - len(manifest_bytes))
                + payload
            )
            with CtoxArtifact(path, verify_tensors=True) as artifact:
                values = artifact.decode_matrix_rows("weight", 0, 1, torch, "cpu")
                self.assertEqual(tuple(values.shape), (1, 64))
                expected = torch.tensor([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0])
                self.assertTrue(torch.allclose(values[0, :4], expected))

    def test_packed_linear_recomputes_codes_and_matches_dense_gradients(self) -> None:
        if torch is None:
            self.skipTest("torch is not installed in the host-only test environment")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "linear.ctoxq"
            packed_code = bytes([0 | (1 << 2) | (2 << 4) | (3 << 6)])
            scale_bytes = (
                torch.tensor([1.0], dtype=torch.float16)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
            payload = scale_bytes + packed_code * 16
            payload += scale_bytes + packed_code * 16
            manifest = {
                "format": "ctox.q2q4.v1",
                "model": "test",
                "revision": "revision",
                "alignment": 64,
                "target": "canonical-b64",
                "tensors": [
                    {
                        "name": "weight",
                        "dtype": "q2_b64",
                        "shape": [2, 64],
                        "offset": 0,
                        "length": 36,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
            manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode()
            data_offset = (HEADER.size + len(manifest_bytes) + 63) & ~63
            path.write_bytes(
                HEADER.pack(
                    MAGIC, 1, ENDIAN_MARKER, len(manifest_bytes), data_offset, 1, 64
                )
                + manifest_bytes
                + b"\0" * (data_offset - HEADER.size - len(manifest_bytes))
                + payload
            )
            with CtoxArtifact(path, verify_tensors=True) as artifact:
                dense_weight = artifact.decode_matrix_rows("weight", 0, 2, torch, "cpu")
                torch.manual_seed(7)
                inputs = torch.randn(2, 3, 64, requires_grad=True)
                s_in = torch.full((64,), 1.1, requires_grad=True)
                s_out = torch.tensor([0.9, 1.2], requires_grad=True)
                bias = torch.tensor([0.1, -0.2], requires_grad=True)
                output = packed_linear(
                    torch,
                    artifact,
                    "weight",
                    inputs,
                    s_in,
                    s_out,
                    bias,
                    rows_per_chunk=1,
                )
                output.square().sum().backward()
                packed_grads = [
                    value.grad.clone() for value in (inputs, s_in, s_out, bias)
                ]

                dense_inputs = inputs.detach().clone().requires_grad_()
                dense_s_in = s_in.detach().clone().requires_grad_()
                dense_s_out = s_out.detach().clone().requires_grad_()
                dense_bias = bias.detach().clone().requires_grad_()
                dense_output = (
                    torch.nn.functional.linear(
                        dense_inputs * dense_s_in, dense_weight, dense_bias
                    )
                    * dense_s_out
                )
                dense_output.square().sum().backward()
                for packed_grad, value in zip(
                    packed_grads, (dense_inputs, dense_s_in, dense_s_out, dense_bias)
                ):
                    self.assertTrue(
                        torch.allclose(packed_grad, value.grad, atol=1e-4, rtol=1e-4)
                    )
                module_class = packed_recovery_linear_class(torch)
                module = module_class(
                    artifact,
                    "weight",
                    torch.ones(64),
                    torch.ones(2),
                    bias=torch.zeros(2),
                    rows_per_chunk=1,
                )
                module(torch.randn(1, 64)).sum().backward()
                self.assertEqual(
                    {name for name, _parameter in module.named_parameters()},
                    {"log_s_in", "log_s_out"},
                )
                self.assertEqual(
                    set(module.correction_tensors()),
                    {"weight.s_in", "weight.s_out"},
                )
                embedding_class = packed_recovery_embedding_class(torch)
                embedding = embedding_class(
                    artifact,
                    "weight",
                    torch.ones(64),
                    torch.ones(2),
                )
                ids = torch.tensor([[1, 0, 1]])
                embedded = embedding(ids)
                expected_embedding = dense_weight.index_select(
                    0, ids.reshape(-1)
                ).reshape(1, 3, 64)
                self.assertTrue(torch.allclose(embedded, expected_embedding))
                embedded.sum().backward()
                self.assertIsNotNone(embedding.log_s_in.grad)
                self.assertIsNotNone(embedding.log_s_out.grad)

    def test_packed_recovery_registry_requires_and_loads_exact_scale_pairs(
        self,
    ) -> None:
        if torch is None:
            self.skipTest("torch is not installed in the host-only test environment")
        artifact = self.FakePackedArtifact(torch)
        registry = PackedRecoveryRegistry(artifact, torch)
        self.assertEqual(registry.weight_names, ["layer.weight"])
        self.assertEqual(registry.scale_parameter_count(), 66)
        artifact.tensors.pop("layer.weight.s_out")
        with self.assertRaisesRegex(ValueError, "lacks"):
            PackedRecoveryRegistry(artifact, torch)

    def test_qwen38_fanout_contract_covers_every_same_input_projection(self) -> None:
        names = self.qwen38_fanout_weight_names()
        groups = qwen38_fanout_groups(names)
        counts = Counter(group["kind"] for group in groups)
        self.assertEqual(
            counts,
            {
                "mlp_gate_up": 65,
                "full_attention_qkv": 17,
                "linear_attention_inputs": 48,
            },
        )
        self.assertEqual(len(groups), 130)
        self.assertEqual(sum(len(group["weights"]) for group in groups), 373)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            qwen38_fanout_groups(
                {
                    "model.language_model.layers.3.self_attn.q_proj.weight",
                    "model.language_model.layers.3.self_attn.k_proj.weight",
                },
                require_frozen_topology=False,
            )

    def test_fanout_tying_uses_one_geometric_mean_parameter_and_exact_alias_gate(
        self,
    ) -> None:
        if torch is None:
            self.skipTest("torch is not installed in the host-only test environment")

        class RecoveryModule(torch.nn.Module):
            def __init__(self, name: str, s_in: float) -> None:
                super().__init__()
                self.name = name
                self.log_s_in = torch.nn.Parameter(
                    torch.full((4,), s_in).log()
                )
                self.log_s_out = torch.nn.Parameter(torch.zeros(2))

        main = torch.nn.Module()
        main.projections = torch.nn.ModuleList(
            [
                RecoveryModule(
                    f"model.language_model.layers.3.self_attn.{projection}.weight",
                    value,
                )
                for projection, value in zip(
                    ("q_proj", "k_proj", "v_proj"),
                    (1.0, 2.0, 4.0),
                )
            ]
        )
        mtp = torch.nn.Module()
        independent = tie_fanout_s_in(main, mtp, torch, INDEPENDENT_POLICY)
        independent_parameters = unique_scale_parameters(main, mtp, independent)
        self.assertEqual(len({id(module.log_s_in) for module in main.projections}), 3)
        self.assertEqual(independent["group_count"], 0)

        evidence = tie_fanout_s_in(
            main,
            mtp,
            torch,
            QWEN38_FANOUT_POLICY,
            require_frozen_topology=False,
        )
        parameters = unique_scale_parameters(main, mtp, evidence)
        expected = torch.tensor(8.0 ** (1.0 / 3.0))
        self.assertTrue(
            torch.allclose(
                main.projections[0].log_s_in.exp(),
                torch.full((4,), expected),
            )
        )
        self.assertEqual(len({id(module.log_s_in) for module in main.projections}), 1)
        self.assertEqual(len(independent_parameters), len(parameters))
        self.assertEqual(evidence["group_count"], 1)
        self.assertEqual(evidence["a8_quantizations_avoided_per_complete_fanout_pass"], 2)
        with self.assertRaisesRegex(ValueError, "aliases differ"):
            unique_scale_parameters(main, mtp, {"policy": INDEPENDENT_POLICY, "groups": []})

    def test_packed_student_assignment_replaces_only_exact_qualified_target(
        self,
    ) -> None:
        if torch is None:
            self.skipTest("torch is not installed in the host-only test environment")
        root = torch.nn.Module()
        root.child = torch.nn.Module()
        root.child.linear = torch.nn.Linear(2, 2)
        replacement = torch.nn.Identity()
        set_submodule(root, "child.linear", replacement)
        self.assertIs(root.child.linear, replacement)
        set_parameter(
            root,
            "child.scale",
            torch.nn.Parameter(torch.ones(2), requires_grad=False),
        )
        self.assertEqual(tuple(root.child.scale.shape), (2,))
        checkpoint_name = "model.language_model.layers.3.mlp.down_proj.weight"
        runtime_name = "model.layers.3.mlp.down_proj.weight"
        self.assertEqual(artifact_to_runtime_name(checkpoint_name), runtime_name)
        self.assertEqual(runtime_to_artifact_name(runtime_name), checkpoint_name)

    def test_local_teacher_provenance_rejects_revision_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            document = {
                "format": "ctox.verified-local-model.v1",
                "revision": "a" * 40,
                "local_root": str(model.resolve()),
                "root_sha256": "b" * 64,
                "files": [{"name": "config.json", "bytes": 2, "sha256": "c" * 64}],
            }
            provenance = root / "provenance.json"
            provenance.write_text(json.dumps(document), encoding="utf-8")
            loaded, digest = validate_local_model_provenance(
                model, "a" * 40, provenance
            )
            self.assertEqual(loaded, document)
            self.assertEqual(len(digest), 64)
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_local_model_provenance(model, "d" * 40, provenance)

    def test_verified_model_root_digest_is_order_independent(self) -> None:
        files = [
            {"name": "b", "bytes": 2, "sha256": "2" * 64},
            {"name": "a", "bytes": 1, "sha256": "1" * 64},
        ]
        self.assertEqual(root_digest(files), root_digest(list(reversed(files))))

    def test_teacher_runtime_pins_fla_and_reserves_gpu_headroom(self) -> None:
        self.assertEqual(len(FLA_KERNEL_REVISION), 40)
        self.assertEqual(
            weight_max_memory(3, 16, 96),
            {0: "16GiB", 1: "16GiB", 2: "16GiB", "cpu": "96GiB"},
        )
        self.assertIsNone(weight_max_memory(3, None, 96))

    def test_german_rag_context_is_part_of_hashed_payload(self) -> None:
        repo = "Beko2210/German-Instruct-Dataset"
        row = {
            "input": "Fasse zusammen",
            "context": "Nur dieser Text",
            "output": "Kurzfassung",
        }
        payload = recovery_payload(row, repo)
        self.assertIn("Kontext:\nNur dieser Text", payload["messages"][0]["content"])
        changed = dict(row, context="Ein anderer Text")
        self.assertNotEqual(canonical_text(row, repo), canonical_text(changed, repo))
        self.assertEqual(
            category_for("default", "train", {"category": "coding"}), "code"
        )
        self.assertEqual(
            category_for("default", "train", {"category": "rag"}), "long_context"
        )

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

    def test_local_generated_payloads_are_grouped_by_source_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generated.jsonl"
            record = {
                "id": "a" * 64,
                "source_repo": "metric-space-ai/CTOX-LLM",
                "source_revision": "b" * 40,
                "subset": "ctox.long-context.v1",
                "split": "calibration",
                "messages": [{"role": "user", "content": "payload"}],
            }
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            groups = load_local_materialized([path])
            key = (
                record["source_repo"],
                record["source_revision"],
                record["subset"],
                record["split"],
            )
            self.assertEqual(groups[key], [record])

    def test_selection_is_balanced_and_order_independent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for category in ("code", "math"):
                path = Path(directory) / f"{category}.jsonl"
                records = [
                    {
                        "id": hashlib.sha256(
                            f"{category}-{index}".encode()
                        ).hexdigest(),
                        "category": category,
                    }
                    for index in range(10)
                ]
                path.write_text(
                    "".join(json.dumps(record) + "\n" for record in records),
                    encoding="utf-8",
                )
                paths.append(path)
            first = select(paths, per_manifest=3, seed="fixed")
            second = select(paths, per_manifest=3, seed="fixed")
            self.assertEqual(first, second)
            self.assertEqual([record["category"] for record in first].count("code"), 3)
            self.assertEqual([record["category"] for record in first].count("math"), 3)

    def test_recovery_split_is_deterministic_disjoint_and_release_eligible(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for source_index in range(2):
                path = Path(temporary) / f"source-{source_index}.jsonl"
                with path.open("w", encoding="utf-8") as output:
                    for record_index in range(8):
                        sample_id = hashlib.sha256(
                            f"{source_index}-{record_index}".encode()
                        ).hexdigest()
                        output.write(
                            json.dumps(
                                {
                                    "id": sample_id,
                                    "category": "agentic",
                                    "source_repo": f"source-{source_index}",
                                    "release_eligible": True,
                                }
                            )
                            + "\n"
                        )
                paths.append(path)
            train, evaluation = split(paths, 4, 2, "fixed-seed")
            repeated_train, repeated_evaluation = split(paths, 4, 2, "fixed-seed")
            self.assertEqual(train, repeated_train)
            self.assertEqual(evaluation, repeated_evaluation)
            self.assertEqual(len(train), 8)
            self.assertEqual(len(evaluation), 4)
            self.assertFalse(
                {record["id"] for record in train}
                & {record["id"] for record in evaluation}
            )

            quarantined = Path(temporary) / "quarantined.jsonl"
            quarantined.write_text(
                json.dumps(
                    {
                        "id": "f" * 64,
                        "category": "agentic",
                        "source_repo": "quarantined",
                        "release_eligible": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "release-ineligible"):
                split([quarantined], 1, 1, "fixed-seed")

    def test_recovery_split_enforces_each_language_stratum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "languages.jsonl"
            records = []
            for language in ("fra", "jpn"):
                for index in range(8):
                    records.append(
                        {
                            "id": hashlib.sha256(
                                f"{language}-{index}".encode()
                            ).hexdigest(),
                            "category": "chat",
                            "language": language,
                            "source_repo": "example/aya",
                            "release_eligible": True,
                        }
                    )
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            train, evaluation = split(
                [path], 3, 2, "language-seed", stratify_field="language"
            )
            for language in ("fra", "jpn"):
                self.assertEqual(sum(r["language"] == language for r in train), 3)
                self.assertEqual(sum(r["language"] == language for r in evaluation), 2)
            self.assertFalse(
                {record["id"] for record in train}
                & {record["id"] for record in evaluation}
            )

    def test_recovery_split_excludes_prior_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.jsonl"
            records = [
                {
                    "id": hashlib.sha256(str(index).encode()).hexdigest(),
                    "category": "chat",
                    "source_repo": "example/source",
                    "release_eligible": True,
                }
                for index in range(10)
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            excluded = {records[0]["id"], records[1]["id"]}
            train, evaluation = split(
                [path], 4, 2, "exclusion-seed", excluded_ids=excluded
            )
            self.assertFalse(excluded & {record["id"] for record in train})
            self.assertFalse(excluded & {record["id"] for record in evaluation})

    def test_manifest_merge_rejects_conflicts_and_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.jsonl"
            second = Path(temporary) / "second.jsonl"
            record = {
                "id": "a" * 64,
                "category": "chat",
                "language": "eng",
                "source_repo": "example/source",
                "release_eligible": True,
            }
            first.write_text(json.dumps(record) + "\n", encoding="utf-8")
            second.write_text(json.dumps(record) + "\n", encoding="utf-8")
            self.assertEqual(merge([first, second]), [record])
            changed = dict(record, category="code")
            second.write_text(json.dumps(changed) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
                merge([first, second])
            second.write_text(
                json.dumps(dict(record, id="b" * 64, release_eligible=False)) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "release-ineligible"):
                merge([first, second])

    def test_openai_tool_call_is_flattened_for_qwen_template(self) -> None:
        call = {
            "id": "call-1",
            "type": "function",
            "function": {"name": "weather", "arguments": '{"city":"Berlin"}'},
        }
        self.assertEqual(
            normalize_tool_call(call),
            {"name": "weather", "arguments": {"city": "Berlin"}},
        )

    def test_empty_tool_calls_are_removed(self) -> None:
        messages = normalize_messages(
            [{"role": "user", "content": "hello", "tool_calls": []}]
        )
        self.assertEqual(messages, [{"role": "user", "content": "hello"}])

    def test_structured_tool_result_content_is_canonical_json(self) -> None:
        messages = normalize_messages(
            [
                {"role": "user", "content": "look it up"},
                {"role": "tool", "content": {"z": 1, "a": ["x", "y"]}},
            ]
        )
        self.assertEqual(messages[1]["content"], '{"a":["x","y"],"z":1}')
        with self.assertRaisesRegex(ValueError, "unsupported type"):
            normalize_content(object(), 0, "content")

    def test_runtime_linear_name_maps_to_checkpoint(self) -> None:
        self.assertEqual(
            checkpoint_weight_name("model.layers.12.mlp.down_proj"),
            "model.language_model.layers.12.mlp.down_proj.weight",
        )

    def test_prefill_ranges_cover_sequence_exactly(self) -> None:
        self.assertEqual(prefill_ranges(10, 4), [(0, 4), (4, 8), (8, 10)])
        self.assertEqual(prefill_ranges(10, 0), [(0, 10)])
        self.assertEqual(prefill_ranges(10, 10), [(0, 10)])
        flattened = [
            position
            for start, stop in prefill_ranges(17, 5)
            for position in range(start, stop)
        ]
        self.assertEqual(flattened, list(range(17)))

    def test_mixed_q2_q4_tensors_remain_calibration_targets(self) -> None:
        plan = {
            "tensors": [
                {
                    "name": "lm_head.weight",
                    "source_shard": "model.safetensors",
                    "dtype": "mixed_q2_q4_b64",
                },
                {
                    "name": "matrix.weight",
                    "source_shard": "model.safetensors",
                    "dtype": "q2_b64",
                },
                {
                    "name": "matrix.weight.s_in",
                    "source_shard": None,
                    "dtype": "f16",
                },
            ]
        }
        self.assertEqual(
            quantized_source_names(plan),
            {"lm_head.weight", "matrix.weight"},
        )
        self.assertEqual(
            [entry["name"] for entry in quantized_entries(plan)],
            ["lm_head.weight", "matrix.weight"],
        )

    def test_merged_activation_metadata_does_not_claim_one_runtime_profile(
        self,
    ) -> None:
        reference = {
            "model": "model",
            "revision": "revision",
            "observed_modules": "2",
            "target_tensors": "2",
            "unobserved_tensors": "[]",
            "input_only_tensors": "[]",
            "row_frequency_tensors": "[]",
            "fla_kernel": "{}",
            "gpu_weight_memory_gib": "16",
        }
        metadata = merged_metadata(
            reference,
            ["a", "b"],
            12,
            ["1" * 64, "2" * 64],
            [
                {"gpu_weight_memory_gib": "16", "max_length": "2048"},
                {"gpu_weight_memory_gib": "9", "max_length": "131072"},
            ],
        )
        self.assertNotIn("gpu_weight_memory_gib", metadata)
        self.assertEqual(metadata["merged_batches"], "2")
        self.assertEqual(
            [
                profile["gpu_weight_memory_gib"]
                for profile in json.loads(metadata["source_runtime_profiles"])
            ],
            ["16", "9"],
        )
        self.assertEqual(
            [
                profile["max_length"]
                for profile in json.loads(metadata["source_runtime_profiles"])
            ],
            ["2048", "131072"],
        )

    def test_nested_activation_merge_preserves_leaf_runtime_profiles(self) -> None:
        metadata = {
            "source_runtime_profiles": json.dumps(
                [
                    {"max_length": "2048", "gpu_weight_memory_gib": "16"},
                    {"max_length": "131072", "gpu_weight_memory_gib": "9"},
                ]
            )
        }
        profiles = source_runtime_profiles(metadata)
        self.assertEqual(len(profiles), 2)
        self.assertEqual(
            [profile["max_length"] for profile in profiles],
            ["2048", "131072"],
        )
        self.assertIn("cuda_memory", profiles[0])

    def test_mtp_names_map_to_frozen_checkpoint(self) -> None:
        self.assertEqual(
            mtp_checkpoint_weight_name("layers.0.eh_proj"),
            "mtp.fc.weight",
        )
        self.assertEqual(
            mtp_checkpoint_weight_name("layers.0.mtp_block.self_attn.q_proj"),
            "mtp.layers.0.self_attn.q_proj.weight",
        )
        mapping = mtp_parameter_mapping(
            {
                "mtp.fc.weight",
                "mtp.pre_fc_norm_embedding.weight",
                "mtp.layers.0.mlp.down_proj.weight",
            },
            1,
        )
        self.assertEqual(mapping["mtp.fc.weight"], "layers.0.eh_proj.weight")
        self.assertEqual(
            mapping["mtp.layers.0.mlp.down_proj.weight"],
            "layers.0.mtp_block.mlp.down_proj.weight",
        )
        with self.assertRaisesRegex(ValueError, "exactly one MTP layer"):
            mtp_parameter_mapping(set(), 2)

    def test_row_group_records_exact_quantized_bytes(self) -> None:
        group = row_group_document(2, 512, 768, 5120, 10.0, 2.0, 100.0)
        self.assertEqual(group["row_start"], 512)
        self.assertEqual(group["row_end"], 768)
        self.assertEqual(group["q2_bytes"], 368640)
        self.assertEqual(group["q4_bytes"], 696320)
        self.assertEqual(group["quality_gain"], 8.0)

    def test_q4_budget_uses_complete_aligned_layout(self) -> None:
        plan = {
            "alignment": 256,
            "tensors": [
                {
                    "name": "matrix.weight",
                    "source_shard": "model.safetensors",
                    "dtype": "q2_b64",
                    "shape": [1, 1024],
                    "length": 288,
                },
                {
                    "name": "matrix.weight.s_in",
                    "source_shard": None,
                    "dtype": "f16",
                    "shape": [1024],
                    "length": 2048,
                },
            ],
        }
        self.assertEqual(layout_bytes(plan, set()), 2560)
        self.assertEqual(layout_bytes(plan, {"matrix.weight"}), 2816)

    def test_q4_optimizer_recomputes_exact_marginal_layout_cost(self) -> None:
        plan = {
            "alignment": 256,
            "tensors": [
                {
                    "name": "a.weight",
                    "source_shard": "model.safetensors",
                    "dtype": "q2_b64",
                    "shape": [1, 1024],
                    "length": 288,
                },
                {
                    "name": "b.weight",
                    "source_shard": "model.safetensors",
                    "dtype": "q2_b64",
                    "shape": [1, 1024],
                    "length": 288,
                },
            ],
        }
        candidates = [
            {
                "name": "a.weight",
                "fixed_q4": False,
                "quality_gain": 2.0,
                "q2_bytes": 288,
                "q4_bytes": 544,
            },
            {
                "name": "b.weight",
                "fixed_q4": False,
                "quality_gain": 5.0,
                "q2_bytes": 288,
                "q4_bytes": 544,
            },
        ]
        base = layout_bytes(plan, set())
        selected, groups, decisions = optimized_selections(
            plan,
            candidates,
            {},
            set(),
            {},
            base + 256,
        )
        self.assertEqual(selected, {"b.weight"})
        self.assertEqual(groups, {})
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]["marginal_layout_bytes"], 256)
        self.assertEqual(decisions[0]["layout_bytes_after"], base + 256)

    def test_mixed_row_groups_change_exact_layout(self) -> None:
        groups = [
            {
                "group_index": 0,
                "q2_bytes": 576,
                "q4_bytes": 1088,
            },
            {
                "group_index": 1,
                "q2_bytes": 576,
                "q4_bytes": 1088,
            },
        ]
        plan = {
            "alignment": 256,
            "tensors": [
                {
                    "name": "embedding.weight",
                    "source_shard": "model.safetensors",
                    "dtype": "q2_b64",
                    "shape": [64, 64],
                    "length": 1152,
                }
            ],
        }
        self.assertEqual(mixed_tensor_bytes(groups, set()), 1152)
        self.assertEqual(mixed_tensor_bytes(groups, {1}), 1664)
        self.assertEqual(
            layout_bytes(
                plan,
                set(),
                {"embedding.weight": groups},
                {"embedding.weight": {1}},
            ),
            1792,
        )
        plan["tensors"][0]["dtype"] = "mixed_q2_q4_b64"
        self.assertEqual(
            layout_bytes(
                plan,
                set(),
                {"embedding.weight": groups},
                {"embedding.weight": {1}},
            ),
            1792,
        )

    def test_fixed_mixed_tensor_selects_every_row_group(self) -> None:
        groups = [{"group_index": 0}, {"group_index": 1}]
        candidates = [
            {
                "name": "lm_head.weight",
                "fixed_q4": True,
                "row_groups": groups,
            }
        ]
        selected, selected_groups = initial_selections(
            candidates,
            {"lm_head.weight": groups},
        )
        self.assertEqual(selected, set())
        self.assertEqual(selected_groups, {"lm_head.weight": {0, 1}})

    def test_release_q4_policy_has_no_architecture_name_exceptions(self) -> None:
        for name in [
            "lm_head.weight",
            "model.language_model.embed_tokens.weight",
            "model.language_model.layers.63.self_attn.k_proj.weight",
            "model.language_model.layers.63.self_attn.v_proj.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.63.mlp.down_proj.weight",
        ]:
            self.assertFalse(fixed_q4(name), name)

    def test_recovery_mixed_ranges_preserve_planned_qcodes(self) -> None:
        entry = {
            "name": "embedding.weight",
            "dtype": "mixed_q2_q4_b64",
            "shape": [12, 64],
            "segments": [
                {"row_start": 0, "row_end": 4, "dtype": "q4_b64"},
                {"row_start": 4, "row_end": 8, "dtype": "q2_b64"},
                {"row_start": 8, "row_end": 12, "dtype": "q4_b64"},
            ],
        }
        self.assertEqual(
            quant_dtype_ranges(entry, 2, 10),
            [(2, 4, "q4_b64"), (4, 8, "q2_b64"), (8, 10, "q4_b64")],
        )

    def test_packer_accepts_only_complete_plan_bound_recovery(self) -> None:
        plan_hash = "a" * 64
        plan = {
            "model": "Qwen/Qwen3.8-27B",
            "revision": "b" * 40,
            "tensors": [
                {
                    "name": "matrix.weight.s_in",
                    "shape": [64],
                    "group": "recovery",
                },
                {
                    "name": "matrix.weight.s_out",
                    "shape": [8],
                    "group": "recovery",
                },
            ],
        }
        metadata = {
            "format": "ctox.recovery.channel-scales.v2",
            "status": "complete",
            "model": plan["model"],
            "revision": plan["revision"],
            "plan_sha256": plan_hash,
            "activation_stats_sha256": "c" * 64,
            "report_sha256": "d" * 64,
            "fixed_logical_qcodes": "true",
        }
        tensors = {
            "matrix.weight.s_in": self.FakeRecoveryTensor((64,)),
            "matrix.weight.s_out": self.FakeRecoveryTensor((8,)),
        }
        descriptor = validate_recovery_source(
            plan,
            plan_hash,
            self.FakeRecovery(metadata, tensors),
        )
        self.assertEqual(descriptor["mode"], "trained")
        self.assertEqual(descriptor["activation_stats_sha256"], "c" * 64)

        fanout_metadata = dict(
            metadata,
            fanout_s_in_policy=INDEPENDENT_POLICY,
            fanout_group_sha256=fanout_group_sha256([]),
        )
        fanout_descriptor = validate_recovery_source(
            plan,
            plan_hash,
            self.FakeRecovery(fanout_metadata, tensors),
        )
        self.assertEqual(
            fanout_descriptor["fanout_s_in_policy"], INDEPENDENT_POLICY
        )
        self.assertEqual(fanout_descriptor["fanout_group_count"], 0)
        with self.assertRaisesRegex(RuntimeError, "group digest differs"):
            validate_recovery_source(
                plan,
                plan_hash,
                self.FakeRecovery(
                    dict(fanout_metadata, fanout_group_sha256="e" * 64),
                    tensors,
                ),
            )

        incomplete = dict(tensors)
        incomplete.pop("matrix.weight.s_out")
        with self.assertRaisesRegex(RuntimeError, "1 missing"):
            validate_recovery_source(
                plan,
                plan_hash,
                self.FakeRecovery(metadata, incomplete),
            )

        wrong_plan = dict(metadata, plan_sha256="e" * 64)
        with self.assertRaisesRegex(RuntimeError, "plan_sha256"):
            validate_recovery_source(
                plan,
                plan_hash,
                self.FakeRecovery(wrong_plan, tensors),
            )

    def test_packer_requires_byte_identical_declared_qwen_fanout_scales(self) -> None:
        if torch is None:
            self.skipTest("torch is not installed in the host-only test environment")
        weight_names = self.qwen38_fanout_weight_names()
        plan = {
            "model": "Qwen/Qwen3.8-27B",
            "revision": "b" * 40,
            "tensors": [],
        }
        tensors = {}
        for name in sorted(weight_names):
            plan["tensors"].extend(
                [
                    {"name": name, "dtype": "q2_b64", "shape": [2, 4]},
                    {
                        "name": f"{name}.s_in",
                        "dtype": "f16",
                        "shape": [4],
                        "group": "recovery",
                    },
                    {
                        "name": f"{name}.s_out",
                        "dtype": "f16",
                        "shape": [2],
                        "group": "recovery",
                    },
                ]
            )
            tensors[f"{name}.s_in"] = torch.ones(4, dtype=torch.float16)
            tensors[f"{name}.s_out"] = torch.ones(2, dtype=torch.float16)
        groups = qwen38_fanout_groups(weight_names)
        metadata = {
            "format": "ctox.recovery.channel-scales.v2",
            "status": "complete",
            "model": plan["model"],
            "revision": plan["revision"],
            "plan_sha256": "a" * 64,
            "activation_stats_sha256": "c" * 64,
            "report_sha256": "d" * 64,
            "fixed_logical_qcodes": "true",
            "fanout_s_in_policy": QWEN38_FANOUT_POLICY,
            "fanout_group_sha256": fanout_group_sha256(groups),
        }
        descriptor = validate_recovery_source(
            plan,
            "a" * 64,
            self.FakeRecovery(metadata, tensors),
        )
        self.assertEqual(descriptor["fanout_group_count"], 130)
        self.assertEqual(descriptor["fanout_logical_s_in_tensors"], 373)
        tensors[groups[0]["scale_names"][0]] = torch.full(
            (4,), 2.0, dtype=torch.float16
        )
        with self.assertRaisesRegex(RuntimeError, "fanout scales differ"):
            validate_recovery_source(
                plan,
                "a" * 64,
                self.FakeRecovery(metadata, tensors),
            )

    def test_fold_package_budget_reserves_manifest_bytes(self) -> None:
        self.assertEqual(CONTAINER_MANIFEST_RESERVE, 2 * 1024 * 1024)
        self.assertEqual(
            FOLD_RESIDENT_LIMIT + CONTAINER_MANIFEST_RESERVE,
            FOLD_PACKAGE_LIMIT,
        )

    def test_vendor_manifest_detects_digest_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "kernel.cu"
            source.write_bytes(b"pinned")
            manifest = root / "UPSTREAM.json"
            manifest.write_text(
                json.dumps(
                    {
                        "files": [
                            {
                                "path": source.name,
                                "sha256": hashlib.sha256(
                                    source.read_bytes()
                                ).hexdigest(),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(verify(manifest), 1)
            source.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                verify(manifest)

    def test_recovery_output_transaction_requires_resume_and_evidence_commits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scales = root / "scales.safetensors"
            report = root / "report.json"
            evidence = root / "evidence.json"
            report.write_text("partial", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "resume from its final checkpoint"):
                prepare_output_transaction(scales, report, evidence, None)
            checkpoint = root / "checkpoint.safetensors"
            checkpoint.write_bytes(b"checkpoint")
            prepare_output_transaction(scales, report, evidence, checkpoint)
            self.assertFalse(report.exists())
            atomic_json(evidence, {"format": "committed"})
            with self.assertRaisesRegex(ValueError, "refusing to overwrite committed"):
                prepare_output_transaction(scales, report, evidence, checkpoint)


if __name__ == "__main__":
    unittest.main()
