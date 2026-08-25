from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
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
from audit_corpus import percentile  # noqa: E402
from build_quant_plan import (  # noqa: E402
    CONTAINER_MANIFEST_RESERVE,
    FOLD_PACKAGE_LIMIT,
    FOLD_RESIDENT_LIMIT,
)
from cache_teacher import (  # noqa: E402
    mtp_target_positions,
    position_sets,
    validate_local_model_provenance,
)
from classify_domains import classification_text, deterministic_labels, quota_gaps  # noqa: E402
from collect_activation_stats import (  # noqa: E402
    checkpoint_weight_name,
    prefill_ranges,
    quantized_source_names,
)
from ctox_artifact import CtoxArtifact, ENDIAN_MARKER, HEADER, MAGIC  # noqa: E402
from materialize_prompts import load_local_materialized, load_manifests  # noqa: E402
from merge_manifests import merge  # noqa: E402
from merge_activation_stats import merged_metadata, source_runtime_profiles  # noqa: E402
from mtp_teacher import mtp_checkpoint_weight_name, mtp_parameter_mapping  # noqa: E402
from optimize_q4_budget import initial_selections, layout_bytes, mixed_tensor_bytes  # noqa: E402
from plan_teacher_cache import sample_tensor_bytes  # noqa: E402
from plan_teacher_batches import batches as plan_teacher_batches  # noqa: E402
from prompt_format import normalize_content, normalize_messages, normalize_tool_call  # noqa: E402
from generate_long_context import generated_record  # noqa: E402
from fit_recovery_scales import quant_dtype_ranges  # noqa: E402
try:  # Optional local training dependency; exercised in the pinned GPU venv.
    import torch  # noqa: E402
    from recovery_modules import (  # noqa: E402
        normalized_hidden_loss,
        end_to_end_recovery_loss,
        sparse_teacher_kl,
        supervised_mtp_token_loss,
        supervised_next_token_loss,
    )
except ModuleNotFoundError:
    torch = None
from pack_checkpoint import validate_recovery_source  # noqa: E402
from packed_recovery_ops import packed_linear  # noqa: E402
from score_quant_sensitivity import quantized_entries, row_group_document  # noqa: E402
from select_manifest import select  # noqa: E402
from select_teacher_smoke import select_ids as select_teacher_smoke_ids  # noqa: E402
from select_primary_domain_supplement import select_supplement  # noqa: E402
from split_manifests import split  # noqa: E402
from teacher_runtime import FLA_KERNEL_REVISION, weight_max_memory  # noqa: E402
from teacher_cache_dataset import VerifiedTeacherCache  # noqa: E402
from verify_vendor_manifest import verify  # noqa: E402
from verify_local_model import root_digest  # noqa: E402
from verify_teacher_cache import expected_tensor_specs  # noqa: E402
from run_teacher_batches import completed_batch_matches  # noqa: E402


class DatasetPipelineTests(unittest.TestCase):
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
            return "\n".join(f"{message['role']}: {message['content']}" for message in messages)

        def __call__(self, text, **_kwargs):
            return type("Encoded", (), {"input_ids": text.replace("\n", " \n ").split()})()

    def test_corpus_percentiles_are_deterministic(self) -> None:
        self.assertEqual(percentile([9, 1, 5, 3], 0.0), 1)
        self.assertEqual(percentile([9, 1, 5, 3], 0.5), 5)
        self.assertEqual(percentile([9, 1, 5, 3], 1.0), 9)

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
                "agentic_tools_search",
                "data_structured_outputs",
                "software_cybersecurity",
            },
        )
        self.assertIn("user: Debug this program", classification_text(record))

    def test_domain_gate_requires_clear_primary_examples_not_only_multilabel_hits(self) -> None:
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

    def test_primary_supplement_closes_each_gap_with_margin_and_confidence(self) -> None:
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

    def test_long_context_generator_is_sized_and_requires_two_retrieval_positions(self) -> None:
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

    def test_teacher_cache_verifier_contract_includes_mtp_and_hidden_layers(self) -> None:
        specs = expected_tensor_specs(100, 20, 10, 19, 7, 64, 5120, [0, 63], True)
        self.assertEqual(specs["input_ids"], ("I32", [1, 100]))
        self.assertEqual(specs["topk_logprobs"], ("BF16", [1, 20, 64]))
        self.assertEqual(specs["hidden_63"], ("BF16", [1, 10, 5120]))
        self.assertEqual(specs["mtp_hidden"], ("BF16", [1, 7, 5120]))
        self.assertEqual(specs["mtp_topk_indices"], ("I32", [1, 19, 64]))

    def test_teacher_smoke_selects_every_domain_and_language_deterministically(self) -> None:
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

    def test_end_to_end_recovery_objective_includes_every_base_and_mtp_family(self) -> None:
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

    def test_verified_teacher_cache_rejects_duplicate_or_changed_artifacts(self) -> None:
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
            self.assertEqual(loaded.manifest()["artifact_root_sha256"], manifest["artifact_root_sha256"])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                VerifiedTeacherCache([verification, verification], "r", "p")
            artifact.write_bytes(b"other")
            with self.assertRaisesRegex(ValueError, "content"):
                dataset.verified_artifact_path(0)

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
                HEADER.pack(MAGIC, 1, ENDIAN_MARKER, len(manifest_bytes), data_offset, 1, 64)
                + manifest_bytes
                + b"\0" * (data_offset - HEADER.size - len(manifest_bytes))
                + payload
            )
            with CtoxArtifact(path, verify_tensors=True) as artifact:
                view = artifact.tensor_bytes("scale")
                self.assertEqual(bytes(view), payload)
                view.release()

    def test_python_ctox_reader_decodes_canonical_q2_codes(self) -> None:
        if torch is None:
            self.skipTest("torch is not installed in the host-only test environment")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "q2.ctoxq"
            packed_code = bytes([0 | (1 << 2) | (2 << 4) | (3 << 6)])
            payload = torch.tensor([1.0], dtype=torch.float16).view(torch.uint8).numpy().tobytes()
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
                HEADER.pack(MAGIC, 1, ENDIAN_MARKER, len(manifest_bytes), data_offset, 1, 64)
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
            scale_bytes = torch.tensor([1.0], dtype=torch.float16).view(torch.uint8).numpy().tobytes()
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
                HEADER.pack(MAGIC, 1, ENDIAN_MARKER, len(manifest_bytes), data_offset, 1, 64)
                + manifest_bytes
                + b"\0" * (data_offset - HEADER.size - len(manifest_bytes))
                + payload
            )
            with CtoxArtifact(path, verify_tensors=True) as artifact:
                dense_weight = artifact.decode_matrix_rows("weight", 0, 2, torch, "cpu")
                inputs = torch.randn(2, 3, 64, requires_grad=True)
                s_in = torch.full((64,), 1.1, requires_grad=True)
                s_out = torch.tensor([0.9, 1.2], requires_grad=True)
                bias = torch.tensor([0.1, -0.2], requires_grad=True)
                output = packed_linear(
                    torch, artifact, "weight", inputs, s_in, s_out, bias, rows_per_chunk=1
                )
                output.square().sum().backward()
                packed_grads = [value.grad.clone() for value in (inputs, s_in, s_out, bias)]

                dense_inputs = inputs.detach().clone().requires_grad_()
                dense_s_in = s_in.detach().clone().requires_grad_()
                dense_s_out = s_out.detach().clone().requires_grad_()
                dense_bias = bias.detach().clone().requires_grad_()
                dense_output = torch.nn.functional.linear(
                    dense_inputs * dense_s_in, dense_weight, dense_bias
                ) * dense_s_out
                dense_output.square().sum().backward()
                for packed_grad, value in zip(
                    packed_grads, (dense_inputs, dense_s_in, dense_s_out, dense_bias)
                ):
                    self.assertTrue(torch.allclose(packed_grad, value.grad, atol=1e-5, rtol=1e-5))

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
        row = {"input": "Fasse zusammen", "context": "Nur dieser Text", "output": "Kurzfassung"}
        payload = recovery_payload(row, repo)
        self.assertIn("Kontext:\nNur dieser Text", payload["messages"][0]["content"])
        changed = dict(row, context="Ein anderer Text")
        self.assertNotEqual(canonical_text(row, repo), canonical_text(changed, repo))
        self.assertEqual(category_for("default", "train", {"category": "coding"}), "code")
        self.assertEqual(category_for("default", "train", {"category": "rag"}), "long_context")

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

    def test_recovery_split_is_deterministic_disjoint_and_release_eligible(self) -> None:
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
                            "id": hashlib.sha256(f"{language}-{index}".encode()).hexdigest(),
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
        messages = normalize_messages([{"role": "user", "content": "hello", "tool_calls": []}])
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
        flattened = [position for start, stop in prefill_ranges(17, 5) for position in range(start, stop)]
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

    def test_merged_activation_metadata_does_not_claim_one_runtime_profile(self) -> None:
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
            [profile["gpu_weight_memory_gib"] for profile in json.loads(metadata["source_runtime_profiles"])],
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
                                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
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


if __name__ == "__main__":
    unittest.main()
