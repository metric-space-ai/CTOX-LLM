from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))

from build_manifest import (  # noqa: E402
    SOURCES,
    canonical_text,
    category_for,
    license_ids,
    recovery_payload,
    source_id_for,
)
from build_quant_plan import (  # noqa: E402
    CONTAINER_MANIFEST_RESERVE,
    FOLD_PACKAGE_LIMIT,
    FOLD_RESIDENT_LIMIT,
)
from collect_activation_stats import checkpoint_weight_name  # noqa: E402
from materialize_prompts import load_manifests  # noqa: E402
from mtp_teacher import mtp_checkpoint_weight_name, mtp_parameter_mapping  # noqa: E402
from optimize_q4_budget import layout_bytes, mixed_tensor_bytes  # noqa: E402
from prompt_format import normalize_messages, normalize_tool_call  # noqa: E402
from generate_long_context import generated_record  # noqa: E402
from score_quant_sensitivity import row_group_document  # noqa: E402
from select_manifest import select  # noqa: E402
from verify_vendor_manifest import verify  # noqa: E402


class DatasetPipelineTests(unittest.TestCase):
    class WordTokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            return "\n".join(f"{message['role']}: {message['content']}" for message in messages)

        def __call__(self, text, **_kwargs):
            return type("Encoded", (), {"input_ids": text.replace("\n", " \n ").split()})()

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

    def test_nested_metadata_supplies_stable_source_id(self) -> None:
        row = {"metadata": {"uuid": "stable-row-id"}}
        self.assertEqual(source_id_for(row, 17), "stable-row-id")
        encoded = {"metadata": json.dumps({"uuid": "encoded-row-id"})}
        self.assertEqual(source_id_for(encoded, 17), "encoded-row-id")

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

    def test_runtime_linear_name_maps_to_checkpoint(self) -> None:
        self.assertEqual(
            checkpoint_weight_name("model.layers.12.mlp.down_proj"),
            "model.language_model.layers.12.mlp.down_proj.weight",
        )

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
