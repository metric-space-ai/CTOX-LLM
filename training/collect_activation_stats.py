#!/usr/bin/env python3
"""Collect diagonal activation statistics for Q2/Q4 sensitivity scoring.

Only channel-wise sums of squares are retained. Source prompts and token-level
activations never enter the output artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from prompt_format import render_record
from run_ledger import GpuRun, require_budget


def checkpoint_weight_name(module_name: str) -> str:
    if module_name.startswith("model.layers."):
        return "model.language_model." + module_name.removeprefix("model.") + ".weight"
    if module_name == "lm_head":
        return "lm_head.weight"
    return module_name + ".weight"


def collect(args: argparse.Namespace, torch: Any, save_file: Any, auto_model: Any, auto_tokenizer: Any) -> None:
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    targets = {
        entry["name"]
        for entry in plan["tensors"]
        if entry["source_shard"] is not None and entry["dtype"] in {"q2_b64", "q4_b64"}
    }
    tokenizer = auto_tokenizer.from_pretrained(args.model, revision=args.revision)
    model = auto_model.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        device_map="balanced",
        low_cpu_mem_usage=True,
    ).eval()
    base_model = getattr(model, "model", None)
    if base_model is None:
        raise RuntimeError("teacher model does not expose its base model")

    accumulators: dict[str, dict[str, Any]] = {}
    hooks = []
    for module_name, module in model.named_modules():
        checkpoint_name = checkpoint_weight_name(module_name)
        if checkpoint_name not in targets or not isinstance(module, torch.nn.Linear):
            continue
        accumulators[checkpoint_name] = {
            "input_sum_sq": None,
            "output_sum_sq": None,
            "count": 0,
        }

        def capture(
            _module: Any,
            inputs: tuple[Any, ...],
            output: Any,
            name: str = checkpoint_name,
        ) -> None:
            values = inputs[0]
            result = output[0] if isinstance(output, tuple) else output
            if values.ndim < 2 or result.ndim < 2:
                raise RuntimeError(f"linear module {name} did not receive token-channel tensors")
            reduce_input = tuple(range(values.ndim - 1))
            reduce_output = tuple(range(result.ndim - 1))
            input_sum_sq = values.float().square().sum(dim=reduce_input).detach()
            output_sum_sq = result.float().square().sum(dim=reduce_output).detach()
            count = values.numel() // values.shape[-1]
            state = accumulators[name]
            state["input_sum_sq"] = (
                input_sum_sq
                if state["input_sum_sq"] is None
                else state["input_sum_sq"] + input_sum_sq
            )
            state["output_sum_sq"] = (
                output_sum_sq
                if state["output_sum_sq"] is None
                else state["output_sum_sq"] + output_sum_sq
            )
            state["count"] += count

        hooks.append(module.register_forward_hook(capture))

    missing_modules = targets - set(accumulators)
    if missing_modules:
        examples = ", ".join(sorted(missing_modules)[:10])
        print(
            f"warning: {len(missing_modules)} quantized tensors are not torch.nn.Linear modules: {examples}",
            file=sys.stderr,
        )

    sample_ids = []
    total_tokens = 0
    with args.input.open(encoding="utf-8") as source, torch.inference_mode():
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = render_record(tokenizer, record)
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )
            input_ids = encoded.input_ids.to(model.device)
            attention_mask = encoded.attention_mask.to(model.device)
            base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            sample_ids.append(record["id"])
            total_tokens += int(input_ids.shape[-1])

    tensors = {}
    observed = 0
    for name, state in accumulators.items():
        if state["count"] == 0:
            continue
        observed += 1
        tensors[f"{name}.input_mean_sq"] = (state["input_sum_sq"] / state["count"]).cpu()
        tensors[f"{name}.output_mean_sq"] = (state["output_sum_sq"] / state["count"]).cpu()
        tensors[f"{name}.token_count"] = torch.tensor([state["count"]], dtype=torch.int64)
    for hook in hooks:
        hook.remove()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        args.output,
        metadata={
            "format": "ctox.activation-diagonal.v1",
            "model": args.model,
            "revision": args.revision,
            "sample_ids": json.dumps(sample_ids, separators=(",", ":")),
            "samples": str(len(sample_ids)),
            "tokens": str(total_tokens),
            "observed_modules": str(observed),
            "target_tensors": str(len(targets)),
        },
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "samples": len(sample_ids),
                "tokens": total_tokens,
                "observed_modules": observed,
                "non_linear_targets": len(missing_modules),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=3)
    parser.add_argument("--reserved-gpu-hours", type=float, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    if args.max_length <= 0:
        raise SystemExit("--max-length must be positive")
    require_budget(args.ledger, args.reserved_gpu_hours)
    try:
        import torch
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit("install training/requirements.in before collecting activations") from error
    with GpuRun(args.ledger, "activation-statistics", args.gpus, sys.argv):
        collect(args, torch, save_file, AutoModelForCausalLM, AutoTokenizer)


if __name__ == "__main__":
    main()
