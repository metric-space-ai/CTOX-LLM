#!/usr/bin/env python3
"""Cache sparse BF16-teacher targets for offline recovery.

Input JSONL records contain `id` and either `messages` or `prompt`. Output is one
safetensors file per sample plus an immutable index. Full-vocabulary logits are
never persisted: top-k log probabilities and their residual mass are sufficient
for the recovery KL term.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from run_ledger import GpuRun, require_budget


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=3)
    parser.add_argument("--reserved-gpu-hours", type=float, required=True)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--hidden-layers", default="0,15,31,47,63")
    args = parser.parse_args()
    require_budget(args.ledger, args.reserved_gpu_hours)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    try:
        import torch
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit("install training/requirements.in before caching teacher targets") from error

    hidden_layers = [int(layer) for layer in args.hidden_layers.split(",") if layer]
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        low_cpu_mem_usage=True,
    ).eval()
    resolved_revision = getattr(model.config, "_commit_hash", None) or args.revision
    index_path = args.output / "index.jsonl"
    with GpuRun(args.ledger, "teacher-cache", args.gpus, sys.argv), index_path.open(
        "x", encoding="utf-8"
    ) as index, args.input.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record["id"])
            if "messages" in record:
                prompt = tokenizer.apply_chat_template(
                    record["messages"], tokenize=False, add_generation_prompt=False
                )
            else:
                prompt = str(record["prompt"])
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )
            input_ids = encoded.input_ids.to(model.device)
            attention_mask = encoded.attention_mask.to(model.device)
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                log_probs = output.logits.float().log_softmax(dim=-1)
                top_values, top_indices = torch.topk(log_probs, args.top_k, dim=-1)
                residual = (1.0 - top_values.exp().sum(dim=-1)).clamp_min(0.0)
            tensors = {
                "input_ids": input_ids.cpu().to(torch.int32),
                "attention_mask": attention_mask.cpu().to(torch.uint8),
                "topk_indices": top_indices.cpu().to(torch.int32),
                "topk_logprobs": top_values.cpu().to(torch.bfloat16),
                "residual_probability": residual.cpu().to(torch.float32),
            }
            for layer in hidden_layers:
                if layer >= len(output.hidden_states):
                    raise RuntimeError(
                        f"hidden layer {layer} unavailable; model returned {len(output.hidden_states)} states"
                    )
                tensors[f"hidden_{layer}"] = output.hidden_states[layer].cpu().to(torch.bfloat16)
            filename = f"{sample_id}.safetensors"
            save_file(
                tensors,
                args.output / filename,
                metadata={
                    "sample_id": sample_id,
                    "teacher_model": args.model,
                    "teacher_revision": str(resolved_revision),
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "top_k": str(args.top_k),
                },
            )
            index.write(
                json.dumps(
                    {
                        "id": sample_id,
                        "file": filename,
                        "tokens": int(input_ids.shape[-1]),
                        "source_line": line_number,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
