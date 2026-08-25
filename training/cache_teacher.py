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
from typing import Any

from prompt_format import render_record
from run_ledger import GpuRun, require_budget


def transformer_layers(model: Any) -> Any:
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if base is None or layers is None or not hasattr(model, "lm_head"):
        raise RuntimeError("teacher model does not expose model.layers and lm_head")
    return base, layers


def sparse_targets(torch: Any, model: Any, hidden: Any, top_k: int, chunk_size: int) -> tuple[Any, Any, Any]:
    indices = []
    log_probabilities = []
    residuals = []
    for start in range(0, hidden.shape[1], chunk_size):
        stop = min(start + chunk_size, hidden.shape[1])
        logits = model.lm_head(hidden[:, start:stop]).float()
        log_normalizer = torch.logsumexp(logits, dim=-1)
        top_values, top_indices = torch.topk(logits, top_k, dim=-1)
        top_log_probabilities = top_values - log_normalizer.unsqueeze(-1)
        residual = (1.0 - top_log_probabilities.exp().sum(dim=-1)).clamp_min(0.0)
        indices.append(top_indices.cpu().to(torch.int32))
        log_probabilities.append(top_log_probabilities.cpu().to(torch.bfloat16))
        residuals.append(residual.cpu().to(torch.float32))
        del logits, log_normalizer, top_values, top_indices, top_log_probabilities, residual
    return (
        torch.cat(indices, dim=1),
        torch.cat(log_probabilities, dim=1),
        torch.cat(residuals, dim=1),
    )


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
    parser.add_argument("--logit-chunk", type=int, default=64)
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
    if args.logit_chunk <= 0:
        raise SystemExit("--logit-chunk must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        low_cpu_mem_usage=True,
    ).eval()
    base_model, layers = transformer_layers(model)
    if any(layer < 0 or layer >= len(layers) for layer in hidden_layers):
        raise SystemExit(f"hidden layer indices must be in [0, {len(layers) - 1}]")
    captured: dict[int, Any] = {}
    hooks = []
    for layer_index in hidden_layers:
        def capture(_module: Any, _inputs: Any, output: Any, index: int = layer_index) -> None:
            captured[index] = output[0] if isinstance(output, tuple) else output

        hooks.append(layers[layer_index].register_forward_hook(capture))
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
            prompt = render_record(tokenizer, record)
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )
            input_ids = encoded.input_ids.to(model.device)
            attention_mask = encoded.attention_mask.to(model.device)
            captured.clear()
            with torch.inference_mode():
                output = base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                top_indices, top_values, residual = sparse_targets(
                    torch, model, output.last_hidden_state, args.top_k, args.logit_chunk
                )
            tensors = {
                "input_ids": input_ids.cpu().to(torch.int32),
                "attention_mask": attention_mask.cpu().to(torch.uint8),
                "topk_indices": top_indices,
                "topk_logprobs": top_values,
                "residual_probability": residual,
            }
            for layer in hidden_layers:
                if layer not in captured:
                    raise RuntimeError(f"hidden layer hook {layer} did not run")
                tensors[f"hidden_{layer}"] = captured[layer].cpu().to(torch.bfloat16)
            filename = f"{sample_id}.safetensors"
            save_file(
                tensors,
                args.output / filename,
                metadata={
                    "sample_id": sample_id,
                    "teacher_model": args.model,
                    "teacher_revision": str(resolved_revision),
                    "source_payload_sha256": str(record["prompt_sha256"]),
                    "rendered_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
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
                        "source_payload_sha256": record["prompt_sha256"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            del output, tensors, top_indices, top_values, residual
    for hook in hooks:
        hook.remove()


if __name__ == "__main__":
    main()
