#!/usr/bin/env python3
"""Collect diagonal activation statistics for Q2/Q4 sensitivity scoring.

Only channel-wise sums of squares are retained. Source prompts and token-level
activations never enter the output artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from mtp_teacher import (
    forward_mtp_activations,
    load_mtp_teacher,
    mtp_checkpoint_weight_name,
)
from prompt_format import render_record
from run_ledger import GpuRun, require_budget
from teacher_runtime import (
    cuda_memory_evidence,
    install_pinned_fla_kernel,
    reset_cuda_memory_peaks,
    weight_max_memory,
)


QUANTIZED_DTYPES = frozenset({"q2_b64", "q4_b64", "mixed_q2_q4_b64"})


def checkpoint_weight_name(module_name: str) -> str:
    if module_name.startswith("model.layers."):
        return "model.language_model." + module_name.removeprefix("model.") + ".weight"
    if module_name == "lm_head":
        return "lm_head.weight"
    return module_name + ".weight"


def quantized_source_names(plan: dict[str, Any]) -> set[str]:
    return {
        entry["name"]
        for entry in plan["tensors"]
        if entry["source_shard"] is not None and entry["dtype"] in QUANTIZED_DTYPES
    }


def prefill_ranges(sequence_length: int, chunk_tokens: int) -> list[tuple[int, int]]:
    """Return complete, ordered causal-prefill ranges for one sequence."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if chunk_tokens < 0:
        raise ValueError("chunk_tokens must be non-negative")
    effective = chunk_tokens if 0 < chunk_tokens < sequence_length else sequence_length
    return [
        (start, min(sequence_length, start + effective))
        for start in range(0, sequence_length, effective)
    ]


def resolve_mtp_device(args: argparse.Namespace, torch: Any, base_model: Any) -> Any:
    """Resolve the explicitly requested MTP device after base weight placement."""

    if args.mtp_device == "auto":
        return next(base_model.layers[-1].parameters()).device
    device = torch.device(args.mtp_device)
    if device.type == "cuda":
        if device.index is None or device.index >= args.gpus:
            raise RuntimeError(
                f"--mtp-device {args.mtp_device} is outside configured GPU count {args.gpus}"
            )
        if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
            raise RuntimeError(f"--mtp-device {args.mtp_device} is unavailable")
    return device


def collect(
    args: argparse.Namespace,
    torch: Any,
    safe_open: Any,
    save_file: Any,
    auto_model: Any,
    auto_tokenizer: Any,
    mtp_cache_type: Any,
) -> None:
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    targets = quantized_source_names(plan)
    tokenizer = auto_tokenizer.from_pretrained(args.model, revision=args.revision)
    kernel_evidence = install_pinned_fla_kernel() if args.use_fla_kernel else None
    model = auto_model.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        device_map="balanced",
        max_memory=weight_max_memory(
            args.gpus,
            args.gpu_weight_memory_gib,
            args.cpu_offload_memory_gib,
        ),
        low_cpu_mem_usage=True,
    ).eval()
    base_model = getattr(model, "model", None)
    if base_model is None:
        raise RuntimeError("teacher model does not expose its base model")
    mtp_device = resolve_mtp_device(args, torch, base_model)
    mtp_model = load_mtp_teacher(model, Path(args.model), mtp_device, safe_open)
    reset_cuda_memory_peaks(torch, args.gpus)

    accumulators: dict[str, dict[str, Any]] = {}
    hooks = []
    for module_name, module in model.named_modules():
        checkpoint_name = checkpoint_weight_name(module_name)
        if (
            checkpoint_name not in targets
            or checkpoint_name == "lm_head.weight"
            or not isinstance(module, torch.nn.Linear)
        ):
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

    for module_name, module in mtp_model.named_modules():
        checkpoint_name = mtp_checkpoint_weight_name(module_name)
        if checkpoint_name not in targets or not isinstance(module, torch.nn.Linear):
            continue
        if checkpoint_name in accumulators:
            raise RuntimeError(f"duplicate activation module for {checkpoint_name}")
        accumulators[checkpoint_name] = {
            "input_sum_sq": None,
            "output_sum_sq": None,
            "count": 0,
        }

        def capture_mtp(
            _module: Any,
            inputs: tuple[Any, ...],
            output: Any,
            name: str = checkpoint_name,
        ) -> None:
            values = inputs[0]
            result = output[0] if isinstance(output, tuple) else output
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

        hooks.append(module.register_forward_hook(capture_mtp))

    lm_head_name = "lm_head.weight"
    lm_head_state = None
    if lm_head_name in targets:
        lm_head_state = {"input_sum_sq": None, "count": 0}

    embedding_name = "model.language_model.embed_tokens.weight"
    embedding_row_count = None
    if embedding_name in targets:
        embedding_row_count = torch.zeros(
            model.get_input_embeddings().num_embeddings,
            dtype=torch.int64,
        )

    special_targets = {
        name
        for name, enabled in (
            (lm_head_name, lm_head_state is not None),
            (embedding_name, embedding_row_count is not None),
        )
        if enabled
    }
    missing_modules = targets - set(accumulators) - special_targets
    if missing_modules:
        examples = ", ".join(sorted(missing_modules)[:10])
        print(
            f"warning: {len(missing_modules)} quantized tensors are not torch.nn.Linear modules: {examples}",
            file=sys.stderr,
        )

    sample_ids = []
    total_tokens = 0
    available_samples = sum(1 for line in args.input.open(encoding="utf-8") if line.strip())
    if args.start_sample >= available_samples:
        raise SystemExit(
            f"--start-sample {args.start_sample} is outside {available_samples} input samples"
        )
    total_samples = min(args.max_samples or available_samples, available_samples - args.start_sample)
    seen_samples = 0
    with args.input.open(encoding="utf-8") as source, torch.inference_mode():
        for line in source:
            if not line.strip():
                continue
            if seen_samples < args.start_sample:
                seen_samples += 1
                continue
            sample_number = len(sample_ids) + 1
            if sample_number > total_samples:
                break
            record = json.loads(line)
            prompt = render_record(tokenizer, record)
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=False,
                add_special_tokens=False,
            )
            if encoded.input_ids.shape[-1] > args.max_length:
                raise RuntimeError(
                    f"sample {record['id']} has {encoded.input_ids.shape[-1]} tokens, above "
                    f"--max-length {args.max_length}; activation collection never truncates records"
                )
            sequence_length = int(encoded.input_ids.shape[-1])
            input_ids = encoded.input_ids.to(model.device)
            attention_mask = encoded.attention_mask.to(model.device)
            if embedding_row_count is not None:
                embedding_row_count += torch.bincount(
                    input_ids.detach().reshape(-1).cpu(),
                    minlength=embedding_row_count.numel(),
                )
            ranges = prefill_ranges(sequence_length, args.prefill_chunk_tokens)
            use_chunk_cache = len(ranges) > 1
            base_cache = None
            mtp_cache = mtp_cache_type(config=mtp_model.config)
            for chunk_start, chunk_stop in ranges:
                outputs = base_model(
                    input_ids=input_ids[:, chunk_start:chunk_stop],
                    attention_mask=attention_mask[:, :chunk_stop],
                    past_key_values=base_cache,
                    use_cache=use_chunk_cache,
                    return_dict=True,
                )
                if use_chunk_cache:
                    base_cache = outputs.past_key_values
                    if base_cache is None:
                        raise RuntimeError("chunked activation forward returned no cache state")
                hidden_states = outputs.last_hidden_state
                if lm_head_state is not None:
                    reduced = hidden_states.float().square().sum(dim=(0, 1)).detach()
                    lm_head_state["input_sum_sq"] = (
                        reduced
                        if lm_head_state["input_sum_sq"] is None
                        else lm_head_state["input_sum_sq"] + reduced
                    )
                    lm_head_state["count"] += hidden_states.numel() // hidden_states.shape[-1]

                # MTP position p consumes main-model hidden p-1. For a base
                # chunk [start, stop), pair all hidden positions that still
                # have a following ground-truth token, and retain its own KV
                # cache across chunks.
                mtp_hidden_stop = min(chunk_stop, sequence_length - 1)
                if chunk_start < mtp_hidden_stop:
                    mtp_length = mtp_hidden_stop - chunk_start
                    mtp_input_ids = input_ids[
                        :, chunk_start + 1 : mtp_hidden_stop + 1
                    ]
                    mtp_hidden = hidden_states[:, :mtp_length, :].to(mtp_device)
                    mtp_position_ids = torch.arange(
                        chunk_start + 1,
                        mtp_hidden_stop + 1,
                        device=mtp_device,
                        dtype=torch.long,
                    ).unsqueeze(0)
                    mtp_output = forward_mtp_activations(
                        mtp_model,
                        mtp_input_ids,
                        mtp_hidden,
                        mtp_position_ids,
                        mtp_cache,
                    )
                    del mtp_output, mtp_input_ids, mtp_hidden, mtp_position_ids
                del outputs, hidden_states
            del base_cache, mtp_cache
            sample_ids.append(record["id"])
            total_tokens += sequence_length
            if sample_number % args.progress_every == 0 or sample_number == total_samples:
                print(
                    f"[{sample_number}/{total_samples}] activation tokens={total_tokens}",
                    flush=True,
                )

    tensors = {}
    observed = 0
    for name, state in accumulators.items():
        if state["count"] == 0:
            continue
        observed += 1
        tensors[f"{name}.input_mean_sq"] = (state["input_sum_sq"] / state["count"]).cpu()
        tensors[f"{name}.output_mean_sq"] = (state["output_sum_sq"] / state["count"]).cpu()
        tensors[f"{name}.token_count"] = torch.tensor([state["count"]], dtype=torch.int64)
    input_only_tensors = []
    if lm_head_state is not None and lm_head_state["count"]:
        observed += 1
        input_only_tensors.append(lm_head_name)
        tensors[f"{lm_head_name}.input_mean_sq"] = (
            lm_head_state["input_sum_sq"] / lm_head_state["count"]
        ).cpu()
        tensors[f"{lm_head_name}.token_count"] = torch.tensor(
            [lm_head_state["count"]], dtype=torch.int64
        )
    row_frequency_tensors = []
    if embedding_row_count is not None and int(embedding_row_count.sum()) > 0:
        observed += 1
        row_frequency_tensors.append(embedding_name)
        tensors[f"{embedding_name}.row_count"] = embedding_row_count
    for hook in hooks:
        hook.remove()
    observed_names = {name for name, state in accumulators.items() if state["count"]}
    observed_names.update(input_only_tensors)
    observed_names.update(row_frequency_tensors)
    unobserved = sorted(targets - observed_names)
    cuda_memory = cuda_memory_evidence(torch, args.gpus)
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
            "unobserved_tensors": json.dumps(unobserved, separators=(",", ":")),
            "input_only_tensors": json.dumps(input_only_tensors, separators=(",", ":")),
            "row_frequency_tensors": json.dumps(row_frequency_tensors, separators=(",", ":")),
            "fla_kernel": json.dumps(kernel_evidence, separators=(",", ":")),
            "max_length": str(args.max_length),
            "start_sample": str(args.start_sample),
            "max_samples": str(args.max_samples),
            "gpu_weight_memory_gib": str(args.gpu_weight_memory_gib),
            "cpu_offload_memory_gib": str(args.cpu_offload_memory_gib),
            "mtp_device": str(mtp_device),
            "prefill_chunk_tokens": str(args.prefill_chunk_tokens),
            "device_map": json.dumps(model.hf_device_map, sort_keys=True, separators=(",", ":")),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "pytorch_cuda_alloc_conf": str(os.environ.get("PYTORCH_CUDA_ALLOC_CONF")),
            "cuda_memory": json.dumps(cuda_memory, separators=(",", ":")),
        },
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "samples": len(sample_ids),
                "tokens": total_tokens,
                "observed_modules": observed,
                "unobserved_targets": len(unobserved),
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
    parser.add_argument("--start-sample", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--use-fla-kernel", action="store_true")
    parser.add_argument("--gpu-weight-memory-gib", type=int)
    parser.add_argument("--cpu-offload-memory-gib", type=int, default=96)
    parser.add_argument("--mtp-device", default="auto")
    parser.add_argument("--prefill-chunk-tokens", type=int, default=0)
    args = parser.parse_args()
    if args.max_length <= 0:
        raise SystemExit("--max-length must be positive")
    if args.start_sample < 0:
        raise SystemExit("--start-sample must be non-negative")
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")
    if args.progress_every <= 0:
        raise SystemExit("--progress-every must be positive")
    if args.gpu_weight_memory_gib is not None and args.gpu_weight_memory_gib <= 0:
        raise SystemExit("--gpu-weight-memory-gib must be positive")
    if args.cpu_offload_memory_gib <= 0:
        raise SystemExit("--cpu-offload-memory-gib must be positive")
    if args.prefill_chunk_tokens < 0:
        raise SystemExit("--prefill-chunk-tokens must be non-negative")
    require_budget(args.ledger, args.reserved_gpu_hours)
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.cache_utils import MtpCache
    except ImportError as error:
        raise SystemExit("install training/requirements.in before collecting activations") from error
    with GpuRun(args.ledger, "activation-statistics", args.gpus, sys.argv):
        collect(
            args,
            torch,
            safe_open,
            save_file,
            AutoModelForCausalLM,
            AutoTokenizer,
            MtpCache,
        )


if __name__ == "__main__":
    main()
