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
import os
import sys
from pathlib import Path
from typing import Any

from mtp_teacher import forward_mtp_activations, load_mtp_teacher
from prompt_format import normalize_messages, render_record
from run_ledger import GpuRun, require_budget
from teacher_runtime import (
    cuda_memory_evidence,
    install_pinned_fla_kernel,
    reset_cuda_memory_peaks,
    weight_max_memory,
)


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_local_model_provenance(
    model_path: Path,
    requested_revision: str,
    provenance_path: Path | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if not model_path.is_dir():
        if provenance_path is not None:
            raise ValueError("--local-model-provenance requires a local --model directory")
        return None, None
    if provenance_path is None:
        raise ValueError("local teacher paths require --local-model-provenance")
    encoded = provenance_path.read_bytes()
    document = json.loads(encoded)
    if document.get("format") != "ctox.verified-local-model.v1":
        raise ValueError("unsupported local model provenance format")
    if document.get("revision") != requested_revision:
        raise ValueError(
            f"local model provenance revision {document.get('revision')} does not match "
            f"requested {requested_revision}"
        )
    if Path(document.get("local_root", "")).resolve() != model_path.resolve():
        raise ValueError("local model provenance root does not match --model")
    files = document.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("local model provenance contains no verified files")
    for entry in files:
        path = model_path / entry["name"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"verified local model file changed size: {entry['name']}")
    return document, hashlib.sha256(encoded).hexdigest()


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


def assistant_prefix(tokenizer: Any, record: dict[str, Any], rendered: str) -> str:
    messages = normalize_messages(record.get("messages", []))
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("assistant target mode requires a final assistant message")
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if record.get("tools"):
        kwargs["tools"] = record["tools"]
    prefix = tokenizer.apply_chat_template(messages[:-1], **kwargs)
    if not rendered.startswith(prefix):
        raise ValueError("assistant generation prefix is not an exact prefix of rendered record")
    return prefix


def position_sets(
    sequence_length: int,
    target_mode: str,
    assistant_prefix_tokens: int | None,
    marker_offsets: list[int],
    marker_window: int,
    uniform_hidden_positions: int,
    assistant_hidden_positions: int = 64,
) -> tuple[list[int], list[int]]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if target_mode == "all":
        positions = list(range(sequence_length))
        return positions, positions
    if target_mode != "assistant":
        raise ValueError(f"unsupported target mode {target_mode}")
    if assistant_prefix_tokens is None or not 1 <= assistant_prefix_tokens < sequence_length:
        raise ValueError("assistant prefix must end inside the rendered sequence")

    # Hidden position p predicts input token p+1. Start at the final prompt
    # token so the first assistant token is supervised; exclude the final
    # sequence position because it has no next-token target in this record.
    logit_positions = list(range(assistant_prefix_tokens - 1, sequence_length - 1))
    if assistant_hidden_positions <= 0:
        raise ValueError("assistant_hidden_positions must be positive")
    if len(logit_positions) <= assistant_hidden_positions:
        hidden_positions = set(logit_positions)
    elif assistant_hidden_positions == 1:
        hidden_positions = {logit_positions[len(logit_positions) // 2]}
    else:
        hidden_positions = {
            logit_positions[
                round(
                    index
                    * (len(logit_positions) - 1)
                    / (assistant_hidden_positions - 1)
                )
            ]
            for index in range(assistant_hidden_positions)
        }
    for offset in marker_offsets:
        if not 0 <= offset < sequence_length:
            raise ValueError(f"marker offset {offset} is outside sequence")
        start = max(0, offset - marker_window)
        stop = min(sequence_length, offset + marker_window + 1)
        hidden_positions.update(range(start, stop))
    if uniform_hidden_positions == 1:
        hidden_positions.add(sequence_length // 2)
    elif uniform_hidden_positions > 1:
        hidden_positions.update(
            round(index * (sequence_length - 1) / (uniform_hidden_positions - 1))
            for index in range(uniform_hidden_positions)
        )
    return logit_positions, sorted(hidden_positions)


def mtp_target_positions(sequence_length: int, logit_positions: list[int]) -> list[int]:
    """Return base-hidden positions whose MTP draft predicts a recorded token."""

    if sequence_length < 3:
        raise ValueError("MTP targets require at least three sequence tokens")
    if logit_positions != sorted(set(logit_positions)):
        raise ValueError("logit positions must be sorted and unique")
    if any(position < 0 or position >= sequence_length - 1 for position in logit_positions):
        raise ValueError("logit position is outside the next-token prediction range")
    # Base hidden p predicts token p+1. MTP consumes that hidden plus token p+1
    # and drafts token p+2, so the final next-token position has no MTP target.
    return [position for position in logit_positions if position < sequence_length - 2]


def resolve_mtp_device(args: argparse.Namespace, torch: Any, base_model: Any) -> Any | None:
    if args.mtp_device is None:
        return None
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


def cache(
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
    try:
        local_provenance, local_provenance_sha256 = validate_local_model_provenance(
            Path(args.model), args.revision, args.local_model_provenance
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    available_samples = sum(1 for line in args.input.open(encoding="utf-8") if line.strip())
    if args.start_sample >= available_samples:
        raise SystemExit(
            f"--start-sample {args.start_sample} is outside {available_samples} input samples"
        )
    selected_samples = min(
        args.max_samples or available_samples,
        available_samples - args.start_sample,
    )
    args.output.mkdir(parents=True)
    hidden_layers = [int(layer) for layer in args.hidden_layers.split(",") if layer]
    tokenizer = auto_tokenizer.from_pretrained(args.model, revision=args.revision)
    kernel_evidence = install_pinned_fla_kernel() if args.use_fla_kernel else None
    max_memory = weight_max_memory(
        args.gpus,
        args.gpu_weight_memory_gib,
        args.cpu_offload_memory_gib,
    )
    model = auto_model.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        device_map="balanced",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    ).eval()
    base_model, layers = transformer_layers(model)
    mtp_device = resolve_mtp_device(args, torch, base_model)
    mtp_model = (
        load_mtp_teacher(model, Path(args.model), mtp_device, safe_open)
        if mtp_device is not None
        else None
    )
    if any(layer < 0 or layer >= len(layers) for layer in hidden_layers):
        raise SystemExit(f"hidden layer indices must be in [0, {len(layers) - 1}]")
    captured: dict[int, list[Any]] = {}
    active_hidden_positions: Any | None = None
    hooks = []
    for layer_index in hidden_layers:

        def capture(_module: Any, _inputs: Any, output: Any, index: int = layer_index) -> None:
            if active_hidden_positions is None:
                raise RuntimeError("hidden-state capture positions were not initialized")
            values = output[0] if isinstance(output, tuple) else output
            selected = values.index_select(1, active_hidden_positions.to(values.device))
            if selected.shape[1]:
                captured.setdefault(index, []).append(
                    selected.detach().cpu().to(torch.bfloat16)
                )

        hooks.append(layers[layer_index].register_forward_hook(capture))
    resolved_revision = (
        local_provenance["revision"]
        if local_provenance is not None
        else getattr(model.config, "_commit_hash", None) or args.revision
    )
    run_manifest = {
        "schema_version": 1,
        "teacher_model": args.model,
        "teacher_revision": str(resolved_revision),
        "local_model_provenance_sha256": local_provenance_sha256,
        "local_model_root_sha256": (
            local_provenance["root_sha256"] if local_provenance is not None else None
        ),
        "architecture": type(model).__name__,
        "dtype": "bfloat16",
        "device_map": {name: str(device) for name, device in model.hf_device_map.items()},
        "top_k": args.top_k,
        "max_length": args.max_length,
        "hidden_layers": hidden_layers,
        "logit_chunk": args.logit_chunk,
        "target_mode": args.target_mode,
        "marker_window": args.marker_window,
        "uniform_hidden_positions": args.uniform_hidden_positions,
        "assistant_hidden_positions": args.assistant_hidden_positions,
        "fla_kernel": kernel_evidence,
        "gpu_weight_memory_gib": args.gpu_weight_memory_gib,
        "cpu_offload_memory_gib": args.cpu_offload_memory_gib,
        "start_sample": args.start_sample,
        "selected_samples": selected_samples,
        "prefill_chunk_tokens": args.prefill_chunk_tokens,
        "mtp_device": str(mtp_device) if mtp_device is not None else None,
        "mtp_targets": mtp_model is not None,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }
    write_json_atomic(args.output / "run.json", run_manifest)
    reset_cuda_memory_peaks(torch, args.gpus)
    index_path = args.output / "index.jsonl"
    seen_samples = 0
    written_samples = 0
    with index_path.open("x", encoding="utf-8") as index, args.input.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            if seen_samples < args.start_sample:
                seen_samples += 1
                continue
            if written_samples >= selected_samples:
                break
            record = json.loads(line)
            sample_id = str(record["id"])
            prompt = render_record(tokenizer, record)
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=False,
                add_special_tokens=False,
            )
            sequence_length = int(encoded.input_ids.shape[-1])
            if sequence_length > args.max_length:
                raise RuntimeError(
                    f"sample {sample_id} has {sequence_length} tokens, above --max-length "
                    f"{args.max_length}; teacher caching never truncates records"
                )
            prefix_tokens = None
            if args.target_mode == "assistant":
                prefix = assistant_prefix(tokenizer, record, prompt)
                prefix_tokens = len(
                    tokenizer(prefix, add_special_tokens=False).input_ids
                )
            logit_position_list, hidden_position_list = position_sets(
                sequence_length,
                args.target_mode,
                prefix_tokens,
                [int(offset) for offset in record.get("marker_token_offsets", [])],
                args.marker_window,
                args.uniform_hidden_positions,
                args.assistant_hidden_positions,
            )
            if not logit_position_list:
                raise RuntimeError(f"sample {sample_id} has no teacher logit positions")
            mtp_position_list = (
                mtp_target_positions(sequence_length, logit_position_list)
                if mtp_model is not None
                else []
            )
            if mtp_model is not None and not mtp_position_list:
                raise RuntimeError(f"sample {sample_id} has no MTP teacher positions")
            mtp_position_set = set(mtp_position_list)
            mtp_hidden_position_list = [
                position
                for position in hidden_position_list
                if position in mtp_position_set
            ]
            input_ids = encoded.input_ids.to(model.device)
            attention_mask = encoded.attention_mask.to(model.device)
            logit_positions = torch.tensor(logit_position_list, dtype=torch.long)
            hidden_positions = torch.tensor(hidden_position_list, dtype=torch.long)
            mtp_positions = torch.tensor(mtp_position_list, dtype=torch.long)
            active_hidden_positions = hidden_positions
            captured.clear()
            selected_final_chunks = []
            selected_mtp_chunks = []
            use_chunk_cache = 0 < args.prefill_chunk_tokens < sequence_length
            chunk_tokens = args.prefill_chunk_tokens if use_chunk_cache else sequence_length
            past_key_values = None
            mtp_cache = mtp_cache_type(config=mtp_model.config) if mtp_model is not None else None
            with torch.inference_mode():
                for chunk_start in range(0, sequence_length, chunk_tokens):
                    chunk_stop = min(sequence_length, chunk_start + chunk_tokens)
                    local_hidden_positions = [
                        position - chunk_start
                        for position in hidden_position_list
                        if chunk_start <= position < chunk_stop
                    ]
                    active_hidden_positions = torch.tensor(
                        local_hidden_positions,
                        dtype=torch.long,
                    )
                    output = base_model(
                        input_ids=input_ids[:, chunk_start:chunk_stop],
                        attention_mask=attention_mask[:, :chunk_stop],
                        past_key_values=past_key_values,
                        use_cache=use_chunk_cache,
                        return_dict=True,
                    )
                    if use_chunk_cache:
                        past_key_values = output.past_key_values
                        if past_key_values is None:
                            raise RuntimeError("chunked teacher forward returned no cache state")
                    local_logit_positions = [
                        position - chunk_start
                        for position in logit_position_list
                        if chunk_start <= position < chunk_stop
                    ]
                    if local_logit_positions:
                        local_positions = torch.tensor(
                            local_logit_positions,
                            dtype=torch.long,
                            device=output.last_hidden_state.device,
                        )
                        selected_final_chunks.append(
                            output.last_hidden_state.index_select(1, local_positions)
                        )
                    if mtp_model is not None:
                        mtp_hidden_stop = min(chunk_stop, sequence_length - 1)
                        if chunk_start < mtp_hidden_stop:
                            mtp_length = mtp_hidden_stop - chunk_start
                            mtp_input_ids = input_ids[
                                :, chunk_start + 1 : mtp_hidden_stop + 1
                            ]
                            mtp_hidden = output.last_hidden_state[:, :mtp_length, :].to(
                                mtp_device
                            )
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
                            local_mtp_positions = [
                                position - chunk_start
                                for position in mtp_position_list
                                if chunk_start <= position < mtp_hidden_stop
                            ]
                            if local_mtp_positions:
                                selected_mtp_chunks.append(
                                    mtp_output.index_select(
                                        1,
                                        torch.tensor(
                                            local_mtp_positions,
                                            dtype=torch.long,
                                            device=mtp_output.device,
                                        ),
                                    )
                                )
                            del mtp_input_ids, mtp_hidden, mtp_position_ids, mtp_output
                    del output
                if not selected_final_chunks:
                    raise RuntimeError(f"sample {sample_id} produced no selected final hidden states")
                selected_final_hidden = torch.cat(selected_final_chunks, dim=1)
                top_indices, top_values, residual = sparse_targets(
                    torch, model, selected_final_hidden, args.top_k, args.logit_chunk
                )
                mtp_hidden_targets = None
                mtp_top_indices = None
                mtp_top_values = None
                mtp_residual = None
                mtp_position_indices = None
                selected_mtp_hidden_positions = None
                if mtp_model is not None:
                    if not selected_mtp_chunks:
                        raise RuntimeError(
                            f"sample {sample_id} produced no selected MTP hidden states"
                        )
                    selected_mtp_hidden = torch.cat(selected_mtp_chunks, dim=1)
                    if selected_mtp_hidden.shape[1] != len(mtp_position_list):
                        raise RuntimeError(
                            f"sample {sample_id} produced {selected_mtp_hidden.shape[1]} MTP "
                            f"targets, expected {len(mtp_position_list)}"
                        )
                    mtp_position_indices = {
                        position: index for index, position in enumerate(mtp_position_list)
                    }
                    selected_mtp_hidden_positions = torch.tensor(
                        [mtp_position_indices[position] for position in mtp_hidden_position_list],
                        dtype=torch.long,
                        device=selected_mtp_hidden.device,
                    )
                    mtp_hidden_targets = (
                        selected_mtp_hidden.index_select(1, selected_mtp_hidden_positions)
                        .detach()
                        .cpu()
                        .to(torch.bfloat16)
                    )
                    # Invoke the Accelerate-wrapped shared head directly.  An
                    # offloaded head deliberately exposes meta parameters
                    # between calls; moving the hidden state to that apparent
                    # device would erase its data before the hook can restore
                    # the real weights and select the execution device.
                    mtp_top_indices, mtp_top_values, mtp_residual = sparse_targets(
                        torch,
                        model,
                        selected_mtp_hidden,
                        args.top_k,
                        args.logit_chunk,
                    )
                    del selected_mtp_hidden
            tensors = {
                "input_ids": input_ids.cpu().to(torch.int32),
                "attention_mask": attention_mask.cpu().to(torch.uint8),
                "logit_positions": logit_positions.to(torch.int32),
                "hidden_positions": hidden_positions.to(torch.int32),
                "topk_indices": top_indices,
                "topk_logprobs": top_values,
                "residual_probability": residual,
            }
            if mtp_hidden_targets is not None:
                tensors.update(
                    {
                        "mtp_positions": mtp_positions.to(torch.int32),
                        "mtp_hidden_positions": torch.tensor(
                            mtp_hidden_position_list,
                            dtype=torch.int32,
                        ),
                        "mtp_hidden": mtp_hidden_targets,
                        "mtp_topk_indices": mtp_top_indices,
                        "mtp_topk_logprobs": mtp_top_values,
                        "mtp_residual_probability": mtp_residual,
                    }
                )
            for layer in hidden_layers:
                if layer not in captured or not captured[layer]:
                    raise RuntimeError(f"hidden layer hook {layer} did not run")
                tensors[f"hidden_{layer}"] = torch.cat(captured[layer], dim=1)
                if tensors[f"hidden_{layer}"].shape[1] != len(hidden_position_list):
                    raise RuntimeError(
                        f"hidden layer {layer} captured {tensors[f'hidden_{layer}'].shape[1]} "
                        f"positions, expected {len(hidden_position_list)}"
                    )
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
                    "target_mode": args.target_mode,
                    "sequence_tokens": str(sequence_length),
                    "logit_target_count": str(len(logit_position_list)),
                    "hidden_target_count": str(len(hidden_position_list)),
                    "mtp_target_count": str(len(mtp_position_list)),
                    "mtp_hidden_target_count": str(len(mtp_hidden_position_list)),
                    "mtp_position_semantics": "base_hidden_p_drafts_token_p_plus_2",
                    "prefill_chunk_tokens": str(chunk_tokens),
                },
            )
            index.write(
                json.dumps(
                    {
                        "id": sample_id,
                        "file": filename,
                        "tokens": sequence_length,
                        "logit_targets": len(logit_position_list),
                        "hidden_targets": len(hidden_position_list),
                        "mtp_targets": len(mtp_position_list),
                        "mtp_hidden_targets": len(mtp_hidden_position_list),
                        "source_line": line_number,
                        "source_payload_sha256": record["prompt_sha256"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            del (
                selected_final_chunks,
                selected_final_hidden,
                tensors,
                top_indices,
                top_values,
                residual,
                past_key_values,
                mtp_cache,
                selected_mtp_chunks,
                mtp_hidden_targets,
                mtp_top_indices,
                mtp_top_values,
                mtp_residual,
                mtp_position_indices,
                selected_mtp_hidden_positions,
            )
            active_hidden_positions = None
            written_samples += 1
    for hook in hooks:
        hook.remove()
    cuda_memory = cuda_memory_evidence(torch, args.gpus)
    if cuda_memory:
        run_manifest["cuda_memory"] = cuda_memory
    run_manifest["written_samples"] = written_samples
    write_json_atomic(args.output / "run.json", run_manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-model-provenance", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=3)
    parser.add_argument("--reserved-gpu-hours", type=float, required=True)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--hidden-layers", default="0,15,31,47,63")
    parser.add_argument("--logit-chunk", type=int, default=64)
    parser.add_argument("--target-mode", choices=("all", "assistant"), default="all")
    parser.add_argument("--marker-window", type=int, default=32)
    parser.add_argument("--uniform-hidden-positions", type=int, default=64)
    parser.add_argument("--assistant-hidden-positions", type=int, default=64)
    parser.add_argument("--start-sample", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--use-fla-kernel", action="store_true")
    parser.add_argument("--gpu-weight-memory-gib", type=int)
    parser.add_argument("--cpu-offload-memory-gib", type=int, default=96)
    parser.add_argument(
        "--mtp-device",
        help="cache resident-MTP hidden and sparse-logit targets on this device (for example cuda:2)",
    )
    parser.add_argument("--prefill-chunk-tokens", type=int, default=0)
    args = parser.parse_args()
    require_budget(args.ledger, args.reserved_gpu_hours)
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    if args.max_length <= 0:
        raise SystemExit("--max-length must be positive")
    if args.logit_chunk <= 0:
        raise SystemExit("--logit-chunk must be positive")
    if args.marker_window < 0 or args.uniform_hidden_positions < 0:
        raise SystemExit("--marker-window and --uniform-hidden-positions must be non-negative")
    if args.assistant_hidden_positions <= 0:
        raise SystemExit("--assistant-hidden-positions must be positive")
    if args.start_sample < 0:
        raise SystemExit("--start-sample must be non-negative")
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")
    if args.gpu_weight_memory_gib is not None and args.gpu_weight_memory_gib <= 0:
        raise SystemExit("--gpu-weight-memory-gib must be positive")
    if args.cpu_offload_memory_gib <= 0:
        raise SystemExit("--cpu-offload-memory-gib must be positive")
    if args.prefill_chunk_tokens < 0:
        raise SystemExit("--prefill-chunk-tokens must be non-negative")

    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.cache_utils import MtpCache
    except ImportError as error:
        raise SystemExit("install training/requirements.in before caching teacher targets") from error

    with GpuRun(args.ledger, "teacher-cache", args.gpus, sys.argv):
        cache(
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
