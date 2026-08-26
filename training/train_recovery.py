#!/usr/bin/env python3
"""Train every packed Qwen Q2/Q4 recovery scale against verified BF16 targets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from ctox_artifact import CtoxArtifact
from cache_teacher import validate_local_model_provenance
from end_to_end_recovery import (
    build_packed_student,
    export_scale_tensors,
    immutable_run_contract,
    load_teacher_file,
    optimizer_parameters,
    restore_training_checkpoint,
    save_training_checkpoint,
    scale_regularization,
    scale_tensor_root,
    sha256_path,
    unique_scale_parameters,
    validate_scale_parameter_contract,
)
from fanout_recovery import INDEPENDENT_POLICY, POLICIES
from recovery_io import atomic_json, durable_replace, prepare_output_transaction
from recovery_training_state import (
    normalize_accumulated_gradients,
    recovery_training_status,
    resolve_sample_indices,
    training_order,
)
from run_ledger import GpuRun, require_budget
from teacher_cache_dataset import VerifiedTeacherCache
from teacher_runtime import install_pinned_fla_kernel


def parse_loss_weights(encoded: str) -> dict[str, float]:
    try:
        document = json.loads(encoded)
        if not isinstance(document, dict):
            raise ValueError("loss weights must be a JSON object")
        return {str(name): float(value) for name, value in document.items()}
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"invalid --loss-weights: {error}") from error


def mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    names = set(metrics[0])
    if any(set(item) != names for item in metrics):
        raise ValueError("recovery metric families changed during the run")
    return {
        name: sum(item[name] for item in metrics) / len(metrics)
        for name in sorted(names)
    }


def validate_artifact_recovery(artifact: CtoxArtifact) -> dict[str, Any]:
    recovery = artifact.manifest.get("recovery")
    if not isinstance(recovery, dict) or not recovery.get("fixed_logical_qcodes"):
        raise ValueError("input CTOX artifact does not bind fixed logical qcodes")
    required_hashes = ("plan_sha256", "activation_stats_sha256", "report_sha256")
    for key in required_hashes:
        value = str(recovery.get(key, ""))
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"input CTOX recovery {key} is not a lowercase SHA-256")
    return recovery


def write_scales(
    output: Path,
    tensors: dict[str, Any],
    metadata: dict[str, str],
) -> None:
    from safetensors.torch import save_file

    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    save_file(tensors, temporary, metadata=metadata)
    durable_replace(temporary, output)


def validate_arguments(args: argparse.Namespace) -> None:
    prepare_output_transaction(
        args.output_scales,
        args.output_report,
        args.output_evidence,
        args.resume_checkpoint,
    )
    if args.epochs <= 0 or args.gpus <= 0:
        raise ValueError("--epochs and --gpus must be positive")
    positive = (
        args.learning_rate,
        args.gradient_accumulation,
        args.gradient_clip,
        args.rows_per_chunk,
        args.logit_chunk,
        args.checkpoint_every,
    )
    if any(float(value) <= 0 for value in positive):
        raise ValueError(
            "learning, accumulation, clipping, chunk, and checkpoint values must be positive"
        )
    if (
        args.scale_regularization < 0
        or args.max_sequence_tokens < 0
        or args.prefill_chunk_tokens < 0
    ):
        raise ValueError(
            "regularization, maximum sequence length, and prefill chunk size must be non-negative"
        )
    if args.prefill_chunk_tokens and args.gradient_checkpointing:
        raise ValueError(
            "--prefill-chunk-tokens and --gradient-checkpointing are mutually exclusive"
        )
    if args.max_optimizer_steps is not None and args.max_optimizer_steps <= 0:
        raise ValueError("--max-optimizer-steps must be positive")
    if args.sample_limit is not None and args.sample_limit <= 0:
        raise ValueError("--sample-limit must be positive")
    if args.sample_ids and args.sample_limit is not None:
        raise ValueError("--sample-id and --sample-limit cannot be combined")
    if args.sample_ids and len(args.sample_ids) != len(set(args.sample_ids)):
        raise ValueError("--sample-id values must be unique")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-model-provenance", type=Path, required=True)
    parser.add_argument("--teacher-cache-set", type=Path, required=True)
    parser.add_argument("--teacher-cache-set-sha256", required=True)
    parser.add_argument("--output-scales", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--reserved-gpu-hours", type=float, required=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--compute-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument(
        "--sample-id",
        dest="sample_ids",
        action="append",
        help="train only the exact verified cache identity; repeatable",
    )
    parser.add_argument("--max-sequence-tokens", type=int, default=0)
    parser.add_argument(
        "--prefill-chunk-tokens",
        type=int,
        default=0,
        help=(
            "stateful truncated-BPTT chunk size for long-context recovery; "
            "zero keeps the monolithic correctness path"
        ),
    )
    parser.add_argument("--oversize-policy", choices=("fail", "skip"), default="fail")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--scale-regularization", type=float, default=1e-4)
    parser.add_argument("--rows-per-chunk", type=int, default=128)
    parser.add_argument("--logit-chunk", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--loss-weights", default="{}")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--use-fla-kernel", action="store_true")
    parser.add_argument(
        "--fanout-s-in-policy",
        choices=POLICIES,
        default=INDEPENDENT_POLICY,
        help=(
            "tie same-input projection corrections for a named, run-bound "
            "Qwen fan-out ablation; independent remains the quality baseline"
        ),
    )
    args = parser.parse_args()
    require_budget(args.ledger, args.reserved_gpu_hours)
    try:
        validate_arguments(args)
        import torch
        from safetensors.torch import save_file  # noqa: F401
    except ImportError as error:
        raise SystemExit("install training/requirements.in before recovery") from error
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA recovery requested but unavailable")
    try:
        loss_weights = parse_loss_weights(args.loss_weights)
        cache = VerifiedTeacherCache.from_manifest(
            args.teacher_cache_set,
            args.teacher_cache_set_sha256,
        )
        hidden_layers = [int(value) for value in cache.settings["hidden_layers"]]
        top_k = int(cache.settings["top_k"])
        selected_indices = resolve_sample_indices(
            (str(item["id"]) for item in cache.artifacts),
            args.sample_ids,
        )
        if not bool(cache.settings["mtp_targets"]):
            raise ValueError("end-to-end recovery requires verified MTP targets")
        artifact_sha256 = sha256_path(args.artifact)
        _local_provenance, local_model_provenance_sha256 = (
            validate_local_model_provenance(
                Path(args.model_source),
                args.revision,
                args.local_model_provenance,
            )
        )
        if local_model_provenance_sha256 is None:
            raise ValueError("recovery requires a verified local BF16 model source")
        if cache.teacher_provenance_sha256 != local_model_provenance_sha256:
            raise ValueError(
                "recovery BF16 model provenance differs from the teacher cache"
            )
        with CtoxArtifact(args.artifact, verify_tensors=True) as artifact:
            if artifact.manifest.get("revision") != args.revision:
                raise ValueError("native artifact and requested model revision differ")
            recovery = validate_artifact_recovery(artifact)
            device = torch.device(args.device)
            compute_dtype = getattr(torch, args.compute_dtype)
            kernel_evidence = (
                install_pinned_fla_kernel() if args.use_fla_kernel else None
            )
            runtime, base_evidence, mtp_evidence, fanout_evidence = build_packed_student(
                args.model_source,
                args.revision,
                artifact,
                device,
                compute_dtype,
                args.rows_per_chunk,
                hidden_layers,
                top_k,
                args.logit_chunk,
                args.gradient_checkpointing,
                args.fanout_s_in_policy,
                torch,
            )
            parameters = unique_scale_parameters(
                runtime.main_model,
                runtime.mtp_model,
                fanout_evidence,
            )
            validate_scale_parameter_contract(parameters, artifact)
            trainable = optimizer_parameters(parameters)
            optimizer = torch.optim.AdamW(
                trainable,
                lr=args.learning_rate,
                weight_decay=0.0,
            )
            run_contract = {
                "format": "ctox.recovery.training-run.v1",
                "model": artifact.manifest["model"],
                "revision": args.revision,
                "local_model_provenance_sha256": local_model_provenance_sha256,
                "artifact_sha256": artifact_sha256,
                "teacher_cache_set_sha256": args.teacher_cache_set_sha256,
                "teacher_artifact_root_sha256": cache.manifest()[
                    "artifact_root_sha256"
                ],
                "hidden_layers": hidden_layers,
                "top_k": top_k,
                "mtp_targets": True,
                "epochs": args.epochs,
                "max_optimizer_steps": args.max_optimizer_steps,
                "sample_limit": args.sample_limit,
                "sample_ids": sorted(args.sample_ids) if args.sample_ids else None,
                "max_sequence_tokens": args.max_sequence_tokens,
                "prefill_chunk_tokens": args.prefill_chunk_tokens,
                "oversize_policy": args.oversize_policy,
                "learning_rate": args.learning_rate,
                "gradient_accumulation": args.gradient_accumulation,
                "gradient_clip": args.gradient_clip,
                "scale_regularization": args.scale_regularization,
                "rows_per_chunk": args.rows_per_chunk,
                "logit_chunk": args.logit_chunk,
                "seed": args.seed,
                "loss_weights": loss_weights,
                "gradient_checkpointing": args.gradient_checkpointing,
                "compute_dtype": args.compute_dtype,
                "use_fla_kernel": args.use_fla_kernel,
                "fanout_s_in_policy": args.fanout_s_in_policy,
                "fanout_group_sha256": fanout_evidence["group_sha256"],
                "fixed_logical_qcodes": True,
            }
            _contract_json, run_contract_sha256 = immutable_run_contract(run_contract)
            cursor = {
                "epoch": 0,
                "next_position": 0,
                "optimizer_steps": 0,
                "samples_seen": 0,
            }
            if args.resume_checkpoint is not None:
                cursor = restore_training_checkpoint(
                    args.resume_checkpoint,
                    parameters,
                    optimizer,
                    run_contract_sha256,
                    torch,
                )
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            optimizer.zero_grad(set_to_none=True)
            started = time.monotonic()
            recent_metrics: list[dict[str, float]] = []
            skipped: list[dict[str, Any]] = []
            accumulated = 0
            bounded_stop = False
            budget_stop = False
            with GpuRun(
                args.ledger,
                "end-to-end-recovery",
                args.gpus,
                sys.argv,
                maximum_gpu_hours=args.reserved_gpu_hours,
            ) as gpu_run:

                def complete_optimizer_step(
                    epoch: int,
                    position: int,
                    sample_id: str,
                    sequence_tokens: int,
                ) -> None:
                    nonlocal accumulated, bounded_stop, budget_stop
                    accumulation_correction = normalize_accumulated_gradients(
                        trainable,
                        accumulated,
                        args.gradient_accumulation,
                    )
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        trainable, args.gradient_clip
                    )
                    if not bool(torch.isfinite(gradient_norm)):
                        raise RuntimeError("non-finite recovery gradient norm")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated = 0
                    cursor["optimizer_steps"] += 1
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "position": position,
                                "sample": sample_id,
                                "tokens": sequence_tokens,
                                "optimizer_steps": cursor["optimizer_steps"],
                                "gradient_norm": float(gradient_norm.detach().cpu()),
                                "accumulation_correction": accumulation_correction,
                                **mean_metrics(recent_metrics[-10:]),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    if cursor["optimizer_steps"] % args.checkpoint_every == 0:
                        checkpoint = args.checkpoint_dir / (
                            f"recovery-step-{cursor['optimizer_steps']:06d}.safetensors"
                        )
                        save_training_checkpoint(
                            checkpoint,
                            parameters,
                            optimizer,
                            cursor,
                            run_contract_sha256,
                            torch,
                        )
                    if (
                        args.max_optimizer_steps is not None
                        and cursor["optimizer_steps"] >= args.max_optimizer_steps
                    ):
                        bounded_stop = True
                    if gpu_run.budget_exhausted():
                        budget_stop = True

                for epoch in range(cursor["epoch"], args.epochs):
                    order = training_order(
                        len(cache.artifacts),
                        epoch,
                        args.seed,
                        selected_indices,
                    )
                    if args.sample_limit is not None:
                        order = order[: args.sample_limit]
                    position_start = (
                        cursor["next_position"] if epoch == cursor["epoch"] else 0
                    )
                    if position_start > len(order):
                        raise ValueError(
                            "resume cursor exceeds deterministic epoch order"
                        )
                    last_trained: tuple[int, str, int] | None = None
                    for position in range(position_start, len(order)):
                        artifact_index = order[position]
                        descriptor = cache.artifacts[artifact_index]
                        teacher_path = cache.verified_artifact_path(artifact_index)
                        teacher = load_teacher_file(teacher_path, torch)
                        sequence_tokens = int(teacher["input_ids"].shape[1])
                        if (
                            args.max_sequence_tokens
                            and sequence_tokens > args.max_sequence_tokens
                        ):
                            if args.oversize_policy == "fail":
                                raise ValueError(
                                    f"teacher sample {descriptor['id']} has "
                                    f"{sequence_tokens} tokens, above "
                                    f"--max-sequence-tokens {args.max_sequence_tokens}"
                                )
                            skipped.append(
                                {"id": descriptor["id"], "tokens": sequence_tokens}
                            )
                            cursor.update(
                                {"epoch": epoch, "next_position": position + 1}
                            )
                            continue
                        regularization = scale_regularization(parameters, torch)
                        chunked = (
                            args.prefill_chunk_tokens > 0
                            and sequence_tokens > args.prefill_chunk_tokens
                        )
                        loss_steps = (
                            runtime.loss_chunks(
                                teacher,
                                args.prefill_chunk_tokens,
                                loss_weights,
                            )
                            if chunked
                            else (runtime.losses(teacher, loss_weights),)
                        )
                        sample_metrics: dict[str, float] = {}
                        objective_value = 0.0
                        chunks_seen = 0
                        for chunks_seen, (total, losses) in enumerate(loss_steps, 1):
                            objective = total
                            if chunks_seen == 1:
                                objective = objective + (
                                    args.scale_regularization * regularization
                                )
                            if not bool(torch.isfinite(objective)):
                                raise RuntimeError(
                                    f"non-finite recovery loss for {descriptor['id']}"
                                )
                            (objective / args.gradient_accumulation).backward()
                            objective_value += float(objective.detach().float().cpu())
                            for name, value in losses.items():
                                sample_metrics[name] = sample_metrics.get(name, 0.0) + float(
                                    value.detach().float().cpu()
                                )
                            del total, losses, objective
                        if chunks_seen == 0:
                            raise RuntimeError(
                                f"recovery produced no loss chunks for {descriptor['id']}"
                            )
                        accumulated += 1
                        last_trained = (
                            position + 1,
                            str(descriptor["id"]),
                            sequence_tokens,
                        )
                        cursor["samples_seen"] += 1
                        cursor.update({"epoch": epoch, "next_position": position + 1})
                        item_metrics = sample_metrics
                        item_metrics["objective"] = objective_value
                        item_metrics["regularization"] = float(
                            regularization.detach().float().cpu()
                        )
                        item_metrics["prefill_chunks"] = float(chunks_seen)
                        recent_metrics.append(item_metrics)
                        if len(recent_metrics) > 100:
                            recent_metrics.pop(0)
                        if accumulated == args.gradient_accumulation:
                            complete_optimizer_step(
                                epoch,
                                position + 1,
                                str(descriptor["id"]),
                                sequence_tokens,
                            )
                            if bounded_stop or budget_stop:
                                break
                        del teacher, regularization, loss_steps, sample_metrics
                    if bounded_stop or budget_stop:
                        break
                    if accumulated:
                        if last_trained is None:
                            raise RuntimeError(
                                "partial gradient group lacks a trained sample"
                            )
                        complete_optimizer_step(epoch, *last_trained)
                        if bounded_stop or budget_stop:
                            break
                    cursor.update({"epoch": epoch + 1, "next_position": 0})
            if budget_stop:
                budget_checkpoint = args.checkpoint_dir / (
                    f"recovery-step-{cursor['optimizer_steps']:06d}.safetensors"
                )
                if not budget_checkpoint.exists():
                    save_training_checkpoint(
                        budget_checkpoint,
                        parameters,
                        optimizer,
                        cursor,
                        run_contract_sha256,
                        torch,
                    )
                raise RuntimeError(
                    "recovery exhausted its admitted GPU-hour reservation at an "
                    f"optimizer boundary; resume from {budget_checkpoint}"
                )
            final_checkpoint = args.checkpoint_dir / (
                f"recovery-final-step-{cursor['optimizer_steps']:06d}.safetensors"
            )
            if final_checkpoint.exists():
                if (
                    args.resume_checkpoint is None
                    or args.resume_checkpoint.resolve() != final_checkpoint.resolve()
                ):
                    raise ValueError(
                        f"refusing to overwrite final checkpoint {final_checkpoint}"
                    )
                final_checkpoint_sha256 = sha256_path(final_checkpoint)
            else:
                final_checkpoint_sha256 = save_training_checkpoint(
                    final_checkpoint,
                    parameters,
                    optimizer,
                    cursor,
                    run_contract_sha256,
                    torch,
                )
            scale_tensors = export_scale_tensors(parameters, torch)
            scale_root = scale_tensor_root(scale_tensors)
            status = recovery_training_status(
                bounded_stop,
                args.sample_limit
                if args.sample_limit is not None
                else (len(selected_indices) if selected_indices is not None else None),
                len(skipped),
            )
            report = {
                **run_contract,
                "run_contract_sha256": run_contract_sha256,
                "status": status,
                "cursor": cursor,
                "elapsed_seconds": time.monotonic() - started,
                "trainable_scale_tensors": len(scale_tensors),
                "trainable_scale_values": sum(
                    tensor.numel() for tensor in scale_tensors.values()
                ),
                "trained_scale_root_sha256": scale_root,
                "final_checkpoint_sha256": final_checkpoint_sha256,
                "recent_mean_losses": mean_metrics(recent_metrics),
                "skipped_oversize_samples": skipped,
                "base_graph": base_evidence,
                "mtp_graph": mtp_evidence,
                "fanout_s_in": fanout_evidence,
                "fla_kernel": kernel_evidence,
            }
            atomic_json(args.output_report, report)
            report_sha256 = sha256_path(args.output_report)
            write_scales(
                args.output_scales,
                scale_tensors,
                {
                    "format": "ctox.recovery.channel-scales.v2",
                    "status": status,
                    "model": str(artifact.manifest["model"]),
                    "revision": args.revision,
                    "local_model_provenance_sha256": local_model_provenance_sha256,
                    "plan_sha256": str(recovery["plan_sha256"]),
                    "activation_stats_sha256": str(recovery["activation_stats_sha256"]),
                    "report_sha256": report_sha256,
                    "teacher_cache_set_sha256": args.teacher_cache_set_sha256,
                    "input_artifact_sha256": artifact_sha256,
                    "run_contract_sha256": run_contract_sha256,
                    "final_checkpoint_sha256": final_checkpoint_sha256,
                    "fixed_logical_qcodes": "true",
                    "fanout_s_in_policy": args.fanout_s_in_policy,
                    "fanout_group_sha256": str(fanout_evidence["group_sha256"]),
                    "trained_scale_root_sha256": scale_root,
                },
            )
            evidence = {
                "format": "ctox.recovery.training-evidence.v1",
                "status": status,
                "run_contract_sha256": run_contract_sha256,
                "report_sha256": report_sha256,
                "scales_sha256": sha256_path(args.output_scales),
                "trained_scale_root_sha256": scale_root,
                "final_checkpoint_sha256": final_checkpoint_sha256,
                "cursor": cursor,
                "skipped_oversize_samples": len(skipped),
                "fixed_logical_qcodes": True,
                "fanout_s_in_policy": args.fanout_s_in_policy,
                "fanout_group_sha256": fanout_evidence["group_sha256"],
            }
            atomic_json(args.output_evidence, evidence)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
