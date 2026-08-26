"""Full packed Qwen recovery graph and crash-safe scale-state helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from cache_teacher import transformer_layers
from fanout_recovery import tie_fanout_s_in, validate_parameter_aliases
from mtp_teacher import forward_mtp_activations
from packed_student_model import install_packed_base_model, install_packed_mtp_model
from recovery_io import durable_replace
from recovery_modules import (
    compose_recovery_losses,
    normalized_hidden_loss,
    normalized_hidden_loss_contribution,
    streamed_sparse_target_losses,
)


def validate_teacher_tensors(
    tensors: dict[str, Any],
    hidden_layers: list[int],
    top_k: int,
    mtp_required: bool = True,
) -> None:
    required = {
        "input_ids",
        "attention_mask",
        "logit_positions",
        "hidden_positions",
        "topk_indices",
        "topk_logprobs",
        "residual_probability",
        *(f"hidden_{layer}" for layer in hidden_layers),
    }
    if mtp_required:
        required.update(
            {
                "mtp_positions",
                "mtp_hidden_positions",
                "mtp_hidden",
                "mtp_topk_indices",
                "mtp_topk_logprobs",
                "mtp_residual_probability",
            }
        )
    missing = sorted(required - set(tensors))
    if missing:
        raise ValueError(f"teacher sample lacks tensors: {missing}")
    input_ids = tensors["input_ids"]
    attention_mask = tensors["attention_mask"]
    if (
        input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or input_ids.shape != attention_mask.shape
    ):
        raise ValueError("teacher input and attention-mask shapes differ")
    sequence = int(input_ids.shape[1])

    def positions(name: str, maximum_offset: int) -> Any:
        value = tensors[name]
        if value.ndim != 1 or value.numel() == 0:
            raise ValueError(f"teacher {name} must be a non-empty vector")
        observed = [int(item) for item in value.tolist()]
        if observed != sorted(set(observed)):
            raise ValueError(f"teacher {name} is not sorted and unique")
        if observed[0] < 0 or observed[-1] + maximum_offset >= sequence:
            raise ValueError(f"teacher {name} exceeds the input sequence")
        return value

    logit_positions = positions("logit_positions", 1)
    hidden_positions = positions("hidden_positions", 0)
    if tensors["topk_indices"].shape != tensors["topk_logprobs"].shape:
        raise ValueError("teacher base sparse tensors differ")
    if tuple(tensors["topk_indices"].shape[:2]) != (1, logit_positions.numel()):
        raise ValueError("teacher base sparse target count differs")
    if int(tensors["topk_indices"].shape[-1]) != top_k:
        raise ValueError("teacher top-k differs from cache-set contract")
    if tuple(tensors["residual_probability"].shape) != (1, logit_positions.numel()):
        raise ValueError("teacher base residual target count differs")
    for layer in hidden_layers:
        hidden = tensors[f"hidden_{layer}"]
        if hidden.ndim != 3 or tuple(hidden.shape[:2]) != (1, hidden_positions.numel()):
            raise ValueError(f"teacher hidden_{layer} target count differs")
    if mtp_required:
        mtp_positions = positions("mtp_positions", 2)
        mtp_hidden_positions = positions("mtp_hidden_positions", 2)
        if tensors["mtp_topk_indices"].shape != tensors["mtp_topk_logprobs"].shape:
            raise ValueError("teacher MTP sparse tensors differ")
        if tuple(tensors["mtp_topk_indices"].shape[:2]) != (1, mtp_positions.numel()):
            raise ValueError("teacher MTP sparse target count differs")
        if int(tensors["mtp_topk_indices"].shape[-1]) != top_k:
            raise ValueError("teacher MTP top-k differs from cache-set contract")
        if tuple(tensors["mtp_residual_probability"].shape) != (
            1,
            mtp_positions.numel(),
        ):
            raise ValueError("teacher MTP residual target count differs")
        if tuple(tensors["mtp_hidden"].shape[:2]) != (
            1,
            mtp_hidden_positions.numel(),
        ):
            raise ValueError("teacher MTP hidden target count differs")


def unique_scale_parameters(
    main_model: Any,
    mtp_model: Any,
    fanout_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    for root in (main_model, mtp_model):
        for module in root.modules():
            name = getattr(module, "name", None)
            if (
                name is None
                or not hasattr(module, "log_s_in")
                or not hasattr(module, "log_s_out")
            ):
                continue
            for suffix in ("s_in", "s_out"):
                parameter = getattr(module, f"log_{suffix}")
                scale_name = f"{name}.{suffix}"
                if scale_name in parameters and parameters[scale_name] is not parameter:
                    raise ValueError(f"duplicate recovery owner for {scale_name}")
                parameters[scale_name] = parameter
    if not parameters:
        raise ValueError("packed student exposes no recovery-scale parameters")
    parameters = dict(sorted(parameters.items()))
    validate_parameter_aliases(
        parameters,
        fanout_evidence
        or {
            "policy": "independent",
            "groups": [],
        },
    )
    return parameters


def validate_scale_parameter_contract(
    parameters: dict[str, Any], artifact: Any
) -> None:
    expected = {
        name
        for name, tensor in artifact.tensors.items()
        if tensor.get("dtype") == "f16"
        and name.endswith((".weight.s_in", ".weight.s_out"))
    }
    if set(parameters) != expected:
        missing = sorted(expected - set(parameters))
        extra = sorted(set(parameters) - expected)
        raise ValueError(
            f"packed trainable scale set differs: {len(missing)} missing, {len(extra)} extra"
        )
    for name, parameter in parameters.items():
        shape = tuple(int(value) for value in artifact.tensors[name]["shape"])
        if tuple(parameter.shape) != shape:
            raise ValueError(f"packed trainable scale shape differs for {name}")


def export_scale_tensors(parameters: dict[str, Any], torch: Any) -> dict[str, Any]:
    output = {}
    for name, parameter in parameters.items():
        value = parameter.detach().float().exp()
        if not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
            raise ValueError(f"trained recovery scale is invalid for {name}")
        output[name] = value.to(torch.float16).cpu().contiguous()
    return output


def scale_tensor_root(tensors: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        header = json.dumps(
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(tensor.contiguous().view(-1).numpy().tobytes())
    return digest.hexdigest()


def scale_regularization(parameters: dict[str, Any], torch: Any) -> Any:
    totals = [parameter.float().square().mean() for parameter in parameters.values()]
    if not totals:
        raise ValueError("scale regularization has no parameters")
    return torch.stack(totals).mean()


class PackedStudentRuntime:
    def __init__(
        self,
        main_model: Any,
        mtp_model: Any,
        base_cache_type: Any,
        mtp_cache_type: Any,
        hidden_layers: list[int],
        top_k: int,
        logit_chunk: int,
        torch: Any,
        device: Any,
        gradient_checkpointing: bool,
    ) -> None:
        if not hidden_layers or logit_chunk <= 0 or top_k <= 0:
            raise ValueError("packed student runtime contract is incomplete")
        self.main_model = main_model
        self.mtp_model = mtp_model
        self.base_cache_type = base_cache_type
        self.mtp_cache_type = mtp_cache_type
        self.hidden_layers = hidden_layers
        self.top_k = top_k
        self.logit_chunk = logit_chunk
        self.torch = torch
        self.device = device
        self.gradient_checkpointing = gradient_checkpointing
        self.base_model, self.layers = transformer_layers(main_model)
        if any(layer < 0 or layer >= len(self.layers) for layer in hidden_layers):
            raise ValueError("recovery hidden layer index is outside the student graph")

    def losses(
        self,
        teacher: dict[str, Any],
        weights: dict[str, float] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        torch = self.torch
        validate_teacher_tensors(teacher, self.hidden_layers, self.top_k, True)
        input_ids = teacher["input_ids"].to(self.device, dtype=torch.long)
        attention_mask = teacher["attention_mask"].to(self.device, dtype=torch.long)
        logit_positions = teacher["logit_positions"].to(self.device, dtype=torch.long)
        hidden_positions = teacher["hidden_positions"].to(self.device, dtype=torch.long)
        captured: dict[int, Any] = {}
        hooks = []
        for layer_index in self.hidden_layers:

            def capture(
                _module: Any,
                _inputs: Any,
                output: Any,
                index: int = layer_index,
            ) -> None:
                values = output[0] if isinstance(output, tuple) else output
                captured[index] = values.index_select(
                    1, hidden_positions.to(values.device)
                )

            hooks.append(self.layers[layer_index].register_forward_hook(capture))
        try:
            output = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        finally:
            for hook in hooks:
                hook.remove()
        if set(captured) != set(self.hidden_layers):
            raise RuntimeError("packed student did not capture every hidden layer")
        final_hidden = output.last_hidden_state
        selected_final = final_hidden.index_select(1, logit_positions)
        kl, ce = streamed_sparse_target_losses(
            self.main_model.lm_head,
            selected_final,
            teacher["topk_indices"].to(self.device),
            teacher["topk_logprobs"].to(self.device),
            teacher["residual_probability"].to(self.device),
            input_ids,
            logit_positions,
            1,
            self.logit_chunk,
        )
        hidden_losses = [
            normalized_hidden_loss(
                captured[layer], teacher[f"hidden_{layer}"].to(self.device)
            )
            for layer in self.hidden_layers
        ]

        mtp_positions = teacher["mtp_positions"].to(self.device, dtype=torch.long)
        mtp_input_ids = input_ids[:, 1:]
        mtp_base_hidden = final_hidden[:, :-1, :]
        mtp_position_ids = torch.arange(
            1,
            input_ids.shape[1],
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(0)
        mtp_cache = self.mtp_cache_type(config=self.mtp_model.config)
        mtp_output = forward_mtp_activations(
            self.mtp_model,
            mtp_input_ids,
            mtp_base_hidden,
            mtp_position_ids,
            mtp_cache,
        )
        selected_mtp = mtp_output.index_select(1, mtp_positions)
        mtp_kl, mtp_ce = streamed_sparse_target_losses(
            self.main_model.lm_head,
            selected_mtp,
            teacher["mtp_topk_indices"].to(self.device),
            teacher["mtp_topk_logprobs"].to(self.device),
            teacher["mtp_residual_probability"].to(self.device),
            input_ids,
            mtp_positions,
            2,
            self.logit_chunk,
        )
        mtp_hidden_positions = teacher["mtp_hidden_positions"].to(
            self.device, dtype=torch.long
        )
        losses = {
            "kl": kl,
            "ce": ce,
            "hidden": torch.stack(hidden_losses).mean(),
            "mtp_kl": mtp_kl,
            "mtp_ce": mtp_ce,
            "mtp_hidden": normalized_hidden_loss(
                mtp_output.index_select(1, mtp_hidden_positions),
                teacher["mtp_hidden"].to(self.device),
            ),
        }
        return compose_recovery_losses(losses, weights), losses

    @staticmethod
    def _chunk_positions(positions: Any, start: int, stop: int) -> tuple[Any, Any]:
        mask = (positions >= start) & (positions < stop)
        indices = mask.nonzero(as_tuple=False).flatten()
        return indices, positions.index_select(0, indices) - start

    @staticmethod
    def _detach_cache(cache: Any) -> None:
        """Truncate autograd history while preserving exact causal state."""

        for layer in cache.layers:
            for attribute in ("keys", "values"):
                value = getattr(layer, attribute, None)
                if value is not None:
                    setattr(layer, attribute, value.detach())
            for attribute in ("conv_states", "recurrent_states"):
                states = getattr(layer, attribute, None)
                if states is None:
                    continue
                for index, value in tuple(states.items()):
                    if value is not None:
                        states[index] = value.detach()

    def loss_chunks(
        self,
        teacher: dict[str, Any],
        chunk_tokens: int,
        weights: dict[str, float] | None = None,
    ) -> Any:
        """Yield exact loss contributions for stateful truncated-BPTT prefill.

        The caller must backpropagate each yielded objective before requesting
        the next item.  Resuming the generator detaches only the causal cache
        tensors, so every chunk trains all fixed-qcode corrections it executes
        while KV and GatedDelta state remain numerically continuous.
        """

        torch = self.torch
        validate_teacher_tensors(teacher, self.hidden_layers, self.top_k, True)
        if chunk_tokens <= 0:
            raise ValueError("recovery prefill chunk size must be positive")
        if self.gradient_checkpointing:
            raise ValueError(
                "stateful recovery chunking cannot be combined with gradient checkpointing"
            )
        input_ids = teacher["input_ids"].to(self.device, dtype=torch.long)
        attention_mask = teacher["attention_mask"].to(self.device, dtype=torch.long)
        sequence = int(input_ids.shape[1])
        if chunk_tokens >= sequence:
            yield self.losses(teacher, weights)
            return

        position_names = (
            "logit_positions",
            "hidden_positions",
            "mtp_positions",
            "mtp_hidden_positions",
        )
        positions = {
            name: teacher[name].to(dtype=torch.long)
            for name in position_names
        }
        totals = {name: int(value.numel()) for name, value in positions.items()}
        hidden_signal = {
            layer: teacher[f"hidden_{layer}"].float().square().sum()
            for layer in self.hidden_layers
        }
        mtp_hidden_signal = teacher["mtp_hidden"].float().square().sum()
        base_cache = self.base_cache_type(config=self.main_model.config)
        mtp_cache = self.mtp_cache_type(config=self.mtp_model.config)

        for chunk_start in range(0, sequence, chunk_tokens):
            chunk_stop = min(sequence, chunk_start + chunk_tokens)
            hidden_indices, local_hidden_positions = self._chunk_positions(
                positions["hidden_positions"], chunk_start, chunk_stop
            )
            captured: dict[int, Any] = {}
            hooks = []
            if hidden_indices.numel():
                for layer_index in self.hidden_layers:

                    def capture(
                        _module: Any,
                        _inputs: Any,
                        output: Any,
                        index: int = layer_index,
                        sink: dict[int, Any] = captured,
                    ) -> None:
                        values = output[0] if isinstance(output, tuple) else output
                        sink[index] = values.index_select(
                            1, local_hidden_positions.to(values.device)
                        )

                    hooks.append(self.layers[layer_index].register_forward_hook(capture))
            try:
                output = self.base_model(
                    input_ids=input_ids[:, chunk_start:chunk_stop],
                    attention_mask=attention_mask[:, :chunk_stop],
                    past_key_values=base_cache,
                    use_cache=True,
                    return_dict=True,
                )
            finally:
                for hook in hooks:
                    hook.remove()
            if hidden_indices.numel() and set(captured) != set(self.hidden_layers):
                raise RuntimeError("chunked student did not capture every hidden layer")
            final_hidden = output.last_hidden_state
            zero = final_hidden.float().sum() * 0.0
            losses = {
                name: zero
                for name in ("kl", "ce", "hidden", "mtp_kl", "mtp_ce", "mtp_hidden")
            }

            logit_indices, local_logit_positions = self._chunk_positions(
                positions["logit_positions"], chunk_start, chunk_stop
            )
            if logit_indices.numel():
                selected_final = final_hidden.index_select(
                    1, local_logit_positions.to(final_hidden.device)
                )
                global_logit_positions = positions["logit_positions"].index_select(
                    0, logit_indices
                ).to(self.device)
                kl, ce = streamed_sparse_target_losses(
                    self.main_model.lm_head,
                    selected_final,
                    teacher["topk_indices"].index_select(1, logit_indices).to(self.device),
                    teacher["topk_logprobs"].index_select(1, logit_indices).to(self.device),
                    teacher["residual_probability"].index_select(1, logit_indices).to(self.device),
                    input_ids,
                    global_logit_positions,
                    1,
                    self.logit_chunk,
                )
                fraction = int(logit_indices.numel()) / totals["logit_positions"]
                losses["kl"] = kl * fraction
                losses["ce"] = ce * fraction

            if hidden_indices.numel():
                losses["hidden"] = torch.stack(
                    [
                        normalized_hidden_loss_contribution(
                            captured[layer],
                            teacher[f"hidden_{layer}"]
                            .index_select(1, hidden_indices)
                            .to(self.device),
                            hidden_signal[layer],
                            totals["hidden_positions"],
                        )
                        for layer in self.hidden_layers
                    ]
                ).mean()

            mtp_hidden_stop = min(chunk_stop, sequence - 1)
            mtp_output = None
            if chunk_start < mtp_hidden_stop:
                mtp_length = mtp_hidden_stop - chunk_start
                mtp_output = forward_mtp_activations(
                    self.mtp_model,
                    input_ids[:, chunk_start + 1 : mtp_hidden_stop + 1],
                    final_hidden[:, :mtp_length, :],
                    torch.arange(
                        chunk_start + 1,
                        mtp_hidden_stop + 1,
                        device=self.device,
                        dtype=torch.long,
                    ).unsqueeze(0),
                    mtp_cache,
                )
                mtp_indices, local_mtp_positions = self._chunk_positions(
                    positions["mtp_positions"], chunk_start, mtp_hidden_stop
                )
                if mtp_indices.numel():
                    selected_mtp = mtp_output.index_select(
                        1, local_mtp_positions.to(mtp_output.device)
                    )
                    global_mtp_positions = positions["mtp_positions"].index_select(
                        0, mtp_indices
                    ).to(self.device)
                    mtp_kl, mtp_ce = streamed_sparse_target_losses(
                        self.main_model.lm_head,
                        selected_mtp,
                        teacher["mtp_topk_indices"].index_select(1, mtp_indices).to(self.device),
                        teacher["mtp_topk_logprobs"].index_select(1, mtp_indices).to(self.device),
                        teacher["mtp_residual_probability"]
                        .index_select(1, mtp_indices)
                        .to(self.device),
                        input_ids,
                        global_mtp_positions,
                        2,
                        self.logit_chunk,
                    )
                    fraction = int(mtp_indices.numel()) / totals["mtp_positions"]
                    losses["mtp_kl"] = mtp_kl * fraction
                    losses["mtp_ce"] = mtp_ce * fraction

                mtp_hidden_indices, local_mtp_hidden_positions = self._chunk_positions(
                    positions["mtp_hidden_positions"], chunk_start, mtp_hidden_stop
                )
                if mtp_hidden_indices.numel():
                    losses["mtp_hidden"] = normalized_hidden_loss_contribution(
                        mtp_output.index_select(
                            1, local_mtp_hidden_positions.to(mtp_output.device)
                        ),
                        teacher["mtp_hidden"].index_select(1, mtp_hidden_indices).to(self.device),
                        mtp_hidden_signal,
                        totals["mtp_hidden_positions"],
                    )

            yield compose_recovery_losses(losses, weights), losses
            del output, final_hidden, captured, losses, mtp_output
            self._detach_cache(base_cache)
            self._detach_cache(mtp_cache)


def build_packed_student(
    model_source: str,
    revision: str,
    artifact: Any,
    device: Any,
    compute_dtype: Any,
    rows_per_chunk: int,
    hidden_layers: list[int],
    top_k: int,
    logit_chunk: int,
    gradient_checkpointing: bool,
    fanout_s_in_policy: str,
    torch: Any,
) -> tuple[PackedStudentRuntime, dict[str, Any], dict[str, Any], dict[str, Any]]:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache, MtpCache

    local = Path(model_source).is_dir()
    config = AutoConfig.from_pretrained(
        model_source,
        revision=revision,
        local_files_only=local,
    )
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    base_evidence = install_packed_base_model(
        model,
        artifact,
        torch,
        str(device),
        compute_dtype,
        rows_per_chunk,
    )
    mtp_model, mtp_evidence = install_packed_mtp_model(
        model,
        artifact,
        torch,
        str(device),
        compute_dtype,
        rows_per_chunk,
    )
    fanout_evidence = tie_fanout_s_in(
        model,
        mtp_model,
        torch,
        fanout_s_in_policy,
    )
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.train()
    mtp_model.train()
    runtime = PackedStudentRuntime(
        model,
        mtp_model,
        DynamicCache,
        MtpCache,
        hidden_layers,
        top_k,
        logit_chunk,
        torch,
        device,
        gradient_checkpointing,
    )
    return runtime, base_evidence, mtp_evidence, fanout_evidence


def optimizer_parameters(parameters: dict[str, Any]) -> list[Any]:
    seen: set[int] = set()
    output = []
    for parameter in parameters.values():
        if id(parameter) not in seen:
            seen.add(id(parameter))
            output.append(parameter)
    return output


def immutable_run_contract(document: dict[str, Any]) -> tuple[str, str]:
    encoded = json.dumps(document, separators=(",", ":"), sort_keys=True)
    return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_teacher_file(path: Path, torch: Any) -> dict[str, Any]:
    from safetensors.torch import load_file

    return load_file(path, device="cpu")


def save_training_checkpoint(
    path: Path,
    parameters: dict[str, Any],
    optimizer: Any,
    cursor: dict[str, int],
    run_contract_sha256: str,
    torch: Any,
) -> str:
    from safetensors.torch import save_file

    if path.exists():
        raise ValueError(f"refusing to overwrite {path}")
    required_cursor = {"epoch", "next_position", "optimizer_steps", "samples_seen"}
    if set(cursor) != required_cursor or any(
        int(value) < 0 for value in cursor.values()
    ):
        raise ValueError("recovery checkpoint cursor is invalid")
    tensors: dict[str, Any] = {}
    for name, parameter in parameters.items():
        tensors[f"parameter/{name}"] = parameter.detach().float().cpu().contiguous()
        state = optimizer.state.get(parameter)
        if not state or not {"step", "exp_avg", "exp_avg_sq"} <= set(state):
            raise ValueError(f"optimizer state is incomplete for {name}")
        tensors[f"optimizer_step/{name}"] = (
            torch.as_tensor(state["step"]).cpu().reshape(1)
        )
        tensors[f"optimizer_exp_avg/{name}"] = (
            state["exp_avg"].detach().cpu().contiguous()
        )
        tensors[f"optimizer_exp_avg_sq/{name}"] = (
            state["exp_avg_sq"].detach().cpu().contiguous()
        )
    metadata = {
        "format": "ctox.recovery.training-checkpoint.v1",
        "run_contract_sha256": run_contract_sha256,
        **{key: str(int(value)) for key, value in cursor.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    save_file(tensors, temporary, metadata=metadata)
    durable_replace(temporary, path)
    return sha256_path(path)


def restore_training_checkpoint(
    path: Path,
    parameters: dict[str, Any],
    optimizer: Any,
    run_contract_sha256: str,
    torch: Any,
) -> dict[str, int]:
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(path, framework="pt", device="cpu") as source:
        metadata = source.metadata() or {}
    if metadata.get("format") != "ctox.recovery.training-checkpoint.v1":
        raise ValueError("unsupported recovery training checkpoint")
    if metadata.get("run_contract_sha256") != run_contract_sha256:
        raise ValueError("recovery checkpoint run contract differs")
    tensors = load_file(path, device="cpu")
    expected = {
        f"{prefix}/{name}"
        for name in parameters
        for prefix in (
            "parameter",
            "optimizer_step",
            "optimizer_exp_avg",
            "optimizer_exp_avg_sq",
        )
    }
    if set(tensors) != expected:
        raise ValueError("recovery checkpoint tensor set differs")
    aliases: dict[int, list[str]] = {}
    for name, parameter in parameters.items():
        aliases.setdefault(id(parameter), []).append(name)
    for names in aliases.values():
        if len(names) < 2:
            continue
        reference = names[0]
        for name in names[1:]:
            for prefix in (
                "parameter",
                "optimizer_step",
                "optimizer_exp_avg",
                "optimizer_exp_avg_sq",
            ):
                if not torch.equal(
                    tensors[f"{prefix}/{reference}"], tensors[f"{prefix}/{name}"]
                ):
                    raise ValueError(
                        f"recovery checkpoint alias state differs for {reference} and {name}"
                    )
    for name, parameter in parameters.items():
        value = tensors[f"parameter/{name}"]
        if tuple(value.shape) != tuple(parameter.shape) or not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(f"recovery checkpoint parameter differs for {name}")
        parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        exp_avg = tensors[f"optimizer_exp_avg/{name}"]
        exp_avg_sq = tensors[f"optimizer_exp_avg_sq/{name}"]
        if tuple(exp_avg.shape) != tuple(parameter.shape) or tuple(
            exp_avg_sq.shape
        ) != tuple(parameter.shape):
            raise ValueError(f"recovery checkpoint optimizer shape differs for {name}")
        optimizer.state[parameter] = {
            "step": tensors[f"optimizer_step/{name}"].reshape(()),
            "exp_avg": exp_avg.to(device=parameter.device, dtype=parameter.dtype),
            "exp_avg_sq": exp_avg_sq.to(device=parameter.device, dtype=parameter.dtype),
        }
    cursor = {}
    for key in ("epoch", "next_position", "optimizer_steps", "samples_seen"):
        try:
            value = int(metadata[key])
        except (KeyError, ValueError) as error:
            raise ValueError(f"recovery checkpoint lacks cursor {key}") from error
        if value < 0:
            raise ValueError(f"recovery checkpoint cursor {key} is negative")
        cursor[key] = value
    return cursor


LossObserver = Callable[[dict[str, float]], None]
