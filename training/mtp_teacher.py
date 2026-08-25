#!/usr/bin/env python3
"""Load the frozen Qwen3.8 MTP teacher from its native checkpoint names.

Transformers can execute generic MTP layers, but the current Qwen3.8 config and
checkpoint use the original `mtp.*` names instead of the generic loader names.
This module keeps that compatibility mapping explicit and fail-closed so MTP
calibration never silently runs with randomly initialized weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def mtp_checkpoint_weight_name(module_name: str) -> str:
    """Map a generic Transformers MTP module to the frozen checkpoint name."""

    prefix = "layers.0."
    if not module_name.startswith(prefix):
        return module_name + ".weight"
    suffix = module_name.removeprefix(prefix)
    if suffix == "eh_proj":
        return "mtp.fc.weight"
    if suffix == "enorm":
        return "mtp.pre_fc_norm_embedding.weight"
    if suffix == "hnorm":
        return "mtp.pre_fc_norm_hidden.weight"
    if suffix == "post_norm":
        return "mtp.norm.weight"
    if suffix.startswith("mtp_block."):
        return "mtp.layers.0." + suffix.removeprefix("mtp_block.") + ".weight"
    return module_name + ".weight"


def mtp_parameter_mapping(checkpoint_names: set[str], num_layers: int) -> dict[str, str]:
    """Return checkpoint-name -> generic-parameter-name for the supported MTP."""

    if num_layers != 1:
        raise ValueError(f"Qwen3.8 recovery currently requires exactly one MTP layer, got {num_layers}")
    mapping: dict[str, str] = {}
    for checkpoint_name in checkpoint_names:
        if checkpoint_name == "mtp.fc.weight":
            target = "layers.0.eh_proj.weight"
        elif checkpoint_name == "mtp.pre_fc_norm_embedding.weight":
            target = "layers.0.enorm.weight"
        elif checkpoint_name == "mtp.pre_fc_norm_hidden.weight":
            target = "layers.0.hnorm.weight"
        elif checkpoint_name == "mtp.norm.weight":
            target = "layers.0.post_norm.weight"
        elif checkpoint_name.startswith("mtp.layers.0."):
            target = "layers.0.mtp_block." + checkpoint_name.removeprefix("mtp.layers.0.")
        else:
            continue
        mapping[checkpoint_name] = target
    return mapping


def load_mtp_teacher(main_model: Any, checkpoint: Path, device: Any, safe_open: Any) -> Any:
    """Instantiate, validate, and load the Qwen3.8 MTP layer on `device`."""

    from transformers.modeling_layers import MtpModel

    text_config = main_model.config.get_text_config()
    num_layers = getattr(text_config, "num_mtp_layers", None)
    if num_layers is None:
        num_layers = getattr(text_config, "mtp_num_hidden_layers", None)
        if num_layers is None:
            raise ValueError("checkpoint config does not declare MTP layers")
        text_config.num_mtp_layers = num_layers
    if getattr(text_config, "mtp_layer_types", None) is None:
        # The frozen checkpoint contains q/k/v/o attention projections and its
        # single MTP block is therefore a full-attention decoder block.
        text_config.mtp_layer_types = ["full_attention"] * num_layers

    mtp = MtpModel(main_model, num_layers)
    mtp.layers.to(device=device, dtype=main_model.config.dtype)
    if getattr(mtp, "shared_post_norm", None) is not None:
        mtp.shared_post_norm.to(device=device, dtype=main_model.config.dtype)

    index = json.loads((checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    checkpoint_names = {name for name in weight_map if name.startswith("mtp.")}
    mapping = mtp_parameter_mapping(checkpoint_names, num_layers)
    parameters = dict(mtp.named_parameters())
    owned_parameters = {
        name
        for name in parameters
        if not name.startswith("embed_tokens.") and not name.startswith("shared_head.")
    }
    mapped_parameters = set(mapping.values())
    missing = sorted(owned_parameters - mapped_parameters)
    unknown = sorted(mapped_parameters - owned_parameters)
    if missing or unknown:
        raise RuntimeError(f"MTP mapping mismatch: missing={missing}, unknown={unknown}")

    shard_groups: dict[str, list[tuple[str, str]]] = {}
    for checkpoint_name, parameter_name in mapping.items():
        shard_groups.setdefault(weight_map[checkpoint_name], []).append(
            (checkpoint_name, parameter_name)
        )
    for shard, entries in shard_groups.items():
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as source:
            for checkpoint_name, parameter_name in entries:
                parameter = parameters[parameter_name]
                value = source.get_tensor(checkpoint_name).to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                if tuple(value.shape) != tuple(parameter.shape):
                    raise RuntimeError(
                        f"MTP shape mismatch for {checkpoint_name}: "
                        f"{tuple(value.shape)} != {tuple(parameter.shape)}"
                    )
                parameter.data.copy_(value)
    mtp.eval()
    return mtp
