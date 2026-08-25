"""Install the native CTOX text artifact into a meta-initialized Qwen graph."""

from __future__ import annotations

from typing import Any

from packed_recovery_model import PackedRecoveryRegistry
from mtp_teacher import mtp_parameter_mapping


def artifact_to_runtime_name(name: str) -> str:
    if name.startswith("model.language_model."):
        return "model." + name.removeprefix("model.language_model.")
    return name


def runtime_to_artifact_name(name: str) -> str:
    if name.startswith("model."):
        return "model.language_model." + name.removeprefix("model.")
    return name


def set_submodule(root: Any, qualified_name: str, module: Any) -> None:
    parent_name, _, attribute = qualified_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, attribute, module)


def set_parameter(root: Any, qualified_name: str, parameter: Any) -> None:
    parent_name, _, attribute = qualified_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, attribute, parameter)


def install_packed_base_model(
    model: Any,
    artifact: Any,
    torch: Any,
    device: str,
    compute_dtype: Any,
    rows_per_chunk: int,
) -> dict[str, Any]:
    registry = PackedRecoveryRegistry(artifact, torch)
    original_modules = dict(model.named_modules())
    installed = []
    for weight_name in registry.weight_names:
        if weight_name.startswith("mtp."):
            continue
        module_name = artifact_to_runtime_name(weight_name[: -len(".weight")])
        original = original_modules.get(module_name)
        if original is None or not hasattr(original, "weight"):
            raise ValueError(f"Qwen graph lacks native CTOX module {module_name}")
        if isinstance(original, torch.nn.Embedding):
            replacement = registry.make_embedding(
                weight_name, device, compute_dtype=compute_dtype
            )
        elif isinstance(original, torch.nn.Linear):
            replacement = registry.make_linear(
                weight_name, device, rows_per_chunk=rows_per_chunk
            )
        else:
            raise ValueError(
                f"native CTOX weight {weight_name} maps to unsupported {type(original).__name__}"
            )
        set_submodule(model, module_name, replacement)
        installed.append(weight_name)

    # Load every remaining frozen parameter from the same native container.
    # Parameters removed with quantized modules no longer appear here.
    loaded_frozen = []
    for name, parameter in list(model.named_parameters()):
        if not parameter.is_meta:
            continue
        artifact_name = runtime_to_artifact_name(name)
        tensor = artifact.tensors.get(artifact_name)
        if tensor is None or tensor["dtype"] not in {"f16", "f32"}:
            raise ValueError(
                f"meta Qwen parameter {name} lacks native float tensor {artifact_name}"
            )
        value = artifact.decode_float_tensor(artifact_name, torch, device)
        if tensor["dtype"] == "f16":
            value = value.to(compute_dtype)
        set_parameter(
            model,
            name,
            torch.nn.Parameter(value, requires_grad=False),
        )
        loaded_frozen.append(name)

    remaining_meta = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    if remaining_meta:
        raise ValueError(f"Qwen graph retains meta parameters: {remaining_meta[:5]}")
    for _name, parameter in model.named_parameters():
        if not _name.endswith(("log_s_in", "log_s_out")):
            parameter.requires_grad_(False)
    for _name, buffer in model.named_buffers():
        if buffer.device.type == "cpu" and str(device) != "cpu":
            parent_name, _, attribute = _name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attribute, buffer.to(device))
    return {
        "installed_quantized_weights": len(installed),
        "loaded_frozen_float_parameters": len(loaded_frozen),
        "trainable_scale_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "mtp_weights_excluded": sum(
            name.startswith("mtp.") for name in registry.weight_names
        ),
    }


def install_packed_mtp_model(
    main_model: Any,
    artifact: Any,
    torch: Any,
    device: str,
    compute_dtype: Any,
    rows_per_chunk: int,
) -> tuple[Any, dict[str, Any]]:
    from accelerate import init_empty_weights
    from transformers.modeling_layers import MtpModel

    text_config = main_model.config.get_text_config()
    num_layers = getattr(text_config, "num_mtp_layers", None)
    if num_layers is None:
        num_layers = getattr(text_config, "mtp_num_hidden_layers", None)
        text_config.num_mtp_layers = num_layers
    if num_layers != 1:
        raise ValueError(f"packed Qwen recovery requires one MTP layer, got {num_layers}")
    if getattr(text_config, "mtp_layer_types", None) is None:
        text_config.mtp_layer_types = ["full_attention"]
    with init_empty_weights():
        mtp = MtpModel(main_model, num_layers)

    registry = PackedRecoveryRegistry(artifact, torch)
    mtp_weights = [name for name in registry.weight_names if name.startswith("mtp.")]
    mapping = mtp_parameter_mapping(set(artifact.tensors), num_layers)
    original_modules = dict(mtp.named_modules())
    installed = []
    for weight_name in mtp_weights:
        parameter_name = mapping.get(weight_name)
        if parameter_name is None or not parameter_name.endswith(".weight"):
            raise ValueError(f"MTP native weight {weight_name} has no runtime mapping")
        module_name = parameter_name[: -len(".weight")]
        original = original_modules.get(module_name)
        if original is None or not isinstance(original, torch.nn.Linear):
            raise ValueError(f"MTP graph lacks linear module {module_name}")
        set_submodule(
            mtp,
            module_name,
            registry.make_linear(weight_name, device, rows_per_chunk=rows_per_chunk),
        )
        installed.append(weight_name)

    reverse_mapping = {parameter_name: checkpoint_name for checkpoint_name, parameter_name in mapping.items()}
    loaded_frozen = []
    for name, parameter in list(mtp.named_parameters()):
        if not parameter.is_meta:
            continue
        checkpoint_name = reverse_mapping.get(name)
        tensor = artifact.tensors.get(checkpoint_name) if checkpoint_name else None
        if tensor is None or tensor["dtype"] not in {"f16", "f32"}:
            raise ValueError(f"meta MTP parameter {name} lacks a native float tensor")
        value = artifact.decode_float_tensor(checkpoint_name, torch, device)
        if tensor["dtype"] == "f16":
            value = value.to(compute_dtype)
        set_parameter(mtp, name, torch.nn.Parameter(value, requires_grad=False))
        loaded_frozen.append(name)

    owned_meta = [
        name
        for name, parameter in mtp.named_parameters()
        if parameter.is_meta
        and not name.startswith("embed_tokens.")
        and not name.startswith("shared_head.")
    ]
    if owned_meta:
        raise ValueError(f"MTP graph retains meta parameters: {owned_meta[:5]}")
    for name, parameter in mtp.named_parameters():
        if not name.endswith(("log_s_in", "log_s_out")):
            parameter.requires_grad_(False)
    mtp.eval()
    return mtp, {
        "installed_quantized_weights": len(installed),
        "loaded_frozen_float_parameters": len(loaded_frozen),
        "owned_trainable_scale_parameters": sum(
            parameter.numel()
            for name, parameter in mtp.named_parameters()
            if parameter.requires_grad
            and not name.startswith("embed_tokens.")
            and not name.startswith("shared_head.")
        ),
        "remaining_owned_meta_parameters": len(owned_meta),
    }
