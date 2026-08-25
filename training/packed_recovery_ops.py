"""Bounded autograd operators over immutable native CTOX Q2/Q4 codes."""

from __future__ import annotations

from typing import Any


_FUNCTIONS: dict[int, Any] = {}


def packed_linear_function(torch: Any) -> Any:
    cached = _FUNCTIONS.get(id(torch))
    if cached is not None:
        return cached

    class PackedLinearFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: Any,
            artifact: Any,
            name: str,
            inputs: Any,
            s_in: Any,
            s_out: Any,
            bias: Any,
            rows_per_chunk: int,
        ) -> Any:
            entry = artifact.tensors[name]
            rows, columns = map(int, entry["shape"])
            if inputs.shape[-1] != columns:
                raise ValueError(f"packed linear {name} input dimension differs")
            if tuple(s_in.shape) != (columns,) or tuple(s_out.shape) != (rows,):
                raise ValueError(f"packed linear {name} recovery scale shape differs")
            if bias is not None and tuple(bias.shape) != (rows,):
                raise ValueError(f"packed linear {name} bias shape differs")
            if rows_per_chunk <= 0:
                raise ValueError("rows_per_chunk must be positive")
            flat = inputs.reshape(-1, columns)
            corrected = flat * s_in.to(device=flat.device, dtype=flat.dtype)
            output = torch.empty(
                (flat.shape[0], rows), device=flat.device, dtype=flat.dtype
            )
            for start in range(0, rows, rows_per_chunk):
                stop = min(rows, start + rows_per_chunk)
                weight = artifact.decode_matrix_rows(
                    name, start, stop, torch, str(flat.device)
                ).to(dtype=flat.dtype)
                local_bias = (
                    bias[start:stop].to(device=flat.device, dtype=flat.dtype)
                    if bias is not None
                    else None
                )
                base = torch.nn.functional.linear(corrected, weight, local_bias)
                output[:, start:stop] = base * s_out[start:stop].to(
                    device=flat.device, dtype=flat.dtype
                )
            bias_storage = (
                bias
                if bias is not None
                else torch.empty(0, device=inputs.device, dtype=inputs.dtype)
            )
            ctx.save_for_backward(inputs, s_in, s_out, bias_storage)
            ctx.artifact = artifact
            ctx.name = name
            ctx.rows_per_chunk = rows_per_chunk
            ctx.has_bias = bias is not None
            ctx.input_shape = tuple(inputs.shape)
            return output.reshape(*inputs.shape[:-1], rows)

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, ...]:
            inputs, s_in, s_out, bias_storage = ctx.saved_tensors
            entry = ctx.artifact.tensors[ctx.name]
            rows, columns = map(int, entry["shape"])
            flat = inputs.reshape(-1, columns)
            grad_flat = grad_output.reshape(-1, rows)
            input_scale = s_in.to(device=flat.device, dtype=flat.dtype)
            corrected = flat * input_scale
            grad_corrected = torch.zeros_like(flat)
            grad_s_out = torch.zeros_like(s_out)
            grad_bias = torch.zeros_like(bias_storage) if ctx.has_bias else None
            for start in range(0, rows, ctx.rows_per_chunk):
                stop = min(rows, start + ctx.rows_per_chunk)
                weight = ctx.artifact.decode_matrix_rows(
                    ctx.name, start, stop, torch, str(flat.device)
                ).to(dtype=flat.dtype)
                local_bias = (
                    bias_storage[start:stop].to(device=flat.device, dtype=flat.dtype)
                    if ctx.has_bias
                    else None
                )
                base = torch.nn.functional.linear(corrected, weight, local_bias)
                local_grad = grad_flat[:, start:stop]
                grad_s_out[start:stop] = (
                    local_grad.float() * base.float()
                ).sum(dim=0).to(grad_s_out.dtype)
                scaled_grad = local_grad * s_out[start:stop].to(
                    device=flat.device, dtype=flat.dtype
                )
                grad_corrected.add_(scaled_grad @ weight)
                if grad_bias is not None:
                    grad_bias[start:stop] = scaled_grad.sum(dim=0).to(grad_bias.dtype)
            grad_inputs = (grad_corrected * input_scale).reshape(ctx.input_shape)
            grad_s_in = (grad_corrected.float() * flat.float()).sum(dim=0).to(s_in.dtype)
            return None, None, grad_inputs, grad_s_in, grad_s_out, grad_bias, None

    _FUNCTIONS[id(torch)] = PackedLinearFunction
    return PackedLinearFunction


def packed_linear(
    torch: Any,
    artifact: Any,
    name: str,
    inputs: Any,
    s_in: Any,
    s_out: Any,
    bias: Any | None = None,
    rows_per_chunk: int = 128,
) -> Any:
    return packed_linear_function(torch).apply(
        artifact,
        name,
        inputs,
        s_in,
        s_out,
        bias,
        rows_per_chunk,
    )
