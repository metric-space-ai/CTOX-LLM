"""Bounded autograd operators over immutable native CTOX Q2/Q4 codes."""

from __future__ import annotations

from typing import Any


_FUNCTIONS: dict[int, Any] = {}
_MODULES: dict[int, Any] = {}
_EMBEDDINGS: dict[int, Any] = {}


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


def packed_recovery_linear_class(torch: Any) -> Any:
    cached = _MODULES.get(id(torch))
    if cached is not None:
        return cached

    class PackedRecoveryLinear(torch.nn.Module):
        """Train only positive channel corrections over an immutable packed matrix."""

        def __init__(
            self,
            artifact: Any,
            name: str,
            initial_s_in: Any,
            initial_s_out: Any,
            bias: Any | None = None,
            rows_per_chunk: int = 128,
        ) -> None:
            super().__init__()
            rows, columns = map(int, artifact.tensors[name]["shape"])
            if tuple(initial_s_in.shape) != (columns,) or tuple(initial_s_out.shape) != (rows,):
                raise ValueError(f"packed recovery scale shape differs for {name}")
            if not bool(torch.isfinite(initial_s_in).all()) or not bool(
                torch.isfinite(initial_s_out).all()
            ):
                raise ValueError(f"packed recovery scales are non-finite for {name}")
            if bool((initial_s_in <= 0).any()) or bool((initial_s_out <= 0).any()):
                raise ValueError(f"packed recovery scales are not positive for {name}")
            self.artifact = artifact
            self.name = name
            self.rows_per_chunk = rows_per_chunk
            self.log_s_in = torch.nn.Parameter(initial_s_in.float().log())
            self.log_s_out = torch.nn.Parameter(initial_s_out.float().log())
            self.register_buffer(
                "bias",
                bias.detach() if bias is not None else None,
                persistent=False,
            )

        def forward(self, inputs: Any) -> Any:
            return packed_linear(
                torch,
                self.artifact,
                self.name,
                inputs,
                self.log_s_in.exp(),
                self.log_s_out.exp(),
                self.bias,
                self.rows_per_chunk,
            )

        def correction_tensors(self) -> dict[str, Any]:
            return {
                f"{self.name}.s_in": self.log_s_in.detach().exp().to(torch.float16).cpu(),
                f"{self.name}.s_out": self.log_s_out.detach().exp().to(torch.float16).cpu(),
            }

    _MODULES[id(torch)] = PackedRecoveryLinear
    return PackedRecoveryLinear


def packed_recovery_embedding_class(torch: Any) -> Any:
    cached = _EMBEDDINGS.get(id(torch))
    if cached is not None:
        return cached

    class PackedRecoveryEmbedding(torch.nn.Module):
        def __init__(
            self,
            artifact: Any,
            name: str,
            initial_s_in: Any,
            initial_s_out: Any,
        ) -> None:
            super().__init__()
            rows, columns = map(int, artifact.tensors[name]["shape"])
            if tuple(initial_s_in.shape) != (columns,) or tuple(initial_s_out.shape) != (rows,):
                raise ValueError(f"packed embedding scale shape differs for {name}")
            if (
                not bool(torch.isfinite(initial_s_in).all())
                or not bool(torch.isfinite(initial_s_out).all())
                or bool((initial_s_in <= 0).any())
                or bool((initial_s_out <= 0).any())
            ):
                raise ValueError(f"packed embedding scales are invalid for {name}")
            self.artifact = artifact
            self.name = name
            self.rows = rows
            self.columns = columns
            self.log_s_in = torch.nn.Parameter(initial_s_in.float().log())
            self.log_s_out = torch.nn.Parameter(initial_s_out.float().log())

        def forward(self, input_ids: Any) -> Any:
            if bool((input_ids < 0).any()) or bool((input_ids >= self.rows).any()):
                raise ValueError(f"packed embedding token id is outside {self.name}")
            flat = input_ids.reshape(-1).to(torch.long)
            unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
            decoded = torch.empty(
                (unique.shape[0], self.columns),
                device=input_ids.device,
                dtype=self.log_s_in.dtype,
            )
            start_index = 0
            while start_index < unique.shape[0]:
                stop_index = start_index + 1
                while (
                    stop_index < unique.shape[0]
                    and int(unique[stop_index]) == int(unique[stop_index - 1]) + 1
                ):
                    stop_index += 1
                row_start = int(unique[start_index])
                row_end = int(unique[stop_index - 1]) + 1
                decoded[start_index:stop_index] = self.artifact.decode_matrix_rows(
                    self.name,
                    row_start,
                    row_end,
                    torch,
                    str(input_ids.device),
                ).to(decoded.dtype)
                start_index = stop_index
            selected = decoded.index_select(0, inverse).reshape(
                *input_ids.shape, self.columns
            )
            input_scale = self.log_s_in.exp().to(selected.dtype)
            output_scale = self.log_s_out.exp().index_select(0, flat).reshape(
                *input_ids.shape, 1
            ).to(selected.dtype)
            return selected * input_scale * output_scale

        def correction_tensors(self) -> dict[str, Any]:
            return {
                f"{self.name}.s_in": self.log_s_in.detach().exp().to(torch.float16).cpu(),
                f"{self.name}.s_out": self.log_s_out.detach().exp().to(torch.float16).cpu(),
            }

    _EMBEDDINGS[id(torch)] = PackedRecoveryEmbedding
    return PackedRecoveryEmbedding
