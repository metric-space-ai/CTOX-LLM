"""Deterministic, host-testable state helpers for end-to-end recovery."""

from __future__ import annotations

import random
from typing import Any


def training_order(samples: int, epoch: int, seed: int) -> list[int]:
    if samples <= 0 or epoch < 0:
        raise ValueError(
            "training order requires positive samples and a non-negative epoch"
        )
    order = list(range(samples))
    random.Random(f"ctox-recovery:{seed}:{epoch}").shuffle(order)
    return order


def recovery_training_status(
    bounded_stop: bool,
    sample_limit: int | None,
    skipped_samples: int,
) -> str:
    """Return a fail-closed status suitable for the final packer gate."""

    if skipped_samples < 0:
        raise ValueError("skipped sample count cannot be negative")
    if bounded_stop:
        return "bounded_run_complete"
    if sample_limit is not None:
        return "subset_run_complete"
    if skipped_samples:
        return "partial_coverage"
    return "complete"


def normalize_accumulated_gradients(
    parameters: list[Any],
    accumulated: int,
    configured_accumulation: int,
) -> float:
    """Correct the final partial accumulation group before clipping.

    Every individual objective is divided by ``configured_accumulation``.
    When an epoch ends with fewer records, multiply the accumulated gradients
    back to the exact mean over the records that actually contributed.
    """

    if not 0 < accumulated <= configured_accumulation:
        raise ValueError("accumulated gradient count is outside its contract")
    factor = configured_accumulation / accumulated
    gradients = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradients += 1
        if factor != 1.0:
            parameter.grad.mul_(factor)
    if gradients == 0:
        raise RuntimeError("optimizer step has no accumulated gradients")
    return factor
