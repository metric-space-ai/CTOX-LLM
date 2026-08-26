"""Plan-bound shared input corrections for Qwen projection fan-outs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Iterable


INDEPENDENT_POLICY = "independent"
QWEN38_FANOUT_POLICY = "qwen38_fanout_s_in_v1"
POLICIES = (INDEPENDENT_POLICY, QWEN38_FANOUT_POLICY)

_GROUP_SPECS = (
    (
        "mlp_gate_up",
        ("gate_proj.weight", "up_proj.weight"),
    ),
    (
        "full_attention_qkv",
        ("q_proj.weight", "k_proj.weight", "v_proj.weight"),
    ),
    (
        "linear_attention_inputs",
        (
            "in_proj_qkv.weight",
            "in_proj_z.weight",
            "in_proj_a.weight",
            "in_proj_b.weight",
        ),
    ),
)

_EXPECTED_QWEN38_GROUPS = {
    "mlp_gate_up": 65,
    "full_attention_qkv": 17,
    "linear_attention_inputs": 48,
}


def _matching_spec(name: str) -> tuple[str, str, tuple[str, ...]] | None:
    for kind, suffixes in _GROUP_SPECS:
        for suffix in suffixes:
            marker = f".{suffix}"
            if name.endswith(marker):
                return kind, name[: -len(marker)], suffixes
    return None


def qwen38_fanout_groups(
    weight_names: Iterable[str],
    *,
    require_frozen_topology: bool = True,
) -> list[dict[str, Any]]:
    """Return exact same-input projection groups in deterministic order.

    Every recognized prefix must contain the complete group. A partial Q/K/V,
    gate/up, or linear-attention input fan-out is a graph-contract error rather
    than an opportunity to share an ambiguous activation.
    """

    names = {str(name) for name in weight_names}
    candidates: dict[tuple[str, str], tuple[str, ...]] = {}
    for name in names:
        match = _matching_spec(name)
        if match is None:
            continue
        kind, prefix, suffixes = match
        candidates[(kind, prefix)] = suffixes

    groups = []
    claimed: set[str] = set()
    for (kind, prefix), suffixes in sorted(candidates.items()):
        members = [f"{prefix}.{suffix}" for suffix in suffixes]
        present = [member for member in members if member in names]
        if len(present) != len(members):
            missing = sorted(set(members) - set(present))
            raise ValueError(
                f"incomplete {kind} recovery fan-out at {prefix}: missing {missing}"
            )
        overlap = claimed.intersection(members)
        if overlap:
            raise ValueError(f"recovery fan-out groups overlap: {sorted(overlap)}")
        claimed.update(members)
        groups.append(
            {
                "kind": kind,
                "prefix": prefix,
                "weights": members,
                "scale_names": [f"{member}.s_in" for member in members],
            }
        )

    counts = Counter(group["kind"] for group in groups)
    if require_frozen_topology and dict(counts) != _EXPECTED_QWEN38_GROUPS:
        raise ValueError(
            "Qwen3.8 fan-out topology differs: "
            f"observed={dict(sorted(counts.items()))} "
            f"expected={dict(sorted(_EXPECTED_QWEN38_GROUPS.items()))}"
        )
    return groups


def fanout_group_sha256(groups: list[dict[str, Any]]) -> str:
    logical = [
        {
            "kind": group["kind"],
            "prefix": group["prefix"],
            "scale_names": list(group["scale_names"]),
        }
        for group in groups
    ]
    encoded = json.dumps(logical, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def recovery_modules(*roots: Any) -> dict[str, Any]:
    modules: dict[str, Any] = {}
    for root in roots:
        for module in root.modules():
            name = getattr(module, "name", None)
            if (
                name is None
                or not hasattr(module, "log_s_in")
                or not hasattr(module, "log_s_out")
            ):
                continue
            prior = modules.get(str(name))
            if prior is not None and prior is not module:
                raise ValueError(f"duplicate packed recovery module {name}")
            modules[str(name)] = module
    return modules


def tie_fanout_s_in(
    main_model: Any,
    mtp_model: Any,
    torch: Any,
    policy: str,
    *,
    require_frozen_topology: bool = True,
) -> dict[str, Any]:
    if policy not in POLICIES:
        raise ValueError(f"unsupported recovery fan-out policy {policy}")
    modules = recovery_modules(main_model, mtp_model)
    groups = (
        qwen38_fanout_groups(
            modules,
            require_frozen_topology=require_frozen_topology,
        )
        if policy == QWEN38_FANOUT_POLICY
        else []
    )
    maximum_initial_log_delta = 0.0
    for group in groups:
        owners = [modules[name] for name in group["weights"]]
        shapes = {tuple(owner.log_s_in.shape) for owner in owners}
        devices = {owner.log_s_in.device for owner in owners}
        dtypes = {owner.log_s_in.dtype for owner in owners}
        if len(shapes) != 1 or len(devices) != 1 or len(dtypes) != 1:
            raise ValueError(
                f"recovery fan-out input-scale contract differs at {group['prefix']}"
            )
        values = torch.stack([owner.log_s_in.detach() for owner in owners])
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"non-finite fan-out initializer at {group['prefix']}")
        shared_value = values.mean(dim=0)
        maximum_initial_log_delta = max(
            maximum_initial_log_delta,
            float((values - shared_value).abs().max().detach().cpu()),
        )
        shared = torch.nn.Parameter(shared_value.clone(), requires_grad=True)
        for owner in owners:
            owner.log_s_in = shared

    scale_names = [name for group in groups for name in group["scale_names"]]
    return {
        "format": "ctox.recovery.fanout-s-in.v1",
        "policy": policy,
        "group_sha256": fanout_group_sha256(groups),
        "groups": groups,
        "group_count": len(groups),
        "logical_s_in_tensors": len(scale_names),
        "unique_s_in_parameters": len(groups),
        "a8_quantizations_avoided_per_complete_fanout_pass": len(scale_names)
        - len(groups),
        "maximum_initial_log_delta": maximum_initial_log_delta,
        "initializer_merge": "geometric_mean" if groups else "none",
    }


def validate_parameter_aliases(
    parameters: dict[str, Any],
    fanout_evidence: dict[str, Any],
) -> None:
    expected = {
        frozenset(str(name) for name in group["scale_names"])
        for group in fanout_evidence.get("groups", [])
    }
    by_identity: dict[int, set[str]] = {}
    for name, parameter in parameters.items():
        by_identity.setdefault(id(parameter), set()).add(name)
    observed = {
        frozenset(names) for names in by_identity.values() if len(names) > 1
    }
    if observed != expected:
        missing = sorted(sorted(group) for group in expected - observed)
        extra = sorted(sorted(group) for group in observed - expected)
        raise ValueError(
            f"recovery parameter aliases differ: missing={missing[:2]} extra={extra[:2]}"
        )
    if any(any(not name.endswith(".weight.s_in") for name in group) for group in observed):
        raise ValueError("only recovery s_in tensors may share a parameter")
