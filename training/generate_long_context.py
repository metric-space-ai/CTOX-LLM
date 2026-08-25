#!/usr/bin/env python3
"""Generate deterministic, license-clean long-context retrieval examples.

The corpus is procedural rather than copied from an external document source.
Every block carries distinct facts and cross-references.  The answer requires
following a link between two records placed at controlled token positions, so
32K/64K/128K coverage cannot be satisfied by padding a short prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from build_manifest import canonical_text
from prompt_format import render_record


FORMAT = "ctox.synthetic-long-context.v1"
LICENSE = "apache-2.0"
DEFAULT_TARGETS = (32_768, 65_536, 131_072)
POSITION_PAIRS = ((0.05, 0.95), (0.25, 0.75), (0.50, 0.90), (0.90, 0.10))

SITES_EN = ("Berlin", "Hamburg", "Dublin", "Oslo", "Lisbon", "Tallinn")
SITES_DE = ("Berlin", "Hamburg", "Dresden", "Köln", "Leipzig", "München")
OWNERS = ("Aster", "Boreal", "Cobalt", "Delta", "Eider", "Fjord", "Ginkgo", "Helix")
STATES_EN = ("approved", "paused", "review", "closed", "scheduled")
STATES_DE = ("freigegeben", "pausiert", "prüfung", "abgeschlossen", "geplant")
MATERIALS_EN = ("sensor array", "power module", "cooling loop", "relay bank", "fiber trunk")
MATERIALS_DE = ("Sensorfeld", "Leistungsmodul", "Kühlkreis", "Relaisbank", "Glasfaserstrang")


def digest_int(*parts: Any) -> int:
    encoded = "\0".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def pick(values: tuple[str, ...], *parts: Any) -> str:
    return values[digest_int(*parts) % len(values)]


def normal_dossier(seed: str, sample_key: str, index: int, language: str) -> str:
    number = digest_int(seed, sample_key, index)
    dossier = f"D-{number % 10_000_000:07d}"
    linked = f"D-{digest_int(seed, sample_key, index, 'link') % 10_000_000:07d}"
    amount = 1000 + number % 89000
    checksum = hashlib.sha256(f"{sample_key}:{index}:{amount}".encode()).hexdigest()[:12]
    owner = pick(OWNERS, seed, sample_key, index, "owner")
    if language == "de":
        return (
            f"[AKTE {index:06d}]\n"
            f"aktenzeichen: {dossier}\nstandort: {pick(SITES_DE, seed, sample_key, index, 'site')}\n"
            f"verantwortung: Team {owner}\nstatus: {pick(STATES_DE, seed, sample_key, index, 'state')}\n"
            f"baugruppe: {pick(MATERIALS_DE, seed, sample_key, index, 'material')}\n"
            f"budget_eur: {amount}\nfolgeakte: {linked}\nprüfsumme: {checksum}\n"
            "notiz: Der Eintrag ist eigenständig; Aktenzeichen, Status, Betrag und Verweis "
            "dürfen nicht mit benachbarten Einträgen zusammengeführt werden.\n"
        )
    return (
        f"[DOSSIER {index:06d}]\n"
        f"dossier_id: {dossier}\nsite: {pick(SITES_EN, seed, sample_key, index, 'site')}\n"
        f"owner: Team {owner}\nstate: {pick(STATES_EN, seed, sample_key, index, 'state')}\n"
        f"assembly: {pick(MATERIALS_EN, seed, sample_key, index, 'material')}\n"
        f"budget_eur: {amount}\nlinked_dossier: {linked}\nchecksum: {checksum}\n"
        "note: This entry is independent; its identifier, state, amount, and link must not "
        "be merged with adjacent records.\n"
    )


def needle_records(seed: str, sample_key: str, language: str) -> tuple[str, str, str, dict[str, Any]]:
    serial = digest_int(seed, sample_key, "needle") % 1_000_000
    first_id = f"ZETA-{serial:06d}-A"
    second_id = f"ZETA-{serial:06d}-B"
    marker_a = f"NEEDLE-{sample_key}-A"
    marker_b = f"NEEDLE-{sample_key}-B"
    amount = 500_000 + digest_int(seed, sample_key, "amount") % 400_000
    code = hashlib.sha256(f"{seed}:{sample_key}:clearance".encode()).hexdigest()[:16].upper()
    site = pick(SITES_DE if language == "de" else SITES_EN, seed, sample_key, "needle-site")
    if language == "de":
        first = (
            f"[{marker_a}]\naktenzeichen: {first_id}\nfolgeakte: {second_id}\n"
            "anweisung: Löse die Folgeakte auf und melde deren Standort, Freigabecode und Budget.\n"
        )
        second = (
            f"[{marker_b}]\naktenzeichen: {second_id}\nstandort: {site}\n"
            f"freigabecode: {code}\nbudget_eur: {amount}\nstatus: freigegeben\n"
        )
        question = (
            f"Suche die Akte mit Marker {marker_a}. Folge ausschließlich ihrem Feld `folgeakte` "
            "zur Zielakte. Antworte als kompaktes JSON mit `aktenzeichen`, `standort`, "
            "`freigabecode` und `budget_eur`."
        )
        answer = {
            "aktenzeichen": second_id,
            "standort": site,
            "freigabecode": code,
            "budget_eur": amount,
        }
    else:
        first = (
            f"[{marker_a}]\ndossier_id: {first_id}\nlinked_dossier: {second_id}\n"
            "instruction: Resolve the linked dossier and report its site, clearance code, and budget.\n"
        )
        second = (
            f"[{marker_b}]\ndossier_id: {second_id}\nsite: {site}\n"
            f"clearance_code: {code}\nbudget_eur: {amount}\nstate: approved\n"
        )
        question = (
            f"Find the dossier carrying marker {marker_a}. Follow only its `linked_dossier` field "
            "to the target dossier. Return compact JSON with `dossier_id`, `site`, "
            "`clearance_code`, and `budget_eur`."
        )
        answer = {
            "dossier_id": second_id,
            "site": site,
            "clearance_code": code,
            "budget_eur": amount,
        }
    return first, second, question, answer


def build_record(
    seed: str,
    target_tokens: int,
    language: str,
    sample_index: int,
    dossier_count: int,
) -> tuple[dict[str, Any], list[str]]:
    sample_key = f"{language}-{target_tokens}-{sample_index:02d}"
    first, second, question, answer = needle_records(seed, sample_key, language)
    position_pair = POSITION_PAIRS[sample_index % len(POSITION_PAIRS)]
    first_index = min(dossier_count - 1, max(0, round(position_pair[0] * (dossier_count - 1))))
    second_index = min(dossier_count - 1, max(0, round(position_pair[1] * (dossier_count - 1))))
    if first_index == second_index:
        second_index = (second_index + max(1, dossier_count // 2)) % dossier_count
    blocks = [normal_dossier(seed, sample_key, index, language) for index in range(dossier_count)]
    blocks[first_index] = first
    blocks[second_index] = second
    heading = (
        "Referenzakten. Jeder abgegrenzte Eintrag enthält unabhängige Felder.\n\n"
        if language == "de"
        else "Reference dossiers. Every delimited entry contains independent fields.\n\n"
    )
    context = heading + "\n---\n".join(blocks)
    user = f"{context}\n\nAUFGABE:\n{question}" if language == "de" else f"{context}\n\nTASK:\n{question}"
    record = {
        "messages": [
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": json.dumps(answer, ensure_ascii=False, separators=(",", ":")),
            },
        ]
    }
    # Include the delimiters in the searched marker so the first match is the
    # context dossier, not the marker repeated later in the user question.
    markers = [f"[NEEDLE-{sample_key}-A]", f"[NEEDLE-{sample_key}-B]"]
    return record, markers


def token_ids(tokenizer: Any, record: dict[str, Any]) -> list[int]:
    rendered = render_record(tokenizer, record)
    encoded = tokenizer(rendered, add_special_tokens=False)
    ids = encoded.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def marker_position(tokenizer: Any, ids: list[int], marker: str) -> int:
    marker_ids = tokenizer(marker, add_special_tokens=False).input_ids
    if marker_ids and isinstance(marker_ids[0], list):
        marker_ids = marker_ids[0]
    for start in range(0, len(ids) - len(marker_ids) + 1):
        if ids[start : start + len(marker_ids)] == list(marker_ids):
            return start
    raise RuntimeError(f"rendered record does not contain marker {marker}")


def sized_example(
    tokenizer: Any,
    seed: str,
    target_tokens: int,
    language: str,
    sample_index: int,
    tolerance: int,
) -> tuple[dict[str, Any], list[int], list[str]]:
    if target_tokens <= tolerance:
        raise ValueError("target_tokens must be larger than tolerance")
    probe, _ = build_record(seed, target_tokens, language, sample_index, 128)
    probe_tokens = len(token_ids(tokenizer, probe))
    if probe_tokens <= 0:
        raise RuntimeError("tokenizer produced no tokens for probe")
    dossier_count = max(16, math.floor(128 * target_tokens / probe_tokens))
    for _ in range(16):
        record, markers = build_record(seed, target_tokens, language, sample_index, dossier_count)
        ids = token_ids(tokenizer, record)
        if target_tokens - tolerance <= len(ids) <= target_tokens:
            return record, ids, markers
        per_dossier = max(1.0, len(ids) / dossier_count)
        delta = target_tokens - tolerance // 2 - len(ids)
        adjustment = math.trunc(delta / per_dossier)
        if adjustment == 0:
            adjustment = 1 if delta > 0 else -1
        dossier_count = max(16, dossier_count + adjustment)
    raise RuntimeError(
        f"could not size {language} sample {sample_index} to {target_tokens}±{tolerance} tokens"
    )


def generated_record(
    tokenizer: Any,
    seed: str,
    target_tokens: int,
    language: str,
    sample_index: int,
    tolerance: int,
    source_revision: str,
    split: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, ids, markers = sized_example(
        tokenizer, seed, target_tokens, language, sample_index, tolerance
    )
    prompt_sha = hashlib.sha256(canonical_text(payload).encode("utf-8")).hexdigest()
    source_id = f"{split}-{language}-{target_tokens}-{sample_index:02d}-{seed}"
    identity = "\0".join((FORMAT, source_revision, source_id, prompt_sha))
    sample_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    marker_offsets = [marker_position(tokenizer, ids, marker) for marker in markers]
    common = {
        "id": sample_id,
        "source_repo": "metric-space-ai/CTOX-LLM",
        "source_revision": source_revision,
        "subset": FORMAT,
        "split": split,
        "source_id": source_id,
        "license": LICENSE,
        "dataset_card_licenses": [LICENSE],
        "generator": FORMAT,
        "generator_seed": seed,
        "category": "long_context",
        "language": language,
        "prompt_sha256": prompt_sha,
        "release_eligible": True,
        "quarantine_reason": None,
        "target_tokens": target_tokens,
        "rendered_tokens": len(ids),
        "marker_token_offsets": marker_offsets,
        "marker_normalized_positions": [round(offset / len(ids), 8) for offset in marker_offsets],
    }
    materialized = dict(common)
    materialized.update(payload)
    return common, materialized


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".partial",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            for row in rows:
                temporary.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.rename(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--split", choices=("calibration", "evaluation"), required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--targets", default=",".join(map(str, DEFAULT_TARGETS)))
    parser.add_argument("--languages", default="en,de")
    parser.add_argument("--samples-per-target", type=int, default=2)
    parser.add_argument("--sample-index-start", type=int, default=0)
    parser.add_argument("--tolerance", type=int, default=256)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.manifest == args.output:
        raise SystemExit("--manifest and --output must be different files")
    if args.samples_per_target <= 0 or args.sample_index_start < 0 or args.tolerance <= 0:
        raise SystemExit(
            "--samples-per-target and --tolerance must be positive; "
            "--sample-index-start must be non-negative"
        )
    targets = tuple(int(value) for value in args.targets.split(",") if value)
    languages = tuple(value for value in args.languages.split(",") if value)
    if not targets or not languages or any(language not in {"en", "de"} for language in languages):
        raise SystemExit("targets must be integers and languages must be en and/or de")

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit("install training/requirements.in before generating long context") from error

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
    manifests = []
    materialized = []
    for target in targets:
        for language in languages:
            for sample_index in range(
                args.sample_index_start,
                args.sample_index_start + args.samples_per_target,
            ):
                manifest, sample = generated_record(
                    tokenizer,
                    args.seed,
                    target,
                    language,
                    sample_index,
                    args.tolerance,
                    args.source_revision,
                    args.split,
                )
                manifests.append(manifest)
                materialized.append(sample)
                print(
                    json.dumps(
                        {
                            "id": manifest["id"],
                            "language": language,
                            "target_tokens": target,
                            "rendered_tokens": manifest["rendered_tokens"],
                            "marker_normalized_positions": manifest[
                                "marker_normalized_positions"
                            ],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    atomic_jsonl(args.manifest, manifests)
    try:
        atomic_jsonl(args.output, materialized)
    except BaseException:
        args.manifest.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
