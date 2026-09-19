#!/usr/bin/env python
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "05y0c-semantic-realized-private-correlation-source-preflight-v4-no-cli"
STATUS = "READ_ONLY_SEMANTIC_SOURCE_PREFLIGHT_NO_DIAGNOSTIC_RESULT_COMPUTED"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

DIR_05Y0 = PAPER5_ROOT / "method_contract" / "05y0"
REPORT_05Y0 = DIR_05Y0 / "realized_rho_preflight.json"
LOCK_05Y0 = DIR_05Y0 / "05y0_preflight_lock.json"

DIR_05Y0B = PAPER5_ROOT / "method_contract" / "05y0b"
REPORT_05Y0B = DIR_05Y0B / "diagnostic_schema_preflight.json"
LOCK_05Y0B = DIR_05Y0B / "05y0b_preflight_lock.json"

SCENARIO_REGISTRY = (
    PAPER5_ROOT / "method_contract" / "05a1" / "final_gate_calibration_scenarios.csv"
)
PROGRESS_05F2 = PAPER5_ROOT / "method_contract" / "05f2" / "full_run_progress_manifest.csv"

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05y0c"
OUTPUT_REPORT = OUTPUT_DIR / "realized_private_correlation_source_preflight.json"
OUTPUT_SOURCE_CONTEXT = OUTPUT_DIR / "realized_private_correlation_source_context.txt"
OUTPUT_README = OUTPUT_DIR / "05y0c_README.txt"
OUTPUT_LOCK = OUTPUT_DIR / "05y0c_preflight_lock.json"

G19 = "G19_CORRELATED_PRIVATE_NULL_N62"

SEARCH_TERMS = (
    "best_private_absolute_correlation",
    "correlated_private_null",
    "correlated_private",
    "private_correlation",
    "private_corr",
    "private_rho",
    "rho_private",
    "realized_private",
    "private_latent",
    "private_factor",
)

SCHEMA_TOKENS = (
    "rho",
    "corr",
    "correlation",
    "private",
    "latent",
    "cov",
    "seed",
    "input",
    "path",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".txt", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def verify_lock(path: Path, expected_status: str) -> dict[str, Any]:
    require_file(path)
    lock = load_json(path)
    actual = str(lock.get("status", "")).strip()
    if actual != expected_status:
        raise RuntimeError(
            f"{relative_path(path)} status mismatch: {actual!r} != {expected_status!r}"
        )
    return lock


def read_g19_scenario_row() -> dict[str, str]:
    require_file(SCENARIO_REGISTRY)
    with SCENARIO_REGISTRY.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if str(row.get("scenario_id", "")).strip() == G19
        ]
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one {G19} scenario row; found {len(rows)}.")
    return rows[0]


def first_g19_shard() -> tuple[dict[str, str], Path, dict[str, Any]]:
    require_file(PROGRESS_05F2)
    selected = None
    with PROGRESS_05F2.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("scenario_id", "")).strip() == G19:
                selected = row
                break
    if selected is None:
        raise RuntimeError(f"No {G19} row in 05f2 progress manifest.")

    raw = Path(str(selected["shard_path"]))
    shard_path = raw if raw.is_absolute() else PROJECT_ROOT / raw
    require_file(shard_path)

    data = shard_path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    expected = str(selected["shard_sha256"]).strip()
    if actual.lower() != expected.lower():
        raise RuntimeError("First G19 shard SHA256 mismatch.")

    return selected, shard_path, json.loads(data)


def relevant_schema_keys(mapping: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, value in mapping.items():
        low = str(key).lower()
        if any(token in low for token in SCHEMA_TOKENS):
            rows.append(
                {
                    "field": str(key),
                    "type": type(value).__name__,
                    "nonnull": value is not None and not (
                        isinstance(value, str)
                        and value.strip().lower() in {"", "nan", "none", "null", "na"}
                    ),
                }
            )
    return sorted(rows, key=lambda x: x["field"])


def source_files() -> list[Path]:
    patterns = (
        "05b*.py",
        "05d2v2*.py",
        "05f0*.py",
        "05f1*.py",
        "05f2*.py",
    )
    found: dict[str, Path] = {}
    for pattern in patterns:
        for path in PAPER5_ROOT.joinpath("scripts").glob(pattern):
            if path.is_file():
                found[str(path.resolve()).lower()] = path
    return sorted(found.values(), key=lambda p: p.name.lower())


def context_windows(path: Path, terms: tuple[str, ...], radius: int = 10) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    hit_indices: set[int] = set()

    for idx, line in enumerate(lines):
        low = line.lower()
        if any(term.lower() in low for term in terms):
            hit_indices.add(idx)

    windows: list[tuple[int, int]] = []
    for idx in sorted(hit_indices):
        start = max(0, idx - radius)
        end = min(len(lines), idx + radius + 1)
        if windows and start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], end))
        else:
            windows.append((start, end))

    output = []
    for start, end in windows:
        output.append(
            {
                "start_line": start + 1,
                "end_line": end,
                "lines": [
                    f"{line_no + 1:05d}: {lines[line_no]}"
                    for line_no in range(start, end)
                ],
            }
        )
    return output


def main() -> None:
    print("=" * 96)
    print("Paper 5 - semantic source preflight for realized correlated-private rho")
    print("=" * 96)
    print(f"Script version: {SCRIPT_VERSION}")
    print("Failure labels inspected: NO")
    print("70-vs-130 comparison computed: NO")
    print("Threshold frontier computed: NO")
    print("Scientific numeric candidate values printed: NO")
    print("Model fitting: NO")
    print("GPU computation: NO")
    print()

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(
            f"05y0c output directory already exists and is non-empty: {OUTPUT_DIR}. "
            "Do not overwrite a versioned preflight."
        )

    verify_lock(
        LOCK_05Y0,
        "READ_ONLY_PREFLIGHT_NO_DIAGNOSTIC_RESULT_COMPUTED",
    )
    verify_lock(
        LOCK_05Y0B,
        "READ_ONLY_SUPPLEMENTAL_PREFLIGHT_NO_DIAGNOSTIC_RESULT_COMPUTED",
    )

    report_05y0 = load_json(REPORT_05Y0)
    report_05y0b = load_json(REPORT_05Y0B)

    candidates = list(
        report_05y0.get("direct_realized_private_correlation_candidates", [])
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "05y0 does not contain exactly one candidate path: "
            + repr(candidates)
        )
    candidate_path = str(candidates[0])
    candidate_leaf = candidate_path.split(".")[-1].replace("[0]", "")

    g19_row = read_g19_scenario_row()
    _, shard_path, shard = first_g19_shard()

    payload = shard.get("scientific_payload")
    if not isinstance(payload, dict):
        raise RuntimeError("First G19 shard lacks scientific_payload.")
    dataset_row = payload.get("dataset_row")
    factor_rows = payload.get("factor_rows")
    if not isinstance(dataset_row, dict) or not isinstance(factor_rows, list) or not factor_rows:
        raise RuntimeError("Unexpected first G19 scientific_payload schema.")

    # Structural classification of the 05y0 candidate.
    factor_present = sum(
        isinstance(row, dict) and candidate_leaf in row
        for row in factor_rows
    )
    factor_nonnull = sum(
        isinstance(row, dict)
        and candidate_leaf in row
        and row[candidate_leaf] is not None
        for row in factor_rows
    )
    dataset_present = candidate_leaf in dataset_row
    dataset_nonnull = dataset_present and dataset_row[candidate_leaf] is not None

    if "[0]" in candidate_path and factor_present == len(factor_rows):
        structural_classification = (
            "FIRST_FACTOR_PROJECTION_OF_A_FACTOR_LEVEL_FIELD__NOT_A_UNIQUE_DATASET_SCALAR"
        )
        permitted_as_primary_realized_rho = False
    elif dataset_present and dataset_nonnull and factor_present == 0:
        structural_classification = "DATASET_LEVEL_SCALAR_CANDIDATE"
        permitted_as_primary_realized_rho = True
    else:
        structural_classification = "AMBIGUOUS_SCOPE"
        permitted_as_primary_realized_rho = False

    # Scenario-registry metadata relevant to private/correlation mechanism.
    scenario_relevant = {
        key: value
        for key, value in g19_row.items()
        if any(token in str(key).lower() for token in ("private", "rho", "corr", "latent"))
        or any(token in str(value).lower() for token in ("private", "rho", "corr"))
    }

    source_context_records = []
    context_text_blocks = []

    for path in source_files():
        windows = context_windows(path, SEARCH_TERMS)
        if not windows:
            continue

        source_context_records.append(
            {
                "path": relative_path(path),
                "sha256": sha256_file(path),
                "n_windows": len(windows),
            }
        )
        context_text_blocks.append("=" * 110)
        context_text_blocks.append(f"SOURCE: {relative_path(path)}")
        context_text_blocks.append(f"SHA256: {sha256_file(path)}")
        context_text_blocks.append("=" * 110)

        for window in windows:
            context_text_blocks.append(
                f"\n--- lines {window['start_line']}-{window['end_line']} ---"
            )
            context_text_blocks.extend(window["lines"])
        context_text_blocks.append("")

    source_context_text = "\n".join(context_text_blocks) + "\n"

    # Search source context semantically enough to flag likely truth-matching usage,
    # but do not claim more than the source text supports.
    low_context = source_context_text.lower()
    best_private_context_present = "best_private_absolute_correlation" in low_context
    nearby_truth_match_tokens = any(
        token in low_context
        for token in (
            "private_truth",
            "truth_match",
            "best_private",
            "absolute correlation",
            "biological_truth",
        )
    )

    factor_schema = relevant_schema_keys(factor_rows[0])
    dataset_schema = relevant_schema_keys(dataset_row)

    report = {
        "script_version": SCRIPT_VERSION,
        "status": STATUS,
        "scientific_result_computed": False,
        "failure_labels_inspected": False,
        "threshold_frontier_computed": False,
        "05y0_candidate": {
            "path": candidate_path,
            "leaf": candidate_leaf,
            "dataset_row_present": dataset_present,
            "dataset_row_nonnull": bool(dataset_nonnull),
            "factor_rows_present_in_first_shard": factor_present,
            "factor_rows_total_in_first_shard": len(factor_rows),
            "factor_rows_nonnull_in_first_shard": factor_nonnull,
            "structural_classification": structural_classification,
            "permitted_as_primary_realized_generator_rho": permitted_as_primary_realized_rho,
            "reason": (
                "A factor_rows[0] path is an arbitrary first-factor projection when the "
                "same field exists on all factor rows. It must not be treated as a unique "
                "dataset-level realized generator correlation without an explicit source definition."
            ),
        },
        "05y0b_schema_summary": {
            "g19_factor_rows": report_05y0b.get("g19_factor_payload", {}).get("factor_rows"),
            "g19_retained_factor_rows": report_05y0b.get("g19_factor_payload", {}).get(
                "retained_factor_rows"
            ),
            "g19_nonretained_factor_rows": report_05y0b.get("g19_factor_payload", {}).get(
                "nonretained_factor_rows"
            ),
            "bidirectional_frontier_available": report_05y0b.get(
                "g19_factor_payload", {}
            ).get("bidirectional_threshold_frontier_available_from_sealed_payload"),
            "standard_factor_rows": report_05y0b.get("standard_factor_payload", {}).get(
                "factor_rows"
            ),
            "standard_retained_factor_rows": report_05y0b.get(
                "standard_factor_payload", {}
            ).get("retained_factor_rows"),
            "standard_nonretained_factor_rows": report_05y0b.get(
                "standard_factor_payload", {}
            ).get("nonretained_factor_rows"),
        },
        "g19_scenario_registry_private_correlation_metadata": scenario_relevant,
        "first_g19_shard": {
            "path": relative_path(shard_path),
            "sha256": sha256_file(shard_path),
            "dataset_relevant_schema": dataset_schema,
            "factor0_relevant_schema": factor_schema,
        },
        "source_context": {
            "records": source_context_records,
            "best_private_identifier_found": best_private_context_present,
            "truth_matching_tokens_present_in_same_source_export": nearby_truth_match_tokens,
            "export_path": relative_path(OUTPUT_SOURCE_CONTEXT),
        },
        "decision": (
            "DO_NOT_FREEZE_05Y1_PRIMARY_REALIZED_RHO_USING_THE_05Y0_CANDIDATE"
            if not permitted_as_primary_realized_rho
            else "CANDIDATE_SCOPE_IS_DATASET_LEVEL__SOURCE_DEFINITION_STILL_MUST_BE_CONFIRMED"
        ),
        "next_step": (
            "Inspect the exported source context to identify either: "
            "(a) an explicit realized private cross-view latent correlation stored by the generator, "
            "or (b) the exact frozen generator function/RNG sequence needed to reconstruct it from "
            "the G19 seed. Freeze that source/formula before any 70-vs-130 calculation."
        ),
        "sources": {
            "05y0_report": file_record(REPORT_05Y0),
            "05y0_lock": file_record(LOCK_05Y0),
            "05y0b_report": file_record(REPORT_05Y0B),
            "05y0b_lock": file_record(LOCK_05Y0B),
            "scenario_registry": file_record(SCENARIO_REGISTRY),
            "05f2_progress_manifest": file_record(PROGRESS_05F2),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_text_atomic(source_context_text, OUTPUT_SOURCE_CONTEXT)
    write_json_atomic(report, OUTPUT_REPORT)

    readme = f"""Paper 5 05y0c — semantic source preflight
==============================================

No failure labels, 70-vs-130 association, threshold frontier, or model fit is computed.

05y0 candidate
--------------
Path: {candidate_path}
Structural classification:
  {structural_classification}

Dataset-row field present:
  {dataset_present}

Same leaf present on factor rows in first G19 shard:
  {factor_present}/{len(factor_rows)}

Permitted as the primary realized generator rho:
  {permitted_as_primary_realized_rho}

Why this matters
----------------
A factor_rows[0] value must not be promoted to a dataset-level realized generator
correlation merely because exactly one value is encountered per shard. If the field
exists on every factor row, factor 0 is only an arbitrary projection.

Next step
---------
Use the source-context export to identify the actual generator-level realized
private cross-view correlation or the exact deterministic reconstruction rule.
Do not freeze 05y1 primary realized-rho analysis until that source is unambiguous.
"""
    write_text_atomic(readme, OUTPUT_README)

    lock = {
        "script_version": SCRIPT_VERSION,
        "script_path": relative_path(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "status": STATUS,
        "scientific_result_computed": False,
        "generated_files": {
            relative_path(path): file_record(path)
            for path in [OUTPUT_REPORT, OUTPUT_SOURCE_CONTEXT, OUTPUT_README]
        },
        "rewrite_policy": (
            "05y0c is a read-only semantic-source preflight. Do not add diagnostic "
            "calculations after inspecting failure labels."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    print("05y0c semantic source preflight: COMPLETE")
    print(f"05y0 candidate: {candidate_path}")
    print(f"Structural classification: {structural_classification}")
    print(
        "Candidate permitted as primary realized generator rho:",
        "YES" if permitted_as_primary_realized_rho else "NO",
    )
    print(f"Dataset-row leaf present: {'YES' if dataset_present else 'NO'}")
    print(
        f"Factor-row leaf present in first G19 shard: "
        f"{factor_present}/{len(factor_rows)}"
    )
    print(f"Source context: {relative_path(OUTPUT_SOURCE_CONTEXT)}")
    print(f"Report:         {relative_path(OUTPUT_REPORT)}")
    print(f"Lock:           {relative_path(OUTPUT_LOCK)}")
    print("Scientific diagnostic result computed: NO")


if __name__ == "__main__":
    main()
