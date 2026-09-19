#!/usr/bin/env python
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "05y0b-realized-rho-and-frontier-schema-preflight-v3-no-cli"
STATUS = "READ_ONLY_SUPPLEMENTAL_PREFLIGHT_NO_DIAGNOSTIC_RESULT_COMPUTED"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

DIR_05Y0 = PAPER5_ROOT / "method_contract" / "05y0"
PREFLIGHT_05Y0 = DIR_05Y0 / "realized_rho_preflight.json"
LOCK_05Y0 = DIR_05Y0 / "05y0_preflight_lock.json"

PROGRESS_05F2 = PAPER5_ROOT / "method_contract" / "05f2" / "full_run_progress_manifest.csv"
LOCK_05F3E = PAPER5_ROOT / "method_contract" / "05f3e" / "final_gate_results_lock.json"
LOCK_05X2B = PAPER5_ROOT / "method_contract" / "05x2b" / "05x2b_results_lock.json"

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05y0b"
OUTPUT_REPORT = OUTPUT_DIR / "diagnostic_schema_preflight.json"
OUTPUT_README = OUTPUT_DIR / "05y0b_README.txt"
OUTPUT_LOCK = OUTPUT_DIR / "05y0b_preflight_lock.json"

G19 = "G19_CORRELATED_PRIVATE_NULL_N62"
EXPECTED_G19_DATASETS = 200

FRONTIER_FIELDS = (
    "median_aligned_correlation",
    "q10_aligned_correlation",
    "inclusion_frequency",
    "passes_stability_gate",
)

TRUTH_CONTEXT_FIELDS = (
    "biological_truth_match",
    "technical_truth_match",
    "truth_factor_class",
    "primary_truth_target",
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


def resolve_project_path(raw: str) -> Path:
    p = Path(str(raw))
    return p if p.is_absolute() else PROJECT_ROOT / p


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


def is_nonnull(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "nan", "none", "null", "na"}
    return True


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (1, 1.0):
        return True
    if value in (0, 0.0):
        return False
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes"}:
        return True
    if text in {"false", "f", "0", "no"}:
        return False
    raise RuntimeError(f"Cannot parse boolean value: {value!r}")


def nested_get(root: Any, dotted_path: str) -> tuple[bool, Any]:
    current = root
    parts = dotted_path.split(".")
    for raw_part in parts:
        if raw_part.endswith("[0]"):
            key = raw_part[:-3]
            if not isinstance(current, dict) or key not in current:
                return False, None
            current = current[key]
            if not isinstance(current, list) or not current:
                return False, None
            current = current[0]
        else:
            if not isinstance(current, dict) or raw_part not in current:
                return False, None
            current = current[raw_part]
    return True, current


def leaf_name(path: str) -> str:
    value = path.split(".")[-1]
    return value.replace("[0]", "")


def selected_rows_for_scenarios(
    scenario_ids: set[str],
) -> tuple[dict[str, list[dict[str, str]]], dict[str, int]]:
    require_file(PROGRESS_05F2)
    rows: dict[str, list[dict[str, str]]] = {sid: [] for sid in scenario_ids}
    all_counts: dict[str, int] = {}
    total_rows = 0

    with PROGRESS_05F2.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset_key", "scenario_id", "shard_path", "shard_sha256"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                "05f2 progress manifest lacks: " + ", ".join(sorted(missing))
            )
        for row in reader:
            total_rows += 1
            sid = str(row["scenario_id"]).strip()
            all_counts[sid] = all_counts.get(sid, 0) + 1
            if sid in rows:
                rows[sid].append(row)

    if total_rows != 3100:
        raise RuntimeError(
            f"05f2 progress manifest has {total_rows} rows; expected frozen total 3100."
        )

    missing_scenarios = [sid for sid, part in rows.items() if len(part) == 0]
    if missing_scenarios:
        raise RuntimeError(
            "Requested scenarios are absent from 05f2 progress manifest: "
            + ", ".join(sorted(missing_scenarios))
        )

    # G19 membership/count is already part of the frozen D01 audit and therefore
    # remains a hard integrity check. STANDARD is used here only for blind schema
    # availability, so its actual frozen count is read from the completed 05f2
    # manifest rather than being guessed.
    if len(rows[G19]) != EXPECTED_G19_DATASETS:
        raise RuntimeError(
            f"Expected frozen G19 count {EXPECTED_G19_DATASETS}; found {len(rows[G19])}."
        )

    return rows, all_counts


def main() -> None:
    print("=" * 94)
    print("Paper 5 - supplemental blind schema preflight for 05y failure diagnostics (v3)")
    print("=" * 94)
    print(f"Script version: {SCRIPT_VERSION}")
    print("Failure labels inspected: NO")
    print("70-vs-130 comparison computed: NO")
    print("Threshold frontier computed: NO")
    print("Model fitting: NO")
    print("GPU computation: NO")
    print()

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(
            f"05y0a output directory already exists and is non-empty: {OUTPUT_DIR}. "
            "Do not overwrite a versioned preflight."
        )

    verify_lock(
        LOCK_05Y0,
        "READ_ONLY_PREFLIGHT_NO_DIAGNOSTIC_RESULT_COMPUTED",
    )
    verify_lock(
        LOCK_05F3E,
        "FROZEN_FINAL_GATE_SCIENTIFIC_RESULTS_NO_RETUNING",
    )
    verify_lock(
        LOCK_05X2B,
        "POSTUNBLINDING_EXPLORATORY_RETENTION_ABLATION_COMPLETE",
    )

    preflight = load_json(PREFLIGHT_05Y0)
    direct = list(preflight.get("direct_realized_private_correlation_candidates", []))
    if len(direct) != 1:
        raise RuntimeError(
            "05y0 did not identify exactly one direct realized-private-correlation "
            f"candidate. Candidates: {direct}"
        )
    candidate_path = str(direct[0])
    candidate_leaf = leaf_name(candidate_path)
    standard_id = str(preflight.get("scenario_ids", {}).get("standard", "")).strip()
    if not standard_id:
        raise RuntimeError("05y0 preflight does not contain the frozen STANDARD scenario ID.")

    scenario_rows, scenario_counts = selected_rows_for_scenarios({G19, standard_id})
    rows = scenario_rows[G19]
    standard_dataset_count = len(scenario_rows[standard_id])

    dataset_candidate_present = 0
    dataset_candidate_nonnull = 0
    dataset_candidate_numeric_finite = 0

    factor_candidate_present = 0
    factor_candidate_nonnull = 0
    factor_candidate_numeric_finite = 0

    n_factor_rows = 0
    n_retained = 0
    n_nonretained = 0

    frontier_present = Counter()
    frontier_nonnull = Counter()
    frontier_nonnull_nonretained = Counter()

    truth_present = Counter()
    truth_nonnull = Counter()

    candidate_direct_path_present = 0
    candidate_direct_path_nonnull = 0
    candidate_direct_path_numeric_finite = 0

    for idx, row in enumerate(rows, start=1):
        shard_path = resolve_project_path(row["shard_path"])
        require_file(shard_path)

        data = shard_path.read_bytes()
        actual_hash = hashlib.sha256(data).hexdigest()
        expected_hash = str(row["shard_sha256"]).strip()
        if actual_hash.lower() != expected_hash.lower():
            raise RuntimeError(f"Shard hash mismatch: {shard_path}")

        shard = json.loads(data)

        exists, direct_value = nested_get(shard, candidate_path)
        if exists:
            candidate_direct_path_present += 1
            if is_nonnull(direct_value):
                candidate_direct_path_nonnull += 1
                try:
                    numeric = float(direct_value)
                    if math.isfinite(numeric):
                        candidate_direct_path_numeric_finite += 1
                except Exception:
                    pass

        payload = shard.get("scientific_payload")
        if not isinstance(payload, dict):
            raise RuntimeError(f"Shard lacks scientific_payload: {shard_path}")

        dataset = payload.get("dataset_row")
        factors = payload.get("factor_rows")
        if not isinstance(dataset, dict) or not isinstance(factors, list):
            raise RuntimeError(f"Unexpected scientific_payload schema: {shard_path}")

        # Check the same candidate leaf explicitly at dataset level.
        if candidate_leaf in dataset:
            dataset_candidate_present += 1
            value = dataset[candidate_leaf]
            if is_nonnull(value):
                dataset_candidate_nonnull += 1
                try:
                    if math.isfinite(float(value)):
                        dataset_candidate_numeric_finite += 1
                except Exception:
                    pass

        for factor in factors:
            if not isinstance(factor, dict):
                raise RuntimeError(f"Non-object factor row: {shard_path}")

            n_factor_rows += 1

            if candidate_leaf in factor:
                factor_candidate_present += 1
                value = factor[candidate_leaf]
                if is_nonnull(value):
                    factor_candidate_nonnull += 1
                    try:
                        if math.isfinite(float(value)):
                            factor_candidate_numeric_finite += 1
                    except Exception:
                        pass

            for field in FRONTIER_FIELDS:
                if field in factor:
                    frontier_present[field] += 1
                    if is_nonnull(factor[field]):
                        frontier_nonnull[field] += 1

            for field in TRUTH_CONTEXT_FIELDS:
                if field in factor:
                    truth_present[field] += 1
                    if is_nonnull(factor[field]):
                        truth_nonnull[field] += 1

            if "passes_stability_gate" not in factor:
                raise RuntimeError(
                    f"Factor row lacks passes_stability_gate: {shard_path}"
                )
            retained = as_bool(factor["passes_stability_gate"])
            if retained:
                n_retained += 1
            else:
                n_nonretained += 1
                for field in FRONTIER_FIELDS[:3]:
                    if field in factor and is_nonnull(factor[field]):
                        frontier_nonnull_nonretained[field] += 1

        if idx == 1 or idx % 50 == 0 or idx == len(rows):
            print(f"Schema verification: {idx:03d}/{len(rows):03d} G19 shards")

    if n_factor_rows <= 0:
        raise RuntimeError("No factor rows found in G19 shards.")

    # Blind schema/count check on frozen STANDARD. This does not inspect failure labels.
    standard_factor_rows = 0
    standard_retained = 0
    standard_nonretained = 0
    standard_frontier_nonnull = Counter()
    standard_frontier_nonnull_nonretained = Counter()
    standard_truth_nonnull = Counter()

    for idx, row in enumerate(scenario_rows[standard_id], start=1):
        shard_path = resolve_project_path(row["shard_path"])
        require_file(shard_path)
        data = shard_path.read_bytes()
        actual_hash = hashlib.sha256(data).hexdigest()
        expected_hash = str(row["shard_sha256"]).strip()
        if actual_hash.lower() != expected_hash.lower():
            raise RuntimeError(f"STANDARD shard hash mismatch: {shard_path}")
        shard = json.loads(data)
        payload = shard.get("scientific_payload")
        if not isinstance(payload, dict) or not isinstance(payload.get("factor_rows"), list):
            raise RuntimeError(f"Unexpected STANDARD payload schema: {shard_path}")
        for factor in payload["factor_rows"]:
            if not isinstance(factor, dict):
                raise RuntimeError(f"Non-object STANDARD factor row: {shard_path}")
            standard_factor_rows += 1
            for field in FRONTIER_FIELDS[:3]:
                if field in factor and is_nonnull(factor[field]):
                    standard_frontier_nonnull[field] += 1
            for field in TRUTH_CONTEXT_FIELDS:
                if field in factor and is_nonnull(factor[field]):
                    standard_truth_nonnull[field] += 1
            if "passes_stability_gate" not in factor:
                raise RuntimeError(f"STANDARD factor lacks passes_stability_gate: {shard_path}")
            retained = as_bool(factor["passes_stability_gate"])
            if retained:
                standard_retained += 1
            else:
                standard_nonretained += 1
                for field in FRONTIER_FIELDS[:3]:
                    if field in factor and is_nonnull(factor[field]):
                        standard_frontier_nonnull_nonretained[field] += 1

        if idx == 1 or idx % 50 == 0 or idx == standard_dataset_count:
            print(
                f"STANDARD schema verification: {idx:03d}/"
                f"{standard_dataset_count:03d} shards"
            )

    all_frontier_metrics_complete = all(
        frontier_nonnull[field] == n_factor_rows
        for field in FRONTIER_FIELDS[:3]
    )
    nonretained_frontier_metrics_complete = all(
        frontier_nonnull_nonretained[field] == n_nonretained
        for field in FRONTIER_FIELDS[:3]
    )

    if candidate_direct_path_nonnull == EXPECTED_G19_DATASETS:
        direct_candidate_scope = "ONE_NON_NULL_DIRECT_VALUE_PER_G19_DATASET"
    elif dataset_candidate_nonnull == EXPECTED_G19_DATASETS:
        direct_candidate_scope = "DATASET_ROW_LEAF_COMPLETE"
    elif factor_candidate_nonnull == n_factor_rows:
        direct_candidate_scope = "FACTOR_ROW_LEAF_COMPLETE"
    else:
        direct_candidate_scope = "INCOMPLETE_OR_AMBIGUOUS"

    frontier_bidirectional = bool(
        all_frontier_metrics_complete
        and nonretained_frontier_metrics_complete
        and n_nonretained > 0
    )

    report = {
        "script_version": SCRIPT_VERSION,
        "status": STATUS,
        "scientific_result_computed": False,
        "failure_labels_inspected": False,
        "threshold_frontier_computed": False,
        "realized_private_correlation_candidate": {
            "exact_path_from_05y0": candidate_path,
            "leaf_name": candidate_leaf,
            "direct_path_present_datasets": candidate_direct_path_present,
            "direct_path_nonnull_datasets": candidate_direct_path_nonnull,
            "direct_path_numeric_finite_datasets": candidate_direct_path_numeric_finite,
            "dataset_row_leaf_present_datasets": dataset_candidate_present,
            "dataset_row_leaf_nonnull_datasets": dataset_candidate_nonnull,
            "dataset_row_leaf_numeric_finite_datasets": dataset_candidate_numeric_finite,
            "factor_row_leaf_present_rows": factor_candidate_present,
            "factor_row_leaf_nonnull_rows": factor_candidate_nonnull,
            "factor_row_leaf_numeric_finite_rows": factor_candidate_numeric_finite,
            "scope_interpretation": direct_candidate_scope,
            "seed_reconstruction_required": False,
        },
        "g19_factor_payload": {
            "datasets": EXPECTED_G19_DATASETS,
            "factor_rows": n_factor_rows,
            "retained_factor_rows": n_retained,
            "nonretained_factor_rows": n_nonretained,
            "frontier_field_present_rows": dict(frontier_present),
            "frontier_field_nonnull_rows": dict(frontier_nonnull),
            "frontier_metrics_nonnull_in_nonretained_rows": dict(
                frontier_nonnull_nonretained
            ),
            "all_factor_rows_have_median_q10_inclusion": all_frontier_metrics_complete,
            "all_nonretained_factor_rows_have_median_q10_inclusion": (
                nonretained_frontier_metrics_complete
            ),
            "bidirectional_threshold_frontier_available_from_sealed_payload": (
                frontier_bidirectional
            ),
            "truth_context_field_present_rows": dict(truth_present),
            "truth_context_field_nonnull_rows": dict(truth_nonnull),
        },
        "standard_factor_payload": {
            "scenario_id": standard_id,
            "datasets": standard_dataset_count,
            "factor_rows": standard_factor_rows,
            "retained_factor_rows": standard_retained,
            "nonretained_factor_rows": standard_nonretained,
            "frontier_metrics_nonnull_rows": dict(standard_frontier_nonnull),
            "frontier_metrics_nonnull_in_nonretained_rows": dict(
                standard_frontier_nonnull_nonretained
            ),
            "truth_context_field_nonnull_rows": dict(standard_truth_nonnull),
            "matched_nonretained_scenario_signature_control_schema_available": bool(
                standard_nonretained > 0
                and all(
                    standard_frontier_nonnull_nonretained[field] == standard_nonretained
                    for field in FRONTIER_FIELDS[:3]
                )
                and n_nonretained > 0
            ),
        },
        "standard_count_source": {
            "source": relative_path(PROGRESS_05F2),
            "scenario_id": standard_id,
            "frozen_dataset_count": standard_dataset_count,
            "reason_not_hardcoded": (
                "STANDARD is used only as a secondary blind schema/control source in "
                "this preflight. Its frozen completed-run count is therefore read from "
                "the authoritative 05f2 progress manifest rather than assumed to be 200."
            ),
        },
        "next_stage_guardrail": (
            "The exact realized-correlation source and the complete factor-level "
            "frontier schema are established without failure-label comparison. "
            "Any 70-vs-130 analysis, factor classifier, anchor calibration, or threshold "
            "sweep must be frozen in a new 05y1 contract before computation."
        ),
        "sources": {
            "05y0_preflight": file_record(PREFLIGHT_05Y0),
            "05y0_lock": file_record(LOCK_05Y0),
            "05f3e_lock": file_record(LOCK_05F3E),
            "05x2b_lock": file_record(LOCK_05X2B),
            "05f2_progress_manifest": file_record(PROGRESS_05F2),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report, OUTPUT_REPORT)

    readme = f"""Paper 5 05y0b — supplemental blind diagnostic-schema preflight
==================================================================

No failure labels were inspected and no diagnostic comparison was computed.

Realized private-correlation candidate
--------------------------------------
Exact path: {candidate_path}
Leaf: {candidate_leaf}
Scope: {direct_candidate_scope}
Seed reconstruction required: NO

G19 factor-row availability
---------------------------
Datasets: {EXPECTED_G19_DATASETS}
Factor rows: {n_factor_rows}
Retained rows: {n_retained}
Non-retained rows: {n_nonretained}

All factor rows have median/q10/inclusion:
  {all_frontier_metrics_complete}

All non-retained rows have median/q10/inclusion:
  {nonretained_frontier_metrics_complete}

Bidirectional threshold frontier available from sealed payload:
  {frontier_bidirectional}

Frozen STANDARD ({standard_id})
-------------------------------
Datasets: {standard_dataset_count}
Factor rows: {standard_factor_rows}
Retained rows: {standard_retained}
Non-retained rows: {standard_nonretained}

Matched non-retained scenario-signature control schema available:
  {bool(
      standard_nonretained > 0
      and all(
          standard_frontier_nonnull_nonretained[field] == standard_nonretained
          for field in FRONTIER_FIELDS[:3]
      )
      and n_nonretained > 0
  )}

Next step
---------
Freeze a new 05y1 diagnostic contract before computing any 70-vs-130,
factor-level classifier, anchor calibration, or threshold-frontier result.
"""
    write_text_atomic(readme, OUTPUT_README)

    lock = {
        "script_version": SCRIPT_VERSION,
        "script_path": relative_path(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "status": STATUS,
        "scientific_result_computed": False,
        "generated_files": {
            relative_path(OUTPUT_REPORT): file_record(OUTPUT_REPORT),
            relative_path(OUTPUT_README): file_record(OUTPUT_README),
        },
        "rewrite_policy": (
            "05y0b is a blind supplemental preflight. Do not add diagnostic "
            "calculations to this version after seeing labels."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    print()
    print("05y0b supplemental preflight: COMPLETE")
    print(f"Exact realized-rho candidate: {candidate_path}")
    print(f"Candidate scope: {direct_candidate_scope}")
    print(f"G19 factor rows: {n_factor_rows}")
    print(f"Retained / non-retained: {n_retained} / {n_nonretained}")
    print(
        "All factor rows have median/q10/inclusion:",
        "YES" if all_frontier_metrics_complete else "NO",
    )
    print(
        "All non-retained rows have median/q10/inclusion:",
        "YES" if nonretained_frontier_metrics_complete else "NO",
    )
    print(
        "Bidirectional threshold frontier available:",
        "YES" if frontier_bidirectional else "NO",
    )
    print(f"Frozen STANDARD scenario: {standard_id}")
    print(f"STANDARD frozen dataset count: {standard_dataset_count}")
    print(
        f"STANDARD retained / non-retained: {standard_retained} / {standard_nonretained}"
    )
    matched_control = bool(
        standard_nonretained > 0
        and all(
            standard_frontier_nonnull_nonretained[field] == standard_nonretained
            for field in FRONTIER_FIELDS[:3]
        )
        and n_nonretained > 0
    )
    print(
        "Matched non-retained scenario-signature control schema available:",
        "YES" if matched_control else "NO",
    )
    print("Seed reconstruction required: NO")
    print(f"Report: {relative_path(OUTPUT_REPORT)}")
    print(f"Lock:   {relative_path(OUTPUT_LOCK)}")
    print("Scientific diagnostic result computed: NO")


if __name__ == "__main__":
    main()
