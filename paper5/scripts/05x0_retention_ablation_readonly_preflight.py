#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_VERSION = "05x0-retention-ablation-readonly-preflight-v1-no-cli"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

DIR_05F2 = PAPER5_ROOT / "method_contract" / "05f2"
PROGRESS_05F2 = DIR_05F2 / "full_run_progress_manifest.csv"

DIR_05F3E = PAPER5_ROOT / "method_contract" / "05f3e"
LOCK_05F3E = DIR_05F3E / "final_gate_results_lock.json"

DIR_05F4B = PAPER5_ROOT / "method_contract" / "05f4b"
LOCK_05F4B = DIR_05F4B / "05f4b_manuscript_assets_lock.json"

SOURCE_SCRIPTS = [
    PAPER5_ROOT / "scripts" / "05a_freeze_ai_method_contract.py",
    PAPER5_ROOT / "scripts" / "05d5_evaluate_sable_mo_stability_gpu.py",
    PAPER5_ROOT / "scripts" / "05e_compare_ablate_and_freeze_method.py",
    PAPER5_ROOT / "scripts" / "05f2_run_full_final_gate_gpu_resumable.py",
    PAPER5_ROOT / "scripts" / "05f3e_aggregate_final_gate_results_cpu_postunblinding_schema_recovery_v3.py",
]

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05x0"
OUTPUT_SCHEMA = OUTPUT_DIR / "sealed_factor_schema_inventory.csv"
OUTPUT_CANDIDATES = OUTPUT_DIR / "retention_ablation_candidate_fields.csv"
OUTPUT_CONTEXT = OUTPUT_DIR / "retention_rule_source_context.txt"
OUTPUT_RESULT = OUTPUT_DIR / "retention_ablation_preflight.json"
OUTPUT_README = OUTPUT_DIR / "05x0_README.txt"
OUTPUT_LOCK = OUTPUT_DIR / "05x0_preflight_lock.json"

EXPECTED_05F3E_STATUS = "FROZEN_FINAL_GATE_SCIENTIFIC_RESULTS_NO_RETUNING"
EXPECTED_DATASETS = 3100

FIELD_TOKENS = (
    "stability",
    "stable",
    "inclusion",
    "median",
    "q10",
    "quantile",
    "correlation",
    "corr",
    "variance",
    "support",
    "retain",
    "retention",
    "loading",
    "factor",
)

SOURCE_PATTERNS = (
    "passes_stability_gate",
    "stability",
    "inclusion",
    "q10",
    "median",
    "correlation",
    "retention",
    "supported_modal",
    "view_support",
    "factor_variance",
)

HIGH_VALUE_CANDIDATES = (
    "passes_stability_gate",
    "median_aligned_correlation",
    "q10_aligned_correlation",
    "inclusion_frequency",
    "factor_variance",
    "n_supported_modalities",
    "predicted_view_support",
    "supported_modalities",
    "view_support_exact",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def write_csv_atomic(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".csv", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        table.to_csv(tmp, index=False)
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


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")


def verify_05f3e_lock() -> dict[str, Any]:
    require_file(LOCK_05F3E)
    lock = load_json(LOCK_05F3E)

    if str(lock.get("status", "")).strip() != EXPECTED_05F3E_STATUS:
        raise RuntimeError("05f3e lock status mismatch.")
    if str(lock.get("core_status", "")).strip() != "CORE_PASS":
        raise RuntimeError("05f3e core status is not CORE_PASS.")

    generated = lock.get("generated_files")
    if not isinstance(generated, dict) or not generated:
        raise RuntimeError("05f3e lock lacks generated_files.")

    for rel, rec in generated.items():
        path = PROJECT_ROOT / Path(rel)
        require_file(path)
        if sha256_file(path).lower() != str(rec.get("sha256", "")).lower():
            raise RuntimeError(f"05f3e locked artifact hash mismatch: {path}")
        if int(path.stat().st_size) != int(rec.get("size_bytes", -1)):
            raise RuntimeError(f"05f3e locked artifact size mismatch: {path}")

    return lock


def verify_05f4b_lock_if_present() -> dict[str, Any] | None:
    if not LOCK_05F4B.is_file():
        return None

    lock = load_json(LOCK_05F4B)
    if str(lock.get("status", "")).strip() != (
        "READ_ONLY_MANUSCRIPT_ASSET_EXPORT_FROM_LOCKED_05F3E"
    ):
        raise RuntimeError("05f4b lock has unexpected status.")

    generated = lock.get("generated_files")
    if not isinstance(generated, dict):
        raise RuntimeError("05f4b lock lacks generated_files.")

    for rel, rec in generated.items():
        path = PROJECT_ROOT / Path(rel)
        require_file(path)
        if sha256_file(path).lower() != str(rec.get("sha256", "")).lower():
            raise RuntimeError(f"05f4b artifact hash mismatch: {path}")

    return lock


def resolve_project_path(raw: str) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def classify_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def scan_sealed_schema(progress: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    key_counts: Counter[str] = Counter()
    nonnull_counts: Counter[str] = Counter()
    types: dict[str, Counter[str]] = defaultdict(Counter)

    n_factor_rows = 0
    dataset_keys: Counter[str] = Counter()

    for ordinal, row in enumerate(progress.itertuples(index=False), start=1):
        shard_path = resolve_project_path(row.shard_path)
        require_file(shard_path)

        expected_hash = str(row.shard_sha256).strip()
        actual_hash = sha256_file(shard_path)
        if actual_hash.lower() != expected_hash.lower():
            raise RuntimeError(f"05f2 shard hash mismatch: {shard_path}")

        shard = load_json(shard_path)
        payload = shard.get("scientific_payload")
        if not isinstance(payload, dict):
            raise RuntimeError(f"Shard lacks scientific_payload: {shard_path}")

        dataset = payload.get("dataset_row")
        factors = payload.get("factor_rows")
        if not isinstance(dataset, dict) or not isinstance(factors, list):
            raise RuntimeError(f"Unexpected scientific payload schema: {shard_path}")

        dataset_keys.update(dataset.keys())

        for factor in factors:
            if not isinstance(factor, dict):
                raise RuntimeError(f"Factor row is not an object: {shard_path}")
            n_factor_rows += 1
            for key, value in factor.items():
                key_counts[key] += 1
                if value is not None:
                    nonnull_counts[key] += 1
                types[key][classify_value(value)] += 1

        if ordinal == 1 or ordinal % 500 == 0 or ordinal == len(progress):
            print(f"Schema scan: {ordinal:04d}/{len(progress):04d} shards verified/read")

    rows = []
    for key in sorted(key_counts):
        rows.append(
            {
                "field": key,
                "present_rows": int(key_counts[key]),
                "nonnull_rows": int(nonnull_counts[key]),
                "total_factor_rows": int(n_factor_rows),
                "present_in_all_factor_rows": bool(key_counts[key] == n_factor_rows),
                "types": ";".join(
                    f"{name}:{count}" for name, count in sorted(types[key].items())
                ),
                "matches_retention_token": bool(
                    any(token in key.lower() for token in FIELD_TOKENS)
                ),
                "high_value_candidate": bool(key in HIGH_VALUE_CANDIDATES),
            }
        )

    table = pd.DataFrame(rows)
    summary = {
        "n_datasets": int(len(progress)),
        "n_factor_rows": int(n_factor_rows),
        "n_factor_fields": int(len(key_counts)),
        "n_dataset_fields": int(len(dataset_keys)),
    }
    return table, summary


def source_context() -> tuple[str, list[dict[str, Any]]]:
    blocks: list[str] = []
    matches_summary: list[dict[str, Any]] = []

    for path in SOURCE_SCRIPTS:
        if not path.is_file():
            blocks.append(f"### MISSING SOURCE: {path}\n")
            continue

        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        hit_lines: set[int] = set()

        for idx, line in enumerate(lines):
            low = line.lower()
            if any(pattern.lower() in low for pattern in SOURCE_PATTERNS):
                hit_lines.add(idx)

        windows: list[tuple[int, int]] = []
        for idx in sorted(hit_lines):
            start = max(0, idx - 5)
            end = min(len(lines), idx + 7)
            if windows and start <= windows[-1][1] + 1:
                windows[-1] = (windows[-1][0], max(windows[-1][1], end))
            else:
                windows.append((start, end))

        matches_summary.append(
            {
                "script": relative_path(path),
                "sha256": sha256_file(path),
                "n_matching_lines": len(hit_lines),
                "n_context_windows": len(windows),
            }
        )

        blocks.append("=" * 100)
        blocks.append(f"SOURCE: {relative_path(path)}")
        blocks.append(f"SHA256: {sha256_file(path)}")
        blocks.append("=" * 100)

        if not windows:
            blocks.append("No retention/stability-related source matches.")
            blocks.append("")
            continue

        for start, end in windows:
            blocks.append(f"\n--- lines {start+1}-{end} ---")
            for lineno in range(start, end):
                blocks.append(f"{lineno+1:05d}: {lines[lineno]}")
        blocks.append("")

    return "\n".join(blocks) + "\n", matches_summary


def build_candidate_table(schema: pd.DataFrame) -> pd.DataFrame:
    candidates = schema[
        schema["matches_retention_token"] | schema["high_value_candidate"]
    ].copy()

    priority = []
    for row in candidates.itertuples(index=False):
        field = str(row.field)
        low = field.lower()

        score = 0
        if field in HIGH_VALUE_CANDIDATES:
            score += 100
        if "stability" in low or "stable" in low:
            score += 50
        if "inclusion" in low:
            score += 40
        if "q10" in low or "quantile" in low:
            score += 35
        if "median" in low:
            score += 30
        if "correlation" in low or "corr" in low:
            score += 25
        if "support" in low:
            score += 20
        if "variance" in low:
            score += 15
        if bool(row.present_in_all_factor_rows):
            score += 5
        priority.append(score)

    candidates["inspection_priority"] = priority
    candidates = candidates.sort_values(
        ["inspection_priority", "field"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return candidates


def build_preflight_result(
    schema: pd.DataFrame,
    scan_summary: dict[str, Any],
    source_matches: list[dict[str, Any]],
    lock_05f4b_present: bool,
) -> dict[str, Any]:
    fields = set(schema["field"].astype(str))

    exact_presence = {
        field: {
            "present": bool(field in fields),
            "present_in_all_factor_rows": bool(
                schema.loc[
                    schema["field"].astype(str).eq(field),
                    "present_in_all_factor_rows",
                ].iloc[0]
            )
            if field in fields
            else False,
        }
        for field in HIGH_VALUE_CANDIDATES
    }

    stability_component_like = [
        field
        for field in fields
        if any(
            token in field.lower()
            for token in ("stability", "inclusion", "q10", "median", "correlation", "corr")
        )
    ]
    support_component_like = [
        field
        for field in fields
        if "support" in field.lower() or "modalit" in field.lower()
    ]

    plausibility = bool(stability_component_like and support_component_like)

    return {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": "READ_ONLY_PREFLIGHT_NO_EXPLORATORY_RESULT_COMPUTED",
        "purpose": (
            "Determine whether a post-unblinding exploratory SABLE-MO retention-rule "
            "ablation can be reconstructed from already-sealed 05f2 factor payloads "
            "without any model refit or GPU computation."
        ),
        "scientific_result_computed": False,
        "ablation_frequency_computed": False,
        "threshold_changed": False,
        "formal_gate_changed": False,
        "05f3e_modified": False,
        "05f4b_lock_present_and_verified": lock_05f4b_present,
        "scan_summary": scan_summary,
        "high_value_candidate_presence": exact_presence,
        "stability_component_like_fields": sorted(stability_component_like),
        "support_component_like_fields": sorted(support_component_like),
        "no_refit_ablation_plausible_from_schema": plausibility,
        "source_context_records": source_matches,
        "execution_decision": (
            "DO_NOT_COMPUTE_ABLATION_YET. Inspect the source-context export and candidate "
            "field inventory first. If the frozen source establishes an unambiguous "
            "stability-only rule from fields present in the sealed payload, define that "
            "rule in a separately versioned 05x1 exploratory contract before opening its "
            "result. If reconstruction requires refitting, defer the ablation and proceed "
            "to the frozen TARGET branch."
        ),
    }


def main() -> None:
    print("=" * 92)
    print("Paper 5 - read-only preflight for post-unblinding retention-rule ablation")
    print("=" * 92)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("GPU computation: NO")
    print("Model fitting: NO")
    print("Exploratory ablation result computed: NO")
    print("05f3e / 05f4b modification: NO")
    print()

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(
            f"05x0 output directory already exists and is non-empty: {OUTPUT_DIR}. "
            "Do not overwrite a versioned preflight."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    verify_05f3e_lock()
    print("05f3e scientific-results lock and covered artifacts: VERIFIED")

    lock_05f4b = verify_05f4b_lock_if_present()
    if lock_05f4b is not None:
        print("05f4b manuscript-asset lock and covered artifacts: VERIFIED")
    else:
        print("05f4b lock: NOT FOUND (non-fatal for 05x0)")

    require_file(PROGRESS_05F2)
    progress = pd.read_csv(PROGRESS_05F2)
    if progress.shape[0] != EXPECTED_DATASETS:
        raise RuntimeError(
            f"Expected {EXPECTED_DATASETS} rows in 05f2 progress manifest, "
            f"found {progress.shape[0]}."
        )

    required_cols = {"shard_path", "shard_sha256"}
    if not required_cols.issubset(progress.columns):
        raise RuntimeError(
            "05f2 progress manifest lacks required columns: "
            + ", ".join(sorted(required_cols - set(progress.columns)))
        )

    print()
    print("Scanning sealed factor-row schemas only...")
    schema, scan_summary = scan_sealed_schema(progress)
    candidates = build_candidate_table(schema)

    context_text, source_matches = source_context()
    result = build_preflight_result(
        schema=schema,
        scan_summary=scan_summary,
        source_matches=source_matches,
        lock_05f4b_present=bool(lock_05f4b is not None),
    )

    write_csv_atomic(schema, OUTPUT_SCHEMA)
    write_csv_atomic(candidates, OUTPUT_CANDIDATES)
    write_text_atomic(context_text, OUTPUT_CONTEXT)
    write_json_atomic(result, OUTPUT_RESULT)

    readme = f"""Paper 5 05x0 — read-only retention-ablation preflight
=========================================================

Script version
--------------
{SCRIPT_VERSION}

Purpose
-------
Check whether the already-sealed 05f2 factor payloads contain enough separately
stored stability and modality-support information to define a no-refit,
post-unblinding exploratory ablation of the SABLE-MO retention rule.

This stage computes NO ablation result and fits NO model.

Verified upstream artifacts
---------------------------
05f3e lock: {relative_path(LOCK_05F3E)}
05f4b lock present: {bool(lock_05f4b is not None)}
05f2 progress manifest: {relative_path(PROGRESS_05F2)}

Scan
----
Datasets: {scan_summary["n_datasets"]}
Factor rows: {scan_summary["n_factor_rows"]}
Distinct factor fields: {scan_summary["n_factor_fields"]}

Interpretation rule
-------------------
05x0 does not infer or invent an ablation formula from field names. The exported
source context must establish the exact frozen retention logic. Only if that logic
can be reconstructed unambiguously from fields already stored in the sealed
payload should a separately versioned 05x1 exploratory contract be written.

If refitting is required, do not use 05x0 to justify it; proceed to the frozen
TARGET branch and treat a refit-based ablation as later exploratory work.
"""
    write_text_atomic(readme, OUTPUT_README)

    generated = [
        OUTPUT_SCHEMA,
        OUTPUT_CANDIDATES,
        OUTPUT_CONTEXT,
        OUTPUT_RESULT,
        OUTPUT_README,
    ]

    lock = {
        "script_version": SCRIPT_VERSION,
        "script_path": relative_path(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "created_at_utc": utc_now_iso(),
        "status": "READ_ONLY_PREFLIGHT_NO_EXPLORATORY_RESULT_COMPUTED",
        "source_05f3e_lock": file_record(LOCK_05F3E),
        "source_05f4b_lock": file_record(LOCK_05F4B) if LOCK_05F4B.is_file() else None,
        "source_05f2_progress_manifest": file_record(PROGRESS_05F2),
        "scientific_result_computed": False,
        "model_refit": False,
        "gpu_computation": False,
        "formal_gate_changed": False,
        "generated_files": {
            relative_path(path): file_record(path)
            for path in generated
        },
        "rewrite_policy": (
            "05x0 is a read-only preflight. Do not add an ablation calculation to "
            "this version after inspecting the schema. Any exploratory calculation "
            "must be defined in a new versioned script/contract."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    print()
    print("05x0 preflight: COMPLETE")
    print(f"Factor schema inventory: {OUTPUT_SCHEMA}")
    print(f"Candidate fields: {OUTPUT_CANDIDATES}")
    print(f"Frozen source context: {OUTPUT_CONTEXT}")
    print(f"Preflight result: {OUTPUT_RESULT}")
    print(f"Preflight lock: {OUTPUT_LOCK}")
    print()
    print(
        "No-refit ablation plausible from schema:",
        "YES" if result["no_refit_ablation_plausible_from_schema"] else "NOT YET ESTABLISHED",
    )
    print()
    print("Highest-priority candidate fields:")
    for row in candidates.head(25).itertuples(index=False):
        print(
            f"  {row.field}: present {row.present_rows}/{row.total_factor_rows}, "
            f"nonnull {row.nonnull_rows}/{row.total_factor_rows}"
        )
    print()
    print("No ablation frequency or new scientific endpoint was computed.")


if __name__ == "__main__":
    main()
