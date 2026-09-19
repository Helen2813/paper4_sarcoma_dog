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


SCRIPT_VERSION = "05y0-preflight-correlated-private-realized-rho-v1-no-cli"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

SCENARIO_REGISTRY = (
    PAPER5_ROOT / "method_contract" / "05a1" / "final_gate_calibration_scenarios.csv"
)
PROGRESS_05F2 = PAPER5_ROOT / "method_contract" / "05f2" / "full_run_progress_manifest.csv"
LOCK_05F3E = PAPER5_ROOT / "method_contract" / "05f3e" / "final_gate_results_lock.json"
LOCK_05X2B = PAPER5_ROOT / "method_contract" / "05x2b" / "05x2b_results_lock.json"
FROZEN_AGGREGATOR = (
    PAPER5_ROOT / "scripts" / "05f3b_aggregate_final_gate_results_cpu_preunblinding_locked.py"
)

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05y0"
OUTPUT_REPORT = OUTPUT_DIR / "realized_rho_preflight.json"
OUTPUT_KEYS = OUTPUT_DIR / "candidate_payload_key_paths.txt"
OUTPUT_ARRAYS = OUTPUT_DIR / "candidate_input_arrays.txt"
OUTPUT_README = OUTPUT_DIR / "05y0_README.txt"
OUTPUT_LOCK = OUTPUT_DIR / "05y0_preflight_lock.json"

G19 = "G19_CORRELATED_PRIVATE_NULL_N62"

KEY_TOKENS = (
    "rho",
    "corr",
    "correlation",
    "private",
    "latent",
    "score",
    "factor",
    "cov",
    "input",
    "path",
    "seed",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
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
    if str(lock.get("status", "")).strip() != expected_status:
        raise RuntimeError(
            f"Unexpected status for {relative_path(path)}: {lock.get('status')!r}"
        )
    return lock


def extract_standard_scenario_id() -> str:
    require_file(FROZEN_AGGREGATOR)
    text = FROZEN_AGGREGATOR.read_text(encoding="utf-8")
    match = re.search(
        r'^\s*STANDARD_SCENARIO_ID\s*=\s*["\']([^"\']+)["\']',
        text,
        flags=re.MULTILINE,
    )
    if not match:
        raise RuntimeError("Could not extract STANDARD_SCENARIO_ID from frozen 05f3b.")
    return match.group(1)


def read_scenario_rows() -> list[dict[str, str]]:
    require_file(SCENARIO_REGISTRY)
    with SCENARIO_REGISTRY.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def read_first_g19_progress_row() -> dict[str, str]:
    require_file(PROGRESS_05F2)
    with PROGRESS_05F2.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if str(row.get("scenario_id", "")).strip() == G19:
                return row
    raise RuntimeError(f"No {G19} row found in 05f2 progress manifest.")


def resolve_project_path(raw: str) -> Path:
    p = Path(str(raw))
    return p if p.is_absolute() else PROJECT_ROOT / p


def flatten_key_paths(
    value: Any,
    prefix: str = "",
    depth: int = 0,
    max_depth: int = 8,
) -> list[tuple[str, str]]:
    if depth > max_depth:
        return []

    rows: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_str = str(key)
            path = f"{prefix}.{key_str}" if prefix else key_str
            rows.append((path, type(child).__name__))
            rows.extend(flatten_key_paths(child, path, depth + 1, max_depth))
    elif isinstance(value, list) and value:
        # Schema-only inspection: inspect at most the first item.
        path = f"{prefix}[0]"
        rows.append((path, type(value[0]).__name__))
        rows.extend(flatten_key_paths(value[0], path, depth + 1, max_depth))
    return rows


def candidate_key_paths(shard: dict[str, Any]) -> list[tuple[str, str]]:
    out = []
    for path, type_name in flatten_key_paths(shard):
        low = path.lower()
        if any(token in low for token in KEY_TOKENS):
            out.append((path, type_name))
    return sorted(set(out))


def find_path_like_strings(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.extend(find_path_like_strings(child, path))
    elif isinstance(value, list):
        for idx, child in enumerate(value[:3]):
            out.extend(find_path_like_strings(child, f"{prefix}[{idx}]"))
    elif isinstance(value, str):
        low = value.lower()
        if any(ext in low for ext in (".npz", ".npy", ".csv", ".parquet")):
            out.append((prefix, value))
    return out


def inspect_npz(path: Path) -> list[dict[str, Any]]:
    try:
        import numpy as np
    except Exception:
        return []

    if not path.is_file() or path.suffix.lower() != ".npz":
        return []

    rows = []
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            arr = data[key]
            low = key.lower()
            if any(token in low for token in KEY_TOKENS):
                rows.append(
                    {
                        "key": key,
                        "shape": list(arr.shape),
                        "dtype": str(arr.dtype),
                    }
                )
    return rows


def inspect_npy_candidate(path: Path) -> dict[str, Any] | None:
    try:
        import numpy as np
    except Exception:
        return None
    if not path.is_file() or path.suffix.lower() != ".npy":
        return None
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "path": relative_path(path),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def main() -> None:
    print("=" * 92)
    print("Paper 5 - preflight realized correlated-private failure-boundary diagnostic")
    print("=" * 92)
    print(f"Script version: {SCRIPT_VERSION}")
    print("Scientific comparison computed: NO")
    print("False-acceptance labels inspected: NO")
    print("Threshold sweep computed: NO")
    print("Model fitting: NO")
    print("GPU computation: NO")
    print()

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(
            f"05y0 output directory already exists and is non-empty: {OUTPUT_DIR}. "
            "Do not overwrite a versioned preflight."
        )

    verify_lock(
        LOCK_05F3E,
        "FROZEN_FINAL_GATE_SCIENTIFIC_RESULTS_NO_RETUNING",
    )
    verify_lock(
        LOCK_05X2B,
        "POSTUNBLINDING_EXPLORATORY_RETENTION_ABLATION_COMPLETE",
    )

    standard_id = extract_standard_scenario_id()
    scenario_rows = read_scenario_rows()
    scenario_by_id = {
        str(row.get("scenario_id", "")).strip(): row
        for row in scenario_rows
    }
    if G19 not in scenario_by_id:
        raise RuntimeError(f"{G19} missing from scenario registry.")
    if standard_id not in scenario_by_id:
        raise RuntimeError(f"{standard_id} missing from scenario registry.")

    progress_row = read_first_g19_progress_row()
    shard_path = resolve_project_path(progress_row["shard_path"])
    require_file(shard_path)

    expected_hash = str(progress_row.get("shard_sha256", "")).strip()
    if sha256_file(shard_path).lower() != expected_hash.lower():
        raise RuntimeError("First G19 shard hash mismatch.")

    shard = load_json(shard_path)
    key_paths = candidate_key_paths(shard)
    path_strings = find_path_like_strings(shard)

    # Inspect any explicitly referenced local NPZ/NPY inputs without reading
    # scientific values; only array names/shapes/dtypes are recorded.
    npz_rows: list[dict[str, Any]] = []
    npy_rows: list[dict[str, Any]] = []

    for key_path, raw in path_strings:
        candidate = resolve_project_path(raw)
        if candidate.suffix.lower() == ".npz" and candidate.is_file():
            for row in inspect_npz(candidate):
                row["source_path"] = relative_path(candidate)
                row["source_key_path"] = key_path
                npz_rows.append(row)
        elif candidate.suffix.lower() == ".npy" and candidate.is_file():
            info = inspect_npy_candidate(candidate)
            if info is not None:
                info["source_key_path"] = key_path
                npy_rows.append(info)

    # Conservative semantic candidates. We do not pick one automatically based
    # on numeric values or association with failure status.
    direct_scalar_candidates = []
    for path, type_name in key_paths:
        low = path.lower()
        if type_name in {"float", "int"} and (
            ("private" in low and ("corr" in low or "rho" in low))
            or "realized_rho" in low
            or "realized_correlation" in low
        ):
            direct_scalar_candidates.append(path)

    latent_array_candidates = [
        row for row in npz_rows
        if "private" in str(row["key"]).lower()
        and any(token in str(row["key"]).lower() for token in ("latent", "score", "factor"))
    ]

    if len(direct_scalar_candidates) == 1:
        availability = "DIRECT_SCALAR_CANDIDATE_FOUND"
    elif latent_array_candidates:
        availability = "PRIVATE_LATENT_ARRAY_CANDIDATES_FOUND"
    elif direct_scalar_candidates:
        availability = "MULTIPLE_DIRECT_SCALAR_CANDIDATES"
    else:
        availability = "NOT_ESTABLISHED_FROM_ONE_SHARD"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    keys_text = "\n".join(
        f"{path}\t{type_name}" for path, type_name in key_paths
    ) + "\n"
    write_text_atomic(keys_text, OUTPUT_KEYS)

    arrays_lines = []
    if path_strings:
        arrays_lines.append("Referenced path-like strings:")
        arrays_lines.extend(f"{k}\t{v}" for k, v in path_strings)
        arrays_lines.append("")
    if npz_rows:
        arrays_lines.append("Candidate NPZ arrays (names/shapes only):")
        arrays_lines.extend(
            f"{row['source_path']}\t{row['key']}\tshape={row['shape']}\tdtype={row['dtype']}"
            for row in npz_rows
        )
        arrays_lines.append("")
    if npy_rows:
        arrays_lines.append("Referenced NPY arrays (paths/shapes only):")
        arrays_lines.extend(
            f"{row['path']}\tshape={row['shape']}\tdtype={row['dtype']}"
            for row in npy_rows
        )
        arrays_lines.append("")
    if not arrays_lines:
        arrays_lines.append("No referenced NPZ/NPY inputs were discoverable from the first G19 shard.")
    write_text_atomic("\n".join(arrays_lines) + "\n", OUTPUT_ARRAYS)

    report = {
        "script_version": SCRIPT_VERSION,
        "status": "READ_ONLY_PREFLIGHT_NO_DIAGNOSTIC_RESULT_COMPUTED",
        "scientific_comparison_computed": False,
        "false_acceptance_labels_inspected": False,
        "threshold_sweep_computed": False,
        "model_fitting": False,
        "GPU_computation": False,
        "scenario_ids": {
            "correlated_private": G19,
            "standard": standard_id,
        },
        "g19_scenario_registry_row": scenario_by_id[G19],
        "standard_scenario_registry_row": scenario_by_id[standard_id],
        "first_g19_shard": {
            "path": relative_path(shard_path),
            "sha256": sha256_file(shard_path),
        },
        "candidate_key_paths_count": len(key_paths),
        "direct_realized_private_correlation_candidates": direct_scalar_candidates,
        "private_latent_array_candidates": latent_array_candidates,
        "realized_rho_availability": availability,
        "next_step_rule": (
            "Freeze 05y1 diagnostic contract only after the primary realized-correlation "
            "quantity has an unambiguous source/formula. Do not choose among candidates "
            "using false-acceptance labels."
        ),
        "source_locks": {
            "05f3e": file_record(LOCK_05F3E),
            "05x2b": file_record(LOCK_05X2B),
            "scenario_registry": file_record(SCENARIO_REGISTRY),
            "frozen_05f3b": file_record(FROZEN_AGGREGATOR),
        },
    }
    write_json_atomic(report, OUTPUT_REPORT)

    readme = f"""Paper 5 05y0 — realized-rho diagnostic preflight
==================================================

This stage computes no failure diagnostic and does not inspect false-acceptance labels.

Correlated-private scenario: {G19}
Frozen STANDARD scenario: {standard_id}

Realized-rho availability: {availability}

Candidate payload keys are written to:
  {relative_path(OUTPUT_KEYS)}

Candidate input-array names/shapes are written to:
  {relative_path(OUTPUT_ARRAYS)}

The next diagnostic contract must freeze the realized-correlation source/formula
before comparing the 70 failing and 130 non-failing G19 datasets.
"""
    write_text_atomic(readme, OUTPUT_README)

    lock = {
        "script_version": SCRIPT_VERSION,
        "script_path": relative_path(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "status": "READ_ONLY_PREFLIGHT_NO_DIAGNOSTIC_RESULT_COMPUTED",
        "generated_files": {
            relative_path(path): file_record(path)
            for path in [OUTPUT_REPORT, OUTPUT_KEYS, OUTPUT_ARRAYS, OUTPUT_README]
        },
        "scientific_comparison_computed": False,
        "rewrite_policy": (
            "05y0 is a read-only preflight. If the realized-rho source is ambiguous, "
            "inspect the exported schema/source information and create a new versioned "
            "preflight or contract; do not add a diagnostic calculation to 05y0."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    print("05y0 preflight: COMPLETE")
    print(f"Frozen STANDARD scenario: {standard_id}")
    print(f"Realized-rho availability: {availability}")
    print(f"Direct scalar candidates: {len(direct_scalar_candidates)}")
    print(f"Private latent array candidates: {len(latent_array_candidates)}")
    print(f"Key-path report: {relative_path(OUTPUT_KEYS)}")
    print(f"Array report:    {relative_path(OUTPUT_ARRAYS)}")
    print(f"Preflight JSON:  {relative_path(OUTPUT_REPORT)}")
    print("Scientific diagnostic result computed: NO")


if __name__ == "__main__":
    main()
