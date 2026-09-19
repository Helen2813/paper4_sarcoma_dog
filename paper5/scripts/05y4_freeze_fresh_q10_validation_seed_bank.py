#!/usr/bin/env python
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "05y4-freeze-fresh-q10-validation-seed-bank-v1-no-cli"
STATUS = "FROZEN_FRESH_Q10_VALIDATION_SEED_BANK_BEFORE_MODEL_EVALUATION"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

DIR_05Y3 = PAPER5_ROOT / "method_contract" / "05y3"
CONTRACT_05Y3 = DIR_05Y3 / "prospective_q10_recalibration_contract.json"
LOCK_05Y3 = DIR_05Y3 / "05y3_contract_lock.json"

SCENARIOS = PAPER5_ROOT / "method_contract" / "05a1" / "final_gate_calibration_scenarios.csv"
PRIOR_SEED_SOURCES = [
    PAPER5_ROOT / "method_contract" / "05a" / "simulation_replicate_registry.csv",
    PAPER5_ROOT / "method_contract" / "05b_v2" / "architecture_screening_data_manifest.csv",
    PAPER5_ROOT / "method_contract" / "05a1" / "final_gate_calibration_replicates.csv",
    PAPER5_ROOT / "method_contract" / "05f2" / "full_run_progress_manifest.csv",
]

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05y4"
OUTPUT_DATASETS = OUTPUT_DIR / "fresh_q10_validation_dataset_registry.csv"
OUTPUT_STABILITY = OUTPUT_DIR / "fresh_q10_validation_stability_seed_registry.csv"
OUTPUT_OVERLAP = OUTPUT_DIR / "fresh_seed_overlap_audit.json"
OUTPUT_MANIFEST = OUTPUT_DIR / "fresh_q10_validation_bank_manifest.json"
OUTPUT_README = OUTPUT_DIR / "05y4_README.txt"
OUTPUT_LOCK = OUTPUT_DIR / "05y4_seed_bank_lock.json"

G19 = "G19_CORRELATED_PRIVATE_NULL_N62"
STANDARD = "G07_IND_N62_S1P00"

MAX_SEED = 2_147_483_647


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
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


def write_csv_atomic(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".csv", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def verify_05y3() -> dict[str, Any]:
    for path in [CONTRACT_05Y3, LOCK_05Y3]:
        require_file(path)
    lock = load_json(LOCK_05Y3)
    expected = "FROZEN_PROSPECTIVE_Q10_RECALIBRATION_BEFORE_FRESH_BANK"
    if str(lock.get("status", "")).strip() != expected:
        raise RuntimeError("05y3 lock status mismatch.")
    if sha256_file(CONTRACT_05Y3).lower() != str(lock["contract"]["sha256"]).lower():
        raise RuntimeError("05y3 contract hash mismatch.")
    contract = load_json(CONTRACT_05Y3)
    if str(contract.get("status", "")).strip() != expected:
        raise RuntimeError("05y3 contract status mismatch.")
    return contract


def collect_seed_column(path: Path) -> tuple[set[int], dict[str, int]]:
    require_file(path)
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        candidates = [
            field for field in fields
            if field.lower() in {"simulation_seed", "dataset_seed", "replicate_seed"}
        ]
        if not candidates:
            return set(), {"rows": 0, "seed_fields": 0, "unique_seeds": 0}
        seeds: set[int] = set()
        rows = 0
        for row in reader:
            rows += 1
            for field in candidates:
                raw = str(row.get(field, "")).strip()
                if raw:
                    seeds.add(int(float(raw)))
        return seeds, {
            "rows": rows,
            "seed_fields": len(candidates),
            "unique_seeds": len(seeds),
        }


def scenario_lookup() -> dict[str, dict[str, str]]:
    require_file(SCENARIOS)
    with SCENARIOS.open("r", encoding="utf-8-sig", newline="") as fh:
        return {
            str(row["scenario_id"]).strip(): row
            for row in csv.DictReader(fh)
        }


def derive_seed(
    *,
    namespace: str,
    master_seed: int,
    scenario_id: str,
    replicate_index: int,
    purpose: str,
    retry: int,
) -> int:
    text = (
        f"{namespace}|{master_seed}|{scenario_id}|{replicate_index}|"
        f"{purpose}|{retry}"
    )
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") % MAX_SEED
    return int(value)


def next_unique_seed(
    *,
    namespace: str,
    master_seed: int,
    scenario_id: str,
    replicate_index: int,
    purpose: str,
    forbidden: set[int],
    used: set[int],
) -> int:
    for retry in range(10000):
        value = derive_seed(
            namespace=namespace,
            master_seed=master_seed,
            scenario_id=scenario_id,
            replicate_index=replicate_index,
            purpose=purpose,
            retry=retry,
        )
        if value == 0:
            continue
        if value in forbidden or value in used:
            continue
        used.add(value)
        return value
    raise RuntimeError(
        f"Could not derive unique seed for {scenario_id}/{replicate_index}/{purpose}."
    )


def main() -> None:
    print("=" * 98)
    print("Paper 5 - freeze fresh q10 validation seed bank")
    print("=" * 98)
    print(f"Script version: {SCRIPT_VERSION}")
    print("Scientific model evaluation: NO")
    print("Fresh simulation arrays materialized: NO")
    print("Fresh seed overlap with prior simulation registries: FORBIDDEN")
    print()

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(
            f"05y4 output directory already exists and is non-empty: {OUTPUT_DIR}"
        )

    contract = verify_05y3()
    bank = contract["fresh_bank_design"]
    master_seed = int(bank["master_seed"])
    fresh_per_scenario = int(bank["scenarios"][0]["fresh_datasets"])
    if fresh_per_scenario != 200:
        raise RuntimeError("05y3 fresh dataset count changed.")
    stability_resamples = int(contract["execution_contract"]["stability_resamples_per_dataset"])
    subsample_fraction = float(contract["execution_contract"]["stability_subsample_fraction"])
    if stability_resamples != 100 or subsample_fraction != 0.80:
        raise RuntimeError("05y3 stability plan changed.")

    prior_simulation_seeds: set[int] = set()
    source_audit: dict[str, Any] = {}
    for source in PRIOR_SEED_SOURCES:
        seeds, audit = collect_seed_column(source)
        prior_simulation_seeds.update(seeds)
        source_audit[relative_path(source)] = audit

    if len(prior_simulation_seeds) < 3100:
        raise RuntimeError(
            f"Prior simulation-seed universe unexpectedly small: {len(prior_simulation_seeds)}"
        )

    scenarios = scenario_lookup()
    for sid in (G19, STANDARD):
        if sid not in scenarios:
            raise RuntimeError(f"Missing scenario {sid}.")

    all_used_seeds: set[int] = set()
    fresh_simulation_seeds: set[int] = set()
    dataset_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []

    scenario_roles = {
        G19: "PRIMARY_CORRELATED_PRIVATE_SPECIFICITY_STRESS",
        STANDARD: "TRUE_SHARED_SENSITIVITY_SAFETY",
    }

    bank_row_index = 0
    for sid in (G19, STANDARD):
        scenario = scenarios[sid]
        n_samples = int(float(scenario["n_samples"]))
        n_subsample = max(2, int(math.floor(n_samples * subsample_fraction)))

        for fresh_rep in range(1, fresh_per_scenario + 1):
            bank_row_index += 1
            sim_seed = next_unique_seed(
                namespace="paper5-05y4-fresh-q10-validation",
                master_seed=master_seed,
                scenario_id=sid,
                replicate_index=fresh_rep,
                purpose="simulation_seed",
                forbidden=prior_simulation_seeds,
                used=all_used_seeds,
            )
            fresh_simulation_seeds.add(sim_seed)

            ref_seed = next_unique_seed(
                namespace="paper5-05y4-fresh-q10-validation",
                master_seed=master_seed,
                scenario_id=sid,
                replicate_index=fresh_rep,
                purpose="reference_fit_seed",
                forbidden=prior_simulation_seeds,
                used=all_used_seeds,
            )

            dataset_key = f"05Y4::{sid}::FRESH_{fresh_rep:04d}::SEED_{sim_seed}"
            dataset_rows.append(
                {
                    "bank_row_index": bank_row_index,
                    "dataset_key": dataset_key,
                    "source_scenario_id": sid,
                    "scenario_role": scenario_roles[sid],
                    "fresh_replicate_index": fresh_rep,
                    "simulation_seed": sim_seed,
                    "n_samples": n_samples,
                    "n_true_biological_shared_factors": int(
                        float(scenario["true_biological_shared_factors"])
                    ),
                    "n_true_technical_shared_factors": int(
                        float(scenario["true_technical_shared_factors"])
                    ),
                    "reference_fit_seed": ref_seed,
                    "stability_resamples": stability_resamples,
                    "stability_subsample_fraction": subsample_fraction,
                    "n_stability_subsample": n_subsample,
                }
            )

            for resample_index in range(1, stability_resamples + 1):
                subsample_seed = next_unique_seed(
                    namespace="paper5-05y4-fresh-q10-validation",
                    master_seed=master_seed,
                    scenario_id=sid,
                    replicate_index=fresh_rep,
                    purpose=f"subsample_seed_{resample_index:03d}",
                    forbidden=prior_simulation_seeds,
                    used=all_used_seeds,
                )
                resample_fit_seed = next_unique_seed(
                    namespace="paper5-05y4-fresh-q10-validation",
                    master_seed=master_seed,
                    scenario_id=sid,
                    replicate_index=fresh_rep,
                    purpose=f"resample_fit_seed_{resample_index:03d}",
                    forbidden=prior_simulation_seeds,
                    used=all_used_seeds,
                )
                stability_rows.append(
                    {
                        "dataset_key": dataset_key,
                        "source_scenario_id": sid,
                        "fresh_replicate_index": fresh_rep,
                        "simulation_seed": sim_seed,
                        "reference_fit_seed": ref_seed,
                        "resample_index": resample_index,
                        "subsample_fraction": subsample_fraction,
                        "n_subsample": n_subsample,
                        "subsample_seed": subsample_seed,
                        "resample_fit_seed": resample_fit_seed,
                        "device": "cuda:0",
                    }
                )

    if len(dataset_rows) != 400:
        raise RuntimeError(f"Fresh dataset registry has {len(dataset_rows)} rows, expected 400.")
    if len(stability_rows) != 400 * 100:
        raise RuntimeError(
            f"Stability seed registry has {len(stability_rows)} rows, expected 40,000."
        )
    if len(fresh_simulation_seeds) != 400:
        raise RuntimeError("Fresh simulation seeds are not unique.")

    overlap = fresh_simulation_seeds & prior_simulation_seeds
    if overlap:
        raise RuntimeError(
            "Fresh simulation seeds overlap prior frozen seeds: "
            + ", ".join(str(x) for x in sorted(overlap)[:20])
        )

    # Dataset-level uniqueness and exact scenario counts.
    counts = {G19: 0, STANDARD: 0}
    keys = set()
    for row in dataset_rows:
        sid = str(row["source_scenario_id"])
        counts[sid] += 1
        key = str(row["dataset_key"])
        if key in keys:
            raise RuntimeError(f"Duplicate dataset_key: {key}")
        keys.add(key)
    if counts != {G19: 200, STANDARD: 200}:
        raise RuntimeError(f"Fresh scenario counts mismatch: {counts}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(dataset_rows, OUTPUT_DATASETS)
    write_csv_atomic(stability_rows, OUTPUT_STABILITY)

    overlap_audit = {
        "fresh_simulation_seed_count": len(fresh_simulation_seeds),
        "prior_unique_simulation_seed_count": len(prior_simulation_seeds),
        "intersection_count": len(overlap),
        "intersection": sorted(overlap),
        "prior_sources": source_audit,
        "all_generated_seed_values_unique_across_simulation_reference_subsample_fit_roles": (
            len(all_used_seeds) == 400 * (1 + 1 + 100 + 100)
        ),
        "total_generated_seed_values": len(all_used_seeds),
    }
    write_json_atomic(overlap_audit, OUTPUT_OVERLAP)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "status": STATUS,
        "scientific_result_computed": False,
        "fresh_arrays_materialized": False,
        "fresh_dataset_registry": file_record(OUTPUT_DATASETS),
        "fresh_stability_registry": file_record(OUTPUT_STABILITY),
        "seed_overlap_audit": file_record(OUTPUT_OVERLAP),
        "05y3_contract": file_record(CONTRACT_05Y3),
        "05y3_lock": file_record(LOCK_05Y3),
        "scope": {
            "fresh_G19_datasets": 200,
            "fresh_STANDARD_datasets": 200,
            "total_fresh_datasets": 400,
            "stability_resamples_per_dataset": 100,
            "stability_rows": 40000,
            "streaming_input_generation": True,
            "permanent_fresh_input_npz_bank": False,
        },
        "guardrails": [
            "No fresh model result was computed while freezing this bank.",
            "Fresh simulation seeds have zero overlap with prior frozen simulation-seed universe.",
            "Seed registry is immutable before 05y5 model evaluation.",
            "No candidate threshold may change after this stage.",
        ],
    }
    write_json_atomic(manifest, OUTPUT_MANIFEST)

    readme = f"""Paper 5 05y4 — fresh q10 validation seed bank
=================================================

Fresh datasets:
  G19-like correlated-private: 200
  STANDARD true-shared:       200
  Total:                      400

Stability:
  100 resamples/dataset
  40,000 stability seed rows

Prior unique simulation seeds audited:
  {len(prior_simulation_seeds)}

Fresh simulation-seed overlap with prior universe:
  0

Permanent fresh NPZ bank:
  NO

05y5 must regenerate each dataset deterministically from this immutable
registry, evaluate one fresh factor/stability result, then apply both
the original q10=.40 rule and candidate q10=.60 rule to the same metrics.
"""
    write_text_atomic(readme, OUTPUT_README)

    lock = {
        "script_version": SCRIPT_VERSION,
        "script_path": relative_path(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "status": STATUS,
        "scientific_result_computed": False,
        "fresh_arrays_materialized": False,
        "dataset_registry": file_record(OUTPUT_DATASETS),
        "stability_registry": file_record(OUTPUT_STABILITY),
        "overlap_audit": file_record(OUTPUT_OVERLAP),
        "manifest": file_record(OUTPUT_MANIFEST),
        "readme": file_record(OUTPUT_README),
        "rewrite_policy": (
            "05y4 seed bank is immutable before 05y5. Any changed seed, dataset count, "
            "scenario membership, resample count, or candidate threshold requires a new prospective branch."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    print("05y4 fresh q10 validation seed bank: FROZEN")
    print(f"Prior unique simulation seeds audited: {len(prior_simulation_seeds):,}")
    print("Fresh simulation seeds: 400")
    print("Fresh overlap with prior simulation seeds: 0")
    print("Fresh dataset rows: 400")
    print("Stability seed rows: 40,000")
    print(f"Total generated unique seed values: {len(all_used_seeds):,}")
    print("Fresh arrays materialized: NO")
    print("Scientific result computed: NO")
    print(f"Datasets:  {relative_path(OUTPUT_DATASETS)}")
    print(f"Stability: {relative_path(OUTPUT_STABILITY)}")
    print(f"Lock:      {relative_path(OUTPUT_LOCK)}")


if __name__ == "__main__":
    main()
