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

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05y6-aggregate-prospective-q10-validation-v1-no-cli"
STATUS = "PROSPECTIVE_Q10_RECALIBRATION_RESULT_FROZEN"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

DIR_05Y3 = PAPER5_ROOT / "method_contract" / "05y3"
CONTRACT_05Y3 = DIR_05Y3 / "prospective_q10_recalibration_contract.json"
LOCK_05Y3 = DIR_05Y3 / "05y3_contract_lock.json"

DIR_05Y4 = PAPER5_ROOT / "method_contract" / "05y4"
LOCK_05Y4 = DIR_05Y4 / "05y4_seed_bank_lock.json"

DIR_05Y5 = PAPER5_ROOT / "method_contract" / "05y5"
COMPLETION_05Y5 = DIR_05Y5 / "05y5_execution_completion_lock.json"
PROGRESS_05Y5 = DIR_05Y5 / "fresh_q10_validation_progress_manifest.csv"

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05y6"
OUT_G19 = OUTPUT_DIR / "fresh_g19_paired_specificity.csv"
OUT_STANDARD = OUTPUT_DIR / "fresh_standard_recovery_by_dataset.csv"
OUT_SUMMARY = OUTPUT_DIR / "prospective_q10_validation_summary.csv"
OUT_RESULT = OUTPUT_DIR / "prospective_q10_validation_result.json"
OUT_README = OUTPUT_DIR / "05y6_README.txt"
OUT_LOCK = OUTPUT_DIR / "05y6_results_lock.json"

EXPECTED_COMPLETION_STATUS = "SEALED_FRESH_Q10_VALIDATION_EXECUTION_COMPLETE_NO_AGGREGATION"
EXPECTED_G19_DATASETS = 200
EXPECTED_STANDARD_DATASETS = 200
EXPECTED_STANDARD_OPPORTUNITIES = 800
ORIGINAL_Q10 = 0.40
CANDIDATE_Q10 = 0.60
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260916
ALPHA = 0.05


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def resolve_project_path(value: Any) -> Path:
    path = Path(str(value).strip())
    return path if path.is_absolute() else PROJECT_ROOT / path


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


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value in (1, 1.0):
        return True
    if value in (0, 0.0):
        return False
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n"}:
        return False
    raise RuntimeError(f"Cannot parse boolean value: {value!r}")


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ) + "\n",
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


def wilson(k: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (
        z
        * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
        / denom
    )
    return max(0.0, center - half), min(1.0, center + half)


def exact_two_sided_nested_mcnemar(rescued: int, reverse: int) -> float:
    discordant = int(rescued + reverse)
    if discordant == 0:
        return 1.0
    if reverse != 0:
        raise RuntimeError(
            "Candidate q10=.60 produced reverse discordance despite being a nested "
            "stricter rule. Scientific implementation integrity is violated."
        )
    # Under H0, rescued and reverse discordances are equiprobable.
    return min(1.0, 2.0 * (0.5 ** discordant))


def bootstrap_paired_reduction(
    original: np.ndarray,
    candidate: np.ndarray,
) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = original.size
    differences = original.astype(float) - candidate.astype(float)
    values = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for i in range(BOOTSTRAP_REPLICATES):
        idx = rng.integers(0, n, size=n)
        values[i] = float(np.mean(differences[idx]))
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    for path in [
        CONTRACT_05Y3,
        LOCK_05Y3,
        LOCK_05Y4,
        COMPLETION_05Y5,
        PROGRESS_05Y5,
    ]:
        require_file(path)

    lock3 = load_json(LOCK_05Y3)
    if str(lock3.get("status", "")).strip() != (
        "FROZEN_PROSPECTIVE_Q10_RECALIBRATION_BEFORE_FRESH_BANK"
    ):
        raise RuntimeError("05y3 lock status mismatch.")
    if sha256_file(CONTRACT_05Y3).lower() != str(lock3["contract"]["sha256"]).lower():
        raise RuntimeError("05y3 contract hash mismatch.")

    contract = load_json(CONTRACT_05Y3)
    if float(contract["single_candidate_rule"]["q10_aligned_correlation_min"]) != CANDIDATE_Q10:
        raise RuntimeError("05y3 candidate q10 is not 0.60.")
    if float(contract["original_rule"]["q10_aligned_correlation_min"]) != ORIGINAL_Q10:
        raise RuntimeError("05y3 original q10 is not 0.40.")

    completion = load_json(COMPLETION_05Y5)
    if str(completion.get("status", "")).strip() != EXPECTED_COMPLETION_STATUS:
        raise RuntimeError("05y5 completion lock status mismatch.")
    if int(completion.get("fresh_datasets_completed", -1)) != 400:
        raise RuntimeError("05y5 completion lock does not represent 400 datasets.")
    if bool(completion.get("scientific_endpoint_aggregated", True)):
        raise RuntimeError("05y5 unexpectedly claims scientific endpoint aggregation.")
    if bool(completion.get("threshold_retuned", True)):
        raise RuntimeError("05y5 indicates threshold retuning.")

    progress = pd.read_csv(PROGRESS_05Y5, dtype=str, low_memory=False).fillna("")
    if progress.shape[0] != 400:
        raise RuntimeError("05y5 progress manifest is not 400 rows.")

    if sha256_file(PROGRESS_05Y5).lower() != str(
        completion["progress_manifest"]["sha256"]
    ).lower():
        raise RuntimeError("05y5 progress manifest hash mismatch.")

    return contract, completion, progress


def candidate_keep(factor: dict[str, Any]) -> bool:
    original = parse_bool(factor["passes_stability_gate"])
    q10 = float(factor["q10_aligned_correlation"])
    return bool(original and q10 >= CANDIDATE_Q10)


def validate_original_rule_fields(factor: dict[str, Any], contract: dict[str, Any]) -> None:
    rule = contract["original_rule"]
    reconstructed = (
        float(factor["median_aligned_correlation"])
        >= float(rule["median_aligned_correlation_min"])
        and float(factor["q10_aligned_correlation"])
        >= float(rule["q10_aligned_correlation_min"])
        and float(factor["inclusion_frequency"])
        >= float(rule["inclusion_frequency_min"])
        and float(factor["reference_factor_variance"])
        >= float(rule["reference_factor_variance_min"])
        and int(factor["n_supported_modalities"])
        >= int(rule["n_supported_modalities_min"])
    )
    stored = parse_bool(factor["passes_stability_gate"])
    if reconstructed != stored:
        raise RuntimeError(
            "Fresh factor stored passes_stability_gate disagrees with frozen original "
            "median/q10/inclusion/variance/modality-support rule."
        )


def main() -> None:
    print("=" * 100)
    print("Paper 5 - aggregate prospective q10=.60 fresh validation")
    print("=" * 100)
    print(f"Script version: {SCRIPT_VERSION}")
    print("This is the first stage allowed to aggregate fresh prospective endpoints.")
    print("Candidate retuning after this point: FORBIDDEN")
    print()

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(
            f"05y6 output directory already exists and is non-empty: {OUTPUT_DIR}. "
            "Do not overwrite prospective results."
        )

    contract, completion, progress = verify_inputs()

    g19_rows: list[dict[str, Any]] = []
    standard_rows: list[dict[str, Any]] = []

    g19_original = []
    g19_candidate = []

    standard_original_total = 0
    standard_candidate_total = 0
    standard_lost_total = 0
    standard_gained_total = 0

    shard_hashes_seen = set()

    progress = progress.copy()
    progress["bank_row_index_num"] = pd.to_numeric(
        progress["bank_row_index"], errors="raise"
    ).astype(int)
    progress = progress.sort_values(
        "bank_row_index_num", kind="mergesort"
    ).reset_index(drop=True)

    for _, row in progress.iterrows():
        shard_path = resolve_project_path(row["shard_path"])
        require_file(shard_path)
        actual_hash = sha256_file(shard_path)
        expected_hash = str(row["shard_sha256"]).strip()
        if actual_hash.lower() != expected_hash.lower():
            raise RuntimeError(f"05y5 shard hash mismatch: {shard_path}")
        if actual_hash in shard_hashes_seen:
            raise RuntimeError("Duplicate 05y5 shard hash encountered.")
        shard_hashes_seen.add(actual_hash)

        shard = load_json(shard_path)
        payload = shard.get("scientific_payload")
        if not isinstance(payload, dict):
            raise RuntimeError("05y5 shard lacks scientific_payload.")
        factors = payload.get("factor_rows")
        if not isinstance(factors, list):
            raise RuntimeError("05y5 shard lacks factor_rows.")

        for factor in factors:
            validate_original_rule_fields(factor, contract)

        sid = str(row["source_scenario_id"])
        dataset_key = str(row["dataset_key"])

        if sid == "G19_CORRELATED_PRIVATE_NULL_N62":
            original_event = False
            candidate_event = False

            for factor in factors:
                false_truth = (
                    not parse_bool(factor["biological_truth_match"])
                    and not parse_bool(factor["technical_truth_match"])
                )
                original_keep = parse_bool(factor["passes_stability_gate"])
                cand_keep = candidate_keep(factor)

                if original_keep and false_truth:
                    original_event = True
                if cand_keep and false_truth:
                    candidate_event = True

            if candidate_event and not original_event:
                raise RuntimeError(
                    f"Nested candidate event without original event in {dataset_key}."
                )

            g19_original.append(int(original_event))
            g19_candidate.append(int(candidate_event))
            g19_rows.append(
                {
                    "dataset_key": dataset_key,
                    "fresh_replicate_index": int(row["fresh_replicate_index"]),
                    "simulation_seed": int(row["simulation_seed"]),
                    "original_false_acceptance_event": int(original_event),
                    "candidate_false_acceptance_event": int(candidate_event),
                    "rescued_by_q10_060": int(original_event and not candidate_event),
                    "reverse_discordance": int(candidate_event and not original_event),
                }
            )

        elif sid == "G07_IND_N62_S1P00":
            original_recovered = 0
            candidate_recovered = 0
            lost = 0
            gained = 0

            for factor in factors:
                biological = parse_bool(factor["biological_truth_match"])
                if not biological:
                    continue
                original_keep = parse_bool(factor["passes_stability_gate"])
                cand_keep = candidate_keep(factor)

                original_recovered += int(original_keep)
                candidate_recovered += int(cand_keep)
                lost += int(original_keep and not cand_keep)
                gained += int(cand_keep and not original_keep)

            if gained != 0:
                raise RuntimeError(
                    f"Nested q10=.60 rule gained STANDARD truth factors in {dataset_key}."
                )

            standard_original_total += original_recovered
            standard_candidate_total += candidate_recovered
            standard_lost_total += lost
            standard_gained_total += gained

            standard_rows.append(
                {
                    "dataset_key": dataset_key,
                    "fresh_replicate_index": int(row["fresh_replicate_index"]),
                    "simulation_seed": int(row["simulation_seed"]),
                    "truth_factor_opportunities": 4,
                    "original_recovered_truth_factors": original_recovered,
                    "candidate_recovered_truth_factors": candidate_recovered,
                    "lost_truth_factors_under_q10_060": lost,
                    "gained_truth_factors_under_q10_060": gained,
                }
            )
        else:
            raise RuntimeError(f"Unexpected 05y5 scenario in progress: {sid}")

    if len(g19_rows) != EXPECTED_G19_DATASETS:
        raise RuntimeError(
            f"Expected {EXPECTED_G19_DATASETS} fresh G19 datasets; found {len(g19_rows)}."
        )
    if len(standard_rows) != EXPECTED_STANDARD_DATASETS:
        raise RuntimeError(
            f"Expected {EXPECTED_STANDARD_DATASETS} fresh STANDARD datasets; "
            f"found {len(standard_rows)}."
        )

    original = np.asarray(g19_original, dtype=int)
    candidate = np.asarray(g19_candidate, dtype=int)

    original_n = int(original.sum())
    candidate_n = int(candidate.sum())
    rescued = int(np.sum((original == 1) & (candidate == 0)))
    reverse = int(np.sum((original == 0) & (candidate == 1)))

    if original_n - candidate_n != rescued - reverse:
        raise RuntimeError("Paired G19 accounting identity failed.")
    if reverse != 0:
        raise RuntimeError("Candidate nesting violated: reverse discordance is nonzero.")

    original_rate = original_n / EXPECTED_G19_DATASETS
    candidate_rate = candidate_n / EXPECTED_G19_DATASETS
    absolute_reduction = original_rate - candidate_rate
    relative_reduction = (
        absolute_reduction / original_rate if original_rate > 0 else None
    )

    orig_ci = wilson(original_n, EXPECTED_G19_DATASETS)
    cand_ci = wilson(candidate_n, EXPECTED_G19_DATASETS)
    reduction_ci = bootstrap_paired_reduction(original, candidate)
    p_value = exact_two_sided_nested_mcnemar(rescued, reverse)

    if standard_original_total > EXPECTED_STANDARD_OPPORTUNITIES:
        raise RuntimeError("Original STANDARD recovered count exceeds 800.")
    if standard_candidate_total > EXPECTED_STANDARD_OPPORTUNITIES:
        raise RuntimeError("Candidate STANDARD recovered count exceeds 800.")
    if standard_original_total - standard_candidate_total != (
        standard_lost_total - standard_gained_total
    ):
        raise RuntimeError("STANDARD paired accounting identity failed.")
    if standard_gained_total != 0:
        raise RuntimeError("Candidate nesting violated for STANDARD truth recovery.")

    standard_original_rate = standard_original_total / EXPECTED_STANDARD_OPPORTUNITIES
    standard_candidate_rate = standard_candidate_total / EXPECTED_STANDARD_OPPORTUNITIES
    standard_candidate_ci = wilson(
        standard_candidate_total,
        EXPECTED_STANDARD_OPPORTUNITIES,
    )

    confirmation = bool(
        p_value < ALPHA
        and candidate_rate < original_rate
        and reverse == 0
        and standard_candidate_rate >= 0.80
    )

    summary_rows = [
        {
            "endpoint": "G19_original_false_acceptance",
            "numerator": original_n,
            "denominator": EXPECTED_G19_DATASETS,
            "rate": original_rate,
            "ci95_low": orig_ci[0],
            "ci95_high": orig_ci[1],
        },
        {
            "endpoint": "G19_candidate_false_acceptance_q10_060",
            "numerator": candidate_n,
            "denominator": EXPECTED_G19_DATASETS,
            "rate": candidate_rate,
            "ci95_low": cand_ci[0],
            "ci95_high": cand_ci[1],
        },
        {
            "endpoint": "STANDARD_original_biological_recovery",
            "numerator": standard_original_total,
            "denominator": EXPECTED_STANDARD_OPPORTUNITIES,
            "rate": standard_original_rate,
            "ci95_low": "",
            "ci95_high": "",
        },
        {
            "endpoint": "STANDARD_candidate_biological_recovery_q10_060",
            "numerator": standard_candidate_total,
            "denominator": EXPECTED_STANDARD_OPPORTUNITIES,
            "rate": standard_candidate_rate,
            "ci95_low": standard_candidate_ci[0],
            "ci95_high": standard_candidate_ci[1],
        },
    ]

    result = {
        "script_version": SCRIPT_VERSION,
        "status": STATUS,
        "analysis_role": "PROSPECTIVE_POST_DISCOVERY_CONFIRMATORY_RECALIBRATION_VALIDATION",
        "original_rule": contract["original_rule"],
        "candidate_rule": contract["single_candidate_rule"],
        "fresh_bank": {
            "G19_datasets": EXPECTED_G19_DATASETS,
            "STANDARD_datasets": EXPECTED_STANDARD_DATASETS,
            "STANDARD_truth_factor_opportunities": EXPECTED_STANDARD_OPPORTUNITIES,
            "seed_overlap_with_prior_bank": 0,
        },
        "primary_specificity_endpoint": {
            "original_false_acceptance_n": original_n,
            "original_false_acceptance_denominator": EXPECTED_G19_DATASETS,
            "original_false_acceptance_rate": original_rate,
            "original_wilson95": list(orig_ci),
            "candidate_false_acceptance_n": candidate_n,
            "candidate_false_acceptance_denominator": EXPECTED_G19_DATASETS,
            "candidate_false_acceptance_rate": candidate_rate,
            "candidate_wilson95": list(cand_ci),
            "rescued_datasets": rescued,
            "reverse_discordance": reverse,
            "absolute_rate_reduction": absolute_reduction,
            "relative_rate_reduction": relative_reduction,
            "paired_bootstrap95_absolute_reduction": list(reduction_ci),
            "exact_two_sided_nested_mcnemar_p": p_value,
            "alpha": ALPHA,
            "required_direction_met": bool(candidate_rate < original_rate),
        },
        "sensitivity_safety_endpoint": {
            "original_recovered_n": standard_original_total,
            "denominator": EXPECTED_STANDARD_OPPORTUNITIES,
            "original_recovery_rate": standard_original_rate,
            "candidate_recovered_n": standard_candidate_total,
            "candidate_recovery_rate": standard_candidate_rate,
            "candidate_wilson95": list(standard_candidate_ci),
            "lost_truth_opportunities": standard_lost_total,
            "gained_truth_opportunities": standard_gained_total,
            "existing_D02_floor": 0.80,
            "floor_met": bool(standard_candidate_rate >= 0.80),
        },
        "prospective_confirmation": {
            "candidate_confirmed": confirmation,
            "definition": (
                "p<0.05 exact paired specificity improvement in required direction, "
                "reverse discordance=0, and candidate STANDARD recovery >=0.80."
            ),
            "retuning_after_result": "FORBIDDEN",
            "formal_05f3e_gate_consequence": "NONE",
        },
        "provenance": {
            "05y3_contract": file_record(CONTRACT_05Y3),
            "05y3_lock": file_record(LOCK_05Y3),
            "05y4_lock": file_record(LOCK_05Y4),
            "05y5_completion": file_record(COMPLETION_05Y5),
            "05y5_progress": file_record(PROGRESS_05Y5),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(g19_rows, OUT_G19)
    write_csv_atomic(standard_rows, OUT_STANDARD)
    write_csv_atomic(summary_rows, OUT_SUMMARY)
    write_json_atomic(result, OUT_RESULT)

    readme = f"""Paper 5 05y6 — prospective q10=.60 validation
==================================================

Fresh G19 specificity
---------------------
Original q10=.40:
  {original_n}/{EXPECTED_G19_DATASETS} = {original_rate:.6f}

Candidate q10=.60:
  {candidate_n}/{EXPECTED_G19_DATASETS} = {candidate_rate:.6f}

Rescued fresh datasets:
  {rescued}

Reverse discordance:
  {reverse}

Absolute paired reduction:
  {absolute_reduction:.6f}

95% paired-bootstrap CI:
  [{reduction_ci[0]:.6f}, {reduction_ci[1]:.6f}]

Exact two-sided paired p:
  {p_value:.12g}

Fresh STANDARD sensitivity safety
---------------------------------
Original:
  {standard_original_total}/{EXPECTED_STANDARD_OPPORTUNITIES}
  = {standard_original_rate:.6f}

Candidate:
  {standard_candidate_total}/{EXPECTED_STANDARD_OPPORTUNITIES}
  = {standard_candidate_rate:.6f}

Existing D02 floor:
  0.80

Floor met:
  {standard_candidate_rate >= 0.80}

Prospective candidate confirmed:
  {confirmation}

The original 05f3e gate remains authoritative and unchanged.
No further threshold selection is permitted on this fresh bank.
"""
    write_text_atomic(readme, OUT_README)

    generated = [
        OUT_G19,
        OUT_STANDARD,
        OUT_SUMMARY,
        OUT_RESULT,
        OUT_README,
    ]
    lock = {
        "script_version": SCRIPT_VERSION,
        "script_path": relative_path(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "status": STATUS,
        "candidate_confirmed": confirmation,
        "formal_05f3e_gate_consequence": "NONE",
        "threshold_retuning_after_fresh_result": False,
        "generated_files": {
            relative_path(path): file_record(path)
            for path in generated
        },
        "rewrite_policy": (
            "05y6 prospective result is immutable. No alternative q10, median, inclusion, "
            "or other frontier point may be selected using this fresh validation bank."
        ),
    }
    write_json_atomic(lock, OUT_LOCK)

    print("05y6 prospective aggregation: COMPLETE")
    print()
    print(
        f"G19 original false acceptance:  {original_n}/{EXPECTED_G19_DATASETS} "
        f"= {original_rate:.3%}"
    )
    print(
        f"G19 q10=.60 false acceptance:   {candidate_n}/{EXPECTED_G19_DATASETS} "
        f"= {candidate_rate:.3%}"
    )
    print(
        f"Paired absolute reduction:      {absolute_reduction:.3%} "
        f"[95% bootstrap {reduction_ci[0]:.3%}, {reduction_ci[1]:.3%}]"
    )
    print(f"Rescued / reverse discordance:  {rescued} / {reverse}")
    print(f"Exact paired p-value:           {p_value:.6g}")
    print()
    print(
        f"STANDARD original recovery:     {standard_original_total}/"
        f"{EXPECTED_STANDARD_OPPORTUNITIES} = {standard_original_rate:.3%}"
    )
    print(
        f"STANDARD q10=.60 recovery:      {standard_candidate_total}/"
        f"{EXPECTED_STANDARD_OPPORTUNITIES} = {standard_candidate_rate:.3%}"
    )
    print(f"Existing D02 >=0.80 floor met:  {standard_candidate_rate >= 0.80}")
    print()
    print(f"PROSPECTIVE q10=.60 CONFIRMED:  {confirmation}")
    print("Formal 05f3e gate changed:       NO")
    print(f"Result: {relative_path(OUT_RESULT)}")
    print(f"Lock:   {relative_path(OUT_LOCK)}")


if __name__ == "__main__":
    main()
