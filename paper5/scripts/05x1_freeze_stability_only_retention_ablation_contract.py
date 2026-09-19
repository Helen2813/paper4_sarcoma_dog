#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_VERSION = "05x1-freeze-stability-only-retention-ablation-contract-v1-no-cli"
CONTRACT_VERSION = "paper5-05x1-postunblinding-exploratory-retention-ablation-v1"
STATUS = "FROZEN_POSTUNBLINDING_EXPLORATORY_ABLATION_BEFORE_RESULT_COMPUTATION"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

CONTRACT_05A = PAPER5_ROOT / "method_contract" / "05a" / "ai_method_contract.json"
HELPER = PAPER5_ROOT / "scripts" / "05d2v2_screen_neural_profiles_gpu_syncfree_pathfix.py"

PROGRESS_05F2 = PAPER5_ROOT / "method_contract" / "05f2" / "full_run_progress_manifest.csv"
LOCK_05F2 = PAPER5_ROOT / "method_contract" / "05f2" / "full_run_completion_lock.json"

LOCK_05F3E = PAPER5_ROOT / "method_contract" / "05f3e" / "final_gate_results_lock.json"
LOCK_05F4B = PAPER5_ROOT / "method_contract" / "05f4b" / "05f4b_manuscript_assets_lock.json"
LOCK_05X0 = PAPER5_ROOT / "method_contract" / "05x0" / "05x0_preflight_lock.json"

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05x1"
OUTPUT_CONTRACT = OUTPUT_DIR / "stability_only_retention_ablation_contract.json"
OUTPUT_README = OUTPUT_DIR / "05x1_README.txt"
OUTPUT_LOCK = OUTPUT_DIR / "05x1_contract_lock.json"

CORRELATED_PRIVATE_SCENARIO_ID = "G19_CORRELATED_PRIVATE_NULL_N62"
EXPECTED_DATASETS_PER_SUBBANK = 200

REQUIRED_THRESHOLDS = {
    "primary_median_factor_correlation": 0.70,
    "primary_q10_factor_correlation": 0.40,
    "primary_inclusion_frequency": 0.80,
    "minimum_factor_variance": 0.0001,
    "minimum_supported_modalities": 2,
    "minimum_loading_energy_fraction_per_supported_modality": 0.15,
}


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


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")


def verify_lock_file(path: Path, expected_status: str | None = None) -> dict[str, Any]:
    require_file(path)
    lock = load_json(path)
    if expected_status is not None and str(lock.get("status", "")).strip() != expected_status:
        raise RuntimeError(
            f"Unexpected lock status for {relative_path(path)}: {lock.get('status')}"
        )
    return lock


def main() -> None:
    print("=" * 92)
    print("Paper 5 - freeze post-unblinding exploratory retention-rule ablation contract")
    print("=" * 92)
    print(f"Script version: {SCRIPT_VERSION}")
    print("Scientific ablation result computed: NO")
    print("Model fitting: NO")
    print("GPU computation: NO")
    print("Formal final-gate modification: FORBIDDEN")
    print()

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(
            f"05x1 output directory already exists and is non-empty: {OUTPUT_DIR}. "
            "Do not overwrite a frozen exploratory contract."
        )

    for path in [
        CONTRACT_05A,
        HELPER,
        PROGRESS_05F2,
        LOCK_05F2,
        LOCK_05F3E,
        LOCK_05F4B,
        LOCK_05X0,
    ]:
        require_file(path)

    lock_05f2 = verify_lock_file(
        LOCK_05F2,
        "FROZEN_ALL_3100_FINAL_GATE_SHARDS_COMPLETE_BEFORE_SCIENTIFIC_AGGREGATION",
    )
    lock_05f3e = verify_lock_file(
        LOCK_05F3E,
        "FROZEN_FINAL_GATE_SCIENTIFIC_RESULTS_NO_RETUNING",
    )
    lock_05f4b = verify_lock_file(
        LOCK_05F4B,
        "READ_ONLY_MANUSCRIPT_ASSET_EXPORT_FROM_LOCKED_05F3E",
    )
    lock_05x0 = verify_lock_file(
        LOCK_05X0,
        "READ_ONLY_PREFLIGHT_NO_EXPLORATORY_RESULT_COMPUTED",
    )

    if str(lock_05f3e.get("core_status", "")).strip() != "CORE_PASS":
        raise RuntimeError("05x1 requires the already-frozen 05f3e CORE_PASS state.")

    preflight = load_json(
        PAPER5_ROOT / "method_contract" / "05x0" / "retention_ablation_preflight.json"
    )
    if not bool(preflight.get("no_refit_ablation_plausible_from_schema", False)):
        raise RuntimeError("05x0 did not establish a no-refit ablation as schema-plausible.")
    if bool(preflight.get("scientific_result_computed", True)):
        raise RuntimeError("05x0 unexpectedly reports a scientific result.")

    method = load_json(CONTRACT_05A)
    thresholds = method.get("stability_gate", {}).get("thresholds", {})
    for key, expected in REQUIRED_THRESHOLDS.items():
        if key not in thresholds:
            raise RuntimeError(f"Frozen 05a threshold missing: {key}")
        actual = thresholds[key]
        if isinstance(expected, int):
            if int(actual) != expected:
                raise RuntimeError(f"Frozen 05a threshold changed for {key}: {actual}")
        else:
            if float(actual) != float(expected):
                raise RuntimeError(f"Frozen 05a threshold changed for {key}: {actual}")

    helper_text = HELPER.read_text(encoding="utf-8", errors="strict")
    required_fragments = [
        '"primary_median_factor_correlation"',
        '"primary_q10_factor_correlation"',
        '"primary_inclusion_frequency"',
        '"minimum_factor_variance"',
        '"minimum_supported_modalities"',
        '"minimum_loading_energy_fraction_per_supported_modality"',
        '"median_aligned_correlation"',
        '"q10_aligned_correlation"',
        '"inclusion_frequency"',
        '"reference_factor_variance"',
        '"n_supported_modalities"',
        '"passes_stability_gate"',
    ]
    missing_fragments = [x for x in required_fragments if x not in helper_text]
    if missing_fragments:
        raise RuntimeError(
            "Frozen helper no longer exposes expected retention components: "
            + ", ".join(missing_fragments)
        )

    progress = pd.read_csv(PROGRESS_05F2, dtype=str, low_memory=False).fillna("")
    if progress.shape[0] != 3100:
        raise RuntimeError(f"05f2 progress manifest has {progress.shape[0]} rows, expected 3100.")
    if "scenario_id" not in progress.columns:
        raise RuntimeError("05f2 progress manifest lacks scenario_id.")

    scenario_counts = progress["scenario_id"].value_counts().to_dict()

    if CORRELATED_PRIVATE_SCENARIO_ID not in scenario_counts:
        raise RuntimeError(
            f"Correlated-private scenario missing: {CORRELATED_PRIVATE_SCENARIO_ID}"
        )
    if int(scenario_counts[CORRELATED_PRIVATE_SCENARIO_ID]) != EXPECTED_DATASETS_PER_SUBBANK:
        raise RuntimeError(
            f"{CORRELATED_PRIVATE_SCENARIO_ID} has "
            f"{scenario_counts[CORRELATED_PRIVATE_SCENARIO_ID]} datasets, expected 200."
        )

    independent_candidates = sorted(
        sid
        for sid in scenario_counts
        if "INDEPENDENT" in sid.upper() and "N62" in sid.upper()
    )
    if len(independent_candidates) != 1:
        raise RuntimeError(
            "Expected exactly one independent-null N62 scenario; found: "
            + ", ".join(independent_candidates)
        )
    independent_n62_id = independent_candidates[0]
    if int(scenario_counts[independent_n62_id]) != EXPECTED_DATASETS_PER_SUBBANK:
        raise RuntimeError(
            f"{independent_n62_id} has {scenario_counts[independent_n62_id]} datasets, expected 200."
        )

    contract = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": STATUS,
        "analysis_role": "POSTUNBLINDING_EXPLORATORY_ABLATION",
        "motivation": (
            "Quantify how much of SABLE-MO retention behavior on the prespecified "
            "correlated-private stress mechanism is attributable to modality-support "
            "and variance constraints beyond factor-wise reproducibility."
        ),
        "scientific_boundary": {
            "formal_05f3e_gate_reopened": False,
            "formal_D01_recomputed_or_reweighted": False,
            "threshold_retuning": False,
            "model_refitting": False,
            "GPU_computation_required": False,
            "TARGET_or_outcomes_accessed": False,
            "primary_manuscript_claim_status": "EXPLORATORY_CONTEXT_ONLY",
        },
        "frozen_subbanks": [
            {
                "subbank_id": "CORRELATED_PRIVATE_N62",
                "scenario_id": CORRELATED_PRIVATE_SCENARIO_ID,
                "expected_datasets": EXPECTED_DATASETS_PER_SUBBANK,
                "role": "PRIMARY_EXPLORATORY_STRESS_MECHANISM",
            },
            {
                "subbank_id": "INDEPENDENT_NULL_N62",
                "scenario_id": independent_n62_id,
                "expected_datasets": EXPECTED_DATASETS_PER_SUBBANK,
                "role": "MATCHED_N_MECHANISM_CONTROL",
            },
        ],
        "frozen_thresholds_from_05a": {
            key: thresholds[key] for key in REQUIRED_THRESHOLDS
        },
        "frozen_rules": [
            {
                "rule_id": "FULL_FROZEN_RETENTION",
                "role": "REFERENCE_ONLY",
                "formula": (
                    "median>=0.70 AND q10>=0.40 AND inclusion>=0.80 "
                    "AND variance>=0.0001 AND n_supported_modalities>=2"
                ),
                "components": [
                    "median_aligned_correlation",
                    "q10_aligned_correlation",
                    "inclusion_frequency",
                    "reference_factor_variance",
                    "n_supported_modalities",
                ],
            },
            {
                "rule_id": "NO_MODALITY_SUPPORT",
                "role": "PRIMARY_EXPLORATORY_ABLATION",
                "formula": (
                    "median>=0.70 AND q10>=0.40 AND inclusion>=0.80 "
                    "AND variance>=0.0001"
                ),
                "removed_from_full_rule": [
                    "n_supported_modalities>=2",
                ],
                "interpretation": (
                    "Isolates the incremental filtering contribution of the "
                    "modality-support requirement while preserving the frozen variance floor."
                ),
            },
            {
                "rule_id": "PURE_FACTOR_REPRODUCIBILITY",
                "role": "SECONDARY_EXPLORATORY_ABLATION",
                "formula": (
                    "median>=0.70 AND q10>=0.40 AND inclusion>=0.80"
                ),
                "removed_from_full_rule": [
                    "variance>=0.0001",
                    "n_supported_modalities>=2",
                ],
                "interpretation": (
                    "Operationalizes factor-wise reproducibility alone, matching the "
                    "conceptual question whether reproducibility by itself is sufficient "
                    "for biological retention."
                ),
            },
        ],
        "false_biological_acceptance_definition": (
            "A factor is a false biological acceptance under a candidate retention rule "
            "iff it is retained by that rule AND technical_truth_match is false AND "
            "biological_truth_match is false. A dataset-level event occurs iff at least "
            "one such factor exists in the dataset."
        ),
        "reporting_plan": {
            "for_each_subbank_and_rule": [
                "dataset_level_false_biological_acceptance_numerator",
                "dataset_level_false_biological_acceptance_denominator",
                "dataset_level_false_biological_acceptance_rate",
                "two_sided_95pct_Wilson_interval",
                "total_retained_factors",
                "total_false_biological_factors",
            ],
            "paired_nested_rule_transitions": [
                "datasets_false_under_ablation_but_not_full",
                "datasets_false_under_full_and_ablation",
                "datasets_false_under_neither",
            ],
            "p_values": "NONE",
            "new_PASS_FAIL_thresholds": "NONE",
        },
        "implementation_checks_required_before_reporting": [
            "Recompute n_supported_modalities from the three stored loading-energy fractions at the frozen 0.15 threshold and reproduce the stored n_supported_modalities.",
            "Recompute FULL_FROZEN_RETENTION from stored factor fields and reproduce passes_stability_gate exactly for every analyzed factor.",
            "Recompute the stored full-rule false_biological_acceptance flag exactly for every analyzed factor.",
            "Reproduce the already-frozen full-rule dataset-level rates on the two selected subbanks before exposing ablation results.",
        ],
        "expected_full_rule_reference_rates_for_integrity_check": {
            "CORRELATED_PRIVATE_N62": {
                "numerator": 70,
                "denominator": 200,
            },
            "INDEPENDENT_NULL_N62": {
                "numerator": 0,
                "denominator": 200,
            },
        },
        "performance_plan": {
            "read_only_shards": 400,
            "parallel_io": True,
            "recommended_workers": "min(32, max(4, 2*logical_cpu_count))",
            "dataframe_materialization": False,
            "GPU": "NOT_USED",
        },
        "frozen_sources": {
            "05a_method_contract": file_record(CONTRACT_05A),
            "frozen_stability_helper": file_record(HELPER),
            "05f2_progress_manifest": file_record(PROGRESS_05F2),
            "05f2_completion_lock": file_record(LOCK_05F2),
            "05f3e_results_lock": file_record(LOCK_05F3E),
            "05f4b_asset_lock": file_record(LOCK_05F4B),
            "05x0_preflight_lock": file_record(LOCK_05X0),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(contract, OUTPUT_CONTRACT)

    readme = f"""Paper 5 05x1 — frozen post-unblinding exploratory retention ablation contract
================================================================================

Status: {STATUS}

No ablation result is computed in 05x1.

Frozen subbanks
---------------
Primary exploratory stress mechanism:
  {CORRELATED_PRIVATE_SCENARIO_ID} ({EXPECTED_DATASETS_PER_SUBBANK} datasets)

Matched-n mechanism control:
  {independent_n62_id} ({EXPECTED_DATASETS_PER_SUBBANK} datasets)

Frozen rules
------------
FULL:
  median>=0.70, q10>=0.40, inclusion>=0.80,
  variance>=0.0001, n_supported_modalities>=2

NO_MODALITY_SUPPORT:
  same as FULL but remove only n_supported_modalities>=2

PURE_FACTOR_REPRODUCIBILITY:
  median>=0.70, q10>=0.40, inclusion>=0.80

The analysis is exploratory, post-unblinding, read-only, and cannot alter 05f3e.
"""
    write_text_atomic(readme, OUTPUT_README)

    lock = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "script_path": relative_path(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "created_at_utc": utc_now_iso(),
        "status": STATUS,
        "contract": file_record(OUTPUT_CONTRACT),
        "readme": file_record(OUTPUT_README),
        "scientific_result_computed": False,
        "rewrite_policy": (
            "05x1 is frozen before the exploratory ablation result is computed. "
            "Do not alter selectors, rules, thresholds, metrics, or expected reference "
            "checks after 05x2 opens the selected sealed payloads."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    print("05x1 exploratory ablation contract: FROZEN")
    print(f"Correlated-private: {CORRELATED_PRIVATE_SCENARIO_ID} (200)")
    print(f"Independent N62:   {independent_n62_id} (200)")
    print("Rules frozen: FULL / NO_MODALITY_SUPPORT / PURE_FACTOR_REPRODUCIBILITY")
    print(f"Contract: {relative_path(OUTPUT_CONTRACT)}")
    print(f"Lock:     {relative_path(OUTPUT_LOCK)}")
    print("Scientific ablation result computed: NO")


if __name__ == "__main__":
    main()
