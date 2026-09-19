#!/usr/bin/env python
from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "05y3-freeze-prospective-q10-recalibration-contract-v1-no-cli"
CONTRACT_VERSION = "paper5-05y3-prospective-q10-recalibration-v1"
STATUS = "FROZEN_PROSPECTIVE_Q10_RECALIBRATION_BEFORE_FRESH_BANK"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

METHOD_05A = PAPER5_ROOT / "method_contract" / "05a" / "ai_method_contract.json"
SCENARIOS_05A1 = PAPER5_ROOT / "method_contract" / "05a1" / "final_gate_calibration_scenarios.csv"
REPLICATES_05A1 = PAPER5_ROOT / "method_contract" / "05a1" / "final_gate_calibration_replicates.csv"
ARCH_REPLICATES_05A = PAPER5_ROOT / "method_contract" / "05a" / "simulation_replicate_registry.csv"
ARCH_MANIFEST_05B = PAPER5_ROOT / "method_contract" / "05b_v2" / "architecture_screening_data_manifest.csv"

GENERATOR_05B = PAPER5_ROOT / "scripts" / "05b_generate_small_n_multiomics_simulations_v2.py"
SCRIPT_05F2 = PAPER5_ROOT / "scripts" / "05f2_run_full_final_gate_gpu_resumable.py"

LOCK_05F3E = PAPER5_ROOT / "method_contract" / "05f3e" / "final_gate_results_lock.json"
LOCK_05F4B = PAPER5_ROOT / "method_contract" / "05f4b" / "05f4b_manuscript_assets_lock.json"
LOCK_05Y2B = PAPER5_ROOT / "method_contract" / "05y2b" / "05y2b_results_lock.json"
RESULT_05Y2B = PAPER5_ROOT / "method_contract" / "05y2b" / "correlated_private_failure_diagnostic_result.json"
FRONTIER_05Y2B = PAPER5_ROOT / "method_contract" / "05y2b" / "retention_threshold_frontier.csv"

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05y3"
OUTPUT_CONTRACT = OUTPUT_DIR / "prospective_q10_recalibration_contract.json"
OUTPUT_API = OUTPUT_DIR / "fresh_validation_execution_api_snapshot.json"
OUTPUT_README = OUTPUT_DIR / "05y3_README.txt"
OUTPUT_LOCK = OUTPUT_DIR / "05y3_contract_lock.json"

G19 = "G19_CORRELATED_PRIVATE_NULL_N62"
STANDARD = "G07_IND_N62_S1P00"

ORIGINAL_RULE = {
    "median_aligned_correlation_min": 0.70,
    "q10_aligned_correlation_min": 0.40,
    "inclusion_frequency_min": 0.80,
    "reference_factor_variance_min": 0.0001,
    "n_supported_modalities_min": 2,
    "loading_energy_fraction_per_supported_modality_min": 0.15,
}
CANDIDATE_RULE = {
    **ORIGINAL_RULE,
    "q10_aligned_correlation_min": 0.60,
}

FRESH_DATASETS_PER_SCENARIO = 200
FRESH_BANK_MASTER_SEED = 20260915
STABILITY_RESAMPLES = 100
STABILITY_SUBSAMPLE_FRACTION = 0.80
PRIMARY_ALPHA = 0.05


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def import_module(path: Path, name: str):
    require_file(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_scenarios() -> dict[str, dict[str, str]]:
    require_file(SCENARIOS_05A1)
    with SCENARIOS_05A1.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = {
            str(row["scenario_id"]).strip(): row
            for row in csv.DictReader(fh)
        }
    for sid in (G19, STANDARD):
        if sid not in rows:
            raise RuntimeError(f"Missing frozen scenario {sid}.")
    return rows


def parse_float(value: Any) -> float:
    return float(str(value).strip())


def exact_frontier_row() -> dict[str, str]:
    require_file(FRONTIER_05Y2B)
    with FRONTIER_05Y2B.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = []
        for row in csv.DictReader(fh):
            if (
                abs(parse_float(row["median_threshold"]) - 0.70) < 1e-12
                and abs(parse_float(row["q10_threshold"]) - 0.60) < 1e-12
                and abs(parse_float(row["inclusion_threshold"]) - 0.80) < 1e-12
            ):
                rows.append(row)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one q10=.60 discovery row; found {len(rows)}.")
    row = rows[0]
    expected = {
        "g19_false_event_numerator": 41,
        "g19_false_event_denominator": 200,
        "standard_recovery_numerator": 368,
        "standard_recovery_denominator": 400,
    }
    for key, value in expected.items():
        if int(float(row[key])) != value:
            raise RuntimeError(
                f"Discovery evidence mismatch for {key}: {row[key]} != {value}"
            )
    return row


def verify_frozen_thresholds() -> None:
    require_file(METHOD_05A)
    method = load_json(METHOD_05A)
    thresholds = method.get("stability_gate", {}).get("thresholds", {})
    expected = {
        "primary_median_factor_correlation": 0.70,
        "primary_q10_factor_correlation": 0.40,
        "primary_inclusion_frequency": 0.80,
        "minimum_factor_variance": 0.0001,
        "minimum_supported_modalities": 2,
        "minimum_loading_energy_fraction_per_supported_modality": 0.15,
    }
    for key, value in expected.items():
        if float(thresholds.get(key, float("nan"))) != float(value):
            raise RuntimeError(f"Frozen 05a threshold mismatch for {key}.")


def inspect_execution_api() -> dict[str, Any]:
    f2 = import_module(SCRIPT_05F2, "paper5_05y3_inspect_05f2")
    runner_path = Path(f2.RUNNER_PATH)
    if not runner_path.is_absolute():
        runner_path = PROJECT_ROOT / runner_path
    require_file(runner_path)
    runner = import_module(runner_path, "paper5_05y3_inspect_05f1_runner")

    epoch_bridge = Path(runner.OUTPUT_EPOCH_BRIDGE)
    if not epoch_bridge.is_absolute():
        epoch_bridge = PROJECT_ROOT / epoch_bridge
    require_file(epoch_bridge)
    epoch_json = load_json(epoch_bridge)
    if "final_gate_fixed_epoch" not in epoch_json:
        raise RuntimeError("Frozen epoch bridge lacks final_gate_fixed_epoch.")

    callable_signatures: dict[str, str] = {}
    for name in sorted(dir(runner)):
        low = name.lower()
        if not any(token in low for token in ("final_gate", "dataset", "upstream", "stability")):
            continue
        obj = getattr(runner, name)
        if callable(obj):
            try:
                callable_signatures[name] = str(inspect.signature(obj))
            except Exception:
                pass

    return {
        "05f2_script": file_record(SCRIPT_05F2),
        "runner_source": file_record(runner_path),
        "runner_path": relative_path(runner_path),
        "epoch_bridge": file_record(epoch_bridge),
        "final_gate_fixed_epoch": int(epoch_json["final_gate_fixed_epoch"]),
        "runner_callable_signatures": callable_signatures,
        "runner_globals": {
            "OUTPUT_SEED_REGISTRY": relative_path(
                Path(runner.OUTPUT_SEED_REGISTRY)
                if Path(runner.OUTPUT_SEED_REGISTRY).is_absolute()
                else PROJECT_ROOT / Path(runner.OUTPUT_SEED_REGISTRY)
            ),
        },
    }


def main() -> None:
    print("=" * 98)
    print("Paper 5 - freeze prospective q10 recalibration contract")
    print("=" * 98)
    print(f"Script version: {SCRIPT_VERSION}")
    print("Fresh validation results accessed: NO")
    print("Fresh datasets generated: NO")
    print("Candidate retuning after fresh results: FORBIDDEN")
    print("Original 05f3e formal gate modification: FORBIDDEN")
    print()

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(
            f"05y3 output directory already exists and is non-empty: {OUTPUT_DIR}"
        )

    for path in [
        METHOD_05A,
        SCENARIOS_05A1,
        REPLICATES_05A1,
        ARCH_REPLICATES_05A,
        ARCH_MANIFEST_05B,
        GENERATOR_05B,
        SCRIPT_05F2,
        RESULT_05Y2B,
        FRONTIER_05Y2B,
    ]:
        require_file(path)

    verify_lock(LOCK_05F3E, "FROZEN_FINAL_GATE_SCIENTIFIC_RESULTS_NO_RETUNING")
    verify_lock(LOCK_05F4B, "READ_ONLY_MANUSCRIPT_ASSET_EXPORT_FROM_LOCKED_05F3E")
    verify_lock(LOCK_05Y2B, "POSTUNBLINDING_EXPLORATORY_FAILURE_DIAGNOSTICS_COMPLETE")

    verify_frozen_thresholds()
    frontier_row = exact_frontier_row()
    scenarios = read_scenarios()
    g19 = scenarios[G19]
    standard = scenarios[STANDARD]

    if int(float(g19["n_samples"])) != 62:
        raise RuntimeError("G19 n_samples changed.")
    if int(float(standard["n_samples"])) != 62:
        raise RuntimeError("STANDARD n_samples changed.")
    if int(float(standard["true_biological_shared_factors"])) != 4:
        raise RuntimeError("STANDARD no longer contains 4 biological truth factors.")
    if int(float(g19["true_biological_shared_factors"])) != 0:
        raise RuntimeError("G19 biological truth-null definition changed.")
    if int(float(g19["true_technical_shared_factors"])) != 0:
        raise RuntimeError("G19 technical truth-null definition changed.")

    api = inspect_execution_api()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(api, OUTPUT_API)

    contract = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": STATUS,
        "analysis_role": "PROSPECTIVE_POST_DISCOVERY_METHOD_RECALIBRATION_VALIDATION",
        "discovery_provenance": {
            "source_stage": "05y2b",
            "candidate_was_selected_after_discovery_frontier": True,
            "discovery_bank_is_forbidden_for_confirmation": True,
            "discovery_observation": {
                "original_G19": [70, 200],
                "candidate_G19": [41, 200],
                "original_STANDARD": [371, 400],
                "candidate_STANDARD": [368, 400],
            },
            "frontier_row": frontier_row,
        },
        "original_rule": ORIGINAL_RULE,
        "single_candidate_rule": CANDIDATE_RULE,
        "only_changed_component": {
            "field": "q10_aligned_correlation_min",
            "original": 0.40,
            "candidate": 0.60,
        },
        "fresh_bank_design": {
            "master_seed": FRESH_BANK_MASTER_SEED,
            "seed_derivation": (
                "SHA256(namespace|master_seed|scenario_id|fresh_replicate_index|purpose|retry) "
                "mapped to positive signed-31-bit integer; reject zero, duplicates, and any "
                "simulation seed present in frozen prior registries."
            ),
            "scenarios": [
                {
                    "source_scenario_id": G19,
                    "role": "PRIMARY_CORRELATED_PRIVATE_SPECIFICITY_STRESS",
                    "fresh_datasets": FRESH_DATASETS_PER_SCENARIO,
                },
                {
                    "source_scenario_id": STANDARD,
                    "role": "TRUE_SHARED_SENSITIVITY_SAFETY",
                    "fresh_datasets": FRESH_DATASETS_PER_SCENARIO,
                },
            ],
            "total_fresh_datasets": 2 * FRESH_DATASETS_PER_SCENARIO,
            "zero_simulation_seed_overlap_required_with": [
                relative_path(ARCH_REPLICATES_05A),
                relative_path(ARCH_MANIFEST_05B),
                relative_path(REPLICATES_05A1),
            ],
            "permanent_input_npz_bank": False,
            "streaming_policy": (
                "Fresh datasets are regenerated deterministically from frozen seed registry "
                "immediately before evaluation and may be discarded after each sealed result shard."
            ),
        },
        "execution_contract": {
            "stability_resamples_per_dataset": STABILITY_RESAMPLES,
            "stability_subsample_fraction": STABILITY_SUBSAMPLE_FRACTION,
            "neural_profile": "EXACT_FROZEN_FINAL_GATE_PROFILE",
            "fixed_epoch": api["final_gate_fixed_epoch"],
            "numerical_path": "EXACT_FROZEN_05F2_05F1_RUNNER_PATH",
            "generator_source": file_record(GENERATOR_05B),
            "05f2_source": api["05f2_script"],
            "05f1_runner_source": api["runner_source"],
            "fresh_thresholds_are_applied_post_fit_to_the_same_factor_metrics": True,
            "duplicate_model_fits_for_original_and_candidate_rules": False,
            "reason": (
                "Original and candidate differ only in q10 threshold; one fresh model/stability "
                "evaluation supplies factor metrics for both nested rules."
            ),
        },
        "prospective_endpoints": {
            "primary_specificity_endpoint": {
                "population": G19,
                "unit": "dataset",
                "original_event": (
                    "at least one original-rule retained factor that is neither biological nor technical truth matched"
                ),
                "candidate_event": (
                    "at least one candidate-rule retained factor that is neither biological nor technical truth matched"
                ),
                "paired_by_same_fresh_dataset": True,
                "nesting_requirement": "candidate_event MUST be subset of original_event",
                "primary_test": {
                    "method": "exact two-sided McNemar/binomial test on discordant pairs",
                    "alpha": PRIMARY_ALPHA,
                    "required_direction": "candidate false-acceptance rate < original rate",
                },
                "report": [
                    "original numerator/200, rate, Wilson 95% CI",
                    "candidate numerator/200, rate, Wilson 95% CI",
                    "rescued datasets original=1 candidate=0",
                    "reverse discordance original=0 candidate=1 (must equal 0)",
                    "paired absolute rate reduction",
                    "bootstrap 95% CI for paired rate reduction",
                    "exact McNemar/binomial p-value",
                ],
            },
            "sensitivity_safety_endpoint": {
                "population": STANDARD,
                "unit": "truth-factor opportunity",
                "truth_factors_per_dataset": 4,
                "fresh_denominator": 800,
                "original_recovery": "retained AND biological_truth_match",
                "candidate_recovery": "candidate-retained AND biological_truth_match",
                "existing_safety_floor": 0.80,
                "floor_provenance": "original frozen D02 threshold",
                "report": [
                    "original numerator/800 and rate",
                    "candidate numerator/800 and rate",
                    "candidate Wilson 95% CI",
                    "paired lost/gained truth opportunities relative to original",
                ],
            },
        },
        "prospective_confirmation_rule": {
            "candidate_confirmed": (
                "TRUE iff (1) exact paired specificity test p<0.05 in the required improvement "
                "direction, (2) reverse discordance is zero as required by nesting, and "
                "(3) candidate STANDARD biological recovery point estimate >=0.80."
            ),
            "no_new_noninferiority_margin": True,
            "if_not_confirmed": (
                "Report prospective failure/inconclusive result unchanged; do not modify q10 or "
                "select another frontier point on this fresh bank."
            ),
        },
        "guardrails": [
            "05f3e remains the authoritative original formal gate.",
            "The 05y2b discovery bank cannot be used as confirmatory evidence for q10=0.60.",
            "Only q10 changes from 0.40 to 0.60; all other retention thresholds remain fixed.",
            "No candidate replacement is allowed after any fresh result is viewed.",
            "Fresh simulation seeds must have zero overlap with all frozen prior simulation registries.",
            "Both original and candidate rules are evaluated on identical fresh datasets and identical fitted factor metrics.",
            "No TARGET clinical outcomes or biological annotations are accessed.",
        ],
        "sources": {
            "05a_method_contract": file_record(METHOD_05A),
            "05a1_scenarios": file_record(SCENARIOS_05A1),
            "05y2b_result": file_record(RESULT_05Y2B),
            "05y2b_frontier": file_record(FRONTIER_05Y2B),
            "05y2b_lock": file_record(LOCK_05Y2B),
            "05f3e_lock": file_record(LOCK_05F3E),
            "05f4b_lock": file_record(LOCK_05F4B),
            "generator_05b": file_record(GENERATOR_05B),
        },
    }

    write_json_atomic(contract, OUTPUT_CONTRACT)

    readme = """Paper 5 05y3 — prospective q10 recalibration freeze
========================================================

DISCOVERY:
  q10 0.40 -> 0.60 was selected after the sealed 05y2b exploratory frontier.

PROSPECTIVE CANDIDATE:
  median >= 0.70
  q10    >= 0.60
  inclusion >= 0.80
  variance >= 0.0001
  supported modalities >= 2
  loading-energy support threshold >= 0.15

FRESH VALIDATION:
  200 correlated-private G19-like datasets
  200 STANDARD true-shared datasets
  zero overlap with prior simulation seeds
  100 stability resamples per dataset
  original and candidate evaluated on the same factor metrics

CONFIRMATION:
  paired G19 specificity improvement at alpha 0.05
  AND candidate STANDARD recovery >= the existing D02 floor of 0.80

No replacement threshold may be selected after fresh results are viewed.
Original 05f3e remains authoritative.
"""
    write_text_atomic(readme, OUTPUT_README)

    lock = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "script_path": relative_path(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "created_at_utc": utc_now_iso(),
        "status": STATUS,
        "scientific_result_computed": False,
        "fresh_dataset_generated": False,
        "contract": file_record(OUTPUT_CONTRACT),
        "api_snapshot": file_record(OUTPUT_API),
        "readme": file_record(OUTPUT_README),
        "rewrite_policy": (
            "05y3 is immutable once frozen. q10=0.60, endpoints, fresh-bank size, "
            "seed algorithm, decision rule, and execution path cannot change after fresh data generation."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    print("05y3 prospective q10 contract: FROZEN")
    print("Only changed threshold: q10 0.40 -> 0.60")
    print("Fresh bank: 200 G19-like + 200 STANDARD")
    print(f"Fresh STANDARD truth-factor denominator: {200 * 4}")
    print("Stability resamples/dataset: 100")
    print(f"Frozen final-gate epoch: {api['final_gate_fixed_epoch']}")
    print(f"Frozen runner: {api['runner_path']}")
    print("Fresh scientific results computed: NO")
    print(f"Contract: {relative_path(OUTPUT_CONTRACT)}")
    print(f"API:      {relative_path(OUTPUT_API)}")
    print(f"Lock:     {relative_path(OUTPUT_LOCK)}")


if __name__ == "__main__":
    main()
