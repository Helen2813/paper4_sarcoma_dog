#!/usr/bin/env python
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "05y1-freeze-correlated-private-failure-diagnostic-contract-v1-no-cli"
CONTRACT_VERSION = "paper5-05y1-postunblinding-correlated-private-diagnostic-v1"
STATUS = "FROZEN_POSTUNBLINDING_EXPLORATORY_DIAGNOSTIC_BEFORE_RESULT_COMPUTATION"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

GENERATOR = PAPER5_ROOT / "scripts" / "05b_generate_small_n_multiomics_simulations_v2.py"
SCENARIO_REGISTRY = PAPER5_ROOT / "method_contract" / "05a1" / "final_gate_calibration_scenarios.csv"
PROGRESS_05F2 = PAPER5_ROOT / "method_contract" / "05f2" / "full_run_progress_manifest.csv"

METHOD_05A = PAPER5_ROOT / "method_contract" / "05a" / "ai_method_contract.json"
ENV_05A = PAPER5_ROOT / "method_contract" / "05a" / "environment_snapshot.json"

LOCK_05F3E = PAPER5_ROOT / "method_contract" / "05f3e" / "final_gate_results_lock.json"
LOCK_05F4B = PAPER5_ROOT / "method_contract" / "05f4b" / "05f4b_manuscript_assets_lock.json"
LOCK_05X2B = PAPER5_ROOT / "method_contract" / "05x2b" / "05x2b_results_lock.json"
LOCK_05Y0B = PAPER5_ROOT / "method_contract" / "05y0b" / "05y0b_preflight_lock.json"
LOCK_05Y0C = PAPER5_ROOT / "method_contract" / "05y0c" / "05y0c_preflight_lock.json"
LOCK_05Y0D = PAPER5_ROOT / "method_contract" / "05y0d" / "05y0d_preflight_lock.json"
REPORT_05Y0D = PAPER5_ROOT / "method_contract" / "05y0d" / "private_score_generator_contract.json"

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05y1"
OUTPUT_CONTRACT = OUTPUT_DIR / "correlated_private_failure_diagnostic_contract.json"
OUTPUT_README = OUTPUT_DIR / "05y1_README.txt"
OUTPUT_LOCK = OUTPUT_DIR / "05y1_contract_lock.json"

G19 = "G19_CORRELATED_PRIVATE_NULL_N62"
STANDARD = "G07_IND_N62_S1P00"

EXPECTED_G19_DATASETS = 200
EXPECTED_STANDARD_DATASETS = 100
EXPECTED_G19_FROZEN_EVENTS = 70
EXPECTED_STANDARD_D02_NUMERATOR = 371
EXPECTED_STANDARD_D02_DENOMINATOR = 400

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260915
GROUP_CV_FOLDS = 5
GROUP_CV_SEED = 20260915

FROZEN_THRESHOLDS = {
    "median_aligned_correlation": 0.70,
    "q10_aligned_correlation": 0.40,
    "inclusion_frequency": 0.80,
    "reference_factor_variance": 0.0001,
    "n_supported_modalities": 2,
}

FRONTIER_GRID = {
    "median_start": 0.00,
    "median_stop": 1.00,
    "median_step": 0.05,
    "q10_start": 0.00,
    "q10_stop": 1.00,
    "q10_step": 0.05,
    "inclusion_start": 0.00,
    "inclusion_stop": 1.00,
    "inclusion_step": 0.05,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def scenario_rows() -> dict[str, dict[str, str]]:
    require_file(SCENARIO_REGISTRY)
    with SCENARIO_REGISTRY.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = {
            str(row["scenario_id"]).strip(): row
            for row in csv.DictReader(fh)
        }
    for sid in (G19, STANDARD):
        if sid not in rows:
            raise RuntimeError(f"Scenario registry missing {sid}.")
    return rows


def count_manifest_scenarios() -> dict[str, int]:
    require_file(PROGRESS_05F2)
    counts: dict[str, int] = {}
    total = 0
    with PROGRESS_05F2.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            total += 1
            sid = str(row["scenario_id"]).strip()
            counts[sid] = counts.get(sid, 0) + 1
    if total != 3100:
        raise RuntimeError(f"05f2 manifest has {total} rows; expected 3100.")
    return counts


def main() -> None:
    print("=" * 96)
    print("Paper 5 - freeze correlated-private failure diagnostic contract")
    print("=" * 96)
    print(f"Script version: {SCRIPT_VERSION}")
    print("Scientific diagnostic result computed: NO")
    print("70-vs-130 association computed: NO")
    print("Threshold frontier computed: NO")
    print("Formal 05f3e gate modification: FORBIDDEN")
    print()

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(
            f"05y1 output directory already exists and is non-empty: {OUTPUT_DIR}. "
            "Do not overwrite a frozen diagnostic contract."
        )

    for path in [
        GENERATOR,
        SCENARIO_REGISTRY,
        PROGRESS_05F2,
        METHOD_05A,
        ENV_05A,
        REPORT_05Y0D,
    ]:
        require_file(path)

    verify_lock(LOCK_05F3E, "FROZEN_FINAL_GATE_SCIENTIFIC_RESULTS_NO_RETUNING")
    verify_lock(LOCK_05F4B, "READ_ONLY_MANUSCRIPT_ASSET_EXPORT_FROM_LOCKED_05F3E")
    verify_lock(LOCK_05X2B, "POSTUNBLINDING_EXPLORATORY_RETENTION_ABLATION_COMPLETE")
    verify_lock(LOCK_05Y0B, "READ_ONLY_SUPPLEMENTAL_PREFLIGHT_NO_DIAGNOSTIC_RESULT_COMPUTED")
    verify_lock(LOCK_05Y0C, "READ_ONLY_SEMANTIC_SOURCE_PREFLIGHT_NO_DIAGNOSTIC_RESULT_COMPUTED")
    verify_lock(LOCK_05Y0D, "READ_ONLY_GENERATOR_SOURCE_EXTRACTION_NO_DIAGNOSTIC_RESULT_COMPUTED")

    y0d = load_json(REPORT_05Y0D)
    selected_generator = y0d.get("selected_generator", {})
    if str(selected_generator.get("path", "")).strip() != relative_path(GENERATOR):
        raise RuntimeError("05y0d selected a different generator source.")
    if str(selected_generator.get("function_name", "")).strip() != "generate_private_scores":
        raise RuntimeError("05y0d generator function is not generate_private_scores.")
    if sha256_file(GENERATOR).lower() != str(selected_generator.get("sha256", "")).lower():
        raise RuntimeError("Frozen 05b generator hash no longer matches 05y0d.")

    rows = scenario_rows()
    g19 = rows[G19]
    standard = rows[STANDARD]

    expected_g19 = {
        "scenario_family": "correlated_private_null",
        "n_samples": "62",
        "n_private_factors_per_view": "2",
        "private_cross_view_correlation": "0.6",
        "correlated_private_mode": "correlated_private_scores",
    }
    for key, expected in expected_g19.items():
        if str(g19.get(key, "")).strip() != expected:
            raise RuntimeError(
                f"G19 metadata changed for {key}: {g19.get(key)!r} != {expected!r}"
            )

    counts = count_manifest_scenarios()
    if counts.get(G19) != EXPECTED_G19_DATASETS:
        raise RuntimeError(f"G19 count changed: {counts.get(G19)}.")
    if counts.get(STANDARD) != EXPECTED_STANDARD_DATASETS:
        raise RuntimeError(f"STANDARD count changed: {counts.get(STANDARD)}.")

    method = load_json(METHOD_05A)
    thresholds = method.get("stability_gate", {}).get("thresholds", {})
    checks = {
        "primary_median_factor_correlation": 0.70,
        "primary_q10_factor_correlation": 0.40,
        "primary_inclusion_frequency": 0.80,
        "minimum_factor_variance": 0.0001,
        "minimum_supported_modalities": 2,
    }
    for key, expected in checks.items():
        if float(thresholds.get(key, float("nan"))) != float(expected):
            raise RuntimeError(f"Frozen 05a threshold mismatch for {key}.")

    contract = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": STATUS,
        "analysis_role": "POSTUNBLINDING_EXPLORATORY_FAILURE_CHARACTERIZATION",
        "scientific_boundary": {
            "formal_05f3e_gate_reopened": False,
            "formal_D01_reweighted": False,
            "threshold_retuning": False,
            "new_threshold_selected_on_same_data": False,
            "model_refit_of_SABLE_MO": False,
            "TARGET_or_outcomes_accessed": False,
            "claim_role": "EXPLORATORY_CHARACTERIZATION_ONLY",
        },
        "scenarios": {
            "correlated_private": {
                "scenario_id": G19,
                "datasets": EXPECTED_G19_DATASETS,
                "nominal_private_cross_view_correlation": 0.6,
                "n_private_factors_per_view": 2,
                "n_samples": 62,
            },
            "standard_true_shared_control": {
                "scenario_id": STANDARD,
                "datasets": EXPECTED_STANDARD_DATASETS,
                "formal_D02_denominator": EXPECTED_STANDARD_D02_DENOMINATOR,
            },
        },
        "primary_dataset_level_diagnostic": {
            "predictor_name": "realized_private_cross_view_correlation",
            "definition": (
                "Replay the exact frozen 05b G19 private-score generator from each dataset's "
                "frozen simulation_seed and exact pre-private RNG call order. For each of the "
                "2 private factors, compute Pearson correlation across each of the 3 unordered "
                "view pairs in the returned standardized private-score matrices. The primary "
                "dataset scalar is the arithmetic mean of these 6 signed Pearson correlations."
            ),
            "view_pairing": "all 3 unordered pairs among the frozen VIEWS",
            "factor_pairing": "same private-factor column only; no cross-factor matching",
            "absolute_value": False,
            "response": (
                "dataset-level false biological acceptance event: at least one factor with "
                "false_biological_acceptance=true"
            ),
            "integrity_reference": {
                "events": EXPECTED_G19_FROZEN_EVENTS,
                "datasets": EXPECTED_G19_DATASETS,
            },
            "primary_statistic": (
                "ROC AUC for higher realized correlation predicting dataset false acceptance"
            ),
            "uncertainty": {
                "method": "stratified dataset bootstrap of AUC",
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "interval": "2.5th and 97.5th percentiles",
            },
            "outcome_blind_curve": {
                "bins": 5,
                "construction": (
                    "realized-correlation quintile cutpoints computed from predictor values only, "
                    "before attaching false-acceptance labels"
                ),
                "report": "numerator/denominator, rate, Wilson 95% CI per quintile",
                "additional_contrast": "Q5 minus Q1 failure-rate difference",
                "no_pass_fail_interpretation": True,
            },
            "secondary_predictor_summaries": [
                "median of the 6 same-factor pairwise correlations",
                "minimum of the 6 same-factor pairwise correlations",
                "maximum of the 6 same-factor pairwise correlations",
                "standard deviation of the 6 same-factor pairwise correlations",
            ],
        },
        "factor_space_diagnostics": {
            "features_primary": [
                "median_aligned_correlation",
                "q10_aligned_correlation",
                "inclusion_frequency",
            ],
            "classifier": {
                "model": "L2 logistic regression",
                "C": 1.0,
                "solver": "lbfgs",
                "class_weight": "balanced",
                "training_standardization": "StandardScaler fit within each training fold only",
                "grouping": "dataset_key",
                "folds": GROUP_CV_FOLDS,
                "fold_assignment_seed": GROUP_CV_SEED,
                "metric": "out-of-fold ROC AUC",
                "cluster_bootstrap": {
                    "unit": "dataset_key",
                    "replicates": BOOTSTRAP_REPLICATES,
                    "seed": BOOTSTRAP_SEED,
                },
            },
            "target_contrast": (
                "G19 retained false-biological factors versus STANDARD retained "
                "biological-truth-matched factors"
            ),
            "negative_scenario_signature_anchor": (
                "G19 non-retained factors versus STANDARD non-retained factors, "
                "using the same features/model/CV"
            ),
            "positive_pipeline_anchor": (
                "within G19: retained false-biological factors versus non-retained factors, "
                "using the same features/model/CV; interpreted only as calibration because "
                "retention itself is defined from these features"
            ),
            "predefined_calibration_outputs": [
                "target_AUC",
                "negative_anchor_AUC",
                "positive_anchor_AUC",
                "absolute_distance_target_to_negative",
                "absolute_distance_target_to_positive",
                "boolean_target_closer_to_negative_than_positive",
            ],
            "claim_limit": (
                "No universal identifiability claim. The strongest allowed wording is empirical "
                "separability within the frozen retention-summary space."
            ),
        },
        "threshold_frontier": {
            "role": "POSTUNBLINDING_EXPLORATORY_FRONTIER_NO_SELECTION",
            "swept_thresholds": {
                "median_aligned_correlation": [0.00, 1.00, 0.05],
                "q10_aligned_correlation": [0.00, 1.00, 0.05],
                "inclusion_frequency": [0.00, 1.00, 0.05],
            },
            "fixed_constraints": {
                "reference_factor_variance_min": 0.0001,
                "n_supported_modalities_min": 2,
            },
            "candidate_rule": (
                "median>=m AND q10>=q AND inclusion>=i AND variance>=0.0001 "
                "AND n_supported_modalities>=2"
            ),
            "G19_metric": (
                "dataset false-acceptance rate: any candidate-retained factor that is neither "
                "technical-truth-matched nor biological-truth-matched"
            ),
            "STANDARD_metric": (
                "candidate-retained biological-truth-matched factor count divided by the frozen "
                "400 STANDARD truth-factor opportunities; implementation must reproduce the "
                "frozen D02 numerator 371 at the frozen point"
            ),
            "frozen_point": {
                "median": 0.70,
                "q10": 0.40,
                "inclusion": 0.80,
                "G19_reference": [70, 200],
                "STANDARD_reference": [371, 400],
            },
            "pareto_definition": (
                "A grid point is dominated if another point has no higher G19 false-acceptance "
                "rate and no lower STANDARD recovery, with at least one strict improvement."
            ),
            "report": [
                "all grid points",
                "Pareto-efficient flag",
                "whether frozen point is Pareto-efficient",
                "number of grid points dominating frozen point",
            ],
            "selection_from_frontier": "FORBIDDEN",
        },
        "interpretation_precommitment": {
            "high_realized_correlation_pattern": (
                "If failure rates rise across predictor-only realized-rho quintiles, discuss this "
                "as consistent with increasing observational similarity of correlated-private and "
                "shared structure within this simulator; do not generalize as a theorem."
            ),
            "broad_realized_correlation_pattern": (
                "If failures are distributed across lower and middle realized-rho quintiles rather "
                "than concentrated toward the upper tail, state that the frozen criterion's failure "
                "is not confined to near-equivalence realizations and that a new information source "
                "would be required for a prospective discriminator."
            ),
            "frontier_pattern": (
                "If reducing G19 false acceptance requires lower STANDARD recovery, characterize "
                "the overlap as an empirical trade-off in the frozen summary space. If grid points "
                "dominate the frozen point, report them descriptively but do not adopt a new threshold "
                "without fresh-seed prospective validation."
            ),
        },
        "implementation_integrity": {
            "generator_source": file_record(GENERATOR),
            "generator_function": "generate_private_scores",
            "pre_private_rng_call": (
                "generate_correlated_scores(root_rng, n, scenario.n_shared_factors, "
                "scenario.shared_factor_correlation)"
            ),
            "rng_initialization": "np.random.default_rng(simulation_seed)",
            "seed_source": (
                "frozen simulation seed associated with each sealed G19 dataset; implementation "
                "may read it from 05f2 manifest or sealed dataset metadata but must verify consistency"
            ),
            "environment_snapshot": file_record(ENV_05A),
            "replay_must_use_exact_frozen_generator_source": True,
        },
        "sources": {
            "05a_method_contract": file_record(METHOD_05A),
            "05a_environment_snapshot": file_record(ENV_05A),
            "05a1_scenario_registry": file_record(SCENARIO_REGISTRY),
            "05f2_progress_manifest": file_record(PROGRESS_05F2),
            "05f3e_lock": file_record(LOCK_05F3E),
            "05f4b_lock": file_record(LOCK_05F4B),
            "05x2b_lock": file_record(LOCK_05X2B),
            "05y0b_lock": file_record(LOCK_05Y0B),
            "05y0c_lock": file_record(LOCK_05Y0C),
            "05y0d_lock": file_record(LOCK_05Y0D),
            "05y0d_report": file_record(REPORT_05Y0D),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(contract, OUTPUT_CONTRACT)

    readme = f"""Paper 5 05y1 — frozen post-unblinding exploratory diagnostic contract
=======================================================================

No diagnostic result is computed here.

Primary dataset diagnostic:
  realized private cross-view correlation reconstructed by exact frozen generator replay
  -> predicts the frozen G19 dataset false-acceptance event.

Primary scalar:
  mean signed Pearson correlation over 2 private factors x 3 unordered view pairs.

Factor-space diagnostics:
  median / q10 / inclusion only, grouped by dataset.
  Target contrast + matched-status scenario-signature anchor + within-G19 positive anchor.

Threshold frontier:
  full 0.00..1.00 grid in 0.05 steps for median/q10/inclusion.
  Variance and modality-support constraints remain fixed.
  No new threshold may be selected on these data.

Formal 05f3e consequence:
  NONE.
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
        "contract": file_record(OUTPUT_CONTRACT),
        "readme": file_record(OUTPUT_README),
        "rewrite_policy": (
            "05y1 is frozen before 05y2 computes any failure diagnostic. "
            "Do not change predictor definition, classifier, anchors, grid, "
            "bootstrap plan, or interpretation rules after viewing results."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    print("05y1 diagnostic contract: FROZEN")
    print("Primary realized-rho scalar: mean of 6 signed same-factor view-pair correlations")
    print("Primary response: frozen G19 dataset false-acceptance event (reference 70/200)")
    print("Factor-space CV: grouped 5-fold logistic; dataset_key groups")
    print("Threshold frontier: 0..1 by 0.05 for median/q10/inclusion")
    print("Frontier selection/adoption on these data: FORBIDDEN")
    print(f"Contract: {relative_path(OUTPUT_CONTRACT)}")
    print(f"Lock:     {relative_path(OUTPUT_LOCK)}")
    print("Scientific diagnostic result computed: NO")


if __name__ == "__main__":
    main()
