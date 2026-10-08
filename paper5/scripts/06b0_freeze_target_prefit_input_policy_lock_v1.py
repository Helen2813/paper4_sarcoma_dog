#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCRIPT_VERSION = "06b0-freeze-target-prefit-input-policy-lock-v1-no-cli"
CONTRACT_VERSION = "paper5-06b0-target-prefit-input-policy-lock-v1"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

INPUT_ROOT = PAPER5_ROOT / "model_inputs" / "TARGET_OS_primary78"
HARMONIZED = INPUT_ROOT / "harmonized"
PREMODEL = INPUT_ROOT / "premodel_qc"

MC05A = PAPER5_ROOT / "method_contract" / "05a"
MC05E = PAPER5_ROOT / "method_contract" / "05e"
MC06A = PAPER5_ROOT / "method_contract" / "06a"
TECH_ROOT = PAPER5_ROOT / "data_manifest" / "TARGET_OS_technical_metadata"

OUT = PAPER5_ROOT / "method_contract" / "06b0"
CONTRACT_OUT = OUT / "target_prefit_input_policy_contract.json"
README_OUT = OUT / "06b0_README.txt"
LOCK_OUT = OUT / "06b0_prefit_lock.json"

# Immutable anchors already established upstream.
EXPECTED_CASE_ORDER_SHA256 = "9a39a40f0ed6be767d9640219b4b3f7ee0813b8e81e6db64880a1fe1bcd44cb4"
EXPECTED_HARMONIZED_LOCK_SHA256 = "04c32b6fefed42f433fc29e4f48413a730dc4504a46cfde8de6e7b6c45c99cac"
EXPECTED_05E_FREEZE_SHA256 = "df01013be4d87fe7b67ea31a5d038ace8642ecb8cfc56d13b1bfa6414cb2d60f"
EXPECTED_06A_CONTRACT_SHA256 = "4459333373437d262fb0d37ec88b6767c7df31dca397f8604756cf2055229dcf"
EXPECTED_06A_LOCK_SHA256 = "d534d48c8ff116e0c45b539e7e252badd573f1a9d600a89e4bb2e12dd26b6f77"

CASE_ORDER = INPUT_ROOT / "TARGET_OS_primary_multimodal_case_order.csv"

RAW_INPUTS = {
    "RNA": {
        "matrix": INPUT_ROOT / "TARGET_OS_primary_rna_log2tpm.npy",
        "features": INPUT_ROOT / "TARGET_OS_primary_rna_features.csv",
        "expected_shape": [78, 50422],
        "pre_z_transform": "identity_log2_TPM",
    },
    "METHYLATION": {
        "matrix": INPUT_ROOT / "TARGET_OS_primary_methylation_beta_corrected.npy",
        "features": INPUT_ROOT / "TARGET_OS_primary_methylation_features_corrected.csv",
        "expected_shape": [78, 486427],
        "pre_z_transform": "beta_to_M_value",
    },
    "CNV": {
        "matrix": INPUT_ROOT / "TARGET_OS_primary_cnv_gene_copy_number.npy",
        "features": INPUT_ROOT / "TARGET_OS_primary_cnv_features.csv",
        "expected_shape": [78, 50385],
        "pre_z_transform": "log2_copy_number_over_2",
    },
}

HARMONIZED_ANCHORS = {
    "RNA": {
        "matrix": HARMONIZED / "RNA_zscore.npy",
        "features": HARMONIZED / "RNA_features.csv",
        "expected_shape": [78, 48670],
    },
    "METHYLATION": {
        "matrix": HARMONIZED / "METHYLATION_Mvalue_zscore.npy",
        "features": HARMONIZED / "METHYLATION_features.csv",
        "expected_shape": [78, 407068],
    },
    "CNV": {
        "matrix": HARMONIZED / "CNV_log2ratio_zscore.npy",
        "features": HARMONIZED / "CNV_features.csv",
        "expected_shape": [78, 49561],
    },
}

HARMONIZED_LOCK = HARMONIZED / "harmonized_input_lock.json"
PREMODEL_MANIFEST = PREMODEL / "prestandardization_variance_audit_manifest.json"
FOLD_LOCAL_CONTRACT = PREMODEL / "fold_local_preprocessing_contract.json"
FEATURE_BUDGET_PLAN = PREMODEL / "representation_feature_budget_plan.csv"
RECONSTRUCTION_AUDIT = PREMODEL / "harmonized_reconstruction_audit.csv"

A05_FILES = [
    MC05A / "ai_method_contract.json",
    MC05A / "ai_method_contract_lock.json",
    MC05A / "model_selection_policy.json",
    MC05A / "model_family_registry.csv",
    MC05A / "baseline_hyperparameter_grid.csv",
    MC05A / "target_cv_split_summary.csv",
    MC05A / "target_outer_cv_assignments.csv",
    MC05A / "target_inner_cv_assignments.csv",
    MC05A / "target_neural_training_assignments.csv",
]

FREEZE_05E = MC05E / "representation_method_freeze.json"
LOCK_05E = MC05E / "representation_method_freeze_lock.json"

CONTRACT_06A = MC06A / "target_application_contract.json"
LOCK_06A = MC06A / "06a_contract_lock.json"

TECH_LOCK = TECH_ROOT / "TARGET_OS_technical_metadata_preflight_lock.json"
TECH_MANIFEST = TECH_ROOT / "TARGET_OS_technical_metadata_preflight_manifest.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen artifact missing: {path}")


def load_json(path: Path) -> dict[str, Any]:
    require(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def boolish(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def project_path(text: str) -> Path:
    s = str(text).replace("\\", "/")
    p = Path(s)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def iter_path_hash_records(value: Any) -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value["path"], value["sha256"]
        for child in value.values():
            yield from iter_path_hash_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_path_hash_records(child)


def verify_embedded_records(doc: dict[str, Any], label: str) -> dict[str, int]:
    seen = 0
    verified = 0
    for path_text, expected in iter_path_hash_records(doc):
        seen += 1
        p = project_path(path_text)
        if not p.is_file():
            # Some upstream documents may deliberately reference non-project/external files.
            # The pre-fit lock only requires present project-local records.
            continue
        observed = sha256_file(p)
        if observed.lower() != expected.lower():
            raise RuntimeError(
                f"{label}: hash mismatch for {p}\n"
                f"expected={expected}\nobserved={observed}"
            )
        verified += 1
    return {"path_hash_records_seen": seen, "project_local_records_verified": verified}


def file_record(path: Path) -> dict[str, Any]:
    require(path)
    return {
        "path": rel(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def matrix_record(path: Path, expected_shape: list[int]) -> dict[str, Any]:
    require(path)
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    shape = [int(x) for x in arr.shape]
    dtype = str(arr.dtype)
    if shape != expected_shape:
        raise RuntimeError(
            f"Unexpected matrix shape for {path}: {shape}; expected {expected_shape}"
        )
    del arr
    record = file_record(path)
    record["shape"] = shape
    record["dtype"] = dtype
    return record


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    require(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def check_case_order() -> dict[str, Any]:
    rows = read_csv_rows(CASE_ORDER)
    if len(rows) != 78:
        raise RuntimeError(f"TARGET case order must contain 78 rows, found {len(rows)}")
    if not rows or not rows[0]:
        raise RuntimeError("TARGET case-order CSV is empty or malformed.")
    first_col = next(iter(rows[0].keys()))
    ids = [str(row.get(first_col, "")).strip() for row in rows]
    if any(not x for x in ids) or len(set(ids)) != 78:
        raise RuntimeError("TARGET case order does not contain 78 unique non-empty first-column IDs.")
    observed = sha256_file(CASE_ORDER)
    if observed.lower() != EXPECTED_CASE_ORDER_SHA256:
        raise RuntimeError(
            f"TARGET case-order SHA256 changed: {observed}; expected {EXPECTED_CASE_ORDER_SHA256}"
        )
    return {
        **file_record(CASE_ORDER),
        "n_cases": 78,
        "first_column": first_col,
        "unique_first_column_ids": 78,
    }


def check_feature_budget() -> tuple[list[dict[str, str]], dict[str, int]]:
    rows = read_csv_rows(FEATURE_BUDGET_PLAN)
    required = {"RNA": 5000, "METHYLATION": 10000, "CNV": 5000}
    found: dict[str, int] = {}
    for row in rows:
        role = str(row.get("budget_role", "")).strip().lower()
        modality = str(row.get("modality", "")).strip().upper()
        if role == "primary" and modality in required:
            found[modality] = int(float(row["n_features"]))
    if found != required:
        raise RuntimeError(f"Primary feature-budget mismatch: observed={found}, expected={required}")
    return rows, found


def check_05a_policy() -> tuple[dict[str, Any], list[dict[str, str]]]:
    policy = load_json(MC05A / "model_selection_policy.json")
    if policy.get("outcome_access") is not False:
        raise RuntimeError("05a model-selection policy unexpectedly allows outcome access.")
    if policy.get("biological_annotation_access") is not False:
        raise RuntimeError("05a model-selection policy unexpectedly allows biological annotation access.")

    nested = policy.get("target_nested_selection", {})
    if nested.get("outer_design") != "5 repeats of 5-fold CV":
        raise RuntimeError("05a outer TARGET design changed.")
    if nested.get("inner_design") != "4-fold CV within each outer-training split":
        raise RuntimeError("05a inner TARGET design changed.")
    if nested.get("outer_test_samples_used_for_selection") is not False:
        raise RuntimeError("05a unexpectedly permits outer-test samples in selection.")

    full = policy.get("final_full_data_method_freeze", {})
    if float(full.get("minimum_single_method_selection_frequency", -1)) != 0.4:
        raise RuntimeError("05a final-method 40% selection-frequency rule changed.")
    if full.get("biological_or_outcome_results_may_not_change_method") is not True:
        raise RuntimeError("05a full-data method freeze no longer blocks biology/outcome changes.")

    registry = read_csv_rows(MC05A / "model_family_registry.csv")
    eligible = [
        row for row in registry
        if boolish(row.get("eligible_for_nested_selection", False))
    ]
    if not eligible:
        raise RuntimeError("05a model registry contains no nested-selection-eligible models.")
    return policy, eligible


def check_05e() -> dict[str, Any]:
    freeze = load_json(FREEZE_05E)
    if sha256_file(FREEZE_05E).lower() != EXPECTED_05E_FREEZE_SHA256:
        raise RuntimeError("05e representation freeze SHA256 changed.")
    if freeze.get("status") != "FROZEN_PRIMARY_PROPOSED_REPRESENTATION_METHOD_BEFORE_FINAL_GATE_AND_TARGET":
        raise RuntimeError("Unexpected 05e status.")
    primary = freeze.get("primary_proposed_representation", {})
    if primary.get("method_id") != "V4_SABLE_MO":
        raise RuntimeError("05e primary proposed method is not V4_SABLE_MO.")
    if primary.get("profile_id") != "ARCH_COMPACT__REG_WEAK":
        raise RuntimeError("05e selected neural profile changed.")
    if freeze.get("scientific_state", {}).get("TARGET_final_method_selected") is not False:
        raise RuntimeError("05e says a TARGET final method was already selected.")
    return freeze


def check_06a() -> dict[str, Any]:
    contract = load_json(CONTRACT_06A)
    if sha256_file(CONTRACT_06A).lower() != EXPECTED_06A_CONTRACT_SHA256:
        raise RuntimeError("06a contract SHA256 changed.")
    if sha256_file(LOCK_06A).lower() != EXPECTED_06A_LOCK_SHA256:
        raise RuntimeError("06a lock SHA256 changed.")
    if contract.get("status") != "FROZEN_BEFORE_TARGET_DATA_MODEL_FIT_OR_OUTCOME_INSPECTION":
        raise RuntimeError("Unexpected 06a status.")
    pr = contract.get("primary_retention_rule", {})
    sr = contract.get("historical_sensitivity_rule", {})
    if float(pr.get("q10_aligned_correlation_min", -1)) != 0.60:
        raise RuntimeError("06a primary q10 is not 0.60.")
    if float(sr.get("q10_aligned_correlation_min", -1)) != 0.40:
        raise RuntimeError("06a historical sensitivity q10 is not 0.40.")
    pol = contract.get("target_application_decision_policy", {})
    if pol.get("sensitivity_may_replace_primary") is not False:
        raise RuntimeError("06a sensitivity unexpectedly may replace primary.")
    if pol.get("threshold_relaxation_after_target_results") is not False:
        raise RuntimeError("06a unexpectedly permits threshold relaxation after TARGET.")
    return contract


def check_technical_preflight() -> tuple[dict[str, Any], dict[str, Any]]:
    lock = load_json(TECH_LOCK)
    manifest = load_json(TECH_MANIFEST)
    if lock.get("status") != "FROZEN_BEFORE_REPRESENTATION_MODEL_FITTING":
        raise RuntimeError("Unexpected TARGET technical-preflight lock status.")
    if manifest.get("status") != "FROZEN_BEFORE_REPRESENTATION_MODEL_FITTING":
        raise RuntimeError("Unexpected TARGET technical-preflight manifest status.")
    cohort = manifest.get("case_cohort", {})
    if int(cohort.get("n_cases", -1)) != 78:
        raise RuntimeError("Technical preflight does not cover 78 TARGET cases.")
    by_modality = cohort.get("selected_files_by_modality", {})
    if by_modality != {"CNV": 78, "METHYLATION": 78, "RNA": 78}:
        raise RuntimeError(f"Unexpected selected-file modality counts: {by_modality}")
    state = manifest.get("scientific_state", {})
    if state.get("outcome_loaded") is not False:
        raise RuntimeError("Technical preflight unexpectedly loaded outcomes.")
    if state.get("technical_covariates_used_for_model_selection") is not False:
        raise RuntimeError("Technical covariates were unexpectedly used for model selection.")
    future = manifest.get("future_factor_audit", {})
    if future.get("model_selection_use") is not False:
        raise RuntimeError("Technical factor audit unexpectedly permits model-selection use.")
    if future.get("automatic_batch_residualization") is not False:
        raise RuntimeError("Technical contract unexpectedly permits automatic residualization.")
    return lock, manifest


def main() -> None:
    print("=" * 104)
    print("Paper 5 - freeze TARGET pre-fit input/policy lock")
    print("=" * 104)
    print(f"Script version: {SCRIPT_VERSION}")
    print("TARGET molecular values summarized/inspected: NO")
    print("TARGET clinical outcomes read: NO")
    print("Biological annotations read: NO")
    print("TARGET model fit: NO")
    print("Runner source authorized here: NO - requires separate pre-execution source lock")
    print()

    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"06b0 output directory already exists and is non-empty: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    # Required files.
    for path in [
        CASE_ORDER, HARMONIZED_LOCK, PREMODEL_MANIFEST, FOLD_LOCAL_CONTRACT,
        FEATURE_BUDGET_PLAN, RECONSTRUCTION_AUDIT, FREEZE_05E, LOCK_05E,
        CONTRACT_06A, LOCK_06A, TECH_LOCK, TECH_MANIFEST, *A05_FILES,
    ]:
        require(path)
    for info in RAW_INPUTS.values():
        require(info["matrix"])
        require(info["features"])
    for info in HARMONIZED_ANCHORS.values():
        require(info["matrix"])
        require(info["features"])

    case_record = check_case_order()

    if sha256_file(HARMONIZED_LOCK).lower() != EXPECTED_HARMONIZED_LOCK_SHA256:
        raise RuntimeError("04b harmonized_input_lock SHA256 changed.")

    harmonized_lock = load_json(HARMONIZED_LOCK)
    premodel_manifest = load_json(PREMODEL_MANIFEST)
    fold_contract = load_json(FOLD_LOCAL_CONTRACT)

    upstream_record_checks = {
        "04b_harmonized_lock": verify_embedded_records(harmonized_lock, "04b"),
        "04c_premodel_manifest": verify_embedded_records(premodel_manifest, "04c"),
        "05a_lock": verify_embedded_records(load_json(MC05A / "ai_method_contract_lock.json"), "05a"),
        "05e_lock": verify_embedded_records(load_json(LOCK_05E), "05e"),
        "06a_lock": verify_embedded_records(load_json(LOCK_06A), "06a"),
        "04d_technical_lock": verify_embedded_records(load_json(TECH_LOCK), "04d"),
    }

    raw_records: dict[str, Any] = {}
    for modality, info in RAW_INPUTS.items():
        raw_records[modality] = {
            "matrix": matrix_record(info["matrix"], info["expected_shape"]),
            "features": file_record(info["features"]),
            "pre_z_transform": info["pre_z_transform"],
        }

    harmonized_records: dict[str, Any] = {}
    for modality, info in HARMONIZED_ANCHORS.items():
        harmonized_records[modality] = {
            "matrix": matrix_record(info["matrix"], info["expected_shape"]),
            "features": file_record(info["features"]),
            "role": "verification/provenance anchor; not direct held-out-CV preprocessed input",
        }

    budget_rows, primary_budgets = check_feature_budget()
    policy_05a, eligible_models = check_05a_policy()
    freeze_05e = check_05e()
    contract_06a = check_06a()
    tech_lock, tech_manifest = check_technical_preflight()

    # All frozen 05a execution artifacts are hashed now.
    a05_records = {path.name: file_record(path) for path in A05_FILES}

    # The CV leak-prevention rule is not newly invented here; it restates 04c/05a.
    nested_execution_policy = {
        "authoritative_source": rel(MC05A / "model_selection_policy.json"),
        "outer_design": policy_05a["target_nested_selection"]["outer_design"],
        "inner_design": policy_05a["target_nested_selection"]["inner_design"],
        "outer_model_seeds": policy_05a["target_nested_selection"]["outer_model_seeds"],
        "inner_model_seeds": policy_05a["target_nested_selection"]["inner_model_seeds"],
        "outer_test_samples_used_for_selection": False,
        "fold_local_preprocessing_required": True,
        "heldout_cv_direct_use_of_global_harmonized_zscores": "FORBIDDEN",
        "heldout_cv_rule": (
            "For every inner/outer held-out evaluation, feature ranking/selection, missing-value handling, "
            "scaling, PCA or other learned preprocessing parameters must be estimated only from that split's "
            "training partition according to the frozen 04c/05a contracts, then applied to validation/test."
        ),
        "raw_locked_molecular_sources_for_cv": {
            modality: raw_records[modality]["matrix"]["path"] for modality in raw_records
        },
        "global_harmonized_matrices_role": (
            "Locked integrity/reconstruction anchors. They may not substitute for fold-local scaling/preprocessing "
            "during held-out nested-CV evaluation."
        ),
    }

    final_full_data_policy = {
        "authoritative_source": rel(MC05A / "model_selection_policy.json"),
        "selection_rule": policy_05a["final_full_data_method_freeze"]["primary_rule"],
        "minimum_single_method_selection_frequency": 0.4,
        "uncertainty_rule": policy_05a["final_full_data_method_freeze"]["uncertainty_rule"],
        "tie_breaker": policy_05a["final_full_data_method_freeze"]["tie_breaker"],
        "full_data_refit": policy_05a["final_full_data_method_freeze"]["full_data_refit"],
        "all_78_cases_without_outcomes": True,
        "outcome_or_biology_may_change_method": False,
    }

    # Critical scope clarification: the q10 retention rule was calibrated on the frozen SABLE-MO pipeline.
    retention_scope = {
        "primary_SABLE_MO_q10": contract_06a["primary_retention_rule"]["q10_aligned_correlation_min"],
        "historical_SABLE_MO_sensitivity_q10": contract_06a["historical_sensitivity_rule"]["q10_aligned_correlation_min"],
        "applies_to": "SABLE-MO factor stability/retention evaluation on TARGET",
        "cross_engine_transfer": "FORBIDDEN",
        "does_not_override_05a_TARGET_model_selection": True,
        "non_SABLE_engines": (
            "Do not apply the SABLE-MO median/q10/inclusion/modality-support retention thresholds to PCA, "
            "BLOCK_PCA, MOFA, RIDGE_GCCA, iCluster2b, MOVE, or other engines unless a separately prospective "
            "engine-specific calibration exists."
        ),
        "zero_factor_policy": contract_06a["target_application_decision_policy"]["rule_if_primary_retains_zero_factors"],
        "few_factor_policy": contract_06a["target_application_decision_policy"]["rule_if_primary_retains_few_nonzero_factors"],
    }

    technical_boundary = {
        "lock_status": tech_lock["status"],
        "n_cases": tech_manifest["case_cohort"]["n_cases"],
        "resolved_covariates": tech_manifest["resolved_covariates"],
        "technical_covariates_used_for_model_selection": False,
        "automatic_batch_residualization": False,
        "future_audit_timing": tech_manifest["future_factor_audit"]["execution_timing"],
        "interpretation": (
            "The frozen 04d preflight found zero strict usable technical covariates. Representation/model selection "
            "therefore remains independent of technical metadata; later stable-factor technical status is not "
            "automatically assumed biological and may remain 'stable_technical_status_unknown'."
        ),
    }

    contract = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "status": "TARGET_PREFIT_INPUT_AND_POLICY_FROZEN_RUNNER_NOT_YET_AUTHORIZED",
        "scientific_state_at_freeze": {
            "target_molecular_values_summarized_or_interpreted": False,
            "target_outcomes_read": False,
            "target_clinical_values_read": False,
            "biological_annotations_read": False,
            "target_model_fit_performed": False,
            "target_retained_factor_result_viewed": False,
            "runner_source_authorized": False,
        },
        "cohort": case_record,
        "raw_locked_molecular_inputs": raw_records,
        "harmonized_integrity_anchors": harmonized_records,
        "preprocessing_contracts": {
            "04b_harmonized_lock": file_record(HARMONIZED_LOCK),
            "04c_prestandardization_manifest": file_record(PREMODEL_MANIFEST),
            "04c_fold_local_preprocessing_contract": file_record(FOLD_LOCAL_CONTRACT),
            "04c_feature_budget_plan": file_record(FEATURE_BUDGET_PLAN),
            "04c_reconstruction_audit": file_record(RECONSTRUCTION_AUDIT),
            "primary_feature_budgets": primary_budgets,
            "all_budget_rows": budget_rows,
        },
        "05a_frozen_target_execution_artifacts": a05_records,
        "nested_cv_execution_policy": nested_execution_policy,
        "nested_selection_eligible_models": eligible_models,
        "05e_primary_proposed_method": {
            "freeze": file_record(FREEZE_05E),
            "lock": file_record(LOCK_05E),
            "method_id": "V4_SABLE_MO",
            "profile_id": "ARCH_COMPACT__REG_WEAK",
            "role": "primary proposed implementation; does not override frozen 05a outcome-blind TARGET method selection",
        },
        "final_full_data_method_freeze": final_full_data_policy,
        "06a_SABLE_MO_retention_scope": retention_scope,
        "04d_technical_metadata_boundary": {
            "lock": file_record(TECH_LOCK),
            "manifest": file_record(TECH_MANIFEST),
            **technical_boundary,
        },
        "upstream_embedded_hash_audit": upstream_record_checks,
        "forbidden_for_06b0_and_next_prefit_source_lock": [
            "TARGET clinical outcome values",
            "survival/metastasis labels",
            "biological program annotations used for model/factor selection",
            "technical covariates used to select representation family",
            "global full-cohort feature ranking/scaling/PCA inside held-out CV",
            "post-TARGET threshold relaxation or model-shopping",
        ],
        "runner_authorization_policy": {
            "target_fit_authorized_by_06b0_alone": False,
            "next_required_stage": "06b1 pre-execution source/implementation lock",
            "require_exact_runner_source_sha256_before_first_TARGET_fit": True,
            "require_no_scientific_TARGET_result_files_before_06b1": True,
            "reason": (
                "06b0 freezes input identity and scientific execution policy. The actual runner must be written/tested "
                "against synthetic fixtures and then source-hash locked before any real TARGET fit."
            ),
        },
        "rewrite_policy": (
            "06b0 is immutable after creation. Any runner implementation must conform to this input/policy lock. "
            "No TARGET representation, retained-factor, biology, technical-association, or outcome result may be used "
            "to alter folds, feature budgets, preprocessing, eligible-model registry, method-selection rules, "
            "SABLE-MO retention thresholds, or technical-metadata boundaries."
        ),
    }

    CONTRACT_OUT.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    README_OUT.write_text(
        "Paper 5 06b0 - TARGET pre-fit input/policy lock\n"
        "================================================\n\n"
        "This stage performs NO TARGET model fitting and reads NO clinical outcomes or biological annotations.\n"
        "It freezes the exact 78-case molecular-input identities, upstream 04b/04c preprocessing contracts,\n"
        "05a nested-CV assignments/model-selection policy, 05e proposed SABLE-MO implementation, 06a q10\n"
        "retention policy, and the frozen 04d technical-metadata boundary.\n\n"
        "Critical leakage guard:\n"
        "Global harmonized z-score matrices are integrity/provenance anchors only for held-out CV. Nested CV must\n"
        "re-estimate ranking/selection, missing-value handling, scaling, PCA/learned preprocessing only from each\n"
        "training partition, then apply those fitted transformations to validation/test.\n\n"
        "Critical scope guard:\n"
        "The 06a q10=0.60 primary and q10=0.40 historical sensitivity retention rules are SABLE-MO-specific and\n"
        "must not be transferred to other representation engines.\n\n"
        "06b0 does NOT authorize a real TARGET fit. A separate source-hash/implementation lock is required next.\n",
        encoding="utf-8",
    )

    lock = {
        "status": "06B0_TARGET_PREFIT_INPUT_POLICY_LOCK_FROZEN",
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "script_source": file_record(SCRIPT_PATH),
        "contract": file_record(CONTRACT_OUT),
        "readme": file_record(README_OUT),
        "target_fit_performed": False,
        "target_outcomes_read": False,
        "target_clinical_values_read": False,
        "biological_annotations_read": False,
        "runner_source_authorized": False,
        "next_required_stage": "06b1 pre-execution source/implementation lock before first real TARGET fit",
    }
    LOCK_OUT.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("06b0 TARGET pre-fit input/policy lock: FROZEN")
    print("TARGET cases:                         78")
    print("Primary feature budgets:              RNA=5000, METHYLATION=10000, CNV=5000")
    print("Nested CV:                            5x5 outer; 4-fold inner")
    print("Held-out direct global z-score use:   FORBIDDEN")
    print("05e proposed method/profile:          V4_SABLE_MO / ARCH_COMPACT__REG_WEAK")
    print("SABLE-MO q10 primary/sensitivity:     0.60 / 0.40")
    print("Cross-engine q10 transfer:            FORBIDDEN")
    print("04d strict usable technical fields:   0")
    print("TARGET outcomes/clinical read:        NO")
    print("TARGET model fit:                     NO")
    print("Runner authorized:                    NO")
    print(f"Contract: {rel(CONTRACT_OUT)}")
    print(f"Lock:     {rel(LOCK_OUT)}")
    print("Next: write/test the real runner on synthetic fixtures, then freeze its exact source in 06b1.")


if __name__ == "__main__":
    main()
