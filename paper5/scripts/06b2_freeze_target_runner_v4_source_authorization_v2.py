#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_VERSION = "06b2-freeze-target-runner-v4-source-authorization-v2-float32-epsilon-canonicalization-no-cli"
CONTRACT_VERSION = "paper5-06b2-target-runner-v4-source-authorization-v2"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# -------------------------------------------------------------------------
# Historical v3 authorization / source
# -------------------------------------------------------------------------
HISTORICAL_06B1_LOCK = (
    PAPER5_ROOT
    / "method_contract"
    / "06b1"
    / "target_runner_source_authorization_lock.json"
)
EXPECTED_06B1_STATUS = "06B1_TARGET_RUNNER_SOURCE_AUTHORIZED_BEFORE_FIRST_TARGET_FIT"
EXPECTED_V3_SHA256 = "1b40573e91b6340edf8c5a210ef029ee6b584aca99489ce69a7041cce8c0e485"
EXPECTED_V3_VERSION = (
    "06b-run-target-outcome-blind-representation-benchmark-gpu-v3-"
    "missing-view-return-schema-recovery-preauthorization-synthetic-gated-no-cli"
)
V3_RUNNER = (
    PAPER5_ROOT
    / "scripts"
    / "06b_run_target_outcome_blind_representation_benchmark_gpu_v3.py"
)

# -------------------------------------------------------------------------
# v4 source + synthetic PASS
# -------------------------------------------------------------------------
V4_RUNNER = (
    PAPER5_ROOT
    / "scripts"
    / "06b_run_target_outcome_blind_representation_benchmark_gpu_v4.py"
)
EXPECTED_V4_SHA256 = "4f69680a2d26d6c8e4bd5b337a5e94e6cdc36d149ea6a4b9af547b3855a93eab"
EXPECTED_V4_VERSION = (
    "06b-run-target-outcome-blind-representation-benchmark-gpu-v4-"
    "frozen-04c-raw-input-fold-local-preprocessing-recovery-"
    "preauthorization-synthetic-gated-no-cli"
)

V4_SELFTEST = (
    PAPER5_ROOT
    / "method_contract"
    / "06b_preflight"
    / "target_runner_synthetic_selftest_v4.json"
)
EXPECTED_SELFTEST_STATUS = (
    "06B_TARGET_RUNNER_SYNTHETIC_SELFTEST_PASS_BEFORE_TARGET_ACCESS"
)

# -------------------------------------------------------------------------
# Prospective anchors
# -------------------------------------------------------------------------
LOCK_06B0 = PAPER5_ROOT / "method_contract" / "06b0" / "06b0_prefit_lock.json"
AUDIT_06B0A = (
    PAPER5_ROOT
    / "method_contract"
    / "06b0a"
    / "target_runner_dependency_audit.json"
)

EXPECTED_06B0_STATUS = "06B0_TARGET_PREFIT_INPUT_POLICY_LOCK_FROZEN"
EXPECTED_06B0A_STATUS = "READ_ONLY_TARGET_RUNNER_DEPENDENCY_AUDIT_COMPLETE"

# -------------------------------------------------------------------------
# 06b2 output expected by the v4 runner
# -------------------------------------------------------------------------
OUT = PAPER5_ROOT / "method_contract" / "06b2"
LOCK_OUT = OUT / "target_runner_v4_source_authorization_lock.json"
README_OUT = OUT / "06b2_README.txt"
RECOVERY_OUT = OUT / "historical_v3_attempt_boundary_record.json"

AUTH_STATUS = "06B2_TARGET_RUNNER_V4_SOURCE_AUTHORIZED_BEFORE_FIRST_TARGET_FIT"

# No scientific TARGET benchmark file may exist before v4 authorization.
REAL_RESULT_ROOT = (
    PAPER5_ROOT / "results" / "target_representation" / "06b_primary_benchmark"
)
REAL_06B_CONTRACT_ROOT = PAPER5_ROOT / "method_contract" / "06b"


EXPECTED_ELIGIBLE_MODELS = [
    "BLOCK_PCA",
    "MOFA_K5",
    "MOFA_K10",
    "MOFA_K15",
    "MOFA_K20",
    "V0_BALANCED_AE",
    "V1_SHARED_PRIVATE_AE",
    "V2_CONTRASTIVE_SPAE",
    "V3_DROPOUT_SPAE",
    "V4_SABLE_MO",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_path(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")


def load_json(path: Path) -> dict[str, Any]:
    require_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_record(path: Path) -> dict[str, Any]:
    require_file(path)
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def top_level_string_literal(path: Path, variable_name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == variable_name
            ):
                try:
                    value = ast.literal_eval(node.value)
                except Exception:
                    continue
                if isinstance(value, str):
                    return value
    return ""


def files_under(root: Path) -> list[str]:
    if not root.exists():
        return []
    return [
        relative_path(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def verify_v3_source_control_flow() -> dict[str, Any]:
    """
    Verify a narrow code-order fact needed for the historical-boundary record:
    v3 resolves prestandardization paths before np.load of molecular matrices.

    This does NOT prove the console traceback by itself.  It only certifies the
    source control-flow boundary that makes the contemporaneously observed
    path-resolution failure occur before matrix loading.
    """
    require_file(V3_RUNNER)
    source = V3_RUNNER.read_text(encoding="utf-8")

    function_start = source.find("def load_authorized_target_inputs")
    if function_start < 0:
        raise RuntimeError("v3 load_authorized_target_inputs() not found.")

    next_function = source.find("\ndef ", function_start + 5)
    if next_function < 0:
        next_function = len(source)
    body = source[function_start:next_function]

    resolve_pos = body.find("discover_prestandardization_matrix")
    load_pos = body.find("np.load")
    if resolve_pos < 0 or load_pos < 0:
        raise RuntimeError(
            "Could not verify v3 path-resolution-before-np.load control flow."
        )
    if resolve_pos >= load_pos:
        raise RuntimeError(
            "Unexpected v3 control flow: np.load occurs before path resolution."
        )

    return {
        "v3_path_resolution_occurs_before_molecular_np_load": True,
        "verification_method": (
            "static source-order verification inside "
            "load_authorized_target_inputs()"
        ),
        "claim_limit": (
            "This source check does not independently reconstruct the console traceback. "
            "It verifies only that a failure during prestandardization path resolution "
            "occurs before molecular np.load in the authorized v3 source."
        ),
    }


def verify_v4_selftest(selftest: dict[str, Any]) -> None:
    if selftest.get("status") != EXPECTED_SELFTEST_STATUS:
        raise RuntimeError(
            f"v4 synthetic self-test is not PASS: {selftest.get('status')}"
        )

    runner = selftest.get("runner_source", {})
    if str(runner.get("sha256", "")).lower() != EXPECTED_V4_SHA256:
        raise RuntimeError(
            "v4 self-test was not produced by the exact runner being authorized."
        )
    if runner.get("path") != relative_path(V4_RUNNER):
        raise RuntimeError("v4 self-test runner path mismatch.")

    scientific = selftest.get("scientific_state", {})
    required_false = [
        "real_TARGET_values_accessed",
        "TARGET_outcomes_accessed",
        "TARGET_clinical_values_accessed",
        "TARGET_biology_accessed",
        "TARGET_model_fitting_performed",
        "scientific_TARGET_result_generated",
        "real_TARGET_fit_authorized",
    ]
    for key in required_false:
        if scientific.get(key) is not False:
            raise RuntimeError(
                f"v4 synthetic scientific boundary failed: "
                f"{key}={scientific.get(key)!r}"
            )

    if selftest.get("eligible_model_ids") != EXPECTED_ELIGIBLE_MODELS:
        raise RuntimeError("v4 synthetic eligible-model set differs from frozen registry.")

    branch = selftest.get("full_10_branch_synthetic_test", {})
    if branch.get("all_10_model_branches_executed") is not True:
        raise RuntimeError("v4 synthetic test did not execute all 10 model branches.")
    if branch.get("target_values_accessed") is not False:
        raise RuntimeError("v4 synthetic branch unexpectedly accessed TARGET values.")
    if branch.get("scientific_target_results_generated") is not False:
        raise RuntimeError(
            "v4 synthetic branch unexpectedly generated scientific TARGET results."
        )

    preprocessing = selftest.get("v4_04c_fold_local_preprocessing_selftest", {})
    expected_true = [
        "RNA_identity",
        "CNV_log2_copy_number_over_2",
        "MOFA_native_missingness_preserved",
        "BLOCK_PCA_all_training_fold_eligible_features",
    ]
    for key in expected_true:
        if preprocessing.get(key) is not True:
            raise RuntimeError(f"v4 04c preprocessing self-test missing PASS: {key}")

    observed_epsilon = float(
        preprocessing.get("methylation_beta_to_M_epsilon", -1.0)
    )
    expected_epsilon_f32 = struct.unpack(
        "!f",
        struct.pack("!f", 1e-6),
    )[0]
    if observed_epsilon != expected_epsilon_f32:
        raise RuntimeError(
            "v4 methylation epsilon differs from frozen 04c float32 semantics. "
            f"Observed={observed_epsilon!r}; "
            f"expected_float32_1e-6={expected_epsilon_f32!r}"
        )
    if int(preprocessing.get("variance_ddof", -1)) != 1:
        raise RuntimeError("v4 variance ddof differs from frozen 04c semantics.")

    recovery = selftest.get("v4_input_preprocessing_recovery", {})
    if recovery.get("historical_v3_authorization_exists") is not True:
        raise RuntimeError("v4 self-test did not observe historical 06b1 authorization.")
    if recovery.get("scientific_metric_definition_changed_by_v4_input_recovery") is not False:
        raise RuntimeError("v4 recovery unexpectedly changed scientific metric definitions.")
    if recovery.get("TARGET_values_accessed_during_v4_synthetic_recovery") is not False:
        raise RuntimeError("v4 recovery unexpectedly accessed real TARGET values.")
    if recovery.get("MOFA_imputation") != "none_native_missingness_preserved":
        raise RuntimeError("v4 self-test does not certify native-missing MOFA semantics.")


def main() -> None:
    print("=" * 106)
    print("Paper 5 - freeze v4 TARGET runner source authorization")
    print("=" * 106)
    print(f"Script version: {SCRIPT_VERSION}")
    print("TARGET molecular values read in 06b2: NO")
    print("TARGET outcomes/clinical/biology read in 06b2: NO")
    print("TARGET model fit in 06b2: NO")
    print("Scientific TARGET results read in 06b2: NO")
    print("Historical 06b1 rewritten: NO")
    print()

    for path in [
        HISTORICAL_06B1_LOCK,
        V3_RUNNER,
        V4_RUNNER,
        V4_SELFTEST,
        LOCK_06B0,
        AUDIT_06B0A,
    ]:
        require_file(path)

    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(
            f"06b2 output directory already exists and is non-empty: {OUT}"
        )

    # ------------------------------------------------------------------
    # Historical v3 identity
    # ------------------------------------------------------------------
    historical = load_json(HISTORICAL_06B1_LOCK)
    if historical.get("status") != EXPECTED_06B1_STATUS:
        raise RuntimeError(
            f"Historical 06b1 status mismatch: {historical.get('status')}"
        )

    historical_runner = historical.get("runner_source", {})
    if str(historical_runner.get("sha256", "")).lower() != EXPECTED_V3_SHA256:
        raise RuntimeError("Historical 06b1 does not authorize expected v3 SHA.")

    if sha256_file(V3_RUNNER).lower() != EXPECTED_V3_SHA256:
        raise RuntimeError("Local historical v3 source changed after 06b1.")
    if top_level_string_literal(V3_RUNNER, "SCRIPT_VERSION") != EXPECTED_V3_VERSION:
        raise RuntimeError("Local historical v3 SCRIPT_VERSION changed.")

    v3_flow = verify_v3_source_control_flow()

    # ------------------------------------------------------------------
    # Verify no scientific output was materialized by the failed v3 attempt
    # ------------------------------------------------------------------
    existing_result_files = files_under(REAL_RESULT_ROOT)
    existing_contract_files = files_under(REAL_06B_CONTRACT_ROOT)
    if existing_result_files or existing_contract_files:
        raise RuntimeError(
            "Scientific 06b files already exist before v4 authorization.\n"
            f"Result files: {existing_result_files}\n"
            f"Contract files: {existing_contract_files}"
        )

    # ------------------------------------------------------------------
    # v4 exact source + synthetic PASS
    # ------------------------------------------------------------------
    observed_v4_sha = sha256_file(V4_RUNNER)
    if observed_v4_sha.lower() != EXPECTED_V4_SHA256:
        raise RuntimeError(
            "v4 runner source SHA mismatch.\n"
            f"Expected: {EXPECTED_V4_SHA256}\n"
            f"Observed: {observed_v4_sha}"
        )

    observed_v4_version = top_level_string_literal(V4_RUNNER, "SCRIPT_VERSION")
    if observed_v4_version != EXPECTED_V4_VERSION:
        raise RuntimeError(
            "v4 runner SCRIPT_VERSION mismatch.\n"
            f"Expected: {EXPECTED_V4_VERSION}\n"
            f"Observed: {observed_v4_version}"
        )

    selftest = load_json(V4_SELFTEST)
    verify_v4_selftest(selftest)

    lock_06b0 = load_json(LOCK_06B0)
    if lock_06b0.get("status") != EXPECTED_06B0_STATUS:
        raise RuntimeError(f"06b0 status mismatch: {lock_06b0.get('status')}")
    audit_06b0a = load_json(AUDIT_06B0A)
    if audit_06b0a.get("status") != EXPECTED_06B0A_STATUS:
        raise RuntimeError(f"06b0a status mismatch: {audit_06b0a.get('status')}")

    OUT.mkdir(parents=True, exist_ok=False)

    # ------------------------------------------------------------------
    # Historical recovery record: conservative / auditable wording
    # ------------------------------------------------------------------
    recovery = {
        "status": "06B2_HISTORICAL_V3_ATTEMPT_BOUNDARY_RECORDED",
        "created_at_utc": utc_now_iso(),
        "historical_06b1": file_record(HISTORICAL_06B1_LOCK),
        "historical_v3_runner": {
            **file_record(V3_RUNNER),
            "script_version": EXPECTED_V3_VERSION,
        },
        "verifiable_disk_state_before_v4_authorization": {
            "06b_scientific_result_files_present": False,
            "06b_scientific_contract_files_present": False,
        },
        "v3_source_control_flow": v3_flow,
        "contemporaneous_execution_context": {
            "reported_failure_stage": (
                "input path resolution in discover_prestandardization_matrix before "
                "successful TARGET molecular matrix loading/model fitting"
            ),
            "reported_failure_reason": (
                "v3 assumed separate materialized prestandardization NPY matrices "
                "that were not part of frozen 04c outputs"
            ),
            "cryptographic_console_log_available_to_06b2": False,
            "interpretation_limit": (
                "The exact traceback text was observed contemporaneously in the execution "
                "session but is not cryptographically ingested by this zero-argument 06b2 "
                "script. Therefore the lock independently claims only the verifiable disk "
                "state plus v3 source-control-flow boundary."
            ),
        },
        "historical_06b1_rewritten": False,
    }
    RECOVERY_OUT.write_text(
        json.dumps(recovery, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # New v4 authorization expected by the v4 runner
    # ------------------------------------------------------------------
    lock = {
        "status": AUTH_STATUS,
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "authorization_scope": (
            "Authorize exactly the immutable v4 runner source for the next outcome-blind "
            "TARGET representation benchmark after v4 synthetic full-path validation of "
            "the frozen 04c raw-to-model transformations and fold-local preprocessing."
        ),
        "runner_source": {
            **file_record(V4_RUNNER),
            "script_version": observed_v4_version,
        },
        "synthetic_selftest": {
            **file_record(V4_SELFTEST),
            "status": selftest["status"],
            "all_10_model_branches_executed": True,
            "exact_04c_fold_local_preprocessing_tested": True,
            "real_TARGET_values_accessed": False,
            "TARGET_model_fitting_performed": False,
        },
        "historical_recovery_record": file_record(RECOVERY_OUT),
        "historical_06b1": {
            **file_record(HISTORICAL_06B1_LOCK),
            "rewritten": False,
            "authorization_scope": "historical exact v3 source only",
        },
        "prospective_anchors": {
            "06b0_prefit_lock": file_record(LOCK_06B0),
            "06b0a_dependency_audit": file_record(AUDIT_06B0A),
        },
        "preauthorization_scientific_output_check": {
            "06b_primary_benchmark_files_present": False,
            "06b_method_contract_files_present": False,
        },
        "real_target_fit_authorized": True,
        "authorization_conditions": {
            "exact_runner_sha_required": EXPECTED_V4_SHA256,
            "runner_may_change_after_authorization": False,
            "TARGET_outcome_access": False,
            "TARGET_clinical_access": False,
            "biological_annotation_access_for_selection": False,
            "technical_covariates_used_for_model_selection": False,
            "q10_060_used_for_cross_family_selection": False,
            "q10_040_used_for_cross_family_selection": False,
            "secondary_comparators_used_for_primary_selection": False,
            "global_full_cohort_rankings_used_in_heldout_CV": False,
            "global_harmonized_zscores_used_as_heldout_CV_input": False,
            "fold_local_feature_ranking_required": True,
            "fold_local_imputation_scaling_PCA_required_where_applicable": True,
            "MOFA_native_missingness_preserved": True,
            "BLOCK_PCA_all_fold_local_eligible_features": True,
            "neural_primary_feature_budgets": {
                "RNA": 5000,
                "METHYLATION": 10000,
                "CNV": 5000,
            },
            "threshold_or_model_retuning_after_TARGET_results": False,
        },
        "rewrite_policy": (
            "06b2 is immutable after creation. Any change to the v4 runner source, "
            "its SHA-256, raw-to-model transformations, fold-local preprocessing, "
            "eligible-model registry, utility definition, one-SE rule, final 40% rule, "
            "or scientific access boundary invalidates this authorization and requires "
            "a new versioned pre-fit authorization before another real TARGET fit."
        ),
    }
    LOCK_OUT.write_text(
        json.dumps(lock, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    README_OUT.write_text(
        "Paper 5 06b2 - v4 TARGET runner source authorization\n"
        "====================================================\n\n"
        f"Authorized v4 runner: {relative_path(V4_RUNNER)}\n"
        f"Runner SHA-256: {observed_v4_sha}\n"
        f"Runner SCRIPT_VERSION: {observed_v4_version}\n\n"
        "Historical 06b1 remains immutable and authorizes v3 only.\n"
        "The v3 attempt produced no frozen scientific 06b result/contract files.\n"
        "v4 then passed a new synthetic-only full-path test including the exact frozen\n"
        "04c raw->model transformations, fold-local ranking, BLOCK_PCA all-eligible\n"
        "feature semantics, native-missing MOFA preprocessing, neural primary budgets,\n"
        "and all 10 nested-selection branches.\n"
"The methylation epsilon check is canonicalized to IEEE-754 float32, matching\n"
"04c METHYLATION_EPSILON = np.float32(1e-6).\n\n"
        "06b2 itself reads no TARGET molecular values, outcomes, clinical values, or biology\n"
        "and performs no model fitting. Only the exact authorized v4 runner may now begin\n"
        "the next outcome-blind real TARGET benchmark.\n",
        encoding="utf-8",
    )

    print("Historical 06b1 identity:                  PASS")
    print("Historical 06b1 rewritten:                 NO")
    print("v3 source path-resolution-before-np.load:  VERIFIED")
    print("Existing scientific 06b result files:      NO")
    print("Existing scientific 06b contract files:    NO")
    print("Exact v4 runner SHA-256:                    PASS")
    print("Exact v4 runner SCRIPT_VERSION:             PASS")
    print("v4 synthetic 10-model full path:            PASS")
    print("v4 exact 04c/fold-local preprocessing test: PASS")
    print("04c methylation epsilon float32 canonicalization: PASS")
    print("Real TARGET values used by v4 self-test:    NO")
    print("TARGET values read in 06b2:                 NO")
    print("TARGET model fit in 06b2:                   NO")
    print("q10 cross-family selection:                 FORBIDDEN")
    print("=" * 106)
    print("Paper 5 06b2 v4 TARGET runner source authorization: FROZEN")
    print("=" * 106)
    print(f"Authorized v4 SHA: {observed_v4_sha}")
    print(f"Recovery record: {relative_path(RECOVERY_OUT)}")
    print(f"Authorization lock: {relative_path(LOCK_OUT)}")
    print("Next: rerun the exact same v4 runner unchanged.")
    print("=" * 106)


if __name__ == "__main__":
    main()
