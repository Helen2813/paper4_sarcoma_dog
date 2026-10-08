#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_VERSION = "06b1-freeze-target-runner-source-authorization-v1-no-cli"
CONTRACT_VERSION = "paper5-06b1-target-runner-source-authorization-v1"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

RUNNER = PAPER5_ROOT / "scripts" / "06b_run_target_outcome_blind_representation_benchmark_gpu_v3.py"
EXPECTED_RUNNER_SHA256 = "1b40573e91b6340edf8c5a210ef029ee6b584aca99489ce69a7041cce8c0e485"
EXPECTED_RUNNER_VERSION = (
    "06b-run-target-outcome-blind-representation-benchmark-gpu-v3-"
    "missing-view-return-schema-recovery-preauthorization-synthetic-gated-no-cli"
)

SELFTEST = (
    PAPER5_ROOT
    / "method_contract"
    / "06b_preflight"
    / "target_runner_synthetic_selftest_v3.json"
)
EXPECTED_SELFTEST_STATUS = (
    "06B_TARGET_RUNNER_SYNTHETIC_SELFTEST_PASS_BEFORE_TARGET_ACCESS"
)

LOCK_06B0 = PAPER5_ROOT / "method_contract" / "06b0" / "06b0_prefit_lock.json"
AUDIT_06B0A = (
    PAPER5_ROOT
    / "method_contract"
    / "06b0a"
    / "target_runner_dependency_audit.json"
)

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "06b1"
OUTPUT_LOCK = OUTPUT_DIR / "target_runner_source_authorization_lock.json"
OUTPUT_README = OUTPUT_DIR / "06b1_README.txt"

# These locations must still contain no scientific TARGET result files at authorization time.
REAL_RESULT_ROOT = (
    PAPER5_ROOT / "results" / "target_representation" / "06b_primary_benchmark"
)
REAL_06B_CONTRACT_DIR = PAPER5_ROOT / "method_contract" / "06b"

EXPECTED_06B0_STATUS = "06B0_TARGET_PREFIT_INPUT_POLICY_LOCK_FROZEN"
EXPECTED_06B0A_STATUS = "READ_ONLY_TARGET_RUNNER_DEPENDENCY_AUDIT_COMPLETE"
AUTH_STATUS = "06B1_TARGET_RUNNER_SOURCE_AUTHORIZED_BEFORE_FIRST_TARGET_FIT"


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
    import ast

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


def list_files_if_any(root: Path) -> list[str]:
    if not root.exists():
        return []
    return [
        relative_path(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]


def main() -> None:
    print("=" * 104)
    print("Paper 5 - freeze exact TARGET runner source authorization")
    print("=" * 104)
    print(f"Script version: {SCRIPT_VERSION}")
    print("TARGET molecular values read: NO")
    print("TARGET outcomes/clinical/biology read: NO")
    print("TARGET model fit in 06b1: NO")
    print("Scientific TARGET results read: NO")
    print()

    for path in [RUNNER, SELFTEST, LOCK_06B0, AUDIT_06B0A]:
        require_file(path)

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(
            f"06b1 output directory already exists and is non-empty: {OUTPUT_DIR}"
        )

    runner_sha = sha256_file(RUNNER)
    if runner_sha.lower() != EXPECTED_RUNNER_SHA256:
        raise RuntimeError(
            "Runner source SHA256 mismatch.\n"
            f"Expected: {EXPECTED_RUNNER_SHA256}\n"
            f"Observed: {runner_sha}"
        )

    runner_version = top_level_string_literal(RUNNER, "SCRIPT_VERSION")
    if runner_version != EXPECTED_RUNNER_VERSION:
        raise RuntimeError(
            "Runner SCRIPT_VERSION mismatch.\n"
            f"Expected: {EXPECTED_RUNNER_VERSION}\n"
            f"Observed: {runner_version}"
        )

    selftest = load_json(SELFTEST)
    if selftest.get("status") != EXPECTED_SELFTEST_STATUS:
        raise RuntimeError(
            f"Synthetic self-test is not PASS: {selftest.get('status')}"
        )

    selftest_runner = selftest.get("runner_source", {})
    if str(selftest_runner.get("sha256", "")).lower() != EXPECTED_RUNNER_SHA256:
        raise RuntimeError(
            "Synthetic self-test was not produced by the exact runner being authorized."
        )
    if selftest_runner.get("path") != relative_path(RUNNER):
        raise RuntimeError(
            "Synthetic self-test runner path differs from the exact runner path."
        )

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
                f"Synthetic self-test scientific boundary not preserved: {key}={scientific.get(key)!r}"
            )

    if selftest.get("eligible_model_ids") != [
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
    ]:
        raise RuntimeError("Synthetic self-test eligible-model registry differs from frozen 10-model set.")

    branch = selftest.get("full_10_branch_synthetic_test", {})
    if branch.get("all_10_model_branches_executed") is not True:
        raise RuntimeError("Synthetic self-test did not execute all 10 frozen model branches.")
    if branch.get("target_values_accessed") is not False:
        raise RuntimeError("Synthetic branch test unexpectedly accessed TARGET values.")
    if branch.get("scientific_target_results_generated") is not False:
        raise RuntimeError("Synthetic branch test unexpectedly generated scientific TARGET results.")

    recovery = selftest.get("v3_adapter_recovery", {})
    if recovery.get("prediction_values_modified") is not False:
        raise RuntimeError("v3 adapter recovery unexpectedly modified prediction values.")
    if recovery.get("scientific_metric_definition_changed") is not False:
        raise RuntimeError("v3 adapter recovery unexpectedly changed a scientific metric.")
    if recovery.get("TARGET_values_accessed_during_recovery") is not False:
        raise RuntimeError("v3 adapter recovery unexpectedly accessed TARGET values.")

    lock_06b0 = load_json(LOCK_06B0)
    if lock_06b0.get("status") != EXPECTED_06B0_STATUS:
        raise RuntimeError(f"06b0 lock has unexpected status: {lock_06b0.get('status')}")
    if lock_06b0.get("target_fit_performed") is not False:
        raise RuntimeError("06b0 indicates that a TARGET fit was already performed.")
    if lock_06b0.get("runner_source_authorized") is not False:
        raise RuntimeError("06b0 unexpectedly says a runner was already authorized.")

    audit_06b0a = load_json(AUDIT_06B0A)
    if audit_06b0a.get("status") != EXPECTED_06B0A_STATUS:
        raise RuntimeError(
            f"06b0a dependency audit has unexpected status: {audit_06b0a.get('status')}"
        )

    # Strong pre-first-fit guard: no scientific 06b output may already exist.
    result_files = list_files_if_any(REAL_RESULT_ROOT)
    contract_files = list_files_if_any(REAL_06B_CONTRACT_DIR)
    if result_files or contract_files:
        raise RuntimeError(
            "Scientific 06b TARGET outputs already exist before authorization.\n"
            f"Result files: {result_files}\n"
            f"Contract files: {contract_files}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    selftest_record = file_record(SELFTEST)
    runner_record = file_record(RUNNER)

    lock = {
        "status": AUTH_STATUS,
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "authorization_scope": (
            "Authorize exactly one immutable runner source for the first outcome-blind "
            "TARGET representation benchmark. Authorization does not permit outcome, clinical, "
            "biological-annotation, technical-covariate-driven, q10-driven cross-family, or "
            "secondary-comparator-driven model selection."
        ),
        "runner_source": {
            **runner_record,
            "script_version": runner_version,
        },
        "synthetic_selftest": {
            **selftest_record,
            "status": selftest["status"],
            "all_10_model_branches_executed": True,
            "real_TARGET_values_accessed": False,
            "TARGET_model_fitting_performed": False,
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
            "exact_runner_sha_required": EXPECTED_RUNNER_SHA256,
            "runner_may_change_after_authorization": False,
            "TARGET_outcome_access": False,
            "TARGET_clinical_access": False,
            "biological_annotation_access_for_selection": False,
            "technical_covariates_used_for_model_selection": False,
            "q10_060_used_for_cross_family_selection": False,
            "q10_040_used_for_cross_family_selection": False,
            "secondary_comparators_used_for_primary_selection": False,
            "threshold_or_model_retuning_after_TARGET_results": False,
        },
        "rewrite_policy": (
            "06b1 is immutable after creation. Any change to the authorized runner source, "
            "its SHA-256, frozen model registry, fold assignments, preprocessing, utility, "
            "one-SE rule, final 40% rule, or scientific execution boundary invalidates this "
            "authorization and requires a new versioned pre-fit authorization before any "
            "new real TARGET fit."
        ),
    }

    OUTPUT_LOCK.write_text(
        json.dumps(lock, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    OUTPUT_README.write_text(
        "Paper 5 06b1 - exact TARGET runner source authorization\n"
        "=======================================================\n\n"
        f"Authorized runner: {relative_path(RUNNER)}\n"
        f"Runner SHA-256: {runner_sha}\n"
        f"Runner SCRIPT_VERSION: {runner_version}\n\n"
        "The exact runner completed a synthetic-only full-path self-test across all 10 frozen\n"
        "primary nested-selection branches before any real TARGET value was accessed.\n\n"
        "06b1 itself does not read TARGET molecular values, outcomes, clinical values, or biology,\n"
        "and performs no model fitting. After this lock exists, the exact authorized runner may\n"
        "begin the first outcome-blind real TARGET representation benchmark.\n\n"
        "q10=0.60/0.40 are NOT authorized for cross-family model selection; those thresholds remain\n"
        "SABLE-MO-specific post-selection retention logic.\n",
        encoding="utf-8",
    )

    print("Exact runner SHA-256:                 PASS")
    print("Exact runner SCRIPT_VERSION:          PASS")
    print("Synthetic full 10-model self-test:    PASS")
    print("Real TARGET values used by self-test: NO")
    print("Existing scientific 06b results:      NO")
    print("TARGET values read in 06b1:           NO")
    print("TARGET model fit in 06b1:             NO")
    print("q10 cross-family selection:           FORBIDDEN")
    print("=" * 104)
    print("Paper 5 06b1 TARGET runner source authorization: FROZEN")
    print("=" * 104)
    print(f"Authorized runner SHA: {runner_sha}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("Real TARGET fit is now authorized ONLY for this exact runner source.")
    print("Next command is to rerun the same v3 runner unchanged.")
    print("=" * 104)


if __name__ == "__main__":
    main()
