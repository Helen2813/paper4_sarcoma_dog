#!/usr/bin/env python3
"""
Paper 5 - freeze TARGET V1 full-data refit contract.

This stage DOES NOT:
- read TARGET clinical outcomes;
- read TARGET biological annotations;
- fit any TARGET model;
- apply SABLE-MO q10 retention to V1;
- modify the completed 06b benchmark.

It reads only frozen 06b outcome-blind benchmark artifacts and frozen
preprocessing/profile contracts, verifies their identity/consistency, and freezes
the deterministic full-data V1 refit protocol before the first all-78 refit.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "06c-freeze-target-v1-full-data-refit-contract-v1-no-cli"

EXPECTED = {
    "06b_summary_sha256": "c2b350f8d0e957cf1a1005d0aa5f490d631a30d9e6607c12a90659c8037ca2e5",
    "06b_selection_sha256": "8d6f17311cc98c3312619d3410f804e6581f152ce7811616b87ab9774a3e68c1",
    "06b_outer_sha256": "b3c04b163392df043f78c8d377ff9f802afea2d7e95dcbff21fc5c87e504f2c5",
    "06b_lock_sha256": "2730c783f1a74f7aa58030d1ee03f324f1d50c4830c72e4229a15e41b60dc121",
    "06b_runner_sha256": "4f69680a2d26d6c8e4bd5b337a5e94e6cdc36d149ea6a4b9af547b3855a93eab",
    "primary_model_id": "V1_SHARED_PRIVATE_AE",
    "primary_selection_count": 17,
    "secondary_model_id": "V0_BALANCED_AE",
    "secondary_selection_count": 8,
    "n_outer_splits": 25,
    "n_outer_rows": 700,
    "n_models": 10,
    "outer_model_seeds": [1103, 2207, 3301],
    "primary_seed": 1103,
    "reproducibility_seeds": [2207, 3301],
    "n_cases": 78,
    "profile_id": "ARCH_COMPACT__REG_WEAK",
    "fixed_epochs": 400,
    "feature_budgets": {
        "RNA": 5000,
        "METHYLATION": 10000,
        "CNV": 5000,
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with require_file(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} mismatch. Expected {expected!r}, observed {actual!r}")


def require_hash(label: str, path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA256 mismatch.\n"
            f"Expected: {expected}\n"
            f"Observed: {actual}\n"
            f"Path: {path}"
        )
    return actual


def relative_to_paper5(path: Path, paper5_root: Path) -> str:
    return path.resolve().relative_to(paper5_root.resolve()).as_posix()


def main() -> None:
    print("=" * 104)
    print("Paper 5 - freeze TARGET V1 full-data refit contract")
    print("=" * 104)
    print("Script version:", SCRIPT_VERSION)
    print("TARGET clinical outcomes read in 06c: NO")
    print("TARGET biological annotations read in 06c: NO")
    print("TARGET full-data model fit in 06c: NO")
    print("SABLE-MO q10 retention applied to V1 in 06c: NO")
    print("Historical/completed 06b rewritten: NO")

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    paper5 = project_root / "paper5"

    summary_path = (
        paper5
        / "results"
        / "target_representation"
        / "06b_primary_benchmark"
        / "target_outcome_blind_benchmark_summary.json"
    )
    selection_path = (
        paper5
        / "results"
        / "target_representation"
        / "06b_primary_benchmark"
        / "target_inner_one_se_selection_by_outer_split.csv"
    )
    outer_path = (
        paper5
        / "results"
        / "target_representation"
        / "06b_primary_benchmark"
        / "target_outer_fixed_model_benchmark.csv"
    )
    lock06b_path = (
        paper5
        / "method_contract"
        / "06b"
        / "target_outcome_blind_benchmark_results_lock.json"
    )
    policy_path = paper5 / "method_contract" / "05a" / "model_selection_policy.json"
    profile_path = paper5 / "method_contract" / "05d3" / "selected_neural_profile.json"
    prestd_manifest_path = (
        paper5
        / "model_inputs"
        / "TARGET_OS_primary78"
        / "premodel_qc"
        / "prestandardization_variance_audit_manifest.json"
    )
    preprocessing_contract_path = (
        paper5
        / "model_inputs"
        / "TARGET_OS_primary78"
        / "premodel_qc"
        / "fold_local_preprocessing_contract.json"
    )
    lock06b0_path = paper5 / "method_contract" / "06b0" / "06b0_prefit_lock.json"
    lock06b2_path = (
        paper5
        / "method_contract"
        / "06b2"
        / "target_runner_v4_source_authorization_lock.json"
    )

    for p in [
        summary_path,
        selection_path,
        outer_path,
        lock06b_path,
        policy_path,
        profile_path,
        prestd_manifest_path,
        preprocessing_contract_path,
        lock06b0_path,
        lock06b2_path,
    ]:
        require_file(p)

    # Exact completed-06b identity.
    observed_hashes = {
        "06b_summary": require_hash(
            "06b summary", summary_path, EXPECTED["06b_summary_sha256"]
        ),
        "06b_selection": require_hash(
            "06b selection", selection_path, EXPECTED["06b_selection_sha256"]
        ),
        "06b_outer": require_hash(
            "06b outer benchmark", outer_path, EXPECTED["06b_outer_sha256"]
        ),
        "06b_lock": require_hash(
            "06b results lock", lock06b_path, EXPECTED["06b_lock_sha256"]
        ),
    }

    summary = load_json(summary_path)
    lock06b = load_json(lock06b_path)
    policy = load_json(policy_path)
    profile = load_json(profile_path)
    prestd_manifest = load_json(prestd_manifest_path)
    preprocessing_contract = load_json(preprocessing_contract_path)
    lock06b0 = load_json(lock06b0_path)
    lock06b2 = load_json(lock06b2_path)

    require_equal(
        "06b summary status",
        summary.get("status"),
        "06B_TARGET_OUTCOME_BLIND_REPRESENTATION_BENCHMARK_COMPLETE",
    )
    require_equal(
        "06b lock status",
        lock06b.get("status"),
        "06B_TARGET_OUTCOME_BLIND_REPRESENTATION_BENCHMARK_RESULTS_FROZEN",
    )
    require_equal(
        "06b locked runner SHA",
        str(lock06b.get("runner_source", {}).get("sha256", "")).lower(),
        EXPECTED["06b_runner_sha256"],
    )

    freeze = summary.get("final_full_data_method_freeze", {})
    require_equal(
        "06b final method-freeze status",
        freeze.get("status"),
        "UNIQUE_PRIMARY_REPRESENTATION",
    )
    require_equal(
        "06b primary model ids",
        freeze.get("primary_model_ids"),
        [EXPECTED["primary_model_id"]],
    )
    top_frequency = float(freeze.get("top_selection_frequency"))
    if not math.isclose(top_frequency, 17 / 25, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError(
            f"Unexpected top selection frequency: {top_frequency}; expected {17/25}"
        )

    selection_counts = freeze.get("selection_counts", {})
    require_equal(
        "V1 selection count",
        int(selection_counts.get(EXPECTED["primary_model_id"], -1)),
        EXPECTED["primary_selection_count"],
    )
    require_equal(
        "V0 selection count",
        int(selection_counts.get(EXPECTED["secondary_model_id"], -1)),
        EXPECTED["secondary_selection_count"],
    )
    for model_id, count in selection_counts.items():
        if model_id not in {
            EXPECTED["primary_model_id"],
            EXPECTED["secondary_model_id"],
        } and int(count) != 0:
            raise RuntimeError(f"Unexpected nonzero selection count: {model_id}={count}")

    guardrails = summary.get("guardrails", {})
    for key in [
        "biology_accessed",
        "clinical_values_accessed",
        "outcomes_accessed",
        "q10_040_used_for_cross_family_selection",
        "q10_060_used_for_cross_family_selection",
        "secondary_comparators_used_for_primary_selection",
        "technical_covariates_used_for_selection",
    ]:
        if as_bool(guardrails.get(key, False)):
            raise RuntimeError(f"06b guardrail violation: {key}={guardrails.get(key)!r}")

    # Selection table integrity.
    selection_rows = load_csv(selection_path)
    require_equal("selection rows", len(selection_rows), EXPECTED["n_outer_splits"])
    split_ids = {r["split_id"] for r in selection_rows}
    require_equal("selection unique splits", len(split_ids), EXPECTED["n_outer_splits"])

    observed_selected_counts = Counter(r["selected_model_id"] for r in selection_rows)
    require_equal(
        "selection-table V1 count",
        observed_selected_counts[EXPECTED["primary_model_id"]],
        EXPECTED["primary_selection_count"],
    )
    require_equal(
        "selection-table V0 count",
        observed_selected_counts[EXPECTED["secondary_model_id"]],
        EXPECTED["secondary_selection_count"],
    )

    # Outer benchmark integrity and V1 epoch/seed audit.
    outer_rows = load_csv(outer_path)
    require_equal("outer rows", len(outer_rows), EXPECTED["n_outer_rows"])
    require_equal(
        "outer unique splits",
        len({r["split_id"] for r in outer_rows}),
        EXPECTED["n_outer_splits"],
    )
    require_equal(
        "outer unique models",
        len({r["model_id"] for r in outer_rows}),
        EXPECTED["n_models"],
    )
    failed_rows = [r for r in outer_rows if as_bool(r.get("failed", False))]
    require_equal("outer failed rows", len(failed_rows), 0)

    v1_rows = [r for r in outer_rows if r["model_id"] == EXPECTED["primary_model_id"]]
    require_equal("V1 outer rows", len(v1_rows), 75)
    require_equal(
        "V1 unique splits",
        len({r["split_id"] for r in v1_rows}),
        EXPECTED["n_outer_splits"],
    )

    seed_counts: Counter[int] = Counter(int(r["seed"]) for r in v1_rows)
    require_equal(
        "V1 outer seed set",
        sorted(seed_counts),
        EXPECTED["outer_model_seeds"],
    )
    for seed in EXPECTED["outer_model_seeds"]:
        require_equal(f"V1 rows for seed {seed}", seed_counts[seed], 25)

    selected_epochs: list[int] = []
    epochs_by_seed: dict[int, list[int]] = defaultdict(list)
    for r in v1_rows:
        e_float = float(r["selected_epoch"])
        if not e_float.is_integer():
            raise RuntimeError(f"Non-integer selected_epoch in V1 row: {e_float}")
        e = int(e_float)
        selected_epochs.append(e)
        epochs_by_seed[int(r["seed"])].append(e)

    epoch_summary = {
        "n": len(selected_epochs),
        "min": min(selected_epochs),
        "median": statistics.median(selected_epochs),
        "mean": statistics.mean(selected_epochs),
        "max": max(selected_epochs),
        "by_seed": {
            str(seed): {
                "n": len(values),
                "min": min(values),
                "median": statistics.median(values),
                "mean": statistics.mean(values),
                "max": max(values),
            }
            for seed, values in sorted(epochs_by_seed.items())
        },
    }

    # Frozen policy/profile verification.
    nested = policy.get("target_nested_selection", {})
    require_equal(
        "frozen outer model seeds",
        [int(x) for x in nested.get("outer_model_seeds", [])],
        EXPECTED["outer_model_seeds"],
    )

    full_data_policy = policy.get("final_full_data_method_freeze", {})
    if "all 78 cases without outcomes" not in str(full_data_policy.get("full_data_refit", "")):
        raise RuntimeError("Frozen 05a full-data refit rule no longer states all 78 cases without outcomes.")
    require_equal(
        "minimum single-method selection frequency",
        float(full_data_policy.get("minimum_single_method_selection_frequency")),
        0.4,
    )

    selected_profile = profile.get("selected_profile", {})
    require_equal(
        "frozen neural profile id",
        selected_profile.get("profile_id"),
        EXPECTED["profile_id"],
    )
    parameters = selected_profile.get("parameters", {})
    require_equal(
        "frozen maximum epochs",
        int(parameters.get("maximum_epochs")),
        EXPECTED["fixed_epochs"],
    )
    require_equal(
        "shared latent dimension",
        int(parameters.get("shared_latent_dim")),
        5,
    )
    require_equal(
        "private latent dimension per view",
        int(parameters.get("private_latent_dim_per_view")),
        2,
    )
    require_equal(
        "PCA components",
        {k: int(v) for k, v in parameters.get("pca_components", {}).items()},
        {"CNV": 10, "METHYLATION": 10, "RNA": 10},
    )

    budget_contract = prestd_manifest.get("budget_contract", {})
    observed_budgets = {
        "RNA": int(budget_contract["RNA"]["primary"]),
        "METHYLATION": int(budget_contract["METHYLATION"]["primary"]),
        "CNV": int(budget_contract["CNV"]["primary"]),
    }
    require_equal("primary feature budgets", observed_budgets, EXPECTED["feature_budgets"])
    require_equal(
        "TARGET case count in preprocessing manifest",
        int(prestd_manifest.get("case_order", {}).get("n_cases")),
        EXPECTED["n_cases"],
    )

    allowed_for = preprocessing_contract.get("global_full_cohort_rankings", {}).get(
        "allowed_for", []
    )
    if not any("final outcome-blind refit" in str(x).lower() for x in allowed_for):
        raise RuntimeError(
            "04c preprocessing contract does not explicitly allow global full-cohort rankings "
            "for the final outcome-blind refit."
        )

    # The V1 outer epoch distribution is recorded as evidence only.
    # It does NOT tune final epochs. The full-data epoch count is inherited directly
    # from the already-frozen profile maximum_epochs=400.
    require_equal("V1 outer epoch median", int(epoch_summary["median"]), 400)

    out_dir = paper5 / "method_contract" / "06c"
    if out_dir.exists():
        existing = list(out_dir.iterdir())
        if existing:
            raise RuntimeError(
                f"06c output directory already exists and is non-empty: {out_dir}\n"
                "Do not overwrite a frozen full-data refit contract."
            )
    else:
        out_dir.mkdir(parents=True, exist_ok=False)

    created_at = datetime.now(timezone.utc).isoformat()

    contract = {
        "analysis_role": "POST_06B_PRE_FULL_DATA_REFIT_METHOD_FREEZE",
        "contract_version": "paper5-06c-target-v1-full-data-refit-v1",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": created_at,
        "status": "FROZEN_BEFORE_TARGET_FULL_DATA_REFIT",
        "selection_result": {
            "final_status": "UNIQUE_PRIMARY_REPRESENTATION",
            "primary_model_id": EXPECTED["primary_model_id"],
            "selection_count": EXPECTED["primary_selection_count"],
            "selection_denominator": EXPECTED["n_outer_splits"],
            "selection_frequency": top_frequency,
            "runner_sha256": EXPECTED["06b_runner_sha256"],
            "outer_failed_rows": 0,
        },
        "full_data_refit": {
            "model_id": EXPECTED["primary_model_id"],
            "model_family": "SABLE_MO_ablation",
            "role": "PRIMARY_TARGET_FULL_DATA_REPRESENTATION",
            "n_cases": EXPECTED["n_cases"],
            "outcome_access": False,
            "clinical_access": False,
            "biological_annotation_access": False,
            "technical_covariates_used_for_fit": False,
            "feature_budget": observed_budgets,
            "feature_ranking_scope": "full 78-case molecular cohort; explicitly allowed by frozen 04c only after model family/budget freeze",
            "imputation_scope": "fit on all 78 molecular cases for the final outcome-blind refit",
            "scaling_scope": "fit on all 78 molecular cases for the final outcome-blind refit",
            "pca_scope": "fit on all 78 molecular cases for the final outcome-blind refit",
            "profile_id": EXPECTED["profile_id"],
            "profile_parameters": parameters,
            "training": {
                "primary_seed": EXPECTED["primary_seed"],
                "primary_seed_rule": "Use the first seed in the already-frozen 05a outer_model_seeds list. The seed is frozen here before any full-data fit and is not selected using full-data fit quality, biology, clinical variables, or outcomes.",
                "reproducibility_seeds": EXPECTED["reproducibility_seeds"],
                "reproducibility_seed_role": "MANDATORY_REPRODUCIBILITY_ONLY; never replace the primary seed based on downstream results.",
                "fixed_epochs": EXPECTED["fixed_epochs"],
                "fixed_epoch_provenance": "Inherited directly from frozen ARCH_COMPACT__REG_WEAK maximum_epochs=400. The observed 06b V1 outer selected-epoch distribution is recorded only as consistency evidence and does not retune this value.",
                "early_stopping_on_full_78": False,
                "post_fit_seed_selection": False,
                "post_fit_epoch_selection": False,
            },
            "representation_output": {
                "primary_seed_representation": "Shared latent representation from V1_SHARED_PRIVATE_AE seed 1103.",
                "reproducibility_seed_representations": [
                    "Shared latent representation from V1_SHARED_PRIVATE_AE seed 2207.",
                    "Shared latent representation from V1_SHARED_PRIVATE_AE seed 3301.",
                ],
                "seed_ensemble_or_averaging": "FORBIDDEN unless separately prospectively frozen; latent coordinates are not averaged ad hoc.",
            },
        },
        "v1_outer_training_evidence": {
            "role": "OUTCOME_BLIND_CONSISTENCY_EVIDENCE_ONLY_NOT_NEW_TUNING",
            "rows": len(v1_rows),
            "splits": EXPECTED["n_outer_splits"],
            "seeds": EXPECTED["outer_model_seeds"],
            "selected_epoch_summary": epoch_summary,
        },
        "sable_retention_boundary": {
            "q10_060_applied_to_v1": False,
            "q10_040_applied_to_v1": False,
            "sable_mo_stability_gate_applied_to_v1": False,
            "reason": "V1_SHARED_PRIVATE_AE is the frozen TARGET-selected representation. SABLE-MO q10/stability retention rules are engine-specific and cross-engine transfer was forbidden before TARGET selection.",
            "06a_q10_contract_rewritten": False,
            "interpretation": "The historical/prospective q10=.40/.60 SABLE-MO findings remain valid for SABLE-MO but do not define V1 factor retention.",
        },
        "decision_firewall": {
            "outcomes_may_change_primary_model": False,
            "biology_may_change_primary_model": False,
            "clinical_variables_may_change_primary_model": False,
            "full_data_fit_quality_may_change_primary_seed": False,
            "full_data_fit_quality_may_change_epoch": False,
            "full_data_fit_quality_may_promote_v4_sable_mo": False,
            "full_data_fit_quality_may_reopen_06b": False,
        },
        "source_evidence": {
            "06b_summary": {
                "path": relative_to_paper5(summary_path, paper5),
                "sha256": observed_hashes["06b_summary"],
            },
            "06b_selection": {
                "path": relative_to_paper5(selection_path, paper5),
                "sha256": observed_hashes["06b_selection"],
            },
            "06b_outer_benchmark": {
                "path": relative_to_paper5(outer_path, paper5),
                "sha256": observed_hashes["06b_outer"],
            },
            "06b_results_lock": {
                "path": relative_to_paper5(lock06b_path, paper5),
                "sha256": observed_hashes["06b_lock"],
            },
            "05a_model_selection_policy": {
                "path": relative_to_paper5(policy_path, paper5),
                "sha256": sha256_file(policy_path),
            },
            "05d3_selected_neural_profile": {
                "path": relative_to_paper5(profile_path, paper5),
                "sha256": sha256_file(profile_path),
            },
            "04c_prestandardization_manifest": {
                "path": relative_to_paper5(prestd_manifest_path, paper5),
                "sha256": sha256_file(prestd_manifest_path),
            },
            "04c_fold_local_preprocessing_contract": {
                "path": relative_to_paper5(preprocessing_contract_path, paper5),
                "sha256": sha256_file(preprocessing_contract_path),
            },
            "06b0_prefit_lock": {
                "path": relative_to_paper5(lock06b0_path, paper5),
                "sha256": sha256_file(lock06b0_path),
                "status": lock06b0.get("status"),
            },
            "06b2_runner_authorization_lock": {
                "path": relative_to_paper5(lock06b2_path, paper5),
                "sha256": sha256_file(lock06b2_path),
                "status": lock06b2.get("status"),
            },
        },
        "rewrite_policy": (
            "06c is immutable after any all-78 V1 full-data model fit begins. "
            "Do not change model family, feature budget, primary seed, fixed epoch count, "
            "or transfer SABLE-MO q10/stability thresholds to V1 in response to full-data, "
            "biological, clinical, or outcome results."
        ),
    }

    contract_path = out_dir / "target_v1_full_data_refit_contract.json"
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    readme_text = f"""Paper 5 06c - frozen TARGET V1 full-data refit contract
================================================================
Script version: {SCRIPT_VERSION}

06b selected V1_SHARED_PRIVATE_AE in 17/25 outer splits (68%), satisfying
the frozen >=40% unique-primary rule. All 700 outer benchmark rows passed.

This stage freezes the first all-78 refit protocol BEFORE that refit occurs:
- model: V1_SHARED_PRIVATE_AE
- profile: ARCH_COMPACT__REG_WEAK
- primary feature budgets: RNA 5000 / METHYLATION 10000 / CNV 5000
- all 78 cases, outcome-blind
- primary seed: 1103 (first already-frozen 05a outer seed)
- reproducibility seeds: 2207 and 3301; never allowed to replace primary
- fixed epochs: 400, inherited from the frozen profile maximum_epochs
- no early stopping or epoch selection on the full 78 cases
- no post-fit seed selection
- no SABLE-MO q10=.60/.40 or stability gate applied to V1

The 06b V1 outer selected-epoch distribution is recorded as outcome-blind
consistency evidence only. It does not tune the final epoch count.

06c performs no full-data model fit and reads no clinical outcomes or biological annotations.
"""
    readme_path = out_dir / "06c_README.txt"
    readme_path.write_text(readme_text, encoding="utf-8")

    lock = {
        "status": "06C_TARGET_V1_FULL_DATA_REFIT_CONTRACT_FROZEN",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": created_at,
        "full_data_fit_performed": False,
        "clinical_outcomes_read": False,
        "biological_annotations_read": False,
        "q10_applied_to_v1": False,
        "primary_model_id": EXPECTED["primary_model_id"],
        "primary_seed": EXPECTED["primary_seed"],
        "fixed_epochs": EXPECTED["fixed_epochs"],
        "contract": {
            "path": relative_to_paper5(contract_path, paper5),
            "sha256": sha256_file(contract_path),
            "size_bytes": contract_path.stat().st_size,
        },
        "readme": {
            "path": relative_to_paper5(readme_path, paper5),
            "sha256": sha256_file(readme_path),
            "size_bytes": readme_path.stat().st_size,
        },
        "script_source": {
            "path": relative_to_paper5(script_path, paper5),
            "sha256": sha256_file(script_path),
            "size_bytes": script_path.stat().st_size,
        },
        "06b_result_identity": {
            "summary_sha256": EXPECTED["06b_summary_sha256"],
            "selection_sha256": EXPECTED["06b_selection_sha256"],
            "outer_sha256": EXPECTED["06b_outer_sha256"],
            "results_lock_sha256": EXPECTED["06b_lock_sha256"],
            "runner_sha256": EXPECTED["06b_runner_sha256"],
        },
        "rewrite_policy": (
            "Do not overwrite 06c after any all-78 V1 fit begins. "
            "Any execution runner must consume this exact frozen contract."
        ),
    }

    lock_path = out_dir / "06c_full_data_refit_contract_lock.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    print("06b result identity: PASS")
    print("06b frozen unique primary: V1_SHARED_PRIVATE_AE")
    print("Selection frequency: 17/25 = 68.0%")
    print("Outer failed rows: 0/700")
    print("Frozen neural profile:", EXPECTED["profile_id"])
    print("Full-data cases:", EXPECTED["n_cases"])
    print(
        "Primary feature budgets: "
        f"RNA={observed_budgets['RNA']}, "
        f"METHYLATION={observed_budgets['METHYLATION']}, "
        f"CNV={observed_budgets['CNV']}"
    )
    print("Primary full-data seed:", EXPECTED["primary_seed"])
    print("Reproducibility-only seeds:", EXPECTED["reproducibility_seeds"])
    print("Fixed full-data epochs:", EXPECTED["fixed_epochs"])
    print(
        "V1 outer selected epochs: "
        f"min={epoch_summary['min']}, median={epoch_summary['median']}, "
        f"mean={epoch_summary['mean']:.6f}, max={epoch_summary['max']}"
    )
    print("SABLE-MO q10 transfer to V1: FORBIDDEN")
    print("TARGET outcomes/clinical/biology read in 06c: NO")
    print("TARGET full-data model fit in 06c: NO")
    print("=" * 104)
    print("Paper 5 06c TARGET V1 full-data refit contract: FROZEN")
    print("=" * 104)
    print("Contract:", relative_to_paper5(contract_path, paper5))
    print("Lock:", relative_to_paper5(lock_path, paper5))
    print("Next: build a new versioned V1 all-78 execution runner that consumes this exact 06c lock.")
    print("=" * 104)


if __name__ == "__main__":
    main()
