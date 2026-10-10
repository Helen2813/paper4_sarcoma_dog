#!/usr/bin/env python3
"""
Paper 5 - 05cmp2 freeze scientific secondary-comparator confirmatory benchmark contract.

This stage is CONTRACT/METADATA ONLY. It freezes the exact post-05e secondary
benchmark scope for iCluster2b and MOVE before any scientific comparator fit is
performed.

Scientific scope frozen here:
- same 360 confirmatory simulation datasets already used by the frozen 05e context;
- confirmatory replicates 11-20 across 36 scenarios;
- iCluster2b and MOVE only;
- common cross-method endpoints only where the 05cmp1 metric registry explicitly
  marks both methods eligible;
- MOVE-native held-out reconstruction remains descriptive and method-specific;
- iCluster2b latent-subspace stability is not invented from a single-seed registry:
  it is recorded as NOT_ESTIMABLE_WITH_FROZEN_SINGLE_SEED_POLICY in this stage;
- no TARGET comparator run is authorized here;
- no outcome/clinical/biology access;
- no influence on 05e, final gate, or TARGET selection.

No model fitting, no GPU use, no R execution, no simulation matrix/truth-array read.
Only frozen metadata/result registries are read.

No command-line arguments.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_VERSION = (
    "05cmp2-freeze-secondary-comparator-confirmatory-benchmark-contract-v1-no-cli"
)
STATUS = "FROZEN_SECONDARY_COMPARATOR_CONFIRMATORY_BENCHMARK_BEFORE_SCIENTIFIC_FITS"

EXPECTED = {
    "cmp1_contract_sha256":
        "418259cedd36cbd1913a4ff1ae9874a3decfddbb7a75e9b3bca2e9623f0aca95",
    "cmp1_lock_sha256":
        "37e0db502a88f7beab52f1457cb0c5367091314faa6b86c7796810d883f31648",
    "cmp1_contract_status":
        "FROZEN_SECONDARY_COMPARATOR_IMPLEMENTATIONS_BEFORE_05E_AND_SCIENTIFIC_RUNS",
    "05e_lock_sha256":
        "e0adaccada7f9b04f0e699badfac252a6dc865f24d2ea39a243b07bc3ddbafbf",
    "05e_status":
        "FROZEN_PRIMARY_PROPOSED_REPRESENTATION_METHOD_BEFORE_FINAL_GATE_AND_TARGET",
    "05d4_dataset_metrics_sha256":
        "9249e41112eb4401e5e0e34ab8c7b3ede2440b5b018cba8e14ab93ed10c43a3f",
    "methods": ["iCluster2b", "MOVE"],
    "n_scenarios": 36,
    "replicates": list(range(11, 21)),
    "n_confirmatory_datasets": 360,
    "latent_dim": 5,
    "view_order": ["RNA", "METHYLATION", "CNV"],
    "common_cross_method_metrics": [
        "shared_subspace_recovery",
        "factor_alignment_correlation",
        "scenario_robustness_summary",
        "missing_view_robustness",
    ],
    "move_method_specific_metrics": [
        "latent_subspace_stability",
        "heldout_reconstruction",
    ],
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen file missing: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def require_empty_or_absent(path: Path, label: str) -> None:
    if not path.exists():
        return
    files = [p for p in path.rglob("*") if p.is_file()]
    if files:
        raise RuntimeError(
            f"{label} already contains files; refusing to freeze a new "
            f"prospective comparator contract: {files[:10]}"
        )


def main() -> None:
    print("=" * 112)
    print("Paper 5 - freeze post-05e scientific secondary-comparator benchmark contract")
    print("=" * 112)
    print("Script version:", SCRIPT_VERSION)
    print("Comparator scientific fits in 05cmp2: NO")
    print("Simulation matrices/truth arrays read in 05cmp2: NO")
    print("TARGET data read in 05cmp2: NO")
    print("Outcomes/clinical/biology read in 05cmp2: NO")
    print("05e/final-gate/TARGET selection influence: FORBIDDEN")

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    paper5 = project_root / "paper5"

    cmp1_contract_path = (
        paper5 / "method_contract" / "05cmp1"
        / "secondary_comparator_implementation_contract.json"
    )
    cmp1_lock_path = (
        paper5 / "method_contract" / "05cmp1"
        / "secondary_comparator_implementation_contract_lock.json"
    )
    cmp1_impl_registry = (
        paper5 / "method_contract" / "05cmp1"
        / "secondary_comparator_implementation_registry.csv"
    )
    cmp1_adapter_registry = (
        paper5 / "method_contract" / "05cmp1"
        / "secondary_comparator_scientific_adapter_registry.csv"
    )
    cmp1_seed_registry = (
        paper5 / "method_contract" / "05cmp1"
        / "secondary_comparator_seed_registry.csv"
    )
    cmp1_metric_registry = (
        paper5 / "method_contract" / "05cmp1"
        / "secondary_comparator_metric_eligibility_registry.csv"
    )
    cmp1_failure_policy = (
        paper5 / "method_contract" / "05cmp1"
        / "secondary_comparator_failure_policy.json"
    )
    cmp0_benchmark_policy = (
        paper5 / "method_contract" / "05cmp0"
        / "secondary_comparator_benchmark_policy.json"
    )

    freeze05e_path = (
        paper5 / "method_contract" / "05e"
        / "representation_method_freeze.json"
    )
    lock05e_path = (
        paper5 / "method_contract" / "05e"
        / "representation_method_freeze_lock.json"
    )

    dataset05d4_path = (
        paper5 / "results" / "simulation_benchmark" / "05d_neural"
        / "confirmatory_ablation"
        / "neural_confirmatory_dataset_variant_metrics.csv.gz"
    )
    manifest05b_path = (
        paper5 / "method_contract" / "05b_v2"
        / "architecture_screening_data_manifest.csv"
    )

    for path in [
        cmp1_contract_path,
        cmp1_lock_path,
        cmp1_impl_registry,
        cmp1_adapter_registry,
        cmp1_seed_registry,
        cmp1_metric_registry,
        cmp1_failure_policy,
        cmp0_benchmark_policy,
        freeze05e_path,
        lock05e_path,
        dataset05d4_path,
        manifest05b_path,
    ]:
        require_file(path)

    out_dir = paper5 / "method_contract" / "05cmp2"
    comparator_result_root = (
        paper5 / "results" / "simulation_benchmark"
        / "05cmp_secondary_comparators"
    )
    require_empty_or_absent(
        out_dir,
        "05cmp2 contract directory",
    )
    require_empty_or_absent(
        comparator_result_root,
        "secondary-comparator scientific result root",
    )

    # ------------------------------------------------------------------
    # Exact frozen 05cmp1 / 05e identity.
    # ------------------------------------------------------------------
    if sha256_file(cmp1_contract_path) != EXPECTED["cmp1_contract_sha256"]:
        raise RuntimeError("05cmp1 implementation contract SHA mismatch.")
    if sha256_file(cmp1_lock_path) != EXPECTED["cmp1_lock_sha256"]:
        raise RuntimeError("05cmp1 implementation lock SHA mismatch.")
    if sha256_file(lock05e_path) != EXPECTED["05e_lock_sha256"]:
        raise RuntimeError("05e representation freeze lock SHA mismatch.")

    cmp1_contract = load_json(cmp1_contract_path)
    cmp1_lock = load_json(cmp1_lock_path)
    freeze05e = load_json(freeze05e_path)
    lock05e = load_json(lock05e_path)

    if cmp1_contract.get("status") != EXPECTED["cmp1_contract_status"]:
        raise RuntimeError("05cmp1 status mismatch.")
    if cmp1_lock.get("status") != EXPECTED["cmp1_contract_status"]:
        raise RuntimeError("05cmp1 lock status mismatch.")
    if lock05e.get("status") != EXPECTED["05e_status"]:
        raise RuntimeError("05e lock status mismatch.")
    if freeze05e.get("status") != EXPECTED["05e_status"]:
        raise RuntimeError("05e freeze status mismatch.")

    # ------------------------------------------------------------------
    # Exact registries. Do not infer methods/metrics from memory.
    # ------------------------------------------------------------------
    impl = pd.read_csv(cmp1_impl_registry, dtype=str).fillna("")
    adapter = pd.read_csv(cmp1_adapter_registry, dtype=str).fillna("")
    seeds = pd.read_csv(cmp1_seed_registry, dtype=str).fillna("")
    metrics = pd.read_csv(cmp1_metric_registry, dtype=str).fillna("")

    if sorted(impl["method_name"].tolist()) != sorted(EXPECTED["methods"]):
        raise RuntimeError(
            f"Unexpected 05cmp1 comparator methods: {impl['method_name'].tolist()}"
        )
    if sorted(adapter["method_name"].tolist()) != sorted(EXPECTED["methods"]):
        raise RuntimeError("05cmp1 adapter method set mismatch.")

    for method in EXPECTED["methods"]:
        row = adapter.loc[adapter["method_name"].eq(method)]
        if row.shape[0] != 1:
            raise RuntimeError(f"Expected exactly one adapter row for {method}.")
        rr = row.iloc[0]
        if rr["view_order"] != "RNA|METHYLATION|CNV":
            raise RuntimeError(f"{method}: view order changed.")
        if int(rr["latent_dimension"]) != EXPECTED["latent_dim"]:
            raise RuntimeError(f"{method}: latent dimension changed.")
        if rr["outcome_or_truth_access"] != "FORBIDDEN":
            raise RuntimeError(f"{method}: outcome/truth access boundary changed.")

    metric_index = metrics.set_index("metric")
    for metric in EXPECTED["common_cross_method_metrics"]:
        if metric not in metric_index.index:
            raise RuntimeError(f"Frozen metric missing: {metric}")
        row = metric_index.loc[metric]
        if row["iCluster2b_eligible"] != "YES" or row["MOVE_eligible"] != "YES":
            raise RuntimeError(
                f"Metric no longer eligible for both comparators: {metric}"
            )

    # Explicitly preserve exclusions.
    forbidden_cross_method = [
        "cross_view_retrieval_MRR",
        "private_subspace_recovery",
        "SABLE_stability_gate_PASS_FAIL",
        "primary_05e_selection_score",
        "final_gate_decision",
    ]
    for metric in forbidden_cross_method:
        if metric not in metric_index.index:
            raise RuntimeError(f"Frozen excluded metric missing: {metric}")

    # Seed policy verification.
    i_rows = seeds.loc[seeds["method_name"].eq("iCluster2b")]
    if i_rows.shape[0] != 1:
        raise RuntimeError("iCluster2b seed registry must contain one row.")
    if (
        i_rows.iloc[0]["seed_label"] != "icluster_fit"
        or i_rows.iloc[0]["base_seed"] != "1103"
    ):
        raise RuntimeError("iCluster2b seed policy changed.")

    move_rows = seeds.loc[seeds["method_name"].eq("MOVE")]
    if set(move_rows["seed_label"]) != {
        "architecture_selection",
        "canonical_fit",
        "stability_rep_2",
        "stability_rep_3",
    }:
        raise RuntimeError("MOVE seed-label set changed.")
    move_seed_map = {
        str(r.seed_label): int(r.base_seed)
        for r in move_rows.itertuples(index=False)
    }
    expected_move_seed_map = {
        "architecture_selection": 1103,
        "canonical_fit": 1103,
        "stability_rep_2": 2207,
        "stability_rep_3": 3301,
    }
    if move_seed_map != expected_move_seed_map:
        raise RuntimeError(f"MOVE seed policy changed: {move_seed_map}")

    # ------------------------------------------------------------------
    # Freeze the exact same 360 confirmatory dataset bank as 05e.
    # Read only identifiers from already-frozen 05d4 dataset-level output.
    # ------------------------------------------------------------------
    if sha256_file(dataset05d4_path) != EXPECTED["05d4_dataset_metrics_sha256"]:
        raise RuntimeError("Frozen 05d4 dataset-variant metrics SHA mismatch.")

    d = pd.read_csv(dataset05d4_path, low_memory=False)
    required_key_cols = {
        "scenario_id",
        "replicate_index",
        "simulation_seed",
        "variant_id",
    }
    if not required_key_cols.issubset(d.columns):
        raise RuntimeError(
            "05d4 dataset table missing required identifier columns: "
            + ", ".join(sorted(required_key_cols - set(d.columns)))
        )

    v4 = d.loc[
        d["variant_id"].astype(str).eq("V4_SABLE_MO"),
        ["scenario_id", "replicate_index", "simulation_seed"],
    ].copy()

    for col in ["replicate_index", "simulation_seed"]:
        v4[col] = pd.to_numeric(v4[col], errors="raise").astype(int)

    v4 = v4.drop_duplicates().sort_values(
        ["scenario_id", "replicate_index", "simulation_seed"],
        kind="mergesort",
    ).reset_index(drop=True)

    if v4.shape[0] != EXPECTED["n_confirmatory_datasets"]:
        raise RuntimeError(
            f"Expected 360 confirmatory keys from 05d4; got {v4.shape[0]}."
        )
    if v4["scenario_id"].nunique() != EXPECTED["n_scenarios"]:
        raise RuntimeError("Confirmatory key set does not contain 36 scenarios.")
    if sorted(v4["replicate_index"].unique().tolist()) != EXPECTED["replicates"]:
        raise RuntimeError(
            "Confirmatory key set does not use exactly replicates 11-20."
        )

    # Join to frozen 05b_v2 manifest without opening any simulation dataset.
    manifest = pd.read_csv(manifest05b_path, dtype=str, low_memory=False).fillna("")
    if manifest.shape[0] != 720:
        raise RuntimeError(
            f"Frozen 05b_v2 manifest row count changed: {manifest.shape[0]} != 720."
        )
    for c in ["scenario_id", "replicate_index", "simulation_seed"]:
        if c not in manifest.columns:
            raise RuntimeError(f"05b_v2 manifest missing key column: {c}")

    manifest["replicate_index"] = pd.to_numeric(
        manifest["replicate_index"], errors="raise"
    ).astype(int)
    manifest["simulation_seed"] = pd.to_numeric(
        manifest["simulation_seed"], errors="raise"
    ).astype(int)

    if manifest.duplicated(
        ["scenario_id", "replicate_index", "simulation_seed"]
    ).any():
        raise RuntimeError("05b_v2 manifest contains duplicate dataset keys.")

    registry = v4.merge(
        manifest,
        on=["scenario_id", "replicate_index", "simulation_seed"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not registry["_merge"].eq("both").all():
        raise RuntimeError(
            "At least one confirmatory key is missing from frozen 05b_v2 manifest."
        )
    registry = registry.drop(columns=["_merge"])
    registry.insert(
        0,
        "comparator_dataset_id",
        [
            f"{sid}__R{rep:02d}__S{seed}"
            for sid, rep, seed in registry[
                ["scenario_id", "replicate_index", "simulation_seed"]
            ].itertuples(index=False, name=None)
        ],
    )
    if registry["comparator_dataset_id"].duplicated().any():
        raise RuntimeError("Derived comparator_dataset_id is not unique.")

    # ------------------------------------------------------------------
    # Freeze the exact prospective scientific protocol.
    # ------------------------------------------------------------------
    contract = {
        "status": STATUS,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),

        "role": (
            "POST_05E_SECONDARY_CONFIRMATORY_SIMULATION_BENCHMARK_ONLY"
        ),

        "scope": {
            "methods": EXPECTED["methods"],
            "n_scenarios": EXPECTED["n_scenarios"],
            "confirmatory_replicates": EXPECTED["replicates"],
            "n_datasets": EXPECTED["n_confirmatory_datasets"],
            "dataset_bank":
                "exact same 360 confirmatory simulation datasets used by frozen 05e context",
            "TARGET_comparator_run_authorized_here": False,
            "final_gate_data_authorized_here": False,
            "outcomes_clinical_biology_authorized_here": False,
        },

        "data_adapter": {
            "view_order": EXPECTED["view_order"],
            "orientation": "samples_x_features_per_view",
            "latent_dimension": 5,
            "standardization": (
                "Within each actual fit/training split, estimate feature center "
                "and scale from observed entries in that fit split only; apply "
                "those parameters to held-out rows. For a full-data scientific "
                "fit, the full dataset is the fit split."
            ),
            "remaining_missing_after_standardization": 0.0,
            "whole_view_missing": "all-zero standardized row",
            "missingness_indicator": False,
            "sample_repair_or_repairing": "FORBIDDEN",
            "truth_use_for_fit_tuning_or_failure_handling": "FORBIDDEN",
            "truth_use_after_fit_for_eligible_simulation_evaluation": "ALLOWED",
        },

        "iCluster2b": {
            "implementation":
                "iClusterPlus::tune.iCluster2b -> iCluster2b",
            "latent_K": 5,
            "regularization": "lasso",
            "tuning": {
                "min_lambda": 10.0,
                "max_lambda": 500.0,
                "lambda_iter": 25,
                "EM_iter": 25,
                "eps": 0.0001,
                "eps2": 1e-6,
                "min_shrinkage_rate_per_view": [0.05, 0.05, 0.05],
            },
            "fit_rule": (
                "One prospectively seeded tune.iCluster2b call per dataset; use "
                "the returned lambda values for the scientific K=5 fit. No "
                "performance/truth/outcome-guided retuning."
            ),
            "seed": {
                "base_seed": 1103,
                "derivation":
                    "SHA256(method|dataset_id|purpose|base_seed), first 8 hex, "
                    "1 + value mod 2147483646",
                "purpose": "dataset_scientific_tuning_and_fit",
                "seed_label": "icluster_fit",
            },
            "latent_subspace_stability": (
                "NOT_ESTIMABLE_WITH_FROZEN_SINGLE_SEED_POLICY_IN_05CMP2"
            ),
            "stability_reason": (
                "05cmp1 froze one deterministic iCluster2b scientific seed and "
                "did not freeze a resampling/replicate schedule for this method. "
                "05cmp2 will not invent one after 05e."
            ),
        },

        "MOVE": {
            "version": "1.5.0",
            "latent_dimension": 5,
            "candidate_hidden_architectures": {
                "MOVE_H64_32": [64, 32],
                "MOVE_H128_64": [128, 64],
                "MOVE_H256_128": [256, 128],
            },
            "fixed_training": {
                "epochs": 300,
                "learning_rate": 0.001,
                "dropout": 0.1,
                "beta": 0.01,
                "kld_warmup_steps": [1, 75, 150, 300],
                "batch_size_rule": "min(16, n_fit_samples)",
                "AMP": False,
                "TF32": False,
                "torch_compile": False,
                "dtype": "float32",
                "device": "cuda:0",
            },
            "architecture_selection_stage_1": {
                "folds": 5,
                "fold_assignment": (
                    "Sort the frozen sample IDs lexicographically within each "
                    "dataset and assign fold = sorted_position mod 5. No truth "
                    "or outcome enters the assignment."
                ),
                "metric":
                    "mean held-out reconstruction SSE over observed standardized entries only",
                "rule":
                    "retain candidates within one standard error of the best mean reconstruction SSE",
                "seed_base": 1103,
                "seed_derivation":
                    "SHA256(method|dataset_id|fold_id|candidate_id|purpose|base_seed)",
            },
            "architecture_selection_stage_2": {
                "full_data_seeds": [1103, 2207, 3301],
                "pairwise_subspace_metric": (
                    "For each seed pair, center latent coordinates, obtain "
                    "orthonormal column-space bases by thin SVD, compute the "
                    "five singular values of Q_A^T Q_B, take their arithmetic "
                    "mean; candidate stability is the median of the three "
                    "pairwise means."
                ),
                "selection":
                    "highest stability; exact ties -> fewer hidden parameters -> lexical candidate_id",
                "canonical_scientific_fit_seed_role":
                    "canonical_fit/base_seed_1103 only",
                "other_seed_role":
                    "stability diagnostic only; never replace canonical_fit",
            },
        },

        "eligible_metrics": {
            "common_cross_method_primary_context":
                EXPECTED["common_cross_method_metrics"],
            "MOVE_method_specific_descriptive":
                EXPECTED["move_method_specific_metrics"],
            "iCluster2b_latent_subspace_stability":
                "NOT_ESTIMABLE_WITH_FROZEN_SINGLE_SEED_POLICY_IN_05CMP2",
            "forbidden_or_not_cross_method": [
                "cross_view_retrieval_MRR",
                "private_subspace_recovery",
                "SABLE_stability_gate_PASS_FAIL",
                "primary_05e_selection_score",
                "final_gate_decision",
            ],
        },

        "metric_semantics": {
            "shared_subspace_recovery": (
                "Use the existing Paper 5 simulation definition: mean canonical "
                "correlation between estimated and true shared latent subspaces; "
                "truth is accessed only after the representation fit is complete."
            ),
            "factor_alignment_correlation": (
                "Sign/permutation-invariant factor alignment on the frozen "
                "simulation truth; runner must reuse the same Paper 5 alignment "
                "semantics/helpers already used in the simulation branch rather "
                "than define a new post-hoc metric."
            ),
            "scenario_robustness_summary": (
                "Aggregate only over the same 36 frozen scenario groups; report "
                "method/scenario summaries and an overall scenario-balanced "
                "descriptive summary. No winner threshold."
            ),
            "missing_view_robustness": (
                "Use only the already-frozen missing-view scenarios and the "
                "fixed zero adapter. No comparator-specific rescue."
            ),
            "heldout_reconstruction": (
                "MOVE-native descriptive endpoint only; do not compare directly "
                "against iCluster2b because 05cmp1 marked it ineligible there."
            ),
        },

        "failure_policy": {
            "environment_mismatch": "STOP before scientific fits",
            "per_dataset_fit_failure": (
                "Retry once with identical configuration and identical seed. "
                "If it fails again, keep the dataset-method failure in the "
                "denominator/manifest."
            ),
            "nonfinite_output": (
                "Record dataset-method failure; no clipping/rescue/retuning."
            ),
            "fallbacks": {
                "iClusterBayes2": "LOCKED_INACTIVE",
                "AUTOENCODIX_varix": "LOCKED_INACTIVE",
            },
        },

        "interpretation_firewall": {
            "may_reopen_05e": False,
            "may_change_final_gate": False,
            "may_change_TARGET_selected_representation": False,
            "may_access_TARGET_outcomes": False,
            "may_declare_cross_family_winner": False,
            "may_create_new_composite_score": False,
            "may_retrofit_SABLE_private_or_retention_metrics_to_comparators": False,
        },

        "source_identity": {
            "05cmp1_contract": {
                "path":
                    "method_contract/05cmp1/secondary_comparator_implementation_contract.json",
                "sha256": sha256_file(cmp1_contract_path),
            },
            "05cmp1_lock": {
                "path":
                    "method_contract/05cmp1/secondary_comparator_implementation_contract_lock.json",
                "sha256": sha256_file(cmp1_lock_path),
            },
            "05cmp1_implementation_registry": {
                "sha256": sha256_file(cmp1_impl_registry),
            },
            "05cmp1_adapter_registry": {
                "sha256": sha256_file(cmp1_adapter_registry),
            },
            "05cmp1_seed_registry": {
                "sha256": sha256_file(cmp1_seed_registry),
            },
            "05cmp1_metric_registry": {
                "sha256": sha256_file(cmp1_metric_registry),
            },
            "05cmp1_failure_policy": {
                "sha256": sha256_file(cmp1_failure_policy),
            },
            "05cmp0_benchmark_policy": {
                "sha256": sha256_file(cmp0_benchmark_policy),
            },
            "05e_lock": {
                "path":
                    "method_contract/05e/representation_method_freeze_lock.json",
                "sha256": sha256_file(lock05e_path),
            },
            "05d4_dataset_variant_metrics": {
                "path":
                    "results/simulation_benchmark/05d_neural/confirmatory_ablation/"
                    "neural_confirmatory_dataset_variant_metrics.csv.gz",
                "sha256": sha256_file(dataset05d4_path),
            },
            "05b_v2_manifest": {
                "path":
                    "method_contract/05b_v2/architecture_screening_data_manifest.csv",
                "sha256": sha256_file(manifest05b_path),
            },
        },

        "scientific_state_at_freeze": {
            "secondary_comparator_scientific_fit_performed": False,
            "secondary_comparator_scientific_result_accessed": False,
            "simulation_matrix_or_truth_array_read_here": False,
            "TARGET_data_accessed_here": False,
            "outcome_clinical_biology_accessed_here": False,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=False)

    registry_path = (
        out_dir / "secondary_comparator_confirmatory_dataset_registry.csv"
    )
    registry.to_csv(registry_path, index=False)

    contract_path = (
        out_dir / "secondary_comparator_confirmatory_benchmark_contract.json"
    )
    contract_path.write_text(
        json.dumps(contract, indent=2) + "\n",
        encoding="utf-8",
    )

    readme_path = out_dir / "05cmp2_README.txt"
    readme = f"""Paper 5 - 05cmp2 secondary comparator confirmatory benchmark contract
================================================================================

Status: {STATUS}

Scope
-----
Methods: iCluster2b, MOVE
Dataset bank: exact 360 confirmatory simulations used by frozen 05e context
Scenarios: 36
Replicates: 11-20

Common cross-method context
---------------------------
shared_subspace_recovery
factor_alignment_correlation
scenario_robustness_summary
missing_view_robustness

Method-specific
---------------
MOVE latent_subspace_stability: allowed under frozen 3-seed architecture/stability policy
MOVE heldout_reconstruction: descriptive only
iCluster2b latent_subspace_stability: NOT ESTIMABLE in 05cmp2 because 05cmp1 froze only
one deterministic scientific seed and no iCluster2b resampling schedule.

Prohibited
----------
No TARGET comparator run in this contract.
No outcome/clinical/biology access.
No 05e/final-gate/TARGET selection consequence.
No cross-family winner.
No new composite score.
No fallback activation after dataset-specific performance.

Next
----
Build a separately source-authorized scientific comparator runner that consumes this exact contract.
"""
    readme_path.write_text(readme, encoding="utf-8")

    lock = {
        "status":
            "05CMP2_SECONDARY_COMPARATOR_CONFIRMATORY_BENCHMARK_CONTRACT_FROZEN",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "contract": {
            "path":
                "method_contract/05cmp2/"
                "secondary_comparator_confirmatory_benchmark_contract.json",
            "sha256": sha256_file(contract_path),
            "size_bytes": contract_path.stat().st_size,
        },
        "dataset_registry": {
            "path":
                "method_contract/05cmp2/"
                "secondary_comparator_confirmatory_dataset_registry.csv",
            "sha256": sha256_file(registry_path),
            "size_bytes": registry_path.stat().st_size,
            "rows": int(registry.shape[0]),
        },
        "readme": {
            "path": "method_contract/05cmp2/05cmp2_README.txt",
            "sha256": sha256_file(readme_path),
            "size_bytes": readme_path.stat().st_size,
        },
        "script_source": {
            "path":
                "scripts/"
                "05cmp2_freeze_secondary_comparator_confirmatory_benchmark_contract_v1.py",
            "sha256": sha256_file(script_path),
            "size_bytes": script_path.stat().st_size,
        },
        "rewrite_policy": (
            "05cmp2 is immutable after creation. Any scientific comparator "
            "protocol change requires a new versioned amendment before additional "
            "scientific comparator results."
        ),
    }
    lock_path = (
        out_dir
        / "secondary_comparator_confirmatory_benchmark_contract_lock.json"
    )
    lock_path.write_text(
        json.dumps(lock, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Exact 05cmp1 implementation contract/lock: PASS")
    print("Exact 05e post-freeze boundary: PASS")
    print("Exact 05d4 360-dataset confirmatory key source: PASS")
    print("Frozen 05b_v2 manifest geometry: PASS [720 rows]")
    print("Comparator confirmatory registry: PASS [360 rows / 36 scenarios / reps 11-20]")
    print("Methods: iCluster2b + MOVE")
    print("Common cross-method metrics:", "; ".join(EXPECTED["common_cross_method_metrics"]))
    print("MOVE method-specific stability: AUTHORIZED")
    print("MOVE heldout reconstruction: DESCRIPTIVE_ONLY")
    print("iCluster2b latent-subspace stability: NOT_ESTIMABLE_WITH_FROZEN_SINGLE_SEED_POLICY")
    print("TARGET comparator run authorized here: NO")
    print("Comparator scientific fits performed here: NO")
    print("Simulation matrices/truth arrays read here: NO")
    print("Outcomes/clinical/biology read here: NO")
    print("Cross-family winner declared: NO")
    print("=" * 112)
    print("Paper 5 05cmp2 secondary comparator confirmatory benchmark contract: FROZEN")
    print("=" * 112)
    print(
        "Contract: method_contract/05cmp2/"
        "secondary_comparator_confirmatory_benchmark_contract.json"
    )
    print(
        "Registry: method_contract/05cmp2/"
        "secondary_comparator_confirmatory_dataset_registry.csv"
    )
    print(
        "Lock: method_contract/05cmp2/"
        "secondary_comparator_confirmatory_benchmark_contract_lock.json"
    )
    print("=" * 112)


if __name__ == "__main__":
    main()
