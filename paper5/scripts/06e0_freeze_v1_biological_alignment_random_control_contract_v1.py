#!/usr/bin/env python3
"""
Paper 5 - 06e0 freeze V1 biological-alignment and matched-random-control contract.

This is a PRE-BIOLOGY contract stage.

It freezes how the already-frozen TARGET V1 representation may later be compared
with the four previously frozen Paper-4 molecular programs M34, M40, M11, M24,
including a matched-random-program null.

This stage DOES NOT:
- open/parse Paper-4 program manifests, gene weights, scoring tables, or expression;
- open TARGET biological annotations;
- open TARGET clinical variables or outcomes;
- read TARGET molecular matrices;
- read the numeric V1 representation array;
- fit/refit/select any representation;
- apply SABLE-MO q10/stability retention to V1;
- run the biological alignment or random controls.

Only non-biological Paper-5 provenance locks are read. Biological input identities
are frozen by path + exact expected SHA-256 and are first verified/read only by a
future separately source-authorized execution stage.

No command-line arguments.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "06e0-freeze-v1-biological-alignment-random-control-contract-v1-no-cli"

EXPECTED = {
    "06c3_lock_sha256":
        "b23d4c630cffca3bc5b1c44328128d4da46028b8db8cb070e6be93d9d0795be5",
    "model_id": "V1_SHARED_PRIVATE_AE",
    "profile_id": "ARCH_COMPACT__REG_WEAK",
    "primary_seed": 1103,
    "n_cases": 78,
    "shared_dim": 5,
    "representation_filename": "target_v1_patient_shared_seed1103.npy",
    "representation_sha256":
        "c05bf08017ad8544e1e4ce0c8f58f086060dd7a6a0a68ed7f8d5a6723768a95f",
    "technical_status":
        "UNKNOWN_NO_STRICT_USABLE_TECHNICAL_COVARIATES",
    "programs": ["M34", "M40", "M11", "M24"],
    "strict_gene_counts": {
        "M34": 162,
        "M40": 111,
        "M11": 7,
        "M24": 7,
    },
    "random_panels_per_program": 1000,
    "random_seed": 42,
    "variability_bins": 10,
    "primary_score_variant": "strict_signed_mean",
    "primary_alignment_metric":
        "max_absolute_spearman_across_5_frozen_v1_shared_coordinates",
}

# These are frozen now but deliberately NOT opened/parsed in 06e0.
BIOLOGICAL_INPUT_IDENTITIES = {
    "paper4_frozen_program_manifest": {
        "path": "results/tables/GSE238110_frozen_canine_transfer_program_manifest.csv",
        "sha256":
            "fcd34b985ea99986f39dfc43ce8e4018db2fde9ecc8a225bb40ac93006d5372c",
        "size_bytes": 14880,
    },
    "paper4_frozen_strict_gene_weights": {
        "path": "results/tables/GSE238110_frozen_transfer_gene_weights_strict.csv",
        "sha256":
            "4f065aa4c4edf117a0c74015840d2b4b2347929f172cd517e1818ba0f6163b91",
        "size_bytes": 50062,
    },
    "paper4_frozen_scoring_specification": {
        "path": "results/tables/GSE238110_frozen_transfer_scoring_specification.csv",
        "sha256":
            "562aab92caa949691f43b0285cf2585c98008e78c62d2c15477a6d156d7f47a7",
        "size_bytes": 14951,
    },
    "paper4_target_expression": {
        "path": "data/processed/human_validation/TARGET_OS_expression_log2_gene_symbol.csv",
        "sha256":
            "0cde0d000cafa4acb40f42f90bcf6b8e82ece92e5da8db1839bf99007eb5e52f",
        "size_bytes": 66594174,
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, indent=2) + "\n",
        encoding="utf-8",
    )


def rel(path: Path, paper5: Path) -> str:
    return path.resolve().relative_to(paper5.resolve()).as_posix()


def main() -> None:
    print("=" * 104)
    print(
        "Paper 5 - 06e0 freeze V1 biological-alignment "
        "and matched-random-control contract"
    )
    print("=" * 104)
    print("Script version:", SCRIPT_VERSION)
    print("Paper-4 biological artifact contents opened in 06e0: NO")
    print("TARGET biological annotations opened in 06e0: NO")
    print("TARGET clinical/outcomes opened in 06e0: NO")
    print("TARGET molecular matrices read in 06e0: NO")
    print("V1 numeric representation array read in 06e0: NO")
    print("Model fit/refit in 06e0: NO")

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    paper5 = project_root / "paper5"

    results_lock_path = (
        paper5 / "method_contract" / "06c3"
        / "target_v1_full_data_refit_results_lock.json"
    )
    scope_lock_path = (
        paper5 / "method_contract" / "06d0"
        / "06d0_v1_postfit_downstream_scope_audit_lock.json"
    )
    technical_lock_path = (
        paper5 / "method_contract" / "06d1_v2"
        / "06d1_v2_v1_postfreeze_technical_status_audit_lock.json"
    )

    require_file(results_lock_path)
    require_file(scope_lock_path)
    require_file(technical_lock_path)

    if sha256_file(results_lock_path) != EXPECTED["06c3_lock_sha256"]:
        raise RuntimeError("06c3 exact result-lock SHA mismatch.")

    results_lock = load_json(results_lock_path)
    scope_lock = load_json(scope_lock_path)
    technical_lock = load_json(technical_lock_path)

    if results_lock.get("status") != (
        "06C3_TARGET_V1_FULL_DATA_REFIT_RESULTS_FROZEN"
    ):
        raise RuntimeError("06c3 result lock is not frozen.")
    if results_lock.get("model_id") != EXPECTED["model_id"]:
        raise RuntimeError("06c3 selected model mismatch.")
    if results_lock.get("profile_id") != EXPECTED["profile_id"]:
        raise RuntimeError("06c3 frozen profile mismatch.")
    if int(results_lock.get("primary_seed")) != EXPECTED["primary_seed"]:
        raise RuntimeError("06c3 primary seed mismatch.")

    rep_record = (
        results_lock.get("result_files", {})
        .get(EXPECTED["representation_filename"], {})
    )
    if str(rep_record.get("sha256", "")).lower() != (
        EXPECTED["representation_sha256"]
    ):
        raise RuntimeError("Frozen primary V1 representation SHA mismatch.")

    if scope_lock.get("status") != (
        "06D0_V1_POSTFIT_DOWNSTREAM_SCOPE_AUDIT_FROZEN"
    ):
        raise RuntimeError("06d0 scope lock is not frozen.")
    if scope_lock.get("q10_060_for_v1") != "FORBIDDEN":
        raise RuntimeError("06d0 no longer forbids q10=.60 transfer to V1.")
    if scope_lock.get("q10_040_for_v1") != "FORBIDDEN":
        raise RuntimeError("06d0 no longer forbids q10=.40 transfer to V1.")
    if scope_lock.get("sable_mo_stability_gate_for_v1") != "FORBIDDEN":
        raise RuntimeError("06d0 no longer forbids SABLE-MO stability transfer to V1.")

    stability_scope = str(
        scope_lock.get("generic_target_resampling_stability_status", "")
    )
    pairing_scope = str(
        scope_lock.get("target_pairing_permutations_status", "")
    )
    if not stability_scope.startswith("NOT_AUTHORIZED_FOR_V1_"):
        raise RuntimeError("06d0 V1 resampling-stability boundary changed.")
    if not pairing_scope.startswith("NOT_AUTHORIZED_FOR_V1_"):
        raise RuntimeError("06d0 V1 pairing-permutation boundary changed.")

    if technical_lock.get("status") != (
        "06D1_V2_V1_POSTFREEZE_TECHNICAL_STATUS_AUDIT_FROZEN"
    ):
        raise RuntimeError("06d1_v2 technical-status lock is not frozen.")
    if technical_lock.get("selected_model") != EXPECTED["model_id"]:
        raise RuntimeError("06d1_v2 model mismatch.")
    if int(technical_lock.get("primary_seed")) != EXPECTED["primary_seed"]:
        raise RuntimeError("06d1_v2 primary seed mismatch.")
    if int(technical_lock.get("strict_usable_technical_covariates", -1)) != 0:
        raise RuntimeError("06d1_v2 no longer records zero strict usable technical fields.")
    if technical_lock.get("technical_association_test_performed") is not False:
        raise RuntimeError("06d1_v2 unexpectedly reports a technical association test.")
    if technical_lock.get("technical_status") != EXPECTED["technical_status"]:
        raise RuntimeError("06d1_v2 technical status mismatch.")

    # Pre-biology discipline: do not hash or parse biological files here.
    # Only assert that the already-known paths exist and sizes match frozen provenance.
    biological_file_metadata = {}
    for label, record in BIOLOGICAL_INPUT_IDENTITIES.items():
        path = project_root / record["path"]
        require_file(path)
        observed_size = path.stat().st_size
        if observed_size != int(record["size_bytes"]):
            raise RuntimeError(
                f"{label} size mismatch before biology opening. "
                f"Expected={record['size_bytes']} Observed={observed_size}"
            )
        biological_file_metadata[label] = {
            "path": record["path"],
            "expected_sha256_for_future_execution": record["sha256"],
            "size_bytes": observed_size,
            "content_opened_or_hashed_in_06e0": False,
        }

    contract = {
        "status":
            "FROZEN_BEFORE_V1_BIOLOGICAL_ALIGNMENT_AND_RANDOM_CONTROLS",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),

        "frozen_representation": {
            "model_id": EXPECTED["model_id"],
            "profile_id": EXPECTED["profile_id"],
            "n_cases": EXPECTED["n_cases"],
            "primary_seed": EXPECTED["primary_seed"],
            "shared_dim": EXPECTED["shared_dim"],
            "representation_filename":
                EXPECTED["representation_filename"],
            "representation_sha256":
                EXPECTED["representation_sha256"],
            "coordinate_ids": [
                f"V1_SHARED_{i:02d}"
                for i in range(1, EXPECTED["shared_dim"] + 1)
            ],
            "coordinate_stability_qualification":
                "NOT_ESTABLISHED_FOR_V1",
            "sable_mo_q10_or_retention_transfer":
                "FORBIDDEN",
        },

        "technical_boundary": {
            "status": EXPECTED["technical_status"],
            "strict_usable_technical_covariates": 0,
            "technical_unknown_means_nontechnical": False,
            "technical_unknown_means_biological": False,
            "biological_alignment_may_override_technical_unknown":
                False,
        },

        "primary_biological_programs": {
            "program_ids": EXPECTED["programs"],
            "program_order_is_frozen": True,
            "source": "previously_frozen_Paper4_programs",
            "strict_gene_counts": EXPECTED["strict_gene_counts"],
            "additional_programs_may_be_added_after_results":
                False,
            "hallmark_or_other_annotation_families_authorized_here":
                False,
        },

        "program_score": {
            "primary_variant": EXPECTED["primary_score_variant"],
            "source_weights":
                "Paper4 frozen strict one-to-one human ortholog weights",
            "gene_level_standardization":
                "within_exact_78_case_TARGET_frame_zscore_ddof1",
            "score_definition":
                "mean(sign_weight * standardized_gene_expression) over available strict genes",
            "score_standardization":
                "within_exact_78_case_TARGET_frame_zscore_ddof1",
            "minimum_program_coverage_fraction": 1.0,
            "missing_strict_gene_policy":
                "FAIL_CLOSED; no reduced-gene fallback in the primary analysis",
            "no_outcome_or_clinical_input": True,
        },

        "case_matching": {
            "paper5_case_count_required": 78,
            "paper4_TARGET_expression_rows_may_exceed_78": True,
            "matching_key":
                "normalized TARGET case identifier",
            "required_overlap":
                "all 78 frozen Paper5 cases exactly once",
            "duplicates":
                "FORBIDDEN",
            "missing_case":
                "FAIL_CLOSED",
            "extra_Paper4_cases":
                "ignored after exact 78-case subset is identified",
        },

        "primary_alignment": {
            "per_coordinate_metric":
                "Spearman correlation between frozen program score and frozen V1 shared coordinate",
            "n_coordinates": 5,
            "n_programs": 4,
            "all_20_raw_signed_correlations_reported": True,
            "primary_program_statistic":
                EXPECTED["primary_alignment_metric"],
            "reason_for_absolute_value":
                "V1 latent-axis sign is arbitrary; magnitude is the invariant alignment quantity.",
            "best_coordinate_id_reported":
                "DESCRIPTIVE_ONLY",
            "best_coordinate_may_change_model_or_seed":
                False,
            "best_coordinate_may_be_called_retained_or_stable":
                False,
        },

        "matched_random_program_controls": {
            "enabled": True,
            "panels_per_program":
                EXPECTED["random_panels_per_program"],
            "random_seed": EXPECTED["random_seed"],
            "gene_universe":
                "genes present in the frozen Paper4 TARGET expression file after exact 78-case subset; finite and positive ddof1 variance",
            "exclude_genes":
                "union of strict genes from M34, M40, M11, M24",
            "matching_variable":
                "pre-standardization sample variance across the exact 78-case TARGET frame",
            "variability_bins":
                EXPECTED["variability_bins"],
            "binning_rule":
                "deterministic equal-frequency variability bins with original gene symbol as deterministic tie-break",
            "sampling_rule":
                "same-bin preference; without replacement within each random panel",
            "fallback_rule":
                "if same-bin candidates are insufficient, expand symmetrically to nearest variability bins; record every fallback draw",
            "panel_size":
                "exact strict-gene count of the corresponding frozen program",
            "weight_and_sign_rule":
                "carry the frozen anchor gene sign/weight positions onto the matched sampled genes so each null preserves the anchor score-weight distribution",
            "null_statistic":
                "maximum absolute Spearman correlation across the same five frozen V1 shared coordinates",
            "selection_correction":
                "the null also maximizes over five coordinates, matching the observed best-coordinate search",
            "empirical_probability":
                "(1 + count(null_statistic >= observed_statistic)) / (1000 + 1)",
            "minimum_attainable_probability":
                1.0 / 1001.0,
            "fallback_count_must_be_reported":
                True,
            "matched_random_results_are":
                "specificity/context evidence, not proof that a coordinate is biological",
        },

        "multiplicity": {
            "family":
                "four primary Paper4 program-level matched-random empirical probabilities",
            "procedure":
                "Benjamini-Hochberg",
            "alpha":
                0.05,
            "bh_q_reported_for_each_program":
                True,
            "raw_20_coordinate_correlations":
                "descriptive; not separately thresholded",
        },

        "interpretation_firewall": {
            "biology_may_change_selected_model": False,
            "biology_may_change_primary_seed": False,
            "biology_may_change_fixed_epochs": False,
            "biology_may_apply_SABLE_q10_or_stability_gate_to_V1":
                False,
            "biology_may_rescue_or_reject_V1_coordinates":
                False,
            "strong_alignment_plus_random_control_support_may_be_called":
                "program-aligned V1 coordinate under technical-status-unknown",
            "strong_alignment_may_be_called_stable_biological_factor":
                False,
            "independent_recurrence_required_for_stronger_biological_claim":
                True,
            "clinical_or_outcome_use":
                "FORBIDDEN",
        },

        "biological_input_identities_for_future_execution":
            biological_file_metadata,

        "source_locks": {
            "06c3": {
                "path": rel(results_lock_path, paper5),
                "sha256": sha256_file(results_lock_path),
            },
            "06d0": {
                "path": rel(scope_lock_path, paper5),
                "sha256": sha256_file(scope_lock_path),
            },
            "06d1_v2": {
                "path": rel(technical_lock_path, paper5),
                "sha256": sha256_file(technical_lock_path),
            },
        },

        "scientific_state_at_freeze": {
            "paper4_biological_artifact_contents_opened_in_06e0":
                False,
            "target_biological_annotations_opened_in_06e0":
                False,
            "target_clinical_or_outcomes_opened_in_06e0":
                False,
            "target_molecular_matrices_read_in_06e0":
                False,
            "v1_numeric_representation_array_read_in_06e0":
                False,
            "model_fit_or_refit_in_06e0":
                False,
            "biological_alignment_computed_in_06e0":
                False,
            "random_controls_computed_in_06e0":
                False,
        },

        "next_stage": {
            "required":
                "separate versioned biological runner with synthetic/schema preflight and source authorization before first biological-result computation",
            "future_runner_first_action":
                "verify exact SHA-256 of every frozen biological input before parsing any scientific row/value",
            "post_result_contract_changes":
                "FORBIDDEN; any scientifically necessary amendment requires a new versioned pre-result contract and preservation of this contract",
        },
    }

    out_dir = paper5 / "method_contract" / "06e0"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(
            f"06e0 output directory already exists and is non-empty: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    contract_path = (
        out_dir
        / "v1_biological_alignment_random_control_contract.json"
    )
    write_json(contract_path, contract)

    lock = {
        "status":
            "06E0_V1_BIOLOGICAL_ALIGNMENT_RANDOM_CONTROL_CONTRACT_FROZEN",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "path": rel(contract_path, paper5),
            "sha256": sha256_file(contract_path),
            "size_bytes": contract_path.stat().st_size,
        },
        "selected_model": EXPECTED["model_id"],
        "primary_seed": EXPECTED["primary_seed"],
        "programs": EXPECTED["programs"],
        "random_panels_per_program":
            EXPECTED["random_panels_per_program"],
        "primary_alignment_metric":
            EXPECTED["primary_alignment_metric"],
        "technical_status":
            EXPECTED["technical_status"],
        "sable_mo_q10_stability_transfer_to_v1":
            "FORBIDDEN",
        "paper4_biological_artifact_contents_opened":
            False,
        "target_biological_annotations_opened":
            False,
        "target_clinical_or_outcomes_opened":
            False,
        "target_molecular_matrices_read":
            False,
        "v1_numeric_representation_array_read":
            False,
        "model_fit_or_refit":
            False,
        "biological_alignment_computed":
            False,
        "random_controls_computed":
            False,
    }

    lock_path = (
        out_dir
        / "06e0_v1_biological_alignment_random_control_contract_lock.json"
    )
    write_json(lock_path, lock)

    print("Exact frozen 06c3 identity: PASS")
    print("06d0 V1 scope boundary: PASS")
    print("06d1_v2 technical-status boundary: PASS")
    print("Frozen V1 representation:", EXPECTED["representation_sha256"])
    print("Primary biological programs:", ", ".join(EXPECTED["programs"]))
    print("Primary score variant:", EXPECTED["primary_score_variant"])
    print(
        "Primary alignment statistic:",
        EXPECTED["primary_alignment_metric"],
    )
    print(
        "Matched random programs per anchor:",
        EXPECTED["random_panels_per_program"],
    )
    print("Variability bins:", EXPECTED["variability_bins"])
    print("Random seed:", EXPECTED["random_seed"])
    print("Technical status:", EXPECTED["technical_status"])
    print("SABLE-MO q10/stability transfer to V1: FORBIDDEN")
    print("Paper-4 biological artifact contents opened in 06e0: NO")
    print("TARGET biological annotations opened in 06e0: NO")
    print("TARGET clinical/outcomes opened in 06e0: NO")
    print("TARGET molecular matrices read in 06e0: NO")
    print("V1 numeric representation array read in 06e0: NO")
    print("Biological alignment computed in 06e0: NO")
    print("Random controls computed in 06e0: NO")
    print("=" * 104)
    print(
        "Paper 5 06e0 V1 biological-alignment/random-control contract: FROZEN"
    )
    print("=" * 104)
    print("Contract:", rel(contract_path, paper5))
    print("Lock:", rel(lock_path, paper5))
    print(
        "Next: build/test/source-authorize the exact biological runner "
        "before first biological-result computation."
    )
    print("=" * 104)


if __name__ == "__main__":
    main()
