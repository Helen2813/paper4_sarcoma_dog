#!/usr/bin/env python3
"""
Paper 5 - 06e3 freeze V1 seed/subspace biological-sensitivity contract.

This stage is CONTRACT-ONLY and runs after the frozen primary seed-1103
biological analysis. It prospectively freezes a post-result sensitivity analysis
using only the two reproducibility seeds that were already frozen before the
full-data fits (2207, 3301).

CRITICAL INTERPRETATION
-----------------------
- The primary biological result remains seed 1103 permanently.
- Seeds 2207 and 3301 are reproducibility diagnostics only.
- No seed may replace 1103, be averaged with it, or be selected by downstream
  biology.
- All 4 programs x 3 seeds must be reported regardless of direction.
- No new p-value/q-value/significance threshold is created for the extra seeds.
- No random-control result is recomputed or transferred from seed 1103.
- No rotation-invariant biology statistic (multiple correlation/canonical
  correlation of a program score with the entire 5D span) is authorized here.
- The stage is descriptive sensitivity/provenance, not a new primary analysis.

This contract script reads ONLY metadata locks / JSON. It does NOT open or hash
the numeric latent .npy files and does NOT open the frozen biological score CSV.

No command-line arguments.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = (
    "06e3-freeze-v1-seed-subspace-biology-sensitivity-contract-v1-no-cli"
)

STATUS = "FROZEN_BEFORE_V1_SEED_SUBSPACE_BIOLOGY_SENSITIVITY"

EXPECTED = {
    "model_id": "V1_SHARED_PRIVATE_AE",
    "profile_id": "ARCH_COMPACT__REG_WEAK",
    "n_cases": 78,
    "shared_dim": 5,
    "primary_seed": 1103,
    "reproducibility_seeds": [2207, 3301],
    "all_seeds": [1103, 2207, 3301],
    "programs": ["M34", "M40", "M11", "M24"],
    "06c3_status": "06C3_TARGET_V1_FULL_DATA_REFIT_RESULTS_FROZEN",
    "06e2_status":
        "06E2_V3_V1_BIOLOGICAL_ALIGNMENT_RANDOM_CONTROLS_FROZEN",
    "06e2a_status":
        "06E2A_V3_V2_FROZEN_BIOLOGICAL_RESULT_AUDIT_LOCKED",
    "case_order": {
        "filename": "target_v1_full_data_case_order.csv",
        "sha256":
            "f259778ecfe8a19e83d366615734d0318b14738267af2162ff45f0d240b26f96",
        "size_bytes": 1648,
    },
    "representations": {
        1103: {
            "filename": "target_v1_patient_shared_seed1103.npy",
            "sha256":
                "c05bf08017ad8544e1e4ce0c8f58f086060dd7a6a0a68ed7f8d5a6723768a95f",
            "size_bytes": 3248,
        },
        2207: {
            "filename": "target_v1_patient_shared_seed2207.npy",
            "sha256":
                "08c10fb0c5c27b19955b735e02abee513002439a892ed4d642178afc2cbf2989",
            "size_bytes": 3248,
        },
        3301: {
            "filename": "target_v1_patient_shared_seed3301.npy",
            "sha256":
                "f0c4b28797f1ed4dd4fc06cbdfff1dfc39139bf060cba34a2c18edd139ce4c06",
            "size_bytes": 3248,
        },
    },
    "primary_biology_files": {
        "v1_frozen_program_scores_78.csv": {
            "sha256":
                "7129ad9719d0f01ed12b3cf854c1a4dff2e7295094986e09f08c7344134b00dc",
            "size_bytes": 7651,
        },
        "v1_program_coordinate_correlations.csv": {
            "sha256":
                "58cadf6ce02603d8d9a05cf06068c9f23e43a0a2dafe8835729407374d89c7f8",
            "size_bytes": 2792,
        },
        "v1_program_alignment_summary.csv": {
            "sha256":
                "a35e24eda531f40d1685f24feea6ae0eecd60f6bc9978feb1c73f5ca0a12b30f",
            "size_bytes": 1501,
        },
        "v1_biological_alignment_summary.json": {
            "sha256":
                "e4e68f5f21b2a54db4d0524fdaba265c42972c0fb6654036d9116f5ed3ca8370",
            "size_bytes": 6053,
        },
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
        raise FileNotFoundError(f"Required metadata file missing: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def require_record(
    records: dict[str, Any],
    filename: str,
    expected_sha: str,
    expected_size: int,
    label: str,
) -> None:
    if filename not in records:
        raise RuntimeError(f"{label}: frozen record missing: {filename}")
    rec = records[filename]
    if str(rec.get("sha256", "")).lower() != expected_sha:
        raise RuntimeError(
            f"{label}: SHA mismatch for {filename}: "
            f"{rec.get('sha256')} != {expected_sha}"
        )
    if int(rec.get("size_bytes", -1)) != int(expected_size):
        raise RuntimeError(
            f"{label}: size mismatch for {filename}: "
            f"{rec.get('size_bytes')} != {expected_size}"
        )


def main() -> None:
    print("=" * 108)
    print("Paper 5 - freeze V1 seed/subspace biological-sensitivity contract")
    print("=" * 108)
    print("Script version:", SCRIPT_VERSION)
    print("Numeric V1 latent arrays opened/hashed here: NO")
    print("Frozen biological score rows opened/hashed here: NO")
    print("TARGET molecular matrices read here: NO")
    print("TARGET clinical/outcomes read here: NO")
    print("Model fit/refit here: NO")
    print("Primary biology result changed here: NO")

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    paper5 = project_root / "paper5"

    lock06c3_path = (
        paper5 / "method_contract" / "06c3"
        / "target_v1_full_data_refit_results_lock.json"
    )
    lock06e2_path = (
        paper5 / "method_contract" / "06e2_v3"
        / "v1_biological_alignment_random_controls_results_lock_v3.json"
    )
    lock06e2a_path = (
        paper5 / "method_contract" / "06e2a_v3_v2"
        / "06e2a_v3_v2_frozen_biological_result_audit_lock.json"
    )

    for path in [lock06c3_path, lock06e2_path, lock06e2a_path]:
        require_file(path)

    lock06c3 = load_json(lock06c3_path)
    lock06e2 = load_json(lock06e2_path)
    lock06e2a = load_json(lock06e2a_path)

    if lock06c3.get("status") != EXPECTED["06c3_status"]:
        raise RuntimeError("06c3 frozen-result status mismatch.")
    if lock06e2.get("status") != EXPECTED["06e2_status"]:
        raise RuntimeError("06e2 frozen primary-biology status mismatch.")
    if lock06e2a.get("status") != EXPECTED["06e2a_status"]:
        raise RuntimeError("06e2a read-only audit status mismatch.")

    # ------------------------------------------------------------------
    # Bind exact representation records WITHOUT opening numeric arrays.
    # ------------------------------------------------------------------
    rep_records = lock06c3.get("result_files", {})
    if not isinstance(rep_records, dict):
        raise RuntimeError("06c3 result_files missing.")

    case = EXPECTED["case_order"]
    require_record(
        rep_records,
        case["filename"],
        case["sha256"],
        case["size_bytes"],
        "06c3",
    )

    for seed in EXPECTED["all_seeds"]:
        rec = EXPECTED["representations"][seed]
        require_record(
            rep_records,
            rec["filename"],
            rec["sha256"],
            rec["size_bytes"],
            "06c3",
        )

    # ------------------------------------------------------------------
    # Bind exact frozen primary-biology files WITHOUT opening score rows.
    # ------------------------------------------------------------------
    bio_records = lock06e2.get("result_files", {})
    if not isinstance(bio_records, dict):
        raise RuntimeError("06e2 result_files missing.")

    for filename, expected in EXPECTED["primary_biology_files"].items():
        require_record(
            bio_records,
            filename,
            expected["sha256"],
            expected["size_bytes"],
            "06e2",
        )

    if lock06e2.get("selected_model") != EXPECTED["model_id"]:
        raise RuntimeError("06e2 selected model mismatch.")
    if int(lock06e2.get("primary_seed", -1)) != EXPECTED["primary_seed"]:
        raise RuntimeError("06e2 primary seed mismatch.")
    if lock06e2.get("programs") != EXPECTED["programs"]:
        raise RuntimeError("06e2 primary program set/order mismatch.")
    if lock06e2.get("technical_status") != (
        "UNKNOWN_NO_STRICT_USABLE_TECHNICAL_COVARIATES"
    ):
        raise RuntimeError("06e2 technical-status boundary changed.")

    # The read-only audit must bind back to the same 06e2 result lock.
    source_lock = lock06e2a.get("source_result_lock", {})
    if str(source_lock.get("sha256", "")).lower() != sha256_file(lock06e2_path):
        raise RuntimeError(
            "06e2a audit does not bind the current frozen 06e2 result lock."
        )

    # ------------------------------------------------------------------
    # Freeze the post-result sensitivity design BEFORE numeric arrays are read.
    # ------------------------------------------------------------------
    contract = {
        "status": STATUS,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),

        "analysis_role": (
            "POST_RESULT_SENSITIVITY_USING_PROSPECTIVELY_FROZEN_"
            "REPRODUCIBILITY_SEEDS"
        ),

        "primary_result_boundary": {
            "primary_seed": 1103,
            "primary_result_remains_seed_1103": True,
            "primary_result_may_be_replaced_by_seed_sensitivity": False,
            "seed_selection_by_biology": "FORBIDDEN",
            "seed_selection_by_subspace_similarity": "FORBIDDEN",
            "seed_averaging_or_ensemble": "FORBIDDEN",
            "primary_06e2_empirical_p_and_BH_q_remain_authoritative": True,
            "new_seed_specific_empirical_p_or_q": "FORBIDDEN_IN_THIS_STAGE",
        },

        "frozen_inputs": {
            "n_cases": EXPECTED["n_cases"],
            "shared_dim": EXPECTED["shared_dim"],
            "model_id": EXPECTED["model_id"],
            "profile_id": EXPECTED["profile_id"],
            "case_order": EXPECTED["case_order"],
            "representations": {
                str(seed): EXPECTED["representations"][seed]
                for seed in EXPECTED["all_seeds"]
            },
            "program_score_file":
                EXPECTED["primary_biology_files"][
                    "v1_frozen_program_scores_78.csv"
                ],
            "primary_coordinate_correlation_file":
                EXPECTED["primary_biology_files"][
                    "v1_program_coordinate_correlations.csv"
                ],
            "programs": EXPECTED["programs"],
        },

        "seed_roles": {
            "1103": "PRIMARY_FROZEN_SCIENTIFIC_REPRESENTATION",
            "2207": "MANDATORY_REPRODUCIBILITY_ONLY",
            "3301": "MANDATORY_REPRODUCIBILITY_ONLY",
            "report_all_seeds": True,
            "report_all_program_seed_cells": True,
        },

        "pairwise_subspace_diagnostics": {
            "seed_pairs": [
                [1103, 2207],
                [1103, 3301],
                [2207, 3301],
            ],
            "input_preparation": (
                "For each frozen 78x5 latent matrix, subtract the column mean. "
                "No outcome/biology-dependent transformation and no refit."
            ),
            "rank_requirement": (
                "Each centered latent matrix must have numerical rank 5 using "
                "the standard SVD machine-epsilon rank criterion. Fail closed "
                "if rank < 5."
            ),
            "canonical_correlations": (
                "Compute orthonormal bases for the centered 5D column spaces "
                "using thin SVD. Compute the five singular values of Q_A^T Q_B, "
                "clip only floating-point spillover to [0,1], and report all "
                "five values in descending order."
            ),
            "principal_angles_degrees": (
                "For each canonical correlation c_j, report "
                "acos(c_j)*180/pi."
            ),
            "subspace_summary_per_pair": [
                "mean_canonical_correlation",
                "median_canonical_correlation",
                "minimum_canonical_correlation",
                "maximum_principal_angle_degrees",
            ],
            "05a_compatibility_summary": (
                "Report the median across the three seed pairs of the pairwise "
                "mean canonical correlation. This is descriptive consistency "
                "with the already-frozen 05a seed_aligned_subspace_similarity "
                "concept; no V1 retention threshold is created or imported."
            ),
            "threshold_or_PASS_FAIL_gate": "FORBIDDEN",
        },

        "procrustes_hungarian_diagnostics": {
            "role": "DESCRIPTIVE_COORDINATE_ORIENTATION_DIAGNOSTIC",
            "reference_direction": (
                "For pairs involving seed 1103, 1103 is the reference. "
                "For pair 2207/3301, 2207 is the reference."
            ),
            "procrustes": (
                "On column-centered latent matrices A (moving) and B "
                "(reference), solve orthogonal Procrustes by SVD of A^T B; "
                "R=U V^T and A_aligned=A R. Report normalized residual "
                "||A_aligned-B||_F / ||B||_F. No scaling fit beyond the "
                "orthogonal rotation/reflection."
            ),
            "hungarian": (
                "After Procrustes alignment, compute the 5x5 Pearson "
                "correlation matrix between aligned moving coordinates and "
                "reference coordinates. Hungarian-match coordinates by "
                "maximizing absolute correlation. Report all five signed and "
                "absolute matched correlations plus their mean/minimum."
            ),
            "sign_flip": (
                "For descriptive aligned-coordinate reporting only, a matched "
                "moving coordinate may be sign-flipped to make its matched "
                "correlation nonnegative. Raw latent arrays are never changed."
            ),
            "threshold_or_PASS_FAIL_gate": "FORBIDDEN",
        },

        "program_seed_sensitivity": {
            "programs": EXPECTED["programs"],
            "seeds": EXPECTED["all_seeds"],
            "n_cells": 12,
            "score_source": (
                "Reuse the exact frozen 78-case strict_signed_mean program "
                "scores from the 06e2 primary analysis; do not reopen Paper-4 "
                "expression or recompute program scores."
            ),
            "raw_seed_statistic": (
                "For each program x seed, compute Spearman correlation between "
                "the frozen program score and each of the five RAW frozen V1 "
                "shared coordinates. Primary descriptive statistic is the "
                "maximum absolute Spearman across the same five coordinates."
            ),
            "best_coordinate": (
                "Report the coordinate attaining max absolute Spearman and its "
                "signed correlation. Coordinate labels across seeds are "
                "descriptive only and are not assumed comparable."
            ),
            "primary_seed_replay": (
                "The seed-1103 4x5 raw correlations and four max-|rho| values "
                "must replay the already-frozen 06e2 primary results exactly "
                "within absolute tolerance 1e-12 before any sensitivity output "
                "is accepted."
            ),
            "across_seed_descriptive_summary": [
                "min_raw_max_abs_spearman",
                "median_raw_max_abs_spearman",
                "max_raw_max_abs_spearman",
                "range_raw_max_abs_spearman",
            ],
            "procrustes_oriented_sensitivity": (
                "For seeds 2207 and 3301 only, additionally Procrustes-align "
                "their centered 5D latent coordinates to centered seed 1103 "
                "using the frozen rule above, then recompute program-to-"
                "coordinate Spearman correlations and max-|rho|. This is an "
                "explicitly exploratory orientation diagnostic and cannot "
                "replace the raw-coordinate primary statistic."
            ),
            "new_significance_threshold": "FORBIDDEN",
            "new_empirical_p_or_BH_q": "FORBIDDEN",
            "claim_upgrade_rule": (
                "No binary 'robust/not robust' threshold is defined. Report "
                "the complete numerical sensitivity pattern regardless of "
                "direction."
            ),
        },

        "explicitly_not_authorized": [
            "refit any representation model",
            "select or replace the primary seed",
            "average or ensemble latent representations",
            "recompute matched random gene panels",
            "transfer seed-1103 random-control p/q values to other seeds",
            "create new seed-specific p-values or q-values",
            "apply SABLE-MO q10/stability thresholds to V1",
            "declare V1 coordinates retained/stable from these diagnostics",
            "use technical UNKNOWN as evidence of no technical confounding",
            "compute rotation-invariant program-to-subspace association "
            "(multiple correlation/canonical correlation) in this stage",
            "open TARGET outcomes or clinical variables",
            "change the frozen primary M34/M40/M11/M24 interpretation",
        ],

        "reporting_language": {
            "required": (
                "Describe this as a post-result sensitivity analysis using "
                "prospectively frozen reproducibility seeds."
            ),
            "forbidden": [
                "pre-specified primary biological analysis",
                "seed-validated primary result",
                "stable biological factor",
                "retained V1 factor",
            ],
        },

        "source_metadata": {
            "06c3_lock": {
                "path":
                    "method_contract/06c3/"
                    "target_v1_full_data_refit_results_lock.json",
                "sha256": sha256_file(lock06c3_path),
                "status": lock06c3.get("status"),
            },
            "06e2_primary_biology_lock": {
                "path":
                    "method_contract/06e2_v3/"
                    "v1_biological_alignment_random_controls_results_lock_v3.json",
                "sha256": sha256_file(lock06e2_path),
                "status": lock06e2.get("status"),
            },
            "06e2a_readonly_audit_lock": {
                "path":
                    "method_contract/06e2a_v3_v2/"
                    "06e2a_v3_v2_frozen_biological_result_audit_lock.json",
                "sha256": sha256_file(lock06e2a_path),
                "status": lock06e2a.get("status"),
            },
        },

        "scientific_state_at_contract_freeze": {
            "numeric_latent_arrays_read_here": False,
            "biological_score_rows_read_here": False,
            "seed2207_biological_result_seen_here": False,
            "seed3301_biological_result_seen_here": False,
            "model_fit_or_refit": False,
            "outcome_or_clinical_access": False,
        },
    }

    out_dir = paper5 / "method_contract" / "06e3"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(
            f"06e3 output directory already exists and is non-empty: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    contract_path = (
        out_dir / "v1_seed_subspace_biology_sensitivity_contract.json"
    )
    contract_path.write_text(
        json.dumps(contract, indent=2) + "\n",
        encoding="utf-8",
    )

    lock = {
        "status":
            "06E3_V1_SEED_SUBSPACE_BIOLOGY_SENSITIVITY_CONTRACT_FROZEN",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "path":
                "method_contract/06e3/"
                "v1_seed_subspace_biology_sensitivity_contract.json",
            "sha256": sha256_file(contract_path),
            "size_bytes": contract_path.stat().st_size,
        },
        "script_source": {
            "path":
                "scripts/"
                "06e3_freeze_v1_seed_subspace_biology_sensitivity_contract_v1.py",
            "sha256": sha256_file(script_path),
            "size_bytes": script_path.stat().st_size,
        },
        "source_metadata_locks": contract["source_metadata"],
        "rewrite_policy": (
            "06e3 is frozen before any seed-2207/3301 biological sensitivity "
            "or pairwise seed-subspace result is read. Do not rewrite it in "
            "response to those results. Any scientific change requires a new "
            "versioned amendment."
        ),
    }

    lock_path = (
        out_dir / "06e3_v1_seed_subspace_biology_sensitivity_contract_lock.json"
    )
    lock_path.write_text(
        json.dumps(lock, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Exact 06c3 representation metadata: PASS")
    print("  case order: PASS")
    print("  seed 1103 representation identity: PASS")
    print("  seed 2207 representation identity: PASS")
    print("  seed 3301 representation identity: PASS")
    print("Exact frozen 06e2 primary-biology metadata: PASS")
    print("Exact 06e2a read-only audit provenance: PASS")
    print("Primary seed remains 1103: YES")
    print("Reproducibility-only seeds: [2207, 3301]")
    print("All 4 programs x 3 seeds reporting required: YES")
    print("Pairwise seed-subspace diagnostics frozen: YES")
    print("Procrustes/Hungarian diagnostics frozen: YES")
    print("New seed-specific p/q values: FORBIDDEN")
    print("Rotation-invariant program-to-subspace biology statistic: FORBIDDEN")
    print("SABLE-MO stability/q10 transfer to V1: FORBIDDEN")
    print("Numeric V1 latent arrays opened/hashed here: NO")
    print("Frozen biological score rows opened/hashed here: NO")
    print("TARGET clinical/outcomes read here: NO")
    print("Model fit/refit here: NO")
    print("=" * 108)
    print("Paper 5 06e3 V1 seed/subspace sensitivity contract: FROZEN")
    print("=" * 108)
    print(
        "Contract: method_contract/06e3/"
        "v1_seed_subspace_biology_sensitivity_contract.json"
    )
    print(
        "Lock: method_contract/06e3/"
        "06e3_v1_seed_subspace_biology_sensitivity_contract_lock.json"
    )
    print(
        "Next: build a separately source-authorized read-only runner that "
        "consumes this exact contract."
    )
    print("=" * 108)


if __name__ == "__main__":
    main()
