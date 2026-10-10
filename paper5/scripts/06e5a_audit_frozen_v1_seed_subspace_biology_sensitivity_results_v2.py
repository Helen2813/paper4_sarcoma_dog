#!/usr/bin/env python3
"""
Paper 5 - 06e5a read-only audit of frozen V1 seed/subspace biology sensitivity.

Audits the completed 06e4 scientific sensitivity and its 06e5 result lock.
This stage is READ-ONLY with respect to all scientific results. It may create
only a new audit namespace under method_contract/06e5a_v1.

It does NOT:
- fit/refit any model;
- reopen TARGET outcomes/clinical variables;
- recompute program scores from expression;
- recompute matched-random panels or p/q values;
- change the seed-1103 primary biological result;
- change any 06e4/06e5 scientific file.

No command-line arguments.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = (
    "06e5a-audit-frozen-v1-seed-subspace-biology-sensitivity-results-v2-no-cli"
)
AUDIT_STATUS = (
    "06E5A_V1_FROZEN_SEED_SUBSPACE_BIOLOGY_SENSITIVITY_AUDIT_PASS"
)
LOCK_STATUS = (
    "06E5A_V1_FROZEN_SEED_SUBSPACE_BIOLOGY_SENSITIVITY_AUDIT_LOCKED"
)

EXPECTED = {
    "result_lock_status":
        "06E5_V1_SEED_SUBSPACE_BIOLOGY_SENSITIVITY_FROZEN",
    "runner_sha256":
        "97a8af048e38c16000f27a078ea976af1c894a21850ac2924a05e3b8c16cee86",
    "contract_sha256":
        "cbac52ec634e566ccee070ec0116577388a23d4ceefd881fbe761b55a39e40a6",
    "primary_seed": 1103,
    "seeds": [1103, 2207, 3301],
    "reproducibility_seeds": [2207, 3301],
    "programs": ["M34", "M40", "M11", "M24"],
    "shared_dim": 5,
    "n_cases": 78,
    "primary_corr_sha256":
        "58cadf6ce02603d8d9a05cf06068c9f23e43a0a2dafe8835729407374d89c7f8",
    "primary_align_sha256":
        "a35e24eda531f40d1685f24feea6ae0eecd60f6bc9978feb1c73f5ca0a12b30f",
    "expected_result_files": [
        "v1_program_coordinate_correlations_by_seed.csv",
        "v1_program_alignment_summary_by_seed.csv",
        "v1_program_raw_seed_sensitivity_summary.csv",
        "v1_pairwise_seed_subspace_summary.csv",
        "v1_pairwise_seed_canonical_correlations.csv",
        "v1_pairwise_seed_hungarian_matches.csv",
        "v1_program_procrustes_aligned_correlations.csv",
        "v1_program_procrustes_aligned_summary.csv",
        "v1_seed_subspace_biology_sensitivity_summary.json",
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


def assert_close(
    observed: float,
    expected: float,
    label: str,
    atol: float = 1e-12,
) -> None:
    if not np.isclose(
        float(observed),
        float(expected),
        rtol=0.0,
        atol=atol,
        equal_nan=False,
    ):
        raise RuntimeError(
            f"{label}: {observed!r} != {expected!r} within atol={atol}"
        )


def main() -> None:
    print("=" * 112)
    print("Paper 5 - 06e5a read-only audit of frozen V1 seed/subspace sensitivity")
    print("=" * 112)
    print("Script version:", SCRIPT_VERSION)
    print("Scientific result rewrite: FORBIDDEN")
    print("Model fit/refit: NO")
    print("TARGET outcomes/clinical access: NO")
    print("New p/q values: NO")
    print("Matched-random recomputation: NO")

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    paper5 = project_root / "paper5"

    result_dir = (
        paper5 / "results" / "target_representation"
        / "06e4_seed_subspace_sensitivity_v1"
    )
    result_lock_path = (
        paper5 / "method_contract" / "06e5"
        / "v1_seed_subspace_biology_sensitivity_results_lock.json"
    )
    auth_lock_path = (
        paper5 / "method_contract" / "06e4a"
        / "v1_seed_subspace_sensitivity_runner_source_authorization_lock.json"
    )
    contract_path = (
        paper5 / "method_contract" / "06e3"
        / "v1_seed_subspace_biology_sensitivity_contract.json"
    )
    primary_dir = (
        paper5 / "results" / "target_representation"
        / "06e1_biological_alignment_v3"
    )
    primary_corr_path = primary_dir / "v1_program_coordinate_correlations.csv"
    primary_align_path = primary_dir / "v1_program_alignment_summary.csv"

    for path in [
        result_lock_path,
        auth_lock_path,
        contract_path,
        primary_corr_path,
        primary_align_path,
    ]:
        require_file(path)

    historical_v1_dir = paper5 / "method_contract" / "06e5a_v1"
    historical_v1_files = (
        [p for p in historical_v1_dir.rglob("*") if p.is_file()]
        if historical_v1_dir.exists()
        else []
    )
    if historical_v1_files:
        raise RuntimeError(
            "Historical failed 06e5a_v1 unexpectedly contains files. "
            "Do not delete or overwrite them; audit before proceeding."
        )

    out_dir = paper5 / "method_contract" / "06e5a_v2"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(
            f"06e5a_v2 audit namespace already exists and is non-empty: {out_dir}"
        )

    result_lock = load_json(result_lock_path)
    auth_lock = load_json(auth_lock_path)
    contract = load_json(contract_path)

    # ------------------------------------------------------------------
    # Provenance / frozen-source identity.
    # ------------------------------------------------------------------
    if result_lock.get("status") != EXPECTED["result_lock_status"]:
        raise RuntimeError(
            f"06e5 status mismatch: {result_lock.get('status')!r}"
        )

    runner_source = result_lock.get("runner_source", {})
    if str(runner_source.get("sha256", "")).lower() != EXPECTED["runner_sha256"]:
        raise RuntimeError("06e5 runner SHA mismatch.")

    contract_rec = result_lock.get("06e3_contract", {})
    if str(contract_rec.get("sha256", "")).lower() != EXPECTED["contract_sha256"]:
        raise RuntimeError("06e5 06e3-contract SHA mismatch.")
    if sha256_file(contract_path) != EXPECTED["contract_sha256"]:
        raise RuntimeError("Current 06e3 contract bytes no longer match frozen SHA.")

    auth_rec = result_lock.get("06e4a_authorization", {})
    if str(auth_rec.get("sha256", "")).lower() != sha256_file(auth_lock_path):
        raise RuntimeError("06e5 authorization-lock SHA binding mismatch.")

    if str(
        auth_lock.get("runner_source", {}).get("sha256", "")
    ).lower() != EXPECTED["runner_sha256"]:
        raise RuntimeError("06e4a authorization no longer binds exact runner SHA.")
    if str(
        auth_lock.get("06e3_contract", {}).get("sha256", "")
    ).lower() != EXPECTED["contract_sha256"]:
        raise RuntimeError("06e4a authorization no longer binds exact 06e3 contract.")

    # ------------------------------------------------------------------
    # Verify every frozen result file hash and exact result-file set.
    # ------------------------------------------------------------------
    result_records = result_lock.get("result_files", {})
    if not isinstance(result_records, dict):
        raise RuntimeError("06e5 result_files is not a dictionary.")

    observed_names = sorted(result_records)
    expected_names = sorted(EXPECTED["expected_result_files"])
    if observed_names != expected_names:
        raise RuntimeError(
            "06e5 frozen result-file set mismatch.\n"
            f"Observed: {observed_names}\nExpected: {expected_names}"
        )

    frozen_file_hashes: dict[str, str] = {}
    for filename in EXPECTED["expected_result_files"]:
        rec = result_records[filename]
        path = result_dir / filename
        require_file(path)

        actual_sha = sha256_file(path)
        actual_size = int(path.stat().st_size)

        if str(rec.get("sha256", "")).lower() != actual_sha:
            raise RuntimeError(f"{filename}: SHA mismatch versus 06e5 lock.")
        if int(rec.get("size_bytes", -1)) != actual_size:
            raise RuntimeError(f"{filename}: size mismatch versus 06e5 lock.")

        frozen_file_hashes[filename] = actual_sha

    # ------------------------------------------------------------------
    # Load only the already-frozen 06e4 outputs.
    # ------------------------------------------------------------------
    raw_corr = pd.read_csv(
        result_dir / "v1_program_coordinate_correlations_by_seed.csv"
    )
    raw_summary = pd.read_csv(
        result_dir / "v1_program_alignment_summary_by_seed.csv"
    )
    across = pd.read_csv(
        result_dir / "v1_program_raw_seed_sensitivity_summary.csv"
    )
    subspace = pd.read_csv(
        result_dir / "v1_pairwise_seed_subspace_summary.csv"
    )
    canonical = pd.read_csv(
        result_dir / "v1_pairwise_seed_canonical_correlations.csv"
    )
    hungarian = pd.read_csv(
        result_dir / "v1_pairwise_seed_hungarian_matches.csv"
    )
    aligned_corr = pd.read_csv(
        result_dir / "v1_program_procrustes_aligned_correlations.csv"
    )
    aligned_summary = pd.read_csv(
        result_dir / "v1_program_procrustes_aligned_summary.csv"
    )
    summary = load_json(
        result_dir / "v1_seed_subspace_biology_sensitivity_summary.json"
    )

    # ------------------------------------------------------------------
    # Exact geometry.
    # ------------------------------------------------------------------
    expected_shapes = {
        "raw_corr": (60, 6),
        "raw_summary": (12, 6),
        "across": (4, 9),
        "subspace": (3, 10),
        "canonical": (15, 5),
        "hungarian": (15, 6),
        "aligned_corr": (40, 6),
        "aligned_summary": (8, 6),
    }
    observed_frames = {
        "raw_corr": raw_corr,
        "raw_summary": raw_summary,
        "across": across,
        "subspace": subspace,
        "canonical": canonical,
        "hungarian": hungarian,
        "aligned_corr": aligned_corr,
        "aligned_summary": aligned_summary,
    }
    for name, shape in expected_shapes.items():
        if observed_frames[name].shape != shape:
            raise RuntimeError(
                f"{name}: frozen geometry changed: "
                f"{observed_frames[name].shape} != {shape}"
            )

    if sorted(raw_corr["seed"].astype(int).unique().tolist()) != EXPECTED["seeds"]:
        raise RuntimeError("Raw-correlation seed set changed.")
    if sorted(raw_summary["seed"].astype(int).unique().tolist()) != EXPECTED["seeds"]:
        raise RuntimeError("Raw-summary seed set changed.")
    if raw_corr["program_id"].astype(str).drop_duplicates().tolist() != (
        EXPECTED["programs"]
    ):
        raise RuntimeError("Raw-correlation program order changed.")
    if across["program_id"].astype(str).tolist() != EXPECTED["programs"]:
        raise RuntimeError("Across-seed program order changed.")

    raw_counts = (
        raw_corr.groupby(["program_id", "seed"], sort=False)
        .size()
    )
    if not raw_counts.eq(5).all() or len(raw_counts) != 12:
        raise RuntimeError("Expected exactly 5 coordinates for all 12 program x seed cells.")

    summary_counts = (
        raw_summary.groupby(["program_id", "seed"], sort=False)
        .size()
    )
    if not summary_counts.eq(1).all() or len(summary_counts) != 12:
        raise RuntimeError("Expected exactly one summary row for all 12 program x seed cells.")

    expected_pairs = {(1103, 2207), (1103, 3301), (2207, 3301)}
    observed_pairs = {
        (int(r.reference_seed), int(r.moving_seed))
        for r in subspace.itertuples(index=False)
    }
    if observed_pairs != expected_pairs:
        raise RuntimeError(
            f"Pairwise seed set mismatch: {observed_pairs} != {expected_pairs}"
        )

    # ------------------------------------------------------------------
    # Replay raw per-program maxima from the 60 raw coordinate rows.
    # ------------------------------------------------------------------
    replay_summary_rows: list[dict[str, Any]] = []
    for program in EXPECTED["programs"]:
        for seed in EXPECTED["seeds"]:
            part = raw_corr.loc[
                raw_corr["program_id"].astype(str).eq(program)
                & raw_corr["seed"].astype(int).eq(seed)
            ].copy()
            if part.shape[0] != 5:
                raise RuntimeError(f"{program}/seed{seed}: expected five raw coordinate rows.")

            abs_rho = pd.to_numeric(
                part["absolute_spearman_rho"], errors="raise"
            ).to_numpy(float)
            rho = pd.to_numeric(
                part["spearman_rho"], errors="raise"
            ).to_numpy(float)

            if not np.isfinite(abs_rho).all() or not np.isfinite(rho).all():
                raise RuntimeError(f"{program}/seed{seed}: non-finite raw correlation.")

            best_pos = int(np.argmax(abs_rho))
            best_row = part.iloc[best_pos]

            frozen = raw_summary.loc[
                raw_summary["program_id"].astype(str).eq(program)
                & raw_summary["seed"].astype(int).eq(seed)
            ]
            if frozen.shape[0] != 1:
                raise RuntimeError(f"{program}/seed{seed}: summary row count != 1.")
            fr = frozen.iloc[0]

            assert_close(
                float(fr["observed_max_abs_spearman"]),
                float(abs_rho[best_pos]),
                f"{program}/seed{seed}: max-|rho| replay",
            )
            assert_close(
                float(fr["best_signed_spearman"]),
                float(rho[best_pos]),
                f"{program}/seed{seed}: signed best-rho replay",
            )
            if str(fr["best_coordinate_id"]) != str(best_row["coordinate_id"]):
                raise RuntimeError(
                    f"{program}/seed{seed}: best-coordinate replay mismatch."
                )

            replay_summary_rows.append(
                {
                    "program_id": program,
                    "seed": seed,
                    "max_abs_spearman_replayed": float(abs_rho[best_pos]),
                    "best_coordinate_id_replayed": str(best_row["coordinate_id"]),
                    "best_signed_spearman_replayed": float(rho[best_pos]),
                }
            )

    # ------------------------------------------------------------------
    # Replay the frozen primary 1103 result against original 06e2 files.
    # ------------------------------------------------------------------
    if sha256_file(primary_corr_path) != EXPECTED["primary_corr_sha256"]:
        raise RuntimeError("Frozen 06e2 primary correlation SHA changed.")
    if sha256_file(primary_align_path) != EXPECTED["primary_align_sha256"]:
        raise RuntimeError("Frozen 06e2 primary alignment SHA changed.")

    primary_corr = pd.read_csv(primary_corr_path)
    primary_align = pd.read_csv(primary_align_path)

    sensitivity_1103 = raw_corr.loc[raw_corr["seed"].astype(int).eq(1103)].copy()
    if sensitivity_1103.shape[0] != 20:
        raise RuntimeError("Seed-1103 sensitivity correlation table must have 20 rows.")

    for row in sensitivity_1103.itertuples(index=False):
        frozen = primary_corr.loc[
            primary_corr["program_id"].astype(str).eq(str(row.program_id))
            & primary_corr["coordinate_id"].astype(str).eq(str(row.coordinate_id))
        ]
        if frozen.shape[0] != 1:
            raise RuntimeError(
                f"Primary frozen cell missing: {row.program_id}/{row.coordinate_id}"
            )
        assert_close(
            float(row.spearman_rho),
            float(frozen.iloc[0]["spearman_rho"]),
            f"Primary replay {row.program_id}/{row.coordinate_id}",
        )

    sensitivity_summary_1103 = raw_summary.loc[
        raw_summary["seed"].astype(int).eq(1103)
    ].copy()
    for row in sensitivity_summary_1103.itertuples(index=False):
        frozen = primary_align.loc[
            primary_align["program_id"].astype(str).eq(str(row.program_id))
        ]
        if frozen.shape[0] != 1:
            raise RuntimeError(f"Primary alignment row missing: {row.program_id}")
        fr = frozen.iloc[0]
        assert_close(
            float(row.observed_max_abs_spearman),
            float(fr["observed_max_abs_spearman"]),
            f"Primary max-|rho| replay {row.program_id}",
        )
        assert_close(
            float(row.best_signed_spearman),
            float(fr["best_signed_spearman"]),
            f"Primary signed-rho replay {row.program_id}",
        )
        if str(row.best_coordinate_id) != str(fr["best_coordinate_id"]):
            raise RuntimeError(
                f"Primary best-coordinate replay mismatch: {row.program_id}"
            )

    # ------------------------------------------------------------------
    # Replay across-seed descriptive summaries from the 12 raw maxima.
    # ------------------------------------------------------------------
    for program in EXPECTED["programs"]:
        vals = (
            raw_summary.loc[
                raw_summary["program_id"].astype(str).eq(program),
                ["seed", "observed_max_abs_spearman"],
            ]
            .sort_values("seed")
        )
        if vals.shape[0] != 3:
            raise RuntimeError(f"{program}: expected three raw seed maxima.")
        row = across.loc[across["program_id"].astype(str).eq(program)]
        if row.shape[0] != 1:
            raise RuntimeError(f"{program}: across-seed summary row count != 1.")
        ar = row.iloc[0]
        by_seed = {
            int(r.seed): float(r.observed_max_abs_spearman)
            for r in vals.itertuples(index=False)
        }
        expected_map = {
            "primary_seed_1103_max_abs_spearman": by_seed[1103],
            "seed2207_max_abs_spearman": by_seed[2207],
            "seed3301_max_abs_spearman": by_seed[3301],
            "min_raw_max_abs_spearman": min(by_seed.values()),
            "median_raw_max_abs_spearman": float(np.median(list(by_seed.values()))),
            "max_raw_max_abs_spearman": max(by_seed.values()),
            "range_raw_max_abs_spearman":
                max(by_seed.values()) - min(by_seed.values()),
        }
        for col, expected_value in expected_map.items():
            assert_close(
                float(ar[col]),
                float(expected_value),
                f"{program}: across-seed replay {col}",
            )

    # ------------------------------------------------------------------
    # Replay subspace summaries from canonical table and Hungarian table.
    # ------------------------------------------------------------------
    for ref_seed, moving_seed in sorted(expected_pairs):
        srow = subspace.loc[
            subspace["reference_seed"].astype(int).eq(ref_seed)
            & subspace["moving_seed"].astype(int).eq(moving_seed)
        ]
        crow = canonical.loc[
            canonical["reference_seed"].astype(int).eq(ref_seed)
            & canonical["moving_seed"].astype(int).eq(moving_seed)
        ].sort_values("canonical_index")
        hrow = hungarian.loc[
            hungarian["reference_seed"].astype(int).eq(ref_seed)
            & hungarian["moving_seed"].astype(int).eq(moving_seed)
        ]

        if srow.shape[0] != 1 or crow.shape[0] != 5 or hrow.shape[0] != 5:
            raise RuntimeError(
                f"Seed pair {ref_seed}/{moving_seed}: geometry mismatch."
            )

        sr = srow.iloc[0]
        c = pd.to_numeric(crow["canonical_correlation"], errors="raise").to_numpy(float)
        angles = pd.to_numeric(
            crow["principal_angle_degrees"], errors="raise"
        ).to_numpy(float)
        matched = pd.to_numeric(
            hrow["matched_absolute_pearson"], errors="raise"
        ).to_numpy(float)

        if not (
            np.isfinite(c).all()
            and np.isfinite(angles).all()
            and np.isfinite(matched).all()
        ):
            raise RuntimeError(
                f"Seed pair {ref_seed}/{moving_seed}: non-finite diagnostics."
            )
        if np.any(c < -1e-15) or np.any(c > 1.0 + 1e-12):
            raise RuntimeError(
                f"Seed pair {ref_seed}/{moving_seed}: canonical correlations outside [0,1]."
            )
        expected_angles = np.degrees(np.arccos(np.clip(c, 0.0, 1.0)))
        if not np.allclose(angles, expected_angles, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                f"Seed pair {ref_seed}/{moving_seed}: principal-angle replay mismatch."
            )

        checks = {
            "mean_canonical_correlation": float(np.mean(c)),
            "median_canonical_correlation": float(np.median(c)),
            "minimum_canonical_correlation": float(np.min(c)),
            "maximum_principal_angle_degrees": float(np.max(angles)),
            "mean_abs_hungarian_matched_correlation": float(np.mean(matched)),
            "minimum_abs_hungarian_matched_correlation": float(np.min(matched)),
        }
        for col, expected_value in checks.items():
            assert_close(
                float(sr[col]),
                expected_value,
                f"Seed pair {ref_seed}/{moving_seed}: {col}",
            )

        residual = float(sr["normalized_procrustes_residual"])
        if not math.isfinite(residual) or residual < 0.0:
            raise RuntimeError(
                f"Seed pair {ref_seed}/{moving_seed}: invalid Procrustes residual."
            )

    # 05a-compatible descriptive scalar is median across 3 pairwise means.
    expected_05a = float(
        np.median(
            pd.to_numeric(
                subspace["mean_canonical_correlation"], errors="raise"
            ).to_numpy(float)
        )
    )
    assert_close(
        float(summary["05a_compatible_seed_aligned_subspace_similarity"]),
        expected_05a,
        "05a-compatible median pairwise mean canonical correlation",
    )

    # ------------------------------------------------------------------
    # Replay aligned per-program maxima from the 40 aligned coordinate rows.
    # ------------------------------------------------------------------
    aligned_counts = (
        aligned_corr.groupby(["program_id", "seed"], sort=False).size()
    )
    if len(aligned_counts) != 8 or not aligned_counts.eq(5).all():
        raise RuntimeError("Aligned output must contain 8 program x seed cells x 5 coords.")

    for program in EXPECTED["programs"]:
        for seed in EXPECTED["reproducibility_seeds"]:
            part = aligned_corr.loc[
                aligned_corr["program_id"].astype(str).eq(program)
                & aligned_corr["seed"].astype(int).eq(seed)
            ].copy()
            vals_abs = pd.to_numeric(
                part["absolute_spearman_rho"], errors="raise"
            ).to_numpy(float)
            vals_signed = pd.to_numeric(
                part["spearman_rho"], errors="raise"
            ).to_numpy(float)
            best_pos = int(np.argmax(vals_abs))
            best_row = part.iloc[best_pos]

            frozen = aligned_summary.loc[
                aligned_summary["program_id"].astype(str).eq(program)
                & aligned_summary["seed"].astype(int).eq(seed)
            ]
            if frozen.shape[0] != 1:
                raise RuntimeError(
                    f"Aligned summary row missing: {program}/seed{seed}"
                )
            fr = frozen.iloc[0]

            assert_close(
                float(fr["aligned_max_abs_spearman"]),
                float(vals_abs[best_pos]),
                f"Aligned max-|rho| replay {program}/seed{seed}",
            )
            assert_close(
                float(fr["best_aligned_signed_spearman"]),
                float(vals_signed[best_pos]),
                f"Aligned signed-rho replay {program}/seed{seed}",
            )
            if str(fr["best_aligned_coordinate_id"]) != str(
                best_row["aligned_coordinate_id"]
            ):
                raise RuntimeError(
                    f"Aligned best-coordinate replay mismatch: {program}/seed{seed}"
                )

    # ------------------------------------------------------------------
    # Summary firewalls: must remain closed.
    # ------------------------------------------------------------------
    if int(summary.get("primary_seed", -1)) != EXPECTED["primary_seed"]:
        raise RuntimeError("Frozen summary primary seed changed.")
    if summary.get("reproducibility_only_seeds") != EXPECTED["reproducibility_seeds"]:
        raise RuntimeError("Frozen summary reproducibility seeds changed.")
    if summary.get("programs") != EXPECTED["programs"]:
        raise RuntimeError("Frozen summary program set/order changed.")
    if int(summary.get("n_cases", -1)) != EXPECTED["n_cases"]:
        raise RuntimeError("Frozen summary n_cases changed.")
    if int(summary.get("shared_dim", -1)) != EXPECTED["shared_dim"]:
        raise RuntimeError("Frozen summary shared_dim changed.")
    if summary.get("primary_seed_replay") != "PASS_20_OF_20":
        raise RuntimeError("Frozen summary primary-seed replay is not PASS_20_OF_20.")

    firewalls = summary.get("firewalls", {})
    expected_false = [
        "primary_result_changed",
        "seed_selected_by_biology",
        "seed_ensemble_or_averaging",
        "new_seed_specific_p_or_q",
        "matched_random_panels_recomputed",
        "rotation_invariant_program_to_subspace_statistic_computed",
        "SABLE_q10_or_stability_gate_applied_to_V1",
        "target_outcomes_or_clinical_read",
        "model_fit_or_refit",
        "technical_UNKNOWN_reinterpreted_as_clean",
    ]
    for key in expected_false:
        if firewalls.get(key) is not False:
            raise RuntimeError(f"Frozen summary firewall is not False: {key}")

    # ------------------------------------------------------------------
    # Create a new audit-only namespace after all checks pass.
    # ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=False)

    audit = {
        "status": AUDIT_STATUS,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "historical_v1_attempt": {
            "script_sha256":
                "43dd7f1aa837bbd56876b6bab18b316dde845da87030c39a181fb1ff446ea9f2",
            "failure_boundary":
                "READ_ONLY_GEOMETRY_ASSERTION_BEFORE_AUDIT_OUTPUT_WRITE",
            "failure_reason":
                "v1 incorrectly expected v1_pairwise_seed_subspace_summary.csv "
                "shape (3,9); frozen file has the correct (3,10) geometry "
                "because analysis_role is the tenth column.",
            "scientific_results_changed": False,
            "historical_06e5a_v1_output_files_present":
                bool(historical_v1_files),
        },
        "source_result_lock": {
            "path":
                "method_contract/06e5/"
                "v1_seed_subspace_biology_sensitivity_results_lock.json",
            "sha256": sha256_file(result_lock_path),
            "status": result_lock.get("status"),
        },
        "runner_sha256": EXPECTED["runner_sha256"],
        "06e3_contract_sha256": EXPECTED["contract_sha256"],
        "frozen_result_file_hashes": frozen_file_hashes,
        "checks": {
            "exact_06e5_result_file_set": "PASS",
            "all_06e5_file_hashes_and_sizes": "PASS",
            "raw_program_coordinate_geometry_60_rows": "PASS",
            "all_12_program_seed_cells_present": "PASS",
            "raw_max_abs_spearman_replay_12_of_12": "PASS",
            "primary_seed_1103_replay_against_06e2_20_of_20": "PASS",
            "primary_seed_1103_summary_replay_4_of_4": "PASS",
            "across_seed_summary_replay_4_of_4": "PASS",
            "pairwise_seed_set_3_of_3": "PASS",
            "canonical_correlation_rows_15": "PASS",
            "principal_angle_algebra_replay_15_of_15": "PASS",
            "pairwise_subspace_summary_replay_3_of_3": "PASS",
            "hungarian_summary_replay_3_of_3": "PASS",
            "05a_compatible_subspace_scalar_replay": "PASS",
            "aligned_program_coordinate_geometry_40_rows": "PASS",
            "aligned_program_summary_replay_8_of_8": "PASS",
            "all_scientific_firewalls_closed": "PASS",
        },
        "scientific_state": {
            "scientific_result_rewrite": False,
            "model_fit_or_refit": False,
            "target_outcomes_or_clinical_read": False,
            "new_p_or_q_computed": False,
            "matched_random_panels_recomputed": False,
            "program_scores_recomputed_from_expression": False,
        },
        "interpretation_boundary": {
            "primary_biology":
                "The seed-1103 06e2 matched-random result remains the primary "
                "biological result.",
            "seed_sensitivity":
                "Seeds 2207/3301 remain descriptive post-result sensitivity "
                "using prospectively frozen reproducibility fits.",
            "subspace":
                "Pairwise canonical-correlation/Procrustes/Hungarian diagnostics "
                "describe seed dependence only; no V1 PASS/FAIL retention gate is created.",
            "no_new_inference":
                "No seed-specific significance claim is authorized by this audit.",
        },
    }

    audit_path = (
        out_dir
        / "v1_seed_subspace_biology_sensitivity_readonly_audit_v2.json"
    )
    audit_path.write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )

    lock = {
        "status": LOCK_STATUS,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "audit": {
            "path":
                "method_contract/06e5a_v2/"
                "v1_seed_subspace_biology_sensitivity_readonly_audit_v2.json",
            "sha256": sha256_file(audit_path),
            "size_bytes": audit_path.stat().st_size,
        },
        "script_source": {
            "path":
                "scripts/"
                "06e5a_audit_frozen_v1_seed_subspace_biology_sensitivity_results_v2.py",
            "sha256": sha256_file(script_path),
            "size_bytes": script_path.stat().st_size,
        },
        "source_result_lock": audit["source_result_lock"],
        "rewrite_policy": (
            "Audit-only namespace. Never overwrite or rewrite frozen 06e4/06e5 "
            "scientific results. Never rerun this audit in place after lock creation."
        ),
    }
    lock_path = (
        out_dir / "06e5a_v2_frozen_seed_subspace_result_audit_lock.json"
    )
    lock_path.write_text(
        json.dumps(lock, indent=2) + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Concise manuscript-facing audit summary.
    # ------------------------------------------------------------------
    m34 = across.loc[across["program_id"].astype(str).eq("M34")].iloc[0]

    print("Historical 06e5a_v1 failure boundary: PRESERVED")
    print("Historical 06e5a_v1 audit output files present: NO")
    print("Corrected frozen subspace geometry: PASS [(3,10)]")
    print("Exact 06e5 result lock status: PASS")
    print("Exact frozen result file set: PASS [9/9]")
    print("Frozen result hashes/sizes: PASS [9/9]")
    print("Raw program-coordinate geometry: PASS [60 rows]")
    print("All program x seed cells: PASS [12/12]")
    print("Raw max-|rho| algebra replay: PASS [12/12]")
    print("Primary seed-1103 replay vs frozen 06e2: PASS [20/20 + 4/4]")
    print("Across-seed descriptive summary replay: PASS [4/4]")
    print("Pairwise seed subspaces: PASS [3/3]")
    print("Canonical-correlation rows: PASS [15/15]")
    print("Principal-angle algebra replay: PASS [15/15]")
    print("Pairwise subspace-summary replay: PASS [3/3]")
    print("Hungarian-summary replay: PASS [3/3]")
    print("05a-compatible seed-aligned subspace scalar replay: PASS")
    print("Aligned program sensitivity replay: PASS [8/8]")
    print("All scientific firewalls: PASS")
    print()
    print("-" * 112)
    print("Frozen headline values (audit replay only)")
    print("-" * 112)
    print(
        "M34 raw max-|rho|: "
        f"seed1103={float(m34['primary_seed_1103_max_abs_spearman']):.6f} | "
        f"seed2207={float(m34['seed2207_max_abs_spearman']):.6f} | "
        f"seed3301={float(m34['seed3301_max_abs_spearman']):.6f}"
    )
    print(
        "Median across seed-pair mean canonical correlations: "
        f"{expected_05a:.6f}"
    )
    print("Primary biological result remains seed 1103: YES")
    print("New seed-specific p/q values computed: NO")
    print("Matched-random panels recomputed: NO")
    print("TARGET outcomes/clinical read: NO")
    print("Model fit/refit: NO")
    print("Scientific 06e4/06e5 results rewritten: NO")
    print("=" * 112)
    print("Paper 5 06e5a frozen seed/subspace result read-only audit: PASS + LOCKED")
    print("=" * 112)
    print(
        "Audit: method_contract/06e5a_v2/"
        "v1_seed_subspace_biology_sensitivity_readonly_audit_v2.json"
    )
    print(
        "Lock: method_contract/06e5a_v2/"
        "06e5a_v2_frozen_seed_subspace_result_audit_lock.json"
    )
    print("=" * 112)


if __name__ == "__main__":
    main()
