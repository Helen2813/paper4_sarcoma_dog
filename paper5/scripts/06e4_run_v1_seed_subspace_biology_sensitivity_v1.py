#!/usr/bin/env python3
"""
Paper 5 - 06e4 V1 seed/subspace biological-sensitivity runner v1.

Two-stage execution:
1) Before source authorization exists, this exact file can ONLY run a synthetic
   preauthorization self-test. It does not open any real V1 latent array or
   biological score row.
2) After a separate 06e4a source-authorization lock binds this exact runner SHA,
   this same unchanged file performs the frozen 06e3 read-only sensitivity.

No model fitting/refitting. No outcomes/clinical variables. No new p/q values.
Primary biological result remains seed 1103.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


SCRIPT_VERSION = (
    "06e4-run-v1-seed-subspace-biology-sensitivity-v1-"
    "two-stage-source-authorized-readonly-no-cli"
)

PREAUTH_STATUS = "06E4_V1_SEED_SUBSPACE_RUNNER_SYNTHETIC_SELFTEST_PASS"
AUTH_STATUS = "06E4A_V1_SEED_SUBSPACE_RUNNER_SOURCE_AUTHORIZED"
RESULT_STATUS = "06E5_V1_SEED_SUBSPACE_BIOLOGY_SENSITIVITY_FROZEN"

EXPECTED = {
    "contract_status":
        "FROZEN_BEFORE_V1_SEED_SUBSPACE_BIOLOGY_SENSITIVITY",
    "contract_lock_status":
        "06E3_V1_SEED_SUBSPACE_BIOLOGY_SENSITIVITY_CONTRACT_FROZEN",
    "primary_seed": 1103,
    "reproducibility_seeds": [2207, 3301],
    "all_seeds": [1103, 2207, 3301],
    "programs": ["M34", "M40", "M11", "M24"],
    "n_cases": 78,
    "shared_dim": 5,
    "case_order_sha256":
        "f259778ecfe8a19e83d366615734d0318b14738267af2162ff45f0d240b26f96",
    "representation_sha256": {
        1103:
            "c05bf08017ad8544e1e4ce0c8f58f086060dd7a6a0a68ed7f8d5a6723768a95f",
        2207:
            "08c10fb0c5c27b19955b735e02abee513002439a892ed4d642178afc2cbf2989",
        3301:
            "f0c4b28797f1ed4dd4fc06cbdfff1dfc39139bf060cba34a2c18edd139ce4c06",
    },
    "program_scores_sha256":
        "7129ad9719d0f01ed12b3cf854c1a4dff2e7295094986e09f08c7344134b00dc",
    "primary_correlations_sha256":
        "58cadf6ce02603d8d9a05cf06068c9f23e43a0a2dafe8835729407374d89c7f8",
    "primary_alignment_summary_sha256":
        "a35e24eda531f40d1685f24feea6ae0eecd60f6bc9978feb1c73f5ca0a12b30f",
    "primary_biology_summary_sha256":
        "e4e68f5f21b2a54db4d0524fdaba265c42972c0fb6654036d9116f5ed3ca8370",
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
        raise FileNotFoundError(f"Required file missing: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def average_ranks(x: np.ndarray) -> np.ndarray:
    return (
        pd.Series(np.asarray(x, dtype=np.float64))
        .rank(method="average", na_option="keep")
        .to_numpy(dtype=np.float64)
    )


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise RuntimeError("Spearman vectors have incompatible shapes.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("Spearman vector contains non-finite values.")
    rx = average_ranks(x)
    ry = average_ranks(y)
    value = float(np.corrcoef(rx, ry)[0, 1])
    if not math.isfinite(value):
        raise RuntimeError("Spearman correlation is non-finite.")
    return value


def pearson_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise RuntimeError("Pearson matrix inputs have incompatible shapes.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError("Pearson matrix input is non-finite.")
    out = np.empty((a.shape[1], b.shape[1]), dtype=np.float64)
    for i in range(a.shape[1]):
        for j in range(b.shape[1]):
            x = a[:, i]
            y = b[:, j]
            sx = float(np.std(x, ddof=1))
            sy = float(np.std(y, ddof=1))
            if sx <= 0.0 or sy <= 0.0:
                raise RuntimeError("Zero-variance coordinate in Pearson matrix.")
            out[i, j] = float(np.corrcoef(x, y)[0, 1])
    return out


def center_latent(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if z.shape != (EXPECTED["n_cases"], EXPECTED["shared_dim"]):
        raise RuntimeError(f"Unexpected latent shape: {z.shape}")
    if not np.isfinite(z).all():
        raise RuntimeError("Latent representation contains non-finite values.")
    return z - np.mean(z, axis=0, keepdims=True)


def svd_rank_and_basis(z_centered: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    u, s, _ = np.linalg.svd(z_centered, full_matrices=False)
    if s.size == 0:
        raise RuntimeError("Empty singular-value vector.")
    tol = float(s[0]) * max(z_centered.shape) * np.finfo(np.float64).eps
    rank = int(np.sum(s > tol))
    return rank, u[:, :rank], s


def canonical_diagnostics(
    a_centered: np.ndarray,
    b_centered: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rank_a, qa, _ = svd_rank_and_basis(a_centered)
    rank_b, qb, _ = svd_rank_and_basis(b_centered)
    if rank_a != EXPECTED["shared_dim"] or rank_b != EXPECTED["shared_dim"]:
        raise RuntimeError(
            f"Latent rank requirement failed: rank_a={rank_a}, rank_b={rank_b}."
        )
    values = np.linalg.svd(qa.T @ qb, compute_uv=False)
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    if values.shape != (EXPECTED["shared_dim"],):
        raise RuntimeError("Unexpected canonical-correlation vector length.")
    angles = np.degrees(np.arccos(values))
    return values, angles


def procrustes_align(
    moving_centered: np.ndarray,
    reference_centered: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    cross = moving_centered.T @ reference_centered
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    rotation = u @ vt
    aligned = moving_centered @ rotation
    denom = float(np.linalg.norm(reference_centered, ord="fro"))
    if denom <= 0.0:
        raise RuntimeError("Reference Procrustes norm is zero.")
    residual = float(
        np.linalg.norm(aligned - reference_centered, ord="fro") / denom
    )
    return aligned, rotation, residual


def hungarian_match(
    aligned_moving: np.ndarray,
    reference_centered: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corr = pearson_matrix(aligned_moving, reference_centered)
    row_ind, col_ind = linear_sum_assignment(-np.abs(corr))
    if sorted(row_ind.tolist()) != list(range(EXPECTED["shared_dim"])):
        raise RuntimeError("Hungarian row assignment is incomplete.")
    if sorted(col_ind.tolist()) != list(range(EXPECTED["shared_dim"])):
        raise RuntimeError("Hungarian column assignment is incomplete.")

    # Reorder matches by reference coordinate for deterministic reporting.
    order = np.argsort(col_ind, kind="mergesort")
    row_ind = row_ind[order]
    col_ind = col_ind[order]
    matched_signed = corr[row_ind, col_ind]
    return row_ind, col_ind, matched_signed


def raw_program_correlations(
    scores: pd.DataFrame,
    latent: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for program in EXPECTED["programs"]:
        score_col = f"{program}__strict_signed_mean_z"
        if score_col not in scores.columns:
            raise RuntimeError(f"Program score column missing: {score_col}")
        score = scores[score_col].to_numpy(dtype=np.float64)

        vals: list[float] = []
        for j in range(EXPECTED["shared_dim"]):
            rho = spearman_rho(score, latent[:, j])
            vals.append(rho)
            rows.append(
                {
                    "program_id": program,
                    "seed": int(seed),
                    "coordinate_id": f"V1_SHARED_{j+1:02d}",
                    "spearman_rho": rho,
                    "absolute_spearman_rho": abs(rho),
                    "analysis_role":
                        "PRIMARY_SEED_REPLAY"
                        if seed == EXPECTED["primary_seed"]
                        else "POST_RESULT_REPRODUCIBILITY_SENSITIVITY",
                }
            )

        abs_vals = np.abs(np.asarray(vals, dtype=np.float64))
        best_index = int(np.argmax(abs_vals))
        summaries.append(
            {
                "program_id": program,
                "seed": int(seed),
                "observed_max_abs_spearman": float(abs_vals[best_index]),
                "best_coordinate_id": f"V1_SHARED_{best_index+1:02d}",
                "best_signed_spearman": float(vals[best_index]),
                "analysis_role":
                    "PRIMARY_SEED_REPLAY"
                    if seed == EXPECTED["primary_seed"]
                    else "POST_RESULT_REPRODUCIBILITY_SENSITIVITY",
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(summaries)


def aligned_program_correlations(
    scores: pd.DataFrame,
    aligned: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for program in EXPECTED["programs"]:
        score = scores[f"{program}__strict_signed_mean_z"].to_numpy(
            dtype=np.float64
        )
        vals: list[float] = []
        for j in range(EXPECTED["shared_dim"]):
            rho = spearman_rho(score, aligned[:, j])
            vals.append(rho)
            rows.append(
                {
                    "program_id": program,
                    "seed": int(seed),
                    "aligned_coordinate_id":
                        f"ALIGNED_TO_1103_SHARED_{j+1:02d}",
                    "spearman_rho": rho,
                    "absolute_spearman_rho": abs(rho),
                    "analysis_role":
                        "EXPLORATORY_PROCRUSTES_ORIENTATION_DIAGNOSTIC",
                }
            )
        abs_vals = np.abs(np.asarray(vals, dtype=np.float64))
        best_index = int(np.argmax(abs_vals))
        summaries.append(
            {
                "program_id": program,
                "seed": int(seed),
                "aligned_max_abs_spearman": float(abs_vals[best_index]),
                "best_aligned_coordinate_id":
                    f"ALIGNED_TO_1103_SHARED_{best_index+1:02d}",
                "best_aligned_signed_spearman": float(vals[best_index]),
                "analysis_role":
                    "EXPLORATORY_PROCRUSTES_ORIENTATION_DIAGNOSTIC",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def run_synthetic_selftest(
    runner_sha: str,
    contract_sha: str,
    preflight_path: Path,
) -> None:
    rng = np.random.default_rng(60401)

    reference = rng.normal(size=(78, 5)).astype(np.float64)
    perm = np.array([4, 2, 0, 3, 1], dtype=int)
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float64)
    perm_signed = reference[:, perm] * signs

    # A genuinely rotated version of the same 5D coordinate span.
    q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    rotated = reference @ q

    ref_c = center_latent(reference)
    ps_c = center_latent(perm_signed)
    rot_c = center_latent(rotated)

    can_ps, ang_ps = canonical_diagnostics(ref_c, ps_c)
    can_rot, ang_rot = canonical_diagnostics(ref_c, rot_c)

    if float(np.min(can_ps)) < 1.0 - 1e-10:
        raise RuntimeError("Synthetic permutation/sign subspace test failed.")
    if float(np.min(can_rot)) < 1.0 - 1e-10:
        raise RuntimeError("Synthetic rotation subspace test failed.")

    aligned_ps, _, residual_ps = procrustes_align(ps_c, ref_c)
    aligned_rot, _, residual_rot = procrustes_align(rot_c, ref_c)
    if residual_ps > 1e-10 or residual_rot > 1e-10:
        raise RuntimeError("Synthetic Procrustes recovery test failed.")

    ri, ci, matched = hungarian_match(aligned_ps, ref_c)
    if float(np.min(np.abs(matched))) < 1.0 - 1e-10:
        raise RuntimeError("Synthetic Hungarian matching test failed.")

    # Spearman sign/permutation invariance of max-|rho|.
    score = rng.normal(size=78)
    raw_ref = np.array(
        [spearman_rho(score, reference[:, j]) for j in range(5)]
    )
    raw_ps = np.array(
        [spearman_rho(score, perm_signed[:, j]) for j in range(5)]
    )
    if abs(float(np.max(np.abs(raw_ref))) - float(np.max(np.abs(raw_ps)))) > 1e-12:
        raise RuntimeError(
            "Synthetic max-|rho| sign/permutation invariance test failed."
        )

    # Rotation sensitivity should generally be observable for this fixed fixture.
    raw_rot = np.array(
        [spearman_rho(score, rotated[:, j]) for j in range(5)]
    )
    rotation_difference = abs(
        float(np.max(np.abs(raw_ref))) - float(np.max(np.abs(raw_rot)))
    )
    if rotation_difference <= 1e-5:
        raise RuntimeError(
            "Synthetic fixture failed to demonstrate coordinate-statistic "
            "rotation sensitivity."
        )

    # Program x seed table geometry on synthetic values.
    fake_scores = pd.DataFrame(
        {
            "case_id": [f"SYNTH-{i:03d}" for i in range(78)],
            **{
                f"{p}__strict_signed_mean_z":
                    rng.normal(size=78)
                for p in EXPECTED["programs"]
            },
        }
    )
    corr, summary = raw_program_correlations(fake_scores, reference, 1103)
    if corr.shape[0] != 20 or summary.shape[0] != 4:
        raise RuntimeError("Synthetic program-correlation geometry failed.")

    report = {
        "status": PREAUTH_STATUS,
        "script_version": SCRIPT_VERSION,
        "runner_sha256": runner_sha,
        "contract_sha256": contract_sha,
        "execution_mode": "SYNTHETIC_ONLY_PREAUTHORIZATION",
        "created_at_utc": utc_now_iso(),
        "real_numeric_latents_opened": False,
        "real_biological_score_rows_opened": False,
        "target_outcomes_or_clinical_read": False,
        "model_fit_or_refit": False,
        "tests": {
            "canonical_correlation_sign_permutation_invariance": "PASS",
            "canonical_correlation_rotation_invariance": "PASS",
            "principal_angle_sign_permutation_fixture_max_degrees":
                float(np.max(ang_ps)),
            "principal_angle_rotation_fixture_max_degrees":
                float(np.max(ang_rot)),
            "procrustes_sign_permutation_recovery": "PASS",
            "procrustes_rotation_recovery": "PASS",
            "hungarian_coordinate_matching": "PASS",
            "max_abs_spearman_sign_permutation_invariance": "PASS",
            "max_abs_spearman_rotation_sensitivity_fixture": "PASS",
            "rotation_fixture_max_abs_spearman_difference":
                rotation_difference,
            "synthetic_program_coordinate_rows": int(corr.shape[0]),
            "synthetic_program_summary_rows": int(summary.shape[0]),
        },
    }

    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(preflight_path, report)

    print("Synthetic canonical-correlation invariance: PASS")
    print("Synthetic Procrustes recovery: PASS")
    print("Synthetic Hungarian matching: PASS")
    print("Synthetic max-|rho| sign/permutation invariance: PASS")
    print("Synthetic max-|rho| rotation-sensitivity fixture: PASS")
    print("Real numeric V1 latent arrays opened: NO")
    print("Real biological score rows opened: NO")
    print("TARGET outcomes/clinical read: NO")
    print("Model fit/refit: NO")
    print("=" * 108)
    print("Paper 5 06e4 seed/subspace runner synthetic self-test: PASS")
    print("=" * 108)
    print(
        "Preflight: method_contract/06e4_preflight/"
        "v1_seed_subspace_runner_synthetic_selftest_v1.json"
    )
    print(
        "Next: freeze this exact runner source with a separate 06e4a "
        "source-authorization stage. Do not edit this runner."
    )
    print("=" * 108)


def main() -> None:
    print("=" * 108)
    print("Paper 5 - V1 seed/subspace biological-sensitivity runner")
    print("=" * 108)
    print("Script version:", SCRIPT_VERSION)

    script_path = Path(__file__).resolve()
    runner_sha = sha256_file(script_path)
    project_root = script_path.parents[2]
    paper5 = project_root / "paper5"

    contract_path = (
        paper5 / "method_contract" / "06e3"
        / "v1_seed_subspace_biology_sensitivity_contract.json"
    )
    contract_lock_path = (
        paper5 / "method_contract" / "06e3"
        / "06e3_v1_seed_subspace_biology_sensitivity_contract_lock.json"
    )
    preflight_path = (
        paper5 / "method_contract" / "06e4_preflight"
        / "v1_seed_subspace_runner_synthetic_selftest_v1.json"
    )
    auth_path = (
        paper5 / "method_contract" / "06e4a"
        / "v1_seed_subspace_sensitivity_runner_source_authorization_lock.json"
    )
    result_dir = (
        paper5 / "results" / "target_representation"
        / "06e4_seed_subspace_sensitivity_v1"
    )
    result_lock_dir = paper5 / "method_contract" / "06e5"

    require_file(contract_path)
    require_file(contract_lock_path)
    contract = load_json(contract_path)
    contract_lock = load_json(contract_lock_path)

    if contract.get("status") != EXPECTED["contract_status"]:
        raise RuntimeError("06e3 contract status mismatch.")
    if contract_lock.get("status") != EXPECTED["contract_lock_status"]:
        raise RuntimeError("06e3 contract-lock status mismatch.")
    if str(
        contract_lock.get("contract", {}).get("sha256", "")
    ).lower() != sha256_file(contract_path):
        raise RuntimeError("06e3 contract hash does not match its lock.")

    contract_sha = sha256_file(contract_path)

    print("Runner SHA-256:", runner_sha)
    print("06e3 contract SHA-256:", contract_sha)

    # ------------------------------------------------------------------
    # PREAUTHORIZATION MODE: no source auth exists yet.
    # ------------------------------------------------------------------
    if not auth_path.exists():
        print("Execution mode: SYNTHETIC_ONLY_PREAUTHORIZATION")
        print("Real numeric V1 latent arrays: FORBIDDEN")
        print("Real biological score rows: FORBIDDEN")
        print("TARGET outcomes/clinical: FORBIDDEN")
        print("Model fit/refit: FORBIDDEN")

        if result_dir.exists() and any(result_dir.iterdir()):
            raise RuntimeError(
                "Scientific sensitivity results exist before source authorization."
            )
        if result_lock_dir.exists() and any(result_lock_dir.iterdir()):
            raise RuntimeError(
                "06e5 result-lock directory exists before source authorization."
            )

        if preflight_path.exists():
            existing = load_json(preflight_path)
            if (
                existing.get("status") != PREAUTH_STATUS
                or existing.get("runner_sha256") != runner_sha
                or existing.get("contract_sha256") != contract_sha
            ):
                raise RuntimeError(
                    "Existing 06e4 preflight does not match this runner/contract."
                )
            print("Existing matching synthetic preflight: PASS")
            print("No real scientific data opened.")
            print(
                "Next: run the separate 06e4a source authorizer. "
                "Do not rerun/edit this runner."
            )
            return

        run_synthetic_selftest(runner_sha, contract_sha, preflight_path)
        return

    # ------------------------------------------------------------------
    # AUTHORIZED REAL READ-ONLY SENSITIVITY MODE.
    # ------------------------------------------------------------------
    auth = load_json(auth_path)
    if auth.get("status") != AUTH_STATUS:
        raise RuntimeError("06e4a source-authorization status mismatch.")
    if str(auth.get("runner_source", {}).get("sha256", "")).lower() != runner_sha:
        raise RuntimeError("06e4a does not authorize this exact runner SHA.")
    if str(auth.get("06e3_contract", {}).get("sha256", "")).lower() != contract_sha:
        raise RuntimeError("06e4a does not bind the current 06e3 contract SHA.")

    require_file(preflight_path)
    preflight = load_json(preflight_path)
    if preflight.get("status") != PREAUTH_STATUS:
        raise RuntimeError("06e4 synthetic preflight status mismatch.")
    if preflight.get("runner_sha256") != runner_sha:
        raise RuntimeError("06e4 preflight runner SHA mismatch.")
    if preflight.get("contract_sha256") != contract_sha:
        raise RuntimeError("06e4 preflight contract SHA mismatch.")
    if str(
        auth.get("synthetic_preflight", {}).get("sha256", "")
    ).lower() != sha256_file(preflight_path):
        raise RuntimeError("06e4a preflight hash binding mismatch.")

    if result_dir.exists() and any(result_dir.iterdir()):
        raise RuntimeError(
            "06e4 scientific result directory already exists and is non-empty; "
            "never rerun/overwrite frozen sensitivity results."
        )
    if result_lock_dir.exists() and any(result_lock_dir.iterdir()):
        raise RuntimeError(
            "06e5 result-lock directory already exists and is non-empty."
        )

    print("Execution mode: AUTHORIZED_READONLY_SENSITIVITY")
    print("Primary seed remains: 1103")
    print("Reproducibility-only seeds: [2207, 3301]")
    print("New p/q values: FORBIDDEN")
    print("Rotation-invariant program-to-subspace biology statistic: FORBIDDEN")
    print("Model fit/refit: NO")
    print("TARGET outcomes/clinical: FORBIDDEN")

    # ------------------------------------------------------------------
    # Frozen real file identities.
    # ------------------------------------------------------------------
    full_data_dir = (
        paper5 / "results" / "target_representation" / "06c_full_data_v1"
    )
    case_order_path = full_data_dir / "target_v1_full_data_case_order.csv"
    rep_paths = {
        seed: full_data_dir / f"target_v1_patient_shared_seed{seed}.npy"
        for seed in EXPECTED["all_seeds"]
    }

    biology_dir = (
        paper5 / "results" / "target_representation"
        / "06e1_biological_alignment_v3"
    )
    scores_path = biology_dir / "v1_frozen_program_scores_78.csv"
    primary_corr_path = biology_dir / "v1_program_coordinate_correlations.csv"
    primary_align_path = biology_dir / "v1_program_alignment_summary.csv"
    primary_summary_path = biology_dir / "v1_biological_alignment_summary.json"

    identity_checks = {
        case_order_path: EXPECTED["case_order_sha256"],
        scores_path: EXPECTED["program_scores_sha256"],
        primary_corr_path: EXPECTED["primary_correlations_sha256"],
        primary_align_path: EXPECTED["primary_alignment_summary_sha256"],
        primary_summary_path: EXPECTED["primary_biology_summary_sha256"],
        **{
            rep_paths[seed]: EXPECTED["representation_sha256"][seed]
            for seed in EXPECTED["all_seeds"]
        },
    }
    for path, expected_sha in identity_checks.items():
        require_file(path)
        actual = sha256_file(path)
        if actual != expected_sha:
            raise RuntimeError(
                f"Frozen scientific input SHA mismatch: {path.name}: "
                f"{actual} != {expected_sha}"
            )

    # ------------------------------------------------------------------
    # Load frozen case order + score table and exact three latent arrays.
    # ------------------------------------------------------------------
    case_order = pd.read_csv(case_order_path)
    if list(case_order.columns) != ["case_index", "case_id"]:
        raise RuntimeError("Frozen case-order schema mismatch.")
    if case_order.shape != (78, 2):
        raise RuntimeError("Frozen case-order shape mismatch.")
    if not np.array_equal(
        pd.to_numeric(case_order["case_index"], errors="raise").to_numpy(int),
        np.arange(78, dtype=int),
    ):
        raise RuntimeError("Frozen case_index is not exactly 0..77.")

    scores = pd.read_csv(scores_path)
    expected_score_cols = [
        "case_id",
        "M34__strict_signed_mean_z",
        "M40__strict_signed_mean_z",
        "M11__strict_signed_mean_z",
        "M24__strict_signed_mean_z",
    ]
    if list(scores.columns) != expected_score_cols:
        raise RuntimeError(
            f"Frozen program-score schema mismatch: {list(scores.columns)}"
        )
    if scores.shape != (78, 5):
        raise RuntimeError("Frozen program-score shape mismatch.")
    if scores["case_id"].astype(str).tolist() != (
        case_order["case_id"].astype(str).tolist()
    ):
        raise RuntimeError("Frozen score/case order mismatch.")

    latents: dict[int, np.ndarray] = {}
    centered: dict[int, np.ndarray] = {}
    for seed in EXPECTED["all_seeds"]:
        z = np.load(rep_paths[seed], allow_pickle=False)
        z = np.asarray(z, dtype=np.float64)
        if z.shape != (78, 5) or not np.isfinite(z).all():
            raise RuntimeError(f"Seed {seed}: invalid frozen latent array.")
        latents[seed] = z
        centered[seed] = center_latent(z)
        rank, _, _ = svd_rank_and_basis(centered[seed])
        if rank != 5:
            raise RuntimeError(f"Seed {seed}: centered latent rank={rank}, expected 5.")

    # ------------------------------------------------------------------
    # Replay primary seed-1103 biology exactly before any new sensitivity.
    # ------------------------------------------------------------------
    primary_frozen = pd.read_csv(primary_corr_path)
    required_primary_cols = {
        "program_id",
        "coordinate_id",
        "spearman_rho",
        "absolute_spearman_rho",
        "is_best_absolute_coordinate",
    }
    if not required_primary_cols.issubset(primary_frozen.columns):
        raise RuntimeError("Frozen primary correlation table schema mismatch.")

    raw_corr_parts: list[pd.DataFrame] = []
    raw_summary_parts: list[pd.DataFrame] = []

    corr1103, summary1103 = raw_program_correlations(
        scores, latents[1103], 1103
    )
    raw_corr_parts.append(corr1103)
    raw_summary_parts.append(summary1103)

    frozen_map = {
        (str(r["program_id"]), str(r["coordinate_id"])):
            float(r["spearman_rho"])
        for r in primary_frozen.to_dict(orient="records")
    }
    if len(frozen_map) != 20:
        raise RuntimeError("Frozen primary correlation map must contain 20 cells.")

    for row in corr1103.to_dict(orient="records"):
        key = (row["program_id"], row["coordinate_id"])
        if key not in frozen_map:
            raise RuntimeError(f"Primary replay cell missing from frozen table: {key}")
        if not np.isclose(
            float(row["spearman_rho"]),
            frozen_map[key],
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(f"Primary seed-1103 replay mismatch: {key}")

    primary_align = pd.read_csv(primary_align_path)
    required_align_cols = {
        "program_id",
        "observed_max_abs_spearman",
        "best_coordinate_id",
        "best_signed_spearman",
    }
    if not required_align_cols.issubset(primary_align.columns):
        raise RuntimeError("Frozen primary alignment-summary schema mismatch.")

    for row in summary1103.to_dict(orient="records"):
        frozen = primary_align.loc[
            primary_align["program_id"].astype(str).eq(row["program_id"])
        ]
        if frozen.shape[0] != 1:
            raise RuntimeError(
                f"Primary alignment row count mismatch for {row['program_id']}."
            )
        fr = frozen.iloc[0]
        for key in ["observed_max_abs_spearman", "best_signed_spearman"]:
            if not np.isclose(
                float(row[key]),
                float(fr[key]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"Primary seed-1103 summary replay mismatch: "
                    f"{row['program_id']} {key}"
                )
        if str(row["best_coordinate_id"]) != str(fr["best_coordinate_id"]):
            raise RuntimeError(
                f"Primary seed-1103 best-coordinate replay mismatch: "
                f"{row['program_id']}"
            )

    print("Primary seed-1103 4x5 biology replay: PASS [20/20]")

    # ------------------------------------------------------------------
    # Raw-coordinate biology sensitivity for 2207/3301.
    # ------------------------------------------------------------------
    for seed in EXPECTED["reproducibility_seeds"]:
        corr, summ = raw_program_correlations(scores, latents[seed], seed)
        raw_corr_parts.append(corr)
        raw_summary_parts.append(summ)

    raw_corr = pd.concat(raw_corr_parts, ignore_index=True)
    raw_summary = pd.concat(raw_summary_parts, ignore_index=True)
    if raw_corr.shape[0] != 60 or raw_summary.shape[0] != 12:
        raise RuntimeError("Unexpected raw program x seed output geometry.")

    # ------------------------------------------------------------------
    # Pairwise subspace / Procrustes / Hungarian diagnostics.
    # ------------------------------------------------------------------
    seed_pairs = [(1103, 2207), (1103, 3301), (2207, 3301)]
    subspace_rows: list[dict[str, Any]] = []
    canonical_rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    aligned_to_1103: dict[int, np.ndarray] = {}

    for ref_seed, moving_seed in seed_pairs:
        reference = centered[ref_seed]
        moving = centered[moving_seed]

        canonical, angles = canonical_diagnostics(moving, reference)
        aligned, rotation, residual = procrustes_align(moving, reference)
        moving_idx, ref_idx, matched_signed = hungarian_match(
            aligned, reference
        )

        subspace_rows.append(
            {
                "reference_seed": ref_seed,
                "moving_seed": moving_seed,
                "mean_canonical_correlation": float(np.mean(canonical)),
                "median_canonical_correlation": float(np.median(canonical)),
                "minimum_canonical_correlation": float(np.min(canonical)),
                "maximum_principal_angle_degrees": float(np.max(angles)),
                "normalized_procrustes_residual": residual,
                "mean_abs_hungarian_matched_correlation":
                    float(np.mean(np.abs(matched_signed))),
                "minimum_abs_hungarian_matched_correlation":
                    float(np.min(np.abs(matched_signed))),
                "analysis_role":
                    "POST_RESULT_REPRODUCIBILITY_SUBSPACE_DIAGNOSTIC",
            }
        )

        for k in range(5):
            canonical_rows.append(
                {
                    "reference_seed": ref_seed,
                    "moving_seed": moving_seed,
                    "canonical_index": k + 1,
                    "canonical_correlation": float(canonical[k]),
                    "principal_angle_degrees": float(angles[k]),
                }
            )

        for order_idx in range(5):
            signed = float(matched_signed[order_idx])
            matched_rows.append(
                {
                    "reference_seed": ref_seed,
                    "moving_seed": moving_seed,
                    "reference_coordinate_id":
                        f"V1_SHARED_{int(ref_idx[order_idx])+1:02d}",
                    "aligned_moving_coordinate_id":
                        f"ALIGNED_MOVING_{int(moving_idx[order_idx])+1:02d}",
                    "matched_signed_pearson": signed,
                    "matched_absolute_pearson": abs(signed),
                }
            )

        if ref_seed == 1103:
            aligned_to_1103[moving_seed] = aligned

    subspace = pd.DataFrame(subspace_rows)
    canonical_table = pd.DataFrame(canonical_rows)
    matched_table = pd.DataFrame(matched_rows)

    if subspace.shape[0] != 3:
        raise RuntimeError("Expected exactly three pairwise subspace rows.")
    if canonical_table.shape[0] != 15:
        raise RuntimeError("Expected exactly 15 canonical-correlation rows.")
    if matched_table.shape[0] != 15:
        raise RuntimeError("Expected exactly 15 Hungarian-match rows.")

    five_a_compatibility = float(
        np.median(subspace["mean_canonical_correlation"].to_numpy(float))
    )

    # ------------------------------------------------------------------
    # Exploratory Procrustes-oriented biology sensitivity to seed 1103.
    # ------------------------------------------------------------------
    aligned_corr_parts: list[pd.DataFrame] = []
    aligned_summary_parts: list[pd.DataFrame] = []
    for seed in EXPECTED["reproducibility_seeds"]:
        if seed not in aligned_to_1103:
            raise RuntimeError(f"Missing seed {seed} alignment to 1103.")
        corr, summ = aligned_program_correlations(
            scores, aligned_to_1103[seed], seed
        )
        aligned_corr_parts.append(corr)
        aligned_summary_parts.append(summ)

    aligned_corr = pd.concat(aligned_corr_parts, ignore_index=True)
    aligned_summary = pd.concat(aligned_summary_parts, ignore_index=True)
    if aligned_corr.shape[0] != 40 or aligned_summary.shape[0] != 8:
        raise RuntimeError("Unexpected aligned program-sensitivity geometry.")

    # ------------------------------------------------------------------
    # Across-seed raw max-|rho| descriptive summaries, all 4 programs.
    # ------------------------------------------------------------------
    across_rows: list[dict[str, Any]] = []
    for program in EXPECTED["programs"]:
        part = raw_summary.loc[
            raw_summary["program_id"].eq(program)
        ].sort_values("seed")
        if part.shape[0] != 3:
            raise RuntimeError(f"{program}: expected 3 raw seed summaries.")
        values = part["observed_max_abs_spearman"].to_numpy(float)
        across_rows.append(
            {
                "program_id": program,
                "primary_seed_1103_max_abs_spearman":
                    float(
                        part.loc[
                            part["seed"].eq(1103),
                            "observed_max_abs_spearman",
                        ].iloc[0]
                    ),
                "seed2207_max_abs_spearman":
                    float(
                        part.loc[
                            part["seed"].eq(2207),
                            "observed_max_abs_spearman",
                        ].iloc[0]
                    ),
                "seed3301_max_abs_spearman":
                    float(
                        part.loc[
                            part["seed"].eq(3301),
                            "observed_max_abs_spearman",
                        ].iloc[0]
                    ),
                "min_raw_max_abs_spearman": float(np.min(values)),
                "median_raw_max_abs_spearman": float(np.median(values)),
                "max_raw_max_abs_spearman": float(np.max(values)),
                "range_raw_max_abs_spearman":
                    float(np.max(values) - np.min(values)),
                "analysis_role":
                    "POST_RESULT_REPRODUCIBILITY_SENSITIVITY_NO_THRESHOLD",
            }
        )
    across = pd.DataFrame(across_rows)

    # ------------------------------------------------------------------
    # Write new sensitivity results only after all checks pass.
    # ------------------------------------------------------------------
    result_dir.mkdir(parents=True, exist_ok=False)
    result_lock_dir.mkdir(parents=True, exist_ok=False)

    files_to_write = {
        "v1_program_coordinate_correlations_by_seed.csv": raw_corr,
        "v1_program_alignment_summary_by_seed.csv": raw_summary,
        "v1_program_raw_seed_sensitivity_summary.csv": across,
        "v1_pairwise_seed_subspace_summary.csv": subspace,
        "v1_pairwise_seed_canonical_correlations.csv": canonical_table,
        "v1_pairwise_seed_hungarian_matches.csv": matched_table,
        "v1_program_procrustes_aligned_correlations.csv": aligned_corr,
        "v1_program_procrustes_aligned_summary.csv": aligned_summary,
    }

    for filename, frame in files_to_write.items():
        frame.to_csv(result_dir / filename, index=False)

    summary = {
        "status": "06E4_V1_SEED_SUBSPACE_BIOLOGY_SENSITIVITY_COMPLETE",
        "script_version": SCRIPT_VERSION,
        "runner_sha256": runner_sha,
        "created_at_utc": utc_now_iso(),
        "analysis_role":
            "POST_RESULT_SENSITIVITY_USING_PROSPECTIVELY_FROZEN_REPRODUCIBILITY_SEEDS",
        "primary_seed": 1103,
        "reproducibility_only_seeds": [2207, 3301],
        "programs": EXPECTED["programs"],
        "n_cases": 78,
        "shared_dim": 5,
        "primary_seed_replay": "PASS_20_OF_20",
        "05a_compatible_seed_aligned_subspace_similarity":
            five_a_compatibility,
        "pairwise_subspace_summary":
            subspace.to_dict(orient="records"),
        "program_raw_seed_sensitivity":
            across.to_dict(orient="records"),
        "firewalls": {
            "primary_result_changed": False,
            "seed_selected_by_biology": False,
            "seed_ensemble_or_averaging": False,
            "new_seed_specific_p_or_q": False,
            "matched_random_panels_recomputed": False,
            "rotation_invariant_program_to_subspace_statistic_computed": False,
            "SABLE_q10_or_stability_gate_applied_to_V1": False,
            "target_outcomes_or_clinical_read": False,
            "model_fit_or_refit": False,
            "technical_UNKNOWN_reinterpreted_as_clean": False,
        },
    }
    summary_path = result_dir / "v1_seed_subspace_biology_sensitivity_summary.json"
    write_json(summary_path, summary)

    generated = [result_dir / name for name in files_to_write] + [summary_path]
    result_lock = {
        "status": RESULT_STATUS,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "runner_source": {
            "path":
                "scripts/06e4_run_v1_seed_subspace_biology_sensitivity_v1.py",
            "sha256": runner_sha,
        },
        "06e3_contract": {
            "path":
                "method_contract/06e3/"
                "v1_seed_subspace_biology_sensitivity_contract.json",
            "sha256": contract_sha,
        },
        "06e4a_authorization": {
            "path":
                "method_contract/06e4a/"
                "v1_seed_subspace_sensitivity_runner_source_authorization_lock.json",
            "sha256": sha256_file(auth_path),
        },
        "result_files": {
            p.name: {
                "path":
                    f"results/target_representation/"
                    f"06e4_seed_subspace_sensitivity_v1/{p.name}",
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
            }
            for p in generated
        },
        "rewrite_policy": (
            "Frozen post-result sensitivity. Never overwrite/delete/rerun in "
            "place. Any scientifically necessary recovery requires a new "
            "versioned stage preserving this lock and all current results."
        ),
    }
    result_lock_path = (
        result_lock_dir
        / "v1_seed_subspace_biology_sensitivity_results_lock.json"
    )
    write_json(result_lock_path, result_lock)

    print("-" * 108)
    print("Pairwise frozen-seed subspace summary")
    print("-" * 108)
    print(
        subspace[
            [
                "reference_seed",
                "moving_seed",
                "mean_canonical_correlation",
                "minimum_canonical_correlation",
                "maximum_principal_angle_degrees",
                "normalized_procrustes_residual",
                "mean_abs_hungarian_matched_correlation",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    print("-" * 108)
    print("Raw-coordinate program max-|Spearman| across frozen seeds")
    print("-" * 108)
    print(
        raw_summary[
            [
                "program_id",
                "seed",
                "observed_max_abs_spearman",
                "best_coordinate_id",
                "best_signed_spearman",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    print("-" * 108)
    print("Exploratory Procrustes-oriented program max-|Spearman|")
    print("-" * 108)
    print(
        aligned_summary[
            [
                "program_id",
                "seed",
                "aligned_max_abs_spearman",
                "best_aligned_coordinate_id",
                "best_aligned_signed_spearman",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    print("Primary biological result remains seed 1103: YES")
    print("New seed-specific p/q values computed: NO")
    print("Rotation-invariant program-to-subspace biology statistic computed: NO")
    print("TARGET outcomes/clinical read: NO")
    print("Model fit/refit: NO")
    print("=" * 108)
    print("Paper 5 06e4 V1 seed/subspace biological sensitivity: COMPLETE")
    print("=" * 108)
    print(
        "Summary: results/target_representation/"
        "06e4_seed_subspace_sensitivity_v1/"
        "v1_seed_subspace_biology_sensitivity_summary.json"
    )
    print(
        "Results lock: method_contract/06e5/"
        "v1_seed_subspace_biology_sensitivity_results_lock.json"
    )
    print("=" * 108)


if __name__ == "__main__":
    main()
