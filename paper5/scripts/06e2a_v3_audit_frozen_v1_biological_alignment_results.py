#!/usr/bin/env python3
"""
Paper 5 - 06e2a_v3 read-only audit of frozen V1 biological-alignment results.

This stage verifies the already-frozen 06e1 v3 biological result without
re-running biology and without modifying any 06e1/06e2 artifact.

It checks:
- chronology: source authorization predates scientific result creation;
- exact runner/source and authorization identities;
- exact six frozen result-file hashes;
- 4 x 5 coordinate-correlation geometry and best-coordinate consistency;
- 4 x 1,000 matched-random draw geometry;
- exact +1 empirical probabilities and BH q-values;
- 287,000 matched-panel membership rows, no within-panel duplicates,
  no primary-anchor genes in the random candidate panels, and fallback replay;
- 78 frozen program-score rows;
- exact replay of all 20 Spearman correlations from the frozen program scores
  and the already-frozen primary V1 representation;
- scientific firewalls (no outcomes, no refit, no SABLE retention transfer,
  technical UNKNOWN not reinterpreted as clean).

It does NOT read Paper-4 expression/weights or TARGET outcomes and does NOT fit
or refit any model.

No command-line arguments.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "06e2a-v3-readonly-audit-v1-no-cli"

EXPECTED = {
    "runner_sha256":
        "77fbe2d681ddaa906db71c93a654c5f6ddac1fed10c5ad96fcd211db48f78d26",
    "runner_script_version":
        "06e1-run-v1-biological-alignment-random-controls-v3-"
        "exact-risk-oriented-weight-schema-provenance-text-fix-two-stage-source-authorized-no-cli",
    "authorization_status":
        "06E1A_V3_TARGET_V1_BIOLOGY_RUNNER_SOURCE_AUTHORIZED",
    "authorization_lock_sha256":
        "0c84dcac02be3299e7f324332419e5e8076763ca7819a6497040e5397e9c4549",
    "summary_status":
        "06E1_V1_BIOLOGICAL_ALIGNMENT_RANDOM_CONTROLS_COMPLETE",
    "result_lock_status":
        "06E2_V3_V1_BIOLOGICAL_ALIGNMENT_RANDOM_CONTROLS_FROZEN",
    "model_id": "V1_SHARED_PRIVATE_AE",
    "profile_id": "ARCH_COMPACT__REG_WEAK",
    "primary_seed": 1103,
    "n_cases": 78,
    "shared_dim": 5,
    "programs": ["M34", "M40", "M11", "M24"],
    "strict_gene_counts": {
        "M34": 162,
        "M40": 111,
        "M11": 7,
        "M24": 7,
    },
    "technical_status":
        "UNKNOWN_NO_STRICT_USABLE_TECHNICAL_COVARIATES",
    "representation_sha256":
        "c05bf08017ad8544e1e4ce0c8f58f086060dd7a6a0a68ed7f8d5a6723768a95f",
    "random_panels_per_program": 1000,
    "expected_membership_rows": 287000,
    "result_files": {
        "v1_program_coordinate_correlations.csv":
            "58cadf6ce02603d8d9a05cf06068c9f23e43a0a2dafe8835729407374d89c7f8",
        "v1_program_alignment_summary.csv":
            "a35e24eda531f40d1685f24feea6ae0eecd60f6bc9978feb1c73f5ca0a12b30f",
        "v1_matched_random_control_draws.csv":
            "382f1d66ab36a478e791b43d2e1c29943748e8497423589d3a526c9316cc6887",
        "v1_matched_random_panel_memberships.csv.gz":
            "61a900a14d975c42bb4c9d43bba3e4b8fa98c56f22663f007dceba904239efac",
        "v1_frozen_program_scores_78.csv":
            "7129ad9719d0f01ed12b3cf854c1a4dff2e7295094986e09f08c7344134b00dc",
        "v1_biological_alignment_summary.json":
            "e4e68f5f21b2a54db4d0524fdaba265c42972c0fb6654036d9116f5ed3ca8370",
    },
    "program_results": {
        "M34": {
            "n_strict_genes": 162,
            "observed_max_abs_spearman": 0.5106665486412322,
            "best_coordinate_id": "V1_SHARED_04",
            "best_signed_spearman": -0.5106665486412322,
            "null_median": 0.3289621770634429,
            "null_q05": 0.26090238875049,
            "null_q95": 0.4065238558909444,
            "null_exceedances": 0,
            "empirical_p_plus1": 0.000999000999000999,
            "bh_q_4": 0.003996003996003996,
            "support": True,
        },
        "M40": {
            "n_strict_genes": 111,
            "observed_max_abs_spearman": 0.3946053946053946,
            "best_coordinate_id": "V1_SHARED_01",
            "best_signed_spearman": 0.3946053946053946,
            "null_median": 0.3118274130932358,
            "null_q05": 0.21963479558416263,
            "null_q95": 0.41190834481973715,
            "null_exceedances": 85,
            "empirical_p_plus1": 0.08591408591408592,
            "bh_q_4": 0.17182817182817184,
            "support": False,
        },
        "M11": {
            "n_strict_genes": 7,
            "observed_max_abs_spearman": 0.1509756066718092,
            "best_coordinate_id": "V1_SHARED_04",
            "best_signed_spearman": -0.1509756066718092,
            "null_median": 0.2870800085989959,
            "null_q05": 0.15067970004678866,
            "null_q95": 0.4553193641801236,
            "null_exceedances": 948,
            "empirical_p_plus1": 0.948051948051948,
            "bh_q_4": 0.948051948051948,
            "support": False,
        },
        "M24": {
            "n_strict_genes": 7,
            "observed_max_abs_spearman": 0.3293036077846204,
            "best_coordinate_id": "V1_SHARED_05",
            "best_signed_spearman": -0.3293036077846204,
            "null_median": 0.28113658493405325,
            "null_q05": 0.1596643862466647,
            "null_q95": 0.4482239279707633,
            "null_exceedances": 302,
            "empirical_p_plus1": 0.3026973026973027,
            "bh_q_4": 0.40359640359640364,
            "support": False,
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
        raise FileNotFoundError(f"Required file missing: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def close(a: float, b: float, tol: float = 1.0e-12) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=0.0, atol=tol))


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
        raise RuntimeError("Spearman audit vectors have incompatible shapes.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("Spearman audit vector contains non-finite values.")
    rx = average_ranks(x)
    ry = average_ranks(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    if not np.isfinite(rho):
        raise RuntimeError("Spearman audit correlation is non-finite.")
    return rho


def bh(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    m = len(ranked)
    q_ranked = np.empty(m, dtype=np.float64)
    running = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        running = min(running, ranked[i] * m / rank)
        q_ranked[i] = min(1.0, running)
    q = np.empty(m, dtype=np.float64)
    q[order] = q_ranked
    return [float(v) for v in q]


def main() -> None:
    print("=" * 108)
    print("Paper 5 - 06e2a_v3 read-only audit of frozen V1 biological-alignment results")
    print("=" * 108)
    print("Script version:", SCRIPT_VERSION)
    print("Paper-4 biological input rows read in audit: NO")
    print("TARGET expression values read in audit: NO")
    print("TARGET clinical/outcomes read in audit: NO")
    print("Model fit/refit in audit: NO")
    print("Frozen scientific outputs rewritten in audit: NO")

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    paper5 = project_root / "paper5"

    runner_path = (
        paper5 / "scripts"
        / "06e1_run_v1_biological_alignment_random_controls_v3.py"
    )
    auth_path = (
        paper5 / "method_contract" / "06e1a_v3"
        / "target_v1_biology_runner_v3_source_authorization_lock.json"
    )
    result_dir = (
        paper5 / "results" / "target_representation"
        / "06e1_biological_alignment_v3"
    )
    result_lock_path = (
        paper5 / "method_contract" / "06e2_v3"
        / "v1_biological_alignment_random_controls_results_lock_v3.json"
    )
    results_06c3_lock_path = (
        paper5 / "method_contract" / "06c3"
        / "target_v1_full_data_refit_results_lock.json"
    )
    v1_rep_path = (
        paper5 / "results" / "target_representation"
        / "06c_full_data_v1"
        / "target_v1_patient_shared_seed1103.npy"
    )
    case_order_path = (
        paper5 / "results" / "target_representation"
        / "06c_full_data_v1"
        / "target_v1_full_data_case_order.csv"
    )

    for path in [
        runner_path,
        auth_path,
        result_lock_path,
        results_06c3_lock_path,
        v1_rep_path,
        case_order_path,
    ]:
        require_file(path)

    if sha256_file(runner_path) != EXPECTED["runner_sha256"]:
        raise RuntimeError("Frozen v3 runner SHA mismatch.")
    if sha256_file(auth_path) != EXPECTED["authorization_lock_sha256"]:
        raise RuntimeError("Frozen v3 authorization-lock SHA mismatch.")

    auth = load_json(auth_path)
    if auth.get("status") != EXPECTED["authorization_status"]:
        raise RuntimeError("Authorization status mismatch.")
    if str(auth.get("runner_source", {}).get("sha256", "")).lower() != (
        EXPECTED["runner_sha256"]
    ):
        raise RuntimeError("Authorization does not bind the exact v3 runner.")

    result_lock = load_json(result_lock_path)
    if result_lock.get("status") != EXPECTED["result_lock_status"]:
        raise RuntimeError("06e2_v3 result-lock status mismatch.")
    if str(result_lock.get("runner_source", {}).get("sha256", "")).lower() != (
        EXPECTED["runner_sha256"]
    ):
        raise RuntimeError("06e2_v3 result lock binds the wrong runner.")
    if result_lock.get("selected_model") != EXPECTED["model_id"]:
        raise RuntimeError("Result-lock selected model mismatch.")
    if int(result_lock.get("primary_seed")) != EXPECTED["primary_seed"]:
        raise RuntimeError("Result-lock primary seed mismatch.")
    if result_lock.get("programs") != EXPECTED["programs"]:
        raise RuntimeError("Result-lock program order/set mismatch.")
    if int(result_lock.get("random_panels_per_program")) != (
        EXPECTED["random_panels_per_program"]
    ):
        raise RuntimeError("Result-lock random-panel count mismatch.")

    # Authorization must strictly precede result creation.
    auth_time = datetime.fromisoformat(auth["created_at_utc"])
    result_time = datetime.fromisoformat(result_lock["created_at_utc"])
    if not auth_time < result_time:
        raise RuntimeError("Authorization does not predate scientific result creation.")

    # Exact expected file set and hashes.
    actual_files = sorted(
        p.name for p in result_dir.iterdir() if p.is_file()
    )
    if actual_files != sorted(EXPECTED["result_files"]):
        raise RuntimeError(
            f"Frozen v3 result file set mismatch. Actual={actual_files}"
        )

    locked_files = result_lock.get("result_files", {})
    if set(locked_files) != set(EXPECTED["result_files"]):
        raise RuntimeError("Result lock contains an unexpected result-file set.")

    for filename, expected_sha in EXPECTED["result_files"].items():
        path = result_dir / filename
        require_file(path)
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Frozen result SHA mismatch for {filename}: "
                f"{actual_sha} != {expected_sha}"
            )
        if str(locked_files[filename].get("sha256", "")).lower() != expected_sha:
            raise RuntimeError(
                f"Result lock SHA mismatch for {filename}."
            )
        if int(locked_files[filename].get("size_bytes")) != path.stat().st_size:
            raise RuntimeError(
                f"Result lock size mismatch for {filename}."
            )

    summary = load_json(result_dir / "v1_biological_alignment_summary.json")
    if summary.get("status") != EXPECTED["summary_status"]:
        raise RuntimeError("Frozen scientific summary status mismatch.")
    if summary.get("script_version") != EXPECTED["runner_script_version"]:
        raise RuntimeError("Frozen scientific summary SCRIPT_VERSION mismatch.")
    if str(summary.get("runner_sha256", "")).lower() != EXPECTED["runner_sha256"]:
        raise RuntimeError("Frozen scientific summary runner SHA mismatch.")
    if summary.get("model_id") != EXPECTED["model_id"]:
        raise RuntimeError("Scientific summary model mismatch.")
    if summary.get("profile_id") != EXPECTED["profile_id"]:
        raise RuntimeError("Scientific summary profile mismatch.")
    if int(summary.get("primary_seed")) != EXPECTED["primary_seed"]:
        raise RuntimeError("Scientific summary primary seed mismatch.")
    if int(summary.get("n_cases")) != EXPECTED["n_cases"]:
        raise RuntimeError("Scientific summary n_cases mismatch.")
    if int(summary.get("shared_dim")) != EXPECTED["shared_dim"]:
        raise RuntimeError("Scientific summary shared_dim mismatch.")
    if summary.get("programs") != EXPECTED["programs"]:
        raise RuntimeError("Scientific summary program order/set mismatch.")
    if summary.get("technical_status") != EXPECTED["technical_status"]:
        raise RuntimeError("Scientific summary technical status mismatch.")
    if summary.get("coordinate_stability_qualification") != (
        "NOT_ESTABLISHED_FOR_V1"
    ):
        raise RuntimeError("V1 stability qualification changed.")

    if str(
        summary.get("authorization", {}).get("authorization_lock_sha256", "")
    ).lower() != EXPECTED["authorization_lock_sha256"]:
        raise RuntimeError("Summary authorization-lock SHA mismatch.")

    # Firewalls must remain exactly conservative.
    firewalls = summary.get("firewalls", {})
    for key in [
        "target_clinical_or_outcomes_read",
        "model_fit_or_refit",
        "representation_selection_changed",
        "primary_seed_changed",
        "SABLE_q10_or_stability_retention_applied_to_V1",
        "V1_coordinate_called_retained_or_stable",
        "technical_UNKNOWN_reinterpreted_as_clean",
    ]:
        if firewalls.get(key) is not False:
            raise RuntimeError(f"Scientific firewall failed: {key}")

    # Load frozen result tables only.
    coord = pd.read_csv(result_dir / "v1_program_coordinate_correlations.csv")
    program_summary = pd.read_csv(result_dir / "v1_program_alignment_summary.csv")
    draws = pd.read_csv(result_dir / "v1_matched_random_control_draws.csv")
    memberships = pd.read_csv(
        result_dir / "v1_matched_random_panel_memberships.csv.gz",
        compression="gzip",
        low_memory=False,
    )
    scores = pd.read_csv(result_dir / "v1_frozen_program_scores_78.csv")

    # 4 x 5 coordinate geometry.
    if coord.shape[0] != 20:
        raise RuntimeError(f"Coordinate table must have 20 rows, got {coord.shape[0]}.")
    expected_coords = {f"V1_SHARED_{i:02d}" for i in range(1, 6)}
    for program in EXPECTED["programs"]:
        part = coord.loc[coord["program_id"].eq(program)].copy()
        if part.shape[0] != 5:
            raise RuntimeError(f"{program}: expected 5 coordinate rows.")
        if set(part["coordinate_id"]) != expected_coords:
            raise RuntimeError(f"{program}: coordinate id set mismatch.")
        if int(part["is_best_absolute_coordinate"].astype(bool).sum()) != 1:
            raise RuntimeError(f"{program}: expected exactly one best coordinate.")
        if not np.allclose(
            pd.to_numeric(part["absolute_spearman_rho"]).to_numpy(float),
            np.abs(pd.to_numeric(part["spearman_rho"]).to_numpy(float)),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(f"{program}: abs(rho) consistency failed.")

    # Program-summary and exact console/JSON values.
    if program_summary.shape[0] != 4:
        raise RuntimeError("Program summary must contain four rows.")
    if set(program_summary["program_id"]) != set(EXPECTED["programs"]):
        raise RuntimeError("Program summary id set mismatch.")

    json_rows = {
        row["program_id"]: row
        for row in summary.get("program_results", [])
    }
    if set(json_rows) != set(EXPECTED["programs"]):
        raise RuntimeError("Summary JSON program-result set mismatch.")

    for program in EXPECTED["programs"]:
        expected = EXPECTED["program_results"][program]
        csv_row = program_summary.loc[
            program_summary["program_id"].eq(program)
        ].iloc[0]
        json_row = json_rows[program]

        for key in [
            "n_strict_genes",
            "best_coordinate_id",
        ]:
            if str(csv_row[key]) != str(expected[key]):
                raise RuntimeError(f"{program}: CSV {key} mismatch.")
            if str(json_row[key]) != str(expected[key]):
                raise RuntimeError(f"{program}: JSON {key} mismatch.")

        for key in [
            "observed_max_abs_spearman",
            "best_signed_spearman",
            "null_median",
            "null_q05",
            "null_q95",
            "empirical_p_plus1",
            "bh_q_4",
        ]:
            if not close(float(csv_row[key]), float(expected[key])):
                raise RuntimeError(f"{program}: CSV {key} mismatch.")
            if not close(float(json_row[key]), float(expected[key])):
                raise RuntimeError(f"{program}: JSON {key} mismatch.")

        for key in ["null_exceedances"]:
            if int(csv_row[key]) != int(expected[key]):
                raise RuntimeError(f"{program}: CSV {key} mismatch.")
            if int(json_row[key]) != int(expected[key]):
                raise RuntimeError(f"{program}: JSON {key} mismatch.")

        if bool(json_row["matched_random_support_q_le_0_05"]) != (
            expected["support"]
        ):
            raise RuntimeError(f"{program}: support classification mismatch.")
        if json_row.get("stable_biological_factor_claim_allowed") is not False:
            raise RuntimeError(
                f"{program}: stable biological factor claim unexpectedly allowed."
            )

    # 4 x 1000 random-draw geometry and exact null replay.
    if draws.shape[0] != 4000:
        raise RuntimeError(f"Expected 4000 random-control rows, got {draws.shape[0]}.")
    replay_p = []
    for program in EXPECTED["programs"]:
        part = draws.loc[draws["program_id"].eq(program)].copy()
        if part.shape[0] != 1000:
            raise RuntimeError(f"{program}: expected 1000 random panels.")
        panel_index = pd.to_numeric(part["panel_index"]).astype(int)
        if sorted(panel_index.tolist()) != list(range(1, 1001)):
            raise RuntimeError(f"{program}: panel index schedule mismatch.")
        null = pd.to_numeric(part["null_max_abs_spearman"]).to_numpy(float)
        if not np.isfinite(null).all():
            raise RuntimeError(f"{program}: non-finite null statistic.")
        if np.any((null < 0.0) | (null > 1.0)):
            raise RuntimeError(f"{program}: invalid max-|rho| null statistic.")
        fallback = pd.to_numeric(part["fallback_draws"]).astype(int)
        if int(fallback.sum()) != 0:
            raise RuntimeError(f"{program}: expected zero fallback draws.")

        expected = EXPECTED["program_results"][program]
        observed = expected["observed_max_abs_spearman"]
        exceed = int(np.sum(null >= observed))
        p = (1.0 + exceed) / 1001.0
        replay_p.append(p)

        if exceed != expected["null_exceedances"]:
            raise RuntimeError(f"{program}: null exceedance replay mismatch.")
        if not close(p, expected["empirical_p_plus1"]):
            raise RuntimeError(f"{program}: +1 empirical p replay mismatch.")
        if not close(np.median(null), expected["null_median"]):
            raise RuntimeError(f"{program}: null median replay mismatch.")
        if not close(np.quantile(null, 0.05), expected["null_q05"]):
            raise RuntimeError(f"{program}: null q05 replay mismatch.")
        if not close(np.quantile(null, 0.95), expected["null_q95"]):
            raise RuntimeError(f"{program}: null q95 replay mismatch.")

    replay_q = bh(replay_p)
    for program, q in zip(EXPECTED["programs"], replay_q):
        if not close(q, EXPECTED["program_results"][program]["bh_q_4"]):
            raise RuntimeError(f"{program}: BH q replay mismatch.")

    # Membership geometry: 287 genes per one four-program draw schedule x 1000.
    if memberships.shape[0] != EXPECTED["expected_membership_rows"]:
        raise RuntimeError(
            f"Expected {EXPECTED['expected_membership_rows']} membership rows, "
            f"got {memberships.shape[0]}."
        )
    anchor_union = set(memberships["anchor_gene"].astype(str))
    random_union = set(memberships["random_gene"].astype(str))
    overlap = anchor_union & random_union
    if overlap:
        raise RuntimeError(
            "Matched random panels contain primary anchor genes: "
            + ",".join(sorted(overlap)[:20])
        )

    for program in EXPECTED["programs"]:
        n_genes = EXPECTED["strict_gene_counts"][program]
        part = memberships.loc[memberships["program_id"].eq(program)].copy()
        if part.shape[0] != n_genes * 1000:
            raise RuntimeError(
                f"{program}: membership row count mismatch."
            )
        grouped = part.groupby("panel_index", sort=False)
        sizes = grouped.size()
        if sizes.shape[0] != 1000 or not (sizes == n_genes).all():
            raise RuntimeError(f"{program}: per-panel membership size mismatch.")
        duplicate_counts = grouped["random_gene"].nunique()
        if not (duplicate_counts == n_genes).all():
            raise RuntimeError(
                f"{program}: duplicate random gene within at least one panel."
            )
        positions = grouped["position"].nunique()
        if not (positions == n_genes).all():
            raise RuntimeError(f"{program}: panel position coverage mismatch.")

    fallback_distance = pd.to_numeric(
        memberships["fallback_distance"]
    ).astype(int)
    if int((fallback_distance > 0).sum()) != 0:
        raise RuntimeError("Membership replay expected zero fallback positions.")
    if not (
        pd.to_numeric(memberships["anchor_bin"]).astype(int).to_numpy()
        == pd.to_numeric(memberships["random_bin"]).astype(int).to_numpy()
    ).all():
        raise RuntimeError("Zero-fallback memberships should remain in the same bin.")

    signs = pd.to_numeric(memberships["sign"]).to_numpy(float)
    if not np.isin(signs, [-1.0, 1.0]).all():
        raise RuntimeError("Membership sign vector contains a value other than +/-1.")

    # Frozen 78-case score table.
    expected_score_columns = [
        "case_id",
        "M34__strict_signed_mean_z",
        "M40__strict_signed_mean_z",
        "M11__strict_signed_mean_z",
        "M24__strict_signed_mean_z",
    ]
    if list(scores.columns) != expected_score_columns:
        raise RuntimeError(
            f"Frozen score-table columns changed: {list(scores.columns)}"
        )
    if scores.shape[0] != 78:
        raise RuntimeError("Frozen score table must contain exactly 78 cases.")
    if scores["case_id"].astype(str).duplicated().any():
        raise RuntimeError("Frozen score table contains duplicate case IDs.")
    score_matrix = scores[expected_score_columns[1:]].to_numpy(float)
    if not np.isfinite(score_matrix).all():
        raise RuntimeError("Frozen program-score table contains non-finite values.")
    if not np.allclose(
        np.mean(score_matrix, axis=0),
        np.zeros(4),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("Frozen program scores are not centered at zero.")
    if not np.allclose(
        np.std(score_matrix, axis=0, ddof=1),
        np.ones(4),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("Frozen program scores do not have ddof1 SD=1.")

    # Replay all 20 Spearman correlations from frozen program scores + frozen V1.
    lock06c3 = load_json(results_06c3_lock_path)
    rep_record = lock06c3.get("result_files", {}).get(
        "target_v1_patient_shared_seed1103.npy",
        {},
    )
    if str(rep_record.get("sha256", "")).lower() != (
        EXPECTED["representation_sha256"]
    ):
        raise RuntimeError("06c3 representation record SHA mismatch.")
    if sha256_file(v1_rep_path) != EXPECTED["representation_sha256"]:
        raise RuntimeError("Frozen V1 representation file SHA mismatch.")

    case_record = lock06c3.get("result_files", {}).get(
        "target_v1_full_data_case_order.csv",
        {},
    )
    if str(case_record.get("sha256", "")).lower() != sha256_file(case_order_path):
        raise RuntimeError("Frozen V1 case-order file SHA mismatch.")

    case_order = pd.read_csv(case_order_path)
    if list(case_order.columns) != ["case_id"]:
        raise RuntimeError("Frozen case-order schema mismatch.")
    if case_order.shape[0] != 78:
        raise RuntimeError("Frozen case-order must contain 78 cases.")
    if case_order["case_id"].astype(str).tolist() != (
        scores["case_id"].astype(str).tolist()
    ):
        raise RuntimeError(
            "Frozen program-score case order does not match frozen V1 case order."
        )

    latent = np.load(v1_rep_path, allow_pickle=False)
    latent = np.asarray(latent, dtype=np.float64)
    if latent.shape != (78, 5) or not np.isfinite(latent).all():
        raise RuntimeError("Frozen V1 representation geometry/value audit failed.")

    for program in EXPECTED["programs"]:
        score_col = f"{program}__strict_signed_mean_z"
        score = scores[score_col].to_numpy(float)
        observed_rows = coord.loc[coord["program_id"].eq(program)].copy()
        observed_rows = observed_rows.set_index("coordinate_id")
        replay = {}
        for j in range(5):
            cid = f"V1_SHARED_{j+1:02d}"
            rho = spearman_rho(score, latent[:, j])
            replay[cid] = rho
            locked_rho = float(observed_rows.loc[cid, "spearman_rho"])
            if not close(rho, locked_rho):
                raise RuntimeError(
                    f"{program}/{cid}: frozen Spearman replay mismatch."
                )
        best_cid = max(replay, key=lambda cid: abs(replay[cid]))
        best_abs = abs(replay[best_cid])
        expected = EXPECTED["program_results"][program]
        if best_cid != expected["best_coordinate_id"]:
            raise RuntimeError(f"{program}: best-coordinate replay mismatch.")
        if not close(best_abs, expected["observed_max_abs_spearman"]):
            raise RuntimeError(f"{program}: max-|rho| replay mismatch.")

    # Write only a new audit namespace after all checks pass.
    out_dir = paper5 / "method_contract" / "06e2a_v3"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(
            f"06e2a_v3 audit directory already exists and is non-empty: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    audit = {
        "status": "06E2A_V3_FROZEN_BIOLOGICAL_RESULT_READONLY_AUDIT_PASS",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": EXPECTED["runner_sha256"],
        "authorization_lock_sha256": EXPECTED["authorization_lock_sha256"],
        "result_lock_sha256": sha256_file(result_lock_path),
        "chronology": {
            "authorization_created_at_utc": auth["created_at_utc"],
            "result_lock_created_at_utc": result_lock["created_at_utc"],
            "authorization_precedes_result": True,
        },
        "checks": {
            "exact_result_file_set_and_hashes": "PASS",
            "coordinate_rows_4x5": "PASS",
            "best_coordinate_consistency": "PASS",
            "random_draw_rows_4x1000": "PASS",
            "plus_one_empirical_probability_replay": "PASS",
            "bh_q4_replay": "PASS",
            "membership_rows_287000": "PASS",
            "random_panels_exclude_primary_anchor_union": "PASS",
            "within_panel_without_replacement": "PASS",
            "zero_fallback_replay": "PASS",
            "score_rows_78": "PASS",
            "score_ddof1_standardization": "PASS",
            "all_20_spearman_correlations_replayed_from_frozen_V1": "PASS",
            "scientific_firewalls": "PASS",
        },
        "program_results": EXPECTED["program_results"],
        "scientific_state": {
            "paper4_expression_or_weight_rows_read_in_audit": False,
            "target_expression_values_read_in_audit": False,
            "target_clinical_or_outcomes_read_in_audit": False,
            "model_fit_or_refit": False,
            "frozen_scientific_outputs_rewritten": False,
        },
    }

    audit_path = out_dir / "v1_biological_alignment_v3_readonly_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )

    audit_lock = {
        "status": "06E2A_V3_FROZEN_BIOLOGICAL_RESULT_AUDIT_LOCKED",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit": {
            "path": "method_contract/06e2a_v3/v1_biological_alignment_v3_readonly_audit.json",
            "sha256": sha256_file(audit_path),
            "size_bytes": audit_path.stat().st_size,
        },
        "source_result_lock": {
            "path":
                "method_contract/06e2_v3/"
                "v1_biological_alignment_random_controls_results_lock_v3.json",
            "sha256": sha256_file(result_lock_path),
        },
        "rewrite_policy":
            "Read-only audit. Never rewrite the frozen 06e1 v3 results or 06e2_v3 lock.",
    }
    audit_lock_path = (
        out_dir / "06e2a_v3_frozen_biological_result_audit_lock.json"
    )
    audit_lock_path.write_text(
        json.dumps(audit_lock, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Authorization predates scientific result: PASS")
    print("Exact v3 runner + authorization identity: PASS")
    print("Exact six frozen result-file hashes: PASS")
    print("4 programs x 5 coordinate correlations: PASS [20 rows]")
    print("4 programs x 1000 matched-random controls: PASS [4000 rows]")
    print("+1 empirical probabilities replay: PASS")
    print("BH q-values across four programs replay: PASS")
    print("Matched-panel memberships: PASS [287000 rows]")
    print("Primary-anchor exclusion from random panels: PASS")
    print("Within-panel sampling without replacement: PASS")
    print("Fallback replay: PASS [0]")
    print("Frozen strict program scores: PASS [78 cases x 4 programs]")
    print("All 20 Spearman correlations replayed from frozen V1: PASS")
    print("TARGET clinical/outcomes read in audit: NO")
    print("Model fit/refit in audit: NO")
    print("Frozen scientific outputs rewritten: NO")
    print("-" * 108)
    for program in EXPECTED["programs"]:
        row = EXPECTED["program_results"][program]
        print(
            f"{program}: max|rho|={row['observed_max_abs_spearman']:.6f}; "
            f"best={row['best_coordinate_id']}; "
            f"p={row['empirical_p_plus1']:.6f}; "
            f"q={row['bh_q_4']:.6f}; "
            f"matched-control-support={'YES' if row['support'] else 'NO'}"
        )
    print("=" * 108)
    print("Paper 5 06e2a_v3 frozen biological-result read-only audit: PASS + LOCKED")
    print("=" * 108)
    print(
        "Audit: method_contract/06e2a_v3/"
        "v1_biological_alignment_v3_readonly_audit.json"
    )
    print(
        "Audit lock: method_contract/06e2a_v3/"
        "06e2a_v3_frozen_biological_result_audit_lock.json"
    )
    print("=" * 108)


if __name__ == "__main__":
    main()
