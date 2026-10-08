#!/usr/bin/env python3
"""
Paper 5 - 06b3 read-only descriptive audit of frozen TARGET one-SE geometry
and outer held-out utility margins.

Purpose
-------
Clarify what the already-frozen 06b TARGET benchmark actually shows without
reopening model selection or fitting anything.

This stage:
- verifies exact frozen 06b summary/selection/outer/lock identities;
- audits 25 one-SE decisions and acceptance-set frequencies;
- aggregates the 700 frozen outer rows to split x model means across the
  already-frozen outer seeds;
- separates between-split variability from within-split seed variability;
- reports descriptive paired V1-minus-comparator outer utility margins;
- reports repeat-level summaries (5 repeated 5-fold CV repeats) without
  pretending the 25 folds are independent;
- never changes the 06b winner, 06c representation, final gate, biology, or
  any threshold.

No TARGET molecular matrix, outcome, clinical variable, or biological label is
read. No model is fit/refit. No new p-value or winner rule is created.

No command-line arguments.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "06b3-readonly-target-margin-audit-v1-descriptive-no-cli"

EXPECTED = {
    "selection_sha256":
        "8d6f17311cc98c3312619d3410f804e6581f152ce7811616b87ab9774a3e68c1",
    "outer_sha256":
        "b3c04b163392df043f78c8d377ff9f802afea2d7e95dcbff21fc5c87e504f2c5",
    "summary_sha256":
        "c2b350f8d0e957cf1a1005d0aa5f490d631a30d9e6607c12a90659c8037ca2e5",
    "lock_sha256":
        "2730c783f1a74f7aa58030d1ee03f324f1d50c4830c72e4229a15e41b60dc121",
    "runner_sha256":
        "4f69680a2d26d6c8e4bd5b337a5e94e6cdc36d149ea6a4b9af547b3855a93eab",
    "n_splits": 25,
    "n_repeats": 5,
    "folds_per_repeat": 5,
    "outer_rows": 700,
    "models": [
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
    ],
    "primary_model": "V1_SHARED_PRIVATE_AE",
    "metrics": [
        "utility",
        "heldout_reconstruction_score",
        "random_adjusted_cross_view_mrr",
        "missing_view_reconstruction_score",
        "modality_balance_score",
    ],
    "expected_outer_rows_per_model": {
        "BLOCK_PCA": 25,
        "MOFA_K5": 75,
        "MOFA_K10": 75,
        "MOFA_K15": 75,
        "MOFA_K20": 75,
        "V0_BALANCED_AE": 75,
        "V1_SHARED_PRIVATE_AE": 75,
        "V2_CONTRASTIVE_SPAE": 75,
        "V3_DROPOUT_SPAE": 75,
        "V4_SABLE_MO": 75,
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
        raise FileNotFoundError(f"Required frozen file missing: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    x = series.astype(str).str.strip().str.lower()
    if not x.isin({"true", "false", "1", "0", "yes", "no"}).all():
        raise RuntimeError(f"Cannot parse boolean series {series.name!r}.")
    return x.isin({"true", "1", "yes"})


def parse_acceptance(value: Any) -> list[str]:
    text = str(value).strip()
    if not text:
        return []
    return [x for x in text.split(";") if x]


def quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.quantile(x, q))


def descriptive_row(x: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise RuntimeError("Descriptive vector is empty/non-finite.")
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)) if x.size > 1 else np.nan,
        "median": float(np.median(x)),
        "q25": quantile(x, 0.25),
        "q75": quantile(x, 0.75),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def main() -> None:
    print("=" * 112)
    print("Paper 5 - 06b3 read-only TARGET one-SE + outer-margin descriptive audit")
    print("=" * 112)
    print("Script version:", SCRIPT_VERSION)
    print("TARGET molecular matrices read: NO")
    print("TARGET outcomes/clinical/biology read: NO")
    print("Model fit/refit: NO")
    print("06b selection changed/reopened: NO")
    print("New inferential p-value/winner rule: NO")

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    paper5 = project_root / "paper5"

    result_root = (
        paper5 / "results" / "target_representation" / "06b_primary_benchmark"
    )
    selection_path = (
        result_root / "target_inner_one_se_selection_by_outer_split.csv"
    )
    outer_path = result_root / "target_outer_fixed_model_benchmark.csv"
    summary_path = result_root / "target_outcome_blind_benchmark_summary.json"
    lock_path = (
        paper5 / "method_contract" / "06b"
        / "target_outcome_blind_benchmark_results_lock.json"
    )

    for path in [selection_path, outer_path, summary_path, lock_path]:
        require_file(path)

    observed_hashes = {
        "selection": sha256_file(selection_path),
        "outer": sha256_file(outer_path),
        "summary": sha256_file(summary_path),
        "lock": sha256_file(lock_path),
    }
    expected_hashes = {
        "selection": EXPECTED["selection_sha256"],
        "outer": EXPECTED["outer_sha256"],
        "summary": EXPECTED["summary_sha256"],
        "lock": EXPECTED["lock_sha256"],
    }
    for key in expected_hashes:
        if observed_hashes[key] != expected_hashes[key]:
            raise RuntimeError(
                f"Frozen 06b {key} SHA mismatch: "
                f"{observed_hashes[key]} != {expected_hashes[key]}"
            )

    summary = load_json(summary_path)
    lock = load_json(lock_path)

    runner_sha = str(
        lock.get("runner_source", {}).get("sha256", "")
    ).strip().lower()
    if runner_sha and runner_sha != EXPECTED["runner_sha256"]:
        raise RuntimeError("06b lock runner SHA mismatch.")

    sel = pd.read_csv(selection_path)
    outer = pd.read_csv(outer_path)

    # ------------------------------------------------------------------
    # Exact schema / firewall checks.
    # ------------------------------------------------------------------
    expected_sel_cols = [
        "repeat_index",
        "outer_fold",
        "split_id",
        "selected_model_id",
        "best_model_id",
        "best_mean",
        "best_se",
        "one_se_cutoff",
        "within_one_se",
    ]
    if list(sel.columns) != expected_sel_cols:
        raise RuntimeError(
            f"Selection schema changed: {list(sel.columns)}"
        )
    if sel.shape != (25, 9):
        raise RuntimeError(f"Selection shape changed: {sel.shape}")
    if sel["split_id"].nunique() != 25:
        raise RuntimeError("Selection split IDs are not unique.")
    if sorted(sel["repeat_index"].unique().tolist()) != [1, 2, 3, 4, 5]:
        raise RuntimeError("Unexpected repeat indices.")
    if sorted(sel["outer_fold"].unique().tolist()) != [1, 2, 3, 4, 5]:
        raise RuntimeError("Unexpected outer-fold indices.")

    required_outer_cols = [
        "biology_access",
        "failed",
        "heldout_reconstruction_score",
        "missing_view_reconstruction_score",
        "modality_balance_score",
        "model_id",
        "outcome_access",
        "q10_retention_applied",
        "random_adjusted_cross_view_mrr",
        "repeat_index",
        "seed",
        "selected_by_inner_cv",
        "split_id",
        "stage",
        "utility",
    ]
    missing_cols = [c for c in required_outer_cols if c not in outer.columns]
    if missing_cols:
        raise RuntimeError(f"Outer table missing columns: {missing_cols}")
    if outer.shape[0] != EXPECTED["outer_rows"]:
        raise RuntimeError(f"Outer row count changed: {outer.shape[0]}")
    if outer["split_id"].nunique() != 25:
        raise RuntimeError("Outer table must contain 25 unique splits.")
    if sorted(outer["model_id"].unique().tolist()) != sorted(EXPECTED["models"]):
        raise RuntimeError("Outer model set changed.")

    failed = bool_series(outer["failed"])
    if failed.any():
        raise RuntimeError(
            f"Frozen outer benchmark contains {int(failed.sum())} failed rows."
        )
    if bool_series(outer["biology_access"]).any():
        raise RuntimeError("Outer table reports biology access.")
    if bool_series(outer["outcome_access"]).any():
        raise RuntimeError("Outer table reports outcome access.")
    if bool_series(outer["q10_retention_applied"]).any():
        raise RuntimeError("Outer table reports q10 retention use.")
    if set(outer["stage"].astype(str)) != {"outer_evaluation"}:
        raise RuntimeError("Outer stage labels changed.")
    if set(outer["script_sha256"].astype(str).str.lower()) != {
        EXPECTED["runner_sha256"]
    }:
        raise RuntimeError("Outer rows contain unexpected runner SHA.")

    counts = outer["model_id"].value_counts().to_dict()
    if counts != EXPECTED["expected_outer_rows_per_model"]:
        raise RuntimeError(
            f"Outer row counts by model changed: {counts}"
        )

    # ------------------------------------------------------------------
    # One-SE geometry.
    # ------------------------------------------------------------------
    geometry_rows: list[dict[str, Any]] = []
    acceptance_counts = {m: 0 for m in EXPECTED["models"]}

    for row in sel.to_dict(orient="records"):
        accepted = parse_acceptance(row["within_one_se"])
        unknown = set(accepted) - set(EXPECTED["models"])
        if unknown:
            raise RuntimeError(
                f"{row['split_id']}: unknown models in one-SE set: {unknown}"
            )
        if row["selected_model_id"] not in accepted:
            raise RuntimeError(
                f"{row['split_id']}: selected model not in one-SE set."
            )
        for model in accepted:
            acceptance_counts[model] += 1

        geometry_rows.append(
            {
                "repeat_index": int(row["repeat_index"]),
                "outer_fold": int(row["outer_fold"]),
                "split_id": row["split_id"],
                "selected_model_id": row["selected_model_id"],
                "best_model_id": row["best_model_id"],
                "best_mean": float(row["best_mean"]),
                "best_se": float(row["best_se"]),
                "one_se_cutoff": float(row["one_se_cutoff"]),
                "one_se_width": float(row["best_se"]),
                "n_models_within_one_se": len(accepted),
                "V0_within_one_se": "V0_BALANCED_AE" in accepted,
                "V1_within_one_se": "V1_SHARED_PRIVATE_AE" in accepted,
                "V2_within_one_se": "V2_CONTRASTIVE_SPAE" in accepted,
                "V3_within_one_se": "V3_DROPOUT_SPAE" in accepted,
                "V4_within_one_se": "V4_SABLE_MO" in accepted,
                "within_one_se": ";".join(accepted),
            }
        )

    geometry = pd.DataFrame(geometry_rows)
    selection_counts = sel["selected_model_id"].value_counts().to_dict()
    best_counts = sel["best_model_id"].value_counts().to_dict()

    # Strong exact statement implied by V0 simpler than V1:
    # when V1 is selected, V0 must be outside the one-SE acceptable set.
    v1_selected = geometry["selected_model_id"].eq("V1_SHARED_PRIVATE_AE")
    if geometry.loc[v1_selected, "V0_within_one_se"].any():
        raise RuntimeError(
            "A V1-selected split unexpectedly contains V0 within one-SE."
        )

    acceptance_frequency = pd.DataFrame(
        [
            {
                "model_id": model,
                "within_one_se_count": int(acceptance_counts[model]),
                "within_one_se_fraction":
                    float(acceptance_counts[model] / EXPECTED["n_splits"]),
                "selected_count": int(selection_counts.get(model, 0)),
                "best_mean_count": int(best_counts.get(model, 0)),
            }
            for model in EXPECTED["models"]
        ]
    )

    # ------------------------------------------------------------------
    # Verify selected_by_inner_cv flags against the frozen selection table.
    # ------------------------------------------------------------------
    selected_flags = bool_series(outer["selected_by_inner_cv"])
    outer_flagged = outer.loc[selected_flags].copy()
    for split_id, part in outer_flagged.groupby("split_id", sort=False):
        models = sorted(part["model_id"].unique().tolist())
        expected_model = str(
            sel.loc[sel["split_id"].eq(split_id), "selected_model_id"].iloc[0]
        )
        if models != [expected_model]:
            raise RuntimeError(
                f"{split_id}: outer selected_by_inner_cv mismatch: {models}"
            )

    # ------------------------------------------------------------------
    # Split x model means across frozen outer seeds.
    # This is descriptive: no new selection uses these test-set means.
    # ------------------------------------------------------------------
    numeric_outer = outer.copy()
    for metric in EXPECTED["metrics"]:
        numeric_outer[metric] = pd.to_numeric(
            numeric_outer[metric], errors="raise"
        )
        if not np.isfinite(numeric_outer[metric].to_numpy(float)).all():
            raise RuntimeError(f"Non-finite outer metric: {metric}")

    split_model_rows: list[dict[str, Any]] = []
    for (repeat_index, outer_fold, split_id, model_id), part in (
        numeric_outer.groupby(
            ["repeat_index", "outer_fold", "split_id", "model_id"],
            sort=False,
        )
    ):
        seeds = sorted(pd.to_numeric(part["seed"]).astype(int).tolist())
        expected_n = 1 if model_id == "BLOCK_PCA" else 3
        if len(seeds) != expected_n or len(set(seeds)) != expected_n:
            raise RuntimeError(
                f"{split_id}/{model_id}: unexpected frozen seed rows {seeds}"
            )
        row: dict[str, Any] = {
            "repeat_index": int(repeat_index),
            "outer_fold": int(outer_fold),
            "split_id": str(split_id),
            "model_id": str(model_id),
            "n_outer_seed_rows": int(part.shape[0]),
            "seeds": ";".join(str(x) for x in seeds),
        }
        for metric in EXPECTED["metrics"]:
            x = part[metric].to_numpy(dtype=float)
            row[f"{metric}_seed_mean"] = float(np.mean(x))
            row[f"{metric}_seed_sd"] = (
                float(np.std(x, ddof=1)) if x.size > 1 else np.nan
            )
            row[f"{metric}_seed_min"] = float(np.min(x))
            row[f"{metric}_seed_max"] = float(np.max(x))
        split_model_rows.append(row)

    split_model = pd.DataFrame(split_model_rows)
    if split_model.shape[0] != 250:
        raise RuntimeError(
            f"Expected 25 x 10 = 250 split-model rows, got {split_model.shape[0]}."
        )

    # ------------------------------------------------------------------
    # Model summaries: between-split variability + within-split seed variability.
    # ------------------------------------------------------------------
    summary_rows: list[dict[str, Any]] = []
    for model in EXPECTED["models"]:
        part = split_model.loc[split_model["model_id"].eq(model)].copy()
        if part.shape[0] != 25:
            raise RuntimeError(f"{model}: expected 25 split-level rows.")
        for metric in EXPECTED["metrics"]:
            split_means = part[f"{metric}_seed_mean"].to_numpy(float)
            desc = descriptive_row(split_means)
            seed_sd = pd.to_numeric(
                part[f"{metric}_seed_sd"], errors="coerce"
            ).to_numpy(float)
            finite_seed_sd = seed_sd[np.isfinite(seed_sd)]
            summary_rows.append(
                {
                    "model_id": model,
                    "metric": metric,
                    "n_splits": 25,
                    "split_mean": desc["mean"],
                    "between_split_sd": desc["sd"],
                    "split_median": desc["median"],
                    "split_q25": desc["q25"],
                    "split_q75": desc["q75"],
                    "split_min": desc["min"],
                    "split_max": desc["max"],
                    "mean_within_split_seed_sd": (
                        float(np.mean(finite_seed_sd))
                        if finite_seed_sd.size else np.nan
                    ),
                    "median_within_split_seed_sd": (
                        float(np.median(finite_seed_sd))
                        if finite_seed_sd.size else np.nan
                    ),
                    "max_within_split_seed_sd": (
                        float(np.max(finite_seed_sd))
                        if finite_seed_sd.size else np.nan
                    ),
                }
            )
    model_metric_summary = pd.DataFrame(summary_rows)

    # ------------------------------------------------------------------
    # Paired V1-minus-comparator outer utility margins.
    # Descriptive only. We report both 25 split deltas and 5 repeat means.
    # ------------------------------------------------------------------
    utility_wide = (
        split_model.pivot(
            index=["repeat_index", "outer_fold", "split_id"],
            columns="model_id",
            values="utility_seed_mean",
        )
        .reset_index()
    )
    if utility_wide.shape[0] != 25:
        raise RuntimeError("Utility pivot did not produce 25 split rows.")

    comparator_models = [
        m for m in EXPECTED["models"] if m != EXPECTED["primary_model"]
    ]
    delta_rows: list[dict[str, Any]] = []
    delta_summary_rows: list[dict[str, Any]] = []

    for comparator in comparator_models:
        delta = (
            utility_wide[EXPECTED["primary_model"]].to_numpy(float)
            - utility_wide[comparator].to_numpy(float)
        )
        for idx, base in utility_wide.iterrows():
            delta_rows.append(
                {
                    "repeat_index": int(base["repeat_index"]),
                    "outer_fold": int(base["outer_fold"]),
                    "split_id": str(base["split_id"]),
                    "reference_model": EXPECTED["primary_model"],
                    "comparator_model": comparator,
                    "v1_utility": float(base[EXPECTED["primary_model"]]),
                    "comparator_utility": float(base[comparator]),
                    "delta_v1_minus_comparator": float(delta[idx]),
                }
            )

        desc = descriptive_row(delta)
        temp = pd.DataFrame(
            {
                "repeat_index": utility_wide["repeat_index"].astype(int),
                "delta": delta,
            }
        )
        repeat_means = (
            temp.groupby("repeat_index", sort=True)["delta"].mean().to_numpy(float)
        )
        if repeat_means.size != 5:
            raise RuntimeError("Expected five repeat-level paired means.")
        rep_desc = descriptive_row(repeat_means)

        delta_summary_rows.append(
            {
                "reference_model": EXPECTED["primary_model"],
                "comparator_model": comparator,
                "n_outer_splits": 25,
                "mean_delta_v1_minus_comparator": desc["mean"],
                "sd_delta_across_splits": desc["sd"],
                "median_delta": desc["median"],
                "q25_delta": desc["q25"],
                "q75_delta": desc["q75"],
                "min_delta": desc["min"],
                "max_delta": desc["max"],
                "fraction_v1_higher": float(np.mean(delta > 0.0)),
                "fraction_equal_exact": float(np.mean(delta == 0.0)),
                "fraction_v1_lower": float(np.mean(delta < 0.0)),
                "repeat_mean_delta": rep_desc["mean"],
                "sd_of_5_repeat_mean_deltas": rep_desc["sd"],
                "min_repeat_mean_delta": rep_desc["min"],
                "max_repeat_mean_delta": rep_desc["max"],
                "interpretation_role":
                    "DESCRIPTIVE_POST_SELECTION_OUTER_TEST_MARGIN_ONLY",
            }
        )

    deltas = pd.DataFrame(delta_rows)
    delta_summary = pd.DataFrame(delta_summary_rows)

    # ------------------------------------------------------------------
    # Direct V4-vs-V1 seed variability comparison.
    # ------------------------------------------------------------------
    seed_var_rows: list[dict[str, Any]] = []
    for metric in EXPECTED["metrics"]:
        v1 = split_model.loc[
            split_model["model_id"].eq("V1_SHARED_PRIVATE_AE"),
            ["split_id", f"{metric}_seed_sd"],
        ].rename(columns={f"{metric}_seed_sd": "V1_seed_sd"})
        v4 = split_model.loc[
            split_model["model_id"].eq("V4_SABLE_MO"),
            ["split_id", f"{metric}_seed_sd"],
        ].rename(columns={f"{metric}_seed_sd": "V4_seed_sd"})
        joined = v1.merge(v4, on="split_id", how="inner", validate="one_to_one")
        if joined.shape[0] != 25:
            raise RuntimeError("V1/V4 seed-SD join did not yield 25 splits.")
        a = joined["V1_seed_sd"].to_numpy(float)
        b = joined["V4_seed_sd"].to_numpy(float)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise RuntimeError("V1/V4 seed variability contains non-finite values.")
        seed_var_rows.append(
            {
                "metric": metric,
                "mean_V1_within_split_seed_sd": float(np.mean(a)),
                "mean_V4_within_split_seed_sd": float(np.mean(b)),
                "median_V1_within_split_seed_sd": float(np.median(a)),
                "median_V4_within_split_seed_sd": float(np.median(b)),
                "mean_delta_V4_minus_V1_seed_sd": float(np.mean(b - a)),
                "median_delta_V4_minus_V1_seed_sd": float(np.median(b - a)),
                "fraction_splits_V4_seed_sd_gt_V1":
                    float(np.mean(b > a)),
                "role": "DESCRIPTIVE_SEED_VARIABILITY_ONLY",
            }
        )
    v1_v4_seed_variability = pd.DataFrame(seed_var_rows)

    # ------------------------------------------------------------------
    # Save only after every check/calculation succeeds.
    # ------------------------------------------------------------------
    out_dir = paper5 / "method_contract" / "06b3"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(
            f"06b3 directory already exists and is non-empty: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "target_inner_one_se_geometry.csv": geometry,
        "target_inner_one_se_model_frequency.csv": acceptance_frequency,
        "target_outer_split_model_metrics.csv": split_model,
        "target_outer_model_metric_summary.csv": model_metric_summary,
        "target_outer_v1_pairwise_utility_deltas.csv": deltas,
        "target_outer_v1_pairwise_utility_summary.csv": delta_summary,
        "target_outer_v1_v4_seed_variability.csv": v1_v4_seed_variability,
    }
    for filename, frame in outputs.items():
        frame.to_csv(out_dir / filename, index=False)

    audit_payload = {
        "status": "06B3_TARGET_READONLY_MARGIN_AUDIT_COMPLETE",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": observed_hashes,
        "scientific_state": {
            "target_molecular_matrices_read": False,
            "target_outcomes_clinical_biology_read": False,
            "model_fit_or_refit": False,
            "06b_selection_reopened": False,
            "new_winner_rule_created": False,
            "new_p_value_created": False,
        },
        "selection_counts": {
            str(k): int(v) for k, v in selection_counts.items()
        },
        "best_model_counts": {
            str(k): int(v) for k, v in best_counts.items()
        },
        "within_one_se_counts": {
            str(k): int(v) for k, v in acceptance_counts.items()
        },
        "key_exact_implications": {
            "v1_selected_count":
                int(selection_counts.get("V1_SHARED_PRIVATE_AE", 0)),
            "v0_selected_count":
                int(selection_counts.get("V0_BALANCED_AE", 0)),
            "v0_below_one_se_cutoff_when_v1_selected_count":
                int(v1_selected.sum()),
            "reason":
                "V0 is simpler than V1; if V1 is selected by the frozen "
                "simplest-within-one-SE rule, V0 cannot be in the accepted set.",
            "v4_best_mean_count":
                int(best_counts.get("V4_SABLE_MO", 0)),
            "v4_within_one_se_count":
                int(acceptance_counts.get("V4_SABLE_MO", 0)),
        },
        "interpretation_boundary": {
            "outer_margins":
                "Post-selection descriptive summaries of already-frozen outer "
                "test rows. They do not alter 06b selection.",
            "variance":
                "Between-split SD and within-split seed SD are descriptive; "
                "no formal variance-equality test is introduced post hoc.",
            "repeated_cv":
                "The 25 outer folds are not treated as independent inferential "
                "replicates. Five repeat-level mean deltas are reported separately.",
        },
    }
    audit_path = out_dir / "target_06b_readonly_margin_audit.json"
    audit_path.write_text(
        json.dumps(audit_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    generated_paths = [out_dir / name for name in outputs] + [audit_path]
    lock_payload = {
        "status": "06B3_TARGET_READONLY_MARGIN_AUDIT_FROZEN",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": observed_hashes,
        "generated_files": {
            p.name: {
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
            }
            for p in generated_paths
        },
        "rewrite_policy":
            "Read-only descriptive audit of frozen 06b outputs. Never rewrite "
            "06b/06c scientific results or use this audit to reopen selection.",
    }
    lock_out = out_dir / "06b3_target_readonly_margin_audit_lock.json"
    lock_out.write_text(
        json.dumps(lock_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Console headline.
    # ------------------------------------------------------------------
    print("Exact frozen 06b hashes: PASS")
    print("Outer failures: 0/700")
    print("Outcome/biology access in outer rows: NO")
    print()
    print("-" * 112)
    print("INNER one-SE geometry")
    print("-" * 112)
    print("Selected-model counts:")
    for model, count in sorted(
        selection_counts.items(), key=lambda kv: (-kv[1], kv[0])
    ):
        print(f"  {model}: {count}/25")
    print("Best-inner-mean counts:")
    for model in EXPECTED["models"]:
        count = int(best_counts.get(model, 0))
        if count:
            print(f"  {model}: {count}/25")
    print("Within-one-SE acceptance counts:")
    for model in EXPECTED["models"]:
        count = int(acceptance_counts.get(model, 0))
        print(f"  {model}: {count}/25")
    print(
        "V0 below one-SE cutoff on every V1-selected split: "
        f"YES [{int(v1_selected.sum())}/25 splits]"
    )
    print(
        "V4 achieved best inner mean: "
        f"{int(best_counts.get('V4_SABLE_MO', 0))}/25"
    )
    print(
        "V4 was nevertheless within one-SE on: "
        f"{int(acceptance_counts.get('V4_SABLE_MO', 0))}/25"
    )

    print()
    print("-" * 112)
    print("OUTER held-out utility: V1 minus comparator [seed-mean within each split]")
    print("-" * 112)
    cols = [
        "comparator_model",
        "mean_delta_v1_minus_comparator",
        "sd_delta_across_splits",
        "median_delta",
        "fraction_v1_higher",
        "repeat_mean_delta",
        "sd_of_5_repeat_mean_deltas",
    ]
    print(delta_summary[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print()
    print("-" * 112)
    print("V1 vs V4 outer utility variability")
    print("-" * 112)
    util_var = v1_v4_seed_variability.loc[
        v1_v4_seed_variability["metric"].eq("utility")
    ].iloc[0]
    v1_between = model_metric_summary.loc[
        model_metric_summary["model_id"].eq("V1_SHARED_PRIVATE_AE")
        & model_metric_summary["metric"].eq("utility")
    ].iloc[0]
    v4_between = model_metric_summary.loc[
        model_metric_summary["model_id"].eq("V4_SABLE_MO")
        & model_metric_summary["metric"].eq("utility")
    ].iloc[0]
    print(
        f"Between-split SD of seed-mean utility: "
        f"V1={v1_between['between_split_sd']:.6f} | "
        f"V4={v4_between['between_split_sd']:.6f}"
    )
    print(
        f"Mean within-split seed SD of utility: "
        f"V1={util_var['mean_V1_within_split_seed_sd']:.6f} | "
        f"V4={util_var['mean_V4_within_split_seed_sd']:.6f}"
    )
    print(
        f"Fraction splits with V4 seed-SD > V1 seed-SD: "
        f"{util_var['fraction_splits_V4_seed_sd_gt_V1']:.3f}"
    )

    print()
    print("TARGET molecular matrices read: NO")
    print("TARGET outcomes/clinical/biology read: NO")
    print("Model fit/refit: NO")
    print("06b winner changed: NO")
    print("=" * 112)
    print("Paper 5 06b3 read-only TARGET margin audit: COMPLETE + FROZEN")
    print("=" * 112)
    print("Output directory: method_contract/06b3")
    print("=" * 112)


if __name__ == "__main__":
    main()
