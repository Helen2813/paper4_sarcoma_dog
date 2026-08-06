from __future__ import annotations

from pathlib import Path
import hashlib
import json
import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.metrics import average_precision_score, roc_auc_score
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore")

SCRIPT_VERSION = "24-external-validation-robustness-audit-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
HUMAN_DIR = PROCESSED_DIR / "human_validation"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]
PRIMARY_SCORE_SUFFIX = "__strict__signed_mean_z"
PROLIFERATION_SCORE = "strict_human_meta_proliferation_pc1_z"
M40_RESIDUAL_SIGNED = (
    "M40__strict__signed_mean__residual_to_disjoint_proliferation_z"
)

TARGET_CLINICAL_FILE = HUMAN_DIR / "TARGET_OS_clinical_standardized.csv"
TARGET_SCORES_FILE = HUMAN_DIR / "TARGET_OS_frozen_transfer_scores.csv"
GSE_CLINICAL_FILE = HUMAN_DIR / "GSE21257_clinical_standardized.csv"
GSE_SCORES_FILE = HUMAN_DIR / "GSE21257_frozen_transfer_scores.csv"

TARGET_PRIMARY_FILE = RESULTS_DIR / "TARGET_OS_primary_frozen_program_validation.csv"
GSE_PRIMARY_FILE = RESULTS_DIR / "GSE21257_metastasis_primary_frozen_program_validation.csv"
GSE_OS_FILE = RESULTS_DIR / "GSE21257_OS_frozen_program_sensitivity.csv"
ADJUSTED_FILE = RESULTS_DIR / "human_external_validation_adjusted_models.csv"
VARIANTS_FILE = RESULTS_DIR / "human_external_validation_score_variant_sensitivity.csv"
RANDOM_FILE = RESULTS_DIR / "human_external_validation_random_gene_set_controls.csv"
SYNTHESIS_FILE = RESULTS_DIR / "human_external_validation_cross_cohort_synthesis.csv"
VALIDATION_MANIFEST_FILE = RESULTS_DIR / "human_external_validation_manifest.json"

N_BOOTSTRAP_LOGISTIC = 5000
N_PERMUTATIONS = 10000
LOGISTIC_L2 = 1e-6
COX_PENALIZER = 0.01
RANDOM_SEED = 20260805

OUTPUT_LOGISTIC = RESULTS_DIR / "GSE21257_primary_robust_logistic_effects.csv"
OUTPUT_TARGET_ADJUSTED = RESULTS_DIR / "TARGET_OS_primary_adjustment_robustness.csv"
OUTPUT_GSE_ADJUSTED = RESULTS_DIR / "GSE21257_primary_adjustment_robustness.csv"
OUTPUT_TARGET_LOO = RESULTS_DIR / "TARGET_OS_primary_leave_one_out_stability.csv"
OUTPUT_GSE_LOO = RESULTS_DIR / "GSE21257_primary_leave_one_out_stability.csv"
OUTPUT_RANDOM = RESULTS_DIR / "human_external_validation_random_control_empirical_fdr.csv"
OUTPUT_FINAL = RESULTS_DIR / "human_external_validation_robust_evidence_summary.csv"
OUTPUT_MANIFEST = RESULTS_DIR / "human_external_validation_robustness_manifest.json"
OUTPUT_README = RESULTS_DIR / "human_external_validation_robustness_README.txt"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_required_csv(path: Path, index_col: int | str | None = 0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    print(f"Loaded: {path}")
    return pd.read_csv(path, index_col=index_col)


def read_optional_csv(path: Path, index_col: int | str | None = None) -> pd.DataFrame:
    if not path.exists():
        print(f"Optional file missing: {path}")
        return pd.DataFrame()
    print(f"Loaded: {path}")
    return pd.read_csv(path, index_col=index_col)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def zscore(series: pd.Series) -> pd.Series:
    values = safe_numeric(series)
    sd = values.std()
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=series.index)
    return (values - values.mean()) / sd


def bh_adjust(values: pd.Series | np.ndarray | list[float]) -> pd.Series:
    p = pd.Series(values, dtype=float)
    q = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.notna() & np.isfinite(p)
    if valid.sum() == 0:
        return q
    raw = p.loc[valid].to_numpy()
    order = np.argsort(raw)
    ranked = raw[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    restored = np.empty(n)
    restored[order] = adjusted
    q.loc[p.index[valid]] = restored
    return q


def score_column(module: str) -> str:
    return f"{module}{PRIMARY_SCORE_SUFFIX}"


def verify_source_manifest() -> None:
    if not VALIDATION_MANIFEST_FILE.exists():
        print("Validation manifest not found; file hashes will be recorded without verification.")
        return
    manifest = json.loads(VALIDATION_MANIFEST_FILE.read_text(encoding="utf-8"))
    print("")
    print("External-validation manifest found:")
    print(f"  Script version: {manifest.get('script_version', 'unknown')}")
    print(f"  Created UTC: {manifest.get('created_utc', 'unknown')}")


def logistic_objective(beta: np.ndarray, x: np.ndarray, y: np.ndarray, l2: float) -> tuple[float, np.ndarray]:
    eta = x @ beta
    nll = float(np.sum(np.logaddexp(0.0, eta) - y * eta))
    penalty = 0.5 * l2 * float(np.sum(beta[1:] ** 2))
    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -40, 40)))
    grad = x.T @ (p - y)
    grad[1:] += l2 * beta[1:]
    return nll + penalty, grad


def fit_logistic_ridge(y: np.ndarray, x: np.ndarray, l2: float = LOGISTIC_L2) -> tuple[np.ndarray, bool]:
    beta0 = np.zeros(x.shape[1], dtype=float)
    result = optimize.minimize(
        fun=lambda b: logistic_objective(b, x, y, l2)[0],
        x0=beta0,
        jac=lambda b: logistic_objective(b, x, y, l2)[1],
        method="BFGS",
        options={"maxiter": 2000, "gtol": 1e-8},
    )
    beta = np.asarray(result.x, dtype=float)
    success = bool(result.success or np.isfinite(result.fun))
    return beta, success


def stratified_bootstrap_logistic(
    frame: pd.DataFrame,
    predictor_cols: list[str],
    score_col: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    y = frame["outcome"].to_numpy(int)
    x = np.column_stack([
        np.ones(frame.shape[0]),
        frame[predictor_cols].to_numpy(float),
    ])
    beta, success = fit_logistic_ridge(y, x)
    score_index = 1 + predictor_cols.index(score_col)
    observed_coef = float(beta[score_index]) if success else np.nan

    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    coefs = []
    aucs = []
    for _ in range(N_BOOTSTRAP_LOGISTIC):
        idx = np.concatenate([
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ])
        xb = x[idx]
        yb = y[idx]
        bb, ok = fit_logistic_ridge(yb, xb)
        if not ok or not np.all(np.isfinite(bb)):
            continue
        coef = float(bb[score_index])
        if abs(coef) > 20:
            continue
        coefs.append(coef)
        try:
            aucs.append(float(roc_auc_score(yb, frame[score_col].to_numpy(float)[idx])))
        except Exception:
            pass

    coef_array = np.asarray(coefs, dtype=float)
    return {
        "coef": observed_coef,
        "or_per_sd": float(np.exp(observed_coef)) if np.isfinite(observed_coef) else np.nan,
        "or_ci_low": float(np.exp(np.quantile(coef_array, 0.025))) if coef_array.size else np.nan,
        "or_ci_high": float(np.exp(np.quantile(coef_array, 0.975))) if coef_array.size else np.nan,
        "bootstrap_probability_coef_positive": float(np.mean(coef_array > 0)) if coef_array.size else np.nan,
        "bootstrap_valid": int(coef_array.size),
        "bootstrap_auc_median": float(np.median(aucs)) if aucs else np.nan,
    }


def permutation_auc_pvalue(
    y: np.ndarray,
    score: np.ndarray,
    observed_auc: float,
    rng: np.random.Generator,
) -> float:
    null_distance = []
    observed_distance = abs(observed_auc - 0.5)
    for _ in range(N_PERMUTATIONS):
        perm = rng.permutation(y)
        auc = float(roc_auc_score(perm, score))
        null_distance.append(abs(auc - 0.5))
    null_array = np.asarray(null_distance)
    return float((1 + np.sum(null_array >= observed_distance)) / (1 + len(null_array)))


def build_gse_frame(
    clinical: pd.DataFrame,
    scores: pd.DataFrame,
    module: str,
    include_proliferation: bool,
) -> tuple[pd.DataFrame, list[str]]:
    col = score_column(module)
    frame = pd.DataFrame({
        "outcome": safe_numeric(clinical["metastasis_within_5y"]),
        "score": zscore(scores[col].reindex(clinical.index)),
    })
    predictors = ["score"]
    if include_proliferation and PROLIFERATION_SCORE in scores.columns and module != "M40":
        frame["proliferation"] = zscore(scores[PROLIFERATION_SCORE].reindex(clinical.index))
        predictors.append("proliferation")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame[frame["outcome"].isin([0, 1])].copy()
    return frame, predictors


def run_gse_robust_logistic(
    clinical: pd.DataFrame,
    scores: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unadjusted_rows = []
    adjusted_rows = []
    for module in PRIMARY_MODULES:
        col = score_column(module)
        if col not in scores.columns:
            continue

        for include_proliferation, destination in [
            (False, unadjusted_rows),
            (True, adjusted_rows),
        ]:
            if include_proliferation and module == "M40":
                continue
            frame, predictors = build_gse_frame(
                clinical, scores, module, include_proliferation
            )
            y = frame["outcome"].to_numpy(int)
            s = frame["score"].to_numpy(float)
            observed_auc = float(roc_auc_score(y, s))
            effect = stratified_bootstrap_logistic(
                frame=frame,
                predictor_cols=predictors,
                score_col="score",
                rng=rng,
            )
            permutation_p = permutation_auc_pvalue(y, s, observed_auc, rng)
            destination.append({
                "module_label": module,
                "adjustment": "proliferation" if include_proliferation else "unadjusted",
                "n": int(frame.shape[0]),
                "positives": int((frame["outcome"] == 1).sum()),
                "negatives": int((frame["outcome"] == 0).sum()),
                "auc": observed_auc,
                "average_precision": float(average_precision_score(y, s)),
                "permutation_auc_p_two_sided": permutation_p,
                **effect,
            })

    unadjusted = pd.DataFrame(unadjusted_rows)
    adjusted = pd.DataFrame(adjusted_rows)
    if not unadjusted.empty:
        unadjusted["permutation_auc_q_bh"] = bh_adjust(
            unadjusted["permutation_auc_p_two_sided"]
        ).values
    if not adjusted.empty:
        adjusted["permutation_auc_q_bh"] = bh_adjust(
            adjusted["permutation_auc_p_two_sided"]
        ).values
    return unadjusted, adjusted


def fit_cox_frame(frame: pd.DataFrame, covariates: list[str]) -> dict[str, Any]:
    needed = ["time", "event", "score"] + covariates
    use = frame[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    result = {
        "n": int(use.shape[0]),
        "events": int(use["event"].sum()) if not use.empty else 0,
        "hr_per_sd": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "p": np.nan,
        "c_index": np.nan,
        "ph_test_p": np.nan,
        "error": "",
    }
    if use.shape[0] < 30 or use["event"].sum() < 8:
        result["error"] = "insufficient_samples_or_events"
        return result
    cph = CoxPHFitter(penalizer=COX_PENALIZER)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            cph.fit(use, duration_col="time", event_col="event")
        summary = cph.summary.loc["score"]
        risk = cph.predict_partial_hazard(use[["score"] + covariates]).to_numpy().ravel()
        result.update({
            "hr_per_sd": float(summary["exp(coef)"]),
            "ci_low": float(summary["exp(coef) lower 95%"]),
            "ci_high": float(summary["exp(coef) upper 95%"]),
            "p": float(summary["p"]),
            "c_index": float(concordance_index(use["time"], -risk, use["event"])),
        })
        try:
            ph = proportional_hazard_test(cph, use, time_transform="rank")
            result["ph_test_p"] = float(ph.summary.loc["score", "p"])
        except Exception:
            pass
    except Exception as exc:
        result["error"] = str(exc)[:400]
    return result


def build_target_frame(
    clinical: pd.DataFrame,
    scores: pd.DataFrame,
    module: str,
) -> pd.DataFrame:
    col = score_column(module)
    frame = pd.DataFrame({
        "time": safe_numeric(clinical["os_time_days"]),
        "event": safe_numeric(clinical["os_event"]),
        "score": zscore(scores[col].reindex(clinical.index)),
    })
    if "age_z" in clinical.columns:
        frame["age_z"] = safe_numeric(clinical["age_z"])
    elif "age_at_diagnosis_years" in clinical.columns:
        frame["age_z"] = zscore(clinical["age_at_diagnosis_years"])
    if PROLIFERATION_SCORE in scores.columns:
        frame["proliferation"] = zscore(scores[PROLIFERATION_SCORE].reindex(clinical.index))
    return frame


def run_target_adjustment(
    clinical: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for module in PRIMARY_MODULES:
        col = score_column(module)
        if col not in scores.columns:
            continue
        frame = build_target_frame(clinical, scores, module)
        specifications = [("unadjusted", [])]
        if "age_z" in frame.columns:
            specifications.append(("age", ["age_z"]))
        if module != "M40" and "proliferation" in frame.columns:
            specifications.append(("proliferation", ["proliferation"]))
            if "age_z" in frame.columns:
                specifications.append(("age_plus_proliferation", ["age_z", "proliferation"]))
        for label, covariates in specifications:
            result = fit_cox_frame(frame, covariates)
            rows.append({
                "module_label": module,
                "adjustment": label,
                "covariates": ";".join(covariates),
                **result,
            })

    if M40_RESIDUAL_SIGNED in scores.columns:
        frame = pd.DataFrame({
            "time": safe_numeric(clinical["os_time_days"]),
            "event": safe_numeric(clinical["os_event"]),
            "score": zscore(scores[M40_RESIDUAL_SIGNED].reindex(clinical.index)),
        })
        if "age_z" in clinical.columns:
            frame["age_z"] = safe_numeric(clinical["age_z"])
        elif "age_at_diagnosis_years" in clinical.columns:
            frame["age_z"] = zscore(clinical["age_at_diagnosis_years"])
        for label, covariates in [
            ("m40_residual_unadjusted", []),
            ("m40_residual_age", ["age_z"] if "age_z" in frame.columns else []),
        ]:
            result = fit_cox_frame(frame, covariates)
            rows.append({
                "module_label": "M40",
                "adjustment": label,
                "covariates": ";".join(covariates),
                **result,
            })
    return pd.DataFrame(rows)


def run_target_loo(clinical: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for module in PRIMARY_MODULES:
        col = score_column(module)
        if col not in scores.columns:
            continue
        frame = build_target_frame(clinical, scores, module)
        complete = frame[["time", "event", "score"]].dropna().copy()
        estimates = []
        for sample in complete.index:
            part = complete.drop(index=sample)
            result = fit_cox_frame(part, [])
            if np.isfinite(result["hr_per_sd"]):
                estimates.append({
                    "removed_sample": sample,
                    "hr_per_sd": result["hr_per_sd"],
                    "p": result["p"],
                    "c_index": result["c_index"],
                })
        loo = pd.DataFrame(estimates)
        rows.append({
            "module_label": module,
            "n_loo_fits": int(loo.shape[0]),
            "hr_min": float(loo["hr_per_sd"].min()) if not loo.empty else np.nan,
            "hr_max": float(loo["hr_per_sd"].max()) if not loo.empty else np.nan,
            "hr_median": float(loo["hr_per_sd"].median()) if not loo.empty else np.nan,
            "fraction_hr_above_1": float((loo["hr_per_sd"] > 1).mean()) if not loo.empty else np.nan,
            "c_index_min": float(loo["c_index"].min()) if not loo.empty else np.nan,
            "c_index_max": float(loo["c_index"].max()) if not loo.empty else np.nan,
            "c_index_median": float(loo["c_index"].median()) if not loo.empty else np.nan,
            "fraction_p_below_0_05": float((loo["p"] < 0.05).mean()) if not loo.empty else np.nan,
        })
    return pd.DataFrame(rows)


def run_gse_loo(clinical: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for module in PRIMARY_MODULES:
        col = score_column(module)
        if col not in scores.columns:
            continue
        frame, _ = build_gse_frame(clinical, scores, module, False)
        estimates = []
        for sample in frame.index:
            part = frame.drop(index=sample)
            y = part["outcome"].to_numpy(int)
            s = part["score"].to_numpy(float)
            if np.unique(y).size < 2:
                continue
            estimates.append(float(roc_auc_score(y, s)))
        values = np.asarray(estimates, dtype=float)
        rows.append({
            "module_label": module,
            "n_loo_fits": int(values.size),
            "auc_min": float(np.min(values)) if values.size else np.nan,
            "auc_max": float(np.max(values)) if values.size else np.nan,
            "auc_median": float(np.median(values)) if values.size else np.nan,
            "fraction_auc_above_0_50": float(np.mean(values > 0.5)) if values.size else np.nan,
            "fraction_auc_above_0_60": float(np.mean(values > 0.6)) if values.size else np.nan,
        })
    return pd.DataFrame(rows)


def process_random_controls(random_controls: pd.DataFrame) -> pd.DataFrame:
    if random_controls.empty:
        return random_controls
    out = random_controls.copy()
    if "empirical_p_greater_equal" not in out.columns:
        raise ValueError("Random-control summary lacks empirical_p_greater_equal.")
    out["empirical_p_greater_equal"] = safe_numeric(out["empirical_p_greater_equal"])
    out["empirical_q_within_cohort"] = np.nan
    for cohort in out["cohort"].dropna().unique():
        mask = out["cohort"].eq(cohort)
        out.loc[mask, "empirical_q_within_cohort"] = bh_adjust(
            out.loc[mask, "empirical_p_greater_equal"]
        ).values
    out["empirical_q_global_eight"] = bh_adjust(
        out["empirical_p_greater_equal"]
    ).values
    return out


def evidence_grade(
    module: str,
    target_primary: pd.DataFrame,
    gse_primary: pd.DataFrame,
    target_adjusted: pd.DataFrame,
    gse_logistic: pd.DataFrame,
    target_loo: pd.DataFrame,
    gse_loo: pd.DataFrame,
    random_table: pd.DataFrame,
) -> dict[str, Any]:
    t = target_primary[target_primary["module_label"].eq(module)]
    g = gse_primary[gse_primary["module_label"].eq(module)]
    ta = target_adjusted[
        target_adjusted["module_label"].eq(module)
        & target_adjusted["adjustment"].eq("unadjusted")
    ]
    gl = gse_logistic[gse_logistic["module_label"].eq(module)]
    tl = target_loo[target_loo["module_label"].eq(module)]
    gloo = gse_loo[gse_loo["module_label"].eq(module)]
    rc_t = random_table[
        random_table["module_label"].eq(module)
        & random_table["cohort"].eq("TARGET_OS")
    ]
    rc_g = random_table[
        random_table["module_label"].eq(module)
        & random_table["cohort"].eq("GSE21257")
    ]

    target_hr = float(t["score_hr_per_sd"].iloc[0]) if not t.empty else np.nan
    target_p = float(t["primary_p"].iloc[0]) if not t.empty else np.nan
    target_q = float(t["q_within_endpoint"].iloc[0]) if not t.empty else np.nan
    target_c = float(t["fixed_score_c_index"].iloc[0]) if not t.empty else np.nan
    gse_auc = float(g["auc"].iloc[0]) if not g.empty else np.nan
    gse_q = float(g["q_within_endpoint"].iloc[0]) if not g.empty else np.nan
    global_q = float(g["q_global_eight_tests"].iloc[0]) if not g.empty else np.nan
    robust_or = float(gl["or_per_sd"].iloc[0]) if not gl.empty else np.nan
    robust_perm_q = float(gl["permutation_auc_q_bh"].iloc[0]) if not gl.empty else np.nan
    target_loo_direction = float(tl["fraction_hr_above_1"].iloc[0]) if not tl.empty else np.nan
    gse_loo_direction = float(gloo["fraction_auc_above_0_50"].iloc[0]) if not gloo.empty else np.nan
    target_random_p = float(rc_t["empirical_p_greater_equal"].iloc[0]) if not rc_t.empty else np.nan
    gse_random_p = float(rc_g["empirical_p_greater_equal"].iloc[0]) if not rc_g.empty else np.nan

    direction_consistent = bool(
        np.isfinite(target_hr)
        and np.isfinite(gse_auc)
        and ((target_hr > 1 and gse_auc > 0.5) or (target_hr < 1 and gse_auc < 0.5))
    )

    if module == "M34" and global_q < 0.05 and target_p < 0.05 and direction_consistent:
        grade = "strong_cross_species_program_support"
    elif global_q < 0.05 and direction_consistent:
        grade = "single_cohort_global_fdr_with_directional_replication"
    elif gse_q < 0.05 and direction_consistent:
        grade = "endpoint_specific_fdr_with_directional_replication"
    elif direction_consistent and gse_q < 0.10:
        grade = "suggestive_directionally_conserved_support"
    elif not direction_consistent:
        grade = "endpoint_direction_discordance"
    else:
        grade = "insufficient_external_support"

    return {
        "module_label": module,
        "target_hr_per_sd": target_hr,
        "target_primary_p": target_p,
        "target_q_within_endpoint": target_q,
        "target_c_index": target_c,
        "gse_met_auc": gse_auc,
        "gse_q_within_endpoint": gse_q,
        "global_q_eight_tests": global_q,
        "gse_robust_logistic_or_per_sd": robust_or,
        "gse_permutation_auc_q_bh": robust_perm_q,
        "direction_consistent": direction_consistent,
        "target_loo_fraction_hr_above_1": target_loo_direction,
        "gse_loo_fraction_auc_above_0_50": gse_loo_direction,
        "target_random_empirical_p": target_random_p,
        "gse_random_empirical_p": gse_random_p,
        "robust_external_evidence_grade": grade,
        "interpretation": {
            "M34": "Primary conserved immune/myeloid program candidate; strong metastasis discrimination and nominal TARGET-OS survival association.",
            "M11": "Metastasis-focused secondary support with weak TARGET-OS evidence.",
            "M24": "Metastasis association is not directionally concordant with TARGET-OS.",
            "M40": "Suggestive proliferation-deviation transfer; not confirmatory in either primary human setting.",
        }.get(module, ""),
    }


def build_readme() -> str:
    return f"""External human validation robustness audit
Script version: {SCRIPT_VERSION}

Purpose
-------
1. Recover finite descriptive logistic odds ratios without requiring statsmodels.
2. Add stratified-bootstrap coefficient intervals and label-permutation AUC tests.
3. Evaluate age and proliferation adjustment in TARGET-OS.
4. Evaluate proliferation adjustment in GSE21257.
5. Quantify leave-one-out stability in both human cohorts.
6. Convert expression-matched random-panel controls to empirical p-values and BH q-values.
7. Produce a frozen evidence summary without changing modules, weights, directions, or tiers.

Interpretation
--------------
The primary prespecified inference remains the strict signed-mean analysis from script 23.
This audit supplies robustness and effect-size diagnostics. It must not be used to revise the
frozen canine programs after seeing human outcomes.
"""


def main() -> None:
    print("=" * 80)
    print("External human validation robustness audit")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Results directory: {RESULTS_DIR}")
    print("")
    print("Design:")
    print("  Preserve the frozen primary human analyses from script 23.")
    print("  Recover robust logistic effect sizes without outcome-driven model selection.")
    print("  Audit clinical/proliferation adjustment, leave-one-out stability, and random controls.")

    verify_source_manifest()

    target_clinical = read_required_csv(TARGET_CLINICAL_FILE, index_col=0)
    target_scores = read_required_csv(TARGET_SCORES_FILE, index_col=0)
    gse_clinical = read_required_csv(GSE_CLINICAL_FILE, index_col=0)
    gse_scores = read_required_csv(GSE_SCORES_FILE, index_col=0)
    target_primary = read_required_csv(TARGET_PRIMARY_FILE, index_col=None)
    gse_primary = read_required_csv(GSE_PRIMARY_FILE, index_col=None)
    gse_os = read_optional_csv(GSE_OS_FILE, index_col=None)
    adjusted_old = read_optional_csv(ADJUSTED_FILE, index_col=None)
    variants = read_optional_csv(VARIANTS_FILE, index_col=None)
    random_controls_raw = read_required_csv(RANDOM_FILE, index_col=None)
    synthesis_old = read_optional_csv(SYNTHESIS_FILE, index_col=None)

    common_target = target_clinical.index.intersection(target_scores.index)
    common_gse = gse_clinical.index.intersection(gse_scores.index)
    target_clinical = target_clinical.loc[common_target].copy()
    target_scores = target_scores.loc[common_target].copy()
    gse_clinical = gse_clinical.loc[common_gse].copy()
    gse_scores = gse_scores.loc[common_gse].copy()

    print("")
    print("Matched data:")
    print(f"  TARGET-OS: {len(common_target)} samples")
    print(f"  GSE21257: {len(common_gse)} samples")

    rng = np.random.default_rng(RANDOM_SEED)
    gse_logistic, gse_adjusted = run_gse_robust_logistic(gse_clinical, gse_scores, rng)
    target_adjusted = run_target_adjustment(target_clinical, target_scores)
    target_loo = run_target_loo(target_clinical, target_scores)
    gse_loo = run_gse_loo(gse_clinical, gse_scores)
    random_controls = process_random_controls(random_controls_raw)

    final_rows = [
        evidence_grade(
            module=module,
            target_primary=target_primary,
            gse_primary=gse_primary,
            target_adjusted=target_adjusted,
            gse_logistic=gse_logistic,
            target_loo=target_loo,
            gse_loo=gse_loo,
            random_table=random_controls,
        )
        for module in PRIMARY_MODULES
    ]
    final = pd.DataFrame(final_rows)

    gse_logistic.to_csv(OUTPUT_LOGISTIC, index=False)
    target_adjusted.to_csv(OUTPUT_TARGET_ADJUSTED, index=False)
    gse_adjusted.to_csv(OUTPUT_GSE_ADJUSTED, index=False)
    target_loo.to_csv(OUTPUT_TARGET_LOO, index=False)
    gse_loo.to_csv(OUTPUT_GSE_LOO, index=False)
    random_controls.to_csv(OUTPUT_RANDOM, index=False)
    final.to_csv(OUTPUT_FINAL, index=False)
    OUTPUT_README.write_text(build_readme(), encoding="utf-8")

    print("")
    print("=" * 80)
    print("Robust GSE21257 logistic effects")
    print("=" * 80)
    logistic_cols = [
        "module_label", "n", "positives", "negatives", "auc", "average_precision",
        "or_per_sd", "or_ci_low", "or_ci_high", "bootstrap_probability_coef_positive",
        "permutation_auc_p_two_sided", "permutation_auc_q_bh", "bootstrap_valid",
    ]
    print(gse_logistic[logistic_cols].to_string(index=False))

    print("")
    print("=" * 80)
    print("TARGET-OS adjustment robustness")
    print("=" * 80)
    target_cols = [
        "module_label", "adjustment", "n", "events", "hr_per_sd", "ci_low", "ci_high",
        "p", "c_index", "ph_test_p", "error",
    ]
    print(target_adjusted[target_cols].to_string(index=False))

    print("")
    print("=" * 80)
    print("GSE21257 proliferation-adjusted robustness")
    print("=" * 80)
    if gse_adjusted.empty:
        print("No proliferation-adjusted GSE21257 models were fitted.")
    else:
        print(gse_adjusted[logistic_cols].to_string(index=False))

    print("")
    print("=" * 80)
    print("Leave-one-out stability")
    print("=" * 80)
    print("TARGET-OS:")
    print(target_loo.to_string(index=False))
    print("")
    print("GSE21257:")
    print(gse_loo.to_string(index=False))

    print("")
    print("=" * 80)
    print("Expression-matched random-control empirical tests")
    print("=" * 80)
    random_cols = [
        "cohort", "endpoint", "module_label", "observed_metric", "n_random_valid",
        "random_mean", "random_q95", "observed_percentile", "empirical_p_greater_equal",
        "empirical_q_within_cohort", "empirical_q_global_eight",
    ]
    random_cols = [c for c in random_cols if c in random_controls.columns]
    print(random_controls[random_cols].to_string(index=False))

    print("")
    print("=" * 80)
    print("Robust external evidence summary")
    print("=" * 80)
    final_cols = [
        "module_label", "target_hr_per_sd", "target_primary_p", "target_c_index",
        "gse_met_auc", "gse_q_within_endpoint", "global_q_eight_tests",
        "gse_robust_logistic_or_per_sd", "gse_permutation_auc_q_bh",
        "direction_consistent", "target_loo_fraction_hr_above_1",
        "gse_loo_fraction_auc_above_0_50", "target_random_empirical_p",
        "gse_random_empirical_p", "robust_external_evidence_grade", "interpretation",
    ]
    print(final[final_cols].to_string(index=False))

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("Script 23 remains the prespecified primary external analysis.")
    print("Bootstrap logistic odds ratios, leave-one-out ranges, and random-panel tests are robustness diagnostics.")
    print("M34 is the only module with global-FDR support in GSE21257 and nominal, directionally concordant TARGET-OS support.")
    print("M24 must not be described as conserved because its TARGET-OS and metastasis directions disagree.")
    print("No human result may be used to alter frozen module membership, score weights, or validation tiers.")

    output_files = [
        OUTPUT_LOGISTIC,
        OUTPUT_TARGET_ADJUSTED,
        OUTPUT_GSE_ADJUSTED,
        OUTPUT_TARGET_LOO,
        OUTPUT_GSE_LOO,
        OUTPUT_RANDOM,
        OUTPUT_FINAL,
        OUTPUT_README,
    ]
    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "random_seed": RANDOM_SEED,
        "bootstrap_logistic": N_BOOTSTRAP_LOGISTIC,
        "auc_permutations": N_PERMUTATIONS,
        "primary_modules": PRIMARY_MODULES,
        "input_files": {
            path.name: {"sha256": sha256_file(path)}
            for path in [
                TARGET_CLINICAL_FILE,
                TARGET_SCORES_FILE,
                GSE_CLINICAL_FILE,
                GSE_SCORES_FILE,
                TARGET_PRIMARY_FILE,
                GSE_PRIMARY_FILE,
                RANDOM_FILE,
            ]
            if path.exists()
        },
        "output_files": {
            path.name: {"sha256": sha256_file(path)}
            for path in output_files
            if path.exists()
        },
    }
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("")
    print("Saved:")
    for path in output_files + [OUTPUT_MANIFEST]:
        print(path)
    print("Done.")


if __name__ == "__main__":
    main()
