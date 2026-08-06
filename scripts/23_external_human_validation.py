from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

warnings.filterwarnings("ignore")

SCRIPT_VERSION = "23-human-external-validation-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
HUMAN_DIR = PROCESSED_DIR / "human_validation"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Frozen canine assets.
FREEZE_JSON_FILE = RESULTS_DIR / "GSE238110_frozen_transfer_program_freeze.json"
FROZEN_MANIFEST_FILE = RESULTS_DIR / "GSE238110_frozen_canine_transfer_program_manifest.csv"
STRICT_WEIGHTS_FILE = RESULTS_DIR / "GSE238110_frozen_transfer_gene_weights_strict.csv"
BROAD_WEIGHTS_FILE = RESULTS_DIR / "GSE238110_frozen_transfer_gene_weights_broad.csv"
ORTHOLOG_QC_FILE = RESULTS_DIR / "GSE238110_RNA_master_candidate_evidence_table_with_ortholog_qc.csv"

# Human cohort preparation assets from script 22.
PREPARATION_MANIFEST_FILE = RESULTS_DIR / "human_validation_cohort_preparation_manifest.json"
TARGET_EXPRESSION_FILE = HUMAN_DIR / "TARGET_OS_expression_log2_gene_symbol.csv"
TARGET_CLINICAL_FILE = HUMAN_DIR / "TARGET_OS_clinical_standardized.csv"
TARGET_SCORES_FILE = HUMAN_DIR / "TARGET_OS_frozen_transfer_scores.csv"
TARGET_COVERAGE_FILE = RESULTS_DIR / "TARGET_OS_frozen_transfer_score_coverage.csv"
GSE_EXPRESSION_FILE = HUMAN_DIR / "GSE21257_expression_gene_symbol.csv"
GSE_CLINICAL_FILE = HUMAN_DIR / "GSE21257_clinical_standardized.csv"
GSE_SCORES_FILE = HUMAN_DIR / "GSE21257_frozen_transfer_scores.csv"
GSE_COVERAGE_FILE = RESULTS_DIR / "GSE21257_frozen_transfer_score_coverage.csv"

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]
SECONDARY_MODULES = ["M28", "M38", "M25", "M17"]
EXPLORATORY_MODULES = ["M20", "M23", "M27", "M30", "M39"]

PRIMARY_SCORE_SUFFIX = "__strict__signed_mean_z"
STRICT_WEIGHTED_SUFFIX = "__strict__canine_pca_weighted_z"
STRICT_PC1_SUFFIX = "__strict__human_pc1_z"
BROAD_SIGNED_SUFFIX = "__broad__signed_mean_z"
BROAD_WEIGHTED_SUFFIX = "__broad__canine_pca_weighted_z"
M40_RESIDUAL_SIGNED = "M40__strict__signed_mean__residual_to_disjoint_proliferation_z"
M40_RESIDUAL_WEIGHTED = "M40__strict__canine_pca_weighted__residual_to_disjoint_proliferation_z"
PROLIFERATION_SCORE = "strict_human_meta_proliferation_pc1_z"
M40_DISJOINT_PROLIFERATION_SCORE = "M40_disjoint_strict_human_meta_proliferation_pc1_z"

COX_PENALIZER = 0.01
N_BOOTSTRAP = 3000
N_RANDOM_GENE_SETS = 1000
RANDOM_SEED = 20260805
MIN_ANALYSIS_N = 30
MIN_SURVIVAL_EVENTS = 8
MIN_CLASS_POS = 8
MIN_CLASS_NEG = 8
PRIMARY_Q_THRESHOLD = 0.05
SUGGESTIVE_Q_THRESHOLD = 0.10

# Output files.
OUTPUT_TARGET_PRIMARY = RESULTS_DIR / "TARGET_OS_primary_frozen_program_validation.csv"
OUTPUT_GSE_MET_PRIMARY = RESULTS_DIR / "GSE21257_metastasis_primary_frozen_program_validation.csv"
OUTPUT_GSE_OS_SENSITIVITY = RESULTS_DIR / "GSE21257_OS_frozen_program_sensitivity.csv"
OUTPUT_SECONDARY = RESULTS_DIR / "human_external_validation_secondary_programs.csv"
OUTPUT_SCORE_VARIANTS = RESULTS_DIR / "human_external_validation_score_variant_sensitivity.csv"
OUTPUT_ADJUSTED = RESULTS_DIR / "human_external_validation_adjusted_models.csv"
OUTPUT_MULTIPLICITY = RESULTS_DIR / "human_external_validation_primary_multiplicity.csv"
OUTPUT_RANDOM_CONTROLS = RESULTS_DIR / "human_external_validation_random_gene_set_controls.csv"
OUTPUT_RANDOM_CONTROL_DISTRIBUTION = RESULTS_DIR / "human_external_validation_random_gene_set_distribution.csv"
OUTPUT_SYNTHESIS = RESULTS_DIR / "human_external_validation_cross_cohort_synthesis.csv"
OUTPUT_MANIFEST = RESULTS_DIR / "human_external_validation_manifest.json"
OUTPUT_README = RESULTS_DIR / "human_external_validation_README.txt"


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


def verify_preparation_assets() -> dict[str, Any]:
    if not PREPARATION_MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Preparation manifest not found: {PREPARATION_MANIFEST_FILE}. Run script 22 first."
        )
    manifest = json.loads(PREPARATION_MANIFEST_FILE.read_text(encoding="utf-8"))
    file_records = manifest.get("files", {})
    required = [
        TARGET_EXPRESSION_FILE,
        TARGET_CLINICAL_FILE,
        TARGET_SCORES_FILE,
        GSE_EXPRESSION_FILE,
        GSE_CLINICAL_FILE,
        GSE_SCORES_FILE,
    ]
    print("")
    print("Human preparation input integrity check:")
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Prepared human input is missing: {path}")
        expected = file_records.get(path.name, {}).get("sha256")
        observed = sha256_file(path)
        if expected and expected != observed:
            raise RuntimeError(
                f"Hash mismatch for {path.name}. Do not continue after modifying prepared data."
            )
        print(f"  {path.name}: {'verified' if expected else 'present_without_recorded_hash'}")
    return manifest


def bh_adjust(pvalues: pd.Series | list[float] | np.ndarray) -> pd.Series:
    p = pd.Series(pvalues, dtype=float)
    q = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.notna() & np.isfinite(p)
    if valid.sum() == 0:
        return q
    values = p.loc[valid].to_numpy()
    order = np.argsort(values)
    ranked = values[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    back = np.empty(n)
    back[order] = adjusted
    q.loc[p.index[valid]] = back
    return q


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def standardize_series(series: pd.Series) -> pd.Series:
    values = safe_numeric(series)
    sd = values.std()
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=series.index)
    return (values - values.mean()) / sd


def clean_scores(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    out = out.drop(columns=["cohort"], errors="ignore")
    for col in out.columns:
        out[col] = safe_numeric(out[col])
    return out


def match_cohort(clinical: pd.DataFrame, scores: pd.DataFrame, expression: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common = clinical.index.intersection(scores.index).intersection(expression.index)
    if common.empty:
        raise RuntimeError("No common samples among clinical, score, and expression tables.")
    return clinical.loc[common].copy(), scores.loc[common].copy(), expression.loc[common].copy()


def bootstrap_c_index(
    time_values: np.ndarray,
    event_values: np.ndarray,
    risk_values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    n = len(time_values)
    estimates = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        t = time_values[idx]
        e = event_values[idx]
        r = risk_values[idx]
        if np.sum(e == 1) < 2 or np.sum(e == 0) < 2:
            continue
        try:
            estimates.append(float(concordance_index(t, -r, e)))
        except Exception:
            continue
    if not estimates:
        return np.nan, np.nan, 0
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
        len(estimates),
    )


def fit_cox_score(
    clinical: pd.DataFrame,
    score: pd.Series,
    time_col: str,
    event_col: str,
    cohort: str,
    endpoint: str,
    module_label: str,
    score_name: str,
    analysis_tier: str,
    covariates: pd.DataFrame | None = None,
    proliferation_score: pd.Series | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    frame = pd.DataFrame({
        "time": safe_numeric(clinical[time_col]),
        "event": safe_numeric(clinical[event_col]),
        "score": standardize_series(score.reindex(clinical.index)),
    })
    model_covariates = []
    if proliferation_score is not None:
        frame["proliferation"] = standardize_series(proliferation_score.reindex(clinical.index))
        model_covariates.append("proliferation")
    if covariates is not None and not covariates.empty:
        for col in covariates.columns:
            frame[col] = safe_numeric(covariates[col].reindex(clinical.index))
            model_covariates.append(col)

    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    row: dict[str, Any] = {
        "cohort": cohort,
        "endpoint": endpoint,
        "module_label": module_label,
        "score_name": score_name,
        "analysis_tier": analysis_tier,
        "n": int(frame.shape[0]),
        "events": int(frame["event"].sum()) if not frame.empty else 0,
        "n_covariates": len(model_covariates),
        "covariates": ";".join(model_covariates),
        "score_coef": np.nan,
        "score_hr_per_sd": np.nan,
        "score_ci_low": np.nan,
        "score_ci_high": np.nan,
        "score_p": np.nan,
        "model_c_index": np.nan,
        "fixed_score_c_index": np.nan,
        "fixed_score_c_index_ci_low": np.nan,
        "fixed_score_c_index_ci_high": np.nan,
        "bootstrap_valid": 0,
        "ph_test_p": np.nan,
        "error": "",
    }
    if frame.shape[0] < MIN_ANALYSIS_N:
        row["error"] = "too_few_complete_samples"
        return row
    if frame["event"].sum() < MIN_SURVIVAL_EVENTS:
        row["error"] = "too_few_events"
        return row
    if frame["score"].std() == 0:
        row["error"] = "zero_variance_score"
        return row

    try:
        fixed_c = float(concordance_index(frame["time"], -frame["score"], frame["event"]))
        row["fixed_score_c_index"] = fixed_c
        if rng is not None:
            low, high, n_valid = bootstrap_c_index(
                frame["time"].to_numpy(float),
                frame["event"].to_numpy(int),
                frame["score"].to_numpy(float),
                N_BOOTSTRAP,
                rng,
            )
            row["fixed_score_c_index_ci_low"] = low
            row["fixed_score_c_index_ci_high"] = high
            row["bootstrap_valid"] = n_valid
    except Exception:
        pass

    cph = CoxPHFitter(penalizer=COX_PENALIZER)
    try:
        fit_cols = ["time", "event", "score"] + model_covariates
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            cph.fit(
                frame[fit_cols],
                duration_col="time",
                event_col="event",
                fit_options={"max_steps": 500},
            )
        s = cph.summary.loc["score"]
        row["score_coef"] = float(s["coef"])
        row["score_hr_per_sd"] = float(s["exp(coef)"])
        row["score_ci_low"] = float(s["exp(coef) lower 95%"])
        row["score_ci_high"] = float(s["exp(coef) upper 95%"])
        row["score_p"] = float(s["p"])
        row["model_c_index"] = float(cph.concordance_index_)
        try:
            ph = proportional_hazard_test(cph, frame[fit_cols], time_transform="rank")
            row["ph_test_p"] = float(ph.summary.loc["score", "p"])
        except Exception:
            pass
    except Exception as exc:
        row["error"] = str(exc)[:500]
    return row


def stratified_bootstrap_auc(
    y: np.ndarray,
    score: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    estimates = []
    for _ in range(n_bootstrap):
        idx_pos = rng.choice(pos, size=len(pos), replace=True)
        idx_neg = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([idx_pos, idx_neg])
        try:
            estimates.append(float(roc_auc_score(y[idx], score[idx])))
        except Exception:
            continue
    if not estimates:
        return np.nan, np.nan, 0
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
        len(estimates),
    )


def fit_logistic_score(
    clinical: pd.DataFrame,
    score: pd.Series,
    outcome_col: str,
    cohort: str,
    endpoint: str,
    module_label: str,
    score_name: str,
    analysis_tier: str,
    covariates: pd.DataFrame | None = None,
    proliferation_score: pd.Series | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    frame = pd.DataFrame({
        "outcome": safe_numeric(clinical[outcome_col]),
        "score": standardize_series(score.reindex(clinical.index)),
    })
    model_covariates = []
    if proliferation_score is not None:
        frame["proliferation"] = standardize_series(proliferation_score.reindex(clinical.index))
        model_covariates.append("proliferation")
    if covariates is not None and not covariates.empty:
        for col in covariates.columns:
            frame[col] = safe_numeric(covariates[col].reindex(clinical.index))
            model_covariates.append(col)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame[frame["outcome"].isin([0, 1])].copy()

    row: dict[str, Any] = {
        "cohort": cohort,
        "endpoint": endpoint,
        "module_label": module_label,
        "score_name": score_name,
        "analysis_tier": analysis_tier,
        "n": int(frame.shape[0]),
        "positives": int((frame["outcome"] == 1).sum()),
        "negatives": int((frame["outcome"] == 0).sum()),
        "n_covariates": len(model_covariates),
        "covariates": ";".join(model_covariates),
        "auc": np.nan,
        "auc_ci_low": np.nan,
        "auc_ci_high": np.nan,
        "average_precision": np.nan,
        "mann_whitney_p_two_sided": np.nan,
        "logistic_coef": np.nan,
        "logistic_or_per_sd": np.nan,
        "logistic_ci_low": np.nan,
        "logistic_ci_high": np.nan,
        "logistic_p": np.nan,
        "bootstrap_valid": 0,
        "error": "",
    }
    if frame.shape[0] < MIN_ANALYSIS_N:
        row["error"] = "too_few_complete_samples"
        return row
    if row["positives"] < MIN_CLASS_POS or row["negatives"] < MIN_CLASS_NEG:
        row["error"] = "insufficient_class_counts"
        return row
    if frame["score"].std() == 0:
        row["error"] = "zero_variance_score"
        return row

    y = frame["outcome"].to_numpy(int)
    s = frame["score"].to_numpy(float)
    try:
        row["auc"] = float(roc_auc_score(y, s))
        row["average_precision"] = float(average_precision_score(y, s))
        if rng is not None:
            low, high, n_valid = stratified_bootstrap_auc(y, s, N_BOOTSTRAP, rng)
            row["auc_ci_low"] = low
            row["auc_ci_high"] = high
            row["bootstrap_valid"] = n_valid
        pos_scores = frame.loc[frame["outcome"] == 1, "score"]
        neg_scores = frame.loc[frame["outcome"] == 0, "score"]
        row["mann_whitney_p_two_sided"] = float(
            stats.mannwhitneyu(pos_scores, neg_scores, alternative="two-sided").pvalue
        )
    except Exception as exc:
        row["error"] = str(exc)[:500]
        return row

    if HAS_STATSMODELS:
        try:
            x_cols = ["score"] + model_covariates
            x = sm.add_constant(frame[x_cols], has_constant="add")
            model = sm.Logit(frame["outcome"], x).fit(disp=False, maxiter=500)
            coef = float(model.params["score"])
            se = float(model.bse["score"])
            row["logistic_coef"] = coef
            row["logistic_or_per_sd"] = float(np.exp(coef))
            row["logistic_ci_low"] = float(np.exp(coef - 1.96 * se))
            row["logistic_ci_high"] = float(np.exp(coef + 1.96 * se))
            row["logistic_p"] = float(model.pvalues["score"])
        except Exception as exc:
            if not row["error"]:
                row["error"] = f"logistic_fit_failed:{str(exc)[:400]}"
    else:
        row["error"] = "statsmodels_not_installed_logistic_effect_not_fitted"
    return row


def parse_metastasis_at_diagnosis(series: pd.Series) -> pd.Series:
    def parse_one(value: Any) -> float:
        text = str(value).strip().lower()
        if not text or text in {"nan", "none", "unknown", "not reported"}:
            return np.nan
        positive = ["yes", "m1", "metastatic", "present", "true", "positive"]
        negative = ["no", "m0", "absent", "false", "negative"]
        if any(token in text for token in positive):
            return 1.0
        if any(token in text for token in negative):
            return 0.0
        return np.nan
    return series.map(parse_one)


def build_target_adjustment_covariates(clinical: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=clinical.index)
    if "age_at_diagnosis_years" in clinical.columns:
        age = safe_numeric(clinical["age_at_diagnosis_years"])
        if age.notna().sum() >= 50 and age.nunique(dropna=True) >= 5:
            out["age_z"] = standardize_series(age)
    if "sex" in clinical.columns:
        sex = clinical["sex"].astype(str).str.strip().str.lower()
        if sex.isin(["male", "female"]).sum() >= 50 and sex[sex.isin(["male", "female"])].nunique() == 2:
            out["sex_male"] = sex.map({"female": 0.0, "male": 1.0})
    if "metastasis_fields_raw" in clinical.columns:
        metastatic = parse_metastasis_at_diagnosis(clinical["metastasis_fields_raw"])
        if metastatic.notna().sum() >= 40 and metastatic.nunique(dropna=True) == 2:
            out["metastatic_at_diagnosis"] = metastatic
    return out


def build_gse_adjustment_covariates(clinical: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=clinical.index)
    if "age_years" in clinical.columns:
        age = safe_numeric(clinical["age_years"])
        if age.notna().sum() >= 40 and age.nunique(dropna=True) >= 5:
            out["age_z"] = standardize_series(age)
    return out


def validation_tier_map(manifest: pd.DataFrame) -> dict[str, str]:
    if "module_label" not in manifest.columns or "validation_tier" not in manifest.columns:
        return {}
    return manifest.set_index("module_label")["validation_tier"].astype(str).to_dict()


def score_column(module: str, suffix: str) -> str:
    return f"{module}{suffix}"


def run_primary_survival(
    clinical: pd.DataFrame,
    scores: pd.DataFrame,
    cohort: str,
    time_col: str,
    event_col: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for module in PRIMARY_MODULES:
        col = score_column(module, PRIMARY_SCORE_SUFFIX)
        if col not in scores.columns:
            rows.append({
                "cohort": cohort,
                "endpoint": "overall_survival",
                "module_label": module,
                "score_name": col,
                "analysis_tier": "primary_confirmatory",
                "error": "primary_score_missing",
            })
            continue
        rows.append(
            fit_cox_score(
                clinical=clinical,
                score=scores[col],
                time_col=time_col,
                event_col=event_col,
                cohort=cohort,
                endpoint="overall_survival",
                module_label=module,
                score_name=col,
                analysis_tier="primary_confirmatory",
                rng=rng,
            )
        )
    return pd.DataFrame(rows)


def run_primary_metastasis(
    clinical: pd.DataFrame,
    scores: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for module in PRIMARY_MODULES:
        col = score_column(module, PRIMARY_SCORE_SUFFIX)
        if col not in scores.columns:
            rows.append({
                "cohort": "GSE21257",
                "endpoint": "metastasis_within_5y",
                "module_label": module,
                "score_name": col,
                "analysis_tier": "primary_confirmatory",
                "error": "primary_score_missing",
            })
            continue
        rows.append(
            fit_logistic_score(
                clinical=clinical,
                score=scores[col],
                outcome_col="metastasis_within_5y",
                cohort="GSE21257",
                endpoint="metastasis_within_5y",
                module_label=module,
                score_name=col,
                analysis_tier="primary_confirmatory",
                rng=rng,
            )
        )
    return pd.DataFrame(rows)


def add_primary_multiplicity(target: pd.DataFrame, gse: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = target.copy()
    gse = gse.copy()
    target["primary_p"] = safe_numeric(target.get("score_p", pd.Series(index=target.index, dtype=float)))
    gse_p_logit = safe_numeric(gse.get("logistic_p", pd.Series(index=gse.index, dtype=float)))
    gse_p_mw = safe_numeric(gse.get("mann_whitney_p_two_sided", pd.Series(index=gse.index, dtype=float)))
    gse["primary_p"] = gse_p_logit.where(gse_p_logit.notna(), gse_p_mw)

    target["q_within_endpoint"] = bh_adjust(target["primary_p"])
    gse["q_within_endpoint"] = bh_adjust(gse["primary_p"])

    combined = pd.concat([
        target[["cohort", "endpoint", "module_label", "score_name", "primary_p"]],
        gse[["cohort", "endpoint", "module_label", "score_name", "primary_p"]],
    ], ignore_index=True)
    combined["q_global_eight_tests"] = bh_adjust(combined["primary_p"])
    combined["direction_consistent"] = False

    target_direction = safe_numeric(target.get("score_hr_per_sd", pd.Series(index=target.index, dtype=float))) > 1
    gse_direction = safe_numeric(gse.get("auc", pd.Series(index=gse.index, dtype=float))) > 0.5
    target["direction_consistent"] = target_direction
    gse["direction_consistent"] = gse_direction

    global_map = combined.set_index(["cohort", "endpoint", "module_label"])["q_global_eight_tests"]
    target["q_global_eight_tests"] = [
        global_map.get((row.cohort, row.endpoint, row.module_label), np.nan)
        for row in target.itertuples()
    ]
    gse["q_global_eight_tests"] = [
        global_map.get((row.cohort, row.endpoint, row.module_label), np.nan)
        for row in gse.itertuples()
    ]

    for table in [target, gse]:
        table["external_support_class"] = np.select(
            [
                table["direction_consistent"] & (table["q_global_eight_tests"] < PRIMARY_Q_THRESHOLD),
                table["direction_consistent"] & (table["q_within_endpoint"] < PRIMARY_Q_THRESHOLD),
                table["direction_consistent"] & (table["q_within_endpoint"] < SUGGESTIVE_Q_THRESHOLD),
                table["direction_consistent"] & (table["primary_p"] < 0.05),
            ],
            [
                "global_fdr_confirmatory_support",
                "within_endpoint_fdr_support",
                "suggestive_fdr_support",
                "nominal_directional_support",
            ],
            default="no_confirmatory_support",
        )
    return target, gse, combined


def run_secondary_programs(
    target_clinical: pd.DataFrame,
    target_scores: pd.DataFrame,
    gse_clinical: pd.DataFrame,
    gse_scores: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for module in SECONDARY_MODULES:
        col = score_column(module, PRIMARY_SCORE_SUFFIX)
        if col in target_scores.columns:
            rows.append(fit_cox_score(
                target_clinical, target_scores[col], "os_time_days", "os_event",
                "TARGET_OS", "overall_survival", module, col,
                "secondary_prespecified", rng=rng,
            ))
        if col in gse_scores.columns:
            rows.append(fit_logistic_score(
                gse_clinical, gse_scores[col], "metastasis_within_5y",
                "GSE21257", "metastasis_within_5y", module, col,
                "secondary_prespecified", rng=rng,
            ))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_for_fdr"] = np.where(
        out["endpoint"].eq("overall_survival"),
        safe_numeric(out.get("score_p", pd.Series(index=out.index, dtype=float))),
        safe_numeric(out.get("logistic_p", pd.Series(index=out.index, dtype=float))).where(
            safe_numeric(out.get("logistic_p", pd.Series(index=out.index, dtype=float))).notna(),
            safe_numeric(out.get("mann_whitney_p_two_sided", pd.Series(index=out.index, dtype=float))),
        ),
    )
    out["q_within_secondary_endpoint"] = np.nan
    for endpoint in out["endpoint"].dropna().unique():
        mask = out["endpoint"].eq(endpoint)
        out.loc[mask, "q_within_secondary_endpoint"] = bh_adjust(out.loc[mask, "p_for_fdr"]).values
    return out


def run_score_variant_sensitivity(
    target_clinical: pd.DataFrame,
    target_scores: pd.DataFrame,
    gse_clinical: pd.DataFrame,
    gse_scores: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    variants = [
        (STRICT_WEIGHTED_SUFFIX, "strict_canine_pca_weighted"),
        (STRICT_PC1_SUFFIX, "strict_human_pc1"),
        (BROAD_SIGNED_SUFFIX, "broad_signed_mean"),
        (BROAD_WEIGHTED_SUFFIX, "broad_canine_pca_weighted"),
    ]
    for module in PRIMARY_MODULES:
        for suffix, variant_label in variants:
            col = score_column(module, suffix)
            if col in target_scores.columns:
                row = fit_cox_score(
                    target_clinical, target_scores[col], "os_time_days", "os_event",
                    "TARGET_OS", "overall_survival", module, col,
                    "score_variant_sensitivity", rng=rng,
                )
                row["score_variant"] = variant_label
                rows.append(row)
            if col in gse_scores.columns:
                row = fit_logistic_score(
                    gse_clinical, gse_scores[col], "metastasis_within_5y",
                    "GSE21257", "metastasis_within_5y", module, col,
                    "score_variant_sensitivity", rng=rng,
                )
                row["score_variant"] = variant_label
                rows.append(row)

    for col, variant_label in [
        (M40_RESIDUAL_SIGNED, "m40_residual_signed"),
        (M40_RESIDUAL_WEIGHTED, "m40_residual_weighted"),
    ]:
        if col in target_scores.columns:
            row = fit_cox_score(
                target_clinical, target_scores[col], "os_time_days", "os_event",
                "TARGET_OS", "overall_survival", "M40", col,
                "mechanistic_sensitivity", rng=rng,
            )
            row["score_variant"] = variant_label
            rows.append(row)
        if col in gse_scores.columns:
            row = fit_logistic_score(
                gse_clinical, gse_scores[col], "metastasis_within_5y",
                "GSE21257", "metastasis_within_5y", "M40", col,
                "mechanistic_sensitivity", rng=rng,
            )
            row["score_variant"] = variant_label
            rows.append(row)
    return pd.DataFrame(rows)


def run_adjusted_models(
    target_clinical: pd.DataFrame,
    target_scores: pd.DataFrame,
    gse_clinical: pd.DataFrame,
    gse_scores: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    target_covariates = build_target_adjustment_covariates(target_clinical)
    gse_covariates = build_gse_adjustment_covariates(gse_clinical)
    print("")
    print("Adjustment covariates detected:")
    print(f"  TARGET-OS: {list(target_covariates.columns)}")
    print(f"  GSE21257: {list(gse_covariates.columns)}")

    for module in PRIMARY_MODULES:
        col = score_column(module, PRIMARY_SCORE_SUFFIX)
        if col in target_scores.columns:
            row = fit_cox_score(
                target_clinical, target_scores[col], "os_time_days", "os_event",
                "TARGET_OS", "overall_survival", module, col,
                "clinical_adjustment_sensitivity", covariates=target_covariates,
            )
            row["adjustment_type"] = "available_clinical_covariates"
            rows.append(row)
            if PROLIFERATION_SCORE in target_scores.columns and module != "M40":
                row = fit_cox_score(
                    target_clinical, target_scores[col], "os_time_days", "os_event",
                    "TARGET_OS", "overall_survival", module, col,
                    "proliferation_adjustment_sensitivity",
                    proliferation_score=target_scores[PROLIFERATION_SCORE],
                )
                row["adjustment_type"] = "human_meta_proliferation_pc1"
                rows.append(row)
        if col in gse_scores.columns:
            row = fit_logistic_score(
                gse_clinical, gse_scores[col], "metastasis_within_5y",
                "GSE21257", "metastasis_within_5y", module, col,
                "clinical_adjustment_sensitivity", covariates=gse_covariates,
            )
            row["adjustment_type"] = "available_clinical_covariates"
            rows.append(row)
            if PROLIFERATION_SCORE in gse_scores.columns and module != "M40":
                row = fit_logistic_score(
                    gse_clinical, gse_scores[col], "metastasis_within_5y",
                    "GSE21257", "metastasis_within_5y", module, col,
                    "proliferation_adjustment_sensitivity",
                    proliferation_score=gse_scores[PROLIFERATION_SCORE],
                )
                row["adjustment_type"] = "human_meta_proliferation_pc1"
                rows.append(row)
    return pd.DataFrame(rows)


def run_gse_os_sensitivity(
    clinical: pd.DataFrame,
    scores: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for module in PRIMARY_MODULES + SECONDARY_MODULES:
        col = score_column(module, PRIMARY_SCORE_SUFFIX)
        if col not in scores.columns:
            continue
        tier = "primary_module_os_sensitivity" if module in PRIMARY_MODULES else "secondary_module_os_sensitivity"
        rows.append(fit_cox_score(
            clinical, scores[col], "os_time_months", "os_event",
            "GSE21257", "overall_survival_sensitivity", module, col, tier, rng=rng,
        ))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_within_gse_os_sensitivity"] = bh_adjust(safe_numeric(out["score_p"]))
    return out


def strict_ortholog_universe(ortholog_qc: pd.DataFrame) -> list[str]:
    human_col = "human_gene_symbol" if "human_gene_symbol" in ortholog_qc.columns else "human_symbol"
    status_col = "ortholog_qc_status"
    if human_col not in ortholog_qc.columns or status_col not in ortholog_qc.columns:
        raise ValueError("Ortholog QC table lacks human symbol or QC status.")
    genes = (
        ortholog_qc.loc[
            ortholog_qc[status_col].eq("strict_symbol_concordant_one_to_one"),
            human_col,
        ]
        .dropna()
        .astype(str)
        .str.upper()
        .drop_duplicates()
        .tolist()
    )
    return genes


def zscore_expression(expression: pd.DataFrame) -> pd.DataFrame:
    x = expression.copy()
    x.columns = x.columns.astype(str).str.upper()
    x = x.loc[:, ~x.columns.duplicated()].copy()
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(axis=0))
    sd = x.std(axis=0).replace(0, np.nan)
    z = (x - x.mean(axis=0)) / sd
    return z.loc[:, z.notna().all(axis=0)]


def assign_expression_bins(expression: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    x = expression.apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(axis=0))
    mean_expression = x.mean(axis=0)
    variance = x.var(axis=0)
    info = pd.DataFrame({"mean_expression": mean_expression, "variance": variance})
    try:
        info["mean_bin"] = pd.qcut(
            info["mean_expression"].rank(method="first"),
            n_bins,
            labels=False,
            duplicates="drop",
        )
    except Exception:
        info["mean_bin"] = 0
    try:
        info["var_bin"] = pd.qcut(
            info["variance"].rank(method="first"),
            n_bins,
            labels=False,
            duplicates="drop",
        )
    except Exception:
        info["var_bin"] = 0
    info["bin_key"] = info["mean_bin"].astype(str) + "_" + info["var_bin"].astype(str)
    return info


def sample_matched_random_genes(
    module_genes: list[str],
    available_universe: list[str],
    bin_info: pd.DataFrame,
    rng: np.random.Generator,
) -> list[str]:
    universe_set = set(available_universe)
    selected: list[str] = []
    used = set(module_genes)
    for gene in module_genes:
        if gene in bin_info.index:
            key = bin_info.loc[gene, "bin_key"]
            candidates = [
                g for g in bin_info.index[bin_info["bin_key"].eq(key)].tolist()
                if g in universe_set and g not in used
            ]
        else:
            candidates = []
        if not candidates:
            candidates = [g for g in available_universe if g not in used]
        if not candidates:
            break
        chosen = str(rng.choice(candidates))
        selected.append(chosen)
        used.add(chosen)
    return selected


def random_control_for_module(
    cohort: str,
    endpoint: str,
    module: str,
    expression: pd.DataFrame,
    clinical: pd.DataFrame,
    strict_weights: pd.DataFrame,
    strict_universe: list[str],
    observed_metric: float,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = expression.copy()
    raw.columns = raw.columns.astype(str).str.upper()
    raw = raw.loc[:, ~raw.columns.duplicated()].copy()
    raw = raw.apply(pd.to_numeric, errors="coerce")
    raw = raw.replace([np.inf, -np.inf], np.nan)
    raw = raw.fillna(raw.median(axis=0))
    z = zscore_expression(raw)
    universe = [gene for gene in strict_universe if gene in z.columns]
    part = strict_weights[strict_weights["module_label"].eq(module)].copy()
    part["human_gene_symbol"] = part["human_gene_symbol"].astype(str).str.upper()
    part = part.drop_duplicates("human_gene_symbol", keep="first")
    part = part[part["human_gene_symbol"].isin(z.columns)].copy()
    module_genes = part["human_gene_symbol"].tolist()
    signs = np.sign(safe_numeric(part["risk_oriented_loading"])).replace(0, 1).to_numpy(float)
    bin_info = assign_expression_bins(raw[universe])
    rows = []
    metrics = []

    for repeat in range(1, N_RANDOM_GENE_SETS + 1):
        genes = sample_matched_random_genes(module_genes, universe, bin_info, rng)
        if len(genes) != len(module_genes) or len(genes) < 3:
            continue
        signs_perm = signs.copy()
        rng.shuffle(signs_perm)
        random_score = z[genes].mul(signs_perm, axis=1).mean(axis=1)
        random_score = standardize_series(random_score)
        metric = np.nan
        if endpoint == "overall_survival":
            frame = pd.DataFrame({
                "time": safe_numeric(clinical["os_time_days"]),
                "event": safe_numeric(clinical["os_event"]),
                "score": random_score.reindex(clinical.index),
            }).dropna()
            if frame.shape[0] >= MIN_ANALYSIS_N and frame["event"].sum() >= MIN_SURVIVAL_EVENTS:
                metric = float(concordance_index(frame["time"], -frame["score"], frame["event"]))
        elif endpoint == "metastasis_within_5y":
            frame = pd.DataFrame({
                "outcome": safe_numeric(clinical["metastasis_within_5y"]),
                "score": random_score.reindex(clinical.index),
            }).dropna()
            frame = frame[frame["outcome"].isin([0, 1])]
            if frame["outcome"].nunique() == 2:
                metric = float(roc_auc_score(frame["outcome"], frame["score"]))
        if np.isfinite(metric):
            metrics.append(metric)
            rows.append({
                "cohort": cohort,
                "endpoint": endpoint,
                "module_label": module,
                "repeat": repeat,
                "panel_size": len(genes),
                "random_metric": metric,
            })

    summary = {
        "cohort": cohort,
        "endpoint": endpoint,
        "module_label": module,
        "observed_metric": observed_metric,
        "n_module_genes_available": len(module_genes),
        "n_random_valid": len(metrics),
        "random_mean": float(np.mean(metrics)) if metrics else np.nan,
        "random_median": float(np.median(metrics)) if metrics else np.nan,
        "random_q90": float(np.quantile(metrics, 0.90)) if metrics else np.nan,
        "random_q95": float(np.quantile(metrics, 0.95)) if metrics else np.nan,
        "observed_percentile": float(np.mean(np.asarray(metrics) <= observed_metric)) if metrics and np.isfinite(observed_metric) else np.nan,
        "empirical_p_greater_equal": float((1 + np.sum(np.asarray(metrics) >= observed_metric)) / (1 + len(metrics))) if metrics and np.isfinite(observed_metric) else np.nan,
    }
    return summary, rows


def run_random_controls(
    target_expression: pd.DataFrame,
    target_clinical: pd.DataFrame,
    target_primary: pd.DataFrame,
    gse_expression: pd.DataFrame,
    gse_clinical: pd.DataFrame,
    gse_primary: pd.DataFrame,
    strict_weights: pd.DataFrame,
    ortholog_qc: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strict_universe = strict_ortholog_universe(ortholog_qc)
    summaries = []
    distributions = []
    print("")
    print("Matched random-gene-set controls:")
    for module in PRIMARY_MODULES:
        target_row = target_primary[target_primary["module_label"].eq(module)]
        target_metric = float(target_row["fixed_score_c_index"].iloc[0]) if not target_row.empty else np.nan
        print(f"  TARGET-OS {module}: {N_RANDOM_GENE_SETS} random sets")
        summary, rows = random_control_for_module(
            "TARGET_OS", "overall_survival", module,
            target_expression, target_clinical, strict_weights, strict_universe,
            target_metric, rng,
        )
        summaries.append(summary)
        distributions.extend(rows)

        gse_row = gse_primary[gse_primary["module_label"].eq(module)]
        gse_metric = float(gse_row["auc"].iloc[0]) if not gse_row.empty else np.nan
        print(f"  GSE21257 {module}: {N_RANDOM_GENE_SETS} random sets")
        summary, rows = random_control_for_module(
            "GSE21257", "metastasis_within_5y", module,
            gse_expression, gse_clinical, strict_weights, strict_universe,
            gse_metric, rng,
        )
        summaries.append(summary)
        distributions.extend(rows)
    return pd.DataFrame(summaries), pd.DataFrame(distributions)


def build_cross_cohort_synthesis(
    target: pd.DataFrame,
    gse_met: pd.DataFrame,
    gse_os: pd.DataFrame,
    random_controls: pd.DataFrame,
) -> pd.DataFrame:
    target_keep = target[[
        "module_label", "score_hr_per_sd", "score_ci_low", "score_ci_high",
        "primary_p", "q_within_endpoint", "q_global_eight_tests",
        "fixed_score_c_index", "fixed_score_c_index_ci_low", "fixed_score_c_index_ci_high",
        "external_support_class",
    ]].copy()
    target_keep = target_keep.rename(columns={
        col: f"target_os_{col}" for col in target_keep.columns if col != "module_label"
    })
    gse_keep = gse_met[[
        "module_label", "auc", "auc_ci_low", "auc_ci_high", "average_precision",
        "logistic_or_per_sd", "logistic_ci_low", "logistic_ci_high",
        "primary_p", "q_within_endpoint", "q_global_eight_tests", "external_support_class",
    ]].copy()
    gse_keep = gse_keep.rename(columns={
        col: f"gse_met_{col}" for col in gse_keep.columns if col != "module_label"
    })
    synthesis = target_keep.merge(gse_keep, on="module_label", how="outer")

    if not gse_os.empty:
        gse_os_keep = gse_os[gse_os["module_label"].isin(PRIMARY_MODULES)][[
            "module_label", "score_hr_per_sd", "score_ci_low", "score_ci_high",
            "score_p", "q_within_gse_os_sensitivity", "fixed_score_c_index",
        ]].copy()
        gse_os_keep = gse_os_keep.rename(columns={
            col: f"gse_os_{col}" for col in gse_os_keep.columns if col != "module_label"
        })
        synthesis = synthesis.merge(gse_os_keep, on="module_label", how="left")

    if not random_controls.empty:
        for cohort, prefix in [("TARGET_OS", "target_random"), ("GSE21257", "gse_random")]:
            part = random_controls[random_controls["cohort"].eq(cohort)][[
                "module_label", "observed_percentile", "empirical_p_greater_equal"
            ]].copy()
            part = part.rename(columns={
                "observed_percentile": f"{prefix}_percentile",
                "empirical_p_greater_equal": f"{prefix}_empirical_p",
            })
            synthesis = synthesis.merge(part, on="module_label", how="left")

    synthesis["direction_consistent_target_and_metastasis"] = (
        safe_numeric(synthesis["target_os_score_hr_per_sd"]) > 1
    ) & (
        safe_numeric(synthesis["gse_met_auc"]) > 0.5
    )
    synthesis["cross_cohort_support_summary"] = np.select(
        [
            synthesis["direction_consistent_target_and_metastasis"]
            & (safe_numeric(synthesis["target_os_q_within_endpoint"]) < 0.05)
            & (safe_numeric(synthesis["gse_met_q_within_endpoint"]) < 0.05),
            synthesis["direction_consistent_target_and_metastasis"]
            & (
                (safe_numeric(synthesis["target_os_primary_p"]) < 0.05)
                | (safe_numeric(synthesis["gse_met_primary_p"]) < 0.05)
            ),
            synthesis["direction_consistent_target_and_metastasis"],
        ],
        [
            "fdr_support_in_both_primary_settings",
            "direction_consistent_with_nominal_support",
            "direction_consistent_without_nominal_support",
        ],
        default="direction_not_consistent",
    )
    return synthesis


def write_readme() -> None:
    text = f"""Human external validation of frozen canine osteosarcoma programs

Script version: {SCRIPT_VERSION}

Primary external score:
- strict one-to-one ortholog signed-mean z-score
- fixed canine risk direction
- no human outcome used for gene selection, weighting, score orientation, or validation-tier revision

Primary external settings:
1. TARGET-OS overall survival: continuous fixed score, Cox HR per SD, and fixed-score Harrell C-index
2. GSE21257 metastasis within five years: continuous fixed score, logistic OR per SD, ROC-AUC, and PR-AUC

Multiplicity:
- BH correction within each primary setting across M34, M11, M24, and M40
- additional global BH correction across all eight primary tests

Sensitivity analyses:
- strict canine-PCA weighted score
- broad mapped score
- human-cohort PC1
- available clinical adjustment
- proliferation adjustment
- M40 residual to disjoint proliferation
- GSE21257 overall-survival association
- expression-matched random gene-set controls

Interpretation:
- External association is not proof of causality or clinical utility.
- GSE21257 is small; metastasis and OS results require replication.
- TARGET-OS has limited sample size and public clinical covariates.
- Random-gene-set controls are descriptive specificity diagnostics.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def save_manifest(output_paths: list[Path], input_manifest: dict[str, Any]) -> None:
    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "preparation_manifest_sha256": sha256_file(PREPARATION_MANIFEST_FILE),
        "freeze_manifest_sha256": sha256_file(FREEZE_JSON_FILE),
        "primary_modules": PRIMARY_MODULES,
        "secondary_modules": SECONDARY_MODULES,
        "primary_score_suffix": PRIMARY_SCORE_SUFFIX,
        "n_bootstrap": N_BOOTSTRAP,
        "n_random_gene_sets": N_RANDOM_GENE_SETS,
        "random_seed": RANDOM_SEED,
        "multiplicity": {
            "within_setting": "Benjamini-Hochberg across four primary modules",
            "global": "Benjamini-Hochberg across eight primary tests",
        },
        "guardrail": "No human outcome revised any frozen gene set, score direction, weight, or validation tier.",
        "input_preparation_manifest": input_manifest,
        "files": {},
    }
    for path in output_paths:
        if path.exists():
            manifest["files"][path.name] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def print_primary_results(target: pd.DataFrame, gse: pd.DataFrame, synthesis: pd.DataFrame) -> None:
    print("")
    print("=" * 80)
    print("TARGET-OS primary frozen-program validation")
    print("=" * 80)
    target_cols = [
        "module_label", "n", "events", "score_hr_per_sd", "score_ci_low", "score_ci_high",
        "primary_p", "q_within_endpoint", "q_global_eight_tests", "fixed_score_c_index",
        "fixed_score_c_index_ci_low", "fixed_score_c_index_ci_high", "ph_test_p",
        "external_support_class",
    ]
    print(target[[c for c in target_cols if c in target.columns]].to_string(index=False))

    print("")
    print("=" * 80)
    print("GSE21257 metastasis primary frozen-program validation")
    print("=" * 80)
    gse_cols = [
        "module_label", "n", "positives", "negatives", "auc", "auc_ci_low", "auc_ci_high",
        "average_precision", "logistic_or_per_sd", "logistic_ci_low", "logistic_ci_high",
        "primary_p", "q_within_endpoint", "q_global_eight_tests", "external_support_class",
    ]
    print(gse[[c for c in gse_cols if c in gse.columns]].to_string(index=False))

    print("")
    print("=" * 80)
    print("Cross-cohort primary synthesis")
    print("=" * 80)
    syn_cols = [
        "module_label", "target_os_score_hr_per_sd", "target_os_q_within_endpoint",
        "target_os_fixed_score_c_index", "gse_met_auc", "gse_met_q_within_endpoint",
        "gse_met_logistic_or_per_sd", "direction_consistent_target_and_metastasis",
        "target_random_percentile", "gse_random_percentile", "cross_cohort_support_summary",
    ]
    print(synthesis[[c for c in syn_cols if c in synthesis.columns]].to_string(index=False))


def main() -> None:
    print("=" * 80)
    print("External human validation of frozen canine osteosarcoma programs")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Human processed directory: {HUMAN_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print("")
    print("Design:")
    print("  Test frozen strict ortholog scores in independent human osteosarcoma cohorts.")
    print("  Use TARGET-OS overall survival and GSE21257 metastasis within five years.")
    print("  Preserve canine score direction and validation tiers.")
    print("  Apply endpoint-specific and global multiplicity control.")
    print("  Run clinical, proliferation, score-variant, and random-gene-set sensitivity analyses.")

    preparation_manifest = verify_preparation_assets()
    if not FREEZE_JSON_FILE.exists():
        raise FileNotFoundError(f"Freeze file not found: {FREEZE_JSON_FILE}")

    manifest = read_required_csv(FROZEN_MANIFEST_FILE, index_col=None)
    strict_weights = read_required_csv(STRICT_WEIGHTS_FILE, index_col=None)
    broad_weights = read_required_csv(BROAD_WEIGHTS_FILE, index_col=None)
    ortholog_qc = read_required_csv(ORTHOLOG_QC_FILE, index_col=None)

    target_expression = read_required_csv(TARGET_EXPRESSION_FILE, index_col=0)
    target_clinical = read_required_csv(TARGET_CLINICAL_FILE, index_col=0)
    target_scores = clean_scores(read_required_csv(TARGET_SCORES_FILE, index_col=0))
    gse_expression = read_required_csv(GSE_EXPRESSION_FILE, index_col=0)
    gse_clinical = read_required_csv(GSE_CLINICAL_FILE, index_col=0)
    gse_scores = clean_scores(read_required_csv(GSE_SCORES_FILE, index_col=0))

    target_clinical, target_scores, target_expression = match_cohort(
        target_clinical, target_scores, target_expression
    )
    gse_clinical, gse_scores, gse_expression = match_cohort(
        gse_clinical, gse_scores, gse_expression
    )

    print("")
    print("Matched human validation data:")
    print(f"  TARGET-OS expression: {target_expression.shape}")
    print(f"  TARGET-OS clinical: {target_clinical.shape}")
    print(f"  TARGET-OS scores: {target_scores.shape}")
    target_complete = target_clinical[["os_time_days", "os_event"]].dropna()
    print(f"  TARGET-OS OS complete: {target_complete.shape[0]}")
    print(f"  TARGET-OS OS events: {int(safe_numeric(target_complete['os_event']).sum())}")
    print(f"  TARGET-OS censored: {int((safe_numeric(target_complete['os_event']) == 0).sum())}")
    print(f"  GSE21257 expression: {gse_expression.shape}")
    print(f"  GSE21257 clinical: {gse_clinical.shape}")
    print(f"  GSE21257 scores: {gse_scores.shape}")
    print(f"  GSE21257 metastasis positive: {int((safe_numeric(gse_clinical['metastasis_within_5y']) == 1).sum())}")
    print(f"  GSE21257 metastasis negative: {int((safe_numeric(gse_clinical['metastasis_within_5y']) == 0).sum())}")
    gse_os_complete = gse_clinical[["os_time_months", "os_event"]].dropna()
    print(f"  GSE21257 OS complete: {gse_os_complete.shape[0]}")
    print(f"  GSE21257 OS events: {int(safe_numeric(gse_os_complete['os_event']).sum())}")

    rng = np.random.default_rng(RANDOM_SEED)

    target_primary = run_primary_survival(
        target_clinical, target_scores, "TARGET_OS", "os_time_days", "os_event", rng
    )
    gse_primary = run_primary_metastasis(gse_clinical, gse_scores, rng)
    target_primary, gse_primary, multiplicity = add_primary_multiplicity(
        target_primary, gse_primary
    )

    secondary = run_secondary_programs(
        target_clinical, target_scores, gse_clinical, gse_scores, rng
    )
    score_variants = run_score_variant_sensitivity(
        target_clinical, target_scores, gse_clinical, gse_scores, rng
    )
    adjusted = run_adjusted_models(
        target_clinical, target_scores, gse_clinical, gse_scores
    )
    gse_os = run_gse_os_sensitivity(gse_clinical, gse_scores, rng)

    random_summary, random_distribution = run_random_controls(
        target_expression, target_clinical, target_primary,
        gse_expression, gse_clinical, gse_primary,
        strict_weights, ortholog_qc, rng,
    )

    synthesis = build_cross_cohort_synthesis(
        target_primary, gse_primary, gse_os, random_summary
    )

    target_primary.to_csv(OUTPUT_TARGET_PRIMARY, index=False)
    gse_primary.to_csv(OUTPUT_GSE_MET_PRIMARY, index=False)
    gse_os.to_csv(OUTPUT_GSE_OS_SENSITIVITY, index=False)
    secondary.to_csv(OUTPUT_SECONDARY, index=False)
    score_variants.to_csv(OUTPUT_SCORE_VARIANTS, index=False)
    adjusted.to_csv(OUTPUT_ADJUSTED, index=False)
    multiplicity.to_csv(OUTPUT_MULTIPLICITY, index=False)
    random_summary.to_csv(OUTPUT_RANDOM_CONTROLS, index=False)
    random_distribution.to_csv(OUTPUT_RANDOM_CONTROL_DISTRIBUTION, index=False)
    synthesis.to_csv(OUTPUT_SYNTHESIS, index=False)
    write_readme()

    output_paths = [
        OUTPUT_TARGET_PRIMARY,
        OUTPUT_GSE_MET_PRIMARY,
        OUTPUT_GSE_OS_SENSITIVITY,
        OUTPUT_SECONDARY,
        OUTPUT_SCORE_VARIANTS,
        OUTPUT_ADJUSTED,
        OUTPUT_MULTIPLICITY,
        OUTPUT_RANDOM_CONTROLS,
        OUTPUT_RANDOM_CONTROL_DISTRIBUTION,
        OUTPUT_SYNTHESIS,
        OUTPUT_README,
    ]
    save_manifest(output_paths, preparation_manifest)

    print_primary_results(target_primary, gse_primary, synthesis)

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("Primary inference uses only strict one-to-one signed-mean scores with frozen canine risk direction.")
    print("TARGET-OS and GSE21257 are treated as distinct primary external settings; global FDR across eight tests is also reported.")
    print("GSE21257 OS, broad mappings, human PC1, weighted scores, clinical adjustment, and proliferation adjustment are sensitivity analyses.")
    print("Random gene-set percentiles are descriptive specificity controls, not independent external cohorts.")
    print("External association does not establish causality, treatment response, or clinical readiness.")
    print("")
    print("Saved:")
    for path in output_paths + [OUTPUT_MANIFEST]:
        print(path)
    print("Done.")


if __name__ == "__main__":
    main()
