from __future__ import annotations

from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

SCRIPT_VERSION = "48-gse39055-pathological-necrosis-response-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HUMAN_DIR = PROJECT_ROOT / "data" / "processed" / "human_validation"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "gse39055_necrosis"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CLINICAL_FILE = HUMAN_DIR / "GSE39055_clinical_standardized.csv"
LOCKED_SCORES_FILE = HUMAN_DIR / "GSE39055_frozen_transfer_scores.csv"
DETECTION_SCORES_FILE = HUMAN_DIR / "GSE39055_detection_aware_frozen_scores.csv"
DETECTION_COVERAGE_FILE = RESULTS_DIR / "GSE39055_detection_aware_module_coverage.csv"
ASSAY_MANIFEST_FILE = RESULTS_DIR / "GSE39055_assay_quality_diagnostic_manifest.json"
FROZEN_MANIFEST_FILE = RESULTS_DIR / "GSE238110_frozen_canine_transfer_program_manifest.csv"

OUTPUT_PRIMARY = RESULTS_DIR / "GSE39055_necrosis_primary_exploratory.csv"
OUTPUT_M40_ASSAY = RESULTS_DIR / "GSE39055_necrosis_M40_assay_rule_sensitivity.csv"
OUTPUT_COMPARATORS = RESULTS_DIR / "GSE39055_necrosis_M40_comparator_sensitivity.csv"
OUTPUT_LOO = RESULTS_DIR / "GSE39055_necrosis_M40_leave_one_out.csv"
OUTPUT_CORRELATIONS = RESULTS_DIR / "GSE39055_necrosis_M40_score_correlations.csv"
OUTPUT_SUMMARY = RESULTS_DIR / "GSE39055_necrosis_analysis_summary.csv"
OUTPUT_README = RESULTS_DIR / "GSE39055_necrosis_analysis_README.txt"
OUTPUT_MANIFEST = RESULTS_DIR / "GSE39055_necrosis_analysis_manifest.json"

SCATTER_PNG = FIGURES_DIR / "GSE39055_M40_percent_necrosis_scatter.png"
SCATTER_PDF = FIGURES_DIR / "GSE39055_M40_percent_necrosis_scatter.pdf"
BOXPLOT_PNG = FIGURES_DIR / "GSE39055_M40_good_response_boxplot.png"
BOXPLOT_PDF = FIGURES_DIR / "GSE39055_M40_good_response_boxplot.pdf"

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]
PRIMARY_TARGET_MODULE = "M40"
CONTINUOUS_ENDPOINT = "percent_necrosis_numeric"
BINARY_ENDPOINT = "good_necrosis_response_ge90"
GOOD_RESPONSE_THRESHOLD = 90.0

N_PERMUTATIONS = 5000
N_BOOTSTRAP = 2000
RANDOM_SEED = 42
MIN_CONTINUOUS_N = 20
MIN_BINARY_PER_CLASS = 5


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_required_csv(path: Path, index_col: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    print(f"Loaded: {path}")
    return pd.read_csv(path, index_col=index_col, low_memory=False)


def read_optional_csv(path: Path, index_col: int | None = None) -> pd.DataFrame:
    if not path.exists():
        print(f"Optional file not found: {path}")
        return pd.DataFrame()
    print(f"Loaded: {path}")
    return pd.read_csv(path, index_col=index_col, low_memory=False)


def zscore_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    sd = numeric.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (numeric - numeric.mean()) / sd


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvalues, errors="coerce")
    result = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.notna() & np.isfinite(p)
    if valid.sum() == 0:
        return result

    values = p.loc[valid].to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    n = ranked.size
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty(n, dtype=float)
    restored[order] = adjusted
    result.loc[valid] = restored
    return result


def normalize_binary(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.where(numeric.isin([0.0, 1.0]))
    return numeric.astype(float)


def safe_spearman(score: pd.Series, outcome: pd.Series) -> float:
    frame = pd.concat(
        [
            pd.to_numeric(score, errors="coerce").rename("score"),
            pd.to_numeric(outcome, errors="coerce").rename("outcome"),
        ],
        axis=1,
    ).dropna()
    if frame.shape[0] < 5:
        return np.nan
    if frame["score"].std() == 0 or frame["outcome"].std() == 0:
        return np.nan
    return float(stats.spearmanr(frame["score"], frame["outcome"]).statistic)


def spearman_permutation_p(
    score: pd.Series,
    outcome: pd.Series,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    frame = pd.concat(
        [
            pd.to_numeric(score, errors="coerce").rename("score"),
            pd.to_numeric(outcome, errors="coerce").rename("outcome"),
        ],
        axis=1,
    ).dropna()
    if frame.shape[0] < MIN_CONTINUOUS_N:
        return np.nan, np.nan

    x = stats.rankdata(frame["score"].to_numpy(dtype=float))
    y = stats.rankdata(frame["outcome"].to_numpy(dtype=float))
    x = (x - x.mean()) / x.std(ddof=1)
    y = (y - y.mean()) / y.std(ddof=1)
    observed = float(np.dot(x, y) / (x.size - 1))

    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(repetitions):
        permuted = rng.permutation(y)
        value = float(np.dot(x, permuted) / (x.size - 1))
        exceed += abs(value) >= abs(observed)

    p = float((exceed + 1) / (repetitions + 1))
    return observed, p


def safe_auc(score: pd.Series, binary: pd.Series) -> float:
    frame = pd.concat(
        [
            pd.to_numeric(score, errors="coerce").rename("score"),
            normalize_binary(binary).rename("binary"),
        ],
        axis=1,
    ).dropna()
    counts = frame["binary"].value_counts()
    if counts.shape[0] != 2 or counts.min() < MIN_BINARY_PER_CLASS:
        return np.nan
    if frame["score"].std() == 0:
        return np.nan
    return float(roc_auc_score(frame["binary"], frame["score"]))


def auc_permutation_p(
    score: pd.Series,
    binary: pd.Series,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    frame = pd.concat(
        [
            pd.to_numeric(score, errors="coerce").rename("score"),
            normalize_binary(binary).rename("binary"),
        ],
        axis=1,
    ).dropna()
    counts = frame["binary"].value_counts()
    if counts.shape[0] != 2 or counts.min() < MIN_BINARY_PER_CLASS:
        return np.nan, np.nan

    x = frame["score"].to_numpy(dtype=float)
    y = frame["binary"].to_numpy(dtype=float)
    observed = float(roc_auc_score(y, x))
    observed_distance = abs(observed - 0.5)

    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(repetitions):
        permuted = rng.permutation(y)
        value = float(roc_auc_score(permuted, x))
        exceed += abs(value - 0.5) >= observed_distance

    p = float((exceed + 1) / (repetitions + 1))
    return observed, p


def logistic_mle(score: pd.Series, binary: pd.Series) -> dict[str, Any]:
    frame = pd.concat(
        [
            zscore_series(score).rename("score"),
            normalize_binary(binary).rename("binary"),
        ],
        axis=1,
    ).dropna()
    counts = frame["binary"].value_counts()
    result = {
        "logistic_or_per_sd": np.nan,
        "logistic_ci_low": np.nan,
        "logistic_ci_high": np.nan,
        "logistic_p": np.nan,
        "logistic_error": "",
    }
    if counts.shape[0] != 2 or counts.min() < MIN_BINARY_PER_CLASS:
        result["logistic_error"] = "insufficient_binary_groups"
        return result

    x = np.column_stack(
        [
            np.ones(frame.shape[0], dtype=float),
            frame["score"].to_numpy(dtype=float),
        ]
    )
    y = frame["binary"].to_numpy(dtype=float)
    beta = np.zeros(2, dtype=float)

    try:
        for _ in range(100):
            probability = np.clip(expit(x @ beta), 1e-8, 1 - 1e-8)
            weights = probability * (1 - probability)
            information = x.T @ (weights[:, None] * x)
            gradient = x.T @ (y - probability)
            step = np.linalg.solve(
                information + np.eye(2) * 1e-8,
                gradient,
            )
            beta_new = beta + step
            if np.max(np.abs(beta_new - beta)) < 1e-9:
                beta = beta_new
                break
            beta = beta_new

        probability = np.clip(expit(x @ beta), 1e-8, 1 - 1e-8)
        weights = probability * (1 - probability)
        information = x.T @ (weights[:, None] * x)
        covariance = np.linalg.inv(information + np.eye(2) * 1e-8)
        se = float(np.sqrt(covariance[1, 1]))
        z_value = float(beta[1] / se)
        p_value = float(2 * stats.norm.sf(abs(z_value)))

        if abs(beta[1]) > 20 or not np.isfinite(se):
            raise RuntimeError("unstable_logistic_fit")

        result.update(
            {
                "logistic_or_per_sd": float(np.exp(beta[1])),
                "logistic_ci_low": float(np.exp(beta[1] - 1.96 * se)),
                "logistic_ci_high": float(np.exp(beta[1] + 1.96 * se)),
                "logistic_p": p_value,
            }
        )
    except Exception as exc:
        result["logistic_error"] = str(exc)[:300]

    return result


def bootstrap_effects(
    score: pd.Series,
    continuous: pd.Series,
    binary: pd.Series,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    frame = pd.concat(
        [
            pd.to_numeric(score, errors="coerce").rename("score"),
            pd.to_numeric(continuous, errors="coerce").rename("continuous"),
            normalize_binary(binary).rename("binary"),
        ],
        axis=1,
    )
    rng = np.random.default_rng(seed)

    continuous_frame = frame[["score", "continuous"]].dropna()
    rho_values: list[float] = []
    slope_values: list[float] = []
    if continuous_frame.shape[0] >= MIN_CONTINUOUS_N:
        for _ in range(repetitions):
            indices = rng.integers(0, continuous_frame.shape[0], continuous_frame.shape[0])
            sample = continuous_frame.iloc[indices]
            score_z = zscore_series(sample["score"])
            if score_z.notna().sum() < MIN_CONTINUOUS_N:
                continue
            rho = safe_spearman(score_z, sample["continuous"])
            if np.isfinite(rho):
                rho_values.append(rho)
            slope = stats.linregress(score_z, sample["continuous"]).slope
            if np.isfinite(slope):
                slope_values.append(float(slope))

    binary_frame = frame[["score", "binary"]].dropna()
    auc_values: list[float] = []
    or_values: list[float] = []
    counts = binary_frame["binary"].value_counts()
    if counts.shape[0] == 2 and counts.min() >= MIN_BINARY_PER_CLASS:
        group_indices = {
            group: np.flatnonzero(binary_frame["binary"].to_numpy() == group)
            for group in [0.0, 1.0]
        }
        for _ in range(repetitions):
            sampled_indices = np.concatenate(
                [
                    rng.choice(indices, size=len(indices), replace=True)
                    for indices in group_indices.values()
                ]
            )
            sample = binary_frame.iloc[sampled_indices]
            auc = safe_auc(sample["score"], sample["binary"])
            if np.isfinite(auc):
                auc_values.append(auc)
            fit = logistic_mle(sample["score"], sample["binary"])
            odds_ratio = fit["logistic_or_per_sd"]
            if np.isfinite(odds_ratio) and 0 < odds_ratio < 1e6:
                or_values.append(float(odds_ratio))

    def interval(values: list[float]) -> tuple[float, float, int]:
        if not values:
            return np.nan, np.nan, 0
        array = np.asarray(values, dtype=float)
        return (
            float(np.quantile(array, 0.025)),
            float(np.quantile(array, 0.975)),
            int(array.size),
        )

    rho_low, rho_high, rho_valid = interval(rho_values)
    slope_low, slope_high, slope_valid = interval(slope_values)
    auc_low, auc_high, auc_valid = interval(auc_values)
    or_low, or_high, or_valid = interval(or_values)

    return {
        "rho_bootstrap_ci_low": rho_low,
        "rho_bootstrap_ci_high": rho_high,
        "rho_bootstrap_valid": rho_valid,
        "slope_bootstrap_ci_low": slope_low,
        "slope_bootstrap_ci_high": slope_high,
        "slope_bootstrap_valid": slope_valid,
        "auc_bootstrap_ci_low": auc_low,
        "auc_bootstrap_ci_high": auc_high,
        "auc_bootstrap_valid": auc_valid,
        "or_bootstrap_ci_low": or_low,
        "or_bootstrap_ci_high": or_high,
        "or_bootstrap_valid": or_valid,
    }


def analyze_score(
    score: pd.Series,
    clinical: pd.DataFrame,
    module_label: str,
    strategy: str,
    score_column: str,
    seed: int,
) -> dict[str, Any]:
    continuous = pd.to_numeric(
        clinical[CONTINUOUS_ENDPOINT],
        errors="coerce",
    )
    binary = normalize_binary(clinical[BINARY_ENDPOINT])
    score_numeric = pd.to_numeric(score, errors="coerce")

    continuous_frame = pd.concat(
        [score_numeric.rename("score"), continuous.rename("outcome")],
        axis=1,
    ).dropna()
    binary_frame = pd.concat(
        [score_numeric.rename("score"), binary.rename("outcome")],
        axis=1,
    ).dropna()

    rho, rho_p = spearman_permutation_p(
        score_numeric,
        continuous,
        N_PERMUTATIONS,
        seed,
    )
    score_z = zscore_series(score_numeric)
    slope = np.nan
    slope_p = np.nan
    if continuous_frame.shape[0] >= MIN_CONTINUOUS_N:
        regression_frame = pd.concat(
            [score_z.rename("score"), continuous.rename("outcome")],
            axis=1,
        ).dropna()
        if regression_frame["score"].std() > 0:
            fit = stats.linregress(
                regression_frame["score"],
                regression_frame["outcome"],
            )
            slope = float(fit.slope)
            slope_p = float(fit.pvalue)

    auc, auc_p = auc_permutation_p(
        score_numeric,
        binary,
        N_PERMUTATIONS,
        seed + 1,
    )
    logistic = logistic_mle(score_numeric, binary)
    bootstrap = bootstrap_effects(
        score_numeric,
        continuous,
        binary,
        N_BOOTSTRAP,
        seed + 2,
    )

    return {
        "module_label": module_label,
        "strategy": strategy,
        "score_column": score_column,
        "n_continuous": int(continuous_frame.shape[0]),
        "spearman_rho": rho,
        "spearman_permutation_p": rho_p,
        "slope_percent_necrosis_per_score_sd": slope,
        "slope_p": slope_p,
        "n_binary": int(binary_frame.shape[0]),
        "n_good_response": int((binary_frame["outcome"] == 1).sum()),
        "n_poor_response": int((binary_frame["outcome"] == 0).sum()),
        "auc_higher_score_predicts_good_response": auc,
        "auc_permutation_p_two_sided": auc_p,
        **logistic,
        **bootstrap,
    }


def find_primary_score_column(scores: pd.DataFrame, module: str) -> str:
    exact = f"{module}__strict__signed_mean_z"
    if exact in scores.columns:
        return exact

    candidates = [
        column
        for column in scores.columns
        if column.startswith(f"{module}__")
        and "strict" in column.lower()
        and "signed_mean" in column.lower()
        and "residual" not in column.lower()
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(
        f"Could not resolve one primary locked score for {module}. "
        f"Candidates: {candidates}"
    )


def find_m40_comparator_columns(scores: pd.DataFrame) -> list[tuple[str, str]]:
    comparators: list[tuple[str, str]] = []

    residual_candidates = [
        column
        for column in scores.columns
        if column.startswith("M40__")
        and "residual_to_disjoint_proliferation" in column.lower()
    ]
    for column in sorted(residual_candidates):
        comparators.append(("M40_residual_to_disjoint_proliferation", column))

    proliferation_candidates = [
        column
        for column in scores.columns
        if "proliferation" in column.lower()
        and not column.startswith("M40__")
        and "residual" not in column.lower()
    ]
    priority = sorted(
        proliferation_candidates,
        key=lambda column: (
            "strict_human_meta_proliferation_pc1_z" not in column.lower(),
            "pc1" not in column.lower(),
            column,
        ),
    )
    if priority:
        comparators.append(("human_proliferation", priority[0]))

    return comparators


def primary_module_analysis(
    clinical: pd.DataFrame,
    locked_scores: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for index, module in enumerate(PRIMARY_MODULES):
        column = find_primary_score_column(locked_scores, module)
        rows.append(
            analyze_score(
                score=locked_scores[column],
                clinical=clinical,
                module_label=module,
                strategy="locked_strict_signed_mean",
                score_column=column,
                seed=RANDOM_SEED + index * 100,
            )
        )

    result = pd.DataFrame(rows)
    result["continuous_q_bh_4"] = bh_adjust(
        result["spearman_permutation_p"]
    )
    result["binary_q_bh_4"] = bh_adjust(
        result["auc_permutation_p_two_sided"]
    )
    result["prespecified_primary_exploratory_module"] = result[
        "module_label"
    ].eq(PRIMARY_TARGET_MODULE)
    return result


def m40_assay_rule_analysis(
    clinical: pd.DataFrame,
    locked_scores: pd.DataFrame,
    detection_scores: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    locked_column = find_primary_score_column(locked_scores, "M40")
    rows.append(
        analyze_score(
            score=locked_scores[locked_column],
            clinical=clinical,
            module_label="M40",
            strategy="locked_strict_signed_mean",
            score_column=locked_column,
            seed=RANDOM_SEED + 1000,
        )
    )

    if not detection_scores.empty:
        for offset, column in enumerate(
            sorted(
                column
                for column in detection_scores.columns
                if column.startswith("M40__")
            )
        ):
            strategy = column.split("__", 1)[1]
            rows.append(
                analyze_score(
                    score=detection_scores[column],
                    clinical=clinical,
                    module_label="M40",
                    strategy=strategy,
                    score_column=column,
                    seed=RANDOM_SEED + 1100 + offset * 20,
                )
            )

    result = pd.DataFrame(rows).drop_duplicates(
        ["strategy", "score_column"],
        keep="first",
    )

    if not coverage.empty:
        coverage_part = coverage[
            coverage["module_label"].astype(str).eq("M40")
        ].copy()
        keep = [
            column
            for column in [
                "strategy",
                "n_frozen_genes",
                "n_available_genes",
                "coverage_fraction",
                "median_gene_detected_fraction_p_lt_0_01",
            ]
            if column in coverage_part.columns
        ]
        result = result.merge(
            coverage_part[keep].drop_duplicates("strategy"),
            on="strategy",
            how="left",
        )

    result["continuous_q_bh_across_strategies"] = bh_adjust(
        result["spearman_permutation_p"]
    )
    result["binary_q_bh_across_strategies"] = bh_adjust(
        result["auc_permutation_p_two_sided"]
    )
    return result


def m40_comparator_analysis(
    clinical: pd.DataFrame,
    locked_scores: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    m40_column = find_primary_score_column(locked_scores, "M40")
    rows.append(
        analyze_score(
            score=locked_scores[m40_column],
            clinical=clinical,
            module_label="M40",
            strategy="locked_M40",
            score_column=m40_column,
            seed=RANDOM_SEED + 2000,
        )
    )

    for offset, (label, column) in enumerate(
        find_m40_comparator_columns(locked_scores)
    ):
        rows.append(
            analyze_score(
                score=locked_scores[column],
                clinical=clinical,
                module_label="M40",
                strategy=label,
                score_column=column,
                seed=RANDOM_SEED + 2100 + offset * 50,
            )
        )

    return pd.DataFrame(rows)


def leave_one_out_m40(
    clinical: pd.DataFrame,
    score: pd.Series,
) -> pd.DataFrame:
    frame = clinical[[CONTINUOUS_ENDPOINT, BINARY_ENDPOINT]].copy()
    frame["score"] = pd.to_numeric(score, errors="coerce")
    rows = []

    for sample_id in frame.index:
        part = frame.drop(index=sample_id)
        rho = safe_spearman(part["score"], part[CONTINUOUS_ENDPOINT])
        auc = safe_auc(part["score"], part[BINARY_ENDPOINT])
        rows.append(
            {
                "sample_removed": sample_id,
                "spearman_rho": rho,
                "auc_higher_score_predicts_good_response": auc,
            }
        )

    return pd.DataFrame(rows)


def score_correlation_matrix(
    locked_scores: pd.DataFrame,
    detection_scores: pd.DataFrame,
) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    locked_column = find_primary_score_column(locked_scores, "M40")
    columns["locked_strict_signed_mean"] = locked_scores[locked_column]

    if not detection_scores.empty:
        for column in sorted(
            column
            for column in detection_scores.columns
            if column.startswith("M40__")
        ):
            columns[column.split("__", 1)[1]] = detection_scores[column]

    matrix = pd.DataFrame(columns).corr(method="spearman")
    matrix.index.name = "strategy"
    return matrix


def build_summary(
    primary: pd.DataFrame,
    assay: pd.DataFrame,
    comparators: pd.DataFrame,
    loo: pd.DataFrame,
) -> pd.DataFrame:
    m40_primary = primary[primary["module_label"].eq("M40")].iloc[0]
    assay_valid = assay[
        assay["spearman_rho"].notna()
        & assay["auc_higher_score_predicts_good_response"].notna()
    ].copy()

    if assay_valid.empty:
        fraction_positive_rho = np.nan
        fraction_auc_above_half = np.nan
        assay_class = "not_estimable"
    else:
        fraction_positive_rho = float(
            (assay_valid["spearman_rho"] > 0).mean()
        )
        fraction_auc_above_half = float(
            (
                assay_valid[
                    "auc_higher_score_predicts_good_response"
                ]
                > 0.5
            ).mean()
        )
        stable_positive = (
            fraction_positive_rho >= 0.80
            and fraction_auc_above_half >= 0.80
        )
        stable_negative = (
            fraction_positive_rho <= 0.20
            and fraction_auc_above_half <= 0.20
        )
        assay_class = (
            "stable_higher_score_higher_necrosis"
            if stable_positive
            else (
                "stable_higher_score_lower_necrosis"
                if stable_negative
                else "assay_rule_sensitive_or_heterogeneous"
            )
        )

    if (
        np.isfinite(m40_primary["continuous_q_bh_4"])
        and m40_primary["continuous_q_bh_4"] < 0.05
    ) or (
        np.isfinite(m40_primary["binary_q_bh_4"])
        and m40_primary["binary_q_bh_4"] < 0.05
    ):
        primary_support = "exploratory_fdr_support"
    elif (
        np.isfinite(m40_primary["spearman_permutation_p"])
        and m40_primary["spearman_permutation_p"] < 0.05
    ) or (
        np.isfinite(m40_primary["auc_permutation_p_two_sided"])
        and m40_primary["auc_permutation_p_two_sided"] < 0.05
    ):
        primary_support = "exploratory_nominal_support"
    else:
        primary_support = "no_nominal_exploratory_support"

    comparator_index = comparators.set_index("strategy")
    residual_rho = (
        comparator_index.loc[
            "M40_residual_to_disjoint_proliferation",
            "spearman_rho",
        ]
        if "M40_residual_to_disjoint_proliferation" in comparator_index.index
        else np.nan
    )
    proliferation_rho = (
        comparator_index.loc["human_proliferation", "spearman_rho"]
        if "human_proliferation" in comparator_index.index
        else np.nan
    )

    return pd.DataFrame(
        [
            {
                "analysis": "GSE39055_pathological_necrosis_response",
                "n_percent_necrosis": int(m40_primary["n_continuous"]),
                "n_good_response": int(m40_primary["n_good_response"]),
                "n_poor_response": int(m40_primary["n_poor_response"]),
                "m40_locked_spearman_rho": m40_primary["spearman_rho"],
                "m40_locked_continuous_p": m40_primary[
                    "spearman_permutation_p"
                ],
                "m40_locked_auc": m40_primary[
                    "auc_higher_score_predicts_good_response"
                ],
                "m40_locked_binary_p": m40_primary[
                    "auc_permutation_p_two_sided"
                ],
                "m40_primary_exploratory_support": primary_support,
                "n_m40_assay_strategies_estimable": int(
                    assay_valid.shape[0]
                ),
                "fraction_m40_strategies_positive_rho": fraction_positive_rho,
                "fraction_m40_strategies_auc_above_0_5": fraction_auc_above_half,
                "m40_assay_rule_response_class": assay_class,
                "m40_residual_spearman_rho": residual_rho,
                "human_proliferation_spearman_rho": proliferation_rho,
                "loo_rho_min": float(loo["spearman_rho"].min()),
                "loo_rho_median": float(loo["spearman_rho"].median()),
                "loo_rho_max": float(loo["spearman_rho"].max()),
                "interpretation_guardrail": (
                    "Association with subsequent pathological necrosis is not "
                    "evidence of chemotherapy benefit because no untreated "
                    "comparator is available. Detection-aware score strategies "
                    "are sensitivity analyses and cannot replace the frozen "
                    "primary score."
                ),
            }
        ]
    )


def create_figures(
    clinical: pd.DataFrame,
    score: pd.Series,
) -> None:
    part = pd.concat(
        [
            zscore_series(score).rename("score"),
            pd.to_numeric(
                clinical[CONTINUOUS_ENDPOINT],
                errors="coerce",
            ).rename("percent_necrosis"),
            normalize_binary(
                clinical[BINARY_ENDPOINT]
            ).rename("good_response"),
        ],
        axis=1,
    )

    continuous = part[["score", "percent_necrosis"]].dropna()
    if continuous.shape[0] >= MIN_CONTINUOUS_N:
        fit = stats.linregress(
            continuous["score"],
            continuous["percent_necrosis"],
        )
        x_grid = np.linspace(
            float(continuous["score"].min()),
            float(continuous["score"].max()),
            100,
        )
        y_grid = fit.intercept + fit.slope * x_grid

        fig, ax = plt.subplots(figsize=(6.0, 4.8))
        ax.scatter(
            continuous["score"],
            continuous["percent_necrosis"],
            alpha=0.75,
        )
        ax.plot(x_grid, y_grid)
        ax.axhline(GOOD_RESPONSE_THRESHOLD, linestyle="--", linewidth=1)
        ax.set_xlabel("Frozen M40 risk-oriented score (SD)")
        ax.set_ylabel("Pathological tumor necrosis (%)")
        ax.set_title("GSE39055 M40 and pathological necrosis")
        fig.tight_layout()
        fig.savefig(SCATTER_PNG, dpi=300, bbox_inches="tight")
        fig.savefig(SCATTER_PDF, bbox_inches="tight")
        plt.close(fig)

    binary = part[["score", "good_response"]].dropna()
    counts = binary["good_response"].value_counts()
    if counts.shape[0] == 2 and counts.min() >= MIN_BINARY_PER_CLASS:
        groups = [
            binary.loc[
                binary["good_response"].eq(0),
                "score",
            ].to_numpy(dtype=float),
            binary.loc[
                binary["good_response"].eq(1),
                "score",
            ].to_numpy(dtype=float),
        ]
        fig, ax = plt.subplots(figsize=(5.4, 4.7))
        ax.boxplot(
            groups,
            tick_labels=["<90%", "≥90%"],
            showfliers=False,
        )
        rng = np.random.default_rng(RANDOM_SEED)
        for position, values in enumerate(groups, start=1):
            jitter = rng.normal(0.0, 0.04, size=len(values))
            ax.scatter(
                np.full(len(values), position) + jitter,
                values,
                alpha=0.70,
                s=22,
            )
        ax.set_ylabel("Frozen M40 risk-oriented score (SD)")
        ax.set_title("GSE39055 M40 by pathological response")
        fig.tight_layout()
        fig.savefig(BOXPLOT_PNG, dpi=300, bbox_inches="tight")
        fig.savefig(BOXPLOT_PDF, bbox_inches="tight")
        plt.close(fig)


def write_readme() -> None:
    text = f"""GSE39055 frozen-program association with pathological necrosis
Script version: {SCRIPT_VERSION}

Purpose
-------
Test whether baseline diagnostic-biopsy frozen program scores are associated
with subsequent pathological tumor necrosis after neoadjuvant chemotherapy.

Primary exploratory hypothesis
------------------------------
M40 is the prespecified primary exploratory module because it is a frozen
proliferation/cycling program with strong independent representation support.

Endpoints
---------
1. Continuous percent tumor necrosis.
2. Good pathological response defined as at least {GOOD_RESPONSE_THRESHOLD:.0f}% necrosis.

Primary molecular analysis
--------------------------
The frozen strict one-to-one M40 signed-mean score is primary. M34, M11, and
M24 are analyzed as a four-module exploratory family with BH correction.

Assay sensitivity
-----------------
All outcome-blind M40 probe-selection strategies from script 31 are tested
without choosing a preferred rule after viewing necrosis. Mixed directions
across these rules indicate assay-rule sensitivity rather than a stable
response association.

Mechanistic comparators
-----------------------
When available, the frozen human proliferation score and the frozen M40
residual-to-disjoint-proliferation score are reported as sensitivity analyses.

Interpretation guardrails
-------------------------
- RNA was measured in diagnostic biopsy specimens before pathological response was observed.
- Percent necrosis is a post-treatment response endpoint.
- Association may be described as association with subsequent pathological response.
- The analysis does not estimate chemotherapy benefit because there is no untreated comparator.
- The cohort contains only 37 patients, so effect estimates are exploratory.
- This analysis cannot modify frozen module membership, score orientation, human outcome tiers, or project-wide multiplicity conclusions.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def create_manifest(input_paths: list[Path], output_paths: list[Path]) -> None:
    payload = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "primary_exploratory_module": PRIMARY_TARGET_MODULE,
        "continuous_endpoint": CONTINUOUS_ENDPOINT,
        "binary_endpoint": BINARY_ENDPOINT,
        "good_response_threshold_percent": GOOD_RESPONSE_THRESHOLD,
        "permutations": N_PERMUTATIONS,
        "bootstrap_repetitions": N_BOOTSTRAP,
        "guardrails": [
            "No score direction or module definition is changed.",
            "Detection-aware strategies are outcome-blind sensitivities.",
            "No untreated comparator is available.",
            "Chemotherapy benefit is not estimated.",
        ],
        "inputs": {},
        "outputs": {},
    }

    for path in input_paths:
        if path.exists():
            payload["inputs"][path.name] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }

    for path in output_paths:
        if path.exists():
            payload["outputs"][path.name] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }

    OUTPUT_MANIFEST.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 80)
    print("GSE39055 frozen programs and pathological necrosis response")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Design:")
    print("  Test frozen baseline-biopsy scores against subsequent necrosis.")
    print("  Treat M40 as the prespecified primary exploratory module.")
    print("  Use continuous necrosis and the fixed 90% response threshold.")
    print("  Audit all outcome-blind M40 assay rules without selecting one.")
    print("  Compare M40 with proliferation and residual M40 when available.")
    print("")

    clinical = read_required_csv(CLINICAL_FILE, index_col=0)
    locked_scores = read_required_csv(LOCKED_SCORES_FILE, index_col=0)
    detection_scores = read_optional_csv(DETECTION_SCORES_FILE, index_col=0)
    coverage = read_optional_csv(DETECTION_COVERAGE_FILE)
    read_optional_csv(FROZEN_MANIFEST_FILE)

    if ASSAY_MANIFEST_FILE.exists():
        print(f"Loaded: {ASSAY_MANIFEST_FILE}")
        assay_manifest = json.loads(
            ASSAY_MANIFEST_FILE.read_text(encoding="utf-8")
        )
        observed_version = str(assay_manifest.get("script_version", ""))
        if observed_version and not observed_version.endswith("v2"):
            raise RuntimeError(
                "The GSE39055 assay manifest is not from the corrected "
                f"script 31 v2. Observed: {observed_version}"
            )

    common = clinical.index.intersection(locked_scores.index)
    if not detection_scores.empty:
        common = common.intersection(detection_scores.index)

    clinical = clinical.loc[common].copy()
    locked_scores = locked_scores.loc[common].copy()
    if not detection_scores.empty:
        detection_scores = detection_scores.loc[common].copy()

    if CONTINUOUS_ENDPOINT not in clinical.columns:
        raise RuntimeError(
            f"Missing continuous necrosis endpoint: {CONTINUOUS_ENDPOINT}"
        )
    if BINARY_ENDPOINT not in clinical.columns:
        raise RuntimeError(
            f"Missing binary necrosis endpoint: {BINARY_ENDPOINT}"
        )

    clinical[CONTINUOUS_ENDPOINT] = pd.to_numeric(
        clinical[CONTINUOUS_ENDPOINT],
        errors="coerce",
    )
    clinical[BINARY_ENDPOINT] = normalize_binary(
        clinical[BINARY_ENDPOINT]
    )

    endpoint_frame = clinical[
        [CONTINUOUS_ENDPOINT, BINARY_ENDPOINT]
    ].copy()
    print("Matched endpoint data:")
    print(f"  Samples: {endpoint_frame.shape[0]}")
    print(
        f"  Continuous necrosis complete: "
        f"{endpoint_frame[CONTINUOUS_ENDPOINT].notna().sum()}"
    )
    print(
        f"  Good response >=90%: "
        f"{int(endpoint_frame[BINARY_ENDPOINT].eq(1).sum())}"
    )
    print(
        f"  Poor response <90%: "
        f"{int(endpoint_frame[BINARY_ENDPOINT].eq(0).sum())}"
    )

    primary = primary_module_analysis(clinical, locked_scores)
    assay = m40_assay_rule_analysis(
        clinical=clinical,
        locked_scores=locked_scores,
        detection_scores=detection_scores,
        coverage=coverage,
    )
    comparators = m40_comparator_analysis(clinical, locked_scores)

    m40_column = find_primary_score_column(locked_scores, "M40")
    loo = leave_one_out_m40(clinical, locked_scores[m40_column])
    correlations = score_correlation_matrix(
        locked_scores,
        detection_scores,
    )
    summary = build_summary(primary, assay, comparators, loo)

    primary.to_csv(OUTPUT_PRIMARY, index=False)
    assay.to_csv(OUTPUT_M40_ASSAY, index=False)
    comparators.to_csv(OUTPUT_COMPARATORS, index=False)
    loo.to_csv(OUTPUT_LOO, index=False)
    correlations.to_csv(OUTPUT_CORRELATIONS)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    create_figures(clinical, locked_scores[m40_column])
    write_readme()

    output_paths = [
        OUTPUT_PRIMARY,
        OUTPUT_M40_ASSAY,
        OUTPUT_COMPARATORS,
        OUTPUT_LOO,
        OUTPUT_CORRELATIONS,
        OUTPUT_SUMMARY,
        OUTPUT_README,
        SCATTER_PNG,
        SCATTER_PDF,
        BOXPLOT_PNG,
        BOXPLOT_PDF,
    ]
    create_manifest(
        input_paths=[
            CLINICAL_FILE,
            LOCKED_SCORES_FILE,
            DETECTION_SCORES_FILE,
            DETECTION_COVERAGE_FILE,
            ASSAY_MANIFEST_FILE,
            FROZEN_MANIFEST_FILE,
        ],
        output_paths=output_paths,
    )

    print("")
    print("=" * 80)
    print("Primary frozen-module necrosis associations")
    print("=" * 80)
    primary_columns = [
        "module_label",
        "n_continuous",
        "spearman_rho",
        "rho_bootstrap_ci_low",
        "rho_bootstrap_ci_high",
        "spearman_permutation_p",
        "continuous_q_bh_4",
        "slope_percent_necrosis_per_score_sd",
        "n_good_response",
        "n_poor_response",
        "auc_higher_score_predicts_good_response",
        "auc_bootstrap_ci_low",
        "auc_bootstrap_ci_high",
        "auc_permutation_p_two_sided",
        "binary_q_bh_4",
        "logistic_or_per_sd",
        "logistic_ci_low",
        "logistic_ci_high",
    ]
    print(primary[primary_columns].to_string(index=False))

    print("")
    print("=" * 80)
    print("M40 assay-rule sensitivity")
    print("=" * 80)
    assay_columns = [
        "strategy",
        "n_available_genes",
        "coverage_fraction",
        "spearman_rho",
        "spearman_permutation_p",
        "auc_higher_score_predicts_good_response",
        "auc_permutation_p_two_sided",
        "logistic_or_per_sd",
    ]
    assay_columns = [
        column for column in assay_columns if column in assay.columns
    ]
    print(assay[assay_columns].to_string(index=False))

    print("")
    print("=" * 80)
    print("M40 versus proliferation comparators")
    print("=" * 80)
    comparator_columns = [
        "strategy",
        "score_column",
        "spearman_rho",
        "spearman_permutation_p",
        "auc_higher_score_predicts_good_response",
        "auc_permutation_p_two_sided",
        "logistic_or_per_sd",
    ]
    print(comparators[comparator_columns].to_string(index=False))

    print("")
    print("=" * 80)
    print("Necrosis analysis summary")
    print("=" * 80)
    print(summary.to_string(index=False))

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("This is an exploratory pathological-response association analysis.")
    print("It does not estimate chemotherapy benefit because no untreated comparator exists.")
    print("Detection-aware score rules are sensitivities and are not selected by outcome.")
    print("No result can change frozen genes, score direction, or locked human evidence tiers.")

    print("")
    print("Saved:")
    for path in output_paths + [OUTPUT_MANIFEST]:
        if path.exists():
            print(path)
    print("Done.")


if __name__ == "__main__":
    main()
