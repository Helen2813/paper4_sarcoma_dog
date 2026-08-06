from __future__ import annotations

from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index

SCRIPT_VERSION = "26-gse39055-rfs-validation-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
HUMAN_PROCESSED_DIR = PROCESSED_DIR / "human_validation"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREPARATION_MANIFEST_FILE = RESULTS_DIR / "GSE39055_preparation_manifest.json"
EXPRESSION_FILE = HUMAN_PROCESSED_DIR / "GSE39055_expression_gene_symbol.csv"
CLINICAL_FILE = HUMAN_PROCESSED_DIR / "GSE39055_clinical_standardized.csv"
SCORES_FILE = HUMAN_PROCESSED_DIR / "GSE39055_frozen_transfer_scores.csv"
COVERAGE_FILE = RESULTS_DIR / "GSE39055_frozen_transfer_score_coverage.csv"

FROZEN_PROGRAM_MANIFEST_FILE = (
    RESULTS_DIR / "GSE238110_frozen_canine_transfer_program_manifest.csv"
)
STRICT_WEIGHTS_FILE = RESULTS_DIR / "GSE238110_frozen_transfer_gene_weights_strict.csv"
BROAD_WEIGHTS_FILE = RESULTS_DIR / "GSE238110_frozen_transfer_gene_weights_broad.csv"
PREVIOUS_ROBUST_SUMMARY_FILE = (
    RESULTS_DIR / "human_external_validation_robust_evidence_summary.csv"
)

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]
SECONDARY_MODULES = ["M28", "M38", "M25", "M17"]

TIME_COL = "rfs_time_months"
EVENT_COL = "recurrence_event"
PRIMARY_SCORE_SUFFIX = "__strict__signed_mean_z"

COX_PENALIZER = 0.05
N_BOOTSTRAP = 5000
N_PERMUTATIONS = 10000
N_RANDOM_SETS = 2000
RANDOM_SEED = 42
EPSILON_TIME_MONTHS = 1.0 / 30.0

OUTPUT_PRIMARY = RESULTS_DIR / "GSE39055_RFS_primary_frozen_program_validation.csv"
OUTPUT_MULTIPLICITY = RESULTS_DIR / "GSE39055_RFS_primary_multiplicity.csv"
OUTPUT_ADJUSTED = RESULTS_DIR / "GSE39055_RFS_adjustment_robustness.csv"
OUTPUT_VARIANTS = RESULTS_DIR / "GSE39055_RFS_score_variant_sensitivity.csv"
OUTPUT_LOO = RESULTS_DIR / "GSE39055_RFS_leave_one_out_stability.csv"
OUTPUT_RANDOM = RESULTS_DIR / "GSE39055_RFS_random_gene_set_controls.csv"
OUTPUT_RANDOM_DISTRIBUTION = (
    RESULTS_DIR / "GSE39055_RFS_random_gene_set_distribution.csv"
)
OUTPUT_ZERO_TIME_AUDIT = RESULTS_DIR / "GSE39055_zero_time_endpoint_audit.csv"
OUTPUT_CROSS_COHORT = RESULTS_DIR / "human_external_validation_three_cohort_synthesis.csv"
OUTPUT_MANIFEST = RESULTS_DIR / "GSE39055_RFS_validation_manifest.json"
OUTPUT_README = RESULTS_DIR / "GSE39055_RFS_validation_README.txt"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_required_csv(path: Path, index_col: int | str | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    print(f"Loaded: {path}")
    return pd.read_csv(path, index_col=index_col)


def bh_adjust(values: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty(n, dtype=float)
    restored[order] = adjusted
    q[valid] = restored
    return q


def verify_preparation_inputs() -> dict[str, Any]:
    if not PREPARATION_MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Preparation manifest is missing: {PREPARATION_MANIFEST_FILE}"
        )
    manifest = json.loads(PREPARATION_MANIFEST_FILE.read_text(encoding="utf-8"))
    recorded = manifest.get("files", {})

    print("")
    print("GSE39055 preparation input integrity check:")
    for path in [EXPRESSION_FILE, CLINICAL_FILE, SCORES_FILE, COVERAGE_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Prepared input is missing: {path}")
        expected = recorded.get(path.name, {}).get("sha256")
        observed = sha256_file(path)
        if expected and observed != expected:
            raise RuntimeError(
                f"Prepared input hash mismatch for {path.name}. "
                "Do not continue after changing a prepared file."
            )
        print(f"  {path.name}: {'verified' if expected else 'present_without_recorded_hash'}")
    return manifest


def zscore_series(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    std = values.std()
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (values - values.mean()) / std


def safe_fixed_direction_c_index(
    time: pd.Series,
    event: pd.Series,
    risk_score: pd.Series,
) -> float:
    frame = pd.concat(
        [
            pd.to_numeric(time, errors="coerce").rename("time"),
            pd.to_numeric(event, errors="coerce").rename("event"),
            pd.to_numeric(risk_score, errors="coerce").rename("risk"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if (
        frame.shape[0] < 5
        or frame["event"].sum() < 2
        or frame["event"].nunique() < 2
    ):
        return np.nan
    return float(
        concordance_index(
            frame["time"].values,
            -frame["risk"].values,
            frame["event"].values,
        )
    )


def fit_cox(
    data: pd.DataFrame,
    score_col: str,
    covariates: list[str] | None = None,
    time_col: str = TIME_COL,
    event_col: str = EVENT_COL,
    penalizer: float = COX_PENALIZER,
) -> dict[str, Any]:
    covariates = covariates or []
    columns = [time_col, event_col, score_col] + covariates
    frame = data[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()

    result: dict[str, Any] = {
        "n": frame.shape[0],
        "events": int(frame[event_col].sum()) if frame.shape[0] else 0,
        "hr_per_sd": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "p": np.nan,
        "coef": np.nan,
        "se_coef": np.nan,
        "c_index": np.nan,
        "ph_test_p": np.nan,
        "error": "",
    }

    if frame.shape[0] < 20:
        result["error"] = "too_few_samples"
        return result
    if frame[event_col].sum() < 5:
        result["error"] = "too_few_events"
        return result
    if frame[score_col].std() == 0:
        result["error"] = "zero_variance_score"
        return result

    model = CoxPHFitter(penalizer=penalizer)
    try:
        model.fit(
            frame,
            duration_col=time_col,
            event_col=event_col,
            fit_options={"max_steps": 500},
        )
        summary = model.summary.loc[score_col]
        result.update(
            {
                "hr_per_sd": float(summary["exp(coef)"]),
                "ci_low": float(summary["exp(coef) lower 95%"]),
                "ci_high": float(summary["exp(coef) upper 95%"]),
                "p": float(summary["p"]),
                "coef": float(summary["coef"]),
                "se_coef": float(summary["se(coef)"]),
                "c_index": float(model.concordance_index_),
            }
        )
        try:
            ph = proportional_hazard_test(
                model,
                frame,
                time_transform="rank",
            )
            result["ph_test_p"] = float(ph.summary.loc[score_col, "p"])
        except Exception:
            pass
    except Exception as exc:
        result["error"] = str(exc)[:500]
    return result


def stratified_bootstrap_c_index(
    data: pd.DataFrame,
    score_col: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float, int]:
    frame = data[[TIME_COL, EVENT_COL, score_col]].dropna().copy()
    event_idx = np.where(frame[EVENT_COL].values == 1)[0]
    censor_idx = np.where(frame[EVENT_COL].values == 0)[0]
    rng = np.random.default_rng(seed)
    values: list[float] = []

    for _ in range(n_bootstrap):
        sampled_event = rng.choice(event_idx, size=len(event_idx), replace=True)
        sampled_censor = rng.choice(censor_idx, size=len(censor_idx), replace=True)
        sampled = np.concatenate([sampled_event, sampled_censor])
        rng.shuffle(sampled)
        part = frame.iloc[sampled]
        value = safe_fixed_direction_c_index(
            part[TIME_COL],
            part[EVENT_COL],
            part[score_col],
        )
        if np.isfinite(value):
            values.append(value)

    observed = safe_fixed_direction_c_index(
        frame[TIME_COL],
        frame[EVENT_COL],
        frame[score_col],
    )
    if not values:
        return observed, np.nan, np.nan, 0
    return (
        observed,
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        len(values),
    )


def permutation_c_index_p(
    data: pd.DataFrame,
    score_col: str,
    n_permutations: int,
    seed: int,
) -> tuple[float, float, float]:
    frame = data[[TIME_COL, EVENT_COL, score_col]].dropna().copy()
    observed = safe_fixed_direction_c_index(
        frame[TIME_COL],
        frame[EVENT_COL],
        frame[score_col],
    )
    if not np.isfinite(observed):
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    null_values = np.empty(n_permutations, dtype=float)
    score = frame[score_col].values.copy()

    for index in range(n_permutations):
        permuted = rng.permutation(score)
        null_values[index] = concordance_index(
            frame[TIME_COL].values,
            -permuted,
            frame[EVENT_COL].values,
        )

    one_sided = (
        1.0 + np.sum(null_values >= observed)
    ) / (n_permutations + 1.0)
    two_sided = (
        1.0
        + np.sum(np.abs(null_values - 0.5) >= abs(observed - 0.5))
    ) / (n_permutations + 1.0)
    return float(observed), float(one_sided), float(two_sided)


def bootstrap_probability_coef_positive(
    data: pd.DataFrame,
    score_col: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float, int]:
    frame = data[[TIME_COL, EVENT_COL, score_col]].dropna().copy()
    event_idx = np.where(frame[EVENT_COL].values == 1)[0]
    censor_idx = np.where(frame[EVENT_COL].values == 0)[0]
    rng = np.random.default_rng(seed)
    coefs: list[float] = []

    for _ in range(n_bootstrap):
        sampled_event = rng.choice(event_idx, size=len(event_idx), replace=True)
        sampled_censor = rng.choice(censor_idx, size=len(censor_idx), replace=True)
        sampled = np.concatenate([sampled_event, sampled_censor])
        rng.shuffle(sampled)
        part = frame.iloc[sampled].copy()
        fit = fit_cox(part, score_col)
        coef = fit["coef"]
        if np.isfinite(coef):
            coefs.append(float(coef))

    if not coefs:
        return np.nan, np.nan, np.nan, 0
    coef_values = np.asarray(coefs, dtype=float)
    return (
        float(np.exp(np.quantile(coef_values, 0.025))),
        float(np.exp(np.quantile(coef_values, 0.975))),
        float(np.mean(coef_values > 0)),
        len(coef_values),
    )


def create_analysis_frames(
    clinical: pd.DataFrame,
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = clinical.join(scores.drop(columns=["cohort"], errors="ignore"), how="inner")
    frame[TIME_COL] = pd.to_numeric(frame[TIME_COL], errors="coerce")
    frame[EVENT_COL] = pd.to_numeric(frame[EVENT_COL], errors="coerce")

    zero_or_negative = frame[
        frame[TIME_COL].notna() & frame[TIME_COL].le(0)
    ].copy()

    primary = frame[
        frame[TIME_COL].gt(0) & frame[EVENT_COL].notna()
    ].copy()

    epsilon = frame[
        frame[TIME_COL].notna() & frame[EVENT_COL].notna()
    ].copy()
    epsilon.loc[epsilon[TIME_COL].le(0), TIME_COL] = EPSILON_TIME_MONTHS

    return primary, epsilon, zero_or_negative


def prepare_covariates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "age_years" in out.columns:
        out["age_z"] = zscore_series(out["age_years"])

    if "sex" in out.columns:
        normalized = out["sex"].astype(str).str.strip().str.lower()
        if normalized[normalized.ne("")].nunique() >= 2:
            out["sex_male"] = normalized.map(
                {
                    "male": 1.0,
                    "m": 1.0,
                    "female": 0.0,
                    "f": 0.0,
                }
            )

    if "percent_necrosis_numeric" in out.columns:
        out["percent_necrosis_z"] = zscore_series(
            out["percent_necrosis_numeric"]
        )

    if "good_necrosis_response_ge90" in out.columns:
        out["good_necrosis_response_ge90"] = pd.to_numeric(
            out["good_necrosis_response_ge90"],
            errors="coerce",
        )

    return out


def primary_validation(
    primary_frame: pd.DataFrame,
    epsilon_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for module_index, module in enumerate(PRIMARY_MODULES):
        score_col = f"{module}{PRIMARY_SCORE_SUFFIX}"
        if score_col not in primary_frame.columns:
            rows.append(
                {
                    "module_label": module,
                    "error": f"missing_primary_score:{score_col}",
                }
            )
            continue

        fit = fit_cox(primary_frame, score_col)
        c_obs, c_low, c_high, c_valid = stratified_bootstrap_c_index(
            primary_frame,
            score_col,
            N_BOOTSTRAP,
            RANDOM_SEED + module_index * 100,
        )
        _, perm_one, perm_two = permutation_c_index_p(
            primary_frame,
            score_col,
            N_PERMUTATIONS,
            RANDOM_SEED + module_index * 1000,
        )
        boot_hr_low, boot_hr_high, boot_prob, boot_valid = (
            bootstrap_probability_coef_positive(
                primary_frame,
                score_col,
                N_BOOTSTRAP,
                RANDOM_SEED + module_index * 10000,
            )
        )
        epsilon_fit = fit_cox(epsilon_frame, score_col)

        rows.append(
            {
                "module_label": module,
                "score_column": score_col,
                "n": fit["n"],
                "events": fit["events"],
                "hr_per_sd": fit["hr_per_sd"],
                "ci_low": fit["ci_low"],
                "ci_high": fit["ci_high"],
                "primary_p": fit["p"],
                "fixed_score_c_index": c_obs,
                "fixed_score_c_index_ci_low": c_low,
                "fixed_score_c_index_ci_high": c_high,
                "c_index_bootstrap_valid": c_valid,
                "ph_test_p": fit["ph_test_p"],
                "bootstrap_hr_ci_low": boot_hr_low,
                "bootstrap_hr_ci_high": boot_hr_high,
                "bootstrap_probability_coef_positive": boot_prob,
                "hr_bootstrap_valid": boot_valid,
                "permutation_c_index_p_one_sided": perm_one,
                "permutation_c_index_p_two_sided": perm_two,
                "epsilon_zero_time_hr_per_sd": epsilon_fit["hr_per_sd"],
                "epsilon_zero_time_ci_low": epsilon_fit["ci_low"],
                "epsilon_zero_time_ci_high": epsilon_fit["ci_high"],
                "epsilon_zero_time_p": epsilon_fit["p"],
                "error": fit["error"],
            }
        )

    results = pd.DataFrame(rows)
    results["q_within_gse39055"] = bh_adjust(results["primary_p"])
    results["permutation_c_index_q_bh"] = bh_adjust(
        results["permutation_c_index_p_two_sided"]
    )

    def classify(row: pd.Series) -> str:
        if not np.isfinite(row.get("hr_per_sd", np.nan)):
            return "not_estimable"
        if row["q_within_gse39055"] < 0.05 and row["hr_per_sd"] > 1:
            return "fdr_directional_rfs_support"
        if row["primary_p"] < 0.05 and row["hr_per_sd"] > 1:
            return "nominal_directional_rfs_support"
        if row["hr_per_sd"] > 1 and row["fixed_score_c_index"] > 0.55:
            return "directionally_supportive_without_nominal_significance"
        if row["hr_per_sd"] < 1:
            return "direction_discordant"
        return "no_support"

    results["gse39055_support_class"] = results.apply(classify, axis=1)
    return results


def adjusted_validation(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_covariates(frame)
    rows = []

    adjustment_sets = {
        "unadjusted": [],
        "age": ["age_z"],
        "age_plus_sex": ["age_z", "sex_male"],
        "proliferation": ["strict_human_meta_proliferation_pc1_z"],
        "age_plus_proliferation": [
            "age_z",
            "strict_human_meta_proliferation_pc1_z",
        ],
        "post_treatment_necrosis_response": [
            "good_necrosis_response_ge90"
        ],
        "age_plus_post_treatment_necrosis_response": [
            "age_z",
            "good_necrosis_response_ge90",
        ],
    }

    for module in PRIMARY_MODULES:
        score_col = f"{module}{PRIMARY_SCORE_SUFFIX}"
        if score_col not in prepared.columns:
            continue

        for label, covariates in adjustment_sets.items():
            usable = [cov for cov in covariates if cov in prepared.columns]
            if len(usable) != len(covariates):
                continue
            fit = fit_cox(prepared, score_col, covariates=usable)
            rows.append(
                {
                    "module_label": module,
                    "score_variant": "strict_signed_mean",
                    "adjustment": label,
                    "covariates": ";".join(usable),
                    **fit,
                    "interpretation_note": (
                        "Percent necrosis is post-treatment and is included only "
                        "as a mechanistic/sensitivity adjustment, not as a primary "
                        "baseline confounder."
                        if "necrosis" in label
                        else ""
                    ),
                }
            )

        if module == "M40":
            residual_candidates = [
                column
                for column in prepared.columns
                if column.startswith(
                    "M40__strict__signed_mean"
                )
                and "residual_to_disjoint_proliferation" in column
            ]
            for residual_col in residual_candidates:
                for label, covariates in {
                    "m40_residual_unadjusted": [],
                    "m40_residual_age": ["age_z"],
                }.items():
                    usable = [
                        cov for cov in covariates if cov in prepared.columns
                    ]
                    if len(usable) != len(covariates):
                        continue
                    fit = fit_cox(
                        prepared,
                        residual_col,
                        covariates=usable,
                    )
                    rows.append(
                        {
                            "module_label": module,
                            "score_variant": residual_col,
                            "adjustment": label,
                            "covariates": ";".join(usable),
                            **fit,
                            "interpretation_note": (
                                "Residualized M40 is a mechanistic sensitivity "
                                "score; PH violations should preclude a simple "
                                "constant-HR interpretation."
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def score_variant_sensitivity(
    frame: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    variant_suffixes = [
        ("strict_signed_mean", "__strict__signed_mean_z"),
        ("strict_canine_pca_weighted", "__strict__canine_pca_weighted_z"),
        ("strict_human_pc1", "__strict__human_pc1_z"),
        ("broad_signed_mean", "__broad__signed_mean_z"),
        ("broad_canine_pca_weighted", "__broad__canine_pca_weighted_z"),
        ("broad_human_pc1", "__broad__human_pc1_z"),
    ]

    for module in PRIMARY_MODULES + SECONDARY_MODULES:
        for variant_label, suffix in variant_suffixes:
            score_col = f"{module}{suffix}"
            if score_col not in frame.columns:
                continue
            fit = fit_cox(frame, score_col)
            rows.append(
                {
                    "module_label": module,
                    "variant": variant_label,
                    "score_column": score_col,
                    **fit,
                }
            )
    return pd.DataFrame(rows)


def leave_one_out_stability(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for module in PRIMARY_MODULES:
        score_col = f"{module}{PRIMARY_SCORE_SUFFIX}"
        if score_col not in frame.columns:
            continue
        values = []
        for sample_id in frame.index:
            part = frame.drop(index=sample_id)
            fit = fit_cox(part, score_col)
            values.append(
                {
                    "sample_removed": sample_id,
                    "hr": fit["hr_per_sd"],
                    "p": fit["p"],
                    "c_index": safe_fixed_direction_c_index(
                        part[TIME_COL],
                        part[EVENT_COL],
                        part[score_col],
                    ),
                }
            )
        table = pd.DataFrame(values)
        rows.append(
            {
                "module_label": module,
                "n_loo_fits": table.shape[0],
                "hr_min": table["hr"].min(),
                "hr_max": table["hr"].max(),
                "hr_median": table["hr"].median(),
                "fraction_hr_above_1": float((table["hr"] > 1).mean()),
                "c_index_min": table["c_index"].min(),
                "c_index_max": table["c_index"].max(),
                "c_index_median": table["c_index"].median(),
                "fraction_p_below_0_05": float(
                    (table["p"] < 0.05).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def expression_bins(expression: pd.DataFrame) -> pd.DataFrame:
    stats = pd.DataFrame(
        {
            "mean": expression.mean(axis=0),
            "sd": expression.std(axis=0),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    stats = stats[stats["sd"] > 0].copy()
    stats["mean_bin"] = pd.qcut(
        stats["mean"],
        q=10,
        labels=False,
        duplicates="drop",
    )
    stats["sd_bin"] = pd.qcut(
        stats["sd"],
        q=10,
        labels=False,
        duplicates="drop",
    )
    stats["bin_key"] = (
        stats["mean_bin"].astype(str) + "_" + stats["sd_bin"].astype(str)
    )
    return stats


def draw_expression_matched_gene_set(
    target_genes: list[str],
    stats: pd.DataFrame,
    excluded: set[str],
    rng: np.random.Generator,
) -> list[str]:
    available_targets = [gene for gene in target_genes if gene in stats.index]
    if not available_targets:
        return []

    target_bins = stats.loc[available_targets, "bin_key"].value_counts()
    selected: list[str] = []
    used = set(excluded)

    for bin_key, count in target_bins.items():
        pool = [
            gene
            for gene in stats.index[stats["bin_key"].eq(bin_key)]
            if gene not in used
        ]
        take = min(int(count), len(pool))
        if take > 0:
            chosen = rng.choice(pool, size=take, replace=False).tolist()
            selected.extend(chosen)
            used.update(chosen)

    needed = len(available_targets) - len(selected)
    if needed > 0:
        fallback = [gene for gene in stats.index if gene not in used]
        if len(fallback) < needed:
            return []
        chosen = rng.choice(fallback, size=needed, replace=False).tolist()
        selected.extend(chosen)

    return selected


def random_gene_set_controls(
    frame: pd.DataFrame,
    expression: pd.DataFrame,
    strict_weights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expression = expression.copy()
    expression.columns = expression.columns.astype(str).str.upper()
    expression = expression.loc[frame.index]
    expression = expression.loc[:, ~expression.columns.duplicated()]
    expression = expression.apply(pd.to_numeric, errors="coerce")
    expression = expression.fillna(expression.median(axis=0))
    expression = expression.loc[:, expression.std(axis=0) > 0]
    z = (expression - expression.mean(axis=0)) / expression.std(axis=0)

    stats = expression_bins(expression)
    rng = np.random.default_rng(RANDOM_SEED)
    summary_rows = []
    distribution_rows = []

    for module_index, module in enumerate(PRIMARY_MODULES):
        score_col = f"{module}{PRIMARY_SCORE_SUFFIX}"
        if score_col not in frame.columns:
            continue

        weights = strict_weights[
            strict_weights["module_label"].eq(module)
        ].copy()
        weights["human_gene_symbol"] = (
            weights["human_gene_symbol"].astype(str).str.upper()
        )
        weights = weights.drop_duplicates("human_gene_symbol", keep="first")
        target_genes = [
            gene
            for gene in weights["human_gene_symbol"].tolist()
            if gene in z.columns
        ]
        loadings = (
            weights.set_index("human_gene_symbol")
            .reindex(target_genes)["risk_oriented_loading"]
            .fillna(0.0)
        )
        signs = np.sign(loadings.values)
        signs[signs == 0] = 1

        observed = safe_fixed_direction_c_index(
            frame[TIME_COL],
            frame[EVENT_COL],
            frame[score_col],
        )
        excluded = set(target_genes)
        random_values: list[float] = []

        for repeat in range(1, N_RANDOM_SETS + 1):
            genes = draw_expression_matched_gene_set(
                target_genes=target_genes,
                stats=stats,
                excluded=excluded,
                rng=rng,
            )
            if len(genes) != len(target_genes):
                continue
            permuted_signs = rng.permutation(signs)
            random_score = z[genes].mul(permuted_signs, axis=1).mean(axis=1)
            value = safe_fixed_direction_c_index(
                frame[TIME_COL],
                frame[EVENT_COL],
                random_score,
            )
            if np.isfinite(value):
                random_values.append(value)
                distribution_rows.append(
                    {
                        "module_label": module,
                        "repeat": repeat,
                        "random_c_index": value,
                        "n_genes": len(genes),
                    }
                )

        random_array = np.asarray(random_values, dtype=float)
        empirical_p = (
            (1.0 + np.sum(random_array >= observed))
            / (len(random_array) + 1.0)
            if len(random_array)
            else np.nan
        )
        summary_rows.append(
            {
                "cohort": "GSE39055",
                "endpoint": "recurrence_free_survival",
                "module_label": module,
                "observed_c_index": observed,
                "n_module_genes_available": len(target_genes),
                "n_random_valid": len(random_array),
                "random_mean": (
                    float(np.mean(random_array)) if len(random_array) else np.nan
                ),
                "random_median": (
                    float(np.median(random_array)) if len(random_array) else np.nan
                ),
                "random_q95": (
                    float(np.quantile(random_array, 0.95))
                    if len(random_array)
                    else np.nan
                ),
                "observed_percentile": (
                    float(np.mean(random_array <= observed))
                    if len(random_array)
                    else np.nan
                ),
                "empirical_p_greater_equal": empirical_p,
            }
        )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["empirical_q_bh"] = bh_adjust(
            summary["empirical_p_greater_equal"]
        )
    return summary, pd.DataFrame(distribution_rows)


def create_cross_cohort_synthesis(
    primary: pd.DataFrame,
    random_summary: pd.DataFrame,
) -> pd.DataFrame:
    current = primary[
        [
            "module_label",
            "hr_per_sd",
            "ci_low",
            "ci_high",
            "primary_p",
            "q_within_gse39055",
            "fixed_score_c_index",
            "ph_test_p",
            "gse39055_support_class",
        ]
    ].copy()
    current = current.rename(
        columns={
            "hr_per_sd": "gse39055_rfs_hr_per_sd",
            "ci_low": "gse39055_rfs_ci_low",
            "ci_high": "gse39055_rfs_ci_high",
            "primary_p": "gse39055_rfs_p",
            "q_within_gse39055": "gse39055_rfs_q",
            "fixed_score_c_index": "gse39055_rfs_c_index",
            "ph_test_p": "gse39055_rfs_ph_p",
        }
    )

    if not random_summary.empty:
        current = current.merge(
            random_summary[
                [
                    "module_label",
                    "observed_percentile",
                    "empirical_p_greater_equal",
                    "empirical_q_bh",
                ]
            ].rename(
                columns={
                    "observed_percentile": "gse39055_random_percentile",
                    "empirical_p_greater_equal": "gse39055_random_empirical_p",
                    "empirical_q_bh": "gse39055_random_empirical_q",
                }
            ),
            on="module_label",
            how="left",
        )

    if PREVIOUS_ROBUST_SUMMARY_FILE.exists():
        previous = pd.read_csv(PREVIOUS_ROBUST_SUMMARY_FILE)
        synthesis = previous.merge(current, on="module_label", how="outer")
    else:
        synthesis = current

    def grade(row: pd.Series) -> str:
        module = row.get("module_label", "")
        current_hr = row.get("gse39055_rfs_hr_per_sd", np.nan)
        current_q = row.get("gse39055_rfs_q", np.nan)
        target_hr = row.get("target_hr_per_sd", np.nan)
        gse_auc = row.get("gse_met_auc", np.nan)

        three_directional = (
            np.isfinite(current_hr)
            and current_hr > 1
            and np.isfinite(target_hr)
            and target_hr > 1
            and np.isfinite(gse_auc)
            and gse_auc > 0.5
        )
        if module == "M34" and three_directional and current_q < 0.10:
            return "strong_three_setting_cross_species_support"
        if three_directional and current_q < 0.10:
            return "three_setting_directional_support"
        if three_directional:
            return "three_setting_directional_but_not_confirmatory"
        if np.isfinite(current_hr) and current_hr < 1:
            return "third_cohort_direction_discordance"
        return "third_cohort_no_clear_support"

    synthesis["three_cohort_evidence_grade"] = synthesis.apply(grade, axis=1)
    return synthesis


def create_manifest(
    preparation_manifest: dict[str, Any],
    outputs: list[Path],
) -> None:
    payload = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "preparation_manifest_sha256": sha256_file(PREPARATION_MANIFEST_FILE),
        "preparation_script_version": preparation_manifest.get("script_version"),
        "analysis_design": {
            "primary_modules": PRIMARY_MODULES,
            "primary_score": "strict one-to-one signed-mean z-score",
            "primary_endpoint": "recurrence-free survival",
            "primary_time_rule": "exclude nonpositive recorded times",
            "zero_time_sensitivity": (
                f"replace nonpositive time with {EPSILON_TIME_MONTHS:.6f} months"
            ),
            "bootstrap_repetitions": N_BOOTSTRAP,
            "permutation_repetitions": N_PERMUTATIONS,
            "random_gene_set_repetitions": N_RANDOM_SETS,
            "multiplicity": "BH across the four frozen primary modules",
        },
        "files": {},
    }

    for path in outputs:
        if path.exists():
            payload["files"][path.name] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }

    OUTPUT_MANIFEST.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def write_readme() -> None:
    text = f"""GSE39055 frozen-program recurrence-free-survival validation
Script version: {SCRIPT_VERSION}

Primary analysis
----------------
- Four frozen primary canine programs: M34, M11, M24, M40
- Strict one-to-one signed-mean human score
- Recurrence-free survival using the GEO recurrence/follow-up time and recurrence event
- Cox HR per SD, fixed-direction C-index, BH FDR across four modules

Zero-time handling
------------------
The primary Cox analysis excludes nonpositive recorded times.
A prespecified sensitivity analysis replaces zero time with one day
({EPSILON_TIME_MONTHS:.6f} months).

Adjustment hierarchy
--------------------
Age and sex are baseline sensitivity covariates.
Human proliferation PC1 is a mechanistic sensitivity adjustment.
Percent necrosis is post-treatment and is not treated as a primary baseline confounder.

Interpretation
--------------
GSE39055 is a small third human cohort. Bootstrap, leave-one-out,
score-variant, and expression-matched random-panel analyses are robustness diagnostics.
No result may be used to change frozen module membership, weights, direction, or tier.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("GSE39055 external RFS validation of frozen canine programs")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Results directory: {RESULTS_DIR}")
    print("")
    print("Design:")
    print("  Preserve frozen canine module definitions and score direction.")
    print("  Test four strict one-to-one signed-mean scores for recurrence-free survival.")
    print("  Apply BH multiplicity control across four primary modules.")
    print("  Audit zero-time handling, baseline/proliferation adjustment, PH, leave-one-out stability, and matched random panels.")
    print("")

    preparation_manifest = verify_preparation_inputs()

    expression = read_required_csv(EXPRESSION_FILE, index_col=0)
    clinical = read_required_csv(CLINICAL_FILE, index_col=0)
    scores = read_required_csv(SCORES_FILE, index_col=0)
    coverage = read_required_csv(COVERAGE_FILE)
    frozen_programs = read_required_csv(FROZEN_PROGRAM_MANIFEST_FILE)
    strict_weights = read_required_csv(STRICT_WEIGHTS_FILE)

    common = clinical.index.intersection(scores.index).intersection(expression.index)
    clinical = clinical.loc[common].copy()
    scores = scores.loc[common].copy()
    expression = expression.loc[common].copy()

    primary_frame, epsilon_frame, zero_time_audit = create_analysis_frames(
        clinical,
        scores,
    )
    zero_time_audit.to_csv(OUTPUT_ZERO_TIME_AUDIT)

    print("")
    print("Matched GSE39055 validation data:")
    print(f"  Expression: {expression.shape}")
    print(f"  Clinical: {clinical.shape}")
    print(f"  Scores: {scores.shape}")
    print(f"  Complete RFS with positive time: {primary_frame.shape[0]}")
    print(f"  Recurrence events in primary frame: {int(primary_frame[EVENT_COL].sum())}")
    print(f"  Nonpositive-time records excluded from primary: {zero_time_audit.shape[0]}")

    primary = primary_validation(primary_frame, epsilon_frame)
    primary.to_csv(OUTPUT_PRIMARY, index=False)
    primary[
        [
            "module_label",
            "primary_p",
            "q_within_gse39055",
            "permutation_c_index_p_two_sided",
            "permutation_c_index_q_bh",
        ]
    ].to_csv(OUTPUT_MULTIPLICITY, index=False)

    adjusted = adjusted_validation(primary_frame)
    adjusted.to_csv(OUTPUT_ADJUSTED, index=False)

    variants = score_variant_sensitivity(primary_frame, coverage)
    variants.to_csv(OUTPUT_VARIANTS, index=False)

    loo = leave_one_out_stability(primary_frame)
    loo.to_csv(OUTPUT_LOO, index=False)

    random_summary, random_distribution = random_gene_set_controls(
        primary_frame,
        expression,
        strict_weights,
    )
    random_summary.to_csv(OUTPUT_RANDOM, index=False)
    random_distribution.to_csv(OUTPUT_RANDOM_DISTRIBUTION, index=False)

    synthesis = create_cross_cohort_synthesis(primary, random_summary)
    synthesis.to_csv(OUTPUT_CROSS_COHORT, index=False)

    write_readme()
    create_manifest(
        preparation_manifest,
        [
            OUTPUT_PRIMARY,
            OUTPUT_MULTIPLICITY,
            OUTPUT_ADJUSTED,
            OUTPUT_VARIANTS,
            OUTPUT_LOO,
            OUTPUT_RANDOM,
            OUTPUT_RANDOM_DISTRIBUTION,
            OUTPUT_ZERO_TIME_AUDIT,
            OUTPUT_CROSS_COHORT,
            OUTPUT_README,
        ],
    )

    print("")
    print("=" * 80)
    print("GSE39055 primary frozen-program RFS validation")
    print("=" * 80)
    display_cols = [
        "module_label",
        "n",
        "events",
        "hr_per_sd",
        "ci_low",
        "ci_high",
        "primary_p",
        "q_within_gse39055",
        "fixed_score_c_index",
        "fixed_score_c_index_ci_low",
        "fixed_score_c_index_ci_high",
        "permutation_c_index_q_bh",
        "ph_test_p",
        "gse39055_support_class",
    ]
    print(primary[display_cols].to_string(index=False))

    print("")
    print("=" * 80)
    print("Zero-time sensitivity")
    print("=" * 80)
    print(
        primary[
            [
                "module_label",
                "hr_per_sd",
                "primary_p",
                "epsilon_zero_time_hr_per_sd",
                "epsilon_zero_time_p",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("GSE39055 adjustment robustness")
    print("=" * 80)
    adjusted_display = adjusted[
        adjusted["module_label"].isin(PRIMARY_MODULES)
    ].copy()
    print(
        adjusted_display[
            [
                "module_label",
                "score_variant",
                "adjustment",
                "n",
                "events",
                "hr_per_sd",
                "ci_low",
                "ci_high",
                "p",
                "c_index",
                "ph_test_p",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("GSE39055 leave-one-out stability")
    print("=" * 80)
    print(loo.to_string(index=False))

    print("")
    print("=" * 80)
    print("GSE39055 expression-matched random controls")
    print("=" * 80)
    print(random_summary.to_string(index=False))

    print("")
    print("=" * 80)
    print("Three-cohort external evidence synthesis")
    print("=" * 80)
    synthesis_cols = [
        "module_label",
        "target_hr_per_sd",
        "gse_met_auc",
        "gse39055_rfs_hr_per_sd",
        "gse39055_rfs_q",
        "gse39055_rfs_c_index",
        "gse39055_random_empirical_p",
        "three_cohort_evidence_grade",
    ]
    synthesis_cols = [
        column for column in synthesis_cols if column in synthesis.columns
    ]
    print(synthesis[synthesis_cols].to_string(index=False))

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("The strict one-to-one signed-mean score and canine risk direction remain frozen.")
    print("Primary Cox analyses exclude nonpositive recorded follow-up times; one-day replacement is only a sensitivity analysis.")
    print("Age and sex are baseline sensitivity covariates; percent necrosis is post-treatment and cannot be treated as a primary baseline confounder.")
    print("GSE39055 contains only 37 samples and 18 recurrences before zero-time exclusion; external effect sizes require cautious interpretation.")
    print("No GSE39055 result may alter module membership, score weights, risk direction, or validation tier.")

    print("")
    print("Saved:")
    for path in [
        OUTPUT_PRIMARY,
        OUTPUT_MULTIPLICITY,
        OUTPUT_ADJUSTED,
        OUTPUT_VARIANTS,
        OUTPUT_LOO,
        OUTPUT_RANDOM,
        OUTPUT_RANDOM_DISTRIBUTION,
        OUTPUT_ZERO_TIME_AUDIT,
        OUTPUT_CROSS_COHORT,
        OUTPUT_README,
        OUTPUT_MANIFEST,
    ]:
        print(path)
    print("Done.")


if __name__ == "__main__":
    main()
