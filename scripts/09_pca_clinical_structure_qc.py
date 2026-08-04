from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_N_VARIABLE_GENES = 5000
N_PCS = 10
MIN_GROUP_SIZE = 5
COX_PENALIZER = 0.05

EXPRESSION_FILE = f"GSE238110_DOG2_expression_log2cpm_matched_top{TOP_N_VARIABLE_GENES}var.csv"
CLINICAL_FILE = "GSE238110_DOG2_clinical_matched_indexed.csv"

CONTINUOUS_VARS = [
    "age",
    "weight",
]

CATEGORICAL_VARS = [
    "gender",
    "PH",
    "ALP",
    "Group",
    "Site",
    "treatment",
    "primary_immune_subtype",
]

ENDPOINTS = {
    "DFI": {
        "time_col": "dfi_time",
        "event_col": "dfi_event",
    },
    "OS": {
        "time_col": "os_time",
        "event_col": "os_event",
    },
}

TRANSCRIPTOMICS_DERIVED_VARS = {
    "primary_immune_subtype",
}


def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)

    valid = np.isfinite(p)
    pv = p[valid]
    n = len(pv)

    if n == 0:
        return q

    order = np.argsort(pv)
    ranked = pv[order]

    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)

    out = np.empty(n)
    out[order] = adjusted
    q[valid] = out

    return q


def load_data():
    expression_path = PROCESSED_DIR / EXPRESSION_FILE
    clinical_path = PROCESSED_DIR / CLINICAL_FILE

    expression = pd.read_csv(expression_path, index_col=0)
    clinical = pd.read_csv(clinical_path, index_col=0)

    common_samples = clinical.index.intersection(expression.index)
    expression = expression.loc[common_samples].copy()
    clinical = clinical.loc[common_samples].copy()

    return expression, clinical


def run_pca(expression):
    x = expression.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(axis=0))

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    pca = PCA(n_components=N_PCS, random_state=42)
    scores = pca.fit_transform(x_scaled)

    pc_cols = [f"PC{i}" for i in range(1, N_PCS + 1)]
    pca_scores = pd.DataFrame(scores, index=expression.index, columns=pc_cols)

    explained = pd.DataFrame({
        "pc": pc_cols,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
    })

    return pca_scores, explained


def test_continuous_pc_association(pca_scores, clinical, variable):
    rows = []

    values = pd.to_numeric(clinical[variable], errors="coerce")

    for pc in pca_scores.columns:
        use = pd.DataFrame({
            "pc": pca_scores[pc],
            "value": values,
        }).dropna()

        if use.shape[0] < 30 or use["value"].nunique() < 3:
            continue

        rho, p_value = stats.spearmanr(use["pc"], use["value"])

        rows.append({
            "variable": variable,
            "variable_type": "continuous",
            "pc": pc,
            "n": use.shape[0],
            "effect_size": float(rho ** 2),
            "statistic": float(rho),
            "p": float(p_value),
            "test": "spearman_rho_squared",
            "is_transcriptomics_derived": variable in TRANSCRIPTOMICS_DERIVED_VARS,
        })

    return rows


def test_categorical_pc_association(pca_scores, clinical, variable):
    rows = []

    values = clinical[variable].astype("object")

    for pc in pca_scores.columns:
        use = pd.DataFrame({
            "pc": pca_scores[pc],
            "group": values,
        }).dropna()

        group_counts = use["group"].value_counts()
        keep = group_counts[group_counts >= MIN_GROUP_SIZE].index
        use = use[use["group"].isin(keep)]

        if use["group"].nunique() < 2:
            continue

        groups = [
            part["pc"].values
            for _, part in use.groupby("group")
        ]

        try:
            f_stat, p_value = stats.f_oneway(*groups)
        except Exception:
            continue

        grand_mean = use["pc"].mean()
        ss_total = ((use["pc"] - grand_mean) ** 2).sum()

        ss_between = 0.0
        for _, part in use.groupby("group"):
            ss_between += part.shape[0] * (part["pc"].mean() - grand_mean) ** 2

        eta_squared = ss_between / ss_total if ss_total > 0 else np.nan

        rows.append({
            "variable": variable,
            "variable_type": "categorical",
            "pc": pc,
            "n": use.shape[0],
            "n_groups": use["group"].nunique(),
            "effect_size": float(eta_squared),
            "statistic": float(f_stat),
            "p": float(p_value),
            "test": "anova_eta_squared",
            "is_transcriptomics_derived": variable in TRANSCRIPTOMICS_DERIVED_VARS,
        })

    return rows


def fit_univariate_cox(df, time_col, event_col, covariates):
    use = df[[time_col, event_col] + covariates].dropna().copy()

    if use.shape[0] < 30:
        return None, "too_few_samples"

    if use[event_col].sum() < 5:
        return None, "too_few_events"

    cph = CoxPHFitter(penalizer=COX_PENALIZER)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            cph.fit(use, duration_col=time_col, event_col=event_col)

        return cph, ""

    except Exception as e:
        return None, str(e)


def test_pc_survival_associations(pca_scores, clinical):
    rows = []

    for endpoint_name, endpoint in ENDPOINTS.items():
        time_col = endpoint["time_col"]
        event_col = endpoint["event_col"]

        clinical[time_col] = pd.to_numeric(clinical[time_col], errors="coerce")
        clinical[event_col] = pd.to_numeric(clinical[event_col], errors="coerce")

        for pc in pca_scores.columns:
            df = clinical[[time_col, event_col]].join(pca_scores[[pc]], how="inner")
            df[pc] = pd.to_numeric(df[pc], errors="coerce")

            cph, error = fit_univariate_cox(df, time_col, event_col, [pc])

            if cph is None:
                rows.append({
                    "endpoint": endpoint_name,
                    "pc": pc,
                    "n": df.dropna().shape[0],
                    "events": np.nan,
                    "coef": np.nan,
                    "hr": np.nan,
                    "p": np.nan,
                    "c_index": np.nan,
                    "error": error,
                })
                continue

            summary = cph.summary.loc[pc]

            rows.append({
                "endpoint": endpoint_name,
                "pc": pc,
                "n": df[[time_col, event_col, pc]].dropna().shape[0],
                "events": int(df[event_col].sum()),
                "coef": float(summary["coef"]),
                "hr": float(summary["exp(coef)"]),
                "p": float(summary["p"]),
                "c_index": float(cph.concordance_index_),
                "error": "",
            })

    out = pd.DataFrame(rows)
    out["q"] = bh_fdr(out["p"].values)

    return out


def prepare_clinical_variable_for_cox(clinical, variable):
    if variable in CONTINUOUS_VARS:
        values = pd.to_numeric(clinical[variable], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)

        if values.dropna().nunique() < 3:
            return None

        z = (values - values.mean()) / values.std()

        return pd.DataFrame({variable: z}, index=clinical.index)

    values = clinical[variable].astype("object")
    counts = values.value_counts(dropna=True)
    keep = counts[counts >= MIN_GROUP_SIZE].index
    values = values.where(values.isin(keep), np.nan)

    if values.dropna().nunique() < 2:
        return None

    dummies = pd.get_dummies(values, prefix=variable, drop_first=True)
    dummies.index = clinical.index

    return dummies


def test_clinical_survival_associations(clinical):
    rows = []

    variables = CONTINUOUS_VARS + CATEGORICAL_VARS

    for endpoint_name, endpoint in ENDPOINTS.items():
        time_col = endpoint["time_col"]
        event_col = endpoint["event_col"]

        clinical[time_col] = pd.to_numeric(clinical[time_col], errors="coerce")
        clinical[event_col] = pd.to_numeric(clinical[event_col], errors="coerce")

        for variable in variables:
            if variable not in clinical.columns:
                continue

            covariates = prepare_clinical_variable_for_cox(clinical, variable)

            if covariates is None or covariates.shape[1] == 0:
                continue

            df = clinical[[time_col, event_col]].join(covariates, how="inner")
            cph, error = fit_univariate_cox(df, time_col, event_col, list(covariates.columns))

            if cph is None:
                rows.append({
                    "endpoint": endpoint_name,
                    "variable": variable,
                    "n": df.dropna().shape[0],
                    "events": np.nan,
                    "n_terms": covariates.shape[1],
                    "min_p": np.nan,
                    "best_term": "",
                    "c_index": np.nan,
                    "error": error,
                    "is_transcriptomics_derived": variable in TRANSCRIPTOMICS_DERIVED_VARS,
                })
                continue

            summary = cph.summary.copy()
            best_term = summary["p"].idxmin()
            min_p = float(summary.loc[best_term, "p"])

            rows.append({
                "endpoint": endpoint_name,
                "variable": variable,
                "n": df.dropna().shape[0],
                "events": int(df[event_col].sum()),
                "n_terms": covariates.shape[1],
                "min_p": min_p,
                "best_term": best_term,
                "c_index": float(cph.concordance_index_),
                "error": "",
                "is_transcriptomics_derived": variable in TRANSCRIPTOMICS_DERIVED_VARS,
            })

    out = pd.DataFrame(rows)
    out["q"] = bh_fdr(out["min_p"].values)

    return out


def build_confounder_flags(pc_assoc, clinical_survival):
    expr_assoc = pc_assoc.copy()

    expr_assoc = expr_assoc[
        (expr_assoc["q"] < 0.05) &
        (expr_assoc["effect_size"] >= 0.05)
    ].copy()

    expr_best = (
        expr_assoc
        .sort_values(["variable", "effect_size"], ascending=[True, False])
        .groupby("variable")
        .head(1)
        .copy()
    )

    survival_sig = clinical_survival[
        (clinical_survival["min_p"] < 0.05)
    ].copy()

    flags = expr_best.merge(
        survival_sig,
        on="variable",
        how="inner",
        suffixes=("_expression_structure", "_survival"),
    )

    if flags.empty:
        return flags

    flags["flag_reason"] = (
        "Variable is associated with expression structure and with survival endpoint."
    )

    flags["recommended_use"] = np.where(
        flags["is_transcriptomics_derived_expression_structure"],
        "biological benchmark only; do not use as ordinary clinical adjustment covariate",
        "candidate clinical adjustment covariate or sensitivity-analysis variable",
    )

    return flags


def main():
    print("=" * 80)
    print("PCA and clinical structure QC")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Processed directory: {PROCESSED_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Top variable genes: {TOP_N_VARIABLE_GENES}")
    print(f"Number of PCs: {N_PCS}")
    print("")

    expression, clinical = load_data()

    print("Input data:")
    print(f"  Expression matrix: {expression.shape}")
    print(f"  Clinical table: {clinical.shape}")
    print("")

    pca_scores, explained = run_pca(expression)

    pca_score_path = RESULTS_DIR / "GSE238110_pca_scores_top5000var.csv"
    explained_path = RESULTS_DIR / "GSE238110_pca_explained_variance_top5000var.csv"

    pca_scores.to_csv(pca_score_path)
    explained.to_csv(explained_path, index=False)

    print("PCA explained variance:")
    print(explained.to_string(index=False))
    print("")

    pc_assoc_rows = []

    for variable in CONTINUOUS_VARS:
        if variable in clinical.columns:
            pc_assoc_rows.extend(test_continuous_pc_association(pca_scores, clinical, variable))

    for variable in CATEGORICAL_VARS:
        if variable in clinical.columns:
            pc_assoc_rows.extend(test_categorical_pc_association(pca_scores, clinical, variable))

    pc_assoc = pd.DataFrame(pc_assoc_rows)

    if not pc_assoc.empty:
        pc_assoc["q"] = bh_fdr(pc_assoc["p"].values)
        pc_assoc = pc_assoc.sort_values(["q", "p", "effect_size"])

    pc_assoc_path = RESULTS_DIR / "GSE238110_pca_clinical_associations_top5000var.csv"
    pc_assoc.to_csv(pc_assoc_path, index=False)

    pc_survival = test_pc_survival_associations(pca_scores, clinical)
    pc_survival_path = RESULTS_DIR / "GSE238110_pca_survival_associations_top5000var.csv"
    pc_survival.to_csv(pc_survival_path, index=False)

    clinical_survival = test_clinical_survival_associations(clinical)
    clinical_survival_path = RESULTS_DIR / "GSE238110_clinical_survival_associations.csv"
    clinical_survival.to_csv(clinical_survival_path, index=False)

    flags = build_confounder_flags(pc_assoc, clinical_survival)
    flags_path = RESULTS_DIR / "GSE238110_potential_confounder_flags.csv"
    flags.to_csv(flags_path, index=False)

    print("=" * 80)
    print("Top PCA-clinical associations")
    print("=" * 80)

    if pc_assoc.empty:
        print("No PCA-clinical associations were tested.")
    else:
        cols = [
            "variable",
            "variable_type",
            "pc",
            "n",
            "n_groups",
            "effect_size",
            "p",
            "q",
            "is_transcriptomics_derived",
        ]
        cols = [c for c in cols if c in pc_assoc.columns]
        print(pc_assoc[cols].head(30).to_string(index=False))

    print("")
    print("=" * 80)
    print("PC survival associations")
    print("=" * 80)

    cols = [
        "endpoint",
        "pc",
        "n",
        "events",
        "coef",
        "hr",
        "p",
        "q",
        "c_index",
    ]
    print(pc_survival[cols].sort_values(["q", "p"]).head(30).to_string(index=False))

    print("")
    print("=" * 80)
    print("Clinical variable survival associations")
    print("=" * 80)

    cols = [
        "endpoint",
        "variable",
        "n",
        "events",
        "n_terms",
        "min_p",
        "q",
        "best_term",
        "c_index",
        "is_transcriptomics_derived",
    ]
    print(clinical_survival[cols].sort_values(["q", "min_p"]).to_string(index=False))

    print("")
    print("=" * 80)
    print("Potential confounder flags")
    print("=" * 80)

    if flags.empty:
        print("No variable met the current confounder-flag rule.")
        print("Flag rule: PCA association q<0.05 and effect_size>=0.05 plus survival p<0.05.")
    else:
        print(flags.to_string(index=False))

    print("")
    print("Saved:")
    print(pca_score_path)
    print(explained_path)
    print(pc_assoc_path)
    print(pc_survival_path)
    print(clinical_survival_path)
    print(flags_path)
    print("Done.")


if __name__ == "__main__":
    main()
