from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXPRESSION_FILE = "GSE238110_DOG2_expression_log2cpm_matched_allgenes.csv"
CLINICAL_FILE = "GSE238110_DOG2_clinical_matched_indexed.csv"
PROLIFERATION_GENE_FILE = "GSE238110_meta_proliferation_gene_set.csv"
PROLIFERATION_SCORE_FILE = "GSE238110_meta_proliferation_score_per_sample.csv"
MODULE_MEMBERSHIP_FILE = "GSE238110_RNA_module_gene_membership.csv"
MODULE_SCORE_FILE = "GSE238110_full_cohort_module_scores_for_proliferation_adjustment.csv"
MODULE_PRIORITY_FILE = "GSE238110_RNA_full_cohort_transferable_module_priority.csv"

N_SPLITS = 5
N_REPEATS = 20
RANDOM_SEED = 42
MIN_GENES_FOR_PCA = 3
MIN_DISJOINT_PROLIFERATION_GENES = 20
COX_PENALIZER = 0.05

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

WEIGHT_COLUMN_CANDIDATES = [
    "weight",
    "Weight",
    "body_weight",
    "body weight",
    "Body weight",
    "body_weight_kg",
]

PRIMARY_REVIEW_MODULES = ["M11", "M24", "M34", "M40", "M28", "M38"]

OUTPUT_OVERLAP_AUDIT = RESULTS_DIR / "GSE238110_module_proliferation_overlap_audit.csv"
OUTPUT_FULL_RESULTS = RESULTS_DIR / "GSE238110_leave_module_out_proliferation_full_cohort.csv"
OUTPUT_CV_FOLDS = RESULTS_DIR / "GSE238110_repeated_cv_proliferation_sensitivity_fold_results.csv"
OUTPUT_CV_SUMMARY = RESULTS_DIR / "GSE238110_repeated_cv_proliferation_sensitivity_summary.csv"
OUTPUT_DECISION_TABLE = RESULTS_DIR / "GSE238110_proliferation_independence_decision_table.csv"


def clean_gene_symbol(gene):
    gene = str(gene)
    tail = gene.rsplit("_", 1)[-1]
    return gene.rsplit("_", 1)[0] if tail.isdigit() else gene


def read_required_csv(path, index_col=None):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    print(f"Loaded: {path}")
    return pd.read_csv(path, index_col=index_col)


def read_optional_csv(path, index_col=None):
    if not path.exists():
        print(f"Optional file not found: {path}")
        return pd.DataFrame()
    print(f"Loaded: {path}")
    return pd.read_csv(path, index_col=index_col)


def bh_adjust(pvalues):
    pvalues = pd.Series(pvalues, dtype=float)
    qvalues = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = pvalues.notna() & np.isfinite(pvalues)
    if valid.sum() == 0:
        return qvalues

    p = pvalues[valid].values
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    q = np.empty(n)
    q[order] = adjusted
    qvalues.loc[pvalues[valid].index] = q
    return qvalues


def safe_corr(a, b):
    frame = pd.concat(
        [pd.Series(a, name="a"), pd.Series(b, name="b")],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.shape[0] < 5:
        return np.nan
    if frame["a"].std() == 0 or frame["b"].std() == 0:
        return np.nan
    return float(frame["a"].corr(frame["b"]))


def find_weight_column(clinical):
    for col in WEIGHT_COLUMN_CANDIDATES:
        if col in clinical.columns:
            return col
    lowered = {str(c).strip().lower(): c for c in clinical.columns}
    for candidate in ["weight", "body weight", "body_weight"]:
        if candidate in lowered:
            return lowered[candidate]
    return None


def find_best_overlap_column(table, valid_values, preferred=None):
    valid_values = set(map(str, valid_values))
    preferred = preferred or []

    for col in preferred:
        if col in table.columns:
            overlap = table[col].dropna().astype(str).isin(valid_values).sum()
            if overlap > 0:
                return col, int(overlap)

    best_col = None
    best_overlap = -1
    for col in table.columns:
        values = table[col].dropna().astype(str)
        overlap = values.isin(valid_values).sum()
        if overlap > best_overlap:
            best_col = col
            best_overlap = int(overlap)

    return best_col, best_overlap


def resolve_gene_values_to_expression(values, expression):
    expression_columns = list(expression.columns)
    exact_set = set(expression_columns)

    symbol_to_columns = {}
    for col in expression_columns:
        symbol = clean_gene_symbol(col).upper()
        symbol_to_columns.setdefault(symbol, []).append(col)

    resolved = []
    for value in pd.Series(values).dropna().astype(str):
        if value in exact_set:
            resolved.append(value)
            continue

        symbol = clean_gene_symbol(value).upper()
        candidates = symbol_to_columns.get(symbol, [])
        if not candidates:
            continue

        if len(candidates) == 1:
            resolved.append(candidates[0])
        else:
            variances = expression[candidates].var(axis=0).sort_values(ascending=False)
            resolved.append(variances.index[0])

    return list(dict.fromkeys(resolved))


def detect_module_columns(module_membership, expression):
    module_candidates = ["module_label", "module", "module_id", "cluster", "cluster_id"]
    module_col = next((c for c in module_candidates if c in module_membership.columns), None)
    if module_col is None:
        raise ValueError(
            "Could not identify a module-label column in the module membership table."
        )

    gene_col, overlap = find_best_overlap_column(
        table=module_membership,
        valid_values=expression.columns,
        preferred=["gene", "gene_id", "expression_gene", "gene_column"],
    )
    if gene_col is None:
        raise ValueError(
            "Could not identify a gene column in the module membership table."
        )

    return module_col, gene_col, overlap


def filter_full_cohort_membership(module_membership):
    scope_candidates = [
        "analysis_scope",
        "scope",
        "analysis_type",
        "source",
        "data_scope",
    ]

    for col in scope_candidates:
        if col not in module_membership.columns:
            continue
        values = module_membership[col].astype(str).str.lower()
        mask = values.str.contains("full", na=False)
        if mask.any():
            print(f"Full-cohort module membership selected using column: {col}")
            return module_membership.loc[mask].copy()

    if "fold" in module_membership.columns:
        numeric_fold = pd.to_numeric(module_membership["fold"], errors="coerce")
        mask = numeric_fold.isna()
        if mask.any():
            print("Full-cohort module membership selected using missing fold values.")
            return module_membership.loc[mask].copy()

    print(
        "No explicit full-cohort scope column was detected. "
        "Module rows will be restricted later to labels present in the full-cohort score file."
    )
    return module_membership.copy()


def get_module_score_mapping(module_scores):
    mapping = {}
    for col in module_scores.columns:
        value = str(col)
        if value.startswith("module_") and value.endswith("_score"):
            label = value[len("module_") : -len("_score")]
            mapping[label] = col
        else:
            mapping[value] = col
    return mapping


def build_module_gene_map(module_membership, expression, module_scores):
    membership = filter_full_cohort_membership(module_membership)
    module_col, gene_col, overlap = detect_module_columns(membership, expression)
    print(f"Module label column: {module_col}")
    print(f"Module gene column: {gene_col}")
    print(f"Exact expression-column overlap in detected gene column: {overlap}")

    score_mapping = get_module_score_mapping(module_scores)
    valid_labels = set(score_mapping)
    membership[module_col] = membership[module_col].astype(str)
    membership = membership[membership[module_col].isin(valid_labels)].copy()

    module_gene_map = {}
    for module_label, part in membership.groupby(module_col):
        genes = resolve_gene_values_to_expression(part[gene_col], expression)
        module_gene_map[str(module_label)] = genes

    missing_labels = sorted(valid_labels - set(module_gene_map))
    if missing_labels:
        print("Module labels without membership rows:")
        for label in missing_labels:
            print(f"  {label}")

    return module_gene_map, score_mapping


def load_proliferation_gene_columns(proliferation_gene_table, expression):
    gene_col, overlap = find_best_overlap_column(
        table=proliferation_gene_table,
        valid_values=expression.columns,
        preferred=[
            "gene",
            "expression_column",
            "gene_column",
            "proliferation_gene",
            "feature",
        ],
    )
    if gene_col is None:
        raise ValueError("Could not identify a gene column in the proliferation gene table.")

    genes = resolve_gene_values_to_expression(proliferation_gene_table[gene_col], expression)
    print(f"Proliferation gene column: {gene_col}")
    print(f"Exact overlap detected in source table: {overlap}")
    print(f"Resolved proliferation expression columns: {len(genes)}")
    return genes


def standardize_train_test(train_x, test_x):
    train_x = train_x.apply(pd.to_numeric, errors="coerce")
    test_x = test_x.apply(pd.to_numeric, errors="coerce")
    train_x = train_x.replace([np.inf, -np.inf], np.nan)
    test_x = test_x.replace([np.inf, -np.inf], np.nan)

    medians = train_x.median(axis=0)
    train_x = train_x.fillna(medians)
    test_x = test_x.fillna(medians)

    means = train_x.mean(axis=0)
    stds = train_x.std(axis=0).replace(0, np.nan)
    train_z = (train_x - means) / stds
    test_z = (test_x - means) / stds

    valid_cols = train_z.columns[
        train_z.notna().all(axis=0) & test_z.notna().all(axis=0)
    ]
    train_z = train_z[valid_cols]
    test_z = test_z[valid_cols]
    return train_z, test_z


def standardize_series_train_test(train_values, test_values):
    train_values = pd.to_numeric(train_values, errors="coerce")
    test_values = pd.to_numeric(test_values, errors="coerce")

    median = train_values.median()
    train_values = train_values.fillna(median)
    test_values = test_values.fillna(median)

    mean = train_values.mean()
    std = train_values.std()
    if not np.isfinite(std) or std == 0:
        return None, None

    train_z = (train_values - mean) / std
    test_z = (test_values - mean) / std
    return train_z, test_z


def train_test_pca_score(
    train_expression,
    test_expression,
    genes,
    reference_train=None,
):
    genes = [g for g in genes if g in train_expression.columns and g in test_expression.columns]
    genes = list(dict.fromkeys(genes))
    if len(genes) < MIN_GENES_FOR_PCA:
        return None

    train_z, test_z = standardize_train_test(
        train_expression[genes].copy(),
        test_expression[genes].copy(),
    )
    genes_used = list(train_z.columns)
    if len(genes_used) < MIN_GENES_FOR_PCA:
        return None

    pca = PCA(n_components=1, random_state=RANDOM_SEED)
    train_score = pd.Series(
        pca.fit_transform(train_z).ravel(),
        index=train_z.index,
        dtype=float,
    )
    test_score = pd.Series(
        pca.transform(test_z).ravel(),
        index=test_z.index,
        dtype=float,
    )

    orientation_reference = None
    if reference_train is not None:
        orientation_reference = pd.Series(reference_train).reindex(train_score.index)
    else:
        orientation_reference = train_z.mean(axis=1)

    correlation = safe_corr(train_score, orientation_reference)
    if np.isfinite(correlation) and correlation < 0:
        train_score = -train_score
        test_score = -test_score

    train_score, test_score = standardize_series_train_test(train_score, test_score)
    if train_score is None:
        return None

    return {
        "train": train_score,
        "test": test_score,
        "genes_used": genes_used,
        "pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
    }


def full_cohort_pca_score(expression, genes, reference=None):
    result = train_test_pca_score(
        train_expression=expression,
        test_expression=expression,
        genes=genes,
        reference_train=reference,
    )
    if result is None:
        return None
    return {
        "score": result["train"],
        "genes_used": result["genes_used"],
        "pc1_explained_variance": result["pc1_explained_variance"],
    }


def fit_linear_residualization_train_test(
    train_score,
    test_score,
    train_covariates,
    test_covariates,
):
    train_frame = pd.concat(
        [pd.Series(train_score, name="score"), train_covariates],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()

    test_frame = pd.concat(
        [pd.Series(test_score, name="score"), test_covariates],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if train_frame.shape[0] < 30 or test_frame.shape[0] < 5:
        return None

    covariate_cols = list(train_covariates.columns)
    x_train = train_frame[covariate_cols].values.astype(float)
    x_test = test_frame[covariate_cols].values.astype(float)
    x_train = np.column_stack([np.ones(x_train.shape[0]), x_train])
    x_test = np.column_stack([np.ones(x_test.shape[0]), x_test])
    y_train = train_frame["score"].values.astype(float)
    y_test = test_frame["score"].values.astype(float)

    try:
        beta, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
    except Exception:
        return None

    train_residual = pd.Series(
        y_train - x_train @ beta,
        index=train_frame.index,
        dtype=float,
    )
    test_residual = pd.Series(
        y_test - x_test @ beta,
        index=test_frame.index,
        dtype=float,
    )

    residual_sd_before_standardization = float(train_residual.std())
    train_residual, test_residual = standardize_series_train_test(
        train_residual,
        test_residual,
    )
    if train_residual is None:
        return None

    return {
        "train": train_residual,
        "test": test_residual,
        "beta": beta,
        "residual_sd_before_standardization": residual_sd_before_standardization,
    }


def residualize_full_cohort(score, covariates):
    result = fit_linear_residualization_train_test(
        train_score=score,
        test_score=score,
        train_covariates=covariates,
        test_covariates=covariates,
    )
    if result is None:
        return None
    return {
        "score": result["train"],
        "residual_sd_before_standardization": result[
            "residual_sd_before_standardization"
        ],
    }


def fit_cox_full(data, time_col, event_col, feature_cols, focal_feature=None):
    needed = [time_col, event_col] + feature_cols
    use = data[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    result = {
        "n": use.shape[0],
        "events": int(use[event_col].sum()) if use.shape[0] else 0,
        "coef": np.nan,
        "hr": np.nan,
        "p": np.nan,
        "se": np.nan,
        "c_index": np.nan,
        "error": "",
    }

    if use.shape[0] < 30:
        result["error"] = "too_few_samples"
        return result
    if use[event_col].sum() < 5:
        result["error"] = "too_few_events"
        return result
    if any(use[col].std() == 0 for col in feature_cols):
        result["error"] = "zero_variance_feature"
        return result

    cph = CoxPHFitter(penalizer=COX_PENALIZER)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            cph.fit(
                use,
                duration_col=time_col,
                event_col=event_col,
                fit_options={"max_steps": 500},
            )

        risk = cph.predict_partial_hazard(use[feature_cols]).values.ravel()
        result["c_index"] = float(
            concordance_index(use[time_col], -risk, use[event_col])
        )

        focal = focal_feature or feature_cols[0]
        if focal in cph.summary.index:
            summary = cph.summary.loc[focal]
            result["coef"] = float(summary["coef"])
            result["hr"] = float(summary["exp(coef)"])
            result["p"] = float(summary["p"])
            result["se"] = float(summary["se(coef)"])
    except Exception as exc:
        result["error"] = str(exc)[:500]

    return result


def fit_and_score_cox_train_test(
    train_clinical,
    test_clinical,
    train_features,
    test_features,
    time_col,
    event_col,
):
    feature_cols = list(train_features.columns)
    if not feature_cols:
        return np.nan, "no_features"

    train_df = train_clinical[[time_col, event_col]].join(
        train_features,
        how="inner",
    ).replace([np.inf, -np.inf], np.nan).dropna()

    test_df = test_clinical[[time_col, event_col]].join(
        test_features,
        how="inner",
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if train_df.shape[0] < 30 or test_df.shape[0] < 5:
        return np.nan, "too_few_samples"
    if train_df[event_col].sum() < 5 or test_df[event_col].sum() < 2:
        return np.nan, "too_few_events"
    if any(train_df[col].std() == 0 for col in feature_cols):
        return np.nan, "zero_variance_feature"

    cph = CoxPHFitter(penalizer=COX_PENALIZER)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            cph.fit(
                train_df[[time_col, event_col] + feature_cols],
                duration_col=time_col,
                event_col=event_col,
                fit_options={"max_steps": 500},
            )

        risk = cph.predict_partial_hazard(test_df[feature_cols]).values.ravel()
        c_index = concordance_index(
            test_df[time_col].values,
            -risk,
            test_df[event_col].values,
        )
        return float(c_index), ""
    except Exception as exc:
        return np.nan, str(exc)[:500]


def prepare_weight_train_test(train_clinical, test_clinical, weight_col):
    if weight_col is None:
        return None
    train_weight, test_weight = standardize_series_train_test(
        train_clinical[weight_col],
        test_clinical[weight_col],
    )
    if train_weight is None:
        return None
    return {
        "train": train_weight,
        "test": test_weight,
    }


def build_overlap_audit(
    module_gene_map,
    proliferation_genes,
    module_scores,
    proliferation_score,
    priority,
):
    proliferation_symbols = {
        clean_gene_symbol(g).upper() for g in proliferation_genes
    }
    rows = []

    score_mapping = get_module_score_mapping(module_scores)
    for module_label, module_genes in sorted(module_gene_map.items()):
        if module_label not in score_mapping:
            continue
        module_symbols = {clean_gene_symbol(g).upper() for g in module_genes}
        overlap_symbols = module_symbols & proliferation_symbols
        disjoint_genes = [
            g
            for g in proliferation_genes
            if clean_gene_symbol(g).upper() not in module_symbols
        ]

        module_score = module_scores[score_mapping[module_label]]
        raw_corr = safe_corr(module_score, proliferation_score)
        residual_variance_fraction = (
            max(0.0, 1.0 - raw_corr ** 2) if np.isfinite(raw_corr) else np.nan
        )

        original_residual = residualize_full_cohort(
            score=module_score,
            covariates=pd.DataFrame(
                {"proliferation": proliferation_score},
                index=module_score.index,
            ),
        )

        if original_residual is None:
            post_corr = np.nan
            residual_sd = np.nan
        else:
            post_corr = safe_corr(
                original_residual["score"],
                proliferation_score,
            )
            residual_sd = original_residual[
                "residual_sd_before_standardization"
            ]

        rows.append(
            {
                "module_label": module_label,
                "n_module_genes_used": len(module_genes),
                "n_proliferation_genes_total": len(proliferation_genes),
                "n_overlap_symbols": len(overlap_symbols),
                "n_disjoint_proliferation_genes": len(disjoint_genes),
                "fraction_module_symbols_in_proliferation": (
                    len(overlap_symbols) / len(module_symbols)
                    if module_symbols
                    else np.nan
                ),
                "fraction_proliferation_symbols_in_module": (
                    len(overlap_symbols) / len(proliferation_symbols)
                    if proliferation_symbols
                    else np.nan
                ),
                "raw_module_proliferation_correlation": raw_corr,
                "orthogonal_variance_fraction_1_minus_r2": residual_variance_fraction,
                "original_residual_sd_before_standardization": residual_sd,
                "original_residual_post_correlation": post_corr,
                "overlap_symbols": ";".join(sorted(overlap_symbols)),
                "disjoint_proliferation_genes": ";".join(disjoint_genes),
            }
        )

    audit = pd.DataFrame(rows)
    if not priority.empty and "module_label" in priority.columns:
        keep = [
            "module_label",
            "module_transfer_qc_tier",
            "transfer_priority_score",
            "n_module_genes",
            "fraction_strict_symbol_concordant",
            "fraction_broad_transferable",
            "strict_human_symbols",
        ]
        keep = [c for c in keep if c in priority.columns]
        audit = audit.merge(
            priority[keep].drop_duplicates("module_label"),
            on="module_label",
            how="left",
        )

    return audit


def run_full_cohort_leave_module_out_analysis(
    expression,
    clinical,
    module_gene_map,
    module_scores,
    proliferation_genes,
    proliferation_score,
    priority,
    weight_col,
):
    rows = []
    score_mapping = get_module_score_mapping(module_scores)

    if weight_col is not None:
        weight = pd.to_numeric(clinical[weight_col], errors="coerce")
        weight = (weight - weight.mean()) / weight.std()
    else:
        weight = None

    for module_label, module_genes in sorted(module_gene_map.items()):
        if module_label not in score_mapping:
            continue

        module_symbols = {clean_gene_symbol(g).upper() for g in module_genes}
        disjoint_genes = [
            gene
            for gene in proliferation_genes
            if clean_gene_symbol(gene).upper() not in module_symbols
        ]

        if len(disjoint_genes) < MIN_DISJOINT_PROLIFERATION_GENES:
            print(
                f"Skipping full-cohort leave-module-out score for {module_label}: "
                f"only {len(disjoint_genes)} proliferation genes remain."
            )
            continue

        disjoint_score_result = full_cohort_pca_score(
            expression=expression,
            genes=disjoint_genes,
            reference=proliferation_score,
        )
        if disjoint_score_result is None:
            continue

        module_score = module_scores[score_mapping[module_label]].reindex(clinical.index)
        disjoint_score = disjoint_score_result["score"].reindex(clinical.index)

        residual_disjoint = residualize_full_cohort(
            score=module_score,
            covariates=pd.DataFrame(
                {"disjoint_proliferation": disjoint_score},
                index=clinical.index,
            ),
        )

        residual_disjoint_weight = None
        if weight is not None:
            residual_disjoint_weight = residualize_full_cohort(
                score=module_score,
                covariates=pd.DataFrame(
                    {
                        "disjoint_proliferation": disjoint_score,
                        "weight": weight,
                    },
                    index=clinical.index,
                ),
            )

        analysis = clinical.copy()
        analysis["module_score"] = module_score
        analysis["original_proliferation"] = proliferation_score.reindex(clinical.index)
        analysis["disjoint_proliferation"] = disjoint_score
        if residual_disjoint is not None:
            analysis["residual_disjoint_proliferation"] = residual_disjoint["score"]
        if residual_disjoint_weight is not None:
            analysis["residual_disjoint_proliferation_weight"] = residual_disjoint_weight[
                "score"
            ]

        model_specs = [
            (
                "module_only",
                ["module_score"],
                "module_score",
            ),
            (
                "module_plus_original_proliferation",
                ["module_score", "original_proliferation"],
                "module_score",
            ),
            (
                "module_plus_disjoint_proliferation",
                ["module_score", "disjoint_proliferation"],
                "module_score",
            ),
        ]

        if residual_disjoint is not None:
            model_specs.append(
                (
                    "residual_to_disjoint_proliferation",
                    ["residual_disjoint_proliferation"],
                    "residual_disjoint_proliferation",
                )
            )
        if residual_disjoint_weight is not None:
            model_specs.append(
                (
                    "residual_to_disjoint_proliferation_and_weight",
                    ["residual_disjoint_proliferation_weight"],
                    "residual_disjoint_proliferation_weight",
                )
            )

        for endpoint_label, endpoint in ENDPOINTS.items():
            time_col = endpoint["time_col"]
            event_col = endpoint["event_col"]
            for model_name, feature_cols, focal_feature in model_specs:
                result = fit_cox_full(
                    data=analysis,
                    time_col=time_col,
                    event_col=event_col,
                    feature_cols=feature_cols,
                    focal_feature=focal_feature,
                )

                rows.append(
                    {
                        "endpoint": endpoint_label,
                        "module_label": module_label,
                        "model": model_name,
                        "n_module_genes_used": len(module_genes),
                        "n_disjoint_proliferation_genes_used": len(
                            disjoint_score_result["genes_used"]
                        ),
                        "disjoint_proliferation_pc1_explained_variance": (
                            disjoint_score_result["pc1_explained_variance"]
                        ),
                        "module_original_proliferation_correlation": safe_corr(
                            module_score,
                            proliferation_score,
                        ),
                        "module_disjoint_proliferation_correlation": safe_corr(
                            module_score,
                            disjoint_score,
                        ),
                        "residual_disjoint_post_correlation": (
                            safe_corr(residual_disjoint["score"], disjoint_score)
                            if residual_disjoint is not None
                            else np.nan
                        ),
                        "residual_disjoint_sd_before_standardization": (
                            residual_disjoint[
                                "residual_sd_before_standardization"
                            ]
                            if residual_disjoint is not None
                            else np.nan
                        ),
                        "n": result["n"],
                        "events": result["events"],
                        "coef": result["coef"],
                        "hr": result["hr"],
                        "p": result["p"],
                        "se": result["se"],
                        "c_index": result["c_index"],
                        "error": result["error"],
                    }
                )

    results = pd.DataFrame(rows)
    for endpoint_label in ENDPOINTS:
        for model_name in results["model"].dropna().unique():
            mask = (
                results["endpoint"].eq(endpoint_label)
                & results["model"].eq(model_name)
            )
            results.loc[mask, "q"] = bh_adjust(results.loc[mask, "p"])

    if not priority.empty and "module_label" in priority.columns:
        keep = [
            "module_label",
            "module_transfer_qc_tier",
            "transfer_priority_score",
            "fraction_strict_symbol_concordant",
            "fraction_broad_transferable",
            "strict_human_symbols",
        ]
        keep = [c for c in keep if c in priority.columns]
        results = results.merge(
            priority[keep].drop_duplicates("module_label"),
            on="module_label",
            how="left",
        )

    return results


def score_model_and_record(
    rows,
    endpoint_label,
    repeat,
    fold,
    module_label,
    model_name,
    train_clinical,
    test_clinical,
    train_features,
    test_features,
    time_col,
    event_col,
    metadata,
):
    c_index, error = fit_and_score_cox_train_test(
        train_clinical=train_clinical,
        test_clinical=test_clinical,
        train_features=train_features,
        test_features=test_features,
        time_col=time_col,
        event_col=event_col,
    )

    rows.append(
        {
            "endpoint": endpoint_label,
            "repeat": repeat,
            "fold": fold,
            "module_label": module_label,
            "model": model_name,
            "c_index": c_index,
            "error": error,
            **metadata,
        }
    )


def run_repeated_cv_sensitivity(
    expression,
    clinical,
    module_gene_map,
    module_scores,
    proliferation_genes,
    proliferation_score,
    weight_col,
):
    rows = []
    score_mapping = get_module_score_mapping(module_scores)

    for endpoint_label, endpoint in ENDPOINTS.items():
        time_col = endpoint["time_col"]
        event_col = endpoint["event_col"]

        clinical_ep = clinical.copy()
        clinical_ep[time_col] = pd.to_numeric(clinical_ep[time_col], errors="coerce")
        clinical_ep[event_col] = pd.to_numeric(clinical_ep[event_col], errors="coerce")
        valid_samples = clinical_ep[[time_col, event_col]].dropna().index
        clinical_ep = clinical_ep.loc[valid_samples].copy()
        expression_ep = expression.loc[valid_samples].copy()
        proliferation_score_ep = proliferation_score.loc[valid_samples].copy()
        module_scores_ep = module_scores.loc[valid_samples].copy()

        samples = np.array(valid_samples)
        event_values = clinical_ep[event_col].astype(int).values

        print("")
        print("=" * 80)
        print(f"Repeated cross-fitted proliferation sensitivity for {endpoint_label}")
        print("=" * 80)
        print(f"Samples: {clinical_ep.shape[0]}")
        print(f"Events: {int(clinical_ep[event_col].sum())}")
        print(f"Repeats: {N_REPEATS}")
        print(f"Folds per repeat: {N_SPLITS}")

        for repeat in range(1, N_REPEATS + 1):
            cv = StratifiedKFold(
                n_splits=N_SPLITS,
                shuffle=True,
                random_state=RANDOM_SEED + repeat - 1,
            )

            print(f"Repeat {repeat}/{N_REPEATS}")
            for fold, (train_idx, test_idx) in enumerate(
                cv.split(samples, event_values),
                start=1,
            ):
                train_samples = samples[train_idx]
                test_samples = samples[test_idx]

                train_clinical = clinical_ep.loc[train_samples].copy()
                test_clinical = clinical_ep.loc[test_samples].copy()
                train_expression = expression_ep.loc[train_samples].copy()
                test_expression = expression_ep.loc[test_samples].copy()

                weight_result = prepare_weight_train_test(
                    train_clinical,
                    test_clinical,
                    weight_col,
                )

                for module_label, module_genes in sorted(module_gene_map.items()):
                    if module_label not in score_mapping:
                        continue

                    module_symbols = {
                        clean_gene_symbol(g).upper() for g in module_genes
                    }
                    disjoint_genes = [
                        gene
                        for gene in proliferation_genes
                        if clean_gene_symbol(gene).upper() not in module_symbols
                    ]
                    if len(disjoint_genes) < MIN_DISJOINT_PROLIFERATION_GENES:
                        continue

                    module_reference_train = module_scores_ep.loc[
                        train_samples,
                        score_mapping[module_label],
                    ]
                    proliferation_reference_train = proliferation_score_ep.loc[
                        train_samples
                    ]

                    module_score_result = train_test_pca_score(
                        train_expression=train_expression,
                        test_expression=test_expression,
                        genes=module_genes,
                        reference_train=module_reference_train,
                    )
                    original_prolif_result = train_test_pca_score(
                        train_expression=train_expression,
                        test_expression=test_expression,
                        genes=proliferation_genes,
                        reference_train=proliferation_reference_train,
                    )
                    disjoint_prolif_result = train_test_pca_score(
                        train_expression=train_expression,
                        test_expression=test_expression,
                        genes=disjoint_genes,
                        reference_train=proliferation_reference_train,
                    )

                    if (
                        module_score_result is None
                        or original_prolif_result is None
                        or disjoint_prolif_result is None
                    ):
                        continue

                    module_train = module_score_result["train"]
                    module_test = module_score_result["test"]
                    original_prolif_train = original_prolif_result["train"]
                    original_prolif_test = original_prolif_result["test"]
                    disjoint_prolif_train = disjoint_prolif_result["train"]
                    disjoint_prolif_test = disjoint_prolif_result["test"]

                    residual_original = fit_linear_residualization_train_test(
                        train_score=module_train,
                        test_score=module_test,
                        train_covariates=pd.DataFrame(
                            {"proliferation": original_prolif_train},
                            index=module_train.index,
                        ),
                        test_covariates=pd.DataFrame(
                            {"proliferation": original_prolif_test},
                            index=module_test.index,
                        ),
                    )

                    residual_disjoint = fit_linear_residualization_train_test(
                        train_score=module_train,
                        test_score=module_test,
                        train_covariates=pd.DataFrame(
                            {"proliferation": disjoint_prolif_train},
                            index=module_train.index,
                        ),
                        test_covariates=pd.DataFrame(
                            {"proliferation": disjoint_prolif_test},
                            index=module_test.index,
                        ),
                    )

                    residual_disjoint_weight = None
                    if weight_result is not None:
                        residual_disjoint_weight = fit_linear_residualization_train_test(
                            train_score=module_train,
                            test_score=module_test,
                            train_covariates=pd.DataFrame(
                                {
                                    "proliferation": disjoint_prolif_train,
                                    "weight": weight_result["train"],
                                },
                                index=module_train.index,
                            ),
                            test_covariates=pd.DataFrame(
                                {
                                    "proliferation": disjoint_prolif_test,
                                    "weight": weight_result["test"],
                                },
                                index=module_test.index,
                            ),
                        )

                    metadata = {
                        "n_train": len(train_samples),
                        "n_test": len(test_samples),
                        "train_events": int(train_clinical[event_col].sum()),
                        "test_events": int(test_clinical[event_col].sum()),
                        "n_module_genes_used": len(module_score_result["genes_used"]),
                        "n_original_proliferation_genes_used": len(
                            original_prolif_result["genes_used"]
                        ),
                        "n_disjoint_proliferation_genes_used": len(
                            disjoint_prolif_result["genes_used"]
                        ),
                        "module_pc1_explained_variance": module_score_result[
                            "pc1_explained_variance"
                        ],
                        "original_proliferation_pc1_explained_variance": (
                            original_prolif_result["pc1_explained_variance"]
                        ),
                        "disjoint_proliferation_pc1_explained_variance": (
                            disjoint_prolif_result["pc1_explained_variance"]
                        ),
                        "train_module_original_proliferation_correlation": safe_corr(
                            module_train,
                            original_prolif_train,
                        ),
                        "train_module_disjoint_proliferation_correlation": safe_corr(
                            module_train,
                            disjoint_prolif_train,
                        ),
                    }

                    score_model_and_record(
                        rows=rows,
                        endpoint_label=endpoint_label,
                        repeat=repeat,
                        fold=fold,
                        module_label=module_label,
                        model_name="module_only",
                        train_clinical=train_clinical,
                        test_clinical=test_clinical,
                        train_features=pd.DataFrame(
                            {"module_score": module_train},
                            index=module_train.index,
                        ),
                        test_features=pd.DataFrame(
                            {"module_score": module_test},
                            index=module_test.index,
                        ),
                        time_col=time_col,
                        event_col=event_col,
                        metadata=metadata,
                    )

                    score_model_and_record(
                        rows=rows,
                        endpoint_label=endpoint_label,
                        repeat=repeat,
                        fold=fold,
                        module_label=module_label,
                        model_name="original_proliferation_only",
                        train_clinical=train_clinical,
                        test_clinical=test_clinical,
                        train_features=pd.DataFrame(
                            {"proliferation": original_prolif_train},
                            index=original_prolif_train.index,
                        ),
                        test_features=pd.DataFrame(
                            {"proliferation": original_prolif_test},
                            index=original_prolif_test.index,
                        ),
                        time_col=time_col,
                        event_col=event_col,
                        metadata=metadata,
                    )

                    score_model_and_record(
                        rows=rows,
                        endpoint_label=endpoint_label,
                        repeat=repeat,
                        fold=fold,
                        module_label=module_label,
                        model_name="module_plus_original_proliferation",
                        train_clinical=train_clinical,
                        test_clinical=test_clinical,
                        train_features=pd.DataFrame(
                            {
                                "module_score": module_train,
                                "proliferation": original_prolif_train,
                            },
                            index=module_train.index,
                        ),
                        test_features=pd.DataFrame(
                            {
                                "module_score": module_test,
                                "proliferation": original_prolif_test,
                            },
                            index=module_test.index,
                        ),
                        time_col=time_col,
                        event_col=event_col,
                        metadata=metadata,
                    )

                    score_model_and_record(
                        rows=rows,
                        endpoint_label=endpoint_label,
                        repeat=repeat,
                        fold=fold,
                        module_label=module_label,
                        model_name="disjoint_proliferation_only",
                        train_clinical=train_clinical,
                        test_clinical=test_clinical,
                        train_features=pd.DataFrame(
                            {"proliferation": disjoint_prolif_train},
                            index=disjoint_prolif_train.index,
                        ),
                        test_features=pd.DataFrame(
                            {"proliferation": disjoint_prolif_test},
                            index=disjoint_prolif_test.index,
                        ),
                        time_col=time_col,
                        event_col=event_col,
                        metadata=metadata,
                    )

                    score_model_and_record(
                        rows=rows,
                        endpoint_label=endpoint_label,
                        repeat=repeat,
                        fold=fold,
                        module_label=module_label,
                        model_name="module_plus_disjoint_proliferation",
                        train_clinical=train_clinical,
                        test_clinical=test_clinical,
                        train_features=pd.DataFrame(
                            {
                                "module_score": module_train,
                                "proliferation": disjoint_prolif_train,
                            },
                            index=module_train.index,
                        ),
                        test_features=pd.DataFrame(
                            {
                                "module_score": module_test,
                                "proliferation": disjoint_prolif_test,
                            },
                            index=module_test.index,
                        ),
                        time_col=time_col,
                        event_col=event_col,
                        metadata=metadata,
                    )

                    if residual_original is not None:
                        score_model_and_record(
                            rows=rows,
                            endpoint_label=endpoint_label,
                            repeat=repeat,
                            fold=fold,
                            module_label=module_label,
                            model_name="residual_to_original_proliferation",
                            train_clinical=train_clinical,
                            test_clinical=test_clinical,
                            train_features=pd.DataFrame(
                                {"module_residual": residual_original["train"]},
                                index=residual_original["train"].index,
                            ),
                            test_features=pd.DataFrame(
                                {"module_residual": residual_original["test"]},
                                index=residual_original["test"].index,
                            ),
                            time_col=time_col,
                            event_col=event_col,
                            metadata={
                                **metadata,
                                "residual_sd_before_standardization": residual_original[
                                    "residual_sd_before_standardization"
                                ],
                            },
                        )

                    if residual_disjoint is not None:
                        score_model_and_record(
                            rows=rows,
                            endpoint_label=endpoint_label,
                            repeat=repeat,
                            fold=fold,
                            module_label=module_label,
                            model_name="residual_to_disjoint_proliferation",
                            train_clinical=train_clinical,
                            test_clinical=test_clinical,
                            train_features=pd.DataFrame(
                                {"module_residual": residual_disjoint["train"]},
                                index=residual_disjoint["train"].index,
                            ),
                            test_features=pd.DataFrame(
                                {"module_residual": residual_disjoint["test"]},
                                index=residual_disjoint["test"].index,
                            ),
                            time_col=time_col,
                            event_col=event_col,
                            metadata={
                                **metadata,
                                "residual_sd_before_standardization": residual_disjoint[
                                    "residual_sd_before_standardization"
                                ],
                            },
                        )

                    if residual_disjoint_weight is not None:
                        score_model_and_record(
                            rows=rows,
                            endpoint_label=endpoint_label,
                            repeat=repeat,
                            fold=fold,
                            module_label=module_label,
                            model_name=(
                                "residual_to_disjoint_proliferation_and_weight"
                            ),
                            train_clinical=train_clinical,
                            test_clinical=test_clinical,
                            train_features=pd.DataFrame(
                                {
                                    "module_residual": residual_disjoint_weight[
                                        "train"
                                    ]
                                },
                                index=residual_disjoint_weight["train"].index,
                            ),
                            test_features=pd.DataFrame(
                                {
                                    "module_residual": residual_disjoint_weight[
                                        "test"
                                    ]
                                },
                                index=residual_disjoint_weight["test"].index,
                            ),
                            time_col=time_col,
                            event_col=event_col,
                            metadata={
                                **metadata,
                                "residual_sd_before_standardization": (
                                    residual_disjoint_weight[
                                        "residual_sd_before_standardization"
                                    ]
                                ),
                            },
                        )

    return pd.DataFrame(rows)


def summarize_cv_results(cv_results):
    valid = cv_results[np.isfinite(cv_results["c_index"])].copy()
    summary = (
        valid.groupby(["endpoint", "module_label", "model"], dropna=False)
        .agg(
            n_valid_folds=("c_index", "count"),
            mean_c_index=("c_index", "mean"),
            std_c_index=("c_index", "std"),
            median_c_index=("c_index", "median"),
            q25_c_index=("c_index", lambda x: np.nanquantile(x, 0.25)),
            q75_c_index=("c_index", lambda x: np.nanquantile(x, 0.75)),
            min_c_index=("c_index", "min"),
            max_c_index=("c_index", "max"),
            fraction_above_0_50=("c_index", lambda x: float((x > 0.50).mean())),
            fraction_above_0_55=("c_index", lambda x: float((x > 0.55).mean())),
            mean_train_module_original_proliferation_correlation=(
                "train_module_original_proliferation_correlation",
                "mean",
            ),
            mean_train_module_disjoint_proliferation_correlation=(
                "train_module_disjoint_proliferation_correlation",
                "mean",
            ),
            mean_n_module_genes_used=("n_module_genes_used", "mean"),
            mean_n_disjoint_proliferation_genes_used=(
                "n_disjoint_proliferation_genes_used",
                "mean",
            ),
        )
        .reset_index()
    )

    pivot = summary.pivot_table(
        index=["endpoint", "module_label"],
        columns="model",
        values="mean_c_index",
    )
    pivot.columns = [f"mean_c_index__{c}" for c in pivot.columns]
    pivot = pivot.reset_index()
    summary = summary.merge(pivot, on=["endpoint", "module_label"], how="left")

    if {
        "mean_c_index__residual_to_disjoint_proliferation",
        "mean_c_index__residual_to_original_proliferation",
    }.issubset(summary.columns):
        summary["delta_disjoint_residual_vs_original_residual"] = (
            summary["mean_c_index__residual_to_disjoint_proliferation"]
            - summary["mean_c_index__residual_to_original_proliferation"]
        )

    if {
        "mean_c_index__module_plus_disjoint_proliferation",
        "mean_c_index__disjoint_proliferation_only",
    }.issubset(summary.columns):
        summary["delta_joint_module_vs_disjoint_proliferation_only"] = (
            summary["mean_c_index__module_plus_disjoint_proliferation"]
            - summary["mean_c_index__disjoint_proliferation_only"]
        )

    return summary


def build_decision_table(audit, full_results, cv_summary, priority):
    full_residual = full_results[
        full_results["model"].eq("residual_to_disjoint_proliferation")
    ][
        [
            "endpoint",
            "module_label",
            "p",
            "q",
            "c_index",
            "module_disjoint_proliferation_correlation",
            "residual_disjoint_sd_before_standardization",
        ]
    ].copy()

    full_residual = full_residual.rename(
        columns={
            "p": "full_disjoint_residual_p",
            "q": "full_disjoint_residual_q",
            "c_index": "full_disjoint_residual_c_index",
        }
    )

    cv_residual = cv_summary[
        cv_summary["model"].eq("residual_to_disjoint_proliferation")
    ][
        [
            "endpoint",
            "module_label",
            "n_valid_folds",
            "mean_c_index",
            "std_c_index",
            "median_c_index",
            "fraction_above_0_50",
            "fraction_above_0_55",
            "mean_train_module_original_proliferation_correlation",
            "mean_train_module_disjoint_proliferation_correlation",
        ]
    ].copy()

    cv_residual = cv_residual.rename(
        columns={
            "mean_c_index": "cv_disjoint_residual_mean_c_index",
            "std_c_index": "cv_disjoint_residual_std_c_index",
            "median_c_index": "cv_disjoint_residual_median_c_index",
            "fraction_above_0_50": "cv_disjoint_residual_fraction_above_0_50",
            "fraction_above_0_55": "cv_disjoint_residual_fraction_above_0_55",
        }
    )

    base = audit.copy()
    endpoints = pd.DataFrame({"endpoint": list(ENDPOINTS)})
    base["_key"] = 1
    endpoints["_key"] = 1
    decision = base.merge(endpoints, on="_key", how="outer").drop(columns="_key")
    decision = decision.merge(
        full_residual,
        on=["endpoint", "module_label"],
        how="left",
    )
    decision = decision.merge(
        cv_residual,
        on=["endpoint", "module_label"],
        how="left",
    )

    if not priority.empty and "module_label" in priority.columns:
        existing = set(decision.columns)
        keep = [
            "module_label",
            "module_transfer_qc_tier",
            "transfer_priority_score",
            "fraction_strict_symbol_concordant",
            "fraction_broad_transferable",
            "strict_human_symbols",
        ]
        keep = [c for c in keep if c in priority.columns and c not in existing]
        if keep:
            decision = decision.merge(
                priority[["module_label"] + keep].drop_duplicates("module_label"),
                on="module_label",
                how="left",
            )

    raw_abs_corr = decision["raw_module_proliferation_correlation"].abs()
    q = pd.to_numeric(decision["full_disjoint_residual_q"], errors="coerce")
    cv_c = pd.to_numeric(
        decision["cv_disjoint_residual_mean_c_index"],
        errors="coerce",
    )
    cv_fraction = pd.to_numeric(
        decision["cv_disjoint_residual_fraction_above_0_50"],
        errors="coerce",
    )
    transfer_priority = pd.to_numeric(
        decision.get("transfer_priority_score", np.nan),
        errors="coerce",
    )

    decision["passes_full_cohort_disjoint_residual_fdr10"] = q < 0.10
    decision["passes_cv_disjoint_residual_mean_055"] = cv_c > 0.55
    decision["passes_cv_majority_above_chance"] = cv_fraction >= 0.60
    decision["is_extreme_proliferation_overlap"] = raw_abs_corr >= 0.90

    clean_primary = (
        (raw_abs_corr < 0.70)
        & (q < 0.10)
        & (cv_c > 0.55)
        & (cv_fraction >= 0.60)
        & (transfer_priority >= 8)
    )
    clean_sensitivity = (
        (raw_abs_corr < 0.70)
        & (q < 0.10)
        & (cv_c > 0.52)
        & (cv_fraction >= 0.55)
    )
    proliferation_residual = (
        (raw_abs_corr >= 0.90)
        & (q < 0.10)
        & (cv_c > 0.55)
        & (cv_fraction >= 0.60)
    )

    decision["recommended_role"] = np.select(
        [
            clean_primary,
            proliferation_residual,
            clean_sensitivity,
            raw_abs_corr >= 0.90,
        ],
        [
            "primary_non_proliferation_transfer_program",
            "proliferation_dominant_with_crossfitted_residual_component",
            "non_proliferation_sensitivity_program",
            "proliferation_reference_only_until_external_validation",
        ],
        default="exploratory_program",
    )

    decision = decision.sort_values(
        [
            "endpoint",
            "recommended_role",
            "cv_disjoint_residual_mean_c_index",
            "full_disjoint_residual_q",
        ],
        ascending=[True, True, False, True],
        na_position="last",
    )
    return decision


def print_key_results(audit, full_results, cv_summary, decision):
    print("")
    print("=" * 80)
    print("Module-proliferation overlap audit")
    print("=" * 80)
    audit_cols = [
        "module_label",
        "module_transfer_qc_tier",
        "transfer_priority_score",
        "n_module_genes_used",
        "n_overlap_symbols",
        "n_disjoint_proliferation_genes",
        "raw_module_proliferation_correlation",
        "orthogonal_variance_fraction_1_minus_r2",
        "original_residual_sd_before_standardization",
        "original_residual_post_correlation",
    ]
    audit_cols = [c for c in audit_cols if c in audit.columns]
    print(
        audit.sort_values(
            "raw_module_proliferation_correlation",
            key=lambda x: x.abs(),
            ascending=False,
        )[audit_cols].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Full-cohort leave-module-out residual results")
    print("=" * 80)
    full_display = full_results[
        full_results["model"].isin(
            [
                "module_plus_disjoint_proliferation",
                "residual_to_disjoint_proliferation",
                "residual_to_disjoint_proliferation_and_weight",
            ]
        )
    ].copy()
    full_cols = [
        "endpoint",
        "module_label",
        "model",
        "module_transfer_qc_tier",
        "transfer_priority_score",
        "module_original_proliferation_correlation",
        "module_disjoint_proliferation_correlation",
        "p",
        "q",
        "c_index",
        "n_disjoint_proliferation_genes_used",
    ]
    full_cols = [c for c in full_cols if c in full_display.columns]
    print(
        full_display.sort_values(
            ["endpoint", "model", "q", "p"],
            ascending=[True, True, True, True],
        )[full_cols].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Repeated cross-fitted residual summary")
    print("=" * 80)
    cv_display = cv_summary[
        cv_summary["model"].isin(
            [
                "module_only",
                "original_proliferation_only",
                "module_plus_original_proliferation",
                "residual_to_original_proliferation",
                "disjoint_proliferation_only",
                "module_plus_disjoint_proliferation",
                "residual_to_disjoint_proliferation",
                "residual_to_disjoint_proliferation_and_weight",
            ]
        )
    ].copy()
    cv_cols = [
        "endpoint",
        "module_label",
        "model",
        "n_valid_folds",
        "mean_c_index",
        "std_c_index",
        "median_c_index",
        "fraction_above_0_50",
        "fraction_above_0_55",
        "mean_train_module_original_proliferation_correlation",
        "mean_train_module_disjoint_proliferation_correlation",
    ]
    cv_cols = [c for c in cv_cols if c in cv_display.columns]
    print(
        cv_display.sort_values(
            ["endpoint", "module_label", "model"],
        )[cv_cols].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Decision table for primary review modules")
    print("=" * 80)
    decision_display = decision[
        decision["module_label"].isin(PRIMARY_REVIEW_MODULES)
    ].copy()
    decision_cols = [
        "endpoint",
        "module_label",
        "module_transfer_qc_tier",
        "transfer_priority_score",
        "raw_module_proliferation_correlation",
        "orthogonal_variance_fraction_1_minus_r2",
        "full_disjoint_residual_q",
        "full_disjoint_residual_c_index",
        "cv_disjoint_residual_mean_c_index",
        "cv_disjoint_residual_std_c_index",
        "cv_disjoint_residual_fraction_above_0_50",
        "recommended_role",
    ]
    decision_cols = [c for c in decision_cols if c in decision_display.columns]
    print(decision_display[decision_cols].to_string(index=False))

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print(
        "A residual score from a module correlated at r approximately 0.98 with proliferation "
        "contains only a small raw orthogonal variance component before re-standardization."
    )
    print(
        "A strong full-cohort p-value after re-standardizing that residual does not by itself "
        "prove a large clinically distinct non-proliferation program."
    )
    print(
        "The leave-module-out proliferation score removes direct gene overlap, and repeated "
        "cross-fitted scoring/residualization tests whether the residual component generalizes."
    )
    print(
        "Module definitions remain canine-discovery features. Human ortholog-mapped external "
        "validation is still required before making a translational claim."
    )


def main():
    print("=" * 80)
    print("Proliferation overlap and cross-fitted sensitivity analysis")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Processed directory: {PROCESSED_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print("")
    print("Design:")
    print("  Audit module/proliferation gene overlap and remaining variance.")
    print("  Rebuild proliferation scores after excluding each module's genes.")
    print("  Fit PCA scores, residualization, and Cox models using train folds only.")
    print("  Evaluate held-out C-index across repeated five-fold cross-validation.")
    print("")

    expression = read_required_csv(
        PROCESSED_DIR / EXPRESSION_FILE,
        index_col=0,
    )
    clinical = read_required_csv(
        PROCESSED_DIR / CLINICAL_FILE,
        index_col=0,
    )
    proliferation_gene_table = read_required_csv(
        RESULTS_DIR / PROLIFERATION_GENE_FILE,
    )
    proliferation_score_table = read_required_csv(
        RESULTS_DIR / PROLIFERATION_SCORE_FILE,
        index_col=0,
    )
    module_membership = read_required_csv(
        RESULTS_DIR / MODULE_MEMBERSHIP_FILE,
    )
    module_scores = read_required_csv(
        RESULTS_DIR / MODULE_SCORE_FILE,
        index_col=0,
    )
    priority = read_optional_csv(RESULTS_DIR / MODULE_PRIORITY_FILE)

    common_samples = clinical.index.intersection(expression.index)
    common_samples = common_samples.intersection(proliferation_score_table.index)
    common_samples = common_samples.intersection(module_scores.index)

    expression = expression.loc[common_samples].copy()
    clinical = clinical.loc[common_samples].copy()
    proliferation_score_table = proliferation_score_table.loc[common_samples].copy()
    module_scores = module_scores.loc[common_samples].copy()

    if "meta_proliferation_score" in proliferation_score_table.columns:
        proliferation_score = proliferation_score_table[
            "meta_proliferation_score"
        ].copy()
    elif proliferation_score_table.shape[1] == 1:
        proliferation_score = proliferation_score_table.iloc[:, 0].copy()
        proliferation_score.name = "meta_proliferation_score"
    else:
        raise ValueError(
            "The proliferation score file contains multiple columns and no "
            "'meta_proliferation_score' column."
        )

    proliferation_score, _ = standardize_series_train_test(
        proliferation_score,
        proliferation_score,
    )
    if proliferation_score is None:
        raise ValueError("The proliferation score has zero or invalid variance.")

    weight_col = find_weight_column(clinical)
    if weight_col is not None:
        print(f"Weight column detected: {weight_col}")
    else:
        print("No weight column detected. Weight-residualized analyses will be skipped.")

    proliferation_genes = load_proliferation_gene_columns(
        proliferation_gene_table,
        expression,
    )
    module_gene_map, score_mapping = build_module_gene_map(
        module_membership,
        expression,
        module_scores,
    )

    module_gene_map = {
        label: genes
        for label, genes in module_gene_map.items()
        if label in score_mapping and len(genes) >= MIN_GENES_FOR_PCA
    }

    print("")
    print("Matched data:")
    print(f"  Expression matrix: {expression.shape}")
    print(f"  Clinical table: {clinical.shape}")
    print(f"  Module score matrix: {module_scores.shape}")
    print(f"  Proliferation score samples: {proliferation_score.shape[0]}")
    print(f"  Proliferation genes: {len(proliferation_genes)}")
    print(f"  Full-cohort modules with usable membership: {len(module_gene_map)}")

    overlap_audit = build_overlap_audit(
        module_gene_map=module_gene_map,
        proliferation_genes=proliferation_genes,
        module_scores=module_scores,
        proliferation_score=proliferation_score,
        priority=priority,
    )

    full_results = run_full_cohort_leave_module_out_analysis(
        expression=expression,
        clinical=clinical,
        module_gene_map=module_gene_map,
        module_scores=module_scores,
        proliferation_genes=proliferation_genes,
        proliferation_score=proliferation_score,
        priority=priority,
        weight_col=weight_col,
    )

    cv_fold_results = run_repeated_cv_sensitivity(
        expression=expression,
        clinical=clinical,
        module_gene_map=module_gene_map,
        module_scores=module_scores,
        proliferation_genes=proliferation_genes,
        proliferation_score=proliferation_score,
        weight_col=weight_col,
    )
    cv_summary = summarize_cv_results(cv_fold_results)

    decision_table = build_decision_table(
        audit=overlap_audit,
        full_results=full_results,
        cv_summary=cv_summary,
        priority=priority,
    )

    overlap_audit.to_csv(OUTPUT_OVERLAP_AUDIT, index=False)
    full_results.to_csv(OUTPUT_FULL_RESULTS, index=False)
    cv_fold_results.to_csv(OUTPUT_CV_FOLDS, index=False)
    cv_summary.to_csv(OUTPUT_CV_SUMMARY, index=False)
    decision_table.to_csv(OUTPUT_DECISION_TABLE, index=False)

    print_key_results(
        audit=overlap_audit,
        full_results=full_results,
        cv_summary=cv_summary,
        decision=decision_table,
    )

    print("")
    print("Saved:")
    print(OUTPUT_OVERLAP_AUDIT)
    print(OUTPUT_FULL_RESULTS)
    print(OUTPUT_CV_FOLDS)
    print(OUTPUT_CV_SUMMARY)
    print(OUTPUT_DECISION_TABLE)
    print("Done.")


if __name__ == "__main__":
    main()
