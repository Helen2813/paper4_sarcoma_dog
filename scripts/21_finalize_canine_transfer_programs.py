from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXPRESSION_FILE = "GSE238110_DOG2_expression_log2cpm_matched_allgenes.csv"
MODULE_MEMBERSHIP_FILE = "GSE238110_RNA_module_gene_membership.csv"
MODULE_SCORE_FILE = "GSE238110_full_cohort_module_scores_for_proliferation_adjustment.csv"
MODULE_ASSOCIATION_FILE = "GSE238110_RNA_full_cohort_module_associations.csv"
MODULE_PRIORITY_FILE = "GSE238110_RNA_full_cohort_transferable_module_priority.csv"
ORTHOLOG_FILE = "GSE238110_RNA_master_candidate_evidence_table_with_orthologs.csv"
OVERLAP_AUDIT_FILE = "GSE238110_module_proliferation_overlap_audit.csv"
CV_FOLD_FILE = "GSE238110_repeated_cv_proliferation_sensitivity_fold_results.csv"
CV_SUMMARY_FILE = "GSE238110_repeated_cv_proliferation_sensitivity_summary.csv"
DECISION_FILE = "GSE238110_proliferation_independence_decision_table.csv"

N_SPLITS = 5
N_REPEATS = 20
BOOTSTRAP_REPS = 5000
RANDOM_SEED = 42
MIN_STRICT_HUMAN_GENES = 3

PRIMARY_CLEAN_MODULES = ["M34", "M11", "M24"]
PRIMARY_PROLIFERATION_AXIS_MODULES = ["M40"]
SECONDARY_SENSITIVITY_MODULES = ["M28", "M38", "M25", "M17"]

PROVISIONAL_PROGRAM_LABELS = {
    "M34": "immune_myeloid_inflammatory_like",
    "M11": "angiogenesis_ecm_remodeling_like",
    "M24": "developmental_neural_signaling_like",
    "M40": "proliferation_cell_cycle_deviation",
    "M28": "secondary_program_M28",
    "M38": "secondary_program_M38",
    "M25": "secondary_program_M25",
    "M17": "stress_hypoxia_like",
}

MODEL_CONTRASTS = [
    {
        "contrast": "incremental_module_beyond_disjoint_proliferation",
        "model_a": "module_plus_disjoint_proliferation",
        "model_b": "disjoint_proliferation_only",
    },
    {
        "contrast": "module_vs_disjoint_proliferation",
        "model_a": "module_only",
        "model_b": "disjoint_proliferation_only",
    },
    {
        "contrast": "joint_model_vs_module_only",
        "model_a": "module_plus_disjoint_proliferation",
        "model_b": "module_only",
    },
    {
        "contrast": "disjoint_residual_vs_module_only",
        "model_a": "residual_to_disjoint_proliferation",
        "model_b": "module_only",
    },
    {
        "contrast": "disjoint_residual_vs_original_residual",
        "model_a": "residual_to_disjoint_proliferation",
        "model_b": "residual_to_original_proliferation",
    },
    {
        "contrast": "weight_adjusted_residual_vs_unadjusted_residual",
        "model_a": "residual_to_disjoint_proliferation_and_weight",
        "model_b": "residual_to_disjoint_proliferation",
    },
]

OUTPUT_REPEAT_SCORES = RESULTS_DIR / "GSE238110_repeated_cv_program_scores_by_repeat.csv"
OUTPUT_CONTRASTS = RESULTS_DIR / "GSE238110_repeated_cv_program_model_contrasts.csv"
OUTPUT_CONTRAST_SUMMARY = RESULTS_DIR / "GSE238110_repeated_cv_program_model_contrast_summary.csv"
OUTPUT_MANIFEST = RESULTS_DIR / "GSE238110_frozen_canine_transfer_program_manifest.csv"
OUTPUT_STRICT_WEIGHTS = RESULTS_DIR / "GSE238110_frozen_transfer_gene_weights_strict.csv"
OUTPUT_BROAD_WEIGHTS = RESULTS_DIR / "GSE238110_frozen_transfer_gene_weights_broad.csv"
OUTPUT_STRICT_GMT = RESULTS_DIR / "GSE238110_frozen_transfer_programs_strict.gmt"
OUTPUT_BROAD_GMT = RESULTS_DIR / "GSE238110_frozen_transfer_programs_broad.gmt"
OUTPUT_SCORING_SPEC = RESULTS_DIR / "GSE238110_frozen_transfer_scoring_specification.csv"
OUTPUT_FREEZE_JSON = RESULTS_DIR / "GSE238110_frozen_transfer_program_freeze.json"
OUTPUT_README = RESULTS_DIR / "GSE238110_frozen_transfer_programs_README.txt"


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


def clean_gene_symbol(gene):
    gene = str(gene)
    tail = gene.rsplit("_", 1)[-1]
    return gene.rsplit("_", 1)[0] if tail.isdigit() else gene


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


def safe_wilcoxon(values, alternative="two-sided"):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    if np.allclose(values, 0):
        return 1.0
    try:
        return float(
            stats.wilcoxon(
                values,
                alternative=alternative,
                zero_method="wilcox",
            ).pvalue
        )
    except Exception:
        return np.nan


def safe_sign_test(values, alternative="two-sided"):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[~np.isclose(values, 0)]
    if len(values) == 0:
        return np.nan
    n_positive = int((values > 0).sum())
    try:
        return float(
            stats.binomtest(
                n_positive,
                n=len(values),
                p=0.5,
                alternative=alternative,
            ).pvalue
        )
    except Exception:
        return np.nan


def bootstrap_mean_ci(values, reps=BOOTSTRAP_REPS, seed=RANDOM_SEED):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(reps, len(values)), replace=True)
    means = sampled.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def corrected_repeated_kfold_ttest(fold_deltas):
    values = np.asarray(fold_deltas, dtype=float)
    values = values[np.isfinite(values)]
    result = {
        "corrected_t": np.nan,
        "corrected_t_df": np.nan,
        "corrected_t_p_two_sided": np.nan,
        "corrected_standard_error": np.nan,
        "correction_factor": np.nan,
    }
    if len(values) < 2:
        return result
    variance = np.var(values, ddof=1)
    test_train_ratio = 1.0 / (N_SPLITS - 1)
    correction = (1.0 / len(values)) + test_train_ratio
    standard_error = np.sqrt(correction * variance)
    if not np.isfinite(standard_error) or standard_error == 0:
        return result
    t_stat = float(np.mean(values) / standard_error)
    df = len(values) - 1
    p_value = float(2 * stats.t.sf(abs(t_stat), df=df))
    result.update(
        {
            "corrected_t": t_stat,
            "corrected_t_df": df,
            "corrected_t_p_two_sided": p_value,
            "corrected_standard_error": float(standard_error),
            "correction_factor": float(correction),
        }
    )
    return result


def repeat_level_scores(cv_folds):
    use = cv_folds.copy()
    use["c_index"] = pd.to_numeric(use["c_index"], errors="coerce")
    use = use[np.isfinite(use["c_index"])].copy()
    return (
        use.groupby(
            ["endpoint", "module_label", "model", "repeat"],
            dropna=False,
        )
        .agg(
            n_valid_folds=("c_index", "count"),
            repeat_mean_c_index=("c_index", "mean"),
            repeat_median_c_index=("c_index", "median"),
            repeat_min_c_index=("c_index", "min"),
            repeat_max_c_index=("c_index", "max"),
        )
        .reset_index()
    )


def summarize_paired_values(repeat_deltas, fold_deltas, seed):
    repeat_deltas = np.asarray(repeat_deltas, dtype=float)
    repeat_deltas = repeat_deltas[np.isfinite(repeat_deltas)]
    fold_deltas = np.asarray(fold_deltas, dtype=float)
    fold_deltas = fold_deltas[np.isfinite(fold_deltas)]
    ci_low, ci_high = bootstrap_mean_ci(repeat_deltas, seed=seed)
    corrected = corrected_repeated_kfold_ttest(fold_deltas)
    return {
        "n_repeats": len(repeat_deltas),
        "n_fold_pairs": len(fold_deltas),
        "mean_delta_c_index": float(np.mean(repeat_deltas)) if len(repeat_deltas) else np.nan,
        "median_delta_c_index": float(np.median(repeat_deltas)) if len(repeat_deltas) else np.nan,
        "std_delta_c_index": float(np.std(repeat_deltas, ddof=1)) if len(repeat_deltas) > 1 else np.nan,
        "bootstrap_repeat_mean_ci_low": ci_low,
        "bootstrap_repeat_mean_ci_high": ci_high,
        "fraction_repeat_deltas_positive": float((repeat_deltas > 0).mean()) if len(repeat_deltas) else np.nan,
        "wilcoxon_repeat_p_two_sided": safe_wilcoxon(repeat_deltas),
        "wilcoxon_repeat_p_greater": safe_wilcoxon(repeat_deltas, alternative="greater"),
        "sign_repeat_p_two_sided": safe_sign_test(repeat_deltas),
        "sign_repeat_p_greater": safe_sign_test(repeat_deltas, alternative="greater"),
        **corrected,
    }


def build_model_contrasts(cv_folds, repeat_scores):
    rows = []
    seed_counter = 0
    endpoints = sorted(cv_folds["endpoint"].dropna().astype(str).unique())
    modules = sorted(cv_folds["module_label"].dropna().astype(str).unique())

    for endpoint in endpoints:
        for module_label in modules:
            fold_part = cv_folds[
                cv_folds["endpoint"].astype(str).eq(endpoint)
                & cv_folds["module_label"].astype(str).eq(module_label)
            ].copy()
            repeat_part = repeat_scores[
                repeat_scores["endpoint"].astype(str).eq(endpoint)
                & repeat_scores["module_label"].astype(str).eq(module_label)
            ].copy()

            for spec in MODEL_CONTRASTS:
                model_a = spec["model_a"]
                model_b = spec["model_b"]

                fold_a = fold_part[fold_part["model"].eq(model_a)][
                    ["repeat", "fold", "c_index"]
                ].rename(columns={"c_index": "c_index_a"})
                fold_b = fold_part[fold_part["model"].eq(model_b)][
                    ["repeat", "fold", "c_index"]
                ].rename(columns={"c_index": "c_index_b"})
                fold_pairs = fold_a.merge(fold_b, on=["repeat", "fold"], how="inner")
                fold_pairs["delta"] = fold_pairs["c_index_a"] - fold_pairs["c_index_b"]

                repeat_a = repeat_part[repeat_part["model"].eq(model_a)][
                    ["repeat", "repeat_mean_c_index"]
                ].rename(columns={"repeat_mean_c_index": "repeat_c_index_a"})
                repeat_b = repeat_part[repeat_part["model"].eq(model_b)][
                    ["repeat", "repeat_mean_c_index"]
                ].rename(columns={"repeat_mean_c_index": "repeat_c_index_b"})
                repeat_pairs = repeat_a.merge(repeat_b, on="repeat", how="inner")
                repeat_pairs["delta"] = (
                    repeat_pairs["repeat_c_index_a"]
                    - repeat_pairs["repeat_c_index_b"]
                )

                if repeat_pairs.empty:
                    continue

                stats_row = summarize_paired_values(
                    repeat_deltas=repeat_pairs["delta"],
                    fold_deltas=fold_pairs["delta"],
                    seed=RANDOM_SEED + seed_counter,
                )
                seed_counter += 1
                rows.append(
                    {
                        "endpoint": endpoint,
                        "module_label": module_label,
                        "contrast": spec["contrast"],
                        "model_a": model_a,
                        "model_b": model_b,
                        "mean_repeat_c_index_a": float(repeat_pairs["repeat_c_index_a"].mean()),
                        "mean_repeat_c_index_b": float(repeat_pairs["repeat_c_index_b"].mean()),
                        **stats_row,
                    }
                )

            residual = repeat_part[
                repeat_part["model"].eq("residual_to_disjoint_proliferation")
            ].copy()
            residual_folds = fold_part[
                fold_part["model"].eq("residual_to_disjoint_proliferation")
            ].copy()
            if not residual.empty:
                repeat_delta = residual["repeat_mean_c_index"] - 0.5
                fold_delta = pd.to_numeric(
                    residual_folds["c_index"], errors="coerce"
                ) - 0.5
                stats_row = summarize_paired_values(
                    repeat_deltas=repeat_delta,
                    fold_deltas=fold_delta,
                    seed=RANDOM_SEED + seed_counter,
                )
                seed_counter += 1
                rows.append(
                    {
                        "endpoint": endpoint,
                        "module_label": module_label,
                        "contrast": "disjoint_residual_above_chance",
                        "model_a": "residual_to_disjoint_proliferation",
                        "model_b": "chance_0.50",
                        "mean_repeat_c_index_a": float(residual["repeat_mean_c_index"].mean()),
                        "mean_repeat_c_index_b": 0.5,
                        **stats_row,
                    }
                )

    contrasts = pd.DataFrame(rows)
    if contrasts.empty:
        return contrasts

    for p_col in [
        "wilcoxon_repeat_p_two_sided",
        "wilcoxon_repeat_p_greater",
        "corrected_t_p_two_sided",
    ]:
        contrasts[f"{p_col}_bh_q"] = np.nan
        for endpoint in contrasts["endpoint"].dropna().unique():
            mask = contrasts["endpoint"].eq(endpoint)
            contrasts.loc[mask, f"{p_col}_bh_q"] = bh_adjust(
                contrasts.loc[mask, p_col]
            )

    contrasts["descriptive_support"] = np.select(
        [
            (contrasts["mean_delta_c_index"] >= 0.02)
            & (contrasts["fraction_repeat_deltas_positive"] >= 0.75)
            & (contrasts["bootstrap_repeat_mean_ci_low"] > 0),
            (contrasts["mean_delta_c_index"] >= 0.01)
            & (contrasts["fraction_repeat_deltas_positive"] >= 0.60),
            (contrasts["mean_delta_c_index"] > 0),
        ],
        [
            "strong_descriptive_support",
            "moderate_descriptive_support",
            "weak_positive_descriptive_support",
        ],
        default="no_consistent_positive_support",
    )
    return contrasts


def build_contrast_summary(contrasts):
    if contrasts.empty:
        return pd.DataFrame()
    keep = contrasts[
        contrasts["module_label"].isin(
            PRIMARY_CLEAN_MODULES
            + PRIMARY_PROLIFERATION_AXIS_MODULES
            + SECONDARY_SENSITIVITY_MODULES
        )
    ].copy()
    return keep.sort_values(
        ["endpoint", "contrast", "mean_delta_c_index"],
        ascending=[True, True, False],
    )


def detect_module_columns(module_membership):
    module_candidates = ["module_label", "module", "module_id", "cluster", "cluster_id"]
    gene_candidates = ["gene", "gene_id", "expression_gene", "gene_column"]
    module_col = next((c for c in module_candidates if c in module_membership.columns), None)
    gene_col = next((c for c in gene_candidates if c in module_membership.columns), None)
    if module_col is None or gene_col is None:
        raise ValueError("Could not identify module and gene columns in module membership.")
    return module_col, gene_col


def filter_full_cohort_membership(module_membership):
    for col in ["analysis_scope", "scope", "analysis_type", "source", "data_scope"]:
        if col in module_membership.columns:
            mask = module_membership[col].astype(str).str.lower().str.contains("full", na=False)
            if mask.any():
                print(f"Full-cohort module membership selected using column: {col}")
                return module_membership.loc[mask].copy()
    if "fold" in module_membership.columns:
        fold = pd.to_numeric(module_membership["fold"], errors="coerce")
        if fold.isna().any():
            print("Full-cohort module membership selected using missing fold values.")
            return module_membership.loc[fold.isna()].copy()
    return module_membership.copy()


def get_module_score_mapping(module_scores):
    mapping = {}
    for col in module_scores.columns:
        value = str(col)
        if value.startswith("module_") and value.endswith("_score"):
            mapping[value[len("module_") : -len("_score")]] = col
        else:
            mapping[value] = col
    return mapping


def resolve_expression_gene(value, expression_columns, symbol_to_columns):
    value = str(value)
    if value in expression_columns:
        return value
    symbol = clean_gene_symbol(value).upper()
    candidates = symbol_to_columns.get(symbol, [])
    if len(candidates) == 1:
        return candidates[0]
    return ""


def build_module_gene_map(module_membership, expression, module_scores):
    membership = filter_full_cohort_membership(module_membership)
    module_col, gene_col = detect_module_columns(membership)
    score_mapping = get_module_score_mapping(module_scores)
    valid_labels = set(score_mapping)
    membership[module_col] = membership[module_col].astype(str)
    membership = membership[membership[module_col].isin(valid_labels)].copy()

    expression_columns = set(expression.columns.astype(str))
    symbol_to_columns = {}
    for col in expression.columns:
        symbol_to_columns.setdefault(clean_gene_symbol(col).upper(), []).append(col)

    module_gene_map = {}
    for module_label, part in membership.groupby(module_col):
        genes = []
        for value in part[gene_col].dropna().astype(str):
            resolved = resolve_expression_gene(value, expression_columns, symbol_to_columns)
            if resolved:
                genes.append(resolved)
        module_gene_map[str(module_label)] = list(dict.fromkeys(genes))
    return module_gene_map, score_mapping


def detect_ortholog_columns(ortholog):
    dog_candidates = ["gene", "canine_gene", "dog_gene", "expression_gene"]
    human_candidates = ["human_gene_symbol", "human_symbol", "human_gene"]
    status_candidates = ["ortholog_qc_status", "mapping_status", "ortholog_status"]
    dog_col = next((c for c in dog_candidates if c in ortholog.columns), None)
    human_col = next((c for c in human_candidates if c in ortholog.columns), None)
    status_col = next((c for c in status_candidates if c in ortholog.columns), None)
    if dog_col is None or human_col is None:
        raise ValueError("Could not identify dog and human ortholog columns.")
    return dog_col, human_col, status_col


def standardize_expression(expression, genes):
    x = expression[genes].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(axis=0))
    means = x.mean(axis=0)
    stds = x.std(axis=0).replace(0, np.nan)
    z = (x - means) / stds
    valid = z.columns[z.notna().all(axis=0)]
    return z[valid]


def safe_corr(a, b):
    frame = pd.concat([pd.Series(a, name="a"), pd.Series(b, name="b")], axis=1).dropna()
    if frame.shape[0] < 5 or frame["a"].std() == 0 or frame["b"].std() == 0:
        return np.nan
    return float(frame["a"].corr(frame["b"]))


def endpoint_module_row(module_associations, module_label, endpoint):
    if module_associations.empty:
        return pd.Series(dtype=object)
    use = module_associations.copy()
    use["module_label"] = use["module_label"].astype(str)
    use["endpoint"] = use["endpoint"].astype(str).str.upper()
    part = use[
        use["module_label"].eq(str(module_label))
        & use["endpoint"].eq(str(endpoint).upper())
    ].copy()
    if part.empty:
        return pd.Series(dtype=object)
    if "p" in part.columns:
        part["p"] = pd.to_numeric(part["p"], errors="coerce")
        part = part.sort_values("p", na_position="last")
    return part.iloc[0]


def get_numeric(row, candidates, default=np.nan):
    for col in candidates:
        if col in row.index:
            value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if np.isfinite(value):
                return float(value)
    return default


def assign_validation_tier(module_label):
    if module_label in PRIMARY_CLEAN_MODULES:
        return "primary_clean_non_proliferation"
    if module_label in PRIMARY_PROLIFERATION_AXIS_MODULES:
        return "primary_proliferation_deviation_axis"
    if module_label in SECONDARY_SENSITIVITY_MODULES:
        return "secondary_sensitivity"
    return "exploratory"


def build_weights_and_manifest(
    expression,
    module_gene_map,
    module_scores,
    module_associations,
    priority,
    ortholog,
    audit,
    cv_summary,
    decision,
):
    dog_col, human_col, status_col = detect_ortholog_columns(ortholog)
    ortholog = ortholog.copy()
    ortholog[dog_col] = ortholog[dog_col].astype(str)
    ortholog["dog_symbol_key"] = ortholog[dog_col].map(clean_gene_symbol).str.upper()
    ortholog[human_col] = ortholog[human_col].astype(str).replace({"nan": "", "None": ""})
    if status_col is None:
        ortholog["_status"] = "unknown_mapping_status"
        status_col = "_status"

    score_mapping = get_module_score_mapping(module_scores)
    weight_rows = []
    manifest_rows = []

    for module_label, genes in sorted(module_gene_map.items()):
        genes = [g for g in genes if g in expression.columns]
        if len(genes) < 3:
            continue

        z = standardize_expression(expression, genes)
        genes_used = list(z.columns)
        if len(genes_used) < 3:
            continue

        pca = PCA(n_components=1, random_state=RANDOM_SEED)
        score = pd.Series(pca.fit_transform(z).ravel(), index=z.index)
        raw_loadings = pd.Series(pca.components_[0], index=genes_used)

        score_col = score_mapping.get(module_label)
        orientation_corr = np.nan
        if score_col in module_scores.columns:
            reference = module_scores[score_col].reindex(score.index)
            orientation_corr = safe_corr(score, reference)
            if np.isfinite(orientation_corr) and orientation_corr < 0:
                score = -score
                raw_loadings = -raw_loadings

        dfi_row = endpoint_module_row(module_associations, module_label, "DFI")
        os_row = endpoint_module_row(module_associations, module_label, "OS")
        dfi_coef = get_numeric(dfi_row, ["coef", "module_coef"])
        os_coef = get_numeric(os_row, ["coef", "module_coef"])
        if not np.isfinite(dfi_coef):
            dfi_hr = get_numeric(dfi_row, ["hr", "hazard_ratio", "exp(coef)"])
            if np.isfinite(dfi_hr) and dfi_hr > 0:
                dfi_coef = float(np.log(dfi_hr))
        if not np.isfinite(os_coef):
            os_hr = get_numeric(os_row, ["hr", "hazard_ratio", "exp(coef)"])
            if np.isfinite(os_hr) and os_hr > 0:
                os_coef = float(np.log(os_hr))
        risk_reference_coef = dfi_coef if np.isfinite(dfi_coef) else os_coef
        risk_multiplier = 1.0 if not np.isfinite(risk_reference_coef) or risk_reference_coef >= 0 else -1.0
        risk_loadings = raw_loadings * risk_multiplier

        module_weight_rows = []
        for canine_gene, loading in risk_loadings.items():
            dog_symbol = clean_gene_symbol(canine_gene).upper()
            mappings = ortholog[ortholog["dog_symbol_key"].eq(dog_symbol)].copy()
            if mappings.empty:
                module_weight_rows.append(
                    {
                        "module_label": module_label,
                        "canine_gene": canine_gene,
                        "canine_gene_symbol": dog_symbol,
                        "human_gene_symbol": "",
                        "ortholog_qc_status": "not_found_in_ortholog_table",
                        "raw_pca_loading": float(raw_loadings.loc[canine_gene]),
                        "risk_oriented_loading": float(loading),
                    }
                )
                continue
            for _, mapping in mappings.iterrows():
                human_symbol = str(mapping[human_col]).strip().upper()
                module_weight_rows.append(
                    {
                        "module_label": module_label,
                        "canine_gene": canine_gene,
                        "canine_gene_symbol": dog_symbol,
                        "human_gene_symbol": human_symbol,
                        "ortholog_qc_status": str(mapping[status_col]),
                        "raw_pca_loading": float(raw_loadings.loc[canine_gene]),
                        "risk_oriented_loading": float(loading),
                    }
                )

        module_weights = pd.DataFrame(module_weight_rows)
        module_weights["absolute_risk_loading"] = module_weights["risk_oriented_loading"].abs()
        module_weights["is_strict_mapping"] = module_weights["ortholog_qc_status"].eq(
            "strict_symbol_concordant_one_to_one"
        ) & module_weights["human_gene_symbol"].ne("")
        module_weights["is_broad_mapping"] = (
            module_weights["human_gene_symbol"].ne("")
            & ~module_weights["ortholog_qc_status"].eq("not_transferable_or_unmapped")
            & ~module_weights["ortholog_qc_status"].eq("not_found_in_ortholog_table")
        )
        weight_rows.append(module_weights)

        strict = module_weights[module_weights["is_strict_mapping"]].copy()
        broad = module_weights[module_weights["is_broad_mapping"]].copy()

        priority_row = pd.Series(dtype=object)
        if not priority.empty and "module_label" in priority.columns:
            part = priority[priority["module_label"].astype(str).eq(module_label)]
            if not part.empty:
                priority_row = part.iloc[0]

        audit_row = pd.Series(dtype=object)
        if not audit.empty:
            part = audit[audit["module_label"].astype(str).eq(module_label)]
            if not part.empty:
                audit_row = part.iloc[0]

        manifest_rows.append(
            {
                "module_label": module_label,
                "validation_tier": assign_validation_tier(module_label),
                "multiplicity_family": (
                    "primary_confirmatory"
                    if module_label in PRIMARY_CLEAN_MODULES + PRIMARY_PROLIFERATION_AXIS_MODULES
                    else "secondary_prespecified"
                    if module_label in SECONDARY_SENSITIVITY_MODULES
                    else "exploratory"
                ),
                "provisional_program_label": PROVISIONAL_PROGRAM_LABELS.get(
                    module_label, f"exploratory_program_{module_label}"
                ),
                "program_label_requires_enrichment_confirmation": True,
                "canine_primary_endpoint": "DFI",
                "canine_secondary_endpoint": "OS_concordance",
                "positive_score_interpretation": "higher_score_higher_canine_DFI_risk",
                "n_canine_genes_used_for_pca": len(genes_used),
                "canine_pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
                "canine_pca_orientation_correlation": orientation_corr,
                "dfi_full_cohort_coef": dfi_coef,
                "os_full_cohort_coef": os_coef,
                "risk_orientation_multiplier": risk_multiplier,
                "n_strict_human_genes": strict["human_gene_symbol"].replace("", np.nan).nunique(),
                "n_broad_human_genes": broad["human_gene_symbol"].replace("", np.nan).nunique(),
                "strict_transfer_eligible": strict["human_gene_symbol"].replace("", np.nan).nunique() >= MIN_STRICT_HUMAN_GENES,
                "module_transfer_qc_tier": priority_row.get("module_transfer_qc_tier", ""),
                "transfer_priority_score": priority_row.get("transfer_priority_score", np.nan),
                "fraction_strict_symbol_concordant": priority_row.get("fraction_strict_symbol_concordant", np.nan),
                "fraction_broad_transferable": priority_row.get("fraction_broad_transferable", np.nan),
                "raw_module_proliferation_correlation": audit_row.get("raw_module_proliferation_correlation", np.nan),
                "orthogonal_variance_fraction_1_minus_r2": audit_row.get("orthogonal_variance_fraction_1_minus_r2", np.nan),
                "n_overlap_symbols_with_proliferation": audit_row.get("n_overlap_symbols", np.nan),
                "primary_human_score": "strict_one_to_one_signed_mean_z",
                "secondary_human_score": "strict_one_to_one_canine_pca_weighted_z",
                "sensitivity_human_scores": "broad_mapped_mean_z;human_cohort_pc1;residual_to_disjoint_proliferation",
                "frozen_after_canine_script": "20_proliferation_overlap_crossfit_sensitivity.py",
            }
        )

    all_weights = pd.concat(weight_rows, axis=0, ignore_index=True)
    manifest = pd.DataFrame(manifest_rows)

    for endpoint in ["DFI", "OS"]:
        if not cv_summary.empty:
            part = cv_summary.copy()
            part["endpoint"] = part["endpoint"].astype(str).str.upper()
            part = part[part["endpoint"].eq(endpoint)].copy()
            for model in [
                "module_only",
                "module_plus_disjoint_proliferation",
                "residual_to_disjoint_proliferation",
                "residual_to_disjoint_proliferation_and_weight",
            ]:
                model_part = part[part["model"].eq(model)][
                    ["module_label", "mean_c_index", "std_c_index", "fraction_above_0_50"]
                ].copy()
                model_part = model_part.rename(
                    columns={
                        "mean_c_index": f"{endpoint.lower()}_{model}_mean_c_index",
                        "std_c_index": f"{endpoint.lower()}_{model}_std_c_index",
                        "fraction_above_0_50": f"{endpoint.lower()}_{model}_fraction_above_0_50",
                    }
                )
                manifest = manifest.merge(model_part, on="module_label", how="left")

    if not decision.empty:
        dfi_decision = decision[decision["endpoint"].astype(str).str.upper().eq("DFI")][
            ["module_label", "recommended_role"]
        ].rename(columns={"recommended_role": "script20_dfi_recommended_role"})
        os_decision = decision[decision["endpoint"].astype(str).str.upper().eq("OS")][
            ["module_label", "recommended_role"]
        ].rename(columns={"recommended_role": "script20_os_recommended_role"})
        manifest = manifest.merge(dfi_decision, on="module_label", how="left")
        manifest = manifest.merge(os_decision, on="module_label", how="left")

    manifest["manual_freeze_reason"] = np.select(
        [
            manifest["module_label"].eq("M34"),
            manifest["module_label"].eq("M11"),
            manifest["module_label"].eq("M24"),
            manifest["module_label"].eq("M40"),
            manifest["module_label"].isin(SECONDARY_SENSITIVITY_MODULES),
        ],
        [
            "strongest clean cross-endpoint non-proliferation program",
            "high transfer readiness and independent DFI signal",
            "compact high-readiness program with cross-endpoint signal",
            "proliferation-dominant axis with reproducible cross-fitted residual component",
            "prespecified sensitivity program based on performance or transferability",
        ],
        default="exploratory program retained without confirmatory status",
    )

    manifest = manifest.sort_values(
        ["multiplicity_family", "validation_tier", "module_label"]
    ).reset_index(drop=True)
    return all_weights, manifest


def deduplicate_human_weights(weights, mapping_flag):
    use = weights[weights[mapping_flag]].copy()
    use = use[use["human_gene_symbol"].astype(str).str.len() > 0].copy()
    if use.empty:
        return use
    use["human_symbol_duplicate_count"] = use.groupby(
        ["module_label", "human_gene_symbol"]
    )["human_gene_symbol"].transform("size")
    use = use.sort_values(
        ["module_label", "human_gene_symbol", "absolute_risk_loading"],
        ascending=[True, True, False],
    )
    use = use.drop_duplicates(["module_label", "human_gene_symbol"], keep="first")
    use["normalized_abs_sum_weight"] = use.groupby("module_label")[
        "risk_oriented_loading"
    ].transform(lambda x: x / x.abs().sum() if x.abs().sum() > 0 else np.nan)
    return use.reset_index(drop=True)


def write_gmt(weights, path, suffix):
    lines = []
    for module_label, part in weights.groupby("module_label"):
        genes = part["human_gene_symbol"].dropna().astype(str).drop_duplicates().tolist()
        if not genes:
            continue
        name = f"CANINE_{module_label}_{suffix}"
        lines.append("\t".join([name, "frozen_canine_ortholog_transfer"] + genes))
    path.write_text("\n".join(lines), encoding="utf-8")


def build_scoring_specification(manifest):
    rows = []
    for _, row in manifest.iterrows():
        rows.extend(
            [
                {
                    "module_label": row["module_label"],
                    "validation_tier": row["validation_tier"],
                    "score_name": "strict_one_to_one_unweighted_mean_z",
                    "analysis_role": "primary",
                    "gene_mapping": "strict_symbol_concordant_one_to_one",
                    "within_cohort_preprocessing": "z-score each available gene across human samples",
                    "score_formula": "arithmetic mean of signed gene z-scores using the sign of each frozen risk-oriented canine PCA loading",
                    "minimum_gene_rule": "at least 3 genes and at least 50% of frozen strict genes",
                    "outcome_use": "no outcome information used to construct the score",
                },
                {
                    "module_label": row["module_label"],
                    "validation_tier": row["validation_tier"],
                    "score_name": "strict_one_to_one_canine_pca_weighted_z",
                    "analysis_role": "secondary_zero_shot",
                    "gene_mapping": "strict_symbol_concordant_one_to_one",
                    "within_cohort_preprocessing": "z-score each available gene across human samples",
                    "score_formula": "sum of gene z-scores multiplied by frozen risk-oriented canine PCA loadings normalized by absolute weight sum",
                    "minimum_gene_rule": "at least 3 genes and at least 50% of frozen strict genes",
                    "outcome_use": "no outcome information used to construct the score",
                },
                {
                    "module_label": row["module_label"],
                    "validation_tier": row["validation_tier"],
                    "score_name": "broad_mapped_unweighted_mean_z",
                    "analysis_role": "mapping_sensitivity",
                    "gene_mapping": "broad transferable mapping including review-status mappings",
                    "within_cohort_preprocessing": "z-score each available gene across human samples",
                    "score_formula": "arithmetic mean of risk-oriented mapped gene z-scores",
                    "minimum_gene_rule": "at least 3 genes and at least 50% of frozen broad genes",
                    "outcome_use": "no outcome information used to construct the score",
                },
            ]
        )
        if row["module_label"] == "M40":
            rows.append(
                {
                    "module_label": row["module_label"],
                    "validation_tier": row["validation_tier"],
                    "score_name": "residual_to_disjoint_proliferation",
                    "analysis_role": "mechanistic_sensitivity",
                    "gene_mapping": "strict module genes plus strict proliferation genes after removing overlap",
                    "within_cohort_preprocessing": "construct both scores without outcomes; residualize M40 score on disjoint proliferation score",
                    "score_formula": "standardized residual from linear regression of M40 score on disjoint proliferation score",
                    "minimum_gene_rule": "same minimum-gene rules for both component scores",
                    "outcome_use": "no outcome information used during residualization",
                }
            )
    return pd.DataFrame(rows)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_readme(manifest):
    primary = manifest[manifest["multiplicity_family"].eq("primary_confirmatory")]
    secondary = manifest[manifest["multiplicity_family"].eq("secondary_prespecified")]
    lines = [
        "Frozen canine-to-human osteosarcoma transfer programs",
        "",
        "Canine primary endpoint: DFI.",
        "Canine OS is a concordance/sensitivity endpoint.",
        "Program definitions, gene membership, score orientation, and validation tiers are frozen after script 20.",
        "No human outcome may be used to change module membership, gene weights, score direction, or tier assignment.",
        "",
        "Primary confirmatory programs:",
    ]
    for _, row in primary.iterrows():
        lines.append(
            f"  {row['module_label']}: {row['provisional_program_label']} | {row['validation_tier']}"
        )
    lines.append("")
    lines.append("Secondary prespecified programs:")
    for _, row in secondary.iterrows():
        lines.append(
            f"  {row['module_label']}: {row['provisional_program_label']} | {row['validation_tier']}"
        )
    lines.extend(
        [
            "",
            "Primary human score: strict one-to-one signed mean z-score using frozen canine loading signs.",
            "Secondary zero-shot score: frozen canine PCA-weighted score.",
            "M40 residualized score is a mechanistic sensitivity analysis, not a replacement for raw M40 or proliferation scores.",
            "TARGET-OS and GSE21257 must be treated as external datasets; cohort-specific preprocessing may not use outcomes.",
            "External validation, not canine repeated-CV p-values, determines translational support.",
        ]
    )
    OUTPUT_README.write_text("\n".join(lines), encoding="utf-8")


def main():
    print("=" * 80)
    print("Finalize canine transfer programs")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Processed directory: {PROCESSED_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print("")
    print("Design:")
    print("  Summarize paired repeated-CV model contrasts without treating 100 folds as independent evidence.")
    print("  Freeze confirmatory and sensitivity module tiers before human outcome analysis.")
    print("  Export strict and broad human ortholog gene sets and frozen canine PCA weights.")
    print("")

    expression = read_required_csv(PROCESSED_DIR / EXPRESSION_FILE, index_col=0)
    module_membership = read_required_csv(RESULTS_DIR / MODULE_MEMBERSHIP_FILE)
    module_scores = read_required_csv(RESULTS_DIR / MODULE_SCORE_FILE, index_col=0)
    module_associations = read_required_csv(RESULTS_DIR / MODULE_ASSOCIATION_FILE)
    priority = read_required_csv(RESULTS_DIR / MODULE_PRIORITY_FILE)
    ortholog = read_required_csv(RESULTS_DIR / ORTHOLOG_FILE)
    audit = read_required_csv(RESULTS_DIR / OVERLAP_AUDIT_FILE)
    cv_folds = read_required_csv(RESULTS_DIR / CV_FOLD_FILE)
    cv_summary = read_required_csv(RESULTS_DIR / CV_SUMMARY_FILE)
    decision = read_required_csv(RESULTS_DIR / DECISION_FILE)

    common_samples = expression.index.intersection(module_scores.index)
    expression = expression.loc[common_samples].copy()
    module_scores = module_scores.loc[common_samples].copy()

    print("")
    print("Matched data:")
    print(f"  Expression matrix: {expression.shape}")
    print(f"  Module score matrix: {module_scores.shape}")
    print(f"  Repeated-CV fold rows: {cv_folds.shape[0]}")

    repeat_scores = repeat_level_scores(cv_folds)
    contrasts = build_model_contrasts(cv_folds, repeat_scores)
    contrast_summary = build_contrast_summary(contrasts)

    module_gene_map, _ = build_module_gene_map(
        module_membership=module_membership,
        expression=expression,
        module_scores=module_scores,
    )
    all_weights, manifest = build_weights_and_manifest(
        expression=expression,
        module_gene_map=module_gene_map,
        module_scores=module_scores,
        module_associations=module_associations,
        priority=priority,
        ortholog=ortholog,
        audit=audit,
        cv_summary=cv_summary,
        decision=decision,
    )

    strict_weights = deduplicate_human_weights(all_weights, "is_strict_mapping")
    broad_weights = deduplicate_human_weights(all_weights, "is_broad_mapping")
    scoring_spec = build_scoring_specification(manifest)

    repeat_scores.to_csv(OUTPUT_REPEAT_SCORES, index=False)
    contrasts.to_csv(OUTPUT_CONTRASTS, index=False)
    contrast_summary.to_csv(OUTPUT_CONTRAST_SUMMARY, index=False)
    manifest.to_csv(OUTPUT_MANIFEST, index=False)
    strict_weights.to_csv(OUTPUT_STRICT_WEIGHTS, index=False)
    broad_weights.to_csv(OUTPUT_BROAD_WEIGHTS, index=False)
    scoring_spec.to_csv(OUTPUT_SCORING_SPEC, index=False)
    write_gmt(strict_weights, OUTPUT_STRICT_GMT, "STRICT")
    write_gmt(broad_weights, OUTPUT_BROAD_GMT, "BROAD")
    write_readme(manifest)

    freeze = {
        "frozen_after_script": "20_proliferation_overlap_crossfit_sensitivity.py",
        "primary_clean_modules": PRIMARY_CLEAN_MODULES,
        "primary_proliferation_axis_modules": PRIMARY_PROLIFERATION_AXIS_MODULES,
        "secondary_sensitivity_modules": SECONDARY_SENSITIVITY_MODULES,
        "primary_canine_endpoint": "DFI",
        "secondary_canine_endpoint": "OS_concordance",
        "primary_human_score": "strict_one_to_one_signed_mean_z",
        "secondary_human_score": "strict_one_to_one_canine_pca_weighted_z",
        "m40_residual_role": "mechanistic_sensitivity",
        "files": {},
    }

    frozen_paths = [
        OUTPUT_REPEAT_SCORES,
        OUTPUT_CONTRASTS,
        OUTPUT_CONTRAST_SUMMARY,
        OUTPUT_MANIFEST,
        OUTPUT_STRICT_WEIGHTS,
        OUTPUT_BROAD_WEIGHTS,
        OUTPUT_STRICT_GMT,
        OUTPUT_BROAD_GMT,
        OUTPUT_SCORING_SPEC,
        OUTPUT_README,
    ]
    for path in frozen_paths:
        freeze["files"][path.name] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    OUTPUT_FREEZE_JSON.write_text(json.dumps(freeze, indent=2), encoding="utf-8")

    print("")
    print("=" * 80)
    print("Frozen transfer program manifest")
    print("=" * 80)
    manifest_cols = [
        "module_label",
        "validation_tier",
        "multiplicity_family",
        "provisional_program_label",
        "n_canine_genes_used_for_pca",
        "n_strict_human_genes",
        "n_broad_human_genes",
        "raw_module_proliferation_correlation",
        "dfi_residual_to_disjoint_proliferation_mean_c_index",
        "os_residual_to_disjoint_proliferation_mean_c_index",
        "manual_freeze_reason",
    ]
    manifest_cols = [c for c in manifest_cols if c in manifest.columns]
    print(manifest[manifest_cols].to_string(index=False))

    print("")
    print("=" * 80)
    print("Key repeated-CV contrasts for frozen programs")
    print("=" * 80)
    key = contrast_summary[
        contrast_summary["contrast"].isin(
            [
                "incremental_module_beyond_disjoint_proliferation",
                "disjoint_residual_above_chance",
                "weight_adjusted_residual_vs_unadjusted_residual",
            ]
        )
    ].copy()
    key_cols = [
        "endpoint",
        "module_label",
        "contrast",
        "mean_repeat_c_index_a",
        "mean_repeat_c_index_b",
        "mean_delta_c_index",
        "bootstrap_repeat_mean_ci_low",
        "bootstrap_repeat_mean_ci_high",
        "fraction_repeat_deltas_positive",
        "wilcoxon_repeat_p_greater",
        "corrected_t_p_two_sided",
        "descriptive_support",
    ]
    key_cols = [c for c in key_cols if c in key.columns]
    print(key[key_cols].to_string(index=False))

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("Repeated cross-validation splits overlap; contrast p-values are descriptive stability diagnostics, not external validation.")
    print("The frozen primary clean programs are M34, M11, and M24.")
    print("M40 is frozen as a separate proliferation-dominant deviation axis.")
    print("M28, M38, M25, and M17 are prespecified sensitivity programs.")
    print("No human outcome may be used to revise module membership, weights, score direction, or validation tier.")

    print("")
    print("Saved:")
    for path in frozen_paths + [OUTPUT_FREEZE_JSON]:
        print(path)
    print("Done.")


if __name__ == "__main__":
    main()
