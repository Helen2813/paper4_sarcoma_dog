from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXPRESSION_FILE = "GSE238110_DOG2_expression_log2cpm_matched_allgenes.csv"
CLINICAL_FILE = "GSE238110_DOG2_clinical_matched_indexed.csv"

PROLIFERATION_SCORE_FILE = "GSE238110_meta_proliferation_score_per_sample.csv"
MODULE_SCORES_FILE = "GSE238110_full_cohort_module_scores_for_proliferation_adjustment.csv"
MODULE_ADJUSTMENT_FILE = "GSE238110_module_proliferation_adjustment_results.csv"
CANDIDATE_ADJUSTMENT_FILE = "GSE238110_candidate_gene_proliferation_adjustment_results.csv"

FULL_MODULE_PRIORITY_FILE = "GSE238110_RNA_full_cohort_transferable_module_priority.csv"
STRICT_CANDIDATES_FILE = "GSE238110_RNA_strict_transferable_candidates.csv"
BROAD_CANDIDATES_FILE = "GSE238110_RNA_broad_transferable_candidates_for_sensitivity.csv"

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

OUTPUT_MODULE_RESIDUAL = RESULTS_DIR / "GSE238110_module_residualized_proliferation_analysis.csv"
OUTPUT_CANDIDATE_RESIDUAL = RESULTS_DIR / "GSE238110_candidate_gene_residualized_proliferation_analysis.csv"
OUTPUT_RESIDUAL_SUMMARY = RESULTS_DIR / "GSE238110_residualized_proliferation_analysis_summary.csv"


def clean_gene_symbol(gene):
    gene = str(gene)
    tail = gene.rsplit("_", 1)[-1]
    return gene.rsplit("_", 1)[0] if tail.isdigit() else gene


def read_csv_if_exists(path, index_col=None):
    if not path.exists():
        print(f"Missing file: {path}")
        return pd.DataFrame()

    print(f"Loaded: {path}")
    return pd.read_csv(path, index_col=index_col)


def q_col_from_p_col(p_col):
    p_col = str(p_col)

    if not p_col.endswith("_p"):
        raise ValueError(f"Expected p-value column ending with '_p', got: {p_col}")

    return p_col[:-2] + "_q"


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


def zscore_series(series):
    series = pd.to_numeric(series, errors="coerce")
    std = series.std()

    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index)

    return (series - series.mean()) / std


def safe_corr(a, b):
    a = pd.Series(a).astype(float)
    b = pd.Series(b).astype(float)

    valid = a.notna() & b.notna()

    if valid.sum() < 5:
        return np.nan

    if a[valid].std() == 0 or b[valid].std() == 0:
        return np.nan

    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def standardize_expression(expression):
    x = expression.apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)

    medians = x.median(axis=0)
    x = x.fillna(medians)

    means = x.mean(axis=0)
    stds = x.std(axis=0).replace(0, np.nan)

    z = (x - means) / stds
    valid_cols = z.columns[z.notna().all(axis=0)]
    z = z[valid_cols]

    return z


def make_symbol_to_columns(expression):
    symbol_to_cols = {}

    for col in expression.columns:
        symbol = clean_gene_symbol(col).upper()
        symbol_to_cols.setdefault(symbol, []).append(col)

    return symbol_to_cols


def find_weight_column(clinical):
    for col in WEIGHT_COLUMN_CANDIDATES:
        if col in clinical.columns:
            return col

    lowered = {str(c).lower(): c for c in clinical.columns}

    for key in ["weight", "body weight", "body_weight"]:
        if key in lowered:
            return lowered[key]

    return None


def residualize_score(score, covariates):
    score = pd.Series(score).astype(float)
    covariates = covariates.copy()

    data = pd.concat(
        [score.rename("score"), covariates],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()

    residual = pd.Series(np.nan, index=score.index, dtype=float)

    if data.shape[0] < 10:
        return residual

    y = data["score"].values.astype(float)
    x = data[covariates.columns].values.astype(float)

    x = np.column_stack([np.ones(x.shape[0]), x])

    try:
        beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        fitted = x @ beta
        resid = y - fitted
        residual.loc[data.index] = resid
        residual = zscore_series(residual)
    except Exception:
        pass

    return residual


def fit_cox_model(data, time_col, event_col, score_col):
    needed = [time_col, event_col, score_col]
    df = data[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()

    result = {
        "n": df.shape[0],
        "events": int(df[event_col].sum()) if event_col in df.columns and df.shape[0] else 0,
        "coef": np.nan,
        "hr": np.nan,
        "p": np.nan,
        "se": np.nan,
        "c_index": np.nan,
        "error": "",
    }

    if df.shape[0] < 30:
        result["error"] = "too_few_samples"
        return result

    if df[event_col].sum() < 5:
        result["error"] = "too_few_events"
        return result

    if df[score_col].std() == 0:
        result["error"] = "zero_variance_score"
        return result

    cph = CoxPHFitter(penalizer=COX_PENALIZER)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            cph.fit(
                df,
                duration_col=time_col,
                event_col=event_col,
                fit_options={"max_steps": 500},
            )

        risk = cph.predict_partial_hazard(df[[score_col]]).values.ravel()

        result["c_index"] = float(concordance_index(df[time_col], -risk, df[event_col]))

        s = cph.summary.loc[score_col]
        result["coef"] = float(s["coef"])
        result["hr"] = float(s["exp(coef)"])
        result["p"] = float(s["p"])
        result["se"] = float(s["se(coef)"])

    except Exception as e:
        result["error"] = str(e)[:500]

    return result


def prepare_analysis_frame(clinical, proliferation_score):
    frame = clinical.copy()

    frame["meta_proliferation_score"] = proliferation_score.reindex(frame.index)
    frame["meta_proliferation_score"] = zscore_series(frame["meta_proliferation_score"])

    weight_col = find_weight_column(frame)

    if weight_col is not None:
        frame["weight_z"] = zscore_series(frame[weight_col])
        print(f"Weight column detected and standardized: {weight_col}")
    else:
        frame["weight_z"] = np.nan
        print("No weight column detected. Weight-residualized analyses will be skipped.")

    return frame, weight_col


def build_score_variants(analysis_frame, raw_score, base_name, weight_col):
    raw_score = zscore_series(raw_score)

    variants = {
        "original": raw_score,
        "residual_after_proliferation": residualize_score(
            raw_score,
            analysis_frame[["meta_proliferation_score"]],
        ),
    }

    if weight_col is not None:
        variants["residual_after_proliferation_and_weight"] = residualize_score(
            raw_score,
            analysis_frame[["meta_proliferation_score", "weight_z"]],
        )

    output = {}

    for variant_name, score in variants.items():
        col = f"{base_name}__{variant_name}"
        output[variant_name] = {
            "column": col,
            "score": score,
        }

    return output


def get_module_label_from_score_column(score_col):
    value = str(score_col)

    if value.startswith("module_") and value.endswith("_score"):
        return value.replace("module_", "").replace("_score", "")

    return value


def run_module_residual_analysis(analysis_frame, module_scores, module_priority, module_adjustment, weight_col):
    rows = []

    metadata_cols = [
        "module_label",
        "module_transfer_qc_tier",
        "transfer_priority_score",
        "n_module_genes",
        "fraction_strict_symbol_concordant",
        "fraction_broad_transferable",
        "n_high_or_medium_rna_evidence",
        "strict_human_symbols",
    ]

    priority = pd.DataFrame()

    if not module_priority.empty:
        metadata_cols = [c for c in metadata_cols if c in module_priority.columns]
        priority = module_priority[metadata_cols].drop_duplicates("module_label").copy()

    adjustment_keep = []

    if not module_adjustment.empty:
        adjustment_keep = [
            "endpoint",
            "module_label",
            "module_proliferation_correlation",
            "module_only_p",
            "module_plus_proliferation_p",
            "module_plus_proliferation_q",
            "module_plus_proliferation_c_index",
            "abs_coef_ratio_after_proliferation_adjustment",
        ]
        adjustment_keep = [c for c in adjustment_keep if c in module_adjustment.columns]

    for score_col in module_scores.columns:
        module_label = get_module_label_from_score_column(score_col)
        raw_score = module_scores[score_col].reindex(analysis_frame.index)

        variants = build_score_variants(
            analysis_frame=analysis_frame,
            raw_score=raw_score,
            base_name=f"module_{module_label}",
            weight_col=weight_col,
        )

        score_frame = analysis_frame.copy()

        for variant in variants.values():
            score_frame[variant["column"]] = variant["score"]

        for endpoint_label, endpoint in ENDPOINTS.items():
            time_col = endpoint["time_col"]
            event_col = endpoint["event_col"]

            for variant_name, variant in variants.items():
                model = fit_cox_model(
                    data=score_frame,
                    time_col=time_col,
                    event_col=event_col,
                    score_col=variant["column"],
                )

                rows.append({
                    "endpoint": endpoint_label,
                    "module_label": module_label,
                    "score_type": variant_name,
                    "score_column": variant["column"],
                    "module_proliferation_correlation_raw": safe_corr(
                        raw_score,
                        analysis_frame["meta_proliferation_score"],
                    ),
                    "score_proliferation_correlation_after_residualization": safe_corr(
                        variant["score"],
                        analysis_frame["meta_proliferation_score"],
                    ),
                    "n": model["n"],
                    "events": model["events"],
                    "coef": model["coef"],
                    "hr": model["hr"],
                    "p": model["p"],
                    "se": model["se"],
                    "c_index": model["c_index"],
                    "error": model["error"],
                })

    results = pd.DataFrame(rows)

    if not priority.empty:
        results = results.merge(priority, on="module_label", how="left")

    if adjustment_keep:
        results = results.merge(
            module_adjustment[adjustment_keep],
            on=["endpoint", "module_label"],
            how="left",
        )

    for endpoint_label in ENDPOINTS:
        for score_type in results["score_type"].dropna().unique():
            mask = results["endpoint"].eq(endpoint_label) & results["score_type"].eq(score_type)
            results.loc[mask, "q"] = bh_adjust(results.loc[mask, "p"])

    return results


def get_candidate_expression_column(row, expression_z, symbol_to_cols):
    gene = str(row.get("gene", ""))

    if gene in expression_z.columns:
        return gene

    symbol = str(row.get("gene_symbol_clean", clean_gene_symbol(gene))).upper()
    cols = [c for c in symbol_to_cols.get(symbol, []) if c in expression_z.columns]

    if not cols:
        return ""

    variances = expression_z[cols].var(axis=0).sort_values(ascending=False)

    return variances.index[0]


def load_candidate_table():
    strict = read_csv_if_exists(RESULTS_DIR / STRICT_CANDIDATES_FILE)

    if strict.empty:
        print("Strict candidate file is missing; candidate residual analysis will be skipped.")
        return pd.DataFrame()

    strict = strict.copy()
    strict["candidate_transfer_set"] = "strict_primary"

    broad = read_csv_if_exists(RESULTS_DIR / BROAD_CANDIDATES_FILE)

    if not broad.empty and "needs_manual_ortholog_review" in broad.columns:
        review = broad[broad["needs_manual_ortholog_review"].fillna(False).astype(bool)].copy()
        review["candidate_transfer_set"] = "broad_review_sensitivity"
        candidates = pd.concat([strict, review], axis=0, ignore_index=True)
    else:
        candidates = strict

    candidates = candidates.drop_duplicates(["gene", "candidate_transfer_set"], keep="first")

    return candidates


def run_candidate_residual_analysis(analysis_frame, expression_z, candidates, symbol_to_cols, candidate_adjustment, weight_col):
    if candidates.empty:
        return pd.DataFrame()

    rows = []

    adjustment_keep = []

    if not candidate_adjustment.empty:
        adjustment_keep = [
            "endpoint",
            "gene",
            "candidate_transfer_set",
            "gene_proliferation_correlation",
            "gene_only_p",
            "gene_plus_proliferation_p",
            "gene_plus_proliferation_q",
            "gene_plus_proliferation_c_index",
            "abs_coef_ratio_after_proliferation_adjustment",
        ]
        adjustment_keep = [c for c in adjustment_keep if c in candidate_adjustment.columns]

    for idx, candidate in candidates.iterrows():
        expr_col = get_candidate_expression_column(candidate, expression_z, symbol_to_cols)

        if not expr_col:
            continue

        raw_score = expression_z[expr_col].reindex(analysis_frame.index)

        variants = build_score_variants(
            analysis_frame=analysis_frame,
            raw_score=raw_score,
            base_name=f"candidate_{idx}",
            weight_col=weight_col,
        )

        score_frame = analysis_frame.copy()

        for variant in variants.values():
            score_frame[variant["column"]] = variant["score"]

        for endpoint_label, endpoint in ENDPOINTS.items():
            time_col = endpoint["time_col"]
            event_col = endpoint["event_col"]

            for variant_name, variant in variants.items():
                model = fit_cox_model(
                    data=score_frame,
                    time_col=time_col,
                    event_col=event_col,
                    score_col=variant["column"],
                )

                rows.append({
                    "endpoint": endpoint_label,
                    "gene": candidate.get("gene", ""),
                    "gene_symbol_clean": candidate.get("gene_symbol_clean", ""),
                    "human_gene_symbol": candidate.get("human_gene_symbol", ""),
                    "candidate_transfer_set": candidate.get("candidate_transfer_set", ""),
                    "rna_evidence_tier": candidate.get("rna_evidence_tier", ""),
                    "rna_evidence_priority_score": candidate.get("rna_evidence_priority_score", np.nan),
                    "expression_column_used": expr_col,
                    "score_type": variant_name,
                    "score_column": variant["column"],
                    "gene_proliferation_correlation_raw": safe_corr(
                        raw_score,
                        analysis_frame["meta_proliferation_score"],
                    ),
                    "score_proliferation_correlation_after_residualization": safe_corr(
                        variant["score"],
                        analysis_frame["meta_proliferation_score"],
                    ),
                    "n": model["n"],
                    "events": model["events"],
                    "coef": model["coef"],
                    "hr": model["hr"],
                    "p": model["p"],
                    "se": model["se"],
                    "c_index": model["c_index"],
                    "error": model["error"],
                })

    results = pd.DataFrame(rows)

    if adjustment_keep:
        results = results.merge(
            candidate_adjustment[adjustment_keep],
            on=["endpoint", "gene", "candidate_transfer_set"],
            how="left",
        )

    for endpoint_label in ENDPOINTS:
        for score_type in results["score_type"].dropna().unique():
            mask = results["endpoint"].eq(endpoint_label) & results["score_type"].eq(score_type)
            results.loc[mask, "q"] = bh_adjust(results.loc[mask, "p"])

    return results


def summarize_residual_results(module_results, candidate_results):
    rows = []

    for endpoint_label in ENDPOINTS:
        for score_type in module_results["score_type"].dropna().unique():
            part = module_results[
                module_results["endpoint"].eq(endpoint_label) &
                module_results["score_type"].eq(score_type)
            ].copy()

            rows.append({
                "analysis": "module",
                "endpoint": endpoint_label,
                "score_type": score_type,
                "n_tested": part.shape[0],
                "n_nominal_p05": int((part["p"] < 0.05).sum()),
                "n_fdr10": int((part["q"] < 0.10).sum()),
                "median_c_index": float(part["c_index"].median()),
                "median_abs_raw_corr_with_proliferation": float(
                    part["module_proliferation_correlation_raw"].abs().median()
                ),
            })

        if not candidate_results.empty:
            for score_type in candidate_results["score_type"].dropna().unique():
                part = candidate_results[
                    candidate_results["endpoint"].eq(endpoint_label) &
                    candidate_results["score_type"].eq(score_type)
                ].copy()

                rows.append({
                    "analysis": "candidate_gene",
                    "endpoint": endpoint_label,
                    "score_type": score_type,
                    "n_tested": part.shape[0],
                    "n_nominal_p05": int((part["p"] < 0.05).sum()),
                    "n_fdr10": int((part["q"] < 0.10).sum()),
                    "median_c_index": float(part["c_index"].median()),
                    "median_abs_raw_corr_with_proliferation": float(
                        part["gene_proliferation_correlation_raw"].abs().median()
                    ),
                })

    return pd.DataFrame(rows)


def print_results(module_results, candidate_results, summary):
    print("")
    print("=" * 80)
    print("Residualized proliferation analysis summary")
    print("=" * 80)
    print(summary.to_string(index=False))

    print("")
    print("=" * 80)
    print("Module residual associations after removing proliferation")
    print("=" * 80)

    module_display = module_results[
        module_results["score_type"].eq("residual_after_proliferation")
    ].copy()

    module_display = module_display.sort_values(
        ["endpoint", "p"],
        ascending=[True, True],
        na_position="last",
    )

    module_cols = [
        "endpoint",
        "module_label",
        "module_transfer_qc_tier",
        "n_module_genes",
        "module_proliferation_correlation_raw",
        "p",
        "q",
        "c_index",
        "module_plus_proliferation_p",
        "module_plus_proliferation_q",
        "transfer_priority_score",
        "strict_human_symbols",
    ]

    module_cols = [c for c in module_cols if c in module_display.columns]
    print(module_display[module_cols].head(40).to_string(index=False))

    if "residual_after_proliferation_and_weight" in set(module_results["score_type"]):
        print("")
        print("=" * 80)
        print("Module residual associations after removing proliferation and weight")
        print("=" * 80)

        module_weight = module_results[
            module_results["score_type"].eq("residual_after_proliferation_and_weight")
        ].copy()

        module_weight = module_weight.sort_values(
            ["endpoint", "p"],
            ascending=[True, True],
            na_position="last",
        )

        print(module_weight[module_cols].head(40).to_string(index=False))

    print("")
    print("=" * 80)
    print("Candidate-gene residual associations after removing proliferation")
    print("=" * 80)

    if candidate_results.empty:
        print("No candidate residual results were computed.")
    else:
        candidate_display = candidate_results[
            candidate_results["score_type"].eq("residual_after_proliferation")
        ].copy()

        candidate_display = candidate_display.sort_values(
            ["endpoint", "candidate_transfer_set", "p"],
            ascending=[True, True, True],
            na_position="last",
        )

        candidate_cols = [
            "endpoint",
            "gene_symbol_clean",
            "human_gene_symbol",
            "candidate_transfer_set",
            "rna_evidence_tier",
            "gene_proliferation_correlation_raw",
            "p",
            "q",
            "c_index",
            "gene_plus_proliferation_p",
            "gene_plus_proliferation_q",
        ]

        candidate_cols = [c for c in candidate_cols if c in candidate_display.columns]
        print(candidate_display[candidate_cols].head(100).to_string(index=False))

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("Residualized scores test the component of a module or gene that is linearly independent of the meta-proliferation score.")
    print("For highly correlated modules such as proliferation/cell-cycle modules, residual analysis is more interpretable than putting both collinear scores in one Cox model.")
    print("Signals that survive residualization are better candidates for non-proliferation prognostic programs.")
    print("This remains full-cohort mechanistic analysis; external human validation is still required.")


def main():
    print("=" * 80)
    print("Residualized proliferation signal analysis")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Processed directory: {PROCESSED_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print("")

    clinical = read_csv_if_exists(PROCESSED_DIR / CLINICAL_FILE, index_col=0)
    expression = read_csv_if_exists(PROCESSED_DIR / EXPRESSION_FILE, index_col=0)
    proliferation_score = read_csv_if_exists(RESULTS_DIR / PROLIFERATION_SCORE_FILE, index_col=0)
    module_scores = read_csv_if_exists(RESULTS_DIR / MODULE_SCORES_FILE, index_col=0)
    module_priority = read_csv_if_exists(RESULTS_DIR / FULL_MODULE_PRIORITY_FILE)
    module_adjustment = read_csv_if_exists(RESULTS_DIR / MODULE_ADJUSTMENT_FILE)
    candidate_adjustment = read_csv_if_exists(RESULTS_DIR / CANDIDATE_ADJUSTMENT_FILE)

    if clinical.empty or expression.empty or proliferation_score.empty or module_scores.empty:
        raise RuntimeError("Required input files are missing.")

    if "meta_proliferation_score" not in proliferation_score.columns:
        proliferation_score.columns = ["meta_proliferation_score"]

    common_samples = clinical.index.intersection(expression.index)
    common_samples = common_samples.intersection(proliferation_score.index)
    common_samples = common_samples.intersection(module_scores.index)

    clinical = clinical.loc[common_samples].copy()
    expression = expression.loc[common_samples].copy()
    proliferation_score = proliferation_score.loc[common_samples, "meta_proliferation_score"].copy()
    module_scores = module_scores.loc[common_samples].copy()

    print("")
    print("Matched data:")
    print(f"  Clinical table: {clinical.shape}")
    print(f"  Expression matrix: {expression.shape}")
    print(f"  Proliferation score samples: {proliferation_score.shape[0]}")
    print(f"  Module score matrix: {module_scores.shape}")

    analysis_frame, weight_col = prepare_analysis_frame(clinical, proliferation_score)

    module_results = run_module_residual_analysis(
        analysis_frame=analysis_frame,
        module_scores=module_scores,
        module_priority=module_priority,
        module_adjustment=module_adjustment,
        weight_col=weight_col,
    )

    expression_z = standardize_expression(expression)
    symbol_to_cols = make_symbol_to_columns(expression_z)
    candidates = load_candidate_table()

    candidate_results = run_candidate_residual_analysis(
        analysis_frame=analysis_frame,
        expression_z=expression_z,
        candidates=candidates,
        symbol_to_cols=symbol_to_cols,
        candidate_adjustment=candidate_adjustment,
        weight_col=weight_col,
    )

    summary = summarize_residual_results(module_results, candidate_results)

    module_results.to_csv(OUTPUT_MODULE_RESIDUAL, index=False)
    candidate_results.to_csv(OUTPUT_CANDIDATE_RESIDUAL, index=False)
    summary.to_csv(OUTPUT_RESIDUAL_SUMMARY, index=False)

    print_results(module_results, candidate_results, summary)

    print("")
    print("Saved:")
    print(OUTPUT_MODULE_RESIDUAL)
    print(OUTPUT_CANDIDATE_RESIDUAL)
    print(OUTPUT_RESIDUAL_SUMMARY)
    print("Done.")


if __name__ == "__main__":
    main()
