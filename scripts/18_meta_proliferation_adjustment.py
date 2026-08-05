from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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

MODULE_MEMBERSHIP_FILE = "GSE238110_RNA_module_gene_membership.csv"
FULL_MODULE_PRIORITY_FILE = "GSE238110_RNA_full_cohort_transferable_module_priority.csv"
STRICT_CANDIDATES_FILE = "GSE238110_RNA_strict_transferable_candidates.csv"
BROAD_CANDIDATES_FILE = "GSE238110_RNA_broad_transferable_candidates_for_sensitivity.csv"

TOP_CORRELATED_PROLIFERATION_GENES = 250
MIN_PROLIFERATION_GENES = 20
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

PROLIFERATION_ANCHOR_SYMBOLS = [
    "PCNA",
    "MKI67",
    "TOP2A",
    "BIRC5",
    "UBE2C",
    "UBE2S",
    "AURKA",
    "AURKB",
    "CDC20",
    "CDC6",
    "CDK1",
    "CCNA2",
    "CCNB1",
    "CCNB2",
    "MCM2",
    "MCM4",
    "MCM5",
    "MCM10",
    "TYMS",
    "RRM2",
    "TK1",
    "PLK1",
    "PLK4",
    "CENPA",
    "CENPE",
    "CENPF",
    "CENPK",
    "CENPV",
    "KIF11",
    "KIF15",
    "KIF18B",
    "KIF23",
    "MELK",
    "MYBL2",
    "BUB1",
    "BUB1B",
    "DLGAP5",
    "SPAG5",
    "STMN1",
]

WEIGHT_COLUMN_CANDIDATES = [
    "weight",
    "Weight",
    "body_weight",
    "body weight",
    "Body weight",
    "body_weight_kg",
]

OUTPUT_PROLIFERATION_GENES = RESULTS_DIR / "GSE238110_meta_proliferation_gene_set.csv"
OUTPUT_PROLIFERATION_SCORE = RESULTS_DIR / "GSE238110_meta_proliferation_score_per_sample.csv"
OUTPUT_PROLIFERATION_MODELS = RESULTS_DIR / "GSE238110_meta_proliferation_only_models.csv"
OUTPUT_MODULE_SCORES = RESULTS_DIR / "GSE238110_full_cohort_module_scores_for_proliferation_adjustment.csv"
OUTPUT_MODULE_RESULTS = RESULTS_DIR / "GSE238110_module_proliferation_adjustment_results.csv"
OUTPUT_CANDIDATE_RESULTS = RESULTS_DIR / "GSE238110_candidate_gene_proliferation_adjustment_results.csv"
OUTPUT_PROLIFERATION_SUMMARY = RESULTS_DIR / "GSE238110_meta_proliferation_adjustment_summary.csv"


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


def make_symbol_to_columns(expression):
    symbol_to_cols = {}

    for col in expression.columns:
        symbol = clean_gene_symbol(col).upper()
        symbol_to_cols.setdefault(symbol, []).append(col)

    return symbol_to_cols


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


def get_anchor_columns(expression_z, symbol_to_cols):
    anchor_cols = []

    for symbol in PROLIFERATION_ANCHOR_SYMBOLS:
        cols = symbol_to_cols.get(symbol.upper(), [])
        cols = [c for c in cols if c in expression_z.columns]
        anchor_cols.extend(cols)

    return sorted(set(anchor_cols))


def build_meta_proliferation_score(expression_z, symbol_to_cols, module_membership):
    anchor_cols = get_anchor_columns(expression_z, symbol_to_cols)
    fallback_used = False

    if len(anchor_cols) < 3:
        print("Too few canonical proliferation anchor genes found.")
        print("Falling back to full-cohort M40 module genes if available.")

        fallback = module_membership[
            (module_membership["analysis"] == "full_cohort") &
            (module_membership["module_label"] == "M40")
        ].copy()

        anchor_cols = [
            g for g in fallback["gene"].dropna().astype(str).tolist()
            if g in expression_z.columns
        ]

        fallback_used = True

    if len(anchor_cols) < 3:
        raise RuntimeError("Could not build proliferation score: insufficient anchor genes.")

    anchor_score = expression_z[anchor_cols].mean(axis=1)
    anchor_score = zscore_series(anchor_score)

    anchor_values = anchor_score.values
    anchor_values = (anchor_values - np.nanmean(anchor_values)) / np.nanstd(anchor_values, ddof=1)

    expr_values = expression_z.values
    corr_values = np.dot(expr_values.T, anchor_values) / (expression_z.shape[0] - 1)

    corr = pd.Series(corr_values, index=expression_z.columns)
    corr = corr.replace([np.inf, -np.inf], np.nan).dropna()
    corr = corr.sort_values(ascending=False)

    top_cols = corr[corr > 0].head(TOP_CORRELATED_PROLIFERATION_GENES).index.tolist()
    proliferation_cols = sorted(set(top_cols + anchor_cols))

    if len(proliferation_cols) < MIN_PROLIFERATION_GENES:
        raise RuntimeError("Could not build proliferation score: too few correlated genes.")

    pca = PCA(n_components=1, random_state=42)
    pc1 = pca.fit_transform(expression_z[proliferation_cols].values).ravel()

    if safe_corr(pc1, anchor_score.values) < 0:
        pc1 = -pc1

    proliferation_score = pd.Series(
        zscore_series(pd.Series(pc1, index=expression_z.index)),
        index=expression_z.index,
        name="meta_proliferation_score",
    )

    gene_rows = []

    for col in proliferation_cols:
        gene_rows.append({
            "gene": col,
            "gene_symbol_clean": clean_gene_symbol(col),
            "is_anchor_gene": col in anchor_cols,
            "correlation_with_anchor_score": float(corr.get(col, np.nan)),
        })

    gene_set = pd.DataFrame(gene_rows)
    gene_set = gene_set.sort_values(
        ["is_anchor_gene", "correlation_with_anchor_score"],
        ascending=[False, False],
        na_position="last",
    )

    info = {
        "n_anchor_columns": len(anchor_cols),
        "n_proliferation_gene_columns": len(proliferation_cols),
        "pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
        "fallback_used": fallback_used,
        "anchor_columns": ";".join(anchor_cols),
    }

    return proliferation_score, gene_set, info


def compute_pc1_score(expression_z, genes, score_name):
    genes = [g for g in genes if g in expression_z.columns]
    genes = sorted(set(genes))

    if len(genes) == 0:
        return None, {}

    if len(genes) == 1:
        score = expression_z[genes[0]].copy()
        score.name = score_name

        info = {
            "n_genes_used": 1,
            "pc1_explained_variance": 1.0,
            "genes_used": genes[0],
        }

        return score, info

    pca = PCA(n_components=1, random_state=42)
    pc1 = pca.fit_transform(expression_z[genes].values).ravel()

    mean_score = expression_z[genes].mean(axis=1)

    if safe_corr(pc1, mean_score.values) < 0:
        pc1 = -pc1

    score = pd.Series(
        zscore_series(pd.Series(pc1, index=expression_z.index)),
        index=expression_z.index,
        name=score_name,
    )

    info = {
        "n_genes_used": len(genes),
        "pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
        "genes_used": ";".join(genes),
    }

    return score, info


def find_weight_column(clinical):
    for col in WEIGHT_COLUMN_CANDIDATES:
        if col in clinical.columns:
            return col

    lowered = {str(c).lower(): c for c in clinical.columns}

    for key in ["weight", "body weight", "body_weight"]:
        if key in lowered:
            return lowered[key]

    return None


def prepare_analysis_frame(clinical, proliferation_score):
    frame = clinical.copy()
    frame["meta_proliferation_score"] = proliferation_score.reindex(frame.index)

    weight_col = find_weight_column(frame)

    if weight_col is not None:
        frame["weight_z"] = zscore_series(frame[weight_col])
        print(f"Weight column detected and standardized: {weight_col}")
    else:
        frame["weight_z"] = np.nan
        print("No weight column detected. Weight-adjusted models will be skipped.")

    return frame, weight_col


def fit_cox_model(data, time_col, event_col, covariates, focal_covariate):
    needed = [time_col, event_col] + covariates
    df = data[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()

    result = {
        "n": df.shape[0],
        "events": int(df[event_col].sum()) if event_col in df.columns and df.shape[0] else 0,
        "c_index": np.nan,
        "log_likelihood": np.nan,
        "focal_coef": np.nan,
        "focal_hr": np.nan,
        "focal_p": np.nan,
        "focal_se": np.nan,
        "proliferation_coef": np.nan,
        "proliferation_p": np.nan,
        "weight_coef": np.nan,
        "weight_p": np.nan,
        "error": "",
    }

    if df.shape[0] < 30:
        result["error"] = "too_few_samples"
        return result

    if df[event_col].sum() < 5:
        result["error"] = "too_few_events"
        return result

    for cov in covariates:
        if df[cov].std() == 0:
            result["error"] = f"zero_variance_covariate:{cov}"
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

        risk = cph.predict_partial_hazard(df[covariates]).values.ravel()

        result["c_index"] = float(concordance_index(df[time_col], -risk, df[event_col]))
        result["log_likelihood"] = float(cph.log_likelihood_)

        if focal_covariate in cph.summary.index:
            s = cph.summary.loc[focal_covariate]
            result["focal_coef"] = float(s["coef"])
            result["focal_hr"] = float(s["exp(coef)"])
            result["focal_p"] = float(s["p"])
            result["focal_se"] = float(s["se(coef)"])

        if "meta_proliferation_score" in cph.summary.index:
            s = cph.summary.loc["meta_proliferation_score"]
            result["proliferation_coef"] = float(s["coef"])
            result["proliferation_p"] = float(s["p"])

        if "weight_z" in cph.summary.index:
            s = cph.summary.loc["weight_z"]
            result["weight_coef"] = float(s["coef"])
            result["weight_p"] = float(s["p"])

    except Exception as e:
        result["error"] = str(e)[:500]

    return result


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


def build_full_cohort_module_scores(expression_z, module_membership):
    full = module_membership[module_membership["analysis"] == "full_cohort"].copy()

    if full.empty:
        raise RuntimeError("No full-cohort modules were found in module membership table.")

    scores = pd.DataFrame(index=expression_z.index)
    info_rows = []

    for module_label, part in full.groupby("module_label"):
        genes = part["gene"].dropna().astype(str).tolist()
        score_name = f"module_{module_label}_score"

        score, info = compute_pc1_score(expression_z, genes, score_name)

        if score is None:
            continue

        scores[score_name] = score

        info_rows.append({
            "module_label": module_label,
            "score_column": score_name,
            **info,
        })

    info = pd.DataFrame(info_rows)

    return scores, info


def get_module_priority_info(module_priority):
    if module_priority.empty:
        return pd.DataFrame(columns=["module_label"])

    keep = [
        "module_label",
        "module_transfer_qc_tier",
        "transfer_priority_score",
        "n_module_genes",
        "fraction_strict_symbol_concordant",
        "fraction_broad_transferable",
        "n_high_or_medium_rna_evidence",
        "dfi_full_module_p",
        "dfi_full_module_c_index",
        "os_full_module_p",
        "os_full_module_c_index",
        "strict_human_symbols",
    ]

    keep = [c for c in keep if c in module_priority.columns]

    return module_priority[keep].drop_duplicates("module_label").copy()


def run_module_adjustment(analysis_frame, module_scores, module_info, module_priority, weight_col):
    rows = []
    priority = get_module_priority_info(module_priority)

    data = analysis_frame.join(module_scores, how="inner")

    for endpoint_label, endpoint in ENDPOINTS.items():
        time_col = endpoint["time_col"]
        event_col = endpoint["event_col"]

        data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
        data[event_col] = pd.to_numeric(data[event_col], errors="coerce")

        for _, module_row in module_info.iterrows():
            module_label = module_row["module_label"]
            score_col = module_row["score_column"]

            module_only = fit_cox_model(
                data=data,
                time_col=time_col,
                event_col=event_col,
                covariates=[score_col],
                focal_covariate=score_col,
            )

            module_plus_prolif = fit_cox_model(
                data=data,
                time_col=time_col,
                event_col=event_col,
                covariates=[score_col, "meta_proliferation_score"],
                focal_covariate=score_col,
            )

            if weight_col is not None:
                module_plus_prolif_weight = fit_cox_model(
                    data=data,
                    time_col=time_col,
                    event_col=event_col,
                    covariates=[score_col, "meta_proliferation_score", "weight_z"],
                    focal_covariate=score_col,
                )
            else:
                module_plus_prolif_weight = {
                    "focal_coef": np.nan,
                    "focal_hr": np.nan,
                    "focal_p": np.nan,
                    "c_index": np.nan,
                    "proliferation_p": np.nan,
                    "weight_p": np.nan,
                    "error": "weight_not_available",
                }

            unadj_coef = module_only["focal_coef"]
            adj_coef = module_plus_prolif["focal_coef"]

            if np.isfinite(unadj_coef) and unadj_coef != 0 and np.isfinite(adj_coef):
                coef_ratio = abs(adj_coef) / abs(unadj_coef)
            else:
                coef_ratio = np.nan

            rows.append({
                "endpoint": endpoint_label,
                "module_label": module_label,
                "score_column": score_col,
                "n_genes_used": module_row.get("n_genes_used", np.nan),
                "module_pc1_explained_variance": module_row.get("pc1_explained_variance", np.nan),
                "module_proliferation_correlation": safe_corr(data[score_col], data["meta_proliferation_score"]),
                "module_only_coef": module_only["focal_coef"],
                "module_only_hr": module_only["focal_hr"],
                "module_only_p": module_only["focal_p"],
                "module_only_c_index": module_only["c_index"],
                "module_plus_proliferation_coef": module_plus_prolif["focal_coef"],
                "module_plus_proliferation_hr": module_plus_prolif["focal_hr"],
                "module_plus_proliferation_p": module_plus_prolif["focal_p"],
                "module_plus_proliferation_c_index": module_plus_prolif["c_index"],
                "module_plus_proliferation_proliferation_p": module_plus_prolif["proliferation_p"],
                "module_plus_proliferation_weight_coef": module_plus_prolif_weight["focal_coef"],
                "module_plus_proliferation_weight_hr": module_plus_prolif_weight["focal_hr"],
                "module_plus_proliferation_weight_p": module_plus_prolif_weight["focal_p"],
                "module_plus_proliferation_weight_c_index": module_plus_prolif_weight["c_index"],
                "module_plus_proliferation_weight_proliferation_p": module_plus_prolif_weight["proliferation_p"],
                "module_plus_proliferation_weight_weight_p": module_plus_prolif_weight["weight_p"],
                "abs_coef_ratio_after_proliferation_adjustment": coef_ratio,
                "retains_nominal_p05_after_proliferation": bool(
                    np.isfinite(module_plus_prolif["focal_p"]) and module_plus_prolif["focal_p"] < 0.05
                ),
                "module_only_error": module_only["error"],
                "module_plus_proliferation_error": module_plus_prolif["error"],
                "module_plus_proliferation_weight_error": module_plus_prolif_weight["error"],
            })

    results = pd.DataFrame(rows)

    if not priority.empty:
        results = results.merge(priority, on="module_label", how="left")

    for endpoint_label in ENDPOINTS:
        mask = results["endpoint"].eq(endpoint_label)

        for p_col in [
            "module_only_p",
            "module_plus_proliferation_p",
            "module_plus_proliferation_weight_p",
        ]:
            q_col = q_col_from_p_col(p_col)
            results.loc[mask, q_col] = bh_adjust(results.loc[mask, p_col])

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
        print("Strict candidate file is missing; candidate adjustment will be skipped.")
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


def run_candidate_adjustment(analysis_frame, expression_z, candidates, symbol_to_cols, weight_col):
    if candidates.empty:
        return pd.DataFrame()

    rows = []

    for _, candidate in candidates.iterrows():
        expr_col = get_candidate_expression_column(candidate, expression_z, symbol_to_cols)

        if not expr_col:
            continue

        gene_score_name = f"gene_score__{candidate['gene']}"
        gene_score = expression_z[expr_col].copy()
        gene_score.name = gene_score_name

        data = analysis_frame.join(gene_score, how="inner")

        for endpoint_label, endpoint in ENDPOINTS.items():
            time_col = endpoint["time_col"]
            event_col = endpoint["event_col"]

            gene_only = fit_cox_model(
                data=data,
                time_col=time_col,
                event_col=event_col,
                covariates=[gene_score_name],
                focal_covariate=gene_score_name,
            )

            gene_plus_prolif = fit_cox_model(
                data=data,
                time_col=time_col,
                event_col=event_col,
                covariates=[gene_score_name, "meta_proliferation_score"],
                focal_covariate=gene_score_name,
            )

            if weight_col is not None:
                gene_plus_prolif_weight = fit_cox_model(
                    data=data,
                    time_col=time_col,
                    event_col=event_col,
                    covariates=[gene_score_name, "meta_proliferation_score", "weight_z"],
                    focal_covariate=gene_score_name,
                )
            else:
                gene_plus_prolif_weight = {
                    "focal_coef": np.nan,
                    "focal_hr": np.nan,
                    "focal_p": np.nan,
                    "c_index": np.nan,
                    "proliferation_p": np.nan,
                    "weight_p": np.nan,
                    "error": "weight_not_available",
                }

            unadj_coef = gene_only["focal_coef"]
            adj_coef = gene_plus_prolif["focal_coef"]

            if np.isfinite(unadj_coef) and unadj_coef != 0 and np.isfinite(adj_coef):
                coef_ratio = abs(adj_coef) / abs(unadj_coef)
            else:
                coef_ratio = np.nan

            rows.append({
                "endpoint": endpoint_label,
                "gene": candidate.get("gene", ""),
                "gene_symbol_clean": candidate.get("gene_symbol_clean", ""),
                "human_gene_symbol": candidate.get("human_gene_symbol", ""),
                "candidate_transfer_set": candidate.get("candidate_transfer_set", ""),
                "rna_evidence_tier": candidate.get("rna_evidence_tier", ""),
                "rna_evidence_priority_score": candidate.get("rna_evidence_priority_score", np.nan),
                "expression_column_used": expr_col,
                "gene_proliferation_correlation": safe_corr(data[gene_score_name], data["meta_proliferation_score"]),
                "gene_only_coef": gene_only["focal_coef"],
                "gene_only_hr": gene_only["focal_hr"],
                "gene_only_p": gene_only["focal_p"],
                "gene_only_c_index": gene_only["c_index"],
                "gene_plus_proliferation_coef": gene_plus_prolif["focal_coef"],
                "gene_plus_proliferation_hr": gene_plus_prolif["focal_hr"],
                "gene_plus_proliferation_p": gene_plus_prolif["focal_p"],
                "gene_plus_proliferation_c_index": gene_plus_prolif["c_index"],
                "gene_plus_proliferation_proliferation_p": gene_plus_prolif["proliferation_p"],
                "gene_plus_proliferation_weight_coef": gene_plus_prolif_weight["focal_coef"],
                "gene_plus_proliferation_weight_hr": gene_plus_prolif_weight["focal_hr"],
                "gene_plus_proliferation_weight_p": gene_plus_prolif_weight["focal_p"],
                "gene_plus_proliferation_weight_c_index": gene_plus_prolif_weight["c_index"],
                "gene_plus_proliferation_weight_proliferation_p": gene_plus_prolif_weight["proliferation_p"],
                "gene_plus_proliferation_weight_weight_p": gene_plus_prolif_weight["weight_p"],
                "abs_coef_ratio_after_proliferation_adjustment": coef_ratio,
                "retains_nominal_p05_after_proliferation": bool(
                    np.isfinite(gene_plus_prolif["focal_p"]) and gene_plus_prolif["focal_p"] < 0.05
                ),
                "gene_only_error": gene_only["error"],
                "gene_plus_proliferation_error": gene_plus_prolif["error"],
                "gene_plus_proliferation_weight_error": gene_plus_prolif_weight["error"],
            })

    results = pd.DataFrame(rows)

    for endpoint_label in ENDPOINTS:
        mask = results["endpoint"].eq(endpoint_label)

        for p_col in [
            "gene_only_p",
            "gene_plus_proliferation_p",
            "gene_plus_proliferation_weight_p",
        ]:
            q_col = q_col_from_p_col(p_col)
            results.loc[mask, q_col] = bh_adjust(results.loc[mask, p_col])

    return results


def run_proliferation_only_models(analysis_frame, weight_col):
    rows = []

    for endpoint_label, endpoint in ENDPOINTS.items():
        time_col = endpoint["time_col"]
        event_col = endpoint["event_col"]

        proliferation_only = fit_cox_model(
            data=analysis_frame,
            time_col=time_col,
            event_col=event_col,
            covariates=["meta_proliferation_score"],
            focal_covariate="meta_proliferation_score",
        )

        rows.append({
            "endpoint": endpoint_label,
            "model": "proliferation_only",
            "n": proliferation_only["n"],
            "events": proliferation_only["events"],
            "proliferation_coef": proliferation_only["focal_coef"],
            "proliferation_hr": proliferation_only["focal_hr"],
            "proliferation_p": proliferation_only["focal_p"],
            "c_index": proliferation_only["c_index"],
            "error": proliferation_only["error"],
        })

        if weight_col is not None:
            proliferation_weight = fit_cox_model(
                data=analysis_frame,
                time_col=time_col,
                event_col=event_col,
                covariates=["meta_proliferation_score", "weight_z"],
                focal_covariate="meta_proliferation_score",
            )

            rows.append({
                "endpoint": endpoint_label,
                "model": "proliferation_plus_weight",
                "n": proliferation_weight["n"],
                "events": proliferation_weight["events"],
                "proliferation_coef": proliferation_weight["focal_coef"],
                "proliferation_hr": proliferation_weight["focal_hr"],
                "proliferation_p": proliferation_weight["focal_p"],
                "weight_p": proliferation_weight["weight_p"],
                "c_index": proliferation_weight["c_index"],
                "error": proliferation_weight["error"],
            })

    return pd.DataFrame(rows)


def summarize_results(prolif_info, prolif_models, module_results, candidate_results):
    print("")
    print("=" * 80)
    print("Meta-proliferation score construction")
    print("=" * 80)
    for key, value in prolif_info.items():
        if key == "anchor_columns":
            print(f"{key}: {str(value)[:500]}")
        else:
            print(f"{key}: {value}")

    print("")
    print("=" * 80)
    print("Proliferation-only Cox models")
    print("=" * 80)
    print(prolif_models.to_string(index=False))

    print("")
    print("=" * 80)
    print("Module associations before and after proliferation adjustment")
    print("=" * 80)

    module_display = module_results.copy()
    module_display = module_display.sort_values(
        ["endpoint", "module_plus_proliferation_p", "module_only_p"],
        ascending=[True, True, True],
        na_position="last",
    )

    module_cols = [
        "endpoint",
        "module_label",
        "module_transfer_qc_tier",
        "n_genes_used",
        "module_proliferation_correlation",
        "module_only_p",
        "module_only_q",
        "module_only_c_index",
        "module_plus_proliferation_p",
        "module_plus_proliferation_q",
        "module_plus_proliferation_c_index",
        "module_plus_proliferation_proliferation_p",
        "abs_coef_ratio_after_proliferation_adjustment",
        "retains_nominal_p05_after_proliferation",
    ]

    module_cols = [c for c in module_cols if c in module_display.columns]
    print(module_display[module_cols].head(40).to_string(index=False))

    print("")
    print("=" * 80)
    print("Strict/broad candidate genes before and after proliferation adjustment")
    print("=" * 80)

    if candidate_results.empty:
        print("No candidate results were computed.")
    else:
        candidate_display = candidate_results.copy()
        candidate_display = candidate_display.sort_values(
            ["endpoint", "candidate_transfer_set", "gene_plus_proliferation_p"],
            ascending=[True, True, True],
            na_position="last",
        )

        candidate_cols = [
            "endpoint",
            "gene_symbol_clean",
            "human_gene_symbol",
            "candidate_transfer_set",
            "rna_evidence_tier",
            "gene_proliferation_correlation",
            "gene_only_p",
            "gene_only_q",
            "gene_only_c_index",
            "gene_plus_proliferation_p",
            "gene_plus_proliferation_q",
            "gene_plus_proliferation_c_index",
            "gene_plus_proliferation_proliferation_p",
            "abs_coef_ratio_after_proliferation_adjustment",
            "retains_nominal_p05_after_proliferation",
        ]

        candidate_cols = [c for c in candidate_cols if c in candidate_display.columns]
        print(candidate_display[candidate_cols].head(100).to_string(index=False))

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("This is a full-cohort mechanistic adjustment, not an external predictive validation.")
    print("Modules or genes that remain associated after proliferation adjustment are better candidates for non-proliferation prognostic biology.")
    print("Modules whose coefficients strongly attenuate after adjustment should be interpreted as largely proliferation-linked.")
    print("Weight-adjusted models are sensitivity analyses because weight may partly proxy breed or body-size biology.")


def safe_count_less_than(df, col, threshold):
    if df.empty or col not in df.columns:
        return 0

    return int((pd.to_numeric(df[col], errors="coerce") < threshold).sum())


def safe_median(df, col):
    if df.empty or col not in df.columns:
        return np.nan

    values = pd.to_numeric(df[col], errors="coerce")
    return float(values.median()) if values.notna().any() else np.nan


def main():
    print("=" * 80)
    print("Meta-proliferation adjustment analysis")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Processed directory: {PROCESSED_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print("")

    expression = read_csv_if_exists(PROCESSED_DIR / EXPRESSION_FILE, index_col=0)
    clinical = read_csv_if_exists(PROCESSED_DIR / CLINICAL_FILE, index_col=0)
    module_membership = read_csv_if_exists(RESULTS_DIR / MODULE_MEMBERSHIP_FILE)
    module_priority = read_csv_if_exists(RESULTS_DIR / FULL_MODULE_PRIORITY_FILE)

    if expression.empty or clinical.empty:
        raise RuntimeError("Expression or clinical data are missing.")

    common_samples = expression.index.intersection(clinical.index)
    expression = expression.loc[common_samples].copy()
    clinical = clinical.loc[common_samples].copy()

    print("")
    print("Matched data:")
    print(f"  Expression matrix: {expression.shape}")
    print(f"  Clinical table: {clinical.shape}")

    expression_z = standardize_expression(expression)
    symbol_to_cols = make_symbol_to_columns(expression_z)

    print(f"  Standardized expression matrix: {expression_z.shape}")

    proliferation_score, proliferation_gene_set, prolif_info = build_meta_proliferation_score(
        expression_z=expression_z,
        symbol_to_cols=symbol_to_cols,
        module_membership=module_membership,
    )

    analysis_frame, weight_col = prepare_analysis_frame(clinical, proliferation_score)

    module_scores, module_info = build_full_cohort_module_scores(
        expression_z=expression_z,
        module_membership=module_membership,
    )

    module_scores.to_csv(OUTPUT_MODULE_SCORES, index=True)
    proliferation_gene_set.to_csv(OUTPUT_PROLIFERATION_GENES, index=False)
    proliferation_score.to_frame().to_csv(OUTPUT_PROLIFERATION_SCORE, index=True)

    proliferation_models = run_proliferation_only_models(
        analysis_frame=analysis_frame,
        weight_col=weight_col,
    )

    module_results = run_module_adjustment(
        analysis_frame=analysis_frame,
        module_scores=module_scores,
        module_info=module_info,
        module_priority=module_priority,
        weight_col=weight_col,
    )

    candidates = load_candidate_table()

    candidate_results = run_candidate_adjustment(
        analysis_frame=analysis_frame,
        expression_z=expression_z,
        candidates=candidates,
        symbol_to_cols=symbol_to_cols,
        weight_col=weight_col,
    )

    summary_rows = []

    for endpoint_label in ENDPOINTS:
        module_part = module_results[module_results["endpoint"].eq(endpoint_label)].copy()

        if candidate_results.empty:
            candidate_part = pd.DataFrame()
        else:
            candidate_part = candidate_results[candidate_results["endpoint"].eq(endpoint_label)].copy()

        summary_rows.append({
            "endpoint": endpoint_label,
            "n_modules_tested": module_part.shape[0],
            "n_modules_nominal_before_proliferation": safe_count_less_than(module_part, "module_only_p", 0.05),
            "n_modules_nominal_after_proliferation": safe_count_less_than(module_part, "module_plus_proliferation_p", 0.05),
            "n_modules_fdr10_before_proliferation": safe_count_less_than(module_part, "module_only_q", 0.10),
            "n_modules_fdr10_after_proliferation": safe_count_less_than(module_part, "module_plus_proliferation_q", 0.10),
            "median_module_abs_coef_ratio_after_proliferation": safe_median(module_part, "abs_coef_ratio_after_proliferation_adjustment"),
            "n_candidates_tested": candidate_part.shape[0],
            "n_candidates_nominal_before_proliferation": safe_count_less_than(candidate_part, "gene_only_p", 0.05),
            "n_candidates_nominal_after_proliferation": safe_count_less_than(candidate_part, "gene_plus_proliferation_p", 0.05),
            "n_candidates_fdr10_before_proliferation": safe_count_less_than(candidate_part, "gene_only_q", 0.10),
            "n_candidates_fdr10_after_proliferation": safe_count_less_than(candidate_part, "gene_plus_proliferation_q", 0.10),
            "median_candidate_abs_coef_ratio_after_proliferation": safe_median(candidate_part, "abs_coef_ratio_after_proliferation_adjustment"),
        })

    summary = pd.DataFrame(summary_rows)

    proliferation_models.to_csv(OUTPUT_PROLIFERATION_MODELS, index=False)
    module_results.to_csv(OUTPUT_MODULE_RESULTS, index=False)
    candidate_results.to_csv(OUTPUT_CANDIDATE_RESULTS, index=False)
    summary.to_csv(OUTPUT_PROLIFERATION_SUMMARY, index=False)

    summarize_results(
        prolif_info=prolif_info,
        prolif_models=proliferation_models,
        module_results=module_results,
        candidate_results=candidate_results,
    )

    print("")
    print("Saved:")
    print(OUTPUT_PROLIFERATION_GENES)
    print(OUTPUT_PROLIFERATION_SCORE)
    print(OUTPUT_PROLIFERATION_MODELS)
    print(OUTPUT_MODULE_SCORES)
    print(OUTPUT_MODULE_RESULTS)
    print(OUTPUT_CANDIDATE_RESULTS)
    print(OUTPUT_PROLIFERATION_SUMMARY)
    print("Done.")


if __name__ == "__main__":
    main()
