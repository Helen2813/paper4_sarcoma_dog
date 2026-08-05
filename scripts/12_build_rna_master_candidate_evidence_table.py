from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


FILES = {
    "dfi_univariate": "GSE238110_dfi_univariate_cox_top5000var.csv",
    "os_univariate": "GSE238110_os_univariate_cox_top5000var.csv",
    "mb_candidates": "GSE238110_mb_candidate_genes_from_univariate_cox.csv",
    "dfi_conditional_full": "GSE238110_dfi_conditional_cox_mb_selected.csv",
    "os_conditional_full": "GSE238110_os_conditional_cox_mb_selected.csv",
    "true_mb_genes": "GSE238110_true_iamb_gsmb_ablation_selected_genes.csv",
    "nested_selected_genes": "GSE238110_nested_cv_selected_genes.csv",
    "nested_stability": "GSE238110_nested_cv_selected_gene_stability.csv",
    "nested_benchmark": "GSE238110_nested_cv_method_benchmark_summary.csv",
    "method_vs_random": "GSE238110_nested_cv_method_vs_random_percentile_summary.csv",
}


OUTPUT_MASTER = "GSE238110_RNA_master_candidate_evidence_table.csv"
OUTPUT_TOP = "GSE238110_RNA_master_candidate_evidence_top100.csv"
OUTPUT_PRIORITY_GENES = "GSE238110_RNA_priority_gene_ids.txt"
OUTPUT_PRIORITY_SYMBOLS = "GSE238110_RNA_priority_gene_symbols.txt"


def read_csv_if_exists(filename):
    path = RESULTS_DIR / filename

    if not path.exists():
        print(f"Missing file: {path}")
        return pd.DataFrame()

    print(f"Loaded: {path}")
    return pd.read_csv(path)


def clean_gene_symbol(gene):
    gene = str(gene)
    tail = gene.rsplit("_", 1)[-1]
    return gene.rsplit("_", 1)[0] if tail.isdigit() else gene


def numeric_series(df, col):
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    return pd.to_numeric(df[col], errors="coerce")


def add_univariate_evidence(master, df, prefix):
    if df.empty or "gene" not in df.columns:
        return master

    use = df.copy()
    use["p"] = numeric_series(use, "p")
    use["q"] = numeric_series(use, "q")
    use["coef"] = numeric_series(use, "coef")
    use["hr_per_sd"] = numeric_series(use, "hr_per_sd")
    use["c_index"] = numeric_series(use, "c_index")

    use = use.sort_values("p", na_position="last").copy()
    use[f"{prefix}_univ_rank"] = np.arange(1, use.shape[0] + 1)

    cols = [
        "gene",
        f"{prefix}_univ_rank",
        "coef",
        "hr_per_sd",
        "p",
        "q",
        "c_index",
    ]

    use = use[[c for c in cols if c in use.columns]].copy()

    rename = {
        "coef": f"{prefix}_univ_coef",
        "hr_per_sd": f"{prefix}_univ_hr_per_sd",
        "p": f"{prefix}_univ_p",
        "q": f"{prefix}_univ_q",
        "c_index": f"{prefix}_univ_c_index",
    }

    use = use.rename(columns=rename)

    return master.merge(use, on="gene", how="left")


def add_candidate_source(master, df):
    if df.empty or "gene" not in df.columns:
        master["in_univariate_candidate_set"] = False
        return master

    use = df.copy()
    use = use.drop_duplicates("gene", keep="first")

    keep_cols = ["gene"]

    if "candidate_rank" in use.columns:
        keep_cols.append("candidate_rank")

    if "candidate_source" in use.columns:
        keep_cols.append("candidate_source")

    use = use[keep_cols].copy()
    use["in_univariate_candidate_set"] = True

    use = use.rename(columns={
        "candidate_rank": "univariate_candidate_rank",
        "candidate_source": "univariate_candidate_source",
    })

    out = master.merge(use, on="gene", how="left")
    out["in_univariate_candidate_set"] = out["in_univariate_candidate_set"].fillna(False)

    return out


def add_full_conditional_selection(master, df, prefix):
    selected_col = f"{prefix}_full_conditional_selected"

    if df.empty or "gene" not in df.columns:
        master[selected_col] = False
        return master

    use = df.copy()
    use = use.drop_duplicates("gene", keep="first")
    use[f"{prefix}_full_conditional_rank"] = np.arange(1, use.shape[0] + 1)
    use[selected_col] = True

    keep_cols = [
        "gene",
        selected_col,
        f"{prefix}_full_conditional_rank",
        "coef",
        "hr_per_sd",
        "p",
        "c_index",
    ]

    use = use[[c for c in keep_cols if c in use.columns]].copy()

    rename = {
        "coef": f"{prefix}_full_conditional_coef",
        "hr_per_sd": f"{prefix}_full_conditional_hr_per_sd",
        "p": f"{prefix}_full_conditional_p",
        "c_index": f"{prefix}_full_conditional_c_index",
    }

    use = use.rename(columns=rename)

    out = master.merge(use, on="gene", how="left")
    out[selected_col] = out[selected_col].fillna(False)

    return out


def add_true_mb_evidence(master, df):
    if df.empty or "gene" not in df.columns:
        for endpoint in ["dfi", "os"]:
            master[f"{endpoint}_true_mb_n_configs"] = 0
            master[f"{endpoint}_true_mb_algorithms"] = ""
            master[f"{endpoint}_true_mb_alphas"] = ""
        return master

    use = df.copy()
    use["endpoint"] = use["endpoint"].astype(str)
    use["algorithm"] = use["algorithm"].astype(str)
    use["alpha"] = use["alpha"].astype(str)

    for endpoint_label, prefix in [("DFI", "dfi"), ("OS", "os")]:
        part = use[use["endpoint"].str.upper() == endpoint_label].copy()

        if part.empty:
            master[f"{prefix}_true_mb_n_configs"] = 0
            master[f"{prefix}_true_mb_algorithms"] = ""
            master[f"{prefix}_true_mb_alphas"] = ""
            continue

        agg = (
            part
            .groupby("gene", dropna=False)
            .agg(
                **{
                    f"{prefix}_true_mb_n_configs": ("gene", "size"),
                    f"{prefix}_true_mb_algorithms": ("algorithm", lambda x: ";".join(sorted(set(x)))),
                    f"{prefix}_true_mb_alphas": ("alpha", lambda x: ";".join(sorted(set(x)))),
                    f"{prefix}_true_mb_mean_rank": ("gene_rank", "mean"),
                }
            )
            .reset_index()
        )

        master = master.merge(agg, on="gene", how="left")
        master[f"{prefix}_true_mb_n_configs"] = master[f"{prefix}_true_mb_n_configs"].fillna(0).astype(int)
        master[f"{prefix}_true_mb_algorithms"] = master[f"{prefix}_true_mb_algorithms"].fillna("")
        master[f"{prefix}_true_mb_alphas"] = master[f"{prefix}_true_mb_alphas"].fillna("")

    return master


def method_to_short(method):
    replacements = {
        "conditional_cox_mb_plus_clinical": "conditional_plus_clinical",
        "conditional_cox_mb": "conditional",
        "elastic_net_cox": "elasticnet",
        "univariate_top10": "univtop10",
        "iamb_alpha0.10": "iamb",
        "gsmb_alpha0.10": "gsmb",
        "clinical_only": "clinical",
        "random_top10_mean20": "random",
        "random_survival_forest_candidate500": "rsf",
    }

    return replacements.get(str(method), str(method).replace(".", "").replace("-", "_"))


def add_nested_stability(master, stability):
    if stability.empty or "gene" not in stability.columns:
        master["nested_any_max_selection_frequency"] = 0.0
        master["nested_any_total_selected_folds"] = 0
        return master

    use = stability.copy()
    use["endpoint"] = use["endpoint"].astype(str).str.lower()
    use["method_short"] = use["method"].map(method_to_short)

    for endpoint in sorted(use["endpoint"].dropna().unique()):
        for method_short in sorted(use["method_short"].dropna().unique()):
            part = use[
                (use["endpoint"] == endpoint) &
                (use["method_short"] == method_short)
            ].copy()

            if part.empty:
                continue

            cols = [
                "gene",
                "selected_in_folds",
                "selection_frequency",
                "mean_gene_rank",
            ]

            part = part[[c for c in cols if c in part.columns]].copy()

            rename = {
                "selected_in_folds": f"nested_{endpoint}_{method_short}_selected_folds",
                "selection_frequency": f"nested_{endpoint}_{method_short}_selection_frequency",
                "mean_gene_rank": f"nested_{endpoint}_{method_short}_mean_rank",
            }

            part = part.rename(columns=rename)
            master = master.merge(part, on="gene", how="left")

    freq_cols = [c for c in master.columns if c.endswith("_selection_frequency")]
    fold_cols = [c for c in master.columns if c.endswith("_selected_folds")]

    for col in freq_cols:
        master[col] = pd.to_numeric(master[col], errors="coerce").fillna(0.0)

    for col in fold_cols:
        master[col] = pd.to_numeric(master[col], errors="coerce").fillna(0).astype(int)

    if freq_cols:
        master["nested_any_max_selection_frequency"] = master[freq_cols].max(axis=1)
    else:
        master["nested_any_max_selection_frequency"] = 0.0

    if fold_cols:
        master["nested_any_total_selected_folds"] = master[fold_cols].sum(axis=1)
    else:
        master["nested_any_total_selected_folds"] = 0

    return master


def add_direction_consistency(master):
    master["dfi_os_both_univariate_available"] = (
        master["dfi_univ_coef"].notna() &
        master["os_univ_coef"].notna()
    )

    master["dfi_os_same_direction"] = (
        np.sign(master["dfi_univ_coef"]) == np.sign(master["os_univ_coef"])
    ) & master["dfi_os_both_univariate_available"]

    master["dfi_os_both_nominal_p05"] = (
        (master["dfi_univ_p"] < 0.05) &
        (master["os_univ_p"] < 0.05)
    )

    master["dfi_os_both_fdr_q10"] = (
        (master["dfi_univ_q"] < 0.10) &
        (master["os_univ_q"] < 0.10)
    )

    master["combined_univ_rank_score"] = (
        master["dfi_univ_rank"].fillna(master["dfi_univ_rank"].max() + 1) +
        master["os_univ_rank"].fillna(master["os_univ_rank"].max() + 1)
    )

    return master


def get_col(df, col, default=0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


def get_bool_col(df, col):
    if col in df.columns:
        return df[col].fillna(False).astype(bool)
    return pd.Series(False, index=df.index)


def compute_priority_score(master):
    score = pd.Series(0.0, index=master.index)

    score += (get_col(master, "dfi_univ_q", 1) < 0.10).astype(float) * 1.0
    score += (get_col(master, "os_univ_q", 1) < 0.10).astype(float) * 1.0
    score += master["dfi_os_same_direction"].astype(float) * 1.0
    score += master["dfi_os_both_fdr_q10"].astype(float) * 1.0

    score += get_bool_col(master, "dfi_full_conditional_selected").astype(float) * 2.0
    score += get_bool_col(master, "os_full_conditional_selected").astype(float) * 1.0

    score += (get_col(master, "dfi_true_mb_n_configs", 0) >= 2).astype(float) * 1.0
    score += (get_col(master, "os_true_mb_n_configs", 0) >= 2).astype(float) * 1.0

    score += (get_col(master, "nested_dfi_conditional_selection_frequency", 0) >= 0.40).astype(float) * 3.0
    score += (get_col(master, "nested_dfi_elasticnet_selection_frequency", 0) >= 0.40).astype(float) * 2.0
    score += (get_col(master, "nested_os_gsmb_selection_frequency", 0) >= 0.40).astype(float) * 1.0
    score += (get_col(master, "nested_os_iamb_selection_frequency", 0) >= 0.40).astype(float) * 1.0
    score += (get_col(master, "nested_any_max_selection_frequency", 0) >= 0.60).astype(float) * 2.0

    master["rna_evidence_priority_score"] = score

    conditions = [
        master["rna_evidence_priority_score"] >= 8,
        master["rna_evidence_priority_score"] >= 5,
        master["rna_evidence_priority_score"] >= 3,
    ]

    choices = [
        "high_rna_evidence",
        "medium_rna_evidence",
        "exploratory_rna_evidence",
    ]

    master["rna_evidence_tier"] = np.select(
        conditions,
        choices,
        default="low_rna_evidence",
    )

    return master


def summarize_method_results(method_vs_random, benchmark):
    print("")
    print("=" * 80)
    print("Method-level RNA benchmark interpretation")
    print("=" * 80)

    if not method_vs_random.empty:
        keep = method_vs_random[
            method_vs_random["random_source"].isin(["train_screen_top1000", "train_screen_top500"])
        ].copy()

        if not keep.empty:
            cols = [
                "endpoint",
                "method",
                "random_source",
                "mean_percentile",
                "median_percentile",
                "mean_delta_vs_random_mean",
                "mean_method_c_index",
                "mean_random_mean_c_index",
            ]
            cols = [c for c in cols if c in keep.columns]

            print("Method vs screened-random controls:")
            print(
                keep[cols]
                .sort_values(["endpoint", "random_source", "mean_percentile"], ascending=[True, True, False])
                .head(30)
                .to_string(index=False)
            )

    if not benchmark.empty:
        cols = [
            "endpoint",
            "method",
            "mean_c_index",
            "std_c_index",
            "median_c_index",
            "mean_n_selected_genes",
        ]
        cols = [c for c in cols if c in benchmark.columns]

        print("")
        print("Nested-CV benchmark:")
        print(
            benchmark[cols]
            .sort_values(["endpoint", "mean_c_index"], ascending=[True, False])
            .to_string(index=False)
        )


def main():
    print("=" * 80)
    print("Build RNA master candidate evidence table")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Results directory: {RESULTS_DIR}")
    print("")

    dfi_univ = read_csv_if_exists(FILES["dfi_univariate"])
    os_univ = read_csv_if_exists(FILES["os_univariate"])
    candidates = read_csv_if_exists(FILES["mb_candidates"])
    dfi_conditional = read_csv_if_exists(FILES["dfi_conditional_full"])
    os_conditional = read_csv_if_exists(FILES["os_conditional_full"])
    true_mb = read_csv_if_exists(FILES["true_mb_genes"])
    nested_selected = read_csv_if_exists(FILES["nested_selected_genes"])
    nested_stability = read_csv_if_exists(FILES["nested_stability"])
    nested_benchmark = read_csv_if_exists(FILES["nested_benchmark"])
    method_vs_random = read_csv_if_exists(FILES["method_vs_random"])

    all_gene_sets = []

    for df in [
        dfi_univ,
        os_univ,
        candidates,
        dfi_conditional,
        os_conditional,
        true_mb,
        nested_selected,
        nested_stability,
    ]:
        if not df.empty and "gene" in df.columns:
            all_gene_sets.extend(df["gene"].dropna().astype(str).tolist())

    unique_genes = sorted(set(all_gene_sets))

    master = pd.DataFrame({"gene": unique_genes})
    master["gene_symbol_clean"] = master["gene"].map(clean_gene_symbol)

    print("")
    print(f"Unique RNA genes in evidence universe: {master.shape[0]}")

    master = add_candidate_source(master, candidates)
    master = add_univariate_evidence(master, dfi_univ, "dfi")
    master = add_univariate_evidence(master, os_univ, "os")
    master = add_full_conditional_selection(master, dfi_conditional, "dfi")
    master = add_full_conditional_selection(master, os_conditional, "os")
    master = add_true_mb_evidence(master, true_mb)
    master = add_nested_stability(master, nested_stability)
    master = add_direction_consistency(master)
    master = compute_priority_score(master)

    master = master.sort_values(
        [
            "rna_evidence_priority_score",
            "nested_any_max_selection_frequency",
            "dfi_univ_q",
            "combined_univ_rank_score",
        ],
        ascending=[False, False, True, True],
        na_position="last",
    ).copy()

    top = master.head(100).copy()

    priority = master[
        master["rna_evidence_tier"].isin([
            "high_rna_evidence",
            "medium_rna_evidence",
        ])
    ].copy()

    master_path = RESULTS_DIR / OUTPUT_MASTER
    top_path = RESULTS_DIR / OUTPUT_TOP
    priority_gene_path = RESULTS_DIR / OUTPUT_PRIORITY_GENES
    priority_symbol_path = RESULTS_DIR / OUTPUT_PRIORITY_SYMBOLS

    master.to_csv(master_path, index=False)
    top.to_csv(top_path, index=False)

    with open(priority_gene_path, "w", encoding="utf-8") as f:
        f.write("\n".join(priority["gene"].astype(str).tolist()))

    with open(priority_symbol_path, "w", encoding="utf-8") as f:
        symbols = priority["gene_symbol_clean"].dropna().astype(str).drop_duplicates().tolist()
        f.write("\n".join(symbols))

    print("")
    print("=" * 80)
    print("RNA evidence tier summary")
    print("=" * 80)
    print(master["rna_evidence_tier"].value_counts().to_string())

    print("")
    print("=" * 80)
    print("Top RNA candidate evidence rows")
    print("=" * 80)

    display_cols = [
        "gene",
        "gene_symbol_clean",
        "rna_evidence_tier",
        "rna_evidence_priority_score",
        "dfi_univ_rank",
        "dfi_univ_q",
        "os_univ_rank",
        "os_univ_q",
        "dfi_os_same_direction",
        "dfi_full_conditional_selected",
        "os_full_conditional_selected",
        "dfi_true_mb_n_configs",
        "os_true_mb_n_configs",
        "nested_dfi_conditional_selection_frequency",
        "nested_dfi_elasticnet_selection_frequency",
        "nested_os_gsmb_selection_frequency",
        "nested_any_max_selection_frequency",
    ]

    display_cols = [c for c in display_cols if c in master.columns]
    print(master[display_cols].head(50).to_string(index=False))

    print("")
    print("=" * 80)
    print("Stable DFI-focused RNA candidates")
    print("=" * 80)

    stable_cols = [
        "gene",
        "gene_symbol_clean",
        "rna_evidence_tier",
        "rna_evidence_priority_score",
        "nested_dfi_conditional_selection_frequency",
        "nested_dfi_elasticnet_selection_frequency",
        "dfi_univ_q",
        "os_univ_q",
        "dfi_full_conditional_selected",
        "dfi_true_mb_n_configs",
    ]

    stable_cols = [c for c in stable_cols if c in master.columns]

    stable = master[
        (
            get_col(master, "nested_dfi_conditional_selection_frequency", 0) >= 0.40
        ) |
        (
            get_col(master, "nested_dfi_elasticnet_selection_frequency", 0) >= 0.40
        ) |
        (
            get_bool_col(master, "dfi_full_conditional_selected")
        )
    ].copy()

    print(stable[stable_cols].head(50).to_string(index=False))

    summarize_method_results(method_vs_random, nested_benchmark)

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("Gene-level RNA panels are exploratory because selected methods do not consistently outperform screened-random panels.")
    print("DFI has the clearest RNA signal; OS should be treated as a sensitivity or concordance endpoint.")
    print("The next major step should prioritize ortholog mapping, pathway/module aggregation, and external human validation.")

    print("")
    print("Saved:")
    print(master_path)
    print(top_path)
    print(priority_gene_path)
    print(priority_symbol_path)
    print("Done.")


if __name__ == "__main__":
    main()
