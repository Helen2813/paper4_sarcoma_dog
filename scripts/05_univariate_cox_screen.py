from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning


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


def clean_gene_symbol(gene):
    gene = str(gene)
    return gene.rsplit("_", 1)[0] if gene.rsplit("_", 1)[-1].isdigit() else gene


def fit_gene_cox(df, time_col, event_col, gene):
    use = df[[time_col, event_col, gene]].dropna().copy()
    use = use.rename(columns={gene: "gene"})

    if use.shape[0] < 30:
        return {"gene": gene, "error": "too_few_samples"}

    if use["gene"].std() == 0 or not np.isfinite(use["gene"].std()):
        return {"gene": gene, "error": "zero_or_invalid_variance"}

    use["gene"] = (use["gene"] - use["gene"].mean()) / use["gene"].std()

    cph = CoxPHFitter()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            cph.fit(use, duration_col=time_col, event_col=event_col)

        row = cph.summary.loc["gene"]

        return {
            "gene": gene,
            "gene_symbol_clean": clean_gene_symbol(gene),
            "n": int(use.shape[0]),
            "events": int(use[event_col].sum()),
            "coef": float(row["coef"]),
            "hr_per_sd": float(row["exp(coef)"]),
            "se_coef": float(row["se(coef)"]),
            "z": float(row["z"]),
            "p": float(row["p"]),
            "ci_lower": float(row["exp(coef) lower 95%"]),
            "ci_upper": float(row["exp(coef) upper 95%"]),
            "c_index": float(cph.concordance_index_),
            "error": "",
        }

    except Exception as e:
        return {
            "gene": gene,
            "gene_symbol_clean": clean_gene_symbol(gene),
            "n": int(use.shape[0]),
            "events": int(use[event_col].sum()),
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", type=str)
    parser.add_argument("--top-n", default=5000, type=int)
    parser.add_argument("--endpoint", choices=["dfi", "os"], default="dfi")
    parser.add_argument("--max-features", default=None, type=int)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    processed_dir = project_root / "data" / "processed"
    results_dir = project_root / "results" / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)

    expr_path = processed_dir / f"GSE238110_DOG2_expression_log2cpm_matched_top{args.top_n}var.csv"
    clinical_path = processed_dir / "GSE238110_DOG2_clinical_matched_indexed.csv"

    expr = pd.read_csv(expr_path, index_col=0)
    clinical = pd.read_csv(clinical_path, index_col=0)

    if args.endpoint == "dfi":
        time_col = "dfi_time"
        event_col = "dfi_event"
    else:
        time_col = "os_time"
        event_col = "os_event"

    clinical[time_col] = pd.to_numeric(clinical[time_col], errors="coerce")
    clinical[event_col] = pd.to_numeric(clinical[event_col], errors="coerce")

    common_samples = clinical.index.intersection(expr.index)
    clinical = clinical.loc[common_samples]
    expr = expr.loc[common_samples]

    genes = list(expr.columns)
    if args.max_features is not None:
        genes = genes[: args.max_features]

    df = clinical[[time_col, event_col]].join(expr[genes], how="inner")

    print("Project root:", project_root)
    print("Endpoint:", args.endpoint)
    print("Time column:", time_col)
    print("Event column:", event_col)
    print("Samples:", df.shape[0])
    print("Genes:", len(genes))
    print("Events:", int(df[event_col].sum()))

    results = []

    for i, gene in enumerate(genes, start=1):
        if i % 250 == 0:
            print(f"Processed {i}/{len(genes)} genes")

        results.append(fit_gene_cox(df, time_col, event_col, gene))

    results = pd.DataFrame(results)

    if "p" in results.columns:
        results["q"] = bh_fdr(results["p"].values)
        results = results.sort_values(["p", "q"], na_position="last")

    out_path = results_dir / f"GSE238110_{args.endpoint}_univariate_cox_top{args.top_n}var.csv"
    results.to_csv(out_path, index=False)

    top_path = results_dir / f"GSE238110_{args.endpoint}_univariate_cox_top_hits_top{args.top_n}var.csv"
    results.head(200).to_csv(top_path, index=False)

    print("Saved:")
    print(out_path)
    print(top_path)


if __name__ == "__main__":
    main()
