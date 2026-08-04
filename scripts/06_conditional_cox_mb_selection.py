from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_N_VARIABLE_GENES = 5000
MAX_SELECTED_FEATURES = 25
FORWARD_ALPHA = 0.01
BACKWARD_ALPHA = 0.05
COX_PENALIZER = 0.05

ENDPOINTS = {
    "dfi": {
        "time_col": "dfi_time",
        "event_col": "dfi_event",
        "label": "DFI",
    },
    "os": {
        "time_col": "os_time",
        "event_col": "os_event",
        "label": "OS",
    },
}


def clean_gene_symbol(gene):
    gene = str(gene)
    tail = gene.rsplit("_", 1)[-1]
    return gene.rsplit("_", 1)[0] if tail.isdigit() else gene


def standardize_expression(x):
    x = x.copy()
    x = x.replace([np.inf, -np.inf], np.nan)

    medians = x.median(axis=0)
    x = x.fillna(medians)

    means = x.mean(axis=0)
    stds = x.std(axis=0).replace(0, np.nan)

    x = (x - means) / stds
    x = x.dropna(axis=1)

    return x


def fit_cox(df, time_col, event_col, feature_cols):
    use = df[[time_col, event_col] + feature_cols].dropna().copy()

    if use.shape[0] < 30:
        return None

    cph = CoxPHFitter(penalizer=COX_PENALIZER)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            cph.fit(use, duration_col=time_col, event_col=event_col)

        return cph

    except Exception:
        return None


def get_feature_pvalue(cph, feature):
    if cph is None:
        return np.nan

    if feature not in cph.summary.index:
        return np.nan

    return float(cph.summary.loc[feature, "p"])


def forward_backward_selection(df, candidate_genes, time_col, event_col, endpoint_label):
    selected = []
    remaining = list(candidate_genes)
    step_log = []

    print("")
    print("=" * 80)
    print(f"Starting conditional Cox selection for {endpoint_label}")
    print("=" * 80)
    print(f"Samples: {df.shape[0]}")
    print(f"Candidate genes: {len(remaining)}")
    print(f"Events: {int(df[event_col].sum())}")
    print(f"Forward alpha: {FORWARD_ALPHA}")
    print(f"Backward alpha: {BACKWARD_ALPHA}")
    print(f"Max selected features: {MAX_SELECTED_FEATURES}")
    print("")

    for step in range(1, MAX_SELECTED_FEATURES + 1):
        print(f"Forward step {step}")
        print(f"Currently selected: {len(selected)}")
        print(f"Remaining candidates: {len(remaining)}")

        tested_rows = []

        for i, gene in enumerate(remaining, start=1):
            cph = fit_cox(df, time_col, event_col, selected + [gene])
            p_value = get_feature_pvalue(cph, gene)

            tested_rows.append({
                "endpoint": endpoint_label,
                "step": step,
                "phase": "forward",
                "gene": gene,
                "gene_symbol_clean": clean_gene_symbol(gene),
                "p": p_value,
                "c_index": np.nan if cph is None else float(cph.concordance_index_),
                "selected_before": len(selected),
            })

            if i % 100 == 0:
                print(f"  Tested {i}/{len(remaining)} candidates")

        tested = pd.DataFrame(tested_rows).sort_values("p", na_position="last")

        if tested.empty or not np.isfinite(tested.iloc[0]["p"]):
            print("No valid candidate models were fitted. Stopping.")
            step_log.extend(tested_rows)
            break

        best = tested.iloc[0]
        best_gene = best["gene"]
        best_p = float(best["p"])
        best_c_index = float(best["c_index"])

        print(f"Best candidate: {best_gene}")
        print(f"Best p-value: {best_p:.4g}")
        print(f"Best C-index: {best_c_index:.4f}")

        step_log.extend(tested_rows)

        if best_p >= FORWARD_ALPHA:
            print("No candidate passed the forward alpha threshold. Stopping forward selection.")
            break

        selected.append(best_gene)
        remaining.remove(best_gene)

        print(f"Added gene: {best_gene}")
        print(f"Selected genes: {selected}")

        while len(selected) > 1:
            cph_full = fit_cox(df, time_col, event_col, selected)

            if cph_full is None:
                print("Backward check skipped because the full selected model failed.")
                break

            pvals = cph_full.summary["p"].reindex(selected)
            worst_gene = pvals.idxmax()
            worst_p = float(pvals.loc[worst_gene])

            step_log.append({
                "endpoint": endpoint_label,
                "step": step,
                "phase": "backward",
                "gene": worst_gene,
                "gene_symbol_clean": clean_gene_symbol(worst_gene),
                "p": worst_p,
                "c_index": float(cph_full.concordance_index_),
                "selected_before": len(selected),
            })

            if worst_p <= BACKWARD_ALPHA:
                print("Backward check passed.")
                break

            selected.remove(worst_gene)
            remaining.append(worst_gene)

            print(f"Removed gene during backward check: {worst_gene}")
            print(f"Backward p-value: {worst_p:.4g}")

        print("")

    final_model = fit_cox(df, time_col, event_col, selected)

    if final_model is None or len(selected) == 0:
        selected_table = pd.DataFrame(columns=[
            "endpoint", "gene", "gene_symbol_clean", "coef",
            "hr_per_sd", "p", "c_index"
        ])
    else:
        selected_table = final_model.summary.loc[selected].reset_index()
        selected_table = selected_table.rename(columns={
            "covariate": "gene",
            "exp(coef)": "hr_per_sd",
        })

        selected_table["endpoint"] = endpoint_label
        selected_table["gene_symbol_clean"] = selected_table["gene"].map(clean_gene_symbol)
        selected_table["c_index"] = float(final_model.concordance_index_)

        keep_cols = [
            "endpoint",
            "gene",
            "gene_symbol_clean",
            "coef",
            "hr_per_sd",
            "se(coef)",
            "z",
            "p",
            "exp(coef) lower 95%",
            "exp(coef) upper 95%",
            "c_index",
        ]

        selected_table = selected_table[[c for c in keep_cols if c in selected_table.columns]]

    step_log = pd.DataFrame(step_log)

    print("")
    print(f"Final selected genes for {endpoint_label}: {len(selected)}")

    if final_model is not None and len(selected) > 0:
        print(f"Final C-index: {final_model.concordance_index_:.4f}")

    for gene in selected:
        print(f"  {gene}")

    return selected_table, step_log


def main():
    print("=" * 80)
    print("Conditional Cox Markov-Blanket-like feature selection")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Processed directory: {PROCESSED_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print("")

    expression_path = PROCESSED_DIR / f"GSE238110_DOG2_expression_log2cpm_matched_top{TOP_N_VARIABLE_GENES}var.csv"
    clinical_path = PROCESSED_DIR / "GSE238110_DOG2_clinical_matched_indexed.csv"
    candidate_path = RESULTS_DIR / "GSE238110_mb_candidate_genes_from_univariate_cox.csv"

    expression = pd.read_csv(expression_path, index_col=0)
    clinical = pd.read_csv(clinical_path, index_col=0)
    candidates = pd.read_csv(candidate_path)

    candidate_genes = candidates["gene"].dropna().astype(str).drop_duplicates().tolist()
    candidate_genes = [g for g in candidate_genes if g in expression.columns]

    print(f"Expression matrix: {expression.shape}")
    print(f"Clinical table: {clinical.shape}")
    print(f"Candidate genes found in expression matrix: {len(candidate_genes)}")

    common_samples = clinical.index.intersection(expression.index)
    clinical = clinical.loc[common_samples].copy()
    expression = expression.loc[common_samples, candidate_genes].copy()

    expression = standardize_expression(expression)
    candidate_genes = [g for g in candidate_genes if g in expression.columns]

    print(f"Matched samples: {len(common_samples)}")
    print(f"Standardized candidate matrix: {expression.shape}")
    print("")

    all_selected = []
    all_steps = []

    for endpoint_key, endpoint in ENDPOINTS.items():
        time_col = endpoint["time_col"]
        event_col = endpoint["event_col"]
        endpoint_label = endpoint["label"]

        clinical[time_col] = pd.to_numeric(clinical[time_col], errors="coerce")
        clinical[event_col] = pd.to_numeric(clinical[event_col], errors="coerce")

        df = clinical[[time_col, event_col]].join(expression, how="inner")

        selected_table, step_log = forward_backward_selection(
            df=df,
            candidate_genes=candidate_genes,
            time_col=time_col,
            event_col=event_col,
            endpoint_label=endpoint_label,
        )

        selected_out = RESULTS_DIR / f"GSE238110_{endpoint_key}_conditional_cox_mb_selected.csv"
        step_out = RESULTS_DIR / f"GSE238110_{endpoint_key}_conditional_cox_mb_step_log.csv"

        selected_table.to_csv(selected_out, index=False)
        step_log.to_csv(step_out, index=False)

        print("")
        print(f"Saved selected table: {selected_out}")
        print(f"Saved step log: {step_out}")
        print("")

        all_selected.append(selected_table)
        all_steps.append(step_log)

    combined_selected = pd.concat(all_selected, axis=0, ignore_index=True)
    combined_out = RESULTS_DIR / "GSE238110_conditional_cox_mb_selected_combined.csv"
    combined_selected.to_csv(combined_out, index=False)

    if not combined_selected.empty:
        dfi_genes = set(combined_selected.loc[combined_selected["endpoint"] == "DFI", "gene"])
        os_genes = set(combined_selected.loc[combined_selected["endpoint"] == "OS", "gene"])
        overlap = sorted(dfi_genes & os_genes)

        print("=" * 80)
        print("Combined selected gene summary")
        print("=" * 80)
        print(f"DFI selected genes: {len(dfi_genes)}")
        print(f"OS selected genes: {len(os_genes)}")
        print(f"Overlap genes: {len(overlap)}")

        if overlap:
            print("Overlap gene list:")
            for gene in overlap:
                print(f"  {gene}")

    print("")
    print(f"Saved combined selected table: {combined_out}")
    print("Done.")


if __name__ == "__main__":
    main()
