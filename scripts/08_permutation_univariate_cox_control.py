from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_N_VARIABLE_GENES = 5000
N_PERMUTATIONS = 10
RANDOM_SEED = 42

ENDPOINTS = {
    "dfi": {
        "time_col": "dfi_time",
        "event_col": "dfi_event",
        "observed_file": "GSE238110_dfi_univariate_cox_top5000var.csv",
        "label": "DFI",
    },
    "os": {
        "time_col": "os_time",
        "event_col": "os_event",
        "observed_file": "GSE238110_os_univariate_cox_top5000var.csv",
        "label": "OS",
    },
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


def standardize_series(x):
    x = pd.to_numeric(x, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median())

    std = x.std()
    if std == 0 or not np.isfinite(std):
        return None

    return (x - x.mean()) / std


def fit_univariate_cox(time_values, event_values, gene_values):
    gene_z = standardize_series(gene_values)

    if gene_z is None:
        return np.nan

    df = pd.DataFrame({
        "time": pd.to_numeric(time_values, errors="coerce"),
        "event": pd.to_numeric(event_values, errors="coerce"),
        "gene": gene_z,
    }).dropna()

    if df.shape[0] < 30:
        return np.nan

    if df["event"].sum() < 5:
        return np.nan

    cph = CoxPHFitter()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            cph.fit(df, duration_col="time", event_col="event")

        return float(cph.summary.loc["gene", "p"])

    except Exception:
        return np.nan


def summarize_pvalues(p_values):
    p = np.asarray(p_values, dtype=float)
    q = bh_fdr(p)

    return {
        "n_tested": int(np.isfinite(p).sum()),
        "p_lt_0_05": int(np.nansum(p < 0.05)),
        "p_lt_0_01": int(np.nansum(p < 0.01)),
        "q_lt_0_25": int(np.nansum(q < 0.25)),
        "q_lt_0_10": int(np.nansum(q < 0.10)),
        "q_lt_0_05": int(np.nansum(q < 0.05)),
        "min_p": float(np.nanmin(p)),
        "min_q": float(np.nanmin(q)),
    }


def observed_summary(endpoint_key, endpoint):
    path = RESULTS_DIR / endpoint["observed_file"]
    observed = pd.read_csv(path)

    observed["p"] = pd.to_numeric(observed["p"], errors="coerce")
    observed["q"] = pd.to_numeric(observed["q"], errors="coerce")

    return {
        "endpoint": endpoint["label"],
        "permutation": "observed",
        "n_tested": int(observed["p"].notna().sum()),
        "p_lt_0_05": int((observed["p"] < 0.05).sum()),
        "p_lt_0_01": int((observed["p"] < 0.01).sum()),
        "q_lt_0_25": int((observed["q"] < 0.25).sum()),
        "q_lt_0_10": int((observed["q"] < 0.10).sum()),
        "q_lt_0_05": int((observed["q"] < 0.05).sum()),
        "min_p": float(observed["p"].min()),
        "min_q": float(observed["q"].min()),
    }


def main():
    print("=" * 80)
    print("Permutation negative control for univariate Cox screening")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Processed directory: {PROCESSED_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Top variable genes: {TOP_N_VARIABLE_GENES}")
    print(f"Permutations per endpoint: {N_PERMUTATIONS}")
    print("")
    print("Permutation design:")
    print("  The time/event outcome pairs are permuted across samples.")
    print("  Gene expression is left unchanged.")
    print("  This preserves censoring and event-time distribution.")
    print("")

    expression_path = PROCESSED_DIR / f"GSE238110_DOG2_expression_log2cpm_matched_top{TOP_N_VARIABLE_GENES}var.csv"
    clinical_path = PROCESSED_DIR / "GSE238110_DOG2_clinical_matched_indexed.csv"

    expression = pd.read_csv(expression_path, index_col=0)
    clinical = pd.read_csv(clinical_path, index_col=0)

    common_samples = clinical.index.intersection(expression.index)
    expression = expression.loc[common_samples].copy()
    clinical = clinical.loc[common_samples].copy()

    print(f"Expression matrix: {expression.shape}")
    print(f"Clinical table: {clinical.shape}")
    print("")

    rng = np.random.default_rng(RANDOM_SEED)

    summary_rows = []
    observed_rows = []

    for endpoint_key, endpoint in ENDPOINTS.items():
        label = endpoint["label"]
        time_col = endpoint["time_col"]
        event_col = endpoint["event_col"]

        print("=" * 80)
        print(f"Endpoint: {label}")
        print("=" * 80)

        clinical[time_col] = pd.to_numeric(clinical[time_col], errors="coerce")
        clinical[event_col] = pd.to_numeric(clinical[event_col], errors="coerce")

        valid = clinical[[time_col, event_col]].dropna().index
        x = expression.loc[valid].copy()
        y_time = clinical.loc[valid, time_col].copy()
        y_event = clinical.loc[valid, event_col].copy()

        print(f"Samples: {len(valid)}")
        print(f"Events: {int(y_event.sum())}")
        print(f"Genes: {x.shape[1]}")

        obs = observed_summary(endpoint_key, endpoint)
        observed_rows.append(obs)

        print("")
        print("Observed screen:")
        print(
            f"  p<0.05: {obs['p_lt_0_05']} | "
            f"q<0.10: {obs['q_lt_0_10']} | "
            f"q<0.05: {obs['q_lt_0_05']} | "
            f"min p: {obs['min_p']:.3e}"
        )
        print("")

        outcome_pairs = pd.DataFrame({
            "time": y_time.values,
            "event": y_event.values,
        }, index=valid)

        for perm_idx in range(1, N_PERMUTATIONS + 1):
            print(f"Running permutation {perm_idx}/{N_PERMUTATIONS} for {label}")

            perm_order = rng.permutation(len(outcome_pairs))
            permuted_time = outcome_pairs["time"].values[perm_order]
            permuted_event = outcome_pairs["event"].values[perm_order]

            p_values = []

            for gene_idx, gene in enumerate(x.columns, start=1):
                if gene_idx % 500 == 0:
                    print(f"  Tested {gene_idx}/{x.shape[1]} genes")

                p = fit_univariate_cox(
                    time_values=permuted_time,
                    event_values=permuted_event,
                    gene_values=x[gene],
                )
                p_values.append(p)

            row = summarize_pvalues(p_values)
            row["endpoint"] = label
            row["permutation"] = perm_idx
            summary_rows.append(row)

            print(
                f"  Result: p<0.05={row['p_lt_0_05']}, "
                f"q<0.10={row['q_lt_0_10']}, "
                f"q<0.05={row['q_lt_0_05']}, "
                f"min p={row['min_p']:.3e}"
            )
            print("")

    observed_df = pd.DataFrame(observed_rows)
    permutation_df = pd.DataFrame(summary_rows)

    combined = pd.concat([observed_df, permutation_df], axis=0, ignore_index=True)

    out_path = RESULTS_DIR / "GSE238110_permutation_univariate_cox_negative_control_summary.csv"
    combined.to_csv(out_path, index=False)

    print("=" * 80)
    print("Negative control summary")
    print("=" * 80)

    for endpoint_label in combined["endpoint"].unique():
        print("")
        print(f"Endpoint: {endpoint_label}")

        obs = combined[
            (combined["endpoint"] == endpoint_label) &
            (combined["permutation"].astype(str) == "observed")
        ]

        perm = combined[
            (combined["endpoint"] == endpoint_label) &
            (combined["permutation"].astype(str) != "observed")
        ]

        print("Observed:")
        print(obs[[
            "p_lt_0_05",
            "p_lt_0_01",
            "q_lt_0_25",
            "q_lt_0_10",
            "q_lt_0_05",
            "min_p",
            "min_q",
        ]].to_string(index=False))

        print("")
        print("Permutations:")
        print(perm[[
            "p_lt_0_05",
            "p_lt_0_01",
            "q_lt_0_25",
            "q_lt_0_10",
            "q_lt_0_05",
            "min_p",
            "min_q",
        ]].describe().to_string())

    print("")
    print("Saved:")
    print(out_path)
    print("Done.")


if __name__ == "__main__":
    main()
