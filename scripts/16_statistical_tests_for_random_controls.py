from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GENE_LEVEL_FILE = "GSE238110_nested_cv_method_vs_random_percentiles.csv"
MODULE_LEVEL_FILE = "GSE238110_RNA_module_vs_random_percentiles.csv"

OUTPUT_TESTS = "GSE238110_random_control_paired_statistical_tests.csv"
OUTPUT_KEY_TESTS = "GSE238110_random_control_key_statistical_tests.csv"


KEY_COMPARISONS = [
    {
        "analysis": "gene_level",
        "endpoint": "DFI",
        "method": "conditional_cox_mb",
        "random_source": "train_screen_top1000",
    },
    {
        "analysis": "gene_level",
        "endpoint": "DFI",
        "method": "conditional_cox_mb",
        "random_source": "train_screen_top500",
    },
    {
        "analysis": "gene_level",
        "endpoint": "DFI",
        "method": "conditional_cox_mb",
        "random_source": "train_variance_top5000",
    },
    {
        "analysis": "module_level",
        "endpoint": "DFI",
        "method": "observed_train_screened_modules",
        "random_source": "train_screen_top1000",
    },
    {
        "analysis": "module_level",
        "endpoint": "DFI",
        "method": "observed_train_screened_modules",
        "random_source": "train_screen_top500",
    },
    {
        "analysis": "module_level",
        "endpoint": "DFI",
        "method": "observed_train_screened_modules",
        "random_source": "train_variance_top5000",
    },
    {
        "analysis": "module_level",
        "endpoint": "OS",
        "method": "observed_train_screened_modules",
        "random_source": "train_screen_top1000",
    },
]


def read_csv_if_exists(filename):
    path = RESULTS_DIR / filename

    if not path.exists():
        print(f"Missing file: {path}")
        return pd.DataFrame()

    print(f"Loaded: {path}")
    return pd.read_csv(path)


def safe_wilcoxon(values, alternative):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    if np.allclose(values, 0):
        return 1.0

    try:
        return float(stats.wilcoxon(values, alternative=alternative, zero_method="wilcox").pvalue)
    except Exception:
        return np.nan


def safe_sign_test(values, alternative):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[~np.isclose(values, 0)]

    n = len(values)

    if n == 0:
        return np.nan

    n_positive = int(np.sum(values > 0))

    try:
        return float(stats.binomtest(n_positive, n=n, p=0.5, alternative=alternative).pvalue)
    except Exception:
        return np.nan


def summarize_differences(df, analysis_name, method_col, delta_col, percentile_col):
    rows = []

    required = {"endpoint", "random_source", method_col, delta_col}

    if not required.issubset(set(df.columns)):
        print(f"Skipping {analysis_name}; missing required columns.")
        return pd.DataFrame()

    df = df.copy()
    df[delta_col] = pd.to_numeric(df[delta_col], errors="coerce")

    if percentile_col in df.columns:
        df[percentile_col] = pd.to_numeric(df[percentile_col], errors="coerce")
    else:
        df[percentile_col] = np.nan

    group_cols = ["endpoint", method_col, "random_source"]

    for keys, part in df.groupby(group_cols, dropna=False):
        endpoint, method, random_source = keys

        deltas = part[delta_col].dropna().astype(float).values
        percentiles = part[percentile_col].dropna().astype(float).values

        n = len(deltas)

        if n == 0:
            continue

        n_positive = int(np.sum(deltas > 0))
        n_negative = int(np.sum(deltas < 0))
        n_zero = int(np.sum(np.isclose(deltas, 0)))

        rows.append({
            "analysis": analysis_name,
            "endpoint": endpoint,
            "method": method,
            "random_source": random_source,
            "n_pairs": n,
            "n_positive_deltas": n_positive,
            "n_negative_deltas": n_negative,
            "n_zero_deltas": n_zero,
            "mean_delta": float(np.mean(deltas)),
            "median_delta": float(np.median(deltas)),
            "std_delta": float(np.std(deltas, ddof=1)) if n > 1 else np.nan,
            "min_delta": float(np.min(deltas)),
            "max_delta": float(np.max(deltas)),
            "mean_percentile": float(np.mean(percentiles)) if len(percentiles) else np.nan,
            "median_percentile": float(np.median(percentiles)) if len(percentiles) else np.nan,
            "wilcoxon_p_greater_than_random": safe_wilcoxon(deltas, alternative="greater"),
            "wilcoxon_p_two_sided": safe_wilcoxon(deltas, alternative="two-sided"),
            "sign_test_p_greater_than_random": safe_sign_test(deltas, alternative="greater"),
            "sign_test_p_two_sided": safe_sign_test(deltas, alternative="two-sided"),
            "all_deltas": ";".join([f"{x:.6f}" for x in deltas]),
        })

    return pd.DataFrame(rows)


def build_gene_level_tests(gene_df):
    if gene_df.empty:
        return pd.DataFrame()

    return summarize_differences(
        df=gene_df,
        analysis_name="gene_level",
        method_col="method",
        delta_col="method_minus_random_mean",
        percentile_col="method_percentile_vs_random",
    )


def build_module_level_tests(module_df):
    if module_df.empty:
        return pd.DataFrame()

    return summarize_differences(
        df=module_df,
        analysis_name="module_level",
        method_col="method",
        delta_col="observed_minus_random_mean",
        percentile_col="observed_percentile_vs_random",
    )


def extract_key_tests(all_tests):
    if all_tests.empty:
        return pd.DataFrame()

    rows = []

    for query in KEY_COMPARISONS:
        subset = all_tests[
            (all_tests["analysis"] == query["analysis"]) &
            (all_tests["endpoint"] == query["endpoint"]) &
            (all_tests["method"] == query["method"]) &
            (all_tests["random_source"] == query["random_source"])
        ].copy()

        if not subset.empty:
            rows.append(subset.iloc[0].to_dict())

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def interpretation_label(row):
    mean_delta = row.get("mean_delta", np.nan)
    mean_percentile = row.get("mean_percentile", np.nan)
    n_positive = row.get("n_positive_deltas", 0)
    n_pairs = row.get("n_pairs", 0)

    if not np.isfinite(mean_delta):
        return "not_interpretable"

    if mean_delta > 0 and mean_percentile >= 0.70 and n_positive >= max(1, n_pairs - 1):
        return "consistent_advantage_over_random"

    if mean_delta > 0 and mean_percentile >= 0.60:
        return "modest_advantage_over_random"

    if abs(mean_delta) < 0.01 and 0.40 <= mean_percentile <= 0.60:
        return "near_random_performance"

    if mean_delta < 0:
        return "worse_than_random_on_average"

    return "mixed_or_weak_advantage"


def main():
    print("=" * 80)
    print("Paired statistical tests for observed-vs-random controls")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Results directory: {RESULTS_DIR}")
    print("")
    print("Important limitation:")
    print("  These tests are descriptive/supportive because the current benchmark has only 5 outer folds.")
    print("  With n=5, even a perfect two-sided sign test cannot go below p=0.0625.")
    print("  Repeated nested CV would be needed for a more powered paired test.")
    print("")

    gene_df = read_csv_if_exists(GENE_LEVEL_FILE)
    module_df = read_csv_if_exists(MODULE_LEVEL_FILE)

    gene_tests = build_gene_level_tests(gene_df)
    module_tests = build_module_level_tests(module_df)

    all_tests = pd.concat([gene_tests, module_tests], axis=0, ignore_index=True)

    if not all_tests.empty:
        all_tests["interpretation_label"] = all_tests.apply(interpretation_label, axis=1)

    key_tests = extract_key_tests(all_tests)

    all_tests_path = RESULTS_DIR / OUTPUT_TESTS
    key_tests_path = RESULTS_DIR / OUTPUT_KEY_TESTS

    all_tests.to_csv(all_tests_path, index=False)
    key_tests.to_csv(key_tests_path, index=False)

    print("=" * 80)
    print("Key paired test results")
    print("=" * 80)

    if key_tests.empty:
        print("No key tests were found.")
    else:
        display_cols = [
            "analysis",
            "endpoint",
            "method",
            "random_source",
            "n_pairs",
            "n_positive_deltas",
            "mean_delta",
            "median_delta",
            "mean_percentile",
            "median_percentile",
            "wilcoxon_p_greater_than_random",
            "sign_test_p_greater_than_random",
            "interpretation_label",
            "all_deltas",
        ]

        display_cols = [c for c in display_cols if c in key_tests.columns]
        print(key_tests[display_cols].to_string(index=False))

    print("")
    print("=" * 80)
    print("All tests, sorted by endpoint and mean delta")
    print("=" * 80)

    if all_tests.empty:
        print("No tests were computed.")
    else:
        display_cols = [
            "analysis",
            "endpoint",
            "method",
            "random_source",
            "n_pairs",
            "mean_delta",
            "mean_percentile",
            "wilcoxon_p_greater_than_random",
            "sign_test_p_greater_than_random",
            "interpretation_label",
        ]

        display_cols = [c for c in display_cols if c in all_tests.columns]

        print(
            all_tests[display_cols]
            .sort_values(["endpoint", "mean_delta"], ascending=[True, False])
            .to_string(index=False)
        )

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("Use these p-values as supportive diagnostics only, not as definitive claims.")
    print("For the current n=5 fold design, effect size, fold-level direction, and random-control percentiles are more informative than p<0.05.")
    print("If the human validation is promising, rerun the nested benchmark with repeated outer CV seeds.")

    print("")
    print("Saved:")
    print(all_tests_path)
    print(key_tests_path)
    print("Done.")


if __name__ == "__main__":
    main()
