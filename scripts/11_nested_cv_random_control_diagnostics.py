from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_GENES_FILE = "GSE238110_DOG2_expression_log2cpm_matched_allgenes.csv"
CLINICAL_FILE = "GSE238110_DOG2_clinical_matched_indexed.csv"

BENCHMARK_FILE = "GSE238110_nested_cv_method_benchmark.csv"
SCREEN_FILE = "GSE238110_nested_cv_train_univariate_screen_top1000_per_fold.csv"
SELECTED_GENES_FILE = "GSE238110_nested_cv_selected_genes.csv"

N_OUTER_SPLITS = 5
TOP_N_TRAIN_VARIANCE = 5000
RANDOM_REPEATS = 100
RANDOM_SEED = 42
COX_PENALIZER = 0.05

RANDOM_SIZES = [4, 5, 6, 7, 8, 10, 12, 15, 20, 25]

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


def clean_gene_symbol(gene):
    gene = str(gene)
    tail = gene.rsplit("_", 1)[-1]
    return gene.rsplit("_", 1)[0] if tail.isdigit() else gene


def load_data():
    expression = pd.read_csv(PROCESSED_DIR / ALL_GENES_FILE, index_col=0)
    clinical = pd.read_csv(PROCESSED_DIR / CLINICAL_FILE, index_col=0)

    common_samples = clinical.index.intersection(expression.index)
    expression = expression.loc[common_samples].copy()
    clinical = clinical.loc[common_samples].copy()

    benchmark = pd.read_csv(RESULTS_DIR / BENCHMARK_FILE)
    screens = pd.read_csv(RESULTS_DIR / SCREEN_FILE)

    selected_genes_path = RESULTS_DIR / SELECTED_GENES_FILE
    if selected_genes_path.exists():
        selected_genes = pd.read_csv(selected_genes_path)
    else:
        selected_genes = pd.DataFrame()

    return expression, clinical, benchmark, screens, selected_genes


def train_variance_genes(train_expression):
    variances = train_expression.var(axis=0).sort_values(ascending=False)
    return variances.head(TOP_N_TRAIN_VARIANCE).index.tolist()


def standardize_train_test(train_x, test_x):
    train_x = train_x.replace([np.inf, -np.inf], np.nan)
    test_x = test_x.replace([np.inf, -np.inf], np.nan)

    medians = train_x.median(axis=0)
    train_x = train_x.fillna(medians)
    test_x = test_x.fillna(medians)

    means = train_x.mean(axis=0)
    stds = train_x.std(axis=0).replace(0, np.nan)

    train_z = (train_x - means) / stds
    test_z = (test_x - means) / stds

    valid_cols = train_z.columns[train_z.notna().all(axis=0)]
    train_z = train_z[valid_cols]
    test_z = test_z[valid_cols]

    return train_z, test_z


def score_gene_set(train_clinical, test_clinical, train_expression, test_expression, genes, time_col, event_col):
    genes = [g for g in genes if g in train_expression.columns and g in test_expression.columns]
    genes = list(dict.fromkeys(genes))

    if len(genes) == 0:
        return np.nan, "no_genes"

    train_x, test_x = standardize_train_test(
        train_expression[genes].copy(),
        test_expression[genes].copy(),
    )

    genes = list(train_x.columns)

    if len(genes) == 0:
        return np.nan, "no_valid_genes_after_standardization"

    train_df = train_clinical[[time_col, event_col]].join(train_x, how="inner").dropna()
    test_df = test_clinical[[time_col, event_col]].join(test_x, how="inner").dropna()

    if train_df.shape[0] < 30 or test_df.shape[0] < 5:
        return np.nan, "too_few_samples"

    if train_df[event_col].sum() < 5 or test_df[event_col].sum() < 2:
        return np.nan, "too_few_events"

    cph = CoxPHFitter(penalizer=COX_PENALIZER)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            cph.fit(
                train_df[[time_col, event_col] + genes],
                duration_col=time_col,
                event_col=event_col,
                fit_options={"max_steps": 500},
            )

        risk = cph.predict_partial_hazard(test_df[genes]).values.ravel()

        c_index = concordance_index(
            test_df[time_col].values,
            -risk,
            test_df[event_col].values,
        )

        return float(c_index), ""

    except Exception as e:
        return np.nan, str(e)


def get_fold_samples(clinical, endpoint_label, time_col, event_col):
    clinical = clinical.copy()
    clinical[time_col] = pd.to_numeric(clinical[time_col], errors="coerce")
    clinical[event_col] = pd.to_numeric(clinical[event_col], errors="coerce")

    valid_samples = clinical[[time_col, event_col]].dropna().index
    endpoint_clinical = clinical.loc[valid_samples].copy()

    y_event = endpoint_clinical[event_col].astype(int).values
    samples = np.array(valid_samples)

    outer_cv = StratifiedKFold(
        n_splits=N_OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    folds = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(samples, y_event), start=1):
        train_samples = samples[train_idx]
        test_samples = samples[test_idx]

        folds.append({
            "endpoint": endpoint_label,
            "fold": fold_idx,
            "train_samples": train_samples,
            "test_samples": test_samples,
        })

    return folds


def random_control_source_genes(
    source_name,
    endpoint_label,
    fold_idx,
    train_expression,
    screens,
):
    if source_name == "train_variance_top5000":
        return train_variance_genes(train_expression)

    fold_screen = screens[
        (screens["endpoint"] == endpoint_label) &
        (screens["fold"] == fold_idx)
    ].copy()

    fold_screen["p"] = pd.to_numeric(fold_screen["p"], errors="coerce")
    fold_screen = fold_screen.sort_values("p", na_position="last")

    if source_name == "train_screen_top500":
        return fold_screen["gene"].dropna().head(500).tolist()

    if source_name == "train_screen_top1000":
        return fold_screen["gene"].dropna().head(1000).tolist()

    raise ValueError(f"Unknown source: {source_name}")


def run_random_diagnostics(expression, clinical, screens):
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    random_sources = [
        "train_variance_top5000",
        "train_screen_top1000",
        "train_screen_top500",
    ]

    for endpoint_label, endpoint in ENDPOINTS.items():
        time_col = endpoint["time_col"]
        event_col = endpoint["event_col"]

        folds = get_fold_samples(clinical, endpoint_label, time_col, event_col)

        print("=" * 80)
        print(f"Random-control diagnostics for {endpoint_label}")
        print("=" * 80)

        for fold in folds:
            fold_idx = fold["fold"]
            train_samples = fold["train_samples"]
            test_samples = fold["test_samples"]

            train_clinical = clinical.loc[train_samples].copy()
            test_clinical = clinical.loc[test_samples].copy()
            train_expression = expression.loc[train_samples].copy()
            test_expression = expression.loc[test_samples].copy()

            print(f"Endpoint {endpoint_label}, fold {fold_idx}")

            for source_name in random_sources:
                source_genes = random_control_source_genes(
                    source_name=source_name,
                    endpoint_label=endpoint_label,
                    fold_idx=fold_idx,
                    train_expression=train_expression,
                    screens=screens,
                )

                source_genes = [g for g in source_genes if g in train_expression.columns]

                print(f"  Source: {source_name}; genes available: {len(source_genes)}")

                for size in RANDOM_SIZES:
                    if len(source_genes) < size:
                        continue

                    for repeat in range(1, RANDOM_REPEATS + 1):
                        genes = rng.choice(source_genes, size=size, replace=False).tolist()

                        c_index, error = score_gene_set(
                            train_clinical=train_clinical,
                            test_clinical=test_clinical,
                            train_expression=train_expression,
                            test_expression=test_expression,
                            genes=genes,
                            time_col=time_col,
                            event_col=event_col,
                        )

                        rows.append({
                            "endpoint": endpoint_label,
                            "fold": fold_idx,
                            "random_source": source_name,
                            "panel_size": size,
                            "repeat": repeat,
                            "c_index": c_index,
                            "error": error,
                            "selected_genes": ";".join(genes),
                        })

                    size_rows = [r for r in rows if r["endpoint"] == endpoint_label and r["fold"] == fold_idx and r["random_source"] == source_name and r["panel_size"] == size]
                    valid_scores = [r["c_index"] for r in size_rows if np.isfinite(r["c_index"])]

                    if valid_scores:
                        print(
                            f"    size={size:2d}: "
                            f"mean={np.mean(valid_scores):.3f}, "
                            f"median={np.median(valid_scores):.3f}, "
                            f"max={np.max(valid_scores):.3f}"
                        )

            print("")

    return pd.DataFrame(rows)


def summarize_random_controls(random_results):
    summary = (
        random_results
        .groupby(["endpoint", "random_source", "panel_size"], dropna=False)
        .agg(
            n_valid=("c_index", "count"),
            mean_c_index=("c_index", "mean"),
            std_c_index=("c_index", "std"),
            median_c_index=("c_index", "median"),
            q75_c_index=("c_index", lambda x: np.nanquantile(x, 0.75)),
            q90_c_index=("c_index", lambda x: np.nanquantile(x, 0.90)),
            q95_c_index=("c_index", lambda x: np.nanquantile(x, 0.95)),
            max_c_index=("c_index", "max"),
        )
        .reset_index()
        .sort_values(["endpoint", "random_source", "panel_size"])
    )

    return summary


def nearest_available_size(size, available_sizes):
    available_sizes = sorted(set(available_sizes))
    return min(available_sizes, key=lambda x: abs(x - size))


def method_vs_random_percentiles(benchmark, random_results):
    rows = []

    valid_random = random_results[np.isfinite(random_results["c_index"])].copy()

    methods_to_compare = benchmark[
        ~benchmark["method"].isin([
            "clinical_only",
            "random_top10_mean20",
            "random_survival_forest_candidate500",
        ])
    ].copy()

    for _, method_row in methods_to_compare.iterrows():
        endpoint = method_row["endpoint"]
        fold = int(method_row["fold"])
        method = method_row["method"]
        c_index = method_row["c_index"]
        n_genes = int(method_row["n_selected_genes"])

        if not np.isfinite(c_index) or n_genes <= 0:
            continue

        for source_name in ["train_variance_top5000", "train_screen_top1000", "train_screen_top500"]:
            subset_source = valid_random[
                (valid_random["endpoint"] == endpoint) &
                (valid_random["fold"] == fold) &
                (valid_random["random_source"] == source_name)
            ]

            if subset_source.empty:
                continue

            size = nearest_available_size(n_genes, subset_source["panel_size"].unique())

            subset = subset_source[subset_source["panel_size"] == size]

            if subset.empty:
                continue

            percentile = float((subset["c_index"] <= c_index).mean())
            random_mean = float(subset["c_index"].mean())
            random_median = float(subset["c_index"].median())
            random_q90 = float(np.nanquantile(subset["c_index"], 0.90))
            random_max = float(subset["c_index"].max())

            rows.append({
                "endpoint": endpoint,
                "fold": fold,
                "method": method,
                "method_c_index": c_index,
                "method_n_genes": n_genes,
                "random_source": source_name,
                "matched_random_panel_size": size,
                "random_mean_c_index": random_mean,
                "random_median_c_index": random_median,
                "random_q90_c_index": random_q90,
                "random_max_c_index": random_max,
                "method_percentile_vs_random": percentile,
                "method_minus_random_mean": c_index - random_mean,
                "method_minus_random_median": c_index - random_median,
            })

    return pd.DataFrame(rows)


def summarize_method_percentiles(percentiles):
    if percentiles.empty:
        return pd.DataFrame()

    summary = (
        percentiles
        .groupby(["endpoint", "method", "random_source"], dropna=False)
        .agg(
            n_folds=("method_percentile_vs_random", "count"),
            mean_percentile=("method_percentile_vs_random", "mean"),
            median_percentile=("method_percentile_vs_random", "median"),
            mean_delta_vs_random_mean=("method_minus_random_mean", "mean"),
            median_delta_vs_random_median=("method_minus_random_median", "median"),
            mean_method_c_index=("method_c_index", "mean"),
            mean_random_mean_c_index=("random_mean_c_index", "mean"),
        )
        .reset_index()
        .sort_values(["endpoint", "random_source", "mean_percentile"], ascending=[True, True, False])
    )

    return summary


def selected_gene_stability(selected_genes):
    if selected_genes.empty:
        return pd.DataFrame()

    required = {"endpoint", "method", "fold", "gene"}
    if not required.issubset(set(selected_genes.columns)):
        return pd.DataFrame()

    total_folds = (
        selected_genes[["endpoint", "method", "fold"]]
        .drop_duplicates()
        .groupby(["endpoint", "method"])
        .size()
        .rename("n_method_folds")
        .reset_index()
    )

    stability = (
        selected_genes
        .groupby(["endpoint", "method", "gene"], dropna=False)
        .agg(
            selected_in_folds=("fold", "nunique"),
            mean_gene_rank=("gene_rank", "mean"),
        )
        .reset_index()
        .merge(total_folds, on=["endpoint", "method"], how="left")
    )

    stability["selection_frequency"] = stability["selected_in_folds"] / stability["n_method_folds"]
    stability["gene_symbol_clean"] = stability["gene"].map(clean_gene_symbol)

    stability = stability.sort_values(
        ["endpoint", "method", "selection_frequency", "selected_in_folds", "mean_gene_rank"],
        ascending=[True, True, False, False, True],
    )

    return stability


def main():
    print("=" * 80)
    print("Nested CV random-control diagnostics")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Processed directory: {PROCESSED_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Random repeats per source/size/fold: {RANDOM_REPEATS}")
    print("")

    expression, clinical, benchmark, screens, selected_genes = load_data()

    print("Loaded data:")
    print(f"  Expression matrix: {expression.shape}")
    print(f"  Clinical table: {clinical.shape}")
    print(f"  Benchmark rows: {benchmark.shape[0]}")
    print(f"  Screen rows: {screens.shape[0]}")
    print(f"  Selected-gene rows: {selected_genes.shape[0]}")
    print("")

    random_results = run_random_diagnostics(
        expression=expression,
        clinical=clinical,
        screens=screens,
    )

    random_summary = summarize_random_controls(random_results)
    percentiles = method_vs_random_percentiles(benchmark, random_results)
    percentile_summary = summarize_method_percentiles(percentiles)
    stability = selected_gene_stability(selected_genes)

    random_results_path = RESULTS_DIR / "GSE238110_nested_cv_expanded_random_controls.csv"
    random_summary_path = RESULTS_DIR / "GSE238110_nested_cv_expanded_random_control_summary.csv"
    percentiles_path = RESULTS_DIR / "GSE238110_nested_cv_method_vs_random_percentiles.csv"
    percentile_summary_path = RESULTS_DIR / "GSE238110_nested_cv_method_vs_random_percentile_summary.csv"
    stability_path = RESULTS_DIR / "GSE238110_nested_cv_selected_gene_stability.csv"

    random_results.to_csv(random_results_path, index=False)
    random_summary.to_csv(random_summary_path, index=False)
    percentiles.to_csv(percentiles_path, index=False)
    percentile_summary.to_csv(percentile_summary_path, index=False)
    stability.to_csv(stability_path, index=False)

    print("=" * 80)
    print("Expanded random-control summary")
    print("=" * 80)
    print(random_summary.to_string(index=False))
    print("")

    print("=" * 80)
    print("Method vs random percentile summary")
    print("=" * 80)

    if percentile_summary.empty:
        print("No method-vs-random percentile summary could be computed.")
    else:
        print(percentile_summary.to_string(index=False))

    print("")
    print("=" * 80)
    print("Selected gene stability, top rows")
    print("=" * 80)

    if stability.empty:
        print("No selected-gene stability table could be computed.")
    else:
        display_cols = [
            "endpoint",
            "method",
            "gene",
            "gene_symbol_clean",
            "selected_in_folds",
            "n_method_folds",
            "selection_frequency",
            "mean_gene_rank",
        ]
        print(stability[display_cols].head(80).to_string(index=False))

    print("")
    print("Saved:")
    print(random_results_path)
    print(random_summary_path)
    print(percentiles_path)
    print(percentile_summary_path)
    print(stability_path)
    print("Done.")


if __name__ == "__main__":
    main()
