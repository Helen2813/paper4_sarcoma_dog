from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
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
SCREEN_FILE = "GSE238110_nested_cv_train_univariate_screen_top1000_per_fold.csv"
OBSERVED_MODULE_FILE = "GSE238110_RNA_nested_cv_module_benchmark.csv"

N_OUTER_SPLITS = 5
N_RANDOM_REPEATS = 30
RANDOM_GENE_COUNT = 300
MIN_MODULE_SIZE = 4
MAX_MODULES_FOR_COX = 20
CORRELATION_DISTANCE_THRESHOLD = 0.70
COX_PENALIZER = 0.05
RANDOM_SEED = 42

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

RANDOM_SOURCES = [
    "train_variance_top5000",
    "train_screen_top1000",
    "train_screen_top500",
]


def load_data():
    expression = pd.read_csv(PROCESSED_DIR / EXPRESSION_FILE, index_col=0)
    clinical = pd.read_csv(PROCESSED_DIR / CLINICAL_FILE, index_col=0)
    screens = pd.read_csv(RESULTS_DIR / SCREEN_FILE)
    observed_modules = pd.read_csv(RESULTS_DIR / OBSERVED_MODULE_FILE)

    common_samples = clinical.index.intersection(expression.index)
    expression = expression.loc[common_samples].copy()
    clinical = clinical.loc[common_samples].copy()

    return expression, clinical, screens, observed_modules


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
        folds.append({
            "endpoint": endpoint_label,
            "fold": fold_idx,
            "train_samples": samples[train_idx],
            "test_samples": samples[test_idx],
        })

    return folds


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


def build_modules(train_expression_z, genes):
    genes = [g for g in genes if g in train_expression_z.columns]
    genes = list(dict.fromkeys(genes))

    if len(genes) < MIN_MODULE_SIZE:
        return {}, pd.DataFrame()

    x = train_expression_z[genes].copy()

    corr = x.corr(method="spearman").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    dist = 1.0 - corr.abs()
    np.fill_diagonal(dist.values, 0.0)

    condensed = squareform(dist.values, checks=False)
    z = linkage(condensed, method="average")

    labels = fcluster(
        z,
        t=CORRELATION_DISTANCE_THRESHOLD,
        criterion="distance",
    )

    module_map = pd.DataFrame({
        "gene": genes,
        "module_id": labels,
    })

    counts = module_map["module_id"].value_counts()
    keep_modules = counts[counts >= MIN_MODULE_SIZE].index
    module_map = module_map[module_map["module_id"].isin(keep_modules)].copy()

    if module_map.empty:
        return {}, module_map

    module_map["module_label"] = "M" + module_map["module_id"].astype(str)

    modules = {
        module_label: sorted(part["gene"].tolist())
        for module_label, part in module_map.groupby("module_label")
    }

    return modules, module_map


def compute_module_scores(train_z, test_z, modules):
    train_scores = pd.DataFrame(index=train_z.index)
    test_scores = pd.DataFrame(index=test_z.index)
    variance_rows = []

    for module_label, genes in modules.items():
        genes = [g for g in genes if g in train_z.columns and g in test_z.columns]

        if len(genes) < MIN_MODULE_SIZE:
            continue

        pca = PCA(n_components=1, random_state=RANDOM_SEED)
        train_values = pca.fit_transform(train_z[genes].values).ravel()
        test_values = pca.transform(test_z[genes].values).ravel()

        train_scores[module_label] = train_values
        test_scores[module_label] = test_values

        variance_rows.append({
            "module_label": module_label,
            "n_genes": len(genes),
            "module_pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
        })

    return train_scores, test_scores, pd.DataFrame(variance_rows)


def score_modules(train_clinical, test_clinical, train_scores, test_scores, time_col, event_col):
    module_cols = list(train_scores.columns)

    if len(module_cols) == 0:
        return np.nan, "no_modules", []

    module_variance = train_scores.var(axis=0).sort_values(ascending=False)
    module_cols = module_variance.head(MAX_MODULES_FOR_COX).index.tolist()

    train_df = train_clinical[[time_col, event_col]].join(train_scores[module_cols], how="inner").dropna()
    test_df = test_clinical[[time_col, event_col]].join(test_scores[module_cols], how="inner").dropna()

    if train_df.shape[0] < 30 or test_df.shape[0] < 5:
        return np.nan, "too_few_samples", module_cols

    if train_df[event_col].sum() < 5 or test_df[event_col].sum() < 2:
        return np.nan, "too_few_events", module_cols

    cph = CoxPHFitter(penalizer=COX_PENALIZER)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            cph.fit(
                train_df[[time_col, event_col] + module_cols],
                duration_col=time_col,
                event_col=event_col,
                fit_options={"max_steps": 500},
            )

        risk = cph.predict_partial_hazard(test_df[module_cols]).values.ravel()

        c_index = concordance_index(
            test_df[time_col].values,
            -risk,
            test_df[event_col].values,
        )

        return float(c_index), "", module_cols

    except Exception as e:
        return np.nan, str(e), module_cols


def score_gene_module_set(train_clinical, test_clinical, train_expression, test_expression, genes, time_col, event_col):
    genes = [g for g in genes if g in train_expression.columns and g in test_expression.columns]
    genes = list(dict.fromkeys(genes))

    if len(genes) < MIN_MODULE_SIZE:
        return {
            "c_index": np.nan,
            "error": "too_few_input_genes",
            "n_input_genes": len(genes),
            "n_modules": 0,
            "n_module_genes": 0,
            "used_modules": "",
            "used_genes": "",
            "mean_module_pc1_explained_variance": np.nan,
        }

    train_z, test_z = standardize_train_test(
        train_expression[genes].copy(),
        test_expression[genes].copy(),
    )

    modules, module_map = build_modules(train_z, genes)

    if len(modules) == 0:
        return {
            "c_index": np.nan,
            "error": "no_modules_after_clustering",
            "n_input_genes": len(genes),
            "n_modules": 0,
            "n_module_genes": 0,
            "used_modules": "",
            "used_genes": "",
            "mean_module_pc1_explained_variance": np.nan,
        }

    train_scores, test_scores, module_variance = compute_module_scores(train_z, test_z, modules)

    c_index, error, used_modules = score_modules(
        train_clinical=train_clinical,
        test_clinical=test_clinical,
        train_scores=train_scores,
        test_scores=test_scores,
        time_col=time_col,
        event_col=event_col,
    )

    used_genes = []
    for module_label in used_modules:
        used_genes.extend(modules.get(module_label, []))

    if not module_variance.empty:
        mean_explained = module_variance[
            module_variance["module_label"].isin(used_modules)
        ]["module_pc1_explained_variance"].mean()
    else:
        mean_explained = np.nan

    return {
        "c_index": c_index,
        "error": error,
        "n_input_genes": len(genes),
        "n_modules": len(used_modules),
        "n_module_genes": len(set(used_genes)),
        "used_modules": ";".join(used_modules),
        "used_genes": ";".join(sorted(set(used_genes))),
        "mean_module_pc1_explained_variance": mean_explained,
    }


def source_genes(source_name, train_expression, screens, endpoint_label, fold_idx):
    if source_name == "train_variance_top5000":
        variances = train_expression.var(axis=0).sort_values(ascending=False)
        return variances.head(5000).index.tolist()

    fold_screen = screens[
        (screens["endpoint"] == endpoint_label) &
        (screens["fold"] == fold_idx)
    ].copy()

    fold_screen["p"] = pd.to_numeric(fold_screen["p"], errors="coerce")
    fold_screen = fold_screen.sort_values("p", na_position="last")

    if source_name == "train_screen_top1000":
        return fold_screen["gene"].dropna().astype(str).head(1000).tolist()

    if source_name == "train_screen_top500":
        return fold_screen["gene"].dropna().astype(str).head(500).tolist()

    raise ValueError(f"Unknown source: {source_name}")


def run_random_module_controls(expression, clinical, screens):
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for endpoint_label, endpoint in ENDPOINTS.items():
        time_col = endpoint["time_col"]
        event_col = endpoint["event_col"]

        folds = get_fold_samples(clinical, endpoint_label, time_col, event_col)

        print("=" * 80)
        print(f"Random module controls for {endpoint_label}")
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

            for source_name in RANDOM_SOURCES:
                pool = source_genes(
                    source_name=source_name,
                    train_expression=train_expression,
                    screens=screens,
                    endpoint_label=endpoint_label,
                    fold_idx=fold_idx,
                )

                pool = [g for g in pool if g in train_expression.columns]

                print(f"  Source: {source_name}; pool genes: {len(pool)}")

                if len(pool) < RANDOM_GENE_COUNT:
                    print("  Skipping source because pool is smaller than random gene count.")
                    continue

                for repeat in range(1, N_RANDOM_REPEATS + 1):
                    genes = rng.choice(pool, size=RANDOM_GENE_COUNT, replace=False).tolist()

                    result = score_gene_module_set(
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
                        "repeat": repeat,
                        **result,
                    })

                subset = pd.DataFrame([
                    row for row in rows
                    if row["endpoint"] == endpoint_label
                    and row["fold"] == fold_idx
                    and row["random_source"] == source_name
                ])

                valid = subset["c_index"].dropna()

                if valid.shape[0] > 0:
                    print(
                        f"    random module C-index: "
                        f"mean={valid.mean():.3f}, "
                        f"median={valid.median():.3f}, "
                        f"q90={valid.quantile(0.90):.3f}, "
                        f"max={valid.max():.3f}"
                    )
                else:
                    print("    no valid random module scores")

            print("")

    return pd.DataFrame(rows)


def summarize_random_controls(random_results):
    if random_results.empty:
        return pd.DataFrame()

    summary = (
        random_results
        .groupby(["endpoint", "random_source"], dropna=False)
        .agg(
            n_valid=("c_index", "count"),
            mean_c_index=("c_index", "mean"),
            std_c_index=("c_index", "std"),
            median_c_index=("c_index", "median"),
            q75_c_index=("c_index", lambda x: np.nanquantile(x, 0.75)),
            q90_c_index=("c_index", lambda x: np.nanquantile(x, 0.90)),
            q95_c_index=("c_index", lambda x: np.nanquantile(x, 0.95)),
            max_c_index=("c_index", "max"),
            mean_n_modules=("n_modules", "mean"),
            mean_n_module_genes=("n_module_genes", "mean"),
            mean_module_pc1_explained_variance=("mean_module_pc1_explained_variance", "mean"),
        )
        .reset_index()
        .sort_values(["endpoint", "random_source"])
    )

    return summary


def method_vs_random_percentiles(observed_modules, random_results):
    rows = []

    observed = observed_modules.copy()
    observed["c_index"] = pd.to_numeric(observed["c_index"], errors="coerce")

    valid_random = random_results[np.isfinite(random_results["c_index"])].copy()

    for _, obs in observed.iterrows():
        endpoint = obs["endpoint"]
        fold = int(obs["fold"])
        obs_c = obs["c_index"]

        if not np.isfinite(obs_c):
            continue

        for source_name in RANDOM_SOURCES:
            subset = valid_random[
                (valid_random["endpoint"] == endpoint) &
                (valid_random["fold"] == fold) &
                (valid_random["random_source"] == source_name)
            ]

            if subset.empty:
                continue

            percentile = float((subset["c_index"] <= obs_c).mean())

            rows.append({
                "endpoint": endpoint,
                "fold": fold,
                "method": "observed_train_screened_modules",
                "observed_c_index": obs_c,
                "observed_n_modules": obs.get("n_modules", np.nan),
                "observed_n_module_genes": obs.get("n_module_genes", np.nan),
                "random_source": source_name,
                "random_mean_c_index": float(subset["c_index"].mean()),
                "random_median_c_index": float(subset["c_index"].median()),
                "random_q75_c_index": float(np.nanquantile(subset["c_index"], 0.75)),
                "random_q90_c_index": float(np.nanquantile(subset["c_index"], 0.90)),
                "random_q95_c_index": float(np.nanquantile(subset["c_index"], 0.95)),
                "random_max_c_index": float(subset["c_index"].max()),
                "observed_percentile_vs_random": percentile,
                "observed_minus_random_mean": obs_c - float(subset["c_index"].mean()),
                "observed_minus_random_median": obs_c - float(subset["c_index"].median()),
            })

    return pd.DataFrame(rows)


def summarize_percentiles(percentiles):
    if percentiles.empty:
        return pd.DataFrame()

    summary = (
        percentiles
        .groupby(["endpoint", "random_source"], dropna=False)
        .agg(
            n_folds=("observed_percentile_vs_random", "count"),
            mean_observed_c_index=("observed_c_index", "mean"),
            mean_random_mean_c_index=("random_mean_c_index", "mean"),
            mean_percentile=("observed_percentile_vs_random", "mean"),
            median_percentile=("observed_percentile_vs_random", "median"),
            mean_delta_vs_random_mean=("observed_minus_random_mean", "mean"),
            median_delta_vs_random_median=("observed_minus_random_median", "median"),
        )
        .reset_index()
        .sort_values(["endpoint", "mean_percentile"], ascending=[True, False])
    )

    return summary


def observed_summary(observed_modules):
    observed = observed_modules.copy()
    observed["c_index"] = pd.to_numeric(observed["c_index"], errors="coerce")

    summary = (
        observed
        .groupby("endpoint", dropna=False)
        .agg(
            n_folds=("c_index", "count"),
            mean_c_index=("c_index", "mean"),
            std_c_index=("c_index", "std"),
            median_c_index=("c_index", "median"),
            min_c_index=("c_index", "min"),
            max_c_index=("c_index", "max"),
            mean_n_modules=("n_modules", "mean"),
            mean_n_module_genes=("n_module_genes", "mean"),
        )
        .reset_index()
    )

    return summary


def main():
    print("=" * 80)
    print("RNA module random-control benchmark")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Processed directory: {PROCESSED_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Random repeats per endpoint/fold/source: {N_RANDOM_REPEATS}")
    print(f"Random gene count per module run: {RANDOM_GENE_COUNT}")
    print("")
    print("Design:")
    print("  Observed modules come from train-screened top 300 genes.")
    print("  Random controls sample 300 genes from train-only pools.")
    print("  Each random set is clustered into modules and evaluated with the same Cox procedure.")
    print("")

    expression, clinical, screens, observed_modules = load_data()

    print("Loaded data:")
    print(f"  Expression matrix: {expression.shape}")
    print(f"  Clinical table: {clinical.shape}")
    print(f"  Screens table: {screens.shape}")
    print(f"  Observed module rows: {observed_modules.shape[0]}")
    print("")

    random_results = run_random_module_controls(
        expression=expression,
        clinical=clinical,
        screens=screens,
    )

    random_summary = summarize_random_controls(random_results)
    percentiles = method_vs_random_percentiles(observed_modules, random_results)
    percentile_summary = summarize_percentiles(percentiles)
    observed = observed_summary(observed_modules)

    random_results_path = RESULTS_DIR / "GSE238110_RNA_module_random_controls.csv"
    random_summary_path = RESULTS_DIR / "GSE238110_RNA_module_random_control_summary.csv"
    percentiles_path = RESULTS_DIR / "GSE238110_RNA_module_vs_random_percentiles.csv"
    percentile_summary_path = RESULTS_DIR / "GSE238110_RNA_module_vs_random_percentile_summary.csv"
    observed_summary_path = RESULTS_DIR / "GSE238110_RNA_observed_module_nested_cv_summary.csv"

    random_results.to_csv(random_results_path, index=False)
    random_summary.to_csv(random_summary_path, index=False)
    percentiles.to_csv(percentiles_path, index=False)
    percentile_summary.to_csv(percentile_summary_path, index=False)
    observed.to_csv(observed_summary_path, index=False)

    print("=" * 80)
    print("Observed module nested-CV summary")
    print("=" * 80)
    print(observed.to_string(index=False))
    print("")

    print("=" * 80)
    print("Random module-control summary")
    print("=" * 80)
    print(random_summary.to_string(index=False))
    print("")

    print("=" * 80)
    print("Observed modules vs random module controls")
    print("=" * 80)
    print(percentile_summary.to_string(index=False))
    print("")

    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("If observed modules exceed random modules from screened pools, module-level RNA signal is stronger than generic screened-gene aggregation.")
    print("If observed modules do not exceed screened random modules, RNA should be framed as broad prognostic programs rather than a uniquely selected module model.")
    print("A strong DFI module result should still be externally tested by ortholog mapping and human validation.")
    print("")

    print("Saved:")
    print(random_results_path)
    print(random_summary_path)
    print(percentiles_path)
    print(percentile_summary_path)
    print(observed_summary_path)
    print("Done.")


if __name__ == "__main__":
    main()
