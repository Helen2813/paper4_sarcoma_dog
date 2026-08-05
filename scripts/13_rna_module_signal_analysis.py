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
MASTER_FILE = "GSE238110_RNA_master_candidate_evidence_table.csv"
NESTED_SCREEN_FILE = "GSE238110_nested_cv_train_univariate_screen_top1000_per_fold.csv"

FULL_MODULE_TOP_GENES = 500
NESTED_MODULE_TOP_GENES = 300
MIN_MODULE_SIZE = 4
MAX_MODULES_FOR_COX = 20
CORRELATION_DISTANCE_THRESHOLD = 0.70
N_OUTER_SPLITS = 5
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


def clean_gene_symbol(gene):
    gene = str(gene)
    tail = gene.rsplit("_", 1)[-1]
    return gene.rsplit("_", 1)[0] if tail.isdigit() else gene


def load_data():
    expression = pd.read_csv(PROCESSED_DIR / EXPRESSION_FILE, index_col=0)
    clinical = pd.read_csv(PROCESSED_DIR / CLINICAL_FILE, index_col=0)
    master = pd.read_csv(RESULTS_DIR / MASTER_FILE)
    screens = pd.read_csv(RESULTS_DIR / NESTED_SCREEN_FILE)

    common_samples = clinical.index.intersection(expression.index)
    expression = expression.loc[common_samples].copy()
    clinical = clinical.loc[common_samples].copy()

    return expression, clinical, master, screens


def standardize_train_apply(train_x, test_x=None):
    train_x = train_x.replace([np.inf, -np.inf], np.nan)
    medians = train_x.median(axis=0)
    train_x = train_x.fillna(medians)

    means = train_x.mean(axis=0)
    stds = train_x.std(axis=0).replace(0, np.nan)

    train_z = (train_x - means) / stds
    valid_cols = train_z.columns[train_z.notna().all(axis=0)]
    train_z = train_z[valid_cols]

    if test_x is None:
        return train_z

    test_x = test_x.replace([np.inf, -np.inf], np.nan)
    test_x = test_x.fillna(medians)
    test_z = (test_x - means) / stds
    test_z = test_z[valid_cols]

    return train_z, test_z


def build_modules(expression_z, genes):
    genes = [g for g in genes if g in expression_z.columns]
    genes = list(dict.fromkeys(genes))

    if len(genes) < MIN_MODULE_SIZE:
        return pd.DataFrame(), {}

    x = expression_z[genes].copy()

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
        "gene_symbol_clean": [clean_gene_symbol(g) for g in genes],
        "module_id": labels,
    })

    counts = module_map["module_id"].value_counts()
    keep_modules = counts[counts >= MIN_MODULE_SIZE].index
    module_map = module_map[module_map["module_id"].isin(keep_modules)].copy()

    module_map["module_label"] = "M" + module_map["module_id"].astype(str)

    modules = {
        module_label: sorted(part["gene"].tolist())
        for module_label, part in module_map.groupby("module_label")
    }

    return module_map, modules


def compute_module_scores(train_expression_z, test_expression_z, modules):
    train_scores = pd.DataFrame(index=train_expression_z.index)
    test_scores = pd.DataFrame(index=test_expression_z.index)

    module_variance = {}

    for module_label, genes in modules.items():
        genes = [g for g in genes if g in train_expression_z.columns and g in test_expression_z.columns]

        if len(genes) < MIN_MODULE_SIZE:
            continue

        pca = PCA(n_components=1, random_state=RANDOM_SEED)
        train_values = pca.fit_transform(train_expression_z[genes].values).ravel()
        test_values = pca.transform(test_expression_z[genes].values).ravel()

        train_scores[module_label] = train_values
        test_scores[module_label] = test_values
        module_variance[module_label] = float(pca.explained_variance_ratio_[0])

    return train_scores, test_scores, module_variance


def fit_and_score_modules(train_clinical, test_clinical, train_scores, test_scores, time_col, event_col):
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


def full_cohort_module_association(expression, clinical, master):
    rows = []
    module_gene_rows = []

    master = master.copy()
    master["rna_evidence_priority_score"] = pd.to_numeric(
        master["rna_evidence_priority_score"],
        errors="coerce",
    ).fillna(0)

    top_genes = (
        master
        .sort_values("rna_evidence_priority_score", ascending=False)
        .head(FULL_MODULE_TOP_GENES)["gene"]
        .dropna()
        .astype(str)
        .tolist()
    )

    expression_z = standardize_train_apply(expression[top_genes].copy())
    module_map, modules = build_modules(expression_z, top_genes)

    if module_map.empty:
        return pd.DataFrame(), pd.DataFrame()

    for _, row in module_map.iterrows():
        module_gene_rows.append({
            "analysis": "full_cohort",
            "module_label": row["module_label"],
            "gene": row["gene"],
            "gene_symbol_clean": row["gene_symbol_clean"],
        })

    scores, _, module_variance = compute_module_scores(expression_z, expression_z, modules)

    for endpoint_label, endpoint in ENDPOINTS.items():
        time_col = endpoint["time_col"]
        event_col = endpoint["event_col"]

        clinical[time_col] = pd.to_numeric(clinical[time_col], errors="coerce")
        clinical[event_col] = pd.to_numeric(clinical[event_col], errors="coerce")

        for module_label in scores.columns:
            df = clinical[[time_col, event_col]].join(scores[[module_label]], how="inner").dropna()

            if df.shape[0] < 30 or df[event_col].sum() < 5:
                continue

            cph = CoxPHFitter(penalizer=COX_PENALIZER)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    cph.fit(df, duration_col=time_col, event_col=event_col)

                summary = cph.summary.loc[module_label]

                rows.append({
                    "endpoint": endpoint_label,
                    "module_label": module_label,
                    "n_genes": len(modules[module_label]),
                    "n": df.shape[0],
                    "events": int(df[event_col].sum()),
                    "coef": float(summary["coef"]),
                    "hr": float(summary["exp(coef)"]),
                    "p": float(summary["p"]),
                    "c_index": float(cph.concordance_index_),
                    "module_pc1_explained_variance": module_variance.get(module_label, np.nan),
                    "genes": ";".join(modules[module_label]),
                    "gene_symbols": ";".join([clean_gene_symbol(g) for g in modules[module_label]]),
                })

            except Exception:
                continue

    return pd.DataFrame(rows), pd.DataFrame(module_gene_rows)


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


def nested_cv_module_benchmark(expression, clinical, screens):
    result_rows = []
    module_gene_rows = []

    for endpoint_label, endpoint in ENDPOINTS.items():
        time_col = endpoint["time_col"]
        event_col = endpoint["event_col"]

        folds = get_fold_samples(clinical, endpoint_label, time_col, event_col)

        print("=" * 80)
        print(f"Nested CV module benchmark for {endpoint_label}")
        print("=" * 80)

        for fold in folds:
            fold_idx = fold["fold"]
            train_samples = fold["train_samples"]
            test_samples = fold["test_samples"]

            train_clinical = clinical.loc[train_samples].copy()
            test_clinical = clinical.loc[test_samples].copy()
            train_expression = expression.loc[train_samples].copy()
            test_expression = expression.loc[test_samples].copy()

            fold_screen = screens[
                (screens["endpoint"] == endpoint_label) &
                (screens["fold"] == fold_idx)
            ].copy()

            fold_screen["p"] = pd.to_numeric(fold_screen["p"], errors="coerce")
            fold_screen = fold_screen.sort_values("p", na_position="last")

            genes = fold_screen["gene"].dropna().head(NESTED_MODULE_TOP_GENES).astype(str).tolist()
            genes = [g for g in genes if g in train_expression.columns]

            print(f"Endpoint {endpoint_label}, fold {fold_idx}")
            print(f"  Train samples: {len(train_samples)}")
            print(f"  Test samples: {len(test_samples)}")
            print(f"  Input genes for modules: {len(genes)}")

            train_z, test_z = standardize_train_apply(
                train_expression[genes].copy(),
                test_expression[genes].copy(),
            )

            module_map, modules = build_modules(train_z, genes)

            print(f"  Modules found: {len(modules)}")

            if len(modules) == 0:
                result_rows.append({
                    "endpoint": endpoint_label,
                    "fold": fold_idx,
                    "method": "train_screened_modules",
                    "n_modules": 0,
                    "n_module_genes": 0,
                    "c_index": np.nan,
                    "error": "no_modules",
                })
                continue

            for _, row in module_map.iterrows():
                module_gene_rows.append({
                    "analysis": "nested_cv",
                    "endpoint": endpoint_label,
                    "fold": fold_idx,
                    "module_label": row["module_label"],
                    "gene": row["gene"],
                    "gene_symbol_clean": row["gene_symbol_clean"],
                })

            train_scores, test_scores, module_variance = compute_module_scores(train_z, test_z, modules)

            c_index, error, used_modules = fit_and_score_modules(
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

            result_rows.append({
                "endpoint": endpoint_label,
                "fold": fold_idx,
                "method": "train_screened_modules",
                "n_modules": len(used_modules),
                "n_module_genes": len(set(used_genes)),
                "c_index": c_index,
                "error": error,
                "used_modules": ";".join(used_modules),
                "used_genes": ";".join(sorted(set(used_genes))),
            })

            print(f"  Used modules: {len(used_modules)}")
            print(f"  Used module genes: {len(set(used_genes))}")
            print(f"  Held-out C-index: {c_index}")
            print(f"  Error: {error}")
            print("")

    return pd.DataFrame(result_rows), pd.DataFrame(module_gene_rows)


def summarize_results(full_assoc, nested_results):
    print("")
    print("=" * 80)
    print("Full-cohort module associations, top rows")
    print("=" * 80)

    if full_assoc.empty:
        print("No full-cohort module associations were found.")
    else:
        cols = [
            "endpoint",
            "module_label",
            "n_genes",
            "p",
            "c_index",
            "module_pc1_explained_variance",
            "gene_symbols",
        ]
        print(
            full_assoc
            .sort_values(["endpoint", "p"])
            [cols]
            .head(30)
            .to_string(index=False)
        )

    print("")
    print("=" * 80)
    print("Nested CV module benchmark")
    print("=" * 80)

    if nested_results.empty:
        print("No nested CV module results were found.")
        return

    print(nested_results.to_string(index=False))

    print("")
    print("Nested CV module summary:")
    summary = (
        nested_results
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

    print(summary.to_string(index=False))


def main():
    print("=" * 80)
    print("RNA module signal analysis")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Processed directory: {PROCESSED_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print("")
    print("Design:")
    print("  Full-cohort module discovery is exploratory.")
    print("  Nested-CV module benchmark uses train-fold screens only.")
    print("")

    expression, clinical, master, screens = load_data()

    print("Loaded data:")
    print(f"  Expression matrix: {expression.shape}")
    print(f"  Clinical table: {clinical.shape}")
    print(f"  Master evidence table: {master.shape}")
    print(f"  Nested screen table: {screens.shape}")
    print("")

    full_assoc, full_module_genes = full_cohort_module_association(
        expression=expression,
        clinical=clinical.copy(),
        master=master,
    )

    nested_results, nested_module_genes = nested_cv_module_benchmark(
        expression=expression,
        clinical=clinical.copy(),
        screens=screens,
    )

    module_genes = pd.concat(
        [full_module_genes, nested_module_genes],
        axis=0,
        ignore_index=True,
    )

    full_assoc_path = RESULTS_DIR / "GSE238110_RNA_full_cohort_module_associations.csv"
    nested_results_path = RESULTS_DIR / "GSE238110_RNA_nested_cv_module_benchmark.csv"
    module_genes_path = RESULTS_DIR / "GSE238110_RNA_module_gene_membership.csv"

    full_assoc.to_csv(full_assoc_path, index=False)
    nested_results.to_csv(nested_results_path, index=False)
    module_genes.to_csv(module_genes_path, index=False)

    summarize_results(full_assoc, nested_results)

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("If module-level nested C-index exceeds gene-level methods, RNA signal is better interpreted as pathway/module-level.")
    print("If module-level nested C-index remains near screened-random controls, the RNA layer should be used mainly for candidate prioritization and cross-species evidence.")
    print("Next steps after this script: ortholog mapping and human RNA validation.")

    print("")
    print("Saved:")
    print(full_assoc_path)
    print(nested_results_path)
    print(module_genes_path)
    print("Done.")


if __name__ == "__main__":
    main()
