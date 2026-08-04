from pathlib import Path
import sys
import time
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
ALPHAS = [0.05, 0.10, 0.20]
ALGORITHMS = ["IAMB", "GSMB"]
COX_PENALIZER = 0.05
RANDOM_SEED = 42

PYCAUSALFS_CANDIDATES = [
    PROJECT_ROOT / "external" / "pyCausalFS" / "pyCausalFS" / "pyCausalFS",
    Path(r"C:\Users\olegk\Desktop\Thesis_master2\pyCausalFS\pyCausalFS\pyCausalFS"),
    Path(r"C:\Users\olegk\Desktop\Thesis_v3\pyCausalFS\pyCausalFS\pyCausalFS"),
]

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


def add_pycausalfs_to_path():
    for path in PYCAUSALFS_CANDIDATES:
        if path.exists():
            sys.path.insert(0, str(path))
            print(f"Using pyCausalFS path: {path}")
            return True

    print("pyCausalFS was not found.")
    print("Expected one of these paths:")
    for path in PYCAUSALFS_CANDIDATES:
        print(f"  {path}")
    print("")
    print("Fix:")
    print("  1. Create external/pyCausalFS inside this project, or")
    print("  2. Edit PYCAUSALFS_CANDIDATES at the top of this script.")
    return False


def patch_fisher_z():
    import CBD.MBs.common.fisher_z_test as fz

    def partial_corr_coef(data, x, y, z=None, ridge_lambda=1e-6):
        if z is None:
            has_z = False
        elif isinstance(z, (int, np.integer)):
            has_z = True
            z = [int(z)]
        elif hasattr(z, "__len__"):
            has_z = len(z) > 0
            if has_z:
                z = [int(zi) for zi in z]
        else:
            has_z = True
            z = [int(z)]

        if not has_z:
            var_x = data[x, x]
            var_y = data[y, y]
            cov_xy = data[x, y]

            if var_x < 1e-10 or var_y < 1e-10:
                return 0.0

            r = cov_xy / np.sqrt(var_x * var_y)
            return float(np.clip(r, -0.999999, 0.999999))

        vars_list = [x, y] + z
        n = len(vars_list)

        sub_cov = np.zeros((n, n))

        for i, vi in enumerate(vars_list):
            for j, vj in enumerate(vars_list):
                sub_cov[i, j] = data[vi, vj]

        sub_cov = sub_cov + ridge_lambda * np.eye(n)

        try:
            precision = np.linalg.inv(sub_cov)
        except np.linalg.LinAlgError:
            precision = np.linalg.pinv(sub_cov)

        p_xx = precision[0, 0]
        p_yy = precision[1, 1]
        p_xy = precision[0, 1]

        if p_xx < 1e-10 or p_yy < 1e-10:
            return 0.0

        r = -p_xy / np.sqrt(p_xx * p_yy)
        return float(np.clip(r, -0.999999, 0.999999))

    fz.partial_corr_coef = partial_corr_coef
    print("Patched Fisher-Z partial correlation.")


def clean_gene_symbol(gene):
    gene = str(gene)
    tail = gene.rsplit("_", 1)[-1]
    return gene.rsplit("_", 1)[0] if tail.isdigit() else gene


def standardize_matrix(x):
    x = x.copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(axis=0))

    means = x.mean(axis=0)
    stds = x.std(axis=0).replace(0, np.nan)

    x = (x - means) / stds
    x = x.dropna(axis=1)

    return x


def build_mb_matrix(expression, target):
    x = expression.values.astype(float)
    y = target.values.astype(float)

    rng = np.random.default_rng(RANDOM_SEED)
    x = x + rng.normal(0.0, 1e-8, size=x.shape)

    y = np.log1p(y)
    y = (y - np.nanmean(y)) / np.nanstd(y)

    full_data = np.column_stack([x, y]).astype(float)
    target_idx = x.shape[1]

    return full_data, target_idx


def run_mb_algorithm(algorithm_name, data_matrix, target_idx, alpha):
    from CBD.MBs.IAMB import IAMB
    from CBD.MBs.GSMB import GSMB

    start = time.time()

    try:
        if algorithm_name == "IAMB":
            result = IAMB(
                data=data_matrix,
                target=target_idx,
                is_discrete=False,
                alaph=alpha,
            )
        elif algorithm_name == "GSMB":
            result = GSMB(
                data=data_matrix,
                target=target_idx,
                is_discrete=False,
                alaph=alpha,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm_name}")

        elapsed = time.time() - start

        if isinstance(result, tuple):
            indices = list(result[0])
        elif result is None:
            indices = []
        else:
            indices = list(result)

        indices = [
            int(i)
            for i in indices
            if int(i) != target_idx and 0 <= int(i) < data_matrix.shape[1] - 1
        ]

        return indices, elapsed, ""

    except Exception as e:
        elapsed = time.time() - start
        return [], elapsed, str(e)


def evaluate_selected_genes(clinical, expression, genes, time_col, event_col):
    if len(genes) == 0:
        return {
            "n_selected": 0,
            "cox_c_index": np.nan,
            "cox_error": "no_genes_selected",
        }

    df = clinical[[time_col, event_col]].join(expression[genes], how="inner").dropna()

    if df.shape[0] < 30:
        return {
            "n_selected": len(genes),
            "cox_c_index": np.nan,
            "cox_error": "too_few_samples",
        }

    cph = CoxPHFitter(penalizer=COX_PENALIZER)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            cph.fit(df, duration_col=time_col, event_col=event_col)

        return {
            "n_selected": len(genes),
            "cox_c_index": float(cph.concordance_index_),
            "cox_error": "",
        }

    except Exception as e:
        return {
            "n_selected": len(genes),
            "cox_c_index": np.nan,
            "cox_error": str(e),
        }


def main():
    print("=" * 80)
    print("True Markov Blanket ablation with IAMB/GSMB")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Processed directory: {PROCESSED_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print("")

    if not add_pycausalfs_to_path():
        raise SystemExit(1)

    patch_fisher_z()

    expression_path = PROCESSED_DIR / f"GSE238110_DOG2_expression_log2cpm_matched_top{TOP_N_VARIABLE_GENES}var.csv"
    clinical_path = PROCESSED_DIR / "GSE238110_DOG2_clinical_matched_indexed.csv"
    candidate_path = RESULTS_DIR / "GSE238110_mb_candidate_genes_from_univariate_cox.csv"

    expression = pd.read_csv(expression_path, index_col=0)
    clinical = pd.read_csv(clinical_path, index_col=0)
    candidates = pd.read_csv(candidate_path)

    candidate_genes = candidates["gene"].dropna().astype(str).drop_duplicates().tolist()
    candidate_genes = [g for g in candidate_genes if g in expression.columns]

    common_samples = clinical.index.intersection(expression.index)
    expression = expression.loc[common_samples, candidate_genes].copy()
    clinical = clinical.loc[common_samples].copy()

    expression = standardize_matrix(expression)
    candidate_genes = [g for g in candidate_genes if g in expression.columns]

    print(f"Expression matrix: {expression.shape}")
    print(f"Clinical table: {clinical.shape}")
    print(f"Candidate genes: {len(candidate_genes)}")
    print("")

    all_results = []
    selected_gene_rows = []

    for endpoint_key, endpoint in ENDPOINTS.items():
        time_col = endpoint["time_col"]
        event_col = endpoint["event_col"]
        endpoint_label = endpoint["label"]

        clinical[time_col] = pd.to_numeric(clinical[time_col], errors="coerce")
        clinical[event_col] = pd.to_numeric(clinical[event_col], errors="coerce")

        valid_samples = clinical[[time_col, event_col]].dropna().index
        clinical_ep = clinical.loc[valid_samples].copy()
        expression_ep = expression.loc[valid_samples].copy()

        print("=" * 80)
        print(f"Endpoint: {endpoint_label}")
        print("=" * 80)
        print(f"Samples: {clinical_ep.shape[0]}")
        print(f"Events: {int(clinical_ep[event_col].sum())}")
        print(f"Features: {expression_ep.shape[1]}")
        print("")

        data_matrix, target_idx = build_mb_matrix(
            expression_ep,
            clinical_ep[time_col],
        )

        feature_names = expression_ep.columns.tolist()

        for alpha in ALPHAS:
            for algorithm in ALGORITHMS:
                print(f"Running {algorithm}, alpha={alpha}")

                indices, elapsed, error = run_mb_algorithm(
                    algorithm_name=algorithm,
                    data_matrix=data_matrix,
                    target_idx=target_idx,
                    alpha=alpha,
                )

                genes = [feature_names[i] for i in indices]
                metrics = evaluate_selected_genes(
                    clinical=clinical_ep,
                    expression=expression_ep,
                    genes=genes,
                    time_col=time_col,
                    event_col=event_col,
                )

                row = {
                    "endpoint": endpoint_label,
                    "algorithm": algorithm,
                    "alpha": alpha,
                    "n_selected": len(genes),
                    "selected_genes": ";".join(genes),
                    "time_sec": elapsed,
                    "cox_c_index": metrics["cox_c_index"],
                    "error": error,
                    "cox_error": metrics["cox_error"],
                }

                all_results.append(row)

                for rank, gene in enumerate(genes, start=1):
                    selected_gene_rows.append({
                        "endpoint": endpoint_label,
                        "algorithm": algorithm,
                        "alpha": alpha,
                        "gene_rank": rank,
                        "gene": gene,
                        "gene_symbol_clean": clean_gene_symbol(gene),
                    })

                print(f"  Selected genes: {len(genes)}")
                print(f"  Cox C-index: {metrics['cox_c_index']}")
                print(f"  Time: {elapsed:.1f} sec")

                if error:
                    print(f"  MB error: {error}")
                if metrics["cox_error"]:
                    print(f"  Cox error: {metrics['cox_error']}")

                if len(genes) > 0:
                    print("  Gene list:")
                    for gene in genes:
                        print(f"    {gene}")

                print("")

    results = pd.DataFrame(all_results)
    selected_genes = pd.DataFrame(selected_gene_rows)

    results_path = RESULTS_DIR / "GSE238110_true_iamb_gsmb_ablation_summary.csv"
    genes_path = RESULTS_DIR / "GSE238110_true_iamb_gsmb_ablation_selected_genes.csv"

    results.to_csv(results_path, index=False)
    selected_genes.to_csv(genes_path, index=False)

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    display_cols = [
        "endpoint",
        "algorithm",
        "alpha",
        "n_selected",
        "cox_c_index",
        "time_sec",
        "error",
        "cox_error",
    ]

    print(results[display_cols].to_string(index=False))

    print("")
    print("Saved:")
    print(results_path)
    print(genes_path)
    print("Done.")


if __name__ == "__main__":
    main()
