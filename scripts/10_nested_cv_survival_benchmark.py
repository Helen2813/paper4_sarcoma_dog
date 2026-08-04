from pathlib import Path
import sys
import time
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

N_OUTER_SPLITS = 5
N_INNER_SPLITS = 3
TOP_N_TRAIN_VARIANCE = 5000
SCREEN_TOP_N = 500
ELASTIC_NET_TOP_N = 250
MAX_CONDITIONAL_SELECTED = 10
CONDITIONAL_FORWARD_ALPHA = 0.01
CONDITIONAL_BACKWARD_ALPHA = 0.05
COX_PENALIZER = 0.05
RANDOM_REPEATS = 20
RANDOM_SEED = 42

ELASTIC_NET_GRID = [
    {"penalizer": 0.01, "l1_ratio": 0.5},
    {"penalizer": 0.05, "l1_ratio": 0.5},
    {"penalizer": 0.10, "l1_ratio": 0.5},
    {"penalizer": 0.50, "l1_ratio": 0.5},
    {"penalizer": 0.01, "l1_ratio": 0.9},
    {"penalizer": 0.05, "l1_ratio": 0.9},
    {"penalizer": 0.10, "l1_ratio": 0.9},
    {"penalizer": 0.50, "l1_ratio": 0.9},
    {"penalizer": 0.01, "l1_ratio": 1.0},
    {"penalizer": 0.05, "l1_ratio": 1.0},
    {"penalizer": 0.10, "l1_ratio": 1.0},
    {"penalizer": 0.50, "l1_ratio": 1.0},
]

ENDPOINTS = {
    "dfi": {
        "label": "DFI",
        "time_col": "dfi_time",
        "event_col": "dfi_event",
    },
    "os": {
        "label": "OS",
        "time_col": "os_time",
        "event_col": "os_event",
    },
}

CLINICAL_CONTINUOUS = ["age", "weight", "PH"]
CLINICAL_CATEGORICAL = ["ALP", "treatment", "gender"]

PYCAUSALFS_CANDIDATES = [
    PROJECT_ROOT / "external" / "pyCausalFS" / "pyCausalFS" / "pyCausalFS",
    Path(r"C:\Users\olegk\Desktop\Thesis_master2\pyCausalFS\pyCausalFS\pyCausalFS"),
    Path(r"C:\Users\olegk\Desktop\Thesis_v3\pyCausalFS\pyCausalFS\pyCausalFS"),
]


try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    from sksurv.metrics import concordance_index_censored
    HAS_SKSURV = True
except Exception:
    HAS_SKSURV = False


def add_pycausalfs_to_path():
    for path in PYCAUSALFS_CANDIDATES:
        if path.exists():
            sys.path.insert(0, str(path))
            print(f"Using pyCausalFS path: {path}")
            return True

    print("pyCausalFS was not found. IAMB/GSMB methods will be skipped.")
    return False


def patch_fisher_z():
    try:
        import CBD.MBs.common.fisher_z_test as fz
    except Exception:
        return False

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
    return True


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

    return expression, clinical


def train_variance_filter(train_expression, all_expression):
    variances = train_expression.var(axis=0).sort_values(ascending=False)
    selected = variances.head(TOP_N_TRAIN_VARIANCE).index.tolist()
    return all_expression[selected].copy(), selected


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


def build_clinical_features(train_clinical, test_clinical):
    train_parts = []
    test_parts = []

    for col in CLINICAL_CONTINUOUS:
        if col not in train_clinical.columns:
            continue

        train_values = pd.to_numeric(train_clinical[col], errors="coerce")
        test_values = pd.to_numeric(test_clinical[col], errors="coerce")

        median = train_values.median()
        train_values = train_values.fillna(median)
        test_values = test_values.fillna(median)

        std = train_values.std()

        if std == 0 or not np.isfinite(std):
            continue

        train_parts.append(((train_values - train_values.mean()) / std).to_frame(col))
        test_parts.append(((test_values - train_values.mean()) / std).to_frame(col))

    for col in CLINICAL_CATEGORICAL:
        if col not in train_clinical.columns:
            continue

        train_values = train_clinical[col].astype("object").fillna("Missing")
        test_values = test_clinical[col].astype("object").fillna("Missing")

        train_dummies = pd.get_dummies(train_values, prefix=col, drop_first=True)
        test_dummies = pd.get_dummies(test_values, prefix=col, drop_first=True)

        test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)

        train_parts.append(train_dummies)
        test_parts.append(test_dummies)

    if train_parts:
        train_features = pd.concat(train_parts, axis=1)
        test_features = pd.concat(test_parts, axis=1)
    else:
        train_features = pd.DataFrame(index=train_clinical.index)
        test_features = pd.DataFrame(index=test_clinical.index)

    return train_features, test_features


def build_model_frames(
    train_clinical,
    test_clinical,
    train_expression,
    test_expression,
    genes,
    time_col,
    event_col,
    include_clinical,
):
    train_parts = []
    test_parts = []

    genes = [g for g in genes if g in train_expression.columns and g in test_expression.columns]

    if genes:
        train_gene, test_gene = standardize_train_test(
            train_expression[genes].copy(),
            test_expression[genes].copy(),
        )

        train_parts.append(train_gene)
        test_parts.append(test_gene)

    if include_clinical:
        train_clin, test_clin = build_clinical_features(train_clinical, test_clinical)
        train_parts.append(train_clin)
        test_parts.append(test_clin)

    if train_parts:
        train_x = pd.concat(train_parts, axis=1)
        test_x = pd.concat(test_parts, axis=1)
    else:
        train_x = pd.DataFrame(index=train_clinical.index)
        test_x = pd.DataFrame(index=test_clinical.index)

    train_x = train_x.loc[:, ~train_x.columns.duplicated()]
    test_x = test_x.reindex(columns=train_x.columns, fill_value=0)

    train_df = train_clinical[[time_col, event_col]].join(train_x, how="inner").dropna()
    test_df = test_clinical[[time_col, event_col]].join(test_x, how="inner").dropna()

    feature_cols = list(train_x.columns)

    return train_df, test_df, feature_cols


def fit_and_score_cox(
    train_clinical,
    test_clinical,
    train_expression,
    test_expression,
    genes,
    time_col,
    event_col,
    include_clinical=False,
    penalizer=COX_PENALIZER,
    l1_ratio=0.0,
):
    train_df, test_df, feature_cols = build_model_frames(
        train_clinical=train_clinical,
        test_clinical=test_clinical,
        train_expression=train_expression,
        test_expression=test_expression,
        genes=genes,
        time_col=time_col,
        event_col=event_col,
        include_clinical=include_clinical,
    )

    if len(feature_cols) == 0:
        return {
            "c_index": np.nan,
            "n_features_model": 0,
            "error": "no_features",
            "model": None,
            "feature_cols": [],
        }

    if train_df.shape[0] < 30 or test_df.shape[0] < 5:
        return {
            "c_index": np.nan,
            "n_features_model": len(feature_cols),
            "error": "too_few_samples",
            "model": None,
            "feature_cols": feature_cols,
        }

    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            cph.fit(
                train_df[[time_col, event_col] + feature_cols],
                duration_col=time_col,
                event_col=event_col,
                fit_options={"max_steps": 500},
            )

        risk = cph.predict_partial_hazard(test_df[feature_cols]).values.ravel()
        c_index = concordance_index(
            test_df[time_col].values,
            -risk,
            test_df[event_col].values,
        )

        return {
            "c_index": float(c_index),
            "n_features_model": len(feature_cols),
            "error": "",
            "model": cph,
            "feature_cols": feature_cols,
        }

    except Exception as e:
        return {
            "c_index": np.nan,
            "n_features_model": len(feature_cols),
            "error": str(e),
            "model": None,
            "feature_cols": feature_cols,
        }


def fit_cox_train_only(train_clinical, train_expression, genes, time_col, event_col):
    train_df, _, feature_cols = build_model_frames(
        train_clinical=train_clinical,
        test_clinical=train_clinical,
        train_expression=train_expression,
        test_expression=train_expression,
        genes=genes,
        time_col=time_col,
        event_col=event_col,
        include_clinical=False,
    )

    if len(feature_cols) == 0:
        return None

    cph = CoxPHFitter(penalizer=COX_PENALIZER)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            cph.fit(
                train_df[[time_col, event_col] + feature_cols],
                duration_col=time_col,
                event_col=event_col,
                fit_options={"max_steps": 500},
            )
        return cph

    except Exception:
        return None


def univariate_gene_pvalue(train_clinical, train_expression, gene, time_col, event_col):
    df = train_clinical[[time_col, event_col]].join(train_expression[[gene]], how="inner").dropna()

    if df.shape[0] < 30 or df[event_col].sum() < 5:
        return np.nan

    x = pd.to_numeric(df[gene], errors="coerce")
    std = x.std()

    if std == 0 or not np.isfinite(std):
        return np.nan

    df[gene] = (x - x.mean()) / std

    cph = CoxPHFitter()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            cph.fit(df[[time_col, event_col, gene]], duration_col=time_col, event_col=event_col)

        return float(cph.summary.loc[gene, "p"])

    except Exception:
        return np.nan


def run_univariate_screen(train_clinical, train_expression, time_col, event_col):
    rows = []

    for i, gene in enumerate(train_expression.columns, start=1):
        if i % 1000 == 0:
            print(f"    Univariate screen tested {i}/{train_expression.shape[1]} genes")

        p = univariate_gene_pvalue(train_clinical, train_expression, gene, time_col, event_col)

        rows.append({
            "gene": gene,
            "p": p,
        })

    out = pd.DataFrame(rows)
    out = out.sort_values("p", na_position="last")

    return out


def select_conditional_cox(train_clinical, train_expression, candidate_genes, time_col, event_col):
    selected = []
    remaining = list(candidate_genes)

    for step in range(1, MAX_CONDITIONAL_SELECTED + 1):
        best_gene = None
        best_p = np.inf

        print(f"    Conditional step {step}; selected={len(selected)}; remaining={len(remaining)}")

        for i, gene in enumerate(remaining, start=1):
            cph = fit_cox_train_only(
                train_clinical=train_clinical,
                train_expression=train_expression,
                genes=selected + [gene],
                time_col=time_col,
                event_col=event_col,
            )

            if cph is None or gene not in cph.summary.index:
                continue

            p = float(cph.summary.loc[gene, "p"])

            if p < best_p:
                best_p = p
                best_gene = gene

            if i % 200 == 0:
                print(f"      Tested {i}/{len(remaining)} candidates")

        if best_gene is None or best_p >= CONDITIONAL_FORWARD_ALPHA:
            print(f"    Stopping conditional selection; best p={best_p:.4g}")
            break

        selected.append(best_gene)
        remaining.remove(best_gene)

        while len(selected) > 1:
            cph = fit_cox_train_only(
                train_clinical=train_clinical,
                train_expression=train_expression,
                genes=selected,
                time_col=time_col,
                event_col=event_col,
            )

            if cph is None:
                break

            pvals = cph.summary["p"].reindex(selected)
            worst_gene = pvals.idxmax()
            worst_p = float(pvals.loc[worst_gene])

            if worst_p <= CONDITIONAL_BACKWARD_ALPHA:
                break

            selected.remove(worst_gene)
            remaining.append(worst_gene)
            print(f"    Removed during backward check: {worst_gene}; p={worst_p:.4g}")

    return selected


def add_pycausal_imports():
    try:
        from CBD.MBs.IAMB import IAMB
        from CBD.MBs.GSMB import GSMB
        return IAMB, GSMB
    except Exception:
        return None, None


def select_true_mb(train_clinical, train_expression, candidate_genes, time_col, algorithm_name, alpha=0.10):
    IAMB, GSMB = add_pycausal_imports()

    if algorithm_name == "IAMB" and IAMB is None:
        return [], "pyCausalFS import failed"

    if algorithm_name == "GSMB" and GSMB is None:
        return [], "pyCausalFS import failed"

    x = train_expression[candidate_genes].copy()
    x = x.replace([np.inf, -np.inf], np.nan).fillna(x.median(axis=0))

    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0).replace(0, np.nan)
    x = ((x - x_mean) / x_std).dropna(axis=1)

    genes = list(x.columns)

    y = pd.to_numeric(train_clinical[time_col], errors="coerce").loc[x.index]
    y = np.log1p(y)
    y = (y - y.mean()) / y.std()

    matrix = np.column_stack([x.values, y.values]).astype(float)
    matrix = matrix + np.random.default_rng(RANDOM_SEED).normal(0.0, 1e-8, size=matrix.shape)

    target_idx = x.shape[1]

    try:
        if algorithm_name == "IAMB":
            result = IAMB(data=matrix, target=target_idx, is_discrete=False, alaph=alpha)
        else:
            result = GSMB(data=matrix, target=target_idx, is_discrete=False, alaph=alpha)

        if isinstance(result, tuple):
            indices = list(result[0])
        elif result is None:
            indices = []
        else:
            indices = list(result)

        indices = [int(i) for i in indices if int(i) != target_idx and 0 <= int(i) < len(genes)]
        selected = [genes[i] for i in indices]

        return selected, ""

    except Exception as e:
        return [], str(e)


def tune_elastic_net(train_clinical, train_expression, candidate_genes, time_col, event_col):
    y = pd.to_numeric(train_clinical[event_col], errors="coerce")
    valid_samples = train_clinical[[time_col, event_col]].dropna().index
    y = y.loc[valid_samples].astype(int)

    if y.nunique() < 2:
        return ELASTIC_NET_GRID[0]

    inner_cv = StratifiedKFold(
        n_splits=N_INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    rows = []

    sample_index = np.array(valid_samples)

    for params in ELASTIC_NET_GRID:
        c_indices = []

        for inner_train_idx, inner_val_idx in inner_cv.split(sample_index, y.values):
            inner_train_samples = sample_index[inner_train_idx]
            inner_val_samples = sample_index[inner_val_idx]

            result = fit_and_score_cox(
                train_clinical=train_clinical.loc[inner_train_samples],
                test_clinical=train_clinical.loc[inner_val_samples],
                train_expression=train_expression.loc[inner_train_samples],
                test_expression=train_expression.loc[inner_val_samples],
                genes=candidate_genes,
                time_col=time_col,
                event_col=event_col,
                include_clinical=False,
                penalizer=params["penalizer"],
                l1_ratio=params["l1_ratio"],
            )

            if np.isfinite(result["c_index"]):
                c_indices.append(result["c_index"])

        mean_c = np.nan if len(c_indices) == 0 else float(np.mean(c_indices))

        rows.append({
            "penalizer": params["penalizer"],
            "l1_ratio": params["l1_ratio"],
            "mean_inner_c_index": mean_c,
        })

    tuning = pd.DataFrame(rows).sort_values("mean_inner_c_index", ascending=False, na_position="last")
    best = tuning.iloc[0].to_dict()

    return {
        "penalizer": float(best["penalizer"]),
        "l1_ratio": float(best["l1_ratio"]),
    }


def select_elastic_net_genes(train_clinical, train_expression, candidate_genes, time_col, event_col):
    best_params = tune_elastic_net(
        train_clinical=train_clinical,
        train_expression=train_expression,
        candidate_genes=candidate_genes,
        time_col=time_col,
        event_col=event_col,
    )

    result = fit_and_score_cox(
        train_clinical=train_clinical,
        test_clinical=train_clinical,
        train_expression=train_expression,
        test_expression=train_expression,
        genes=candidate_genes,
        time_col=time_col,
        event_col=event_col,
        include_clinical=False,
        penalizer=best_params["penalizer"],
        l1_ratio=best_params["l1_ratio"],
    )

    model = result["model"]

    if model is None:
        return [], best_params, "elastic_net_fit_failed"

    coefs = model.params_.copy()
    coefs = coefs[coefs.index.isin(candidate_genes)]
    coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index)

    nonzero = coefs[coefs.abs() > 1e-6]

    if nonzero.shape[0] == 0:
        selected = coefs.head(10).index.tolist()
        note = "no_nonzero_coefficients_used_top10_abs_coef"
    else:
        selected = nonzero.head(25).index.tolist()
        note = ""

    return selected, best_params, note


def score_random_sets(train_clinical, test_clinical, train_expression, test_expression, candidate_genes, time_col, event_col, rng):
    rows = []

    for repeat in range(1, RANDOM_REPEATS + 1):
        selected = list(rng.choice(candidate_genes, size=min(10, len(candidate_genes)), replace=False))

        result = fit_and_score_cox(
            train_clinical=train_clinical,
            test_clinical=test_clinical,
            train_expression=train_expression,
            test_expression=test_expression,
            genes=selected,
            time_col=time_col,
            event_col=event_col,
            include_clinical=False,
            penalizer=COX_PENALIZER,
            l1_ratio=0.0,
        )

        rows.append({
            "repeat": repeat,
            "selected_genes": selected,
            "c_index": result["c_index"],
            "error": result["error"],
        })

    valid = [r["c_index"] for r in rows if np.isfinite(r["c_index"])]
    mean_c = np.nan if len(valid) == 0 else float(np.mean(valid))

    return rows, mean_c


def run_rsf_if_available(train_clinical, test_clinical, train_expression, test_expression, candidate_genes, time_col, event_col):
    if not HAS_SKSURV:
        return {
            "c_index": np.nan,
            "error": "scikit-survival_not_installed",
            "selected_genes": [],
        }

    genes = candidate_genes[:min(500, len(candidate_genes))]

    x_train, x_test = standardize_train_test(
        train_expression[genes].copy(),
        test_expression[genes].copy(),
    )

    y_train_df = train_clinical[[time_col, event_col]].dropna()
    y_test_df = test_clinical[[time_col, event_col]].dropna()

    common_train = x_train.index.intersection(y_train_df.index)
    common_test = x_test.index.intersection(y_test_df.index)

    x_train = x_train.loc[common_train]
    x_test = x_test.loc[common_test]
    y_train_df = y_train_df.loc[common_train]
    y_test_df = y_test_df.loc[common_test]

    y_train = Surv.from_arrays(
        event=y_train_df[event_col].astype(bool).values,
        time=y_train_df[time_col].astype(float).values,
    )

    try:
        model = RandomSurvivalForest(
            n_estimators=300,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )

        model.fit(x_train, y_train)

        risk = model.predict(x_test)

        c_index = concordance_index_censored(
            y_test_df[event_col].astype(bool).values,
            y_test_df[time_col].astype(float).values,
            risk,
        )[0]

        selected_genes = []

        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=x_train.columns)
            selected_genes = importances.sort_values(ascending=False).head(25).index.tolist()

        return {
            "c_index": float(c_index),
            "error": "",
            "selected_genes": selected_genes,
        }

    except Exception as e:
        return {
            "c_index": np.nan,
            "error": str(e),
            "selected_genes": [],
        }


def record_result(rows, endpoint, fold, method, selected_genes, c_index, error, extra=None):
    extra = extra or {}

    rows.append({
        "endpoint": endpoint,
        "fold": fold,
        "method": method,
        "n_selected_genes": len(selected_genes),
        "selected_genes": ";".join(selected_genes),
        "c_index": c_index,
        "error": error,
        **extra,
    })


def main():
    print("=" * 80)
    print("Nested CV survival benchmark")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Processed directory: {PROCESSED_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print("")
    print("Leakage-control design:")
    print("  Train-only variance filtering")
    print("  Train-only univariate screening")
    print("  Train-only feature selection")
    print("  Held-out fold C-index only")
    print("")
    print(f"Outer folds: {N_OUTER_SPLITS}")
    print(f"Top train-variance genes: {TOP_N_TRAIN_VARIANCE}")
    print(f"Screened candidates per fold: {SCREEN_TOP_N}")
    print("")

    pycausal_available = add_pycausalfs_to_path()
    if pycausal_available:
        if patch_fisher_z():
            print("Patched Fisher-Z partial correlation.")
        else:
            print("Fisher-Z patch was skipped.")

    if HAS_SKSURV:
        print("scikit-survival is available. RSF baseline will run.")
    else:
        print("scikit-survival is not installed. RSF baseline will be skipped.")
        print("Optional install later: pip install scikit-survival")
    print("")

    expression_all, clinical = load_data()

    print(f"All-gene expression matrix: {expression_all.shape}")
    print(f"Clinical table: {clinical.shape}")
    print("")

    rng = np.random.default_rng(RANDOM_SEED)
    all_results = []
    selected_gene_rows = []
    screen_rows = []

    for endpoint_key, endpoint in ENDPOINTS.items():
        endpoint_label = endpoint["label"]
        time_col = endpoint["time_col"]
        event_col = endpoint["event_col"]

        clinical[time_col] = pd.to_numeric(clinical[time_col], errors="coerce")
        clinical[event_col] = pd.to_numeric(clinical[event_col], errors="coerce")

        valid_samples = clinical[[time_col, event_col]].dropna().index
        endpoint_clinical = clinical.loc[valid_samples].copy()
        endpoint_expression = expression_all.loc[valid_samples].copy()

        y_event = endpoint_clinical[event_col].astype(int).values
        samples = np.array(valid_samples)

        outer_cv = StratifiedKFold(
            n_splits=N_OUTER_SPLITS,
            shuffle=True,
            random_state=RANDOM_SEED,
        )

        print("=" * 80)
        print(f"Endpoint: {endpoint_label}")
        print("=" * 80)
        print(f"Samples: {endpoint_clinical.shape[0]}")
        print(f"Events: {int(endpoint_clinical[event_col].sum())}")
        print("")

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(samples, y_event), start=1):
            fold_start = time.time()

            train_samples = samples[train_idx]
            test_samples = samples[test_idx]

            train_clinical = endpoint_clinical.loc[train_samples].copy()
            test_clinical = endpoint_clinical.loc[test_samples].copy()

            train_expression_all = endpoint_expression.loc[train_samples].copy()
            test_expression_all = endpoint_expression.loc[test_samples].copy()

            filtered_all, variance_genes = train_variance_filter(
                train_expression=train_expression_all,
                all_expression=endpoint_expression,
            )

            train_expression = filtered_all.loc[train_samples].copy()
            test_expression = filtered_all.loc[test_samples].copy()

            print("-" * 80)
            print(f"Endpoint {endpoint_label}, fold {fold_idx}/{N_OUTER_SPLITS}")
            print("-" * 80)
            print(f"Train samples: {train_clinical.shape[0]}")
            print(f"Test samples: {test_clinical.shape[0]}")
            print(f"Train events: {int(train_clinical[event_col].sum())}")
            print(f"Test events: {int(test_clinical[event_col].sum())}")
            print(f"Train-variance genes: {len(variance_genes)}")
            print("")

            print("  Running train-only univariate screen")
            screen = run_univariate_screen(
                train_clinical=train_clinical,
                train_expression=train_expression,
                time_col=time_col,
                event_col=event_col,
            )

            screen["endpoint"] = endpoint_label
            screen["fold"] = fold_idx
            screen_rows.append(screen.head(1000))

            candidate_genes = screen["gene"].dropna().head(SCREEN_TOP_N).tolist()
            elastic_genes = screen["gene"].dropna().head(ELASTIC_NET_TOP_N).tolist()

            print(f"  Candidate genes: {len(candidate_genes)}")
            print("")

            result = fit_and_score_cox(
                train_clinical=train_clinical,
                test_clinical=test_clinical,
                train_expression=train_expression,
                test_expression=test_expression,
                genes=[],
                time_col=time_col,
                event_col=event_col,
                include_clinical=True,
            )

            record_result(
                rows=all_results,
                endpoint=endpoint_label,
                fold=fold_idx,
                method="clinical_only",
                selected_genes=[],
                c_index=result["c_index"],
                error=result["error"],
            )

            top10 = screen["gene"].dropna().head(10).tolist()
            result = fit_and_score_cox(
                train_clinical=train_clinical,
                test_clinical=test_clinical,
                train_expression=train_expression,
                test_expression=test_expression,
                genes=top10,
                time_col=time_col,
                event_col=event_col,
                include_clinical=False,
            )

            record_result(
                rows=all_results,
                endpoint=endpoint_label,
                fold=fold_idx,
                method="univariate_top10",
                selected_genes=top10,
                c_index=result["c_index"],
                error=result["error"],
            )

            print("  Running conditional Cox selection")
            conditional_genes = select_conditional_cox(
                train_clinical=train_clinical,
                train_expression=train_expression,
                candidate_genes=candidate_genes,
                time_col=time_col,
                event_col=event_col,
            )

            result = fit_and_score_cox(
                train_clinical=train_clinical,
                test_clinical=test_clinical,
                train_expression=train_expression,
                test_expression=test_expression,
                genes=conditional_genes,
                time_col=time_col,
                event_col=event_col,
                include_clinical=False,
            )

            record_result(
                rows=all_results,
                endpoint=endpoint_label,
                fold=fold_idx,
                method="conditional_cox_mb",
                selected_genes=conditional_genes,
                c_index=result["c_index"],
                error=result["error"],
            )

            result = fit_and_score_cox(
                train_clinical=train_clinical,
                test_clinical=test_clinical,
                train_expression=train_expression,
                test_expression=test_expression,
                genes=conditional_genes,
                time_col=time_col,
                event_col=event_col,
                include_clinical=True,
            )

            record_result(
                rows=all_results,
                endpoint=endpoint_label,
                fold=fold_idx,
                method="conditional_cox_mb_plus_clinical",
                selected_genes=conditional_genes,
                c_index=result["c_index"],
                error=result["error"],
            )

            if pycausal_available:
                for algorithm in ["IAMB", "GSMB"]:
                    print(f"  Running true MB method: {algorithm}")
                    mb_genes, mb_error = select_true_mb(
                        train_clinical=train_clinical,
                        train_expression=train_expression,
                        candidate_genes=candidate_genes,
                        time_col=time_col,
                        algorithm_name=algorithm,
                        alpha=0.10,
                    )

                    result = fit_and_score_cox(
                        train_clinical=train_clinical,
                        test_clinical=test_clinical,
                        train_expression=train_expression,
                        test_expression=test_expression,
                        genes=mb_genes,
                        time_col=time_col,
                        event_col=event_col,
                        include_clinical=False,
                    )

                    error = mb_error if mb_error else result["error"]

                    record_result(
                        rows=all_results,
                        endpoint=endpoint_label,
                        fold=fold_idx,
                        method=f"{algorithm.lower()}_alpha0.10",
                        selected_genes=mb_genes,
                        c_index=result["c_index"],
                        error=error,
                    )

            print("  Running Elastic Net Cox selection")
            enet_genes, enet_params, enet_note = select_elastic_net_genes(
                train_clinical=train_clinical,
                train_expression=train_expression,
                candidate_genes=elastic_genes,
                time_col=time_col,
                event_col=event_col,
            )

            result = fit_and_score_cox(
                train_clinical=train_clinical,
                test_clinical=test_clinical,
                train_expression=train_expression,
                test_expression=test_expression,
                genes=enet_genes,
                time_col=time_col,
                event_col=event_col,
                include_clinical=False,
            )

            record_result(
                rows=all_results,
                endpoint=endpoint_label,
                fold=fold_idx,
                method="elastic_net_cox",
                selected_genes=enet_genes,
                c_index=result["c_index"],
                error=result["error"] if not enet_note else enet_note,
                extra={
                    "enet_penalizer": enet_params["penalizer"],
                    "enet_l1_ratio": enet_params["l1_ratio"],
                },
            )

            print("  Running random gene-set control")
            random_rows, random_mean_c = score_random_sets(
                train_clinical=train_clinical,
                test_clinical=test_clinical,
                train_expression=train_expression,
                test_expression=test_expression,
                candidate_genes=candidate_genes,
                time_col=time_col,
                event_col=event_col,
                rng=rng,
            )

            record_result(
                rows=all_results,
                endpoint=endpoint_label,
                fold=fold_idx,
                method="random_top10_mean20",
                selected_genes=[],
                c_index=random_mean_c,
                error="",
            )

            print("  Running optional Random Survival Forest baseline")
            rsf_result = run_rsf_if_available(
                train_clinical=train_clinical,
                test_clinical=test_clinical,
                train_expression=train_expression,
                test_expression=test_expression,
                candidate_genes=candidate_genes,
                time_col=time_col,
                event_col=event_col,
            )

            record_result(
                rows=all_results,
                endpoint=endpoint_label,
                fold=fold_idx,
                method="random_survival_forest_candidate500",
                selected_genes=rsf_result["selected_genes"],
                c_index=rsf_result["c_index"],
                error=rsf_result["error"],
            )

            methods_this_fold = [
                row for row in all_results
                if row["endpoint"] == endpoint_label and row["fold"] == fold_idx
            ]

            print("")
            print("  Fold result summary:")
            for row in methods_this_fold:
                print(
                    f"    {row['method']:35s} "
                    f"C-index={row['c_index']} "
                    f"n_genes={row['n_selected_genes']} "
                    f"error={row['error']}"
                )

            print(f"  Fold runtime: {(time.time() - fold_start) / 60:.1f} min")
            print("")

            for method_row in methods_this_fold:
                genes = method_row["selected_genes"].split(";") if method_row["selected_genes"] else []

                for rank, gene in enumerate(genes, start=1):
                    selected_gene_rows.append({
                        "endpoint": endpoint_label,
                        "fold": fold_idx,
                        "method": method_row["method"],
                        "gene_rank": rank,
                        "gene": gene,
                        "gene_symbol_clean": clean_gene_symbol(gene),
                    })

    results = pd.DataFrame(all_results)
    selected_genes = pd.DataFrame(selected_gene_rows)
    screens = pd.concat(screen_rows, axis=0, ignore_index=True)

    summary = (
        results
        .groupby(["endpoint", "method"], dropna=False)
        .agg(
            n_folds=("c_index", "count"),
            mean_c_index=("c_index", "mean"),
            std_c_index=("c_index", "std"),
            median_c_index=("c_index", "median"),
            min_c_index=("c_index", "min"),
            max_c_index=("c_index", "max"),
            mean_n_selected_genes=("n_selected_genes", "mean"),
        )
        .reset_index()
        .sort_values(["endpoint", "mean_c_index"], ascending=[True, False])
    )

    results_path = RESULTS_DIR / "GSE238110_nested_cv_method_benchmark.csv"
    summary_path = RESULTS_DIR / "GSE238110_nested_cv_method_benchmark_summary.csv"
    selected_genes_path = RESULTS_DIR / "GSE238110_nested_cv_selected_genes.csv"
    screen_path = RESULTS_DIR / "GSE238110_nested_cv_train_univariate_screen_top1000_per_fold.csv"

    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    selected_genes.to_csv(selected_genes_path, index=False)
    screens.to_csv(screen_path, index=False)

    print("=" * 80)
    print("Nested CV benchmark summary")
    print("=" * 80)
    print(summary.to_string(index=False))
    print("")

    print("Saved:")
    print(results_path)
    print(summary_path)
    print(selected_genes_path)
    print(screen_path)
    print("Done.")


if __name__ == "__main__":
    main()
