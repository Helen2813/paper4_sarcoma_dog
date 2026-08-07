from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

SCRIPT_VERSION = "45-dog2-pathological-necrosis-response-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "pathological_response"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

EXPRESSION_FILE = (
    PROCESSED_DIR / "GSE238110_DOG2_expression_log2cpm_matched_allgenes.csv"
)
CLINICAL_FILE = (
    PROCESSED_DIR / "GSE238110_DOG2_clinical_matched_indexed.csv"
)
STRICT_WEIGHTS_FILE = (
    RESULTS_DIR / "GSE238110_frozen_transfer_gene_weights_strict.csv"
)
FREEZE_FILE = (
    RESULTS_DIR / "GSE238110_frozen_transfer_program_freeze.json"
)
PROLIFERATION_SCORE_FILE = (
    RESULTS_DIR / "GSE238110_meta_proliferation_score_per_sample.csv"
)
PROLIFERATION_GENE_FILE = (
    RESULTS_DIR / "GSE238110_meta_proliferation_gene_set.csv"
)

PATKAR_FILENAME = "ccr-24-1854_supplementary_table_s10_suppts10.xlsx"
PATKAR_DOWNLOAD_URL = (
    "https://aacr.figshare.com/ndownloader/files/51205537"
)

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]
PRIMARY_TARGET_MODULE = "M40"
GOOD_RESPONSE_THRESHOLD = 90.0

N_PERMUTATIONS = 10000
N_BOOTSTRAP = 2000
N_RANDOM_PANELS = 1000
N_VARIABILITY_BINS = 10
RANDOM_SEED = 42

MIN_CONTINUOUS_N = 25
MIN_BINARY_CLASS = 8
MIN_DISJOINT_PROLIFERATION_GENES = 20

OUTPUT_COLUMN_AUDIT = (
    RESULTS_DIR / "DOG2_pathological_response_column_audit.csv"
)
OUTPUT_MATCHED = (
    RESULTS_DIR / "DOG2_pathological_response_frozen_program_scores.csv"
)
OUTPUT_COVERAGE = (
    RESULTS_DIR / "DOG2_pathological_response_score_coverage.csv"
)
OUTPUT_PRIMARY = (
    RESULTS_DIR / "DOG2_pathological_response_primary_module_tests.csv"
)
OUTPUT_SENSITIVITY = (
    RESULTS_DIR / "DOG2_pathological_response_score_variant_sensitivity.csv"
)
OUTPUT_LOO = (
    RESULTS_DIR / "DOG2_pathological_response_M40_leave_one_out.csv"
)
OUTPUT_RANDOM = (
    RESULTS_DIR / "DOG2_pathological_response_M40_random_panel_summary.csv"
)
OUTPUT_RANDOM_DISTRIBUTION = (
    RESULTS_DIR / "DOG2_pathological_response_M40_random_panel_distribution.csv"
)
OUTPUT_RESPONSE_SUMMARY = (
    RESULTS_DIR / "DOG2_pathological_response_endpoint_summary.csv"
)
OUTPUT_README = (
    RESULTS_DIR / "DOG2_pathological_response_README.txt"
)
OUTPUT_MANIFEST = (
    RESULTS_DIR / "DOG2_pathological_response_manifest.json"
)

SCATTER_PNG = FIGURES_DIR / "DOG2_M40_percent_necrosis_scatter.png"
SCATTER_PDF = FIGURES_DIR / "DOG2_M40_percent_necrosis_scatter.pdf"
BINARY_PNG = FIGURES_DIR / "DOG2_M40_good_response_boxplot.png"
BINARY_PDF = FIGURES_DIR / "DOG2_M40_good_response_boxplot.pdf"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_required_csv(
    path: Path,
    index_col: int | str | None = None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    print(f"Loaded: {path}")
    return pd.read_csv(path, index_col=index_col)


def read_optional_csv(
    path: Path,
    index_col: int | str | None = None,
) -> pd.DataFrame:
    if not path.exists():
        print(f"Optional file not found: {path}")
        return pd.DataFrame()
    print(f"Loaded: {path}")
    return pd.read_csv(path, index_col=index_col)


def bh_adjust(values: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)

    if valid.sum() == 0:
        return q

    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    restored = np.empty(n, dtype=float)
    restored[order] = adjusted
    q[valid] = restored
    return q


def zscore_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    sd = numeric.std()
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (numeric - numeric.mean()) / sd


def zscore_columns(expression: pd.DataFrame) -> pd.DataFrame:
    x = expression.apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(axis=0))
    std = x.std(axis=0).replace(0, np.nan)
    z = (x - x.mean(axis=0)) / std
    return z.loc[:, z.notna().all(axis=0)]


def normalize_patient_id(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na"}:
        return ""

    try:
        numeric = float(text)
        if np.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    except ValueError:
        pass

    groups = re.findall(r"\d+", text)
    if not groups:
        return ""
    return str(int(groups[-1]))


def parse_percent(value: Any) -> float:
    if pd.isna(value):
        return np.nan

    text = str(value).strip().replace(",", ".")
    if not text or text.lower() in {
        "na",
        "nan",
        "none",
        "not available",
        "unknown",
        "not done",
    }:
        return np.nan

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not numbers:
        return np.nan

    parsed = [float(number) for number in numbers]
    if len(parsed) >= 2 and any(
        separator in text for separator in ["-", "–", "to"]
    ):
        value_numeric = float(np.mean(parsed[:2]))
    else:
        value_numeric = float(parsed[0])

    if value_numeric <= 1.0 and "%" not in text:
        value_numeric *= 100.0

    if value_numeric < 0 or value_numeric > 100:
        return np.nan
    return value_numeric


def normalized_column_name(column: Any) -> str:
    text = str(column).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def response_column_candidates(
    table: pd.DataFrame,
    source: str,
    sheet: str = "",
) -> pd.DataFrame:
    rows = []

    for column in table.columns:
        normalized = normalized_column_name(column)
        if not any(
            token in normalized
            for token in [
                "necrosis",
                "histologic_response",
                "histological_response",
                "pathologic_response",
                "pathological_response",
            ]
        ):
            continue

        parsed = table[column].map(parse_percent)
        valid = parsed.dropna()

        score = 0.0
        if "necrosis" in normalized:
            score += 10.0
        if "percent" in normalized or "pct" in normalized:
            score += 5.0
        if "tumor" in normalized:
            score += 2.0
        if "response" in normalized:
            score += 1.0
        score += min(5.0, valid.shape[0] / 20.0)

        rows.append(
            {
                "source": source,
                "sheet": sheet,
                "column": str(column),
                "normalized_column": normalized,
                "n_rows": table.shape[0],
                "n_numeric_valid": int(valid.shape[0]),
                "n_unique_valid": int(valid.nunique()),
                "minimum": float(valid.min()) if not valid.empty else np.nan,
                "median": float(valid.median()) if not valid.empty else np.nan,
                "maximum": float(valid.max()) if not valid.empty else np.nan,
                "candidate_score": score,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "source",
                "sheet",
                "column",
                "normalized_column",
                "n_rows",
                "n_numeric_valid",
                "n_unique_valid",
                "minimum",
                "median",
                "maximum",
                "candidate_score",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        ["candidate_score", "n_numeric_valid"],
        ascending=[False, False],
    )


def locate_or_download_patkar_table() -> Path:
    candidates = [
        RAW_DIR / "canine_clinical_DOG2" / PATKAR_FILENAME,
        RAW_DIR / "DOG2" / PATKAR_FILENAME,
        RAW_DIR / PATKAR_FILENAME,
        PROJECT_ROOT / PATKAR_FILENAME,
    ]

    for path in candidates:
        if path.exists():
            print(f"Patkar Table S10 found: {path}")
            return path

    recursive = list(PROJECT_ROOT.rglob(PATKAR_FILENAME))
    if recursive:
        print(f"Patkar Table S10 found: {recursive[0]}")
        return recursive[0]

    destination = (
        RAW_DIR / "canine_clinical_DOG2" / PATKAR_FILENAME
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(
        "Patkar Table S10 not found locally; "
        f"downloading to: {destination}"
    )
    urllib.request.urlretrieve(PATKAR_DOWNLOAD_URL, destination)
    return destination


def detect_patient_id_column(table: pd.DataFrame) -> str | None:
    exact = [
        "patient_id",
        "patient id",
        "patientid",
        "dog_id",
        "dog id",
        "sample_id",
        "sample id",
    ]
    normalized_map = {
        normalized_column_name(column): str(column)
        for column in table.columns
    }

    for candidate in exact:
        normalized = normalized_column_name(candidate)
        if normalized in normalized_map:
            return normalized_map[normalized]

    best_column = None
    best_nonempty = 0
    for column in table.columns:
        normalized = normalized_column_name(column)
        if not any(
            token in normalized
            for token in ["patient", "dog", "sample"]
        ):
            continue
        parsed = table[column].map(normalize_patient_id)
        nonempty = int(parsed.ne("").sum())
        if nonempty > best_nonempty:
            best_nonempty = nonempty
            best_column = str(column)

    return best_column


def load_response_endpoint(
    clinical: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any], Path | None]:
    clinical_audit = response_column_candidates(
        clinical,
        source="processed_clinical",
    )

    if not clinical_audit.empty:
        top = clinical_audit.iloc[0]
        if int(top["n_numeric_valid"]) >= MIN_CONTINUOUS_N:
            column = str(top["column"])
            response = clinical[column].map(parse_percent)
            response.index = clinical.index
            return (
                response,
                clinical_audit,
                {
                    "response_source": "processed_clinical",
                    "response_sheet": "",
                    "response_column": column,
                    "patient_id_column": "__index__",
                },
                None,
            )

    patkar_path = locate_or_download_patkar_table()
    workbook = pd.ExcelFile(patkar_path)
    audits = []
    sheet_tables: dict[str, pd.DataFrame] = {}

    for sheet_name in workbook.sheet_names:
        table = pd.read_excel(patkar_path, sheet_name=sheet_name)
        sheet_tables[sheet_name] = table
        audit = response_column_candidates(
            table,
            source="Patkar_Table_S10",
            sheet=sheet_name,
        )
        if not audit.empty:
            audits.append(audit)

    if not audits:
        raise RuntimeError(
            "No pathological-necrosis or response column was found "
            "in the processed clinical table or Patkar Table S10."
        )

    raw_audit = pd.concat(audits, ignore_index=True).sort_values(
        ["candidate_score", "n_numeric_valid"],
        ascending=[False, False],
    )
    combined_audit = pd.concat(
        [clinical_audit, raw_audit],
        ignore_index=True,
        sort=False,
    )

    target_ids = {
        normalize_patient_id(value)
        for value in clinical.index
    }
    target_ids.discard("")

    selected = None
    for row in raw_audit.itertuples(index=False):
        table = sheet_tables[str(row.sheet)]
        patient_column = detect_patient_id_column(table)
        if patient_column is None:
            continue

        patient_ids = table[patient_column].map(normalize_patient_id)
        overlap = int(patient_ids.isin(target_ids).sum())
        if overlap >= min(20, int(row.n_numeric_valid)):
            selected = {
                "sheet": str(row.sheet),
                "column": str(row.column),
                "patient_column": patient_column,
                "overlap": overlap,
            }
            break

    if selected is None:
        raise RuntimeError(
            "A response column was found in Patkar Table S10, "
            "but no patient-ID column could be matched to DOG2."
        )

    table = sheet_tables[selected["sheet"]].copy()
    table["patient_id_normalized"] = table[
        selected["patient_column"]
    ].map(normalize_patient_id)
    table["percent_necrosis"] = table[
        selected["column"]
    ].map(parse_percent)
    table = table[
        table["patient_id_normalized"].ne("")
    ].drop_duplicates("patient_id_normalized", keep="first")

    clinical_ids = pd.Series(
        clinical.index,
        index=clinical.index,
    ).map(normalize_patient_id)
    response_map = table.set_index(
        "patient_id_normalized"
    )["percent_necrosis"]
    response = clinical_ids.map(response_map)
    response.index = clinical.index

    return (
        response,
        combined_audit,
        {
            "response_source": "Patkar_Table_S10",
            "response_sheet": selected["sheet"],
            "response_column": selected["column"],
            "patient_id_column": selected["patient_column"],
            "n_patient_id_overlap": selected["overlap"],
        },
        patkar_path,
    )


def compute_frozen_scores(
    expression: pd.DataFrame,
    weights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.DataFrame(index=expression.index)
    coverage_rows = []

    expression_z = zscore_columns(expression)

    for module in PRIMARY_MODULES:
        part = weights[
            weights["module_label"].astype(str).eq(module)
        ].copy()
        part["canine_gene"] = part["canine_gene"].astype(str)
        part = part.drop_duplicates("canine_gene", keep="first")

        requested = part["canine_gene"].tolist()
        available = [
            gene for gene in requested if gene in expression_z.columns
        ]

        coverage_rows.append(
            {
                "module_label": module,
                "n_frozen_genes": len(requested),
                "n_available_genes": len(available),
                "coverage_fraction": (
                    len(available) / len(requested)
                    if requested
                    else np.nan
                ),
                "available_genes": ";".join(available),
                "missing_genes": ";".join(
                    gene for gene in requested if gene not in available
                ),
            }
        )

        if len(available) < 3:
            continue

        module_expression = expression_z[available]
        indexed = part.set_index("canine_gene").loc[available]
        loadings = pd.to_numeric(
            indexed["risk_oriented_loading"],
            errors="coerce",
        ).fillna(0.0)

        signs = np.sign(loadings).replace(0, 1)
        signed_mean = module_expression.mul(
            signs,
            axis=1,
        ).mean(axis=1)

        if loadings.abs().sum() > 0:
            normalized_weights = loadings / loadings.abs().sum()
            weighted = module_expression.mul(
                normalized_weights,
                axis=1,
            ).sum(axis=1)
        else:
            weighted = pd.Series(
                np.nan,
                index=module_expression.index,
            )

        scores[f"{module}__strict_signed_mean_z"] = zscore_series(
            signed_mean
        )
        scores[
            f"{module}__strict_canine_weighted_z"
        ] = zscore_series(weighted)

    return scores, pd.DataFrame(coverage_rows)


def detect_gene_column(table: pd.DataFrame) -> str:
    candidates = [
        "gene",
        "canine_gene",
        "gene_symbol",
        "symbol",
        "expression_column",
    ]

    for candidate in candidates:
        if candidate in table.columns:
            return candidate

    best_column = None
    best_overlap = -1
    for column in table.columns:
        nonempty = table[column].astype(str).str.strip().ne("").sum()
        if nonempty > best_overlap:
            best_overlap = int(nonempty)
            best_column = str(column)

    if best_column is None:
        raise ValueError("Could not detect proliferation gene column.")
    return best_column


def build_disjoint_proliferation_score(
    expression: pd.DataFrame,
    weights: pd.DataFrame,
    proliferation_genes: pd.DataFrame,
    original_score: pd.DataFrame,
) -> tuple[pd.Series, dict[str, Any]]:
    if proliferation_genes.empty:
        return (
            pd.Series(np.nan, index=expression.index, dtype=float),
            {"estimable": False, "reason": "gene_file_missing"},
        )

    gene_column = detect_gene_column(proliferation_genes)
    candidate_genes = (
        proliferation_genes[gene_column]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    m40_genes = set(
        weights.loc[
            weights["module_label"].astype(str).eq("M40"),
            "canine_gene",
        ].astype(str)
    )
    disjoint = [
        gene
        for gene in candidate_genes
        if gene in expression.columns and gene not in m40_genes
    ]

    if len(disjoint) < MIN_DISJOINT_PROLIFERATION_GENES:
        return (
            pd.Series(np.nan, index=expression.index, dtype=float),
            {
                "estimable": False,
                "reason": "too_few_disjoint_genes",
                "n_disjoint_genes": len(disjoint),
            },
        )

    x = zscore_columns(expression[disjoint])
    pca = PCA(n_components=1, random_state=RANDOM_SEED)
    values = pca.fit_transform(x).ravel()
    score = pd.Series(values, index=x.index, dtype=float)

    original = pd.Series(np.nan, index=expression.index, dtype=float)
    if not original_score.empty:
        if "meta_proliferation_score" in original_score.columns:
            original = original_score[
                "meta_proliferation_score"
            ].reindex(expression.index)
        elif original_score.shape[1] == 1:
            original = original_score.iloc[:, 0].reindex(
                expression.index
            )

    correlation = score.corr(original)
    if np.isfinite(correlation) and correlation < 0:
        score = -score
        correlation = -correlation

    score = zscore_series(score)

    return (
        score,
        {
            "estimable": True,
            "n_source_proliferation_genes": len(candidate_genes),
            "n_m40_overlap_removed": len(
                set(candidate_genes).intersection(m40_genes)
            ),
            "n_disjoint_genes": len(disjoint),
            "pc1_explained_variance": float(
                pca.explained_variance_ratio_[0]
            ),
            "correlation_with_original_meta_score": (
                float(correlation)
                if np.isfinite(correlation)
                else np.nan
            ),
        },
    )


def residualize_score(
    score: pd.Series,
    covariate: pd.Series,
) -> pd.Series:
    frame = pd.DataFrame(
        {
            "score": pd.to_numeric(score, errors="coerce"),
            "covariate": pd.to_numeric(
                covariate,
                errors="coerce",
            ),
        }
    ).dropna()

    result = pd.Series(np.nan, index=score.index, dtype=float)
    if frame.shape[0] < 10 or frame["covariate"].std() == 0:
        return result

    model = LinearRegression()
    model.fit(
        frame[["covariate"]].to_numpy(),
        frame["score"].to_numpy(),
    )
    residuals = (
        frame["score"].to_numpy()
        - model.predict(frame[["covariate"]].to_numpy())
    )
    result.loc[frame.index] = residuals
    return zscore_series(result)


def permutation_spearman_p(
    x: np.ndarray,
    y: np.ndarray,
    observed: float,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    count = 0

    for _ in range(N_PERMUTATIONS):
        permuted = rng.permutation(y)
        value = stats.spearmanr(x, permuted).statistic
        if abs(value) >= abs(observed):
            count += 1

    return float((count + 1) / (N_PERMUTATIONS + 1))


def permutation_auc_p(
    x: np.ndarray,
    y: np.ndarray,
    observed_auc: float,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    observed_distance = abs(observed_auc - 0.5)
    count = 0

    for _ in range(N_PERMUTATIONS):
        permuted = rng.permutation(y)
        auc = roc_auc_score(permuted, x)
        if abs(auc - 0.5) >= observed_distance:
            count += 1

    return float((count + 1) / (N_PERMUTATIONS + 1))


def bootstrap_continuous(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    rhos = []
    slopes = []

    for _ in range(N_BOOTSTRAP):
        indices = rng.integers(0, n, size=n)
        xb = x[indices]
        yb = y[indices]

        if np.std(xb) == 0 or np.std(yb) == 0:
            continue

        rho = stats.spearmanr(xb, yb).statistic
        slope = LinearRegression().fit(
            xb.reshape(-1, 1),
            yb,
        ).coef_[0]

        if np.isfinite(rho):
            rhos.append(float(rho))
        if np.isfinite(slope):
            slopes.append(float(slope))

    return {
        "rho_ci_low": (
            float(np.quantile(rhos, 0.025)) if rhos else np.nan
        ),
        "rho_ci_high": (
            float(np.quantile(rhos, 0.975)) if rhos else np.nan
        ),
        "slope_ci_low": (
            float(np.quantile(slopes, 0.025))
            if slopes
            else np.nan
        ),
        "slope_ci_high": (
            float(np.quantile(slopes, 0.975))
            if slopes
            else np.nan
        ),
    }


def logistic_or(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(x.reshape(-1, 1), y)
    return float(np.exp(model.coef_[0, 0]))


def bootstrap_binary(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    positive = np.where(y == 1)[0]
    negative = np.where(y == 0)[0]

    aucs = []
    odds_ratios = []

    for _ in range(N_BOOTSTRAP):
        indices = np.concatenate(
            [
                rng.choice(
                    positive,
                    size=len(positive),
                    replace=True,
                ),
                rng.choice(
                    negative,
                    size=len(negative),
                    replace=True,
                ),
            ]
        )
        rng.shuffle(indices)

        xb = x[indices]
        yb = y[indices]

        if np.std(xb) == 0:
            continue

        try:
            aucs.append(float(roc_auc_score(yb, xb)))
            odds_ratios.append(logistic_or(xb, yb))
        except Exception:
            continue

    return {
        "auc_ci_low": (
            float(np.quantile(aucs, 0.025)) if aucs else np.nan
        ),
        "auc_ci_high": (
            float(np.quantile(aucs, 0.975)) if aucs else np.nan
        ),
        "or_ci_low": (
            float(np.quantile(odds_ratios, 0.025))
            if odds_ratios
            else np.nan
        ),
        "or_ci_high": (
            float(np.quantile(odds_ratios, 0.975))
            if odds_ratios
            else np.nan
        ),
    }


def analyze_score(
    score: pd.Series,
    percent_necrosis: pd.Series,
    module_label: str,
    score_variant: str,
    seed: int,
) -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            "score": pd.to_numeric(score, errors="coerce"),
            "percent_necrosis": pd.to_numeric(
                percent_necrosis,
                errors="coerce",
            ),
        }
    ).dropna()

    result: dict[str, Any] = {
        "module_label": module_label,
        "score_variant": score_variant,
        "n_continuous": int(frame.shape[0]),
        "n_good_response": np.nan,
        "n_poor_response": np.nan,
        "spearman_rho": np.nan,
        "spearman_asymptotic_p": np.nan,
        "spearman_permutation_p": np.nan,
        "rho_ci_low": np.nan,
        "rho_ci_high": np.nan,
        "slope_percent_necrosis_per_score_sd": np.nan,
        "slope_ci_low": np.nan,
        "slope_ci_high": np.nan,
        "auc_higher_score_predicts_good_response": np.nan,
        "auc_ci_low": np.nan,
        "auc_ci_high": np.nan,
        "logistic_or_good_response_per_score_sd": np.nan,
        "or_ci_low": np.nan,
        "or_ci_high": np.nan,
        "auc_permutation_p": np.nan,
        "error": "",
    }

    if frame.shape[0] < MIN_CONTINUOUS_N:
        result["error"] = "too_few_continuous_observations"
        return result

    x = zscore_series(frame["score"]).to_numpy(dtype=float)
    y = frame["percent_necrosis"].to_numpy(dtype=float)

    spearman = stats.spearmanr(x, y)
    slope = LinearRegression().fit(
        x.reshape(-1, 1),
        y,
    ).coef_[0]
    continuous_bootstrap = bootstrap_continuous(
        x,
        y,
        seed,
    )

    result.update(
        {
            "spearman_rho": float(spearman.statistic),
            "spearman_asymptotic_p": float(spearman.pvalue),
            "spearman_permutation_p": permutation_spearman_p(
                x,
                y,
                float(spearman.statistic),
                seed + 1,
            ),
            "slope_percent_necrosis_per_score_sd": float(slope),
            **continuous_bootstrap,
        }
    )

    binary = (
        frame["percent_necrosis"] >= GOOD_RESPONSE_THRESHOLD
    ).astype(int)
    n_good = int(binary.sum())
    n_poor = int((1 - binary).sum())

    result["n_good_response"] = n_good
    result["n_poor_response"] = n_poor

    if min(n_good, n_poor) < MIN_BINARY_CLASS:
        result["error"] = (
            result["error"] + ";"
            if result["error"]
            else ""
        ) + "too_few_binary_class_observations"
        return result

    y_binary = binary.to_numpy(dtype=int)
    auc = roc_auc_score(y_binary, x)
    odds_ratio = logistic_or(x, y_binary)
    binary_bootstrap = bootstrap_binary(
        x,
        y_binary,
        seed + 2,
    )

    result.update(
        {
            "auc_higher_score_predicts_good_response": float(auc),
            "logistic_or_good_response_per_score_sd": odds_ratio,
            "auc_permutation_p": permutation_auc_p(
                x,
                y_binary,
                auc,
                seed + 3,
            ),
            **binary_bootstrap,
        }
    )
    return result


def leave_one_out_m40(
    score: pd.Series,
    percent_necrosis: pd.Series,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "score": pd.to_numeric(score, errors="coerce"),
            "percent_necrosis": pd.to_numeric(
                percent_necrosis,
                errors="coerce",
            ),
        }
    ).dropna()

    rows = []
    for left_out in frame.index:
        part = frame.drop(index=left_out)
        x = zscore_series(part["score"]).to_numpy(dtype=float)
        y = part["percent_necrosis"].to_numpy(dtype=float)
        binary = (
            part["percent_necrosis"] >= GOOD_RESPONSE_THRESHOLD
        ).astype(int)

        rho = (
            float(stats.spearmanr(x, y).statistic)
            if len(part) >= MIN_CONTINUOUS_N
            else np.nan
        )
        auc = (
            float(roc_auc_score(binary, x))
            if binary.nunique() == 2
            and min(binary.sum(), (1 - binary).sum())
            >= MIN_BINARY_CLASS - 1
            else np.nan
        )

        rows.append(
            {
                "left_out_sample": str(left_out),
                "n": part.shape[0],
                "spearman_rho": rho,
                "auc_higher_score_predicts_good_response": auc,
            }
        )

    return pd.DataFrame(rows)


def variability_bins(expression: pd.DataFrame) -> pd.Series:
    variance = expression.var(axis=0)
    ranks = variance.rank(method="average")
    return pd.qcut(
        ranks,
        q=min(N_VARIABILITY_BINS, variance.shape[0]),
        labels=False,
        duplicates="drop",
    )


def random_panel_controls(
    expression: pd.DataFrame,
    weights: pd.DataFrame,
    percent_necrosis: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expression_z = zscore_columns(expression)
    bins = variability_bins(expression_z)

    m40 = weights[
        weights["module_label"].astype(str).eq("M40")
    ].copy()
    m40["canine_gene"] = m40["canine_gene"].astype(str)
    m40 = m40.drop_duplicates("canine_gene", keep="first")
    m40 = m40[m40["canine_gene"].isin(expression_z.columns)]

    all_primary_genes = set(
        weights.loc[
            weights["module_label"].isin(PRIMARY_MODULES),
            "canine_gene",
        ].astype(str)
    )

    candidate_genes = [
        gene
        for gene in expression_z.columns
        if gene not in all_primary_genes
    ]

    target_genes = m40["canine_gene"].tolist()
    target_bins = bins.reindex(target_genes).to_numpy()
    loadings = pd.to_numeric(
        m40.set_index("canine_gene").loc[
            target_genes,
            "risk_oriented_loading",
        ],
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)

    response_frame = pd.DataFrame(
        {
            "percent_necrosis": percent_necrosis,
        }
    ).dropna()
    response_index = response_frame.index.intersection(
        expression_z.index
    )
    y = percent_necrosis.loc[response_index].to_numpy(dtype=float)
    y_binary = (y >= GOOD_RESPONSE_THRESHOLD).astype(int)

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    candidates_by_bin = {
        bin_value: [
            gene
            for gene in candidate_genes
            if bins.get(gene, np.nan) == bin_value
        ]
        for bin_value in pd.Series(target_bins).dropna().unique()
    }

    for repeat in range(N_RANDOM_PANELS):
        selected: list[str] = []
        used: set[str] = set()

        for target_bin in target_bins:
            pool = [
                gene
                for gene in candidates_by_bin.get(target_bin, [])
                if gene not in used
            ]
            if not pool:
                pool = [
                    gene
                    for gene in candidate_genes
                    if gene not in used
                ]
            if not pool:
                selected = []
                break

            gene = str(rng.choice(pool))
            selected.append(gene)
            used.add(gene)

        if len(selected) != len(target_genes):
            continue

        permuted_loadings = rng.permutation(loadings)
        denominator = np.sum(np.abs(permuted_loadings))
        if denominator <= 0:
            continue

        score = (
            expression_z.loc[
                response_index,
                selected,
            ].to_numpy(dtype=float)
            @ permuted_loadings
            / denominator
        )
        score = (
            score - np.mean(score)
        ) / np.std(score)

        rho = stats.spearmanr(score, y).statistic
        auc = (
            roc_auc_score(y_binary, score)
            if len(np.unique(y_binary)) == 2
            else np.nan
        )

        rows.append(
            {
                "random_panel": repeat + 1,
                "n_genes": len(selected),
                "spearman_rho": float(rho),
                "absolute_spearman_rho": float(abs(rho)),
                "auc_higher_score_predicts_good_response": (
                    float(auc)
                    if np.isfinite(auc)
                    else np.nan
                ),
                "absolute_auc_distance_from_0_5": (
                    float(abs(auc - 0.5))
                    if np.isfinite(auc)
                    else np.nan
                ),
            }
        )

    distribution = pd.DataFrame(rows)

    m40_part = m40.set_index("canine_gene")
    m40_loadings = pd.to_numeric(
        m40_part.loc[
            target_genes,
            "risk_oriented_loading",
        ],
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)

    m40_score = (
        expression_z.loc[
            response_index,
            target_genes,
        ].to_numpy(dtype=float)
        @ m40_loadings
        / np.sum(np.abs(m40_loadings))
    )
    m40_score = (
        m40_score - np.mean(m40_score)
    ) / np.std(m40_score)

    observed_rho = float(
        stats.spearmanr(m40_score, y).statistic
    )
    observed_auc = float(
        roc_auc_score(y_binary, m40_score)
    )

    summary = pd.DataFrame(
        [
            {
                "module_label": "M40",
                "n_response_samples": len(y),
                "n_random_panels": distribution.shape[0],
                "observed_spearman_rho": observed_rho,
                "observed_absolute_spearman_rho": abs(
                    observed_rho
                ),
                "random_absolute_rho_median": float(
                    distribution[
                        "absolute_spearman_rho"
                    ].median()
                ),
                "random_absolute_rho_q95": float(
                    distribution[
                        "absolute_spearman_rho"
                    ].quantile(0.95)
                ),
                "rho_empirical_p": float(
                    (
                        1
                        + (
                            distribution[
                                "absolute_spearman_rho"
                            ]
                            >= abs(observed_rho)
                        ).sum()
                    )
                    / (distribution.shape[0] + 1)
                ),
                "observed_auc": observed_auc,
                "observed_absolute_auc_distance_from_0_5": abs(
                    observed_auc - 0.5
                ),
                "random_auc_distance_median": float(
                    distribution[
                        "absolute_auc_distance_from_0_5"
                    ].median()
                ),
                "random_auc_distance_q95": float(
                    distribution[
                        "absolute_auc_distance_from_0_5"
                    ].quantile(0.95)
                ),
                "auc_empirical_p": float(
                    (
                        1
                        + (
                            distribution[
                                "absolute_auc_distance_from_0_5"
                            ]
                            >= abs(observed_auc - 0.5)
                        ).sum()
                    )
                    / (distribution.shape[0] + 1)
                ),
            }
        ]
    )
    return summary, distribution


def create_figures(
    frame: pd.DataFrame,
    score_column: str,
) -> None:
    part = frame[
        [score_column, "percent_necrosis"]
    ].dropna()

    if part.shape[0] >= MIN_CONTINUOUS_N:
        x = zscore_series(part[score_column])
        y = part["percent_necrosis"]
        model = LinearRegression().fit(
            x.to_numpy().reshape(-1, 1),
            y.to_numpy(),
        )
        x_grid = np.linspace(
            float(x.min()),
            float(x.max()),
            100,
        )
        y_grid = model.predict(x_grid.reshape(-1, 1))

        fig, ax = plt.subplots(figsize=(6.0, 4.8))
        ax.scatter(x, y, alpha=0.7)
        ax.plot(x_grid, y_grid)
        ax.axhline(
            GOOD_RESPONSE_THRESHOLD,
            linestyle="--",
            linewidth=1,
        )
        ax.set_xlabel("Frozen M40 risk-oriented score (SD)")
        ax.set_ylabel("Pathological tumor necrosis (%)")
        ax.set_title("DOG2 M40 and pathological necrosis")
        fig.tight_layout()
        fig.savefig(
            SCATTER_PNG,
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            SCATTER_PDF,
            bbox_inches="tight",
        )
        plt.close(fig)

        binary = (
            part["percent_necrosis"] >= GOOD_RESPONSE_THRESHOLD
        )
        groups = [
            x[~binary].to_numpy(),
            x[binary].to_numpy(),
        ]

        fig, ax = plt.subplots(figsize=(5.4, 4.7))
        ax.boxplot(
            groups,
            tick_labels=[
                f"<{GOOD_RESPONSE_THRESHOLD:.0f}%",
                f"≥{GOOD_RESPONSE_THRESHOLD:.0f}%",
            ],
            showfliers=False,
        )
        rng = np.random.default_rng(RANDOM_SEED)
        for position, values in enumerate(groups, start=1):
            jitter = rng.normal(0.0, 0.04, size=len(values))
            ax.scatter(
                np.full(len(values), position) + jitter,
                values,
                alpha=0.65,
                s=20,
            )
        ax.set_ylabel("Frozen M40 risk-oriented score (SD)")
        ax.set_title("DOG2 M40 by pathological response")
        fig.tight_layout()
        fig.savefig(
            BINARY_PNG,
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            BINARY_PDF,
            bbox_inches="tight",
        )
        plt.close(fig)


def write_readme(
    response_metadata: dict[str, Any],
    proliferation_metadata: dict[str, Any],
) -> None:
    text = f"""DOG2 frozen-program association with pathological necrosis
Script version: {SCRIPT_VERSION}

Purpose
-------
Test whether the frozen canine transcriptional programs are associated with
subsequent pathological tumor necrosis after chemotherapy.

Response source
---------------
{json.dumps(response_metadata, indent=2)}

Primary exploratory hypothesis
------------------------------
M40 is tested first because prior analyses and single-cell localization identify
it as a cycling/proliferation axis.

Primary response representations
--------------------------------
1. Continuous percent tumor necrosis.
2. Good pathological response defined as at least
   {GOOD_RESPONSE_THRESHOLD:.0f}% necrosis.

Primary molecular score
-----------------------
Frozen strict one-to-one canine signed-mean score with the previously fixed
risk orientation.

Multiplicity
------------
- BH across four modules for continuous response.
- BH across four modules for binary response.
- Global BH across all eight module-by-response tests.
- M40 is the prespecified primary exploratory module.
- Frozen weighted scores, meta-proliferation, and residual scores are
  sensitivity analyses.

Disjoint proliferation sensitivity
----------------------------------
{json.dumps(proliferation_metadata, indent=2)}

Random controls
---------------
M40 is compared with variability-matched random panels of the same gene count.
All genes from M34, M11, M24, and M40 are excluded from the random candidate
pool, and the frozen M40 loading values are permuted across selected genes.

Interpretation guardrails
-------------------------
- Percent necrosis is a post-treatment response endpoint.
- Association does not establish chemotherapy benefit because there is no
  untreated comparator.
- This is a same-cohort exploratory translational analysis, not independent
  external validation.
- No response result can modify frozen module genes, loadings, direction,
  human evidence tiers, or the project-wide multiplicity results.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("DOG2 frozen programs and pathological necrosis response")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Design:")
    print("  Detect percent-necrosis data without outcome-guided score changes.")
    print("  Reconstruct frozen strict canine program scores.")
    print("  Test continuous necrosis and the prespecified ≥90% response threshold.")
    print("  Treat M40 as the primary exploratory response hypothesis.")
    print("  Compare M40 with proliferation and matched random panels.")
    print("")

    expression = read_required_csv(
        EXPRESSION_FILE,
        index_col=0,
    )
    clinical = read_required_csv(
        CLINICAL_FILE,
        index_col=0,
    )
    weights = read_required_csv(STRICT_WEIGHTS_FILE)
    original_proliferation = read_optional_csv(
        PROLIFERATION_SCORE_FILE,
        index_col=0,
    )
    proliferation_genes = read_optional_csv(
        PROLIFERATION_GENE_FILE,
    )

    if FREEZE_FILE.exists():
        freeze = json.loads(
            FREEZE_FILE.read_text(encoding="utf-8")
        )
    else:
        freeze = {}

    common = expression.index.intersection(clinical.index)
    expression = expression.loc[common].copy()
    clinical = clinical.loc[common].copy()

    percent_necrosis, column_audit, response_metadata, patkar_path = (
        load_response_endpoint(clinical)
    )
    percent_necrosis = percent_necrosis.reindex(common)

    column_audit.to_csv(OUTPUT_COLUMN_AUDIT, index=False)

    scores, coverage = compute_frozen_scores(
        expression=expression,
        weights=weights,
    )
    coverage.to_csv(OUTPUT_COVERAGE, index=False)

    disjoint_proliferation, proliferation_metadata = (
        build_disjoint_proliferation_score(
            expression=expression,
            weights=weights,
            proliferation_genes=proliferation_genes,
            original_score=original_proliferation,
        )
    )

    original_meta_score = pd.Series(
        np.nan,
        index=common,
        dtype=float,
    )
    if not original_proliferation.empty:
        if (
            "meta_proliferation_score"
            in original_proliferation.columns
        ):
            original_meta_score = original_proliferation[
                "meta_proliferation_score"
            ].reindex(common)
        elif original_proliferation.shape[1] == 1:
            original_meta_score = original_proliferation.iloc[
                :, 0
            ].reindex(common)
        original_meta_score = zscore_series(
            original_meta_score
        )

    m40_primary_column = "M40__strict_signed_mean_z"
    if m40_primary_column not in scores.columns:
        raise RuntimeError(
            "Frozen M40 primary score could not be constructed."
        )

    m40_residual = residualize_score(
        scores[m40_primary_column],
        disjoint_proliferation,
    )

    matched = clinical.copy()
    matched["percent_necrosis"] = percent_necrosis
    matched["good_response_ge90"] = np.where(
        percent_necrosis.notna(),
        (
            percent_necrosis >= GOOD_RESPONSE_THRESHOLD
        ).astype(float),
        np.nan,
    )
    matched = matched.join(scores, how="left")
    matched[
        "meta_proliferation_score"
    ] = original_meta_score
    matched[
        "disjoint_meta_proliferation_score"
    ] = disjoint_proliferation
    matched[
        "M40__residual_after_disjoint_proliferation_z"
    ] = m40_residual
    matched.to_csv(OUTPUT_MATCHED)

    primary_rows = []
    for module_index, module in enumerate(PRIMARY_MODULES):
        column = f"{module}__strict_signed_mean_z"
        if column not in matched.columns:
            continue
        primary_rows.append(
            analyze_score(
                score=matched[column],
                percent_necrosis=matched["percent_necrosis"],
                module_label=module,
                score_variant="strict_signed_mean_z",
                seed=RANDOM_SEED + module_index * 100,
            )
        )

    primary = pd.DataFrame(primary_rows)
    primary["continuous_q_bh_4"] = bh_adjust(
        primary["spearman_permutation_p"]
    )
    primary["binary_q_bh_4"] = bh_adjust(
        primary["auc_permutation_p"]
    )

    global_p = pd.concat(
        [
            primary[
                [
                    "module_label",
                    "spearman_permutation_p",
                ]
            ].rename(
                columns={
                    "spearman_permutation_p": "p"
                }
            ).assign(response_test="continuous_necrosis"),
            primary[
                [
                    "module_label",
                    "auc_permutation_p",
                ]
            ].rename(
                columns={"auc_permutation_p": "p"}
            ).assign(response_test="good_response_ge90"),
        ],
        ignore_index=True,
    )
    global_p["global_q_bh_8"] = bh_adjust(global_p["p"])
    primary = primary.merge(
        global_p[
            global_p["response_test"].eq(
                "continuous_necrosis"
            )
        ][
            ["module_label", "global_q_bh_8"]
        ].rename(
            columns={
                "global_q_bh_8": (
                    "continuous_global_q_bh_8"
                )
            }
        ),
        on="module_label",
        how="left",
    )
    primary = primary.merge(
        global_p[
            global_p["response_test"].eq(
                "good_response_ge90"
            )
        ][
            ["module_label", "global_q_bh_8"]
        ].rename(
            columns={
                "global_q_bh_8": (
                    "binary_global_q_bh_8"
                )
            }
        ),
        on="module_label",
        how="left",
    )
    primary["prespecified_primary_exploratory_module"] = (
        primary["module_label"].eq(PRIMARY_TARGET_MODULE)
    )
    primary.to_csv(OUTPUT_PRIMARY, index=False)

    sensitivity_specs = [
        (
            "M40",
            "strict_canine_weighted_z",
            matched.get(
                "M40__strict_canine_weighted_z",
                pd.Series(np.nan, index=matched.index),
            ),
        ),
        (
            "M40",
            "residual_after_disjoint_proliferation_z",
            m40_residual,
        ),
        (
            "META_PROLIFERATION",
            "original_meta_proliferation_z",
            original_meta_score,
        ),
        (
            "META_PROLIFERATION",
            "disjoint_meta_proliferation_z",
            disjoint_proliferation,
        ),
    ]

    sensitivity_rows = []
    for index, (
        label,
        variant,
        score,
    ) in enumerate(sensitivity_specs):
        sensitivity_rows.append(
            analyze_score(
                score=score,
                percent_necrosis=matched["percent_necrosis"],
                module_label=label,
                score_variant=variant,
                seed=RANDOM_SEED + 1000 + index * 100,
            )
        )

    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(
        OUTPUT_SENSITIVITY,
        index=False,
    )

    loo = leave_one_out_m40(
        matched[m40_primary_column],
        matched["percent_necrosis"],
    )
    loo.to_csv(OUTPUT_LOO, index=False)

    random_summary, random_distribution = random_panel_controls(
        expression=expression,
        weights=weights,
        percent_necrosis=matched["percent_necrosis"],
    )
    random_summary.to_csv(OUTPUT_RANDOM, index=False)
    random_distribution.to_csv(
        OUTPUT_RANDOM_DISTRIBUTION,
        index=False,
    )

    response_values = matched["percent_necrosis"].dropna()
    good = response_values >= GOOD_RESPONSE_THRESHOLD
    endpoint_summary = pd.DataFrame(
        [
            {
                "response_source": response_metadata[
                    "response_source"
                ],
                "response_sheet": response_metadata.get(
                    "response_sheet",
                    "",
                ),
                "response_column": response_metadata[
                    "response_column"
                ],
                "n_total_dogs": matched.shape[0],
                "n_necrosis_complete": response_values.shape[0],
                "n_good_response_ge90": int(good.sum()),
                "n_poor_response_lt90": int((~good).sum()),
                "minimum_percent_necrosis": float(
                    response_values.min()
                ),
                "median_percent_necrosis": float(
                    response_values.median()
                ),
                "maximum_percent_necrosis": float(
                    response_values.max()
                ),
            }
        ]
    )
    endpoint_summary.to_csv(
        OUTPUT_RESPONSE_SUMMARY,
        index=False,
    )

    create_figures(
        frame=matched,
        score_column=m40_primary_column,
    )
    write_readme(
        response_metadata=response_metadata,
        proliferation_metadata=proliferation_metadata,
    )

    input_paths = [
        EXPRESSION_FILE,
        CLINICAL_FILE,
        STRICT_WEIGHTS_FILE,
    ]
    for path in [
        FREEZE_FILE,
        PROLIFERATION_SCORE_FILE,
        PROLIFERATION_GENE_FILE,
    ]:
        if path.exists():
            input_paths.append(path)
    if patkar_path is not None:
        input_paths.append(patkar_path)

    output_paths = [
        OUTPUT_COLUMN_AUDIT,
        OUTPUT_MATCHED,
        OUTPUT_COVERAGE,
        OUTPUT_PRIMARY,
        OUTPUT_SENSITIVITY,
        OUTPUT_LOO,
        OUTPUT_RANDOM,
        OUTPUT_RANDOM_DISTRIBUTION,
        OUTPUT_RESPONSE_SUMMARY,
        OUTPUT_README,
        SCATTER_PNG,
        SCATTER_PDF,
        BINARY_PNG,
        BINARY_PDF,
    ]

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "response_metadata": response_metadata,
        "primary_module": PRIMARY_TARGET_MODULE,
        "good_response_threshold_percent": (
            GOOD_RESPONSE_THRESHOLD
        ),
        "frozen_program_definition": freeze,
        "permutations_per_test": N_PERMUTATIONS,
        "bootstrap_repetitions": N_BOOTSTRAP,
        "random_panels": N_RANDOM_PANELS,
        "outcome_role": (
            "Post-treatment pathological response; exploratory "
            "association, not treatment-benefit estimation."
        ),
        "guardrails": [
            "No module gene, loading, or direction was changed.",
            "M40 was specified before response analysis as the primary exploratory module.",
            "Percent necrosis is post-treatment and is not a baseline confounder.",
            "No untreated comparator is available, so chemotherapy benefit is not estimated.",
            "This analysis cannot revise locked human evidence tiers.",
        ],
        "inputs": {
            str(path): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in input_paths
        },
        "outputs": {
            str(path): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in output_paths
            if path.exists()
        },
    }
    OUTPUT_MANIFEST.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("")
    print("=" * 80)
    print("Pathological-response endpoint audit")
    print("=" * 80)
    print(endpoint_summary.to_string(index=False))
    print("")
    print("Selected response metadata:")
    print(json.dumps(response_metadata, indent=2))

    print("")
    print("=" * 80)
    print("Primary frozen-module response associations")
    print("=" * 80)
    display_columns = [
        "module_label",
        "n_continuous",
        "spearman_rho",
        "rho_ci_low",
        "rho_ci_high",
        "spearman_permutation_p",
        "continuous_q_bh_4",
        "continuous_global_q_bh_8",
        "slope_percent_necrosis_per_score_sd",
        "n_good_response",
        "n_poor_response",
        "auc_higher_score_predicts_good_response",
        "auc_ci_low",
        "auc_ci_high",
        "logistic_or_good_response_per_score_sd",
        "or_ci_low",
        "or_ci_high",
        "auc_permutation_p",
        "binary_q_bh_4",
        "binary_global_q_bh_8",
    ]
    print(
        primary[
            [
                column
                for column in display_columns
                if column in primary.columns
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("M40 and proliferation sensitivity")
    print("=" * 80)
    print(
        sensitivity[
            [
                "module_label",
                "score_variant",
                "n_continuous",
                "spearman_rho",
                "spearman_permutation_p",
                "slope_percent_necrosis_per_score_sd",
                "auc_higher_score_predicts_good_response",
                "logistic_or_good_response_per_score_sd",
                "auc_permutation_p",
                "error",
            ]
        ].to_string(index=False)
    )
    print("")
    print("Disjoint proliferation construction:")
    print(json.dumps(proliferation_metadata, indent=2))

    print("")
    print("=" * 80)
    print("M40 leave-one-out stability")
    print("=" * 80)
    print(
        pd.DataFrame(
            [
                {
                    "n_leave_one_out_runs": loo.shape[0],
                    "minimum_spearman_rho": loo[
                        "spearman_rho"
                    ].min(),
                    "median_spearman_rho": loo[
                        "spearman_rho"
                    ].median(),
                    "maximum_spearman_rho": loo[
                        "spearman_rho"
                    ].max(),
                    "fraction_same_rho_direction": float(
                        (
                            np.sign(loo["spearman_rho"])
                            == np.sign(
                                primary.loc[
                                    primary[
                                        "module_label"
                                    ].eq("M40"),
                                    "spearman_rho",
                                ].iloc[0]
                            )
                        ).mean()
                    ),
                    "minimum_auc": loo[
                        "auc_higher_score_predicts_good_response"
                    ].min(),
                    "median_auc": loo[
                        "auc_higher_score_predicts_good_response"
                    ].median(),
                    "maximum_auc": loo[
                        "auc_higher_score_predicts_good_response"
                    ].max(),
                }
            ]
        ).to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("M40 variability-matched random controls")
    print("=" * 80)
    print(random_summary.to_string(index=False))

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("Percent necrosis is a post-treatment pathological response endpoint.")
    print("Association may be described as association with subsequent pathological response, not chemotherapy benefit.")
    print("M40 is the prespecified primary exploratory response hypothesis; other modules are exploratory.")
    print("Frozen human transfer results and validation tiers remain unchanged.")
    print("Random panels are specificity controls within the same DOG2 cohort, not external validation.")

    print("")
    print("Saved:")
    for path in output_paths + [OUTPUT_MANIFEST]:
        if path.exists():
            print(path)
    print("Done.")


if __name__ == "__main__":
    main()
