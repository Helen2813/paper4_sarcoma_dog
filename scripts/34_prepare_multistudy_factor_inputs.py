from __future__ import annotations

from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_VERSION = "34-prepare-multistudy-factor-inputs-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
HUMAN_DIR = PROCESSED_DIR / "human_validation"
MSFA_DIR = PROCESSED_DIR / "multistudy_factor"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"

MSFA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CANINE_EXPRESSION_FILE = (
    PROCESSED_DIR / "GSE238110_DOG2_expression_log2cpm_matched_allgenes.csv"
)
TARGET_EXPRESSION_FILE = HUMAN_DIR / "TARGET_OS_expression_log2_gene_symbol.csv"
GSE21257_EXPRESSION_FILE = HUMAN_DIR / "GSE21257_expression_gene_symbol.csv"
GSE39055_EXPRESSION_FILE = HUMAN_DIR / "GSE39055_expression_gene_symbol.csv"

ORTHOLOG_QC_FILE = (
    RESULTS_DIR / "GSE238110_RNA_master_candidate_evidence_table_with_ortholog_qc.csv"
)
STRICT_WEIGHTS_FILE = (
    RESULTS_DIR / "GSE238110_frozen_transfer_gene_weights_strict.csv"
)
GSE39055_PROBE_SELECTION_FILE = (
    RESULTS_DIR / "GSE39055_gene_probe_selection_comparison.csv"
)
ASSAY_AWARE_LOCK_FILE = (
    RESULTS_DIR / "paper4_assay_aware_locked_module_evidence_summary.csv"
)

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]
COHORT_ORDER = ["DOG2", "TARGET_OS", "GSE21257", "GSE39055"]

STRICT_ORTHOLOG_STATUS = "strict_symbol_concordant_one_to_one"
TOP_BACKGROUND_GENES = 500
TOP_ASSAY_AWARE_GENES = 350
MIN_GSE39055_DETECTION_FRACTION = 0.50
MAX_PCA_COMPONENTS = 20
RANDOM_SEED = 42

ANALYSIS_SETS = {
    "four_cohort_core_plus_frozen": {
        "cohorts": ["DOG2", "TARGET_OS", "GSE21257", "GSE39055"],
        "background_n": TOP_BACKGROUND_GENES,
        "require_gse39055_detection": False,
    },
    "four_cohort_detection_aware": {
        "cohorts": ["DOG2", "TARGET_OS", "GSE21257", "GSE39055"],
        "background_n": TOP_ASSAY_AWARE_GENES,
        "require_gse39055_detection": True,
    },
    "three_cohort_no_ffpe": {
        "cohorts": ["DOG2", "TARGET_OS", "GSE21257"],
        "background_n": TOP_BACKGROUND_GENES,
        "require_gse39055_detection": False,
    },
}

OUTPUT_GENE_UNIVERSE = RESULTS_DIR / "multistudy_factor_strict_ortholog_universe.csv"
OUTPUT_SET_SUMMARY = RESULTS_DIR / "multistudy_factor_input_set_summary.csv"
OUTPUT_MODULE_COVERAGE = RESULTS_DIR / "multistudy_factor_frozen_module_coverage.csv"
OUTPUT_PCA_DIAGNOSTICS = RESULTS_DIR / "multistudy_factor_pca_rank_diagnostics.csv"
OUTPUT_SHIFT_DIAGNOSTICS = RESULTS_DIR / "multistudy_factor_pairwise_shift_diagnostics.csv"
OUTPUT_CONFIG = RESULTS_DIR / "multistudy_factor_model_config.json"
OUTPUT_MANIFEST = RESULTS_DIR / "multistudy_factor_input_manifest.json"
OUTPUT_README = RESULTS_DIR / "multistudy_factor_input_README.txt"


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


def clean_human_symbol(value: Any) -> str:
    text = str(value).strip().upper()
    if text in {"", "NAN", "NONE", "NA"}:
        return ""
    return text


def robust_scale_rank_gaussian(expression: pd.DataFrame) -> pd.DataFrame:
    x = expression.apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(axis=0))

    transformed = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    n = x.shape[0]

    for column in x.columns:
        values = x[column].to_numpy(dtype=float)
        ranks = stats.rankdata(values, method="average")
        probabilities = (ranks - 0.5) / n
        probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
        transformed[column] = stats.norm.ppf(probabilities)

    means = transformed.mean(axis=0)
    stds = transformed.std(axis=0).replace(0, np.nan)
    transformed = (transformed - means) / stds
    transformed = transformed.loc[:, transformed.notna().all(axis=0)]
    return transformed


def robust_variability(expression: pd.DataFrame) -> pd.Series:
    x = expression.apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(axis=0))

    values = {}
    for column in x.columns:
        series = x[column].to_numpy(dtype=float)
        median = np.median(series)
        mad = np.median(np.abs(series - median))
        iqr = np.quantile(series, 0.75) - np.quantile(series, 0.25)
        value = mad if mad > 0 else iqr / 1.349 if iqr > 0 else np.std(series)
        values[column] = float(value)

    return pd.Series(values, dtype=float)


def percentile_rank(values: pd.Series) -> pd.Series:
    valid = values.replace([np.inf, -np.inf], np.nan)
    return valid.rank(method="average", pct=True)


def detect_ortholog_columns(table: pd.DataFrame) -> tuple[str, str, str]:
    dog_candidates = [
        "gene",
        "canine_gene",
        "dog_gene",
        "dog_gene_symbol",
        "canine_gene_symbol",
    ]
    human_candidates = [
        "human_gene_symbol",
        "human_symbol",
        "human_gene",
    ]
    status_candidates = [
        "ortholog_qc_status",
        "mapping_status",
        "qc_status",
    ]

    dog_col = next((column for column in dog_candidates if column in table.columns), None)
    human_col = next((column for column in human_candidates if column in table.columns), None)
    status_col = next((column for column in status_candidates if column in table.columns), None)

    if dog_col is None or human_col is None or status_col is None:
        raise ValueError(
            "Could not detect ortholog columns. "
            f"Columns: {list(table.columns)}"
        )
    return dog_col, human_col, status_col


def load_expression_matrices() -> dict[str, pd.DataFrame]:
    matrices = {
        "DOG2": read_required_csv(CANINE_EXPRESSION_FILE, index_col=0),
        "TARGET_OS": read_required_csv(TARGET_EXPRESSION_FILE, index_col=0),
        "GSE21257": read_required_csv(GSE21257_EXPRESSION_FILE, index_col=0),
        "GSE39055": read_required_csv(GSE39055_EXPRESSION_FILE, index_col=0),
    }

    for cohort in ["TARGET_OS", "GSE21257", "GSE39055"]:
        matrix = matrices[cohort].copy()
        matrix.columns = matrix.columns.astype(str).str.upper()
        matrix = matrix.loc[:, ~matrix.columns.duplicated(keep="first")]
        matrices[cohort] = matrix

    matrices["DOG2"].columns = matrices["DOG2"].columns.astype(str)
    matrices["DOG2"] = matrices["DOG2"].loc[
        :,
        ~matrices["DOG2"].columns.duplicated(keep="first"),
    ]
    return matrices


def build_strict_ortholog_universe(
    ortholog_qc: pd.DataFrame,
    matrices: dict[str, pd.DataFrame],
    detection_table: pd.DataFrame,
    strict_weights: pd.DataFrame,
) -> pd.DataFrame:
    dog_col, human_col, status_col = detect_ortholog_columns(ortholog_qc)

    mapping = ortholog_qc[
        ortholog_qc[status_col].astype(str).eq(STRICT_ORTHOLOG_STATUS)
    ].copy()
    mapping["canine_gene"] = mapping[dog_col].astype(str)
    mapping["human_gene_symbol"] = mapping[human_col].map(clean_human_symbol)
    mapping = mapping[mapping["human_gene_symbol"].ne("")].copy()
    mapping = mapping.drop_duplicates("canine_gene", keep=False)
    mapping = mapping.drop_duplicates("human_gene_symbol", keep=False)

    for cohort in COHORT_ORDER:
        if cohort == "DOG2":
            mapping[f"available_{cohort}"] = mapping["canine_gene"].isin(
                matrices[cohort].columns
            )
        else:
            mapping[f"available_{cohort}"] = mapping["human_gene_symbol"].isin(
                matrices[cohort].columns
            )

    detection = detection_table.copy()
    detection["gene_symbol"] = detection["gene_symbol"].map(clean_human_symbol)
    detection_col = "best_detected_detected_fraction_p_lt_0_01"
    if detection_col not in detection.columns:
        raise ValueError(
            f"Expected column {detection_col} not found in "
            f"{GSE39055_PROBE_SELECTION_FILE.name}."
        )

    detection = detection[
        ["gene_symbol", detection_col]
    ].drop_duplicates("gene_symbol", keep="first")
    detection = detection.rename(
        columns={
            "gene_symbol": "human_gene_symbol",
            detection_col: "gse39055_best_detected_fraction_p_lt_0_01",
        }
    )
    mapping = mapping.merge(detection, on="human_gene_symbol", how="left")

    weights = strict_weights.copy()
    weights["human_gene_symbol"] = weights["human_gene_symbol"].map(clean_human_symbol)
    primary_genes = set(
        weights[
            weights["module_label"].isin(PRIMARY_MODULES)
        ]["human_gene_symbol"].dropna()
    )
    mapping["is_primary_frozen_gene"] = mapping["human_gene_symbol"].isin(
        primary_genes
    )

    return mapping[
        [
            "canine_gene",
            "human_gene_symbol",
            *[f"available_{cohort}" for cohort in COHORT_ORDER],
            "gse39055_best_detected_fraction_p_lt_0_01",
            "is_primary_frozen_gene",
        ]
    ].sort_values("human_gene_symbol").reset_index(drop=True)


def add_variability_metrics(
    universe: pd.DataFrame,
    matrices: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    result = universe.copy()

    for cohort in COHORT_ORDER:
        if cohort == "DOG2":
            genes = result["canine_gene"].tolist()
            matrix = matrices[cohort].reindex(columns=genes)
            matrix.columns = result["human_gene_symbol"].tolist()
        else:
            genes = result["human_gene_symbol"].tolist()
            matrix = matrices[cohort].reindex(columns=genes)

        variability = robust_variability(matrix)
        result[f"{cohort}_robust_variability"] = result[
            "human_gene_symbol"
        ].map(variability)
        result[f"{cohort}_variability_percentile"] = percentile_rank(
            result[f"{cohort}_robust_variability"]
        )

    percentile_cols = [
        f"{cohort}_variability_percentile" for cohort in COHORT_ORDER
    ]
    result["minimum_four_cohort_variability_percentile"] = result[
        percentile_cols
    ].min(axis=1)
    result["median_four_cohort_variability_percentile"] = result[
        percentile_cols
    ].median(axis=1)
    result["minimum_three_cohort_variability_percentile"] = result[
        [f"{cohort}_variability_percentile" for cohort in COHORT_ORDER[:3]]
    ].min(axis=1)
    result["median_three_cohort_variability_percentile"] = result[
        [f"{cohort}_variability_percentile" for cohort in COHORT_ORDER[:3]]
    ].median(axis=1)
    return result


def select_analysis_set(
    universe: pd.DataFrame,
    set_name: str,
    specification: dict[str, Any],
) -> pd.DataFrame:
    cohorts = specification["cohorts"]
    available_cols = [f"available_{cohort}" for cohort in cohorts]
    eligible = universe[universe[available_cols].all(axis=1)].copy()

    if specification["require_gse39055_detection"]:
        eligible = eligible[
            eligible["gse39055_best_detected_fraction_p_lt_0_01"].ge(
                MIN_GSE39055_DETECTION_FRACTION
            )
        ].copy()

    if len(cohorts) == 4:
        sort_cols = [
            "minimum_four_cohort_variability_percentile",
            "median_four_cohort_variability_percentile",
        ]
    else:
        sort_cols = [
            "minimum_three_cohort_variability_percentile",
            "median_three_cohort_variability_percentile",
        ]

    background = (
        eligible[~eligible["is_primary_frozen_gene"]]
        .sort_values(sort_cols, ascending=False)
        .head(int(specification["background_n"]))
    )
    frozen = eligible[eligible["is_primary_frozen_gene"]].copy()

    selected = pd.concat([background, frozen], axis=0, ignore_index=True)
    selected = selected.drop_duplicates("human_gene_symbol", keep="first")
    selected["analysis_set"] = set_name
    selected["selection_role"] = np.where(
        selected["is_primary_frozen_gene"],
        "forced_primary_frozen_gene",
        "outcome_blind_cross_study_variable_gene",
    )
    selected = selected.sort_values(
        ["selection_role", "human_gene_symbol"]
    ).reset_index(drop=True)

    if selected.shape[0] < 50:
        raise RuntimeError(
            f"Analysis set {set_name} retained only {selected.shape[0]} genes."
        )
    return selected


def matrix_for_set(
    cohort: str,
    selected: pd.DataFrame,
    matrices: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    human_symbols = selected["human_gene_symbol"].tolist()

    if cohort == "DOG2":
        dog_genes = selected.set_index("human_gene_symbol").loc[
            human_symbols, "canine_gene"
        ].tolist()
        matrix = matrices[cohort].reindex(columns=dog_genes).copy()
        matrix.columns = human_symbols
    else:
        matrix = matrices[cohort].reindex(columns=human_symbols).copy()

    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    all_missing = matrix.columns[matrix.isna().all(axis=0)]
    if len(all_missing):
        raise RuntimeError(
            f"{cohort} has all-missing selected genes: {list(all_missing[:10])}"
        )

    transformed = robust_scale_rank_gaussian(matrix)
    transformed = transformed.reindex(columns=human_symbols)

    if transformed.isna().any().any():
        raise RuntimeError(f"NaN values remain after transformation for {cohort}.")
    return transformed


def pca_diagnostics(
    matrix: pd.DataFrame,
    set_name: str,
    cohort: str,
) -> pd.DataFrame:
    x = matrix.to_numpy(dtype=float)
    singular_values = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    eigenvalues = singular_values ** 2
    proportions = eigenvalues / eigenvalues.sum()

    positive = proportions[proportions > 0]
    effective_rank = float(np.exp(-np.sum(positive * np.log(positive))))
    cumulative = np.cumsum(proportions)

    rows = []
    for component in range(min(MAX_PCA_COMPONENTS, len(proportions))):
        rows.append(
            {
                "analysis_set": set_name,
                "cohort": cohort,
                "n_samples": matrix.shape[0],
                "n_genes": matrix.shape[1],
                "effective_rank": effective_rank,
                "component": component + 1,
                "variance_explained": float(proportions[component]),
                "cumulative_variance_explained": float(cumulative[component]),
            }
        )
    return pd.DataFrame(rows)


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[indices]


def rv_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    numerator = np.trace(a.T @ b)
    denominator = np.sqrt(np.trace(a.T @ a) * np.trace(b.T @ b))
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)


def pairwise_shift_diagnostics(
    matrices: dict[str, pd.DataFrame],
    set_name: str,
) -> pd.DataFrame:
    rows = []

    correlation_matrices = {
        cohort: np.corrcoef(matrix.to_numpy(dtype=float), rowvar=False)
        for cohort, matrix in matrices.items()
    }

    for cohort_a, cohort_b in combinations(matrices.keys(), 2):
        corr_a = correlation_matrices[cohort_a]
        corr_b = correlation_matrices[cohort_b]
        edges_a = upper_triangle_values(corr_a)
        edges_b = upper_triangle_values(corr_b)

        spearman = stats.spearmanr(edges_a, edges_b).statistic
        pearson = np.corrcoef(edges_a, edges_b)[0, 1]
        frobenius = np.linalg.norm(corr_a - corr_b, ord="fro") / corr_a.shape[0]

        rows.append(
            {
                "analysis_set": set_name,
                "cohort_a": cohort_a,
                "cohort_b": cohort_b,
                "edge_spearman": float(spearman),
                "edge_pearson": float(pearson),
                "correlation_matrix_rv": rv_coefficient(corr_a, corr_b),
                "normalized_frobenius_distance": float(frobenius),
            }
        )
    return pd.DataFrame(rows)


def module_coverage(
    selected_sets: dict[str, pd.DataFrame],
    strict_weights: pd.DataFrame,
) -> pd.DataFrame:
    weights = strict_weights.copy()
    weights["human_gene_symbol"] = weights["human_gene_symbol"].map(
        clean_human_symbol
    )

    rows = []
    for set_name, selected in selected_sets.items():
        selected_genes = set(selected["human_gene_symbol"])
        for module in PRIMARY_MODULES:
            module_genes = set(
                weights[
                    weights["module_label"].eq(module)
                ]["human_gene_symbol"].dropna()
            )
            available = sorted(module_genes.intersection(selected_genes))
            rows.append(
                {
                    "analysis_set": set_name,
                    "module_label": module,
                    "n_frozen_genes": len(module_genes),
                    "n_selected_genes": len(available),
                    "coverage_fraction": (
                        len(available) / len(module_genes)
                        if module_genes else np.nan
                    ),
                    "selected_genes": ";".join(available),
                }
            )
    return pd.DataFrame(rows)


def write_readme() -> None:
    text = f"""Outcome-blind multi-study factor input preparation
Script version: {SCRIPT_VERSION}

Purpose
-------
Prepare harmonized canine and human osteosarcoma matrices for multi-study
factor analysis that separates shared and study-specific latent transcriptional
variation.

No outcomes are loaded or used by this script.

Feature universe
----------------
Strict symbol-concordant one-to-one canine-human orthologs from the locked
ortholog-QC table.

Analysis sets
-------------
1. four_cohort_core_plus_frozen
   Top {TOP_BACKGROUND_GENES} cross-study variable background genes plus all
   available frozen primary-module genes across DOG2, TARGET-OS, GSE21257,
   and GSE39055.

2. four_cohort_detection_aware
   Top {TOP_ASSAY_AWARE_GENES} cross-study variable genes among probes detected
   at P < 0.01 in at least {MIN_GSE39055_DETECTION_FRACTION:.0%} of GSE39055
   samples, plus available frozen primary-module genes that meet the same rule.

3. three_cohort_no_ffpe
   DOG2, TARGET-OS, and GSE21257 sensitivity analysis excluding the FFPE DASL
   cohort.

Transformation
--------------
Within each cohort, each gene is transformed by a rank-based inverse-normal
transformation and standardized. Cohort mean differences are therefore removed;
the analysis targets shared versus cohort-specific covariance structure.

Interpretation
--------------
The prepared matrices support unsupervised statistical machine learning.
Subsequent latent factors must be related to outcomes only after model fitting.
Factor-space alignment with frozen modules should use rotation-invariant
subspace projection in addition to individual-factor correlations.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("Prepare outcome-blind multi-study factor-analysis inputs")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {MSFA_DIR}")
    print("")
    print("Design:")
    print("  Use strict one-to-one canine-human orthologs.")
    print("  Select cross-study variable genes without outcomes.")
    print("  Force available frozen primary-module genes into the core feature set.")
    print("  Rank-Gaussian transform and standardize within each cohort.")
    print("  Prepare four-cohort, detection-aware, and no-FFPE analysis sets.")
    print("")

    matrices = load_expression_matrices()
    ortholog_qc = read_required_csv(ORTHOLOG_QC_FILE)
    strict_weights = read_required_csv(STRICT_WEIGHTS_FILE)
    detection_table = read_required_csv(GSE39055_PROBE_SELECTION_FILE)

    universe = build_strict_ortholog_universe(
        ortholog_qc=ortholog_qc,
        matrices=matrices,
        detection_table=detection_table,
        strict_weights=strict_weights,
    )
    universe = add_variability_metrics(universe, matrices)
    universe.to_csv(OUTPUT_GENE_UNIVERSE, index=False)

    selected_sets: dict[str, pd.DataFrame] = {}
    set_summary_rows = []
    pca_tables = []
    shift_tables = []
    generated_matrix_paths: list[Path] = []
    generated_metadata_paths: list[Path] = []

    for set_name, specification in ANALYSIS_SETS.items():
        print("")
        print("=" * 80)
        print(f"Preparing analysis set: {set_name}")
        print("=" * 80)

        selected = select_analysis_set(
            universe=universe,
            set_name=set_name,
            specification=specification,
        )
        selected_sets[set_name] = selected

        set_dir = MSFA_DIR / set_name
        set_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = set_dir / "gene_metadata.csv"
        selected.to_csv(metadata_path, index=False)
        generated_metadata_paths.append(metadata_path)

        transformed_matrices: dict[str, pd.DataFrame] = {}
        for cohort in specification["cohorts"]:
            transformed = matrix_for_set(
                cohort=cohort,
                selected=selected,
                matrices=matrices,
            )
            transformed_matrices[cohort] = transformed

            matrix_path = set_dir / f"{cohort}_rank_gaussian_z.csv"
            transformed.to_csv(matrix_path)
            generated_matrix_paths.append(matrix_path)

            pca_tables.append(
                pca_diagnostics(
                    matrix=transformed,
                    set_name=set_name,
                    cohort=cohort,
                )
            )
            print(
                f"  {cohort}: {transformed.shape[0]} samples x "
                f"{transformed.shape[1]} genes"
            )

        shift_tables.append(
            pairwise_shift_diagnostics(
                matrices=transformed_matrices,
                set_name=set_name,
            )
        )

        set_summary_rows.append(
            {
                "analysis_set": set_name,
                "cohorts": ";".join(specification["cohorts"]),
                "n_cohorts": len(specification["cohorts"]),
                "n_selected_genes": selected.shape[0],
                "n_background_genes": int(
                    selected["selection_role"].eq(
                        "outcome_blind_cross_study_variable_gene"
                    ).sum()
                ),
                "n_forced_primary_frozen_genes": int(
                    selected["selection_role"].eq(
                        "forced_primary_frozen_gene"
                    ).sum()
                ),
                "gse39055_detection_required": bool(
                    specification["require_gse39055_detection"]
                ),
            }
        )

    set_summary = pd.DataFrame(set_summary_rows)
    set_summary.to_csv(OUTPUT_SET_SUMMARY, index=False)

    coverage = module_coverage(selected_sets, strict_weights)
    coverage.to_csv(OUTPUT_MODULE_COVERAGE, index=False)

    pca_diagnostics_table = pd.concat(pca_tables, ignore_index=True)
    pca_diagnostics_table.to_csv(OUTPUT_PCA_DIAGNOSTICS, index=False)

    shift_diagnostics = pd.concat(shift_tables, ignore_index=True)
    shift_diagnostics.to_csv(OUTPUT_SHIFT_DIAGNOSTICS, index=False)

    config = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "analysis_sets": {
            set_name: {
                "directory": str(MSFA_DIR / set_name),
                "cohorts": specification["cohorts"],
                "gene_metadata": str(MSFA_DIR / set_name / "gene_metadata.csv"),
                "matrices": {
                    cohort: str(
                        MSFA_DIR
                        / set_name
                        / f"{cohort}_rank_gaussian_z.csv"
                    )
                    for cohort in specification["cohorts"]
                },
            }
            for set_name, specification in ANALYSIS_SETS.items()
        },
        "suggested_vimsfa_grids": [
            {"name": "compact", "K_common": 4, "J_specific": 2},
            {"name": "medium", "K_common": 6, "J_specific": 3},
            {"name": "expanded", "K_common": 8, "J_specific": 4},
        ],
        "primary_modules": PRIMARY_MODULES,
        "outcome_loaded": False,
    }
    OUTPUT_CONFIG.write_text(json.dumps(config, indent=2), encoding="utf-8")

    write_readme()

    input_paths = [
        CANINE_EXPRESSION_FILE,
        TARGET_EXPRESSION_FILE,
        GSE21257_EXPRESSION_FILE,
        GSE39055_EXPRESSION_FILE,
        ORTHOLOG_QC_FILE,
        STRICT_WEIGHTS_FILE,
        GSE39055_PROBE_SELECTION_FILE,
    ]
    output_paths = [
        OUTPUT_GENE_UNIVERSE,
        OUTPUT_SET_SUMMARY,
        OUTPUT_MODULE_COVERAGE,
        OUTPUT_PCA_DIAGNOSTICS,
        OUTPUT_SHIFT_DIAGNOSTICS,
        OUTPUT_CONFIG,
        OUTPUT_README,
        *generated_matrix_paths,
        *generated_metadata_paths,
    ]

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "outcome_loaded": False,
        "strict_ortholog_status": STRICT_ORTHOLOG_STATUS,
        "inputs": {},
        "outputs": {},
    }
    for path in input_paths:
        manifest["inputs"][path.name] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    for path in output_paths:
        manifest["outputs"][str(path.relative_to(PROJECT_ROOT))] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    OUTPUT_MANIFEST.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("")
    print("=" * 80)
    print("Multi-study factor input-set summary")
    print("=" * 80)
    print(set_summary.to_string(index=False))

    print("")
    print("=" * 80)
    print("Frozen primary-module coverage")
    print("=" * 80)
    print(
        coverage[
            [
                "analysis_set",
                "module_label",
                "n_frozen_genes",
                "n_selected_genes",
                "coverage_fraction",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("PCA effective-rank diagnostics")
    print("=" * 80)
    effective_rank = (
        pca_diagnostics_table[
            [
                "analysis_set",
                "cohort",
                "n_samples",
                "n_genes",
                "effective_rank",
            ]
        ]
        .drop_duplicates()
        .sort_values(["analysis_set", "cohort"])
    )
    print(effective_rank.to_string(index=False))

    print("")
    print("=" * 80)
    print("Outcome-blind pairwise covariance-shift diagnostics")
    print("=" * 80)
    print(
        shift_diagnostics[
            [
                "analysis_set",
                "cohort_a",
                "cohort_b",
                "edge_spearman",
                "correlation_matrix_rv",
                "normalized_frobenius_distance",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("No clinical endpoint or outcome label was loaded.")
    print("Feature selection used ortholog status, cross-study variability, frozen-module membership, and assay detectability only.")
    print("Within-cohort rank-Gaussian transformation removes mean-scale differences and targets covariance structure.")
    print("The four-cohort detection-aware set is a sensitivity analysis, not a replacement for the core set.")
    print("Latent-factor outcome associations must be analyzed only after the unsupervised model is frozen.")

    print("")
    print("Saved:")
    for path in [
        OUTPUT_GENE_UNIVERSE,
        OUTPUT_SET_SUMMARY,
        OUTPUT_MODULE_COVERAGE,
        OUTPUT_PCA_DIAGNOSTICS,
        OUTPUT_SHIFT_DIAGNOSTICS,
        OUTPUT_CONFIG,
        OUTPUT_MANIFEST,
        OUTPUT_README,
    ]:
        print(path)
    print(f"Matrix directories: {MSFA_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
