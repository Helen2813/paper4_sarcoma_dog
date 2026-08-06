from __future__ import annotations

from itertools import combinations
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_VERSION = "37-prepare-variable-only-mofa-inputs-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
HUMAN_DIR = PROCESSED_DIR / "human_validation"
OUTPUT_ROOT = PROCESSED_DIR / "multistudy_factor_variable_only"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CANINE_EXPRESSION_FILE = (
    PROCESSED_DIR / "GSE238110_DOG2_expression_log2cpm_matched_allgenes.csv"
)
TARGET_EXPRESSION_FILE = HUMAN_DIR / "TARGET_OS_expression_log2_gene_symbol.csv"
GSE21257_EXPRESSION_FILE = HUMAN_DIR / "GSE21257_expression_gene_symbol.csv"
GSE39055_EXPRESSION_FILE = HUMAN_DIR / "GSE39055_expression_gene_symbol.csv"

ORTHOLOG_UNIVERSE_FILE = (
    RESULTS_DIR / "multistudy_factor_strict_ortholog_universe.csv"
)
STRICT_WEIGHTS_FILE = (
    RESULTS_DIR / "GSE238110_frozen_transfer_gene_weights_strict.csv"
)
SCRIPT34_MANIFEST_FILE = (
    RESULTS_DIR / "multistudy_factor_input_manifest.json"
)

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]
COHORT_ORDER = ["DOG2", "TARGET_OS", "GSE21257", "GSE39055"]

MIN_GSE39055_DETECTION_FRACTION = 0.50

ANALYSIS_SETS = {
    "four_cohort_variable_only_1500": {
        "cohorts": ["DOG2", "TARGET_OS", "GSE21257", "GSE39055"],
        "n_genes": 1500,
        "require_gse39055_detection": False,
        "variability_min_col": "minimum_four_cohort_variability_percentile",
        "variability_median_col": "median_four_cohort_variability_percentile",
    },
    "four_cohort_detection_aware_variable_only_700": {
        "cohorts": ["DOG2", "TARGET_OS", "GSE21257", "GSE39055"],
        "n_genes": 700,
        "require_gse39055_detection": True,
        "variability_min_col": "minimum_four_cohort_variability_percentile",
        "variability_median_col": "median_four_cohort_variability_percentile",
    },
    "three_cohort_no_ffpe_variable_only_1500": {
        "cohorts": ["DOG2", "TARGET_OS", "GSE21257"],
        "n_genes": 1500,
        "require_gse39055_detection": False,
        "variability_min_col": "minimum_three_cohort_variability_percentile",
        "variability_median_col": "median_three_cohort_variability_percentile",
    },
}

OUTPUT_SELECTION = (
    RESULTS_DIR / "multistudy_factor_variable_only_gene_selection.csv"
)
OUTPUT_SET_SUMMARY = (
    RESULTS_DIR / "multistudy_factor_variable_only_set_summary.csv"
)
OUTPUT_MODULE_COVERAGE = (
    RESULTS_DIR / "multistudy_factor_variable_only_module_coverage.csv"
)
OUTPUT_SHIFT = (
    RESULTS_DIR / "multistudy_factor_variable_only_pairwise_shift.csv"
)
OUTPUT_CONFIG = (
    RESULTS_DIR / "multistudy_factor_variable_only_model_config.json"
)
OUTPUT_README = (
    RESULTS_DIR / "multistudy_factor_variable_only_README.txt"
)
OUTPUT_MANIFEST = (
    RESULTS_DIR / "multistudy_factor_variable_only_manifest.json"
)


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


def rank_gaussian_z(expression: pd.DataFrame) -> pd.DataFrame:
    x = expression.apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(axis=0))

    transformed = pd.DataFrame(
        index=x.index,
        columns=x.columns,
        dtype=float,
    )
    n_samples = x.shape[0]

    for column in x.columns:
        values = x[column].to_numpy(dtype=float)
        ranks = stats.rankdata(values, method="average")
        probabilities = (ranks - 0.5) / n_samples
        probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
        transformed[column] = stats.norm.ppf(probabilities)

    stds = transformed.std(axis=0).replace(0, np.nan)
    transformed = (
        transformed - transformed.mean(axis=0)
    ) / stds
    transformed = transformed.loc[
        :,
        transformed.notna().all(axis=0),
    ]
    return transformed


def load_expression_matrices() -> dict[str, pd.DataFrame]:
    matrices = {
        "DOG2": read_required_csv(CANINE_EXPRESSION_FILE, index_col=0),
        "TARGET_OS": read_required_csv(TARGET_EXPRESSION_FILE, index_col=0),
        "GSE21257": read_required_csv(GSE21257_EXPRESSION_FILE, index_col=0),
        "GSE39055": read_required_csv(GSE39055_EXPRESSION_FILE, index_col=0),
    }

    matrices["DOG2"].columns = matrices["DOG2"].columns.astype(str)
    matrices["DOG2"] = matrices["DOG2"].loc[
        :,
        ~matrices["DOG2"].columns.duplicated(keep="first"),
    ]

    for cohort in ["TARGET_OS", "GSE21257", "GSE39055"]:
        matrices[cohort].columns = (
            matrices[cohort].columns.astype(str).str.upper()
        )
        matrices[cohort] = matrices[cohort].loc[
            :,
            ~matrices[cohort].columns.duplicated(keep="first"),
        ]

    return matrices


def select_genes(
    universe: pd.DataFrame,
    set_name: str,
    specification: dict[str, Any],
) -> pd.DataFrame:
    result = universe.copy()
    availability_cols = [
        f"available_{cohort}"
        for cohort in specification["cohorts"]
    ]
    result = result[result[availability_cols].all(axis=1)].copy()

    if specification["require_gse39055_detection"]:
        result = result[
            result[
                "gse39055_best_detected_fraction_p_lt_0_01"
            ].ge(MIN_GSE39055_DETECTION_FRACTION)
        ].copy()

    min_col = specification["variability_min_col"]
    median_col = specification["variability_median_col"]

    result = result[
        result[min_col].notna()
        & result[median_col].notna()
    ].copy()

    # No frozen-module membership is used for selection.
    selected = (
        result.sort_values(
            [min_col, median_col, "human_gene_symbol"],
            ascending=[False, False, True],
        )
        .head(int(specification["n_genes"]))
        .copy()
    )

    selected["analysis_set"] = set_name
    selected["selection_rule"] = (
        "top_cross_study_variable_strict_ortholog_no_forced_module_genes"
    )
    selected["selection_rank"] = np.arange(
        1,
        selected.shape[0] + 1,
    )
    return selected.reset_index(drop=True)


def matrix_for_set(
    cohort: str,
    selected: pd.DataFrame,
    matrices: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    human_symbols = selected["human_gene_symbol"].astype(str).str.upper().tolist()

    if cohort == "DOG2":
        mapping = selected.set_index("human_gene_symbol")
        dog_genes = mapping.loc[
            human_symbols,
            "canine_gene",
        ].astype(str).tolist()
        matrix = matrices[cohort].reindex(columns=dog_genes).copy()
        matrix.columns = human_symbols
    else:
        matrix = matrices[cohort].reindex(columns=human_symbols).copy()

    if matrix.isna().all(axis=0).any():
        missing = matrix.columns[matrix.isna().all(axis=0)].tolist()
        raise RuntimeError(
            f"{cohort} contains all-missing selected genes: {missing[:10]}"
        )

    transformed = rank_gaussian_z(matrix)
    transformed = transformed.reindex(columns=human_symbols)

    if transformed.isna().any().any():
        raise RuntimeError(
            f"NaN values remain after transformation for {cohort}."
        )
    return transformed


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[indices]


def rv_coefficient(first: np.ndarray, second: np.ndarray) -> float:
    numerator = float(np.sum(first * second))
    denominator = float(
        np.sqrt(np.sum(first * first) * np.sum(second * second))
    )
    if denominator <= 0 or not np.isfinite(denominator):
        return np.nan
    return numerator / denominator


def pairwise_shift(
    matrices: dict[str, pd.DataFrame],
    analysis_set: str,
) -> pd.DataFrame:
    correlations = {
        cohort: np.corrcoef(
            matrix.to_numpy(dtype=float),
            rowvar=False,
        )
        for cohort, matrix in matrices.items()
    }

    rows = []
    for cohort_a, cohort_b in combinations(matrices.keys(), 2):
        corr_a = correlations[cohort_a]
        corr_b = correlations[cohort_b]

        edge_a = upper_triangle(corr_a)
        edge_b = upper_triangle(corr_b)

        rows.append(
            {
                "analysis_set": analysis_set,
                "cohort_a": cohort_a,
                "cohort_b": cohort_b,
                "edge_spearman": float(
                    stats.spearmanr(edge_a, edge_b).statistic
                ),
                "edge_pearson": float(
                    np.corrcoef(edge_a, edge_b)[0, 1]
                ),
                "correlation_matrix_rv": rv_coefficient(
                    corr_a,
                    corr_b,
                ),
                "normalized_frobenius_distance": float(
                    np.linalg.norm(
                        corr_a - corr_b,
                        ord="fro",
                    ) / corr_a.shape[0]
                ),
            }
        )

    return pd.DataFrame(rows)


def module_coverage(
    selected_sets: dict[str, pd.DataFrame],
    strict_weights: pd.DataFrame,
) -> pd.DataFrame:
    weights = strict_weights.copy()
    weights["human_gene_symbol"] = (
        weights["human_gene_symbol"].astype(str).str.upper()
    )

    rows = []
    for set_name, selected in selected_sets.items():
        selected_genes = set(
            selected["human_gene_symbol"].astype(str).str.upper()
        )

        for module in PRIMARY_MODULES:
            module_genes = set(
                weights[
                    weights["module_label"].eq(module)
                ]["human_gene_symbol"].drop_duplicates()
            )
            overlap = sorted(module_genes.intersection(selected_genes))

            rows.append(
                {
                    "analysis_set": set_name,
                    "module_label": module,
                    "n_frozen_genes": len(module_genes),
                    "n_naturally_selected_module_genes": len(overlap),
                    "natural_coverage_fraction": (
                        len(overlap) / len(module_genes)
                        if module_genes
                        else np.nan
                    ),
                    "naturally_selected_genes": ";".join(overlap),
                }
            )

    return pd.DataFrame(rows)


def write_readme() -> None:
    text = f"""Variable-only outcome-blind MOFA input preparation
Script version: {SCRIPT_VERSION}

Purpose
-------
Create a stricter baseline for independent latent-factor recurrence.

Unlike script 34, this script never forces frozen M34, M11, M24, or M40 genes
into the factor-analysis feature space. Genes are selected only by:
- strict one-to-one ortholog status,
- availability in the requested cohorts,
- outcome-blind cross-study variability,
- and, for the detection-aware set, GSE39055 Detection P-value coverage.

Analysis sets
-------------
- four_cohort_variable_only_1500
- four_cohort_detection_aware_variable_only_700
- three_cohort_no_ffpe_variable_only_1500

Interpretation
--------------
Any frozen-module genes present in these sets were selected naturally by the
same outcome-blind variability rule as all other genes. Subsequent module-to-
factor alignment is therefore a stronger test of latent recurrence than the
frozen-program-enriched analysis.

A module with very low natural gene coverage is not interpretable in this
baseline. No outcome is loaded or used.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("Prepare variable-only outcome-blind MOFA input sets")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Design:")
    print("  Select strict orthologs only by cross-study variability.")
    print("  Do not force any frozen module gene into the feature space.")
    print("  Prepare four-cohort, detection-aware, and no-FFPE sets.")
    print("  Rank-Gaussian transform within each cohort.")
    print("  Audit natural frozen-module coverage before model fitting.")
    print("")

    matrices = load_expression_matrices()
    universe = read_required_csv(ORTHOLOG_UNIVERSE_FILE)
    strict_weights = read_required_csv(STRICT_WEIGHTS_FILE)

    universe["human_gene_symbol"] = (
        universe["human_gene_symbol"].astype(str).str.upper()
    )

    selected_sets: dict[str, pd.DataFrame] = {}
    selection_tables = []
    set_summary_rows = []
    shift_tables = []
    generated_paths: list[Path] = []

    for set_name, specification in ANALYSIS_SETS.items():
        print("")
        print("=" * 80)
        print(f"Preparing set: {set_name}")
        print("=" * 80)

        selected = select_genes(
            universe=universe,
            set_name=set_name,
            specification=specification,
        )
        selected_sets[set_name] = selected
        selection_tables.append(selected)

        set_dir = OUTPUT_ROOT / set_name
        set_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = set_dir / "gene_metadata.csv"
        selected.to_csv(metadata_path, index=False)
        generated_paths.append(metadata_path)

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
            generated_paths.append(matrix_path)

            print(
                f"  {cohort}: {transformed.shape[0]} samples x "
                f"{transformed.shape[1]} genes"
            )

        shift_tables.append(
            pairwise_shift(
                matrices=transformed_matrices,
                analysis_set=set_name,
            )
        )

        set_summary_rows.append(
            {
                "analysis_set": set_name,
                "cohorts": ";".join(specification["cohorts"]),
                "n_cohorts": len(specification["cohorts"]),
                "requested_n_genes": specification["n_genes"],
                "selected_n_genes": selected.shape[0],
                "gse39055_detection_required": bool(
                    specification["require_gse39055_detection"]
                ),
                "selection_used_frozen_membership": False,
            }
        )

    selection = pd.concat(selection_tables, ignore_index=True)
    set_summary = pd.DataFrame(set_summary_rows)
    coverage = module_coverage(selected_sets, strict_weights)
    shift = pd.concat(shift_tables, ignore_index=True)

    selection.to_csv(OUTPUT_SELECTION, index=False)
    set_summary.to_csv(OUTPUT_SET_SUMMARY, index=False)
    coverage.to_csv(OUTPUT_MODULE_COVERAGE, index=False)
    shift.to_csv(OUTPUT_SHIFT, index=False)

    config = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "outcome_loaded": False,
        "selection_used_frozen_membership": False,
        "analysis_sets": {
            set_name: {
                "directory": str(OUTPUT_ROOT / set_name),
                "cohorts": specification["cohorts"],
                "gene_metadata": str(
                    OUTPUT_ROOT / set_name / "gene_metadata.csv"
                ),
                "matrices": {
                    cohort: str(
                        OUTPUT_ROOT
                        / set_name
                        / f"{cohort}_rank_gaussian_z.csv"
                    )
                    for cohort in specification["cohorts"]
                },
            }
            for set_name, specification in ANALYSIS_SETS.items()
        },
        "initial_factor_grids": [8, 12, 16],
    }
    OUTPUT_CONFIG.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    write_readme()

    input_paths = [
        CANINE_EXPRESSION_FILE,
        TARGET_EXPRESSION_FILE,
        GSE21257_EXPRESSION_FILE,
        GSE39055_EXPRESSION_FILE,
        ORTHOLOG_UNIVERSE_FILE,
        STRICT_WEIGHTS_FILE,
    ]
    if SCRIPT34_MANIFEST_FILE.exists():
        input_paths.append(SCRIPT34_MANIFEST_FILE)

    output_paths = [
        OUTPUT_SELECTION,
        OUTPUT_SET_SUMMARY,
        OUTPUT_MODULE_COVERAGE,
        OUTPUT_SHIFT,
        OUTPUT_CONFIG,
        OUTPUT_README,
        *generated_paths,
    ]

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "outcome_loaded": False,
        "selection_used_frozen_membership": False,
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
    print("Variable-only input-set summary")
    print("=" * 80)
    print(set_summary.to_string(index=False))

    print("")
    print("=" * 80)
    print("Natural frozen-module coverage")
    print("=" * 80)
    print(
        coverage[
            [
                "analysis_set",
                "module_label",
                "n_frozen_genes",
                "n_naturally_selected_module_genes",
                "natural_coverage_fraction",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Variable-only pairwise covariance shift")
    print("=" * 80)
    print(
        shift[
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
    print("No clinical outcome or endpoint was loaded.")
    print("Frozen module membership was not used for feature selection.")
    print("Frozen-module coverage is reported only after selection.")
    print("Low-coverage modules must not be interpreted in the variable-only baseline.")
    print("The next model fit must retain all predefined factor ranks without outcome-based selection.")

    print("")
    print("Saved:")
    for path in [
        OUTPUT_SELECTION,
        OUTPUT_SET_SUMMARY,
        OUTPUT_MODULE_COVERAGE,
        OUTPUT_SHIFT,
        OUTPUT_CONFIG,
        OUTPUT_README,
        OUTPUT_MANIFEST,
    ]:
        print(path)
    print(f"Matrix directory: {OUTPUT_ROOT}")
    print("Done.")


if __name__ == "__main__":
    main()
