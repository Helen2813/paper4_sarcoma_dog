from __future__ import annotations

from pathlib import Path
import gzip
import hashlib
import json
from datetime import datetime, timezone
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

SCRIPT_VERSION = "42-score-ammons-single-cell-localization-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "single_cell"
    / "Ammons_GSE252470"
)
PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "single_cell"
    / "Ammons_GSE252470"
)
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "single_cell_localization"
)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

METADATA_FILE = RAW_DIR / "meta.tsv"
EXPRESSION_FILE = RAW_DIR / "exprMatrix.tsv.gz"
PREFLIGHT_MANIFEST_FILE = (
    RESULTS_DIR / "Ammons_scRNA_preflight_manifest.json"
)
STRICT_WEIGHTS_FILE = (
    RESULTS_DIR / "GSE238110_frozen_transfer_gene_weights_strict.csv"
)

PRIMARY_MODULES = ["M34", "M40"]
SECONDARY_MODULES = ["M11", "M24"]
ALL_MODULES = PRIMARY_MODULES + SECONDARY_MODULES

DOG_COLUMN = "orig.ident"
CELL_ID_COLUMN = "barcode"
ANNOTATION_LEVELS = ["celltype.l1", "celltype.l2", "celltype.l3"]

MIN_CELLS_PER_DOG_CELLTYPE = 25
MIN_DOGS_PER_CELLTYPE = 3
MIN_CELLS_PER_DOG_COMPARTMENT = 50
MIN_DOGS_PER_TARGETED_CONTRAST = 4

RANK_GAUSSIAN_EPSILON = 1e-6
N_BOOTSTRAP = 5000
RANDOM_SEED = 42

OUTPUT_SELECTED_EXPRESSION = (
    PROCESSED_DIR / "Ammons_primary_frozen_gene_expression_rank_gaussian.npz"
)
OUTPUT_SELECTED_EXPRESSION_METADATA = (
    PROCESSED_DIR / "Ammons_primary_frozen_gene_expression_metadata.json"
)
OUTPUT_CELL_SCORES = (
    PROCESSED_DIR / "Ammons_frozen_program_cell_scores.csv.gz"
)
OUTPUT_DOG_CELLTYPE = (
    RESULTS_DIR / "Ammons_scRNA_dog_celltype_pseudobulk_scores.csv"
)
OUTPUT_CELLTYPE_SUMMARY = (
    RESULTS_DIR / "Ammons_scRNA_celltype_score_summary.csv"
)
OUTPUT_COMPARTMENT = (
    RESULTS_DIR / "Ammons_scRNA_dog_compartment_scores.csv"
)
OUTPUT_TARGETED = (
    RESULTS_DIR / "Ammons_scRNA_targeted_localization_tests.csv"
)
OUTPUT_GENE_LOCALIZATION = (
    RESULTS_DIR / "Ammons_scRNA_gene_celltype_localization.csv"
)
OUTPUT_RANK_STABILITY = (
    RESULTS_DIR / "Ammons_scRNA_celltype_rank_stability.csv"
)
OUTPUT_SCORE_COVERAGE = (
    RESULTS_DIR / "Ammons_scRNA_score_gene_coverage.csv"
)
OUTPUT_README = (
    RESULTS_DIR / "Ammons_scRNA_localization_README.txt"
)
OUTPUT_MANIFEST = (
    RESULTS_DIR / "Ammons_scRNA_localization_manifest.json"
)

HEATMAP_PNG = (
    FIGURES_DIR / "Ammons_scRNA_l1_program_localization_heatmap.png"
)
HEATMAP_PDF = (
    FIGURES_DIR / "Ammons_scRNA_l1_program_localization_heatmap.pdf"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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


def clean_gene_symbol(value: str) -> str:
    text = str(value).strip().upper()

    if "|" in text:
        parts = [
            part.strip()
            for part in text.split("|")
            if part.strip()
        ]
        symbol_like = [
            part
            for part in parts
            if not part.upper().startswith("ENSCAFG")
        ]
        text = symbol_like[-1] if symbol_like else parts[-1]

    return text


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    for path in [
        METADATA_FILE,
        EXPRESSION_FILE,
        PREFLIGHT_MANIFEST_FILE,
        STRICT_WEIGHTS_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    metadata = pd.read_csv(
        METADATA_FILE,
        sep="\t",
        dtype=str,
        low_memory=False,
    )
    weights = pd.read_csv(STRICT_WEIGHTS_FILE)
    manifest = json.loads(
        PREFLIGHT_MANIFEST_FILE.read_text(encoding="utf-8")
    )

    if bool(manifest.get("outcome_loaded", True)):
        raise RuntimeError(
            "The single-cell preflight manifest does not confirm "
            "outcome_loaded=false."
        )

    required_metadata_columns = [
        CELL_ID_COLUMN,
        DOG_COLUMN,
        *ANNOTATION_LEVELS,
    ]
    missing = [
        column
        for column in required_metadata_columns
        if column not in metadata.columns
    ]
    if missing:
        raise ValueError(
            f"Required metadata columns are missing: {missing}"
        )

    weights["human_gene_symbol"] = (
        weights["human_gene_symbol"].astype(str).str.upper()
    )
    weights["risk_oriented_loading"] = pd.to_numeric(
        weights["risk_oriented_loading"],
        errors="coerce",
    )
    weights = weights[
        weights["module_label"].isin(ALL_MODULES)
        & weights["risk_oriented_loading"].notna()
    ].copy()

    return metadata, weights, manifest


def expression_header() -> tuple[list[str], str]:
    with gzip.open(
        EXPRESSION_FILE,
        "rt",
        encoding="utf-8",
    ) as handle:
        header = handle.readline().rstrip("\n\r").split("\t")

    if len(header) < 2:
        raise RuntimeError(
            "Expression matrix header contains fewer than two columns."
        )

    return header[1:], header[0]


def extract_selected_expression(
    requested_genes: set[str],
    expected_cells: int,
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    values_by_gene: dict[str, list[np.ndarray]] = {}
    row_counts: dict[str, int] = {}

    with gzip.open(
        EXPRESSION_FILE,
        "rt",
        encoding="utf-8",
    ) as handle:
        handle.readline()

        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue

            gene_token, values_text = line.split("\t", 1)
            gene = clean_gene_symbol(gene_token)

            if gene not in requested_genes:
                continue

            values = np.fromstring(
                values_text,
                sep="\t",
                dtype=np.float32,
            )
            if values.size != expected_cells:
                raise RuntimeError(
                    f"Gene {gene} on line {line_number} has "
                    f"{values.size} values; expected {expected_cells}."
                )

            values_by_gene.setdefault(gene, []).append(values)
            row_counts[gene] = row_counts.get(gene, 0) + 1

    genes = sorted(values_by_gene)
    matrix = np.empty(
        (len(genes), expected_cells),
        dtype=np.float32,
    )

    for index, gene in enumerate(genes):
        arrays = values_by_gene[gene]
        if len(arrays) == 1:
            matrix[index] = arrays[0]
        else:
            matrix[index] = np.mean(
                np.stack(arrays, axis=0),
                axis=0,
                dtype=np.float32,
            )

    return genes, matrix, row_counts


def rank_gaussian_transform(
    expression: np.ndarray,
) -> np.ndarray:
    transformed = np.empty_like(expression, dtype=np.float32)
    n_cells = expression.shape[1]

    for gene_index in range(expression.shape[0]):
        values = expression[gene_index].astype(float)
        finite = np.isfinite(values)

        if finite.sum() < 2 or np.nanstd(values[finite]) == 0:
            transformed[gene_index] = 0.0
            continue

        fill_value = float(np.nanmedian(values[finite]))
        values[~finite] = fill_value

        ranks = stats.rankdata(values, method="average")
        probabilities = (ranks - 0.5) / n_cells
        probabilities = np.clip(
            probabilities,
            RANK_GAUSSIAN_EPSILON,
            1.0 - RANK_GAUSSIAN_EPSILON,
        )
        transformed[gene_index] = stats.norm.ppf(
            probabilities
        ).astype(np.float32)

    return transformed


def build_score_weights(
    weights: pd.DataFrame,
    detected_genes: list[str],
) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    gene_index = {
        gene: index for index, gene in enumerate(detected_genes)
    }
    score_weights: dict[str, dict[str, np.ndarray]] = {}
    coverage_rows = []

    for module in ALL_MODULES:
        part = weights[
            weights["module_label"].eq(module)
        ].drop_duplicates(
            "human_gene_symbol",
            keep="first",
        )

        available = part[
            part["human_gene_symbol"].isin(gene_index)
        ].copy()
        indices = np.asarray(
            [
                gene_index[gene]
                for gene in available["human_gene_symbol"]
            ],
            dtype=int,
        )
        loadings = available[
            "risk_oriented_loading"
        ].to_numpy(dtype=float)

        positive_mask = loadings > 0
        negative_mask = loadings < 0

        score_weights[module] = {
            "indices": indices,
            "loadings": loadings,
            "positive_indices": indices[positive_mask],
            "positive_weights": loadings[positive_mask],
            "negative_indices": indices[negative_mask],
            "negative_weights": np.abs(loadings[negative_mask]),
        }

        coverage_rows.append(
            {
                "module_label": module,
                "n_frozen_genes": int(part.shape[0]),
                "n_detected_genes": int(available.shape[0]),
                "coverage_fraction": float(
                    available.shape[0] / part.shape[0]
                ),
                "n_positive_detected": int(positive_mask.sum()),
                "n_negative_detected": int(negative_mask.sum()),
                "signed_score_estimable": bool(
                    available.shape[0] >= 3
                ),
                "positive_component_estimable": bool(
                    positive_mask.sum() >= 2
                ),
                "negative_component_estimable": bool(
                    negative_mask.sum() >= 2
                ),
                "component_guardrail": (
                    "M34 positive component is descriptive because "
                    "only two positive-loading genes are available."
                    if module == "M34"
                    else ""
                ),
            }
        )

    return score_weights, pd.DataFrame(coverage_rows)


def weighted_component(
    expression: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    if indices.size == 0:
        return np.full(
            expression.shape[1],
            np.nan,
            dtype=np.float32,
        )

    denominator = float(np.sum(np.abs(weights)))
    if denominator <= 0:
        return np.full(
            expression.shape[1],
            np.nan,
            dtype=np.float32,
        )

    return (
        weights.astype(np.float32)
        @ expression[indices]
        / denominator
    ).astype(np.float32)


def calculate_cell_scores(
    transformed_expression: np.ndarray,
    score_weights: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    score_columns: dict[str, np.ndarray] = {}

    for module, specification in score_weights.items():
        indices = specification["indices"]
        loadings = specification["loadings"]

        signed = weighted_component(
            transformed_expression,
            indices,
            loadings,
        )
        positive = weighted_component(
            transformed_expression,
            specification["positive_indices"],
            specification["positive_weights"],
        )
        negative_expression = weighted_component(
            transformed_expression,
            specification["negative_indices"],
            specification["negative_weights"],
        )

        score_columns[f"{module}_signed_risk_score"] = signed
        score_columns[
            f"{module}_positive_component_expression"
        ] = positive
        score_columns[
            f"{module}_negative_component_expression"
        ] = negative_expression

    return pd.DataFrame(score_columns)


def broad_compartment(label: Any) -> str:
    text = str(label).strip().lower()

    if "osteoblast" in text:
        return "osteoblast_lineage"
    if "fibroblast" in text or "endothelial" in text:
        return "stromal_vascular"
    if (
        text == "tam"
        or text == "tim"
        or text == "dc"
        or "neutrophil" in text
        or "macrophage" in text
        or "dendritic" in text
    ):
        return "myeloid_innate"
    if (
        "t cell" in text
        or text == "t_cycling"
        or "b cell" in text
        or "lymph" in text
    ):
        return "lymphoid"
    if (
        "_oc" in text
        or text.endswith("oc")
        or "osteoclast" in text
    ):
        return "osteoclast"

    return "other"


def aggregate_scores(
    cells: pd.DataFrame,
    group_columns: list[str],
    score_columns: list[str],
    minimum_cells: int,
) -> pd.DataFrame:
    grouped = cells.groupby(
        group_columns,
        dropna=False,
        observed=True,
    )

    counts = grouped.size().rename("n_cells")
    means = grouped[score_columns].mean()
    medians = grouped[score_columns].median()
    detection = grouped[
        [f"{module}_detected_gene_fraction" for module in ALL_MODULES]
    ].mean()

    means.columns = [f"{column}_mean" for column in means.columns]
    medians.columns = [
        f"{column}_median" for column in medians.columns
    ]
    detection.columns = [
        f"{column}_mean" for column in detection.columns
    ]

    result = pd.concat(
        [counts, means, medians, detection],
        axis=1,
    ).reset_index()
    return result[
        result["n_cells"].ge(minimum_cells)
    ].reset_index(drop=True)


def exact_sign_flip_test(
    differences: np.ndarray,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan, np.nan

    observed = float(np.mean(values))
    n = values.size

    if n <= 20:
        statistics = []
        for signs in product([-1.0, 1.0], repeat=n):
            statistics.append(
                float(
                    np.mean(
                        values * np.asarray(signs, dtype=float)
                    )
                )
            )
        statistics_array = np.asarray(statistics)
        p = float(
            np.mean(
                np.abs(statistics_array)
                >= abs(observed) - 1e-12
            )
        )
    else:
        rng = np.random.default_rng(RANDOM_SEED)
        signs = rng.choice(
            [-1.0, 1.0],
            size=(100000, n),
        )
        statistics_array = np.mean(
            signs * values[None, :],
            axis=1,
        )
        p = float(
            (
                1
                + np.sum(
                    np.abs(statistics_array)
                    >= abs(observed)
                )
            )
            / (statistics_array.size + 1)
        )

    return observed, p


def bootstrap_mean_ci(
    differences: np.ndarray,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.integers(
        0,
        values.size,
        size=(N_BOOTSTRAP, values.size),
    )
    means = values[indices].mean(axis=1)

    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def paired_contrast(
    pseudobulk: pd.DataFrame,
    group_column: str,
    group_a: str,
    group_b: str,
    score_column: str,
    contrast_name: str,
    hypothesis_direction: str,
) -> dict[str, Any]:
    part = pseudobulk[
        pseudobulk[group_column].isin([group_a, group_b])
    ][
        [DOG_COLUMN, group_column, score_column, "n_cells"]
    ].copy()

    pivot = part.pivot_table(
        index=DOG_COLUMN,
        columns=group_column,
        values=score_column,
        aggfunc="mean",
    )
    pivot = pivot.dropna(subset=[group_a, group_b])

    differences = (
        pivot[group_a] - pivot[group_b]
    ).to_numpy(dtype=float)

    mean_difference, p = exact_sign_flip_test(differences)
    ci_low, ci_high = bootstrap_mean_ci(differences)

    return {
        "contrast_name": contrast_name,
        "score_column": score_column,
        "group_column": group_column,
        "group_a": group_a,
        "group_b": group_b,
        "difference_definition": f"{group_a} minus {group_b}",
        "hypothesis_direction": hypothesis_direction,
        "n_paired_dogs": int(differences.size),
        "mean_paired_difference": mean_difference,
        "median_paired_difference": (
            float(np.median(differences))
            if differences.size
            else np.nan
        ),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "exact_sign_flip_p": p,
        "dog_level_differences": ";".join(
            f"{value:.6g}" for value in differences
        ),
        "estimable": bool(
            differences.size >= MIN_DOGS_PER_TARGETED_CONTRAST
        ),
    }


def targeted_tests(
    compartment_scores: pd.DataFrame,
    l1_scores: pd.DataFrame,
) -> pd.DataFrame:
    tests = []

    tests.append(
        paired_contrast(
            pseudobulk=compartment_scores,
            group_column="broad_compartment",
            group_a="osteoblast_lineage",
            group_b="immune_combined",
            score_column="M34_signed_risk_score_mean",
            contrast_name=(
                "M34 signed risk: osteoblast lineage versus immune"
            ),
            hypothesis_direction=(
                "Higher in osteoblast lineage"
            ),
        )
    )
    tests.append(
        paired_contrast(
            pseudobulk=compartment_scores,
            group_column="broad_compartment",
            group_a="immune_combined",
            group_b="osteoblast_lineage",
            score_column=(
                "M34_negative_component_expression_mean"
            ),
            contrast_name=(
                "M34 negative-loading expression: immune versus "
                "osteoblast lineage"
            ),
            hypothesis_direction="Higher in immune compartment",
        )
    )
    tests.append(
        paired_contrast(
            pseudobulk=l1_scores,
            group_column="celltype.l1",
            group_a="Osteoblast_cycling",
            group_b="Osteoblast",
            score_column="M40_signed_risk_score_mean",
            contrast_name=(
                "M40 signed risk: cycling versus non-cycling "
                "osteoblast"
            ),
            hypothesis_direction="Higher in cycling osteoblast",
        )
    )
    tests.append(
        paired_contrast(
            pseudobulk=l1_scores,
            group_column="celltype.l1",
            group_a="Osteoblast_cycling",
            group_b="Osteoblast",
            score_column=(
                "M40_positive_component_expression_mean"
            ),
            contrast_name=(
                "M40 positive-loading expression: cycling versus "
                "non-cycling osteoblast"
            ),
            hypothesis_direction="Higher in cycling osteoblast",
        )
    )

    result = pd.DataFrame(tests)
    result.loc[
        ~result["estimable"],
        "exact_sign_flip_p",
    ] = np.nan
    result["primary_bh_q"] = bh_adjust(
        result["exact_sign_flip_p"]
    )
    return result


def build_celltype_summary(
    dog_celltype: pd.DataFrame,
    score_columns: list[str],
) -> pd.DataFrame:
    rows = []

    for annotation_level in ANNOTATION_LEVELS:
        part = dog_celltype[
            dog_celltype["annotation_level"].eq(annotation_level)
        ]

        for cell_type, cell_part in part.groupby(
            "cell_type",
            observed=True,
        ):
            if cell_part[DOG_COLUMN].nunique() < MIN_DOGS_PER_CELLTYPE:
                continue

            for score_column in score_columns:
                values = cell_part[
                    f"{score_column}_mean"
                ].to_numpy(dtype=float)
                values = values[np.isfinite(values)]

                if values.size == 0:
                    continue

                rows.append(
                    {
                        "annotation_level": annotation_level,
                        "cell_type": cell_type,
                        "score_column": score_column,
                        "n_dogs": int(
                            cell_part[DOG_COLUMN].nunique()
                        ),
                        "total_cells": int(
                            cell_part["n_cells"].sum()
                        ),
                        "median_dog_score": float(
                            np.median(values)
                        ),
                        "mean_dog_score": float(
                            np.mean(values)
                        ),
                        "iqr_low": float(
                            np.quantile(values, 0.25)
                        ),
                        "iqr_high": float(
                            np.quantile(values, 0.75)
                        ),
                        "minimum_dog_score": float(
                            np.min(values)
                        ),
                        "maximum_dog_score": float(
                            np.max(values)
                        ),
                    }
                )

    return pd.DataFrame(rows)


def rank_stability(
    dog_celltype: pd.DataFrame,
    score_columns: list[str],
) -> pd.DataFrame:
    rows = []

    for annotation_level in ANNOTATION_LEVELS:
        part = dog_celltype[
            dog_celltype["annotation_level"].eq(annotation_level)
        ].copy()

        eligible_types = (
            part.groupby("cell_type")[DOG_COLUMN]
            .nunique()
            .loc[lambda series: series >= MIN_DOGS_PER_CELLTYPE]
            .index.tolist()
        )
        part = part[part["cell_type"].isin(eligible_types)]

        dogs = sorted(part[DOG_COLUMN].dropna().unique())
        if len(dogs) < MIN_DOGS_PER_CELLTYPE:
            continue

        for score_column in score_columns:
            score_name = f"{score_column}_mean"
            rank_records = []

            full_means = (
                part.groupby("cell_type")[score_name]
                .mean()
                .sort_values(ascending=False)
            )
            full_ranks = pd.Series(
                np.arange(1, len(full_means) + 1),
                index=full_means.index,
            )

            for left_out_dog in dogs:
                loo = part[
                    part[DOG_COLUMN].ne(left_out_dog)
                ]
                loo_means = (
                    loo.groupby("cell_type")[score_name]
                    .mean()
                    .sort_values(ascending=False)
                )
                loo_ranks = pd.Series(
                    np.arange(1, len(loo_means) + 1),
                    index=loo_means.index,
                )

                common = full_ranks.index.intersection(
                    loo_ranks.index
                )
                for cell_type in common:
                    rank_records.append(
                        {
                            "cell_type": cell_type,
                            "left_out_dog": left_out_dog,
                            "full_rank": int(
                                full_ranks.loc[cell_type]
                            ),
                            "loo_rank": int(
                                loo_ranks.loc[cell_type]
                            ),
                        }
                    )

            rank_table = pd.DataFrame(rank_records)
            if rank_table.empty:
                continue

            for cell_type, cell_part in rank_table.groupby(
                "cell_type"
            ):
                rows.append(
                    {
                        "annotation_level": annotation_level,
                        "score_column": score_column,
                        "cell_type": cell_type,
                        "full_rank": int(
                            cell_part["full_rank"].iloc[0]
                        ),
                        "minimum_loo_rank": int(
                            cell_part["loo_rank"].min()
                        ),
                        "maximum_loo_rank": int(
                            cell_part["loo_rank"].max()
                        ),
                        "median_loo_rank": float(
                            cell_part["loo_rank"].median()
                        ),
                        "maximum_absolute_rank_change": int(
                            np.max(
                                np.abs(
                                    cell_part["loo_rank"]
                                    - cell_part["full_rank"]
                                )
                            )
                        ),
                    }
                )

    return pd.DataFrame(rows)


def gene_celltype_localization(
    transformed_expression: np.ndarray,
    genes: list[str],
    metadata: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    gene_to_index = {
        gene: index for index, gene in enumerate(genes)
    }
    rows = []

    metadata_l1 = metadata["celltype.l1"].astype(str)
    eligible_types = (
        metadata_l1.value_counts()
        .loc[lambda series: series >= 100]
        .index.tolist()
    )

    for module in PRIMARY_MODULES:
        module_weights = weights[
            weights["module_label"].eq(module)
        ].drop_duplicates("human_gene_symbol")

        for weight_row in module_weights.itertuples(index=False):
            gene = str(weight_row.human_gene_symbol)
            if gene not in gene_to_index:
                continue

            values = transformed_expression[
                gene_to_index[gene]
            ]

            type_means = {}
            for cell_type in eligible_types:
                mask = metadata_l1.eq(cell_type).to_numpy()
                type_means[cell_type] = float(
                    np.mean(values[mask])
                )

            if not type_means:
                continue

            ordered = sorted(
                type_means.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            for cell_type, mean_value in ordered:
                rows.append(
                    {
                        "module_label": module,
                        "gene_symbol": gene,
                        "risk_oriented_loading": float(
                            weight_row.risk_oriented_loading
                        ),
                        "loading_sign": (
                            "positive"
                            if weight_row.risk_oriented_loading > 0
                            else "negative"
                        ),
                        "cell_type_l1": cell_type,
                        "mean_rank_gaussian_expression": mean_value,
                        "highest_expression_cell_type": ordered[0][0],
                        "highest_expression_value": ordered[0][1],
                        "specificity_range": (
                            ordered[0][1] - ordered[-1][1]
                        ),
                    }
                )

    return pd.DataFrame(rows)


def create_heatmap(
    celltype_summary: pd.DataFrame,
) -> None:
    selected_scores = [
        "M34_signed_risk_score",
        "M34_negative_component_expression",
        "M40_signed_risk_score",
        "M40_positive_component_expression",
    ]

    part = celltype_summary[
        celltype_summary["annotation_level"].eq("celltype.l1")
        & celltype_summary["score_column"].isin(selected_scores)
        & celltype_summary["n_dogs"].ge(MIN_DOGS_PER_CELLTYPE)
    ].copy()

    if part.empty:
        return

    matrix = part.pivot(
        index="cell_type",
        columns="score_column",
        values="median_dog_score",
    ).reindex(columns=selected_scores)

    matrix = matrix.loc[
        matrix.notna().any(axis=1)
    ].copy()

    order = matrix[
        "M34_signed_risk_score"
    ].sort_values(ascending=False).index
    matrix = matrix.loc[order]

    figure_height = max(5.0, 0.35 * matrix.shape[0] + 1.5)
    fig, ax = plt.subplots(
        figsize=(8.5, figure_height)
    )
    image = ax.imshow(
        matrix.to_numpy(dtype=float),
        aspect="auto",
    )

    labels = [
        "M34 signed risk",
        "M34 negative component",
        "M40 signed risk",
        "M40 positive component",
    ]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index.tolist())
    ax.set_title(
        "Ammons canine OS atlas: dog-level median cell-type localization"
    )

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix.iloc[row_index, column_index]
            text = "NA" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=7,
            )

    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(
        HEATMAP_PNG,
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        HEATMAP_PDF,
        bbox_inches="tight",
    )
    plt.close(fig)


def create_paired_plot(
    pseudobulk: pd.DataFrame,
    group_column: str,
    group_a: str,
    group_b: str,
    score_column: str,
    title: str,
    filename_stem: str,
) -> list[Path]:
    part = pseudobulk[
        pseudobulk[group_column].isin([group_a, group_b])
    ][
        [DOG_COLUMN, group_column, score_column]
    ].copy()

    pivot = part.pivot_table(
        index=DOG_COLUMN,
        columns=group_column,
        values=score_column,
        aggfunc="mean",
    ).dropna(subset=[group_a, group_b])

    if pivot.empty:
        return []

    fig, ax = plt.subplots(figsize=(5.4, 4.6))

    for dog, row in pivot.iterrows():
        ax.plot(
            [0, 1],
            [row[group_a], row[group_b]],
            marker="o",
            label=str(dog),
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels([group_a, group_b], rotation=15)
    ax.set_ylabel(score_column)
    ax.set_title(title)
    fig.tight_layout()

    png = FIGURES_DIR / f"{filename_stem}.png"
    pdf = FIGURES_DIR / f"{filename_stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return [png, pdf]


def write_readme(
    score_coverage: pd.DataFrame,
    targeted: pd.DataFrame,
) -> None:
    text = f"""Ammons canine osteosarcoma single-cell localization
Script version: {SCRIPT_VERSION}

Data
----
- 32,613 cells from the UCSC Cell Browser "All Cells" leaf.
- Six dog/sample identifiers are expected in orig.ident.
- Three annotation levels: celltype.l1, celltype.l2, celltype.l3.

Scoring
-------
Only frozen primary-program genes are extracted from the compressed matrix.
Each gene is rank-Gaussian transformed across all cells. Frozen risk-oriented
loadings are then applied without outcome information.

For each module:
- signed_risk_score uses signed frozen loadings normalized by their absolute sum;
- positive_component_expression uses positive loadings only;
- negative_component_expression uses the absolute magnitude of negative
  loadings and therefore represents expression of the protective/opposite-sign
  component.

Biological replication
----------------------
Primary summaries use dog x cell type or dog x broad compartment pseudobulks.
Individual cells are never treated as independent biological replicates.

Primary targeted contrasts
--------------------------
1. M34 signed risk: osteoblast lineage versus immune compartment.
2. M34 negative-loading expression: immune versus osteoblast lineage.
3. M40 signed risk: cycling versus non-cycling osteoblast.
4. M40 positive-loading expression: cycling versus non-cycling osteoblast.

The exact sign-flip test uses paired dog-level differences, and BH correction
is applied across the four targeted contrasts.

Important guardrails
--------------------
- M34 has only two detected positive-loading genes; that positive component is
  descriptive and is not a primary test.
- M11 and M24 contain no detected negative-loading genes and remain secondary.
- Single-cell localization is same-species biological annotation, not external
  outcome validation.
- A high M34 score in osteoblast-lineage cells and high negative-component
  expression in immune cells would support an immune-depletion/exclusion
  interpretation, not prove causal immune exclusion.

Score coverage:
{score_coverage.to_string(index=False)}

Targeted tests:
{targeted.to_string(index=False)}
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("Score frozen programs in Ammons canine OS single-cell atlas")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Design:")
    print("  Extract only frozen program genes from the 563.7-MB matrix.")
    print("  Rank-Gaussian transform each gene across cells.")
    print("  Compute signed, positive, and negative component scores.")
    print("  Aggregate by dog x cell type and dog x broad compartment.")
    print("  Use exact paired dog-level tests for four targeted contrasts.")
    print("")

    metadata, weights, preflight_manifest = read_inputs()
    expression_cells, gene_column = expression_header()

    if len(expression_cells) != metadata.shape[0]:
        raise RuntimeError(
            "Expression and metadata cell counts differ."
        )

    metadata_indexed = (
        metadata.set_index(CELL_ID_COLUMN, drop=False)
        .reindex(expression_cells)
    )
    if metadata_indexed[CELL_ID_COLUMN].isna().any():
        raise RuntimeError(
            "Some expression cells are missing from metadata."
        )
    metadata = metadata_indexed.reset_index(drop=True)

    requested_genes = set(
        weights["human_gene_symbol"].dropna()
    )
    genes, raw_expression, duplicate_rows = (
        extract_selected_expression(
            requested_genes=requested_genes,
            expected_cells=len(expression_cells),
        )
    )

    if not genes:
        raise RuntimeError(
            "No frozen-program genes were extracted."
        )

    print(
        f"Extracted {len(genes)} unique frozen-program genes "
        f"across {len(expression_cells)} cells."
    )

    transformed_expression = rank_gaussian_transform(
        raw_expression
    )
    score_weights, score_coverage = build_score_weights(
        weights=weights,
        detected_genes=genes,
    )
    cell_score_values = calculate_cell_scores(
        transformed_expression=transformed_expression,
        score_weights=score_weights,
    )

    cell_scores = metadata[
        [
            CELL_ID_COLUMN,
            DOG_COLUMN,
            *ANNOTATION_LEVELS,
        ]
    ].copy()
    cell_scores["broad_compartment"] = (
        cell_scores["celltype.l1"].map(broad_compartment)
    )
    cell_scores["immune_combined"] = np.where(
        cell_scores["broad_compartment"].isin(
            ["myeloid_innate", "lymphoid"]
        ),
        "immune_combined",
        cell_scores["broad_compartment"],
    )

    for column in cell_score_values.columns:
        cell_scores[column] = cell_score_values[column].to_numpy()

    gene_to_index = {
        gene: index for index, gene in enumerate(genes)
    }
    for module in ALL_MODULES:
        module_genes = (
            weights[
                weights["module_label"].eq(module)
                & weights["human_gene_symbol"].isin(gene_to_index)
            ]["human_gene_symbol"]
            .drop_duplicates()
            .tolist()
        )
        if module_genes:
            indices = [
                gene_to_index[gene] for gene in module_genes
            ]
            detected_fraction = np.mean(
                raw_expression[indices] > 0,
                axis=0,
            )
        else:
            detected_fraction = np.full(
                raw_expression.shape[1],
                np.nan,
            )
        cell_scores[
            f"{module}_detected_gene_fraction"
        ] = detected_fraction.astype(np.float32)

    score_columns = [
        column
        for column in cell_score_values.columns
        if column.endswith(
            (
                "_signed_risk_score",
                "_positive_component_expression",
                "_negative_component_expression",
            )
        )
    ]

    dog_celltype_tables = []
    for annotation_level in ANNOTATION_LEVELS:
        aggregated = aggregate_scores(
            cells=cell_scores,
            group_columns=[DOG_COLUMN, annotation_level],
            score_columns=score_columns,
            minimum_cells=MIN_CELLS_PER_DOG_CELLTYPE,
        )
        aggregated = aggregated.rename(
            columns={annotation_level: "cell_type"}
        )
        aggregated.insert(
            1,
            "annotation_level",
            annotation_level,
        )
        dog_celltype_tables.append(aggregated)

    dog_celltype = pd.concat(
        dog_celltype_tables,
        ignore_index=True,
    )

    l1_scores = dog_celltype[
        dog_celltype["annotation_level"].eq("celltype.l1")
    ].rename(columns={"cell_type": "celltype.l1"})

    compartment_source = cell_scores.copy()
    compartment_source["broad_compartment"] = (
        compartment_source["immune_combined"]
    )
    compartment_scores = aggregate_scores(
        cells=compartment_source,
        group_columns=[DOG_COLUMN, "broad_compartment"],
        score_columns=score_columns,
        minimum_cells=MIN_CELLS_PER_DOG_COMPARTMENT,
    )

    targeted = targeted_tests(
        compartment_scores=compartment_scores,
        l1_scores=l1_scores,
    )
    celltype_summary = build_celltype_summary(
        dog_celltype=dog_celltype,
        score_columns=score_columns,
    )
    stability = rank_stability(
        dog_celltype=dog_celltype,
        score_columns=score_columns,
    )
    gene_localization = gene_celltype_localization(
        transformed_expression=transformed_expression,
        genes=genes,
        metadata=metadata,
        weights=weights,
    )

    np.savez_compressed(
        OUTPUT_SELECTED_EXPRESSION,
        genes=np.asarray(genes, dtype=object),
        cell_ids=np.asarray(expression_cells, dtype=object),
        rank_gaussian_expression=transformed_expression,
    )
    OUTPUT_SELECTED_EXPRESSION_METADATA.write_text(
        json.dumps(
            {
                "script_version": SCRIPT_VERSION,
                "gene_column": gene_column,
                "n_genes": len(genes),
                "n_cells": len(expression_cells),
                "duplicate_expression_rows": duplicate_rows,
                "expression_sha256": sha256_file(EXPRESSION_FILE),
                "metadata_sha256": sha256_file(METADATA_FILE),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cell_scores.to_csv(
        OUTPUT_CELL_SCORES,
        index=False,
        compression="gzip",
    )
    dog_celltype.to_csv(OUTPUT_DOG_CELLTYPE, index=False)
    celltype_summary.to_csv(OUTPUT_CELLTYPE_SUMMARY, index=False)
    compartment_scores.to_csv(OUTPUT_COMPARTMENT, index=False)
    targeted.to_csv(OUTPUT_TARGETED, index=False)
    gene_localization.to_csv(
        OUTPUT_GENE_LOCALIZATION,
        index=False,
    )
    stability.to_csv(OUTPUT_RANK_STABILITY, index=False)
    score_coverage.to_csv(OUTPUT_SCORE_COVERAGE, index=False)

    create_heatmap(celltype_summary)

    figure_paths = [HEATMAP_PNG, HEATMAP_PDF]
    figure_paths.extend(
        create_paired_plot(
            pseudobulk=compartment_scores,
            group_column="broad_compartment",
            group_a="osteoblast_lineage",
            group_b="immune_combined",
            score_column="M34_signed_risk_score_mean",
            title=(
                "M34 signed risk: osteoblast lineage versus immune"
            ),
            filename_stem=(
                "Ammons_M34_signed_osteoblast_vs_immune"
            ),
        )
    )
    figure_paths.extend(
        create_paired_plot(
            pseudobulk=compartment_scores,
            group_column="broad_compartment",
            group_a="immune_combined",
            group_b="osteoblast_lineage",
            score_column=(
                "M34_negative_component_expression_mean"
            ),
            title=(
                "M34 negative-loading expression: immune versus "
                "osteoblast lineage"
            ),
            filename_stem=(
                "Ammons_M34_negative_immune_vs_osteoblast"
            ),
        )
    )
    figure_paths.extend(
        create_paired_plot(
            pseudobulk=l1_scores,
            group_column="celltype.l1",
            group_a="Osteoblast_cycling",
            group_b="Osteoblast",
            score_column="M40_signed_risk_score_mean",
            title=(
                "M40 signed risk: cycling versus non-cycling osteoblast"
            ),
            filename_stem=(
                "Ammons_M40_signed_cycling_vs_osteoblast"
            ),
        )
    )
    figure_paths.extend(
        create_paired_plot(
            pseudobulk=l1_scores,
            group_column="celltype.l1",
            group_a="Osteoblast_cycling",
            group_b="Osteoblast",
            score_column=(
                "M40_positive_component_expression_mean"
            ),
            title=(
                "M40 positive component: cycling versus "
                "non-cycling osteoblast"
            ),
            filename_stem=(
                "Ammons_M40_positive_cycling_vs_osteoblast"
            ),
        )
    )

    write_readme(
        score_coverage=score_coverage,
        targeted=targeted,
    )

    input_paths = [
        METADATA_FILE,
        EXPRESSION_FILE,
        PREFLIGHT_MANIFEST_FILE,
        STRICT_WEIGHTS_FILE,
    ]
    output_paths = [
        OUTPUT_SELECTED_EXPRESSION,
        OUTPUT_SELECTED_EXPRESSION_METADATA,
        OUTPUT_CELL_SCORES,
        OUTPUT_DOG_CELLTYPE,
        OUTPUT_CELLTYPE_SUMMARY,
        OUTPUT_COMPARTMENT,
        OUTPUT_TARGETED,
        OUTPUT_GENE_LOCALIZATION,
        OUTPUT_RANK_STABILITY,
        OUTPUT_SCORE_COVERAGE,
        OUTPUT_README,
        *[path for path in figure_paths if path.exists()],
    ]

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "outcome_loaded": False,
        "biological_replicate": DOG_COLUMN,
        "cell_id_column": CELL_ID_COLUMN,
        "annotation_levels": ANNOTATION_LEVELS,
        "scoring": {
            "gene_transform": (
                "rank-Gaussian across all cells"
            ),
            "signed_score": (
                "sum(risk-oriented loading * transformed expression) "
                "/ sum(abs(loading))"
            ),
            "positive_component": (
                "positive-loading weighted mean"
            ),
            "negative_component": (
                "absolute negative-loading weighted mean expression"
            ),
        },
        "targeted_tests": targeted[
            [
                "contrast_name",
                "score_column",
                "n_paired_dogs",
                "mean_paired_difference",
                "exact_sign_flip_p",
                "primary_bh_q",
            ]
        ].to_dict(orient="records"),
        "guardrails": [
            "No clinical outcome or endpoint was loaded.",
            "Frozen program loadings and signs were not changed.",
            "Individual cells were not used as biological replicates.",
            "Primary inference used paired dog-level pseudobulk contrasts.",
            "M34 positive component is descriptive because only two positive-loading genes were detected.",
            "Single-cell localization does not establish causality or external outcome validation.",
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
    print("Score gene coverage")
    print("=" * 80)
    print(score_coverage.to_string(index=False))

    print("")
    print("=" * 80)
    print("Primary targeted single-cell localization tests")
    print("=" * 80)
    print(
        targeted[
            [
                "contrast_name",
                "n_paired_dogs",
                "mean_paired_difference",
                "median_paired_difference",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "exact_sign_flip_p",
                "primary_bh_q",
                "estimable",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Top cell types by dog-level median score")
    print("=" * 80)
    selected_summary = celltype_summary[
        celltype_summary["annotation_level"].eq("celltype.l1")
        & celltype_summary["score_column"].isin(
            [
                "M34_signed_risk_score",
                "M34_negative_component_expression",
                "M40_signed_risk_score",
                "M40_positive_component_expression",
            ]
        )
    ]
    top = (
        selected_summary.sort_values(
            ["score_column", "median_dog_score"],
            ascending=[True, False],
        )
        .groupby("score_column", as_index=False)
        .head(10)
    )
    print(
        top[
            [
                "score_column",
                "cell_type",
                "n_dogs",
                "total_cells",
                "median_dog_score",
                "iqr_low",
                "iqr_high",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Leave-one-dog-out rank stability")
    print("=" * 80)
    stable_top = (
        stability[
            stability["annotation_level"].eq("celltype.l1")
        ]
        .sort_values(
            ["score_column", "full_rank"],
        )
        .groupby("score_column", as_index=False)
        .head(10)
    )
    print(
        stable_top[
            [
                "score_column",
                "cell_type",
                "full_rank",
                "minimum_loo_rank",
                "maximum_loo_rank",
                "maximum_absolute_rank_change",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("No clinical outcome was loaded.")
    print("Dog, not cell, is the unit of primary inference.")
    print("M34 positive component is descriptive because it contains only two detected genes.")
    print("The M34 negative component represents expression of opposite-sign/protective genes.")
    print("Single-cell localization can support immune-depletion or proliferation localization, but cannot prove causality.")

    print("")
    print("Saved:")
    for path in [
        OUTPUT_SELECTED_EXPRESSION,
        OUTPUT_SELECTED_EXPRESSION_METADATA,
        OUTPUT_CELL_SCORES,
        OUTPUT_DOG_CELLTYPE,
        OUTPUT_CELLTYPE_SUMMARY,
        OUTPUT_COMPARTMENT,
        OUTPUT_TARGETED,
        OUTPUT_GENE_LOCALIZATION,
        OUTPUT_RANK_STABILITY,
        OUTPUT_SCORE_COVERAGE,
        OUTPUT_README,
        OUTPUT_MANIFEST,
        HEATMAP_PNG,
        HEATMAP_PDF,
    ]:
        print(path)
    print(f"Figures directory: {FIGURES_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
