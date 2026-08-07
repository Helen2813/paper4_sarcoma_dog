from __future__ import annotations

from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_VERSION = "44-recompute-ammons-six-dog-localization-v1"

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
    / "single_cell_localization_six_dogs"
)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

METADATA_FILE = RAW_DIR / "meta.tsv"
CELL_SCORES_FILE = (
    PROCESSED_DIR / "Ammons_frozen_program_cell_scores.csv.gz"
)
SCORE_COVERAGE_FILE = (
    RESULTS_DIR / "Ammons_scRNA_score_gene_coverage.csv"
)
ORIGINAL_TARGETED_FILE = (
    RESULTS_DIR / "Ammons_scRNA_targeted_localization_tests.csv"
)
SCRIPT42_MANIFEST_FILE = (
    RESULTS_DIR / "Ammons_scRNA_localization_manifest.json"
)
MULTIDIMENSIONAL_FILE = (
    RESULTS_DIR / "paper4_locked_multidimensional_transport_evidence.csv"
)

CELL_ID_COLUMN = "barcode"
LIBRARY_COLUMN = "orig.ident"
DOG_COLUMN = "name"

ANNOTATION_LEVELS = ["celltype.l1", "celltype.l2", "celltype.l3"]
MIN_CELLS_PER_DOG_CELLTYPE = 25
MIN_CELLS_PER_DOG_COMPARTMENT = 50
MIN_DOGS_PER_CELLTYPE = 3
MIN_DOGS_PER_CONTRAST = 4
N_BOOTSTRAP = 5000
RANDOM_SEED = 42

OUTPUT_MAPPING = (
    RESULTS_DIR / "Ammons_scRNA_confirmed_library_to_dog_mapping.csv"
)
OUTPUT_DOG_COUNTS = (
    RESULTS_DIR / "Ammons_scRNA_six_dog_cell_counts.csv"
)
OUTPUT_DOG_CELLTYPE = (
    RESULTS_DIR
    / "Ammons_scRNA_six_dog_celltype_pseudobulk_scores.csv"
)
OUTPUT_DOG_COMPARTMENT = (
    RESULTS_DIR / "Ammons_scRNA_six_dog_compartment_scores.csv"
)
OUTPUT_TARGETED = (
    RESULTS_DIR / "Ammons_scRNA_targeted_localization_tests_six_dogs.csv"
)
OUTPUT_COMPARISON = (
    RESULTS_DIR
    / "Ammons_scRNA_targeted_eight_sample_vs_six_dog_comparison.csv"
)
OUTPUT_CELLTYPE_SUMMARY = (
    RESULTS_DIR / "Ammons_scRNA_celltype_score_summary_six_dogs.csv"
)
OUTPUT_RANK_STABILITY = (
    RESULTS_DIR / "Ammons_scRNA_celltype_rank_stability_six_dogs.csv"
)
OUTPUT_LOCKED_SINGLE_CELL = (
    RESULTS_DIR
    / "paper4_locked_single_cell_biological_localization_six_dogs.csv"
)
OUTPUT_UPDATED_MASTER = (
    RESULTS_DIR
    / "paper4_locked_multidimensional_transport_evidence_with_single_cell_six_dogs.csv"
)
OUTPUT_SENTENCES = (
    RESULTS_DIR
    / "paper4_locked_single_cell_results_sentences_six_dogs.txt"
)
OUTPUT_README = (
    RESULTS_DIR / "Ammons_scRNA_six_dog_localization_README.txt"
)
OUTPUT_MANIFEST = (
    RESULTS_DIR / "Ammons_scRNA_six_dog_localization_manifest.json"
)

M34_PAIRED_PNG = (
    FIGURES_DIR / "Ammons_M34_signed_osteoblast_vs_immune_six_dogs.png"
)
M34_PAIRED_PDF = (
    FIGURES_DIR / "Ammons_M34_signed_osteoblast_vs_immune_six_dogs.pdf"
)
M40_PAIRED_PNG = (
    FIGURES_DIR / "Ammons_M40_signed_cycling_vs_osteoblast_six_dogs.png"
)
M40_PAIRED_PDF = (
    FIGURES_DIR / "Ammons_M40_signed_cycling_vs_osteoblast_six_dogs.pdf"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_required_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    print(f"Loaded: {path}")
    return pd.read_csv(path, **kwargs)


def read_required_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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


def exact_sign_flip_test(
    differences: np.ndarray,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan, np.nan

    observed = float(np.mean(values))
    statistics = []

    for signs in product([-1.0, 1.0], repeat=values.size):
        statistics.append(
            float(
                np.mean(
                    values * np.asarray(signs, dtype=float)
                )
            )
        )

    permutation_statistics = np.asarray(statistics, dtype=float)
    p = float(
        np.mean(
            np.abs(permutation_statistics)
            >= abs(observed) - 1e-12
        )
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


def estimable_score_columns(
    coverage: pd.DataFrame,
) -> list[str]:
    columns: list[str] = []

    for row in coverage.itertuples(index=False):
        module = str(row.module_label)

        if bool(row.signed_score_estimable):
            columns.append(f"{module}_signed_risk_score")
        if bool(row.positive_component_estimable):
            columns.append(
                f"{module}_positive_component_expression"
            )
        if bool(row.negative_component_estimable):
            columns.append(
                f"{module}_negative_component_expression"
            )

    return sorted(set(columns))


def validate_six_dog_mapping(
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [CELL_ID_COLUMN, LIBRARY_COLUMN, DOG_COLUMN]
    missing = [
        column for column in required if column not in metadata.columns
    ]
    if missing:
        raise ValueError(
            f"Required metadata columns are missing: {missing}"
        )

    mapping = (
        metadata[[LIBRARY_COLUMN, DOG_COLUMN]]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values([DOG_COLUMN, LIBRARY_COLUMN])
        .reset_index(drop=True)
    )

    if mapping[LIBRARY_COLUMN].nunique() != 8:
        raise RuntimeError(
            "Expected eight Cell Ranger library identifiers."
        )
    if mapping[DOG_COLUMN].nunique() != 6:
        raise RuntimeError(
            "Expected the metadata name column to define six dogs."
        )

    libraries_per_dog = (
        mapping.groupby(DOG_COLUMN)[LIBRARY_COLUMN]
        .nunique()
        .sort_values(ascending=False)
    )

    if sorted(libraries_per_dog.tolist()) != [1, 1, 1, 1, 2, 2]:
        raise RuntimeError(
            "Expected two dogs with two libraries and four dogs "
            "with one library."
        )

    counts = (
        metadata[DOG_COLUMN]
        .astype(str)
        .value_counts()
        .rename_axis("dog_id")
        .reset_index(name="n_cells")
        .sort_values("dog_id")
    )
    counts["fraction_cells"] = counts["n_cells"] / metadata.shape[0]
    counts["n_libraries"] = counts["dog_id"].map(libraries_per_dog)

    return mapping.rename(columns={DOG_COLUMN: "dog_id"}), counts


def merge_dog_ids(
    metadata: pd.DataFrame,
    cell_scores: pd.DataFrame,
) -> pd.DataFrame:
    dog_map = (
        metadata[
            [CELL_ID_COLUMN, LIBRARY_COLUMN, DOG_COLUMN]
        ]
        .dropna()
        .astype(str)
        .drop_duplicates(CELL_ID_COLUMN)
        .rename(columns={DOG_COLUMN: "dog_id"})
    )

    scores = cell_scores.copy()
    scores[CELL_ID_COLUMN] = scores[CELL_ID_COLUMN].astype(str)

    if "dog_id" in scores.columns:
        scores = scores.drop(columns=["dog_id"])

    merged = scores.merge(
        dog_map,
        on=CELL_ID_COLUMN,
        how="left",
        suffixes=("", "_metadata"),
        validate="one_to_one",
    )

    if merged["dog_id"].isna().any():
        raise RuntimeError(
            "Some scored cells could not be mapped to a biological dog."
        )

    if (
        LIBRARY_COLUMN in merged.columns
        and f"{LIBRARY_COLUMN}_metadata" in merged.columns
    ):
        mismatch = (
            merged[LIBRARY_COLUMN].astype(str)
            != merged[f"{LIBRARY_COLUMN}_metadata"].astype(str)
        )
        if mismatch.any():
            raise RuntimeError(
                "orig.ident mismatch between score table and metadata."
            )
        merged = merged.drop(
            columns=[f"{LIBRARY_COLUMN}_metadata"]
        )
    elif f"{LIBRARY_COLUMN}_metadata" in merged.columns:
        merged = merged.rename(
            columns={
                f"{LIBRARY_COLUMN}_metadata": LIBRARY_COLUMN
            }
        )

    return merged


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

    means.columns = [
        f"{column}_mean" for column in means.columns
    ]
    medians.columns = [
        f"{column}_median" for column in medians.columns
    ]

    result = pd.concat(
        [counts, means, medians],
        axis=1,
    ).reset_index()

    return result[
        result["n_cells"].ge(minimum_cells)
    ].reset_index(drop=True)


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
        ["dog_id", group_column, score_column, "n_cells"]
    ].copy()

    pivot = part.pivot_table(
        index="dog_id",
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

    n_positive = int(np.sum(differences > 0))
    n_negative = int(np.sum(differences < 0))
    n_zero = int(np.sum(differences == 0))

    return {
        "contrast_name": contrast_name,
        "score_column": score_column,
        "group_column": group_column,
        "group_a": group_a,
        "group_b": group_b,
        "difference_definition": f"{group_a} minus {group_b}",
        "hypothesis_direction": hypothesis_direction,
        "n_paired_dogs": int(differences.size),
        "n_positive_differences": n_positive,
        "n_negative_differences": n_negative,
        "n_zero_differences": n_zero,
        "all_same_nonzero_sign": bool(
            differences.size > 0
            and n_zero == 0
            and (
                n_positive == differences.size
                or n_negative == differences.size
            )
        ),
        "mean_paired_difference": mean_difference,
        "median_paired_difference": (
            float(np.median(differences))
            if differences.size
            else np.nan
        ),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "exact_sign_flip_p": p,
        "minimum_attainable_two_sided_p": (
            2.0 / (2.0 ** differences.size)
            if differences.size
            else np.nan
        ),
        "dog_level_differences": ";".join(
            f"{dog}:{difference:.8g}"
            for dog, difference in zip(
                pivot.index.astype(str),
                differences,
            )
        ),
        "estimable": bool(
            differences.size >= MIN_DOGS_PER_CONTRAST
        ),
    }


def targeted_tests(
    compartment_scores: pd.DataFrame,
    l1_scores: pd.DataFrame,
) -> pd.DataFrame:
    tests = [
        paired_contrast(
            pseudobulk=compartment_scores,
            group_column="analysis_compartment",
            group_a="osteoblast_lineage",
            group_b="immune_combined",
            score_column="M34_signed_risk_score_mean",
            contrast_name=(
                "M34 signed risk: osteoblast lineage versus immune"
            ),
            hypothesis_direction="Higher in osteoblast lineage",
        ),
        paired_contrast(
            pseudobulk=compartment_scores,
            group_column="analysis_compartment",
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
        ),
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
        ),
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
        ),
    ]

    result = pd.DataFrame(tests)
    result.loc[
        ~result["estimable"],
        "exact_sign_flip_p",
    ] = np.nan
    result["primary_bh_q"] = bh_adjust(
        result["exact_sign_flip_p"]
    )
    return result


def celltype_summary(
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
            n_dogs = int(cell_part["dog_id"].nunique())
            if n_dogs < MIN_DOGS_PER_CELLTYPE:
                continue

            for score_column in score_columns:
                values = pd.to_numeric(
                    cell_part[f"{score_column}_mean"],
                    errors="coerce",
                ).to_numpy(dtype=float)
                values = values[np.isfinite(values)]

                if values.size == 0:
                    continue

                rows.append(
                    {
                        "annotation_level": annotation_level,
                        "cell_type": cell_type,
                        "score_column": score_column,
                        "n_dogs": n_dogs,
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
            part.groupby("cell_type")["dog_id"]
            .nunique()
            .loc[lambda values: values >= MIN_DOGS_PER_CELLTYPE]
            .index.tolist()
        )
        part = part[part["cell_type"].isin(eligible_types)]

        dogs = sorted(part["dog_id"].dropna().unique())
        if len(dogs) < MIN_DOGS_PER_CELLTYPE:
            continue

        for score_column in score_columns:
            score_name = f"{score_column}_mean"

            full_means = (
                part.groupby("cell_type")[score_name]
                .mean()
                .sort_values(ascending=False)
            )
            full_ranks = pd.Series(
                np.arange(1, len(full_means) + 1),
                index=full_means.index,
            )

            records = []
            for left_out_dog in dogs:
                loo = part[part["dog_id"].ne(left_out_dog)]
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
                    records.append(
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

            record_table = pd.DataFrame(records)
            if record_table.empty:
                continue

            for cell_type, cell_part in record_table.groupby(
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


def compare_with_eight_sample(
    six_dog: pd.DataFrame,
    eight_sample: pd.DataFrame,
) -> pd.DataFrame:
    left = eight_sample[
        [
            "contrast_name",
            "n_paired_dogs",
            "mean_paired_difference",
            "median_paired_difference",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "exact_sign_flip_p",
            "primary_bh_q",
        ]
    ].rename(
        columns={
            "n_paired_dogs": "n_orig_ident_units",
            "mean_paired_difference": (
                "eight_sample_mean_difference"
            ),
            "median_paired_difference": (
                "eight_sample_median_difference"
            ),
            "bootstrap_ci_low": "eight_sample_ci_low",
            "bootstrap_ci_high": "eight_sample_ci_high",
            "exact_sign_flip_p": "eight_sample_exact_p",
            "primary_bh_q": "eight_sample_bh_q",
        }
    )

    right = six_dog[
        [
            "contrast_name",
            "n_paired_dogs",
            "mean_paired_difference",
            "median_paired_difference",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "exact_sign_flip_p",
            "primary_bh_q",
            "all_same_nonzero_sign",
        ]
    ].rename(
        columns={
            "n_paired_dogs": "n_biological_dogs",
            "mean_paired_difference": (
                "six_dog_mean_difference"
            ),
            "median_paired_difference": (
                "six_dog_median_difference"
            ),
            "bootstrap_ci_low": "six_dog_ci_low",
            "bootstrap_ci_high": "six_dog_ci_high",
            "exact_sign_flip_p": "six_dog_exact_p",
            "primary_bh_q": "six_dog_bh_q",
        }
    )

    comparison = left.merge(
        right,
        on="contrast_name",
        how="outer",
    )
    comparison["mean_effect_change"] = (
        comparison["six_dog_mean_difference"]
        - comparison["eight_sample_mean_difference"]
    )
    comparison["mean_effect_ratio"] = (
        comparison["six_dog_mean_difference"]
        / comparison["eight_sample_mean_difference"]
    )
    comparison["supersession_note"] = (
        "The six-dog result supersedes the eight-orig.ident "
        "analysis because dogs 1 and 2 each had two technical "
        "replicate libraries."
    )
    return comparison


def build_locked_single_cell(
    targeted: pd.DataFrame,
) -> pd.DataFrame:
    indexed = targeted.set_index("contrast_name")

    m34_signed_name = (
        "M34 signed risk: osteoblast lineage versus immune"
    )
    m34_negative_name = (
        "M34 negative-loading expression: immune versus "
        "osteoblast lineage"
    )
    m40_signed_name = (
        "M40 signed risk: cycling versus non-cycling osteoblast"
    )
    m40_positive_name = (
        "M40 positive-loading expression: cycling versus "
        "non-cycling osteoblast"
    )

    rows = [
        {
            "module_label": "M34",
            "single_cell_localization_class": (
                "immune_negative_component_with_"
                "osteoblast_high_signed_risk"
            ),
            "n_biological_dogs": 6,
            "primary_contrast_1": m34_signed_name,
            "primary_effect_1": indexed.loc[
                m34_signed_name,
                "mean_paired_difference",
            ],
            "primary_q_1": indexed.loc[
                m34_signed_name,
                "primary_bh_q",
            ],
            "primary_contrast_2": m34_negative_name,
            "primary_effect_2": indexed.loc[
                m34_negative_name,
                "mean_paired_difference",
            ],
            "primary_q_2": indexed.loc[
                m34_negative_name,
                "primary_bh_q",
            ],
            "locked_single_cell_interpretation": (
                "Across six biological dogs, the M34 signed "
                "risk-oriented score was consistently higher in "
                "osteoblast-lineage than immune compartments, while "
                "the negative-loading component was consistently "
                "higher in immune than osteoblast-lineage cells. "
                "Because 153 of 155 detected M34 genes have negative "
                "loadings, M34 is best interpreted as an inverse "
                "immune-lineage or immune-depletion/exclusion axis."
            ),
            "replicate_guardrail": (
                "Dogs 1 and 2 each had two technical replicate "
                "libraries, which were combined by metadata name "
                "before dog-level pseudobulk inference."
            ),
        },
        {
            "module_label": "M40",
            "single_cell_localization_class": (
                "pan_cycling_program_with_"
                "cycling_osteoblast_enrichment"
            ),
            "n_biological_dogs": 6,
            "primary_contrast_1": m40_signed_name,
            "primary_effect_1": indexed.loc[
                m40_signed_name,
                "mean_paired_difference",
            ],
            "primary_q_1": indexed.loc[
                m40_signed_name,
                "primary_bh_q",
            ],
            "primary_contrast_2": m40_positive_name,
            "primary_effect_2": indexed.loc[
                m40_positive_name,
                "mean_paired_difference",
            ],
            "primary_q_2": indexed.loc[
                m40_positive_name,
                "primary_bh_q",
            ],
            "locked_single_cell_interpretation": (
                "Across six biological dogs, M40 signed and "
                "positive-loading scores were consistently higher in "
                "cycling than non-cycling osteoblasts. Together with "
                "high scores in cycling T-cell and osteoclast "
                "populations, M40 is a broad cycling/proliferation "
                "axis with clear tumor-lineage enrichment rather than "
                "an osteoblast-specific program."
            ),
            "replicate_guardrail": (
                "Dogs 1 and 2 each had two technical replicate "
                "libraries, which were combined by metadata name "
                "before dog-level pseudobulk inference."
            ),
        },
        {
            "module_label": "M11",
            "single_cell_localization_class": (
                "secondary_positive_component_only"
            ),
            "n_biological_dogs": 6,
            "primary_contrast_1": "",
            "primary_effect_1": np.nan,
            "primary_q_1": np.nan,
            "primary_contrast_2": "",
            "primary_effect_2": np.nan,
            "primary_q_2": np.nan,
            "locked_single_cell_interpretation": (
                "M11 remains a secondary positive-component-only "
                "localization and does not alter its locked "
                "cross-species evidence grade."
            ),
            "replicate_guardrail": (
                "No primary single-cell localization test was "
                "specified for M11."
            ),
        },
        {
            "module_label": "M24",
            "single_cell_localization_class": (
                "secondary_positive_component_only"
            ),
            "n_biological_dogs": 6,
            "primary_contrast_1": "",
            "primary_effect_1": np.nan,
            "primary_q_1": np.nan,
            "primary_contrast_2": "",
            "primary_effect_2": np.nan,
            "primary_q_2": np.nan,
            "locked_single_cell_interpretation": (
                "M24 remains a secondary positive-component-only "
                "localization and cannot compensate for limited "
                "cross-species representation and outcome evidence."
            ),
            "replicate_guardrail": (
                "No primary single-cell localization test was "
                "specified for M24."
            ),
        },
    ]
    return pd.DataFrame(rows)


def paired_plot(
    pseudobulk: pd.DataFrame,
    group_column: str,
    group_a: str,
    group_b: str,
    score_column: str,
    title: str,
    png_path: Path,
    pdf_path: Path,
) -> None:
    pivot = (
        pseudobulk[
            pseudobulk[group_column].isin([group_a, group_b])
        ]
        .pivot_table(
            index="dog_id",
            columns=group_column,
            values=score_column,
            aggfunc="mean",
        )
        .dropna(subset=[group_a, group_b])
    )

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(5.5, 4.7))

    for dog_id, row in pivot.iterrows():
        ax.plot(
            [0, 1],
            [row[group_a], row[group_b]],
            marker="o",
        )
        ax.text(
            1.03,
            row[group_b],
            str(dog_id),
            va="center",
            fontsize=8,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels([group_a, group_b], rotation=15)
    ax.set_ylabel(score_column)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def write_sentences(
    locked: pd.DataFrame,
    targeted: pd.DataFrame,
) -> None:
    indexed = locked.set_index("module_label")

    all_floor = bool(
        targeted["all_same_nonzero_sign"].all()
        and np.allclose(
            targeted["exact_sign_flip_p"].to_numpy(dtype=float),
            0.03125,
        )
    )

    lines = [
        "Locked six-dog single-cell localization results",
        "================================================",
        "",
        "Replicate correction",
        "--------------------",
        (
            "The Cell Browser dataset contained eight Cell Ranger "
            "libraries but six biological dogs. Dogs 1 and 2 each "
            "contributed two technical replicate libraries, which "
            "were combined using the metadata name column before "
            "biological inference."
        ),
        "",
        "M34",
        "---",
        indexed.loc[
            "M34",
            "locked_single_cell_interpretation",
        ],
        "",
        "M40",
        "---",
        indexed.loc[
            "M40",
            "locked_single_cell_interpretation",
        ],
        "",
        "Statistical interpretation",
        "--------------------------",
        (
            "All four prespecified contrasts retained the same "
            "direction in all six biological dogs. The exact "
            "two-sided sign-flip P value was 0.03125, the minimum "
            "attainable value for six paired dogs."
            if all_floor
            else
            "Dog-level exact paired results are reported in "
            "Ammons_scRNA_targeted_localization_tests_six_dogs.csv."
        ),
        "",
        "Supersession rule",
        "-----------------",
        (
            "The six-dog analysis supersedes the earlier "
            "eight-orig.ident P=0.0078125 analysis."
        ),
    ]

    OUTPUT_SENTENCES.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_readme(
    mapping: pd.DataFrame,
    targeted: pd.DataFrame,
) -> None:
    text = f"""Ammons six-dog single-cell localization
Script version: {SCRIPT_VERSION}

Why this analysis is required
-----------------------------
The official study generated eight raw scRNA-seq samples from six dogs.
Dogs 1 and 2 each had two technical replicate libraries. The Cell Browser
metadata reflects this structure:
- 8 orig.ident library identifiers
- 6 name values

This script combines technical replicate libraries by name before any
biological inference. The earlier eight-orig.ident targeted tests are retained
only for audit comparison and are superseded by this six-dog analysis.

Statistical implication
-----------------------
For six paired biological dogs, the minimum attainable two-sided exact sign-flip
P value is 2 / 2^6 = 0.03125.

Mapping
-------
{mapping.to_string(index=False)}

Corrected targeted results
--------------------------
{targeted.to_string(index=False)}

Guardrails
----------
- Dog is the biological replicate.
- Individual cells and technical libraries are not independent replicates.
- The four targeted contrasts are biologically related and are not four
  independent replications.
- Single-cell localization supports biological interpretation, not causal or
  external prognostic validation.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("Recompute Ammons localization using six biological dogs")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Design:")
    print("  Collapse eight Cell Ranger libraries to six dogs using metadata name.")
    print("  Combine technical replicate cells before pseudobulk aggregation.")
    print("  Recalculate all four prespecified paired contrasts.")
    print("  Replace eight-orig.ident P values with six-dog exact inference.")
    print("")

    metadata = read_required_csv(
        METADATA_FILE,
        sep="\t",
        dtype=str,
        low_memory=False,
    )
    cell_scores = read_required_csv(
        CELL_SCORES_FILE,
        compression="gzip",
        low_memory=False,
    )
    coverage = read_required_csv(SCORE_COVERAGE_FILE)
    original_targeted = read_required_csv(
        ORIGINAL_TARGETED_FILE
    )
    script42_manifest = read_required_json(
        SCRIPT42_MANIFEST_FILE
    )

    if bool(script42_manifest.get("outcome_loaded", True)):
        raise RuntimeError(
            "Script 42 manifest does not confirm outcome_loaded=false."
        )

    mapping, dog_counts = validate_six_dog_mapping(metadata)
    cells = merge_dog_ids(metadata, cell_scores)

    if cells["dog_id"].nunique() != 6:
        raise RuntimeError(
            "Merged cell-score table does not contain six dogs."
        )

    score_columns = estimable_score_columns(coverage)
    missing_scores = [
        column
        for column in score_columns
        if column not in cells.columns
    ]
    if missing_scores:
        raise ValueError(
            f"Expected score columns are missing: {missing_scores}"
        )

    dog_celltype_tables = []
    for annotation_level in ANNOTATION_LEVELS:
        aggregated = aggregate_scores(
            cells=cells,
            group_columns=["dog_id", annotation_level],
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

    compartment_cells = cells.copy()
    if "immune_combined" not in compartment_cells.columns:
        raise ValueError(
            "The saved cell-score table is missing immune_combined."
        )
    compartment_cells["analysis_compartment"] = (
        compartment_cells["immune_combined"]
    )

    compartment_scores = aggregate_scores(
        cells=compartment_cells,
        group_columns=["dog_id", "analysis_compartment"],
        score_columns=score_columns,
        minimum_cells=MIN_CELLS_PER_DOG_COMPARTMENT,
    )

    targeted = targeted_tests(
        compartment_scores=compartment_scores,
        l1_scores=l1_scores,
    )
    comparison = compare_with_eight_sample(
        six_dog=targeted,
        eight_sample=original_targeted,
    )
    summary = celltype_summary(
        dog_celltype=dog_celltype,
        score_columns=score_columns,
    )
    stability = rank_stability(
        dog_celltype=dog_celltype,
        score_columns=score_columns,
    )
    locked = build_locked_single_cell(targeted)

    mapping.to_csv(OUTPUT_MAPPING, index=False)
    dog_counts.to_csv(OUTPUT_DOG_COUNTS, index=False)
    dog_celltype.to_csv(OUTPUT_DOG_CELLTYPE, index=False)
    compartment_scores.to_csv(
        OUTPUT_DOG_COMPARTMENT,
        index=False,
    )
    targeted.to_csv(OUTPUT_TARGETED, index=False)
    comparison.to_csv(OUTPUT_COMPARISON, index=False)
    summary.to_csv(OUTPUT_CELLTYPE_SUMMARY, index=False)
    stability.to_csv(OUTPUT_RANK_STABILITY, index=False)
    locked.to_csv(OUTPUT_LOCKED_SINGLE_CELL, index=False)

    if MULTIDIMENSIONAL_FILE.exists():
        multidimensional = pd.read_csv(
            MULTIDIMENSIONAL_FILE
        )
        updated = multidimensional.merge(
            locked[
                [
                    "module_label",
                    "single_cell_localization_class",
                    "n_biological_dogs",
                    "locked_single_cell_interpretation",
                    "replicate_guardrail",
                ]
            ],
            on="module_label",
            how="left",
        )
        updated.to_csv(
            OUTPUT_UPDATED_MASTER,
            index=False,
        )
    else:
        updated = pd.DataFrame()

    paired_plot(
        pseudobulk=compartment_scores,
        group_column="analysis_compartment",
        group_a="osteoblast_lineage",
        group_b="immune_combined",
        score_column="M34_signed_risk_score_mean",
        title=(
            "M34 signed risk: osteoblast lineage versus immune "
            "(six dogs)"
        ),
        png_path=M34_PAIRED_PNG,
        pdf_path=M34_PAIRED_PDF,
    )
    paired_plot(
        pseudobulk=l1_scores,
        group_column="celltype.l1",
        group_a="Osteoblast_cycling",
        group_b="Osteoblast",
        score_column="M40_signed_risk_score_mean",
        title=(
            "M40 signed risk: cycling versus non-cycling "
            "osteoblast (six dogs)"
        ),
        png_path=M40_PAIRED_PNG,
        pdf_path=M40_PAIRED_PDF,
    )

    write_sentences(locked, targeted)
    write_readme(mapping, targeted)

    input_paths = [
        METADATA_FILE,
        CELL_SCORES_FILE,
        SCORE_COVERAGE_FILE,
        ORIGINAL_TARGETED_FILE,
        SCRIPT42_MANIFEST_FILE,
    ]
    if MULTIDIMENSIONAL_FILE.exists():
        input_paths.append(MULTIDIMENSIONAL_FILE)

    output_paths = [
        OUTPUT_MAPPING,
        OUTPUT_DOG_COUNTS,
        OUTPUT_DOG_CELLTYPE,
        OUTPUT_DOG_COMPARTMENT,
        OUTPUT_TARGETED,
        OUTPUT_COMPARISON,
        OUTPUT_CELLTYPE_SUMMARY,
        OUTPUT_RANK_STABILITY,
        OUTPUT_LOCKED_SINGLE_CELL,
        OUTPUT_SENTENCES,
        OUTPUT_README,
        M34_PAIRED_PNG,
        M34_PAIRED_PDF,
        M40_PAIRED_PNG,
        M40_PAIRED_PDF,
    ]
    if not updated.empty:
        output_paths.append(OUTPUT_UPDATED_MASTER)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "outcome_loaded": False,
        "biological_replicate_column": DOG_COLUMN,
        "n_cell_ranger_libraries": 8,
        "n_biological_dogs": 6,
        "technical_replicate_structure": {
            "library_counts_per_dog": (
                mapping.groupby("dog_id")[LIBRARY_COLUMN]
                .nunique()
                .to_dict()
            ),
        },
        "supersedes": [
            "Ammons_scRNA_targeted_localization_tests.csv",
            "paper4_locked_single_cell_biological_localization.csv",
        ],
        "guardrails": [
            "Dogs 1 and 2 had two technical replicate libraries.",
            "Technical replicate cells were combined before dog-level pseudobulk inference.",
            "The six-dog analysis supersedes the eight-orig.ident exact tests.",
            "No clinical outcome or endpoint was loaded.",
            "Single-cell localization is biological annotation, not prognostic validation.",
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

    print("=" * 80)
    print("Confirmed library-to-dog mapping")
    print("=" * 80)
    print(mapping.to_string(index=False))

    print("")
    print("=" * 80)
    print("Corrected six-dog targeted tests")
    print("=" * 80)
    print(
        targeted[
            [
                "contrast_name",
                "n_paired_dogs",
                "n_positive_differences",
                "n_negative_differences",
                "mean_paired_difference",
                "median_paired_difference",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "exact_sign_flip_p",
                "minimum_attainable_two_sided_p",
                "primary_bh_q",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Eight-sample versus six-dog comparison")
    print("=" * 80)
    print(
        comparison[
            [
                "contrast_name",
                "eight_sample_mean_difference",
                "six_dog_mean_difference",
                "mean_effect_change",
                "eight_sample_exact_p",
                "six_dog_exact_p",
                "six_dog_bh_q",
                "all_same_nonzero_sign",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Corrected locked single-cell localization")
    print("=" * 80)
    print(
        locked[
            [
                "module_label",
                "single_cell_localization_class",
                "n_biological_dogs",
                "primary_effect_1",
                "primary_q_1",
                "primary_effect_2",
                "primary_q_2",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("The metadata name column defines six biological dogs.")
    print("Dogs 1 and 2 each contributed two technical replicate libraries.")
    print("The earlier eight-orig.ident P=0.0078125 results are superseded.")
    print("For six paired dogs, the exact two-sided floor is P=0.03125.")
    print("Biological interpretation remains valid only if dog-level directions remain concordant.")

    print("")
    print("Saved:")
    for path in [
        OUTPUT_MAPPING,
        OUTPUT_DOG_COUNTS,
        OUTPUT_DOG_CELLTYPE,
        OUTPUT_DOG_COMPARTMENT,
        OUTPUT_TARGETED,
        OUTPUT_COMPARISON,
        OUTPUT_CELLTYPE_SUMMARY,
        OUTPUT_RANK_STABILITY,
        OUTPUT_LOCKED_SINGLE_CELL,
        OUTPUT_UPDATED_MASTER,
        OUTPUT_SENTENCES,
        OUTPUT_README,
        OUTPUT_MANIFEST,
        M34_PAIRED_PNG,
        M34_PAIRED_PDF,
        M40_PAIRED_PNG,
        M40_PAIRED_PDF,
    ]:
        if path.exists():
            print(path)
    print("Done.")


if __name__ == "__main__":
    main()
