from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = (
    "04c-prestandardization-feature-variance-audit-v1"
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

INPUT_DIR = (
    PAPER5_ROOT
    / "model_inputs"
    / "TARGET_OS_primary78"
)

HARMONIZED_DIR = (
    INPUT_DIR
    / "harmonized"
)

OUTPUT_DIR = (
    INPUT_DIR
    / "premodel_qc"
)

FEATURE_SET_DIR = (
    OUTPUT_DIR
    / "feature_sets"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FEATURE_SET_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CASE_ORDER_FILE = (
    INPUT_DIR
    / "TARGET_OS_primary_multimodal_case_order.csv"
)

HARMONIZED_LOCK_FILE = (
    HARMONIZED_DIR
    / "harmonized_input_lock.json"
)


OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "prestandardization_variance_summary.csv"
)

OUTPUT_RECONSTRUCTION = (
    OUTPUT_DIR
    / "harmonized_reconstruction_audit.csv"
)

OUTPUT_FEATURE_TYPE_SUMMARY = (
    OUTPUT_DIR
    / "prestandardization_feature_type_summary.csv"
)

OUTPUT_BUDGET_PLAN = (
    OUTPUT_DIR
    / "representation_feature_budget_plan.csv"
)

OUTPUT_FOLD_CONTRACT = (
    OUTPUT_DIR
    / "fold_local_preprocessing_contract.json"
)

OUTPUT_MANIFEST = (
    OUTPUT_DIR
    / "prestandardization_variance_audit_manifest.json"
)

OUTPUT_README = (
    OUTPUT_DIR
    / "prestandardization_variance_audit_README.txt"
)


EXPECTED_CASES = 78
DEFAULT_CHUNK_SIZE = 25_000

METHYLATION_EPSILON = np.float32(1e-6)
RECONSTRUCTION_TOLERANCE = 5e-5


# ---------------------------------------------------------------------
# Modality specification
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ModalitySpec:
    key: str

    raw_matrix: Path
    raw_features: Path
    raw_feature_column: str

    harmonized_matrix: Path
    harmonized_features: Path
    harmonized_feature_column: str

    native_scale: str
    model_scale: str
    transform_name: str

    maximum_missing_fraction: float
    require_cg_probe_for_primary_models: bool

    low_budget: int
    primary_budget: int
    high_budget: int


MODALITIES = [
    ModalitySpec(
        key="RNA",
        raw_matrix=(
            INPUT_DIR
            / "TARGET_OS_primary_rna_log2tpm.npy"
        ),
        raw_features=(
            INPUT_DIR
            / "TARGET_OS_primary_rna_features.csv"
        ),
        raw_feature_column="feature",
        harmonized_matrix=(
            HARMONIZED_DIR
            / "RNA_zscore.npy"
        ),
        harmonized_features=(
            HARMONIZED_DIR
            / "RNA_features.csv"
        ),
        harmonized_feature_column="feature",
        native_scale="log2_TPM",
        model_scale="log2_TPM_before_zscore",
        transform_name="identity",
        maximum_missing_fraction=0.0,
        require_cg_probe_for_primary_models=False,
        low_budget=2_000,
        primary_budget=5_000,
        high_budget=10_000,
    ),
    ModalitySpec(
        key="METHYLATION",
        raw_matrix=(
            INPUT_DIR
            / "TARGET_OS_primary_methylation_beta_corrected.npy"
        ),
        raw_features=(
            INPUT_DIR
            / "TARGET_OS_primary_methylation_features_corrected.csv"
        ),
        raw_feature_column="probe_id",
        harmonized_matrix=(
            HARMONIZED_DIR
            / "METHYLATION_Mvalue_zscore.npy"
        ),
        harmonized_features=(
            HARMONIZED_DIR
            / "METHYLATION_features.csv"
        ),
        harmonized_feature_column="probe_id",
        native_scale="beta_value",
        model_scale="M_value_before_zscore",
        transform_name="beta_to_M_value",
        maximum_missing_fraction=0.20,
        require_cg_probe_for_primary_models=True,
        low_budget=5_000,
        primary_budget=10_000,
        high_budget=20_000,
    ),
    ModalitySpec(
        key="CNV",
        raw_matrix=(
            INPUT_DIR
            / "TARGET_OS_primary_cnv_gene_copy_number.npy"
        ),
        raw_features=(
            INPUT_DIR
            / "TARGET_OS_primary_cnv_features.csv"
        ),
        raw_feature_column="feature",
        harmonized_matrix=(
            HARMONIZED_DIR
            / "CNV_log2ratio_zscore.npy"
        ),
        harmonized_features=(
            HARMONIZED_DIR
            / "CNV_features.csv"
        ),
        harmonized_feature_column="feature",
        native_scale="gene_level_copy_number",
        model_scale="log2_copy_number_over_2_before_zscore",
        transform_name="log2_copy_number_over_2",
        maximum_missing_fraction=0.0,
        require_cg_probe_for_primary_models=False,
        low_budget=2_000,
        primary_budget=5_000,
        high_budget=10_000,
    ),
]


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "null",
        "na",
    }:
        return ""

    return text


def relative_path(path: Path) -> str:
    try:
        return str(
            path.resolve().relative_to(
                PROJECT_ROOT.resolve()
            )
        )
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                4 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def hash_strings(values: list[str]) -> str:
    digest = hashlib.sha256()

    for value in values:
        digest.update(
            str(value).encode("utf-8")
        )
        digest.update(b"\n")

    return digest.hexdigest()


def write_csv_gzip(
    table: pd.DataFrame,
    path: Path,
) -> None:
    compression = {
        "method": "gzip",
        "compresslevel": 6,
        "mtime": 0,
    }

    try:
        table.to_csv(
            path,
            index=False,
            float_format="%.12g",
            compression=compression,
        )
    except TypeError:
        # Compatibility fallback for older pandas releases.
        table.to_csv(
            path,
            index=False,
            float_format="%.12g",
            compression="gzip",
        )


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def load_feature_names(
    path: Path,
    column: str,
) -> list[str]:
    require_file(path)

    table = pd.read_csv(
        path,
        dtype=str,
    ).fillna("")

    if column not in table.columns:
        raise ValueError(
            f"{path.name} does not contain "
            f"feature column '{column}'. "
            f"Columns: {table.columns.tolist()}"
        )

    features = (
        table[column]
        .astype(str)
        .str.strip()
        .tolist()
    )

    if any(not feature for feature in features):
        raise RuntimeError(
            f"Empty feature identifiers found in {path}"
        )

    if len(features) != len(set(features)):
        duplicates = (
            pd.Series(features)
            .value_counts()
            .loc[lambda values: values > 1]
            .head(20)
            .index
            .tolist()
        )

        raise RuntimeError(
            f"Duplicate feature identifiers found in "
            f"{path}: {duplicates}"
        )

    return features


# ---------------------------------------------------------------------
# Input lock validation
# ---------------------------------------------------------------------

def load_case_order() -> list[str]:
    require_file(CASE_ORDER_FILE)

    table = pd.read_csv(
        CASE_ORDER_FILE,
        dtype=str,
    ).fillna("")

    if "case_id" not in table.columns:
        raise ValueError(
            "The case-order file does not contain case_id."
        )

    case_ids = (
        table["case_id"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    if len(case_ids) != EXPECTED_CASES:
        raise RuntimeError(
            f"Expected {EXPECTED_CASES} cases, "
            f"found {len(case_ids)}."
        )

    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError(
            "Duplicate case IDs detected in the locked order."
        )

    return case_ids


def load_and_validate_harmonized_lock(
    case_ids: list[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    require_file(HARMONIZED_LOCK_FILE)

    lock = json.loads(
        HARMONIZED_LOCK_FILE.read_text(
            encoding="utf-8"
        )
    )

    if bool(lock.get("outcome_loaded", True)):
        raise RuntimeError(
            "The harmonized lock does not confirm "
            "outcome_loaded=false."
        )

    if bool(lock.get("outcome_guided_processing", True)):
        raise RuntimeError(
            "The harmonized lock does not confirm "
            "outcome_guided_processing=false."
        )

    if bool(lock.get("model_fitting", True)):
        raise RuntimeError(
            "The harmonized lock does not confirm "
            "model_fitting=false."
        )

    observed_case_hash = hash_strings(
        case_ids
    )

    expected_case_hash = clean_text(
        lock.get(
            "case_order",
            {},
        ).get(
            "case_hash",
            "",
        )
    )

    if observed_case_hash != expected_case_hash:
        raise RuntimeError(
            "The current case order does not match "
            "the 04b harmonized lock."
        )

    observed_hashes: dict[str, str] = {}

    for spec in MODALITIES:
        require_file(
            spec.harmonized_matrix
        )

        expected_matrix_hash = clean_text(
            lock.get(
                "modalities",
                {},
            ).get(
                spec.key,
                {},
            ).get(
                "matrix_sha256",
                "",
            )
        )

        if not expected_matrix_hash:
            raise RuntimeError(
                f"No locked matrix hash was found "
                f"for {spec.key}."
            )

        print(
            f"Verifying locked matrix hash: {spec.key}"
        )

        observed_matrix_hash = sha256_file(
            spec.harmonized_matrix
        )

        observed_hashes[
            spec.key
        ] = observed_matrix_hash

        if (
            observed_matrix_hash
            != expected_matrix_hash
        ):
            raise RuntimeError(
                f"{spec.key} harmonized matrix hash "
                "does not match the 04b lock."
            )

    return lock, observed_hashes


# ---------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------

def transform_to_model_space(
    spec: ModalitySpec,
    native_values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        native_values,
        dtype=np.float32,
    )

    if spec.transform_name == "identity":
        return values

    if spec.transform_name == "beta_to_M_value":
        clipped = np.clip(
            values,
            METHYLATION_EPSILON,
            np.float32(1.0)
            - METHYLATION_EPSILON,
        )

        transformed = np.log2(
            clipped
            / (
                np.float32(1.0)
                - clipped
            )
        )

        return transformed.astype(
            np.float32,
            copy=False,
        )

    if (
        spec.transform_name
        == "log2_copy_number_over_2"
    ):
        finite = np.isfinite(values)

        if (
            finite.any()
            and np.min(
                values[finite]
            ) < 0
        ):
            raise RuntimeError(
                "Negative gene-level copy-number values "
                "were found before log transformation."
            )

        transformed = np.log2(
            np.clip(
                values,
                np.float32(1e-6),
                None,
            )
            / np.float32(2.0)
        )

        return transformed.astype(
            np.float32,
            copy=False,
        )

    raise ValueError(
        f"Unsupported transform: "
        f"{spec.transform_name}"
    )


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def safe_feature_statistics(
    values: np.ndarray,
) -> dict[str, np.ndarray]:
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    finite = np.isfinite(x)
    n_finite = finite.sum(axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=RuntimeWarning,
        )

        mean = np.nanmean(
            x,
            axis=0,
        )

        variance = np.nanvar(
            x,
            axis=0,
            ddof=1,
        )

        median = np.nanmedian(
            x,
            axis=0,
        )

        quartiles = np.nanpercentile(
            x,
            [
                25.0,
                75.0,
            ],
            axis=0,
        )

        minimum = np.nanmin(
            x,
            axis=0,
        )

        maximum = np.nanmax(
            x,
            axis=0,
        )

        absolute_deviation = np.abs(
            x
            - median[
                np.newaxis,
                :
            ]
        )

        mad = np.nanmedian(
            absolute_deviation,
            axis=0,
        )

    q25 = quartiles[0]
    q75 = quartiles[1]

    missing_fraction = (
        1.0
        - (
            n_finite
            / x.shape[0]
        )
    )

    zero_count = (
        finite
        & np.isclose(
            x,
            0.0,
            rtol=0.0,
            atol=1e-12,
        )
    ).sum(axis=0)

    zero_fraction = np.divide(
        zero_count,
        n_finite,
        out=np.full(
            n_finite.shape,
            np.nan,
            dtype=float,
        ),
        where=n_finite > 0,
    )

    return {
        "n_finite": (
            n_finite.astype(int)
        ),
        "missing_fraction": (
            missing_fraction
        ),
        "mean": mean,
        "variance": variance,
        "median": median,
        "mad": mad,
        "q25": q25,
        "q75": q75,
        "iqr": (
            q75 - q25
        ),
        "min": minimum,
        "max": maximum,
        "zero_fraction": (
            zero_fraction
        ),
    }


def feature_type_for_names(
    spec: ModalitySpec,
    features: list[str],
) -> list[str]:
    if spec.key != "METHYLATION":
        return [
            "gene"
            for _ in features
        ]

    output: list[str] = []

    for feature in features:
        lower = feature.lower()

        if lower.startswith("cg"):
            output.append("cg")
        elif lower.startswith("ch"):
            output.append("ch")
        elif lower.startswith("rs"):
            output.append("rs")
        else:
            output.append("other")

    return output


def assign_deterministic_rank(
    table: pd.DataFrame,
    metric_column: str,
    rank_column: str,
) -> None:
    table[rank_column] = pd.Series(
        pd.array(
            [
                pd.NA
                for _ in range(
                    table.shape[0]
                )
            ],
            dtype="Int64",
        ),
        index=table.index,
    )

    eligible = table[
        table[
            "primary_model_eligible"
        ].eq(True)
        & np.isfinite(
            table[
                metric_column
            ]
        )
    ].copy()

    eligible = eligible.sort_values(
        [
            metric_column,
            "feature_index",
        ],
        ascending=[
            False,
            True,
        ],
        kind="mergesort",
    )

    ranks = pd.Series(
        np.arange(
            1,
            eligible.shape[0] + 1,
            dtype=int,
        ),
        index=eligible.index,
    )

    table.loc[
        eligible.index,
        rank_column,
    ] = ranks

    table[rank_column] = (
        table[rank_column]
        .astype("Int64")
    )


# ---------------------------------------------------------------------
# Modality processing
# ---------------------------------------------------------------------

def process_modality(
    spec: ModalitySpec,
    case_ids: list[str],
    chunk_size: int,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    list[dict[str, Any]],
    list[Path],
]:
    print()
    print("-" * 88)
    print(
        f"Processing {spec.key}"
    )
    print("-" * 88)

    for path in [
        spec.raw_matrix,
        spec.raw_features,
        spec.harmonized_matrix,
        spec.harmonized_features,
    ]:
        require_file(path)

    raw_features = load_feature_names(
        spec.raw_features,
        spec.raw_feature_column,
    )

    retained_features = load_feature_names(
        spec.harmonized_features,
        spec.harmonized_feature_column,
    )

    raw_matrix = np.load(
        spec.raw_matrix,
        mmap_mode="r",
    )

    harmonized_matrix = np.load(
        spec.harmonized_matrix,
        mmap_mode="r",
    )

    if raw_matrix.ndim != 2:
        raise RuntimeError(
            f"{spec.key} raw matrix is not two-dimensional."
        )

    if harmonized_matrix.ndim != 2:
        raise RuntimeError(
            f"{spec.key} harmonized matrix "
            "is not two-dimensional."
        )

    if raw_matrix.shape[0] != len(case_ids):
        raise RuntimeError(
            f"{spec.key} raw sample count mismatch: "
            f"{raw_matrix.shape[0]} vs {len(case_ids)}."
        )

    if (
        harmonized_matrix.shape[0]
        != len(case_ids)
    ):
        raise RuntimeError(
            f"{spec.key} harmonized sample count mismatch: "
            f"{harmonized_matrix.shape[0]} "
            f"vs {len(case_ids)}."
        )

    if raw_matrix.shape[1] != len(raw_features):
        raise RuntimeError(
            f"{spec.key} raw feature-count mismatch."
        )

    if (
        harmonized_matrix.shape[1]
        != len(retained_features)
    ):
        raise RuntimeError(
            f"{spec.key} harmonized feature-count mismatch."
        )

    raw_feature_index = pd.Index(
        raw_features,
        dtype=str,
    )

    retained_to_raw = raw_feature_index.get_indexer(
        retained_features
    )

    missing_from_raw = np.flatnonzero(
        retained_to_raw < 0
    )

    if missing_from_raw.size > 0:
        examples = [
            retained_features[index]
            for index in missing_from_raw[:20]
        ]

        raise RuntimeError(
            f"{spec.key} harmonized features not found "
            f"in the raw feature universe: {examples}"
        )

    print(
        f"Raw features: "
        f"{len(raw_features):,}"
    )
    print(
        f"Harmonized retained features: "
        f"{len(retained_features):,}"
    )
    print(
        f"Native ranking scale: "
        f"{spec.native_scale}"
    )
    print(
        f"Model pre-z-score scale: "
        f"{spec.model_scale}"
    )

    ranking_parts: list[pd.DataFrame] = []

    reconstruction_max_abs_difference = 0.0
    reconstruction_sum_abs_difference = 0.0
    reconstruction_compared_values = 0
    reconstruction_finite_mask_mismatches = 0

    n_features = len(
        retained_features
    )

    for start in range(
        0,
        n_features,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            n_features,
        )

        retained_slice = slice(
            start,
            end,
        )

        raw_columns = retained_to_raw[
            start:end
        ]

        native_values = np.asarray(
            raw_matrix[
                :,
                raw_columns,
            ],
            dtype=np.float32,
        )

        model_values = transform_to_model_space(
            spec,
            native_values,
        )

        expected_zscores = np.asarray(
            harmonized_matrix[
                :,
                retained_slice,
            ],
            dtype=np.float32,
        )

        # Reproduce the exact standardization used in script 04.
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                category=RuntimeWarning,
            )

            model_mean = np.nanmean(
                model_values,
                axis=0,
            )

            model_std = np.nanstd(
                model_values,
                axis=0,
                ddof=1,
            )

        model_std[
            (~np.isfinite(model_std))
            | (model_std == 0)
        ] = np.float32(1.0)

        reconstructed_zscores = (
            model_values
            - model_mean
        ) / model_std

        reconstructed_finite = np.isfinite(
            reconstructed_zscores
        )

        expected_finite = np.isfinite(
            expected_zscores
        )

        finite_mask_mismatch = (
            reconstructed_finite
            != expected_finite
        )

        reconstruction_finite_mask_mismatches += int(
            finite_mask_mismatch.sum()
        )

        both_finite = (
            reconstructed_finite
            & expected_finite
        )

        if both_finite.any():
            absolute_difference = np.abs(
                reconstructed_zscores[
                    both_finite
                ].astype(np.float64)
                - expected_zscores[
                    both_finite
                ].astype(np.float64)
            )

            reconstruction_max_abs_difference = max(
                reconstruction_max_abs_difference,
                float(
                    absolute_difference.max()
                ),
            )

            reconstruction_sum_abs_difference += float(
                absolute_difference.sum()
            )

            reconstruction_compared_values += int(
                absolute_difference.size
            )

        native_stats = safe_feature_statistics(
            native_values
        )

        model_stats = safe_feature_statistics(
            model_values
        )

        chunk_features = retained_features[
            start:end
        ]

        chunk_types = feature_type_for_names(
            spec,
            chunk_features,
        )

        finite_native = np.isfinite(
            native_values
        )

        n_finite_native = finite_native.sum(
            axis=0
        )

        methylation_boundary_fraction = np.full(
            end - start,
            np.nan,
            dtype=float,
        )

        cnv_non_diploid_fraction = np.full(
            end - start,
            np.nan,
            dtype=float,
        )

        if spec.key == "METHYLATION":
            boundary_count = (
                finite_native
                & (
                    (
                        native_values
                        <= METHYLATION_EPSILON
                    )
                    | (
                        native_values
                        >= (
                            np.float32(1.0)
                            - METHYLATION_EPSILON
                        )
                    )
                )
            ).sum(axis=0)

            methylation_boundary_fraction = np.divide(
                boundary_count,
                n_finite_native,
                out=np.full(
                    boundary_count.shape,
                    np.nan,
                    dtype=float,
                ),
                where=n_finite_native > 0,
            )

        if spec.key == "CNV":
            altered_count = (
                finite_native
                & ~np.isclose(
                    native_values,
                    np.float32(2.0),
                    rtol=0.0,
                    atol=1e-6,
                )
            ).sum(axis=0)

            cnv_non_diploid_fraction = np.divide(
                altered_count,
                n_finite_native,
                out=np.full(
                    altered_count.shape,
                    np.nan,
                    dtype=float,
                ),
                where=n_finite_native > 0,
            )

        part = pd.DataFrame(
            {
                "feature_index": np.arange(
                    start,
                    end,
                    dtype=int,
                ),
                "raw_feature_index": raw_columns,
                "feature": chunk_features,
                "feature_type": chunk_types,
                "n_samples": len(case_ids),
                "n_finite": (
                    model_stats[
                        "n_finite"
                    ]
                ),
                "missing_fraction": (
                    model_stats[
                        "missing_fraction"
                    ]
                ),
                "native_mean": (
                    native_stats[
                        "mean"
                    ]
                ),
                "native_variance": (
                    native_stats[
                        "variance"
                    ]
                ),
                "native_median": (
                    native_stats[
                        "median"
                    ]
                ),
                "native_mad": (
                    native_stats[
                        "mad"
                    ]
                ),
                "native_iqr": (
                    native_stats[
                        "iqr"
                    ]
                ),
                "native_min": (
                    native_stats[
                        "min"
                    ]
                ),
                "native_max": (
                    native_stats[
                        "max"
                    ]
                ),
                "native_zero_fraction": (
                    native_stats[
                        "zero_fraction"
                    ]
                ),
                "prestandardization_mean": (
                    model_stats[
                        "mean"
                    ]
                ),
                "prestandardization_variance": (
                    model_stats[
                        "variance"
                    ]
                ),
                "prestandardization_median": (
                    model_stats[
                        "median"
                    ]
                ),
                "prestandardization_mad": (
                    model_stats[
                        "mad"
                    ]
                ),
                "prestandardization_iqr": (
                    model_stats[
                        "iqr"
                    ]
                ),
                "prestandardization_min": (
                    model_stats[
                        "min"
                    ]
                ),
                "prestandardization_max": (
                    model_stats[
                        "max"
                    ]
                ),
                "methylation_boundary_fraction": (
                    methylation_boundary_fraction
                ),
                "cnv_non_diploid_fraction": (
                    cnv_non_diploid_fraction
                ),
            }
        )

        ranking_parts.append(part)

        print(
            f" Features "
            f"{start:,}-"
            f"{end:,}/"
            f"{n_features:,}"
        )

    ranking = pd.concat(
        ranking_parts,
        ignore_index=True,
    )

    minimum_finite_count = int(
        math.ceil(
            (
                1.0
                - spec.maximum_missing_fraction
            )
            * len(case_ids)
            - 1e-12
        )
    )

    eligible = (
        ranking[
            "n_finite"
        ].ge(
            minimum_finite_count
        )
        & ranking[
            "missing_fraction"
        ].le(
            spec.maximum_missing_fraction
            + 1e-12
        )
        & np.isfinite(
            ranking[
                "prestandardization_variance"
            ]
        )
        & ranking[
            "prestandardization_variance"
        ].gt(0)
    )

    if (
        spec.require_cg_probe_for_primary_models
    ):
        eligible &= ranking[
            "feature_type"
        ].eq("cg")

    ranking[
        "primary_model_eligible"
    ] = eligible

    ranking[
        "eligibility_reason"
    ] = np.where(
        ranking[
            "primary_model_eligible"
        ],
        "eligible",
        "excluded_by_missingness_variance_or_feature_type_rule",
    )

    assign_deterministic_rank(
        ranking,
        metric_column=(
            "prestandardization_variance"
        ),
        rank_column="variance_rank",
    )

    assign_deterministic_rank(
        ranking,
        metric_column=(
            "prestandardization_mad"
        ),
        rank_column="robust_mad_rank",
    )

    budget_definitions = [
        (
            "sensitivity_low",
            spec.low_budget,
        ),
        (
            "primary",
            spec.primary_budget,
        ),
        (
            "sensitivity_high",
            spec.high_budget,
        ),
    ]

    for role, budget in budget_definitions:
        rank_numeric = pd.to_numeric(
            ranking[
                "variance_rank"
            ],
            errors="coerce",
        )

        ranking[
            f"in_variance_{role}_budget"
        ] = (
            rank_numeric.le(
                budget
            )
        )

    robust_rank_numeric = pd.to_numeric(
        ranking[
            "robust_mad_rank"
        ],
        errors="coerce",
    )

    ranking[
        "in_robust_mad_primary_budget"
    ] = (
        robust_rank_numeric.le(
            spec.primary_budget
        )
    )

    eligible_count = int(
        ranking[
            "primary_model_eligible"
        ].sum()
    )

    if eligible_count < spec.high_budget:
        raise RuntimeError(
            f"{spec.key} has only "
            f"{eligible_count:,} eligible features, "
            f"below the prespecified high budget "
            f"of {spec.high_budget:,}."
        )

    ranking_path = (
        OUTPUT_DIR
        / (
            f"{spec.key}_"
            "prestandardization_"
            "feature_ranking.csv.gz"
        )
    )

    write_csv_gzip(
        ranking,
        ranking_path,
    )

    generated_paths = [
        ranking_path,
    ]

    candidate_rows: list[
        dict[str, Any]
    ] = []

    selection_columns = [
        "feature_index",
        "raw_feature_index",
        "feature",
        "feature_type",
        "n_finite",
        "missing_fraction",
        "prestandardization_variance",
        "prestandardization_mad",
        "native_variance",
        "methylation_boundary_fraction",
        "cnv_non_diploid_fraction",
    ]

    for role, budget in budget_definitions:
        rank_column = "variance_rank"

        rank_values = pd.to_numeric(
            ranking[rank_column],
            errors="coerce",
        )

        selected = ranking.loc[
            rank_values.le(budget),
            [
                *selection_columns,
                rank_column,
            ],
        ].copy()

        selected = selected.sort_values(
            rank_column,
        )

        selected = selected.rename(
            columns={
                rank_column: (
                    "selection_rank"
                )
            }
        )

        selected[
            "selection_rank"
        ] = (
            selected[
                "selection_rank"
            ]
            .astype(int)
        )

        selected_path = (
            FEATURE_SET_DIR
            / (
                f"{spec.key}_"
                f"{role}_"
                f"top{budget}_"
                "prestandardization_variance.csv"
            )
        )

        selected.to_csv(
            selected_path,
            index=False,
            float_format="%.12g",
        )

        generated_paths.append(
            selected_path
        )

        candidate_rows.append(
            {
                "modality": spec.key,
                "ranking_method": (
                    "prestandardization_variance"
                ),
                "budget_role": role,
                "n_features": budget,
                "candidate_file": (
                    relative_path(
                        selected_path
                    )
                ),
                "full_cohort_use": (
                    "final_refit_only"
                ),
                "held_out_benchmark_use": (
                    "forbidden_recompute_within_training_fold"
                ),
            }
        )

    robust_selected = ranking.loc[
        robust_rank_numeric.le(
            spec.primary_budget
        ),
        [
            *selection_columns,
            "robust_mad_rank",
        ],
    ].copy()

    robust_selected = robust_selected.sort_values(
        "robust_mad_rank",
    )

    robust_selected = robust_selected.rename(
        columns={
            "robust_mad_rank": (
                "selection_rank"
            )
        }
    )

    robust_selected[
        "selection_rank"
    ] = (
        robust_selected[
            "selection_rank"
        ]
        .astype(int)
    )

    robust_path = (
        FEATURE_SET_DIR
        / (
            f"{spec.key}_"
            f"primary_top{spec.primary_budget}_"
            "prestandardization_robust_mad.csv"
        )
    )

    robust_selected.to_csv(
        robust_path,
        index=False,
        float_format="%.12g",
    )

    generated_paths.append(
        robust_path
    )

    candidate_rows.append(
        {
            "modality": spec.key,
            "ranking_method": (
                "prestandardization_robust_MAD"
            ),
            "budget_role": (
                "primary_metric_sensitivity"
            ),
            "n_features": (
                spec.primary_budget
            ),
            "candidate_file": (
                relative_path(
                    robust_path
                )
            ),
            "full_cohort_use": (
                "final_refit_sensitivity_only"
            ),
            "held_out_benchmark_use": (
                "forbidden_recompute_within_training_fold"
            ),
        }
    )

    if reconstruction_compared_values > 0:
        reconstruction_mean_abs_difference = (
            reconstruction_sum_abs_difference
            / reconstruction_compared_values
        )
    else:
        reconstruction_mean_abs_difference = np.nan

    reconstruction_result = {
        "modality": spec.key,
        "n_samples": len(case_ids),
        "n_features": len(retained_features),
        "n_compared_finite_values": (
            reconstruction_compared_values
        ),
        "finite_mask_mismatches": (
            reconstruction_finite_mask_mismatches
        ),
        "maximum_absolute_difference": (
            reconstruction_max_abs_difference
        ),
        "mean_absolute_difference": (
            reconstruction_mean_abs_difference
        ),
        "tolerance": (
            RECONSTRUCTION_TOLERANCE
        ),
        "status": (
            "PASS"
            if (
                reconstruction_finite_mask_mismatches
                == 0
                and reconstruction_max_abs_difference
                <= RECONSTRUCTION_TOLERANCE
            )
            else "FAIL"
        ),
    }

    if reconstruction_result[
        "status"
    ] != "PASS":
        raise RuntimeError(
            f"{spec.key} pre-z-score transformation "
            "does not reproduce the locked harmonized "
            "matrix. "
            f"Finite-mask mismatches: "
            f"{reconstruction_finite_mask_mismatches}; "
            f"max absolute difference: "
            f"{reconstruction_max_abs_difference:.8g}."
        )

    eligible_variance = ranking.loc[
        ranking[
            "primary_model_eligible"
        ],
        "prestandardization_variance",
    ].to_numpy(
        dtype=float
    )

    eligible_mad = ranking.loc[
        ranking[
            "primary_model_eligible"
        ],
        "prestandardization_mad",
    ].to_numpy(
        dtype=float
    )

    summary_result = {
        "modality": spec.key,
        "n_cases": len(case_ids),
        "n_raw_features": len(raw_features),
        "n_harmonized_features": (
            len(retained_features)
        ),
        "n_primary_model_eligible": (
            eligible_count
        ),
        "native_scale": spec.native_scale,
        "prestandardization_model_scale": (
            spec.model_scale
        ),
        "ranking_metric": (
            "prestandardization_variance"
        ),
        "robust_sensitivity_metric": (
            "prestandardization_MAD"
        ),
        "mean_missing_fraction": float(
            ranking[
                "missing_fraction"
            ].mean()
        ),
        "maximum_missing_fraction": float(
            ranking[
                "missing_fraction"
            ].max()
        ),
        "variance_q05": float(
            np.nanquantile(
                eligible_variance,
                0.05,
            )
        ),
        "variance_median": float(
            np.nanmedian(
                eligible_variance
            )
        ),
        "variance_q95": float(
            np.nanquantile(
                eligible_variance,
                0.95,
            )
        ),
        "mad_median": float(
            np.nanmedian(
                eligible_mad
            )
        ),
        "low_budget": spec.low_budget,
        "primary_budget": (
            spec.primary_budget
        ),
        "high_budget": spec.high_budget,
        "post_standardization_variance_used": False,
        "full_cohort_ranking_allowed_in_cv": False,
        "ranking_file": relative_path(
            ranking_path
        ),
    }

    type_summary = (
        ranking.groupby(
            "feature_type",
            dropna=False,
        )
        .agg(
            n_features=(
                "feature",
                "size",
            ),
            n_primary_model_eligible=(
                "primary_model_eligible",
                "sum",
            ),
            mean_missing_fraction=(
                "missing_fraction",
                "mean",
            ),
            median_prestandardization_variance=(
                "prestandardization_variance",
                "median",
            ),
        )
        .reset_index()
    )

    type_summary.insert(
        0,
        "modality",
        spec.key,
    )

    input_record = {
        "raw_matrix": {
            "path": relative_path(
                spec.raw_matrix
            ),
            "sha256": sha256_file(
                spec.raw_matrix
            ),
            "shape": list(
                raw_matrix.shape
            ),
        },
        "raw_features": {
            "path": relative_path(
                spec.raw_features
            ),
            "sha256": sha256_file(
                spec.raw_features
            ),
            "feature_hash": hash_strings(
                raw_features
            ),
            "n_features": len(
                raw_features
            ),
        },
        "harmonized_matrix": {
            "path": relative_path(
                spec.harmonized_matrix
            ),
            "sha256": sha256_file(
                spec.harmonized_matrix
            ),
            "shape": list(
                harmonized_matrix.shape
            ),
        },
        "harmonized_features": {
            "path": relative_path(
                spec.harmonized_features
            ),
            "sha256": sha256_file(
                spec.harmonized_features
            ),
            "feature_hash": hash_strings(
                retained_features
            ),
            "n_features": len(
                retained_features
            ),
        },
    }

    return (
        ranking,
        summary_result,
        reconstruction_result,
        type_summary,
        candidate_rows,
        generated_paths,
        input_record,
    )


# ---------------------------------------------------------------------
# Preprocessing contract
# ---------------------------------------------------------------------

def build_fold_local_contract() -> dict[str, Any]:
    return {
        "contract_version": (
            "paper5-fold-local-preprocessing-v1"
        ),
        "purpose": (
            "Prevent information leakage during "
            "held-out representation benchmarking."
        ),
        "global_full_cohort_rankings": {
            "allowed_for": [
                (
                    "descriptive full-cohort audit"
                ),
                (
                    "final outcome-blind refit after "
                    "model family, rank, and feature "
                    "budget are frozen"
                ),
            ],
            "forbidden_for": [
                (
                    "outer-fold held-out evaluation"
                ),
                (
                    "cross-validated model selection"
                ),
                (
                    "held-out reconstruction scoring"
                ),
            ],
        },
        "held_out_benchmark_order": [
            (
                "Split samples into training and "
                "held-out folds before any "
                "data-dependent preprocessing."
            ),
            (
                "Apply the fixed missingness and "
                "feature-type eligibility rules using "
                "training samples only."
            ),
            (
                "Calculate pre-standardization "
                "variance or robust MAD using training "
                "samples only."
            ),
            (
                "Select the prespecified feature "
                "budget using the training-fold "
                "ranking only."
            ),
            (
                "Fit median imputation on training "
                "samples only for PCA, block PCA, "
                "AE, and VAE."
            ),
            (
                "Use KNN imputation only as a "
                "prespecified sensitivity analysis and "
                "fit it on training samples only."
            ),
            (
                "Fit centering, scaling, and any "
                "PCA compression on training samples "
                "only."
            ),
            (
                "Apply frozen training-fold "
                "preprocessing parameters to the "
                "held-out samples."
            ),
            (
                "Calculate held-out reconstruction "
                "and coherence metrics without "
                "refitting on held-out samples."
            ),
        ],
        "model_specific_rules": {
            "modality_specific_PCA": {
                "feature_budget": (
                    "all training-fold eligible "
                    "features"
                ),
                "imputation": (
                    "training-fold median"
                ),
                "scaling": (
                    "training-fold mean and standard "
                    "deviation"
                ),
            },
            "block_scaled_PCA": {
                "feature_budget": (
                    "all training-fold eligible "
                    "features"
                ),
                "imputation": (
                    "training-fold median"
                ),
                "block_scaling": (
                    "equalize total training-fold "
                    "variance contributed by each "
                    "modality before concatenation"
                ),
            },
            "MOFA": {
                "feature_budget": (
                    "prespecified low, primary, or "
                    "high budget, ranked within the "
                    "training fold"
                ),
                "missing_values": (
                    "retained for native probabilistic "
                    "handling"
                ),
                "imputation": "none",
            },
            "shared_private_AE": {
                "feature_budget": (
                    "prespecified low, primary, or "
                    "high budget, ranked within the "
                    "training fold"
                ),
                "imputation": (
                    "training-fold median"
                ),
                "encoder_input": (
                    "training-fold PCA-compressed "
                    "modality representations, not "
                    "raw high-dimensional features"
                ),
            },
            "VAE": {
                "role": "secondary_exploratory",
                "feature_budget": (
                    "same fold-local rule as the AE"
                ),
                "encoder_input": (
                    "training-fold PCA-compressed "
                    "modality representations"
                ),
            },
            "regularized_CCA": {
                "role": "secondary_exploratory",
                "ordinary_CCA_allowed": False,
                "regularization": (
                    "ridge or sparse CCA only; "
                    "penalty selection inside the "
                    "training fold"
                ),
            },
        },
        "forbidden_operations": [
            (
                "Variance ranking after z-score "
                "standardization"
            ),
            (
                "Global imputation before "
                "cross-validation"
            ),
            (
                "Global PCA before held-out "
                "evaluation"
            ),
            (
                "Use of survival, metastasis, "
                "response, treatment, or other "
                "outcomes in feature ranking"
            ),
        ],
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and freeze pre-standardization "
            "feature-variance rankings for the "
            "TARGET-OS Paper 5 multi-omics inputs."
        )
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "Number of features processed per chunk. "
            f"Default: {DEFAULT_CHUNK_SIZE}."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.chunk_size <= 0:
        raise ValueError(
            "--chunk-size must be positive."
        )

    print("=" * 88)
    print(
        "Paper 5 - pre-standardization "
        "feature variance audit"
    )
    print("=" * 88)
    print(
        f"Script version: {SCRIPT_VERSION}"
    )
    print(
        f"Project root: {PROJECT_ROOT}"
    )
    print(
        f"Input directory: {INPUT_DIR}"
    )
    print(
        f"Output directory: {OUTPUT_DIR}"
    )
    print(
        f"Chunk size: {args.chunk_size:,}"
    )
    print()

    case_ids = load_case_order()

    print(
        f"Locked cases: {len(case_ids)}"
    )
    print(
        f"Case-order hash: "
        f"{hash_strings(case_ids)}"
    )
    print()

    (
        harmonized_lock,
        verified_harmonized_hashes,
    ) = load_and_validate_harmonized_lock(
        case_ids
    )

    print()
    print(
        "04b harmonized lock verification: PASS"
    )
    print(
        "Post-standardization variance will not "
        "be used for feature ranking."
    )

    summary_rows: list[
        dict[str, Any]
    ] = []

    reconstruction_rows: list[
        dict[str, Any]
    ] = []

    feature_type_tables: list[
        pd.DataFrame
    ] = []

    candidate_rows: list[
        dict[str, Any]
    ] = []

    generated_paths: list[
        Path
    ] = []

    modality_inputs: dict[
        str,
        dict[str, Any],
    ] = {}

    for spec in MODALITIES:
        (
            _ranking,
            summary_result,
            reconstruction_result,
            type_summary,
            modality_candidates,
            modality_generated_paths,
            input_record,
        ) = process_modality(
            spec=spec,
            case_ids=case_ids,
            chunk_size=int(
                args.chunk_size
            ),
        )

        summary_rows.append(
            summary_result
        )

        reconstruction_rows.append(
            reconstruction_result
        )

        feature_type_tables.append(
            type_summary
        )

        candidate_rows.extend(
            modality_candidates
        )

        generated_paths.extend(
            modality_generated_paths
        )

        modality_inputs[
            spec.key
        ] = input_record

    summary = pd.DataFrame(
        summary_rows
    )

    reconstruction = pd.DataFrame(
        reconstruction_rows
    )

    feature_type_summary = pd.concat(
        feature_type_tables,
        ignore_index=True,
    )

    candidate_plan = pd.DataFrame(
        candidate_rows
    )

    # Linear PCA baselines use all fold-local eligible features.
    linear_rows: list[
        dict[str, Any]
    ] = []

    for spec in MODALITIES:
        n_retained = int(
            summary.loc[
                summary[
                    "modality"
                ].eq(spec.key),
                "n_harmonized_features",
            ].iloc[0]
        )

        for model_family in [
            "modality_specific_PCA",
            "block_scaled_PCA",
        ]:
            linear_rows.append(
                {
                    "modality": spec.key,
                    "ranking_method": (
                        "no_global_budget_all_"
                        "training_fold_eligible_features"
                    ),
                    "budget_role": (
                        "all_fold_local_eligible"
                    ),
                    "n_features": n_retained,
                    "candidate_file": "",
                    "full_cohort_use": (
                        "descriptive_and_final_refit"
                    ),
                    "held_out_benchmark_use": (
                        "recompute_eligibility_"
                        "imputation_scaling_and_PCA_"
                        "inside_training_fold"
                    ),
                    "model_family": model_family,
                }
            )

    linear_plan = pd.DataFrame(
        linear_rows
    )

    candidate_plan[
        "model_family"
    ] = (
        "MOFA;shared_private_AE;VAE"
    )

    budget_plan = pd.concat(
        [
            linear_plan,
            candidate_plan,
        ],
        ignore_index=True,
        sort=False,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        float_format="%.12g",
    )

    reconstruction.to_csv(
        OUTPUT_RECONSTRUCTION,
        index=False,
        float_format="%.12g",
    )

    feature_type_summary.to_csv(
        OUTPUT_FEATURE_TYPE_SUMMARY,
        index=False,
        float_format="%.12g",
    )

    budget_plan.to_csv(
        OUTPUT_BUDGET_PLAN,
        index=False,
    )

    generated_paths.extend(
        [
            OUTPUT_SUMMARY,
            OUTPUT_RECONSTRUCTION,
            OUTPUT_FEATURE_TYPE_SUMMARY,
            OUTPUT_BUDGET_PLAN,
        ]
    )

    fold_contract = (
        build_fold_local_contract()
    )

    OUTPUT_FOLD_CONTRACT.write_text(
        json.dumps(
            fold_contract,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    generated_paths.append(
        OUTPUT_FOLD_CONTRACT
    )

    print()
    print("-" * 88)
    print(
        "Pre-standardization variance summary"
    )
    print("-" * 88)

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        220,
    ):
        print(
            summary.to_string(
                index=False
            )
        )

    print()
    print("-" * 88)
    print(
        "Transformation reconstruction audit"
    )
    print("-" * 88)

    print(
        reconstruction.to_string(
            index=False
        )
    )

    if not reconstruction[
        "status"
    ].eq("PASS").all():
        raise RuntimeError(
            "At least one reconstruction audit failed."
        )

    print()
    print("-" * 88)
    print(
        "Prespecified computational feature budgets"
    )
    print("-" * 88)

    print(
        candidate_plan[
            [
                "modality",
                "ranking_method",
                "budget_role",
                "n_features",
            ]
        ].to_string(
            index=False
        )
    )

    output_hashes = {
        relative_path(path): {
            "sha256": sha256_file(
                path
            ),
            "size_bytes": int(
                path.stat().st_size
            ),
        }
        for path in sorted(
            set(generated_paths)
        )
        if path.exists()
    }

    manifest = {
        "script_version": (
            SCRIPT_VERSION
        ),
        "created_at_utc": (
            utc_now_iso()
        ),
        "project_root": str(
            PROJECT_ROOT
        ),
        "outcome_loaded": False,
        "outcome_guided_feature_selection": False,
        "post_standardization_variance_used_for_selection": False,
        "model_fitting_performed": False,
        "cross_validation_performed": False,
        "case_order": {
            "n_cases": len(case_ids),
            "case_hash": hash_strings(
                case_ids
            ),
            "source": relative_path(
                CASE_ORDER_FILE
            ),
        },
        "upstream_lock": {
            "path": relative_path(
                HARMONIZED_LOCK_FILE
            ),
            "sha256": sha256_file(
                HARMONIZED_LOCK_FILE
            ),
            "script_version": (
                harmonized_lock.get(
                    "script_version"
                )
            ),
            "verified_harmonized_matrix_hashes": (
                verified_harmonized_hashes
            ),
        },
        "ranking_contract": {
            "primary_metric": (
                "sample variance calculated on "
                "the transformed model input "
                "before z-standardization"
            ),
            "robust_sensitivity_metric": (
                "median absolute deviation "
                "calculated before "
                "z-standardization"
            ),
            "tie_breaker": (
                "original retained feature index "
                "ascending"
            ),
            "methylation_primary_feature_type": (
                "cg probes only"
            ),
            "RNA_model_scale": (
                "log2 TPM before z-score"
            ),
            "methylation_model_scale": (
                "M-value before z-score"
            ),
            "CNV_model_scale": (
                "log2(copy number / 2) "
                "before z-score"
            ),
        },
        "budget_contract": {
            "RNA": {
                "sensitivity_low": 2_000,
                "primary": 5_000,
                "sensitivity_high": 10_000,
            },
            "METHYLATION": {
                "sensitivity_low": 5_000,
                "primary": 10_000,
                "sensitivity_high": 20_000,
            },
            "CNV": {
                "sensitivity_low": 2_000,
                "primary": 5_000,
                "sensitivity_high": 10_000,
            },
            "intended_models": [
                "MOFA",
                "shared_private_AE",
                "VAE_exploratory",
            ],
        },
        "critical_cv_restriction": (
            "Full-cohort rankings and feature lists "
            "must not be used in held-out model "
            "evaluation. Missingness filtering, "
            "ranking, imputation, scaling, and PCA "
            "must be fitted independently within "
            "each training fold."
        ),
        "modality_inputs": (
            modality_inputs
        ),
        "guardrails": [
            (
                "No clinical outcome or endpoint "
                "was loaded."
            ),
            (
                "Post-z-score variance was not used "
                "for feature ranking."
            ),
            (
                "Feature rankings were calculated "
                "only from molecular measurements."
            ),
            (
                "The global rankings are descriptive "
                "and may be used only for the final "
                "outcome-blind refit after model "
                "choices are frozen."
            ),
            (
                "Held-out benchmark preprocessing "
                "must be recomputed using training "
                "samples only."
            ),
            (
                "Ordinary unregularized CCA is "
                "forbidden because p greatly "
                "exceeds n."
            ),
            (
                "AE and VAE models must consume "
                "fold-local PCA-compressed modality "
                "representations rather than raw "
                "high-dimensional vectors."
            ),
        ],
        "outputs": output_hashes,
    }

    OUTPUT_MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    readme_lines = [
        "Paper 5 pre-standardization feature variance audit",
        "=" * 72,
        "",
        f"Script version: {SCRIPT_VERSION}",
        f"Created UTC: {manifest['created_at_utc']}",
        "",
        "Purpose",
        "-------",
        (
            "Create outcome-blind feature rankings "
            "using molecular variance before "
            "z-standardization."
        ),
        (
            "The script also verifies that the "
            "pre-z-score transformations reproduce "
            "the matrices frozen by script 04b."
        ),
        "",
        "Primary ranking rule",
        "--------------------",
        (
            "RNA: variance of log2 TPM before z-score."
        ),
        (
            "Methylation: variance of M-values before "
            "z-score; primary candidate budgets are "
            "restricted to cg probes."
        ),
        (
            "CNV: variance of log2(copy number / 2) "
            "before z-score."
        ),
        (
            "Robust MAD rankings are retained as "
            "prespecified sensitivity sets."
        ),
        "",
        "Critical held-out evaluation rule",
        "---------------------------------",
        (
            "The full-cohort rankings written here "
            "must not be used in cross-validated "
            "held-out evaluation."
        ),
        (
            "Within each benchmark fold, feature "
            "eligibility, variance ranking, "
            "imputation, scaling, and PCA must be "
            "fitted using training samples only."
        ),
        "",
        "Scientific state",
        "----------------",
        "outcome_loaded: false",
        "model_fitting_performed: false",
        "post-standardization variance used for ranking: false",
        "",
        "Summary",
        "-------",
    ]

    for _, row in summary.iterrows():
        readme_lines.append(
            (
                f"{row['modality']}: "
                f"{int(row['n_harmonized_features'])} "
                f"retained; "
                f"{int(row['n_primary_model_eligible'])} "
                "primary-model eligible"
            )
        )

    readme_lines.extend(
        [
            "",
            "Outputs",
            "-------",
            relative_path(
                OUTPUT_SUMMARY
            ),
            relative_path(
                OUTPUT_RECONSTRUCTION
            ),
            relative_path(
                OUTPUT_FEATURE_TYPE_SUMMARY
            ),
            relative_path(
                OUTPUT_BUDGET_PLAN
            ),
            relative_path(
                OUTPUT_FOLD_CONTRACT
            ),
            relative_path(
                OUTPUT_MANIFEST
            ),
            "",
        ]
    )

    OUTPUT_README.write_text(
        "\n".join(
            readme_lines
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print(
        "Pre-standardization variance audit: PASS"
    )
    print("=" * 88)
    print(
        "Post-z-score variance used for ranking: NO"
    )
    print(
        "Full-cohort rankings permitted in held-out CV: NO"
    )
    print(
        "Fold-local preprocessing contract: "
        f"{relative_path(OUTPUT_FOLD_CONTRACT)}"
    )
    print(
        "Manifest: "
        f"{relative_path(OUTPUT_MANIFEST)}"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
