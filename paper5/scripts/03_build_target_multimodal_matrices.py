from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


SCRIPT_VERSION = "03-build-target-multimodal-matrices-v1"

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

MANIFEST_DIR = PAPER5_ROOT / "data_manifest"
MODEL_INPUT_DIR = (
    PAPER5_ROOT
    / "model_inputs"
    / "TARGET_OS_primary78"
)

MODEL_INPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FILE_INVENTORY = (
    MANIFEST_DIR
    / "TARGET_OS_gdc_file_inventory.csv"
)

COMBINATION_FILE = (
    MANIFEST_DIR
    / "TARGET_OS_multimodal_case_combinations.csv"
)

OUTPUT_CASE_ORDER = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_multimodal_case_order.csv"
)

OUTPUT_FILE_MAP = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_multimodal_file_map.csv"
)

OUTPUT_FILE_SELECTION_AUDIT = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_multimodal_file_selection_audit.csv"
)

OUTPUT_RNA_MATRIX = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_rna_log2tpm.npy"
)

OUTPUT_RNA_FEATURES = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_rna_features.csv"
)

OUTPUT_RNA_SAMPLE_QC = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_rna_sample_qc.csv"
)

OUTPUT_RNA_FEATURE_QC = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_rna_feature_qc.csv"
)

OUTPUT_METH_MATRIX = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_methylation_beta.npy"
)

OUTPUT_METH_FEATURES = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_methylation_features.csv"
)

OUTPUT_METH_SAMPLE_QC = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_methylation_sample_qc.csv"
)

OUTPUT_METH_FEATURE_QC = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_methylation_feature_qc.csv"
)

OUTPUT_CNV_MATRIX = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_cnv_gene_copy_number.npy"
)

OUTPUT_CNV_FEATURES = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_cnv_features.csv"
)

OUTPUT_CNV_SAMPLE_QC = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_cnv_sample_qc.csv"
)

OUTPUT_CNV_FEATURE_QC = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_cnv_feature_qc.csv"
)

OUTPUT_MODALITY_SUMMARY = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_multimodal_matrix_summary.csv"
)

OUTPUT_MANIFEST = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_multimodal_matrix_manifest.json"
)

OUTPUT_README = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_multimodal_matrix_README.txt"
)

PRIMARY_COMBINATION = (
    "RNA + methylation + CNV"
)

PRIMARY_SCOPE = "primary_tumor"

PRIMARY_MODALITIES = [
    "rna",
    "methylation",
    "cnv_gene",
]

MIN_PRIMARY_CASES = 30

FEATURE_QC_CHUNK = 50000


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def clean_text(
    value: Any,
) -> str:
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


def normalize_column(
    value: Any,
) -> str:
    text = clean_text(
        value
    ).lower()

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    ).strip("_")


def clean_gene_symbol(
    value: Any,
) -> str:
    text = clean_text(
        value
    ).upper()

    if not text:
        return ""

    text = re.sub(
        r"\.\d+$",
        "",
        text,
    )

    return text


def clean_ensembl_id(
    value: Any,
) -> str:
    text = clean_text(
        value
    ).upper()

    return re.sub(
        r"\.\d+$",
        "",
        text,
    )


def relative_path(
    path: Path,
) -> str:
    try:
        return str(
            path.resolve().relative_to(
                PROJECT_ROOT.resolve()
            )
        )
    except ValueError:
        return str(
            path.resolve()
        )


def split_ids(
    value: Any,
) -> list[str]:
    text = clean_text(
        value
    )

    if not text:
        return []

    return [
        item.strip()
        for item in text.split(";")
        if item.strip()
    ]


def parse_bool(
    value: Any,
) -> bool:
    if isinstance(
        value,
        bool,
    ):
        return value

    return (
        clean_text(value).lower()
        in {
            "true",
            "1",
            "yes",
        }
    )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def resolve_local_path(
    value: Any,
) -> Path:
    text = clean_text(
        value
    )

    if not text:
        raise ValueError(
            "Empty local path."
        )

    path = Path(text)

    if not path.is_absolute():
        path = (
            PROJECT_ROOT
            / path
        )

    return path


# ---------------------------------------------------------------------
# Case selection
# ---------------------------------------------------------------------

def load_primary_cases() -> list[str]:
    if not COMBINATION_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {COMBINATION_FILE}"
        )

    table = pd.read_csv(
        COMBINATION_FILE,
        dtype=str,
    ).fillna("")

    required = {
        "scope",
        "combination",
        "n_cases",
        "case_ids",
    }

    missing = required - set(
        table.columns
    )

    if missing:
        raise ValueError(
            "Combination table missing: "
            + ", ".join(
                sorted(missing)
            )
        )

    part = table[
        table[
            "scope"
        ].eq(
            PRIMARY_SCOPE
        )
        & table[
            "combination"
        ].eq(
            PRIMARY_COMBINATION
        )
    ]

    if part.shape[0] != 1:
        raise RuntimeError(
            "Could not uniquely identify "
            "the primary TARGET combination."
        )

    row = part.iloc[0]

    case_ids = sorted(
        set(
            split_ids(
                row["case_ids"]
            )
        )
    )

    expected = int(
        float(
            row["n_cases"]
        )
    )

    if len(case_ids) != expected:
        raise RuntimeError(
            "Case-ID count does not match "
            f"reported n_cases: "
            f"{len(case_ids)} vs {expected}"
        )

    if len(case_ids) < MIN_PRIMARY_CASES:
        raise RuntimeError(
            "Too few primary multimodal "
            f"cases: {len(case_ids)}"
        )

    return case_ids


# ---------------------------------------------------------------------
# File mapping
# ---------------------------------------------------------------------

def row_primary_score(
    row: pd.Series,
) -> int:
    score = 0

    tissue = clean_text(
        row.get(
            "tissue_type"
        )
    ).lower()

    descriptor = clean_text(
        row.get(
            "tumor_descriptor"
        )
    ).lower()

    samples = clean_text(
        row.get(
            "parsed_sample_ids"
        )
    ).upper()

    if "tumor" in tissue:
        score += 4

    if "primary" in descriptor:
        score += 4

    if re.search(
        r"-01[A-Z0-9]*",
        samples,
    ):
        score += 2

    return score


def explode_inventory_cases(
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in inventory.iterrows():
        case_ids = split_ids(
            row.get(
                "parsed_case_ids"
            )
        )

        for case_id in case_ids:
            record = (
                row.to_dict()
            )

            record[
                "case_key"
            ] = case_id.upper()

            record[
                "primary_selection_score"
            ] = row_primary_score(
                row
            )

            rows.append(
                record
            )

    return pd.DataFrame(
        rows
    )


def build_file_map(
    case_ids: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    if not FILE_INVENTORY.exists():
        raise FileNotFoundError(
            f"Missing: {FILE_INVENTORY}"
        )

    inventory = pd.read_csv(
        FILE_INVENTORY,
        dtype=str,
        low_memory=False,
    ).fillna("")

    required = {
        "modality",
        "parsed_case_ids",
        "downloaded",
        "local_path",
        "file_name",
    }

    missing = required - set(
        inventory.columns
    )

    if missing:
        raise ValueError(
            "GDC inventory missing: "
            + ", ".join(
                sorted(missing)
            )
        )

    inventory = inventory[
        inventory[
            "modality"
        ].isin(
            PRIMARY_MODALITIES
        )
        & inventory[
            "downloaded"
        ].map(
            parse_bool
        )
    ].copy()

    exploded = explode_inventory_cases(
        inventory
    )

    rows = []
    audit_rows = []

    for case_id in case_ids:
        for modality in PRIMARY_MODALITIES:
            candidates = exploded[
                exploded[
                    "case_key"
                ].eq(
                    case_id
                )
                & exploded[
                    "modality"
                ].eq(
                    modality
                )
            ].copy()

            if candidates.empty:
                raise RuntimeError(
                    "No downloaded file for "
                    f"{case_id} / {modality}"
                )

            candidates = (
                candidates
                .sort_values(
                    [
                        "primary_selection_score",
                        "file_name",
                    ],
                    ascending=[
                        False,
                        True,
                    ],
                )
                .reset_index(
                    drop=True
                )
            )

            selected = (
                candidates.iloc[0]
            )

            local_path = (
                resolve_local_path(
                    selected[
                        "local_path"
                    ]
                )
            )

            if not local_path.exists():
                raise FileNotFoundError(
                    f"Selected file missing: "
                    f"{local_path}"
                )

            rows.append(
                {
                    "case_id": case_id,
                    "modality": modality,
                    "file_name": (
                        selected[
                            "file_name"
                        ]
                    ),
                    "local_path": (
                        relative_path(
                            local_path
                        )
                    ),
                    "selection_score": int(
                        selected[
                            "primary_selection_score"
                        ]
                    ),
                    "n_candidates": int(
                        candidates.shape[0]
                    ),
                    "tissue_type": (
                        selected.get(
                            "tissue_type",
                            "",
                        )
                    ),
                    "tumor_descriptor": (
                        selected.get(
                            "tumor_descriptor",
                            "",
                        )
                    ),
                    "parsed_sample_ids": (
                        selected.get(
                            "parsed_sample_ids",
                            "",
                        )
                    ),
                }
            )

            audit_rows.append(
                {
                    "case_id": case_id,
                    "modality": modality,
                    "n_candidates": int(
                        candidates.shape[0]
                    ),
                    "selected_file": (
                        selected[
                            "file_name"
                        ]
                    ),
                    "all_candidate_files": (
                        ";".join(
                            candidates[
                                "file_name"
                            ].astype(str)
                        )
                    ),
                    "all_candidate_scores": (
                        ";".join(
                            candidates[
                                "primary_selection_score"
                            ]
                            .astype(str)
                        )
                    ),
                }
            )

    file_map = pd.DataFrame(
        rows
    )

    audit = pd.DataFrame(
        audit_rows
    )

    expected_rows = (
        len(case_ids)
        * len(
            PRIMARY_MODALITIES
        )
    )

    if file_map.shape[0] != expected_rows:
        raise RuntimeError(
            "Incomplete file mapping."
        )

    return file_map, audit


# ---------------------------------------------------------------------
# RNA
# ---------------------------------------------------------------------

def read_rna_series(
    path: Path,
) -> tuple[
    pd.Series,
    dict[str, Any],
]:
    table = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        low_memory=False,
    )

    table.columns = [
        normalize_column(
            column
        )
        for column in table.columns
    ]

    gene_column = next(
        (
            column
            for column in [
                "gene_name",
                "gene_symbol",
                "symbol",
                "gene_id",
            ]
            if column
            in table.columns
        ),
        None,
    )

    if gene_column is None:
        raise ValueError(
            f"No RNA gene identifier "
            f"column in {path.name}"
        )

    value_column = next(
        (
            column
            for column in [
                "tpm_unstranded",
                "tpm",
                "fpkm_uq_unstranded",
                "fpkm_unstranded",
                "unstranded",
            ]
            if column
            in table.columns
        ),
        None,
    )

    if value_column is None:
        raise ValueError(
            f"No supported RNA value "
            f"column in {path.name}. "
            f"Columns: "
            f"{table.columns.tolist()}"
        )

    if gene_column in {
        "gene_name",
        "gene_symbol",
        "symbol",
    }:
        features = (
            table[
                gene_column
            ]
            .map(
                clean_gene_symbol
            )
        )

    else:
        features = (
            table[
                gene_column
            ]
            .map(
                clean_ensembl_id
            )
        )

    values = pd.to_numeric(
        table[
            value_column
        ],
        errors="coerce",
    )

    valid = (
        features.ne("")
        & ~features.str.startswith(
            "N_"
        )
        & values.notna()
    )

    data = pd.DataFrame(
        {
            "feature": (
                features[
                    valid
                ]
            ),
            "value": (
                values[
                    valid
                ]
            ),
        }
    )

    data = (
        data.groupby(
            "feature",
            sort=False,
        )["value"]
        .mean()
    )

    # GDC normalized expression values are transformed
    # outcome-blind for representation learning.
    if (
        "tpm"
        in value_column
        or "fpkm"
        in value_column
    ):
        data = np.log2(
            data.clip(
                lower=0
            )
            + 1.0
        )

        transform = (
            f"log2({value_column}+1)"
        )
    else:
        data = np.log2(
            data.clip(
                lower=0
            )
            + 1.0
        )

        transform = (
            f"log2({value_column}+1)"
        )

    return (
        data.astype(
            np.float32
        ),
        {
            "gene_column": (
                gene_column
            ),
            "value_column": (
                value_column
            ),
            "transform": (
                transform
            ),
        },
    )


# ---------------------------------------------------------------------
# Methylation
# ---------------------------------------------------------------------

def read_methylation_series(
    path: Path,
) -> tuple[
    pd.Series,
    dict[str, Any],
]:
    table = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        low_memory=False,
    )

    original_columns = list(
        table.columns
    )

    table.columns = [
        normalize_column(
            column
        )
        for column in table.columns
    ]

    feature_column = next(
        (
            column
            for column in [
                "composite_element",
                "composite_element_ref",
                "probe_id",
                "probe",
            ]
            if column
            in table.columns
        ),
        None,
    )

    value_column = next(
        (
            column
            for column in [
                "beta_value",
                "beta",
            ]
            if column
            in table.columns
        ),
        None,
    )

    if (
        feature_column is None
        and len(
            table.columns
        ) >= 1
    ):
        feature_column = (
            table.columns[0]
        )

    if (
        value_column is None
        and len(
            table.columns
        ) >= 2
    ):
        value_column = (
            table.columns[1]
        )

    if (
        feature_column is None
        or value_column is None
    ):
        raise ValueError(
            f"Could not identify "
            f"methylation columns "
            f"in {path.name}. "
            f"Original columns: "
            f"{original_columns}"
        )

    features = (
        table[
            feature_column
        ]
        .map(
            clean_text
        )
    )

    values = pd.to_numeric(
        table[
            value_column
        ],
        errors="coerce",
    )

    # Beta values outside [0, 1] are invalid.
    values = values.where(
        values.between(
            0.0,
            1.0,
            inclusive="both",
        )
    )

    valid_feature = (
        features.ne("")
    )

    data = pd.DataFrame(
        {
            "feature": (
                features[
                    valid_feature
                ]
            ),
            "value": (
                values[
                    valid_feature
                ]
            ),
        }
    )

    data = (
        data.groupby(
            "feature",
            sort=False,
        )["value"]
        .mean()
    )

    return (
        data.astype(
            np.float32
        ),
        {
            "feature_column": (
                feature_column
            ),
            "value_column": (
                value_column
            ),
            "transform": (
                "none_beta_0_1"
            ),
        },
    )


# ---------------------------------------------------------------------
# CNV
# ---------------------------------------------------------------------

def read_cnv_series(
    path: Path,
) -> tuple[
    pd.Series,
    dict[str, Any],
]:
    table = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        low_memory=False,
    )

    table.columns = [
        normalize_column(
            column
        )
        for column in table.columns
    ]

    feature_column = next(
        (
            column
            for column in [
                "gene_name",
                "gene_symbol",
                "symbol",
                "gene_id",
            ]
            if column
            in table.columns
        ),
        None,
    )

    value_column = next(
        (
            column
            for column in [
                "copy_number",
                "copynumber",
            ]
            if column
            in table.columns
        ),
        None,
    )

    if feature_column is None:
        raise ValueError(
            f"No CNV gene identifier "
            f"in {path.name}. "
            f"Columns: "
            f"{table.columns.tolist()}"
        )

    if value_column is None:
        raise ValueError(
            f"No CNV copy-number column "
            f"in {path.name}. "
            f"Columns: "
            f"{table.columns.tolist()}"
        )

    if feature_column in {
        "gene_name",
        "gene_symbol",
        "symbol",
    }:
        features = (
            table[
                feature_column
            ]
            .map(
                clean_gene_symbol
            )
        )

    else:
        features = (
            table[
                feature_column
            ]
            .map(
                clean_ensembl_id
            )
        )

    values = pd.to_numeric(
        table[
            value_column
        ],
        errors="coerce",
    )

    valid = (
        features.ne("")
    )

    data = pd.DataFrame(
        {
            "feature": (
                features[
                    valid
                ]
            ),
            "value": (
                values[
                    valid
                ]
            ),
        }
    )

    # A gene can overlap more than one segment.
    # Median aggregation is deterministic and
    # prevents duplicate identifiers in the matrix.
    data = (
        data.groupby(
            "feature",
            sort=False,
        )["value"]
        .median()
    )

    return (
        data.astype(
            np.float32
        ),
        {
            "feature_column": (
                feature_column
            ),
            "value_column": (
                value_column
            ),
            "transform": (
                "none_integer_copy_number"
            ),
        },
    )


# ---------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------

Reader = Callable[
    [Path],
    tuple[
        pd.Series,
        dict[str, Any],
    ],
]


def modality_file_table(
    file_map: pd.DataFrame,
    case_ids: list[str],
    modality: str,
) -> pd.DataFrame:
    part = file_map[
        file_map[
            "modality"
        ].eq(
            modality
        )
    ].copy()

    part = (
        part.set_index(
            "case_id"
        )
        .loc[
            case_ids
        ]
        .reset_index()
    )

    if part.shape[0] != len(
        case_ids
    ):
        raise RuntimeError(
            f"File map incomplete "
            f"for {modality}."
        )

    return part


def determine_common_features(
    file_table: pd.DataFrame,
    reader: Reader,
    modality: str,
) -> tuple[
    list[str],
    dict[str, Any],
    pd.DataFrame,
]:
    common: set[str] | None = None
    first_order: list[str] = []
    reader_metadata: dict[str, Any] = {}

    rows = []

    print(
        f"  First pass: determining "
        f"shared {modality} features..."
    )

    for index, row in (
        file_table.iterrows()
    ):
        case_id = row[
            "case_id"
        ]

        path = resolve_local_path(
            row[
                "local_path"
            ]
        )

        series, metadata = reader(
            path
        )

        feature_set = set(
            series.index.astype(
                str
            )
        )

        if common is None:
            common = feature_set
            first_order = (
                series.index
                .astype(str)
                .tolist()
            )

            reader_metadata = (
                metadata
            )
        else:
            common &= feature_set

        rows.append(
            {
                "case_id": (
                    case_id
                ),
                "n_features_in_file": (
                    len(
                        feature_set
                    )
                ),
                "n_nonmissing_values": int(
                    series.notna().sum()
                ),
                "file_name": path.name,
            }
        )

        if (
            (index + 1) % 10 == 0
            or (
                index + 1
                == file_table.shape[0]
            )
        ):
            print(
                f"    inspected "
                f"{index + 1}/"
                f"{file_table.shape[0]}"
            )

    if common is None:
        raise RuntimeError(
            f"No features found "
            f"for {modality}."
        )

    features = [
        feature
        for feature in first_order
        if feature in common
    ]

    if not features:
        raise RuntimeError(
            f"No common features "
            f"for {modality}."
        )

    return (
        features,
        reader_metadata,
        pd.DataFrame(
            rows
        ),
    )


def write_feature_qc(
    matrix_path: Path,
    feature_names: list[str],
    output_path: Path,
) -> pd.DataFrame:
    matrix = np.load(
        matrix_path,
        mmap_mode="r",
    )

    n_samples, n_features = (
        matrix.shape
    )

    chunks = []

    for start in range(
        0,
        n_features,
        FEATURE_QC_CHUNK,
    ):
        end = min(
            start
            + FEATURE_QC_CHUNK,
            n_features,
        )

        block = np.asarray(
            matrix[
                :,
                start:end,
            ],
            dtype=np.float64,
        )

        finite = np.isfinite(
            block
        )

        n_finite = finite.sum(
            axis=0
        )

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                category=RuntimeWarning,
            )

            means = np.nanmean(
                block,
                axis=0,
            )

            variances = np.nanvar(
                block,
                axis=0,
                ddof=1,
            )

            minimum = np.nanmin(
                block,
                axis=0,
            )

            maximum = np.nanmax(
                block,
                axis=0,
            )

        part = pd.DataFrame(
            {
                "feature": (
                    feature_names[
                        start:end
                    ]
                ),
                "n_finite": (
                    n_finite
                ),
                "missing_fraction": (
                    1.0
                    - n_finite
                    / n_samples
                ),
                "mean": means,
                "variance": variances,
                "min": minimum,
                "max": maximum,
            }
        )

        chunks.append(
            part
        )

    qc = pd.concat(
        chunks,
        ignore_index=True,
    )

    qc[
        "zero_or_undefined_variance"
    ] = (
        ~np.isfinite(
            qc["variance"]
        )
        | qc[
            "variance"
        ].le(0)
    )

    qc.to_csv(
        output_path,
        index=False,
    )

    return qc


def build_modality_matrix(
    case_ids: list[str],
    file_map: pd.DataFrame,
    modality: str,
    reader: Reader,
    matrix_output: Path,
    features_output: Path,
    sample_qc_output: Path,
    feature_qc_output: Path,
) -> dict[str, Any]:
    print()
    print(
        f"Building modality: "
        f"{modality}"
    )

    files = modality_file_table(
        file_map,
        case_ids,
        modality,
    )

    (
        features,
        reader_metadata,
        first_pass_qc,
    ) = determine_common_features(
        files,
        reader,
        modality,
    )

    print(
        f"  Shared features: "
        f"{len(features):,}"
    )

    feature_index = pd.Index(
        features,
        dtype=str,
    )

    matrix = (
        np.lib.format.open_memmap(
            matrix_output,
            mode="w+",
            dtype=np.float32,
            shape=(
                len(case_ids),
                len(features),
            ),
        )
    )

    sample_rows = []

    print(
        "  Second pass: "
        "writing aligned matrix..."
    )

    for row_index, row in (
        files.iterrows()
    ):
        case_id = row[
            "case_id"
        ]

        path = resolve_local_path(
            row[
                "local_path"
            ]
        )

        series, metadata = reader(
            path
        )

        aligned = (
            series.reindex(
                feature_index
            )
            .to_numpy(
                dtype=np.float32
            )
        )

        matrix[
            row_index,
            :,
        ] = aligned

        finite = np.isfinite(
            aligned
        )

        sample_rows.append(
            {
                "case_id": (
                    case_id
                ),
                "modality": (
                    modality
                ),
                "n_features": (
                    len(features)
                ),
                "n_finite": int(
                    finite.sum()
                ),
                "missing_fraction": (
                    1.0
                    - finite.mean()
                ),
                "mean": (
                    float(
                        np.nanmean(
                            aligned
                        )
                    )
                    if finite.any()
                    else np.nan
                ),
                "sd": (
                    float(
                        np.nanstd(
                            aligned,
                            ddof=1,
                        )
                    )
                    if finite.sum() > 1
                    else np.nan
                ),
                "file_name": (
                    path.name
                ),
            }
        )

        if (
            (row_index + 1) % 10
            == 0
            or (
                row_index + 1
                == files.shape[0]
            )
        ):
            print(
                f"    wrote "
                f"{row_index + 1}/"
                f"{files.shape[0]}"
            )

    matrix.flush()
    del matrix

    feature_table = pd.DataFrame(
        {
            "feature_index": np.arange(
                len(features),
                dtype=int,
            ),
            "feature": features,
        }
    )

    feature_table.to_csv(
        features_output,
        index=False,
    )

    sample_qc = pd.DataFrame(
        sample_rows
    )

    sample_qc.to_csv(
        sample_qc_output,
        index=False,
    )

    feature_qc = write_feature_qc(
        matrix_output,
        features,
        feature_qc_output,
    )

    return {
        "modality": modality,
        "n_cases": len(
            case_ids
        ),
        "n_features": len(
            features
        ),
        "matrix_path": (
            relative_path(
                matrix_output
            )
        ),
        "matrix_sha256": (
            sha256_file(
                matrix_output
            )
        ),
        "features_path": (
            relative_path(
                features_output
            )
        ),
        "reader_metadata": (
            reader_metadata
        ),
        "mean_sample_missing_fraction": (
            float(
                sample_qc[
                    "missing_fraction"
                ].mean()
            )
        ),
        "max_sample_missing_fraction": (
            float(
                sample_qc[
                    "missing_fraction"
                ].max()
            )
        ),
        "n_zero_or_undefined_variance_features": int(
            feature_qc[
                "zero_or_undefined_variance"
            ].sum()
        ),
        "n_features_with_any_missing": int(
            feature_qc[
                "missing_fraction"
            ].gt(0).sum()
        ),
        "first_pass_min_features": int(
            first_pass_qc[
                "n_features_in_file"
            ].min()
        ),
        "first_pass_max_features": int(
            first_pass_qc[
                "n_features_in_file"
            ].max()
        ),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 88)
    print(
        "Paper 5 - build TARGET-OS "
        "multimodal matrices"
    )
    print("=" * 88)

    print(
        f"Script version: "
        f"{SCRIPT_VERSION}"
    )

    print(
        f"Project root: "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Output directory: "
        f"{MODEL_INPUT_DIR}"
    )

    print()

    case_ids = load_primary_cases()

    print(
        "Primary patient-matched "
        "TARGET cohort:"
    )

    print(
        f"  RNA + methylation + CNV: "
        f"{len(case_ids)} cases"
    )

    case_table = pd.DataFrame(
        {
            "sample_index": np.arange(
                len(case_ids),
                dtype=int,
            ),
            "case_id": case_ids,
        }
    )

    case_table.to_csv(
        OUTPUT_CASE_ORDER,
        index=False,
    )

    file_map, selection_audit = (
        build_file_map(
            case_ids
        )
    )

    file_map.to_csv(
        OUTPUT_FILE_MAP,
        index=False,
    )

    selection_audit.to_csv(
        OUTPUT_FILE_SELECTION_AUDIT,
        index=False,
    )

    print()
    print(
        "Selected molecular files:"
    )

    print(
        file_map.groupby(
            "modality"
        )[
            "case_id"
        ]
        .nunique()
        .to_string()
    )

    duplicate_selections = (
        selection_audit[
            "n_candidates"
        ].astype(int).gt(1)
    )

    print(
        "Case/modality mappings with "
        "multiple candidate files: "
        f"{int(duplicate_selections.sum())}"
    )

    results = []

    results.append(
        build_modality_matrix(
            case_ids=case_ids,
            file_map=file_map,
            modality="rna",
            reader=read_rna_series,
            matrix_output=OUTPUT_RNA_MATRIX,
            features_output=OUTPUT_RNA_FEATURES,
            sample_qc_output=OUTPUT_RNA_SAMPLE_QC,
            feature_qc_output=OUTPUT_RNA_FEATURE_QC,
        )
    )

    results.append(
        build_modality_matrix(
            case_ids=case_ids,
            file_map=file_map,
            modality="methylation",
            reader=read_methylation_series,
            matrix_output=OUTPUT_METH_MATRIX,
            features_output=OUTPUT_METH_FEATURES,
            sample_qc_output=OUTPUT_METH_SAMPLE_QC,
            feature_qc_output=OUTPUT_METH_FEATURE_QC,
        )
    )

    results.append(
        build_modality_matrix(
            case_ids=case_ids,
            file_map=file_map,
            modality="cnv_gene",
            reader=read_cnv_series,
            matrix_output=OUTPUT_CNV_MATRIX,
            features_output=OUTPUT_CNV_FEATURES,
            sample_qc_output=OUTPUT_CNV_SAMPLE_QC,
            feature_qc_output=OUTPUT_CNV_FEATURE_QC,
        )
    )

    summary = pd.DataFrame(
        [
            {
                "modality": result[
                    "modality"
                ],
                "n_cases": result[
                    "n_cases"
                ],
                "n_features": result[
                    "n_features"
                ],
                "mean_sample_missing_fraction": result[
                    "mean_sample_missing_fraction"
                ],
                "max_sample_missing_fraction": result[
                    "max_sample_missing_fraction"
                ],
                "n_features_with_any_missing": result[
                    "n_features_with_any_missing"
                ],
                "n_zero_or_undefined_variance_features": result[
                    "n_zero_or_undefined_variance_features"
                ],
                "matrix_path": result[
                    "matrix_path"
                ],
            }
            for result in results
        ]
    )

    summary.to_csv(
        OUTPUT_MODALITY_SUMMARY,
        index=False,
    )

    print()
    print("-" * 88)
    print(
        "TARGET multimodal matrix summary"
    )
    print("-" * 88)

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        180,
    ):
        print(
            summary.to_string(
                index=False
            )
        )

    # All three matrices must use identical case order.
    for result in results:
        if result[
            "n_cases"
        ] != len(case_ids):
            raise RuntimeError(
                "Modality case-count mismatch."
            )

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "outcome_loaded": False,
        "outcome_guided_case_selection": False,
        "outcome_guided_feature_selection": False,
        "model_fitting_performed": False,
        "primary_cohort_definition": {
            "scope": PRIMARY_SCOPE,
            "combination": (
                PRIMARY_COMBINATION
            ),
            "n_cases": len(
                case_ids
            ),
            "case_order_file": (
                relative_path(
                    OUTPUT_CASE_ORDER
                )
            ),
        },
        "processing_contract": {
            "rna": {
                "preferred_value": (
                    "tpm_unstranded"
                ),
                "transform": (
                    "log2(value + 1)"
                ),
                "duplicate_gene_rule": (
                    "mean"
                ),
                "feature_universe": (
                    "intersection across all "
                    "primary cases"
                ),
            },
            "methylation": {
                "value": "Beta Value",
                "valid_range": [
                    0.0,
                    1.0,
                ],
                "duplicate_probe_rule": (
                    "mean"
                ),
                "feature_universe": (
                    "intersection across all "
                    "primary cases"
                ),
                "missing_values": (
                    "retained for later "
                    "outcome-blind imputation"
                ),
            },
            "cnv_gene": {
                "value": "copy_number",
                "duplicate_gene_rule": (
                    "median"
                ),
                "feature_universe": (
                    "intersection across all "
                    "primary cases"
                ),
            },
        },
        "guardrails": [
            (
                "Clinical outcomes are not "
                "loaded."
            ),
            (
                "The 78-case cohort is defined "
                "only by primary-tumor molecular "
                "modality availability."
            ),
            (
                "No feature is selected using "
                "survival, metastasis, response, "
                "or treatment data."
            ),
            (
                "No variance filtering or "
                "representation-learning model "
                "is performed in this script."
            ),
            (
                "All modalities use the exact "
                "same frozen case order."
            ),
            (
                "Methylation remains in CpG "
                "probe space at this stage."
            ),
        ],
        "modalities": {
            result["modality"]: result
            for result in results
        },
        "outputs": {
            "case_order": (
                relative_path(
                    OUTPUT_CASE_ORDER
                )
            ),
            "file_map": (
                relative_path(
                    OUTPUT_FILE_MAP
                )
            ),
            "file_selection_audit": (
                relative_path(
                    OUTPUT_FILE_SELECTION_AUDIT
                )
            ),
            "summary": (
                relative_path(
                    OUTPUT_MODALITY_SUMMARY
                )
            ),
        },
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
        "TARGET-OS patient-matched multimodal matrices",
        "=" * 68,
        "",
        f"Script version: {SCRIPT_VERSION}",
        f"Created UTC: {manifest['created_at_utc']}",
        "",
        "Primary cohort",
        "--------------",
        (
            f"{len(case_ids)} primary TARGET-OS "
            "cases with RNA, methylation, "
            "and gene-level CNV."
        ),
        "",
        "Scientific state",
        "----------------",
        "outcome_loaded: false",
        "outcome-guided case selection: false",
        "outcome-guided feature selection: false",
        "model fitting: false",
        "",
        "Matrices",
        "--------",
    ]

    for result in results:
        readme_lines.extend(
            [
                (
                    f"{result['modality']}: "
                    f"{result['n_cases']} x "
                    f"{result['n_features']}"
                ),
                (
                    f"  {result['matrix_path']}"
                ),
            ]
        )

    readme_lines.extend(
        [
            "",
            "Important",
            "---------",
            (
                "These are full aligned molecular "
                "matrices, not final ML features."
            ),
            (
                "Outcome-blind missing-value handling, "
                "variance filtering, scaling, and "
                "representation-learning feature "
                "construction are deferred to the "
                "next analysis stage."
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
        "TARGET multimodal matrix "
        "construction: PASS"
    )
    print("=" * 88)

    print(
        f"Cases: {len(case_ids)}"
    )

    for result in results:
        print(
            f"{result['modality']}: "
            f"{result['n_cases']} x "
            f"{result['n_features']:,}"
        )

    print()
    print(
        f"Manifest: "
        f"{relative_path(OUTPUT_MANIFEST)}"
    )

    print("=" * 88)


if __name__ == "__main__":
    main()
