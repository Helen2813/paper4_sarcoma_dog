from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "02-build-multimodal-patient-overlap-v2"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

DATA_MANIFEST_DIR = PAPER5_ROOT / "data_manifest"
RESULTS_DIR = PAPER5_ROOT / "results"

DATA_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_ROOT = PAPER5_ROOT / "human" / "TARGET_OS"

DOG2_EXPRESSION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "GSE238110_DOG2_expression_log2cpm_matched_allgenes.csv"
)

DOG2_CLINICAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "GSE238110_DOG2_clinical_matched_indexed.csv"
)

OUTPUT_TARGET_FILES = (
    DATA_MANIFEST_DIR / "TARGET_OS_gdc_file_inventory.csv"
)
OUTPUT_TARGET_CASES = (
    DATA_MANIFEST_DIR / "TARGET_OS_case_modality_availability.csv"
)
OUTPUT_TARGET_SAMPLES = (
    DATA_MANIFEST_DIR / "TARGET_OS_sample_modality_availability.csv"
)
OUTPUT_TARGET_OVERLAP = (
    DATA_MANIFEST_DIR / "TARGET_OS_modality_overlap_summary.csv"
)
OUTPUT_TARGET_COMBINATIONS = (
    DATA_MANIFEST_DIR / "TARGET_OS_multimodal_case_combinations.csv"
)
OUTPUT_TARGET_MAPPING_AUDIT = (
    DATA_MANIFEST_DIR / "TARGET_OS_case_mapping_audit.csv"
)

OUTPUT_DOG2_IDS = (
    DATA_MANIFEST_DIR / "DOG2_subject_ids.csv"
)
OUTPUT_CANINE_ID_EVIDENCE = (
    DATA_MANIFEST_DIR / "canine_downloaded_subject_id_evidence.csv"
)
OUTPUT_CANINE_OVERLAP = (
    DATA_MANIFEST_DIR / "DOG2_cross_platform_subject_overlap.csv"
)
OUTPUT_DOG2_SCAN_AUDIT = (
    DATA_MANIFEST_DIR / "DOG2_subject_id_source_audit.csv"
)

OUTPUT_EXTERNAL_DATASETS = (
    DATA_MANIFEST_DIR / "external_multimodal_dataset_inventory.csv"
)
OUTPUT_MASTER_READINESS = (
    DATA_MANIFEST_DIR / "paper5_multimodal_training_readiness.csv"
)
OUTPUT_MANIFEST = (
    DATA_MANIFEST_DIR / "paper5_multimodal_overlap_manifest.json"
)
OUTPUT_README = (
    DATA_MANIFEST_DIR / "paper5_multimodal_overlap_README.txt"
)

PRIMARY_TARGET_MODALITIES = [
    "rna",
    "methylation",
    "cnv_gene",
]

SECONDARY_TARGET_MODALITIES = [
    "mutation",
    "mirna",
]

TARGET_ALL_MODALITIES = (
    PRIMARY_TARGET_MODALITIES
    + SECONDARY_TARGET_MODALITIES
)

ACCESSION_RULES = {
    "GSE295311": {
        "species": "canine",
        "modality": "GeoMx_spatial_expression",
        "role": "DOG2_paired_spatial_candidate",
    },
    "GSE149679": {
        "species": "canine_and_human",
        "modality": "DNA_methylation",
        "role": "independent_cross_species_epigenetic_validation",
    },
    "GSE244116": {
        "species": "canine",
        "modality": "single_nucleus_RNA_ATAC_multiome",
        "role": "mechanistic_multiome_annotation",
    },
    "GSE299494": {
        "species": "canine",
        "modality": "spatial_transcriptomics",
        "role": "independent_spatial_validation",
    },
}

ICDC_RULES = {
    "COTC021": {
        "species": "canine",
        "modality": "NanoString_or_ICDC_molecular",
        "role": "DOG2_cross_platform_candidate",
    },
    "COTC022": {
        "species": "canine",
        "modality": "NanoString_or_ICDC_molecular",
        "role": "DOG2_cross_platform_candidate",
    },
}

TARGET_CASE_PATTERN = re.compile(
    r"TARGET-\d{2}-[A-Z0-9]+",
    flags=re.IGNORECASE,
)

TARGET_SAMPLE_PATTERN = re.compile(
    r"TARGET-\d{2}-[A-Z0-9]+-[A-Z0-9]+",
    flags=re.IGNORECASE,
)

COTC_PATTERN = re.compile(
    r"\b(COTC0?\d{2,3})[-_ ]+(\d{3,6})",
    flags=re.IGNORECASE,
)

TEXT_SUFFIXES = {
    ".csv",
    ".tsv",
    ".txt",
}

EXCEL_SUFFIXES = {
    ".xlsx",
    ".xls",
}

GZIP_TEXT_SUFFIXES = {
    ".csv.gz",
    ".tsv.gz",
    ".txt.gz",
}

MAX_GENERAL_TEXT_SCAN_BYTES = 120 * 1024 * 1024
MAX_EXCEL_SCAN_BYTES = 120 * 1024 * 1024
MAX_HASH_BYTES = 256 * 1024 * 1024


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def normalize_column_name(value: Any) -> str:
    text = clean_text(value).lower()
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    ).strip("_")


def relative_path(path: Path | None) -> str:
    if path is None:
        return ""

    try:
        return str(
            path.resolve().relative_to(
                PROJECT_ROOT.resolve()
            )
        )
    except ValueError:
        return str(path.resolve())


def sha256_if_small(path: Path) -> str:
    if not path.exists():
        return ""

    if path.stat().st_size > MAX_HASH_BYTES:
        return ""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def list_files(root: Path) -> list[Path]:
    if not root.exists():
        return []

    return [
        path
        for path in root.rglob("*")
        if path.is_file()
    ]


def find_files_by_name(
    root: Path,
    patterns: list[str],
) -> list[Path]:
    if not root.exists():
        return []

    patterns = [
        pattern.lower()
        for pattern in patterns
    ]

    output = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        lower = path.name.lower()

        if any(
            pattern in lower
            for pattern in patterns
        ):
            output.append(path)

    return sorted(set(output))


# ---------------------------------------------------------------------
# TARGET identifier parsing
# ---------------------------------------------------------------------

def extract_target_case_ids(
    *values: Any,
) -> list[str]:
    ids: list[str] = []

    for value in values:
        text = clean_text(value).upper()

        if not text:
            continue

        matches = TARGET_CASE_PATTERN.findall(
            text
        )

        for match in matches:
            match = match.upper()

            if match not in ids:
                ids.append(match)

    return ids


def extract_target_sample_ids(
    *values: Any,
) -> list[str]:
    ids: list[str] = []

    for value in values:
        text = clean_text(value).upper()

        if not text:
            continue

        matches = TARGET_SAMPLE_PATTERN.findall(
            text
        )

        for match in matches:
            match = match.upper()

            if match not in ids:
                ids.append(match)

    return ids


def split_metadata_values(
    value: Any,
) -> list[str]:
    text = clean_text(value)

    if not text:
        return []

    parts = re.split(
        r"\s*,\s*|\s*;\s*|\s*\|\s*",
        text,
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def metadata_contains(
    value: Any,
    token: str,
) -> bool:
    token = token.lower()

    return any(
        token in part.lower()
        for part in split_metadata_values(
            value
        )
    )


# ---------------------------------------------------------------------
# GDC sample sheets
# ---------------------------------------------------------------------

def harmonize_gdc_columns(
    table: pd.DataFrame,
) -> pd.DataFrame:
    x = table.copy()

    x.columns = [
        normalize_column_name(column)
        for column in x.columns
    ]

    aliases = {
        "file_id": [
            "file_id",
            "id",
        ],
        "file_name": [
            "file_name",
            "filename",
        ],
        "data_category": [
            "data_category",
        ],
        "data_type": [
            "data_type",
        ],
        "project_id": [
            "project_id",
        ],
        "case_id": [
            "case_id",
        ],
        "sample_id": [
            "sample_id",
        ],
        "tissue_type": [
            "tissue_type",
        ],
        "tumor_descriptor": [
            "tumor_descriptor",
        ],
        "specimen_type": [
            "specimen_type",
        ],
        "preservation_method": [
            "preservation_method",
        ],
    }

    output = pd.DataFrame(
        index=x.index
    )

    for canonical, candidates in aliases.items():
        found = next(
            (
                column
                for column in candidates
                if column in x.columns
            ),
            None,
        )

        if found is None:
            output[canonical] = ""
        else:
            output[canonical] = (
                x[found]
                .map(clean_text)
            )

    return output


def load_all_gdc_sample_sheets(
    root: Path,
) -> tuple[pd.DataFrame, list[Path]]:
    paths = find_files_by_name(
        root,
        [
            "gdc_sample_sheet",
            "sample_sheet",
        ],
    )

    tables = []
    used_paths = []

    for path in paths:
        try:
            table = pd.read_csv(
                path,
                sep="\t",
                dtype=str,
                low_memory=False,
            )

            if table.empty:
                continue

            harmonized = harmonize_gdc_columns(
                table
            )

            if (
                harmonized["file_name"]
                .str.len()
                .eq(0)
                .all()
            ):
                continue

            harmonized[
                "source_sample_sheet"
            ] = relative_path(path)

            tables.append(harmonized)
            used_paths.append(path)

        except Exception as exc:
            print(
                f"Warning: could not read "
                f"{path}: {exc}"
            )

    if not tables:
        return pd.DataFrame(), used_paths

    combined = pd.concat(
        tables,
        ignore_index=True,
    )

    combined["_key"] = np.where(
        combined["file_id"].ne(""),
        combined["file_id"],
        combined["file_name"],
    )

    combined = (
        combined
        .drop_duplicates(
            "_key",
            keep="first",
        )
        .drop(columns="_key")
        .reset_index(drop=True)
    )

    return combined, used_paths


def classify_gdc_modality(
    row: pd.Series,
) -> str:
    text = " ".join(
        [
            clean_text(
                row.get("data_category")
            ).lower(),
            clean_text(
                row.get("data_type")
            ).lower(),
            clean_text(
                row.get("file_name")
            ).lower(),
        ]
    )

    file_name = clean_text(
        row.get("file_name")
    ).lower()

    if (
        "methylation beta" in text
        or "level3betas" in text
    ):
        return "methylation"

    if (
        file_name.endswith(".idat")
        or "_red.idat" in file_name
        or "_grn.idat" in file_name
        or "raw methylation" in text
    ):
        return "methylation_raw"

    if (
        "mirna" in text
        or "microrna" in text
    ):
        return "mirna"

    if (
        "gene level copy number" in text
        or "gene_level_copy_number" in text
    ):
        return "cnv_gene"

    if (
        "copy number" in text
        or "copy_number" in text
        or "ascat" in text
    ):
        return "cnv_segment"

    if (
        "masked somatic mutation" in text
        or file_name.endswith(".maf")
        or ".maf." in file_name
    ):
        return "mutation"

    if (
        "gene expression quantification" in text
        or "star_gene_counts" in text
        or "rna_seq" in text
    ):
        return "rna"

    if "clinical" in text:
        return "clinical"

    if "biospecimen" in text:
        return "biospecimen"

    return "other"


def build_local_file_lookup(
    root: Path,
) -> dict[str, list[Path]]:
    lookup: dict[
        str,
        list[Path],
    ] = defaultdict(list)

    for path in list_files(root):
        lookup[path.name].append(path)

    return lookup


def build_target_file_inventory(
    sample_sheet: pd.DataFrame,
    target_root: Path,
) -> pd.DataFrame:
    lookup = build_local_file_lookup(
        target_root
    )

    rows = []

    for _, row in sample_sheet.iterrows():
        file_name = clean_text(
            row["file_name"]
        )

        matches = lookup.get(
            file_name,
            [],
        )

        case_ids = extract_target_case_ids(
            row.get("case_id"),
            row.get("sample_id"),
            file_name,
        )

        sample_ids = extract_target_sample_ids(
            row.get("sample_id"),
            file_name,
        )

        rows.append(
            {
                **row.to_dict(),
                "modality": (
                    classify_gdc_modality(
                        row
                    )
                ),
                "parsed_case_ids": ";".join(
                    case_ids
                ),
                "n_parsed_case_ids": len(
                    case_ids
                ),
                "parsed_sample_ids": ";".join(
                    sample_ids
                ),
                "n_parsed_sample_ids": len(
                    sample_ids
                ),
                "downloaded": bool(matches),
                "n_local_matches": len(
                    matches
                ),
                "local_path": (
                    relative_path(
                        matches[0]
                    )
                    if matches
                    else ""
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# TARGET sample/case status
# ---------------------------------------------------------------------

def row_contains_tumor(
    row: pd.Series,
) -> bool:
    if metadata_contains(
        row.get("tissue_type"),
        "tumor",
    ):
        return True

    for sample_id in split_metadata_values(
        row.get("parsed_sample_ids")
    ):
        # TARGET code 01 is primary tumor in
        # the existing TARGET-OS sample barcodes.
        if re.search(
            r"-01[A-Z0-9]*$",
            sample_id,
            flags=re.IGNORECASE,
        ):
            return True

    return False


def row_contains_primary_tumor(
    row: pd.Series,
) -> bool:
    tissue_has_tumor = metadata_contains(
        row.get("tissue_type"),
        "tumor",
    )

    descriptor_has_primary = (
        metadata_contains(
            row.get("tumor_descriptor"),
            "primary",
        )
    )

    if (
        tissue_has_tumor
        and descriptor_has_primary
    ):
        return True

    for sample_id in split_metadata_values(
        row.get("parsed_sample_ids")
    ):
        if re.search(
            r"-01[A-Z0-9]*$",
            sample_id,
            flags=re.IGNORECASE,
        ):
            return True

    return False


def explode_target_file_cases(
    file_inventory: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in file_inventory.iterrows():
        case_ids = split_metadata_values(
            row["parsed_case_ids"]
        )

        if not case_ids:
            rows.append(
                {
                    **row.to_dict(),
                    "case_key": "",
                }
            )
            continue

        for case_id in case_ids:
            rows.append(
                {
                    **row.to_dict(),
                    "case_key": (
                        case_id.upper()
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_target_case_availability(
    file_inventory: pd.DataFrame,
) -> pd.DataFrame:
    x = explode_target_file_cases(
        file_inventory
    )

    x = x[
        x["downloaded"].eq(True)
        & x["case_key"].ne("")
    ].copy()

    x["contains_tumor"] = x.apply(
        row_contains_tumor,
        axis=1,
    )

    x["contains_primary_tumor"] = (
        x.apply(
            row_contains_primary_tumor,
            axis=1,
        )
    )

    rows = []

    for case_key, part in x.groupby(
        "case_key"
    ):
        modalities_any = set(
            part["modality"]
        )

        modalities_tumor = set(
            part.loc[
                part["contains_tumor"],
                "modality",
            ]
        )

        modalities_primary = set(
            part.loc[
                part[
                    "contains_primary_tumor"
                ],
                "modality",
            ]
        )

        record = {
            "case_key": case_key,
            "n_files": int(
                part.shape[0]
            ),
            "sample_ids": ";".join(
                sorted(
                    {
                        sample_id
                        for value in part[
                            "parsed_sample_ids"
                        ]
                        for sample_id in split_metadata_values(
                            value
                        )
                    }
                )
            ),
            "tissue_types": ";".join(
                sorted(
                    {
                        clean_text(value)
                        for value in part[
                            "tissue_type"
                        ]
                        if clean_text(value)
                    }
                )
            ),
            "tumor_descriptors": ";".join(
                sorted(
                    {
                        clean_text(value)
                        for value in part[
                            "tumor_descriptor"
                        ]
                        if clean_text(value)
                    }
                )
            ),
            "modalities_any_sample": ";".join(
                sorted(
                    modalities_any
                )
            ),
            "modalities_tumor": ";".join(
                sorted(
                    modalities_tumor
                )
            ),
            "modalities_primary_tumor": (
                ";".join(
                    sorted(
                        modalities_primary
                    )
                )
            ),
        }

        for modality in TARGET_ALL_MODALITIES:
            record[
                f"has_{modality}"
            ] = (
                modality
                in modalities_any
            )

            record[
                f"tumor_has_{modality}"
            ] = (
                modality
                in modalities_tumor
            )

            record[
                f"primary_has_{modality}"
            ] = (
                modality
                in modalities_primary
            )

        rows.append(record)

    return (
        pd.DataFrame(rows)
        .sort_values("case_key")
        .reset_index(drop=True)
    )


def build_target_sample_availability(
    file_inventory: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in file_inventory.iterrows():
        if not bool(
            row["downloaded"]
        ):
            continue

        case_ids = split_metadata_values(
            row["parsed_case_ids"]
        )

        sample_ids = split_metadata_values(
            row["parsed_sample_ids"]
        )

        rows.append(
            {
                "file_name": row["file_name"],
                "modality": row["modality"],
                "case_ids": ";".join(
                    case_ids
                ),
                "sample_ids": ";".join(
                    sample_ids
                ),
                "tissue_type": (
                    row["tissue_type"]
                ),
                "tumor_descriptor": (
                    row["tumor_descriptor"]
                ),
                "contains_tumor": (
                    row_contains_tumor(
                        row
                    )
                ),
                "contains_primary_tumor": (
                    row_contains_primary_tumor(
                        row
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def build_target_mapping_audit(
    file_inventory: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for modality, part in (
        file_inventory.groupby(
            "modality"
        )
    ):
        downloaded = part[
            part["downloaded"].eq(
                True
            )
        ]

        rows.append(
            {
                "modality": modality,
                "n_files": int(
                    part.shape[0]
                ),
                "n_downloaded": int(
                    downloaded.shape[0]
                ),
                "n_with_case_id": int(
                    downloaded[
                        "n_parsed_case_ids"
                    ].gt(0).sum()
                ),
                "n_with_multiple_case_ids": int(
                    downloaded[
                        "n_parsed_case_ids"
                    ].gt(1).sum()
                ),
                "n_contains_tumor": int(
                    downloaded.apply(
                        row_contains_tumor,
                        axis=1,
                    ).sum()
                ),
                "n_contains_primary_tumor": int(
                    downloaded.apply(
                        row_contains_primary_tumor,
                        axis=1,
                    ).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_target_overlap_summary(
    case_table: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    scopes = {
        "any_sample": "",
        "tumor": "tumor_",
        "primary_tumor": "primary_",
    }

    for scope, prefix in scopes.items():
        sets = {}

        for modality in TARGET_ALL_MODALITIES:
            column = (
                f"{prefix}has_{modality}"
            )

            sets[modality] = set(
                case_table.loc[
                    case_table[
                        column
                    ].fillna(False),
                    "case_key",
                ]
            )

        for i, left in enumerate(
            TARGET_ALL_MODALITIES
        ):
            for right in TARGET_ALL_MODALITIES[
                i + 1:
            ]:
                a = sets[left]
                b = sets[right]

                rows.append(
                    {
                        "scope": scope,
                        "modality_a": left,
                        "modality_b": right,
                        "n_a": len(a),
                        "n_b": len(b),
                        "n_intersection": len(
                            a & b
                        ),
                        "n_union": len(
                            a | b
                        ),
                        "jaccard": (
                            len(a & b)
                            / len(a | b)
                            if a | b
                            else np.nan
                        ),
                    }
                )

    return pd.DataFrame(rows)


def build_target_combination_summary(
    case_table: pd.DataFrame,
) -> pd.DataFrame:
    definitions = [
        (
            "RNA",
            ["rna"],
        ),
        (
            "RNA + methylation",
            [
                "rna",
                "methylation",
            ],
        ),
        (
            "RNA + CNV",
            [
                "rna",
                "cnv_gene",
            ],
        ),
        (
            "RNA + methylation + CNV",
            [
                "rna",
                "methylation",
                "cnv_gene",
            ],
        ),
        (
            (
                "RNA + methylation + CNV "
                "+ mutation"
            ),
            [
                "rna",
                "methylation",
                "cnv_gene",
                "mutation",
            ],
        ),
        (
            (
                "RNA + methylation + CNV "
                "+ mutation + miRNA"
            ),
            [
                "rna",
                "methylation",
                "cnv_gene",
                "mutation",
                "mirna",
            ],
        ),
    ]

    scopes = {
        "any_sample": "",
        "tumor": "tumor_",
        "primary_tumor": "primary_",
    }

    rows = []

    for scope, prefix in scopes.items():
        for label, modalities in definitions:
            mask = np.ones(
                case_table.shape[0],
                dtype=bool,
            )

            for modality in modalities:
                column = (
                    f"{prefix}has_{modality}"
                )

                mask &= (
                    case_table[column]
                    .fillna(False)
                    .to_numpy(dtype=bool)
                )

            ids = (
                case_table.loc[
                    mask,
                    "case_key",
                ]
                .astype(str)
                .tolist()
            )

            rows.append(
                {
                    "scope": scope,
                    "combination": label,
                    "modalities": ";".join(
                        modalities
                    ),
                    "n_cases": len(ids),
                    "case_ids": ";".join(
                        ids
                    ),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# COTC parsing
# ---------------------------------------------------------------------

def normalize_cotc_id(
    study: str,
    subject: str,
) -> str:
    study = study.upper()

    digits = re.sub(
        r"\D",
        "",
        study[4:],
    )

    if not digits:
        return ""

    if len(digits) == 2:
        digits = (
            "0" + digits
        )

    subject = re.sub(
        r"\D",
        "",
        subject,
    )

    if not subject:
        return ""

    return (
        f"COTC{digits}-{subject}"
    )


def extract_cotc_ids_from_text(
    value: Any,
) -> set[str]:
    text = clean_text(value)

    if not text:
        return set()

    output = set()

    for match in COTC_PATTERN.finditer(
        text
    ):
        normalized = normalize_cotc_id(
            match.group(1),
            match.group(2),
        )

        if normalized:
            output.add(
                normalized
            )

    return output


def read_text_file(
    path: Path,
) -> str:
    lower = path.name.lower()

    if lower.endswith(".gz"):
        with gzip.open(
            path,
            "rt",
            encoding="utf-8",
            errors="ignore",
        ) as handle:
            return handle.read()

    return path.read_text(
        encoding="utf-8",
        errors="ignore",
    )


def scan_table_for_cotc_ids(
    path: Path,
) -> set[str]:
    output = set()

    try:
        if path.suffix.lower() in {
            ".xlsx",
            ".xls",
        }:
            sheets = pd.read_excel(
                path,
                sheet_name=None,
                dtype=str,
            )

            for table in sheets.values():
                for column in table.columns:
                    output.update(
                        extract_cotc_ids_from_text(
                            column
                        )
                    )

                    for value in (
                        table[column]
                        .dropna()
                        .astype(str)
                    ):
                        output.update(
                            extract_cotc_ids_from_text(
                                value
                            )
                        )

            return output

        separator = (
            "\t"
            if (
                ".tsv" in path.name.lower()
                or ".txt" in path.name.lower()
            )
            else ","
        )

        table = pd.read_csv(
            path,
            sep=separator,
            dtype=str,
            low_memory=False,
        )

        # Scan every metadata column, not only the first few.
        for column in table.columns:
            output.update(
                extract_cotc_ids_from_text(
                    column
                )
            )

            values = (
                table[column]
                .dropna()
                .astype(str)
            )

            for value in values:
                output.update(
                    extract_cotc_ids_from_text(
                        value
                    )
                )

    except Exception:
        try:
            if (
                path.stat().st_size
                <= MAX_GENERAL_TEXT_SCAN_BYTES
            ):
                output.update(
                    extract_cotc_ids_from_text(
                        read_text_file(
                            path
                        )
                    )
                )
        except Exception:
            pass

    return output


def candidate_dog2_source_files() -> list[Path]:
    candidates = set()

    if DOG2_EXPRESSION_FILE.exists():
        candidates.add(
            DOG2_EXPRESSION_FILE
        )

    if DOG2_CLINICAL_FILE.exists():
        candidates.add(
            DOG2_CLINICAL_FILE
        )

    search_roots = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "results",
    ]

    tokens = [
        "gse238110",
        "dog2",
        "cotc021",
        "cotc022",
    ]

    for root in search_roots:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            lower = str(path).lower()

            if not any(
                token in lower
                for token in tokens
            ):
                continue

            if (
                path.stat().st_size
                <= MAX_GENERAL_TEXT_SCAN_BYTES
            ):
                candidates.add(
                    path
                )

    return sorted(
        candidates
    )


def extract_dog2_subject_ids(
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    evidence: dict[
        str,
        set[str],
    ] = defaultdict(set)

    audit_rows = []

    for path in candidate_dog2_source_files():
        suffix = path.suffix.lower()
        ids = set()

        if (
            suffix in TEXT_SUFFIXES
            or path.name.lower().endswith(
                tuple(
                    GZIP_TEXT_SUFFIXES
                )
            )
            or suffix in EXCEL_SUFFIXES
        ):
            ids = scan_table_for_cotc_ids(
                path
            )

        if not ids:
            # Path names can also contain COTC IDs.
            ids.update(
                extract_cotc_ids_from_text(
                    str(path)
                )
            )

        audit_rows.append(
            {
                "source_file": (
                    relative_path(path)
                ),
                "size_bytes": int(
                    path.stat().st_size
                ),
                "n_cotc_ids": len(ids),
                "cotc_ids": ";".join(
                    sorted(ids)
                ),
            }
        )

        for subject_id in ids:
            evidence[
                subject_id
            ].add(
                relative_path(path)
            )

    ids_table = pd.DataFrame(
        [
            {
                "dog2_subject_id": subject_id,
                "evidence_sources": ";".join(
                    sorted(sources)
                ),
            }
            for subject_id, sources in sorted(
                evidence.items()
            )
        ]
    )

    audit = pd.DataFrame(
        audit_rows
    )

    return ids_table, audit


# ---------------------------------------------------------------------
# Downloaded canine data
# ---------------------------------------------------------------------

def infer_dataset_label(
    path: Path,
) -> str:
    text = str(path).upper()

    for accession in ACCESSION_RULES:
        if accession in text:
            return accession

    for cotc in ICDC_RULES:
        if cotc in text:
            return cotc

    if "GEOMX" in text:
        return "GSE295311"

    if "MULTIOME" in text:
        return "GSE244116"

    if "SPATIAL" in text:
        return "GSE299494"

    return "UNCLASSIFIED"


def scan_downloaded_file_for_cotc_ids(
    path: Path,
) -> set[str]:
    output = set()

    output.update(
        extract_cotc_ids_from_text(
            str(path)
        )
    )

    lower = path.name.lower()

    try:
        if (
            path.suffix.lower()
            in EXCEL_SUFFIXES
        ):
            if (
                path.stat().st_size
                <= MAX_EXCEL_SCAN_BYTES
            ):
                output.update(
                    scan_table_for_cotc_ids(
                        path
                    )
                )

        elif (
            path.suffix.lower()
            in TEXT_SUFFIXES
            or lower.endswith(
                tuple(
                    GZIP_TEXT_SUFFIXES
                )
            )
        ):
            if (
                path.stat().st_size
                <= MAX_GENERAL_TEXT_SCAN_BYTES
            ):
                output.update(
                    scan_table_for_cotc_ids(
                        path
                    )
                )

    except Exception:
        pass

    return output


def scan_paper5_for_canine_ids(
) -> pd.DataFrame:
    rows = []

    for path in list_files(
        PAPER5_ROOT
    ):
        if (
            "TARGET_OS"
            in str(path).upper()
        ):
            continue

        dataset = infer_dataset_label(
            path
        )

        ids = (
            scan_downloaded_file_for_cotc_ids(
                path
            )
        )

        for subject_id in sorted(ids):
            rows.append(
                {
                    "dataset": dataset,
                    "subject_id": (
                        subject_id
                    ),
                    "source_file": (
                        relative_path(
                            path
                        )
                    ),
                    "source_file_size": int(
                        path.stat().st_size
                    ),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "subject_id",
                "source_file",
                "source_file_size",
            ]
        )

    return (
        pd.DataFrame(rows)
        .drop_duplicates()
        .sort_values(
            [
                "dataset",
                "subject_id",
                "source_file",
            ]
        )
        .reset_index(drop=True)
    )


def build_canine_overlap(
    dog2_ids: pd.DataFrame,
    downloaded_ids: pd.DataFrame,
) -> pd.DataFrame:
    if dog2_ids.empty:
        return pd.DataFrame()

    dog2_set = set(
        dog2_ids[
            "dog2_subject_id"
        ].astype(str)
    )

    datasets = sorted(
        set(
            downloaded_ids[
                "dataset"
            ].astype(str)
        )
        - {
            "UNCLASSIFIED",
        }
    )

    rows = []

    for dataset in datasets:
        observed = set(
            downloaded_ids.loc[
                downloaded_ids[
                    "dataset"
                ].eq(dataset),
                "subject_id",
            ].astype(str)
        )

        overlap = sorted(
            dog2_set
            & observed
        )

        rows.append(
            {
                "dataset": dataset,
                "n_dog2_subjects": len(
                    dog2_set
                ),
                "n_downloaded_subject_ids": len(
                    observed
                ),
                "n_exact_patient_overlap": len(
                    overlap
                ),
                "overlap_subject_ids": ";".join(
                    overlap
                ),
                "interpretation": (
                    "paired_with_DOG2"
                    if overlap
                    else (
                        "no_exact_COTC_"
                        "overlap_detected"
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# External dataset inventory
# ---------------------------------------------------------------------

def accession_file_summary(
    accession: str,
) -> dict[str, Any]:
    matches = [
        path
        for path in list_files(
            PAPER5_ROOT
        )
        if accession.upper()
        in str(path).upper()
    ]

    return {
        "n_files": len(matches),
        "total_bytes": int(
            sum(
                path.stat().st_size
                for path in matches
            )
        ),
        "example_paths": ";".join(
            relative_path(path)
            for path in matches[:5]
        ),
    }


def build_external_dataset_inventory(
    canine_overlap: pd.DataFrame,
) -> pd.DataFrame:
    overlap_map = {}

    if not canine_overlap.empty:
        overlap_map = (
            canine_overlap
            .set_index("dataset")
            ["n_exact_patient_overlap"]
            .to_dict()
        )

    rows = []

    specifications = {
        **ACCESSION_RULES,
        **ICDC_RULES,
    }

    for dataset, specification in (
        specifications.items()
    ):
        summary = (
            accession_file_summary(
                dataset
            )
        )

        rows.append(
            {
                "dataset": dataset,
                **specification,
                **summary,
                "dog2_patient_overlap": int(
                    overlap_map.get(
                        dataset,
                        0,
                    )
                ),
                "download_detected": bool(
                    summary["n_files"] > 0
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------

def combination_n(
    combinations: pd.DataFrame,
    scope: str,
    label: str,
) -> int:
    part = combinations[
        combinations["scope"].eq(
            scope
        )
        & combinations[
            "combination"
        ].eq(label)
    ]

    if part.empty:
        return 0

    return int(
        part.iloc[0]["n_cases"]
    )


def build_training_readiness(
    target_combinations: pd.DataFrame,
    canine_overlap: pd.DataFrame,
    external_inventory: pd.DataFrame,
) -> pd.DataFrame:
    target_three = combination_n(
        target_combinations,
        "primary_tumor",
        "RNA + methylation + CNV",
    )

    target_four = combination_n(
        target_combinations,
        "primary_tumor",
        (
            "RNA + methylation + CNV "
            "+ mutation"
        ),
    )

    target_five = combination_n(
        target_combinations,
        "primary_tumor",
        (
            "RNA + methylation + CNV "
            "+ mutation + miRNA"
        ),
    )

    dog2_geomx = 0

    if not canine_overlap.empty:
        part = canine_overlap[
            canine_overlap[
                "dataset"
            ].eq("GSE295311")
        ]

        if not part.empty:
            dog2_geomx = int(
                part.iloc[0][
                    "n_exact_patient_overlap"
                ]
            )

    downloaded = set(
        external_inventory.loc[
            external_inventory[
                "download_detected"
            ].eq(True),
            "dataset",
        ]
    )

    return pd.DataFrame(
        [
            {
                "analysis_component": (
                    "TARGET primary-tumor "
                    "RNA+methylation+CNV"
                ),
                "n_paired_subjects": (
                    target_three
                ),
                "ready": (
                    target_three >= 30
                ),
                "recommended_role": (
                    "primary human multi-omics "
                    "representation-learning block"
                ),
            },
            {
                "analysis_component": (
                    "TARGET primary-tumor "
                    "RNA+methylation+CNV+mutation"
                ),
                "n_paired_subjects": (
                    target_four
                ),
                "ready": (
                    target_four >= 30
                ),
                "recommended_role": (
                    "extended human multi-omics "
                    "sensitivity analysis"
                ),
            },
            {
                "analysis_component": (
                    "TARGET primary-tumor "
                    "five-modality block"
                ),
                "n_paired_subjects": (
                    target_five
                ),
                "ready": (
                    target_five >= 30
                ),
                "recommended_role": (
                    "use only if miRNA files "
                    "are actually available"
                ),
            },
            {
                "analysis_component": (
                    "DOG2 + GSE295311 GeoMx"
                ),
                "n_paired_subjects": (
                    dog2_geomx
                ),
                "ready": (
                    dog2_geomx > 0
                ),
                "recommended_role": (
                    "paired canine spatial "
                    "validation"
                ),
            },
            {
                "analysis_component": (
                    "GSE149679 dog-human "
                    "methylation"
                ),
                "n_paired_subjects": np.nan,
                "ready": (
                    "GSE149679"
                    in downloaded
                ),
                "recommended_role": (
                    "independent cross-species "
                    "epigenetic validation"
                ),
            },
            {
                "analysis_component": (
                    "GSE244116 canine RNA+ATAC"
                ),
                "n_paired_subjects": np.nan,
                "ready": (
                    "GSE244116"
                    in downloaded
                ),
                "recommended_role": (
                    "mechanistic chromatin "
                    "annotation"
                ),
            },
            {
                "analysis_component": (
                    "GSE299494 canine spatial"
                ),
                "n_paired_subjects": np.nan,
                "ready": (
                    "GSE299494"
                    in downloaded
                ),
                "recommended_role": (
                    "independent spatial "
                    "validation"
                ),
            },
        ]
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 88)
    print(
        "Paper 5 - multimodal patient "
        "overlap audit"
    )
    print("=" * 88)
    print(
        f"Script version: "
        f"{SCRIPT_VERSION}"
    )
    print(
        f"Project root: {PROJECT_ROOT}"
    )
    print(
        f"Paper 5 root: {PAPER5_ROOT}"
    )
    print()

    # -------------------------------------------------------------
    # TARGET
    # -------------------------------------------------------------

    print("-" * 88)
    print(
        "TARGET-OS: loading GDC sample sheets"
    )
    print("-" * 88)

    if not TARGET_ROOT.exists():
        raise FileNotFoundError(
            f"TARGET root not found: "
            f"{TARGET_ROOT}"
        )

    sample_sheet, sheet_paths = (
        load_all_gdc_sample_sheets(
            TARGET_ROOT
        )
    )

    if sample_sheet.empty:
        raise RuntimeError(
            "No GDC sample sheets found."
        )

    print(
        f"Sample sheets loaded: "
        f"{len(sheet_paths)}"
    )
    print(
        f"Unique GDC file records: "
        f"{sample_sheet.shape[0]}"
    )

    target_files = (
        build_target_file_inventory(
            sample_sheet,
            TARGET_ROOT,
        )
    )

    target_files.to_csv(
        OUTPUT_TARGET_FILES,
        index=False,
    )

    completeness = (
        target_files.groupby(
            "modality"
        )
        .agg(
            expected_files=(
                "file_name",
                "size",
            ),
            downloaded_files=(
                "downloaded",
                "sum",
            ),
        )
        .reset_index()
    )

    print()
    print(
        "TARGET-OS downloaded file "
        "completeness:"
    )
    print(
        completeness.to_string(
            index=False
        )
    )

    mapping_audit = (
        build_target_mapping_audit(
            target_files
        )
    )

    mapping_audit.to_csv(
        OUTPUT_TARGET_MAPPING_AUDIT,
        index=False,
    )

    print()
    print(
        "TARGET-OS identifier / tumor "
        "mapping audit:"
    )
    print(
        mapping_audit.to_string(
            index=False
        )
    )

    target_samples = (
        build_target_sample_availability(
            target_files
        )
    )

    target_samples.to_csv(
        OUTPUT_TARGET_SAMPLES,
        index=False,
    )

    target_cases = (
        build_target_case_availability(
            target_files
        )
    )

    target_cases.to_csv(
        OUTPUT_TARGET_CASES,
        index=False,
    )

    target_overlap = (
        build_target_overlap_summary(
            target_cases
        )
    )

    target_overlap.to_csv(
        OUTPUT_TARGET_OVERLAP,
        index=False,
    )

    target_combinations = (
        build_target_combination_summary(
            target_cases
        )
    )

    target_combinations.to_csv(
        OUTPUT_TARGET_COMBINATIONS,
        index=False,
    )

    print()
    print(
        "TARGET-OS case-level "
        "multimodal combinations:"
    )

    print(
        target_combinations[
            [
                "scope",
                "combination",
                "n_cases",
            ]
        ].to_string(
            index=False
        )
    )

    # -------------------------------------------------------------
    # DOG2
    # -------------------------------------------------------------

    print()
    print("-" * 88)
    print(
        "DOG2 / downloaded canine "
        "patient-ID overlap audit"
    )
    print("-" * 88)

    dog2_ids, dog2_scan_audit = (
        extract_dog2_subject_ids()
    )

    dog2_ids.to_csv(
        OUTPUT_DOG2_IDS,
        index=False,
    )

    dog2_scan_audit.to_csv(
        OUTPUT_DOG2_SCAN_AUDIT,
        index=False,
    )

    print(
        f"DOG2 COTC subject IDs detected: "
        f"{dog2_ids.shape[0]}"
    )

    if not dog2_scan_audit.empty:
        informative = (
            dog2_scan_audit[
                dog2_scan_audit[
                    "n_cotc_ids"
                ].gt(0)
            ]
        )

        print(
            "DOG2 files containing COTC IDs: "
            f"{informative.shape[0]}"
        )

        if not informative.empty:
            print(
                informative[
                    [
                        "source_file",
                        "n_cotc_ids",
                    ]
                ]
                .sort_values(
                    "n_cotc_ids",
                    ascending=False,
                )
                .head(20)
                .to_string(
                    index=False
                )
            )

    canine_evidence = (
        scan_paper5_for_canine_ids()
    )

    canine_evidence.to_csv(
        OUTPUT_CANINE_ID_EVIDENCE,
        index=False,
    )

    n_downloaded_ids = (
        canine_evidence[
            "subject_id"
        ].nunique()
        if not canine_evidence.empty
        else 0
    )

    print(
        "COTC subject IDs found in "
        f"downloaded Paper 5 data: "
        f"{n_downloaded_ids}"
    )

    canine_overlap = (
        build_canine_overlap(
            dog2_ids,
            canine_evidence,
        )
    )

    canine_overlap.to_csv(
        OUTPUT_CANINE_OVERLAP,
        index=False,
    )

    if not canine_overlap.empty:
        print()
        print(
            canine_overlap[
                [
                    "dataset",
                    "n_downloaded_subject_ids",
                    "n_exact_patient_overlap",
                    "overlap_subject_ids",
                ]
            ].to_string(
                index=False
            )
        )

    # -------------------------------------------------------------
    # External datasets
    # -------------------------------------------------------------

    print()
    print("-" * 88)
    print(
        "External multimodal dataset inventory"
    )
    print("-" * 88)

    external_inventory = (
        build_external_dataset_inventory(
            canine_overlap
        )
    )

    external_inventory.to_csv(
        OUTPUT_EXTERNAL_DATASETS,
        index=False,
    )

    print(
        external_inventory[
            [
                "dataset",
                "modality",
                "role",
                "n_files",
                "download_detected",
                "dog2_patient_overlap",
            ]
        ].to_string(
            index=False
        )
    )

    # -------------------------------------------------------------
    # Readiness
    # -------------------------------------------------------------

    readiness = (
        build_training_readiness(
            target_combinations,
            canine_overlap,
            external_inventory,
        )
    )

    readiness.to_csv(
        OUTPUT_MASTER_READINESS,
        index=False,
    )

    print()
    print("-" * 88)
    print(
        "Paper 5 representation-learning "
        "readiness"
    )
    print("-" * 88)

    print(
        readiness.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------
    # Manifest
    # -------------------------------------------------------------

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "project_root": str(
            PROJECT_ROOT
        ),
        "paper5_root": str(
            PAPER5_ROOT
        ),
        "outcome_loaded": False,
        "outcome_guided_subject_selection": False,
        "outcome_guided_modality_selection": False,
        "feature_selection_performed": False,
        "model_fitting_performed": False,
        "target_mapping_rules": {
            "case_identifier": (
                "Extract unique "
                "TARGET-XX-XXXX participant "
                "barcode from Case ID, "
                "Sample ID, or file name."
            ),
            "multiple_case_entries": (
                "Repeated comma-separated "
                "case IDs are deduplicated."
            ),
            "tumor_status": (
                "Use Tissue Type with "
                "TARGET sample-barcode fallback."
            ),
            "primary_tumor_status": (
                "Use Tumor Descriptor=Primary "
                "with TARGET -01 sample-code "
                "fallback."
            ),
        },
        "guardrails": [
            (
                "No target clinical outcome is "
                "loaded."
            ),
            (
                "Molecular overlap is defined "
                "from identifiers and molecular "
                "file metadata only."
            ),
            (
                "No subject is selected based "
                "on prognosis, response, or "
                "survival."
            ),
            (
                "No representation-learning "
                "model is fitted in this script."
            ),
            (
                "DOG2 cross-platform pairing "
                "requires exact normalized "
                "COTC participant identifiers."
            ),
            (
                "Independent datasets are not "
                "treated as patient-paired "
                "without identifier evidence."
            ),
        ],
        "target_os": {
            "n_sample_sheets": len(
                sheet_paths
            ),
            "n_unique_file_records": int(
                target_files.shape[0]
            ),
            "n_cases": int(
                target_cases.shape[0]
            ),
        },
        "dog2": {
            "n_detected_cotc_subject_ids": int(
                dog2_ids.shape[0]
            ),
        },
        "outputs": {
            "target_file_inventory": (
                relative_path(
                    OUTPUT_TARGET_FILES
                )
            ),
            "target_mapping_audit": (
                relative_path(
                    OUTPUT_TARGET_MAPPING_AUDIT
                )
            ),
            "target_case_availability": (
                relative_path(
                    OUTPUT_TARGET_CASES
                )
            ),
            "target_sample_availability": (
                relative_path(
                    OUTPUT_TARGET_SAMPLES
                )
            ),
            "target_overlap": (
                relative_path(
                    OUTPUT_TARGET_OVERLAP
                )
            ),
            "target_combinations": (
                relative_path(
                    OUTPUT_TARGET_COMBINATIONS
                )
            ),
            "dog2_subject_ids": (
                relative_path(
                    OUTPUT_DOG2_IDS
                )
            ),
            "dog2_scan_audit": (
                relative_path(
                    OUTPUT_DOG2_SCAN_AUDIT
                )
            ),
            "canine_id_evidence": (
                relative_path(
                    OUTPUT_CANINE_ID_EVIDENCE
                )
            ),
            "canine_overlap": (
                relative_path(
                    OUTPUT_CANINE_OVERLAP
                )
            ),
            "external_inventory": (
                relative_path(
                    OUTPUT_EXTERNAL_DATASETS
                )
            ),
            "readiness": (
                relative_path(
                    OUTPUT_MASTER_READINESS
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
        "Paper 5 multimodal patient overlap audit",
        "=" * 72,
        "",
        f"Script version: {SCRIPT_VERSION}",
        f"Created UTC: {manifest['created_at_utc']}",
        "",
        "Scientific state",
        "----------------",
        "outcome_loaded: false",
        "feature_selection_performed: false",
        "model_fitting_performed: false",
        "",
        "TARGET-OS combinations",
        "----------------------",
    ]

    for _, row in target_combinations.iterrows():
        readme_lines.append(
            f"{row['scope']} | "
            f"{row['combination']}: "
            f"{row['n_cases']} cases"
        )

    readme_lines.extend(
        [
            "",
            "Canine paired evidence",
            "----------------------",
        ]
    )

    if canine_overlap.empty:
        readme_lines.append(
            "No exact DOG2 cross-platform "
            "COTC overlap was estimable."
        )
    else:
        for _, row in canine_overlap.iterrows():
            readme_lines.append(
                f"{row['dataset']}: "
                f"{row['n_exact_patient_overlap']} "
                f"exact DOG2 overlaps"
            )

    readme_lines.extend(
        [
            "",
            "Interpretation",
            "--------------",
            (
                "TARGET-OS RNA, methylation, "
                "gene-level CNV, and mutation "
                "availability are evaluated "
                "using normalized participant IDs."
            ),
            (
                "miRNA is not assumed available "
                "unless actual downloaded files "
                "are present."
            ),
            (
                "DOG2 cross-platform pairing "
                "requires exact COTC participant "
                "identifier evidence."
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
        "Paper 5 multimodal overlap "
        "audit complete"
    )
    print("=" * 88)

    print("Outputs:")

    for path in [
        OUTPUT_TARGET_FILES,
        OUTPUT_TARGET_MAPPING_AUDIT,
        OUTPUT_TARGET_CASES,
        OUTPUT_TARGET_SAMPLES,
        OUTPUT_TARGET_OVERLAP,
        OUTPUT_TARGET_COMBINATIONS,
        OUTPUT_DOG2_IDS,
        OUTPUT_DOG2_SCAN_AUDIT,
        OUTPUT_CANINE_ID_EVIDENCE,
        OUTPUT_CANINE_OVERLAP,
        OUTPUT_EXTERNAL_DATASETS,
        OUTPUT_MASTER_READINESS,
        OUTPUT_MANIFEST,
        OUTPUT_README,
    ]:
        print(
            f"  {relative_path(path)}"
        )

    print("=" * 88)


if __name__ == "__main__":
    main()
