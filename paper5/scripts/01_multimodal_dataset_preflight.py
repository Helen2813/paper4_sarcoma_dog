from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_VERSION = "01-multimodal-dataset-preflight-v1"

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

CONFIG_DIR = PAPER5_ROOT / "configs"
DATA_MANIFEST_DIR = PAPER5_ROOT / "data_manifest"
RESULTS_DIR = PAPER5_ROOT / "results"
FIGURES_DIR = PAPER5_ROOT / "figures"

for directory in [
    CONFIG_DIR,
    DATA_MANIFEST_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

REGISTRY_FILE = CONFIG_DIR / "dataset_registry.csv"

OUTPUT_DATASET_INVENTORY = (
    DATA_MANIFEST_DIR / "dataset_inventory.csv"
)
OUTPUT_FILE_INVENTORY = (
    DATA_MANIFEST_DIR / "dataset_file_inventory.csv"
)
OUTPUT_MODALITY_OVERLAP = (
    DATA_MANIFEST_DIR / "modality_overlap.csv"
)
OUTPUT_PROGRAM_COVERAGE = (
    DATA_MANIFEST_DIR / "frozen_program_coverage.csv"
)
OUTPUT_SAMPLE_MATCHING = (
    DATA_MANIFEST_DIR / "sample_matching_audit.csv"
)
OUTPUT_READINESS = (
    DATA_MANIFEST_DIR / "representation_learning_readiness.csv"
)
OUTPUT_MANIFEST = (
    DATA_MANIFEST_DIR / "paper5_preflight_manifest.json"
)
OUTPUT_README = (
    DATA_MANIFEST_DIR / "paper5_preflight_README.txt"
)

FROZEN_WEIGHTS_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "GSE238110_frozen_transfer_gene_weights_strict.csv"
)

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]

CORE_DATASETS = {
    "GSE238110_DOG2",
    "GSE239948_CCOGC",
}

OUTCOME_FILE_TOKENS = {
    "clinical",
    "survival",
    "outcome",
    "response",
    "rfs",
    "overall_survival",
    "disease_free",
    "dfi",
    "necrosis",
    "event",
    "metastasis",
}

NON_EXPRESSION_TOKENS = {
    "clinical",
    "survival",
    "outcome",
    "response",
    "score",
    "scores",
    "mapping",
    "coverage",
    "manifest",
    "readme",
    "result",
    "results",
    "targeted",
    "metadata",
    "meta",
    "annotation",
    "weights",
    "loading",
    "loadings",
    "classification",
}

EXPRESSION_POSITIVE_TOKENS = {
    "expression",
    "matrix",
    "normalized",
    "log2",
    "log2cpm",
    "counts",
    "count",
    "rna",
}

SUPPORTED_MATRIX_SUFFIXES = (
    ".csv",
    ".csv.gz",
    ".tsv",
    ".tsv.gz",
    ".txt",
    ".txt.gz",
    ".npy",
    ".npz",
)

DEFAULT_MAX_HASH_MB = 512

# Prevent failures on very wide single-cell headers.
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10


# ---------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------

DEFAULT_DATASETS = [
    {
        "dataset_id": "GSE238110_DOG2",
        "accession": "GSE238110",
        "species": "canine",
        "modality": "bulk_rna",
        "biological_unit": "tumor",
        "current_role": "source_anchor",
        "preferred_expression_path": (
            "data/processed/"
            "GSE238110_DOG2_expression_log2cpm_matched_allgenes.csv"
        ),
        "search_tokens": "GSE238110;DOG2",
        "orientation_hint": "samples_by_features",
        "feature_space": "gene_symbol",
        "precomputed_coverage_path": "",
    },
    {
        "dataset_id": "GSE239948_CCOGC",
        "accession": "GSE239948",
        "species": "canine",
        "modality": "bulk_rna",
        "biological_unit": "tumor",
        "current_role": "external_canine_validation",
        "preferred_expression_path": (
            "data/processed/"
            "canine_validation_GSE239948_expression_log2_symbol.csv"
        ),
        "search_tokens": "GSE239948;CCOGC",
        "orientation_hint": "samples_by_features",
        "feature_space": "gene_symbol",
        "precomputed_coverage_path": "",
    },
    {
        "dataset_id": "TARGET_OS",
        "accession": "TARGET-OS",
        "species": "human",
        "modality": "bulk_rna",
        "biological_unit": "tumor",
        "current_role": "external_human_validation",
        "preferred_expression_path": "",
        "search_tokens": "TARGET_OS;TARGET-OS;TARGET",
        "orientation_hint": "auto",
        "feature_space": "gene_symbol",
        "precomputed_coverage_path": "",
    },
    {
        "dataset_id": "GSE21257",
        "accession": "GSE21257",
        "species": "human",
        "modality": "bulk_rna",
        "biological_unit": "tumor",
        "current_role": "external_human_validation",
        "preferred_expression_path": "",
        "search_tokens": "GSE21257",
        "orientation_hint": "auto",
        "feature_space": "gene_symbol",
        "precomputed_coverage_path": "",
    },
    {
        "dataset_id": "GSE39055",
        "accession": "GSE39055",
        "species": "human",
        "modality": "expression_array",
        "biological_unit": "tumor",
        "current_role": "external_human_assay_sensitive_validation",
        "preferred_expression_path": "",
        "search_tokens": "GSE39055",
        "orientation_hint": "auto",
        "feature_space": "gene_symbol_or_probe",
        "precomputed_coverage_path": "",
    },
    {
        "dataset_id": "GSE252470_AMMONS",
        "accession": "GSE252470",
        "species": "canine",
        "modality": "single_cell_rna",
        "biological_unit": "cell",
        "current_role": "biological_annotation_only",
        "preferred_expression_path": (
            "data/raw/single_cell/"
            "Ammons_GSE252470/exprMatrix.tsv.gz"
        ),
        "search_tokens": "GSE252470;Ammons",
        "orientation_hint": "features_by_samples",
        "feature_space": "gene_symbol",
        "precomputed_coverage_path": (
            "results/tables/"
            "Ammons_scRNA_score_gene_coverage.csv"
        ),
    },
]


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------

@dataclass
class MatrixInspection:
    path: Path
    format: str
    orientation: str
    n_samples: int | None
    n_features: int | None
    feature_names: set[str] | None
    sample_ids: list[str] | None
    error: str | None = None


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def clean_gene_symbol(value: Any) -> str:
    text = clean_text(value).upper()

    if not text:
        return ""

    if "|" in text:
        parts = [
            part.strip()
            for part in text.split("|")
            if part.strip()
        ]

        symbol_like = [
            part
            for part in parts
            if not part.startswith("ENS")
        ]

        if symbol_like:
            text = symbol_like[-1]
        elif parts:
            text = parts[-1]

    # Remove common version suffixes.
    text = re.sub(r"\.\d+$", "", text)

    return text


def normalize_sample_id(value: Any) -> str:
    text = clean_text(value).upper()
    return re.sub(r"[^A-Z0-9]+", "", text)


def split_tokens(value: Any) -> list[str]:
    text = clean_text(value)

    if not text:
        return []

    return [
        token.strip()
        for token in text.split(";")
        if token.strip()
    ]


def relative_path(path: Path | None) -> str:
    if path is None:
        return ""

    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def get_git_state() -> dict[str, Any]:
    result: dict[str, Any] = {
        "git_available": False,
        "commit": None,
        "branch": None,
        "dirty": None,
    }

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        result.update(
            {
                "git_available": True,
                "commit": commit,
                "branch": branch,
                "dirty": bool(status),
            }
        )
    except Exception:
        pass

    return result


def sha256_file(
    path: Path,
    max_hash_bytes: int,
    force_full_hash: bool,
) -> dict[str, Any]:
    size = path.stat().st_size

    result = {
        "path": relative_path(path),
        "size_bytes": int(size),
        "sha256": None,
        "hash_status": None,
    }

    if size > max_hash_bytes and not force_full_hash:
        result["hash_status"] = "skipped_large_file"
        return result

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    result["sha256"] = digest.hexdigest()
    result["hash_status"] = "complete"

    return result


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------

def write_default_registry(
    overwrite: bool = False,
) -> None:
    if REGISTRY_FILE.exists() and not overwrite:
        return

    pd.DataFrame(DEFAULT_DATASETS).to_csv(
        REGISTRY_FILE,
        index=False,
    )

    print(f"Created registry: {REGISTRY_FILE}")


def load_registry() -> pd.DataFrame:
    write_default_registry(overwrite=False)

    registry = pd.read_csv(
        REGISTRY_FILE,
        dtype=str,
    ).fillna("")

    required = [
        "dataset_id",
        "accession",
        "species",
        "modality",
        "biological_unit",
        "current_role",
        "preferred_expression_path",
        "search_tokens",
        "orientation_hint",
        "feature_space",
        "precomputed_coverage_path",
    ]

    missing = [
        column
        for column in required
        if column not in registry.columns
    ]

    if missing:
        raise ValueError(
            "Dataset registry is missing columns: "
            + ", ".join(missing)
        )

    if registry["dataset_id"].duplicated().any():
        duplicates = registry.loc[
            registry["dataset_id"].duplicated(),
            "dataset_id",
        ].tolist()

        raise ValueError(
            f"Duplicate dataset_id values: {duplicates}"
        )

    return registry


# ---------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------

def is_supported_matrix(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(SUPPORTED_MATRIX_SUFFIXES)


def path_contains_any(
    path: Path,
    tokens: Iterable[str],
) -> bool:
    text = str(path).lower()

    return any(
        token.lower() in text
        for token in tokens
    )


def candidate_expression_score(
    path: Path,
    search_tokens: list[str],
) -> float:
    name = path.name.lower()
    full = str(path).lower()

    if not is_supported_matrix(path):
        return -np.inf

    if ".soft" in name:
        return -np.inf

    if any(
        token in name
        for token in NON_EXPRESSION_TOKENS
    ):
        return -np.inf

    score = 0.0

    for token in search_tokens:
        if token.lower() in full:
            score += 10.0

    for token in EXPRESSION_POSITIVE_TOKENS:
        if token in name:
            score += 3.0

    if "data/processed" in full.replace("\\", "/"):
        score += 4.0

    if "data/raw" in full.replace("\\", "/"):
        score += 1.0

    if "allgenes" in name:
        score += 3.0

    if "symbol" in name:
        score += 2.0

    return score


def discover_expression_file(
    row: pd.Series,
) -> tuple[Path | None, list[dict[str, Any]]]:
    preferred = clean_text(
        row["preferred_expression_path"]
    )

    candidate_rows: list[dict[str, Any]] = []

    if preferred:
        preferred_path = PROJECT_ROOT / preferred

        if preferred_path.exists():
            candidate_rows.append(
                {
                    "path": relative_path(preferred_path),
                    "score": 1000.0,
                    "selected": True,
                    "selection_reason": "preferred_path",
                }
            )

            return preferred_path, candidate_rows

    search_tokens = split_tokens(
        row["search_tokens"]
    )

    roots = [
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "raw",
    ]

    candidates: list[tuple[float, Path]] = []

    for root in roots:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            score = candidate_expression_score(
                path,
                search_tokens,
            )

            if not np.isfinite(score):
                continue

            if score <= 0:
                continue

            candidates.append((score, path))

    candidates.sort(
        key=lambda item: (
            item[0],
            -len(str(item[1])),
        ),
        reverse=True,
    )

    selected = (
        candidates[0][1]
        if candidates
        else None
    )

    for score, path in candidates[:20]:
        candidate_rows.append(
            {
                "path": relative_path(path),
                "score": float(score),
                "selected": bool(
                    selected is not None
                    and path == selected
                ),
                "selection_reason": (
                    "highest_expression_candidate_score"
                    if selected is not None
                    and path == selected
                    else "candidate"
                ),
            }
        )

    return selected, candidate_rows


def discover_outcome_artifacts(
    row: pd.Series,
) -> list[Path]:
    tokens = split_tokens(
        row["search_tokens"]
    )

    roots = [
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "raw",
    ]

    matches: list[Path] = []

    for root in roots:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            full = str(path).lower()

            if not any(
                token.lower() in full
                for token in tokens
            ):
                continue

            if any(
                outcome_token in path.name.lower()
                for outcome_token in OUTCOME_FILE_TOKENS
            ):
                matches.append(path)

    return sorted(set(matches))


# ---------------------------------------------------------------------
# Matrix inspection
# ---------------------------------------------------------------------

def infer_delimiter(path: Path) -> str:
    lower = path.name.lower()

    if lower.endswith(".csv") or lower.endswith(".csv.gz"):
        return ","

    if (
        lower.endswith(".tsv")
        or lower.endswith(".tsv.gz")
        or lower.endswith(".txt")
        or lower.endswith(".txt.gz")
    ):
        return "\t"

    return ","


def open_text_file(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(
            path,
            "rt",
            encoding="utf-8",
            errors="replace",
        )

    return path.open(
        "rt",
        encoding="utf-8",
        errors="replace",
    )


def inspect_delimited_matrix(
    path: Path,
    orientation_hint: str,
    collect_features: bool,
    collect_sample_ids: bool,
) -> MatrixInspection:
    delimiter = infer_delimiter(path)

    with open_text_file(path) as handle:
        header_line = handle.readline()

        if not header_line:
            raise RuntimeError(
                "Matrix has no header."
            )

        header = next(
            csv.reader(
                [header_line],
                delimiter=delimiter,
            )
        )

        n_columns = len(header)

        n_rows = sum(
            1
            for line in handle
            if line.strip()
        )

    if n_columns < 2 or n_rows < 1:
        raise RuntimeError(
            f"Unexpected matrix dimensions: "
            f"{n_rows} rows x {n_columns} columns."
        )

    orientation = orientation_hint

    if orientation not in {
        "samples_by_features",
        "features_by_samples",
    }:
        # Strong asymmetry is enough for preflight inference.
        if n_rows <= 0.20 * n_columns:
            orientation = "samples_by_features"
        elif n_columns <= 0.20 * n_rows:
            orientation = "features_by_samples"
        else:
            orientation = "unknown"

    feature_names: set[str] | None = None
    sample_ids: list[str] | None = None

    if orientation == "samples_by_features":
        n_samples = n_rows
        n_features = n_columns - 1

        if collect_features:
            feature_names = {
                clean_gene_symbol(value)
                for value in header[1:]
                if clean_gene_symbol(value)
            }

        if collect_sample_ids and n_rows <= 10000:
            ids: list[str] = []

            with open_text_file(path) as handle:
                reader = csv.reader(
                    handle,
                    delimiter=delimiter,
                )
                next(reader, None)

                for row in reader:
                    if not row:
                        continue
                    ids.append(clean_text(row[0]))

            sample_ids = ids

    elif orientation == "features_by_samples":
        n_samples = n_columns - 1
        n_features = n_rows

        if collect_features:
            features: set[str] = set()

            with open_text_file(path) as handle:
                handle.readline()

                for line in handle:
                    if not line.strip():
                        continue

                    first = line.split(
                        delimiter,
                        1,
                    )[0]

                    symbol = clean_gene_symbol(first)

                    if symbol:
                        features.add(symbol)

            feature_names = features

        # Avoid keeping huge single-cell barcode vectors.
        if collect_sample_ids and n_samples <= 10000:
            sample_ids = [
                clean_text(value)
                for value in header[1:]
            ]

    else:
        n_samples = None
        n_features = None

    return MatrixInspection(
        path=path,
        format="delimited_text",
        orientation=orientation,
        n_samples=n_samples,
        n_features=n_features,
        feature_names=feature_names,
        sample_ids=sample_ids,
    )


def inspect_numpy_matrix(
    path: Path,
    orientation_hint: str,
) -> MatrixInspection:
    lower = path.name.lower()

    if lower.endswith(".npy"):
        array = np.load(
            path,
            mmap_mode="r",
        )

        shape = array.shape

    else:
        with np.load(
            path,
            allow_pickle=False,
        ) as archive:
            candidates = [
                archive[key]
                for key in archive.files
                if archive[key].ndim == 2
            ]

            if not candidates:
                raise RuntimeError(
                    "NPZ file contains no 2-D array."
                )

            shape = candidates[0].shape

    if len(shape) != 2:
        raise RuntimeError(
            f"Expected a 2-D array, found {shape}."
        )

    orientation = orientation_hint

    if orientation == "samples_by_features":
        n_samples, n_features = shape
    elif orientation == "features_by_samples":
        n_features, n_samples = shape
    else:
        n_samples = None
        n_features = None
        orientation = "unknown"

    return MatrixInspection(
        path=path,
        format="numpy",
        orientation=orientation,
        n_samples=(
            int(n_samples)
            if n_samples is not None
            else None
        ),
        n_features=(
            int(n_features)
            if n_features is not None
            else None
        ),
        feature_names=None,
        sample_ids=None,
    )


def inspect_matrix(
    path: Path,
    orientation_hint: str,
    collect_features: bool,
    collect_sample_ids: bool,
) -> MatrixInspection:
    lower = path.name.lower()

    try:
        if lower.endswith(
            (
                ".csv",
                ".csv.gz",
                ".tsv",
                ".tsv.gz",
                ".txt",
                ".txt.gz",
            )
        ):
            return inspect_delimited_matrix(
                path,
                orientation_hint,
                collect_features,
                collect_sample_ids,
            )

        if lower.endswith((".npy", ".npz")):
            return inspect_numpy_matrix(
                path,
                orientation_hint,
            )

        return MatrixInspection(
            path=path,
            format="unsupported",
            orientation="unknown",
            n_samples=None,
            n_features=None,
            feature_names=None,
            sample_ids=None,
            error="Unsupported matrix format.",
        )

    except Exception as exc:
        return MatrixInspection(
            path=path,
            format="inspection_failed",
            orientation="unknown",
            n_samples=None,
            n_features=None,
            feature_names=None,
            sample_ids=None,
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------
# Frozen programs
# ---------------------------------------------------------------------

def load_frozen_programs() -> tuple[
    pd.DataFrame,
    dict[str, dict[str, set[str]]],
]:
    if not FROZEN_WEIGHTS_FILE.exists():
        raise FileNotFoundError(
            "Frozen weights file not found: "
            f"{FROZEN_WEIGHTS_FILE}"
        )

    weights = pd.read_csv(
        FROZEN_WEIGHTS_FILE
    )

    if "module_label" not in weights.columns:
        raise ValueError(
            "Frozen weights do not contain module_label."
        )

    canine_column = next(
        (
            column
            for column in [
                "canine_gene_symbol",
                "canine_gene",
                "gene",
            ]
            if column in weights.columns
        ),
        None,
    )

    human_column = next(
        (
            column
            for column in [
                "human_gene_symbol",
                "human_gene",
            ]
            if column in weights.columns
        ),
        None,
    )

    if canine_column is None:
        raise ValueError(
            "Frozen weights do not contain a canine gene-symbol column."
        )

    programs: dict[str, dict[str, set[str]]] = {}

    for module in PRIMARY_MODULES:
        part = weights[
            weights["module_label"]
            .astype(str)
            .eq(module)
        ]

        canine = {
            clean_gene_symbol(value)
            for value in part[canine_column]
            if clean_gene_symbol(value)
        }

        human: set[str] = set()

        if human_column is not None:
            human = {
                clean_gene_symbol(value)
                for value in part[human_column]
                if clean_gene_symbol(value)
            }

        programs[module] = {
            "canine": canine,
            "human": human,
        }

    return weights, programs


def coverage_from_feature_set(
    dataset_id: str,
    species: str,
    feature_names: set[str],
    programs: dict[str, dict[str, set[str]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if species not in {"canine", "human"}:
        return rows

    cleaned_features = {
        clean_gene_symbol(value)
        for value in feature_names
        if clean_gene_symbol(value)
    }

    for module in PRIMARY_MODULES:
        frozen = programs[module].get(
            species,
            set(),
        )

        if not frozen:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "module_label": module,
                    "species": species,
                    "n_frozen_genes": np.nan,
                    "n_detected_genes": np.nan,
                    "coverage_fraction": np.nan,
                    "coverage_source": "not_available",
                    "estimable_ge_3": False,
                }
            )
            continue

        detected = frozen & cleaned_features

        rows.append(
            {
                "dataset_id": dataset_id,
                "module_label": module,
                "species": species,
                "n_frozen_genes": len(frozen),
                "n_detected_genes": len(detected),
                "coverage_fraction": (
                    len(detected) / len(frozen)
                    if frozen
                    else np.nan
                ),
                "coverage_source": "expression_feature_intersection",
                "estimable_ge_3": len(detected) >= 3,
            }
        )

    return rows


def coverage_from_precomputed(
    dataset_id: str,
    species: str,
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    table = pd.read_csv(path)

    required = {
        "module_label",
        "n_frozen_genes",
        "n_detected_genes",
        "coverage_fraction",
    }

    if not required.issubset(table.columns):
        return []

    output: list[dict[str, Any]] = []

    for module in PRIMARY_MODULES:
        part = table[
            table["module_label"]
            .astype(str)
            .eq(module)
        ]

        if part.empty:
            continue

        row = part.iloc[0]

        detected = pd.to_numeric(
            pd.Series(
                [row["n_detected_genes"]]
            ),
            errors="coerce",
        ).iloc[0]

        output.append(
            {
                "dataset_id": dataset_id,
                "module_label": module,
                "species": species,
                "n_frozen_genes": row["n_frozen_genes"],
                "n_detected_genes": row["n_detected_genes"],
                "coverage_fraction": row["coverage_fraction"],
                "coverage_source": relative_path(path),
                "estimable_ge_3": bool(
                    np.isfinite(detected)
                    and detected >= 3
                ),
            }
        )

    return output


# ---------------------------------------------------------------------
# Pairwise audits
# ---------------------------------------------------------------------

def build_feature_overlap(
    runtime: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    ids = list(runtime)
    rows: list[dict[str, Any]] = []

    for i, left_id in enumerate(ids):
        for right_id in ids[i + 1:]:
            left = runtime[left_id]
            right = runtime[right_id]

            left_features = left.get("feature_names")
            right_features = right.get("feature_names")

            same_species = (
                left["species"]
                == right["species"]
            )

            row: dict[str, Any] = {
                "dataset_a": left_id,
                "dataset_b": right_id,
                "species_a": left["species"],
                "species_b": right["species"],
                "modality_a": left["modality"],
                "modality_b": right["modality"],
                "comparison_scope": None,
                "n_features_a": (
                    len(left_features)
                    if left_features is not None
                    else np.nan
                ),
                "n_features_b": (
                    len(right_features)
                    if right_features is not None
                    else np.nan
                ),
                "n_shared_features": np.nan,
                "jaccard": np.nan,
                "coverage_a_in_b": np.nan,
                "coverage_b_in_a": np.nan,
                "note": "",
            }

            if not same_species:
                row["comparison_scope"] = (
                    "cross_species_requires_ortholog_mapping"
                )
                row["note"] = (
                    "Raw symbol overlap intentionally not used."
                )

            elif (
                left_features is None
                or right_features is None
            ):
                row["comparison_scope"] = (
                    "feature_names_unavailable"
                )

            else:
                a = set(left_features)
                b = set(right_features)

                shared = a & b
                union = a | b

                row["comparison_scope"] = (
                    "same_species_gene_symbol_overlap"
                )
                row["n_shared_features"] = len(shared)
                row["jaccard"] = (
                    len(shared) / len(union)
                    if union
                    else np.nan
                )
                row["coverage_a_in_b"] = (
                    len(shared) / len(a)
                    if a
                    else np.nan
                )
                row["coverage_b_in_a"] = (
                    len(shared) / len(b)
                    if b
                    else np.nan
                )

            rows.append(row)

    return pd.DataFrame(rows)


def build_sample_matching_audit(
    runtime: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    ids = list(runtime)
    rows: list[dict[str, Any]] = []

    for i, left_id in enumerate(ids):
        for right_id in ids[i + 1:]:
            left = runtime[left_id]
            right = runtime[right_id]

            left_ids = left.get("sample_ids")
            right_ids = right.get("sample_ids")

            same_species = (
                left["species"]
                == right["species"]
            )

            same_unit = (
                left["biological_unit"]
                == right["biological_unit"]
            )

            row = {
                "dataset_a": left_id,
                "dataset_b": right_id,
                "species_match": same_species,
                "biological_unit_match": same_unit,
                "sample_id_a_available": left_ids is not None,
                "sample_id_b_available": right_ids is not None,
                "exact_normalized_overlap_count": np.nan,
                "overlap_ids": "",
                "interpretation": "",
            }

            if not same_species:
                row["interpretation"] = (
                    "not_comparable_cross_species"
                )

            elif not same_unit:
                row["interpretation"] = (
                    "not_comparable_biological_unit"
                )

            elif left_ids is None or right_ids is None:
                row["interpretation"] = (
                    "sample_ids_unavailable"
                )

            else:
                a = {
                    normalize_sample_id(value)
                    for value in left_ids
                    if normalize_sample_id(value)
                }

                b = {
                    normalize_sample_id(value)
                    for value in right_ids
                    if normalize_sample_id(value)
                }

                overlap = sorted(a & b)

                row[
                    "exact_normalized_overlap_count"
                ] = len(overlap)

                row["overlap_ids"] = ";".join(
                    overlap[:50]
                )

                row["interpretation"] = (
                    "exact_id_overlap_detected"
                    if overlap
                    else "no_exact_id_overlap_detected"
                )

            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------

def build_readiness_table(
    inventory: pd.DataFrame,
    sample_matching: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    found = inventory[
        inventory["expression_file_exists"].eq(True)
        & inventory["inspection_ok"].eq(True)
    ].copy()

    bulk_like = found[
        found["modality"].isin(
            [
                "bulk_rna",
                "expression_array",
            ]
        )
    ]

    canine_bulk = bulk_like[
        bulk_like["species"].eq("canine")
    ]

    human_bulk = bulk_like[
        bulk_like["species"].eq("human")
    ]

    single_cell = found[
        found["modality"].eq("single_cell_rna")
    ]

    core_ids = set(
        found["dataset_id"].tolist()
    )

    core_canine_ready = (
        CORE_DATASETS.issubset(core_ids)
    )

    cross_species_bulk_ready = (
        canine_bulk.shape[0] >= 2
        and human_bulk.shape[0] >= 1
    )

    paired_multimodal_overlap = False

    if not sample_matching.empty:
        matched = sample_matching[
            sample_matching[
                "exact_normalized_overlap_count"
            ]
            .fillna(0)
            .astype(float)
            .gt(0)
        ]

        if not matched.empty:
            modality_map = inventory.set_index(
                "dataset_id"
            )["modality"].to_dict()

            for _, row in matched.iterrows():
                if (
                    modality_map.get(row["dataset_a"])
                    != modality_map.get(row["dataset_b"])
                ):
                    paired_multimodal_overlap = True
                    break

    rows = [
        {
            "readiness_item": "core_canine_reference_pair",
            "ready": core_canine_ready,
            "details": (
                "GSE238110 DOG2 and GSE239948 available"
                if core_canine_ready
                else "One or both core canine matrices are missing"
            ),
        },
        {
            "readiness_item": "multi_study_bulk_representation",
            "ready": bulk_like.shape[0] >= 3,
            "details": (
                f"{bulk_like.shape[0]} bulk-like expression datasets available"
            ),
        },
        {
            "readiness_item": "cross_species_bulk_representation",
            "ready": cross_species_bulk_ready,
            "details": (
                f"{canine_bulk.shape[0]} canine and "
                f"{human_bulk.shape[0]} human bulk-like datasets available"
            ),
        },
        {
            "readiness_item": "single_cell_annotation",
            "ready": single_cell.shape[0] >= 1,
            "details": (
                f"{single_cell.shape[0]} single-cell dataset(s) available"
            ),
        },
        {
            "readiness_item": "paired_multimodal_learning",
            "ready": paired_multimodal_overlap,
            "details": (
                "Exact cross-modality subject overlap detected"
                if paired_multimodal_overlap
                else (
                    "No evidence yet for paired cross-modality "
                    "subject-level training"
                )
            ),
        },
    ]

    if cross_species_bulk_ready:
        recommendation = (
            "Proceed next with outcome-blind multi-study bulk "
            "representation harmonization. Keep human outcomes "
            "outside representation learning. Use single-cell "
            "data as biological annotation unless paired "
            "multimodal subjects are demonstrated."
        )
    elif core_canine_ready:
        recommendation = (
            "Canine representation work is ready, but human "
            "expression matrices must be resolved before a "
            "cross-species representation-learning model is frozen."
        )
    else:
        recommendation = (
            "Resolve missing core molecular matrices before "
            "representation-learning model development."
        )

    summary = {
        "n_expression_datasets_found": int(found.shape[0]),
        "n_bulk_like_datasets_found": int(bulk_like.shape[0]),
        "n_canine_bulk_like": int(canine_bulk.shape[0]),
        "n_human_bulk_like": int(human_bulk.shape[0]),
        "n_single_cell": int(single_cell.shape[0]),
        "core_canine_ready": bool(core_canine_ready),
        "cross_species_bulk_ready": bool(
            cross_species_bulk_ready
        ),
        "paired_multimodal_ready": bool(
            paired_multimodal_overlap
        ),
        "recommendation": recommendation,
    }

    return pd.DataFrame(rows), summary


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def run_preflight(
    full_hash: bool,
    max_hash_mb: int,
    strict: bool,
) -> None:
    print("=" * 80)
    print("Paper 5 - multimodal / multi-study dataset preflight")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Paper 5 root: {PAPER5_ROOT}")
    print()

    registry = load_registry()

    print(f"Loaded registry: {REGISTRY_FILE}")
    print(f"Registered datasets: {registry.shape[0]}")
    print()

    weights, programs = load_frozen_programs()

    print(f"Loaded frozen programs: {FROZEN_WEIGHTS_FILE}")
    print(
        "Frozen modules: "
        + ", ".join(PRIMARY_MODULES)
    )
    print()

    inventory_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    runtime: dict[str, dict[str, Any]] = {}

    max_hash_bytes = (
        int(max_hash_mb)
        * 1024
        * 1024
    )

    selected_input_hashes: list[dict[str, Any]] = []

    selected_input_hashes.append(
        sha256_file(
            FROZEN_WEIGHTS_FILE,
            max_hash_bytes=max_hash_bytes,
            force_full_hash=full_hash,
        )
    )

    for _, registry_row in registry.iterrows():
        dataset_id = registry_row["dataset_id"]
        species = registry_row["species"].lower()
        modality = registry_row["modality"].lower()
        role = registry_row["current_role"]
        biological_unit = registry_row["biological_unit"]

        print("-" * 80)
        print(dataset_id)
        print("-" * 80)

        expression_file, candidates = (
            discover_expression_file(
                registry_row
            )
        )

        for candidate in candidates:
            file_rows.append(
                {
                    "dataset_id": dataset_id,
                    "file_role": "expression_candidate",
                    **candidate,
                }
            )

        outcome_artifacts = (
            discover_outcome_artifacts(
                registry_row
            )
        )

        for path in outcome_artifacts:
            file_rows.append(
                {
                    "dataset_id": dataset_id,
                    "file_role": (
                        "outcome_or_clinical_artifact_not_loaded"
                    ),
                    "path": relative_path(path),
                    "score": np.nan,
                    "selected": False,
                    "selection_reason": (
                        "recorded_only_outcome_blind_guardrail"
                    ),
                }
            )

        inspection: MatrixInspection | None = None

        precomputed_path_text = clean_text(
            registry_row[
                "precomputed_coverage_path"
            ]
        )

        precomputed_path = (
            PROJECT_ROOT
            / precomputed_path_text
            if precomputed_path_text
            else None
        )

        use_precomputed_coverage = bool(
            precomputed_path is not None
            and precomputed_path.exists()
        )

        if expression_file is not None:
            # Never select an outcome-like file as molecular input.
            lower_name = expression_file.name.lower()

            if any(
                token in lower_name
                for token in OUTCOME_FILE_TOKENS
            ):
                raise RuntimeError(
                    "Outcome-blind guardrail failed: "
                    f"selected molecular input looks outcome-related: "
                    f"{expression_file}"
                )

            # For large single-cell data, precomputed frozen-program
            # coverage is preferred over scanning the entire gene matrix.
            collect_features = not (
                modality == "single_cell_rna"
                and use_precomputed_coverage
            )

            collect_sample_ids = (
                biological_unit == "tumor"
                and modality in {
                    "bulk_rna",
                    "expression_array",
                }
            )

            inspection = inspect_matrix(
                expression_file,
                registry_row["orientation_hint"],
                collect_features=collect_features,
                collect_sample_ids=collect_sample_ids,
            )

            selected_input_hashes.append(
                sha256_file(
                    expression_file,
                    max_hash_bytes=max_hash_bytes,
                    force_full_hash=full_hash,
                )
            )

            print(
                f"Selected expression: "
                f"{relative_path(expression_file)}"
            )

            if inspection.error:
                print(
                    f"Inspection: FAILED - "
                    f"{inspection.error}"
                )
            else:
                print(
                    "Inspection: "
                    f"{inspection.orientation}, "
                    f"samples/cells={inspection.n_samples}, "
                    f"features={inspection.n_features}"
                )

        else:
            print("Selected expression: NOT FOUND")

        if use_precomputed_coverage:
            rows = coverage_from_precomputed(
                dataset_id=dataset_id,
                species=species,
                path=precomputed_path,
            )

            coverage_rows.extend(rows)

            selected_input_hashes.append(
                sha256_file(
                    precomputed_path,
                    max_hash_bytes=max_hash_bytes,
                    force_full_hash=full_hash,
                )
            )

        elif (
            inspection is not None
            and inspection.feature_names is not None
            and registry_row["feature_space"]
            in {"gene_symbol", "human_symbol", "canine_symbol"}
        ):
            coverage_rows.extend(
                coverage_from_feature_set(
                    dataset_id=dataset_id,
                    species=species,
                    feature_names=inspection.feature_names,
                    programs=programs,
                )
            )

        expression_exists = bool(
            expression_file is not None
            and expression_file.exists()
        )

        inspection_ok = bool(
            inspection is not None
            and inspection.error is None
        )

        inventory_rows.append(
            {
                "dataset_id": dataset_id,
                "accession": registry_row["accession"],
                "species": species,
                "modality": modality,
                "biological_unit": biological_unit,
                "current_role": role,
                "expression_file": (
                    relative_path(expression_file)
                    if expression_file is not None
                    else ""
                ),
                "expression_file_exists": expression_exists,
                "matrix_format": (
                    inspection.format
                    if inspection is not None
                    else ""
                ),
                "orientation": (
                    inspection.orientation
                    if inspection is not None
                    else ""
                ),
                "n_samples_or_cells": (
                    inspection.n_samples
                    if inspection is not None
                    else np.nan
                ),
                "n_features": (
                    inspection.n_features
                    if inspection is not None
                    else np.nan
                ),
                "feature_space": registry_row["feature_space"],
                "feature_names_available": bool(
                    inspection is not None
                    and inspection.feature_names is not None
                ),
                "sample_ids_available": bool(
                    inspection is not None
                    and inspection.sample_ids is not None
                ),
                "precomputed_coverage_file": (
                    relative_path(precomputed_path)
                    if use_precomputed_coverage
                    else ""
                ),
                "outcome_artifact_count": len(
                    outcome_artifacts
                ),
                "outcome_artifacts_recorded_only": True,
                "outcomes_loaded": False,
                "inspection_ok": inspection_ok,
                "inspection_error": (
                    inspection.error
                    if inspection is not None
                    and inspection.error
                    else ""
                ),
            }
        )

        runtime[dataset_id] = {
            "species": species,
            "modality": modality,
            "biological_unit": biological_unit,
            "current_role": role,
            "feature_names": (
                inspection.feature_names
                if inspection is not None
                else None
            ),
            "sample_ids": (
                inspection.sample_ids
                if inspection is not None
                else None
            ),
            "n_samples": (
                inspection.n_samples
                if inspection is not None
                else None
            ),
            "n_features": (
                inspection.n_features
                if inspection is not None
                else None
            ),
        }

        if outcome_artifacts:
            print(
                f"Outcome/clinical artifacts recorded but not loaded: "
                f"{len(outcome_artifacts)}"
            )

        print()

    inventory = pd.DataFrame(
        inventory_rows
    ).sort_values("dataset_id")

    file_inventory = pd.DataFrame(
        file_rows
    )

    coverage = pd.DataFrame(
        coverage_rows
    )

    feature_overlap = build_feature_overlap(
        runtime
    )

    sample_matching = build_sample_matching_audit(
        runtime
    )

    readiness, readiness_summary = (
        build_readiness_table(
            inventory,
            sample_matching,
        )
    )

    # Strict mode protects the two established canine anchors.
    if strict:
        missing_core = []

        for dataset_id in sorted(CORE_DATASETS):
            part = inventory[
                inventory["dataset_id"].eq(
                    dataset_id
                )
            ]

            if (
                part.empty
                or not bool(
                    part.iloc[0][
                        "expression_file_exists"
                    ]
                )
                or not bool(
                    part.iloc[0][
                        "inspection_ok"
                    ]
                )
            ):
                missing_core.append(dataset_id)

        if missing_core:
            raise RuntimeError(
                "Strict preflight failed. Missing or invalid core datasets: "
                + ", ".join(missing_core)
            )

    inventory.to_csv(
        OUTPUT_DATASET_INVENTORY,
        index=False,
    )

    file_inventory.to_csv(
        OUTPUT_FILE_INVENTORY,
        index=False,
    )

    feature_overlap.to_csv(
        OUTPUT_MODALITY_OVERLAP,
        index=False,
    )

    coverage.to_csv(
        OUTPUT_PROGRAM_COVERAGE,
        index=False,
    )

    sample_matching.to_csv(
        OUTPUT_SAMPLE_MATCHING,
        index=False,
    )

    readiness.to_csv(
        OUTPUT_READINESS,
        index=False,
    )

    # Remove duplicate hash entries caused by reused files.
    hashes_by_path: dict[str, dict[str, Any]] = {}

    for entry in selected_input_hashes:
        hashes_by_path[
            str(entry["path"])
        ] = entry

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "project_root": str(PROJECT_ROOT),
        "paper5_root": str(PAPER5_ROOT),
        "outcome_loaded": False,
        "feature_selection_using_outcome": False,
        "outcome_guided_orientation": False,
        "outcome_guided_dataset_selection": False,
        "git": get_git_state(),
        "guardrails": [
            "Paper 4 scripts and frozen result artifacts are read-only inputs.",
            "All new outputs are written under paper5/.",
            "Clinical and outcome artifacts may be discovered by filename but are never loaded.",
            "Target outcomes are not used for expression-file selection.",
            "Target outcomes are not used for feature selection.",
            "Target outcomes are not used for score orientation.",
            "Cross-species raw gene-symbol overlap is not interpreted without ortholog mapping.",
            "Single-cell data are treated as biological annotation unless paired multimodal training is independently demonstrated.",
        ],
        "registered_datasets": registry.to_dict(
            orient="records"
        ),
        "input_files": list(
            hashes_by_path.values()
        ),
        "readiness": readiness_summary,
        "outputs": {
            "dataset_inventory": relative_path(
                OUTPUT_DATASET_INVENTORY
            ),
            "dataset_file_inventory": relative_path(
                OUTPUT_FILE_INVENTORY
            ),
            "modality_overlap": relative_path(
                OUTPUT_MODALITY_OVERLAP
            ),
            "frozen_program_coverage": relative_path(
                OUTPUT_PROGRAM_COVERAGE
            ),
            "sample_matching_audit": relative_path(
                OUTPUT_SAMPLE_MATCHING
            ),
            "representation_learning_readiness": relative_path(
                OUTPUT_READINESS
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
        "Paper 5 multimodal / multi-study preflight",
        "=" * 60,
        "",
        f"Script version: {SCRIPT_VERSION}",
        f"Created UTC: {manifest['created_at_utc']}",
        "",
        "Outcome-blind state",
        "-------------------",
        "outcome_loaded: false",
        "outcome-guided feature selection: false",
        "outcome-guided orientation: false",
        "",
        "Representation-learning readiness",
        "---------------------------------",
    ]

    for _, row in readiness.iterrows():
        readme_lines.append(
            f"{row['readiness_item']}: "
            f"{'READY' if row['ready'] else 'NOT READY'}"
        )
        readme_lines.append(
            f"  {row['details']}"
        )

    readme_lines.extend(
        [
            "",
            "Recommended next step",
            "---------------------",
            readiness_summary["recommendation"],
            "",
            "Important",
            "---------",
            (
                "This preflight does not define the final Paper 5 "
                "representation-learning architecture."
            ),
            (
                "Its purpose is to establish which molecular datasets "
                "and feature spaces are actually available before a "
                "model is frozen."
            ),
            "",
        ]
    )

    OUTPUT_README.write_text(
        "\n".join(readme_lines),
        encoding="utf-8",
    )

    print("=" * 80)
    print("Paper 5 preflight complete")
    print("=" * 80)

    display_columns = [
        "dataset_id",
        "species",
        "modality",
        "current_role",
        "expression_file_exists",
        "n_samples_or_cells",
        "n_features",
        "inspection_ok",
    ]

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        180,
    ):
        print(
            inventory[
                display_columns
            ].to_string(index=False)
        )

    print()
    print("Readiness:")
    for _, row in readiness.iterrows():
        print(
            f"  {row['readiness_item']}: "
            f"{'READY' if row['ready'] else 'NOT READY'}"
        )

    print()
    print("Recommendation:")
    print(
        f"  {readiness_summary['recommendation']}"
    )

    print()
    print("Outputs:")
    for path in [
        OUTPUT_DATASET_INVENTORY,
        OUTPUT_FILE_INVENTORY,
        OUTPUT_MODALITY_OVERLAP,
        OUTPUT_PROGRAM_COVERAGE,
        OUTPUT_SAMPLE_MATCHING,
        OUTPUT_READINESS,
        OUTPUT_MANIFEST,
        OUTPUT_README,
    ]:
        print(f"  {relative_path(path)}")

    print("=" * 80)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Outcome-blind dataset and modality preflight "
            "for Paper 5."
        )
    )

    parser.add_argument(
        "--rewrite-registry",
        action="store_true",
        help=(
            "Replace paper5/configs/dataset_registry.csv "
            "with the default registry."
        ),
    )

    parser.add_argument(
        "--full-hash",
        action="store_true",
        help=(
            "Hash all selected inputs regardless of file size."
        ),
    )

    parser.add_argument(
        "--max-hash-mb",
        type=int,
        default=DEFAULT_MAX_HASH_MB,
        help=(
            "Maximum file size hashed by default. "
            f"Default: {DEFAULT_MAX_HASH_MB} MB."
        ),
    )

    parser.add_argument(
        "--no-strict",
        action="store_true",
        help=(
            "Do not fail when the two core canine expression "
            "matrices are missing."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.rewrite_registry:
        write_default_registry(
            overwrite=True
        )
        print(
            f"Registry rewritten: {REGISTRY_FILE}"
        )

    run_preflight(
        full_hash=bool(args.full_hash),
        max_hash_mb=int(args.max_hash_mb),
        strict=not bool(args.no_strict),
    )


if __name__ == "__main__":
    main()
