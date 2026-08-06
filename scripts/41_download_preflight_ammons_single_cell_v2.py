from __future__ import annotations

from pathlib import Path
import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.parse import urljoin

import pandas as pd

SCRIPT_VERSION = "41-download-preflight-ammons-single-cell-v2"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "single_cell"
    / "Ammons_GSE252470"
)
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"

RAW_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STRICT_WEIGHTS_FILE = (
    RESULTS_DIR / "GSE238110_frozen_transfer_gene_weights_strict.csv"
)

CELL_BROWSER_ROOT = "https://cells.ucsc.edu/"
COLLECTION_NAME = "canine-os-atlas"
COLLECTION_JSON_URL = (
    f"{CELL_BROWSER_ROOT}{COLLECTION_NAME}/dataset.json"
)

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]

OUTPUT_DISCOVERY = (
    RESULTS_DIR / "Ammons_scRNA_cellbrowser_dataset_discovery.csv"
)
OUTPUT_METADATA_SUMMARY = (
    RESULTS_DIR / "Ammons_scRNA_metadata_preflight.csv"
)
OUTPUT_MODULE_COVERAGE = (
    RESULTS_DIR / "Ammons_scRNA_frozen_module_gene_coverage.csv"
)
OUTPUT_CELLTYPE_COUNTS = (
    RESULTS_DIR / "Ammons_scRNA_celltype_counts.csv"
)
OUTPUT_README = (
    RESULTS_DIR / "Ammons_scRNA_preflight_README.txt"
)
OUTPUT_MANIFEST = (
    RESULTS_DIR / "Ammons_scRNA_preflight_manifest.json"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def require_requests():
    try:
        import requests
    except ImportError as error:
        raise ImportError(
            "The requests package is required. Install it with:\n"
            f'"{sys.executable}" -m pip install requests'
        ) from error
    return requests


def request_json(url: str) -> dict[str, Any]:
    requests = require_requests()
    response = requests.get(
        url,
        timeout=(30, 120),
        allow_redirects=True,
    )
    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object from {url}.")
    return payload


def url_exists(url: str) -> bool:
    requests = require_requests()

    try:
        response = requests.head(
            url,
            timeout=(20, 60),
            allow_redirects=True,
        )
        if response.status_code == 405:
            response = requests.get(
                url,
                timeout=(20, 60),
                allow_redirects=True,
                stream=True,
                headers={"Range": "bytes=0-0"},
            )
        return response.status_code in {200, 206}
    except Exception:
        return False


def dataset_json_url(dataset_name: str) -> str:
    return urljoin(
        CELL_BROWSER_ROOT,
        f"{dataset_name.strip('/')}/dataset.json",
    )


def discover_leaf_datasets(
    dataset_name: str,
    inherited_label: str | None = None,
    visited: set[str] | None = None,
) -> list[dict[str, Any]]:
    if visited is None:
        visited = set()

    normalized = dataset_name.strip("/")
    if normalized in visited:
        return []
    visited.add(normalized)

    payload = request_json(dataset_json_url(normalized))
    short_label = str(
        payload.get("shortLabel")
        or payload.get("label")
        or inherited_label
        or normalized
    )
    children = payload.get("datasets") or []

    if children:
        leaves: list[dict[str, Any]] = []
        for child in children:
            child_name = str(child.get("name", "")).strip("/")
            if not child_name:
                continue
            child_label = str(
                child.get("shortLabel")
                or child.get("label")
                or child_name
            )
            leaves.extend(
                discover_leaf_datasets(
                    dataset_name=child_name,
                    inherited_label=child_label,
                    visited=visited,
                )
            )
        return leaves

    return [
        {
            "dataset_name": normalized,
            "short_label": short_label,
            "sample_count": int(payload.get("sampleCount") or 0),
            "dataset_json": payload,
        }
    ]


def basename_from_file_version(
    payload: dict[str, Any],
    preferred_keys: list[str],
) -> str | None:
    versions = payload.get("fileVersions") or {}

    for key in preferred_keys:
        entry = versions.get(key)
        if isinstance(entry, dict):
            fname = entry.get("fname")
            if fname:
                return Path(str(fname)).name

    return None


def candidate_data_files(
    leaf: dict[str, Any],
) -> dict[str, Any]:
    payload = leaf["dataset_json"]
    dataset_name = leaf["dataset_name"].strip("/")
    base_url = urljoin(
        CELL_BROWSER_ROOT,
        f"{dataset_name}/",
    )

    metadata_basename = basename_from_file_version(
        payload,
        ["outMeta", "inMeta"],
    ) or "meta.tsv"

    expression_basename = basename_from_file_version(
        payload,
        ["outMatrix", "inMatrix"],
    )

    expression_candidates = []
    if expression_basename:
        expression_candidates.append(expression_basename)

    expression_candidates.extend(
        [
            "exprMatrix.tsv.gz",
            "matrix.mtx.gz",
            "exprMatrix.mtx.gz",
            "matrix.mtx",
        ]
    )
    expression_candidates = list(
        dict.fromkeys(expression_candidates)
    )

    expression_url = None
    selected_expression_basename = None
    for candidate in expression_candidates:
        candidate_url = urljoin(base_url, candidate)
        if url_exists(candidate_url):
            expression_url = candidate_url
            selected_expression_basename = candidate
            break

    metadata_url = urljoin(base_url, metadata_basename)
    if not url_exists(metadata_url):
        fallback_url = urljoin(base_url, "meta.tsv")
        if url_exists(fallback_url):
            metadata_url = fallback_url
            metadata_basename = "meta.tsv"
        else:
            metadata_url = None

    dataset_json_url_value = urljoin(base_url, "dataset.json")
    desc_json_url = urljoin(base_url, "desc.json")

    return {
        **leaf,
        "base_url": base_url,
        "metadata_basename": metadata_basename,
        "metadata_url": metadata_url,
        "expression_basename": selected_expression_basename,
        "expression_url": expression_url,
        "dataset_json_url": dataset_json_url_value,
        "desc_json_url": (
            desc_json_url if url_exists(desc_json_url) else None
        ),
        "is_downloadable_leaf": bool(
            metadata_url and expression_url
        ),
    }


def select_full_annotated_leaf(
    leaves: list[dict[str, Any]],
) -> dict[str, Any]:
    downloadable = [
        leaf for leaf in leaves if leaf["is_downloadable_leaf"]
    ]

    if not downloadable:
        details = "\n".join(
            f"- {leaf['dataset_name']}: "
            f"meta={leaf['metadata_url']}, "
            f"expression={leaf['expression_url']}"
            for leaf in leaves
        )
        raise RuntimeError(
            "No downloadable Cell Browser leaf dataset was found.\n"
            + details
        )

    preference_tokens = [
        "full",
        "complete",
        "annotated",
        "all cells",
        "all_cells",
        "naive",
        "n6",
    ]

    def selection_key(leaf: dict[str, Any]) -> tuple[int, int]:
        text = (
            f"{leaf['dataset_name']} {leaf['short_label']}"
        ).lower()
        preference_score = sum(
            token in text for token in preference_tokens
        )
        return preference_score, int(leaf["sample_count"])

    return max(downloadable, key=selection_key)


def download_with_resume(
    url: str,
    destination: Path,
    required: bool,
) -> bool:
    requests = require_requests()

    if destination.exists() and destination.stat().st_size > 0:
        print(
            f"Using existing: {destination} "
            f"({destination.stat().st_size / 1024 / 1024:.1f} MB)"
        )
        return True

    partial = destination.with_suffix(
        destination.suffix + ".part"
    )
    downloaded = partial.stat().st_size if partial.exists() else 0

    headers = {}
    mode = "wb"
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"
        mode = "ab"

    print(f"Downloading: {url}")

    try:
        with requests.get(
            url,
            stream=True,
            timeout=(30, 240),
            headers=headers,
            allow_redirects=True,
        ) as response:
            if response.status_code == 416 and partial.exists():
                partial.replace(destination)
                return True

            if response.status_code not in {200, 206}:
                message = (
                    f"Download failed with HTTP "
                    f"{response.status_code}: {url}"
                )
                if required:
                    raise RuntimeError(message)
                print(f"Optional file skipped: {message}")
                return False

            if downloaded > 0 and response.status_code == 200:
                downloaded = 0
                mode = "wb"

            total_header = response.headers.get("content-length")
            total = (
                int(total_header) + downloaded
                if total_header is not None
                else None
            )

            written = downloaded
            next_report = written + 100 * 1024 * 1024

            with partial.open(mode) as handle:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if not chunk:
                        continue

                    handle.write(chunk)
                    written += len(chunk)

                    if written >= next_report:
                        if total:
                            percent = 100.0 * written / total
                            print(
                                f"  {written / 1024 / 1024:.1f} MB "
                                f"({percent:.1f}%)"
                            )
                        else:
                            print(
                                f"  {written / 1024 / 1024:.1f} MB"
                            )
                        next_report += 100 * 1024 * 1024

        partial.replace(destination)
        print(
            f"Saved: {destination} "
            f"({destination.stat().st_size / 1024 / 1024:.1f} MB)"
        )
        return True

    except Exception:
        if required:
            raise
        return False


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


def candidate_columns(
    columns: Iterable[str],
    exact_candidates: list[str],
    contains_candidates: list[str],
) -> list[str]:
    columns_list = list(columns)
    lower_map = {
        str(column).lower(): str(column)
        for column in columns_list
    }

    found = []
    for candidate in exact_candidates:
        if candidate.lower() in lower_map:
            found.append(lower_map[candidate.lower()])

    for column in columns_list:
        lower = str(column).lower()
        if any(token in lower for token in contains_candidates):
            if str(column) not in found:
                found.append(str(column))

    return found


def metadata_preflight(
    metadata_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = pd.read_csv(
        metadata_path,
        sep="\t",
        dtype=str,
        low_memory=False,
    )

    if metadata.empty:
        raise RuntimeError(
            "The Cell Browser metadata table is empty."
        )

    cell_id_candidates = candidate_columns(
        metadata.columns,
        exact_candidates=[
            "cell",
            "cell_id",
            "cellid",
            "barcode",
            "Cell",
        ],
        contains_candidates=["cellid", "barcode"],
    )
    cell_id_column = (
        cell_id_candidates[0]
        if cell_id_candidates
        else str(metadata.columns[0])
    )

    celltype_columns = candidate_columns(
        metadata.columns,
        exact_candidates=[
            "celltype.l1",
            "celltype.l2",
            "celltype.l3",
            "cell_type",
            "celltype",
            "majorID",
            "majorID_sub",
            "majorID_subWclus",
        ],
        contains_candidates=[
            "celltype",
            "cell_type",
            "majorid",
            "annotation",
        ],
    )

    dog_columns = candidate_columns(
        metadata.columns,
        exact_candidates=[
            "orig.ident",
            "sample",
            "sample_id",
            "sampleID",
            "dog",
            "dog_id",
            "patient",
        ],
        contains_candidates=[
            "orig.ident",
            "sample",
            "dog",
            "patient",
            "donor",
        ],
    )

    summary_rows = []
    for column in metadata.columns:
        series = metadata[column]
        summary_rows.append(
            {
                "column": str(column),
                "n_nonmissing": int(series.notna().sum()),
                "n_unique": int(series.nunique(dropna=True)),
                "example_values": "; ".join(
                    series.dropna()
                    .astype(str)
                    .drop_duplicates()
                    .head(5)
                ),
                "is_cell_id_candidate": (
                    str(column) == cell_id_column
                ),
                "is_celltype_candidate": (
                    str(column) in celltype_columns
                ),
                "is_dog_candidate": str(column) in dog_columns,
            }
        )

    diagnostics = {
        "n_cells_metadata": int(metadata.shape[0]),
        "n_metadata_columns": int(metadata.shape[1]),
        "cell_id_column": cell_id_column,
        "celltype_columns": celltype_columns,
        "dog_columns": dog_columns,
    }
    return pd.DataFrame(summary_rows), diagnostics


def read_tsv_expression_header(
    expression_path: Path,
) -> tuple[list[str], str]:
    with gzip.open(
        expression_path,
        "rt",
        encoding="utf-8",
    ) as handle:
        header = handle.readline().rstrip("\n\r").split("\t")

    if len(header) < 2:
        raise RuntimeError(
            "The expression matrix header has fewer than two columns."
        )

    return header[1:], header[0]


def scan_tsv_expression_genes(
    expression_path: Path,
    requested_genes: set[str],
) -> tuple[set[str], int]:
    found = set()
    n_rows = 0

    with gzip.open(
        expression_path,
        "rt",
        encoding="utf-8",
    ) as handle:
        handle.readline()
        for line in handle:
            if not line.strip():
                continue

            gene = clean_gene_symbol(
                line.split("\t", 1)[0]
            )
            n_rows += 1

            if gene in requested_genes:
                found.add(gene)

    return found, n_rows


def build_module_coverage(
    weights: pd.DataFrame,
    found_genes: set[str],
) -> pd.DataFrame:
    rows = []

    for module in PRIMARY_MODULES:
        part = weights[
            weights["module_label"].astype(str).eq(module)
        ].copy()
        part["human_gene_symbol"] = (
            part["human_gene_symbol"].astype(str).str.upper()
        )

        module_genes = sorted(
            set(part["human_gene_symbol"].dropna())
        )
        present = sorted(
            set(module_genes).intersection(found_genes)
        )
        missing = sorted(
            set(module_genes).difference(found_genes)
        )

        loadings = pd.to_numeric(
            part["risk_oriented_loading"],
            errors="coerce",
        )
        positive = set(
            part.loc[
                loadings > 0,
                "human_gene_symbol",
            ]
        )
        negative = set(
            part.loc[
                loadings < 0,
                "human_gene_symbol",
            ]
        )

        rows.append(
            {
                "module_label": module,
                "n_frozen_genes": len(module_genes),
                "n_detected_in_sc_matrix": len(present),
                "coverage_fraction": (
                    len(present) / len(module_genes)
                    if module_genes
                    else float("nan")
                ),
                "n_positive_loading_genes": len(positive),
                "n_positive_loading_genes_detected": len(
                    positive.intersection(found_genes)
                ),
                "n_negative_loading_genes": len(negative),
                "n_negative_loading_genes_detected": len(
                    negative.intersection(found_genes)
                ),
                "detected_genes": ";".join(present),
                "missing_genes": ";".join(missing),
            }
        )

    return pd.DataFrame(rows)


def build_celltype_counts(
    metadata_path: Path,
    celltype_columns: list[str],
) -> pd.DataFrame:
    metadata = pd.read_csv(
        metadata_path,
        sep="\t",
        dtype=str,
        low_memory=False,
    )

    rows = []
    for column in celltype_columns:
        counts = (
            metadata[column]
            .fillna("NA")
            .value_counts(dropna=False)
        )

        for label, count in counts.items():
            rows.append(
                {
                    "annotation_column": column,
                    "cell_type": str(label),
                    "n_cells": int(count),
                    "fraction_cells": float(
                        count / metadata.shape[0]
                    ),
                }
            )

    return pd.DataFrame(rows)


def write_readme(
    diagnostics: dict[str, Any],
    selected_leaf: dict[str, Any],
) -> None:
    text = f"""Ammons canine osteosarcoma single-cell atlas preflight
Script version: {SCRIPT_VERSION}

Source
------
UCSC Cell Browser collection: {COLLECTION_NAME}
Resolved leaf dataset: {selected_leaf["dataset_name"]}
Resolved label: {selected_leaf["short_label"]}
Publication dataset accession: GSE252470

Why v2 was needed
-----------------
The public URL points to a Cell Browser collection. Collection directories
contain child datasets but do not themselves contain meta.tsv or an expression
matrix. The script now resolves the collection hierarchy through dataset.json
and selects the downloadable full leaf dataset before downloading files.

Detected metadata
-----------------
Cell-ID column: {diagnostics.get("cell_id_column")}
Candidate cell-type columns:
{json.dumps(diagnostics.get("celltype_columns", []), indent=2)}
Candidate dog/sample columns:
{json.dumps(diagnostics.get("dog_columns", []), indent=2)}

Purpose
-------
This preflight does not calculate module scores. It verifies:
- Python-readable expression and metadata files,
- cell identifier consistency,
- available cell-type and dog/sample annotations,
- frozen M34/M11/M24/M40 gene coverage,
- positive and negative loading coverage.

No clinical outcome is loaded.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print(
        "Discover, download, and preflight Ammons canine OS "
        "single-cell atlas"
    )
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Design:")
    print("  Resolve the UCSC Cell Browser collection hierarchy.")
    print("  Select the full downloadable leaf dataset.")
    print("  Download metadata and expression matrix from that leaf.")
    print("  Detect cell-type and dog/sample metadata columns.")
    print("  Scan frozen-module coverage without loading the full matrix.")
    print("")

    if not STRICT_WEIGHTS_FILE.exists():
        raise FileNotFoundError(
            f"Frozen weights not found: {STRICT_WEIGHTS_FILE}"
        )

    print(f"Resolving collection: {COLLECTION_JSON_URL}")
    raw_leaves = discover_leaf_datasets(COLLECTION_NAME)
    leaves = [
        candidate_data_files(leaf)
        for leaf in raw_leaves
    ]

    discovery_table = pd.DataFrame(
        [
            {
                "dataset_name": leaf["dataset_name"],
                "short_label": leaf["short_label"],
                "sample_count": leaf["sample_count"],
                "base_url": leaf["base_url"],
                "metadata_url": leaf["metadata_url"],
                "expression_url": leaf["expression_url"],
                "expression_basename": leaf[
                    "expression_basename"
                ],
                "is_downloadable_leaf": leaf[
                    "is_downloadable_leaf"
                ],
            }
            for leaf in leaves
        ]
    )
    discovery_table.to_csv(OUTPUT_DISCOVERY, index=False)

    selected = select_full_annotated_leaf(leaves)

    print("")
    print("Selected Cell Browser leaf:")
    print(f"  name: {selected['dataset_name']}")
    print(f"  label: {selected['short_label']}")
    print(f"  cells: {selected['sample_count']}")
    print(f"  metadata: {selected['metadata_url']}")
    print(f"  expression: {selected['expression_url']}")

    metadata_path = RAW_DIR / str(
        selected["metadata_basename"]
    )
    expression_path = RAW_DIR / str(
        selected["expression_basename"]
    )
    leaf_dataset_json_path = RAW_DIR / (
        "resolved_leaf_dataset.json"
    )
    leaf_desc_json_path = RAW_DIR / (
        "resolved_leaf_desc.json"
    )

    download_with_resume(
        selected["metadata_url"],
        metadata_path,
        required=True,
    )
    download_with_resume(
        selected["expression_url"],
        expression_path,
        required=True,
    )
    download_with_resume(
        selected["dataset_json_url"],
        leaf_dataset_json_path,
        required=True,
    )
    if selected["desc_json_url"]:
        download_with_resume(
            selected["desc_json_url"],
            leaf_desc_json_path,
            required=False,
        )

    if expression_path.name not in {
        "exprMatrix.tsv.gz",
    } and not expression_path.name.endswith(".tsv.gz"):
        raise RuntimeError(
            "The resolved Cell Browser leaf uses an MTX-format "
            "matrix rather than exprMatrix.tsv.gz. The download "
            "discovery succeeded, but this preflight version only "
            "scans TSV matrices. Send the printed selected leaf and "
            "expression filename so the MTX scanner can be added."
        )

    weights = pd.read_csv(STRICT_WEIGHTS_FILE)
    weights["human_gene_symbol"] = (
        weights["human_gene_symbol"].astype(str).str.upper()
    )
    requested_genes = set(
        weights[
            weights["module_label"].isin(PRIMARY_MODULES)
        ]["human_gene_symbol"]
    )

    metadata_summary, diagnostics = metadata_preflight(
        metadata_path
    )
    expression_cell_ids, gene_column = (
        read_tsv_expression_header(expression_path)
    )

    diagnostics["resolved_dataset_name"] = selected[
        "dataset_name"
    ]
    diagnostics["resolved_dataset_label"] = selected[
        "short_label"
    ]
    diagnostics["expression_gene_column"] = gene_column
    diagnostics["n_expression_cells"] = len(
        expression_cell_ids
    )

    metadata = pd.read_csv(
        metadata_path,
        sep="\t",
        dtype=str,
        low_memory=False,
    )
    metadata_cell_ids = set(
        metadata[
            diagnostics["cell_id_column"]
        ].astype(str)
    )
    expression_cell_id_set = set(
        map(str, expression_cell_ids)
    )

    intersection = metadata_cell_ids.intersection(
        expression_cell_id_set
    )
    diagnostics["n_cell_ids_intersection"] = len(
        intersection
    )
    diagnostics["metadata_cell_match_fraction"] = (
        len(intersection) / max(1, len(metadata_cell_ids))
    )
    diagnostics["expression_cell_match_fraction"] = (
        len(intersection)
        / max(1, len(expression_cell_id_set))
    )

    found_genes, n_expression_gene_rows = (
        scan_tsv_expression_genes(
            expression_path,
            requested_genes,
        )
    )
    diagnostics["n_expression_gene_rows"] = (
        n_expression_gene_rows
    )

    module_coverage = build_module_coverage(
        weights=weights,
        found_genes=found_genes,
    )
    celltype_counts = build_celltype_counts(
        metadata_path=metadata_path,
        celltype_columns=diagnostics[
            "celltype_columns"
        ],
    )

    metadata_summary.to_csv(
        OUTPUT_METADATA_SUMMARY,
        index=False,
    )
    module_coverage.to_csv(
        OUTPUT_MODULE_COVERAGE,
        index=False,
    )
    celltype_counts.to_csv(
        OUTPUT_CELLTYPE_COUNTS,
        index=False,
    )
    write_readme(diagnostics, selected)

    input_paths = [
        STRICT_WEIGHTS_FILE,
        metadata_path,
        expression_path,
        leaf_dataset_json_path,
    ]
    if leaf_desc_json_path.exists():
        input_paths.append(leaf_desc_json_path)

    output_paths = [
        OUTPUT_DISCOVERY,
        OUTPUT_METADATA_SUMMARY,
        OUTPUT_MODULE_COVERAGE,
        OUTPUT_CELLTYPE_COUNTS,
        OUTPUT_README,
    ]

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "source": {
            "collection": COLLECTION_NAME,
            "collection_json": COLLECTION_JSON_URL,
            "resolved_dataset_name": selected[
                "dataset_name"
            ],
            "resolved_dataset_label": selected[
                "short_label"
            ],
            "resolved_base_url": selected["base_url"],
            "geo_accession": "GSE252470",
        },
        "outcome_loaded": False,
        "diagnostics": diagnostics,
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
        },
    }
    OUTPUT_MANIFEST.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("")
    print("=" * 80)
    print("Discovered Cell Browser leaves")
    print("=" * 80)
    print(discovery_table.to_string(index=False))

    print("")
    print("=" * 80)
    print("Metadata preflight")
    print("=" * 80)
    print(json.dumps(diagnostics, indent=2))

    print("")
    print("=" * 80)
    print("Frozen-module single-cell gene coverage")
    print("=" * 80)
    print(
        module_coverage[
            [
                "module_label",
                "n_frozen_genes",
                "n_detected_in_sc_matrix",
                "coverage_fraction",
                "n_positive_loading_genes_detected",
                "n_negative_loading_genes_detected",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Largest detected annotation groups")
    print("=" * 80)
    if celltype_counts.empty:
        print(
            "No cell-type annotation column was detected."
        )
    else:
        print(
            celltype_counts.sort_values(
                ["annotation_column", "n_cells"],
                ascending=[True, False],
            )
            .groupby(
                "annotation_column",
                as_index=False,
            )
            .head(15)
            .to_string(index=False)
        )

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("No clinical outcome or endpoint was loaded.")
    print(
        "The collection itself is not treated as a data leaf."
    )
    print(
        "Single cells must not be treated as independent "
        "biological replicates."
    )
    print(
        "Primary scoring will use dog-by-cell-type "
        "pseudobulk summaries."
    )
    print(
        "M34 and M40 positive and negative loading "
        "components will be analyzed separately."
    )

    print("")
    print("Saved:")
    for path in [
        OUTPUT_DISCOVERY,
        OUTPUT_METADATA_SUMMARY,
        OUTPUT_MODULE_COVERAGE,
        OUTPUT_CELLTYPE_COUNTS,
        OUTPUT_README,
        OUTPUT_MANIFEST,
    ]:
        print(path)
    print(f"Raw data directory: {RAW_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
