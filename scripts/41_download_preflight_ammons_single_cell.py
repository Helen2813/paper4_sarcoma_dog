from __future__ import annotations

from pathlib import Path
import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

SCRIPT_VERSION = "41-download-preflight-ammons-single-cell-v1"

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

BASE_URL = "https://cells.ucsc.edu/canine-os-atlas"
FILES = {
    "metadata": {
        "url": f"{BASE_URL}/meta.tsv",
        "path": RAW_DIR / "meta.tsv",
    },
    "expression": {
        "url": f"{BASE_URL}/exprMatrix.tsv.gz",
        "path": RAW_DIR / "exprMatrix.tsv.gz",
    },
    "dataset_json": {
        "url": f"{BASE_URL}/dataset.json",
        "path": RAW_DIR / "dataset.json",
    },
    "description_json": {
        "url": f"{BASE_URL}/desc.json",
        "path": RAW_DIR / "desc.json",
    },
}

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]

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

    partial = destination.with_suffix(destination.suffix + ".part")
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
            timeout=(30, 180),
            headers=headers,
            allow_redirects=True,
        ) as response:
            if response.status_code == 416 and partial.exists():
                partial.replace(destination)
                return True

            if response.status_code not in {200, 206}:
                message = (
                    f"Download failed with HTTP {response.status_code}: "
                    f"{url}"
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
        # Cell Browser matrices sometimes store SYMBOL|ENSEMBL.
        text = text.split("|", 1)[0].strip()
    return text


def candidate_columns(
    columns: Iterable[str],
    exact_candidates: list[str],
    contains_candidates: list[str],
) -> list[str]:
    columns_list = list(columns)
    lower_map = {str(column).lower(): str(column) for column in columns_list}

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


def metadata_preflight(metadata_path: Path) -> tuple[pd.DataFrame, dict]:
    metadata = pd.read_csv(
        metadata_path,
        sep="\t",
        dtype=str,
        low_memory=False,
    )

    if metadata.empty:
        raise RuntimeError("The Cell Browser metadata table is empty.")

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
    if not cell_id_candidates:
        cell_id_column = str(metadata.columns[0])
    else:
        cell_id_column = cell_id_candidates[0]

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
                    series.dropna().astype(str).drop_duplicates().head(5)
                ),
                "is_cell_id_candidate": str(column) == cell_id_column,
                "is_celltype_candidate": str(column) in celltype_columns,
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


def read_expression_header(
    expression_path: Path,
) -> tuple[list[str], str]:
    with gzip.open(expression_path, "rt", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n\r").split("\t")

    if len(header) < 2:
        raise RuntimeError(
            "The expression matrix header has fewer than two columns."
        )

    first_column = header[0]
    cell_ids = header[1:]
    return cell_ids, first_column


def scan_expression_genes(
    expression_path: Path,
    requested_genes: set[str],
) -> tuple[set[str], int]:
    found = set()
    n_rows = 0

    with gzip.open(expression_path, "rt", encoding="utf-8") as handle:
        header = handle.readline()
        for line in handle:
            if not line.strip():
                continue
            gene = clean_gene_symbol(line.split("\t", 1)[0])
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
        present = sorted(set(module_genes).intersection(found_genes))
        missing = sorted(set(module_genes).difference(found_genes))

        positive = set(
            part.loc[
                pd.to_numeric(
                    part["risk_oriented_loading"],
                    errors="coerce",
                ) > 0,
                "human_gene_symbol",
            ]
        )
        negative = set(
            part.loc[
                pd.to_numeric(
                    part["risk_oriented_loading"],
                    errors="coerce",
                ) < 0,
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
                    "fraction_cells": float(count / metadata.shape[0]),
                }
            )

    return pd.DataFrame(rows)


def write_readme(diagnostics: dict) -> None:
    text = f"""Ammons canine osteosarcoma single-cell atlas preflight
Script version: {SCRIPT_VERSION}

Source
------
UCSC Cell Browser dataset: canine-os-atlas
Publication dataset accession: GSE252470

Downloaded files
----------------
- meta.tsv
- exprMatrix.tsv.gz
- dataset.json and desc.json when available

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

The next analysis should:
1. read only frozen program genes from the compressed matrix;
2. compute separate positive-loading and negative-loading scores;
3. compute the signed risk-oriented score;
4. summarize cells by dog x cell type using pseudobulk statistics;
5. use dog as the biological replicate, not individual cells;
6. test M34 and M40 as primary single-cell localization analyses;
7. retain M11/M24 as secondary or non-interpretable when coverage is limited.

No clinical outcome is loaded.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("Download and preflight Ammons canine OS single-cell atlas")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Design:")
    print("  Download Python-readable UCSC Cell Browser files.")
    print("  Avoid the 2.1-GB Seurat RDS because R is not required.")
    print("  Detect cell-type and dog/sample metadata columns.")
    print("  Scan frozen-module gene coverage without loading the full matrix.")
    print("  Calculate no outcomes and no single-cell scores yet.")
    print("")

    if not STRICT_WEIGHTS_FILE.exists():
        raise FileNotFoundError(
            f"Frozen weights not found: {STRICT_WEIGHTS_FILE}"
        )

    download_with_resume(
        FILES["metadata"]["url"],
        FILES["metadata"]["path"],
        required=True,
    )
    download_with_resume(
        FILES["expression"]["url"],
        FILES["expression"]["path"],
        required=True,
    )
    download_with_resume(
        FILES["dataset_json"]["url"],
        FILES["dataset_json"]["path"],
        required=False,
    )
    download_with_resume(
        FILES["description_json"]["url"],
        FILES["description_json"]["path"],
        required=False,
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
        FILES["metadata"]["path"]
    )
    expression_cell_ids, gene_column = read_expression_header(
        FILES["expression"]["path"]
    )

    diagnostics["expression_gene_column"] = gene_column
    diagnostics["n_expression_cells"] = len(expression_cell_ids)

    metadata = pd.read_csv(
        FILES["metadata"]["path"],
        sep="\t",
        dtype=str,
        low_memory=False,
    )
    metadata_cell_ids = set(
        metadata[diagnostics["cell_id_column"]].astype(str)
    )
    expression_cell_id_set = set(map(str, expression_cell_ids))

    diagnostics["n_cell_ids_intersection"] = len(
        metadata_cell_ids.intersection(expression_cell_id_set)
    )
    diagnostics["metadata_cell_match_fraction"] = (
        diagnostics["n_cell_ids_intersection"]
        / max(1, len(metadata_cell_ids))
    )
    diagnostics["expression_cell_match_fraction"] = (
        diagnostics["n_cell_ids_intersection"]
        / max(1, len(expression_cell_id_set))
    )

    found_genes, n_expression_gene_rows = scan_expression_genes(
        FILES["expression"]["path"],
        requested_genes,
    )
    diagnostics["n_expression_gene_rows"] = n_expression_gene_rows

    module_coverage = build_module_coverage(
        weights=weights,
        found_genes=found_genes,
    )
    celltype_counts = build_celltype_counts(
        metadata_path=FILES["metadata"]["path"],
        celltype_columns=diagnostics["celltype_columns"],
    )

    metadata_summary.to_csv(OUTPUT_METADATA_SUMMARY, index=False)
    module_coverage.to_csv(OUTPUT_MODULE_COVERAGE, index=False)
    celltype_counts.to_csv(OUTPUT_CELLTYPE_COUNTS, index=False)
    write_readme(diagnostics)

    input_paths = [
        STRICT_WEIGHTS_FILE,
        FILES["metadata"]["path"],
        FILES["expression"]["path"],
    ]
    for optional_key in ["dataset_json", "description_json"]:
        optional_path = FILES[optional_key]["path"]
        if optional_path.exists():
            input_paths.append(optional_path)

    output_paths = [
        OUTPUT_METADATA_SUMMARY,
        OUTPUT_MODULE_COVERAGE,
        OUTPUT_CELLTYPE_COUNTS,
        OUTPUT_README,
    ]

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "source": {
            "dataset": "Ammons canine osteosarcoma scRNA-seq atlas",
            "geo_accession": "GSE252470",
            "cell_browser": BASE_URL,
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
        print("No cell-type annotation column was detected.")
    else:
        print(
            celltype_counts.sort_values(
                ["annotation_column", "n_cells"],
                ascending=[True, False],
            )
            .groupby("annotation_column", as_index=False)
            .head(15)
            .to_string(index=False)
        )

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("No clinical outcome or endpoint was loaded.")
    print("This script performs download and data-structure validation only.")
    print("Single cells must not be treated as independent biological replicates.")
    print("Primary scoring will use dog-by-cell-type pseudobulk summaries.")
    print("M34 and M40 positive and negative loading components will be analyzed separately.")

    print("")
    print("Saved:")
    for path in [
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
