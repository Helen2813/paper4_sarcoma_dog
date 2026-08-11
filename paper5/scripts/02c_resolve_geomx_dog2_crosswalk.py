from __future__ import annotations

import gzip
import json
import re
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_VERSION = "02c-resolve-geomx-dog2-crosswalk-v1"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

MANIFEST_DIR = PAPER5_ROOT / "data_manifest"
CACHE_DIR = MANIFEST_DIR / "geo_cache"

MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DOG2_SUBJECT_FILE = (
    MANIFEST_DIR
    / "DOG2_subject_ids_authoritative.csv"
)

GSE295311_SOFT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE295nnn/GSE295311/soft/"
    "GSE295311_family.soft.gz"
)

GSE295311_SOFT_FILE = (
    CACHE_DIR / "GSE295311_family.soft.gz"
)

OUTPUT_GEO_METADATA = (
    MANIFEST_DIR
    / "GSE295311_GEO_sample_metadata.csv"
)

OUTPUT_LOCAL_EVIDENCE = (
    MANIFEST_DIR
    / "GSE295311_local_identifier_evidence.csv"
)

OUTPUT_ICDC_ALIAS_MAP = (
    MANIFEST_DIR
    / "ICDC_COTC_subject_alias_map.csv"
)

OUTPUT_CROSSWALK = (
    MANIFEST_DIR
    / "GSE295311_DOG2_crosswalk.csv"
)

OUTPUT_UNRESOLVED = (
    MANIFEST_DIR
    / "GSE295311_unresolved_identifiers.csv"
)

OUTPUT_MANIFEST = (
    MANIFEST_DIR
    / "GSE295311_DOG2_crosswalk_manifest.json"
)

OUTPUT_README = (
    MANIFEST_DIR
    / "GSE295311_DOG2_crosswalk_README.txt"
)

COTC_PATTERN = re.compile(
    r"\b(COTC0?\d{2,3})[-_ ]+(\d{3,6})",
    flags=re.IGNORECASE,
)

GSM_PATTERN = re.compile(
    r"\bGSM\d+\b",
    flags=re.IGNORECASE,
)

ID_COLUMN_PATTERN = re.compile(
    r"(id|patient|dog|animal|case|subject|sample|specimen)",
    flags=re.IGNORECASE,
)

FORBIDDEN_COLUMN_PATTERN = re.compile(
    r"(survival|dfi|outcome|response|death|event|"
    r"metast|recurr|prognos|treatment|therapy)",
    flags=re.IGNORECASE,
)

MAX_TABLE_BYTES = 150 * 1024 * 1024

PUBLISHED_EXPECTED_OVERLAP = 9


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


def normalize_cotc_id(
    study: str,
    subject: str,
) -> str:
    study_digits = re.sub(
        r"\D",
        "",
        study.upper()[4:],
    )

    subject_digits = re.sub(
        r"\D",
        "",
        subject,
    )

    if not study_digits or not subject_digits:
        return ""

    if len(study_digits) == 2:
        study_digits = "0" + study_digits

    return (
        f"COTC{study_digits}-{subject_digits}"
    )


def extract_cotc_ids(
    value: Any,
) -> set[str]:
    text = clean_text(value)

    output: set[str] = set()

    for match in COTC_PATTERN.finditer(text):
        subject_id = normalize_cotc_id(
            match.group(1),
            match.group(2),
        )

        if subject_id:
            output.add(subject_id)

    return output


def extract_gsm_ids(
    value: Any,
) -> set[str]:
    return {
        match.upper()
        for match in GSM_PATTERN.findall(
            clean_text(value)
        )
    }


def normalize_alias(
    value: Any,
) -> str:
    text = clean_text(value).upper()

    if not text:
        return ""

    text = re.sub(
        r"[^A-Z0-9]+",
        "",
        text,
    )

    if len(text) < 4:
        return ""

    # Reject generic values.
    if text in {
        "TRUE",
        "FALSE",
        "PRIMARY",
        "TUMOR",
        "TUMOUR",
        "CANINE",
        "OSTEOSARCOMA",
        "MALE",
        "FEMALE",
        "HIGH",
        "LOW",
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


# ---------------------------------------------------------------------
# GSE295311 GEO metadata
# ---------------------------------------------------------------------

def download_gse295311_soft() -> None:
    if GSE295311_SOFT_FILE.exists():
        print(
            "Using cached GSE295311 SOFT: "
            f"{GSE295311_SOFT_FILE}"
        )
        return

    print(
        "Downloading authoritative "
        "GSE295311 GEO metadata..."
    )

    urllib.request.urlretrieve(
        GSE295311_SOFT_URL,
        GSE295311_SOFT_FILE,
    )

    print(
        f"Saved: {GSE295311_SOFT_FILE}"
    )


def parse_gse295311_soft() -> pd.DataFrame:
    records = []

    current: dict[str, Any] | None = None

    with gzip.open(
        GSE295311_SOFT_FILE,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as handle:

        for raw_line in handle:
            line = raw_line.rstrip(
                "\r\n"
            )

            if line.startswith(
                "^SAMPLE = "
            ):
                if current is not None:
                    records.append(current)

                current = {
                    "gsm": (
                        line.split(
                            "=",
                            1,
                        )[1].strip()
                    ),
                    "title": "",
                    "source": "",
                    "characteristics": [],
                }

            elif (
                current is not None
                and line.startswith(
                    "!Sample_title = "
                )
            ):
                current["title"] = (
                    line.split(
                        "=",
                        1,
                    )[1].strip()
                )

            elif (
                current is not None
                and line.startswith(
                    "!Sample_source_name_ch1 = "
                )
            ):
                current["source"] = (
                    line.split(
                        "=",
                        1,
                    )[1].strip()
                )

            elif (
                current is not None
                and line.startswith(
                    "!Sample_characteristics_ch1 = "
                )
            ):
                current[
                    "characteristics"
                ].append(
                    line.split(
                        "=",
                        1,
                    )[1].strip()
                )

    if current is not None:
        records.append(current)

    rows = []

    for record in records:
        metadata_text = " | ".join(
            [
                record["gsm"],
                record["title"],
                record["source"],
                *record[
                    "characteristics"
                ],
            ]
        )

        cotc_ids = sorted(
            extract_cotc_ids(
                metadata_text
            )
        )

        rows.append(
            {
                "gsm": record["gsm"],
                "title": record["title"],
                "source": record["source"],
                "characteristics": "; ".join(
                    record[
                        "characteristics"
                    ]
                ),
                "cotc_ids": ";".join(
                    cotc_ids
                ),
                "n_cotc_ids": len(
                    cotc_ids
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Table readers
# ---------------------------------------------------------------------

def is_geomx_candidate(
    path: Path,
) -> bool:
    text = str(path).lower()
    name = path.name.lower()

    if "data_manifest" in text:
        return False

    return (
        "gse295311" in text
        or "geomx" in text
        or name
        == "nihms2085260-supplement-1.xlsx"
    )


def is_icdc_candidate(
    path: Path,
) -> bool:
    text = str(path).lower()

    if "data_manifest" in text:
        return False

    return (
        "cotc021" in text
        or "cotc022" in text
    )


def read_table_file(
    path: Path,
) -> list[
    tuple[str, pd.DataFrame]
]:
    if (
        not path.exists()
        or path.stat().st_size
        > MAX_TABLE_BYTES
    ):
        return []

    lower = path.name.lower()

    try:
        if lower.endswith(
            (".xlsx", ".xls")
        ):
            sheets = pd.read_excel(
                path,
                sheet_name=None,
                dtype=str,
            )

            return [
                (
                    str(sheet_name),
                    table.fillna(""),
                )
                for (
                    sheet_name,
                    table,
                ) in sheets.items()
            ]

        if lower.endswith(
            ".csv"
        ):
            table = pd.read_csv(
                path,
                dtype=str,
                low_memory=False,
            ).fillna("")

            return [
                ("table", table)
            ]

        if lower.endswith(
            (
                ".tsv",
                ".txt",
            )
        ):
            table = pd.read_csv(
                path,
                sep="\t",
                dtype=str,
                low_memory=False,
            ).fillna("")

            return [
                ("table", table)
            ]

    except Exception as exc:
        print(
            f"Warning: could not parse "
            f"{path}: {exc}"
        )

    return []


def identifier_columns(
    table: pd.DataFrame,
) -> list[str]:
    output = []

    for column in table.columns:
        name = str(column)

        if FORBIDDEN_COLUMN_PATTERN.search(
            name
        ):
            continue

        if ID_COLUMN_PATTERN.search(
            name
        ):
            output.append(column)

    return output


# ---------------------------------------------------------------------
# ICDC aliases
# ---------------------------------------------------------------------

def build_icdc_alias_map() -> pd.DataFrame:
    rows = []

    for path in PAPER5_ROOT.rglob(
        "*"
    ):
        if (
            not path.is_file()
            or not is_icdc_candidate(
                path
            )
        ):
            continue

        for sheet, table in read_table_file(
            path
        ):
            if table.empty:
                continue

            id_columns = identifier_columns(
                table
            )

            for row_index, row in (
                table.iterrows()
            ):
                row_text = " | ".join(
                    clean_text(value)
                    for value in row.values
                )

                cotc_ids = (
                    extract_cotc_ids(
                        row_text
                    )
                )

                if not cotc_ids:
                    continue

                # Only identifier-like columns are
                # allowed to contribute aliases.
                aliases = set()

                for column in id_columns:
                    alias = normalize_alias(
                        row.get(
                            column,
                            "",
                        )
                    )

                    if alias:
                        aliases.add(alias)

                for cotc_id in cotc_ids:
                    aliases.add(
                        normalize_alias(
                            cotc_id
                        )
                    )

                    for alias in aliases:
                        rows.append(
                            {
                                "cotc_subject_id": (
                                    cotc_id
                                ),
                                "alias": alias,
                                "source_file": (
                                    relative_path(
                                        path
                                    )
                                ),
                                "sheet": sheet,
                                "row_index": int(
                                    row_index
                                ),
                            }
                        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "cotc_subject_id",
                "alias",
                "source_file",
                "sheet",
                "row_index",
                "alias_unique",
            ]
        )

    table = (
        pd.DataFrame(rows)
        .drop_duplicates()
    )

    alias_counts = (
        table.groupby(
            "alias"
        )[
            "cotc_subject_id"
        ]
        .nunique()
    )

    table[
        "alias_unique"
    ] = (
        table["alias"]
        .map(alias_counts)
        .eq(1)
    )

    return table.sort_values(
        [
            "cotc_subject_id",
            "alias",
        ]
    ).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------
# GeoMx local evidence
# ---------------------------------------------------------------------

def build_geomx_identifier_evidence(
) -> pd.DataFrame:
    rows = []

    for path in PAPER5_ROOT.rglob(
        "*"
    ):
        if (
            not path.is_file()
            or not is_geomx_candidate(
                path
            )
        ):
            continue

        tables = read_table_file(
            path
        )

        for sheet, table in tables:
            if table.empty:
                continue

            id_columns = identifier_columns(
                table
            )

            for column in id_columns:
                for row_index, value in (
                    table[column]
                    .items()
                ):
                    text = clean_text(
                        value
                    )

                    if not text:
                        continue

                    rows.append(
                        {
                            "source_file": (
                                relative_path(
                                    path
                                )
                            ),
                            "sheet": sheet,
                            "column": str(
                                column
                            ),
                            "row_index": int(
                                row_index
                            ),
                            "raw_identifier": (
                                text
                            ),
                            "normalized_alias": (
                                normalize_alias(
                                    text
                                )
                            ),
                            "direct_cotc_ids": (
                                ";".join(
                                    sorted(
                                        extract_cotc_ids(
                                            text
                                        )
                                    )
                                )
                            ),
                            "gsm_ids": (
                                ";".join(
                                    sorted(
                                        extract_gsm_ids(
                                            text
                                        )
                                    )
                                )
                            ),
                        }
                    )

    if not rows:
        return pd.DataFrame(
            columns=[
                "source_file",
                "sheet",
                "column",
                "row_index",
                "raw_identifier",
                "normalized_alias",
                "direct_cotc_ids",
                "gsm_ids",
            ]
        )

    return (
        pd.DataFrame(rows)
        .drop_duplicates()
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Crosswalk
# ---------------------------------------------------------------------

def build_crosswalk(
    dog2: pd.DataFrame,
    aliases: pd.DataFrame,
    geomx: pd.DataFrame,
    geo_metadata: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    dog2_ids = set(
        dog2[
            "dog2_subject_id"
        ].astype(str)
    )

    unique_alias_map = {}

    if not aliases.empty:
        unique_aliases = aliases[
            aliases[
                "alias_unique"
            ].eq(True)
        ]

        for alias, part in (
            unique_aliases.groupby(
                "alias"
            )
        ):
            subjects = set(
                part[
                    "cotc_subject_id"
                ].astype(str)
            )

            if len(subjects) == 1:
                unique_alias_map[
                    alias
                ] = next(
                    iter(subjects)
                )

    resolved_rows = []
    unresolved_rows = []

    for _, row in geomx.iterrows():
        candidates = set()

        direct = [
            value
            for value in (
                clean_text(
                    row[
                        "direct_cotc_ids"
                    ]
                ).split(";")
            )
            if value
        ]

        for cotc_id in direct:
            if cotc_id in dog2_ids:
                candidates.add(
                    (
                        cotc_id,
                        "direct_COTC_identifier",
                    )
                )

        alias = clean_text(
            row[
                "normalized_alias"
            ]
        )

        if (
            alias
            and alias
            in unique_alias_map
        ):
            cotc_id = (
                unique_alias_map[
                    alias
                ]
            )

            if cotc_id in dog2_ids:
                candidates.add(
                    (
                        cotc_id,
                        "unique_ICDC_alias",
                    )
                )

        if candidates:
            for (
                cotc_id,
                method,
            ) in sorted(candidates):
                resolved_rows.append(
                    {
                        **row.to_dict(),
                        "dog2_subject_id": (
                            cotc_id
                        ),
                        "mapping_method": (
                            method
                        ),
                    }
                )
        else:
            unresolved_rows.append(
                row.to_dict()
            )

    # Add any direct COTC identifiers
    # publicly exposed by GEO itself.
    for _, row in (
        geo_metadata.iterrows()
    ):
        cotc_ids = [
            value
            for value in clean_text(
                row["cotc_ids"]
            ).split(";")
            if value
        ]

        for cotc_id in cotc_ids:
            if cotc_id not in dog2_ids:
                continue

            resolved_rows.append(
                {
                    "source_file": (
                        "GSE295311 GEO SOFT"
                    ),
                    "sheet": "",
                    "column": (
                        "GEO sample metadata"
                    ),
                    "row_index": "",
                    "raw_identifier": (
                        row["title"]
                    ),
                    "normalized_alias": "",
                    "direct_cotc_ids": (
                        cotc_id
                    ),
                    "gsm_ids": (
                        row["gsm"]
                    ),
                    "dog2_subject_id": (
                        cotc_id
                    ),
                    "mapping_method": (
                        "GEO_direct_COTC"
                    ),
                }
            )

    crosswalk = pd.DataFrame(
        resolved_rows
    )

    if not crosswalk.empty:
        crosswalk = (
            crosswalk
            .drop_duplicates(
                [
                    "dog2_subject_id",
                    "source_file",
                    "raw_identifier",
                    "mapping_method",
                ]
            )
            .sort_values(
                [
                    "dog2_subject_id",
                    "mapping_method",
                ]
            )
            .reset_index(drop=True)
        )

    unresolved = pd.DataFrame(
        unresolved_rows
    )

    return crosswalk, unresolved


def main() -> None:
    print("=" * 84)
    print(
        "Paper 5 - GSE295311 / DOG2 "
        "crosswalk audit"
    )
    print("=" * 84)
    print(
        f"Script version: "
        f"{SCRIPT_VERSION}"
    )
    print()

    if not DOG2_SUBJECT_FILE.exists():
        raise FileNotFoundError(
            "Run script 02b first. Missing: "
            f"{DOG2_SUBJECT_FILE}"
        )

    dog2 = pd.read_csv(
        DOG2_SUBJECT_FILE,
        dtype=str,
    ).fillna("")

    print(
        f"Authoritative DOG2 subjects: "
        f"{dog2.shape[0]}"
    )

    download_gse295311_soft()

    geo_metadata = (
        parse_gse295311_soft()
    )

    geo_metadata.to_csv(
        OUTPUT_GEO_METADATA,
        index=False,
    )

    print(
        f"GSE295311 GEO samples parsed: "
        f"{geo_metadata.shape[0]}"
    )

    n_geo_cotc = int(
        geo_metadata[
            "n_cotc_ids"
        ].gt(0).sum()
    )

    print(
        "GEO samples exposing COTC ID: "
        f"{n_geo_cotc}"
    )

    print()
    print(
        "Building ICDC identifier alias map..."
    )

    alias_map = (
        build_icdc_alias_map()
    )

    alias_map.to_csv(
        OUTPUT_ICDC_ALIAS_MAP,
        index=False,
    )

    print(
        f"ICDC alias records: "
        f"{alias_map.shape[0]}"
    )

    if not alias_map.empty:
        print(
            "Unique aliases usable for "
            "deterministic mapping: "
            f"{alias_map['alias_unique'].sum()}"
        )

    print()
    print(
        "Scanning local GSE295311 / "
        "GeoMx identifier fields..."
    )

    geomx_evidence = (
        build_geomx_identifier_evidence()
    )

    geomx_evidence.to_csv(
        OUTPUT_LOCAL_EVIDENCE,
        index=False,
    )

    print(
        f"GeoMx identifier records: "
        f"{geomx_evidence.shape[0]}"
    )

    crosswalk, unresolved = (
        build_crosswalk(
            dog2=dog2,
            aliases=alias_map,
            geomx=geomx_evidence,
            geo_metadata=geo_metadata,
        )
    )

    crosswalk.to_csv(
        OUTPUT_CROSSWALK,
        index=False,
    )

    unresolved.to_csv(
        OUTPUT_UNRESOLVED,
        index=False,
    )

    if crosswalk.empty:
        n_exact = 0
    else:
        n_exact = int(
            crosswalk[
                "dog2_subject_id"
            ].nunique()
        )

    print()
    print(
        "Resolved exact DOG2 / GeoMx "
        f"subjects: {n_exact}"
    )

    if not crosswalk.empty:
        summary = (
            crosswalk[
                [
                    "dog2_subject_id",
                    "mapping_method",
                ]
            ]
            .drop_duplicates()
        )

        print()
        print(
            summary.to_string(
                index=False
            )
        )

    print()
    print(
        "Published paired bulk/GeoMx "
        f"count: {PUBLISHED_EXPECTED_OVERLAP}"
    )

    if n_exact == PUBLISHED_EXPECTED_OVERLAP:
        status = (
            "PASS_EXACT_PUBLIC_CROSSWALK"
        )

        print(
            "Crosswalk sanity check: PASS"
        )

    elif n_exact == 0:
        status = (
            "PUBLIC_CROSSWALK_NOT_RESOLVED"
        )

        print(
            "Crosswalk sanity check: "
            "NOT RESOLVED"
        )

        print(
            "Do not assign the published "
            "nine paired cases by inference."
        )

        print(
            "GeoMx should remain an external "
            "spatial validation layer until "
            "a patient-ID crosswalk is found."
        )

    else:
        status = (
            "PARTIAL_PUBLIC_CROSSWALK"
        )

        print(
            "Crosswalk sanity check: PARTIAL"
        )

        print(
            "Use only the deterministically "
            "resolved patients; do not force "
            "the count to nine."
        )

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "outcome_loaded": False,
        "outcome_guided_matching": False,
        "published_expected_overlap": (
            PUBLISHED_EXPECTED_OVERLAP
        ),
        "resolved_exact_overlap": (
            n_exact
        ),
        "crosswalk_status": status,
        "guardrails": [
            (
                "Only identifier-like columns "
                "are used for alias construction."
            ),
            (
                "Survival, DFI, response, "
                "metastasis, recurrence, and "
                "treatment columns are excluded "
                "from matching."
            ),
            (
                "An ICDC alias is accepted only "
                "when it maps uniquely to one "
                "COTC subject."
            ),
            (
                "The published overlap count of "
                "nine is a sanity check and is "
                "never used to force assignments."
            ),
            (
                "No clinical outcome is loaded "
                "for molecular analysis."
            ),
        ],
        "outputs": {
            "geo_metadata": (
                relative_path(
                    OUTPUT_GEO_METADATA
                )
            ),
            "icdc_alias_map": (
                relative_path(
                    OUTPUT_ICDC_ALIAS_MAP
                )
            ),
            "local_identifier_evidence": (
                relative_path(
                    OUTPUT_LOCAL_EVIDENCE
                )
            ),
            "crosswalk": (
                relative_path(
                    OUTPUT_CROSSWALK
                )
            ),
            "unresolved": (
                relative_path(
                    OUTPUT_UNRESOLVED
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
        "GSE295311 / DOG2 crosswalk audit",
        "=" * 60,
        "",
        f"Script version: {SCRIPT_VERSION}",
        f"DOG2 subjects: {dog2.shape[0]}",
        (
            "GSE295311 GEO samples: "
            f"{geo_metadata.shape[0]}"
        ),
        (
            "Exact resolved DOG2/GeoMx "
            f"subjects: {n_exact}"
        ),
        (
            "Published cases with both "
            "GeoMx and bulk RNA: "
            f"{PUBLISHED_EXPECTED_OVERLAP}"
        ),
        f"Status: {status}",
        "",
        (
            "The published count is used only "
            "as a validation target, never to "
            "infer patient identities."
        ),
        "",
    ]

    OUTPUT_README.write_text(
        "\n".join(readme_lines),
        encoding="utf-8",
    )

    print()
    print("Outputs:")

    for path in [
        OUTPUT_GEO_METADATA,
        OUTPUT_ICDC_ALIAS_MAP,
        OUTPUT_LOCAL_EVIDENCE,
        OUTPUT_CROSSWALK,
        OUTPUT_UNRESOLVED,
        OUTPUT_MANIFEST,
        OUTPUT_README,
    ]:
        print(
            f"  {relative_path(path)}"
        )

    print("=" * 84)


if __name__ == "__main__":
    main()
