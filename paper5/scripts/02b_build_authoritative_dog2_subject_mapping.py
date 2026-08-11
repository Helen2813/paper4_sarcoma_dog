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


SCRIPT_VERSION = "02b-build-authoritative-dog2-subject-mapping-v1"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

DATA_MANIFEST_DIR = PAPER5_ROOT / "data_manifest"
CACHE_DIR = PAPER5_ROOT / "data_manifest" / "geo_cache"

DATA_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

GSE238110_SOFT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE238nnn/GSE238110/soft/GSE238110_family.soft.gz"
)

GSE238110_SOFT_FILE = (
    CACHE_DIR / "GSE238110_family.soft.gz"
)

CANINE_EVIDENCE_FILE = (
    DATA_MANIFEST_DIR
    / "canine_downloaded_subject_id_evidence.csv"
)

OUTPUT_SAMPLE_MAPPING = (
    DATA_MANIFEST_DIR
    / "DOG2_GEO_GSM_COTC_mapping.csv"
)

OUTPUT_SUBJECTS = (
    DATA_MANIFEST_DIR
    / "DOG2_subject_ids_authoritative.csv"
)

OUTPUT_OVERLAP = (
    DATA_MANIFEST_DIR
    / "DOG2_cross_platform_subject_overlap_authoritative.csv"
)

OUTPUT_MANIFEST = (
    DATA_MANIFEST_DIR
    / "DOG2_authoritative_mapping_manifest.json"
)

OUTPUT_README = (
    DATA_MANIFEST_DIR
    / "DOG2_authoritative_mapping_README.txt"
)

COTC_PATTERN = re.compile(
    r"\b(COTC0?\d{2,3})[-_ ]+(\d{3,6})(?:[-_ ]+[A-Z])?\b",
    flags=re.IGNORECASE,
)


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
    study = study.upper()

    study_digits = re.sub(
        r"\D",
        "",
        study[4:],
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


def extract_cotc_id(
    value: Any,
) -> str:
    text = clean_text(value)

    if not text:
        return ""

    match = COTC_PATTERN.search(text)

    if match is None:
        return ""

    return normalize_cotc_id(
        match.group(1),
        match.group(2),
    )


def download_soft() -> None:
    if GSE238110_SOFT_FILE.exists():
        print(
            f"Using cached GEO SOFT: "
            f"{GSE238110_SOFT_FILE}"
        )
        return

    print("Downloading authoritative GSE238110 GEO metadata...")
    print(GSE238110_SOFT_URL)

    urllib.request.urlretrieve(
        GSE238110_SOFT_URL,
        GSE238110_SOFT_FILE,
    )

    print(
        f"Saved: {GSE238110_SOFT_FILE}"
    )


def parse_gse238110_soft() -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    current_gsm = ""
    current_title = ""

    with gzip.open(
        GSE238110_SOFT_FILE,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for raw_line in handle:
            line = raw_line.rstrip(
                "\n\r"
            )

            if line.startswith("^SAMPLE = "):
                if current_gsm:
                    rows.append(
                        {
                            "gsm": current_gsm,
                            "sample_title": current_title,
                        }
                    )

                current_gsm = (
                    line.split(
                        "=",
                        1,
                    )[1]
                    .strip()
                )

                current_title = ""

            elif line.startswith(
                "!Sample_title = "
            ):
                current_title = (
                    line.split(
                        "=",
                        1,
                    )[1]
                    .strip()
                )

        if current_gsm:
            rows.append(
                {
                    "gsm": current_gsm,
                    "sample_title": current_title,
                }
            )

    mapping = pd.DataFrame(rows)

    if mapping.empty:
        raise RuntimeError(
            "No GEO samples were parsed from GSE238110 SOFT."
        )

    mapping["cotc_subject_id"] = (
        mapping["sample_title"]
        .map(extract_cotc_id)
    )

    mapping["has_cotc_id"] = (
        mapping["cotc_subject_id"]
        .ne("")
    )

    return mapping


def validate_mapping(
    mapping: pd.DataFrame,
) -> None:
    n_samples = int(
        mapping.shape[0]
    )

    n_with_cotc = int(
        mapping["has_cotc_id"].sum()
    )

    n_unique_subjects = int(
        mapping.loc[
            mapping["has_cotc_id"],
            "cotc_subject_id",
        ].nunique()
    )

    print()
    print("Authoritative GSE238110 mapping:")
    print(f"  GEO samples: {n_samples}")
    print(
        f"  Samples with COTC ID: "
        f"{n_with_cotc}"
    )
    print(
        f"  Unique COTC subjects: "
        f"{n_unique_subjects}"
    )

    if n_samples != 186:
        raise RuntimeError(
            "Expected 186 GSE238110 samples, "
            f"found {n_samples}."
        )

    if n_with_cotc != 186:
        missing = mapping.loc[
            ~mapping["has_cotc_id"],
            [
                "gsm",
                "sample_title",
            ],
        ]

        missing_file = (
            DATA_MANIFEST_DIR
            / "DOG2_GEO_samples_missing_COTC_id.csv"
        )

        missing.to_csv(
            missing_file,
            index=False,
        )

        raise RuntimeError(
            "Not all 186 GEO samples contain a "
            "parseable COTC identifier. "
            f"See {missing_file}"
        )

    if n_unique_subjects != 186:
        duplicates = (
            mapping[
                "cotc_subject_id"
            ]
            .value_counts()
            .loc[lambda x: x > 1]
        )

        print()
        print(
            "Warning: repeated COTC subject IDs:"
        )
        print(duplicates)


def build_subject_table(
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for subject_id, part in mapping.groupby(
        "cotc_subject_id"
    ):
        rows.append(
            {
                "dog2_subject_id": subject_id,
                "n_geo_samples": int(
                    part.shape[0]
                ),
                "gsm_ids": ";".join(
                    sorted(
                        part["gsm"]
                        .astype(str)
                    )
                ),
                "sample_titles": ";".join(
                    sorted(
                        part["sample_title"]
                        .astype(str)
                    )
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "dog2_subject_id"
        )
        .reset_index(drop=True)
    )


def build_cross_platform_overlap(
    subjects: pd.DataFrame,
) -> pd.DataFrame:
    if not CANINE_EVIDENCE_FILE.exists():
        raise FileNotFoundError(
            "Run script 02 first. Missing: "
            f"{CANINE_EVIDENCE_FILE}"
        )

    evidence = pd.read_csv(
        CANINE_EVIDENCE_FILE,
        dtype=str,
    ).fillna("")

    required = {
        "dataset",
        "subject_id",
        "source_file",
    }

    missing = required - set(
        evidence.columns
    )

    if missing:
        raise ValueError(
            "Canine evidence table is missing: "
            + ", ".join(
                sorted(missing)
            )
        )

    dog2_set = set(
        subjects[
            "dog2_subject_id"
        ].astype(str)
    )

    rows = []

    for dataset, part in evidence.groupby(
        "dataset"
    ):
        observed = set(
            part[
                "subject_id"
            ].astype(str)
        )

        overlap = sorted(
            dog2_set
            & observed
        )

        source_files = sorted(
            set(
                part.loc[
                    part[
                        "subject_id"
                    ].isin(overlap),
                    "source_file",
                ].astype(str)
            )
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
                "n_exact_dog2_overlap": len(
                    overlap
                ),
                "overlap_subject_ids": ";".join(
                    overlap
                ),
                "overlap_source_files": ";".join(
                    source_files
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "n_exact_dog2_overlap",
                "dataset",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def main() -> None:
    print("=" * 84)
    print(
        "Paper 5 - authoritative DOG2 patient mapping"
    )
    print("=" * 84)
    print(
        f"Script version: {SCRIPT_VERSION}"
    )
    print()

    download_soft()

    mapping = parse_gse238110_soft()

    validate_mapping(
        mapping
    )

    mapping.to_csv(
        OUTPUT_SAMPLE_MAPPING,
        index=False,
    )

    subjects = build_subject_table(
        mapping
    )

    subjects.to_csv(
        OUTPUT_SUBJECTS,
        index=False,
    )

    overlap = (
        build_cross_platform_overlap(
            subjects
        )
    )

    overlap.to_csv(
        OUTPUT_OVERLAP,
        index=False,
    )

    print()
    print("Cross-platform DOG2 overlaps:")
    print()

    if overlap.empty:
        print(
            "No downloaded COTC datasets "
            "were available for comparison."
        )
    else:
        print(
            overlap[
                [
                    "dataset",
                    "n_downloaded_subject_ids",
                    "n_exact_dog2_overlap",
                    "overlap_subject_ids",
                ]
            ].to_string(
                index=False
            )
        )

    geomx_overlap = 0

    if not overlap.empty:
        part = overlap[
            overlap[
                "dataset"
            ].eq("GSE295311")
        ]

        if not part.empty:
            geomx_overlap = int(
                part.iloc[0][
                    "n_exact_dog2_overlap"
                ]
            )

    print()
    print(
        f"GSE295311 exact DOG2 overlap: "
        f"{geomx_overlap}"
    )

    if geomx_overlap != 9:
        print(
            "Note: the GeoMx publication reports "
            "9 cases with both GeoMx and bulk RNA. "
            "If the exact-ID overlap is not 9, "
            "inspect the downloaded GeoMx metadata "
            "before model construction."
        )

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "outcome_loaded": False,
        "mapping_source": {
            "accession": "GSE238110",
            "url": GSE238110_SOFT_URL,
            "cached_file": str(
                GSE238110_SOFT_FILE
            ),
        },
        "n_geo_samples": int(
            mapping.shape[0]
        ),
        "n_samples_with_cotc_id": int(
            mapping[
                "has_cotc_id"
            ].sum()
        ),
        "n_unique_dog2_subjects": int(
            subjects.shape[0]
        ),
        "gse295311_exact_overlap": (
            geomx_overlap
        ),
        "guardrails": [
            (
                "DOG2 COTC identifiers are derived "
                "from authoritative GSE238110 GEO "
                "sample metadata."
            ),
            (
                "No clinical outcome is loaded or "
                "used for patient matching."
            ),
            (
                "Cross-platform pairing requires an "
                "exact normalized COTC subject ID."
            ),
        ],
        "outputs": {
            "sample_mapping": str(
                OUTPUT_SAMPLE_MAPPING
            ),
            "subject_table": str(
                OUTPUT_SUBJECTS
            ),
            "cross_platform_overlap": str(
                OUTPUT_OVERLAP
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

    readme = [
        "Paper 5 authoritative DOG2 subject mapping",
        "=" * 60,
        "",
        f"Script version: {SCRIPT_VERSION}",
        "",
        f"GEO samples: {mapping.shape[0]}",
        (
            "Samples with COTC ID: "
            f"{mapping['has_cotc_id'].sum()}"
        ),
        (
            "Unique DOG2 subjects: "
            f"{subjects.shape[0]}"
        ),
        (
            "GSE295311 exact DOG2 overlap: "
            f"{geomx_overlap}"
        ),
        "",
        "No clinical outcome was loaded.",
        "",
    ]

    OUTPUT_README.write_text(
        "\n".join(readme),
        encoding="utf-8",
    )

    print()
    print("Outputs:")
    print(
        f"  {OUTPUT_SAMPLE_MAPPING}"
    )
    print(
        f"  {OUTPUT_SUBJECTS}"
    )
    print(
        f"  {OUTPUT_OVERLAP}"
    )
    print(
        f"  {OUTPUT_MANIFEST}"
    )
    print("=" * 84)


if __name__ == "__main__":
    main()
