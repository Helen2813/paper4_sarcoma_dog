from __future__ import annotations

import gzip
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_VERSION = (
    "02d-audit-gse149679-pairing-v1"
)


SCRIPT_PATH = Path(__file__).resolve()

PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]


MANIFEST_DIR = (
    PAPER5_ROOT
    / "data_manifest"
)


CACHE_DIR = (
    MANIFEST_DIR
    / "geo_cache"
)


CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


GSE149679_SOFT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE149nnn/GSE149679/soft/"
    "GSE149679_family.soft.gz"
)


SOFT_FILE = (
    CACHE_DIR
    / "GSE149679_family.soft.gz"
)


DOG2_SUBJECT_FILE = (
    MANIFEST_DIR
    / "DOG2_subject_ids_authoritative.csv"
)


OUTPUT_GEO_METADATA = (
    MANIFEST_DIR
    / "GSE149679_sample_metadata.csv"
)


OUTPUT_COTC_MATCHES = (
    MANIFEST_DIR
    / "GSE149679_DOG2_COTC_overlap.csv"
)


OUTPUT_SPECIES_SUMMARY = (
    MANIFEST_DIR
    / "GSE149679_species_summary.csv"
)


OUTPUT_IDENTIFIER_AUDIT = (
    MANIFEST_DIR
    / "GSE149679_identifier_audit.csv"
)


OUTPUT_MANIFEST = (
    MANIFEST_DIR
    / "GSE149679_pairing_audit_manifest.json"
)


OUTPUT_README = (
    MANIFEST_DIR
    / "GSE149679_pairing_audit_README.txt"
)


COTC_PATTERN = re.compile(
    r"\b(COTC0?\d{2,3})[-_ ]+(\d{3,6})",
    flags=re.IGNORECASE,
)


DOG_PATTERN = re.compile(
    r"(dog|canine|oss|osteosarcoma)",
    flags=re.IGNORECASE,
)



def utc_now_iso():
    return datetime.now(
        timezone.utc
    ).isoformat()



def clean_text(value: Any):

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



def relative_path(path: Path):

    try:
        return str(
            path.resolve()
            .relative_to(
                PROJECT_ROOT.resolve()
            )
        )

    except ValueError:
        return str(path.resolve())



def normalize_cotc(
    study,
    subject,
):

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
        study_digits = (
            "0" + study_digits
        )

    return (
        f"COTC{study_digits}-"
        f"{subject_digits}"
    )



def extract_cotc_ids(
    value
):

    text = clean_text(value)

    output = set()

    for match in COTC_PATTERN.finditer(
        text
    ):

        output.add(
            normalize_cotc(
                match.group(1),
                match.group(2),
            )
        )

    return {
        x
        for x in output
        if x
    }



def download_soft():

    if SOFT_FILE.exists():

        print(
            f"Using cached GEO SOFT: {SOFT_FILE}"
        )

        return


    print(
        "Downloading GSE149679 GEO metadata..."
    )

    urllib.request.urlretrieve(
        GSE149679_SOFT_URL,
        SOFT_FILE,
    )


    print(
        f"Saved: {SOFT_FILE}"
    )



def parse_soft():

    records = []

    current = None


    with gzip.open(
        SOFT_FILE,
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
                    records.append(
                        current
                    )


                current = {

                    "gsm":
                        line.split(
                            "=",
                            1,
                        )[1].strip(),

                    "title":
                        "",

                    "source":
                        "",

                    "characteristics":
                        [],

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
                    )[1]
                    .strip()
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
                    )[1]
                    .strip()
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
                    )[1]
                    .strip()
                )


    if current is not None:
        records.append(
            current
        )


    rows = []


    for record in records:

        combined = " | ".join(
            [
                record["gsm"],
                record["title"],
                record["source"],
                *record[
                    "characteristics"
                ],
            ]
        )


        cotc = sorted(
            extract_cotc_ids(
                combined
            )
        )


        rows.append(
            {

                "gsm":
                    record["gsm"],

                "title":
                    record["title"],

                "source":
                    record["source"],

                "characteristics":
                    "; ".join(
                        record[
                            "characteristics"
                        ]
                    ),

                "cotc_ids":
                    ";".join(
                        cotc
                    ),

                "n_cotc_ids":
                    len(cotc),

                "contains_canine_keyword":
                    bool(
                        DOG_PATTERN.search(
                            combined
                        )
                    ),

            }
        )


    return pd.DataFrame(
        rows
    )



def load_dog2_subjects():

    if not DOG2_SUBJECT_FILE.exists():

        raise FileNotFoundError(
            DOG2_SUBJECT_FILE
        )


    table = pd.read_csv(
        DOG2_SUBJECT_FILE,
        dtype=str,
    ).fillna("")


    return set(
        table[
            "dog2_subject_id"
        ]
        .astype(str)
    )



def build_species_summary(
    metadata,
):

    rows = []

    rows.append(
        {
            "category":
                "total_samples",

            "n":
                len(metadata),
        }
    )


    rows.append(
        {
            "category":
                "samples_with_COTC",

            "n":
                int(
                    metadata[
                        "n_cotc_ids"
                    ]
                    .gt(0)
                    .sum()
                ),
        }
    )


    rows.append(
        {
            "category":
                "samples_with_canine_keyword",

            "n":
                int(
                    metadata[
                        "contains_canine_keyword"
                    ]
                    .sum()
                ),
        }
    )


    return pd.DataFrame(
        rows
    )



def build_overlap(
    metadata,
    dog2_ids,
):

    rows = []


    for _, row in metadata.iterrows():

        ids = [
            x
            for x in clean_text(
                row["cotc_ids"]
            )
            .split(";")
            if x
        ]


        overlap = sorted(
            set(ids)
            &
            dog2_ids
        )


        if overlap:

            rows.append(
                {
                    "gsm":
                        row["gsm"],

                    "title":
                        row["title"],

                    "cotc_ids":
                        ";".join(
                            ids
                        ),

                    "dog2_overlap":
                        ";".join(
                            overlap
                        ),

                }
            )


    return pd.DataFrame(
        rows
    )



def main():

    print("=" * 88)

    print(
        "Paper 5 - GSE149679 pairing audit"
    )

    print("=" * 88)

    print(
        f"Script version: {SCRIPT_VERSION}"
    )


    download_soft()


    metadata = parse_soft()


    metadata.to_csv(
        OUTPUT_GEO_METADATA,
        index=False,
    )


    print()

    print(
        f"GSE149679 samples parsed: "
        f"{metadata.shape[0]}"
    )


    dog2_ids = load_dog2_subjects()


    print(
        f"DOG2 authoritative subjects: "
        f"{len(dog2_ids)}"
    )


    summary = (
        build_species_summary(
            metadata
        )
    )


    summary.to_csv(
        OUTPUT_SPECIES_SUMMARY,
        index=False,
    )


    print()

    print(
        "Sample summary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )


    overlap = (
        build_overlap(
            metadata,
            dog2_ids,
        )
    )


    overlap.to_csv(
        OUTPUT_COTC_MATCHES,
        index=False,
    )


    print()

    print(
        "DOG2 exact COTC overlaps:"
    )


    if overlap.empty:

        print(
            "No exact COTC overlap detected."
        )

    else:

        print(
            overlap.to_string(
                index=False
            )
        )


    identifier_audit = metadata[
        [
            "gsm",
            "title",
            "source",
            "cotc_ids",
            "contains_canine_keyword",
        ]
    ]


    identifier_audit.to_csv(
        OUTPUT_IDENTIFIER_AUDIT,
        index=False,
    )


    manifest = {

        "script_version":
            SCRIPT_VERSION,

        "created_at_utc":
            utc_now_iso(),

        "outcome_loaded":
            False,

        "model_fitting":
            False,


        "inputs":

            {
                "geo_soft":
                    relative_path(
                        SOFT_FILE
                    ),

                "dog2_mapping":
                    relative_path(
                        DOG2_SUBJECT_FILE
                    ),
            },


        "results":

            {

                "n_gse149679_samples":
                    int(
                        metadata.shape[0]
                    ),

                "n_samples_with_cotc":
                    int(
                        metadata[
                            "n_cotc_ids"
                        ]
                        .gt(0)
                        .sum()
                    ),

                "n_exact_dog2_overlap":
                    int(
                        overlap.shape[0]
                    ),

            },


        "guardrails":

            [

                "No outcome data loaded",

                "No methylation-expression modeling performed",

                "Only identifier overlap audited",

                "Exact COTC matching required",

            ],

    }


    OUTPUT_MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


    OUTPUT_README.write_text(
        "\n".join(
            [
                "GSE149679 pairing audit",
                "=" * 40,
                "",
                "Purpose:",
                "Determine whether canine methylation samples",
                "can be linked to DOG2 RNA samples.",
                "",
                "No modeling performed.",
                "No outcomes loaded.",
            ]
        ),
        encoding="utf-8",
    )


    print()

    print("=" * 88)

    print(
        "GSE149679 pairing audit complete"
    )

    print("=" * 88)



if __name__ == "__main__":
    main()
