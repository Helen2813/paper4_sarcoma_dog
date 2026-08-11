from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_VERSION = "03d-audit-repair-target-methylation-files-v1"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

TARGET_ROOT = (
    PAPER5_ROOT
    / "human"
    / "TARGET_OS"
)

MODEL_INPUT_DIR = (
    PAPER5_ROOT
    / "model_inputs"
    / "TARGET_OS_primary78"
)

FILE_MAP = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_multimodal_file_map.csv"
)

OUTPUT_AUDIT = (
    MODEL_INPUT_DIR
    / "TARGET_OS_methylation_download_integrity_audit.csv"
)

OUTPUT_MANIFEST = (
    MODEL_INPUT_DIR
    / "TARGET_OS_methylation_download_integrity_manifest.json"
)

OUTPUT_README = (
    MODEL_INPUT_DIR
    / "TARGET_OS_methylation_download_integrity_README.txt"
)

METHYLATION_SUFFIX = (
    ".methylation_array.sesame.level3betas.txt"
)

EXPECTED_PROCESSED_FILES = 86

DOWNLOAD_ENDPOINT = (
    "https://api.gdc.cancer.gov/data/{file_id}"
)


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


def normalize_column(value: Any) -> str:
    return (
        clean_text(value)
        .lower()
        .replace(" ", "_")
    )


def relative_path(path: Path) -> str:
    try:
        return str(
            path.resolve().relative_to(
                PROJECT_ROOT.resolve()
            )
        )
    except ValueError:
        return str(path.resolve())


def md5_file(path: Path) -> str:
    digest = hashlib.md5()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                4 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


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


def load_manifest_records() -> pd.DataFrame:
    paths = sorted(
        TARGET_ROOT.rglob(
            "gdc_manifest*.txt"
        )
    )

    if not paths:
        raise FileNotFoundError(
            "No gdc_manifest*.txt files were found below "
            f"{TARGET_ROOT}"
        )

    tables = []

    for path in paths:
        try:
            table = pd.read_csv(
                path,
                sep="\t",
                dtype=str,
                low_memory=False,
            ).fillna("")
        except Exception as exc:
            print(
                f"Warning: could not read "
                f"{path}: {exc}"
            )
            continue

        table.columns = [
            normalize_column(column)
            for column in table.columns
        ]

        required = {
            "id",
            "filename",
            "md5",
            "size",
        }

        if not required.issubset(
            table.columns
        ):
            continue

        table = table[
            table["filename"]
            .astype(str)
            .str.lower()
            .str.endswith(
                METHYLATION_SUFFIX.lower()
            )
        ].copy()

        if table.empty:
            continue

        table["source_manifest"] = (
            relative_path(path)
        )

        tables.append(table)

    if not tables:
        raise RuntimeError(
            "No processed SeSAMe beta-value records "
            "were found in the GDC manifests."
        )

    combined = pd.concat(
        tables,
        ignore_index=True,
    )

    # A file can appear in more than one browser-generated manifest.
    # All repeated records must agree.
    conflicts = []

    for filename, part in combined.groupby(
        "filename"
    ):
        ids = set(
            part["id"].astype(str)
        )

        md5s = set(
            part["md5"].astype(str)
        )

        sizes = set(
            part["size"].astype(str)
        )

        if (
            len(ids) > 1
            or len(md5s) > 1
            or len(sizes) > 1
        ):
            conflicts.append(
                filename
            )

    if conflicts:
        raise RuntimeError(
            "Conflicting GDC manifest records found for: "
            + ", ".join(
                conflicts[:10]
            )
        )

    combined = (
        combined.sort_values(
            [
                "filename",
                "source_manifest",
            ]
        )
        .drop_duplicates(
            "filename",
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    combined["expected_size"] = pd.to_numeric(
        combined["size"],
        errors="raise",
    ).astype("int64")

    if (
        combined.shape[0]
        != EXPECTED_PROCESSED_FILES
    ):
        raise RuntimeError(
            "Expected "
            f"{EXPECTED_PROCESSED_FILES} unique processed "
            "methylation files in manifests; found "
            f"{combined.shape[0]}."
        )

    return combined


def build_local_lookup() -> dict[
    str,
    list[Path],
]:
    lookup: dict[
        str,
        list[Path],
    ] = {}

    for path in TARGET_ROOT.rglob(
        f"*{METHYLATION_SUFFIX}"
    ):
        if not path.is_file():
            continue

        lookup.setdefault(
            path.name,
            [],
        ).append(path)

    return lookup


def load_case_map() -> dict[str, str]:
    if not FILE_MAP.exists():
        return {}

    table = pd.read_csv(
        FILE_MAP,
        dtype=str,
    ).fillna("")

    required = {
        "case_id",
        "modality",
        "file_name",
    }

    if not required.issubset(
        table.columns
    ):
        return {}

    table = table[
        table["modality"]
        .eq("methylation")
    ]

    return {
        str(row["file_name"]): str(
            row["case_id"]
        )
        for _, row in table.iterrows()
    }


def inspect_all_files(
    records: pd.DataFrame,
) -> pd.DataFrame:
    local_lookup = (
        build_local_lookup()
    )

    case_map = (
        load_case_map()
    )

    rows = []

    print()
    print(
        "Auditing local files against GDC manifest..."
    )

    for index, record in (
        records.iterrows()
    ):
        filename = str(
            record["filename"]
        )

        file_id = str(
            record["id"]
        )

        expected_md5 = str(
            record["md5"]
        ).lower()

        expected_size = int(
            record["expected_size"]
        )

        matches = local_lookup.get(
            filename,
            [],
        )

        local_path = (
            matches[0]
            if matches
            else None
        )

        duplicate_local_files = (
            len(matches)
        )

        if local_path is None:
            observed_size = None
            observed_md5 = ""
            size_match = False
            md5_match = False
            status = "MISSING"

        else:
            observed_size = int(
                local_path.stat().st_size
            )

            size_match = (
                observed_size
                == expected_size
            )

            # Hash even a size-mismatched file so the audit
            # records exactly what was present locally.
            observed_md5 = (
                md5_file(
                    local_path
                )
            )

            md5_match = (
                observed_md5.lower()
                == expected_md5
            )

            if (
                size_match
                and md5_match
                and duplicate_local_files == 1
            ):
                status = "PASS"

            elif (
                size_match
                and md5_match
                and duplicate_local_files > 1
            ):
                status = (
                    "PASS_DUPLICATE_LOCAL_COPY"
                )

            elif not size_match:
                status = "SIZE_MISMATCH"

            else:
                status = "MD5_MISMATCH"

        rows.append(
            {
                "case_id": (
                    case_map.get(
                        filename,
                        "",
                    )
                ),
                "file_id": file_id,
                "filename": filename,
                "state": clean_text(
                    record.get(
                        "state",
                        "",
                    )
                ),
                "expected_size": (
                    expected_size
                ),
                "observed_size": (
                    observed_size
                ),
                "size_difference": (
                    (
                        observed_size
                        - expected_size
                    )
                    if observed_size
                    is not None
                    else None
                ),
                "expected_md5": (
                    expected_md5
                ),
                "observed_md5": (
                    observed_md5
                ),
                "size_match": (
                    size_match
                ),
                "md5_match": (
                    md5_match
                ),
                "n_local_copies": (
                    duplicate_local_files
                ),
                "local_path": (
                    relative_path(
                        local_path
                    )
                    if local_path
                    is not None
                    else ""
                ),
                "source_manifest": (
                    record[
                        "source_manifest"
                    ]
                ),
                "status": status,
            }
        )

        if (
            status != "PASS"
            or (
                index + 1
            ) % 10 == 0
            or index + 1
            == records.shape[0]
        ):
            print(
                f"{index + 1:02d}/"
                f"{records.shape[0]} "
                f"{case_map.get(filename, '') or 'unmapped'} "
                f"{status} "
                f"observed="
                f"{observed_size if observed_size is not None else 'NA'} "
                f"expected="
                f"{expected_size}"
            )

    return pd.DataFrame(
        rows
    )


def choose_destination(
    row: pd.Series,
) -> Path:
    local_path = clean_text(
        row[
            "local_path"
        ]
    )

    if local_path:
        path = Path(
            local_path
        )

        if not path.is_absolute():
            path = (
                PROJECT_ROOT
                / path
            )

        return path

    # If a file is fully absent, preserve the GDC UUID layout.
    destination = (
        TARGET_ROOT
        / "methilation"
        / str(
            row["file_id"]
        )
        / str(
            row["filename"]
        )
    )

    return destination


def download_file(
    file_id: str,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    url = DOWNLOAD_ENDPOINT.format(
        file_id=file_id
    )

    print(
        f"  URL: {url}"
    )

    fd, temporary_name = (
        tempfile.mkstemp(
            prefix=(
                destination.name
                + "."
            ),
            suffix=".part",
            dir=str(
                destination.parent
            ),
        )
    )

    os.close(fd)

    temporary = Path(
        temporary_name
    )

    try:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "paper5-target-methylation-"
                    "integrity-audit/1.0"
                )
            },
        )

        with urllib.request.urlopen(
            request,
            timeout=120,
        ) as response:
            with temporary.open(
                "wb"
            ) as handle:
                while True:
                    chunk = response.read(
                        4 * 1024 * 1024
                    )

                    if not chunk:
                        break

                    handle.write(
                        chunk
                    )

        os.replace(
            temporary,
            destination,
        )

    finally:
        if temporary.exists():
            temporary.unlink()


def repair_failed_files(
    audit: pd.DataFrame,
) -> None:
    failed = audit[
        ~audit["status"].isin(
            [
                "PASS",
                "PASS_DUPLICATE_LOCAL_COPY",
            ]
        )
    ].copy()

    if failed.empty:
        print()
        print(
            "No files require repair."
        )
        return

    print()
    print(
        "Repairing failed methylation downloads:"
    )

    for _, row in failed.iterrows():
        case_id = clean_text(
            row["case_id"]
        )

        filename = str(
            row["filename"]
        )

        file_id = str(
            row["file_id"]
        )

        destination = (
            choose_destination(
                row
            )
        )

        print()
        print(
            f"Downloading "
            f"{case_id or 'unmapped'}:"
        )

        print(
            f"  {filename}"
        )

        download_file(
            file_id=file_id,
            destination=destination,
        )

        observed_size = int(
            destination.stat().st_size
        )

        observed_md5 = (
            md5_file(
                destination
            )
        )

        expected_size = int(
            row["expected_size"]
        )

        expected_md5 = str(
            row["expected_md5"]
        ).lower()

        if (
            observed_size
            != expected_size
        ):
            raise RuntimeError(
                "Repair download has wrong size for "
                f"{filename}: "
                f"{observed_size} vs "
                f"{expected_size}"
            )

        if (
            observed_md5.lower()
            != expected_md5
        ):
            raise RuntimeError(
                "Repair download has wrong MD5 for "
                f"{filename}: "
                f"{observed_md5} vs "
                f"{expected_md5}"
            )

        print(
            "  Repair verification: PASS"
        )


def write_outputs(
    audit: pd.DataFrame,
    repair_requested: bool,
) -> None:
    audit.to_csv(
        OUTPUT_AUDIT,
        index=False,
    )

    status_counts = (
        audit["status"]
        .value_counts()
        .to_dict()
    )

    n_pass = int(
        audit["status"]
        .isin(
            [
                "PASS",
                "PASS_DUPLICATE_LOCAL_COPY",
            ]
        )
        .sum()
    )

    n_failed = int(
        audit.shape[0]
        - n_pass
    )

    manifest = {
        "script_version": (
            SCRIPT_VERSION
        ),
        "created_at_utc": (
            utc_now_iso()
        ),
        "outcome_loaded": False,
        "model_fitting_performed": False,
        "feature_selection_performed": False,
        "repair_requested": bool(
            repair_requested
        ),
        "n_expected_files": int(
            audit.shape[0]
        ),
        "n_integrity_pass": (
            n_pass
        ),
        "n_integrity_fail": (
            n_failed
        ),
        "status_counts": {
            str(key): int(value)
            for key, value
            in status_counts.items()
        },
        "integrity_contract": {
            "expected_metadata_source": (
                "GDC download manifests"
            ),
            "checks": [
                "filename",
                "file size",
                "MD5 digest",
            ],
            "repair_endpoint": (
                "https://api.gdc.cancer.gov/data/{file_id}"
            ),
            "atomic_replacement": True,
        },
        "guardrails": [
            (
                "No clinical outcome is loaded."
            ),
            (
                "No molecular feature is selected."
            ),
            (
                "Only files failing GDC manifest "
                "size or MD5 integrity are eligible "
                "for repair."
            ),
            (
                "Downloaded repairs must match both "
                "the GDC manifest size and MD5 before "
                "being accepted."
            ),
        ],
        "output": (
            relative_path(
                OUTPUT_AUDIT
            )
        ),
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
        "TARGET-OS methylation download integrity audit",
        "=" * 68,
        "",
        f"Script version: {SCRIPT_VERSION}",
        "",
        f"Expected processed files: {audit.shape[0]}",
        f"Integrity PASS: {n_pass}",
        f"Integrity FAIL: {n_failed}",
        "",
        "Status counts",
        "-------------",
    ]

    for status, count in sorted(
        status_counts.items()
    ):
        readme_lines.append(
            f"{status}: {count}"
        )

    readme_lines.extend(
        [
            "",
            (
                "Files are validated against the "
                "original GDC manifest size and MD5."
            ),
            (
                "No outcome data or feature selection "
                "is involved."
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit TARGET-OS SeSAMe beta-value "
            "downloads against GDC manifest size "
            "and MD5; optionally repair only failed files."
        )
    )

    parser.add_argument(
        "--repair",
        action="store_true",
        help=(
            "Redownload only files that fail "
            "the GDC size/MD5 integrity audit."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 88)
    print(
        "Paper 5 - TARGET methylation "
        "download integrity audit"
    )
    print("=" * 88)

    print(
        f"Script version: "
        f"{SCRIPT_VERSION}"
    )

    print(
        f"Repair mode: "
        f"{bool(args.repair)}"
    )

    records = (
        load_manifest_records()
    )

    print(
        f"Unique processed methylation "
        f"records in GDC manifests: "
        f"{records.shape[0]}"
    )

    audit_before = (
        inspect_all_files(
            records
        )
    )

    failed_before = (
        ~audit_before[
            "status"
        ].isin(
            [
                "PASS",
                "PASS_DUPLICATE_LOCAL_COPY",
            ]
        )
    )

    print()
    print(
        "Initial integrity failures: "
        f"{int(failed_before.sum())}"
    )

    if failed_before.any():
        print()

        columns = [
            "case_id",
            "filename",
            "expected_size",
            "observed_size",
            "size_difference",
            "status",
        ]

        print(
            audit_before.loc[
                failed_before,
                columns,
            ].to_string(
                index=False
            )
        )

    if (
        args.repair
        and failed_before.any()
    ):
        repair_failed_files(
            audit_before
        )

        print()
        print(
            "Re-running full integrity audit "
            "after repair..."
        )

        audit_final = (
            inspect_all_files(
                records
            )
        )

    else:
        audit_final = (
            audit_before
        )

    write_outputs(
        audit_final,
        repair_requested=bool(
            args.repair
        ),
    )

    failed_final = (
        ~audit_final[
            "status"
        ].isin(
            [
                "PASS",
                "PASS_DUPLICATE_LOCAL_COPY",
            ]
        )
    )

    print()
    print("-" * 88)
    print(
        "Final integrity summary"
    )
    print("-" * 88)

    print(
        f"PASS: "
        f"{int((~failed_final).sum())}/"
        f"{audit_final.shape[0]}"
    )

    print(
        f"FAIL: "
        f"{int(failed_final.sum())}/"
        f"{audit_final.shape[0]}"
    )

    if failed_final.any():
        print()
        print(
            "Integrity audit: FAIL"
        )

        print(
            "Do not rebuild the methylation "
            "matrix yet."
        )

    else:
        print()
        print(
            "Integrity audit: PASS"
        )

        print(
            "All processed methylation files "
            "match their GDC manifest."
        )

    print()
    print(
        f"Audit table: "
        f"{relative_path(OUTPUT_AUDIT)}"
    )

    print(
        f"Manifest: "
        f"{relative_path(OUTPUT_MANIFEST)}"
    )

    print("=" * 88)


if __name__ == "__main__":
    main()
