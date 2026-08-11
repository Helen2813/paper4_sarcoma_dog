from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "03b-rebuild-target-methylation-matrix-v1"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

MODEL_INPUT_DIR = (
    PAPER5_ROOT
    / "model_inputs"
    / "TARGET_OS_primary78"
)

FILE_MAP = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_multimodal_file_map.csv"
)

CASE_ORDER = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_multimodal_case_order.csv"
)

OLD_MATRIX = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_methylation_beta.npy"
)

OUTPUT_MATRIX = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_methylation_beta_corrected.npy"
)

OUTPUT_FEATURES = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_methylation_features_corrected.csv"
)

OUTPUT_SAMPLE_QC = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_methylation_sample_qc_corrected.csv"
)

OUTPUT_FEATURE_QC = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_methylation_feature_qc_corrected.csv"
)

OUTPUT_FILE_AUDIT = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_methylation_file_probe_audit.csv"
)

OUTPUT_MANIFEST = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_methylation_correction_manifest.json"
)

OUTPUT_README = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_methylation_correction_README.txt"
)

EXPECTED_CASES = 78

# TARGET files downloaded here contain approximately 486k probe rows.
# This threshold is an assay-integrity guardrail, not feature selection.
MIN_EXPECTED_COMMON_PROBES = 400_000

FEATURE_QC_CHUNK = 50_000


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


def resolve_path(value: Any) -> Path:
    path = Path(
        clean_text(value)
    )

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_beta_file(
    path: Path,
) -> pd.Series:
    """
    Read a GDC SeSAMe beta-value file.

    Important:
    These TARGET-OS files do not contain a header row.
    Column 1 is the CpG probe identifier and column 2 is
    the beta value.
    """

    table = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=[
            "probe_id",
            "beta_value",
        ],
        dtype={
            "probe_id": str,
        },
        low_memory=False,
    )

    if table.shape[1] != 2:
        raise RuntimeError(
            f"Expected exactly two columns in {path.name}; "
            f"found {table.shape[1]}."
        )

    probes = (
        table["probe_id"]
        .astype(str)
        .str.strip()
    )

    beta = pd.to_numeric(
        table["beta_value"],
        errors="coerce",
    )

    # Values outside the beta-value domain are invalid.
    beta = beta.where(
        beta.between(
            0.0,
            1.0,
            inclusive="both",
        )
    )

    valid_probe = (
        probes.ne("")
        & probes.ne("nan")
    )

    data = pd.DataFrame(
        {
            "probe_id": probes[
                valid_probe
            ],
            "beta_value": beta[
                valid_probe
            ],
        }
    )

    if data[
        "probe_id"
    ].duplicated().any():
        data = (
            data.groupby(
                "probe_id",
                sort=False,
            )["beta_value"]
            .mean()
        )
    else:
        data = data.set_index(
            "probe_id"
        )["beta_value"]

    return data.astype(
        np.float32
    )


def load_inputs() -> tuple[
    list[str],
    pd.DataFrame,
]:
    if not CASE_ORDER.exists():
        raise FileNotFoundError(
            f"Missing case-order file: {CASE_ORDER}"
        )

    if not FILE_MAP.exists():
        raise FileNotFoundError(
            f"Missing file map: {FILE_MAP}"
        )

    cases = pd.read_csv(
        CASE_ORDER,
        dtype=str,
    )

    file_map = pd.read_csv(
        FILE_MAP,
        dtype=str,
    ).fillna("")

    if "case_id" not in cases.columns:
        raise ValueError(
            "Case-order file does not contain case_id."
        )

    case_ids = (
        cases["case_id"]
        .astype(str)
        .tolist()
    )

    if len(case_ids) != EXPECTED_CASES:
        raise RuntimeError(
            f"Expected {EXPECTED_CASES} cases; "
            f"found {len(case_ids)}."
        )

    methylation = file_map[
        file_map["modality"]
        .eq("methylation")
    ].copy()

    methylation = (
        methylation.set_index(
            "case_id"
        )
        .loc[case_ids]
        .reset_index()
    )

    if methylation.shape[0] != EXPECTED_CASES:
        raise RuntimeError(
            "Methylation file map is incomplete."
        )

    return case_ids, methylation


def audit_probe_universe(
    case_ids: list[str],
    file_map: pd.DataFrame,
) -> tuple[
    list[str],
    pd.DataFrame,
]:
    common: set[str] | None = None
    first_order: list[str] = []

    audit_rows = []

    print(
        "First pass: auditing methylation probe universes..."
    )

    for index, row in file_map.iterrows():
        case_id = row["case_id"]

        path = resolve_path(
            row["local_path"]
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing methylation file: {path}"
            )

        series = read_beta_file(
            path
        )

        probes = (
            series.index
            .astype(str)
            .tolist()
        )

        probe_set = set(
            probes
        )

        if common is None:
            common = probe_set
            first_order = probes
        else:
            common &= probe_set

        audit_rows.append(
            {
                "case_id": case_id,
                "file_name": path.name,
                "n_probe_rows": int(
                    len(series)
                ),
                "n_unique_probes": int(
                    len(probe_set)
                ),
                "n_finite_beta": int(
                    series.notna().sum()
                ),
                "missing_fraction": float(
                    series.isna().mean()
                ),
                "min_beta": float(
                    series.min(
                        skipna=True
                    )
                ),
                "max_beta": float(
                    series.max(
                        skipna=True
                    )
                ),
            }
        )

        if (
            (index + 1) % 10 == 0
            or index + 1
            == file_map.shape[0]
        ):
            print(
                f"  inspected "
                f"{index + 1}/"
                f"{file_map.shape[0]}"
            )

    if common is None:
        raise RuntimeError(
            "No methylation probes found."
        )

    common_ordered = [
        probe
        for probe in first_order
        if probe in common
    ]

    print()
    print(
        "Probe-universe audit:"
    )

    print(
        f"  first sample probes: "
        f"{len(first_order):,}"
    )

    print(
        f"  common probes across "
        f"{len(case_ids)} cases: "
        f"{len(common_ordered):,}"
    )

    if (
        len(common_ordered)
        < MIN_EXPECTED_COMMON_PROBES
    ):
        raise RuntimeError(
            "Unexpectedly small common methylation "
            f"probe universe: {len(common_ordered):,}. "
            f"Expected at least "
            f"{MIN_EXPECTED_COMMON_PROBES:,}. "
            "Stopping before matrix construction."
        )

    audit = pd.DataFrame(
        audit_rows
    )

    return common_ordered, audit


def build_matrix(
    case_ids: list[str],
    file_map: pd.DataFrame,
    probes: list[str],
) -> pd.DataFrame:
    probe_index = pd.Index(
        probes,
        dtype=str,
    )

    matrix = (
        np.lib.format.open_memmap(
            OUTPUT_MATRIX,
            mode="w+",
            dtype=np.float32,
            shape=(
                len(case_ids),
                len(probes),
            ),
        )
    )

    sample_rows = []

    print()
    print(
        "Second pass: writing corrected "
        "methylation matrix..."
    )

    for row_index, row in (
        file_map.iterrows()
    ):
        case_id = row[
            "case_id"
        ]

        path = resolve_path(
            row[
                "local_path"
            ]
        )

        series = read_beta_file(
            path
        )

        aligned = (
            series.reindex(
                probe_index
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
                "case_id": case_id,
                "n_probes": len(
                    probes
                ),
                "n_finite": int(
                    finite.sum()
                ),
                "n_missing": int(
                    (~finite).sum()
                ),
                "missing_fraction": float(
                    1.0
                    - finite.mean()
                ),
                "mean_beta": float(
                    np.nanmean(
                        aligned
                    )
                ),
                "sd_beta": float(
                    np.nanstd(
                        aligned,
                        ddof=1,
                    )
                ),
                "file_name": path.name,
            }
        )

        if (
            (row_index + 1) % 10
            == 0
            or row_index + 1
            == file_map.shape[0]
        ):
            print(
                f"  wrote "
                f"{row_index + 1}/"
                f"{file_map.shape[0]}"
            )

    matrix.flush()
    del matrix

    return pd.DataFrame(
        sample_rows
    )


def build_feature_qc(
    probes: list[str],
) -> pd.DataFrame:
    matrix = np.load(
        OUTPUT_MATRIX,
        mmap_mode="r",
    )

    n_samples, n_features = (
        matrix.shape
    )

    parts = []

    print()
    print(
        "Computing methylation feature QC..."
    )

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

        with np.errstate(
            invalid="ignore"
        ):
            mean = np.nanmean(
                block,
                axis=0,
            )

            variance = np.nanvar(
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

        parts.append(
            pd.DataFrame(
                {
                    "probe_id": (
                        probes[
                            start:end
                        ]
                    ),
                    "n_finite": (
                        n_finite
                    ),
                    "missing_fraction": (
                        1.0
                        - (
                            n_finite
                            / n_samples
                        )
                    ),
                    "mean_beta": mean,
                    "variance": variance,
                    "min_beta": minimum,
                    "max_beta": maximum,
                }
            )
        )

        print(
            f"  features "
            f"{start:,}-"
            f"{end:,}/"
            f"{n_features:,}"
        )

    qc = pd.concat(
        parts,
        ignore_index=True,
    )

    qc[
        "zero_or_undefined_variance"
    ] = (
        ~np.isfinite(
            qc[
                "variance"
            ]
        )
        | qc[
            "variance"
        ].le(0)
    )

    return qc


def main() -> None:
    print("=" * 88)
    print(
        "Paper 5 - corrected TARGET-OS "
        "methylation matrix"
    )
    print("=" * 88)

    print(
        f"Script version: "
        f"{SCRIPT_VERSION}"
    )

    print()

    case_ids, file_map = (
        load_inputs()
    )

    print(
        f"Primary TARGET cases: "
        f"{len(case_ids)}"
    )

    print(
        f"Methylation files: "
        f"{file_map.shape[0]}"
    )

    probes, file_audit = (
        audit_probe_universe(
            case_ids,
            file_map,
        )
    )

    file_audit.to_csv(
        OUTPUT_FILE_AUDIT,
        index=False,
    )

    pd.DataFrame(
        {
            "feature_index": np.arange(
                len(probes),
                dtype=int,
            ),
            "probe_id": probes,
        }
    ).to_csv(
        OUTPUT_FEATURES,
        index=False,
    )

    sample_qc = build_matrix(
        case_ids,
        file_map,
        probes,
    )

    sample_qc.to_csv(
        OUTPUT_SAMPLE_QC,
        index=False,
    )

    feature_qc = build_feature_qc(
        probes
    )

    feature_qc.to_csv(
        OUTPUT_FEATURE_QC,
        index=False,
    )

    matrix = np.load(
        OUTPUT_MATRIX,
        mmap_mode="r",
    )

    print()
    print("-" * 88)
    print(
        "Corrected methylation summary"
    )
    print("-" * 88)

    print(
        f"Matrix shape: "
        f"{matrix.shape[0]} x "
        f"{matrix.shape[1]:,}"
    )

    print(
        "Mean sample missing fraction: "
        f"{sample_qc['missing_fraction'].mean():.6f}"
    )

    print(
        "Maximum sample missing fraction: "
        f"{sample_qc['missing_fraction'].max():.6f}"
    )

    print(
        "Features with any missing value: "
        f"{feature_qc['missing_fraction'].gt(0).sum():,}"
    )

    print(
        "Features with zero/undefined variance: "
        f"{feature_qc['zero_or_undefined_variance'].sum():,}"
    )

    old_shape = None

    if OLD_MATRIX.exists():
        try:
            old_shape = list(
                np.load(
                    OLD_MATRIX,
                    mmap_mode="r",
                ).shape
            )
        except Exception:
            old_shape = None

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "outcome_loaded": False,
        "outcome_guided_feature_selection": False,
        "model_fitting_performed": False,
        "correction_reason": (
            "GDC SeSAMe TARGET-OS methylation beta "
            "TXT files contain no header row. The first "
            "probe/value pair had previously been interpreted "
            "as column names."
        ),
        "primary_cases": len(
            case_ids
        ),
        "common_probe_count": len(
            probes
        ),
        "old_matrix_shape": (
            old_shape
        ),
        "corrected_matrix_shape": list(
            matrix.shape
        ),
        "mean_sample_missing_fraction": float(
            sample_qc[
                "missing_fraction"
            ].mean()
        ),
        "max_sample_missing_fraction": float(
            sample_qc[
                "missing_fraction"
            ].max()
        ),
        "n_features_with_any_missing": int(
            feature_qc[
                "missing_fraction"
            ].gt(0).sum()
        ),
        "n_zero_or_undefined_variance_features": int(
            feature_qc[
                "zero_or_undefined_variance"
            ].sum()
        ),
        "processing_contract": {
            "input_format": (
                "headerless two-column TXT"
            ),
            "column_1": (
                "CpG probe identifier"
            ),
            "column_2": (
                "SeSAMe beta value"
            ),
            "beta_range": [
                0.0,
                1.0,
            ],
            "missing_values": (
                "retained"
            ),
            "feature_universe": (
                "probe identifiers present "
                "in all 78 files"
            ),
            "imputation": (
                "not performed"
            ),
            "variance_filtering": (
                "not performed"
            ),
        },
        "guardrails": [
            (
                "No clinical outcome is loaded."
            ),
            (
                "Missing beta values remain missing."
            ),
            (
                "No CpG is selected using outcome data."
            ),
            (
                "No variance filtering is performed."
            ),
            (
                "A minimum common probe count of "
                f"{MIN_EXPECTED_COMMON_PROBES} is enforced "
                "as an assay-integrity diagnostic."
            ),
        ],
        "outputs": {
            "matrix": (
                relative_path(
                    OUTPUT_MATRIX
                )
            ),
            "features": (
                relative_path(
                    OUTPUT_FEATURES
                )
            ),
            "sample_qc": (
                relative_path(
                    OUTPUT_SAMPLE_QC
                )
            ),
            "feature_qc": (
                relative_path(
                    OUTPUT_FEATURE_QC
                )
            ),
            "file_probe_audit": (
                relative_path(
                    OUTPUT_FILE_AUDIT
                )
            ),
        },
        "matrix_sha256": (
            sha256_file(
                OUTPUT_MATRIX
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

    readme = [
        "Corrected TARGET-OS methylation matrix",
        "=" * 64,
        "",
        f"Script version: {SCRIPT_VERSION}",
        "",
        "Correction",
        "----------",
        (
            "The downloaded GDC SeSAMe beta-value files "
            "are headerless two-column files."
        ),
        (
            "The previous parser treated the first "
            "CpG/beta pair as a header."
        ),
        "",
        "Corrected matrix",
        "----------------",
        (
            f"Cases: {len(case_ids)}"
        ),
        (
            f"Common CpG probes: "
            f"{len(probes)}"
        ),
        (
            "No clinical outcomes were loaded."
        ),
        (
            "No imputation or feature selection "
            "was performed."
        ),
        "",
    ]

    OUTPUT_README.write_text(
        "\n".join(
            readme
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print(
        "Corrected TARGET methylation "
        "construction: PASS"
    )
    print("=" * 88)

    print(
        f"Corrected matrix: "
        f"{relative_path(OUTPUT_MATRIX)}"
    )

    print(
        f"Manifest: "
        f"{relative_path(OUTPUT_MANIFEST)}"
    )

    print("=" * 88)


if __name__ == "__main__":
    main()
