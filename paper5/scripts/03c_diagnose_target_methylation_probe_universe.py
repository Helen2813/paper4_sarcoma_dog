from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "03c-diagnose-target-methylation-probe-universe-v1"

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

OUTPUT_FILE_AUDIT = (
    MODEL_INPUT_DIR
    / "TARGET_OS_methylation_probe_universe_diagnostic.csv"
)

OUTPUT_FINGERPRINTS = (
    MODEL_INPUT_DIR
    / "TARGET_OS_methylation_probe_set_fingerprints.csv"
)

OUTPUT_RUNNING_INTERSECTION = (
    MODEL_INPUT_DIR
    / "TARGET_OS_methylation_running_intersection.csv"
)

OUTPUT_MANIFEST = (
    MODEL_INPUT_DIR
    / "TARGET_OS_methylation_probe_universe_diagnostic_manifest.json"
)

OUTPUT_README = (
    MODEL_INPUT_DIR
    / "TARGET_OS_methylation_probe_universe_diagnostic_README.txt"
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


def probe_fingerprint(
    probes: set[str],
) -> str:
    digest = hashlib.sha256()

    for probe in sorted(probes):
        digest.update(
            probe.encode(
                "utf-8"
            )
        )
        digest.update(b"\n")

    return digest.hexdigest()


def ordered_probe_fingerprint(
    probes: list[str],
) -> str:
    digest = hashlib.sha256()

    for probe in probes:
        digest.update(
            probe.encode(
                "utf-8"
            )
        )
        digest.update(b"\n")

    return digest.hexdigest()


def read_probe_file(
    path: Path,
) -> dict[str, Any]:
    """
    GDC SeSAMe beta files are treated as headerless:
    column 1 = probe ID
    column 2 = beta value.
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
            f"{path.name}: expected 2 columns, "
            f"found {table.shape[1]}"
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

    valid_probe = (
        probes.ne("")
        & probes.ne("nan")
        & probes.ne("None")
    )

    probes = probes[
        valid_probe
    ].tolist()

    beta = beta[
        valid_probe
    ]

    probe_set = set(
        probes
    )

    prefixes = Counter(
        probe[:2].lower()
        if len(probe) >= 2
        else probe.lower()
        for probe in probes
    )

    cg_count = sum(
        probe.lower().startswith(
            "cg"
        )
        for probe in probes
    )

    ch_count = sum(
        probe.lower().startswith(
            "ch"
        )
        for probe in probes
    )

    rs_count = sum(
        probe.lower().startswith(
            "rs"
        )
        for probe in probes
    )

    return {
        "ordered_probes": probes,
        "probe_set": probe_set,
        "n_rows": int(
            len(probes)
        ),
        "n_unique_probes": int(
            len(probe_set)
        ),
        "n_duplicate_probe_rows": int(
            len(probes)
            - len(probe_set)
        ),
        "n_finite_beta": int(
            beta.notna().sum()
        ),
        "missing_beta_fraction": float(
            beta.isna().mean()
        ),
        "n_cg": int(
            cg_count
        ),
        "n_ch": int(
            ch_count
        ),
        "n_rs": int(
            rs_count
        ),
        "first_probe": (
            probes[0]
            if probes
            else ""
        ),
        "last_probe": (
            probes[-1]
            if probes
            else ""
        ),
        "set_fingerprint": (
            probe_fingerprint(
                probe_set
            )
        ),
        "ordered_fingerprint": (
            ordered_probe_fingerprint(
                probes
            )
        ),
        "prefix_summary": ";".join(
            f"{prefix}:{count}"
            for prefix, count
            in prefixes.most_common(10)
        ),
    }


def load_methylation_file_map() -> pd.DataFrame:
    if not FILE_MAP.exists():
        raise FileNotFoundError(
            f"Missing file map: "
            f"{FILE_MAP}"
        )

    if not CASE_ORDER.exists():
        raise FileNotFoundError(
            f"Missing case order: "
            f"{CASE_ORDER}"
        )

    cases = pd.read_csv(
        CASE_ORDER,
        dtype=str,
    )

    case_ids = (
        cases["case_id"]
        .astype(str)
        .tolist()
    )

    file_map = pd.read_csv(
        FILE_MAP,
        dtype=str,
    ).fillna("")

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

    if methylation.shape[0] != len(
        case_ids
    ):
        raise RuntimeError(
            "Methylation file map "
            "does not match case order."
        )

    return methylation


def main() -> None:
    print("=" * 88)
    print(
        "Paper 5 - TARGET methylation "
        "probe-universe diagnostic"
    )
    print("=" * 88)

    print(
        f"Script version: "
        f"{SCRIPT_VERSION}"
    )

    file_map = (
        load_methylation_file_map()
    )

    print(
        f"Methylation files: "
        f"{file_map.shape[0]}"
    )

    audit_rows = []
    running_rows = []

    reference_set: set[str] | None = None
    running_intersection: set[str] | None = None
    previous_intersection_size: int | None = None

    fingerprints: dict[
        str,
        list[str],
    ] = {}

    for index, row in (
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

        if not path.exists():
            raise FileNotFoundError(
                f"Missing: {path}"
            )

        info = read_probe_file(
            path
        )

        probe_set = info[
            "probe_set"
        ]

        if reference_set is None:
            reference_set = set(
                probe_set
            )

        overlap_with_reference = (
            len(
                reference_set
                & probe_set
            )
        )

        union_with_reference = (
            len(
                reference_set
                | probe_set
            )
        )

        jaccard_reference = (
            overlap_with_reference
            / union_with_reference
            if union_with_reference
            else np.nan
        )

        if running_intersection is None:
            running_intersection = set(
                probe_set
            )
        else:
            running_intersection &= (
                probe_set
            )

        current_intersection_size = (
            len(
                running_intersection
            )
        )

        if previous_intersection_size is None:
            probes_lost_this_step = 0
        else:
            probes_lost_this_step = (
                previous_intersection_size
                - current_intersection_size
            )

        previous_intersection_size = (
            current_intersection_size
        )

        fingerprints.setdefault(
            info[
                "set_fingerprint"
            ],
            [],
        ).append(
            case_id
        )

        audit_rows.append(
            {
                "order": int(
                    index
                ),
                "case_id": (
                    case_id
                ),
                "file_name": (
                    path.name
                ),
                "file_size_bytes": int(
                    path.stat().st_size
                ),
                "n_rows": (
                    info[
                        "n_rows"
                    ]
                ),
                "n_unique_probes": (
                    info[
                        "n_unique_probes"
                    ]
                ),
                "n_duplicate_probe_rows": (
                    info[
                        "n_duplicate_probe_rows"
                    ]
                ),
                "n_finite_beta": (
                    info[
                        "n_finite_beta"
                    ]
                ),
                "missing_beta_fraction": (
                    info[
                        "missing_beta_fraction"
                    ]
                ),
                "n_cg": (
                    info[
                        "n_cg"
                    ]
                ),
                "n_ch": (
                    info[
                        "n_ch"
                    ]
                ),
                "n_rs": (
                    info[
                        "n_rs"
                    ]
                ),
                "first_probe": (
                    info[
                        "first_probe"
                    ]
                ),
                "last_probe": (
                    info[
                        "last_probe"
                    ]
                ),
                "set_fingerprint": (
                    info[
                        "set_fingerprint"
                    ]
                ),
                "ordered_fingerprint": (
                    info[
                        "ordered_fingerprint"
                    ]
                ),
                "n_overlap_with_first_file": int(
                    overlap_with_reference
                ),
                "jaccard_with_first_file": float(
                    jaccard_reference
                ),
                "prefix_summary": (
                    info[
                        "prefix_summary"
                    ]
                ),
            }
        )

        running_rows.append(
            {
                "order": int(
                    index
                ),
                "case_id": (
                    case_id
                ),
                "n_probes_current_file": int(
                    len(
                        probe_set
                    )
                ),
                "running_common_probes": int(
                    current_intersection_size
                ),
                "probes_lost_at_this_step": int(
                    probes_lost_this_step
                ),
            }
        )

        print(
            f"{index + 1:02d}/"
            f"{file_map.shape[0]} "
            f"{case_id}: "
            f"probes="
            f"{len(probe_set):,}, "
            f"overlap_first="
            f"{overlap_with_reference:,}, "
            f"J="
            f"{jaccard_reference:.4f}, "
            f"running_common="
            f"{current_intersection_size:,}, "
            f"lost="
            f"{probes_lost_this_step:,}"
        )

    audit = pd.DataFrame(
        audit_rows
    )

    running = pd.DataFrame(
        running_rows
    )

    audit.to_csv(
        OUTPUT_FILE_AUDIT,
        index=False,
    )

    running.to_csv(
        OUTPUT_RUNNING_INTERSECTION,
        index=False,
    )

    fingerprint_rows = []

    for fingerprint, cases in (
        fingerprints.items()
    ):
        subset = audit[
            audit[
                "set_fingerprint"
            ].eq(
                fingerprint
            )
        ]

        fingerprint_rows.append(
            {
                "set_fingerprint": (
                    fingerprint
                ),
                "n_cases": len(
                    cases
                ),
                "case_ids": ";".join(
                    cases
                ),
                "n_unique_probes": int(
                    subset[
                        "n_unique_probes"
                    ].iloc[0]
                ),
                "first_probe": (
                    subset[
                        "first_probe"
                    ].iloc[0]
                ),
                "last_probe": (
                    subset[
                        "last_probe"
                    ].iloc[0]
                ),
            }
        )

    fingerprint_table = (
        pd.DataFrame(
            fingerprint_rows
        )
        .sort_values(
            [
                "n_cases",
                "n_unique_probes",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    fingerprint_table.to_csv(
        OUTPUT_FINGERPRINTS,
        index=False,
    )

    print()
    print("-" * 88)
    print(
        "Probe universe diagnostic summary"
    )
    print("-" * 88)

    print(
        "Unique probe-set fingerprints: "
        f"{fingerprint_table.shape[0]}"
    )

    print()
    print(
        fingerprint_table[
            [
                "n_cases",
                "n_unique_probes",
                "first_probe",
                "last_probe",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Per-file probe-count range: "
        f"{audit['n_unique_probes'].min():,}"
        " - "
        f"{audit['n_unique_probes'].max():,}"
    )

    print(
        "Overlap with first-file range: "
        f"{audit['n_overlap_with_first_file'].min():,}"
        " - "
        f"{audit['n_overlap_with_first_file'].max():,}"
    )

    print(
        "Jaccard with first-file range: "
        f"{audit['jaccard_with_first_file'].min():.6f}"
        " - "
        f"{audit['jaccard_with_first_file'].max():.6f}"
    )

    final_common = int(
        running[
            "running_common_probes"
        ].iloc[-1]
    )

    print(
        "Final 78-case intersection: "
        f"{final_common:,}"
    )

    biggest_loss = (
        running.sort_values(
            "probes_lost_at_this_step",
            ascending=False,
        )
        .iloc[0]
    )

    print()
    print(
        "Largest intersection collapse:"
    )

    print(
        f"  case: "
        f"{biggest_loss['case_id']}"
    )

    print(
        f"  probes lost at that step: "
        f"{int(biggest_loss['probes_lost_at_this_step']):,}"
    )

    print(
        f"  remaining common probes: "
        f"{int(biggest_loss['running_common_probes']):,}"
    )

    interpretation = ""

    if (
        fingerprint_table.shape[0]
        == 1
        and final_common
        == int(
            audit[
                "n_unique_probes"
            ].iloc[0]
        )
    ):
        interpretation = (
            "All 78 files have the same probe universe."
        )

    elif (
        audit[
            "n_unique_probes"
        ].min()
        < 100_000
    ):
        interpretation = (
            "At least one file has a substantially "
            "smaller probe universe, consistent with "
            "mixed assay generations or an anomalous file."
        )

    elif (
        audit[
            "jaccard_with_first_file"
        ].min()
        < 0.5
    ):
        interpretation = (
            "Files have similarly large row counts but "
            "strongly different probe identifiers. "
            "Platform/manifest provenance must be audited "
            "before constructing a joint methylation matrix."
        )

    else:
        interpretation = (
            "Probe universes differ moderately across files; "
            "use missing-aware union construction rather than "
            "requiring complete-case probe intersection."
        )

    print()
    print("Interpretation:")
    print(
        f"  {interpretation}"
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
        "n_cases": int(
            audit.shape[0]
        ),
        "n_probe_set_fingerprints": int(
            fingerprint_table.shape[0]
        ),
        "probe_count_min": int(
            audit[
                "n_unique_probes"
            ].min()
        ),
        "probe_count_max": int(
            audit[
                "n_unique_probes"
            ].max()
        ),
        "minimum_jaccard_with_first": float(
            audit[
                "jaccard_with_first_file"
            ].min()
        ),
        "final_common_probe_count": (
            final_common
        ),
        "largest_intersection_loss_case": (
            clean_text(
                biggest_loss[
                    "case_id"
                ]
            )
        ),
        "largest_intersection_loss": int(
            biggest_loss[
                "probes_lost_at_this_step"
            ]
        ),
        "interpretation": (
            interpretation
        ),
        "guardrails": [
            (
                "Only probe identifiers and beta-value "
                "missingness are examined."
            ),
            (
                "No clinical outcome is loaded."
            ),
            (
                "No probe is selected for modeling."
            ),
            (
                "No imputation, scaling, or representation "
                "learning is performed."
            ),
        ],
        "outputs": {
            "file_audit": (
                relative_path(
                    OUTPUT_FILE_AUDIT
                )
            ),
            "fingerprints": (
                relative_path(
                    OUTPUT_FINGERPRINTS
                )
            ),
            "running_intersection": (
                relative_path(
                    OUTPUT_RUNNING_INTERSECTION
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

    readme = [
        "TARGET-OS methylation probe-universe diagnostic",
        "=" * 68,
        "",
        f"Script version: {SCRIPT_VERSION}",
        "",
        f"Cases: {audit.shape[0]}",
        (
            "Unique probe-set fingerprints: "
            f"{fingerprint_table.shape[0]}"
        ),
        (
            "Probe-count range: "
            f"{audit['n_unique_probes'].min()} - "
            f"{audit['n_unique_probes'].max()}"
        ),
        (
            "Final complete intersection: "
            f"{final_common}"
        ),
        "",
        "Interpretation",
        "--------------",
        interpretation,
        "",
        "No clinical outcomes were loaded.",
        "No feature selection was performed.",
        "",
    ]

    OUTPUT_README.write_text(
        "\n".join(readme),
        encoding="utf-8",
    )

    print()
    print("Outputs:")

    for path in [
        OUTPUT_FILE_AUDIT,
        OUTPUT_FINGERPRINTS,
        OUTPUT_RUNNING_INTERSECTION,
        OUTPUT_MANIFEST,
        OUTPUT_README,
    ]:
        print(
            f"  {relative_path(path)}"
        )

    print("=" * 88)


if __name__ == "__main__":
    main()
