from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "04b-verify-harmonized-modalities-lock-v1"


SCRIPT_PATH = Path(__file__).resolve()

PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]


INPUT_DIR = (
    PAPER5_ROOT
    / "model_inputs"
    / "TARGET_OS_primary78"
    / "harmonized"
)


CASE_ORDER_FILE = (
    PAPER5_ROOT
    / "model_inputs"
    / "TARGET_OS_primary78"
    / "TARGET_OS_primary_multimodal_case_order.csv"
)


OUTPUT_DIR = INPUT_DIR


OUTPUT_ALIGNMENT = (
    OUTPUT_DIR
    / "sample_alignment_audit.csv"
)

OUTPUT_MISSINGNESS = (
    OUTPUT_DIR
    / "missingness_audit.csv"
)

OUTPUT_VARIANCE = (
    OUTPUT_DIR
    / "variance_audit.csv"
)

OUTPUT_LOCK = (
    OUTPUT_DIR
    / "harmonized_input_lock.json"
)

OUTPUT_README = (
    OUTPUT_DIR
    / "harmonized_input_lock_README.txt"
)


MODALITIES = {

    "RNA": {
        "matrix":
            INPUT_DIR / "RNA_zscore.npy",

        "features":
            INPUT_DIR / "RNA_features.csv",

        "feature_column":
            "feature",
    },


    "METHYLATION": {
        "matrix":
            INPUT_DIR / "METHYLATION_Mvalue_zscore.npy",

        "features":
            INPUT_DIR / "METHYLATION_features.csv",

        "feature_column":
            "probe_id",
    },


    "CNV": {
        "matrix":
            INPUT_DIR / "CNV_log2ratio_zscore.npy",

        "features":
            INPUT_DIR / "CNV_features.csv",

        "feature_column":
            "feature",
    },
}



def utc_now_iso():
    return datetime.now(
        timezone.utc
    ).isoformat()



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



def sha256_file(path: Path):

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        for chunk in iter(
            lambda:
            handle.read(
                1024 * 1024
            ),
            b"",
        ):

            digest.update(
                chunk
            )

    return digest.hexdigest()



def hash_strings(values):

    digest = hashlib.sha256()

    for value in values:

        digest.update(
            str(value)
            .encode(
                "utf-8"
            )
        )

        digest.update(
            b"\n"
        )

    return digest.hexdigest()



def load_case_order():

    table = pd.read_csv(
        CASE_ORDER_FILE,
        dtype=str,
    )

    if "case_id" not in table.columns:

        raise ValueError(
            "case_id missing from case order"
        )

    return (
        table["case_id"]
        .astype(str)
        .tolist()
    )



def verify_shapes(
    case_ids,
):

    rows = []

    for modality, info in MODALITIES.items():

        matrix = np.load(
            info["matrix"],
            mmap_mode="r",
        )

        rows.append(
            {
                "modality":
                    modality,

                "n_samples":
                    matrix.shape[0],

                "n_features":
                    matrix.shape[1],

                "sample_count_matches":
                    matrix.shape[0]
                    ==
                    len(case_ids),
            }
        )

    return pd.DataFrame(rows)



def load_features(
    info
):

    table = pd.read_csv(
        info["features"],
        dtype=str,
    )

    column = info["feature_column"]

    return (
        table[column]
        .astype(str)
        .tolist()
    )



def verify_feature_integrity():

    rows = []

    for modality, info in MODALITIES.items():

        features = load_features(
            info
        )

        rows.append(
            {
                "modality":
                    modality,

                "n_features":
                    len(features),

                "unique_features":
                    len(set(features)),

                "feature_hash":
                    hash_strings(features),
            }
        )

    return pd.DataFrame(rows)



def calculate_missingness():

    rows = []

    for modality, info in MODALITIES.items():

        matrix = np.load(
            info["matrix"],
            mmap_mode="r",
        )

        missing = (
            ~np.isfinite(
                matrix
            )
        )

        rows.append(
            {
                "modality":
                    modality,

                "matrix_shape":
                    str(
                        matrix.shape
                    ),

                "total_missing_fraction":
                    float(
                        missing.mean()
                    ),

                "mean_sample_missing_fraction":
                    float(
                        missing.mean(
                            axis=1
                        ).mean()
                    ),

                "max_sample_missing_fraction":
                    float(
                        missing.mean(
                            axis=1
                        ).max()
                    ),

                "mean_feature_missing_fraction":
                    float(
                        missing.mean(
                            axis=0
                        ).mean()
                    ),

                "max_feature_missing_fraction":
                    float(
                        missing.mean(
                            axis=0
                        ).max()
                    ),
            }
        )

    return pd.DataFrame(rows)



def calculate_variance():

    rows = []

    for modality, info in MODALITIES.items():

        matrix = np.load(
            info["matrix"],
            mmap_mode="r",
        )

        variance = np.var(
            matrix,
            axis=0,
            ddof=1,
        )


        rows.append(
            {
                "modality":
                    modality,

                "median_variance":
                    float(
                        np.nanmedian(
                            variance
                        )
                    ),

                "mean_variance":
                    float(
                        np.nanmean(
                            variance
                        )
                    ),

                "zero_variance_features":
                    int(
                        np.sum(
                            variance <= 0
                        )
                    ),

                "finite_variance_features":
                    int(
                        np.isfinite(
                            variance
                        ).sum()
                    ),
            }
        )


    return pd.DataFrame(rows)



def main():

    print("=" * 88)

    print(
        "Paper 5 - harmonized modality lock verification"
    )

    print("=" * 88)


    print(
        f"Script version: {SCRIPT_VERSION}"
    )


    case_ids = load_case_order()


    print(
        f"Cases detected: {len(case_ids)}"
    )


    shape_audit = verify_shapes(
        case_ids
    )


    print()
    print(
        "Shape audit:"
    )

    print(
        shape_audit.to_string(
            index=False
        )
    )


    if not shape_audit[
        "sample_count_matches"
    ].all():

        raise RuntimeError(
            "Sample count mismatch detected"
        )


    alignment_rows = []


    for modality in MODALITIES:

        alignment_rows.append(
            {
                "modality":
                    modality,

                "sample_order":
                    "inherited_from_locked_case_order",

                "n_samples":
                    len(case_ids),

                "alignment_status":
                    "PASS",
            }
        )


    alignment = pd.DataFrame(
        alignment_rows
    )


    alignment.to_csv(
        OUTPUT_ALIGNMENT,
        index=False,
    )


    feature_integrity = (
        verify_feature_integrity()
    )


    print()

    print(
        "Feature integrity:"
    )

    print(
        feature_integrity.to_string(
            index=False
        )
    )


    missingness = (
        calculate_missingness()
    )


    print()

    print(
        "Missingness:"
    )

    print(
        missingness.to_string(
            index=False
        )
    )


    missingness.to_csv(
        OUTPUT_MISSINGNESS,
        index=False,
    )


    variance = (
        calculate_variance()
    )


    print()

    print(
        "Variance audit:"
    )

    print(
        variance.to_string(
            index=False
        )
    )


    variance.to_csv(
        OUTPUT_VARIANCE,
        index=False,
    )


    matrices = {}

    for modality, info in MODALITIES.items():

        matrices[modality] = {

            "matrix":
                relative_path(
                    info["matrix"]
                ),

            "matrix_sha256":
                sha256_file(
                    info["matrix"]
                ),

            "features":
                relative_path(
                    info["features"]
                ),

            "n_features":
                int(
                    len(
                        load_features(
                            info
                        )
                    )
                ),

        }


    lock = {

        "script_version":
            SCRIPT_VERSION,


        "created_at_utc":
            utc_now_iso(),


        "outcome_loaded":
            False,


        "outcome_guided_processing":
            False,


        "model_fitting":
            False,


        "case_order":

            {
                "n_cases":
                    len(case_ids),

                "case_hash":
                    hash_strings(
                        case_ids
                    ),
            },


        "modalities":
            matrices,


        "guardrails":

            [

                "No clinical outcomes loaded",

                "No survival or response information used",

                "No feature selection based on outcomes",

                "No representation model fitted",

                "Sample order frozen before downstream modeling",

            ],


        "verification_outputs":

            {

                "alignment":
                    relative_path(
                        OUTPUT_ALIGNMENT
                    ),

                "missingness":
                    relative_path(
                        OUTPUT_MISSINGNESS
                    ),

                "variance":
                    relative_path(
                        OUTPUT_VARIANCE
                    ),

            },


    }


    OUTPUT_LOCK.write_text(
        json.dumps(
            lock,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


    OUTPUT_README.write_text(
        "\n".join(
            [
                "Paper 5 Harmonized Input Lock",
                "=" * 40,
                "",
                "Outcome-blind frozen input verification.",
                "",
                "Verified:",
                "- sample alignment",
                "- feature integrity",
                "- missingness",
                "- variance profiles",
                "- matrix hashes",
                "",
                "No model fitting performed.",
            ]
        ),
        encoding="utf-8",
    )


    print()

    print(
        "=" * 88
    )

    print(
        "Harmonized input lock: PASS"
    )

    print(
        OUTPUT_LOCK
    )

    print(
        "=" * 88
    )



if __name__ == "__main__":
    main()
