from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "04-outcome-blind-multimodal-qc-harmonization-v1"


SCRIPT_PATH = Path(__file__).resolve()

PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]


INPUT_DIR = (
    PAPER5_ROOT
    / "model_inputs"
    / "TARGET_OS_primary78"
)


OUTPUT_DIR = (
    PAPER5_ROOT
    / "model_inputs"
    / "TARGET_OS_primary78"
    / "harmonized"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CASE_ORDER = (
    INPUT_DIR
    / "TARGET_OS_primary_multimodal_case_order.csv"
)


RNA_MATRIX = (
    INPUT_DIR
    / "TARGET_OS_primary_rna_log2tpm.npy"
)

RNA_FEATURES = (
    INPUT_DIR
    / "TARGET_OS_primary_rna_features.csv"
)


METH_MATRIX = (
    INPUT_DIR
    / "TARGET_OS_primary_methylation_beta_corrected.npy"
)

METH_FEATURES = (
    INPUT_DIR
    / "TARGET_OS_primary_methylation_features_corrected.csv"
)


CNV_MATRIX = (
    INPUT_DIR
    / "TARGET_OS_primary_cnv_gene_copy_number.npy"
)

CNV_FEATURES = (
    INPUT_DIR
    / "TARGET_OS_primary_cnv_features.csv"
)


OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "multimodal_qc_summary.csv"
)

OUTPUT_MANIFEST = (
    OUTPUT_DIR
    / "multimodal_qc_manifest.json"
)

OUTPUT_README = (
    OUTPUT_DIR
    / "multimodal_qc_README.txt"
)


MISSING_THRESHOLD = 0.20


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
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()



def zscore_matrix(
    matrix: np.ndarray,
):
    mean = np.nanmean(
        matrix,
        axis=0,
    )

    std = np.nanstd(
        matrix,
        axis=0,
        ddof=1,
    )

    std[
        (~np.isfinite(std))
        |
        (std == 0)
    ] = 1.0

    return (
        matrix - mean
    ) / std



def remove_bad_features(
    matrix,
    features,
    missing_threshold,
):

    missing_fraction = (
        np.mean(
            ~np.isfinite(matrix),
            axis=0,
        )
    )


    variance = np.nanvar(
        matrix,
        axis=0,
        ddof=1,
    )


    keep = (
        (missing_fraction <= missing_threshold)
        &
        np.isfinite(variance)
        &
        (variance > 0)
    )


    qc = pd.DataFrame(
        {
            "feature": features,
            "missing_fraction": missing_fraction,
            "variance": variance,
            "kept": keep,
        }
    )


    return (
        matrix[:, keep],
        features[keep],
        qc,
    )



def beta_to_mvalue(beta):

    eps = 1e-6

    beta = np.clip(
        beta,
        eps,
        1 - eps,
    )

    return np.log2(
        beta /
        (1 - beta)
    )



def save_matrix(
    path,
    matrix,
):

    np.save(
        path,
        matrix.astype(
            np.float32
        ),
    )



def process_rna():

    matrix = np.load(
        RNA_MATRIX,
    )

    features = (
        pd.read_csv(
            RNA_FEATURES
        )
        ["feature"]
        .astype(str)
        .to_numpy()
    )


    matrix, features, qc = (
        remove_bad_features(
            matrix,
            features,
            0.0,
        )
    )


    matrix = zscore_matrix(
        matrix
    )


    out_matrix = (
        OUTPUT_DIR
        /
        "RNA_zscore.npy"
    )

    out_features = (
        OUTPUT_DIR
        /
        "RNA_features.csv"
    )

    save_matrix(
        out_matrix,
        matrix,
    )

    pd.DataFrame(
        {
            "feature": features
        }
    ).to_csv(
        out_features,
        index=False,
    )


    qc.to_csv(
        OUTPUT_DIR
        /
        "RNA_feature_qc.csv",
        index=False,
    )


    return {
        "modality": "RNA",
        "before_features": int(
            len(qc)
        ),
        "after_features": int(
            len(features)
        ),
        "matrix": relative_path(
            out_matrix
        ),
        "missing_removed": int(
            (~qc.kept).sum()
        ),
    }



def process_methylation():

    matrix = np.load(
        METH_MATRIX,
    )

    features = (
        pd.read_csv(
            METH_FEATURES
        )
        ["probe_id"]
        .astype(str)
        .to_numpy()
    )


    matrix, features, qc = (
        remove_bad_features(
            matrix,
            features,
            MISSING_THRESHOLD,
        )
    )


    matrix = beta_to_mvalue(
        matrix
    )


    matrix = zscore_matrix(
        matrix
    )


    out_matrix = (
        OUTPUT_DIR
        /
        "METHYLATION_Mvalue_zscore.npy"
    )


    save_matrix(
        out_matrix,
        matrix,
    )


    pd.DataFrame(
        {
            "probe_id": features
        }
    ).to_csv(
        OUTPUT_DIR
        /
        "METHYLATION_features.csv",
        index=False,
    )


    qc.to_csv(
        OUTPUT_DIR
        /
        "METHYLATION_feature_qc.csv",
        index=False,
    )


    return {
        "modality": "METHYLATION",
        "before_features": int(
            len(qc)
        ),
        "after_features": int(
            len(features)
        ),
        "matrix": relative_path(
            out_matrix
        ),
        "missing_removed": int(
            (~qc.kept).sum()
        ),
    }



def process_cnv():

    matrix = np.load(
        CNV_MATRIX,
    )


    features = (
        pd.read_csv(
            CNV_FEATURES
        )
        ["feature"]
        .astype(str)
        .to_numpy()
    )


    matrix = np.log2(
        np.clip(
            matrix,
            1e-6,
            None,
        )
        /
        2.0
    )


    matrix, features, qc = (
        remove_bad_features(
            matrix,
            features,
            0.0,
        )
    )


    matrix = zscore_matrix(
        matrix
    )


    out_matrix = (
        OUTPUT_DIR
        /
        "CNV_log2ratio_zscore.npy"
    )


    save_matrix(
        out_matrix,
        matrix,
    )


    pd.DataFrame(
        {
            "feature": features
        }
    ).to_csv(
        OUTPUT_DIR
        /
        "CNV_features.csv",
        index=False,
    )


    qc.to_csv(
        OUTPUT_DIR
        /
        "CNV_feature_qc.csv",
        index=False,
    )


    return {
        "modality": "CNV",
        "before_features": int(
            len(qc)
        ),
        "after_features": int(
            len(features)
        ),
        "matrix": relative_path(
            out_matrix
        ),
        "missing_removed": int(
            (~qc.kept).sum()
        ),
    }



def main():

    print("=" * 88)
    print(
        "Paper 5 - outcome blind multimodal QC"
    )
    print("=" * 88)

    print(
        f"Script version: {SCRIPT_VERSION}"
    )

    if not CASE_ORDER.exists():
        raise FileNotFoundError(
            CASE_ORDER
        )


    cases = pd.read_csv(
        CASE_ORDER
    )


    print(
        f"Cases: {cases.shape[0]}"
    )


    results = []


    print()
    print(
        "Processing RNA..."
    )

    results.append(
        process_rna()
    )


    print(
        "Processing methylation..."
    )

    results.append(
        process_methylation()
    )


    print(
        "Processing CNV..."
    )

    results.append(
        process_cnv()
    )


    summary = pd.DataFrame(
        results
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )


    print()
    print(
        summary.to_string(
            index=False
        )
    )


    manifest = {

        "script_version":
            SCRIPT_VERSION,

        "created_at_utc":
            utc_now_iso(),

        "outcome_loaded":
            False,

        "feature_selection_using_outcome":
            False,

        "model_fitting":
            False,


        "rules": {

            "RNA":
                "z-score after zero variance removal",

            "methylation":
                (
                    "remove >20% missing, "
                    "beta to M-value, z-score"
                ),

            "CNV":
                (
                    "log2(copy_number/2), "
                    "z-score"
                ),

        },


        "guardrails": [

            "No clinical outcome loaded",

            "No survival-guided feature selection",

            "No model fitting",

            "No cohort filtering",

        ],


        "outputs": {

            "summary":
                relative_path(
                    OUTPUT_SUMMARY
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


    OUTPUT_README.write_text(
        "\n".join(
            [
                "Paper 5 multimodal QC",
                "=" * 40,
                "",
                "Outcome blind processing.",
                "",
                "RNA: z-score",
                "Methylation: beta->M-value->z-score",
                "CNV: log2(CN/2)->z-score",
                "",
                "No outcome information used.",
            ]
        ),
        encoding="utf-8",
    )


    print()
    print(
        "QC harmonization PASS"
    )

    print(
        OUTPUT_DIR
    )



if __name__ == "__main__":
    main()
