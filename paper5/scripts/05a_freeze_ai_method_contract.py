from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05a-freeze-ai-method-contract-v1"
METHOD_NAME = "SABLE-MO"
METHOD_NAME_STATUS = "provisional"


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

INPUT_DIR = (
    PAPER5_ROOT
    / "model_inputs"
    / "TARGET_OS_primary78"
)

HARMONIZED_DIR = (
    INPUT_DIR
    / "harmonized"
)

PREMODEL_DIR = (
    INPUT_DIR
    / "premodel_qc"
)

OUTPUT_DIR = (
    PAPER5_ROOT
    / "method_contract"
    / "05a"
)

ARCHIVE_DIR = (
    OUTPUT_DIR
    / "archive"
)

CASE_ORDER_FILE = (
    INPUT_DIR
    / "TARGET_OS_primary_multimodal_case_order.csv"
)

HARMONIZED_LOCK_FILE = (
    HARMONIZED_DIR
    / "harmonized_input_lock.json"
)

PREMODEL_MANIFEST_FILE = (
    PREMODEL_DIR
    / "prestandardization_variance_audit_manifest.json"
)

FOLD_LOCAL_CONTRACT_FILE = (
    PREMODEL_DIR
    / "fold_local_preprocessing_contract.json"
)

FEATURE_BUDGET_PLAN_FILE = (
    PREMODEL_DIR
    / "representation_feature_budget_plan.csv"
)

PREMODEL_SUMMARY_FILE = (
    PREMODEL_DIR
    / "prestandardization_variance_summary.csv"
)

RECONSTRUCTION_AUDIT_FILE = (
    PREMODEL_DIR
    / "harmonized_reconstruction_audit.csv"
)

OUTPUT_OUTER_ASSIGNMENTS = (
    OUTPUT_DIR
    / "target_outer_cv_assignments.csv"
)

OUTPUT_INNER_ASSIGNMENTS = (
    OUTPUT_DIR
    / "target_inner_cv_assignments.csv"
)

OUTPUT_SPLIT_SUMMARY = (
    OUTPUT_DIR
    / "target_cv_split_summary.csv"
)

OUTPUT_NEURAL_ASSIGNMENTS = (
    OUTPUT_DIR
    / "target_neural_training_assignments.csv"
)

OUTPUT_MODEL_REGISTRY = (
    OUTPUT_DIR
    / "model_family_registry.csv"
)

OUTPUT_ABLATION_REGISTRY = (
    OUTPUT_DIR
    / "sable_mo_ablation_registry.csv"
)

OUTPUT_BASELINE_GRID = (
    OUTPUT_DIR
    / "baseline_hyperparameter_grid.csv"
)

OUTPUT_ARCHITECTURE_PROFILES = (
    OUTPUT_DIR
    / "sable_mo_architecture_profiles.csv"
)

OUTPUT_REGULARIZATION_PROFILES = (
    OUTPUT_DIR
    / "sable_mo_regularization_profiles.csv"
)

OUTPUT_SIMULATION_SCENARIOS = (
    OUTPUT_DIR
    / "simulation_scenarios.csv"
)

OUTPUT_SIMULATION_REPLICATES = (
    OUTPUT_DIR
    / "simulation_replicate_registry.csv"
)

OUTPUT_METRIC_REGISTRY = (
    OUTPUT_DIR
    / "metric_registry.csv"
)

OUTPUT_FAILURE_GATES = (
    OUTPUT_DIR
    / "failure_gate_registry.csv"
)

OUTPUT_RANDOM_CONTROLS = (
    OUTPUT_DIR
    / "random_control_registry.csv"
)

OUTPUT_MULTIPLICITY = (
    OUTPUT_DIR
    / "multiplicity_family_registry.csv"
)

OUTPUT_SELECTION_POLICY = (
    OUTPUT_DIR
    / "model_selection_policy.json"
)

OUTPUT_EXTERNAL_BENCHMARK = (
    OUTPUT_DIR
    / "external_method_benchmark_contract.json"
)

OUTPUT_ENVIRONMENT = (
    OUTPUT_DIR
    / "environment_snapshot.json"
)

OUTPUT_CONTRACT = (
    OUTPUT_DIR
    / "ai_method_contract.json"
)

OUTPUT_README = (
    OUTPUT_DIR
    / "ai_method_contract_README.txt"
)

OUTPUT_LOCK = (
    OUTPUT_DIR
    / "ai_method_contract_lock.json"
)


# ---------------------------------------------------------------------
# Frozen design constants
# ---------------------------------------------------------------------

EXPECTED_CASES = 78
MASTER_SPLIT_SEED = 20260811
SIMULATION_MASTER_SEED = 20260812

OUTER_REPEATS = 5
OUTER_FOLDS = 5
INNER_FOLDS = 4
EARLY_STOP_FRACTION = 0.20

INNER_MODEL_SEEDS = [1103, 2207]
OUTER_MODEL_SEEDS = [1103, 2207, 3301]
FINAL_FULL_DATA_SEEDS = [
    1103,
    2207,
    3301,
    4409,
    5519,
    6637,
    7753,
    8863,
    9973,
    11027,
]

PAIRING_PERMUTATIONS = 100
FINAL_STABILITY_RESAMPLES = 100
FINAL_STABILITY_SUBSAMPLE_FRACTION = 0.80

FEATURE_BUDGETS = {
    "RNA": {
        "sensitivity_low": 2_000,
        "primary": 5_000,
        "sensitivity_high": 10_000,
    },
    "METHYLATION": {
        "sensitivity_low": 5_000,
        "primary": 10_000,
        "sensitivity_high": 20_000,
    },
    "CNV": {
        "sensitivity_low": 2_000,
        "primary": 5_000,
        "sensitivity_high": 10_000,
    },
}

PRIMARY_SELECTION_WEIGHTS = {
    "heldout_reconstruction_score": 0.30,
    "random_adjusted_cross_view_mrr": 0.30,
    "missing_view_reconstruction_score": 0.25,
    "modality_balance_score": 0.15,
}

STABILITY_THRESHOLDS = {
    "primary_median_factor_correlation": 0.70,
    "primary_q10_factor_correlation": 0.40,
    "primary_inclusion_frequency": 0.80,
    "minimum_factor_variance": 1e-4,
    "minimum_supported_modalities": 2,
    "minimum_loading_energy_fraction_per_supported_modality": 0.15,
    "sensitivity_low_median_factor_correlation": 0.60,
    "sensitivity_high_median_factor_correlation": 0.80,
}


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def utc_stamp() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")


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


def resolve_project_path(value: Any) -> Path:
    path = Path(
        clean_text(value)
    )

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


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


def hash_strings(values: list[str]) -> str:
    digest = hashlib.sha256()

    for value in values:
        digest.update(
            str(value).encode("utf-8")
        )
        digest.update(b"\n")

    return digest.hexdigest()


def stable_seed(
    master_seed: int,
    *parts: Any,
) -> int:
    digest = hashlib.sha256()
    digest.update(
        str(master_seed).encode("utf-8")
    )

    for part in parts:
        digest.update(b"|")
        digest.update(
            str(part).encode("utf-8")
        )

    value = int.from_bytes(
        digest.digest()[:8],
        byteorder="big",
        signed=False,
    )

    return int(
        value % (2**32 - 1)
    )


def write_json(
    value: Any,
    path: Path,
) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_table(
    table: pd.DataFrame,
    path: Path,
) -> None:
    table.to_csv(
        path,
        index=False,
        lineterminator="\n",
    )


def package_version(
    distribution_name: str,
) -> dict[str, Any]:
    try:
        version = importlib.metadata.version(
            distribution_name
        )
        return {
            "installed": True,
            "version": version,
        }
    except importlib.metadata.PackageNotFoundError:
        return {
            "installed": False,
            "version": None,
        }


# ---------------------------------------------------------------------
# Existing-contract protection
# ---------------------------------------------------------------------


def verify_existing_contract() -> None:
    require_file(OUTPUT_LOCK)
    require_file(OUTPUT_CONTRACT)

    lock = json.loads(
        OUTPUT_LOCK.read_text(
            encoding="utf-8"
        )
    )

    contract = json.loads(
        OUTPUT_CONTRACT.read_text(
            encoding="utf-8"
        )
    )

    failures: list[str] = []

    for relative_name, record in lock.get(
        "generated_files",
        {},
    ).items():
        path = resolve_project_path(
            relative_name
        )

        if not path.exists():
            failures.append(
                f"Missing generated file: {relative_name}"
            )
            continue

        observed_hash = sha256_file(path)
        expected_hash = clean_text(
            record.get("sha256")
        )

        if observed_hash != expected_hash:
            failures.append(
                "Generated-file hash mismatch: "
                f"{relative_name}"
            )

    for relative_name, record in contract.get(
        "upstream_artifacts",
        {},
    ).items():
        path = resolve_project_path(
            relative_name
        )

        if not path.exists():
            failures.append(
                f"Missing upstream artifact: {relative_name}"
            )
            continue

        observed_hash = sha256_file(path)
        expected_hash = clean_text(
            record.get("sha256")
        )

        if observed_hash != expected_hash:
            failures.append(
                "Upstream-artifact hash mismatch: "
                f"{relative_name}"
            )

    expected_contract_hash = clean_text(
        lock.get("contract_sha256")
    )
    observed_contract_hash = sha256_file(
        OUTPUT_CONTRACT
    )

    if (
        expected_contract_hash
        != observed_contract_hash
    ):
        failures.append(
            "Contract JSON hash mismatch."
        )

    if failures:
        raise RuntimeError(
            "Existing AI method contract verification failed:\n"
            + "\n".join(
                f"- {failure}"
                for failure in failures
            )
        )

    print("=" * 88)
    print(
        "Existing Paper 5 AI method contract: PASS"
    )
    print("=" * 88)
    print(
        f"Contract: {relative_path(OUTPUT_CONTRACT)}"
    )
    print(
        f"Contract SHA256: {observed_contract_hash}"
    )
    print(
        "No files were rewritten."
    )
    print("=" * 88)


def archive_existing_contract(
    amendment_reason: str,
) -> Path:
    archive_target = (
        ARCHIVE_DIR
        / utc_stamp()
    )

    archive_target.mkdir(
        parents=True,
        exist_ok=False,
    )

    for path in OUTPUT_DIR.iterdir():
        if path.name == "archive":
            continue

        destination = (
            archive_target
            / path.name
        )

        if path.is_file():
            shutil.copy2(
                path,
                destination,
            )
        elif path.is_dir():
            shutil.copytree(
                path,
                destination,
            )

    (
        archive_target
        / "amendment_reason.txt"
    ).write_text(
        amendment_reason.strip()
        + "\n",
        encoding="utf-8",
    )

    return archive_target


# ---------------------------------------------------------------------
# Upstream validation
# ---------------------------------------------------------------------


def load_case_order() -> list[str]:
    require_file(CASE_ORDER_FILE)

    table = pd.read_csv(
        CASE_ORDER_FILE,
        dtype=str,
    ).fillna("")

    if "case_id" not in table.columns:
        raise ValueError(
            "The locked case-order file does not "
            "contain case_id."
        )

    case_ids = (
        table["case_id"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    if len(case_ids) != EXPECTED_CASES:
        raise RuntimeError(
            f"Expected {EXPECTED_CASES} cases; "
            f"found {len(case_ids)}."
        )

    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError(
            "Duplicate case IDs were found in the "
            "locked case order."
        )

    return case_ids


def verify_premodel_outputs(
    manifest: dict[str, Any],
) -> None:
    outputs = manifest.get(
        "outputs",
        {},
    )

    if not outputs:
        raise RuntimeError(
            "The 04c manifest does not contain "
            "output hashes."
        )

    print(
        "Verifying 04c output artifacts..."
    )

    for relative_name, record in sorted(
        outputs.items()
    ):
        path = resolve_project_path(
            relative_name
        )
        require_file(path)

        observed_hash = sha256_file(path)
        expected_hash = clean_text(
            record.get("sha256")
        )

        if observed_hash != expected_hash:
            raise RuntimeError(
                "04c output hash mismatch: "
                f"{relative_name}"
            )


def validate_feature_budget_plan() -> pd.DataFrame:
    require_file(
        FEATURE_BUDGET_PLAN_FILE
    )

    table = pd.read_csv(
        FEATURE_BUDGET_PLAN_FILE,
        dtype=str,
    ).fillna("")

    required = {
        "modality",
        "ranking_method",
        "budget_role",
        "n_features",
        "held_out_benchmark_use",
    }

    missing = required - set(
        table.columns
    )

    if missing:
        raise ValueError(
            "Feature-budget plan is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    variance_rows = table[
        table["ranking_method"].eq(
            "prestandardization_variance"
        )
    ].copy()

    observed: dict[
        tuple[str, str],
        int,
    ] = {}

    for _, row in variance_rows.iterrows():
        key = (
            str(row["modality"]),
            str(row["budget_role"]),
        )
        observed[key] = int(
            float(row["n_features"])
        )

    expected: dict[
        tuple[str, str],
        int,
    ] = {}

    for modality, roles in (
        FEATURE_BUDGETS.items()
    ):
        for role, count in roles.items():
            expected[
                (modality, role)
            ] = count

    if observed != expected:
        raise RuntimeError(
            "The 04c feature budgets do not match "
            "the frozen Paper 5 budget contract.\n"
            f"Observed: {observed}\n"
            f"Expected: {expected}"
        )

    global_candidate_rows = table[
        table["candidate_file"].astype(str).ne("")
    ] if "candidate_file" in table.columns else table.iloc[0:0]

    if not global_candidate_rows.empty:
        invalid = global_candidate_rows[
            ~global_candidate_rows[
                "held_out_benchmark_use"
            ].astype(str).str.contains(
                "forbidden",
                case=False,
                regex=False,
            )
        ]

        if not invalid.empty:
            raise RuntimeError(
                "At least one full-cohort candidate "
                "feature list is not explicitly "
                "forbidden in held-out evaluation."
            )

    return table


def validate_upstream_state(
    case_ids: list[str],
) -> dict[str, dict[str, Any]]:
    for path in [
        HARMONIZED_LOCK_FILE,
        PREMODEL_MANIFEST_FILE,
        FOLD_LOCAL_CONTRACT_FILE,
        FEATURE_BUDGET_PLAN_FILE,
        PREMODEL_SUMMARY_FILE,
        RECONSTRUCTION_AUDIT_FILE,
    ]:
        require_file(path)

    harmonized_lock = json.loads(
        HARMONIZED_LOCK_FILE.read_text(
            encoding="utf-8"
        )
    )

    premodel_manifest = json.loads(
        PREMODEL_MANIFEST_FILE.read_text(
            encoding="utf-8"
        )
    )

    fold_local_contract = json.loads(
        FOLD_LOCAL_CONTRACT_FILE.read_text(
            encoding="utf-8"
        )
    )

    expected_case_hash = hash_strings(
        case_ids
    )

    lock_case_hash = clean_text(
        harmonized_lock.get(
            "case_order",
            {},
        ).get(
            "case_hash",
            "",
        )
    )

    premodel_case_hash = clean_text(
        premodel_manifest.get(
            "case_order",
            {},
        ).get(
            "case_hash",
            "",
        )
    )

    if (
        lock_case_hash
        != expected_case_hash
        or premodel_case_hash
        != expected_case_hash
    ):
        raise RuntimeError(
            "Case-order hash mismatch across "
            "upstream locks."
        )

    outcome_flags = {
        "harmonized_outcome_loaded": bool(
            harmonized_lock.get(
                "outcome_loaded",
                True,
            )
        ),
        "harmonized_outcome_guided_processing": bool(
            harmonized_lock.get(
                "outcome_guided_processing",
                True,
            )
        ),
        "premodel_outcome_loaded": bool(
            premodel_manifest.get(
                "outcome_loaded",
                True,
            )
        ),
        "premodel_outcome_guided_feature_selection": bool(
            premodel_manifest.get(
                "outcome_guided_feature_selection",
                True,
            )
        ),
        "premodel_post_standardization_variance_used": bool(
            premodel_manifest.get(
                "post_standardization_variance_used_for_selection",
                True,
            )
        ),
        "premodel_model_fitting": bool(
            premodel_manifest.get(
                "model_fitting_performed",
                True,
            )
        ),
    }

    if any(outcome_flags.values()):
        raise RuntimeError(
            "An upstream guardrail is not in the "
            "required false state:\n"
            + json.dumps(
                outcome_flags,
                indent=2,
                sort_keys=True,
            )
        )

    expected_lock_hash = clean_text(
        premodel_manifest.get(
            "upstream_lock",
            {},
        ).get(
            "sha256",
            "",
        )
    )

    observed_lock_hash = sha256_file(
        HARMONIZED_LOCK_FILE
    )

    if expected_lock_hash != observed_lock_hash:
        raise RuntimeError(
            "The 04b lock hash does not match "
            "the hash recorded by 04c."
        )

    forbidden_operations = [
        clean_text(value)
        for value in fold_local_contract.get(
            "forbidden_operations",
            [],
        )
    ]

    required_forbidden_phrase = (
        "Variance ranking after z-score standardization"
    )

    if required_forbidden_phrase not in forbidden_operations:
        raise RuntimeError(
            "The fold-local preprocessing contract "
            "does not explicitly forbid post-z-score "
            "variance ranking."
        )

    verify_premodel_outputs(
        premodel_manifest
    )

    validate_feature_budget_plan()

    upstream_paths = [
        CASE_ORDER_FILE,
        HARMONIZED_LOCK_FILE,
        PREMODEL_MANIFEST_FILE,
        FOLD_LOCAL_CONTRACT_FILE,
        FEATURE_BUDGET_PLAN_FILE,
        PREMODEL_SUMMARY_FILE,
        RECONSTRUCTION_AUDIT_FILE,
    ]

    return {
        relative_path(path): {
            "sha256": sha256_file(path),
            "size_bytes": int(
                path.stat().st_size
            ),
        }
        for path in upstream_paths
    }


# ---------------------------------------------------------------------
# Deterministic nested split construction
# ---------------------------------------------------------------------


def balanced_fold_labels(
    n_items: int,
    n_folds: int,
    seed: int,
) -> np.ndarray:
    if n_folds < 2:
        raise ValueError(
            "n_folds must be at least 2."
        )

    if n_items < n_folds:
        raise ValueError(
            "Number of items is smaller than "
            "the requested number of folds."
        )

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_items)

    labels = np.empty(
        n_items,
        dtype=int,
    )

    for fold_index, indices in enumerate(
        np.array_split(
            permutation,
            n_folds,
        )
    ):
        labels[indices] = fold_index

    return labels


def split_fit_and_early_stop(
    indices: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[set[int], set[int]]:
    if not 0.0 < fraction < 0.5:
        raise ValueError(
            "Early-stopping fraction must be between 0 and 0.5."
        )

    if indices.size < 5:
        raise ValueError(
            "Too few samples for a separate early-stopping subset."
        )

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(indices)

    n_early = max(
        1,
        int(math.ceil(indices.size * fraction)),
    )

    if indices.size - n_early < 3:
        n_early = indices.size - 3

    early = {
        int(value)
        for value in shuffled[:n_early]
    }
    fit = {
        int(value)
        for value in shuffled[n_early:]
    }

    if not early or not fit:
        raise RuntimeError(
            "Could not create non-empty fit and early-stopping subsets."
        )

    return fit, early


def build_nested_cv_splits(
    case_ids: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    outer_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    neural_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    outer_assignment_hashes: set[str] = set()

    for repeat_index in range(
        1,
        OUTER_REPEATS + 1,
    ):
        outer_seed = stable_seed(
            MASTER_SPLIT_SEED,
            "outer",
            repeat_index,
        )

        outer_labels = balanced_fold_labels(
            n_items=len(case_ids),
            n_folds=OUTER_FOLDS,
            seed=outer_seed,
        )

        assignment_hash = hash_strings(
            [
                f"{case_id}:{int(label)}"
                for case_id, label in zip(
                    case_ids,
                    outer_labels,
                )
            ]
        )

        if assignment_hash in outer_assignment_hashes:
            raise RuntimeError(
                "Duplicate outer-fold assignment "
                "was generated across repeats."
            )

        outer_assignment_hashes.add(
            assignment_hash
        )

        for outer_fold_index in range(
            1,
            OUTER_FOLDS + 1,
        ):
            split_id = (
                f"R{repeat_index:02d}_"
                f"F{outer_fold_index:02d}"
            )

            test_mask = (
                outer_labels
                == (outer_fold_index - 1)
            )

            train_indices = np.flatnonzero(
                ~test_mask
            )
            test_indices = np.flatnonzero(
                test_mask
            )

            outer_early_seed = stable_seed(
                MASTER_SPLIT_SEED,
                "outer_early_stop",
                repeat_index,
                outer_fold_index,
            )

            (
                outer_fit_set,
                outer_early_set,
            ) = split_fit_and_early_stop(
                indices=train_indices,
                fraction=EARLY_STOP_FRACTION,
                seed=outer_early_seed,
            )

            for case_index, case_id in enumerate(case_ids):
                if case_index in set(test_indices.tolist()):
                    neural_role = "test"
                elif case_index in outer_early_set:
                    neural_role = "early_stop"
                else:
                    neural_role = "fit"

                neural_rows.append(
                    {
                        "stage": "outer_evaluation",
                        "repeat_index": repeat_index,
                        "outer_fold": outer_fold_index,
                        "split_id": split_id,
                        "inner_fold": 0,
                        "partition_seed": outer_early_seed,
                        "case_index": case_index,
                        "case_id": case_id,
                        "role": neural_role,
                    }
                )

            inner_seed = stable_seed(
                MASTER_SPLIT_SEED,
                "inner",
                repeat_index,
                outer_fold_index,
            )

            inner_labels = balanced_fold_labels(
                n_items=train_indices.size,
                n_folds=INNER_FOLDS,
                seed=inner_seed,
            )

            for case_index, case_id in enumerate(
                case_ids
            ):
                outer_rows.append(
                    {
                        "repeat_index": repeat_index,
                        "outer_fold": outer_fold_index,
                        "split_id": split_id,
                        "outer_seed": outer_seed,
                        "case_index": case_index,
                        "case_id": case_id,
                        "outer_role": (
                            "test"
                            if test_mask[case_index]
                            else "train"
                        ),
                    }
                )

            inner_fit_sizes: list[int] = []
            inner_early_sizes: list[int] = []

            for inner_fold_index in range(
                1,
                INNER_FOLDS + 1,
            ):
                validation_mask = (
                    inner_labels
                    == (inner_fold_index - 1)
                )

                inner_validation_indices = train_indices[
                    validation_mask
                ]
                inner_training_indices = train_indices[
                    ~validation_mask
                ]

                inner_early_seed = stable_seed(
                    MASTER_SPLIT_SEED,
                    "inner_early_stop",
                    repeat_index,
                    outer_fold_index,
                    inner_fold_index,
                )

                (
                    inner_fit_set,
                    inner_early_set,
                ) = split_fit_and_early_stop(
                    indices=inner_training_indices,
                    fraction=EARLY_STOP_FRACTION,
                    seed=inner_early_seed,
                )

                inner_fit_sizes.append(
                    len(inner_fit_set)
                )
                inner_early_sizes.append(
                    len(inner_early_set)
                )

                validation_set = {
                    int(value)
                    for value in inner_validation_indices
                }

                for local_index, case_index in enumerate(
                    train_indices
                ):
                    case_id = case_ids[
                        int(case_index)
                    ]

                    role = (
                        "validation"
                        if validation_mask[local_index]
                        else "train"
                    )

                    inner_rows.append(
                        {
                            "repeat_index": repeat_index,
                            "outer_fold": outer_fold_index,
                            "split_id": split_id,
                            "inner_fold": inner_fold_index,
                            "inner_seed": inner_seed,
                            "case_index": int(case_index),
                            "case_id": case_id,
                            "inner_role": role,
                        }
                    )

                for case_index, case_id in enumerate(case_ids):
                    if case_index in set(test_indices.tolist()):
                        neural_role = "outer_test_excluded"
                    elif case_index in validation_set:
                        neural_role = "validation"
                    elif case_index in inner_early_set:
                        neural_role = "early_stop"
                    else:
                        neural_role = "fit"

                    neural_rows.append(
                        {
                            "stage": "inner_model_selection",
                            "repeat_index": repeat_index,
                            "outer_fold": outer_fold_index,
                            "split_id": split_id,
                            "inner_fold": inner_fold_index,
                            "partition_seed": inner_early_seed,
                            "case_index": case_index,
                            "case_id": case_id,
                            "role": neural_role,
                        }
                    )

            summary_rows.append(
                {
                    "repeat_index": repeat_index,
                    "outer_fold": outer_fold_index,
                    "split_id": split_id,
                    "outer_seed": outer_seed,
                    "inner_seed": inner_seed,
                    "n_outer_train": int(
                        train_indices.size
                    ),
                    "n_outer_test": int(
                        test_indices.size
                    ),
                    "n_inner_folds": INNER_FOLDS,
                    "n_outer_fit_before_refit": len(outer_fit_set),
                    "n_outer_early_stop": len(outer_early_set),
                    "minimum_inner_fit_before_refit": min(inner_fit_sizes),
                    "maximum_inner_fit_before_refit": max(inner_fit_sizes),
                    "minimum_inner_early_stop": min(inner_early_sizes),
                    "maximum_inner_early_stop": max(inner_early_sizes),
                    "minimum_inner_validation_size": int(
                        math.floor(
                            train_indices.size
                            / INNER_FOLDS
                        )
                    ),
                    "maximum_inner_validation_size": int(
                        math.ceil(
                            train_indices.size
                            / INNER_FOLDS
                        )
                    ),
                    "stratification_used": False,
                    "outcome_used_for_split": False,
                }
            )

    outer = pd.DataFrame(
        outer_rows
    )
    inner = pd.DataFrame(
        inner_rows
    )
    neural = pd.DataFrame(
        neural_rows
    )
    summary = pd.DataFrame(
        summary_rows
    )

    for repeat_index, part in outer.groupby(
        "repeat_index"
    ):
        test_counts = (
            part[
                part["outer_role"].eq("test")
            ]
            .groupby("case_id")
            .size()
        )

        if (
            test_counts.shape[0]
            != len(case_ids)
            or not test_counts.eq(1).all()
        ):
            raise RuntimeError(
                "Each case must appear exactly once "
                "as outer test per repeat. "
                f"Failed repeat: {repeat_index}"
            )

    test_sizes = summary[
        "n_outer_test"
    ].to_numpy(dtype=int)

    if test_sizes.max() - test_sizes.min() > 1:
        raise RuntimeError(
            "Outer test-fold sizes are not balanced."
        )

    return outer, inner, neural, summary


# ---------------------------------------------------------------------
# Model registries
# ---------------------------------------------------------------------


def build_model_registry() -> pd.DataFrame:
    rows = [
        {
            "model_id": "PCA_RNA",
            "family": "modality_specific_PCA",
            "role": "primary_linear_baseline",
            "views": "RNA",
            "integrated_representation": False,
            "eligible_for_nested_selection": False,
            "eligible_for_final_freeze": False,
            "complexity_rank": 0,
            "hyperparameter_source": "fixed",
            "outcome_access": False,
            "notes": "Descriptive single-view baseline.",
        },
        {
            "model_id": "PCA_METHYLATION",
            "family": "modality_specific_PCA",
            "role": "primary_linear_baseline",
            "views": "METHYLATION",
            "integrated_representation": False,
            "eligible_for_nested_selection": False,
            "eligible_for_final_freeze": False,
            "complexity_rank": 0,
            "hyperparameter_source": "fixed",
            "outcome_access": False,
            "notes": "Descriptive single-view baseline.",
        },
        {
            "model_id": "PCA_CNV",
            "family": "modality_specific_PCA",
            "role": "primary_linear_baseline",
            "views": "CNV",
            "integrated_representation": False,
            "eligible_for_nested_selection": False,
            "eligible_for_final_freeze": False,
            "complexity_rank": 0,
            "hyperparameter_source": "fixed",
            "outcome_access": False,
            "notes": "Descriptive single-view baseline.",
        },
        {
            "model_id": "BLOCK_PCA",
            "family": "block_scaled_PCA",
            "role": "primary_integrated_baseline",
            "views": "RNA;METHYLATION;CNV",
            "integrated_representation": True,
            "eligible_for_nested_selection": True,
            "eligible_for_final_freeze": True,
            "complexity_rank": 1,
            "hyperparameter_source": "fixed",
            "outcome_access": False,
            "notes": (
                "Each fold-local standardized block is scaled to equal total "
                "training variance before concatenation."
            ),
        },
        {
            "model_id": "MOFA_K5",
            "family": "MOFA_plus",
            "role": "primary_probabilistic_baseline",
            "views": "RNA;METHYLATION;CNV",
            "integrated_representation": True,
            "eligible_for_nested_selection": True,
            "eligible_for_final_freeze": True,
            "complexity_rank": 2,
            "hyperparameter_source": "prespecified_rank_sweep",
            "outcome_access": False,
            "notes": "Gaussian views; K=5; native missing-value handling.",
        },
        {
            "model_id": "MOFA_K10",
            "family": "MOFA_plus",
            "role": "primary_probabilistic_baseline",
            "views": "RNA;METHYLATION;CNV",
            "integrated_representation": True,
            "eligible_for_nested_selection": True,
            "eligible_for_final_freeze": True,
            "complexity_rank": 3,
            "hyperparameter_source": "prespecified_rank_sweep",
            "outcome_access": False,
            "notes": "Gaussian views; K=10; native missing-value handling.",
        },
        {
            "model_id": "MOFA_K15",
            "family": "MOFA_plus",
            "role": "primary_probabilistic_baseline",
            "views": "RNA;METHYLATION;CNV",
            "integrated_representation": True,
            "eligible_for_nested_selection": True,
            "eligible_for_final_freeze": True,
            "complexity_rank": 4,
            "hyperparameter_source": "prespecified_rank_sweep",
            "outcome_access": False,
            "notes": "Gaussian views; K=15; native missing-value handling.",
        },
        {
            "model_id": "MOFA_K20",
            "family": "MOFA_plus",
            "role": "primary_probabilistic_baseline",
            "views": "RNA;METHYLATION;CNV",
            "integrated_representation": True,
            "eligible_for_nested_selection": True,
            "eligible_for_final_freeze": True,
            "complexity_rank": 5,
            "hyperparameter_source": "prespecified_rank_sweep",
            "outcome_access": False,
            "notes": "Gaussian views; K=20; native missing-value handling.",
        },
        {
            "model_id": "RIDGE_GCCA",
            "family": "ridge_generalized_CCA",
            "role": "secondary_regularized_baseline",
            "views": "RNA;METHYLATION;CNV",
            "integrated_representation": True,
            "eligible_for_nested_selection": False,
            "eligible_for_final_freeze": False,
            "complexity_rank": 5,
            "hyperparameter_source": "inner_cv",
            "outcome_access": False,
            "notes": (
                "Exploratory regularized baseline. Ordinary unregularized CCA "
                "is forbidden because p greatly exceeds n."
            ),
        },
        {
            "model_id": "V0_BALANCED_AE",
            "family": "SABLE_MO_ablation",
            "role": "primary_nonlinear_ablation",
            "views": "RNA;METHYLATION;CNV",
            "integrated_representation": True,
            "eligible_for_nested_selection": True,
            "eligible_for_final_freeze": True,
            "complexity_rank": 6,
            "hyperparameter_source": "simulation_selected",
            "outcome_access": False,
            "notes": "Block-balanced reconstruction only.",
        },
        {
            "model_id": "V1_SHARED_PRIVATE_AE",
            "family": "SABLE_MO_ablation",
            "role": "primary_nonlinear_ablation",
            "views": "RNA;METHYLATION;CNV",
            "integrated_representation": True,
            "eligible_for_nested_selection": True,
            "eligible_for_final_freeze": True,
            "complexity_rank": 7,
            "hyperparameter_source": "simulation_selected",
            "outcome_access": False,
            "notes": "V0 plus shared/private decomposition.",
        },
        {
            "model_id": "V2_CONTRASTIVE_SPAE",
            "family": "SABLE_MO_ablation",
            "role": "primary_nonlinear_ablation",
            "views": "RNA;METHYLATION;CNV",
            "integrated_representation": True,
            "eligible_for_nested_selection": True,
            "eligible_for_final_freeze": True,
            "complexity_rank": 8,
            "hyperparameter_source": "simulation_selected",
            "outcome_access": False,
            "notes": "V1 plus paired cross-view contrastive alignment.",
        },
        {
            "model_id": "V3_DROPOUT_SPAE",
            "family": "SABLE_MO_ablation",
            "role": "primary_nonlinear_ablation",
            "views": "RNA;METHYLATION;CNV",
            "integrated_representation": True,
            "eligible_for_nested_selection": True,
            "eligible_for_final_freeze": True,
            "complexity_rank": 9,
            "hyperparameter_source": "simulation_selected",
            "outcome_access": False,
            "notes": "V2 plus modality dropout and missing-view reconstruction.",
        },
        {
            "model_id": "V4_SABLE_MO",
            "family": "SABLE_MO",
            "role": "proposed_primary_method",
            "views": "RNA;METHYLATION;CNV",
            "integrated_representation": True,
            "eligible_for_nested_selection": True,
            "eligible_for_final_freeze": True,
            "complexity_rank": 10,
            "hyperparameter_source": "simulation_selected",
            "outcome_access": False,
            "notes": (
                "V3 plus resampling-based factor alignment and stability gate. "
                "Proposed method is not guaranteed to win the benchmark."
            ),
        },
        {
            "model_id": "V5_SABLE_VAE",
            "family": "SABLE_MO_variational_extension",
            "role": "secondary_exploratory",
            "views": "RNA;METHYLATION;CNV",
            "integrated_representation": True,
            "eligible_for_nested_selection": False,
            "eligible_for_final_freeze": False,
            "complexity_rank": 11,
            "hyperparameter_source": "simulation_selected",
            "outcome_access": False,
            "notes": (
                "Variational extension. Exploratory because n=78 adds substantial "
                "posterior and optimization uncertainty."
            ),
        },
    ]

    return pd.DataFrame(rows)


def build_ablation_registry() -> pd.DataFrame:
    rows = [
        {
            "variant_id": "V0_BALANCED_AE",
            "block_balanced_reconstruction": True,
            "shared_private_decomposition": False,
            "paired_contrastive_alignment": False,
            "modality_dropout": False,
            "missing_view_reconstruction": False,
            "stability_gate": False,
            "variational_latent": False,
            "prespecified_component_hypothesis": (
                "Block normalization prevents methylation dominance and improves "
                "held-out reconstruction fairness."
            ),
        },
        {
            "variant_id": "V1_SHARED_PRIVATE_AE",
            "block_balanced_reconstruction": True,
            "shared_private_decomposition": True,
            "paired_contrastive_alignment": False,
            "modality_dropout": False,
            "missing_view_reconstruction": False,
            "stability_gate": False,
            "variational_latent": False,
            "prespecified_component_hypothesis": (
                "Shared/private decomposition separates cross-view structure from "
                "assay-specific variation."
            ),
        },
        {
            "variant_id": "V2_CONTRASTIVE_SPAE",
            "block_balanced_reconstruction": True,
            "shared_private_decomposition": True,
            "paired_contrastive_alignment": True,
            "modality_dropout": False,
            "missing_view_reconstruction": False,
            "stability_gate": False,
            "variational_latent": False,
            "prespecified_component_hypothesis": (
                "Paired contrastive alignment improves same-patient cross-view retrieval."
            ),
        },
        {
            "variant_id": "V3_DROPOUT_SPAE",
            "block_balanced_reconstruction": True,
            "shared_private_decomposition": True,
            "paired_contrastive_alignment": True,
            "modality_dropout": True,
            "missing_view_reconstruction": True,
            "stability_gate": False,
            "variational_latent": False,
            "prespecified_component_hypothesis": (
                "Modality dropout improves partial-view robustness and RNA-only projection."
            ),
        },
        {
            "variant_id": "V4_SABLE_MO",
            "block_balanced_reconstruction": True,
            "shared_private_decomposition": True,
            "paired_contrastive_alignment": True,
            "modality_dropout": True,
            "missing_view_reconstruction": True,
            "stability_gate": True,
            "variational_latent": False,
            "prespecified_component_hypothesis": (
                "The stability gate removes attractive but non-reproducible latent factors."
            ),
        },
        {
            "variant_id": "V5_SABLE_VAE",
            "block_balanced_reconstruction": True,
            "shared_private_decomposition": True,
            "paired_contrastive_alignment": True,
            "modality_dropout": True,
            "missing_view_reconstruction": True,
            "stability_gate": True,
            "variational_latent": True,
            "prespecified_component_hypothesis": (
                "The variational extension tests whether posterior regularization improves "
                "generalization despite the small sample size."
            ),
        },
    ]

    return pd.DataFrame(rows)


def build_baseline_grid() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for rank in [5, 10, 15, 20]:
        rows.append(
            {
                "model_family": "MOFA_plus",
                "parameter_set_id": f"MOFA_K{rank}",
                "latent_rank": rank,
                "ridge_penalty": "",
                "feature_budget_role": "primary",
                "likelihoods": "gaussian;gaussian;gaussian",
                "native_missingness": True,
                "inner_cv_required": False,
                "notes": "ARD and spike-and-slab sparsity enabled.",
            }
        )

    for rank in [5, 10, 15]:
        for penalty in [0.1, 1.0, 10.0, 100.0]:
            rows.append(
                {
                    "model_family": "ridge_generalized_CCA",
                    "parameter_set_id": (
                        f"RGCCA_K{rank}_L{penalty:g}"
                    ),
                    "latent_rank": rank,
                    "ridge_penalty": penalty,
                    "feature_budget_role": "primary",
                    "likelihoods": "not_applicable",
                    "native_missingness": False,
                    "inner_cv_required": True,
                    "notes": (
                        "Fold-local median imputation and PCA compression are required."
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_architecture_profiles() -> pd.DataFrame:
    rows = [
        {
            "architecture_profile": "compact",
            "rna_pca_components": 10,
            "methylation_pca_components": 10,
            "cnv_pca_components": 10,
            "shared_latent_dim": 5,
            "private_latent_dim_per_view": 2,
            "encoder_hidden_dim": 32,
            "decoder_hidden_dim": 32,
            "activation": "GELU",
            "layer_normalization": True,
            "full_batch_training": True,
            "maximum_epochs": 400,
            "early_stopping_patience": 30,
            "early_stopping_min_delta": 1e-4,
            "complexity_rank": 1,
        },
        {
            "architecture_profile": "standard",
            "rna_pca_components": 20,
            "methylation_pca_components": 20,
            "cnv_pca_components": 20,
            "shared_latent_dim": 10,
            "private_latent_dim_per_view": 5,
            "encoder_hidden_dim": 64,
            "decoder_hidden_dim": 64,
            "activation": "GELU",
            "layer_normalization": True,
            "full_batch_training": True,
            "maximum_epochs": 500,
            "early_stopping_patience": 40,
            "early_stopping_min_delta": 1e-4,
            "complexity_rank": 2,
        },
        {
            "architecture_profile": "expanded",
            "rna_pca_components": 30,
            "methylation_pca_components": 30,
            "cnv_pca_components": 30,
            "shared_latent_dim": 15,
            "private_latent_dim_per_view": 5,
            "encoder_hidden_dim": 96,
            "decoder_hidden_dim": 96,
            "activation": "GELU",
            "layer_normalization": True,
            "full_batch_training": True,
            "maximum_epochs": 600,
            "early_stopping_patience": 50,
            "early_stopping_min_delta": 1e-4,
            "complexity_rank": 3,
        },
    ]

    return pd.DataFrame(rows)


def build_regularization_profiles() -> pd.DataFrame:
    rows = [
        {
            "regularization_profile": "weak",
            "reconstruction_weight": 1.0,
            "shared_alignment_weight": 0.5,
            "contrastive_weight": 0.10,
            "shared_private_orthogonality_weight": 0.05,
            "private_separation_weight": 0.05,
            "cross_reconstruction_weight": 0.25,
            "modality_dropout_probability": 0.15,
            "contrastive_temperature": 0.20,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "complexity_rank": 1,
        },
        {
            "regularization_profile": "standard",
            "reconstruction_weight": 1.0,
            "shared_alignment_weight": 1.0,
            "contrastive_weight": 0.25,
            "shared_private_orthogonality_weight": 0.10,
            "private_separation_weight": 0.10,
            "cross_reconstruction_weight": 0.50,
            "modality_dropout_probability": 0.25,
            "contrastive_temperature": 0.10,
            "learning_rate": 5e-4,
            "weight_decay": 5e-4,
            "complexity_rank": 2,
        },
        {
            "regularization_profile": "strong",
            "reconstruction_weight": 1.0,
            "shared_alignment_weight": 2.0,
            "contrastive_weight": 0.50,
            "shared_private_orthogonality_weight": 0.20,
            "private_separation_weight": 0.20,
            "cross_reconstruction_weight": 1.00,
            "modality_dropout_probability": 0.35,
            "contrastive_temperature": 0.07,
            "learning_rate": 3e-4,
            "weight_decay": 1e-3,
            "complexity_rank": 3,
        },
    ]

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Simulation contract
# ---------------------------------------------------------------------


def build_simulation_scenarios() -> pd.DataFrame:
    base = {
        "n_samples": 75,
        "n_shared_factors": 5,
        "n_private_factors_per_view": 3,
        "n_features_rna": 500,
        "n_features_methylation": 4_000,
        "n_features_cnv": 500,
        "shared_signal_strength": 1.0,
        "private_signal_strength": 1.0,
        "noise_sd_rna": 1.0,
        "noise_sd_methylation": 1.0,
        "noise_sd_cnv": 1.0,
        "missing_fraction_rna": 0.0,
        "missing_fraction_methylation": 0.0,
        "missing_fraction_cnv": 0.0,
        "pair_corruption_fraction": 0.0,
        "uninformative_modality": "none",
        "block_scale_rna": 1.0,
        "block_scale_methylation": 1.0,
        "block_scale_cnv": 1.0,
        "replicates": 20,
    }

    specifications = [
        (
            "S01_BASE",
            "Base small-n, methylation-dominant feature count.",
            {},
        ),
        (
            "S02_SMALL_N50",
            "Smaller sample size.",
            {"n_samples": 50},
        ),
        (
            "S03_LARGER_N100",
            "Larger small-data benchmark.",
            {"n_samples": 100},
        ),
        (
            "S04_METH_MISSING10",
            "Ten percent methylation missingness.",
            {"missing_fraction_methylation": 0.10},
        ),
        (
            "S05_METH_MISSING20",
            "Twenty percent methylation missingness.",
            {"missing_fraction_methylation": 0.20},
        ),
        (
            "S06_ALL_VIEWS_MISSING10",
            "Ten percent missingness in every modality.",
            {
                "missing_fraction_rna": 0.10,
                "missing_fraction_methylation": 0.10,
                "missing_fraction_cnv": 0.10,
            },
        ),
        (
            "S07_WEAK_SHARED",
            "Weak shared signal with unchanged private signal.",
            {"shared_signal_strength": 0.50},
        ),
        (
            "S08_PRIVATE_DOMINANT",
            "Private factors dominate shared factors.",
            {
                "shared_signal_strength": 0.50,
                "private_signal_strength": 1.50,
            },
        ),
        (
            "S09_NOISY_METHYLATION",
            "Methylation view has doubled noise.",
            {"noise_sd_methylation": 2.0},
        ),
        (
            "S10_METHYLATION_SCALE_DOMINANCE",
            "Methylation values have fourfold block scale.",
            {"block_scale_methylation": 4.0},
        ),
        (
            "S11_PAIR_CORRUPTION10",
            "Ten percent patient-pair corruption.",
            {"pair_corruption_fraction": 0.10},
        ),
        (
            "S12_PAIR_CORRUPTION25",
            "Twenty-five percent patient-pair corruption.",
            {"pair_corruption_fraction": 0.25},
        ),
        (
            "S13_UNINFORMATIVE_CNV",
            "CNV carries noise but no shared or private signal.",
            {"uninformative_modality": "CNV"},
        ),
        (
            "S14_BALANCED_FEATURE_COUNTS",
            "All views have equal feature counts.",
            {
                "n_features_rna": 1_000,
                "n_features_methylation": 1_000,
                "n_features_cnv": 1_000,
            },
        ),
        (
            "S15_EXTREME_METHYLATION_DIMENSION",
            "Methylation has sixteenfold more features than each other view.",
            {"n_features_methylation": 8_000},
        ),
        (
            "S16_COMBINED_STRESS",
            "Small n, weak shared signal, missingness, noise, scale imbalance, and pairing error.",
            {
                "n_samples": 50,
                "shared_signal_strength": 0.50,
                "private_signal_strength": 1.50,
                "noise_sd_methylation": 2.0,
                "missing_fraction_methylation": 0.15,
                "pair_corruption_fraction": 0.10,
                "block_scale_methylation": 4.0,
            },
        ),
    ]

    rows: list[dict[str, Any]] = []

    for scenario_id, description, updates in specifications:
        record = {
            "scenario_id": scenario_id,
            "description": description,
            **base,
        }
        record.update(updates)
        rows.append(record)

    return pd.DataFrame(rows)


def build_simulation_replicates(
    scenarios: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for scenario_index, row in scenarios.iterrows():
        scenario_id = str(
            row["scenario_id"]
        )
        n_replicates = int(
            row["replicates"]
        )

        for replicate_index in range(
            1,
            n_replicates + 1,
        ):
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_order": int(
                        scenario_index + 1
                    ),
                    "replicate_index": replicate_index,
                    "simulation_seed": stable_seed(
                        SIMULATION_MASTER_SEED,
                        scenario_id,
                        replicate_index,
                    ),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Evaluation and inference registries
# ---------------------------------------------------------------------


def build_metric_registry() -> pd.DataFrame:
    rows = [
        {
            "metric_id": "heldout_normalized_reconstruction_error",
            "phase": "target_outer_cv",
            "direction": "lower",
            "primary_status": "primary",
            "formula": (
                "Mean across views of held-out MSE divided by the fold-local "
                "training-mean reconstruction MSE for that view."
            ),
            "aggregation": "mean across views, seeds, folds, and repeats",
            "outcome_used": False,
        },
        {
            "metric_id": "heldout_reconstruction_score",
            "phase": "target_inner_and_outer_cv",
            "direction": "higher",
            "primary_status": "primary_selection_component",
            "formula": (
                "clip(1 - heldout_normalized_reconstruction_error, 0, 1)"
            ),
            "aggregation": "mean across inner folds or outer folds",
            "outcome_used": False,
        },
        {
            "metric_id": "cross_view_top1_accuracy",
            "phase": "target_outer_cv",
            "direction": "higher",
            "primary_status": "secondary",
            "formula": (
                "Fraction of held-out patients whose true paired sample is the "
                "nearest sample in another view by cosine similarity."
            ),
            "aggregation": "mean across ordered view pairs",
            "outcome_used": False,
        },
        {
            "metric_id": "cross_view_mrr",
            "phase": "target_inner_and_outer_cv",
            "direction": "higher",
            "primary_status": "primary",
            "formula": (
                "Mean reciprocal rank of the true same-patient match across "
                "ordered pairs of modalities."
            ),
            "aggregation": "mean across ordered view pairs",
            "outcome_used": False,
        },
        {
            "metric_id": "random_adjusted_cross_view_mrr",
            "phase": "target_inner_and_outer_cv",
            "direction": "higher",
            "primary_status": "primary_selection_component",
            "formula": (
                "clip((MRR - H_n/n) / (1 - H_n/n), 0, 1), where H_n/n is "
                "the exact random-ranking expectation for n held-out samples."
            ),
            "aggregation": "mean across ordered view pairs",
            "outcome_used": False,
        },
        {
            "metric_id": "same_patient_similarity_margin",
            "phase": "target_outer_cv",
            "direction": "higher",
            "primary_status": "secondary",
            "formula": (
                "Median cosine similarity for true pairs minus median similarity "
                "for non-matching held-out pairs."
            ),
            "aggregation": "median across ordered view pairs",
            "outcome_used": False,
        },
        {
            "metric_id": "missing_view_r2",
            "phase": "target_inner_and_outer_cv",
            "direction": "higher",
            "primary_status": "primary",
            "formula": (
                "Held-out R2 for reconstructing one omitted view from the remaining "
                "view or views in fold-local compressed space."
            ),
            "aggregation": "mean across omitted views",
            "outcome_used": False,
        },
        {
            "metric_id": "missing_view_reconstruction_score",
            "phase": "target_inner_and_outer_cv",
            "direction": "higher",
            "primary_status": "primary_selection_component",
            "formula": "mean across views of clip(missing_view_r2, 0, 1)",
            "aggregation": "mean across omitted views",
            "outcome_used": False,
        },
        {
            "metric_id": "modality_contribution_entropy",
            "phase": "target_inner_and_outer_cv",
            "direction": "higher",
            "primary_status": "primary",
            "formula": (
                "Entropy of non-negative per-view loading or decoder energy, "
                "normalized by log(number_of_views)."
            ),
            "aggregation": "median across shared factors",
            "outcome_used": False,
        },
        {
            "metric_id": "modality_balance_score",
            "phase": "target_inner_and_outer_cv",
            "direction": "higher",
            "primary_status": "primary_selection_component",
            "formula": "clip(modality_contribution_entropy, 0, 1)",
            "aggregation": "median across shared factors",
            "outcome_used": False,
        },
        {
            "metric_id": "latent_effective_rank",
            "phase": "target_outer_cv",
            "direction": "descriptive",
            "primary_status": "failure_diagnostic",
            "formula": (
                "exp(entropy of normalized latent singular values squared)."
            ),
            "aggregation": "per fit and across seeds",
            "outcome_used": False,
        },
        {
            "metric_id": "seed_aligned_subspace_similarity",
            "phase": "target_outer_cv_and_final_stability",
            "direction": "higher",
            "primary_status": "stability",
            "formula": (
                "Mean canonical correlation after orthogonal Procrustes alignment "
                "of latent spaces fitted with different seeds."
            ),
            "aggregation": "median across seed pairs",
            "outcome_used": False,
        },
        {
            "metric_id": "simulation_shared_subspace_recovery",
            "phase": "simulation",
            "direction": "higher",
            "primary_status": "primary_simulation_metric",
            "formula": (
                "Mean canonical correlation between estimated and true shared "
                "latent subspaces after Procrustes alignment."
            ),
            "aggregation": "mean across simulation replicates and scenarios",
            "outcome_used": False,
        },
        {
            "metric_id": "simulation_private_subspace_recovery",
            "phase": "simulation",
            "direction": "higher",
            "primary_status": "secondary_simulation_metric",
            "formula": (
                "Mean canonical correlation between estimated and true private "
                "subspaces for each view."
            ),
            "aggregation": "mean across views and replicates",
            "outcome_used": False,
        },
        {
            "metric_id": "simulation_false_shared_discovery_rate",
            "phase": "simulation",
            "direction": "lower",
            "primary_status": "simulation_failure_gate",
            "formula": (
                "Fraction of retained shared factors not matched to a true shared "
                "factor above the prespecified correlation threshold."
            ),
            "aggregation": "mean across replicates",
            "outcome_used": False,
        },
    ]

    return pd.DataFrame(rows)


def build_failure_gates() -> pd.DataFrame:
    rows = [
        {
            "gate_id": "G01_NONFINITE_OPTIMIZATION",
            "applies_to": "all fitted models",
            "threshold": "any non-finite loss, latent value, loading, or reconstruction",
            "assessment_scope": "per fit",
            "consequence": "fit failure; exclude fit but retain audit record",
        },
        {
            "gate_id": "G02_LATENT_COLLAPSE",
            "applies_to": "integrated latent models",
            "threshold": (
                "shared latent effective rank < 2 or median shared-dimension "
                "variance < 1e-4"
            ),
            "assessment_scope": "per fit",
            "consequence": "model configuration fails that split",
        },
        {
            "gate_id": "G03_RECONSTRUCTION_WORSE_THAN_NULL",
            "applies_to": "all reconstructive models",
            "threshold": "mean held-out normalized reconstruction error >= 1.0",
            "assessment_scope": "aggregated outer CV",
            "consequence": "ineligible for final representation freeze",
        },
        {
            "gate_id": "G04_RETRIEVAL_NOT_ABOVE_RANDOM",
            "applies_to": "integrated multi-view models",
            "threshold": "random-adjusted cross-view MRR <= 0",
            "assessment_scope": "aggregated outer CV",
            "consequence": "ineligible for final representation freeze",
        },
        {
            "gate_id": "G05_SINGLE_VIEW_DOMINATION",
            "applies_to": "integrated shared representations",
            "threshold": "median normalized modality contribution entropy < 0.60",
            "assessment_scope": "aggregated outer CV",
            "consequence": "flag severe view domination; fail proposed-method claim",
        },
        {
            "gate_id": "G06_SEED_INSTABILITY",
            "applies_to": "stochastic models",
            "threshold": "median seed-aligned subspace similarity < 0.60",
            "assessment_scope": "aggregated outer CV",
            "consequence": "ineligible for final representation freeze",
        },
        {
            "gate_id": "G07_PAIRING_NULL_FAILURE",
            "applies_to": "selected integrated method",
            "threshold": (
                "observed cross-view retrieval does not exceed the 95th percentile "
                "of the pairing-permutation null"
            ),
            "assessment_scope": "post-selection outcome-blind permutation audit",
            "consequence": "no paired cross-view structure claim",
        },
        {
            "gate_id": "G08_NO_STABLE_FACTORS",
            "applies_to": "final selected representation",
            "threshold": "zero factors pass the primary stability gate",
            "assessment_scope": "final full-data stability analysis",
            "consequence": "do not proceed to factor-level biological or outcome claims",
        },
        {
            "gate_id": "G09_SIMULATION_FALSE_SHARED_DISCOVERY",
            "applies_to": "SABLE-MO architecture-profile selection",
            "threshold": "mean simulation false shared discovery rate > 0.10",
            "assessment_scope": "simulation benchmark",
            "consequence": "architecture/regularization profile is ineligible",
        },
    ]

    return pd.DataFrame(rows)


def build_random_control_registry() -> pd.DataFrame:
    rows = [
        {
            "control_id": "RC01_PAIRING_PERMUTATION",
            "phase": "representation_benchmark",
            "n_replicates": PAIRING_PERMUTATIONS,
            "unit_permuted": "patient pairing between modalities within training data",
            "preserved_structure": (
                "sample count, each view marginal distribution, feature identities, "
                "and outer held-out samples"
            ),
            "primary_metric": "cross_view_mrr;missing_view_r2",
            "outcome_used": False,
        },
        {
            "control_id": "RC02_MATCHED_RANDOM_GENE_PANELS",
            "phase": "biological_alignment",
            "n_replicates": 10_000,
            "unit_permuted": "gene or probe membership",
            "preserved_structure": (
                "panel size, modality, mean abundance/detection bin, and "
                "prestandardization variance bin"
            ),
            "primary_metric": "latent-program alignment statistic",
            "outcome_used": False,
        },
        {
            "control_id": "RC03_LATENT_LABEL_PERMUTATION",
            "phase": "biological_alignment",
            "n_replicates": 1_000,
            "unit_permuted": "sample labels of frozen latent factors",
            "preserved_structure": "factor marginal distribution",
            "primary_metric": "program alignment and annotation statistics",
            "outcome_used": False,
        },
    ]

    return pd.DataFrame(rows)


def build_multiplicity_registry() -> pd.DataFrame:
    rows = [
        {
            "family_id": "MF01_REPRESENTATION_PAIRING",
            "analysis_stage": "representation benchmark",
            "tests_grouped": (
                "pairing-permutation tests across prespecified integrated models "
                "and primary cross-view metrics"
            ),
            "adjustment": "Benjamini-Hochberg",
            "separate_from": "biological, external, and outcome families",
        },
        {
            "family_id": "MF02_PRIMARY_PROGRAM_ALIGNMENT",
            "analysis_stage": "biological alignment",
            "tests_grouped": (
                "all stable frozen factors by M34, M40, M11, and M24 in the "
                "primary TARGET representation"
            ),
            "adjustment": "Benjamini-Hochberg",
            "separate_from": "external-cohort and outcome families",
        },
        {
            "family_id": "MF03_EXTERNAL_COHORT_ALIGNMENT",
            "analysis_stage": "external validation",
            "tests_grouped": (
                "stable factor by frozen program tests within each external cohort"
            ),
            "adjustment": (
                "Benjamini-Hochberg separately within each prespecified cohort "
                "and assay family"
            ),
            "separate_from": "primary representation and outcome families",
        },
        {
            "family_id": "MF04_SPATIAL_ATAC_EXPLORATORY",
            "analysis_stage": "spatial and chromatin annotation",
            "tests_grouped": (
                "spatial-region and ATAC-support tests for frozen stable factors"
            ),
            "adjustment": "Benjamini-Hochberg within data type",
            "separate_from": "confirmatory bulk-expression families",
        },
        {
            "family_id": "MF05_OUTCOME_TRANSPORT",
            "analysis_stage": "outcome analysis after representation freeze",
            "tests_grouped": (
                "frozen stable factors by prespecified endpoint within each cohort"
            ),
            "adjustment": (
                "Benjamini-Hochberg within endpoint family; OS, DFI/metastasis, "
                "and RFS are not pooled"
            ),
            "separate_from": "all outcome-blind representation families",
        },
    ]

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Selection, stability, and external benchmark contracts
# ---------------------------------------------------------------------


def build_selection_policy() -> dict[str, Any]:
    return {
        "policy_version": "paper5-model-selection-policy-v1",
        "outcome_access": False,
        "biological_annotation_access": False,
        "simulation_profile_selection": {
            "candidate_architecture_profiles": [
                "compact",
                "standard",
                "expanded",
            ],
            "candidate_regularization_profiles": [
                "weak",
                "standard",
                "strong",
            ],
            "profile_selected_for": "V4_SABLE_MO",
            "profile_reused_by": [
                "V0_BALANCED_AE",
                "V1_SHARED_PRIVATE_AE",
                "V2_CONTRASTIVE_SPAE",
                "V3_DROPOUT_SPAE",
                "V5_SABLE_VAE",
            ],
            "primary_simulation_utility": {
                "simulation_shared_subspace_recovery": 0.40,
                "simulation_private_subspace_recovery": 0.20,
                "random_adjusted_cross_view_mrr": 0.20,
                "missing_view_reconstruction_score": 0.10,
                "modality_balance_score": 0.10,
            },
            "hard_gate": (
                "simulation_false_shared_discovery_rate <= 0.10 and no latent collapse"
            ),
            "selection_rule": (
                "Among eligible profiles, choose the lowest-complexity profile "
                "within one standard error of the highest mean simulation utility."
            ),
            "target_data_may_not_change_profile": True,
        },
        "target_nested_selection": {
            "outer_design": (
                f"{OUTER_REPEATS} repeats of {OUTER_FOLDS}-fold CV"
            ),
            "inner_design": (
                f"{INNER_FOLDS}-fold CV within each outer-training split"
            ),
            "eligible_model_source": (
                "model_family_registry.csv rows with "
                "eligible_for_nested_selection=true"
            ),
            "inner_model_seeds": INNER_MODEL_SEEDS,
            "outer_model_seeds": OUTER_MODEL_SEEDS,
            "primary_utility_components": PRIMARY_SELECTION_WEIGHTS,
            "component_formulas": {
                "heldout_reconstruction_score": (
                    "clip(1 - mean_view_normalized_MSE, 0, 1)"
                ),
                "random_adjusted_cross_view_mrr": (
                    "clip((MRR - H_n/n) / (1 - H_n/n), 0, 1)"
                ),
                "missing_view_reconstruction_score": (
                    "mean_view clip(missing_view_R2, 0, 1)"
                ),
                "modality_balance_score": (
                    "normalized entropy of view contribution energy"
                ),
            },
            "failure_gates_applied_before_selection": True,
            "one_standard_error_rule": (
                "Choose the lowest-complexity eligible model whose mean inner-CV "
                "utility is at least best_mean - standard_error_of_best."
            ),
            "tie_breaker": (
                "Lower complexity_rank, then lexicographically smaller model_id."
            ),
            "outer_test_samples_used_for_selection": False,
            "neural_early_stopping_rule": (
                "Each inner model-selection split has a deterministic fit, early-stop, "
                "and validation partition. The validation partition is never used for "
                "early stopping. After selecting an epoch on the early-stop subset, "
                "refit on fit plus early-stop samples for that fixed epoch before "
                "scoring the inner validation subset."
            ),
            "outer_test_evaluation": (
                "Each outer-training split has a deterministic fit and early-stop "
                "partition. Select the epoch without outer-test access, refit on all "
                "outer-training samples for that fixed epoch, then evaluate once on "
                "the untouched outer test fold."
            ),
        },
        "fixed_model_benchmark": {
            "all_prespecified_models_reported": True,
            "failed_models_retained_in_audit": True,
            "poor_models_may_be_summarized_in_supplement": True,
            "poor_models_may_not_be_silently_omitted": True,
        },
        "final_full_data_method_freeze": {
            "primary_rule": (
                "Choose the model_id selected most frequently by inner-CV one-SE "
                "selection across all outer splits."
            ),
            "tie_breaker": (
                "Lower complexity_rank, then lexicographically smaller model_id."
            ),
            "minimum_single_method_selection_frequency": 0.40,
            "uncertainty_rule": (
                "If no model reaches 40% selection frequency, do not claim a unique "
                "winner. Freeze the two most frequently selected non-failed methods "
                "as co-primary representations and report method uncertainty."
            ),
            "full_data_refit": (
                "Use the frozen method family, simulation-selected neural profile, "
                "primary feature budget, and all 78 cases without outcomes."
            ),
            "biological_or_outcome_results_may_not_change_method": True,
        },
    }


def build_external_benchmark_contract() -> dict[str, Any]:
    return {
        "contract_version": "paper5-external-method-benchmark-v1",
        "purpose": (
            "Determine whether the proposed method behaves sensibly outside "
            "the TARGET-OS osteosarcoma application."
        ),
        "status": "required_before_general_method_claim",
        "not_required_for": (
            "osteosarcoma-specific audited framework claim"
        ),
        "minimum_number_of_cohorts": 2,
        "eligibility_rule": {
            "required_modalities": [
                "RNA",
                "DNA methylation",
                "gene-level CNV",
            ],
            "minimum_complete_cases": 100,
            "public_or_reproducibly_accessible": True,
            "selected_without_outcomes": True,
            "selected_without_viewing_SABLE_MO_results": True,
            "osteosarcoma_excluded": True,
        },
        "small_n_downsampling": [
            50,
            75,
            100,
        ],
        "minimum_downsampling_repeats": 20,
        "primary_metrics": [
            "heldout_normalized_reconstruction_error",
            "cross_view_mrr",
            "missing_view_r2",
            "seed_aligned_subspace_similarity",
        ],
        "use_for_TARGET_model_selection": False,
        "outcome_access": False,
    }


def build_environment_snapshot() -> dict[str, Any]:
    distributions = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "torch",
        "mofapy2",
        "statsmodels",
    ]

    packages = {
        name: package_version(name)
        for name in distributions
    }

    return {
        "created_at_utc": utc_now_iso(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": packages,
        "readiness": {
            "linear_baselines_ready": all(
                packages[name]["installed"]
                for name in [
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                ]
            ),
            "mofa_ready": packages[
                "mofapy2"
            ]["installed"],
            "neural_models_ready": packages[
                "torch"
            ]["installed"],
        },
    }


# ---------------------------------------------------------------------
# Main contract construction
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the outcome-blind AI method, simulation, nested-CV, "
            "selection, failure-gate, and multiplicity contract for Paper 5."
        )
    )

    parser.add_argument(
        "--rewrite-contract",
        action="store_true",
        help=(
            "Archive and replace an existing contract. Requires "
            "--amendment-reason."
        ),
    )

    parser.add_argument(
        "--amendment-reason",
        type=str,
        default="",
        help=(
            "Scientific or technical reason for rewriting a frozen contract."
        ),
    )

    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help=(
            "Verify the existing frozen contract and exit without rewriting files."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if (
        args.rewrite_contract
        and args.verify_existing
    ):
        raise ValueError(
            "Use either --rewrite-contract or "
            "--verify-existing, not both."
        )

    if OUTPUT_LOCK.exists():
        if args.rewrite_contract:
            reason = clean_text(
                args.amendment_reason
            )

            if not reason:
                raise ValueError(
                    "--rewrite-contract requires a non-empty "
                    "--amendment-reason."
                )

            archived = archive_existing_contract(
                reason
            )

            print(
                "Archived previous contract: "
                f"{archived}"
            )
        else:
            verify_existing_contract()
            return
    elif args.verify_existing:
        raise FileNotFoundError(
            "No existing contract lock was found to verify."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 88)
    print(
        "Paper 5 - freeze AI method contract"
    )
    print("=" * 88)
    print(
        f"Script version: {SCRIPT_VERSION}"
    )
    print(
        f"Provisional method name: {METHOD_NAME}"
    )
    print(
        f"Project root: {PROJECT_ROOT}"
    )
    print(
        f"Output directory: {OUTPUT_DIR}"
    )
    print()

    case_ids = load_case_order()
    case_hash = hash_strings(
        case_ids
    )

    print(
        f"Locked cases: {len(case_ids)}"
    )
    print(
        f"Case-order hash: {case_hash}"
    )
    print()

    upstream_artifacts = validate_upstream_state(
        case_ids
    )

    print()
    print(
        "Upstream outcome-blind state: PASS"
    )
    print(
        "04b and 04c hashes: PASS"
    )
    print(
        "Post-standardization variance selection: FORBIDDEN"
    )

    (
        outer,
        inner,
        neural_assignments,
        split_summary,
    ) = build_nested_cv_splits(
        case_ids
    )

    model_registry = build_model_registry()
    ablation_registry = build_ablation_registry()
    baseline_grid = build_baseline_grid()
    architecture_profiles = (
        build_architecture_profiles()
    )
    regularization_profiles = (
        build_regularization_profiles()
    )
    simulation_scenarios = (
        build_simulation_scenarios()
    )
    simulation_replicates = (
        build_simulation_replicates(
            simulation_scenarios
        )
    )
    metric_registry = build_metric_registry()
    failure_gates = build_failure_gates()
    random_controls = (
        build_random_control_registry()
    )
    multiplicity = (
        build_multiplicity_registry()
    )
    selection_policy = (
        build_selection_policy()
    )
    external_benchmark = (
        build_external_benchmark_contract()
    )
    environment = (
        build_environment_snapshot()
    )

    write_table(
        outer,
        OUTPUT_OUTER_ASSIGNMENTS,
    )
    write_table(
        inner,
        OUTPUT_INNER_ASSIGNMENTS,
    )
    write_table(
        split_summary,
        OUTPUT_SPLIT_SUMMARY,
    )
    write_table(
        neural_assignments,
        OUTPUT_NEURAL_ASSIGNMENTS,
    )
    write_table(
        model_registry,
        OUTPUT_MODEL_REGISTRY,
    )
    write_table(
        ablation_registry,
        OUTPUT_ABLATION_REGISTRY,
    )
    write_table(
        baseline_grid,
        OUTPUT_BASELINE_GRID,
    )
    write_table(
        architecture_profiles,
        OUTPUT_ARCHITECTURE_PROFILES,
    )
    write_table(
        regularization_profiles,
        OUTPUT_REGULARIZATION_PROFILES,
    )
    write_table(
        simulation_scenarios,
        OUTPUT_SIMULATION_SCENARIOS,
    )
    write_table(
        simulation_replicates,
        OUTPUT_SIMULATION_REPLICATES,
    )
    write_table(
        metric_registry,
        OUTPUT_METRIC_REGISTRY,
    )
    write_table(
        failure_gates,
        OUTPUT_FAILURE_GATES,
    )
    write_table(
        random_controls,
        OUTPUT_RANDOM_CONTROLS,
    )
    write_table(
        multiplicity,
        OUTPUT_MULTIPLICITY,
    )
    write_json(
        selection_policy,
        OUTPUT_SELECTION_POLICY,
    )
    write_json(
        external_benchmark,
        OUTPUT_EXTERNAL_BENCHMARK,
    )
    write_json(
        environment,
        OUTPUT_ENVIRONMENT,
    )

    generated_paths = [
        OUTPUT_OUTER_ASSIGNMENTS,
        OUTPUT_INNER_ASSIGNMENTS,
        OUTPUT_SPLIT_SUMMARY,
        OUTPUT_NEURAL_ASSIGNMENTS,
        OUTPUT_MODEL_REGISTRY,
        OUTPUT_ABLATION_REGISTRY,
        OUTPUT_BASELINE_GRID,
        OUTPUT_ARCHITECTURE_PROFILES,
        OUTPUT_REGULARIZATION_PROFILES,
        OUTPUT_SIMULATION_SCENARIOS,
        OUTPUT_SIMULATION_REPLICATES,
        OUTPUT_METRIC_REGISTRY,
        OUTPUT_FAILURE_GATES,
        OUTPUT_RANDOM_CONTROLS,
        OUTPUT_MULTIPLICITY,
        OUTPUT_SELECTION_POLICY,
        OUTPUT_EXTERNAL_BENCHMARK,
        OUTPUT_ENVIRONMENT,
    ]

    contract = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "contract_status": "FROZEN_BEFORE_MODEL_FITTING",
        "method": {
            "provisional_name": METHOD_NAME,
            "name_status": METHOD_NAME_STATUS,
            "proposed_method_id": "V4_SABLE_MO",
            "proposed_method_automatically_declared_winner": False,
            "methodological_question": (
                "Can stability-aware, block-balanced, shared/private latent "
                "learning improve honest held-out multi-view reconstruction, "
                "paired cross-view coherence, missing-view recovery, and factor "
                "reproducibility under extreme p >> n?"
            ),
            "proposed_components": [
                "fold-local pre-standardization feature ranking",
                "block-balanced reconstruction",
                "shared/private latent decomposition",
                "paired contrastive alignment",
                "modality dropout",
                "missing-view reconstruction",
                "resampling-based factor alignment",
                "stability-gated factor retention",
            ],
        },
        "paper_scope": {
            "paper5_primary_scope": (
                "Outcome-blind small-n multi-omics representation learning, "
                "biological interpretation, and cross-platform recurrence."
            ),
            "outcomes_accessed_only_after_representation_freeze": True,
            "paper6_transfer_learning_is_out_of_scope": True,
            "GSE149679_individual_level_RNA_methylation_pairing_status": (
                "not established by the public GEO/COTC identifier audit; "
                "supplementary crosswalk remains unresolved"
            ),
        },
        "scientific_state": {
            "outcome_loaded": False,
            "clinical_labels_loaded": False,
            "biological_program_labels_used_for_model_selection": False,
            "model_fitting_performed": False,
            "simulation_results_viewed": False,
            "target_benchmark_results_viewed": False,
            "post_standardization_variance_used_for_selection": False,
            "global_preprocessing_allowed_in_heldout_evaluation": False,
        },
        "data_lock": {
            "n_cases": len(case_ids),
            "case_hash": case_hash,
            "modalities": [
                "RNA",
                "METHYLATION",
                "CNV",
            ],
            "feature_budgets": FEATURE_BUDGETS,
        },
        "nested_cv": {
            "master_seed": MASTER_SPLIT_SEED,
            "outer_repeats": OUTER_REPEATS,
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "early_stop_fraction": EARLY_STOP_FRACTION,
            "stratification": "none",
            "outcome_used_for_splits": False,
            "inner_model_seeds": INNER_MODEL_SEEDS,
            "outer_model_seeds": OUTER_MODEL_SEEDS,
            "outer_assignment_file": relative_path(
                OUTPUT_OUTER_ASSIGNMENTS
            ),
            "inner_assignment_file": relative_path(
                OUTPUT_INNER_ASSIGNMENTS
            ),
            "split_summary_file": relative_path(
                OUTPUT_SPLIT_SUMMARY
            ),
            "neural_training_assignment_file": relative_path(
                OUTPUT_NEURAL_ASSIGNMENTS
            ),
        },
        "preprocessing": {
            "heldout_rule": (
                "Missingness filtering, feature ranking, median imputation, "
                "scaling, block scaling, and PCA are fitted on training samples only."
            ),
            "primary_imputation": "training-fold feature median",
            "imputation_sensitivity": [
                "training-fold KNN k=5",
                "training-fold KNN k=10",
            ],
            "MOFA_missingness": "native probabilistic handling without imputation",
            "variance_ranking_scale": {
                "RNA": "log2 TPM before z-score",
                "METHYLATION": "M-value before z-score",
                "CNV": "log2(copy number / 2) before z-score",
            },
            "post_zscore_variance_selection_forbidden": True,
            "neural_raw_high_dimensional_input_forbidden": True,
            "neural_input": (
                "fold-local PCA-compressed per-view representations"
            ),
            "block_PCA_scaling": (
                "After fold-local standardization, multiply each view by the "
                "inverse square root of its total training variance before concatenation."
            ),
        },
        "model_registry": relative_path(
            OUTPUT_MODEL_REGISTRY
        ),
        "ablation_registry": relative_path(
            OUTPUT_ABLATION_REGISTRY
        ),
        "baseline_grid": relative_path(
            OUTPUT_BASELINE_GRID
        ),
        "simulation": {
            "master_seed": SIMULATION_MASTER_SEED,
            "scenario_file": relative_path(
                OUTPUT_SIMULATION_SCENARIOS
            ),
            "replicate_registry": relative_path(
                OUTPUT_SIMULATION_REPLICATES
            ),
            "architecture_profile_file": relative_path(
                OUTPUT_ARCHITECTURE_PROFILES
            ),
            "regularization_profile_file": relative_path(
                OUTPUT_REGULARIZATION_PROFILES
            ),
            "neural_profile_selected_using_target_data": False,
        },
        "evaluation": {
            "metric_registry": relative_path(
                OUTPUT_METRIC_REGISTRY
            ),
            "failure_gate_registry": relative_path(
                OUTPUT_FAILURE_GATES
            ),
            "selection_policy": relative_path(
                OUTPUT_SELECTION_POLICY
            ),
            "primary_selection_weights": PRIMARY_SELECTION_WEIGHTS,
            "pairing_permutations": PAIRING_PERMUTATIONS,
            "all_prespecified_models_reported": True,
            "silent_omission_of_failed_models_forbidden": True,
        },
        "stability_gate": {
            "final_resamples": FINAL_STABILITY_RESAMPLES,
            "subsample_fraction": FINAL_STABILITY_SUBSAMPLE_FRACTION,
            "full_data_seeds": FINAL_FULL_DATA_SEEDS,
            "factor_alignment": (
                "orthogonal Procrustes, Hungarian matching, and sign alignment"
            ),
            "thresholds": STABILITY_THRESHOLDS,
            "stability_evaluated_before_biology_and_outcomes": True,
        },
        "random_controls": relative_path(
            OUTPUT_RANDOM_CONTROLS
        ),
        "multiplicity_families": relative_path(
            OUTPUT_MULTIPLICITY
        ),
        "external_method_benchmark": relative_path(
            OUTPUT_EXTERNAL_BENCHMARK
        ),
        "environment_snapshot": relative_path(
            OUTPUT_ENVIRONMENT
        ),
        "upstream_artifacts": upstream_artifacts,
        "amendment_policy": {
            "silent_rewrite_forbidden": True,
            "rewrite_requires_archived_previous_contract": True,
            "rewrite_requires_documented_reason": True,
            "changes_after_viewing_biological_or_outcome_results_forbidden": True,
        },
        "guardrails": [
            "No outcomes or clinical labels are used in scripts 05a-08.",
            "No biological program annotation is used to select a model family.",
            "No full-cohort feature ranking is used in held-out evaluation.",
            "No post-z-score variance is used for feature selection.",
            "No raw 500k-dimensional vector is supplied directly to a neural encoder.",
            "Ordinary unregularized CCA is forbidden.",
            "Neural architecture profiles are selected on simulations, not TARGET results.",
            "Outer test samples are never used for preprocessing or model selection.",
            "Prespecified failed models remain visible in the audit trail.",
            "The proposed method is not guaranteed to be selected or declared superior.",
        ],
        "generated_artifacts": [
            relative_path(path)
            for path in generated_paths
        ],
    }

    write_json(
        contract,
        OUTPUT_CONTRACT,
    )

    generated_paths.append(
        OUTPUT_CONTRACT
    )

    readme_lines = [
        "Paper 5 AI method contract",
        "=" * 72,
        "",
        f"Script version: {SCRIPT_VERSION}",
        f"Created UTC: {contract['created_at_utc']}",
        f"Provisional method name: {METHOD_NAME}",
        "",
        "Scientific state",
        "----------------",
        "outcome_loaded: false",
        "clinical_labels_loaded: false",
        "model_fitting_performed: false",
        "biological labels used for selection: false",
        "post-standardization variance used for selection: false",
        "",
        "Frozen target evaluation",
        "------------------------",
        (
            f"{OUTER_REPEATS} repeats of {OUTER_FOLDS}-fold outer CV; "
            f"{INNER_FOLDS}-fold inner CV."
        ),
        "Splits are balanced and use no outcomes or clinical labels.",
        "Neural fits use a separate deterministic early-stopping subset.",
        "The metric-validation subset is never reused for early stopping.",
        "All data-dependent preprocessing is fitted within training folds.",
        "",
        "Model families",
        "--------------",
        "- modality-specific PCA",
        "- block-scaled PCA",
        "- MOFA+ rank sweep",
        "- ridge generalized CCA (secondary)",
        "- V0-V4 SABLE-MO ablation family",
        "- SABLE-VAE (secondary exploratory)",
        "",
        "Method-development rule",
        "-----------------------",
        (
            "Neural architecture and regularization profiles are selected on "
            "ground-truth simulations before TARGET benchmarking."
        ),
        (
            "TARGET inner CV uses an outcome-blind composite of held-out "
            "reconstruction, paired retrieval, missing-view recovery, and "
            "modality balance."
        ),
        (
            "The one-standard-error rule selects the simplest eligible model."
        ),
        "",
        "Transparency rule",
        "-----------------",
        (
            "All prespecified model families and failed variants remain in the "
            "benchmark audit. Poor results may be summarized in the supplement "
            "but may not be silently omitted."
        ),
        "",
        "Next scripts",
        "------------",
        "05b_generate_small_n_multiomics_simulations.py",
        "05c_fit_linear_mofa_baselines.py",
        "05d_fit_sable_mo_variants.py",
        "05e_compare_ablate_and_freeze_method.py",
        "",
    ]

    OUTPUT_README.write_text(
        "\n".join(readme_lines),
        encoding="utf-8",
    )

    generated_paths.append(
        OUTPUT_README
    )

    generated_file_records = {
        relative_path(path): {
            "sha256": sha256_file(path),
            "size_bytes": int(
                path.stat().st_size
            ),
        }
        for path in sorted(
            generated_paths,
            key=lambda item: str(item),
        )
    }

    contract_hash = sha256_file(
        OUTPUT_CONTRACT
    )

    lock = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "contract_status": "FROZEN_BEFORE_MODEL_FITTING",
        "contract_path": relative_path(
            OUTPUT_CONTRACT
        ),
        "contract_sha256": contract_hash,
        "case_hash": case_hash,
        "generated_files": generated_file_records,
        "upstream_artifacts": upstream_artifacts,
        "rewrite_policy": (
            "Run with --rewrite-contract and a documented --amendment-reason. "
            "The previous contract will be archived."
        ),
    }

    write_json(
        lock,
        OUTPUT_LOCK,
    )

    print()
    print("-" * 88)
    print(
        "Frozen nested-CV design"
    )
    print("-" * 88)
    print(
        split_summary[
            [
                "repeat_index",
                "outer_fold",
                "n_outer_train",
                "n_outer_test",
                "n_outer_fit_before_refit",
                "n_outer_early_stop",
                "minimum_inner_validation_size",
                "maximum_inner_validation_size",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("-" * 88)
    print(
        "Frozen model families"
    )
    print("-" * 88)
    print(
        model_registry[
            [
                "model_id",
                "role",
                "eligible_for_nested_selection",
                "eligible_for_final_freeze",
                "complexity_rank",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("-" * 88)
    print(
        "Environment readiness"
    )
    print("-" * 88)
    print(
        json.dumps(
            environment["readiness"],
            indent=2,
            sort_keys=True,
        )
    )

    if not environment[
        "readiness"
    ]["mofa_ready"]:
        print()
        print(
            "MOFA backend status: NOT INSTALLED"
        )
        print(
            "Install mofapy2 before script 05c."
        )

    if not environment[
        "readiness"
    ]["neural_models_ready"]:
        print()
        print(
            "PyTorch status: NOT INSTALLED"
        )
        print(
            "Install torch before script 05d."
        )

    print()
    print("=" * 88)
    print(
        "Paper 5 AI method contract: FROZEN"
    )
    print("=" * 88)
    print(
        f"Contract: {relative_path(OUTPUT_CONTRACT)}"
    )
    print(
        f"Contract SHA256: {contract_hash}"
    )
    print(
        f"Lock: {relative_path(OUTPUT_LOCK)}"
    )
    print(
        "Outcome access before representation freeze: FORBIDDEN"
    )
    print(
        "Silent model omission: FORBIDDEN"
    )
    print(
        "Silent contract rewrite: FORBIDDEN"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
