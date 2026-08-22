from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_VERSION = (
    "05a1-freeze-simulation-technical-confound-addendum-v3-no-cli"
)
ADDENDUM_VERSION = (
    "sable-mo-prospective-simulation-technical-addendum-v3-no-cli"
)
METHOD_NAME = "SABLE-MO"

# Fixed zero-argument execution policy.
# Existing data-only 05b v1 simulations are automatically subjected to
# a full SHA-256 audit, disclosed, and excluded from confirmatory analyses.
FULL_PILOT_HASH_AUDIT = True


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

BASE_CONTRACT_DIR = (
    PAPER5_ROOT
    / "method_contract"
    / "05a"
)

BASE_CONTRACT_FILE = (
    BASE_CONTRACT_DIR
    / "ai_method_contract.json"
)

BASE_LOCK_FILE = (
    BASE_CONTRACT_DIR
    / "ai_method_contract_lock.json"
)

BASE_SCENARIOS_FILE = (
    BASE_CONTRACT_DIR
    / "simulation_scenarios.csv"
)

BASE_REPLICATES_FILE = (
    BASE_CONTRACT_DIR
    / "simulation_replicate_registry.csv"
)

BASE_METRICS_FILE = (
    BASE_CONTRACT_DIR
    / "metric_registry.csv"
)

BASE_FAILURE_GATES_FILE = (
    BASE_CONTRACT_DIR
    / "failure_gate_registry.csv"
)

BASE_MULTIPLICITY_FILE = (
    BASE_CONTRACT_DIR
    / "multiplicity_family_registry.csv"
)

BASE_EXTERNAL_BENCHMARK_FILE = (
    BASE_CONTRACT_DIR
    / "external_method_benchmark_contract.json"
)

OUTPUT_DIR = (
    PAPER5_ROOT
    / "method_contract"
    / "05a1"
)

OUTPUT_ARCHITECTURE_SCENARIOS = (
    OUTPUT_DIR
    / "architecture_screening_addendum_scenarios.csv"
)

OUTPUT_ARCHITECTURE_REPLICATES = (
    OUTPUT_DIR
    / "architecture_screening_addendum_replicates.csv"
)

OUTPUT_GATE_SCENARIOS = (
    OUTPUT_DIR
    / "final_gate_calibration_scenarios.csv"
)

OUTPUT_GATE_REPLICATES = (
    OUTPUT_DIR
    / "final_gate_calibration_replicates.csv"
)

OUTPUT_COMBINED_ARCHITECTURE_SCENARIOS = (
    OUTPUT_DIR
    / "combined_architecture_screening_scenarios.csv"
)

OUTPUT_COMBINED_ARCHITECTURE_REPLICATES = (
    OUTPUT_DIR
    / "combined_architecture_screening_replicates.csv"
)

OUTPUT_TRUTH_TAXONOMY = (
    OUTPUT_DIR
    / "simulation_truth_taxonomy.csv"
)

OUTPUT_GATE_METRICS = (
    OUTPUT_DIR
    / "stability_gate_operating_metric_registry.csv"
)

OUTPUT_GATE_DECISIONS = (
    OUTPUT_DIR
    / "stability_gate_decision_registry.csv"
)

OUTPUT_TECHNICAL_COVARIATES = (
    OUTPUT_DIR
    / "target_technical_covariate_registry.csv"
)

OUTPUT_TECHNICAL_CONTRACT = (
    OUTPUT_DIR
    / "target_technical_metadata_preflight_contract.json"
)

OUTPUT_EXTERNAL_BENCHMARK_AMENDMENT = (
    OUTPUT_DIR
    / "external_method_benchmark_addendum.json"
)

OUTPUT_NOVELTY_CONTRACT = (
    OUTPUT_DIR
    / "novelty_positioning_contract.json"
)

OUTPUT_ENVIRONMENT = (
    OUTPUT_DIR
    / "addendum_environment_snapshot.json"
)

OUTPUT_ADDENDUM = (
    OUTPUT_DIR
    / "simulation_and_technical_confound_addendum.json"
)

OUTPUT_README = (
    OUTPUT_DIR
    / "simulation_and_technical_confound_addendum_README.txt"
)

OUTPUT_LOCK = (
    OUTPUT_DIR
    / "simulation_and_technical_confound_addendum_lock.json"
)

PILOT_05B_DIR = (
    PAPER5_ROOT
    / "method_contract"
    / "05b"
)

PILOT_05B_LOCK = (
    PILOT_05B_DIR
    / "simulation_generation_lock.json"
)

PILOT_05B_MANIFEST = (
    PILOT_05B_DIR
    / "simulation_data_manifest.csv"
)

PILOT_SIMULATION_ROOT = (
    PAPER5_ROOT
    / "simulations"
    / "sable_mo_v1"
)

OUTPUT_PILOT_DISCLOSURE = (
    OUTPUT_DIR
    / "pre_addendum_05b_pilot_materialization_disclosure.json"
)


# ---------------------------------------------------------------------
# Frozen addendum constants
# ---------------------------------------------------------------------

ADDENDUM_MASTER_SEED = 20260814
GENERATOR_REQUIRED_VERSION = "sable-mo-simulation-generator-v2"

VIEWS = (
    "RNA",
    "METHYLATION",
    "CNV",
)

BASE_FEATURE_COUNTS = {
    "RNA": 500,
    "METHYLATION": 4_000,
    "CNV": 500,
}

BASE_SCENARIO_DEFAULTS: dict[str, Any] = {
    "n_samples": 62,
    "n_shared_factors": 4,
    "n_private_factors_per_view": 2,
    "n_features_rna": BASE_FEATURE_COUNTS["RNA"],
    "n_features_methylation": BASE_FEATURE_COUNTS[
        "METHYLATION"
    ],
    "n_features_cnv": BASE_FEATURE_COUNTS["CNV"],
    "shared_signal_strength": 1.0,
    "private_signal_strength": 1.0,
    "noise_sd_rna": 1.0,
    "noise_sd_methylation": 1.0,
    "noise_sd_cnv": 1.0,
    "missing_fraction_rna": 0.0,
    "missing_fraction_methylation": 0.0,
    "missing_fraction_cnv": 0.0,
    "pair_corruption_fraction": 0.0,
    "uninformative_modality": "NONE",
    "block_scale_rna": 1.0,
    "block_scale_methylation": 1.0,
    "block_scale_cnv": 1.0,
    "nonlinear_mode": "none",
    "shared_factor_correlation": 0.0,
    "shared_strength_pattern": "equal",
    "shared_support_pattern": "ALL3",
    "shared_support_by_factor": (
        "ALL3|ALL3|ALL3|ALL3"
    ),
    "private_cross_view_correlation": 0.0,
    "correlated_private_mode": "none",
    "shared_loading_fraction": 0.10,
    "private_loading_fraction": 0.10,
    "loading_regime": "moderate_10pct",
    "missingness_mode": "none",
    "missingness_target_view": "NONE",
    "low_quality_sample_fraction": 0.0,
    "low_quality_feature_fraction": 0.0,
    "low_quality_missing_fraction": 0.0,
    "partial_view_sample_fraction": 0.0,
    "partial_view_missing_fraction": 0.0,
    "batch_mode": "none",
    "batch_cardinality": 2,
    "batch_prevalence": 0.50,
    "batch_effect_strength": 0.0,
    "batch_view_support": "NONE",
    "batch_signal_correlation": 0.0,
    "batch_missingness_increment": 0.0,
    "true_biological_shared_factors": 4,
    "true_technical_shared_factors": 0,
    "factor_level_identifiable": True,
    "subspace_level_identifiable": True,
    "primary_truth_target": "factor_and_subspace",
    "architecture_screening_replicates": 20,
    "stability_resamples_per_dataset": 20,
    "materialization_policy": "persist_npz",
    "generator_required_version": (
        GENERATOR_REQUIRED_VERSION
    ),
}

ARCHITECTURE_SCREENING_REPLICATES = 20
ARCHITECTURE_STABILITY_RESAMPLES = 20
FINAL_GATE_STABILITY_RESAMPLES = 100
FINAL_GATE_NON_NULL_REPLICATES = 100
FINAL_GATE_NULL_REPLICATES = 200

TRUTH_FACTOR_MATCH_ABS_CORRELATION = 0.50
TRUTH_BATCH_MATCH_ABS_CORRELATION = 0.50

OPERATING_TARGETS = {
    "null_familywise_false_acceptance_max": 0.10,
    "standard_n62_signal1_tpr_min": 0.80,
    "standard_n62_signal1_fnr_max": 0.20,
    "batch_null_false_acceptance_max": 0.10,
    "rotational_n62_signal1_subspace_tpr_min": 0.80,
    "view_support_classification_accuracy_min": 0.80,
}

TECHNICAL_ASSOCIATION_THRESHOLDS = {
    "bh_q_value": 0.05,
    "absolute_spearman_rho": 0.30,
    "eta_squared": 0.10,
    "standardized_mean_difference": 0.50,
    "minimum_level_size": 5,
    "maximum_missing_fraction": 0.30,
    "maximum_unique_fraction": 0.50,
    "maximum_categorical_levels": 10,
    "permutations": 10_000,
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
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )


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
        return str(
            path.resolve()
        )


def resolve_project_path(value: Any) -> Path:
    path = Path(
        clean_text(value)
    )

    if not path.is_absolute():
        path = (
            PROJECT_ROOT
            / path
        )

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


def stable_seed(
    master_seed: int,
    *parts: Any,
) -> int:
    digest = hashlib.sha256()
    digest.update(
        str(master_seed).encode(
            "utf-8"
        )
    )

    for part in parts:
        digest.update(b"|")
        digest.update(
            str(part).encode(
                "utf-8"
            )
        )

    return int(
        int.from_bytes(
            digest.digest()[:8],
            byteorder="big",
            signed=False,
        )
        % (2**32 - 1)
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
        return {
            "installed": True,
            "version": (
                importlib.metadata.version(
                    distribution_name
                )
            ),
        }
    except importlib.metadata.PackageNotFoundError:
        return {
            "installed": False,
            "version": None,
        }


# ---------------------------------------------------------------------
# Base-contract verification
# ---------------------------------------------------------------------


def validate_base_contract() -> dict[str, Any]:
    required = [
        BASE_CONTRACT_FILE,
        BASE_LOCK_FILE,
        BASE_SCENARIOS_FILE,
        BASE_REPLICATES_FILE,
        BASE_METRICS_FILE,
        BASE_FAILURE_GATES_FILE,
        BASE_MULTIPLICITY_FILE,
        BASE_EXTERNAL_BENCHMARK_FILE,
    ]

    for path in required:
        require_file(path)

    contract = json.loads(
        BASE_CONTRACT_FILE.read_text(
            encoding="utf-8"
        )
    )

    lock = json.loads(
        BASE_LOCK_FILE.read_text(
            encoding="utf-8"
        )
    )

    observed_contract_hash = sha256_file(
        BASE_CONTRACT_FILE
    )

    expected_contract_hash = clean_text(
        lock.get(
            "contract_sha256"
        )
    )

    if (
        observed_contract_hash
        != expected_contract_hash
    ):
        raise RuntimeError(
            "The 05a contract does not match "
            "its frozen lock."
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
                f"missing: {relative_name}"
            )
            continue

        observed_hash = sha256_file(
            path
        )

        expected_hash = clean_text(
            record.get("sha256")
        )

        if observed_hash != expected_hash:
            failures.append(
                f"hash mismatch: {relative_name}"
            )

    if failures:
        raise RuntimeError(
            "05a generated-file verification failed:\n"
            + "\n".join(
                f"- {value}"
                for value in failures
            )
        )

    state = contract.get(
        "scientific_state",
        {},
    )

    scope = contract.get(
        "paper_scope",
        {},
    )

    if bool(
        state.get(
            "outcome_loaded",
            True,
        )
    ):
        raise RuntimeError(
            "05a does not confirm "
            "outcome_loaded=false."
        )

    if bool(
        state.get(
            "model_fitting_performed",
            True,
        )
    ):
        raise RuntimeError(
            "05a does not confirm "
            "model_fitting_performed=false."
        )

    if not bool(
        scope.get(
            "outcomes_accessed_only_after_representation_freeze",
            False,
        )
    ):
        raise RuntimeError(
            "05a does not keep outcomes inaccessible "
            "until representation freeze."
        )

    if clean_text(
        contract.get("contract_status")
    ) != "FROZEN_BEFORE_MODEL_FITTING":
        raise RuntimeError(
            "Unexpected 05a contract status."
        )

    base_scenarios = pd.read_csv(
        BASE_SCENARIOS_FILE,
        dtype=str,
    ).fillna("")

    base_replicates = pd.read_csv(
        BASE_REPLICATES_FILE,
        dtype=str,
    ).fillna("")

    if base_scenarios.empty:
        raise RuntimeError(
            "The 05a base scenario registry is empty."
        )

    if base_scenarios[
        "scenario_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate scenario IDs exist in 05a."
        )

    if base_replicates.empty:
        raise RuntimeError(
            "The 05a base replicate registry is empty."
        )

    return {
        "contract": contract,
        "lock": lock,
        "contract_sha256": (
            observed_contract_hash
        ),
        "lock_sha256": sha256_file(
            BASE_LOCK_FILE
        ),
        "base_scenarios": base_scenarios,
        "base_replicates": base_replicates,
        "verified_files": {
            relative_path(path): {
                "sha256": sha256_file(path),
                "size_bytes": int(
                    path.stat().st_size
                ),
            }
            for path in required
        },
    }


def find_preexisting_model_results() -> list[str]:
    """Find fitted-model or benchmark outputs that invalidate a prospective addendum."""
    failures: list[str] = []

    candidate_paths = [
        PAPER5_ROOT / "method_contract" / "05c",
        PAPER5_ROOT / "method_contract" / "05d",
        PAPER5_ROOT / "method_contract" / "05e",
        PAPER5_ROOT / "results" / "representation_benchmark",
        PAPER5_ROOT / "results" / "sable_mo",
        PAPER5_ROOT / "results" / "simulation_benchmark",
    ]

    for path in candidate_paths:
        if path.is_file():
            if path.stat().st_size > 0:
                failures.append(relative_path(path))
        elif path.is_dir():
            if any(item.is_file() for item in path.rglob("*")):
                failures.append(relative_path(path))

    return failures


def preexisting_05b_pilot_present() -> bool:
    if PILOT_05B_LOCK.exists() or PILOT_05B_MANIFEST.exists():
        return True
    if PILOT_SIMULATION_ROOT.exists():
        return any(
            path.is_file()
            and path.suffix.lower() in {".npz", ".npy", ".parquet"}
            for path in PILOT_SIMULATION_ROOT.rglob("*")
        )
    return False


def validate_existing_05b_pilot(
    *,
    full_hash_audit: bool,
) -> dict[str, Any]:
    """
    Validate a data-only 05b v1 materialization.

    This is allowed only when no representation or simulation model was fitted.
    The materialized v1 datasets are disclosed as developmental pilot inputs and
    are excluded from confirmatory SABLE-MO selection and final gate calibration.
    """
    require_file(PILOT_05B_LOCK)
    require_file(PILOT_05B_MANIFEST)

    lock = json.loads(
        PILOT_05B_LOCK.read_text(encoding="utf-8")
    )

    failures: list[str] = []

    if bool(lock.get("outcome_access", True)):
        failures.append("05b v1 lock does not confirm outcome_access=false.")
    if bool(lock.get("model_fitting_performed", True)):
        failures.append(
            "05b v1 lock indicates that model fitting was performed."
        )
    if bool(lock.get("stability_gate_thresholds_retuned", True)):
        failures.append(
            "05b v1 lock indicates stability-gate threshold retuning."
        )

    expected_manifest_hash = clean_text(
        lock.get("data_manifest_sha256")
    )
    observed_manifest_hash = sha256_file(
        PILOT_05B_MANIFEST
    )
    if observed_manifest_hash != expected_manifest_hash:
        failures.append("05b v1 data-manifest SHA256 mismatch.")

    manifest = pd.read_csv(
        PILOT_05B_MANIFEST,
        dtype=str,
        low_memory=False,
    ).fillna("")

    required_columns = {
        "scenario_id",
        "scenario_family",
        "path",
        "size_bytes",
        "sha256",
    }
    missing_columns = required_columns - set(manifest.columns)
    if missing_columns:
        failures.append(
            "05b v1 manifest is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    if failures:
        raise RuntimeError(
            "Existing 05b pilot materialization failed contract validation:\n"
            + "\n".join(f"- {value}" for value in failures)
        )

    expected_n = int(lock.get("n_datasets", -1))
    if manifest.shape[0] != expected_n:
        failures.append(
            f"05b v1 manifest row count {manifest.shape[0]} "
            f"does not match lock n_datasets={expected_n}."
        )

    if manifest["path"].duplicated().any():
        failures.append("Duplicate dataset paths exist in the 05b v1 manifest.")

    manifest_paths: set[str] = set()
    total_size = 0
    hash_checked = 0

    print()
    print("Validating pre-addendum 05b v1 pilot materialization...")

    for row_index, row in manifest.iterrows():
        raw_path = clean_text(row["path"])
        path = Path(raw_path)
        if not path.is_absolute():
            normalized_parts = [
                part
                for part in raw_path.replace("\\", "/").split("/")
                if part
            ]
            path = PROJECT_ROOT.joinpath(*normalized_parts)
        manifest_paths.add(str(path.resolve()))

        if not path.exists():
            failures.append(
                f"Missing pilot simulation file: {relative_path(path)}"
            )
            continue

        expected_size = int(float(row["size_bytes"]))
        observed_size = int(path.stat().st_size)
        total_size += observed_size

        if observed_size != expected_size:
            failures.append(
                f"Pilot simulation size mismatch: {relative_path(path)} "
                f"({observed_size} vs {expected_size})"
            )
            continue

        if full_hash_audit:
            observed_hash = sha256_file(path)
            hash_checked += 1
            if observed_hash != clean_text(row["sha256"]):
                failures.append(
                    f"Pilot simulation SHA256 mismatch: {relative_path(path)}"
                )

        current = row_index + 1
        if current == 1 or current % 50 == 0 or current == manifest.shape[0]:
            suffix = " + SHA256" if full_hash_audit else ""
            print(
                f" {current:04d}/{manifest.shape[0]:04d} "
                f"existence/size{suffix}: PASS"
            )

    all_materialized_files: list[Path] = []
    if PILOT_SIMULATION_ROOT.exists():
        all_materialized_files = [
            path
            for path in PILOT_SIMULATION_ROOT.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".npz", ".npy", ".parquet"}
        ]

    extra_files = sorted(
        [
            path
            for path in all_materialized_files
            if str(path.resolve()) not in manifest_paths
        ],
        key=lambda value: str(value),
    )

    if failures:
        raise RuntimeError(
            "Existing 05b pilot materialization failed file validation:\n"
            + "\n".join(f"- {value}" for value in failures[:50])
        )

    family_counts = {
        str(key): int(value)
        for key, value in manifest.groupby("scenario_family").size().to_dict().items()
    }

    return {
        "status": "VALIDATED_DATA_ONLY_PILOT_MATERIALIZATION",
        "materialized_before_05a1": True,
        "prospective_before_simulation_materialization": False,
        "prospective_before_model_fitting": True,
        "eligible_for_primary_model_selection": False,
        "eligible_for_final_gate_calibration": False,
        "allowed_uses": [
            "generator regression testing",
            "developmental pipeline smoke testing",
            "non-confirmatory descriptive QA",
        ],
        "forbidden_uses": [
            "primary architecture selection",
            "confirmatory method comparison",
            "final stability-gate operating-characteristic estimation",
            "headline manuscript performance claims",
        ],
        "lock_path": relative_path(PILOT_05B_LOCK),
        "lock_sha256": sha256_file(PILOT_05B_LOCK),
        "manifest_path": relative_path(PILOT_05B_MANIFEST),
        "manifest_sha256": observed_manifest_hash,
        "n_locked_manifest_datasets": int(manifest.shape[0]),
        "n_all_materialized_files": int(len(all_materialized_files)),
        "n_unlocked_extra_files": int(len(extra_files)),
        "extra_files": [relative_path(path) for path in extra_files],
        "scenario_family_counts": family_counts,
        "locked_total_size_bytes": int(total_size),
        "full_file_hash_audit_performed": bool(full_hash_audit),
        "n_dataset_hashes_recomputed": int(hash_checked),
        "outcome_access": bool(lock.get("outcome_access", False)),
        "model_fitting_performed": bool(lock.get("model_fitting_performed", False)),
        "stability_gate_thresholds_retuned": bool(
            lock.get("stability_gate_thresholds_retuned", False)
        ),
        "interpretation": (
            "Simulation arrays were materialized before the expanded 05a1 design "
            "was frozen, but no simulation model, TARGET representation model, "
            "biological analysis, or outcome analysis had been fitted. The v1 "
            "datasets are therefore retained as an auditable developmental pilot "
            "and excluded from confirmatory evaluation."
        ),
    }


def find_preexisting_simulation_or_model_results() -> list[str]:
    """Backward-compatible combined finder used by rewrite protection."""
    failures = find_preexisting_model_results()
    if preexisting_05b_pilot_present():
        failures.append(
            "A pre-existing 05b pilot materialization is present."
        )
    return failures


# ---------------------------------------------------------------------
# Existing-addendum protection
# ---------------------------------------------------------------------


def verify_existing_addendum() -> None:
    require_file(OUTPUT_LOCK)
    require_file(OUTPUT_ADDENDUM)

    lock = json.loads(
        OUTPUT_LOCK.read_text(
            encoding="utf-8"
        )
    )

    addendum = json.loads(
        OUTPUT_ADDENDUM.read_text(
            encoding="utf-8"
        )
    )

    failures: list[str] = []

    expected_addendum_hash = clean_text(
        lock.get("addendum_sha256")
    )

    observed_addendum_hash = sha256_file(
        OUTPUT_ADDENDUM
    )

    if (
        observed_addendum_hash
        != expected_addendum_hash
    ):
        failures.append(
            "Addendum JSON hash mismatch."
        )

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

        if sha256_file(path) != clean_text(
            record.get("sha256")
        ):
            failures.append(
                f"Generated-file hash mismatch: {relative_name}"
            )

    base_reference = addendum.get(
        "base_05a_reference",
        {},
    )

    if sha256_file(
        BASE_CONTRACT_FILE
    ) != clean_text(
        base_reference.get(
            "contract_sha256"
        )
    ):
        failures.append(
            "The frozen 05a contract changed after 05a1."
        )

    if sha256_file(
        BASE_LOCK_FILE
    ) != clean_text(
        base_reference.get(
            "lock_sha256"
        )
    ):
        failures.append(
            "The frozen 05a lock changed after 05a1."
        )

    if failures:
        raise RuntimeError(
            "Existing 05a1 addendum verification failed:\n"
            + "\n".join(
                f"- {value}"
                for value in failures
            )
        )

    print("=" * 88)
    print(
        "Existing Paper 5 simulation and technical "
        "addendum: PASS"
    )
    print("=" * 88)
    print(
        f"Addendum: {relative_path(OUTPUT_ADDENDUM)}"
    )
    print(
        f"Lock: {relative_path(OUTPUT_LOCK)}"
    )
    print("=" * 88)


# ---------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------


def support_string(
    supports: Iterable[str],
) -> str:
    return "|".join(
        str(value)
        for value in supports
    )


def make_scenario(
    scenario_id: str,
    description: str,
    scenario_family: str,
    calibration_role: str,
    updates: dict[str, Any] | None = None,
    *,
    stage: str = "architecture_screening",
    planned_replicates: int = (
        ARCHITECTURE_SCREENING_REPLICATES
    ),
    stability_resamples: int = (
        ARCHITECTURE_STABILITY_RESAMPLES
    ),
    materialization_policy: str = "persist_npz",
) -> dict[str, Any]:
    record = {
        "scenario_id": scenario_id,
        "description": description,
        "scenario_origin": "05a1_addendum",
        "scenario_family": scenario_family,
        "calibration_role": calibration_role,
        "simulation_stage": stage,
        **BASE_SCENARIO_DEFAULTS,
    }

    if updates:
        record.update(updates)

    record[
        "architecture_screening_replicates"
    ] = planned_replicates

    record[
        "planned_replicates"
    ] = planned_replicates

    record[
        "stability_resamples_per_dataset"
    ] = stability_resamples

    record[
        "materialization_policy"
    ] = materialization_policy

    record[
        "generator_required_version"
    ] = GENERATOR_REQUIRED_VERSION

    return record


def build_architecture_screening_scenarios() -> pd.DataFrame:
    scenarios: list[dict[str, Any]] = []

    scenarios.append(
        make_scenario(
            "A01_N62_STANDARD",
            (
                "Realistic TARGET outer-training sample size "
                "with independent fully shared factors."
            ),
            "operating_point",
            "architecture_screening",
        )
    )

    scenarios.append(
        make_scenario(
            "A02_SHARED_RHO05",
            (
                "Moderately correlated fully shared factors "
                "with factor-level identifiability retained."
            ),
            "correlated_shared_factors",
            "factor_and_subspace_screening",
            {
                "shared_factor_correlation": 0.50,
            },
        )
    )

    scenarios.append(
        make_scenario(
            "A03_ROTATIONAL_RHO08",
            (
                "Equal-strength, highly correlated shared factors "
                "with rotational ambiguity."
            ),
            "rotational_ambiguity",
            "subspace_recovery_screening",
            {
                "n_shared_factors": 2,
                "shared_factor_correlation": 0.80,
                "shared_strength_pattern": (
                    "equal_near_degenerate"
                ),
                "shared_support_by_factor": (
                    "ALL3|ALL3"
                ),
                "true_biological_shared_factors": 2,
                "factor_level_identifiable": False,
                "subspace_level_identifiable": True,
                "primary_truth_target": "subspace",
            },
        )
    )

    partial_specs = [
        (
            "A04_PARTIAL_RNA_METH",
            "RNA_METH",
        ),
        (
            "A05_PARTIAL_RNA_CNV",
            "RNA_CNV",
        ),
        (
            "A06_PARTIAL_METH_CNV",
            "METH_CNV",
        ),
    ]

    for scenario_id, support in partial_specs:
        scenarios.append(
            make_scenario(
                scenario_id,
                (
                    "True biological factors shared by exactly "
                    f"two views: {support}."
                ),
                "partial_view_sharing",
                "view_support_screening",
                {
                    "n_shared_factors": 2,
                    "shared_support_pattern": support,
                    "shared_support_by_factor": (
                        f"{support}|{support}"
                    ),
                    "true_biological_shared_factors": 2,
                },
            )
        )

    scenarios.append(
        make_scenario(
            "A07_MIXED_VIEW_SUPPORT",
            (
                "Mixed truth with one all-view factor and one "
                "factor for each pair of views."
            ),
            "partial_view_sharing",
            "view_support_screening",
            {
                "n_shared_factors": 4,
                "shared_support_pattern": "MIXED",
                "shared_support_by_factor": (
                    "ALL3|RNA_METH|RNA_CNV|METH_CNV"
                ),
                "true_biological_shared_factors": 4,
            },
        )
    )

    scenarios.append(
        make_scenario(
            "A08_CORRELATED_PRIVATE_NULL",
            (
                "No true shared factor; private factors across "
                "views are correlated and may mimic sharing."
            ),
            "correlated_private_null",
            "false_shared_screening",
            {
                "n_shared_factors": 0,
                "shared_signal_strength": 0.0,
                "shared_support_by_factor": "",
                "private_cross_view_correlation": 0.60,
                "correlated_private_mode": (
                    "correlated_private_scores"
                ),
                "true_biological_shared_factors": 0,
                "factor_level_identifiable": False,
                "subspace_level_identifiable": False,
                "primary_truth_target": "null_specificity",
            },
        )
    )

    scenarios.append(
        make_scenario(
            "A09_FEATURE_STRUCTURED_MISSING",
            (
                "A subset of methylation features has elevated "
                "missingness rather than cell-wise MCAR."
            ),
            "structured_missingness",
            "missingness_screening",
            {
                "missingness_mode": "feature_structured",
                "missingness_target_view": "METHYLATION",
                "low_quality_feature_fraction": 0.15,
                "low_quality_missing_fraction": 0.30,
            },
        )
    )

    scenarios.append(
        make_scenario(
            "A10_SAMPLE_QUALITY_MISSING",
            (
                "A subset of samples has multi-view low-quality "
                "profiles with structured missingness."
            ),
            "structured_missingness",
            "missingness_screening",
            {
                "missingness_mode": "sample_quality",
                "missingness_target_view": "ALL3",
                "low_quality_sample_fraction": 0.20,
                "low_quality_missing_fraction": 0.20,
            },
        )
    )

    scenarios.append(
        make_scenario(
            "A11_PARTIAL_VIEW_MISSING",
            (
                "A subset of samples has a substantially missing "
                "methylation view."
            ),
            "structured_missingness",
            "missing_view_screening",
            {
                "missingness_mode": "partial_view",
                "missingness_target_view": "METHYLATION",
                "partial_view_sample_fraction": 0.20,
                "partial_view_missing_fraction": 0.50,
            },
        )
    )

    scenarios.append(
        make_scenario(
            "A12_BATCH_DEPENDENT_MISSING",
            (
                "Methylation missingness is increased within one "
                "technical batch."
            ),
            "structured_missingness",
            "technical_missingness_screening",
            {
                "missingness_mode": "batch_dependent",
                "missingness_target_view": "METHYLATION",
                "batch_mode": "shared_all_views",
                "batch_view_support": "ALL3",
                "batch_effect_strength": 0.75,
                "batch_missingness_increment": 0.15,
                "true_technical_shared_factors": 1,
            },
        )
    )

    scenarios.append(
        make_scenario(
            "A13_SPARSE_LOADING_2PCT",
            (
                "Sparse biological programs with strong loadings "
                "on two percent of features."
            ),
            "loading_density",
            "sparse_loading_screening",
            {
                "shared_loading_fraction": 0.02,
                "private_loading_fraction": 0.02,
                "loading_regime": "sparse_2pct",
                "shared_signal_strength": 1.25,
            },
        )
    )

    scenarios.append(
        make_scenario(
            "A14_DENSE_LOADING_40PCT",
            (
                "Dense biological programs with weaker loadings "
                "on forty percent of features."
            ),
            "loading_density",
            "dense_loading_screening",
            {
                "shared_loading_fraction": 0.40,
                "private_loading_fraction": 0.40,
                "loading_regime": "dense_40pct",
                "shared_signal_strength": 0.75,
            },
        )
    )

    scenarios.append(
        make_scenario(
            "A15_BATCH_SHARED_NULL",
            (
                "No biological shared factor; a reproducible "
                "technical batch affects all views."
            ),
            "technical_batch_null",
            "technical_false_acceptance_screening",
            {
                "n_shared_factors": 0,
                "shared_signal_strength": 0.0,
                "shared_support_by_factor": "",
                "batch_mode": "shared_all_views",
                "batch_view_support": "ALL3",
                "batch_effect_strength": 1.00,
                "true_biological_shared_factors": 0,
                "true_technical_shared_factors": 1,
                "factor_level_identifiable": False,
                "subspace_level_identifiable": False,
                "primary_truth_target": (
                    "technical_null_specificity"
                ),
            },
        )
    )

    scenarios.append(
        make_scenario(
            "A16_BATCH_PARTIAL_NULL",
            (
                "No biological shared factor; a technical batch "
                "affects RNA and methylation only."
            ),
            "technical_batch_null",
            "partial_technical_false_acceptance_screening",
            {
                "n_shared_factors": 0,
                "shared_signal_strength": 0.0,
                "shared_support_by_factor": "",
                "batch_mode": "partial_two_views",
                "batch_view_support": "RNA_METH",
                "batch_effect_strength": 1.00,
                "true_biological_shared_factors": 0,
                "true_technical_shared_factors": 1,
                "factor_level_identifiable": False,
                "subspace_level_identifiable": False,
                "primary_truth_target": (
                    "technical_null_specificity"
                ),
            },
        )
    )

    scenarios.append(
        make_scenario(
            "A17_BATCH_SIGNAL_CORRELATED",
            (
                "A true biological shared factor is correlated "
                "with a technical batch score."
            ),
            "technical_batch_signal_correlation",
            "technical_disentanglement_screening",
            {
                "batch_mode": "signal_correlated",
                "batch_view_support": "ALL3",
                "batch_effect_strength": 1.00,
                "batch_signal_correlation": 0.50,
                "true_technical_shared_factors": 1,
            },
        )
    )

    scenarios.append(
        make_scenario(
            "A18_BATCH_IMBALANCED_N50",
            (
                "Small n with an imbalanced 80/20 technical "
                "batch and no biological shared factor."
            ),
            "technical_batch_null",
            "imbalanced_batch_specificity_screening",
            {
                "n_samples": 50,
                "n_shared_factors": 0,
                "shared_signal_strength": 0.0,
                "shared_support_by_factor": "",
                "batch_mode": "imbalanced_shared_all_views",
                "batch_view_support": "ALL3",
                "batch_prevalence": 0.20,
                "batch_effect_strength": 1.00,
                "true_biological_shared_factors": 0,
                "true_technical_shared_factors": 1,
                "factor_level_identifiable": False,
                "subspace_level_identifiable": False,
                "primary_truth_target": (
                    "technical_null_specificity"
                ),
            },
        )
    )

    scenarios.append(
        make_scenario(
            "A19_BATCH_IMBALANCED_N62",
            (
                "TARGET-like n with an imbalanced 80/20 "
                "technical batch and no biological shared factor."
            ),
            "technical_batch_null",
            "imbalanced_batch_specificity_screening",
            {
                "n_shared_factors": 0,
                "shared_signal_strength": 0.0,
                "shared_support_by_factor": "",
                "batch_mode": "imbalanced_shared_all_views",
                "batch_view_support": "ALL3",
                "batch_prevalence": 0.20,
                "batch_effect_strength": 1.00,
                "true_biological_shared_factors": 0,
                "true_technical_shared_factors": 1,
                "factor_level_identifiable": False,
                "subspace_level_identifiable": False,
                "primary_truth_target": (
                    "technical_null_specificity"
                ),
            },
        )
    )

    scenarios.append(
        make_scenario(
            "A20_COMBINED_ROTATIONAL_BATCH_MISSING",
            (
                "Rotationally ambiguous biology, correlated batch, "
                "and batch-dependent methylation missingness."
            ),
            "combined_stress",
            "combined_failure_mode_screening",
            {
                "n_samples": 50,
                "n_shared_factors": 2,
                "shared_factor_correlation": 0.80,
                "shared_strength_pattern": (
                    "equal_near_degenerate"
                ),
                "shared_support_by_factor": (
                    "ALL3|ALL3"
                ),
                "true_biological_shared_factors": 2,
                "factor_level_identifiable": False,
                "subspace_level_identifiable": True,
                "primary_truth_target": "subspace",
                "batch_mode": "signal_correlated",
                "batch_view_support": "ALL3",
                "batch_effect_strength": 1.00,
                "batch_signal_correlation": 0.50,
                "true_technical_shared_factors": 1,
                "missingness_mode": "batch_dependent",
                "missingness_target_view": "METHYLATION",
                "batch_missingness_increment": 0.15,
            },
        )
    )

    table = pd.DataFrame(
        scenarios
    )

    validate_scenario_table(
        table,
        table_name="architecture screening addendum",
    )

    return table


def signal_token(value: float) -> str:
    return (
        f"{value:.2f}"
        .replace(".", "P")
    )


def build_final_gate_calibration_scenarios() -> pd.DataFrame:
    scenarios: list[dict[str, Any]] = []

    order = 0

    for n_samples in [
        50,
        62,
        75,
        100,
    ]:
        for signal in [
            0.0,
            0.25,
            1.0,
            1.5,
        ]:
            order += 1
            is_null = signal == 0.0
            replicates = (
                FINAL_GATE_NULL_REPLICATES
                if is_null
                else FINAL_GATE_NON_NULL_REPLICATES
            )

            updates = {
                "n_samples": n_samples,
                "shared_signal_strength": signal,
            }

            if is_null:
                updates.update(
                    {
                        "n_shared_factors": 0,
                        "shared_support_by_factor": "",
                        "true_biological_shared_factors": 0,
                        "factor_level_identifiable": False,
                        "subspace_level_identifiable": False,
                        "primary_truth_target": (
                            "null_specificity"
                        ),
                    }
                )

            scenarios.append(
                make_scenario(
                    (
                        f"G{order:02d}_IND_N{n_samples}_"
                        f"S{signal_token(signal)}"
                    ),
                    (
                        "Final gate calibration under independent "
                        f"shared-factor structure, n={n_samples}, "
                        f"signal={signal:.2f}."
                    ),
                    "final_independent_gate_curve",
                    (
                        "gate_specificity"
                        if is_null
                        else "gate_sensitivity"
                    ),
                    updates,
                    stage="final_gate_calibration",
                    planned_replicates=replicates,
                    stability_resamples=(
                        FINAL_GATE_STABILITY_RESAMPLES
                    ),
                    materialization_policy=(
                        "stream_or_temporary_after_architecture_freeze"
                    ),
                )
            )

    special_specs = [
        (
            "ROTATIONAL_N62_S1P00",
            (
                "Final calibration of subspace recovery for "
                "near-degenerate factors."
            ),
            "rotational_ambiguity",
            "subspace_gate_sensitivity",
            {
                "n_samples": 62,
                "n_shared_factors": 2,
                "shared_signal_strength": 1.0,
                "shared_factor_correlation": 0.80,
                "shared_strength_pattern": (
                    "equal_near_degenerate"
                ),
                "shared_support_by_factor": (
                    "ALL3|ALL3"
                ),
                "true_biological_shared_factors": 2,
                "factor_level_identifiable": False,
                "subspace_level_identifiable": True,
                "primary_truth_target": "subspace",
            },
            FINAL_GATE_NON_NULL_REPLICATES,
        ),
        (
            "MIXED_SUPPORT_N62_S1P00",
            (
                "Final calibration of view-support classification "
                "under mixed all-view and pair-shared truth."
            ),
            "partial_view_sharing",
            "view_support_gate_sensitivity",
            {
                "n_samples": 62,
                "n_shared_factors": 4,
                "shared_support_pattern": "MIXED",
                "shared_support_by_factor": (
                    "ALL3|RNA_METH|RNA_CNV|METH_CNV"
                ),
                "true_biological_shared_factors": 4,
            },
            FINAL_GATE_NON_NULL_REPLICATES,
        ),
        (
            "CORRELATED_PRIVATE_NULL_N62",
            (
                "Final false-shared calibration under "
                "correlated private factors."
            ),
            "correlated_private_null",
            "gate_specificity",
            {
                "n_samples": 62,
                "n_shared_factors": 0,
                "shared_signal_strength": 0.0,
                "shared_support_by_factor": "",
                "private_cross_view_correlation": 0.60,
                "correlated_private_mode": (
                    "correlated_private_scores"
                ),
                "true_biological_shared_factors": 0,
                "factor_level_identifiable": False,
                "subspace_level_identifiable": False,
                "primary_truth_target": "null_specificity",
            },
            FINAL_GATE_NULL_REPLICATES,
        ),
        (
            "BATCH_SHARED_NULL_N62",
            (
                "Final technical false-acceptance calibration "
                "for an all-view batch factor."
            ),
            "technical_batch_null",
            "technical_gate_specificity",
            {
                "n_samples": 62,
                "n_shared_factors": 0,
                "shared_signal_strength": 0.0,
                "shared_support_by_factor": "",
                "batch_mode": "shared_all_views",
                "batch_view_support": "ALL3",
                "batch_effect_strength": 1.00,
                "true_biological_shared_factors": 0,
                "true_technical_shared_factors": 1,
                "factor_level_identifiable": False,
                "subspace_level_identifiable": False,
                "primary_truth_target": (
                    "technical_null_specificity"
                ),
            },
            FINAL_GATE_NULL_REPLICATES,
        ),
        (
            "BATCH_PARTIAL_NULL_N62",
            (
                "Final technical false-acceptance calibration "
                "for a two-view batch factor."
            ),
            "technical_batch_null",
            "technical_gate_specificity",
            {
                "n_samples": 62,
                "n_shared_factors": 0,
                "shared_signal_strength": 0.0,
                "shared_support_by_factor": "",
                "batch_mode": "partial_two_views",
                "batch_view_support": "RNA_METH",
                "batch_effect_strength": 1.00,
                "true_biological_shared_factors": 0,
                "true_technical_shared_factors": 1,
                "factor_level_identifiable": False,
                "subspace_level_identifiable": False,
                "primary_truth_target": (
                    "technical_null_specificity"
                ),
            },
            FINAL_GATE_NULL_REPLICATES,
        ),
        (
            "BATCH_SIGNAL_CORRELATED_N62",
            (
                "Final calibration when biology and batch are "
                "correlated but both are present."
            ),
            "technical_batch_signal_correlation",
            "technical_disentanglement_sensitivity",
            {
                "n_samples": 62,
                "batch_mode": "signal_correlated",
                "batch_view_support": "ALL3",
                "batch_effect_strength": 1.00,
                "batch_signal_correlation": 0.50,
                "true_technical_shared_factors": 1,
            },
            FINAL_GATE_NON_NULL_REPLICATES,
        ),
        (
            "SPARSE_LOADING_N62_S1P00",
            (
                "Final sensitivity calibration for sparse "
                "two-percent biological loadings."
            ),
            "loading_density",
            "sparse_gate_sensitivity",
            {
                "n_samples": 62,
                "shared_loading_fraction": 0.02,
                "private_loading_fraction": 0.02,
                "loading_regime": "sparse_2pct",
                "shared_signal_strength": 1.0,
            },
            FINAL_GATE_NON_NULL_REPLICATES,
        ),
        (
            "DENSE_LOADING_N62_S1P00",
            (
                "Final sensitivity calibration for dense "
                "forty-percent biological loadings."
            ),
            "loading_density",
            "dense_gate_sensitivity",
            {
                "n_samples": 62,
                "shared_loading_fraction": 0.40,
                "private_loading_fraction": 0.40,
                "loading_regime": "dense_40pct",
                "shared_signal_strength": 1.0,
            },
            FINAL_GATE_NON_NULL_REPLICATES,
        ),
    ]

    for (
        suffix,
        description,
        family,
        role,
        updates,
        replicates,
    ) in special_specs:
        order += 1
        scenarios.append(
            make_scenario(
                f"G{order:02d}_{suffix}",
                description,
                family,
                role,
                updates,
                stage="final_gate_calibration",
                planned_replicates=replicates,
                stability_resamples=(
                    FINAL_GATE_STABILITY_RESAMPLES
                ),
                materialization_policy=(
                    "stream_or_temporary_after_architecture_freeze"
                ),
            )
        )

    table = pd.DataFrame(
        scenarios
    )

    validate_scenario_table(
        table,
        table_name="final gate calibration",
    )

    return table


def validate_scenario_table(
    table: pd.DataFrame,
    table_name: str,
) -> None:
    required = {
        "scenario_id",
        "description",
        "scenario_family",
        "calibration_role",
        "simulation_stage",
        "n_samples",
        "n_shared_factors",
        "n_private_factors_per_view",
        "n_features_rna",
        "n_features_methylation",
        "n_features_cnv",
        "shared_signal_strength",
        "private_signal_strength",
        "noise_sd_rna",
        "noise_sd_methylation",
        "noise_sd_cnv",
        "missing_fraction_rna",
        "missing_fraction_methylation",
        "missing_fraction_cnv",
        "shared_factor_correlation",
        "shared_support_by_factor",
        "private_cross_view_correlation",
        "shared_loading_fraction",
        "private_loading_fraction",
        "missingness_mode",
        "batch_mode",
        "batch_effect_strength",
        "true_biological_shared_factors",
        "true_technical_shared_factors",
        "factor_level_identifiable",
        "subspace_level_identifiable",
        "planned_replicates",
        "stability_resamples_per_dataset",
    }

    missing = required - set(
        table.columns
    )

    if missing:
        raise RuntimeError(
            f"{table_name} scenario table is missing: "
            + ", ".join(
                sorted(missing)
            )
        )

    if table.empty:
        raise RuntimeError(
            f"{table_name} scenario table is empty."
        )

    if table[
        "scenario_id"
    ].duplicated().any():
        duplicates = table.loc[
            table[
                "scenario_id"
            ].duplicated(),
            "scenario_id",
        ].tolist()

        raise RuntimeError(
            f"Duplicate {table_name} scenario IDs: "
            f"{duplicates}"
        )

    probability_columns = [
        "missing_fraction_rna",
        "missing_fraction_methylation",
        "missing_fraction_cnv",
        "pair_corruption_fraction",
        "shared_factor_correlation",
        "private_cross_view_correlation",
        "shared_loading_fraction",
        "private_loading_fraction",
        "low_quality_sample_fraction",
        "low_quality_feature_fraction",
        "low_quality_missing_fraction",
        "partial_view_sample_fraction",
        "partial_view_missing_fraction",
        "batch_prevalence",
        "batch_signal_correlation",
        "batch_missingness_increment",
    ]

    for column in probability_columns:
        values = pd.to_numeric(
            table[column],
            errors="raise",
        )

        if (
            values.lt(0).any()
            or values.gt(1).any()
        ):
            raise RuntimeError(
                f"{table_name}: {column} must be in [0, 1]."
            )

    positive_integer_columns = [
        "n_samples",
        "n_private_factors_per_view",
        "n_features_rna",
        "n_features_methylation",
        "n_features_cnv",
        "planned_replicates",
        "stability_resamples_per_dataset",
    ]

    for column in positive_integer_columns:
        values = pd.to_numeric(
            table[column],
            errors="raise",
        )

        if values.le(0).any():
            raise RuntimeError(
                f"{table_name}: {column} must be positive."
            )

    shared_counts = pd.to_numeric(
        table["n_shared_factors"],
        errors="raise",
    )

    if shared_counts.lt(0).any():
        raise RuntimeError(
            f"{table_name}: n_shared_factors cannot be negative."
        )


def build_replicate_registry(
    scenarios: pd.DataFrame,
    registry_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for scenario_order, row in scenarios.reset_index(
        drop=True
    ).iterrows():
        scenario_id = clean_text(
            row["scenario_id"]
        )

        n_replicates = int(
            row["planned_replicates"]
        )

        for replicate_index in range(
            1,
            n_replicates + 1,
        ):
            rows.append(
                {
                    "registry_label": registry_label,
                    "simulation_stage": row[
                        "simulation_stage"
                    ],
                    "scenario_id": scenario_id,
                    "scenario_order": int(
                        scenario_order + 1
                    ),
                    "replicate_index": replicate_index,
                    "simulation_seed": stable_seed(
                        ADDENDUM_MASTER_SEED,
                        registry_label,
                        scenario_id,
                        replicate_index,
                    ),
                    "stability_resamples_per_dataset": int(
                        row[
                            "stability_resamples_per_dataset"
                        ]
                    ),
                    "materialization_policy": row[
                        "materialization_policy"
                    ],
                }
            )

    table = pd.DataFrame(rows)

    if table.empty:
        raise RuntimeError(
            f"Replicate registry {registry_label} is empty."
        )

    duplicate_mask = table.duplicated(
        [
            "scenario_id",
            "replicate_index",
        ]
    )

    if duplicate_mask.any():
        raise RuntimeError(
            f"Duplicate entries in {registry_label} replicate registry."
        )

    return table


def augment_base_scenarios(
    base_scenarios: pd.DataFrame,
) -> pd.DataFrame:
    table = base_scenarios.copy()

    defaults = {
        "scenario_origin": "05a_base",
        "scenario_family": "base_05a_screening",
        "calibration_role": "architecture_screening",
        "simulation_stage": "architecture_screening",
        "nonlinear_mode": "none",
        "shared_factor_correlation": 0.0,
        "shared_strength_pattern": "equal",
        "shared_support_pattern": "ALL3",
        "private_cross_view_correlation": 0.0,
        "correlated_private_mode": "none",
        "shared_loading_fraction": 0.10,
        "private_loading_fraction": 0.10,
        "loading_regime": "moderate_10pct",
        "missingness_mode": "mcar_cell",
        "missingness_target_view": "FROM_BASE_COLUMNS",
        "low_quality_sample_fraction": 0.0,
        "low_quality_feature_fraction": 0.0,
        "low_quality_missing_fraction": 0.0,
        "partial_view_sample_fraction": 0.0,
        "partial_view_missing_fraction": 0.0,
        "batch_mode": "none",
        "batch_cardinality": 2,
        "batch_prevalence": 0.50,
        "batch_effect_strength": 0.0,
        "batch_view_support": "NONE",
        "batch_signal_correlation": 0.0,
        "batch_missingness_increment": 0.0,
        "true_technical_shared_factors": 0,
        "factor_level_identifiable": True,
        "subspace_level_identifiable": True,
        "primary_truth_target": "factor_and_subspace",
        "stability_resamples_per_dataset": (
            ARCHITECTURE_STABILITY_RESAMPLES
        ),
        "materialization_policy": "persist_npz",
        "generator_required_version": (
            GENERATOR_REQUIRED_VERSION
        ),
    }

    for column, value in defaults.items():
        if column not in table.columns:
            table[column] = value

    table["planned_replicates"] = pd.to_numeric(
        table["replicates"],
        errors="raise",
    ).astype(int)

    table[
        "architecture_screening_replicates"
    ] = table["planned_replicates"]

    shared_counts = pd.to_numeric(
        table["n_shared_factors"],
        errors="raise",
    ).astype(int)

    table[
        "true_biological_shared_factors"
    ] = shared_counts

    table[
        "shared_support_by_factor"
    ] = [
        support_string(
            ["ALL3"] * count
        )
        if count > 0
        else ""
        for count in shared_counts
    ]

    null_mask = shared_counts.eq(0)

    table.loc[
        null_mask,
        "factor_level_identifiable",
    ] = False

    table.loc[
        null_mask,
        "subspace_level_identifiable",
    ] = False

    table.loc[
        null_mask,
        "primary_truth_target",
    ] = "null_specificity"

    if "uninformative_modality" in table.columns:
        table[
            "uninformative_modality"
        ] = table[
            "uninformative_modality"
        ].astype(str).str.upper().replace(
            {
                "NONE": "NONE",
                "NAN": "NONE",
                "": "NONE",
            }
        )

    return table


def build_combined_architecture_replicates(
    base_replicates: pd.DataFrame,
    addendum_replicates: pd.DataFrame,
) -> pd.DataFrame:
    base = base_replicates.copy()

    base.insert(
        0,
        "registry_label",
        "05a_base_architecture_screening",
    )

    base[
        "simulation_stage"
    ] = "architecture_screening"

    base[
        "stability_resamples_per_dataset"
    ] = ARCHITECTURE_STABILITY_RESAMPLES

    base[
        "materialization_policy"
    ] = "persist_npz"

    columns = [
        "registry_label",
        "simulation_stage",
        "scenario_id",
        "scenario_order",
        "replicate_index",
        "simulation_seed",
        "stability_resamples_per_dataset",
        "materialization_policy",
    ]

    base = base[columns]
    addendum = addendum_replicates[columns]

    combined = pd.concat(
        [
            base,
            addendum,
        ],
        ignore_index=True,
    )

    if combined.duplicated(
        [
            "registry_label",
            "scenario_id",
            "replicate_index",
        ]
    ).any():
        raise RuntimeError(
            "Duplicate combined architecture replicate keys."
        )

    return combined


# ---------------------------------------------------------------------
# Truth, metric, decision, and technical registries
# ---------------------------------------------------------------------


def build_truth_taxonomy() -> pd.DataFrame:
    rows = [
        {
            "truth_id": "BIO_SHARED_ALL3",
            "truth_class": "biological_shared",
            "view_support": "ALL3",
            "expected_gate_action": "retain_if_stable",
            "factor_level_evaluable": True,
            "subspace_level_evaluable": True,
            "technical": False,
        },
        {
            "truth_id": "BIO_SHARED_RNA_METH",
            "truth_class": "biological_partial_shared",
            "view_support": "RNA_METH",
            "expected_gate_action": "retain_if_stable_and_classify_support",
            "factor_level_evaluable": True,
            "subspace_level_evaluable": True,
            "technical": False,
        },
        {
            "truth_id": "BIO_SHARED_RNA_CNV",
            "truth_class": "biological_partial_shared",
            "view_support": "RNA_CNV",
            "expected_gate_action": "retain_if_stable_and_classify_support",
            "factor_level_evaluable": True,
            "subspace_level_evaluable": True,
            "technical": False,
        },
        {
            "truth_id": "BIO_SHARED_METH_CNV",
            "truth_class": "biological_partial_shared",
            "view_support": "METH_CNV",
            "expected_gate_action": "retain_if_stable_and_classify_support",
            "factor_level_evaluable": True,
            "subspace_level_evaluable": True,
            "technical": False,
        },
        {
            "truth_id": "BIO_PRIVATE",
            "truth_class": "biological_private",
            "view_support": "ONE_VIEW",
            "expected_gate_action": "exclude_from_shared_gate",
            "factor_level_evaluable": True,
            "subspace_level_evaluable": True,
            "technical": False,
        },
        {
            "truth_id": "CORRELATED_PRIVATE_NULL",
            "truth_class": "false_shared_mechanism",
            "view_support": "NONE_TRUE_SHARED",
            "expected_gate_action": "reject",
            "factor_level_evaluable": False,
            "subspace_level_evaluable": False,
            "technical": False,
        },
        {
            "truth_id": "TECHNICAL_SHARED_ALL3",
            "truth_class": "technical_shared",
            "view_support": "ALL3",
            "expected_gate_action": "flag_as_technical_not_biological",
            "factor_level_evaluable": True,
            "subspace_level_evaluable": True,
            "technical": True,
        },
        {
            "truth_id": "TECHNICAL_SHARED_PAIR",
            "truth_class": "technical_partial_shared",
            "view_support": "TWO_VIEWS",
            "expected_gate_action": "flag_as_technical_not_biological",
            "factor_level_evaluable": True,
            "subspace_level_evaluable": True,
            "technical": True,
        },
        {
            "truth_id": "ROTATIONAL_SHARED_SUBSPACE",
            "truth_class": "biological_shared_subspace",
            "view_support": "ALL3",
            "expected_gate_action": "retain_subspace_not_individual_factor",
            "factor_level_evaluable": False,
            "subspace_level_evaluable": True,
            "technical": False,
        },
    ]

    return pd.DataFrame(rows)


def build_gate_metric_registry() -> pd.DataFrame:
    rows = [
        {
            "metric_id": "gate_true_positive_rate",
            "direction": "higher",
            "truth_scope": "non_null_shared_factors",
            "formula": (
                "number of true shared factors matched and retained / "
                "number of true shared factors"
            ),
            "primary_status": "primary_operating_characteristic",
            "confidence_interval": "Wilson 95% binomial interval",
        },
        {
            "metric_id": "gate_false_negative_rate",
            "direction": "lower",
            "truth_scope": "non_null_shared_factors",
            "formula": "1 - gate_true_positive_rate",
            "primary_status": "primary_operating_characteristic",
            "confidence_interval": "Wilson 95% binomial interval",
        },
        {
            "metric_id": "gate_false_discovery_proportion",
            "direction": "lower",
            "truth_scope": "retained_shared_factors",
            "formula": (
                "retained factors without a true biological shared match / "
                "all retained shared factors"
            ),
            "primary_status": "primary_operating_characteristic",
            "confidence_interval": "bootstrap percentile 95% interval across datasets",
        },
        {
            "metric_id": "gate_familywise_false_acceptance",
            "direction": "lower",
            "truth_scope": "null_datasets",
            "formula": (
                "probability of retaining at least one false biological "
                "shared factor in a null dataset"
            ),
            "primary_status": "specificity_failure_gate",
            "confidence_interval": "Wilson 95% binomial interval",
        },
        {
            "metric_id": "factor_count_bias",
            "direction": "absolute_closer_to_zero",
            "truth_scope": "all_simulations",
            "formula": (
                "number retained as biological shared minus number of true "
                "biological shared factors"
            ),
            "primary_status": "secondary_operating_characteristic",
            "confidence_interval": "bootstrap percentile 95% interval",
        },
        {
            "metric_id": "factor_level_truth_match",
            "direction": "higher",
            "truth_scope": "factor_identifiable_scenarios",
            "formula": (
                "absolute correlation after Hungarian matching and sign alignment; "
                f"truth match threshold={TRUTH_FACTOR_MATCH_ABS_CORRELATION:.2f}"
            ),
            "primary_status": "factor_recovery",
            "confidence_interval": "bootstrap percentile 95% interval",
        },
        {
            "metric_id": "subspace_mean_principal_angle_cosine",
            "direction": "higher",
            "truth_scope": "subspace_identifiable_scenarios",
            "formula": (
                "mean cosine of principal angles between estimated and true "
                "shared subspaces"
            ),
            "primary_status": "subspace_recovery",
            "confidence_interval": "bootstrap percentile 95% interval",
        },
        {
            "metric_id": "subspace_projection_distance",
            "direction": "lower",
            "truth_scope": "subspace_identifiable_scenarios",
            "formula": (
                "Frobenius norm of the difference between estimated and true "
                "orthogonal projection matrices, normalized by true rank"
            ),
            "primary_status": "subspace_recovery",
            "confidence_interval": "bootstrap percentile 95% interval",
        },
        {
            "metric_id": "view_support_classification_accuracy",
            "direction": "higher",
            "truth_scope": "partial_sharing_scenarios",
            "formula": (
                "fraction of matched retained factors assigned to the correct "
                "all-view or pair-view support class"
            ),
            "primary_status": "view_support_recovery",
            "confidence_interval": "Wilson 95% binomial interval",
        },
        {
            "metric_id": "private_to_shared_leakage_rate",
            "direction": "lower",
            "truth_scope": "private_and_correlated_private_scenarios",
            "formula": (
                "fraction of retained shared factors matched more strongly to a "
                "private truth factor than to any shared truth factor"
            ),
            "primary_status": "false_shared_diagnostic",
            "confidence_interval": "Wilson 95% binomial interval",
        },
        {
            "metric_id": "batch_false_biological_acceptance_rate",
            "direction": "lower",
            "truth_scope": "technical_batch_null_scenarios",
            "formula": (
                "fraction of technical batch factors retained and not flagged as "
                "technical; batch truth match threshold="
                f"{TRUTH_BATCH_MATCH_ABS_CORRELATION:.2f}"
            ),
            "primary_status": "technical_specificity_failure_gate",
            "confidence_interval": "Wilson 95% binomial interval",
        },
        {
            "metric_id": "technical_factor_detection_sensitivity",
            "direction": "higher",
            "truth_scope": "batch_present_scenarios",
            "formula": (
                "fraction of true technical factors correctly flagged as technical"
            ),
            "primary_status": "technical_audit_metric",
            "confidence_interval": "Wilson 95% binomial interval",
        },
    ]

    return pd.DataFrame(rows)


def build_gate_decision_registry() -> pd.DataFrame:
    rows = [
        {
            "decision_id": "D01_NULL_SPECIFICITY",
            "scenario_selector": (
                "true_biological_shared_factors == 0 and "
                "true_technical_shared_factors == 0"
            ),
            "metric": "gate_familywise_false_acceptance",
            "required_threshold": (
                "upper operating estimate <= 0.10"
            ),
            "failure_consequence": (
                "stability gate is insufficiently specific; do not claim a "
                "validated biological factor acceptance rule"
            ),
            "threshold_retuning_allowed": False,
        },
        {
            "decision_id": "D02_STANDARD_SENSITIVITY",
            "scenario_selector": (
                "n_samples == 62 and shared_signal_strength == 1.0 and "
                "shared_factor_correlation == 0 and batch_mode == 'none'"
            ),
            "metric": "gate_true_positive_rate",
            "required_threshold": "point estimate >= 0.80",
            "failure_consequence": (
                "stability gate has inadequate sensitivity at the TARGET-like "
                "operating point"
            ),
            "threshold_retuning_allowed": False,
        },
        {
            "decision_id": "D03_STANDARD_FNR",
            "scenario_selector": (
                "n_samples == 62 and shared_signal_strength == 1.0 and "
                "shared_factor_correlation == 0 and batch_mode == 'none'"
            ),
            "metric": "gate_false_negative_rate",
            "required_threshold": "point estimate <= 0.20",
            "failure_consequence": (
                "stability gate rejects too many true factors at the "
                "TARGET-like operating point"
            ),
            "threshold_retuning_allowed": False,
        },
        {
            "decision_id": "D04_ROTATIONAL_SUBSPACE",
            "scenario_selector": (
                "primary_truth_target == 'subspace' and n_samples == 62 and "
                "shared_signal_strength == 1.0"
            ),
            "metric": "subspace_mean_principal_angle_cosine",
            "required_threshold": "point estimate >= 0.80",
            "failure_consequence": (
                "rotationally ambiguous shared structure is not recovered "
                "reliably; individual factor claims remain forbidden"
            ),
            "threshold_retuning_allowed": False,
        },
        {
            "decision_id": "D05_VIEW_SUPPORT",
            "scenario_selector": (
                "shared_support_pattern == 'MIXED'"
            ),
            "metric": "view_support_classification_accuracy",
            "required_threshold": "point estimate >= 0.80",
            "failure_consequence": (
                "view-support labels are exploratory and cannot define primary "
                "shared-versus-partial biological claims"
            ),
            "threshold_retuning_allowed": False,
        },
        {
            "decision_id": "D06_BATCH_SPECIFICITY",
            "scenario_selector": (
                "true_technical_shared_factors > 0 and "
                "true_biological_shared_factors == 0"
            ),
            "metric": "batch_false_biological_acceptance_rate",
            "required_threshold": "point estimate <= 0.10",
            "failure_consequence": (
                "technical-confound flagging is insufficient; real-data stable "
                "factors cannot be assumed biological without external recurrence"
            ),
            "threshold_retuning_allowed": False,
        },
        {
            "decision_id": "D07_NO_POSTHOC_RETUNING",
            "scenario_selector": "all final calibration scenarios",
            "metric": "all operating metrics",
            "required_threshold": (
                "thresholds fixed before any simulation model result is viewed"
            ),
            "failure_consequence": (
                "any alternative threshold becomes a labeled exploratory "
                "sensitivity analysis, not the primary gate"
            ),
            "threshold_retuning_allowed": False,
        },
    ]

    return pd.DataFrame(rows)


def build_technical_covariate_registry() -> pd.DataFrame:
    rows = [
        {
            "covariate_id": "sequencing_or_processing_center",
            "candidate_alias_regex": (
                "center|sequencing_center|processing_center|tissue_source_site"
            ),
            "data_type": "categorical",
            "primary_status": "candidate",
            "allowed_for_model_selection": False,
            "allowed_for_factor_flagging": True,
            "notes": (
                "Tissue source site is retained as an ambiguous technical/site "
                "covariate and interpreted cautiously."
            ),
        },
        {
            "covariate_id": "platform",
            "candidate_alias_regex": (
                "platform|instrument_model|array_platform"
            ),
            "data_type": "categorical",
            "primary_status": "candidate",
            "allowed_for_model_selection": False,
            "allowed_for_factor_flagging": True,
            "notes": "Use only if at least two non-sparse levels exist.",
        },
        {
            "covariate_id": "workflow_type",
            "candidate_alias_regex": (
                "workflow_type|workflow|pipeline|analysis_workflow"
            ),
            "data_type": "categorical",
            "primary_status": "candidate",
            "allowed_for_model_selection": False,
            "allowed_for_factor_flagging": True,
            "notes": "Constant-within-view fields are documented but not tested.",
        },
        {
            "covariate_id": "experimental_strategy",
            "candidate_alias_regex": (
                "experimental_strategy|assay_type|data_type"
            ),
            "data_type": "categorical",
            "primary_status": "descriptive_only_if_constant",
            "allowed_for_model_selection": False,
            "allowed_for_factor_flagging": True,
            "notes": "Do not confuse modality identity with a sample-level batch.",
        },
        {
            "covariate_id": "array_plate_or_chip",
            "candidate_alias_regex": (
                "plate|chip|sentrix|slide|array_id|batch_id"
            ),
            "data_type": "categorical",
            "primary_status": "candidate",
            "allowed_for_model_selection": False,
            "allowed_for_factor_flagging": True,
            "notes": "Reject identifier-like fields with excessive cardinality.",
        },
        {
            "covariate_id": "preservation_method",
            "candidate_alias_regex": (
                "preservation_method|preservation|freezing_method"
            ),
            "data_type": "categorical",
            "primary_status": "candidate",
            "allowed_for_model_selection": False,
            "allowed_for_factor_flagging": True,
            "notes": "Use only as a technical specimen-handling covariate.",
        },
        {
            "covariate_id": "analyte_type",
            "candidate_alias_regex": (
                "analyte_type|analyte|molecule_type"
            ),
            "data_type": "categorical",
            "primary_status": "candidate",
            "allowed_for_model_selection": False,
            "allowed_for_factor_flagging": True,
            "notes": "Constant fields remain descriptive only.",
        },
        {
            "covariate_id": "specimen_or_portion_type",
            "candidate_alias_regex": (
                "specimen_type|portion_type|sample_type|tissue_type"
            ),
            "data_type": "categorical",
            "primary_status": "candidate_ambiguous",
            "allowed_for_model_selection": False,
            "allowed_for_factor_flagging": True,
            "notes": (
                "May contain biological content; technical flagging must be "
                "reported as ambiguous rather than automatically corrected."
            ),
        },
        {
            "covariate_id": "submission_or_creation_time_bin",
            "candidate_alias_regex": (
                "created_datetime|updated_datetime|submission_date|created_date"
            ),
            "data_type": "ordered_categorical",
            "primary_status": "candidate_derived",
            "allowed_for_model_selection": False,
            "allowed_for_factor_flagging": True,
            "notes": (
                "Dates may be binned by calendar quarter only; raw unique "
                "timestamps are forbidden."
            ),
        },
        {
            "covariate_id": "file_case_sample_aliquot_uuid",
            "candidate_alias_regex": (
                "uuid|file_id|case_id|sample_id|aliquot_id|submitter_id"
            ),
            "data_type": "identifier",
            "primary_status": "forbidden",
            "allowed_for_model_selection": False,
            "allowed_for_factor_flagging": False,
            "notes": "Identifiers may be used only for linkage and integrity checks.",
        },
        {
            "covariate_id": "clinical_or_outcome_field",
            "candidate_alias_regex": (
                "survival|death|event|metast|recurr|response|therapy|treatment|"
                "age|sex|race|stage|grade|histology|prognos"
            ),
            "data_type": "clinical_or_outcome",
            "primary_status": "forbidden_before_representation_freeze",
            "allowed_for_model_selection": False,
            "allowed_for_factor_flagging": False,
            "notes": "Never loaded in the technical metadata preflight.",
        },
    ]

    return pd.DataFrame(rows)


def build_technical_metadata_contract() -> dict[str, Any]:
    return {
        "contract_version": (
            "paper5-target-technical-metadata-preflight-v1"
        ),
        "status": "FROZEN_BEFORE_MODEL_FITTING",
        "purpose": (
            "Inventory usable sample-level technical covariates without "
            "accessing clinical outcomes or using technical labels for model "
            "family selection."
        ),
        "source_roots": [
            relative_path(
                PAPER5_ROOT
                / "human"
                / "TARGET_OS"
            ),
            relative_path(
                PAPER5_ROOT
                / "data_manifest"
            ),
        ],
        "permitted_input_patterns": [
            "gdc_sample_sheet*.tsv",
            "metadata*.json",
            "files-table*.tsv",
            "biospecimen metadata files if available",
        ],
        "forbidden_input_patterns": [
            "clinical outcome tables",
            "survival files",
            "response files",
            "metastasis files",
            "treatment files",
        ],
        "field_usability_rules": {
            "maximum_missing_fraction": (
                TECHNICAL_ASSOCIATION_THRESHOLDS[
                    "maximum_missing_fraction"
                ]
            ),
            "maximum_unique_fraction": (
                TECHNICAL_ASSOCIATION_THRESHOLDS[
                    "maximum_unique_fraction"
                ]
            ),
            "minimum_level_size": (
                TECHNICAL_ASSOCIATION_THRESHOLDS[
                    "minimum_level_size"
                ]
            ),
            "maximum_categorical_levels": (
                TECHNICAL_ASSOCIATION_THRESHOLDS[
                    "maximum_categorical_levels"
                ]
            ),
            "constant_fields": "document_but_do_not_test",
            "identifier_like_fields": "exclude",
            "raw_dates": "forbidden",
            "date_derivation": (
                "calendar-quarter bins only if at least two usable levels and "
                "minimum level size is met"
            ),
        },
        "post_representation_factor_association": {
            "continuous": (
                "absolute Spearman correlation with 10000 label permutations"
            ),
            "binary": (
                "standardized mean difference and permutation p-value"
            ),
            "multicategory": (
                "eta-squared from one-way model and permutation p-value"
            ),
            "multiplicity": (
                "Benjamini-Hochberg within the separate technical-confound family"
            ),
            "q_value_threshold": (
                TECHNICAL_ASSOCIATION_THRESHOLDS[
                    "bh_q_value"
                ]
            ),
            "continuous_effect_threshold": (
                TECHNICAL_ASSOCIATION_THRESHOLDS[
                    "absolute_spearman_rho"
                ]
            ),
            "categorical_effect_threshold": (
                TECHNICAL_ASSOCIATION_THRESHOLDS[
                    "eta_squared"
                ]
            ),
            "binary_effect_threshold": (
                TECHNICAL_ASSOCIATION_THRESHOLDS[
                    "standardized_mean_difference"
                ]
            ),
        },
        "factor_status_taxonomy": {
            "unstable": (
                "fails the upstream factor or subspace stability gate"
            ),
            "stable_technical_flag": (
                "passes stability but has technical q<0.05 and exceeds the "
                "prespecified effect-size threshold"
            ),
            "stable_nontechnical": (
                "passes stability and no usable technical covariate meets the "
                "flagging rule"
            ),
            "stable_technical_status_unknown": (
                "passes stability but no usable sample-level technical metadata "
                "are available"
            ),
            "stable_ambiguous_specimen_association": (
                "passes stability and associates only with an ambiguous specimen "
                "or site variable that may contain biological information"
            ),
        },
        "biological_claim_rule": (
            "Stable factors with a technical flag are retained in the audit "
            "but excluded from primary biological claims. Stable factors with "
            "unknown technical status require external recurrence evidence and "
            "an explicit limitation."
        ),
        "batch_adjustment_policy": {
            "primary_analysis": "no automatic batch residualization",
            "adversarial_batch_removal": "not prespecified",
            "sensitivity_analysis": (
                "fold-local residualization may be performed only after the "
                "technical audit identifies a usable covariate; parameters are "
                "fit on training samples and applied unchanged to held-out samples"
            ),
            "outcome_access": False,
        },
        "guardrails": [
            "Technical labels are not used to select the model family.",
            "Clinical and outcome fields remain inaccessible.",
            "High-cardinality identifiers cannot be used as covariates.",
            "A reproducible factor is not assumed biological solely because it is stable.",
        ],
    }


def build_external_benchmark_addendum() -> dict[str, Any]:
    return {
        "addendum_version": (
            "paper5-external-method-benchmark-addendum-v1"
        ),
        "status": "FROZEN_BEFORE_MODEL_FITTING",
        "amends_base_05a_section": "external_method_benchmark",
        "reason": (
            "Add a disease-related sarcoma benchmark and prohibit label-selected "
            "MLOmics feature scales before viewing SABLE-MO results."
        ),
        "benchmark_hierarchy": [
            {
                "priority": 1,
                "cohort": "TCGA-SARC",
                "role": "primary_disease_related_external_method_benchmark",
                "source_preference": (
                    "direct GDC or reproducibly linked multi-omics source"
                ),
                "required_modalities": [
                    "RNA",
                    "DNA methylation",
                    "gene-level CNV",
                ],
                "minimum_complete_cases_for_primary_role": 50,
                "selection_uses_outcome": False,
                "status": "requires_preflight",
            },
            {
                "priority": 2,
                "cohort": "MLOmics_Original_generalization_cohort",
                "role": "non_sarcoma_generalization_benchmark",
                "selection_rule": (
                    "Select the eligible non-sarcoma cancer type with the largest "
                    "complete-case RNA-methylation-CNV sample count; break ties "
                    "alphabetically before any method result is viewed."
                ),
                "minimum_complete_cases": 100,
                "allowed_scale": "Original",
                "forbidden_scale": "Top",
                "forbidden_reason": (
                    "Top uses cancer-type-label ANOVA feature selection."
                ),
                "selection_uses_outcome": False,
            },
        ],
        "downsampling_sample_sizes": [
            50,
            75,
            100,
        ],
        "minimum_downsampling_repeats": 20,
        "target_os_use": (
            "flagship osteosarcoma biological application, not the sole proof "
            "of a general method claim"
        ),
        "guardrails": [
            "Do not assume SARC is a standalone MLOmics task without preflight confirmation.",
            "Do not use MLOmics Top features.",
            "Do not use external outcomes for architecture selection.",
            "External cohorts are selected before SABLE-MO benchmark results are viewed.",
        ],
    }


def build_novelty_positioning_contract() -> dict[str, Any]:
    return {
        "contract_version": (
            "paper5-sable-mo-novelty-positioning-v1"
        ),
        "status": "PROVISIONAL_UNTIL_SYSTEMATIC_REVIEW_UPDATE",
        "allowed_provisional_claim": (
            "SABLE-MO is explicitly designed and prospectively characterized for "
            "extreme-small-sample same-patient multi-omics representation learning "
            "using block-balanced optimization, shared/private decomposition, "
            "missing-view recovery, and factor-and-subspace reproducibility gates."
        ),
        "forbidden_claims": [
            "first stability-based multi-omics method",
            "first shared/private multi-omics autoencoder",
            "existing methods were never evaluated below n=100",
            "theoretically guaranteed factor stability",
            "SABLE-MO solves small-n multi-omics universally",
        ],
        "terminology": {
            "preferred": (
                "simulation-characterized operating properties"
            ),
            "avoid_without_theorem": (
                "theoretically guaranteed stability"
            ),
        },
        "related_work_distinctions": {
            "MOFA_plus": (
                "probabilistic shared and view-specific factors with native "
                "missing-value handling and ARD/sparsity"
            ),
            "CODE_AE": (
                "shared/private domain separation between cell lines and patient "
                "tissues for drug-response transfer, not same-patient omics views"
            ),
            "Flexynesis": (
                "modular bulk multi-omics deep-learning toolkit; not the same "
                "prospective factor acceptance and rejection framework"
            ),
            "stability_based_multiomics": (
                "stability concepts pre-exist; novelty cannot rest on stability alone"
            ),
        },
        "headline_novelty_components": [
            "extreme-small-n operating regime",
            "block-balanced loss",
            "same-patient cross-view contrastive alignment",
            "shared/private decomposition",
            "modality dropout and missing-view recovery",
            "factor-level and subspace-level stability",
            "prospective TPR/FNR/FDP and false-acceptance characterization",
            "explicit technical-confound flagging",
            "outcome-blind model selection before biological annotation",
        ],
        "final_first_or_unique_claim_rule": (
            "No first or unique claim is permitted until a documented systematic "
            "novelty search is repeated immediately before manuscript submission."
        ),
    }


def build_environment_snapshot() -> dict[str, Any]:
    distributions = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "mofapy2",
        "torch",
    ]

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
        "packages": {
            name: package_version(name)
            for name in distributions
        },
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    # This script intentionally accepts no command-line parameters.
    # A frozen addendum is verified and never overwritten.
    if OUTPUT_LOCK.exists():
        verify_existing_addendum()
        return

    print("=" * 88)
    print(
        "Paper 5 - freeze simulation and technical-confound addendum"
    )
    print("=" * 88)
    print(
        f"Script version: {SCRIPT_VERSION}"
    )
    print(
        f"Method: {METHOD_NAME}"
    )
    print(
        f"Project root: {PROJECT_ROOT}"
    )
    print(
        f"Output directory: {OUTPUT_DIR}"
    )
    print()

    base = validate_base_contract()

    model_results = find_preexisting_model_results()

    if model_results:
        raise RuntimeError(
            "05a1 cannot be frozen after simulation or representation-model "
            "results exist. Found:\n"
            + "\n".join(
                f"- {value}"
                for value in model_results
            )
        )

    pilot_present = preexisting_05b_pilot_present()
    pilot_disclosure: dict[str, Any] | None = None

    if pilot_present:
        print(
            "A pre-addendum 05b v1 data-only materialization was detected. "
            "It will be fully hash-audited, disclosed, preserved, and excluded "
            "from confirmatory analyses."
        )

        pilot_disclosure = validate_existing_05b_pilot(
            full_hash_audit=FULL_PILOT_HASH_AUDIT
        )

    print(
        "05a base contract and generated-file hashes: PASS"
    )
    print(
        "Fitted simulation/TARGET model results before addendum: NO"
    )
    print(
        "Pre-addendum 05b data-only materialization: "
        + ("ACKNOWLEDGED" if pilot_disclosure is not None else "NO")
    )
    print(
        "Outcome access before representation freeze: FORBIDDEN"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    architecture_scenarios = (
        build_architecture_screening_scenarios()
    )

    architecture_replicates = (
        build_replicate_registry(
            architecture_scenarios,
            "05a1_architecture_screening_addendum",
        )
    )

    gate_scenarios = (
        build_final_gate_calibration_scenarios()
    )

    gate_replicates = (
        build_replicate_registry(
            gate_scenarios,
            "05a1_final_gate_calibration",
        )
    )

    base_augmented = augment_base_scenarios(
        base["base_scenarios"]
    )

    combined_architecture_scenarios = pd.concat(
        [
            base_augmented,
            architecture_scenarios,
        ],
        ignore_index=True,
        sort=False,
    )

    if combined_architecture_scenarios[
        "scenario_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate scenario IDs exist after combining 05a and 05a1."
        )

    combined_architecture_replicates = (
        build_combined_architecture_replicates(
            base["base_replicates"],
            architecture_replicates,
        )
    )

    truth_taxonomy = build_truth_taxonomy()
    gate_metrics = build_gate_metric_registry()
    gate_decisions = build_gate_decision_registry()
    technical_covariates = (
        build_technical_covariate_registry()
    )
    technical_contract = (
        build_technical_metadata_contract()
    )
    external_benchmark_addendum = (
        build_external_benchmark_addendum()
    )
    novelty_contract = (
        build_novelty_positioning_contract()
    )
    environment = build_environment_snapshot()

    write_table(
        architecture_scenarios,
        OUTPUT_ARCHITECTURE_SCENARIOS,
    )
    write_table(
        architecture_replicates,
        OUTPUT_ARCHITECTURE_REPLICATES,
    )
    write_table(
        gate_scenarios,
        OUTPUT_GATE_SCENARIOS,
    )
    write_table(
        gate_replicates,
        OUTPUT_GATE_REPLICATES,
    )
    write_table(
        combined_architecture_scenarios,
        OUTPUT_COMBINED_ARCHITECTURE_SCENARIOS,
    )
    write_table(
        combined_architecture_replicates,
        OUTPUT_COMBINED_ARCHITECTURE_REPLICATES,
    )
    write_table(
        truth_taxonomy,
        OUTPUT_TRUTH_TAXONOMY,
    )
    write_table(
        gate_metrics,
        OUTPUT_GATE_METRICS,
    )
    write_table(
        gate_decisions,
        OUTPUT_GATE_DECISIONS,
    )
    write_table(
        technical_covariates,
        OUTPUT_TECHNICAL_COVARIATES,
    )
    write_json(
        technical_contract,
        OUTPUT_TECHNICAL_CONTRACT,
    )
    write_json(
        external_benchmark_addendum,
        OUTPUT_EXTERNAL_BENCHMARK_AMENDMENT,
    )
    write_json(
        novelty_contract,
        OUTPUT_NOVELTY_CONTRACT,
    )
    write_json(
        environment,
        OUTPUT_ENVIRONMENT,
    )

    if pilot_disclosure is not None:
        write_json(
            pilot_disclosure,
            OUTPUT_PILOT_DISCLOSURE,
        )

    generated_paths = [
        OUTPUT_ARCHITECTURE_SCENARIOS,
        OUTPUT_ARCHITECTURE_REPLICATES,
        OUTPUT_GATE_SCENARIOS,
        OUTPUT_GATE_REPLICATES,
        OUTPUT_COMBINED_ARCHITECTURE_SCENARIOS,
        OUTPUT_COMBINED_ARCHITECTURE_REPLICATES,
        OUTPUT_TRUTH_TAXONOMY,
        OUTPUT_GATE_METRICS,
        OUTPUT_GATE_DECISIONS,
        OUTPUT_TECHNICAL_COVARIATES,
        OUTPUT_TECHNICAL_CONTRACT,
        OUTPUT_EXTERNAL_BENCHMARK_AMENDMENT,
        OUTPUT_NOVELTY_CONTRACT,
        OUTPUT_ENVIRONMENT,
    ]

    if pilot_disclosure is not None:
        generated_paths.append(
            OUTPUT_PILOT_DISCLOSURE
        )

    architecture_total_scenarios = int(
        combined_architecture_scenarios.shape[0]
    )
    architecture_total_replicates = int(
        combined_architecture_replicates.shape[0]
    )
    gate_total_scenarios = int(
        gate_scenarios.shape[0]
    )
    gate_total_replicates = int(
        gate_replicates.shape[0]
    )

    if pilot_disclosure is None:
        freeze_status = (
            "FROZEN_BEFORE_SIMULATION_AND_MODEL_FITTING"
        )
        amendment_reason = (
            "Prospective additions were frozen before any simulation data were "
            "materialized and before any representation-model result was generated."
        )
        materialization_timing = (
            "after 05a1 freeze and before target-model benchmarking"
        )
    else:
        freeze_status = (
            "FROZEN_BEFORE_MODEL_FITTING_AFTER_DISCLOSED_PILOT_MATERIALIZATION"
        )
        amendment_reason = (
            "Methodological additions were frozen after a disclosed data-only 05b "
            "v1 pilot materialization but before any simulation model, TARGET "
            "representation model, biological analysis, or outcome analysis was "
            "fitted. The v1 pilot is excluded from confirmatory evaluation."
        )
        materialization_timing = (
            "05b v1 pilot materialized before 05a1 and excluded; all confirmatory "
            "v2 datasets are generated after the 05a1 freeze"
        )

    addendum = {
        "script_version": SCRIPT_VERSION,
        "addendum_version": ADDENDUM_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": freeze_status,
        "method": {
            "name": METHOD_NAME,
            "name_status": "provisional",
        },
        "scientific_state": {
            "outcome_loaded": False,
            "clinical_labels_loaded": False,
            "biological_annotations_loaded": False,
            "simulation_model_fitting_performed": False,
            "target_model_fitting_performed": False,
            "simulation_results_viewed": False,
            "target_representation_results_viewed": False,
            "simulation_data_materialized_before_addendum": bool(
                pilot_disclosure is not None
            ),
            "prospective_before_model_fitting": True,
            "prospective_before_simulation_materialization": bool(
                pilot_disclosure is None
            ),
        },
        "base_05a_reference": {
            "contract_path": relative_path(
                BASE_CONTRACT_FILE
            ),
            "contract_sha256": (
                base["contract_sha256"]
            ),
            "lock_path": relative_path(
                BASE_LOCK_FILE
            ),
            "lock_sha256": (
                base["lock_sha256"]
            ),
            "base_contract_status": (
                base["contract"].get(
                    "contract_status"
                )
            ),
        },
        "amendment_scope": {
            "base_05a_is_modified": False,
            "sections_extended": [
                "simulation scenario taxonomy",
                "factor-level and subspace-level truth recovery",
                "stability-gate operating characteristics",
                "technical batch-confound stress tests",
                "TARGET technical metadata preflight",
                "external method benchmark hierarchy",
                "novelty positioning guardrails",
            ],
            "reason": amendment_reason,
        },
        "architecture_screening": {
            "base_05a_scenarios": int(
                base["base_scenarios"].shape[0]
            ),
            "addendum_scenarios": int(
                architecture_scenarios.shape[0]
            ),
            "combined_scenarios": (
                architecture_total_scenarios
            ),
            "combined_replicates": (
                architecture_total_replicates
            ),
            "replicates_per_addendum_scenario": (
                ARCHITECTURE_SCREENING_REPLICATES
            ),
            "stability_resamples_per_dataset": (
                ARCHITECTURE_STABILITY_RESAMPLES
            ),
            "scenario_file": relative_path(
                OUTPUT_COMBINED_ARCHITECTURE_SCENARIOS
            ),
            "replicate_registry": relative_path(
                OUTPUT_COMBINED_ARCHITECTURE_REPLICATES
            ),
            "materialization_timing": materialization_timing,
        },
        "pre_addendum_05b_pilot": (
            {
                "present": False,
                "disclosure_file": "",
                "eligible_for_confirmatory_analysis": False,
            }
            if pilot_disclosure is None
            else {
                "present": True,
                "disclosure_file": relative_path(
                    OUTPUT_PILOT_DISCLOSURE
                ),
                "eligible_for_confirmatory_analysis": False,
                "n_locked_manifest_datasets": int(
                    pilot_disclosure[
                        "n_locked_manifest_datasets"
                    ]
                ),
                "n_unlocked_extra_files": int(
                    pilot_disclosure[
                        "n_unlocked_extra_files"
                    ]
                ),
                "interpretation": pilot_disclosure[
                    "interpretation"
                ],
            }
        ),
        "final_gate_calibration": {
            "execution_timing": (
                "only after the model architecture is frozen from the broad "
                "simulation benchmark"
            ),
            "n_scenarios": gate_total_scenarios,
            "n_planned_datasets": gate_total_replicates,
            "non_null_replicates": (
                FINAL_GATE_NON_NULL_REPLICATES
            ),
            "null_and_batch_replicates": (
                FINAL_GATE_NULL_REPLICATES
            ),
            "stability_resamples_per_dataset": (
                FINAL_GATE_STABILITY_RESAMPLES
            ),
            "scenario_file": relative_path(
                OUTPUT_GATE_SCENARIOS
            ),
            "replicate_registry": relative_path(
                OUTPUT_GATE_REPLICATES
            ),
            "confidence_intervals": (
                "Wilson 95% intervals for binomial operating properties and "
                "bootstrap percentile intervals for continuous metrics"
            ),
            "threshold_retuning_after_results": False,
        },
        "truth_matching": {
            "factor_alignment": (
                "Hungarian matching followed by sign alignment"
            ),
            "factor_truth_match_absolute_correlation": (
                TRUTH_FACTOR_MATCH_ABS_CORRELATION
            ),
            "subspace_alignment": (
                "principal angles and normalized projection-matrix distance"
            ),
            "rotational_rule": (
                "When individual factors are not identifiable but the latent "
                "subspace is stable, retain a stable subspace label and forbid "
                "individual factor claims."
            ),
            "view_support_truth": (
                "ALL3, RNA_METH, RNA_CNV, METH_CNV, or private"
            ),
        },
        "operating_targets": OPERATING_TARGETS,
        "technical_confound": {
            "simulation_truth_available": True,
            "batch_truth_match_absolute_correlation": (
                TRUTH_BATCH_MATCH_ABS_CORRELATION
            ),
            "real_data_preflight_contract": relative_path(
                OUTPUT_TECHNICAL_CONTRACT
            ),
            "candidate_covariate_registry": relative_path(
                OUTPUT_TECHNICAL_COVARIATES
            ),
            "technical_association_thresholds": (
                TECHNICAL_ASSOCIATION_THRESHOLDS
            ),
            "technical_family_is_separate_multiplicity_family": True,
            "stable_factor_assumed_biological": False,
        },
        "external_benchmark_amendment": relative_path(
            OUTPUT_EXTERNAL_BENCHMARK_AMENDMENT
        ),
        "novelty_positioning_contract": relative_path(
            OUTPUT_NOVELTY_CONTRACT
        ),
        "generator_requirement": {
            "required_version": GENERATOR_REQUIRED_VERSION,
            "current_05b_v1_full_generation_allowed": False,
            "preexisting_05b_v1_pilot_allowed_for_primary_results": False,
            "required_v2_output_root": (
                "paper5/simulations/sable_mo_v2"
            ),
            "next_required_generator": (
                "05b_generate_small_n_multiomics_simulations.py v2 reading "
                "both 05a and 05a1 registries and writing to a new v2 root"
            ),
        },
        "guardrails": [
            "05a remains immutable and is referenced by hash.",
            "Any pre-addendum 05b v1 data-only materialization is disclosed and excluded from confirmatory analysis.",
            "No simulation model was fitted before this addendum was frozen.",
            "No TARGET representation model was fitted before this addendum was frozen.",
            "No outcome, clinical label, or biological annotation is used in model selection.",
            "A stable factor is not automatically interpreted as biological.",
            "Factor-level failure under rotational ambiguity does not imply subspace failure.",
            "Technical batch factors must be flagged rather than promoted as biology.",
            "Primary stability thresholds cannot be retuned after simulation results are viewed.",
            "Final gate calibration is performed only for the architecture frozen after broad screening.",
            "MLOmics Top features are forbidden because they use label-based ANOVA selection.",
            "No first-method claim is allowed without a final systematic novelty search.",
        ],
        "generated_artifacts": [
            relative_path(path)
            for path in generated_paths
        ],
    }

    write_json(
        addendum,
        OUTPUT_ADDENDUM,
    )
    generated_paths.append(
        OUTPUT_ADDENDUM
    )

    readme_lines = [
        "Paper 5 prospective simulation and technical-confound addendum",
        "=" * 76,
        "",
        f"Script version: {SCRIPT_VERSION}",
        f"Created UTC: {addendum['created_at_utc']}",
        "",
        "Status",
        "------",
        f"Freeze status: {freeze_status}",
        "Frozen before simulation model fitting: yes",
        "Frozen before TARGET representation fitting: yes",
        (
            "Simulation data materialized before addendum: yes; disclosed "
            "05b v1 pilot excluded from confirmatory analyses"
            if pilot_disclosure is not None
            else "Simulation data materialized before addendum: no"
        ),
        "Outcome loaded: no",
        "Biological annotation used for selection: no",
        "",
        "Why this addendum exists",
        "------------------------",
        (
            "The base 05a contract is retained unchanged. This prospective "
            "addendum adds the realistic n=62 operating point, rotationally "
            "ambiguous factors, partial sharing, correlated-private nulls, "
            "structured missingness, sparse/dense loadings, and technical "
            "batch-confound scenarios."
        ),
        "",
        "Two-stage simulation policy",
        "---------------------------",
        (
            f"Architecture screening: {architecture_total_scenarios} combined "
            f"scenarios and {architecture_total_replicates} datasets."
        ),
        (
            f"Final gate calibration: {gate_total_scenarios} scenarios and "
            f"{gate_total_replicates} planned datasets, executed only after "
            "the architecture is frozen."
        ),
        "",
        "Stability interpretation",
        "------------------------",
        (
            "Factor-level recovery uses Hungarian matching and sign alignment."
        ),
        (
            "Subspace-level recovery uses principal angles and projection distance."
        ),
        (
            "Rotationally stable subspaces may be retained without assigning "
            "stable identities to individual factors."
        ),
        "",
        "Technical-confound rule",
        "-----------------------",
        (
            "Reproducibility does not establish biological meaning. Stable "
            "factors are audited against usable sample-level technical metadata "
            "after representation freeze and before primary biological claims."
        ),
        "",
        "Next required scripts",
        "---------------------",
        "04d_target_technical_metadata_preflight.py",
        "05b_generate_small_n_multiomics_simulations.py v2",
        "",
        "Do not use or rerun 05b v1 for confirmatory analyses after creating this addendum.",
        "The required v2 generator must write to paper5/simulations/sable_mo_v2.",
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
            key=lambda value: str(value),
        )
    }

    lock = {
        "script_version": SCRIPT_VERSION,
        "addendum_version": ADDENDUM_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": freeze_status,
        "addendum_path": relative_path(
            OUTPUT_ADDENDUM
        ),
        "addendum_sha256": sha256_file(
            OUTPUT_ADDENDUM
        ),
        "base_05a_contract_sha256": (
            base["contract_sha256"]
        ),
        "base_05a_lock_sha256": (
            base["lock_sha256"]
        ),
        "generated_files": generated_file_records,
        "rewrite_policy": (
            "This no-argument script never rewrites a frozen addendum. Any future "
            "scientifically justified amendment must use a new versioned script "
            "and preserve the current locked contract. A disclosed data-only "
            "05b v1 pilot remains excluded from confirmatory analyses."
        ),
    }

    write_json(
        lock,
        OUTPUT_LOCK,
    )

    print()
    print("-" * 88)
    print(
        "Prospective simulation extension"
    )
    print("-" * 88)
    print(
        f"05a base scenarios: {base['base_scenarios'].shape[0]}"
    )
    print(
        f"05a1 architecture scenarios: {architecture_scenarios.shape[0]}"
    )
    print(
        f"Combined architecture scenarios: {architecture_total_scenarios}"
    )
    print(
        f"Combined architecture datasets: {architecture_total_replicates}"
    )
    print(
        f"Final gate-calibration scenarios: {gate_total_scenarios}"
    )
    print(
        f"Final gate-calibration datasets: {gate_total_replicates}"
    )

    print()
    print("-" * 88)
    print(
        "Frozen methodological additions"
    )
    print("-" * 88)
    print(
        "TARGET-like operating point n=62: INCLUDED"
    )
    print(
        "Factor-level and subspace-level recovery: INCLUDED"
    )
    print(
        "Partial-sharing truth: INCLUDED"
    )
    print(
        "Correlated-private null: INCLUDED"
    )
    print(
        "Structured missingness: INCLUDED"
    )
    print(
        "Sparse/dense loading regimes: INCLUDED"
    )
    print(
        "Technical batch-confound family: INCLUDED"
    )
    print(
        "Post-result threshold retuning: FORBIDDEN"
    )

    print()
    print("=" * 88)
    print(
        "Paper 5 simulation and technical-confound addendum: FROZEN"
    )
    print("=" * 88)
    print(
        f"Addendum: {relative_path(OUTPUT_ADDENDUM)}"
    )
    print(
        f"Lock: {relative_path(OUTPUT_LOCK)}"
    )
    print(
        "Pre-addendum 05b v1 pilot: "
        + ("DISCLOSED / EXCLUDED" if pilot_disclosure is not None else "ABSENT")
    )
    print(
        "Full 05b v1 generation permitted: NO"
    )
    print(
        "Next simulation generator requirement: v2"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
