
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05d1-freeze-neural-simulation-protocol-v1-no-cli"
CONTRACT_VERSION = "paper5-neural-simulation-protocol-v1"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# ---------------------------------------------------------------------
# Frozen upstream artifacts
# ---------------------------------------------------------------------

CONTRACT_05A_DIR = PAPER5_ROOT / "method_contract" / "05a"
CONTRACT_05A = CONTRACT_05A_DIR / "ai_method_contract.json"
LOCK_05A = CONTRACT_05A_DIR / "ai_method_contract_lock.json"
MODEL_REGISTRY = CONTRACT_05A_DIR / "model_family_registry.csv"
ABLATION_REGISTRY = CONTRACT_05A_DIR / "sable_mo_ablation_registry.csv"
ARCHITECTURE_PROFILES = CONTRACT_05A_DIR / "sable_mo_architecture_profiles.csv"
REGULARIZATION_PROFILES = CONTRACT_05A_DIR / "sable_mo_regularization_profiles.csv"
SELECTION_POLICY = CONTRACT_05A_DIR / "model_selection_policy.json"
FAILURE_GATES = CONTRACT_05A_DIR / "failure_gate_registry.csv"

CONTRACT_05A1_DIR = PAPER5_ROOT / "method_contract" / "05a1"
CONTRACT_05A1 = (
    CONTRACT_05A1_DIR
    / "simulation_and_technical_confound_addendum.json"
)
LOCK_05A1 = (
    CONTRACT_05A1_DIR
    / "simulation_and_technical_confound_addendum_lock.json"
)

SIMULATION_MANIFEST = (
    PAPER5_ROOT
    / "method_contract"
    / "05b_v2"
    / "architecture_screening_data_manifest.csv"
)
SIMULATION_LOCK = (
    PAPER5_ROOT
    / "method_contract"
    / "05b_v2"
    / "simulation_generation_v2_lock.json"
)

LINEAR_SPLITS = (
    PAPER5_ROOT
    / "results"
    / "simulation_benchmark"
    / "05c1_linear_baselines"
    / "linear_simulation_split_registry.csv.gz"
)
LINEAR_RESULTS_MANIFEST = (
    PAPER5_ROOT
    / "method_contract"
    / "05c1"
    / "linear_baseline_results_manifest.json"
)
LINEAR_LOCK = (
    PAPER5_ROOT
    / "method_contract"
    / "05c1"
    / "linear_simulation_baseline_lock.json"
)

MOFA_RESULTS_MANIFEST = (
    PAPER5_ROOT
    / "method_contract"
    / "05c2b"
    / "mofa_baseline_results_manifest.json"
)
MOFA_RESULTS_LOCK = (
    PAPER5_ROOT
    / "method_contract"
    / "05c2b"
    / "mofa_simulation_baseline_lock.json"
)

NEURAL_ENV_CONTRACT = (
    PAPER5_ROOT
    / "method_contract"
    / "05d0"
    / "neural_environment_contract.json"
)
NEURAL_ENV_LOCK = (
    PAPER5_ROOT
    / "method_contract"
    / "05d0"
    / "neural_environment_lock.json"
)

FINAL_GATE_DATA_ROOT = (
    PAPER5_ROOT
    / "simulations"
    / "sable_mo_v2"
    / "final_gate_calibration"
)

FUTURE_RESULT_PATHS = (
    PAPER5_ROOT / "method_contract" / "05d2",
    PAPER5_ROOT / "method_contract" / "05d3",
    PAPER5_ROOT / "method_contract" / "05d4",
    PAPER5_ROOT / "method_contract" / "05d5",
    PAPER5_ROOT / "method_contract" / "05e",
    PAPER5_ROOT / "results" / "simulation_benchmark" / "05d_neural",
    PAPER5_ROOT / "results" / "representation_benchmark",
)

# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05d1"

OUTPUT_PROFILE_GRID = (
    OUTPUT_DIR
    / "neural_architecture_regularization_profile_grid.csv"
)
OUTPUT_DATASET_ROLES = (
    OUTPUT_DIR
    / "neural_simulation_dataset_role_registry.csv"
)
OUTPUT_DEVELOPMENT_PARTITIONS = (
    OUTPUT_DIR
    / "neural_profile_development_partition_registry.csv"
)
OUTPUT_PROFILE_SEEDS = (
    OUTPUT_DIR
    / "neural_profile_screen_fit_seed_registry.csv"
)
OUTPUT_PROFILE_STABILITY = (
    OUTPUT_DIR
    / "neural_profile_stability_resample_registry.csv"
)
OUTPUT_CONFIRMATORY_SEEDS = (
    OUTPUT_DIR
    / "neural_confirmatory_ablation_fit_seed_registry.csv"
)
OUTPUT_CONFIRMATORY_STABILITY = (
    OUTPUT_DIR
    / "neural_confirmatory_stability_resample_registry.csv"
)
OUTPUT_LOSS_CONTRACT = (
    OUTPUT_DIR
    / "neural_loss_and_architecture_contract.json"
)
OUTPUT_SELECTION_CONTRACT = (
    OUTPUT_DIR
    / "neural_profile_selection_contract.json"
)
OUTPUT_PROTOCOL = (
    OUTPUT_DIR
    / "neural_simulation_evaluation_protocol.json"
)
OUTPUT_README = (
    OUTPUT_DIR
    / "neural_simulation_protocol_README.txt"
)
OUTPUT_LOCK = (
    OUTPUT_DIR
    / "neural_simulation_protocol_lock.json"
)

# ---------------------------------------------------------------------
# Prospective neural constants
# ---------------------------------------------------------------------

EXPECTED_SCENARIOS = 36
EXPECTED_DATASETS = 720
REPLICATES_PER_SCENARIO = 20

DEVELOPMENT_REPLICATES = tuple(
    range(1, 11)
)
CONFIRMATORY_REPLICATES = tuple(
    range(11, 21)
)

EXPECTED_DEVELOPMENT_DATASETS = (
    EXPECTED_SCENARIOS
    * len(
        DEVELOPMENT_REPLICATES
    )
)
EXPECTED_CONFIRMATORY_DATASETS = (
    EXPECTED_SCENARIOS
    * len(
        CONFIRMATORY_REPLICATES
    )
)

# Development profile selection:
# use one frozen stochastic seed because the required 20-resample
# stability analysis is the principal stochastic robustness layer.
PROFILE_SCREEN_BASE_SEED = 1103

# Confirmatory V0-V5 comparison uses two frozen seeds from the
# original 05a INNER_MODEL_SEEDS.
CONFIRMATORY_MODEL_BASE_SEEDS = (
    1103,
    2207,
)

# Profile-development split reuses exact 05c1 folds:
# fold 1 = untouched validation,
# fold 2 = early stopping,
# folds 3-5 = fit.
PROFILE_VALIDATION_FOLD = 1
PROFILE_EARLY_STOP_FOLD = 2
PROFILE_FIT_FOLDS = (
    3,
    4,
    5,
)

ARCHITECTURE_STABILITY_RESAMPLES = 20
STABILITY_SUBSAMPLE_FRACTION = 0.80

TRUTH_FACTOR_MATCH_ABS_CORRELATION = 0.50
TRUTH_BATCH_MATCH_ABS_CORRELATION = 0.50

NEURAL_VARIANTS = (
    "V0_BALANCED_AE",
    "V1_SHARED_PRIVATE_AE",
    "V2_CONTRASTIVE_SPAE",
    "V3_DROPOUT_SPAE",
    "V4_SABLE_MO",
    "V5_SABLE_VAE",
)

PRIMARY_ABLATION_VARIANTS = (
    "V0_BALANCED_AE",
    "V1_SHARED_PRIVATE_AE",
    "V2_CONTRASTIVE_SPAE",
    "V3_DROPOUT_SPAE",
    "V4_SABLE_MO",
)

VAE_VARIANT = "V5_SABLE_VAE"

GRADIENT_CLIP_NORM = 5.0
MINIMUM_NULL_RECONSTRUCTION_MSE = 1e-8
MINIMUM_NUMERICAL_VARIANCE = 1e-10

# Fixed exploratory VAE extension. It is not tuned on TARGET or on
# the simulation profile grid.
VAE_KL_FINAL_WEIGHT = 0.01
VAE_KL_WARMUP_FRACTION = 0.25

# A stability match is counted as present in an individual resample
# if the aligned factor correlation exceeds the already-frozen
# simulation truth correspondence threshold.
STABILITY_RESAMPLE_MATCH_THRESHOLD = (
    TRUTH_FACTOR_MATCH_ABS_CORRELATION
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def clean_text(
    value: Any,
) -> str:
    if value is None:
        return ""

    text = str(
        value
    ).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "null",
        "na",
    }:
        return ""

    return text


def parse_bool(
    value: Any,
) -> bool:
    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return bool(
            value
        )

    return clean_text(
        value
    ).lower() in {
        "true",
        "1",
        "yes",
    }


def relative_path(
    path: Path,
) -> str:
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


def resolve_project_path(
    value: Any,
) -> Path:
    path = Path(
        clean_text(
            value
        )
    )

    if not path.is_absolute():
        path = (
            PROJECT_ROOT
            / path
        )

    return path


def require_file(
    path: Path,
) -> None:
    if (
        not path.exists()
        or not path.is_file()
    ):
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                4 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def load_json(
    path: Path,
) -> dict[str, Any]:
    require_file(
        path
    )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_json_atomic(
    value: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, temporary_name = (
        tempfile.mkstemp(
            prefix=(
                path.stem
                + "."
            ),
            suffix=".tmp.json",
            dir=str(
                path.parent
            ),
        )
    )

    os.close(
        descriptor
    )

    temporary = Path(
        temporary_name
    )

    try:
        temporary.write_text(
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        os.replace(
            temporary,
            path,
        )

    finally:
        if temporary.exists():
            temporary.unlink()


def stable_seed(
    master_seed: int,
    *parts: Any,
) -> int:
    digest = hashlib.sha256()

    digest.update(
        str(
            master_seed
        ).encode(
            "utf-8"
        )
    )

    for part in parts:
        digest.update(
            b"|"
        )
        digest.update(
            str(
                part
            ).encode(
                "utf-8"
            )
        )

    value = int.from_bytes(
        digest.digest()[
            :8
        ],
        byteorder="big",
        signed=False,
    )

    return int(
        value
        % (
            2**32
            - 1
        )
    )


def is_nonempty_path(
    path: Path,
) -> bool:
    if not path.exists():
        return False

    if path.is_file():
        return (
            path.stat().st_size
            > 0
        )

    return any(
        item.is_file()
        for item in path.rglob(
            "*"
        )
    )


def verify_generated_files(
    lock: dict[str, Any],
    label: str,
) -> None:
    failures = []

    for relative_name, record in lock.get(
        "generated_files",
        {},
    ).items():
        candidate = resolve_project_path(
            relative_name
        )

        if not candidate.exists():
            failures.append(
                f"missing: {relative_name}"
            )
            continue

        expected_size = record.get(
            "size_bytes"
        )

        if (
            expected_size
            is not None
            and int(
                candidate.stat().st_size
            )
            != int(
                expected_size
            )
        ):
            failures.append(
                f"size mismatch: {relative_name}"
            )
            continue

        expected_hash = clean_text(
            record.get(
                "sha256"
            )
        )

        if (
            expected_hash
            and sha256_file(
                candidate
            )
            != expected_hash
        ):
            failures.append(
                f"SHA256 mismatch: {relative_name}"
            )

    if failures:
        raise RuntimeError(
            f"{label} generated-file verification failed:\n"
            + "\n".join(
                f"- {value}"
                for value in failures
            )
        )


def validate_future_state() -> None:
    found = [
        relative_path(
            path
        )
        for path in FUTURE_RESULT_PATHS
        if is_nonempty_path(
            path
        )
    ]

    if found:
        raise RuntimeError(
            "05d1 must be frozen before any 05d2+ neural result exists. Found:\n- "
            + "\n- ".join(
                found
            )
        )

    if (
        FINAL_GATE_DATA_ROOT.exists()
        and any(
            item.is_file()
            for item in FINAL_GATE_DATA_ROOT.rglob(
                "*"
            )
        )
    ):
        raise RuntimeError(
            "Final gate-calibration simulations were materialized before architecture freeze."
        )


# ---------------------------------------------------------------------
# Upstream validation
# ---------------------------------------------------------------------


def verify_upstream() -> dict[str, Any]:
    for path in [
        CONTRACT_05A,
        LOCK_05A,
        MODEL_REGISTRY,
        ABLATION_REGISTRY,
        ARCHITECTURE_PROFILES,
        REGULARIZATION_PROFILES,
        SELECTION_POLICY,
        FAILURE_GATES,
        CONTRACT_05A1,
        LOCK_05A1,
        SIMULATION_MANIFEST,
        SIMULATION_LOCK,
        LINEAR_SPLITS,
        LINEAR_RESULTS_MANIFEST,
        LINEAR_LOCK,
        MOFA_RESULTS_MANIFEST,
        MOFA_RESULTS_LOCK,
        NEURAL_ENV_CONTRACT,
        NEURAL_ENV_LOCK,
    ]:
        require_file(
            path
        )

    validate_future_state()

    contract_05a = load_json(
        CONTRACT_05A
    )
    lock_05a = load_json(
        LOCK_05A
    )

    expected_05a = clean_text(
        lock_05a.get(
            "contract_sha256"
        )
    )

    if (
        not expected_05a
        or sha256_file(
            CONTRACT_05A
        )
        != expected_05a
    ):
        raise RuntimeError(
            "05a contract verification failed."
        )

    verify_generated_files(
        lock_05a,
        "05a",
    )

    addendum = load_json(
        CONTRACT_05A1
    )
    lock_05a1 = load_json(
        LOCK_05A1
    )

    expected_05a1 = clean_text(
        lock_05a1.get(
            "addendum_sha256"
        )
    )

    if (
        not expected_05a1
        or sha256_file(
            CONTRACT_05A1
        )
        != expected_05a1
    ):
        raise RuntimeError(
            "05a1 addendum verification failed."
        )

    verify_generated_files(
        lock_05a1,
        "05a1",
    )

    architecture_screening = (
        addendum.get(
            "architecture_screening",
            {}
        )
    )

    if int(
        architecture_screening.get(
            "combined_scenarios",
            -1,
        )
    ) != EXPECTED_SCENARIOS:
        raise RuntimeError(
            "05a1 architecture scenario count differs from 36."
        )

    if int(
        architecture_screening.get(
            "combined_replicates",
            -1,
        )
    ) != EXPECTED_DATASETS:
        raise RuntimeError(
            "05a1 architecture dataset count differs from 720."
        )

    if int(
        architecture_screening.get(
            "stability_resamples_per_dataset",
            -1,
        )
    ) != ARCHITECTURE_STABILITY_RESAMPLES:
        raise RuntimeError(
            "05a1 stability-resample count differs from 20."
        )

    truth_matching = addendum.get(
        "truth_matching",
        {},
    )

    if float(
        truth_matching.get(
            "factor_truth_match_absolute_correlation",
            float(
                "nan"
            ),
        )
    ) != TRUTH_FACTOR_MATCH_ABS_CORRELATION:
        raise RuntimeError(
            "05a1 biological truth-match threshold differs from 0.50."
        )

    if float(
        addendum.get(
            "technical_confound",
            {},
        ).get(
            "batch_truth_match_absolute_correlation",
            float(
                "nan"
            ),
        )
    ) != TRUTH_BATCH_MATCH_ABS_CORRELATION:
        raise RuntimeError(
            "05a1 technical truth-match threshold differs from 0.50."
        )

    simulation_lock = load_json(
        SIMULATION_LOCK
    )

    if clean_text(
        simulation_lock.get(
            "status"
        )
    ) != (
        "FROZEN_ARCHITECTURE_SCREENING_INPUTS_BEFORE_MODEL_FITTING"
    ):
        raise RuntimeError(
            "Unexpected 05b v2 lock status."
        )

    linear_lock = load_json(
        LINEAR_LOCK
    )

    if clean_text(
        linear_lock.get(
            "status"
        )
    ) != (
        "FROZEN_LINEAR_BASELINE_RESULTS_BEFORE_MOFA_AND_NEURAL_FITTING"
    ):
        raise RuntimeError(
            "Unexpected 05c1 lock status."
        )

    mofa_lock = load_json(
        MOFA_RESULTS_LOCK
    )

    if clean_text(
        mofa_lock.get(
            "status"
        )
    ) != (
        "FROZEN_MOFA_BASELINE_RESULTS_BEFORE_NEURAL_FITTING_AND_ARCHITECTURE_SELECTION"
    ):
        raise RuntimeError(
            "Unexpected 05c2b lock status."
        )

    mofa_manifest = load_json(
        MOFA_RESULTS_MANIFEST
    )

    mofa_scope = mofa_manifest.get(
        "scope",
        {},
    )

    if (
        int(
            mofa_scope.get(
                "n_fit_pass",
                -1,
            )
        )
        != 17280
        or int(
            mofa_scope.get(
                "n_fit_fail",
                -1,
            )
        )
        != 0
    ):
        raise RuntimeError(
            "05c2b fit counts do not match 17,280 PASS / 0 FAIL."
        )

    neural_env = load_json(
        NEURAL_ENV_CONTRACT
    )
    neural_env_lock = load_json(
        NEURAL_ENV_LOCK
    )

    if clean_text(
        neural_env_lock.get(
            "status"
        )
    ) != (
        "FROZEN_NEURAL_RUNTIME_ENVIRONMENT_BEFORE_NEURAL_MODEL_FITTING"
    ):
        raise RuntimeError(
            "Unexpected 05d0 lock status."
        )

    expected_env_hash = clean_text(
        neural_env_lock.get(
            "contract",
            {},
        ).get(
            "sha256"
        )
    )

    if (
        not expected_env_hash
        or sha256_file(
            NEURAL_ENV_CONTRACT
        )
        != expected_env_hash
    ):
        raise RuntimeError(
            "05d0 neural environment contract hash mismatch."
        )

    verify_generated_files(
        neural_env_lock,
        "05d0",
    )

    state = neural_env.get(
        "scientific_state",
        {},
    )

    forbidden_true = [
        "outcomes_loaded",
        "clinical_labels_loaded",
        "biological_annotations_loaded",
        "TARGET_model_fitting_performed",
        "neural_simulation_model_fitting_performed",
        "architecture_selected",
        "stability_gate_thresholds_retuned",
        "final_gate_calibration_data_used",
    ]

    active = [
        key
        for key in forbidden_true
        if bool(
            state.get(
                key,
                False,
            )
        )
    ]

    if active:
        raise RuntimeError(
            "05d0 pre-neural state violated: "
            + ", ".join(
                active
            )
        )

    runtime = neural_env.get(
        "runtime_policy",
        {},
    )

    required_runtime = {
        "device_policy": (
            "CUDA required for confirmatory neural benchmark"
        ),
        "dtype": "float32",
        "automatic_mixed_precision": False,
        "torch_compile": False,
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "allow_tf32_matmul": False,
        "allow_tf32_cudnn": False,
        "dataloader_workers_primary": 0,
    }

    for key, expected in required_runtime.items():
        if runtime.get(
            key
        ) != expected:
            raise RuntimeError(
                f"05d0 runtime policy mismatch for {key}."
            )

    architecture = pd.read_csv(
        ARCHITECTURE_PROFILES,
        dtype=str,
    ).fillna(
        ""
    )

    regularization = pd.read_csv(
        REGULARIZATION_PROFILES,
        dtype=str,
    ).fillna(
        ""
    )

    ablations = pd.read_csv(
        ABLATION_REGISTRY,
        dtype=str,
    ).fillna(
        ""
    )

    models = pd.read_csv(
        MODEL_REGISTRY,
        dtype=str,
    ).fillna(
        ""
    )

    selection = load_json(
        SELECTION_POLICY
    )

    simulation_manifest = pd.read_csv(
        SIMULATION_MANIFEST,
        dtype=str,
        low_memory=False,
    ).fillna(
        ""
    )

    split_table = pd.read_csv(
        LINEAR_SPLITS,
        dtype=str,
        low_memory=False,
    ).fillna(
        ""
    )

    return {
        "contract_05a": contract_05a,
        "addendum": addendum,
        "neural_env": neural_env,
        "architecture": architecture,
        "regularization": regularization,
        "ablations": ablations,
        "models": models,
        "selection": selection,
        "simulation_manifest": (
            simulation_manifest
        ),
        "split_table": split_table,
        "hashes": {
            relative_path(
                path
            ): sha256_file(
                path
            )
            for path in [
                CONTRACT_05A,
                LOCK_05A,
                MODEL_REGISTRY,
                ABLATION_REGISTRY,
                ARCHITECTURE_PROFILES,
                REGULARIZATION_PROFILES,
                SELECTION_POLICY,
                FAILURE_GATES,
                CONTRACT_05A1,
                LOCK_05A1,
                SIMULATION_MANIFEST,
                SIMULATION_LOCK,
                LINEAR_SPLITS,
                LINEAR_RESULTS_MANIFEST,
                LINEAR_LOCK,
                MOFA_RESULTS_MANIFEST,
                MOFA_RESULTS_LOCK,
                NEURAL_ENV_CONTRACT,
                NEURAL_ENV_LOCK,
            ]
        },
    }


# ---------------------------------------------------------------------
# Validation of the frozen 05a neural design
# ---------------------------------------------------------------------


def validate_05a_neural_design(
    upstream: dict[str, Any],
) -> None:
    architecture = upstream[
        "architecture"
    ].copy()

    regularization = upstream[
        "regularization"
    ].copy()

    ablations = upstream[
        "ablations"
    ].copy()

    models = upstream[
        "models"
    ].copy()

    expected_architectures = {
        "compact": {
            "rna_pca_components": 10,
            "methylation_pca_components": 10,
            "cnv_pca_components": 10,
            "shared_latent_dim": 5,
            "private_latent_dim_per_view": 2,
            "encoder_hidden_dim": 32,
            "decoder_hidden_dim": 32,
            "maximum_epochs": 400,
            "early_stopping_patience": 30,
            "complexity_rank": 1,
        },
        "standard": {
            "rna_pca_components": 20,
            "methylation_pca_components": 20,
            "cnv_pca_components": 20,
            "shared_latent_dim": 10,
            "private_latent_dim_per_view": 5,
            "encoder_hidden_dim": 64,
            "decoder_hidden_dim": 64,
            "maximum_epochs": 500,
            "early_stopping_patience": 40,
            "complexity_rank": 2,
        },
        "expanded": {
            "rna_pca_components": 30,
            "methylation_pca_components": 30,
            "cnv_pca_components": 30,
            "shared_latent_dim": 15,
            "private_latent_dim_per_view": 5,
            "encoder_hidden_dim": 96,
            "decoder_hidden_dim": 96,
            "maximum_epochs": 600,
            "early_stopping_patience": 50,
            "complexity_rank": 3,
        },
    }

    if set(
        architecture[
            "architecture_profile"
        ]
    ) != set(
        expected_architectures
    ):
        raise RuntimeError(
            "05a architecture-profile names differ from compact/standard/expanded."
        )

    for profile, expected in expected_architectures.items():
        row = architecture[
            architecture[
                "architecture_profile"
            ].eq(
                profile
            )
        ].iloc[
            0
        ]

        for column, value in expected.items():
            observed = int(
                float(
                    row[
                        column
                    ]
                )
            )

            if observed != value:
                raise RuntimeError(
                    f"05a {profile} {column} changed: {observed} vs {value}."
                )

        if clean_text(
            row[
                "activation"
            ]
        ) != "GELU":
            raise RuntimeError(
                f"05a {profile} activation is not GELU."
            )

        if not parse_bool(
            row[
                "layer_normalization"
            ]
        ):
            raise RuntimeError(
                f"05a {profile} no longer requires LayerNorm."
            )

        if not parse_bool(
            row[
                "full_batch_training"
            ]
        ):
            raise RuntimeError(
                f"05a {profile} no longer requires full-batch training."
            )

        if float(
            row[
                "early_stopping_min_delta"
            ]
        ) != 1e-4:
            raise RuntimeError(
                f"05a {profile} early-stopping min delta changed."
            )

    expected_regularization = {
        "weak",
        "standard",
        "strong",
    }

    if set(
        regularization[
            "regularization_profile"
        ]
    ) != expected_regularization:
        raise RuntimeError(
            "05a regularization profiles differ from weak/standard/strong."
        )

    observed_variants = set(
        ablations[
            "variant_id"
        ]
    )

    if observed_variants != set(
        NEURAL_VARIANTS
    ):
        raise RuntimeError(
            "05a V0-V5 ablation registry changed."
        )

    registry_variants = set(
        models.loc[
            models[
                "model_id"
            ].isin(
                NEURAL_VARIANTS
            ),
            "model_id",
        ]
    )

    if registry_variants != set(
        NEURAL_VARIANTS
    ):
        raise RuntimeError(
            "05a model registry does not contain all V0-V5 variants."
        )

    policy = upstream[
        "selection"
    ].get(
        "simulation_profile_selection",
        {},
    )

    if clean_text(
        policy.get(
            "profile_selected_for"
        )
    ) != "V4_SABLE_MO":
        raise RuntimeError(
            "05a no longer selects the neural profile for V4_SABLE_MO."
        )

    reused = set(
        policy.get(
            "profile_reused_by",
            []
        )
    )

    expected_reused = {
        "V0_BALANCED_AE",
        "V1_SHARED_PRIVATE_AE",
        "V2_CONTRASTIVE_SPAE",
        "V3_DROPOUT_SPAE",
        "V5_SABLE_VAE",
    }

    if reused != expected_reused:
        raise RuntimeError(
            "05a profile-reuse rule for neural ablations changed."
        )


# ---------------------------------------------------------------------
# Profile grid
# ---------------------------------------------------------------------


def build_profile_grid(
    architecture: pd.DataFrame,
    regularization: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, architecture_row in (
        architecture.sort_values(
            "complexity_rank",
            kind="mergesort",
        ).iterrows()
    ):
        for _, regularization_row in (
            regularization.sort_values(
                "complexity_rank",
                kind="mergesort",
            ).iterrows()
        ):
            arch_name = clean_text(
                architecture_row[
                    "architecture_profile"
                ]
            )

            reg_name = clean_text(
                regularization_row[
                    "regularization_profile"
                ]
            )

            arch_rank = int(
                float(
                    architecture_row[
                        "complexity_rank"
                    ]
                )
            )

            reg_rank = int(
                float(
                    regularization_row[
                        "complexity_rank"
                    ]
                )
            )

            rows.append(
                {
                    "profile_id": (
                        f"ARCH_{arch_name.upper()}__REG_{reg_name.upper()}"
                    ),
                    "architecture_profile": (
                        arch_name
                    ),
                    "regularization_profile": (
                        reg_name
                    ),
                    "architecture_complexity_rank": (
                        arch_rank
                    ),
                    "regularization_complexity_rank": (
                        reg_rank
                    ),
                    "combined_complexity_rank": (
                        arch_rank
                        + reg_rank
                    ),
                    **{
                        f"arch_{column}": (
                            architecture_row[
                                column
                            ]
                        )
                        for column in architecture.columns
                        if column
                        not in {
                            "architecture_profile",
                            "complexity_rank",
                        }
                    },
                    **{
                        f"reg_{column}": (
                            regularization_row[
                                column
                            ]
                        )
                        for column in regularization.columns
                        if column
                        not in {
                            "regularization_profile",
                            "complexity_rank",
                        }
                    },
                }
            )

    table = pd.DataFrame(
        rows
    )

    if table.shape[
        0
    ] != 9:
        raise RuntimeError(
            f"Expected 9 architecture x regularization profiles; found {table.shape[0]}."
        )

    if table[
        "profile_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate neural profile IDs."
        )

    return table.sort_values(
        [
            "combined_complexity_rank",
            "architecture_complexity_rank",
            "regularization_complexity_rank",
            "profile_id",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------
# Simulation development/confirmatory split
# ---------------------------------------------------------------------


def build_dataset_role_registry(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    table = manifest.copy()

    for column in [
        "replicate_index",
        "simulation_seed",
    ]:
        table[
            column
        ] = pd.to_numeric(
            table[
                column
            ],
            errors="raise",
        ).astype(
            "int64"
        )

    if table.shape[
        0
    ] != EXPECTED_DATASETS:
        raise RuntimeError(
            f"Expected {EXPECTED_DATASETS} simulation datasets; found {table.shape[0]}."
        )

    scenario_counts = (
        table.groupby(
            "scenario_id"
        )[
            "replicate_index"
        ].nunique()
    )

    if (
        scenario_counts.shape[
            0
        ]
        != EXPECTED_SCENARIOS
        or not scenario_counts.eq(
            REPLICATES_PER_SCENARIO
        ).all()
    ):
        raise RuntimeError(
            "Every scenario must contain exactly 20 unique replicates."
        )

    for scenario_id, part in table.groupby(
        "scenario_id"
    ):
        observed = sorted(
            part[
                "replicate_index"
            ].tolist()
        )

        if observed != list(
            range(
                1,
                21,
            )
        ):
            raise RuntimeError(
                f"{scenario_id}: replicate indices are not exactly 1..20."
            )

    table[
        "neural_dataset_role"
    ] = np.where(
        table[
            "replicate_index"
        ].isin(
            DEVELOPMENT_REPLICATES
        ),
        "profile_development",
        "confirmatory_ablation",
    )

    table[
        "eligible_for_neural_profile_selection"
    ] = table[
        "neural_dataset_role"
    ].eq(
        "profile_development"
    )

    table[
        "eligible_for_confirmatory_neural_ablation"
    ] = table[
        "neural_dataset_role"
    ].eq(
        "confirmatory_ablation"
    )

    development = int(
        table[
            "eligible_for_neural_profile_selection"
        ].sum()
    )

    confirmatory = int(
        table[
            "eligible_for_confirmatory_neural_ablation"
        ].sum()
    )

    if (
        development
        != EXPECTED_DEVELOPMENT_DATASETS
        or confirmatory
        != EXPECTED_CONFIRMATORY_DATASETS
    ):
        raise RuntimeError(
            "Neural development/confirmatory dataset split is not 360/360."
        )

    return table


# ---------------------------------------------------------------------
# Reuse exact 05c1 folds
# ---------------------------------------------------------------------


def parse_indices(
    text: str,
) -> list[int]:
    return [
        int(
            value
        )
        for value in clean_text(
            text
        ).split(
            ";"
        )
        if value
    ]


def index_string(
    values: list[int],
) -> str:
    return ";".join(
        str(
            int(
                value
            )
        )
        for value in values
    )


def build_split_lookup(
    split_table: pd.DataFrame,
) -> dict[
    tuple[
        str,
        int,
        int,
    ],
    dict[
        int,
        list[int],
    ],
]:
    required = {
        "scenario_id",
        "replicate_index",
        "simulation_seed",
        "fold_index",
        "test_indices",
    }

    missing = (
        required
        - set(
            split_table.columns
        )
    )

    if missing:
        raise RuntimeError(
            "05c1 split registry missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    lookup: dict[
        tuple[
            str,
            int,
            int,
        ],
        dict[
            int,
            list[int],
        ],
    ] = {}

    for _, row in split_table.iterrows():
        key = (
            clean_text(
                row[
                    "scenario_id"
                ]
            ),
            int(
                float(
                    row[
                        "replicate_index"
                    ]
                )
            ),
            int(
                float(
                    row[
                        "simulation_seed"
                    ]
                )
            ),
        )

        fold = int(
            float(
                row[
                    "fold_index"
                ]
            )
        )

        lookup.setdefault(
            key,
            {},
        )[
            fold
        ] = parse_indices(
            row[
                "test_indices"
            ]
        )

    if len(
        lookup
    ) != EXPECTED_DATASETS:
        raise RuntimeError(
            "05c1 split lookup does not contain 720 datasets."
        )

    for key, folds in lookup.items():
        if set(
            folds
        ) != {
            1,
            2,
            3,
            4,
            5,
        }:
            raise RuntimeError(
                f"05c1 fold coverage invalid for {key}."
            )

    return lookup


def build_development_partitions(
    roles: pd.DataFrame,
    split_lookup: dict[
        tuple[
            str,
            int,
            int,
        ],
        dict[
            int,
            list[int],
        ],
    ],
) -> pd.DataFrame:
    rows = []

    development = roles[
        roles[
            "neural_dataset_role"
        ].eq(
            "profile_development"
        )
    ].copy()

    for _, row in development.iterrows():
        key = (
            clean_text(
                row[
                    "scenario_id"
                ]
            ),
            int(
                row[
                    "replicate_index"
                ]
            ),
            int(
                row[
                    "simulation_seed"
                ]
            ),
        )

        folds = split_lookup[
            key
        ]

        validation = sorted(
            folds[
                PROFILE_VALIDATION_FOLD
            ]
        )

        early_stop = sorted(
            folds[
                PROFILE_EARLY_STOP_FOLD
            ]
        )

        fit = sorted(
            value
            for fold in PROFILE_FIT_FOLDS
            for value in folds[
                fold
            ]
        )

        fit_set = set(
            fit
        )
        early_set = set(
            early_stop
        )
        validation_set = set(
            validation
        )

        if (
            fit_set
            & early_set
            or fit_set
            & validation_set
            or early_set
            & validation_set
        ):
            raise RuntimeError(
                f"Development partition overlap for {key}."
            )

        n_samples = int(
            row[
                "n_samples"
            ]
        )

        covered = (
            fit_set
            | early_set
            | validation_set
        )

        if covered != set(
            range(
                n_samples
            )
        ):
            raise RuntimeError(
                f"Development partition does not cover all samples for {key}."
            )

        rows.append(
            {
                "scenario_id": key[
                    0
                ],
                "replicate_index": key[
                    1
                ],
                "simulation_seed": key[
                    2
                ],
                "n_samples": (
                    n_samples
                ),
                "fit_source_folds": (
                    "3;4;5"
                ),
                "early_stop_source_fold": (
                    PROFILE_EARLY_STOP_FOLD
                ),
                "validation_source_fold": (
                    PROFILE_VALIDATION_FOLD
                ),
                "n_fit": len(
                    fit
                ),
                "n_early_stop": len(
                    early_stop
                ),
                "n_validation": len(
                    validation
                ),
                "fit_indices": index_string(
                    fit
                ),
                "early_stop_indices": (
                    index_string(
                        early_stop
                    )
                ),
                "validation_indices": (
                    index_string(
                        validation
                    )
                ),
                "profile_selection_data": True,
                "confirmatory_ablation_data": False,
            }
        )

    table = pd.DataFrame(
        rows
    )

    if table.shape[
        0
    ] != EXPECTED_DEVELOPMENT_DATASETS:
        raise RuntimeError(
            "Expected 360 profile-development partitions."
        )

    return table


# ---------------------------------------------------------------------
# Deterministic seed registries
# ---------------------------------------------------------------------


def build_profile_fit_seeds(
    development: pd.DataFrame,
    profiles: pd.DataFrame,
    master_seed: int,
) -> pd.DataFrame:
    rows = []

    for _, dataset in development.iterrows():
        scenario_id = clean_text(
            dataset[
                "scenario_id"
            ]
        )

        replicate_index = int(
            dataset[
                "replicate_index"
            ]
        )

        simulation_seed = int(
            dataset[
                "simulation_seed"
            ]
        )

        for _, profile in profiles.iterrows():
            profile_id = clean_text(
                profile[
                    "profile_id"
                ]
            )

            fit_seed = stable_seed(
                master_seed,
                "05d2_profile_screen",
                scenario_id,
                replicate_index,
                simulation_seed,
                profile_id,
                PROFILE_SCREEN_BASE_SEED,
            )

            rows.append(
                {
                    "scenario_id": (
                        scenario_id
                    ),
                    "replicate_index": (
                        replicate_index
                    ),
                    "simulation_seed": (
                        simulation_seed
                    ),
                    "profile_id": (
                        profile_id
                    ),
                    "base_model_seed": (
                        PROFILE_SCREEN_BASE_SEED
                    ),
                    "fit_seed": (
                        fit_seed
                    ),
                    "early_stop_seed": (
                        fit_seed
                    ),
                    "refit_seed": (
                        fit_seed
                    ),
                    "device": "cuda:0",
                }
            )

    table = pd.DataFrame(
        rows
    )

    expected = (
        EXPECTED_DEVELOPMENT_DATASETS
        * profiles.shape[
            0
        ]
    )

    if table.shape[
        0
    ] != expected:
        raise RuntimeError(
            f"Expected {expected} profile-screen fit seed rows."
        )

    return table


def build_profile_stability_registry(
    development: pd.DataFrame,
    profiles: pd.DataFrame,
    master_seed: int,
) -> pd.DataFrame:
    rows = []

    for _, dataset in development.iterrows():
        scenario_id = clean_text(
            dataset[
                "scenario_id"
            ]
        )

        replicate_index = int(
            dataset[
                "replicate_index"
            ]
        )

        simulation_seed = int(
            dataset[
                "simulation_seed"
            ]
        )

        n_samples = int(
            dataset[
                "n_samples"
            ]
        )

        n_subsample = max(
            2,
            int(
                math.floor(
                    n_samples
                    * STABILITY_SUBSAMPLE_FRACTION
                )
            ),
        )

        for _, profile in profiles.iterrows():
            profile_id = clean_text(
                profile[
                    "profile_id"
                ]
            )

            reference_seed = stable_seed(
                master_seed,
                "05d2_profile_stability_reference",
                scenario_id,
                replicate_index,
                simulation_seed,
                profile_id,
            )

            for resample_index in range(
                1,
                ARCHITECTURE_STABILITY_RESAMPLES
                + 1,
            ):
                subsample_seed = stable_seed(
                    master_seed,
                    "05d2_profile_stability_subsample",
                    scenario_id,
                    replicate_index,
                    simulation_seed,
                    profile_id,
                    resample_index,
                )

                fit_seed = stable_seed(
                    master_seed,
                    "05d2_profile_stability_fit",
                    scenario_id,
                    replicate_index,
                    simulation_seed,
                    profile_id,
                    resample_index,
                )

                rows.append(
                    {
                        "scenario_id": (
                            scenario_id
                        ),
                        "replicate_index": (
                            replicate_index
                        ),
                        "simulation_seed": (
                            simulation_seed
                        ),
                        "profile_id": (
                            profile_id
                        ),
                        "reference_fit_seed": (
                            reference_seed
                        ),
                        "resample_index": (
                            resample_index
                        ),
                        "subsample_fraction": (
                            STABILITY_SUBSAMPLE_FRACTION
                        ),
                        "n_subsample": (
                            n_subsample
                        ),
                        "subsample_seed": (
                            subsample_seed
                        ),
                        "resample_fit_seed": (
                            fit_seed
                        ),
                        "device": "cuda:0",
                    }
                )

    table = pd.DataFrame(
        rows
    )

    expected = (
        EXPECTED_DEVELOPMENT_DATASETS
        * profiles.shape[
            0
        ]
        * ARCHITECTURE_STABILITY_RESAMPLES
    )

    if table.shape[
        0
    ] != expected:
        raise RuntimeError(
            f"Expected {expected} profile stability-resample rows."
        )

    return table


def build_confirmatory_fit_seeds(
    confirmatory: pd.DataFrame,
    master_seed: int,
) -> pd.DataFrame:
    rows = []

    for _, dataset in confirmatory.iterrows():
        scenario_id = clean_text(
            dataset[
                "scenario_id"
            ]
        )

        replicate_index = int(
            dataset[
                "replicate_index"
            ]
        )

        simulation_seed = int(
            dataset[
                "simulation_seed"
            ]
        )

        for variant in NEURAL_VARIANTS:
            for fold_index in range(
                1,
                6,
            ):
                for base_seed in (
                    CONFIRMATORY_MODEL_BASE_SEEDS
                ):
                    seed = stable_seed(
                        master_seed,
                        "05d4_confirmatory_ablation",
                        scenario_id,
                        replicate_index,
                        simulation_seed,
                        variant,
                        fold_index,
                        base_seed,
                    )

                    partition_seed = stable_seed(
                        master_seed,
                        "05d4_confirmatory_early_stop_partition",
                        scenario_id,
                        replicate_index,
                        simulation_seed,
                        variant,
                        fold_index,
                        base_seed,
                    )

                    rows.append(
                        {
                            "scenario_id": (
                                scenario_id
                            ),
                            "replicate_index": (
                                replicate_index
                            ),
                            "simulation_seed": (
                                simulation_seed
                            ),
                            "variant_id": (
                                variant
                            ),
                            "fold_index": (
                                fold_index
                            ),
                            "base_model_seed": (
                                base_seed
                            ),
                            "fit_seed": (
                                seed
                            ),
                            "refit_seed": (
                                seed
                            ),
                            "early_stop_partition_seed": (
                                partition_seed
                            ),
                            "device": "cuda:0",
                        }
                    )

    table = pd.DataFrame(
        rows
    )

    expected = (
        EXPECTED_CONFIRMATORY_DATASETS
        * len(
            NEURAL_VARIANTS
        )
        * 5
        * len(
            CONFIRMATORY_MODEL_BASE_SEEDS
        )
    )

    if table.shape[
        0
    ] != expected:
        raise RuntimeError(
            f"Expected {expected} confirmatory ablation fit seed rows."
        )

    return table


def build_confirmatory_stability_registry(
    confirmatory: pd.DataFrame,
    master_seed: int,
) -> pd.DataFrame:
    rows = []

    for _, dataset in confirmatory.iterrows():
        scenario_id = clean_text(
            dataset[
                "scenario_id"
            ]
        )

        replicate_index = int(
            dataset[
                "replicate_index"
            ]
        )

        simulation_seed = int(
            dataset[
                "simulation_seed"
            ]
        )

        n_samples = int(
            dataset[
                "n_samples"
            ]
        )

        n_subsample = max(
            2,
            int(
                math.floor(
                    n_samples
                    * STABILITY_SUBSAMPLE_FRACTION
                )
            ),
        )

        reference_seed = stable_seed(
            master_seed,
            "05d5_confirmatory_stability_reference",
            scenario_id,
            replicate_index,
            simulation_seed,
        )

        for resample_index in range(
            1,
            ARCHITECTURE_STABILITY_RESAMPLES
            + 1,
        ):
            rows.append(
                {
                    "scenario_id": (
                        scenario_id
                    ),
                    "replicate_index": (
                        replicate_index
                    ),
                    "simulation_seed": (
                        simulation_seed
                    ),
                    "variant_id": (
                        "V4_SABLE_MO"
                    ),
                    "profile_id": (
                        "FROZEN_BY_05D3"
                    ),
                    "reference_fit_seed": (
                        reference_seed
                    ),
                    "resample_index": (
                        resample_index
                    ),
                    "subsample_fraction": (
                        STABILITY_SUBSAMPLE_FRACTION
                    ),
                    "n_subsample": (
                        n_subsample
                    ),
                    "subsample_seed": stable_seed(
                        master_seed,
                        "05d5_confirmatory_stability_subsample",
                        scenario_id,
                        replicate_index,
                        simulation_seed,
                        resample_index,
                    ),
                    "resample_fit_seed": stable_seed(
                        master_seed,
                        "05d5_confirmatory_stability_fit",
                        scenario_id,
                        replicate_index,
                        simulation_seed,
                        resample_index,
                    ),
                    "device": "cuda:0",
                }
            )

    table = pd.DataFrame(
        rows
    )

    expected = (
        EXPECTED_CONFIRMATORY_DATASETS
        * ARCHITECTURE_STABILITY_RESAMPLES
    )

    if table.shape[
        0
    ] != expected:
        raise RuntimeError(
            f"Expected {expected} confirmatory stability-resample rows."
        )

    return table


# ---------------------------------------------------------------------
# Neural architecture and loss contract
# ---------------------------------------------------------------------


def build_loss_contract(
    upstream: dict[str, Any],
) -> dict[str, Any]:
    ablation_table = upstream[
        "ablations"
    ]

    ablation_records = {}

    for _, row in ablation_table.iterrows():
        variant = clean_text(
            row[
                "variant_id"
            ]
        )

        ablation_records[
            variant
        ] = {
            column: (
                parse_bool(
                    row[
                        column
                    ]
                )
                if column
                in {
                    "block_balanced_reconstruction",
                    "shared_private_decomposition",
                    "paired_contrastive_alignment",
                    "modality_dropout",
                    "missing_view_reconstruction",
                    "stability_gate",
                    "variational_latent",
                }
                else clean_text(
                    row[
                        column
                    ]
                )
            )
            for column in ablation_table.columns
            if column
            != "variant_id"
        }

    return {
        "contract_version": (
            "paper5-neural-loss-architecture-v1"
        ),
        "device": "cuda:0",
        "heavy_computation_device_policy": (
            "All neural forward passes, backward passes, optimizer steps, "
            "contrastive similarity matrices, reconstruction losses, and "
            "latent projections must execute on CUDA. Quiet CPU fallback is forbidden."
        ),
        "cpu_only_allowed_operations": [
            "file I/O",
            "pandas audit tables",
            "fold-local deterministic feature ranking",
            "training-fold median imputation",
            "deterministic dual-PCA preprocessing",
            "hashing and provenance",
        ],
        "precision": {
            "dtype": "float32",
            "automatic_mixed_precision": False,
            "tf32": False,
            "torch_compile": False,
        },
        "parameter_initialization": {
            "linear_weights": (
                "torch.nn.init.xavier_uniform_ using the frozen fit seed"
            ),
            "linear_bias": "zeros",
            "layernorm_weight": "ones",
            "layernorm_bias": "zeros",
            "stochastic_layer_dropout": False,
        },
        "encoder": {
            "per_view": True,
            "body": (
                "Linear(PCA_dim, hidden_dim) -> LayerNorm -> GELU"
            ),
            "V0_output": (
                "shared latent head only"
            ),
            "V1_to_V4_output": (
                "separate shared and private latent heads"
            ),
            "V5_output": (
                "shared mean/log-variance and private mean/log-variance heads"
            ),
        },
        "shared_representation": {
            "per_view_shared_embedding": True,
            "patient_shared_embedding": (
                "arithmetic mean of available per-view shared embeddings"
            ),
            "missing_view_rule": (
                "dropped or unavailable views are excluded from the shared mean"
            ),
            "minimum_available_views": 1,
        },
        "decoder": {
            "per_view": True,
            "V0_input": "patient shared embedding",
            "V1_to_V5_observed_view_input": (
                "concatenate(patient shared embedding, own private embedding)"
            ),
            "missing_view_input": (
                "concatenate(patient shared embedding from available views, "
                "zero private vector for the omitted target view)"
            ),
            "body": (
                "Linear(latent_input, hidden_dim) -> LayerNorm -> GELU "
                "-> Linear(hidden_dim, PCA_dim)"
            ),
        },
        "fold_local_preprocessing": {
            "maximum_feature_missing_fraction": 0.20,
            "variance_ranking": (
                "training-fold pre-standardization variance only"
            ),
            "feature_budget_role": "primary",
            "imputation": (
                "training-fold feature median"
            ),
            "scaling": (
                "training-fold mean and sample standard deviation after imputation"
            ),
            "PCA": (
                "deterministic training-fold dual PCA with sign orientation"
            ),
            "effective_PCA_components": (
                "min(profile component target, n_training - 1, numerical matrix rank)"
            ),
            "post_zscore_variance_selection": False,
            "heldout_preprocessing": False,
        },
        "block_balanced_reconstruction": {
            "per_view_MSE": (
                "mean squared error in the fold-local PCA coordinate system"
            ),
            "null_error": (
                "fit-subset MSE from predicting the zero/mean PCA vector for that view"
            ),
            "null_error_floor": (
                MINIMUM_NULL_RECONSTRUCTION_MSE
            ),
            "normalized_view_loss": (
                "view_MSE / max(view_null_MSE, 1e-8)"
            ),
            "aggregate": (
                "unweighted mean of normalized view losses"
            ),
        },
        "shared_alignment_loss": {
            "applies_to": [
                "V1_SHARED_PRIVATE_AE",
                "V2_CONTRASTIVE_SPAE",
                "V3_DROPOUT_SPAE",
                "V4_SABLE_MO",
                "V5_SABLE_VAE",
            ],
            "formula": (
                "mean squared distance from each available per-view shared embedding "
                "to the patient shared mean"
            ),
        },
        "shared_private_orthogonality_loss": {
            "applies_to": [
                "V1_SHARED_PRIVATE_AE",
                "V2_CONTRASTIVE_SPAE",
                "V3_DROPOUT_SPAE",
                "V4_SABLE_MO",
                "V5_SABLE_VAE",
            ],
            "formula": (
                "mean squared Frobenius norm of centered shared-private "
                "cross-covariance within each available view"
            ),
        },
        "private_separation_loss": {
            "applies_to": [
                "V1_SHARED_PRIVATE_AE",
                "V2_CONTRASTIVE_SPAE",
                "V3_DROPOUT_SPAE",
                "V4_SABLE_MO",
                "V5_SABLE_VAE",
            ],
            "formula": (
                "mean squared Frobenius norm of centered cross-view private "
                "cross-covariance across available ordered view pairs"
            ),
        },
        "paired_contrastive_loss": {
            "applies_to": [
                "V2_CONTRASTIVE_SPAE",
                "V3_DROPOUT_SPAE",
                "V4_SABLE_MO",
                "V5_SABLE_VAE",
            ],
            "formula": (
                "symmetric full-batch InfoNCE over L2-normalized per-view shared "
                "embeddings; same patient is the positive and all other patients "
                "in the fit batch are negatives"
            ),
            "temperature_source": (
                "05a regularization profile"
            ),
        },
        "modality_dropout": {
            "applies_to": [
                "V3_DROPOUT_SPAE",
                "V4_SABLE_MO",
                "V5_SABLE_VAE",
            ],
            "unit": "patient x epoch",
            "rule": (
                "with the frozen profile probability, independently mark a patient "
                "for dropout and remove exactly one uniformly sampled modality; "
                "never drop all modalities"
            ),
            "randomness": (
                "CUDA torch.Generator seeded by the frozen fit seed and epoch"
            ),
        },
        "missing_view_reconstruction_loss": {
            "applies_to": [
                "V3_DROPOUT_SPAE",
                "V4_SABLE_MO",
                "V5_SABLE_VAE",
            ],
            "training_target": (
                "deliberately dropped target modality in PCA space"
            ),
            "decoder_private_input": (
                "zero vector for omitted modality"
            ),
            "weight_source": (
                "05a cross_reconstruction_weight"
            ),
        },
        "VAE_extension": {
            "variant": (
                VAE_VARIANT
            ),
            "prior": (
                "independent standard normal for shared and private latent variables"
            ),
            "reparameterization": (
                "standard Gaussian reparameterization trick on CUDA"
            ),
            "KL_final_weight": (
                VAE_KL_FINAL_WEIGHT
            ),
            "KL_warmup": (
                f"linear from 0 to {VAE_KL_FINAL_WEIGHT} over the first "
                f"{VAE_KL_WARMUP_FRACTION:.2f} of fixed/training epochs"
            ),
            "profile_tuned_KL_weight": False,
            "role": "secondary_exploratory",
        },
        "optimizer": {
            "type": "AdamW",
            "learning_rate_source": (
                "05a regularization profile"
            ),
            "weight_decay_source": (
                "05a regularization profile"
            ),
            "gradient_clip_global_norm": (
                GRADIENT_CLIP_NORM
            ),
            "batching": "full batch",
        },
        "early_stopping": {
            "maximum_epochs_source": (
                "05a architecture profile"
            ),
            "patience_source": (
                "05a architecture profile"
            ),
            "min_delta": 1e-4,
            "monitor": (
                "deterministic early-stop total objective evaluated with "
                "modality dropout disabled; V3-V5 missing-view term is evaluated "
                "by systematic leave-one-view-out reconstruction"
            ),
            "metric_validation_reuse": False,
            "refit": (
                "after epoch selection, reinitialize with the same frozen fit seed "
                "and refit on fit + early-stop samples for exactly the selected epoch count"
            ),
        },
        "stability_gate": {
            "applies_to": [
                "V4_SABLE_MO",
                "V5_SABLE_VAE",
            ],
            "training_component": False,
            "resampling_component": True,
            "subsample_fraction": (
                STABILITY_SUBSAMPLE_FRACTION
            ),
            "resamples_per_dataset": (
                ARCHITECTURE_STABILITY_RESAMPLES
            ),
            "alignment": (
                "orthogonal Procrustes on shared scores, then Hungarian factor matching "
                "and sign orientation"
            ),
            "resample_match_threshold": (
                STABILITY_RESAMPLE_MATCH_THRESHOLD
            ),
            "factor_retention_thresholds": (
                upstream[
                    "contract_05a"
                ][
                    "stability_gate"
                ][
                    "thresholds"
                ]
            ),
            "factor_variance_source": (
                "reference full-data shared scores"
            ),
            "modality_support_energy": (
                "squared L2 norm of the shared-dimension column in each view's "
                "first decoder affine layer, normalized across views"
            ),
            "rotational_rule": (
                "if factor_level_identifiable=false but subspace_level_identifiable=true, "
                "report and gate the shared subspace; individual-factor claims are forbidden"
            ),
        },
        "technical_confound_simulation_status": {
            "technical_truth_used_for_training": False,
            "technical_truth_used_for_profile_utility": False,
            "technical_truth_used_after_stability": True,
            "technical_flag_rule": (
                "stable factor with absolute correlation >= 0.50 to known technical "
                "simulation truth is labeled stable_technical_flag"
            ),
            "false_biological_acceptance": (
                "stable nontechnical factor that fails biological truth matching"
            ),
        },
        "ablation_components": (
            ablation_records
        ),
    }


# ---------------------------------------------------------------------
# Selection contract
# ---------------------------------------------------------------------


def build_selection_contract(
    upstream: dict[str, Any],
    profiles: pd.DataFrame,
) -> dict[str, Any]:
    selection = upstream[
        "selection"
    ][
        "simulation_profile_selection"
    ]

    utility = selection[
        "primary_simulation_utility"
    ]

    expected_utility = {
        "simulation_shared_subspace_recovery": 0.40,
        "simulation_private_subspace_recovery": 0.20,
        "random_adjusted_cross_view_mrr": 0.20,
        "missing_view_reconstruction_score": 0.10,
        "modality_balance_score": 0.10,
    }

    if utility != expected_utility:
        raise RuntimeError(
            "05a simulation utility weights changed."
        )

    return {
        "contract_version": (
            "paper5-neural-profile-selection-v1"
        ),
        "profile_selection_dataset_role": (
            "profile_development only; replicate indices 1-10 within each scenario"
        ),
        "confirmatory_neural_dataset_role": (
            "confirmatory_ablation only; replicate indices 11-20 within each scenario"
        ),
        "development_dataset_count": (
            EXPECTED_DEVELOPMENT_DATASETS
        ),
        "confirmatory_dataset_count": (
            EXPECTED_CONFIRMATORY_DATASETS
        ),
        "development_partition": {
            "fit_folds": list(
                PROFILE_FIT_FOLDS
            ),
            "early_stop_fold": (
                PROFILE_EARLY_STOP_FOLD
            ),
            "validation_fold": (
                PROFILE_VALIDATION_FOLD
            ),
            "source": (
                "exact 05c1 frozen five-fold split registry"
            ),
            "validation_never_used_for_early_stopping": True,
        },
        "candidate_profiles": (
            profiles[
                [
                    "profile_id",
                    "architecture_profile",
                    "regularization_profile",
                    "combined_complexity_rank",
                    "architecture_complexity_rank",
                    "regularization_complexity_rank",
                ]
            ].to_dict(
                orient="records"
            )
        ),
        "profile_selected_for": (
            "V4_SABLE_MO"
        ),
        "profile_reused_by": [
            "V0_BALANCED_AE",
            "V1_SHARED_PRIVATE_AE",
            "V2_CONTRASTIVE_SPAE",
            "V3_DROPOUT_SPAE",
            "V5_SABLE_VAE",
        ],
        "screening_seed_count": 1,
        "screening_base_seed": (
            PROFILE_SCREEN_BASE_SEED
        ),
        "primary_utility": (
            expected_utility
        ),
        "dataset_utility": (
            "weighted sum of the five frozen components after validation scoring"
        ),
        "scenario_aggregation": (
            "average dataset utility within each of the 36 scenarios"
        ),
        "overall_aggregation": (
            "equal-weight mean of the 36 scenario means"
        ),
        "standard_error_unit": (
            "36 scenario-level mean utilities, not 360 individual datasets"
        ),
        "hard_eligibility": {
            "latent_collapse": (
                "zero development reference fits may satisfy G02_LATENT_COLLAPSE"
            ),
            "stability_gated_false_shared_discovery": (
                "mean false biological shared discovery fraction <= 0.10 "
                "among factor-level-identifiable or true-null development datasets"
            ),
            "nonfinite_optimization": (
                "all failed fits retained; profile is ineligible if any required "
                "profile-development reference or stability fit is non-finite"
            ),
        },
        "stability_profile_screen": {
            "all_9_profiles_evaluated": True,
            "development_datasets_per_profile": (
                EXPECTED_DEVELOPMENT_DATASETS
            ),
            "reference_full_data_fit_per_dataset_profile": 1,
            "resamples_per_dataset_profile": (
                ARCHITECTURE_STABILITY_RESAMPLES
            ),
            "subsample_fraction": (
                STABILITY_SUBSAMPLE_FRACTION
            ),
            "technical_truth_is_not_training_information": True,
            "rotational_scenarios_use_subspace_metrics": True,
        },
        "one_standard_error_rule": (
            "Among hard-gate-eligible profiles, identify the profile with the "
            "highest overall mean utility. Eligible profiles with mean utility >= "
            "best_mean - SE(best) form the one-SE set. Select the profile with the "
            "lowest combined_complexity_rank; ties use architecture_complexity_rank, "
            "then regularization_complexity_rank, then profile_id lexicographically."
        ),
        "combined_complexity_rank": (
            "architecture_complexity_rank + regularization_complexity_rank"
        ),
        "target_data_may_change_profile": False,
        "baseline_results_may_change_profile": False,
        "confirmatory_replicates_may_change_profile": False,
        "biological_or_outcome_labels_may_change_profile": False,
    }


# ---------------------------------------------------------------------
# Main protocol
# ---------------------------------------------------------------------


def build_protocol(
    upstream: dict[str, Any],
    profiles: pd.DataFrame,
    roles: pd.DataFrame,
    development_partitions: pd.DataFrame,
    profile_seeds: pd.DataFrame,
    profile_stability: pd.DataFrame,
    confirmatory_seeds: pd.DataFrame,
    confirmatory_stability: pd.DataFrame,
) -> dict[str, Any]:
    simulation_master_seed = int(
        upstream[
            "contract_05a"
        ][
            "simulation"
        ][
            "master_seed"
        ]
    )

    return {
        "contract_version": (
            CONTRACT_VERSION
        ),
        "script_version": (
            SCRIPT_VERSION
        ),
        "created_at_utc": (
            utc_now_iso()
        ),
        "scientific_state": {
            "outcomes_loaded": False,
            "clinical_labels_loaded": False,
            "biological_annotations_loaded": False,
            "TARGET_model_fitting_performed": False,
            "neural_simulation_model_fitting_performed": False,
            "neural_profile_selected": False,
            "architecture_selected": False,
            "stability_gate_thresholds_retuned": False,
            "final_gate_calibration_data_used": False,
        },
        "gpu_execution": {
            "required_for_heavy_neural_compute": True,
            "device": "cuda:0",
            "quiet_cpu_fallback_allowed": False,
            "frozen_environment_contract": (
                relative_path(
                    NEURAL_ENV_CONTRACT
                )
            ),
            "frozen_environment_sha256": (
                sha256_file(
                    NEURAL_ENV_CONTRACT
                )
            ),
        },
        "simulation_scope": {
            "total_scenarios": (
                EXPECTED_SCENARIOS
            ),
            "total_datasets": (
                EXPECTED_DATASETS
            ),
            "profile_development_datasets": (
                EXPECTED_DEVELOPMENT_DATASETS
            ),
            "confirmatory_ablation_datasets": (
                EXPECTED_CONFIRMATORY_DATASETS
            ),
            "development_replicates_per_scenario": list(
                DEVELOPMENT_REPLICATES
            ),
            "confirmatory_replicates_per_scenario": list(
                CONFIRMATORY_REPLICATES
            ),
            "split_created_before_neural_results": True,
        },
        "profile_grid": {
            "n_profiles": int(
                profiles.shape[
                    0
                ]
            ),
            "file": relative_path(
                OUTPUT_PROFILE_GRID
            ),
        },
        "profile_development": {
            "dataset_registry": relative_path(
                OUTPUT_DATASET_ROLES
            ),
            "partition_registry": relative_path(
                OUTPUT_DEVELOPMENT_PARTITIONS
            ),
            "fit_seed_registry": relative_path(
                OUTPUT_PROFILE_SEEDS
            ),
            "stability_resample_registry": relative_path(
                OUTPUT_PROFILE_STABILITY
            ),
            "development_partition_rule": (
                "05c1 folds 3-5 fit; fold 2 early-stop; fold 1 validation"
            ),
            "profile_screen_fit_evaluations": int(
                profile_seeds.shape[
                    0
                ]
            ),
            "profile_screen_optimization_runs": int(
                profile_seeds.shape[
                    0
                ]
                * 2
            ),
            "profile_stability_reference_fits": (
                EXPECTED_DEVELOPMENT_DATASETS
                * profiles.shape[
                    0
                ]
            ),
            "profile_stability_resample_fits": int(
                profile_stability.shape[
                    0
                ]
            ),
        },
        "confirmatory_ablation": {
            "dataset_registry": relative_path(
                OUTPUT_DATASET_ROLES
            ),
            "fit_seed_registry": relative_path(
                OUTPUT_CONFIRMATORY_SEEDS
            ),
            "stability_resample_registry": relative_path(
                OUTPUT_CONFIRMATORY_STABILITY
            ),
            "variants": list(
                NEURAL_VARIANTS
            ),
            "seeds_per_variant_fold": len(
                CONFIRMATORY_MODEL_BASE_SEEDS
            ),
            "folds": 5,
            "fit_evaluations": int(
                confirmatory_seeds.shape[
                    0
                ]
            ),
            "optimization_runs_including_refit": int(
                confirmatory_seeds.shape[
                    0
                ]
                * 2
            ),
            "V4_confirmatory_stability_reference_fits": (
                EXPECTED_CONFIRMATORY_DATASETS
            ),
            "V4_confirmatory_stability_resample_fits": int(
                confirmatory_stability.shape[
                    0
                ]
            ),
        },
        "master_seed": (
            simulation_master_seed
        ),
        "runtime_rule": (
            "All neural optimization and tensor-heavy representation calculations "
            "must execute on the frozen CUDA device; deterministic CPU preprocessing "
            "is allowed where it is scientifically preferable."
        ),
        "loss_contract": relative_path(
            OUTPUT_LOSS_CONTRACT
        ),
        "selection_contract": relative_path(
            OUTPUT_SELECTION_CONTRACT
        ),
        "stage_order": [
            {
                "stage": "05d2",
                "script": (
                    "05d2_screen_neural_profiles_gpu.py"
                ),
                "purpose": (
                    "GPU profile screening and 20-resample stability audit on "
                    "development simulation replicates only"
                ),
                "neural_profile_selected_at_end": True,
            },
            {
                "stage": "05d3",
                "script": (
                    "05d3_freeze_selected_neural_profile.py"
                ),
                "purpose": (
                    "freeze the outcome-blind simulation-selected neural profile "
                    "before confirmatory neural ablations"
                ),
            },
            {
                "stage": "05d4",
                "script": (
                    "05d4_fit_neural_ablation_benchmark_gpu.py"
                ),
                "purpose": (
                    "GPU V0-V5 confirmatory held-out benchmark on replicate indices 11-20"
                ),
            },
            {
                "stage": "05d5",
                "script": (
                    "05d5_evaluate_sable_mo_stability_gpu.py"
                ),
                "purpose": (
                    "GPU 20-resample confirmatory V4 stability-gate evaluation"
                ),
            },
            {
                "stage": "05e",
                "script": (
                    "05e_compare_ablate_and_freeze_method.py"
                ),
                "purpose": (
                    "compare frozen linear, MOFA, and neural results without outcomes "
                    "and freeze the representation method"
                ),
            },
        ],
        "forbidden_operations": [
            "Using simulation replicates 11-20 to select the neural profile.",
            "Using simulation replicates 1-10 as confirmatory neural ablation evidence.",
            "Using TARGET-OS results to change architecture or regularization profiles.",
            "Changing the 05a architecture or regularization grids.",
            "Using biological program labels for neural profile selection.",
            "Using survival, metastasis, treatment, or other outcomes.",
            "Using final gate-calibration simulations before architecture freeze.",
            "Quietly falling back to CPU for neural optimization.",
            "Using automatic mixed precision, TF32, or torch.compile in the confirmatory neural branch.",
            "Changing stability thresholds after viewing neural results.",
            "Silently omitting failed V0-V5 variants.",
        ],
        "upstream_hashes": (
            upstream[
                "hashes"
            ]
        ),
        "generated_registries": {
            "profile_grid": relative_path(
                OUTPUT_PROFILE_GRID
            ),
            "dataset_roles": relative_path(
                OUTPUT_DATASET_ROLES
            ),
            "development_partitions": relative_path(
                OUTPUT_DEVELOPMENT_PARTITIONS
            ),
            "profile_fit_seeds": relative_path(
                OUTPUT_PROFILE_SEEDS
            ),
            "profile_stability": relative_path(
                OUTPUT_PROFILE_STABILITY
            ),
            "confirmatory_fit_seeds": relative_path(
                OUTPUT_CONFIRMATORY_SEEDS
            ),
            "confirmatory_stability": relative_path(
                OUTPUT_CONFIRMATORY_STABILITY
            ),
        },
    }


# ---------------------------------------------------------------------
# Existing lock
# ---------------------------------------------------------------------


def verify_existing_lock() -> None:
    lock = load_json(
        OUTPUT_LOCK
    )

    if clean_text(
        lock.get(
            "script_version"
        )
    ) != SCRIPT_VERSION:
        raise RuntimeError(
            "Existing 05d1 script-version mismatch."
        )

    failures = []

    for relative_name, record in lock.get(
        "generated_files",
        {},
    ).items():
        candidate = resolve_project_path(
            relative_name
        )

        if not candidate.exists():
            failures.append(
                f"missing: {relative_name}"
            )
            continue

        if int(
            candidate.stat().st_size
        ) != int(
            record[
                "size_bytes"
            ]
        ):
            failures.append(
                f"size mismatch: {relative_name}"
            )
            continue

        if sha256_file(
            candidate
        ) != clean_text(
            record[
                "sha256"
            ]
        ):
            failures.append(
                f"SHA256 mismatch: {relative_name}"
            )

    if failures:
        raise RuntimeError(
            "Existing 05d1 lock verification failed:\n"
            + "\n".join(
                f"- {value}"
                for value in failures
            )
        )

    print(
        "="
        * 88
    )

    print(
        "Existing Paper 5 neural simulation protocol: PASS"
    )

    print(
        "="
        * 88
    )

    print(
        f"Lock: {relative_path(OUTPUT_LOCK)}"
    )

    print(
        "No files were rewritten."
    )

    print(
        "="
        * 88
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    print(
        "="
        * 88
    )

    print(
        "Paper 5 - freeze neural simulation protocol"
    )

    print(
        "="
        * 88
    )

    print(
        f"Script version: {SCRIPT_VERSION}"
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT_LOCK.exists():
        verify_existing_lock()
        return

    upstream = verify_upstream()

    validate_05a_neural_design(
        upstream
    )

    print(
        "05a neural architecture/regularization grids: PASS"
    )

    print(
        "05a1 simulation/stability addendum: PASS"
    )

    print(
        "05b v2 simulation bank: PASS"
    )

    print(
        "05c1 exact five-fold split registry: PASS"
    )

    print(
        "05c2b MOFA result lock: PASS"
    )

    print(
        "05d0 CUDA neural environment lock: PASS"
    )

    print(
        "Neural model results before 05d1: NO"
    )

    print(
        "Architecture/profile selected before 05d1: NO"
    )

    print(
        "Final gate-calibration data accessed: NO"
    )

    print(
        "Outcome access: FORBIDDEN"
    )

    print()

    profiles = build_profile_grid(
        upstream[
            "architecture"
        ],
        upstream[
            "regularization"
        ],
    )

    roles = build_dataset_role_registry(
        upstream[
            "simulation_manifest"
        ]
    )

    split_lookup = build_split_lookup(
        upstream[
            "split_table"
        ]
    )

    development_partitions = (
        build_development_partitions(
            roles,
            split_lookup,
        )
    )

    development = roles[
        roles[
            "neural_dataset_role"
        ].eq(
            "profile_development"
        )
    ].copy()

    confirmatory = roles[
        roles[
            "neural_dataset_role"
        ].eq(
            "confirmatory_ablation"
        )
    ].copy()

    master_seed = int(
        upstream[
            "contract_05a"
        ][
            "simulation"
        ][
            "master_seed"
        ]
    )

    profile_seeds = (
        build_profile_fit_seeds(
            development,
            profiles,
            master_seed,
        )
    )

    profile_stability = (
        build_profile_stability_registry(
            development,
            profiles,
            master_seed,
        )
    )

    confirmatory_seeds = (
        build_confirmatory_fit_seeds(
            confirmatory,
            master_seed,
        )
    )

    confirmatory_stability = (
        build_confirmatory_stability_registry(
            confirmatory,
            master_seed,
        )
    )

    profiles.to_csv(
        OUTPUT_PROFILE_GRID,
        index=False,
        lineterminator="\n",
    )

    roles.to_csv(
        OUTPUT_DATASET_ROLES,
        index=False,
        lineterminator="\n",
    )

    development_partitions.to_csv(
        OUTPUT_DEVELOPMENT_PARTITIONS,
        index=False,
        lineterminator="\n",
    )

    profile_seeds.to_csv(
        OUTPUT_PROFILE_SEEDS,
        index=False,
        lineterminator="\n",
    )

    profile_stability.to_csv(
        OUTPUT_PROFILE_STABILITY,
        index=False,
        lineterminator="\n",
    )

    confirmatory_seeds.to_csv(
        OUTPUT_CONFIRMATORY_SEEDS,
        index=False,
        lineterminator="\n",
    )

    confirmatory_stability.to_csv(
        OUTPUT_CONFIRMATORY_STABILITY,
        index=False,
        lineterminator="\n",
    )

    loss_contract = build_loss_contract(
        upstream
    )

    selection_contract = (
        build_selection_contract(
            upstream,
            profiles,
        )
    )

    write_json_atomic(
        loss_contract,
        OUTPUT_LOSS_CONTRACT,
    )

    write_json_atomic(
        selection_contract,
        OUTPUT_SELECTION_CONTRACT,
    )

    protocol = build_protocol(
        upstream=upstream,
        profiles=profiles,
        roles=roles,
        development_partitions=(
            development_partitions
        ),
        profile_seeds=(
            profile_seeds
        ),
        profile_stability=(
            profile_stability
        ),
        confirmatory_seeds=(
            confirmatory_seeds
        ),
        confirmatory_stability=(
            confirmatory_stability
        ),
    )

    write_json_atomic(
        protocol,
        OUTPUT_PROTOCOL,
    )

    readme_lines = [
        "Paper 5 neural simulation protocol",
        "=" * 72,
        "",
        f"Script version: {SCRIPT_VERSION}",
        "",
        "Scientific state",
        "----------------",
        "Neural model fitting performed: NO",
        "Neural profile selected: NO",
        "TARGET model fitting performed: NO",
        "Final gate-calibration simulations accessed: NO",
        "Outcome access: NO",
        "",
        "GPU rule",
        "--------",
        "All neural forward/backward/optimizer work must run on frozen cuda:0.",
        "Quiet CPU fallback is forbidden.",
        "CPU is allowed for deterministic preprocessing, I/O, tables, and hashing.",
        "",
        "Simulation split",
        "----------------",
        "Replicates 1-10 of every scenario: neural profile development only.",
        "Replicates 11-20 of every scenario: confirmatory V0-V5 neural ablation only.",
        "This split is frozen before any neural result exists.",
        "",
        "Development partition",
        "---------------------",
        "Reuse exact 05c1 folds.",
        "Folds 3-5: fit.",
        "Fold 2: early stopping.",
        "Fold 1: untouched profile-validation metrics.",
        "",
        "Profile grid",
        "------------",
        "3 frozen 05a architecture profiles x 3 frozen 05a regularization profiles = 9.",
        "Profile is selected for V4 SABLE-MO and reused by V0-V3 and V5.",
        "",
        "Stability",
        "---------",
        "All 9 candidate profiles receive the frozen 20-resample development stability audit.",
        "Confirmatory V4 receives an independent 20-resample stability audit on replicates 11-20.",
        "",
        "Next stage",
        "----------",
        "05d2_screen_neural_profiles_gpu.py",
        "",
    ]

    OUTPUT_README.write_text(
        "\n".join(
            readme_lines
        ),
        encoding="utf-8",
    )

    generated = [
        OUTPUT_PROFILE_GRID,
        OUTPUT_DATASET_ROLES,
        OUTPUT_DEVELOPMENT_PARTITIONS,
        OUTPUT_PROFILE_SEEDS,
        OUTPUT_PROFILE_STABILITY,
        OUTPUT_CONFIRMATORY_SEEDS,
        OUTPUT_CONFIRMATORY_STABILITY,
        OUTPUT_LOSS_CONTRACT,
        OUTPUT_SELECTION_CONTRACT,
        OUTPUT_PROTOCOL,
        OUTPUT_README,
    ]

    lock = {
        "contract_version": (
            CONTRACT_VERSION
        ),
        "script_version": (
            SCRIPT_VERSION
        ),
        "created_at_utc": (
            utc_now_iso()
        ),
        "status": (
            "FROZEN_NEURAL_SIMULATION_PROTOCOL_BEFORE_NEURAL_MODEL_FITTING"
        ),
        "scientific_state": (
            protocol[
                "scientific_state"
            ]
        ),
        "protocol": {
            "path": relative_path(
                OUTPUT_PROTOCOL
            ),
            "sha256": sha256_file(
                OUTPUT_PROTOCOL
            ),
        },
        "generated_files": {
            relative_path(
                path
            ): {
                "sha256": sha256_file(
                    path
                ),
                "size_bytes": int(
                    path.stat().st_size
                ),
            }
            for path in generated
        },
        "rewrite_policy": (
            "This zero-argument stage never rewrites a frozen neural simulation protocol. "
            "Any change requires a new versioned prospective protocol before neural refitting."
        ),
    }

    write_json_atomic(
        lock,
        OUTPUT_LOCK,
    )

    development_optimization_runs = (
        int(
            profile_seeds.shape[
                0
            ]
        )
        * 2
        + EXPECTED_DEVELOPMENT_DATASETS
        * profiles.shape[
            0
        ]
        + int(
            profile_stability.shape[
                0
            ]
        )
    )

    confirmatory_optimization_runs = (
        int(
            confirmatory_seeds.shape[
                0
            ]
        )
        * 2
        + EXPECTED_CONFIRMATORY_DATASETS
        + int(
            confirmatory_stability.shape[
                0
            ]
        )
    )

    print(
        "-"
        * 88
    )

    print(
        "Frozen neural simulation design"
    )

    print(
        "-"
        * 88
    )

    print(
        f"Neural candidate profiles: {profiles.shape[0]}"
    )

    print(
        f"Development datasets: {development.shape[0]}"
    )

    print(
        f"Confirmatory datasets: {confirmatory.shape[0]}"
    )

    print(
        "Development replicate indices: 1-10"
    )

    print(
        "Confirmatory replicate indices: 11-20"
    )

    print(
        "Profile-development fit / early-stop / validation folds: "
        "3-5 / 2 / 1"
    )

    print(
        f"Profile-screen fit evaluations: {profile_seeds.shape[0]:,}"
    )

    print(
        f"Profile stability resample fits: {profile_stability.shape[0]:,}"
    )

    print(
        f"Estimated development neural optimization runs: "
        f"{development_optimization_runs:,}"
    )

    print(
        f"Confirmatory V0-V5 fold-seed evaluations: "
        f"{confirmatory_seeds.shape[0]:,}"
    )

    print(
        f"Confirmatory V4 stability resample fits: "
        f"{confirmatory_stability.shape[0]:,}"
    )

    print(
        f"Estimated confirmatory neural optimization runs: "
        f"{confirmatory_optimization_runs:,}"
    )

    print(
        "Heavy neural computation device: cuda:0"
    )

    print(
        "Quiet CPU fallback: FORBIDDEN"
    )

    print()

    print(
        "="
        * 88
    )

    print(
        "Paper 5 neural simulation protocol: FROZEN"
    )

    print(
        "="
        * 88
    )

    print(
        f"Protocol: {relative_path(OUTPUT_PROTOCOL)}"
    )

    print(
        f"Loss contract: {relative_path(OUTPUT_LOSS_CONTRACT)}"
    )

    print(
        f"Selection contract: {relative_path(OUTPUT_SELECTION_CONTRACT)}"
    )

    print(
        f"Lock: {relative_path(OUTPUT_LOCK)}"
    )

    print(
        "Neural simulation model fitting performed: NO"
    )

    print(
        "Neural profile selected: NO"
    )

    print(
        "Final gate-calibration data accessed: NO"
    )

    print(
        "Next stage: 05d2_screen_neural_profiles_gpu.py"
    )

    print(
        "="
        * 88
    )


if __name__ == "__main__":
    main()
