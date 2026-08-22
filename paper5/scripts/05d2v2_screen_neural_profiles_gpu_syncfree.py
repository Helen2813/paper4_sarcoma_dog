
from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import tempfile
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Deterministic CUDA GEMM configuration must be set before CUDA work starts.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_VERSION = "05d2v2-screen-neural-profiles-gpu-syncfree-v2-no-cli"
CONTRACT_VERSION = "paper5-neural-profile-screen-v2-syncfree"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

VIEWS = ("RNA", "METHYLATION", "CNV")
VIEW_PREFIX = {
    "RNA": "rna",
    "METHYLATION": "methylation",
    "CNV": "cnv",
}
VIEW_INDEX = {
    "RNA": 0,
    "METHYLATION": 1,
    "CNV": 2,
}

PRIMARY_FEATURE_BUDGETS = {
    "RNA": 5_000,
    "METHYLATION": 10_000,
    "CNV": 5_000,
}
MAX_FEATURE_MISSING_FRACTION = 0.20
MIN_FINITE_VALUES = 3
MIN_PREIMPUTATION_VARIANCE = 1e-10
MIN_POSTIMPUTATION_STD = 1e-8
PCA_NUMERICAL_TOLERANCE = 1e-10

VIEW_SUPPORT_ENERGY_THRESHOLD = 0.15
TRUTH_MATCH_THRESHOLD = 0.50
TECHNICAL_MATCH_THRESHOLD = 0.50
GRADIENT_CLIP_NORM = 5.0
NULL_MSE_FLOOR = 1e-8

EXPECTED_PROFILES = 9
EXPECTED_DEVELOPMENT_DATASETS = 360
EXPECTED_SHARDS = (
    EXPECTED_PROFILES
    * EXPECTED_DEVELOPMENT_DATASETS
)
STABILITY_RESAMPLES = 20

# ---------------------------------------------------------------------
# Frozen upstream files
# ---------------------------------------------------------------------

CONTRACT_05A = (
    PAPER5_ROOT
    / "method_contract"
    / "05a"
    / "ai_method_contract.json"
)

SIMULATION_MANIFEST = (
    PAPER5_ROOT
    / "method_contract"
    / "05b_v2"
    / "architecture_screening_data_manifest.csv"
)

LINEAR_SCRIPT = (
    PAPER5_ROOT
    / "scripts"
    / "05c1_fit_linear_simulation_baselines.py"
)

ENV_CONTRACT = (
    PAPER5_ROOT
    / "method_contract"
    / "05d0"
    / "neural_environment_contract.json"
)

ENV_LOCK = (
    PAPER5_ROOT
    / "method_contract"
    / "05d0"
    / "neural_environment_lock.json"
)

PROTOCOL_05D1 = (
    PAPER5_ROOT
    / "method_contract"
    / "05d1"
    / "neural_simulation_evaluation_protocol.json"
)

LOSS_CONTRACT = (
    PAPER5_ROOT
    / "method_contract"
    / "05d1"
    / "neural_loss_and_architecture_contract.json"
)

SELECTION_CONTRACT = (
    PAPER5_ROOT
    / "method_contract"
    / "05d1"
    / "neural_profile_selection_contract.json"
)

LOCK_05D1 = (
    PAPER5_ROOT
    / "method_contract"
    / "05d1"
    / "neural_simulation_protocol_lock.json"
)

PROFILE_GRID = (
    PAPER5_ROOT
    / "method_contract"
    / "05d1"
    / "neural_architecture_regularization_profile_grid.csv"
)

DATASET_ROLES = (
    PAPER5_ROOT
    / "method_contract"
    / "05d1"
    / "neural_simulation_dataset_role_registry.csv"
)

DEVELOPMENT_PARTITIONS = (
    PAPER5_ROOT
    / "method_contract"
    / "05d1"
    / "neural_profile_development_partition_registry.csv"
)

PROFILE_FIT_SEEDS = (
    PAPER5_ROOT
    / "method_contract"
    / "05d1"
    / "neural_profile_screen_fit_seed_registry.csv"
)

PROFILE_STABILITY = (
    PAPER5_ROOT
    / "method_contract"
    / "05d1"
    / "neural_profile_stability_resample_registry.csv"
)

FINAL_GATE_DATA_ROOT = (
    PAPER5_ROOT
    / "simulations"
    / "sable_mo_v2"
    / "final_gate_calibration"
)

FUTURE_RESULT_PATHS = (
    PAPER5_ROOT / "method_contract" / "05d3",
    PAPER5_ROOT / "method_contract" / "05d4",
    PAPER5_ROOT / "method_contract" / "05d5",
    PAPER5_ROOT / "method_contract" / "05e",
    PAPER5_ROOT / "results" / "simulation_benchmark" / "05d_neural" / "confirmatory",
)

# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------

OUTPUT_CONTRACT_DIR = (
    PAPER5_ROOT
    / "method_contract"
    / "05d2_v2"
)

OUTPUT_RESULTS_DIR = (
    PAPER5_ROOT
    / "results"
    / "simulation_benchmark"
    / "05d_neural"
    / "profile_screen_v2"
)

OUTPUT_SHARD_DIR = (
    OUTPUT_RESULTS_DIR
    / "shards"
)

OUTPUT_PROTOCOL = (
    OUTPUT_CONTRACT_DIR
    / "neural_profile_screen_execution_protocol.json"
)

OUTPUT_FIXTURE = (
    OUTPUT_CONTRACT_DIR
    / "neural_profile_screen_reference_fixture.json"
)

OUTPUT_SHARD_MANIFEST = (
    OUTPUT_CONTRACT_DIR
    / "neural_profile_screen_shard_manifest.csv"
)

OUTPUT_RESULT_MANIFEST = (
    OUTPUT_CONTRACT_DIR
    / "neural_profile_screen_results_manifest.json"
)

OUTPUT_LOCK = (
    OUTPUT_CONTRACT_DIR
    / "neural_profile_screen_results_lock.json"
)

OUTPUT_README = (
    OUTPUT_CONTRACT_DIR
    / "neural_profile_screen_README.txt"
)

OUTPUT_IN_PROGRESS = (
    OUTPUT_CONTRACT_DIR
    / ".neural_profile_screen_in_progress.json"
)

OUTPUT_DATASET_METRICS = (
    OUTPUT_RESULTS_DIR
    / "neural_profile_dataset_metrics.csv.gz"
)

OUTPUT_STABILITY_FACTORS = (
    OUTPUT_RESULTS_DIR
    / "neural_profile_stability_factor_audit.csv.gz"
)

OUTPUT_STABILITY_SUMMARY = (
    OUTPUT_RESULTS_DIR
    / "neural_profile_stability_dataset_summary.csv.gz"
)

OUTPUT_FIT_AUDIT = (
    OUTPUT_RESULTS_DIR
    / "neural_profile_fit_audit.csv.gz"
)

OUTPUT_PREPROCESSING = (
    OUTPUT_RESULTS_DIR
    / "neural_profile_preprocessing_audit.csv.gz"
)

OUTPUT_SCENARIO_SUMMARY = (
    OUTPUT_RESULTS_DIR
    / "neural_profile_scenario_summary.csv"
)

OUTPUT_PROFILE_SUMMARY = (
    OUTPUT_RESULTS_DIR
    / "neural_profile_overall_summary.csv"
)

OUTPUT_SELECTION_TABLE = (
    OUTPUT_RESULTS_DIR
    / "neural_profile_selection_table.csv"
)

OUTPUT_SELECTION_RESULT = (
    OUTPUT_RESULTS_DIR
    / "neural_profile_selection_result.json"
)

ABORTED_V1_CONTRACT_DIR = (
    PAPER5_ROOT
    / "method_contract"
    / "05d2"
)

ABORTED_V1_LOCK = (
    ABORTED_V1_CONTRACT_DIR
    / "neural_profile_screen_results_lock.json"
)

ABORTED_V1_SHARD_DIR = (
    PAPER5_ROOT
    / "results"
    / "simulation_benchmark"
    / "05d_neural"
    / "profile_screen"
    / "shards"
)

OUTPUT_EXECUTION_AMENDMENT = (
    OUTPUT_CONTRACT_DIR
    / "runtime_optimization_amendment.json"
)


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class NeuralViewPreprocessor:
    view: str
    selected_indices: np.ndarray
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    pca: Any
    pca_components: int
    n_eligible_before_budget: int
    feature_budget: int

    def transform_observed(
        self,
        matrix: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(
            matrix[
                :,
                self.selected_indices,
            ],
            dtype=np.float64,
        ).copy()

        missing = ~np.isfinite(
            values
        )

        if missing.any():
            row_index, column_index = np.where(
                missing
            )

            values[
                row_index,
                column_index,
            ] = self.medians[
                column_index
            ]

        values = (
            values
            - self.means
        ) / self.scales

        scores = self.pca.transform(
            values,
            self.pca_components,
        )

        if not np.isfinite(
            scores
        ).all():
            raise RuntimeError(
                f"{self.view}: non-finite PCA scores."
            )

        return scores.astype(
            np.float32,
            copy=False,
        )

    def transform_complete(
        self,
        matrix: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(
            matrix[
                :,
                self.selected_indices,
            ],
            dtype=np.float64,
        ).copy()

        if not np.isfinite(
            values
        ).all():
            raise RuntimeError(
                f"{self.view}: complete truth contains non-finite values."
            )

        values = (
            values
            - self.means
        ) / self.scales

        scores = self.pca.transform(
            values,
            self.pca_components,
        )

        if not np.isfinite(
            scores
        ).all():
            raise RuntimeError(
                f"{self.view}: non-finite complete PCA scores."
            )

        return scores.astype(
            np.float32,
            copy=False,
        )


@dataclass(frozen=True)
class ProfileSpec:
    profile_id: str
    architecture_profile: str
    regularization_profile: str
    pca_components: dict[str, int]
    shared_dim: int
    private_dim: int
    hidden_dim: int
    maximum_epochs: int
    patience: int
    min_delta: float
    reconstruction_weight: float
    shared_alignment_weight: float
    contrastive_weight: float
    orthogonality_weight: float
    private_separation_weight: float
    cross_reconstruction_weight: float
    modality_dropout_probability: float
    contrastive_temperature: float
    learning_rate: float
    weight_decay: float
    combined_complexity_rank: int
    architecture_complexity_rank: int
    regularization_complexity_rank: int


@dataclass
class FitResult:
    model: nn.Module
    selected_epoch: int
    history: list[dict[str, float]]
    wall_seconds: float
    fit_status: str
    error_type: str
    error_message: str


@dataclass
class GpuDualPCA:
    mean: torch.Tensor
    components: torch.Tensor
    singular_values: torch.Tensor
    explained_variance: torch.Tensor
    total_variance: float

    def transform(
        self,
        matrix: np.ndarray,
        rank: int,
    ) -> np.ndarray:
        values = torch.as_tensor(
            np.asarray(matrix, dtype=np.float64),
            dtype=torch.float64,
            device="cuda:0",
        )

        k = min(
            int(rank),
            int(self.components.shape[0]),
        )

        scores = (
            values
            - self.mean
        ) @ self.components[
            :k,
            :
        ].T

        if not bool(torch.isfinite(scores).all()):
            raise RuntimeError(
                "GPU dual-PCA transform produced non-finite scores."
            )

        return (
            scores.detach()
            .cpu()
            .numpy()
        )

    def inverse_transform(
        self,
        scores: np.ndarray,
        rank: int,
    ) -> np.ndarray:
        score_tensor = torch.as_tensor(
            np.asarray(scores, dtype=np.float64),
            dtype=torch.float64,
            device="cuda:0",
        )

        k = min(
            int(rank),
            int(self.components.shape[0]),
        )

        reconstructed = (
            score_tensor[:, :k]
            @ self.components[:k, :]
            + self.mean
        )

        return (
            reconstructed.detach()
            .cpu()
            .numpy()
        )


def fit_dual_pca_gpu(
    training_matrix: np.ndarray,
    maximum_rank: int,
) -> GpuDualPCA:
    matrix = np.asarray(
        training_matrix,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "GPU PCA input must be two-dimensional."
        )

    if not np.isfinite(matrix).all():
        raise RuntimeError(
            "GPU PCA input contains non-finite values."
        )

    values = torch.as_tensor(
        matrix,
        dtype=torch.float64,
        device="cuda:0",
    )

    mean = torch.mean(
        values,
        dim=0,
    )

    centered = (
        values
        - mean
    )

    gram = (
        centered
        @ centered.T
    )

    eigenvalues, eigenvectors = (
        torch.linalg.eigh(
            gram
        )
    )

    order = torch.argsort(
        eigenvalues,
        descending=True,
    )

    eigenvalues = eigenvalues[
        order
    ]

    eigenvectors = eigenvectors[
        :,
        order,
    ]

    positive = (
        eigenvalues
        > PCA_NUMERICAL_TOLERANCE
    )

    eigenvalues = eigenvalues[
        positive
    ]

    eigenvectors = eigenvectors[
        :,
        positive
    ]

    available_rank = min(
        int(maximum_rank),
        int(eigenvalues.shape[0]),
        int(centered.shape[1]),
    )

    if available_rank < 1:
        raise RuntimeError(
            "GPU dual PCA has zero numerical rank."
        )

    eigenvalues = eigenvalues[
        :available_rank
    ]

    eigenvectors = eigenvectors[
        :,
        :available_rank,
    ]

    singular_values = torch.sqrt(
        eigenvalues
    )

    components = (
        eigenvectors.T
        @ centered
    ) / singular_values[
        :,
        None,
    ]

    component_norms = torch.linalg.vector_norm(
        components,
        ord=2,
        dim=1,
    )

    components = (
        components
        / component_norms[
            :,
            None,
        ]
    )

    singular_values = (
        singular_values
        * component_norms
    )

    # Same deterministic sign convention as 05c1:
    # the largest absolute loading is positive.
    pivots = torch.argmax(
        torch.abs(
            components
        ),
        dim=1,
    )

    rows = torch.arange(
        components.shape[0],
        device=components.device,
    )

    pivot_values = components[
        rows,
        pivots,
    ]

    signs = torch.where(
        pivot_values
        < 0,
        torch.full_like(
            pivot_values,
            -1.0,
        ),
        torch.ones_like(
            pivot_values,
        ),
    )

    components = (
        components
        * signs[
            :,
            None,
        ]
    )

    denominator = max(
        1,
        int(
            matrix.shape[0]
            - 1
        ),
    )

    explained_variance = (
        singular_values
        ** 2
    ) / denominator

    total_variance = float(
        torch.sum(
            torch.var(
                centered,
                dim=0,
                correction=1,
            )
        )
        .detach()
        .cpu()
    )

    if not bool(
        torch.isfinite(
            components
        ).all()
    ):
        raise RuntimeError(
            "GPU dual PCA produced non-finite components."
        )

    return GpuDualPCA(
        mean=mean.detach(),
        components=(
            components.detach()
        ),
        singular_values=(
            singular_values.detach()
        ),
        explained_variance=(
            explained_variance.detach()
        ),
        total_variance=(
            total_variance
        ),
    )


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def utc_now_iso() -> str:
    from datetime import datetime, timezone

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
        for block in iter(
            lambda: handle.read(
                4
                * 1024
                * 1024
            ),
            b"",
        ):
            digest.update(
                block
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
    value: Any,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=(
            path.stem
            + "."
        ),
        suffix=".tmp.json",
        dir=str(
            path.parent
        ),
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
                to_json_safe(
                    value
                ),
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


def to_json_safe(
    value: Any,
) -> Any:
    if isinstance(
        value,
        dict,
    ):
        return {
            str(
                key
            ): to_json_safe(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            to_json_safe(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        np.ndarray,
    ):
        return to_json_safe(
            value.tolist()
        )

    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):
        numeric = float(
            value
        )

        return (
            numeric
            if math.isfinite(
                numeric
            )
            else None
        )

    if isinstance(
        value,
        float,
    ):
        return (
            value
            if math.isfinite(
                value
            )
            else None
        )

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(
            value
        )

    return value


def write_csv_gzip(
    table: pd.DataFrame,
    path: Path,
) -> None:
    compression: Any = {
        "method": "gzip",
        "compresslevel": 6,
        "mtime": 0,
    }

    try:
        table.to_csv(
            path,
            index=False,
            float_format="%.12g",
            compression=compression,
        )

    except TypeError:
        table.to_csv(
            path,
            index=False,
            float_format="%.12g",
            compression="gzip",
        )


def stable_seed(
    *parts: Any,
) -> int:
    payload = "|".join(
        str(
            part
        )
        for part in parts
    )

    digest = hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).digest()

    return int.from_bytes(
        digest[
            :4
        ],
        "little",
    )


def seed_everything(
    seed: int,
) -> None:
    random.seed(
        seed
    )
    np.random.seed(
        seed
    )
    torch.manual_seed(
        seed
    )
    torch.cuda.manual_seed_all(
        seed
    )


def assert_cuda_runtime() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Quiet CPU fallback is forbidden."
        )

    if torch.cuda.device_count() < 1:
        raise RuntimeError(
            "No CUDA device is visible."
        )

    torch.cuda.set_device(
        0
    )

    torch.use_deterministic_algorithms(
        True,
        warn_only=False,
    )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if hasattr(
        torch.backends,
        "cuda",
    ) and hasattr(
        torch.backends.cuda,
        "matmul",
    ):
        torch.backends.cuda.matmul.allow_tf32 = False

    if hasattr(
        torch.backends.cudnn,
        "allow_tf32",
    ):
        torch.backends.cudnn.allow_tf32 = False

    if hasattr(
        torch,
        "set_float32_matmul_precision",
    ):
        torch.set_float32_matmul_precision(
            "highest"
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


def load_linear_helpers() -> Any:
    require_file(
        LINEAR_SCRIPT
    )

    module_name = (
        "paper5_05c1_neural_metric_helpers"
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        LINEAR_SCRIPT,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Could not load 05c1 helpers."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        module_name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


# ---------------------------------------------------------------------
# Frozen upstream verification
# ---------------------------------------------------------------------

def verify_generated_files(
    lock: dict[str, Any],
    label: str,
) -> None:
    failures: list[str] = []

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


def verify_upstream_state() -> dict[str, Any]:
    required_paths = [
        CONTRACT_05A,
        SIMULATION_MANIFEST,
        LINEAR_SCRIPT,
        ENV_CONTRACT,
        ENV_LOCK,
        PROTOCOL_05D1,
        LOSS_CONTRACT,
        SELECTION_CONTRACT,
        LOCK_05D1,
        PROFILE_GRID,
        DATASET_ROLES,
        DEVELOPMENT_PARTITIONS,
        PROFILE_FIT_SEEDS,
        PROFILE_STABILITY,
    ]

    for candidate in required_paths:
        require_file(
            candidate
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
            "Final gate-calibration simulations were materialized before profile freeze."
        )

    future_found = [
        relative_path(
            candidate
        )
        for candidate in FUTURE_RESULT_PATHS
        if is_nonempty_path(
            candidate
        )
    ]

    if future_found:
        raise RuntimeError(
            "05d2 must precede all 05d3+ neural result stages. Found:\n- "
            + "\n- ".join(
                future_found
            )
        )

    env_contract = load_json(
        ENV_CONTRACT
    )

    env_lock = load_json(
        ENV_LOCK
    )

    if clean_text(
        env_lock.get(
            "status"
        )
    ) != (
        "FROZEN_NEURAL_RUNTIME_ENVIRONMENT_BEFORE_NEURAL_MODEL_FITTING"
    ):
        raise RuntimeError(
            "Unexpected 05d0 lock status."
        )

    expected_env_hash = clean_text(
        env_lock.get(
            "contract",
            {},
        ).get(
            "sha256"
        )
    )

    if (
        not expected_env_hash
        or sha256_file(
            ENV_CONTRACT
        )
        != expected_env_hash
    ):
        raise RuntimeError(
            "05d0 environment contract hash mismatch."
        )

    verify_generated_files(
        env_lock,
        "05d0",
    )

    protocol_05d1 = load_json(
        PROTOCOL_05D1
    )

    loss_contract = load_json(
        LOSS_CONTRACT
    )

    selection_contract = load_json(
        SELECTION_CONTRACT
    )

    lock_05d1 = load_json(
        LOCK_05D1
    )

    if clean_text(
        lock_05d1.get(
            "status"
        )
    ) != (
        "FROZEN_NEURAL_SIMULATION_PROTOCOL_BEFORE_NEURAL_MODEL_FITTING"
    ):
        raise RuntimeError(
            "Unexpected 05d1 lock status."
        )

    expected_protocol_hash = clean_text(
        lock_05d1.get(
            "protocol",
            {},
        ).get(
            "sha256"
        )
    )

    if (
        not expected_protocol_hash
        or sha256_file(
            PROTOCOL_05D1
        )
        != expected_protocol_hash
    ):
        raise RuntimeError(
            "05d1 protocol hash mismatch."
        )

    verify_generated_files(
        lock_05d1,
        "05d1",
    )

    scientific_state = protocol_05d1.get(
        "scientific_state",
        {},
    )

    forbidden_true = [
        "outcomes_loaded",
        "clinical_labels_loaded",
        "biological_annotations_loaded",
        "TARGET_model_fitting_performed",
        "neural_simulation_model_fitting_performed",
        "neural_profile_selected",
        "architecture_selected",
        "stability_gate_thresholds_retuned",
        "final_gate_calibration_data_used",
    ]

    active = [
        key
        for key in forbidden_true
        if bool(
            scientific_state.get(
                key,
                False,
            )
        )
    ]

    if active:
        raise RuntimeError(
            "05d1 pre-fit scientific state violated: "
            + ", ".join(
                active
            )
        )

    gpu_contract = protocol_05d1.get(
        "gpu_execution",
        {},
    )

    if (
        not bool(
            gpu_contract.get(
                "required_for_heavy_neural_compute",
                False,
            )
        )
        or clean_text(
            gpu_contract.get(
                "device"
            )
        )
        != "cuda:0"
        or bool(
            gpu_contract.get(
                "quiet_cpu_fallback_allowed",
                True,
            )
        )
    ):
        raise RuntimeError(
            "05d1 GPU execution contract changed."
        )

    if clean_text(
        env_contract.get(
            "device_freeze",
            {},
        ).get(
            "confirmatory_device"
        )
    ) != "cuda:0":
        raise RuntimeError(
            "05d0 did not freeze cuda:0."
        )

    frozen_software = env_contract.get(
        "software_freeze",
        {},
    )

    frozen_torch = clean_text(
        frozen_software.get(
            "torch_version"
        )
    )

    frozen_cuda = clean_text(
        frozen_software.get(
            "torch_cuda_build_version"
        )
    )

    if clean_text(
        torch.__version__
    ) != frozen_torch:
        raise RuntimeError(
            "Active PyTorch version differs from 05d0: "
            f"active={torch.__version__}, frozen={frozen_torch}"
        )

    if clean_text(
        torch.version.cuda
    ) != frozen_cuda:
        raise RuntimeError(
            "Active PyTorch CUDA build differs from 05d0: "
            f"active={torch.version.cuda}, frozen={frozen_cuda}"
        )

    assert_cuda_runtime()

    frozen_gpu = clean_text(
        env_contract.get(
            "device_freeze",
            {},
        ).get(
            "gpu_name"
        )
    )

    active_gpu = clean_text(
        torch.cuda.get_device_name(
            0
        )
    )

    if active_gpu != frozen_gpu:
        raise RuntimeError(
            "Active GPU differs from 05d0: "
            f"active={active_gpu}, frozen={frozen_gpu}"
        )

    profile_grid = pd.read_csv(
        PROFILE_GRID,
        dtype=str,
    ).fillna(
        ""
    )

    roles = pd.read_csv(
        DATASET_ROLES,
        dtype=str,
        low_memory=False,
    ).fillna(
        ""
    )

    development_partitions = pd.read_csv(
        DEVELOPMENT_PARTITIONS,
        dtype=str,
    ).fillna(
        ""
    )

    profile_fit_seeds = pd.read_csv(
        PROFILE_FIT_SEEDS,
        dtype=str,
    ).fillna(
        ""
    )

    profile_stability = pd.read_csv(
        PROFILE_STABILITY,
        dtype=str,
        low_memory=False,
    ).fillna(
        ""
    )

    simulation_manifest = pd.read_csv(
        SIMULATION_MANIFEST,
        dtype=str,
        low_memory=False,
    ).fillna(
        ""
    )

    if profile_grid.shape[
        0
    ] != EXPECTED_PROFILES:
        raise RuntimeError(
            f"Expected {EXPECTED_PROFILES} profiles; found {profile_grid.shape[0]}."
        )

    development_roles = roles[
        roles[
            "neural_dataset_role"
        ].eq(
            "profile_development"
        )
    ].copy()

    if development_roles.shape[
        0
    ] != EXPECTED_DEVELOPMENT_DATASETS:
        raise RuntimeError(
            "05d1 development dataset count is not 360."
        )

    if development_partitions.shape[
        0
    ] != EXPECTED_DEVELOPMENT_DATASETS:
        raise RuntimeError(
            "05d1 development partition count is not 360."
        )

    if profile_fit_seeds.shape[
        0
    ] != EXPECTED_SHARDS:
        raise RuntimeError(
            "05d1 profile fit-seed registry row count is not 3,240."
        )

    expected_stability_rows = (
        EXPECTED_SHARDS
        * STABILITY_RESAMPLES
    )

    if profile_stability.shape[
        0
    ] != expected_stability_rows:
        raise RuntimeError(
            "05d1 profile stability registry row count mismatch."
        )

    if (
        simulation_manifest.shape[
            0
        ]
        != 720
    ):
        raise RuntimeError(
            "Frozen simulation manifest no longer contains 720 datasets."
        )

    if bool(
        loss_contract.get(
            "precision",
            {},
        ).get(
            "automatic_mixed_precision",
            True,
        )
    ):
        raise RuntimeError(
            "05d1 unexpectedly permits AMP."
        )

    if bool(
        loss_contract.get(
            "precision",
            {},
        ).get(
            "tf32",
            True,
        )
    ):
        raise RuntimeError(
            "05d1 unexpectedly permits TF32."
        )

    if bool(
        loss_contract.get(
            "precision",
            {},
        ).get(
            "torch_compile",
            True,
        )
    ):
        raise RuntimeError(
            "05d1 unexpectedly permits torch.compile."
        )

    return {
        "env_contract": env_contract,
        "protocol_05d1": (
            protocol_05d1
        ),
        "loss_contract": (
            loss_contract
        ),
        "selection_contract": (
            selection_contract
        ),
        "profile_grid": (
            profile_grid
        ),
        "roles": roles,
        "development_roles": (
            development_roles
        ),
        "development_partitions": (
            development_partitions
        ),
        "profile_fit_seeds": (
            profile_fit_seeds
        ),
        "profile_stability": (
            profile_stability
        ),
        "simulation_manifest": (
            simulation_manifest
        ),
        "hashes": {
            relative_path(
                candidate
            ): sha256_file(
                candidate
            )
            for candidate in [
                CONTRACT_05A,
                SIMULATION_MANIFEST,
                LINEAR_SCRIPT,
                ENV_CONTRACT,
                ENV_LOCK,
                PROTOCOL_05D1,
                LOSS_CONTRACT,
                SELECTION_CONTRACT,
                LOCK_05D1,
                PROFILE_GRID,
                DATASET_ROLES,
                DEVELOPMENT_PARTITIONS,
                PROFILE_FIT_SEEDS,
                PROFILE_STABILITY,
            ]
        },
    }


# ---------------------------------------------------------------------
# Frozen profile parsing
# ---------------------------------------------------------------------

def _int_value(
    row: pd.Series,
    column: str,
) -> int:
    return int(
        float(
            row[
                column
            ]
        )
    )


def _float_value(
    row: pd.Series,
    column: str,
) -> float:
    return float(
        row[
            column
        ]
    )


def parse_profile(
    row: pd.Series,
) -> ProfileSpec:
    required = {
        "profile_id",
        "architecture_profile",
        "regularization_profile",
        "architecture_complexity_rank",
        "regularization_complexity_rank",
        "combined_complexity_rank",
        "arch_rna_pca_components",
        "arch_methylation_pca_components",
        "arch_cnv_pca_components",
        "arch_shared_latent_dim",
        "arch_private_latent_dim_per_view",
        "arch_encoder_hidden_dim",
        "arch_decoder_hidden_dim",
        "arch_maximum_epochs",
        "arch_early_stopping_patience",
        "arch_early_stopping_min_delta",
        "reg_reconstruction_weight",
        "reg_shared_alignment_weight",
        "reg_contrastive_weight",
        "reg_shared_private_orthogonality_weight",
        "reg_private_separation_weight",
        "reg_cross_reconstruction_weight",
        "reg_modality_dropout_probability",
        "reg_contrastive_temperature",
        "reg_learning_rate",
        "reg_weight_decay",
    }

    missing = (
        required
        - set(
            row.index
        )
    )

    if missing:
        raise RuntimeError(
            "Profile grid missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    encoder_hidden = _int_value(
        row,
        "arch_encoder_hidden_dim",
    )

    decoder_hidden = _int_value(
        row,
        "arch_decoder_hidden_dim",
    )

    if encoder_hidden != decoder_hidden:
        raise RuntimeError(
            "05d2 implementation requires equal frozen encoder/decoder hidden dimensions."
        )

    return ProfileSpec(
        profile_id=clean_text(
            row[
                "profile_id"
            ]
        ),
        architecture_profile=clean_text(
            row[
                "architecture_profile"
            ]
        ),
        regularization_profile=clean_text(
            row[
                "regularization_profile"
            ]
        ),
        pca_components={
            "RNA": _int_value(
                row,
                "arch_rna_pca_components",
            ),
            "METHYLATION": _int_value(
                row,
                "arch_methylation_pca_components",
            ),
            "CNV": _int_value(
                row,
                "arch_cnv_pca_components",
            ),
        },
        shared_dim=_int_value(
            row,
            "arch_shared_latent_dim",
        ),
        private_dim=_int_value(
            row,
            "arch_private_latent_dim_per_view",
        ),
        hidden_dim=(
            encoder_hidden
        ),
        maximum_epochs=_int_value(
            row,
            "arch_maximum_epochs",
        ),
        patience=_int_value(
            row,
            "arch_early_stopping_patience",
        ),
        min_delta=_float_value(
            row,
            "arch_early_stopping_min_delta",
        ),
        reconstruction_weight=_float_value(
            row,
            "reg_reconstruction_weight",
        ),
        shared_alignment_weight=_float_value(
            row,
            "reg_shared_alignment_weight",
        ),
        contrastive_weight=_float_value(
            row,
            "reg_contrastive_weight",
        ),
        orthogonality_weight=_float_value(
            row,
            "reg_shared_private_orthogonality_weight",
        ),
        private_separation_weight=_float_value(
            row,
            "reg_private_separation_weight",
        ),
        cross_reconstruction_weight=_float_value(
            row,
            "reg_cross_reconstruction_weight",
        ),
        modality_dropout_probability=_float_value(
            row,
            "reg_modality_dropout_probability",
        ),
        contrastive_temperature=_float_value(
            row,
            "reg_contrastive_temperature",
        ),
        learning_rate=_float_value(
            row,
            "reg_learning_rate",
        ),
        weight_decay=_float_value(
            row,
            "reg_weight_decay",
        ),
        combined_complexity_rank=_int_value(
            row,
            "combined_complexity_rank",
        ),
        architecture_complexity_rank=_int_value(
            row,
            "architecture_complexity_rank",
        ),
        regularization_complexity_rank=_int_value(
            row,
            "regularization_complexity_rank",
        ),
    )


def build_profile_lookup(
    table: pd.DataFrame,
) -> dict[str, ProfileSpec]:
    output: dict[
        str,
        ProfileSpec,
    ] = {}

    for _, row in table.iterrows():
        profile = parse_profile(
            row
        )

        if profile.profile_id in output:
            raise RuntimeError(
                f"Duplicate profile ID: {profile.profile_id}"
            )

        output[
            profile.profile_id
        ] = profile

    if len(
        output
    ) != EXPECTED_PROFILES:
        raise RuntimeError(
            "Profile lookup does not contain 9 profiles."
        )

    return output


# ---------------------------------------------------------------------
# Frozen registry lookup
# ---------------------------------------------------------------------

def build_manifest_lookup(
    table: pd.DataFrame,
) -> dict[
    tuple[str, int, int],
    pd.Series,
]:
    output: dict[
        tuple[str, int, int],
        pd.Series,
    ] = {}

    for _, row in table.iterrows():
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

        if key in output:
            raise RuntimeError(
                f"Duplicate simulation manifest key: {key}"
            )

        output[
            key
        ] = row

    return output


def build_partition_lookup(
    table: pd.DataFrame,
) -> dict[
    tuple[str, int, int],
    dict[str, np.ndarray],
]:
    output = {}

    for _, row in table.iterrows():
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

        def parse_indices(
            text: str,
        ) -> np.ndarray:
            values = [
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

            return np.asarray(
                values,
                dtype=int,
            )

        output[
            key
        ] = {
            "fit": parse_indices(
                row[
                    "fit_indices"
                ]
            ),
            "early_stop": parse_indices(
                row[
                    "early_stop_indices"
                ]
            ),
            "validation": parse_indices(
                row[
                    "validation_indices"
                ]
            ),
        }

    if len(
        output
    ) != EXPECTED_DEVELOPMENT_DATASETS:
        raise RuntimeError(
            "Development partition lookup does not contain 360 datasets."
        )

    return output


def build_profile_seed_lookup(
    table: pd.DataFrame,
) -> dict[
    tuple[str, int, int, str],
    int,
]:
    output = {}

    for _, row in table.iterrows():
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
            clean_text(
                row[
                    "profile_id"
                ]
            ),
        )

        if key in output:
            raise RuntimeError(
                f"Duplicate profile fit seed: {key}"
            )

        output[
            key
        ] = int(
            float(
                row[
                    "fit_seed"
                ]
            )
        )

    if len(
        output
    ) != EXPECTED_SHARDS:
        raise RuntimeError(
            "Profile fit-seed lookup does not contain 3,240 rows."
        )

    return output


def build_stability_lookup(
    table: pd.DataFrame,
) -> dict[
    tuple[str, int, int, str],
    dict[str, Any],
]:
    output: dict[
        tuple[str, int, int, str],
        dict[str, Any],
    ] = {}

    grouped = table.groupby(
        [
            "scenario_id",
            "replicate_index",
            "simulation_seed",
            "profile_id",
        ],
        sort=False,
    )

    for keys, part in grouped:
        scenario_id = clean_text(
            keys[
                0
            ]
        )

        replicate_index = int(
            float(
                keys[
                    1
                ]
            )
        )

        simulation_seed = int(
            float(
                keys[
                    2
                ]
            )
        )

        profile_id = clean_text(
            keys[
                3
            ]
        )

        part = part.copy()

        part[
            "resample_index"
        ] = pd.to_numeric(
            part[
                "resample_index"
            ],
            errors="raise",
        ).astype(
            int
        )

        part = part.sort_values(
            "resample_index",
            kind="mergesort",
        )

        if part.shape[
            0
        ] != STABILITY_RESAMPLES:
            raise RuntimeError(
                "Every dataset-profile stability group must contain 20 resamples."
            )

        reference_seeds = set(
            pd.to_numeric(
                part[
                    "reference_fit_seed"
                ],
                errors="raise",
            ).astype(
                int
            )
        )

        if len(
            reference_seeds
        ) != 1:
            raise RuntimeError(
                "Stability reference seed is not unique within a dataset-profile group."
            )

        output[
            (
                scenario_id,
                replicate_index,
                simulation_seed,
                profile_id,
            )
        ] = {
            "reference_fit_seed": int(
                next(
                    iter(
                        reference_seeds
                    )
                )
            ),
            "rows": part,
        }

    if len(
        output
    ) != EXPECTED_SHARDS:
        raise RuntimeError(
            "Stability lookup does not contain 3,240 dataset-profile groups."
        )

    return output


# ---------------------------------------------------------------------
# Fold-local budgeted preprocessing and deterministic dual PCA
# ---------------------------------------------------------------------

def fit_view_preprocessor(
    linear: Any,
    view: str,
    training_matrix: np.ndarray,
    feature_budget: int,
    pca_target_components: int,
) -> NeuralViewPreprocessor:
    values = np.asarray(
        training_matrix,
        dtype=np.float64,
    )

    if values.ndim != 2:
        raise RuntimeError(
            f"{view}: training matrix is not two-dimensional."
        )

    finite = np.isfinite(
        values
    )

    n_finite = finite.sum(
        axis=0
    )

    missing_fraction = (
        1.0
        - n_finite
        / values.shape[
            0
        ]
    )

    with np.errstate(
        all="ignore"
    ):
        pre_variance = np.nanvar(
            values,
            axis=0,
            ddof=1,
        )

    eligible = (
        (
            n_finite
            >= MIN_FINITE_VALUES
        )
        & (
            missing_fraction
            <= MAX_FEATURE_MISSING_FRACTION
            + 1e-12
        )
        & np.isfinite(
            pre_variance
        )
        & (
            pre_variance
            > MIN_PREIMPUTATION_VARIANCE
        )
    )

    eligible_indices = np.flatnonzero(
        eligible
    )

    if eligible_indices.size == 0:
        raise RuntimeError(
            f"{view}: no fold-local eligible features."
        )

    # Prespecified ranking:
    # pre-standardization variance descending,
    # original feature index ascending as deterministic tie-breaker.
    order = np.lexsort(
        (
            eligible_indices,
            -pre_variance[
                eligible_indices
            ],
        )
    )

    ranked_indices = eligible_indices[
        order
    ]

    selected_indices = ranked_indices[
        : min(
            int(
                feature_budget
            ),
            ranked_indices.size,
        )
    ]

    selected = values[
        :,
        selected_indices,
    ].copy()

    with np.errstate(
        all="ignore"
    ):
        medians = np.nanmedian(
            selected,
            axis=0,
        )

    if not np.isfinite(
        medians
    ).all():
        raise RuntimeError(
            f"{view}: non-finite median after eligibility filtering."
        )

    missing = ~np.isfinite(
        selected
    )

    if missing.any():
        row_index, column_index = np.where(
            missing
        )

        selected[
            row_index,
            column_index,
        ] = medians[
            column_index
        ]

    means = np.mean(
        selected,
        axis=0,
    )

    scales = np.std(
        selected,
        axis=0,
        ddof=1,
    )

    numerical_keep = (
        np.isfinite(
            means
        )
        & np.isfinite(
            scales
        )
        & (
            scales
            > MIN_POSTIMPUTATION_STD
        )
    )

    if not numerical_keep.any():
        raise RuntimeError(
            f"{view}: no selected features survived numerical checks."
        )

    selected_indices = selected_indices[
        numerical_keep
    ]

    medians = medians[
        numerical_keep
    ]

    means = means[
        numerical_keep
    ]

    scales = scales[
        numerical_keep
    ]

    selected = selected[
        :,
        numerical_keep,
    ]

    standardized = (
        selected
        - means
    ) / scales

    if not np.isfinite(
        standardized
    ).all():
        raise RuntimeError(
            f"{view}: non-finite standardized training data."
        )

    maximum_rank = min(
        int(
            pca_target_components
        ),
        max(
            1,
            standardized.shape[
                0
            ]
            - 1,
        ),
        standardized.shape[
            1
        ],
    )

    # The numerically heavy Gram/eigendecomposition and PCA projection
    # run on the frozen CUDA device. Feature eligibility/ranking and
    # median imputation remain deterministic CPU operations.
    pca = fit_dual_pca_gpu(
        standardized,
        maximum_rank,
    )

    actual_components = min(
        int(
            pca_target_components
        ),
        int(
            pca.components.shape[
                0
            ]
        ),
    )

    if actual_components < 1:
        raise RuntimeError(
            f"{view}: PCA has zero retained components."
        )

    return NeuralViewPreprocessor(
        view=view,
        selected_indices=(
            selected_indices.astype(
                np.int64
            )
        ),
        medians=medians,
        means=means,
        scales=scales,
        pca=pca,
        pca_components=(
            actual_components
        ),
        n_eligible_before_budget=int(
            eligible_indices.size
        ),
        feature_budget=int(
            feature_budget
        ),
    )


def fit_preprocessors(
    linear: Any,
    arrays: dict[str, np.ndarray],
    training_indices: np.ndarray,
    profile: ProfileSpec,
) -> dict[
    str,
    NeuralViewPreprocessor,
]:
    output: dict[
        str,
        NeuralViewPreprocessor,
    ] = {}

    for view in VIEWS:
        output[
            view
        ] = fit_view_preprocessor(
            linear=linear,
            view=view,
            training_matrix=(
                arrays[
                    f"X_{VIEW_PREFIX[view]}"
                ][
                    training_indices,
                    :,
                ]
            ),
            feature_budget=(
                PRIMARY_FEATURE_BUDGETS[
                    view
                ]
            ),
            pca_target_components=(
                profile.pca_components[
                    view
                ]
            ),
        )

    return output


def transform_views(
    arrays: dict[str, np.ndarray],
    preprocessors: dict[
        str,
        NeuralViewPreprocessor,
    ],
    indices: np.ndarray,
    *,
    complete: bool,
) -> dict[
    str,
    np.ndarray,
]:
    output: dict[
        str,
        np.ndarray,
    ] = {}

    for view in VIEWS:
        prefix = VIEW_PREFIX[
            view
        ]

        matrix = arrays[
            (
                f"X_complete_{prefix}"
                if complete
                else f"X_{prefix}"
            )
        ][
            indices,
            :,
        ]

        if complete:
            output[
                view
            ] = preprocessors[
                view
            ].transform_complete(
                matrix
            )

        else:
            output[
                view
            ] = preprocessors[
                view
            ].transform_observed(
                matrix
            )

    return output


def view_availability(
    arrays: dict[str, np.ndarray],
    preprocessors: dict[
        str,
        NeuralViewPreprocessor,
    ],
    indices: np.ndarray,
) -> np.ndarray:
    availability = np.zeros(
        (
            indices.shape[
                0
            ],
            len(
                VIEWS
            ),
        ),
        dtype=bool,
    )

    for view in VIEWS:
        prefix = VIEW_PREFIX[
            view
        ]

        selected_indices = preprocessors[
            view
        ].selected_indices

        matrix = arrays[
            f"X_{prefix}"
        ][
            indices,
            :,
        ][
            :,
            selected_indices,
        ]

        availability[
            :,
            VIEW_INDEX[
                view
            ],
        ] = np.any(
            np.isfinite(
                matrix
            ),
            axis=1,
        )

    if np.any(
        availability.sum(
            axis=1
        )
        < 1
    ):
        raise RuntimeError(
            "At least one sample has zero available modalities after feature selection."
        )

    return availability


def preprocessing_audit_rows(
    arrays: dict[str, np.ndarray],
    preprocessors: dict[
        str,
        NeuralViewPreprocessor,
    ],
    training_indices: np.ndarray,
    profile: ProfileSpec,
    stage: str,
) -> list[
    dict[str, Any]
]:
    rows = []

    for view in VIEWS:
        prep = preprocessors[
            view
        ]

        selected_raw = arrays[
            f"X_{VIEW_PREFIX[view]}"
        ][
            training_indices,
            :,
        ][
            :,
            prep.selected_indices,
        ]

        rows.append(
            {
                "preprocessing_stage": (
                    stage
                ),
                "profile_id": (
                    profile.profile_id
                ),
                "view": view,
                "n_training_samples": int(
                    training_indices.shape[
                        0
                    ]
                ),
                "n_input_features": int(
                    arrays[
                        f"X_{VIEW_PREFIX[view]}"
                    ].shape[
                        1
                    ]
                ),
                "n_eligible_before_budget": (
                    prep.n_eligible_before_budget
                ),
                "feature_budget": (
                    prep.feature_budget
                ),
                "n_selected_features": int(
                    prep.selected_indices.shape[
                        0
                    ]
                ),
                "target_pca_components": int(
                    profile.pca_components[
                        view
                    ]
                ),
                "actual_pca_components": int(
                    prep.pca_components
                ),
                "selected_feature_missing_fraction_mean": float(
                    np.mean(
                        ~np.isfinite(
                            selected_raw
                        )
                    )
                ),
                "imputation": (
                    "training_subset_feature_median"
                ),
                "ranking": (
                    "training_subset_prestandardization_variance"
                ),
                "post_zscore_variance_ranking": False,
                "PCA_solver": (
                    "CUDA_float64_dual_PCA_with_05c1_sign_convention"
                ),
            }
        )

    return rows


# ---------------------------------------------------------------------
# CUDA SABLE-MO profile-screen model
# ---------------------------------------------------------------------

class ViewEncoder(
    nn.Module
):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        shared_dim: int,
        private_dim: int,
    ) -> None:
        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.LayerNorm(
                hidden_dim
            ),
            nn.GELU(),
        )

        self.shared_head = nn.Linear(
            hidden_dim,
            shared_dim,
        )

        self.private_head = nn.Linear(
            hidden_dim,
            private_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden = self.body(
            x
        )

        return (
            self.shared_head(
                hidden
            ),
            self.private_head(
                hidden
            ),
        )


class ViewDecoder(
    nn.Module
):
    def __init__(
        self,
        shared_dim: int,
        private_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()

        self.first = nn.Linear(
            shared_dim
            + private_dim,
            hidden_dim,
        )

        self.norm = nn.LayerNorm(
            hidden_dim
        )

        self.activation = nn.GELU()

        self.output = nn.Linear(
            hidden_dim,
            output_dim,
        )

    def forward(
        self,
        shared: torch.Tensor,
        private: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat(
            [
                shared,
                private,
            ],
            dim=1,
        )

        hidden = self.first(
            combined
        )

        hidden = self.norm(
            hidden
        )

        hidden = self.activation(
            hidden
        )

        return self.output(
            hidden
        )


class SableMOProfileModel(
    nn.Module
):
    def __init__(
        self,
        input_dims: dict[str, int],
        profile: ProfileSpec,
    ) -> None:
        super().__init__()

        self.profile = profile

        self.encoders = nn.ModuleDict(
            {
                view: ViewEncoder(
                    input_dim=int(
                        input_dims[
                            view
                        ]
                    ),
                    hidden_dim=(
                        profile.hidden_dim
                    ),
                    shared_dim=(
                        profile.shared_dim
                    ),
                    private_dim=(
                        profile.private_dim
                    ),
                )
                for view in VIEWS
            }
        )

        self.decoders = nn.ModuleDict(
            {
                view: ViewDecoder(
                    shared_dim=(
                        profile.shared_dim
                    ),
                    private_dim=(
                        profile.private_dim
                    ),
                    hidden_dim=(
                        profile.hidden_dim
                    ),
                    output_dim=int(
                        input_dims[
                            view
                        ]
                    ),
                )
                for view in VIEWS
            }
        )

        self.reset_frozen_parameters()

    def reset_frozen_parameters(
        self,
    ) -> None:
        for module in self.modules():
            if isinstance(
                module,
                nn.Linear,
            ):
                nn.init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(
                module,
                nn.LayerNorm,
            ):
                if module.weight is not None:
                    nn.init.ones_(
                        module.weight
                    )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

    def encode(
        self,
        views: dict[
            str,
            torch.Tensor,
        ],
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        shared: dict[
            str,
            torch.Tensor,
        ] = {}

        private: dict[
            str,
            torch.Tensor,
        ] = {}

        for view in VIEWS:
            (
                shared[
                    view
                ],
                private[
                    view
                ],
            ) = self.encoders[
                view
            ](
                views[
                    view
                ]
            )

        return (
            shared,
            private,
        )

    def patient_shared(
        self,
        shared: dict[
            str,
            torch.Tensor,
        ],
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        stack = torch.stack(
            [
                shared[
                    view
                ]
                for view in VIEWS
            ],
            dim=1,
        )

        mask = active_mask.to(
            dtype=stack.dtype
        ).unsqueeze(
            -1
        )

        denominator = mask.sum(
            dim=1
        ).clamp_min(
            1.0
        )

        return (
            stack
            * mask
        ).sum(
            dim=1
        ) / denominator

    def decoder_shared_energy(
        self,
    ) -> np.ndarray:
        energy = np.zeros(
            (
                self.profile.shared_dim,
                len(
                    VIEWS
                ),
            ),
            dtype=np.float64,
        )

        with torch.no_grad():
            for view in VIEWS:
                weight = (
                    self.decoders[
                        view
                    ].first.weight[
                        :,
                        : self.profile.shared_dim,
                    ]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )

                energy[
                    :,
                    VIEW_INDEX[
                        view
                    ],
                ] = np.sum(
                    weight
                    ** 2,
                    axis=0,
                )

        total = np.sum(
            energy,
            axis=1,
            keepdims=True,
        )

        return np.divide(
            energy,
            total,
            out=np.zeros_like(
                energy
            ),
            where=(
                total
                > 0
            ),
        )


# ---------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------

def numpy_views_to_cuda(
    views: dict[str, np.ndarray],
) -> dict[
    str,
    torch.Tensor,
]:
    output = {}

    for view in VIEWS:
        tensor = torch.as_tensor(
            views[
                view
            ],
            dtype=torch.float32,
            device="cuda:0",
        )

        if tensor.device.type != "cuda":
            raise RuntimeError(
                "Heavy neural tensor was not placed on CUDA."
            )

        output[
            view
        ] = tensor

    return output


def availability_to_cuda(
    availability: np.ndarray,
) -> torch.Tensor:
    tensor = torch.as_tensor(
        availability,
        dtype=torch.bool,
        device="cuda:0",
    )

    if tensor.device.type != "cuda":
        raise RuntimeError(
            "Availability tensor was not placed on CUDA."
        )

    return tensor


def calculate_null_mse(
    views: dict[
        str,
        torch.Tensor,
    ],
) -> dict[
    str,
    float,
]:
    output = {}

    for view in VIEWS:
        value = float(
            torch.mean(
                views[
                    view
                ]
                ** 2
            )
            .detach()
            .cpu()
        )

        if (
            not math.isfinite(
                value
            )
            or value
            <= 0
        ):
            value = (
                NULL_MSE_FLOOR
            )

        output[
            view
        ] = max(
            value,
            NULL_MSE_FLOOR,
        )

    return output


def normalized_entropy(
    fractions: np.ndarray,
) -> np.ndarray:
    output = np.zeros(
        fractions.shape[
            0
        ],
        dtype=np.float64,
    )

    for index in range(
        fractions.shape[
            0
        ]
    ):
        row = fractions[
            index,
            :,
        ]

        positive = (
            row
            > 0
        )

        if positive.any():
            output[
                index
            ] = -float(
                np.sum(
                    row[
                        positive
                    ]
                    * np.log(
                        row[
                            positive
                        ]
                    )
                )
                / math.log(
                    len(
                        VIEWS
                    )
                )
            )

    return output


# ---------------------------------------------------------------------
# Frozen V4 objective
# ---------------------------------------------------------------------

def zero_scalar(
    reference: torch.Tensor,
) -> torch.Tensor:
    return torch.zeros(
        (),
        dtype=reference.dtype,
        device=reference.device,
    )


def covariance_penalty(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    if left.shape[
        0
    ] < 2:
        return zero_scalar(
            left
        )

    left_centered = (
        left
        - left.mean(
            dim=0,
            keepdim=True,
        )
    )

    right_centered = (
        right
        - right.mean(
            dim=0,
            keepdim=True,
        )
    )

    covariance = (
        left_centered.T
        @ right_centered
    ) / max(
        1,
        left.shape[
            0
        ]
        - 1,
    )

    # Frozen contract: squared Frobenius norm.
    return torch.sum(
        covariance
        ** 2
    )


def symmetric_info_nce(
    left: torch.Tensor,
    right: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if left.shape[
        0
    ] < 2:
        return zero_scalar(
            left
        )

    left_normalized = F.normalize(
        left,
        p=2,
        dim=1,
        eps=1e-8,
    )

    right_normalized = F.normalize(
        right,
        p=2,
        dim=1,
        eps=1e-8,
    )

    logits = (
        left_normalized
        @ right_normalized.T
    ) / float(
        temperature
    )

    targets = torch.arange(
        logits.shape[
            0
        ],
        dtype=torch.long,
        device=logits.device,
    )

    return 0.5 * (
        F.cross_entropy(
            logits,
            targets,
        )
        + F.cross_entropy(
            logits.T,
            targets,
        )
    )


def make_modality_dropout(
    availability: torch.Tensor,
    probability: float,
    fit_seed: int,
    epoch: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    # Entire dropout mask construction remains on CUDA and contains no
    # device-to-host synchronization. The random-number order matches v1:
    # one N-vector draw for the trigger, followed by one N x 3 draw for
    # the uniformly selected available modality.
    if probability <= 0:
        return (
            availability,
            torch.zeros_like(
                availability,
                dtype=torch.bool,
            ),
        )

    available_count = availability.sum(
        dim=1
    )

    candidates = (
        available_count
        >= 2
    )

    generator = torch.Generator(
        device="cuda"
    )

    generator.manual_seed(
        stable_seed(
            fit_seed,
            "modality_dropout",
            epoch,
        )
    )

    trigger = (
        torch.rand(
            (
                availability.shape[
                    0
                ],
            ),
            generator=generator,
            device=availability.device,
            dtype=torch.float32,
        )
        < float(
            probability
        )
    ) & candidates

    random_values = torch.rand(
        availability.shape,
        generator=generator,
        device=availability.device,
        dtype=torch.float32,
    )

    random_values = torch.where(
        availability,
        random_values,
        torch.full_like(
            random_values,
            2.0,
        ),
    )

    selected_view = torch.argmin(
        random_values,
        dim=1,
    )

    selected_one_hot = F.one_hot(
        selected_view,
        num_classes=len(
            VIEWS
        ),
    ).to(
        dtype=torch.bool
    )

    dropped = (
        selected_one_hot
        & trigger.unsqueeze(
            1
        )
    )

    active = (
        availability
        & ~dropped
    )

    # By construction trigger can only be true when >=2 modalities were
    # originally available, so at least one modality always remains.
    return (
        active,
        dropped,
    )


def masked_covariance_penalty(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    weights = mask.to(
        dtype=left.dtype
    )

    count = weights.sum()

    denominator_mean = count.clamp_min(
        1.0
    )

    left_mean = (
        (
            left
            * weights.unsqueeze(
                1
            )
        ).sum(
            dim=0,
            keepdim=True,
        )
        / denominator_mean
    )

    right_mean = (
        (
            right
            * weights.unsqueeze(
                1
            )
        ).sum(
            dim=0,
            keepdim=True,
        )
        / denominator_mean
    )

    left_centered = (
        left
        - left_mean
    ) * weights.unsqueeze(
        1
    )

    right_centered = (
        right
        - right_mean
    ) * weights.unsqueeze(
        1
    )

    covariance = (
        left_centered.T
        @ right_centered
    ) / (
        count
        - 1.0
    ).clamp_min(
        1.0
    )

    valid = (
        count
        >= 2.0
    ).to(
        dtype=left.dtype
    )

    return (
        torch.sum(
            covariance
            ** 2
        )
        * valid,
        valid,
    )


def masked_symmetric_info_nce(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    left_normalized = F.normalize(
        left,
        p=2,
        dim=1,
        eps=1e-8,
    )

    right_normalized = F.normalize(
        right,
        p=2,
        dim=1,
        eps=1e-8,
    )

    logits = (
        left_normalized
        @ right_normalized.T
    ) / float(
        temperature
    )

    valid = mask.to(
        dtype=logits.dtype
    )

    count = valid.sum()

    pair_valid = (
        count
        >= 2.0
    ).to(
        dtype=logits.dtype
    )

    # Mask invalid candidate columns without creating a dynamic-length
    # CUDA tensor. This is algebraically the same InfoNCE candidate set
    # as subsetting to the available patients.
    negative_large = torch.full_like(
        logits,
        -1.0e9,
    )

    candidate_mask = mask.unsqueeze(
        0
    )

    logits_lr = torch.where(
        candidate_mask,
        logits,
        negative_large,
    )

    logits_rl = torch.where(
        candidate_mask,
        logits.T,
        negative_large,
    )

    positive = torch.diagonal(
        logits
    )

    loss_lr_per_row = (
        -positive
        + torch.logsumexp(
            logits_lr,
            dim=1,
        )
    )

    loss_rl_per_row = (
        -positive
        + torch.logsumexp(
            logits_rl,
            dim=1,
        )
    )

    denominator = count.clamp_min(
        1.0
    )

    loss_lr = (
        loss_lr_per_row
        * valid
    ).sum() / denominator

    loss_rl = (
        loss_rl_per_row
        * valid
    ).sum() / denominator

    return (
        0.5
        * (
            loss_lr
            + loss_rl
        )
        * pair_valid,
        pair_valid,
    )


def reconstruction_loss(
    model: SableMOProfileModel,
    views: dict[
        str,
        torch.Tensor,
    ],
    private: dict[
        str,
        torch.Tensor,
    ],
    patient_shared: torch.Tensor,
    active_mask: torch.Tensor,
    null_mse: dict[str, float],
) -> torch.Tensor:
    view_losses = []
    view_validity = []

    for view in VIEWS:
        mask = active_mask[
            :,
            VIEW_INDEX[
                view
            ],
        ].to(
            dtype=patient_shared.dtype
        )

        prediction = model.decoders[
            view
        ](
            patient_shared,
            private[
                view
            ],
        )

        per_sample_mse = torch.mean(
            (
                prediction
                - views[
                    view
                ]
            )
            ** 2,
            dim=1,
        )

        count = mask.sum()

        valid = (
            count
            > 0.0
        ).to(
            dtype=patient_shared.dtype
        )

        mse = (
            per_sample_mse
            * mask
        ).sum() / count.clamp_min(
            1.0
        )

        view_losses.append(
            (
                mse
                / max(
                    float(
                        null_mse[
                            view
                        ]
                    ),
                    NULL_MSE_FLOOR,
                )
            )
            * valid
        )

        view_validity.append(
            valid
        )

    losses = torch.stack(
        view_losses
    )

    validity = torch.stack(
        view_validity
    )

    return losses.sum() / validity.sum().clamp_min(
        1.0
    )


def shared_alignment_loss(
    shared: dict[
        str,
        torch.Tensor,
    ],
    patient_shared: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    view_losses = []
    view_validity = []

    for view in VIEWS:
        mask = active_mask[
            :,
            VIEW_INDEX[
                view
            ],
        ].to(
            dtype=patient_shared.dtype
        )

        per_sample_mse = torch.mean(
            (
                shared[
                    view
                ]
                - patient_shared
            )
            ** 2,
            dim=1,
        )

        count = mask.sum()

        valid = (
            count
            > 0.0
        ).to(
            dtype=patient_shared.dtype
        )

        loss = (
            per_sample_mse
            * mask
        ).sum() / count.clamp_min(
            1.0
        )

        view_losses.append(
            loss
            * valid
        )

        view_validity.append(
            valid
        )

    losses = torch.stack(
        view_losses
    )

    validity = torch.stack(
        view_validity
    )

    return losses.sum() / validity.sum().clamp_min(
        1.0
    )


def orthogonality_loss(
    shared: dict[
        str,
        torch.Tensor,
    ],
    private: dict[
        str,
        torch.Tensor,
    ],
    active_mask: torch.Tensor,
) -> torch.Tensor:
    losses = []
    validity = []

    for view in VIEWS:
        loss, valid = (
            masked_covariance_penalty(
                shared[
                    view
                ],
                private[
                    view
                ],
                active_mask[
                    :,
                    VIEW_INDEX[
                        view
                    ],
                ],
            )
        )

        losses.append(
            loss
        )

        validity.append(
            valid
        )

    losses_tensor = torch.stack(
        losses
    )

    valid_tensor = torch.stack(
        validity
    )

    return losses_tensor.sum() / valid_tensor.sum().clamp_min(
        1.0
    )


def private_separation_loss(
    private: dict[
        str,
        torch.Tensor,
    ],
    active_mask: torch.Tensor,
) -> torch.Tensor:
    losses = []
    validity = []

    for left_index, left_view in enumerate(
        VIEWS
    ):
        for right_view in VIEWS[
            left_index
            + 1 :
        ]:
            mask = (
                active_mask[
                    :,
                    VIEW_INDEX[
                        left_view
                    ],
                ]
                & active_mask[
                    :,
                    VIEW_INDEX[
                        right_view
                    ],
                ]
            )

            loss, valid = (
                masked_covariance_penalty(
                    private[
                        left_view
                    ],
                    private[
                        right_view
                    ],
                    mask,
                )
            )

            losses.append(
                loss
            )

            validity.append(
                valid
            )

    losses_tensor = torch.stack(
        losses
    )

    valid_tensor = torch.stack(
        validity
    )

    return losses_tensor.sum() / valid_tensor.sum().clamp_min(
        1.0
    )


def contrastive_loss(
    shared: dict[
        str,
        torch.Tensor,
    ],
    active_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    losses = []
    validity = []

    for left_index, left_view in enumerate(
        VIEWS
    ):
        for right_view in VIEWS[
            left_index
            + 1 :
        ]:
            mask = (
                active_mask[
                    :,
                    VIEW_INDEX[
                        left_view
                    ],
                ]
                & active_mask[
                    :,
                    VIEW_INDEX[
                        right_view
                    ],
                ]
            )

            loss, valid = (
                masked_symmetric_info_nce(
                    shared[
                        left_view
                    ],
                    shared[
                        right_view
                    ],
                    mask,
                    temperature=(
                        temperature
                    ),
                )
            )

            losses.append(
                loss
            )

            validity.append(
                valid
            )

    losses_tensor = torch.stack(
        losses
    )

    valid_tensor = torch.stack(
        validity
    )

    return losses_tensor.sum() / valid_tensor.sum().clamp_min(
        1.0
    )


def dropped_view_reconstruction_loss(
    model: SableMOProfileModel,
    views: dict[
        str,
        torch.Tensor,
    ],
    patient_shared: torch.Tensor,
    dropped_mask: torch.Tensor,
    null_mse: dict[str, float],
) -> torch.Tensor:
    view_losses = []
    view_validity = []

    zero_private = torch.zeros(
        (
            patient_shared.shape[
                0
            ],
            model.profile.private_dim,
        ),
        dtype=torch.float32,
        device=patient_shared.device,
    )

    for view in VIEWS:
        mask = dropped_mask[
            :,
            VIEW_INDEX[
                view
            ],
        ].to(
            dtype=patient_shared.dtype
        )

        prediction = model.decoders[
            view
        ](
            patient_shared,
            zero_private,
        )

        per_sample_mse = torch.mean(
            (
                prediction
                - views[
                    view
                ]
            )
            ** 2,
            dim=1,
        )

        count = mask.sum()

        valid = (
            count
            > 0.0
        ).to(
            dtype=patient_shared.dtype
        )

        mse = (
            per_sample_mse
            * mask
        ).sum() / count.clamp_min(
            1.0
        )

        view_losses.append(
            (
                mse
                / max(
                    float(
                        null_mse[
                            view
                        ]
                    ),
                    NULL_MSE_FLOOR,
                )
            )
            * valid
        )

        view_validity.append(
            valid
        )

    losses = torch.stack(
        view_losses
    )

    validity = torch.stack(
        view_validity
    )

    return losses.sum() / validity.sum().clamp_min(
        1.0
    )


def systematic_missing_view_loss(
    model: SableMOProfileModel,
    views: dict[
        str,
        torch.Tensor,
    ],
    shared: dict[
        str,
        torch.Tensor,
    ],
    availability: torch.Tensor,
    null_mse: dict[str, float],
) -> torch.Tensor:
    view_losses = []
    view_validity = []

    zero_private = torch.zeros(
        (
            availability.shape[
                0
            ],
            model.profile.private_dim,
        ),
        dtype=torch.float32,
        device=availability.device,
    )

    available_count = availability.sum(
        dim=1
    )

    for omitted_view in VIEWS:
        target_index = VIEW_INDEX[
            omitted_view
        ]

        target_available = availability[
            :,
            target_index,
        ]

        remaining_available = (
            available_count
            - target_available.to(
                dtype=available_count.dtype
            )
            >= 1
        )

        mask_bool = (
            target_available
            & remaining_available
        )

        mask = mask_bool.to(
            dtype=zero_private.dtype
        )

        omitted_active = availability.clone()

        omitted_active[
            :,
            target_index,
        ] = False

        patient_shared = model.patient_shared(
            shared,
            omitted_active,
        )

        prediction = model.decoders[
            omitted_view
        ](
            patient_shared,
            zero_private,
        )

        per_sample_mse = torch.mean(
            (
                prediction
                - views[
                    omitted_view
                ]
            )
            ** 2,
            dim=1,
        )

        count = mask.sum()

        valid = (
            count
            > 0.0
        ).to(
            dtype=zero_private.dtype
        )

        mse = (
            per_sample_mse
            * mask
        ).sum() / count.clamp_min(
            1.0
        )

        view_losses.append(
            (
                mse
                / max(
                    float(
                        null_mse[
                            omitted_view
                        ]
                    ),
                    NULL_MSE_FLOOR,
                )
            )
            * valid
        )

        view_validity.append(
            valid
        )

    losses = torch.stack(
        view_losses
    )

    validity = torch.stack(
        view_validity
    )

    return losses.sum() / validity.sum().clamp_min(
        1.0
    )


def compute_objective(
    model: SableMOProfileModel,
    views: dict[
        str,
        torch.Tensor,
    ],
    availability: torch.Tensor,
    null_mse: dict[str, float],
    *,
    stochastic_dropout: bool,
    fit_seed: int,
    epoch: int,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    shared, private = model.encode(
        views
    )

    if stochastic_dropout:
        (
            active_mask,
            dropped_mask,
        ) = make_modality_dropout(
            availability,
            probability=(
                model.profile.modality_dropout_probability
            ),
            fit_seed=fit_seed,
            epoch=epoch,
        )

    else:
        active_mask = availability
        dropped_mask = torch.zeros_like(
            availability,
            dtype=torch.bool,
        )

    patient_shared = model.patient_shared(
        shared,
        active_mask,
    )

    reconstruction = reconstruction_loss(
        model=model,
        views=views,
        private=private,
        patient_shared=(
            patient_shared
        ),
        active_mask=(
            active_mask
        ),
        null_mse=null_mse,
    )

    alignment = shared_alignment_loss(
        shared=shared,
        patient_shared=(
            patient_shared
        ),
        active_mask=(
            active_mask
        ),
    )

    orthogonality = orthogonality_loss(
        shared=shared,
        private=private,
        active_mask=(
            active_mask
        ),
    )

    separation = private_separation_loss(
        private=private,
        active_mask=(
            active_mask
        ),
    )

    contrastive = contrastive_loss(
        shared=shared,
        active_mask=(
            active_mask
        ),
        temperature=(
            model.profile.contrastive_temperature
        ),
    )

    if stochastic_dropout:
        cross_reconstruction = (
            dropped_view_reconstruction_loss(
                model=model,
                views=views,
                patient_shared=(
                    patient_shared
                ),
                dropped_mask=(
                    dropped_mask
                ),
                null_mse=(
                    null_mse
                ),
            )
        )

    else:
        cross_reconstruction = (
            systematic_missing_view_loss(
                model=model,
                views=views,
                shared=shared,
                availability=(
                    availability
                ),
                null_mse=(
                    null_mse
                ),
            )
        )

    total = (
        model.profile.reconstruction_weight
        * reconstruction
        + model.profile.shared_alignment_weight
        * alignment
        + model.profile.contrastive_weight
        * contrastive
        + model.profile.orthogonality_weight
        * orthogonality
        + model.profile.private_separation_weight
        * separation
        + model.profile.cross_reconstruction_weight
        * cross_reconstruction
    )

    components = {
        "total": total,
        "reconstruction": (
            reconstruction
        ),
        "shared_alignment": (
            alignment
        ),
        "contrastive": (
            contrastive
        ),
        "orthogonality": (
            orthogonality
        ),
        "private_separation": (
            separation
        ),
        "cross_reconstruction": (
            cross_reconstruction
        ),
    }

    # No device-to-host synchronization here. Non-finite total objectives
    # propagate to gradients and are trapped by clip_grad_norm_(
    # error_if_nonfinite=True) during optimization. Evaluation objectives
    # are synchronized once when the scalar is actually needed.

    return (
        total,
        components,
    )


# ---------------------------------------------------------------------
# Deterministic CUDA fitting
# ---------------------------------------------------------------------

def construct_model(
    input_dims: dict[str, int],
    profile: ProfileSpec,
    seed: int,
) -> SableMOProfileModel:
    seed_everything(
        seed
    )

    model = SableMOProfileModel(
        input_dims=input_dims,
        profile=profile,
    ).to(
        device="cuda:0",
        dtype=torch.float32,
    )

    if next(
        model.parameters()
    ).device.type != "cuda":
        raise RuntimeError(
            "SABLE-MO model is not on CUDA."
        )

    return model


def make_optimizer(
    model: nn.Module,
    profile: ProfileSpec,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=(
            profile.learning_rate
        ),
        weight_decay=(
            profile.weight_decay
        ),
        foreach=False,
        fused=False,
    )


def train_with_early_stopping(
    profile: ProfileSpec,
    fit_views_np: dict[str, np.ndarray],
    fit_availability_np: np.ndarray,
    early_views_np: dict[str, np.ndarray],
    early_availability_np: np.ndarray,
    fit_seed: int,
) -> tuple[
    int,
    dict[str, Any],
]:
    input_dims = {
        view: int(
            fit_views_np[
                view
            ].shape[
                1
            ]
        )
        for view in VIEWS
    }

    fit_views = numpy_views_to_cuda(
        fit_views_np
    )

    fit_availability = availability_to_cuda(
        fit_availability_np
    )

    early_views = numpy_views_to_cuda(
        early_views_np
    )

    early_availability = availability_to_cuda(
        early_availability_np
    )

    null_mse = calculate_null_mse(
        fit_views
    )

    model = construct_model(
        input_dims=input_dims,
        profile=profile,
        seed=fit_seed,
    )

    optimizer = make_optimizer(
        model,
        profile,
    )

    best_value = float(
        "inf"
    )

    best_epoch = 1
    epochs_without_improvement = 0
    epochs_run = 0
    final_training_value = None
    final_training_tensor = None
    final_early_value = None

    start = time.perf_counter()

    for epoch in range(
        1,
        profile.maximum_epochs
        + 1,
    ):
        epochs_run = epoch

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        total, _ = compute_objective(
            model=model,
            views=fit_views,
            availability=(
                fit_availability
            ),
            null_mse=(
                null_mse
            ),
            stochastic_dropout=True,
            fit_seed=fit_seed,
            epoch=epoch,
        )

        total.backward()

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=(
                    GRADIENT_CLIP_NORM
                ),
                error_if_nonfinite=True,
            )
        )

        # error_if_nonfinite=True above is the frozen failure gate.
        # Avoid a second GPU->CPU scalar synchronization here.
        optimizer.step()

        final_training_tensor = total.detach()

        model.eval()

        with torch.no_grad():
            early_total, _ = (
                compute_objective(
                    model=model,
                    views=early_views,
                    availability=(
                        early_availability
                    ),
                    null_mse=(
                        null_mse
                    ),
                    stochastic_dropout=False,
                    fit_seed=fit_seed,
                    epoch=epoch,
                )
            )

        early_value = float(
            early_total.detach().cpu()
        )

        final_early_value = (
            early_value
        )

        if (
            early_value
            < best_value
            - profile.min_delta
        ):
            best_value = (
                early_value
            )

            best_epoch = epoch
            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= profile.patience
        ):
            break

    if final_training_tensor is not None:
        final_training_value = float(
            final_training_tensor.cpu()
        )

    # early_value was synchronized once per epoch because early stopping
    # genuinely requires a CPU decision. No additional device-wide
    # synchronization is necessary.
    wall_seconds = (
        time.perf_counter()
        - start
    )

    if not math.isfinite(
        best_value
    ):
        raise RuntimeError(
            "No finite early-stopping objective was obtained."
        )

    audit = {
        "fit_stage": (
            "profile_early_stop"
        ),
        "fit_seed": int(
            fit_seed
        ),
        "profile_id": (
            profile.profile_id
        ),
        "epochs_run": int(
            epochs_run
        ),
        "selected_epoch": int(
            best_epoch
        ),
        "best_early_stop_objective": float(
            best_value
        ),
        "final_early_stop_objective": (
            final_early_value
        ),
        "final_training_objective": (
            final_training_value
        ),
        "early_stopped": bool(
            epochs_run
            < profile.maximum_epochs
        ),
        "maximum_epochs": int(
            profile.maximum_epochs
        ),
        "patience": int(
            profile.patience
        ),
        "min_delta": float(
            profile.min_delta
        ),
        "wall_seconds": float(
            wall_seconds
        ),
        "device": "cuda:0",
        "fit_status": "PASS",
    }

    del optimizer
    del model
    del fit_views
    del early_views
    del fit_availability
    del early_availability

    torch.cuda.empty_cache()

    return (
        int(
            best_epoch
        ),
        audit,
    )


def train_fixed_epochs(
    profile: ProfileSpec,
    training_views_np: dict[
        str,
        np.ndarray,
    ],
    availability_np: np.ndarray,
    fit_seed: int,
    epochs: int,
    fit_stage: str,
) -> tuple[
    SableMOProfileModel,
    dict[str, Any],
]:
    if epochs < 1:
        raise ValueError(
            "Fixed epoch count must be positive."
        )

    input_dims = {
        view: int(
            training_views_np[
                view
            ].shape[
                1
            ]
        )
        for view in VIEWS
    }

    views = numpy_views_to_cuda(
        training_views_np
    )

    availability = availability_to_cuda(
        availability_np
    )

    null_mse = calculate_null_mse(
        views
    )

    model = construct_model(
        input_dims=input_dims,
        profile=profile,
        seed=fit_seed,
    )

    optimizer = make_optimizer(
        model,
        profile,
    )

    final_total = None
    final_total_tensor = None

    start = time.perf_counter()

    for epoch in range(
        1,
        int(
            epochs
        )
        + 1,
    ):
        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        total, _ = compute_objective(
            model=model,
            views=views,
            availability=availability,
            null_mse=null_mse,
            stochastic_dropout=True,
            fit_seed=fit_seed,
            epoch=epoch,
        )

        total.backward()

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=(
                    GRADIENT_CLIP_NORM
                ),
                error_if_nonfinite=True,
            )
        )

        # clip_grad_norm_(error_if_nonfinite=True) is sufficient to
        # enforce the frozen non-finite failure policy without a host sync.
        optimizer.step()

        final_total_tensor = total.detach()

    final_total = float(
        final_total_tensor.cpu()
    )

    # The scalar transfer above synchronizes the final training step.
    # Avoid device-wide torch.cuda.synchronize(), which serialized every
    # tiny fixed-epoch fit in v1.
    wall_seconds = (
        time.perf_counter()
        - start
    )

    model.eval()

    audit = {
        "fit_stage": (
            fit_stage
        ),
        "fit_seed": int(
            fit_seed
        ),
        "profile_id": (
            profile.profile_id
        ),
        "epochs_run": int(
            epochs
        ),
        "selected_epoch": int(
            epochs
        ),
        "final_training_objective": (
            final_total
        ),
        "wall_seconds": float(
            wall_seconds
        ),
        "device": "cuda:0",
        "fit_status": "PASS",
    }

    del optimizer
    del views
    del availability

    return (
        model,
        audit,
    )


def encode_numpy(
    model: SableMOProfileModel,
    views_np: dict[str, np.ndarray],
    availability_np: np.ndarray,
) -> dict[str, Any]:
    views = numpy_views_to_cuda(
        views_np
    )

    availability = availability_to_cuda(
        availability_np
    )

    model.eval()

    with torch.no_grad():
        shared, private = model.encode(
            views
        )

        patient_shared = (
            model.patient_shared(
                shared,
                availability,
            )
        )

    output = {
        "patient_shared": (
            patient_shared.detach()
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        ),
        "shared_by_view": {
            view: (
                shared[
                    view
                ]
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64
                )
            )
            for view in VIEWS
        },
        "private_by_view": {
            view: (
                private[
                    view
                ]
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64
                )
            )
            for view in VIEWS
        },
    }

    del views
    del availability

    return output


# ---------------------------------------------------------------------
# Held-out profile-validation metrics
# ---------------------------------------------------------------------

def null_mse_from_numpy(
    views: dict[str, np.ndarray],
) -> dict[str, float]:
    output = {}

    for view in VIEWS:
        value = float(
            np.mean(
                np.asarray(
                    views[
                        view
                    ],
                    dtype=np.float64,
                )
                ** 2
            )
        )

        output[
            view
        ] = max(
            value,
            NULL_MSE_FLOOR,
        )

    return output


def normalized_reconstruction_against_training_null(
    prediction: np.ndarray,
    truth: np.ndarray,
    training_null_mse: float,
) -> dict[str, float | None]:
    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    truth = np.asarray(
        truth,
        dtype=np.float64,
    )

    if prediction.shape != truth.shape:
        raise RuntimeError(
            "Prediction/truth shape mismatch."
        )

    if (
        not np.isfinite(
            prediction
        ).all()
        or not np.isfinite(
            truth
        ).all()
    ):
        return {
            "mse": None,
            "normalized_error": None,
            "score": None,
            "r2": None,
        }

    mse = float(
        np.mean(
            (
                prediction
                - truth
            )
            ** 2
        )
    )

    denominator = max(
        float(
            training_null_mse
        ),
        NULL_MSE_FLOOR,
    )

    normalized_error = (
        mse
        / denominator
    )

    return {
        "mse": mse,
        "normalized_error": float(
            normalized_error
        ),
        "score": float(
            np.clip(
                1.0
                - normalized_error,
                0.0,
                1.0,
            )
        ),
        "r2": float(
            1.0
            - normalized_error
        ),
    }


def predict_reconstruction_numpy(
    model: SableMOProfileModel,
    views_np: dict[str, np.ndarray],
    availability_np: np.ndarray,
) -> dict[str, np.ndarray]:
    views = numpy_views_to_cuda(
        views_np
    )

    availability = availability_to_cuda(
        availability_np
    )

    model.eval()

    output = {}

    with torch.no_grad():
        shared, private = model.encode(
            views
        )

        patient_shared = (
            model.patient_shared(
                shared,
                availability,
            )
        )

        for view in VIEWS:
            view_index = VIEW_INDEX[
                view
            ]

            available = availability[
                :,
                view_index,
            ]

            private_input = private[
                view
            ].clone()

            if bool(
                (
                    ~available
                ).any()
            ):
                private_input[
                    ~available,
                    :,
                ] = 0.0

            prediction = model.decoders[
                view
            ](
                patient_shared,
                private_input,
            )

            output[
                view
            ] = (
                prediction.detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64
                )
            )

    del views
    del availability

    return output


def predict_missing_view_numpy(
    model: SableMOProfileModel,
    views_np: dict[str, np.ndarray],
    availability_np: np.ndarray,
    omitted_view: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    views = numpy_views_to_cuda(
        views_np
    )

    availability = availability_to_cuda(
        availability_np
    )

    model.eval()

    with torch.no_grad():
        shared, _ = model.encode(
            views
        )

        omitted_active = (
            availability.clone()
        )

        omitted_active[
            :,
            VIEW_INDEX[
                omitted_view
            ],
        ] = False

        valid = (
            omitted_active.sum(
                dim=1
            )
            >= 1
        )

        patient_shared = (
            model.patient_shared(
                shared,
                omitted_active,
            )
        )

        zero_private = torch.zeros(
            (
                patient_shared.shape[
                    0
                ],
                model.profile.private_dim,
            ),
            dtype=torch.float32,
            device=patient_shared.device,
        )

        prediction = model.decoders[
            omitted_view
        ](
            patient_shared,
            zero_private,
        )

        prediction_np = (
            prediction.detach()
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )

        valid_np = (
            valid.detach()
            .cpu()
            .numpy()
            .astype(
                bool
            )
        )

    del views
    del availability

    return (
        prediction_np,
        valid_np,
    )


def validation_metrics(
    linear: Any,
    arrays: dict[str, np.ndarray],
    profile: ProfileSpec,
    model: SableMOProfileModel,
    preprocessors: dict[
        str,
        NeuralViewPreprocessor,
    ],
    training_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> dict[str, Any]:
    training_views = transform_views(
        arrays,
        preprocessors,
        training_indices,
        complete=False,
    )

    validation_views = transform_views(
        arrays,
        preprocessors,
        validation_indices,
        complete=False,
    )

    validation_truth = transform_views(
        arrays,
        preprocessors,
        validation_indices,
        complete=True,
    )

    training_availability = (
        view_availability(
            arrays,
            preprocessors,
            training_indices,
        )
    )

    validation_availability = (
        view_availability(
            arrays,
            preprocessors,
            validation_indices,
        )
    )

    null_mse = null_mse_from_numpy(
        training_views
    )

    train_encoded = encode_numpy(
        model,
        training_views,
        training_availability,
    )

    validation_encoded = encode_numpy(
        model,
        validation_views,
        validation_availability,
    )

    reconstruction_prediction = (
        predict_reconstruction_numpy(
            model,
            validation_views,
            validation_availability,
        )
    )

    reconstruction_metrics = {}

    for view in VIEWS:
        reconstruction_metrics[
            view
        ] = (
            normalized_reconstruction_against_training_null(
                reconstruction_prediction[
                    view
                ],
                validation_truth[
                    view
                ],
                null_mse[
                    view
                ],
            )
        )

    mean_reconstruction_error = (
        linear.nanmean_or_none(
            reconstruction_metrics[
                view
            ][
                "normalized_error"
            ]
            for view in VIEWS
        )
    )

    heldout_reconstruction_score = (
        float(
            np.clip(
                1.0
                - mean_reconstruction_error,
                0.0,
                1.0,
            )
        )
        if mean_reconstruction_error
        is not None
        else None
    )

    row_source = {
        view: arrays[
            f"row_source_{VIEW_PREFIX[view]}"
        ].astype(
            int
        )
        for view in VIEWS
    }

    retrieval = (
        linear.cross_view_retrieval_metrics(
            train_encoded[
                "shared_by_view"
            ],
            validation_encoded[
                "shared_by_view"
            ],
            row_source,
            training_indices,
            validation_indices,
        )
    )

    missing_view_r2 = {}

    for omitted_view in VIEWS:
        (
            prediction,
            valid,
        ) = predict_missing_view_numpy(
            model,
            validation_views,
            validation_availability,
            omitted_view,
        )

        if not valid.any():
            missing_view_r2[
                omitted_view
            ] = None
            continue

        metric = (
            normalized_reconstruction_against_training_null(
                prediction[
                    valid,
                    :,
                ],
                validation_truth[
                    omitted_view
                ][
                    valid,
                    :,
                ],
                null_mse[
                    omitted_view
                ],
            )
        )

        missing_view_r2[
            omitted_view
        ] = metric[
            "r2"
        ]

    missing_view_score = (
        linear.nanmean_or_none(
            np.clip(
                value,
                0.0,
                1.0,
            )
            for value in (
                missing_view_r2.values()
            )
            if value
            is not None
        )
    )

    energy_fractions = (
        model.decoder_shared_energy()
    )

    entropy = normalized_entropy(
        energy_fractions
    )

    modality_balance_score = float(
        np.clip(
            np.median(
                entropy
            ),
            0.0,
            1.0,
        )
    )

    biological_truth, _ = (
        linear.integrated_anchor_truth(
            arrays,
            "biological",
        )
    )

    technical_truth, _ = (
        linear.integrated_anchor_truth(
            arrays,
            "technical",
        )
    )

    validation_shared = (
        validation_encoded[
            "patient_shared"
        ]
    )

    biological_validation = (
        biological_truth[
            validation_indices,
            :,
        ]
    )

    technical_validation = (
        technical_truth[
            validation_indices,
            :,
        ]
    )

    shared_subspace = (
        linear.subspace_metrics(
            validation_shared,
            biological_validation,
        )
    )

    technical_subspace = (
        linear.subspace_metrics(
            validation_shared,
            technical_validation,
        )
    )

    private_values = []

    for view in VIEWS:
        private_truth, _ = (
            linear.row_aligned_truth(
                arrays,
                view,
                "private",
            )
        )

        private_validation = (
            private_truth[
                validation_indices,
                :,
            ]
        )

        result = (
            linear.subspace_metrics(
                validation_encoded[
                    "private_by_view"
                ][
                    view
                ],
                private_validation,
            )
        )

        value = result[
            "mean_canonical_correlation"
        ]

        if value is not None:
            private_values.append(
                float(
                    value
                )
            )

    private_subspace_recovery = (
        float(
            np.mean(
                private_values
            )
        )
        if private_values
        else None
    )

    raw_shared_recovery = (
        shared_subspace[
            "mean_canonical_correlation"
        ]
    )

    # The raw subspace metric remains NA when no biological shared
    # truth exists. For the frozen five-component utility, absence of
    # a true shared subspace contributes zero rather than silently
    # dropping/reweighting the prespecified component.
    shared_utility_component = (
        float(
            raw_shared_recovery
        )
        if raw_shared_recovery
        is not None
        else 0.0
    )

    private_utility_component = (
        float(
            private_subspace_recovery
        )
        if private_subspace_recovery
        is not None
        else 0.0
    )

    retrieval_component = (
        retrieval[
            "random_adjusted_cross_view_mrr"
        ]
    )

    missing_component = (
        missing_view_score
    )

    utility_components = [
        (
            0.40,
            shared_utility_component,
        ),
        (
            0.20,
            private_utility_component,
        ),
        (
            0.20,
            retrieval_component,
        ),
        (
            0.10,
            missing_component,
        ),
        (
            0.10,
            modality_balance_score,
        ),
    ]

    if all(
        value is not None
        and math.isfinite(
            float(
                value
            )
        )
        for _, value in utility_components
    ):
        utility = float(
            sum(
                weight
                * float(
                    np.clip(
                        value,
                        0.0,
                        1.0,
                    )
                )
                for (
                    weight,
                    value,
                ) in utility_components
            )
        )

    else:
        utility = None

    shared_variance = np.var(
        validation_shared,
        axis=0,
        ddof=1,
    )

    return {
        "heldout_normalized_reconstruction_error": (
            mean_reconstruction_error
        ),
        "heldout_reconstruction_score": (
            heldout_reconstruction_score
        ),
        "heldout_reconstruction_error_RNA": (
            reconstruction_metrics[
                "RNA"
            ][
                "normalized_error"
            ]
        ),
        "heldout_reconstruction_error_METHYLATION": (
            reconstruction_metrics[
                "METHYLATION"
            ][
                "normalized_error"
            ]
        ),
        "heldout_reconstruction_error_CNV": (
            reconstruction_metrics[
                "CNV"
            ][
                "normalized_error"
            ]
        ),
        "cross_view_mrr": (
            retrieval[
                "cross_view_mrr"
            ]
        ),
        "random_expected_cross_view_mrr": (
            retrieval[
                "random_expected_cross_view_mrr"
            ]
        ),
        "random_adjusted_cross_view_mrr": (
            retrieval[
                "random_adjusted_cross_view_mrr"
            ]
        ),
        "cross_view_top1_accuracy": (
            retrieval[
                "cross_view_top1_accuracy"
            ]
        ),
        "same_patient_similarity_margin": (
            retrieval[
                "same_patient_similarity_margin"
            ]
        ),
        "missing_view_r2_RNA": (
            missing_view_r2[
                "RNA"
            ]
        ),
        "missing_view_r2_METHYLATION": (
            missing_view_r2[
                "METHYLATION"
            ]
        ),
        "missing_view_r2_CNV": (
            missing_view_r2[
                "CNV"
            ]
        ),
        "missing_view_reconstruction_score": (
            missing_view_score
        ),
        "modality_contribution_entropy": float(
            np.median(
                entropy
            )
        ),
        "modality_balance_score": (
            modality_balance_score
        ),
        "shared_subspace_recovery": (
            raw_shared_recovery
        ),
        "private_subspace_recovery": (
            private_subspace_recovery
        ),
        "technical_subspace_recovery": (
            technical_subspace[
                "mean_canonical_correlation"
            ]
        ),
        "validation_latent_effective_rank": (
            linear.effective_rank(
                validation_shared
            )
        ),
        "validation_median_shared_variance": float(
            np.median(
                shared_variance
            )
        ),
        "simulation_utility": (
            utility
        ),
        "utility_shared_component": (
            shared_utility_component
        ),
        "utility_private_component": (
            private_utility_component
        ),
        "utility_retrieval_component": (
            retrieval_component
        ),
        "utility_missing_view_component": (
            missing_component
        ),
        "utility_modality_balance_component": (
            modality_balance_score
        ),
    }


# ---------------------------------------------------------------------
# Stability alignment
# ---------------------------------------------------------------------

def standardized_matrix(
    matrix: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        matrix,
        dtype=np.float64,
    )

    centered = (
        values
        - np.mean(
            values,
            axis=0,
        )
    )

    scales = np.std(
        centered,
        axis=0,
        ddof=1,
    )

    valid = (
        np.isfinite(
            scales
        )
        & (
            scales
            > 1e-12
        )
    )

    output = np.zeros_like(
        centered,
        dtype=np.float64,
    )

    if valid.any():
        output[
            :,
            valid,
        ] = (
            centered[
                :,
                valid,
            ]
            / scales[
                valid
            ]
        )

    return output


def align_resample_to_reference(
    linear: Any,
    resample_scores: np.ndarray,
    reference_scores: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float | None,
]:
    resample_standardized = (
        standardized_matrix(
            resample_scores
        )
    )

    reference_standardized = (
        standardized_matrix(
            reference_scores
        )
    )

    mapping = linear.orthogonal_mapping(
        resample_standardized,
        reference_standardized,
    )

    aligned = (
        resample_standardized
        @ mapping
    )

    correlation = (
        linear.absolute_correlation_matrix(
            aligned,
            reference_standardized,
        )
    )

    if correlation.shape[
        0
    ] != correlation.shape[
        1
    ]:
        raise RuntimeError(
            "Stability alignment expects equal shared latent dimensions."
        )

    row_indices, column_indices = (
        linear_sum_assignment(
            -correlation
        )
    )

    by_reference = np.full(
        correlation.shape[
            1
        ],
        np.nan,
        dtype=np.float64,
    )

    for row_index, column_index in zip(
        row_indices,
        column_indices,
    ):
        by_reference[
            int(
                column_index
            )
        ] = float(
            correlation[
                row_index,
                column_index,
            ]
        )

    subspace = linear.subspace_metrics(
        aligned,
        reference_standardized,
    )

    return (
        aligned,
        by_reference,
        subspace[
            "mean_canonical_correlation"
        ],
    )


def reference_factor_truth_status(
    linear: Any,
    arrays: dict[str, np.ndarray],
    reference_scores: np.ndarray,
) -> tuple[
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
]:
    biological_truth, biological_ids = (
        linear.integrated_anchor_truth(
            arrays,
            "biological",
        )
    )

    technical_truth, technical_ids = (
        linear.integrated_anchor_truth(
            arrays,
            "technical",
        )
    )

    (
        biological_summary,
        biological_rows,
    ) = linear.factor_matching(
        reference_scores,
        biological_truth,
        biological_ids,
        TRUTH_MATCH_THRESHOLD,
    )

    del biological_summary

    technical_correlation = (
        linear.absolute_correlation_matrix(
            reference_scores,
            technical_truth,
        )
    )

    technical_best = np.zeros(
        reference_scores.shape[
            1
        ],
        dtype=np.float64,
    )

    if technical_correlation.shape[
        1
    ] > 0:
        technical_best = np.max(
            technical_correlation,
            axis=1,
        )

    biological_best = np.zeros(
        reference_scores.shape[
            1
        ],
        dtype=np.float64,
    )

    biological_correlation = (
        linear.absolute_correlation_matrix(
            reference_scores,
            biological_truth,
        )
    )

    if biological_correlation.shape[
        1
    ] > 0:
        biological_best = np.max(
            biological_correlation,
            axis=1,
        )

    biological_by_estimated = {
        int(
            row[
                "estimated_factor_index"
            ]
        ): row
        for row in biological_rows
    }

    rows = []

    for factor_index in range(
        reference_scores.shape[
            1
        ]
    ):
        match = biological_by_estimated.get(
            factor_index,
            {},
        )

        rows.append(
            {
                "factor_index": (
                    factor_index
                ),
                "biological_truth_factor_index": (
                    match.get(
                        "truth_factor_index"
                    )
                ),
                "biological_truth_factor_id": clean_text(
                    match.get(
                        "truth_factor_id"
                    )
                ),
                "biological_assigned_absolute_correlation": (
                    match.get(
                        "absolute_correlation"
                    )
                ),
                "biological_truth_match": bool(
                    match.get(
                        "passes_truth_match_threshold",
                        False,
                    )
                ),
                "best_biological_absolute_correlation": float(
                    biological_best[
                        factor_index
                    ]
                ),
                "best_technical_absolute_correlation": float(
                    technical_best[
                        factor_index
                    ]
                ),
                "technical_truth_match": bool(
                    technical_best[
                        factor_index
                    ]
                    >= TECHNICAL_MATCH_THRESHOLD
                ),
                "n_biological_truth_factors": int(
                    biological_truth.shape[
                        1
                    ]
                ),
                "n_technical_truth_factors": int(
                    technical_truth.shape[
                        1
                    ]
                ),
                "technical_truth_factor_ids": (
                    ";".join(
                        technical_ids
                    )
                ),
            }
        )

    return (
        rows,
        biological_truth,
        technical_truth,
    )


def stability_audit(
    linear: Any,
    arrays: dict[str, np.ndarray],
    profile: ProfileSpec,
    selected_epoch: int,
    stability_group: dict[str, Any],
    stability_thresholds: dict[str, Any],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    all_indices = np.arange(
        arrays[
            "sample_index"
        ].shape[
            0
        ],
        dtype=int,
    )

    reference_seed = int(
        stability_group[
            "reference_fit_seed"
        ]
    )

    reference_preprocessors = (
        fit_preprocessors(
            linear,
            arrays,
            all_indices,
            profile,
        )
    )

    reference_views = transform_views(
        arrays,
        reference_preprocessors,
        all_indices,
        complete=False,
    )

    reference_availability = (
        view_availability(
            arrays,
            reference_preprocessors,
            all_indices,
        )
    )

    reference_model, reference_fit_audit = (
        train_fixed_epochs(
            profile=profile,
            training_views_np=(
                reference_views
            ),
            availability_np=(
                reference_availability
            ),
            fit_seed=reference_seed,
            epochs=selected_epoch,
            fit_stage=(
                "profile_stability_reference_full_data"
            ),
        )
    )

    reference_preprocessing_rows = (
        preprocessing_audit_rows(
            arrays,
            reference_preprocessors,
            all_indices,
            profile,
            (
                "profile_stability_reference_full_data"
            ),
        )
    )

    reference_encoded = encode_numpy(
        reference_model,
        reference_views,
        reference_availability,
    )

    reference_scores = (
        reference_encoded[
            "patient_shared"
        ]
    )

    reference_effective_rank = (
        linear.effective_rank(
            reference_scores
        )
    )

    reference_variances = np.var(
        reference_scores,
        axis=0,
        ddof=1,
    )

    median_reference_variance = float(
        np.median(
            reference_variances
        )
    )

    latent_collapse = bool(
        reference_effective_rank
        < 2.0
        or median_reference_variance
        < 1e-4
    )

    energy_fractions = (
        reference_model.decoder_shared_energy()
    )

    (
        truth_rows,
        biological_truth,
        technical_truth,
    ) = reference_factor_truth_status(
        linear,
        arrays,
        reference_scores,
    )

    biological_subspace = (
        linear.subspace_metrics(
            reference_scores,
            biological_truth,
        )
    )

    technical_subspace = (
        linear.subspace_metrics(
            reference_scores,
            technical_truth,
        )
    )

    per_factor_correlations = [
        []
        for _ in range(
            profile.shared_dim
        )
    ]

    subspace_correlations = []

    fit_audits = [
        reference_fit_audit
    ]

    preprocessing_rows = list(
        reference_preprocessing_rows
    )

    stability_rows = (
        stability_group[
            "rows"
        ]
    )

    n_samples = int(
        all_indices.shape[
            0
        ]
    )

    for _, registry_row in (
        stability_rows.iterrows()
    ):
        resample_index = int(
            registry_row[
                "resample_index"
            ]
        )

        n_subsample = int(
            float(
                registry_row[
                    "n_subsample"
                ]
            )
        )

        subsample_seed = int(
            float(
                registry_row[
                    "subsample_seed"
                ]
            )
        )

        resample_fit_seed = int(
            float(
                registry_row[
                    "resample_fit_seed"
                ]
            )
        )

        rng = np.random.default_rng(
            subsample_seed
        )

        subset = np.sort(
            rng.choice(
                n_samples,
                size=n_subsample,
                replace=False,
            ).astype(
                int
            )
        )

        resample_preprocessors = (
            fit_preprocessors(
                linear,
                arrays,
                subset,
                profile,
            )
        )

        resample_views = transform_views(
            arrays,
            resample_preprocessors,
            subset,
            complete=False,
        )

        resample_availability = (
            view_availability(
                arrays,
                resample_preprocessors,
                subset,
            )
        )

        resample_model, fit_audit = (
            train_fixed_epochs(
                profile=profile,
                training_views_np=(
                    resample_views
                ),
                availability_np=(
                    resample_availability
                ),
                fit_seed=(
                    resample_fit_seed
                ),
                epochs=(
                    selected_epoch
                ),
                fit_stage=(
                    "profile_stability_resample"
                ),
            )
        )

        fit_audit[
            "resample_index"
        ] = resample_index

        fit_audit[
            "subsample_seed"
        ] = subsample_seed

        fit_audits.append(
            fit_audit
        )

        resample_preprocessing = (
            preprocessing_audit_rows(
                arrays,
                resample_preprocessors,
                subset,
                profile,
                (
                    "profile_stability_resample"
                ),
            )
        )

        for row in resample_preprocessing:
            row[
                "resample_index"
            ] = resample_index

            row[
                "subsample_seed"
            ] = subsample_seed

        preprocessing_rows.extend(
            resample_preprocessing
        )

        resample_encoded = encode_numpy(
            resample_model,
            resample_views,
            resample_availability,
        )

        resample_scores = (
            resample_encoded[
                "patient_shared"
            ]
        )

        reference_subset_scores = (
            reference_scores[
                subset,
                :,
            ]
        )

        (
            aligned,
            correlations,
            subspace_correlation,
        ) = align_resample_to_reference(
            linear,
            resample_scores,
            reference_subset_scores,
        )

        del aligned

        for factor_index, value in enumerate(
            correlations
        ):
            if np.isfinite(
                value
            ):
                per_factor_correlations[
                    factor_index
                ].append(
                    float(
                        value
                    )
                )

        if (
            subspace_correlation
            is not None
            and math.isfinite(
                float(
                    subspace_correlation
                )
            )
        ):
            subspace_correlations.append(
                float(
                    subspace_correlation
                )
            )

        del resample_model
        del resample_views
        del resample_preprocessors
        del resample_encoded

        # Keep the CUDA caching allocator warm across the 20 tiny resample
        # fits; forcing empty_cache() here would add substantial synchronization
        # overhead. Python garbage collection is amortized every five resamples.
        if resample_index % 5 == 0:
            gc.collect()

    thresholds = {
        "median": float(
            stability_thresholds[
                "primary_median_factor_correlation"
            ]
        ),
        "q10": float(
            stability_thresholds[
                "primary_q10_factor_correlation"
            ]
        ),
        "inclusion": float(
            stability_thresholds[
                "primary_inclusion_frequency"
            ]
        ),
        "variance": float(
            stability_thresholds[
                "minimum_factor_variance"
            ]
        ),
        "minimum_supported_modalities": int(
            stability_thresholds[
                "minimum_supported_modalities"
            ]
        ),
        "minimum_loading_energy": float(
            stability_thresholds[
                "minimum_loading_energy_fraction_per_supported_modality"
            ]
        ),
    }

    truth_by_factor = {
        int(
            row[
                "factor_index"
            ]
        ): row
        for row in truth_rows
    }

    factor_rows = []

    stable_nontechnical = 0
    false_biological = 0
    n_stable = 0
    n_stable_technical = 0
    n_stable_biological = 0

    for factor_index in range(
        profile.shared_dim
    ):
        values = np.asarray(
            per_factor_correlations[
                factor_index
            ],
            dtype=np.float64,
        )

        if values.size != STABILITY_RESAMPLES:
            raise RuntimeError(
                "A stability factor does not contain all 20 aligned correlations."
            )

        median_correlation = float(
            np.median(
                values
            )
        )

        q10_correlation = float(
            np.quantile(
                values,
                0.10,
            )
        )

        inclusion_frequency = float(
            np.mean(
                values
                >= TRUTH_MATCH_THRESHOLD
            )
        )

        variance = float(
            reference_variances[
                factor_index
            ]
        )

        factor_energy = energy_fractions[
            factor_index,
            :,
        ]

        supported = (
            factor_energy
            >= thresholds[
                "minimum_loading_energy"
            ]
        )

        n_supported = int(
            np.sum(
                supported
            )
        )

        stable = bool(
            median_correlation
            >= thresholds[
                "median"
            ]
            and q10_correlation
            >= thresholds[
                "q10"
            ]
            and inclusion_frequency
            >= thresholds[
                "inclusion"
            ]
            and variance
            >= thresholds[
                "variance"
            ]
            and n_supported
            >= thresholds[
                "minimum_supported_modalities"
            ]
        )

        truth = truth_by_factor[
            factor_index
        ]

        biological_match = bool(
            truth[
                "biological_truth_match"
            ]
        )

        technical_match = bool(
            truth[
                "technical_truth_match"
            ]
        )

        if not stable:
            status = (
                "unstable"
            )

        elif technical_match:
            status = (
                "stable_technical_flag"
            )

            n_stable += 1
            n_stable_technical += 1

        elif biological_match:
            status = (
                "stable_biological_match"
            )

            n_stable += 1
            n_stable_biological += 1
            stable_nontechnical += 1

        else:
            status = (
                "stable_unmatched_nontechnical"
            )

            n_stable += 1
            stable_nontechnical += 1
            false_biological += 1

        factor_rows.append(
            {
                **truth,
                "median_aligned_correlation": (
                    median_correlation
                ),
                "q10_aligned_correlation": (
                    q10_correlation
                ),
                "inclusion_frequency": (
                    inclusion_frequency
                ),
                "reference_factor_variance": (
                    variance
                ),
                "loading_energy_RNA": float(
                    factor_energy[
                        VIEW_INDEX[
                            "RNA"
                        ]
                    ]
                ),
                "loading_energy_METHYLATION": float(
                    factor_energy[
                        VIEW_INDEX[
                            "METHYLATION"
                        ]
                    ]
                ),
                "loading_energy_CNV": float(
                    factor_energy[
                        VIEW_INDEX[
                            "CNV"
                        ]
                    ]
                ),
                "supported_modalities": (
                    ";".join(
                        view
                        for view in VIEWS
                        if supported[
                            VIEW_INDEX[
                                view
                            ]
                        ]
                    )
                ),
                "n_supported_modalities": (
                    n_supported
                ),
                "passes_stability_gate": (
                    stable
                ),
                "factor_status": (
                    status
                ),
                "false_biological_acceptance": bool(
                    stable
                    and not technical_match
                    and not biological_match
                ),
            }
        )

    false_discovery_fraction = (
        float(
            false_biological
            / stable_nontechnical
        )
        if stable_nontechnical
        > 0
        else 0.0
    )

    summary = {
        "reference_effective_rank": (
            reference_effective_rank
        ),
        "reference_median_shared_variance": (
            median_reference_variance
        ),
        "latent_collapse": (
            latent_collapse
        ),
        "reference_shared_subspace_recovery": (
            biological_subspace[
                "mean_canonical_correlation"
            ]
        ),
        "reference_technical_subspace_recovery": (
            technical_subspace[
                "mean_canonical_correlation"
            ]
        ),
        "stability_subspace_median_canonical_correlation": (
            float(
                np.median(
                    subspace_correlations
                )
            )
            if subspace_correlations
            else None
        ),
        "stability_subspace_q10_canonical_correlation": (
            float(
                np.quantile(
                    subspace_correlations,
                    0.10,
                )
            )
            if subspace_correlations
            else None
        ),
        "n_stable_factors": (
            n_stable
        ),
        "n_stable_biological_factors": (
            n_stable_biological
        ),
        "n_stable_technical_factors": (
            n_stable_technical
        ),
        "n_stable_unmatched_nontechnical_factors": (
            false_biological
        ),
        "stability_gated_false_biological_discovery_fraction": (
            false_discovery_fraction
        ),
        "n_stability_resamples": (
            STABILITY_RESAMPLES
        ),
        "selected_epoch_used_for_stability": int(
            selected_epoch
        ),
    }

    del reference_model
    del reference_views
    del reference_preprocessors
    del reference_encoded

    torch.cuda.empty_cache()

    gc.collect()

    return (
        summary,
        factor_rows,
        fit_audits,
        preprocessing_rows,
    )


# ---------------------------------------------------------------------
# Dataset x profile evaluation
# ---------------------------------------------------------------------

def manifest_metadata(
    row: pd.Series,
    dataset_path: Path,
    observed_hash: str,
) -> dict[str, Any]:
    return {
        "registry_label": clean_text(
            row[
                "registry_label"
            ]
        ),
        "scenario_id": clean_text(
            row[
                "scenario_id"
            ]
        ),
        "scenario_origin": clean_text(
            row[
                "scenario_origin"
            ]
        ),
        "scenario_family": clean_text(
            row[
                "scenario_family"
            ]
        ),
        "calibration_role": clean_text(
            row[
                "calibration_role"
            ]
        ),
        "replicate_index": int(
            float(
                row[
                    "replicate_index"
                ]
            )
        ),
        "simulation_seed": int(
            float(
                row[
                    "simulation_seed"
                ]
            )
        ),
        "n_samples": int(
            float(
                row[
                    "n_samples"
                ]
            )
        ),
        "n_true_biological_shared_factors": int(
            float(
                row[
                    "n_biological_shared_factors"
                ]
            )
        ),
        "n_true_private_factors_per_view": int(
            float(
                row[
                    "n_private_factors_per_view"
                ]
            )
        ),
        "n_true_technical_shared_factors": int(
            float(
                row[
                    "n_technical_shared_factors"
                ]
            )
        ),
        "factor_level_identifiable": parse_bool(
            row[
                "factor_level_identifiable"
            ]
        ),
        "subspace_level_identifiable": parse_bool(
            row[
                "subspace_level_identifiable"
            ]
        ),
        "primary_truth_target": clean_text(
            row[
                "primary_truth_target"
            ]
        ),
        "input_path": relative_path(
            dataset_path
        ),
        "input_sha256": (
            observed_hash
        ),
    }


def attach_metadata(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    profile: ProfileSpec,
) -> None:
    for row in rows:
        row.update(
            metadata
        )

        row[
            "profile_id"
        ] = (
            profile.profile_id
        )

        row[
            "architecture_profile"
        ] = (
            profile.architecture_profile
        )

        row[
            "regularization_profile"
        ] = (
            profile.regularization_profile
        )

        row[
            "shared_latent_dim"
        ] = (
            profile.shared_dim
        )

        row[
            "private_latent_dim_per_view"
        ] = (
            profile.private_dim
        )

        row[
            "combined_complexity_rank"
        ] = (
            profile.combined_complexity_rank
        )


def evaluate_dataset_profile(
    linear: Any,
    manifest_row: pd.Series,
    profile: ProfileSpec,
    partitions: dict[str, np.ndarray],
    profile_fit_seed: int,
    stability_group: dict[str, Any],
    stability_thresholds: dict[str, Any],
) -> dict[str, Any]:
    dataset_path = resolve_project_path(
        manifest_row[
            "path"
        ]
    )

    require_file(
        dataset_path
    )

    expected_size = int(
        float(
            manifest_row[
                "size_bytes"
            ]
        )
    )

    if int(
        dataset_path.stat().st_size
    ) != expected_size:
        raise RuntimeError(
            "Simulation file size differs from frozen manifest: "
            f"{relative_path(dataset_path)}"
        )

    observed_hash = sha256_file(
        dataset_path
    )

    expected_hash = clean_text(
        manifest_row[
            "sha256"
        ]
    )

    if observed_hash != expected_hash:
        raise RuntimeError(
            "Simulation SHA256 differs from frozen manifest: "
            f"{relative_path(dataset_path)}"
        )

    arrays = linear.load_simulation_arrays(
        dataset_path
    )

    metadata = manifest_metadata(
        manifest_row,
        dataset_path,
        observed_hash,
    )

    fit_indices = partitions[
        "fit"
    ]

    early_indices = partitions[
        "early_stop"
    ]

    validation_indices = partitions[
        "validation"
    ]

    if (
        set(
            fit_indices.tolist()
        )
        & set(
            early_indices.tolist()
        )
        or set(
            fit_indices.tolist()
        )
        & set(
            validation_indices.tolist()
        )
        or set(
            early_indices.tolist()
        )
        & set(
            validation_indices.tolist()
        )
    ):
        raise RuntimeError(
            "Development fit/early-stop/validation partitions overlap."
        )

    all_partitioned = set(
        fit_indices.tolist()
    ) | set(
        early_indices.tolist()
    ) | set(
        validation_indices.tolist()
    )

    if all_partitioned != set(
        range(
            metadata[
                "n_samples"
            ]
        )
    ):
        raise RuntimeError(
            "Development partition does not cover every sample exactly once."
        )

    torch.cuda.reset_peak_memory_stats(
        0
    )

    fit_audits: list[
        dict[str, Any]
    ] = []

    preprocessing_rows: list[
        dict[str, Any]
    ] = []

    # -------------------------------------------------------------
    # 1. Fit-only preprocessing and deterministic early stopping.
    # -------------------------------------------------------------

    early_preprocessors = fit_preprocessors(
        linear,
        arrays,
        fit_indices,
        profile,
    )

    preprocessing_rows.extend(
        preprocessing_audit_rows(
            arrays,
            early_preprocessors,
            fit_indices,
            profile,
            "profile_early_stop_fit_only",
        )
    )

    fit_views = transform_views(
        arrays,
        early_preprocessors,
        fit_indices,
        complete=False,
    )

    fit_availability = view_availability(
        arrays,
        early_preprocessors,
        fit_indices,
    )

    early_views = transform_views(
        arrays,
        early_preprocessors,
        early_indices,
        complete=False,
    )

    early_availability = view_availability(
        arrays,
        early_preprocessors,
        early_indices,
    )

    (
        selected_epoch,
        early_fit_audit,
    ) = train_with_early_stopping(
        profile=profile,
        fit_views_np=fit_views,
        fit_availability_np=(
            fit_availability
        ),
        early_views_np=(
            early_views
        ),
        early_availability_np=(
            early_availability
        ),
        fit_seed=(
            profile_fit_seed
        ),
    )

    fit_audits.append(
        early_fit_audit
    )

    # -------------------------------------------------------------
    # 2. Refit on fit + early-stop, keeping validation untouched.
    # -------------------------------------------------------------

    refit_indices = np.sort(
        np.concatenate(
            [
                fit_indices,
                early_indices,
            ]
        ).astype(
            int
        )
    )

    refit_preprocessors = fit_preprocessors(
        linear,
        arrays,
        refit_indices,
        profile,
    )

    preprocessing_rows.extend(
        preprocessing_audit_rows(
            arrays,
            refit_preprocessors,
            refit_indices,
            profile,
            "profile_validation_refit_fit_plus_early",
        )
    )

    refit_views = transform_views(
        arrays,
        refit_preprocessors,
        refit_indices,
        complete=False,
    )

    refit_availability = view_availability(
        arrays,
        refit_preprocessors,
        refit_indices,
    )

    refit_model, refit_audit = (
        train_fixed_epochs(
            profile=profile,
            training_views_np=(
                refit_views
            ),
            availability_np=(
                refit_availability
            ),
            fit_seed=(
                profile_fit_seed
            ),
            epochs=(
                selected_epoch
            ),
            fit_stage=(
                "profile_validation_refit_fit_plus_early"
            ),
        )
    )

    fit_audits.append(
        refit_audit
    )

    validation = validation_metrics(
        linear=linear,
        arrays=arrays,
        profile=profile,
        model=refit_model,
        preprocessors=(
            refit_preprocessors
        ),
        training_indices=(
            refit_indices
        ),
        validation_indices=(
            validation_indices
        ),
    )

    del refit_model
    del refit_views
    del refit_preprocessors

    torch.cuda.empty_cache()

    gc.collect()

    # -------------------------------------------------------------
    # 3. Independent full-data + 20-resample stability audit.
    # -------------------------------------------------------------

    (
        stability_summary,
        factor_rows,
        stability_fit_audits,
        stability_preprocessing_rows,
    ) = stability_audit(
        linear=linear,
        arrays=arrays,
        profile=profile,
        selected_epoch=(
            selected_epoch
        ),
        stability_group=(
            stability_group
        ),
        stability_thresholds=(
            stability_thresholds
        ),
    )

    fit_audits.extend(
        stability_fit_audits
    )

    preprocessing_rows.extend(
        stability_preprocessing_rows
    )

    factor_level_primary_eligible = bool(
        metadata[
            "factor_level_identifiable"
        ]
        or metadata[
            "n_true_biological_shared_factors"
        ]
        == 0
    )

    max_memory = int(
        torch.cuda.max_memory_allocated(
            0
        )
    )

    if len(fit_audits) != (2 + 1 + STABILITY_RESAMPLES):
        raise RuntimeError(
            "Successful dataset-profile shard does not contain exactly 23 neural fit audits."
        )

    if len(preprocessing_rows) != (3 + 3 + 3 + 3 * STABILITY_RESAMPLES):
        raise RuntimeError(
            "Successful dataset-profile shard does not contain exactly 69 preprocessing audit rows."
        )

    if len(factor_rows) != profile.shared_dim:
        raise RuntimeError(
            "Successful dataset-profile shard factor-audit row count differs from shared latent dimension."
        )

    dataset_row = {
        **metadata,
        "profile_id": (
            profile.profile_id
        ),
        "architecture_profile": (
            profile.architecture_profile
        ),
        "regularization_profile": (
            profile.regularization_profile
        ),
        "combined_complexity_rank": (
            profile.combined_complexity_rank
        ),
        "architecture_complexity_rank": (
            profile.architecture_complexity_rank
        ),
        "regularization_complexity_rank": (
            profile.regularization_complexity_rank
        ),
        "shared_latent_dim": (
            profile.shared_dim
        ),
        "private_latent_dim_per_view": (
            profile.private_dim
        ),
        "hidden_dim": (
            profile.hidden_dim
        ),
        "selected_epoch": (
            selected_epoch
        ),
        "profile_fit_seed": (
            profile_fit_seed
        ),
        "n_fit_samples": int(
            fit_indices.shape[
                0
            ]
        ),
        "n_early_stop_samples": int(
            early_indices.shape[
                0
            ]
        ),
        "n_validation_samples": int(
            validation_indices.shape[
                0
            ]
        ),
        "factor_level_metrics_primary_eligible": (
            factor_level_primary_eligible
        ),
        "required_fit_count": (
            2
            + 1
            + STABILITY_RESAMPLES
        ),
        "all_required_fits_passed": True,
        "max_cuda_memory_allocated_bytes": (
            max_memory
        ),
        **validation,
        **stability_summary,
        "shard_status": "PASS",
        "shard_error_type": "",
        "shard_error_message": "",
    }

    attach_metadata(
        fit_audits,
        metadata,
        profile,
    )

    attach_metadata(
        preprocessing_rows,
        metadata,
        profile,
    )

    attach_metadata(
        factor_rows,
        metadata,
        profile,
    )

    for row in factor_rows:
        row[
            "factor_level_metrics_primary_eligible"
        ] = (
            factor_level_primary_eligible
        )

    return {
        "script_version": (
            SCRIPT_VERSION
        ),
        "scenario_id": (
            metadata[
                "scenario_id"
            ]
        ),
        "replicate_index": (
            metadata[
                "replicate_index"
            ]
        ),
        "simulation_seed": (
            metadata[
                "simulation_seed"
            ]
        ),
        "profile_id": (
            profile.profile_id
        ),
        "input_path": (
            metadata[
                "input_path"
            ]
        ),
        "input_sha256": (
            metadata[
                "input_sha256"
            ]
        ),
        "dataset_row": (
            dataset_row
        ),
        "fit_audits": (
            fit_audits
        ),
        "preprocessing_rows": (
            preprocessing_rows
        ),
        "factor_rows": (
            factor_rows
        ),
        "stability_summary": {
            **metadata,
            "profile_id": (
                profile.profile_id
            ),
            "architecture_profile": (
                profile.architecture_profile
            ),
            "regularization_profile": (
                profile.regularization_profile
            ),
            "selected_epoch": (
                selected_epoch
            ),
            "factor_level_metrics_primary_eligible": (
                factor_level_primary_eligible
            ),
            **stability_summary,
        },
    }


def failed_shard_record(
    manifest_row: pd.Series,
    profile: ProfileSpec,
    error: BaseException,
) -> dict[str, Any]:
    dataset_path = resolve_project_path(
        manifest_row[
            "path"
        ]
    )

    observed_hash = (
        sha256_file(
            dataset_path
        )
        if dataset_path.exists()
        else ""
    )

    metadata = manifest_metadata(
        manifest_row,
        dataset_path,
        observed_hash,
    )

    dataset_row = {
        **metadata,
        "profile_id": (
            profile.profile_id
        ),
        "architecture_profile": (
            profile.architecture_profile
        ),
        "regularization_profile": (
            profile.regularization_profile
        ),
        "combined_complexity_rank": (
            profile.combined_complexity_rank
        ),
        "architecture_complexity_rank": (
            profile.architecture_complexity_rank
        ),
        "regularization_complexity_rank": (
            profile.regularization_complexity_rank
        ),
        "shared_latent_dim": (
            profile.shared_dim
        ),
        "private_latent_dim_per_view": (
            profile.private_dim
        ),
        "hidden_dim": (
            profile.hidden_dim
        ),
        "selected_epoch": None,
        "profile_fit_seed": None,
        "factor_level_metrics_primary_eligible": bool(
            metadata[
                "factor_level_identifiable"
            ]
            or metadata[
                "n_true_biological_shared_factors"
            ]
            == 0
        ),
        "all_required_fits_passed": False,
        "heldout_normalized_reconstruction_error": None,
        "heldout_reconstruction_score": None,
        "random_adjusted_cross_view_mrr": None,
        "missing_view_reconstruction_score": None,
        "modality_contribution_entropy": None,
        "modality_balance_score": None,
        "shared_subspace_recovery": None,
        "private_subspace_recovery": None,
        "technical_subspace_recovery": None,
        "simulation_utility": None,
        "reference_effective_rank": None,
        "reference_median_shared_variance": None,
        "latent_collapse": None,
        "stability_subspace_median_canonical_correlation": None,
        "stability_subspace_q10_canonical_correlation": None,
        "stability_gated_false_biological_discovery_fraction": None,
        "n_stable_factors": None,
        "shard_status": "FAIL",
        "shard_error_type": (
            type(
                error
            ).__name__
        ),
        "shard_error_message": clean_text(
            str(
                error
            )
        )[
            :2000
        ],
        "max_cuda_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated(
                0
            )
        )
        if torch.cuda.is_available()
        else None,
    }

    fit_audit = {
        **metadata,
        "profile_id": (
            profile.profile_id
        ),
        "architecture_profile": (
            profile.architecture_profile
        ),
        "regularization_profile": (
            profile.regularization_profile
        ),
        "fit_stage": (
            "dataset_profile_failure"
        ),
        "fit_status": "FAIL",
        "fit_error_type": (
            type(
                error
            ).__name__
        ),
        "fit_error_message": clean_text(
            str(
                error
            )
        )[
            :2000
        ],
        "fit_traceback_tail": "\n".join(
            traceback.format_exc().splitlines()[
                -20:
            ]
        ),
    }

    return {
        "script_version": (
            SCRIPT_VERSION
        ),
        "scenario_id": (
            metadata[
                "scenario_id"
            ]
        ),
        "replicate_index": (
            metadata[
                "replicate_index"
            ]
        ),
        "simulation_seed": (
            metadata[
                "simulation_seed"
            ]
        ),
        "profile_id": (
            profile.profile_id
        ),
        "input_path": (
            metadata[
                "input_path"
            ]
        ),
        "input_sha256": (
            metadata[
                "input_sha256"
            ]
        ),
        "dataset_row": (
            dataset_row
        ),
        "fit_audits": [
            fit_audit
        ],
        "preprocessing_rows": [],
        "factor_rows": [],
        "stability_summary": {
            **metadata,
            "profile_id": (
                profile.profile_id
            ),
            "stability_status": "FAIL",
            "error_type": (
                type(
                    error
                ).__name__
            ),
            "error_message": clean_text(
                str(
                    error
                )
            )[
                :2000
            ],
        },
    }


# ---------------------------------------------------------------------
# Prospective execution protocol
# ---------------------------------------------------------------------

def build_execution_protocol(
    upstream: dict[str, Any],
) -> dict[str, Any]:
    stability_thresholds = load_json(
        CONTRACT_05A
    )[
        "stability_gate"
    ][
        "thresholds"
    ]

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
        "scientific_state_before_fit": {
            "outcomes_loaded": False,
            "clinical_labels_loaded": False,
            "biological_annotations_loaded": False,
            "TARGET_model_fitting_performed": False,
            "neural_profile_screening_performed": False,
            "neural_profile_selected": False,
            "representation_architecture_selected": False,
            "final_gate_calibration_data_used": False,
        },
        "scope": {
            "model": (
                "V4_SABLE_MO"
            ),
            "candidate_profiles": (
                EXPECTED_PROFILES
            ),
            "development_datasets": (
                EXPECTED_DEVELOPMENT_DATASETS
            ),
            "dataset_profile_shards": (
                EXPECTED_SHARDS
            ),
            "stability_resamples_per_shard": (
                STABILITY_RESAMPLES
            ),
            "expected_optimization_runs": (
                EXPECTED_SHARDS
                * (
                    2
                    + 1
                    + STABILITY_RESAMPLES
                )
            ),
        },
        "gpu_execution": {
            "heavy_neural_compute_device": (
                "cuda:0"
            ),
            "quiet_cpu_fallback": False,
            "forward_backward_optimizer": (
                "CUDA"
            ),
            "contrastive_similarity": (
                "CUDA"
            ),
            "neural_reconstruction": (
                "CUDA"
            ),
            "deterministic_dual_PCA": (
                "CUDA float64 required for Gram/eigendecomposition/projection"
            ),
            "fold_local_feature_ranking_and_imputation": (
                "CPU allowed"
            ),
        },
        "profile_development_partition": {
            "fit": (
                "05c1 folds 3,4,5"
            ),
            "early_stop": (
                "05c1 fold 2"
            ),
            "validation": (
                "05c1 fold 1"
            ),
            "validation_used_for_early_stopping": False,
            "validation_used_for_preprocessing": False,
            "selected_epoch_refit": (
                "reinitialize with same frozen profile fit seed; "
                "refit on fit+early-stop for selected epoch count"
            ),
        },
        "preprocessing": {
            "feature_budgets": (
                PRIMARY_FEATURE_BUDGETS
            ),
            "ranking": (
                "training-subset pre-standardization variance descending; "
                "original feature index deterministic tie-break"
            ),
            "missingness_threshold": (
                MAX_FEATURE_MISSING_FRACTION
            ),
            "imputation": (
                "training-subset feature median"
            ),
            "centering_scaling": (
                "training-subset mean and sample standard deviation after imputation"
            ),
            "PCA": (
                "algebraically matched dual PCA with 05c1 sign convention; "
                "Gram matrix, eigendecomposition, and projection run in float64 on cuda:0"
            ),
            "global_rankings_used": False,
            "complete_truth_used_for_preprocessing": False,
        },
        "objective": {
            "source": relative_path(
                LOSS_CONTRACT
            ),
            "variant": (
                "V4_SABLE_MO"
            ),
            "gradient_clip_norm": (
                GRADIENT_CLIP_NORM
            ),
            "optimizer": (
                "AdamW foreach=False fused=False"
            ),
            "early_stop_monitor": (
                "deterministic total objective with modality dropout disabled "
                "and systematic leave-one-view-out cross reconstruction"
            ),
        },
        "utility": {
            "weights": {
                "shared_subspace_recovery": 0.40,
                "private_subspace_recovery": 0.20,
                "random_adjusted_cross_view_mrr": 0.20,
                "missing_view_reconstruction_score": 0.10,
                "modality_balance_score": 0.10,
            },
            "true_zero_dimensional_truth_rule": (
                "Raw subspace recovery remains NA when the corresponding truth "
                "subspace has zero dimensions. For the prespecified fixed-weight "
                "selection utility only, that absent truth component contributes "
                "0.0 rather than dropping or reweighting the component."
            ),
            "scenario_aggregation": (
                "mean across 10 development replicates"
            ),
            "overall_aggregation": (
                "equal-weight mean across 36 scenario means"
            ),
            "standard_error_unit": (
                "36 scenario means"
            ),
        },
        "stability": {
            "reference_fit": (
                "one full-data fit per dataset-profile using frozen reference seed "
                "and the profile-specific selected epoch count"
            ),
            "resamples": (
                STABILITY_RESAMPLES
            ),
            "subsample_fraction": 0.80,
            "resample_preprocessing_refit": True,
            "resample_neural_refit": True,
            "alignment": (
                "standardize shared scores, orthogonal Procrustes resample to "
                "reference, Hungarian factor matching, absolute correlation"
            ),
            "thresholds": (
                stability_thresholds
            ),
            "biological_truth_match_absolute_correlation": (
                TRUTH_MATCH_THRESHOLD
            ),
            "technical_truth_match_absolute_correlation": (
                TECHNICAL_MATCH_THRESHOLD
            ),
            "false_biological_discovery_denominator": (
                "stable nontechnical factors"
            ),
            "stable_technical_factors_counted_as_false_biological": False,
            "rotational_nonnull_factor_false_discovery_primary_eligible": False,
        },
        "hard_profile_eligibility": {
            "required_fit_failure_count": 0,
            "development_reference_latent_collapse_count": 0,
            "mean_stability_gated_false_biological_discovery_fraction_maximum": (
                0.10
            ),
            "false_discovery_eligible_datasets": (
                "factor-level-identifiable datasets plus true biological null datasets"
            ),
        },
        "selection": {
            "rule": (
                "one-standard-error rule from frozen 05d1 contract"
            ),
            "best_profile": (
                "highest mean utility among hard-gate-eligible profiles"
            ),
            "one_se_threshold": (
                "best mean utility minus SE(best) across 36 scenario means"
            ),
            "selection_within_one_se_set": (
                "lowest combined complexity rank; then architecture complexity; "
                "then regularization complexity; then profile_id lexicographically"
            ),
            "TARGET_data_used": False,
            "confirmatory_replicates_11_to_20_used": False,
        },
        "runtime_optimization_v2": {
            "reason": (
                "Interrupted v1 runtime pilot stabilized near 170-180 hours "
                "after approximately 30 PASS shards; no scientific performance "
                "metric values were inspected before this execution rewrite."
            ),
            "scientific_contract_changed": False,
            "changes": [
                "remove per-epoch CUDA-to-host mask/count synchronizations",
                "remove per-epoch fixed-fit loss scalar transfers",
                "replace dynamic-length masked losses with algebraically equivalent fixed-shape CUDA masks",
                "replace modality-dropout nonzero indexing with synchronization-free one-hot masking",
                "retain deterministic float32 neural arithmetic, float64 CUDA dual PCA, AdamW, seeds, profiles, losses, resamples and one-SE rule",
            ],
            "v1_partial_shards_reused": False,
            "v1_partial_shards_preserved_for_audit": True,
        },
        "resumability": {
            "unit": (
                "simulation dataset x profile"
            ),
            "n_shards": (
                EXPECTED_SHARDS
            ),
            "existing_shards_verified_by": [
                "script version",
                "execution protocol SHA256",
                "input SHA256",
                "scenario/replicate",
                "profile ID",
            ],
        },
        "upstream_hashes": (
            upstream[
                "hashes"
            ]
        ),
    }


def protocol_core(
    protocol: dict[str, Any],
) -> dict[str, Any]:
    output = dict(
        protocol
    )

    output.pop(
        "created_at_utc",
        None,
    )

    return output


def initialize_or_verify_protocol(
    upstream: dict[str, Any],
) -> tuple[
    dict[str, Any],
    str,
]:
    proposed = build_execution_protocol(
        upstream
    )

    if OUTPUT_PROTOCOL.exists():
        existing = load_json(
            OUTPUT_PROTOCOL
        )

        if json.dumps(
            protocol_core(
                existing
            ),
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        ) != json.dumps(
            protocol_core(
                proposed
            ),
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        ):
            raise RuntimeError(
                "Existing prospective 05d2 execution protocol differs from the current script. "
                "A new versioned stage is required."
            )

        protocol = existing

    else:
        write_json_atomic(
            proposed,
            OUTPUT_PROTOCOL,
        )

        protocol = proposed

    return (
        protocol,
        sha256_file(
            OUTPUT_PROTOCOL
        ),
    )


# ---------------------------------------------------------------------
# CUDA implementation reference fixture
# ---------------------------------------------------------------------

def run_reference_fixture(
    profile: ProfileSpec,
) -> dict[str, Any]:
    rng = np.random.default_rng(
        20521
    )

    # Verify the GPU implementation of the frozen dual-PCA algebra before
    # any development simulation result is generated.
    pca_matrix = rng.normal(
        size=(
            42,
            240,
        )
    ).astype(
        np.float64
    )

    pca_a = fit_dual_pca_gpu(
        pca_matrix,
        15,
    )

    pca_scores_a = pca_a.transform(
        pca_matrix,
        15,
    )

    pca_b = fit_dual_pca_gpu(
        pca_matrix,
        15,
    )

    pca_scores_b = pca_b.transform(
        pca_matrix,
        15,
    )

    exact_pca_components = bool(
        torch.equal(
            pca_a.components,
            pca_b.components,
        )
    )

    exact_pca_scores = bool(
        np.array_equal(
            pca_scores_a,
            pca_scores_b,
        )
    )

    if not (
        exact_pca_components
        and exact_pca_scores
    ):
        raise RuntimeError(
            "05d2 CUDA dual-PCA fixture is not exactly repeatable."
        )

    del pca_a
    del pca_b

    torch.cuda.empty_cache()

    n = 36

    input_dims = {
        "RNA": min(
            10,
            profile.pca_components[
                "RNA"
            ],
        ),
        "METHYLATION": min(
            10,
            profile.pca_components[
                "METHYLATION"
            ],
        ),
        "CNV": min(
            10,
            profile.pca_components[
                "CNV"
            ],
        ),
    }

    latent = rng.normal(
        size=(
            n,
            3,
        )
    )

    views = {}

    for view in VIEWS:
        weights = rng.normal(
            scale=0.7,
            size=(
                3,
                input_dims[
                    view
                ],
            ),
        )

        views[
            view
        ] = (
            latent
            @ weights
            + rng.normal(
                scale=0.5,
                size=(
                    n,
                    input_dims[
                        view
                    ],
                ),
            )
        ).astype(
            np.float32
        )

    availability = np.ones(
        (
            n,
            3,
        ),
        dtype=bool,
    )

    availability[
        0,
        2,
    ] = False

    availability[
        1,
        1,
    ] = False

    seed = 20260821
    epochs = 12

    model_a, audit_a = (
        train_fixed_epochs(
            profile=profile,
            training_views_np=views,
            availability_np=(
                availability
            ),
            fit_seed=seed,
            epochs=epochs,
            fit_stage=(
                "reference_fixture_a"
            ),
        )
    )

    state_a = {
        key: value.detach()
        .cpu()
        .clone()
        for key, value in (
            model_a.state_dict().items()
        )
    }

    encoded_a = encode_numpy(
        model_a,
        views,
        availability,
    )

    del model_a

    torch.cuda.empty_cache()

    model_b, audit_b = (
        train_fixed_epochs(
            profile=profile,
            training_views_np=views,
            availability_np=(
                availability
            ),
            fit_seed=seed,
            epochs=epochs,
            fit_stage=(
                "reference_fixture_b"
            ),
        )
    )

    state_b = {
        key: value.detach()
        .cpu()
        .clone()
        for key, value in (
            model_b.state_dict().items()
        )
    }

    encoded_b = encode_numpy(
        model_b,
        views,
        availability,
    )

    exact_state = (
        list(
            state_a
        )
        == list(
            state_b
        )
        and all(
            torch.equal(
                state_a[
                    key
                ],
                state_b[
                    key
                ],
            )
            for key in state_a
        )
    )

    exact_scores = bool(
        np.array_equal(
            encoded_a[
                "patient_shared"
            ],
            encoded_b[
                "patient_shared"
            ],
        )
    )

    if not (
        exact_state
        and exact_scores
    ):
        raise RuntimeError(
            "05d2 CUDA implementation fixture is not exactly repeatable."
        )

    # Compatibility guard: the modality-dropout path must actually have
    # executed successfully on CUDA during both fixed-epoch fixture fits.
    if (
        clean_text(audit_a.get("fit_status")) != "PASS"
        or clean_text(audit_b.get("fit_status")) != "PASS"
    ):
        raise RuntimeError(
            "05d2 modality-dropout CUDA fixture did not complete successfully."
        )

    del model_b

    torch.cuda.empty_cache()

    return {
        "fixture_id": (
            "05d2-v4-cuda-profile-screen-v1"
        ),
        "profile_id": (
            profile.profile_id
        ),
        "seed": seed,
        "epochs": epochs,
        "device": "cuda:0",
        "torch_version": (
            torch.__version__
        ),
        "cuda_build": (
            torch.version.cuda
        ),
        "exact_gpu_dual_pca_component_match": (
            exact_pca_components
        ),
        "exact_gpu_dual_pca_score_match": (
            exact_pca_scores
        ),
        "exact_state_dict_match": (
            exact_state
        ),
        "exact_shared_score_match": (
            exact_scores
        ),
        "final_objective_a": (
            audit_a[
                "final_training_objective"
            ]
        ),
        "final_objective_b": (
            audit_b[
                "final_training_objective"
            ]
        ),
        "status": "PASS",
    }


# ---------------------------------------------------------------------
# Resumable dataset x profile shards
# ---------------------------------------------------------------------

def shard_path(
    manifest_row: pd.Series,
    profile: ProfileSpec,
) -> Path:
    dataset_path = Path(
        clean_text(
            manifest_row[
                "path"
            ]
        )
    )

    return (
        OUTPUT_SHARD_DIR
        / clean_text(
            manifest_row[
                "scenario_family"
            ]
        )
        / clean_text(
            manifest_row[
                "scenario_id"
            ]
        )
        / profile.profile_id
        / (
            dataset_path.stem
            + "__"
            + profile.profile_id
            + ".json"
        )
    )


def validate_existing_shard(
    shard: dict[str, Any],
    manifest_row: pd.Series,
    profile: ProfileSpec,
    protocol_hash: str,
) -> None:
    if clean_text(
        shard.get(
            "script_version"
        )
    ) != SCRIPT_VERSION:
        raise RuntimeError(
            "Existing 05d2 shard script-version mismatch."
        )

    if clean_text(
        shard.get(
            "protocol_sha256"
        )
    ) != protocol_hash:
        raise RuntimeError(
            "Existing 05d2 shard protocol hash mismatch."
        )

    if clean_text(
        shard.get(
            "profile_id"
        )
    ) != profile.profile_id:
        raise RuntimeError(
            "Existing 05d2 shard profile mismatch."
        )

    if clean_text(
        shard.get(
            "scenario_id"
        )
    ) != clean_text(
        manifest_row[
            "scenario_id"
        ]
    ):
        raise RuntimeError(
            "Existing 05d2 shard scenario mismatch."
        )

    if int(
        shard.get(
            "replicate_index"
        )
    ) != int(
        float(
            manifest_row[
                "replicate_index"
            ]
        )
    ):
        raise RuntimeError(
            "Existing 05d2 shard replicate mismatch."
        )

    expected_input_hash = clean_text(
        manifest_row[
            "sha256"
        ]
    )

    if clean_text(
        shard.get(
            "input_sha256"
        )
    ) != expected_input_hash:
        raise RuntimeError(
            "Existing 05d2 shard input hash mismatch."
        )


def materialize_shard(
    linear: Any,
    manifest_row: pd.Series,
    profile: ProfileSpec,
    partitions: dict[str, np.ndarray],
    profile_fit_seed: int,
    stability_group: dict[str, Any],
    stability_thresholds: dict[str, Any],
    protocol_hash: str,
) -> tuple[
    dict[str, Any],
    str,
]:
    candidate = shard_path(
        manifest_row,
        profile,
    )

    if candidate.exists():
        shard = load_json(
            candidate
        )

        validate_existing_shard(
            shard,
            manifest_row,
            profile,
            protocol_hash,
        )

        return (
            shard,
            "verified_existing",
        )

    try:
        shard = evaluate_dataset_profile(
            linear=linear,
            manifest_row=(
                manifest_row
            ),
            profile=profile,
            partitions=partitions,
            profile_fit_seed=(
                profile_fit_seed
            ),
            stability_group=(
                stability_group
            ),
            stability_thresholds=(
                stability_thresholds
            ),
        )

    except (
        KeyboardInterrupt,
        SystemExit,
    ):
        raise

    except Exception as exc:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        shard = failed_shard_record(
            manifest_row,
            profile,
            exc,
        )

    shard[
        "protocol_sha256"
    ] = protocol_hash

    shard[
        "created_at_utc"
    ] = utc_now_iso()

    write_json_atomic(
        shard,
        candidate,
    )

    serialized = load_json(
        candidate
    )

    validate_existing_shard(
        serialized,
        manifest_row,
        profile,
        protocol_hash,
    )

    return (
        serialized,
        "generated",
    )


# ---------------------------------------------------------------------
# Aggregate shard outputs
# ---------------------------------------------------------------------

def aggregate_shards(
    shards: list[
        dict[str, Any]
    ],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    dataset_rows = []

    stability_rows = []

    factor_rows = []

    fit_rows = []

    preprocessing_rows = []

    for shard in shards:
        dataset_rows.append(
            shard[
                "dataset_row"
            ]
        )

        stability_rows.append(
            shard[
                "stability_summary"
            ]
        )

        factor_rows.extend(
            shard[
                "factor_rows"
            ]
        )

        fit_rows.extend(
            shard[
                "fit_audits"
            ]
        )

        preprocessing_rows.extend(
            shard[
                "preprocessing_rows"
            ]
        )

    return (
        pd.DataFrame(
            dataset_rows
        ),
        pd.DataFrame(
            stability_rows
        ),
        pd.DataFrame(
            factor_rows
        ),
        pd.DataFrame(
            fit_rows
        ),
        pd.DataFrame(
            preprocessing_rows
        ),
    )


# ---------------------------------------------------------------------
# Scenario summaries and one-SE profile selection
# ---------------------------------------------------------------------

def scenario_summary(
    dataset_table: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "simulation_utility",
        "heldout_reconstruction_score",
        "random_adjusted_cross_view_mrr",
        "missing_view_reconstruction_score",
        "modality_balance_score",
        "shared_subspace_recovery",
        "private_subspace_recovery",
        "technical_subspace_recovery",
        "stability_subspace_median_canonical_correlation",
        "stability_gated_false_biological_discovery_fraction",
    ]

    rows = []

    group_columns = [
        "scenario_id",
        "scenario_origin",
        "scenario_family",
        "profile_id",
        "architecture_profile",
        "regularization_profile",
        "combined_complexity_rank",
        "architecture_complexity_rank",
        "regularization_complexity_rank",
    ]

    for keys, part in dataset_table.groupby(
        group_columns,
        sort=True,
        dropna=False,
    ):
        row = dict(
            zip(
                group_columns,
                keys,
            )
        )

        row[
            "n_datasets"
        ] = int(
            part.shape[
                0
            ]
        )

        row[
            "n_passed_shards"
        ] = int(
            part[
                "shard_status"
            ].eq(
                "PASS"
            ).sum()
        )

        row[
            "n_failed_shards"
        ] = int(
            part[
                "shard_status"
            ].eq(
                "FAIL"
            ).sum()
        )

        row[
            "n_latent_collapse"
        ] = int(
            part[
                "latent_collapse"
            ]
            .map(
                parse_bool
            )
            .sum()
        )

        for metric in metrics:
            values = pd.to_numeric(
                part[
                    metric
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

            finite = values[
                np.isfinite(
                    values
                )
            ]

            row[
                f"{metric}_n"
            ] = int(
                finite.size
            )

            row[
                f"{metric}_mean"
            ] = (
                float(
                    np.mean(
                        finite
                    )
                )
                if finite.size
                else None
            )

            row[
                f"{metric}_median"
            ] = (
                float(
                    np.median(
                        finite
                    )
                )
                if finite.size
                else None
            )

        rows.append(
            row
        )

    output = pd.DataFrame(
        rows
    )

    expected = (
        36
        * EXPECTED_PROFILES
    )

    if output.shape[
        0
    ] != expected:
        raise RuntimeError(
            f"Expected {expected} scenario-profile summary rows; found {output.shape[0]}."
        )

    return output


def profile_overall_summary(
    dataset_table: pd.DataFrame,
    scenario_table: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    profile_columns = [
        "profile_id",
        "architecture_profile",
        "regularization_profile",
        "combined_complexity_rank",
        "architecture_complexity_rank",
        "regularization_complexity_rank",
    ]

    for keys, part in dataset_table.groupby(
        profile_columns,
        sort=True,
    ):
        row = dict(
            zip(
                profile_columns,
                keys,
            )
        )

        profile_id = clean_text(
            row[
                "profile_id"
            ]
        )

        scenario_part = scenario_table[
            scenario_table[
                "profile_id"
            ].eq(
                profile_id
            )
        ].copy()

        if scenario_part.shape[
            0
        ] != 36:
            raise RuntimeError(
                f"{profile_id}: expected 36 scenario means."
            )

        scenario_utilities = pd.to_numeric(
            scenario_part[
                "simulation_utility_mean"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        finite_scenario_utilities = (
            scenario_utilities[
                np.isfinite(
                    scenario_utilities
                )
            ]
        )

        n_failed = int(
            part[
                "shard_status"
            ].eq(
                "FAIL"
            ).sum()
        )

        n_collapse = int(
            part[
                "latent_collapse"
            ]
            .map(
                parse_bool
            )
            .sum()
        )

        n_missing_utility = int(
            pd.to_numeric(
                part[
                    "simulation_utility"
                ],
                errors="coerce",
            )
            .isna()
            .sum()
        )

        false_discovery_eligible = part[
            part[
                "factor_level_metrics_primary_eligible"
            ].map(
                parse_bool
            )
            & part[
                "shard_status"
            ].eq(
                "PASS"
            )
        ]

        false_discovery_values = (
            pd.to_numeric(
                false_discovery_eligible[
                    "stability_gated_false_biological_discovery_fraction"
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )
        )

        false_discovery_values = (
            false_discovery_values[
                np.isfinite(
                    false_discovery_values
                )
            ]
        )

        mean_false_discovery = (
            float(
                np.mean(
                    false_discovery_values
                )
            )
            if false_discovery_values.size
            else None
        )

        if finite_scenario_utilities.size == 36:
            overall_mean = float(
                np.mean(
                    finite_scenario_utilities
                )
            )

            overall_sd = float(
                np.std(
                    finite_scenario_utilities,
                    ddof=1,
                )
            )

            overall_se = float(
                overall_sd
                / math.sqrt(
                    36
                )
            )

        else:
            overall_mean = None
            overall_sd = None
            overall_se = None

        hard_gate_nonfinite = bool(
            n_failed
            > 0
            or n_missing_utility
            > 0
            or finite_scenario_utilities.size
            != 36
        )

        hard_gate_collapse = bool(
            n_collapse
            > 0
        )

        hard_gate_false_discovery = bool(
            mean_false_discovery
            is None
            or mean_false_discovery
            > 0.10
        )

        eligible = bool(
            not hard_gate_nonfinite
            and not hard_gate_collapse
            and not hard_gate_false_discovery
        )

        row.update(
            {
                "n_datasets": int(
                    part.shape[
                        0
                    ]
                ),
                "n_failed_shards": (
                    n_failed
                ),
                "n_missing_utility": (
                    n_missing_utility
                ),
                "n_latent_collapse": (
                    n_collapse
                ),
                "n_false_discovery_eligible_datasets": int(
                    false_discovery_values.size
                ),
                "mean_stability_gated_false_biological_discovery_fraction": (
                    mean_false_discovery
                ),
                "overall_mean_simulation_utility": (
                    overall_mean
                ),
                "scenario_mean_utility_sd": (
                    overall_sd
                ),
                "scenario_mean_utility_se": (
                    overall_se
                ),
                "hard_gate_nonfinite_failed": (
                    hard_gate_nonfinite
                ),
                "hard_gate_latent_collapse_failed": (
                    hard_gate_collapse
                ),
                "hard_gate_false_shared_discovery_failed": (
                    hard_gate_false_discovery
                ),
                "hard_gate_eligible": (
                    eligible
                ),
            }
        )

        rows.append(
            row
        )

    output = pd.DataFrame(
        rows
    )

    if output.shape[
        0
    ] != EXPECTED_PROFILES:
        raise RuntimeError(
            "Overall profile summary does not contain 9 profiles."
        )

    return output


def select_profile_one_se(
    profile_summary: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    table = profile_summary.copy()

    table[
        "within_one_se_of_best"
    ] = False

    table[
        "selected_profile"
    ] = False

    eligible = table[
        table[
            "hard_gate_eligible"
        ].map(
            parse_bool
        )
    ].copy()

    if eligible.empty:
        result = {
            "status": (
                "NO_HARD_GATE_ELIGIBLE_PROFILE"
            ),
            "selected_profile_id": None,
            "best_profile_id": None,
            "best_mean_simulation_utility": None,
            "best_standard_error": None,
            "one_se_threshold": None,
            "n_hard_gate_eligible_profiles": 0,
            "n_within_one_se_profiles": 0,
            "outcome_access": False,
            "TARGET_access": False,
            "confirmatory_replicates_11_to_20_accessed": False,
        }

        return (
            table,
            result,
        )

    eligible[
        "overall_mean_simulation_utility"
    ] = pd.to_numeric(
        eligible[
            "overall_mean_simulation_utility"
        ],
        errors="raise",
    )

    eligible[
        "scenario_mean_utility_se"
    ] = pd.to_numeric(
        eligible[
            "scenario_mean_utility_se"
        ],
        errors="raise",
    )

    best_row = eligible.sort_values(
        [
            "overall_mean_simulation_utility",
            "combined_complexity_rank",
            "architecture_complexity_rank",
            "regularization_complexity_rank",
            "profile_id",
        ],
        ascending=[
            False,
            True,
            True,
            True,
            True,
        ],
        kind="mergesort",
    ).iloc[
        0
    ]

    best_mean = float(
        best_row[
            "overall_mean_simulation_utility"
        ]
    )

    best_se = float(
        best_row[
            "scenario_mean_utility_se"
        ]
    )

    one_se_threshold = (
        best_mean
        - best_se
    )

    within = eligible[
        eligible[
            "overall_mean_simulation_utility"
        ]
        >= one_se_threshold
        - 1e-12
    ].copy()

    within = within.sort_values(
        [
            "combined_complexity_rank",
            "architecture_complexity_rank",
            "regularization_complexity_rank",
            "profile_id",
        ],
        ascending=[
            True,
            True,
            True,
            True,
        ],
        kind="mergesort",
    )

    selected_row = within.iloc[
        0
    ]

    selected_profile = clean_text(
        selected_row[
            "profile_id"
        ]
    )

    table.loc[
        table[
            "profile_id"
        ].isin(
            within[
                "profile_id"
            ]
        ),
        "within_one_se_of_best",
    ] = True

    table.loc[
        table[
            "profile_id"
        ].eq(
            selected_profile
        ),
        "selected_profile",
    ] = True

    result = {
        "status": (
            "PROFILE_SELECTED_BY_FROZEN_ONE_SE_RULE"
        ),
        "selected_profile_id": (
            selected_profile
        ),
        "selected_architecture_profile": clean_text(
            selected_row[
                "architecture_profile"
            ]
        ),
        "selected_regularization_profile": clean_text(
            selected_row[
                "regularization_profile"
            ]
        ),
        "selected_combined_complexity_rank": int(
            selected_row[
                "combined_complexity_rank"
            ]
        ),
        "best_profile_id": clean_text(
            best_row[
                "profile_id"
            ]
        ),
        "best_mean_simulation_utility": (
            best_mean
        ),
        "best_standard_error": (
            best_se
        ),
        "one_se_threshold": (
            one_se_threshold
        ),
        "selected_mean_simulation_utility": float(
            selected_row[
                "overall_mean_simulation_utility"
            ]
        ),
        "n_hard_gate_eligible_profiles": int(
            eligible.shape[
                0
            ]
        ),
        "n_within_one_se_profiles": int(
            within.shape[
                0
            ]
        ),
        "selection_order": [
            "hard-gate eligibility",
            "within one SE of highest eligible mean utility",
            "lowest combined complexity rank",
            "lowest architecture complexity rank",
            "lowest regularization complexity rank",
            "lexicographically smaller profile_id",
        ],
        "profile_selected_for": (
            "V4_SABLE_MO"
        ),
        "profile_reused_by_confirmatory_variants": [
            "V0_BALANCED_AE",
            "V1_SHARED_PRIVATE_AE",
            "V2_CONTRASTIVE_SPAE",
            "V3_DROPOUT_SPAE",
            "V5_SABLE_VAE",
        ],
        "outcome_access": False,
        "TARGET_access": False,
        "confirmatory_replicates_11_to_20_accessed": False,
        "final_gate_calibration_accessed": False,
        "representation_method_selected": False,
    }

    return (
        table,
        result,
    )


# ---------------------------------------------------------------------
# Frozen-result verification
# ---------------------------------------------------------------------

def verify_existing_lock() -> None:
    lock = load_json(
        OUTPUT_LOCK
    )

    failures = []

    if clean_text(
        lock.get(
            "script_version"
        )
    ) != SCRIPT_VERSION:
        failures.append(
            "script version mismatch"
        )

    for relative_name, record in lock.get(
        "generated_files",
        {},
    ).items():
        candidate = resolve_project_path(
            relative_name
        )

        if not candidate.exists():
            failures.append(
                f"missing output: {relative_name}"
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

    shard_record = lock.get(
        "shard_manifest",
        {},
    )

    shard_manifest_path = resolve_project_path(
        shard_record.get(
            "path",
            "",
        )
    )

    if (
        not shard_manifest_path.exists()
        or not shard_manifest_path.is_file()
    ):
        failures.append(
            "shard manifest missing"
        )

    else:
        shard_manifest = pd.read_csv(
            shard_manifest_path,
            dtype=str,
        ).fillna(
            ""
        )

        if shard_manifest.shape[
            0
        ] != EXPECTED_SHARDS:
            failures.append(
                "shard manifest row-count mismatch"
            )

        else:
            for _, row in shard_manifest.iterrows():
                candidate = resolve_project_path(
                    row[
                        "shard_path"
                    ]
                )

                if not candidate.exists():
                    failures.append(
                        f"missing shard: {row['shard_path']}"
                    )
                    break

                if int(
                    candidate.stat().st_size
                ) != int(
                    float(
                        row[
                            "size_bytes"
                        ]
                    )
                ):
                    failures.append(
                        f"shard size mismatch: {row['shard_path']}"
                    )
                    break

                if sha256_file(
                    candidate
                ) != clean_text(
                    row[
                        "sha256"
                    ]
                ):
                    failures.append(
                        f"shard hash mismatch: {row['shard_path']}"
                    )
                    break

    if failures:
        raise RuntimeError(
            "Existing 05d2 frozen-result verification failed:\n"
            + "\n".join(
                f"- {failure}"
                for failure in failures
            )
        )

    print(
        "="
        * 88
    )

    print(
        "Existing Paper 5 neural profile screen: PASS"
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
# README
# ---------------------------------------------------------------------

def write_readme(
    selection_result: dict[str, Any],
    n_passed_shards: int,
    n_failed_shards: int,
) -> None:
    selected = selection_result.get(
        "selected_profile_id"
    )

    lines = [
        "Paper 5 neural profile simulation screen",
        "=" * 72,
        "",
        f"Script version: {SCRIPT_VERSION}",
        "",
        "Scope",
        "-----",
        "V4 SABLE-MO profile selection only.",
        "Development simulation replicates 1-10 only.",
        "Confirmatory replicates 11-20 were not accessed.",
        "TARGET-OS data were not accessed.",
        "Outcomes and biological annotation were not used for selection.",
        "",
        "GPU execution",
        "-------------",
        "All neural forward passes, backward passes, AdamW updates, contrastive",
        "similarity matrices, and neural reconstruction operations ran on cuda:0.",
        "Quiet CPU fallback was forbidden.",
        "CPU was used only for feature eligibility/ranking, median imputation, audit I/O and hashing; dual PCA remained on CUDA.",
        "",
        "Runtime optimization v2",
        "-----------------------",
        "The interrupted v1 execution was excluded from profile selection.",
        "v2 removes per-epoch CUDA-to-host synchronization while preserving the frozen scientific protocol.",
        "",
        "Profile screen",
        "--------------",
        f"Candidate profiles: {EXPECTED_PROFILES}",
        f"Development datasets: {EXPECTED_DEVELOPMENT_DATASETS}",
        f"Dataset x profile shards: {EXPECTED_SHARDS}",
        f"Passed shards: {n_passed_shards}",
        f"Failed shards: {n_failed_shards}",
        f"Stability resamples per shard: {STABILITY_RESAMPLES}",
        "",
        "Selection",
        "---------",
        f"Status: {selection_result.get('status')}",
        f"Selected profile: {selected}",
        "Selection used the frozen hard gates and one-standard-error rule.",
        "The selected profile is not yet the final Paper 5 representation method.",
        "",
        "Next stage",
        "----------",
        (
            "05d3_freeze_selected_neural_profile.py"
            if selected
            else "STOP: no hard-gate-eligible neural profile"
        ),
        "",
    ]

    OUTPUT_README.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
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
        "Paper 5 - GPU neural profile simulation screen v2 (sync-free execution)"
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
        f"Results directory: {OUTPUT_RESULTS_DIR}"
    )

    print()

    OUTPUT_CONTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_SHARD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT_LOCK.exists():
        verify_existing_lock()
        return

    assert_cuda_runtime()

    if ABORTED_V1_LOCK.exists():
        raise RuntimeError(
            "A completed/frozen 05d2 v1 result lock exists. "
            "The sync-free v2 execution is intended only for the interrupted runtime pilot case."
        )

    aborted_v1_shards = (
        sorted(
            ABORTED_V1_SHARD_DIR.rglob("*.json")
        )
        if ABORTED_V1_SHARD_DIR.exists()
        else []
    )

    upstream = verify_upstream_state()

    linear = load_linear_helpers()

    profile_lookup = build_profile_lookup(
        upstream[
            "profile_grid"
        ]
    )

    manifest_lookup = build_manifest_lookup(
        upstream[
            "simulation_manifest"
        ]
    )

    partition_lookup = build_partition_lookup(
        upstream[
            "development_partitions"
        ]
    )

    seed_lookup = build_profile_seed_lookup(
        upstream[
            "profile_fit_seeds"
        ]
    )

    stability_lookup = build_stability_lookup(
        upstream[
            "profile_stability"
        ]
    )

    contract_05a = load_json(
        CONTRACT_05A
    )

    stability_thresholds = contract_05a[
        "stability_gate"
    ][
        "thresholds"
    ]

    print(
        "05d0 frozen CUDA runtime: PASS"
    )

    print(
        "05d1 frozen neural simulation protocol: PASS"
    )

    print(
        "05c1 deterministic metric/PCA implementation: LOADED"
    )

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"CUDA build: {torch.version.cuda}"
    )

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        "Heavy neural computation device: cuda:0"
    )

    print(
        "Quiet CPU fallback: FORBIDDEN"
    )

    print(
        "Automatic mixed precision: OFF"
    )

    print(
        "TF32: OFF"
    )

    print(
        "torch.compile: OFF"
    )

    print(
        "Outcomes/TARGET/confirmatory replicates: FORBIDDEN"
    )

    print()

    execution_protocol, protocol_hash = (
        initialize_or_verify_protocol(
            upstream
        )
    )

    runtime_amendment = {
        "contract_version": "paper5-05d2-runtime-optimization-amendment-v2",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "reason": (
            "05d2 v1 was manually interrupted after its ETA stabilized near "
            "170-180 hours. Only shard PASS status and runtime/ETA were inspected; "
            "no utility, truth-recovery, stability, biological, technical, TARGET "
            "or outcome metric values were inspected."
        ),
        "scientific_protocol_changed": False,
        "v1_partial_shards_found": len(aborted_v1_shards),
        "v1_partial_shards_reused": False,
        "v1_partial_shards_directory": (
            relative_path(ABORTED_V1_SHARD_DIR)
            if ABORTED_V1_SHARD_DIR.exists()
            else None
        ),
        "v2_results_directory": relative_path(OUTPUT_RESULTS_DIR),
        "optimization_changes": [
            "synchronization-free CUDA modality-dropout masking",
            "fixed-shape masked reconstruction/alignment/covariance/InfoNCE losses",
            "no per-epoch host scalar transfer in fixed-epoch stability fits",
            "no redundant per-component or gradient finite host checks",
            "no device-wide synchronize after each fixed-epoch fit",
        ],
        "unchanged_scientific_elements": [
            "05d1 development datasets and partitions",
            "all 9 frozen architecture x regularization profiles",
            "all frozen fit and resample seeds",
            "20 stability resamples per dataset-profile",
            "V4 loss terms and frozen regularization weights",
            "AdamW learning rates and weight decay",
            "early-stopping patience/min_delta and selected-epoch refit rule",
            "stability thresholds and truth-match thresholds",
            "profile hard gates and one-standard-error selection rule",
            "CUDA float32 neural precision, TF32 OFF, AMP OFF, torch.compile OFF",
        ],
    }

    if OUTPUT_EXECUTION_AMENDMENT.exists():
        existing_amendment = load_json(
            OUTPUT_EXECUTION_AMENDMENT
        )
        comparable_existing = dict(
            existing_amendment
        )
        comparable_proposed = dict(
            runtime_amendment
        )
        comparable_existing.pop(
            "created_at_utc",
            None,
        )
        comparable_proposed.pop(
            "created_at_utc",
            None,
        )
        if json.dumps(
            comparable_existing,
            sort_keys=True,
            separators=(",", ":"),
        ) != json.dumps(
            comparable_proposed,
            sort_keys=True,
            separators=(",", ":"),
        ):
            raise RuntimeError(
                "Existing 05d2 v2 runtime amendment differs from this script."
            )
    else:
        write_json_atomic(
            runtime_amendment,
            OUTPUT_EXECUTION_AMENDMENT,
        )

    print(
        "Prospective 05d2 v2 execution protocol: FROZEN"
    )

    print(
        f"Interrupted v1 shards preserved/excluded: {len(aborted_v1_shards)}"
    )

    print(
        "Per-epoch CUDA->CPU synchronization in fixed fits: REMOVED"
    )

    print(
        f"Candidate profiles: {EXPECTED_PROFILES}"
    )

    print(
        f"Development datasets: {EXPECTED_DEVELOPMENT_DATASETS}"
    )

    print(
        f"Dataset x profile shards: {EXPECTED_SHARDS:,}"
    )

    print(
        "Optimization runs if all shards pass: "
        f"{EXPECTED_SHARDS * (2 + 1 + STABILITY_RESAMPLES):,}"
    )

    print(
        f"Stability resamples: {STABILITY_RESAMPLES} per dataset-profile"
    )

    print()

    # Lowest-complexity frozen candidate is used only as a deterministic
    # implementation fixture. No performance result is used.
    fixture_profile = sorted(
        profile_lookup.values(),
        key=lambda profile: (
            profile.combined_complexity_rank,
            profile.architecture_complexity_rank,
            profile.regularization_complexity_rank,
            profile.profile_id,
        ),
    )[
        0
    ]

    print(
        "Running deterministic V4 CUDA implementation fixture..."
    )

    fixture = run_reference_fixture(
        fixture_profile
    )

    write_json_atomic(
        fixture,
        OUTPUT_FIXTURE,
    )

    print(
        "V4 CUDA implementation fixture: PASS"
    )

    print(
        "Exact repeated GPU dual-PCA components/scores: PASS"
    )

    print(
        "Exact repeated model state: PASS"
    )

    print(
        "Exact repeated shared scores: PASS"
    )

    print()

    write_json_atomic(
        {
            "script_version": (
                SCRIPT_VERSION
            ),
            "created_at_utc": (
                utc_now_iso()
            ),
            "protocol_sha256": (
                protocol_hash
            ),
            "planned_shards": (
                EXPECTED_SHARDS
            ),
            "planned_optimization_runs": (
                EXPECTED_SHARDS
                * (
                    2
                    + 1
                    + STABILITY_RESAMPLES
                )
            ),
            "status": "IN_PROGRESS",
        },
        OUTPUT_IN_PROGRESS,
    )

    development = upstream[
        "development_roles"
    ].copy()

    for column in [
        "replicate_index",
        "simulation_seed",
    ]:
        development[
            column
        ] = pd.to_numeric(
            development[
                column
            ],
            errors="raise",
        ).astype(
            int
        )

    development = development.sort_values(
        [
            "scenario_id",
            "replicate_index",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    ordered_profiles = sorted(
        profile_lookup.values(),
        key=lambda profile: (
            profile.combined_complexity_rank,
            profile.architecture_complexity_rank,
            profile.regularization_complexity_rank,
            profile.profile_id,
        ),
    )

    shards = []

    shard_manifest_rows = []

    current = 0

    generated_count = 0

    generated_seconds = 0.0

    total_shards = EXPECTED_SHARDS

    for _, role_row in development.iterrows():
        dataset_key = (
            clean_text(
                role_row[
                    "scenario_id"
                ]
            ),
            int(
                role_row[
                    "replicate_index"
                ]
            ),
            int(
                role_row[
                    "simulation_seed"
                ]
            ),
        )

        if dataset_key not in manifest_lookup:
            raise RuntimeError(
                f"Development dataset missing from simulation manifest: {dataset_key}"
            )

        if dataset_key not in partition_lookup:
            raise RuntimeError(
                f"Development partition missing: {dataset_key}"
            )

        manifest_row = manifest_lookup[
            dataset_key
        ]

        partitions = partition_lookup[
            dataset_key
        ]

        for profile in ordered_profiles:
            current += 1

            profile_key = (
                dataset_key[
                    0
                ],
                dataset_key[
                    1
                ],
                dataset_key[
                    2
                ],
                profile.profile_id,
            )

            if profile_key not in seed_lookup:
                raise RuntimeError(
                    f"Profile fit seed missing: {profile_key}"
                )

            if profile_key not in stability_lookup:
                raise RuntimeError(
                    f"Profile stability registry missing: {profile_key}"
                )

            started = time.perf_counter()

            shard, generation_status = materialize_shard(
                linear=linear,
                manifest_row=(
                    manifest_row
                ),
                profile=profile,
                partitions=partitions,
                profile_fit_seed=(
                    seed_lookup[
                        profile_key
                    ]
                ),
                stability_group=(
                    stability_lookup[
                        profile_key
                    ]
                ),
                stability_thresholds=(
                    stability_thresholds
                ),
                protocol_hash=(
                    protocol_hash
                ),
            )

            elapsed = (
                time.perf_counter()
                - started
            )

            if generation_status == "generated":
                generated_count += 1

                generated_seconds += (
                    elapsed
                )

            shards.append(
                shard
            )

            candidate = shard_path(
                manifest_row,
                profile,
            )

            shard_status = clean_text(
                shard[
                    "dataset_row"
                ].get(
                    "shard_status"
                )
            )

            shard_manifest_rows.append(
                {
                    "scenario_id": (
                        dataset_key[
                            0
                        ]
                    ),
                    "scenario_family": clean_text(
                        manifest_row[
                            "scenario_family"
                        ]
                    ),
                    "replicate_index": (
                        dataset_key[
                            1
                        ]
                    ),
                    "simulation_seed": (
                        dataset_key[
                            2
                        ]
                    ),
                    "profile_id": (
                        profile.profile_id
                    ),
                    "architecture_profile": (
                        profile.architecture_profile
                    ),
                    "regularization_profile": (
                        profile.regularization_profile
                    ),
                    "input_path": clean_text(
                        manifest_row[
                            "path"
                        ]
                    ),
                    "input_sha256": clean_text(
                        manifest_row[
                            "sha256"
                        ]
                    ),
                    "shard_path": relative_path(
                        candidate
                    ),
                    "size_bytes": int(
                        candidate.stat().st_size
                    ),
                    "sha256": sha256_file(
                        candidate
                    ),
                    "generation_status": (
                        generation_status
                    ),
                    "shard_status": (
                        shard_status
                    ),
                    "wall_seconds": (
                        elapsed
                    ),
                }
            )

            if (
                current
                == 1
                or current
                % 10
                == 0
                or shard_status
                == "FAIL"
                or current
                == total_shards
            ):
                if generated_count > 0:
                    mean_seconds = (
                        generated_seconds
                        / generated_count
                    )

                    remaining = (
                        total_shards
                        - current
                    )

                    eta_hours = (
                        remaining
                        * mean_seconds
                        / 3600.0
                    )

                    eta_text = (
                        f" | ETA~{eta_hours:.1f}h"
                    )

                else:
                    eta_text = ""

                print(
                    f"{current:04d}/{total_shards:04d} "
                    f"{dataset_key[0]} "
                    f"R{dataset_key[1]:03d} "
                    f"{profile.profile_id}: "
                    f"{generation_status}/{shard_status}"
                    f"{eta_text}"
                )

                if shard_status == "FAIL":
                    print(
                        "    ERROR: "
                        + clean_text(
                            shard[
                                "dataset_row"
                            ].get(
                                "shard_error_type"
                            )
                        )
                        + ": "
                        + clean_text(
                            shard[
                                "dataset_row"
                            ].get(
                                "shard_error_message"
                            )
                        )[
                            :500
                        ]
                    )

    shard_manifest = pd.DataFrame(
        shard_manifest_rows
    )

    if shard_manifest.shape[
        0
    ] != EXPECTED_SHARDS:
        raise RuntimeError(
            "05d2 shard manifest does not contain 3,240 rows."
        )

    shard_manifest.to_csv(
        OUTPUT_SHARD_MANIFEST,
        index=False,
        lineterminator="\n",
    )

    (
        dataset_table,
        stability_table,
        factor_table,
        fit_table,
        preprocessing_table,
    ) = aggregate_shards(
        shards
    )

    if dataset_table.shape[
        0
    ] != EXPECTED_SHARDS:
        raise RuntimeError(
            "05d2 dataset metric table does not contain 3,240 rows."
        )

    if stability_table.shape[
        0
    ] != EXPECTED_SHARDS:
        raise RuntimeError(
            "05d2 stability summary does not contain 3,240 rows."
        )

    preliminary_failed_shards = int(
        dataset_table["shard_status"].eq("FAIL").sum()
    )

    if preliminary_failed_shards == 0:
        expected_fit_rows = EXPECTED_SHARDS * (2 + 1 + STABILITY_RESAMPLES)
        expected_preprocessing_rows = EXPECTED_SHARDS * (
            3 + 3 + 3 + 3 * STABILITY_RESAMPLES
        )
        expected_factor_rows = (
            EXPECTED_DEVELOPMENT_DATASETS
            * sum(profile.shared_dim for profile in profile_lookup.values())
        )

        if fit_table.shape[0] != expected_fit_rows:
            raise RuntimeError(
                f"Expected {expected_fit_rows:,} neural fit-audit rows after an all-PASS screen; "
                f"found {fit_table.shape[0]:,}."
            )

        if preprocessing_table.shape[0] != expected_preprocessing_rows:
            raise RuntimeError(
                f"Expected {expected_preprocessing_rows:,} preprocessing audit rows after an all-PASS screen; "
                f"found {preprocessing_table.shape[0]:,}."
            )

        if factor_table.shape[0] != expected_factor_rows:
            raise RuntimeError(
                f"Expected {expected_factor_rows:,} stability factor rows after an all-PASS screen; "
                f"found {factor_table.shape[0]:,}."
            )

    write_csv_gzip(
        dataset_table,
        OUTPUT_DATASET_METRICS,
    )

    write_csv_gzip(
        stability_table,
        OUTPUT_STABILITY_SUMMARY,
    )

    write_csv_gzip(
        factor_table,
        OUTPUT_STABILITY_FACTORS,
    )

    write_csv_gzip(
        fit_table,
        OUTPUT_FIT_AUDIT,
    )

    write_csv_gzip(
        preprocessing_table,
        OUTPUT_PREPROCESSING,
    )

    scenario_table = scenario_summary(
        dataset_table
    )

    scenario_table.to_csv(
        OUTPUT_SCENARIO_SUMMARY,
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )

    overall_table = profile_overall_summary(
        dataset_table,
        scenario_table,
    )

    overall_table.to_csv(
        OUTPUT_PROFILE_SUMMARY,
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )

    (
        selection_table,
        selection_result,
    ) = select_profile_one_se(
        overall_table
    )

    selection_table.to_csv(
        OUTPUT_SELECTION_TABLE,
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )

    write_json_atomic(
        selection_result,
        OUTPUT_SELECTION_RESULT,
    )

    n_passed_shards = int(
        dataset_table[
            "shard_status"
        ].eq(
            "PASS"
        ).sum()
    )

    n_failed_shards = int(
        dataset_table[
            "shard_status"
        ].eq(
            "FAIL"
        ).sum()
    )

    write_readme(
        selection_result,
        n_passed_shards,
        n_failed_shards,
    )

    output_files = [
        OUTPUT_PROTOCOL,
        OUTPUT_EXECUTION_AMENDMENT,
        OUTPUT_FIXTURE,
        OUTPUT_SHARD_MANIFEST,
        OUTPUT_DATASET_METRICS,
        OUTPUT_STABILITY_SUMMARY,
        OUTPUT_STABILITY_FACTORS,
        OUTPUT_FIT_AUDIT,
        OUTPUT_PREPROCESSING,
        OUTPUT_SCENARIO_SUMMARY,
        OUTPUT_PROFILE_SUMMARY,
        OUTPUT_SELECTION_TABLE,
        OUTPUT_SELECTION_RESULT,
        OUTPUT_README,
    ]

    profile_selected = bool(
        selection_result.get(
            "selected_profile_id"
        )
    )

    result_manifest = {
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
            "biological_annotations_used_for_selection": False,
            "TARGET_model_fitting_performed": False,
            "neural_profile_screening_performed": True,
            "neural_profile_selected": (
                profile_selected
            ),
            "representation_method_selected": False,
            "final_gate_calibration_data_used": False,
        },
        "scope": {
            "candidate_profiles": (
                EXPECTED_PROFILES
            ),
            "development_datasets": (
                EXPECTED_DEVELOPMENT_DATASETS
            ),
            "dataset_profile_shards": int(
                dataset_table.shape[
                    0
                ]
            ),
            "passed_shards": (
                n_passed_shards
            ),
            "failed_shards": (
                n_failed_shards
            ),
            "fit_audit_rows": int(
                fit_table.shape[
                    0
                ]
            ),
            "preprocessing_audit_rows": int(
                preprocessing_table.shape[
                    0
                ]
            ),
            "stability_factor_rows": int(
                factor_table.shape[
                    0
                ]
            ),
            "selected_profile_id": (
                selection_result.get(
                    "selected_profile_id"
                )
            ),
        },
        "execution_protocol": {
            "path": relative_path(
                OUTPUT_PROTOCOL
            ),
            "sha256": (
                protocol_hash
            ),
        },
        "selection_result": {
            "path": relative_path(
                OUTPUT_SELECTION_RESULT
            ),
            "sha256": sha256_file(
                OUTPUT_SELECTION_RESULT
            ),
            "status": (
                selection_result[
                    "status"
                ]
            ),
            "selected_profile_id": (
                selection_result.get(
                    "selected_profile_id"
                )
            ),
        },
        "shard_manifest": {
            "path": relative_path(
                OUTPUT_SHARD_MANIFEST
            ),
            "sha256": sha256_file(
                OUTPUT_SHARD_MANIFEST
            ),
            "n_rows": int(
                shard_manifest.shape[
                    0
                ]
            ),
        },
        "outputs": {
            relative_path(
                candidate
            ): {
                "sha256": sha256_file(
                    candidate
                ),
                "size_bytes": int(
                    candidate.stat().st_size
                ),
            }
            for candidate in output_files
        },
        "upstream_hashes": (
            upstream[
                "hashes"
            ]
        ),
        "guardrails": [
            "Interrupted 05d2 v1 shards were preserved for audit but excluded from all v2 aggregation and selection.",
            "Only development simulation replicates 1-10 were used.",
            "Confirmatory replicates 11-20 were not accessed.",
            "TARGET-OS matrices were not accessed.",
            "Outcomes and clinical labels were not accessed.",
            "Heavy neural computation executed on frozen cuda:0.",
            "Quiet CPU fallback was forbidden.",
            "All data-dependent preprocessing was local to the applicable fit subset.",
            "Validation fold 1 was never used for early stopping or preprocessing.",
            "All candidate profiles and failed shards remain visible.",
            "Final gate-calibration simulations were not accessed.",
            "The selected neural profile does not constitute final representation-method selection.",
        ],
    }

    write_json_atomic(
        result_manifest,
        OUTPUT_RESULT_MANIFEST,
    )

    generated_files = {
        relative_path(
            candidate
        ): {
            "sha256": sha256_file(
                candidate
            ),
            "size_bytes": int(
                candidate.stat().st_size
            ),
        }
        for candidate in [
            *output_files,
            OUTPUT_RESULT_MANIFEST,
        ]
    }

    lock_status = (
        "FROZEN_NEURAL_PROFILE_SCREEN_RESULTS_BEFORE_CONFIRMATORY_ABLATION"
        if profile_selected
        else "FROZEN_NEURAL_PROFILE_SCREEN_RESULTS_NO_ELIGIBLE_PROFILE"
    )

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
            lock_status
        ),
        "scientific_state": (
            result_manifest[
                "scientific_state"
            ]
        ),
        "result_manifest": {
            "path": relative_path(
                OUTPUT_RESULT_MANIFEST
            ),
            "sha256": sha256_file(
                OUTPUT_RESULT_MANIFEST
            ),
        },
        "selection_result": (
            result_manifest[
                "selection_result"
            ]
        ),
        "shard_manifest": (
            result_manifest[
                "shard_manifest"
            ]
        ),
        "generated_files": (
            generated_files
        ),
        "rewrite_policy": (
            "This zero-argument stage never rewrites frozen 05d2 results. "
            "Any scientifically justified implementation change requires a new "
            "versioned prospective execution stage and full neural refitting."
        ),
    }

    write_json_atomic(
        lock,
        OUTPUT_LOCK,
    )

    if OUTPUT_IN_PROGRESS.exists():
        OUTPUT_IN_PROGRESS.unlink()

    print()

    print(
        "-"
        * 88
    )

    print(
        "Neural profile screen summary"
    )

    print(
        "-"
        * 88
    )

    print(
        f"Dataset x profile shards: {dataset_table.shape[0]:,}"
    )

    print(
        f"Passed shards: {n_passed_shards:,}"
    )

    print(
        f"Failed shards: {n_failed_shards:,}"
    )

    print(
        f"Fit-audit rows: {fit_table.shape[0]:,}"
    )

    print(
        f"Stability factor rows: {factor_table.shape[0]:,}"
    )

    print()

    display_columns = [
        "profile_id",
        "overall_mean_simulation_utility",
        "scenario_mean_utility_se",
        "mean_stability_gated_false_biological_discovery_fraction",
        "n_latent_collapse",
        "n_failed_shards",
        "hard_gate_eligible",
        "within_one_se_of_best",
        "selected_profile",
    ]

    print(
        selection_table[
            display_columns
        ].sort_values(
            [
                "selected_profile",
                "hard_gate_eligible",
                "overall_mean_simulation_utility",
            ],
            ascending=[
                False,
                False,
                False,
            ],
            kind="mergesort",
        ).to_string(
            index=False
        )
    )

    print()

    print(
        "Selection status: "
        f"{selection_result['status']}"
    )

    print(
        "Selected profile: "
        f"{selection_result.get('selected_profile_id')}"
    )

    print(
        "TARGET accessed: NO"
    )

    print(
        "Confirmatory replicates 11-20 accessed: NO"
    )

    print(
        "Final gate-calibration data accessed: NO"
    )

    print(
        "Representation method selected: NO"
    )

    print()

    print(
        "="
        * 88
    )

    if profile_selected:
        print(
            "Paper 5 neural profile simulation screen: PASS"
        )

        print(
            "="
            * 88
        )

        print(
            f"Selected neural profile: {selection_result['selected_profile_id']}"
        )

        print(
            f"Selection result: {relative_path(OUTPUT_SELECTION_RESULT)}"
        )

        print(
            f"Results manifest: {relative_path(OUTPUT_RESULT_MANIFEST)}"
        )

        print(
            f"Lock: {relative_path(OUTPUT_LOCK)}"
        )

        print(
            "Next stage: 05d3_freeze_selected_neural_profile.py"
        )

    else:
        print(
            "Paper 5 neural profile simulation screen: NO ELIGIBLE PROFILE"
        )

        print(
            "="
            * 88
        )

        print(
            f"Selection result: {relative_path(OUTPUT_SELECTION_RESULT)}"
        )

        print(
            f"Results manifest: {relative_path(OUTPUT_RESULT_MANIFEST)}"
        )

        print(
            f"Lock: {relative_path(OUTPUT_LOCK)}"
        )

        print(
            "STOP: do not proceed to confirmatory neural ablations."
        )

    print(
        "="
        * 88
    )


if __name__ == "__main__":
    main()
