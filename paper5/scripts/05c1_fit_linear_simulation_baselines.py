from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import KFold


SCRIPT_VERSION = "05c1-fit-linear-simulation-baselines-v1-no-cli"
CONTRACT_VERSION = "paper5-linear-simulation-baseline-benchmark-v1"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# ---------------------------------------------------------------------
# Frozen upstream contracts
# ---------------------------------------------------------------------

CONTRACT_05A_DIR = PAPER5_ROOT / "method_contract" / "05a"
CONTRACT_05A = CONTRACT_05A_DIR / "ai_method_contract.json"
LOCK_05A = CONTRACT_05A_DIR / "ai_method_contract_lock.json"
MODEL_REGISTRY_FILE = CONTRACT_05A_DIR / "model_family_registry.csv"
ARCHITECTURE_PROFILES_FILE = (
    CONTRACT_05A_DIR / "sable_mo_architecture_profiles.csv"
)
METRIC_REGISTRY_FILE = CONTRACT_05A_DIR / "metric_registry.csv"
FAILURE_GATE_FILE = CONTRACT_05A_DIR / "failure_gate_registry.csv"
SELECTION_POLICY_FILE = CONTRACT_05A_DIR / "model_selection_policy.json"

CONTRACT_05A1_DIR = PAPER5_ROOT / "method_contract" / "05a1"
CONTRACT_05A1 = (
    CONTRACT_05A1_DIR
    / "simulation_and_technical_confound_addendum.json"
)
LOCK_05A1 = (
    CONTRACT_05A1_DIR
    / "simulation_and_technical_confound_addendum_lock.json"
)

TECHNICAL_DIR = (
    PAPER5_ROOT
    / "data_manifest"
    / "TARGET_OS_technical_metadata"
)
TECHNICAL_MANIFEST_FILE = (
    TECHNICAL_DIR
    / "TARGET_OS_technical_metadata_preflight_manifest.json"
)
TECHNICAL_LOCK_FILE = (
    TECHNICAL_DIR
    / "TARGET_OS_technical_metadata_preflight_lock.json"
)

CONTRACT_05B_DIR = PAPER5_ROOT / "method_contract" / "05b_v2"
SIMULATION_LOCK_FILE = (
    CONTRACT_05B_DIR
    / "simulation_generation_v2_lock.json"
)
SIMULATION_MANIFEST_FILE = (
    CONTRACT_05B_DIR
    / "architecture_screening_data_manifest.csv"
)
FINAL_GATE_DATA_ROOT = (
    PAPER5_ROOT
    / "simulations"
    / "sable_mo_v2"
    / "final_gate_calibration"
)

# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------

OUTPUT_CONTRACT_DIR = PAPER5_ROOT / "method_contract" / "05c1"
OUTPUT_RESULTS_DIR = (
    PAPER5_ROOT
    / "results"
    / "simulation_benchmark"
    / "05c1_linear_baselines"
)
OUTPUT_SHARD_DIR = OUTPUT_RESULTS_DIR / "shards"

OUTPUT_PROTOCOL = (
    OUTPUT_CONTRACT_DIR
    / "linear_simulation_evaluation_protocol.json"
)
OUTPUT_REFERENCE_FIXTURE = (
    OUTPUT_CONTRACT_DIR
    / "linear_baseline_reference_fixture.json"
)
OUTPUT_ENVIRONMENT = (
    OUTPUT_CONTRACT_DIR
    / "linear_baseline_environment_snapshot.json"
)
OUTPUT_SHARD_MANIFEST = (
    OUTPUT_CONTRACT_DIR
    / "linear_baseline_shard_manifest.csv"
)
OUTPUT_RESULT_MANIFEST = (
    OUTPUT_CONTRACT_DIR
    / "linear_baseline_results_manifest.json"
)
OUTPUT_README = (
    OUTPUT_CONTRACT_DIR
    / "linear_simulation_baseline_README.txt"
)
OUTPUT_LOCK = (
    OUTPUT_CONTRACT_DIR
    / "linear_simulation_baseline_lock.json"
)
OUTPUT_IN_PROGRESS = (
    OUTPUT_CONTRACT_DIR
    / ".linear_simulation_baseline_in_progress.json"
)

OUTPUT_SPLITS = (
    OUTPUT_RESULTS_DIR
    / "linear_simulation_split_registry.csv.gz"
)
OUTPUT_PREPROCESSING = (
    OUTPUT_RESULTS_DIR
    / "linear_preprocessing_audit.csv.gz"
)
OUTPUT_FOLD_METRICS = (
    OUTPUT_RESULTS_DIR
    / "linear_fold_metrics.csv.gz"
)
OUTPUT_DATASET_METRICS = (
    OUTPUT_RESULTS_DIR
    / "linear_dataset_metrics.csv.gz"
)
OUTPUT_FACTOR_AUDIT = (
    OUTPUT_RESULTS_DIR
    / "linear_factor_truth_match_audit.csv.gz"
)
OUTPUT_FAILURE_AUDIT = (
    OUTPUT_RESULTS_DIR
    / "linear_failure_gate_audit.csv.gz"
)
OUTPUT_SCENARIO_SUMMARY = (
    OUTPUT_RESULTS_DIR
    / "linear_scenario_summary.csv"
)

# ---------------------------------------------------------------------
# Scientific constants
# ---------------------------------------------------------------------

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

SIMULATION_CV_FOLDS = 5
TEST_FRACTION_DESCRIPTION = (
    "Five-fold deterministic held-out evaluation; every sample is held out "
    "exactly once per simulation dataset."
)

MAX_FEATURE_MISSING_FRACTION = 0.20
MIN_FINITE_VALUES = 3
MIN_PREIMPUTATION_VARIANCE = 1e-10
MIN_POSTIMPUTATION_STD = 1e-8
PCA_NUMERICAL_TOLERANCE = 1e-10
VIEW_SUPPORT_ENERGY_THRESHOLD = 0.15
MODALITY_DOMINATION_ENTROPY_THRESHOLD = 0.60
PRIMARY_RANK_FALLBACK = 10

REQUIRED_BASE_MODEL_IDS = {
    "PCA_RNA",
    "PCA_METHYLATION",
    "PCA_CNV",
    "BLOCK_PCA",
}

REQUIRED_SIMULATION_MANIFEST_COLUMNS = {
    "registry_label",
    "simulation_stage",
    "scenario_id",
    "scenario_origin",
    "scenario_family",
    "calibration_role",
    "replicate_index",
    "simulation_seed",
    "n_samples",
    "n_biological_shared_factors",
    "n_private_factors_per_view",
    "n_technical_shared_factors",
    "factor_level_identifiable",
    "subspace_level_identifiable",
    "primary_truth_target",
    "path",
    "size_bytes",
    "sha256",
}

# Paths that must not exist before this stage is frozen.
FUTURE_MODEL_RESULT_PATHS = (
    PAPER5_ROOT / "method_contract" / "05c2",
    PAPER5_ROOT / "method_contract" / "05d",
    PAPER5_ROOT / "method_contract" / "05e",
    PAPER5_ROOT / "results" / "simulation_benchmark" / "05c2_mofa",
    PAPER5_ROOT / "results" / "simulation_benchmark" / "05d_sable_mo",
    PAPER5_ROOT / "results" / "representation_benchmark",
)


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ViewPreprocessor:
    view: str
    selected_indices: np.ndarray
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    training_missing_fraction: np.ndarray
    training_preimputation_variance: np.ndarray

    def transform_observed(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(
            matrix[:, self.selected_indices],
            dtype=np.float64,
        ).copy()

        missing = ~np.isfinite(values)
        if missing.any():
            row_index, column_index = np.where(missing)
            values[row_index, column_index] = self.medians[
                column_index
            ]

        values -= self.means
        values /= self.scales

        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"{self.view}: non-finite values after preprocessing."
            )

        return values

    def transform_complete(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(
            matrix[:, self.selected_indices],
            dtype=np.float64,
        ).copy()

        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"{self.view}: complete truth matrix contains non-finite values."
            )

        values -= self.means
        values /= self.scales

        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"{self.view}: non-finite complete values after preprocessing."
            )

        return values


@dataclass(frozen=True)
class DualPCA:
    mean: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    explained_variance: np.ndarray
    total_variance: float

    def transform(
        self,
        matrix: np.ndarray,
        rank: int,
    ) -> np.ndarray:
        k = min(rank, self.components.shape[0])
        centered = np.asarray(matrix, dtype=np.float64) - self.mean
        return centered @ self.components[:k, :].T

    def inverse_transform(
        self,
        scores: np.ndarray,
        rank: int,
    ) -> np.ndarray:
        k = min(rank, self.components.shape[0])
        return scores[:, :k] @ self.components[:k, :] + self.mean


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

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


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    return clean_text(value).lower() in {
        "true",
        "1",
        "yes",
    }


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
    path = Path(clean_text(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def require_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "little")


def to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): to_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_json_safe(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, (np.bool_,)):
        return bool(value)

    return value


def write_json_atomic(
    value: Any,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.stem + ".",
        suffix=".tmp.json",
        dir=str(path.parent),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)

    try:
        temporary.write_text(
            json.dumps(
                to_json_safe(value),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_csv_gzip(
    table: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

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


def package_version(name: str) -> dict[str, Any]:
    try:
        return {
            "installed": True,
            "version": importlib.metadata.version(name),
        }
    except importlib.metadata.PackageNotFoundError:
        return {
            "installed": False,
            "version": None,
        }


def environment_snapshot() -> dict[str, Any]:
    packages = {
        name: package_version(name)
        for name in [
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
        ]
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
        "linear_dependencies_ready": all(
            packages[name]["installed"]
            for name in packages
        ),
    }


def nanmean_or_none(values: Iterable[Any]) -> float | None:
    array = pd.to_numeric(
        pd.Series(list(values)),
        errors="coerce",
    ).to_numpy(dtype=float)

    finite = np.isfinite(array)
    if not finite.any():
        return None

    return float(np.mean(array[finite]))


def nanmedian_or_none(values: Iterable[Any]) -> float | None:
    array = pd.to_numeric(
        pd.Series(list(values)),
        errors="coerce",
    ).to_numpy(dtype=float)

    finite = np.isfinite(array)
    if not finite.any():
        return None

    return float(np.median(array[finite]))


def harmonic_number(n: int) -> float:
    if n <= 0:
        return float("nan")
    return float(
        np.sum(1.0 / np.arange(1, n + 1, dtype=float))
    )


# ---------------------------------------------------------------------
# Frozen upstream validation
# ---------------------------------------------------------------------

def load_json(path: Path) -> dict[str, Any]:
    require_file(path)
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def validate_upstream_state() -> dict[str, Any]:
    for path in [
        CONTRACT_05A,
        LOCK_05A,
        MODEL_REGISTRY_FILE,
        ARCHITECTURE_PROFILES_FILE,
        METRIC_REGISTRY_FILE,
        FAILURE_GATE_FILE,
        SELECTION_POLICY_FILE,
        CONTRACT_05A1,
        LOCK_05A1,
        TECHNICAL_MANIFEST_FILE,
        TECHNICAL_LOCK_FILE,
        SIMULATION_LOCK_FILE,
        SIMULATION_MANIFEST_FILE,
    ]:
        require_file(path)

    contract_05a = load_json(CONTRACT_05A)
    lock_05a = load_json(LOCK_05A)
    addendum = load_json(CONTRACT_05A1)
    addendum_lock = load_json(LOCK_05A1)
    technical_manifest = load_json(TECHNICAL_MANIFEST_FILE)
    technical_lock = load_json(TECHNICAL_LOCK_FILE)
    simulation_lock = load_json(SIMULATION_LOCK_FILE)

    if sha256_file(TECHNICAL_MANIFEST_FILE) != clean_text(
        technical_lock.get("manifest_sha256")
    ):
        raise RuntimeError(
            "The 04d technical manifest does not match its frozen lock."
        )

    scientific_state = simulation_lock.get(
        "scientific_state",
        {},
    )

    forbidden_true_fields = [
        "outcomes_generated",
        "clinical_labels_generated",
        "biological_annotations_loaded",
        "simulation_model_fitting_performed",
        "target_model_fitting_performed",
        "stability_gate_thresholds_retuned",
        "architecture_results_viewed",
    ]

    failures = [
        key
        for key in forbidden_true_fields
        if bool(scientific_state.get(key, False))
    ]

    if failures:
        raise RuntimeError(
            "The 05b v2 lock does not preserve the required pre-fit state: "
            + ", ".join(failures)
        )

    if clean_text(simulation_lock.get("status")) != (
        "FROZEN_ARCHITECTURE_SCREENING_INPUTS_BEFORE_MODEL_FITTING"
    ):
        raise RuntimeError(
            "Unexpected 05b v2 lock status."
        )

    locked_manifest = simulation_lock.get(
        "data_manifest",
        {},
    )
    expected_manifest_hash = clean_text(
        locked_manifest.get("sha256")
    )
    observed_manifest_hash = sha256_file(
        SIMULATION_MANIFEST_FILE
    )

    if observed_manifest_hash != expected_manifest_hash:
        raise RuntimeError(
            "The 05b v2 simulation manifest does not match its frozen lock."
        )

    manifest = pd.read_csv(
        SIMULATION_MANIFEST_FILE,
        dtype=str,
        low_memory=False,
    ).fillna("")

    missing_columns = (
        REQUIRED_SIMULATION_MANIFEST_COLUMNS
        - set(manifest.columns)
    )
    if missing_columns:
        raise ValueError(
            "The simulation manifest is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    expected_rows = int(
        simulation_lock.get("scope", {}).get(
            "n_datasets",
            0,
        )
    )
    if manifest.shape[0] != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} simulation datasets, "
            f"found {manifest.shape[0]}."
        )

    if manifest["path"].duplicated().any():
        raise RuntimeError(
            "Duplicate simulation paths were found in the frozen manifest."
        )

    if not manifest["simulation_stage"].eq(
        "architecture_screening"
    ).all():
        raise RuntimeError(
            "The 05c1 baseline benchmark may use architecture-screening "
            "simulations only."
        )

    if FINAL_GATE_DATA_ROOT.exists():
        final_gate_files = [
            path
            for path in FINAL_GATE_DATA_ROOT.rglob("*")
            if path.is_file()
        ]
        if final_gate_files:
            raise RuntimeError(
                "Final gate-calibration simulation data were materialized "
                "before architecture selection."
            )

    for path in FUTURE_MODEL_RESULT_PATHS:
        if path.exists():
            files = [
                item
                for item in path.rglob("*")
                if item.is_file()
            ]
            if files:
                raise RuntimeError(
                    "A future model-result stage exists before 05c1 was frozen: "
                    f"{relative_path(path)}"
                )

    model_registry = pd.read_csv(
        MODEL_REGISTRY_FILE,
        dtype=str,
    ).fillna("")

    observed_model_ids = set(
        model_registry["model_id"].astype(str)
    )
    if not REQUIRED_BASE_MODEL_IDS.issubset(observed_model_ids):
        raise RuntimeError(
            "Required linear baselines are missing from the 05a model registry."
        )

    architecture_profiles = pd.read_csv(
        ARCHITECTURE_PROFILES_FILE,
        dtype=str,
    ).fillna("")

    ranks = sorted(
        {
            int(value)
            for value in pd.to_numeric(
                architecture_profiles["shared_latent_dim"],
                errors="raise",
            ).tolist()
        }
    )

    if ranks != [5, 10, 15]:
        raise RuntimeError(
            f"Expected frozen shared latent ranks [5, 10, 15], found {ranks}."
        )

    standard = architecture_profiles[
        architecture_profiles[
            "architecture_profile"
        ].eq("standard")
    ]
    if standard.shape[0] != 1:
        raise RuntimeError(
            "Could not identify the frozen standard architecture profile."
        )

    primary_rank = int(
        float(
            standard.iloc[0]["shared_latent_dim"]
        )
    )

    truth_match_threshold = float(
        addendum.get(
            "truth_matching",
            {},
        ).get(
            "factor_truth_match_absolute_correlation",
            float("nan"),
        )
    )

    if not math.isfinite(truth_match_threshold):
        raise RuntimeError(
            "The 05a1 truth-matching threshold is unavailable."
        )

    technical_state = technical_manifest.get(
        "scientific_state",
        {},
    )
    if bool(technical_state.get("outcome_loaded", False)):
        raise RuntimeError(
            "The technical preflight accessed outcome values."
        )
    if bool(technical_state.get("clinical_values_retained", False)):
        raise RuntimeError(
            "The technical preflight retained clinical values."
        )
    if bool(
        technical_state.get(
            "representation_model_fitting_performed",
            False,
        )
    ):
        raise RuntimeError(
            "The technical preflight reports representation-model fitting."
        )

    return {
        "contract_05a": contract_05a,
        "lock_05a": lock_05a,
        "addendum": addendum,
        "addendum_lock": addendum_lock,
        "technical_manifest": technical_manifest,
        "technical_lock": technical_lock,
        "simulation_lock": simulation_lock,
        "manifest": manifest,
        "ranks": ranks,
        "primary_rank": primary_rank,
        "truth_match_threshold": truth_match_threshold,
        "upstream_hashes": {
            relative_path(path): sha256_file(path)
            for path in [
                CONTRACT_05A,
                LOCK_05A,
                MODEL_REGISTRY_FILE,
                ARCHITECTURE_PROFILES_FILE,
                METRIC_REGISTRY_FILE,
                FAILURE_GATE_FILE,
                SELECTION_POLICY_FILE,
                CONTRACT_05A1,
                LOCK_05A1,
                TECHNICAL_MANIFEST_FILE,
                TECHNICAL_LOCK_FILE,
                SIMULATION_LOCK_FILE,
                SIMULATION_MANIFEST_FILE,
            ]
        },
    }


# ---------------------------------------------------------------------
# Prospective linear-evaluation protocol
# ---------------------------------------------------------------------

def build_protocol(upstream: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "scientific_state": {
            "outcomes_loaded": False,
            "clinical_labels_loaded": False,
            "biological_annotations_loaded": False,
            "TARGET_model_fitting_performed": False,
            "MOFA_model_fitting_performed": False,
            "neural_model_fitting_performed": False,
            "architecture_selected": False,
            "stability_gate_thresholds_retuned": False,
        },
        "scope": {
            "simulation_stage": "architecture_screening",
            "n_datasets": int(upstream["manifest"].shape[0]),
            "model_families": [
                "modality_specific_PCA",
                "block_scaled_PCA",
            ],
            "base_model_ids": [
                "PCA_RNA",
                "PCA_METHYLATION",
                "PCA_CNV",
                "BLOCK_PCA",
            ],
            "latent_rank_grid": upstream["ranks"],
            "primary_linear_rank": upstream["primary_rank"],
        },
        "heldout_design": {
            "folds": SIMULATION_CV_FOLDS,
            "description": TEST_FRACTION_DESCRIPTION,
            "split_seed_rule": (
                "SHA256-derived deterministic seed from script version, "
                "scenario ID, replicate index, and simulation seed"
            ),
            "stratification": "none; simulations contain no outcome labels",
            "every_sample_held_out_once": True,
        },
        "fold_local_preprocessing": {
            "maximum_feature_missing_fraction": (
                MAX_FEATURE_MISSING_FRACTION
            ),
            "minimum_finite_training_values": MIN_FINITE_VALUES,
            "minimum_preimputation_variance": (
                MIN_PREIMPUTATION_VARIANCE
            ),
            "imputation": (
                "training-fold feature median; all-NaN features are excluded"
            ),
            "centering": "training-fold mean after median imputation",
            "scaling": (
                "training-fold sample standard deviation after median imputation"
            ),
            "feature_selection_after_zscore": False,
            "complete_truth_used_for_preprocessing": False,
            "heldout_samples_used_for_preprocessing": False,
        },
        "PCA": {
            "solver": (
                "deterministic dual PCA from the training sample Gram matrix"
            ),
            "sign_rule": (
                "largest absolute loading in each component is positive"
            ),
            "rank_sensitivities": upstream["ranks"],
            "primary_rank": upstream["primary_rank"],
        },
        "block_scaled_PCA": {
            "block_scaling": (
                "Each fold-local standardized view is divided by the square "
                "root of its total training variance before concatenation."
            ),
            "equal_total_view_variance": True,
            "naive_unscaled_concatenation_allowed": False,
            "view_contribution_embedding": (
                "Each view contributes X_view_scaled @ V_view.T in the common "
                "block-PCA coordinate system."
            ),
            "cross_view_alignment": (
                "Training-only orthogonal Procrustes alignment of view-contribution "
                "embeddings before held-out retrieval."
            ),
        },
        "simulation_truth_evaluation": {
            "full_data_fit_allowed_for_truth_recovery": True,
            "full_data_truth_recovery_not_used_as_heldout_reconstruction": True,
            "heldout_reconstruction_uses_complete_known_simulation_truth": True,
            "pair_corruption_truth": (
                "Cross-view retrieval is scored against row_source arrays, not "
                "nominal row index. Nominal-index retrieval is retained only as a "
                "diagnostic."
            ),
            "integrated_sample_anchor": "RNA row_source",
            "factor_truth_match_absolute_correlation": (
                upstream["truth_match_threshold"]
            ),
            "factor_level_and_subspace_level_metrics_separate": True,
            "rotationally_ambiguous_factor_claims_allowed": False,
        },
        "primary_metrics": {
            "heldout_reconstruction": (
                "MSE relative to the training-mean null on complete simulation truth"
            ),
            "shared_subspace_recovery": (
                "mean canonical correlation between estimated and true biological "
                "shared score subspaces"
            ),
            "private_subspace_recovery": (
                "mean canonical correlation between view-specific estimated "
                "contributions and true private score subspaces"
            ),
            "cross_view_retrieval": (
                "true-source same-patient MRR, top-1 accuracy, and similarity margin"
            ),
            "missing_view_reconstruction": (
                "held-out R2 for reconstructing an omitted standardized view from "
                "the other block contributions"
            ),
            "modality_balance": (
                "normalized entropy of per-view loading energy"
            ),
            "technical_contamination": (
                "canonical correlation with known technical latent truth"
            ),
        },
        "failure_gate_policy": {
            "linear_baselines_are_not_silently_omitted": True,
            "failures_are_reported_not_hidden": True,
            "G09_false_shared_gate_is_diagnostic_for_ungated_PCA": True,
            "architecture_selection_performed_here": False,
        },
        "upstream_hashes": upstream["upstream_hashes"],
        "rewrite_policy": (
            "This zero-argument script never rewrites a frozen 05c1 lock. "
            "Any scientifically justified change requires a new versioned stage."
        ),
    }


def protocol_core(protocol: dict[str, Any]) -> dict[str, Any]:
    output = dict(protocol)
    output.pop("created_at_utc", None)
    return output


def initialize_or_verify_protocol(
    upstream: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    proposed = build_protocol(upstream)

    if OUTPUT_PROTOCOL.exists():
        existing = load_json(OUTPUT_PROTOCOL)
        if canonical_json(protocol_core(existing)) != canonical_json(
            protocol_core(proposed)
        ):
            raise RuntimeError(
                "The existing prospective 05c1 protocol differs from the "
                "current script. A new versioned stage is required."
            )
        protocol = existing
    else:
        write_json_atomic(proposed, OUTPUT_PROTOCOL)
        protocol = proposed

    return protocol, sha256_file(OUTPUT_PROTOCOL)


# ---------------------------------------------------------------------
# Simulation loading
# ---------------------------------------------------------------------

def load_simulation_arrays(path: Path) -> dict[str, np.ndarray]:
    required = {
        "sample_index",
        "Z_biological_shared",
        "biological_support_mask",
        "biological_factor_ids",
        "biological_factor_truth_class",
        "Z_technical",
        "technical_support_mask",
        "technical_factor_ids",
        "factor_level_identifiable",
        "subspace_level_identifiable",
        "primary_truth_target",
    }

    for view in VIEWS:
        prefix = VIEW_PREFIX[view]
        required.update(
            {
                f"X_{prefix}",
                f"X_complete_{prefix}",
                f"Z_private_{prefix}",
                f"row_source_{prefix}",
            }
        )

    with np.load(path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise RuntimeError(
                f"Simulation file {path.name} is missing arrays: "
                + ", ".join(sorted(missing))
            )

        arrays = {
            key: archive[key]
            for key in required
        }

    n = int(arrays["sample_index"].shape[0])

    for view in VIEWS:
        prefix = VIEW_PREFIX[view]
        observed = arrays[f"X_{prefix}"]
        complete = arrays[f"X_complete_{prefix}"]
        row_source = arrays[f"row_source_{prefix}"].astype(int)

        if observed.shape != complete.shape:
            raise RuntimeError(
                f"{view}: observed/complete shape mismatch."
            )
        if observed.shape[0] != n:
            raise RuntimeError(
                f"{view}: sample-count mismatch."
            )
        if not np.all(np.isfinite(complete)):
            raise RuntimeError(
                f"{view}: complete truth contains non-finite values."
            )
        if set(row_source.tolist()) != set(range(n)):
            raise RuntimeError(
                f"{view}: row_source is not a permutation."
            )

    return arrays


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

def fit_view_preprocessor(
    view: str,
    training_matrix: np.ndarray,
) -> ViewPreprocessor:
    values = np.asarray(
        training_matrix,
        dtype=np.float64,
    )

    finite = np.isfinite(values)
    n_finite = finite.sum(axis=0)
    missing_fraction = 1.0 - (
        n_finite / values.shape[0]
    )

    with np.errstate(all="ignore"):
        pre_variance = np.nanvar(
            values,
            axis=0,
            ddof=1,
        )

    initial_keep = (
        (n_finite >= MIN_FINITE_VALUES)
        & (missing_fraction <= MAX_FEATURE_MISSING_FRACTION + 1e-12)
        & np.isfinite(pre_variance)
        & (pre_variance > MIN_PREIMPUTATION_VARIANCE)
    )

    initial_indices = np.flatnonzero(initial_keep)
    if initial_indices.size == 0:
        raise RuntimeError(
            f"{view}: no features passed fold-local eligibility."
        )

    selected = values[:, initial_indices].copy()

    with np.errstate(all="ignore"):
        medians = np.nanmedian(selected, axis=0)

    if not np.all(np.isfinite(medians)):
        raise RuntimeError(
            f"{view}: non-finite training medians after eligibility filtering."
        )

    missing = ~np.isfinite(selected)
    if missing.any():
        rows, columns = np.where(missing)
        selected[rows, columns] = medians[columns]

    means = np.mean(selected, axis=0)
    scales = np.std(selected, axis=0, ddof=1)

    post_keep = (
        np.isfinite(means)
        & np.isfinite(scales)
        & (scales > MIN_POSTIMPUTATION_STD)
    )

    if not post_keep.any():
        raise RuntimeError(
            f"{view}: no features retained after post-imputation numerical checks."
        )

    retained_indices = initial_indices[post_keep]

    return ViewPreprocessor(
        view=view,
        selected_indices=retained_indices.astype(np.int64),
        medians=medians[post_keep],
        means=means[post_keep],
        scales=scales[post_keep],
        training_missing_fraction=missing_fraction[retained_indices],
        training_preimputation_variance=pre_variance[retained_indices],
    )


def fit_all_preprocessors(
    arrays: dict[str, np.ndarray],
    training_indices: np.ndarray,
) -> dict[str, ViewPreprocessor]:
    output: dict[str, ViewPreprocessor] = {}

    for view in VIEWS:
        prefix = VIEW_PREFIX[view]
        output[view] = fit_view_preprocessor(
            view,
            arrays[f"X_{prefix}"][training_indices, :],
        )

    return output


def transformed_views(
    arrays: dict[str, np.ndarray],
    preprocessors: dict[str, ViewPreprocessor],
    indices: np.ndarray,
    *,
    complete: bool,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}

    for view in VIEWS:
        prefix = VIEW_PREFIX[view]
        matrix = arrays[
            f"X_complete_{prefix}"
            if complete
            else f"X_{prefix}"
        ][indices, :]

        if complete:
            output[view] = preprocessors[view].transform_complete(matrix)
        else:
            output[view] = preprocessors[view].transform_observed(matrix)

    return output


def calculate_block_scales(
    training_views: dict[str, np.ndarray],
) -> dict[str, float]:
    scales: dict[str, float] = {}

    for view, matrix in training_views.items():
        variance = np.var(matrix, axis=0, ddof=1)
        total_variance = float(np.sum(variance))

        if not math.isfinite(total_variance) or total_variance <= 0:
            raise RuntimeError(
                f"{view}: invalid total variance for block scaling."
            )

        scales[view] = math.sqrt(total_variance)

    return scales


def scale_blocks(
    views: dict[str, np.ndarray],
    block_scales: dict[str, float],
) -> dict[str, np.ndarray]:
    return {
        view: matrix / block_scales[view]
        for view, matrix in views.items()
    }


# ---------------------------------------------------------------------
# Deterministic PCA
# ---------------------------------------------------------------------

def fit_dual_pca(
    training_matrix: np.ndarray,
    maximum_rank: int,
) -> DualPCA:
    matrix = np.asarray(
        training_matrix,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "PCA input must be two-dimensional."
        )
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError(
            "PCA input contains non-finite values."
        )

    mean = np.mean(matrix, axis=0)
    centered = matrix - mean

    gram = centered @ centered.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    positive = eigenvalues > PCA_NUMERICAL_TOLERANCE
    eigenvalues = eigenvalues[positive]
    eigenvectors = eigenvectors[:, positive]

    available_rank = min(
        maximum_rank,
        eigenvalues.shape[0],
        centered.shape[1],
    )

    if available_rank < 1:
        raise RuntimeError(
            "PCA has zero numerical rank."
        )

    eigenvalues = eigenvalues[:available_rank]
    eigenvectors = eigenvectors[:, :available_rank]
    singular_values = np.sqrt(eigenvalues)

    components = (
        eigenvectors.T @ centered
    ) / singular_values[:, np.newaxis]

    component_norms = np.linalg.norm(
        components,
        axis=1,
    )
    components /= component_norms[:, np.newaxis]
    singular_values *= component_norms

    # Deterministic sign convention.
    for index in range(components.shape[0]):
        pivot = int(
            np.argmax(np.abs(components[index, :]))
        )
        if components[index, pivot] < 0:
            components[index, :] *= -1.0
            eigenvectors[:, index] *= -1.0

    denominator = max(1, matrix.shape[0] - 1)
    explained_variance = (
        singular_values ** 2
    ) / denominator
    total_variance = float(
        np.sum(np.var(centered, axis=0, ddof=1))
    )

    return DualPCA(
        mean=mean,
        components=components,
        singular_values=singular_values,
        explained_variance=explained_variance,
        total_variance=total_variance,
    )


def effective_rank(scores: np.ndarray) -> float:
    centered = np.asarray(scores, dtype=np.float64)
    centered -= np.mean(centered, axis=0)

    singular_values = np.linalg.svd(
        centered,
        compute_uv=False,
        full_matrices=False,
    )
    energy = singular_values ** 2
    total = float(np.sum(energy))

    if total <= 0 or not math.isfinite(total):
        return 0.0

    probabilities = energy / total
    positive = probabilities > 0
    entropy = -float(
        np.sum(
            probabilities[positive]
            * np.log(probabilities[positive])
        )
    )
    return float(math.exp(entropy))


# ---------------------------------------------------------------------
# Truth alignment metrics
# ---------------------------------------------------------------------

def centered_orthonormal_basis(
    matrix: np.ndarray,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)

    if values.ndim != 2 or values.shape[1] == 0:
        return np.zeros(
            (values.shape[0], 0),
            dtype=np.float64,
        )

    values = values - np.mean(values, axis=0)
    u, singular_values, _ = np.linalg.svd(
        values,
        full_matrices=False,
    )

    if singular_values.size == 0:
        return np.zeros(
            (values.shape[0], 0),
            dtype=np.float64,
        )

    tolerance = (
        max(values.shape)
        * np.finfo(float).eps
        * singular_values[0]
    )
    rank = int(np.sum(singular_values > tolerance))
    return u[:, :rank]


def subspace_metrics(
    estimated: np.ndarray,
    truth: np.ndarray,
) -> dict[str, float | None]:
    estimated_basis = centered_orthonormal_basis(
        estimated
    )
    truth_basis = centered_orthonormal_basis(
        truth
    )

    if (
        estimated_basis.shape[1] == 0
        or truth_basis.shape[1] == 0
    ):
        return {
            "mean_canonical_correlation": None,
            "minimum_canonical_correlation": None,
            "mean_principal_angle_degrees": None,
            "maximum_principal_angle_degrees": None,
            "normalized_projection_distance": None,
            "estimated_subspace_rank": int(
                estimated_basis.shape[1]
            ),
            "truth_subspace_rank": int(
                truth_basis.shape[1]
            ),
        }

    singular_values = np.linalg.svd(
        estimated_basis.T @ truth_basis,
        compute_uv=False,
    )
    singular_values = np.clip(
        singular_values,
        0.0,
        1.0,
    )
    angles = np.degrees(
        np.arccos(singular_values)
    )

    projection_estimated = (
        estimated_basis @ estimated_basis.T
    )
    projection_truth = truth_basis @ truth_basis.T
    denominator = math.sqrt(
        estimated_basis.shape[1]
        + truth_basis.shape[1]
    )
    distance = float(
        np.linalg.norm(
            projection_estimated - projection_truth,
            ord="fro",
        )
        / denominator
    )

    return {
        "mean_canonical_correlation": float(
            np.mean(singular_values)
        ),
        "minimum_canonical_correlation": float(
            np.min(singular_values)
        ),
        "mean_principal_angle_degrees": float(
            np.mean(angles)
        ),
        "maximum_principal_angle_degrees": float(
            np.max(angles)
        ),
        "normalized_projection_distance": distance,
        "estimated_subspace_rank": int(
            estimated_basis.shape[1]
        ),
        "truth_subspace_rank": int(
            truth_basis.shape[1]
        ),
    }


def standardized_columns(matrix: np.ndarray) -> np.ndarray:
    """Standardize while preserving the original column index.

    Constant columns are represented by zeros rather than dropped. This is
    important for factor-audit rows because estimated component indices must
    remain aligned with loading-energy and truth-matching records.
    """
    values = np.asarray(matrix, dtype=np.float64)

    if values.ndim != 2 or values.shape[1] == 0:
        return np.zeros(
            (values.shape[0], 0),
            dtype=np.float64,
        )

    centered = values - np.mean(values, axis=0)
    scales = np.std(centered, axis=0, ddof=1)
    valid = np.isfinite(scales) & (scales > 1e-12)

    standardized = np.zeros_like(centered, dtype=np.float64)
    if valid.any():
        standardized[:, valid] = (
            centered[:, valid] / scales[valid]
        )

    return standardized


def absolute_correlation_matrix(
    estimated: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray:
    left = standardized_columns(estimated)
    right = standardized_columns(truth)

    if left.shape[1] == 0 or right.shape[1] == 0:
        return np.zeros(
            (left.shape[1], right.shape[1]),
            dtype=np.float64,
        )

    denominator = max(1, left.shape[0] - 1)
    return np.abs(
        (left.T @ right) / denominator
    )


def factor_matching(
    estimated: np.ndarray,
    truth: np.ndarray,
    truth_ids: list[str],
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    correlation = absolute_correlation_matrix(
        estimated,
        truth,
    )

    n_estimated = int(estimated.shape[1])
    n_truth = int(truth.shape[1])

    match_rows: list[dict[str, Any]] = []

    if n_estimated == 0:
        summary = {
            "n_estimated_factors": 0,
            "n_truth_factors": n_truth,
            "n_assigned_pairs": 0,
            "n_threshold_matches": 0,
            "true_factor_recovery_rate": (
                None if n_truth == 0 else 0.0
            ),
            "estimated_factor_match_fraction": None,
            "false_shared_discovery_fraction": None,
            "mean_matched_absolute_correlation": None,
            "minimum_matched_absolute_correlation": None,
        }
        return summary, match_rows

    if n_truth == 0:
        for estimated_index in range(n_estimated):
            match_rows.append(
                {
                    "estimated_factor_index": estimated_index,
                    "truth_factor_index": None,
                    "truth_factor_id": "",
                    "absolute_correlation": None,
                    "passes_truth_match_threshold": False,
                }
            )

        summary = {
            "n_estimated_factors": n_estimated,
            "n_truth_factors": 0,
            "n_assigned_pairs": 0,
            "n_threshold_matches": 0,
            "true_factor_recovery_rate": None,
            "estimated_factor_match_fraction": 0.0,
            "false_shared_discovery_fraction": 1.0,
            "mean_matched_absolute_correlation": None,
            "minimum_matched_absolute_correlation": None,
        }
        return summary, match_rows

    row_indices, column_indices = linear_sum_assignment(
        -correlation
    )

    assigned_by_estimated: dict[int, tuple[int, float]] = {
        int(row): (
            int(column),
            float(correlation[row, column]),
        )
        for row, column in zip(
            row_indices,
            column_indices,
        )
    }

    threshold_matches = 0
    matched_correlations: list[float] = []

    for estimated_index in range(n_estimated):
        if estimated_index in assigned_by_estimated:
            truth_index, value = assigned_by_estimated[
                estimated_index
            ]
            passes = value >= threshold
            threshold_matches += int(passes)
            matched_correlations.append(value)
            truth_id = (
                truth_ids[truth_index]
                if truth_index < len(truth_ids)
                else f"TRUTH_{truth_index + 1:03d}"
            )
        else:
            truth_index = None
            truth_id = ""
            value = None
            passes = False

        match_rows.append(
            {
                "estimated_factor_index": estimated_index,
                "truth_factor_index": truth_index,
                "truth_factor_id": truth_id,
                "absolute_correlation": value,
                "passes_truth_match_threshold": passes,
            }
        )

    summary = {
        "n_estimated_factors": n_estimated,
        "n_truth_factors": n_truth,
        "n_assigned_pairs": int(len(row_indices)),
        "n_threshold_matches": int(threshold_matches),
        "true_factor_recovery_rate": float(
            threshold_matches / n_truth
        ),
        "estimated_factor_match_fraction": float(
            threshold_matches / n_estimated
        ),
        "false_shared_discovery_fraction": float(
            1.0 - threshold_matches / n_estimated
        ),
        "mean_matched_absolute_correlation": (
            float(np.mean(matched_correlations))
            if matched_correlations
            else None
        ),
        "minimum_matched_absolute_correlation": (
            float(np.min(matched_correlations))
            if matched_correlations
            else None
        ),
    }

    return summary, match_rows


def row_aligned_truth(
    arrays: dict[str, np.ndarray],
    view: str,
    kind: str,
) -> tuple[np.ndarray, list[str]]:
    prefix = VIEW_PREFIX[view]
    row_source = arrays[f"row_source_{prefix}"].astype(int)

    if kind == "biological":
        support = arrays[
            "biological_support_mask"
        ].astype(bool)
        columns = np.flatnonzero(
            support[:, VIEW_INDEX[view]]
        )
        truth = arrays[
            "Z_biological_shared"
        ][row_source, :][:, columns]
        ids = [
            str(value)
            for value in arrays[
                "biological_factor_ids"
            ][columns].tolist()
        ]
        return truth, ids

    if kind == "private":
        truth = arrays[
            f"Z_private_{prefix}"
        ][row_source, :]
        ids = [
            f"PRIVATE_{view}_{index + 1:03d}"
            for index in range(truth.shape[1])
        ]
        return truth, ids

    if kind == "technical":
        support = arrays[
            "technical_support_mask"
        ].astype(bool)
        columns = np.flatnonzero(
            support[:, VIEW_INDEX[view]]
        )
        truth = arrays[
            "Z_technical"
        ][row_source, :][:, columns]
        ids = [
            str(value)
            for value in arrays[
                "technical_factor_ids"
            ][columns].tolist()
        ]
        return truth, ids

    raise ValueError(f"Unknown truth kind: {kind}")


def integrated_anchor_truth(
    arrays: dict[str, np.ndarray],
    kind: str,
) -> tuple[np.ndarray, list[str]]:
    anchor_source = arrays[
        "row_source_rna"
    ].astype(int)

    if kind == "biological":
        truth = arrays[
            "Z_biological_shared"
        ][anchor_source, :]
        ids = [
            str(value)
            for value in arrays[
                "biological_factor_ids"
            ].tolist()
        ]
        return truth, ids

    if kind == "technical":
        truth = arrays[
            "Z_technical"
        ][anchor_source, :]
        ids = [
            str(value)
            for value in arrays[
                "technical_factor_ids"
            ].tolist()
        ]
        return truth, ids

    raise ValueError(f"Unknown integrated truth kind: {kind}")


# ---------------------------------------------------------------------
# Reconstruction and retrieval metrics
# ---------------------------------------------------------------------

def normalized_reconstruction_metrics(
    reconstructed: np.ndarray,
    complete_truth: np.ndarray,
) -> dict[str, float | None]:
    prediction = np.asarray(
        reconstructed,
        dtype=np.float64,
    )
    truth = np.asarray(
        complete_truth,
        dtype=np.float64,
    )

    if prediction.shape != truth.shape:
        raise ValueError(
            "Reconstruction and truth shapes differ."
        )

    if not np.all(np.isfinite(prediction)):
        return {
            "mse": None,
            "null_mse": None,
            "normalized_error": None,
            "reconstruction_score": None,
            "r2": None,
        }

    residual = truth - prediction
    mse = float(np.mean(residual ** 2))
    null_mse = float(np.mean(truth ** 2))

    if null_mse <= 0 or not math.isfinite(null_mse):
        normalized = None
        score = None
        r2 = None
    else:
        normalized = float(mse / null_mse)
        score = float(
            np.clip(1.0 - normalized, 0.0, 1.0)
        )
        r2 = float(1.0 - normalized)

    return {
        "mse": mse,
        "null_mse": null_mse,
        "normalized_error": normalized,
        "reconstruction_score": score,
        "r2": r2,
    }


def standardize_embedding_train_test(
    training: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    training_values = np.asarray(training, dtype=np.float64)
    test_values = np.asarray(test, dtype=np.float64)

    mean = np.mean(training_values, axis=0)
    scale = np.std(training_values, axis=0, ddof=1)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0

    return (
        (training_values - mean) / scale,
        (test_values - mean) / scale,
    )


def orthogonal_mapping(
    source_training: np.ndarray,
    target_training: np.ndarray,
) -> np.ndarray:
    cross_product = source_training.T @ target_training
    left, _, right_transpose = np.linalg.svd(
        cross_product,
        full_matrices=False,
    )
    return left @ right_transpose


def cosine_similarity_matrix(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)

    left_norm = np.linalg.norm(left_values, axis=1)
    right_norm = np.linalg.norm(right_values, axis=1)
    left_norm[left_norm <= 1e-12] = 1.0
    right_norm[right_norm <= 1e-12] = 1.0

    return (
        left_values / left_norm[:, None]
    ) @ (
        right_values / right_norm[:, None]
    ).T


def one_direction_retrieval(
    source_training: np.ndarray,
    target_training: np.ndarray,
    source_test: np.ndarray,
    target_test: np.ndarray,
    source_truth_ids: np.ndarray,
    target_truth_ids: np.ndarray,
) -> dict[str, float | int | None]:
    source_training_std, source_test_std = (
        standardize_embedding_train_test(
            source_training,
            source_test,
        )
    )
    target_training_std, target_test_std = (
        standardize_embedding_train_test(
            target_training,
            target_test,
        )
    )

    mapping = orthogonal_mapping(
        source_training_std,
        target_training_std,
    )
    aligned_source_test = source_test_std @ mapping

    similarity = cosine_similarity_matrix(
        aligned_source_test,
        target_test_std,
    )

    target_lookup = {
        int(subject_id): index
        for index, subject_id in enumerate(
            target_truth_ids.astype(int).tolist()
        )
    }

    reciprocal_ranks: list[float] = []
    top1: list[float] = []
    margins: list[float] = []
    nominal_reciprocal_ranks: list[float] = []

    for query_index, subject_id in enumerate(
        source_truth_ids.astype(int).tolist()
    ):
        if int(subject_id) not in target_lookup:
            continue

        true_index = target_lookup[int(subject_id)]
        order = np.argsort(
            -similarity[query_index, :],
            kind="mergesort",
        )
        true_rank = int(
            np.flatnonzero(order == true_index)[0]
        ) + 1
        reciprocal_ranks.append(1.0 / true_rank)
        top1.append(float(true_rank == 1))

        nonmatch = np.delete(
            similarity[query_index, :],
            true_index,
        )
        if nonmatch.size > 0:
            margins.append(
                float(
                    similarity[query_index, true_index]
                    - np.median(nonmatch)
                )
            )

        # Nominal row-index pairing is diagnostic only.
        if query_index < similarity.shape[1]:
            nominal_order = order
            nominal_rank = int(
                np.flatnonzero(
                    nominal_order == query_index
                )[0]
            ) + 1
            nominal_reciprocal_ranks.append(
                1.0 / nominal_rank
            )

    n_candidates = int(target_test.shape[0])
    random_mrr = (
        harmonic_number(n_candidates)
        / n_candidates
        if n_candidates > 0
        else float("nan")
    )

    if not reciprocal_ranks:
        return {
            "n_valid_queries": 0,
            "n_candidates": n_candidates,
            "mrr": None,
            "random_expected_mrr": random_mrr,
            "random_adjusted_mrr": None,
            "top1_accuracy": None,
            "similarity_margin": None,
            "nominal_index_mrr": None,
        }

    mrr = float(np.mean(reciprocal_ranks))
    denominator = 1.0 - random_mrr
    adjusted = (
        float(
            np.clip(
                (mrr - random_mrr) / denominator,
                0.0,
                1.0,
            )
        )
        if denominator > 0
        else None
    )

    return {
        "n_valid_queries": int(len(reciprocal_ranks)),
        "n_candidates": n_candidates,
        "mrr": mrr,
        "random_expected_mrr": random_mrr,
        "random_adjusted_mrr": adjusted,
        "top1_accuracy": float(np.mean(top1)),
        "similarity_margin": (
            float(np.median(margins))
            if margins
            else None
        ),
        "nominal_index_mrr": (
            float(np.mean(nominal_reciprocal_ranks))
            if nominal_reciprocal_ranks
            else None
        ),
    }


def cross_view_retrieval_metrics(
    training_embeddings: dict[str, np.ndarray],
    test_embeddings: dict[str, np.ndarray],
    row_source: dict[str, np.ndarray],
    training_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict[str, Any]:
    pair_rows: list[dict[str, Any]] = []

    for source_view in VIEWS:
        for target_view in VIEWS:
            if source_view == target_view:
                continue

            result = one_direction_retrieval(
                training_embeddings[source_view],
                training_embeddings[target_view],
                test_embeddings[source_view],
                test_embeddings[target_view],
                row_source[source_view][test_indices],
                row_source[target_view][test_indices],
            )
            pair_rows.append(
                {
                    "source_view": source_view,
                    "target_view": target_view,
                    **result,
                }
            )

    valid = [
        row
        for row in pair_rows
        if row["mrr"] is not None
    ]

    return {
        "ordered_pair_results": pair_rows,
        "cross_view_mrr": nanmean_or_none(
            row["mrr"] for row in valid
        ),
        "random_expected_cross_view_mrr": nanmean_or_none(
            row["random_expected_mrr"]
            for row in valid
        ),
        "random_adjusted_cross_view_mrr": nanmean_or_none(
            row["random_adjusted_mrr"]
            for row in valid
        ),
        "cross_view_top1_accuracy": nanmean_or_none(
            row["top1_accuracy"]
            for row in valid
        ),
        "same_patient_similarity_margin": nanmedian_or_none(
            row["similarity_margin"]
            for row in valid
        ),
        "nominal_index_cross_view_mrr": nanmean_or_none(
            row["nominal_index_mrr"]
            for row in valid
        ),
        "n_ordered_view_pairs": int(len(valid)),
        "n_valid_retrieval_queries": int(
            sum(
                int(row["n_valid_queries"])
                for row in valid
            )
        ),
    }


def modality_loading_energy(
    components: np.ndarray,
    view_slices: dict[str, slice],
) -> tuple[np.ndarray, np.ndarray]:
    energy = np.zeros(
        (components.shape[0], len(VIEWS)),
        dtype=np.float64,
    )

    for view in VIEWS:
        energy[:, VIEW_INDEX[view]] = np.sum(
            components[:, view_slices[view]] ** 2,
            axis=1,
        )

    total = np.sum(energy, axis=1, keepdims=True)
    fractions = np.divide(
        energy,
        total,
        out=np.zeros_like(energy),
        where=total > 0,
    )

    entropy = np.zeros(components.shape[0], dtype=float)
    for index in range(components.shape[0]):
        positive = fractions[index, :] > 0
        if positive.any():
            entropy[index] = -float(
                np.sum(
                    fractions[index, positive]
                    * np.log(
                        fractions[index, positive]
                    )
                )
                / math.log(len(VIEWS))
            )

    return fractions, entropy


def support_jaccard(
    predicted: np.ndarray,
    truth: np.ndarray,
) -> float:
    union = np.logical_or(predicted, truth)
    intersection = np.logical_and(predicted, truth)

    if not union.any():
        return 1.0

    return float(
        np.sum(intersection) / np.sum(union)
    )


# ---------------------------------------------------------------------
# Full-data truth evaluation
# ---------------------------------------------------------------------

def single_view_full_truth_metrics(
    arrays: dict[str, np.ndarray],
    view: str,
    pca: DualPCA,
    transformed: np.ndarray,
    rank: int,
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scores = pca.transform(transformed, rank)

    biological_truth, biological_ids = row_aligned_truth(
        arrays,
        view,
        "biological",
    )
    private_truth, private_ids = row_aligned_truth(
        arrays,
        view,
        "private",
    )
    technical_truth, technical_ids = row_aligned_truth(
        arrays,
        view,
        "technical",
    )

    biological_subspace = subspace_metrics(
        scores,
        biological_truth,
    )
    biological_matching, factor_rows = factor_matching(
        scores,
        biological_truth,
        biological_ids,
        threshold,
    )
    private_subspace = subspace_metrics(
        scores,
        private_truth,
    )
    technical_subspace = subspace_metrics(
        scores,
        technical_truth,
    )

    biological_corr = absolute_correlation_matrix(
        scores,
        biological_truth,
    )
    private_corr = absolute_correlation_matrix(
        scores,
        private_truth,
    )
    technical_corr = absolute_correlation_matrix(
        scores,
        technical_truth,
    )

    for row in factor_rows:
        index = int(row["estimated_factor_index"])
        row["best_biological_absolute_correlation"] = (
            float(np.max(biological_corr[index, :]))
            if biological_corr.shape[1] > 0
            else None
        )
        row["best_private_absolute_correlation"] = (
            float(np.max(private_corr[index, :]))
            if private_corr.shape[1] > 0
            else None
        )
        row["best_technical_absolute_correlation"] = (
            float(np.max(technical_corr[index, :]))
            if technical_corr.shape[1] > 0
            else None
        )
        row["loading_energy_RNA"] = (
            1.0 if view == "RNA" else 0.0
        )
        row["loading_energy_METHYLATION"] = (
            1.0 if view == "METHYLATION" else 0.0
        )
        row["loading_energy_CNV"] = (
            1.0 if view == "CNV" else 0.0
        )
        row["modality_contribution_entropy"] = 0.0
        row["predicted_view_support"] = view
        row["truth_view_support"] = ""
        row["view_support_jaccard"] = None
        row["view_support_exact"] = None

    metrics = {
        "full_effective_rank": effective_rank(scores),
        "shared_subspace_recovery": biological_subspace[
            "mean_canonical_correlation"
        ],
        "shared_subspace_minimum_correlation": biological_subspace[
            "minimum_canonical_correlation"
        ],
        "shared_subspace_mean_angle_degrees": biological_subspace[
            "mean_principal_angle_degrees"
        ],
        "shared_subspace_max_angle_degrees": biological_subspace[
            "maximum_principal_angle_degrees"
        ],
        "shared_subspace_projection_distance": biological_subspace[
            "normalized_projection_distance"
        ],
        "private_subspace_recovery": private_subspace[
            "mean_canonical_correlation"
        ],
        "technical_subspace_recovery": technical_subspace[
            "mean_canonical_correlation"
        ],
        "true_factor_recovery_rate": biological_matching[
            "true_factor_recovery_rate"
        ],
        "estimated_factor_match_fraction": biological_matching[
            "estimated_factor_match_fraction"
        ],
        "false_shared_discovery_fraction": biological_matching[
            "false_shared_discovery_fraction"
        ],
        "mean_matched_absolute_correlation": biological_matching[
            "mean_matched_absolute_correlation"
        ],
        "minimum_matched_absolute_correlation": biological_matching[
            "minimum_matched_absolute_correlation"
        ],
        "modality_contribution_entropy": 0.0,
        "view_support_jaccard": None,
        "view_support_exact_accuracy": None,
    }

    return metrics, factor_rows


def block_full_truth_metrics(
    arrays: dict[str, np.ndarray],
    pca: DualPCA,
    block_views: dict[str, np.ndarray],
    view_slices: dict[str, slice],
    rank: int,
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    concatenated = np.concatenate(
        [block_views[view] for view in VIEWS],
        axis=1,
    )
    total_scores = pca.transform(concatenated, rank)

    k = total_scores.shape[1]
    components = pca.components[:k, :]

    contribution_scores: dict[str, np.ndarray] = {}
    for view in VIEWS:
        component_slice = components[
            :,
            view_slices[view],
        ]
        mean_slice = pca.mean[
            view_slices[view]
        ]
        contribution_scores[view] = (
            block_views[view] - mean_slice
        ) @ component_slice.T

    biological_truth, biological_ids = integrated_anchor_truth(
        arrays,
        "biological",
    )
    technical_truth, _ = integrated_anchor_truth(
        arrays,
        "technical",
    )

    biological_subspace = subspace_metrics(
        total_scores,
        biological_truth,
    )
    biological_matching, factor_rows = factor_matching(
        total_scores,
        biological_truth,
        biological_ids,
        threshold,
    )
    technical_subspace = subspace_metrics(
        total_scores,
        technical_truth,
    )

    private_recoveries: list[float] = []
    private_corr_by_view: dict[str, np.ndarray] = {}
    for view in VIEWS:
        private_truth, _ = row_aligned_truth(
            arrays,
            view,
            "private",
        )
        private_result = subspace_metrics(
            contribution_scores[view],
            private_truth,
        )
        value = private_result[
            "mean_canonical_correlation"
        ]
        if value is not None:
            private_recoveries.append(float(value))
        private_corr_by_view[view] = absolute_correlation_matrix(
            contribution_scores[view],
            private_truth,
        )

    biological_corr = absolute_correlation_matrix(
        total_scores,
        biological_truth,
    )
    technical_corr = absolute_correlation_matrix(
        total_scores,
        technical_truth,
    )

    energy_fractions, entropy = modality_loading_energy(
        components,
        view_slices,
    )

    biological_support = arrays[
        "biological_support_mask"
    ].astype(bool)
    truth_classes = [
        str(value)
        for value in arrays[
            "biological_factor_truth_class"
        ].tolist()
    ]

    support_jaccards: list[float] = []
    support_exact: list[float] = []

    for row in factor_rows:
        estimated_index = int(
            row["estimated_factor_index"]
        )
        truth_index_raw = row["truth_factor_index"]

        row["best_biological_absolute_correlation"] = (
            float(np.max(biological_corr[estimated_index, :]))
            if biological_corr.shape[1] > 0
            else None
        )
        row["best_private_absolute_correlation"] = nanmean_or_none(
            (
                float(np.max(private_corr_by_view[view][estimated_index, :]))
                if private_corr_by_view[view].shape[1] > 0
                else None
            )
            for view in VIEWS
        )
        row["best_technical_absolute_correlation"] = (
            float(np.max(technical_corr[estimated_index, :]))
            if technical_corr.shape[1] > 0
            else None
        )

        for view in VIEWS:
            row[f"loading_energy_{view}"] = float(
                energy_fractions[
                    estimated_index,
                    VIEW_INDEX[view],
                ]
            )

        row["modality_contribution_entropy"] = float(
            entropy[estimated_index]
        )

        predicted_support = (
            energy_fractions[estimated_index, :]
            >= VIEW_SUPPORT_ENERGY_THRESHOLD
        )
        row["predicted_view_support"] = ";".join(
            view
            for view in VIEWS
            if predicted_support[VIEW_INDEX[view]]
        )

        if truth_index_raw is None:
            row["truth_view_support"] = ""
            row["truth_factor_class"] = ""
            row["view_support_jaccard"] = None
            row["view_support_exact"] = None
            continue

        truth_index = int(truth_index_raw)
        truth_support = biological_support[truth_index, :]
        row["truth_view_support"] = ";".join(
            view
            for view in VIEWS
            if truth_support[VIEW_INDEX[view]]
        )
        row["truth_factor_class"] = (
            truth_classes[truth_index]
            if truth_index < len(truth_classes)
            else ""
        )

        jaccard = support_jaccard(
            predicted_support,
            truth_support,
        )
        exact = float(
            np.array_equal(
                predicted_support,
                truth_support,
            )
        )
        row["view_support_jaccard"] = jaccard
        row["view_support_exact"] = bool(exact)

        if bool(row["passes_truth_match_threshold"]):
            support_jaccards.append(jaccard)
            support_exact.append(exact)

    metrics = {
        "full_effective_rank": effective_rank(total_scores),
        "shared_subspace_recovery": biological_subspace[
            "mean_canonical_correlation"
        ],
        "shared_subspace_minimum_correlation": biological_subspace[
            "minimum_canonical_correlation"
        ],
        "shared_subspace_mean_angle_degrees": biological_subspace[
            "mean_principal_angle_degrees"
        ],
        "shared_subspace_max_angle_degrees": biological_subspace[
            "maximum_principal_angle_degrees"
        ],
        "shared_subspace_projection_distance": biological_subspace[
            "normalized_projection_distance"
        ],
        "private_subspace_recovery": (
            float(np.mean(private_recoveries))
            if private_recoveries
            else None
        ),
        "technical_subspace_recovery": technical_subspace[
            "mean_canonical_correlation"
        ],
        "true_factor_recovery_rate": biological_matching[
            "true_factor_recovery_rate"
        ],
        "estimated_factor_match_fraction": biological_matching[
            "estimated_factor_match_fraction"
        ],
        "false_shared_discovery_fraction": biological_matching[
            "false_shared_discovery_fraction"
        ],
        "mean_matched_absolute_correlation": biological_matching[
            "mean_matched_absolute_correlation"
        ],
        "minimum_matched_absolute_correlation": biological_matching[
            "minimum_matched_absolute_correlation"
        ],
        "modality_contribution_entropy": float(
            np.median(entropy)
        ),
        "view_support_jaccard": (
            float(np.mean(support_jaccards))
            if support_jaccards
            else None
        ),
        "view_support_exact_accuracy": (
            float(np.mean(support_exact))
            if support_exact
            else None
        ),
    }

    return metrics, factor_rows


# ---------------------------------------------------------------------
# Fold-local model evaluation
# ---------------------------------------------------------------------

def build_view_slices(
    block_views: dict[str, np.ndarray],
) -> dict[str, slice]:
    slices: dict[str, slice] = {}
    start = 0

    for view in VIEWS:
        end = start + block_views[view].shape[1]
        slices[view] = slice(start, end)
        start = end

    return slices


def block_contributions(
    pca: DualPCA,
    block_views: dict[str, np.ndarray],
    view_slices: dict[str, slice],
    rank: int,
) -> dict[str, np.ndarray]:
    k = min(rank, pca.components.shape[0])
    components = pca.components[:k, :]
    output: dict[str, np.ndarray] = {}

    for view in VIEWS:
        matrix_slice = view_slices[view]
        output[view] = (
            block_views[view]
            - pca.mean[matrix_slice]
        ) @ components[:, matrix_slice].T

    return output


def evaluate_fold(
    arrays: dict[str, np.ndarray],
    training_indices: np.ndarray,
    test_indices: np.ndarray,
    fold_index: int,
    ranks: list[int],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    preprocessors = fit_all_preprocessors(
        arrays,
        training_indices,
    )

    training_views = transformed_views(
        arrays,
        preprocessors,
        training_indices,
        complete=False,
    )
    test_views = transformed_views(
        arrays,
        preprocessors,
        test_indices,
        complete=False,
    )
    test_truth_views = transformed_views(
        arrays,
        preprocessors,
        test_indices,
        complete=True,
    )

    maximum_rank = max(ranks)

    per_view_pca = {
        view: fit_dual_pca(
            training_views[view],
            maximum_rank,
        )
        for view in VIEWS
    }

    block_scales = calculate_block_scales(
        training_views
    )
    block_training = scale_blocks(
        training_views,
        block_scales,
    )
    block_test = scale_blocks(
        test_views,
        block_scales,
    )
    view_slices = build_view_slices(block_training)

    block_training_concatenated = np.concatenate(
        [block_training[view] for view in VIEWS],
        axis=1,
    )
    block_test_concatenated = np.concatenate(
        [block_test[view] for view in VIEWS],
        axis=1,
    )
    block_pca = fit_dual_pca(
        block_training_concatenated,
        maximum_rank,
    )

    preprocessing_rows = []
    for view in VIEWS:
        preprocessor = preprocessors[view]
        preprocessing_rows.append(
            {
                "fold_index": fold_index,
                "view": view,
                "n_training_samples": int(
                    training_indices.shape[0]
                ),
                "n_test_samples": int(
                    test_indices.shape[0]
                ),
                "n_input_features": int(
                    arrays[f"X_{VIEW_PREFIX[view]}"].shape[1]
                ),
                "n_retained_features": int(
                    preprocessor.selected_indices.shape[0]
                ),
                "mean_training_missing_fraction": float(
                    np.mean(
                        preprocessor.training_missing_fraction
                    )
                ),
                "maximum_training_missing_fraction": float(
                    np.max(
                        preprocessor.training_missing_fraction
                    )
                ),
                "median_training_preimputation_variance": float(
                    np.median(
                        preprocessor.training_preimputation_variance
                    )
                ),
                "block_scale": block_scales[view],
            }
        )

    fold_rows: list[dict[str, Any]] = []

    for rank in ranks:
        # Single-view PCA baselines.
        for view in VIEWS:
            model_id = f"PCA_{view}"
            pca = per_view_pca[view]
            scores = pca.transform(
                test_views[view],
                rank,
            )
            reconstruction = pca.inverse_transform(
                scores,
                rank,
            )
            reconstruction_metrics = (
                normalized_reconstruction_metrics(
                    reconstruction,
                    test_truth_views[view],
                )
            )

            fold_rows.append(
                {
                    "fold_index": fold_index,
                    "model_id": model_id,
                    "model_family": "modality_specific_PCA",
                    "rank": rank,
                    "view": view,
                    "n_training_samples": int(
                        training_indices.shape[0]
                    ),
                    "n_test_samples": int(
                        test_indices.shape[0]
                    ),
                    "heldout_normalized_reconstruction_error": (
                        reconstruction_metrics["normalized_error"]
                    ),
                    "heldout_reconstruction_score": (
                        reconstruction_metrics["reconstruction_score"]
                    ),
                    "heldout_reconstruction_r2": (
                        reconstruction_metrics["r2"]
                    ),
                    "heldout_reconstruction_error_RNA": (
                        reconstruction_metrics["normalized_error"]
                        if view == "RNA"
                        else None
                    ),
                    "heldout_reconstruction_error_METHYLATION": (
                        reconstruction_metrics["normalized_error"]
                        if view == "METHYLATION"
                        else None
                    ),
                    "heldout_reconstruction_error_CNV": (
                        reconstruction_metrics["normalized_error"]
                        if view == "CNV"
                        else None
                    ),
                    "cross_view_mrr": None,
                    "random_expected_cross_view_mrr": None,
                    "random_adjusted_cross_view_mrr": None,
                    "cross_view_top1_accuracy": None,
                    "same_patient_similarity_margin": None,
                    "nominal_index_cross_view_mrr": None,
                    "n_valid_retrieval_queries": 0,
                    "missing_view_r2_RNA": None,
                    "missing_view_r2_METHYLATION": None,
                    "missing_view_r2_CNV": None,
                    "missing_view_reconstruction_score": None,
                    "modality_contribution_entropy": 0.0,
                    "fold_latent_effective_rank": effective_rank(scores),
                    "fit_status": "PASS",
                }
            )

        # Block-scaled PCA baseline.
        total_scores = block_pca.transform(
            block_test_concatenated,
            rank,
        )
        total_reconstruction = block_pca.inverse_transform(
            total_scores,
            rank,
        )

        view_reconstruction_metrics: dict[str, dict[str, Any]] = {}
        for view in VIEWS:
            reconstructed_scaled = total_reconstruction[
                :,
                view_slices[view],
            ]
            reconstructed_standardized = (
                reconstructed_scaled
                * block_scales[view]
            )
            view_reconstruction_metrics[view] = (
                normalized_reconstruction_metrics(
                    reconstructed_standardized,
                    test_truth_views[view],
                )
            )

        normalized_errors = [
            view_reconstruction_metrics[view][
                "normalized_error"
            ]
            for view in VIEWS
        ]
        mean_normalized_error = nanmean_or_none(
            normalized_errors
        )
        reconstruction_score = (
            float(
                np.clip(
                    1.0 - mean_normalized_error,
                    0.0,
                    1.0,
                )
            )
            if mean_normalized_error is not None
            else None
        )

        training_contributions = block_contributions(
            block_pca,
            block_training,
            view_slices,
            rank,
        )
        test_contributions = block_contributions(
            block_pca,
            block_test,
            view_slices,
            rank,
        )

        row_source = {
            view: arrays[
                f"row_source_{VIEW_PREFIX[view]}"
            ].astype(int)
            for view in VIEWS
        }

        retrieval = cross_view_retrieval_metrics(
            training_contributions,
            test_contributions,
            row_source,
            training_indices,
            test_indices,
        )

        missing_view_results: dict[str, float | None] = {}
        for omitted_view in VIEWS:
            remaining_scores = np.sum(
                [
                    test_contributions[view]
                    for view in VIEWS
                    if view != omitted_view
                ],
                axis=0,
            )
            component_slice = block_pca.components[
                : min(rank, block_pca.components.shape[0]),
                view_slices[omitted_view],
            ]
            reconstructed_scaled = (
                remaining_scores @ component_slice
                + block_pca.mean[
                    view_slices[omitted_view]
                ]
            )
            reconstructed_standardized = (
                reconstructed_scaled
                * block_scales[omitted_view]
            )
            metric = normalized_reconstruction_metrics(
                reconstructed_standardized,
                test_truth_views[omitted_view],
            )
            missing_view_results[omitted_view] = metric["r2"]

        missing_view_score = nanmean_or_none(
            np.clip(value, 0.0, 1.0)
            for value in missing_view_results.values()
            if value is not None
        )

        _, entropy = modality_loading_energy(
            block_pca.components[
                : min(rank, block_pca.components.shape[0]),
                :,
            ],
            view_slices,
        )

        fold_rows.append(
            {
                "fold_index": fold_index,
                "model_id": "BLOCK_PCA",
                "model_family": "block_scaled_PCA",
                "rank": rank,
                "view": "ALL3",
                "n_training_samples": int(
                    training_indices.shape[0]
                ),
                "n_test_samples": int(
                    test_indices.shape[0]
                ),
                "heldout_normalized_reconstruction_error": (
                    mean_normalized_error
                ),
                "heldout_reconstruction_score": reconstruction_score,
                "heldout_reconstruction_r2": (
                    None
                    if mean_normalized_error is None
                    else 1.0 - mean_normalized_error
                ),
                "heldout_reconstruction_error_RNA": (
                    view_reconstruction_metrics["RNA"][
                        "normalized_error"
                    ]
                ),
                "heldout_reconstruction_error_METHYLATION": (
                    view_reconstruction_metrics["METHYLATION"][
                        "normalized_error"
                    ]
                ),
                "heldout_reconstruction_error_CNV": (
                    view_reconstruction_metrics["CNV"][
                        "normalized_error"
                    ]
                ),
                "cross_view_mrr": retrieval["cross_view_mrr"],
                "random_expected_cross_view_mrr": retrieval[
                    "random_expected_cross_view_mrr"
                ],
                "random_adjusted_cross_view_mrr": retrieval[
                    "random_adjusted_cross_view_mrr"
                ],
                "cross_view_top1_accuracy": retrieval[
                    "cross_view_top1_accuracy"
                ],
                "same_patient_similarity_margin": retrieval[
                    "same_patient_similarity_margin"
                ],
                "nominal_index_cross_view_mrr": retrieval[
                    "nominal_index_cross_view_mrr"
                ],
                "n_valid_retrieval_queries": retrieval[
                    "n_valid_retrieval_queries"
                ],
                "missing_view_r2_RNA": missing_view_results["RNA"],
                "missing_view_r2_METHYLATION": (
                    missing_view_results["METHYLATION"]
                ),
                "missing_view_r2_CNV": missing_view_results["CNV"],
                "missing_view_reconstruction_score": (
                    missing_view_score
                ),
                "modality_contribution_entropy": float(
                    np.median(entropy)
                ),
                "fold_latent_effective_rank": effective_rank(
                    total_scores
                ),
                "fit_status": "PASS",
            }
        )

    return fold_rows, preprocessing_rows


# ---------------------------------------------------------------------
# Dataset-level evaluation
# ---------------------------------------------------------------------

def deterministic_folds(
    n_samples: int,
    scenario_id: str,
    replicate_index: int,
    simulation_seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    seed = stable_seed(
        SCRIPT_VERSION,
        "simulation_cv",
        scenario_id,
        replicate_index,
        simulation_seed,
    )
    splitter = KFold(
        n_splits=SIMULATION_CV_FOLDS,
        shuffle=True,
        random_state=seed,
    )

    indices = np.arange(n_samples, dtype=int)
    return [
        (
            training.astype(int),
            test.astype(int),
        )
        for training, test in splitter.split(indices)
    ]


def aggregate_fold_metrics(
    fold_rows: list[dict[str, Any]],
    ranks: list[int],
) -> list[dict[str, Any]]:
    table = pd.DataFrame(fold_rows)
    output: list[dict[str, Any]] = []

    metric_columns = [
        "heldout_normalized_reconstruction_error",
        "heldout_reconstruction_score",
        "heldout_reconstruction_r2",
        "heldout_reconstruction_error_RNA",
        "heldout_reconstruction_error_METHYLATION",
        "heldout_reconstruction_error_CNV",
        "cross_view_mrr",
        "random_expected_cross_view_mrr",
        "random_adjusted_cross_view_mrr",
        "cross_view_top1_accuracy",
        "same_patient_similarity_margin",
        "nominal_index_cross_view_mrr",
        "missing_view_r2_RNA",
        "missing_view_r2_METHYLATION",
        "missing_view_r2_CNV",
        "missing_view_reconstruction_score",
        "modality_contribution_entropy",
        "fold_latent_effective_rank",
    ]

    for (model_id, rank), part in table.groupby(
        ["model_id", "rank"],
        sort=True,
    ):
        row: dict[str, Any] = {
            "model_id": str(model_id),
            "model_family": str(
                part.iloc[0]["model_family"]
            ),
            "rank": int(rank),
            "view": str(part.iloc[0]["view"]),
            "n_folds": int(part.shape[0]),
            "all_fold_fits_passed": bool(
                part["fit_status"].eq("PASS").all()
            ),
            "total_heldout_samples": int(
                pd.to_numeric(
                    part["n_test_samples"],
                    errors="coerce",
                ).sum()
            ),
            "total_valid_retrieval_queries": int(
                pd.to_numeric(
                    part["n_valid_retrieval_queries"],
                    errors="coerce",
                ).fillna(0).sum()
            ),
        }

        for column in metric_columns:
            row[column] = nanmean_or_none(
                part[column].tolist()
            )

        output.append(row)

    expected = 4 * len(ranks)
    if len(output) != expected:
        raise RuntimeError(
            f"Expected {expected} aggregated linear model rows, "
            f"found {len(output)}."
        )

    return output


def full_data_truth_evaluation(
    arrays: dict[str, np.ndarray],
    ranks: list[int],
    threshold: float,
) -> tuple[
    dict[tuple[str, int], dict[str, Any]],
    list[dict[str, Any]],
]:
    all_indices = np.arange(
        arrays["sample_index"].shape[0],
        dtype=int,
    )
    preprocessors = fit_all_preprocessors(
        arrays,
        all_indices,
    )
    observed_views = transformed_views(
        arrays,
        preprocessors,
        all_indices,
        complete=False,
    )

    maximum_rank = max(ranks)
    truth_metrics: dict[
        tuple[str, int],
        dict[str, Any],
    ] = {}
    factor_rows: list[dict[str, Any]] = []

    for view in VIEWS:
        pca = fit_dual_pca(
            observed_views[view],
            maximum_rank,
        )

        for rank in ranks:
            metrics, matches = single_view_full_truth_metrics(
                arrays,
                view,
                pca,
                observed_views[view],
                rank,
                threshold,
            )
            truth_metrics[(f"PCA_{view}", rank)] = metrics

            for match in matches:
                factor_rows.append(
                    {
                        "model_id": f"PCA_{view}",
                        "model_family": "modality_specific_PCA",
                        "rank": rank,
                        "view": view,
                        **match,
                    }
                )

    block_scales = calculate_block_scales(observed_views)
    block_views = scale_blocks(
        observed_views,
        block_scales,
    )
    view_slices = build_view_slices(block_views)
    concatenated = np.concatenate(
        [block_views[view] for view in VIEWS],
        axis=1,
    )
    block_pca = fit_dual_pca(
        concatenated,
        maximum_rank,
    )

    for rank in ranks:
        metrics, matches = block_full_truth_metrics(
            arrays,
            block_pca,
            block_views,
            view_slices,
            rank,
            threshold,
        )
        truth_metrics[("BLOCK_PCA", rank)] = metrics

        for match in matches:
            factor_rows.append(
                {
                    "model_id": "BLOCK_PCA",
                    "model_family": "block_scaled_PCA",
                    "rank": rank,
                    "view": "ALL3",
                    **match,
                }
            )

    return truth_metrics, factor_rows


def apply_failure_gates(
    dataset_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for row in dataset_rows:
        model_id = str(row["model_id"])
        integrated = model_id == "BLOCK_PCA"

        essential_values = [
            row.get("heldout_normalized_reconstruction_error"),
            row.get("full_effective_rank"),
        ]
        nonfinite = any(
            value is None
            or not math.isfinite(float(value))
            for value in essential_values
        )

        gate_definitions = [
            (
                "G01_NONFINITE_OPTIMIZATION",
                True,
                nonfinite,
                "all fitted models",
            ),
            (
                "G02_LATENT_COLLAPSE",
                integrated,
                (
                    row.get("full_effective_rank") is not None
                    and float(row["full_effective_rank"]) < 2.0
                ),
                "integrated latent models",
            ),
            (
                "G03_RECONSTRUCTION_WORSE_THAN_NULL",
                True,
                (
                    row.get(
                        "heldout_normalized_reconstruction_error"
                    ) is not None
                    and float(
                        row[
                            "heldout_normalized_reconstruction_error"
                        ]
                    ) >= 1.0
                ),
                "all reconstructed models",
            ),
            (
                "G04_RETRIEVAL_NOT_ABOVE_RANDOM",
                integrated,
                (
                    row.get(
                        "random_adjusted_cross_view_mrr"
                    ) is not None
                    and float(
                        row[
                            "random_adjusted_cross_view_mrr"
                        ]
                    ) <= 0.0
                ),
                "integrated multi-view models",
            ),
            (
                "G05_SINGLE_VIEW_DOMINATION",
                integrated,
                (
                    row.get(
                        "modality_contribution_entropy"
                    ) is not None
                    and float(
                        row[
                            "modality_contribution_entropy"
                        ]
                    ) < MODALITY_DOMINATION_ENTROPY_THRESHOLD
                ),
                "integrated shared representations",
            ),
            (
                "G09_SIMULATION_FALSE_SHARED_DISCOVERY",
                integrated,
                (
                    row.get(
                        "false_shared_discovery_fraction"
                    ) is not None
                    and float(
                        row[
                            "false_shared_discovery_fraction"
                        ]
                    ) > 0.10
                ),
                "diagnostic only for ungated PCA",
            ),
        ]

        for gate_id, applicable, failed, scope in gate_definitions:
            output.append(
                {
                    "model_id": model_id,
                    "rank": int(row["rank"]),
                    "gate_id": gate_id,
                    "applicable": bool(applicable),
                    "failed": bool(failed) if applicable else False,
                    "assessment_scope": scope,
                    "consequence": (
                        "diagnostic_only_for_linear_baseline"
                        if gate_id
                        == "G09_SIMULATION_FALSE_SHARED_DISCOVERY"
                        else "report_failure_without_silent_omission"
                    ),
                }
            )

    return output


def evaluate_one_dataset(
    manifest_row: pd.Series,
    ranks: list[int],
    primary_rank: int,
    truth_match_threshold: float,
) -> dict[str, Any]:
    scenario_id = clean_text(
        manifest_row["scenario_id"]
    )
    replicate_index = int(
        float(manifest_row["replicate_index"])
    )
    simulation_seed = int(
        float(manifest_row["simulation_seed"])
    )
    dataset_path = resolve_project_path(
        manifest_row["path"]
    )

    require_file(dataset_path)

    expected_size = int(
        float(manifest_row["size_bytes"])
    )
    if dataset_path.stat().st_size != expected_size:
        raise RuntimeError(
            f"Simulation size mismatch: {relative_path(dataset_path)}"
        )

    observed_hash = sha256_file(dataset_path)
    expected_hash = clean_text(
        manifest_row["sha256"]
    )
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"Simulation SHA256 mismatch: {relative_path(dataset_path)}"
        )

    arrays = load_simulation_arrays(dataset_path)
    n_samples = int(arrays["sample_index"].shape[0])

    folds = deterministic_folds(
        n_samples,
        scenario_id,
        replicate_index,
        simulation_seed,
    )

    split_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    preprocessing_rows: list[dict[str, Any]] = []

    for fold_index, (training, test) in enumerate(
        folds,
        start=1,
    ):
        split_rows.append(
            {
                "fold_index": fold_index,
                "n_training_samples": int(training.shape[0]),
                "n_test_samples": int(test.shape[0]),
                "training_indices": ";".join(
                    str(value)
                    for value in training.tolist()
                ),
                "test_indices": ";".join(
                    str(value)
                    for value in test.tolist()
                ),
            }
        )

        fold_result, preprocessing_result = evaluate_fold(
            arrays,
            training,
            test,
            fold_index,
            ranks,
        )
        fold_rows.extend(fold_result)
        preprocessing_rows.extend(preprocessing_result)

    dataset_rows = aggregate_fold_metrics(
        fold_rows,
        ranks,
    )

    truth_metrics, factor_rows = full_data_truth_evaluation(
        arrays,
        ranks,
        truth_match_threshold,
    )

    base_metadata = {
        "registry_label": clean_text(
            manifest_row["registry_label"]
        ),
        "scenario_id": scenario_id,
        "scenario_origin": clean_text(
            manifest_row["scenario_origin"]
        ),
        "scenario_family": clean_text(
            manifest_row["scenario_family"]
        ),
        "calibration_role": clean_text(
            manifest_row["calibration_role"]
        ),
        "replicate_index": replicate_index,
        "simulation_seed": simulation_seed,
        "n_samples": n_samples,
        "n_true_biological_shared_factors": int(
            float(
                manifest_row[
                    "n_biological_shared_factors"
                ]
            )
        ),
        "n_true_private_factors_per_view": int(
            float(
                manifest_row[
                    "n_private_factors_per_view"
                ]
            )
        ),
        "n_true_technical_shared_factors": int(
            float(
                manifest_row[
                    "n_technical_shared_factors"
                ]
            )
        ),
        "factor_level_identifiable": parse_bool(
            manifest_row["factor_level_identifiable"]
        ),
        "subspace_level_identifiable": parse_bool(
            manifest_row["subspace_level_identifiable"]
        ),
        "primary_truth_target": clean_text(
            manifest_row["primary_truth_target"]
        ),
        "input_path": relative_path(dataset_path),
        "input_sha256": observed_hash,
    }

    for row in split_rows:
        row.update(base_metadata)

    for row in preprocessing_rows:
        row.update(base_metadata)

    for row in fold_rows:
        row.update(base_metadata)
        row["is_primary_rank"] = int(row["rank"]) == primary_rank

    for row in dataset_rows:
        truth = truth_metrics[
            (str(row["model_id"]), int(row["rank"]))
        ]
        row.update(truth)
        row.update(base_metadata)
        row["is_primary_rank"] = int(row["rank"]) == primary_rank

        # Preserve descriptive factor matching, but do not treat individual
        # factor recovery as a primary metric in rotationally ambiguous
        # non-null scenarios. Null-specificity remains evaluable even when
        # factor identities are undefined.
        row["factor_level_metrics_primary_eligible"] = bool(
            base_metadata["factor_level_identifiable"]
            or base_metadata["n_true_biological_shared_factors"] == 0
        )
        if (
            not base_metadata["factor_level_identifiable"]
            and base_metadata["n_true_biological_shared_factors"] > 0
        ):
            for metric_name in [
                "true_factor_recovery_rate",
                "estimated_factor_match_fraction",
                "false_shared_discovery_fraction",
                "mean_matched_absolute_correlation",
                "minimum_matched_absolute_correlation",
            ]:
                row[f"{metric_name}_descriptive"] = row.get(metric_name)
                row[metric_name] = None

        if row["model_id"] == "BLOCK_PCA":
            utility_components = [
                (0.40, row.get("shared_subspace_recovery")),
                (0.20, row.get("private_subspace_recovery")),
                (0.20, row.get("random_adjusted_cross_view_mrr")),
                (0.10, row.get("missing_view_reconstruction_score")),
                (0.10, row.get("modality_contribution_entropy")),
            ]
            if all(
                value is not None
                and math.isfinite(float(value))
                for _, value in utility_components
            ):
                row["simulation_utility"] = float(
                    sum(
                        weight
                        * float(
                            np.clip(value, 0.0, 1.0)
                        )
                        for weight, value in utility_components
                    )
                )
            else:
                row["simulation_utility"] = None
        else:
            row["simulation_utility"] = None

    for row in factor_rows:
        row.update(base_metadata)
        row["is_primary_rank"] = int(row["rank"]) == primary_rank

    failure_rows = apply_failure_gates(dataset_rows)
    for row in failure_rows:
        row.update(base_metadata)
        row["is_primary_rank"] = int(row["rank"]) == primary_rank

    return {
        "script_version": SCRIPT_VERSION,
        "input_path": relative_path(dataset_path),
        "input_sha256": observed_hash,
        "scenario_id": scenario_id,
        "replicate_index": replicate_index,
        "simulation_seed": simulation_seed,
        "split_rows": split_rows,
        "preprocessing_rows": preprocessing_rows,
        "fold_rows": fold_rows,
        "dataset_rows": dataset_rows,
        "factor_rows": factor_rows,
        "failure_rows": failure_rows,
    }


# ---------------------------------------------------------------------
# Shards and aggregation
# ---------------------------------------------------------------------

def shard_path_for_manifest_row(
    row: pd.Series,
) -> Path:
    dataset_path = Path(clean_text(row["path"]))
    return (
        OUTPUT_SHARD_DIR
        / clean_text(row["scenario_family"])
        / clean_text(row["scenario_id"])
        / (dataset_path.stem + ".json")
    )


def validate_existing_shard(
    shard: dict[str, Any],
    row: pd.Series,
    protocol_hash: str,
) -> None:
    if clean_text(shard.get("script_version")) != SCRIPT_VERSION:
        raise RuntimeError(
            "Existing shard script version mismatch."
        )
    if clean_text(shard.get("protocol_sha256")) != protocol_hash:
        raise RuntimeError(
            "Existing shard protocol hash mismatch."
        )
    if clean_text(shard.get("input_sha256")) != clean_text(
        row["sha256"]
    ):
        raise RuntimeError(
            "Existing shard input hash mismatch."
        )
    if clean_text(shard.get("scenario_id")) != clean_text(
        row["scenario_id"]
    ):
        raise RuntimeError(
            "Existing shard scenario mismatch."
        )
    if int(shard.get("replicate_index")) != int(
        float(row["replicate_index"])
    ):
        raise RuntimeError(
            "Existing shard replicate mismatch."
        )


def materialize_shard(
    row: pd.Series,
    ranks: list[int],
    primary_rank: int,
    truth_match_threshold: float,
    protocol_hash: str,
) -> tuple[dict[str, Any], str]:
    path = shard_path_for_manifest_row(row)

    if path.exists():
        shard = load_json(path)
        validate_existing_shard(
            shard,
            row,
            protocol_hash,
        )
        return shard, "verified_existing"

    result = evaluate_one_dataset(
        row,
        ranks,
        primary_rank,
        truth_match_threshold,
    )
    result["protocol_sha256"] = protocol_hash
    result["created_at_utc"] = utc_now_iso()

    write_json_atomic(result, path)

    serialized = load_json(path)
    validate_existing_shard(
        serialized,
        row,
        protocol_hash,
    )

    return serialized, "generated"


def aggregate_shards(
    shard_records: list[dict[str, Any]],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    split_rows: list[dict[str, Any]] = []
    preprocessing_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    factor_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for shard in shard_records:
        split_rows.extend(shard["split_rows"])
        preprocessing_rows.extend(shard["preprocessing_rows"])
        fold_rows.extend(shard["fold_rows"])
        dataset_rows.extend(shard["dataset_rows"])
        factor_rows.extend(shard["factor_rows"])
        failure_rows.extend(shard["failure_rows"])

    return (
        pd.DataFrame(split_rows),
        pd.DataFrame(preprocessing_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(dataset_rows),
        pd.DataFrame(factor_rows),
        pd.DataFrame(failure_rows),
    )


def summarize_scenarios(
    dataset_metrics: pd.DataFrame,
) -> pd.DataFrame:
    key_metrics = [
        "heldout_normalized_reconstruction_error",
        "heldout_reconstruction_score",
        "random_adjusted_cross_view_mrr",
        "cross_view_top1_accuracy",
        "missing_view_reconstruction_score",
        "modality_contribution_entropy",
        "shared_subspace_recovery",
        "private_subspace_recovery",
        "technical_subspace_recovery",
        "true_factor_recovery_rate",
        "false_shared_discovery_fraction",
        "view_support_jaccard",
        "simulation_utility",
    ]

    rows: list[dict[str, Any]] = []

    group_columns = [
        "scenario_id",
        "scenario_origin",
        "scenario_family",
        "calibration_role",
        "model_id",
        "model_family",
        "rank",
        "is_primary_rank",
    ]

    for keys, part in dataset_metrics.groupby(
        group_columns,
        dropna=False,
        sort=True,
    ):
        row = dict(zip(group_columns, keys))
        row["n_datasets"] = int(part.shape[0])
        row["n_all_fold_fits_passed"] = int(
            part["all_fold_fits_passed"].map(parse_bool).sum()
        )

        for metric in key_metrics:
            values = pd.to_numeric(
                part[metric],
                errors="coerce",
            ).to_numpy(dtype=float)
            finite = values[np.isfinite(values)]

            row[f"{metric}_n"] = int(finite.size)
            if finite.size == 0:
                row[f"{metric}_mean"] = None
                row[f"{metric}_sd"] = None
                row[f"{metric}_se"] = None
                row[f"{metric}_median"] = None
            else:
                row[f"{metric}_mean"] = float(
                    np.mean(finite)
                )
                row[f"{metric}_sd"] = (
                    float(np.std(finite, ddof=1))
                    if finite.size > 1
                    else 0.0
                )
                row[f"{metric}_se"] = (
                    float(
                        np.std(finite, ddof=1)
                        / math.sqrt(finite.size)
                    )
                    if finite.size > 1
                    else 0.0
                )
                row[f"{metric}_median"] = float(
                    np.median(finite)
                )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Reference fixture
# ---------------------------------------------------------------------

def run_reference_fixture() -> dict[str, Any]:
    rng = np.random.default_rng(17051)
    n = 40
    shared = rng.normal(size=(n, 3))

    arrays: dict[str, np.ndarray] = {
        "sample_index": np.arange(n, dtype=np.int32),
        "Z_biological_shared": shared.astype(np.float32),
        "biological_support_mask": np.ones(
            (3, 3),
            dtype=np.uint8,
        ),
        "biological_factor_ids": np.asarray(
            ["BIO_1", "BIO_2", "BIO_3"],
            dtype="U16",
        ),
        "biological_factor_truth_class": np.asarray(
            ["ALL3", "ALL3", "ALL3"],
            dtype="U16",
        ),
        "Z_technical": np.zeros((n, 0), dtype=np.float32),
        "technical_support_mask": np.zeros(
            (0, 3),
            dtype=np.uint8,
        ),
        "technical_factor_ids": np.asarray([], dtype="U16"),
        "factor_level_identifiable": np.asarray(True),
        "subspace_level_identifiable": np.asarray(True),
        "primary_truth_target": np.asarray("factor_and_subspace"),
    }

    feature_counts = {
        "RNA": 60,
        "METHYLATION": 90,
        "CNV": 50,
    }

    for view in VIEWS:
        prefix = VIEW_PREFIX[view]
        private = rng.normal(size=(n, 2))
        w_shared = rng.normal(
            scale=0.8,
            size=(feature_counts[view], 3),
        )
        w_private = rng.normal(
            scale=0.3,
            size=(feature_counts[view], 2),
        )
        complete = (
            shared @ w_shared.T
            + private @ w_private.T
            + rng.normal(
                scale=0.5,
                size=(n, feature_counts[view]),
            )
        ).astype(np.float32)
        observed = complete.copy()
        if view == "METHYLATION":
            mask = rng.random(observed.shape) < 0.05
            observed[mask] = np.nan

        arrays[f"X_{prefix}"] = observed
        arrays[f"X_complete_{prefix}"] = complete
        arrays[f"Z_private_{prefix}"] = private.astype(np.float32)
        arrays[f"row_source_{prefix}"] = np.arange(
            n,
            dtype=np.int32,
        )

    folds = deterministic_folds(
        n,
        "REFERENCE_FIXTURE",
        1,
        17051,
    )
    fold_rows, preprocessing_rows = evaluate_fold(
        arrays,
        folds[0][0],
        folds[0][1],
        1,
        [5, 10, 15],
    )

    block_rows = [
        row
        for row in fold_rows
        if row["model_id"] == "BLOCK_PCA"
        and int(row["rank"]) == 5
    ]
    if len(block_rows) != 1:
        raise RuntimeError(
            "Reference fixture did not produce the expected block-PCA row."
        )

    row = block_rows[0]
    required_finite = [
        row["heldout_normalized_reconstruction_error"],
        row["random_adjusted_cross_view_mrr"],
        row["modality_contribution_entropy"],
    ]
    if not all(
        value is not None
        and math.isfinite(float(value))
        for value in required_finite
    ):
        raise RuntimeError(
            "Reference fixture produced non-finite primary metrics."
        )

    preprocess_table = pd.DataFrame(preprocessing_rows)
    block_scales = preprocess_table.set_index("view")[
        "block_scale"
    ].astype(float)

    if not (block_scales > 0).all():
        raise RuntimeError(
            "Reference fixture produced invalid block scales."
        )

    return {
        "fixture_id": "05c1-linear-reference-v1",
        "n_samples": n,
        "feature_counts": feature_counts,
        "block_pca_rank5": {
            "heldout_normalized_reconstruction_error": row[
                "heldout_normalized_reconstruction_error"
            ],
            "random_adjusted_cross_view_mrr": row[
                "random_adjusted_cross_view_mrr"
            ],
            "modality_contribution_entropy": row[
                "modality_contribution_entropy"
            ],
        },
        "block_scales": {
            view: float(block_scales.loc[view])
            for view in VIEWS
        },
        "status": "PASS",
    }


# ---------------------------------------------------------------------
# Frozen-lock verification
# ---------------------------------------------------------------------

def verify_existing_lock() -> None:
    lock = load_json(OUTPUT_LOCK)

    failures: list[str] = []

    if clean_text(lock.get("script_version")) != SCRIPT_VERSION:
        failures.append("script version mismatch")

    generated_files = lock.get("generated_files", {})
    for relative_name, specification in generated_files.items():
        path = resolve_project_path(relative_name)
        if not path.exists():
            failures.append(f"missing output: {relative_name}")
            continue

        expected_size = int(specification["size_bytes"])
        observed_size = int(path.stat().st_size)
        if observed_size != expected_size:
            failures.append(
                f"size mismatch: {relative_name}"
            )
            continue

        expected_hash = clean_text(specification["sha256"])
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            failures.append(
                f"SHA256 mismatch: {relative_name}"
            )

    shard_manifest_text = clean_text(
        lock.get("shard_manifest", {}).get("path", "")
    )
    if not shard_manifest_text:
        failures.append("shard manifest path missing from lock")
        shard_manifest_path = Path("__missing__")
    else:
        shard_manifest_path = resolve_project_path(
            shard_manifest_text
        )

    if not shard_manifest_path.exists() or not shard_manifest_path.is_file():
        failures.append("shard manifest missing")
    else:
        shard_manifest = pd.read_csv(
            shard_manifest_path,
            dtype=str,
        ).fillna("")
        expected_shards = int(
            lock.get("scope", {}).get("n_datasets", 0)
        )
        if shard_manifest.shape[0] != expected_shards:
            failures.append(
                "shard manifest row-count mismatch"
            )
        else:
            for _, row in shard_manifest.iterrows():
                path = resolve_project_path(row["shard_path"])
                if not path.exists():
                    failures.append(
                        f"missing shard: {row['shard_path']}"
                    )
                    break
                if int(path.stat().st_size) != int(
                    float(row["size_bytes"])
                ):
                    failures.append(
                        f"shard size mismatch: {row['shard_path']}"
                    )
                    break
                if sha256_file(path) != clean_text(row["sha256"]):
                    failures.append(
                        f"shard SHA256 mismatch: {row['shard_path']}"
                    )
                    break

    if failures:
        raise RuntimeError(
            "Existing 05c1 lock verification failed:\n"
            + "\n".join(
                f"- {failure}"
                for failure in failures
            )
        )

    print("=" * 88)
    print("Existing Paper 5 linear simulation benchmark: PASS")
    print("=" * 88)
    print(
        f"Lock: {relative_path(OUTPUT_LOCK)}"
    )
    print("=" * 88)


# ---------------------------------------------------------------------
# README and main
# ---------------------------------------------------------------------

def write_readme(
    n_datasets: int,
    n_scenarios: int,
    ranks: list[int],
    primary_rank: int,
) -> None:
    lines = [
        "Paper 5 linear simulation baseline benchmark",
        "=" * 72,
        "",
        f"Script version: {SCRIPT_VERSION}",
        "",
        "Scope",
        "-----",
        f"Confirmatory architecture-screening datasets: {n_datasets}",
        f"Scenarios: {n_scenarios}",
        f"Latent rank sensitivities: {ranks}",
        f"Primary linear rank: {primary_rank}",
        "",
        "Models",
        "------",
        "PCA_RNA",
        "PCA_METHYLATION",
        "PCA_CNV",
        "BLOCK_PCA with equal total training variance per modality",
        "",
        "Evaluation",
        "----------",
        "Five-fold deterministic held-out reconstruction.",
        "Full-data simulation-truth recovery is reported separately.",
        "Cross-view retrieval is scored against true row_source pairing.",
        "Complete simulation truth is used only for evaluation, never preprocessing.",
        "",
        "Scientific state",
        "----------------",
        "Outcome access: false",
        "TARGET model fitting: false",
        "MOFA fitting: false",
        "Neural fitting: false",
        "Architecture selection: false",
        "Final gate-calibration simulations: not used",
        "",
        "Interpretation",
        "--------------",
        (
            "Linear baselines are reported whether they perform well or poorly. "
            "No prespecified model family is silently omitted."
        ),
        (
            "The false-shared-discovery gate is diagnostic for ungated PCA; "
            "it is not used to hide or remove the linear baseline."
        ),
        "",
    ]

    OUTPUT_README.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 88)
    print("Paper 5 - fit linear simulation baselines")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Results directory: {OUTPUT_RESULTS_DIR}")
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

    upstream = validate_upstream_state()

    print("05a AI method contract: PASS")
    print("05a1 simulation/confound addendum: PASS")
    print("04d technical metadata preflight: PASS")
    print("05b v2 confirmatory simulation lock: PASS")
    print("Outcome access: FORBIDDEN")
    print("TARGET model fitting in this stage: NO")
    print("MOFA/neural fitting in this stage: NO")
    print()

    environment = environment_snapshot()
    if not environment["linear_dependencies_ready"]:
        raise RuntimeError(
            "NumPy, pandas, SciPy and scikit-learn are required."
        )

    protocol, protocol_hash = initialize_or_verify_protocol(
        upstream
    )

    print("Prospective 05c1 evaluation protocol: FROZEN")
    print(f"Latent ranks: {upstream['ranks']}")
    print(f"Primary linear rank: {upstream['primary_rank']}")
    print(f"Simulation CV folds: {SIMULATION_CV_FOLDS}")
    print(
        "Naive unscaled multi-omics concatenation: FORBIDDEN"
    )
    print(
        "Complete simulation truth used for preprocessing: NO"
    )
    print()

    print("Running deterministic linear-baseline reference fixture...")
    fixture = run_reference_fixture()
    write_json_atomic(fixture, OUTPUT_REFERENCE_FIXTURE)
    print("Linear-baseline reference fixture: PASS")
    print()

    write_json_atomic(environment, OUTPUT_ENVIRONMENT)

    manifest = upstream["manifest"].copy()
    manifest = manifest.sort_values(
        ["scenario_id", "replicate_index"],
        kind="mergesort",
    ).reset_index(drop=True)

    in_progress = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "protocol_sha256": protocol_hash,
        "n_datasets": int(manifest.shape[0]),
        "status": "IN_PROGRESS",
    }
    write_json_atomic(in_progress, OUTPUT_IN_PROGRESS)

    shard_records: list[dict[str, Any]] = []
    shard_manifest_rows: list[dict[str, Any]] = []
    total = int(manifest.shape[0])

    for index, row in manifest.iterrows():
        shard, generation_status = materialize_shard(
            row,
            upstream["ranks"],
            upstream["primary_rank"],
            upstream["truth_match_threshold"],
            protocol_hash,
        )
        shard_records.append(shard)

        shard_path = shard_path_for_manifest_row(row)
        shard_manifest_rows.append(
            {
                "registry_label": clean_text(row["registry_label"]),
                "scenario_id": clean_text(row["scenario_id"]),
                "scenario_family": clean_text(row["scenario_family"]),
                "replicate_index": int(
                    float(row["replicate_index"])
                ),
                "simulation_seed": int(
                    float(row["simulation_seed"])
                ),
                "input_path": clean_text(row["path"]),
                "input_sha256": clean_text(row["sha256"]),
                "shard_path": relative_path(shard_path),
                "size_bytes": int(shard_path.stat().st_size),
                "sha256": sha256_file(shard_path),
                "generation_status": generation_status,
            }
        )

        current = index + 1
        if (
            current == 1
            or current % 10 == 0
            or current == total
        ):
            print(
                f" {current:04d}/{total:04d} "
                f"{clean_text(row['scenario_id'])} "
                f"R{int(float(row['replicate_index'])):03d}: "
                f"{generation_status}"
            )

    shard_manifest = pd.DataFrame(shard_manifest_rows)
    shard_manifest.to_csv(
        OUTPUT_SHARD_MANIFEST,
        index=False,
    )

    (
        split_table,
        preprocessing_table,
        fold_table,
        dataset_table,
        factor_table,
        failure_table,
    ) = aggregate_shards(shard_records)

    expected_dataset_rows = (
        total
        * 4
        * len(upstream["ranks"])
    )
    if dataset_table.shape[0] != expected_dataset_rows:
        raise RuntimeError(
            f"Expected {expected_dataset_rows} dataset-level model rows, "
            f"found {dataset_table.shape[0]}."
        )

    write_csv_gzip(split_table, OUTPUT_SPLITS)
    write_csv_gzip(
        preprocessing_table,
        OUTPUT_PREPROCESSING,
    )
    write_csv_gzip(fold_table, OUTPUT_FOLD_METRICS)
    write_csv_gzip(
        dataset_table,
        OUTPUT_DATASET_METRICS,
    )
    write_csv_gzip(
        factor_table,
        OUTPUT_FACTOR_AUDIT,
    )
    write_csv_gzip(
        failure_table,
        OUTPUT_FAILURE_AUDIT,
    )

    scenario_summary = summarize_scenarios(
        dataset_table
    )
    scenario_summary.to_csv(
        OUTPUT_SCENARIO_SUMMARY,
        index=False,
        float_format="%.12g",
    )

    n_scenarios = int(
        manifest["scenario_id"].nunique()
    )
    write_readme(
        total,
        n_scenarios,
        upstream["ranks"],
        upstream["primary_rank"],
    )

    output_files = [
        OUTPUT_PROTOCOL,
        OUTPUT_REFERENCE_FIXTURE,
        OUTPUT_ENVIRONMENT,
        OUTPUT_SHARD_MANIFEST,
        OUTPUT_SPLITS,
        OUTPUT_PREPROCESSING,
        OUTPUT_FOLD_METRICS,
        OUTPUT_DATASET_METRICS,
        OUTPUT_FACTOR_AUDIT,
        OUTPUT_FAILURE_AUDIT,
        OUTPUT_SCENARIO_SUMMARY,
        OUTPUT_README,
    ]

    result_manifest = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "scientific_state": {
            "outcomes_loaded": False,
            "clinical_labels_loaded": False,
            "biological_annotations_loaded": False,
            "TARGET_model_fitting_performed": False,
            "MOFA_model_fitting_performed": False,
            "neural_model_fitting_performed": False,
            "architecture_selected": False,
            "stability_gate_thresholds_retuned": False,
            "final_gate_calibration_data_used": False,
        },
        "scope": {
            "n_scenarios": n_scenarios,
            "n_datasets": total,
            "n_models_per_rank": 4,
            "ranks": upstream["ranks"],
            "primary_rank": upstream["primary_rank"],
            "n_dataset_model_rows": int(
                dataset_table.shape[0]
            ),
            "n_fold_rows": int(fold_table.shape[0]),
            "n_factor_audit_rows": int(
                factor_table.shape[0]
            ),
        },
        "upstream_hashes": upstream["upstream_hashes"],
        "protocol": {
            "path": relative_path(OUTPUT_PROTOCOL),
            "sha256": protocol_hash,
        },
        "outputs": {
            relative_path(path): {
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
            for path in output_files
        },
        "shard_manifest": {
            "path": relative_path(
                OUTPUT_SHARD_MANIFEST
            ),
            "sha256": sha256_file(
                OUTPUT_SHARD_MANIFEST
            ),
            "n_rows": int(shard_manifest.shape[0]),
        },
        "guardrails": [
            "Only frozen sable_mo_v2 architecture-screening simulations were used.",
            "The disclosed sable_mo_v1 pilot was not used.",
            "Final gate-calibration simulations were not accessed.",
            "All preprocessing was fitted within the training fold for held-out metrics.",
            "Complete simulation truth was used for evaluation only.",
            "No outcome, clinical variable or biological annotation was accessed.",
            "All prespecified linear baselines are retained regardless of performance.",
            "No architecture or stability threshold was selected or changed.",
        ],
    }
    write_json_atomic(
        result_manifest,
        OUTPUT_RESULT_MANIFEST,
    )

    generated_files = {
        relative_path(path): {
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for path in [
            *output_files,
            OUTPUT_RESULT_MANIFEST,
        ]
    }

    lock = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": (
            "FROZEN_LINEAR_BASELINE_RESULTS_BEFORE_MOFA_AND_NEURAL_FITTING"
        ),
        "scientific_state": result_manifest[
            "scientific_state"
        ],
        "scope": result_manifest["scope"],
        "result_manifest": {
            "path": relative_path(
                OUTPUT_RESULT_MANIFEST
            ),
            "sha256": sha256_file(
                OUTPUT_RESULT_MANIFEST
            ),
        },
        "shard_manifest": result_manifest[
            "shard_manifest"
        ],
        "generated_files": generated_files,
        "rewrite_policy": (
            "This zero-argument script never rewrites a frozen 05c1 result. "
            "Any scientifically justified change requires a new versioned stage."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    if OUTPUT_IN_PROGRESS.exists():
        OUTPUT_IN_PROGRESS.unlink()

    primary_summary = scenario_summary[
        scenario_summary["is_primary_rank"].map(parse_bool)
    ]

    print()
    print("-" * 88)
    print("Linear simulation baseline summary")
    print("-" * 88)
    print(f"Scenarios: {n_scenarios}")
    print(f"Datasets: {total}")
    print(
        "Dataset × model × rank rows: "
        f"{dataset_table.shape[0]}"
    )
    print(f"Fold-level rows: {fold_table.shape[0]}")
    print(
        "Primary-rank scenario summary rows: "
        f"{primary_summary.shape[0]}"
    )
    print("Architecture selected: NO")
    print("MOFA fitted: NO")
    print("Neural models fitted: NO")
    print("Final gate-calibration data accessed: NO")

    print()
    print("=" * 88)
    print("Linear simulation baseline benchmark: PASS")
    print("=" * 88)
    print(
        f"Results manifest: {relative_path(OUTPUT_RESULT_MANIFEST)}"
    )
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("=" * 88)


if __name__ == "__main__":
    main()
