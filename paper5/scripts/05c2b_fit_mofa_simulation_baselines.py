from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import sys
import tempfile
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05c2b-fit-mofa-simulation-baselines-v1-no-cli"
CONTRACT_VERSION = "paper5-mofa-simulation-baseline-benchmark-v1"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

VIEWS = ("RNA", "METHYLATION", "CNV")
VIEW_PREFIX = {"RNA": "rna", "METHYLATION": "methylation", "CNV": "cnv"}
VIEW_INDEX = {"RNA": 0, "METHYLATION": 1, "CNV": 2}
MOFA_RANKS = (5, 10, 15, 20)
EXPECTED_DATASETS = 720
CV_FOLDS = 5
EXPECTED_FITS = EXPECTED_DATASETS * len(MOFA_RANKS) * (CV_FOLDS + 1)
EXPECTED_SHARDS = EXPECTED_DATASETS * len(MOFA_RANKS)
VIEW_SUPPORT_ENERGY_THRESHOLD = 0.15
MODALITY_DOMINATION_ENTROPY_THRESHOLD = 0.60

# Frozen upstream artifacts.
CONTRACT_05A1 = PAPER5_ROOT / "method_contract" / "05a1" / "simulation_and_technical_confound_addendum.json"
SIMULATION_MANIFEST = PAPER5_ROOT / "method_contract" / "05b_v2" / "architecture_screening_data_manifest.csv"
LINEAR_RESULTS_DIR = PAPER5_ROOT / "results" / "simulation_benchmark" / "05c1_linear_baselines"
LINEAR_SPLITS = LINEAR_RESULTS_DIR / "linear_simulation_split_registry.csv.gz"
LINEAR_DATASET_METRICS = LINEAR_RESULTS_DIR / "linear_dataset_metrics.csv.gz"
LINEAR_SCRIPT = PAPER5_ROOT / "scripts" / "05c1_fit_linear_simulation_baselines.py"

MOFA_PROTOCOL_DIR = PAPER5_ROOT / "method_contract" / "05c2a"
MOFA_PROTOCOL = MOFA_PROTOCOL_DIR / "mofa_out_of_sample_projection_protocol.json"
MOFA_PROTOCOL_LOCK = MOFA_PROTOCOL_DIR / "mofa_out_of_sample_projection_lock.json"
MOFA_SEEDS = MOFA_PROTOCOL_DIR / "mofa_simulation_fit_seed_registry.csv"
MOFA_PROJECTION_FIXTURE = MOFA_PROTOCOL_DIR / "mofa_gaussian_projection_reference_fixture.json"

FINAL_GATE_DATA_ROOT = PAPER5_ROOT / "simulations" / "sable_mo_v2" / "final_gate_calibration"
FUTURE_PATHS = (
    PAPER5_ROOT / "method_contract" / "05d",
    PAPER5_ROOT / "method_contract" / "05e",
    PAPER5_ROOT / "results" / "simulation_benchmark" / "05d_sable_mo",
    PAPER5_ROOT / "results" / "representation_benchmark",
)

# Outputs.
OUTPUT_CONTRACT_DIR = PAPER5_ROOT / "method_contract" / "05c2b"
OUTPUT_RESULTS_DIR = PAPER5_ROOT / "results" / "simulation_benchmark" / "05c2_mofa"
OUTPUT_SHARD_DIR = OUTPUT_RESULTS_DIR / "shards"

OUTPUT_PROTOCOL = OUTPUT_CONTRACT_DIR / "mofa_simulation_evaluation_protocol.json"
OUTPUT_REFERENCE_FIXTURE = OUTPUT_CONTRACT_DIR / "mofa_baseline_reference_fixture.json"
OUTPUT_ENVIRONMENT = OUTPUT_CONTRACT_DIR / "mofa_baseline_environment_snapshot.json"
OUTPUT_SHARD_MANIFEST = OUTPUT_CONTRACT_DIR / "mofa_baseline_shard_manifest.csv"
OUTPUT_RESULTS_MANIFEST = OUTPUT_CONTRACT_DIR / "mofa_baseline_results_manifest.json"
OUTPUT_README = OUTPUT_CONTRACT_DIR / "mofa_simulation_baseline_README.txt"
OUTPUT_LOCK = OUTPUT_CONTRACT_DIR / "mofa_simulation_baseline_lock.json"
OUTPUT_IN_PROGRESS = OUTPUT_CONTRACT_DIR / ".mofa_simulation_baseline_in_progress.json"

OUTPUT_PREPROCESSING = OUTPUT_RESULTS_DIR / "mofa_preprocessing_audit.csv.gz"
OUTPUT_FIT_AUDIT = OUTPUT_RESULTS_DIR / "mofa_fit_audit.csv.gz"
OUTPUT_FOLD_METRICS = OUTPUT_RESULTS_DIR / "mofa_fold_metrics.csv.gz"
OUTPUT_DATASET_METRICS = OUTPUT_RESULTS_DIR / "mofa_dataset_metrics.csv.gz"
OUTPUT_FACTOR_AUDIT = OUTPUT_RESULTS_DIR / "mofa_factor_truth_match_audit.csv.gz"
OUTPUT_FAILURE_AUDIT = OUTPUT_RESULTS_DIR / "mofa_failure_gate_audit.csv.gz"
OUTPUT_SCENARIO_SUMMARY = OUTPUT_RESULTS_DIR / "mofa_scenario_summary.csv"
OUTPUT_PAIRED_LINEAR = OUTPUT_RESULTS_DIR / "mofa_vs_block_pca_same_rank_dataset_comparison.csv.gz"
OUTPUT_PAIRED_LINEAR_SUMMARY = OUTPUT_RESULTS_DIR / "mofa_vs_block_pca_same_rank_scenario_summary.csv"


@dataclass(frozen=True)
class MofaViewPreprocessor:
    view: str
    selected_indices: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    training_missing_fraction: np.ndarray
    training_variance: np.ndarray
    n_eligible_before_budget: int
    feature_budget: int

    def transform_observed(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix[:, self.selected_indices], dtype=np.float64)
        values = (values - self.means) / self.scales
        if not (np.isfinite(values) | np.isnan(values)).all():
            raise RuntimeError(f"{self.view}: invalid transformed values.")
        return values

    def transform_complete(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix[:, self.selected_indices], dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(f"{self.view}: non-finite complete truth.")
        values = (values - self.means) / self.scales
        if not np.isfinite(values).all():
            raise RuntimeError(f"{self.view}: non-finite transformed complete truth.")
        return values


@dataclass
class MofaFit:
    rank: int
    seed: int
    weights: dict[str, np.ndarray]
    tau: dict[str, np.ndarray]
    training_factors: np.ndarray
    audit: dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na"}:
        return ""
    return text


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return clean_text(value).lower() in {"true", "1", "yes"}


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_project_path(value: Any) -> Path:
    path = Path(clean_text(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.stem + ".", suffix=".tmp.json", dir=str(path.parent)
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


def write_csv_gzip(table: pd.DataFrame, path: Path) -> None:
    compression: Any = {"method": "gzip", "compresslevel": 6, "mtime": 0}
    try:
        table.to_csv(
            path, index=False, float_format="%.12g", compression=compression
        )
    except TypeError:
        table.to_csv(path, index=False, float_format="%.12g", compression="gzip")


def load_json(path: Path) -> dict[str, Any]:
    require_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def package_version(name: str) -> dict[str, Any]:
    try:
        return {"installed": True, "version": importlib.metadata.version(name)}
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None}


def environment_snapshot() -> dict[str, Any]:
    packages = {
        name: package_version(name)
        for name in ["numpy", "pandas", "scipy", "scikit-learn", "mofapy2"]
    }
    return {
        "created_at_utc": utc_now_iso(),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": packages,
        "ready": all(record["installed"] for record in packages.values()),
    }


def nanmean_or_none(values: Any) -> float | None:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(float)
    finite = np.isfinite(array)
    return float(np.mean(array[finite])) if finite.any() else None


def is_nonempty_path(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.stat().st_size > 0
    return any(item.is_file() for item in path.rglob("*"))


def load_linear_helpers() -> Any:
    require_file(LINEAR_SCRIPT)
    spec = importlib.util.spec_from_file_location("paper5_05c1_helpers", LINEAR_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load 05c1 metric helpers.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_05c2a_state() -> dict[str, Any]:
    for path in [
        CONTRACT_05A1,
        SIMULATION_MANIFEST,
        LINEAR_SPLITS,
        LINEAR_DATASET_METRICS,
        LINEAR_SCRIPT,
        MOFA_PROTOCOL,
        MOFA_PROTOCOL_LOCK,
        MOFA_SEEDS,
        MOFA_PROJECTION_FIXTURE,
    ]:
        require_file(path)

    protocol = load_json(MOFA_PROTOCOL)
    lock = load_json(MOFA_PROTOCOL_LOCK)

    if clean_text(lock.get("status")) != (
        "FROZEN_MOFA_OOS_PROJECTION_PROTOCOL_BEFORE_MOFA_FITTING"
    ):
        raise RuntimeError("Unexpected 05c2a lock status.")

    if sha256_file(MOFA_PROTOCOL) != clean_text(lock["protocol"]["sha256"]):
        raise RuntimeError("05c2a protocol hash mismatch.")

    for relative_name, record in lock.get("generated_files", {}).items():
        candidate = resolve_project_path(relative_name)
        require_file(candidate)
        if int(candidate.stat().st_size) != int(record["size_bytes"]):
            raise RuntimeError(f"05c2a generated-file size mismatch: {relative_name}")
        if sha256_file(candidate) != clean_text(record["sha256"]):
            raise RuntimeError(f"05c2a generated-file hash mismatch: {relative_name}")

    for relative_name, expected_hash in protocol.get("upstream_hashes", {}).items():
        candidate = resolve_project_path(relative_name)
        require_file(candidate)
        if sha256_file(candidate) != clean_text(expected_hash):
            raise RuntimeError(f"05c2a upstream hash mismatch: {relative_name}")

    state = protocol.get("scientific_state", {})
    forbidden = [
        "outcomes_loaded",
        "clinical_labels_loaded",
        "biological_annotations_loaded",
        "TARGET_model_fitting_performed",
        "MOFA_model_fitting_performed",
        "neural_model_fitting_performed",
        "architecture_selected",
        "stability_gate_thresholds_retuned",
        "final_gate_calibration_data_used",
    ]
    active = [key for key in forbidden if bool(state.get(key, False))]
    if active:
        raise RuntimeError("05c2a pre-fit state violated: " + ", ".join(active))

    if protocol["out_of_sample_projection"]["raw_X_times_W_projection_allowed"]:
        raise RuntimeError("05c2a no longer forbids raw X @ W projection.")
    if protocol["out_of_sample_projection"]["test_sample_refitting"]:
        raise RuntimeError("05c2a no longer forbids test-sample refitting.")

    if sha256_file(LINEAR_SPLITS) != clean_text(
        protocol["paired_comparison_with_05c1"]["split_registry_sha256"]
    ):
        raise RuntimeError("05c1 split hash differs from 05c2a.")
    if sha256_file(MOFA_SEEDS) != clean_text(lock["fit_seed_registry"]["sha256"]):
        raise RuntimeError("05c2a seed registry hash mismatch.")

    environment = environment_snapshot()
    if not environment["ready"]:
        raise RuntimeError("Active environment is not ready for MOFA.")

    frozen_version = clean_text(
        protocol["mofapy2_configuration"]["mofapy2_version"]
    )
    observed_version = clean_text(environment["packages"]["mofapy2"]["version"])
    if observed_version != frozen_version:
        raise RuntimeError(
            f"mofapy2 version mismatch: observed={observed_version}, frozen={frozen_version}"
        )

    if FINAL_GATE_DATA_ROOT.exists() and any(
        item.is_file() for item in FINAL_GATE_DATA_ROOT.rglob("*")
    ):
        raise RuntimeError("Final gate-calibration data were materialized too early.")

    for future in FUTURE_PATHS:
        if is_nonempty_path(future):
            raise RuntimeError(f"Future result stage already exists: {relative_path(future)}")

    manifest = pd.read_csv(SIMULATION_MANIFEST, dtype=str, low_memory=False).fillna("")
    splits = pd.read_csv(LINEAR_SPLITS, dtype=str, low_memory=False).fillna("")
    seeds = pd.read_csv(MOFA_SEEDS, dtype=str, low_memory=False).fillna("")

    if manifest.shape[0] != EXPECTED_DATASETS:
        raise RuntimeError(f"Expected {EXPECTED_DATASETS} simulations.")
    if splits.shape[0] != EXPECTED_DATASETS * CV_FOLDS:
        raise RuntimeError("Frozen split registry row count mismatch.")
    if seeds.shape[0] != EXPECTED_FITS:
        raise RuntimeError(f"Expected {EXPECTED_FITS} frozen MOFA fit seeds.")

    addendum = load_json(CONTRACT_05A1)
    truth_threshold = float(
        addendum.get("truth_matching", {}).get(
            "factor_truth_match_absolute_correlation", float("nan")
        )
    )
    if not math.isfinite(truth_threshold):
        raise RuntimeError("05a1 truth-match threshold is unavailable.")

    return {
        "protocol": protocol,
        "lock": lock,
        "environment": environment,
        "manifest": manifest,
        "splits": splits,
        "seeds": seeds,
        "truth_threshold": truth_threshold,
        "protocol_sha256": sha256_file(MOFA_PROTOCOL),
        "split_sha256": sha256_file(LINEAR_SPLITS),
        "seed_sha256": sha256_file(MOFA_SEEDS),
        "linear_script_sha256": sha256_file(LINEAR_SCRIPT),
    }


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
            "MOFA_simulation_model_fitting_performed_before_protocol_freeze": False,
            "neural_model_fitting_performed": False,
            "architecture_selected": False,
            "stability_gate_thresholds_retuned": False,
            "final_gate_calibration_data_used": False,
        },
        "scope": {
            "n_datasets": EXPECTED_DATASETS,
            "ranks": list(MOFA_RANKS),
            "heldout_folds_per_dataset": CV_FOLDS,
            "full_data_truth_fit_per_dataset_and_rank": True,
            "planned_total_mofa_fits": EXPECTED_FITS,
            "planned_rank_shards": EXPECTED_SHARDS,
            "model_files_saved": False,
            "architecture_selection_performed_here": False,
        },
        "frozen_05c2a": {
            "path": relative_path(MOFA_PROTOCOL),
            "sha256": upstream["protocol_sha256"],
        },
        "exact_05c1_split_reuse": {
            "path": relative_path(LINEAR_SPLITS),
            "sha256": upstream["split_sha256"],
        },
        "fit_seed_registry": {
            "path": relative_path(MOFA_SEEDS),
            "sha256": upstream["seed_sha256"],
        },
        "metric_implementation": {
            "05c1_helper_source": relative_path(LINEAR_SCRIPT),
            "05c1_helper_source_sha256": upstream["linear_script_sha256"],
            "reuse": [
                "simulation loading",
                "reconstruction metric",
                "cross-view retrieval",
                "subspace recovery",
                "factor matching",
                "truth alignment",
                "effective rank",
                "support Jaccard",
            ],
        },
        "fold_local_preprocessing": upstream["protocol"]["fold_local_preprocessing"],
        "mofapy2_configuration": upstream["protocol"]["mofapy2_configuration"],
        "out_of_sample_projection": upstream["protocol"]["out_of_sample_projection"],
        "truth_evaluation": {
            "factor_truth_match_absolute_correlation": upstream["truth_threshold"],
            "integrated_truth_anchor": "RNA row_source, identical to 05c1 BLOCK_PCA",
            "private_metric_semantics": (
                "view-specific private-signal capture from single-view posterior "
                "projections; MOFA is not described as having an explicit private block"
            ),
            "rotationally_ambiguous_factor_metrics_primary": False,
            "subspace_metrics_retained_under_rotational_ambiguity": True,
        },
        "failure_policy": {
            "failed_fits_retained": True,
            "partial_fold_success_not_used_as_primary_dataset_metric": True,
            "G09_false_shared_discovery": "diagnostic_only_for_probabilistic_baseline",
            "silent_omission_forbidden": True,
        },
        "comparison_policy": {
            "same_rank_BLOCK_PCA": "descriptive paired deltas for K=5,10,15 only",
            "MOFA_K20_linear_same_rank_comparator": "none",
            "model_selection_performed_here": False,
        },
        "prohibited_operations": [
            "Use of sable_mo_v1 pilot simulations.",
            "Access to final gate-calibration simulations.",
            "Changing frozen MOFA ranks or training settings.",
            "Held-out preprocessing.",
            "Pre-MOFA imputation.",
            "Raw X @ W held-out projection.",
            "Held-out MOFA refitting.",
            "Outcome, clinical, or biological-label model selection.",
            "Architecture selection in 05c2b.",
        ],
    }


def protocol_core(protocol: dict[str, Any]) -> dict[str, Any]:
    result = dict(protocol)
    result.pop("created_at_utc", None)
    return result


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
                "Existing 05c2b protocol differs from current script. "
                "A new versioned stage is required."
            )
        protocol = existing
    else:
        write_json_atomic(proposed, OUTPUT_PROTOCOL)
        protocol = proposed
    return protocol, sha256_file(OUTPUT_PROTOCOL)


# ---------------------------------------------------------------------
# Fold-local preprocessing
# ---------------------------------------------------------------------

def fit_view_preprocessor(
    view: str,
    training_matrix: np.ndarray,
    feature_budget: int,
    contract: dict[str, Any],
) -> MofaViewPreprocessor:
    values = np.asarray(training_matrix, dtype=np.float64)
    finite = np.isfinite(values)
    n_finite = finite.sum(axis=0)
    missing_fraction = 1.0 - n_finite / values.shape[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        variance = np.nanvar(values, axis=0, ddof=1)

    eligible = (
        (n_finite >= int(contract["minimum_finite_training_values"]))
        & (
            missing_fraction
            <= float(contract["maximum_feature_missing_fraction"]) + 1e-12
        )
        & np.isfinite(variance)
        & (variance > float(contract["minimum_prestandardization_variance"]))
    )
    eligible_indices = np.flatnonzero(eligible)
    if eligible_indices.size == 0:
        raise RuntimeError(f"{view}: no eligible features.")

    # Variance descending, original index ascending.
    order = np.lexsort((eligible_indices, -variance[eligible_indices]))
    selected_indices = eligible_indices[
        order[: min(feature_budget, eligible_indices.size)]
    ]

    selected = values[:, selected_indices]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        means = np.nanmean(selected, axis=0)
        scales = np.nanstd(selected, axis=0, ddof=1)

    numerical_keep = (
        np.isfinite(means)
        & np.isfinite(scales)
        & (scales > float(contract["minimum_training_standard_deviation"]))
    )
    if not numerical_keep.any():
        raise RuntimeError(f"{view}: no features passed numerical checks.")

    selected_indices = selected_indices[numerical_keep]
    means = means[numerical_keep]
    scales = scales[numerical_keep]

    return MofaViewPreprocessor(
        view=view,
        selected_indices=selected_indices.astype(np.int64),
        means=means,
        scales=scales,
        training_missing_fraction=missing_fraction[selected_indices],
        training_variance=variance[selected_indices],
        n_eligible_before_budget=int(eligible_indices.size),
        feature_budget=int(feature_budget),
    )


def fit_all_preprocessors(
    arrays: dict[str, np.ndarray],
    training_indices: np.ndarray,
    contract: dict[str, Any],
) -> dict[str, MofaViewPreprocessor]:
    budgets = contract["feature_budgets"]
    result: dict[str, MofaViewPreprocessor] = {}

    for view in VIEWS:
        result[view] = fit_view_preprocessor(
            view=view,
            training_matrix=arrays[f"X_{VIEW_PREFIX[view]}"][training_indices, :],
            feature_budget=int(budgets[view]),
            contract=contract,
        )
    return result


def transformed_views(
    arrays: dict[str, np.ndarray],
    preprocessors: dict[str, MofaViewPreprocessor],
    indices: np.ndarray,
    *,
    complete: bool,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for view in VIEWS:
        prefix = VIEW_PREFIX[view]
        key = f"X_complete_{prefix}" if complete else f"X_{prefix}"
        matrix = arrays[key][indices, :]
        if complete:
            result[view] = preprocessors[view].transform_complete(matrix)
        else:
            result[view] = preprocessors[view].transform_observed(matrix)
    return result


def build_preprocessing_rows(
    arrays: dict[str, np.ndarray],
    preprocessors: dict[str, MofaViewPreprocessor],
    fit_stage: str,
    fold_index: int,
    rank: int,
    n_training: int,
    n_test: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for view in VIEWS:
        prep = preprocessors[view]
        rows.append(
            {
                "fit_stage": fit_stage,
                "fold_index": fold_index,
                "rank": rank,
                "view": view,
                "n_training_samples": n_training,
                "n_test_samples": n_test,
                "n_input_features": int(
                    arrays[f"X_{VIEW_PREFIX[view]}"].shape[1]
                ),
                "n_eligible_before_budget": prep.n_eligible_before_budget,
                "feature_budget": prep.feature_budget,
                "n_selected_features": int(prep.selected_indices.size),
                "mean_training_missing_fraction": float(
                    np.mean(prep.training_missing_fraction)
                ),
                "maximum_training_missing_fraction": float(
                    np.max(prep.training_missing_fraction)
                ),
                "median_training_prestandardization_variance": float(
                    np.median(prep.training_variance)
                ),
                "imputation_performed": False,
                "post_zscore_variance_used_for_ranking": False,
            }
        )
    return rows


# ---------------------------------------------------------------------
# Frozen Gaussian posterior projection
# ---------------------------------------------------------------------

def gaussian_posterior_project(
    observed_by_view: dict[str, np.ndarray],
    weights_by_view: dict[str, np.ndarray],
    precision_by_view: dict[str, np.ndarray],
    views_used: tuple[str, ...],
) -> np.ndarray:
    n_samples = int(observed_by_view[views_used[0]].shape[0])
    rank = int(weights_by_view[views_used[0]].shape[1])
    output = np.zeros((n_samples, rank), dtype=np.float64)
    identity = np.eye(rank, dtype=np.float64)

    for sample_index in range(n_samples):
        precision_matrix = identity.copy()
        information_vector = np.zeros(rank, dtype=np.float64)

        for view in views_used:
            x = np.asarray(
                observed_by_view[view][sample_index, :], dtype=np.float64
            )
            w = np.asarray(weights_by_view[view], dtype=np.float64)
            tau = np.asarray(precision_by_view[view], dtype=np.float64).reshape(-1)

            if x.shape[0] != w.shape[0] or x.shape[0] != tau.shape[0]:
                raise RuntimeError(f"{view}: projection dimensionality mismatch.")

            observed = (
                np.isfinite(x)
                & np.isfinite(tau)
                & (tau > 0)
                & np.all(np.isfinite(w), axis=1)
            )
            if not observed.any():
                continue

            x_obs = x[observed]
            w_obs = w[observed, :]
            tau_obs = tau[observed]

            precision_matrix += w_obs.T @ (tau_obs[:, None] * w_obs)
            information_vector += w_obs.T @ (tau_obs * x_obs)

        try:
            output[sample_index, :] = np.linalg.solve(
                precision_matrix, information_vector
            )
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Gaussian posterior projection solve failed.") from exc

    return output


# ---------------------------------------------------------------------
# MOFA fitting and expectation extraction
# ---------------------------------------------------------------------

def normalize_weight_matrix(
    values: np.ndarray,
    n_features: int,
    rank: int,
    view: str,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape == (n_features, rank):
        output = matrix
    elif matrix.shape == (rank, n_features):
        output = matrix.T
    else:
        raise RuntimeError(
            f"{view}: unexpected W shape {matrix.shape}; "
            f"expected {(n_features, rank)}."
        )
    if not np.isfinite(output).all():
        raise RuntimeError(f"{view}: non-finite W expectation.")
    return output


def normalize_factor_matrix(
    values: np.ndarray,
    n_samples: int,
    rank: int,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape == (n_samples, rank):
        output = matrix
    elif matrix.shape == (rank, n_samples):
        output = matrix.T
    else:
        raise RuntimeError(
            f"Unexpected Z shape {matrix.shape}; expected {(n_samples, rank)}."
        )
    if not np.isfinite(output).all():
        raise RuntimeError("Non-finite Z expectation.")
    return output


def normalize_tau_vector(
    values: np.ndarray,
    n_samples: int,
    n_features: int,
    view: str,
) -> tuple[np.ndarray, float]:
    array = np.asarray(values, dtype=np.float64)
    maximum_range = 0.0

    if array.ndim == 1:
        if array.shape[0] != n_features:
            raise RuntimeError(f"{view}: unexpected Tau vector shape {array.shape}.")
        tau = array

    elif array.ndim == 2:
        if array.shape[1] == n_features and array.shape[0] in {1, n_samples}:
            if array.shape[0] > 1:
                maximum_range = float(np.nanmax(np.ptp(array, axis=0)))
            tau = np.nanmedian(array, axis=0)
        elif array.shape[0] == n_features and array.shape[1] in {1, n_samples}:
            if array.shape[1] > 1:
                maximum_range = float(np.nanmax(np.ptp(array, axis=1)))
            tau = np.nanmedian(array, axis=1)
        else:
            raise RuntimeError(f"{view}: unexpected Tau matrix shape {array.shape}.")
    else:
        raise RuntimeError(f"{view}: unsupported Tau dimensionality {array.ndim}.")

    tau = np.asarray(tau, dtype=np.float64).reshape(-1)
    if tau.shape[0] != n_features:
        raise RuntimeError(f"{view}: Tau feature-count mismatch.")
    if not np.isfinite(tau).all() or np.any(tau <= 0):
        raise RuntimeError(f"{view}: invalid Tau expectation.")

    return tau, maximum_range


def summarize_training_stats(model: Any) -> dict[str, Any]:
    stats = model.getTrainingStats()
    elbo = np.asarray(stats.get("elbo", []), dtype=np.float64).reshape(-1)
    times = np.asarray(stats.get("time", []), dtype=np.float64).reshape(-1)
    factors = np.asarray(
        stats.get("number_factors", []), dtype=np.float64
    ).reshape(-1)

    finite_elbo = elbo[np.isfinite(elbo)]
    finite_time = times[np.isfinite(times)]
    finite_factors = factors[np.isfinite(factors)]

    decreases = (
        int(np.sum(np.diff(finite_elbo) < 0))
        if finite_elbo.size > 1
        else 0
    )

    return {
        "model_trained_flag": bool(getattr(model, "trained", False)),
        "n_training_iterations_recorded": int(finite_time.size),
        "training_time_seconds_sum": (
            float(np.sum(finite_time)) if finite_time.size else None
        ),
        "n_elbo_records": int(finite_elbo.size),
        "elbo_initial": float(finite_elbo[0]) if finite_elbo.size else None,
        "elbo_final": float(finite_elbo[-1]) if finite_elbo.size else None,
        "elbo_maximum": float(np.max(finite_elbo)) if finite_elbo.size else None,
        "n_elbo_decreases": decreases,
        "minimum_recorded_factor_count": (
            int(np.min(finite_factors)) if finite_factors.size else None
        ),
        "maximum_recorded_factor_count": (
            int(np.max(finite_factors)) if finite_factors.size else None
        ),
    }


def fit_mofa(
    training_views: dict[str, np.ndarray],
    selected_feature_indices: dict[str, np.ndarray],
    rank: int,
    seed: int,
    frozen_protocol: dict[str, Any],
    sample_ids: list[str],
) -> MofaFit:
    from mofapy2.run.entry_point import entry_point

    configuration = frozen_protocol["mofapy2_configuration"]
    data_options = dict(configuration["data_options"])
    model_options = dict(configuration["model_options"])
    train_options = dict(configuration["train_options"])

    model_options["factors"] = int(rank)
    train_options["seed"] = int(seed)
    train_options["verbose"] = False
    train_options["outfile"] = None
    train_options["save_interrupted"] = False

    data = [
        [np.asarray(training_views[view], dtype=np.float32)]
        for view in VIEWS
    ]
    feature_names = [
        [
            f"{view}_F{int(index):07d}"
            for index in selected_feature_indices[view].tolist()
        ]
        for view in VIEWS
    ]
    sample_names = [list(sample_ids)]

    start = time.perf_counter()

    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            ent = entry_point()
            ent.set_data_options(**data_options)
            ent.set_data_matrix(
                data,
                likelihoods=["gaussian", "gaussian", "gaussian"],
                views_names=list(VIEWS),
                groups_names=["single_group"],
                samples_names=sample_names,
                features_names=feature_names,
            )
            ent.set_model_options(**model_options)
            ent.set_train_options(**train_options)
            ent.build()
            ent.run()

    wall_seconds = time.perf_counter() - start
    model = ent.model
    if model is None or not bool(getattr(model, "trained", False)):
        raise RuntimeError("mofapy2 did not return a trained model.")

    w_expect = model.nodes["W"].getExpectations()
    tau_expect = model.nodes["Tau"].getExpectations()
    z_expect = model.nodes["Z"].getExpectations()

    if len(w_expect) != len(VIEWS) or len(tau_expect) != len(VIEWS):
        raise RuntimeError("Unexpected MOFA multiview expectation count.")

    weights: dict[str, np.ndarray] = {}
    tau: dict[str, np.ndarray] = {}
    tau_ranges: dict[str, float] = {}

    for view_index, view in enumerate(VIEWS):
        n_features = int(training_views[view].shape[1])
        weights[view] = normalize_weight_matrix(
            w_expect[view_index]["E"], n_features, rank, view
        )
        tau[view], tau_ranges[view] = normalize_tau_vector(
            tau_expect[view_index]["E"],
            n_samples=int(training_views[view].shape[0]),
            n_features=n_features,
            view=view,
        )

    factors = normalize_factor_matrix(
        z_expect["E"], n_samples=len(sample_ids), rank=rank
    )
    stats = summarize_training_stats(model)

    audit = {
        "fit_status": "PASS",
        "rank": rank,
        "mofa_seed": seed,
        "wall_time_seconds": wall_seconds,
        **stats,
        "tau_max_within_feature_range_RNA": tau_ranges["RNA"],
        "tau_max_within_feature_range_METHYLATION": tau_ranges["METHYLATION"],
        "tau_max_within_feature_range_CNV": tau_ranges["CNV"],
        "n_retained_factors": int(factors.shape[1]),
    }

    del model
    del ent

    return MofaFit(
        rank=rank,
        seed=seed,
        weights=weights,
        tau=tau,
        training_factors=factors,
        audit=audit,
    )


# ---------------------------------------------------------------------
# Frozen split and seed lookup
# ---------------------------------------------------------------------

def parse_indices(text: str) -> np.ndarray:
    values = [value for value in clean_text(text).split(";") if value]
    return np.asarray([int(value) for value in values], dtype=int)


def build_split_lookup(
    table: pd.DataFrame,
) -> dict[tuple[str, int, int], list[tuple[int, np.ndarray, np.ndarray]]]:
    required = {
        "scenario_id",
        "replicate_index",
        "simulation_seed",
        "fold_index",
        "training_indices",
        "test_indices",
    }
    missing = required - set(table.columns)
    if missing:
        raise RuntimeError("Split registry missing: " + ", ".join(sorted(missing)))

    result: dict[
        tuple[str, int, int],
        list[tuple[int, np.ndarray, np.ndarray]],
    ] = {}

    for _, row in table.iterrows():
        key = (
            clean_text(row["scenario_id"]),
            int(float(row["replicate_index"])),
            int(float(row["simulation_seed"])),
        )
        result.setdefault(key, []).append(
            (
                int(float(row["fold_index"])),
                parse_indices(row["training_indices"]),
                parse_indices(row["test_indices"]),
            )
        )

    if len(result) != EXPECTED_DATASETS:
        raise RuntimeError("Split registry dataset count mismatch.")

    for key, folds in result.items():
        folds.sort(key=lambda item: item[0])
        if [item[0] for item in folds] != [1, 2, 3, 4, 5]:
            raise RuntimeError(f"Invalid fold coverage for {key}.")

    return result


def build_seed_lookup(
    table: pd.DataFrame,
) -> dict[tuple[str, str, int, int, int, int], int]:
    required = {
        "fit_stage",
        "scenario_id",
        "replicate_index",
        "simulation_seed",
        "latent_rank",
        "fold_index",
        "mofa_seed",
    }
    missing = required - set(table.columns)
    if missing:
        raise RuntimeError("Seed registry missing: " + ", ".join(sorted(missing)))

    result: dict[tuple[str, str, int, int, int, int], int] = {}
    for _, row in table.iterrows():
        key = (
            clean_text(row["fit_stage"]),
            clean_text(row["scenario_id"]),
            int(float(row["replicate_index"])),
            int(float(row["simulation_seed"])),
            int(float(row["latent_rank"])),
            int(float(row["fold_index"])),
        )
        if key in result:
            raise RuntimeError(f"Duplicate fit seed key: {key}")
        result[key] = int(float(row["mofa_seed"]))

    if len(result) != EXPECTED_FITS:
        raise RuntimeError("Frozen fit-seed count mismatch.")
    return result


# ---------------------------------------------------------------------
# MOFA-specific metrics
# ---------------------------------------------------------------------

def modality_loading_energy(
    weights: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    rank = int(next(iter(weights.values())).shape[1])
    energy = np.zeros((rank, len(VIEWS)), dtype=np.float64)

    for view in VIEWS:
        energy[:, VIEW_INDEX[view]] = np.sum(weights[view] ** 2, axis=0)

    total = np.sum(energy, axis=1, keepdims=True)
    fractions = np.divide(
        energy,
        total,
        out=np.zeros_like(energy),
        where=total > 0,
    )

    entropy = np.zeros(rank, dtype=np.float64)
    for factor_index in range(rank):
        positive = fractions[factor_index, :] > 0
        if positive.any():
            entropy[factor_index] = -float(
                np.sum(
                    fractions[factor_index, positive]
                    * np.log(fractions[factor_index, positive])
                )
                / math.log(len(VIEWS))
            )
    return fractions, entropy


# ---------------------------------------------------------------------
# Held-out and full-truth evaluation
# ---------------------------------------------------------------------

def evaluate_heldout_fit(
    arrays: dict[str, np.ndarray],
    training_indices: np.ndarray,
    test_indices: np.ndarray,
    fold_index: int,
    rank: int,
    seed: int,
    frozen_protocol: dict[str, Any],
    linear: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    prep_contract = frozen_protocol["fold_local_preprocessing"]
    preprocessors = fit_all_preprocessors(arrays, training_indices, prep_contract)

    training_views = transformed_views(
        arrays, preprocessors, training_indices, complete=False
    )
    test_views = transformed_views(
        arrays, preprocessors, test_indices, complete=False
    )
    test_truth = transformed_views(
        arrays, preprocessors, test_indices, complete=True
    )

    pre_rows = build_preprocessing_rows(
        arrays=arrays,
        preprocessors=preprocessors,
        fit_stage="heldout_cv",
        fold_index=fold_index,
        rank=rank,
        n_training=int(training_indices.size),
        n_test=int(test_indices.size),
    )

    selected_indices = {
        view: preprocessors[view].selected_indices
        for view in VIEWS
    }

    fit = fit_mofa(
        training_views=training_views,
        selected_feature_indices=selected_indices,
        rank=rank,
        seed=seed,
        frozen_protocol=frozen_protocol,
        sample_ids=[f"TRAIN_{int(index):04d}" for index in training_indices],
    )

    all_view_scores = gaussian_posterior_project(
        test_views, fit.weights, fit.tau, VIEWS
    )

    reconstruction: dict[str, dict[str, Any]] = {}
    for view in VIEWS:
        predicted = all_view_scores @ fit.weights[view].T
        reconstruction[view] = linear.normalized_reconstruction_metrics(
            predicted, test_truth[view]
        )

    mean_error = linear.nanmean_or_none(
        reconstruction[view]["normalized_error"] for view in VIEWS
    )
    reconstruction_score = (
        float(np.clip(1.0 - mean_error, 0.0, 1.0))
        if mean_error is not None
        else None
    )

    training_embeddings = {
        view: gaussian_posterior_project(
            training_views, fit.weights, fit.tau, (view,)
        )
        for view in VIEWS
    }
    test_embeddings = {
        view: gaussian_posterior_project(
            test_views, fit.weights, fit.tau, (view,)
        )
        for view in VIEWS
    }
    row_source = {
        view: arrays[f"row_source_{VIEW_PREFIX[view]}"].astype(int)
        for view in VIEWS
    }

    retrieval = linear.cross_view_retrieval_metrics(
        training_embeddings,
        test_embeddings,
        row_source,
        training_indices,
        test_indices,
    )

    missing_view_r2: dict[str, float | None] = {}
    for omitted_view in VIEWS:
        remaining = tuple(view for view in VIEWS if view != omitted_view)
        scores = gaussian_posterior_project(
            test_views, fit.weights, fit.tau, remaining
        )
        predicted = scores @ fit.weights[omitted_view].T
        metric = linear.normalized_reconstruction_metrics(
            predicted, test_truth[omitted_view]
        )
        missing_view_r2[omitted_view] = metric["r2"]

    missing_score = linear.nanmean_or_none(
        np.clip(value, 0.0, 1.0)
        for value in missing_view_r2.values()
        if value is not None
    )

    _, entropy = modality_loading_energy(fit.weights)

    fold_row = {
        "fit_stage": "heldout_cv",
        "fold_index": fold_index,
        "model_id": f"MOFA_K{rank}",
        "model_family": "MOFA_plus",
        "rank": rank,
        "view": "ALL3",
        "n_training_samples": int(training_indices.size),
        "n_test_samples": int(test_indices.size),
        "heldout_normalized_reconstruction_error": mean_error,
        "heldout_reconstruction_score": reconstruction_score,
        "heldout_reconstruction_r2": (
            None if mean_error is None else 1.0 - mean_error
        ),
        "heldout_reconstruction_error_RNA": reconstruction["RNA"]["normalized_error"],
        "heldout_reconstruction_error_METHYLATION": reconstruction["METHYLATION"]["normalized_error"],
        "heldout_reconstruction_error_CNV": reconstruction["CNV"]["normalized_error"],
        "cross_view_mrr": retrieval["cross_view_mrr"],
        "random_expected_cross_view_mrr": retrieval["random_expected_cross_view_mrr"],
        "random_adjusted_cross_view_mrr": retrieval["random_adjusted_cross_view_mrr"],
        "cross_view_top1_accuracy": retrieval["cross_view_top1_accuracy"],
        "same_patient_similarity_margin": retrieval["same_patient_similarity_margin"],
        "nominal_index_cross_view_mrr": retrieval["nominal_index_cross_view_mrr"],
        "n_valid_retrieval_queries": retrieval["n_valid_retrieval_queries"],
        "missing_view_r2_RNA": missing_view_r2["RNA"],
        "missing_view_r2_METHYLATION": missing_view_r2["METHYLATION"],
        "missing_view_r2_CNV": missing_view_r2["CNV"],
        "missing_view_reconstruction_score": missing_score,
        "modality_contribution_entropy": float(np.median(entropy)),
        "fold_latent_effective_rank": linear.effective_rank(all_view_scores),
        "fit_status": "PASS",
        "fit_error_type": "",
        "fit_error_message": "",
    }

    fit_audit = {
        "fit_stage": "heldout_cv",
        "fold_index": fold_index,
        "model_id": f"MOFA_K{rank}",
        **fit.audit,
    }

    return fold_row, pre_rows, fit_audit


def failed_fold_row(
    fold_index: int,
    rank: int,
    n_training: int,
    n_test: int,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "fit_stage": "heldout_cv",
        "fold_index": fold_index,
        "model_id": f"MOFA_K{rank}",
        "model_family": "MOFA_plus",
        "rank": rank,
        "view": "ALL3",
        "n_training_samples": n_training,
        "n_test_samples": n_test,
        "heldout_normalized_reconstruction_error": None,
        "heldout_reconstruction_score": None,
        "heldout_reconstruction_r2": None,
        "heldout_reconstruction_error_RNA": None,
        "heldout_reconstruction_error_METHYLATION": None,
        "heldout_reconstruction_error_CNV": None,
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
        "modality_contribution_entropy": None,
        "fold_latent_effective_rank": None,
        "fit_status": "FAIL",
        "fit_error_type": type(error).__name__,
        "fit_error_message": clean_text(str(error))[:1000],
    }


def evaluate_full_truth_fit(
    arrays: dict[str, np.ndarray],
    rank: int,
    seed: int,
    frozen_protocol: dict[str, Any],
    truth_threshold: float,
    linear: Any,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    all_indices = np.arange(arrays["sample_index"].shape[0], dtype=int)
    prep_contract = frozen_protocol["fold_local_preprocessing"]
    preprocessors = fit_all_preprocessors(arrays, all_indices, prep_contract)
    observed = transformed_views(
        arrays, preprocessors, all_indices, complete=False
    )

    pre_rows = build_preprocessing_rows(
        arrays=arrays,
        preprocessors=preprocessors,
        fit_stage="full_data_truth_recovery",
        fold_index=0,
        rank=rank,
        n_training=int(all_indices.size),
        n_test=0,
    )

    fit = fit_mofa(
        training_views=observed,
        selected_feature_indices={
            view: preprocessors[view].selected_indices for view in VIEWS
        },
        rank=rank,
        seed=seed,
        frozen_protocol=frozen_protocol,
        sample_ids=[f"FULL_{index:04d}" for index in all_indices],
    )

    total_scores = fit.training_factors
    biological_truth, biological_ids = linear.integrated_anchor_truth(
        arrays, "biological"
    )
    technical_truth, _ = linear.integrated_anchor_truth(arrays, "technical")

    biological_subspace = linear.subspace_metrics(total_scores, biological_truth)
    matching, factor_rows = linear.factor_matching(
        total_scores, biological_truth, biological_ids, truth_threshold
    )
    technical_subspace = linear.subspace_metrics(total_scores, technical_truth)

    biological_corr = linear.absolute_correlation_matrix(
        total_scores, biological_truth
    )
    technical_corr = linear.absolute_correlation_matrix(
        total_scores, technical_truth
    )

    one_view_scores = {
        view: gaussian_posterior_project(
            observed, fit.weights, fit.tau, (view,)
        )
        for view in VIEWS
    }

    private_recoveries: list[float] = []
    private_corr: dict[str, np.ndarray] = {}

    for view in VIEWS:
        private_truth, _ = linear.row_aligned_truth(arrays, view, "private")
        private_result = linear.subspace_metrics(
            one_view_scores[view], private_truth
        )
        value = private_result["mean_canonical_correlation"]
        if value is not None:
            private_recoveries.append(float(value))
        private_corr[view] = linear.absolute_correlation_matrix(
            one_view_scores[view], private_truth
        )

    energy, entropy = modality_loading_energy(fit.weights)
    truth_support = arrays["biological_support_mask"].astype(bool)
    truth_classes = [
        str(value)
        for value in arrays["biological_factor_truth_class"].tolist()
    ]

    support_jaccards: list[float] = []
    support_exact: list[float] = []

    for factor_row in factor_rows:
        estimated_index = int(factor_row["estimated_factor_index"])
        truth_index_raw = factor_row["truth_factor_index"]

        factor_row["best_biological_absolute_correlation"] = (
            float(np.max(biological_corr[estimated_index, :]))
            if biological_corr.shape[1] > 0
            else None
        )
        factor_row["best_private_absolute_correlation"] = linear.nanmean_or_none(
            (
                float(np.max(private_corr[view][estimated_index, :]))
                if private_corr[view].shape[1] > 0
                else None
            )
            for view in VIEWS
        )
        factor_row["best_technical_absolute_correlation"] = (
            float(np.max(technical_corr[estimated_index, :]))
            if technical_corr.shape[1] > 0
            else None
        )

        for view in VIEWS:
            factor_row[f"loading_energy_{view}"] = float(
                energy[estimated_index, VIEW_INDEX[view]]
            )
        factor_row["modality_contribution_entropy"] = float(
            entropy[estimated_index]
        )

        predicted_support = (
            energy[estimated_index, :] >= VIEW_SUPPORT_ENERGY_THRESHOLD
        )
        factor_row["predicted_view_support"] = ";".join(
            view
            for view in VIEWS
            if predicted_support[VIEW_INDEX[view]]
        )

        if truth_index_raw is None:
            factor_row["truth_view_support"] = ""
            factor_row["truth_factor_class"] = ""
            factor_row["view_support_jaccard"] = None
            factor_row["view_support_exact"] = None
            continue

        truth_index = int(truth_index_raw)
        actual_support = truth_support[truth_index, :]
        factor_row["truth_view_support"] = ";".join(
            view for view in VIEWS if actual_support[VIEW_INDEX[view]]
        )
        factor_row["truth_factor_class"] = (
            truth_classes[truth_index]
            if truth_index < len(truth_classes)
            else ""
        )

        jaccard = linear.support_jaccard(predicted_support, actual_support)
        exact = float(np.array_equal(predicted_support, actual_support))
        factor_row["view_support_jaccard"] = jaccard
        factor_row["view_support_exact"] = bool(exact)

        if bool(factor_row["passes_truth_match_threshold"]):
            support_jaccards.append(jaccard)
            support_exact.append(exact)

    metrics = {
        "full_fit_status": "PASS",
        "full_fit_error_type": "",
        "full_fit_error_message": "",
        "full_effective_rank": linear.effective_rank(total_scores),
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
        "private_metric_semantics": (
            "view_specific_signal_capture_not_explicit_private_block"
        ),
        "technical_subspace_recovery": technical_subspace[
            "mean_canonical_correlation"
        ],
        "true_factor_recovery_rate": matching["true_factor_recovery_rate"],
        "estimated_factor_match_fraction": matching[
            "estimated_factor_match_fraction"
        ],
        "false_shared_discovery_fraction": matching[
            "false_shared_discovery_fraction"
        ],
        "mean_matched_absolute_correlation": matching[
            "mean_matched_absolute_correlation"
        ],
        "minimum_matched_absolute_correlation": matching[
            "minimum_matched_absolute_correlation"
        ],
        "modality_contribution_entropy": float(np.median(entropy)),
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

    fit_audit = {
        "fit_stage": "full_data_truth_recovery",
        "fold_index": 0,
        "model_id": f"MOFA_K{rank}",
        **fit.audit,
    }

    return metrics, factor_rows, pre_rows, fit_audit


def failed_full_metrics(error: BaseException) -> dict[str, Any]:
    return {
        "full_fit_status": "FAIL",
        "full_fit_error_type": type(error).__name__,
        "full_fit_error_message": clean_text(str(error))[:1000],
        "full_effective_rank": None,
        "shared_subspace_recovery": None,
        "shared_subspace_minimum_correlation": None,
        "shared_subspace_mean_angle_degrees": None,
        "shared_subspace_max_angle_degrees": None,
        "shared_subspace_projection_distance": None,
        "private_subspace_recovery": None,
        "private_metric_semantics": (
            "view_specific_signal_capture_not_explicit_private_block"
        ),
        "technical_subspace_recovery": None,
        "true_factor_recovery_rate": None,
        "estimated_factor_match_fraction": None,
        "false_shared_discovery_fraction": None,
        "mean_matched_absolute_correlation": None,
        "minimum_matched_absolute_correlation": None,
        "modality_contribution_entropy": None,
        "view_support_jaccard": None,
        "view_support_exact_accuracy": None,
    }


FOLD_METRICS = [
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


def aggregate_folds(
    fold_rows: list[dict[str, Any]],
    rank: int,
    linear: Any,
) -> dict[str, Any]:
    table = pd.DataFrame(fold_rows)
    if table.shape[0] != CV_FOLDS:
        raise RuntimeError("Rank shard must contain five held-out folds.")

    successful = table[table["fit_status"].eq("PASS")]
    all_passed = bool(successful.shape[0] == CV_FOLDS)

    row: dict[str, Any] = {
        "model_id": f"MOFA_K{rank}",
        "model_family": "MOFA_plus",
        "rank": rank,
        "view": "ALL3",
        "n_folds": CV_FOLDS,
        "n_successful_folds": int(successful.shape[0]),
        "all_fold_fits_passed": all_passed,
        "total_heldout_samples": int(
            pd.to_numeric(table["n_test_samples"], errors="coerce").sum()
        ),
        "total_valid_retrieval_queries": int(
            pd.to_numeric(
                successful.get(
                    "n_valid_retrieval_queries", pd.Series(dtype=float)
                ),
                errors="coerce",
            )
            .fillna(0)
            .sum()
        ),
    }

    for metric in FOLD_METRICS:
        descriptive = linear.nanmean_or_none(
            successful[metric].tolist() if metric in successful.columns else []
        )
        row[f"{metric}_successful_fold_mean_descriptive"] = descriptive
        row[metric] = descriptive if all_passed else None

    return row


def apply_failure_gates(dataset_row: dict[str, Any]) -> list[dict[str, Any]]:
    g01 = (
        not bool(dataset_row.get("all_fold_fits_passed", False))
        or clean_text(dataset_row.get("full_fit_status")) != "PASS"
    )
    if not g01:
        for value in [
            dataset_row.get("heldout_normalized_reconstruction_error"),
            dataset_row.get("full_effective_rank"),
        ]:
            if value is None or not math.isfinite(float(value)):
                g01 = True
                break

    definitions = [
        (
            "G01_NONFINITE_OPTIMIZATION",
            g01,
            "all fitted models",
            "report_failure_without_silent_omission",
        ),
        (
            "G02_LATENT_COLLAPSE",
            (
                dataset_row.get("full_effective_rank") is not None
                and float(dataset_row["full_effective_rank"]) < 2.0
            ),
            "integrated latent models",
            "report_failure_without_silent_omission",
        ),
        (
            "G03_RECONSTRUCTION_WORSE_THAN_NULL",
            (
                dataset_row.get("heldout_normalized_reconstruction_error")
                is not None
                and float(
                    dataset_row["heldout_normalized_reconstruction_error"]
                )
                >= 1.0
            ),
            "all reconstructed models",
            "report_failure_without_silent_omission",
        ),
        (
            "G04_RETRIEVAL_NOT_ABOVE_RANDOM",
            (
                dataset_row.get("random_adjusted_cross_view_mrr") is not None
                and float(dataset_row["random_adjusted_cross_view_mrr"]) <= 0.0
            ),
            "integrated multi-view models",
            "report_failure_without_silent_omission",
        ),
        (
            "G05_SINGLE_VIEW_DOMINATION",
            (
                dataset_row.get("modality_contribution_entropy") is not None
                and float(dataset_row["modality_contribution_entropy"])
                < MODALITY_DOMINATION_ENTROPY_THRESHOLD
            ),
            "integrated shared representations",
            "report_failure_without_silent_omission",
        ),
        (
            "G09_SIMULATION_FALSE_SHARED_DISCOVERY",
            (
                dataset_row.get("false_shared_discovery_fraction") is not None
                and float(dataset_row["false_shared_discovery_fraction"]) > 0.10
            ),
            "diagnostic only for MOFA probabilistic baseline",
            "diagnostic_only_for_probabilistic_baseline",
        ),
    ]

    return [
        {
            "model_id": dataset_row["model_id"],
            "rank": int(dataset_row["rank"]),
            "gate_id": gate_id,
            "applicable": True,
            "failed": bool(failed),
            "assessment_scope": scope,
            "consequence": consequence,
        }
        for gate_id, failed, scope, consequence in definitions
    ]


# ---------------------------------------------------------------------
# Dataset-rank evaluation and resumable shards
# ---------------------------------------------------------------------

def build_base_metadata(
    row: pd.Series,
    dataset_path: Path,
    dataset_hash: str,
) -> dict[str, Any]:
    return {
        "registry_label": clean_text(row["registry_label"]),
        "scenario_id": clean_text(row["scenario_id"]),
        "scenario_origin": clean_text(row["scenario_origin"]),
        "scenario_family": clean_text(row["scenario_family"]),
        "calibration_role": clean_text(row["calibration_role"]),
        "replicate_index": int(float(row["replicate_index"])),
        "simulation_seed": int(float(row["simulation_seed"])),
        "n_samples": int(float(row["n_samples"])),
        "n_true_biological_shared_factors": int(
            float(row["n_biological_shared_factors"])
        ),
        "n_true_private_factors_per_view": int(
            float(row["n_private_factors_per_view"])
        ),
        "n_true_technical_shared_factors": int(
            float(row["n_technical_shared_factors"])
        ),
        "factor_level_identifiable": parse_bool(
            row["factor_level_identifiable"]
        ),
        "subspace_level_identifiable": parse_bool(
            row["subspace_level_identifiable"]
        ),
        "primary_truth_target": clean_text(row["primary_truth_target"]),
        "input_path": relative_path(dataset_path),
        "input_sha256": dataset_hash,
    }


def evaluate_one_rank(
    manifest_row: pd.Series,
    rank: int,
    folds: list[tuple[int, np.ndarray, np.ndarray]],
    seed_lookup: dict[tuple[str, str, int, int, int, int], int],
    frozen_protocol: dict[str, Any],
    truth_threshold: float,
    linear: Any,
) -> dict[str, Any]:
    scenario_id = clean_text(manifest_row["scenario_id"])
    replicate_index = int(float(manifest_row["replicate_index"]))
    simulation_seed = int(float(manifest_row["simulation_seed"]))

    dataset_path = resolve_project_path(manifest_row["path"])
    require_file(dataset_path)

    expected_size = int(float(manifest_row["size_bytes"]))
    if int(dataset_path.stat().st_size) != expected_size:
        raise RuntimeError(f"Simulation size mismatch: {relative_path(dataset_path)}")

    dataset_hash = sha256_file(dataset_path)
    if dataset_hash != clean_text(manifest_row["sha256"]):
        raise RuntimeError(f"Simulation hash mismatch: {relative_path(dataset_path)}")

    arrays = linear.load_simulation_arrays(dataset_path)
    metadata = build_base_metadata(manifest_row, dataset_path, dataset_hash)

    fold_rows: list[dict[str, Any]] = []
    preprocessing_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []

    for fold_index, training_indices, test_indices in folds:
        seed_key = (
            "heldout_cv",
            scenario_id,
            replicate_index,
            simulation_seed,
            rank,
            fold_index,
        )
        if seed_key not in seed_lookup:
            raise RuntimeError(f"Missing frozen fit seed: {seed_key}")
        seed = seed_lookup[seed_key]

        try:
            fold_row, pre_rows, fit_audit = evaluate_heldout_fit(
                arrays=arrays,
                training_indices=training_indices,
                test_indices=test_indices,
                fold_index=fold_index,
                rank=rank,
                seed=seed,
                frozen_protocol=frozen_protocol,
                linear=linear,
            )
            preprocessing_rows.extend(pre_rows)
            fit_rows.append(fit_audit)

        except (KeyboardInterrupt, SystemExit):
            raise

        except Exception as exc:
            fold_row = failed_fold_row(
                fold_index=fold_index,
                rank=rank,
                n_training=int(training_indices.size),
                n_test=int(test_indices.size),
                error=exc,
            )
            fit_rows.append(
                {
                    "fit_stage": "heldout_cv",
                    "fold_index": fold_index,
                    "model_id": f"MOFA_K{rank}",
                    "rank": rank,
                    "mofa_seed": seed,
                    "fit_status": "FAIL",
                    "fit_error_type": type(exc).__name__,
                    "fit_error_message": clean_text(str(exc))[:1000],
                    "fit_traceback_tail": "\n".join(
                        traceback.format_exc().splitlines()[-12:]
                    ),
                }
            )

        fold_row.update(metadata)
        fold_rows.append(fold_row)
        gc.collect()

    dataset_row = aggregate_folds(fold_rows, rank, linear)

    full_seed_key = (
        "full_data_truth_recovery",
        scenario_id,
        replicate_index,
        simulation_seed,
        rank,
        0,
    )
    if full_seed_key not in seed_lookup:
        raise RuntimeError(f"Missing frozen full-data fit seed: {full_seed_key}")
    full_seed = seed_lookup[full_seed_key]

    factor_rows: list[dict[str, Any]] = []

    try:
        (
            truth_metrics,
            factor_rows,
            full_pre_rows,
            full_fit_audit,
        ) = evaluate_full_truth_fit(
            arrays=arrays,
            rank=rank,
            seed=full_seed,
            frozen_protocol=frozen_protocol,
            truth_threshold=truth_threshold,
            linear=linear,
        )
        preprocessing_rows.extend(full_pre_rows)
        fit_rows.append(full_fit_audit)

    except (KeyboardInterrupt, SystemExit):
        raise

    except Exception as exc:
        truth_metrics = failed_full_metrics(exc)
        fit_rows.append(
            {
                "fit_stage": "full_data_truth_recovery",
                "fold_index": 0,
                "model_id": f"MOFA_K{rank}",
                "rank": rank,
                "mofa_seed": full_seed,
                "fit_status": "FAIL",
                "fit_error_type": type(exc).__name__,
                "fit_error_message": clean_text(str(exc))[:1000],
                "fit_traceback_tail": "\n".join(
                    traceback.format_exc().splitlines()[-12:]
                ),
            }
        )

    dataset_row.update(truth_metrics)
    dataset_row.update(metadata)

    factor_primary = bool(
        metadata["factor_level_identifiable"]
        or metadata["n_true_biological_shared_factors"] == 0
    )
    dataset_row["factor_level_metrics_primary_eligible"] = factor_primary

    if (
        not factor_primary
        and metadata["n_true_biological_shared_factors"] > 0
    ):
        for metric in [
            "true_factor_recovery_rate",
            "estimated_factor_match_fraction",
            "false_shared_discovery_fraction",
            "mean_matched_absolute_correlation",
            "minimum_matched_absolute_correlation",
        ]:
            dataset_row[f"{metric}_descriptive"] = dataset_row.get(metric)
            dataset_row[metric] = None

    utility = [
        (0.40, dataset_row.get("shared_subspace_recovery")),
        (0.20, dataset_row.get("private_subspace_recovery")),
        (0.20, dataset_row.get("random_adjusted_cross_view_mrr")),
        (0.10, dataset_row.get("missing_view_reconstruction_score")),
        (0.10, dataset_row.get("modality_contribution_entropy")),
    ]

    if (
        bool(dataset_row["all_fold_fits_passed"])
        and clean_text(dataset_row.get("full_fit_status")) == "PASS"
        and all(
            value is not None and math.isfinite(float(value))
            for _, value in utility
        )
    ):
        dataset_row["simulation_utility"] = float(
            sum(
                weight * float(np.clip(value, 0.0, 1.0))
                for weight, value in utility
            )
        )
    else:
        dataset_row["simulation_utility"] = None

    failure_rows = apply_failure_gates(dataset_row)

    for collection in [
        preprocessing_rows,
        fit_rows,
        factor_rows,
        failure_rows,
    ]:
        for item in collection:
            item.update(metadata)
            item["rank"] = rank
            item["model_id"] = f"MOFA_K{rank}"

    return {
        "script_version": SCRIPT_VERSION,
        "scenario_id": scenario_id,
        "replicate_index": replicate_index,
        "simulation_seed": simulation_seed,
        "rank": rank,
        "model_id": f"MOFA_K{rank}",
        "input_path": relative_path(dataset_path),
        "input_sha256": dataset_hash,
        "fold_rows": fold_rows,
        "preprocessing_rows": preprocessing_rows,
        "fit_rows": fit_rows,
        "dataset_row": dataset_row,
        "factor_rows": factor_rows,
        "failure_rows": failure_rows,
    }


def shard_path(row: pd.Series, rank: int) -> Path:
    dataset_path = Path(clean_text(row["path"]))
    return (
        OUTPUT_SHARD_DIR
        / clean_text(row["scenario_family"])
        / clean_text(row["scenario_id"])
        / (dataset_path.stem + f"__MOFA_K{rank}.json")
    )


def validate_existing_shard(
    shard: dict[str, Any],
    row: pd.Series,
    rank: int,
    protocol_hash: str,
) -> None:
    if clean_text(shard.get("script_version")) != SCRIPT_VERSION:
        raise RuntimeError("Shard script-version mismatch.")
    if clean_text(shard.get("protocol_sha256")) != protocol_hash:
        raise RuntimeError("Shard protocol hash mismatch.")
    if int(shard.get("rank")) != rank:
        raise RuntimeError("Shard rank mismatch.")
    if clean_text(shard.get("input_sha256")) != clean_text(row["sha256"]):
        raise RuntimeError("Shard input hash mismatch.")
    if clean_text(shard.get("scenario_id")) != clean_text(row["scenario_id"]):
        raise RuntimeError("Shard scenario mismatch.")
    if int(shard.get("replicate_index")) != int(float(row["replicate_index"])):
        raise RuntimeError("Shard replicate mismatch.")

    input_path = resolve_project_path(row["path"])
    require_file(input_path)
    if int(input_path.stat().st_size) != int(float(row["size_bytes"])):
        raise RuntimeError("Existing shard input-file size mismatch.")
    if sha256_file(input_path) != clean_text(row["sha256"]):
        raise RuntimeError("Existing shard input-file SHA256 mismatch.")


def materialize_shard(
    row: pd.Series,
    rank: int,
    folds: list[tuple[int, np.ndarray, np.ndarray]],
    seed_lookup: dict[tuple[str, str, int, int, int, int], int],
    frozen_protocol: dict[str, Any],
    truth_threshold: float,
    protocol_hash: str,
    linear: Any,
) -> tuple[dict[str, Any], str]:
    path = shard_path(row, rank)

    if path.exists():
        shard = load_json(path)
        validate_existing_shard(shard, row, rank, protocol_hash)
        return shard, "verified_existing"

    result = evaluate_one_rank(
        manifest_row=row,
        rank=rank,
        folds=folds,
        seed_lookup=seed_lookup,
        frozen_protocol=frozen_protocol,
        truth_threshold=truth_threshold,
        linear=linear,
    )
    result["protocol_sha256"] = protocol_hash
    result["created_at_utc"] = utc_now_iso()
    write_json_atomic(result, path)

    serialized = load_json(path)
    validate_existing_shard(serialized, row, rank, protocol_hash)
    return serialized, "generated"


def aggregate_shards(
    shards: list[dict[str, Any]],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    preprocessing: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    datasets: list[dict[str, Any]] = []
    factors: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for shard in shards:
        preprocessing.extend(shard["preprocessing_rows"])
        fits.extend(shard["fit_rows"])
        folds.extend(shard["fold_rows"])
        datasets.append(shard["dataset_row"])
        factors.extend(shard["factor_rows"])
        failures.extend(shard["failure_rows"])

    return (
        pd.DataFrame(preprocessing),
        pd.DataFrame(fits),
        pd.DataFrame(folds),
        pd.DataFrame(datasets),
        pd.DataFrame(factors),
        pd.DataFrame(failures),
    )


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

SUMMARY_METRICS = [
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


def summarize_scenarios(dataset_table: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "scenario_id",
        "scenario_origin",
        "scenario_family",
        "calibration_role",
        "model_id",
        "model_family",
        "rank",
    ]
    rows: list[dict[str, Any]] = []

    for keys, part in dataset_table.groupby(
        group_columns, dropna=False, sort=True
    ):
        row = dict(zip(group_columns, keys))
        row["n_datasets"] = int(part.shape[0])
        row["n_all_fold_fits_passed"] = int(
            part["all_fold_fits_passed"].map(parse_bool).sum()
        )
        row["n_full_fits_passed"] = int(
            part["full_fit_status"].eq("PASS").sum()
        )

        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(part[metric], errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            row[f"{metric}_n"] = int(finite.size)

            if finite.size == 0:
                row[f"{metric}_mean"] = None
                row[f"{metric}_sd"] = None
                row[f"{metric}_se"] = None
                row[f"{metric}_median"] = None
            else:
                row[f"{metric}_mean"] = float(np.mean(finite))
                row[f"{metric}_sd"] = (
                    float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
                )
                row[f"{metric}_se"] = (
                    float(np.std(finite, ddof=1) / math.sqrt(finite.size))
                    if finite.size > 1
                    else 0.0
                )
                row[f"{metric}_median"] = float(np.median(finite))

        rows.append(row)

    return pd.DataFrame(rows)


def build_paired_linear_comparison(
    mofa: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    linear = pd.read_csv(LINEAR_DATASET_METRICS, low_memory=False)
    linear = linear[
        linear["model_id"].eq("BLOCK_PCA")
        & linear["rank"].isin([5, 10, 15])
    ].copy()
    mofa = mofa[mofa["rank"].isin([5, 10, 15])].copy()

    keys = ["scenario_id", "replicate_index", "simulation_seed", "rank"]
    metrics = [
        "heldout_reconstruction_score",
        "random_adjusted_cross_view_mrr",
        "missing_view_reconstruction_score",
        "modality_contribution_entropy",
        "shared_subspace_recovery",
        "private_subspace_recovery",
        "technical_subspace_recovery",
        "simulation_utility",
    ]

    paired = mofa[
        [*keys, "model_id", *metrics]
    ].merge(
        linear[[*keys, *metrics]],
        on=keys,
        how="inner",
        suffixes=("_mofa", "_block_pca"),
        validate="one_to_one",
    )

    if paired.shape[0] != EXPECTED_DATASETS * 3:
        raise RuntimeError("Same-rank MOFA/BLOCK_PCA paired row count mismatch.")

    for metric in metrics:
        paired[f"{metric}_delta_mofa_minus_block_pca"] = (
            pd.to_numeric(paired[f"{metric}_mofa"], errors="coerce")
            - pd.to_numeric(paired[f"{metric}_block_pca"], errors="coerce")
        )

    delta_columns = [
        column
        for column in paired.columns
        if column.endswith("_delta_mofa_minus_block_pca")
    ]
    rows: list[dict[str, Any]] = []

    for (scenario_id, rank), part in paired.groupby(
        ["scenario_id", "rank"], sort=True
    ):
        row: dict[str, Any] = {
            "scenario_id": scenario_id,
            "rank": int(rank),
            "model_id": f"MOFA_K{int(rank)}",
            "linear_comparator": "BLOCK_PCA",
            "n_paired_datasets": int(part.shape[0]),
            "comparison_role": "descriptive_only_no_model_selection",
        }

        for column in delta_columns:
            values = pd.to_numeric(part[column], errors="coerce").to_numpy(float)
            values = values[np.isfinite(values)]
            row[f"{column}_mean"] = (
                float(np.mean(values)) if values.size else None
            )
            row[f"{column}_median"] = (
                float(np.median(values)) if values.size else None
            )
            row[f"{column}_se"] = (
                float(np.std(values, ddof=1) / math.sqrt(values.size))
                if values.size > 1
                else (0.0 if values.size == 1 else None)
            )
        rows.append(row)

    return paired, pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Installed-mofapy2 reference fixture
# ---------------------------------------------------------------------

def run_reference_fixture(
    frozen_protocol: dict[str, Any],
    linear: Any,
) -> dict[str, Any]:
    rng = np.random.default_rng(17071)
    n = 32
    truth_rank = 2
    fit_rank = 3
    z = rng.normal(size=(n, truth_rank))

    feature_counts = {"RNA": 24, "METHYLATION": 32, "CNV": 20}
    views: dict[str, np.ndarray] = {}
    selected: dict[str, np.ndarray] = {}

    for view in VIEWS:
        p = feature_counts[view]
        weights = rng.normal(scale=0.7, size=(p, truth_rank))
        matrix = (
            z @ weights.T
            + rng.normal(scale=0.6, size=(n, p))
        )

        if view == "METHYLATION":
            mask = rng.random(matrix.shape) < 0.05
            matrix[mask] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(matrix, axis=0)
            scale = np.nanstd(matrix, axis=0, ddof=1)

        scale[~np.isfinite(scale) | (scale <= 1e-8)] = 1.0
        views[view] = (matrix - mean) / scale
        selected[view] = np.arange(p, dtype=int)

    fit = fit_mofa(
        training_views=views,
        selected_feature_indices=selected,
        rank=fit_rank,
        seed=17071,
        frozen_protocol=frozen_protocol,
        sample_ids=[f"FIXTURE_{index:03d}" for index in range(n)],
    )

    projection = gaussian_posterior_project(
        views, fit.weights, fit.tau, VIEWS
    )

    if projection.shape != (n, fit_rank):
        raise RuntimeError("Reference projection shape mismatch.")
    if not np.isfinite(projection).all():
        raise RuntimeError("Reference projection is non-finite.")

    reconstruction_mse: dict[str, float] = {}
    for view in VIEWS:
        predicted = projection @ fit.weights[view].T
        observed = views[view]
        finite = np.isfinite(observed)
        reconstruction_mse[view] = float(
            np.mean((predicted[finite] - observed[finite]) ** 2)
        )

    if not all(math.isfinite(value) for value in reconstruction_mse.values()):
        raise RuntimeError("Reference reconstruction metric is non-finite.")

    return {
        "fixture_id": "05c2b-installed-mofapy2-reference-v1",
        "seed": 17071,
        "n_samples": n,
        "rank": fit_rank,
        "feature_counts": feature_counts,
        "model_trained_flag": fit.audit["model_trained_flag"],
        "n_retained_factors": fit.audit["n_retained_factors"],
        "n_elbo_records": fit.audit["n_elbo_records"],
        "elbo_final": fit.audit["elbo_final"],
        "projection_effective_rank": linear.effective_rank(projection),
        "reconstruction_mse": reconstruction_mse,
        "weights_shapes": {
            view: list(fit.weights[view].shape) for view in VIEWS
        },
        "tau_shapes": {
            view: list(fit.tau[view].shape) for view in VIEWS
        },
        "status": "PASS",
    }


# ---------------------------------------------------------------------
# Existing frozen result verification
# ---------------------------------------------------------------------

def verify_existing_lock() -> None:
    lock = load_json(OUTPUT_LOCK)
    failures: list[str] = []

    if clean_text(lock.get("script_version")) != SCRIPT_VERSION:
        failures.append("script version mismatch")

    for relative_name, record in lock.get("generated_files", {}).items():
        candidate = resolve_project_path(relative_name)
        if not candidate.exists():
            failures.append(f"missing output: {relative_name}")
            continue
        if int(candidate.stat().st_size) != int(record["size_bytes"]):
            failures.append(f"size mismatch: {relative_name}")
            continue
        if sha256_file(candidate) != clean_text(record["sha256"]):
            failures.append(f"SHA256 mismatch: {relative_name}")

    shard_record = lock.get("shard_manifest", {})
    shard_text = clean_text(shard_record.get("path"))
    if not shard_text:
        failures.append("shard manifest missing from lock")
    else:
        shard_manifest_path = resolve_project_path(shard_text)
        if not shard_manifest_path.exists():
            failures.append("shard manifest missing")
        else:
            shard_manifest = pd.read_csv(
                shard_manifest_path, dtype=str
            ).fillna("")
            if shard_manifest.shape[0] != EXPECTED_SHARDS:
                failures.append("shard manifest row-count mismatch")
            else:
                for _, row in shard_manifest.iterrows():
                    candidate = resolve_project_path(row["shard_path"])
                    if not candidate.exists():
                        failures.append(f"missing shard: {row['shard_path']}")
                        break
                    if int(candidate.stat().st_size) != int(
                        float(row["size_bytes"])
                    ):
                        failures.append(
                            f"shard size mismatch: {row['shard_path']}"
                        )
                        break
                    if sha256_file(candidate) != clean_text(row["sha256"]):
                        failures.append(
                            f"shard hash mismatch: {row['shard_path']}"
                        )
                        break

    if failures:
        raise RuntimeError(
            "Existing 05c2b lock verification failed:\n"
            + "\n".join(f"- {failure}" for failure in failures)
        )

    print("=" * 88)
    print("Existing Paper 5 MOFA simulation benchmark: PASS")
    print("=" * 88)
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("No files were rewritten.")
    print("=" * 88)


def write_readme() -> None:
    lines = [
        "Paper 5 MOFA simulation baseline benchmark",
        "=" * 72,
        "",
        f"Script version: {SCRIPT_VERSION}",
        "",
        "Scope",
        "-----",
        "Confirmatory sable_mo_v2 architecture-screening bank only.",
        "720 datasets; MOFA K=5,10,15,20.",
        "Five held-out fits plus one full-data truth fit per dataset and rank.",
        f"Total frozen MOFA fits: {EXPECTED_FITS}.",
        "",
        "Held-out preprocessing",
        "----------------------",
        "Feature eligibility and pre-z-score variance ranking use training samples only.",
        "Primary feature budgets: RNA 5000; methylation 10000; CNV 5000.",
        "No median or KNN imputation is used for MOFA.",
        "Missing observations remain NaN.",
        "Centering and scaling parameters come from the training fold only.",
        "",
        "Out-of-sample inference",
        "-----------------------",
        "Raw X @ W is forbidden.",
        "Test scores use the 05c2a fixed-parameter Gaussian posterior mean.",
        "Held-out samples never update W, Tau, sparsity, ARD, or preprocessing.",
        "",
        "Truth evaluation",
        "----------------",
        "Shared factor/subspace recovery uses the same definitions as 05c1.",
        "Private recovery means view-specific private-signal capture, not an explicit MOFA private block.",
        "Technical-factor contamination is evaluated from known simulation truth.",
        "",
        "Transparency",
        "------------",
        "Failed fits remain in the audit.",
        "G09 is diagnostic for the MOFA baseline.",
        "No architecture is selected in this stage.",
        "Final gate-calibration simulations are not accessed.",
        "",
    ]
    OUTPUT_README.write_text("\n".join(lines), encoding="utf-8")


# Redefine the loader with explicit sys.modules registration for dataclass imports.
def load_linear_helpers() -> Any:
    require_file(LINEAR_SCRIPT)
    module_name = "paper5_05c1_helpers"
    spec = importlib.util.spec_from_file_location(module_name, LINEAR_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load 05c1 metric helpers.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 88)
    print("Paper 5 - fit MOFA simulation baselines")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Results directory: {OUTPUT_RESULTS_DIR}")
    print()

    OUTPUT_CONTRACT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SHARD_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_LOCK.exists():
        verify_existing_lock()
        return

    upstream = verify_05c2a_state()
    linear = load_linear_helpers()

    print("05a AI method contract: PASS")
    print("05a1 simulation/confound addendum: PASS")
    print("04d technical metadata preflight: PASS")
    print("05b v2 confirmatory simulation bank: PASS")
    print("05c1 frozen linear benchmark and split registry: PASS")
    print("05c2a MOFA projection protocol: PASS")
    print("Outcome access: FORBIDDEN")
    print("TARGET model fitting in this stage: NO")
    print("Neural model fitting in this stage: NO")
    print("Final gate-calibration data accessed: NO")
    print()

    protocol, protocol_hash = initialize_or_verify_protocol(upstream)

    print("Prospective 05c2b evaluation protocol: FROZEN")
    print(f"MOFA ranks: {list(MOFA_RANKS)}")
    print(f"Planned MOFA fits: {EXPECTED_FITS:,}")
    print(f"Resumable dataset x rank shards: {EXPECTED_SHARDS:,}")
    print("Exact 05c1 held-out folds reused: YES")
    print("Raw X @ W held-out projection: FORBIDDEN")
    print("Test-sample MOFA refitting: FORBIDDEN")
    print()

    write_json_atomic(upstream["environment"], OUTPUT_ENVIRONMENT)

    print("Running installed-mofapy2 reference fixture...")
    fixture = run_reference_fixture(upstream["protocol"], linear)
    write_json_atomic(fixture, OUTPUT_REFERENCE_FIXTURE)
    print("Installed-mofapy2 reference fixture: PASS")
    print("W/Z/Tau expectation extraction: PASS")
    print("Gaussian held-out projection after a real MOFA fit: PASS")
    print()

    seed_lookup = build_seed_lookup(upstream["seeds"])
    split_lookup = build_split_lookup(upstream["splits"])

    manifest = upstream["manifest"].copy()
    manifest = manifest.sort_values(
        ["scenario_id", "replicate_index"], kind="mergesort"
    ).reset_index(drop=True)

    write_json_atomic(
        {
            "script_version": SCRIPT_VERSION,
            "created_at_utc": utc_now_iso(),
            "protocol_sha256": protocol_hash,
            "n_datasets": int(manifest.shape[0]),
            "ranks": list(MOFA_RANKS),
            "planned_rank_shards": EXPECTED_SHARDS,
            "planned_mofa_fits": EXPECTED_FITS,
            "status": "IN_PROGRESS",
        },
        OUTPUT_IN_PROGRESS,
    )

    shards: list[dict[str, Any]] = []
    shard_manifest_rows: list[dict[str, Any]] = []
    current = 0
    generated_count = 0
    generated_seconds = 0.0

    for _, row in manifest.iterrows():
        key = (
            clean_text(row["scenario_id"]),
            int(float(row["replicate_index"])),
            int(float(row["simulation_seed"])),
        )
        if key not in split_lookup:
            raise RuntimeError(f"Missing frozen 05c1 folds for {key}.")
        folds = split_lookup[key]

        for rank in MOFA_RANKS:
            current += 1
            started = time.perf_counter()

            shard, status = materialize_shard(
                row=row,
                rank=rank,
                folds=folds,
                seed_lookup=seed_lookup,
                frozen_protocol=upstream["protocol"],
                truth_threshold=upstream["truth_threshold"],
                protocol_hash=protocol_hash,
                linear=linear,
            )

            elapsed = time.perf_counter() - started
            if status == "generated":
                generated_count += 1
                generated_seconds += elapsed

            shards.append(shard)
            candidate = shard_path(row, rank)
            shard_manifest_rows.append(
                {
                    "registry_label": clean_text(row["registry_label"]),
                    "scenario_id": clean_text(row["scenario_id"]),
                    "scenario_family": clean_text(row["scenario_family"]),
                    "replicate_index": int(float(row["replicate_index"])),
                    "simulation_seed": int(float(row["simulation_seed"])),
                    "rank": rank,
                    "model_id": f"MOFA_K{rank}",
                    "input_path": clean_text(row["path"]),
                    "input_sha256": clean_text(row["sha256"]),
                    "shard_path": relative_path(candidate),
                    "size_bytes": int(candidate.stat().st_size),
                    "sha256": sha256_file(candidate),
                    "generation_status": status,
                    "shard_wall_seconds": elapsed,
                }
            )

            if current == 1 or current % 10 == 0 or current == EXPECTED_SHARDS:
                eta = None
                if generated_count > 0:
                    eta = (
                        (EXPECTED_SHARDS - current)
                        * (generated_seconds / generated_count)
                        / 3600.0
                    )
                eta_text = f" | approximate ETA={eta:.1f}h" if eta is not None else ""
                print(
                    f"{current:04d}/{EXPECTED_SHARDS:04d} "
                    f"{clean_text(row['scenario_id'])} "
                    f"R{int(float(row['replicate_index'])):03d} "
                    f"MOFA_K{rank}: {status}{eta_text}"
                )

    shard_manifest = pd.DataFrame(shard_manifest_rows)
    shard_manifest.to_csv(
        OUTPUT_SHARD_MANIFEST, index=False, lineterminator="\n"
    )

    (
        preprocessing_table,
        fit_table,
        fold_table,
        dataset_table,
        factor_table,
        failure_table,
    ) = aggregate_shards(shards)

    if dataset_table.shape[0] != EXPECTED_SHARDS:
        raise RuntimeError("Dataset-rank output row count mismatch.")
    if fold_table.shape[0] != EXPECTED_SHARDS * CV_FOLDS:
        raise RuntimeError("Held-out fold output row count mismatch.")
    if fit_table.shape[0] != EXPECTED_FITS:
        raise RuntimeError("Fit-audit row count mismatch.")

    write_csv_gzip(preprocessing_table, OUTPUT_PREPROCESSING)
    write_csv_gzip(fit_table, OUTPUT_FIT_AUDIT)
    write_csv_gzip(fold_table, OUTPUT_FOLD_METRICS)
    write_csv_gzip(dataset_table, OUTPUT_DATASET_METRICS)
    write_csv_gzip(factor_table, OUTPUT_FACTOR_AUDIT)
    write_csv_gzip(failure_table, OUTPUT_FAILURE_AUDIT)

    scenario_summary = summarize_scenarios(dataset_table)
    scenario_summary.to_csv(
        OUTPUT_SCENARIO_SUMMARY,
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )

    paired, paired_summary = build_paired_linear_comparison(dataset_table)
    write_csv_gzip(paired, OUTPUT_PAIRED_LINEAR)
    paired_summary.to_csv(
        OUTPUT_PAIRED_LINEAR_SUMMARY,
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )

    write_readme()

    output_files = [
        OUTPUT_PROTOCOL,
        OUTPUT_REFERENCE_FIXTURE,
        OUTPUT_ENVIRONMENT,
        OUTPUT_SHARD_MANIFEST,
        OUTPUT_PREPROCESSING,
        OUTPUT_FIT_AUDIT,
        OUTPUT_FOLD_METRICS,
        OUTPUT_DATASET_METRICS,
        OUTPUT_FACTOR_AUDIT,
        OUTPUT_FAILURE_AUDIT,
        OUTPUT_SCENARIO_SUMMARY,
        OUTPUT_PAIRED_LINEAR,
        OUTPUT_PAIRED_LINEAR_SUMMARY,
        OUTPUT_README,
    ]

    n_scenarios = int(manifest["scenario_id"].nunique())
    n_fit_pass = int(fit_table["fit_status"].eq("PASS").sum())
    n_fit_fail = int(fit_table["fit_status"].eq("FAIL").sum())
    n_all_fold_pass = int(
        dataset_table["all_fold_fits_passed"].map(parse_bool).sum()
    )
    n_full_pass = int(dataset_table["full_fit_status"].eq("PASS").sum())

    result_manifest = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "scientific_state": {
            "outcomes_loaded": False,
            "clinical_labels_loaded": False,
            "biological_annotations_loaded": False,
            "TARGET_model_fitting_performed": False,
            "MOFA_simulation_model_fitting_performed": True,
            "neural_model_fitting_performed": False,
            "architecture_selected": False,
            "stability_gate_thresholds_retuned": False,
            "final_gate_calibration_data_used": False,
        },
        "scope": {
            "n_scenarios": n_scenarios,
            "n_datasets": int(manifest.shape[0]),
            "ranks": list(MOFA_RANKS),
            "n_dataset_rank_rows": int(dataset_table.shape[0]),
            "n_fold_rows": int(fold_table.shape[0]),
            "n_fit_audit_rows": int(fit_table.shape[0]),
            "n_fit_pass": n_fit_pass,
            "n_fit_fail": n_fit_fail,
            "n_dataset_rank_all_fold_pass": n_all_fold_pass,
            "n_full_truth_fit_pass": n_full_pass,
        },
        "protocol": {
            "path": relative_path(OUTPUT_PROTOCOL),
            "sha256": protocol_hash,
        },
        "upstream": {
            "05c2a_protocol": {
                "path": relative_path(MOFA_PROTOCOL),
                "sha256": upstream["protocol_sha256"],
            },
            "05c1_split_registry": {
                "path": relative_path(LINEAR_SPLITS),
                "sha256": upstream["split_sha256"],
            },
            "05c2a_seed_registry": {
                "path": relative_path(MOFA_SEEDS),
                "sha256": upstream["seed_sha256"],
            },
            "05c1_metric_source": {
                "path": relative_path(LINEAR_SCRIPT),
                "sha256": upstream["linear_script_sha256"],
            },
        },
        "outputs": {
            relative_path(candidate): {
                "sha256": sha256_file(candidate),
                "size_bytes": int(candidate.stat().st_size),
            }
            for candidate in output_files
        },
        "shard_manifest": {
            "path": relative_path(OUTPUT_SHARD_MANIFEST),
            "sha256": sha256_file(OUTPUT_SHARD_MANIFEST),
            "n_rows": int(shard_manifest.shape[0]),
        },
        "guardrails": [
            "Only frozen sable_mo_v2 architecture-screening simulations were used.",
            "The disclosed sable_mo_v1 pilot was not used.",
            "Final gate-calibration simulations were not accessed.",
            "Exact 05c1 held-out folds were reused.",
            "All held-out preprocessing was fitted on training samples only.",
            "Missing observations were not imputed for MOFA.",
            "Held-out factors used the frozen Gaussian posterior projection, never raw X @ W.",
            "Held-out samples never updated MOFA parameters.",
            "Complete simulation truth was used only for evaluation.",
            "Failed fits remain visible in the audit.",
            "No architecture or stability threshold was selected or changed.",
        ],
    }
    write_json_atomic(result_manifest, OUTPUT_RESULTS_MANIFEST)

    generated_files = {
        relative_path(candidate): {
            "sha256": sha256_file(candidate),
            "size_bytes": int(candidate.stat().st_size),
        }
        for candidate in [*output_files, OUTPUT_RESULTS_MANIFEST]
    }

    lock = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": (
            "FROZEN_MOFA_BASELINE_RESULTS_BEFORE_NEURAL_FITTING_AND_ARCHITECTURE_SELECTION"
        ),
        "scientific_state": result_manifest["scientific_state"],
        "scope": result_manifest["scope"],
        "result_manifest": {
            "path": relative_path(OUTPUT_RESULTS_MANIFEST),
            "sha256": sha256_file(OUTPUT_RESULTS_MANIFEST),
        },
        "shard_manifest": result_manifest["shard_manifest"],
        "generated_files": generated_files,
        "rewrite_policy": (
            "This zero-argument script never rewrites a frozen 05c2b result. "
            "Any scientifically justified change requires a new versioned stage."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    if OUTPUT_IN_PROGRESS.exists():
        OUTPUT_IN_PROGRESS.unlink()

    print()
    print("-" * 88)
    print("MOFA simulation baseline summary")
    print("-" * 88)
    print(f"Scenarios: {n_scenarios}")
    print(f"Datasets: {manifest.shape[0]}")
    print(f"Dataset x MOFA-rank rows: {dataset_table.shape[0]}")
    print(f"Held-out fold rows: {fold_table.shape[0]}")
    print(f"MOFA fit-audit rows: {fit_table.shape[0]}")
    print(f"Successful MOFA fits: {n_fit_pass}")
    print(f"Failed MOFA fits: {n_fit_fail}")
    print("Architecture selected: NO")
    print("Neural models fitted: NO")
    print("Final gate-calibration data accessed: NO")

    print()
    print("=" * 88)
    print("MOFA simulation baseline benchmark: PASS")
    print("=" * 88)
    print(f"Results manifest: {relative_path(OUTPUT_RESULTS_MANIFEST)}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print(
        "Next stage: freeze/install PyTorch environment, then 05d neural ablation benchmark"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
