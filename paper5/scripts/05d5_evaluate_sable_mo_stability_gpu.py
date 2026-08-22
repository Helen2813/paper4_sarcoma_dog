from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

# Deterministic CUDA GEMM configuration must be set before CUDA work begins.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pandas as pd
import torch


SCRIPT_VERSION = "05d5-evaluate-sable-mo-confirmatory-stability-gpu-v1-no-cli"
CONTRACT_VERSION = "paper5-neural-confirmatory-stability-v1"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

V4_VARIANT = "V4_SABLE_MO"
CONFIRMATORY_REPLICATES = tuple(range(11, 21))
EXPECTED_SCENARIOS = 36
EXPECTED_CONFIRMATORY_DATASETS = 360
STABILITY_RESAMPLES = 20
STABILITY_SUBSAMPLE_FRACTION = 0.80
EXPECTED_STABILITY_REGISTRY_ROWS = EXPECTED_CONFIRMATORY_DATASETS * STABILITY_RESAMPLES
EXPECTED_REFERENCE_FITS = EXPECTED_CONFIRMATORY_DATASETS
EXPECTED_STABILITY_OPTIMIZATION_RUNS = EXPECTED_CONFIRMATORY_DATASETS * (1 + STABILITY_RESAMPLES)
EXPECTED_FIT_AUDIT_ROWS_IF_ALL_PASS = EXPECTED_STABILITY_OPTIMIZATION_RUNS
EXPECTED_PREPROCESSING_ROWS_IF_ALL_PASS = EXPECTED_CONFIRMATORY_DATASETS * (3 + 3 * STABILITY_RESAMPLES)
EXPECTED_05D4_V4_FOLD_SEED_ROWS = EXPECTED_CONFIRMATORY_DATASETS * 5 * 2
EXPECTED_05D4_V4_FOLD_SEED_ROWS_PER_DATASET = 10

# -----------------------------------------------------------------------------
# Frozen upstream artifacts
# -----------------------------------------------------------------------------

CONTRACT_05A = PAPER5_ROOT / "method_contract" / "05a" / "ai_method_contract.json"
ENV_CONTRACT = PAPER5_ROOT / "method_contract" / "05d0" / "neural_environment_contract.json"
ENV_LOCK = PAPER5_ROOT / "method_contract" / "05d0" / "neural_environment_lock.json"
PROTOCOL_05D1 = PAPER5_ROOT / "method_contract" / "05d1" / "neural_simulation_evaluation_protocol.json"
LOSS_CONTRACT_05D1 = PAPER5_ROOT / "method_contract" / "05d1" / "neural_loss_and_architecture_contract.json"
LOCK_05D1 = PAPER5_ROOT / "method_contract" / "05d1" / "neural_simulation_protocol_lock.json"
PROFILE_GRID_05D1 = PAPER5_ROOT / "method_contract" / "05d1" / "neural_architecture_regularization_profile_grid.csv"
DATASET_ROLES_05D1 = PAPER5_ROOT / "method_contract" / "05d1" / "neural_simulation_dataset_role_registry.csv"
CONFIRMATORY_STABILITY_05D1 = PAPER5_ROOT / "method_contract" / "05d1" / "neural_confirmatory_stability_resample_registry.csv"

SIMULATION_MANIFEST = PAPER5_ROOT / "method_contract" / "05b_v2" / "architecture_screening_data_manifest.csv"
NEURAL_IMPL_SCRIPT = PAPER5_ROOT / "scripts" / "05d2v2_screen_neural_profiles_gpu_syncfree_pathfix.py"
LOCK_05D2 = PAPER5_ROOT / "method_contract" / "05d2_v2" / "neural_profile_screen_results_lock.json"
CONTRACT_05D3 = PAPER5_ROOT / "method_contract" / "05d3" / "selected_neural_profile.json"
LOCK_05D3 = PAPER5_ROOT / "method_contract" / "05d3" / "selected_neural_profile_lock.json"
EXPECTED_05D3_STATUS = "FROZEN_OUTCOME_BLIND_NEURAL_PROFILE_BEFORE_CONFIRMATORY_ABLATION"

RESULT_MANIFEST_05D4 = PAPER5_ROOT / "method_contract" / "05d4" / "neural_confirmatory_ablation_results_manifest.json"
LOCK_05D4 = PAPER5_ROOT / "method_contract" / "05d4" / "neural_confirmatory_ablation_results_lock.json"
METRICS_05D4 = PAPER5_ROOT / "results" / "simulation_benchmark" / "05d_neural" / "confirmatory_ablation" / "neural_confirmatory_fold_seed_metrics.csv.gz"
DATASET_VARIANT_05D4 = PAPER5_ROOT / "results" / "simulation_benchmark" / "05d_neural" / "confirmatory_ablation" / "neural_confirmatory_dataset_variant_metrics.csv.gz"

FINAL_GATE_DATA_ROOT = PAPER5_ROOT / "simulations" / "sable_mo_v2" / "final_gate_calibration"
FUTURE_RESULT_PATHS = (
    PAPER5_ROOT / "method_contract" / "05e",
    PAPER5_ROOT / "results" / "representation_benchmark",
)

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

OUTPUT_CONTRACT_DIR = PAPER5_ROOT / "method_contract" / "05d5"
OUTPUT_RESULTS_DIR = PAPER5_ROOT / "results" / "simulation_benchmark" / "05d_neural" / "confirmatory_stability"
OUTPUT_SHARD_DIR = OUTPUT_RESULTS_DIR / "shards" / "_compact"

OUTPUT_EPOCH_REGISTRY = OUTPUT_CONTRACT_DIR / "confirmatory_stability_epoch_registry.csv"
OUTPUT_PROTOCOL = OUTPUT_CONTRACT_DIR / "neural_confirmatory_stability_execution_protocol.json"
OUTPUT_FIXTURE = OUTPUT_CONTRACT_DIR / "neural_confirmatory_stability_reference_fixture.json"
OUTPUT_SHARD_MANIFEST = OUTPUT_CONTRACT_DIR / "neural_confirmatory_stability_shard_manifest.csv"
OUTPUT_RESULT_MANIFEST = OUTPUT_CONTRACT_DIR / "neural_confirmatory_stability_results_manifest.json"
OUTPUT_LOCK = OUTPUT_CONTRACT_DIR / "neural_confirmatory_stability_results_lock.json"
OUTPUT_README = OUTPUT_CONTRACT_DIR / "neural_confirmatory_stability_README.txt"
OUTPUT_IN_PROGRESS = OUTPUT_CONTRACT_DIR / ".neural_confirmatory_stability_in_progress.json"

OUTPUT_DATASET_SUMMARY = OUTPUT_RESULTS_DIR / "neural_confirmatory_stability_dataset_summary.csv.gz"
OUTPUT_FACTOR_AUDIT = OUTPUT_RESULTS_DIR / "neural_confirmatory_stability_factor_audit.csv.gz"
OUTPUT_FIT_AUDIT = OUTPUT_RESULTS_DIR / "neural_confirmatory_stability_fit_audit.csv.gz"
OUTPUT_PREPROCESSING_AUDIT = OUTPUT_RESULTS_DIR / "neural_confirmatory_stability_preprocessing_audit.csv.gz"
OUTPUT_SCENARIO_SUMMARY = OUTPUT_RESULTS_DIR / "neural_confirmatory_stability_scenario_summary.csv"
OUTPUT_OVERALL_SUMMARY = OUTPUT_RESULTS_DIR / "neural_confirmatory_stability_overall_summary.csv"


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "na"} else text


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


def load_json(path: Path) -> dict[str, Any]:
    require_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tmp_", suffix=".json", dir=str(path.parent)
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


def write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tmp_", suffix=".txt", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_csv_gzip(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression: Any = {"method": "gzip", "compresslevel": 6, "mtime": 0}
    try:
        table.to_csv(path, index=False, float_format="%.12g", compression=compression)
    except TypeError:
        table.to_csv(path, index=False, float_format="%.12g", compression="gzip")


def file_record(path: Path) -> dict[str, Any]:
    require_file(path)
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def is_nonempty_path(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.stat().st_size > 0
    return any(item.is_file() for item in path.rglob("*"))


def load_module(path: Path, module_name: str) -> Any:
    require_file(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def verify_generated_files(lock: dict[str, Any], label: str) -> None:
    failures: list[str] = []
    for relative_name, record in lock.get("generated_files", {}).items():
        candidate = resolve_project_path(relative_name)
        if not candidate.exists():
            failures.append(f"missing: {relative_name}")
            continue
        expected_size = record.get("size_bytes")
        if expected_size is not None and int(candidate.stat().st_size) != int(expected_size):
            failures.append(f"size mismatch: {relative_name}")
            continue
        expected_hash = clean_text(record.get("sha256"))
        if expected_hash and sha256_file(candidate) != expected_hash:
            failures.append(f"SHA256 mismatch: {relative_name}")
    if failures:
        raise RuntimeError(
            f"{label} generated-file verification failed:\n" + "\n".join(f"- {x}" for x in failures)
        )


def normalized_path_key(value: Any) -> str:
    return clean_text(value).replace("\\", "/").lower()


def assert_cuda_runtime(neural: Any, env_contract: dict[str, Any]) -> None:
    neural.assert_cuda_runtime()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is unavailable. Quiet CPU fallback is forbidden.")

    frozen = env_contract.get("software_freeze", {})
    frozen_torch = clean_text(frozen.get("torch_version"))
    frozen_cuda = clean_text(frozen.get("torch_cuda_build_version"))
    frozen_gpu = clean_text(env_contract.get("device_freeze", {}).get("gpu_name"))

    if clean_text(torch.__version__) != frozen_torch:
        raise RuntimeError(f"Active PyTorch differs from 05d0: {torch.__version__} vs {frozen_torch}")
    if clean_text(torch.version.cuda) != frozen_cuda:
        raise RuntimeError(f"Active CUDA build differs from 05d0: {torch.version.cuda} vs {frozen_cuda}")
    if clean_text(torch.cuda.get_device_name(0)) != frozen_gpu:
        raise RuntimeError(
            f"Active GPU differs from 05d0: {torch.cuda.get_device_name(0)} vs {frozen_gpu}"
        )


def build_manifest_lookup(table: pd.DataFrame) -> dict[tuple[str, int, int], pd.Series]:
    output: dict[tuple[str, int, int], pd.Series] = {}
    for _, row in table.iterrows():
        key = (
            clean_text(row["scenario_id"]),
            int(float(row["replicate_index"])),
            int(float(row["simulation_seed"])),
        )
        if key in output:
            raise RuntimeError(f"Duplicate simulation manifest key: {key}")
        output[key] = row
    return output


def dataset_metadata(row: pd.Series, dataset_path: Path, observed_hash: str) -> dict[str, Any]:
    return {
        "registry_label": clean_text(row.get("registry_label")),
        "scenario_id": clean_text(row["scenario_id"]),
        "scenario_origin": clean_text(row.get("scenario_origin")),
        "scenario_family": clean_text(row.get("scenario_family")),
        "calibration_role": clean_text(row.get("calibration_role")),
        "replicate_index": int(float(row["replicate_index"])),
        "simulation_seed": int(float(row["simulation_seed"])),
        "n_samples": int(float(row["n_samples"])),
        "n_true_biological_shared_factors": int(float(row["n_biological_shared_factors"])),
        "n_true_private_factors_per_view": int(float(row["n_private_factors_per_view"])),
        "n_true_technical_shared_factors": int(float(row["n_technical_shared_factors"])),
        "factor_level_identifiable": parse_bool(row["factor_level_identifiable"]),
        "subspace_level_identifiable": parse_bool(row["subspace_level_identifiable"]),
        "primary_truth_target": clean_text(row["primary_truth_target"]),
        "input_path": relative_path(dataset_path),
        "input_sha256": observed_hash,
    }


def finite_values(values: Iterable[Any]) -> np.ndarray:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


# -----------------------------------------------------------------------------
# 05d4 -> 05d5 deterministic epoch bridge
# -----------------------------------------------------------------------------

def build_epoch_registry(
    raw_05d4: pd.DataFrame,
    dataset_05d4: pd.DataFrame,
    profile: Any,
) -> pd.DataFrame:
    """
    05d1 froze one full-data reference fit + 20 resample fits per confirmatory
    dataset, but did not explicitly state how the ten 05d4 fold/seed-selected
    epoch counts should be collapsed to the one fixed epoch count needed by the
    full-data 05d5 stability refits.

    This execution bridge is frozen *before any 05d5 stability result exists*.
    It uses only the 05d4 V4 selected_epoch diagnostics, not any utility/truth/
    stability/outcome metric: per dataset, take the arithmetic mean of the ten
    frozen fold-seed selected epochs and round to nearest integer, half-up.
    """
    required_raw = {
        "scenario_id",
        "replicate_index",
        "simulation_seed",
        "variant_id",
        "profile_id",
        "fold_index",
        "base_model_seed",
        "selected_epoch",
        "shard_status",
        "stability_gate_applied_in_05d4",
    }
    missing = required_raw - set(raw_05d4.columns)
    if missing:
        raise RuntimeError("05d4 raw metric table missing: " + ", ".join(sorted(missing)))

    v4 = raw_05d4[raw_05d4["variant_id"].eq(V4_VARIANT)].copy()
    if v4.shape[0] != EXPECTED_05D4_V4_FOLD_SEED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_05D4_V4_FOLD_SEED_ROWS:,} V4 05d4 rows; found {v4.shape[0]:,}."
        )
    if not v4["shard_status"].eq("PASS").all():
        raise RuntimeError("05d5 requires all V4 05d4 fold-seed rows to be PASS.")
    if not v4["profile_id"].eq(profile.profile_id).all():
        raise RuntimeError("05d4 V4 rows do not all use the frozen 05d3 profile.")
    if v4["stability_gate_applied_in_05d4"].map(parse_bool).any():
        raise RuntimeError("05d4 unexpectedly applied the V4 stability gate.")

    required_dataset = {
        "scenario_id",
        "replicate_index",
        "simulation_seed",
        "variant_id",
        "profile_id",
        "selected_epoch_mean",
        "selected_epoch_sd",
        "n_fold_seed_evaluations",
        "n_passed_fold_seed_evaluations",
        "n_failed_fold_seed_evaluations",
    }
    missing = required_dataset - set(dataset_05d4.columns)
    if missing:
        raise RuntimeError("05d4 dataset-variant table missing: " + ", ".join(sorted(missing)))

    v4_dataset = dataset_05d4[dataset_05d4["variant_id"].eq(V4_VARIANT)].copy()
    if v4_dataset.shape[0] != EXPECTED_CONFIRMATORY_DATASETS:
        raise RuntimeError("05d4 V4 dataset summary does not contain 360 rows.")

    dataset_lookup: dict[tuple[str, int, int], pd.Series] = {}
    for _, row in v4_dataset.iterrows():
        key = (
            clean_text(row["scenario_id"]),
            int(float(row["replicate_index"])),
            int(float(row["simulation_seed"])),
        )
        if key in dataset_lookup:
            raise RuntimeError(f"Duplicate V4 dataset summary key: {key}")
        dataset_lookup[key] = row

    rows: list[dict[str, Any]] = []
    grouped = v4.groupby(["scenario_id", "replicate_index", "simulation_seed"], sort=True)
    for keys, part in grouped:
        key = (clean_text(keys[0]), int(float(keys[1])), int(float(keys[2])))
        if part.shape[0] != EXPECTED_05D4_V4_FOLD_SEED_ROWS_PER_DATASET:
            raise RuntimeError(f"{key}: expected exactly 10 V4 fold-seed selected epochs.")
        if set(pd.to_numeric(part["fold_index"], errors="raise").astype(int)) != set(range(1, 6)):
            raise RuntimeError(f"{key}: V4 05d4 fold coverage is not exactly 1..5.")
        if set(pd.to_numeric(part["base_model_seed"], errors="raise").astype(int)) != {1103, 2207}:
            raise RuntimeError(f"{key}: V4 05d4 base model seeds differ from 1103/2207.")
        fold_seed_pairs = part[["fold_index", "base_model_seed"]].copy()
        fold_seed_pairs["fold_index"] = pd.to_numeric(fold_seed_pairs["fold_index"], errors="raise").astype(int)
        fold_seed_pairs["base_model_seed"] = pd.to_numeric(fold_seed_pairs["base_model_seed"], errors="raise").astype(int)
        if fold_seed_pairs.duplicated().any():
            raise RuntimeError(f"{key}: duplicate V4 fold/base-seed row in 05d4 metrics.")

        epochs = pd.to_numeric(part["selected_epoch"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(epochs).all():
            raise RuntimeError(f"{key}: non-finite V4 selected epoch.")
        if not np.allclose(epochs, np.round(epochs), atol=0.0, rtol=0.0):
            raise RuntimeError(f"{key}: non-integer V4 selected epoch.")
        if np.any(epochs < 1) or np.any(epochs > int(profile.maximum_epochs)):
            raise RuntimeError(f"{key}: V4 selected epoch outside frozen profile bounds.")

        epoch_mean = float(np.mean(epochs))
        epoch_sd = float(np.std(epochs, ddof=1))
        epoch_min = int(np.min(epochs))
        epoch_max = int(np.max(epochs))
        selected_epoch = int(math.floor(epoch_mean + 0.5))
        selected_epoch = min(max(1, selected_epoch), int(profile.maximum_epochs))

        if key not in dataset_lookup:
            raise RuntimeError(f"{key}: missing V4 05d4 dataset summary.")
        drow = dataset_lookup[key]
        if int(float(drow["n_fold_seed_evaluations"])) != 10:
            raise RuntimeError(f"{key}: 05d4 dataset summary does not contain 10 evaluations.")
        if int(float(drow["n_passed_fold_seed_evaluations"])) != 10 or int(
            float(drow["n_failed_fold_seed_evaluations"])
        ) != 0:
            raise RuntimeError(f"{key}: 05d4 V4 dataset summary is not 10 PASS / 0 FAIL.")
        if clean_text(drow["profile_id"]) != profile.profile_id:
            raise RuntimeError(f"{key}: 05d4 dataset summary profile mismatch.")

        observed_mean = float(drow["selected_epoch_mean"])
        observed_sd = float(drow["selected_epoch_sd"])
        if not math.isclose(observed_mean, epoch_mean, rel_tol=0.0, abs_tol=1e-8):
            raise RuntimeError(f"{key}: selected_epoch_mean does not replay from raw 05d4 rows.")
        if not math.isclose(observed_sd, epoch_sd, rel_tol=0.0, abs_tol=1e-8):
            raise RuntimeError(f"{key}: selected_epoch_sd does not replay from raw 05d4 rows.")

        rows.append(
            {
                "scenario_id": key[0],
                "replicate_index": key[1],
                "simulation_seed": key[2],
                "variant_id": V4_VARIANT,
                "profile_id": profile.profile_id,
                "n_05d4_fold_seed_selected_epochs": 10,
                "selected_epoch_mean_05d4": epoch_mean,
                "selected_epoch_sd_05d4": epoch_sd,
                "selected_epoch_min_05d4": epoch_min,
                "selected_epoch_max_05d4": epoch_max,
                "selected_epoch_for_05d5": selected_epoch,
                "rounding_rule": "nearest_integer_half_up=floor(mean+0.5)",
                "source": relative_path(METRICS_05D4),
                "performance_metrics_used_to_choose_epoch": False,
                "stability_results_used_to_choose_epoch": False,
                "TARGET_or_outcomes_used_to_choose_epoch": False,
            }
        )

    output = pd.DataFrame(rows).sort_values(
        ["scenario_id", "replicate_index", "simulation_seed"], kind="mergesort"
    ).reset_index(drop=True)
    if output.shape[0] != EXPECTED_CONFIRMATORY_DATASETS:
        raise RuntimeError("05d5 epoch registry does not contain 360 datasets.")
    if output["scenario_id"].nunique() != EXPECTED_SCENARIOS:
        raise RuntimeError("05d5 epoch registry does not contain 36 scenarios.")
    if set(output["replicate_index"].astype(int).unique()) != set(CONFIRMATORY_REPLICATES):
        raise RuntimeError("05d5 epoch registry does not use exactly replicate indices 11-20.")
    return output


def initialize_or_verify_epoch_registry(table: pd.DataFrame) -> None:
    text = table.to_csv(index=False, float_format="%.12g", lineterminator="\n")
    if OUTPUT_EPOCH_REGISTRY.exists():
        existing = OUTPUT_EPOCH_REGISTRY.read_text(encoding="utf-8")
        if existing != text:
            raise RuntimeError(
                "Existing 05d5 epoch bridge differs from the deterministic replay of frozen 05d4 V4 epochs."
            )
    else:
        write_text_atomic(text, OUTPUT_EPOCH_REGISTRY)


def build_epoch_lookup(table: pd.DataFrame) -> dict[tuple[str, int, int], dict[str, Any]]:
    output: dict[tuple[str, int, int], dict[str, Any]] = {}
    for _, row in table.iterrows():
        key = (
            clean_text(row["scenario_id"]),
            int(row["replicate_index"]),
            int(row["simulation_seed"]),
        )
        if key in output:
            raise RuntimeError(f"Duplicate 05d5 epoch key: {key}")
        output[key] = row.to_dict()
    return output


# -----------------------------------------------------------------------------
# Frozen 05d1 confirmatory stability registry
# -----------------------------------------------------------------------------

def build_stability_lookup(
    table: pd.DataFrame,
    confirmatory_roles: pd.DataFrame,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    required = {
        "scenario_id",
        "replicate_index",
        "simulation_seed",
        "variant_id",
        "profile_id",
        "reference_fit_seed",
        "resample_index",
        "subsample_fraction",
        "n_subsample",
        "subsample_seed",
        "resample_fit_seed",
        "device",
    }
    missing = required - set(table.columns)
    if missing:
        raise RuntimeError("05d1 confirmatory stability registry missing: " + ", ".join(sorted(missing)))
    if table.shape[0] != EXPECTED_STABILITY_REGISTRY_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_STABILITY_REGISTRY_ROWS:,} 05d1 confirmatory stability rows; found {table.shape[0]:,}."
        )

    role_lookup = {
        (
            clean_text(row["scenario_id"]),
            int(float(row["replicate_index"])),
            int(float(row["simulation_seed"])),
        ): int(float(row["n_samples"]))
        for _, row in confirmatory_roles.iterrows()
    }

    output: dict[tuple[str, int, int], dict[str, Any]] = {}
    grouped = table.groupby(["scenario_id", "replicate_index", "simulation_seed"], sort=True)
    for keys, part in grouped:
        key = (clean_text(keys[0]), int(float(keys[1])), int(float(keys[2])))
        part = part.copy()
        part["resample_index"] = pd.to_numeric(part["resample_index"], errors="raise").astype(int)
        part = part.sort_values("resample_index", kind="mergesort")

        if key not in role_lookup:
            raise RuntimeError(f"05d1 stability registry contains non-confirmatory dataset: {key}")
        if part.shape[0] != STABILITY_RESAMPLES:
            raise RuntimeError(f"{key}: stability registry does not contain exactly 20 resamples.")
        if list(part["resample_index"]) != list(range(1, STABILITY_RESAMPLES + 1)):
            raise RuntimeError(f"{key}: resample indices are not exactly 1..20.")
        if not part["variant_id"].eq(V4_VARIANT).all():
            raise RuntimeError(f"{key}: confirmatory stability registry contains a non-V4 variant.")
        if not part["profile_id"].eq("FROZEN_BY_05D3").all():
            raise RuntimeError(f"{key}: confirmatory stability registry profile placeholder changed.")
        if not part["device"].eq("cuda:0").all():
            raise RuntimeError(f"{key}: confirmatory stability registry device is not cuda:0.")

        fractions = pd.to_numeric(part["subsample_fraction"], errors="raise").to_numpy(dtype=float)
        if not np.allclose(fractions, STABILITY_SUBSAMPLE_FRACTION, atol=0.0, rtol=0.0):
            raise RuntimeError(f"{key}: stability subsample fraction changed from 0.80.")

        expected_n_subsample = max(2, int(math.floor(role_lookup[key] * STABILITY_SUBSAMPLE_FRACTION)))
        n_subsamples = set(pd.to_numeric(part["n_subsample"], errors="raise").astype(int))
        if n_subsamples != {expected_n_subsample}:
            raise RuntimeError(
                f"{key}: n_subsample differs from floor(0.80*n)={expected_n_subsample}."
            )

        reference_seeds = set(pd.to_numeric(part["reference_fit_seed"], errors="raise").astype(int))
        if len(reference_seeds) != 1:
            raise RuntimeError(f"{key}: reference_fit_seed is not unique.")
        if pd.to_numeric(part["subsample_seed"], errors="raise").astype(int).duplicated().any():
            raise RuntimeError(f"{key}: duplicate subsample seed.")
        if pd.to_numeric(part["resample_fit_seed"], errors="raise").astype(int).duplicated().any():
            raise RuntimeError(f"{key}: duplicate resample fit seed.")

        output[key] = {
            "reference_fit_seed": int(next(iter(reference_seeds))),
            "rows": part,
        }

    if len(output) != EXPECTED_CONFIRMATORY_DATASETS:
        raise RuntimeError("05d1 confirmatory stability lookup does not contain 360 dataset groups.")
    return output


# -----------------------------------------------------------------------------
# Upstream verification
# -----------------------------------------------------------------------------

def verify_upstream_state(neural: Any) -> dict[str, Any]:
    required_paths = [
        CONTRACT_05A,
        ENV_CONTRACT,
        ENV_LOCK,
        PROTOCOL_05D1,
        LOSS_CONTRACT_05D1,
        LOCK_05D1,
        PROFILE_GRID_05D1,
        DATASET_ROLES_05D1,
        CONFIRMATORY_STABILITY_05D1,
        SIMULATION_MANIFEST,
        NEURAL_IMPL_SCRIPT,
        LOCK_05D2,
        CONTRACT_05D3,
        LOCK_05D3,
        RESULT_MANIFEST_05D4,
        LOCK_05D4,
        METRICS_05D4,
        DATASET_VARIANT_05D4,
    ]
    for path in required_paths:
        require_file(path)

    if FINAL_GATE_DATA_ROOT.exists() and any(item.is_file() for item in FINAL_GATE_DATA_ROOT.rglob("*")):
        raise RuntimeError(
            "Final gate-calibration simulations already exist. 05d5 is frozen to confirmatory architecture simulations only."
        )
    future = [relative_path(path) for path in FUTURE_RESULT_PATHS if is_nonempty_path(path)]
    if future:
        raise RuntimeError("05d5 must precede 05e/final representation results. Found:\n- " + "\n- ".join(future))

    env_contract = load_json(ENV_CONTRACT)
    env_lock = load_json(ENV_LOCK)
    if clean_text(env_lock.get("status")) != "FROZEN_NEURAL_RUNTIME_ENVIRONMENT_BEFORE_NEURAL_MODEL_FITTING":
        raise RuntimeError("Unexpected 05d0 lock status.")
    verify_generated_files(env_lock, "05d0")
    expected_env_hash = clean_text(env_lock.get("contract", {}).get("sha256"))
    if not expected_env_hash or sha256_file(ENV_CONTRACT) != expected_env_hash:
        raise RuntimeError("05d0 environment contract hash mismatch.")
    assert_cuda_runtime(neural, env_contract)

    protocol_05d1 = load_json(PROTOCOL_05D1)
    loss_contract = load_json(LOSS_CONTRACT_05D1)
    lock_05d1 = load_json(LOCK_05D1)
    if clean_text(lock_05d1.get("status")) != "FROZEN_NEURAL_SIMULATION_PROTOCOL_BEFORE_NEURAL_MODEL_FITTING":
        raise RuntimeError("Unexpected 05d1 lock status.")
    verify_generated_files(lock_05d1, "05d1")
    expected_05d1_hash = clean_text(lock_05d1.get("protocol", {}).get("sha256"))
    if not expected_05d1_hash or sha256_file(PROTOCOL_05D1) != expected_05d1_hash:
        raise RuntimeError("05d1 protocol hash mismatch.")

    for relative_name, expected_hash in protocol_05d1.get("upstream_hashes", {}).items():
        candidate = resolve_project_path(relative_name)
        require_file(candidate)
        if sha256_file(candidate) != clean_text(expected_hash):
            raise RuntimeError(f"05d1 frozen upstream hash mismatch: {relative_name}")

    stages = [clean_text(item.get("stage")) for item in protocol_05d1.get("stage_order", [])]
    if stages != ["05d2", "05d3", "05d4", "05d5", "05e"]:
        raise RuntimeError(f"Frozen 05d1 stage order changed: {stages}")

    cscope = protocol_05d1.get("confirmatory_ablation", {})
    if int(cscope.get("V4_confirmatory_stability_reference_fits", -1)) != EXPECTED_REFERENCE_FITS:
        raise RuntimeError("05d1 V4 confirmatory stability reference-fit count changed.")
    if int(cscope.get("V4_confirmatory_stability_resample_fits", -1)) != EXPECTED_STABILITY_REGISTRY_ROWS:
        raise RuntimeError("05d1 V4 confirmatory stability resample-fit count changed.")
    if normalized_path_key(cscope.get("stability_resample_registry")) != normalized_path_key(relative_path(CONFIRMATORY_STABILITY_05D1)):
        raise RuntimeError("05d1 confirmatory stability registry path changed.")

    stability_contract = loss_contract.get("stability_gate", {})
    if bool(stability_contract.get("training_component", True)):
        raise RuntimeError("05d1 stability gate unexpectedly became a training component.")
    if not bool(stability_contract.get("resampling_component", False)):
        raise RuntimeError("05d1 stability gate is no longer a resampling component.")
    if int(stability_contract.get("resamples_per_dataset", -1)) != STABILITY_RESAMPLES:
        raise RuntimeError("05d1 stability resample count changed.")
    if float(stability_contract.get("subsample_fraction", float("nan"))) != STABILITY_SUBSAMPLE_FRACTION:
        raise RuntimeError("05d1 stability subsample fraction changed.")

    lock_05d2 = load_json(LOCK_05D2)
    if clean_text(lock_05d2.get("status")) != "FROZEN_NEURAL_PROFILE_SCREEN_RESULTS_BEFORE_CONFIRMATORY_ABLATION":
        raise RuntimeError("Unexpected 05d2 v2 lock status.")
    verify_generated_files(lock_05d2, "05d2 v2")

    lock_05d3 = load_json(LOCK_05D3)
    if clean_text(lock_05d3.get("status")) != EXPECTED_05D3_STATUS:
        raise RuntimeError("Unexpected 05d3 lock status.")
    verify_generated_files(lock_05d3, "05d3")
    expected_05d3_hash = clean_text(lock_05d3.get("contract", {}).get("sha256"))
    if not expected_05d3_hash or sha256_file(CONTRACT_05D3) != expected_05d3_hash:
        raise RuntimeError("05d3 selected-profile contract hash mismatch.")

    contract_05d3 = load_json(CONTRACT_05D3)
    if clean_text(contract_05d3.get("status")) != EXPECTED_05D3_STATUS:
        raise RuntimeError("05d3 contract status mismatch.")
    selected_profile = contract_05d3.get("selected_profile", {})
    selected_profile_id = clean_text(selected_profile.get("profile_id"))
    if not selected_profile_id or clean_text(lock_05d3.get("selected_profile_id")) != selected_profile_id:
        raise RuntimeError("05d3 selected profile ID missing or inconsistent.")

    profile_grid = pd.read_csv(PROFILE_GRID_05D1, dtype=str).fillna("")
    selected_rows = profile_grid[profile_grid["profile_id"].eq(selected_profile_id)]
    if selected_rows.shape[0] != 1:
        raise RuntimeError("05d3 selected profile is not unique in the 05d1 profile grid.")
    source_row = selected_profile.get("source_profile_grid_row", {})
    selected_row = selected_rows.iloc[0]
    for column in profile_grid.columns:
        if clean_text(source_row.get(column)) != clean_text(selected_row[column]):
            raise RuntimeError(f"05d3 selected profile differs from frozen 05d1 grid at {column}.")
    profile = neural.parse_profile(selected_row)

    lock_05d4 = load_json(LOCK_05D4)
    if clean_text(lock_05d4.get("status")) != "FROZEN_NEURAL_CONFIRMATORY_ABLATION_RESULTS_BEFORE_05D5":
        raise RuntimeError("05d4 is not a clean all-PASS frozen result ready for 05d5.")
    verify_generated_files(lock_05d4, "05d4")
    expected_05d4_manifest_hash = clean_text(lock_05d4.get("result_manifest", {}).get("sha256"))
    if not expected_05d4_manifest_hash or sha256_file(RESULT_MANIFEST_05D4) != expected_05d4_manifest_hash:
        raise RuntimeError("05d4 result manifest hash mismatch.")
    if clean_text(lock_05d4.get("profile_id")) != profile.profile_id:
        raise RuntimeError("05d4 lock profile differs from 05d3 frozen profile.")

    result_05d4 = load_json(RESULT_MANIFEST_05D4)
    state_05d4 = result_05d4.get("scientific_state", {})
    required_state = {
        "confirmatory_replicates_11_to_20_accessed": True,
        "confirmatory_neural_ablation_performed": True,
        "confirmatory_V4_stability_performed": False,
        "final_gate_calibration_data_used": False,
        "representation_method_selected": False,
        "TARGET_model_fitting_performed": False,
    }
    for key, expected in required_state.items():
        if bool(state_05d4.get(key, False)) != expected:
            raise RuntimeError(f"05d4 scientific state mismatch for {key}.")
    scope_05d4 = result_05d4.get("scope", {})
    if int(scope_05d4.get("passed_fold_seed_evaluations", -1)) != 21600:
        raise RuntimeError("05d4 passed fold-seed count is not 21,600.")
    if int(scope_05d4.get("failed_fold_seed_evaluations", -1)) != 0:
        raise RuntimeError("05d4 contains failed fold-seed evaluations.")

    # Ensure the exact sync-free V4 implementation reused by 05d5 is the same
    # helper whose hash was frozen into 05d4 provenance.
    for relative_name, expected_hash in result_05d4.get("upstream_hashes", {}).items():
        candidate = resolve_project_path(relative_name)
        require_file(candidate)
        if sha256_file(candidate) != clean_text(expected_hash):
            raise RuntimeError(f"05d4 frozen upstream hash mismatch before 05d5: {relative_name}")

    upstream_hashes_05d4 = {
        normalized_path_key(key): clean_text(value)
        for key, value in result_05d4.get("upstream_hashes", {}).items()
    }
    helper_key = normalized_path_key(relative_path(NEURAL_IMPL_SCRIPT))
    expected_helper_hash = upstream_hashes_05d4.get(helper_key)
    if not expected_helper_hash:
        raise RuntimeError("05d4 provenance does not contain the 05d2v2 helper hash.")
    if sha256_file(NEURAL_IMPL_SCRIPT) != expected_helper_hash:
        raise RuntimeError("05d2v2 helper changed after 05d4; refusing 05d5 execution.")

    roles = pd.read_csv(DATASET_ROLES_05D1, dtype=str, low_memory=False).fillna("")
    confirmatory_roles = roles[roles["neural_dataset_role"].eq("confirmatory_ablation")].copy()
    if confirmatory_roles.shape[0] != EXPECTED_CONFIRMATORY_DATASETS:
        raise RuntimeError("05d1 confirmatory dataset count is not 360.")
    confirmatory_roles["replicate_index"] = pd.to_numeric(
        confirmatory_roles["replicate_index"], errors="raise"
    ).astype(int)
    confirmatory_roles["simulation_seed"] = pd.to_numeric(
        confirmatory_roles["simulation_seed"], errors="raise"
    ).astype(int)
    if set(confirmatory_roles["replicate_index"].unique()) != set(CONFIRMATORY_REPLICATES):
        raise RuntimeError("Confirmatory roles are not exactly replicate indices 11-20.")
    if confirmatory_roles["scenario_id"].nunique() != EXPECTED_SCENARIOS:
        raise RuntimeError("Confirmatory roles do not contain 36 scenarios.")
    scenario_counts = confirmatory_roles.groupby("scenario_id")["replicate_index"].nunique()
    if not scenario_counts.eq(10).all():
        raise RuntimeError("Every scenario must contain exactly 10 confirmatory replicates.")

    manifest = pd.read_csv(SIMULATION_MANIFEST, dtype=str, low_memory=False).fillna("")
    if manifest.shape[0] != 720:
        raise RuntimeError("Frozen simulation manifest no longer contains 720 datasets.")
    manifest_lookup = build_manifest_lookup(manifest)
    for _, row in confirmatory_roles.iterrows():
        key = (clean_text(row["scenario_id"]), int(row["replicate_index"]), int(row["simulation_seed"]))
        if key not in manifest_lookup:
            raise RuntimeError(f"Confirmatory dataset missing from simulation manifest: {key}")

    stability_registry = pd.read_csv(CONFIRMATORY_STABILITY_05D1, dtype=str, low_memory=False).fillna("")
    stability_lookup = build_stability_lookup(stability_registry, confirmatory_roles)

    raw_05d4 = pd.read_csv(METRICS_05D4, dtype=str, low_memory=False).fillna("")
    dataset_05d4 = pd.read_csv(DATASET_VARIANT_05D4, dtype=str, low_memory=False).fillna("")
    epoch_registry = build_epoch_registry(raw_05d4, dataset_05d4, profile)

    thresholds = load_json(CONTRACT_05A).get("stability_gate", {}).get("thresholds", {})
    required_thresholds = {
        "primary_median_factor_correlation",
        "primary_q10_factor_correlation",
        "primary_inclusion_frequency",
        "minimum_factor_variance",
        "minimum_supported_modalities",
        "minimum_loading_energy_fraction_per_supported_modality",
    }
    if required_thresholds - set(thresholds):
        raise RuntimeError("05a stability-gate thresholds are incomplete.")

    upstream_paths = required_paths
    return {
        "env_contract": env_contract,
        "protocol_05d1": protocol_05d1,
        "loss_contract": loss_contract,
        "contract_05d3": contract_05d3,
        "result_05d4": result_05d4,
        "profile": profile,
        "confirmatory_roles": confirmatory_roles,
        "manifest_lookup": manifest_lookup,
        "stability_lookup": stability_lookup,
        "stability_thresholds": thresholds,
        "epoch_registry": epoch_registry,
        "hashes": {relative_path(path): sha256_file(path) for path in upstream_paths},
    }


# -----------------------------------------------------------------------------
# Prospective 05d5 execution protocol
# -----------------------------------------------------------------------------

def build_execution_protocol(upstream: dict[str, Any]) -> dict[str, Any]:
    profile = upstream["profile"]
    return {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "scientific_state_before_fit": {
            "outcomes_loaded": False,
            "clinical_labels_loaded": False,
            "biological_annotations_used_for_model_or_profile_selection": False,
            "TARGET_model_fitting_performed": False,
            "neural_profile_frozen": True,
            "confirmatory_neural_ablation_performed": True,
            "confirmatory_V4_stability_performed": False,
            "final_gate_calibration_data_used": False,
            "representation_method_selected": False,
        },
        "scope": {
            "variant": V4_VARIANT,
            "profile_id": profile.profile_id,
            "confirmatory_replicate_indices": list(CONFIRMATORY_REPLICATES),
            "confirmatory_datasets": EXPECTED_CONFIRMATORY_DATASETS,
            "scenarios": EXPECTED_SCENARIOS,
            "reference_full_data_fits": EXPECTED_REFERENCE_FITS,
            "stability_resamples_per_dataset": STABILITY_RESAMPLES,
            "stability_resample_fits": EXPECTED_STABILITY_REGISTRY_ROWS,
            "total_neural_optimization_runs": EXPECTED_STABILITY_OPTIMIZATION_RUNS,
        },
        "frozen_sources": {
            "05d1_confirmatory_stability_registry": file_record(CONFIRMATORY_STABILITY_05D1),
            "05d3_selected_profile": file_record(CONTRACT_05D3),
            "05d4_result_manifest": file_record(RESULT_MANIFEST_05D4),
            "05d4_V4_fold_seed_metrics": file_record(METRICS_05D4),
            "05d4_V4_dataset_summary": file_record(DATASET_VARIANT_05D4),
            "05d2v2_V4_implementation": file_record(NEURAL_IMPL_SCRIPT),
        },
        "selected_epoch_bridge": {
            "registry": file_record(OUTPUT_EPOCH_REGISTRY),
            "reason": (
                "05d1 froze one full-data reference + 20 stability resample fits per confirmatory dataset, "
                "but did not explicitly define how the ten 05d4 V4 fold/seed-selected epoch counts transfer "
                "to one fixed full-data stability epoch count. This execution-level bridge is frozen before "
                "any 05d5 stability result exists."
            ),
            "input": "the ten 05d4 V4 selected_epoch diagnostics per dataset only",
            "rule": "arithmetic mean across 5 folds x 2 frozen seeds, then nearest integer half-up via floor(mean+0.5)",
            "dataset_specific": True,
            "bounded_by_frozen_profile_maximum_epochs": True,
            "05d4_performance_metrics_used": False,
            "05d5_stability_metrics_used": False,
            "TARGET_or_outcomes_used": False,
        },
        "stability_contract": {
            "training_component": False,
            "post_fit_resampling_component": True,
            "subsample_fraction": STABILITY_SUBSAMPLE_FRACTION,
            "resamples_per_dataset": STABILITY_RESAMPLES,
            "reference_seed_source": relative_path(CONFIRMATORY_STABILITY_05D1),
            "subsample_seed_source": relative_path(CONFIRMATORY_STABILITY_05D1),
            "resample_fit_seed_source": relative_path(CONFIRMATORY_STABILITY_05D1),
            "preprocessing_refit_within_each_resample": True,
            "neural_refit_within_each_resample": True,
            "alignment": "standardize shared scores -> orthogonal Procrustes -> Hungarian factor matching -> absolute correlation",
            "thresholds": upstream["stability_thresholds"],
            "biological_truth_match_absolute_correlation": 0.50,
            "technical_truth_match_absolute_correlation": 0.50,
            "technical_factors_counted_as_false_biological": False,
            "false_biological_discovery_denominator": "stable nontechnical factors",
            "factor_level_primary_eligibility": "factor-level-identifiable datasets OR true biological null datasets",
            "rotational_nonnull_factor_results": "descriptive_only_not_primary",
        },
        "gpu_execution": {
            "device": "cuda:0",
            "quiet_cpu_fallback": False,
            "neural_dtype": "float32",
            "dual_PCA_dtype": "float64",
            "automatic_mixed_precision": False,
            "TF32": False,
            "torch_compile": False,
            "deterministic_algorithms": True,
            "CPU_allowed": [
                "file I/O",
                "pandas audit tables and aggregation",
                "fold/resample-local feature eligibility and ranking",
                "training-subset median imputation",
                "hashing/provenance",
            ],
        },
        "aggregation": {
            "dataset": "one full-data reference + 20 frozen 80% resamples",
            "scenario": "equal descriptive aggregation across the 10 confirmatory replicate datasets",
            "overall_subspace_metrics": "equal mean across 36 scenario means; SE unit is 36 scenario means",
            "factor_level_false_discovery_primary": "mean only across PASS datasets eligible for factor-level primary interpretation",
            "inferential_p_values": False,
            "representation_method_selection": False,
        },
        "resumability": {
            "unit": "one confirmatory simulation dataset",
            "n_shards": EXPECTED_CONFIRMATORY_DATASETS,
            "compact_SHA256_derived_filenames": True,
            "existing_shards_verified_by": [
                "script version",
                "prospective 05d5 protocol SHA256",
                "simulation input SHA256",
                "frozen profile ID",
                "scenario/replicate/simulation seed",
                "05d5 selected epoch",
            ],
        },
        "guardrails": [
            "Only confirmatory simulation replicate indices 11-20 are used.",
            "Only frozen V4 SABLE-MO is stability-resampled in 05d5.",
            "The 05d3-selected profile is not retuned.",
            "The 05a stability thresholds are not retuned.",
            "05d4 V4-minus-V3 is not treated as the stability-gate effect.",
            "Failed 05d5 datasets remain visible and are never silently omitted.",
            "TARGET matrices, outcomes, clinical labels, biological program annotations, and final-gate calibration data are not used.",
            "05d5 does not select the final representation method; that remains reserved for 05e.",
        ],
        "upstream_hashes": upstream["hashes"],
        "implementation_source": {
            "path": relative_path(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "next_required_stage_if_all_datasets_pass": "05e_compare_ablate_and_freeze_method.py",
    }


def protocol_core(protocol: dict[str, Any]) -> dict[str, Any]:
    output = json.loads(json.dumps(protocol))
    output.pop("created_at_utc", None)
    implementation = output.get("implementation_source")
    if isinstance(implementation, dict):
        implementation.pop("path", None)
    return output


def initialize_or_verify_protocol(upstream: dict[str, Any]) -> tuple[dict[str, Any], str]:
    proposed = build_execution_protocol(upstream)
    if OUTPUT_PROTOCOL.exists():
        existing = load_json(OUTPUT_PROTOCOL)
        existing_core = json.dumps(protocol_core(existing), sort_keys=True, separators=(",", ":"))
        proposed_core = json.dumps(protocol_core(proposed), sort_keys=True, separators=(",", ":"))
        if existing_core != proposed_core:
            raise RuntimeError(
                "Existing prospective 05d5 execution protocol differs from the current implementation. "
                "Refusing to alter a started confirmatory stability analysis."
            )
        protocol = existing
    else:
        write_json_atomic(proposed, OUTPUT_PROTOCOL)
        protocol = proposed
    return protocol, sha256_file(OUTPUT_PROTOCOL)


# -----------------------------------------------------------------------------
# Deterministic implementation fixture
# -----------------------------------------------------------------------------

def run_reference_fixture(neural: Any, linear: Any, profile: Any) -> dict[str, Any]:
    neural_fixture = neural.run_reference_fixture(profile)
    if clean_text(neural_fixture.get("status")) != "PASS":
        raise RuntimeError("05d5 inherited V4 CUDA fixture did not PASS.")

    rng = np.random.default_rng(20525)
    reference = rng.normal(size=(48, int(profile.shared_dim))).astype(np.float64)
    permutation = np.arange(int(profile.shared_dim))[::-1]
    signs = np.where(np.arange(int(profile.shared_dim)) % 2 == 0, 1.0, -1.0)
    resample = reference[:, permutation] * signs
    _, correlations, subspace = neural.align_resample_to_reference(linear, resample, reference)
    min_corr = float(np.min(correlations))
    if subspace is None or min_corr < 1.0 - 1e-10 or float(subspace) < 1.0 - 1e-10:
        raise RuntimeError("05d5 Procrustes/Hungarian stability alignment fixture failed.")

    return {
        "fixture_id": "paper5-05d5-v4-confirmatory-stability-fixture-v1",
        "profile_id": profile.profile_id,
        "device": "cuda:0",
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "inherited_V4_cuda_fixture": neural_fixture,
        "alignment_permuted_signed_latent_min_absolute_correlation": min_corr,
        "alignment_subspace_mean_canonical_correlation": float(subspace),
        "status": "PASS",
    }


# -----------------------------------------------------------------------------
# Resumable dataset-level stability shards
# -----------------------------------------------------------------------------

def shard_path(metadata: dict[str, Any], selected_epoch: int, protocol_hash: str) -> Path:
    key = (
        f"{metadata['scenario_id']}|{metadata['replicate_index']}|{metadata['simulation_seed']}|"
        f"{V4_VARIANT}|{metadata['profile_id']}|{metadata['input_sha256']}|E{selected_epoch}|{protocol_hash}"
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:28]
    return OUTPUT_SHARD_DIR / f"shard_{digest}.json"


def validate_existing_shard(
    shard: dict[str, Any],
    metadata: dict[str, Any],
    selected_epoch: int,
    protocol_hash: str,
) -> None:
    checks = {
        "script_version": (clean_text(shard.get("script_version")), SCRIPT_VERSION),
        "protocol_sha256": (clean_text(shard.get("protocol_sha256")), protocol_hash),
        "scenario_id": (clean_text(shard.get("scenario_id")), clean_text(metadata["scenario_id"])),
        "replicate_index": (int(shard.get("replicate_index", -1)), int(metadata["replicate_index"])),
        "simulation_seed": (int(shard.get("simulation_seed", -1)), int(metadata["simulation_seed"])),
        "variant_id": (clean_text(shard.get("variant_id")), V4_VARIANT),
        "profile_id": (clean_text(shard.get("profile_id")), clean_text(metadata["profile_id"])),
        "input_sha256": (clean_text(shard.get("input_sha256")), clean_text(metadata["input_sha256"])),
        "selected_epoch": (int(shard.get("selected_epoch", -1)), int(selected_epoch)),
    }
    failures = [name for name, (observed, expected) in checks.items() if observed != expected]
    if failures:
        raise RuntimeError("Existing 05d5 shard verification failed: " + ", ".join(failures))


def attach_metadata(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    profile: Any,
    selected_epoch: int,
    primary_eligible: bool,
) -> None:
    for row in rows:
        row.update(metadata)
        row.update(
            {
                "execution_stage": "05d5_confirmatory_V4_stability",
                "variant_id": V4_VARIANT,
                "profile_id": profile.profile_id,
                "architecture_profile": profile.architecture_profile,
                "regularization_profile": profile.regularization_profile,
                "shared_latent_dim": int(profile.shared_dim),
                "private_latent_dim_per_view": int(profile.private_dim),
                "selected_epoch_for_05d5": int(selected_epoch),
                "factor_level_metrics_primary_eligible": bool(primary_eligible),
            }
        )


def evaluate_dataset_stability(
    neural: Any,
    linear: Any,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    profile: Any,
    selected_epoch_record: dict[str, Any],
    stability_group: dict[str, Any],
    stability_thresholds: dict[str, Any],
) -> dict[str, Any]:
    selected_epoch = int(selected_epoch_record["selected_epoch_for_05d5"])
    primary_eligible = bool(
        metadata["factor_level_identifiable"] or metadata["n_true_biological_shared_factors"] == 0
    )

    torch.cuda.reset_peak_memory_stats(0)
    summary, factor_rows, fit_audits, preprocessing_rows = neural.stability_audit(
        linear=linear,
        arrays=arrays,
        profile=profile,
        selected_epoch=selected_epoch,
        stability_group=stability_group,
        stability_thresholds=stability_thresholds,
    )

    if len(fit_audits) != 1 + STABILITY_RESAMPLES:
        raise RuntimeError("Successful 05d5 dataset does not contain exactly 21 stability fit audits.")
    if len(preprocessing_rows) != 3 + 3 * STABILITY_RESAMPLES:
        raise RuntimeError("Successful 05d5 dataset does not contain exactly 63 preprocessing rows.")
    if len(factor_rows) != int(profile.shared_dim):
        raise RuntimeError("Successful 05d5 factor row count differs from frozen shared latent dimension.")
    if int(summary.get("n_stability_resamples", -1)) != STABILITY_RESAMPLES:
        raise RuntimeError("05d5 stability summary does not report 20 resamples.")
    if int(summary.get("selected_epoch_used_for_stability", -1)) != selected_epoch:
        raise RuntimeError("05d5 helper did not use the frozen dataset-specific stability epoch.")

    attach_metadata(fit_audits, metadata, profile, selected_epoch, primary_eligible)
    attach_metadata(preprocessing_rows, metadata, profile, selected_epoch, primary_eligible)
    attach_metadata(factor_rows, metadata, profile, selected_epoch, primary_eligible)
    for row in factor_rows:
        row["factor_level_interpretation"] = (
            "primary_factor_level"
            if primary_eligible
            else "descriptive_only_rotational_nonnull"
        )

    max_memory = int(torch.cuda.max_memory_allocated(0))
    dataset_row = {
        **metadata,
        "variant_id": V4_VARIANT,
        "profile_id": profile.profile_id,
        "architecture_profile": profile.architecture_profile,
        "regularization_profile": profile.regularization_profile,
        "shared_latent_dim": int(profile.shared_dim),
        "private_latent_dim_per_view": int(profile.private_dim),
        "hidden_dim": int(profile.hidden_dim),
        "factor_level_metrics_primary_eligible": primary_eligible,
        "selected_epoch_for_05d5": selected_epoch,
        "selected_epoch_mean_05d4": float(selected_epoch_record["selected_epoch_mean_05d4"]),
        "selected_epoch_sd_05d4": float(selected_epoch_record["selected_epoch_sd_05d4"]),
        "selected_epoch_min_05d4": int(selected_epoch_record["selected_epoch_min_05d4"]),
        "selected_epoch_max_05d4": int(selected_epoch_record["selected_epoch_max_05d4"]),
        "reference_fit_seed": int(stability_group["reference_fit_seed"]),
        "required_reference_fits": 1,
        "required_resample_fits": STABILITY_RESAMPLES,
        "all_required_fits_passed": True,
        "max_cuda_memory_allocated_bytes": max_memory,
        **summary,
        "shard_status": "PASS",
        "shard_error_type": "",
        "shard_error_message": "",
    }

    return {
        "script_version": SCRIPT_VERSION,
        "scenario_id": metadata["scenario_id"],
        "replicate_index": metadata["replicate_index"],
        "simulation_seed": metadata["simulation_seed"],
        "variant_id": V4_VARIANT,
        "profile_id": profile.profile_id,
        "input_path": metadata["input_path"],
        "input_sha256": metadata["input_sha256"],
        "selected_epoch": selected_epoch,
        "dataset_row": dataset_row,
        "factor_rows": factor_rows,
        "fit_audits": fit_audits,
        "preprocessing_rows": preprocessing_rows,
    }


def failed_shard_record(
    metadata: dict[str, Any],
    profile: Any,
    selected_epoch_record: dict[str, Any],
    stability_group: dict[str, Any],
    error: BaseException,
) -> dict[str, Any]:
    selected_epoch = int(selected_epoch_record["selected_epoch_for_05d5"])
    primary_eligible = bool(
        metadata["factor_level_identifiable"] or metadata["n_true_biological_shared_factors"] == 0
    )
    dataset_row = {
        **metadata,
        "variant_id": V4_VARIANT,
        "profile_id": profile.profile_id,
        "architecture_profile": profile.architecture_profile,
        "regularization_profile": profile.regularization_profile,
        "shared_latent_dim": int(profile.shared_dim),
        "private_latent_dim_per_view": int(profile.private_dim),
        "hidden_dim": int(profile.hidden_dim),
        "factor_level_metrics_primary_eligible": primary_eligible,
        "selected_epoch_for_05d5": selected_epoch,
        "selected_epoch_mean_05d4": float(selected_epoch_record["selected_epoch_mean_05d4"]),
        "selected_epoch_sd_05d4": float(selected_epoch_record["selected_epoch_sd_05d4"]),
        "selected_epoch_min_05d4": int(selected_epoch_record["selected_epoch_min_05d4"]),
        "selected_epoch_max_05d4": int(selected_epoch_record["selected_epoch_max_05d4"]),
        "reference_fit_seed": int(stability_group["reference_fit_seed"]),
        "all_required_fits_passed": False,
        "reference_effective_rank": None,
        "reference_median_shared_variance": None,
        "latent_collapse": None,
        "reference_shared_subspace_recovery": None,
        "reference_technical_subspace_recovery": None,
        "stability_subspace_median_canonical_correlation": None,
        "stability_subspace_q10_canonical_correlation": None,
        "n_stable_factors": None,
        "n_stable_biological_factors": None,
        "n_stable_technical_factors": None,
        "n_stable_unmatched_nontechnical_factors": None,
        "stability_gated_false_biological_discovery_fraction": None,
        "n_stability_resamples": None,
        "selected_epoch_used_for_stability": None,
        "shard_status": "FAIL",
        "shard_error_type": type(error).__name__,
        "shard_error_message": clean_text(str(error))[:2000],
        "max_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(0))
        if torch.cuda.is_available()
        else None,
    }
    audit = {
        **metadata,
        "execution_stage": "05d5_confirmatory_V4_stability",
        "variant_id": V4_VARIANT,
        "profile_id": profile.profile_id,
        "selected_epoch_for_05d5": selected_epoch,
        "factor_level_metrics_primary_eligible": primary_eligible,
        "fit_stage": "05d5_dataset_stability_failure",
        "fit_status": "FAIL",
        "fit_error_type": type(error).__name__,
        "fit_error_message": clean_text(str(error))[:2000],
        "fit_traceback_tail": "\n".join(traceback.format_exc().splitlines()[-20:]),
    }
    return {
        "script_version": SCRIPT_VERSION,
        "scenario_id": metadata["scenario_id"],
        "replicate_index": metadata["replicate_index"],
        "simulation_seed": metadata["simulation_seed"],
        "variant_id": V4_VARIANT,
        "profile_id": profile.profile_id,
        "input_path": metadata["input_path"],
        "input_sha256": metadata["input_sha256"],
        "selected_epoch": selected_epoch,
        "dataset_row": dataset_row,
        "factor_rows": [],
        "fit_audits": [audit],
        "preprocessing_rows": [],
    }


def materialize_shard(
    *,
    neural: Any,
    linear: Any,
    dataset_path: Path,
    metadata: dict[str, Any],
    profile: Any,
    selected_epoch_record: dict[str, Any],
    stability_group: dict[str, Any],
    stability_thresholds: dict[str, Any],
    protocol_hash: str,
) -> tuple[dict[str, Any], str]:
    selected_epoch = int(selected_epoch_record["selected_epoch_for_05d5"])
    candidate = shard_path(metadata, selected_epoch, protocol_hash)
    if candidate.exists():
        shard = load_json(candidate)
        validate_existing_shard(shard, metadata, selected_epoch, protocol_hash)
        return shard, "verified_existing"

    try:
        arrays = linear.load_simulation_arrays(dataset_path)
        shard = evaluate_dataset_stability(
            neural=neural,
            linear=linear,
            arrays=arrays,
            metadata=metadata,
            profile=profile,
            selected_epoch_record=selected_epoch_record,
            stability_group=stability_group,
            stability_thresholds=stability_thresholds,
        )
        del arrays
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        shard = failed_shard_record(
            metadata=metadata,
            profile=profile,
            selected_epoch_record=selected_epoch_record,
            stability_group=stability_group,
            error=exc,
        )

    shard["protocol_sha256"] = protocol_hash
    shard["created_at_utc"] = utc_now_iso()
    write_json_atomic(shard, candidate)
    serialized = load_json(candidate)
    validate_existing_shard(serialized, metadata, selected_epoch, protocol_hash)
    return serialized, "generated"


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------

def aggregate_shards(
    shards: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_rows: list[dict[str, Any]] = []
    factor_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    preprocessing_rows: list[dict[str, Any]] = []
    for shard in shards:
        dataset_rows.append(shard["dataset_row"])
        factor_rows.extend(shard.get("factor_rows", []))
        fit_rows.extend(shard.get("fit_audits", []))
        preprocessing_rows.extend(shard.get("preprocessing_rows", []))
    return (
        pd.DataFrame(dataset_rows),
        pd.DataFrame(factor_rows),
        pd.DataFrame(fit_rows),
        pd.DataFrame(preprocessing_rows),
    )


SCENARIO_SUBSPACE_METRICS = [
    "reference_effective_rank",
    "reference_median_shared_variance",
    "reference_shared_subspace_recovery",
    "reference_technical_subspace_recovery",
    "stability_subspace_median_canonical_correlation",
    "stability_subspace_q10_canonical_correlation",
]

PRIMARY_FACTOR_METRICS = [
    "n_stable_factors",
    "n_stable_biological_factors",
    "n_stable_technical_factors",
    "n_stable_unmatched_nontechnical_factors",
    "stability_gated_false_biological_discovery_fraction",
]


def scenario_summary(dataset_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["scenario_id", "scenario_origin", "scenario_family", "calibration_role"]
    for keys, part in dataset_table.groupby(group_columns, sort=True, dropna=False):
        row = dict(zip(group_columns, keys))
        if part.shape[0] != 10:
            raise RuntimeError(f"{row['scenario_id']}: expected exactly 10 confirmatory datasets.")
        passed = part[part["shard_status"].eq("PASS")].copy()
        primary = passed[passed["factor_level_metrics_primary_eligible"].map(parse_bool)].copy()

        row["n_confirmatory_datasets"] = int(part.shape[0])
        row["n_passed_datasets"] = int(passed.shape[0])
        row["n_failed_datasets"] = int(part["shard_status"].eq("FAIL").sum())
        row["n_latent_collapse"] = int(passed["latent_collapse"].map(parse_bool).sum()) if not passed.empty else 0
        row["n_factor_level_primary_eligible_datasets"] = int(primary.shape[0])

        epochs = finite_values(passed["selected_epoch_for_05d5"]) if not passed.empty else np.asarray([], dtype=float)
        row["selected_epoch_mean"] = float(np.mean(epochs)) if epochs.size else None
        row["selected_epoch_sd"] = float(np.std(epochs, ddof=1)) if epochs.size > 1 else None

        for metric in SCENARIO_SUBSPACE_METRICS:
            values = finite_values(passed[metric]) if metric in passed.columns else np.asarray([], dtype=float)
            row[f"{metric}_n"] = int(values.size)
            row[f"{metric}_mean"] = float(np.mean(values)) if values.size else None
            row[f"{metric}_median"] = float(np.median(values)) if values.size else None

        for metric in PRIMARY_FACTOR_METRICS:
            values = finite_values(primary[metric]) if metric in primary.columns else np.asarray([], dtype=float)
            row[f"primary_{metric}_n"] = int(values.size)
            row[f"primary_{metric}_mean"] = float(np.mean(values)) if values.size else None
            row[f"primary_{metric}_median"] = float(np.median(values)) if values.size else None

        rows.append(row)

    output = pd.DataFrame(rows)
    if output.shape[0] != EXPECTED_SCENARIOS:
        raise RuntimeError(f"Expected {EXPECTED_SCENARIOS} scenario summaries; found {output.shape[0]}.")
    return output


def overall_summary(dataset_table: pd.DataFrame, scenario_table: pd.DataFrame) -> pd.DataFrame:
    row: dict[str, Any] = {
        "variant_id": V4_VARIANT,
        "profile_id": clean_text(dataset_table.iloc[0]["profile_id"]),
        "n_scenarios": int(scenario_table.shape[0]),
        "n_confirmatory_datasets": int(dataset_table.shape[0]),
        "n_passed_datasets": int(dataset_table["shard_status"].eq("PASS").sum()),
        "n_failed_datasets": int(dataset_table["shard_status"].eq("FAIL").sum()),
        "n_latent_collapse": int(
            dataset_table.loc[dataset_table["shard_status"].eq("PASS"), "latent_collapse"].map(parse_bool).sum()
        ),
        "representation_method_selected": False,
    }

    passed = dataset_table[dataset_table["shard_status"].eq("PASS")].copy()
    primary = passed[passed["factor_level_metrics_primary_eligible"].map(parse_bool)].copy()
    row["n_factor_level_primary_eligible_datasets"] = int(primary.shape[0])

    epochs = finite_values(passed["selected_epoch_for_05d5"]) if not passed.empty else np.asarray([], dtype=float)
    row["selected_epoch_mean_across_datasets"] = float(np.mean(epochs)) if epochs.size else None
    row["selected_epoch_sd_across_datasets"] = float(np.std(epochs, ddof=1)) if epochs.size > 1 else None

    for metric in SCENARIO_SUBSPACE_METRICS:
        column = f"{metric}_mean"
        values = finite_values(scenario_table[column]) if column in scenario_table.columns else np.asarray([], dtype=float)
        row[f"overall_mean_{metric}_across_scenarios"] = (
            float(np.mean(values)) if values.size else None
        )
        row[f"scenario_mean_{metric}_sd"] = (
            float(np.std(values, ddof=1)) if values.size > 1 else None
        )
        row[f"scenario_mean_{metric}_se"] = (
            float(np.std(values, ddof=1) / math.sqrt(values.size))
            if values.size > 1
            else None
        )
        row[f"n_finite_scenario_means_{metric}"] = int(values.size)

    # Primary factor-level summaries are intentionally based only on datasets
    # where factor identities are interpretable (or biological truth is null).
    for metric in PRIMARY_FACTOR_METRICS:
        values = finite_values(primary[metric]) if metric in primary.columns else np.asarray([], dtype=float)
        row[f"primary_dataset_mean_{metric}"] = float(np.mean(values)) if values.size else None
        row[f"primary_dataset_median_{metric}"] = float(np.median(values)) if values.size else None
        row[f"primary_dataset_n_{metric}"] = int(values.size)

    # A scenario-balanced companion for primary FDR is useful descriptively,
    # but no 05d5 hard-gate or method selection decision is made here.
    fdr_column = "primary_stability_gated_false_biological_discovery_fraction_mean"
    fdr_scenario = finite_values(scenario_table[fdr_column]) if fdr_column in scenario_table.columns else np.asarray([], dtype=float)
    row["scenario_balanced_primary_false_biological_discovery_fraction"] = (
        float(np.mean(fdr_scenario)) if fdr_scenario.size else None
    )
    row["n_scenarios_with_primary_false_discovery_estimate"] = int(fdr_scenario.size)

    return pd.DataFrame([row])


# -----------------------------------------------------------------------------
# Frozen-result verification and README
# -----------------------------------------------------------------------------

def verify_existing_lock() -> None:
    lock = load_json(OUTPUT_LOCK)
    allowed_status = {
        "FROZEN_NEURAL_CONFIRMATORY_STABILITY_RESULTS_BEFORE_05E",
        "FROZEN_NEURAL_CONFIRMATORY_STABILITY_RESULTS_WITH_FAILURES",
    }
    failures: list[str] = []
    if clean_text(lock.get("script_version")) != SCRIPT_VERSION:
        failures.append("script version mismatch")
    if clean_text(lock.get("status")) not in allowed_status:
        failures.append("unexpected lock status")

    for relative_name, record in lock.get("generated_files", {}).items():
        candidate = resolve_project_path(relative_name)
        if not candidate.exists():
            failures.append(f"missing output: {relative_name}")
            continue
        if int(candidate.stat().st_size) != int(record.get("size_bytes", -1)):
            failures.append(f"size mismatch: {relative_name}")
            continue
        if sha256_file(candidate) != clean_text(record.get("sha256")):
            failures.append(f"SHA256 mismatch: {relative_name}")

    shard_record = lock.get("shard_manifest", {})
    shard_manifest_path = resolve_project_path(shard_record.get("path", ""))
    if not shard_manifest_path.exists():
        failures.append("shard manifest missing")
    else:
        table = pd.read_csv(shard_manifest_path, dtype=str, low_memory=False).fillna("")
        if table.shape[0] != EXPECTED_CONFIRMATORY_DATASETS:
            failures.append("shard manifest row-count mismatch")
        else:
            for _, row in table.iterrows():
                candidate = resolve_project_path(row["shard_path"])
                if not candidate.exists():
                    failures.append(f"missing shard: {row['shard_path']}")
                    break
                if int(candidate.stat().st_size) != int(float(row["size_bytes"])):
                    failures.append(f"shard size mismatch: {row['shard_path']}")
                    break
                if sha256_file(candidate) != clean_text(row["sha256"]):
                    failures.append(f"shard hash mismatch: {row['shard_path']}")
                    break

    if failures:
        raise RuntimeError("Existing 05d5 frozen-result verification failed:\n" + "\n".join(f"- {x}" for x in failures))

    print("=" * 88)
    print("Existing Paper 5 V4 confirmatory stability result lock: VERIFIED")
    print("=" * 88)
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("No files were rewritten.")
    print("=" * 88)


def write_readme(profile: Any, n_pass: int, n_fail: int) -> None:
    lines = [
        "Paper 5 V4 confirmatory stability benchmark",
        "=" * 72,
        "",
        f"Script version: {SCRIPT_VERSION}",
        "",
        "Scope",
        "-----",
        "Confirmatory simulation replicate indices 11-20 only.",
        f"Confirmatory datasets: {EXPECTED_CONFIRMATORY_DATASETS}",
        f"Variant: {V4_VARIANT}",
        f"Frozen profile: {profile.profile_id}",
        f"Reference fits: {EXPECTED_REFERENCE_FITS}",
        f"Resample fits: {EXPECTED_STABILITY_REGISTRY_ROWS}",
        f"Total neural optimization runs if all datasets pass: {EXPECTED_STABILITY_OPTIMIZATION_RUNS}",
        "",
        "Epoch bridge",
        "------------",
        "05d1 did not explicitly specify how the ten 05d4 V4 fold/seed-selected epoch counts",
        "should transfer to one full-data 05d5 stability epoch count. Before any 05d5 stability",
        "result is generated, this script freezes a deterministic dataset-specific registry:",
        "mean of the ten 05d4 selected_epoch diagnostics, rounded nearest integer half-up.",
        "No 05d4 utility/truth metric, 05d5 stability result, TARGET value, or outcome is used by this rule.",
        "",
        "Stability",
        "---------",
        "Each dataset receives one frozen-seed full-data V4 reference fit and 20 frozen-seed 80% subsample refits.",
        "Each resample refits preprocessing and the neural model.",
        "Shared scores are aligned by orthogonal Procrustes and Hungarian factor matching.",
        "The already-frozen 05a stability thresholds are applied without retuning.",
        "",
        "Interpretation",
        "--------------",
        "Factor-level primary summaries are restricted to factor-level-identifiable datasets and true biological nulls.",
        "Rotational non-null factor-level results remain descriptive; subspace stability remains interpretable.",
        "This stage does not select the final representation method.",
        "",
        "GPU execution",
        "-------------",
        "All neural forward/backward/AdamW work runs on frozen cuda:0 in float32.",
        "Dual-PCA Gram/eigendecomposition/projection runs in float64 on cuda:0.",
        "AMP: OFF; TF32: OFF; torch.compile: OFF; quiet CPU fallback: FORBIDDEN.",
        "",
        "Results",
        "-------",
        f"Passed dataset shards: {n_pass}",
        f"Failed dataset shards: {n_fail}",
        "",
        "Next stage",
        "----------",
        "05e_compare_ablate_and_freeze_method.py" if n_fail == 0 else "STOP: review 05d5 failures before 05e.",
        "",
    ]
    OUTPUT_README.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    print("=" * 88)
    print("Paper 5 - GPU V4 confirmatory stability benchmark")
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

    neural = load_module(NEURAL_IMPL_SCRIPT, "paper5_05d2v2_05d5_helpers")
    linear = neural.load_linear_helpers()
    upstream = verify_upstream_state(neural)
    profile = upstream["profile"]

    # Freeze the previously under-specified 05d4 -> 05d5 selected-epoch bridge
    # before any 05d5 reference/resample result is generated.
    initialize_or_verify_epoch_registry(upstream["epoch_registry"])
    epoch_lookup = build_epoch_lookup(upstream["epoch_registry"])

    print("05d0 frozen CUDA runtime: PASS")
    print("05d1 frozen confirmatory stability registry: PASS")
    print("05d2 v2 frozen V4 implementation/profile-screen results: PASS")
    print("05d3 frozen selected profile: PASS")
    print("05d4 frozen confirmatory ablation results: PASS")
    print(f"Frozen profile: {profile.profile_id}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Heavy neural computation device: cuda:0")
    print("Quiet CPU fallback: FORBIDDEN")
    print("Automatic mixed precision: OFF")
    print("TF32: OFF")
    print("torch.compile: OFF")
    print("TARGET/outcomes/final-gate data: FORBIDDEN")
    print()

    print("05d4 -> 05d5 selected-epoch bridge: FROZEN BEFORE 05d5 STABILITY FITS")
    print(f"Epoch registry: {relative_path(OUTPUT_EPOCH_REGISTRY)}")
    print("Rule: mean of 10 V4 fold-seed selected epochs, nearest integer half-up")
    print("05d4 performance metrics used by epoch rule: NO")
    print("05d5 stability results used by epoch rule: NO")
    print()

    execution_protocol, protocol_hash = initialize_or_verify_protocol(upstream)
    del execution_protocol
    print("Prospective 05d5 confirmatory stability protocol: FROZEN")
    print(f"Confirmatory datasets: {EXPECTED_CONFIRMATORY_DATASETS}")
    print(f"Reference fits: {EXPECTED_REFERENCE_FITS}")
    print(f"Stability resamples per dataset: {STABILITY_RESAMPLES}")
    print(f"Resample fits: {EXPECTED_STABILITY_REGISTRY_ROWS:,}")
    print(f"Planned optimization runs: {EXPECTED_STABILITY_OPTIMIZATION_RUNS:,}")
    print()

    print("Running deterministic V4 CUDA + stability-alignment fixture...")
    fixture = run_reference_fixture(neural, linear, profile)
    write_json_atomic(fixture, OUTPUT_FIXTURE)
    print("05d5 deterministic V4 CUDA fixture: PASS")
    print("05d5 Procrustes/Hungarian alignment fixture: PASS")
    print()

    write_json_atomic(
        {
            "script_version": SCRIPT_VERSION,
            "created_at_utc": utc_now_iso(),
            "protocol_sha256": protocol_hash,
            "epoch_registry_sha256": sha256_file(OUTPUT_EPOCH_REGISTRY),
            "planned_dataset_shards": EXPECTED_CONFIRMATORY_DATASETS,
            "planned_optimization_runs": EXPECTED_STABILITY_OPTIMIZATION_RUNS,
            "status": "IN_PROGRESS",
        },
        OUTPUT_IN_PROGRESS,
    )

    confirmatory = upstream["confirmatory_roles"].copy()
    confirmatory = confirmatory.sort_values(
        ["scenario_id", "replicate_index"], kind="mergesort"
    ).reset_index(drop=True)

    shards: list[dict[str, Any]] = []
    shard_manifest_rows: list[dict[str, Any]] = []
    generated_count = 0
    generated_seconds = 0.0

    for current, (_, role_row) in enumerate(confirmatory.iterrows(), start=1):
        key = (
            clean_text(role_row["scenario_id"]),
            int(role_row["replicate_index"]),
            int(role_row["simulation_seed"]),
        )
        if key not in upstream["manifest_lookup"]:
            raise RuntimeError(f"Confirmatory dataset missing from manifest: {key}")
        if key not in upstream["stability_lookup"]:
            raise RuntimeError(f"Confirmatory stability group missing: {key}")
        if key not in epoch_lookup:
            raise RuntimeError(f"05d5 epoch record missing: {key}")

        manifest_row = upstream["manifest_lookup"][key]
        dataset_path = resolve_project_path(manifest_row["path"])
        require_file(dataset_path)
        expected_size = int(float(manifest_row["size_bytes"]))
        if int(dataset_path.stat().st_size) != expected_size:
            raise RuntimeError(f"Simulation size differs from frozen manifest: {relative_path(dataset_path)}")
        observed_hash = sha256_file(dataset_path)
        if observed_hash != clean_text(manifest_row["sha256"]):
            raise RuntimeError(f"Simulation SHA256 differs from frozen manifest: {relative_path(dataset_path)}")

        metadata = dataset_metadata(manifest_row, dataset_path, observed_hash)
        metadata["profile_id"] = profile.profile_id
        selected_epoch_record = epoch_lookup[key]
        selected_epoch = int(selected_epoch_record["selected_epoch_for_05d5"])
        candidate = shard_path(metadata, selected_epoch, protocol_hash)
        started = time.perf_counter()

        if candidate.exists():
            shard = load_json(candidate)
            validate_existing_shard(shard, metadata, selected_epoch, protocol_hash)
            generation_status = "verified_existing"
        else:
            shard, generation_status = materialize_shard(
                neural=neural,
                linear=linear,
                dataset_path=dataset_path,
                metadata=metadata,
                profile=profile,
                selected_epoch_record=selected_epoch_record,
                stability_group=upstream["stability_lookup"][key],
                stability_thresholds=upstream["stability_thresholds"],
                protocol_hash=protocol_hash,
            )

        elapsed = time.perf_counter() - started
        if generation_status == "generated":
            generated_count += 1
            generated_seconds += elapsed

        shards.append(shard)
        shard_status = clean_text(shard["dataset_row"].get("shard_status"))
        shard_manifest_rows.append(
            {
                "scenario_id": key[0],
                "scenario_family": clean_text(manifest_row.get("scenario_family")),
                "replicate_index": key[1],
                "simulation_seed": key[2],
                "variant_id": V4_VARIANT,
                "profile_id": profile.profile_id,
                "selected_epoch_for_05d5": selected_epoch,
                "input_path": relative_path(dataset_path),
                "input_sha256": observed_hash,
                "shard_path": relative_path(candidate),
                "size_bytes": int(candidate.stat().st_size),
                "sha256": sha256_file(candidate),
                "generation_status": generation_status,
                "shard_status": shard_status,
                "wall_seconds": elapsed,
            }
        )

        if current == 1 or current % 5 == 0 or shard_status == "FAIL" or current == EXPECTED_CONFIRMATORY_DATASETS:
            if generated_count > 0:
                mean_seconds = generated_seconds / generated_count
                remaining = EXPECTED_CONFIRMATORY_DATASETS - current
                eta_text = f" | ETA~{remaining * mean_seconds / 3600.0:.1f}h"
            else:
                eta_text = ""
            print(
                f"{current:04d}/{EXPECTED_CONFIRMATORY_DATASETS:04d} "
                f"{key[0]} R{key[1]:03d} E{selected_epoch:03d}: "
                f"{generation_status}/{shard_status}{eta_text}"
            )
            if shard_status == "FAIL":
                print(
                    "    ERROR: "
                    + clean_text(shard["dataset_row"].get("shard_error_type"))
                    + ": "
                    + clean_text(shard["dataset_row"].get("shard_error_message"))[:500]
                )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    shard_manifest = pd.DataFrame(shard_manifest_rows)
    if shard_manifest.shape[0] != EXPECTED_CONFIRMATORY_DATASETS:
        raise RuntimeError("05d5 shard manifest does not contain 360 rows.")
    shard_manifest.to_csv(
        OUTPUT_SHARD_MANIFEST, index=False, float_format="%.12g", lineterminator="\n"
    )

    dataset_table, factor_table, fit_table, preprocessing_table = aggregate_shards(shards)
    if dataset_table.shape[0] != EXPECTED_CONFIRMATORY_DATASETS:
        raise RuntimeError("05d5 dataset summary does not contain 360 rows.")

    n_pass = int(dataset_table["shard_status"].eq("PASS").sum())
    n_fail = int(dataset_table["shard_status"].eq("FAIL").sum())
    if n_fail == 0:
        if fit_table.shape[0] != EXPECTED_FIT_AUDIT_ROWS_IF_ALL_PASS:
            raise RuntimeError(
                f"All-PASS 05d5 expected {EXPECTED_FIT_AUDIT_ROWS_IF_ALL_PASS:,} fit-audit rows; "
                f"found {fit_table.shape[0]:,}."
            )
        if preprocessing_table.shape[0] != EXPECTED_PREPROCESSING_ROWS_IF_ALL_PASS:
            raise RuntimeError(
                f"All-PASS 05d5 expected {EXPECTED_PREPROCESSING_ROWS_IF_ALL_PASS:,} preprocessing rows; "
                f"found {preprocessing_table.shape[0]:,}."
            )
        expected_factor_rows = EXPECTED_CONFIRMATORY_DATASETS * int(profile.shared_dim)
        if factor_table.shape[0] != expected_factor_rows:
            raise RuntimeError(
                f"All-PASS 05d5 expected {expected_factor_rows:,} factor-audit rows; found {factor_table.shape[0]:,}."
            )

    write_csv_gzip(dataset_table, OUTPUT_DATASET_SUMMARY)
    write_csv_gzip(factor_table, OUTPUT_FACTOR_AUDIT)
    write_csv_gzip(fit_table, OUTPUT_FIT_AUDIT)
    write_csv_gzip(preprocessing_table, OUTPUT_PREPROCESSING_AUDIT)

    scenario_table = scenario_summary(dataset_table)
    scenario_table.to_csv(
        OUTPUT_SCENARIO_SUMMARY, index=False, float_format="%.12g", lineterminator="\n"
    )
    overall_table = overall_summary(dataset_table, scenario_table)
    overall_table.to_csv(
        OUTPUT_OVERALL_SUMMARY, index=False, float_format="%.12g", lineterminator="\n"
    )
    write_readme(profile, n_pass, n_fail)

    output_files = [
        OUTPUT_EPOCH_REGISTRY,
        OUTPUT_PROTOCOL,
        OUTPUT_FIXTURE,
        OUTPUT_SHARD_MANIFEST,
        OUTPUT_DATASET_SUMMARY,
        OUTPUT_FACTOR_AUDIT,
        OUTPUT_FIT_AUDIT,
        OUTPUT_PREPROCESSING_AUDIT,
        OUTPUT_SCENARIO_SUMMARY,
        OUTPUT_OVERALL_SUMMARY,
        OUTPUT_README,
    ]

    result_manifest = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "scientific_state": {
            "outcomes_loaded": False,
            "clinical_labels_loaded": False,
            "biological_annotations_used_for_model_or_profile_selection": False,
            "TARGET_model_fitting_performed": False,
            "neural_profile_frozen": True,
            "confirmatory_replicates_11_to_20_accessed": True,
            "confirmatory_neural_ablation_performed": True,
            "confirmatory_V4_stability_performed": True,
            "final_gate_calibration_data_used": False,
            "representation_method_selected": False,
        },
        "scope": {
            "variant_id": V4_VARIANT,
            "profile_id": profile.profile_id,
            "confirmatory_datasets": EXPECTED_CONFIRMATORY_DATASETS,
            "passed_datasets": n_pass,
            "failed_datasets": n_fail,
            "reference_fits_planned": EXPECTED_REFERENCE_FITS,
            "resample_fits_planned": EXPECTED_STABILITY_REGISTRY_ROWS,
            "fit_audit_rows": int(fit_table.shape[0]),
            "preprocessing_audit_rows": int(preprocessing_table.shape[0]),
            "factor_audit_rows": int(factor_table.shape[0]),
            "scenario_summary_rows": int(scenario_table.shape[0]),
            "overall_summary_rows": int(overall_table.shape[0]),
        },
        "epoch_registry": file_record(OUTPUT_EPOCH_REGISTRY),
        "execution_protocol": {"path": relative_path(OUTPUT_PROTOCOL), "sha256": protocol_hash},
        "reference_fixture": file_record(OUTPUT_FIXTURE),
        "shard_manifest": {
            "path": relative_path(OUTPUT_SHARD_MANIFEST),
            "sha256": sha256_file(OUTPUT_SHARD_MANIFEST),
            "n_rows": int(shard_manifest.shape[0]),
        },
        "outputs": {
            relative_path(path): {
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
            for path in output_files
        },
        "upstream_hashes": upstream["hashes"],
        "interpretation_guardrails": [
            "05d5 uses confirmatory simulation replicate indices 11-20 only.",
            "Only frozen V4 SABLE-MO is stability-resampled.",
            "The 05d3-selected profile and 05a stability thresholds are not retuned.",
            "The 05d4-to-05d5 selected-epoch bridge is frozen before any 05d5 stability result and uses selected_epoch diagnostics only.",
            "Factor-level primary summaries exclude rotational non-null datasets, while subspace metrics remain reportable.",
            "No TARGET, outcome, clinical, biological-program, or final-gate calibration data are used.",
            "No final representation method is selected in 05d5.",
        ],
        "next_required_stage": "05e_compare_ablate_and_freeze_method.py" if n_fail == 0 else None,
    }
    write_json_atomic(result_manifest, OUTPUT_RESULT_MANIFEST)

    generated_files = {
        relative_path(path): {
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for path in [*output_files, OUTPUT_RESULT_MANIFEST]
    }
    lock_status = (
        "FROZEN_NEURAL_CONFIRMATORY_STABILITY_RESULTS_BEFORE_05E"
        if n_fail == 0
        else "FROZEN_NEURAL_CONFIRMATORY_STABILITY_RESULTS_WITH_FAILURES"
    )
    lock = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": lock_status,
        "scientific_state": result_manifest["scientific_state"],
        "profile_id": profile.profile_id,
        "variant_id": V4_VARIANT,
        "result_manifest": {
            "path": relative_path(OUTPUT_RESULT_MANIFEST),
            "sha256": sha256_file(OUTPUT_RESULT_MANIFEST),
        },
        "epoch_registry": result_manifest["epoch_registry"],
        "shard_manifest": result_manifest["shard_manifest"],
        "generated_files": generated_files,
        "rewrite_policy": (
            "This zero-argument stage never rewrites frozen 05d5 results. Any scientifically justified change "
            "requires a new versioned confirmatory stability stage rather than silent replacement."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    if OUTPUT_IN_PROGRESS.exists():
        OUTPUT_IN_PROGRESS.unlink()

    print()
    print("-" * 88)
    print("V4 confirmatory stability summary")
    print("-" * 88)
    print(f"Dataset shards: {dataset_table.shape[0]:,}")
    print(f"Passed datasets: {n_pass:,}")
    print(f"Failed datasets: {n_fail:,}")
    print(f"Fit-audit rows: {fit_table.shape[0]:,}")
    print(f"Preprocessing-audit rows: {preprocessing_table.shape[0]:,}")
    print(f"Factor-audit rows: {factor_table.shape[0]:,}")
    print()

    display_columns = [
        "n_passed_datasets",
        "n_failed_datasets",
        "n_latent_collapse",
        "overall_mean_stability_subspace_median_canonical_correlation_across_scenarios",
        "overall_mean_stability_subspace_q10_canonical_correlation_across_scenarios",
        "primary_dataset_mean_stability_gated_false_biological_discovery_fraction",
        "n_factor_level_primary_eligible_datasets",
    ]
    available_display = [column for column in display_columns if column in overall_table.columns]
    print(overall_table[available_display].to_string(index=False))
    print()
    print("Representation method selected: NO")
    print("TARGET accessed: NO")
    print("Final gate-calibration data accessed: NO")
    print()
    print("=" * 88)
    if n_fail == 0:
        print("Paper 5 V4 confirmatory stability benchmark: PASS")
        print("=" * 88)
        print(f"Results manifest: {relative_path(OUTPUT_RESULT_MANIFEST)}")
        print(f"Lock: {relative_path(OUTPUT_LOCK)}")
        print("Next stage: 05e_compare_ablate_and_freeze_method.py")
    else:
        print("Paper 5 V4 confirmatory stability benchmark: COMPLETED WITH FAILURES")
        print("=" * 88)
        print(f"Results manifest: {relative_path(OUTPUT_RESULT_MANIFEST)}")
        print(f"Lock: {relative_path(OUTPUT_LOCK)}")
        print("STOP: review 05d5 failures before 05e.")
    print("=" * 88)


if __name__ == "__main__":
    main()
