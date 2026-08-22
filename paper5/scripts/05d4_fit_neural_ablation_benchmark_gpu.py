from __future__ import annotations

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_VERSION = "05d4-fit-neural-ablation-benchmark-gpu-v1-no-cli"
CONTRACT_VERSION = "paper5-neural-confirmatory-ablation-benchmark-v1"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

VIEWS = ("RNA", "METHYLATION", "CNV")
VIEW_PREFIX = {"RNA": "rna", "METHYLATION": "methylation", "CNV": "cnv"}
VIEW_INDEX = {"RNA": 0, "METHYLATION": 1, "CNV": 2}

NEURAL_VARIANTS = (
    "V0_BALANCED_AE",
    "V1_SHARED_PRIVATE_AE",
    "V2_CONTRASTIVE_SPAE",
    "V3_DROPOUT_SPAE",
    "V4_SABLE_MO",
    "V5_SABLE_VAE",
)
PRIMARY_ABLATION_VARIANTS = NEURAL_VARIANTS[:5]
VAE_VARIANT = "V5_SABLE_VAE"
CONFIRMATORY_MODEL_BASE_SEEDS = (1103, 2207)
CONFIRMATORY_REPLICATES = tuple(range(11, 21))
PRIMARY_FEATURE_BUDGETS = {"RNA": 5_000, "METHYLATION": 10_000, "CNV": 5_000}

EXPECTED_SCENARIOS = 36
EXPECTED_CONFIRMATORY_DATASETS = 360
EXPECTED_FOLDS = 5
EXPECTED_SEEDS_PER_VARIANT_FOLD = 2
EXPECTED_FOLD_SEED_EVALUATIONS = (
    EXPECTED_CONFIRMATORY_DATASETS
    * len(NEURAL_VARIANTS)
    * EXPECTED_FOLDS
    * EXPECTED_SEEDS_PER_VARIANT_FOLD
)
EXPECTED_OPTIMIZATION_RUNS = EXPECTED_FOLD_SEED_EVALUATIONS * 2
EXPECTED_PREPROCESSING_ROWS_IF_ALL_PASS = EXPECTED_FOLD_SEED_EVALUATIONS * 6
EXPECTED_DATASET_VARIANT_ROWS = EXPECTED_CONFIRMATORY_DATASETS * len(NEURAL_VARIANTS)
EXPECTED_SCENARIO_VARIANT_ROWS = EXPECTED_SCENARIOS * len(NEURAL_VARIANTS)

EARLY_STOP_FRACTION = 0.20
GRADIENT_CLIP_NORM = 5.0
NULL_MSE_FLOOR = 1e-8
VAE_KL_FINAL_WEIGHT = 0.01
VAE_KL_WARMUP_FRACTION = 0.25

# -----------------------------------------------------------------------------
# Frozen upstream artifacts
# -----------------------------------------------------------------------------

CONTRACT_05A = PAPER5_ROOT / "method_contract" / "05a" / "ai_method_contract.json"
ABLATION_REGISTRY = PAPER5_ROOT / "method_contract" / "05a" / "sable_mo_ablation_registry.csv"
SIMULATION_MANIFEST = PAPER5_ROOT / "method_contract" / "05b_v2" / "architecture_screening_data_manifest.csv"
LINEAR_SCRIPT = PAPER5_ROOT / "scripts" / "05c1_fit_linear_simulation_baselines.py"
LINEAR_SPLITS = PAPER5_ROOT / "results" / "simulation_benchmark" / "05c1_linear_baselines" / "linear_simulation_split_registry.csv.gz"
ENV_CONTRACT = PAPER5_ROOT / "method_contract" / "05d0" / "neural_environment_contract.json"
ENV_LOCK = PAPER5_ROOT / "method_contract" / "05d0" / "neural_environment_lock.json"
PROTOCOL_05D1 = PAPER5_ROOT / "method_contract" / "05d1" / "neural_simulation_evaluation_protocol.json"
LOSS_CONTRACT_05D1 = PAPER5_ROOT / "method_contract" / "05d1" / "neural_loss_and_architecture_contract.json"
LOCK_05D1 = PAPER5_ROOT / "method_contract" / "05d1" / "neural_simulation_protocol_lock.json"
PROFILE_GRID_05D1 = PAPER5_ROOT / "method_contract" / "05d1" / "neural_architecture_regularization_profile_grid.csv"
DATASET_ROLES_05D1 = PAPER5_ROOT / "method_contract" / "05d1" / "neural_simulation_dataset_role_registry.csv"
CONFIRMATORY_SEEDS_05D1 = PAPER5_ROOT / "method_contract" / "05d1" / "neural_confirmatory_ablation_fit_seed_registry.csv"
NEURAL_IMPL_SCRIPT = PAPER5_ROOT / "scripts" / "05d2v2_screen_neural_profiles_gpu_syncfree_pathfix.py"
LOCK_05D2 = PAPER5_ROOT / "method_contract" / "05d2_v2" / "neural_profile_screen_results_lock.json"
RESULT_MANIFEST_05D2 = PAPER5_ROOT / "method_contract" / "05d2_v2" / "neural_profile_screen_results_manifest.json"
CONTRACT_05D3 = PAPER5_ROOT / "method_contract" / "05d3" / "selected_neural_profile.json"
PROFILE_TABLE_05D3 = PAPER5_ROOT / "method_contract" / "05d3" / "selected_neural_profile.csv"
LOCK_05D3 = PAPER5_ROOT / "method_contract" / "05d3" / "selected_neural_profile_lock.json"
EXPECTED_05D3_STATUS = "FROZEN_OUTCOME_BLIND_NEURAL_PROFILE_BEFORE_CONFIRMATORY_ABLATION"

FINAL_GATE_DATA_ROOT = PAPER5_ROOT / "simulations" / "sable_mo_v2" / "final_gate_calibration"
FUTURE_RESULT_PATHS = (
    PAPER5_ROOT / "method_contract" / "05d5",
    PAPER5_ROOT / "method_contract" / "05e",
    PAPER5_ROOT / "results" / "representation_benchmark",
)

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

OUTPUT_CONTRACT_DIR = PAPER5_ROOT / "method_contract" / "05d4"
OUTPUT_RESULTS_DIR = PAPER5_ROOT / "results" / "simulation_benchmark" / "05d_neural" / "confirmatory_ablation"
OUTPUT_SHARD_DIR = OUTPUT_RESULTS_DIR / "shards" / "_compact"
OUTPUT_PROTOCOL = OUTPUT_CONTRACT_DIR / "neural_confirmatory_ablation_execution_protocol.json"
OUTPUT_FIXTURE = OUTPUT_CONTRACT_DIR / "neural_confirmatory_ablation_reference_fixture.json"
OUTPUT_SHARD_MANIFEST = OUTPUT_CONTRACT_DIR / "neural_confirmatory_ablation_shard_manifest.csv"
OUTPUT_RESULT_MANIFEST = OUTPUT_CONTRACT_DIR / "neural_confirmatory_ablation_results_manifest.json"
OUTPUT_LOCK = OUTPUT_CONTRACT_DIR / "neural_confirmatory_ablation_results_lock.json"
OUTPUT_README = OUTPUT_CONTRACT_DIR / "neural_confirmatory_ablation_README.txt"
OUTPUT_IN_PROGRESS = OUTPUT_CONTRACT_DIR / ".neural_confirmatory_ablation_in_progress.json"
OUTPUT_FOLD_SEED_METRICS = OUTPUT_RESULTS_DIR / "neural_confirmatory_fold_seed_metrics.csv.gz"
OUTPUT_FIT_AUDIT = OUTPUT_RESULTS_DIR / "neural_confirmatory_fit_audit.csv.gz"
OUTPUT_PREPROCESSING_AUDIT = OUTPUT_RESULTS_DIR / "neural_confirmatory_preprocessing_audit.csv.gz"
OUTPUT_DATASET_VARIANT = OUTPUT_RESULTS_DIR / "neural_confirmatory_dataset_variant_metrics.csv.gz"
OUTPUT_SCENARIO_VARIANT = OUTPUT_RESULTS_DIR / "neural_confirmatory_scenario_variant_summary.csv"
OUTPUT_VARIANT_SUMMARY = OUTPUT_RESULTS_DIR / "neural_confirmatory_variant_summary.csv"
OUTPUT_ABLATION_DELTAS = OUTPUT_RESULTS_DIR / "neural_confirmatory_ablation_delta_summary.csv"


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    block_balanced_reconstruction: bool
    shared_private_decomposition: bool
    paired_contrastive_alignment: bool
    modality_dropout: bool
    missing_view_reconstruction: bool
    stability_gate: bool
    variational_latent: bool
    prespecified_component_hypothesis: str


@dataclass
class FoldRefitData:
    training_indices: np.ndarray
    test_indices: np.ndarray
    preprocessors: dict[str, Any]
    training_views: dict[str, np.ndarray]
    training_availability: np.ndarray
    test_views: dict[str, np.ndarray]
    test_truth: dict[str, np.ndarray]
    test_availability: np.ndarray
    preprocessing_rows: list[dict[str, Any]]


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
    descriptor, temporary_name = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=str(path.parent))
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(to_json_safe(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
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
    return {"path": relative_path(path), "sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)}


def is_nonempty_path(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.stat().st_size > 0
    return any(item.is_file() for item in path.rglob("*"))


def nanmean_or_none(values: Iterable[Any]) -> float | None:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else None


def nanstd_or_none(values: Iterable[Any], ddof: int = 1) -> float | None:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.std(finite, ddof=ddof)) if finite.size > ddof else None


def verify_generated_files(lock: dict[str, Any], label: str) -> None:
    failures: list[str] = []
    for relative_name, record in lock.get("generated_files", {}).items():
        candidate = resolve_project_path(relative_name)
        if not candidate.exists():
            failures.append(f"missing: {relative_name}")
            continue
        if record.get("size_bytes") is not None and int(candidate.stat().st_size) != int(record["size_bytes"]):
            failures.append(f"size mismatch: {relative_name}")
            continue
        expected_hash = clean_text(record.get("sha256"))
        if expected_hash and sha256_file(candidate) != expected_hash:
            failures.append(f"SHA256 mismatch: {relative_name}")
    if failures:
        raise RuntimeError(f"{label} generated-file verification failed:\n" + "\n".join(f"- {x}" for x in failures))


def load_module(path: Path, module_name: str) -> Any:
    require_file(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def assert_cuda_runtime(env_contract: dict[str, Any]) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is unavailable. Quiet CPU fallback is forbidden.")
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    frozen = env_contract.get("software_freeze", {})
    if clean_text(torch.__version__) != clean_text(frozen.get("torch_version")):
        raise RuntimeError(f"Active PyTorch differs from 05d0: {torch.__version__} vs {frozen.get('torch_version')}")
    if clean_text(torch.version.cuda) != clean_text(frozen.get("torch_cuda_build_version")):
        raise RuntimeError(f"Active CUDA build differs from 05d0: {torch.version.cuda} vs {frozen.get('torch_cuda_build_version')}")
    frozen_gpu = clean_text(env_contract.get("device_freeze", {}).get("gpu_name"))
    if clean_text(torch.cuda.get_device_name(0)) != frozen_gpu:
        raise RuntimeError(f"Active GPU differs from 05d0: {torch.cuda.get_device_name(0)} vs {frozen_gpu}")


def build_variant_specs(loss_contract: dict[str, Any]) -> dict[str, VariantSpec]:
    records = loss_contract.get("ablation_components", {})
    output: dict[str, VariantSpec] = {}
    for variant_id in NEURAL_VARIANTS:
        if variant_id not in records:
            raise RuntimeError(f"05d1 loss contract missing {variant_id}.")
        row = records[variant_id]
        output[variant_id] = VariantSpec(
            variant_id=variant_id,
            block_balanced_reconstruction=parse_bool(row.get("block_balanced_reconstruction")),
            shared_private_decomposition=parse_bool(row.get("shared_private_decomposition")),
            paired_contrastive_alignment=parse_bool(row.get("paired_contrastive_alignment")),
            modality_dropout=parse_bool(row.get("modality_dropout")),
            missing_view_reconstruction=parse_bool(row.get("missing_view_reconstruction")),
            stability_gate=parse_bool(row.get("stability_gate")),
            variational_latent=parse_bool(row.get("variational_latent")),
            prespecified_component_hypothesis=clean_text(row.get("prespecified_component_hypothesis")),
        )
    expected = {
        "V0_BALANCED_AE": (False, False, False, False, False),
        "V1_SHARED_PRIVATE_AE": (True, False, False, False, False),
        "V2_CONTRASTIVE_SPAE": (True, True, False, False, False),
        "V3_DROPOUT_SPAE": (True, True, True, True, False),
        "V4_SABLE_MO": (True, True, True, True, False),
        "V5_SABLE_VAE": (True, True, True, True, True),
    }
    for variant_id, flags in expected.items():
        spec = output[variant_id]
        observed = (
            spec.shared_private_decomposition,
            spec.paired_contrastive_alignment,
            spec.modality_dropout,
            spec.missing_view_reconstruction,
            spec.variational_latent,
        )
        if observed != flags or not spec.block_balanced_reconstruction:
            raise RuntimeError(f"Frozen ablation flags changed for {variant_id}: {observed}.")
    if not output["V4_SABLE_MO"].stability_gate or not output["V5_SABLE_VAE"].stability_gate:
        raise RuntimeError("Frozen V4/V5 stability-gate flags changed.")
    return output


def parse_index_string(text: Any) -> np.ndarray:
    return np.asarray([int(x) for x in clean_text(text).split(";") if x], dtype=int)


def build_split_lookup(table: pd.DataFrame) -> dict[tuple[str, int, int], dict[int, np.ndarray]]:
    required = {"scenario_id", "replicate_index", "simulation_seed", "fold_index", "test_indices"}
    missing = required - set(table.columns)
    if missing:
        raise RuntimeError("05c1 split registry missing: " + ", ".join(sorted(missing)))
    output: dict[tuple[str, int, int], dict[int, np.ndarray]] = {}
    for _, row in table.iterrows():
        key = (clean_text(row["scenario_id"]), int(float(row["replicate_index"])), int(float(row["simulation_seed"])))
        fold = int(float(row["fold_index"]))
        if fold in output.setdefault(key, {}):
            raise RuntimeError(f"Duplicate 05c1 fold: {key}, {fold}")
        output[key][fold] = parse_index_string(row["test_indices"])
    return output


def build_manifest_lookup(table: pd.DataFrame) -> dict[tuple[str, int, int], pd.Series]:
    output: dict[tuple[str, int, int], pd.Series] = {}
    for _, row in table.iterrows():
        key = (clean_text(row["scenario_id"]), int(float(row["replicate_index"])), int(float(row["simulation_seed"])))
        if key in output:
            raise RuntimeError(f"Duplicate simulation manifest key: {key}")
        output[key] = row
    return output


def build_seed_lookup(table: pd.DataFrame) -> dict[tuple[str, int, int, str, int, int], dict[str, int]]:
    required = {"scenario_id", "replicate_index", "simulation_seed", "variant_id", "fold_index", "base_model_seed", "fit_seed", "refit_seed", "early_stop_partition_seed", "device"}
    missing = required - set(table.columns)
    if missing:
        raise RuntimeError("05d1 confirmatory seed registry missing: " + ", ".join(sorted(missing)))
    output: dict[tuple[str, int, int, str, int, int], dict[str, int]] = {}
    for _, row in table.iterrows():
        key = (
            clean_text(row["scenario_id"]), int(float(row["replicate_index"])), int(float(row["simulation_seed"])),
            clean_text(row["variant_id"]), int(float(row["fold_index"])), int(float(row["base_model_seed"])),
        )
        fit_seed = int(float(row["fit_seed"]))
        refit_seed = int(float(row["refit_seed"]))
        if key in output or fit_seed != refit_seed or clean_text(row["device"]) != "cuda:0":
            raise RuntimeError(f"Invalid frozen confirmatory seed row: {key}")
        output[key] = {"fit_seed": fit_seed, "refit_seed": refit_seed, "early_stop_partition_seed": int(float(row["early_stop_partition_seed"]))}
    if len(output) != EXPECTED_FOLD_SEED_EVALUATIONS:
        raise RuntimeError(f"Expected {EXPECTED_FOLD_SEED_EVALUATIONS:,} seed rows; found {len(output):,}.")
    return output


def deterministic_early_stop_partition(training_indices: np.ndarray, partition_seed: int) -> tuple[np.ndarray, np.ndarray]:
    training = np.sort(np.asarray(training_indices, dtype=int))
    if training.size < 4:
        raise RuntimeError("Training fold too small for early-stop split.")
    n_early = max(1, int(math.ceil(EARLY_STOP_FRACTION * training.size)))
    n_early = min(n_early, training.size - 2)
    permuted = np.random.default_rng(int(partition_seed)).permutation(training)
    early = np.sort(permuted[:n_early].astype(int))
    fit = np.sort(permuted[n_early:].astype(int))
    if (set(fit) & set(early)) or (set(fit) | set(early) != set(training)):
        raise RuntimeError("Invalid deterministic early-stop partition.")
    return fit, early


def verify_upstream_state(neural: Any) -> dict[str, Any]:
    required_paths = [
        CONTRACT_05A,
        ABLATION_REGISTRY,
        SIMULATION_MANIFEST,
        LINEAR_SCRIPT,
        LINEAR_SPLITS,
        ENV_CONTRACT,
        ENV_LOCK,
        PROTOCOL_05D1,
        LOSS_CONTRACT_05D1,
        LOCK_05D1,
        PROFILE_GRID_05D1,
        DATASET_ROLES_05D1,
        CONFIRMATORY_SEEDS_05D1,
        NEURAL_IMPL_SCRIPT,
        LOCK_05D2,
        RESULT_MANIFEST_05D2,
        CONTRACT_05D3,
        PROFILE_TABLE_05D3,
        LOCK_05D3,
    ]
    for path in required_paths:
        require_file(path)

    if FINAL_GATE_DATA_ROOT.exists() and any(item.is_file() for item in FINAL_GATE_DATA_ROOT.rglob("*")):
        raise RuntimeError(
            "Final gate-calibration simulations already exist. 05d4 is the frozen next stage and does not consume them."
        )

    future = [relative_path(path) for path in FUTURE_RESULT_PATHS if is_nonempty_path(path)]
    if future:
        raise RuntimeError("05d4 must precede 05d5/05e. Found:\n- " + "\n- ".join(future))

    env_contract = load_json(ENV_CONTRACT)
    env_lock = load_json(ENV_LOCK)
    if clean_text(env_lock.get("status")) != "FROZEN_NEURAL_RUNTIME_ENVIRONMENT_BEFORE_NEURAL_MODEL_FITTING":
        raise RuntimeError("Unexpected 05d0 lock status.")
    verify_generated_files(env_lock, "05d0")
    expected_env_hash = clean_text(env_lock.get("contract", {}).get("sha256"))
    if not expected_env_hash or sha256_file(ENV_CONTRACT) != expected_env_hash:
        raise RuntimeError("05d0 environment contract hash mismatch.")
    assert_cuda_runtime(env_contract)

    protocol_05d1 = load_json(PROTOCOL_05D1)
    loss_contract = load_json(LOSS_CONTRACT_05D1)
    lock_05d1 = load_json(LOCK_05D1)
    if clean_text(lock_05d1.get("status")) != "FROZEN_NEURAL_SIMULATION_PROTOCOL_BEFORE_NEURAL_MODEL_FITTING":
        raise RuntimeError("Unexpected 05d1 lock status.")
    verify_generated_files(lock_05d1, "05d1")
    expected_protocol_hash = clean_text(lock_05d1.get("protocol", {}).get("sha256"))
    if not expected_protocol_hash or sha256_file(PROTOCOL_05D1) != expected_protocol_hash:
        raise RuntimeError("05d1 protocol hash mismatch.")

    upstream_hashes_05d1 = protocol_05d1.get("upstream_hashes", {})
    for relative_name, expected_hash in upstream_hashes_05d1.items():
        candidate = resolve_project_path(relative_name)
        require_file(candidate)
        if sha256_file(candidate) != clean_text(expected_hash):
            raise RuntimeError(f"05d1 frozen upstream hash mismatch: {relative_name}")

    stages = [clean_text(item.get("stage")) for item in protocol_05d1.get("stage_order", [])]
    if stages != ["05d2", "05d3", "05d4", "05d5", "05e"]:
        raise RuntimeError(f"Frozen 05d1 stage order changed: {stages}")

    scope = protocol_05d1.get("confirmatory_ablation", {})
    if int(scope.get("fit_evaluations", -1)) != EXPECTED_FOLD_SEED_EVALUATIONS:
        raise RuntimeError("05d1 confirmatory fit-evaluation count changed.")
    if int(scope.get("optimization_runs_including_refit", -1)) != EXPECTED_OPTIMIZATION_RUNS:
        raise RuntimeError("05d1 confirmatory optimization-run count changed.")
    if list(scope.get("variants", [])) != list(NEURAL_VARIANTS):
        raise RuntimeError("05d1 confirmatory variant order changed.")

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
    state_05d3 = contract_05d3.get("scientific_state", {})
    forbidden_true = [
        "TARGET_model_fitting_performed",
        "confirmatory_replicates_11_to_20_accessed",
        "confirmatory_neural_ablation_performed",
        "final_gate_calibration_data_used",
        "representation_method_selected",
    ]
    active = [key for key in forbidden_true if bool(state_05d3.get(key, False))]
    if active:
        raise RuntimeError("05d3 prospective state violated: " + ", ".join(active))

    profile_grid = pd.read_csv(PROFILE_GRID_05D1, dtype=str).fillna("")
    selected_profile = contract_05d3.get("selected_profile", {})
    selected_profile_id = clean_text(selected_profile.get("profile_id"))
    if not selected_profile_id or clean_text(lock_05d3.get("selected_profile_id")) != selected_profile_id:
        raise RuntimeError("05d3 selected-profile ID is missing or inconsistent.")
    selected_rows = profile_grid[profile_grid["profile_id"].eq(selected_profile_id)]
    if selected_rows.shape[0] != 1:
        raise RuntimeError("05d3 selected profile is not unique in the 05d1 grid.")
    selected_row = selected_rows.iloc[0]
    source_row = selected_profile.get("source_profile_grid_row", {})
    for column in profile_grid.columns:
        if clean_text(source_row.get(column)) != clean_text(selected_row[column]):
            raise RuntimeError(f"05d3 selected profile differs from 05d1 at {column}.")
    profile = neural.parse_profile(selected_row)

    roles = pd.read_csv(DATASET_ROLES_05D1, dtype=str, low_memory=False).fillna("")
    confirmatory_roles = roles[roles["neural_dataset_role"].eq("confirmatory_ablation")].copy()
    if confirmatory_roles.shape[0] != EXPECTED_CONFIRMATORY_DATASETS:
        raise RuntimeError("05d1 confirmatory dataset count is not 360.")
    confirmatory_roles["replicate_index"] = pd.to_numeric(confirmatory_roles["replicate_index"], errors="raise").astype(int)
    confirmatory_roles["simulation_seed"] = pd.to_numeric(confirmatory_roles["simulation_seed"], errors="raise").astype(int)
    if set(confirmatory_roles["replicate_index"].unique()) != set(range(11, 21)):
        raise RuntimeError("Confirmatory roles are not exactly replicate indices 11-20.")

    manifest = pd.read_csv(SIMULATION_MANIFEST, dtype=str, low_memory=False).fillna("")
    if manifest.shape[0] != 720:
        raise RuntimeError("Frozen simulation manifest no longer contains 720 datasets.")
    manifest_lookup = build_manifest_lookup(manifest)

    split_table = pd.read_csv(LINEAR_SPLITS, dtype=str, low_memory=False).fillna("")
    split_lookup = build_split_lookup(split_table)

    seed_table = pd.read_csv(CONFIRMATORY_SEEDS_05D1, dtype=str, low_memory=False).fillna("")
    seed_lookup = build_seed_lookup(seed_table)

    for _, row in confirmatory_roles.iterrows():
        key = (clean_text(row["scenario_id"]), int(row["replicate_index"]), int(row["simulation_seed"]))
        if key not in manifest_lookup:
            raise RuntimeError(f"Confirmatory dataset missing from manifest: {key}")
        if key not in split_lookup or set(split_lookup[key]) != set(range(1, 6)):
            raise RuntimeError(f"Confirmatory dataset missing exact 05c1 five-fold split: {key}")
        n_samples = int(float(row["n_samples"]))
        fold_sets = [set(split_lookup[key][fold].tolist()) for fold in range(1, 6)]
        for left in range(5):
            for right in range(left + 1, 5):
                if fold_sets[left] & fold_sets[right]:
                    raise RuntimeError(f"05c1 test folds overlap for {key}.")
        if set().union(*fold_sets) != set(range(n_samples)):
            raise RuntimeError(f"05c1 test folds do not cover all samples for {key}.")
        for variant_id in NEURAL_VARIANTS:
            for fold in range(1, 6):
                for base_seed in CONFIRMATORY_MODEL_BASE_SEEDS:
                    if (*key, variant_id, fold, base_seed) not in seed_lookup:
                        raise RuntimeError(f"Frozen confirmatory seed missing: {(*key, variant_id, fold, base_seed)}")

    variant_specs = build_variant_specs(loss_contract)
    stability = loss_contract.get("stability_gate", {})
    if bool(stability.get("training_component", True)) or not bool(stability.get("resampling_component", False)):
        raise RuntimeError("05d1 stability-gate training/resampling semantics changed.")

    return {
        "env_contract": env_contract,
        "protocol_05d1": protocol_05d1,
        "loss_contract": loss_contract,
        "contract_05d3": contract_05d3,
        "profile": profile,
        "variant_specs": variant_specs,
        "confirmatory_roles": confirmatory_roles,
        "manifest_lookup": manifest_lookup,
        "split_lookup": split_lookup,
        "seed_lookup": seed_lookup,
        "hashes": {relative_path(path): sha256_file(path) for path in required_paths},
    }


# -----------------------------------------------------------------------------
# V0-V5 model family
# -----------------------------------------------------------------------------

class SharedOnlyEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, shared_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.shared_head = nn.Linear(hidden_dim, shared_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared_head(self.body(x))


class SharedPrivateEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, shared_dim: int, private_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.shared_head = nn.Linear(hidden_dim, shared_dim)
        self.private_head = nn.Linear(hidden_dim, private_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(x)
        return self.shared_head(hidden), self.private_head(hidden)


class VariationalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, shared_dim: int, private_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.shared_mean = nn.Linear(hidden_dim, shared_dim)
        self.shared_logvar = nn.Linear(hidden_dim, shared_dim)
        self.private_mean = nn.Linear(hidden_dim, private_dim)
        self.private_logvar = nn.Linear(hidden_dim, private_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.body(x)
        return self.shared_mean(hidden), self.shared_logvar(hidden), self.private_mean(hidden), self.private_logvar(hidden)


class VariantDecoder(nn.Module):
    def __init__(self, shared_dim: int, private_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.private_dim = int(private_dim)
        self.first = nn.Linear(shared_dim + private_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, shared: torch.Tensor, private: torch.Tensor | None = None) -> torch.Tensor:
        if self.private_dim > 0:
            if private is None:
                raise RuntimeError("Private latent input is required by this decoder.")
            combined = torch.cat([shared, private], dim=1)
        else:
            combined = shared
        return self.output(self.activation(self.norm(self.first(combined))))


class NeuralAblationModel(nn.Module):
    def __init__(self, input_dims: dict[str, int], profile: Any, variant: VariantSpec) -> None:
        super().__init__()
        self.profile = profile
        self.variant = variant
        self.private_dim = int(profile.private_dim) if variant.shared_private_decomposition else 0
        if variant.variational_latent:
            self.encoders = nn.ModuleDict({
                view: VariationalEncoder(int(input_dims[view]), int(profile.hidden_dim), int(profile.shared_dim), int(profile.private_dim))
                for view in VIEWS
            })
        elif variant.shared_private_decomposition:
            self.encoders = nn.ModuleDict({
                view: SharedPrivateEncoder(int(input_dims[view]), int(profile.hidden_dim), int(profile.shared_dim), int(profile.private_dim))
                for view in VIEWS
            })
        else:
            self.encoders = nn.ModuleDict({
                view: SharedOnlyEncoder(int(input_dims[view]), int(profile.hidden_dim), int(profile.shared_dim))
                for view in VIEWS
            })
        self.decoders = nn.ModuleDict({
            view: VariantDecoder(int(profile.shared_dim), self.private_dim, int(profile.hidden_dim), int(input_dims[view]))
            for view in VIEWS
        })
        self.reset_frozen_parameters()

    def reset_frozen_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)

    def encode(
        self,
        views: dict[str, torch.Tensor],
        *,
        stochastic_latent: bool,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
        shared: dict[str, torch.Tensor] = {}
        private: dict[str, torch.Tensor] = {}
        stats: dict[str, dict[str, torch.Tensor]] = {}
        for view in VIEWS:
            if self.variant.variational_latent:
                sm, sl, pm, pl = self.encoders[view](views[view])
                shared[view] = self.reparameterize(sm, sl) if stochastic_latent else sm
                private[view] = self.reparameterize(pm, pl) if stochastic_latent else pm
                stats[view] = {"shared_mean": sm, "shared_logvar": sl, "private_mean": pm, "private_logvar": pl}
            elif self.variant.shared_private_decomposition:
                shared[view], private[view] = self.encoders[view](views[view])
            else:
                shared[view] = self.encoders[view](views[view])
                private[view] = torch.zeros((views[view].shape[0], 0), dtype=views[view].dtype, device=views[view].device)
        return shared, private, stats

    def patient_shared(self, shared: dict[str, torch.Tensor], active_mask: torch.Tensor) -> torch.Tensor:
        stack = torch.stack([shared[view] for view in VIEWS], dim=1)
        mask = active_mask.to(dtype=stack.dtype).unsqueeze(-1)
        return (stack * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def decode(self, view: str, patient_shared: torch.Tensor, private: torch.Tensor | None) -> torch.Tensor:
        return self.decoders[view](patient_shared, private)

    def zero_private(self, n_samples: int, device: torch.device) -> torch.Tensor | None:
        if self.private_dim <= 0:
            return None
        return torch.zeros((n_samples, self.private_dim), dtype=torch.float32, device=device)

    def decoder_shared_energy(self) -> np.ndarray:
        energy = np.zeros((int(self.profile.shared_dim), len(VIEWS)), dtype=np.float64)
        with torch.no_grad():
            for view in VIEWS:
                weight = self.decoders[view].first.weight[:, : int(self.profile.shared_dim)].detach().float().cpu().numpy()
                energy[:, VIEW_INDEX[view]] = np.sum(weight**2, axis=0)
        total = np.sum(energy, axis=1, keepdims=True)
        return np.divide(energy, total, out=np.zeros_like(energy), where=total > 0)


# -----------------------------------------------------------------------------
# CUDA objective and fitting
# -----------------------------------------------------------------------------

def numpy_views_to_cuda(views: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for view in VIEWS:
        tensor = torch.as_tensor(views[view], dtype=torch.float32, device="cuda:0")
        if tensor.device.type != "cuda":
            raise RuntimeError("Heavy neural tensor was not placed on CUDA.")
        output[view] = tensor
    return output


def availability_to_cuda(availability: np.ndarray) -> torch.Tensor:
    tensor = torch.as_tensor(availability, dtype=torch.bool, device="cuda:0")
    if tensor.device.type != "cuda":
        raise RuntimeError("Availability tensor was not placed on CUDA.")
    return tensor


def calculate_null_mse(views: dict[str, torch.Tensor]) -> dict[str, float]:
    output: dict[str, float] = {}
    for view in VIEWS:
        value = float(torch.mean(views[view] ** 2).detach().cpu())
        output[view] = max(value, NULL_MSE_FLOOR) if math.isfinite(value) and value > 0 else NULL_MSE_FLOOR
    return output


def zero_scalar(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), dtype=reference.dtype, device=reference.device)


def masked_covariance_penalty(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weights = mask.to(dtype=left.dtype)
    count = weights.sum()
    denom = count.clamp_min(1.0)
    left_mean = (left * weights.unsqueeze(1)).sum(dim=0, keepdim=True) / denom
    right_mean = (right * weights.unsqueeze(1)).sum(dim=0, keepdim=True) / denom
    left_centered = (left - left_mean) * weights.unsqueeze(1)
    right_centered = (right - right_mean) * weights.unsqueeze(1)
    covariance = (left_centered.T @ right_centered) / (count - 1.0).clamp_min(1.0)
    valid = (count >= 2.0).to(dtype=left.dtype)
    return torch.sum(covariance**2) * valid, valid


def masked_symmetric_info_nce(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    left = F.normalize(left, p=2, dim=1, eps=1e-8)
    right = F.normalize(right, p=2, dim=1, eps=1e-8)
    logits = (left @ right.T) / float(temperature)
    valid = mask.to(dtype=logits.dtype)
    count = valid.sum()
    pair_valid = (count >= 2.0).to(dtype=logits.dtype)
    negative_large = torch.full_like(logits, -1.0e9)
    candidate_mask = mask.unsqueeze(0)
    logits_lr = torch.where(candidate_mask, logits, negative_large)
    logits_rl = torch.where(candidate_mask, logits.T, negative_large)
    positive = torch.diagonal(logits)
    loss_lr = ((-positive + torch.logsumexp(logits_lr, dim=1)) * valid).sum() / count.clamp_min(1.0)
    loss_rl = ((-positive + torch.logsumexp(logits_rl, dim=1)) * valid).sum() / count.clamp_min(1.0)
    return 0.5 * (loss_lr + loss_rl) * pair_valid, pair_valid


def make_modality_dropout(
    neural: Any,
    availability: torch.Tensor,
    probability: float,
    fit_seed: int,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if probability <= 0:
        return availability, torch.zeros_like(availability, dtype=torch.bool)
    candidates = availability.sum(dim=1) >= 2
    generator = torch.Generator(device="cuda")
    generator.manual_seed(neural.stable_seed(fit_seed, "modality_dropout", epoch))
    trigger = (
        torch.rand((availability.shape[0],), generator=generator, device=availability.device, dtype=torch.float32)
        < float(probability)
    ) & candidates
    random_values = torch.rand(availability.shape, generator=generator, device=availability.device, dtype=torch.float32)
    random_values = torch.where(availability, random_values, torch.full_like(random_values, 2.0))
    selected_view = torch.argmin(random_values, dim=1)
    selected_one_hot = F.one_hot(selected_view, num_classes=len(VIEWS)).to(dtype=torch.bool)
    dropped = selected_one_hot & trigger.unsqueeze(1)
    return availability & ~dropped, dropped


def reconstruction_loss(
    model: NeuralAblationModel,
    views: dict[str, torch.Tensor],
    private: dict[str, torch.Tensor],
    patient_shared: torch.Tensor,
    active_mask: torch.Tensor,
    null_mse: dict[str, float],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    validity: list[torch.Tensor] = []
    for view in VIEWS:
        mask = active_mask[:, VIEW_INDEX[view]].to(dtype=patient_shared.dtype)
        prediction = model.decode(view, patient_shared, private[view] if model.private_dim > 0 else None)
        per_sample = torch.mean((prediction - views[view]) ** 2, dim=1)
        count = mask.sum()
        valid = (count > 0).to(dtype=patient_shared.dtype)
        mse = (per_sample * mask).sum() / count.clamp_min(1.0)
        losses.append((mse / max(float(null_mse[view]), NULL_MSE_FLOOR)) * valid)
        validity.append(valid)
    return torch.stack(losses).sum() / torch.stack(validity).sum().clamp_min(1.0)


def shared_alignment_loss(shared: dict[str, torch.Tensor], patient_shared: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    validity: list[torch.Tensor] = []
    for view in VIEWS:
        mask = active_mask[:, VIEW_INDEX[view]].to(dtype=patient_shared.dtype)
        per_sample = torch.mean((shared[view] - patient_shared) ** 2, dim=1)
        count = mask.sum()
        valid = (count > 0).to(dtype=patient_shared.dtype)
        losses.append(((per_sample * mask).sum() / count.clamp_min(1.0)) * valid)
        validity.append(valid)
    return torch.stack(losses).sum() / torch.stack(validity).sum().clamp_min(1.0)


def orthogonality_loss(shared: dict[str, torch.Tensor], private: dict[str, torch.Tensor], active_mask: torch.Tensor) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    validity: list[torch.Tensor] = []
    for view in VIEWS:
        loss, valid = masked_covariance_penalty(shared[view], private[view], active_mask[:, VIEW_INDEX[view]])
        losses.append(loss)
        validity.append(valid)
    return torch.stack(losses).sum() / torch.stack(validity).sum().clamp_min(1.0)


def private_separation_loss(private: dict[str, torch.Tensor], active_mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    validity: list[torch.Tensor] = []
    for left_index, left_view in enumerate(VIEWS):
        for right_view in VIEWS[left_index + 1 :]:
            mask = active_mask[:, VIEW_INDEX[left_view]] & active_mask[:, VIEW_INDEX[right_view]]
            loss, valid = masked_covariance_penalty(private[left_view], private[right_view], mask)
            losses.append(loss)
            validity.append(valid)
    return torch.stack(losses).sum() / torch.stack(validity).sum().clamp_min(1.0) if losses else zero_scalar(reference)


def contrastive_loss(shared: dict[str, torch.Tensor], active_mask: torch.Tensor, temperature: float) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    validity: list[torch.Tensor] = []
    for left_index, left_view in enumerate(VIEWS):
        for right_view in VIEWS[left_index + 1 :]:
            mask = active_mask[:, VIEW_INDEX[left_view]] & active_mask[:, VIEW_INDEX[right_view]]
            loss, valid = masked_symmetric_info_nce(shared[left_view], shared[right_view], mask, temperature)
            losses.append(loss)
            validity.append(valid)
    return torch.stack(losses).sum() / torch.stack(validity).sum().clamp_min(1.0) if losses else zero_scalar(shared[VIEWS[0]])


def dropped_view_reconstruction_loss(
    model: NeuralAblationModel,
    views: dict[str, torch.Tensor],
    patient_shared: torch.Tensor,
    dropped_mask: torch.Tensor,
    null_mse: dict[str, float],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    validity: list[torch.Tensor] = []
    zero_private = model.zero_private(patient_shared.shape[0], patient_shared.device)
    for view in VIEWS:
        mask = dropped_mask[:, VIEW_INDEX[view]].to(dtype=patient_shared.dtype)
        prediction = model.decode(view, patient_shared, zero_private)
        per_sample = torch.mean((prediction - views[view]) ** 2, dim=1)
        count = mask.sum()
        valid = (count > 0).to(dtype=patient_shared.dtype)
        mse = (per_sample * mask).sum() / count.clamp_min(1.0)
        losses.append((mse / max(float(null_mse[view]), NULL_MSE_FLOOR)) * valid)
        validity.append(valid)
    return torch.stack(losses).sum() / torch.stack(validity).sum().clamp_min(1.0)


def systematic_missing_view_loss(
    model: NeuralAblationModel,
    views: dict[str, torch.Tensor],
    shared: dict[str, torch.Tensor],
    availability: torch.Tensor,
    null_mse: dict[str, float],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    validity: list[torch.Tensor] = []
    zero_private = model.zero_private(availability.shape[0], availability.device)
    available_count = availability.sum(dim=1)
    for omitted_view in VIEWS:
        target_index = VIEW_INDEX[omitted_view]
        target_available = availability[:, target_index]
        valid_bool = target_available & (available_count - target_available.to(available_count.dtype) >= 1)
        mask = valid_bool.to(dtype=torch.float32)
        omitted_active = availability.clone()
        omitted_active[:, target_index] = False
        patient_shared = model.patient_shared(shared, omitted_active)
        prediction = model.decode(omitted_view, patient_shared, zero_private)
        per_sample = torch.mean((prediction - views[omitted_view]) ** 2, dim=1)
        count = mask.sum()
        valid = (count > 0).to(dtype=torch.float32)
        mse = (per_sample * mask).sum() / count.clamp_min(1.0)
        losses.append((mse / max(float(null_mse[omitted_view]), NULL_MSE_FLOOR)) * valid)
        validity.append(valid)
    return torch.stack(losses).sum() / torch.stack(validity).sum().clamp_min(1.0)


def kl_loss(stats: dict[str, dict[str, torch.Tensor]], active_mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if not stats:
        return zero_scalar(reference)
    losses: list[torch.Tensor] = []
    validity: list[torch.Tensor] = []
    for view in VIEWS:
        record = stats[view]
        shared_kl = -0.5 * (1.0 + record["shared_logvar"] - record["shared_mean"] ** 2 - torch.exp(record["shared_logvar"]))
        private_kl = -0.5 * (1.0 + record["private_logvar"] - record["private_mean"] ** 2 - torch.exp(record["private_logvar"]))
        per_sample = 0.5 * (torch.mean(shared_kl, dim=1) + torch.mean(private_kl, dim=1))
        mask = active_mask[:, VIEW_INDEX[view]].to(dtype=per_sample.dtype)
        count = mask.sum()
        valid = (count > 0).to(dtype=per_sample.dtype)
        losses.append(((per_sample * mask).sum() / count.clamp_min(1.0)) * valid)
        validity.append(valid)
    return torch.stack(losses).sum() / torch.stack(validity).sum().clamp_min(1.0)


def vae_kl_weight(epoch: int, planned_epochs: int) -> float:
    warmup_epochs = max(1, int(math.ceil(VAE_KL_WARMUP_FRACTION * int(planned_epochs))))
    return VAE_KL_FINAL_WEIGHT * min(1.0, max(0.0, float(epoch) / warmup_epochs))


def compute_objective(
    neural: Any,
    model: NeuralAblationModel,
    views: dict[str, torch.Tensor],
    availability: torch.Tensor,
    null_mse: dict[str, float],
    *,
    stochastic_dropout: bool,
    stochastic_latent: bool,
    fit_seed: int,
    epoch: int,
    planned_epochs: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    variant = model.variant
    shared, private, stats = model.encode(views, stochastic_latent=stochastic_latent)
    if stochastic_dropout and variant.modality_dropout:
        active_mask, dropped_mask = make_modality_dropout(
            neural, availability, float(model.profile.modality_dropout_probability), fit_seed, epoch
        )
    else:
        active_mask = availability
        dropped_mask = torch.zeros_like(availability, dtype=torch.bool)
    patient_shared = model.patient_shared(shared, active_mask)
    reconstruction = reconstruction_loss(model, views, private, patient_shared, active_mask, null_mse)
    alignment = shared_alignment_loss(shared, patient_shared, active_mask) if variant.shared_private_decomposition else zero_scalar(patient_shared)
    orthogonality = orthogonality_loss(shared, private, active_mask) if variant.shared_private_decomposition else zero_scalar(patient_shared)
    separation = private_separation_loss(private, active_mask, patient_shared) if variant.shared_private_decomposition else zero_scalar(patient_shared)
    contrastive = contrastive_loss(shared, active_mask, float(model.profile.contrastive_temperature)) if variant.paired_contrastive_alignment else zero_scalar(patient_shared)
    if variant.missing_view_reconstruction:
        cross_reconstruction = (
            dropped_view_reconstruction_loss(model, views, patient_shared, dropped_mask, null_mse)
            if stochastic_dropout and variant.modality_dropout
            else systematic_missing_view_loss(model, views, shared, availability, null_mse)
        )
    else:
        cross_reconstruction = zero_scalar(patient_shared)
    kl = kl_loss(stats, active_mask, patient_shared) if variant.variational_latent else zero_scalar(patient_shared)
    kl_weight = vae_kl_weight(epoch, planned_epochs) if variant.variational_latent else 0.0

    total = float(model.profile.reconstruction_weight) * reconstruction
    if variant.shared_private_decomposition:
        total = total + float(model.profile.shared_alignment_weight) * alignment
        total = total + float(model.profile.orthogonality_weight) * orthogonality
        total = total + float(model.profile.private_separation_weight) * separation
    if variant.paired_contrastive_alignment:
        total = total + float(model.profile.contrastive_weight) * contrastive
    if variant.missing_view_reconstruction:
        total = total + float(model.profile.cross_reconstruction_weight) * cross_reconstruction
    if variant.variational_latent:
        total = total + float(kl_weight) * kl

    components = {
        "total": total,
        "reconstruction": reconstruction,
        "shared_alignment": alignment,
        "contrastive": contrastive,
        "orthogonality": orthogonality,
        "private_separation": separation,
        "cross_reconstruction": cross_reconstruction,
        "kl": kl,
        "kl_weight": torch.as_tensor(kl_weight, dtype=total.dtype, device=total.device),
    }
    return total, components


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def construct_model(input_dims: dict[str, int], profile: Any, variant: VariantSpec, seed: int) -> NeuralAblationModel:
    seed_everything(seed)
    model = NeuralAblationModel(input_dims, profile, variant).to(device="cuda:0", dtype=torch.float32)
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("Neural ablation model is not on CUDA.")
    return model


def make_optimizer(model: nn.Module, profile: Any) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(profile.learning_rate),
        weight_decay=float(profile.weight_decay),
        foreach=False,
        fused=False,
    )


def train_with_early_stopping(
    neural: Any,
    profile: Any,
    variant: VariantSpec,
    fit_views_np: dict[str, np.ndarray],
    fit_availability_np: np.ndarray,
    early_views_np: dict[str, np.ndarray],
    early_availability_np: np.ndarray,
    fit_seed: int,
) -> tuple[int, dict[str, Any]]:
    input_dims = {view: int(fit_views_np[view].shape[1]) for view in VIEWS}
    fit_views = numpy_views_to_cuda(fit_views_np)
    fit_availability = availability_to_cuda(fit_availability_np)
    early_views = numpy_views_to_cuda(early_views_np)
    early_availability = availability_to_cuda(early_availability_np)
    null_mse = calculate_null_mse(fit_views)
    model = construct_model(input_dims, profile, variant, fit_seed)
    optimizer = make_optimizer(model, profile)

    best_value = float("inf")
    best_epoch = 1
    no_improvement = 0
    epochs_run = 0
    final_training_tensor: torch.Tensor | None = None
    final_early_value: float | None = None
    early_components: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()

    for epoch in range(1, int(profile.maximum_epochs) + 1):
        epochs_run = epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, _ = compute_objective(
            neural, model, fit_views, fit_availability, null_mse,
            stochastic_dropout=True,
            stochastic_latent=bool(variant.variational_latent),
            fit_seed=fit_seed,
            epoch=epoch,
            planned_epochs=int(profile.maximum_epochs),
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM, error_if_nonfinite=True)
        optimizer.step()
        final_training_tensor = total.detach()

        model.eval()
        with torch.no_grad():
            early_total, early_components = compute_objective(
                neural, model, early_views, early_availability, null_mse,
                stochastic_dropout=False,
                stochastic_latent=False,
                fit_seed=fit_seed,
                epoch=epoch,
                planned_epochs=int(profile.maximum_epochs),
            )
        early_value = float(early_total.detach().cpu())
        final_early_value = early_value
        if early_value < best_value - float(profile.min_delta):
            best_value = early_value
            best_epoch = epoch
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= int(profile.patience):
            break

    if final_training_tensor is None or early_components is None or not math.isfinite(best_value):
        raise RuntimeError("No finite early-stopping objective was obtained.")
    final_training_value = float(final_training_tensor.cpu())
    final_components = {name: float(value.detach().cpu()) for name, value in early_components.items()}
    wall_seconds = time.perf_counter() - started
    audit = {
        "fit_stage": "confirmatory_early_stop",
        "variant_id": variant.variant_id,
        "fit_seed": int(fit_seed),
        "profile_id": profile.profile_id,
        "epochs_run": int(epochs_run),
        "selected_epoch": int(best_epoch),
        "best_early_stop_objective": float(best_value),
        "final_early_stop_objective": final_early_value,
        "final_training_objective": final_training_value,
        "early_stopped": bool(epochs_run < int(profile.maximum_epochs)),
        "maximum_epochs": int(profile.maximum_epochs),
        "patience": int(profile.patience),
        "min_delta": float(profile.min_delta),
        "final_monitor_reconstruction": final_components["reconstruction"],
        "final_monitor_shared_alignment": final_components["shared_alignment"],
        "final_monitor_contrastive": final_components["contrastive"],
        "final_monitor_orthogonality": final_components["orthogonality"],
        "final_monitor_private_separation": final_components["private_separation"],
        "final_monitor_cross_reconstruction": final_components["cross_reconstruction"],
        "final_monitor_kl": final_components["kl"],
        "final_monitor_kl_weight": final_components["kl_weight"],
        "wall_seconds": float(wall_seconds),
        "device": "cuda:0",
        "fit_status": "PASS",
    }
    del optimizer, model, fit_views, fit_availability, early_views, early_availability
    return int(best_epoch), audit


def train_fixed_epochs(
    neural: Any,
    profile: Any,
    variant: VariantSpec,
    training_views_np: dict[str, np.ndarray],
    availability_np: np.ndarray,
    fit_seed: int,
    epochs: int,
) -> tuple[NeuralAblationModel, dict[str, Any]]:
    if epochs < 1:
        raise ValueError("Fixed epoch count must be positive.")
    input_dims = {view: int(training_views_np[view].shape[1]) for view in VIEWS}
    views = numpy_views_to_cuda(training_views_np)
    availability = availability_to_cuda(availability_np)
    null_mse = calculate_null_mse(views)
    model = construct_model(input_dims, profile, variant, fit_seed)
    optimizer = make_optimizer(model, profile)
    final_total: torch.Tensor | None = None
    final_components: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()

    for epoch in range(1, int(epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, components = compute_objective(
            neural, model, views, availability, null_mse,
            stochastic_dropout=True,
            stochastic_latent=bool(variant.variational_latent),
            fit_seed=fit_seed,
            epoch=epoch,
            planned_epochs=int(epochs),
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM, error_if_nonfinite=True)
        optimizer.step()
        final_total = total.detach()
        final_components = {name: value.detach() for name, value in components.items()}

    if final_total is None or final_components is None:
        raise RuntimeError("Fixed-epoch refit produced no optimization step.")
    final_value = float(final_total.cpu())
    component_values = {name: float(value.cpu()) for name, value in final_components.items()}
    wall_seconds = time.perf_counter() - started
    model.eval()
    audit = {
        "fit_stage": "confirmatory_refit_full_training_fold",
        "variant_id": variant.variant_id,
        "fit_seed": int(fit_seed),
        "profile_id": profile.profile_id,
        "epochs_run": int(epochs),
        "selected_epoch": int(epochs),
        "final_training_objective": final_value,
        "final_training_reconstruction": component_values["reconstruction"],
        "final_training_shared_alignment": component_values["shared_alignment"],
        "final_training_contrastive": component_values["contrastive"],
        "final_training_orthogonality": component_values["orthogonality"],
        "final_training_private_separation": component_values["private_separation"],
        "final_training_cross_reconstruction": component_values["cross_reconstruction"],
        "final_training_kl": component_values["kl"],
        "final_training_kl_weight": component_values["kl_weight"],
        "wall_seconds": float(wall_seconds),
        "device": "cuda:0",
        "fit_status": "PASS",
    }
    del optimizer, views, availability
    return model, audit


# -----------------------------------------------------------------------------
# Held-out confirmatory scoring
# -----------------------------------------------------------------------------

def encode_numpy(model: NeuralAblationModel, views_np: dict[str, np.ndarray], availability_np: np.ndarray) -> dict[str, Any]:
    views = numpy_views_to_cuda(views_np)
    availability = availability_to_cuda(availability_np)
    model.eval()
    with torch.no_grad():
        shared, private, _ = model.encode(views, stochastic_latent=False)
        patient_shared = model.patient_shared(shared, availability)
    output = {
        "patient_shared": patient_shared.detach().cpu().numpy().astype(np.float64),
        "shared_by_view": {view: shared[view].detach().cpu().numpy().astype(np.float64) for view in VIEWS},
        "private_by_view": {view: private[view].detach().cpu().numpy().astype(np.float64) for view in VIEWS},
    }
    del views, availability
    return output


def null_mse_from_numpy(views: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        view: max(float(np.mean(np.asarray(views[view], dtype=np.float64) ** 2)), NULL_MSE_FLOOR)
        for view in VIEWS
    }


def normalized_reconstruction_against_training_null(
    prediction: np.ndarray, truth: np.ndarray, training_null_mse: float
) -> dict[str, float | None]:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if prediction.shape != truth.shape:
        raise RuntimeError("Prediction/truth shape mismatch.")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        return {"mse": None, "normalized_error": None, "score": None, "r2": None}
    mse = float(np.mean((prediction - truth) ** 2))
    normalized_error = mse / max(float(training_null_mse), NULL_MSE_FLOOR)
    return {
        "mse": mse,
        "normalized_error": float(normalized_error),
        "score": float(np.clip(1.0 - normalized_error, 0.0, 1.0)),
        "r2": float(1.0 - normalized_error),
    }


def predict_reconstruction_numpy(
    model: NeuralAblationModel, views_np: dict[str, np.ndarray], availability_np: np.ndarray
) -> dict[str, np.ndarray]:
    views = numpy_views_to_cuda(views_np)
    availability = availability_to_cuda(availability_np)
    model.eval()
    output: dict[str, np.ndarray] = {}
    with torch.no_grad():
        shared, private, _ = model.encode(views, stochastic_latent=False)
        patient_shared = model.patient_shared(shared, availability)
        for view in VIEWS:
            private_input: torch.Tensor | None
            if model.private_dim > 0:
                private_input = private[view].clone()
                unavailable = ~availability[:, VIEW_INDEX[view]]
                if bool(unavailable.any()):
                    private_input[unavailable, :] = 0.0
            else:
                private_input = None
            prediction = model.decode(view, patient_shared, private_input)
            output[view] = prediction.detach().cpu().numpy().astype(np.float64)
    del views, availability
    return output


def predict_missing_view_numpy(
    model: NeuralAblationModel,
    views_np: dict[str, np.ndarray],
    availability_np: np.ndarray,
    omitted_view: str,
) -> tuple[np.ndarray, np.ndarray]:
    views = numpy_views_to_cuda(views_np)
    availability = availability_to_cuda(availability_np)
    model.eval()
    with torch.no_grad():
        shared, _, _ = model.encode(views, stochastic_latent=False)
        omitted_active = availability.clone()
        omitted_active[:, VIEW_INDEX[omitted_view]] = False
        valid = omitted_active.sum(dim=1) >= 1
        patient_shared = model.patient_shared(shared, omitted_active)
        prediction = model.decode(
            omitted_view,
            patient_shared,
            model.zero_private(patient_shared.shape[0], patient_shared.device),
        )
        prediction_np = prediction.detach().cpu().numpy().astype(np.float64)
        valid_np = valid.detach().cpu().numpy().astype(bool)
    del views, availability
    return prediction_np, valid_np


def evaluate_heldout(
    neural: Any,
    linear: Any,
    arrays: dict[str, np.ndarray],
    variant: VariantSpec,
    model: NeuralAblationModel,
    fold_data: FoldRefitData,
) -> dict[str, Any]:
    training_views = fold_data.training_views
    training_availability = fold_data.training_availability
    test_views = fold_data.test_views
    test_truth = fold_data.test_truth
    test_availability = fold_data.test_availability
    training_indices = fold_data.training_indices
    test_indices = fold_data.test_indices

    null_mse = null_mse_from_numpy(training_views)
    train_encoded = encode_numpy(model, training_views, training_availability)
    test_encoded = encode_numpy(model, test_views, test_availability)
    reconstruction_prediction = predict_reconstruction_numpy(model, test_views, test_availability)

    reconstruction: dict[str, dict[str, float | None]] = {}
    for view in VIEWS:
        reconstruction[view] = normalized_reconstruction_against_training_null(
            reconstruction_prediction[view], test_truth[view], null_mse[view]
        )
    mean_reconstruction_error = linear.nanmean_or_none(reconstruction[view]["normalized_error"] for view in VIEWS)
    heldout_reconstruction_score = (
        float(np.clip(1.0 - mean_reconstruction_error, 0.0, 1.0))
        if mean_reconstruction_error is not None
        else None
    )

    row_source = {view: arrays[f"row_source_{VIEW_PREFIX[view]}"].astype(int) for view in VIEWS}
    retrieval = linear.cross_view_retrieval_metrics(
        train_encoded["shared_by_view"], test_encoded["shared_by_view"], row_source, training_indices, test_indices
    )

    missing_view_r2: dict[str, float | None] = {}
    for omitted_view in VIEWS:
        prediction, valid = predict_missing_view_numpy(model, test_views, test_availability, omitted_view)
        if not valid.any():
            missing_view_r2[omitted_view] = None
        else:
            metric = normalized_reconstruction_against_training_null(
                prediction[valid, :], test_truth[omitted_view][valid, :], null_mse[omitted_view]
            )
            missing_view_r2[omitted_view] = metric["r2"]
    missing_view_score = linear.nanmean_or_none(
        np.clip(value, 0.0, 1.0) for value in missing_view_r2.values() if value is not None
    )

    energy = model.decoder_shared_energy()
    entropy = neural.normalized_entropy(energy)
    modality_balance_score = float(np.clip(np.median(entropy), 0.0, 1.0))

    biological_truth, _ = linear.integrated_anchor_truth(arrays, "biological")
    technical_truth, _ = linear.integrated_anchor_truth(arrays, "technical")
    test_shared = test_encoded["patient_shared"]
    shared_subspace = linear.subspace_metrics(test_shared, biological_truth[test_indices, :])
    technical_subspace = linear.subspace_metrics(test_shared, technical_truth[test_indices, :])

    private_values: list[float] = []
    if variant.shared_private_decomposition:
        for view in VIEWS:
            private_truth, _ = linear.row_aligned_truth(arrays, view, "private")
            result = linear.subspace_metrics(test_encoded["private_by_view"][view], private_truth[test_indices, :])
            if result["mean_canonical_correlation"] is not None:
                private_values.append(float(result["mean_canonical_correlation"]))
    private_recovery = float(np.mean(private_values)) if private_values else None

    raw_shared = shared_subspace["mean_canonical_correlation"]
    shared_component = float(raw_shared) if raw_shared is not None else 0.0
    private_component = float(private_recovery) if private_recovery is not None else 0.0
    retrieval_component = retrieval["random_adjusted_cross_view_mrr"]
    missing_component = missing_view_score
    utility_components = [
        (0.40, shared_component),
        (0.20, private_component),
        (0.20, retrieval_component),
        (0.10, missing_component),
        (0.10, modality_balance_score),
    ]
    simulation_utility = None
    if all(value is not None and math.isfinite(float(value)) for _, value in utility_components):
        simulation_utility = float(sum(weight * float(np.clip(value, 0.0, 1.0)) for weight, value in utility_components))

    shared_variance = np.var(test_shared, axis=0, ddof=1)
    effective_rank = linear.effective_rank(test_shared)
    latent_collapse = bool(effective_rank < 2.0 or float(np.median(shared_variance)) < 1e-4)

    return {
        "heldout_normalized_reconstruction_error": mean_reconstruction_error,
        "heldout_reconstruction_score": heldout_reconstruction_score,
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
        "missing_view_reconstruction_score": missing_view_score,
        "modality_contribution_entropy": float(np.median(entropy)),
        "modality_balance_score": modality_balance_score,
        "shared_subspace_recovery": raw_shared,
        "shared_subspace_minimum_correlation": shared_subspace["minimum_canonical_correlation"],
        "private_subspace_recovery": private_recovery,
        "technical_subspace_recovery": technical_subspace["mean_canonical_correlation"],
        "heldout_latent_effective_rank": effective_rank,
        "heldout_median_shared_variance": float(np.median(shared_variance)),
        "heldout_latent_collapse": latent_collapse,
        "simulation_utility": simulation_utility,
        "utility_shared_component": shared_component,
        "utility_private_component": private_component,
        "utility_retrieval_component": retrieval_component,
        "utility_missing_view_component": missing_component,
        "utility_modality_balance_component": modality_balance_score,
    }


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


def attach_audit_metadata(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    profile: Any,
    variant: VariantSpec,
    fold_index: int,
    base_model_seed: int,
    partition_seed: int,
) -> None:
    for row in rows:
        row.update(metadata)
        row.update(
            {
                "profile_id": profile.profile_id,
                "architecture_profile": profile.architecture_profile,
                "regularization_profile": profile.regularization_profile,
                "variant_id": variant.variant_id,
                "fold_index": int(fold_index),
                "base_model_seed": int(base_model_seed),
                "early_stop_partition_seed": int(partition_seed),
                "shared_latent_dim": int(profile.shared_dim),
                "private_latent_dim_per_view": int(profile.private_dim) if variant.shared_private_decomposition else 0,
            }
        )


def evaluate_fold_seed(
    neural: Any,
    linear: Any,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    profile: Any,
    variant: VariantSpec,
    fold_index: int,
    base_model_seed: int,
    seed_record: dict[str, int],
    fold_data: FoldRefitData,
) -> dict[str, Any]:
    fit_seed = seed_record["fit_seed"]
    refit_seed = seed_record["refit_seed"]
    partition_seed = seed_record["early_stop_partition_seed"]
    if fit_seed != refit_seed:
        raise RuntimeError("Frozen fit/refit seeds differ.")

    training_indices = fold_data.training_indices
    test_indices = fold_data.test_indices
    fit_indices, early_indices = deterministic_early_stop_partition(training_indices, partition_seed)
    if set(test_indices) & set(training_indices):
        raise RuntimeError("Confirmatory training/test overlap.")
    if set(test_indices) | set(training_indices) != set(range(metadata["n_samples"])):
        raise RuntimeError("Confirmatory split does not cover all samples.")

    torch.cuda.reset_peak_memory_stats(0)
    early_preprocessors = neural.fit_preprocessors(linear, arrays, fit_indices, profile)
    early_preprocessing = neural.preprocessing_audit_rows(
        arrays, early_preprocessors, fit_indices, profile, "confirmatory_early_stop_fit_only"
    )
    fit_views = neural.transform_views(arrays, early_preprocessors, fit_indices, complete=False)
    fit_availability = neural.view_availability(arrays, early_preprocessors, fit_indices)
    early_views = neural.transform_views(arrays, early_preprocessors, early_indices, complete=False)
    early_availability = neural.view_availability(arrays, early_preprocessors, early_indices)

    selected_epoch, early_audit = train_with_early_stopping(
        neural, profile, variant, fit_views, fit_availability, early_views, early_availability, fit_seed
    )
    del fit_views, fit_availability, early_views, early_availability, early_preprocessors

    model, refit_audit = train_fixed_epochs(
        neural, profile, variant, fold_data.training_views, fold_data.training_availability, refit_seed, selected_epoch
    )
    metrics = evaluate_heldout(neural, linear, arrays, variant, model, fold_data)
    max_memory = int(torch.cuda.max_memory_allocated(0))

    fit_audits = [early_audit, refit_audit]
    preprocessing_rows = [*early_preprocessing, *[dict(row) for row in fold_data.preprocessing_rows]]
    attach_audit_metadata(fit_audits, metadata, profile, variant, fold_index, base_model_seed, partition_seed)
    attach_audit_metadata(preprocessing_rows, metadata, profile, variant, fold_index, base_model_seed, partition_seed)

    metric_row = {
        **metadata,
        "profile_id": profile.profile_id,
        "architecture_profile": profile.architecture_profile,
        "regularization_profile": profile.regularization_profile,
        "variant_id": variant.variant_id,
        "primary_ablation_variant": bool(variant.variant_id in PRIMARY_ABLATION_VARIANTS),
        "secondary_exploratory_variant": bool(variant.variant_id == VAE_VARIANT),
        "stability_gate_declared": bool(variant.stability_gate),
        "stability_gate_applied_in_05d4": False,
        "variational_latent": bool(variant.variational_latent),
        "fold_index": int(fold_index),
        "base_model_seed": int(base_model_seed),
        "fit_seed": int(fit_seed),
        "refit_seed": int(refit_seed),
        "early_stop_partition_seed": int(partition_seed),
        "selected_epoch": int(selected_epoch),
        "n_fit_samples": int(fit_indices.size),
        "n_early_stop_samples": int(early_indices.size),
        "n_refit_training_samples": int(training_indices.size),
        "n_test_samples": int(test_indices.size),
        "shared_latent_dim": int(profile.shared_dim),
        "private_latent_dim_per_view": int(profile.private_dim) if variant.shared_private_decomposition else 0,
        "hidden_dim": int(profile.hidden_dim),
        "max_cuda_memory_allocated_bytes": max_memory,
        **metrics,
        "shard_status": "PASS",
        "shard_error_type": "",
        "shard_error_message": "",
    }
    del model
    return {
        "script_version": SCRIPT_VERSION,
        "scenario_id": metadata["scenario_id"],
        "replicate_index": metadata["replicate_index"],
        "simulation_seed": metadata["simulation_seed"],
        "variant_id": variant.variant_id,
        "fold_index": int(fold_index),
        "base_model_seed": int(base_model_seed),
        "profile_id": profile.profile_id,
        "input_path": metadata["input_path"],
        "input_sha256": metadata["input_sha256"],
        "metric_row": metric_row,
        "fit_audits": fit_audits,
        "preprocessing_rows": preprocessing_rows,
    }


def failed_shard_record(
    metadata: dict[str, Any],
    profile: Any,
    variant: VariantSpec,
    fold_index: int,
    base_model_seed: int,
    seed_record: dict[str, int],
    error: BaseException,
) -> dict[str, Any]:
    metric_row = {
        **metadata,
        "profile_id": profile.profile_id,
        "architecture_profile": profile.architecture_profile,
        "regularization_profile": profile.regularization_profile,
        "variant_id": variant.variant_id,
        "primary_ablation_variant": bool(variant.variant_id in PRIMARY_ABLATION_VARIANTS),
        "secondary_exploratory_variant": bool(variant.variant_id == VAE_VARIANT),
        "stability_gate_declared": bool(variant.stability_gate),
        "stability_gate_applied_in_05d4": False,
        "variational_latent": bool(variant.variational_latent),
        "fold_index": int(fold_index),
        "base_model_seed": int(base_model_seed),
        "fit_seed": int(seed_record["fit_seed"]),
        "refit_seed": int(seed_record["refit_seed"]),
        "early_stop_partition_seed": int(seed_record["early_stop_partition_seed"]),
        "selected_epoch": None,
        "shared_latent_dim": int(profile.shared_dim),
        "private_latent_dim_per_view": int(profile.private_dim) if variant.shared_private_decomposition else 0,
        "hidden_dim": int(profile.hidden_dim),
        "simulation_utility": None,
        "heldout_reconstruction_score": None,
        "random_adjusted_cross_view_mrr": None,
        "missing_view_reconstruction_score": None,
        "modality_balance_score": None,
        "shared_subspace_recovery": None,
        "private_subspace_recovery": None,
        "technical_subspace_recovery": None,
        "heldout_latent_collapse": None,
        "shard_status": "FAIL",
        "shard_error_type": type(error).__name__,
        "shard_error_message": clean_text(str(error))[:2000],
    }
    audit = {
        **metadata,
        "profile_id": profile.profile_id,
        "variant_id": variant.variant_id,
        "fold_index": int(fold_index),
        "base_model_seed": int(base_model_seed),
        "fit_seed": int(seed_record["fit_seed"]),
        "refit_seed": int(seed_record["refit_seed"]),
        "early_stop_partition_seed": int(seed_record["early_stop_partition_seed"]),
        "fit_stage": "confirmatory_fold_seed_failure",
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
        "variant_id": variant.variant_id,
        "fold_index": int(fold_index),
        "base_model_seed": int(base_model_seed),
        "profile_id": profile.profile_id,
        "input_path": metadata["input_path"],
        "input_sha256": metadata["input_sha256"],
        "metric_row": metric_row,
        "fit_audits": [audit],
        "preprocessing_rows": [],
    }


# -----------------------------------------------------------------------------
# Prospective confirmatory execution protocol
# -----------------------------------------------------------------------------

def build_execution_protocol(upstream: dict[str, Any]) -> dict[str, Any]:
    variant_specs = upstream["variant_specs"]

    return {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "scientific_state_before_fit": {
            "outcomes_loaded": False,
            "clinical_labels_loaded": False,
            "biological_annotations_used_for_model_selection": False,
            "TARGET_model_fitting_performed": False,
            "neural_profile_frozen": True,
            "confirmatory_replicates_11_to_20_accessed": False,
            "confirmatory_neural_ablation_performed": False,
            "confirmatory_V4_stability_performed": False,
            "final_gate_calibration_data_used": False,
            "representation_method_selected": False,
        },
        "scope": {
            "simulation_scenarios": EXPECTED_SCENARIOS,
            "confirmatory_datasets": EXPECTED_CONFIRMATORY_DATASETS,
            "confirmatory_replicate_indices": list(CONFIRMATORY_REPLICATES),
            "variants": list(NEURAL_VARIANTS),
            "primary_ablation_variants": list(PRIMARY_ABLATION_VARIANTS),
            "secondary_exploratory_variant": VAE_VARIANT,
            "folds_per_dataset": EXPECTED_FOLDS,
            "model_seeds_per_variant_fold": len(CONFIRMATORY_MODEL_BASE_SEEDS),
            "base_model_seeds": list(CONFIRMATORY_MODEL_BASE_SEEDS),
            "fold_seed_evaluations": EXPECTED_FOLD_SEED_EVALUATIONS,
            "optimization_runs_including_selected_epoch_refits": EXPECTED_OPTIMIZATION_RUNS,
            "stability_resample_fits_in_05d4": 0,
        },
        "frozen_profile": {
            "profile_id": upstream["profile"].profile_id,
            "architecture_profile": upstream["profile"].architecture_profile,
            "regularization_profile": upstream["profile"].regularization_profile,
            "source": relative_path(CONTRACT_05D3),
            "source_sha256": sha256_file(CONTRACT_05D3),
            "reused_without_retuning_by_all_V0_to_V5": True,
            "confirmatory_results_may_not_change_profile": True,
        },
        "heldout_design": {
            "outer_unit": "one frozen 05c1 test fold within each confirmatory simulation dataset",
            "fold_source": relative_path(LINEAR_SPLITS),
            "folds": EXPECTED_FOLDS,
            "every_sample_held_out_exactly_once_per_seed_and_variant": True,
            "within_training_early_stop_fraction": EARLY_STOP_FRACTION,
            "within_training_early_stop_count_rule": "ceil(0.20 * n_training), bounded to leave at least two fit samples",
            "early_stop_partition_rng": "numpy default_rng using frozen 05d1 early_stop_partition_seed",
            "heldout_test_used_for_epoch_selection": False,
            "heldout_test_used_for_preprocessing": False,
            "selected_epoch_refit": (
                "after early stopping, refit from frozen initialization on the entire training fold "
                "for exactly the selected epoch count, then evaluate the untouched test fold once"
            ),
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
            "optimizer": "AdamW foreach=False fused=False",
            "gradient_clip_global_norm": GRADIENT_CLIP_NORM,
            "CPU_allowed": [
                "file I/O",
                "pandas audit tables",
                "fold-local feature eligibility/ranking",
                "training-subset median imputation",
                "hashing/provenance",
                "aggregation",
            ],
        },
        "preprocessing": {
            "implementation_source": relative_path(NEURAL_IMPL_SCRIPT),
            "feature_budgets": PRIMARY_FEATURE_BUDGETS,
            "maximum_feature_missing_fraction": 0.20,
            "ranking": "training-subset pre-standardization variance descending with original-index tie break",
            "imputation": "training-subset feature median",
            "centering_scaling": "training-subset mean and sample standard deviation",
            "PCA": "deterministic CUDA float64 dual PCA with 05c1 sign orientation",
            "profile_PCA_targets": dict(upstream["profile"].pca_components),
            "global_rankings_used": False,
            "complete_simulation_truth_used_for_preprocessing": False,
        },
        "variant_contract": {
            variant_id: {
                "block_balanced_reconstruction": spec.block_balanced_reconstruction,
                "shared_private_decomposition": spec.shared_private_decomposition,
                "paired_contrastive_alignment": spec.paired_contrastive_alignment,
                "modality_dropout": spec.modality_dropout,
                "missing_view_reconstruction": spec.missing_view_reconstruction,
                "stability_gate_declared": spec.stability_gate,
                "stability_gate_is_training_component": False,
                "variational_latent": spec.variational_latent,
                "prespecified_component_hypothesis": spec.prespecified_component_hypothesis,
            }
            for variant_id, spec in variant_specs.items()
        },
        "V4_interpretation_guard": {
            "V4_training_objective_equals_V3_training_objective": True,
            "reason": "the frozen stability gate is a post-fit resampling component, not a training loss",
            "V4_minus_V3_may_be_called_stability_gate_effect_in_05d4": False,
            "confirmatory_stability_is_reserved_for": "05d5_evaluate_sable_mo_stability_gpu.py",
        },
        "V5_variational_extension": {
            "role": "secondary_exploratory",
            "KL_final_weight": VAE_KL_FINAL_WEIGHT,
            "KL_warmup_fraction": VAE_KL_WARMUP_FRACTION,
            "KL_weight_profile_tuned": False,
            "eligible_for_final_freeze": False,
        },
        "heldout_metrics": {
            "simulation_utility_weights": {
                "shared_subspace_recovery": 0.40,
                "private_subspace_recovery": 0.20,
                "random_adjusted_cross_view_mrr": 0.20,
                "missing_view_reconstruction_score": 0.10,
                "modality_balance_score": 0.10,
            },
            "V0_private_subspace_utility_component": 0.0,
            "shared_zero_truth_utility_component": 0.0,
            "truth_metrics_used_only_for_simulation_evaluation": True,
            "truth_metrics_used_for_training": False,
        },
        "aggregation": {
            "dataset_variant": "equal mean across 5 folds x 2 frozen seeds (10 held-out evaluations)",
            "scenario_variant": "equal mean across the 10 confirmatory replicate datasets",
            "overall_variant": "equal mean across 36 scenario means",
            "standard_error_unit": "36 scenario means",
            "model_or_profile_selection_in_05d4": False,
        },
        "prespecified_descriptive_ablation_deltas": [
            {"contrast_id": "C1_SHARED_PRIVATE", "higher": "V1_SHARED_PRIVATE_AE", "lower": "V0_BALANCED_AE"},
            {"contrast_id": "C2_CONTRASTIVE", "higher": "V2_CONTRASTIVE_SPAE", "lower": "V1_SHARED_PRIVATE_AE"},
            {"contrast_id": "C3_DROPOUT_MISSING_VIEW", "higher": "V3_DROPOUT_SPAE", "lower": "V2_CONTRASTIVE_SPAE"},
        ],
        "non_ablation_comparisons": {
            "V4_minus_V3": "not interpreted as a stability-gate ablation in 05d4",
            "V5": "secondary exploratory variational comparison only",
            "inferential_p_values_in_05d4": False,
        },
        "resumability": {
            "unit": "dataset x variant x fold x base model seed",
            "n_shards": EXPECTED_FOLD_SEED_EVALUATIONS,
            "compact_SHA256_derived_filenames": True,
            "existing_shards_verified_by": [
                "script version",
                "prospective execution protocol SHA256",
                "simulation input SHA256",
                "frozen profile ID",
                "scenario/replicate",
                "variant/fold/base seed",
            ],
        },
        "guardrails": [
            "Only simulation replicate indices 11-20 are eligible in this stage.",
            "Simulation replicate indices 1-10 are not used as confirmatory evidence.",
            "The 05d3 frozen profile is reused by all six neural variants without retuning.",
            "TARGET matrices, outcomes, clinical labels, and biological program annotations are not loaded.",
            "Final gate-calibration simulations are not accessed.",
            "The V4 stability resampling gate is not executed here; it is reserved for 05d5.",
            "All failed V0-V5 fold-seed evaluations remain visible and are never silently omitted.",
            "This stage does not choose the final representation method.",
        ],
        "upstream_hashes": upstream["hashes"],
        "implementation_source": {
            "path": relative_path(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "next_required_stage_if_all_fits_pass": "05d5_evaluate_sable_mo_stability_gpu.py",
    }


def protocol_core(protocol: dict[str, Any]) -> dict[str, Any]:
    output = dict(protocol)
    output.pop("created_at_utc", None)
    # The running source file is intentionally included in provenance, but
    # source hashing must not make a prospective protocol impossible to replay
    # after an identical file is copied to another filesystem timestamp.
    return output


def initialize_or_verify_protocol(upstream: dict[str, Any]) -> tuple[dict[str, Any], str]:
    proposed = build_execution_protocol(upstream)
    if OUTPUT_PROTOCOL.exists():
        existing = load_json(OUTPUT_PROTOCOL)
        if json.dumps(protocol_core(existing), sort_keys=True, separators=(",", ":")) != json.dumps(
            protocol_core(proposed), sort_keys=True, separators=(",", ":")
        ):
            raise RuntimeError(
                "Existing prospective 05d4 execution protocol differs from the current script. "
                "A new versioned stage is required rather than silently changing the confirmatory benchmark."
            )
        protocol = existing
    else:
        write_json_atomic(proposed, OUTPUT_PROTOCOL)
        protocol = proposed
    return protocol, sha256_file(OUTPUT_PROTOCOL)


# -----------------------------------------------------------------------------
# Deterministic implementation fixture
# -----------------------------------------------------------------------------

def run_reference_fixture(
    neural: Any,
    profile: Any,
    variant_specs: dict[str, VariantSpec],
) -> dict[str, Any]:
    rng = np.random.default_rng(20524)
    n = 34
    input_dims = {"RNA": 7, "METHYLATION": 8, "CNV": 6}
    latent = rng.normal(size=(n, 3))
    views: dict[str, np.ndarray] = {}
    for view in VIEWS:
        weights = rng.normal(scale=0.7, size=(3, input_dims[view]))
        views[view] = (latent @ weights + rng.normal(scale=0.45, size=(n, input_dims[view]))).astype(np.float32)

    availability = np.ones((n, len(VIEWS)), dtype=bool)
    availability[0, VIEW_INDEX["CNV"]] = False
    availability[1, VIEW_INDEX["METHYLATION"]] = False
    availability[2, VIEW_INDEX["RNA"]] = False

    checks: dict[str, Any] = {}
    for variant_id in ("V0_BALANCED_AE", "V4_SABLE_MO", "V5_SABLE_VAE"):
        variant = variant_specs[variant_id]
        seed = 20260824 + NEURAL_VARIANTS.index(variant_id)
        epochs = 7

        model_a, audit_a = train_fixed_epochs(
            neural=neural,
            profile=profile,
            variant=variant,
            training_views_np=views,
            availability_np=availability,
            fit_seed=seed,
            epochs=epochs,
        )
        state_a = {key: value.detach().cpu().clone() for key, value in model_a.state_dict().items()}
        scores_a = encode_numpy(model_a, variant, views, availability)["patient_shared"]
        del model_a

        model_b, audit_b = train_fixed_epochs(
            neural=neural,
            profile=profile,
            variant=variant,
            training_views_np=views,
            availability_np=availability,
            fit_seed=seed,
            epochs=epochs,
        )
        state_b = {key: value.detach().cpu().clone() for key, value in model_b.state_dict().items()}
        scores_b = encode_numpy(model_b, variant, views, availability)["patient_shared"]
        del model_b

        exact_state = list(state_a) == list(state_b) and all(torch.equal(state_a[key], state_b[key]) for key in state_a)
        exact_scores = bool(np.array_equal(scores_a, scores_b))
        exact_objective = audit_a["final_training_objective"] == audit_b["final_training_objective"]
        if not (exact_state and exact_scores and exact_objective):
            raise RuntimeError(f"05d4 deterministic CUDA fixture failed for {variant_id}.")

        checks[variant_id] = {
            "seed": seed,
            "epochs": epochs,
            "exact_state_dict_match": exact_state,
            "exact_patient_shared_score_match": exact_scores,
            "exact_final_objective_match": exact_objective,
            "final_objective": audit_a["final_training_objective"],
        }

    return {
        "fixture_id": "paper5-05d4-v0-v5-cuda-confirmatory-fixture-v1",
        "profile_id": profile.profile_id,
        "device": "cuda:0",
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "variants_checked": list(checks),
        "checks": checks,
        "status": "PASS",
    }


# -----------------------------------------------------------------------------
# Resumable confirmatory fold-seed shards
# -----------------------------------------------------------------------------

def shard_path(
    metadata: dict[str, Any],
    variant_id: str,
    fold_index: int,
    base_model_seed: int,
) -> Path:
    key = (
        f"{metadata['scenario_id']}|{metadata['replicate_index']}|{metadata['simulation_seed']}|"
        f"{variant_id}|{fold_index}|{base_model_seed}|{metadata['input_sha256']}|{metadata['profile_id']}"
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:28]
    return OUTPUT_SHARD_DIR / f"shard_{digest}.json"


def validate_existing_shard(
    shard: dict[str, Any],
    metadata: dict[str, Any],
    variant_id: str,
    fold_index: int,
    base_model_seed: int,
    protocol_hash: str,
) -> None:
    checks = {
        "script_version": (clean_text(shard.get("script_version")), SCRIPT_VERSION),
        "protocol_sha256": (clean_text(shard.get("protocol_sha256")), protocol_hash),
        "scenario_id": (clean_text(shard.get("scenario_id")), clean_text(metadata["scenario_id"])),
        "replicate_index": (int(shard.get("replicate_index", -1)), int(metadata["replicate_index"])),
        "simulation_seed": (int(shard.get("simulation_seed", -1)), int(metadata["simulation_seed"])),
        "variant_id": (clean_text(shard.get("variant_id")), variant_id),
        "fold_index": (int(shard.get("fold_index", -1)), int(fold_index)),
        "base_model_seed": (int(shard.get("base_model_seed", -1)), int(base_model_seed)),
        "profile_id": (clean_text(shard.get("profile_id")), clean_text(metadata["profile_id"])),
        "input_sha256": (clean_text(shard.get("input_sha256")), clean_text(metadata["input_sha256"])),
    }
    failures = [name for name, (observed, expected) in checks.items() if observed != expected]
    if failures:
        raise RuntimeError("Existing 05d4 shard verification failed: " + ", ".join(failures))


def materialize_shard(
    *,
    neural: Any,
    linear: Any,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    profile: Any,
    variant: VariantSpec,
    fold_index: int,
    base_model_seed: int,
    seed_record: dict[str, int],
    fold_data: FoldRefitData,
    protocol_hash: str,
) -> tuple[dict[str, Any], str]:
    candidate = shard_path(metadata, variant.variant_id, fold_index, base_model_seed)
    if candidate.exists():
        shard = load_json(candidate)
        validate_existing_shard(
            shard, metadata, variant.variant_id, fold_index, base_model_seed, protocol_hash
        )
        return shard, "verified_existing"

    try:
        shard = evaluate_fold_seed(
            neural=neural,
            linear=linear,
            arrays=arrays,
            metadata=metadata,
            profile=profile,
            variant=variant,
            fold_index=fold_index,
            base_model_seed=base_model_seed,
            seed_record=seed_record,
            fold_data=fold_data,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        shard = failed_shard_record(
            metadata=metadata,
            profile=profile,
            variant=variant,
            fold_index=fold_index,
            base_model_seed=base_model_seed,
            seed_record=seed_record,
            error=exc,
        )

    shard["protocol_sha256"] = protocol_hash
    shard["created_at_utc"] = utc_now_iso()
    write_json_atomic(shard, candidate)

    serialized = load_json(candidate)
    validate_existing_shard(
        serialized, metadata, variant.variant_id, fold_index, base_model_seed, protocol_hash
    )
    return serialized, "generated"


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------

def aggregate_shards(shards: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    preprocessing_rows: list[dict[str, Any]] = []
    for shard in shards:
        metric_rows.append(shard["metric_row"])
        fit_rows.extend(shard.get("fit_audits", []))
        preprocessing_rows.extend(shard.get("preprocessing_rows", []))
    return pd.DataFrame(metric_rows), pd.DataFrame(fit_rows), pd.DataFrame(preprocessing_rows)


AGGREGATED_METRICS = [
    "simulation_utility",
    "heldout_reconstruction_score",
    "heldout_normalized_reconstruction_error",
    "random_adjusted_cross_view_mrr",
    "cross_view_mrr",
    "cross_view_top1_accuracy",
    "same_patient_similarity_margin",
    "missing_view_reconstruction_score",
    "missing_view_r2_RNA",
    "missing_view_r2_METHYLATION",
    "missing_view_r2_CNV",
    "modality_balance_score",
    "modality_contribution_entropy",
    "shared_subspace_recovery",
    "private_subspace_recovery",
    "technical_subspace_recovery",
    "heldout_latent_effective_rank",
    "heldout_median_shared_variance",
]


def dataset_variant_summary(metric_table: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "scenario_id",
        "scenario_origin",
        "scenario_family",
        "calibration_role",
        "replicate_index",
        "simulation_seed",
        "n_samples",
        "n_true_biological_shared_factors",
        "n_true_private_factors_per_view",
        "n_true_technical_shared_factors",
        "factor_level_identifiable",
        "subspace_level_identifiable",
        "primary_truth_target",
        "profile_id",
        "architecture_profile",
        "regularization_profile",
        "variant_id",
        "primary_ablation_variant",
        "secondary_exploratory_variant",
        "stability_gate_declared",
        "stability_gate_applied_in_05d4",
        "variational_latent",
    ]
    rows: list[dict[str, Any]] = []
    for keys, part in metric_table.groupby(group_columns, sort=True, dropna=False):
        row = dict(zip(group_columns, keys))
        row["n_fold_seed_evaluations"] = int(part.shape[0])
        row["n_passed_fold_seed_evaluations"] = int(part["shard_status"].eq("PASS").sum())
        row["n_failed_fold_seed_evaluations"] = int(part["shard_status"].eq("FAIL").sum())
        row["all_fold_seed_evaluations_passed"] = bool(part["shard_status"].eq("PASS").all())
        selected_epochs = pd.to_numeric(part["selected_epoch"], errors="coerce").to_numpy(dtype=float)
        selected_epochs = selected_epochs[np.isfinite(selected_epochs)]
        row["selected_epoch_mean"] = float(np.mean(selected_epochs)) if selected_epochs.size else None
        row["selected_epoch_sd"] = float(np.std(selected_epochs, ddof=1)) if selected_epochs.size > 1 else None
        row["n_heldout_latent_collapse"] = int(part["heldout_latent_collapse"].map(parse_bool).sum())
        for metric in AGGREGATED_METRICS:
            values = pd.to_numeric(part[metric], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{metric}_n"] = int(finite.size)
            row[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else None
            row[f"{metric}_sd"] = float(np.std(finite, ddof=1)) if finite.size > 1 else None
        rows.append(row)

    output = pd.DataFrame(rows)
    expected = EXPECTED_CONFIRMATORY_DATASETS * len(NEURAL_VARIANTS)
    if output.shape[0] != expected:
        raise RuntimeError(f"Expected {expected:,} dataset-variant rows; found {output.shape[0]:,}.")
    counts = pd.to_numeric(output["n_fold_seed_evaluations"], errors="raise").astype(int)
    if not counts.eq(EXPECTED_FOLDS * len(CONFIRMATORY_MODEL_BASE_SEEDS)).all():
        raise RuntimeError("Every dataset-variant summary must contain exactly 10 fold-seed evaluations.")
    return output


def scenario_variant_summary(dataset_table: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "scenario_id",
        "scenario_origin",
        "scenario_family",
        "profile_id",
        "architecture_profile",
        "regularization_profile",
        "variant_id",
        "primary_ablation_variant",
        "secondary_exploratory_variant",
        "stability_gate_declared",
        "stability_gate_applied_in_05d4",
        "variational_latent",
    ]
    metric_columns = [f"{metric}_mean" for metric in AGGREGATED_METRICS]
    rows: list[dict[str, Any]] = []
    for keys, part in dataset_table.groupby(group_columns, sort=True, dropna=False):
        row = dict(zip(group_columns, keys))
        row["n_confirmatory_datasets"] = int(part.shape[0])
        row["n_datasets_all_fold_seed_pass"] = int(part["all_fold_seed_evaluations_passed"].map(parse_bool).sum())
        row["n_datasets_with_any_failure"] = int((~part["all_fold_seed_evaluations_passed"].map(parse_bool)).sum())
        for metric_column in metric_columns:
            values = pd.to_numeric(part[metric_column], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[metric_column] = float(np.mean(finite)) if finite.size else None
            row[f"{metric_column}_sd_across_replicates"] = float(np.std(finite, ddof=1)) if finite.size > 1 else None
            row[f"{metric_column}_n_replicates"] = int(finite.size)
        rows.append(row)

    output = pd.DataFrame(rows)
    expected = EXPECTED_SCENARIOS * len(NEURAL_VARIANTS)
    if output.shape[0] != expected:
        raise RuntimeError(f"Expected {expected} scenario-variant rows; found {output.shape[0]}.")
    if not pd.to_numeric(output["n_confirmatory_datasets"], errors="raise").astype(int).eq(10).all():
        raise RuntimeError("Every scenario-variant summary must contain exactly 10 confirmatory datasets.")
    return output


def variant_summary(scenario_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant_id in NEURAL_VARIANTS:
        part = scenario_table[scenario_table["variant_id"].eq(variant_id)].copy()
        if part.shape[0] != EXPECTED_SCENARIOS:
            raise RuntimeError(f"{variant_id}: expected 36 scenario rows.")
        row: dict[str, Any] = {
            "variant_id": variant_id,
            "profile_id": clean_text(part.iloc[0]["profile_id"]),
            "architecture_profile": clean_text(part.iloc[0]["architecture_profile"]),
            "regularization_profile": clean_text(part.iloc[0]["regularization_profile"]),
            "primary_ablation_variant": parse_bool(part.iloc[0]["primary_ablation_variant"]),
            "secondary_exploratory_variant": parse_bool(part.iloc[0]["secondary_exploratory_variant"]),
            "stability_gate_declared": parse_bool(part.iloc[0]["stability_gate_declared"]),
            "stability_gate_applied_in_05d4": False,
            "n_scenarios": int(part.shape[0]),
            "n_scenarios_with_any_failed_dataset": int((pd.to_numeric(part["n_datasets_with_any_failure"], errors="raise") > 0).sum()),
        }
        for metric in AGGREGATED_METRICS:
            column = f"{metric}_mean"
            values = pd.to_numeric(part[column], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"overall_mean_{metric}"] = float(np.mean(finite)) if finite.size == EXPECTED_SCENARIOS else None
            row[f"scenario_mean_{metric}_sd"] = float(np.std(finite, ddof=1)) if finite.size == EXPECTED_SCENARIOS else None
            row[f"scenario_mean_{metric}_se"] = (
                float(np.std(finite, ddof=1) / math.sqrt(EXPECTED_SCENARIOS))
                if finite.size == EXPECTED_SCENARIOS
                else None
            )
            row[f"n_finite_scenario_means_{metric}"] = int(finite.size)
        rows.append(row)

    output = pd.DataFrame(rows)
    if list(output["variant_id"]) != list(NEURAL_VARIANTS):
        raise RuntimeError("Variant summary order differs from frozen V0-V5 order.")
    return output


def ablation_delta_summary(scenario_table: pd.DataFrame) -> pd.DataFrame:
    contrasts = [
        ("C1_SHARED_PRIVATE", "V1_SHARED_PRIVATE_AE", "V0_BALANCED_AE", "shared/private decomposition"),
        ("C2_CONTRASTIVE", "V2_CONTRASTIVE_SPAE", "V1_SHARED_PRIVATE_AE", "paired contrastive alignment"),
        ("C3_DROPOUT_MISSING_VIEW", "V3_DROPOUT_SPAE", "V2_CONTRASTIVE_SPAE", "modality dropout + missing-view reconstruction"),
    ]
    metric_names = [
        "simulation_utility",
        "heldout_reconstruction_score",
        "random_adjusted_cross_view_mrr",
        "missing_view_reconstruction_score",
        "modality_balance_score",
        "shared_subspace_recovery",
        "private_subspace_recovery",
    ]
    rows: list[dict[str, Any]] = []
    for contrast_id, higher, lower, component in contrasts:
        left = scenario_table[scenario_table["variant_id"].eq(higher)].set_index("scenario_id")
        right = scenario_table[scenario_table["variant_id"].eq(lower)].set_index("scenario_id")
        shared_ids = sorted(set(left.index) & set(right.index))
        if len(shared_ids) != EXPECTED_SCENARIOS:
            raise RuntimeError(f"{contrast_id}: scenario pairing is incomplete.")
        row: dict[str, Any] = {
            "contrast_id": contrast_id,
            "added_component": component,
            "higher_variant": higher,
            "lower_variant": lower,
            "n_paired_scenarios": len(shared_ids),
            "interpretation": "descriptive confirmatory ablation contrast; no p-value or method selection is performed here",
        }
        for metric in metric_names:
            column = f"{metric}_mean"
            a = pd.to_numeric(left.loc[shared_ids, column], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(right.loc[shared_ids, column], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(a) & np.isfinite(b)
            delta = a[valid] - b[valid]
            row[f"{metric}_delta_n"] = int(delta.size)
            row[f"{metric}_delta_mean"] = float(np.mean(delta)) if delta.size else None
            row[f"{metric}_delta_sd"] = float(np.std(delta, ddof=1)) if delta.size > 1 else None
            row[f"{metric}_delta_se"] = float(np.std(delta, ddof=1) / math.sqrt(delta.size)) if delta.size > 1 else None
            row[f"{metric}_fraction_positive"] = float(np.mean(delta > 0)) if delta.size else None
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Frozen-result verification and README
# -----------------------------------------------------------------------------

def verify_existing_lock() -> None:
    lock = load_json(OUTPUT_LOCK)
    allowed_status = {
        "FROZEN_NEURAL_CONFIRMATORY_ABLATION_RESULTS_BEFORE_05D5",
        "FROZEN_NEURAL_CONFIRMATORY_ABLATION_RESULTS_WITH_FAILURES",
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
        if table.shape[0] != EXPECTED_FOLD_SEED_EVALUATIONS:
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
        raise RuntimeError("Existing 05d4 frozen-result verification failed:\n" + "\n".join(f"- {x}" for x in failures))

    print("=" * 88)
    print("Existing Paper 5 neural confirmatory ablation benchmark: PASS")
    print("=" * 88)
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("No files were rewritten.")
    print("=" * 88)


def write_readme(profile: Any, n_pass: int, n_fail: int) -> None:
    lines = [
        "Paper 5 neural confirmatory ablation benchmark",
        "=" * 72,
        "",
        f"Script version: {SCRIPT_VERSION}",
        "",
        "Scope",
        "-----",
        "Confirmatory simulation replicate indices 11-20 only.",
        f"Confirmatory datasets: {EXPECTED_CONFIRMATORY_DATASETS}",
        "Variants: V0-V5.",
        f"Five frozen folds x two frozen seeds = {EXPECTED_FOLD_SEED_EVALUATIONS:,} held-out fold-seed evaluations.",
        f"Early-stop fit + selected-epoch refit = {EXPECTED_OPTIMIZATION_RUNS:,} planned neural optimization runs.",
        "",
        "Frozen profile",
        "--------------",
        f"Profile: {profile.profile_id}",
        f"Architecture: {profile.architecture_profile}",
        f"Regularization: {profile.regularization_profile}",
        "The same 05d3-frozen profile is reused by all V0-V5 variants without retuning.",
        "",
        "Confirmatory separation",
        "-----------------------",
        "Development simulation replicates 1-10 are not used as confirmatory evidence.",
        "TARGET-OS matrices, outcomes, clinical labels, and biological program annotations are not accessed.",
        "Final gate-calibration simulations are not accessed.",
        "",
        "GPU execution",
        "-------------",
        "All neural forward/backward/AdamW work runs on frozen cuda:0 in float32.",
        "Dual-PCA Gram/eigendecomposition/projection runs in float64 on cuda:0.",
        "AMP: OFF; TF32: OFF; torch.compile: OFF; quiet CPU fallback: FORBIDDEN.",
        "",
        "Important V4 rule",
        "-----------------",
        "The V4 stability gate is not a training component and is not executed in 05d4.",
        "V4-versus-V3 is therefore not interpreted here as the effect of the stability gate.",
        "Independent confirmatory V4 stability resampling is reserved for 05d5.",
        "",
        "Results",
        "-------",
        f"Passed fold-seed shards: {n_pass}",
        f"Failed fold-seed shards: {n_fail}",
        "No final representation method is selected in this stage.",
        "",
        "Next stage",
        "----------",
        "05d5_evaluate_sable_mo_stability_gpu.py" if n_fail == 0 else "STOP: resolve/report 05d4 failures before 05d5.",
        "",
    ]
    OUTPUT_README.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    print("=" * 88)
    print("Paper 5 - GPU neural confirmatory ablation benchmark")
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

    neural = load_module(NEURAL_IMPL_SCRIPT, "paper5_05d2v2_confirmatory_helpers")
    linear = neural.load_linear_helpers()
    upstream = verify_upstream_state(neural)
    profile = upstream["profile"]
    variant_specs = upstream["variant_specs"]

    print("05d0 frozen CUDA runtime: PASS")
    print("05d1 frozen confirmatory simulation protocol: PASS")
    print("05d2 v2 frozen neural profile screen: PASS")
    print("05d3 frozen selected neural profile: PASS")
    print(f"Frozen profile reused by V0-V5: {profile.profile_id}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Heavy neural computation device: cuda:0")
    print("Quiet CPU fallback: FORBIDDEN")
    print("Automatic mixed precision: OFF")
    print("TF32: OFF")
    print("torch.compile: OFF")
    print("TARGET/outcomes/final-gate data: FORBIDDEN")
    print("Confirmatory V4 stability resampling in 05d4: NO (reserved for 05d5)")
    print()

    execution_protocol, protocol_hash = initialize_or_verify_protocol(upstream)
    del execution_protocol
    print("Prospective 05d4 confirmatory execution protocol: FROZEN")
    print(f"Confirmatory datasets: {EXPECTED_CONFIRMATORY_DATASETS}")
    print(f"Variants: {len(NEURAL_VARIANTS)}")
    print(f"Folds: {EXPECTED_FOLDS}")
    print(f"Seeds per variant-fold: {EXPECTED_SEEDS_PER_VARIANT_FOLD}")
    print(f"Fold-seed evaluations: {EXPECTED_FOLD_SEED_EVALUATIONS:,}")
    print(f"Planned optimization runs including refits: {EXPECTED_OPTIMIZATION_RUNS:,}")
    print()

    print("Running deterministic V0/V4/V5 CUDA implementation fixture...")
    fixture = run_reference_fixture(neural, profile, variant_specs)
    write_json_atomic(fixture, OUTPUT_FIXTURE)
    print("05d4 deterministic CUDA fixture: PASS")
    print("V0 exact repeated state/scores: PASS")
    print("V4 exact repeated state/scores: PASS")
    print("V5 exact repeated state/scores: PASS")
    print()

    write_json_atomic(
        {
            "script_version": SCRIPT_VERSION,
            "created_at_utc": utc_now_iso(),
            "protocol_sha256": protocol_hash,
            "planned_shards": EXPECTED_FOLD_SEED_EVALUATIONS,
            "planned_optimization_runs": EXPECTED_OPTIMIZATION_RUNS,
            "status": "IN_PROGRESS",
        },
        OUTPUT_IN_PROGRESS,
    )

    confirmatory = upstream["confirmatory_roles"].copy()
    confirmatory = confirmatory.sort_values(["scenario_id", "replicate_index"], kind="mergesort").reset_index(drop=True)
    shards: list[dict[str, Any]] = []
    shard_manifest_rows: list[dict[str, Any]] = []
    current = 0
    generated_count = 0
    generated_seconds = 0.0

    for dataset_number, (_, role_row) in enumerate(confirmatory.iterrows(), start=1):
        key = (
            clean_text(role_row["scenario_id"]),
            int(role_row["replicate_index"]),
            int(role_row["simulation_seed"]),
        )
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
        arrays: dict[str, np.ndarray] | None = None

        for fold_index in range(1, EXPECTED_FOLDS + 1):
            test_indices = np.sort(upstream["split_lookup"][key][fold_index].astype(int))
            training_indices = np.setdiff1d(np.arange(metadata["n_samples"], dtype=int), test_indices, assume_unique=False)
            fold_data: FoldRefitData | None = None

            for variant_id in NEURAL_VARIANTS:
                variant = variant_specs[variant_id]
                for base_model_seed in CONFIRMATORY_MODEL_BASE_SEEDS:
                    current += 1
                    seed_key = (*key, variant_id, fold_index, base_model_seed)
                    seed_record = upstream["seed_lookup"][seed_key]
                    candidate = shard_path(metadata, variant_id, fold_index, base_model_seed)
                    started = time.perf_counter()

                    if candidate.exists():
                        shard = load_json(candidate)
                        validate_existing_shard(
                            shard, metadata, variant_id, fold_index, base_model_seed, protocol_hash
                        )
                        generation_status = "verified_existing"
                    else:
                        if arrays is None:
                            arrays = linear.load_simulation_arrays(dataset_path)
                        if fold_data is None:
                            preprocessors = neural.fit_preprocessors(linear, arrays, training_indices, profile)
                            training_views = neural.transform_views(arrays, preprocessors, training_indices, complete=False)
                            training_availability = neural.view_availability(arrays, preprocessors, training_indices)
                            test_views = neural.transform_views(arrays, preprocessors, test_indices, complete=False)
                            test_truth = neural.transform_views(arrays, preprocessors, test_indices, complete=True)
                            test_availability = neural.view_availability(arrays, preprocessors, test_indices)
                            preprocessing_rows = neural.preprocessing_audit_rows(
                                arrays,
                                preprocessors,
                                training_indices,
                                profile,
                                "confirmatory_refit_full_training_fold",
                            )
                            fold_data = FoldRefitData(
                                training_indices=training_indices,
                                test_indices=test_indices,
                                preprocessors=preprocessors,
                                training_views=training_views,
                                training_availability=training_availability,
                                test_views=test_views,
                                test_truth=test_truth,
                                test_availability=test_availability,
                                preprocessing_rows=preprocessing_rows,
                            )

                        shard, generation_status = materialize_shard(
                            neural=neural,
                            linear=linear,
                            arrays=arrays,
                            metadata=metadata,
                            profile=profile,
                            variant=variant,
                            fold_index=fold_index,
                            base_model_seed=base_model_seed,
                            seed_record=seed_record,
                            fold_data=fold_data,
                            protocol_hash=protocol_hash,
                        )

                    elapsed = time.perf_counter() - started
                    if generation_status == "generated":
                        generated_count += 1
                        generated_seconds += elapsed
                    shards.append(shard)
                    shard_status = clean_text(shard["metric_row"].get("shard_status"))
                    shard_manifest_rows.append(
                        {
                            "scenario_id": key[0],
                            "scenario_family": clean_text(manifest_row.get("scenario_family")),
                            "replicate_index": key[1],
                            "simulation_seed": key[2],
                            "profile_id": profile.profile_id,
                            "variant_id": variant_id,
                            "fold_index": fold_index,
                            "base_model_seed": base_model_seed,
                            "fit_seed": seed_record["fit_seed"],
                            "refit_seed": seed_record["refit_seed"],
                            "early_stop_partition_seed": seed_record["early_stop_partition_seed"],
                            "input_path": clean_text(manifest_row["path"]),
                            "input_sha256": observed_hash,
                            "shard_path": relative_path(candidate),
                            "size_bytes": int(candidate.stat().st_size),
                            "sha256": sha256_file(candidate),
                            "generation_status": generation_status,
                            "shard_status": shard_status,
                            "wall_seconds": float(elapsed),
                        }
                    )

                    if current == 1 or current % 25 == 0 or shard_status == "FAIL" or current == EXPECTED_FOLD_SEED_EVALUATIONS:
                        if generated_count > 0:
                            mean_seconds = generated_seconds / generated_count
                            remaining = EXPECTED_FOLD_SEED_EVALUATIONS - current
                            eta_text = f" | ETA~{remaining * mean_seconds / 3600.0:.1f}h"
                        else:
                            eta_text = ""
                        print(
                            f"{current:05d}/{EXPECTED_FOLD_SEED_EVALUATIONS:05d} "
                            f"{key[0]} R{key[1]:03d} {variant_id} F{fold_index} S{base_model_seed}: "
                            f"{generation_status}/{shard_status}{eta_text}"
                        )
                        if shard_status == "FAIL":
                            print(
                                "    ERROR: "
                                + clean_text(shard["metric_row"].get("shard_error_type"))
                                + ": "
                                + clean_text(shard["metric_row"].get("shard_error_message"))[:500]
                            )

            if fold_data is not None:
                del fold_data

        if arrays is not None:
            del arrays
        # Keep the CUDA caching allocator warm across the five folds of a
        # dataset. Release cached blocks only at dataset boundaries; forcing
        # empty_cache() after every tiny fold-seed fit adds synchronization
        # overhead without changing the scientific calculation.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if dataset_number % 10 == 0 or dataset_number == EXPECTED_CONFIRMATORY_DATASETS:
            print(f"Completed confirmatory datasets: {dataset_number}/{EXPECTED_CONFIRMATORY_DATASETS}")

    shard_manifest = pd.DataFrame(shard_manifest_rows)
    if shard_manifest.shape[0] != EXPECTED_FOLD_SEED_EVALUATIONS:
        raise RuntimeError(
            f"05d4 shard manifest expected {EXPECTED_FOLD_SEED_EVALUATIONS:,} rows; found {shard_manifest.shape[0]:,}."
        )
    shard_manifest.to_csv(OUTPUT_SHARD_MANIFEST, index=False, float_format="%.12g", lineterminator="\n")

    metric_table, fit_table, preprocessing_table = aggregate_shards(shards)
    if metric_table.shape[0] != EXPECTED_FOLD_SEED_EVALUATIONS:
        raise RuntimeError("05d4 fold-seed metric table row count mismatch.")

    for column in AGGREGATED_METRICS:
        if column not in metric_table.columns:
            metric_table[column] = np.nan
    if "heldout_latent_collapse" not in metric_table.columns:
        metric_table["heldout_latent_collapse"] = False

    n_pass = int(metric_table["shard_status"].eq("PASS").sum())
    n_fail = int(metric_table["shard_status"].eq("FAIL").sum())
    if n_fail == 0:
        if fit_table.shape[0] != EXPECTED_OPTIMIZATION_RUNS:
            raise RuntimeError(
                f"All-PASS 05d4 expected {EXPECTED_OPTIMIZATION_RUNS:,} fit-audit rows; found {fit_table.shape[0]:,}."
            )
        if preprocessing_table.shape[0] != EXPECTED_PREPROCESSING_ROWS_IF_ALL_PASS:
            raise RuntimeError(
                f"All-PASS 05d4 expected {EXPECTED_PREPROCESSING_ROWS_IF_ALL_PASS:,} preprocessing rows; "
                f"found {preprocessing_table.shape[0]:,}."
            )

    write_csv_gzip(metric_table, OUTPUT_FOLD_SEED_METRICS)
    write_csv_gzip(fit_table, OUTPUT_FIT_AUDIT)
    write_csv_gzip(preprocessing_table, OUTPUT_PREPROCESSING_AUDIT)

    dataset_table = dataset_variant_summary(metric_table)
    scenario_table = scenario_variant_summary(dataset_table)
    variant_table = variant_summary(scenario_table)
    delta_table = ablation_delta_summary(scenario_table)

    write_csv_gzip(dataset_table, OUTPUT_DATASET_VARIANT)
    scenario_table.to_csv(OUTPUT_SCENARIO_VARIANT, index=False, float_format="%.12g", lineterminator="\n")
    variant_table.to_csv(OUTPUT_VARIANT_SUMMARY, index=False, float_format="%.12g", lineterminator="\n")
    delta_table.to_csv(OUTPUT_ABLATION_DELTAS, index=False, float_format="%.12g", lineterminator="\n")
    write_readme(profile, n_pass, n_fail)

    output_files = [
        OUTPUT_PROTOCOL,
        OUTPUT_FIXTURE,
        OUTPUT_SHARD_MANIFEST,
        OUTPUT_FOLD_SEED_METRICS,
        OUTPUT_FIT_AUDIT,
        OUTPUT_PREPROCESSING_AUDIT,
        OUTPUT_DATASET_VARIANT,
        OUTPUT_SCENARIO_VARIANT,
        OUTPUT_VARIANT_SUMMARY,
        OUTPUT_ABLATION_DELTAS,
        OUTPUT_README,
    ]

    result_manifest = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "scientific_state": {
            "outcomes_loaded": False,
            "clinical_labels_loaded": False,
            "biological_annotations_used_for_model_selection": False,
            "TARGET_model_fitting_performed": False,
            "neural_profile_frozen": True,
            "confirmatory_replicates_11_to_20_accessed": True,
            "confirmatory_neural_ablation_performed": True,
            "confirmatory_V4_stability_performed": False,
            "final_gate_calibration_data_used": False,
            "representation_method_selected": False,
        },
        "scope": {
            "profile_id": profile.profile_id,
            "confirmatory_datasets": EXPECTED_CONFIRMATORY_DATASETS,
            "variants": list(NEURAL_VARIANTS),
            "fold_seed_evaluations": int(metric_table.shape[0]),
            "passed_fold_seed_evaluations": n_pass,
            "failed_fold_seed_evaluations": n_fail,
            "fit_audit_rows": int(fit_table.shape[0]),
            "preprocessing_audit_rows": int(preprocessing_table.shape[0]),
            "dataset_variant_rows": int(dataset_table.shape[0]),
            "scenario_variant_rows": int(scenario_table.shape[0]),
            "variant_summary_rows": int(variant_table.shape[0]),
            "descriptive_ablation_contrasts": int(delta_table.shape[0]),
        },
        "execution_protocol": {"path": relative_path(OUTPUT_PROTOCOL), "sha256": protocol_hash},
        "reference_fixture": file_record(OUTPUT_FIXTURE),
        "shard_manifest": {
            "path": relative_path(OUTPUT_SHARD_MANIFEST),
            "sha256": sha256_file(OUTPUT_SHARD_MANIFEST),
            "n_rows": int(shard_manifest.shape[0]),
        },
        "outputs": {relative_path(path): {"sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)} for path in output_files},
        "upstream_hashes": upstream["hashes"],
        "interpretation_guardrails": [
            "05d4 uses confirmatory simulation replicate indices 11-20 only.",
            "The 05d3-frozen ARCH_COMPACT__REG_WEAK profile is reused without retuning.",
            "V4 stability is not applied in 05d4 because the frozen stability gate is a resampling component, not a training loss.",
            "V4-minus-V3 is not interpreted as a stability-gate ablation in this stage.",
            "V5 remains secondary exploratory and is not eligible for final method freeze.",
            "No TARGET, outcome, clinical, biological-program, or final-gate calibration data were used.",
            "No final representation method is selected in 05d4.",
        ],
        "next_required_stage": "05d5_evaluate_sable_mo_stability_gpu.py" if n_fail == 0 else None,
    }
    write_json_atomic(result_manifest, OUTPUT_RESULT_MANIFEST)

    generated_files = {
        relative_path(path): {"sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)}
        for path in [*output_files, OUTPUT_RESULT_MANIFEST]
    }
    lock_status = (
        "FROZEN_NEURAL_CONFIRMATORY_ABLATION_RESULTS_BEFORE_05D5"
        if n_fail == 0
        else "FROZEN_NEURAL_CONFIRMATORY_ABLATION_RESULTS_WITH_FAILURES"
    )
    lock = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": lock_status,
        "scientific_state": result_manifest["scientific_state"],
        "profile_id": profile.profile_id,
        "result_manifest": {"path": relative_path(OUTPUT_RESULT_MANIFEST), "sha256": sha256_file(OUTPUT_RESULT_MANIFEST)},
        "shard_manifest": result_manifest["shard_manifest"],
        "generated_files": generated_files,
        "rewrite_policy": (
            "This zero-argument stage never rewrites frozen 05d4 results. Any scientifically justified change requires "
            "a new versioned confirmatory stage rather than silent replacement."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    if OUTPUT_IN_PROGRESS.exists():
        OUTPUT_IN_PROGRESS.unlink()

    print()
    print("-" * 88)
    print("Neural confirmatory ablation summary")
    print("-" * 88)
    print(f"Fold-seed shards: {metric_table.shape[0]:,}")
    print(f"Passed shards: {n_pass:,}")
    print(f"Failed shards: {n_fail:,}")
    print(f"Fit-audit rows: {fit_table.shape[0]:,}")
    print(f"Preprocessing-audit rows: {preprocessing_table.shape[0]:,}")
    print()

    display = [
        "variant_id",
        "overall_mean_simulation_utility",
        "scenario_mean_simulation_utility_se",
        "overall_mean_heldout_reconstruction_score",
        "overall_mean_random_adjusted_cross_view_mrr",
        "overall_mean_missing_view_reconstruction_score",
        "overall_mean_modality_balance_score",
        "n_scenarios_with_any_failed_dataset",
    ]
    print(variant_table[display].to_string(index=False))
    print()
    print("V4 stability gate applied in 05d4: NO")
    print("Representation method selected: NO")
    print("TARGET accessed: NO")
    print("Final gate-calibration data accessed: NO")
    print()
    print("=" * 88)
    if n_fail == 0:
        print("Paper 5 neural confirmatory ablation benchmark: PASS")
        print("=" * 88)
        print(f"Results manifest: {relative_path(OUTPUT_RESULT_MANIFEST)}")
        print(f"Lock: {relative_path(OUTPUT_LOCK)}")
        print("Next stage: 05d5_evaluate_sable_mo_stability_gpu.py")
    else:
        print("Paper 5 neural confirmatory ablation benchmark: COMPLETED WITH FAILURES")
        print("=" * 88)
        print(f"Results manifest: {relative_path(OUTPUT_RESULT_MANIFEST)}")
        print(f"Lock: {relative_path(OUTPUT_LOCK)}")
        print("STOP: do not proceed to 05d5 until 05d4 failures are reviewed.")
    print("=" * 88)


if __name__ == "__main__":
    main()
