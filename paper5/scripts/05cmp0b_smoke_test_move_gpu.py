from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Deterministic CUDA GEMM configuration must be set before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch


SCRIPT_VERSION = "05cmp0b-smoke-test-move-gpu-v1-no-cli"
CONTRACT_VERSION = "sable-mo-move-gpu-synthetic-smoke-v1"
STATUS = "PASS_SYNTHETIC_MOVE_GPU_SMOKE_BEFORE_05CMP1"

EXPECTED_MOVE_VERSION = "1.5.0"
EXPECTED_SYNTHETIC_SAMPLES = 24
VIEWS = ("RNA", "METHYLATION", "CNV")
VIEW_KEYS = ("rna", "methylation", "cnv")

# Development-only smoke parameters. These are explicitly NOT scientific
# comparator hyperparameters and must not be reused as such in 05cmp1.
SMOKE_SEED = 1103
SMOKE_BATCH_SIZE = 8
SMOKE_EPOCHS = 12
SMOKE_KLD_WARMUP_STEPS = (1, 4, 8, 12)
SMOKE_HIDDEN = (32, 16)
SMOKE_LATENT_DIM = 5
SMOKE_BETA = 0.01
SMOKE_DROPOUT = 0.10
SMOKE_LR = 1e-3

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# -----------------------------------------------------------------------------
# Frozen prospective comparator anchors
# -----------------------------------------------------------------------------

DIR_05CMP0 = PAPER5_ROOT / "method_contract" / "05cmp0"
LOCK_05CMP0 = DIR_05CMP0 / "secondary_comparator_selection_contract_lock.json"
CONTRACT_05CMP0 = DIR_05CMP0 / "secondary_comparator_selection_contract.json"

DIR_05CMP0A = PAPER5_ROOT / "method_contract" / "05cmp0a"
LOCK_05CMP0A = DIR_05CMP0A / "secondary_comparator_continuous_data_amendment_lock.json"
AMENDMENT_05CMP0A = DIR_05CMP0A / "secondary_comparator_continuous_data_amendment.json"
EFFECTIVE_SELECTED_05CMP0A = DIR_05CMP0A / "effective_secondary_comparator_selected_registry.csv"

DIR_05DEV0 = PAPER5_ROOT / "method_contract" / "05dev0"
LOCK_05DEV0 = DIR_05DEV0 / "target_input_synthetic_fixture_lock.json"
CONTRACT_05DEV0 = DIR_05DEV0 / "target_input_synthetic_fixture_contract.json"

FIXTURE_ROOT = PAPER5_ROOT / "development_fixtures" / "target_input_synthetic_v1"
SAMPLE_MANIFEST = FIXTURE_ROOT / "sample_manifest.csv"
AVAILABILITY = FIXTURE_ROOT / "view_availability.csv"
RNA_NPY = FIXTURE_ROOT / "rna.npy"
METH_NPY = FIXTURE_ROOT / "methylation.npy"
CNV_NPY = FIXTURE_ROOT / "cnv.npy"

DIR_05CMP1 = PAPER5_ROOT / "method_contract" / "05cmp1"
SECONDARY_RESULTS = PAPER5_ROOT / "results" / "secondary_comparators"
DIR_05E = PAPER5_ROOT / "method_contract" / "05e"

# -----------------------------------------------------------------------------
# Outputs: implementation smoke only, no scientific latent arrays are persisted
# -----------------------------------------------------------------------------

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05cmp0b"
OUTPUT_ENVIRONMENT = OUTPUT_DIR / "move_gpu_smoke_environment.json"
OUTPUT_RESULT = OUTPUT_DIR / "move_gpu_smoke_result.json"
OUTPUT_README = OUTPUT_DIR / "05cmp0b_README.txt"
OUTPUT_LOCK = OUTPUT_DIR / "move_gpu_smoke_lock.json"


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "na"} else text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(b"|")
    digest.update(str(tuple(values.shape)).encode("utf-8"))
    digest.update(b"|")
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_project_path(value: Any) -> Path:
    path = Path(clean_text(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_json(path: Path) -> dict[str, Any]:
    require_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def file_record(path: Path) -> dict[str, Any]:
    require_file(path)
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def verify_generated_files(lock: dict[str, Any], label: str) -> None:
    failures: list[str] = []
    for relative_name, record in lock.get("generated_files", {}).items():
        candidate = resolve_project_path(relative_name)
        if not candidate.is_file():
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
            f"{label} generated-file verification failed:\n- "
            + "\n- ".join(failures[:100])
        )


def any_files(path: Path) -> bool:
    if path.is_file():
        return True
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def parse_bool_series(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    unknown = sorted(set(values) - set(mapping))
    if unknown:
        raise RuntimeError(f"Unparseable boolean values: {unknown[:10]}")
    return values.map(mapping).astype(bool)


# -----------------------------------------------------------------------------
# Existing PASS verification
# -----------------------------------------------------------------------------

def verify_existing_smoke() -> None:
    require_file(OUTPUT_LOCK)
    require_file(OUTPUT_RESULT)
    require_file(OUTPUT_ENVIRONMENT)

    lock = load_json(OUTPUT_LOCK)
    result = load_json(OUTPUT_RESULT)

    if clean_text(lock.get("status")) != STATUS:
        raise RuntimeError("Unexpected existing 05cmp0b lock status.")
    if clean_text(result.get("status")) != STATUS:
        raise RuntimeError("Unexpected existing 05cmp0b result status.")
    if clean_text(lock.get("script_version")) != SCRIPT_VERSION:
        raise RuntimeError("Existing 05cmp0b script version mismatch.")

    verify_generated_files(lock, "05cmp0b")

    script_record = lock.get("script_source", {})
    if clean_text(script_record.get("sha256")) and sha256_file(SCRIPT_PATH) != clean_text(script_record.get("sha256")):
        raise RuntimeError("Existing 05cmp0b script-source hash mismatch.")

    for relative_name, expected_hash in lock.get("upstream_hashes", {}).items():
        candidate = resolve_project_path(relative_name)
        require_file(candidate)
        if sha256_file(candidate) != clean_text(expected_hash):
            raise RuntimeError(f"05cmp0b upstream hash mismatch: {relative_name}")

    print("=" * 88)
    print("Existing Paper 5 MOVE GPU synthetic smoke test: VERIFIED PASS")
    print("=" * 88)
    print(f"Result: {relative_path(OUTPUT_RESULT)}")
    print(f"Environment: {relative_path(OUTPUT_ENVIRONMENT)}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("No files were rewritten.")
    print("=" * 88)


# -----------------------------------------------------------------------------
# Frozen upstream verification
# -----------------------------------------------------------------------------

def verify_upstream() -> dict[str, Any]:
    required = [
        LOCK_05CMP0,
        CONTRACT_05CMP0,
        LOCK_05CMP0A,
        AMENDMENT_05CMP0A,
        EFFECTIVE_SELECTED_05CMP0A,
        LOCK_05DEV0,
        CONTRACT_05DEV0,
        SAMPLE_MANIFEST,
        AVAILABILITY,
        RNA_NPY,
        METH_NPY,
        CNV_NPY,
    ]
    for path in required:
        require_file(path)

    if any_files(DIR_05CMP1):
        raise RuntimeError(
            "05cmp1 already contains files. This synthetic implementation smoke belongs before 05cmp1."
        )
    if any_files(SECONDARY_RESULTS):
        raise RuntimeError(
            "Secondary-comparator scientific results already exist; refusing a pre-05cmp1 smoke run."
        )

    lock0 = load_json(LOCK_05CMP0)
    contract0 = load_json(CONTRACT_05CMP0)
    expected0 = "FROZEN_SECONDARY_COMPARATOR_SELECTION_BEFORE_05E_AND_TARGET_RESULTS"
    if clean_text(lock0.get("status")) != expected0 or clean_text(contract0.get("status")) != expected0:
        raise RuntimeError("05cmp0 is not in the expected frozen state.")
    if sha256_file(CONTRACT_05CMP0) != clean_text(lock0.get("contract_sha256")):
        raise RuntimeError("05cmp0 contract hash mismatch.")
    verify_generated_files(lock0, "05cmp0")

    lock0a = load_json(LOCK_05CMP0A)
    amendment0a = load_json(AMENDMENT_05CMP0A)
    expected0a = "FROZEN_COMPARATOR_METHOD_CORRECTION_BEFORE_SCIENTIFIC_RUNS"
    if clean_text(lock0a.get("status")) != expected0a or clean_text(amendment0a.get("status")) != expected0a:
        raise RuntimeError("05cmp0a is not in the expected frozen state.")
    if sha256_file(AMENDMENT_05CMP0A) != clean_text(lock0a.get("amendment_sha256")):
        raise RuntimeError("05cmp0a amendment hash mismatch.")
    verify_generated_files(lock0a, "05cmp0a")

    selected = pd.read_csv(EFFECTIVE_SELECTED_05CMP0A, dtype=str).fillna("")
    deep = selected[selected["comparator_role"].eq("required_external_deep")]
    if deep.shape[0] != 1 or clean_text(deep.iloc[0]["method_name"]) != "MOVE":
        raise RuntimeError("Effective comparator registry does not contain MOVE as the unique required deep comparator.")
    if clean_text(deep.iloc[0]["synthetic_smoke_test_allowed_before_05e"]).lower() not in {"true", "1", "yes"}:
        raise RuntimeError("05cmp0/05cmp0a no longer permit the MOVE synthetic smoke test.")
    if clean_text(deep.iloc[0]["scientific_run_allowed_before_05e"]).lower() in {"true", "1", "yes"}:
        raise RuntimeError("Unexpected permission for a pre-05e scientific MOVE run.")

    lockdev = load_json(LOCK_05DEV0)
    contractdev = load_json(CONTRACT_05DEV0)
    expecteddev = "FROZEN_SYNTHETIC_TARGET_INPUT_FIXTURES_BEFORE_TARGET_CODE"
    if clean_text(lockdev.get("status")) != expecteddev or clean_text(contractdev.get("status")) != expecteddev:
        raise RuntimeError("05dev0 is not in the expected frozen state.")
    if sha256_file(CONTRACT_05DEV0) != clean_text(lockdev.get("contract_sha256")):
        raise RuntimeError("05dev0 contract hash mismatch.")
    verify_generated_files(lockdev, "05dev0")

    science = contractdev.get("scientific_state", {})
    forbidden_true = [
        "real_TARGET_values_accessed",
        "TARGET_outcomes_accessed",
        "biological_annotations_accessed",
        "final_representation_method_selected_by_fixture",
        "latent_output_schema_frozen_by_fixture",
        "scientific_TARGET_splits_defined_by_fixture",
        "scientific_TARGET_seeds_defined_by_fixture",
    ]
    active = [key for key in forbidden_true if bool(science.get(key, False))]
    if active:
        raise RuntimeError("05dev0 scientific boundary violated: " + ", ".join(active))

    return {
        "selected_registry": selected,
        "05e_state": "FILES_PRESENT" if any_files(DIR_05E) else "NO_FILES_DETECTED",
        "upstream_hashes": {
            relative_path(path): sha256_file(path)
            for path in required
            if path.suffix.lower() in {".json", ".csv", ".npy"}
        },
    }


# -----------------------------------------------------------------------------
# Environment and official MOVE API verification
# -----------------------------------------------------------------------------

def import_move_api() -> dict[str, Any]:
    try:
        move_version = importlib.metadata.version("move-dl")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "move-dl is not installed in the active Python environment. "
            "Activate the isolated MOVE environment and install move-dl==1.5.0."
        ) from exc

    if move_version != EXPECTED_MOVE_VERSION:
        raise RuntimeError(
            f"MOVE version mismatch: active move-dl={move_version}, expected={EXPECTED_MOVE_VERSION}. "
            "Do not silently change the comparator version before 05cmp1."
        )

    from move.data.dataloaders import make_dataloader
    from move.models.vae import VAE
    from move.training.training_loop import training_loop

    expected_vae_args = {
        "continuous_shapes",
        "continuous_weights",
        "num_hidden",
        "num_latent",
        "beta",
        "dropout",
        "cuda",
    }
    vae_parameters = set(inspect.signature(VAE.__init__).parameters)
    missing_vae = expected_vae_args - vae_parameters
    if missing_vae:
        raise RuntimeError("MOVE VAE API mismatch; missing parameters: " + ", ".join(sorted(missing_vae)))

    expected_loop_args = {"model", "train_dataloader", "lr", "num_epochs", "kld_warmup_steps"}
    loop_parameters = set(inspect.signature(training_loop).parameters)
    missing_loop = expected_loop_args - loop_parameters
    if missing_loop:
        raise RuntimeError("MOVE training_loop API mismatch; missing parameters: " + ", ".join(sorted(missing_loop)))

    source_paths = {
        "vae": Path(inspect.getfile(VAE)).resolve(),
        "training_loop": Path(inspect.getfile(training_loop)).resolve(),
        "dataloaders": Path(inspect.getfile(make_dataloader)).resolve(),
    }
    for path in source_paths.values():
        require_file(path)

    return {
        "VAE": VAE,
        "training_loop": training_loop,
        "make_dataloader": make_dataloader,
        "move_version": move_version,
        "source_paths": source_paths,
    }


def configure_cuda() -> dict[str, Any]:
    in_virtual_environment = bool(
        sys.prefix != getattr(sys, "base_prefix", sys.prefix)
        or os.environ.get("CONDA_PREFIX")
        or os.environ.get("VIRTUAL_ENV")
    )
    if not in_virtual_environment:
        raise RuntimeError(
            "The MOVE smoke test must run in an isolated virtual/Conda environment, not the global Python installation."
        )

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(
            "CUDA is unavailable in the active MOVE environment. Quiet CPU fallback is forbidden for this GPU smoke test."
        )

    torch.cuda.set_device(0)
    torch.manual_seed(SMOKE_SEED)
    torch.cuda.manual_seed_all(SMOKE_SEED)

    # Preserve reproducibility where PyTorch supports it without redefining MOVE.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")

    props = torch.cuda.get_device_properties(0)
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "sys_base_prefix": getattr(sys, "base_prefix", ""),
        "virtual_environment_detected": in_virtual_environment,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "device": "cuda:0",
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": int(props.total_memory),
        "gpu_compute_capability": [int(props.major), int(props.minor)],
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "tf32_matmul": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
        "tf32_cudnn": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
    }


# -----------------------------------------------------------------------------
# 05dev0 adapter validation
# -----------------------------------------------------------------------------

def load_and_validate_fixture() -> dict[str, Any]:
    sample_manifest = pd.read_csv(SAMPLE_MANIFEST, dtype=str).fillna("")
    availability = pd.read_csv(AVAILABILITY, dtype=str).fillna("")

    if sample_manifest.shape[0] != EXPECTED_SYNTHETIC_SAMPLES:
        raise RuntimeError("05dev0 sample-manifest count changed.")
    if availability.shape[0] != EXPECTED_SYNTHETIC_SAMPLES:
        raise RuntimeError("05dev0 availability-row count changed.")

    expected_ids = sample_manifest["synthetic_sample_id"].astype(str).tolist()
    observed_ids = availability["synthetic_sample_id"].astype(str).tolist()
    if observed_ids != expected_ids:
        raise RuntimeError("05dev0 availability/sample-manifest order mismatch.")
    if len(set(expected_ids)) != EXPECTED_SYNTHETIC_SAMPLES:
        raise RuntimeError("05dev0 contains duplicate synthetic sample IDs.")

    availability_bool = pd.DataFrame(
        {
            "rna": parse_bool_series(availability["rna_available"]),
            "methylation": parse_bool_series(availability["methylation_available"]),
            "cnv": parse_bool_series(availability["cnv_available"]),
        }
    )
    if (availability_bool.sum(axis=1) < 1).any():
        raise RuntimeError("05dev0 valid fixture contains an all-views-missing sample.")

    matrices = {
        "rna": np.load(RNA_NPY, allow_pickle=False),
        "methylation": np.load(METH_NPY, allow_pickle=False),
        "cnv": np.load(CNV_NPY, allow_pickle=False),
    }
    expected_features = {"rna": 40, "methylation": 50, "cnv": 30}

    audit: dict[str, Any] = {}
    move_inputs: list[np.ndarray] = []
    for key in VIEW_KEYS:
        matrix = np.asarray(matrices[key], dtype=np.float32)
        expected_shape = (EXPECTED_SYNTHETIC_SAMPLES, expected_features[key])
        if matrix.shape != expected_shape:
            raise RuntimeError(f"05dev0 {key} matrix shape changed: {matrix.shape} != {expected_shape}")

        available = availability_bool[key].to_numpy(dtype=bool)
        all_nan = np.isnan(matrix).all(axis=1)
        if np.any(all_nan[available]):
            raise RuntimeError(f"05dev0 {key}: available row is all NaN.")
        if np.any(~all_nan[~available]):
            raise RuntimeError(f"05dev0 {key}: unavailable row is not all NaN.")

        feature_level_nan_available = int(np.isnan(matrix[available, :]).sum())
        whole_view_missing_rows = int((~available).sum())

        # MOVE's published continuous-data missingness convention represents missing
        # encoded values by zero. 05dev0 is already standardized synthetic input, so
        # this smoke adapter ONLY converts NaNs to zero; it does not define scientific
        # 05cmp1 preprocessing, feature weighting, or architecture selection.
        encoded = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        if not np.isfinite(encoded).all():
            raise RuntimeError(f"MOVE adapter produced non-finite {key} input.")
        if np.any(encoded[~available, :] != 0.0):
            raise RuntimeError(f"MOVE adapter did not zero-code unavailable {key} rows.")

        move_inputs.append(encoded)
        audit[key] = {
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "whole_view_missing_rows": whole_view_missing_rows,
            "feature_level_nan_count_in_available_rows": feature_level_nan_available,
            "adapter": "NaN_to_zero_only_on_already_standardized_05dev0_fixture",
            "encoded_finite": True,
            "encoded_sha256": sha256_array(encoded),
        }

    return {
        "sample_ids": expected_ids,
        "availability": availability_bool,
        "move_inputs": move_inputs,
        "continuous_shapes": [int(matrix.shape[1]) for matrix in move_inputs],
        "audit": audit,
    }


# -----------------------------------------------------------------------------
# MOVE low-level GPU smoke run
# -----------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(make_dataloader: Any, continuous: list[np.ndarray], *, shuffle: bool, seed: int) -> Any:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return make_dataloader(
        con_list=continuous,
        batch_size=SMOKE_BATCH_SIZE,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )


def run_move_once(api: dict[str, Any], fixture: dict[str, Any], seed: int) -> dict[str, Any]:
    VAE = api["VAE"]
    training_loop = api["training_loop"]
    make_dataloader = api["make_dataloader"]

    seed_everything(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    train_loader = make_loader(
        make_dataloader,
        fixture["move_inputs"],
        shuffle=True,
        seed=seed,
    )
    full_loader = make_loader(
        make_dataloader,
        fixture["move_inputs"],
        shuffle=False,
        seed=seed,
    )

    model = VAE(
        categorical_shapes=None,
        continuous_shapes=list(fixture["continuous_shapes"]),
        categorical_weights=None,
        continuous_weights=[1, 1, 1],
        num_hidden=list(SMOKE_HIDDEN),
        num_latent=SMOKE_LATENT_DIM,
        beta=SMOKE_BETA,
        dropout=SMOKE_DROPOUT,
        cuda=True,
    )

    # The low-level MOVE VAE constructor records self.device but does not itself
    # move parameters. Explicit .to(model.device) is runtime plumbing, not a
    # scientific modification of the published architecture/objective.
    model = model.to(model.device)
    parameter_devices = sorted({parameter.device.type for parameter in model.parameters()})
    if parameter_devices != ["cuda"]:
        raise RuntimeError(f"MOVE model parameters are not exclusively on CUDA: {parameter_devices}")

    started = time.perf_counter()
    losses, bce_losses, sse_losses, kld_losses, final_kld_weight = training_loop(
        model=model,
        train_dataloader=train_loader,
        valid_dataloader=None,
        lr=SMOKE_LR,
        num_epochs=SMOKE_EPOCHS,
        batch_dilation_steps=[],
        kld_warmup_steps=list(SMOKE_KLD_WARMUP_STEPS),
        early_stopping=False,
        patience=0,
    )
    torch.cuda.synchronize(0)
    wall_seconds = time.perf_counter() - started

    histories = {
        "loss": np.asarray(losses, dtype=np.float64),
        "bce": np.asarray(bce_losses, dtype=np.float64),
        "sse": np.asarray(sse_losses, dtype=np.float64),
        "kld": np.asarray(kld_losses, dtype=np.float64),
    }
    for name, values in histories.items():
        if values.shape != (SMOKE_EPOCHS,):
            raise RuntimeError(f"MOVE {name} history length changed: {values.shape}")
        if not np.isfinite(values).all():
            raise RuntimeError(f"MOVE {name} history contains non-finite values.")

    latent = np.asarray(model.project(full_loader), dtype=np.float32)
    expected_latent_shape = (EXPECTED_SYNTHETIC_SAMPLES, SMOKE_LATENT_DIM)
    if latent.shape != expected_latent_shape:
        raise RuntimeError(f"MOVE latent shape mismatch: {latent.shape} != {expected_latent_shape}")
    if not np.isfinite(latent).all():
        raise RuntimeError("MOVE latent output contains non-finite values.")

    # Deterministic mean-latent reconstruction check using public encode/decode
    # methods; this avoids persisting any scientific-looking reconstruction data.
    concatenated = np.concatenate(fixture["move_inputs"], axis=1).astype(np.float32, copy=False)
    tensor = torch.from_numpy(concatenated).to(device=model.device, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(tensor)
        _, continuous_reconstruction = model.decode(mu)
    if continuous_reconstruction is None:
        raise RuntimeError("MOVE continuous reconstruction was unexpectedly None.")
    expected_recon_shape = concatenated.shape
    if tuple(continuous_reconstruction.shape) != expected_recon_shape:
        raise RuntimeError(
            f"MOVE reconstruction shape mismatch: {tuple(continuous_reconstruction.shape)} != {expected_recon_shape}"
        )
    if not bool(torch.isfinite(continuous_reconstruction).all()):
        raise RuntimeError("MOVE reconstruction contains non-finite values.")
    if not bool(torch.isfinite(logvar).all()):
        raise RuntimeError("MOVE latent log-variance contains non-finite values.")

    max_memory = int(torch.cuda.max_memory_allocated(0))
    if max_memory <= 0:
        raise RuntimeError("MOVE smoke test did not allocate CUDA memory.")

    result = {
        "seed": int(seed),
        "device": str(model.device),
        "parameter_devices": parameter_devices,
        "epochs_requested": SMOKE_EPOCHS,
        "epochs_recorded": int(histories["loss"].size),
        "final_kld_weight": float(final_kld_weight),
        "final_loss": float(histories["loss"][-1]),
        "final_sse": float(histories["sse"][-1]),
        "final_kld": float(histories["kld"][-1]),
        "latent_shape": list(latent.shape),
        "latent_all_finite": True,
        "latent_sha256": sha256_array(latent),
        "reconstruction_shape": list(continuous_reconstruction.shape),
        "reconstruction_all_finite": True,
        "wall_seconds": float(wall_seconds),
        "max_cuda_memory_allocated_bytes": max_memory,
        "latent": latent,
    }

    del continuous_reconstruction, logvar, mu, tensor, model, train_loader, full_loader
    gc.collect()
    torch.cuda.empty_cache()
    return result


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    print("=" * 88)
    print("Paper 5 - MOVE GPU synthetic implementation smoke test")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Synthetic fixture root: {FIXTURE_ROOT}")
    print("Real TARGET values accessed: NO")
    print("TARGET outcomes/biology accessed: NO")
    print("Scientific comparator performance generated: NO")
    print("05e primary-method influence: FORBIDDEN")
    print("Quiet CPU fallback: FORBIDDEN")
    print()

    if OUTPUT_LOCK.exists():
        verify_existing_smoke()
        return

    upstream = verify_upstream()
    print("05cmp0 frozen comparator selection: PASS")
    print("05cmp0a continuous-data correction: PASS")
    print("05dev0 frozen synthetic fixture: PASS")
    print(f"05e state: {upstream['05e_state']}")
    print("05cmp1 files present: NO")
    print("Secondary comparator scientific results present: NO")
    print()

    api = import_move_api()
    cuda = configure_cuda()

    environment = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "move_distribution": "move-dl",
        "move_version": api["move_version"],
        "expected_move_version": EXPECTED_MOVE_VERSION,
        "official_stable_version_match": api["move_version"] == EXPECTED_MOVE_VERSION,
        "move_source_files": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
            for name, path in api["source_paths"].items()
        },
        **cuda,
    }
    write_json_atomic(environment, OUTPUT_ENVIRONMENT)

    print(f"MOVE: move-dl {api['move_version']}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("MOVE VAE runtime device: cuda:0")
    print("Model parameters on CUDA: REQUIRED")
    print()

    fixture = load_and_validate_fixture()
    print("05dev0 -> MOVE adapter validation: PASS")
    print(f"Synthetic samples: {len(fixture['sample_ids'])}")
    print(f"Continuous shapes: {fixture['continuous_shapes']}")
    print("Missing-value adapter: NaN -> 0 on already-standardized 05dev0 only")
    print("Scientific preprocessing/weights/hyperparameters frozen here: NO")
    print()

    print("Running MOVE GPU smoke fit #1...")
    first = run_move_once(api, fixture, SMOKE_SEED)
    print("MOVE GPU smoke fit #1: PASS")
    print("Running same-seed repeat for implementation-repeatability diagnostic...")
    second = run_move_once(api, fixture, SMOKE_SEED)
    print("MOVE GPU smoke fit #2: PASS")

    latent_a = first.pop("latent")
    latent_b = second.pop("latent")
    max_abs_difference = float(np.max(np.abs(latent_a - latent_b)))
    exact_match = bool(np.array_equal(latent_a, latent_b))
    close_1e6 = bool(np.allclose(latent_a, latent_b, rtol=1e-6, atol=1e-6))

    # Repeatability is diagnostic at this stage, not a scientific/fallback gate.
    repeatability = {
        "same_seed": SMOKE_SEED,
        "exact_latent_match": exact_match,
        "allclose_rtol_1e-6_atol_1e-6": close_1e6,
        "maximum_absolute_latent_difference": max_abs_difference,
        "run1_latent_sha256": first["latent_sha256"],
        "run2_latent_sha256": second["latent_sha256"],
        "gate_consequence": "NONE_IN_05CMP0B; seed/stability policy must be frozen prospectively in 05cmp1",
    }

    result = {
        "script_version": SCRIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": STATUS,
        "comparator": {
            "method": "MOVE",
            "distribution": "move-dl",
            "version": api["move_version"],
            "role": "required_external_deep_secondary_comparator",
            "fallback_activated": False,
        },
        "scientific_boundaries": {
            "synthetic_05dev0_only": True,
            "real_TARGET_values_accessed": False,
            "TARGET_outcomes_accessed": False,
            "biological_annotations_accessed": False,
            "scientific_comparator_result_generated": False,
            "may_influence_05e": False,
            "may_influence_final_gate": False,
            "scientific_hyperparameters_frozen_here": False,
            "scientific_preprocessing_frozen_here": False,
            "scientific_feature_weights_frozen_here": False,
            "latent_values_persisted": False,
        },
        "development_smoke_configuration": {
            "seed": SMOKE_SEED,
            "batch_size": SMOKE_BATCH_SIZE,
            "epochs": SMOKE_EPOCHS,
            "kld_warmup_steps": list(SMOKE_KLD_WARMUP_STEPS),
            "num_hidden": list(SMOKE_HIDDEN),
            "num_latent": SMOKE_LATENT_DIM,
            "beta": SMOKE_BETA,
            "dropout": SMOKE_DROPOUT,
            "learning_rate": SMOKE_LR,
            "continuous_weights": [1, 1, 1],
            "purpose": "implementation smoke only; equal continuous weights only exercise the documented MOVE multi-dataset loss path and may not be treated as 05cmp1 scientific weights/hyperparameters",
        },
        "fixture_adapter_audit": fixture["audit"],
        "run_1": first,
        "run_2": second,
        "same_seed_repeatability_diagnostic": repeatability,
        "pass_checks": {
            "move_version_exactly_1_5_0": api["move_version"] == EXPECTED_MOVE_VERSION,
            "CUDA_available": True,
            "model_parameters_on_CUDA": True,
            "training_history_finite": True,
            "latent_rows_equal_24": first["latent_shape"][0] == EXPECTED_SYNTHETIC_SAMPLES,
            "latent_finite": bool(first["latent_all_finite"] and second["latent_all_finite"]),
            "decoder_output_finite": bool(first["reconstruction_all_finite"] and second["reconstruction_all_finite"]),
            "whole_view_missingness_adapter_exercised": any(
                fixture["audit"][key]["whole_view_missing_rows"] > 0 for key in VIEW_KEYS
            ),
            "feature_level_missingness_adapter_exercised": (
                fixture["audit"]["methylation"]["feature_level_nan_count_in_available_rows"] > 0
            ),
            "CUDA_memory_allocated": (
                first["max_cuda_memory_allocated_bytes"] > 0
                and second["max_cuda_memory_allocated_bytes"] > 0
            ),
        },
        "upstream_hashes": upstream["upstream_hashes"],
        "environment": {
            "path": relative_path(OUTPUT_ENVIRONMENT),
            "sha256": sha256_file(OUTPUT_ENVIRONMENT),
        },
        "next_stage": (
            "Complete the iCluster2b synthetic smoke test, then freeze exact MOVE/iCluster2b "
            "versions, adapters, seeds, preprocessing and scientific benchmark rules in 05cmp1 "
            "before any scientific secondary-comparator run."
        ),
    }

    if not all(bool(value) for value in result["pass_checks"].values()):
        failed = [name for name, value in result["pass_checks"].items() if not bool(value)]
        raise RuntimeError("MOVE GPU smoke hard checks failed: " + ", ".join(failed))

    write_json_atomic(result, OUTPUT_RESULT)

    readme = f"""Paper 5 - MOVE GPU synthetic implementation smoke test
==============================================================================
Script version: {SCRIPT_VERSION}
Status: {STATUS}

PURPOSE
-------
Verify that the frozen required external deep comparator, MOVE, can be imported,
trained and projected on CUDA using ONLY the frozen 05dev0 synthetic fixture.

WHAT THIS PROVES
----------------
- move-dl {EXPECTED_MOVE_VERSION} imports under the active isolated environment;
- CUDA is visible and MOVE model parameters/train tensors execute on cuda:0;
- the 24-row synthetic three-view input is accepted after the development-only
  NaN-to-zero MOVE missing-value adapter;
- whole-view missingness and feature-level methylation missingness are exercised;
- training completes with finite diagnostics;
- the latent projection has one finite row per synthetic sample;
- the decoder produces finite continuous output;
- same-seed repeatability is recorded for 05cmp1 design, but is NOT a 05cmp0b
  scientific/fallback decision criterion.

WHAT THIS DOES NOT DO
---------------------
- no real TARGET value is read;
- no outcome, biology or clinical label is read;
- no scientific comparator performance estimate is generated;
- no MOVE hyperparameter, feature-weight, preprocessing or stability policy is
  frozen for scientific analysis;
- no latent array is persisted;
- MOVE cannot influence 05e or the final-gate decision.

DEVELOPMENT-ONLY SMOKE CONFIGURATION
------------------------------------
seed: {SMOKE_SEED}
batch size: {SMOKE_BATCH_SIZE}
epochs: {SMOKE_EPOCHS}
hidden: {list(SMOKE_HIDDEN)}
latent dim: {SMOKE_LATENT_DIM}
beta: {SMOKE_BETA}
dropout: {SMOKE_DROPOUT}
learning rate: {SMOKE_LR}
continuous weights: [1, 1, 1]

NEXT
----
Complete the iCluster2b synthetic smoke test, then create 05cmp1 to freeze exact
comparator implementations and scientific adapters BEFORE scientific comparator
runs.
"""
    write_text_atomic(readme, OUTPUT_README)

    generated = [OUTPUT_ENVIRONMENT, OUTPUT_RESULT, OUTPUT_README]
    lock = {
        "script_version": SCRIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": STATUS,
        "result": file_record(OUTPUT_RESULT),
        "environment": file_record(OUTPUT_ENVIRONMENT),
        "generated_files": {
            relative_path(path): file_record(path)
            for path in generated
        },
        "upstream_hashes": upstream["upstream_hashes"],
        "script_source": {
            "path": relative_path(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "rewrite_policy": (
            "A successful 05cmp0b smoke record is immutable. If MOVE installation/runtime changes, "
            "create a new versioned smoke/amendment rather than silently rewriting this PASS."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    print()
    print("-" * 88)
    print("MOVE GPU synthetic smoke summary")
    print("-" * 88)
    print(f"MOVE version: {api['move_version']}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Synthetic rows projected: {first['latent_shape'][0]}")
    print(f"Latent dimensions (development-only): {first['latent_shape'][1]}")
    print(f"Run 1 CUDA peak allocation: {first['max_cuda_memory_allocated_bytes'] / (1024**2):.1f} MiB")
    print(f"Run 2 CUDA peak allocation: {second['max_cuda_memory_allocated_bytes'] / (1024**2):.1f} MiB")
    print(f"Same-seed exact latent repeat: {'YES' if exact_match else 'NO'}")
    print(f"Same-seed max |delta|: {max_abs_difference:.8g}")
    print("Scientific performance interpretation: FORBIDDEN")
    print("MOVE fallback activated: NO")
    print()
    print("=" * 88)
    print("Paper 5 MOVE GPU synthetic implementation smoke: PASS")
    print("=" * 88)
    print(f"Result: {relative_path(OUTPUT_RESULT)}")
    print(f"Environment: {relative_path(OUTPUT_ENVIRONMENT)}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("Next: iCluster2b synthetic smoke -> 05cmp1 freeze.")
    print("=" * 88)


if __name__ == "__main__":
    main()
