
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_VERSION = "05d0-freeze-pytorch-neural-environment-v1-no-cli"
CONTRACT_VERSION = "paper5-neural-runtime-environment-v1"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

CONTRACT_05A = PAPER5_ROOT / "method_contract" / "05a" / "ai_method_contract.json"
LOCK_05A = PAPER5_ROOT / "method_contract" / "05a" / "ai_method_contract_lock.json"
CONTRACT_05A1 = PAPER5_ROOT / "method_contract" / "05a1" / "simulation_and_technical_confound_addendum.json"
LOCK_05A1 = PAPER5_ROOT / "method_contract" / "05a1" / "simulation_and_technical_confound_addendum_lock.json"
SIMULATION_LOCK = PAPER5_ROOT / "method_contract" / "05b_v2" / "simulation_generation_v2_lock.json"
SIMULATION_MANIFEST = PAPER5_ROOT / "method_contract" / "05b_v2" / "architecture_screening_data_manifest.csv"
LINEAR_RESULTS_MANIFEST = PAPER5_ROOT / "method_contract" / "05c1" / "linear_baseline_results_manifest.json"
LINEAR_LOCK = PAPER5_ROOT / "method_contract" / "05c1" / "linear_simulation_baseline_lock.json"
MOFA_PROTOCOL = PAPER5_ROOT / "method_contract" / "05c2a" / "mofa_out_of_sample_projection_protocol.json"
MOFA_PROTOCOL_LOCK = PAPER5_ROOT / "method_contract" / "05c2a" / "mofa_out_of_sample_projection_lock.json"
MOFA_RESULTS_MANIFEST = PAPER5_ROOT / "method_contract" / "05c2b" / "mofa_baseline_results_manifest.json"
MOFA_RESULTS_LOCK = PAPER5_ROOT / "method_contract" / "05c2b" / "mofa_simulation_baseline_lock.json"

FINAL_GATE_DATA_ROOT = PAPER5_ROOT / "simulations" / "sable_mo_v2" / "final_gate_calibration"
FUTURE_RESULT_PATHS = (
    PAPER5_ROOT / "method_contract" / "05d",
    PAPER5_ROOT / "method_contract" / "05e",
    PAPER5_ROOT / "results" / "simulation_benchmark" / "05d_sable_mo",
    PAPER5_ROOT / "results" / "representation_benchmark",
)

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05d0"
OUTPUT_CONTRACT = OUTPUT_DIR / "neural_environment_contract.json"
OUTPUT_ENVIRONMENT = OUTPUT_DIR / "neural_environment_snapshot.json"
OUTPUT_FIXTURE = OUTPUT_DIR / "neural_determinism_reference_fixture.json"
OUTPUT_README = OUTPUT_DIR / "neural_environment_README.txt"
OUTPUT_LOCK = OUTPUT_DIR / "neural_environment_lock.json"

CUBLAS_WORKSPACE_CONFIG = ":4096:8"
REFERENCE_SEED = 20260816
REFERENCE_STEPS = 12

RUNTIME_POLICY = {
    "device_policy": "CUDA required for confirmatory neural benchmark",
    "dtype": "float32",
    "automatic_mixed_precision": False,
    "torch_compile": False,
    "deterministic_algorithms": True,
    "deterministic_warn_only": False,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "allow_tf32_matmul": False,
    "allow_tf32_cudnn": False,
    "float32_matmul_precision": "highest",
    "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
    "dataloader_workers_primary": 0,
    "gradient_scaler": False,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na"}:
        return ""
    return text


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
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    require_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(value: dict[str, Any], path: Path) -> None:
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
                value,
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


def package_version(package_name: str) -> dict[str, Any]:
    try:
        return {
            "installed": True,
            "version": importlib.metadata.version(package_name),
        }
    except importlib.metadata.PackageNotFoundError:
        return {
            "installed": False,
            "version": None,
        }


def is_nonempty_path(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.stat().st_size > 0
    return any(item.is_file() for item in path.rglob("*"))


def validate_future_state() -> None:
    found = [
        relative_path(path)
        for path in FUTURE_RESULT_PATHS
        if is_nonempty_path(path)
    ]
    if found:
        raise RuntimeError(
            "05d0 must be frozen before 05d/05e/representation results exist. Found:\n- "
            + "\n- ".join(found)
        )

    if FINAL_GATE_DATA_ROOT.exists() and any(
        item.is_file() for item in FINAL_GATE_DATA_ROOT.rglob("*")
    ):
        raise RuntimeError(
            "Final gate-calibration simulations were materialized before architecture freeze."
        )


def verify_upstream() -> dict[str, Any]:
    for path in [
        CONTRACT_05A,
        LOCK_05A,
        CONTRACT_05A1,
        LOCK_05A1,
        SIMULATION_LOCK,
        SIMULATION_MANIFEST,
        LINEAR_RESULTS_MANIFEST,
        LINEAR_LOCK,
        MOFA_PROTOCOL,
        MOFA_PROTOCOL_LOCK,
        MOFA_RESULTS_MANIFEST,
        MOFA_RESULTS_LOCK,
    ]:
        require_file(path)

    validate_future_state()

    lock_05a = load_json(LOCK_05A)
    expected_05a = clean_text(lock_05a.get("contract_sha256"))
    if not expected_05a or sha256_file(CONTRACT_05A) != expected_05a:
        raise RuntimeError("05a AI method contract verification failed.")

    lock_05a1 = load_json(LOCK_05A1)
    expected_05a1 = clean_text(lock_05a1.get("addendum_sha256"))
    if not expected_05a1 or sha256_file(CONTRACT_05A1) != expected_05a1:
        raise RuntimeError("05a1 addendum verification failed.")

    simulation_lock = load_json(SIMULATION_LOCK)
    if clean_text(simulation_lock.get("status")) != (
        "FROZEN_ARCHITECTURE_SCREENING_INPUTS_BEFORE_MODEL_FITTING"
    ):
        raise RuntimeError("Unexpected 05b v2 lock status.")

    linear_lock = load_json(LINEAR_LOCK)
    if clean_text(linear_lock.get("status")) != (
        "FROZEN_LINEAR_BASELINE_RESULTS_BEFORE_MOFA_AND_NEURAL_FITTING"
    ):
        raise RuntimeError("Unexpected 05c1 lock status.")
    if sha256_file(LINEAR_RESULTS_MANIFEST) != clean_text(
        linear_lock.get("result_manifest", {}).get("sha256")
    ):
        raise RuntimeError("05c1 result manifest verification failed.")

    mofa_protocol_lock = load_json(MOFA_PROTOCOL_LOCK)
    if clean_text(mofa_protocol_lock.get("status")) != (
        "FROZEN_MOFA_OOS_PROJECTION_PROTOCOL_BEFORE_MOFA_FITTING"
    ):
        raise RuntimeError("Unexpected 05c2a lock status.")
    if sha256_file(MOFA_PROTOCOL) != clean_text(
        mofa_protocol_lock.get("protocol", {}).get("sha256")
    ):
        raise RuntimeError("05c2a protocol verification failed.")

    mofa_result_lock = load_json(MOFA_RESULTS_LOCK)
    if clean_text(mofa_result_lock.get("status")) != (
        "FROZEN_MOFA_BASELINE_RESULTS_BEFORE_NEURAL_FITTING_AND_ARCHITECTURE_SELECTION"
    ):
        raise RuntimeError("Unexpected 05c2b result lock status.")
    if sha256_file(MOFA_RESULTS_MANIFEST) != clean_text(
        mofa_result_lock.get("result_manifest", {}).get("sha256")
    ):
        raise RuntimeError("05c2b result manifest verification failed.")

    mofa_results = load_json(MOFA_RESULTS_MANIFEST)
    scope = mofa_results.get("scope", {})
    if int(scope.get("n_fit_fail", -1)) != 0:
        raise RuntimeError("05c2b reports failed MOFA fits.")
    if int(scope.get("n_fit_pass", -1)) != 17280:
        raise RuntimeError("05c2b successful-fit count differs from frozen benchmark.")

    state = mofa_results.get("scientific_state", {})
    if bool(state.get("architecture_selected", True)):
        raise RuntimeError("Architecture was selected before 05d0.")
    if bool(state.get("neural_model_fitting_performed", True)):
        raise RuntimeError("Neural model fitting occurred before 05d0.")
    if bool(state.get("final_gate_calibration_data_used", True)):
        raise RuntimeError("Final gate-calibration data were used before 05d0.")

    return {
        "05a_contract_sha256": sha256_file(CONTRACT_05A),
        "05a1_addendum_sha256": sha256_file(CONTRACT_05A1),
        "05b_lock_sha256": sha256_file(SIMULATION_LOCK),
        "05c1_result_manifest_sha256": sha256_file(LINEAR_RESULTS_MANIFEST),
        "05c2a_protocol_sha256": sha256_file(MOFA_PROTOCOL),
        "05c2b_result_manifest_sha256": sha256_file(MOFA_RESULTS_MANIFEST),
        "05c2b_successful_fits": int(scope["n_fit_pass"]),
        "05c2b_failed_fits": int(scope["n_fit_fail"]),
    }


def nvidia_smi_query() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "rows": [],
        }

    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            rows.append(
                {
                    "name": parts[0],
                    "driver_version": parts[1],
                    "memory_total_mib": float(parts[2]),
                }
            )

    return {
        "available": True,
        "error": None,
        "rows": rows,
    }


def import_torch_or_fail() -> Any:
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError(
            "PyTorch is not installed in this venv. "
            "Install the intended CUDA-enabled stable PyTorch build, then rerun 05d0."
        )

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

    import torch

    return torch


def configure_determinism(torch: Any) -> None:
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if hasattr(torch.backends, "cuda") and hasattr(
        torch.backends.cuda, "matmul"
    ):
        torch.backends.cuda.matmul.allow_tf32 = False

    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def torch_environment_snapshot(torch: Any) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0

    devices = []
    if cuda_available:
        for index in range(device_count):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "compute_capability": [
                        int(properties.major),
                        int(properties.minor),
                    ],
                    "total_memory_bytes": int(properties.total_memory),
                    "multi_processor_count": int(properties.multi_processor_count),
                }
            )

    cudnn_version = (
        torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available()
        else None
    )

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
            for name in [
                "torch",
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
            ]
        },
        "torch": {
            "version": torch.__version__,
            "cuda_build_version": torch.version.cuda,
            "git_version": clean_text(getattr(torch.version, "git_version", "")),
            "cuda_available": cuda_available,
            "cuda_device_count": device_count,
            "cudnn_available": bool(torch.backends.cudnn.is_available()),
            "cudnn_version": cudnn_version,
            "deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "deterministic_warn_only": bool(
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "float32_matmul_precision": (
                torch.get_float32_matmul_precision()
                if hasattr(torch, "get_float32_matmul_precision")
                else None
            ),
        },
        "cuda_devices": devices,
        "nvidia_smi": nvidia_smi_query(),
        "environment_variables": {
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
        },
        "runtime_policy": RUNTIME_POLICY,
    }


def seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one_fixture(torch: Any, seed: int) -> dict[str, Any]:
    seed_everything(torch, seed)
    device = torch.device("cuda")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    x_cpu = torch.randn(
        48,
        18,
        generator=generator,
        dtype=torch.float32,
    )
    target_cpu = torch.randn(
        48,
        18,
        generator=generator,
        dtype=torch.float32,
    )

    x = x_cpu.to(device=device)
    target = target_cpu.to(device=device)

    model = torch.nn.Sequential(
        torch.nn.Linear(18, 14),
        torch.nn.ReLU(),
        torch.nn.Linear(14, 7),
        torch.nn.Tanh(),
        torch.nn.Linear(7, 14),
        torch.nn.ReLU(),
        torch.nn.Linear(14, 18),
    ).to(device=device, dtype=torch.float32)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )

    losses = []

    for _ in range(REFERENCE_STEPS):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = torch.nn.functional.mse_loss(
            prediction,
            target,
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    torch.cuda.synchronize()

    state = {
        key: tensor.detach().cpu().clone()
        for key, tensor in model.state_dict().items()
    }
    prediction = model(x).detach().cpu().clone()

    return {
        "losses": losses,
        "state": state,
        "prediction": prediction,
    }


def tensor_sha256(tensor: Any) -> str:
    values = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(str(tuple(values.shape)).encode("utf-8"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def run_determinism_fixture(torch: Any) -> dict[str, Any]:
    first = run_one_fixture(torch, REFERENCE_SEED)
    second = run_one_fixture(torch, REFERENCE_SEED)

    state_keys_equal = list(first["state"]) == list(second["state"])
    exact_state_match = state_keys_equal and all(
        torch.equal(
            first["state"][key],
            second["state"][key],
        )
        for key in first["state"]
    )

    exact_prediction_match = bool(
        torch.equal(
            first["prediction"],
            second["prediction"],
        )
    )

    exact_loss_match = first["losses"] == second["losses"]

    if not (
        exact_state_match
        and exact_prediction_match
        and exact_loss_match
    ):
        raise RuntimeError(
            "PyTorch deterministic reference fixture failed exact repeatability."
        )

    return {
        "fixture_id": "05d0-deterministic-cuda-mlp-v1",
        "seed": REFERENCE_SEED,
        "steps": REFERENCE_STEPS,
        "device": "cuda",
        "dtype": "float32",
        "optimizer": "Adam",
        "automatic_mixed_precision": False,
        "torch_compile": False,
        "exact_state_dict_match": exact_state_match,
        "exact_prediction_match": exact_prediction_match,
        "exact_loss_history_match": exact_loss_match,
        "final_loss": first["losses"][-1],
        "prediction_sha256": tensor_sha256(
            first["prediction"]
        ),
        "state_dict_sha256": {
            key: tensor_sha256(tensor)
            for key, tensor in first["state"].items()
        },
        "status": "PASS",
    }


def verify_existing_lock() -> None:
    lock = load_json(OUTPUT_LOCK)

    if clean_text(lock.get("script_version")) != SCRIPT_VERSION:
        raise RuntimeError("Existing 05d0 script-version mismatch.")

    failures = []

    for relative_name, record in lock.get("generated_files", {}).items():
        candidate = resolve_project_path(relative_name)

        if not candidate.exists():
            failures.append(f"missing: {relative_name}")
            continue

        if int(candidate.stat().st_size) != int(record["size_bytes"]):
            failures.append(f"size mismatch: {relative_name}")
            continue

        if sha256_file(candidate) != clean_text(record["sha256"]):
            failures.append(f"SHA256 mismatch: {relative_name}")

    if failures:
        raise RuntimeError(
            "Existing 05d0 lock verification failed:\n"
            + "\n".join(f"- {item}" for item in failures)
        )

    print("=" * 88)
    print("Existing Paper 5 neural environment lock: PASS")
    print("=" * 88)
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("No files were rewritten.")
    print("=" * 88)


def main() -> None:
    print("=" * 88)
    print("Paper 5 - freeze PyTorch neural environment")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_LOCK.exists():
        verify_existing_lock()
        return

    upstream = verify_upstream()

    print("05a AI method contract: PASS")
    print("05a1 simulation/confound addendum: PASS")
    print("05b v2 confirmatory simulation bank: PASS")
    print("05c1 linear baseline results: PASS")
    print("05c2a MOFA projection protocol: PASS")
    print("05c2b MOFA baseline results: PASS")
    print(f"MOFA successful fits: {upstream['05c2b_successful_fits']:,}")
    print(f"MOFA failed fits: {upstream['05c2b_failed_fits']:,}")
    print("Neural model results before 05d0: NO")
    print("Architecture selected before 05d0: NO")
    print("Final gate-calibration data accessed: NO")
    print("Outcome access: FORBIDDEN")
    print()

    torch = import_torch_or_fail()
    configure_determinism(torch)

    environment = torch_environment_snapshot(torch)

    print(f"PyTorch version: {environment['torch']['version']}")
    print(f"CUDA build: {environment['torch']['cuda_build_version']}")
    print(f"CUDA available: {environment['torch']['cuda_available']}")

    if not environment["torch"]["cuda_available"]:
        raise RuntimeError(
            "CUDA is not available to PyTorch. "
            "Confirmatory neural benchmark is not permitted on this environment."
        )

    if environment["torch"]["cuda_device_count"] < 1:
        raise RuntimeError("No CUDA device is visible to PyTorch.")

    primary_gpu = environment["cuda_devices"][0]

    print(f"CUDA device 0: {primary_gpu['name']}")
    print(
        "Compute capability: "
        f"{primary_gpu['compute_capability'][0]}."
        f"{primary_gpu['compute_capability'][1]}"
    )
    print(
        "Total GPU memory: "
        f"{primary_gpu['total_memory_bytes'] / (1024 ** 3):.2f} GiB"
    )
    print(
        "Deterministic algorithms enabled: "
        f"{environment['torch']['deterministic_algorithms_enabled']}"
    )
    print(f"cuDNN benchmark: {environment['torch']['cudnn_benchmark']}")
    print(f"cuDNN deterministic: {environment['torch']['cudnn_deterministic']}")
    print(
        "CUBLAS_WORKSPACE_CONFIG: "
        f"{environment['environment_variables']['CUBLAS_WORKSPACE_CONFIG']}"
    )
    print()

    print("Running deterministic CUDA neural reference fixture...")
    fixture = run_determinism_fixture(torch)
    print("Deterministic CUDA neural reference fixture: PASS")
    print("Exact repeated model state: PASS")
    print("Exact repeated predictions: PASS")
    print("Exact repeated loss trajectory: PASS")
    print()

    contract = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "scientific_state": {
            "outcomes_loaded": False,
            "clinical_labels_loaded": False,
            "biological_annotations_loaded": False,
            "TARGET_model_fitting_performed": False,
            "neural_simulation_model_fitting_performed": False,
            "architecture_selected": False,
            "stability_gate_thresholds_retuned": False,
            "final_gate_calibration_data_used": False,
        },
        "runtime_policy": RUNTIME_POLICY,
        "device_freeze": {
            "confirmatory_device": "cuda:0",
            "gpu_name": primary_gpu["name"],
            "compute_capability": primary_gpu["compute_capability"],
            "total_memory_bytes": primary_gpu["total_memory_bytes"],
        },
        "software_freeze": {
            "python_version": environment["python"]["version"],
            "torch_version": environment["torch"]["version"],
            "torch_cuda_build_version": environment["torch"][
                "cuda_build_version"
            ],
            "cudnn_version": environment["torch"]["cudnn_version"],
        },
        "determinism_contract": {
            "python_random_seed_required": True,
            "numpy_seed_required": True,
            "torch_manual_seed_required": True,
            "torch_cuda_manual_seed_all_required": True,
            "torch_use_deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "automatic_mixed_precision_allowed": False,
            "torch_compile_allowed": False,
            "tf32_allowed": False,
            "primary_dtype": "float32",
            "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG,
            "primary_dataloader_workers": 0,
        },
        "upstream_hashes": upstream,
        "next_required_stage": "05d1_freeze_neural_simulation_protocol.py",
    }

    write_json_atomic(environment, OUTPUT_ENVIRONMENT)
    write_json_atomic(fixture, OUTPUT_FIXTURE)
    write_json_atomic(contract, OUTPUT_CONTRACT)

    readme_lines = [
        "Paper 5 neural runtime environment lock",
        "=" * 72,
        "",
        f"Script version: {SCRIPT_VERSION}",
        "",
        "Purpose",
        "-------",
        "Freeze the PyTorch/CUDA execution environment before any confirmatory neural fit.",
        "",
        "Runtime policy",
        "--------------",
        "CUDA device 0 only.",
        "float32 only.",
        "Automatic mixed precision disabled.",
        "torch.compile disabled.",
        "TF32 disabled.",
        "torch.use_deterministic_algorithms(True, warn_only=False).",
        "cuDNN benchmark disabled.",
        "cuDNN deterministic enabled.",
        f"CUBLAS_WORKSPACE_CONFIG={CUBLAS_WORKSPACE_CONFIG}",
        "Primary DataLoader workers = 0.",
        "",
        "Scientific state",
        "----------------",
        "Neural simulation model fitting performed: NO",
        "Architecture selected: NO",
        "Final gate-calibration data used: NO",
        "Outcomes accessed: NO",
        "",
        "Next stage",
        "----------",
        "05d1_freeze_neural_simulation_protocol.py",
        "",
    ]

    OUTPUT_README.write_text(
        "\n".join(readme_lines),
        encoding="utf-8",
    )

    generated = [
        OUTPUT_CONTRACT,
        OUTPUT_ENVIRONMENT,
        OUTPUT_FIXTURE,
        OUTPUT_README,
    ]

    lock = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": (
            "FROZEN_NEURAL_RUNTIME_ENVIRONMENT_BEFORE_NEURAL_MODEL_FITTING"
        ),
        "scientific_state": contract["scientific_state"],
        "contract": {
            "path": relative_path(OUTPUT_CONTRACT),
            "sha256": sha256_file(OUTPUT_CONTRACT),
        },
        "generated_files": {
            relative_path(path): {
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
            for path in generated
        },
        "rewrite_policy": (
            "This zero-argument stage never rewrites a frozen neural runtime lock. "
            "Any change in PyTorch, CUDA, GPU, determinism settings, or precision "
            "requires a new versioned environment contract before neural refitting."
        ),
    }

    write_json_atomic(lock, OUTPUT_LOCK)

    print("-" * 88)
    print("Frozen neural runtime policy")
    print("-" * 88)
    print("Confirmatory device: cuda:0")
    print("Primary dtype: float32")
    print("Automatic mixed precision: FORBIDDEN")
    print("torch.compile: FORBIDDEN")
    print("TF32: FORBIDDEN")
    print("Deterministic algorithms: REQUIRED")
    print("cuDNN benchmark: OFF")
    print("Primary DataLoader workers: 0")
    print()

    print("=" * 88)
    print("Paper 5 neural runtime environment: FROZEN")
    print("=" * 88)
    print(f"Contract: {relative_path(OUTPUT_CONTRACT)}")
    print(f"Environment: {relative_path(OUTPUT_ENVIRONMENT)}")
    print(f"Fixture: {relative_path(OUTPUT_FIXTURE)}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("Neural simulation model fitting performed: NO")
    print("Architecture selected: NO")
    print("Final gate-calibration data accessed: NO")
    print("Next stage: 05d1_freeze_neural_simulation_protocol.py")
    print("=" * 88)


if __name__ == "__main__":
    main()
