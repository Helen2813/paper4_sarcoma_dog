from __future__ import annotations

import dataclasses
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Frozen deterministic CUDA GEMM configuration must be set before CUDA work.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pandas as pd
import torch


SCRIPT_VERSION = "05f1-run-final-gate-runtime-pilot-gpu-v1-no-cli"
CONTRACT_VERSION = "paper5-final-gate-runtime-pilot-v1"
PRELOCK_STATUS = "FROZEN_FINAL_GATE_NUMERICAL_IMPLEMENTATION_BEFORE_FIRST_PILOT_FIT"
PILOT_LOCK_STATUS = "FROZEN_FINAL_GATE_RUNTIME_PILOT_COMPLETE_BEFORE_FULL_RUN"

EXPECTED_SCENARIOS = 24
EXPECTED_DATASETS = 3_100
EXPECTED_STABILITY_RESAMPLES = 100
EXPECTED_PILOT_DATASETS = 3
EXPECTED_METHOD_ID = "V4_SABLE_MO"
EXPECTED_PROFILE_ID = "ARCH_COMPACT__REG_WEAK"
EXPECTED_FINAL_STAGE = "final_gate_calibration"
EXPECTED_FINAL_MATERIALIZATION_POLICY = "stream_or_temporary_after_architecture_freeze"
EXPECTED_05F0_STATUS = "FROZEN_FINAL_GATE_STREAMING_EXECUTION_PLAN_BEFORE_RUNTIME_PILOT"
EXPECTED_05E_STATUS = "FROZEN_PRIMARY_PROPOSED_REPRESENTATION_METHOD_BEFORE_FINAL_GATE_AND_TARGET"
EXPECTED_05A2_STATUS = "FROZEN_BEFORE_FINAL_GATE_MATERIALIZATION"
EXPECTED_05D5_STATUS = "FROZEN_NEURAL_CONFIRMATORY_STABILITY_RESULTS_BEFORE_05E"
EXPECTED_05D3_STATUS = "FROZEN_OUTCOME_BLIND_NEURAL_PROFILE_BEFORE_CONFIRMATORY_ABLATION"
EXPECTED_05D0_STATUS = "FROZEN_NEURAL_RUNTIME_ENVIRONMENT_BEFORE_NEURAL_MODEL_FITTING"
EXPECTED_05D1_STATUS = "FROZEN_NEURAL_SIMULATION_PROTOCOL_BEFORE_NEURAL_MODEL_FITTING"
EXPECTED_05B_STATUS = "FROZEN_ARCHITECTURE_SCREENING_INPUTS_BEFORE_MODEL_FITTING"
EXPECTED_GENERATOR_VERSION = "sable-mo-simulation-generator-v2"
EXPECTED_GENERATOR_SCRIPT_VERSION = "05b-generate-small-n-multiomics-simulations-v2-no-cli"
EXPECTED_NEURAL_SCRIPT_VERSION = "05d2v2-screen-neural-profiles-gpu-syncfree-v2-no-cli"
EXPECTED_HELPER_BASE_STABILITY_RESAMPLES = 20

# The same frozen SABLE-MO stability scheme is evaluated more precisely at the
# final gate: 80% subsamples, now with the preregistered 100 resamples/dataset.
STABILITY_SUBSAMPLE_FRACTION = 0.80

# Prospective final-gate stability seed namespaces. These are frozen before the
# first pilot fit and reused unchanged by the full final-gate run.
REFERENCE_SEED_NAMESPACE = "paper5_final_gate_stability_reference_v1"
SUBSAMPLE_SEED_NAMESPACE = "paper5_final_gate_stability_subsample_v1"
FIT_SEED_NAMESPACE = "paper5_final_gate_stability_fit_v1"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# -----------------------------------------------------------------------------
# Frozen upstream files
# -----------------------------------------------------------------------------

DIR_05A = PAPER5_ROOT / "method_contract" / "05a"
CONTRACT_05A = DIR_05A / "ai_method_contract.json"
LOCK_05A = DIR_05A / "ai_method_contract_lock.json"

DIR_05A1 = PAPER5_ROOT / "method_contract" / "05a1"
CONTRACT_05A1 = DIR_05A1 / "simulation_and_technical_confound_addendum.json"
LOCK_05A1 = DIR_05A1 / "simulation_and_technical_confound_addendum_lock.json"
FINAL_GATE_SCENARIOS = DIR_05A1 / "final_gate_calibration_scenarios.csv"
FINAL_GATE_REPLICATES = DIR_05A1 / "final_gate_calibration_replicates.csv"

DIR_05A2 = PAPER5_ROOT / "method_contract" / "05a2"
AMENDMENT_05A2 = DIR_05A2 / "inference_final_gate_implementation_amendment.json"
RUNTIME_POLICY_05A2 = DIR_05A2 / "final_gate_runtime_pilot_policy.json"
LOCK_05A2 = DIR_05A2 / "inference_final_gate_implementation_amendment_lock.json"

DIR_05B = PAPER5_ROOT / "method_contract" / "05b_v2"
LOCK_05B = DIR_05B / "simulation_generation_v2_lock.json"
FROZEN_GENERATOR_FIXTURE = DIR_05B / "simulation_reference_fixture_hashes.json"
GENERATOR_SCRIPT = PAPER5_ROOT / "scripts" / "05b_generate_small_n_multiomics_simulations_v2.py"

DIR_05D0 = PAPER5_ROOT / "method_contract" / "05d0"
ENV_CONTRACT = DIR_05D0 / "neural_environment_contract.json"
ENV_LOCK = DIR_05D0 / "neural_environment_lock.json"

DIR_05D1 = PAPER5_ROOT / "method_contract" / "05d1"
PROTOCOL_05D1 = DIR_05D1 / "neural_simulation_evaluation_protocol.json"
LOSS_CONTRACT_05D1 = DIR_05D1 / "neural_loss_and_architecture_contract.json"
LOCK_05D1 = DIR_05D1 / "neural_simulation_protocol_lock.json"
PROFILE_GRID_05D1 = DIR_05D1 / "neural_architecture_regularization_profile_grid.csv"

NEURAL_IMPL_SCRIPT = PAPER5_ROOT / "scripts" / "05d2v2_screen_neural_profiles_gpu_syncfree_pathfix.py"
LINEAR_IMPL_SCRIPT = PAPER5_ROOT / "scripts" / "05c1_fit_linear_simulation_baselines.py"
DIR_05D2 = PAPER5_ROOT / "method_contract" / "05d2_v2"
PROTOCOL_05D2 = DIR_05D2 / "neural_profile_screen_execution_protocol.json"
LOCK_05D2 = DIR_05D2 / "neural_profile_screen_results_lock.json"

DIR_05D3 = PAPER5_ROOT / "method_contract" / "05d3"
CONTRACT_05D3 = DIR_05D3 / "selected_neural_profile.json"
LOCK_05D3 = DIR_05D3 / "selected_neural_profile_lock.json"

DIR_05D5 = PAPER5_ROOT / "method_contract" / "05d5"
LOCK_05D5 = DIR_05D5 / "neural_confirmatory_stability_results_lock.json"
MANIFEST_05D5 = DIR_05D5 / "neural_confirmatory_stability_results_manifest.json"
EPOCH_REGISTRY_05D5 = DIR_05D5 / "confirmatory_stability_epoch_registry.csv"

DIR_05E = PAPER5_ROOT / "method_contract" / "05e"
FREEZE_05E = DIR_05E / "representation_method_freeze.json"
LOCK_05E = DIR_05E / "representation_method_freeze_lock.json"

DIR_05F0 = PAPER5_ROOT / "method_contract" / "05f0"
PLAN_05F0 = DIR_05F0 / "final_gate_streaming_execution_plan.json"
PILOT_REGISTRY_05F0 = DIR_05F0 / "final_gate_runtime_pilot_registry.csv"
LOCK_05F0 = DIR_05F0 / "final_gate_streaming_execution_plan_lock.json"
SCRIPT_05F0 = PAPER5_ROOT / "scripts" / "05f0_freeze_final_gate_streaming_execution_plan.py"

FINAL_GATE_DATA_ROOT = PAPER5_ROOT / "simulations" / "sable_mo_v2" / "final_gate_calibration"

# -----------------------------------------------------------------------------
# 05f1 outputs
# -----------------------------------------------------------------------------

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05f1"
RESULTS_ROOT = PAPER5_ROOT / "results" / "simulation_benchmark" / "05f_final_gate"
SHARD_DIR = RESULTS_ROOT / "shards" / "_compact"

OUTPUT_EPOCH_BRIDGE = OUTPUT_DIR / "final_gate_fixed_epoch_bridge.json"
OUTPUT_SEED_REGISTRY = OUTPUT_DIR / "final_gate_stability_seed_registry.csv.gz"
OUTPUT_NUMERICAL_CONTRACT = OUTPUT_DIR / "final_gate_numerical_runtime_contract.json"
OUTPUT_PRELOCK = OUTPUT_DIR / "final_gate_runtime_pilot_prelock.json"
OUTPUT_FIXTURE = OUTPUT_DIR / "final_gate_runtime_reference_fixture.json"
OUTPUT_PILOT_MANIFEST = OUTPUT_DIR / "final_gate_runtime_pilot_manifest.csv"
OUTPUT_PILOT_SUMMARY = OUTPUT_DIR / "final_gate_runtime_pilot_summary.json"
OUTPUT_README = OUTPUT_DIR / "05f1_runtime_pilot_README.txt"
OUTPUT_LOCK = OUTPUT_DIR / "final_gate_runtime_pilot_lock.json"
OUTPUT_IN_PROGRESS = OUTPUT_DIR / ".final_gate_runtime_pilot_in_progress.json"
OUTPUT_FAILURE = OUTPUT_DIR / "final_gate_runtime_pilot_failure.json"


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


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return clean_text(value).lower() in {"true", "1", "yes", "y"}


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def resolve_project_path(value: Any) -> Path:
    path = Path(clean_text(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(to_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def load_json(path: Path) -> dict[str, Any]:
    require_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


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
        number = float(value)
        return number if np.isfinite(number) else None
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
            json.dumps(to_json_safe(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".tmp_", suffix=".txt", dir=str(path.parent))
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_csv_atomic(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".tmp_", suffix=".csv", dir=str(path.parent))
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        table.to_csv(temporary, index=False, float_format="%.12g", lineterminator="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_csv_gzip_atomic(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".tmp_", suffix=".csv.gz", dir=str(path.parent))
    os.close(descriptor)
    temporary = Path(temporary_name)
    compression: Any = {"method": "gzip", "compresslevel": 6, "mtime": 0}
    try:
        try:
            table.to_csv(
                temporary,
                index=False,
                float_format="%.12g",
                lineterminator="\n",
                compression=compression,
            )
        except TypeError:
            table.to_csv(
                temporary,
                index=False,
                float_format="%.12g",
                lineterminator="\n",
                compression="gzip",
            )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def file_record(path: Path) -> dict[str, Any]:
    require_file(path)
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def verify_generated_files(lock: dict[str, Any], label: str) -> None:
    records = lock.get("generated_files", {})
    if not isinstance(records, dict) or not records:
        raise RuntimeError(f"{label} lock has no generated_files registry.")
    failures: list[str] = []
    for relative_name, record in records.items():
        path = resolve_project_path(relative_name)
        if not path.is_file():
            failures.append(f"missing: {relative_name}")
            continue
        expected_size = record.get("size_bytes")
        if expected_size is not None and int(path.stat().st_size) != int(expected_size):
            failures.append(f"size mismatch: {relative_name}")
            continue
        expected_hash = clean_text(record.get("sha256"))
        if expected_hash and sha256_file(path) != expected_hash:
            failures.append(f"SHA256 mismatch: {relative_name}")
    if failures:
        raise RuntimeError(f"{label} generated-file verification failed:\n- " + "\n- ".join(failures[:100]))


def import_module(path: Path, module_name: str) -> Any:
    require_file(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def any_files(path: Path) -> bool:
    return path.is_file() or (path.is_dir() and any(item.is_file() for item in path.rglob("*")))


def normalized_path_key(value: Any) -> str:
    return clean_text(value).replace("\\", "/").lower()


def function_source_hash(functions: Iterable[Any]) -> str:
    pieces: list[str] = []
    for function in functions:
        pieces.append(f"### {function.__name__}\n{inspect.getsource(function).strip()}\n")
    return sha256_text("\n".join(pieces))


# -----------------------------------------------------------------------------
# Upstream verification and frozen runtime load
# -----------------------------------------------------------------------------

def verify_upstream_and_load() -> dict[str, Any]:
    required = [
        CONTRACT_05A,
        LOCK_05A,
        CONTRACT_05A1,
        LOCK_05A1,
        FINAL_GATE_SCENARIOS,
        FINAL_GATE_REPLICATES,
        AMENDMENT_05A2,
        RUNTIME_POLICY_05A2,
        LOCK_05A2,
        LOCK_05B,
        FROZEN_GENERATOR_FIXTURE,
        GENERATOR_SCRIPT,
        ENV_CONTRACT,
        ENV_LOCK,
        PROTOCOL_05D1,
        LOSS_CONTRACT_05D1,
        LOCK_05D1,
        PROFILE_GRID_05D1,
        NEURAL_IMPL_SCRIPT,
        LINEAR_IMPL_SCRIPT,
        PROTOCOL_05D2,
        LOCK_05D2,
        CONTRACT_05D3,
        LOCK_05D3,
        LOCK_05D5,
        MANIFEST_05D5,
        EPOCH_REGISTRY_05D5,
        FREEZE_05E,
        LOCK_05E,
        PLAN_05F0,
        PILOT_REGISTRY_05F0,
        LOCK_05F0,
        SCRIPT_05F0,
    ]
    for path in required:
        require_file(path)

    # 05f0 is the immediate frozen execution plan.
    lock_05f0 = load_json(LOCK_05F0)
    if clean_text(lock_05f0.get("status")) != EXPECTED_05F0_STATUS:
        raise RuntimeError("05f0 streaming plan is not frozen in the expected state.")
    if clean_text(lock_05f0.get("script_sha256")) != sha256_file(SCRIPT_05F0):
        raise RuntimeError("Current 05f0 script differs from its frozen script hash.")
    verify_generated_files(lock_05f0, "05f0")
    plan_05f0 = load_json(PLAN_05F0)
    if clean_text(plan_05f0.get("status")) != EXPECTED_05F0_STATUS:
        raise RuntimeError("05f0 plan JSON status mismatch.")
    scope_05f0 = plan_05f0.get("frozen_input_registry", {})
    if int(scope_05f0.get("n_scenarios", -1)) != EXPECTED_SCENARIOS:
        raise RuntimeError("05f0 scenario count changed.")
    if int(scope_05f0.get("n_datasets", -1)) != EXPECTED_DATASETS:
        raise RuntimeError("05f0 dataset count changed.")
    if int(scope_05f0.get("stability_resamples_per_dataset", -1)) != EXPECTED_STABILITY_RESAMPLES:
        raise RuntimeError("05f0 stability-resample count changed.")
    if clean_text(scope_05f0.get("materialization_policy")) != EXPECTED_FINAL_MATERIALIZATION_POLICY:
        raise RuntimeError("05f0 materialization policy changed.")
    pilot_scope = plan_05f0.get("runtime_pilot", {})
    if int(pilot_scope.get("dataset_count", -1)) != EXPECTED_PILOT_DATASETS:
        raise RuntimeError("05f0 pilot dataset count changed.")
    if int(pilot_scope.get("planned_optimization_runs", -1)) != EXPECTED_PILOT_DATASETS * (1 + EXPECTED_STABILITY_RESAMPLES):
        raise RuntimeError("05f0 pilot optimization-run count changed.")
    if clean_text(pilot_scope.get("scientific_console_output")) != "FORBIDDEN":
        raise RuntimeError("05f0 no longer forbids scientific pilot console output.")

    # No persistent final-gate input bank is allowed.
    if any_files(FINAL_GATE_DATA_ROOT):
        raise RuntimeError("Permanent final-gate input files exist; frozen streaming policy forbids this.")

    # Frozen 05a2 implementation clarification / pilot reuse rules.
    amendment_05a2 = load_json(AMENDMENT_05A2)
    lock_05a2 = load_json(LOCK_05A2)
    runtime_policy = load_json(RUNTIME_POLICY_05A2)
    if clean_text(amendment_05a2.get("status")) != EXPECTED_05A2_STATUS:
        raise RuntimeError("05a2 amendment status mismatch.")
    if clean_text(lock_05a2.get("status")) != EXPECTED_05A2_STATUS:
        raise RuntimeError("05a2 lock status mismatch.")
    if sha256_file(AMENDMENT_05A2) != clean_text(lock_05a2.get("amendment_sha256")):
        raise RuntimeError("05a2 amendment hash mismatch.")
    verify_generated_files(lock_05a2, "05a2")
    if clean_text(runtime_policy.get("status")) != "FROZEN_BEFORE_FINAL_GATE_RUNTIME_PILOT":
        raise RuntimeError("05a2 runtime-pilot policy status mismatch.")
    if "first prespecified dataset ids" not in clean_text(runtime_policy.get("pilot_dataset_selection")).lower():
        raise RuntimeError("05a2 pilot selection is no longer first frozen registry IDs.")
    visibility = clean_text(runtime_policy.get("pilot_scientific_output_visibility")).lower()
    if "do not print" not in visibility or "pass/fail" not in visibility:
        raise RuntimeError("05a2 pilot output visibility rule changed.")

    # Frozen proposed method/profile.
    freeze_05e = load_json(FREEZE_05E)
    lock_05e = load_json(LOCK_05E)
    if clean_text(freeze_05e.get("status")) != EXPECTED_05E_STATUS:
        raise RuntimeError("05e freeze status mismatch.")
    if clean_text(lock_05e.get("status")) != EXPECTED_05E_STATUS:
        raise RuntimeError("05e lock status mismatch.")
    verify_generated_files(lock_05e, "05e")
    if clean_text(lock_05e.get("primary_proposed_method_id")) != EXPECTED_METHOD_ID:
        raise RuntimeError("05e proposed method is not V4_SABLE_MO.")
    if clean_text(lock_05e.get("selected_profile_id")) != EXPECTED_PROFILE_ID:
        raise RuntimeError("05e profile is not ARCH_COMPACT__REG_WEAK.")

    # Confirmatory stability result is all-PASS and supplies the pre-final-gate
    # selected-epoch diagnostics only; no stability metric is used for the bridge.
    lock_05d5 = load_json(LOCK_05D5)
    manifest_05d5 = load_json(MANIFEST_05D5)
    if clean_text(lock_05d5.get("status")) != EXPECTED_05D5_STATUS:
        raise RuntimeError("05d5 is not the clean all-PASS frozen result.")
    verify_generated_files(lock_05d5, "05d5")
    scope_05d5 = manifest_05d5.get("scope", {})
    if int(scope_05d5.get("confirmatory_datasets", -1)) != 360:
        raise RuntimeError("05d5 confirmatory dataset count changed from 360.")
    if int(scope_05d5.get("passed_datasets", -1)) != 360 or int(scope_05d5.get("failed_datasets", -1)) != 0:
        raise RuntimeError("05d5 is not 360 PASS / 0 FAIL.")
    if clean_text(lock_05d5.get("variant_id")) != EXPECTED_METHOD_ID:
        raise RuntimeError("05d5 variant mismatch.")
    if clean_text(lock_05d5.get("profile_id")) != EXPECTED_PROFILE_ID:
        raise RuntimeError("05d5 profile mismatch.")

    # Frozen neural runtime and protocol.
    env_contract = load_json(ENV_CONTRACT)
    env_lock = load_json(ENV_LOCK)
    if clean_text(env_lock.get("status")) != EXPECTED_05D0_STATUS:
        raise RuntimeError("05d0 environment lock status mismatch.")
    verify_generated_files(env_lock, "05d0")
    expected_env_hash = clean_text(env_lock.get("contract", {}).get("sha256"))
    if not expected_env_hash or sha256_file(ENV_CONTRACT) != expected_env_hash:
        raise RuntimeError("05d0 environment contract hash mismatch.")

    protocol_05d1 = load_json(PROTOCOL_05D1)
    loss_05d1 = load_json(LOSS_CONTRACT_05D1)
    lock_05d1 = load_json(LOCK_05D1)
    if clean_text(lock_05d1.get("status")) != EXPECTED_05D1_STATUS:
        raise RuntimeError("05d1 lock status mismatch.")
    verify_generated_files(lock_05d1, "05d1")
    stability_contract = loss_05d1.get("stability_gate", {})
    if bool(stability_contract.get("training_component", True)):
        raise RuntimeError("05d1 stability gate unexpectedly became a training component.")
    if not bool(stability_contract.get("resampling_component", False)):
        raise RuntimeError("05d1 stability gate is no longer a resampling component.")
    if not math.isclose(float(stability_contract.get("subsample_fraction", float("nan"))), STABILITY_SUBSAMPLE_FRACTION, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("Frozen stability subsample fraction differs from 0.80.")

    # 05b generator provenance.
    lock_05b = load_json(LOCK_05B)
    if clean_text(lock_05b.get("status")) != EXPECTED_05B_STATUS:
        raise RuntimeError("05b v2 lock status mismatch.")
    verify_generated_files(lock_05b, "05b v2")

    generator = import_module(GENERATOR_SCRIPT, "paper5_05b_v2_generator_for_05f1")
    if clean_text(getattr(generator, "SCRIPT_VERSION", "")) != EXPECTED_GENERATOR_SCRIPT_VERSION:
        raise RuntimeError("05b generator script version changed.")
    if clean_text(getattr(generator, "GENERATOR_VERSION", "")) != EXPECTED_GENERATOR_VERSION:
        raise RuntimeError("05b generator version changed.")

    # Replay the frozen generator reference fixture before any final-gate input.
    frozen_fixture = load_json(FROZEN_GENERATOR_FIXTURE)
    current_fixture = generator.run_reference_fixture()
    fixture_keys = ["scenario_sha256", "npz_sha256", "npz_size_bytes", "array_sha256", "status"]
    mismatch = [key for key in fixture_keys if current_fixture.get(key) != frozen_fixture.get(key)]
    if mismatch:
        raise RuntimeError("05b generator no longer reproduces frozen fixture: " + ", ".join(mismatch))

    # Reuse 05f0's exact final-scenario parser/registry interpretation.
    f0 = import_module(SCRIPT_05F0, "paper5_05f0_streaming_plan_for_05f1")
    scenarios, scenarios_table, replicates, scenario_hashes = f0.load_final_gate_registries(generator)
    if len(scenarios) != EXPECTED_SCENARIOS or replicates.shape[0] != EXPECTED_DATASETS:
        raise RuntimeError("05f0 replay no longer yields 24 scenarios / 3,100 datasets.")

    pilot = pd.read_csv(PILOT_REGISTRY_05F0, dtype=str, low_memory=False).fillna("")
    if pilot.shape[0] != EXPECTED_PILOT_DATASETS:
        raise RuntimeError("05f0 pilot registry does not contain exactly 3 rows.")
    expected_pilot_keys = replicates.iloc[:EXPECTED_PILOT_DATASETS]["dataset_key"].astype(str).tolist()
    observed_pilot_keys = pilot["dataset_key"].astype(str).tolist()
    if observed_pilot_keys != expected_pilot_keys:
        raise RuntimeError("05f0 pilot registry is not the first three final-gate registry rows.")

    # Neural implementation / linear helper exact source path.
    neural = import_module(NEURAL_IMPL_SCRIPT, "paper5_05d2v2_final_gate_helpers")
    if clean_text(getattr(neural, "SCRIPT_VERSION", "")) != EXPECTED_NEURAL_SCRIPT_VERSION:
        raise RuntimeError("Frozen neural helper script version changed.")
    if int(getattr(neural, "STABILITY_RESAMPLES", -1)) != EXPECTED_HELPER_BASE_STABILITY_RESAMPLES:
        raise RuntimeError("Neural helper base stability-resample constant is no longer the frozen architecture value 20.")
    if Path(neural.LINEAR_SCRIPT).resolve() != LINEAR_IMPL_SCRIPT.resolve():
        raise RuntimeError("Neural helper no longer points to the expected 05c1 linear helper.")
    linear = neural.load_linear_helpers()

    # Check the 05d2 prospective protocol still freezes this exact linear script.
    protocol_05d2 = load_json(PROTOCOL_05D2)
    upstream_05d2 = {
        normalized_path_key(key): clean_text(value)
        for key, value in protocol_05d2.get("upstream_hashes", {}).items()
    }
    linear_key = normalized_path_key(relative_path(LINEAR_IMPL_SCRIPT))
    if linear_key not in upstream_05d2:
        raise RuntimeError("05d2 protocol does not contain the 05c1 helper hash.")
    if sha256_file(LINEAR_IMPL_SCRIPT) != upstream_05d2[linear_key]:
        raise RuntimeError("05c1 numerical helper changed after the frozen neural benchmark.")

    # Frozen profile reconstruction.
    contract_05d3 = load_json(CONTRACT_05D3)
    lock_05d3 = load_json(LOCK_05D3)
    if clean_text(contract_05d3.get("status")) != EXPECTED_05D3_STATUS or clean_text(lock_05d3.get("status")) != EXPECTED_05D3_STATUS:
        raise RuntimeError("05d3 selected-profile status mismatch.")
    verify_generated_files(lock_05d3, "05d3")
    profile_table = pd.read_csv(PROFILE_GRID_05D1, dtype=str, low_memory=False).fillna("")
    selected = profile_table[profile_table["profile_id"].eq(EXPECTED_PROFILE_ID)]
    if selected.shape[0] != 1:
        raise RuntimeError("Frozen profile is not unique in the 05d1 profile grid.")
    source_profile = contract_05d3.get("selected_profile", {}).get("source_profile_grid_row", {})
    selected_row = selected.iloc[0]
    for column in profile_table.columns:
        if clean_text(source_profile.get(column)) != clean_text(selected_row[column]):
            raise RuntimeError(f"05d3 frozen profile differs from 05d1 grid at {column}.")
    profile = neural.parse_profile(selected_row)
    if profile.profile_id != EXPECTED_PROFILE_ID:
        raise RuntimeError("Parsed profile ID mismatch.")

    # Frozen stability thresholds from 05a.
    contract_05a = load_json(CONTRACT_05A)
    lock_05a = load_json(LOCK_05A)
    if sha256_file(CONTRACT_05A) != clean_text(lock_05a.get("contract_sha256")):
        raise RuntimeError("05a contract hash mismatch.")
    verify_generated_files(lock_05a, "05a")
    thresholds = contract_05a.get("stability_gate", {}).get("thresholds", {})
    required_thresholds = {
        "primary_median_factor_correlation",
        "primary_q10_factor_correlation",
        "primary_inclusion_frequency",
        "minimum_factor_variance",
        "minimum_supported_modalities",
        "minimum_loading_energy_fraction_per_supported_modality",
    }
    if required_thresholds - set(thresholds):
        raise RuntimeError("05a stability thresholds are incomplete.")

    # Exact frozen runtime check. This initializes CUDA but performs no pilot fit.
    neural.assert_cuda_runtime()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA unavailable; quiet CPU fallback is forbidden.")
    frozen_software = env_contract.get("software_freeze", {})
    if clean_text(torch.__version__) != clean_text(frozen_software.get("torch_version")):
        raise RuntimeError("Active PyTorch version differs from 05d0.")
    if clean_text(torch.version.cuda) != clean_text(frozen_software.get("torch_cuda_build_version")):
        raise RuntimeError("Active CUDA build differs from 05d0.")
    frozen_gpu = clean_text(env_contract.get("device_freeze", {}).get("gpu_name"))
    if clean_text(torch.cuda.get_device_name(0)) != frozen_gpu:
        raise RuntimeError("Active GPU differs from the frozen 05d0 GPU.")

    return {
        "generator": generator,
        "f0": f0,
        "neural": neural,
        "linear": linear,
        "profile": profile,
        "stability_thresholds": thresholds,
        "contract_05a": contract_05a,
        "amendment_05a2": amendment_05a2,
        "runtime_policy": runtime_policy,
        "plan_05f0": plan_05f0,
        "scenarios": scenarios,
        "scenarios_table": scenarios_table,
        "replicates": replicates,
        "scenario_hashes": scenario_hashes,
        "pilot": pilot,
        "generator_fixture": current_fixture,
        "upstream_hashes": {relative_path(path): sha256_file(path) for path in required},
    }


# -----------------------------------------------------------------------------
# Prospective bridges frozen before first final-gate fit
# -----------------------------------------------------------------------------

def build_fixed_epoch_bridge(profile: Any) -> dict[str, Any]:
    table = pd.read_csv(EPOCH_REGISTRY_05D5, dtype=str, low_memory=False).fillna("")
    required = {
        "scenario_id",
        "replicate_index",
        "simulation_seed",
        "profile_id",
        "selected_epoch_mean_05d4",
        "selected_epoch_for_05d5",
        "performance_metrics_used_to_choose_epoch",
        "stability_results_used_to_choose_epoch",
        "TARGET_or_outcomes_used_to_choose_epoch",
    }
    missing = required - set(table.columns)
    if missing:
        raise RuntimeError("05d5 epoch registry missing: " + ", ".join(sorted(missing)))
    if table.shape[0] != 360:
        raise RuntimeError("05d5 epoch registry no longer contains 360 datasets.")
    if not table["profile_id"].eq(EXPECTED_PROFILE_ID).all():
        raise RuntimeError("05d5 epoch registry profile mismatch.")
    if table["performance_metrics_used_to_choose_epoch"].map(parse_bool).any():
        raise RuntimeError("05d5 epoch bridge unexpectedly used performance metrics.")
    if table["stability_results_used_to_choose_epoch"].map(parse_bool).any():
        raise RuntimeError("05d5 epoch bridge unexpectedly used stability results.")
    if table["TARGET_or_outcomes_used_to_choose_epoch"].map(parse_bool).any():
        raise RuntimeError("05d5 epoch bridge unexpectedly used TARGET/outcomes.")

    means = pd.to_numeric(table["selected_epoch_mean_05d4"], errors="raise").to_numpy(dtype=np.float64)
    rounded_05d5 = pd.to_numeric(table["selected_epoch_for_05d5"], errors="raise").to_numpy(dtype=np.int64)
    if not np.isfinite(means).all():
        raise RuntimeError("Non-finite confirmatory selected-epoch diagnostic.")
    if np.any(means < 1.0) or np.any(means > float(profile.maximum_epochs)):
        raise RuntimeError("Confirmatory selected-epoch means exceed frozen profile bounds.")

    # Equal-weight mean across the 360 confirmatory dataset means is equivalent
    # to the mean across all 3,600 V4 fold/seed selected-epoch diagnostics because
    # each dataset contributes exactly ten values. Half-up matches the frozen 05d5
    # bridge convention. No final-gate result can influence this value.
    grand_mean = float(np.mean(means))
    selected_epoch = int(math.floor(grand_mean + 0.5))
    selected_epoch = min(max(1, selected_epoch), int(profile.maximum_epochs))

    return {
        "bridge_version": "paper5-final-gate-fixed-epoch-bridge-v1",
        "created_at_utc": utc_now_iso(),
        "status": "FROZEN_BEFORE_FIRST_FINAL_GATE_FIT",
        "method_id": EXPECTED_METHOD_ID,
        "profile_id": EXPECTED_PROFILE_ID,
        "source_registry": file_record(EPOCH_REGISTRY_05D5),
        "source_confirmatory_datasets": 360,
        "source_fold_seed_selected_epoch_opportunities": 3_600,
        "source_selected_epoch_mean_grand_mean": grand_mean,
        "source_selected_epoch_mean_min": float(np.min(means)),
        "source_selected_epoch_mean_max": float(np.max(means)),
        "source_05d5_rounded_epoch_mean": float(np.mean(rounded_05d5)),
        "final_gate_fixed_epoch": selected_epoch,
        "rule": (
            "Arithmetic mean across the 360 frozen 05d5 selected_epoch_mean_05d4 diagnostics "
            "(equivalent to all 3,600 confirmatory V4 fold/seed selected epochs), then nearest integer "
            "half-up via floor(mean+0.5), bounded to the frozen profile maximum."
        ),
        "dataset_specific_final_gate_epoch": False,
        "05d4_performance_metrics_used": False,
        "05d5_stability_metrics_used": False,
        "final_gate_results_used": False,
        "TARGET_or_outcomes_used": False,
        "threshold_retuning_performed": False,
    }


def build_full_stability_seed_registry(upstream: dict[str, Any]) -> pd.DataFrame:
    generator = upstream["generator"]
    replicates = upstream["replicates"]
    scenarios = upstream["scenarios"]
    master_seed = int(upstream["contract_05a"]["simulation"]["master_seed"])

    rows: list[dict[str, Any]] = []
    for dataset in replicates.itertuples(index=False):
        scenario_id = clean_text(dataset.scenario_id)
        replicate_index = int(dataset.replicate_index)
        simulation_seed = int(dataset.simulation_seed)
        scenario = scenarios[scenario_id]
        n_samples = int(scenario.n_samples)
        n_subsample = max(2, int(math.floor(n_samples * STABILITY_SUBSAMPLE_FRACTION)))

        reference_seed = generator.stable_seed(
            master_seed,
            REFERENCE_SEED_NAMESPACE,
            scenario_id,
            replicate_index,
            simulation_seed,
        )

        for resample_index in range(1, EXPECTED_STABILITY_RESAMPLES + 1):
            rows.append(
                {
                    "registry_row_index": int(dataset.registry_row_index),
                    "dataset_key": clean_text(dataset.dataset_key),
                    "scenario_id": scenario_id,
                    "replicate_index": replicate_index,
                    "simulation_seed": simulation_seed,
                    "variant_id": EXPECTED_METHOD_ID,
                    "profile_id": EXPECTED_PROFILE_ID,
                    "reference_fit_seed": int(reference_seed),
                    "resample_index": resample_index,
                    "subsample_fraction": STABILITY_SUBSAMPLE_FRACTION,
                    "n_subsample": n_subsample,
                    "subsample_seed": int(
                        generator.stable_seed(
                            master_seed,
                            SUBSAMPLE_SEED_NAMESPACE,
                            scenario_id,
                            replicate_index,
                            simulation_seed,
                            resample_index,
                        )
                    ),
                    "resample_fit_seed": int(
                        generator.stable_seed(
                            master_seed,
                            FIT_SEED_NAMESPACE,
                            scenario_id,
                            replicate_index,
                            simulation_seed,
                            resample_index,
                        )
                    ),
                    "device": "cuda:0",
                }
            )

    table = pd.DataFrame(rows)
    expected_rows = EXPECTED_DATASETS * EXPECTED_STABILITY_RESAMPLES
    if table.shape[0] != expected_rows:
        raise RuntimeError(f"Expected {expected_rows:,} final-gate stability seed rows; found {table.shape[0]:,}.")
    if table.duplicated(["dataset_key", "resample_index"]).any():
        raise RuntimeError("Duplicate final-gate dataset/resample seed rows.")
    if table.groupby("dataset_key").size().ne(EXPECTED_STABILITY_RESAMPLES).any():
        raise RuntimeError("Every final-gate dataset must have exactly 100 stability seed rows.")
    grouped_reference = table.groupby("dataset_key")["reference_fit_seed"].nunique()
    if not grouped_reference.eq(1).all():
        raise RuntimeError("Reference fit seed is not unique within every final-gate dataset.")
    if table["subsample_seed"].duplicated().any():
        raise RuntimeError("Duplicate final-gate subsample seeds detected.")
    if table["resample_fit_seed"].duplicated().any():
        raise RuntimeError("Duplicate final-gate resample fit seeds detected.")
    return table


def configure_final_gate_neural_helper(neural: Any) -> None:
    # The frozen sync-free helper was written for 20 architecture/confirmatory
    # resamples and uses this module constant only to validate/label the number of
    # correlations accumulated by stability_audit. Final-gate scientific design
    # prospectively requires 100. No training, preprocessing, alignment, reduction,
    # threshold, dtype, optimizer or model code is changed.
    observed = int(getattr(neural, "STABILITY_RESAMPLES", -1))
    if observed not in {EXPECTED_HELPER_BASE_STABILITY_RESAMPLES, EXPECTED_STABILITY_RESAMPLES}:
        raise RuntimeError(f"Unexpected inherited neural STABILITY_RESAMPLES={observed}.")
    neural.STABILITY_RESAMPLES = EXPECTED_STABILITY_RESAMPLES
    if int(neural.STABILITY_RESAMPLES) != EXPECTED_STABILITY_RESAMPLES:
        raise RuntimeError("Failed to configure the frozen helper for 100 final-gate resamples.")


def array_bundle_sha256(generator: Any, arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(generator.deterministic_npy_bytes(np.asarray(arrays[key])))
        digest.update(b"\0")
    return digest.hexdigest()


def validate_final_gate_arrays(
    generator: Any,
    arrays: dict[str, np.ndarray],
    scenario: Any,
    registry_label: str,
    replicate_index: int,
    simulation_seed: int,
) -> dict[str, Any]:
    expected_scalar = {
        "scenario_id": scenario.scenario_id,
        "scenario_family": scenario.scenario_family,
        "registry_label": registry_label,
        "simulation_stage": EXPECTED_FINAL_STAGE,
        "generator_version": EXPECTED_GENERATOR_VERSION,
        "primary_truth_target": scenario.primary_truth_target,
    }
    for key, expected in expected_scalar.items():
        if key not in arrays or str(arrays[key].item()) != str(expected):
            raise RuntimeError(f"Final-gate generated array metadata mismatch: {key}.")
    if int(arrays["replicate_index"].item()) != int(replicate_index):
        raise RuntimeError("Final-gate replicate index mismatch.")
    if int(arrays["simulation_seed"].item()) != int(simulation_seed):
        raise RuntimeError("Final-gate simulation seed mismatch.")

    expected_scenario_hash = generator.sha256_text(generator.canonical_json(scenario.canonical_dict()))
    if str(arrays["scenario_sha256"].item()) != expected_scenario_hash:
        raise RuntimeError("Final-gate scenario canonical hash mismatch.")

    n = int(scenario.n_samples)
    k_bio = int(scenario.n_shared_factors)
    k_private = int(scenario.n_private_factors_per_view)
    k_tech = int(scenario.true_technical_shared_factors)

    if not np.array_equal(arrays["sample_index"], np.arange(n, dtype=np.int32)):
        raise RuntimeError("Final-gate sample_index mismatch.")
    if arrays["Z_biological_shared"].shape != (n, k_bio):
        raise RuntimeError("Final-gate biological truth shape mismatch.")
    if arrays["Z_technical"].shape != (n, k_tech):
        raise RuntimeError("Final-gate technical truth shape mismatch.")
    if arrays["batch_labels"].shape != (n, k_tech):
        raise RuntimeError("Final-gate batch-label shape mismatch.")
    if arrays["biological_support_mask"].shape != (k_bio, len(generator.VIEWS)):
        raise RuntimeError("Final-gate biological support-mask shape mismatch.")
    if arrays["technical_support_mask"].shape != (k_tech, len(generator.VIEWS)):
        raise RuntimeError("Final-gate technical support-mask shape mismatch.")
    if not np.all(np.isfinite(arrays["Z_biological_shared"])) or not np.all(np.isfinite(arrays["Z_technical"])):
        raise RuntimeError("Final-gate latent truth contains non-finite values.")

    total_missing = 0
    total_corrupted_rows = 0
    for view in generator.VIEWS:
        prefix = generator.VIEW_PREFIX[view]
        p = int(scenario.n_features(view))
        observed = arrays[f"X_{prefix}"]
        complete = arrays[f"X_complete_{prefix}"]
        reason = arrays[f"missing_reason_{prefix}"]
        private = arrays[f"Z_private_{prefix}"]
        w_bio = arrays[f"W_biological_shared_{prefix}"]
        w_private = arrays[f"W_private_{prefix}"]
        w_technical = arrays[f"W_technical_{prefix}"]
        row_source = arrays[f"row_source_{prefix}"]

        if observed.shape != (n, p) or observed.dtype != generator.DEFAULT_DTYPE:
            raise RuntimeError(f"{view}: final-gate observed matrix shape/dtype mismatch.")
        if complete.shape != (n, p) or complete.dtype != generator.DEFAULT_DTYPE:
            raise RuntimeError(f"{view}: final-gate complete matrix shape/dtype mismatch.")
        if not np.all(np.isfinite(complete)):
            raise RuntimeError(f"{view}: final-gate complete matrix contains non-finite values.")
        if reason.shape != (n, p) or reason.dtype != np.uint8:
            raise RuntimeError(f"{view}: final-gate missing-reason shape/dtype mismatch.")
        missing = ~np.isfinite(observed)
        if not np.array_equal(missing, reason > 0):
            raise RuntimeError(f"{view}: final-gate NaN and missing-reason masks disagree.")
        finite = ~missing
        if finite.any() and not np.array_equal(observed[finite], complete[finite]):
            raise RuntimeError(f"{view}: final-gate finite observed values differ from complete truth.")
        if private.shape != (n, k_private):
            raise RuntimeError(f"{view}: final-gate private-score shape mismatch.")
        if w_bio.shape != (p, k_bio) or w_private.shape != (p, k_private) or w_technical.shape != (p, k_tech):
            raise RuntimeError(f"{view}: final-gate loading shape mismatch.")
        if row_source.shape != (n,) or set(row_source.astype(int).tolist()) != set(range(n)):
            raise RuntimeError(f"{view}: final-gate row_source is not a permutation.")

        total_missing += int(missing.sum())
        total_corrupted_rows += int(np.sum(row_source != np.arange(n, dtype=np.int32)))

    return {
        "n_samples": n,
        "n_biological_shared_factors": k_bio,
        "n_private_factors_per_view": k_private,
        "n_technical_shared_factors": k_tech,
        "total_missing_cells": total_missing,
        "total_corrupted_view_rows": total_corrupted_rows,
        "scenario_sha256": expected_scenario_hash,
    }


def build_stability_group(seed_rows: pd.DataFrame, expected_n_samples: int) -> dict[str, Any]:
    if seed_rows.shape[0] != EXPECTED_STABILITY_RESAMPLES:
        raise RuntimeError("Final-gate stability group does not contain exactly 100 rows.")
    rows = seed_rows.copy()
    rows["resample_index"] = pd.to_numeric(rows["resample_index"], errors="raise").astype(int)
    rows = rows.sort_values("resample_index", kind="mergesort").reset_index(drop=True)
    if rows["resample_index"].tolist() != list(range(1, EXPECTED_STABILITY_RESAMPLES + 1)):
        raise RuntimeError("Final-gate resample indices are not exactly 1..100.")
    fractions = pd.to_numeric(rows["subsample_fraction"], errors="raise").to_numpy(dtype=float)
    if not np.allclose(fractions, STABILITY_SUBSAMPLE_FRACTION, rtol=0.0, atol=0.0):
        raise RuntimeError("Final-gate subsample fraction differs from 0.80.")
    expected_n = max(2, int(math.floor(expected_n_samples * STABILITY_SUBSAMPLE_FRACTION)))
    if set(pd.to_numeric(rows["n_subsample"], errors="raise").astype(int)) != {expected_n}:
        raise RuntimeError("Final-gate n_subsample differs from floor(0.80*n).")
    references = set(pd.to_numeric(rows["reference_fit_seed"], errors="raise").astype(int))
    if len(references) != 1:
        raise RuntimeError("Final-gate reference seed is not unique.")
    if pd.to_numeric(rows["subsample_seed"], errors="raise").astype(int).duplicated().any():
        raise RuntimeError("Duplicate subsample seed within final-gate dataset.")
    if pd.to_numeric(rows["resample_fit_seed"], errors="raise").astype(int).duplicated().any():
        raise RuntimeError("Duplicate resample fit seed within final-gate dataset.")
    return {"reference_fit_seed": int(next(iter(references))), "rows": rows}



def enrich_final_gate_truth_audit(
    neural: Any,
    linear: Any,
    arrays: dict[str, np.ndarray],
    reference_scores: np.ndarray,
    factor_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Add metric-sufficient truth audit without any additional model fit.

    The frozen neural stability helper already computed the full-data reference
    representation and transferred its shared scores to NumPy. 05f1 captures
    those exact scores during the inherited stability_audit call and uses them
    only for deterministic CPU truth-audit calculations required by the frozen
    05a1 operating-metric registry. No training/preprocessing/alignment/stability
    calculation is changed and no additional neural optimization run is added.
    """
    scores = np.asarray(reference_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(factor_rows):
        raise RuntimeError("Captured final-gate reference-score shape does not match factor rows.")
    if not np.isfinite(scores).all():
        raise RuntimeError("Captured final-gate reference scores contain non-finite values.")

    biological_truth, _ = linear.integrated_anchor_truth(arrays, "biological")
    technical_truth, technical_ids = linear.integrated_anchor_truth(arrays, "technical")
    biological_subspace = linear.subspace_metrics(scores, biological_truth)

    private_best_by_factor = np.zeros(scores.shape[1], dtype=np.float64)
    private_truth_present = False
    for view in neural.VIEWS:
        private_truth, _ = linear.row_aligned_truth(arrays, view, "private")
        correlation = linear.absolute_correlation_matrix(scores, private_truth)
        if correlation.shape[1] > 0:
            private_truth_present = True
            private_best_by_factor = np.maximum(
                private_best_by_factor,
                np.max(correlation, axis=1),
            )

    technical_rows_by_estimated: dict[int, dict[str, Any]] = {}
    if technical_truth.shape[1] > 0:
        _, technical_rows = linear.factor_matching(
            scores,
            technical_truth,
            technical_ids,
            float(neural.TECHNICAL_MATCH_THRESHOLD),
        )
        technical_rows_by_estimated = {
            int(row["estimated_factor_index"]): row
            for row in technical_rows
        }

    truth_support = arrays["biological_support_mask"].astype(bool)
    truth_classes = [
        str(value)
        for value in arrays["biological_factor_truth_class"].tolist()
    ]

    for factor_row in factor_rows:
        factor_index = int(factor_row["factor_index"])
        if factor_index < 0 or factor_index >= scores.shape[1]:
            raise RuntimeError("Final-gate factor index outside captured reference-score columns.")

        best_private = (
            float(private_best_by_factor[factor_index])
            if private_truth_present
            else None
        )
        best_biological_raw = factor_row.get("best_biological_absolute_correlation")
        best_biological = (
            float(best_biological_raw)
            if best_biological_raw is not None and np.isfinite(float(best_biological_raw))
            else None
        )
        factor_row["best_private_absolute_correlation"] = best_private
        factor_row["private_stronger_than_shared_truth"] = bool(
            best_private is not None
            and (best_biological is None or best_private > best_biological)
        )

        predicted_support = np.asarray(
            [
                float(factor_row[f"loading_energy_{view}"])
                >= float(neural.VIEW_SUPPORT_ENERGY_THRESHOLD)
                for view in neural.VIEWS
            ],
            dtype=bool,
        )
        factor_row["predicted_view_support"] = ";".join(
            view
            for view in neural.VIEWS
            if predicted_support[neural.VIEW_INDEX[view]]
        )

        truth_index_raw = factor_row.get("biological_truth_factor_index")
        if truth_index_raw is None:
            factor_row["truth_view_support"] = ""
            factor_row["truth_factor_class"] = ""
            factor_row["view_support_jaccard"] = None
            factor_row["view_support_exact"] = None
        else:
            truth_index = int(truth_index_raw)
            if truth_index < 0 or truth_index >= truth_support.shape[0]:
                raise RuntimeError("Final-gate biological truth index outside support mask.")
            actual_support = truth_support[truth_index, :]
            factor_row["truth_view_support"] = ";".join(
                view
                for view in neural.VIEWS
                if actual_support[neural.VIEW_INDEX[view]]
            )
            factor_row["truth_factor_class"] = (
                truth_classes[truth_index]
                if truth_index < len(truth_classes)
                else ""
            )
            factor_row["view_support_jaccard"] = float(
                linear.support_jaccard(predicted_support, actual_support)
            )
            factor_row["view_support_exact"] = bool(
                np.array_equal(predicted_support, actual_support)
            )

        technical_match = technical_rows_by_estimated.get(factor_index)
        if technical_match is None:
            factor_row["technical_truth_factor_index"] = None
            factor_row["technical_truth_factor_id"] = ""
            factor_row["technical_assigned_absolute_correlation"] = None
            factor_row["technical_assigned_truth_match"] = False
        else:
            factor_row["technical_truth_factor_index"] = technical_match.get("truth_factor_index")
            factor_row["technical_truth_factor_id"] = clean_text(technical_match.get("truth_factor_id"))
            factor_row["technical_assigned_absolute_correlation"] = technical_match.get("absolute_correlation")
            factor_row["technical_assigned_truth_match"] = bool(
                technical_match.get("passes_truth_match_threshold", False)
            )

    return {
        "reference_shared_subspace_minimum_correlation": biological_subspace.get("minimum_canonical_correlation"),
        "reference_shared_subspace_mean_angle_degrees": biological_subspace.get("mean_principal_angle_degrees"),
        "reference_shared_subspace_max_angle_degrees": biological_subspace.get("maximum_principal_angle_degrees"),
        "reference_shared_subspace_projection_distance": biological_subspace.get("normalized_projection_distance"),
        "private_truth_audit_available": bool(private_truth_present),
        "view_support_truth_audit_available": bool(truth_support.shape[0] > 0),
        "technical_truth_assignment_audit_available": bool(technical_truth.shape[1] > 0),
    }


def attach_final_gate_metadata(
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
                "execution_stage": "final_gate_calibration",
                "variant_id": EXPECTED_METHOD_ID,
                "profile_id": profile.profile_id,
                "architecture_profile": profile.architecture_profile,
                "regularization_profile": profile.regularization_profile,
                "shared_latent_dim": int(profile.shared_dim),
                "private_latent_dim_per_view": int(profile.private_dim),
                "selected_epoch_for_final_gate": int(selected_epoch),
                "factor_level_metrics_primary_eligible": bool(primary_eligible),
            }
        )


def evaluate_final_gate_dataset(
    neural: Any,
    linear: Any,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    profile: Any,
    selected_epoch: int,
    stability_group: dict[str, Any],
    stability_thresholds: dict[str, Any],
) -> dict[str, Any]:
    # Same frozen V4 stability_audit numerical path as 05d5, with the single
    # preregistered final-gate count change 20 -> 100 applied before execution.
    # Capture the already-computed full-data reference shared scores at the
    # helper's truth-audit boundary. The wrapper returns the original helper
    # result unchanged; the captured NumPy scores are used only after the
    # stability audit to seal the remaining frozen truth-audit diagnostics.
    captured_reference: dict[str, np.ndarray] = {}
    original_truth_status = neural.reference_factor_truth_status

    def capture_reference_truth_status(
        linear_arg: Any,
        arrays_arg: dict[str, np.ndarray],
        reference_scores_arg: np.ndarray,
    ) -> Any:
        captured_reference["scores"] = np.asarray(
            reference_scores_arg, dtype=np.float64
        ).copy()
        return original_truth_status(linear_arg, arrays_arg, reference_scores_arg)

    neural.reference_factor_truth_status = capture_reference_truth_status
    try:
        summary, factor_rows, fit_audits, preprocessing_rows = neural.stability_audit(
            linear=linear,
            arrays=arrays,
            profile=profile,
            selected_epoch=int(selected_epoch),
            stability_group=stability_group,
            stability_thresholds=stability_thresholds,
        )
    finally:
        neural.reference_factor_truth_status = original_truth_status

    if "scores" not in captured_reference:
        raise RuntimeError("Final-gate stability audit did not expose its reference-score truth-audit boundary.")

    truth_audit = enrich_final_gate_truth_audit(
        neural=neural,
        linear=linear,
        arrays=arrays,
        reference_scores=captured_reference["scores"],
        factor_rows=factor_rows,
    )

    if len(fit_audits) != 1 + EXPECTED_STABILITY_RESAMPLES:
        raise RuntimeError("Final-gate dataset does not contain 101 required fit audits.")
    if len(preprocessing_rows) != 3 + 3 * EXPECTED_STABILITY_RESAMPLES:
        raise RuntimeError("Final-gate dataset does not contain 303 required preprocessing rows.")
    if len(factor_rows) != int(profile.shared_dim):
        raise RuntimeError("Final-gate factor-audit row count differs from frozen shared dimension.")
    if int(summary.get("n_stability_resamples", -1)) != EXPECTED_STABILITY_RESAMPLES:
        raise RuntimeError("Final-gate helper did not report 100 stability resamples.")
    if int(summary.get("selected_epoch_used_for_stability", -1)) != int(selected_epoch):
        raise RuntimeError("Final-gate helper did not use the frozen fixed epoch.")

    primary_eligible = bool(
        metadata["factor_level_identifiable"]
        or int(metadata["n_true_biological_shared_factors"]) == 0
    )
    attach_final_gate_metadata(fit_audits, metadata, profile, selected_epoch, primary_eligible)
    attach_final_gate_metadata(preprocessing_rows, metadata, profile, selected_epoch, primary_eligible)
    attach_final_gate_metadata(factor_rows, metadata, profile, selected_epoch, primary_eligible)
    for row in factor_rows:
        row["factor_level_interpretation"] = (
            "primary_factor_level" if primary_eligible else "descriptive_only_rotational_nonnull"
        )

    dataset_row = {
        **metadata,
        "variant_id": EXPECTED_METHOD_ID,
        "profile_id": profile.profile_id,
        "architecture_profile": profile.architecture_profile,
        "regularization_profile": profile.regularization_profile,
        "shared_latent_dim": int(profile.shared_dim),
        "private_latent_dim_per_view": int(profile.private_dim),
        "hidden_dim": int(profile.hidden_dim),
        "factor_level_metrics_primary_eligible": primary_eligible,
        "selected_epoch_for_final_gate": int(selected_epoch),
        "reference_fit_seed": int(stability_group["reference_fit_seed"]),
        "required_reference_fits": 1,
        "required_resample_fits": EXPECTED_STABILITY_RESAMPLES,
        "all_required_fits_completed": True,
        **summary,
        **truth_audit,
        "compute_status": "COMPLETE",
    }

    return {
        "dataset_row": dataset_row,
        "factor_rows": factor_rows,
        "fit_audits": fit_audits,
        "preprocessing_rows": preprocessing_rows,
    }


# -----------------------------------------------------------------------------
# Prospective prelock / numerical implementation hash
# -----------------------------------------------------------------------------

def numerical_implementation_record(
    upstream: dict[str, Any],
    selected_epoch: int,
    seed_registry_sha256: str,
) -> dict[str, Any]:
    profile = upstream["profile"]
    thresholds = upstream["stability_thresholds"]

    local_numeric_source_hash = function_source_hash(
        [
            build_full_stability_seed_registry,
            configure_final_gate_neural_helper,
            array_bundle_sha256,
            validate_final_gate_arrays,
            build_stability_group,
            enrich_final_gate_truth_audit,
            attach_final_gate_metadata,
            evaluate_final_gate_dataset,
        ]
    )

    record = {
        "numerical_implementation_version": "paper5-final-gate-numerical-path-v1",
        "method_id": EXPECTED_METHOD_ID,
        "profile_id": EXPECTED_PROFILE_ID,
        "selected_epoch": int(selected_epoch),
        "stability_resamples_per_dataset": EXPECTED_STABILITY_RESAMPLES,
        "stability_subsample_fraction": STABILITY_SUBSAMPLE_FRACTION,
        "seed_namespaces": {
            "reference": REFERENCE_SEED_NAMESPACE,
            "subsample": SUBSAMPLE_SEED_NAMESPACE,
            "fit": FIT_SEED_NAMESPACE,
        },
        "seed_registry_sha256": seed_registry_sha256,
        "profile": {
            "pca_components": dict(profile.pca_components),
            "shared_dim": int(profile.shared_dim),
            "private_dim": int(profile.private_dim),
            "hidden_dim": int(profile.hidden_dim),
            "maximum_epochs": int(profile.maximum_epochs),
            "reconstruction_weight": float(profile.reconstruction_weight),
            "shared_alignment_weight": float(profile.shared_alignment_weight),
            "contrastive_weight": float(profile.contrastive_weight),
            "orthogonality_weight": float(profile.orthogonality_weight),
            "private_separation_weight": float(profile.private_separation_weight),
            "cross_reconstruction_weight": float(profile.cross_reconstruction_weight),
            "modality_dropout_probability": float(profile.modality_dropout_probability),
            "contrastive_temperature": float(profile.contrastive_temperature),
            "learning_rate": float(profile.learning_rate),
            "weight_decay": float(profile.weight_decay),
        },
        "stability_thresholds": thresholds,
        "runtime": {
            "device": "cuda:0",
            "neural_dtype": "float32",
            "dual_pca_dtype": "float64",
            "automatic_mixed_precision": False,
            "TF32": False,
            "torch_compile": False,
            "deterministic_algorithms": True,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "source_hashes": {
            "generator_script": sha256_file(GENERATOR_SCRIPT),
            "neural_syncfree_helper": sha256_file(NEURAL_IMPL_SCRIPT),
            "linear_metric_pca_helper": sha256_file(LINEAR_IMPL_SCRIPT),
            "local_final_gate_numeric_functions": local_numeric_source_hash,
        },
        "helper_adaptation": (
            "Set the frozen sync-free helper module STABILITY_RESAMPLES from its architecture value 20 to the "
            "preregistered final-gate value 100 before any pilot fit. The underlying stability_audit training, "
            "preprocessing, alignment, threshold and truth-matching implementation is unchanged."
        ),
        "metric_sufficiency_audit": {
            "reference_score_capture": (
                "Capture the exact NumPy full-data reference shared scores at the inherited reference_factor_truth_status "
                "boundary without changing the helper return value or adding a neural fit."
            ),
            "sealed_secondary_operating_metrics": [
                "subspace principal-angle/projection diagnostics",
                "view-support exact/Jaccard audit",
                "private-to-shared leakage audit",
                "assigned technical-truth audit",
            ],
            "additional_neural_optimization_runs": 0,
            "view_support_energy_threshold": float(upstream["neural"].VIEW_SUPPORT_ENERGY_THRESHOLD),
            "private_leakage_comparison": (
                "For each retained shared factor, compare its strongest absolute correlation to any view-specific private "
                "truth factor against its strongest biological-shared truth correlation."
            ),
        },
    }
    record["numerical_implementation_hash"] = sha256_text(canonical_json(record))
    return record


def scientific_contract_hash(
    upstream: dict[str, Any],
    epoch_bridge_hash: str,
    seed_registry_hash: str,
    numerical_contract_hash: str,
) -> str:
    record = {
        "05a": sha256_file(CONTRACT_05A),
        "05a1": sha256_file(CONTRACT_05A1),
        "05a2": sha256_file(AMENDMENT_05A2),
        "05d3": sha256_file(CONTRACT_05D3),
        "05d5_epoch_registry": sha256_file(EPOCH_REGISTRY_05D5),
        "05e": sha256_file(FREEZE_05E),
        "05f0_plan": sha256_file(PLAN_05F0),
        "05f0_pilot_registry": sha256_file(PILOT_REGISTRY_05F0),
        "final_gate_scenario_registry": sha256_file(FINAL_GATE_SCENARIOS),
        "final_gate_replicate_registry": sha256_file(FINAL_GATE_REPLICATES),
        "epoch_bridge": epoch_bridge_hash,
        "full_seed_registry": seed_registry_hash,
        "numerical_contract": numerical_contract_hash,
        "method_id": EXPECTED_METHOD_ID,
        "profile_id": EXPECTED_PROFILE_ID,
        "n_datasets": EXPECTED_DATASETS,
        "n_stability_resamples": EXPECTED_STABILITY_RESAMPLES,
    }
    return sha256_text(canonical_json(record))


def initialize_or_verify_prepilot_freeze(upstream: dict[str, Any]) -> dict[str, Any]:
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PRELOCK.exists():
        prelock = load_json(OUTPUT_PRELOCK)
        if clean_text(prelock.get("status")) != PRELOCK_STATUS:
            raise RuntimeError("Existing 05f1 prelock has unexpected status.")
        if clean_text(prelock.get("script_sha256")) != sha256_file(SCRIPT_PATH):
            raise RuntimeError("Current 05f1 script differs from the pre-pilot frozen script hash.")
        for key in ["epoch_bridge", "seed_registry", "numerical_contract"]:
            record = prelock.get(key, {})
            path = resolve_project_path(record.get("path", ""))
            require_file(path)
            if sha256_file(path) != clean_text(record.get("sha256")):
                raise RuntimeError(f"Existing 05f1 prelock {key} hash mismatch.")
        return prelock

    # If any reusable final-gate shard exists without a prelock, provenance is
    # incomplete; never silently adopt it.
    existing_shards = list(SHARD_DIR.glob("shard_*.json"))
    if existing_shards:
        raise RuntimeError("Final-gate result shards exist but 05f1 prelock is absent; inspect provenance before continuing.")

    epoch_bridge = build_fixed_epoch_bridge(upstream["profile"])
    write_json_atomic(epoch_bridge, OUTPUT_EPOCH_BRIDGE)

    seed_registry = build_full_stability_seed_registry(upstream)
    write_csv_gzip_atomic(seed_registry, OUTPUT_SEED_REGISTRY)
    # Replay exact row count after serialization.
    serialized_seed_registry = pd.read_csv(OUTPUT_SEED_REGISTRY, dtype=str, low_memory=False).fillna("")
    if serialized_seed_registry.shape[0] != EXPECTED_DATASETS * EXPECTED_STABILITY_RESAMPLES:
        raise RuntimeError("Serialized final-gate seed registry row count mismatch.")

    selected_epoch = int(epoch_bridge["final_gate_fixed_epoch"])
    numerical = numerical_implementation_record(
        upstream,
        selected_epoch=selected_epoch,
        seed_registry_sha256=sha256_file(OUTPUT_SEED_REGISTRY),
    )
    write_json_atomic(numerical, OUTPUT_NUMERICAL_CONTRACT)

    sci_hash = scientific_contract_hash(
        upstream,
        epoch_bridge_hash=sha256_file(OUTPUT_EPOCH_BRIDGE),
        seed_registry_hash=sha256_file(OUTPUT_SEED_REGISTRY),
        numerical_contract_hash=sha256_file(OUTPUT_NUMERICAL_CONTRACT),
    )

    prelock = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "script_path": relative_path(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "created_at_utc": utc_now_iso(),
        "status": PRELOCK_STATUS,
        "scientific_contract_hash": sci_hash,
        "05a2_amendment_hash": sha256_file(AMENDMENT_05A2),
        "numerical_implementation_hash": numerical["numerical_implementation_hash"],
        "epoch_bridge": file_record(OUTPUT_EPOCH_BRIDGE),
        "seed_registry": file_record(OUTPUT_SEED_REGISTRY),
        "numerical_contract": file_record(OUTPUT_NUMERICAL_CONTRACT),
        "runtime_environment": {
            "PyTorch_version": torch.__version__,
            "CUDA_build": torch.version.cuda,
            "GPU_model": torch.cuda.get_device_name(0),
        },
        "scientific_state_before_first_pilot_fit": {
            "pilot_final_gate_datasets_generated": 0,
            "pilot_model_fits_performed": 0,
            "final_gate_scientific_metrics_printed": False,
            "final_gate_scientific_metrics_aggregated": False,
            "TARGET_accessed": False,
            "outcomes_accessed": False,
            "biology_annotations_accessed": False,
            "threshold_retuning_performed": False,
        },
        "reuse_rule": upstream["runtime_policy"].get("reuse_rule"),
        "invalidation_rule": upstream["runtime_policy"].get("invalidation_rule"),
        "required_shard_provenance": upstream["runtime_policy"].get("required_shard_provenance", []),
        "rewrite_policy": (
            "This prelock is written before the first final-gate pilot fit and is immutable. Any numerical-path change "
            "requires a new versioned pilot and invalidates all shards created under this numerical implementation hash."
        ),
    }
    write_json_atomic(prelock, OUTPUT_PRELOCK)
    return prelock


# -----------------------------------------------------------------------------
# Streaming final-gate dataset generation / shard execution
# -----------------------------------------------------------------------------

def dataset_metadata(
    scenario: Any,
    replicate_row: pd.Series,
    validation: dict[str, Any],
    input_sha256: str,
) -> dict[str, Any]:
    return {
        "registry_label": clean_text(replicate_row.get("registry_label")),
        "registry_row_index": int(replicate_row["registry_row_index"]),
        "dataset_key": clean_text(replicate_row["dataset_key"]),
        "simulation_stage": EXPECTED_FINAL_STAGE,
        "materialization_policy": EXPECTED_FINAL_MATERIALIZATION_POLICY,
        "scenario_id": scenario.scenario_id,
        "scenario_origin": scenario.scenario_origin,
        "scenario_family": scenario.scenario_family,
        "calibration_role": scenario.calibration_role,
        "replicate_index": int(replicate_row["replicate_index"]),
        "simulation_seed": int(replicate_row["simulation_seed"]),
        "n_samples": int(scenario.n_samples),
        "n_true_biological_shared_factors": int(scenario.true_biological_shared_factors),
        "n_true_private_factors_per_view": int(scenario.n_private_factors_per_view),
        "n_true_technical_shared_factors": int(scenario.true_technical_shared_factors),
        "factor_level_identifiable": bool(scenario.factor_level_identifiable),
        "subspace_level_identifiable": bool(scenario.subspace_level_identifiable),
        "primary_truth_target": scenario.primary_truth_target,
        "input_mode": "deterministic_in_memory_stream",
        "input_path": "",
        "input_sha256": input_sha256,
        "scenario_sha256": validation["scenario_sha256"],
    }


def final_gate_shard_path(
    metadata: dict[str, Any],
    selected_epoch: int,
    prelock: dict[str, Any],
) -> Path:
    key = "|".join(
        [
            metadata["dataset_key"],
            metadata["input_sha256"],
            EXPECTED_METHOD_ID,
            EXPECTED_PROFILE_ID,
            f"E{int(selected_epoch)}",
            clean_text(prelock["scientific_contract_hash"]),
            clean_text(prelock["numerical_implementation_hash"]),
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    return SHARD_DIR / f"shard_{digest}.json"


def validate_existing_final_gate_shard(
    shard: dict[str, Any],
    metadata: dict[str, Any],
    selected_epoch: int,
    prelock: dict[str, Any],
) -> None:
    checks = {
        "dataset_key": (clean_text(shard.get("dataset_key")), metadata["dataset_key"]),
        "scenario_id": (clean_text(shard.get("scenario_id")), metadata["scenario_id"]),
        "replicate_index": (int(shard.get("replicate_index", -1)), int(metadata["replicate_index"])),
        "simulation_seed": (int(shard.get("simulation_seed", -1)), int(metadata["simulation_seed"])),
        "input_sha256": (clean_text(shard.get("input_sha256")), metadata["input_sha256"]),
        "variant_id": (clean_text(shard.get("variant_id")), EXPECTED_METHOD_ID),
        "profile_id": (clean_text(shard.get("profile_id")), EXPECTED_PROFILE_ID),
        "selected_epoch": (int(shard.get("selected_epoch", -1)), int(selected_epoch)),
        "scientific_contract_hash": (
            clean_text(shard.get("scientific_contract_hash")),
            clean_text(prelock["scientific_contract_hash"]),
        ),
        "05a2_amendment_hash": (
            clean_text(shard.get("05a2_amendment_hash")),
            clean_text(prelock["05a2_amendment_hash"]),
        ),
        "numerical_implementation_hash": (
            clean_text(shard.get("numerical_implementation_hash")),
            clean_text(prelock["numerical_implementation_hash"]),
        ),
        "PyTorch_version": (clean_text(shard.get("PyTorch_version")), clean_text(torch.__version__)),
        "CUDA_build": (clean_text(shard.get("CUDA_build")), clean_text(torch.version.cuda)),
        "GPU_model": (clean_text(shard.get("GPU_model")), clean_text(torch.cuda.get_device_name(0))),
        "compute_status": (clean_text(shard.get("compute_status")), "COMPLETE"),
    }
    failures = [name for name, (observed, expected) in checks.items() if observed != expected]
    if failures:
        raise RuntimeError("Existing final-gate shard provenance mismatch: " + ", ".join(failures))
    payload = shard.get("scientific_payload", {})
    if not isinstance(payload, dict):
        raise RuntimeError("Existing final-gate shard lacks scientific payload object.")
    if int(payload.get("dataset_row", {}).get("n_stability_resamples", -1)) != EXPECTED_STABILITY_RESAMPLES:
        raise RuntimeError("Existing final-gate shard does not contain 100 stability resamples.")
    if len(payload.get("fit_audits", [])) != 1 + EXPECTED_STABILITY_RESAMPLES:
        raise RuntimeError("Existing final-gate shard fit-audit count mismatch.")
    if len(payload.get("preprocessing_rows", [])) != 3 + 3 * EXPECTED_STABILITY_RESAMPLES:
        raise RuntimeError("Existing final-gate shard preprocessing-audit count mismatch.")


def execute_or_verify_dataset_shard(
    upstream: dict[str, Any],
    prelock: dict[str, Any],
    seed_registry: pd.DataFrame,
    replicate_row: pd.Series,
    selected_epoch: int,
) -> tuple[dict[str, Any], str, float, int]:
    generator = upstream["generator"]
    neural = upstream["neural"]
    linear = upstream["linear"]
    profile = upstream["profile"]
    thresholds = upstream["stability_thresholds"]

    scenario_id = clean_text(replicate_row["scenario_id"])
    scenario = upstream["scenarios"][scenario_id]
    registry_label = clean_text(replicate_row["registry_label"])
    replicate_index = int(replicate_row["replicate_index"])
    simulation_seed = int(replicate_row["simulation_seed"])

    # Deterministic streamed input generation is intentionally repeated on resume;
    # no permanent final-gate input bank is created.
    arrays = generator.generate_dataset(
        scenario,
        registry_label,
        replicate_index,
        simulation_seed,
    )
    validation = validate_final_gate_arrays(
        generator,
        arrays,
        scenario,
        registry_label,
        replicate_index,
        simulation_seed,
    )
    input_hash = array_bundle_sha256(generator, arrays)
    metadata = dataset_metadata(scenario, replicate_row, validation, input_hash)
    candidate = final_gate_shard_path(metadata, selected_epoch, prelock)

    if candidate.exists():
        shard = load_json(candidate)
        validate_existing_final_gate_shard(shard, metadata, selected_epoch, prelock)
        runtime = shard.get("runtime", {})
        wall_seconds = float(runtime.get("wall_seconds_full_dataset", 0.0))
        peak_bytes = int(runtime.get("peak_cuda_memory_allocated_bytes", 0))
        del arrays
        return shard, "verified_existing", wall_seconds, peak_bytes

    dataset_seed_rows = seed_registry[seed_registry["dataset_key"].eq(metadata["dataset_key"])].copy()
    stability_group = build_stability_group(dataset_seed_rows, int(scenario.n_samples))

    # Runtime synchronization is a timing wrapper only; the numerical functions
    # hashed above are unchanged by this measurement boundary.
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize(0)
    started = time.perf_counter()

    payload = evaluate_final_gate_dataset(
        neural=neural,
        linear=linear,
        arrays=arrays,
        metadata=metadata,
        profile=profile,
        selected_epoch=selected_epoch,
        stability_group=stability_group,
        stability_thresholds=thresholds,
    )

    torch.cuda.synchronize(0)
    wall_seconds = time.perf_counter() - started
    peak_bytes = int(torch.cuda.max_memory_allocated(0))

    shard = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "script_hash": sha256_file(SCRIPT_PATH),
        "created_at_utc": utc_now_iso(),
        "compute_status": "COMPLETE",
        "dataset_key": metadata["dataset_key"],
        "scenario_id": metadata["scenario_id"],
        "replicate_index": metadata["replicate_index"],
        "simulation_seed": metadata["simulation_seed"],
        "variant_id": EXPECTED_METHOD_ID,
        "profile_id": EXPECTED_PROFILE_ID,
        "selected_epoch": int(selected_epoch),
        "input_sha256": input_hash,
        "input_mode": "deterministic_in_memory_stream",
        "scientific_contract_hash": prelock["scientific_contract_hash"],
        "05a2_amendment_hash": prelock["05a2_amendment_hash"],
        "numerical_implementation_hash": prelock["numerical_implementation_hash"],
        "epoch_bridge_sha256": sha256_file(OUTPUT_EPOCH_BRIDGE),
        "seed_registry_sha256": sha256_file(OUTPUT_SEED_REGISTRY),
        "PyTorch_version": torch.__version__,
        "CUDA_build": torch.version.cuda,
        "GPU_model": torch.cuda.get_device_name(0),
        "runtime": {
            "wall_seconds_full_dataset": float(wall_seconds),
            "peak_cuda_memory_allocated_bytes": peak_bytes,
            "optimization_runs": 1 + EXPECTED_STABILITY_RESAMPLES,
        },
        # Scientific values are sealed per-dataset for later all-3,100 aggregation.
        # 05f1 never prints or aggregates this payload.
        "scientific_payload": payload,
    }
    write_json_atomic(shard, candidate)
    serialized = load_json(candidate)
    validate_existing_final_gate_shard(serialized, metadata, selected_epoch, prelock)
    del arrays
    return serialized, "generated", float(wall_seconds), peak_bytes


# -----------------------------------------------------------------------------
# Runtime-only pilot completion / verification
# -----------------------------------------------------------------------------

def runtime_summary(manifest: pd.DataFrame) -> dict[str, Any]:
    gpu_wall = pd.to_numeric(
        manifest["wall_seconds_full_dataset"], errors="raise"
    ).to_numpy(dtype=float)
    end_to_end = pd.to_numeric(
        manifest["wrapper_wall_seconds"], errors="raise"
    ).to_numpy(dtype=float)
    peak = pd.to_numeric(
        manifest["peak_cuda_memory_allocated_bytes"], errors="raise"
    ).to_numpy(dtype=float)
    if (
        gpu_wall.size != EXPECTED_PILOT_DATASETS
        or end_to_end.size != EXPECTED_PILOT_DATASETS
        or not np.isfinite(gpu_wall).all()
        or not np.isfinite(end_to_end).all()
        or np.any(gpu_wall <= 0)
        or np.any(end_to_end <= 0)
    ):
        raise RuntimeError("Pilot runtime measurements are incomplete/non-positive.")

    total_gpu = float(np.sum(gpu_wall))
    total_end = float(np.sum(end_to_end))
    mean_gpu_dataset = float(np.mean(gpu_wall))
    mean_end_dataset = float(np.mean(end_to_end))
    median_end_dataset = float(np.median(end_to_end))
    mean_gpu_fit = total_gpu / float(
        EXPECTED_PILOT_DATASETS * (1 + EXPECTED_STABILITY_RESAMPLES)
    )
    projected_seconds = mean_end_dataset * EXPECTED_DATASETS

    return {
        "summary_version": "paper5-final-gate-runtime-pilot-summary-v1",
        "created_at_utc": utc_now_iso(),
        "status": "RUNTIME_ONLY_COMPLETE_NO_SCIENTIFIC_AGGREGATION",
        "pilot_dataset_count": EXPECTED_PILOT_DATASETS,
        "pilot_optimization_runs": EXPECTED_PILOT_DATASETS * (1 + EXPECTED_STABILITY_RESAMPLES),
        # Backward-simple names use end-to-end runtime because the full streaming
        # run includes deterministic generation/validation/shard wrapping too.
        "total_wall_seconds": total_end,
        "mean_wall_seconds_per_dataset": mean_end_dataset,
        "median_wall_seconds_per_dataset": median_end_dataset,
        "total_gpu_path_wall_seconds": total_gpu,
        "mean_gpu_path_wall_seconds_per_dataset": mean_gpu_dataset,
        "mean_gpu_path_wall_seconds_per_optimization_run": mean_gpu_fit,
        "peak_cuda_memory_allocated_bytes": int(np.max(peak)),
        "full_run_planned_datasets": EXPECTED_DATASETS,
        "full_run_planned_optimization_runs": EXPECTED_DATASETS * (1 + EXPECTED_STABILITY_RESAMPLES),
        "rough_full_run_extrapolated_seconds": projected_seconds,
        "rough_full_run_extrapolated_hours": projected_seconds / 3600.0,
        "rough_full_run_extrapolated_days": projected_seconds / 86400.0,
        "extrapolation_basis": "mean end-to-end streamed dataset wrapper time",
        "extrapolation_role": (
            "runtime planning only; first-three frozen pilot datasets are not selected for runtime representativeness"
        ),
        "scientific_metrics_printed": False,
        "scientific_metrics_aggregated": False,
        "D01_D02_D03_evaluated": False,
        "final_gate_PASS_FAIL_evaluated": False,
        "TARGET_accessed": False,
        "outcomes_accessed": False,
        "threshold_retuning_performed": False,
    }


def build_readme(summary: dict[str, Any], prelock: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Paper 5 - 05f1 final-gate GPU runtime pilot",
            "=" * 76,
            "",
            f"Script version: {SCRIPT_VERSION}",
            f"Numerical implementation hash: {prelock['numerical_implementation_hash']}",
            "",
            "Scope",
            "-----",
            "The pilot executes the first three dataset IDs frozen by 05f0.",
            "Each dataset traverses the complete final-gate path: one full-data V4 reference fit",
            "plus all 100 frozen 80% stability resample fits.",
            "Final-gate simulation inputs are generated deterministically in memory and not persisted as an input bank.",
            "",
            "Scientific visibility",
            "---------------------",
            "Per-dataset scientific payloads are sealed in reusable result shards but are not printed or aggregated by 05f1.",
            "No D01/D02/D03, TPR, FNR, FDP, familywise false acceptance, subgroup result, or scientific PASS/FAIL is evaluated here.",
            "",
            "Numerical freeze",
            "----------------",
            "The exact generator, neural helper, linear/PCA helper, fixed-epoch bridge, full 310,000-row",
            "stability seed registry, profile, thresholds, dtype/runtime settings, and local final-gate numerical wrappers",
            "are hash-frozen before the first pilot fit.",
            "Pilot shards are reusable only under the unchanged numerical implementation hash.",
            "",
            "Runtime-only summary",
            "--------------------",
            f"Pilot datasets: {summary['pilot_dataset_count']}",
            f"Pilot optimization runs: {summary['pilot_optimization_runs']}",
            f"Mean wall seconds/dataset: {summary['mean_wall_seconds_per_dataset']:.3f}",
            f"Peak CUDA allocation: {summary['peak_cuda_memory_allocated_bytes'] / (1024**2):.1f} MiB",
            f"Rough full-run extrapolation: {summary['rough_full_run_extrapolated_days']:.2f} days",
            "",
            "Next",
            "----",
            "If runtime is acceptable, run the full final-gate streaming evaluation using this exact numerical implementation hash.",
            "If runtime-only optimization is needed, create a new versioned numerical path and recompute all pilot shards.",
            "",
        ]
    )


def verify_existing_pilot_lock() -> None:
    lock = load_json(OUTPUT_LOCK)
    if clean_text(lock.get("status")) != PILOT_LOCK_STATUS:
        raise RuntimeError("Existing 05f1 pilot lock status mismatch.")
    if clean_text(lock.get("script_sha256")) != sha256_file(SCRIPT_PATH):
        raise RuntimeError("Current 05f1 script differs from frozen pilot script hash.")
    verify_generated_files(lock, "05f1 runtime pilot")

    manifest = pd.read_csv(OUTPUT_PILOT_MANIFEST, dtype=str, low_memory=False).fillna("")
    if manifest.shape[0] != EXPECTED_PILOT_DATASETS:
        raise RuntimeError("Existing 05f1 pilot manifest row count mismatch.")
    for _, row in manifest.iterrows():
        shard_path = resolve_project_path(row["shard_path"])
        require_file(shard_path)
        if int(shard_path.stat().st_size) != int(float(row["shard_size_bytes"])):
            raise RuntimeError("Existing pilot shard size mismatch.")
        if sha256_file(shard_path) != clean_text(row["shard_sha256"]):
            raise RuntimeError("Existing pilot shard hash mismatch.")

    summary = load_json(OUTPUT_PILOT_SUMMARY)
    print("=" * 88)
    print("Existing Paper 5 05f1 final-gate runtime pilot: VERIFIED")
    print("=" * 88)
    print(f"Pilot datasets: {EXPECTED_PILOT_DATASETS}")
    print(f"Pilot optimization runs: {EXPECTED_PILOT_DATASETS * (1 + EXPECTED_STABILITY_RESAMPLES):,}")
    print(f"Mean wall time/dataset: {float(summary['mean_wall_seconds_per_dataset']):.1f}s")
    print(f"Peak CUDA allocation: {int(summary['peak_cuda_memory_allocated_bytes']) / (1024**2):.1f} MiB")
    print(f"Rough full-run timing extrapolation: {float(summary['rough_full_run_extrapolated_days']):.2f} days")
    print("Scientific metrics printed/aggregated: NO")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("No files were rewritten.")
    print("=" * 88)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    print("=" * 88)
    print("Paper 5 - GPU final-gate runtime pilot")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Pilot datasets: frozen first 3 final-gate registry rows")
    print("Stability resamples per pilot dataset: 100")
    print("Permanent final-gate input bank: FORBIDDEN")
    print("Scientific gate output during pilot: FORBIDDEN")
    print("TARGET/outcomes/biology annotations: FORBIDDEN")
    print("Threshold retuning: FORBIDDEN")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_LOCK.exists():
        verify_existing_pilot_lock()
        return

    upstream = verify_upstream_and_load()
    prelock = initialize_or_verify_prepilot_freeze(upstream)

    # Freeze is now durable before the first pilot fit. Only after this line may
    # the inherited helper be configured for the preregistered 100 resamples.
    configure_final_gate_neural_helper(upstream["neural"])

    epoch_bridge = load_json(OUTPUT_EPOCH_BRIDGE)
    selected_epoch = int(epoch_bridge["final_gate_fixed_epoch"])
    numerical = load_json(OUTPUT_NUMERICAL_CONTRACT)
    if clean_text(numerical.get("numerical_implementation_hash")) != clean_text(prelock.get("numerical_implementation_hash")):
        raise RuntimeError("05f1 numerical contract/prelock hash mismatch.")

    seed_registry = pd.read_csv(OUTPUT_SEED_REGISTRY, dtype=str, low_memory=False).fillna("")
    if seed_registry.shape[0] != EXPECTED_DATASETS * EXPECTED_STABILITY_RESAMPLES:
        raise RuntimeError("Frozen final-gate stability seed registry row count mismatch.")

    # Runtime fixture after numerical prelock, before pilot datasets. The fixture
    # tests inherited CUDA determinism/alignment only and exposes no final-gate result.
    if OUTPUT_FIXTURE.exists():
        fixture = load_json(OUTPUT_FIXTURE)
        if clean_text(fixture.get("numerical_implementation_hash")) != clean_text(prelock["numerical_implementation_hash"]):
            raise RuntimeError("Existing 05f1 runtime fixture numerical hash mismatch.")
    else:
        neural_fixture = upstream["neural"].run_reference_fixture(upstream["profile"])
        if clean_text(neural_fixture.get("status")) != "PASS":
            raise RuntimeError("Inherited V4 CUDA deterministic fixture failed before pilot.")
        rng = np.random.default_rng(20551)
        reference = rng.normal(size=(48, int(upstream["profile"].shared_dim))).astype(np.float64)
        permutation = np.arange(int(upstream["profile"].shared_dim))[::-1]
        signs = np.where(np.arange(int(upstream["profile"].shared_dim)) % 2 == 0, 1.0, -1.0)
        resample = reference[:, permutation] * signs
        _, correlations, subspace = upstream["neural"].align_resample_to_reference(
            upstream["linear"], resample, reference
        )
        if subspace is None or float(np.min(correlations)) < 1.0 - 1e-10 or float(subspace) < 1.0 - 1e-10:
            raise RuntimeError("Inherited Procrustes/Hungarian alignment fixture failed before pilot.")
        fixture = {
            "fixture_version": "paper5-05f1-final-gate-runtime-fixture-v1",
            "created_at_utc": utc_now_iso(),
            "numerical_implementation_hash": prelock["numerical_implementation_hash"],
            "inherited_neural_fixture_status": "PASS",
            "alignment_fixture_status": "PASS",
            "PyTorch_version": torch.__version__,
            "CUDA_build": torch.version.cuda,
            "GPU_model": torch.cuda.get_device_name(0),
            "final_gate_scientific_data_used": False,
        }
        write_json_atomic(fixture, OUTPUT_FIXTURE)

    print("05f0 streaming plan: VERIFIED")
    print("05a2 runtime-pilot policy: VERIFIED")
    print("05e V4/profile freeze: VERIFIED")
    print("05b generator deterministic replay: VERIFIED")
    print("05d0/05d1/05d3/05d5 numerical lineage: VERIFIED")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Numerical implementation prelock: FROZEN BEFORE FIRST PILOT FIT")
    print("Full 3,100-dataset stability seed registry: FROZEN BEFORE FIRST PILOT FIT")
    print("Final-gate fixed-epoch bridge: FROZEN BEFORE FIRST PILOT FIT")
    print("Scientific metrics below this point will not be printed or aggregated.")
    print()

    write_json_atomic(
        {
            "script_version": SCRIPT_VERSION,
            "started_at_utc": utc_now_iso(),
            "status": "IN_PROGRESS",
            "pilot_dataset_count": EXPECTED_PILOT_DATASETS,
            "pilot_optimization_runs": EXPECTED_PILOT_DATASETS * (1 + EXPECTED_STABILITY_RESAMPLES),
            "scientific_contract_hash": prelock["scientific_contract_hash"],
            "numerical_implementation_hash": prelock["numerical_implementation_hash"],
            "scientific_console_output": False,
        },
        OUTPUT_IN_PROGRESS,
    )

    pilot = upstream["pilot"].copy()
    # Numeric columns are serialized as strings by 05f0; recover exact values.
    for column in ["registry_row_index", "replicate_index", "simulation_seed"]:
        pilot[column] = pd.to_numeric(pilot[column], errors="raise").astype(np.int64)

    manifest_rows: list[dict[str, Any]] = []
    generated_runtime_seconds = 0.0
    generated_count = 0

    try:
        for ordinal, (_, row) in enumerate(pilot.iterrows(), start=1):
            dataset_key = clean_text(row["dataset_key"])
            started_wrapper = time.perf_counter()
            shard, generation_status, wall_seconds, peak_bytes = execute_or_verify_dataset_shard(
                upstream=upstream,
                prelock=prelock,
                seed_registry=seed_registry,
                replicate_row=row,
                selected_epoch=selected_epoch,
            )
            wrapper_seconds = time.perf_counter() - started_wrapper

            candidate = final_gate_shard_path(
                shard["scientific_payload"]["dataset_row"],
                selected_epoch,
                prelock,
            )
            # final_gate_shard_path only requires dataset_key/input_sha256 fields;
            # top-level shard is authoritative for validation.
            if not candidate.exists():
                # Recompute path directly from the saved shard top-level metadata.
                pseudo_metadata = {
                    "dataset_key": shard["dataset_key"],
                    "input_sha256": shard["input_sha256"],
                }
                candidate = final_gate_shard_path(pseudo_metadata, selected_epoch, prelock)
            require_file(candidate)

            if generation_status == "generated":
                generated_count += 1
                generated_runtime_seconds += float(wall_seconds)

            manifest_rows.append(
                {
                    "pilot_ordinal": ordinal,
                    "dataset_key": dataset_key,
                    "generation_status": generation_status,
                    "optimization_runs": 1 + EXPECTED_STABILITY_RESAMPLES,
                    "wall_seconds_full_dataset": float(wall_seconds),
                    "wrapper_wall_seconds": float(wrapper_seconds),
                    "peak_cuda_memory_allocated_bytes": int(peak_bytes),
                    "shard_path": relative_path(candidate),
                    "shard_size_bytes": int(candidate.stat().st_size),
                    "shard_sha256": sha256_file(candidate),
                    "scientific_metrics_printed": False,
                    "scientific_metrics_aggregated": False,
                }
            )

            # Runtime/progress output only, exactly as frozen by 05a2/05f0.
            if generated_count > 0:
                mean_runtime = generated_runtime_seconds / generated_count
                remaining = EXPECTED_PILOT_DATASETS - ordinal
                eta_minutes = remaining * mean_runtime / 60.0
                eta_text = f" | ETA~{eta_minutes:.1f}m"
            else:
                eta_text = ""
            print(
                f"{ordinal:02d}/{EXPECTED_PILOT_DATASETS:02d} "
                f"runtime={wall_seconds:.1f}s | "
                f"VRAM={peak_bytes / (1024**2):.1f} MiB | "
                f"shard={candidate.stat().st_size / (1024**2):.2f} MiB | "
                f"{generation_status}{eta_text}"
            )

            torch.cuda.empty_cache()
            gc.collect()

    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        failure = {
            "script_version": SCRIPT_VERSION,
            "created_at_utc": utc_now_iso(),
            "status": "RUNTIME_IMPLEMENTATION_FAILURE_NOT_SCIENTIFIC_GATE_RESULT",
            "error_type": type(exc).__name__,
            "error_message": clean_text(str(exc))[:4000],
            "traceback_tail": "\n".join(traceback.format_exc().splitlines()[-40:]),
            "scientific_metrics_printed": False,
            "scientific_metrics_aggregated": False,
            "threshold_retuning_performed": False,
            "numerical_implementation_hash": prelock["numerical_implementation_hash"],
        }
        write_json_atomic(failure, OUTPUT_FAILURE)
        raise

    manifest = pd.DataFrame(manifest_rows)
    if manifest.shape[0] != EXPECTED_PILOT_DATASETS:
        raise RuntimeError("05f1 runtime pilot manifest does not contain exactly 3 rows.")
    write_csv_atomic(manifest, OUTPUT_PILOT_MANIFEST)

    summary = runtime_summary(manifest)
    write_json_atomic(summary, OUTPUT_PILOT_SUMMARY)
    write_text_atomic(build_readme(summary, prelock), OUTPUT_README)

    generated_files = [
        OUTPUT_EPOCH_BRIDGE,
        OUTPUT_SEED_REGISTRY,
        OUTPUT_NUMERICAL_CONTRACT,
        OUTPUT_PRELOCK,
        OUTPUT_FIXTURE,
        OUTPUT_PILOT_MANIFEST,
        OUTPUT_PILOT_SUMMARY,
        OUTPUT_README,
    ]

    lock = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "script_path": relative_path(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "created_at_utc": utc_now_iso(),
        "status": PILOT_LOCK_STATUS,
        "scientific_contract_hash": prelock["scientific_contract_hash"],
        "05a2_amendment_hash": prelock["05a2_amendment_hash"],
        "numerical_implementation_hash": prelock["numerical_implementation_hash"],
        "pilot_dataset_count": EXPECTED_PILOT_DATASETS,
        "pilot_optimization_runs": EXPECTED_PILOT_DATASETS * (1 + EXPECTED_STABILITY_RESAMPLES),
        "scientific_state": {
            "final_gate_pilot_datasets_streamed": EXPECTED_PILOT_DATASETS,
            "final_gate_pilot_model_fits_performed": EXPECTED_PILOT_DATASETS * (1 + EXPECTED_STABILITY_RESAMPLES),
            "per_dataset_scientific_payloads_sealed_in_shards": True,
            "scientific_metrics_printed": False,
            "scientific_metrics_aggregated": False,
            "D01_D02_D03_evaluated": False,
            "final_gate_PASS_FAIL_evaluated": False,
            "TARGET_accessed": False,
            "outcomes_accessed": False,
            "biology_annotations_accessed": False,
            "threshold_retuning_performed": False,
        },
        "runtime_summary": file_record(OUTPUT_PILOT_SUMMARY),
        "pilot_manifest": file_record(OUTPUT_PILOT_MANIFEST),
        "generated_files": {relative_path(path): file_record(path) for path in generated_files},
        "pilot_shards": {
            clean_text(row["dataset_key"]): {
                "path": clean_text(row["shard_path"]),
                "sha256": clean_text(row["shard_sha256"]),
                "size_bytes": int(float(row["shard_size_bytes"])),
            }
            for _, row in manifest.iterrows()
        },
        "reuse_rule": upstream["runtime_policy"].get("reuse_rule"),
        "invalidation_rule": upstream["runtime_policy"].get("invalidation_rule"),
        "rewrite_policy": (
            "05f1 runtime-pilot outputs are immutable. Pilot shards may enter the full final-gate run only if the "
            "numerical implementation hash is exactly unchanged and every shard provenance/hash validation passes."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    if OUTPUT_IN_PROGRESS.exists():
        OUTPUT_IN_PROGRESS.unlink()
    if OUTPUT_FAILURE.exists():
        OUTPUT_FAILURE.unlink()

    print()
    print("-" * 88)
    print("Final-gate runtime pilot summary (runtime only)")
    print("-" * 88)
    print(f"Pilot datasets completed: {EXPECTED_PILOT_DATASETS}")
    print(f"Pilot optimization runs: {EXPECTED_PILOT_DATASETS * (1 + EXPECTED_STABILITY_RESAMPLES):,}")
    print(f"Total measured end-to-end pilot time: {summary['total_wall_seconds'] / 60.0:.1f} min")
    print(f"Mean wall time per dataset: {summary['mean_wall_seconds_per_dataset']:.1f}s")
    print(f"Peak CUDA allocation: {summary['peak_cuda_memory_allocated_bytes'] / (1024**2):.1f} MiB")
    print(f"Rough full-run timing extrapolation: {summary['rough_full_run_extrapolated_days']:.2f} days")
    print("Scientific metrics printed: NO")
    print("Scientific metrics aggregated: NO")
    print("Final-gate PASS/FAIL evaluated: NO")
    print("Pilot shards reusable if numerical implementation hash is unchanged: YES")
    print()
    print("=" * 88)
    print("Paper 5 05f1 final-gate GPU runtime pilot: COMPLETE")
    print("=" * 88)
    print(f"Numerical prelock: {relative_path(OUTPUT_PRELOCK)}")
    print(f"Runtime summary: {relative_path(OUTPUT_PILOT_SUMMARY)}")
    print(f"Pilot manifest: {relative_path(OUTPUT_PILOT_MANIFEST)}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("Next: review runtime only, then full final-gate streaming evaluation under the unchanged numerical hash.")
    print("=" * 88)


if __name__ == "__main__":
    main()
