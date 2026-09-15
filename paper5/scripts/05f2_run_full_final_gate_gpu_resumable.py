from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Preserve the frozen deterministic CUDA configuration before importing torch.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pandas as pd
import torch


SCRIPT_VERSION = "05f2-run-full-final-gate-gpu-resumable-v1-sequential-05f1-path-no-cli"
CONTRACT_VERSION = "paper5-05f2-full-final-gate-streaming-run-v1"

EXPECTED_DATASETS = 3_100
EXPECTED_SCENARIOS = 24
EXPECTED_STABILITY_RESAMPLES = 100
EXPECTED_OPTIMIZATION_RUNS = EXPECTED_DATASETS * (1 + EXPECTED_STABILITY_RESAMPLES)
EXPECTED_REUSED_PILOT_DATASETS = 3

EXPECTED_05F1_SCRIPT_VERSION = (
    "05f1-run-final-gate-runtime-pilot-gpu-v2-seed-collision-aware-no-cli"
)
EXPECTED_05F1_LOCK_STATUS = "FROZEN_FINAL_GATE_RUNTIME_PILOT_COMPLETE_BEFORE_FULL_RUN"
EXPECTED_05F1D_SCRIPT_VERSION = "05f1d-validate-two-stream-full-path-gpu-v1-no-cli"
EXPECTED_05F1D_LOCK_STATUS = (
    "FROZEN_TWO_STREAM_FULL_PATH_PILOT_VALIDATED_BEFORE_FULL_FINAL_GATE"
)

EXECUTION_PRELOCK_STATUS = (
    "FROZEN_RUNTIME_ONLY_SELECTION_OF_ORIGINAL_05F1_SEQUENTIAL_PATH_BEFORE_FULL_RUN"
)
COMPLETION_LOCK_STATUS = (
    "FROZEN_ALL_3100_FINAL_GATE_SHARDS_COMPLETE_BEFORE_SCIENTIFIC_AGGREGATION"
)

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

RUNNER_PATH = PAPER5_ROOT / "scripts" / "05f1_run_final_gate_runtime_pilot_gpu_v2.py"
TWO_STREAM_PATH = PAPER5_ROOT / "scripts" / "05f1d_validate_two_stream_full_path_gpu.py"

DIR_05F1 = PAPER5_ROOT / "method_contract" / "05f1"
DIR_05F1D = PAPER5_ROOT / "method_contract" / "05f1d"

TWO_STREAM_SUMMARY = DIR_05F1D / "two_stream_full_path_runtime_summary.json"
TWO_STREAM_LOCK = DIR_05F1D / "two_stream_full_path_validation_lock.json"

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05f2"
OUTPUT_EXECUTION_PRELOCK = OUTPUT_DIR / "full_run_execution_choice_prelock.json"
OUTPUT_PROGRESS_MANIFEST = OUTPUT_DIR / "full_run_progress_manifest.csv"
OUTPUT_PROGRESS_STATE = OUTPUT_DIR / "full_run_progress_state.json"
OUTPUT_COMPLETION_SUMMARY = OUTPUT_DIR / "full_run_completion_summary.json"
OUTPUT_README = OUTPUT_DIR / "05f2_README.txt"
OUTPUT_LOCK = OUTPUT_DIR / "full_run_completion_lock.json"
OUTPUT_FAILURE = OUTPUT_DIR / "full_run_failure.json"

# The numerical result shards stay in the original frozen 05f1 compact-shard
# directory because 05f2 deliberately reuses the exact 05f1 numerical path/hash.
# This is what makes the first three pilot shards reusable without rewriting them.


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
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
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


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
            )
            + "\n",
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


def write_csv_atomic(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tmp_", suffix=".csv", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        table.to_csv(
            temporary,
            index=False,
            float_format="%.12g",
            lineterminator="\n",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def resolve_project_path(value: Any) -> Path:
    path = Path(clean_text(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def import_module(path: Path, module_name: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required script not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


# -----------------------------------------------------------------------------
# Frozen runtime-only execution choice
# -----------------------------------------------------------------------------

def verify_05f1_and_05f1d_runtime_evidence(runner: Any) -> dict[str, Any]:
    if clean_text(getattr(runner, "SCRIPT_VERSION", "")) != EXPECTED_05F1_SCRIPT_VERSION:
        raise RuntimeError("Unexpected 05f1 runner version.")
    if not runner.OUTPUT_LOCK.is_file():
        raise RuntimeError("Completed 05f1 pilot lock is required.")

    old_lock = runner.load_json(runner.OUTPUT_LOCK)
    if clean_text(old_lock.get("status")) != EXPECTED_05F1_LOCK_STATUS:
        raise RuntimeError("05f1 pilot lock has unexpected status.")
    if clean_text(old_lock.get("script_sha256")) != sha256_file(RUNNER_PATH):
        raise RuntimeError("Current 05f1 runner differs from frozen pilot script hash.")

    old_summary_record = old_lock.get("runtime_summary", {})
    old_summary_path = resolve_project_path(old_summary_record.get("path"))
    if old_summary_path.resolve() != runner.OUTPUT_PILOT_SUMMARY.resolve():
        raise RuntimeError("05f1 lock points to unexpected runtime summary.")
    if sha256_file(old_summary_path) != clean_text(old_summary_record.get("sha256")):
        raise RuntimeError("05f1 runtime summary hash mismatch.")
    old_summary = load_json(old_summary_path)
    if parse_bool(old_summary.get("scientific_metrics_aggregated")):
        raise RuntimeError("05f1 runtime summary unexpectedly aggregated scientific metrics.")
    if parse_bool(old_summary.get("final_gate_PASS_FAIL_evaluated")):
        raise RuntimeError("05f1 runtime summary unexpectedly evaluated PASS/FAIL.")

    if not TWO_STREAM_PATH.is_file():
        raise FileNotFoundError("05f1d script is required for runtime-only path comparison.")
    if not TWO_STREAM_LOCK.is_file() or not TWO_STREAM_SUMMARY.is_file():
        raise FileNotFoundError("Completed 05f1d lock/summary is required.")

    stream_lock = load_json(TWO_STREAM_LOCK)
    if clean_text(stream_lock.get("status")) != EXPECTED_05F1D_LOCK_STATUS:
        raise RuntimeError("05f1d validation lock has unexpected status.")
    if clean_text(stream_lock.get("script_version")) != EXPECTED_05F1D_SCRIPT_VERSION:
        raise RuntimeError("05f1d script version mismatch.")
    if clean_text(stream_lock.get("script_sha256")) != sha256_file(TWO_STREAM_PATH):
        raise RuntimeError("Current 05f1d script differs from frozen validation hash.")
    if not bool(stream_lock.get("blind_exact_equivalence_all_three", False)):
        raise RuntimeError("05f1d did not achieve exact equivalence on all three pilot datasets.")
    state = stream_lock.get("scientific_state", {})
    if parse_bool(state.get("scientific_metrics_aggregated")):
        raise RuntimeError("05f1d unexpectedly aggregated scientific metrics.")
    if parse_bool(state.get("final_gate_PASS_FAIL_evaluated")):
        raise RuntimeError("05f1d unexpectedly evaluated final gate PASS/FAIL.")

    stream_summary_record = stream_lock.get("runtime_summary", {})
    stream_summary_path = resolve_project_path(stream_summary_record.get("path"))
    if stream_summary_path.resolve() != TWO_STREAM_SUMMARY.resolve():
        raise RuntimeError("05f1d lock points to unexpected runtime summary.")
    if sha256_file(stream_summary_path) != clean_text(stream_summary_record.get("sha256")):
        raise RuntimeError("05f1d runtime summary hash mismatch.")
    stream_summary = load_json(stream_summary_path)

    original_mean = float(old_summary["mean_wall_seconds_per_dataset"])
    stream_mean = float(stream_summary["mean_wall_seconds_per_dataset"])
    if not (math.isfinite(original_mean) and original_mean > 0):
        raise RuntimeError("05f1 mean runtime is invalid.")
    if not (math.isfinite(stream_mean) and stream_mean > 0):
        raise RuntimeError("05f1d mean runtime is invalid.")

    # 05f2 is intentionally the original sequential numerical path. If the
    # completed three-dataset full-path evidence no longer supports that runtime
    # choice, refuse rather than silently changing execution semantics.
    if original_mean > stream_mean:
        raise RuntimeError(
            "05f1d two-stream full-path runtime is now faster than 05f1 sequential. "
            "This 05f2 script is specifically frozen to the original 05f1 path; "
            "use a newly versioned full-run orchestrator instead of changing this file."
        )

    return {
        "05f1_lock": old_lock,
        "05f1_summary": old_summary,
        "05f1_summary_path": old_summary_path,
        "05f1d_lock": stream_lock,
        "05f1d_summary": stream_summary,
        "05f1d_summary_path": stream_summary_path,
        "original_mean_wall_seconds_per_dataset": original_mean,
        "two_stream_mean_wall_seconds_per_dataset": stream_mean,
    }


def build_execution_choice_prelock(
    runner: Any,
    runtime_evidence: dict[str, Any],
) -> dict[str, Any]:
    old_prelock = runner.load_json(runner.OUTPUT_PRELOCK)
    numerical = runner.load_json(runner.OUTPUT_NUMERICAL_CONTRACT)
    old_lock = runtime_evidence["05f1_lock"]

    record = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "script_path": relative_path(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "created_at_utc": utc_now_iso(),
        "status": EXECUTION_PRELOCK_STATUS,
        "selection_scope": "runtime_execution_only_no_scientific_metric_used",
        "selected_execution_path": {
            "name": "original_05f1_sequential_final_gate_numerical_path",
            "runner_path": relative_path(RUNNER_PATH),
            "runner_sha256": sha256_file(RUNNER_PATH),
            "scientific_contract_hash": clean_text(old_prelock["scientific_contract_hash"]),
            "numerical_implementation_hash": clean_text(
                old_prelock["numerical_implementation_hash"]
            ),
            "seed_registry_sha256": sha256_file(runner.OUTPUT_SEED_REGISTRY),
            "epoch_bridge_sha256": sha256_file(runner.OUTPUT_EPOCH_BRIDGE),
            "numerical_contract_sha256": sha256_file(runner.OUTPUT_NUMERICAL_CONTRACT),
            "reuse_first_three_05f1_pilot_shards": True,
        },
        "runtime_only_comparison": {
            "05f1_original_sequential_mean_wall_seconds_per_dataset": float(
                runtime_evidence["original_mean_wall_seconds_per_dataset"]
            ),
            "05f1_original_sequential_rough_full_run_days": float(
                runtime_evidence["05f1_summary"]["rough_full_run_extrapolated_days"]
            ),
            "05f1d_two_stream_mean_wall_seconds_per_dataset": float(
                runtime_evidence["two_stream_mean_wall_seconds_per_dataset"]
            ),
            "05f1d_two_stream_rough_full_run_days": float(
                runtime_evidence["05f1d_summary"]["rough_full_run_extrapolated_days"]
            ),
            "05f1d_two_stream_exact_scientific_payload_equivalence": True,
            "decision": (
                "Do not adopt two-stream execution because the completed three-dataset "
                "full-path pilot was slower than the already-frozen original sequential path."
            ),
            "scientific_metrics_used_for_decision": False,
            "scientific_thresholds_used_for_decision": False,
            "final_gate_PASS_FAIL_used_for_decision": False,
        },
        "runtime_environment": {
            "PyTorch_version": torch.__version__,
            "CUDA_build": torch.version.cuda,
            "GPU_model": torch.cuda.get_device_name(0),
        },
        "upstream_records": {
            "05f1_prelock": file_record(runner.OUTPUT_PRELOCK),
            "05f1_numerical_contract": file_record(runner.OUTPUT_NUMERICAL_CONTRACT),
            "05f1_seed_registry": file_record(runner.OUTPUT_SEED_REGISTRY),
            "05f1_epoch_bridge": file_record(runner.OUTPUT_EPOCH_BRIDGE),
            "05f1_runtime_summary": file_record(
                runtime_evidence["05f1_summary_path"]
            ),
            "05f1_runtime_lock": file_record(runner.OUTPUT_LOCK),
            "05f1d_runtime_summary": file_record(
                runtime_evidence["05f1d_summary_path"]
            ),
            "05f1d_validation_lock": file_record(TWO_STREAM_LOCK),
        },
        "scientific_state_before_first_new_full_run_fit": {
            "scientific_metrics_printed_by_05f2": False,
            "scientific_metrics_aggregated_by_05f2": False,
            "D01_D02_D03_evaluated_by_05f2": False,
            "final_gate_PASS_FAIL_evaluated_by_05f2": False,
            "TARGET_accessed_by_05f2": False,
            "outcomes_accessed_by_05f2": False,
            "threshold_retuning_performed_by_05f2": False,
        },
        "resumption_policy": {
            "completed_dataset_shard_is_atomic_and_reused_on_resume": True,
            "progress_manifest_is_atomic_and_contiguous_in_frozen_registry_order": True,
            "interrupted_incomplete_dataset_is_recomputed": True,
            "scientific_payload_remains_sealed_until_post_run_aggregation_stage": True,
        },
        "full_run_scope": {
            "n_scenarios": EXPECTED_SCENARIOS,
            "n_datasets": EXPECTED_DATASETS,
            "stability_resamples_per_dataset": EXPECTED_STABILITY_RESAMPLES,
            "optimization_runs_represented": EXPECTED_OPTIMIZATION_RUNS,
            "pilot_shards_reused": EXPECTED_REUSED_PILOT_DATASETS,
        },
        "numerical_contract_version": clean_text(
            numerical.get("numerical_implementation_version")
        ),
        "full_run_scientific_aggregation_in_05f2": False,
    }

    hash_basis = json.loads(json.dumps(record))
    hash_basis.pop("created_at_utc", None)
    record["execution_choice_hash"] = sha256_text(canonical_json(hash_basis))
    return record


def initialize_or_verify_execution_choice_prelock(
    runner: Any,
    runtime_evidence: dict[str, Any],
) -> dict[str, Any]:
    proposed = build_execution_choice_prelock(runner, runtime_evidence)

    if OUTPUT_EXECUTION_PRELOCK.exists():
        existing = load_json(OUTPUT_EXECUTION_PRELOCK)
        existing_core = json.loads(json.dumps(existing))
        proposed_core = json.loads(json.dumps(proposed))
        existing_core.pop("created_at_utc", None)
        proposed_core.pop("created_at_utc", None)
        if canonical_json(existing_core) != canonical_json(proposed_core):
            raise RuntimeError(
                "Existing 05f2 execution-choice prelock differs from the current script/runtime "
                "evidence. Do not modify a started full run; create a new versioned orchestrator."
            )
        return existing

    # Before first new full-run fit, only the three already-frozen 05f1 pilot
    # shards may exist in the selected compact shard directory.
    observed = list(runner.SHARD_DIR.glob("shard_*.json"))
    pilot_manifest = pd.read_csv(
        runner.OUTPUT_PILOT_MANIFEST,
        dtype=str,
        low_memory=False,
    ).fillna("")
    if pilot_manifest.shape[0] != EXPECTED_REUSED_PILOT_DATASETS:
        raise RuntimeError("05f1 pilot manifest no longer contains exactly 3 rows.")
    expected_paths = {
        str(resolve_project_path(path).resolve()).lower()
        for path in pilot_manifest["shard_path"].tolist()
    }
    observed_paths = {
        str(path.resolve()).lower()
        for path in observed
    }
    if observed_paths != expected_paths:
        raise RuntimeError(
            "Before the first 05f2 full-run fit, the compact final-gate shard directory "
            "must contain exactly the three frozen 05f1 pilot shards."
        )

    write_json_atomic(proposed, OUTPUT_EXECUTION_PRELOCK)
    return proposed


# -----------------------------------------------------------------------------
# Progress manifest / resume integrity
# -----------------------------------------------------------------------------

PROGRESS_COLUMNS = [
    "registry_row_index",
    "dataset_key",
    "scenario_id",
    "replicate_index",
    "simulation_seed",
    "completion_source",
    "optimization_runs",
    "wall_seconds_full_dataset",
    "wrapper_wall_seconds",
    "peak_cuda_memory_allocated_bytes",
    "shard_path",
    "shard_sha256",
    "shard_size_bytes",
    "completed_at_utc",
    "scientific_metrics_printed",
    "scientific_metrics_aggregated",
]


def normalized_replicates(upstream: dict[str, Any]) -> pd.DataFrame:
    table = upstream["replicates"].copy()
    required = {
        "registry_row_index",
        "dataset_key",
        "scenario_id",
        "replicate_index",
        "simulation_seed",
    }
    missing = required - set(table.columns)
    if missing:
        raise RuntimeError(
            "Frozen final-gate replicate registry missing columns: "
            + ", ".join(sorted(missing))
        )
    for column in ["registry_row_index", "replicate_index", "simulation_seed"]:
        table[column] = pd.to_numeric(
            table[column],
            errors="raise",
        ).astype(np.int64)
    table = table.sort_values("registry_row_index", kind="mergesort").reset_index(drop=True)
    if table.shape[0] != EXPECTED_DATASETS:
        raise RuntimeError("Final-gate replicate registry is not 3,100 rows.")
    if table["registry_row_index"].duplicated().any():
        raise RuntimeError("Duplicate final-gate registry_row_index.")
    if table["dataset_key"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate final-gate dataset_key.")
    return table


def initialize_progress_from_05f1_pilot(
    runner: Any,
    replicates: pd.DataFrame,
) -> pd.DataFrame:
    pilot_manifest = pd.read_csv(
        runner.OUTPUT_PILOT_MANIFEST,
        dtype=str,
        low_memory=False,
    ).fillna("")
    if pilot_manifest.shape[0] != EXPECTED_REUSED_PILOT_DATASETS:
        raise RuntimeError("05f1 pilot manifest no longer contains exactly three rows.")

    rows: list[dict[str, Any]] = []
    for ordinal in range(EXPECTED_REUSED_PILOT_DATASETS):
        registry_row = replicates.iloc[ordinal]
        pilot_row = pilot_manifest.iloc[ordinal]

        dataset_key = clean_text(registry_row["dataset_key"])
        if clean_text(pilot_row["dataset_key"]) != dataset_key:
            raise RuntimeError("05f1 pilot manifest no longer matches first three registry rows.")

        shard_path = resolve_project_path(pilot_row["shard_path"])
        if not shard_path.is_file():
            raise FileNotFoundError(f"05f1 pilot shard missing: {shard_path}")
        expected_hash = clean_text(pilot_row["shard_sha256"])
        if sha256_file(shard_path) != expected_hash:
            raise RuntimeError("05f1 pilot shard hash mismatch during 05f2 initialization.")

        rows.append(
            {
                "registry_row_index": int(registry_row["registry_row_index"]),
                "dataset_key": dataset_key,
                "scenario_id": clean_text(registry_row["scenario_id"]),
                "replicate_index": int(registry_row["replicate_index"]),
                "simulation_seed": int(registry_row["simulation_seed"]),
                "completion_source": "reused_frozen_05f1_pilot_shard",
                "optimization_runs": 1 + EXPECTED_STABILITY_RESAMPLES,
                "wall_seconds_full_dataset": float(
                    pilot_row["wall_seconds_full_dataset"]
                ),
                "wrapper_wall_seconds": float(
                    pilot_row["wrapper_wall_seconds"]
                ),
                "peak_cuda_memory_allocated_bytes": int(
                    float(pilot_row["peak_cuda_memory_allocated_bytes"])
                ),
                "shard_path": relative_path(shard_path),
                "shard_sha256": expected_hash,
                "shard_size_bytes": int(shard_path.stat().st_size),
                "completed_at_utc": "",
                "scientific_metrics_printed": False,
                "scientific_metrics_aggregated": False,
            }
        )

    table = pd.DataFrame(rows, columns=PROGRESS_COLUMNS)
    write_csv_atomic(table, OUTPUT_PROGRESS_MANIFEST)
    return table


def validate_progress_manifest(
    table: pd.DataFrame,
    replicates: pd.DataFrame,
) -> pd.DataFrame:
    if set(PROGRESS_COLUMNS) - set(table.columns):
        raise RuntimeError("05f2 progress manifest schema is incomplete.")
    table = table[PROGRESS_COLUMNS].copy()

    n = table.shape[0]
    if n < EXPECTED_REUSED_PILOT_DATASETS or n > EXPECTED_DATASETS:
        raise RuntimeError("05f2 progress manifest row count is invalid.")

    for column in [
        "registry_row_index",
        "replicate_index",
        "simulation_seed",
        "optimization_runs",
        "peak_cuda_memory_allocated_bytes",
        "shard_size_bytes",
    ]:
        table[column] = pd.to_numeric(
            table[column],
            errors="raise",
        ).astype(np.int64)

    for column in [
        "wall_seconds_full_dataset",
        "wrapper_wall_seconds",
    ]:
        table[column] = pd.to_numeric(
            table[column],
            errors="raise",
        ).astype(float)

    expected = replicates.iloc[:n].reset_index(drop=True)
    observed = table.reset_index(drop=True)

    for index in range(n):
        for column in [
            "registry_row_index",
            "dataset_key",
            "scenario_id",
            "replicate_index",
            "simulation_seed",
        ]:
            left = clean_text(observed.iloc[index][column])
            right = clean_text(expected.iloc[index][column])
            if left != right:
                raise RuntimeError(
                    f"05f2 progress manifest is not contiguous in frozen registry order "
                    f"at row {index + 1}, field {column}."
                )

    if observed["dataset_key"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate dataset_key in 05f2 progress manifest.")

    # Hash-check every completed shard. This is intentionally disk-only on resume:
    # no already-completed scientific payload is aggregated or interpreted.
    for _, row in observed.iterrows():
        shard_path = resolve_project_path(row["shard_path"])
        if not shard_path.is_file():
            raise FileNotFoundError(f"Completed 05f2 shard missing: {shard_path}")
        if int(shard_path.stat().st_size) != int(row["shard_size_bytes"]):
            raise RuntimeError("Completed 05f2 shard size mismatch.")
        if sha256_file(shard_path) != clean_text(row["shard_sha256"]):
            raise RuntimeError("Completed 05f2 shard SHA256 mismatch.")

    if n >= EXPECTED_REUSED_PILOT_DATASETS:
        if not observed.iloc[:EXPECTED_REUSED_PILOT_DATASETS][
            "completion_source"
        ].eq("reused_frozen_05f1_pilot_shard").all():
            raise RuntimeError("First three 05f2 progress rows are not frozen 05f1 pilot reuse.")

    return observed


def load_or_initialize_progress(
    runner: Any,
    replicates: pd.DataFrame,
) -> pd.DataFrame:
    if OUTPUT_PROGRESS_MANIFEST.exists():
        table = pd.read_csv(
            OUTPUT_PROGRESS_MANIFEST,
            dtype=str,
            low_memory=False,
        ).fillna("")
        return validate_progress_manifest(table, replicates)

    # No progress manifest: initialize by importing only the already-frozen
    # three 05f1 pilot shards. Any additional orphan shard from a crash after
    # shard write but before progress update will be recovered naturally when
    # its registry row is next executed.
    return initialize_progress_from_05f1_pilot(runner, replicates)


# -----------------------------------------------------------------------------
# Disk preflight / runtime state
# -----------------------------------------------------------------------------

def disk_preflight(
    runner: Any,
    progress: pd.DataFrame,
) -> dict[str, Any]:
    pilot_manifest = pd.read_csv(
        runner.OUTPUT_PILOT_MANIFEST,
        dtype=str,
        low_memory=False,
    ).fillna("")
    pilot_sizes = pd.to_numeric(
        pilot_manifest["shard_size_bytes"],
        errors="raise",
    ).to_numpy(dtype=np.int64)
    mean_shard_bytes = float(np.mean(pilot_sizes))

    remaining = EXPECTED_DATASETS - int(progress.shape[0])
    projected_remaining = mean_shard_bytes * remaining
    # 35% overhead plus 1 GiB safety buffer for manifests/temp writes.
    required_free = int(math.ceil(projected_remaining * 1.35 + 1 * 1024**3))
    usage = shutil.disk_usage(runner.RESULTS_ROOT)
    free_bytes = int(usage.free)

    record = {
        "mean_pilot_shard_size_bytes": mean_shard_bytes,
        "remaining_datasets": remaining,
        "projected_remaining_shard_bytes": int(projected_remaining),
        "required_free_bytes_with_buffer": required_free,
        "observed_free_bytes": free_bytes,
        "pass": bool(free_bytes >= required_free),
    }
    if not record["pass"]:
        raise RuntimeError(
            "Insufficient free disk space for the remaining final-gate shards. "
            f"Need at least {required_free / (1024**3):.2f} GiB free under the conservative "
            f"buffer; observed {free_bytes / (1024**3):.2f} GiB."
        )
    return record


def build_progress_state(
    progress: pd.DataFrame,
    session_started_at: float,
    session_generated_times: list[float],
    disk_record: dict[str, Any],
    execution_prelock: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    completed = int(progress.shape[0])
    remaining = EXPECTED_DATASETS - completed

    if session_generated_times:
        mean_current = float(np.mean(session_generated_times))
    elif completed > 0:
        mean_current = float(
            pd.to_numeric(
                progress["wrapper_wall_seconds"],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan).dropna().mean()
        )
    else:
        mean_current = float("nan")

    if math.isfinite(mean_current) and mean_current > 0:
        eta_seconds = remaining * mean_current
    else:
        eta_seconds = None

    return {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "updated_at_utc": utc_now_iso(),
        "status": status,
        "completed_datasets": completed,
        "remaining_datasets": remaining,
        "completion_fraction": completed / EXPECTED_DATASETS,
        "optimization_runs_represented": completed * (1 + EXPECTED_STABILITY_RESAMPLES),
        "session_generated_datasets": len(session_generated_times),
        "session_elapsed_seconds": time.perf_counter() - session_started_at,
        "current_runtime_mean_seconds_per_new_dataset": (
            mean_current if math.isfinite(mean_current) else None
        ),
        "rough_remaining_eta_seconds": eta_seconds,
        "rough_remaining_eta_days": (
            eta_seconds / 86400.0 if eta_seconds is not None else None
        ),
        "progress_manifest": file_record(OUTPUT_PROGRESS_MANIFEST),
        "execution_choice_hash": clean_text(execution_prelock["execution_choice_hash"]),
        "scientific_contract_hash": clean_text(
            execution_prelock["selected_execution_path"]["scientific_contract_hash"]
        ),
        "numerical_implementation_hash": clean_text(
            execution_prelock["selected_execution_path"]["numerical_implementation_hash"]
        ),
        "disk_preflight": disk_record,
        "scientific_metrics_printed": False,
        "scientific_metrics_aggregated": False,
        "D01_D02_D03_evaluated": False,
        "final_gate_PASS_FAIL_evaluated": False,
    }


# -----------------------------------------------------------------------------
# Dataset execution
# -----------------------------------------------------------------------------

def shard_path_from_saved_shard(
    runner: Any,
    shard: dict[str, Any],
    selected_epoch: int,
    old_prelock: dict[str, Any],
) -> Path:
    pseudo_metadata = {
        "dataset_key": clean_text(shard["dataset_key"]),
        "input_sha256": clean_text(shard["input_sha256"]),
    }
    path = runner.final_gate_shard_path(
        pseudo_metadata,
        selected_epoch,
        old_prelock,
    )
    if not path.is_file():
        raise FileNotFoundError(f"Expected final-gate shard not found after execution: {path}")
    return path


def run_one_registry_row(
    runner: Any,
    upstream: dict[str, Any],
    old_prelock: dict[str, Any],
    seed_registry: pd.DataFrame,
    selected_epoch: int,
    registry_row: pd.Series,
) -> dict[str, Any]:
    wrapper_started = time.perf_counter()
    shard, generation_status, wall_seconds, peak_bytes = (
        runner.execute_or_verify_dataset_shard(
            upstream=upstream,
            prelock=old_prelock,
            seed_registry=seed_registry,
            replicate_row=registry_row,
            selected_epoch=selected_epoch,
        )
    )
    wrapper_seconds = time.perf_counter() - wrapper_started
    shard_path = shard_path_from_saved_shard(
        runner,
        shard,
        selected_epoch,
        old_prelock,
    )

    expected_dataset_key = clean_text(registry_row["dataset_key"])
    if clean_text(shard["dataset_key"]) != expected_dataset_key:
        raise RuntimeError("Saved shard dataset key differs from frozen registry row.")

    return {
        "registry_row_index": int(registry_row["registry_row_index"]),
        "dataset_key": expected_dataset_key,
        "scenario_id": clean_text(registry_row["scenario_id"]),
        "replicate_index": int(registry_row["replicate_index"]),
        "simulation_seed": int(registry_row["simulation_seed"]),
        "completion_source": (
            "generated_05f2_full_run"
            if generation_status == "generated"
            else "verified_existing_orphan_shard"
        ),
        "optimization_runs": 1 + EXPECTED_STABILITY_RESAMPLES,
        "wall_seconds_full_dataset": float(wall_seconds),
        "wrapper_wall_seconds": float(wrapper_seconds),
        "peak_cuda_memory_allocated_bytes": int(peak_bytes),
        "shard_path": relative_path(shard_path),
        "shard_sha256": sha256_file(shard_path),
        "shard_size_bytes": int(shard_path.stat().st_size),
        "completed_at_utc": utc_now_iso(),
        "scientific_metrics_printed": False,
        "scientific_metrics_aggregated": False,
    }


# -----------------------------------------------------------------------------
# Completion
# -----------------------------------------------------------------------------

def build_completion_summary(
    progress: pd.DataFrame,
    execution_prelock: dict[str, Any],
) -> dict[str, Any]:
    if progress.shape[0] != EXPECTED_DATASETS:
        raise RuntimeError("Cannot complete 05f2 before all 3,100 datasets are present.")

    wall = pd.to_numeric(
        progress["wall_seconds_full_dataset"],
        errors="raise",
    ).to_numpy(dtype=float)
    wrapper = pd.to_numeric(
        progress["wrapper_wall_seconds"],
        errors="raise",
    ).to_numpy(dtype=float)
    sizes = pd.to_numeric(
        progress["shard_size_bytes"],
        errors="raise",
    ).to_numpy(dtype=np.int64)
    peak = pd.to_numeric(
        progress["peak_cuda_memory_allocated_bytes"],
        errors="raise",
    ).to_numpy(dtype=np.int64)

    return {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": "FULL_FINAL_GATE_SHARD_GENERATION_COMPLETE_NO_SCIENTIFIC_AGGREGATION",
        "n_scenarios": EXPECTED_SCENARIOS,
        "n_datasets": EXPECTED_DATASETS,
        "stability_resamples_per_dataset": EXPECTED_STABILITY_RESAMPLES,
        "optimization_runs_represented": EXPECTED_OPTIMIZATION_RUNS,
        "reused_frozen_05f1_pilot_datasets": int(
            progress["completion_source"]
            .astype(str)
            .eq("reused_frozen_05f1_pilot_shard")
            .sum()
        ),
        "generated_or_recovered_full_run_datasets": int(
            EXPECTED_DATASETS
            - progress["completion_source"]
            .astype(str)
            .eq("reused_frozen_05f1_pilot_shard")
            .sum()
        ),
        "sum_stored_dataset_wall_seconds": float(np.sum(wall)),
        "mean_stored_dataset_wall_seconds": float(np.mean(wall)),
        "median_stored_dataset_wall_seconds": float(np.median(wall)),
        "sum_stored_wrapper_wall_seconds": float(np.sum(wrapper)),
        "peak_cuda_memory_allocated_bytes": int(np.max(peak)),
        "total_shard_bytes": int(np.sum(sizes)),
        "total_shard_gib": float(np.sum(sizes) / (1024**3)),
        "execution_choice_hash": clean_text(
            execution_prelock["execution_choice_hash"]
        ),
        "scientific_contract_hash": clean_text(
            execution_prelock["selected_execution_path"]["scientific_contract_hash"]
        ),
        "numerical_implementation_hash": clean_text(
            execution_prelock["selected_execution_path"]["numerical_implementation_hash"]
        ),
        "scientific_metrics_printed": False,
        "scientific_metrics_aggregated": False,
        "D01_D02_D03_evaluated": False,
        "final_gate_PASS_FAIL_evaluated": False,
        "TARGET_accessed": False,
        "outcomes_accessed": False,
        "threshold_retuning_performed": False,
        "next_stage": (
            "05f3 read all 3,100 sealed shards once, compute the frozen D01/D02/D03 "
            "and descriptive final-gate operating metrics, then emit numerator/denominator/"
            "CI/PASS-FAIL without retuning."
        ),
    }


def build_readme(
    completion_summary: dict[str, Any],
    execution_prelock: dict[str, Any],
) -> str:
    return (
        "Paper 5 05f2 — resumable full final-gate shard generation\n"
        "==========================================================\n\n"
        "Execution path\n"
        "--------------\n"
        "05f2 uses the original frozen 05f1 sequential numerical implementation.\n"
        "The exact-equivalent two-stream alternative was not adopted because the\n"
        "completed three-dataset full-path runtime pilot was slower. This choice\n"
        "used runtime information only; no final-gate scientific metric was viewed.\n\n"
        "Resumption\n"
        "----------\n"
        "Each completed dataset is stored as an atomic sealed shard. The progress\n"
        "manifest is updated atomically after each completed dataset. If execution is\n"
        "interrupted, rerun the same 05f2 command. Completed contiguous rows are\n"
        "hash-verified and skipped. A shard written immediately before interruption\n"
        "but not yet entered into the progress manifest is recovered automatically.\n\n"
        "Scientific blinding\n"
        "-------------------\n"
        "05f2 does not aggregate, print or evaluate D01, D02, D03, FDP, TPR, FNR or\n"
        "final PASS/FAIL. Those values remain sealed in per-dataset shards until 05f3.\n\n"
        f"Datasets complete: {completion_summary['n_datasets']}\n"
        f"Optimization runs represented: {completion_summary['optimization_runs_represented']}\n"
        f"Stored shard volume: {completion_summary['total_shard_gib']:.3f} GiB\n"
        f"Scientific contract hash: "
        f"{execution_prelock['selected_execution_path']['scientific_contract_hash']}\n"
        f"Numerical implementation hash: "
        f"{execution_prelock['selected_execution_path']['numerical_implementation_hash']}\n"
    )


def verify_existing_completion_lock() -> None:
    lock = load_json(OUTPUT_LOCK)
    if clean_text(lock.get("status")) != COMPLETION_LOCK_STATUS:
        raise RuntimeError("Existing 05f2 completion lock has unexpected status.")
    if clean_text(lock.get("script_sha256")) != sha256_file(SCRIPT_PATH):
        raise RuntimeError("Current 05f2 script differs from completed-run hash.")

    for key in ["execution_prelock", "progress_manifest", "completion_summary"]:
        record = lock.get(key, {})
        path = resolve_project_path(record.get("path"))
        if not path.is_file():
            raise FileNotFoundError(f"05f2 completion artifact missing: {path}")
        if sha256_file(path) != clean_text(record.get("sha256")):
            raise RuntimeError(f"05f2 completion artifact hash mismatch: {key}")

    summary = load_json(OUTPUT_COMPLETION_SUMMARY)
    print("=" * 88)
    print("Existing Paper 5 05f2 full final-gate shard run: VERIFIED")
    print("=" * 88)
    print(f"Datasets complete: {int(summary['n_datasets']):,}/{EXPECTED_DATASETS:,}")
    print(
        f"Optimization runs represented: "
        f"{int(summary['optimization_runs_represented']):,}"
    )
    print(f"Stored shard volume: {float(summary['total_shard_gib']):.2f} GiB")
    print("Scientific metrics printed/aggregated: NO")
    print("Final-gate PASS/FAIL evaluated: NO")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("No files were rewritten.")
    print("Next: 05f3 frozen final-gate scientific aggregation.")
    print("=" * 88)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    print("=" * 88)
    print("Paper 5 - resumable full final-gate GPU streaming run")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Final-gate datasets: {EXPECTED_DATASETS:,}")
    print(f"Stability resamples per dataset: {EXPECTED_STABILITY_RESAMPLES}")
    print(f"Planned optimization runs represented: {EXPECTED_OPTIMIZATION_RUNS:,}")
    print("Execution path: original frozen 05f1 sequential numerical path")
    print("Resumable per completed dataset: YES")
    print("Permanent final-gate input bank: NO")
    print("Scientific metrics printed during full run: NO")
    print("Scientific metrics aggregated during full run: NO")
    print("Final-gate PASS/FAIL evaluated during full run: NO")
    print("Threshold retuning: FORBIDDEN")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_LOCK.exists():
        verify_existing_completion_lock()
        return

    runner = import_module(RUNNER_PATH, "paper5_05f2_frozen_05f1_runner")
    runtime_evidence = verify_05f1_and_05f1d_runtime_evidence(runner)

    # Full upstream replay verifies the scientific registries, frozen generator,
    # neural helper, profile, thresholds, CUDA runtime and the 05f0 streaming policy.
    upstream = runner.verify_upstream_and_load()

    execution_prelock = initialize_or_verify_execution_choice_prelock(
        runner,
        runtime_evidence,
    )

    old_prelock = runner.load_json(runner.OUTPUT_PRELOCK)
    if clean_text(old_prelock["scientific_contract_hash"]) != clean_text(
        execution_prelock["selected_execution_path"]["scientific_contract_hash"]
    ):
        raise RuntimeError("05f2 execution prelock scientific hash differs from frozen 05f1.")
    if clean_text(old_prelock["numerical_implementation_hash"]) != clean_text(
        execution_prelock["selected_execution_path"]["numerical_implementation_hash"]
    ):
        raise RuntimeError("05f2 execution prelock numerical hash differs from frozen 05f1.")

    runner.configure_final_gate_neural_helper(upstream["neural"])

    selected_epoch = int(
        runner.load_json(runner.OUTPUT_EPOCH_BRIDGE)["final_gate_fixed_epoch"]
    )
    seed_registry = pd.read_csv(
        runner.OUTPUT_SEED_REGISTRY,
        dtype=str,
        low_memory=False,
    ).fillna("")
    if seed_registry.shape[0] != EXPECTED_DATASETS * EXPECTED_STABILITY_RESAMPLES:
        raise RuntimeError("Frozen final-gate seed registry row count mismatch.")

    replicates = normalized_replicates(upstream)
    progress = load_or_initialize_progress(runner, replicates)
    progress = validate_progress_manifest(progress, replicates)

    disk_record = disk_preflight(runner, progress)

    print("05f1 frozen numerical path: VERIFIED")
    print("05f1d exact-equivalent two-stream alternative: VERIFIED BUT NOT ADOPTED")
    print(
        "Runtime-only path choice: sequential "
        f"{runtime_evidence['original_mean_wall_seconds_per_dataset']:.1f}s/dataset "
        f"vs two-stream {runtime_evidence['two_stream_mean_wall_seconds_per_dataset']:.1f}s/dataset"
    )
    print(
        "Scientific metrics used for execution choice: NO"
    )
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Execution-choice prelock: FROZEN | "
        f"numerical hash={clean_text(old_prelock['numerical_implementation_hash'])}"
    )
    print(
        f"Resume state: {progress.shape[0]:,}/{EXPECTED_DATASETS:,} datasets already complete"
    )
    print(
        f"Disk preflight: PASS | free={disk_record['observed_free_bytes'] / (1024**3):.2f} GiB | "
        f"conservative required={disk_record['required_free_bytes_with_buffer'] / (1024**3):.2f} GiB"
    )
    print("Scientific values remain sealed; console below is runtime/progress only.")
    print()

    session_started_at = time.perf_counter()
    session_generated_times: list[float] = []

    state = build_progress_state(
        progress,
        session_started_at,
        session_generated_times,
        disk_record,
        execution_prelock,
        status="RUNNING_RESUMABLE_FULL_FINAL_GATE",
    )
    write_json_atomic(state, OUTPUT_PROGRESS_STATE)

    try:
        start_index = int(progress.shape[0])
        for zero_index in range(start_index, EXPECTED_DATASETS):
            registry_row = replicates.iloc[zero_index]
            ordinal = zero_index + 1

            row_record = run_one_registry_row(
                runner=runner,
                upstream=upstream,
                old_prelock=old_prelock,
                seed_registry=seed_registry,
                selected_epoch=selected_epoch,
                registry_row=registry_row,
            )

            progress = pd.concat(
                [
                    progress,
                    pd.DataFrame([row_record], columns=PROGRESS_COLUMNS),
                ],
                ignore_index=True,
            )
            # Per-dataset persistence must stay O(1): the newly written shard was
            # already validated and SHA256-hashed by run_one_registry_row(). Do not
            # re-hash every prior shard after every dataset. Full contiguous/hash
            # validation is done once at process start/resume and once at completion.
            if progress.shape[0] != ordinal:
                raise RuntimeError("05f2 progress row count is not contiguous after append.")
            just_added = progress.iloc[-1]
            if clean_text(just_added["dataset_key"]) != clean_text(registry_row["dataset_key"]):
                raise RuntimeError("05f2 appended progress row differs from current registry dataset.")
            if int(just_added["registry_row_index"]) != int(registry_row["registry_row_index"]):
                raise RuntimeError("05f2 appended progress row has wrong registry_row_index.")
            write_csv_atomic(progress, OUTPUT_PROGRESS_MANIFEST)

            if row_record["completion_source"] == "generated_05f2_full_run":
                session_generated_times.append(
                    float(row_record["wrapper_wall_seconds"])
                )

            state = build_progress_state(
                progress,
                session_started_at,
                session_generated_times,
                disk_record,
                execution_prelock,
                status="RUNNING_RESUMABLE_FULL_FINAL_GATE",
            )
            write_json_atomic(state, OUTPUT_PROGRESS_STATE)

            eta_days = state.get("rough_remaining_eta_days")
            eta_text = (
                f" | ETA~{float(eta_days):.2f}d"
                if eta_days is not None
                else ""
            )
            print(
                f"{ordinal:04d}/{EXPECTED_DATASETS:04d} "
                f"runtime={float(row_record['wall_seconds_full_dataset']):.1f}s | "
                f"VRAM={int(row_record['peak_cuda_memory_allocated_bytes']) / (1024**2):.1f} MiB | "
                f"{row_record['completion_source']}"
                f"{eta_text}"
            )

            torch.cuda.empty_cache()
            gc.collect()

    except KeyboardInterrupt:
        state = build_progress_state(
            progress,
            session_started_at,
            session_generated_times,
            disk_record,
            execution_prelock,
            status="INTERRUPTED_RESUMABLE_SAFE_TO_RERUN",
        )
        write_json_atomic(state, OUTPUT_PROGRESS_STATE)
        print()
        print("=" * 88)
        print("05f2 interrupted after the last completed atomic dataset shard.")
        print(f"Completed: {progress.shape[0]:,}/{EXPECTED_DATASETS:,}")
        print("Safe resume command:")
        print(r"python paper5\scripts\05f2_run_full_final_gate_gpu_resumable.py")
        print("No scientific metric was aggregated or printed.")
        print("=" * 88)
        return

    except Exception as exc:
        write_json_atomic(
            {
                "contract_version": CONTRACT_VERSION,
                "script_version": SCRIPT_VERSION,
                "created_at_utc": utc_now_iso(),
                "status": "RUNTIME_IMPLEMENTATION_FAILURE_RESUMABLE_NOT_SCIENTIFIC_GATE_RESULT",
                "completed_datasets_before_failure": int(progress.shape[0]),
                "error_type": type(exc).__name__,
                "error_message": clean_text(str(exc))[:4000],
                "traceback_tail": "\n".join(
                    traceback.format_exc().splitlines()[-60:]
                ),
                "scientific_metrics_printed": False,
                "scientific_metrics_aggregated": False,
                "D01_D02_D03_evaluated": False,
                "final_gate_PASS_FAIL_evaluated": False,
                "threshold_retuning_performed": False,
                "safe_to_resume_same_script_if_environment_and_hashes_are_unchanged": True,
            },
            OUTPUT_FAILURE,
        )
        raise

    # Full completion integrity. Progress validation SHA-checks all 3,100 shards
    # without aggregating any scientific values.
    progress = validate_progress_manifest(progress, replicates)
    if progress.shape[0] != EXPECTED_DATASETS:
        raise RuntimeError("05f2 loop ended without all 3,100 completed datasets.")

    completion_summary = build_completion_summary(
        progress,
        execution_prelock,
    )
    write_json_atomic(completion_summary, OUTPUT_COMPLETION_SUMMARY)
    write_text_atomic(
        build_readme(completion_summary, execution_prelock),
        OUTPUT_README,
    )

    completion_lock = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "script_path": relative_path(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "created_at_utc": utc_now_iso(),
        "status": COMPLETION_LOCK_STATUS,
        "execution_choice_hash": clean_text(
            execution_prelock["execution_choice_hash"]
        ),
        "scientific_contract_hash": clean_text(
            execution_prelock["selected_execution_path"]["scientific_contract_hash"]
        ),
        "numerical_implementation_hash": clean_text(
            execution_prelock["selected_execution_path"]["numerical_implementation_hash"]
        ),
        "n_datasets": EXPECTED_DATASETS,
        "optimization_runs_represented": EXPECTED_OPTIMIZATION_RUNS,
        "execution_prelock": file_record(OUTPUT_EXECUTION_PRELOCK),
        "progress_manifest": file_record(OUTPUT_PROGRESS_MANIFEST),
        "completion_summary": file_record(OUTPUT_COMPLETION_SUMMARY),
        "scientific_state": {
            "scientific_metrics_printed": False,
            "scientific_metrics_aggregated": False,
            "D01_D02_D03_evaluated": False,
            "final_gate_PASS_FAIL_evaluated": False,
            "TARGET_accessed": False,
            "outcomes_accessed": False,
            "threshold_retuning_performed": False,
        },
        "next_stage": (
            "05f3 frozen final-gate aggregation from all 3,100 sealed shards."
        ),
    }
    write_json_atomic(completion_lock, OUTPUT_LOCK)

    if OUTPUT_FAILURE.exists():
        OUTPUT_FAILURE.unlink()

    final_state = build_progress_state(
        progress,
        session_started_at,
        session_generated_times,
        disk_record,
        execution_prelock,
        status="COMPLETE_ALL_3100_SHARDS_SEALED_BEFORE_SCIENTIFIC_AGGREGATION",
    )
    write_json_atomic(final_state, OUTPUT_PROGRESS_STATE)

    print()
    print("-" * 88)
    print("Full final-gate shard-generation summary (runtime only)")
    print("-" * 88)
    print(f"Datasets complete: {EXPECTED_DATASETS:,}/{EXPECTED_DATASETS:,}")
    print(f"Optimization runs represented: {EXPECTED_OPTIMIZATION_RUNS:,}")
    print(
        f"Reused frozen 05f1 pilot datasets: "
        f"{int(completion_summary['reused_frozen_05f1_pilot_datasets'])}"
    )
    print(
        f"Stored shard volume: "
        f"{float(completion_summary['total_shard_gib']):.2f} GiB"
    )
    print("Scientific metrics printed: NO")
    print("Scientific metrics aggregated: NO")
    print("D01/D02/D03 evaluated: NO")
    print("Final-gate PASS/FAIL evaluated: NO")
    print()
    print("=" * 88)
    print("Paper 5 05f2 full final-gate GPU shard run: COMPLETE")
    print("=" * 88)
    print(f"Progress manifest: {relative_path(OUTPUT_PROGRESS_MANIFEST)}")
    print(f"Completion summary: {relative_path(OUTPUT_COMPLETION_SUMMARY)}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("Next: 05f3 aggregate the frozen final-gate scientific results once.")
    print("=" * 88)


if __name__ == "__main__":
    main()
