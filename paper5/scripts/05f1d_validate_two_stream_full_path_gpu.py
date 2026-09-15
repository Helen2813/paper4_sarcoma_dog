from __future__ import annotations

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
from typing import Any, Callable

# Preserve the already-frozen deterministic CUDA GEMM configuration before torch import.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pandas as pd
import torch


SCRIPT_VERSION = "05f1d-validate-two-stream-full-path-gpu-v1-no-cli"
CONTRACT_VERSION = "paper5-05f1d-two-stream-full-path-validation-v1"
PRELOCK_STATUS = "FROZEN_TWO_STREAM_EXECUTION_PATH_BEFORE_05F1D_PILOT_FITS"
LOCK_STATUS = "FROZEN_TWO_STREAM_FULL_PATH_PILOT_VALIDATED_BEFORE_FULL_FINAL_GATE"

STREAM_WIDTH = 2
EXPECTED_PILOT_DATASETS = 3
EXPECTED_STABILITY_RESAMPLES = 100
EXPECTED_OPTIMIZATION_RUNS = EXPECTED_PILOT_DATASETS * (1 + EXPECTED_STABILITY_RESAMPLES)
EXPECTED_05F1_SCRIPT_VERSION = (
    "05f1-run-final-gate-runtime-pilot-gpu-v2-seed-collision-aware-no-cli"
)
EXPECTED_05F1_LOCK_STATUS = "FROZEN_FINAL_GATE_RUNTIME_PILOT_COMPLETE_BEFORE_FULL_RUN"
EXPECTED_05F1C2_SCRIPT_VERSION = (
    "05f1c2-benchmark-single-process-cuda-streams-gpu-v1-no-cli"
)
EXPECTED_05F1C2_LOCK_STATUS = "RUNTIME_ONLY_CUDA_STREAM_BENCHMARK_COMPLETE"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

RUNNER_PATH = PAPER5_ROOT / "scripts" / "05f1_run_final_gate_runtime_pilot_gpu_v2.py"
STREAM_BENCHMARK_PATH = (
    PAPER5_ROOT / "scripts" / "05f1c2_benchmark_single_process_cuda_streams_gpu.py"
)
STREAM_BENCHMARK_DIR = PAPER5_ROOT / "method_contract" / "05f1c2"
STREAM_BENCHMARK_RESULT = (
    STREAM_BENCHMARK_DIR / "single_process_cuda_stream_runtime_benchmark.json"
)
STREAM_BENCHMARK_LOCK = (
    STREAM_BENCHMARK_DIR / "single_process_cuda_stream_runtime_benchmark_lock.json"
)

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05f1d"
OUTPUT_PRELOCK = OUTPUT_DIR / "two_stream_execution_prelock.json"
OUTPUT_MANIFEST = OUTPUT_DIR / "two_stream_full_path_pilot_manifest.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "two_stream_full_path_runtime_summary.json"
OUTPUT_README = OUTPUT_DIR / "05f1d_README.txt"
OUTPUT_LOCK = OUTPUT_DIR / "two_stream_full_path_validation_lock.json"
OUTPUT_IN_PROGRESS = OUTPUT_DIR / ".two_stream_full_path_in_progress.json"
OUTPUT_FAILURE = OUTPUT_DIR / "two_stream_full_path_failure.json"

RESULTS_ROOT = PAPER5_ROOT / "results" / "simulation_benchmark" / "05f_final_gate"
SHARD_DIR = RESULTS_ROOT / "shards" / "_stream2_validated_pilot"


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
        table.to_csv(temporary, index=False)
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


def strip_runtime_only_fields(value: Any) -> Any:
    """Remove only runtime/provenance fields before blind payload equivalence hashing."""
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"wall_seconds", "created_at_utc"}:
                continue
            output[str(key)] = strip_runtime_only_fields(item)
        return output
    if isinstance(value, list):
        return [strip_runtime_only_fields(item) for item in value]
    if isinstance(value, np.ndarray):
        return strip_runtime_only_fields(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def scientific_payload_hash(value: Any) -> str:
    payload = json.dumps(
        strip_runtime_only_fields(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def numpy_bundle_hash(views: dict[str, np.ndarray], availability: np.ndarray) -> str:
    digest = hashlib.sha256()
    for view in sorted(views):
        array = np.asarray(views[view])
        digest.update(view.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
        digest.update(b"\0")
    array = np.asarray(availability)
    digest.update(b"availability\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def state_dict_hash(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def function_source_hash(functions: list[Callable[..., Any]]) -> str:
    chunks: list[str] = []
    for function in functions:
        chunks.append(function.__name__)
        chunks.append(inspect.getsource(function))
    return sha256_text("\n\n".join(chunks))


# -----------------------------------------------------------------------------
# Two-stream execution engine
# -----------------------------------------------------------------------------

def prepare_resample_jobs(
    runner: Any,
    upstream: dict[str, Any],
    arrays: dict[str, np.ndarray],
    stability_group: dict[str, Any],
    selected_epoch: int,
) -> list[dict[str, Any]]:
    """Prepare exactly the frozen 100 resample inputs used by stability_audit.

    This duplicates only the deterministic subset/preprocessing input preparation.
    The canonical helper is still replayed afterward to generate the scientific
    payload; its resample train_fixed_epochs calls are replaced by exact cached
    model states from this two-stream engine.
    """
    neural = upstream["neural"]
    linear = upstream["linear"]
    profile = upstream["profile"]
    n_samples = int(arrays["sample_index"].shape[0])

    rows = stability_group["rows"].copy()
    rows["resample_index"] = pd.to_numeric(
        rows["resample_index"], errors="raise"
    ).astype(int)
    rows["n_subsample"] = pd.to_numeric(
        rows["n_subsample"], errors="raise"
    ).astype(int)
    rows["subsample_seed"] = pd.to_numeric(
        rows["subsample_seed"], errors="raise"
    ).astype(np.int64)
    rows["resample_fit_seed"] = pd.to_numeric(
        rows["resample_fit_seed"], errors="raise"
    ).astype(np.int64)
    rows = rows.sort_values("resample_index", kind="mergesort").reset_index(drop=True)

    if rows.shape[0] != EXPECTED_STABILITY_RESAMPLES:
        raise RuntimeError("Two-stream job preparation requires exactly 100 resamples.")
    if rows["resample_index"].tolist() != list(range(1, EXPECTED_STABILITY_RESAMPLES + 1)):
        raise RuntimeError("Two-stream job preparation resample order is not exactly 1..100.")

    jobs: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        resample_index = int(row["resample_index"])
        n_subsample = int(row["n_subsample"])
        subsample_seed = int(row["subsample_seed"])
        fit_seed = int(row["resample_fit_seed"])

        rng = np.random.default_rng(subsample_seed)
        subset = np.sort(
            rng.choice(n_samples, size=n_subsample, replace=False).astype(int)
        )

        preprocessors = neural.fit_preprocessors(
            linear,
            arrays,
            subset,
            profile,
        )
        views = neural.transform_views(
            arrays,
            preprocessors,
            subset,
            complete=False,
        )
        availability = neural.view_availability(
            arrays,
            preprocessors,
            subset,
        )

        views_copy = {
            key: np.asarray(value).copy()
            for key, value in views.items()
        }
        availability_copy = np.asarray(availability).copy()
        jobs.append(
            {
                "resample_index": resample_index,
                "subsample_seed": subsample_seed,
                "fit_seed": fit_seed,
                "epochs": int(selected_epoch),
                "views": views_copy,
                "availability": availability_copy,
                "input_bundle_hash": numpy_bundle_hash(
                    views_copy,
                    availability_copy,
                ),
            }
        )

        del preprocessors
        del views
        del availability

    fit_seeds = [int(job["fit_seed"]) for job in jobs]
    if len(set(fit_seeds)) != EXPECTED_STABILITY_RESAMPLES:
        raise RuntimeError("Two-stream prepared fit seeds are not unique within dataset.")
    return jobs


def prepare_async_fit_state(
    neural: Any,
    profile: Any,
    job: dict[str, Any],
) -> dict[str, Any]:
    input_dims = {
        view: int(np.asarray(job["views"][view]).shape[1])
        for view in neural.VIEWS
    }
    views = neural.numpy_views_to_cuda(job["views"])
    availability = neural.availability_to_cuda(job["availability"])
    null_mse = neural.calculate_null_mse(views)
    model = neural.construct_model(
        input_dims=input_dims,
        profile=profile,
        seed=int(job["fit_seed"]),
    )
    optimizer = neural.make_optimizer(model, profile)
    return {
        "job": job,
        "input_dims": input_dims,
        "views": views,
        "availability": availability,
        "null_mse": null_mse,
        "model": model,
        "optimizer": optimizer,
        "final_total_tensor": None,
    }


def enqueue_async_epoch(neural: Any, state: dict[str, Any], epoch: int) -> None:
    job = state["job"]
    model = state["model"]
    optimizer = state["optimizer"]

    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, _ = neural.compute_objective(
        model=model,
        views=state["views"],
        availability=state["availability"],
        null_mse=state["null_mse"],
        stochastic_dropout=True,
        fit_seed=int(job["fit_seed"]),
        epoch=int(epoch),
    )
    total.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=float(neural.GRADIENT_CLIP_NORM),
        error_if_nonfinite=True,
    )
    optimizer.step()
    state["final_total_tensor"] = total.detach()


def finalize_async_fit_to_cpu_cache(
    profile: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    job = state["job"]
    model = state["model"]
    final_total_tensor = state["final_total_tensor"]
    if final_total_tensor is None:
        raise RuntimeError("Two-stream fit produced no final objective tensor.")

    final_total = float(final_total_tensor.cpu())
    cpu_state = {
        key: tensor.detach().cpu().clone()
        for key, tensor in model.state_dict().items()
    }
    audit = {
        "fit_stage": "profile_stability_resample",
        "fit_seed": int(job["fit_seed"]),
        "profile_id": profile.profile_id,
        "epochs_run": int(job["epochs"]),
        "selected_epoch": int(job["epochs"]),
        "final_training_objective": float(final_total),
        "wall_seconds": 0.0,
        "device": "cuda:0",
        "fit_status": "PASS",
    }
    return {
        "resample_index": int(job["resample_index"]),
        "fit_seed": int(job["fit_seed"]),
        "epochs": int(job["epochs"]),
        "input_bundle_hash": clean_text(job["input_bundle_hash"]),
        "input_dims": dict(state["input_dims"]),
        "state_dict": cpu_state,
        "state_dict_sha256": state_dict_hash(cpu_state),
        "audit": audit,
        "consumed": False,
    }


def release_async_fit_state(state: dict[str, Any]) -> None:
    state["optimizer"] = None
    state["model"] = None
    state["views"] = None
    state["availability"] = None
    state["null_mse"] = None
    state["final_total_tensor"] = None


def run_two_stream_training_cache(
    neural: Any,
    profile: Any,
    jobs: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], float, int]:
    """Train independent resample fits two-at-a-time in one CUDA context."""
    if len(jobs) != EXPECTED_STABILITY_RESAMPLES:
        raise RuntimeError("Two-stream training cache requires exactly 100 jobs.")

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize(0)

    started = time.perf_counter()
    cache: dict[int, dict[str, Any]] = {}

    for start_index in range(0, len(jobs), STREAM_WIDTH):
        chunk = jobs[start_index : start_index + STREAM_WIDTH]
        states = [
            prepare_async_fit_state(neural, profile, job)
            for job in chunk
        ]

        # All model/input construction is deterministic on the default stream.
        # Complete it before independent training streams consume those tensors.
        torch.cuda.synchronize(0)

        streams = [
            torch.cuda.Stream(device=0)
            for _ in states
        ]
        max_epochs = max(int(state["job"]["epochs"]) for state in states)

        for epoch in range(1, max_epochs + 1):
            for state, stream in zip(states, streams):
                if epoch > int(state["job"]["epochs"]):
                    continue
                with torch.cuda.stream(stream):
                    enqueue_async_epoch(neural, state, epoch)

        for stream in streams:
            stream.synchronize()

        for state in states:
            entry = finalize_async_fit_to_cpu_cache(profile, state)
            fit_seed = int(entry["fit_seed"])
            if fit_seed in cache:
                raise RuntimeError("Duplicate fit seed while constructing two-stream cache.")
            cache[fit_seed] = entry

        for state in states:
            release_async_fit_state(state)
        del states
        del streams
        torch.cuda.empty_cache()
        gc.collect()

    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - started
    peak_bytes = int(torch.cuda.max_memory_allocated(0))

    if len(cache) != EXPECTED_STABILITY_RESAMPLES:
        raise RuntimeError("Two-stream cache does not contain exactly 100 trained resample models.")
    return cache, float(elapsed), peak_bytes


def make_cached_train_fixed_epochs(
    neural: Any,
    profile: Any,
    cache: dict[int, dict[str, Any]],
    original_train: Callable[..., Any],
) -> Callable[..., Any]:
    """Return a train_fixed_epochs replacement backed by exact cached states.

    Reference/full-data fitting remains on the original frozen helper path.
    Only profile_stability_resample calls are served from models already trained
    with the exact two-stream kernel schedule validated in 05f1c2.
    """

    def cached_train_fixed_epochs(
        profile: Any,
        training_views_np: dict[str, np.ndarray],
        availability_np: np.ndarray,
        fit_seed: int,
        epochs: int,
        fit_stage: str,
    ) -> tuple[torch.nn.Module, dict[str, Any]]:
        stage = clean_text(fit_stage)
        seed = int(fit_seed)

        if stage != "profile_stability_resample":
            return original_train(
                profile=profile,
                training_views_np=training_views_np,
                availability_np=availability_np,
                fit_seed=seed,
                epochs=int(epochs),
                fit_stage=fit_stage,
            )

        if seed not in cache:
            raise RuntimeError(
                f"Canonical stability_audit requested unprepared resample fit seed {seed}."
            )
        entry = cache[seed]
        if bool(entry["consumed"]):
            raise RuntimeError(f"Cached resample fit seed {seed} was requested more than once.")
        if clean_text(profile.profile_id) != clean_text(entry["audit"]["profile_id"]):
            raise RuntimeError("Cached resample profile mismatch.")
        if int(epochs) != int(entry["epochs"]):
            raise RuntimeError("Cached resample selected epoch mismatch.")

        observed_hash = numpy_bundle_hash(
            {
                key: np.asarray(value)
                for key, value in training_views_np.items()
            },
            np.asarray(availability_np),
        )
        if observed_hash != clean_text(entry["input_bundle_hash"]):
            raise RuntimeError(
                "Canonical stability_audit resample input differs from precomputed two-stream input."
            )

        input_dims = {
            view: int(np.asarray(training_views_np[view]).shape[1])
            for view in neural.VIEWS
        }
        if input_dims != entry["input_dims"]:
            raise RuntimeError("Cached resample input dimensions mismatch.")

        # construct_model performs the same frozen deterministic model creation
        # that train_fixed_epochs would have performed before optimization.
        model = neural.construct_model(
            input_dims=input_dims,
            profile=profile,
            seed=seed,
        )
        model.load_state_dict(entry["state_dict"], strict=True)
        model.eval()

        if state_dict_hash(model.state_dict()) != clean_text(entry["state_dict_sha256"]):
            raise RuntimeError("GPU-reconstructed cached model state hash mismatch.")

        entry["consumed"] = True
        audit = dict(entry["audit"])
        return model, audit

    return cached_train_fixed_epochs


def evaluate_two_stream_full_path_dataset(
    runner: Any,
    upstream: dict[str, Any],
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    selected_epoch: int,
    stability_group: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute the candidate full scientific path without exposing metrics."""
    neural = upstream["neural"]
    profile = upstream["profile"]

    jobs = prepare_resample_jobs(
        runner=runner,
        upstream=upstream,
        arrays=arrays,
        stability_group=stability_group,
        selected_epoch=selected_epoch,
    )

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize(0)
    total_started = time.perf_counter()

    cache, parallel_training_seconds, cache_peak_bytes = run_two_stream_training_cache(
        neural=neural,
        profile=profile,
        jobs=jobs,
    )

    original_train = neural.train_fixed_epochs
    neural.train_fixed_epochs = make_cached_train_fixed_epochs(
        neural=neural,
        profile=profile,
        cache=cache,
        original_train=original_train,
    )

    replay_started = time.perf_counter()
    try:
        payload = runner.evaluate_final_gate_dataset(
            neural=neural,
            linear=upstream["linear"],
            arrays=arrays,
            metadata=metadata,
            profile=profile,
            selected_epoch=int(selected_epoch),
            stability_group=stability_group,
            stability_thresholds=upstream["stability_thresholds"],
        )
        torch.cuda.synchronize(0)
    finally:
        neural.train_fixed_epochs = original_train
    replay_seconds = time.perf_counter() - replay_started

    unconsumed = [
        int(seed)
        for seed, entry in cache.items()
        if not bool(entry["consumed"])
    ]
    if unconsumed:
        raise RuntimeError(
            f"Canonical stability_audit did not consume {len(unconsumed)} cached resample fits."
        )

    total_seconds = time.perf_counter() - total_started
    peak_bytes = max(
        int(cache_peak_bytes),
        int(torch.cuda.max_memory_allocated(0)),
    )

    # Release CPU model caches only after canonical helper replay is complete.
    for entry in cache.values():
        entry["state_dict"] = {}
    cache.clear()
    del jobs
    gc.collect()
    torch.cuda.empty_cache()

    runtime = {
        "wall_seconds_full_dataset": float(total_seconds),
        "parallel_resample_training_seconds": float(parallel_training_seconds),
        "canonical_helper_replay_seconds": float(replay_seconds),
        "stream_width": STREAM_WIDTH,
        "optimization_runs": 1 + EXPECTED_STABILITY_RESAMPLES,
        "parallelized_resample_fits": EXPECTED_STABILITY_RESAMPLES,
        "sequential_reference_fits": 1,
        "peak_cuda_memory_allocated_bytes": int(peak_bytes),
    }
    return payload, runtime


# -----------------------------------------------------------------------------
# Prospective execution prelock
# -----------------------------------------------------------------------------

def execution_core_hash() -> str:
    return function_source_hash(
        [
            prepare_resample_jobs,
            prepare_async_fit_state,
            enqueue_async_epoch,
            finalize_async_fit_to_cpu_cache,
            release_async_fit_state,
            run_two_stream_training_cache,
            make_cached_train_fixed_epochs,
            evaluate_two_stream_full_path_dataset,
        ]
    )


def verify_stream_benchmark() -> dict[str, Any]:
    if not STREAM_BENCHMARK_PATH.is_file():
        raise FileNotFoundError("05f1c2 benchmark script not found.")
    if not STREAM_BENCHMARK_RESULT.is_file() or not STREAM_BENCHMARK_LOCK.is_file():
        raise FileNotFoundError("Completed 05f1c2 benchmark result/lock is required.")

    lock = load_json(STREAM_BENCHMARK_LOCK)
    if clean_text(lock.get("status")) != EXPECTED_05F1C2_LOCK_STATUS:
        raise RuntimeError("05f1c2 benchmark lock has unexpected status.")
    if clean_text(lock.get("script_sha256")) != sha256_file(STREAM_BENCHMARK_PATH):
        raise RuntimeError("05f1c2 script differs from its completed benchmark hash.")

    result_record = lock.get("result", {})
    result_path = resolve_project_path(result_record.get("path"))
    if result_path.resolve() != STREAM_BENCHMARK_RESULT.resolve():
        raise RuntimeError("05f1c2 lock points to an unexpected result path.")
    if sha256_file(result_path) != clean_text(result_record.get("sha256")):
        raise RuntimeError("05f1c2 result hash mismatch.")

    result = load_json(STREAM_BENCHMARK_RESULT)
    if clean_text(result.get("script_version")) != EXPECTED_05F1C2_SCRIPT_VERSION:
        raise RuntimeError("Unexpected 05f1c2 benchmark script version.")

    recommendation = result.get("recommendation", {})
    if int(recommendation.get("stream_width", -1)) != STREAM_WIDTH:
        raise RuntimeError("05f1c2 did not recommend stream width 2.")

    candidate = None
    for row in result.get("candidates", []):
        if isinstance(row, dict) and int(row.get("stream_width", -1)) == STREAM_WIDTH:
            candidate = row
            break
    if candidate is None:
        raise RuntimeError("05f1c2 lacks the stream-width-2 candidate.")
    if clean_text(candidate.get("status")) != "PASS":
        raise RuntimeError("05f1c2 stream-width-2 candidate did not PASS.")
    equivalence = candidate.get("equivalence", {})
    if not bool(equivalence.get("all_exact", False)):
        raise RuntimeError("05f1c2 stream-width-2 candidate was not exact-equivalent.")

    return {
        "lock": lock,
        "result": result,
        "candidate": candidate,
    }


def build_execution_prelock(
    runner: Any,
    stream_benchmark: dict[str, Any],
) -> dict[str, Any]:
    old_lock = load_json(runner.OUTPUT_LOCK)
    if clean_text(old_lock.get("status")) != EXPECTED_05F1_LOCK_STATUS:
        raise RuntimeError("05f1 completed pilot lock has unexpected status.")
    if clean_text(old_lock.get("script_version")) != EXPECTED_05F1_SCRIPT_VERSION:
        raise RuntimeError("05f1 completed pilot script version mismatch.")
    if clean_text(old_lock.get("script_sha256")) != sha256_file(RUNNER_PATH):
        raise RuntimeError("05f1 runner differs from its completed pilot hash.")

    old_prelock = load_json(runner.OUTPUT_PRELOCK)
    old_epoch_bridge = load_json(runner.OUTPUT_EPOCH_BRIDGE)

    record = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": PRELOCK_STATUS,
        "scientific_contract_hash": clean_text(old_prelock["scientific_contract_hash"]),
        "05a2_amendment_hash": clean_text(old_prelock["05a2_amendment_hash"]),
        "selected_epoch": int(old_epoch_bridge["final_gate_fixed_epoch"]),
        "seed_registry": {
            "path": relative_path(runner.OUTPUT_SEED_REGISTRY),
            "sha256": sha256_file(runner.OUTPUT_SEED_REGISTRY),
            "changed_from_05f1": False,
        },
        "epoch_bridge": {
            "path": relative_path(runner.OUTPUT_EPOCH_BRIDGE),
            "sha256": sha256_file(runner.OUTPUT_EPOCH_BRIDGE),
            "changed_from_05f1": False,
        },
        "execution_path": {
            "name": "single_process_two_cuda_stream_cached_resample_training_plus_canonical_helper_replay",
            "stream_width": STREAM_WIDTH,
            "python_processes": 1,
            "cuda_contexts": 1,
            "reference_fit_execution": "original frozen sequential train_fixed_epochs",
            "resample_fit_execution": (
                "100 independent frozen fixed-epoch fits trained two-at-a-time in CUDA streams; "
                "exact CPU state_dict cache replayed through the canonical stability_audit"
            ),
            "canonical_scientific_aggregation": "unchanged runner.evaluate_final_gate_dataset",
            "AMP": False,
            "TF32": False,
            "torch_compile": False,
            "preprocessing_changed": False,
            "architecture_changed": False,
            "optimizer_changed": False,
            "fit_seeds_changed": False,
            "selected_epoch_changed": False,
            "alignment_changed": False,
            "stability_thresholds_changed": False,
        },
        "runtime_selection_evidence": {
            "05f1c2_script": {
                "path": relative_path(STREAM_BENCHMARK_PATH),
                "sha256": sha256_file(STREAM_BENCHMARK_PATH),
            },
            "05f1c2_result": {
                "path": relative_path(STREAM_BENCHMARK_RESULT),
                "sha256": sha256_file(STREAM_BENCHMARK_RESULT),
            },
            "selected_stream_width": STREAM_WIDTH,
            "selection_used_scientific_metrics": False,
            "candidate_exact_equivalence": True,
            "candidate_speedup_vs_official_sequential": float(
                stream_benchmark["candidate"]["speedup_vs_official_sequential"]
            ),
        },
        "numerical_sources": {
            "05f1_runner": {
                "path": relative_path(RUNNER_PATH),
                "sha256": sha256_file(RUNNER_PATH),
            },
            "neural_helper": {
                "path": relative_path(runner.NEURAL_IMPL_SCRIPT),
                "sha256": sha256_file(runner.NEURAL_IMPL_SCRIPT),
            },
            "linear_helper": {
                "path": relative_path(runner.LINEAR_IMPL_SCRIPT),
                "sha256": sha256_file(runner.LINEAR_IMPL_SCRIPT),
            },
            "two_stream_execution_core_sha256": execution_core_hash(),
        },
        "pilot_validation_requirement": {
            "datasets": EXPECTED_PILOT_DATASETS,
            "optimization_runs": EXPECTED_OPTIMIZATION_RUNS,
            "blind_payload_equivalence_against_05f1_required": True,
            "runtime_fields_stripped_for_equivalence": ["wall_seconds", "created_at_utc"],
            "scientific_metrics_may_be_printed": False,
            "scientific_metrics_may_be_aggregated": False,
            "full_run_authorized_by_prelock_alone": False,
        },
    }

    hash_basis = json.loads(json.dumps(record))
    hash_basis.pop("created_at_utc", None)
    numerical_hash = sha256_text(canonical_json(hash_basis))
    record["two_stream_numerical_implementation_hash"] = numerical_hash
    return record


def initialize_or_verify_prelock(
    runner: Any,
    stream_benchmark: dict[str, Any],
) -> dict[str, Any]:
    proposed = build_execution_prelock(runner, stream_benchmark)
    if OUTPUT_PRELOCK.exists():
        existing = load_json(OUTPUT_PRELOCK)
        existing_core = json.loads(json.dumps(existing))
        proposed_core = json.loads(json.dumps(proposed))
        existing_core.pop("created_at_utc", None)
        proposed_core.pop("created_at_utc", None)
        if canonical_json(existing_core) != canonical_json(proposed_core):
            raise RuntimeError(
                "Existing 05f1d prelock differs from current two-stream implementation. "
                "Refusing to alter a started validation pilot."
            )
        return existing

    # No 05f1d result shard may predate the prospective prelock.
    existing_shards = list(SHARD_DIR.glob("shard_*.json"))
    if existing_shards:
        raise RuntimeError(
            "Two-stream pilot shards exist without a frozen 05f1d prelock."
        )
    write_json_atomic(proposed, OUTPUT_PRELOCK)
    return proposed


# -----------------------------------------------------------------------------
# Dataset execution / blind equivalence
# -----------------------------------------------------------------------------

def new_shard_path(
    metadata: dict[str, Any],
    prelock: dict[str, Any],
) -> Path:
    key = "|".join(
        [
            clean_text(metadata["dataset_key"]),
            clean_text(metadata["input_sha256"]),
            clean_text(prelock["scientific_contract_hash"]),
            clean_text(prelock["two_stream_numerical_implementation_hash"]),
            f"streams={STREAM_WIDTH}",
        ]
    )
    digest = sha256_text(key)[:32]
    return SHARD_DIR / f"shard_{digest}.json"


def load_old_pilot_manifest(runner: Any) -> dict[str, dict[str, Any]]:
    table = pd.read_csv(
        runner.OUTPUT_PILOT_MANIFEST,
        dtype=str,
        low_memory=False,
    ).fillna("")
    if table.shape[0] != EXPECTED_PILOT_DATASETS:
        raise RuntimeError("05f1 pilot manifest no longer contains exactly 3 rows.")

    lookup: dict[str, dict[str, Any]] = {}
    for _, row in table.iterrows():
        dataset_key = clean_text(row["dataset_key"])
        shard_path = resolve_project_path(row["shard_path"])
        if not shard_path.is_file():
            raise FileNotFoundError(f"Frozen 05f1 pilot shard missing: {shard_path}")
        if sha256_file(shard_path) != clean_text(row["shard_sha256"]):
            raise RuntimeError("Frozen 05f1 pilot shard hash mismatch.")
        lookup[dataset_key] = {
            "path": shard_path,
            "sha256": clean_text(row["shard_sha256"]),
        }
    if len(lookup) != EXPECTED_PILOT_DATASETS:
        raise RuntimeError("05f1 pilot manifest dataset keys are not unique.")
    return lookup


def validate_existing_new_shard(
    shard: dict[str, Any],
    metadata: dict[str, Any],
    prelock: dict[str, Any],
) -> None:
    checks = {
        "dataset_key": (
            clean_text(shard.get("dataset_key")),
            clean_text(metadata["dataset_key"]),
        ),
        "input_sha256": (
            clean_text(shard.get("input_sha256")),
            clean_text(metadata["input_sha256"]),
        ),
        "scientific_contract_hash": (
            clean_text(shard.get("scientific_contract_hash")),
            clean_text(prelock["scientific_contract_hash"]),
        ),
        "two_stream_numerical_implementation_hash": (
            clean_text(shard.get("two_stream_numerical_implementation_hash")),
            clean_text(prelock["two_stream_numerical_implementation_hash"]),
        ),
        "stream_width": (
            int(shard.get("stream_width", -1)),
            STREAM_WIDTH,
        ),
        "compute_status": (
            clean_text(shard.get("compute_status")),
            "COMPLETE_EXACT_EQUIVALENT",
        ),
    }
    failures = [
        name
        for name, (observed, expected) in checks.items()
        if observed != expected
    ]
    if failures:
        raise RuntimeError(
            "Existing 05f1d shard provenance mismatch: " + ", ".join(failures)
        )
    equivalence = shard.get("blind_equivalence", {})
    if not bool(equivalence.get("exact_hash_match_after_runtime_field_stripping", False)):
        raise RuntimeError("Existing 05f1d shard lacks exact blind payload equivalence.")
    payload = shard.get("scientific_payload")
    if not isinstance(payload, dict):
        raise RuntimeError("Existing 05f1d shard lacks scientific payload.")


def execute_or_verify_pilot_dataset(
    runner: Any,
    upstream: dict[str, Any],
    prelock: dict[str, Any],
    seed_registry: pd.DataFrame,
    replicate_row: pd.Series,
    selected_epoch: int,
    old_lookup: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    generator = upstream["generator"]
    scenario_id = clean_text(replicate_row["scenario_id"])
    scenario = upstream["scenarios"][scenario_id]
    registry_label = clean_text(replicate_row["registry_label"])
    replicate_index = int(replicate_row["replicate_index"])
    simulation_seed = int(replicate_row["simulation_seed"])

    arrays = generator.generate_dataset(
        scenario,
        registry_label,
        replicate_index,
        simulation_seed,
    )
    validation = runner.validate_final_gate_arrays(
        generator,
        arrays,
        scenario,
        registry_label,
        replicate_index,
        simulation_seed,
    )
    input_hash = runner.array_bundle_sha256(generator, arrays)
    metadata = runner.dataset_metadata(
        scenario,
        replicate_row,
        validation,
        input_hash,
    )
    candidate = new_shard_path(metadata, prelock)

    if candidate.exists():
        shard = load_json(candidate)
        validate_existing_new_shard(shard, metadata, prelock)
        del arrays
        return shard, "verified_existing"

    dataset_key = clean_text(metadata["dataset_key"])
    if dataset_key not in old_lookup:
        raise RuntimeError("05f1d dataset is absent from frozen 05f1 pilot lookup.")
    old_shard = load_json(old_lookup[dataset_key]["path"])
    if clean_text(old_shard.get("dataset_key")) != dataset_key:
        raise RuntimeError("Frozen 05f1 pilot shard dataset key mismatch.")
    if clean_text(old_shard.get("input_sha256")) != input_hash:
        raise RuntimeError("Frozen 05f1 pilot input hash mismatch.")

    dataset_seed_rows = seed_registry[
        seed_registry["dataset_key"].eq(dataset_key)
    ].copy()
    stability_group = runner.build_stability_group(
        dataset_seed_rows,
        int(scenario.n_samples),
    )

    payload, runtime = evaluate_two_stream_full_path_dataset(
        runner=runner,
        upstream=upstream,
        arrays=arrays,
        metadata=metadata,
        selected_epoch=selected_epoch,
        stability_group=stability_group,
    )

    new_payload_hash = scientific_payload_hash(payload)
    old_payload_hash = scientific_payload_hash(old_shard["scientific_payload"])
    exact = bool(new_payload_hash == old_payload_hash)
    if not exact:
        raise RuntimeError(
            "Two-stream full-path scientific payload differs from frozen 05f1 payload "
            "after stripping runtime-only fields."
        )

    shard = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(SCRIPT_PATH),
        "created_at_utc": utc_now_iso(),
        "compute_status": "COMPLETE_EXACT_EQUIVALENT",
        "dataset_key": dataset_key,
        "scenario_id": scenario_id,
        "replicate_index": replicate_index,
        "simulation_seed": simulation_seed,
        "input_sha256": input_hash,
        "scientific_contract_hash": prelock["scientific_contract_hash"],
        "05a2_amendment_hash": prelock["05a2_amendment_hash"],
        "two_stream_numerical_implementation_hash": prelock[
            "two_stream_numerical_implementation_hash"
        ],
        "stream_width": STREAM_WIDTH,
        "PyTorch_version": torch.__version__,
        "CUDA_build": torch.version.cuda,
        "GPU_model": torch.cuda.get_device_name(0),
        "runtime": runtime,
        "blind_equivalence": {
            "frozen_05f1_shard_path": relative_path(old_lookup[dataset_key]["path"]),
            "frozen_05f1_shard_sha256": old_lookup[dataset_key]["sha256"],
            "frozen_payload_hash_after_runtime_field_stripping": old_payload_hash,
            "two_stream_payload_hash_after_runtime_field_stripping": new_payload_hash,
            "exact_hash_match_after_runtime_field_stripping": True,
            "scientific_values_displayed": False,
            "scientific_metrics_aggregated": False,
        },
        "scientific_payload": payload,
    }
    write_json_atomic(shard, candidate)

    serialized = load_json(candidate)
    validate_existing_new_shard(serialized, metadata, prelock)
    del arrays
    return serialized, "generated"


# -----------------------------------------------------------------------------
# Completion / lock
# -----------------------------------------------------------------------------

def build_runtime_summary(manifest: pd.DataFrame) -> dict[str, Any]:
    wall = pd.to_numeric(
        manifest["wall_seconds_full_dataset"],
        errors="raise",
    ).to_numpy(dtype=float)
    parallel = pd.to_numeric(
        manifest["parallel_resample_training_seconds"],
        errors="raise",
    ).to_numpy(dtype=float)
    replay = pd.to_numeric(
        manifest["canonical_helper_replay_seconds"],
        errors="raise",
    ).to_numpy(dtype=float)
    peak = pd.to_numeric(
        manifest["peak_cuda_memory_allocated_bytes"],
        errors="raise",
    ).to_numpy(dtype=np.int64)

    mean_wall = float(np.mean(wall))
    return {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": "RUNTIME_ONLY_TWO_STREAM_FULL_PATH_PILOT_COMPLETE",
        "pilot_datasets": EXPECTED_PILOT_DATASETS,
        "pilot_optimization_runs": EXPECTED_OPTIMIZATION_RUNS,
        "stream_width": STREAM_WIDTH,
        "total_wall_seconds": float(np.sum(wall)),
        "mean_wall_seconds_per_dataset": mean_wall,
        "median_wall_seconds_per_dataset": float(np.median(wall)),
        "mean_parallel_resample_training_seconds": float(np.mean(parallel)),
        "mean_canonical_helper_replay_seconds": float(np.mean(replay)),
        "peak_cuda_memory_allocated_bytes": int(np.max(peak)),
        "rough_full_run_extrapolated_days": float(
            mean_wall * 3_100 / 86_400.0
        ),
        "scientific_metrics_printed": False,
        "scientific_metrics_aggregated": False,
        "final_gate_PASS_FAIL_evaluated": False,
        "blind_exact_equivalence_all_three": bool(
            manifest["exact_payload_equivalence"]
            .astype(str)
            .str.lower()
            .eq("true")
            .all()
        ),
    }


def build_readme(summary: dict[str, Any], prelock: dict[str, Any]) -> str:
    return (
        "Paper 5 05f1d — two-stream full-path validation pilot\n"
        "======================================================\n\n"
        "Purpose\n"
        "-------\n"
        "Validate the single-process two-CUDA-stream resample-training execution path\n"
        "on the same three frozen 05f1 pilot datasets before any 3,100-dataset full run.\n\n"
        "Scientific design\n"
        "-----------------\n"
        "No scientific design element changes. Seeds, selected epoch, preprocessing,\n"
        "architecture, optimizer, alignment, stability thresholds and scientific\n"
        "aggregation remain frozen. Only independent resample neural fits are scheduled\n"
        "two-at-a-time in one CUDA context. The canonical stability_audit is replayed\n"
        "using exact cached model states.\n\n"
        "Validation criterion\n"
        "--------------------\n"
        "For each of the three pilot datasets, the complete scientific payload must hash\n"
        "exactly to the already-frozen 05f1 payload after removing only wall_seconds and\n"
        "created_at_utc. Scientific values are not printed or aggregated.\n\n"
        f"Two-stream numerical implementation hash: "
        f"{prelock['two_stream_numerical_implementation_hash']}\n"
        f"Mean full-path runtime per dataset: "
        f"{summary['mean_wall_seconds_per_dataset']:.3f} s\n"
        f"Rough 3,100-dataset extrapolation: "
        f"{summary['rough_full_run_extrapolated_days']:.3f} days\n"
        f"All-three exact equivalence: "
        f"{summary['blind_exact_equivalence_all_three']}\n"
    )


def verify_existing_lock() -> None:
    lock = load_json(OUTPUT_LOCK)
    if clean_text(lock.get("status")) != LOCK_STATUS:
        raise RuntimeError("Existing 05f1d lock has unexpected status.")
    if clean_text(lock.get("script_sha256")) != sha256_file(SCRIPT_PATH):
        raise RuntimeError("Current 05f1d script differs from its frozen validation hash.")

    summary_record = lock.get("runtime_summary", {})
    summary_path = resolve_project_path(summary_record.get("path"))
    if (
        not summary_path.is_file()
        or sha256_file(summary_path) != clean_text(summary_record.get("sha256"))
    ):
        raise RuntimeError("Existing 05f1d runtime summary hash mismatch.")
    summary = load_json(summary_path)
    if not bool(summary.get("blind_exact_equivalence_all_three", False)):
        raise RuntimeError("Existing 05f1d lock does not certify all-three exact equivalence.")

    print("=" * 88)
    print("Existing Paper 5 05f1d two-stream full-path validation: VERIFIED")
    print("=" * 88)
    print(f"Pilot datasets: {summary['pilot_datasets']}")
    print(f"Stream width: {summary['stream_width']}")
    print(
        f"Mean wall time per dataset: "
        f"{float(summary['mean_wall_seconds_per_dataset']):.1f}s"
    )
    print(
        f"Rough full-run extrapolation: "
        f"{float(summary['rough_full_run_extrapolated_days']):.2f} days"
    )
    print("Blind exact payload equivalence across all three: PASS")
    print("Scientific metrics printed/aggregated: NO")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("No files were rewritten.")
    print("=" * 88)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    print("=" * 88)
    print("Paper 5 - validate two-stream full final-gate execution path")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Pilot datasets: frozen first {EXPECTED_PILOT_DATASETS} 05f1 registry rows")
    print(f"Stability resamples per dataset: {EXPECTED_STABILITY_RESAMPLES}")
    print(f"CUDA stream width: {STREAM_WIDTH}")
    print("Python processes: 1")
    print("CUDA contexts: 1")
    print("Reference fit: frozen sequential helper")
    print("Resample fits: two-at-a-time CUDA streams")
    print("Canonical stability aggregation: frozen helper replay")
    print("Scientific metrics printed: NO")
    print("Scientific metrics aggregated: NO")
    print("Final-gate PASS/FAIL evaluated: NO")
    print("Threshold retuning: FORBIDDEN")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_LOCK.exists():
        verify_existing_lock()
        return

    if not RUNNER_PATH.is_file():
        raise FileNotFoundError("05f1 v2 runner is required.")
    runner = import_module(RUNNER_PATH, "paper5_05f1d_runner")
    if clean_text(getattr(runner, "SCRIPT_VERSION", "")) != EXPECTED_05F1_SCRIPT_VERSION:
        raise RuntimeError("Unexpected 05f1 runner version.")
    if not runner.OUTPUT_LOCK.is_file():
        raise RuntimeError("Completed 05f1 pilot lock is required before 05f1d.")

    stream_benchmark = verify_stream_benchmark()
    upstream = runner.verify_upstream_and_load()
    runner.configure_final_gate_neural_helper(upstream["neural"])

    prelock = initialize_or_verify_prelock(
        runner=runner,
        stream_benchmark=stream_benchmark,
    )

    selected_epoch = int(
        load_json(runner.OUTPUT_EPOCH_BRIDGE)["final_gate_fixed_epoch"]
    )
    if selected_epoch != int(prelock["selected_epoch"]):
        raise RuntimeError("05f1d selected epoch differs from prelock.")

    seed_registry = pd.read_csv(
        runner.OUTPUT_SEED_REGISTRY,
        dtype=str,
        low_memory=False,
    ).fillna("")
    if seed_registry.shape[0] != runner.EXPECTED_DATASETS * EXPECTED_STABILITY_RESAMPLES:
        raise RuntimeError("Frozen final-gate seed registry row count mismatch.")

    old_lookup = load_old_pilot_manifest(runner)

    print("05f1 frozen scientific/numerical lineage: VERIFIED")
    print("05f1c2 stream-width-2 exact benchmark: VERIFIED")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("AMP: OFF | TF32: OFF | torch.compile: OFF")
    print("Two-stream execution prelock: FROZEN BEFORE FIRST 05f1d FIT")
    print(
        "Two-stream numerical implementation hash: "
        + clean_text(prelock["two_stream_numerical_implementation_hash"])
    )
    print("Scientific values remain sealed; only runtime and equivalence status will print.")
    print()

    write_json_atomic(
        {
            "script_version": SCRIPT_VERSION,
            "started_at_utc": utc_now_iso(),
            "status": "IN_PROGRESS",
            "pilot_datasets": EXPECTED_PILOT_DATASETS,
            "optimization_runs": EXPECTED_OPTIMIZATION_RUNS,
            "stream_width": STREAM_WIDTH,
            "scientific_contract_hash": prelock["scientific_contract_hash"],
            "two_stream_numerical_implementation_hash": prelock[
                "two_stream_numerical_implementation_hash"
            ],
            "scientific_console_output": False,
        },
        OUTPUT_IN_PROGRESS,
    )

    pilot = upstream["pilot"].copy()
    for column in ["registry_row_index", "replicate_index", "simulation_seed"]:
        pilot[column] = pd.to_numeric(
            pilot[column],
            errors="raise",
        ).astype(np.int64)
    if pilot.shape[0] != EXPECTED_PILOT_DATASETS:
        raise RuntimeError("Frozen 05f0/05f1 pilot registry no longer contains 3 rows.")

    manifest_rows: list[dict[str, Any]] = []

    try:
        for ordinal, (_, row) in enumerate(pilot.iterrows(), start=1):
            shard, generation_status = execute_or_verify_pilot_dataset(
                runner=runner,
                upstream=upstream,
                prelock=prelock,
                seed_registry=seed_registry,
                replicate_row=row,
                selected_epoch=selected_epoch,
                old_lookup=old_lookup,
            )
            runtime = shard["runtime"]
            candidate = new_shard_path(
                shard["scientific_payload"]["dataset_row"],
                prelock,
            )
            if not candidate.is_file():
                # dataset_row contains the same dataset_key/input_sha256 metadata,
                # but reconstruct directly from top-level fields if needed.
                candidate = new_shard_path(
                    {
                        "dataset_key": shard["dataset_key"],
                        "input_sha256": shard["input_sha256"],
                    },
                    prelock,
                )
            if not candidate.is_file():
                raise RuntimeError("05f1d saved shard cannot be resolved.")

            manifest_rows.append(
                {
                    "pilot_ordinal": ordinal,
                    "dataset_key": clean_text(shard["dataset_key"]),
                    "generation_status": generation_status,
                    "optimization_runs": 1 + EXPECTED_STABILITY_RESAMPLES,
                    "stream_width": STREAM_WIDTH,
                    "wall_seconds_full_dataset": float(
                        runtime["wall_seconds_full_dataset"]
                    ),
                    "parallel_resample_training_seconds": float(
                        runtime["parallel_resample_training_seconds"]
                    ),
                    "canonical_helper_replay_seconds": float(
                        runtime["canonical_helper_replay_seconds"]
                    ),
                    "peak_cuda_memory_allocated_bytes": int(
                        runtime["peak_cuda_memory_allocated_bytes"]
                    ),
                    "exact_payload_equivalence": bool(
                        shard["blind_equivalence"][
                            "exact_hash_match_after_runtime_field_stripping"
                        ]
                    ),
                    "shard_path": relative_path(candidate),
                    "shard_sha256": sha256_file(candidate),
                    "shard_size_bytes": int(candidate.stat().st_size),
                    "scientific_metrics_printed": False,
                    "scientific_metrics_aggregated": False,
                }
            )

            print(
                f"{ordinal:02d}/{EXPECTED_PILOT_DATASETS:02d} "
                f"runtime={float(runtime['wall_seconds_full_dataset']):.1f}s | "
                f"train={float(runtime['parallel_resample_training_seconds']):.1f}s | "
                f"replay={float(runtime['canonical_helper_replay_seconds']):.1f}s | "
                f"VRAM={int(runtime['peak_cuda_memory_allocated_bytes']) / (1024**2):.1f} MiB | "
                f"exact-equivalence=PASS | {generation_status}"
            )

            torch.cuda.empty_cache()
            gc.collect()

    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        write_json_atomic(
            {
                "script_version": SCRIPT_VERSION,
                "created_at_utc": utc_now_iso(),
                "status": "RUNTIME_IMPLEMENTATION_FAILURE_NOT_SCIENTIFIC_GATE_RESULT",
                "error_type": type(exc).__name__,
                "error_message": clean_text(str(exc))[:4000],
                "traceback_tail": "\n".join(
                    traceback.format_exc().splitlines()[-50:]
                ),
                "scientific_metrics_printed": False,
                "scientific_metrics_aggregated": False,
                "final_gate_PASS_FAIL_evaluated": False,
                "threshold_retuning_performed": False,
                "two_stream_numerical_implementation_hash": prelock[
                    "two_stream_numerical_implementation_hash"
                ],
            },
            OUTPUT_FAILURE,
        )
        raise

    manifest = pd.DataFrame(manifest_rows)
    if manifest.shape[0] != EXPECTED_PILOT_DATASETS:
        raise RuntimeError("05f1d manifest does not contain exactly 3 rows.")
    if not manifest["exact_payload_equivalence"].astype(bool).all():
        raise RuntimeError("Not all 05f1d pilot datasets are exact-equivalent.")
    write_csv_atomic(manifest, OUTPUT_MANIFEST)

    summary = build_runtime_summary(manifest)
    if not bool(summary["blind_exact_equivalence_all_three"]):
        raise RuntimeError("05f1d all-three exact-equivalence summary failed.")
    write_json_atomic(summary, OUTPUT_SUMMARY)
    write_text_atomic(build_readme(summary, prelock), OUTPUT_README)

    lock = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "script_path": relative_path(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "created_at_utc": utc_now_iso(),
        "status": LOCK_STATUS,
        "scientific_contract_hash": prelock["scientific_contract_hash"],
        "05a2_amendment_hash": prelock["05a2_amendment_hash"],
        "two_stream_numerical_implementation_hash": prelock[
            "two_stream_numerical_implementation_hash"
        ],
        "stream_width": STREAM_WIDTH,
        "pilot_dataset_count": EXPECTED_PILOT_DATASETS,
        "pilot_optimization_runs": EXPECTED_OPTIMIZATION_RUNS,
        "blind_exact_equivalence_all_three": True,
        "scientific_state": {
            "scientific_metrics_printed": False,
            "scientific_metrics_aggregated": False,
            "D01_D02_D03_evaluated": False,
            "final_gate_PASS_FAIL_evaluated": False,
            "TARGET_accessed": False,
            "outcomes_accessed": False,
            "biology_annotations_accessed": False,
            "threshold_retuning_performed": False,
        },
        "prelock": {
            "path": relative_path(OUTPUT_PRELOCK),
            "sha256": sha256_file(OUTPUT_PRELOCK),
        },
        "runtime_summary": {
            "path": relative_path(OUTPUT_SUMMARY),
            "sha256": sha256_file(OUTPUT_SUMMARY),
        },
        "pilot_manifest": {
            "path": relative_path(OUTPUT_MANIFEST),
            "sha256": sha256_file(OUTPUT_MANIFEST),
        },
        "pilot_shards": {
            clean_text(row["dataset_key"]): {
                "path": clean_text(row["shard_path"]),
                "sha256": clean_text(row["shard_sha256"]),
                "size_bytes": int(float(row["shard_size_bytes"])),
            }
            for _, row in manifest.iterrows()
        },
        "full_run_reuse_rule": (
            "05f1d shards may be reused by the full final-gate run only if 05f2 imports "
            "this exact 05f1d execution engine, verifies this lock and preserves the exact "
            "two_stream_numerical_implementation_hash, scientific contract hash, epoch bridge, "
            "seed registry, PyTorch/CUDA runtime and all upstream helper hashes."
        ),
        "full_run_authorized": True,
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    if OUTPUT_IN_PROGRESS.exists():
        OUTPUT_IN_PROGRESS.unlink()
    if OUTPUT_FAILURE.exists():
        OUTPUT_FAILURE.unlink()

    print()
    print("-" * 88)
    print("Two-stream full-path validation summary (runtime/equivalence only)")
    print("-" * 88)
    print(f"Pilot datasets completed: {EXPECTED_PILOT_DATASETS}")
    print(f"Pilot optimization runs: {EXPECTED_OPTIMIZATION_RUNS}")
    print("Blind exact scientific-payload equivalence: 3/3 PASS")
    print(
        f"Total pilot time: "
        f"{float(summary['total_wall_seconds']) / 60.0:.1f} min"
    )
    print(
        f"Mean wall time per dataset: "
        f"{float(summary['mean_wall_seconds_per_dataset']):.1f}s"
    )
    print(
        f"Peak CUDA allocation: "
        f"{int(summary['peak_cuda_memory_allocated_bytes']) / (1024**2):.1f} MiB"
    )
    print(
        f"Rough 3,100-dataset timing extrapolation: "
        f"{float(summary['rough_full_run_extrapolated_days']):.2f} days"
    )
    print("Scientific metrics printed: NO")
    print("Scientific metrics aggregated: NO")
    print("Final-gate PASS/FAIL evaluated: NO")
    print("Full final-gate run under this unchanged execution hash: AUTHORIZED")
    print()
    print("=" * 88)
    print("Paper 5 05f1d two-stream full-path validation: PASS")
    print("=" * 88)
    print(f"Prelock: {relative_path(OUTPUT_PRELOCK)}")
    print(f"Runtime summary: {relative_path(OUTPUT_SUMMARY)}")
    print(f"Pilot manifest: {relative_path(OUTPUT_MANIFEST)}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("Next: 05f2 resumable 3,100-dataset final-gate streaming run using this exact 05f1d engine.")
    print("=" * 88)


if __name__ == "__main__":
    main()
