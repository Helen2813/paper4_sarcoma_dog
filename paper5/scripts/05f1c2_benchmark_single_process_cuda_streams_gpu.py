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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Preserve the frozen deterministic CUDA GEMM configuration before torch import.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pandas as pd
import torch


SCRIPT_VERSION = "05f1c2-benchmark-single-process-cuda-streams-gpu-v1-no-cli"
CONTRACT_VERSION = "paper5-05f1c2-single-process-cuda-stream-runtime-benchmark-v1"

BENCHMARK_RESAMPLES = 12
CANDIDATE_STREAM_WIDTHS = (1, 2, 4, 8, 12)
EXPECTED_DOMINANT_CATEGORY = "neural_training"
MIN_DOMINANT_FRACTION = 0.95
EXPECTED_05F1_RUNNER_VERSION = "05f1-run-final-gate-runtime-pilot-gpu-v2-seed-collision-aware-no-cli"
EXPECTED_05F1_LOCK_STATUS = "FROZEN_FINAL_GATE_RUNTIME_PILOT_COMPLETE_BEFORE_FULL_RUN"
EXPECTED_05F1A_LOCK_STATUS = "RUNTIME_ONLY_PROFILE_COMPLETE_NO_SCIENTIFIC_OUTPUT"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

RUNNER_PATH = PAPER5_ROOT / "scripts" / "05f1_run_final_gate_runtime_pilot_gpu_v2.py"
PROFILE_PATH = PAPER5_ROOT / "scripts" / "05f1a_profile_final_gate_runtime_breakdown_gpu.py"
PROFILE_DIR = PAPER5_ROOT / "method_contract" / "05f1a"
PROFILE_RESULT = PROFILE_DIR / "final_gate_runtime_breakdown.json"
PROFILE_LOCK = PROFILE_DIR / "final_gate_runtime_breakdown_lock.json"

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05f1c2"
OUTPUT_RESULT = OUTPUT_DIR / "single_process_cuda_stream_runtime_benchmark.json"
OUTPUT_LOCK = OUTPUT_DIR / "single_process_cuda_stream_runtime_benchmark_lock.json"


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


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
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


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


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


def strip_runtime_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): strip_runtime_fields(item)
            for key, item in value.items()
            if key not in {"wall_seconds", "created_at_utc"}
        }
    if isinstance(value, list):
        return [strip_runtime_fields(item) for item in value]
    if isinstance(value, np.ndarray):
        return strip_runtime_fields(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def tensor_state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def encoded_bundle_hash(encoded: dict[str, Any]) -> str:
    digest = hashlib.sha256()

    def add_array(label: str, array: np.ndarray) -> None:
        values = np.asarray(array)
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes(order="C"))
        digest.update(b"\0")

    add_array("patient_shared", np.asarray(encoded["patient_shared"]))
    for section in ("shared_by_view", "private_by_view"):
        mapping = encoded[section]
        for view in sorted(mapping):
            add_array(f"{section}:{view}", np.asarray(mapping[view]))
    return digest.hexdigest()


def official_run_fit(neural: Any, profile: Any, job: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    model, audit = neural.train_fixed_epochs(
        profile=profile,
        training_views_np=job["views"],
        availability_np=job["availability"],
        fit_seed=int(job["fit_seed"]),
        epochs=int(job["epochs"]),
        fit_stage="profile_stability_resample",
    )
    encoded = neural.encode_numpy(
        model,
        job["views"],
        job["availability"],
    )
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - started
    result = {
        "resample_index": int(job["resample_index"]),
        "state_hash": tensor_state_hash(model),
        "encoded_hash": encoded_bundle_hash(encoded),
        "audit": strip_runtime_fields(audit),
        "fit_wall_seconds": float(elapsed),
    }
    del encoded
    del model
    gc.collect()
    return result


def prepare_async_state(
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
        "views": views,
        "availability": availability,
        "null_mse": null_mse,
        "model": model,
        "optimizer": optimizer,
        "final_total_tensor": None,
    }


def enqueue_one_epoch(neural: Any, state: dict[str, Any], epoch: int) -> None:
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


def finalize_async_state(
    neural: Any,
    profile: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    job = state["job"]
    model = state["model"]
    final_total_tensor = state["final_total_tensor"]
    if final_total_tensor is None:
        raise RuntimeError("CUDA-stream fit produced no final objective tensor.")

    final_total = float(final_total_tensor.cpu())
    model.eval()
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
    encoded = neural.encode_numpy(
        model,
        job["views"],
        job["availability"],
    )
    result = {
        "resample_index": int(job["resample_index"]),
        "state_hash": tensor_state_hash(model),
        "encoded_hash": encoded_bundle_hash(encoded),
        "audit": strip_runtime_fields(audit),
    }
    del encoded
    return result


def cleanup_async_state(state: dict[str, Any]) -> None:
    state["optimizer"] = None
    state["model"] = None
    state["views"] = None
    state["availability"] = None
    state["final_total_tensor"] = None


def run_stream_width(
    neural: Any,
    profile: Any,
    jobs: list[dict[str, Any]],
    width: int,
) -> tuple[list[dict[str, Any]], float, int]:
    if width < 1:
        raise ValueError("Stream width must be positive.")

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize(0)

    started = time.perf_counter()
    results: list[dict[str, Any]] = []

    for start_index in range(0, len(jobs), width):
        chunk = jobs[start_index : start_index + width]
        states = [prepare_async_state(neural, profile, job) for job in chunk]

        # Model/input setup is deterministic on the default stream; complete it
        # before independent training streams consume those tensors.
        torch.cuda.synchronize(0)

        streams = [torch.cuda.Stream(device=0) for _ in states]
        max_epochs = max(int(state["job"]["epochs"]) for state in states)

        # Interleave one epoch per fit so tiny kernels from independent fits may
        # overlap in one CUDA context. No fit shares parameters, optimizer state,
        # tensors, or the per-epoch modality-dropout Generator.
        for epoch in range(1, max_epochs + 1):
            for state, stream in zip(states, streams):
                if epoch > int(state["job"]["epochs"]):
                    continue
                with torch.cuda.stream(stream):
                    enqueue_one_epoch(neural, state, epoch)

        for stream in streams:
            stream.synchronize()

        for state in states:
            results.append(finalize_async_state(neural, profile, state))

        for state in states:
            cleanup_async_state(state)
        del states
        del streams
        gc.collect()

    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - started
    peak_bytes = int(torch.cuda.max_memory_allocated(0))
    return results, float(elapsed), peak_bytes


def compare_results(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    base = {int(row["resample_index"]): row for row in baseline}
    cand = {int(row["resample_index"]): row for row in candidate}
    if set(base) != set(cand):
        return {
            "same_resample_set": False,
            "exact_state_hash": False,
            "exact_encoded_bundle": False,
            "exact_fit_audit_without_runtime": False,
            "all_exact": False,
        }

    state_ok = True
    encoded_ok = True
    audit_ok = True
    for index in sorted(base):
        left = base[index]
        right = cand[index]
        state_ok = state_ok and clean_text(left["state_hash"]) == clean_text(right["state_hash"])
        encoded_ok = encoded_ok and clean_text(left["encoded_hash"]) == clean_text(right["encoded_hash"])
        audit_ok = audit_ok and canonical_json(left["audit"]) == canonical_json(right["audit"])

    return {
        "same_resample_set": True,
        "exact_state_hash": bool(state_ok),
        "exact_encoded_bundle": bool(encoded_ok),
        "exact_fit_audit_without_runtime": bool(audit_ok),
        "all_exact": bool(state_ok and encoded_ok and audit_ok),
    }


def verify_existing_lock() -> None:
    lock = load_json(OUTPUT_LOCK)
    if clean_text(lock.get("status")) != "RUNTIME_ONLY_CUDA_STREAM_BENCHMARK_COMPLETE":
        raise RuntimeError("Existing 05f1c2 lock has unexpected status.")
    if clean_text(lock.get("script_sha256")) != sha256_file(SCRIPT_PATH):
        raise RuntimeError("Current 05f1c2 script differs from its frozen benchmark hash.")
    result_record = lock.get("result", {})
    result_path = PROJECT_ROOT / clean_text(result_record.get("path"))
    if not result_path.is_file() or sha256_file(result_path) != clean_text(result_record.get("sha256")):
        raise RuntimeError("Existing 05f1c2 result hash mismatch.")
    result = load_json(OUTPUT_RESULT)
    recommendation = result.get("recommendation", {})
    print("=" * 88)
    print("Existing Paper 5 05f1c2 single-process CUDA-stream runtime benchmark: VERIFIED")
    print("=" * 88)
    print(f"Recommended stream width: {recommendation.get('stream_width', 'NONE')}")
    print(f"Measured speedup: {float(recommendation.get('speedup_vs_official_sequential', 0.0)):.2f}x")
    print("Scientific metrics printed/aggregated: NO")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("No files were rewritten.")
    print("=" * 88)


def main() -> None:
    print("=" * 88)
    print("Paper 5 - single-process CUDA-stream neural-fit runtime benchmark")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Benchmark resamples: first {BENCHMARK_RESAMPLES} from first frozen 05f1 pilot dataset")
    print(f"Candidate CUDA stream widths: {', '.join(str(x) for x in CANDIDATE_STREAM_WIDTHS)}")
    print("Python processes: 1")
    print("CUDA contexts: 1")
    print("Scientific metrics printed: NO")
    print("Scientific metrics aggregated: NO")
    print("Final-gate PASS/FAIL evaluated: NO")
    print("Threshold retuning: FORBIDDEN")
    print("Purpose: runtime-only test after multiprocessing showed limited speedup/resource ceiling")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_LOCK.exists():
        verify_existing_lock()
        return

    if not RUNNER_PATH.is_file() or not PROFILE_RESULT.is_file() or not PROFILE_LOCK.is_file():
        raise FileNotFoundError("05f1 v2 and completed 05f1a profile are required.")

    profile_lock = load_json(PROFILE_LOCK)
    if clean_text(profile_lock.get("status")) != EXPECTED_05F1A_LOCK_STATUS:
        raise RuntimeError("05f1a runtime profile is not frozen COMPLETE.")
    if clean_text(profile_lock.get("script_sha256")) != sha256_file(PROFILE_PATH):
        raise RuntimeError("Current 05f1a profiler differs from its frozen hash.")
    profile_result = load_json(PROFILE_RESULT)
    if clean_text(profile_result.get("dominant_category")) != EXPECTED_DOMINANT_CATEGORY:
        raise RuntimeError("05f1a no longer identifies neural_training as the dominant runtime category.")
    categories = {
        clean_text(row.get("category")): row
        for row in profile_result.get("categories", [])
        if isinstance(row, dict)
    }
    neural_fraction = float(categories[EXPECTED_DOMINANT_CATEGORY]["fraction_of_profiled_wall"])
    if neural_fraction < MIN_DOMINANT_FRACTION:
        raise RuntimeError("05f1a neural_training fraction is below the prespecified optimization trigger.")

    runner = import_module(RUNNER_PATH, "paper5_05f1c_parent_runner")
    if clean_text(getattr(runner, "SCRIPT_VERSION", "")) != EXPECTED_05F1_RUNNER_VERSION:
        raise RuntimeError("Unexpected 05f1 runner version.")
    if not runner.OUTPUT_LOCK.is_file():
        raise RuntimeError("Completed 05f1 pilot lock is required.")
    pilot_lock = runner.load_json(runner.OUTPUT_LOCK)
    if clean_text(pilot_lock.get("status")) != EXPECTED_05F1_LOCK_STATUS:
        raise RuntimeError("05f1 pilot is not frozen COMPLETE.")
    if clean_text(pilot_lock.get("script_sha256")) != sha256_file(RUNNER_PATH):
        raise RuntimeError("05f1 runner differs from its frozen pilot hash.")

    manifest = pd.read_csv(runner.OUTPUT_PILOT_MANIFEST, dtype=str, low_memory=False).fillna("")
    expected_shards = {
        str((PROJECT_ROOT / clean_text(path)).resolve()).lower()
        for path in manifest["shard_path"].tolist()
    }
    observed_shards = {
        str(path.resolve()).lower()
        for path in runner.SHARD_DIR.glob("shard_*.json")
    }
    if observed_shards != expected_shards:
        raise RuntimeError(
            "05f1c2 requires exactly the three frozen pilot shards and no additional full-run shards."
        )

    upstream = runner.verify_upstream_and_load()
    runner.configure_final_gate_neural_helper(upstream["neural"])
    epoch_bridge = runner.load_json(runner.OUTPUT_EPOCH_BRIDGE)
    selected_epoch = int(epoch_bridge["final_gate_fixed_epoch"])
    seed_registry = pd.read_csv(runner.OUTPUT_SEED_REGISTRY, dtype=str, low_memory=False).fillna("")

    pilot = upstream["pilot"].copy()
    for column in ["registry_row_index", "replicate_index", "simulation_seed"]:
        pilot[column] = pd.to_numeric(pilot[column], errors="raise").astype(np.int64)
    first_row = pilot.iloc[0]
    scenario_id = clean_text(first_row["scenario_id"])
    scenario = upstream["scenarios"][scenario_id]
    registry_label = clean_text(first_row["registry_label"])
    replicate_index = int(first_row["replicate_index"])
    simulation_seed = int(first_row["simulation_seed"])

    arrays = upstream["generator"].generate_dataset(
        scenario,
        registry_label,
        replicate_index,
        simulation_seed,
    )
    validation = runner.validate_final_gate_arrays(
        upstream["generator"],
        arrays,
        scenario,
        registry_label,
        replicate_index,
        simulation_seed,
    )
    input_hash = runner.array_bundle_sha256(upstream["generator"], arrays)
    metadata = runner.dataset_metadata(scenario, first_row, validation, input_hash)
    dataset_seed_rows = seed_registry[
        seed_registry["dataset_key"].eq(metadata["dataset_key"])
    ].copy()
    stability_group = runner.build_stability_group(dataset_seed_rows, int(scenario.n_samples))

    neural = upstream["neural"]
    linear = upstream["linear"]
    profile = upstream["profile"]
    n_samples = int(arrays["sample_index"].shape[0])
    jobs: list[dict[str, Any]] = []
    rows = stability_group["rows"].iloc[:BENCHMARK_RESAMPLES].copy()
    for _, registry_row in rows.iterrows():
        resample_index = int(registry_row["resample_index"])
        n_subsample = int(float(registry_row["n_subsample"]))
        subsample_seed = int(float(registry_row["subsample_seed"]))
        fit_seed = int(float(registry_row["resample_fit_seed"]))
        rng = np.random.default_rng(subsample_seed)
        subset = np.sort(
            rng.choice(n_samples, size=n_subsample, replace=False).astype(int)
        )
        preprocessors = neural.fit_preprocessors(linear, arrays, subset, profile)
        views = neural.transform_views(arrays, preprocessors, subset, complete=False)
        availability = neural.view_availability(arrays, preprocessors, subset)
        jobs.append(
            {
                "resample_index": resample_index,
                "fit_seed": fit_seed,
                "epochs": selected_epoch,
                "views": {key: np.asarray(value).copy() for key, value in views.items()},
                "availability": np.asarray(availability).copy(),
            }
        )
        del preprocessors
        del views
        del availability

    print("05f1/05f1a frozen runtime evidence: VERIFIED")
    print(f"05f1a neural_training fraction: {100.0 * neural_fraction:.1f}%")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("AMP: OFF | TF32: OFF | torch.compile: OFF")
    print("Benchmark scientific values will remain sealed and unreported.")
    print()

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize(0)
    baseline_started = time.perf_counter()
    baseline_results = [official_run_fit(neural, profile, job) for job in jobs]
    torch.cuda.synchronize(0)
    baseline_seconds = time.perf_counter() - baseline_started
    print(
        f"Official sequential baseline: {baseline_seconds:.1f}s | "
        f"{BENCHMARK_RESAMPLES / baseline_seconds:.3f} fits/s | exact reference established"
    )

    candidate_records: list[dict[str, Any]] = []

    for width in CANDIDATE_STREAM_WIDTHS:
        try:
            candidate_results, timed_seconds, peak_bytes = run_stream_width(
                neural,
                profile,
                jobs,
                int(width),
            )
            comparison = compare_results(baseline_results, candidate_results)
            speedup = baseline_seconds / timed_seconds
            eligible = bool(comparison["all_exact"])
            if width == 1 and not eligible:
                raise RuntimeError(
                    "Single-stream rewritten training path is not exactly equivalent to "
                    "the frozen helper; wider CUDA-stream candidates are invalid."
                )
            record = {
                "stream_width": int(width),
                "status": "PASS",
                "timed_batch_seconds": float(timed_seconds),
                "fits_per_second": float(BENCHMARK_RESAMPLES / timed_seconds),
                "speedup_vs_official_sequential": float(speedup),
                "peak_cuda_allocated_bytes": int(peak_bytes),
                "equivalence": comparison,
                "eligible_for_recommendation": bool(eligible and width > 1),
            }
            candidate_records.append(record)
            print(
                f"Streams={width}: {timed_seconds:.1f}s | {speedup:.2f}x | "
                f"peak={peak_bytes / (1024**2):.1f} MiB | "
                f"exact numerical equivalence={'PASS' if eligible else 'FAIL'}"
            )
        except Exception as exc:
            torch.cuda.empty_cache()
            gc.collect()
            record = {
                "stream_width": int(width),
                "status": "FAILED_RUNTIME_BENCHMARK",
                "error_type": type(exc).__name__,
                "error_message": clean_text(str(exc))[:2000],
                "eligible_for_recommendation": False,
            }
            candidate_records.append(record)
            print(
                f"Streams={width}: FAILED | {type(exc).__name__}: "
                f"{clean_text(str(exc))[:500]}"
            )
            if width == 1:
                raise

    eligible = [
        row for row in candidate_records
        if row.get("eligible_for_recommendation")
    ]
    original_summary = runner.load_json(runner.OUTPUT_PILOT_SUMMARY)
    original_days = float(original_summary["rough_full_run_extrapolated_days"])

    if eligible:
        best = min(eligible, key=lambda row: float(row["timed_batch_seconds"]))
        recommendation = {
            "status": "RUNTIME_ONLY_RECOMMENDATION_FOR_NEW_VERSIONED_PILOT",
            "stream_width": int(best["stream_width"]),
            "speedup_vs_official_sequential": float(best["speedup_vs_official_sequential"]),
            "original_full_run_extrapolated_days": float(original_days),
            "rough_stream_full_run_days_if_speedup_generalizes": float(
                original_days / float(best["speedup_vs_official_sequential"])
            ),
            "selection_basis": (
                "fastest single-process CUDA-stream candidate with exact model-state, "
                "full-encoding and fit-audit equivalence on the frozen runtime-only benchmark subset"
            ),
            "scientific_metrics_used": False,
            "scientific_thresholds_used": False,
            "automatic_full_run_authorization": False,
            "next_required_action": (
                "if materially faster than the 4-process benchmark, implement a new versioned "
                "full-path 3-dataset pilot under a new execution/numerical hash"
            ),
        }
    else:
        recommendation = {
            "status": "NO_EXACT_MULTI_STREAM_CANDIDATE",
            "stream_width": None,
            "speedup_vs_official_sequential": 0.0,
            "original_full_run_extrapolated_days": float(original_days),
            "selection_basis": "no width > 1 preserved exact implementation outputs",
            "scientific_metrics_used": False,
            "scientific_thresholds_used": False,
            "automatic_full_run_authorization": False,
            "next_required_action": "do not adopt CUDA streams; benchmark another runtime-only optimization",
        }

    result = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": "RUNTIME_ONLY_CUDA_STREAM_BENCHMARK_COMPLETE",
        "scope": {
            "benchmark_dataset_key": metadata["dataset_key"],
            "benchmark_resamples": BENCHMARK_RESAMPLES,
            "candidate_stream_widths": list(CANDIDATE_STREAM_WIDTHS),
            "python_processes": 1,
            "cuda_contexts": 1,
            "official_sequential_seconds": float(baseline_seconds),
            "official_sequential_fits_per_second": float(BENCHMARK_RESAMPLES / baseline_seconds),
            "05f1a_neural_training_fraction": float(neural_fraction),
        },
        "implementation_constraints": {
            "same_frozen_profile": True,
            "same_preprocessing": True,
            "same_fit_seeds": True,
            "same_selected_epoch": True,
            "same_float32_neural_dtype": True,
            "same_AdamW_foreach_false_fused_false": True,
            "same_gradient_clip_norm": float(neural.GRADIENT_CLIP_NORM),
            "same_per_epoch_modality_dropout_seed_derivation": True,
            "AMP": False,
            "TF32": False,
            "torch_compile": False,
            "single_stream_rewrite_exact_to_official_required": True,
            "scientific_metrics_computed": False,
        },
        "candidates": candidate_records,
        "recommendation": recommendation,
        "guardrails": [
            "No D01/D02/D03 or other final-gate scientific metric was computed, printed or aggregated.",
            "No final-gate PASS/FAIL decision was evaluated.",
            "No threshold, seed, epoch, architecture, optimizer, preprocessing or alignment rule was changed.",
            "This benchmark does not authorize the full final-gate run.",
            "Any adopted execution rewrite requires a new versioned 3-dataset full-path pilot before the full run.",
        ],
    }
    write_json_atomic(result, OUTPUT_RESULT)

    lock = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": "RUNTIME_ONLY_CUDA_STREAM_BENCHMARK_COMPLETE",
        "script_sha256": sha256_file(SCRIPT_PATH),
        "result": {
            "path": relative_path(OUTPUT_RESULT),
            "sha256": sha256_file(OUTPUT_RESULT),
        },
        "scientific_metrics_used": False,
        "automatic_full_run_authorization": False,
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    print()
    print("-" * 88)
    print("CUDA-stream runtime benchmark summary")
    print("-" * 88)
    print(f"Official sequential baseline: {baseline_seconds:.1f}s")
    if recommendation["stream_width"] is None:
        print("Recommended CUDA stream width: NONE")
    else:
        print(f"Recommended CUDA stream width: {recommendation['stream_width']}")
        print(
            f"Measured exact-equivalent speedup: "
            f"{recommendation['speedup_vs_official_sequential']:.2f}x"
        )
        print(
            f"Rough full-run extrapolation if it generalizes: "
            f"{recommendation['rough_stream_full_run_days_if_speedup_generalizes']:.2f} days"
        )
    print("Scientific metrics printed/aggregated: NO")
    print("Final-gate PASS/FAIL evaluated: NO")
    print()
    print("=" * 88)
    print("Paper 5 05f1c2 single-process CUDA-stream runtime benchmark: COMPLETE")
    print("=" * 88)
    print(f"Result: {relative_path(OUTPUT_RESULT)}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("Next: compare runtime-only options; then rerun a new full-path 3-dataset pilot under the chosen execution path.")
    print("=" * 88)


if __name__ == "__main__":
    main()
