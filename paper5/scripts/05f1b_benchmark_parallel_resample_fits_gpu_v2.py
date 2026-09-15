from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Preserve the frozen deterministic CUDA GEMM configuration before torch import.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pandas as pd
import torch


SCRIPT_VERSION = "05f1b-benchmark-parallel-resample-fits-gpu-v2-side-effect-free-workers-no-cli"
CONTRACT_VERSION = "paper5-05f1b-parallel-resample-runtime-benchmark-v2-side-effect-free-workers"

BENCHMARK_RESAMPLES = 12
CANDIDATE_WORKERS = (2, 4, 8)
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

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05f1b_v2"
OUTPUT_RESULT = OUTPUT_DIR / "parallel_resample_fit_runtime_benchmark_v2.json"
OUTPUT_LOCK = OUTPUT_DIR / "parallel_resample_fit_runtime_benchmark_v2_lock.json"

# Worker-local state. Every spawned process has its own Python, CPU RNG, CUDA
# context and torch RNG state, avoiding cross-fit seed races.
_WORKER_NEURAL: Any | None = None
_WORKER_PROFILE: Any | None = None


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


def run_fit(neural: Any, profile: Any, job: dict[str, Any]) -> dict[str, Any]:
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
    # Synchronize only at the benchmark timing boundary. train_fixed_epochs has
    # already synchronized its final objective, but encode_numpy returns NumPy
    # and therefore also completes the required device transfer.
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - started
    result = {
        "resample_index": int(job["resample_index"]),
        "state_hash": tensor_state_hash(model),
        "patient_shared": np.asarray(encoded["patient_shared"]).copy(),
        "audit": strip_runtime_fields(audit),
        "fit_wall_seconds": float(elapsed),
        "pid": int(os.getpid()),
    }
    del encoded
    del model
    gc.collect()
    return result


def profile_contract_dict(profile: Any) -> dict[str, Any]:
    return {
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
    }


def worker_initializer() -> None:
    global _WORKER_NEURAL, _WORKER_PROFILE
    # Keep CPU helper thread pools small so the benchmark measures GPU fit
    # concurrency rather than accidental CPU oversubscription.
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    # IMPORTANT: worker initialization must be side-effect free. 05f1's full
    # verify_upstream_and_load() replays the 05b deterministic fixture using
    # fixed temporary names (.fixture_a.npz/.fixture_b.npz). That verification
    # is intentionally run once in the parent before spawning workers. Running
    # it concurrently in multiple Windows processes races on those fixed files.
    # Workers therefore reconstruct only the already-verified frozen neural
    # helper/profile from read-only artifacts and independently verify the exact
    # hashes/runtime recorded by the frozen 05f1 prelock.
    runner = import_module(
        RUNNER_PATH,
        f"paper5_05f1b_v2_worker_runner_{os.getpid()}",
    )
    if clean_text(getattr(runner, "SCRIPT_VERSION", "")) != EXPECTED_05F1_RUNNER_VERSION:
        raise RuntimeError("Worker imported an unexpected 05f1 runner version.")

    prelock = load_json(runner.OUTPUT_PRELOCK)
    numerical = load_json(runner.OUTPUT_NUMERICAL_CONTRACT)
    numerical_record = prelock.get("numerical_contract", {})
    expected_numerical_hash = clean_text(numerical_record.get("sha256"))
    if not expected_numerical_hash or sha256_file(runner.OUTPUT_NUMERICAL_CONTRACT) != expected_numerical_hash:
        raise RuntimeError("Worker sees a numerical contract that differs from the frozen 05f1 prelock.")
    if clean_text(numerical.get("numerical_implementation_hash")) != clean_text(prelock.get("numerical_implementation_hash")):
        raise RuntimeError("Worker numerical implementation hash mismatch.")

    runtime = prelock.get("runtime_environment", {})
    if clean_text(torch.__version__) != clean_text(runtime.get("PyTorch_version")):
        raise RuntimeError("Worker PyTorch version differs from the frozen 05f1 prelock.")
    if clean_text(torch.version.cuda) != clean_text(runtime.get("CUDA_build")):
        raise RuntimeError("Worker CUDA build differs from the frozen 05f1 prelock.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("Worker CUDA unavailable; CPU fallback is forbidden.")
    if clean_text(torch.cuda.get_device_name(0)) != clean_text(runtime.get("GPU_model")):
        raise RuntimeError("Worker GPU differs from the frozen 05f1 prelock.")

    neural = import_module(
        runner.NEURAL_IMPL_SCRIPT,
        f"paper5_05f1b_v2_worker_neural_{os.getpid()}",
    )
    if clean_text(getattr(neural, "SCRIPT_VERSION", "")) != clean_text(runner.EXPECTED_NEURAL_SCRIPT_VERSION):
        raise RuntimeError("Worker imported an unexpected frozen neural helper version.")
    expected_neural_hash = clean_text(
        numerical.get("source_hashes", {}).get("neural_syncfree_helper")
    )
    if not expected_neural_hash or sha256_file(runner.NEURAL_IMPL_SCRIPT) != expected_neural_hash:
        raise RuntimeError("Worker neural helper hash differs from the frozen 05f1 numerical contract.")

    profile_table = pd.read_csv(
        runner.PROFILE_GRID_05D1, dtype=str, low_memory=False
    ).fillna("")
    selected = profile_table[
        profile_table["profile_id"].eq(runner.EXPECTED_PROFILE_ID)
    ]
    if selected.shape[0] != 1:
        raise RuntimeError("Worker could not uniquely reconstruct the frozen neural profile.")
    profile = neural.parse_profile(selected.iloc[0])
    if clean_text(profile.profile_id) != clean_text(runner.EXPECTED_PROFILE_ID):
        raise RuntimeError("Worker parsed profile ID mismatch.")
    if canonical_json(profile_contract_dict(profile)) != canonical_json(numerical.get("profile", {})):
        raise RuntimeError("Worker profile parameters differ from the frozen 05f1 numerical contract.")

    runner.configure_final_gate_neural_helper(neural)
    neural.assert_cuda_runtime()
    _WORKER_NEURAL = neural
    _WORKER_PROFILE = profile

    # Touch/synchronize the worker CUDA context before timed jobs.
    torch.zeros(1, device="cuda:0")
    torch.cuda.synchronize(0)


def worker_probe(_: int) -> dict[str, Any]:
    if _WORKER_NEURAL is None or _WORKER_PROFILE is None:
        raise RuntimeError("Worker initializer did not complete.")
    torch.cuda.synchronize(0)
    return {
        "pid": int(os.getpid()),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def worker_run_job(job: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_NEURAL is None or _WORKER_PROFILE is None:
        raise RuntimeError("Worker initializer did not complete.")
    return run_fit(_WORKER_NEURAL, _WORKER_PROFILE, job)


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
            "exact_patient_shared": False,
            "exact_fit_audit_without_runtime": False,
            "all_exact": False,
        }

    state_ok = True
    latent_ok = True
    audit_ok = True
    for index in sorted(base):
        left = base[index]
        right = cand[index]
        state_ok = state_ok and clean_text(left["state_hash"]) == clean_text(right["state_hash"])
        latent_ok = latent_ok and bool(
            np.array_equal(
                np.asarray(left["patient_shared"]),
                np.asarray(right["patient_shared"]),
            )
        )
        audit_ok = audit_ok and canonical_json(left["audit"]) == canonical_json(right["audit"])

    return {
        "same_resample_set": True,
        "exact_state_hash": bool(state_ok),
        "exact_patient_shared": bool(latent_ok),
        "exact_fit_audit_without_runtime": bool(audit_ok),
        "all_exact": bool(state_ok and latent_ok and audit_ok),
    }


def verify_existing_lock() -> None:
    lock = load_json(OUTPUT_LOCK)
    if clean_text(lock.get("status")) != "RUNTIME_ONLY_PARALLEL_FIT_BENCHMARK_COMPLETE":
        raise RuntimeError("Existing 05f1b lock has unexpected status.")
    if clean_text(lock.get("script_sha256")) != sha256_file(SCRIPT_PATH):
        raise RuntimeError("Current 05f1b script differs from its frozen benchmark hash.")
    result_record = lock.get("result", {})
    result_path = PROJECT_ROOT / clean_text(result_record.get("path"))
    if not result_path.is_file() or sha256_file(result_path) != clean_text(result_record.get("sha256")):
        raise RuntimeError("Existing 05f1b result hash mismatch.")
    result = load_json(OUTPUT_RESULT)
    recommendation = result.get("recommendation", {})
    print("=" * 88)
    print("Existing Paper 5 05f1b parallel-fit runtime benchmark: VERIFIED")
    print("=" * 88)
    print(f"Recommended workers: {recommendation.get('worker_count', 'NONE')}")
    print(f"Measured speedup: {float(recommendation.get('speedup_vs_sequential', 0.0)):.2f}x")
    print("Scientific metrics printed/aggregated: NO")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("No files were rewritten.")
    print("=" * 88)


def main() -> None:
    print("=" * 88)
    print("Paper 5 - GPU parallel-resample neural-fit runtime benchmark v2")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Benchmark resamples: first {BENCHMARK_RESAMPLES} from first frozen 05f1 pilot dataset")
    print(f"Candidate worker counts: {', '.join(str(x) for x in CANDIDATE_WORKERS)}")
    print("Scientific metrics printed: NO")
    print("Scientific metrics aggregated: NO")
    print("Final-gate PASS/FAIL evaluated: NO")
    print("Threshold retuning: FORBIDDEN")
    print("Purpose: runtime-only test of independent-fit concurrency after 05f1a profiling")
    print("Worker initialization: side-effect-free read-only reconstruction; 05b fixture replay parent-only")
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

    runner = import_module(RUNNER_PATH, "paper5_05f1b_parent_runner")
    if clean_text(getattr(runner, "SCRIPT_VERSION", "")) != EXPECTED_05F1_RUNNER_VERSION:
        raise RuntimeError("Unexpected 05f1 runner version.")
    if not runner.OUTPUT_LOCK.is_file():
        raise RuntimeError("Completed 05f1 pilot lock is required.")
    pilot_lock = runner.load_json(runner.OUTPUT_LOCK)
    if clean_text(pilot_lock.get("status")) != EXPECTED_05F1_LOCK_STATUS:
        raise RuntimeError("05f1 pilot is not frozen COMPLETE.")
    if clean_text(pilot_lock.get("script_sha256")) != sha256_file(RUNNER_PATH):
        raise RuntimeError("05f1 runner differs from its frozen pilot hash.")

    # Ensure runtime engineering remains confined to the already-approved pilot
    # scope: no extra full-run shards may exist before this benchmark.
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
            "05f1b requires exactly the three frozen pilot shards and no additional full-run shards."
        )

    upstream = runner.verify_upstream_and_load()
    runner.configure_final_gate_neural_helper(upstream["neural"])
    prelock = runner.load_json(runner.OUTPUT_PRELOCK)
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

    # Build exactly the same resample subsets and preprocessing as stability_audit.
    # Only the expensive independent train_fixed_epochs calls are benchmarked in
    # parallel; no scientific alignment/threshold/aggregation is performed.
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
    print("Benchmark scientific values will remain sealed and unreported.")
    print()

    # Sequential exact baseline on the same prepared jobs.
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize(0)
    baseline_started = time.perf_counter()
    baseline_results = [run_fit(neural, profile, job) for job in jobs]
    torch.cuda.synchronize(0)
    baseline_seconds = time.perf_counter() - baseline_started
    baseline_rate = BENCHMARK_RESAMPLES / baseline_seconds
    print(
        f"Sequential baseline: {baseline_seconds:.1f}s | "
        f"{baseline_rate:.3f} fits/s | exact reference established"
    )

    candidate_records: list[dict[str, Any]] = []
    context = mp.get_context("spawn")

    for workers in CANDIDATE_WORKERS:
        # Each candidate gets a fresh pool so worker CUDA contexts/RNG state are
        # independent. Pool initialization is measured separately because a full
        # 3,100-dataset run would amortize it once over hundreds of thousands of fits.
        pool_started = time.perf_counter()
        with ProcessPoolExecutor(
            max_workers=int(workers),
            mp_context=context,
            initializer=worker_initializer,
        ) as executor:
            # Force worker creation and CUDA-context warmup before timed jobs.
            probes = list(executor.map(worker_probe, range(int(workers)), chunksize=1))
            warmup_seconds = time.perf_counter() - pool_started
            unique_pids = len({int(row["pid"]) for row in probes})

            torch.cuda.synchronize(0)
            batch_started = time.perf_counter()
            candidate_results = list(executor.map(worker_run_job, jobs, chunksize=1))
            timed_seconds = time.perf_counter() - batch_started

        comparison = compare_results(baseline_results, candidate_results)
        rate = BENCHMARK_RESAMPLES / timed_seconds
        speedup = baseline_seconds / timed_seconds
        record = {
            "worker_count_requested": int(workers),
            "worker_processes_observed_during_warmup": int(unique_pids),
            "pool_initialization_and_cuda_warmup_seconds": float(warmup_seconds),
            "timed_batch_seconds": float(timed_seconds),
            "fits_per_second": float(rate),
            "speedup_vs_sequential": float(speedup),
            "equivalence": comparison,
            "eligible_for_recommendation": bool(comparison["all_exact"]),
        }
        candidate_records.append(record)
        print(
            f"Workers={workers}: {timed_seconds:.1f}s | {speedup:.2f}x | "
            f"exact numerical equivalence={'PASS' if comparison['all_exact'] else 'FAIL'}"
        )

    eligible = [row for row in candidate_records if row["eligible_for_recommendation"]]
    if eligible:
        best = min(eligible, key=lambda row: float(row["timed_batch_seconds"]))
        recommended_workers: int | None = int(best["worker_count_requested"])
        recommended_speedup = float(best["speedup_vs_sequential"])
        original_summary = runner.load_json(runner.OUTPUT_PILOT_SUMMARY)
        original_days = float(original_summary["rough_full_run_extrapolated_days"])
        projected_days = original_days / recommended_speedup
        recommendation = {
            "status": "RUNTIME_ONLY_RECOMMENDATION_FOR_NEW_VERSIONED_PILOT",
            "worker_count": recommended_workers,
            "speedup_vs_sequential": recommended_speedup,
            "original_full_run_extrapolated_days": original_days,
            "rough_parallel_full_run_days_if_speedup_generalizes": projected_days,
            "selection_basis": "fastest candidate with exact state/latent/audit equivalence on the frozen runtime-only benchmark subset",
            "scientific_metrics_used": False,
            "scientific_thresholds_used": False,
            "automatic_full_run_authorization": False,
            "next_required_action": "implement a new versioned full-path 3-dataset pilot under a new numerical implementation hash",
        }
    else:
        recommendation = {
            "status": "NO_EXACT_PARALLEL_CANDIDATE",
            "worker_count": None,
            "speedup_vs_sequential": 0.0,
            "selection_basis": "no candidate preserved exact state/latent/audit equivalence",
            "scientific_metrics_used": False,
            "scientific_thresholds_used": False,
            "automatic_full_run_authorization": False,
            "next_required_action": "do not adopt multiprocessing; benchmark another runtime-only optimization",
        }

    result = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": "RUNTIME_ONLY_PARALLEL_FIT_BENCHMARK_COMPLETE",
        "scope": {
            "dataset_key": metadata["dataset_key"],
            "benchmark_resamples": BENCHMARK_RESAMPLES,
            "resample_indices": [int(job["resample_index"]) for job in jobs],
            "selected_epoch": int(selected_epoch),
            "candidate_worker_counts": list(CANDIDATE_WORKERS),
            "operation_benchmarked": "independent fixed-epoch resample neural fits only",
            "preprocessing_reused_identically_before_timed_fit_batch": True,
            "alignment_or_scientific_gate_calculation_performed": False,
        },
        "runtime_trigger": {
            "05f1a_dominant_category": clean_text(profile_result["dominant_category"]),
            "05f1a_neural_training_fraction": neural_fraction,
            "05f1a_total_profiled_wall_seconds": float(
                profile_result["runtime"]["total_profiled_wall_seconds"]
            ),
        },
        "sequential_baseline": {
            "timed_seconds": float(baseline_seconds),
            "fits_per_second": float(baseline_rate),
            "fit_count": BENCHMARK_RESAMPLES,
        },
        "candidates": candidate_records,
        "recommendation": recommendation,
        "guardrails": {
            "scientific_metrics_printed": False,
            "scientific_metrics_aggregated": False,
            "D01_D02_D03_evaluated": False,
            "final_gate_PASS_FAIL_evaluated": False,
            "TARGET_accessed": False,
            "outcomes_accessed": False,
            "threshold_retuning_performed": False,
            "existing_05f1_pilot_shards_rewritten": False,
            "new_final_gate_result_shards_written": False,
            "benchmark_based_only_on_runtime_and_exact_numerical_equivalence": True,
            "worker_initializers_write_upstream_fixture_files": False,
        },
        "provenance": {
            "05f1_runner_path": relative_path(RUNNER_PATH),
            "05f1_runner_sha256": sha256_file(RUNNER_PATH),
            "05f1_lock_sha256": sha256_file(runner.OUTPUT_LOCK),
            "05f1a_result_sha256": sha256_file(PROFILE_RESULT),
            "05f1a_lock_sha256": sha256_file(PROFILE_LOCK),
            "05f1_numerical_implementation_hash": clean_text(prelock["numerical_implementation_hash"]),
            "worker_upstream_verification_mode": "parent full verification once; workers read-only reconstruct frozen neural helper/profile from locked hashes",
            "PyTorch_version": torch.__version__,
            "CUDA_build": torch.version.cuda,
            "GPU_model": torch.cuda.get_device_name(0),
        },
    }
    write_json_atomic(result, OUTPUT_RESULT)

    lock = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": "RUNTIME_ONLY_PARALLEL_FIT_BENCHMARK_COMPLETE",
        "script_path": relative_path(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "result": {
            "path": relative_path(OUTPUT_RESULT),
            "sha256": sha256_file(OUTPUT_RESULT),
            "size_bytes": int(OUTPUT_RESULT.stat().st_size),
        },
        "scientific_output": False,
        "rewrite_policy": (
            "05f1b is an immutable runtime-only engineering benchmark. It cannot alter scientific criteria or authorize "
            "the full final-gate run. Any adopted optimization requires a new versioned full-path 3-dataset pilot and "
            "a new numerical implementation hash before full execution."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    print()
    print("-" * 88)
    print("Parallel neural-fit runtime benchmark summary")
    print("-" * 88)
    print(f"Sequential baseline: {baseline_seconds:.1f}s for {BENCHMARK_RESAMPLES} fits")
    if recommendation["worker_count"] is None:
        print("Recommended parallel worker count: NONE")
        print("Exact-equivalent parallel optimization found: NO")
    else:
        print(f"Recommended parallel worker count: {recommendation['worker_count']}")
        print(f"Measured benchmark speedup: {recommendation['speedup_vs_sequential']:.2f}x")
        print(
            "Rough full-run extrapolation if benchmark speedup generalizes: "
            f"{recommendation['rough_parallel_full_run_days_if_speedup_generalizes']:.2f} days"
        )
        print("Exact-equivalent parallel optimization found: YES")
    print("Scientific metrics printed/aggregated: NO")
    print("Final-gate PASS/FAIL evaluated: NO")
    print("Existing 05f1 pilot shards modified: NO")
    print()
    print("=" * 88)
    print("Paper 5 05f1b parallel-resample runtime benchmark: COMPLETE")
    print("=" * 88)
    print(f"Result: {relative_path(OUTPUT_RESULT)}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("Next: if an exact candidate is found, implement a new 3-dataset full-path pilot under a new numerical hash.")
    print("=" * 88)

    del baseline_results
    del jobs
    del arrays
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    mp.freeze_support()
    main()
