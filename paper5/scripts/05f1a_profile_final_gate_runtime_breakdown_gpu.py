from __future__ import annotations

import functools
import gc
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

SCRIPT_VERSION = "05f1a-profile-final-gate-runtime-breakdown-gpu-v1-no-cli"
CONTRACT_VERSION = "paper5-05f1a-final-gate-runtime-profile-v1"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

RUNNER_PATH = PAPER5_ROOT / "scripts" / "05f1_run_final_gate_runtime_pilot_gpu_v2.py"
OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05f1a"
OUTPUT_RESULT = OUTPUT_DIR / "final_gate_runtime_breakdown.json"
OUTPUT_LOCK = OUTPUT_DIR / "final_gate_runtime_breakdown_lock.json"

EXPECTED_RUNNER_VERSION = "05f1-run-final-gate-runtime-pilot-gpu-v2-seed-collision-aware-no-cli"
EXPECTED_PILOT_LOCK_STATUS = "FROZEN_FINAL_GATE_RUNTIME_PILOT_COMPLETE_BEFORE_FULL_RUN"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na"}:
        return ""
    return text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def import_runner() -> Any:
    if not RUNNER_PATH.is_file():
        raise FileNotFoundError(f"Required 05f1 v2 runner not found: {RUNNER_PATH}")
    spec = importlib.util.spec_from_file_location("paper5_05f1_v2_runtime_profiler", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import frozen 05f1 v2 runner.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if clean_text(getattr(module, "SCRIPT_VERSION", "")) != EXPECTED_RUNNER_VERSION:
        raise RuntimeError(
            "05f1a requires the completed 05f1 v2 runner; observed version="
            + clean_text(getattr(module, "SCRIPT_VERSION", ""))
        )
    return module


def strip_runtime_only_fields(value: Any) -> Any:
    """Remove timing/provenance fields before numerical-equivalence hashing.

    The profiler must not print or inspect scientific values. It only verifies
    that the re-executed scientific payload is byte-equivalent after removing
    runtime-only fields whose values necessarily differ on a repeated run.
    """
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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        strip_runtime_only_fields(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class TimingLedger:
    def __init__(self) -> None:
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)

    def add(self, label: str, elapsed: float) -> None:
        self.seconds[label] += float(elapsed)
        self.calls[label] += 1

    def wrap(self, owner: Any, attribute: str, label: str) -> Callable[[], None]:
        original = getattr(owner, attribute)

        @functools.wraps(original)
        def timed(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.add(label, time.perf_counter() - started)

        setattr(owner, attribute, timed)

        def restore() -> None:
            setattr(owner, attribute, original)

        return restore


def main() -> None:
    print("=" * 88)
    print("Paper 5 - GPU final-gate runtime breakdown profiler")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("Scope: first frozen 05f1 pilot dataset only")
    print("Optimization runs: 101 (1 reference + 100 stability resamples)")
    print("Scientific metrics printed: NO")
    print("Scientific metrics aggregated: NO")
    print("Final-gate PASS/FAIL evaluated: NO")
    print("Threshold retuning: FORBIDDEN")
    print()

    if OUTPUT_LOCK.exists():
        lock = load_json(OUTPUT_LOCK)
        if clean_text(lock.get("status")) != "RUNTIME_ONLY_PROFILE_COMPLETE_NO_SCIENTIFIC_OUTPUT":
            raise RuntimeError("Existing 05f1a lock has unexpected status.")
        if clean_text(lock.get("script_sha256")) != sha256_file(SCRIPT_PATH):
            raise RuntimeError("Current 05f1a script differs from frozen profiler hash.")
        result = load_json(OUTPUT_RESULT)
        print("Existing 05f1a runtime profile: VERIFIED")
        print(f"Total profiled wall time: {float(result['total_profiled_wall_seconds']):.1f}s")
        print(f"Dominant category: {result['dominant_category']}")
        print("Scientific metrics printed/aggregated: NO")
        print(f"Lock: {relative_path(OUTPUT_LOCK)}")
        print("=" * 88)
        return

    runner = import_runner()
    if not runner.OUTPUT_LOCK.is_file():
        raise RuntimeError("Completed 05f1 v2 pilot lock is required before runtime profiling.")
    pilot_lock = runner.load_json(runner.OUTPUT_LOCK)
    if clean_text(pilot_lock.get("status")) != EXPECTED_PILOT_LOCK_STATUS:
        raise RuntimeError("05f1 v2 pilot is not frozen COMPLETE.")
    if clean_text(pilot_lock.get("script_version")) != EXPECTED_RUNNER_VERSION:
        raise RuntimeError("05f1 v2 pilot lock runner version mismatch.")
    if clean_text(pilot_lock.get("script_sha256")) != sha256_file(RUNNER_PATH):
        raise RuntimeError("05f1 v2 runner file changed after pilot freeze.")

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

    generation_started = time.perf_counter()
    arrays = upstream["generator"].generate_dataset(
        scenario, registry_label, replicate_index, simulation_seed
    )
    validation = runner.validate_final_gate_arrays(
        upstream["generator"], arrays, scenario, registry_label, replicate_index, simulation_seed
    )
    input_hash = runner.array_bundle_sha256(upstream["generator"], arrays)
    metadata = runner.dataset_metadata(scenario, first_row, validation, input_hash)
    generation_seconds = time.perf_counter() - generation_started

    dataset_seed_rows = seed_registry[seed_registry["dataset_key"].eq(metadata["dataset_key"])].copy()
    stability_group = runner.build_stability_group(dataset_seed_rows, int(scenario.n_samples))

    # Existing frozen pilot shard is used only for blind numerical-equivalence
    # hashing after this repeated execution; no scientific field is printed.
    existing_shard_path = runner.final_gate_shard_path(metadata, selected_epoch, prelock)
    if not existing_shard_path.is_file():
        raise RuntimeError("Expected first frozen 05f1 pilot shard is missing.")
    existing_shard = runner.load_json(existing_shard_path)
    runner.validate_existing_final_gate_shard(existing_shard, metadata, selected_epoch, prelock)

    ledger = TimingLedger()
    restores: list[Callable[[], None]] = []
    neural = upstream["neural"]

    # These are non-overlapping top-level operations in stability_audit.
    for attribute, label in [
        ("fit_preprocessors", "preprocessing_fit"),
        ("transform_views", "preprocessing_transform"),
        ("view_availability", "availability"),
        ("preprocessing_audit_rows", "preprocessing_audit"),
        ("train_fixed_epochs", "neural_training"),
        ("encode_numpy", "encoding"),
        ("align_resample_to_reference", "alignment"),
        ("reference_factor_truth_status", "reference_truth_audit"),
    ]:
        restores.append(ledger.wrap(neural, attribute, label))

    # Final truth audit is a runner-level CPU calculation after stability_audit.
    restores.append(ledger.wrap(runner, "enrich_final_gate_truth_audit", "final_truth_audit"))

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize(0)
    started = time.perf_counter()
    try:
        payload = runner.evaluate_final_gate_dataset(
            neural=neural,
            linear=upstream["linear"],
            arrays=arrays,
            metadata=metadata,
            profile=upstream["profile"],
            selected_epoch=selected_epoch,
            stability_group=stability_group,
            stability_thresholds=upstream["stability_thresholds"],
        )
        torch.cuda.synchronize(0)
    finally:
        total_seconds = time.perf_counter() - started
        for restore in reversed(restores):
            restore()

    peak_bytes = int(torch.cuda.max_memory_allocated(0))

    observed_hash = canonical_hash(payload)
    frozen_hash = canonical_hash(existing_shard["scientific_payload"])
    equivalent = bool(observed_hash == frozen_hash)
    if not equivalent:
        raise RuntimeError(
            "Runtime instrumentation changed the numerical scientific payload; refusing profile freeze."
        )

    categories = []
    for label in sorted(ledger.seconds):
        seconds = float(ledger.seconds[label])
        calls = int(ledger.calls[label])
        categories.append(
            {
                "category": label,
                "calls": calls,
                "seconds": seconds,
                "mean_seconds_per_call": seconds / max(1, calls),
                "fraction_of_profiled_wall": seconds / total_seconds if total_seconds > 0 else 0.0,
            }
        )

    categories_sorted = sorted(categories, key=lambda row: row["seconds"], reverse=True)
    accounted_seconds = float(sum(row["seconds"] for row in categories))
    unaccounted_seconds = max(0.0, float(total_seconds - accounted_seconds))
    dominant = categories_sorted[0]["category"] if categories_sorted else "unclassified"

    result = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": "RUNTIME_ONLY_PROFILE_COMPLETE_NO_SCIENTIFIC_OUTPUT",
        "scope": {
            "dataset_key": metadata["dataset_key"],
            "registry_row_index": int(first_row["registry_row_index"]),
            "optimization_runs": 101,
            "stability_resamples": 100,
            "selected_epoch": selected_epoch,
        },
        "runtime": {
            "input_generation_and_validation_seconds": float(generation_seconds),
            "total_profiled_wall_seconds": float(total_seconds),
            "accounted_category_seconds": accounted_seconds,
            "unaccounted_seconds": unaccounted_seconds,
            "peak_cuda_memory_allocated_bytes": peak_bytes,
        },
        "categories": categories_sorted,
        "dominant_category": dominant,
        "numerical_equivalence": {
            "runtime_fields_stripped": ["wall_seconds", "created_at_utc"],
            "recomputed_payload_hash": observed_hash,
            "frozen_05f1_payload_hash": frozen_hash,
            "exact_hash_match_after_runtime_field_stripping": equivalent,
            "scientific_values_displayed": False,
        },
        "guardrails": {
            "scientific_metrics_printed": False,
            "scientific_metrics_aggregated": False,
            "D01_D02_D03_evaluated": False,
            "final_gate_PASS_FAIL_evaluated": False,
            "TARGET_accessed": False,
            "outcomes_accessed": False,
            "threshold_retuning_performed": False,
            "numerical_path_changed": False,
        },
        "upstream": {
            "05f1_runner_path": relative_path(RUNNER_PATH),
            "05f1_runner_sha256": sha256_file(RUNNER_PATH),
            "05f1_pilot_lock_sha256": sha256_file(runner.OUTPUT_LOCK),
            "numerical_implementation_hash": clean_text(prelock["numerical_implementation_hash"]),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(result, OUTPUT_RESULT)
    lock = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": "RUNTIME_ONLY_PROFILE_COMPLETE_NO_SCIENTIFIC_OUTPUT",
        "script_sha256": sha256_file(SCRIPT_PATH),
        "result": {
            "path": relative_path(OUTPUT_RESULT),
            "sha256": sha256_file(OUTPUT_RESULT),
            "size_bytes": int(OUTPUT_RESULT.stat().st_size),
        },
        "numerical_implementation_hash": clean_text(prelock["numerical_implementation_hash"]),
        "scientific_output": False,
        "rewrite_policy": "Immutable runtime-only diagnostic; no scientific selection or threshold decision may use this profile.",
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    print("05f1 frozen pilot: VERIFIED")
    print("Numerical implementation: UNCHANGED")
    print("Blind scientific-payload equivalence after removing runtime-only fields: PASS")
    print()
    print("-" * 88)
    print("Runtime breakdown (runtime only)")
    print("-" * 88)
    print(f"Total profiled wall time: {total_seconds:.1f}s")
    print(f"Input generation/validation before profile: {generation_seconds:.1f}s")
    for row in categories_sorted:
        print(
            f"{row['category']:<24} {row['seconds']:8.1f}s | "
            f"{100.0 * row['fraction_of_profiled_wall']:5.1f}% | calls={row['calls']}"
        )
    print(f"{'unaccounted_wrapper':<24} {unaccounted_seconds:8.1f}s | {100.0 * unaccounted_seconds / total_seconds:5.1f}%")
    print(f"Peak CUDA allocation: {peak_bytes / (1024**2):.1f} MiB")
    print(f"Dominant category: {dominant}")
    print("Scientific metrics printed/aggregated: NO")
    print("Final-gate PASS/FAIL evaluated: NO")
    print()
    print("=" * 88)
    print("Paper 5 05f1a runtime breakdown profile: COMPLETE")
    print("=" * 88)
    print(f"Result: {relative_path(OUTPUT_RESULT)}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("Next: choose a runtime-only optimization based on the dominant measured category, then rerun a new versioned 3-dataset pilot.")
    print("=" * 88)

    del payload
    del arrays
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
