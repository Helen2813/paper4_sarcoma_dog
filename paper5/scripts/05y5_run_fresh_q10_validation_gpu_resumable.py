#!/usr/bin/env python
from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05y5-run-fresh-q10-validation-gpu-resumable-v1-no-cli"

PRELOCK_STATUS = "FROZEN_FRESH_Q10_VALIDATION_EXECUTION_PRELOCK_BEFORE_FIRST_FIT"
SHARD_STATUS = "SEALED_FRESH_Q10_VALIDATION_DATASET_COMPLETE"
COMPLETION_STATUS = "SEALED_FRESH_Q10_VALIDATION_EXECUTION_COMPLETE_NO_AGGREGATION"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

DIR_05Y3 = PAPER5_ROOT / "method_contract" / "05y3"
CONTRACT_05Y3 = DIR_05Y3 / "prospective_q10_recalibration_contract.json"
API_05Y3 = DIR_05Y3 / "fresh_validation_execution_api_snapshot.json"
LOCK_05Y3 = DIR_05Y3 / "05y3_contract_lock.json"

DIR_05Y4 = PAPER5_ROOT / "method_contract" / "05y4"
DATASETS_05Y4 = DIR_05Y4 / "fresh_q10_validation_dataset_registry.csv"
STABILITY_05Y4 = DIR_05Y4 / "fresh_q10_validation_stability_seed_registry.csv"
MANIFEST_05Y4 = DIR_05Y4 / "fresh_q10_validation_bank_manifest.json"
LOCK_05Y4 = DIR_05Y4 / "05y4_seed_bank_lock.json"

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05y5"
OUTPUT_PRELOCK = OUTPUT_DIR / "05y5_execution_prelock.json"
OUTPUT_PROGRESS = OUTPUT_DIR / "fresh_q10_validation_progress_manifest.csv"
OUTPUT_COMPLETION = OUTPUT_DIR / "05y5_execution_completion_lock.json"
OUTPUT_README = OUTPUT_DIR / "05y5_README.txt"

RESULTS_ROOT = PAPER5_ROOT / "results" / "simulation_benchmark" / "05y5_fresh_q10_validation"
SHARD_DIR = RESULTS_ROOT / "shards"

REGISTRY_LABEL = "05y4_fresh_q10_validation"
EXPECTED_DATASETS = 400
EXPECTED_RESAMPLES = 100
EXPECTED_FITS_PER_DATASET = 101
EXPECTED_PREPROCESSING_ROWS = 303
EXPECTED_EPOCH = 248


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def resolve_project_path(value: Any) -> Path:
    path = Path(str(value).strip())
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".txt", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_csv_atomic(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fields = [
            "bank_row_index",
            "dataset_key",
            "source_scenario_id",
            "fresh_replicate_index",
            "simulation_seed",
            "execution_status",
            "shard_path",
            "shard_sha256",
            "shard_size_bytes",
            "wall_seconds_full_dataset",
            "peak_cuda_memory_allocated_bytes",
            "optimization_runs",
        ]
    else:
        fields = list(rows[0].keys())

    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".csv", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def import_module(path: Path, name: str):
    require_file(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_freezes() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for path in [
        CONTRACT_05Y3,
        API_05Y3,
        LOCK_05Y3,
        DATASETS_05Y4,
        STABILITY_05Y4,
        MANIFEST_05Y4,
        LOCK_05Y4,
    ]:
        require_file(path)

    lock3 = load_json(LOCK_05Y3)
    expected3 = "FROZEN_PROSPECTIVE_Q10_RECALIBRATION_BEFORE_FRESH_BANK"
    if str(lock3.get("status", "")).strip() != expected3:
        raise RuntimeError("05y3 lock status mismatch.")
    if sha256_file(CONTRACT_05Y3).lower() != str(lock3["contract"]["sha256"]).lower():
        raise RuntimeError("05y3 contract hash mismatch.")
    if sha256_file(API_05Y3).lower() != str(lock3["api_snapshot"]["sha256"]).lower():
        raise RuntimeError("05y3 API snapshot hash mismatch.")

    lock4 = load_json(LOCK_05Y4)
    expected4 = "FROZEN_FRESH_Q10_VALIDATION_SEED_BANK_BEFORE_MODEL_EVALUATION"
    if str(lock4.get("status", "")).strip() != expected4:
        raise RuntimeError("05y4 lock status mismatch.")

    checks4 = {
        DATASETS_05Y4: lock4["dataset_registry"]["sha256"],
        STABILITY_05Y4: lock4["stability_registry"]["sha256"],
        MANIFEST_05Y4: lock4["manifest"]["sha256"],
    }
    for path, expected_hash in checks4.items():
        if sha256_file(path).lower() != str(expected_hash).lower():
            raise RuntimeError(f"05y4 frozen file hash mismatch: {path.name}")

    contract = load_json(CONTRACT_05Y3)
    api = load_json(API_05Y3)
    bank_manifest = load_json(MANIFEST_05Y4)

    if int(contract["execution_contract"]["fixed_epoch"]) != EXPECTED_EPOCH:
        raise RuntimeError("05y3 fixed epoch changed.")
    if int(contract["execution_contract"]["stability_resamples_per_dataset"]) != EXPECTED_RESAMPLES:
        raise RuntimeError("05y3 resample count changed.")

    return contract, api, bank_manifest


def load_registries() -> tuple[pd.DataFrame, pd.DataFrame]:
    datasets = pd.read_csv(DATASETS_05Y4, dtype=str, low_memory=False).fillna("")
    stability = pd.read_csv(STABILITY_05Y4, dtype=str, low_memory=False).fillna("")

    if datasets.shape[0] != EXPECTED_DATASETS:
        raise RuntimeError(
            f"05y4 dataset registry has {datasets.shape[0]} rows; expected {EXPECTED_DATASETS}."
        )
    if stability.shape[0] != EXPECTED_DATASETS * EXPECTED_RESAMPLES:
        raise RuntimeError(
            f"05y4 stability registry has {stability.shape[0]} rows; "
            f"expected {EXPECTED_DATASETS * EXPECTED_RESAMPLES}."
        )
    if datasets["dataset_key"].duplicated().any():
        raise RuntimeError("05y4 dataset registry has duplicate dataset_key.")

    grouped = stability.groupby("dataset_key", sort=False).size()
    if grouped.shape[0] != EXPECTED_DATASETS or not grouped.eq(EXPECTED_RESAMPLES).all():
        raise RuntimeError("05y4 stability registry is not exactly 100 rows/dataset.")

    return datasets, stability


def initialize_or_verify_prelock(
    *,
    contract: dict[str, Any],
    api: dict[str, Any],
    runner_path: Path,
    torch: Any,
) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PRELOCK.exists():
        prelock = load_json(OUTPUT_PRELOCK)
        if str(prelock.get("status", "")).strip() != PRELOCK_STATUS:
            raise RuntimeError("Existing 05y5 prelock status mismatch.")
        if str(prelock.get("script_sha256", "")).strip().lower() != sha256_file(SCRIPT_PATH).lower():
            raise RuntimeError("Current 05y5 script differs from frozen prelock.")
        frozen_files = [
            (CONTRACT_05Y3, prelock["05y3_contract"]["sha256"]),
            (LOCK_05Y3, prelock["05y3_lock"]["sha256"]),
            (LOCK_05Y4, prelock["05y4_lock"]["sha256"]),
            (DATASETS_05Y4, prelock["05y4_dataset_registry"]["sha256"]),
            (STABILITY_05Y4, prelock["05y4_stability_registry"]["sha256"]),
            (runner_path, prelock["frozen_05f1_runner"]["sha256"]),
        ]
        for path, expected_hash in frozen_files:
            if sha256_file(path).lower() != str(expected_hash).lower():
                raise RuntimeError(f"05y5 prelock upstream hash mismatch: {path}")
        return prelock

    if any(SHARD_DIR.glob("shard_*.json")):
        raise RuntimeError(
            "05y5 result shards exist without an execution prelock. "
            "Do not adopt untracked shards."
        )

    scientific_hash = sha256_text(
        canonical_json(
            {
                "05y3_contract_sha256": sha256_file(CONTRACT_05Y3),
                "05y4_lock_sha256": sha256_file(LOCK_05Y4),
                "05y4_dataset_registry_sha256": sha256_file(DATASETS_05Y4),
                "05y4_stability_registry_sha256": sha256_file(STABILITY_05Y4),
                "runner_sha256": sha256_file(runner_path),
                "fixed_epoch": EXPECTED_EPOCH,
                "resamples": EXPECTED_RESAMPLES,
                "candidate_q10": float(
                    contract["single_candidate_rule"]["q10_aligned_correlation_min"]
                ),
            }
        )
    )

    prelock = {
        "script_version": SCRIPT_VERSION,
        "script_path": relative_path(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "created_at_utc": utc_now_iso(),
        "status": PRELOCK_STATUS,
        "scientific_execution_hash": scientific_hash,
        "05y3_contract": file_record(CONTRACT_05Y3),
        "05y3_lock": file_record(LOCK_05Y3),
        "05y3_api_snapshot": file_record(API_05Y3),
        "05y4_lock": file_record(LOCK_05Y4),
        "05y4_dataset_registry": file_record(DATASETS_05Y4),
        "05y4_stability_registry": file_record(STABILITY_05Y4),
        "frozen_05f1_runner": file_record(runner_path),
        "fixed_epoch": EXPECTED_EPOCH,
        "stability_resamples_per_dataset": EXPECTED_RESAMPLES,
        "runtime_environment": {
            "PyTorch_version": str(torch.__version__),
            "CUDA_build": str(torch.version.cuda),
            "GPU_model": str(torch.cuda.get_device_name(0)),
        },
        "scientific_state_before_first_fresh_fit": {
            "fresh_datasets_evaluated": 0,
            "fresh_model_fits_performed": 0,
            "fresh_scientific_metrics_printed": False,
            "fresh_scientific_metrics_aggregated": False,
            "candidate_threshold_retuned": False,
            "TARGET_accessed": False,
            "outcomes_accessed": False,
        },
        "rewrite_policy": (
            "This prelock is immutable before the first fresh fit. Any numerical-path, "
            "seed-bank, model, epoch, resample-count, threshold, or endpoint change "
            "requires a new prospective branch."
        ),
    }
    write_json_atomic(prelock, OUTPUT_PRELOCK)
    return prelock


def shard_path(
    *,
    dataset_key: str,
    input_sha256: str,
    prelock: dict[str, Any],
) -> Path:
    key = "|".join(
        [
            dataset_key,
            input_sha256,
            str(EXPECTED_EPOCH),
            str(prelock["scientific_execution_hash"]),
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    return SHARD_DIR / f"shard_{digest}.json"


def validate_shard(
    shard: dict[str, Any],
    *,
    dataset_key: str,
    scenario_id: str,
    replicate_index: int,
    simulation_seed: int,
    input_sha256: str,
    prelock: dict[str, Any],
    profile_shared_dim: int,
) -> None:
    expected = {
        "status": SHARD_STATUS,
        "dataset_key": dataset_key,
        "source_scenario_id": scenario_id,
        "fresh_replicate_index": int(replicate_index),
        "simulation_seed": int(simulation_seed),
        "input_sha256": input_sha256,
        "selected_epoch": EXPECTED_EPOCH,
        "scientific_execution_hash": str(prelock["scientific_execution_hash"]),
    }
    for key, value in expected.items():
        if shard.get(key) != value:
            raise RuntimeError(
                f"05y5 shard provenance mismatch for {key}: "
                f"{shard.get(key)!r} != {value!r}"
            )

    payload = shard.get("scientific_payload")
    if not isinstance(payload, dict):
        raise RuntimeError("05y5 shard lacks scientific_payload.")
    dataset_row = payload.get("dataset_row")
    factor_rows = payload.get("factor_rows")
    fit_audits = payload.get("fit_audits")
    preprocessing_rows = payload.get("preprocessing_rows")

    if not isinstance(dataset_row, dict):
        raise RuntimeError("05y5 shard lacks dataset_row.")
    if not isinstance(factor_rows, list) or len(factor_rows) != profile_shared_dim:
        raise RuntimeError("05y5 shard factor-row count mismatch.")
    if not isinstance(fit_audits, list) or len(fit_audits) != EXPECTED_FITS_PER_DATASET:
        raise RuntimeError("05y5 shard fit-audit count mismatch.")
    if not isinstance(preprocessing_rows, list) or len(preprocessing_rows) != EXPECTED_PREPROCESSING_ROWS:
        raise RuntimeError("05y5 shard preprocessing-row count mismatch.")
    if int(dataset_row.get("n_stability_resamples", -1)) != EXPECTED_RESAMPLES:
        raise RuntimeError("05y5 shard dataset_row resample count mismatch.")
    if int(dataset_row.get("selected_epoch_for_final_gate", -1)) != EXPECTED_EPOCH:
        raise RuntimeError("05y5 shard dataset_row epoch mismatch.")


def execute_or_verify_one(
    *,
    f1: Any,
    upstream: dict[str, Any],
    dataset_row: pd.Series,
    seed_rows: pd.DataFrame,
    prelock: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    generator = upstream["generator"]
    scenario_id = str(dataset_row["source_scenario_id"])
    scenario = upstream["scenarios"][scenario_id]
    replicate_index = int(dataset_row["fresh_replicate_index"])
    simulation_seed = int(dataset_row["simulation_seed"])
    dataset_key = str(dataset_row["dataset_key"])

    arrays = generator.generate_dataset(
        scenario,
        REGISTRY_LABEL,
        replicate_index,
        simulation_seed,
    )

    validation = f1.validate_final_gate_arrays(
        generator,
        arrays,
        scenario,
        REGISTRY_LABEL,
        replicate_index,
        simulation_seed,
    )
    input_hash = f1.array_bundle_sha256(generator, arrays)

    replicate_series = pd.Series(
        {
            "registry_label": REGISTRY_LABEL,
            "registry_row_index": int(dataset_row["bank_row_index"]),
            "dataset_key": dataset_key,
            "scenario_id": scenario_id,
            "replicate_index": replicate_index,
            "simulation_seed": simulation_seed,
        }
    )

    metadata = f1.dataset_metadata(
        scenario,
        replicate_series,
        validation,
        input_hash,
    )

    candidate = shard_path(
        dataset_key=dataset_key,
        input_sha256=input_hash,
        prelock=prelock,
    )

    if candidate.exists():
        shard = load_json(candidate)
        validate_shard(
            shard,
            dataset_key=dataset_key,
            scenario_id=scenario_id,
            replicate_index=replicate_index,
            simulation_seed=simulation_seed,
            input_sha256=input_hash,
            prelock=prelock,
            profile_shared_dim=int(upstream["profile"].shared_dim),
        )
        del arrays
        return shard, "reused"

    stability_group = f1.build_stability_group(
        seed_rows,
        expected_n_samples=int(scenario.n_samples),
    )

    torch = f1.torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    payload = f1.evaluate_final_gate_dataset(
        neural=upstream["neural"],
        linear=upstream["linear"],
        arrays=arrays,
        metadata=metadata,
        profile=upstream["profile"],
        selected_epoch=EXPECTED_EPOCH,
        stability_group=stability_group,
        stability_thresholds=upstream["stability_thresholds"],
    )

    torch.cuda.synchronize()
    wall_seconds = float(time.perf_counter() - started)
    peak_bytes = int(torch.cuda.max_memory_allocated())

    shard = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": SHARD_STATUS,
        "analysis_role": "SEALED_PROSPECTIVE_FRESH_Q10_VALIDATION_INPUT_FOR_05Y6",
        "dataset_key": dataset_key,
        "source_scenario_id": scenario_id,
        "fresh_replicate_index": replicate_index,
        "simulation_seed": simulation_seed,
        "registry_label": REGISTRY_LABEL,
        "input_mode": "deterministic_in_memory_stream",
        "input_sha256": input_hash,
        "selected_epoch": EXPECTED_EPOCH,
        "scientific_execution_hash": str(prelock["scientific_execution_hash"]),
        "05y3_contract_sha256": sha256_file(CONTRACT_05Y3),
        "05y4_seed_bank_lock_sha256": sha256_file(LOCK_05Y4),
        "frozen_05f1_runner_sha256": sha256_file(resolve_project_path(
            load_json(API_05Y3)["runner_path"]
        )),
        "PyTorch_version": str(torch.__version__),
        "CUDA_build": str(torch.version.cuda),
        "GPU_model": str(torch.cuda.get_device_name(0)),
        "runtime": {
            "wall_seconds_full_dataset": wall_seconds,
            "peak_cuda_memory_allocated_bytes": peak_bytes,
            "optimization_runs": EXPECTED_FITS_PER_DATASET,
        },
        # No prospective endpoint is aggregated or printed here.
        "scientific_payload": payload,
    }
    write_json_atomic(shard, candidate)

    serialized = load_json(candidate)
    validate_shard(
        serialized,
        dataset_key=dataset_key,
        scenario_id=scenario_id,
        replicate_index=replicate_index,
        simulation_seed=simulation_seed,
        input_sha256=input_hash,
        prelock=prelock,
        profile_shared_dim=int(upstream["profile"].shared_dim),
    )

    del payload
    del arrays
    gc.collect()
    torch.cuda.empty_cache()

    return serialized, "generated"


def progress_row(
    dataset_row: pd.Series,
    shard: dict[str, Any],
    path: Path,
    execution_status: str,
) -> dict[str, Any]:
    runtime = shard["runtime"]
    return {
        "bank_row_index": int(dataset_row["bank_row_index"]),
        "dataset_key": str(dataset_row["dataset_key"]),
        "source_scenario_id": str(dataset_row["source_scenario_id"]),
        "fresh_replicate_index": int(dataset_row["fresh_replicate_index"]),
        "simulation_seed": int(dataset_row["simulation_seed"]),
        "execution_status": execution_status,
        "shard_path": relative_path(path),
        "shard_sha256": sha256_file(path),
        "shard_size_bytes": int(path.stat().st_size),
        "wall_seconds_full_dataset": float(runtime["wall_seconds_full_dataset"]),
        "peak_cuda_memory_allocated_bytes": int(runtime["peak_cuda_memory_allocated_bytes"]),
        "optimization_runs": int(runtime["optimization_runs"]),
    }


def verify_completion_against_progress(
    datasets: pd.DataFrame,
    progress_rows: list[dict[str, Any]],
) -> None:
    if len(progress_rows) != EXPECTED_DATASETS:
        raise RuntimeError("05y5 progress does not contain all 400 datasets.")
    expected_keys = datasets.sort_values("bank_row_index", key=lambda s: pd.to_numeric(s))[
        "dataset_key"
    ].astype(str).tolist()
    observed_keys = [
        str(row["dataset_key"])
        for row in sorted(progress_rows, key=lambda r: int(r["bank_row_index"]))
    ]
    if observed_keys != expected_keys:
        raise RuntimeError("05y5 progress dataset order/membership mismatch.")
    for row in progress_rows:
        path = resolve_project_path(row["shard_path"])
        require_file(path)
        if sha256_file(path).lower() != str(row["shard_sha256"]).lower():
            raise RuntimeError(f"05y5 final shard hash mismatch: {path}")


def main() -> None:
    print("=" * 100)
    print("Paper 5 - prospective fresh q10 validation GPU runner")
    print("=" * 100)
    print(f"Script version: {SCRIPT_VERSION}")
    print("Fresh datasets: 400")
    print("Optimization runs/dataset: 101")
    print("Planned optimization runs: 40,400")
    print("Scientific endpoint printing: FORBIDDEN")
    print("Scientific aggregation: FORBIDDEN")
    print("Threshold retuning: FORBIDDEN")
    print()

    contract, api, _ = verify_freezes()
    runner_path = resolve_project_path(api["runner_path"])
    require_file(runner_path)
    expected_runner_hash = str(api["runner_source"]["sha256"])
    if sha256_file(runner_path).lower() != expected_runner_hash.lower():
        raise RuntimeError("Frozen 05f1 runner hash differs from 05y3 API snapshot.")

    f1 = import_module(runner_path, "paper5_05y5_frozen_05f1_runner")

    if not f1.torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 05y5.")
    if int(api["final_gate_fixed_epoch"]) != EXPECTED_EPOCH:
        raise RuntimeError("05y3 API fixed epoch mismatch.")

    datasets, stability = load_registries()

    prelock = initialize_or_verify_prelock(
        contract=contract,
        api=api,
        runner_path=runner_path,
        torch=f1.torch,
    )

    if OUTPUT_COMPLETION.exists():
        completion = load_json(OUTPUT_COMPLETION)
        if str(completion.get("status", "")).strip() != COMPLETION_STATUS:
            raise RuntimeError("Existing 05y5 completion lock status mismatch.")
        print("05y5 completion lock already exists: VERIFIED")
        print("Scientific endpoint aggregation remains deferred to 05y6.")
        return

    upstream = f1.verify_upstream_and_load()
    f1.configure_final_gate_neural_helper(upstream["neural"])

    # Same deterministic CUDA/alignment fixtures used by the frozen 05f1 path.
    neural_fixture = upstream["neural"].run_reference_fixture(upstream["profile"])
    if str(neural_fixture.get("status", "")).strip() != "PASS":
        raise RuntimeError("Frozen neural CUDA fixture failed before 05y5.")

    rng = np.random.default_rng(20551)
    reference = rng.normal(
        size=(48, int(upstream["profile"].shared_dim))
    ).astype(np.float64)
    permutation = np.arange(int(upstream["profile"].shared_dim))[::-1]
    signs = np.where(
        np.arange(int(upstream["profile"].shared_dim)) % 2 == 0,
        1.0,
        -1.0,
    )
    resample = reference[:, permutation] * signs
    _, correlations, subspace = upstream["neural"].align_resample_to_reference(
        upstream["linear"],
        resample,
        reference,
    )
    if (
        subspace is None
        or float(np.min(correlations)) < 1.0 - 1e-10
        or float(subspace) < 1.0 - 1e-10
    ):
        raise RuntimeError("Frozen alignment fixture failed before 05y5.")

    # Ensure 05y4 scenario IDs exist in the exact frozen 05f1 upstream.
    needed = set(datasets["source_scenario_id"].astype(str))
    missing = needed - set(upstream["scenarios"])
    if missing:
        raise RuntimeError(
            "Fresh validation scenario IDs absent from frozen 05f1 upstream: "
            + ", ".join(sorted(missing))
        )

    datasets = datasets.copy()
    datasets["bank_row_index_num"] = pd.to_numeric(
        datasets["bank_row_index"], errors="raise"
    ).astype(int)
    datasets = datasets.sort_values(
        "bank_row_index_num",
        kind="mergesort",
    ).reset_index(drop=True)

    stability_groups = {
        str(key): group.copy()
        for key, group in stability.groupby("dataset_key", sort=False)
    }

    progress_map: dict[str, dict[str, Any]] = {}
    run_started = time.perf_counter()
    generated_this_run = 0
    reused_this_run = 0
    completed_wall_times: list[float] = []

    for i, dataset_row in datasets.iterrows():
        ordinal = i + 1
        key = str(dataset_row["dataset_key"])
        seed_rows = stability_groups.get(key)
        if seed_rows is None:
            raise RuntimeError(f"Missing 05y4 stability rows for {key}")

        shard, status = execute_or_verify_one(
            f1=f1,
            upstream=upstream,
            dataset_row=dataset_row,
            seed_rows=seed_rows,
            prelock=prelock,
        )

        candidate_path = shard_path(
            dataset_key=key,
            input_sha256=str(shard["input_sha256"]),
            prelock=prelock,
        )

        row = progress_row(
            dataset_row,
            shard,
            candidate_path,
            status,
        )
        progress_map[key] = row

        if status == "generated":
            generated_this_run += 1
        else:
            reused_this_run += 1

        completed_wall_times.append(float(row["wall_seconds_full_dataset"]))

        ordered_progress = [
            progress_map[str(ds["dataset_key"])]
            for _, ds in datasets.iloc[:ordinal].iterrows()
            if str(ds["dataset_key"]) in progress_map
        ]
        write_csv_atomic(ordered_progress, OUTPUT_PROGRESS)

        elapsed = time.perf_counter() - run_started
        mean_wall = float(np.mean(completed_wall_times[-20:]))
        remaining = EXPECTED_DATASETS - ordinal
        eta_seconds = max(0.0, mean_wall * remaining)
        peak_gb = float(row["peak_cuda_memory_allocated_bytes"]) / (1024 ** 3)

        print(
            f"[{ordinal:03d}/{EXPECTED_DATASETS:03d}] "
            f"{status:<9} "
            f"wall={float(row['wall_seconds_full_dataset']):7.1f}s "
            f"peak={peak_gb:5.2f} GiB "
            f"elapsed={elapsed/3600:6.2f}h "
            f"ETA~{eta_seconds/3600:6.2f}h"
        )

    progress_rows = [
        progress_map[str(row["dataset_key"])]
        for _, row in datasets.iterrows()
    ]
    write_csv_atomic(progress_rows, OUTPUT_PROGRESS)
    verify_completion_against_progress(datasets, progress_rows)

    completion = {
        "script_version": SCRIPT_VERSION,
        "script_path": relative_path(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "created_at_utc": utc_now_iso(),
        "status": COMPLETION_STATUS,
        "scientific_endpoint_aggregated": False,
        "scientific_endpoint_printed": False,
        "threshold_retuned": False,
        "fresh_datasets_completed": EXPECTED_DATASETS,
        "optimization_runs_represented": EXPECTED_DATASETS * EXPECTED_FITS_PER_DATASET,
        "generated_this_run": generated_this_run,
        "reused_this_run": reused_this_run,
        "execution_prelock": file_record(OUTPUT_PRELOCK),
        "progress_manifest": file_record(OUTPUT_PROGRESS),
        "05y3_contract": file_record(CONTRACT_05Y3),
        "05y4_lock": file_record(LOCK_05Y4),
        "sealed_shards": {
            str(row["dataset_key"]): {
                "path": str(row["shard_path"]),
                "sha256": str(row["shard_sha256"]),
                "size_bytes": int(row["shard_size_bytes"]),
            }
            for row in progress_rows
        },
        "next_stage": (
            "05y6 may open sealed scientific payloads and aggregate the prospectively "
            "frozen original-vs-q10=.60 endpoints. No candidate retuning is permitted."
        ),
    }
    write_json_atomic(completion, OUTPUT_COMPLETION)

    readme = """Paper 5 05y5 — fresh q10 prospective execution
====================================================

400 fresh datasets were evaluated through the exact frozen 05f1 numerical path:
  200 correlated-private G19-like
  200 STANDARD true-shared

Each dataset:
  1 reference fit + 100 stability resample fits = 101 optimization runs.

Scientific prospective endpoint values are sealed inside per-dataset shards.
05y5 neither prints nor aggregates them.

05y6 is the first stage allowed to aggregate:
  original q10=.40 versus candidate q10=.60.
"""
    write_text_atomic(readme, OUTPUT_README)

    print()
    print("05y5 prospective execution: COMPLETE")
    print(f"Fresh datasets completed: {EXPECTED_DATASETS}")
    print(f"Optimization runs represented: {EXPECTED_DATASETS * EXPECTED_FITS_PER_DATASET:,}")
    print("Scientific endpoint aggregation: NO")
    print("Scientific endpoint printing: NO")
    print(f"Progress:   {relative_path(OUTPUT_PROGRESS)}")
    print(f"Completion: {relative_path(OUTPUT_COMPLETION)}")
    print("Next: run 05y6 aggregation exactly once.")


if __name__ == "__main__":
    main()
