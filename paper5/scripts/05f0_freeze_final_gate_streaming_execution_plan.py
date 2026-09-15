from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05f0-freeze-final-gate-streaming-execution-plan-v1-no-cli"
CONTRACT_VERSION = "paper5-final-gate-streaming-execution-plan-v1"
STATUS = "FROZEN_FINAL_GATE_STREAMING_EXECUTION_PLAN_BEFORE_RUNTIME_PILOT"

EXPECTED_SCENARIOS = 24
EXPECTED_DATASETS = 3_100
EXPECTED_STABILITY_RESAMPLES = 100
EXPECTED_METHOD_ID = "V4_SABLE_MO"
EXPECTED_PROFILE_ID = "ARCH_COMPACT__REG_WEAK"
EXPECTED_05E_STATUS = "FROZEN_PRIMARY_PROPOSED_REPRESENTATION_METHOD_BEFORE_FINAL_GATE_AND_TARGET"
EXPECTED_05A2_STATUS = "FROZEN_BEFORE_FINAL_GATE_MATERIALIZATION"
EXPECTED_05B_STATUS = "FROZEN_ARCHITECTURE_SCREENING_INPUTS_BEFORE_MODEL_FITTING"
EXPECTED_GENERATOR_VERSION = "sable-mo-simulation-generator-v2"
EXPECTED_GENERATOR_SCRIPT_VERSION = "05b-generate-small-n-multiomics-simulations-v2-no-cli"
EXPECTED_FINAL_STAGE = "final_gate_calibration"
EXPECTED_FINAL_MATERIALIZATION_POLICY = "stream_or_temporary_after_architecture_freeze"

# Operational runtime-pilot size. This is frozen before any final-gate fit/result.
# Each pilot dataset must traverse the complete numerical path: one reference fit
# plus all 100 frozen stability resamples. Three consecutive frozen-registry rows
# provide a small repeatability/timing sample without altering scientific design.
PILOT_DATASET_COUNT = 3

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# -----------------------------------------------------------------------------
# Frozen upstream inputs
# -----------------------------------------------------------------------------

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

DIR_05E = PAPER5_ROOT / "method_contract" / "05e"
FREEZE_05E = DIR_05E / "representation_method_freeze.json"
LOCK_05E = DIR_05E / "representation_method_freeze_lock.json"

# Frozen policy names this location in earlier stages, but the corrected execution
# plan must NOT populate it as a permanent 3,100-file input bank.
FINAL_GATE_DATA_ROOT = PAPER5_ROOT / "simulations" / "sable_mo_v2" / "final_gate_calibration"

# -----------------------------------------------------------------------------
# 05f0 outputs
# -----------------------------------------------------------------------------

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05f0"
OUTPUT_PREFLIGHT = OUTPUT_DIR / "final_gate_streaming_preflight.json"
OUTPUT_PILOT_REGISTRY = OUTPUT_DIR / "final_gate_runtime_pilot_registry.csv"
OUTPUT_PLAN = OUTPUT_DIR / "final_gate_streaming_execution_plan.json"
OUTPUT_README = OUTPUT_DIR / "05f0_streaming_README.txt"
OUTPUT_LOCK = OUTPUT_DIR / "final_gate_streaming_execution_plan_lock.json"

# Names from the withdrawn/incorrect persistent-materialization implementation.
LEGACY_MATERIALIZATION_LOCK = OUTPUT_DIR / "final_gate_calibration_materialization_lock.json"
LEGACY_MATERIALIZATION_MANIFEST = OUTPUT_DIR / "final_gate_calibration_data_manifest.csv"
LEGACY_MATERIALIZATION_IN_PROGRESS = OUTPUT_DIR / ".final_gate_calibration_materialization_in_progress.json"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "na"} else text


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
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


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
        table.to_csv(temporary, index=False, lineterminator="\n")
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


def any_files(path: Path) -> bool:
    return path.is_file() or (path.is_dir() and any(item.is_file() for item in path.rglob("*")))


def import_generator() -> Any:
    require_file(GENERATOR_SCRIPT)
    spec = importlib.util.spec_from_file_location("paper5_05b_v2_generator_for_05f0", GENERATOR_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the frozen 05b v2 generator module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if clean_text(getattr(module, "SCRIPT_VERSION", "")) != EXPECTED_GENERATOR_SCRIPT_VERSION:
        raise RuntimeError("05b v2 generator script version changed.")
    if clean_text(getattr(module, "GENERATOR_VERSION", "")) != EXPECTED_GENERATOR_VERSION:
        raise RuntimeError("05b v2 generator version changed.")
    return module


# -----------------------------------------------------------------------------
# Upstream verification
# -----------------------------------------------------------------------------


def verify_upstream() -> dict[str, Any]:
    for path in [
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
        FREEZE_05E,
        LOCK_05E,
    ]:
        require_file(path)

    addendum = load_json(CONTRACT_05A1)
    lock_05a1 = load_json(LOCK_05A1)
    expected_addendum_hash = clean_text(lock_05a1.get("addendum_sha256"))
    if not expected_addendum_hash or sha256_file(CONTRACT_05A1) != expected_addendum_hash:
        raise RuntimeError("05a1 addendum hash mismatch.")
    verify_generated_files(lock_05a1, "05a1")

    final_gate = addendum.get("final_gate_calibration", {})
    if int(final_gate.get("n_scenarios", -1)) != EXPECTED_SCENARIOS:
        raise RuntimeError("05a1 final-gate scenario count is not 24.")
    if int(final_gate.get("n_planned_datasets", -1)) != EXPECTED_DATASETS:
        raise RuntimeError("05a1 final-gate dataset count is not 3,100.")
    if int(final_gate.get("stability_resamples_per_dataset", -1)) != EXPECTED_STABILITY_RESAMPLES:
        raise RuntimeError("05a1 final-gate stability-resample count is not 100.")
    if bool(final_gate.get("threshold_retuning_after_results", True)):
        raise RuntimeError("05a1 unexpectedly permits final-gate threshold retuning.")

    amendment = load_json(AMENDMENT_05A2)
    runtime_policy = load_json(RUNTIME_POLICY_05A2)
    lock_05a2 = load_json(LOCK_05A2)
    if clean_text(amendment.get("status")) != EXPECTED_05A2_STATUS:
        raise RuntimeError("05a2 amendment is not frozen before final-gate execution.")
    if clean_text(lock_05a2.get("status")) != EXPECTED_05A2_STATUS:
        raise RuntimeError("05a2 lock is not in the expected frozen state.")
    if sha256_file(AMENDMENT_05A2) != clean_text(lock_05a2.get("amendment_sha256")):
        raise RuntimeError("05a2 amendment hash mismatch.")
    verify_generated_files(lock_05a2, "05a2")
    if clean_text(runtime_policy.get("status")) != "FROZEN_BEFORE_FINAL_GATE_RUNTIME_PILOT":
        raise RuntimeError("05a2 runtime-pilot policy is not frozen.")

    pilot_selection = clean_text(runtime_policy.get("pilot_dataset_selection"))
    if "first prespecified dataset ids" not in pilot_selection.lower():
        raise RuntimeError("05a2 runtime policy no longer freezes pilot selection to first registry IDs.")
    visibility = clean_text(runtime_policy.get("pilot_scientific_output_visibility"))
    if "do not print" not in visibility.lower() or "pass/fail" not in visibility.lower():
        raise RuntimeError("05a2 runtime policy no longer forbids scientific pilot output visibility.")

    lock_05b = load_json(LOCK_05B)
    if clean_text(lock_05b.get("status")) != EXPECTED_05B_STATUS:
        raise RuntimeError("Unexpected 05b v2 lock status.")
    verify_generated_files(lock_05b, "05b v2")
    scope_05b = lock_05b.get("scope", {})
    if bool(scope_05b.get("final_gate_calibration_materialized", True)):
        raise RuntimeError("05b v2 lock unexpectedly says final-gate data were already materialized.")
    if int(scope_05b.get("final_gate_registry_rows", -1)) != EXPECTED_DATASETS:
        raise RuntimeError("05b v2 final-gate registry row count differs from 3,100.")

    freeze_05e = load_json(FREEZE_05E)
    lock_05e = load_json(LOCK_05E)
    if clean_text(freeze_05e.get("status")) != EXPECTED_05E_STATUS:
        raise RuntimeError("05e freeze is not in the required state.")
    if clean_text(lock_05e.get("status")) != EXPECTED_05E_STATUS:
        raise RuntimeError("05e lock is not in the required state.")
    verify_generated_files(lock_05e, "05e")
    expected_freeze_hash = clean_text(lock_05e.get("freeze", {}).get("sha256"))
    if not expected_freeze_hash or sha256_file(FREEZE_05E) != expected_freeze_hash:
        raise RuntimeError("05e freeze JSON hash mismatch.")
    if clean_text(lock_05e.get("primary_proposed_method_id")) != EXPECTED_METHOD_ID:
        raise RuntimeError("05e primary proposed method is not V4_SABLE_MO.")
    if clean_text(lock_05e.get("selected_profile_id")) != EXPECTED_PROFILE_ID:
        raise RuntimeError("05e selected profile is not ARCH_COMPACT__REG_WEAK.")

    return {
        "addendum": addendum,
        "amendment": amendment,
        "runtime_policy": runtime_policy,
        "freeze_05e": freeze_05e,
        "hashes": {
            relative_path(CONTRACT_05A1): sha256_file(CONTRACT_05A1),
            relative_path(LOCK_05A1): sha256_file(LOCK_05A1),
            relative_path(FINAL_GATE_SCENARIOS): sha256_file(FINAL_GATE_SCENARIOS),
            relative_path(FINAL_GATE_REPLICATES): sha256_file(FINAL_GATE_REPLICATES),
            relative_path(AMENDMENT_05A2): sha256_file(AMENDMENT_05A2),
            relative_path(RUNTIME_POLICY_05A2): sha256_file(RUNTIME_POLICY_05A2),
            relative_path(LOCK_05A2): sha256_file(LOCK_05A2),
            relative_path(LOCK_05B): sha256_file(LOCK_05B),
            relative_path(FROZEN_GENERATOR_FIXTURE): sha256_file(FROZEN_GENERATOR_FIXTURE),
            relative_path(GENERATOR_SCRIPT): sha256_file(GENERATOR_SCRIPT),
            relative_path(FREEZE_05E): sha256_file(FREEZE_05E),
            relative_path(LOCK_05E): sha256_file(LOCK_05E),
        },
    }


# -----------------------------------------------------------------------------
# Frozen final-gate registry parsing
# -----------------------------------------------------------------------------


def load_final_gate_registries(generator: Any) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, str]]:
    scenarios_table = pd.read_csv(FINAL_GATE_SCENARIOS, dtype=str, low_memory=False).fillna("")
    replicates = pd.read_csv(FINAL_GATE_REPLICATES, dtype=str, low_memory=False).fillna("")

    if scenarios_table.shape[0] != EXPECTED_SCENARIOS:
        raise RuntimeError(f"Final-gate scenario registry has {scenarios_table.shape[0]} rows, expected 24.")
    if replicates.shape[0] != EXPECTED_DATASETS:
        raise RuntimeError(f"Final-gate replicate registry has {replicates.shape[0]} rows, expected 3,100.")
    if scenarios_table["scenario_id"].duplicated().any():
        raise RuntimeError("Duplicate final-gate scenario IDs.")
    if replicates.duplicated(["scenario_id", "replicate_index"]).any():
        raise RuntimeError("Duplicate final-gate scenario/replicate rows.")
    if replicates["simulation_seed"].duplicated().any():
        raise RuntimeError("Duplicate final-gate simulation seeds.")

    if not scenarios_table["simulation_stage"].astype(str).eq(EXPECTED_FINAL_STAGE).all():
        raise RuntimeError("Final-gate scenario registry contains a non-final_gate_calibration stage row.")
    if not replicates["simulation_stage"].astype(str).eq(EXPECTED_FINAL_STAGE).all():
        raise RuntimeError("Final-gate replicate registry contains a non-final_gate_calibration stage row.")

    # This is the critical prospective policy frozen by 05a1. Do not silently
    # reinterpret it as a permanent persist_npz bank.
    if not scenarios_table["materialization_policy"].astype(str).eq(EXPECTED_FINAL_MATERIALIZATION_POLICY).all():
        observed = sorted(set(scenarios_table["materialization_policy"].astype(str)))
        raise RuntimeError("Unexpected final-gate scenario materialization policy: " + ", ".join(observed))
    if not replicates["materialization_policy"].astype(str).eq(EXPECTED_FINAL_MATERIALIZATION_POLICY).all():
        observed = sorted(set(replicates["materialization_policy"].astype(str)))
        raise RuntimeError("Unexpected final-gate replicate materialization policy: " + ", ".join(observed))

    for column in ["scenario_order", "replicate_index", "simulation_seed", "stability_resamples_per_dataset"]:
        replicates[column] = pd.to_numeric(replicates[column], errors="raise").astype("int64")
    if not replicates["stability_resamples_per_dataset"].eq(EXPECTED_STABILITY_RESAMPLES).all():
        raise RuntimeError("Not every final-gate dataset has exactly 100 frozen stability resamples.")

    scenarios: dict[str, Any] = {}
    scenario_hashes: dict[str, str] = {}
    for _, source_row in scenarios_table.iterrows():
        # 05b-v2's parser deliberately guards architecture-screening materialization
        # and therefore accepts only architecture_screening + persist_npz. Use those
        # two execution tokens in-memory solely to reuse its exact scientific-field
        # parser, then restore the *frozen* final-gate stage/policy before any future
        # generation. No CSV or upstream contract is rewritten.
        parser_row = source_row.copy()
        parser_row["simulation_stage"] = "architecture_screening"
        parser_row["materialization_policy"] = "persist_npz"
        parsed = generator.scenario_from_row(parser_row)
        final_scenario = dataclasses.replace(
            parsed,
            simulation_stage=EXPECTED_FINAL_STAGE,
            materialization_policy=EXPECTED_FINAL_MATERIALIZATION_POLICY,
        )
        if clean_text(final_scenario.generator_required_version) != EXPECTED_GENERATOR_VERSION:
            raise RuntimeError(f"{final_scenario.scenario_id}: unexpected generator version.")
        if int(final_scenario.stability_resamples_per_dataset) != EXPECTED_STABILITY_RESAMPLES:
            raise RuntimeError(f"{final_scenario.scenario_id}: stability-resample count differs from 100.")
        if clean_text(final_scenario.materialization_policy) != EXPECTED_FINAL_MATERIALIZATION_POLICY:
            raise RuntimeError(f"{final_scenario.scenario_id}: final policy restoration failed.")
        scenarios[final_scenario.scenario_id] = final_scenario
        scenario_hashes[final_scenario.scenario_id] = sha256_text(canonical_json(final_scenario.canonical_dict()))

    if len(scenarios) != EXPECTED_SCENARIOS:
        raise RuntimeError("Parsed final-gate scenario count is not 24.")

    unknown = sorted(set(replicates["scenario_id"].astype(str)) - set(scenarios))
    if unknown:
        raise RuntimeError("Final-gate replicate registry contains unknown scenarios: " + ", ".join(unknown))

    observed_counts = replicates.groupby("scenario_id").size().to_dict()
    for scenario_id, scenario in scenarios.items():
        observed = int(observed_counts.get(scenario_id, 0))
        if observed != int(scenario.planned_replicates):
            raise RuntimeError(
                f"{scenario_id}: replicate count {observed} does not match frozen planned {scenario.planned_replicates}."
            )

    replicates = replicates.reset_index(drop=True).copy()
    replicates.insert(0, "registry_row_index", np.arange(replicates.shape[0], dtype=np.int64))
    replicates["dataset_key"] = [
        f"{clean_text(row.scenario_id)}|R{int(row.replicate_index):03d}|S{int(row.simulation_seed):010d}"
        for row in replicates.itertuples(index=False)
    ]
    if replicates["dataset_key"].duplicated().any():
        raise RuntimeError("Final-gate dataset keys are not unique.")

    return scenarios, scenarios_table, replicates, scenario_hashes


# -----------------------------------------------------------------------------
# Generator regression / physical-state guards
# -----------------------------------------------------------------------------


def run_generator_regression(generator: Any) -> dict[str, Any]:
    frozen = load_json(FROZEN_GENERATOR_FIXTURE)
    current = generator.run_reference_fixture()
    required_keys = ["scenario_sha256", "npz_sha256", "npz_size_bytes", "array_sha256", "status"]
    mismatches = [key for key in required_keys if current.get(key) != frozen.get(key)]
    if mismatches:
        raise RuntimeError(
            "Current 05b-v2 generator no longer reproduces its frozen reference fixture: " + ", ".join(mismatches)
        )
    return {
        "status": "PASS",
        "generator_script": relative_path(GENERATOR_SCRIPT),
        "generator_script_sha256": sha256_file(GENERATOR_SCRIPT),
        "generator_script_version": clean_text(getattr(generator, "SCRIPT_VERSION", "")),
        "generator_version": clean_text(getattr(generator, "GENERATOR_VERSION", "")),
        "frozen_fixture": relative_path(FROZEN_GENERATOR_FIXTURE),
        "frozen_fixture_sha256": sha256_file(FROZEN_GENERATOR_FIXTURE),
        "reference_npz_sha256": current["npz_sha256"],
        "reference_scenario_sha256": current["scenario_sha256"],
    }


def verify_no_final_gate_input_bank() -> dict[str, Any]:
    existing_files: list[Path] = []
    if FINAL_GATE_DATA_ROOT.exists():
        existing_files = sorted(item for item in FINAL_GATE_DATA_ROOT.rglob("*") if item.is_file())
    if existing_files:
        raise RuntimeError(
            "Permanent/leftover final-gate input files already exist, but the frozen 05a1 policy is "
            f"{EXPECTED_FINAL_MATERIALIZATION_POLICY!r}. Inspect before continuing:\n- "
            + "\n- ".join(relative_path(path) for path in existing_files[:50])
        )

    incompatible_artifacts = [
        path
        for path in [LEGACY_MATERIALIZATION_LOCK, LEGACY_MATERIALIZATION_MANIFEST, LEGACY_MATERIALIZATION_IN_PROGRESS]
        if path.is_file()
    ]
    if incompatible_artifacts:
        raise RuntimeError(
            "Artifacts from the withdrawn persistent-materialization 05f0 implementation exist. "
            "Do not delete them automatically; inspect provenance first:\n- "
            + "\n- ".join(relative_path(path) for path in incompatible_artifacts)
        )

    return {
        "final_gate_data_root_exists": FINAL_GATE_DATA_ROOT.exists(),
        "final_gate_data_file_count": 0,
        "legacy_persistent_materialization_artifacts": 0,
    }


# -----------------------------------------------------------------------------
# Freeze outputs
# -----------------------------------------------------------------------------


def build_pilot_registry(replicates: pd.DataFrame) -> pd.DataFrame:
    if PILOT_DATASET_COUNT < 1 or PILOT_DATASET_COUNT > replicates.shape[0]:
        raise RuntimeError("Invalid frozen runtime-pilot dataset count.")
    pilot = replicates.iloc[:PILOT_DATASET_COUNT].copy()
    pilot.insert(0, "pilot_ordinal", np.arange(1, PILOT_DATASET_COUNT + 1, dtype=np.int64))
    pilot["pilot_role"] = "runtime_only_no_scientific_decision"
    pilot["reference_fits_planned"] = 1
    pilot["stability_resample_fits_planned"] = EXPECTED_STABILITY_RESAMPLES
    pilot["optimization_runs_planned"] = 1 + EXPECTED_STABILITY_RESAMPLES
    return pilot


def build_readme(pilot: pd.DataFrame) -> str:
    pilot_ids = ", ".join(pilot["dataset_key"].astype(str).tolist())
    return "\n".join(
        [
            "Paper 5 - 05f0 final-gate streaming execution plan",
            "=" * 78,
            "",
            f"Script version: {SCRIPT_VERSION}",
            f"Status: {STATUS}",
            "",
            "Correction captured by this stage",
            "---------------------------------",
            "The frozen 05a1 final-gate materialization policy is:",
            f"  {EXPECTED_FINAL_MATERIALIZATION_POLICY}",
            "Therefore Paper 5 will NOT create a permanent 3,100-NPZ final-gate input bank.",
            "The earlier 05f0 persistent-materialization implementation is withdrawn before any final-gate dataset was created.",
            "",
            "Execution model",
            "---------------",
            "Each final-gate dataset is generated deterministically from its frozen registry seed only when needed.",
            "The preferred path is in-memory generation. A temporary NPZ is allowed only as an execution cache,",
            "must not become a frozen scientific input bank, and must be discarded after the validated result shard exists.",
            "Persistent outputs are per-dataset result/provenance shards, not regenerated simulation inputs.",
            "",
            "Frozen scope",
            "------------",
            f"Final-gate scenarios: {EXPECTED_SCENARIOS}",
            f"Final-gate datasets: {EXPECTED_DATASETS}",
            f"Stability resamples per dataset: {EXPECTED_STABILITY_RESAMPLES}",
            f"Full planned reference+resample fits: {EXPECTED_DATASETS * (1 + EXPECTED_STABILITY_RESAMPLES):,}",
            "Scientific thresholds/decisions: unchanged from 05a1 + 05a2",
            "TARGET/outcomes/biology: forbidden",
            "",
            "Runtime pilot",
            "-------------",
            f"Pilot datasets: first {PILOT_DATASET_COUNT} rows in the frozen replicate registry",
            f"Pilot full-path fits: {PILOT_DATASET_COUNT * (1 + EXPECTED_STABILITY_RESAMPLES):,}",
            f"Pilot IDs: {pilot_ids}",
            "Pilot console output may contain timing, VRAM and progress only.",
            "No D01/D02/D03, TPR, FNR, FDP, false-acceptance or PASS/FAIL may be printed/aggregated during pilot execution.",
            "",
            "Numerical versioning",
            "--------------------",
            "05f1 must freeze its numerical implementation hash immediately before the first pilot fit.",
            "Pilot shards may be reused in the full run only if that hash remains unchanged.",
            "Any numerical-path change invalidates and recomputes pilot shards, per frozen 05a2.",
            "",
        ]
    )


def verify_existing_lock() -> None:
    require_file(OUTPUT_LOCK)
    lock = load_json(OUTPUT_LOCK)
    if clean_text(lock.get("status")) != STATUS:
        raise RuntimeError("Unexpected existing 05f0 streaming lock status.")
    if clean_text(lock.get("script_sha256")) != sha256_file(SCRIPT_PATH):
        raise RuntimeError("Current 05f0 script differs from the frozen streaming-plan script hash.")
    verify_generated_files(lock, "05f0 streaming plan")
    if any_files(FINAL_GATE_DATA_ROOT):
        raise RuntimeError("Final-gate input files appeared after the 05f0 streaming-plan freeze.")
    print("=" * 88)
    print("Existing Paper 5 05f0 streaming execution plan: VERIFIED")
    print("=" * 88)
    print(f"Pilot datasets: {lock.get('pilot_dataset_count')}")
    print("Permanent final-gate input bank: ABSENT")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("No files were rewritten.")
    print("=" * 88)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    print("=" * 88)
    print("Paper 5 - freeze final-gate streaming execution plan")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("GPU computation in 05f0: NO")
    print("Model fitting in 05f0: NO")
    print("Final-gate scientific metrics in 05f0: NO")
    print("Final-gate datasets generated in 05f0: NO")
    print("TARGET/outcome/biology access in 05f0: NO")
    print("Threshold retuning in 05f0: FORBIDDEN")
    print()

    if OUTPUT_LOCK.exists():
        verify_existing_lock()
        return

    physical_state = verify_no_final_gate_input_bank()
    upstream = verify_upstream()
    generator = import_generator()
    scenarios, scenarios_table, replicates, scenario_hashes = load_final_gate_registries(generator)
    regression = run_generator_regression(generator)

    print("05a1 frozen final-gate registry: PASS")
    print(f"05a1 final-gate materialization policy: {EXPECTED_FINAL_MATERIALIZATION_POLICY}")
    print("05a2 frozen gate/CI/failure/runtime clarification: PASS")
    print("05b v2 frozen numerical generator replay: PASS")
    print("05e frozen primary proposed V4_SABLE_MO: PASS")
    print(f"Final-gate scenarios: {len(scenarios):,}")
    print(f"Final-gate datasets: {replicates.shape[0]:,}")
    print(f"Stability resamples per dataset: {EXPECTED_STABILITY_RESAMPLES}")
    print("Permanent final-gate input files present: 0")
    print()

    pilot = build_pilot_registry(replicates)

    preflight = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": "PASS_BEFORE_FINAL_GATE_RUNTIME_PILOT",
        "scientific_state": {
            "final_gate_datasets_generated": False,
            "final_gate_model_fitting_performed": False,
            "final_gate_scientific_metrics_evaluated": False,
            "TARGET_accessed": False,
            "outcomes_accessed": False,
            "biology_accessed": False,
            "threshold_retuning_performed": False,
        },
        "registry": {
            "n_scenarios": int(scenarios_table.shape[0]),
            "n_datasets": int(replicates.shape[0]),
            "stability_resamples_per_dataset": EXPECTED_STABILITY_RESAMPLES,
            "simulation_stage": EXPECTED_FINAL_STAGE,
            "materialization_policy": EXPECTED_FINAL_MATERIALIZATION_POLICY,
            "scenario_registry_sha256": sha256_file(FINAL_GATE_SCENARIOS),
            "replicate_registry_sha256": sha256_file(FINAL_GATE_REPLICATES),
            "scenario_canonical_hash_registry_sha256": sha256_text(canonical_json(scenario_hashes)),
        },
        "physical_state": physical_state,
        "generator_regression": regression,
        "upstream_hashes": upstream["hashes"],
    }

    plan = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": STATUS,
        "method": EXPECTED_METHOD_ID,
        "profile": EXPECTED_PROFILE_ID,
        "frozen_input_registry": {
            "scenarios": relative_path(FINAL_GATE_SCENARIOS),
            "replicates": relative_path(FINAL_GATE_REPLICATES),
            "n_scenarios": EXPECTED_SCENARIOS,
            "n_datasets": EXPECTED_DATASETS,
            "stability_resamples_per_dataset": EXPECTED_STABILITY_RESAMPLES,
            "materialization_policy": EXPECTED_FINAL_MATERIALIZATION_POLICY,
        },
        "execution_policy": {
            "permanent_input_bank": "FORBIDDEN_BY_EFFECTIVE_05f0_INTERPRETATION_OF_FROZEN_POLICY",
            "preferred_dataset_generation": "deterministic_in_memory_from_frozen_registry_seed",
            "temporary_npz": (
                "Permitted only as an execution cache after architecture freeze; must not become the frozen scientific "
                "input bank and must be removed after a validated result shard is durable."
            ),
            "persistent_artifact": "per_dataset_result_and_provenance_shard",
            "resume_unit": "validated result shard keyed by frozen dataset_key",
            "full_run_registry_order": "exact final_gate_calibration_replicates.csv order",
        },
        "runtime_pilot": {
            "dataset_count": PILOT_DATASET_COUNT,
            "selection": "first prespecified dataset IDs in frozen replicate-registry order",
            "registry_path": relative_path(OUTPUT_PILOT_REGISTRY),
            "reference_fits_per_dataset": 1,
            "stability_resample_fits_per_dataset": EXPECTED_STABILITY_RESAMPLES,
            "planned_optimization_runs": PILOT_DATASET_COUNT * (1 + EXPECTED_STABILITY_RESAMPLES),
            "scientific_console_output": "FORBIDDEN",
            "allowed_console_output": ["progress", "wall_time", "VRAM", "ETA", "shard_size"],
            "scientific_decision_during_pilot": "FORBIDDEN",
        },
        "full_run": {
            "planned_optimization_runs": EXPECTED_DATASETS * (1 + EXPECTED_STABILITY_RESAMPLES),
            "D01_D02_D03_rules": "read frozen 05a2 effective registries; unchanged",
            "threshold_retuning": "FORBIDDEN",
        },
        "numerical_versioning": {
            "numerical_implementation_hash_frozen_here": False,
            "required_next": (
                "05f1 must compute/freeze the exact numerical implementation hash before its first GPU pilot fit, "
                "embed it in every pilot shard, and reuse pilot shards only if unchanged."
            ),
            "pilot_invalidation_rule": upstream["runtime_policy"].get("invalidation_rule"),
            "pilot_reuse_rule": upstream["runtime_policy"].get("reuse_rule"),
            "required_shard_provenance": upstream["runtime_policy"].get("required_shard_provenance", []),
        },
        "scientific_guardrails": [
            "No final-gate dataset is generated or fit by 05f0.",
            "No D01/D02/D03 or subgroup scientific metric is evaluated by 05f0.",
            "No final-gate threshold, selector, denominator, dataset count, or resample count is changed.",
            "No TARGET, biological annotation, clinical label, or outcome is accessed.",
            "The failed prior persistent-materialization script created no final-gate dataset and is superseded by this versioned plan.",
        ],
        "upstream_hashes": upstream["hashes"],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(pilot, OUTPUT_PILOT_REGISTRY)
    write_json_atomic(preflight, OUTPUT_PREFLIGHT)
    write_json_atomic(plan, OUTPUT_PLAN)
    write_text_atomic(build_readme(pilot), OUTPUT_README)

    generated = [OUTPUT_PREFLIGHT, OUTPUT_PILOT_REGISTRY, OUTPUT_PLAN, OUTPUT_README]
    lock = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "script_path": relative_path(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "created_at_utc": utc_now_iso(),
        "status": STATUS,
        "plan_path": relative_path(OUTPUT_PLAN),
        "plan_sha256": sha256_file(OUTPUT_PLAN),
        "pilot_registry_path": relative_path(OUTPUT_PILOT_REGISTRY),
        "pilot_registry_sha256": sha256_file(OUTPUT_PILOT_REGISTRY),
        "pilot_dataset_count": PILOT_DATASET_COUNT,
        "permanent_final_gate_input_bank": False,
        "generated_files": {relative_path(path): file_record(path) for path in generated},
        "rewrite_policy": (
            "This zero-argument script never rewrites a frozen 05f0 streaming plan. Any future change requires a new "
            "versioned plan preserving this lock and may not be chosen using final-gate scientific results."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    print("-" * 88)
    print("Frozen final-gate streaming execution")
    print("-" * 88)
    print(f"Frozen materialization policy: {EXPECTED_FINAL_MATERIALIZATION_POLICY}")
    print("Permanent 3,100-NPZ input bank: NO")
    print("Final-gate datasets generated by 05f0: 0")
    print(f"Runtime pilot datasets: first {PILOT_DATASET_COUNT} frozen registry rows")
    print(f"Runtime pilot full-path fits: {PILOT_DATASET_COUNT * (1 + EXPECTED_STABILITY_RESAMPLES):,}")
    print(f"Full final-gate planned reference+resample fits: {EXPECTED_DATASETS * (1 + EXPECTED_STABILITY_RESAMPLES):,}")
    print("Scientific metrics visible during pilot: NO")
    print("Numerical implementation hash frozen by 05f1 before first pilot fit: REQUIRED")
    print()
    print("=" * 88)
    print("Paper 5 05f0 final-gate streaming execution plan: FROZEN")
    print("=" * 88)
    print(f"Plan: {relative_path(OUTPUT_PLAN)}")
    print(f"Pilot registry: {relative_path(OUTPUT_PILOT_REGISTRY)}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("Next: 05f1 GPU runtime pilot on the frozen first-3 dataset registry; no scientific gate output.")
    print("=" * 88)


if __name__ == "__main__":
    main()
