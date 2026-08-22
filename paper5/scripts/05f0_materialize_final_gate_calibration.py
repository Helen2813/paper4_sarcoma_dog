from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05f0-materialize-final-gate-calibration-v1-no-cli"
CONTRACT_VERSION = "paper5-final-gate-calibration-materialization-v1"
STATUS = "FROZEN_FINAL_GATE_CALIBRATION_INPUTS_BEFORE_RUNTIME_PILOT"

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
FINAL_STAGE = "final_gate_calibration"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# Frozen upstream contracts.
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

# Materialized simulation data.
FINAL_GATE_DATA_ROOT = PAPER5_ROOT / "simulations" / "sable_mo_v2" / "final_gate_calibration"

# Outputs.
OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05f0"
OUTPUT_MANIFEST = OUTPUT_DIR / "final_gate_calibration_data_manifest.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "final_gate_calibration_materialization_summary.csv"
OUTPUT_GENERATOR_REGRESSION = OUTPUT_DIR / "final_gate_generator_regression.json"
OUTPUT_ENVIRONMENT = OUTPUT_DIR / "final_gate_materialization_environment.json"
OUTPUT_README = OUTPUT_DIR / "05f0_README.txt"
OUTPUT_LOCK = OUTPUT_DIR / "final_gate_calibration_materialization_lock.json"
OUTPUT_IN_PROGRESS = OUTPUT_DIR / ".final_gate_calibration_materialization_in_progress.json"


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


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = clean_text(value).lower()
    if not text:
        return bool(default)
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


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
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
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
        raise RuntimeError(f"{label} generated-file verification failed:\n- " + "\n- ".join(failures[:100]))


def import_generator() -> Any:
    require_file(GENERATOR_SCRIPT)
    spec = importlib.util.spec_from_file_location("paper5_05b_v2_generator", GENERATOR_SCRIPT)
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
# Frozen-state verification
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
    lock_05a2 = load_json(LOCK_05A2)
    runtime_policy = load_json(RUNTIME_POLICY_05A2)
    if clean_text(amendment.get("status")) != EXPECTED_05A2_STATUS or clean_text(lock_05a2.get("status")) != EXPECTED_05A2_STATUS:
        raise RuntimeError("05a2 is not frozen before final-gate materialization.")
    if sha256_file(AMENDMENT_05A2) != clean_text(lock_05a2.get("amendment_sha256")):
        raise RuntimeError("05a2 amendment hash mismatch.")
    verify_generated_files(lock_05a2, "05a2")
    if clean_text(runtime_policy.get("status")) != "FROZEN_BEFORE_FINAL_GATE_RUNTIME_PILOT":
        raise RuntimeError("05a2 runtime-pilot policy is not frozen.")

    lock_05b = load_json(LOCK_05B)
    if clean_text(lock_05b.get("status")) != EXPECTED_05B_STATUS:
        raise RuntimeError("Unexpected 05b v2 lock status.")
    verify_generated_files(lock_05b, "05b v2")
    scope_05b = lock_05b.get("scope", {})
    if bool(scope_05b.get("final_gate_calibration_materialized", True)):
        raise RuntimeError("05b v2 lock unexpectedly says final-gate data were already materialized.")
    if int(scope_05b.get("final_gate_registry_rows", -1)) != EXPECTED_DATASETS:
        raise RuntimeError("05b v2 lock final-gate registry row count differs from 3,100.")

    freeze_05e = load_json(FREEZE_05E)
    lock_05e = load_json(LOCK_05E)
    if clean_text(freeze_05e.get("status")) != EXPECTED_05E_STATUS or clean_text(lock_05e.get("status")) != EXPECTED_05E_STATUS:
        raise RuntimeError("05e proposed-method freeze is not in the required frozen state.")
    verify_generated_files(lock_05e, "05e")
    expected_freeze_hash = clean_text(lock_05e.get("freeze", {}).get("sha256"))
    if not expected_freeze_hash or sha256_file(FREEZE_05E) != expected_freeze_hash:
        raise RuntimeError("05e freeze JSON hash mismatch.")
    if clean_text(lock_05e.get("primary_proposed_method_id")) != EXPECTED_METHOD_ID:
        raise RuntimeError("05e primary proposed method is not V4_SABLE_MO.")
    if clean_text(lock_05e.get("selected_profile_id")) != EXPECTED_PROFILE_ID:
        raise RuntimeError("05e selected profile is not ARCH_COMPACT__REG_WEAK.")

    freeze_next = freeze_05e.get("final_gate_required_next", {})
    if int(freeze_next.get("final_gate_datasets", -1)) != EXPECTED_DATASETS:
        raise RuntimeError("05e final-gate dataset count changed.")
    if int(freeze_next.get("stability_resamples_per_dataset", -1)) != EXPECTED_STABILITY_RESAMPLES:
        raise RuntimeError("05e final-gate stability-resample count changed.")

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
# Final-gate registry parsing using the exact 05b-v2 generator parser
# -----------------------------------------------------------------------------


def load_final_gate_registries(generator: Any) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    scenarios_table = pd.read_csv(FINAL_GATE_SCENARIOS, dtype=str, low_memory=False).fillna("")
    replicates = pd.read_csv(FINAL_GATE_REPLICATES, dtype=str, low_memory=False).fillna("")

    if scenarios_table.shape[0] != EXPECTED_SCENARIOS:
        raise RuntimeError(f"Final-gate scenario registry has {scenarios_table.shape[0]} rows, expected {EXPECTED_SCENARIOS}.")
    if replicates.shape[0] != EXPECTED_DATASETS:
        raise RuntimeError(f"Final-gate replicate registry has {replicates.shape[0]} rows, expected {EXPECTED_DATASETS}.")
    if scenarios_table["scenario_id"].duplicated().any():
        raise RuntimeError("Duplicate scenario IDs exist in final-gate scenario registry.")
    if replicates.duplicated(["scenario_id", "replicate_index"]).any():
        raise RuntimeError("Duplicate scenario/replicate rows exist in final-gate registry.")
    if replicates["simulation_seed"].duplicated().any():
        raise RuntimeError("Duplicate simulation seeds exist in final-gate registry.")
    if not scenarios_table["simulation_stage"].astype(str).eq(FINAL_STAGE).all():
        raise RuntimeError("Final-gate scenario registry contains a non-final_gate_calibration stage row.")
    if not replicates["simulation_stage"].astype(str).eq(FINAL_STAGE).all():
        raise RuntimeError("Final-gate replicate registry contains a non-final_gate_calibration stage row.")
    if not replicates["materialization_policy"].astype(str).eq("persist_npz").all():
        raise RuntimeError("Final-gate replicate registry contains unsupported materialization policy.")

    for column in ["scenario_order", "replicate_index", "simulation_seed", "stability_resamples_per_dataset"]:
        replicates[column] = pd.to_numeric(replicates[column], errors="raise").astype("int64")
    if not replicates["stability_resamples_per_dataset"].eq(EXPECTED_STABILITY_RESAMPLES).all():
        raise RuntimeError("Not every final-gate dataset has exactly 100 frozen stability resamples.")

    scenarios: dict[str, Any] = {}
    for _, row in scenarios_table.iterrows():
        # Reuse the exact frozen 05b-v2 parsing and validation for every scientific field.
        # Only the stage token is temporarily changed for that parser because 05b-v2 was
        # intentionally restricted to architecture-screening materialization.
        parser_row = row.copy()
        parser_row["simulation_stage"] = "architecture_screening"
        parsed = generator.scenario_from_row(parser_row)
        final_scenario = dataclasses.replace(parsed, simulation_stage=FINAL_STAGE)
        if clean_text(final_scenario.generator_required_version) != EXPECTED_GENERATOR_VERSION:
            raise RuntimeError(f"{final_scenario.scenario_id}: unexpected generator-required version.")
        if int(final_scenario.stability_resamples_per_dataset) != EXPECTED_STABILITY_RESAMPLES:
            raise RuntimeError(f"{final_scenario.scenario_id}: scenario stability-resample count differs from 100.")
        scenarios[final_scenario.scenario_id] = final_scenario

    unknown = sorted(set(replicates["scenario_id"].astype(str)) - set(scenarios))
    if unknown:
        raise RuntimeError("Final-gate replicate registry contains unknown scenarios: " + ", ".join(unknown))

    observed_counts = replicates.groupby("scenario_id").size().to_dict()
    for scenario_id, scenario in scenarios.items():
        observed = int(observed_counts.get(scenario_id, 0))
        if observed != int(scenario.planned_replicates):
            raise RuntimeError(
                f"{scenario_id}: final-gate replicate count {observed} does not match planned {scenario.planned_replicates}."
            )

    return scenarios, scenarios_table, replicates


# -----------------------------------------------------------------------------
# Generator regression and data validation
# -----------------------------------------------------------------------------


def run_generator_regression(generator: Any) -> dict[str, Any]:
    frozen = load_json(FROZEN_GENERATOR_FIXTURE)
    current = generator.run_reference_fixture()
    keys = ["scenario_sha256", "npz_sha256", "npz_size_bytes", "array_sha256", "status"]
    mismatches = [key for key in keys if current.get(key) != frozen.get(key)]
    if mismatches:
        raise RuntimeError(
            "Current 05b-v2 numerical generator no longer reproduces the frozen reference fixture: "
            + ", ".join(mismatches)
        )
    return {
        "status": "PASS",
        "generator_script": relative_path(GENERATOR_SCRIPT),
        "generator_script_sha256": sha256_file(GENERATOR_SCRIPT),
        "generator_script_version": generator.SCRIPT_VERSION,
        "generator_version": generator.GENERATOR_VERSION,
        "frozen_fixture_path": relative_path(FROZEN_GENERATOR_FIXTURE),
        "frozen_fixture_sha256": sha256_file(FROZEN_GENERATOR_FIXTURE),
        "reference_npz_sha256": current["npz_sha256"],
        "reference_scenario_sha256": current["scenario_sha256"],
        "reference_array_hashes_match": True,
    }


def final_gate_dataset_path(generator: Any, scenario: Any, replicate_index: int, simulation_seed: int) -> Path:
    return (
        FINAL_GATE_DATA_ROOT
        / scenario.scenario_family
        / scenario.scenario_id
        / generator.dataset_filename(scenario.scenario_id, replicate_index, simulation_seed)
    )


def validate_final_gate_arrays(
    generator: Any,
    arrays: dict[str, np.ndarray],
    scenario: Any,
    registry_label: str,
    replicate_index: int,
    simulation_seed: int,
) -> dict[str, Any]:
    if "simulation_stage" not in arrays:
        raise RuntimeError("Final-gate dataset lacks simulation_stage scalar.")
    observed_stage = str(arrays["simulation_stage"].item())
    if observed_stage != FINAL_STAGE:
        raise RuntimeError(f"Final-gate simulation-stage mismatch: {observed_stage!r}.")

    # Exact 05b-v2 array validator is reused. Its only architecture-specific check is
    # the scalar stage token, so use an in-memory validation shim for that token while
    # leaving the real final-gate arrays and final-stage scenario hash unchanged.
    shim = dict(arrays)
    shim["simulation_stage"] = np.asarray("architecture_screening", dtype=arrays["simulation_stage"].dtype)
    return generator.validate_dataset_arrays(
        shim,
        scenario,
        registry_label,
        replicate_index,
        simulation_seed,
    )


def materialize_one(
    generator: Any,
    scenario: Any,
    registry_label: str,
    registry_row_index: int,
    replicate_index: int,
    simulation_seed: int,
) -> dict[str, Any]:
    path = final_gate_dataset_path(generator, scenario, replicate_index, simulation_seed)

    if path.exists():
        arrays = generator.read_npz_arrays(path)
        validation = validate_final_gate_arrays(
            generator, arrays, scenario, registry_label, replicate_index, simulation_seed
        )
        status = "verified_existing"
    else:
        arrays = generator.generate_dataset(scenario, registry_label, replicate_index, simulation_seed)
        validation = validate_final_gate_arrays(
            generator, arrays, scenario, registry_label, replicate_index, simulation_seed
        )
        generator.write_deterministic_npz(path, arrays)
        serialized = generator.read_npz_arrays(path)
        validate_final_gate_arrays(
            generator, serialized, scenario, registry_label, replicate_index, simulation_seed
        )
        status = "generated"

    return {
        "registry_row_index": int(registry_row_index),
        "dataset_key": f"{scenario.scenario_id}|{replicate_index}|{simulation_seed}",
        "registry_label": registry_label,
        "simulation_stage": FINAL_STAGE,
        "scenario_id": scenario.scenario_id,
        "scenario_origin": scenario.scenario_origin,
        "scenario_family": scenario.scenario_family,
        "calibration_role": scenario.calibration_role,
        "replicate_index": int(replicate_index),
        "simulation_seed": int(simulation_seed),
        "n_samples": int(scenario.n_samples),
        "n_features_rna": int(scenario.n_features_rna),
        "n_features_methylation": int(scenario.n_features_methylation),
        "n_features_cnv": int(scenario.n_features_cnv),
        "n_biological_shared_factors": int(scenario.n_shared_factors),
        "n_private_factors_per_view": int(scenario.n_private_factors_per_view),
        "n_technical_shared_factors": int(scenario.true_technical_shared_factors),
        "shared_factor_correlation": float(scenario.shared_factor_correlation),
        "shared_support_by_factor": scenario.shared_support_by_factor,
        "loading_regime": scenario.loading_regime,
        "missingness_mode": scenario.missingness_mode,
        "batch_mode": scenario.batch_mode,
        "pair_corruption_fraction": float(scenario.pair_corruption_fraction),
        "factor_level_identifiable": bool(scenario.factor_level_identifiable),
        "subspace_level_identifiable": bool(scenario.subspace_level_identifiable),
        "primary_truth_target": scenario.primary_truth_target,
        "stability_resamples_per_dataset": int(scenario.stability_resamples_per_dataset),
        "scenario_sha256": generator.sha256_text(generator.canonical_json(scenario.canonical_dict())),
        "path": relative_path(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "generation_status": status,
        **validation,
    }


def estimate_missing_storage(generator: Any, scenarios: dict[str, Any], replicates: pd.DataFrame) -> tuple[int, int]:
    estimated_bytes = 0
    missing_datasets = 0
    for _, row in replicates.iterrows():
        scenario = scenarios[clean_text(row["scenario_id"])]
        replicate_index = int(row["replicate_index"])
        simulation_seed = int(row["simulation_seed"])
        path = final_gate_dataset_path(generator, scenario, replicate_index, simulation_seed)
        if not path.exists():
            missing_datasets += 1
            estimated_bytes += int(generator.estimate_dataset_storage_bytes(scenario))
    return estimated_bytes, missing_datasets


# -----------------------------------------------------------------------------
# Existing frozen 05f0 verification
# -----------------------------------------------------------------------------


def verify_existing_lock() -> None:
    require_file(OUTPUT_LOCK)
    require_file(OUTPUT_MANIFEST)
    lock = load_json(OUTPUT_LOCK)
    if clean_text(lock.get("status")) != STATUS:
        raise RuntimeError("Unexpected existing 05f0 lock status.")
    verify_generated_files(lock, "05f0")

    manifest = pd.read_csv(OUTPUT_MANIFEST, dtype=str, low_memory=False).fillna("")
    if manifest.shape[0] != EXPECTED_DATASETS:
        raise RuntimeError("Existing 05f0 manifest row count differs from 3,100.")
    if manifest["dataset_key"].duplicated().any() or manifest["path"].duplicated().any():
        raise RuntimeError("Existing 05f0 manifest contains duplicate dataset keys or paths.")

    print("Verifying frozen final-gate calibration datasets...")
    failures: list[str] = []
    for index, row in manifest.iterrows():
        path = resolve_project_path(row["path"])
        if not path.is_file():
            failures.append(f"missing: {row['path']}")
            continue
        if int(path.stat().st_size) != int(float(row["size_bytes"])):
            failures.append(f"size mismatch: {row['path']}")
            continue
        if sha256_file(path) != clean_text(row["sha256"]):
            failures.append(f"SHA256 mismatch: {row['path']}")
        current = index + 1
        if current == 1 or current % 100 == 0 or current == EXPECTED_DATASETS:
            print(f" {current:04d}/{EXPECTED_DATASETS:04d} existence/size/SHA256: PASS")
    if failures:
        raise RuntimeError("Frozen 05f0 dataset verification failed:\n- " + "\n- ".join(failures[:100]))

    print()
    print("=" * 88)
    print("Existing Paper 5 final-gate calibration materialization: VERIFIED")
    print("=" * 88)
    print(f"Datasets: {manifest.shape[0]:,}")
    print(f"Manifest: {relative_path(OUTPUT_MANIFEST)}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("No files were rewritten.")
    print("=" * 88)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    print("=" * 88)
    print("Paper 5 - materialize preregistered final-gate calibration simulations")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data root: {FINAL_GATE_DATA_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("GPU computation in 05f0: NO")
    print("Model fitting in 05f0: NO")
    print("Final-gate scientific metrics in 05f0: NO")
    print("TARGET/outcome/biology access in 05f0: NO")
    print("Threshold retuning in 05f0: FORBIDDEN")
    print()

    if OUTPUT_LOCK.exists():
        verify_existing_lock()
        return

    upstream = verify_upstream()
    generator = import_generator()
    scenarios, scenarios_table, replicates = load_final_gate_registries(generator)

    print("05a1 frozen final-gate registry: PASS")
    print("05a2 frozen final-gate implementation clarification: PASS")
    print("05a2 runtime-pilot policy: PASS")
    print("05b v2 frozen numerical generator anchor: PASS")
    print("05e frozen proposed method V4_SABLE_MO: PASS")
    print(f"Final-gate scenarios: {len(scenarios):,}")
    print(f"Final-gate datasets: {replicates.shape[0]:,}")
    print(f"Stability resamples per dataset: {EXPECTED_STABILITY_RESAMPLES}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_GATE_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    regression = run_generator_regression(generator)
    write_json_atomic(regression, OUTPUT_GENERATOR_REGRESSION)
    print("Frozen 05b-v2 generator reference replay: PASS")
    print()

    estimated_bytes, missing_datasets = estimate_missing_storage(generator, scenarios, replicates)
    free_bytes = shutil.disk_usage(FINAL_GATE_DATA_ROOT.parent).free
    required_bytes = int(estimated_bytes * 1.20 + 1024 * 1024 * 1024)

    print("Disk preflight:")
    print(f" Datasets not yet present: {missing_datasets:,}")
    print(f" Estimated additional bytes: {estimated_bytes:,}")
    print(f" Free bytes: {free_bytes:,}")
    print(f" Required with safety headroom: {required_bytes:,}")
    if free_bytes < required_bytes:
        raise RuntimeError(
            "Insufficient free disk space for final-gate calibration materialization. "
            f"Need about {required_bytes:,} bytes including safety headroom."
        )
    print()

    in_progress = {
        "script_version": SCRIPT_VERSION,
        "started_at_utc": utc_now_iso(),
        "expected_scenarios": EXPECTED_SCENARIOS,
        "expected_datasets": EXPECTED_DATASETS,
        "stability_resamples_per_dataset": EXPECTED_STABILITY_RESAMPLES,
        "output_root": relative_path(FINAL_GATE_DATA_ROOT),
        "model_fitting_performed": False,
        "scientific_gate_metrics_evaluated": False,
        "TARGET_access": False,
        "outcome_access": False,
    }
    write_json_atomic(in_progress, OUTPUT_IN_PROGRESS)

    manifest_rows: list[dict[str, Any]] = []
    total = int(replicates.shape[0])
    for ordinal, (_, row) in enumerate(replicates.iterrows(), start=1):
        scenario_id = clean_text(row["scenario_id"])
        scenario = scenarios[scenario_id]
        registry_label = clean_text(row["registry_label"])
        replicate_index = int(row["replicate_index"])
        simulation_seed = int(row["simulation_seed"])

        record = materialize_one(
            generator,
            scenario,
            registry_label,
            ordinal - 1,
            replicate_index,
            simulation_seed,
        )
        manifest_rows.append(record)

        if ordinal == 1 or ordinal % 25 == 0 or ordinal == total:
            print(
                f" {ordinal:04d}/{total:04d} {scenario_id} R{replicate_index:03d}: "
                f"{record['generation_status']}"
            )

    manifest = pd.DataFrame(manifest_rows)
    if manifest.shape[0] != EXPECTED_DATASETS:
        raise RuntimeError("Final-gate materialization manifest row-count mismatch.")
    if manifest["dataset_key"].duplicated().any() or manifest["path"].duplicated().any():
        raise RuntimeError("Duplicate dataset key/path created in final-gate materialization.")
    if not manifest["generation_status"].isin(["generated", "verified_existing"]).all():
        raise RuntimeError("Unexpected final-gate generation status.")
    if not manifest["simulation_stage"].eq(FINAL_STAGE).all():
        raise RuntimeError("Final-gate manifest contains wrong simulation stage.")

    # Preserve the exact frozen replicate-registry order. The runtime-pilot policy
    # selects the first prespecified dataset IDs in that order.
    manifest = manifest.sort_values("registry_row_index", kind="mergesort").reset_index(drop=True)
    write_csv_atomic(manifest, OUTPUT_MANIFEST)

    summary = (
        manifest.groupby(
            ["scenario_id", "scenario_origin", "scenario_family", "calibration_role"],
            sort=False,
            dropna=False,
        )
        .agg(
            n_datasets=("path", "size"),
            n_generated=("generation_status", lambda x: int(np.sum(x == "generated"))),
            n_verified_existing=("generation_status", lambda x: int(np.sum(x == "verified_existing"))),
            total_size_bytes=("size_bytes", "sum"),
            n_samples=("n_samples", "first"),
            n_biological_shared_factors=("n_biological_shared_factors", "first"),
            n_technical_shared_factors=("n_technical_shared_factors", "first"),
            mean_rna_missing_fraction=("rna_missing_fraction", "mean"),
            mean_methylation_missing_fraction=("methylation_missing_fraction", "mean"),
            mean_cnv_missing_fraction=("cnv_missing_fraction", "mean"),
        )
        .reset_index()
    )
    if summary.shape[0] != EXPECTED_SCENARIOS:
        raise RuntimeError("Final-gate scenario summary row-count mismatch.")
    write_csv_atomic(summary, OUTPUT_SUMMARY)

    environment = {
        "created_at_utc": utc_now_iso(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "GPU_used": False,
    }
    write_json_atomic(environment, OUTPUT_ENVIRONMENT)

    total_bytes = int(pd.to_numeric(manifest["size_bytes"], errors="raise").sum())
    readme = "\n".join(
        [
            "Paper 5 - 05f0 final-gate calibration materialization",
            "=" * 78,
            "",
            f"Script version: {SCRIPT_VERSION}",
            f"Status: {STATUS}",
            "",
            "Scientific state",
            "----------------",
            "05e primary proposed method frozen: V4_SABLE_MO / ARCH_COMPACT__REG_WEAK",
            "final-gate scenarios materialized: 24",
            "final-gate datasets materialized: 3,100",
            "stability resamples per dataset: 100 (registry only; no model fits here)",
            "scientific final-gate metrics evaluated: NO",
            "TARGET/outcomes/biology accessed: NO",
            "threshold retuning: FORBIDDEN",
            "",
            "Generator identity",
            "------------------",
            "The exact 05b-v2 deterministic generator implementation is reused.",
            "Its frozen reference fixture is replayed and must match before materialization.",
            "The final-gate stage token is the only intentional stage-level difference from 05b-v2 architecture inputs.",
            "",
            "Runtime pilot boundary",
            "----------------------",
            "The next stage is a timing/VRAM runtime pilot on the first prespecified final-gate dataset IDs.",
            "The pilot must not print or aggregate scientific gate metrics.",
            "Pilot shards are reusable only if the numerical implementation hash remains unchanged.",
            "",
            f"Total final-gate input bytes: {total_bytes}",
            "",
        ]
    )
    write_text_atomic(readme, OUTPUT_README)

    generated_contract_files = [
        OUTPUT_MANIFEST,
        OUTPUT_SUMMARY,
        OUTPUT_GENERATOR_REGRESSION,
        OUTPUT_ENVIRONMENT,
        OUTPUT_README,
    ]
    lock = {
        "script_version": SCRIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": STATUS,
        "scientific_state": {
            "primary_proposed_method": EXPECTED_METHOD_ID,
            "selected_profile": EXPECTED_PROFILE_ID,
            "final_gate_inputs_materialized": True,
            "final_gate_model_fitting_performed": False,
            "final_gate_scientific_metrics_evaluated": False,
            "TARGET_accessed": False,
            "outcomes_accessed": False,
            "biological_annotations_accessed": False,
            "threshold_retuning_performed": False,
        },
        "scope": {
            "n_scenarios": EXPECTED_SCENARIOS,
            "n_datasets": EXPECTED_DATASETS,
            "stability_resamples_per_dataset": EXPECTED_STABILITY_RESAMPLES,
            "total_size_bytes": total_bytes,
            "data_root": relative_path(FINAL_GATE_DATA_ROOT),
            "manifest": file_record(OUTPUT_MANIFEST),
        },
        "generator": regression,
        "upstream_hashes": upstream["hashes"],
        "script_source": file_record(SCRIPT_PATH),
        "generated_files": {
            relative_path(path): file_record(path)
            for path in generated_contract_files
        },
        "runtime_pilot_policy": file_record(RUNTIME_POLICY_05A2),
        "rewrite_policy": (
            "05f0 is immutable after complete materialization. Existing dataset shards are verified and reused only "
            "when their size/SHA256 and deterministic truth validation pass. Scientific final-gate results may not "
            "rewrite these inputs."
        ),
        "next_stage": "final-gate GPU runtime pilot under frozen 05a2 runtime policy",
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    if OUTPUT_IN_PROGRESS.exists():
        OUTPUT_IN_PROGRESS.unlink()

    print()
    print("-" * 88)
    print("Final-gate calibration materialization summary")
    print("-" * 88)
    print(f"Scenarios: {EXPECTED_SCENARIOS}")
    print(f"Datasets: {EXPECTED_DATASETS:,}")
    print(f"Total size bytes: {total_bytes:,}")
    print(f"Generated this run: {int(np.sum(manifest['generation_status'] == 'generated')):,}")
    print(f"Verified existing: {int(np.sum(manifest['generation_status'] == 'verified_existing')):,}")
    print("Simulation/model fitting performed: NO")
    print("Final-gate scientific metrics evaluated: NO")
    print("TARGET/outcomes accessed: NO")
    print("Thresholds retuned: NO")
    print()
    print("=" * 88)
    print("Paper 5 final-gate calibration input materialization: PASS")
    print("=" * 88)
    print(f"Manifest: {relative_path(OUTPUT_MANIFEST)}")
    print(f"Summary: {relative_path(OUTPUT_SUMMARY)}")
    print(f"Generator regression: {relative_path(OUTPUT_GENERATOR_REGRESSION)}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("Next: final-gate GPU runtime pilot (timing/VRAM only; no scientific gate output).")
    print("=" * 88)


if __name__ == "__main__":
    main()
