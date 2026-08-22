from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05e-compare-ablate-and-freeze-method-v1-no-cli"
CONTRACT_VERSION = "paper5-primary-proposed-representation-method-freeze-v1"
STATUS = "FROZEN_PRIMARY_PROPOSED_REPRESENTATION_METHOD_BEFORE_FINAL_GATE_AND_TARGET"

PROPOSED_METHOD_ID = "V4_SABLE_MO"
EXPLORATORY_METHOD_ID = "V5_SABLE_VAE"
EXPECTED_PROFILE_ID = "ARCH_COMPACT__REG_WEAK"
EXPECTED_SCENARIOS = 36
EXPECTED_CONFIRMATORY_DATASETS = 360
EXPECTED_CONFIRMATORY_REPLICATES = set(range(11, 21))
EXPECTED_NEURAL_VARIANTS = (
    "V0_BALANCED_AE",
    "V1_SHARED_PRIVATE_AE",
    "V2_CONTRASTIVE_SPAE",
    "V3_DROPOUT_SPAE",
    "V4_SABLE_MO",
    "V5_SABLE_VAE",
)
EXPECTED_PRIMARY_ABLATION_CONTRASTS = (
    "C1_SHARED_PRIVATE",
    "C2_CONTRASTIVE",
    "C3_DROPOUT_MISSING_VIEW",
)

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# -----------------------------------------------------------------------------
# Frozen upstream artifacts
# -----------------------------------------------------------------------------

DIR_05A = PAPER5_ROOT / "method_contract" / "05a"
CONTRACT_05A = DIR_05A / "ai_method_contract.json"
LOCK_05A = DIR_05A / "ai_method_contract_lock.json"
MODEL_REGISTRY_05A = DIR_05A / "model_family_registry.csv"
SELECTION_POLICY_05A = DIR_05A / "model_selection_policy.json"

DIR_05A2 = PAPER5_ROOT / "method_contract" / "05a2"
AMENDMENT_05A2 = DIR_05A2 / "inference_final_gate_implementation_amendment.json"
FAILURE_POLICY_05A2 = DIR_05A2 / "final_gate_failure_consequence_policy.json"
LOCK_05A2 = DIR_05A2 / "inference_final_gate_implementation_amendment_lock.json"

DIR_05C1 = PAPER5_ROOT / "method_contract" / "05c1"
LOCK_05C1 = DIR_05C1 / "linear_simulation_baseline_lock.json"
MANIFEST_05C1 = DIR_05C1 / "linear_baseline_results_manifest.json"
RESULTS_05C1 = PAPER5_ROOT / "results" / "simulation_benchmark" / "05c1_linear_baselines"
DATASET_05C1 = RESULTS_05C1 / "linear_dataset_metrics.csv.gz"

DIR_05C2B = PAPER5_ROOT / "method_contract" / "05c2b"
LOCK_05C2B = DIR_05C2B / "mofa_simulation_baseline_lock.json"
MANIFEST_05C2B = DIR_05C2B / "mofa_baseline_results_manifest.json"
RESULTS_05C2B = PAPER5_ROOT / "results" / "simulation_benchmark" / "05c2_mofa"
DATASET_05C2B = RESULTS_05C2B / "mofa_dataset_metrics.csv.gz"

DIR_05D1 = PAPER5_ROOT / "method_contract" / "05d1"
PROTOCOL_05D1 = DIR_05D1 / "neural_simulation_evaluation_protocol.json"
LOCK_05D1 = DIR_05D1 / "neural_simulation_protocol_lock.json"

DIR_05D3 = PAPER5_ROOT / "method_contract" / "05d3"
CONTRACT_05D3 = DIR_05D3 / "selected_neural_profile.json"
LOCK_05D3 = DIR_05D3 / "selected_neural_profile_lock.json"

DIR_05D4 = PAPER5_ROOT / "method_contract" / "05d4"
LOCK_05D4 = DIR_05D4 / "neural_confirmatory_ablation_results_lock.json"
MANIFEST_05D4 = DIR_05D4 / "neural_confirmatory_ablation_results_manifest.json"
RESULTS_05D4 = PAPER5_ROOT / "results" / "simulation_benchmark" / "05d_neural" / "confirmatory_ablation"
DATASET_05D4 = RESULTS_05D4 / "neural_confirmatory_dataset_variant_metrics.csv.gz"
VARIANT_SUMMARY_05D4 = RESULTS_05D4 / "neural_confirmatory_variant_summary.csv"

DIR_05D4A = PAPER5_ROOT / "method_contract" / "05d4a"
LOCK_05D4A = DIR_05D4A / "05d4a_results_lock.json"
EXECUTION_05D4A = DIR_05D4A / "05d4a_execution_record.json"
RESULTS_05D4A = PAPER5_ROOT / "results" / "simulation_benchmark" / "05d_neural" / "confirmatory_ablation_audit"
HEADLINE_05D4A = RESULTS_05D4A / "05d4a_headline_contrast_summary.csv"

DIR_05D5 = PAPER5_ROOT / "method_contract" / "05d5"
LOCK_05D5 = DIR_05D5 / "neural_confirmatory_stability_results_lock.json"
MANIFEST_05D5 = DIR_05D5 / "neural_confirmatory_stability_results_manifest.json"
RESULTS_05D5 = PAPER5_ROOT / "results" / "simulation_benchmark" / "05d_neural" / "confirmatory_stability"
OVERALL_05D5 = RESULTS_05D5 / "neural_confirmatory_stability_overall_summary.csv"
DATASET_05D5 = RESULTS_05D5 / "neural_confirmatory_stability_dataset_summary.csv.gz"

DIR_05CMP1 = PAPER5_ROOT / "method_contract" / "05cmp1"
CONTRACT_05CMP1 = DIR_05CMP1 / "secondary_comparator_implementation_contract.json"
LOCK_05CMP1 = DIR_05CMP1 / "secondary_comparator_implementation_contract_lock.json"

FINAL_GATE_DATA_ROOT = PAPER5_ROOT / "simulations" / "sable_mo_v2" / "final_gate_calibration"

# These must still be empty before the first 05e freeze. 05e itself writes only
# to paper5/method_contract/05e and deliberately does not create a benchmark-result
# directory.
FORBIDDEN_FUTURE_RESULT_ROOTS = (
    PAPER5_ROOT / "results" / "representation_benchmark",
    PAPER5_ROOT / "results" / "target_representation",
    PAPER5_ROOT / "results" / "biological_alignment",
    PAPER5_ROOT / "results" / "outcome_analysis",
    PAPER5_ROOT / "results" / "outcome_transport",
    PAPER5_ROOT / "results" / "secondary_comparators",
)

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05e"
OUTPUT_CONTEXT = OUTPUT_DIR / "confirmatory_method_context_summary.csv"
OUTPUT_COMPONENT = OUTPUT_DIR / "neural_confirmatory_component_evidence.csv"
OUTPUT_STABILITY = OUTPUT_DIR / "sable_mo_confirmatory_stability_evidence.csv"
OUTPUT_FREEZE = OUTPUT_DIR / "representation_method_freeze.json"
OUTPUT_README = OUTPUT_DIR / "05e_README.txt"
OUTPUT_LOCK = OUTPUT_DIR / "representation_method_freeze_lock.json"

COMMON_CONTEXT_METRICS = (
    "heldout_reconstruction_score",
    "random_adjusted_cross_view_mrr",
    "missing_view_reconstruction_score",
    "modality_contribution_entropy",
    "modality_balance_score",
    "shared_subspace_recovery",
    "private_subspace_recovery",
    "technical_subspace_recovery",
    "simulation_utility",
)

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
    text = clean_text(value).lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
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


def any_files(path: Path) -> bool:
    if path.is_file():
        return True
    return path.is_dir() and any(candidate.is_file() for candidate in path.rglob("*"))


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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


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
        raise RuntimeError(
            f"{label} generated-file verification failed:\n- "
            + "\n- ".join(failures[:100])
        )


def finite_values(values: Iterable[Any]) -> np.ndarray:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


def confirmatory_key(
    scenario_id: Any,
    replicate_index: Any,
    simulation_seed: Any,
) -> str:
    return (
        f"{clean_text(scenario_id)}|"
        f"{int(float(replicate_index))}|"
        f"{int(float(simulation_seed))}"
    )


def attach_confirmatory_key(table: pd.DataFrame) -> pd.DataFrame:
    required = {"scenario_id", "replicate_index", "simulation_seed"}
    missing = required - set(table.columns)
    if missing:
        raise RuntimeError("Table missing confirmatory key columns: " + ", ".join(sorted(missing)))
    result = table.copy()
    result["_confirmatory_key"] = [
        confirmatory_key(s, r, seed)
        for s, r, seed in zip(
            result["scenario_id"],
            result["replicate_index"],
            result["simulation_seed"],
        )
    ]
    return result


def validate_model_dataset_geometry(
    table: pd.DataFrame,
    model_id: str,
    *,
    expected_rows: int = EXPECTED_CONFIRMATORY_DATASETS,
) -> None:
    part = table[table["model_id"].eq(model_id)].copy()
    if part.shape[0] != expected_rows:
        raise RuntimeError(
            f"{model_id}: expected {expected_rows} confirmatory dataset rows; found {part.shape[0]}."
        )
    if part["_confirmatory_key"].duplicated().any():
        raise RuntimeError(f"{model_id}: duplicate confirmatory dataset key.")
    if part["scenario_id"].nunique() != EXPECTED_SCENARIOS:
        raise RuntimeError(f"{model_id}: expected {EXPECTED_SCENARIOS} scenarios.")
    scenario_counts = part.groupby("scenario_id")["_confirmatory_key"].nunique()
    if not scenario_counts.eq(10).all():
        raise RuntimeError(f"{model_id}: every scenario must contain exactly 10 confirmatory datasets.")
    replicates = set(pd.to_numeric(part["replicate_index"], errors="raise").astype(int).unique())
    if replicates != EXPECTED_CONFIRMATORY_REPLICATES:
        raise RuntimeError(f"{model_id}: confirmatory replicate set differs from 11..20.")


# -----------------------------------------------------------------------------
# Existing 05e verification
# -----------------------------------------------------------------------------


def verify_existing_lock() -> None:
    require_file(OUTPUT_LOCK)
    require_file(OUTPUT_FREEZE)
    lock = load_json(OUTPUT_LOCK)
    freeze = load_json(OUTPUT_FREEZE)

    failures: list[str] = []
    if clean_text(lock.get("status")) != STATUS:
        failures.append("unexpected lock status")
    if clean_text(freeze.get("status")) != STATUS:
        failures.append("unexpected freeze status")
    if clean_text(lock.get("script_version")) != SCRIPT_VERSION:
        failures.append("script-version mismatch")

    expected_freeze_hash = clean_text(lock.get("freeze", {}).get("sha256"))
    if not expected_freeze_hash or sha256_file(OUTPUT_FREEZE) != expected_freeze_hash:
        failures.append("freeze JSON hash mismatch")

    script_record = lock.get("script_source", {})
    expected_script_hash = clean_text(script_record.get("sha256"))
    if expected_script_hash and sha256_file(SCRIPT_PATH) != expected_script_hash:
        failures.append("05e script source hash mismatch")

    try:
        verify_generated_files(lock, "05e")
    except Exception as exc:
        failures.append(str(exc))

    if failures:
        raise RuntimeError(
            "Existing 05e representation-method freeze verification failed:\n- "
            + "\n- ".join(failures)
        )

    print("=" * 88)
    print("Existing Paper 5 05e representation-method freeze: VERIFIED")
    print("=" * 88)
    print(f"Freeze: {relative_path(OUTPUT_FREEZE)}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("No files were rewritten.")
    print("=" * 88)


# -----------------------------------------------------------------------------
# Upstream verification
# -----------------------------------------------------------------------------


def verify_05a() -> dict[str, Any]:
    for path in [CONTRACT_05A, LOCK_05A, MODEL_REGISTRY_05A, SELECTION_POLICY_05A]:
        require_file(path)

    contract = load_json(CONTRACT_05A)
    lock = load_json(LOCK_05A)
    expected_hash = clean_text(lock.get("contract_sha256"))
    if not expected_hash or sha256_file(CONTRACT_05A) != expected_hash:
        raise RuntimeError("05a contract hash mismatch.")
    verify_generated_files(lock, "05a")

    if clean_text(contract.get("contract_status")) != "FROZEN_BEFORE_MODEL_FITTING":
        raise RuntimeError("Unexpected 05a contract status.")

    method = contract.get("method", {})
    if clean_text(method.get("proposed_method_id")) != PROPOSED_METHOD_ID:
        raise RuntimeError("05a proposed method is no longer V4_SABLE_MO.")
    if bool(method.get("proposed_method_automatically_declared_winner", True)):
        raise RuntimeError("05a unexpectedly declares the proposed method an automatic winner.")

    state = contract.get("scientific_state", {})
    forbidden_true = [
        "outcome_loaded",
        "clinical_labels_loaded",
        "biological_program_labels_used_for_model_selection",
        "target_benchmark_results_viewed",
        "post_standardization_variance_used_for_selection",
        "global_preprocessing_allowed_in_heldout_evaluation",
    ]
    active = [key for key in forbidden_true if bool(state.get(key, False))]
    if active:
        raise RuntimeError("05a outcome-blind scientific state violated: " + ", ".join(active))

    registry = pd.read_csv(MODEL_REGISTRY_05A, dtype=str, low_memory=False).fillna("")
    required_columns = {
        "model_id",
        "family",
        "role",
        "eligible_for_nested_selection",
        "eligible_for_final_freeze",
        "complexity_rank",
    }
    missing = required_columns - set(registry.columns)
    if missing:
        raise RuntimeError("05a model registry missing: " + ", ".join(sorted(missing)))
    if registry["model_id"].duplicated().any():
        raise RuntimeError("05a model registry contains duplicate model_id values.")

    v4 = registry[registry["model_id"].eq(PROPOSED_METHOD_ID)]
    if v4.shape[0] != 1:
        raise RuntimeError("05a does not contain exactly one V4_SABLE_MO row.")
    if clean_text(v4.iloc[0]["role"]) != "proposed_primary_method":
        raise RuntimeError("V4 role changed from proposed_primary_method.")
    if not parse_bool(v4.iloc[0]["eligible_for_nested_selection"]):
        raise RuntimeError("V4 is no longer eligible for TARGET nested selection.")
    if not parse_bool(v4.iloc[0]["eligible_for_final_freeze"]):
        raise RuntimeError("V4 is no longer eligible for TARGET final freeze.")

    v5 = registry[registry["model_id"].eq(EXPLORATORY_METHOD_ID)]
    if v5.shape[0] != 1:
        raise RuntimeError("05a does not contain exactly one V5_SABLE_VAE row.")
    if clean_text(v5.iloc[0]["role"]) != "secondary_exploratory":
        raise RuntimeError("V5 is no longer marked secondary exploratory.")
    if parse_bool(v5.iloc[0]["eligible_for_nested_selection"]) or parse_bool(
        v5.iloc[0]["eligible_for_final_freeze"]
    ):
        raise RuntimeError("V5 unexpectedly became eligible for primary/final selection.")

    selection_policy = load_json(SELECTION_POLICY_05A)
    target_nested = selection_policy.get("target_nested_selection", {})
    final_target = selection_policy.get("final_full_data_method_freeze", {})
    if not target_nested or not final_target:
        raise RuntimeError("05a selection policy lacks TARGET nested/final freeze rules.")
    if bool(target_nested.get("outer_test_samples_used_for_selection", True)):
        raise RuntimeError("05a TARGET selection no longer protects outer test samples.")
    if not bool(final_target.get("biological_or_outcome_results_may_not_change_method", False)):
        raise RuntimeError("05a TARGET final method is no longer protected from biology/outcome changes.")

    return {
        "contract": contract,
        "lock": lock,
        "registry": registry,
        "selection_policy": selection_policy,
        "hashes": {
            relative_path(CONTRACT_05A): sha256_file(CONTRACT_05A),
            relative_path(LOCK_05A): sha256_file(LOCK_05A),
            relative_path(MODEL_REGISTRY_05A): sha256_file(MODEL_REGISTRY_05A),
            relative_path(SELECTION_POLICY_05A): sha256_file(SELECTION_POLICY_05A),
        },
    }


def verify_05a2() -> dict[str, Any]:
    for path in [AMENDMENT_05A2, FAILURE_POLICY_05A2, LOCK_05A2]:
        require_file(path)
    amendment = load_json(AMENDMENT_05A2)
    failure_policy = load_json(FAILURE_POLICY_05A2)
    lock = load_json(LOCK_05A2)

    expected_status = "FROZEN_BEFORE_FINAL_GATE_MATERIALIZATION"
    if clean_text(lock.get("status")) != expected_status or clean_text(amendment.get("status")) != expected_status:
        raise RuntimeError("05a2 is not in the expected prospective frozen state.")
    if sha256_file(AMENDMENT_05A2) != clean_text(lock.get("amendment_sha256")):
        raise RuntimeError("05a2 amendment hash mismatch.")
    verify_generated_files(lock, "05a2")

    unchanged = amendment.get("unchanged_scientific_design", {})
    if int(unchanged.get("final_gate_datasets", -1)) != 3100:
        raise RuntimeError("05a2 final-gate dataset count changed.")
    if int(unchanged.get("stability_resamples_per_dataset", -1)) != 100:
        raise RuntimeError("05a2 final-gate stability resample count changed.")
    if not bool(unchanged.get("outcome_blinding", False)):
        raise RuntimeError("05a2 outcome blinding changed.")

    if not clean_text(failure_policy.get("governing_rule")):
        raise RuntimeError("05a2 failure policy is unavailable.")

    return {
        "amendment": amendment,
        "failure_policy": failure_policy,
        "lock": lock,
        "hashes": {
            relative_path(AMENDMENT_05A2): sha256_file(AMENDMENT_05A2),
            relative_path(FAILURE_POLICY_05A2): sha256_file(FAILURE_POLICY_05A2),
            relative_path(LOCK_05A2): sha256_file(LOCK_05A2),
        },
    }


def verify_baseline_stage(
    lock_path: Path,
    manifest_path: Path,
    dataset_path: Path,
    *,
    expected_status: str,
    label: str,
) -> dict[str, Any]:
    for path in [lock_path, manifest_path, dataset_path]:
        require_file(path)
    lock = load_json(lock_path)
    manifest = load_json(manifest_path)
    if clean_text(lock.get("status")) != expected_status:
        raise RuntimeError(f"Unexpected {label} lock status.")
    verify_generated_files(lock, label)
    expected_manifest_hash = clean_text(lock.get("result_manifest", {}).get("sha256"))
    if not expected_manifest_hash or sha256_file(manifest_path) != expected_manifest_hash:
        raise RuntimeError(f"{label} result-manifest hash mismatch.")

    state = manifest.get("scientific_state", {})
    forbidden_true = [
        "outcomes_loaded",
        "clinical_labels_loaded",
        "biological_annotations_loaded",
        "TARGET_model_fitting_performed",
        "architecture_selected",
        "stability_gate_thresholds_retuned",
        "final_gate_calibration_data_used",
    ]
    active = [key for key in forbidden_true if bool(state.get(key, False))]
    if active:
        raise RuntimeError(f"{label} scientific-state violation: " + ", ".join(active))

    table = pd.read_csv(dataset_path, low_memory=False)
    return {
        "lock": lock,
        "manifest": manifest,
        "table": table,
        "hashes": {
            relative_path(lock_path): sha256_file(lock_path),
            relative_path(manifest_path): sha256_file(manifest_path),
            relative_path(dataset_path): sha256_file(dataset_path),
        },
    }


def verify_05d1() -> dict[str, Any]:
    for path in [PROTOCOL_05D1, LOCK_05D1]:
        require_file(path)
    protocol = load_json(PROTOCOL_05D1)
    lock = load_json(LOCK_05D1)
    if clean_text(lock.get("status")) != "FROZEN_NEURAL_SIMULATION_PROTOCOL_BEFORE_NEURAL_MODEL_FITTING":
        raise RuntimeError("Unexpected 05d1 lock status.")
    verify_generated_files(lock, "05d1")
    expected_hash = clean_text(lock.get("protocol", {}).get("sha256"))
    if not expected_hash or sha256_file(PROTOCOL_05D1) != expected_hash:
        raise RuntimeError("05d1 protocol hash mismatch.")

    stages = [clean_text(item.get("stage")) for item in protocol.get("stage_order", [])]
    if stages != ["05d2", "05d3", "05d4", "05d5", "05e"]:
        raise RuntimeError(f"05d1 stage order changed: {stages}")

    return {
        "protocol": protocol,
        "lock": lock,
        "hashes": {
            relative_path(PROTOCOL_05D1): sha256_file(PROTOCOL_05D1),
            relative_path(LOCK_05D1): sha256_file(LOCK_05D1),
        },
    }


def verify_05d3() -> dict[str, Any]:
    for path in [CONTRACT_05D3, LOCK_05D3]:
        require_file(path)
    contract = load_json(CONTRACT_05D3)
    lock = load_json(LOCK_05D3)
    expected_status = "FROZEN_OUTCOME_BLIND_NEURAL_PROFILE_BEFORE_CONFIRMATORY_ABLATION"
    if clean_text(contract.get("status")) != expected_status or clean_text(lock.get("status")) != expected_status:
        raise RuntimeError("05d3 selected profile is not in the expected frozen state.")
    verify_generated_files(lock, "05d3")
    expected_hash = clean_text(lock.get("contract", {}).get("sha256"))
    if not expected_hash or sha256_file(CONTRACT_05D3) != expected_hash:
        raise RuntimeError("05d3 selected-profile contract hash mismatch.")

    selected = contract.get("selected_profile", {})
    profile_id = clean_text(selected.get("profile_id"))
    if profile_id != EXPECTED_PROFILE_ID:
        raise RuntimeError(f"Unexpected 05d3 selected profile: {profile_id!r}")
    if clean_text(lock.get("selected_profile_id")) != profile_id:
        raise RuntimeError("05d3 lock/profile ID mismatch.")

    state = contract.get("scientific_state", {})
    forbidden_true = [
        "TARGET_model_fitting_performed",
        "confirmatory_replicates_11_to_20_accessed",
        "confirmatory_neural_ablation_performed",
        "final_gate_calibration_data_used",
        "representation_method_selected",
    ]
    active = [key for key in forbidden_true if bool(state.get(key, False))]
    if active:
        raise RuntimeError("05d3 prospective state violated: " + ", ".join(active))

    return {
        "contract": contract,
        "lock": lock,
        "profile_id": profile_id,
        "hashes": {
            relative_path(CONTRACT_05D3): sha256_file(CONTRACT_05D3),
            relative_path(LOCK_05D3): sha256_file(LOCK_05D3),
        },
    }


def verify_05d4() -> dict[str, Any]:
    for path in [LOCK_05D4, MANIFEST_05D4, DATASET_05D4, VARIANT_SUMMARY_05D4]:
        require_file(path)
    lock = load_json(LOCK_05D4)
    manifest = load_json(MANIFEST_05D4)
    if clean_text(lock.get("status")) != "FROZEN_NEURAL_CONFIRMATORY_ABLATION_RESULTS_BEFORE_05D5":
        raise RuntimeError("05d4 is not a clean all-PASS confirmatory result.")
    verify_generated_files(lock, "05d4")
    expected_manifest_hash = clean_text(lock.get("result_manifest", {}).get("sha256"))
    if not expected_manifest_hash or sha256_file(MANIFEST_05D4) != expected_manifest_hash:
        raise RuntimeError("05d4 result-manifest hash mismatch.")
    if clean_text(lock.get("profile_id")) != EXPECTED_PROFILE_ID:
        raise RuntimeError("05d4 profile differs from 05d3 frozen profile.")

    state = manifest.get("scientific_state", {})
    required_state = {
        "confirmatory_replicates_11_to_20_accessed": True,
        "confirmatory_neural_ablation_performed": True,
        "confirmatory_V4_stability_performed": False,
        "final_gate_calibration_data_used": False,
        "representation_method_selected": False,
        "TARGET_model_fitting_performed": False,
    }
    for key, expected in required_state.items():
        if bool(state.get(key, False)) != expected:
            raise RuntimeError(f"05d4 scientific-state mismatch for {key}.")

    scope = manifest.get("scope", {})
    if int(scope.get("confirmatory_datasets", -1)) != EXPECTED_CONFIRMATORY_DATASETS:
        raise RuntimeError("05d4 confirmatory dataset count is not 360.")
    if int(scope.get("passed_fold_seed_evaluations", -1)) != 21600:
        raise RuntimeError("05d4 passed fold-seed count is not 21,600.")
    if int(scope.get("failed_fold_seed_evaluations", -1)) != 0:
        raise RuntimeError("05d4 contains failed fold-seed evaluations.")

    table = pd.read_csv(DATASET_05D4, low_memory=False)
    table = attach_confirmatory_key(table)
    required_columns = {
        "variant_id",
        "profile_id",
        "all_fold_seed_evaluations_passed",
        "n_fold_seed_evaluations",
        "secondary_exploratory_variant",
        "stability_gate_applied_in_05d4",
    }
    required_columns |= {f"{metric}_mean" for metric in COMMON_CONTEXT_METRICS}
    missing = required_columns - set(table.columns)
    if missing:
        raise RuntimeError("05d4 dataset table missing: " + ", ".join(sorted(missing)))

    variants = tuple(sorted(table["variant_id"].astype(str).unique().tolist()))
    if variants != tuple(sorted(EXPECTED_NEURAL_VARIANTS)):
        raise RuntimeError(f"Unexpected 05d4 V0-V5 variant set: {variants}")
    if table.shape[0] != EXPECTED_CONFIRMATORY_DATASETS * len(EXPECTED_NEURAL_VARIANTS):
        raise RuntimeError("05d4 dataset-variant row count differs from 360 x 6.")
    if not table["profile_id"].astype(str).eq(EXPECTED_PROFILE_ID).all():
        raise RuntimeError("05d4 does not consistently use the frozen selected profile.")
    if not table["all_fold_seed_evaluations_passed"].map(parse_bool).all():
        raise RuntimeError("05d4 contains a dataset-variant row with failed fold-seed evaluation.")
    if not pd.to_numeric(table["n_fold_seed_evaluations"], errors="raise").astype(int).eq(10).all():
        raise RuntimeError("05d4 dataset rows do not each contain exactly ten fold-seed evaluations.")

    for variant in EXPECTED_NEURAL_VARIANTS:
        validate_model_dataset_geometry(table.rename(columns={"variant_id": "model_id"}), variant)

    v4 = table[table["variant_id"].eq(PROPOSED_METHOD_ID)]
    if v4["stability_gate_applied_in_05d4"].map(parse_bool).any():
        raise RuntimeError("05d4 unexpectedly applied the V4 stability gate.")
    v5 = table[table["variant_id"].eq(EXPLORATORY_METHOD_ID)]
    if not v5["secondary_exploratory_variant"].map(parse_bool).all():
        raise RuntimeError("V5 is not consistently marked secondary exploratory in 05d4.")

    keys = set(table.loc[table["variant_id"].eq(PROPOSED_METHOD_ID), "_confirmatory_key"].tolist())
    if len(keys) != EXPECTED_CONFIRMATORY_DATASETS:
        raise RuntimeError("05d4 confirmatory key set does not contain 360 unique datasets.")

    return {
        "lock": lock,
        "manifest": manifest,
        "table": table,
        "confirmatory_keys": keys,
        "hashes": {
            relative_path(LOCK_05D4): sha256_file(LOCK_05D4),
            relative_path(MANIFEST_05D4): sha256_file(MANIFEST_05D4),
            relative_path(DATASET_05D4): sha256_file(DATASET_05D4),
            relative_path(VARIANT_SUMMARY_05D4): sha256_file(VARIANT_SUMMARY_05D4),
        },
    }


def verify_05d4a() -> dict[str, Any]:
    for path in [LOCK_05D4A, EXECUTION_05D4A, HEADLINE_05D4A]:
        require_file(path)
    lock = load_json(LOCK_05D4A)
    execution = load_json(EXECUTION_05D4A)
    expected_status = "DERIVED_READ_ONLY_AUDIT_OF_FROZEN_05D4_RESULTS"
    if clean_text(lock.get("status")) != expected_status or clean_text(execution.get("status")) != expected_status:
        raise RuntimeError("05d4a is not in the expected frozen read-only state.")
    verify_generated_files(lock, "05d4a")

    state = execution.get("scientific_state", {})
    forbidden_true = [
        "model_fitting_performed",
        "GPU_computation_performed",
        "TARGET_accessed",
        "outcomes_accessed",
        "final_gate_data_accessed",
        "method_selection_performed",
        "V4_stability_gate_interpreted_from_05d4",
        "V5_promoted_from_secondary_exploratory",
    ]
    active = [key for key in forbidden_true if bool(state.get(key, False))]
    if active:
        raise RuntimeError("05d4a scientific-state violation: " + ", ".join(active))

    table = pd.read_csv(HEADLINE_05D4A, low_memory=False)
    required = {
        "contrast_id",
        "contrast_role",
        "higher_variant",
        "lower_variant",
        "added_component_or_comparison",
        "metric",
        "hypothesis_matched_headline_metric",
        "n_paired",
        "delta_mean",
        "bootstrap_mean_ci95_low",
        "bootstrap_mean_ci95_high",
        "fraction_positive",
        "interpretation_guardrail",
    }
    missing = required - set(table.columns)
    if missing:
        raise RuntimeError("05d4a headline table missing: " + ", ".join(sorted(missing)))

    for contrast in EXPECTED_PRIMARY_ABLATION_CONTRASTS:
        if table[table["contrast_id"].eq(contrast)].empty:
            raise RuntimeError(f"05d4a lacks primary contrast {contrast}.")

    return {
        "lock": lock,
        "execution": execution,
        "table": table,
        "hashes": {
            relative_path(LOCK_05D4A): sha256_file(LOCK_05D4A),
            relative_path(EXECUTION_05D4A): sha256_file(EXECUTION_05D4A),
            relative_path(HEADLINE_05D4A): sha256_file(HEADLINE_05D4A),
        },
    }


def verify_05d5() -> dict[str, Any]:
    for path in [LOCK_05D5, MANIFEST_05D5, OVERALL_05D5, DATASET_05D5]:
        require_file(path)
    lock = load_json(LOCK_05D5)
    manifest = load_json(MANIFEST_05D5)
    if clean_text(lock.get("status")) != "FROZEN_NEURAL_CONFIRMATORY_STABILITY_RESULTS_BEFORE_05E":
        raise RuntimeError(
            "05d5 is not a clean all-PASS confirmatory stability result; review failures before 05e."
        )
    verify_generated_files(lock, "05d5")
    expected_manifest_hash = clean_text(lock.get("result_manifest", {}).get("sha256"))
    if not expected_manifest_hash or sha256_file(MANIFEST_05D5) != expected_manifest_hash:
        raise RuntimeError("05d5 result-manifest hash mismatch.")
    if clean_text(lock.get("profile_id")) != EXPECTED_PROFILE_ID:
        raise RuntimeError("05d5 profile differs from frozen 05d3 profile.")
    if clean_text(lock.get("variant_id")) != PROPOSED_METHOD_ID:
        raise RuntimeError("05d5 variant differs from V4_SABLE_MO.")

    state = manifest.get("scientific_state", {})
    forbidden_true = [
        "TARGET_model_fitting_performed",
        "final_gate_calibration_data_used",
        "representation_method_selected",
    ]
    active = [key for key in forbidden_true if bool(state.get(key, False))]
    if active:
        raise RuntimeError("05d5 scientific-state violation: " + ", ".join(active))

    scope = manifest.get("scope", {})
    if int(scope.get("confirmatory_datasets", -1)) != EXPECTED_CONFIRMATORY_DATASETS:
        raise RuntimeError("05d5 confirmatory dataset count is not 360.")
    if int(scope.get("passed_datasets", -1)) != EXPECTED_CONFIRMATORY_DATASETS:
        raise RuntimeError("05d5 did not PASS all 360 confirmatory datasets.")
    if int(scope.get("failed_datasets", -1)) != 0:
        raise RuntimeError("05d5 contains failed confirmatory datasets.")

    overall = pd.read_csv(OVERALL_05D5, low_memory=False)
    if overall.shape[0] != 1:
        raise RuntimeError("05d5 overall summary must contain exactly one row.")
    row = overall.iloc[0]
    required_columns = {
        "variant_id",
        "profile_id",
        "n_scenarios",
        "n_confirmatory_datasets",
        "n_passed_datasets",
        "n_failed_datasets",
        "n_latent_collapse",
        "n_factor_level_primary_eligible_datasets",
        "overall_mean_stability_subspace_median_canonical_correlation_across_scenarios",
        "overall_mean_stability_subspace_q10_canonical_correlation_across_scenarios",
        "primary_dataset_mean_stability_gated_false_biological_discovery_fraction",
    }
    missing = required_columns - set(overall.columns)
    if missing:
        raise RuntimeError("05d5 overall summary missing: " + ", ".join(sorted(missing)))

    if clean_text(row["variant_id"]) != PROPOSED_METHOD_ID or clean_text(row["profile_id"]) != EXPECTED_PROFILE_ID:
        raise RuntimeError("05d5 overall V4/profile identity mismatch.")
    if int(float(row["n_scenarios"])) != EXPECTED_SCENARIOS:
        raise RuntimeError("05d5 overall scenario count is not 36.")
    if int(float(row["n_confirmatory_datasets"])) != EXPECTED_CONFIRMATORY_DATASETS:
        raise RuntimeError("05d5 overall dataset count is not 360.")
    if int(float(row["n_passed_datasets"])) != EXPECTED_CONFIRMATORY_DATASETS:
        raise RuntimeError("05d5 overall PASS count is not 360.")
    if int(float(row["n_failed_datasets"])) != 0:
        raise RuntimeError("05d5 overall FAIL count is not zero.")
    if int(float(row["n_latent_collapse"])) != 0:
        raise RuntimeError("05d5 reports latent collapse in confirmatory V4 stability.")

    dataset = pd.read_csv(DATASET_05D5, low_memory=False)
    dataset = attach_confirmatory_key(dataset)
    if dataset.shape[0] != EXPECTED_CONFIRMATORY_DATASETS:
        raise RuntimeError("05d5 dataset table does not contain 360 rows.")
    if not dataset["shard_status"].astype(str).eq("PASS").all():
        raise RuntimeError("05d5 dataset table contains a non-PASS shard.")
    if dataset["_confirmatory_key"].duplicated().any():
        raise RuntimeError("05d5 dataset table contains duplicate confirmatory keys.")

    return {
        "lock": lock,
        "manifest": manifest,
        "overall": overall,
        "dataset": dataset,
        "hashes": {
            relative_path(LOCK_05D5): sha256_file(LOCK_05D5),
            relative_path(MANIFEST_05D5): sha256_file(MANIFEST_05D5),
            relative_path(OVERALL_05D5): sha256_file(OVERALL_05D5),
            relative_path(DATASET_05D5): sha256_file(DATASET_05D5),
        },
    }


def verify_05cmp1() -> dict[str, Any]:
    for path in [CONTRACT_05CMP1, LOCK_05CMP1]:
        require_file(path)
    contract = load_json(CONTRACT_05CMP1)
    lock = load_json(LOCK_05CMP1)
    expected_status = "FROZEN_SECONDARY_COMPARATOR_IMPLEMENTATIONS_BEFORE_05E_AND_SCIENTIFIC_RUNS"
    if clean_text(contract.get("status")) != expected_status or clean_text(lock.get("status")) != expected_status:
        raise RuntimeError("05cmp1 is not in the expected pre-05e frozen state.")
    verify_generated_files(lock, "05cmp1")
    expected_hash = clean_text(lock.get("contract", {}).get("sha256"))
    if not expected_hash or sha256_file(CONTRACT_05CMP1) != expected_hash:
        raise RuntimeError("05cmp1 contract hash mismatch.")

    state = contract.get("scientific_state", {})
    forbidden_true = [
        "scientific_secondary_comparator_results_accessed",
        "real_TARGET_values_accessed",
        "TARGET_outcomes_accessed",
        "biological_annotations_accessed",
        "05e_results_accessed",
        "primary_method_selected_here",
        "final_gate_changed_here",
    ]
    active = [key for key in forbidden_true if bool(state.get(key, False))]
    if active:
        raise RuntimeError("05cmp1 scientific-state violation: " + ", ".join(active))

    return {
        "contract": contract,
        "lock": lock,
        "hashes": {
            relative_path(CONTRACT_05CMP1): sha256_file(CONTRACT_05CMP1),
            relative_path(LOCK_05CMP1): sha256_file(LOCK_05CMP1),
        },
    }


def verify_future_state() -> None:
    materialized = []
    if FINAL_GATE_DATA_ROOT.exists():
        materialized = [path for path in FINAL_GATE_DATA_ROOT.rglob("*") if path.is_file()]
    if materialized:
        raise RuntimeError(
            "Final-gate simulation data already exist before the 05e freeze. "
            "05e must freeze the proposed method before final-gate materialization.\n- "
            + "\n- ".join(relative_path(path) for path in materialized[:50])
        )

    future = [relative_path(root) for root in FORBIDDEN_FUTURE_RESULT_ROOTS if any_files(root)]
    if future:
        raise RuntimeError(
            "A future scientific/comparator result root already contains files before the first 05e freeze:\n- "
            + "\n- ".join(future)
        )


# -----------------------------------------------------------------------------
# Confirmatory context harmonization
# -----------------------------------------------------------------------------


def scenario_balanced_metrics(
    part: pd.DataFrame,
    column_map: dict[str, str | None],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in COMMON_CONTEXT_METRICS:
        source = column_map.get(metric)
        if source is None or source not in part.columns:
            output[f"scenario_balanced_mean_{metric}"] = None
            output[f"n_finite_scenario_means_{metric}"] = 0
            continue

        per_scenario: list[float] = []
        for _, scenario_part in part.groupby("scenario_id", sort=True):
            values = finite_values(scenario_part[source])
            if values.size:
                per_scenario.append(float(np.mean(values)))

        output[f"n_finite_scenario_means_{metric}"] = int(len(per_scenario))
        output[f"scenario_balanced_mean_{metric}"] = (
            float(np.mean(per_scenario)) if per_scenario else None
        )
    return output


def build_context_summary(
    registry: pd.DataFrame,
    linear: pd.DataFrame,
    mofa: pd.DataFrame,
    neural: pd.DataFrame,
    confirmatory_keys: set[str],
) -> pd.DataFrame:
    linear = attach_confirmatory_key(linear)
    mofa = attach_confirmatory_key(mofa)

    # 05c1 generated all models at ranks 5/10/15; the frozen primary linear rank is 10.
    if "is_primary_rank" in linear.columns:
        linear_primary = linear[linear["is_primary_rank"].map(parse_bool)].copy()
    else:
        linear_primary = linear[pd.to_numeric(linear["rank"], errors="raise").astype(int).eq(10)].copy()
    linear_primary = linear_primary[linear_primary["_confirmatory_key"].isin(confirmatory_keys)].copy()

    mofa_confirm = mofa[mofa["_confirmatory_key"].isin(confirmatory_keys)].copy()
    neural_confirm = neural[neural["_confirmatory_key"].isin(confirmatory_keys)].copy()

    expected_linear_models = {"PCA_RNA", "PCA_METHYLATION", "PCA_CNV", "BLOCK_PCA"}
    if set(linear_primary["model_id"].astype(str).unique()) != expected_linear_models:
        raise RuntimeError("05c1 primary-rank model set differs from the frozen four linear baselines.")
    for model_id in sorted(expected_linear_models):
        validate_model_dataset_geometry(linear_primary, model_id)

    expected_mofa_models = {"MOFA_K5", "MOFA_K10", "MOFA_K15", "MOFA_K20"}
    if set(mofa_confirm["model_id"].astype(str).unique()) != expected_mofa_models:
        raise RuntimeError("05c2b model set differs from MOFA_K5/K10/K15/K20.")
    for model_id in sorted(expected_mofa_models):
        validate_model_dataset_geometry(mofa_confirm, model_id)

    neural_confirm = neural_confirm.rename(columns={"variant_id": "model_id"})
    for model_id in EXPECTED_NEURAL_VARIANTS:
        validate_model_dataset_geometry(neural_confirm, model_id)

    # Exact same confirmatory dataset bank must underlie all integrated context rows.
    for source_name, table in [
        ("05c1", linear_primary),
        ("05c2b", mofa_confirm),
        ("05d4", neural_confirm),
    ]:
        observed_keys = set(table["_confirmatory_key"].unique())
        if observed_keys != confirmatory_keys:
            raise RuntimeError(f"{source_name} confirmatory dataset-key set differs from neural 05d4.")

    linear_map = {
        "heldout_reconstruction_score": "heldout_reconstruction_score",
        "random_adjusted_cross_view_mrr": "random_adjusted_cross_view_mrr",
        "missing_view_reconstruction_score": "missing_view_reconstruction_score",
        "modality_contribution_entropy": "modality_contribution_entropy",
        "modality_balance_score": "modality_contribution_entropy",
        "shared_subspace_recovery": "shared_subspace_recovery",
        "private_subspace_recovery": "private_subspace_recovery",
        "technical_subspace_recovery": "technical_subspace_recovery",
        "simulation_utility": "simulation_utility",
    }
    mofa_map = dict(linear_map)
    neural_map = {metric: f"{metric}_mean" for metric in COMMON_CONTEXT_METRICS}

    rows: list[dict[str, Any]] = []
    for _, model_row in registry.iterrows():
        model_id = clean_text(model_row["model_id"])
        base: dict[str, Any] = {
            "model_id": model_id,
            "family": clean_text(model_row["family"]),
            "role": clean_text(model_row["role"]),
            "eligible_for_nested_selection": parse_bool(model_row["eligible_for_nested_selection"]),
            "eligible_for_final_freeze": parse_bool(model_row["eligible_for_final_freeze"]),
            "complexity_rank": int(float(model_row["complexity_rank"])),
            "confirmatory_context_only": True,
            "cross_family_selection_score_defined_in_05e": False,
            "cross_family_winner_claim_permitted": False,
        }

        if model_id in expected_linear_models:
            part = linear_primary[linear_primary["model_id"].eq(model_id)].copy()
            base.update(
                {
                    "source_stage": "05c1",
                    "source_result_status": "COMPLETE_FROZEN_SIMULATION_BASELINE",
                    "n_confirmatory_datasets": int(part.shape[0]),
                    "n_scenarios": int(part["scenario_id"].nunique()),
                    "all_required_fits_passed": bool(
                        part.get("all_fold_fits_passed", pd.Series([True] * len(part))).map(parse_bool).all()
                    ),
                    "05e_role": "CONTEXT_ONLY_NO_CROSS_FAMILY_SELECTION_RULE",
                }
            )
            base.update(scenario_balanced_metrics(part, linear_map))

        elif model_id in expected_mofa_models:
            part = mofa_confirm[mofa_confirm["model_id"].eq(model_id)].copy()
            all_folds = part.get("all_fold_fits_passed", pd.Series([False] * len(part))).map(parse_bool).all()
            full_pass = part.get("full_fit_status", pd.Series([""] * len(part))).astype(str).eq("PASS").all()
            base.update(
                {
                    "source_stage": "05c2b",
                    "source_result_status": "COMPLETE_FROZEN_SIMULATION_BASELINE",
                    "n_confirmatory_datasets": int(part.shape[0]),
                    "n_scenarios": int(part["scenario_id"].nunique()),
                    "all_required_fits_passed": bool(all_folds and full_pass),
                    "05e_role": "CONTEXT_ONLY_NO_CROSS_FAMILY_SELECTION_RULE",
                }
            )
            base.update(scenario_balanced_metrics(part, mofa_map))

        elif model_id in EXPECTED_NEURAL_VARIANTS:
            part = neural_confirm[neural_confirm["model_id"].eq(model_id)].copy()
            base.update(
                {
                    "source_stage": "05d4",
                    "source_result_status": (
                        "COMPLETE_FROZEN_CONFIRMATORY_NEURAL_ABLATION_SECONDARY_EXPLORATORY"
                        if model_id == EXPLORATORY_METHOD_ID
                        else "COMPLETE_FROZEN_CONFIRMATORY_NEURAL_ABLATION"
                    ),
                    "n_confirmatory_datasets": int(part.shape[0]),
                    "n_scenarios": int(part["scenario_id"].nunique()),
                    "all_required_fits_passed": bool(
                        part["all_fold_seed_evaluations_passed"].map(parse_bool).all()
                    ),
                    "05e_role": (
                        "PROSPECTIVE_PROPOSED_METHOD_CONFIRMATORY_ANCHOR"
                        if model_id == PROPOSED_METHOD_ID
                        else (
                            "SECONDARY_EXPLORATORY_INELIGIBLE_CONTEXT_ONLY"
                            if model_id == EXPLORATORY_METHOD_ID
                            else "CONTEXT_ONLY_NO_CROSS_FAMILY_SELECTION_RULE"
                        )
                    ),
                }
            )
            base.update(scenario_balanced_metrics(part, neural_map))

        elif model_id == "RIDGE_GCCA":
            base.update(
                {
                    "source_stage": "05a_registry_only",
                    "source_result_status": "NOT_FIT_IN_COMPLETED_SIMULATION_BRANCH__SECONDARY_INELIGIBLE",
                    "n_confirmatory_datasets": 0,
                    "n_scenarios": 0,
                    "all_required_fits_passed": None,
                    "05e_role": "VISIBLE_SECONDARY_METHOD_NOT_A_05E_BLOCKER",
                }
            )
            for metric in COMMON_CONTEXT_METRICS:
                base[f"scenario_balanced_mean_{metric}"] = None
                base[f"n_finite_scenario_means_{metric}"] = 0

        else:
            raise RuntimeError(f"05e does not recognize frozen model registry row: {model_id}")

        rows.append(base)

    output = pd.DataFrame(rows)
    if output.shape[0] != registry.shape[0]:
        raise RuntimeError("05e context summary silently omitted a frozen model-registry row.")
    if list(output["model_id"]) != list(registry["model_id"]):
        raise RuntimeError("05e context summary changed the frozen model-registry order.")
    return output


def build_component_evidence(headline: pd.DataFrame) -> pd.DataFrame:
    output = headline.copy()
    output["method_selection_consequence"] = "NONE"
    output["cross_family_winner_consequence"] = "NONE"
    output["V4_minus_V3_as_stability_gate_ablation_permitted"] = False
    output["05e_interpretation"] = np.where(
        output["contrast_id"].isin(EXPECTED_PRIMARY_ABLATION_CONTRASTS),
        "PRIMARY_CONFIRMATORY_ABLATION_EVIDENCE_BY_HYPOTHESIS_MATCHED_ENDPOINT",
        "SECONDARY_EXPLORATORY_CONTEXT_ONLY",
    )
    return output


def build_stability_evidence(overall: pd.DataFrame) -> pd.DataFrame:
    row = overall.iloc[0].to_dict()
    keep = {
        "variant_id": row.get("variant_id"),
        "profile_id": row.get("profile_id"),
        "n_scenarios": row.get("n_scenarios"),
        "n_confirmatory_datasets": row.get("n_confirmatory_datasets"),
        "n_passed_datasets": row.get("n_passed_datasets"),
        "n_failed_datasets": row.get("n_failed_datasets"),
        "n_latent_collapse": row.get("n_latent_collapse"),
        "n_factor_level_primary_eligible_datasets": row.get("n_factor_level_primary_eligible_datasets"),
        "overall_mean_stability_subspace_median_canonical_correlation_across_scenarios": row.get(
            "overall_mean_stability_subspace_median_canonical_correlation_across_scenarios"
        ),
        "overall_mean_stability_subspace_q10_canonical_correlation_across_scenarios": row.get(
            "overall_mean_stability_subspace_q10_canonical_correlation_across_scenarios"
        ),
        "primary_dataset_mean_stability_gated_false_biological_discovery_fraction": row.get(
            "primary_dataset_mean_stability_gated_false_biological_discovery_fraction"
        ),
        "scenario_balanced_primary_false_biological_discovery_fraction": row.get(
            "scenario_balanced_primary_false_biological_discovery_fraction"
        ),
        "formal_final_gate_consequence": "NONE__05D5_IS_CONFIRMATORY_STABILITY_NOT_FINAL_GATE",
        "aggregate_threshold_comparison_permitted": False,
        "factor_threshold_note": (
            "Frozen factor-level stability rules were applied inside 05d5 where defined; "
            "aggregate scenario means are descriptive and are not reinterpreted as new pass/fail thresholds."
        ),
    }
    return pd.DataFrame([keep])


# -----------------------------------------------------------------------------
# Freeze construction
# -----------------------------------------------------------------------------


def merge_hashes(*sources: dict[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    for source in sources:
        for key, value in source.get("hashes", {}).items():
            if key in output and output[key] != value:
                raise RuntimeError(f"Conflicting upstream hash record for {key}")
            output[key] = value
    return output


def build_freeze(
    source_05a: dict[str, Any],
    source_05a2: dict[str, Any],
    source_05d1: dict[str, Any],
    source_05d3: dict[str, Any],
    source_05d4: dict[str, Any],
    source_05d4a: dict[str, Any],
    source_05d5: dict[str, Any],
    source_cmp1: dict[str, Any],
    upstream_hashes: dict[str, str],
) -> dict[str, Any]:
    selection_policy = source_05a["selection_policy"]
    final_gate = source_05a2["amendment"].get("effective_overrides", {})
    overall = source_05d5["overall"].iloc[0]

    return {
        "script_version": SCRIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": STATUS,
        "method": "SABLE-MO",
        "primary_proposed_representation": {
            "method_id": PROPOSED_METHOD_ID,
            "profile_id": EXPECTED_PROFILE_ID,
            "05a_role": "proposed_primary_method",
            "freeze_role": (
                "Primary proposed SABLE-MO implementation entering the preregistered final-gate "
                "calibration and subsequent outcome-blind TARGET benchmark."
            ),
            "not_a_cross_family_winner_declaration": True,
            "not_the_final_TARGET_selected_representation": True,
            "not_yet_final_gate_validated": True,
        },
        "scientific_state": {
            "outcomes_accessed": False,
            "clinical_labels_accessed_for_selection": False,
            "biological_program_annotations_accessed_for_selection": False,
            "TARGET_representation_results_accessed": False,
            "TARGET_pairing_permutation_results_accessed": False,
            "final_gate_simulation_data_accessed": False,
            "final_gate_results_accessed": False,
            "secondary_comparator_scientific_results_accessed": False,
            "cross_family_selection_score_created_in_05e": False,
            "cross_family_winner_declared": False,
            "TARGET_final_method_selected": False,
        },
        "freeze_basis": {
            "prospective_identity": (
                "05a designated V4_SABLE_MO as the proposed primary method before model fitting and explicitly "
                "did not guarantee it would win."
            ),
            "profile_selection": (
                "05d2/05d3 selected and froze ARCH_COMPACT__REG_WEAK from development simulations under the "
                "pre-existing hard-gate plus one-standard-error profile rule."
            ),
            "confirmatory_neural_ablation": (
                "05d4 completed all prespecified V0-V5 confirmatory held-out evaluations; 05d4a provides paired, "
                "hypothesis-matched component evidence without adding a new decision threshold."
            ),
            "confirmatory_stability": (
                "05d5 completed the frozen V4 post-fit resampling stability analysis on all 360 confirmatory datasets "
                "with zero failed dataset shards and zero latent-collapse datasets."
            ),
            "cross_family_context": (
                "05c1 BLOCK_PCA/PCA, 05c2b MOFA, and 05d4 neural results are summarized on the same confirmatory "
                "dataset bank for context only. No cross-family selection score/rule was prospectively frozen for 05e, "
                "so none is invented here."
            ),
            "secondary_external_comparators": (
                "05cmp1 froze iCluster2b and MOVE implementations before 05e; no scientific comparator result exists "
                "or influences this freeze."
            ),
        },
        "confirmatory_stability_snapshot": {
            "n_confirmatory_datasets": int(float(overall["n_confirmatory_datasets"])),
            "n_passed_datasets": int(float(overall["n_passed_datasets"])),
            "n_failed_datasets": int(float(overall["n_failed_datasets"])),
            "n_latent_collapse": int(float(overall["n_latent_collapse"])),
            "n_factor_level_primary_eligible_datasets": int(
                float(overall["n_factor_level_primary_eligible_datasets"])
            ),
            "overall_mean_stability_subspace_median_canonical_correlation_across_scenarios": float(
                overall["overall_mean_stability_subspace_median_canonical_correlation_across_scenarios"]
            ),
            "overall_mean_stability_subspace_q10_canonical_correlation_across_scenarios": float(
                overall["overall_mean_stability_subspace_q10_canonical_correlation_across_scenarios"]
            ),
            "primary_dataset_mean_stability_gated_false_biological_discovery_fraction": float(
                overall["primary_dataset_mean_stability_gated_false_biological_discovery_fraction"]
            ),
            "aggregate_metrics_are_descriptive_not_new_factor_thresholds": True,
        },
        "claims_not_made_by_05e": [
            "V4 is the best method across all model families.",
            "V4 has the highest raw confirmatory simulation utility.",
            "V4-minus-V3 pre-gate 05d4 metrics estimate the effect of the stability gate.",
            "V5 may be promoted from its prospectively frozen secondary-exploratory role.",
            "The final gate has passed.",
            "The final TARGET representation method has been selected.",
            "TARGET, biology, or outcomes validate the representation.",
        ],
        "TARGET_selection_policy_preserved": {
            "source": relative_path(SELECTION_POLICY_05A),
            "source_sha256": sha256_file(SELECTION_POLICY_05A),
            "target_nested_selection": selection_policy["target_nested_selection"],
            "final_full_data_method_freeze": selection_policy["final_full_data_method_freeze"],
            "05e_override": "NONE",
            "interpretation": (
                "05e freezes the proposed SABLE-MO implementation before final-gate/TARGET work; it does not "
                "replace the already-frozen outcome-blind TARGET nested-CV and final full-data method-selection rules."
            ),
        },
        "final_gate_required_next": {
            "05a2_status": source_05a2["amendment"]["status"],
            "final_gate_datasets": 3100,
            "stability_resamples_per_dataset": 100,
            "D01": final_gate.get("D01_NULL_SPECIFICITY", {}),
            "D02": final_gate.get("D02_STANDARD_SENSITIVITY", {}),
            "D03": final_gate.get("D03_STANDARD_FNR", {}),
            "failure_policy": relative_path(FAILURE_POLICY_05A2),
            "failure_rule": clean_text(source_05a2["failure_policy"].get("governing_rule")),
            "retuning_after_result": "FORBIDDEN",
            "outcomes_used_to_rescue_failed_gate": "FORBIDDEN",
        },
        "secondary_comparator_boundary": {
            "implementation_contract": relative_path(CONTRACT_05CMP1),
            "implementation_contract_sha256": sha256_file(CONTRACT_05CMP1),
            "scientific_runs_before_05e": "FORBIDDEN_AND_NONE_EXIST",
            "scientific_runs_after_05e": (
                "PERMITTED as secondary post-freeze benchmarking only; they cannot reopen 05e or alter the final gate."
            ),
            "05e_influence": "NONE",
        },
        "RIDGE_GCCA_disclosure": (
            "RIDGE_GCCA remains visible in the frozen 05a registry as a secondary regularized baseline, "
            "but no completed scientific simulation-fit branch exists for it in 05c1/05c2b/05d4. Because 05a made it "
            "ineligible for nested selection and final freeze, this absence is disclosed rather than silently hidden and "
            "does not block the V4 proposed-method freeze."
        ),
        "evidence_outputs": {
            "confirmatory_method_context_summary": relative_path(OUTPUT_CONTEXT),
            "neural_confirmatory_component_evidence": relative_path(OUTPUT_COMPONENT),
            "sable_mo_confirmatory_stability_evidence": relative_path(OUTPUT_STABILITY),
        },
        "upstream_hashes": upstream_hashes,
        "guardrails": [
            "No outcome, clinical label, biological program, TARGET benchmark result, or final-gate result is used by 05e.",
            "No new cross-family score, threshold, one-SE rule, p-value, or winner rule is invented in 05e.",
            "V5 remains secondary exploratory and ineligible for primary/final selection.",
            "V4 05d4 metrics are pre-gate; the stability component is represented only by 05d5 post-fit resampling evidence.",
            "The 05d5 scenario-average stability metrics are not compared directly to factor-level 0.70/0.40/0.80 retention thresholds.",
            "The frozen 05a TARGET nested-CV/final-full-data selection policy remains authoritative and unchanged.",
            "The preregistered 05a2 final gate must be materialized/evaluated next; failure changes the conclusion, not the criterion.",
            "Scientific secondary comparator runs may begin only after this freeze and remain secondary reporting analyses.",
        ],
        "next_primary_scientific_stage": (
            "Materialize/evaluate the preregistered final-gate calibration bank under 05a + 05a1 + 05a2, "
            "before scientific TARGET representation fitting."
        ),
    }


def build_readme(freeze: dict[str, Any]) -> str:
    s = freeze["confirmatory_stability_snapshot"]
    lines = [
        "Paper 5 - 05e primary proposed representation-method freeze",
        "=" * 78,
        "",
        f"Script version: {SCRIPT_VERSION}",
        f"Status: {STATUS}",
        f"Created UTC: {freeze['created_at_utc']}",
        "",
        "What is frozen",
        "--------------",
        f"Primary proposed method: {PROPOSED_METHOD_ID}",
        f"Frozen neural profile: {EXPECTED_PROFILE_ID}",
        "Role: fixed SABLE-MO implementation entering the preregistered final gate and later outcome-blind TARGET benchmark.",
        "",
        "What 05e does NOT claim",
        "-----------------------",
        "- no cross-family winner is declared;",
        "- no new cross-family score or threshold is introduced;",
        "- V5 remains secondary exploratory/ineligible;",
        "- 05d4 V4-minus-V3 is not a stability-gate ablation;",
        "- the final gate has not yet passed;",
        "- the final TARGET-selected representation has not yet been chosen.",
        "",
        "Confirmatory V4 stability snapshot",
        "---------------------------------",
        f"Datasets: {s['n_confirmatory_datasets']}",
        f"Passed: {s['n_passed_datasets']}",
        f"Failed: {s['n_failed_datasets']}",
        f"Latent collapse: {s['n_latent_collapse']}",
        f"Factor-level primary-eligible datasets: {s['n_factor_level_primary_eligible_datasets']}",
        "Aggregate stability metrics are descriptive; they are not reinterpreted as factor-level hard thresholds.",
        "",
        "TARGET policy preserved",
        "-----------------------",
        "The frozen 05a outcome-blind TARGET nested-CV and final-full-data method-selection rules remain unchanged.",
        "05e does not pre-select a TARGET winner and does not use TARGET data.",
        "",
        "Secondary comparators",
        "---------------------",
        "05cmp1 froze iCluster2b and MOVE before 05e. No scientific secondary-comparator result was used here.",
        "After 05e they may be run only as secondary post-freeze benchmarking and cannot reopen 05e/final-gate rules.",
        "",
        "Next primary scientific stage",
        "-----------------------------",
        "Materialize/evaluate the frozen 3,100-dataset final-gate calibration bank under 05a + 05a1 + 05a2.",
        "If a preregistered gate fails, change the scientific conclusion rather than the threshold.",
        "",
    ]
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    print("=" * 88)
    print("Paper 5 - compare frozen confirmatory evidence and freeze proposed method")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("GPU computation in 05e: NO")
    print("Model fitting in 05e: NO")
    print("TARGET/outcome/biology access in 05e: NO")
    print("Final-gate data access in 05e: NO")
    print("Secondary-comparator scientific result access in 05e: NO")
    print()

    # Existing frozen 05e is verified before any downstream/future-state checks,
    # so a later rerun remains valid after downstream stages have legitimately run.
    if OUTPUT_LOCK.exists():
        verify_existing_lock()
        return

    verify_future_state()

    source_05a = verify_05a()
    source_05a2 = verify_05a2()
    source_05c1 = verify_baseline_stage(
        LOCK_05C1,
        MANIFEST_05C1,
        DATASET_05C1,
        expected_status="FROZEN_LINEAR_BASELINE_RESULTS_BEFORE_MOFA_AND_NEURAL_FITTING",
        label="05c1",
    )
    source_05c2b = verify_baseline_stage(
        LOCK_05C2B,
        MANIFEST_05C2B,
        DATASET_05C2B,
        expected_status="FROZEN_MOFA_BASELINE_RESULTS_BEFORE_NEURAL_FITTING_AND_ARCHITECTURE_SELECTION",
        label="05c2b",
    )
    source_05d1 = verify_05d1()
    source_05d3 = verify_05d3()
    source_05d4 = verify_05d4()
    source_05d4a = verify_05d4a()
    source_05d5 = verify_05d5()
    source_cmp1 = verify_05cmp1()

    print("05a frozen AI/method + TARGET selection contract: PASS")
    print("05a2 frozen final-gate clarification/failure policy: PASS")
    print("05c1 frozen linear simulation baselines: PASS")
    print("05c2b frozen MOFA simulation baselines: PASS")
    print("05d1 frozen neural stage order/protocol: PASS")
    print("05d3 frozen neural profile: PASS")
    print("05d4 frozen V0-V5 confirmatory ablation: PASS")
    print("05d4a read-only paired component audit: PASS")
    print("05d5 frozen V4 confirmatory stability: PASS")
    print("05cmp1 frozen secondary-comparator implementations: PASS")
    print("Final-gate materialized files present: 0")
    print("Future TARGET/outcome/comparator scientific results present: NO")
    print()

    # Align every fitted context model to the exact same 360 confirmatory datasets
    # defined by 05d4 V4.
    context = build_context_summary(
        registry=source_05a["registry"],
        linear=source_05c1["table"],
        mofa=source_05c2b["table"],
        neural=source_05d4["table"],
        confirmatory_keys=source_05d4["confirmatory_keys"],
    )
    component = build_component_evidence(source_05d4a["table"])
    stability = build_stability_evidence(source_05d5["overall"])

    # Explicitly verify 05d5 used exactly the same 360 confirmatory simulation datasets.
    if set(source_05d5["dataset"]["_confirmatory_key"].tolist()) != source_05d4["confirmatory_keys"]:
        raise RuntimeError("05d5 confirmatory dataset-key set differs from 05d4 V4.")

    upstream_hashes = merge_hashes(
        source_05a,
        source_05a2,
        source_05c1,
        source_05c2b,
        source_05d1,
        source_05d3,
        source_05d4,
        source_05d4a,
        source_05d5,
        source_cmp1,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(context, OUTPUT_CONTEXT)
    write_csv_atomic(component, OUTPUT_COMPONENT)
    write_csv_atomic(stability, OUTPUT_STABILITY)

    freeze = build_freeze(
        source_05a=source_05a,
        source_05a2=source_05a2,
        source_05d1=source_05d1,
        source_05d3=source_05d3,
        source_05d4=source_05d4,
        source_05d4a=source_05d4a,
        source_05d5=source_05d5,
        source_cmp1=source_cmp1,
        upstream_hashes=upstream_hashes,
    )
    write_json_atomic(freeze, OUTPUT_FREEZE)
    write_text_atomic(build_readme(freeze), OUTPUT_README)

    generated = [
        OUTPUT_CONTEXT,
        OUTPUT_COMPONENT,
        OUTPUT_STABILITY,
        OUTPUT_FREEZE,
        OUTPUT_README,
    ]
    lock = {
        "script_version": SCRIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": STATUS,
        "primary_proposed_method_id": PROPOSED_METHOD_ID,
        "selected_profile_id": EXPECTED_PROFILE_ID,
        "freeze": file_record(OUTPUT_FREEZE),
        "script_source": file_record(SCRIPT_PATH),
        "generated_files": {
            relative_path(path): file_record(path)
            for path in generated
        },
        "upstream_hashes": upstream_hashes,
        "rewrite_policy": (
            "05e is immutable after creation. Any scientifically justified change requires a new versioned amendment. "
            "Do not rewrite 05e in response to final-gate, TARGET, biology, outcome, or secondary-comparator results."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    srow = stability.iloc[0]
    print("-" * 88)
    print("05e freeze summary")
    print("-" * 88)
    print(f"Frozen primary proposed method: {PROPOSED_METHOD_ID}")
    print(f"Frozen neural profile: {EXPECTED_PROFILE_ID}")
    print(f"Confirmatory V4 datasets passed: {int(float(srow['n_passed_datasets']))}/{EXPECTED_CONFIRMATORY_DATASETS}")
    print(f"Confirmatory V4 latent collapse: {int(float(srow['n_latent_collapse']))}")
    print("Cross-family winner declared: NO")
    print("New cross-family selection score created: NO")
    print("V5 promoted from secondary exploratory: NO")
    print("TARGET final method selected: NO")
    print("Final gate validation completed: NO")
    print("Secondary comparator scientific results used: NO")
    print("05a TARGET nested-CV/final-full-data selection policy overridden: NO")
    print()

    print("=" * 88)
    print("Paper 5 05e primary proposed representation-method freeze: FROZEN")
    print("=" * 88)
    print(f"Context summary: {relative_path(OUTPUT_CONTEXT)}")
    print(f"Component evidence: {relative_path(OUTPUT_COMPONENT)}")
    print(f"Stability evidence: {relative_path(OUTPUT_STABILITY)}")
    print(f"Freeze: {relative_path(OUTPUT_FREEZE)}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("Next primary scientific stage: final-gate calibration/materialization under frozen 05a2.")
    print("Scientific secondary-comparator runs are now permitted only as post-freeze secondary benchmarking.")
    print("=" * 88)


if __name__ == "__main__":
    main()
