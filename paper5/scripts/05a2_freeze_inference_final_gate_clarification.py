from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_VERSION = "05a2-freeze-inference-final-gate-clarification-v1-no-cli"
AMENDMENT_VERSION = "sable-mo-inference-final-gate-clarification-v1"
AMENDMENT_STATUS = "FROZEN_BEFORE_FINAL_GATE_MATERIALIZATION"

PAIRING_PERMUTATIONS_BASE = 100
PAIRING_PERMUTATIONS_EFFECTIVE = 1_000

D01_LIMIT = 0.10
D01_WILSON_Z = 1.959963984540054
D01_CONFIDENCE_LEVEL = 0.95

D02_TPR_MIN = 0.80
D03_FNR_MAX = 0.20

EXPECTED_FINAL_GATE_SCENARIOS = 24
EXPECTED_FINAL_GATE_DATASETS = 3_100
EXPECTED_FINAL_GATE_STABILITY_RESAMPLES = 100
EXPECTED_D01_DATASETS = 1_000
EXPECTED_D01_INDEPENDENT_DATASETS = 800
EXPECTED_D01_CORRELATED_PRIVATE_DATASETS = 200
EXPECTED_STANDARD_D02_D03_DATASETS = 100
EXPECTED_STANDARD_TRUE_FACTOR_OPPORTUNITIES = 400


SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# -----------------------------------------------------------------------------
# Frozen upstream artifacts
# -----------------------------------------------------------------------------

DIR_05A = PAPER5_ROOT / "method_contract" / "05a"
CONTRACT_05A = DIR_05A / "ai_method_contract.json"
LOCK_05A = DIR_05A / "ai_method_contract_lock.json"
RANDOM_CONTROLS_05A = DIR_05A / "random_control_registry.csv"
MULTIPLICITY_05A = DIR_05A / "multiplicity_family_registry.csv"

DIR_05A1 = PAPER5_ROOT / "method_contract" / "05a1"
CONTRACT_05A1 = DIR_05A1 / "simulation_and_technical_confound_addendum.json"
LOCK_05A1 = DIR_05A1 / "simulation_and_technical_confound_addendum_lock.json"
FINAL_GATE_SCENARIOS = DIR_05A1 / "final_gate_calibration_scenarios.csv"
FINAL_GATE_REPLICATES = DIR_05A1 / "final_gate_calibration_replicates.csv"
GATE_METRICS_05A1 = DIR_05A1 / "stability_gate_operating_metric_registry.csv"
GATE_DECISIONS_05A1 = DIR_05A1 / "stability_gate_decision_registry.csv"

DIR_05B_V2 = PAPER5_ROOT / "method_contract" / "05b_v2"
LOCK_05B_V2 = DIR_05B_V2 / "simulation_generation_v2_lock.json"

DIR_05D3 = PAPER5_ROOT / "method_contract" / "05d3"
LOCK_05D3 = DIR_05D3 / "selected_neural_profile_lock.json"

DIR_05D4 = PAPER5_ROOT / "method_contract" / "05d4"
LOCK_05D4 = DIR_05D4 / "neural_confirmatory_ablation_results_lock.json"

DIR_05D4A = PAPER5_ROOT / "method_contract" / "05d4a"
LOCK_05D4A = DIR_05D4A / "05d4a_results_lock.json"

DIR_05D4B = PAPER5_ROOT / "method_contract" / "05d4b"
LOCK_05D4B = DIR_05D4B / "05d4b_results_lock.json"
AUDIT_05D4B = DIR_05D4B / "05d4b_final_gate_design_audit.json"
REQUIREMENTS_05D4B = DIR_05D4B / "05d4b_05a2_amendment_requirements.json"
SELECTORS_05D4B = DIR_05D4B / "05d4b_final_gate_selector_membership.csv"

DIR_05D5 = PAPER5_ROOT / "method_contract" / "05d5"
LOCK_05D5 = DIR_05D5 / "neural_confirmatory_stability_results_lock.json"
IN_PROGRESS_05D5 = DIR_05D5 / ".neural_confirmatory_stability_in_progress.json"

FINAL_GATE_DATA_ROOT = PAPER5_ROOT / "simulations" / "sable_mo_v2" / "final_gate_calibration"

# Future result roots whose existence would invalidate this prospective amendment.
FORBIDDEN_FUTURE_RESULT_ROOTS = (
    PAPER5_ROOT / "method_contract" / "05e",
    PAPER5_ROOT / "results" / "representation_benchmark",
    PAPER5_ROOT / "results" / "target_representation",
    PAPER5_ROOT / "results" / "biological_alignment",
    PAPER5_ROOT / "results" / "outcome_analysis",
    PAPER5_ROOT / "results" / "outcome_transport",
)

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05a2"
OUTPUT_AMENDMENT = OUTPUT_DIR / "inference_final_gate_implementation_amendment.json"
OUTPUT_RANDOM_CONTROLS = OUTPUT_DIR / "effective_random_control_registry.csv"
OUTPUT_GATE_METRICS = OUTPUT_DIR / "effective_stability_gate_operating_metric_registry.csv"
OUTPUT_GATE_DECISIONS = OUTPUT_DIR / "effective_stability_gate_decision_registry.csv"
OUTPUT_REPORTING = OUTPUT_DIR / "final_gate_reporting_registry.csv"
OUTPUT_FAILURE_POLICY = OUTPUT_DIR / "final_gate_failure_consequence_policy.json"
OUTPUT_RUNTIME_POLICY = OUTPUT_DIR / "final_gate_runtime_pilot_policy.json"
OUTPUT_PRECONDITION = OUTPUT_DIR / "amendment_precondition_snapshot.json"
OUTPUT_README = OUTPUT_DIR / "05a2_README.txt"
OUTPUT_LOCK = OUTPUT_DIR / "inference_final_gate_implementation_amendment_lock.json"


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na"}:
        return ""
    return text


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = clean_text(value).lower()
    if not text:
        return default
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean: {value!r}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_project_path(value: Any) -> Path:
    path = Path(clean_text(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_json(path: Path) -> dict[str, Any]:
    require_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv_atomic(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    table.to_csv(
        temporary,
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    os.replace(temporary, path)


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def verify_generated_files(lock: dict[str, Any], label: str) -> None:
    failures: list[str] = []
    for relative_name, record in lock.get("generated_files", {}).items():
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
            failures.append(f"SHA-256 mismatch: {relative_name}")

    if failures:
        raise RuntimeError(
            f"{label} generated-file verification failed:\n- "
            + "\n- ".join(failures[:100])
        )


def any_files(path: Path) -> bool:
    return path.is_file() or (path.is_dir() and any(item.is_file() for item in path.rglob("*")))


# -----------------------------------------------------------------------------
# Upstream verification
# -----------------------------------------------------------------------------

def verify_05a() -> dict[str, Any]:
    for path in [
        CONTRACT_05A,
        LOCK_05A,
        RANDOM_CONTROLS_05A,
        MULTIPLICITY_05A,
    ]:
        require_file(path)

    contract = load_json(CONTRACT_05A)
    lock = load_json(LOCK_05A)

    observed_contract_hash = sha256_file(CONTRACT_05A)
    expected_contract_hash = clean_text(lock.get("contract_sha256"))
    if observed_contract_hash != expected_contract_hash:
        raise RuntimeError("05a contract hash does not match its frozen lock.")

    verify_generated_files(lock, "05a")

    state = contract.get("scientific_state", {})
    if bool(state.get("outcome_loaded", True)):
        raise RuntimeError("05a no longer confirms outcome_loaded=false.")
    if bool(state.get("target_benchmark_results_viewed", True)):
        raise RuntimeError("05a unexpectedly says TARGET benchmark results were viewed at freeze.")

    random_controls = pd.read_csv(RANDOM_CONTROLS_05A, dtype=str).fillna("")
    rc01 = random_controls[random_controls["control_id"].eq("RC01_PAIRING_PERMUTATION")]
    if rc01.shape[0] != 1:
        raise RuntimeError("Expected exactly one RC01_PAIRING_PERMUTATION row.")
    if int(float(rc01.iloc[0]["n_replicates"])) != PAIRING_PERMUTATIONS_BASE:
        raise RuntimeError("Frozen 05a pairing-permutation count is no longer 100.")

    multiplicity = pd.read_csv(MULTIPLICITY_05A, dtype=str).fillna("")
    mf01 = multiplicity[multiplicity["family_id"].eq("MF01_REPRESENTATION_PAIRING")]
    if mf01.shape[0] != 1:
        raise RuntimeError("Expected exactly one MF01_REPRESENTATION_PAIRING row.")
    if "benjamini-hochberg" not in clean_text(mf01.iloc[0]["adjustment"]).lower():
        raise RuntimeError("MF01 pairing multiplicity is no longer Benjamini-Hochberg.")

    return {
        "contract": contract,
        "lock": lock,
        "contract_sha256": observed_contract_hash,
        "lock_sha256": sha256_file(LOCK_05A),
        "random_controls": random_controls,
        "multiplicity": multiplicity,
    }


def verify_05a1() -> dict[str, Any]:
    for path in [
        CONTRACT_05A1,
        LOCK_05A1,
        FINAL_GATE_SCENARIOS,
        FINAL_GATE_REPLICATES,
        GATE_METRICS_05A1,
        GATE_DECISIONS_05A1,
    ]:
        require_file(path)

    addendum = load_json(CONTRACT_05A1)
    lock = load_json(LOCK_05A1)

    observed_hash = sha256_file(CONTRACT_05A1)
    expected_hash = clean_text(lock.get("addendum_sha256"))
    if observed_hash != expected_hash:
        raise RuntimeError("05a1 addendum hash does not match its frozen lock.")

    verify_generated_files(lock, "05a1")

    final_gate = addendum.get("final_gate_calibration", {})
    if int(final_gate.get("n_scenarios", -1)) != EXPECTED_FINAL_GATE_SCENARIOS:
        raise RuntimeError("05a1 final-gate scenario count is no longer 24.")
    if int(final_gate.get("n_planned_datasets", -1)) != EXPECTED_FINAL_GATE_DATASETS:
        raise RuntimeError("05a1 final-gate dataset count is no longer 3,100.")
    if int(final_gate.get("stability_resamples_per_dataset", -1)) != EXPECTED_FINAL_GATE_STABILITY_RESAMPLES:
        raise RuntimeError("05a1 final-gate stability resamples are no longer 100.")
    if bool(final_gate.get("threshold_retuning_after_results", True)):
        raise RuntimeError("05a1 unexpectedly permits post-result threshold retuning.")

    scenarios = pd.read_csv(FINAL_GATE_SCENARIOS, dtype=str, low_memory=False).fillna("")
    replicates = pd.read_csv(FINAL_GATE_REPLICATES, dtype=str, low_memory=False).fillna("")
    metrics = pd.read_csv(GATE_METRICS_05A1, dtype=str, low_memory=False).fillna("")
    decisions = pd.read_csv(GATE_DECISIONS_05A1, dtype=str, low_memory=False).fillna("")

    if scenarios.shape[0] != EXPECTED_FINAL_GATE_SCENARIOS:
        raise RuntimeError("05a1 scenario registry does not contain exactly 24 rows.")
    if replicates.shape[0] != EXPECTED_FINAL_GATE_DATASETS:
        raise RuntimeError("05a1 replicate registry does not contain exactly 3,100 rows.")

    return {
        "addendum": addendum,
        "lock": lock,
        "addendum_sha256": observed_hash,
        "lock_sha256": sha256_file(LOCK_05A1),
        "scenarios": scenarios,
        "replicates": replicates,
        "metrics": metrics,
        "decisions": decisions,
    }


def verify_05b_v2_and_no_final_gate_data() -> dict[str, Any]:
    require_file(LOCK_05B_V2)
    lock = load_json(LOCK_05B_V2)
    verify_generated_files(lock, "05b v2")

    scope = lock.get("scope", {})
    if bool(scope.get("final_gate_calibration_materialized", True)):
        raise RuntimeError("05b v2 lock says final-gate data were materialized.")
    if int(scope.get("final_gate_registry_rows", -1)) != EXPECTED_FINAL_GATE_DATASETS:
        raise RuntimeError("05b v2 lock final-gate registry count is not 3,100.")

    materialized: list[Path] = []
    if FINAL_GATE_DATA_ROOT.exists():
        materialized = [path for path in FINAL_GATE_DATA_ROOT.rglob("*") if path.is_file()]
    if materialized:
        raise RuntimeError(
            "Final-gate simulation data already exist; 05a2 must not be frozen now.\n- "
            + "\n- ".join(relative_path(path) for path in materialized[:50])
        )

    for root in FORBIDDEN_FUTURE_RESULT_ROOTS:
        if any_files(root):
            raise RuntimeError(
                "A future result root already contains files, so this amendment is no longer prospective:\n"
                f"- {relative_path(root)}"
            )

    return {
        "lock_sha256": sha256_file(LOCK_05B_V2),
        "final_gate_materialized_files": 0,
        "future_result_roots_with_files": 0,
    }


def verify_05d3_05d4_05d4a_05d4b() -> dict[str, Any]:
    for path in [LOCK_05D3, LOCK_05D4, LOCK_05D4A, LOCK_05D4B, AUDIT_05D4B, REQUIREMENTS_05D4B, SELECTORS_05D4B]:
        require_file(path)

    lock_05d3 = load_json(LOCK_05D3)
    if clean_text(lock_05d3.get("status")) != "FROZEN_OUTCOME_BLIND_NEURAL_PROFILE_BEFORE_CONFIRMATORY_ABLATION":
        raise RuntimeError("Unexpected 05d3 lock status.")
    verify_generated_files(lock_05d3, "05d3")

    lock_05d4 = load_json(LOCK_05D4)
    if clean_text(lock_05d4.get("status")) != "FROZEN_NEURAL_CONFIRMATORY_ABLATION_RESULTS_BEFORE_05D5":
        raise RuntimeError("05d4 is not a clean frozen confirmatory-ablation result.")
    verify_generated_files(lock_05d4, "05d4")

    lock_05d4a = load_json(LOCK_05D4A)
    if clean_text(lock_05d4a.get("status")) != "DERIVED_READ_ONLY_AUDIT_OF_FROZEN_05D4_RESULTS":
        raise RuntimeError("Unexpected 05d4a lock status.")
    verify_generated_files(lock_05d4a, "05d4a")

    lock_05d4b = load_json(LOCK_05D4B)
    if clean_text(lock_05d4b.get("status")) != "DERIVED_READ_ONLY_FINAL_GATE_DESIGN_AUDIT":
        raise RuntimeError("Unexpected 05d4b lock status.")
    verify_generated_files(lock_05d4b, "05d4b")

    audit = load_json(AUDIT_05D4B)
    scientific_state = audit.get("scientific_state", {})
    forbidden_true = [
        "GPU_computation_performed",
        "model_fitting_performed",
        "TARGET_accessed",
        "outcomes_accessed",
        "biological_annotations_accessed",
        "final_gate_simulation_data_accessed",
        "final_gate_simulation_data_materialized",
        "threshold_retuning_performed",
        "decision_rule_rewritten",
    ]
    active = [key for key in forbidden_true if bool(scientific_state.get(key, False))]
    if active:
        raise RuntimeError("05d4b prospective state violated: " + ", ".join(active))

    selectors = pd.read_csv(SELECTORS_05D4B, dtype=str, low_memory=False).fillna("")
    d01 = selectors[selectors["D01_current_selector"].str.lower().eq("true")]
    d01_ind = selectors[selectors["D01_independent_null_component"].str.lower().eq("true")]
    d01_corr = selectors[selectors["D01_correlated_private_component"].str.lower().eq("true")]
    d02_current = selectors[selectors["D02_current_selector"].str.lower().eq("true")]
    d02_standard = selectors[
        selectors["D02_D03_standard_candidate_selector"].str.lower().eq("true")
    ]

    def n_datasets(table: pd.DataFrame) -> int:
        return int(pd.to_numeric(table["replicate_rows"], errors="raise").sum())

    def n_true_factor_opportunities(table: pd.DataFrame) -> int:
        return int(
            (
                pd.to_numeric(table["replicate_rows"], errors="raise")
                * pd.to_numeric(table["true_biological_shared_factors"], errors="raise")
            ).sum()
        )

    if n_datasets(d01) != EXPECTED_D01_DATASETS:
        raise RuntimeError("05d4b D01 selector no longer resolves to 1,000 datasets.")
    if n_datasets(d01_ind) != EXPECTED_D01_INDEPENDENT_DATASETS:
        raise RuntimeError("05d4b independent D01 subset no longer resolves to 800 datasets.")
    if n_datasets(d01_corr) != EXPECTED_D01_CORRELATED_PRIVATE_DATASETS:
        raise RuntimeError("05d4b correlated-private D01 subset no longer resolves to 200 datasets.")
    if d02_current.shape[0] != 4:
        raise RuntimeError("05d4b current D02 selector no longer matches four scenarios.")
    if d02_standard.shape[0] != 1 or n_datasets(d02_standard) != EXPECTED_STANDARD_D02_D03_DATASETS:
        raise RuntimeError("05d4b exact STANDARD selector no longer resolves to one 100-dataset scenario.")
    if n_true_factor_opportunities(d02_standard) != EXPECTED_STANDARD_TRUE_FACTOR_OPPORTUNITIES:
        raise RuntimeError("Exact STANDARD D02/D03 scenario no longer contains 400 true-factor opportunities.")

    return {
        "05d3_lock_sha256": sha256_file(LOCK_05D3),
        "05d4_lock_sha256": sha256_file(LOCK_05D4),
        "05d4a_lock_sha256": sha256_file(LOCK_05D4A),
        "05d4b_lock_sha256": sha256_file(LOCK_05D4B),
        "05d4b_audit_sha256": sha256_file(AUDIT_05D4B),
        "05d4b_requirements_sha256": sha256_file(REQUIREMENTS_05D4B),
        "05d4b_selector_sha256": sha256_file(SELECTORS_05D4B),
        "d01_datasets": n_datasets(d01),
        "d01_independent_datasets": n_datasets(d01_ind),
        "d01_correlated_private_datasets": n_datasets(d01_corr),
        "d02_current_scenarios": d02_current["scenario_id"].tolist(),
        "d02_standard_scenarios": d02_standard["scenario_id"].tolist(),
        "d02_standard_datasets": n_datasets(d02_standard),
        "d02_standard_true_factor_opportunities": n_true_factor_opportunities(d02_standard),
    }


# -----------------------------------------------------------------------------
# Effective amended registries
# -----------------------------------------------------------------------------

def build_effective_random_controls(base: pd.DataFrame) -> pd.DataFrame:
    table = base.copy()
    table["base_n_replicates"] = table["n_replicates"].astype(str)
    table["amended_by"] = ""
    table["amendment_reason"] = ""

    mask = table["control_id"].eq("RC01_PAIRING_PERMUTATION")
    if mask.sum() != 1:
        raise RuntimeError("Expected one RC01 row when building the effective registry.")

    table.loc[mask, "n_replicates"] = str(PAIRING_PERMUTATIONS_EFFECTIVE)
    table.loc[mask, "amended_by"] = "05a2"
    table.loc[mask, "amendment_reason"] = (
        "Monte Carlo p-value resolution only; null construction, metrics, BH family, "
        "model registry, and outcome blinding are unchanged."
    )
    return table


def build_effective_gate_metrics(base: pd.DataFrame) -> pd.DataFrame:
    table = base.copy()
    table["effective_aggregation_rule"] = ""
    table["effective_confidence_interval_rule"] = ""
    table["decision_uses_confidence_bound"] = ""
    table["amended_by"] = ""
    table["amendment_note"] = ""

    def set_row(metric_id: str, **updates: Any) -> None:
        mask = table["metric_id"].eq(metric_id)
        if mask.sum() != 1:
            raise RuntimeError(f"Expected one metric row for {metric_id}.")
        for column, value in updates.items():
            table.loc[mask, column] = value

    set_row(
        "gate_familywise_false_acceptance",
        effective_aggregation_rule=(
            "Primary D01 pools dataset-level false-acceptance events across every dataset "
            "matching the frozen D01 selector in the prespecified final-gate null registry."
        ),
        effective_confidence_interval_rule=(
            "Wilson score interval; two-sided 95%; z=1.959963984540054; "
            "no continuity correction."
        ),
        decision_uses_confidence_bound="upper endpoint",
        amended_by="05a2",
        amendment_note=(
            "Clarifies previously unspecified pooling and Wilson implementation. "
            "Point estimate and numerator/denominator are reported before CI and PASS/FAIL."
        ),
    )

    set_row(
        "gate_true_positive_rate",
        effective_aggregation_rule=(
            "Within exact STANDARD scenario G07_IND_N62_S1P00, pool factor-level truth opportunities: "
            "total true biological shared factors matched-and-retained divided by total true biological "
            "shared-factor opportunities across the 100 datasets (400 opportunities). Because each dataset "
            "contains exactly four true biological shared factors, this point estimate is algebraically equal "
            "to the unweighted mean of dataset-level TPR values."
        ),
        effective_confidence_interval_rule=(
            "Retain the base registry's Wilson 95% binomial interval for reporting; "
            "the D02 PASS/FAIL rule uses the point estimate, not a confidence bound."
        ),
        decision_uses_confidence_bound="NO",
        amended_by="05a2",
        amendment_note=(
            "Clarifies cross-dataset aggregation before final-gate results. "
            "No sensitivity threshold is changed."
        ),
    )

    set_row(
        "gate_false_negative_rate",
        effective_aggregation_rule=(
            "Defined exactly as 1 - gate_true_positive_rate on the same G07 pooled "
            "factor-opportunity denominator."
        ),
        effective_confidence_interval_rule=(
            "Mirrors the TPR reporting convention; D03 PASS/FAIL uses the point estimate."
        ),
        decision_uses_confidence_bound="NO",
        amended_by="05a2",
        amendment_note=(
            "D03 is a mirrored restatement of D02 on the same denominator and is not "
            "independent sensitivity evidence."
        ),
    )

    return table


def build_effective_gate_decisions(base: pd.DataFrame) -> pd.DataFrame:
    table = base.copy()
    table["base_scenario_selector"] = table["scenario_selector"].astype(str)
    table["effective_aggregation_rule"] = ""
    table["effective_ci_rule"] = ""
    table["decision_independence_note"] = ""
    table["amended_by"] = ""
    table["amendment_rationale"] = ""

    def set_row(decision_id: str, **updates: Any) -> None:
        mask = table["decision_id"].eq(decision_id)
        if mask.sum() != 1:
            raise RuntimeError(f"Expected one decision row for {decision_id}.")
        for column, value in updates.items():
            table.loc[mask, column] = value

    set_row(
        "D01_NULL_SPECIFICITY",
        scenario_selector=(
            "true_biological_shared_factors == 0 and true_technical_shared_factors == 0"
        ),
        effective_aggregation_rule=(
            "Pool all dataset-level false-acceptance events across the complete prespecified "
            "D01 null registry selected by this expression (N=1000)."
        ),
        effective_ci_rule=(
            "upper endpoint of two-sided 95% Wilson score interval, "
            "z=1.959963984540054, no continuity correction; PASS if upper <= 0.10"
        ),
        amended_by="05a2",
        amendment_rationale=(
            "The frozen selector and 10% limit are unchanged. 05a2 only fixes aggregation "
            "and CI implementation before final-gate data exist."
        ),
    )

    standard_selector = (
        "scenario_family == 'final_independent_gate_curve' and "
        "n_samples == 62 and shared_signal_strength == 1.0 and "
        "shared_factor_correlation == 0 and batch_mode == 'none'"
    )

    rationale = (
        "The base selector unintentionally also matched MIXED_SUPPORT, SPARSE_LOADING, and "
        "DENSE_LOADING stress scenarios. Restricting to the exact STANDARD independent "
        "operating point probably makes D02/D03 easier to pass, and this favorable direction "
        "is explicitly acknowledged. It also reduces the selected dataset count from 400 to 100. "
        "The clarification is nevertheless adopted because the prospectively frozen decision "
        "names use STANDARD and the operating-target names use standard_n62_signal1, while no "
        "final-gate dataset exists at this amendment freeze."
    )

    set_row(
        "D02_STANDARD_SENSITIVITY",
        scenario_selector=standard_selector,
        effective_aggregation_rule=(
            "G07 only; total matched-and-retained true biological shared factors / 400 "
            "true biological shared-factor opportunities across 100 datasets."
        ),
        effective_ci_rule=(
            "Point estimate >= 0.80 determines PASS/FAIL; CI is reported but does not determine the decision."
        ),
        decision_independence_note=(
            "Primary sensitivity operating target; D03 is its algebraic mirror, not independent evidence."
        ),
        amended_by="05a2",
        amendment_rationale=rationale,
    )

    set_row(
        "D03_STANDARD_FNR",
        scenario_selector=standard_selector,
        effective_aggregation_rule=(
            "G07 only; FNR = 1 - D02 TPR on the identical pooled 400-factor denominator."
        ),
        effective_ci_rule=(
            "Point estimate <= 0.20 determines PASS/FAIL; CI is reported but does not determine the decision."
        ),
        decision_independence_note=(
            "Mirrored restatement of D02 on the same selector and denominator; not a second independent sensitivity test."
        ),
        amended_by="05a2",
        amendment_rationale=rationale,
    )

    return table


def build_reporting_registry() -> pd.DataFrame:
    rows = [
        {
            "report_id": "R01_D01_PRIMARY_POOLED",
            "role": "primary_decision",
            "metric": "gate_familywise_false_acceptance",
            "selector": (
                "true_biological_shared_factors == 0 and true_technical_shared_factors == 0"
            ),
            "expected_datasets": 1000,
            "reported_statistic": "point estimate; numerator/denominator; two-sided 95% Wilson CI; PASS/FAIL",
            "formal_gate_consequence": "D01",
            "notes": (
                "Estimand is familywise false biological acceptance over the prespecified final-gate null registry."
            ),
        },
        {
            "report_id": "R02_D01_INDEPENDENT_NULL_POOLED",
            "role": "mandatory_secondary",
            "metric": "gate_familywise_false_acceptance",
            "selector": (
                "D01 selector AND scenario_family == 'final_independent_gate_curve'"
            ),
            "expected_datasets": 800,
            "reported_statistic": "point estimate; numerator/denominator; two-sided 95% Wilson CI",
            "formal_gate_consequence": "NONE",
            "notes": "Separates ordinary independent nulls from the correlated-private stress condition.",
        },
        {
            "report_id": "R03_D01_CORRELATED_PRIVATE",
            "role": "mandatory_secondary",
            "metric": "gate_familywise_false_acceptance",
            "selector": "D01 selector AND scenario_family == 'correlated_private_null'",
            "expected_datasets": 200,
            "reported_statistic": "point estimate; numerator/denominator; two-sided 95% Wilson CI",
            "formal_gate_consequence": "NONE",
            "notes": "Stress-null component of the primary mixture.",
        },
    ]

    for n_value in [50, 62, 75, 100]:
        rows.append(
            {
                "report_id": f"R_D01_INDEPENDENT_N{n_value}",
                "role": "mandatory_descriptive",
                "metric": "gate_familywise_false_acceptance",
                "selector": (
                    "D01 selector AND scenario_family == 'final_independent_gate_curve' "
                    f"AND n_samples == {n_value}"
                ),
                "expected_datasets": 200,
                "reported_statistic": "point estimate; numerator/denominator; two-sided 95% Wilson CI",
                "formal_gate_consequence": "NONE",
                "notes": (
                    "Descriptive sample-size breakdown. No claim of homogeneity is permitted from these cells."
                ),
            }
        )

    rows.extend(
        [
            {
                "report_id": "R08_D01_SAMPLE_SIZE_TREND",
                "role": "mandatory_descriptive",
                "metric": "dataset_level_false_biological_acceptance_event",
                "selector": "four final_independent_gate_curve null scenarios only",
                "expected_datasets": 800,
                "reported_statistic": (
                    "binomial logistic regression; predictor=n_samples/25; OR per +25 samples; "
                    "two-sided 95% CI"
                ),
                "formal_gate_consequence": "NONE",
                "notes": (
                    "Raw n scale fixed prospectively. Correlated-private null is excluded to avoid confounding n with null mechanism."
                ),
            },
            {
                "report_id": "R09_D02_STANDARD_TPR",
                "role": "primary_decision",
                "metric": "gate_true_positive_rate",
                "selector": "G07_IND_N62_S1P00 only",
                "expected_datasets": 100,
                "reported_statistic": (
                    "pooled factor-level TPR over 400 true-factor opportunities; point estimate first; "
                    "CI second; PASS/FAIL third"
                ),
                "formal_gate_consequence": "D02",
                "notes": "Point-estimate decision threshold remains >=0.80.",
            },
            {
                "report_id": "R10_D03_STANDARD_FNR",
                "role": "mirrored_primary_decision",
                "metric": "gate_false_negative_rate",
                "selector": "G07_IND_N62_S1P00 only",
                "expected_datasets": 100,
                "reported_statistic": "1 - R09 TPR; point estimate first; CI second; PASS/FAIL third",
                "formal_gate_consequence": "D03",
                "notes": "Not independent evidence from D02.",
            },
        ]
    )

    return pd.DataFrame(rows)


def build_failure_policy() -> dict[str, Any]:
    return {
        "policy_version": "sable-mo-final-gate-failure-consequence-policy-v1",
        "status": "FROZEN_BEFORE_FINAL_GATE_RESULTS",
        "governing_rule": (
            "A preregistered final-gate failure changes the scientific conclusion, not the criterion. "
            "No threshold, architecture, stability rule, eligibility rule, or selector may be retuned "
            "in response to final-gate results."
        ),
        "core_outcomes": [
            {
                "condition": "D01 PASS and D02 PASS and D03 PASS",
                "interpretation": (
                    "The frozen gate meets the preregistered core specificity and standard-operating-point "
                    "sensitivity criteria in the final simulation bank."
                ),
                "downstream": (
                    "Proceed with the planned outcome-blind TARGET representation benchmark and subsequent "
                    "frozen stability/biology pipeline, subject to all other frozen gates."
                ),
            },
            {
                "condition": "D01 FAIL and D02/D03 PASS",
                "interpretation": (
                    "The gate can recover true shared signal at the standard operating point but does not "
                    "achieve the preregistered control of false biological acceptance over the prespecified null registry."
                ),
                "downstream": (
                    "Do not retune the gate. TARGET representation analysis may be reported only as outcome-blind "
                    "descriptive characterization. Do not use biological or outcome results to rescue a validated "
                    "factor-acceptance claim, and do not make primary factor-level biological/outcome claims."
                ),
            },
            {
                "condition": "D01 PASS and D02/D03 FAIL",
                "interpretation": (
                    "The gate is sufficiently specific under the preregistered null criterion but is too insensitive "
                    "or too conservative at the exact TARGET-like standard operating point."
                ),
                "downstream": (
                    "Do not loosen the gate. TARGET representation analysis may be descriptive, but the paper must "
                    "state the sensitivity limitation and may not claim reliable recovery of true shared factors in "
                    "the target small-n regime."
                ),
            },
            {
                "condition": "D01 FAIL and D02/D03 FAIL",
                "interpretation": (
                    "The frozen method is not validated for the claimed extreme-small-n factor-acceptance regime."
                ),
                "downstream": (
                    "Paper 5 becomes a negative/boundary methods result on the limits of stable multi-omics "
                    "representation learning. No post-hoc rescue retuning is permitted, and outcomes are not opened "
                    "for the purpose of rescuing the method."
                ),
            },
        ],
        "narrow_gate_failures": {
            "D04_ROTATIONAL_SUBSPACE": (
                "If failed, do not claim reliable recovery of rotationally ambiguous subspaces; individual-factor "
                "claims remain forbidden as already specified."
            ),
            "D05_VIEW_SUPPORT": (
                "If failed, view-support labels remain exploratory and cannot support primary shared-versus-partial claims."
            ),
            "D06_BATCH_SPECIFICITY": (
                "If failed, stable factors cannot be assumed biological without external recurrence; technical-confound "
                "limitations become a primary conclusion rather than a hidden caveat."
            ),
            "D07_NO_POSTHOC_RETUNING": (
                "Any alternative threshold explored after results is explicitly exploratory and cannot replace the "
                "primary frozen gate."
            ),
        },
        "stress_subgroup_rule": (
            "A favorable pooled D01 result may not suppress poor prespecified subgroup behavior. Per-n and null-mechanism "
            "results are mandatory descriptive reporting, but they do not create new post-hoc PASS/FAIL thresholds."
        ),
        "outcome_boundary": (
            "Outcomes are not opened to rescue a failed core representation/gate validation. Outcome analysis, if reached, "
            "is a minimal downstream validation of a representation already frozen without outcomes."
        ),
    }


def build_runtime_policy() -> dict[str, Any]:
    return {
        "policy_version": "sable-mo-final-gate-runtime-pilot-policy-v1",
        "status": "FROZEN_BEFORE_FINAL_GATE_RUNTIME_PILOT",
        "purpose": "Measure production wall time, VRAM, shard size, and ETA without making scientific decisions.",
        "pilot_dataset_selection": (
            "Use the first prespecified dataset IDs in frozen final_gate_calibration_replicates.csv order; "
            "do not select pilot datasets based on observed scientific results or runtime difficulty."
        ),
        "pilot_scientific_output_visibility": (
            "During runtime pilot execution, print timing/VRAM/progress only. Do not print or aggregate TPR, FNR, FDP, "
            "familywise false acceptance, or final-gate PASS/FAIL."
        ),
        "reuse_rule": (
            "Pilot shards may be retained as final shards only when the numerical implementation hash is unchanged."
        ),
        "invalidation_rule": (
            "Any change to the numerical/GPU code path, dtype, reduction order, preprocessing math, model fitting path, "
            "alignment math, or stability computation invalidates all pilot shards; they must be recomputed."
        ),
        "non_numerical_change_rule": (
            "Logging/progress/checkpoint-wrapper changes may retain pilot shards only if the numerical implementation "
            "hash is unchanged and the shard validation hash passes."
        ),
        "required_shard_provenance": [
            "scientific_contract_hash",
            "05a2_amendment_hash",
            "script_hash",
            "numerical_implementation_hash",
            "PyTorch_version",
            "CUDA_build",
            "GPU_model",
        ],
    }


# -----------------------------------------------------------------------------
# Precondition snapshot and amendment
# -----------------------------------------------------------------------------

def build_precondition_snapshot(
    source_05a: dict[str, Any],
    source_05a1: dict[str, Any],
    source_05b: dict[str, Any],
    source_audits: dict[str, Any],
) -> dict[str, Any]:
    return {
        "snapshot_version": "paper5-05a2-precondition-snapshot-v1",
        "created_at_utc": utc_now_iso(),
        "scientific_timing": {
            "05d4_confirmatory_ablation_results_exist_and_were_available": True,
            "05d4a_read_only_ablation_audit_exists": True,
            "05d4b_final_gate_design_audit_exists": True,
            "05d5_status_at_amendment": (
                "COMPLETE"
                if LOCK_05D5.is_file()
                else ("IN_PROGRESS" if IN_PROGRESS_05D5.is_file() else "NOT_STARTED_OR_NOT_DETECTED")
            ),
            "05d5_results_used_to_define_05a2": False,
            "final_gate_datasets_materialized": False,
            "final_gate_results_viewed": False,
            "TARGET_representation_results_viewed": False,
            "TARGET_pairing_permutation_results_viewed": False,
            "biological_alignment_results_viewed": False,
            "outcomes_viewed": False,
        },
        "explicit_nonblind_disclosure": (
            "05a2 is frozen after 05d4 confirmatory-ablation results and the derived 05d4a audit were available. "
            "The amendment is prospective specifically with respect to the unmaterialized final-gate simulation bank, "
            "future TARGET representation benchmarking, pairing-permutation results, biological interpretation, and outcomes."
        ),
        "hashes": {
            "05a_contract": source_05a["contract_sha256"],
            "05a_lock": source_05a["lock_sha256"],
            "05a1_addendum": source_05a1["addendum_sha256"],
            "05a1_lock": source_05a1["lock_sha256"],
            "05b_v2_lock": source_05b["lock_sha256"],
            "05d3_lock": source_audits["05d3_lock_sha256"],
            "05d4_lock": source_audits["05d4_lock_sha256"],
            "05d4a_lock": source_audits["05d4a_lock_sha256"],
            "05d4b_lock": source_audits["05d4b_lock_sha256"],
            "05d4b_audit": source_audits["05d4b_audit_sha256"],
            "05d4b_requirements": source_audits["05d4b_requirements_sha256"],
            "05d4b_selector_membership": source_audits["05d4b_selector_sha256"],
        },
        "physical_absence_checks": {
            "final_gate_materialized_files": source_05b["final_gate_materialized_files"],
            "future_result_roots_with_files": source_05b["future_result_roots_with_files"],
        },
    }


def build_amendment(
    precondition: dict[str, Any],
    source_audits: dict[str, Any],
) -> dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "amendment_version": AMENDMENT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": AMENDMENT_STATUS,
        "method": "SABLE-MO",
        "amends": [
            "05a random-control implementation count for RC01 pairing permutations",
            "05a1 D01 final-gate aggregation and confidence-interval implementation",
            "05a1 D02/D03 exact STANDARD selector and aggregation implementation",
            "final-gate reporting and failure-consequence semantics",
            "final-gate runtime-pilot numerical-versioning policy",
        ],
        "does_not_rewrite": [
            "05a",
            "05a1",
            "05d1",
            "05d3",
            "05d4",
            "05d4a",
            "05d4b",
        ],
        "effective_overrides": {
            "RC01_PAIRING_PERMUTATION": {
                "base_n_replicates": PAIRING_PERMUTATIONS_BASE,
                "effective_n_replicates": PAIRING_PERMUTATIONS_EFFECTIVE,
                "reason": "Monte Carlo p-value resolution only.",
                "unchanged": [
                    "unit permuted",
                    "preserved structure",
                    "primary metrics",
                    "outcome blinding",
                    "MF01 Benjamini-Hochberg multiplicity family",
                ],
            },
            "D01_NULL_SPECIFICITY": {
                "selector": (
                    "true_biological_shared_factors == 0 and true_technical_shared_factors == 0"
                ),
                "primary_estimand": (
                    "familywise false biological acceptance over the prespecified final-gate null registry"
                ),
                "event": (
                    "dataset retains at least one false biological shared factor in a null dataset"
                ),
                "aggregation": "pool all matching dataset-level events; N=1000",
                "confidence_interval": {
                    "method": "Wilson score",
                    "confidence_level": D01_CONFIDENCE_LEVEL,
                    "sidedness": "two-sided",
                    "z": D01_WILSON_Z,
                    "continuity_correction": False,
                    "decision_endpoint": "upper",
                    "equivalent_one_sided_confidence_level": 0.975,
                },
                "threshold": D01_LIMIT,
                "pass_rule": "upper two-sided 95% Wilson endpoint <= 0.10",
                "mandatory_secondary_reporting": [
                    "independent-null pooled N=800",
                    "correlated-private stress null N=200",
                    "independent-null n=50,62,75,100 separately, N=200 each",
                    "descriptive logistic trend on raw n scale, OR per +25 samples",
                ],
            },
            "D02_STANDARD_SENSITIVITY": {
                "selector": (
                    "scenario_family == 'final_independent_gate_curve' and "
                    "n_samples == 62 and shared_signal_strength == 1.0 and "
                    "shared_factor_correlation == 0 and batch_mode == 'none'"
                ),
                "resolved_scenario": "G07_IND_N62_S1P00",
                "datasets": EXPECTED_STANDARD_D02_D03_DATASETS,
                "true_factor_opportunities": EXPECTED_STANDARD_TRUE_FACTOR_OPPORTUNITIES,
                "aggregation": (
                    "total true biological shared factors matched-and-retained / "
                    "400 true biological shared-factor opportunities"
                ),
                "threshold": D02_TPR_MIN,
                "pass_rule": "point estimate >= 0.80",
                "confidence_bound_used_for_decision": False,
                "favorable_direction_disclosure": (
                    "This selector clarification probably increases TPR by excluding harder mixed-support "
                    "and sparse/dense loading stress scenarios; it also reduces selected datasets from 400 to 100. "
                    "The clarification is frozen before final-gate data exist because STANDARD and "
                    "standard_n62_signal1 were already prospectively named."
                ),
            },
            "D03_STANDARD_FNR": {
                "selector": "identical to D02",
                "resolved_scenario": "G07_IND_N62_S1P00",
                "aggregation": "FNR = 1 - D02 TPR on the identical pooled denominator",
                "threshold": D03_FNR_MAX,
                "pass_rule": "point estimate <= 0.20",
                "confidence_bound_used_for_decision": False,
                "independence_status": "algebraic mirror of D02, not independent evidence",
            },
            "factor_level_stability_thresholds": {
                "decision_type": "point-estimate factor rules",
                "confidence_bounds_used_for_factor_acceptance": False,
                "unchanged_thresholds": {
                    "median_aligned_correlation_min": 0.70,
                    "q10_aligned_correlation_min": 0.40,
                    "inclusion_frequency_min": 0.80,
                },
            },
        },
        "specificity_sensitivity_evidence_asymmetry": (
            "D01 uses an uncertainty-aware upper confidence bound, whereas D02/D03 use point-estimate "
            "operating targets. This asymmetry is preserved rather than harmonized post hoc because it was "
            "already present in the frozen design and reflects the preregistered priority on avoiding false "
            "biological acceptance."
        ),
        "unchanged_scientific_design": {
            "final_gate_scenarios": EXPECTED_FINAL_GATE_SCENARIOS,
            "final_gate_datasets": EXPECTED_FINAL_GATE_DATASETS,
            "stability_resamples_per_dataset": EXPECTED_FINAL_GATE_STABILITY_RESAMPLES,
            "D01_threshold": D01_LIMIT,
            "D02_threshold": D02_TPR_MIN,
            "D03_threshold": D03_FNR_MAX,
            "architecture_profile": "unchanged from 05d3",
            "representation_method_selection": "unchanged",
            "outcome_blinding": True,
        },
        "precondition_snapshot": relative_path(OUTPUT_PRECONDITION),
        "effective_registries": {
            "random_controls": relative_path(OUTPUT_RANDOM_CONTROLS),
            "gate_metrics": relative_path(OUTPUT_GATE_METRICS),
            "gate_decisions": relative_path(OUTPUT_GATE_DECISIONS),
            "reporting": relative_path(OUTPUT_REPORTING),
            "failure_policy": relative_path(OUTPUT_FAILURE_POLICY),
            "runtime_pilot_policy": relative_path(OUTPUT_RUNTIME_POLICY),
        },
        "guardrails": [
            "No final-gate simulation dataset existed at the 05a2 freeze.",
            "05a2 is not blind to 05d4/05d4a results; this is disclosed explicitly.",
            "05d5 results are not used to define any 05a2 rule even if 05d5 is running or later completes.",
            "No TARGET representation result, pairing-permutation result, biological result, or outcome result is used.",
            "A failed frozen criterion changes the conclusion, not the criterion.",
            "No final-gate threshold or dataset count is changed by 05a2.",
            "No V5 exploratory result can alter the primary method hierarchy.",
        ],
        "selector_audit_reference": {
            "D01_datasets": source_audits["d01_datasets"],
            "D01_independent_datasets": source_audits["d01_independent_datasets"],
            "D01_correlated_private_datasets": source_audits["d01_correlated_private_datasets"],
            "D02_current_scenarios": source_audits["d02_current_scenarios"],
            "D02_standard_scenarios": source_audits["d02_standard_scenarios"],
            "D02_standard_true_factor_opportunities": source_audits[
                "d02_standard_true_factor_opportunities"
            ],
        },
    }


def build_readme(amendment: dict[str, Any], precondition: dict[str, Any]) -> str:
    lines = [
        "Paper 5 - 05a2 inference and final-gate implementation clarification",
        "=" * 78,
        "",
        f"Script version: {SCRIPT_VERSION}",
        f"Amendment version: {AMENDMENT_VERSION}",
        f"Created UTC: {amendment['created_at_utc']}",
        "",
        "Status",
        "------",
        AMENDMENT_STATUS,
        "Final-gate datasets materialized at freeze: NO",
        "TARGET representation results available at freeze: NO",
        "Outcomes available at freeze: NO",
        f"05d5 state at freeze: {precondition['scientific_timing']['05d5_status_at_amendment']}",
        "05d5 results used to define this amendment: NO",
        "",
        "Honest timing disclosure",
        "------------------------",
        "05a2 is frozen after 05d4 confirmatory ablation and the 05d4a read-only audit were available.",
        "It is prospective with respect to the unmaterialized final-gate bank, future TARGET representation",
        "benchmarking, pairing-permutation results, biology, and outcomes.",
        "",
        "Effective clarifications",
        "------------------------",
        "1. RC01 pairing permutations: 100 -> 1,000 for Monte Carlo resolution only.",
        "2. D01: all frozen-selector null datasets are pooled for the primary decision (N=1,000).",
        "3. D01: upper endpoint of a two-sided 95% Wilson score interval, z=1.959963984540054,",
        "   no continuity correction; PASS iff upper endpoint <= 0.10.",
        "4. D01 reporting: point estimate and numerator/denominator first, CI second, PASS/FAIL third;",
        "   independent-null, correlated-private, and per-n breakdowns are mandatory descriptive results.",
        "5. D02/D03: exact STANDARD scenario is G07_IND_N62_S1P00 only.",
        "6. D02: pooled factor-level denominator is 400 true-factor opportunities across 100 datasets.",
        "7. D03: exactly 1-D02 on the same denominator; it is mirrored, not independent evidence.",
        "8. Factor-level 0.70/0.40/0.80 stability thresholds remain point-estimate rules.",
        "9. PASS/FAIL consequences are frozen before results; failures change the conclusion, not thresholds.",
        "10. Runtime pilot shards are reusable only under an unchanged numerical implementation hash.",
        "",
        "D02/D03 favorable-direction disclosure",
        "---------------------------------------",
        "The selector clarification excludes mixed-support and sparse/dense loading stress scenarios and therefore",
        "probably makes the STANDARD TPR easier to pass. It also reduces selected datasets from 400 to 100.",
        "This is accepted before final-gate data exist because the frozen decision/target names explicitly said",
        "STANDARD and standard_n62_signal1; stress scenarios remain mandatory descriptive analyses.",
        "",
        "Unchanged",
        "---------",
        "Final-gate datasets: 3,100",
        "Final-gate stability resamples per dataset: 100",
        "D01 limit: 0.10",
        "D02 TPR minimum: 0.80",
        "D03 FNR maximum: 0.20",
        "Neural profile/architecture: unchanged",
        "Outcome blinding: unchanged",
        "",
        "Downstream rule",
        "---------------",
        "Every final-gate script must read 05a + 05a1 + 05a2 and treat the 05a2 effective registries as",
        "the implementation clarification layer. 05a and 05a1 remain immutable provenance artifacts.",
        "",
    ]
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Existing amendment verification
# -----------------------------------------------------------------------------

def verify_existing_amendment() -> None:
    require_file(OUTPUT_LOCK)
    require_file(OUTPUT_AMENDMENT)

    lock = load_json(OUTPUT_LOCK)
    amendment = load_json(OUTPUT_AMENDMENT)

    if clean_text(lock.get("status")) != AMENDMENT_STATUS:
        raise RuntimeError("Unexpected existing 05a2 lock status.")
    if clean_text(amendment.get("status")) != AMENDMENT_STATUS:
        raise RuntimeError("Unexpected existing 05a2 amendment status.")

    expected_amendment_hash = clean_text(lock.get("amendment_sha256"))
    observed_amendment_hash = sha256_file(OUTPUT_AMENDMENT)
    if expected_amendment_hash != observed_amendment_hash:
        raise RuntimeError("05a2 amendment JSON hash mismatch.")

    verify_generated_files(lock, "05a2")

    print("=" * 88)
    print("Existing Paper 5 05a2 inference/final-gate amendment: PASS")
    print("=" * 88)
    print(f"Amendment: {relative_path(OUTPUT_AMENDMENT)}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("No files were rewritten.")
    print("=" * 88)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    if OUTPUT_LOCK.exists():
        verify_existing_amendment()
        return

    print("=" * 88)
    print("Paper 5 - freeze inference and final-gate implementation clarification")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("GPU computation in 05a2: NO")
    print("Model fitting in 05a2: NO")
    print("Final-gate data access in 05a2: NO")
    print("TARGET/outcome access in 05a2: NO")
    print()

    source_05a = verify_05a()
    source_05a1 = verify_05a1()
    source_05b = verify_05b_v2_and_no_final_gate_data()
    source_audits = verify_05d3_05d4_05d4a_05d4b()

    print("05a frozen method contract: PASS")
    print("05a1 frozen simulation/final-gate addendum: PASS")
    print("05b v2 final-gate materialization status: NONE")
    print("05d3 frozen neural profile: PASS")
    print("05d4 confirmatory ablation result: PASS")
    print("05d4a read-only paired audit: PASS")
    print("05d4b read-only final-gate design audit: PASS")
    print("Final-gate simulation files present: 0")
    print()

    random_controls = build_effective_random_controls(source_05a["random_controls"])
    effective_metrics = build_effective_gate_metrics(source_05a1["metrics"])
    effective_decisions = build_effective_gate_decisions(source_05a1["decisions"])
    reporting = build_reporting_registry()
    failure_policy = build_failure_policy()
    runtime_policy = build_runtime_policy()

    precondition = build_precondition_snapshot(
        source_05a,
        source_05a1,
        source_05b,
        source_audits,
    )
    amendment = build_amendment(precondition, source_audits)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_csv_atomic(random_controls, OUTPUT_RANDOM_CONTROLS)
    write_csv_atomic(effective_metrics, OUTPUT_GATE_METRICS)
    write_csv_atomic(effective_decisions, OUTPUT_GATE_DECISIONS)
    write_csv_atomic(reporting, OUTPUT_REPORTING)
    write_json_atomic(failure_policy, OUTPUT_FAILURE_POLICY)
    write_json_atomic(runtime_policy, OUTPUT_RUNTIME_POLICY)
    write_json_atomic(precondition, OUTPUT_PRECONDITION)
    write_json_atomic(amendment, OUTPUT_AMENDMENT)

    OUTPUT_README.write_text(
        build_readme(amendment, precondition),
        encoding="utf-8",
    )

    generated = [
        OUTPUT_RANDOM_CONTROLS,
        OUTPUT_GATE_METRICS,
        OUTPUT_GATE_DECISIONS,
        OUTPUT_REPORTING,
        OUTPUT_FAILURE_POLICY,
        OUTPUT_RUNTIME_POLICY,
        OUTPUT_PRECONDITION,
        OUTPUT_AMENDMENT,
        OUTPUT_README,
    ]

    lock = {
        "script_version": SCRIPT_VERSION,
        "amendment_version": AMENDMENT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": AMENDMENT_STATUS,
        "amendment_path": relative_path(OUTPUT_AMENDMENT),
        "amendment_sha256": sha256_file(OUTPUT_AMENDMENT),
        "precondition_snapshot_sha256": sha256_file(OUTPUT_PRECONDITION),
        "base_hashes": precondition["hashes"],
        "generated_files": {
            relative_path(path): file_record(path)
            for path in generated
        },
        "rewrite_policy": (
            "This no-argument script never rewrites a frozen 05a2 amendment. "
            "Any future scientific change requires a new versioned amendment that preserves 05a2."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    print("-" * 88)
    print("05a2 frozen clarification summary")
    print("-" * 88)
    print("RC01 pairing permutations: 100 -> 1,000")
    print("D01 primary pool: 1,000 prespecified null-registry datasets")
    print("D01 CI: two-sided 95% Wilson, z=1.959963984540054, no continuity correction")
    print("D01 PASS: upper Wilson endpoint <= 0.10")
    print("D02/D03 exact STANDARD scenario: G07_IND_N62_S1P00")
    print("D02/D03 selected datasets: 100")
    print("D02 pooled true-factor opportunities: 400")
    print("D02 PASS: point estimate >= 0.80")
    print("D03: 1 - D02 on same denominator; mirrored, not independent evidence")
    print("Final-gate datasets changed: NO")
    print("Final-gate stability resamples changed: NO")
    print("Primary stability thresholds changed: NO")
    print("05d5 results used by amendment: NO")
    print()

    print("=" * 88)
    print("Paper 5 05a2 inference/final-gate clarification: FROZEN")
    print("=" * 88)
    print(f"Amendment: {relative_path(OUTPUT_AMENDMENT)}")
    print(f"Effective random controls: {relative_path(OUTPUT_RANDOM_CONTROLS)}")
    print(f"Effective gate metrics: {relative_path(OUTPUT_GATE_METRICS)}")
    print(f"Effective gate decisions: {relative_path(OUTPUT_GATE_DECISIONS)}")
    print(f"Reporting registry: {relative_path(OUTPUT_REPORTING)}")
    print(f"Failure policy: {relative_path(OUTPUT_FAILURE_POLICY)}")
    print(f"Runtime policy: {relative_path(OUTPUT_RUNTIME_POLICY)}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("Final-gate materialization is permitted only after this frozen 05a2 layer.")
    print("05d5 may continue running independently: YES")
    print("=" * 88)


if __name__ == "__main__":
    main()
