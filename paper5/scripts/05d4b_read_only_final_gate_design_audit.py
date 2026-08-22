from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom


SCRIPT_VERSION = "05d4b-read-only-final-gate-design-audit-v1-no-cli"
AUDIT_VERSION = "sable-mo-final-gate-design-audit-v1"

TWO_SIDED_95_Z = 1.959963984540054
ONE_SIDED_95_Z = 1.6448536269514722
D01_LIMIT = 0.10
D02_TPR_MIN = 0.80
D03_FNR_MAX = 0.20

D01_TRUE_RATE_GRID = (0.00, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12)
D01_N_GRID = (200, 800, 1000)
D02_TRUE_TPR_GRID = (0.75, 0.78, 0.80, 0.82, 0.85, 0.90)
D02_DENOMINATOR_REFERENCE_GRID = (100, 400)

EXPECTED_FINAL_GATE_SCENARIOS = 24
EXPECTED_FINAL_GATE_DATASETS = 3_100
EXPECTED_FINAL_GATE_STABILITY_RESAMPLES = 100
EXPECTED_PAIRING_PERMUTATIONS = 100


SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

CONTRACT_05A_DIR = PAPER5_ROOT / "method_contract" / "05a"
CONTRACT_05A = CONTRACT_05A_DIR / "ai_method_contract.json"
LOCK_05A = CONTRACT_05A_DIR / "ai_method_contract_lock.json"
RANDOM_CONTROLS_05A = CONTRACT_05A_DIR / "random_control_registry.csv"
MULTIPLICITY_05A = CONTRACT_05A_DIR / "multiplicity_family_registry.csv"

CONTRACT_05A1_DIR = PAPER5_ROOT / "method_contract" / "05a1"
CONTRACT_05A1 = CONTRACT_05A1_DIR / "simulation_and_technical_confound_addendum.json"
LOCK_05A1 = CONTRACT_05A1_DIR / "simulation_and_technical_confound_addendum_lock.json"
FINAL_GATE_SCENARIOS = CONTRACT_05A1_DIR / "final_gate_calibration_scenarios.csv"
FINAL_GATE_REPLICATES = CONTRACT_05A1_DIR / "final_gate_calibration_replicates.csv"
GATE_METRICS = CONTRACT_05A1_DIR / "stability_gate_operating_metric_registry.csv"
GATE_DECISIONS = CONTRACT_05A1_DIR / "stability_gate_decision_registry.csv"

CONTRACT_05B_V2_DIR = PAPER5_ROOT / "method_contract" / "05b_v2"
LOCK_05B_V2 = CONTRACT_05B_V2_DIR / "simulation_generation_v2_lock.json"
FINAL_GATE_DATA_ROOT = PAPER5_ROOT / "simulations" / "sable_mo_v2" / "final_gate_calibration"

SOURCE_05D4A_LOCK = PAPER5_ROOT / "method_contract" / "05d4a" / "05d4a_results_lock.json"

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05d4b"
OUTPUT_SELECTOR_MEMBERSHIP = OUTPUT_DIR / "05d4b_final_gate_selector_membership.csv"
OUTPUT_D01_WILSON_THRESHOLDS = OUTPUT_DIR / "05d4b_d01_wilson_thresholds.csv"
OUTPUT_D01_POWER = OUTPUT_DIR / "05d4b_d01_pass_probability_curve.csv"
OUTPUT_D02_REFERENCE = OUTPUT_DIR / "05d4b_d02_point_estimate_reference.csv"
OUTPUT_AUDIT = OUTPUT_DIR / "05d4b_final_gate_design_audit.json"
OUTPUT_AMENDMENT_REQUIREMENTS = OUTPUT_DIR / "05d4b_05a2_amendment_requirements.json"
OUTPUT_README = OUTPUT_DIR / "05d4b_README.txt"
OUTPUT_LOCK = OUTPUT_DIR / "05d4b_results_lock.json"


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
    raise ValueError(f"Cannot parse boolean value: {value!r}")


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


def write_json_atomic(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
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


# -----------------------------------------------------------------------------
# Frozen-source verification
# -----------------------------------------------------------------------------

def verify_05a() -> dict[str, Any]:
    for path in [CONTRACT_05A, LOCK_05A, RANDOM_CONTROLS_05A, MULTIPLICITY_05A]:
        require_file(path)

    contract = load_json(CONTRACT_05A)
    lock = load_json(LOCK_05A)

    expected = clean_text(lock.get("contract_sha256"))
    observed = sha256_file(CONTRACT_05A)
    if not expected or expected != observed:
        raise RuntimeError("05a AI method contract does not match its frozen lock.")

    verify_generated_files(lock, "05a")

    random_controls = pd.read_csv(RANDOM_CONTROLS_05A, dtype=str).fillna("")
    pairing = random_controls[
        random_controls["control_id"].astype(str).eq("RC01_PAIRING_PERMUTATION")
    ]
    if pairing.shape[0] != 1:
        raise RuntimeError("Expected exactly one RC01_PAIRING_PERMUTATION row.")

    current_pairing_permutations = int(float(pairing.iloc[0]["n_replicates"]))
    if current_pairing_permutations != EXPECTED_PAIRING_PERMUTATIONS:
        raise RuntimeError(
            f"Unexpected frozen pairing-permutation count: {current_pairing_permutations}"
        )

    multiplicity = pd.read_csv(MULTIPLICITY_05A, dtype=str).fillna("")
    mf01 = multiplicity[
        multiplicity["family_id"].astype(str).eq("MF01_REPRESENTATION_PAIRING")
    ]
    if mf01.shape[0] != 1:
        raise RuntimeError("Expected exactly one MF01_REPRESENTATION_PAIRING row.")
    if "benjamini-hochberg" not in clean_text(mf01.iloc[0]["adjustment"]).lower():
        raise RuntimeError("MF01 pairing multiplicity is no longer Benjamini-Hochberg.")

    return {
        "contract_sha256": observed,
        "lock_sha256": sha256_file(LOCK_05A),
        "pairing_permutations": current_pairing_permutations,
        "pairing_multiplicity": clean_text(mf01.iloc[0]["adjustment"]),
    }


def verify_05a1() -> dict[str, Any]:
    required = [
        CONTRACT_05A1,
        LOCK_05A1,
        FINAL_GATE_SCENARIOS,
        FINAL_GATE_REPLICATES,
        GATE_METRICS,
        GATE_DECISIONS,
    ]
    for path in required:
        require_file(path)

    addendum = load_json(CONTRACT_05A1)
    lock = load_json(LOCK_05A1)

    expected = clean_text(lock.get("addendum_sha256"))
    observed = sha256_file(CONTRACT_05A1)
    if not expected or expected != observed:
        raise RuntimeError("05a1 addendum does not match its frozen lock.")

    verify_generated_files(lock, "05a1")

    final_gate = addendum.get("final_gate_calibration", {})
    if int(final_gate.get("n_scenarios", -1)) != EXPECTED_FINAL_GATE_SCENARIOS:
        raise RuntimeError("05a1 final-gate scenario count differs from 24.")
    if int(final_gate.get("n_planned_datasets", -1)) != EXPECTED_FINAL_GATE_DATASETS:
        raise RuntimeError("05a1 final-gate dataset count differs from 3,100.")
    if int(final_gate.get("stability_resamples_per_dataset", -1)) != EXPECTED_FINAL_GATE_STABILITY_RESAMPLES:
        raise RuntimeError("05a1 final-gate stability-resample count differs from 100.")
    if bool(final_gate.get("threshold_retuning_after_results", True)):
        raise RuntimeError("05a1 unexpectedly permits post-result threshold retuning.")

    scenarios = pd.read_csv(FINAL_GATE_SCENARIOS, dtype=str, low_memory=False).fillna("")
    replicates = pd.read_csv(FINAL_GATE_REPLICATES, dtype=str, low_memory=False).fillna("")
    metrics = pd.read_csv(GATE_METRICS, dtype=str, low_memory=False).fillna("")
    decisions = pd.read_csv(GATE_DECISIONS, dtype=str, low_memory=False).fillna("")

    if scenarios.shape[0] != EXPECTED_FINAL_GATE_SCENARIOS:
        raise RuntimeError(f"Final-gate scenario registry has {scenarios.shape[0]} rows, expected 24.")
    if replicates.shape[0] != EXPECTED_FINAL_GATE_DATASETS:
        raise RuntimeError(f"Final-gate replicate registry has {replicates.shape[0]} rows, expected 3,100.")

    required_decisions = {
        "D01_NULL_SPECIFICITY",
        "D02_STANDARD_SENSITIVITY",
        "D03_STANDARD_FNR",
    }
    observed_decisions = set(decisions["decision_id"].astype(str))
    missing = required_decisions - observed_decisions
    if missing:
        raise RuntimeError("Missing required final-gate decisions: " + ", ".join(sorted(missing)))

    return {
        "addendum": addendum,
        "addendum_sha256": observed,
        "lock_sha256": sha256_file(LOCK_05A1),
        "scenarios": scenarios,
        "replicates": replicates,
        "metrics": metrics,
        "decisions": decisions,
        "scenario_sha256": sha256_file(FINAL_GATE_SCENARIOS),
        "replicate_sha256": sha256_file(FINAL_GATE_REPLICATES),
        "metric_sha256": sha256_file(GATE_METRICS),
        "decision_sha256": sha256_file(GATE_DECISIONS),
    }


def verify_05b_v2_no_final_gate_materialization() -> dict[str, Any]:
    require_file(LOCK_05B_V2)
    lock = load_json(LOCK_05B_V2)

    if clean_text(lock.get("status")) != "FROZEN_ARCHITECTURE_SCREENING_INPUTS_BEFORE_MODEL_FITTING":
        raise RuntimeError("Unexpected 05b v2 lock status.")

    verify_generated_files(lock, "05b v2")

    scope = lock.get("scope", {})
    if bool(scope.get("final_gate_calibration_materialized", True)):
        raise RuntimeError("05b v2 lock says final-gate data were materialized.")
    if int(scope.get("final_gate_registry_rows", -1)) != EXPECTED_FINAL_GATE_DATASETS:
        raise RuntimeError("05b v2 final-gate registry count differs from 3,100.")

    materialized_files: list[Path] = []
    if FINAL_GATE_DATA_ROOT.exists():
        materialized_files = [path for path in FINAL_GATE_DATA_ROOT.rglob("*") if path.is_file()]
    if materialized_files:
        raise RuntimeError(
            "Final-gate simulation files already exist; prospective design audit must stop.\n- "
            + "\n- ".join(relative_path(path) for path in materialized_files[:50])
        )

    return {
        "lock_sha256": sha256_file(LOCK_05B_V2),
        "final_gate_data_root": relative_path(FINAL_GATE_DATA_ROOT),
        "materialized_files": 0,
        "lock_final_gate_materialized": False,
    }


# -----------------------------------------------------------------------------
# Statistical helpers
# -----------------------------------------------------------------------------

def wilson_interval(successes: int, total: int, z: float) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    if successes < 0 or successes > total:
        raise ValueError("successes must lie in [0, total].")

    p_hat = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            (p_hat * (1.0 - p_hat) + z2 / (4.0 * total)) / total
        )
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def max_d01_pass_count(total: int, z: float) -> int:
    passing = -1
    for successes in range(total + 1):
        _, upper = wilson_interval(successes, total, z)
        if upper <= D01_LIMIT + 1e-15:
            passing = successes
        else:
            break
    return passing


def exact_probability_d01_pass(total: int, true_rate: float, z: float) -> float:
    maximum = max_d01_pass_count(total, z)
    if maximum < 0:
        return 0.0
    return float(binom.cdf(maximum, total, true_rate))


def exact_probability_point_estimate_tpr_pass(total: int, true_tpr: float) -> float:
    minimum_successes = int(math.ceil(D02_TPR_MIN * total - 1e-12))
    return float(binom.sf(minimum_successes - 1, total, true_tpr))


def minimum_n_for_d01_power(
    true_rate: float,
    desired_pass_probability: float,
    *,
    z: float,
    start_n: int = 50,
    max_n: int = 10_000,
) -> dict[str, Any]:
    for total in range(start_n, max_n + 1):
        maximum = max_d01_pass_count(total, z)
        probability = float(binom.cdf(maximum, total, true_rate))
        if probability >= desired_pass_probability:
            _, upper = wilson_interval(maximum, total, z)
            return {
                "n": total,
                "max_pass_count": maximum,
                "max_pass_observed_rate": maximum / total,
                "pass_probability": probability,
                "wilson_upper_at_boundary": upper,
            }
    raise RuntimeError("Requested D01 power was not reached within max_n.")


# -----------------------------------------------------------------------------
# Selector audit
# -----------------------------------------------------------------------------

def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="raise")


def normalized_lower(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def build_selector_membership(
    scenarios: pd.DataFrame,
    replicates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    table = scenarios.copy()

    for column in [
        "n_samples",
        "shared_signal_strength",
        "shared_factor_correlation",
        "true_biological_shared_factors",
        "true_technical_shared_factors",
        "n_shared_factors",
        "planned_replicates",
    ]:
        table[column] = numeric(table[column])

    d01 = (
        table["true_biological_shared_factors"].eq(0)
        & table["true_technical_shared_factors"].eq(0)
    )

    d02_current = (
        table["n_samples"].eq(62)
        & np.isclose(table["shared_signal_strength"], 1.0)
        & np.isclose(table["shared_factor_correlation"], 0.0)
        & normalized_lower(table["batch_mode"]).eq("none")
    )

    d02_standard_candidate = (
        d02_current
        & table["scenario_family"].astype(str).eq("final_independent_gate_curve")
    )

    independent_null = (
        d01
        & table["scenario_family"].astype(str).eq("final_independent_gate_curve")
    )
    correlated_private_null = (
        d01
        & table["scenario_family"].astype(str).eq("correlated_private_null")
    )

    replicate_counts = (
        replicates.groupby("scenario_id", sort=False)
        .size()
        .rename("replicate_rows")
    )

    output = table.merge(
        replicate_counts,
        how="left",
        left_on="scenario_id",
        right_index=True,
        validate="one_to_one",
    )
    output["replicate_rows"] = output["replicate_rows"].fillna(0).astype(int)
    output["D01_current_selector"] = d01.to_numpy()
    output["D01_independent_null_component"] = independent_null.to_numpy()
    output["D01_correlated_private_component"] = correlated_private_null.to_numpy()
    output["D02_current_selector"] = d02_current.to_numpy()
    output["D03_current_selector"] = d02_current.to_numpy()
    output["D02_D03_standard_candidate_selector"] = d02_standard_candidate.to_numpy()

    report_columns = [
        "scenario_id",
        "scenario_family",
        "calibration_role",
        "n_samples",
        "shared_signal_strength",
        "shared_factor_correlation",
        "batch_mode",
        "loading_regime",
        "shared_support_pattern",
        "true_biological_shared_factors",
        "true_technical_shared_factors",
        "n_shared_factors",
        "replicate_rows",
        "D01_current_selector",
        "D01_independent_null_component",
        "D01_correlated_private_component",
        "D02_current_selector",
        "D03_current_selector",
        "D02_D03_standard_candidate_selector",
    ]
    output = output[report_columns].copy()

    d01_rows = output[output["D01_current_selector"]]
    d01_independent = output[output["D01_independent_null_component"]]
    d01_correlated = output[output["D01_correlated_private_component"]]
    d02_rows = output[output["D02_current_selector"]]
    d02_standard = output[output["D02_D03_standard_candidate_selector"]]

    def total_dataset_rows(part: pd.DataFrame) -> int:
        return int(part["replicate_rows"].sum())

    def total_true_factor_opportunities(part: pd.DataFrame) -> int:
        return int((part["replicate_rows"] * part["true_biological_shared_factors"]).sum())

    summary = {
        "D01": {
            "scenario_ids": d01_rows["scenario_id"].astype(str).tolist(),
            "n_scenarios": int(d01_rows.shape[0]),
            "n_datasets": total_dataset_rows(d01_rows),
            "independent_null_scenario_ids": d01_independent["scenario_id"].astype(str).tolist(),
            "independent_null_datasets": total_dataset_rows(d01_independent),
            "correlated_private_scenario_ids": d01_correlated["scenario_id"].astype(str).tolist(),
            "correlated_private_datasets": total_dataset_rows(d01_correlated),
        },
        "D02_D03_current": {
            "scenario_ids": d02_rows["scenario_id"].astype(str).tolist(),
            "n_scenarios": int(d02_rows.shape[0]),
            "n_datasets": total_dataset_rows(d02_rows),
            "n_true_factor_opportunities": total_true_factor_opportunities(d02_rows),
        },
        "D02_D03_standard_candidate": {
            "scenario_ids": d02_standard["scenario_id"].astype(str).tolist(),
            "n_scenarios": int(d02_standard.shape[0]),
            "n_datasets": total_dataset_rows(d02_standard),
            "n_true_factor_opportunities": total_true_factor_opportunities(d02_standard),
        },
    }

    if summary["D01"]["n_datasets"] != 1000:
        raise RuntimeError(
            f"D01 current selector resolves to {summary['D01']['n_datasets']} datasets, expected 1000 from the frozen registry."
        )
    if summary["D01"]["independent_null_datasets"] != 800:
        raise RuntimeError("D01 independent-null component does not contain 800 datasets.")
    if summary["D01"]["correlated_private_datasets"] != 200:
        raise RuntimeError("D01 correlated-private component does not contain 200 datasets.")
    if summary["D02_D03_current"]["n_scenarios"] != 4:
        raise RuntimeError(
            "D02/D03 current selector no longer resolves to the expected four frozen scenarios."
        )
    if summary["D02_D03_standard_candidate"]["n_scenarios"] != 1:
        raise RuntimeError(
            "The exact STANDARD independent n=62 signal=1 candidate selector does not resolve to exactly one scenario."
        )

    return output, summary


def audit_metric_and_decision_semantics(
    metrics: pd.DataFrame,
    decisions: pd.DataFrame,
) -> dict[str, Any]:
    def metric_row(metric_id: str) -> pd.Series:
        part = metrics[metrics["metric_id"].astype(str).eq(metric_id)]
        if part.shape[0] != 1:
            raise RuntimeError(f"Expected one metric row for {metric_id}.")
        return part.iloc[0]

    def decision_row(decision_id: str) -> pd.Series:
        part = decisions[decisions["decision_id"].astype(str).eq(decision_id)]
        if part.shape[0] != 1:
            raise RuntimeError(f"Expected one decision row for {decision_id}.")
        return part.iloc[0]

    tpr = metric_row("gate_true_positive_rate")
    fnr = metric_row("gate_false_negative_rate")
    fwer = metric_row("gate_familywise_false_acceptance")

    d01 = decision_row("D01_NULL_SPECIFICITY")
    d02 = decision_row("D02_STANDARD_SENSITIVITY")
    d03 = decision_row("D03_STANDARD_FNR")

    fnr_formula = clean_text(fnr["formula"]).replace(" ", "").lower()
    d03_is_mirror = fnr_formula in {
        "1-gate_true_positive_rate",
        "1.0-gate_true_positive_rate",
    } and clean_text(d02["scenario_selector"]) == clean_text(d03["scenario_selector"])

    fwer_formula = clean_text(fwer["formula"]).lower()
    familywise_event_explicit = (
        "at least one false biological" in fwer_formula
        and "null dataset" in fwer_formula
    )

    d01_ci = clean_text(fwer["confidence_interval"])
    d01_ci_underspecified = (
        "wilson" in d01_ci.lower()
        and "95%" in d01_ci.lower()
        and all(token not in d01_ci.lower() for token in ["two-sided", "one-sided", "continuity"])
    )

    tpr_formula = clean_text(tpr["formula"])
    tpr_denominator_factor_based = (
        "number of true shared factors" in tpr_formula.lower()
    )
    tpr_aggregation_unit_explicit = any(
        token in tpr_formula.lower()
        for token in [
            "pooled across datasets",
            "mean across datasets",
            "dataset-level",
            "per dataset",
        ]
    )

    return {
        "D01": {
            "selector": clean_text(d01["scenario_selector"]),
            "metric": clean_text(d01["metric"]),
            "required_threshold": clean_text(d01["required_threshold"]),
            "metric_formula": clean_text(fwer["formula"]),
            "confidence_interval_text": d01_ci,
            "familywise_event_definition_explicit": familywise_event_explicit,
            "wilson_sidedness_and_continuity_correction_explicit": not d01_ci_underspecified,
            "implementation_ambiguity_present": d01_ci_underspecified,
        },
        "D02": {
            "selector": clean_text(d02["scenario_selector"]),
            "metric": clean_text(d02["metric"]),
            "required_threshold": clean_text(d02["required_threshold"]),
            "metric_formula": tpr_formula,
            "confidence_interval_text": clean_text(tpr["confidence_interval"]),
            "formula_is_factor_based": tpr_denominator_factor_based,
            "aggregation_unit_explicit": tpr_aggregation_unit_explicit,
            "aggregation_unit_ambiguity_present": (
                tpr_denominator_factor_based and not tpr_aggregation_unit_explicit
            ),
        },
        "D03": {
            "selector": clean_text(d03["scenario_selector"]),
            "metric": clean_text(d03["metric"]),
            "required_threshold": clean_text(d03["required_threshold"]),
            "metric_formula": clean_text(fnr["formula"]),
            "confidence_interval_text": clean_text(fnr["confidence_interval"]),
            "algebraic_mirror_of_D02_on_same_selector": d03_is_mirror,
        },
    }


# -----------------------------------------------------------------------------
# Derived design tables
# -----------------------------------------------------------------------------

def build_wilson_threshold_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    definitions = [
        ("two_sided_95", TWO_SIDED_95_Z),
        ("one_sided_95", ONE_SIDED_95_Z),
    ]
    for label, z in definitions:
        for total in D01_N_GRID:
            maximum = max_d01_pass_count(total, z)
            _, upper = wilson_interval(maximum, total, z)
            next_upper = None
            if maximum + 1 <= total:
                _, next_upper = wilson_interval(maximum + 1, total, z)

            rows.append(
                {
                    "interval_interpretation": label,
                    "z": z,
                    "continuity_correction": False,
                    "n": total,
                    "D01_limit": D01_LIMIT,
                    "max_passing_false_acceptances": maximum,
                    "max_passing_observed_rate": maximum / total,
                    "wilson_upper_at_max_pass": upper,
                    "wilson_upper_at_next_count": next_upper,
                }
            )
    return pd.DataFrame(rows)


def build_d01_power_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for total in D01_N_GRID:
        maximum = max_d01_pass_count(total, TWO_SIDED_95_Z)
        for true_rate in D01_TRUE_RATE_GRID:
            rows.append(
                {
                    "wilson_interpretation_for_design_audit": "two-sided 95%, z=1.959963984540054, no continuity correction",
                    "n": total,
                    "D01_limit": D01_LIMIT,
                    "max_passing_false_acceptances": maximum,
                    "max_passing_observed_rate": maximum / total,
                    "true_false_acceptance_rate": true_rate,
                    "probability_D01_pass": exact_probability_d01_pass(
                        total,
                        true_rate,
                        TWO_SIDED_95_Z,
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_d02_reference_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for denominator in D02_DENOMINATOR_REFERENCE_GRID:
        minimum_successes = int(math.ceil(D02_TPR_MIN * denominator - 1e-12))
        for true_tpr in D02_TRUE_TPR_GRID:
            rows.append(
                {
                    "reference_only": True,
                    "binomial_denominator": denominator,
                    "D02_point_estimate_threshold": D02_TPR_MIN,
                    "minimum_successes_to_pass": minimum_successes,
                    "true_TPR": true_tpr,
                    "probability_point_estimate_D02_pass": exact_probability_point_estimate_tpr_pass(
                        denominator,
                        true_tpr,
                    ),
                    "interpretation": (
                        "N=100 corresponds to one Bernoulli event per dataset; "
                        "N=400 corresponds to four true-factor opportunities across 100 standard datasets. "
                        "The frozen metric formula is factor-based but the cross-dataset aggregation unit is not explicit."
                    ),
                }
            )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Audit synthesis
# -----------------------------------------------------------------------------

def build_amendment_requirements(
    source_05a: dict[str, Any],
    selector_summary: dict[str, Any],
    semantics: dict[str, Any],
    power_80_at_true_08: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "DESIGN_AUDIT_RECOMMENDATIONS_NOT_YET_FROZEN",
        "must_be_frozen_before_final_gate_materialization": True,
        "recommended_05a2_scope": [
            {
                "item": "pairing_permutation_resolution",
                "current": int(source_05a["pairing_permutations"]),
                "recommended": 1000,
                "reason": (
                    "Increase Monte Carlo p-value resolution only. "
                    "Do not change the pairing null, primary metrics, multiplicity family, model registry, or outcome blinding."
                ),
            },
            {
                "item": "D01_primary_aggregation",
                "current": "selector is frozen, aggregation across matching scenarios is not explicit",
                "recommended": (
                    "pool every dataset satisfying the frozen D01 selector for the primary decision; "
                    "describe the estimand as familywise false biological acceptance over the prespecified final-gate null registry"
                ),
                "frozen_registry_counts": selector_summary["D01"],
            },
            {
                "item": "D01_Wilson_implementation",
                "current": semantics["D01"]["confidence_interval_text"],
                "recommended": {
                    "method": "Wilson score interval",
                    "confidence_level": 0.95,
                    "sidedness": "two-sided",
                    "z": TWO_SIDED_95_Z,
                    "continuity_correction": False,
                    "decision_quantity": "upper endpoint",
                    "pass_rule": "upper endpoint <= 0.10",
                    "equivalent_statement": "upper endpoint of a two-sided 95% interval, equivalently a one-sided 97.5% upper bound",
                },
            },
            {
                "item": "D01_mandatory_reporting",
                "recommended": [
                    "point estimate, numerator/denominator, two-sided 95% Wilson CI, then PASS/FAIL",
                    "primary pooled prespecified D01 null registry (N=1000 datasets)",
                    "ordinary independent nulls pooled (N=800 datasets)",
                    "correlated-private stress null alone (N=200 datasets)",
                    "independent null n=50,62,75,100 separately (N=200 each)",
                    "descriptive logistic trend across the four independent-null n values only; raw n scale, OR per +25 samples, no gate consequence",
                ],
            },
            {
                "item": "D02_D03_standard_selector",
                "current_matching": selector_summary["D02_D03_current"],
                "recommended_matching": selector_summary["D02_D03_standard_candidate"],
                "recommended_selector": (
                    "scenario_family == 'final_independent_gate_curve' and "
                    "n_samples == 62 and shared_signal_strength == 1.0 and "
                    "shared_factor_correlation == 0 and batch_mode == 'none'"
                ),
                "rationale_disclosure": (
                    "The clarification probably makes D02/D03 easier by excluding harder mixed-support and loading-density stress scenarios, "
                    "but it also reduces the selected dataset count. The clarification is justified by the prospectively frozen STANDARD decision names "
                    "and standard_n62_signal1 operating-target names, and is made while final-gate datasets do not yet exist."
                ),
            },
            {
                "item": "D02_D03_aggregation_unit",
                "current": {
                    "metric_formula": semantics["D02"]["metric_formula"],
                    "aggregation_unit_explicit": semantics["D02"]["aggregation_unit_explicit"],
                    "current_selector_dataset_count": selector_summary["D02_D03_current"]["n_datasets"],
                    "current_selector_true_factor_opportunities": selector_summary["D02_D03_current"]["n_true_factor_opportunities"],
                    "standard_selector_dataset_count": selector_summary["D02_D03_standard_candidate"]["n_datasets"],
                    "standard_selector_true_factor_opportunities": selector_summary["D02_D03_standard_candidate"]["n_true_factor_opportunities"],
                },
                "recommended": (
                    "05a2 must state the exact numerator/denominator and cross-dataset aggregation unit for TPR/FNR. "
                    "The frozen formula is factor-based, but the current registry does not state whether factor events are pooled across datasets "
                    "or dataset-level TPRs are averaged. Do not leave this choice to the post-result implementation."
                ),
            },
            {
                "item": "D03_mirror_status",
                "current": semantics["D03"],
                "recommended": (
                    "Preserve D03 because it is frozen, but state explicitly whether it is the algebraic mirror of D02 rather than an independent sensitivity criterion."
                ),
            },
            {
                "item": "specificity_sensitivity_asymmetry",
                "recommended": (
                    "State prospectively that D01 uses an uncertainty-aware upper Wilson bound, whereas D02/D03 use point-estimate thresholds, "
                    "reflecting the preregistered priority on control of false biological acceptance. Do not change D02/D03 to CI-bound rules."
                ),
            },
            {
                "item": "prewritten_failure_consequence_policy",
                "recommended": (
                    "Freeze scientific consequences for D01/D02/D03 PASS/FAIL before final-gate results. "
                    "A failed criterion changes the conclusion, not the criterion; no rescue retuning or outcome-guided rescue is permitted."
                ),
            },
            {
                "item": "runtime_pilot_numerical_versioning",
                "recommended": (
                    "Pilot shards may become final shards only if the numerical implementation hash is unchanged. "
                    "Any change to numerical/GPU code path invalidates and recomputes all pilot shards; logging-only changes may retain them only when the numerical hash is unchanged."
                ),
            },
        ],
        "design_resolution_reference": {
            "minimum_n_for_80_percent_probability_of_D01_pass_when_true_rate_is_0.08": power_80_at_true_08,
            "note": (
                "This reference is descriptive design calibration only. "
                "The audit does not recommend increasing the frozen 3,100-dataset bank at this time."
            ),
        },
        "forbidden_changes_from_this_audit": [
            "No final-gate threshold is changed.",
            "No final-gate dataset count is changed.",
            "No stability-resample count is changed.",
            "No architecture/profile is changed.",
            "No TARGET result is accessed.",
            "No biological or outcome result is accessed.",
        ],
    }


def build_readme(
    selector_summary: dict[str, Any],
    semantics: dict[str, Any],
    power_80: dict[str, Any],
    source_05a: dict[str, Any],
) -> str:
    d02_current = ", ".join(selector_summary["D02_D03_current"]["scenario_ids"])
    d02_standard = ", ".join(selector_summary["D02_D03_standard_candidate"]["scenario_ids"])

    lines = [
        "Paper 5 - 05d4b read-only final-gate decision-design audit",
        "=" * 78,
        "",
        f"Script version: {SCRIPT_VERSION}",
        f"Audit version: {AUDIT_VERSION}",
        f"Created UTC: {utc_now_iso()}",
        "",
        "Scientific state",
        "----------------",
        "GPU computation: NO",
        "Model fitting: NO",
        "Final-gate simulation data accessed: NO",
        "TARGET accessed: NO",
        "Outcomes accessed: NO",
        "Threshold retuning: NO",
        "",
        "Purpose",
        "-------",
        "This audit reads only frozen 05a/05a1/05b-v2 registries and contracts.",
        "It evaluates whether the already-frozen final-gate decision rules are operationally unambiguous and how much resolution they have before any final-gate simulation dataset is materialized.",
        "",
        "D01",
        "----",
        f"Current frozen selector resolves to {selector_summary['D01']['n_scenarios']} scenarios / {selector_summary['D01']['n_datasets']} datasets.",
        f"Independent null component: {selector_summary['D01']['independent_null_datasets']} datasets.",
        f"Correlated-private stress component: {selector_summary['D01']['correlated_private_datasets']} datasets.",
        f"Frozen event formula: {semantics['D01']['metric_formula']}",
        f"Frozen CI text: {semantics['D01']['confidence_interval_text']}",
        "The CI text does not specify sidedness, z, or continuity correction; this is an implementation ambiguity that must be frozen before final-gate materialization.",
        "The design audit therefore reports both common interpretations and uses two-sided 95% Wilson only for prospective power/reference calculations, not as a rewritten frozen rule.",
        "",
        "D02 / D03",
        "-----------",
        f"Current selector matches: {d02_current}",
        f"Exact STANDARD independent candidate matches: {d02_standard}",
        f"Current matching datasets: {selector_summary['D02_D03_current']['n_datasets']}",
        f"Exact STANDARD matching datasets: {selector_summary['D02_D03_standard_candidate']['n_datasets']}",
        f"Current matching true-factor opportunities: {selector_summary['D02_D03_current']['n_true_factor_opportunities']}",
        f"Exact STANDARD true-factor opportunities: {selector_summary['D02_D03_standard_candidate']['n_true_factor_opportunities']}",
        f"D03 is an algebraic mirror of D02 on the same selector: {semantics['D03']['algebraic_mirror_of_D02_on_same_selector']}",
        f"TPR formula is factor-based: {semantics['D02']['formula_is_factor_based']}",
        f"Cross-dataset aggregation unit explicit in frozen formula: {semantics['D02']['aggregation_unit_explicit']}",
        "The exact D02/D03 aggregation denominator must therefore be frozen in 05a2 before results.",
        "",
        "Pairing permutation resolution",
        "------------------------------",
        f"Frozen RC01 pairing permutations: {source_05a['pairing_permutations']}",
        f"Frozen multiplicity adjustment: {source_05a['pairing_multiplicity']}",
        "A prospective 100 -> 1,000 resolution amendment is recommended without changing the null construction or multiplicity family.",
        "",
        "Reference design calculation",
        "----------------------------",
        "Under a two-sided 95% Wilson score interval with no continuity correction:",
        "N=200 permits at most 11 false-acceptance datasets while keeping the upper CI <= 0.10.",
        "N=1000 permits at most 81 false-acceptance datasets while keeping the upper CI <= 0.10.",
        (
            "Minimum N for at least 80% probability of passing D01 when the true rate is exactly 8% "
            f"is {power_80['n']} under this reference implementation."
        ),
        "This is a design-resolution diagnostic, not a recommendation to alter the frozen 3,100 datasets.",
        "",
        "Next step",
        "---------",
        "Use this audit plus 05d4a to write a narrow 05a2 inference/final-gate implementation clarification amendment before any final-gate data are materialized.",
        "",
    ]
    return "\n".join(lines)


def verify_existing_lock() -> None:
    lock = load_json(OUTPUT_LOCK)
    if clean_text(lock.get("status")) != "DERIVED_READ_ONLY_FINAL_GATE_DESIGN_AUDIT":
        raise RuntimeError("Unexpected existing 05d4b lock status.")

    failures: list[str] = []
    for relative_name, record in lock.get("generated_files", {}).items():
        path = resolve_project_path(relative_name)
        if not path.is_file():
            failures.append(f"missing: {relative_name}")
            continue
        if int(path.stat().st_size) != int(record.get("size_bytes", -1)):
            failures.append(f"size mismatch: {relative_name}")
            continue
        if sha256_file(path) != clean_text(record.get("sha256")):
            failures.append(f"hash mismatch: {relative_name}")

    if failures:
        raise RuntimeError("Existing 05d4b verification failed:\n- " + "\n- ".join(failures))

    print("=" * 88)
    print("Existing Paper 5 05d4b final-gate design audit: PASS")
    print("=" * 88)
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("No recomputation performed.")
    print("=" * 88)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    print("=" * 88)
    print("Paper 5 - read-only final-gate CI / power / decision-rule audit")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("GPU computation in 05d4b: NO")
    print("Model fitting in 05d4b: NO")
    print("TARGET/outcomes accessed: NO")
    print("Final-gate simulation data accessed: NO")
    print("Threshold retuning in 05d4b: FORBIDDEN")
    print()

    if OUTPUT_LOCK.exists():
        verify_existing_lock()
        return

    source_05a = verify_05a()
    source_05a1 = verify_05a1()
    source_05b = verify_05b_v2_no_final_gate_materialization()

    if SOURCE_05D4A_LOCK.is_file():
        source_05d4a = {
            "path": relative_path(SOURCE_05D4A_LOCK),
            "sha256": sha256_file(SOURCE_05D4A_LOCK),
            "present": True,
        }
    else:
        source_05d4a = {
            "path": relative_path(SOURCE_05D4A_LOCK),
            "sha256": None,
            "present": False,
        }

    print("05a AI method contract + RC01/MF01: PASS")
    print("05a1 final-gate scenario/replicate/metric/decision registries: PASS")
    print("05b v2 architecture-only materialization lock: PASS")
    print("Final gate-calibration datasets materialized: NO")
    print(f"05d4a paired ablation audit present: {'YES' if source_05d4a['present'] else 'NO'}")
    print()

    selector_membership, selector_summary = build_selector_membership(
        source_05a1["scenarios"],
        source_05a1["replicates"],
    )
    semantics = audit_metric_and_decision_semantics(
        source_05a1["metrics"],
        source_05a1["decisions"],
    )

    wilson_thresholds = build_wilson_threshold_table()
    d01_power = build_d01_power_table()
    d02_reference = build_d02_reference_table()
    power_80 = minimum_n_for_d01_power(
        0.08,
        0.80,
        z=TWO_SIDED_95_Z,
    )

    amendment_requirements = build_amendment_requirements(
        source_05a,
        selector_summary,
        semantics,
        power_80,
    )

    audit = {
        "audit_version": AUDIT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": "DERIVED_READ_ONLY_FINAL_GATE_DESIGN_AUDIT",
        "scientific_state": {
            "GPU_computation_performed": False,
            "model_fitting_performed": False,
            "TARGET_accessed": False,
            "outcomes_accessed": False,
            "biological_annotations_accessed": False,
            "final_gate_simulation_data_accessed": False,
            "final_gate_simulation_data_materialized": False,
            "threshold_retuning_performed": False,
            "decision_rule_rewritten": False,
        },
        "source_provenance": {
            "05a_contract": {
                "path": relative_path(CONTRACT_05A),
                "sha256": source_05a["contract_sha256"],
            },
            "05a_lock": {
                "path": relative_path(LOCK_05A),
                "sha256": source_05a["lock_sha256"],
            },
            "05a1_addendum": {
                "path": relative_path(CONTRACT_05A1),
                "sha256": source_05a1["addendum_sha256"],
            },
            "05a1_lock": {
                "path": relative_path(LOCK_05A1),
                "sha256": source_05a1["lock_sha256"],
            },
            "final_gate_scenarios": {
                "path": relative_path(FINAL_GATE_SCENARIOS),
                "sha256": source_05a1["scenario_sha256"],
            },
            "final_gate_replicates": {
                "path": relative_path(FINAL_GATE_REPLICATES),
                "sha256": source_05a1["replicate_sha256"],
            },
            "gate_metrics": {
                "path": relative_path(GATE_METRICS),
                "sha256": source_05a1["metric_sha256"],
            },
            "gate_decisions": {
                "path": relative_path(GATE_DECISIONS),
                "sha256": source_05a1["decision_sha256"],
            },
            "05b_v2_lock": {
                "path": relative_path(LOCK_05B_V2),
                "sha256": source_05b["lock_sha256"],
            },
            "05d4a_lock_if_present": source_05d4a,
        },
        "frozen_scope": {
            "final_gate_scenarios": EXPECTED_FINAL_GATE_SCENARIOS,
            "final_gate_datasets": EXPECTED_FINAL_GATE_DATASETS,
            "stability_resamples_per_dataset": EXPECTED_FINAL_GATE_STABILITY_RESAMPLES,
            "pairing_permutations_current": source_05a["pairing_permutations"],
        },
        "selector_resolution": selector_summary,
        "metric_and_decision_semantics": semantics,
        "reference_statistical_implementation": {
            "role": (
                "design audit only; becomes primary only if separately frozen in 05a2 before final-gate materialization"
            ),
            "D01_two_sided_95_Wilson_z": TWO_SIDED_95_Z,
            "D01_one_sided_95_Wilson_z_for_ambiguity_comparison": ONE_SIDED_95_Z,
            "continuity_correction": False,
            "D01_limit": D01_LIMIT,
            "D02_point_estimate_min": D02_TPR_MIN,
            "D03_point_estimate_max": D03_FNR_MAX,
            "minimum_n_for_80pct_D01_pass_probability_at_true_rate_0p08": power_80,
        },
        "newly_detected_ambiguities": [
            {
                "id": "A01_D01_WILSON_IMPLEMENTATION",
                "present": bool(semantics["D01"]["implementation_ambiguity_present"]),
                "detail": (
                    "Frozen text says Wilson 95% binomial interval but does not specify sidedness, z, or continuity correction."
                ),
            },
            {
                "id": "A02_D02_D03_SELECTOR_SCOPE",
                "present": selector_summary["D02_D03_current"]["n_scenarios"] > 1,
                "detail": (
                    "The frozen STANDARD selector also matches mixed-support, sparse-loading, and dense-loading stress scenarios."
                ),
            },
            {
                "id": "A03_D02_D03_AGGREGATION_UNIT",
                "present": bool(semantics["D02"]["aggregation_unit_ambiguity_present"]),
                "detail": (
                    "TPR/FNR formulas are factor-based, but the frozen registry does not specify whether true-factor events are pooled across datasets or dataset-level rates are averaged."
                ),
            },
            {
                "id": "A04_D03_MIRRORS_D02",
                "present": bool(semantics["D03"]["algebraic_mirror_of_D02_on_same_selector"]),
                "detail": (
                    "D03 is algebraically 1-TPR on the same selector and should not be presented as independent evidence if this remains true in the implementation."
                ),
            },
        ],
        "guardrails": [
            "This audit does not materialize or read any final-gate simulation dataset.",
            "This audit does not choose a favorable implementation after results.",
            "The two-sided Wilson calculation is a prospective reference calculation until separately frozen by 05a2.",
            "The audit does not change 3,100 datasets, 100 stability resamples, 0.70/0.40/0.80 factor thresholds, architecture, or profile.",
            "Any 05a2 amendment must be frozen before final-gate materialization.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(selector_membership, OUTPUT_SELECTOR_MEMBERSHIP)
    write_csv_atomic(wilson_thresholds, OUTPUT_D01_WILSON_THRESHOLDS)
    write_csv_atomic(d01_power, OUTPUT_D01_POWER)
    write_csv_atomic(d02_reference, OUTPUT_D02_REFERENCE)
    write_json_atomic(audit, OUTPUT_AUDIT)
    write_json_atomic(amendment_requirements, OUTPUT_AMENDMENT_REQUIREMENTS)

    readme = build_readme(
        selector_summary,
        semantics,
        power_80,
        source_05a,
    )
    OUTPUT_README.write_text(readme, encoding="utf-8")

    generated = [
        OUTPUT_SELECTOR_MEMBERSHIP,
        OUTPUT_D01_WILSON_THRESHOLDS,
        OUTPUT_D01_POWER,
        OUTPUT_D02_REFERENCE,
        OUTPUT_AUDIT,
        OUTPUT_AMENDMENT_REQUIREMENTS,
        OUTPUT_README,
    ]

    lock = {
        "audit_version": AUDIT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": "DERIVED_READ_ONLY_FINAL_GATE_DESIGN_AUDIT",
        "scientific_state": audit["scientific_state"],
        "source_provenance": audit["source_provenance"],
        "generated_files": {
            relative_path(path): file_record(path)
            for path in generated
        },
        "rewrite_policy": (
            "05d4b is an immutable read-only design audit. "
            "It does not rewrite 05a/05a1 and does not itself freeze the recommended 05a2 clarifications. "
            "Any change requires a new versioned audit or the separately versioned 05a2 amendment."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    d01_two = wilson_thresholds[
        wilson_thresholds["interval_interpretation"].eq("two_sided_95")
    ].set_index("n")
    d01_one = wilson_thresholds[
        wilson_thresholds["interval_interpretation"].eq("one_sided_95")
    ].set_index("n")

    print("-" * 88)
    print("Final-gate selector audit")
    print("-" * 88)
    print(
        f"D01 current selector: {selector_summary['D01']['n_scenarios']} scenarios / "
        f"{selector_summary['D01']['n_datasets']} datasets"
    )
    print(
        f"  independent nulls: {selector_summary['D01']['independent_null_datasets']} datasets"
    )
    print(
        f"  correlated-private stress null: {selector_summary['D01']['correlated_private_datasets']} datasets"
    )
    print(
        f"D02/D03 current selector: {selector_summary['D02_D03_current']['n_scenarios']} scenarios / "
        f"{selector_summary['D02_D03_current']['n_datasets']} datasets"
    )
    print("  current matches: " + ", ".join(selector_summary["D02_D03_current"]["scenario_ids"]))
    print(
        f"D02/D03 exact STANDARD candidate: {selector_summary['D02_D03_standard_candidate']['n_scenarios']} scenario / "
        f"{selector_summary['D02_D03_standard_candidate']['n_datasets']} datasets"
    )
    print("  standard match: " + ", ".join(selector_summary["D02_D03_standard_candidate"]["scenario_ids"]))
    print(
        "D02/D03 true-factor opportunities: "
        f"current={selector_summary['D02_D03_current']['n_true_factor_opportunities']}, "
        f"standard={selector_summary['D02_D03_standard_candidate']['n_true_factor_opportunities']}"
    )
    print(
        "D02/D03 aggregation unit explicitly frozen: "
        + ("YES" if semantics["D02"]["aggregation_unit_explicit"] else "NO")
    )
    print(
        "D03 algebraic mirror of D02 on same selector: "
        + ("YES" if semantics["D03"]["algebraic_mirror_of_D02_on_same_selector"] else "NO")
    )
    print()

    print("-" * 88)
    print("D01 Wilson implementation ambiguity")
    print("-" * 88)
    for total in (200, 1000):
        print(
            f"N={total}: two-sided 95% max PASS={int(d01_two.loc[total, 'max_passing_false_acceptances'])}; "
            f"one-sided 95% max PASS={int(d01_one.loc[total, 'max_passing_false_acceptances'])}"
        )
    print(
        "Reference two-sided 95% Wilson: minimum N for >=80% probability of D01 PASS "
        f"when true false-acceptance rate=8%: {power_80['n']}"
    )
    print()

    print("-" * 88)
    print("05a2 requirements identified by audit")
    print("-" * 88)
    print("1. Freeze exact D01 pooling and Wilson implementation.")
    print("2. Freeze exact D02/D03 STANDARD selector.")
    print("3. Freeze exact D02/D03 aggregation numerator/denominator.")
    print("4. Preserve D03 as mirrored if FNR remains exactly 1-TPR.")
    print("5. Increase RC01 pairing permutations 100 -> 1,000 for Monte Carlo resolution only.")
    print("6. Freeze mandatory subgroup reporting and prewritten failure consequences.")
    print()

    print("=" * 88)
    print("Paper 5 05d4b final-gate design audit: PASS")
    print("=" * 88)
    print(f"Audit: {relative_path(OUTPUT_AUDIT)}")
    print(f"Selector membership: {relative_path(OUTPUT_SELECTOR_MEMBERSHIP)}")
    print(f"D01 Wilson thresholds: {relative_path(OUTPUT_D01_WILSON_THRESHOLDS)}")
    print(f"D01 power curve: {relative_path(OUTPUT_D01_POWER)}")
    print(f"D02 reference: {relative_path(OUTPUT_D02_REFERENCE)}")
    print(f"05a2 requirements: {relative_path(OUTPUT_AMENDMENT_REQUIREMENTS)}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("05d5 may continue running independently: YES")
    print("Final-gate materialization remains FORBIDDEN until 05a2 is frozen.")
    print("=" * 88)


if __name__ == "__main__":
    main()
