from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05f3b-aggregate-final-gate-results-cpu-v1-poststart-preunblinding-locked-no-cli"
CONTRACT_VERSION = "paper5-05f3b-final-gate-scientific-aggregation-v1"

EXPECTED_05F2_SCRIPT_VERSION = (
    "05f2-run-full-final-gate-gpu-resumable-v1-sequential-05f1-path-no-cli"
)
EXPECTED_05F2_LOCK_STATUS = (
    "FROZEN_ALL_3100_FINAL_GATE_SHARDS_COMPLETE_BEFORE_SCIENTIFIC_AGGREGATION"
)
EXPECTED_05F2A_LOCK_STATUS = (
    "FROZEN_FINAL_GATE_AGGREGATION_IMPLEMENTATION_POSTSTART_PREUNBLINDING"
)
EXPECTED_05A2_STATUS = "FROZEN_BEFORE_FINAL_GATE_MATERIALIZATION"

EXPECTED_DATASETS = 3_100
EXPECTED_SCENARIOS = 24
EXPECTED_STABILITY_RESAMPLES = 100

D01_LIMIT = 0.10
D01_WILSON_Z = 1.959963984540054
D02_TPR_MIN = 0.80
D03_FNR_MAX = 0.20
D04_SUBSPACE_MIN = 0.80
D05_VIEW_SUPPORT_MIN = 0.80
D06_BATCH_FALSE_ACCEPT_MAX = 0.10

EXPECTED_D01_DATASETS = 1_000
EXPECTED_D01_INDEPENDENT_DATASETS = 800
EXPECTED_D01_CORRELATED_PRIVATE_DATASETS = 200
EXPECTED_D01_PER_N_DATASETS = 200
EXPECTED_D02_DATASETS = 100
EXPECTED_D02_FACTOR_OPPORTUNITIES = 400
EXPECTED_D04_DATASETS = 100
EXPECTED_D05_DATASETS = 100
EXPECTED_D06_DATASETS = 400
EXPECTED_D06_TECHNICAL_OPPORTUNITIES = 400

STANDARD_SCENARIO_ID = "G07_IND_N62_S1P00"
ROTATIONAL_SCENARIO_ID = "G17_ROTATIONAL_N62_S1P00"
MIXED_SUPPORT_SCENARIO_ID = "G18_MIXED_SUPPORT_N62_S1P00"
CORRELATED_PRIVATE_SCENARIO_ID = "G19_CORRELATED_PRIVATE_NULL_N62"
BATCH_NULL_SCENARIO_IDS = (
    "G20_BATCH_SHARED_NULL_N62",
    "G21_BATCH_PARTIAL_NULL_N62",
)

# Fixed in the pre-existing 05f3 v1 source and re-locked by 05f2b before scientific unblinding.
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_MASTER_SEED = 20260822

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

DIR_05A1 = PAPER5_ROOT / "method_contract" / "05a1"
FINAL_GATE_SCENARIOS = DIR_05A1 / "final_gate_calibration_scenarios.csv"
FINAL_GATE_REPLICATES = DIR_05A1 / "final_gate_calibration_replicates.csv"

DIR_05A2 = PAPER5_ROOT / "method_contract" / "05a2"
AMENDMENT_05A2 = DIR_05A2 / "inference_final_gate_implementation_amendment.json"
EFFECTIVE_METRICS_05A2 = DIR_05A2 / "effective_stability_gate_operating_metric_registry.csv"
EFFECTIVE_DECISIONS_05A2 = DIR_05A2 / "effective_stability_gate_decision_registry.csv"
REPORTING_05A2 = DIR_05A2 / "final_gate_reporting_registry.csv"
FAILURE_POLICY_05A2 = DIR_05A2 / "final_gate_failure_consequence_policy.json"
LOCK_05A2 = DIR_05A2 / "inference_final_gate_implementation_amendment_lock.json"

DIR_05F2 = PAPER5_ROOT / "method_contract" / "05f2"
SCRIPT_05F2 = PAPER5_ROOT / "scripts" / "05f2_run_full_final_gate_gpu_resumable.py"
PROGRESS_05F2 = DIR_05F2 / "full_run_progress_manifest.csv"
SUMMARY_05F2 = DIR_05F2 / "full_run_completion_summary.json"
LOCK_05F2 = DIR_05F2 / "full_run_completion_lock.json"

DIR_05F2A = PAPER5_ROOT / "method_contract" / "05f2b"
LOCK_05F2A = DIR_05F2A / "final_gate_aggregation_preunblinding_lock.json"

PREEXISTING_05F3 = PAPER5_ROOT / "scripts" / "05f3_aggregate_final_gate_results_cpu.py"
EXPECTED_PREEXISTING_05F3_SHA256 = "443b553e2e9e95a9f4d3ac7de5fe7b4f33479d6552264555e3edc276b077d5c2"

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05f3b"
OUTPUT_PRIMARY_DECISIONS = OUTPUT_DIR / "final_gate_primary_decisions.csv"
OUTPUT_MANDATORY_REPORTING = OUTPUT_DIR / "final_gate_mandatory_reporting.csv"
OUTPUT_SCENARIO_SUMMARY = OUTPUT_DIR / "final_gate_scenario_summary.csv"
OUTPUT_OPERATING_METRICS = OUTPUT_DIR / "final_gate_operating_metric_summary.csv"
OUTPUT_D01_TREND = OUTPUT_DIR / "d01_independent_null_sample_size_trend.json"
OUTPUT_RESULT = OUTPUT_DIR / "final_gate_result.json"
OUTPUT_FAILURE_CONSEQUENCE = OUTPUT_DIR / "final_gate_failure_consequence.json"
OUTPUT_README = OUTPUT_DIR / "05f3b_README.txt"
OUTPUT_LOCK = OUTPUT_DIR / "final_gate_results_lock.json"


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


def nullable_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    text = clean_text(value)
    if not text:
        return None
    return int(float(text))


def nullable_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256()
    digest.update(str(BOOTSTRAP_MASTER_SEED).encode("utf-8"))
    for part in parts:
        digest.update(b"|")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big", signed=False) % (2**32 - 1)


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


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(
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
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_csv_atomic(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".csv", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        table.to_csv(
            tmp,
            index=False,
            float_format="%.12g",
            lineterminator="\n",
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


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def resolve_project_path(value: Any) -> Path:
    path = Path(clean_text(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


# -----------------------------------------------------------------------------
# Statistical helpers
# -----------------------------------------------------------------------------

def wilson_interval(
    numerator: int,
    denominator: int,
    z: float = D01_WILSON_Z,
) -> tuple[float, float]:
    if denominator <= 0:
        raise ValueError("Wilson denominator must be positive.")
    if numerator < 0 or numerator > denominator:
        raise ValueError("Wilson numerator outside denominator.")
    n = float(denominator)
    phat = float(numerator) / n
    z2 = float(z) ** 2
    center = (phat + z2 / (2.0 * n)) / (1.0 + z2 / n)
    half = (
        z
        * math.sqrt(
            phat * (1.0 - phat) / n
            + z2 / (4.0 * n * n)
        )
        / (1.0 + z2 / n)
    )
    return max(0.0, center - half), min(1.0, center + half)


def percentile_bootstrap_mean(
    values: np.ndarray,
    label: str,
) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise RuntimeError(f"Bootstrap metric {label} has fewer than two finite observations.")
    rng = np.random.default_rng(stable_seed("bootstrap", label))
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    n = int(x.size)
    # Chunked to avoid a large B x N allocation for future-proofing.
    chunk = 500
    filled = 0
    while filled < BOOTSTRAP_REPLICATES:
        b = min(chunk, BOOTSTRAP_REPLICATES - filled)
        indices = rng.integers(0, n, size=(b, n), endpoint=False)
        draws[filled : filled + b] = np.mean(x[indices], axis=1)
        filled += b
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def fit_logistic_sample_size_trend(
    n_samples: np.ndarray,
    events: np.ndarray,
) -> dict[str, Any]:
    n = np.asarray(n_samples, dtype=np.float64)
    y = np.asarray(events, dtype=np.float64)
    if n.shape != y.shape or n.ndim != 1:
        raise RuntimeError("D01 trend inputs have incompatible shapes.")
    if n.size != EXPECTED_D01_INDEPENDENT_DATASETS:
        raise RuntimeError("D01 trend must use exactly 800 independent-null datasets.")
    if not set(np.unique(n).astype(int).tolist()).issubset({50, 62, 75, 100}):
        raise RuntimeError("Unexpected n value in D01 independent-null trend.")
    if not np.isin(y, [0.0, 1.0]).all():
        raise RuntimeError("D01 trend event must be binary.")

    # Frozen predictor: raw sample size divided by 25, so exp(beta) is the OR
    # per +25 samples. Newton/IRLS MLE, unpenalized.
    x = n / 25.0
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2, dtype=np.float64)

    converged = False
    for iteration in range(1, 101):
        eta = np.clip(X @ beta, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(p * (1.0 - p), 1e-12)
        score = X.T @ (y - p)
        information = X.T @ (w[:, None] * X)
        step = np.linalg.pinv(information) @ score
        beta_new = beta + step
        if float(np.max(np.abs(step))) < 1e-10:
            beta = beta_new
            converged = True
            break
        beta = beta_new

    eta = np.clip(X @ beta, -35.0, 35.0)
    p = 1.0 / (1.0 + np.exp(-eta))
    w = np.maximum(p * (1.0 - p), 1e-12)
    information = X.T @ (w[:, None] * X)
    covariance = np.linalg.pinv(information)
    se_beta = float(math.sqrt(max(0.0, covariance[1, 1])))
    beta_n25 = float(beta[1])
    or_n25 = float(math.exp(beta_n25))
    ci_low = float(math.exp(beta_n25 - D01_WILSON_Z * se_beta))
    ci_high = float(math.exp(beta_n25 + D01_WILSON_Z * se_beta))

    return {
        "model": "unpenalized_logistic_regression",
        "outcome": "dataset_level_false_biological_acceptance_event",
        "predictor": "n_samples / 25",
        "interpretation": "odds ratio per +25 samples",
        "n_datasets": int(n.size),
        "event_count": int(y.sum()),
        "iterations": int(iteration),
        "converged": bool(converged),
        "log_odds_coefficient_per_25": beta_n25,
        "standard_error": se_beta,
        "odds_ratio_per_25": or_n25,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "formal_gate_consequence": "NONE_DESCRIPTIVE_ONLY",
    }


# -----------------------------------------------------------------------------
# Frozen artifact verification
# -----------------------------------------------------------------------------

def verify_05a2() -> dict[str, Any]:
    required = [
        AMENDMENT_05A2,
        EFFECTIVE_METRICS_05A2,
        EFFECTIVE_DECISIONS_05A2,
        REPORTING_05A2,
        FAILURE_POLICY_05A2,
        LOCK_05A2,
        FINAL_GATE_SCENARIOS,
        FINAL_GATE_REPLICATES,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen artifact missing: {path}")

    amendment = load_json(AMENDMENT_05A2)
    lock = load_json(LOCK_05A2)
    if clean_text(amendment.get("status")) != EXPECTED_05A2_STATUS:
        raise RuntimeError("05a2 amendment status mismatch.")
    if clean_text(lock.get("status")) != EXPECTED_05A2_STATUS:
        raise RuntimeError("05a2 lock status mismatch.")
    if sha256_file(AMENDMENT_05A2) != clean_text(lock.get("amendment_sha256")):
        raise RuntimeError("05a2 amendment SHA256 mismatch.")

    generated = lock.get("generated_files", {})
    for relative_name, record in generated.items():
        path = resolve_project_path(relative_name)
        if not path.is_file():
            raise FileNotFoundError(f"05a2 generated file missing: {path}")
        if sha256_file(path) != clean_text(record.get("sha256")):
            raise RuntimeError(f"05a2 generated file hash mismatch: {relative_name}")

    return {
        "amendment": amendment,
        "lock": lock,
        "failure_policy": load_json(FAILURE_POLICY_05A2),
        "reporting": pd.read_csv(REPORTING_05A2, dtype=str, low_memory=False).fillna(""),
        "decisions": pd.read_csv(EFFECTIVE_DECISIONS_05A2, dtype=str, low_memory=False).fillna(""),
        "metrics": pd.read_csv(EFFECTIVE_METRICS_05A2, dtype=str, low_memory=False).fillna(""),
        "scenarios": pd.read_csv(FINAL_GATE_SCENARIOS, dtype=str, low_memory=False).fillna(""),
        "replicates": pd.read_csv(FINAL_GATE_REPLICATES, dtype=str, low_memory=False).fillna(""),
    }


def verify_05f2_and_05f2a() -> dict[str, Any]:
    required = [
        SCRIPT_05F2,
        PROGRESS_05F2,
        SUMMARY_05F2,
        LOCK_05F2,
        LOCK_05F2A,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(
                "05f3b cannot run until the full 05f2 run is complete and 05f2b exists. "
                f"Missing: {path}"
            )

    lock_05f2a = load_json(LOCK_05F2A)
    if clean_text(lock_05f2a.get("status")) != EXPECTED_05F2A_LOCK_STATUS:
        raise RuntimeError("05f2b aggregation-plan lock has unexpected status.")
    if clean_text(lock_05f2a.get("aggregation_script_sha256")) != sha256_file(SCRIPT_PATH):
        raise RuntimeError(
            "Current 05f3b script differs from the version frozen by 05f2b before scientific unblinding. "
            "Do not aggregate with a modified script."
        )

    if clean_text(lock_05f2a.get("preexisting_05f3_sha256")) != EXPECTED_PREEXISTING_05F3_SHA256:
        raise RuntimeError("05f2b lock does not certify the pre-existing 05f3 source hash.")
    if not PREEXISTING_05F3.is_file() or sha256_file(PREEXISTING_05F3) != EXPECTED_PREEXISTING_05F3_SHA256:
        raise RuntimeError("Pre-existing 05f3 source changed after the 05f2b freeze.")
    if not parse_bool(lock_05f2a.get("scientific_core_ast_equivalent_to_preexisting_05f3")):
        raise RuntimeError("05f2b lock does not certify scientific-core equivalence to pre-existing 05f3.")

    lock = load_json(LOCK_05F2)
    if clean_text(lock.get("status")) != EXPECTED_05F2_LOCK_STATUS:
        raise RuntimeError("05f2 full-run lock status mismatch.")
    if clean_text(lock.get("script_version")) != EXPECTED_05F2_SCRIPT_VERSION:
        raise RuntimeError("05f2 full-run script version mismatch.")
    if clean_text(lock.get("script_sha256")) != sha256_file(SCRIPT_05F2):
        raise RuntimeError("Current 05f2 script differs from the completed-run hash.")
    if int(lock.get("n_datasets", -1)) != EXPECTED_DATASETS:
        raise RuntimeError("05f2 completion lock dataset count mismatch.")

    for key, expected_path in [
        ("progress_manifest", PROGRESS_05F2),
        ("completion_summary", SUMMARY_05F2),
    ]:
        record = lock.get(key, {})
        path = resolve_project_path(record.get("path"))
        if path.resolve() != expected_path.resolve():
            raise RuntimeError(f"05f2 lock {key} path mismatch.")
        if sha256_file(path) != clean_text(record.get("sha256")):
            raise RuntimeError(f"05f2 lock {key} hash mismatch.")

    state = lock.get("scientific_state", {})
    forbidden_true = [
        "scientific_metrics_printed",
        "scientific_metrics_aggregated",
        "D01_D02_D03_evaluated",
        "final_gate_PASS_FAIL_evaluated",
        "TARGET_accessed",
        "outcomes_accessed",
        "threshold_retuning_performed",
    ]
    if any(parse_bool(state.get(key)) for key in forbidden_true):
        raise RuntimeError("05f2 scientific blinding state is not clean before aggregation.")

    progress = pd.read_csv(PROGRESS_05F2, dtype=str, low_memory=False).fillna("")
    if progress.shape[0] != EXPECTED_DATASETS:
        raise RuntimeError("05f2 progress manifest must contain exactly 3,100 completed rows.")
    if progress["dataset_key"].duplicated().any():
        raise RuntimeError("05f2 progress manifest contains duplicate dataset keys.")

    return {
        "lock_05f2": lock,
        "lock_05f2a": lock_05f2a,
        "progress": progress,
    }


# -----------------------------------------------------------------------------
# Sealed shard extraction
# -----------------------------------------------------------------------------

def extract_sealed_results(
    progress: pd.DataFrame,
    scenarios: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario_lookup = scenarios.set_index("scenario_id", drop=False).to_dict(orient="index")
    dataset_rows: list[dict[str, Any]] = []
    factor_rows: list[dict[str, Any]] = []

    for ordinal, row in enumerate(progress.itertuples(index=False), start=1):
        shard_path = resolve_project_path(row.shard_path)
        if not shard_path.is_file():
            raise FileNotFoundError(f"Final-gate shard missing: {shard_path}")
        expected_hash = clean_text(row.shard_sha256)
        if sha256_file(shard_path) != expected_hash:
            raise RuntimeError(f"Final-gate shard hash mismatch: {shard_path}")

        shard = load_json(shard_path)
        payload = shard.get("scientific_payload")
        if not isinstance(payload, dict):
            raise RuntimeError("Final-gate shard lacks scientific_payload.")

        dataset = payload.get("dataset_row")
        factors = payload.get("factor_rows")
        if not isinstance(dataset, dict) or not isinstance(factors, list):
            raise RuntimeError("Final-gate scientific payload schema mismatch.")
        if clean_text(dataset.get("dataset_key")) != clean_text(row.dataset_key):
            raise RuntimeError("Final-gate dataset key mismatch between progress and shard.")
        if int(dataset.get("n_stability_resamples", -1)) != EXPECTED_STABILITY_RESAMPLES:
            raise RuntimeError("Final-gate shard does not contain 100 stability resamples.")
        if clean_text(dataset.get("compute_status")) != "COMPLETE":
            raise RuntimeError("Final-gate dataset scientific payload is not COMPLETE.")

        scenario_id = clean_text(dataset.get("scenario_id"))
        if scenario_id not in scenario_lookup:
            raise RuntimeError(f"Unknown scenario ID in final-gate shard: {scenario_id}")
        scenario_record = scenario_lookup[scenario_id]

        merged_dataset = dict(dataset)
        for key, value in scenario_record.items():
            merged_dataset[f"scenario_{key}"] = value
        dataset_rows.append(merged_dataset)

        for factor in factors:
            if not isinstance(factor, dict):
                raise RuntimeError("Final-gate factor row is not an object.")
            merged_factor = dict(factor)
            for key, value in scenario_record.items():
                merged_factor[f"scenario_{key}"] = value
            factor_rows.append(merged_factor)

        if ordinal == 1 or ordinal % 250 == 0 or ordinal == EXPECTED_DATASETS:
            print(f"Shard extraction: {ordinal:04d}/{EXPECTED_DATASETS:04d} verified/read")

    datasets = pd.DataFrame(dataset_rows)
    factors = pd.DataFrame(factor_rows)

    if datasets.shape[0] != EXPECTED_DATASETS:
        raise RuntimeError("Extracted final-gate dataset row count mismatch.")
    if datasets["dataset_key"].duplicated().any():
        raise RuntimeError("Extracted final-gate dataset keys are not unique.")
    if factors.empty:
        raise RuntimeError("Extracted final-gate factor table is empty.")

    return datasets, factors


# -----------------------------------------------------------------------------
# Event / opportunity definitions
# -----------------------------------------------------------------------------

def with_numeric_scenario_fields(datasets: pd.DataFrame) -> pd.DataFrame:
    table = datasets.copy()
    for column in [
        "scenario_n_samples",
        "scenario_shared_signal_strength",
        "scenario_shared_factor_correlation",
        "scenario_true_biological_shared_factors",
        "scenario_true_technical_shared_factors",
    ]:
        if column not in table.columns:
            raise RuntimeError(f"Missing scenario field after merge: {column}")
        table[column] = pd.to_numeric(table[column], errors="raise")
    return table


def add_dataset_false_acceptance_event(
    datasets: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    table = datasets.copy()
    count_col = pd.to_numeric(
        table["n_stable_unmatched_nontechnical_factors"],
        errors="raise",
    ).astype(int)
    table["false_biological_acceptance_event"] = count_col.gt(0)

    # Cross-check against factor rows; the event is the familywise "at least one"
    # false biological factor event frozen for D01.
    factor_events = (
        factors.assign(
            false_event=factors["false_biological_acceptance"].map(parse_bool)
        )
        .groupby("dataset_key", sort=False)["false_event"]
        .any()
    )
    observed = table["dataset_key"].map(factor_events).fillna(False).astype(bool)
    if not np.array_equal(
        observed.to_numpy(dtype=bool),
        table["false_biological_acceptance_event"].to_numpy(dtype=bool),
    ):
        raise RuntimeError(
            "Dataset-level false biological acceptance event disagrees with factor-level sealed audit."
        )
    return table


def pooled_wilson_record(
    report_id: str,
    role: str,
    selected: pd.DataFrame,
    formal_gate_consequence: str,
    decision_rule: str = "",
    threshold: float | None = None,
    decision_direction: str = "",
) -> dict[str, Any]:
    numerator = int(selected["false_biological_acceptance_event"].sum())
    denominator = int(selected.shape[0])
    low, high = wilson_interval(numerator, denominator)
    point = numerator / denominator

    passed: bool | None = None
    if formal_gate_consequence == "D01":
        if threshold is None or decision_direction != "upper_ci_le":
            raise RuntimeError("D01 decision rule not configured.")
        passed = bool(high <= threshold)

    return {
        "report_id": report_id,
        "role": role,
        "metric": "gate_familywise_false_acceptance",
        "numerator": numerator,
        "denominator": denominator,
        "point_estimate": point,
        "ci_method": "two-sided 95% Wilson score interval; z=1.959963984540054; no continuity correction",
        "ci95_low": low,
        "ci95_high": high,
        "formal_gate_consequence": formal_gate_consequence,
        "decision_rule": decision_rule,
        "PASS": passed,
    }


def biological_recovery_count(
    factor_subset: pd.DataFrame,
) -> int:
    stable = factor_subset["passes_stability_gate"].map(parse_bool)
    matched = factor_subset["biological_truth_match"].map(parse_bool)
    selected = factor_subset[stable & matched].copy()

    # Hungarian matching makes biological_truth_factor_index unique within a
    # dataset among assigned pairs. Cross-check rather than silently assuming.
    if selected.empty:
        return 0
    keys = selected[["dataset_key", "biological_truth_factor_index"]].copy()
    keys["biological_truth_factor_index"] = keys["biological_truth_factor_index"].map(nullable_int)
    if keys["biological_truth_factor_index"].isna().any():
        raise RuntimeError("Stable biological match lacks truth-factor index.")
    if keys.duplicated().any():
        raise RuntimeError("Duplicate retained biological truth match within dataset.")
    return int(keys.shape[0])


def technical_opportunity_counts(
    dataset_subset: pd.DataFrame,
    factor_subset: pd.DataFrame,
) -> tuple[int, int, int]:
    false_accept = 0
    correctly_flagged = 0
    denominator = 0

    grouped = {
        key: group.copy()
        for key, group in factor_subset.groupby("dataset_key", sort=False)
    }

    for dataset in dataset_subset.itertuples(index=False):
        n_truth = int(dataset.n_true_technical_shared_factors)
        denominator += n_truth
        if n_truth <= 0:
            continue
        rows = grouped.get(clean_text(dataset.dataset_key))
        if rows is None:
            raise RuntimeError("Technical-truth dataset has no factor rows.")

        for truth_index in range(n_truth):
            assigned = rows[
                rows["technical_truth_factor_index"].map(nullable_int).eq(truth_index)
            ].copy()
            if assigned.shape[0] > 1:
                raise RuntimeError("Technical Hungarian assignment is not unique.")
            if assigned.empty:
                continue

            factor = assigned.iloc[0]
            retained = parse_bool(factor["passes_stability_gate"])
            flagged = parse_bool(factor["technical_assigned_truth_match"])

            if retained and flagged:
                correctly_flagged += 1
            elif retained and not flagged:
                false_accept += 1

    return false_accept, correctly_flagged, denominator


# -----------------------------------------------------------------------------
# Formal gates and mandatory reporting
# -----------------------------------------------------------------------------

def compute_formal_results(
    datasets: pd.DataFrame,
    factors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    datasets = with_numeric_scenario_fields(datasets)
    datasets = add_dataset_false_acceptance_event(datasets, factors)

    d01_selector = (
        datasets["scenario_true_biological_shared_factors"].eq(0)
        & datasets["scenario_true_technical_shared_factors"].eq(0)
    )
    d01 = datasets[d01_selector].copy()
    if d01.shape[0] != EXPECTED_D01_DATASETS:
        raise RuntimeError(f"D01 selector expected 1,000 datasets; found {d01.shape[0]}.")

    independent = d01[
        d01["scenario_scenario_family"].eq("final_independent_gate_curve")
    ].copy()
    correlated_private = d01[
        d01["scenario_scenario_family"].eq("correlated_private_null")
    ].copy()
    if independent.shape[0] != EXPECTED_D01_INDEPENDENT_DATASETS:
        raise RuntimeError("D01 independent-null pool is not 800 datasets.")
    if correlated_private.shape[0] != EXPECTED_D01_CORRELATED_PRIVATE_DATASETS:
        raise RuntimeError("D01 correlated-private pool is not 200 datasets.")
    if set(correlated_private["scenario_id"].unique()) != {CORRELATED_PRIVATE_SCENARIO_ID}:
        raise RuntimeError("Unexpected correlated-private D01 scenario membership.")

    mandatory_rows: list[dict[str, Any]] = []
    primary_d01 = pooled_wilson_record(
        "R01_D01_PRIMARY_POOLED",
        "primary_decision",
        d01,
        "D01",
        decision_rule="PASS iff upper endpoint of two-sided 95% Wilson CI <= 0.10",
        threshold=D01_LIMIT,
        decision_direction="upper_ci_le",
    )
    mandatory_rows.append(primary_d01)
    mandatory_rows.append(
        pooled_wilson_record(
            "R02_D01_INDEPENDENT_NULL_POOLED",
            "mandatory_secondary",
            independent,
            "NONE",
        )
    )
    mandatory_rows.append(
        pooled_wilson_record(
            "R03_D01_CORRELATED_PRIVATE",
            "mandatory_secondary",
            correlated_private,
            "NONE",
        )
    )

    for n_value in [50, 62, 75, 100]:
        selected = independent[
            independent["scenario_n_samples"].eq(n_value)
        ].copy()
        if selected.shape[0] != EXPECTED_D01_PER_N_DATASETS:
            raise RuntimeError(
                f"D01 independent n={n_value} expected 200 datasets; found {selected.shape[0]}."
            )
        mandatory_rows.append(
            pooled_wilson_record(
                f"R_D01_INDEPENDENT_N{n_value}",
                "mandatory_descriptive",
                selected,
                "NONE",
            )
        )

    trend = fit_logistic_sample_size_trend(
        independent["scenario_n_samples"].to_numpy(dtype=float),
        independent["false_biological_acceptance_event"].astype(int).to_numpy(),
    )

    # D02/D03 exact STANDARD selector.
    standard_datasets = datasets[
        datasets["scenario_id"].eq(STANDARD_SCENARIO_ID)
    ].copy()
    standard_factors = factors[
        factors["scenario_id"].eq(STANDARD_SCENARIO_ID)
    ].copy()
    if standard_datasets.shape[0] != EXPECTED_D02_DATASETS:
        raise RuntimeError("D02/D03 STANDARD selector is not 100 datasets.")

    denominator_d02 = int(
        pd.to_numeric(
            standard_datasets["n_true_biological_shared_factors"],
            errors="raise",
        ).sum()
    )
    if denominator_d02 != EXPECTED_D02_FACTOR_OPPORTUNITIES:
        raise RuntimeError("D02 denominator is not exactly 400 true-factor opportunities.")
    numerator_d02 = biological_recovery_count(standard_factors)
    tpr = numerator_d02 / denominator_d02
    tpr_low, tpr_high = wilson_interval(numerator_d02, denominator_d02)
    fn = denominator_d02 - numerator_d02
    fnr = fn / denominator_d02
    fnr_low = 1.0 - tpr_high
    fnr_high = 1.0 - tpr_low
    d02_pass = bool(tpr >= D02_TPR_MIN)
    d03_pass = bool(fnr <= D03_FNR_MAX)
    if d02_pass != d03_pass:
        raise RuntimeError("D02 and algebraic-mirror D03 decisions disagree unexpectedly.")

    # D04 rotational subspace. The sealed dataset-level
    # reference_shared_subspace_recovery is the mean canonical correlation,
    # i.e. mean cosine of principal angles.
    d04_datasets = datasets[
        datasets["scenario_id"].eq(ROTATIONAL_SCENARIO_ID)
    ].copy()
    if d04_datasets.shape[0] != EXPECTED_D04_DATASETS:
        raise RuntimeError("D04 rotational selector is not 100 datasets.")
    d04_values = pd.to_numeric(
        d04_datasets["reference_shared_subspace_recovery"],
        errors="coerce",
    ).to_numpy(dtype=float)
    if not np.isfinite(d04_values).all():
        raise RuntimeError("D04 subspace recovery contains non-finite values.")
    d04_point = float(np.mean(d04_values))
    d04_low, d04_high = percentile_bootstrap_mean(d04_values, "D04_ROTATIONAL_SUBSPACE")
    d04_pass = bool(d04_point >= D04_SUBSPACE_MIN)

    # D05 mixed support: denominator is matched-and-retained biological factors,
    # exactly as frozen in the operating metric definition.
    d05_datasets = datasets[
        datasets["scenario_id"].eq(MIXED_SUPPORT_SCENARIO_ID)
    ].copy()
    d05_factors = factors[
        factors["scenario_id"].eq(MIXED_SUPPORT_SCENARIO_ID)
    ].copy()
    if d05_datasets.shape[0] != EXPECTED_D05_DATASETS:
        raise RuntimeError("D05 mixed-support selector is not 100 datasets.")
    d05_stable = d05_factors["passes_stability_gate"].map(parse_bool)
    d05_match = d05_factors["biological_truth_match"].map(parse_bool)
    d05_eligible = d05_factors[d05_stable & d05_match].copy()
    if d05_eligible.empty:
        d05_num = 0
        d05_den = 0
        d05_point = None
        d05_low = None
        d05_high = None
        d05_pass = False
    else:
        exact = d05_eligible["view_support_exact"].map(parse_bool)
        d05_num = int(exact.sum())
        d05_den = int(exact.shape[0])
        d05_point = d05_num / d05_den
        d05_low, d05_high = wilson_interval(d05_num, d05_den)
        d05_pass = bool(d05_point >= D05_VIEW_SUPPORT_MIN)

    # D06 technical-null specificity, pooled over the two frozen technical-null
    # scenarios. Each true technical factor is one opportunity. A false
    # biological technical acceptance occurs when its Hungarian-assigned estimated
    # factor is retained by the stability gate but fails the frozen technical
    # truth-match threshold used to flag it as technical.
    d06_datasets = datasets[
        datasets["scenario_id"].isin(BATCH_NULL_SCENARIO_IDS)
    ].copy()
    d06_factors = factors[
        factors["scenario_id"].isin(BATCH_NULL_SCENARIO_IDS)
    ].copy()
    if d06_datasets.shape[0] != EXPECTED_D06_DATASETS:
        raise RuntimeError("D06 technical-null selector is not 400 datasets.")
    d06_num, tech_detect_num, d06_den = technical_opportunity_counts(
        d06_datasets,
        d06_factors,
    )
    if d06_den != EXPECTED_D06_TECHNICAL_OPPORTUNITIES:
        raise RuntimeError("D06 technical-factor denominator is not 400.")
    d06_point = d06_num / d06_den
    d06_low, d06_high = wilson_interval(d06_num, d06_den)
    d06_pass = bool(d06_point <= D06_BATCH_FALSE_ACCEPT_MAX)
    tech_sensitivity = tech_detect_num / d06_den
    tech_low, tech_high = wilson_interval(tech_detect_num, d06_den)

    decision_rows = [
        {
            "decision_id": "D01_NULL_SPECIFICITY",
            "metric": "gate_familywise_false_acceptance",
            "numerator": int(primary_d01["numerator"]),
            "denominator": int(primary_d01["denominator"]),
            "point_estimate": float(primary_d01["point_estimate"]),
            "ci95_low": float(primary_d01["ci95_low"]),
            "ci95_high": float(primary_d01["ci95_high"]),
            "decision_quantity": "upper endpoint of two-sided 95% Wilson CI",
            "threshold": D01_LIMIT,
            "pass_rule": "ci95_high <= 0.10",
            "PASS": bool(primary_d01["PASS"]),
            "independence_note": "primary core specificity decision",
        },
        {
            "decision_id": "D02_STANDARD_SENSITIVITY",
            "metric": "gate_true_positive_rate",
            "numerator": numerator_d02,
            "denominator": denominator_d02,
            "point_estimate": tpr,
            "ci95_low": tpr_low,
            "ci95_high": tpr_high,
            "decision_quantity": "point estimate",
            "threshold": D02_TPR_MIN,
            "pass_rule": "point_estimate >= 0.80",
            "PASS": d02_pass,
            "independence_note": "primary core sensitivity decision",
        },
        {
            "decision_id": "D03_STANDARD_FNR",
            "metric": "gate_false_negative_rate",
            "numerator": fn,
            "denominator": denominator_d02,
            "point_estimate": fnr,
            "ci95_low": fnr_low,
            "ci95_high": fnr_high,
            "decision_quantity": "point estimate",
            "threshold": D03_FNR_MAX,
            "pass_rule": "point_estimate <= 0.20",
            "PASS": d03_pass,
            "independence_note": "algebraic mirror of D02 on identical denominator; not independent evidence",
        },
        {
            "decision_id": "D04_ROTATIONAL_SUBSPACE",
            "metric": "subspace_mean_principal_angle_cosine",
            "numerator": None,
            "denominator": int(d04_datasets.shape[0]),
            "point_estimate": d04_point,
            "ci95_low": d04_low,
            "ci95_high": d04_high,
            "decision_quantity": "mean dataset-level mean canonical correlation",
            "threshold": D04_SUBSPACE_MIN,
            "pass_rule": "point_estimate >= 0.80",
            "PASS": d04_pass,
            "independence_note": "narrow rotational-subspace gate",
        },
        {
            "decision_id": "D05_VIEW_SUPPORT",
            "metric": "view_support_classification_accuracy",
            "numerator": d05_num,
            "denominator": d05_den,
            "point_estimate": d05_point,
            "ci95_low": d05_low,
            "ci95_high": d05_high,
            "decision_quantity": "point estimate",
            "threshold": D05_VIEW_SUPPORT_MIN,
            "pass_rule": "point_estimate >= 0.80",
            "PASS": d05_pass,
            "independence_note": "narrow mixed-support gate",
        },
        {
            "decision_id": "D06_BATCH_SPECIFICITY",
            "metric": "batch_false_biological_acceptance_rate",
            "numerator": d06_num,
            "denominator": d06_den,
            "point_estimate": d06_point,
            "ci95_low": d06_low,
            "ci95_high": d06_high,
            "decision_quantity": "point estimate",
            "threshold": D06_BATCH_FALSE_ACCEPT_MAX,
            "pass_rule": "point_estimate <= 0.10",
            "PASS": d06_pass,
            "independence_note": "narrow technical-confound specificity gate",
        },
        {
            "decision_id": "D07_NO_POSTHOC_RETUNING",
            "metric": "process_integrity",
            "numerator": None,
            "denominator": None,
            "point_estimate": None,
            "ci95_low": None,
            "ci95_high": None,
            "decision_quantity": "process audit",
            "threshold": None,
            "pass_rule": "all thresholds/selection rules unchanged from frozen 05a2/05f2a",
            "PASS": True,
            "independence_note": "process gate, not a performance metric",
        },
    ]

    auxiliary = {
        "technical_factor_detection_sensitivity_D06_pool": {
            "numerator": tech_detect_num,
            "denominator": d06_den,
            "point_estimate": tech_sensitivity,
            "ci95_low": tech_low,
            "ci95_high": tech_high,
        },
        "d01_trend": trend,
    }

    return (
        pd.DataFrame(decision_rows),
        pd.DataFrame(mandatory_rows),
        auxiliary,
    )


# -----------------------------------------------------------------------------
# Descriptive scenario / operating metric summaries
# -----------------------------------------------------------------------------

def build_scenario_summary(
    datasets: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    datasets = with_numeric_scenario_fields(datasets)
    datasets = add_dataset_false_acceptance_event(datasets, factors)

    rows: list[dict[str, Any]] = []
    for scenario_id, dset in datasets.groupby("scenario_id", sort=False):
        fset = factors[factors["scenario_id"].eq(scenario_id)].copy()

        n_true_bio = int(
            pd.to_numeric(dset["n_true_biological_shared_factors"], errors="raise").sum()
        )
        recovered = biological_recovery_count(fset) if n_true_bio > 0 else 0
        stable_total = int(pd.to_numeric(dset["n_stable_factors"], errors="raise").sum())
        stable_bio = int(pd.to_numeric(dset["n_stable_biological_factors"], errors="raise").sum())
        stable_tech = int(pd.to_numeric(dset["n_stable_technical_factors"], errors="raise").sum())
        stable_false = int(
            pd.to_numeric(
                dset["n_stable_unmatched_nontechnical_factors"],
                errors="raise",
            ).sum()
        )
        stable_nontechnical = stable_bio + stable_false

        subspace = pd.to_numeric(
            dset["reference_shared_subspace_recovery"],
            errors="coerce",
        ).to_numpy(dtype=float)
        finite_subspace = subspace[np.isfinite(subspace)]

        support_stable = fset["passes_stability_gate"].map(parse_bool)
        support_match = fset["biological_truth_match"].map(parse_bool)
        support_rows = fset[support_stable & support_match]
        support_accuracy = (
            float(support_rows["view_support_exact"].map(parse_bool).mean())
            if not support_rows.empty
            else None
        )

        rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_family": clean_text(dset["scenario_scenario_family"].iloc[0]),
                "calibration_role": clean_text(dset["scenario_calibration_role"].iloc[0]),
                "n_samples": int(float(dset["scenario_n_samples"].iloc[0])),
                "shared_signal_strength": float(
                    dset["scenario_shared_signal_strength"].iloc[0]
                ),
                "n_datasets": int(dset.shape[0]),
                "true_biological_factor_opportunities": n_true_bio,
                "matched_retained_biological_factors": recovered,
                "biological_factor_TPR": (
                    recovered / n_true_bio if n_true_bio > 0 else None
                ),
                "familywise_false_acceptance_numerator": int(
                    dset["false_biological_acceptance_event"].sum()
                ),
                "familywise_false_acceptance_rate": float(
                    dset["false_biological_acceptance_event"].mean()
                ),
                "stable_factors_total": stable_total,
                "stable_biological_factors": stable_bio,
                "stable_technical_factors": stable_tech,
                "stable_unmatched_nontechnical_factors": stable_false,
                "pooled_stability_gated_FDP": (
                    stable_false / stable_nontechnical
                    if stable_nontechnical > 0
                    else 0.0
                ),
                "mean_reference_shared_subspace_recovery": (
                    float(np.mean(finite_subspace))
                    if finite_subspace.size > 0
                    else None
                ),
                "matched_retained_view_support_accuracy": support_accuracy,
            }
        )

    table = pd.DataFrame(rows)
    if table.shape[0] != EXPECTED_SCENARIOS:
        raise RuntimeError("Scenario summary does not contain 24 scenarios.")
    return table


def build_operating_metric_summary(
    datasets: pd.DataFrame,
    factors: pd.DataFrame,
    decisions: pd.DataFrame,
    auxiliary: dict[str, Any],
) -> pd.DataFrame:
    stable = factors["passes_stability_gate"].map(parse_bool)
    stable_factors = factors[stable].copy()

    # Pooled FDP across all stable nontechnical biological-claim-eligible retained
    # factors: stable biological matches plus stable unmatched nontechnical factors.
    n_stable_bio = int(
        pd.to_numeric(datasets["n_stable_biological_factors"], errors="raise").sum()
    )
    n_stable_false = int(
        pd.to_numeric(
            datasets["n_stable_unmatched_nontechnical_factors"],
            errors="raise",
        ).sum()
    )
    nontechnical_den = n_stable_bio + n_stable_false

    count_bias = (
        pd.to_numeric(datasets["n_stable_biological_factors"], errors="raise")
        - pd.to_numeric(datasets["n_true_biological_shared_factors"], errors="raise")
    ).to_numpy(dtype=float)
    bias_low, bias_high = percentile_bootstrap_mean(count_bias, "FACTOR_COUNT_BIAS")

    leakage_eligible = stable_factors[
        stable_factors["private_truth_audit_available"].map(parse_bool)
    ].copy()
    leakage_num = int(
        leakage_eligible["private_stronger_than_shared_truth"].map(parse_bool).sum()
    )
    leakage_den = int(leakage_eligible.shape[0])
    if leakage_den > 0:
        leakage_low, leakage_high = wilson_interval(leakage_num, leakage_den)
        leakage_point = leakage_num / leakage_den
    else:
        leakage_low = leakage_high = leakage_point = None

    rows = [
        {
            "metric_id": "pooled_stability_gated_false_discovery_proportion",
            "scope": "all final-gate datasets; stable nontechnical factors",
            "numerator": n_stable_false,
            "denominator": nontechnical_den,
            "point_estimate": (
                n_stable_false / nontechnical_den if nontechnical_den > 0 else 0.0
            ),
            "ci95_low": None,
            "ci95_high": None,
            "role": "descriptive_operating_metric",
        },
        {
            "metric_id": "factor_count_bias",
            "scope": "all 3,100 datasets",
            "numerator": None,
            "denominator": int(datasets.shape[0]),
            "point_estimate": float(np.mean(count_bias)),
            "ci95_low": bias_low,
            "ci95_high": bias_high,
            "role": "descriptive_operating_metric",
        },
        {
            "metric_id": "private_to_shared_leakage_rate",
            "scope": "stable factors where private truth audit is available",
            "numerator": leakage_num,
            "denominator": leakage_den,
            "point_estimate": leakage_point,
            "ci95_low": leakage_low,
            "ci95_high": leakage_high,
            "role": "descriptive_operating_metric",
        },
        {
            "metric_id": "technical_factor_detection_sensitivity",
            "scope": "D06 technical-null pool",
            "numerator": auxiliary["technical_factor_detection_sensitivity_D06_pool"]["numerator"],
            "denominator": auxiliary["technical_factor_detection_sensitivity_D06_pool"]["denominator"],
            "point_estimate": auxiliary["technical_factor_detection_sensitivity_D06_pool"]["point_estimate"],
            "ci95_low": auxiliary["technical_factor_detection_sensitivity_D06_pool"]["ci95_low"],
            "ci95_high": auxiliary["technical_factor_detection_sensitivity_D06_pool"]["ci95_high"],
            "role": "technical_audit_metric",
        },
    ]

    # Also carry formal gate results into the operating-metric summary for a
    # single manuscript-facing table.
    for row in decisions.to_dict(orient="records"):
        if row["decision_id"] == "D07_NO_POSTHOC_RETUNING":
            continue
        rows.append(
            {
                "metric_id": row["metric"],
                "scope": row["decision_id"],
                "numerator": row["numerator"],
                "denominator": row["denominator"],
                "point_estimate": row["point_estimate"],
                "ci95_low": row["ci95_low"],
                "ci95_high": row["ci95_high"],
                "role": "formal_gate_metric",
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Failure-policy routing
# -----------------------------------------------------------------------------

def select_failure_policy(
    failure_policy: dict[str, Any],
    decisions: pd.DataFrame,
) -> dict[str, Any]:
    decision_map = {
        row["decision_id"]: bool(row["PASS"])
        for row in decisions.to_dict(orient="records")
    }
    d01 = decision_map["D01_NULL_SPECIFICITY"]
    d02 = decision_map["D02_STANDARD_SENSITIVITY"]
    d03 = decision_map["D03_STANDARD_FNR"]

    if d02 != d03:
        raise RuntimeError("Frozen mirrored D02/D03 status mismatch.")

    if d01 and d02 and d03:
        index = 0
        core_status = "CORE_PASS"
    elif (not d01) and d02 and d03:
        index = 1
        core_status = "CORE_FAIL_SPECIFICITY"
    elif d01 and (not d02) and (not d03):
        index = 2
        core_status = "CORE_FAIL_SENSITIVITY"
    else:
        index = 3
        core_status = "CORE_FAIL_BOTH"

    core_outcomes = failure_policy.get("core_outcomes", [])
    if not isinstance(core_outcomes, list) or len(core_outcomes) < 4:
        raise RuntimeError("Frozen 05a2 failure policy lacks four core outcomes.")

    narrow = {
        row["decision_id"]: bool(row["PASS"])
        for row in decisions.to_dict(orient="records")
        if row["decision_id"] in {
            "D04_ROTATIONAL_SUBSPACE",
            "D05_VIEW_SUPPORT",
            "D06_BATCH_SPECIFICITY",
            "D07_NO_POSTHOC_RETUNING",
        }
    }

    return {
        "core_status": core_status,
        "D01_PASS": d01,
        "D02_PASS": d02,
        "D03_PASS": d03,
        "frozen_core_outcome": core_outcomes[index],
        "narrow_gate_status": narrow,
        "frozen_narrow_gate_failure_policy": failure_policy.get("narrow_gate_failures", {}),
        "stress_subgroup_rule": failure_policy.get("stress_subgroup_rule"),
        "outcome_boundary": failure_policy.get("outcome_boundary"),
    }


# -----------------------------------------------------------------------------
# Existing result verification
# -----------------------------------------------------------------------------

def verify_existing_results_lock() -> None:
    lock = load_json(OUTPUT_LOCK)
    if clean_text(lock.get("status")) != "FROZEN_FINAL_GATE_SCIENTIFIC_RESULTS_NO_RETUNING":
        raise RuntimeError("Existing 05f3 results lock has unexpected status.")
    if clean_text(lock.get("script_sha256")) != sha256_file(SCRIPT_PATH):
        raise RuntimeError("Current 05f3 script differs from frozen result hash.")

    generated = lock.get("generated_files", {})
    for relative_name, record in generated.items():
        path = resolve_project_path(relative_name)
        if not path.is_file():
            raise FileNotFoundError(f"Frozen 05f3 output missing: {path}")
        if sha256_file(path) != clean_text(record.get("sha256")):
            raise RuntimeError(f"Frozen 05f3 output hash mismatch: {relative_name}")

    result = load_json(OUTPUT_RESULT)
    print("=" * 88)
    print("Existing Paper 5 05f3 final-gate scientific results: VERIFIED")
    print("=" * 88)
    print(f"Core status: {result['failure_policy_route']['core_status']}")
    for row in result["formal_decisions"]:
        point = row.get("point_estimate")
        point_text = "N/A" if point is None else f"{100.0 * float(point):.2f}%"
        print(f"{row['decision_id']}: {point_text} | {'PASS' if row['PASS'] else 'FAIL'}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("No files were rewritten.")
    print("=" * 88)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    print("=" * 88)
    print("Paper 5 - aggregate frozen final-gate scientific results (pre-unblinding locked)")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("GPU computation in 05f3b: NO")
    print("Model fitting in 05f3b: NO")
    print("TARGET/outcome/biology access in 05f3b: NO")
    print("Threshold retuning in 05f3b: FORBIDDEN")
    print("This is the first all-3,100 scientific final-gate aggregation stage.")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_LOCK.exists():
        verify_existing_results_lock()
        return

    frozen = verify_05a2()
    run = verify_05f2_and_05f2a()

    scenarios = frozen["scenarios"].copy()
    if scenarios.shape[0] != EXPECTED_SCENARIOS:
        raise RuntimeError("Frozen final-gate scenario registry is not 24 rows.")

    # Validate all expected IDs used by the frozen implementation.
    scenario_ids = set(scenarios["scenario_id"].astype(str))
    required_ids = {
        STANDARD_SCENARIO_ID,
        ROTATIONAL_SCENARIO_ID,
        MIXED_SUPPORT_SCENARIO_ID,
        CORRELATED_PRIVATE_SCENARIO_ID,
        *BATCH_NULL_SCENARIO_IDS,
    }
    missing_ids = required_ids - scenario_ids
    if missing_ids:
        raise RuntimeError(
            "Frozen final-gate registry lacks expected scenario IDs: "
            + ", ".join(sorted(missing_ids))
        )

    print("05a2 frozen decision/report/failure-policy layer: VERIFIED")
    print("05f2b post-start/pre-unblinding aggregation script hash: VERIFIED")
    print("05f2 3,100-dataset completion lock: VERIFIED")
    print("Reading sealed scientific payloads exactly once...")
    print()

    datasets, factors = extract_sealed_results(
        run["progress"],
        scenarios,
    )

    decisions, mandatory, auxiliary = compute_formal_results(
        datasets,
        factors,
    )
    scenario_summary = build_scenario_summary(
        datasets,
        factors,
    )
    operating = build_operating_metric_summary(
        datasets,
        factors,
        decisions,
        auxiliary,
    )
    failure_route = select_failure_policy(
        frozen["failure_policy"],
        decisions,
    )

    write_csv_atomic(decisions, OUTPUT_PRIMARY_DECISIONS)
    write_csv_atomic(mandatory, OUTPUT_MANDATORY_REPORTING)
    write_csv_atomic(scenario_summary, OUTPUT_SCENARIO_SUMMARY)
    write_csv_atomic(operating, OUTPUT_OPERATING_METRICS)
    write_json_atomic(auxiliary["d01_trend"], OUTPUT_D01_TREND)
    write_json_atomic(failure_route, OUTPUT_FAILURE_CONSEQUENCE)

    result = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": "FROZEN_FINAL_GATE_SCIENTIFIC_RESULTS_NO_RETUNING",
        "scope": {
            "n_scenarios": EXPECTED_SCENARIOS,
            "n_datasets": EXPECTED_DATASETS,
            "stability_resamples_per_dataset": EXPECTED_STABILITY_RESAMPLES,
            "method": "V4_SABLE_MO",
            "profile": "ARCH_COMPACT__REG_WEAK",
        },
        "formal_decisions": decisions.to_dict(orient="records"),
        "mandatory_D01_reporting": mandatory.to_dict(orient="records"),
        "D01_independent_null_sample_size_trend": auxiliary["d01_trend"],
        "failure_policy_route": failure_route,
        "bootstrap_implementation": {
            "role": "continuous descriptive/narrow-gate confidence intervals only",
            "replicates": BOOTSTRAP_REPLICATES,
            "master_seed": BOOTSTRAP_MASTER_SEED,
            "interval": "2.5th and 97.5th percentile of bootstrap means",
            "fixed_in_preexisting_05f3_and_relocked_by_05f2b_before_unblinding": True,
            "05f2_had_started_before_05f2b_freeze": True,
            "formal_D01_decision_uses_bootstrap": False,
            "formal_D02_D03_decisions_use_bootstrap": False,
        },
        "scientific_integrity": {
            "aggregation_lock_timing": "post-05f2-start but before scientific unblinding",
            "preexisting_05f3_source_sha256": EXPECTED_PREEXISTING_05F3_SHA256,
            "scientific_core_ast_equivalent_to_preexisting_05f3": True,
            "thresholds_retuned_after_results": False,
            "selectors_changed_after_results": False,
            "D01_Wilson_rule_changed": False,
            "D02_D03_STANDARD_selector_changed": False,
            "outcomes_used_to_rescue_gate": False,
            "TARGET_used_to_rescue_gate": False,
        },
        "source_records": {
            "05a2_amendment": file_record(AMENDMENT_05A2),
            "05a2_reporting_registry": file_record(REPORTING_05A2),
            "05a2_failure_policy": file_record(FAILURE_POLICY_05A2),
            "05f2_completion_lock": file_record(LOCK_05F2),
            "05f2b_aggregation_plan_lock": file_record(LOCK_05F2A),
        },
    }
    write_json_atomic(result, OUTPUT_RESULT)

    readme = (
        "Paper 5 05f3b — final-gate scientific aggregation\n"
        "================================================\n\n"
        "05f3b is the first stage that opens and aggregates scientific values from all\n"
        "3,100 sealed final-gate shards. It performs no model fitting and accesses no\n"
        "TARGET data, clinical outcomes, or biological annotations.\n\n"
        "Core gates\n"
        "----------\n"
        "D01: familywise false biological acceptance over the frozen N=1,000 null pool;\n"
        "PASS iff upper endpoint of the two-sided 95% Wilson interval <= 0.10.\n"
        "D02: G07 STANDARD pooled matched-and-retained true factors / 400 opportunities;\n"
        "PASS iff point estimate >= 0.80.\n"
        "D03: 1-D02 on the same denominator; PASS iff point estimate <= 0.20.\n\n"
        "Narrow gates\n"
        "------------\n"
        "D04 rotational subspace point estimate >= 0.80.\n"
        "D05 mixed-support classification accuracy >= 0.80.\n"
        "D06 batch false biological acceptance <= 0.10.\n"
        "D07 confirms no post-hoc retuning.\n\n"
        "The frozen 05a2 failure-consequence policy determines downstream interpretation.\n"
        "A failed criterion changes the conclusion, not the criterion.\n"
    )
    write_text_atomic(readme, OUTPUT_README)

    generated_paths = [
        OUTPUT_PRIMARY_DECISIONS,
        OUTPUT_MANDATORY_REPORTING,
        OUTPUT_SCENARIO_SUMMARY,
        OUTPUT_OPERATING_METRICS,
        OUTPUT_D01_TREND,
        OUTPUT_RESULT,
        OUTPUT_FAILURE_CONSEQUENCE,
        OUTPUT_README,
    ]

    lock = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "script_path": relative_path(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "created_at_utc": utc_now_iso(),
        "status": "FROZEN_FINAL_GATE_SCIENTIFIC_RESULTS_NO_RETUNING",
        "05f2_completion_lock_sha256": sha256_file(LOCK_05F2),
        "05f2b_aggregation_plan_lock_sha256": sha256_file(LOCK_05F2A),
        "05a2_amendment_sha256": sha256_file(AMENDMENT_05A2),
        "core_status": failure_route["core_status"],
        "generated_files": {
            relative_path(path): file_record(path)
            for path in generated_paths
        },
        "rewrite_policy": (
            "05f3b results are immutable. No threshold, selector, CI decision rule, "
            "scenario membership, or aggregation denominator may be changed after "
            "these results are opened. Any later sensitivity analysis must be "
            "separately versioned and labeled exploratory."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    print()
    print("-" * 88)
    print("Final-gate formal decisions")
    print("-" * 88)
    for row in decisions.to_dict(orient="records"):
        point = row["point_estimate"]
        if point is None:
            point_text = "process"
        else:
            point_text = f"{100.0 * float(point):.2f}%"
        numerator = row["numerator"]
        denominator = row["denominator"]
        count_text = (
            ""
            if numerator is None or denominator is None
            else f" ({int(numerator)}/{int(denominator)})"
        )
        ci_text = (
            ""
            if row["ci95_low"] is None or row["ci95_high"] is None
            else f" | 95% CI [{100.0*float(row['ci95_low']):.2f}%, {100.0*float(row['ci95_high']):.2f}%]"
        )
        print(
            f"{row['decision_id']}: {point_text}{count_text}{ci_text} | "
            f"{'PASS' if bool(row['PASS']) else 'FAIL'}"
        )

    print()
    print(f"Core final-gate status: {failure_route['core_status']}")
    frozen_outcome = failure_route["frozen_core_outcome"]
    print(f"Frozen interpretation: {clean_text(frozen_outcome.get('interpretation'))}")
    print(f"Frozen downstream rule: {clean_text(frozen_outcome.get('downstream'))}")
    print("Threshold retuning after results: NO")
    print("TARGET/outcomes used to rescue result: NO")
    print()
    print("=" * 88)
    print("Paper 5 05f3b final-gate scientific aggregation: FROZEN")
    print("=" * 88)
    print(f"Primary decisions: {relative_path(OUTPUT_PRIMARY_DECISIONS)}")
    print(f"Mandatory reporting: {relative_path(OUTPUT_MANDATORY_REPORTING)}")
    print(f"Scenario summary: {relative_path(OUTPUT_SCENARIO_SUMMARY)}")
    print(f"Result: {relative_path(OUTPUT_RESULT)}")
    print(f"Failure route: {relative_path(OUTPUT_FAILURE_CONSEQUENCE)}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("Next scientific stage is determined by the frozen 05a2 failure-consequence route.")
    print("=" * 88)


if __name__ == "__main__":
    main()
