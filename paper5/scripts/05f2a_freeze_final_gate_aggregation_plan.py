from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_VERSION = "05f2a-freeze-final-gate-aggregation-plan-v1-no-cli"
PLAN_VERSION = "paper5-final-gate-aggregation-plan-v1"
LOCK_STATUS = "FROZEN_FINAL_GATE_AGGREGATION_IMPLEMENTATION_BEFORE_FULL_RUN_RESULTS"

EXPECTED_05A2_STATUS = "FROZEN_BEFORE_FINAL_GATE_MATERIALIZATION"
EXPECTED_05F3_SCRIPT_VERSION = (
    "05f3-aggregate-final-gate-results-cpu-v1-prospectively-locked-no-cli"
)
EXPECTED_05F3_SHA256 = (
    "443b553e2e9e95a9f4d3ac7de5fe7b4f33479d6552264555e3edc276b077d5c2"
)

EXPECTED_FINAL_GATE_SCENARIOS = 24
EXPECTED_FINAL_GATE_DATASETS = 3_100
EXPECTED_STABILITY_RESAMPLES = 100
MAX_ALLOWED_05F2_PROGRESS_ROWS_AT_FREEZE = 3

# These values must agree exactly with frozen 05a2 / 05a1 and with the
# prospectively frozen 05f3 implementation.
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
EXPECTED_D02_DATASETS = 100
EXPECTED_D02_FACTOR_OPPORTUNITIES = 400

EXPECTED_STANDARD_SCENARIO_ID = "G07_IND_N62_S1P00"
EXPECTED_ROTATIONAL_SCENARIO_ID = "G17_ROTATIONAL_N62_S1P00"
EXPECTED_MIXED_SUPPORT_SCENARIO_ID = "G18_MIXED_SUPPORT_N62_S1P00"
EXPECTED_CORRELATED_PRIVATE_SCENARIO_ID = "G19_CORRELATED_PRIVATE_NULL_N62"
EXPECTED_BATCH_NULL_SCENARIO_IDS = (
    "G20_BATCH_SHARED_NULL_N62",
    "G21_BATCH_PARTIAL_NULL_N62",
)

EXPECTED_BOOTSTRAP_REPLICATES = 10_000
EXPECTED_BOOTSTRAP_MASTER_SEED = 20260822


SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

DIR_05A1 = PAPER5_ROOT / "method_contract" / "05a1"
FINAL_GATE_SCENARIOS = DIR_05A1 / "final_gate_calibration_scenarios.csv"
FINAL_GATE_REPLICATES = DIR_05A1 / "final_gate_calibration_replicates.csv"
GATE_METRICS_05A1 = DIR_05A1 / "stability_gate_operating_metric_registry.csv"
GATE_DECISIONS_05A1 = DIR_05A1 / "stability_gate_decision_registry.csv"

DIR_05A2 = PAPER5_ROOT / "method_contract" / "05a2"
AMENDMENT_05A2 = DIR_05A2 / "inference_final_gate_implementation_amendment.json"
EFFECTIVE_METRICS_05A2 = DIR_05A2 / "effective_stability_gate_operating_metric_registry.csv"
EFFECTIVE_DECISIONS_05A2 = DIR_05A2 / "effective_stability_gate_decision_registry.csv"
REPORTING_05A2 = DIR_05A2 / "final_gate_reporting_registry.csv"
FAILURE_POLICY_05A2 = DIR_05A2 / "final_gate_failure_consequence_policy.json"
LOCK_05A2 = DIR_05A2 / "inference_final_gate_implementation_amendment_lock.json"

AGGREGATION_SCRIPT = PAPER5_ROOT / "scripts" / "05f3_aggregate_final_gate_results_cpu.py"

DIR_05F2 = PAPER5_ROOT / "method_contract" / "05f2"
PROGRESS_05F2 = DIR_05F2 / "full_run_progress_manifest.csv"
LOCK_05F2 = DIR_05F2 / "full_run_completion_lock.json"

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05f2a"
OUTPUT_PLAN = OUTPUT_DIR / "final_gate_aggregation_plan.json"
OUTPUT_SCRIPT_SNAPSHOT = OUTPUT_DIR / "05f3_aggregation_script_snapshot.json"
OUTPUT_README = OUTPUT_DIR / "05f2a_README.txt"
OUTPUT_LOCK = OUTPUT_DIR / "final_gate_aggregation_plan_lock.json"


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
    if isinstance(value, bool):
        return value
    text = clean_text(value).lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"Cannot parse boolean: {value!r}")


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


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".tmp_", suffix=".json", dir=str(path.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(
                value,
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
    fd, temporary_name = tempfile.mkstemp(
        prefix=".tmp_", suffix=".txt", dir=str(path.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def extract_literal_constants(path: Path) -> dict[str, Any]:
    """Read top-level literal constants without importing/running 05f3."""
    if not path.is_file():
        raise FileNotFoundError(
            "The exact 05f3 aggregation script must be placed in paper5/scripts "
            f"before 05f2a is run. Missing: {path}"
        )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    constants: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            constants[target.id] = ast.literal_eval(node.value)
        except Exception:
            continue
    return constants


# -----------------------------------------------------------------------------
# Frozen upstream verification
# -----------------------------------------------------------------------------

def verify_05a2() -> dict[str, Any]:
    required = [
        FINAL_GATE_SCENARIOS,
        FINAL_GATE_REPLICATES,
        GATE_METRICS_05A1,
        GATE_DECISIONS_05A1,
        AMENDMENT_05A2,
        EFFECTIVE_METRICS_05A2,
        EFFECTIVE_DECISIONS_05A2,
        REPORTING_05A2,
        FAILURE_POLICY_05A2,
        LOCK_05A2,
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
    if not isinstance(generated, dict) or not generated:
        raise RuntimeError("05a2 lock lacks generated_files registry.")
    for relative_name, record in generated.items():
        path = resolve_project_path(relative_name)
        if not path.is_file():
            raise FileNotFoundError(f"05a2 generated file missing: {path}")
        if sha256_file(path) != clean_text(record.get("sha256")):
            raise RuntimeError(f"05a2 generated file hash mismatch: {relative_name}")

    decisions = pd.read_csv(
        EFFECTIVE_DECISIONS_05A2,
        dtype=str,
        low_memory=False,
    ).fillna("")
    reporting = pd.read_csv(
        REPORTING_05A2,
        dtype=str,
        low_memory=False,
    ).fillna("")
    scenarios = pd.read_csv(
        FINAL_GATE_SCENARIOS,
        dtype=str,
        low_memory=False,
    ).fillna("")
    replicates = pd.read_csv(
        FINAL_GATE_REPLICATES,
        dtype=str,
        low_memory=False,
    ).fillna("")

    if scenarios.shape[0] != EXPECTED_FINAL_GATE_SCENARIOS:
        raise RuntimeError("Final-gate scenario registry is not 24 rows.")
    if replicates.shape[0] != EXPECTED_FINAL_GATE_DATASETS:
        raise RuntimeError("Final-gate replicate registry is not 3,100 rows.")

    scenario_ids = set(scenarios["scenario_id"].astype(str))
    required_ids = {
        EXPECTED_STANDARD_SCENARIO_ID,
        EXPECTED_ROTATIONAL_SCENARIO_ID,
        EXPECTED_MIXED_SUPPORT_SCENARIO_ID,
        EXPECTED_CORRELATED_PRIVATE_SCENARIO_ID,
        *EXPECTED_BATCH_NULL_SCENARIO_IDS,
    }
    missing = required_ids - scenario_ids
    if missing:
        raise RuntimeError(
            "Frozen scenario registry lacks IDs required by 05f3: "
            + ", ".join(sorted(missing))
        )

    decision_lookup = decisions.set_index("decision_id", drop=False)
    for decision_id in [
        "D01_NULL_SPECIFICITY",
        "D02_STANDARD_SENSITIVITY",
        "D03_STANDARD_FNR",
        "D04_ROTATIONAL_SUBSPACE",
        "D05_VIEW_SUPPORT",
        "D06_BATCH_SPECIFICITY",
        "D07_NO_POSTHOC_RETUNING",
    ]:
        if decision_id not in decision_lookup.index:
            raise RuntimeError(f"05a2 effective decision missing: {decision_id}")

    d01 = decision_lookup.loc["D01_NULL_SPECIFICITY"]
    if "upper endpoint of two-sided 95% Wilson" not in clean_text(
        d01.get("effective_ci_rule")
    ):
        raise RuntimeError("05a2 D01 Wilson implementation rule changed.")

    d02 = decision_lookup.loc["D02_STANDARD_SENSITIVITY"]
    if "G07 only" not in clean_text(d02.get("effective_aggregation_rule")):
        raise RuntimeError("05a2 D02 no longer fixes G07-only aggregation.")
    if "400" not in clean_text(d02.get("effective_aggregation_rule")):
        raise RuntimeError("05a2 D02 no longer fixes the 400-factor denominator.")

    d03 = decision_lookup.loc["D03_STANDARD_FNR"]
    if "1 - D02" not in clean_text(d03.get("effective_aggregation_rule")):
        raise RuntimeError("05a2 D03 no longer fixes algebraic mirroring of D02.")

    # Mandatory D01 reporting was frozen in 05a2; verify the key denominators.
    reports = reporting.set_index("report_id", drop=False)
    expected_reporting = {
        "R01_D01_PRIMARY_POOLED": EXPECTED_D01_DATASETS,
        "R02_D01_INDEPENDENT_NULL_POOLED": EXPECTED_D01_INDEPENDENT_DATASETS,
        "R03_D01_CORRELATED_PRIVATE": EXPECTED_D01_CORRELATED_PRIVATE_DATASETS,
    }
    for report_id, expected_n in expected_reporting.items():
        if report_id not in reports.index:
            raise RuntimeError(f"05a2 mandatory reporting row missing: {report_id}")
        observed = int(float(reports.loc[report_id, "expected_datasets"]))
        if observed != expected_n:
            raise RuntimeError(
                f"05a2 reporting denominator changed for {report_id}: {observed}."
            )

    return {
        "amendment": amendment,
        "lock": lock,
        "decisions": decisions,
        "reporting": reporting,
        "scenarios": scenarios,
        "replicates": replicates,
    }


def verify_05f3_exact_script() -> dict[str, Any]:
    observed_hash = sha256_file(AGGREGATION_SCRIPT)
    if observed_hash != EXPECTED_05F3_SHA256:
        raise RuntimeError(
            "05f3 aggregation script SHA256 does not match the prospectively reviewed "
            "implementation. Expected:\n"
            f"{EXPECTED_05F3_SHA256}\nObserved:\n{observed_hash}\n"
            "Do not continue with a modified aggregation implementation."
        )

    constants = extract_literal_constants(AGGREGATION_SCRIPT)
    expected_constants = {
        "SCRIPT_VERSION": EXPECTED_05F3_SCRIPT_VERSION,
        "EXPECTED_DATASETS": EXPECTED_FINAL_GATE_DATASETS,
        "EXPECTED_SCENARIOS": EXPECTED_FINAL_GATE_SCENARIOS,
        "EXPECTED_STABILITY_RESAMPLES": EXPECTED_STABILITY_RESAMPLES,
        "D01_LIMIT": D01_LIMIT,
        "D01_WILSON_Z": D01_WILSON_Z,
        "D02_TPR_MIN": D02_TPR_MIN,
        "D03_FNR_MAX": D03_FNR_MAX,
        "D04_SUBSPACE_MIN": D04_SUBSPACE_MIN,
        "D05_VIEW_SUPPORT_MIN": D05_VIEW_SUPPORT_MIN,
        "D06_BATCH_FALSE_ACCEPT_MAX": D06_BATCH_FALSE_ACCEPT_MAX,
        "EXPECTED_D01_DATASETS": EXPECTED_D01_DATASETS,
        "EXPECTED_D01_INDEPENDENT_DATASETS": EXPECTED_D01_INDEPENDENT_DATASETS,
        "EXPECTED_D01_CORRELATED_PRIVATE_DATASETS": (
            EXPECTED_D01_CORRELATED_PRIVATE_DATASETS
        ),
        "EXPECTED_D02_DATASETS": EXPECTED_D02_DATASETS,
        "EXPECTED_D02_FACTOR_OPPORTUNITIES": EXPECTED_D02_FACTOR_OPPORTUNITIES,
        "STANDARD_SCENARIO_ID": EXPECTED_STANDARD_SCENARIO_ID,
        "ROTATIONAL_SCENARIO_ID": EXPECTED_ROTATIONAL_SCENARIO_ID,
        "MIXED_SUPPORT_SCENARIO_ID": EXPECTED_MIXED_SUPPORT_SCENARIO_ID,
        "CORRELATED_PRIVATE_SCENARIO_ID": EXPECTED_CORRELATED_PRIVATE_SCENARIO_ID,
        "BATCH_NULL_SCENARIO_IDS": EXPECTED_BATCH_NULL_SCENARIO_IDS,
        "BOOTSTRAP_REPLICATES": EXPECTED_BOOTSTRAP_REPLICATES,
        "BOOTSTRAP_MASTER_SEED": EXPECTED_BOOTSTRAP_MASTER_SEED,
    }
    mismatches: list[str] = []
    for key, expected in expected_constants.items():
        observed = constants.get(key, "<MISSING>")
        if observed != expected:
            mismatches.append(f"{key}: observed={observed!r}, expected={expected!r}")
    if mismatches:
        raise RuntimeError(
            "05f3 literal-constant audit failed:\n- " + "\n- ".join(mismatches)
        )

    return {
        "sha256": observed_hash,
        "constants": {
            key: constants[key]
            for key in expected_constants
        },
    }


def verify_no_full_run_results_yet() -> dict[str, Any]:
    if LOCK_05F2.exists():
        raise RuntimeError(
            "05f2 completion lock already exists. 05f2a must be frozen before "
            "the full-run scientific result bank is complete."
        )

    progress_rows = 0
    progress_status = "ABSENT"
    if PROGRESS_05F2.exists():
        progress = pd.read_csv(
            PROGRESS_05F2,
            dtype=str,
            low_memory=False,
        ).fillna("")
        progress_rows = int(progress.shape[0])
        progress_status = "PRESENT"
        if progress_rows > MAX_ALLOWED_05F2_PROGRESS_ROWS_AT_FREEZE:
            raise RuntimeError(
                "05f2 has already progressed beyond the three previously frozen runtime-pilot "
                f"datasets ({progress_rows} rows found). Refusing to freeze the scientific "
                "aggregation implementation after new full-run results have accumulated."
            )

    # We deliberately do not inspect any 05f1/05f2 scientific shard here.
    return {
        "05f2_completion_lock_present": False,
        "05f2_progress_manifest_status": progress_status,
        "05f2_progress_rows": progress_rows,
        "maximum_allowed_progress_rows_at_05f2a_freeze": (
            MAX_ALLOWED_05F2_PROGRESS_ROWS_AT_FREEZE
        ),
        "scientific_shards_opened_or_read": False,
        "scientific_metrics_aggregated": False,
        "scientific_PASS_FAIL_evaluated": False,
    }


# -----------------------------------------------------------------------------
# Plan construction
# -----------------------------------------------------------------------------

def build_plan(
    upstream: dict[str, Any],
    aggregation: dict[str, Any],
    timing: dict[str, Any],
) -> dict[str, Any]:
    return {
        "plan_version": PLAN_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": LOCK_STATUS,
        "scientific_timing": timing,
        "aggregation_script": {
            "path": relative_path(AGGREGATION_SCRIPT),
            "sha256": aggregation["sha256"],
            "script_version": EXPECTED_05F3_SCRIPT_VERSION,
            "execution_timing": "only after 05f2 is locked COMPLETE at 3100/3100",
            "script_modification_after_05f2a": "FORBIDDEN_FOR_PRIMARY_AGGREGATION",
        },
        "core_decisions": {
            "D01_NULL_SPECIFICITY": {
                "selector": (
                    "true_biological_shared_factors == 0 and "
                    "true_technical_shared_factors == 0"
                ),
                "expected_datasets": EXPECTED_D01_DATASETS,
                "event": (
                    "dataset retains at least one stable unmatched nontechnical "
                    "factor classified as a false biological acceptance"
                ),
                "point_estimate": "event datasets / 1000",
                "confidence_interval": {
                    "method": "Wilson score interval",
                    "confidence_level": 0.95,
                    "sidedness": "two-sided",
                    "z": D01_WILSON_Z,
                    "continuity_correction": False,
                },
                "PASS_rule": "upper endpoint <= 0.10",
                "threshold": D01_LIMIT,
            },
            "D02_STANDARD_SENSITIVITY": {
                "scenario_id": EXPECTED_STANDARD_SCENARIO_ID,
                "expected_datasets": EXPECTED_D02_DATASETS,
                "aggregation_unit": "pooled true biological factor opportunities",
                "expected_factor_opportunities": EXPECTED_D02_FACTOR_OPPORTUNITIES,
                "numerator": (
                    "true biological factors whose Hungarian-assigned estimated "
                    "factor passes the biological truth-match threshold and the "
                    "frozen stability gate"
                ),
                "PASS_rule": "point estimate >= 0.80",
                "threshold": D02_TPR_MIN,
                "confidence_interval_role": "reported but not decision-determining",
            },
            "D03_STANDARD_FNR": {
                "scenario_id": EXPECTED_STANDARD_SCENARIO_ID,
                "definition": "1 - D02 TPR on the identical 400-factor denominator",
                "independence_status": "algebraic mirror of D02; not independent evidence",
                "PASS_rule": "point estimate <= 0.20",
                "threshold": D03_FNR_MAX,
            },
        },
        "narrow_decisions": {
            "D04_ROTATIONAL_SUBSPACE": {
                "scenario_id": EXPECTED_ROTATIONAL_SCENARIO_ID,
                "metric": (
                    "mean across datasets of reference_shared_subspace_recovery "
                    "(mean canonical correlation / mean cosine of principal angles)"
                ),
                "PASS_rule": "point estimate >= 0.80",
                "threshold": D04_SUBSPACE_MIN,
                "confidence_interval": (
                    "fixed-seed percentile bootstrap of dataset-level means"
                ),
            },
            "D05_VIEW_SUPPORT": {
                "scenario_id": EXPECTED_MIXED_SUPPORT_SCENARIO_ID,
                "metric": (
                    "exact view-support classification accuracy among biological "
                    "factors that are both truth-matched and retained by the frozen stability gate"
                ),
                "PASS_rule": "point estimate >= 0.80",
                "threshold": D05_VIEW_SUPPORT_MIN,
                "confidence_interval": "two-sided 95% Wilson interval",
            },
            "D06_BATCH_SPECIFICITY": {
                "scenario_ids": list(EXPECTED_BATCH_NULL_SCENARIO_IDS),
                "metric": (
                    "per true technical-factor opportunity, false biological acceptance "
                    "when the Hungarian-assigned estimated factor is stability-retained "
                    "but does not satisfy the frozen technical truth-match flag"
                ),
                "PASS_rule": "point estimate <= 0.10",
                "threshold": D06_BATCH_FALSE_ACCEPT_MAX,
                "confidence_interval": "two-sided 95% Wilson interval",
            },
            "D07_NO_POSTHOC_RETUNING": {
                "metric": "process integrity",
                "PASS_rule": (
                    "05f3 script hash and frozen 05a2 thresholds/selectors/denominators "
                    "remain unchanged"
                ),
            },
        },
        "mandatory_D01_reporting": {
            "primary_pooled": {
                "n": EXPECTED_D01_DATASETS,
                "statistics": "numerator/denominator; point; two-sided 95% Wilson CI; PASS/FAIL",
            },
            "independent_null_pooled": {
                "n": EXPECTED_D01_INDEPENDENT_DATASETS,
                "formal_gate_consequence": "NONE",
            },
            "correlated_private_null": {
                "scenario_id": EXPECTED_CORRELATED_PRIVATE_SCENARIO_ID,
                "n": EXPECTED_D01_CORRELATED_PRIVATE_DATASETS,
                "formal_gate_consequence": "NONE",
            },
            "independent_null_by_n": {
                "n_values": [50, 62, 75, 100],
                "datasets_per_n": 200,
                "formal_gate_consequence": "NONE",
            },
            "sample_size_trend": {
                "pool": "800 independent-null datasets only",
                "model": "unpenalized logistic regression via deterministic Newton/IRLS",
                "outcome": "dataset-level false biological acceptance",
                "predictor": "n_samples / 25",
                "reported_effect": "odds ratio per +25 samples with two-sided 95% Wald CI",
                "formal_gate_consequence": "NONE_DESCRIPTIVE_ONLY",
            },
        },
        "continuous_interval_implementation": {
            "bootstrap_replicates": EXPECTED_BOOTSTRAP_REPLICATES,
            "bootstrap_master_seed": EXPECTED_BOOTSTRAP_MASTER_SEED,
            "resampling_unit": "dataset",
            "statistic": "mean",
            "interval": "2.5th and 97.5th percentiles of bootstrap means",
            "use_for_D01_decision": False,
            "use_for_D02_D03_decisions": False,
            "role": "continuous descriptive/narrow-gate confidence intervals only",
        },
        "additional_descriptive_outputs": [
            "24-scenario operating summary",
            "pooled stability-gated false discovery proportion",
            "factor-count bias with fixed-seed bootstrap interval",
            "private-to-shared leakage rate",
            "technical-factor detection sensitivity",
        ],
        "failure_policy": {
            "source": relative_path(FAILURE_POLICY_05A2),
            "source_sha256": sha256_file(FAILURE_POLICY_05A2),
            "rule": (
                "Route D01/D02/D03 result through the already-frozen 05a2 "
                "core_outcomes table; narrow-gate failures use the already-frozen "
                "D04/D05/D06/D07 consequence rules."
            ),
            "criterion_failure_changes_conclusion_not_criterion": True,
            "post_result_retuning": False,
            "TARGET_or_outcome_rescue": False,
        },
        "guardrails": [
            "05f2a does not read any final-gate scientific shard.",
            "05f2a does not compute any scientific metric.",
            "05f2a does not evaluate any final-gate PASS/FAIL result.",
            "05f3 may run only after 05f2 is COMPLETE at 3100/3100.",
            "The exact SHA256-locked 05f3 file must be used for primary aggregation.",
            "No threshold, selector, denominator, CI decision rule or scenario membership may change after results are opened.",
            "Any later alternative analysis is separately versioned and explicitly exploratory.",
        ],
        "source_records": {
            "05a2_amendment": file_record(AMENDMENT_05A2),
            "05a2_effective_decisions": file_record(EFFECTIVE_DECISIONS_05A2),
            "05a2_reporting": file_record(REPORTING_05A2),
            "05a2_failure_policy": file_record(FAILURE_POLICY_05A2),
            "final_gate_scenarios": file_record(FINAL_GATE_SCENARIOS),
            "final_gate_replicates": file_record(FINAL_GATE_REPLICATES),
            "05f3_script": file_record(AGGREGATION_SCRIPT),
        },
    }


def build_readme(plan: dict[str, Any]) -> str:
    return (
        "Paper 5 05f2a — prospective final-gate aggregation implementation freeze\n"
        "=======================================================================\n\n"
        f"Script version: {SCRIPT_VERSION}\n"
        f"Status: {LOCK_STATUS}\n\n"
        "Purpose\n"
        "-------\n"
        "Freeze the exact post-05f2 scientific aggregation implementation before\n"
        "new full-run final-gate results accumulate. This prevents the long GPU run\n"
        "from being followed by post-result changes to selectors, denominators,\n"
        "confidence-interval semantics, narrow-gate metrics, or failure routing.\n\n"
        "Scientific visibility\n"
        "---------------------\n"
        "05f2a reads frozen contracts/registries and the 05f3 source code only.\n"
        "It does not read any final-gate scientific result shard, compute any gate\n"
        "metric, or evaluate PASS/FAIL.\n\n"
        "Execution order\n"
        "---------------\n"
        "1. Freeze 05f2a (this stage).\n"
        "2. Run/resume 05f2 until 3100/3100 sealed shards are complete.\n"
        "3. Run the unchanged SHA256-locked 05f3 aggregation script exactly once.\n"
        "4. Route downstream work through the already-frozen 05a2 failure policy.\n\n"
        f"Frozen 05f3 SHA256: {plan['aggregation_script']['sha256']}\n"
        f"Frozen 05f3 version: {plan['aggregation_script']['script_version']}\n"
        f"Bootstrap replicates: {EXPECTED_BOOTSTRAP_REPLICATES}\n"
        f"Bootstrap master seed: {EXPECTED_BOOTSTRAP_MASTER_SEED}\n"
    )


# -----------------------------------------------------------------------------
# Existing freeze verification
# -----------------------------------------------------------------------------

def verify_existing_freeze() -> None:
    lock = load_json(OUTPUT_LOCK)
    if clean_text(lock.get("status")) != LOCK_STATUS:
        raise RuntimeError("Existing 05f2a lock has unexpected status.")
    if clean_text(lock.get("script_sha256")) != sha256_file(SCRIPT_PATH):
        raise RuntimeError("Current 05f2a script differs from its frozen hash.")
    if clean_text(lock.get("aggregation_script_sha256")) != sha256_file(
        AGGREGATION_SCRIPT
    ):
        raise RuntimeError("Current 05f3 script differs from the 05f2a frozen hash.")

    generated = lock.get("generated_files", {})
    for relative_name, record in generated.items():
        path = resolve_project_path(relative_name)
        if not path.is_file():
            raise FileNotFoundError(f"Frozen 05f2a output missing: {path}")
        if sha256_file(path) != clean_text(record.get("sha256")):
            raise RuntimeError(f"Frozen 05f2a output hash mismatch: {relative_name}")

    print("=" * 88)
    print("Existing Paper 5 05f2a final-gate aggregation plan: VERIFIED")
    print("=" * 88)
    print(f"05f3 script version: {EXPECTED_05F3_SCRIPT_VERSION}")
    print(f"05f3 SHA256: {sha256_file(AGGREGATION_SCRIPT)}")
    print("Scientific final-gate shards read: NO")
    print("Scientific metrics computed: NO")
    print("PASS/FAIL evaluated: NO")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("No files were rewritten.")
    print("=" * 88)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    print("=" * 88)
    print("Paper 5 - freeze final-gate scientific aggregation implementation")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("GPU computation in 05f2a: NO")
    print("Model fitting in 05f2a: NO")
    print("Final-gate scientific shard access in 05f2a: NO")
    print("Scientific metric calculation in 05f2a: NO")
    print("Final-gate PASS/FAIL evaluation in 05f2a: NO")
    print("TARGET/outcome/biology access in 05f2a: NO")
    print("Threshold retuning: FORBIDDEN")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_LOCK.exists():
        verify_existing_freeze()
        return

    upstream = verify_05a2()
    aggregation = verify_05f3_exact_script()
    timing = verify_no_full_run_results_yet()

    print("05a2 frozen decision/report/failure-policy layer: PASS")
    print("Final-gate registry: 24 scenarios / 3,100 datasets / 100 resamples: PASS")
    print("05f3 exact source SHA256/version/constants: PASS")
    print(
        "05f2 full-run progress at aggregation freeze: "
        f"{timing['05f2_progress_rows']} rows"
    )
    print("Final-gate scientific result shards read: NO")
    print("Scientific results used to define aggregation: NO")
    print()

    plan = build_plan(
        upstream=upstream,
        aggregation=aggregation,
        timing=timing,
    )
    snapshot = {
        "snapshot_version": "paper5-05f3-source-snapshot-v1",
        "created_at_utc": utc_now_iso(),
        "status": LOCK_STATUS,
        "aggregation_script": file_record(AGGREGATION_SCRIPT),
        "script_version": EXPECTED_05F3_SCRIPT_VERSION,
        "literal_constant_audit": aggregation["constants"],
        "source_code_executed_in_05f2a": False,
        "scientific_shards_read": False,
        "scientific_results_used": False,
    }

    write_json_atomic(plan, OUTPUT_PLAN)
    write_json_atomic(snapshot, OUTPUT_SCRIPT_SNAPSHOT)
    write_text_atomic(build_readme(plan), OUTPUT_README)

    generated_paths = [
        OUTPUT_PLAN,
        OUTPUT_SCRIPT_SNAPSHOT,
        OUTPUT_README,
    ]
    lock = {
        "plan_version": PLAN_VERSION,
        "script_version": SCRIPT_VERSION,
        "script_path": relative_path(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "created_at_utc": utc_now_iso(),
        "status": LOCK_STATUS,
        "aggregation_script_path": relative_path(AGGREGATION_SCRIPT),
        "aggregation_script_version": EXPECTED_05F3_SCRIPT_VERSION,
        "aggregation_script_sha256": aggregation["sha256"],
        "05a2_amendment_sha256": sha256_file(AMENDMENT_05A2),
        "final_gate_scenarios_sha256": sha256_file(FINAL_GATE_SCENARIOS),
        "final_gate_replicates_sha256": sha256_file(FINAL_GATE_REPLICATES),
        "scientific_state_at_freeze": {
            "05f2_completion_lock_present": False,
            "05f2_progress_rows": timing["05f2_progress_rows"],
            "final_gate_scientific_shards_read": False,
            "scientific_metrics_computed": False,
            "D01_D02_D03_evaluated": False,
            "final_gate_PASS_FAIL_evaluated": False,
            "TARGET_accessed": False,
            "outcomes_accessed": False,
            "biology_annotations_accessed": False,
            "threshold_retuning_performed": False,
        },
        "generated_files": {
            relative_path(path): file_record(path)
            for path in generated_paths
        },
        "rewrite_policy": (
            "This no-argument script never rewrites a frozen 05f2a plan. "
            "Primary final-gate aggregation must use the exact SHA256-locked 05f3 "
            "script. Any later alternative aggregation is exploratory and must be "
            "separately versioned after the primary frozen result."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    print("-" * 88)
    print("Frozen final-gate aggregation implementation")
    print("-" * 88)
    print(f"05f3 version: {EXPECTED_05F3_SCRIPT_VERSION}")
    print(f"05f3 SHA256: {aggregation['sha256']}")
    print("D01: N=1000; two-sided 95% Wilson upper endpoint <= 0.10")
    print("D02: G07 only; 400 pooled true-factor opportunities; TPR >= 0.80")
    print("D03: 1-D02 on same denominator; FNR <= 0.20")
    print("D04: G17 rotational subspace mean >= 0.80")
    print("D05: G18 matched-retained view-support accuracy >= 0.80")
    print("D06: G20+G21 technical false biological acceptance <= 0.10")
    print("Continuous bootstrap: 10,000 fixed-seed dataset resamples")
    print("Post-result aggregation changes: FORBIDDEN")
    print()
    print("=" * 88)
    print("Paper 5 05f2a final-gate aggregation implementation: FROZEN")
    print("=" * 88)
    print(f"Plan: {relative_path(OUTPUT_PLAN)}")
    print(f"05f3 snapshot: {relative_path(OUTPUT_SCRIPT_SNAPSHOT)}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("Next: start/resume 05f2 full GPU run. Do NOT run 05f3 until 05f2 is COMPLETE.")
    print("=" * 88)


if __name__ == "__main__":
    main()
