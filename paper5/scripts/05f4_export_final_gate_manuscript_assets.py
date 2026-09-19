#!/usr/bin/env python
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


SCRIPT_VERSION = "05f4-export-final-gate-manuscript-assets-v1-read-only-no-cli"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

SOURCE_DIR = PAPER5_ROOT / "method_contract" / "05f3e"
SOURCE_LOCK = SOURCE_DIR / "final_gate_results_lock.json"
SOURCE_RESULT = SOURCE_DIR / "final_gate_result.json"
SOURCE_DECISIONS = SOURCE_DIR / "final_gate_primary_decisions.csv"
SOURCE_MANDATORY = SOURCE_DIR / "final_gate_mandatory_reporting.csv"
SOURCE_OPERATING = SOURCE_DIR / "final_gate_operating_metric_summary.csv"
SOURCE_SCENARIO = SOURCE_DIR / "final_gate_scenario_summary.csv"
SOURCE_TREND = SOURCE_DIR / "d01_independent_null_sample_size_trend.json"

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05f4"
OUTPUT_GATE_SUMMARY = OUTPUT_DIR / "manuscript_final_gate_summary.csv"
OUTPUT_D01_MECHANISM = OUTPUT_DIR / "d01_mechanism_breakdown.csv"
OUTPUT_D01_PER_N = OUTPUT_DIR / "d01_independent_null_per_n.csv"
OUTPUT_D01_TREND = OUTPUT_DIR / "d01_trend_manuscript_reporting.json"
OUTPUT_EVIDENCE = OUTPUT_DIR / "effective_evidence_structure.csv"
OUTPUT_RECOVERY_AUDIT = OUTPUT_DIR / "postunblinding_recovery_audit.csv"
OUTPUT_D05_CONTEXT = OUTPUT_DIR / "d05_reporting_context.json"
OUTPUT_SCENARIO_COPY = OUTPUT_DIR / "scenario_operating_characteristics_locked_copy.csv"
OUTPUT_SCENARIO_COLUMNS = OUTPUT_DIR / "scenario_operating_characteristics_columns.json"
OUTPUT_MACROS = OUTPUT_DIR / "final_gate_numbers.tex"
OUTPUT_README = OUTPUT_DIR / "05f4_README.txt"
OUTPUT_LOCK = OUTPUT_DIR / "05f4_manuscript_assets_lock.json"

FIG_D01_MECHANISM_PNG = OUTPUT_DIR / "fig_d01_mechanism_breakdown.png"
FIG_D01_MECHANISM_PDF = OUTPUT_DIR / "fig_d01_mechanism_breakdown.pdf"
FIG_D01_PER_N_PNG = OUTPUT_DIR / "fig_d01_independent_by_n.png"
FIG_D01_PER_N_PDF = OUTPUT_DIR / "fig_d01_independent_by_n.pdf"

EXPECTED_SOURCE_STATUS = "FROZEN_FINAL_GATE_SCIENTIFIC_RESULTS_NO_RETUNING"
EXPECTED_CORE_STATUS = "CORE_PASS"
EXPECTED_SOURCE_SCRIPT_VERSION = (
    "05f3e-aggregate-final-gate-results-cpu-v3-postunblinding-schema-recovery-no-cli"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
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
                json_safe(value),
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


def write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".tmp_", suffix=path.suffix or ".txt", dir=str(path.parent)
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(text, encoding="utf-8")
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
        table.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required locked input missing: {path}")


def verify_source_lock() -> dict[str, Any]:
    required = [
        SOURCE_LOCK,
        SOURCE_RESULT,
        SOURCE_DECISIONS,
        SOURCE_MANDATORY,
        SOURCE_OPERATING,
        SOURCE_SCENARIO,
        SOURCE_TREND,
    ]
    for path in required:
        require_file(path)

    lock = load_json(SOURCE_LOCK)

    if str(lock.get("status", "")).strip() != EXPECTED_SOURCE_STATUS:
        raise RuntimeError("05f3e source lock has unexpected status.")
    if str(lock.get("core_status", "")).strip() != EXPECTED_CORE_STATUS:
        raise RuntimeError("05f3e source lock is not CORE_PASS.")
    if str(lock.get("script_version", "")).strip() != EXPECTED_SOURCE_SCRIPT_VERSION:
        raise RuntimeError("05f3e source lock script version mismatch.")

    generated = lock.get("generated_files")
    if not isinstance(generated, dict) or not generated:
        raise RuntimeError("05f3e lock has no generated_files manifest.")

    for rel, record in generated.items():
        path = PROJECT_ROOT / Path(rel)
        if not path.is_file():
            raise FileNotFoundError(f"Locked 05f3e artifact missing: {path}")
        expected_hash = str(record.get("sha256", "")).lower()
        expected_size = int(record.get("size_bytes", -1))
        if sha256_file(path).lower() != expected_hash:
            raise RuntimeError(f"Locked 05f3e artifact hash mismatch: {path}")
        if int(path.stat().st_size) != expected_size:
            raise RuntimeError(f"Locked 05f3e artifact size mismatch: {path}")

    if relative_path(SOURCE_RESULT) not in generated:
        raise RuntimeError("05f3e lock does not cover final_gate_result.json.")

    return lock


def get_decision_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = result.get("formal_decisions")
    if not isinstance(rows, list):
        raise RuntimeError("final_gate_result.json lacks formal_decisions.")

    out = {}
    for row in rows:
        decision_id = str(row.get("decision_id", "")).strip()
        if not decision_id:
            raise RuntimeError("Formal decision lacks decision_id.")
        out[decision_id] = row

    expected = {
        "D01_NULL_SPECIFICITY",
        "D02_STANDARD_SENSITIVITY",
        "D03_STANDARD_FNR",
        "D04_ROTATIONAL_SUBSPACE",
        "D05_VIEW_SUPPORT",
        "D06_BATCH_SPECIFICITY",
        "D07_NO_POSTHOC_RETUNING",
    }
    if set(out) != expected:
        raise RuntimeError(
            "Unexpected formal-decision set: "
            + ", ".join(sorted(set(out) ^ expected))
        )
    return out


def build_gate_summary(result: dict[str, Any]) -> pd.DataFrame:
    evidence_notes = {
        "D01_NULL_SPECIFICITY": (
            "Primary core specificity evidence; pooled null-bank decision with "
            "mandatory mechanism-specific reporting."
        ),
        "D02_STANDARD_SENSITIVITY": (
            "Primary core sensitivity evidence at the frozen standard operating point."
        ),
        "D03_STANDARD_FNR": (
            "Algebraic mirror of D02 on the identical 400 opportunities; "
            "not independent evidence."
        ),
        "D04_ROTATIONAL_SUBSPACE": (
            "Narrow rotational-subspace gate; continuous subspace-recovery quantity."
        ),
        "D05_VIEW_SUPPORT": (
            "Narrow mixed-support gate; conditional on retained, biological-truth-matched "
            "factors. Failed frozen threshold; view-support labels remain exploratory."
        ),
        "D06_BATCH_SPECIFICITY": (
            "Narrow technical-confound specificity gate."
        ),
        "D07_NO_POSTHOC_RETUNING": (
            "Process-integrity gate; not a performance observation."
        ),
    }

    rows = []
    for row in result["formal_decisions"]:
        decision_id = row["decision_id"]
        rows.append(
            {
                "decision_id": decision_id,
                "metric": row.get("metric"),
                "numerator": row.get("numerator"),
                "denominator": row.get("denominator"),
                "point_estimate": row.get("point_estimate"),
                "ci95_low": row.get("ci95_low"),
                "ci95_high": row.get("ci95_high"),
                "threshold": row.get("threshold"),
                "pass_rule": row.get("pass_rule"),
                "PASS": row.get("PASS"),
                "decision_quantity": row.get("decision_quantity"),
                "independence_note": row.get("independence_note"),
                "manuscript_evidence_note": evidence_notes[decision_id],
            }
        )
    return pd.DataFrame(rows)


def mandatory_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = result.get("mandatory_D01_reporting")
    if not isinstance(rows, list):
        raise RuntimeError("final_gate_result.json lacks mandatory_D01_reporting.")
    return {str(row["report_id"]): row for row in rows}


def build_d01_mechanism_breakdown(result: dict[str, Any]) -> pd.DataFrame:
    reports = mandatory_map(result)
    wanted = [
        ("R01_D01_PRIMARY_POOLED", "pooled_frozen_null_bank", "primary_decision"),
        ("R02_D01_INDEPENDENT_NULL_POOLED", "independent_null", "mandatory_secondary"),
        ("R03_D01_CORRELATED_PRIVATE", "correlated_private_null", "mandatory_secondary"),
    ]

    rows = []
    for report_id, label, role in wanted:
        if report_id not in reports:
            raise RuntimeError(f"Missing mandatory D01 report: {report_id}")
        r = reports[report_id]
        rows.append(
            {
                "report_id": report_id,
                "mechanism": label,
                "role": role,
                "numerator": int(r["numerator"]),
                "denominator": int(r["denominator"]),
                "point_estimate": float(r["point_estimate"]),
                "ci95_low": float(r["ci95_low"]),
                "ci95_high": float(r["ci95_high"]),
                "formal_gate_consequence": r.get("formal_gate_consequence"),
                "composition_note": (
                    "Frozen pooled endpoint; do not reweight post-unblinding."
                    if report_id == "R01_D01_PRIMARY_POOLED"
                    else "Mandatory prespecified mechanism-specific reporting."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_d01_per_n(result: dict[str, Any]) -> pd.DataFrame:
    reports = mandatory_map(result)
    rows = []
    prefix = "R_D01_INDEPENDENT_N"

    for report_id, r in reports.items():
        if not report_id.startswith(prefix):
            continue
        n = int(report_id[len(prefix):])
        rows.append(
            {
                "n_samples": n,
                "numerator": int(r["numerator"]),
                "denominator": int(r["denominator"]),
                "point_estimate": float(r["point_estimate"]),
                "ci95_low": float(r["ci95_low"]),
                "ci95_high": float(r["ci95_high"]),
                "role": r.get("role"),
            }
        )

    table = pd.DataFrame(rows).sort_values("n_samples").reset_index(drop=True)
    if table["n_samples"].tolist() != [50, 62, 75, 100]:
        raise RuntimeError("Independent-null per-n reporting is incomplete.")
    return table


def build_d01_trend_report(
    result: dict[str, Any],
    per_n: pd.DataFrame,
) -> dict[str, Any]:
    raw = result.get("D01_independent_null_sample_size_trend")
    if not isinstance(raw, dict):
        raise RuntimeError("Missing D01 independent-null sample-size trend.")

    event_count = int(raw.get("event_count", -1))
    converged = bool(raw.get("converged", False))
    all_cells_zero = bool((per_n["numerator"] == 0).all())

    if event_count == 0 and all_cells_zero:
        status = "NON_ESTIMABLE_ZERO_EVENTS"
        reason = (
            "The prespecified unpenalized logistic sample-size trend is not "
            "estimable in a meaningful way because there were zero false-acceptance "
            "events in all 800 independent-null datasets and in every prespecified "
            "sample-size cell. Do not report the raw OR or its odds-scale Wald CI."
        )
        report_or = None
        report_ci_low = None
        report_ci_high = None
    elif not converged:
        status = "NON_ESTIMABLE_NONCONVERGENCE"
        reason = (
            "The prespecified unpenalized logistic trend did not converge. "
            "Do not report the raw OR or its odds-scale Wald CI as an inferential result."
        )
        report_or = None
        report_ci_low = None
        report_ci_high = None
    else:
        status = "ESTIMABLE_AS_PRESPECIFIED"
        reason = "Prespecified model converged."
        report_or = raw.get("odds_ratio_per_25")
        report_ci_low = raw.get("ci95_low")
        report_ci_high = raw.get("ci95_high")

    return {
        "reporting_status": status,
        "reason": reason,
        "n_datasets": int(raw.get("n_datasets", len(per_n) * 200)),
        "event_count": event_count,
        "converged": converged,
        "iterations": int(raw.get("iterations", 0)),
        "manuscript_odds_ratio_per_25": report_or,
        "manuscript_ci95_low": report_ci_low,
        "manuscript_ci95_high": report_ci_high,
        "per_n_event_counts": [
            {
                "n_samples": int(row.n_samples),
                "events": int(row.numerator),
                "datasets": int(row.denominator),
            }
            for row in per_n.itertuples(index=False)
        ],
        "raw_prespecified_model_record_source": relative_path(SOURCE_TREND),
        "raw_record_retained_for_audit": True,
        "posthoc_replacement_model_fitted": False,
    }


def build_effective_evidence_structure() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "evidence_family": "E1_CORE_NULL_SPECIFICITY",
                "decision_ids": "D01",
                "role": "core_performance_evidence",
                "independence_or_conditioning": (
                    "Primary pooled null decision; mandatory independent-null and "
                    "correlated-private mechanism-specific estimates are descriptive "
                    "components of the same family."
                ),
            },
            {
                "evidence_family": "E2_CORE_STANDARD_RECOVERY",
                "decision_ids": "D02;D03",
                "role": "core_performance_evidence",
                "independence_or_conditioning": (
                    "D02 and D03 use the same 400 true-factor opportunities; D03 is "
                    "the algebraic complement of D02 and is not independent evidence."
                ),
            },
            {
                "evidence_family": "E3_ROTATIONAL_SUBSPACE",
                "decision_ids": "D04",
                "role": "narrow_performance_evidence",
                "independence_or_conditioning": (
                    "Rotational-subspace stress gate using a continuous "
                    "subspace-recovery quantity."
                ),
            },
            {
                "evidence_family": "E4_VIEW_SUPPORT",
                "decision_ids": "D05",
                "role": "narrow_conditional_performance_evidence",
                "independence_or_conditioning": (
                    "Conditional on factors that are both retained by the stability "
                    "gate and matched to biological truth; failed the frozen utility threshold."
                ),
            },
            {
                "evidence_family": "E5_TECHNICAL_CONFOUND",
                "decision_ids": "D06",
                "role": "narrow_performance_evidence",
                "independence_or_conditioning": (
                    "Technical-confound specificity stress gate."
                ),
            },
            {
                "evidence_family": "E6_PROCESS_INTEGRITY",
                "decision_ids": "D07",
                "role": "process_audit_not_performance_evidence",
                "independence_or_conditioning": (
                    "Confirms no post-hoc retuning; should not be counted as a "
                    "separate performance result."
                ),
            },
        ]
    )


def build_recovery_audit() -> pd.DataFrame:
    specs = [
        (
            "05f3b",
            PAPER5_ROOT / "scripts" / "05f3b_aggregate_final_gate_results_cpu_preunblinding_locked.py",
            "PREUNBLINDING_LOCKED_AGGREGATOR",
            (
                "Stopped after verified reading of 3,100 shards because exponentiation "
                "of the descriptive D01 logistic Wald upper bound overflowed."
            ),
            "No threshold, selector, denominator, or formal decision was changed.",
        ),
        (
            "05f3c",
            PAPER5_ROOT / "scripts" / "05f3c_aggregate_final_gate_results_cpu_postunblinding_numeric_recovery.py",
            "POSTUNBLINDING_NUMERIC_RECOVERY_V1",
            (
                "Stopped at provenance verification because the recovery comparison "
                "treated equivalent SHA-256 strings with different letter case as unequal."
            ),
            "No scientific payload was opened by this execution.",
        ),
        (
            "05f3d",
            PAPER5_ROOT / "scripts" / "05f3d_aggregate_final_gate_results_cpu_postunblinding_numeric_recovery_v2.py",
            "POSTUNBLINDING_NUMERIC_RECOVERY_V2",
            (
                "Passed provenance checks and numerical reporting recovery, read all "
                "3,100 shards, then stopped on a dataset-level versus factor-level schema "
                "mismatch for the descriptive private-to-shared leakage metric."
            ),
            "Formal gate definitions remained unchanged.",
        ),
        (
            "05f3e",
            PAPER5_ROOT / "scripts" / "05f3e_aggregate_final_gate_results_cpu_postunblinding_schema_recovery_v3.py",
            "POSTUNBLINDING_NUMERIC_PLUS_SCHEMA_RECOVERY_V3",
            (
                "Deterministically propagated the already-stored dataset-level "
                "private_truth_audit_available flag to factor rows; all scientific "
                "artifacts and the final results lock were written. A later "
                "console-formatting statement failed on int(NaN) for a non-count gate."
            ),
            "Locked scientific outputs are the authoritative final-gate results.",
        ),
    ]

    rows = []
    for stage, path, role, execution_note, scientific_scope in specs:
        rows.append(
            {
                "stage": stage,
                "script_path": relative_path(path) if path.exists() else path.as_posix(),
                "script_exists": bool(path.is_file()),
                "script_sha256": sha256_file(path) if path.is_file() else None,
                "role": role,
                "execution_note_source": "observed_terminal_log_posthoc_audit_note",
                "execution_note": execution_note,
                "scientific_scope_note": scientific_scope,
            }
        )
    return pd.DataFrame(rows)


def build_d05_context(
    decisions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    d05 = decisions["D05_VIEW_SUPPORT"]
    return {
        "decision_id": "D05_VIEW_SUPPORT",
        "point_estimate": d05.get("point_estimate"),
        "ci95_low": d05.get("ci95_low"),
        "ci95_high": d05.get("ci95_high"),
        "numerator": d05.get("numerator"),
        "denominator": d05.get("denominator"),
        "threshold": d05.get("threshold"),
        "PASS": d05.get("PASS"),
        "frozen_consequence": (
            "View-support labels remain exploratory and cannot support primary "
            "shared-versus-partial claims."
        ),
        "chance_baseline_status": "NOT_DEFINED_OR_COMPUTED_IN_05F4",
        "chance_baseline_reason": (
            "The locked 05f3e summary does not define a chance model for the "
            "conditional view-support classification task. 05f4 is a read-only "
            "manuscript-packaging stage and does not introduce a new post-unblinding "
            "chance estimand. A separately versioned exploratory analysis may compute "
            "a baseline from the frozen mixed-support truth-class distribution if that "
            "estimand is specified before inspecting its result."
        ),
    }


def pct(x: Any, digits: int = 2) -> str:
    if x is None:
        return ""
    return f"{100.0 * float(x):.{digits}f}"


def build_tex_macros(
    decisions: dict[str, dict[str, Any]],
    mech: pd.DataFrame,
) -> str:
    d01 = decisions["D01_NULL_SPECIFICITY"]
    d02 = decisions["D02_STANDARD_SENSITIVITY"]
    d03 = decisions["D03_STANDARD_FNR"]
    d04 = decisions["D04_ROTATIONAL_SUBSPACE"]
    d05 = decisions["D05_VIEW_SUPPORT"]
    d06 = decisions["D06_BATCH_SPECIFICITY"]

    mech_idx = mech.set_index("mechanism")
    independent = mech_idx.loc["independent_null"]
    correlated = mech_idx.loc["correlated_private_null"]

    return "\n".join(
        [
            "% Auto-generated by 05f4. Do not edit scientific values by hand.",
            rf"\newcommand{{\DZeroOnePointPct}}{{{pct(d01['point_estimate'])}\%}}",
            rf"\newcommand{{\DZeroOneCILowPct}}{{{pct(d01['ci95_low'])}\%}}",
            rf"\newcommand{{\DZeroOneCIHighPct}}{{{pct(d01['ci95_high'])}\%}}",
            rf"\newcommand{{\DZeroTwoPointPct}}{{{pct(d02['point_estimate'])}\%}}",
            rf"\newcommand{{\DZeroTwoCILowPct}}{{{pct(d02['ci95_low'])}\%}}",
            rf"\newcommand{{\DZeroTwoCIHighPct}}{{{pct(d02['ci95_high'])}\%}}",
            rf"\newcommand{{\DZeroThreePointPct}}{{{pct(d03['point_estimate'])}\%}}",
            rf"\newcommand{{\DZeroFourPoint}}{{{float(d04['point_estimate']):.3f}}}",
            rf"\newcommand{{\DZeroFourCILow}}{{{float(d04['ci95_low']):.3f}}}",
            rf"\newcommand{{\DZeroFourCIHigh}}{{{float(d04['ci95_high']):.3f}}}",
            rf"\newcommand{{\DZeroFivePointPct}}{{{pct(d05['point_estimate'], 1)}\%}}",
            rf"\newcommand{{\DZeroFiveCILowPct}}{{{pct(d05['ci95_low'], 1)}\%}}",
            rf"\newcommand{{\DZeroFiveCIHighPct}}{{{pct(d05['ci95_high'], 1)}\%}}",
            rf"\newcommand{{\DZeroSixPointPct}}{{{pct(d06['point_estimate'])}\%}}",
            rf"\newcommand{{\IndependentNullPointPct}}{{{pct(independent['point_estimate'], 1)}\%}}",
            rf"\newcommand{{\CorrelatedPrivatePointPct}}{{{pct(correlated['point_estimate'], 1)}\%}}",
            "",
        ]
    )


def create_d01_figures(
    mech: pd.DataFrame,
    per_n: pd.DataFrame,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        warning = (
            "Matplotlib was unavailable; figure generation was skipped.\n"
            f"Import error: {type(exc).__name__}: {exc}\n"
        )
        path = OUTPUT_DIR / "figure_generation_skipped.txt"
        write_text_atomic(warning, path)
        return [path]

    created: list[Path] = []

    plot_mech = mech[
        mech["mechanism"].isin(["independent_null", "correlated_private_null"])
    ].copy()
    labels = ["Independent null", "Correlated-private null"]
    y = 100.0 * plot_mech["point_estimate"].to_numpy(dtype=float)
    low = 100.0 * plot_mech["ci95_low"].to_numpy(dtype=float)
    high = 100.0 * plot_mech["ci95_high"].to_numpy(dtype=float)
    yerr = np.vstack([y - low, high - y])

    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, y)
    ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=4)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Dataset-level false biological acceptance (%)")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(FIG_D01_MECHANISM_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(FIG_D01_MECHANISM_PDF, bbox_inches="tight")
    plt.close(fig)
    created.extend([FIG_D01_MECHANISM_PNG, FIG_D01_MECHANISM_PDF])

    y = 100.0 * per_n["point_estimate"].to_numpy(dtype=float)
    low = 100.0 * per_n["ci95_low"].to_numpy(dtype=float)
    high = 100.0 * per_n["ci95_high"].to_numpy(dtype=float)
    yerr = np.vstack([y - low, high - y])

    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    x = per_n["n_samples"].to_numpy(dtype=int)
    ax.errorbar(x, y, yerr=yerr, marker="o", capsize=4)
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Independent-null false acceptance (%)")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(FIG_D01_PER_N_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(FIG_D01_PER_N_PDF, bbox_inches="tight")
    plt.close(fig)
    created.extend([FIG_D01_PER_N_PNG, FIG_D01_PER_N_PDF])

    return created


def verify_internal_consistency(
    result: dict[str, Any],
    decisions_csv: pd.DataFrame,
    mandatory_csv: pd.DataFrame,
) -> None:
    decision_map = get_decision_map(result)

    csv_ids = set(decisions_csv["decision_id"].astype(str))
    if csv_ids != set(decision_map):
        raise RuntimeError("Decision IDs differ between locked JSON and CSV.")

    for decision_id, row in decision_map.items():
        csv_row = decisions_csv.loc[
            decisions_csv["decision_id"].astype(str).eq(decision_id)
        ].iloc[0]

        for field in ["point_estimate", "ci95_low", "ci95_high", "threshold"]:
            j = row.get(field)
            c = csv_row.get(field)

            if pd.isna(c):
                c = None
            if j is None and c is None:
                continue
            if (
                j is None
                or c is None
                or not math.isclose(
                    float(j),
                    float(c),
                    rel_tol=0,
                    abs_tol=1e-12,
                )
            ):
                raise RuntimeError(
                    f"Locked JSON/CSV mismatch for {decision_id} field {field}."
                )

    json_reports = mandatory_map(result)
    csv_reports = set(mandatory_csv["report_id"].astype(str))
    if csv_reports != set(json_reports):
        raise RuntimeError(
            "Mandatory D01 report IDs differ between locked JSON and CSV."
        )


def main() -> None:
    print("=" * 88)
    print("Paper 5 - export manuscript assets from locked 05f3e final-gate results")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Source lock: {SOURCE_LOCK}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("GPU computation: NO")
    print("Model fitting: NO")
    print("Scientific payload modification: NO")
    print("Threshold/selector/denominator changes: FORBIDDEN")
    print()

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(
            f"05f4 output directory already exists and is non-empty: {OUTPUT_DIR}. "
            "Do not overwrite versioned manuscript assets."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    verify_source_lock()
    print("05f3e final scientific-results lock and all covered artifacts: VERIFIED")

    result = load_json(SOURCE_RESULT)
    decisions_csv = pd.read_csv(SOURCE_DECISIONS)
    mandatory_csv = pd.read_csv(SOURCE_MANDATORY)
    scenario = pd.read_csv(SOURCE_SCENARIO)
    pd.read_csv(SOURCE_OPERATING)

    if str(result.get("status", "")).strip() != EXPECTED_SOURCE_STATUS:
        raise RuntimeError("final_gate_result.json status mismatch.")
    if str(result.get("script_version", "")).strip() != EXPECTED_SOURCE_SCRIPT_VERSION:
        raise RuntimeError("final_gate_result.json script version mismatch.")
    if result.get("failure_policy_route", {}).get("core_status") != EXPECTED_CORE_STATUS:
        raise RuntimeError("final_gate_result.json is not CORE_PASS.")

    verify_internal_consistency(result, decisions_csv, mandatory_csv)
    print("Locked JSON/CSV decision consistency: VERIFIED")

    decisions = get_decision_map(result)
    gate_summary = build_gate_summary(result)
    mechanism = build_d01_mechanism_breakdown(result)
    per_n = build_d01_per_n(result)
    trend_report = build_d01_trend_report(result, per_n)
    evidence = build_effective_evidence_structure()
    recovery = build_recovery_audit()
    d05_context = build_d05_context(decisions)

    write_csv_atomic(gate_summary, OUTPUT_GATE_SUMMARY)
    write_csv_atomic(mechanism, OUTPUT_D01_MECHANISM)
    write_csv_atomic(per_n, OUTPUT_D01_PER_N)
    write_json_atomic(trend_report, OUTPUT_D01_TREND)
    write_csv_atomic(evidence, OUTPUT_EVIDENCE)
    write_csv_atomic(recovery, OUTPUT_RECOVERY_AUDIT)
    write_json_atomic(d05_context, OUTPUT_D05_CONTEXT)

    write_csv_atomic(scenario, OUTPUT_SCENARIO_COPY)
    write_json_atomic(
        {
            "source": relative_path(SOURCE_SCENARIO),
            "source_sha256": sha256_file(SOURCE_SCENARIO),
            "n_rows": int(scenario.shape[0]),
            "columns": [str(c) for c in scenario.columns],
            "note": (
                "Exact read-only copy of the locked 05f3e scenario summary. "
                "No new PASS/FAIL rule is defined in 05f4."
            ),
        },
        OUTPUT_SCENARIO_COLUMNS,
    )

    write_text_atomic(build_tex_macros(decisions, mechanism), OUTPUT_MACROS)

    figure_paths = create_d01_figures(mechanism, per_n)

    d01_ind = mechanism.loc[
        mechanism["mechanism"].eq("independent_null")
    ].iloc[0]
    d01_corr = mechanism.loc[
        mechanism["mechanism"].eq("correlated_private_null")
    ].iloc[0]
    d05 = decisions["D05_VIEW_SUPPORT"]

    readme = f"""Paper 5 05f4 — manuscript asset export from locked 05f3e
================================================================

Script version
--------------
{SCRIPT_VERSION}

Scope
-----
05f4 is a read-only packaging stage. It verifies the frozen 05f3e results lock
and every artifact covered by that lock, checks JSON/CSV consistency, and exports
manuscript-facing tables, figures, and LaTeX macros.

05f4 performs:
- no GPU computation;
- no model fitting;
- no threshold or selector changes;
- no denominator changes;
- no reweighting of the frozen D01 null bank;
- no post-hoc replacement of the non-estimable D01 logistic trend;
- no new D05 chance model.

Key locked observations copied for manuscript use
-------------------------------------------------
D01 pooled false acceptance: {100*float(decisions["D01_NULL_SPECIFICITY"]["point_estimate"]):.2f}%
D01 independent-null false acceptance: {100*float(d01_ind["point_estimate"]):.2f}%
D01 correlated-private false acceptance: {100*float(d01_corr["point_estimate"]):.2f}%
D02 standard sensitivity: {100*float(decisions["D02_STANDARD_SENSITIVITY"]["point_estimate"]):.2f}%
D03 standard FNR: {100*float(decisions["D03_STANDARD_FNR"]["point_estimate"]):.2f}%
D04 rotational-subspace recovery: {float(decisions["D04_ROTATIONAL_SUBSPACE"]["point_estimate"]):.6f}
D05 view-support accuracy: {100*float(d05["point_estimate"]):.2f}% (frozen threshold {100*float(d05["threshold"]):.0f}%; FAIL)
D06 batch false biological acceptance: {100*float(decisions["D06_BATCH_SPECIFICITY"]["point_estimate"]):.2f}%

D01 sample-size trend
---------------------
Manuscript status: {trend_report["reporting_status"]}
Events: {trend_report["event_count"]}/{trend_report["n_datasets"]}
The raw prespecified model record remains preserved in 05f3e for audit, but 05f4
does not export its OR≈1 or huge odds-scale Wald interval as a manuscript result
when the model is non-estimable due to zero events.

D05 chance baseline
-------------------
05f4 intentionally does not invent a post-unblinding chance model. If a D05 chance
baseline is desired, define its estimand first and compute it in a separately
versioned exploratory script from frozen mixed-support truth labels.

Recovery audit
--------------
postunblinding_recovery_audit.csv records the observed 05f3b→05f3e recovery chain.
Execution-note text is explicitly labeled as a post-hoc audit note from the
observed terminal logs; script hashes are computed directly from the current files.

Authoritative source
--------------------
{relative_path(SOURCE_LOCK)}
SHA256: {sha256_file(SOURCE_LOCK)}
"""
    write_text_atomic(readme, OUTPUT_README)

    generated_paths = [
        OUTPUT_GATE_SUMMARY,
        OUTPUT_D01_MECHANISM,
        OUTPUT_D01_PER_N,
        OUTPUT_D01_TREND,
        OUTPUT_EVIDENCE,
        OUTPUT_RECOVERY_AUDIT,
        OUTPUT_D05_CONTEXT,
        OUTPUT_SCENARIO_COPY,
        OUTPUT_SCENARIO_COLUMNS,
        OUTPUT_MACROS,
        OUTPUT_README,
        *figure_paths,
    ]

    asset_lock = {
        "script_version": SCRIPT_VERSION,
        "script_path": relative_path(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "created_at_utc": utc_now_iso(),
        "status": "READ_ONLY_MANUSCRIPT_ASSET_EXPORT_FROM_LOCKED_05F3E",
        "source_05f3e_lock": file_record(SOURCE_LOCK),
        "source_05f3e_result": file_record(SOURCE_RESULT),
        "source_core_status": EXPECTED_CORE_STATUS,
        "scientific_mutation": False,
        "new_formal_decisions": False,
        "d01_reweighting": False,
        "d01_replacement_trend_model": False,
        "d05_new_chance_estimand": False,
        "generated_files": {
            relative_path(path): file_record(path)
            for path in generated_paths
        },
        "rewrite_policy": (
            "05f4 assets are derived read-only from the locked 05f3e results. "
            "If manuscript-facing formatting changes are needed later, create a new "
            "versioned export rather than altering scientific values."
        ),
    }
    write_json_atomic(asset_lock, OUTPUT_LOCK)

    print()
    print("05f4 manuscript assets: WRITTEN")
    print(f"Gate summary: {OUTPUT_GATE_SUMMARY}")
    print(f"D01 mechanism table: {OUTPUT_D01_MECHANISM}")
    print(f"D01 per-n table: {OUTPUT_D01_PER_N}")
    print(f"Evidence structure: {OUTPUT_EVIDENCE}")
    print(f"Recovery audit: {OUTPUT_RECOVERY_AUDIT}")
    print(f"Scenario locked copy: {OUTPUT_SCENARIO_COPY}")
    print(f"LaTeX macros: {OUTPUT_MACROS}")
    print(f"05f4 lock: {OUTPUT_LOCK}")
    print()
    print("Key read-only manuscript facts:")
    print(
        f"  D01 independent null: {int(d01_ind['numerator'])}/"
        f"{int(d01_ind['denominator'])} "
        f"({100*float(d01_ind['point_estimate']):.1f}%)"
    )
    print(
        f"  D01 correlated-private: {int(d01_corr['numerator'])}/"
        f"{int(d01_corr['denominator'])} "
        f"({100*float(d01_corr['point_estimate']):.1f}%)"
    )
    print(f"  D01 trend: {trend_report['reporting_status']}")
    print(
        f"  D05: {100*float(d05['point_estimate']):.1f}% -> "
        f"{'PASS' if d05['PASS'] else 'FAIL'}"
    )
    print()
    print("No 05f3 scientific artifact was modified.")


if __name__ == "__main__":
    main()
