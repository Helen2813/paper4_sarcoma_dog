#!/usr/bin/env python
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "05x1b-freeze-stability-only-retention-ablation-contract-v2-no-cli"
CONTRACT_VERSION = "paper5-05x1b-postunblinding-exploratory-retention-ablation-v2"
STATUS = "FROZEN_POSTUNBLINDING_EXPLORATORY_ABLATION_BEFORE_RESULT_COMPUTATION"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

CONTRACT_05A = PAPER5_ROOT / "method_contract" / "05a" / "ai_method_contract.json"
SCENARIO_REGISTRY = (
    PAPER5_ROOT / "method_contract" / "05a1" / "final_gate_calibration_scenarios.csv"
)
HELPER = PAPER5_ROOT / "scripts" / "05d2v2_screen_neural_profiles_gpu_syncfree_pathfix.py"

PROGRESS_05F2 = PAPER5_ROOT / "method_contract" / "05f2" / "full_run_progress_manifest.csv"
LOCK_05F2 = PAPER5_ROOT / "method_contract" / "05f2" / "full_run_completion_lock.json"
LOCK_05F3E = PAPER5_ROOT / "method_contract" / "05f3e" / "final_gate_results_lock.json"
LOCK_05F4B = PAPER5_ROOT / "method_contract" / "05f4b" / "05f4b_manuscript_assets_lock.json"
LOCK_05X0 = PAPER5_ROOT / "method_contract" / "05x0" / "05x0_preflight_lock.json"

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05x1b"
OUTPUT_CONTRACT = OUTPUT_DIR / "stability_only_retention_ablation_contract.json"
OUTPUT_README = OUTPUT_DIR / "05x1b_README.txt"
OUTPUT_LOCK = OUTPUT_DIR / "05x1b_contract_lock.json"

CORRELATED_PRIVATE_SCENARIO_ID = "G19_CORRELATED_PRIVATE_NULL_N62"
EXPECTED_DATASETS_PER_SUBBANK = 200

REQUIRED_THRESHOLDS = {
    "primary_median_factor_correlation": 0.70,
    "primary_q10_factor_correlation": 0.40,
    "primary_inclusion_frequency": 0.80,
    "minimum_factor_variance": 0.0001,
    "minimum_supported_modalities": 2,
    "minimum_loading_energy_fraction_per_supported_modality": 0.15,
}


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


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n",
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


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")


def require_lock_status(path: Path, expected: str) -> dict[str, Any]:
    require_file(path)
    value = load_json(path)
    actual = str(value.get("status", "")).strip()
    if actual != expected:
        raise RuntimeError(
            f"{relative_path(path)} status mismatch: {actual!r} != {expected!r}"
        )
    return value


def as_int(row: dict[str, str], key: str) -> int:
    raw = str(row.get(key, "")).strip()
    if raw == "":
        raise RuntimeError(f"Scenario registry missing numeric value {key}.")
    return int(float(raw))


def discover_frozen_subbanks() -> tuple[str, str]:
    with SCENARIO_REGISTRY.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError("Final-gate scenario registry is empty.")

    required = {
        "scenario_id",
        "scenario_family",
        "n_samples",
        "true_biological_shared_factors",
        "true_technical_shared_factors",
    }
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(
            "Scenario registry lacks fields required by the frozen 05f3b D01 selector: "
            + ", ".join(sorted(missing))
        )

    correlated = [
        row for row in rows
        if str(row["scenario_id"]).strip() == CORRELATED_PRIVATE_SCENARIO_ID
    ]
    if len(correlated) != 1:
        raise RuntimeError(
            f"Expected exactly one {CORRELATED_PRIVATE_SCENARIO_ID} row; found {len(correlated)}."
        )
    corr = correlated[0]
    if (
        str(corr["scenario_family"]).strip() != "correlated_private_null"
        or as_int(corr, "n_samples") != 62
        or as_int(corr, "true_biological_shared_factors") != 0
        or as_int(corr, "true_technical_shared_factors") != 0
    ):
        raise RuntimeError("Correlated-private scenario no longer matches frozen D01 membership.")

    # Exact frozen 05f3b independent-null selector:
    #   D01 truth null AND scenario_family == final_independent_gate_curve,
    # then n_samples == 62.
    independent_n62 = [
        row for row in rows
        if str(row["scenario_family"]).strip() == "final_independent_gate_curve"
        and as_int(row, "n_samples") == 62
        and as_int(row, "true_biological_shared_factors") == 0
        and as_int(row, "true_technical_shared_factors") == 0
    ]
    if len(independent_n62) != 1:
        ids = [str(row.get("scenario_id", "")).strip() for row in independent_n62]
        raise RuntimeError(
            "Frozen 05f3b selector expected exactly one independent-null n=62 scenario; "
            f"found {len(independent_n62)}: {ids}"
        )

    return CORRELATED_PRIVATE_SCENARIO_ID, str(independent_n62[0]["scenario_id"]).strip()


def progress_counts() -> dict[str, int]:
    with PROGRESS_05F2.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "scenario_id" not in (reader.fieldnames or []):
            raise RuntimeError("05f2 progress manifest lacks scenario_id.")
        counts: dict[str, int] = {}
        n_rows = 0
        for row in reader:
            n_rows += 1
            sid = str(row["scenario_id"]).strip()
            counts[sid] = counts.get(sid, 0) + 1

    if n_rows != 3100:
        raise RuntimeError(f"05f2 progress manifest has {n_rows} rows, expected 3100.")
    return counts


def main() -> None:
    print("=" * 94)
    print("Paper 5 - freeze exploratory retention ablation contract (exact frozen D01 selector)")
    print("=" * 94)
    print(f"Script version: {SCRIPT_VERSION}")
    print("Scientific ablation result computed: NO")
    print("Model fitting: NO")
    print("GPU computation: NO")
    print("Formal 05f3e gate modification: FORBIDDEN")
    print()

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(
            f"05x1b output directory already exists and is non-empty: {OUTPUT_DIR}. "
            "Do not overwrite a frozen exploratory contract."
        )

    for path in [
        CONTRACT_05A,
        SCENARIO_REGISTRY,
        HELPER,
        PROGRESS_05F2,
        LOCK_05F2,
        LOCK_05F3E,
        LOCK_05F4B,
        LOCK_05X0,
    ]:
        require_file(path)

    require_lock_status(
        LOCK_05F2,
        "FROZEN_ALL_3100_FINAL_GATE_SHARDS_COMPLETE_BEFORE_SCIENTIFIC_AGGREGATION",
    )
    lock_05f3e = require_lock_status(
        LOCK_05F3E,
        "FROZEN_FINAL_GATE_SCIENTIFIC_RESULTS_NO_RETUNING",
    )
    require_lock_status(
        LOCK_05F4B,
        "READ_ONLY_MANUSCRIPT_ASSET_EXPORT_FROM_LOCKED_05F3E",
    )
    require_lock_status(
        LOCK_05X0,
        "READ_ONLY_PREFLIGHT_NO_EXPLORATORY_RESULT_COMPUTED",
    )

    if str(lock_05f3e.get("core_status", "")).strip() != "CORE_PASS":
        raise RuntimeError("05x1b requires the already-frozen 05f3e CORE_PASS state.")

    preflight = load_json(
        PAPER5_ROOT / "method_contract" / "05x0" / "retention_ablation_preflight.json"
    )
    if not bool(preflight.get("no_refit_ablation_plausible_from_schema", False)):
        raise RuntimeError("05x0 did not establish schema-plausible no-refit ablation.")
    if bool(preflight.get("scientific_result_computed", True)):
        raise RuntimeError("05x0 unexpectedly reports a scientific result.")

    method = load_json(CONTRACT_05A)
    thresholds = method.get("stability_gate", {}).get("thresholds", {})
    for key, expected in REQUIRED_THRESHOLDS.items():
        if key not in thresholds:
            raise RuntimeError(f"Frozen 05a threshold missing: {key}")
        actual = thresholds[key]
        if float(actual) != float(expected):
            raise RuntimeError(f"Frozen 05a threshold changed for {key}: {actual}")

    helper_text = HELPER.read_text(encoding="utf-8", errors="strict")
    required_fragments = [
        '"median_aligned_correlation"',
        '"q10_aligned_correlation"',
        '"inclusion_frequency"',
        '"reference_factor_variance"',
        '"n_supported_modalities"',
        '"passes_stability_gate"',
        '"minimum_loading_energy_fraction_per_supported_modality"',
    ]
    missing_fragments = [frag for frag in required_fragments if frag not in helper_text]
    if missing_fragments:
        raise RuntimeError(
            "Frozen helper no longer exposes expected retention components: "
            + ", ".join(missing_fragments)
        )

    correlated_id, independent_n62_id = discover_frozen_subbanks()
    counts = progress_counts()
    for sid in [correlated_id, independent_n62_id]:
        if counts.get(sid, 0) != EXPECTED_DATASETS_PER_SUBBANK:
            raise RuntimeError(
                f"{sid} has {counts.get(sid, 0)} datasets in 05f2, expected 200."
            )

    contract = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": STATUS,
        "analysis_role": "POSTUNBLINDING_EXPLORATORY_ABLATION",
        "supersedes_failed_pre_result_selector_attempt": {
            "stage": "05x1",
            "reason": (
                "05x1 searched for the literal token INDEPENDENT in scenario_id and stopped "
                "before creating a contract or computing any ablation result. 05x1b instead "
                "uses the exact frozen 05f3b selector fields from the scenario registry."
            ),
            "scientific_result_was_computed": False,
        },
        "scientific_boundary": {
            "formal_05f3e_gate_reopened": False,
            "formal_D01_recomputed_or_reweighted": False,
            "threshold_retuning": False,
            "model_refitting": False,
            "GPU_computation_required": False,
            "TARGET_or_outcomes_accessed": False,
            "claim_role": "EXPLORATORY_CONTEXT_ONLY",
        },
        "frozen_subbanks": [
            {
                "subbank_id": "CORRELATED_PRIVATE_N62",
                "scenario_id": correlated_id,
                "expected_datasets": 200,
                "role": "PRIMARY_EXPLORATORY_STRESS_MECHANISM",
                "selector_provenance": (
                    "Frozen 05f3b D01 correlated-private membership; exact scenario ID."
                ),
            },
            {
                "subbank_id": "INDEPENDENT_NULL_N62",
                "scenario_id": independent_n62_id,
                "expected_datasets": 200,
                "role": "MATCHED_N_MECHANISM_CONTROL",
                "selector_provenance": (
                    "Frozen 05f3b independent-null D01 pool, restricted to scenario_n_samples==62."
                ),
            },
        ],
        "frozen_thresholds_from_05a": {
            key: thresholds[key] for key in REQUIRED_THRESHOLDS
        },
        "frozen_rules": [
            {
                "rule_id": "FULL_FROZEN_RETENTION",
                "role": "REFERENCE_ONLY",
                "formula": (
                    "median>=0.70 AND q10>=0.40 AND inclusion>=0.80 "
                    "AND variance>=0.0001 AND n_supported_modalities>=2"
                ),
            },
            {
                "rule_id": "NO_MODALITY_SUPPORT",
                "role": "PRIMARY_EXPLORATORY_ABLATION",
                "formula": (
                    "median>=0.70 AND q10>=0.40 AND inclusion>=0.80 "
                    "AND variance>=0.0001"
                ),
                "removed_from_full_rule": ["n_supported_modalities>=2"],
            },
            {
                "rule_id": "PURE_FACTOR_REPRODUCIBILITY",
                "role": "SECONDARY_EXPLORATORY_ABLATION",
                "formula": "median>=0.70 AND q10>=0.40 AND inclusion>=0.80",
                "removed_from_full_rule": [
                    "variance>=0.0001",
                    "n_supported_modalities>=2",
                ],
            },
        ],
        "false_biological_acceptance_definition": (
            "A factor is a false biological acceptance under a candidate rule iff "
            "that rule retains it AND technical_truth_match is false AND "
            "biological_truth_match is false. A dataset-level event occurs iff at "
            "least one such factor exists."
        ),
        "reporting_plan": {
            "for_each_subbank_and_rule": [
                "dataset_level_false_biological_acceptance_numerator",
                "dataset_level_false_biological_acceptance_denominator",
                "dataset_level_false_biological_acceptance_rate",
                "two_sided_95pct_Wilson_interval",
                "total_retained_factors",
                "total_false_biological_factors",
            ],
            "paired_nested_rule_transitions": True,
            "p_values": "NONE",
            "new_PASS_FAIL_thresholds": "NONE",
        },
        "integrity_checks_before_result_reporting": [
            "Recompute support count from stored RNA/METHYLATION/CNV loading-energy fractions at 0.15.",
            "Recompute FULL_FROZEN_RETENTION and reproduce stored passes_stability_gate for every selected factor.",
            "Recompute factor-level stored false_biological_acceptance for every selected factor.",
            "Reproduce frozen full-rule subbank rates: correlated-private 70/200; independent n=62 0/200.",
        ],
        "expected_full_rule_reference_rates": {
            "CORRELATED_PRIVATE_N62": {"numerator": 70, "denominator": 200},
            "INDEPENDENT_NULL_N62": {"numerator": 0, "denominator": 200},
        },
        "performance_plan": {
            "selected_shards": 400,
            "parallel_file_io_and_hashing": True,
            "workers": "min(32, max(4, 2*logical_cpu_count))",
            "GPU": "NOT_USED",
        },
        "frozen_sources": {
            "05a_method_contract": file_record(CONTRACT_05A),
            "05a1_scenario_registry": file_record(SCENARIO_REGISTRY),
            "stability_helper": file_record(HELPER),
            "05f2_progress_manifest": file_record(PROGRESS_05F2),
            "05f2_completion_lock": file_record(LOCK_05F2),
            "05f3e_results_lock": file_record(LOCK_05F3E),
            "05f4b_asset_lock": file_record(LOCK_05F4B),
            "05x0_preflight_lock": file_record(LOCK_05X0),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(contract, OUTPUT_CONTRACT)

    readme = f"""Paper 5 05x1b — frozen exploratory retention ablation contract
=================================================================

No ablation result is computed in this stage.

Primary exploratory stress mechanism:
  {correlated_id} (200 datasets)

Matched-n independent-null control:
  {independent_n62_id} (200 datasets)

Independent-null selector provenance:
  exact frozen 05f3b family selector + n_samples==62 + D01 truth-null conditions.

Rules:
  FULL: median/q10/inclusion + variance + modality support
  NO_MODALITY_SUPPORT: median/q10/inclusion + variance
  PURE_FACTOR_REPRODUCIBILITY: median/q10/inclusion

This analysis is post-unblinding exploratory and cannot alter frozen 05f3e.
"""
    write_text_atomic(readme, OUTPUT_README)

    lock = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "script_path": relative_path(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "created_at_utc": utc_now_iso(),
        "status": STATUS,
        "contract": file_record(OUTPUT_CONTRACT),
        "readme": file_record(OUTPUT_README),
        "scientific_result_computed": False,
        "rewrite_policy": (
            "05x1b is frozen before 05x2b computes the exploratory result. "
            "Do not alter subbanks, rules, thresholds, metrics, or reference checks."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    print("05x1b exploratory ablation contract: FROZEN")
    print(f"Correlated-private: {correlated_id} (200)")
    print(f"Independent n=62:  {independent_n62_id} (200)")
    print("Rules: FULL / NO_MODALITY_SUPPORT / PURE_FACTOR_REPRODUCIBILITY")
    print(f"Contract: {relative_path(OUTPUT_CONTRACT)}")
    print(f"Lock:     {relative_path(OUTPUT_LOCK)}")
    print("Scientific ablation result computed: NO")


if __name__ == "__main__":
    main()
