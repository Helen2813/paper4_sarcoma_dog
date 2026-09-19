#!/usr/bin/env python
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "05x2-run-stability-only-retention-ablation-fast-cpu-v1-no-cli"
STATUS = "POSTUNBLINDING_EXPLORATORY_RETENTION_ABLATION_COMPLETE"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

DIR_05X1 = PAPER5_ROOT / "method_contract" / "05x1"
CONTRACT_05X1 = DIR_05X1 / "stability_only_retention_ablation_contract.json"
LOCK_05X1 = DIR_05X1 / "05x1_contract_lock.json"

PROGRESS_05F2 = PAPER5_ROOT / "method_contract" / "05f2" / "full_run_progress_manifest.csv"

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05x2"
OUTPUT_SUMMARY = OUTPUT_DIR / "retention_ablation_summary.csv"
OUTPUT_TRANSITIONS = OUTPUT_DIR / "retention_ablation_paired_transitions.csv"
OUTPUT_RESULT = OUTPUT_DIR / "retention_ablation_result.json"
OUTPUT_README = OUTPUT_DIR / "05x2_README.txt"
OUTPUT_LOCK = OUTPUT_DIR / "05x2_results_lock.json"

EXPECTED_CONTRACT_STATUS = (
    "FROZEN_POSTUNBLINDING_EXPLORATORY_ABLATION_BEFORE_RESULT_COMPUTATION"
)

REQUIRED_FACTOR_FIELDS = {
    "median_aligned_correlation",
    "q10_aligned_correlation",
    "inclusion_frequency",
    "reference_factor_variance",
    "loading_energy_RNA",
    "loading_energy_METHYLATION",
    "loading_energy_CNV",
    "n_supported_modalities",
    "passes_stability_gate",
    "technical_truth_match",
    "biological_truth_match",
    "false_biological_acceptance",
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


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (1, 1.0):
        return True
    if value in (0, 0.0):
        return False
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes"}:
        return True
    if text in {"false", "f", "0", "no"}:
        return False
    raise RuntimeError(f"Cannot parse boolean value: {value!r}")


def wilson_interval(numerator: int, denominator: int) -> tuple[float, float]:
    if denominator <= 0:
        raise RuntimeError("Wilson denominator must be positive.")
    z = 1.959963984540054
    n = float(denominator)
    p = float(numerator) / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (
        z
        * math.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n)))
        / denom
    )
    return max(0.0, center - half), min(1.0, center + half)


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


def write_csv_atomic(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    fields = list(rows[0].keys())
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".csv", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
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


def resolve_project_path(raw: str) -> Path:
    p = Path(str(raw))
    return p if p.is_absolute() else PROJECT_ROOT / p


def verify_contract() -> dict[str, Any]:
    for path in [CONTRACT_05X1, LOCK_05X1, PROGRESS_05F2]:
        require_file(path)

    lock = load_json(LOCK_05X1)
    if str(lock.get("status", "")).strip() != EXPECTED_CONTRACT_STATUS:
        raise RuntimeError("05x1 lock has unexpected status.")

    contract_record = lock.get("contract", {})
    if sha256_file(CONTRACT_05X1).lower() != str(contract_record.get("sha256", "")).lower():
        raise RuntimeError("05x1 contract hash mismatch.")

    contract = load_json(CONTRACT_05X1)
    if str(contract.get("status", "")).strip() != EXPECTED_CONTRACT_STATUS:
        raise RuntimeError("05x1 contract status mismatch.")

    if contract.get("analysis_role") != "POSTUNBLINDING_EXPLORATORY_ABLATION":
        raise RuntimeError("05x1 analysis role changed.")

    return contract


def selected_progress_rows(contract: dict[str, Any]) -> list[dict[str, str]]:
    wanted = {
        str(row["scenario_id"]): str(row["subbank_id"])
        for row in contract["frozen_subbanks"]
    }

    selected: list[dict[str, str]] = []
    with PROGRESS_05F2.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset_key", "scenario_id", "shard_path", "shard_sha256"}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError(
                "05f2 progress manifest lacks: "
                + ", ".join(sorted(required - set(reader.fieldnames or [])))
            )
        for row in reader:
            scenario_id = str(row["scenario_id"]).strip()
            if scenario_id in wanted:
                row["_subbank_id"] = wanted[scenario_id]
                selected.append(row)

    counts: dict[str, int] = {}
    for row in selected:
        counts[row["_subbank_id"]] = counts.get(row["_subbank_id"], 0) + 1

    for subbank in contract["frozen_subbanks"]:
        sid = str(subbank["subbank_id"])
        expected = int(subbank["expected_datasets"])
        if counts.get(sid, 0) != expected:
            raise RuntimeError(
                f"Selected {counts.get(sid, 0)} datasets for {sid}; expected {expected}."
            )

    if len(selected) != 400:
        raise RuntimeError(f"Expected 400 selected datasets total; found {len(selected)}.")
    return selected


def read_one(
    row: dict[str, str],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    path = resolve_project_path(row["shard_path"])
    require_file(path)

    expected_hash = str(row["shard_sha256"]).strip()
    if sha256_file(path).lower() != expected_hash.lower():
        raise RuntimeError(f"Shard hash mismatch: {path}")

    shard = load_json(path)
    payload = shard.get("scientific_payload")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Shard lacks scientific_payload: {path}")

    dataset = payload.get("dataset_row")
    factors = payload.get("factor_rows")
    if not isinstance(dataset, dict) or not isinstance(factors, list):
        raise RuntimeError(f"Unexpected payload schema: {path}")

    if str(dataset.get("dataset_key", "")).strip() != str(row["dataset_key"]).strip():
        raise RuntimeError(f"Dataset key mismatch: {path}")
    if str(dataset.get("scenario_id", "")).strip() != str(row["scenario_id"]).strip():
        raise RuntimeError(f"Scenario ID mismatch: {path}")

    t_median = float(thresholds["primary_median_factor_correlation"])
    t_q10 = float(thresholds["primary_q10_factor_correlation"])
    t_inclusion = float(thresholds["primary_inclusion_frequency"])
    t_variance = float(thresholds["minimum_factor_variance"])
    t_energy = float(thresholds["minimum_loading_energy_fraction_per_supported_modality"])
    t_support = int(thresholds["minimum_supported_modalities"])

    rule_ids = (
        "FULL_FROZEN_RETENTION",
        "NO_MODALITY_SUPPORT",
        "PURE_FACTOR_REPRODUCIBILITY",
    )
    retained = {rule: 0 for rule in rule_ids}
    false_count = {rule: 0 for rule in rule_ids}
    event = {rule: False for rule in rule_ids}

    for factor in factors:
        if not isinstance(factor, dict):
            raise RuntimeError(f"Non-object factor row: {path}")
        missing = REQUIRED_FACTOR_FIELDS - set(factor)
        if missing:
            raise RuntimeError(
                f"Factor row missing fields in {path}: " + ", ".join(sorted(missing))
            )

        median = float(factor["median_aligned_correlation"])
        q10 = float(factor["q10_aligned_correlation"])
        inclusion = float(factor["inclusion_frequency"])
        variance = float(factor["reference_factor_variance"])

        energies = [
            float(factor["loading_energy_RNA"]),
            float(factor["loading_energy_METHYLATION"]),
            float(factor["loading_energy_CNV"]),
        ]
        recomputed_n_supported = sum(value >= t_energy for value in energies)
        stored_n_supported = int(factor["n_supported_modalities"])
        if recomputed_n_supported != stored_n_supported:
            raise RuntimeError(
                f"n_supported_modalities mismatch in {path}: "
                f"stored={stored_n_supported}, recomputed={recomputed_n_supported}"
            )

        reproducibility = (
            median >= t_median
            and q10 >= t_q10
            and inclusion >= t_inclusion
        )
        no_support = reproducibility and variance >= t_variance
        full = no_support and stored_n_supported >= t_support

        stored_full = parse_bool(factor["passes_stability_gate"])
        if full != stored_full:
            raise RuntimeError(
                f"Full retention rule does not reproduce passes_stability_gate in {path}."
            )

        technical = parse_bool(factor["technical_truth_match"])
        biological = parse_bool(factor["biological_truth_match"])

        stored_false = parse_bool(factor["false_biological_acceptance"])
        recomputed_stored_false = full and (not technical) and (not biological)
        if stored_false != recomputed_stored_false:
            raise RuntimeError(
                f"Stored false_biological_acceptance mismatch in {path}."
            )

        rules = {
            "FULL_FROZEN_RETENTION": full,
            "NO_MODALITY_SUPPORT": no_support,
            "PURE_FACTOR_REPRODUCIBILITY": reproducibility,
        }

        for rule, keep in rules.items():
            if keep:
                retained[rule] += 1
                is_false = (not technical) and (not biological)
                if is_false:
                    false_count[rule] += 1
                    event[rule] = True

    stored_dataset_event = parse_bool(dataset["false_biological_acceptance_event"])
    if stored_dataset_event != event["FULL_FROZEN_RETENTION"]:
        raise RuntimeError(
            f"Full rule does not reproduce dataset false_biological_acceptance_event in {path}."
        )

    return {
        "dataset_key": str(row["dataset_key"]).strip(),
        "scenario_id": str(row["scenario_id"]).strip(),
        "subbank_id": str(row["_subbank_id"]),
        "event": event,
        "retained": retained,
        "false_count": false_count,
        "n_factors": len(factors),
    }


def summarize(
    records: list[dict[str, Any]],
    contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rule_order = [
        "FULL_FROZEN_RETENTION",
        "NO_MODALITY_SUPPORT",
        "PURE_FACTOR_REPRODUCIBILITY",
    ]

    summary_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []

    by_subbank: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_subbank.setdefault(record["subbank_id"], []).append(record)

    expected_reference = contract["expected_full_rule_reference_rates_for_integrity_check"]

    for subbank in contract["frozen_subbanks"]:
        subbank_id = str(subbank["subbank_id"])
        part = by_subbank.get(subbank_id, [])
        denominator = int(subbank["expected_datasets"])
        if len(part) != denominator:
            raise RuntimeError(
                f"{subbank_id}: {len(part)} records, expected {denominator}."
            )

        for rule in rule_order:
            numerator = sum(bool(r["event"][rule]) for r in part)
            low, high = wilson_interval(numerator, denominator)
            summary_rows.append(
                {
                    "subbank_id": subbank_id,
                    "scenario_id": subbank["scenario_id"],
                    "subbank_role": subbank["role"],
                    "rule_id": rule,
                    "rule_role": next(
                        x["role"] for x in contract["frozen_rules"] if x["rule_id"] == rule
                    ),
                    "dataset_false_accept_numerator": numerator,
                    "dataset_false_accept_denominator": denominator,
                    "dataset_false_accept_rate": numerator / denominator,
                    "ci95_low": low,
                    "ci95_high": high,
                    "total_retained_factors": sum(r["retained"][rule] for r in part),
                    "total_false_biological_factors": sum(r["false_count"][rule] for r in part),
                }
            )

        full_num = sum(bool(r["event"]["FULL_FROZEN_RETENTION"]) for r in part)
        expected = expected_reference[subbank_id]
        if full_num != int(expected["numerator"]) or denominator != int(expected["denominator"]):
            raise RuntimeError(
                f"{subbank_id}: full-rule reference mismatch "
                f"({full_num}/{denominator} vs "
                f"{expected['numerator']}/{expected['denominator']})."
            )

        for ablation in [
            "NO_MODALITY_SUPPORT",
            "PURE_FACTOR_REPRODUCIBILITY",
        ]:
            both = sum(
                r["event"]["FULL_FROZEN_RETENTION"] and r["event"][ablation]
                for r in part
            )
            ablation_only = sum(
                (not r["event"]["FULL_FROZEN_RETENTION"]) and r["event"][ablation]
                for r in part
            )
            full_only = sum(
                r["event"]["FULL_FROZEN_RETENTION"] and (not r["event"][ablation])
                for r in part
            )
            neither = denominator - both - ablation_only - full_only

            # Since the ablations remove constraints from the full rule, FULL must be nested.
            if full_only != 0:
                raise RuntimeError(
                    f"{subbank_id}/{ablation}: nested-rule violation; full_only={full_only}."
                )

            transition_rows.append(
                {
                    "subbank_id": subbank_id,
                    "ablation_rule_id": ablation,
                    "both_full_and_ablation_false_event": both,
                    "ablation_false_event_but_not_full": ablation_only,
                    "full_false_event_but_not_ablation": full_only,
                    "neither_false_event": neither,
                    "denominator": denominator,
                }
            )

    result = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": STATUS,
        "analysis_role": "POSTUNBLINDING_EXPLORATORY_ABLATION",
        "formal_final_gate_consequence": "NONE",
        "model_refit": False,
        "GPU_computation": False,
        "selected_datasets": len(records),
        "selected_factor_rows": sum(r["n_factors"] for r in records),
        "summary": summary_rows,
        "paired_transitions": transition_rows,
        "interpretation_guardrails": [
            "05x2 is exploratory and cannot replace, reweight, or reinterpret frozen D01.",
            "No p-values or new PASS/FAIL thresholds are introduced.",
            "FULL_FROZEN_RETENTION is reproduced only as an integrity reference.",
            "NO_MODALITY_SUPPORT isolates removal of the modality-support count requirement while retaining the variance floor.",
            "PURE_FACTOR_REPRODUCIBILITY reports the prespecified correlation/inclusion screen alone and is secondary exploratory context.",
        ],
    }
    return summary_rows, transition_rows, result


def main() -> None:
    print("=" * 92)
    print("Paper 5 - FAST read-only CPU retention-rule ablation on frozen subbanks")
    print("=" * 92)
    print(f"Script version: {SCRIPT_VERSION}")
    print("Scientific role: POST-UNBLINDING EXPLORATORY")
    print("GPU computation: NO")
    print("Model fitting: NO")
    print("Selected frozen shards: 400")
    print()

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(
            f"05x2 output directory already exists and is non-empty: {OUTPUT_DIR}. "
            "Do not overwrite exploratory results."
        )

    contract = verify_contract()
    rows = selected_progress_rows(contract)
    thresholds = contract["frozen_thresholds_from_05a"]

    logical = os.cpu_count() or 4
    workers = min(32, max(4, 2 * logical))
    print(f"Parallel JSON/hash workers: {workers}")
    print("Reading/verifying selected shards in parallel...")

    def job(row: dict[str, str]) -> dict[str, Any]:
        return read_one(row, thresholds)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(job, rows, chunksize=1))

    print(f"Verified/read datasets: {len(records)}/400")
    print(f"Factor rows evaluated: {sum(r['n_factors'] for r in records):,}")

    summary_rows, transition_rows, result = summarize(records, contract)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(summary_rows, OUTPUT_SUMMARY)
    write_csv_atomic(transition_rows, OUTPUT_TRANSITIONS)
    write_json_atomic(result, OUTPUT_RESULT)

    readme_lines = [
        "Paper 5 05x2 — post-unblinding exploratory retention-rule ablation",
        "=" * 74,
        "",
        f"Script version: {SCRIPT_VERSION}",
        f"Status: {STATUS}",
        "",
        "No model was refit. No GPU computation was performed.",
        "Only 400 already-frozen sealed final-gate shards were read.",
        "The frozen 05f3e final gate remains authoritative and unchanged.",
        "",
        "Results",
        "-------",
    ]

    for row in summary_rows:
        readme_lines.append(
            f"{row['subbank_id']} | {row['rule_id']}: "
            f"{row['dataset_false_accept_numerator']}/"
            f"{row['dataset_false_accept_denominator']} = "
            f"{100*row['dataset_false_accept_rate']:.1f}% "
            f"[95% CI {100*row['ci95_low']:.1f}%, {100*row['ci95_high']:.1f}%]"
        )

    readme_lines.extend(
        [
            "",
            "Interpretation boundary",
            "-----------------------",
            "Exploratory context only; no new formal PASS/FAIL decision.",
            "No D01 reweighting. No threshold retuning. No outcome/TARGET access.",
            "",
        ]
    )
    write_text_atomic("\n".join(readme_lines), OUTPUT_README)

    generated = [
        OUTPUT_SUMMARY,
        OUTPUT_TRANSITIONS,
        OUTPUT_RESULT,
        OUTPUT_README,
    ]
    lock = {
        "script_version": SCRIPT_VERSION,
        "script_path": relative_path(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "created_at_utc": utc_now_iso(),
        "status": STATUS,
        "05x1_contract": file_record(CONTRACT_05X1),
        "05x1_lock": file_record(LOCK_05X1),
        "scientific_role": "POSTUNBLINDING_EXPLORATORY_ABLATION",
        "formal_final_gate_consequence": "NONE",
        "model_refit": False,
        "GPU_computation": False,
        "generated_files": {
            relative_path(path): file_record(path) for path in generated
        },
        "rewrite_policy": (
            "05x2 results are immutable. Any additional post-unblinding exploratory "
            "retention analysis requires a new versioned stage."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    print()
    print("-" * 92)
    print("Exploratory retention ablation results")
    print("-" * 92)
    for row in summary_rows:
        print(
            f"{row['subbank_id']:24s} | {row['rule_id']:28s} | "
            f"{row['dataset_false_accept_numerator']:3d}/"
            f"{row['dataset_false_accept_denominator']:3d} "
            f"= {100*row['dataset_false_accept_rate']:5.1f}%"
        )
    print("-" * 92)
    print(f"Summary: {relative_path(OUTPUT_SUMMARY)}")
    print(f"Transitions: {relative_path(OUTPUT_TRANSITIONS)}")
    print(f"Result: {relative_path(OUTPUT_RESULT)}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("05f3e formal gate modified: NO")


if __name__ == "__main__":
    main()
