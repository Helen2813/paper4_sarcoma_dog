#!/usr/bin/env python3
"""
Paper 5 - 06d0 read-only audit of V1 post-fit downstream scope.

Purpose
-------
Determine, from already-frozen Paper-5 contracts only, which post-fit TARGET
operations remain scientifically authorized after TARGET nested selection chose
V1_SHARED_PRIVATE_AE rather than V4_SABLE_MO.

This stage DOES NOT:
- read TARGET molecular matrices;
- read clinical variables, outcomes, biology, histology, mutation, or annotations;
- fit/refit any model;
- generate or score any scientific representation;
- run resampling, permutations, retention, or biological analyses;
- rewrite 05a/05e/06a/06b/06c/06c3.

The audit is deliberately conservative:
- q10=.60/.40 and the SABLE-MO stability/retention gate are hard-forbidden for V1
  because the frozen 06c contract says so explicitly;
- generic TARGET resampling stability and TARGET pairing permutations are not
  automatically authorized merely because similar machinery existed for SABLE-MO.
  They are classified only from explicit pre-existing frozen-contract evidence;
- ambiguous scope is reported as UNRESOLVED rather than silently generalized
  across engines.

No command-line arguments.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "06d0-audit-v1-postfit-downstream-scope-v1-no-cli"

EXPECTED = {
    "06c_contract_sha256":
        "8ab8516e30964f220d0f1204ca892e29ca337691e06cbcbf29315e1cb6aec2e5",
    "06c_lock_sha256":
        "1c3b4fb36706dc6afd0208fda23b1dc6d51c2953b4ccb9541099466aea13a374",
    "06c_summary_sha256":
        "17687ee34483f01876a85d1690c92bedb6a897ad02f62b9f0c804c7bc015ea4a",
    "06c3_lock_sha256":
        "b23d4c630cffca3bc5b1c44328128d4da46028b8db8cb070e6be93d9d0795be5",
    "runner_v7_sha256":
        "006f718de8b0a1f768f8b4b7c35e4951719970f4d91ef43fa75541c74c8dc3bb",
    "model_id": "V1_SHARED_PRIVATE_AE",
    "profile_id": "ARCH_COMPACT__REG_WEAK",
    "n_cases": 78,
    "primary_seed": 1103,
    "reproducibility_seeds": [2207, 3301],
    "fixed_epochs": 400,
}

EVIDENCE_DIRS = (
    "05a",
    "05a1",
    "05a2",
    "05d1",
    "05d5",
    "05e",
    "06a",
    "06b0",
    "06c",
)

TEXT_SUFFIXES = {".json", ".csv", ".txt", ".md"}

CONCEPT_PATTERNS = {
    "stability": re.compile(
        r"stability|resampl|subsampl|procrustes|hungarian|sign alignment",
        re.I,
    ),
    "retention_q10": re.compile(
        r"q10|q0[._]?10|0\.60|0\.40|retention|retain(?:ed|ing)?",
        re.I,
    ),
    "pairing": re.compile(
        r"pairing|patient[-_ ]pair|permutation",
        re.I,
    ),
    "target": re.compile(r"\bTARGET(?:-OS)?\b|target_", re.I),
    "v1": re.compile(r"\bV1_SHARED_PRIVATE_AE\b|shared/private", re.I),
    "v4_sable": re.compile(r"\bV4_SABLE_MO\b|\bSABLE[-_ ]MO\b", re.I),
    "selected_representation": re.compile(
        r"selected representation|after selection|after.*representation.*freeze|"
        r"full representation freeze|selected model|final full-data",
        re.I,
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    return path


def require_hash(label: str, path: Path, expected: str) -> str:
    observed = sha256_file(path)
    if observed.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA256 mismatch.\n"
            f"Expected: {expected}\n"
            f"Observed: {observed}\n"
            f"Path: {path}"
        )
    return observed


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((json_safe(v) for v in value), key=lambda x: repr(x))
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2) + "\n",
        encoding="utf-8",
    )


def relative_to_paper5(path: Path, paper5: Path) -> str:
    return path.resolve().relative_to(paper5.resolve()).as_posix()


def flatten_json(
    value: Any,
    prefix: str = "$",
) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}"
            rows.append((child, item))
            rows.extend(flatten_json(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            rows.append((child, item))
            rows.extend(flatten_json(item, child))
    return rows


def discover_evidence_files(
    paper5: Path,
) -> list[Path]:
    files: list[Path] = []
    base = paper5 / "method_contract"

    for dirname in EVIDENCE_DIRS:
        directory = base / dirname
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() in TEXT_SUFFIXES:
                files.append(path)

    return files


def text_snippets(
    path: Path,
    *,
    context: int = 1,
) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    hits: list[dict[str, Any]] = []

    for i, line in enumerate(lines):
        matched = [
            name
            for name, pattern in CONCEPT_PATTERNS.items()
            if pattern.search(line)
        ]
        if not matched:
            continue

        start = max(0, i - context)
        end = min(len(lines), i + context + 1)

        hits.append(
            {
                "line_number": i + 1,
                "concepts": matched,
                "text": "\n".join(
                    f"{j+1}: {lines[j]}"
                    for j in range(start, end)
                ),
            }
        )

    return hits


def evidence_summary(
    path: Path,
    paper5: Path,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": relative_to_paper5(path, paper5),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "suffix": path.suffix.lower(),
        "snippets": text_snippets(path),
    }

    if path.suffix.lower() == ".json":
        try:
            data = load_json(path)
            flat = flatten_json(data)
            record["json_key_value_evidence"] = [
                {
                    "json_path": key,
                    "value": value,
                }
                for key, value in flat
                if isinstance(value, (str, int, float, bool))
                and any(
                    pattern.search(f"{key}={value}")
                    for pattern in CONCEPT_PATTERNS.values()
                )
            ][:500]
        except Exception as exc:
            record["json_parse_error"] = (
                f"{type(exc).__name__}: {exc}"
            )

    return record


def compact_evidence_index(
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    index: list[dict[str, Any]] = []

    for file_record in evidence:
        path = file_record["path"]

        for hit in file_record.get("snippets", []):
            concepts = set(hit.get("concepts", []))
            if not concepts.intersection(
                {
                    "stability",
                    "retention_q10",
                    "pairing",
                    "target",
                    "v1",
                    "v4_sable",
                    "selected_representation",
                }
            ):
                continue

            index.append(
                {
                    "path": path,
                    "line_number": hit["line_number"],
                    "concepts": sorted(concepts),
                    "text": hit["text"],
                }
            )

    return index


def has_joint_concepts(
    evidence_index: list[dict[str, Any]],
    required: set[str],
    *,
    forbidden: set[str] | None = None,
) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    forbidden = forbidden or set()

    for item in evidence_index:
        concepts = set(item["concepts"])
        if not required.issubset(concepts):
            continue
        if forbidden.intersection(concepts):
            continue
        found.append(item)

    return found


def classify_scope(
    contract_06c: dict[str, Any],
    summary_06c: dict[str, Any],
    lock_06c3: dict[str, Any],
    evidence_index: list[dict[str, Any]],
) -> dict[str, Any]:
    full = contract_06c.get("full_data_refit", {})
    boundary = contract_06c.get("sable_retention_boundary", {})
    firewall = contract_06c.get("decision_firewall", {})

    if full.get("model_id") != EXPECTED["model_id"]:
        raise RuntimeError("06c selected model is no longer V1.")

    q10_060_forbidden = (
        boundary.get("q10_060_applied_to_v1") is False
    )
    q10_040_forbidden = (
        boundary.get("q10_040_applied_to_v1") is False
    )
    sable_gate_forbidden = (
        boundary.get("sable_mo_stability_gate_applied_to_v1")
        is False
    )

    if not (
        q10_060_forbidden
        and q10_040_forbidden
        and sable_gate_forbidden
    ):
        raise RuntimeError(
            "06c no longer contains the expected V1 SABLE-retention firewall."
        )

    target_selected_stability = has_joint_concepts(
        evidence_index,
        {"target", "stability", "selected_representation"},
        forbidden={"v4_sable"},
    )
    target_v1_stability = has_joint_concepts(
        evidence_index,
        {"target", "stability", "v1"},
        forbidden={"v4_sable"},
    )
    target_selected_pairing = has_joint_concepts(
        evidence_index,
        {"target", "pairing", "selected_representation"},
        forbidden={"v4_sable"},
    )
    target_v1_pairing = has_joint_concepts(
        evidence_index,
        {"target", "pairing", "v1"},
        forbidden={"v4_sable"},
    )

    sable_stability_evidence = has_joint_concepts(
        evidence_index,
        {"stability", "v4_sable"},
    )
    sable_pairing_evidence = has_joint_concepts(
        evidence_index,
        {"pairing", "v4_sable"},
    )

    if target_v1_stability or target_selected_stability:
        generic_stability_status = (
            "PRESPECIFIED_DESCRIPTIVE_SCOPE_EVIDENCE_FOUND__"
            "NO_SABLE_RETENTION_ALLOWED"
        )
    elif sable_stability_evidence:
        generic_stability_status = (
            "NOT_AUTHORIZED_FOR_V1_FROM_AVAILABLE_FROZEN_EVIDENCE__"
            "STABILITY_EVIDENCE_IS_SABLE_SPECIFIC"
        )
    else:
        generic_stability_status = (
            "UNRESOLVED__NO_EXPLICIT_ENGINE_AGNOSTIC_TARGET_V1_EVIDENCE"
        )

    if target_v1_pairing or target_selected_pairing:
        pairing_status = (
            "PRESPECIFIED_TARGET_PAIRING_SCOPE_EVIDENCE_FOUND__"
            "MUST_NOT_CHANGE_MODEL_OR_RETENTION"
        )
    elif sable_pairing_evidence:
        pairing_status = (
            "NOT_AUTHORIZED_FOR_V1_FROM_AVAILABLE_FROZEN_EVIDENCE__"
            "PAIRING_EVIDENCE_IS_SABLE_SPECIFIC"
        )
    else:
        pairing_status = (
            "UNRESOLVED__NO_EXPLICIT_ENGINE_AGNOSTIC_TARGET_V1_EVIDENCE"
        )

    return {
        "selected_representation": {
            "model_id": full.get("model_id"),
            "profile_id": full.get("profile_id"),
            "primary_seed": (
                full.get("training", {}).get("primary_seed")
            ),
            "representation_sha256": (
                lock_06c3.get("result_files", {})
                .get(
                    "target_v1_patient_shared_seed1103.npy",
                    {},
                )
                .get("sha256")
            ),
        },
        "hard_frozen_boundaries": {
            "q10_060_for_v1": "FORBIDDEN",
            "q10_040_for_v1": "FORBIDDEN",
            "sable_mo_stability_gate_for_v1": "FORBIDDEN",
            "sable_mo_factor_retention_thresholds_for_v1": "FORBIDDEN",
            "outcomes_may_change_primary_model":
                firewall.get("outcomes_may_change_primary_model"),
            "biology_may_change_primary_model":
                firewall.get("biology_may_change_primary_model"),
            "clinical_variables_may_change_primary_model":
                firewall.get("clinical_variables_may_change_primary_model"),
            "full_data_fit_quality_may_reopen_06b":
                firewall.get("full_data_fit_quality_may_reopen_06b"),
        },
        "generic_target_resampling_stability": {
            "status": generic_stability_status,
            "allowed_interpretation_if_run": (
                "Descriptive/reproducibility only. It may not apply "
                "SABLE-MO q10/retention thresholds, rescue/reject V1 factors, "
                "replace seed 1103, or reopen 06b."
            ),
            "target_v1_evidence_count":
                len(target_v1_stability),
            "target_selected_representation_evidence_count":
                len(target_selected_stability),
            "sable_specific_evidence_count":
                len(sable_stability_evidence),
            "supporting_evidence": (
                target_v1_stability
                + target_selected_stability
                + sable_stability_evidence
            )[:50],
        },
        "target_pairing_permutations": {
            "status": pairing_status,
            "allowed_interpretation_if_run": (
                "Outcome-blind falsification/sensitivity only. It may not "
                "change the selected V1 engine/profile/seed/epochs or import "
                "SABLE-MO retention rules."
            ),
            "target_v1_evidence_count":
                len(target_v1_pairing),
            "target_selected_representation_evidence_count":
                len(target_selected_pairing),
            "sable_specific_evidence_count":
                len(sable_pairing_evidence),
            "supporting_evidence": (
                target_v1_pairing
                + target_selected_pairing
                + sable_pairing_evidence
            )[:50],
        },
        "downstream_blinding": {
            "biology_access_in_06d0": False,
            "clinical_access_in_06d0": False,
            "outcome_access_in_06d0": False,
            "target_molecular_values_read_in_06d0": False,
            "model_fit_in_06d0": False,
            "representation_generated_in_06d0": False,
        },
        "summary_crosscheck": {
            "summary_status": summary_06c.get("status"),
            "summary_model_id": summary_06c.get("model_id"),
            "summary_primary_seed": summary_06c.get("primary_seed"),
            "summary_no_q10_060":
                summary_06c.get("q10_060_applied_to_v1"),
            "summary_no_q10_040":
                summary_06c.get("q10_040_applied_to_v1"),
            "summary_no_sable_stability_gate":
                summary_06c.get(
                    "sable_mo_stability_gate_applied_to_v1"
                ),
        },
    }


def main() -> None:
    print("=" * 104)
    print(
        "Paper 5 - 06d0 read-only audit of V1 post-fit downstream scope"
    )
    print("=" * 104)
    print("Script version:", SCRIPT_VERSION)
    print("TARGET molecular values read in 06d0: NO")
    print("TARGET outcomes/clinical/biology read in 06d0: NO")
    print("TARGET model fit in 06d0: NO")
    print("Scientific representation generated in 06d0: NO")
    print("06b/06c/06c3 rewritten: NO")

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    paper5 = project_root / "paper5"

    contract_path = (
        paper5
        / "method_contract"
        / "06c"
        / "target_v1_full_data_refit_contract.json"
    )
    contract_lock_path = (
        paper5
        / "method_contract"
        / "06c"
        / "06c_full_data_refit_contract_lock.json"
    )
    summary_path = (
        paper5
        / "results"
        / "target_representation"
        / "06c_full_data_v1"
        / "target_v1_full_data_refit_summary.json"
    )
    results_lock_path = (
        paper5
        / "method_contract"
        / "06c3"
        / "target_v1_full_data_refit_results_lock.json"
    )

    require_hash(
        "06c contract",
        contract_path,
        EXPECTED["06c_contract_sha256"],
    )
    require_hash(
        "06c lock",
        contract_lock_path,
        EXPECTED["06c_lock_sha256"],
    )
    require_hash(
        "06c summary",
        summary_path,
        EXPECTED["06c_summary_sha256"],
    )
    require_hash(
        "06c3 lock",
        results_lock_path,
        EXPECTED["06c3_lock_sha256"],
    )

    contract_06c = load_json(contract_path)
    lock_06c = load_json(contract_lock_path)
    summary_06c = load_json(summary_path)
    lock_06c3 = load_json(results_lock_path)

    if contract_06c.get("status") != (
        "FROZEN_BEFORE_TARGET_FULL_DATA_REFIT"
    ):
        raise RuntimeError("06c contract status mismatch.")
    if lock_06c.get("status") != (
        "06C_TARGET_V1_FULL_DATA_REFIT_CONTRACT_FROZEN"
    ):
        raise RuntimeError("06c lock status mismatch.")
    if summary_06c.get("status") != (
        "06C_TARGET_V1_FULL_DATA_REFIT_COMPLETE"
    ):
        raise RuntimeError("06c full-data summary is not COMPLETE.")
    if lock_06c3.get("status") != (
        "06C3_TARGET_V1_FULL_DATA_REFIT_RESULTS_FROZEN"
    ):
        raise RuntimeError("06c3 result lock is not FROZEN.")

    if summary_06c.get("model_id") != EXPECTED["model_id"]:
        raise RuntimeError("06c summary model mismatch.")
    if summary_06c.get("profile_id") != EXPECTED["profile_id"]:
        raise RuntimeError("06c summary profile mismatch.")
    if int(summary_06c.get("n_cases")) != EXPECTED["n_cases"]:
        raise RuntimeError("06c summary n_cases mismatch.")
    if int(summary_06c.get("primary_seed")) != EXPECTED["primary_seed"]:
        raise RuntimeError("06c primary seed mismatch.")
    if list(
        summary_06c.get("reproducibility_only_seeds", [])
    ) != EXPECTED["reproducibility_seeds"]:
        raise RuntimeError("06c reproducibility seeds mismatch.")
    if int(summary_06c.get("fixed_epochs")) != EXPECTED["fixed_epochs"]:
        raise RuntimeError("06c fixed epochs mismatch.")
    if summary_06c.get(
        "post_fit_seed_selection_performed"
    ) is not False:
        raise RuntimeError(
            "06c summary no longer preserves no-post-fit-seed-selection."
        )
    if summary_06c.get(
        "seed_ensemble_or_averaging_performed"
    ) is not False:
        raise RuntimeError(
            "06c summary no longer preserves no-seed-ensemble rule."
        )
    if summary_06c.get("outcomes_accessed") is not False:
        raise RuntimeError("06c summary reports outcome access.")
    if summary_06c.get("clinical_values_accessed") is not False:
        raise RuntimeError("06c summary reports clinical access.")
    if summary_06c.get(
        "biological_annotations_accessed"
    ) is not False:
        raise RuntimeError("06c summary reports biology access.")

    locked_runner_sha = str(
        lock_06c3.get("runner_source", {}).get("sha256", "")
    ).lower()
    if locked_runner_sha != EXPECTED["runner_v7_sha256"]:
        raise RuntimeError(
            "06c3 runner SHA differs from the frozen v7 runner."
        )

    evidence_files = discover_evidence_files(paper5)
    if not evidence_files:
        raise RuntimeError(
            "No frozen method-contract evidence files were discovered."
        )

    evidence = [
        evidence_summary(path, paper5)
        for path in evidence_files
    ]
    evidence_index = compact_evidence_index(evidence)

    scope = classify_scope(
        contract_06c,
        summary_06c,
        lock_06c3,
        evidence_index,
    )

    out_dir = paper5 / "method_contract" / "06d0"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(
            f"06d0 output directory already exists and is non-empty: {out_dir}\n"
            "Do not overwrite a completed/partial audit."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "status": "06D0_V1_POSTFIT_DOWNSTREAM_SCOPE_AUDIT_COMPLETE",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_state": {
            "target_molecular_values_read": False,
            "target_outcomes_clinical_biology_read": False,
            "target_model_fit": False,
            "scientific_representation_generated": False,
            "06b_06c_06c3_rewritten": False,
        },
        "frozen_06c_identity": {
            "contract_path": relative_to_paper5(
                contract_path,
                paper5,
            ),
            "contract_sha256": sha256_file(contract_path),
            "contract_lock_path": relative_to_paper5(
                contract_lock_path,
                paper5,
            ),
            "contract_lock_sha256": sha256_file(
                contract_lock_path
            ),
            "summary_path": relative_to_paper5(
                summary_path,
                paper5,
            ),
            "summary_sha256": sha256_file(summary_path),
            "results_lock_path": relative_to_paper5(
                results_lock_path,
                paper5,
            ),
            "results_lock_sha256": sha256_file(
                results_lock_path
            ),
        },
        "scope_decision": scope,
        "evidence_directories": list(EVIDENCE_DIRS),
        "evidence_file_count": len(evidence_files),
        "evidence_files": evidence,
        "evidence_index": evidence_index,
        "next_stage_rule": {
            "if_generic_target_resampling_stability_is_prespecified": (
                "A separate versioned source-authorized runner may be built "
                "for descriptive V1 reproducibility only. It must not use "
                "SABLE-MO q10 or factor-retention decisions."
            ),
            "if_target_pairing_permutations_are_prespecified": (
                "A separate versioned source-authorized outcome-blind pairing "
                "falsification runner may be built. It must not reopen 06b or "
                "change the V1 seed/profile/epochs."
            ),
            "if_scope_is_unresolved_or_sable_specific": (
                "Do not invent a new post-hoc V1 gate. Skip that branch in the "
                "frozen core and proceed only to the next already-prespecified "
                "outcome-blind downstream stage."
            ),
            "biology_clinical_outcomes_remain_closed": True,
        },
    }

    report_path = (
        out_dir
        / "v1_postfit_downstream_scope_audit.json"
    )
    write_json(report_path, report)

    evidence_csv_path = (
        out_dir
        / "v1_postfit_downstream_scope_evidence.csv"
    )
    with evidence_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "line_number",
                "concepts",
                "text",
            ],
        )
        writer.writeheader()
        for row in evidence_index:
            writer.writerow(
                {
                    "path": row["path"],
                    "line_number": row["line_number"],
                    "concepts": ";".join(row["concepts"]),
                    "text": row["text"].replace("\n", " | "),
                }
            )

    lock_payload = {
        "status": "06D0_V1_POSTFIT_DOWNSTREAM_SCOPE_AUDIT_FROZEN",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_report": {
            "path": relative_to_paper5(
                report_path,
                paper5,
            ),
            "sha256": sha256_file(report_path),
            "size_bytes": report_path.stat().st_size,
        },
        "evidence_table": {
            "path": relative_to_paper5(
                evidence_csv_path,
                paper5,
            ),
            "sha256": sha256_file(evidence_csv_path),
            "size_bytes": evidence_csv_path.stat().st_size,
        },
        "selected_model": EXPECTED["model_id"],
        "primary_seed": EXPECTED["primary_seed"],
        "q10_060_for_v1": "FORBIDDEN",
        "q10_040_for_v1": "FORBIDDEN",
        "sable_mo_stability_gate_for_v1": "FORBIDDEN",
        "generic_target_resampling_stability_status": (
            scope["generic_target_resampling_stability"]["status"]
        ),
        "target_pairing_permutations_status": (
            scope["target_pairing_permutations"]["status"]
        ),
        "target_molecular_values_read": False,
        "target_outcomes_clinical_biology_read": False,
        "target_model_fit": False,
        "scientific_representation_generated": False,
        "06b_06c_06c3_rewritten": False,
    }

    lock_path = (
        out_dir
        / "06d0_v1_postfit_downstream_scope_audit_lock.json"
    )
    write_json(lock_path, lock_payload)

    print("Exact 06c contract SHA: PASS")
    print("Exact 06c lock SHA: PASS")
    print("Exact 06c summary SHA: PASS")
    print("Exact 06c3 result-lock SHA: PASS")
    print("Selected model:", EXPECTED["model_id"])
    print("Primary seed:", EXPECTED["primary_seed"])
    print("q10=.60 transfer to V1: FORBIDDEN")
    print("q10=.40 transfer to V1: FORBIDDEN")
    print("SABLE-MO stability gate transfer to V1: FORBIDDEN")
    print(
        "Generic TARGET resampling stability:",
        scope[
            "generic_target_resampling_stability"
        ]["status"],
    )
    print(
        "TARGET pairing permutations:",
        scope["target_pairing_permutations"]["status"],
    )
    print("Frozen evidence files inspected:", len(evidence_files))
    print("Evidence snippets indexed:", len(evidence_index))
    print("TARGET molecular values read in 06d0: NO")
    print("TARGET outcomes/clinical/biology read in 06d0: NO")
    print("TARGET model fit in 06d0: NO")
    print("=" * 104)
    print("Paper 5 06d0 V1 post-fit downstream scope audit: COMPLETE")
    print("=" * 104)
    print(
        "Report:",
        relative_to_paper5(report_path, paper5),
    )
    print(
        "Evidence:",
        relative_to_paper5(
            evidence_csv_path,
            paper5,
        ),
    )
    print(
        "Lock:",
        relative_to_paper5(lock_path, paper5),
    )
    print("=" * 104)


if __name__ == "__main__":
    main()
