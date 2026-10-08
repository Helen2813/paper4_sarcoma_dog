#!/usr/bin/env python3
"""
Paper 5 - 06d1 post-freeze TARGET technical-status audit for frozen V1.

Read-only scientific audit:
- verifies frozen 06c/06c3 V1 result identity;
- verifies 06d0 forbids SABLE-MO q10/stability and TARGET pairing/stability transfer to V1;
- verifies the frozen 04d technical-metadata preflight and its generated-file hashes;
- if 04d has zero strict usable technical covariates, performs NO association test and
  assigns technical status UNKNOWN to all five frozen V1 shared coordinates.

No molecular matrices, biological annotations, clinical values, outcomes, or model fitting.
No command-line arguments.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "06d1-audit-v1-postfreeze-technical-status-v2-no-cli"

EXPECTED = {
    "06c_summary_sha256":
        "17687ee34483f01876a85d1690c92bedb6a897ad02f62b9f0c804c7bc015ea4a",
    "06c3_lock_sha256":
        "b23d4c630cffca3bc5b1c44328128d4da46028b8db8cb070e6be93d9d0795be5",
    "runner_v7_sha256":
        "006f718de8b0a1f768f8b4b7c35e4951719970f4d91ef43fa75541c74c8dc3bb",
    "model_id": "V1_SHARED_PRIVATE_AE",
    "profile_id": "ARCH_COMPACT__REG_WEAK",
    "primary_seed": 1103,
    "shared_dim": 5,
    "04d_contract_version":
        "paper5-target-technical-metadata-preflight-execution-v1",
    "04d_manifest_sha256":
        "73232610dcd3f0e2c9a45d7c3a66c0eaac138766a9cf339ed03cd67d282ef532",
    "04d_resolved_registry_sha256":
        "563ef2360755591eeb692940d9af250cb665c23640c718694d4f07977430a70c",
    "04d_case_matrix_sha256":
        "6ca088d5c287c2d89aafebb806dc09921e22d5bf38372fb1c3c7b7b0bece9be2",
    "04d_covariate_summary_sha256":
        "0f7063d900467a92dbc43586370910dd4d87f43a9facaadf6eb861d2605be76e",
    "04d_cross_view_sha256":
        "b5149619601105a3237143f6d2a3c32c41082e5d56a8d58b086fb17f60c24125",
    "strict_status":
        "ELIGIBLE_FOR_POSTFREEZE_FACTOR_FLAGGING",
    "ambiguous_status":
        "ELIGIBLE_AMBIGUOUS_SPECIMEN_ASSOCIATION",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    return path


def require_hash(label: str, path: Path, expected: str) -> None:
    observed = sha256_file(require_file(path))
    if observed.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA256 mismatch. Expected={expected} Observed={observed} Path={path}"
        )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def rel(path: Path, paper5: Path) -> str:
    return path.resolve().relative_to(paper5.resolve()).as_posix()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with require_file(path).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return fields, rows


def resolve_column(fields: list[str], choices: list[str], label: str) -> str:
    for name in choices:
        if name in fields:
            return name
    raise RuntimeError(f"Could not resolve {label} column. Columns={fields}")


def verify_04d_generated_files(
    paper5: Path,
    lock: dict[str, Any],
) -> dict[str, Any]:
    generated = lock.get("generated_files")
    if not isinstance(generated, dict) or not generated:
        raise RuntimeError("04d lock has no generated_files mapping.")

    checked: dict[str, Any] = {}

    for raw, rec in generated.items():
        if not isinstance(rec, dict):
            raise RuntimeError(f"Malformed 04d generated-file record: {raw}")

        normalized = str(raw).replace("\\", "/")
        marker = "paper5/"
        pos = normalized.lower().find(marker)
        if pos < 0:
            raise RuntimeError(f"04d generated-file path lacks paper5/: {raw}")

        local = paper5 / normalized[pos + len(marker):]
        require_file(local)

        expected = str(rec.get("sha256", "")).lower()
        observed = sha256_file(local)
        if not expected or observed != expected:
            raise RuntimeError(
                f"04d generated-file SHA mismatch: {raw} "
                f"Expected={expected} Observed={observed}"
            )

        if rec.get("size_bytes") is not None:
            if int(rec["size_bytes"]) != local.stat().st_size:
                raise RuntimeError(f"04d generated-file size mismatch: {raw}")

        checked[rel(local, paper5)] = {
            "sha256": observed,
            "size_bytes": local.stat().st_size,
        }

    return checked


def main() -> None:
    print("=" * 104)
    print("Paper 5 - 06d1 post-freeze TARGET technical-status audit for frozen V1")
    print("=" * 104)
    print("Script version:", SCRIPT_VERSION)
    print("TARGET molecular matrices read in 06d1: NO")
    print("Biological annotations read in 06d1: NO")
    print("Clinical/outcome values read in 06d1: NO")
    print("TARGET model fit/refit in 06d1: NO")
    print("SABLE-MO q10/stability transfer to V1: FORBIDDEN")

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    paper5 = project_root / "paper5"

    summary_path = (
        paper5 / "results" / "target_representation" / "06c_full_data_v1"
        / "target_v1_full_data_refit_summary.json"
    )
    results_lock_path = (
        paper5 / "method_contract" / "06c3"
        / "target_v1_full_data_refit_results_lock.json"
    )
    scope_lock_path = (
        paper5 / "method_contract" / "06d0"
        / "06d0_v1_postfit_downstream_scope_audit_lock.json"
    )

    tech_dir = paper5 / "data_manifest" / "TARGET_OS_technical_metadata"
    tech_lock_path = tech_dir / "TARGET_OS_technical_metadata_preflight_lock.json"
    tech_manifest_path = tech_dir / "TARGET_OS_technical_metadata_preflight_manifest.json"
    registry_path = tech_dir / "TARGET_OS_resolved_technical_covariate_registry.csv"
    case_matrix_path = tech_dir / "TARGET_OS_technical_covariate_case_matrix.csv"
    summary_tech_path = tech_dir / "TARGET_OS_technical_covariate_summary.csv"
    cross_view_path = tech_dir / "TARGET_OS_technical_covariate_cross_view_concordance.csv"

    require_hash("06c summary", summary_path, EXPECTED["06c_summary_sha256"])
    require_hash("06c3 lock", results_lock_path, EXPECTED["06c3_lock_sha256"])
    require_hash("04d manifest", tech_manifest_path, EXPECTED["04d_manifest_sha256"])
    require_hash("04d registry", registry_path, EXPECTED["04d_resolved_registry_sha256"])
    require_hash("04d case matrix", case_matrix_path, EXPECTED["04d_case_matrix_sha256"])
    require_hash("04d summary", summary_tech_path, EXPECTED["04d_covariate_summary_sha256"])
    require_hash("04d cross-view", cross_view_path, EXPECTED["04d_cross_view_sha256"])

    summary = load_json(summary_path)
    results_lock = load_json(results_lock_path)
    scope_lock = load_json(scope_lock_path)
    tech_lock = load_json(tech_lock_path)
    tech_manifest = load_json(tech_manifest_path)

    if summary.get("status") != "06C_TARGET_V1_FULL_DATA_REFIT_COMPLETE":
        raise RuntimeError("06c summary is not COMPLETE.")
    if results_lock.get("status") != "06C3_TARGET_V1_FULL_DATA_REFIT_RESULTS_FROZEN":
        raise RuntimeError("06c3 lock is not FROZEN.")
    if summary.get("model_id") != EXPECTED["model_id"]:
        raise RuntimeError("Frozen model mismatch.")
    if summary.get("profile_id") != EXPECTED["profile_id"]:
        raise RuntimeError("Frozen profile mismatch.")
    if int(summary.get("primary_seed")) != EXPECTED["primary_seed"]:
        raise RuntimeError("Frozen primary seed mismatch.")
    if str(results_lock.get("runner_source", {}).get("sha256", "")).lower() != (
        EXPECTED["runner_v7_sha256"]
    ):
        raise RuntimeError("06c3 runner SHA mismatch.")

    if scope_lock.get("status") != "06D0_V1_POSTFIT_DOWNSTREAM_SCOPE_AUDIT_FROZEN":
        raise RuntimeError("06d0 scope lock is not FROZEN.")

    for key in (
        "q10_060_for_v1",
        "q10_040_for_v1",
        "sable_mo_stability_gate_for_v1",
    ):
        if scope_lock.get(key) != "FORBIDDEN":
            raise RuntimeError(f"06d0 no longer forbids {key}.")

    stability_scope = str(
        scope_lock.get("generic_target_resampling_stability_status", "")
    )
    pairing_scope = str(
        scope_lock.get("target_pairing_permutations_status", "")
    )
    if not stability_scope.startswith("NOT_AUTHORIZED_FOR_V1_"):
        raise RuntimeError("06d0 V1 stability boundary changed.")
    if not pairing_scope.startswith("NOT_AUTHORIZED_FOR_V1_"):
        raise RuntimeError("06d0 V1 pairing boundary changed.")

    if tech_lock.get("contract_version") != EXPECTED["04d_contract_version"]:
        raise RuntimeError("04d contract version mismatch.")
    if tech_lock.get("status") != "FROZEN_BEFORE_REPRESENTATION_MODEL_FITTING":
        raise RuntimeError("04d lock status mismatch.")
    if tech_manifest.get("status") != "FROZEN_BEFORE_REPRESENTATION_MODEL_FITTING":
        raise RuntimeError("04d manifest status mismatch.")
    scientific_state = tech_manifest.get("scientific_state")
    if not isinstance(scientific_state, dict):
        raise RuntimeError("04d manifest scientific_state is missing or malformed.")

    required_false_scientific_state = (
        "biological_labels_used",
        "clinical_values_retained",
        "identifiers_used_as_covariates",
        "outcome_loaded",
        "raw_dates_retained",
        "representation_model_fitting_performed",
        "technical_covariates_used_for_model_selection",
    )
    for key in required_false_scientific_state:
        if scientific_state.get(key) is not False:
            raise RuntimeError(
                f"04d scientific_state[{key!r}] is not explicitly FALSE: "
                f"{scientific_state.get(key)!r}"
            )

    resolved_covariates = tech_manifest.get("resolved_covariates")
    if not isinstance(resolved_covariates, dict):
        raise RuntimeError("04d manifest resolved_covariates is missing or malformed.")

    manifest_n_strict = int(resolved_covariates.get("n_strict_usable", -1))
    manifest_n_ambiguous = int(resolved_covariates.get("n_ambiguous_usable", -1))
    manifest_n_cross_view = int(
        resolved_covariates.get("n_potential_cross_view_batch_candidates", -1)
    )
    manifest_no_usable_rule = resolved_covariates.get(
        "technical_status_if_no_strict_usable_fields"
    )

    if manifest_n_strict != 0:
        raise RuntimeError(
            "04d manifest no longer records zero strict usable technical covariates."
        )
    if manifest_n_ambiguous != 0:
        raise RuntimeError(
            "04d manifest no longer records zero ambiguous usable technical covariates."
        )
    if manifest_n_cross_view != 0:
        raise RuntimeError(
            "04d manifest no longer records zero potential cross-view batch candidates."
        )
    if manifest_no_usable_rule != "stable_technical_status_unknown":
        raise RuntimeError(
            "04d no-strict-usable technical-status rule changed: "
            f"{manifest_no_usable_rule!r}"
        )

    checked_04d = verify_04d_generated_files(paper5, tech_lock)

    fields, rows = read_csv(registry_path)
    eligibility_col = resolve_column(
        fields,
        [
            "eligibility_status",
            "eligibility",
            "postfreeze_factor_flagging_eligibility",
            "technical_analysis_eligibility",
        ],
        "eligibility",
    )
    covariate_col = resolve_column(
        fields,
        ["covariate_id", "technical_covariate_id", "field_id", "covariate"],
        "covariate id",
    )
    modality_col = next(
        (x for x in ["modality", "view", "omic_view"] if x in fields),
        None,
    )

    strict_rows = [
        r for r in rows
        if str(r.get(eligibility_col, "")).strip() == EXPECTED["strict_status"]
    ]
    ambiguous_rows = [
        r for r in rows
        if str(r.get(eligibility_col, "")).strip() == EXPECTED["ambiguous_status"]
    ]

    if strict_rows:
        raise RuntimeError(
            "04d now contains strict usable technical covariates. "
            "This v1 audit is locked to the prospectively frozen zero-strict-usable state "
            "and refuses to switch post hoc into association testing."
        )

    if len(strict_rows) != manifest_n_strict:
        raise RuntimeError(
            "04d strict-usable count disagrees between manifest and resolved registry: "
            f"manifest={manifest_n_strict} registry={len(strict_rows)}"
        )
    if len(ambiguous_rows) != manifest_n_ambiguous:
        raise RuntimeError(
            "04d ambiguous-usable count disagrees between manifest and resolved registry: "
            f"manifest={manifest_n_ambiguous} registry={len(ambiguous_rows)}"
        )

    coordinates = []
    for i in range(1, EXPECTED["shared_dim"] + 1):
        coordinates.append({
            "coordinate_id": f"V1_SHARED_{i:02d}",
            "model_id": EXPECTED["model_id"],
            "primary_seed": EXPECTED["primary_seed"],
            "technical_status":
                "UNKNOWN_NO_STRICT_USABLE_TECHNICAL_COVARIATES",
            "technical_association_test_performed": False,
            "eligible_for_primary_biological_interpretation_from_technical_audit_alone":
                False,
            "interpretation":
                "Technical status unknown; absence of a usable technical label is not evidence of non-technical origin.",
        })

    out_dir = paper5 / "method_contract" / "06d1_v2"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(
            f"06d1_v2 output directory already exists and is non-empty: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    status_csv = out_dir / "v1_shared_coordinate_technical_status_v2.csv"
    with status_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(coordinates[0].keys()))
        writer.writeheader()
        writer.writerows(coordinates)

    report = {
        "status": "06D1_V2_V1_POSTFREEZE_TECHNICAL_STATUS_AUDIT_COMPLETE",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_representation": {
            "model_id": EXPECTED["model_id"],
            "profile_id": EXPECTED["profile_id"],
            "primary_seed": EXPECTED["primary_seed"],
            "shared_dim": EXPECTED["shared_dim"],
            "representation_sha256": (
                results_lock.get("result_files", {})
                .get("target_v1_patient_shared_seed1103.npy", {})
                .get("sha256")
            ),
        },
        "06d1_v1_failed_audit_boundary": {
            "script_sha256":
                "e91978672e17be2bc620b1903a02374f05e1f957291fc1875ee0b2472fb34c0d",
            "script_version":
                "06d1-audit-v1-postfreeze-technical-status-v1-no-cli",
            "failure_type": "RuntimeError",
            "failure_message":
                "04d technical metadata were used for model selection.",
            "cause":
                "v1 incorrectly looked for technical_covariates_used_for_model_selection "
                "at the top level of the 04d manifest; the frozen 04d schema stores it "
                "under scientific_state.",
            "failure_before_06d1_output_directory_creation": True,
            "target_molecular_matrices_read": False,
            "biological_annotations_read": False,
            "clinical_or_outcome_values_read": False,
            "target_model_fit_or_refit": False,
            "scientific_consequence": "NONE",
        },
        "06d0_scope_boundary": {
            "generic_target_resampling_stability": stability_scope,
            "target_pairing_permutations": pairing_scope,
            "q10_060_for_v1": "FORBIDDEN",
            "q10_040_for_v1": "FORBIDDEN",
            "sable_mo_stability_gate_for_v1": "FORBIDDEN",
        },
        "technical_metadata_preflight": {
            "contract_version": tech_lock.get("contract_version"),
            "status": tech_lock.get("status"),
            "strict_usable_technical_covariates": len(strict_rows),
            "ambiguous_specimen_technical_covariates": len(ambiguous_rows),
            "potential_cross_view_batch_candidates": manifest_n_cross_view,
            "technical_status_if_no_strict_usable_fields":
                manifest_no_usable_rule,
            "strict_inventory": [
                {
                    "modality": r.get(modality_col, "") if modality_col else "",
                    "covariate_id": r.get(covariate_col, ""),
                    "eligibility_status": r.get(eligibility_col, ""),
                }
                for r in strict_rows
            ],
            "ambiguous_inventory": [
                {
                    "modality": r.get(modality_col, "") if modality_col else "",
                    "covariate_id": r.get(covariate_col, ""),
                    "eligibility_status": r.get(eligibility_col, ""),
                }
                for r in ambiguous_rows
            ],
            "generated_files_verified": len(checked_04d),
        },
        "technical_association_family": {
            "performed": False,
            "reason":
                "No strict usable technical covariate existed in the prospectively frozen 04d registry.",
            "posthoc_covariate_creation": False,
            "identifier_use_as_covariate": False,
            "raw_timestamp_use_as_covariate": False,
        },
        "technical_status_rule": {
            "all_v1_shared_coordinates":
                "UNKNOWN_NO_STRICT_USABLE_TECHNICAL_COVARIATES",
            "unknown_means_clean": False,
            "unknown_means_biological": False,
            "technical_status_may_change_selected_model": False,
            "technical_status_may_change_primary_seed": False,
            "technical_status_may_apply_sable_retention_to_v1": False,
        },
        "scientific_state": {
            "target_molecular_matrices_read": False,
            "biological_annotations_read": False,
            "clinical_or_outcome_values_read": False,
            "target_model_fit_or_refit": False,
            "technical_association_test_performed": False,
            "representation_selection_changed": False,
            "representation_seed_changed": False,
            "06b_06c_06c3_rewritten": False,
        },
        "next_stage_boundary": {
            "technical_status_is_frozen_before_biology": True,
            "biology_may_not_rescue_or_reselect_v1_coordinates": True,
            "factor_level_biology_cannot_claim_stability_qualification": True,
            "recommended_next_stage":
                "freeze outcome-blind biological-alignment/random-control contract before opening biological annotations",
        },
        "verified_04d_generated_files": checked_04d,
    }

    report_path = out_dir / "v1_postfreeze_technical_status_audit_v2.json"
    write_json(report_path, report)

    lock_payload = {
        "status": "06D1_V2_V1_POSTFREEZE_TECHNICAL_STATUS_AUDIT_FROZEN",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_report": {
            "path": rel(report_path, paper5),
            "sha256": sha256_file(report_path),
            "size_bytes": report_path.stat().st_size,
        },
        "coordinate_status_table": {
            "path": rel(status_csv, paper5),
            "sha256": sha256_file(status_csv),
            "size_bytes": status_csv.stat().st_size,
        },
        "selected_model": EXPECTED["model_id"],
        "primary_seed": EXPECTED["primary_seed"],
        "strict_usable_technical_covariates": len(strict_rows),
        "technical_association_test_performed": False,
        "technical_status":
            "UNKNOWN_NO_STRICT_USABLE_TECHNICAL_COVARIATES",
        "q10_060_for_v1": "FORBIDDEN",
        "q10_040_for_v1": "FORBIDDEN",
        "sable_mo_stability_gate_for_v1": "FORBIDDEN",
        "target_molecular_matrices_read": False,
        "biological_annotations_read": False,
        "clinical_or_outcome_values_read": False,
        "target_model_fit_or_refit": False,
        "06b_06c_06c3_rewritten": False,
    }

    lock_path = out_dir / "06d1_v2_v1_postfreeze_technical_status_audit_lock.json"
    write_json(lock_path, lock_payload)

    print("Exact frozen 06c summary identity: PASS")
    print("Exact frozen 06c3 lock identity: PASS")
    print("06d0 V1 downstream-scope boundary: PASS")
    print("Exact frozen 04d manifest identity: PASS")
    print("04d nested scientific_state firewall: PASS")
    print("04d nested resolved_covariates zero-usable boundary: PASS")
    print("Exact frozen 04d resolved registry identity: PASS")
    print("04d generated-file lock replay: PASS")
    print("Strict usable technical covariates:", len(strict_rows))
    print("Ambiguous specimen/site technical covariates:", len(ambiguous_rows))
    print("Technical association tests performed: NO")
    print(
        "Frozen V1 technical status:",
        "UNKNOWN_NO_STRICT_USABLE_TECHNICAL_COVARIATES",
    )
    print("Technical UNKNOWN interpreted as non-technical/biological: NO")
    print("TARGET molecular matrices read in 06d1: NO")
    print("Biological annotations read in 06d1: NO")
    print("Clinical/outcome values read in 06d1: NO")
    print("TARGET model fit/refit in 06d1: NO")
    print("=" * 104)
    print("Paper 5 06d1_v2 V1 post-freeze technical-status audit: FROZEN")
    print("=" * 104)
    print("Report:", rel(report_path, paper5))
    print("Coordinate status:", rel(status_csv, paper5))
    print("Lock:", rel(lock_path, paper5))
    print(
        "Next: freeze the outcome-blind biological-alignment/random-control "
        "contract before opening biological annotations."
    )
    print("=" * 104)


if __name__ == "__main__":
    main()
