#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_VERSION = "06a-freeze-target-application-contract-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]
MC = ROOT / "method_contract"

F05F3E = MC / "05f3e" / "final_gate_results_lock.json"
F05Y6_RESULT = MC / "05y6" / "prospective_q10_validation_result.json"
F05Y6_LOCK = MC / "05y6" / "05y6_results_lock.json"
F05Y7_LOCK = MC / "05y7" / "05y7_manuscript_assets_lock.json"
F05Y8_LOCK = MC / "05y8" / "05y8_bytecode_archive_lock.json"

EXPECTED_05F3E_SHA256 = "6d691237662d633a4f8d8294126a33b32c5694372a6e58de71bd45731d67c397"
EXPECTED_05Y6_RESULT_SHA256 = "daa11e17a0a273d45b33c6c3ab14a2ae1bf6dc559bb722b18816a906e7f86052"
EXPECTED_05Y6_LOCK_SHA256 = "a93aeebcdcb07bd654096b5926ec3aa7b23b9b8a4bf51835991a843473ac21b1"

OUT = MC / "06a"
CONTRACT = OUT / "target_application_contract.json"
README = OUT / "06a_README.txt"
LOCK = OUT / "06a_contract_lock.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen artifact missing: {path}")


def require_sha(path: Path, expected: str) -> str:
    observed = sha256_file(path)
    if observed.lower() != expected.lower():
        raise RuntimeError(
            f"SHA256 mismatch for {path}: observed={observed}, expected={expected}"
        )
    return observed


def main() -> None:
    print("=" * 100)
    print("Paper 5 - freeze pre-TARGET application contract")
    print("=" * 100)
    print(f"Script version: {SCRIPT_VERSION}")
    print("TARGET data read: NO")
    print("TARGET outcomes read: NO")
    print("TARGET model fit: NO")
    print("Threshold selection/retuning: NO")
    print()

    for p in [F05F3E, F05Y6_RESULT, F05Y6_LOCK, F05Y7_LOCK, F05Y8_LOCK]:
        require_file(p)

    sha_05f3e = require_sha(F05F3E, EXPECTED_05F3E_SHA256)
    sha_05y6_result = require_sha(F05Y6_RESULT, EXPECTED_05Y6_RESULT_SHA256)
    sha_05y6_lock = require_sha(F05Y6_LOCK, EXPECTED_05Y6_LOCK_SHA256)

    y6 = json.loads(F05Y6_RESULT.read_text(encoding="utf-8"))
    y6_lock = json.loads(F05Y6_LOCK.read_text(encoding="utf-8"))

    if y6["status"] != "PROSPECTIVE_Q10_RECALIBRATION_RESULT_FROZEN":
        raise RuntimeError("Unexpected 05y6 result status.")
    if not y6["prospective_confirmation"]["candidate_confirmed"]:
        raise RuntimeError("05y6 candidate is not prospectively confirmed.")
    if y6["prospective_confirmation"]["formal_05f3e_gate_consequence"] != "NONE":
        raise RuntimeError("05y6 unexpectedly changes the historical 05f3e gate.")
    if y6_lock["threshold_retuning_after_fresh_result"] is not False:
        raise RuntimeError("05y6 lock does not forbid post-result retuning as expected.")

    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"06a output directory already exists and is non-empty: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    primary_rule = {
        "role": "PRIMARY_TARGET_RETENTION_RULE",
        "provenance": "single q10 recalibration prospectively confirmed in frozen 05y6 fresh-seed validation",
        "inclusion_frequency_min": 0.8,
        "loading_energy_fraction_per_supported_modality_min": 0.15,
        "median_aligned_correlation_min": 0.7,
        "n_supported_modalities_min": 2,
        "q10_aligned_correlation_min": 0.6,
        "reference_factor_variance_min": 0.0001,
    }
    sensitivity_rule = {
        "role": "PRESPECIFIED_HISTORICAL_SENSITIVITY_ONLY",
        "provenance": "historical frozen 05f3e rule",
        "inclusion_frequency_min": 0.8,
        "loading_energy_fraction_per_supported_modality_min": 0.15,
        "median_aligned_correlation_min": 0.7,
        "n_supported_modalities_min": 2,
        "q10_aligned_correlation_min": 0.4,
        "reference_factor_variance_min": 0.0001,
    }

    contract = {
        "analysis_role": "PRE_TARGET_APPLICATION_CONTRACT",
        "contract_version": "paper5-06a-target-application-v1",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_BEFORE_TARGET_DATA_MODEL_FIT_OR_OUTCOME_INSPECTION",
        "scientific_rationale": {
            "primary_rule_choice": (
                "q10=0.60 is primary because it was the single post-discovery recalibration "
                "frozen before a fresh independent-seed validation bank and was prospectively confirmed in 05y6."
            ),
            "historical_rule_role": (
                "q10=0.40 remains a prespecified historical sensitivity analysis and never replaces "
                "the primary q10=0.60 result based on TARGET behavior."
            ),
            "scope_limit": (
                "q10=0.60 is a targeted prospectively validated recalibration, not a rerun or replacement "
                "of the complete historical 05f3e simulation gate."
            ),
        },
        "primary_retention_rule": primary_rule,
        "historical_sensitivity_rule": sensitivity_rule,
        "target_application_decision_policy": {
            "primary_rule_always_executed_first_and_reported": True,
            "historical_sensitivity_also_executed": True,
            "sensitivity_may_replace_primary": False,
            "threshold_relaxation_after_target_results": False,
            "alternative_frontier_point_selection_after_target_results": False,
            "outcome_guided_threshold_selection": False,
            "rule_if_primary_retains_zero_factors": (
                "Report zero retained factors as the primary result. Do not relax q10 or substitute q10=0.40. "
                "Any downstream factor-specific analysis requiring at least one retained factor is reported as "
                "not estimable under the primary rule. Run q10=0.40 only as the separately labeled prespecified sensitivity."
            ),
            "rule_if_primary_retains_few_nonzero_factors": (
                "Do not invent a minimum retained-factor count and do not relax thresholds. "
                "Use all factors retained by the frozen primary rule for analyses that are mathematically and "
                "methodologically defined under the downstream frozen plan; otherwise report the corresponding "
                "analysis as not estimable. The q10=0.40 analysis remains sensitivity only."
            ),
            "rule_if_primary_and_sensitivity_disagree": (
                "Report both results with their prespecified labels. Do not select whichever rule yields the more favorable "
                "biological or outcome result; primary interpretation follows q10=0.60."
            ),
        },
        "historical_gate_boundaries": {
            "formal_05f3e_gate_remains_historical": True,
            "D04_and_D06_and_other_historical_gate_decisions_reestimated_for_q10_060": False,
            "claim_for_q10_060": (
                "Only the targeted fresh G19 specificity endpoint and fresh STANDARD truth-factor recovery safety endpoint "
                "were prospectively evaluated in 05y6."
            ),
            "D05_interpretation": {
                "historical_restriction_carries_forward": True,
                "historical_numeric_basis": "67.5% conditional mixed-support localization under the historical q10=0.40 gate",
                "numeric_basis_reestimated_for_q10_060": False,
                "allowed_wording": (
                    "Mixed/view-specific support attribution remains conservatively interpreted because its quantitative "
                    "simulation basis was established under the historical q10=0.40 gate and was not recharacterized for q10=0.60."
                ),
            },
        },
        "target_data_and_outcome_firewall": {
            "target_expression_or_multiomics_read_at_06a": False,
            "target_clinical_outcomes_read_at_06a": False,
            "target_model_fit_at_06a": False,
            "target_sample_count_used_to_choose_rule": False,
            "target_result_may_modify_thresholds": False,
        },
        "frozen_sources": {
            "05f3e_historical_gate_lock": {
                "path": str(F05F3E.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha_05f3e,
                "size_bytes": F05F3E.stat().st_size,
            },
            "05y6_prospective_result": {
                "path": str(F05Y6_RESULT.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha_05y6_result,
                "size_bytes": F05Y6_RESULT.stat().st_size,
            },
            "05y6_results_lock": {
                "path": str(F05Y6_LOCK.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha_05y6_lock,
                "size_bytes": F05Y6_LOCK.stat().st_size,
                "rewrite_policy": y6_lock["rewrite_policy"],
            },
            "05y7_manuscript_assets_lock": {
                "path": str(F05Y7_LOCK.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(F05Y7_LOCK),
                "size_bytes": F05Y7_LOCK.stat().st_size,
            },
            "05y8_bytecode_provenance_lock": {
                "path": str(F05Y8_LOCK.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(F05Y8_LOCK),
                "size_bytes": F05Y8_LOCK.stat().st_size,
            },
        },
        "rewrite_policy": (
            "06a is immutable after any TARGET model fit, retained-factor result, biological annotation result, "
            "or clinical outcome result is viewed. q10=0.60 remains primary and q10=0.40 remains sensitivity regardless "
            "of which produces a more favorable TARGET result."
        ),
    }

    CONTRACT.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    README.write_text(
        "Paper 5 06a - frozen pre-TARGET application contract\n"
        "=====================================================\n\n"
        "Primary TARGET retention rule: q10=0.60 with all other frozen retention criteria unchanged.\n"
        "Prespecified historical sensitivity: q10=0.40.\n\n"
        "The sensitivity rule is never a fallback or replacement for the primary rule.\n"
        "If q10=0.60 retains zero factors, zero is reported as the primary result and factor-specific downstream "
        "analyses are not estimable under the primary rule; q10=0.40 is still reported only as sensitivity.\n\n"
        "Historical 05f3e gate results remain historical q10=0.40 results. The q10=0.60 recalibration was prospectively "
        "validated only for the targeted G19 specificity and STANDARD truth-factor recovery safety endpoints.\n"
        "D05's quantitative basis was not re-estimated under q10=0.60.\n\n"
        "No TARGET data, TARGET outcomes, or TARGET model fit are accessed by 06a.\n",
        encoding="utf-8",
    )

    self_path = Path(__file__).resolve()
    lock = {
        "status": "06A_TARGET_APPLICATION_CONTRACT_FROZEN",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "path": str(CONTRACT.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(CONTRACT),
            "size_bytes": CONTRACT.stat().st_size,
        },
        "readme": {
            "path": str(README.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(README),
            "size_bytes": README.stat().st_size,
        },
        "script_source": {
            "path": str(self_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(self_path),
            "size_bytes": self_path.stat().st_size,
        },
        "primary_q10": 0.6,
        "historical_sensitivity_q10": 0.4,
        "sensitivity_is_fallback": False,
        "target_data_read": False,
        "target_outcomes_read": False,
        "target_fit_performed": False,
        "threshold_retuning_after_target": False,
    }
    LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    print("06a TARGET application contract: FROZEN")
    print("Primary q10:                  0.60")
    print("Historical sensitivity q10:   0.40")
    print("Sensitivity fallback:          NO")
    print("TARGET data/outcomes read:     NO")
    print("TARGET model fit:              NO")
    print(f"Contract: {CONTRACT.relative_to(ROOT)}")
    print(f"Lock:     {LOCK.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
