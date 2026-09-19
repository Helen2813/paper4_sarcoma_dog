#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_VERSION = "05y7-freeze-manuscript-q10-recalibration-assets-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]
MC = ROOT / "method_contract"

IN_DIR = MC / "05y6"
RESULT = IN_DIR / "prospective_q10_validation_result.json"
LOCK = IN_DIR / "05y6_results_lock.json"
AUDIT_DIR = MC / "05y6_preunblinding_audit"

OUT_DIR = MC / "05y7"
SUMMARY_JSON = OUT_DIR / "manuscript_q10_recalibration_summary.json"
SUMMARY_CSV = OUT_DIR / "manuscript_q10_recalibration_summary.csv"
TEXT_TXT = OUT_DIR / "manuscript_q10_recalibration_text.txt"
LOCK_JSON = OUT_DIR / "05y7_manuscript_assets_lock.json"
README = OUT_DIR / "05y7_README.txt"

EXPECTED_05Y6_RESULT_SHA256 = "daa11e17a0a273d45b33c6c3ab14a2ae1bf6dc559bb722b18816a906e7f86052"
EXPECTED_05Y6_LOCK_SHA256 = "a93aeebcdcb07bd654096b5926ec3aa7b23b9b8a4bf51835991a843473ac21b1"
EXPECTED_05Y6_AGGREGATOR_SHA256 = "bad3ee4833918b578c76daa013ee8e92de72f945c0f2b976db7dc75873ce39e8"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")


def require_equal(label: str, observed, expected) -> None:
    if observed != expected:
        raise RuntimeError(f"{label} mismatch: observed={observed!r}, expected={expected!r}")


def pct(x: float, digits: int = 3) -> str:
    return f"{100.0 * x:.{digits}f}%"


def pp(x: float, digits: int = 1) -> str:
    return f"{100.0 * x:.{digits}f} percentage points"


def main() -> None:
    print("=" * 100)
    print("Paper 5 - freeze manuscript-facing prospective q10 recalibration assets")
    print("=" * 100)
    print(f"Script version: {SCRIPT_VERSION}")
    print("Scientific endpoint recomputation from shards: FORBIDDEN")
    print("Threshold retuning: FORBIDDEN")
    print()

    require_file(RESULT)
    require_file(LOCK)

    if OUT_DIR.exists():
        contents = list(OUT_DIR.iterdir())
        if contents:
            raise RuntimeError(
                f"05y7 output directory already exists and is non-empty: {OUT_DIR}. "
                "Do not overwrite manuscript-facing frozen assets."
            )
    else:
        OUT_DIR.mkdir(parents=True)

    result_sha = sha256_file(RESULT)
    lock_sha = sha256_file(LOCK)
    require_equal("05y6 result SHA256", result_sha, EXPECTED_05Y6_RESULT_SHA256)
    require_equal("05y6 lock SHA256", lock_sha, EXPECTED_05Y6_LOCK_SHA256)

    result = json.loads(RESULT.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))

    require_equal("05y6 status", result["status"], "PROSPECTIVE_Q10_RECALIBRATION_RESULT_FROZEN")
    require_equal("05y6 lock status", lock["status"], "PROSPECTIVE_Q10_RECALIBRATION_RESULT_FROZEN")
    require_equal("candidate confirmed in result", result["prospective_confirmation"]["candidate_confirmed"], True)
    require_equal("candidate confirmed in lock", lock["candidate_confirmed"], True)
    require_equal("formal 05f3e consequence", result["prospective_confirmation"]["formal_05f3e_gate_consequence"], "NONE")
    require_equal("05y6 aggregator SHA256", lock["script_sha256"], EXPECTED_05Y6_AGGREGATOR_SHA256)
    require_equal("retuning after result", result["prospective_confirmation"]["retuning_after_result"], "FORBIDDEN")
    require_equal("threshold retuning after fresh result", lock["threshold_retuning_after_fresh_result"], False)

    require_equal("only original q10", result["original_rule"]["q10_aligned_correlation_min"], 0.4)
    require_equal("only candidate q10", result["candidate_rule"]["q10_aligned_correlation_min"], 0.6)

    other_fields = [
        "inclusion_frequency_min",
        "loading_energy_fraction_per_supported_modality_min",
        "median_aligned_correlation_min",
        "n_supported_modalities_min",
        "reference_factor_variance_min",
    ]
    for field in other_fields:
        require_equal(
            f"unchanged rule component {field}",
            result["candidate_rule"][field],
            result["original_rule"][field],
        )

    require_equal("fresh G19 datasets", result["fresh_bank"]["G19_datasets"], 200)
    require_equal("fresh STANDARD datasets", result["fresh_bank"]["STANDARD_datasets"], 200)
    require_equal("fresh STANDARD truth opportunities", result["fresh_bank"]["STANDARD_truth_factor_opportunities"], 800)
    require_equal("fresh seed overlap", result["fresh_bank"]["seed_overlap_with_prior_bank"], 0)

    primary = result["primary_specificity_endpoint"]
    safety = result["sensitivity_safety_endpoint"]

    require_equal("G19 original denominator", primary["original_false_acceptance_denominator"], 200)
    require_equal("G19 candidate denominator", primary["candidate_false_acceptance_denominator"], 200)
    require_equal("reverse discordance", primary["reverse_discordance"], 0)
    require_equal("required direction met", primary["required_direction_met"], True)
    require_equal("STANDARD denominator", safety["denominator"], 800)
    require_equal("D02 safety floor", safety["existing_D02_floor"], 0.8)
    require_equal("D02 safety floor met", safety["floor_met"], True)
    require_equal("candidate confirmation", result["prospective_confirmation"]["candidate_confirmed"], True)

    summary = {
        "analysis_role": "MANUSCRIPT_FACING_LOCKED_SUMMARY_OF_05Y6_PROSPECTIVE_Q10_RECALIBRATION",
        "script_version": SCRIPT_VERSION,
        "source_status": result["status"],
        "candidate_confirmed": True,
        "formal_05f3e_gate_consequence": "NONE",
        "retuning_after_result": "FORBIDDEN",
        "change_tested": {
            "field": "q10_aligned_correlation_min",
            "original": 0.4,
            "candidate": 0.6,
            "all_other_rule_components_unchanged": True,
        },
        "fresh_validation_bank": {
            "G19_correlated_private_null_datasets": 200,
            "STANDARD_true_shared_datasets": 200,
            "STANDARD_truth_factor_opportunities": 800,
            "seed_overlap_with_prior_bank": 0,
        },
        "specificity": {
            "original_false_acceptance_n": primary["original_false_acceptance_n"],
            "original_false_acceptance_denominator": primary["original_false_acceptance_denominator"],
            "original_false_acceptance_rate": primary["original_false_acceptance_rate"],
            "original_wilson95": primary["original_wilson95"],
            "candidate_false_acceptance_n": primary["candidate_false_acceptance_n"],
            "candidate_false_acceptance_denominator": primary["candidate_false_acceptance_denominator"],
            "candidate_false_acceptance_rate": primary["candidate_false_acceptance_rate"],
            "candidate_wilson95": primary["candidate_wilson95"],
            "absolute_rate_reduction": primary["absolute_rate_reduction"],
            "paired_bootstrap95_absolute_reduction": primary["paired_bootstrap95_absolute_reduction"],
            "relative_rate_reduction": primary["relative_rate_reduction"],
            "rescued_datasets": primary["rescued_datasets"],
            "reverse_discordance": primary["reverse_discordance"],
            "exact_two_sided_nested_mcnemar_p": primary["exact_two_sided_nested_mcnemar_p"],
        },
        "sensitivity_safety": {
            "original_recovered_n": safety["original_recovered_n"],
            "candidate_recovered_n": safety["candidate_recovered_n"],
            "denominator": safety["denominator"],
            "original_recovery_rate": safety["original_recovery_rate"],
            "candidate_recovery_rate": safety["candidate_recovery_rate"],
            "candidate_wilson95": safety["candidate_wilson95"],
            "lost_truth_opportunities": safety["lost_truth_opportunities"],
            "gained_truth_opportunities": safety["gained_truth_opportunities"],
            "existing_D02_floor": safety["existing_D02_floor"],
            "floor_met": safety["floor_met"],
        },
        "source_provenance": {
            "05y6_result": {
                "path": str(RESULT.relative_to(ROOT)).replace("\\", "/"),
                "sha256": result_sha,
                "size_bytes": RESULT.stat().st_size,
            },
            "05y6_lock": {
                "path": str(LOCK.relative_to(ROOT)).replace("\\", "/"),
                "sha256": lock_sha,
                "size_bytes": LOCK.stat().st_size,
            },
            "05y6_frozen_aggregator_sha256": EXPECTED_05Y6_AGGREGATOR_SHA256,
        },
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    rows = [
        ("G19 original false acceptance", primary["original_false_acceptance_n"], primary["original_false_acceptance_denominator"], primary["original_false_acceptance_rate"]),
        ("G19 q10=.60 false acceptance", primary["candidate_false_acceptance_n"], primary["candidate_false_acceptance_denominator"], primary["candidate_false_acceptance_rate"]),
        ("STANDARD original biological recovery", safety["original_recovered_n"], safety["denominator"], safety["original_recovery_rate"]),
        ("STANDARD q10=.60 biological recovery", safety["candidate_recovered_n"], safety["denominator"], safety["candidate_recovery_rate"]),
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["endpoint", "numerator", "denominator", "rate"])
        w.writerows(rows)

    methods_text = (
        "Post-discovery recalibration was evaluated prospectively under a frozen contract. "
        "The only modified retention criterion was the lower-tail stability threshold "
        "(q10 aligned correlation, 0.40 to 0.60); all other retention criteria were unchanged. "
        "The single candidate was evaluated on 400 independently seeded fresh datasets "
        "(200 correlated-private null stress datasets and 200 true-shared STANDARD datasets), "
        "with zero simulation-seed overlap with prior frozen banks. The prespecified confirmation "
        "rule required an exact paired specificity improvement at p<0.05 in the required direction, "
        "zero reverse discordance under the nested rules, and candidate STANDARD biological recovery "
        "of at least 0.80. No further threshold retuning was permitted after fresh-result unblinding."
    )

    results_text = (
        f"In the fresh correlated-private null stress set, false acceptance decreased from "
        f"{primary['original_false_acceptance_n']}/{primary['original_false_acceptance_denominator']} "
        f"({pct(primary['original_false_acceptance_rate'], 1)}) under q10=0.40 to "
        f"{primary['candidate_false_acceptance_n']}/{primary['candidate_false_acceptance_denominator']} "
        f"({pct(primary['candidate_false_acceptance_rate'], 1)}) under q10=0.60. "
        f"The paired absolute reduction was {pp(primary['absolute_rate_reduction'], 1)} "
        f"(95% bootstrap CI {100*primary['paired_bootstrap95_absolute_reduction'][0]:.1f} to "
        f"{100*primary['paired_bootstrap95_absolute_reduction'][1]:.1f} percentage points), "
        f"with {primary['rescued_datasets']} rescued datasets, "
        f"{primary['reverse_discordance']} reverse-discordant datasets, and an exact paired "
        f"two-sided McNemar/binomial p value of {primary['exact_two_sided_nested_mcnemar_p']:.3g}. "
        f"In the true-shared safety set, biological recovery changed from "
        f"{safety['original_recovered_n']}/{safety['denominator']} "
        f"({pct(safety['original_recovery_rate'], 3)}) to "
        f"{safety['candidate_recovered_n']}/{safety['denominator']} "
        f"({pct(safety['candidate_recovery_rate'], 3)}), with "
        f"{safety['lost_truth_opportunities']} lost and {safety['gained_truth_opportunities']} gained "
        f"truth-factor opportunities. Candidate recovery remained above the prespecified 0.80 safety floor. "
        f"The prospective q10=0.60 recalibration therefore met all prespecified confirmation criteria. "
        f"This prospective recalibration does not alter the historical formal 05f3e gate result."
    )

    discussion_text = (
        "The prospective result supports q10=0.60 as a separately validated post-discovery recalibration "
        "for the identified correlated-private specificity failure mode. The improvement was obtained without "
        "material loss of true-shared recovery in the prespecified safety condition. The original 05f3e formal "
        "gate remains unchanged and should be reported as the historical frozen result; the 05y3-05y6 sequence "
        "should be described explicitly as post-discovery recalibration followed by independent prospective confirmation."
    )

    TEXT_TXT.write_text(
        "METHODS\n" + methods_text + "\n\n"
        "RESULTS\n" + results_text + "\n\n"
        "DISCUSSION / INTERPRETATION\n" + discussion_text + "\n",
        encoding="utf-8",
    )

    README.write_text(
        "Paper 5 05y7 - manuscript-facing prospective q10 recalibration assets\n"
        "=====================================================================\n\n"
        "05y7 does not recompute scientific endpoints and does not read 05y5 shards.\n"
        "It validates the frozen 05y6 result/lock hashes and emits manuscript-facing assets only.\n\n"
        "Authoritative scientific source:\n"
        "  method_contract/05y6/prospective_q10_validation_result.json\n\n"
        "Interpretation:\n"
        "  q10=.60 prospective recalibration CONFIRMED under the frozen 05y3 rule.\n"
        "  historical formal 05f3e gate consequence: NONE.\n"
        "  retuning after the fresh result: FORBIDDEN.\n",
        encoding="utf-8",
    )

    generated = {}
    for path in [SUMMARY_JSON, SUMMARY_CSV, TEXT_TXT, README]:
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        generated[rel] = {
            "path": rel,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    self_path = Path(__file__).resolve()
    lock_out = {
        "status": "05Y7_MANUSCRIPT_Q10_RECALIBRATION_ASSETS_FROZEN",
        "script_version": SCRIPT_VERSION,
        "script_source": {
            "path": str(self_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(self_path),
            "size_bytes": self_path.stat().st_size,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_endpoint_recomputation": False,
        "threshold_retuning": False,
        "candidate_confirmed": True,
        "formal_05f3e_gate_consequence": "NONE",
        "source_files": {
            "05y6_result": {
                "path": str(RESULT.relative_to(ROOT)).replace("\\", "/"),
                "sha256": result_sha,
                "size_bytes": RESULT.stat().st_size,
            },
            "05y6_results_lock": {
                "path": str(LOCK.relative_to(ROOT)).replace("\\", "/"),
                "sha256": lock_sha,
                "size_bytes": LOCK.stat().st_size,
            },
        },
        "generated_files": generated,
        "rewrite_policy": (
            "05y7 manuscript-facing assets are immutable summaries of the frozen 05y6 result. "
            "Do not select an alternative threshold or modify endpoint values using the 05y5 fresh bank."
        ),
    }
    LOCK_JSON.write_text(json.dumps(lock_out, indent=2) + "\n", encoding="utf-8")

    print("05y7 manuscript assets: COMPLETE")
    print()
    print(f"Candidate confirmed:                  {summary['candidate_confirmed']}")
    print(f"Formal 05f3e gate consequence:        {summary['formal_05f3e_gate_consequence']}")
    print(
        "G19 false acceptance:                 "
        f"{primary['original_false_acceptance_n']}/200 -> {primary['candidate_false_acceptance_n']}/200"
    )
    print(
        "STANDARD biological recovery:         "
        f"{safety['original_recovered_n']}/800 -> {safety['candidate_recovered_n']}/800"
    )
    print(f"Output: {OUT_DIR.relative_to(ROOT)}")
    print(f"Lock:   {LOCK_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
