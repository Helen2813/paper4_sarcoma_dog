from __future__ import annotations

from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_VERSION = "29-lock-cross-species-evidence-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]
HUMAN_COHORTS = ["TARGET_OS", "GSE21257", "GSE39055"]

TARGET_FILE = RESULTS_DIR / "TARGET_OS_primary_frozen_program_validation.csv"
GSE21257_PRIMARY_FILE = (
    RESULTS_DIR / "GSE21257_metastasis_primary_frozen_program_validation.csv"
)
GSE21257_ROBUST_FILE = (
    RESULTS_DIR / "GSE21257_primary_robust_logistic_effects.csv"
)
GSE39055_FILE = RESULTS_DIR / "GSE39055_RFS_primary_frozen_program_validation.csv"
PRESERVATION_FILE = (
    RESULTS_DIR / "cross_cohort_conservative_preservation_classification.csv"
)
FROZEN_PROGRAM_FILE = (
    RESULTS_DIR / "GSE238110_frozen_canine_transfer_program_manifest.csv"
)

OUTPUT_OUTCOME_LONG = RESULTS_DIR / "paper4_locked_human_outcome_evidence.csv"
OUTPUT_STRUCTURE_LONG = RESULTS_DIR / "paper4_locked_structure_evidence.csv"
OUTPUT_SUMMARY = RESULTS_DIR / "paper4_locked_module_evidence_summary.csv"
OUTPUT_INTERPRETATION = RESULTS_DIR / "paper4_locked_module_interpretation.csv"
OUTPUT_SENTENCES = RESULTS_DIR / "paper4_locked_results_sentences.txt"
OUTPUT_MANIFEST = RESULTS_DIR / "paper4_analysis_freeze_manifest.json"
OUTPUT_README = RESULTS_DIR / "paper4_analysis_freeze_README.txt"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_required_csv(
    path: Path,
    index_col: int | str | None = None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    print(f"Loaded: {path}")
    return pd.read_csv(path, index_col=index_col)


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def classify_target(row: pd.Series) -> tuple[str, bool, bool]:
    hr = row.get("score_hr_per_sd", np.nan)
    p = row.get("primary_p", np.nan)
    q = row.get("q_within_endpoint", np.nan)

    concordant = finite(hr) and float(hr) > 1.0
    significant = (
        concordant
        and (
            (finite(q) and float(q) < 0.05)
            or (finite(p) and float(p) < 0.05)
        )
    )

    if finite(q) and float(q) < 0.05 and concordant:
        label = "endpoint_fdr_directional_support"
    elif finite(p) and float(p) < 0.05 and concordant:
        label = "nominal_directional_support"
    elif concordant:
        label = "directional_without_nominal_support"
    elif finite(hr) and float(hr) < 1.0:
        label = "direction_discordant"
    else:
        label = "not_estimable"

    return label, concordant, significant


def classify_gse21257(
    primary_row: pd.Series,
    robust_row: pd.Series,
) -> tuple[str, bool, bool]:
    odds_ratio = robust_row.get("or_per_sd", np.nan)
    p = primary_row.get("primary_p", np.nan)
    q_within = primary_row.get("q_within_endpoint", np.nan)
    q_global = primary_row.get("q_global_eight_tests", np.nan)

    concordant = finite(odds_ratio) and float(odds_ratio) > 1.0
    significant = (
        concordant
        and (
            (finite(q_global) and float(q_global) < 0.05)
            or (finite(q_within) and float(q_within) < 0.05)
            or (finite(p) and float(p) < 0.05)
        )
    )

    if finite(q_global) and float(q_global) < 0.05 and concordant:
        label = "global_fdr_directional_support"
    elif finite(q_within) and float(q_within) < 0.05 and concordant:
        label = "endpoint_fdr_directional_support"
    elif finite(p) and float(p) < 0.05 and concordant:
        label = "nominal_directional_support"
    elif concordant:
        label = "directional_without_nominal_support"
    elif finite(odds_ratio) and float(odds_ratio) < 1.0:
        label = "direction_discordant"
    else:
        label = "not_estimable"

    return label, concordant, significant


def classify_gse39055(row: pd.Series) -> tuple[str, bool, bool]:
    hr = row.get("hr_per_sd", np.nan)
    p = row.get("primary_p", np.nan)
    q = row.get("q_within_gse39055", np.nan)

    concordant = finite(hr) and float(hr) > 1.0
    significant = (
        concordant
        and (
            (finite(q) and float(q) < 0.05)
            or (finite(p) and float(p) < 0.05)
        )
    )

    if finite(q) and float(q) < 0.05 and concordant:
        label = "endpoint_fdr_directional_support"
    elif finite(p) and float(p) < 0.05 and concordant:
        label = "nominal_directional_support"
    elif concordant:
        label = "directional_without_nominal_support"
    elif finite(hr) and float(hr) < 1.0:
        label = "direction_discordant"
    else:
        label = "not_estimable"

    return label, concordant, significant


def build_outcome_table(
    target: pd.DataFrame,
    gse_primary: pd.DataFrame,
    gse_robust: pd.DataFrame,
    gse39055: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    target_indexed = target.set_index("module_label")
    gse_primary_indexed = gse_primary.set_index("module_label")
    gse_robust_indexed = gse_robust.set_index("module_label")
    gse39055_indexed = gse39055.set_index("module_label")

    for module in PRIMARY_MODULES:
        target_row = target_indexed.loc[module]
        label, concordant, significant = classify_target(target_row)
        rows.append(
            {
                "module_label": module,
                "cohort": "TARGET_OS",
                "endpoint": "overall_survival",
                "effect_type": "hazard_ratio_per_sd",
                "effect": target_row.get("score_hr_per_sd", np.nan),
                "ci_low": target_row.get("score_ci_low", np.nan),
                "ci_high": target_row.get("score_ci_high", np.nan),
                "primary_p": target_row.get("primary_p", np.nan),
                "endpoint_q": target_row.get("q_within_endpoint", np.nan),
                "global_q": target_row.get("q_global_eight_tests", np.nan),
                "discrimination_type": "fixed_direction_c_index",
                "discrimination": target_row.get("fixed_score_c_index", np.nan),
                "outcome_support_class": label,
                "direction_concordant": concordant,
                "statistically_supported": significant,
            }
        )

        gse_primary_row = gse_primary_indexed.loc[module]
        gse_robust_row = gse_robust_indexed.loc[module]
        label, concordant, significant = classify_gse21257(
            gse_primary_row,
            gse_robust_row,
        )
        rows.append(
            {
                "module_label": module,
                "cohort": "GSE21257",
                "endpoint": "metastasis_within_5y",
                "effect_type": "odds_ratio_per_sd",
                "effect": gse_robust_row.get("or_per_sd", np.nan),
                "ci_low": gse_robust_row.get("or_ci_low", np.nan),
                "ci_high": gse_robust_row.get("or_ci_high", np.nan),
                "primary_p": gse_primary_row.get("primary_p", np.nan),
                "endpoint_q": gse_primary_row.get("q_within_endpoint", np.nan),
                "global_q": gse_primary_row.get("q_global_eight_tests", np.nan),
                "discrimination_type": "auc",
                "discrimination": gse_primary_row.get("auc", np.nan),
                "outcome_support_class": label,
                "direction_concordant": concordant,
                "statistically_supported": significant,
            }
        )

        gse39055_row = gse39055_indexed.loc[module]
        label, concordant, significant = classify_gse39055(gse39055_row)
        rows.append(
            {
                "module_label": module,
                "cohort": "GSE39055",
                "endpoint": "recurrence_free_survival",
                "effect_type": "hazard_ratio_per_sd",
                "effect": gse39055_row.get("hr_per_sd", np.nan),
                "ci_low": gse39055_row.get("ci_low", np.nan),
                "ci_high": gse39055_row.get("ci_high", np.nan),
                "primary_p": gse39055_row.get("primary_p", np.nan),
                "endpoint_q": gse39055_row.get("q_within_gse39055", np.nan),
                "global_q": np.nan,
                "discrimination_type": "fixed_direction_c_index",
                "discrimination": gse39055_row.get(
                    "fixed_score_c_index",
                    np.nan,
                ),
                "outcome_support_class": label,
                "direction_concordant": concordant,
                "statistically_supported": significant,
            }
        )

    return pd.DataFrame(rows)


def structure_rank(value: str) -> int:
    ranks = {
        "strong_cross_cohort_representation_preservation": 3,
        "partial_cross_cohort_representation_preservation": 2,
        "limited_cross_cohort_representation_evidence": 1,
        "no_clear_cross_cohort_representation_preservation": 0,
    }
    return ranks.get(str(value), 0)


def module_grade(row: pd.Series) -> str:
    n_strong = int(row["n_strong_structure_settings"])
    n_partial_or_strong = int(row["n_partial_or_strong_structure_settings"])
    n_concordant = int(row["n_direction_concordant_settings"])
    n_supported = int(row["n_statistically_supported_settings"])
    gse39055_discordant = bool(row["gse39055_direction_discordant"])

    if (
        n_strong >= 2
        and n_concordant >= 2
        and n_supported >= 2
        and gse39055_discordant
    ):
        return "multi_setting_transfer_with_third_setting_heterogeneity"

    if (
        n_strong >= 3
        and n_concordant == 2
        and gse39055_discordant
    ):
        return "structure_preserved_but_outcome_heterogeneous"

    if n_concordant == 3 and n_supported <= 1:
        return "directionally_consistent_but_weakly_supported"

    if n_concordant <= 1:
        return "limited_or_inconsistent_cross_species_transfer"

    if n_partial_or_strong >= 2 and n_concordant >= 2:
        return "partial_cross_species_transfer"

    return "limited_or_inconsistent_cross_species_transfer"


def interpretation_for_grade(
    module: str,
    grade: str,
) -> str:
    if grade == "multi_setting_transfer_with_third_setting_heterogeneity":
        return (
            f"{module} showed strong representation and outcome transfer in "
            "TARGET-OS and GSE21257, but attenuated representation and an "
            "opposite recurrence-free-survival direction in GSE39055. It "
            "should be described as a replicated but cohort- and "
            "endpoint-heterogeneous cross-species program, not as a universal "
            "prognostic biomarker."
        )

    if grade == "structure_preserved_but_outcome_heterogeneous":
        return (
            f"{module} retained strong cross-cohort transcriptional structure "
            "in all three human settings, while its frozen prognostic direction "
            "was concordant in TARGET-OS and GSE21257 but discordant in "
            "GSE39055. This is direct evidence that representation "
            "preservation does not guarantee prognostic transport."
        )

    if grade == "directionally_consistent_but_weakly_supported":
        return (
            f"{module} retained the same risk direction in all three human "
            "settings, but statistical and structural support was limited or "
            "incomplete outside GSE21257. It should be treated as a secondary "
            "directional-consistency finding."
        )

    if grade == "partial_cross_species_transfer":
        return (
            f"{module} showed partial cross-species transfer, with at least "
            "two directionally concordant human settings but incomplete "
            "structural or statistical support."
        )

    return (
        f"{module} did not show sufficiently consistent structural and outcome "
        "evidence to support a general cross-species prognostic claim."
    )


def build_module_summary(
    outcome: pd.DataFrame,
    structure: pd.DataFrame,
) -> pd.DataFrame:
    structure = structure.copy()
    structure["structure_rank"] = structure[
        "conservative_preservation_class"
    ].map(structure_rank)

    rows: list[dict[str, Any]] = []

    for module in PRIMARY_MODULES:
        outcome_part = outcome[outcome["module_label"].eq(module)].copy()
        structure_part = structure[
            structure["module_label"].eq(module)
        ].copy()

        gse39055_outcome = outcome_part[
            outcome_part["cohort"].eq("GSE39055")
        ]
        gse39055_discordant = (
            not gse39055_outcome.empty
            and not bool(
                gse39055_outcome.iloc[0]["direction_concordant"]
            )
        )

        row = {
            "module_label": module,
            "n_human_settings": outcome_part.shape[0],
            "n_direction_concordant_settings": int(
                outcome_part["direction_concordant"].sum()
            ),
            "n_statistically_supported_settings": int(
                outcome_part["statistically_supported"].sum()
            ),
            "n_strong_structure_settings": int(
                (
                    structure_part["structure_rank"] == 3
                ).sum()
            ),
            "n_partial_or_strong_structure_settings": int(
                (
                    structure_part["structure_rank"] >= 2
                ).sum()
            ),
            "n_limited_or_better_structure_settings": int(
                (
                    structure_part["structure_rank"] >= 1
                ).sum()
            ),
            "gse39055_direction_discordant": gse39055_discordant,
            "target_outcome_class": outcome_part.loc[
                outcome_part["cohort"].eq("TARGET_OS"),
                "outcome_support_class",
            ].iloc[0],
            "gse21257_outcome_class": outcome_part.loc[
                outcome_part["cohort"].eq("GSE21257"),
                "outcome_support_class",
            ].iloc[0],
            "gse39055_outcome_class": outcome_part.loc[
                outcome_part["cohort"].eq("GSE39055"),
                "outcome_support_class",
            ].iloc[0],
            "target_structure_class": structure_part.loc[
                structure_part["cohort"].eq("TARGET_OS"),
                "conservative_preservation_class",
            ].iloc[0],
            "gse21257_structure_class": structure_part.loc[
                structure_part["cohort"].eq("GSE21257"),
                "conservative_preservation_class",
            ].iloc[0],
            "gse39055_structure_class": structure_part.loc[
                structure_part["cohort"].eq("GSE39055"),
                "conservative_preservation_class",
            ].iloc[0],
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["locked_evidence_grade"] = summary.apply(module_grade, axis=1)
    summary["locked_interpretation"] = summary.apply(
        lambda row: interpretation_for_grade(
            row["module_label"],
            row["locked_evidence_grade"],
        ),
        axis=1,
    )
    return summary


def write_results_sentences(
    summary: pd.DataFrame,
    outcome: pd.DataFrame,
    structure: pd.DataFrame,
) -> None:
    summary_indexed = summary.set_index("module_label")

    lines = [
        "Locked cross-species evidence statements",
        "=======================================",
        "",
        "These statements summarize distinct endpoints and are not a formal meta-analysis.",
        "",
    ]

    for module in PRIMARY_MODULES:
        lines.append(f"{module}")
        lines.append("-" * len(module))
        lines.append(summary_indexed.loc[module, "locked_interpretation"])
        lines.append("")

    lines.extend(
        [
            "Primary manuscript-level conclusion",
            "-----------------------------------",
            (
                "Frozen canine transcriptional programs showed heterogeneous "
                "human transfer. M34 combined strong representation and outcome "
                "support in TARGET-OS and GSE21257 with attenuated representation "
                "and opposite RFS direction in GSE39055. M40 retained strong "
                "transcriptional structure across all three human cohorts despite "
                "discordant RFS direction in GSE39055, demonstrating that "
                "cross-cohort representation preservation does not guarantee "
                "prognostic transport. M11 showed the most consistent direction "
                "of association but only limited or partial structural and "
                "statistical support. M24 did not support a general conserved "
                "prognostic claim."
            ),
            "",
            "Guardrail",
            "---------",
            (
                "Overall survival, five-year metastasis, and recurrence-free "
                "survival are not pooled into one effect estimate. Frozen score "
                "directions are not reversed after viewing human outcomes."
            ),
        ]
    )

    OUTPUT_SENTENCES.write_text("\n".join(lines), encoding="utf-8")


def create_manifest(input_paths: list[Path], output_paths: list[Path]) -> None:
    payload = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "analysis_status": "locked_descriptive_synthesis",
        "primary_modules": PRIMARY_MODULES,
        "human_settings": HUMAN_COHORTS,
        "guardrails": [
            "No formal meta-analysis across non-identical endpoints.",
            "No post hoc reversal of frozen score direction.",
            "No change to frozen module membership, weights, or validation tier.",
            "Script 28 conservative structure classes supersede script 27 classes for manuscript claims.",
        ],
        "inputs": {},
        "outputs": {},
    }

    for path in input_paths:
        payload["inputs"][path.name] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    for path in output_paths:
        if path.exists():
            payload["outputs"][path.name] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }

    OUTPUT_MANIFEST.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def write_readme() -> None:
    text = f"""Paper 4 locked cross-species evidence synthesis
Script version: {SCRIPT_VERSION}

Purpose
-------
This script performs no new feature selection, score orientation, model fitting,
or hypothesis testing. It freezes the interpretation of the human validation
and conservative structure-preservation analyses.

Evidence layers
---------------
1. TARGET-OS overall survival.
2. GSE21257 metastasis within five years.
3. GSE39055 recurrence-free survival.
4. Conservative canine-human representation preservation from script 28.

Important restriction
---------------------
The three human endpoints are scientifically related but not identical.
They are summarized by triangulation, not pooled into a formal meta-analysis.

Manuscript hierarchy
--------------------
- Script 23 provides the prespecified primary TARGET-OS and GSE21257 analyses.
- Script 24 provides robustness diagnostics.
- Script 26 provides the third-cohort GSE39055 RFS analysis.
- Script 28 provides manuscript-ready conservative representation classes.
- Script 29 locks the final descriptive evidence hierarchy.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("Lock Paper 4 cross-species evidence hierarchy")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Design:")
    print("  Perform no new outcome modelling or feature selection.")
    print("  Combine primary human outcome results with conservative structure classes.")
    print("  Keep OS, metastasis, and RFS as distinct external settings.")
    print("  Freeze module-level manuscript interpretations and input hashes.")
    print("")

    target = read_required_csv(TARGET_FILE)
    gse_primary = read_required_csv(GSE21257_PRIMARY_FILE)
    gse_robust = read_required_csv(GSE21257_ROBUST_FILE)
    gse39055 = read_required_csv(GSE39055_FILE)
    structure = read_required_csv(PRESERVATION_FILE)
    frozen_programs = read_required_csv(FROZEN_PROGRAM_FILE)

    outcome = build_outcome_table(
        target=target,
        gse_primary=gse_primary,
        gse_robust=gse_robust,
        gse39055=gse39055,
    )

    structure_locked = structure[
        structure["module_label"].isin(PRIMARY_MODULES)
        & structure["cohort"].isin(HUMAN_COHORTS)
    ].copy()

    summary = build_module_summary(
        outcome=outcome,
        structure=structure_locked,
    )

    interpretation = summary[
        [
            "module_label",
            "locked_evidence_grade",
            "locked_interpretation",
        ]
    ].copy()

    outcome.to_csv(OUTPUT_OUTCOME_LONG, index=False)
    structure_locked.to_csv(OUTPUT_STRUCTURE_LONG, index=False)
    summary.to_csv(OUTPUT_SUMMARY, index=False)
    interpretation.to_csv(OUTPUT_INTERPRETATION, index=False)

    write_results_sentences(
        summary=summary,
        outcome=outcome,
        structure=structure_locked,
    )
    write_readme()

    input_paths = [
        TARGET_FILE,
        GSE21257_PRIMARY_FILE,
        GSE21257_ROBUST_FILE,
        GSE39055_FILE,
        PRESERVATION_FILE,
        FROZEN_PROGRAM_FILE,
    ]
    output_paths = [
        OUTPUT_OUTCOME_LONG,
        OUTPUT_STRUCTURE_LONG,
        OUTPUT_SUMMARY,
        OUTPUT_INTERPRETATION,
        OUTPUT_SENTENCES,
        OUTPUT_README,
    ]
    create_manifest(input_paths, output_paths)

    print("")
    print("=" * 80)
    print("Locked module evidence summary")
    print("=" * 80)
    display_columns = [
        "module_label",
        "n_direction_concordant_settings",
        "n_statistically_supported_settings",
        "n_strong_structure_settings",
        "n_partial_or_strong_structure_settings",
        "target_outcome_class",
        "gse21257_outcome_class",
        "gse39055_outcome_class",
        "target_structure_class",
        "gse21257_structure_class",
        "gse39055_structure_class",
        "locked_evidence_grade",
    ]
    print(summary[display_columns].to_string(index=False))

    print("")
    print("=" * 80)
    print("Locked manuscript interpretations")
    print("=" * 80)
    for _, row in interpretation.iterrows():
        print(f"{row['module_label']}: {row['locked_interpretation']}")
        print("")

    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("No formal meta-analysis is performed across OS, metastasis, and RFS.")
    print("Frozen score directions are not reversed after viewing human outcomes.")
    print("Script 28 conservative structure classes supersede script 27 descriptive classes for manuscript claims.")
    print("No result changes module membership, gene weights, or validation tiers.")

    print("")
    print("Saved:")
    for path in [
        OUTPUT_OUTCOME_LONG,
        OUTPUT_STRUCTURE_LONG,
        OUTPUT_SUMMARY,
        OUTPUT_INTERPRETATION,
        OUTPUT_SENTENCES,
        OUTPUT_README,
        OUTPUT_MANIFEST,
    ]:
        print(path)
    print("Done.")


if __name__ == "__main__":
    main()
