from __future__ import annotations

from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_VERSION = "33-projectwide-multiplicity-assay-aware-lock-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]
HUMAN_COHORTS = ["TARGET_OS", "GSE21257", "GSE39055"]

TARGET_FILE = RESULTS_DIR / "TARGET_OS_primary_frozen_program_validation.csv"
GSE21257_FILE = (
    RESULTS_DIR / "GSE21257_metastasis_primary_frozen_program_validation.csv"
)
GSE39055_FILE = (
    RESULTS_DIR / "GSE39055_RFS_primary_frozen_program_validation.csv"
)
ASSAY_SUMMARY_FILE = (
    RESULTS_DIR / "GSE39055_assay_quality_module_summary.csv"
)
ASSAY_RFS_FILE = (
    RESULTS_DIR / "GSE39055_detection_aware_RFS_sensitivity.csv"
)
STRUCTURE_FILE = (
    RESULTS_DIR / "cross_cohort_conservative_preservation_classification.csv"
)
LOCKED_SUMMARY_FILE = (
    RESULTS_DIR / "paper4_locked_module_evidence_summary.csv"
)
PATKAR_OMNIBUS_FILE = (
    RESULTS_DIR / "Patkar_TME_module_omnibus_associations.csv"
)
PATKAR_TARGETED_FILE = (
    RESULTS_DIR / "Patkar_TME_targeted_convergence_tests.csv"
)
PATKAR_MEDIANS_FILE = (
    RESULTS_DIR / "Patkar_TME_subtype_score_summary.csv"
)

OUTPUT_MULTIPLICITY = (
    RESULTS_DIR / "paper4_projectwide_primary_multiplicity.csv"
)
OUTPUT_ASSAY = (
    RESULTS_DIR / "paper4_gse39055_assay_stability_classification.csv"
)
OUTPUT_TYPOLOGY = (
    RESULTS_DIR / "paper4_representation_outcome_decoupling_typology.csv"
)
OUTPUT_SUMMARY = (
    RESULTS_DIR / "paper4_assay_aware_locked_module_evidence_summary.csv"
)
OUTPUT_INTERPRETATION = (
    RESULTS_DIR / "paper4_assay_aware_locked_module_interpretation.csv"
)
OUTPUT_SENTENCES = (
    RESULTS_DIR / "paper4_assay_aware_locked_results_sentences.txt"
)
OUTPUT_README = (
    RESULTS_DIR / "paper4_assay_aware_lock_README.txt"
)
OUTPUT_MANIFEST = (
    RESULTS_DIR / "paper4_assay_aware_lock_manifest.json"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    print(f"Loaded: {path}")
    return pd.read_csv(path)


def bh_adjust(values: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q

    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    restored = np.empty(n, dtype=float)
    restored[order] = adjusted
    q[valid] = restored
    return q


def build_projectwide_multiplicity(
    target: pd.DataFrame,
    gse21257: pd.DataFrame,
    gse39055: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    target_index = target.set_index("module_label")
    gse_index = gse21257.set_index("module_label")
    rfs_index = gse39055.set_index("module_label")

    for module in PRIMARY_MODULES:
        rows.append(
            {
                "cohort": "TARGET_OS",
                "endpoint": "overall_survival",
                "module_label": module,
                "primary_p": target_index.loc[module, "primary_p"],
                "effect": target_index.loc[module, "score_hr_per_sd"],
                "effect_type": "hazard_ratio_per_sd",
                "direction_concordant": bool(
                    target_index.loc[module, "score_hr_per_sd"] > 1
                ),
            }
        )
        rows.append(
            {
                "cohort": "GSE21257",
                "endpoint": "metastasis_within_5y",
                "module_label": module,
                "primary_p": gse_index.loc[module, "primary_p"],
                "effect": gse_index.loc[module, "auc"],
                "effect_type": "auc",
                "direction_concordant": bool(
                    gse_index.loc[module, "auc"] > 0.5
                ),
            }
        )
        rows.append(
            {
                "cohort": "GSE39055",
                "endpoint": "recurrence_free_survival",
                "module_label": module,
                "primary_p": rfs_index.loc[module, "primary_p"],
                "effect": rfs_index.loc[module, "hr_per_sd"],
                "effect_type": "hazard_ratio_per_sd",
                "direction_concordant": bool(
                    rfs_index.loc[module, "hr_per_sd"] > 1
                ),
            }
        )

    result = pd.DataFrame(rows)
    result["projectwide_q_12"] = bh_adjust(result["primary_p"])
    result["projectwide_fdr_supported"] = (
        result["projectwide_q_12"] < 0.05
    )
    result["nominal_supported"] = result["primary_p"] < 0.05

    result = result.sort_values(
        ["projectwide_q_12", "primary_p", "cohort", "module_label"]
    ).reset_index(drop=True)
    return result


def coverage_class(value: float) -> str:
    if not np.isfinite(value):
        return "not_available"
    if value >= 0.70:
        return "adequate"
    if value >= 0.30:
        return "limited"
    return "poor"


def classify_assay_stability(
    assay_summary: pd.DataFrame,
    assay_rfs: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for module in PRIMARY_MODULES:
        summary_row = assay_summary[
            assay_summary["module_label"].eq(module)
        ].iloc[0]
        part = assay_rfs[
            assay_rfs["module_label"].eq(module)
        ].copy()

        valid = part[
            part["direction_concordant_with_canine"].notna()
        ].copy()
        n_strategies = valid.shape[0]
        n_concordant = int(
            valid["direction_concordant_with_canine"].astype(bool).sum()
        )
        fraction = (
            n_concordant / n_strategies
            if n_strategies
            else np.nan
        )

        filtered = part[
            part["strategy"].str.contains("p01_ge", regex=False)
        ].copy()
        n_filtered_estimable = filtered.shape[0]

        coverage_50 = float(
            summary_row["best_detected_50_coverage_fraction"]
        )
        coverage_80 = float(
            summary_row["best_detected_80_coverage_fraction"]
        )
        coverage_status = coverage_class(coverage_50)

        if n_filtered_estimable == 0:
            assay_class = "not_detection_filter_estimable"
        elif np.isfinite(fraction) and fraction == 1.0:
            assay_class = "stable_concordant_direction"
        elif np.isfinite(fraction) and fraction == 0.0:
            assay_class = "stable_discordant_direction"
        else:
            assay_class = "assay_rule_sensitive_direction"

        if assay_class == "stable_concordant_direction":
            interpretation = (
                "The canine risk direction was preserved under every "
                "estimable assay rule."
            )
        elif assay_class == "stable_discordant_direction":
            interpretation = (
                "The opposite GSE39055 direction persisted under every "
                "estimable assay rule, but interpretation remains constrained "
                f"by {coverage_status} detection-aware coverage."
            )
        elif assay_class == "assay_rule_sensitive_direction":
            interpretation = (
                "The GSE39055 direction changed across outcome-blind "
                "probe-detection rules and should not be presented as a "
                "stable biological reversal."
            )
        else:
            interpretation = (
                "Too few frozen genes remained after detection filtering to "
                "evaluate direction robustness."
            )

        rows.append(
            {
                "module_label": module,
                "locked_hr_per_sd": summary_row["locked_hr_per_sd"],
                "n_estimable_strategies": n_strategies,
                "n_filtered_estimable_strategies": n_filtered_estimable,
                "n_direction_concordant_strategies": n_concordant,
                "fraction_direction_concordant_strategies": fraction,
                "minimum_score_correlation_with_locked": summary_row[
                    "minimum_score_correlation_with_locked"
                ],
                "median_score_correlation_with_locked": summary_row[
                    "median_score_correlation_with_locked"
                ],
                "best_detected_50_coverage_fraction": coverage_50,
                "best_detected_80_coverage_fraction": coverage_80,
                "detection_coverage_class": coverage_status,
                "gse39055_assay_stability_class": assay_class,
                "assay_aware_interpretation": interpretation,
            }
        )

    return pd.DataFrame(rows)


def patkar_annotation(
    omnibus: pd.DataFrame,
    targeted: pd.DataFrame,
    medians: pd.DataFrame,
) -> pd.DataFrame:
    omnibus_primary = omnibus[
        omnibus["score_variant"].eq("strict_signed_mean_z")
    ].set_index("module_label")

    medians_primary = medians[
        medians["score_variant"].eq("strict_signed_mean_z")
    ].copy()
    median_matrix = medians_primary.pivot(
        index="module_label",
        columns="primary_immune_subtype",
        values="median",
    )

    targeted_primary = targeted[
        targeted["score_variant"].eq("strict_signed_mean_z")
    ].copy()

    rows: list[dict[str, Any]] = []
    for module in PRIMARY_MODULES:
        q = (
            omnibus_primary.loc[module, "permutation_kruskal_q_bh"]
            if module in omnibus_primary.index
            else np.nan
        )
        epsilon = (
            omnibus_primary.loc[module, "epsilon_squared"]
            if module in omnibus_primary.index
            else np.nan
        )

        median_id = (
            median_matrix.loc[module, "ID"]
            if module in median_matrix.index
            else np.nan
        )
        median_ie = (
            median_matrix.loc[module, "IE"]
            if module in median_matrix.index
            else np.nan
        )
        median_ie_ecm = (
            median_matrix.loc[module, "IE-ECM"]
            if module in median_matrix.index
            else np.nan
        )

        if module == "M34":
            label = "strong_immune_desert_risk_axis"
            note = (
                "M34 was substantially higher in immune-desert tumors and "
                "lower in immune-enriched tumors."
            )
        elif module == "M40" and np.isfinite(q) and q < 0.05:
            label = "exploratory_proliferation_immune_desert_axis"
            note = (
                "M40 was higher in immune-desert tumors; this contrast was "
                "exploratory."
            )
        elif module == "M11" and np.isfinite(q) and q < 0.05:
            label = "tme_subtype_associated_with_ie_ecm_trend"
            note = (
                "M11 differed across subtypes, with higher scores in IE-ECM, "
                "but the prespecified IE-ECM versus IE contrast was not "
                "statistically significant."
            )
        else:
            label = "no_clear_patkar_tme_convergence"
            note = "No clear association with the Patkar TME taxonomy."

        rows.append(
            {
                "module_label": module,
                "patkar_permutation_omnibus_q": q,
                "patkar_epsilon_squared": epsilon,
                "median_ID": median_id,
                "median_IE": median_ie,
                "median_IE_ECM": median_ie_ecm,
                "patkar_biological_annotation": label,
                "patkar_interpretation": note,
            }
        )

    return pd.DataFrame(rows)


def simplified_structure_class(value: str) -> str:
    mapping = {
        "strong_cross_cohort_representation_preservation": "preserved",
        "partial_cross_cohort_representation_preservation": "partial",
        "limited_cross_cohort_representation_evidence": "limited",
        "no_clear_cross_cohort_representation_preservation": "not_preserved",
    }
    return mapping.get(str(value), "unknown")


def build_typology(
    structure: pd.DataFrame,
    target: pd.DataFrame,
    gse21257: pd.DataFrame,
    gse39055: pd.DataFrame,
    assay: pd.DataFrame,
) -> pd.DataFrame:
    target_index = target.set_index("module_label")
    gse_index = gse21257.set_index("module_label")
    rfs_index = gse39055.set_index("module_label")
    assay_index = assay.set_index("module_label")

    rows: list[dict[str, Any]] = []

    for cohort in HUMAN_COHORTS:
        for module in PRIMARY_MODULES:
            structure_row = structure[
                structure["cohort"].eq(cohort)
                & structure["module_label"].eq(module)
            ].iloc[0]
            structure_simple = simplified_structure_class(
                structure_row["conservative_preservation_class"]
            )

            if cohort == "TARGET_OS":
                direction = bool(
                    target_index.loc[module, "score_hr_per_sd"] > 1
                )
                outcome_status = (
                    "nominal_or_fdr_supported"
                    if target_index.loc[module, "primary_p"] < 0.05
                    else "direction_only"
                )
                assay_status = "not_applicable"
            elif cohort == "GSE21257":
                direction = bool(
                    gse_index.loc[module, "auc"] > 0.5
                )
                outcome_status = (
                    "nominal_or_fdr_supported"
                    if gse_index.loc[module, "primary_p"] < 0.05
                    else "direction_only"
                )
                assay_status = "not_applicable"
            else:
                direction = bool(
                    rfs_index.loc[module, "hr_per_sd"] > 1
                )
                outcome_status = (
                    "nominal_or_fdr_supported"
                    if rfs_index.loc[module, "primary_p"] < 0.05
                    else "direction_only"
                )
                assay_status = assay_index.loc[
                    module,
                    "gse39055_assay_stability_class",
                ]

            if cohort == "GSE39055" and assay_status in {
                "assay_rule_sensitive_direction",
                "not_detection_filter_estimable",
            }:
                quadrant = "outcome_transport_not_stably_estimable"
            elif structure_simple in {"preserved", "partial"} and direction:
                quadrant = "representation_preserved_outcome_concordant"
            elif structure_simple in {"preserved", "partial"} and not direction:
                quadrant = "representation_outcome_decoupling"
            elif structure_simple in {"limited", "not_preserved"} and direction:
                quadrant = "directional_concordance_without_structure"
            else:
                quadrant = "representation_and_outcome_not_transported"

            rows.append(
                {
                    "cohort": cohort,
                    "module_label": module,
                    "conservative_structure_class": structure_row[
                        "conservative_preservation_class"
                    ],
                    "structure_class_simplified": structure_simple,
                    "outcome_direction_concordant": direction,
                    "outcome_support_status": outcome_status,
                    "assay_stability_status": assay_status,
                    "decoupling_typology_class": quadrant,
                }
            )

    return pd.DataFrame(rows)


def updated_grade(module: str) -> str:
    grades = {
        "M34": (
            "projectwide_fdr_support_in_one_setting_with_"
            "assay_limited_third_cohort_discordance"
        ),
        "M11": (
            "directionally_consistent_with_assay_robust_"
            "third_cohort_direction"
        ),
        "M24": (
            "endpoint_specific_support_with_third_cohort_"
            "not_detection_filter_estimable"
        ),
        "M40": (
            "structure_preserved_but_third_cohort_outcome_"
            "assay_rule_sensitive"
        ),
    }
    return grades[module]


def updated_interpretation(module: str) -> str:
    interpretations = {
        "M34": (
            "M34 retained strong representation and outcome support in "
            "TARGET-OS and GSE21257. It was the only program with "
            "project-wide FDR-controlled human outcome support, through the "
            "GSE21257 metastasis analysis. Its opposite GSE39055 RFS direction "
            "persisted across detection-aware scoring rules, but stringent "
            "probe filtering retained only limited module coverage; the third "
            "cohort should therefore be presented as assay-limited "
            "heterogeneity rather than definitive biological reversal."
        ),
        "M11": (
            "M11 retained the canine risk direction in all three human "
            "settings and under every estimable GSE39055 assay rule. A "
            "detection-aware GSE39055 sensitivity score was nominally "
            "associated with RFS, but this alternative assay rule was not a "
            "primary test. M11 remains a secondary directional-consistency "
            "finding."
        ),
        "M24": (
            "M24 showed endpoint-specific support in GSE21257 and discordant "
            "directions in the locked TARGET-OS and GSE39055 analyses. "
            "However, only one of seven frozen genes passed the principal "
            "GSE39055 detection filter, so the third-cohort direction is not "
            "considered a stable assay-quality-tested result."
        ),
        "M40": (
            "M40 retained conservative transcriptional-structure preservation "
            "across all three human cohorts. Its locked GSE39055 RFS direction "
            "was opposite to the canine direction, but detection-aware scores "
            "changed direction across probe-quality rules. M40 therefore "
            "supports separation of representation preservation from stable "
            "prognostic transport, but not a definitive biological reversal "
            "in GSE39055."
        ),
    }
    return interpretations[module]


def build_updated_summary(
    locked_summary: pd.DataFrame,
    multiplicity: pd.DataFrame,
    assay: pd.DataFrame,
    patkar: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for module in PRIMARY_MODULES:
        locked_row = locked_summary[
            locked_summary["module_label"].eq(module)
        ].iloc[0]
        assay_row = assay[
            assay["module_label"].eq(module)
        ].iloc[0]
        patkar_row = patkar[
            patkar["module_label"].eq(module)
        ].iloc[0]
        mult_part = multiplicity[
            multiplicity["module_label"].eq(module)
        ]

        rows.append(
            {
                "module_label": module,
                "n_nominal_or_fdr_supported_human_settings": int(
                    mult_part["nominal_supported"].sum()
                ),
                "n_projectwide_fdr_supported_human_settings": int(
                    mult_part["projectwide_fdr_supported"].sum()
                ),
                "minimum_projectwide_q_12": float(
                    mult_part["projectwide_q_12"].min()
                ),
                "n_direction_concordant_settings_locked": locked_row[
                    "n_direction_concordant_settings"
                ],
                "n_strong_structure_settings": locked_row[
                    "n_strong_structure_settings"
                ],
                "gse39055_assay_stability_class": assay_row[
                    "gse39055_assay_stability_class"
                ],
                "gse39055_detection_coverage_class": assay_row[
                    "detection_coverage_class"
                ],
                "patkar_biological_annotation": patkar_row[
                    "patkar_biological_annotation"
                ],
                "assay_aware_locked_evidence_grade": updated_grade(module),
                "assay_aware_locked_interpretation": updated_interpretation(
                    module
                ),
            }
        )

    return pd.DataFrame(rows)


def write_sentences(summary: pd.DataFrame) -> None:
    indexed = summary.set_index("module_label")

    lines = [
        "Assay-aware locked cross-species evidence statements",
        "===================================================",
        "",
        "Multiplicity statement",
        "----------------------",
        (
            "Across all 12 frozen primary human outcome tests, only the "
            "GSE21257 M34 metastasis association remained significant after "
            "project-wide Benjamini-Hochberg correction."
        ),
        "",
    ]

    for module in PRIMARY_MODULES:
        lines.append(module)
        lines.append("-" * len(module))
        lines.append(
            indexed.loc[
                module,
                "assay_aware_locked_interpretation",
            ]
        )
        lines.append("")

    lines.extend(
        [
            "Biological convergence",
            "-----------------------",
            (
                "Within DOG2, the frozen risk-oriented M34 score showed strong "
                "same-cohort orthogonal-method convergence with the Patkar "
                "tumor-microenvironment taxonomy: scores were highest in "
                "immune-desert tumors and markedly lower in immune-enriched "
                "tumors. M40 also differed strongly across TME subtypes, with "
                "higher scores in immune-desert tumors, whereas M24 showed no "
                "clear TME-subtype association."
            ),
            "",
            "Primary conceptual conclusion",
            "-----------------------------",
            (
                "Cross-species transcriptional representation and prognostic "
                "transport are separable quantities. M34 showed the strongest "
                "outcome evidence but third-cohort assay-limited heterogeneity. "
                "M40 preserved transcriptional structure while its third-cohort "
                "outcome direction was unstable across assay-quality rules. "
                "Thus, preserved representation did not guarantee a stable "
                "transported prognostic effect."
            ),
        ]
    )

    OUTPUT_SENTENCES.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_readme() -> None:
    text = f"""Paper 4 project-wide multiplicity and assay-aware evidence lock
Script version: {SCRIPT_VERSION}

Purpose
-------
1. Apply one Benjamini-Hochberg correction across all 12 frozen primary human
   outcome tests: four modules in TARGET-OS, GSE21257, and GSE39055.
2. Integrate the outcome-blind GSE39055 Detection P-value diagnostic.
3. Add same-cohort Patkar TME-subtype biological convergence annotations.
4. Update the representation-outcome decoupling typology without changing any
   frozen program, score orientation, primary outcome model, or raw result.

Interpretation hierarchy
------------------------
- Script 23 remains the prespecified TARGET-OS and GSE21257 primary analysis.
- Script 26 remains the locked GSE39055 primary RFS analysis.
- Script 31 determines whether GSE39055 direction is robust to assay rules.
- Script 28 remains the conservative structure-preservation analysis.
- Script 32 is biological convergence within overlapping DOG2 samples, not
  independent validation.
- Script 33 updates manuscript wording and multiplicity accounting only.

Important restriction
---------------------
A diagnostic detection-aware sensitivity cannot replace a locked primary
analysis. However, assay-rule-sensitive or non-estimable direction should not
be presented as definitive biological heterogeneity.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def create_manifest(
    input_paths: list[Path],
    output_paths: list[Path],
) -> None:
    payload = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "primary_test_count": 12,
        "multiplicity_method": "Benjamini-Hochberg across all 12 frozen primary human outcome tests",
        "guardrails": [
            "No new feature selection or outcome model fitting.",
            "No frozen score reversal.",
            "No replacement of scripts 23 or 26 primary analyses.",
            "Detection-aware analyses modify interpretation, not primary results.",
            "Patkar convergence uses overlapping DOG2 data and is not external validation.",
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


def main() -> None:
    print("=" * 80)
    print("Project-wide multiplicity and assay-aware Paper 4 evidence lock")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Design:")
    print("  Correct all 12 frozen primary human tests together.")
    print("  Classify GSE39055 direction stability across detection-aware rules.")
    print("  Integrate conservative structure preservation and Patkar TME convergence.")
    print("  Update manuscript interpretation without changing primary results.")
    print("")

    target = read_required_csv(TARGET_FILE)
    gse21257 = read_required_csv(GSE21257_FILE)
    gse39055 = read_required_csv(GSE39055_FILE)
    assay_summary = read_required_csv(ASSAY_SUMMARY_FILE)
    assay_rfs = read_required_csv(ASSAY_RFS_FILE)
    structure = read_required_csv(STRUCTURE_FILE)
    locked_summary = read_required_csv(LOCKED_SUMMARY_FILE)
    patkar_omnibus = read_required_csv(PATKAR_OMNIBUS_FILE)
    patkar_targeted = read_required_csv(PATKAR_TARGETED_FILE)
    patkar_medians = read_required_csv(PATKAR_MEDIANS_FILE)

    multiplicity = build_projectwide_multiplicity(
        target=target,
        gse21257=gse21257,
        gse39055=gse39055,
    )
    multiplicity.to_csv(OUTPUT_MULTIPLICITY, index=False)

    assay = classify_assay_stability(
        assay_summary=assay_summary,
        assay_rfs=assay_rfs,
    )
    assay.to_csv(OUTPUT_ASSAY, index=False)

    patkar = patkar_annotation(
        omnibus=patkar_omnibus,
        targeted=patkar_targeted,
        medians=patkar_medians,
    )

    typology = build_typology(
        structure=structure,
        target=target,
        gse21257=gse21257,
        gse39055=gse39055,
        assay=assay,
    )
    typology.to_csv(OUTPUT_TYPOLOGY, index=False)

    summary = build_updated_summary(
        locked_summary=locked_summary,
        multiplicity=multiplicity,
        assay=assay,
        patkar=patkar,
    )
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    summary[
        [
            "module_label",
            "assay_aware_locked_evidence_grade",
            "assay_aware_locked_interpretation",
        ]
    ].to_csv(OUTPUT_INTERPRETATION, index=False)

    write_sentences(summary)
    write_readme()

    input_paths = [
        TARGET_FILE,
        GSE21257_FILE,
        GSE39055_FILE,
        ASSAY_SUMMARY_FILE,
        ASSAY_RFS_FILE,
        STRUCTURE_FILE,
        LOCKED_SUMMARY_FILE,
        PATKAR_OMNIBUS_FILE,
        PATKAR_TARGETED_FILE,
        PATKAR_MEDIANS_FILE,
    ]
    output_paths = [
        OUTPUT_MULTIPLICITY,
        OUTPUT_ASSAY,
        OUTPUT_TYPOLOGY,
        OUTPUT_SUMMARY,
        OUTPUT_INTERPRETATION,
        OUTPUT_SENTENCES,
        OUTPUT_README,
    ]
    create_manifest(input_paths, output_paths)

    print("")
    print("=" * 80)
    print("Project-wide multiplicity across 12 primary human tests")
    print("=" * 80)
    print(
        multiplicity[
            [
                "cohort",
                "endpoint",
                "module_label",
                "primary_p",
                "projectwide_q_12",
                "nominal_supported",
                "projectwide_fdr_supported",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("GSE39055 assay-aware direction classification")
    print("=" * 80)
    print(
        assay[
            [
                "module_label",
                "n_estimable_strategies",
                "n_filtered_estimable_strategies",
                "fraction_direction_concordant_strategies",
                "best_detected_50_coverage_fraction",
                "detection_coverage_class",
                "gse39055_assay_stability_class",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Assay-aware locked module evidence summary")
    print("=" * 80)
    print(
        summary[
            [
                "module_label",
                "n_nominal_or_fdr_supported_human_settings",
                "n_projectwide_fdr_supported_human_settings",
                "minimum_projectwide_q_12",
                "gse39055_assay_stability_class",
                "patkar_biological_annotation",
                "assay_aware_locked_evidence_grade",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Representation-outcome decoupling typology")
    print("=" * 80)
    print(
        typology[
            [
                "cohort",
                "module_label",
                "structure_class_simplified",
                "outcome_direction_concordant",
                "assay_stability_status",
                "decoupling_typology_class",
            ]
        ].sort_values(
            ["module_label", "cohort"]
        ).to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("Only GSE21257 M34 may be described as project-wide FDR supported.")
    print("M40 GSE39055 direction is assay-rule sensitive and is not a definitive biological reversal.")
    print("M24 GSE39055 is not adequately estimable after detection filtering.")
    print("M34 GSE39055 remains discordant across rules, but stringent filtering leaves limited coverage.")
    print("Patkar results are same-cohort biological convergence, not external validation.")

    print("")
    print("Saved:")
    for path in output_paths + [OUTPUT_MANIFEST]:
        print(path)
    print("Done.")


if __name__ == "__main__":
    main()
