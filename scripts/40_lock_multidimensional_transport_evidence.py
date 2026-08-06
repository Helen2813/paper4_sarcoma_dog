from __future__ import annotations

from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_VERSION = "40-lock-multidimensional-transport-evidence-v1"

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

MODULES = ["M34", "M11", "M24", "M40"]

FILES = {
    "outcome": TABLES / "paper4_assay_aware_locked_module_evidence_summary.csv",
    "multiplicity": TABLES / "paper4_projectwide_primary_multiplicity.csv",
    "assay": TABLES / "paper4_gse39055_assay_stability_classification.csv",
    "structure": TABLES / "cross_cohort_conservative_preservation_classification.csv",
    "targeted_mofa": TABLES / "multigroup_mofa_module_representation_interpretation.csv",
    "variable_mofa": TABLES / "multigroup_mofa_variable_only_representation_interpretation.csv",
    "variable_rank": TABLES / "multigroup_mofa_variable_only_module_capture_rank_summary.csv",
    "variable_factor": TABLES / "multigroup_mofa_variable_only_module_factor_rank_summary.csv",
    "targeted_manifest": TABLES / "multigroup_mofa_alignment_manifest.json",
    "variable_manifest": TABLES / "multigroup_mofa_variable_only_alignment_manifest.json",
}

OUT_AI = TABLES / "paper4_locked_ai_representation_evidence.csv"
OUT_MASTER = TABLES / "paper4_locked_multidimensional_transport_evidence.csv"
OUT_TYPOLOGY = TABLES / "paper4_locked_multidimensional_transport_typology.csv"
OUT_SENTENCES = TABLES / "paper4_locked_multidimensional_results_sentences.txt"
OUT_CLAIMS = TABLES / "paper4_locked_ai_novelty_claims.txt"
OUT_README = TABLES / "paper4_locked_multidimensional_README.txt"
OUT_MANIFEST = TABLES / "paper4_locked_multidimensional_manifest.json"
OUT_FIG_PNG = FIGURES / "paper4_multidimensional_transport_heatmap.png"
OUT_FIG_PDF = FIGURES / "paper4_multidimensional_transport_heatmap.pdf"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"Loaded: {path}")
    return pd.read_csv(path)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def rank_value(table: pd.DataFrame, analysis_set: str, module: str, subspace: str, column: str) -> float:
    part = table[
        table["analysis_set"].eq(analysis_set)
        & table["module_label"].eq(module)
        & table["subspace_name"].eq(subspace)
    ]
    return float(part.iloc[0][column]) if not part.empty else np.nan


def factor_value(table: pd.DataFrame, analysis_set: str, module: str, column: str) -> float:
    part = table[
        table["analysis_set"].eq(analysis_set)
        & table["module_label"].eq(module)
    ]
    return float(part.iloc[0][column]) if not part.empty else np.nan


def structure_summary(table: pd.DataFrame, module: str) -> tuple[int, int, int, str]:
    part = table[table["module_label"].eq(module)]
    strong = int(part["conservative_preservation_class"].eq(
        "strong_cross_cohort_representation_preservation"
    ).sum())
    partial = int(part["conservative_preservation_class"].eq(
        "partial_cross_cohort_representation_preservation"
    ).sum())
    limited = int(len(part) - strong - partial)
    if strong >= 2:
        label = "strong_multicohort_structure_preservation"
    elif strong + partial >= 2:
        label = "partial_multicohort_structure_preservation"
    else:
        label = "limited_structure_preservation"
    return strong, partial, limited, label


def structure_score(label: str) -> float:
    return {
        "strong_multicohort_structure_preservation": 1.0,
        "partial_multicohort_structure_preservation": 0.6,
        "limited_structure_preservation": 0.2,
    }[label]


def latent_score(label: str) -> float:
    return {
        "independent_ubiquitous_latent_recurrence": 1.0,
        "independent_non_ffpe_recurrence_with_detection_aware_restoration": 0.8,
        "independent_non_ffpe_latent_recurrence": 0.7,
        "limited_or_no_independent_latent_recurrence": 0.2,
        "not_interpretable_low_natural_coverage": np.nan,
    }.get(label, np.nan)


def interpretation(module: str) -> str:
    return {
        "M34": (
            "M34 independently recurred as an ubiquitous latent transcriptional representation "
            "in a variable-only ortholog space selected without frozen-module membership. "
            "Nevertheless, prognostic transport remained endpoint- and platform-dependent: "
            "GSE21257 provided the only project-wide FDR-controlled human outcome result, "
            "TARGET-OS showed nominal directional concordance, and GSE39055 retained an "
            "assay-limited discordant RFS direction."
        ),
        "M40": (
            "M40 independently recurred across DOG2, TARGET-OS, and GSE21257, was attenuated "
            "in unfiltered GSE39055, and re-emerged after detection-aware filtering. Its "
            "GSE39055 outcome direction remained assay-rule sensitive, showing that recoverable "
            "shared representation did not imply stable prognostic transport."
        ),
        "M11": (
            "M11 retained the canine risk direction across human outcome settings and every "
            "estimable GSE39055 assay rule, yet it did not emerge as a dominant independently "
            "learned latent representation. Its statistically unusual best-factor alignments "
            "were small in absolute magnitude."
        ),
        "M24": (
            "M24 lacked sufficient natural gene coverage for a reliable variable-only latent-"
            "recurrence test and showed limited structure and outcome transport outside GSE21257."
        ),
    }[module]


def integrated_class(module: str) -> str:
    return {
        "M34": "shared_representation_with_endpoint_and_platform_dependent_outcome_transport",
        "M40": "assay_sensitive_representation_with_unstable_prognostic_transport",
        "M11": "directional_outcome_concordance_without_dominant_latent_representation",
        "M24": "limited_representation_and_endpoint_specific_outcome_evidence",
    }[module]


def main() -> None:
    print("=" * 80)
    print("Lock multidimensional cross-species transport evidence")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {ROOT}")
    print()

    outcome = load_csv(FILES["outcome"])
    multiplicity = load_csv(FILES["multiplicity"])
    assay = load_csv(FILES["assay"])
    structure = load_csv(FILES["structure"])
    targeted = load_csv(FILES["targeted_mofa"])
    variable = load_csv(FILES["variable_mofa"])
    rank = load_csv(FILES["variable_rank"])
    factor = load_csv(FILES["variable_factor"])

    targeted_manifest = load_json(FILES["targeted_manifest"])
    variable_manifest = load_json(FILES["variable_manifest"])
    if targeted_manifest.get("outcome_loaded", True):
        raise RuntimeError("Targeted MOFA manifest does not confirm outcome_loaded=false.")
    if variable_manifest.get("outcome_loaded", True):
        raise RuntimeError("Variable-only MOFA manifest does not confirm outcome_loaded=false.")
    if variable_manifest.get("selection_used_frozen_membership", True):
        raise RuntimeError("Variable-only MOFA manifest does not confirm frozen-membership-free selection.")

    core = "four_cohort_variable_only_1500"
    detect = "four_cohort_detection_aware_variable_only_700"
    no_ffpe = "three_cohort_no_ffpe_variable_only_1500"

    ai_rows = []
    master_rows = []
    typology_rows = []

    for module in MODULES:
        variable_row = variable[variable["module_label"].eq(module)].iloc[0]
        targeted_row = targeted[targeted["module_label"].eq(module)].iloc[0]
        outcome_row = outcome[outcome["module_label"].eq(module)].iloc[0]
        assay_row = assay[assay["module_label"].eq(module)].iloc[0]
        mult = multiplicity[multiplicity["module_label"].eq(module)]
        strong, partial, limited, structure_class = structure_summary(structure, module)

        ai_row = {
            "module_label": module,
            "targeted_mofa_representation_class": targeted_row["mofa_representation_class"],
            "variable_only_representation_class": variable_row["variable_only_representation_class"],
            "core_natural_coverage_fraction": variable_row["core_natural_coverage_fraction"],
            "core_ubiquitous_median_capture": rank_value(rank, core, module, "ubiquitous_all_groups", "median_capture"),
            "core_non_ffpe_median_capture": rank_value(rank, core, module, "shared_non_ffpe_only", "median_capture"),
            "detection_aware_ubiquitous_median_capture": rank_value(rank, detect, module, "ubiquitous_all_groups", "median_capture"),
            "no_ffpe_ubiquitous_median_capture": rank_value(rank, no_ffpe, module, "ubiquitous_all_groups", "median_capture"),
            "core_median_max_absolute_factor_cosine": factor_value(factor, core, module, "median_max_absolute_cosine"),
            "core_fraction_ranks_best_factor_global_q_lt_0_05": factor_value(factor, core, module, "fraction_ranks_global_q_lt_0_05"),
        }
        ai_rows.append(ai_row)

        latent_class = str(variable_row["variable_only_representation_class"])
        outcome_class = str(outcome_row["assay_aware_locked_evidence_grade"])
        assay_class = str(assay_row["gse39055_assay_stability_class"])

        outcome_score = {"M34": 1.0, "M11": 0.6, "M24": 0.3, "M40": 0.3}[module]
        assay_score = {"M34": 0.5, "M11": 1.0, "M24": np.nan, "M40": 0.4}[module]

        master_rows.append({
            "module_label": module,
            "n_strong_structure_settings": strong,
            "n_partial_structure_settings": partial,
            "n_limited_structure_settings": limited,
            "structure_evidence_class": structure_class,
            "variable_only_latent_class": latent_class,
            "targeted_mofa_representation_class": targeted_row["mofa_representation_class"],
            "core_natural_coverage_fraction": ai_row["core_natural_coverage_fraction"],
            "core_ubiquitous_median_capture": ai_row["core_ubiquitous_median_capture"],
            "core_non_ffpe_median_capture": ai_row["core_non_ffpe_median_capture"],
            "detection_aware_ubiquitous_median_capture": ai_row["detection_aware_ubiquitous_median_capture"],
            "no_ffpe_ubiquitous_median_capture": ai_row["no_ffpe_ubiquitous_median_capture"],
            "core_median_max_absolute_factor_cosine": ai_row["core_median_max_absolute_factor_cosine"],
            "n_projectwide_fdr_supported_human_settings": int(mult["projectwide_fdr_supported"].sum()),
            "n_nominal_supported_human_settings": int(mult["nominal_supported"].sum()),
            "minimum_projectwide_q_12": float(mult["projectwide_q_12"].min()),
            "outcome_evidence_class": outcome_class,
            "gse39055_assay_stability_class": assay_class,
            "multidimensional_transport_class": integrated_class(module),
            "locked_multidimensional_interpretation": interpretation(module),
            "structure_evidence_score": structure_score(structure_class),
            "independent_latent_recurrence_score": latent_score(latent_class),
            "outcome_transport_evidence_score": outcome_score,
            "gse39055_assay_stability_score": assay_score,
        })

        typology_rows.append({
            "module_label": module,
            "network_structure": structure_class,
            "independent_latent_recurrence": latent_class,
            "outcome_transport": outcome_class,
            "gse39055_assay_stability": assay_class,
            "integrated_typology": integrated_class(module),
        })

    ai = pd.DataFrame(ai_rows)
    master = pd.DataFrame(master_rows)
    typology = pd.DataFrame(typology_rows)

    ai.to_csv(OUT_AI, index=False)
    master.to_csv(OUT_MASTER, index=False)
    typology.to_csv(OUT_TYPOLOGY, index=False)

    lines = [
        "Locked multidimensional transport results",
        "========================================",
        "",
        "Primary conceptual result",
        "-------------------------",
        (
            "Outcome-blind multi-group MOFA2 identified stable latent transcriptional "
            "representations across canine and human osteosarcoma cohorts. Variable-only "
            "models confirmed that the principal M34- and M40-like representations were not "
            "artifacts of forced frozen-gene inclusion. Latent recurrence, network preservation, "
            "assay stability, and prognostic transport nevertheless separated into distinct "
            "module-specific patterns."
        ),
        "",
    ]
    for module in MODULES:
        lines += [module, "-" * len(module), interpretation(module), ""]
    lines += [
        "Terminology guardrail",
        "---------------------",
        (
            "Independent latent recurrence means independent of frozen-module feature selection "
            "and clinical outcomes within the same expression cohorts. It does not mean independent "
            "external patient-cohort validation."
        ),
    ]
    OUT_SENTENCES.write_text("\n".join(lines), encoding="utf-8")

    OUT_CLAIMS.write_text(
        """Recommended AI/ML novelty claims
================================

Preferred abstract-level claim
------------------------------
We used outcome-blind multi-group MOFA2 to distinguish ubiquitous, partially
shared, and assay-sensitive latent transcriptional representations across canine
and human osteosarcoma cohorts. Variable-only models selected without frozen-
program membership independently recovered M34- and M40-like latent
representations, while demonstrating that latent recurrence, network
preservation, and prognostic transport were separable properties.

Claims to avoid
---------------
- The modules were independently validated in new patients.
- MOFA causally identified prognostic programs.
- MOFA outperformed PCA.
- All four canine programs were conserved in humans.
- GSE39055 proved a biological reversal of M40.
""",
        encoding="utf-8",
    )

    OUT_README.write_text(
        f"""Paper 4 multidimensional evidence lock
Script version: {SCRIPT_VERSION}

This script integrates locked outcome evidence, conservative structure
preservation, GSE39055 assay stability, targeted MOFA, and variable-only MOFA.
No new factor or outcome model is fitted.

The numerical heatmap scores are descriptive display scores only. They are not
pooled effect sizes, posterior probabilities, or formal meta-analysis results.
""",
        encoding="utf-8",
    )

    matrix = master.set_index("module_label").reindex(MODULES)[[
        "structure_evidence_score",
        "independent_latent_recurrence_score",
        "outcome_transport_evidence_score",
        "gse39055_assay_stability_score",
    ]]
    labels = ["Network\nstructure", "Independent\nlatent recurrence", "Outcome\ntransport", "GSE39055 assay\nstability"]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(MODULES)))
    ax.set_yticklabels(MODULES)
    ax.set_title("Multidimensional cross-species transport evidence")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            ax.text(j, i, "NA" if not np.isfinite(value) else f"{value:.1f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT_FIG_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG_PDF, bbox_inches="tight")
    plt.close(fig)

    output_paths = [OUT_AI, OUT_MASTER, OUT_TYPOLOGY, OUT_SENTENCES, OUT_CLAIMS, OUT_README, OUT_FIG_PNG, OUT_FIG_PDF]
    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now(),
        "new_model_fitting": False,
        "guardrails": [
            "No new outcome or factor model was fitted.",
            "Variable-only feature selection did not use frozen-module membership.",
            "Independent latent recurrence is not external patient-cohort validation.",
            "Heatmap scores are descriptive only.",
            "No causal claim is made.",
        ],
        "inputs": {str(path): {"sha256": sha256(path), "size_bytes": path.stat().st_size} for path in FILES.values()},
        "outputs": {str(path): {"sha256": sha256(path), "size_bytes": path.stat().st_size} for path in output_paths},
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 80)
    print("Locked multidimensional evidence")
    print("=" * 80)
    print(master[[
        "module_label",
        "structure_evidence_class",
        "variable_only_latent_class",
        "outcome_evidence_class",
        "gse39055_assay_stability_class",
        "multidimensional_transport_class",
    ]].to_string(index=False))
    print()
    print("=" * 80)
    print("Key AI/ML metrics")
    print("=" * 80)
    print(ai[[
        "module_label",
        "core_natural_coverage_fraction",
        "core_ubiquitous_median_capture",
        "core_non_ffpe_median_capture",
        "detection_aware_ubiquitous_median_capture",
        "no_ffpe_ubiquitous_median_capture",
        "core_median_max_absolute_factor_cosine",
    ]].to_string(index=False))
    print()
    print("Saved:")
    for path in output_paths + [OUT_MANIFEST]:
        print(path)
    print("Done.")


if __name__ == "__main__":
    main()
