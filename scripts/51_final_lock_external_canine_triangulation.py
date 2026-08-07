from __future__ import annotations

from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "51-final-lock-external-canine-triangulation-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SCRIPT47_FILE = RESULTS_DIR / "paper4_locked_independent_canine_representation.csv"
SCRIPT47_MANIFEST = RESULTS_DIR / "paper4_external_canine_evidence_manifest.json"

SCRIPT49_FILE = RESULTS_DIR / "GSE239948_blind_frozen_program_rediscovery.csv"
SCRIPT49_MANIFEST = RESULTS_DIR / "GSE239948_blind_de_novo_rediscovery_manifest.json"

SCRIPT50_FILE = RESULTS_DIR / "GSE239948_WGCNA_module_preservation_signed.csv"
SCRIPT50_CONCORDANCE_FILE = RESULTS_DIR / "paper4_GSE239948_preservation_method_concordance.csv"
SCRIPT50_MANIFEST = RESULTS_DIR / "GSE239948_WGCNA_module_preservation_manifest.json"

NECROSIS_PRIMARY_FILE = RESULTS_DIR / "GSE39055_necrosis_primary_exploratory.csv"

MASTER_CANDIDATES = [
    RESULTS_DIR / "paper4_locked_multidimensional_transport_evidence_with_single_cell_and_external_canine.csv",
    RESULTS_DIR / "paper4_locked_multidimensional_transport_evidence_with_single_cell_six_dogs.csv",
    RESULTS_DIR / "paper4_locked_multidimensional_transport_evidence.csv",
]

OUTPUT_TRIANGULATION = RESULTS_DIR / "paper4_locked_external_canine_triangulation.csv"
OUTPUT_FINAL_MASTER = RESULTS_DIR / "paper4_locked_multidimensional_transport_evidence_final.csv"
OUTPUT_INTERPRETATION = RESULTS_DIR / "paper4_locked_final_module_interpretation.csv"
OUTPUT_SENTENCES = RESULTS_DIR / "paper4_locked_final_results_sentences.txt"
OUTPUT_GUARDRAILS = RESULTS_DIR / "paper4_locked_final_claim_guardrails.txt"
OUTPUT_README = RESULTS_DIR / "paper4_final_analysis_lock_README.txt"
OUTPUT_MANIFEST = RESULTS_DIR / "paper4_final_analysis_lock_manifest.json"

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]

EXPECTED_MANIFEST_VERSIONS = {
    SCRIPT47_MANIFEST: "47-lock-gse239948-independent-canine-evidence-v2",
    SCRIPT49_MANIFEST: "49-gse239948-blind-de-novo-rediscovery-v2",
    SCRIPT50_MANIFEST: "50-wgcna-module-preservation-benchmark-v1",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_required_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    print(f"Loaded: {path}")
    return pd.read_csv(path, **kwargs)


def read_optional_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        print(f"Optional file not found: {path}")
        return pd.DataFrame()
    print(f"Loaded: {path}")
    return pd.read_csv(path, **kwargs)


def read_required_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def choose_master_file() -> Path:
    for path in MASTER_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No multidimensional master file was found. Expected one of: "
        + ", ".join(str(path) for path in MASTER_CANDIDATES)
    )


def verify_manifest_versions() -> list[dict[str, Any]]:
    rows = []
    for path, expected in EXPECTED_MANIFEST_VERSIONS.items():
        payload = read_required_json(path)
        observed = str(payload.get("script_version", ""))
        if observed != expected:
            raise RuntimeError(
                f"Unexpected script version in {path.name}: "
                f"{observed}. Expected: {expected}"
            )
        if bool(payload.get("outcome_loaded", False)):
            raise RuntimeError(
                f"{path.name} unexpectedly reports outcome_loaded=true."
            )
        rows.append(
            {
                "manifest": path.name,
                "expected_script_version": expected,
                "observed_script_version": observed,
                "verified": True,
            }
        )
    return rows


def require_columns(table: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def numeric_value(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(pd.Series([row.get(column, np.nan)]), errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) else np.nan


def build_necrosis_annotation(necrosis: pd.DataFrame) -> pd.DataFrame:
    if necrosis.empty:
        return pd.DataFrame({"module_label": PRIMARY_MODULES})

    require_columns(necrosis, ["module_label"], "Necrosis table")
    keep = [
        "module_label",
        "spearman_rho",
        "spearman_permutation_p",
        "continuous_q_bh_4",
        "auc_higher_score_predicts_good_response",
        "auc_permutation_p_two_sided",
        "binary_q_bh_4",
        "logistic_or_per_sd",
    ]
    keep = [column for column in keep if column in necrosis.columns]

    result = necrosis[keep].copy()
    rename = {
        column: f"gse39055_necrosis_{column}"
        for column in keep
        if column != "module_label"
    }
    result = result.rename(columns=rename)

    classes = []
    for _, row in result.iterrows():
        continuous_p = numeric_value(row, "gse39055_necrosis_spearman_permutation_p")
        binary_p = numeric_value(row, "gse39055_necrosis_auc_permutation_p_two_sided")
        if (
            (np.isfinite(continuous_p) and continuous_p < 0.05)
            or (np.isfinite(binary_p) and binary_p < 0.05)
        ):
            classes.append("nominal_exploratory_pathological_response_support")
        else:
            classes.append("no_nominal_exploratory_pathological_response_support")
    result["gse39055_necrosis_exploratory_class"] = classes
    return result


def build_triangulation(
    script47: pd.DataFrame,
    script49: pd.DataFrame,
    wgcna: pd.DataFrame,
    method_concordance: pd.DataFrame,
    necrosis: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        script47,
        [
            "module_label",
            "edge_spearman",
            "loading_spearman",
            "external_canine_representation_class",
        ],
        "Script 47 table",
    )
    require_columns(
        script49,
        [
            "module_label",
            "best_match_f1",
            "empirical_max_match_q_bh_4",
            "blind_rediscovery_class",
        ],
        "Script 49 table",
    )
    require_columns(
        wgcna,
        [
            "module_label",
            "module_size",
            "zsummary_pres",
            "zdensity_pres",
            "zconnectivity_pres",
            "median_rank_pres",
            "wgcna_preservation_class",
        ],
        "Script 50 WGCNA table",
    )

    left_keep = [
        "module_label",
        "n_frozen_genes",
        "n_common_genes",
        "coverage_fraction",
        "edge_spearman",
        "edge_q_bh_8",
        "loading_spearman",
        "loading_q_bh_8",
        "split_half_median",
        "random_panel_empirical_p",
        "external_canine_representation_class",
    ]
    left_keep = [column for column in left_keep if column in script47.columns]

    blind_keep = [
        "module_label",
        "n_frozen_canine_genes",
        "n_frozen_genes_in_blind_discovery_universe",
        "discovery_universe_coverage_fraction",
        "best_discovered_module_id",
        "best_discovered_module_size",
        "overlap_genes",
        "frozen_gene_recall_within_discovery_universe",
        "discovered_module_precision",
        "best_match_jaccard",
        "best_match_f1",
        "best_module_subsample_stability_median_jaccard",
        "random_max_f1_q95",
        "empirical_max_match_p",
        "empirical_max_match_q_bh_4",
        "blind_rediscovery_class",
    ]
    blind_keep = [column for column in blind_keep if column in script49.columns]

    wgcna_keep = [
        "module_label",
        "module_size",
        "zsummary_pres",
        "zdensity_pres",
        "zconnectivity_pres",
        "median_rank_pres",
        "wgcna_preservation_class",
        "small_module_guardrail",
    ]
    wgcna_keep = [column for column in wgcna_keep if column in wgcna.columns]

    result = pd.DataFrame({"module_label": PRIMARY_MODULES})
    result = result.merge(script47[left_keep], on="module_label", how="left")
    result = result.merge(script49[blind_keep], on="module_label", how="left")
    result = result.merge(wgcna[wgcna_keep], on="module_label", how="left")

    if (
        not method_concordance.empty
        and "module_label" in method_concordance.columns
        and "preservation_method_concordance_class" in method_concordance.columns
    ):
        result = result.merge(
            method_concordance[
                ["module_label", "preservation_method_concordance_class"]
            ],
            on="module_label",
            how="left",
        )

    necrosis_annotation = build_necrosis_annotation(necrosis)
    result = result.merge(necrosis_annotation, on="module_label", how="left")

    final_classes = []
    manuscript_labels = []
    main_text_status = []

    for _, row in result.iterrows():
        module = str(row["module_label"])
        zsummary = numeric_value(row, "zsummary_pres")
        zdensity = numeric_value(row, "zdensity_pres")
        zconnectivity = numeric_value(row, "zconnectivity_pres")
        blind_q = numeric_value(row, "empirical_max_match_q_bh_4")
        blind_f1 = numeric_value(row, "best_match_f1")
        recall = numeric_value(row, "frozen_gene_recall_within_discovery_universe")
        external_class = str(row.get("external_canine_representation_class", ""))
        blind_class = str(row.get("blind_rediscovery_class", ""))
        module_size = numeric_value(row, "module_size")

        tri_method = bool(
            np.isfinite(zsummary)
            and zsummary >= 10
            and external_class == "strong_external_canine_representation_preservation"
            and np.isfinite(blind_q)
            and blind_q < 0.05
            and blind_class == "strong_blind_independent_rediscovery"
        )

        if tri_method and module == "M40":
            final_class = (
                "triangulated_strong_external_canine_replication_"
                "with_substantial_blind_recovery"
            )
            manuscript = (
                "M40 showed concordant external canine preservation across three "
                "complementary analyses: strong direct edge/loading preservation, "
                "strong standard WGCNA preservation, and significant outcome-blind "
                "de novo rediscovery with substantial recovery of the naturally "
                "available frozen genes. This supports M40 as a highly reproducible "
                "cycling/proliferation-related molecular architecture, while its "
                "clinical-outcome transport remains assay- and context-sensitive."
            )
            status = "main_text"
        elif tri_method and module == "M34":
            final_class = (
                "triangulated_strong_external_canine_replication_"
                "with_significant_partial_blind_core_recovery"
            )
            manuscript = (
                "M34 showed strong direct and WGCNA preservation in the independent "
                "canine cohort, while blind de novo clustering recovered a statistically "
                "specific and stable M34-related core rather than the full frozen module "
                "boundary. This supports robust external canine representation replication "
                "without implying exact de novo recovery of the complete 162-gene program."
            )
            status = "main_text"
        elif np.isfinite(module_size) and module_size < 10:
            if np.isfinite(zsummary) and zsummary >= 2:
                final_class = (
                    "small_module_wgcna_support_without_independent_blind_replication"
                )
                manuscript = (
                    f"{module} showed moderate WGCNA preservation by Zsummary, but the "
                    "module contained only six shared genes and did not show concordant "
                    "support in the direct custom preservation or blind rediscovery analyses. "
                    "The result is therefore treated as small-module supportive evidence, "
                    "not as independent structural replication."
                )
            else:
                final_class = "small_module_no_clear_external_replication"
                manuscript = (
                    f"{module} did not show clear independent canine replication. "
                    "Inference remains limited by the very small number of shared genes."
                )
            status = "secondary_or_supplementary"
        elif external_class == "strong_external_canine_representation_preservation":
            final_class = "direct_external_canine_preservation_without_full_triangulation"
            manuscript = (
                f"{module} showed direct external canine preservation, but the full "
                "three-method triangulation criterion was not met."
            )
            status = "secondary_or_supplementary"
        else:
            final_class = "no_clear_triangulated_external_canine_replication"
            manuscript = (
                f"{module} did not show clear concordant external canine replication "
                "across the direct, blind, and WGCNA analyses."
            )
            status = "secondary_or_supplementary"

        final_classes.append(final_class)
        manuscript_labels.append(manuscript)
        main_text_status.append(status)

    result["external_canine_triangulation_class"] = final_classes
    result["external_canine_manuscript_interpretation"] = manuscript_labels
    result["external_canine_reporting_status"] = main_text_status

    result["wgcna_density_connectivity_pattern"] = result.apply(
        lambda row: density_connectivity_pattern(
            numeric_value(row, "zdensity_pres"),
            numeric_value(row, "zconnectivity_pres"),
        ),
        axis=1,
    )

    return result


def density_connectivity_pattern(zdensity: float, zconnectivity: float) -> str:
    density_strong = np.isfinite(zdensity) and zdensity >= 10
    density_moderate = np.isfinite(zdensity) and 2 <= zdensity < 10
    connectivity_strong = np.isfinite(zconnectivity) and zconnectivity >= 10
    connectivity_moderate = (
        np.isfinite(zconnectivity) and 2 <= zconnectivity < 10
    )

    if density_strong and connectivity_strong:
        return "strong_density_and_connectivity_preservation"
    if density_strong and not connectivity_strong:
        return "strong_density_with_weaker_connectivity_preservation"
    if density_moderate and connectivity_moderate:
        return "moderate_density_and_connectivity_preservation"
    if density_moderate and not connectivity_moderate:
        return "moderate_density_with_limited_connectivity_preservation"
    if not density_moderate and connectivity_moderate:
        return "limited_density_with_moderate_connectivity_preservation"
    return "limited_or_unclear_density_connectivity_preservation"


def final_multidimensional_class(module: str) -> str:
    mapping = {
        "M34": (
            "replicated_canine_and_cross_species_representation_"
            "with_endpoint_and_platform_dependent_outcome_transport"
        ),
        "M40": (
            "triangulated_canine_and_cross_species_cycling_representation_"
            "with_assay_sensitive_outcome_transport"
        ),
        "M11": (
            "directional_outcome_concordance_with_small_module_"
            "representation_uncertainty"
        ),
        "M24": (
            "limited_representation_and_endpoint_specific_outcome_evidence"
        ),
    }
    return mapping[module]


def final_interpretation(module: str) -> str:
    mapping = {
        "M34": (
            "M34 is the principal translational program. Its molecular representation "
            "is supported by human network/latent analyses and by independent canine "
            "replication using direct preservation, a standard WGCNA benchmark, and "
            "significant blind recovery of an M34-related core. Human outcome transport "
            "is not universal: GSE21257 provides the only project-wide FDR-controlled "
            "primary outcome result, TARGET-OS provides nominal directional support, and "
            "GSE39055 remains an assay-limited discordant setting."
        ),
        "M40": (
            "M40 is the strongest architecture-reproducibility result. It shows very "
            "strong independent canine preservation by direct metrics and WGCNA and was "
            "substantially rediscovered without frozen-module membership. Its molecular "
            "representation is therefore highly reproducible, whereas prognostic transport "
            "is assay- and context-sensitive. The exploratory GSE39055 pathological-necrosis "
            "analysis does not provide nominal evidence for a chemotherapy-response association."
        ),
        "M11": (
            "M11 retains the most directionally consistent human outcome pattern, but "
            "independent representation evidence is limited. Moderate WGCNA preservation "
            "is interpreted cautiously because only six shared genes were available and "
            "custom preservation and blind rediscovery were not supported."
        ),
        "M24": (
            "M24 remains an endpoint-specific program with limited general representation "
            "evidence. Its six-gene external WGCNA result is size-sensitive and is not "
            "supported by the custom or blind independent-canine analyses."
        ),
    }
    return mapping[module]


def update_master(
    master: pd.DataFrame,
    triangulation: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(master, ["module_label"], "Multidimensional master")

    columns_to_add = [
        "module_label",
        "module_size",
        "zsummary_pres",
        "zdensity_pres",
        "zconnectivity_pres",
        "median_rank_pres",
        "wgcna_preservation_class",
        "best_match_f1",
        "empirical_max_match_q_bh_4",
        "blind_rediscovery_class",
        "external_canine_triangulation_class",
        "external_canine_reporting_status",
        "wgcna_density_connectivity_pattern",
        "gse39055_necrosis_exploratory_class",
    ]
    columns_to_add = [
        column for column in columns_to_add if column in triangulation.columns
    ]

    renamed = triangulation[columns_to_add].copy()
    rename_map = {
        column: f"gse239948_{column}"
        for column in columns_to_add
        if column != "module_label"
        and not column.startswith("gse39055_")
    }
    renamed = renamed.rename(columns=rename_map)

    updated = master.merge(renamed, on="module_label", how="left")

    updated["final_multidimensional_transport_class"] = updated["module_label"].map(
        final_multidimensional_class
    )
    updated["final_locked_interpretation"] = updated["module_label"].map(
        final_interpretation
    )
    return updated


def build_interpretation_table(
    triangulation: pd.DataFrame,
    master: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for module in PRIMARY_MODULES:
        tri = triangulation[triangulation["module_label"].eq(module)].iloc[0]
        master_row = master[master["module_label"].eq(module)].iloc[0]
        rows.append(
            {
                "module_label": module,
                "external_canine_triangulation_class": tri[
                    "external_canine_triangulation_class"
                ],
                "external_canine_reporting_status": tri[
                    "external_canine_reporting_status"
                ],
                "final_multidimensional_transport_class": master_row[
                    "final_multidimensional_transport_class"
                ],
                "final_locked_interpretation": master_row[
                    "final_locked_interpretation"
                ],
            }
        )
    return pd.DataFrame(rows)


def write_sentences(triangulation: pd.DataFrame) -> None:
    indexed = triangulation.set_index("module_label")

    lines = [
        "Paper 4 final locked results sentences",
        "=====================================",
        "",
        "Independent canine replication",
        "------------------------------",
        (
            "Independent canine validation converged across complementary methods. "
            f"M34 showed WGCNA Zsummary={indexed.loc['M34', 'zsummary_pres']:.2f}, "
            f"direct edge/loading concordance of "
            f"{indexed.loc['M34', 'edge_spearman']:.3f}/"
            f"{indexed.loc['M34', 'loading_spearman']:.3f}, and a significant "
            f"blind de novo best-match test "
            f"(BH q={indexed.loc['M34', 'empirical_max_match_q_bh_4']:.4f}). "
            f"M40 showed WGCNA Zsummary={indexed.loc['M40', 'zsummary_pres']:.2f}, "
            f"direct edge/loading concordance of "
            f"{indexed.loc['M40', 'edge_spearman']:.3f}/"
            f"{indexed.loc['M40', 'loading_spearman']:.3f}, and significant "
            f"blind rediscovery "
            f"(BH q={indexed.loc['M40', 'empirical_max_match_q_bh_4']:.4f})."
        ),
        "",
        "M34 wording",
        "-----------",
        (
            "M34 should be described as having strong external canine representation "
            "replication with statistically supported partial de novo recovery of an "
            "M34-related core. The blind analysis does not justify claiming exact "
            "reconstruction of the full frozen module boundary."
        ),
        "",
        "M40 wording",
        "-----------",
        (
            "M40 provides the clearest evidence that molecular architecture and clinical "
            "transport are separable: the cycling-related program is highly reproducible "
            "across independent canine and human expression settings, while its prognostic "
            "association is assay- and context-sensitive."
        ),
        "",
        "Small modules",
        "-------------",
        (
            "M11 and M24 showed moderate WGCNA Zsummary values, but each contained only "
            "six shared genes and neither was supported by the direct custom preservation "
            "or blind rediscovery analyses. These WGCNA results are therefore supportive "
            "small-module sensitivities rather than independent structural replications."
        ),
        "",
        "Pathological response",
        "---------------------",
        (
            "The exploratory GSE39055 pathological-necrosis analysis did not provide "
            "nominal evidence that M40 is a pathological chemotherapy-response marker. "
            "This negative result should be retained as an exploratory limitation rather "
            "than optimized further."
        ),
    ]

    OUTPUT_SENTENCES.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_guardrails() -> None:
    text = """Paper 4 final claim guardrails
==============================

1. Do not call M34 or M40 universal prognostic biomarkers.
2. Do not describe GSE239948 as an outcome-validation cohort.
3. Do not claim donor independence from expression fingerprinting alone.
4. Do not state that blind clustering reconstructed the full M34 module.
5. Do not upgrade M11 or M24 to strong preservation based only on WGCNA Zsummary.
6. Do not interpret WGCNA Zsummary as a module-size-independent significance test.
7. Do not pool TARGET-OS, GSE21257, and GSE39055 into one meta-analytic effect.
8. Do not present M40 as a chemotherapy-benefit or pathological-response biomarker.
9. Keep the project-wide 12-test multiplicity result unchanged.
10. No additional outcome-driven feature selection, score reversal, or assay-rule optimization.
"""
    OUTPUT_GUARDRAILS.write_text(text, encoding="utf-8")


def write_readme(master_file: Path) -> None:
    text = f"""Paper 4 final multidimensional evidence lock
Script version: {SCRIPT_VERSION}

Purpose
-------
Integrate the independent-canine evidence from scripts 47, 49, and 50 into the
latest locked multidimensional master table.

This script performs no new:
- feature selection,
- clustering,
- network fitting,
- outcome fitting,
- score orientation,
- multiplicity testing,
- module redefinition.

Evidence hierarchy
------------------
Script 47:
Direct outcome-blind external canine edge/loading preservation and sample-overlap audit.

Script 49:
Blind de novo GSE239948 module discovery followed only afterward by frozen-program
matching against a best-of-many variance-matched random-panel null.

Script 50:
Standard WGCNA modulePreservation() reviewer-facing benchmark with 200 permutations.

Interpretation
--------------
M34 and M40 receive the strongest external canine architecture support because all
three approaches converge.

For M34, blind de novo recovery is statistically specific but boundary-level overlap is
partial, so manuscript wording should refer to recovery of an M34-related core.

For M40, blind recovery is substantially stronger and complements exceptionally strong
direct and WGCNA preservation.

M11 and M24 each have only six shared genes in the WGCNA benchmark. Their moderate
Zsummary values are retained as supportive small-module sensitivities but do not override
the negative custom and blind analyses.

Pathological response
---------------------
If available, the script records the frozen GSE39055 necrosis analysis only as an
exploratory annotation. It does not alter any module evidence tier.

Input multidimensional master
-----------------------------
{master_file}

Final restriction
-----------------
This is an evidence-lock script. The next project stage should be manuscript figure/table
generation and writing rather than additional outcome-driven discovery.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def create_manifest(
    input_paths: list[Path],
    output_paths: list[Path],
    master_file: Path,
) -> None:
    payload = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "analysis_type": "evidence_lock_only",
        "new_statistical_tests": False,
        "feature_selection": False,
        "outcome_model_fitting": False,
        "module_redefinition": False,
        "source_master": master_file.name,
        "inputs": {},
        "outputs": {},
        "guardrails": [
            "Scripts 47, 49, and 50 are integrated without rerunning their analyses.",
            "M34 blind rediscovery is described as statistically supported partial core recovery.",
            "M40 is the strongest tri-method external canine architecture replication.",
            "M11 and M24 WGCNA results remain small-module supportive sensitivities.",
            "Existing project-wide human multiplicity conclusions are unchanged.",
            "No additional outcome-driven tuning is permitted by this lock.",
        ],
    }

    for path in input_paths:
        if path.exists():
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
    print("Final lock of external canine preservation triangulation")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Design:")
    print("  Perform no new statistical tests.")
    print("  Verify scripts 47, 49, and 50 manifests.")
    print("  Integrate direct, blind, and WGCNA external canine evidence.")
    print("  Preserve project-wide human outcome conclusions.")
    print("  Freeze manuscript-ready multidimensional interpretations.")
    print("")

    manifest_audit = verify_manifest_versions()
    master_file = choose_master_file()

    master = read_required_csv(master_file)
    script47 = read_required_csv(SCRIPT47_FILE)
    script49 = read_required_csv(SCRIPT49_FILE)
    wgcna = read_required_csv(SCRIPT50_FILE)
    concordance = read_required_csv(SCRIPT50_CONCORDANCE_FILE)
    necrosis = read_optional_csv(NECROSIS_PRIMARY_FILE)

    triangulation = build_triangulation(
        script47=script47,
        script49=script49,
        wgcna=wgcna,
        method_concordance=concordance,
        necrosis=necrosis,
    )
    final_master = update_master(master, triangulation)
    interpretation = build_interpretation_table(triangulation, final_master)

    triangulation.to_csv(OUTPUT_TRIANGULATION, index=False)
    final_master.to_csv(OUTPUT_FINAL_MASTER, index=False)
    interpretation.to_csv(OUTPUT_INTERPRETATION, index=False)
    write_sentences(triangulation)
    write_guardrails()
    write_readme(master_file)

    print("=" * 80)
    print("Manifest version audit")
    print("=" * 80)
    print(pd.DataFrame(manifest_audit).to_string(index=False))

    print("")
    print("=" * 80)
    print("External canine tri-method evidence")
    print("=" * 80)
    display_columns = [
        "module_label",
        "edge_spearman",
        "loading_spearman",
        "zsummary_pres",
        "zdensity_pres",
        "zconnectivity_pres",
        "best_match_f1",
        "empirical_max_match_q_bh_4",
        "external_canine_triangulation_class",
    ]
    display_columns = [
        column for column in display_columns if column in triangulation.columns
    ]
    print(triangulation[display_columns].to_string(index=False))

    print("")
    print("=" * 80)
    print("Final multidimensional interpretations")
    print("=" * 80)
    print(
        interpretation[
            [
                "module_label",
                "final_multidimensional_transport_class",
                "final_locked_interpretation",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("M34 blind rediscovery is partial/core recovery, not exact full-module recovery.")
    print("M40 has the strongest architecture replication but not stable prognostic transport.")
    print("M11 and M24 WGCNA results are small-module supportive sensitivities.")
    print("No project-wide human outcome result or multiplicity conclusion is changed.")
    print("No additional outcome-driven tuning should follow this lock.")

    input_paths = [
        master_file,
        SCRIPT47_FILE,
        SCRIPT47_MANIFEST,
        SCRIPT49_FILE,
        SCRIPT49_MANIFEST,
        SCRIPT50_FILE,
        SCRIPT50_CONCORDANCE_FILE,
        SCRIPT50_MANIFEST,
    ]
    if NECROSIS_PRIMARY_FILE.exists():
        input_paths.append(NECROSIS_PRIMARY_FILE)

    output_paths = [
        OUTPUT_TRIANGULATION,
        OUTPUT_FINAL_MASTER,
        OUTPUT_INTERPRETATION,
        OUTPUT_SENTENCES,
        OUTPUT_GUARDRAILS,
        OUTPUT_README,
    ]
    create_manifest(
        input_paths=input_paths,
        output_paths=output_paths,
        master_file=master_file,
    )

    print("")
    print("Saved:")
    for path in output_paths + [OUTPUT_MANIFEST]:
        print(path)
    print("Done.")


if __name__ == "__main__":
    main()
