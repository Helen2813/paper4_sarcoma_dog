from __future__ import annotations

from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "53-generate-paper4-manuscript-draft-v1"
TARGET_JOURNAL = "Cancer Letters"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"
MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

MAIN_RESULTS_FILE = RESULTS_DIR / "Paper4_final_main_results_table.csv"
HUMAN_RESULTS_FILE = RESULTS_DIR / "Paper4_final_human_primary_outcomes_table.csv"
CANINE_TRIANGULATION_FILE = RESULTS_DIR / "Paper4_final_external_canine_triangulation_table.csv"
DECOUPLING_FILE = RESULTS_DIR / "Paper4_final_representation_outcome_typology_table.csv"
FINAL_INTERPRETATION_FILE = RESULTS_DIR / "paper4_locked_final_module_interpretation.csv"
FIGURE_CAPTIONS_FILE = RESULTS_DIR / "Paper4_final_figure_captions.txt"
RESULTS_OUTLINE_FILE = RESULTS_DIR / "Paper4_final_results_section_outline.txt"
ASSET_MANIFEST_FILE = RESULTS_DIR / "Paper4_final_manuscript_assets_manifest.json"
FINAL_LOCK_MANIFEST_FILE = RESULTS_DIR / "paper4_final_analysis_lock_manifest.json"

OUTPUT_MANUSCRIPT = MANUSCRIPT_DIR / "Paper4_manuscript_draft_v1.md"
OUTPUT_TITLE_OPTIONS = MANUSCRIPT_DIR / "Paper4_title_options.txt"
OUTPUT_HIGHLIGHTS = MANUSCRIPT_DIR / "Paper4_highlights.txt"
OUTPUT_CLAIM_MATRIX = MANUSCRIPT_DIR / "Paper4_claim_matrix.csv"
OUTPUT_AUTHOR_CHECKLIST = MANUSCRIPT_DIR / "Paper4_authorship_and_submission_checklist.txt"
OUTPUT_MANIFEST = MANUSCRIPT_DIR / "Paper4_manuscript_draft_manifest.json"

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    print(f"Loaded: {path}")
    return pd.read_csv(path)


def read_required_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    print(f"Loaded: {path}")
    return path.read_text(encoding="utf-8")


def read_required_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required manifest: {path}")
    print(f"Loaded: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require_columns(table: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def safe_float(value: Any) -> float:
    result = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(result) if np.isfinite(result) else np.nan


def fmt(value: Any, digits: int = 3) -> str:
    value = safe_float(value)
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def fmt_p(value: Any) -> str:
    value = safe_float(value)
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def verify_lock_manifests(
    final_lock: dict[str, Any],
    asset_manifest: dict[str, Any],
) -> None:
    expected_final = "51-final-lock-external-canine-triangulation-v1"
    expected_assets = "52-generate-final-paper4-manuscript-assets-v1"

    observed_final = str(final_lock.get("script_version", ""))
    observed_assets = str(asset_manifest.get("script_version", ""))

    if observed_final != expected_final:
        raise RuntimeError(
            f"Unexpected script 51 lock version: {observed_final}. "
            f"Expected: {expected_final}"
        )

    if observed_assets != expected_assets:
        raise RuntimeError(
            f"Unexpected script 52 asset version: {observed_assets}. "
            f"Expected: {expected_assets}"
        )

    for payload, label in [
        (final_lock, "script 51"),
        (asset_manifest, "script 52"),
    ]:
        if bool(payload.get("new_statistical_tests", False)):
            raise RuntimeError(f"{label} unexpectedly reports new statistical tests.")
        if bool(payload.get("outcome_model_fitting", False)):
            raise RuntimeError(f"{label} unexpectedly reports outcome model fitting.")
        if bool(payload.get("feature_selection", False)):
            raise RuntimeError(f"{label} unexpectedly reports feature selection.")


def extract_locked_numbers(
    human: pd.DataFrame,
    canine: pd.DataFrame,
) -> dict[str, Any]:
    require_columns(
        human,
        [
            "module_label",
            "TARGET_OS_HR_per_SD",
            "TARGET_OS_CI_low",
            "TARGET_OS_CI_high",
            "TARGET_OS_primary_p",
            "TARGET_OS_projectwide_q12",
            "GSE21257_AUC",
            "GSE21257_AUC_CI_low",
            "GSE21257_AUC_CI_high",
            "GSE21257_primary_p",
            "GSE21257_projectwide_q12",
            "GSE39055_RFS_HR_per_SD",
            "GSE39055_RFS_CI_low",
            "GSE39055_RFS_CI_high",
            "GSE39055_RFS_primary_p",
            "GSE39055_RFS_projectwide_q12",
        ],
        "Human results",
    )

    require_columns(
        canine,
        [
            "module_label",
            "edge_spearman",
            "loading_spearman",
            "zsummary_pres",
            "zdensity_pres",
            "zconnectivity_pres",
            "best_match_f1",
            "empirical_max_match_q_bh_4",
        ],
        "External canine results",
    )

    h = human.set_index("module_label")
    c = canine.set_index("module_label")

    result: dict[str, Any] = {}

    for module in PRIMARY_MODULES:
        result[module] = {
            "target_hr": safe_float(h.loc[module, "TARGET_OS_HR_per_SD"]),
            "target_ci_low": safe_float(h.loc[module, "TARGET_OS_CI_low"]),
            "target_ci_high": safe_float(h.loc[module, "TARGET_OS_CI_high"]),
            "target_p": safe_float(h.loc[module, "TARGET_OS_primary_p"]),
            "target_q12": safe_float(h.loc[module, "TARGET_OS_projectwide_q12"]),
            "gse_auc": safe_float(h.loc[module, "GSE21257_AUC"]),
            "gse_auc_low": safe_float(h.loc[module, "GSE21257_AUC_CI_low"]),
            "gse_auc_high": safe_float(h.loc[module, "GSE21257_AUC_CI_high"]),
            "gse_p": safe_float(h.loc[module, "GSE21257_primary_p"]),
            "gse_q12": safe_float(h.loc[module, "GSE21257_projectwide_q12"]),
            "rfs_hr": safe_float(h.loc[module, "GSE39055_RFS_HR_per_SD"]),
            "rfs_ci_low": safe_float(h.loc[module, "GSE39055_RFS_CI_low"]),
            "rfs_ci_high": safe_float(h.loc[module, "GSE39055_RFS_CI_high"]),
            "rfs_p": safe_float(h.loc[module, "GSE39055_RFS_primary_p"]),
            "rfs_q12": safe_float(h.loc[module, "GSE39055_RFS_projectwide_q12"]),
            "edge": safe_float(c.loc[module, "edge_spearman"]),
            "loading": safe_float(c.loc[module, "loading_spearman"]),
            "zsummary": safe_float(c.loc[module, "zsummary_pres"]),
            "zdensity": safe_float(c.loc[module, "zdensity_pres"]),
            "zconnectivity": safe_float(c.loc[module, "zconnectivity_pres"]),
            "blind_f1": safe_float(c.loc[module, "best_match_f1"]),
            "blind_q": safe_float(c.loc[module, "empirical_max_match_q_bh_4"]),
        }

    return result


def validate_locked_story(numbers: dict[str, Any]) -> None:
    supported = []
    for module in PRIMARY_MODULES:
        values = numbers[module]
        for cohort_key, q_key in [
            ("TARGET_OS", "target_q12"),
            ("GSE21257", "gse_q12"),
            ("GSE39055", "rfs_q12"),
        ]:
            q_value = values[q_key]
            if np.isfinite(q_value) and q_value < 0.05:
                supported.append((cohort_key, module, q_value))

    if supported != [("GSE21257", "M34", numbers["M34"]["gse_q12"])]:
        raise RuntimeError(
            "Locked project-wide multiplicity pattern changed unexpectedly. "
            f"Observed supported tests: {supported}"
        )

    for module in ["M34", "M40"]:
        if not (
            numbers[module]["zsummary"] >= 10
            and numbers[module]["blind_q"] < 0.05
        ):
            raise RuntimeError(
                f"{module} no longer meets the expected external canine triangulation pattern."
            )


def build_title_options() -> str:
    titles = [
        "Cross-species osteosarcoma transcriptional programs show conserved molecular architecture but heterogeneous prognostic transport",
        "Molecular preservation does not guarantee prognostic transport across canine and human osteosarcoma",
        "Cross-species representation analysis identifies conserved osteosarcoma programs with context-dependent clinical transport",
        "Conserved osteosarcoma transcriptional programs decouple molecular reproducibility from prognostic transport across species",
        "Comparative osteosarcoma transcriptomics reveals reproducible molecular programs with endpoint- and platform-dependent clinical transport",
    ]

    lines = [
        "Paper 4 title options",
        "=====================",
        "",
        f"Target journal: {TARGET_JOURNAL}",
        "",
    ]

    for index, title in enumerate(titles, start=1):
        lines.append(f"{index}. {title}")

    lines.extend(
        [
            "",
            "Recommended working title:",
            titles[0],
            "",
            "Title guardrail:",
            "Avoid calling the modules universal prognostic biomarkers because only one of the 12 frozen human primary tests is project-wide FDR supported.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_highlights() -> str:
    lines = [
        "Paper 4 highlights",
        "==================",
        "",
        "- Canine osteosarcoma programs were frozen before human outcome evaluation.",
        "- M34 showed the strongest cross-species translational evidence and the only project-wide FDR-supported human primary outcome test.",
        "- M34 and M40 showed strong independent canine molecular preservation by direct, blind de novo, and WGCNA analyses.",
        "- M40 was highly reproducible as a molecular architecture despite assay- and context-sensitive prognostic transport.",
        "- Molecular representation preservation and clinical-outcome transport emerged as distinct, non-interchangeable properties.",
    ]
    return "\n".join(lines) + "\n"


def build_claim_matrix(
    numbers: dict[str, Any],
    interpretation: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        interpretation,
        [
            "module_label",
            "final_multidimensional_transport_class",
            "final_locked_interpretation",
        ],
        "Final interpretation table",
    )

    interp = interpretation.set_index("module_label")

    rows = [
        {
            "claim_id": "C1",
            "claim": "M34 is the principal translational program.",
            "primary_evidence": (
                f"GSE21257 AUC={numbers['M34']['gse_auc']:.3f}, "
                f"project-wide q12={numbers['M34']['gse_q12']:.4f}; "
                f"TARGET-OS HR={numbers['M34']['target_hr']:.2f}, "
                f"p={numbers['M34']['target_p']:.3f}."
            ),
            "status": "main_text",
            "guardrail": "Do not call M34 universally prognostic because GSE39055 is discordant.",
        },
        {
            "claim_id": "C2",
            "claim": "M34 has strong external canine molecular replication.",
            "primary_evidence": (
                f"edge rho={numbers['M34']['edge']:.3f}; "
                f"loading rho={numbers['M34']['loading']:.3f}; "
                f"WGCNA Zsummary={numbers['M34']['zsummary']:.2f}; "
                f"blind q={numbers['M34']['blind_q']:.4f}."
            ),
            "status": "main_text",
            "guardrail": "Describe blind recovery as a stable M34-related core, not exact full-module reconstruction.",
        },
        {
            "claim_id": "C3",
            "claim": "M40 has the strongest molecular-architecture reproducibility.",
            "primary_evidence": (
                f"edge rho={numbers['M40']['edge']:.3f}; "
                f"loading rho={numbers['M40']['loading']:.3f}; "
                f"WGCNA Zsummary={numbers['M40']['zsummary']:.2f}; "
                f"blind F1={numbers['M40']['blind_f1']:.3f}, "
                f"q={numbers['M40']['blind_q']:.4f}."
            ),
            "status": "main_text",
            "guardrail": "Do not equate molecular reproducibility with stable prognosis or chemotherapy benefit.",
        },
        {
            "claim_id": "C4",
            "claim": "Molecular representation and prognostic transport are separable.",
            "primary_evidence": (
                "M40 is strongly preserved structurally but has assay/context-sensitive outcome transport; "
                "M34 is strongly represented but not outcome-concordant in all human settings."
            ),
            "status": "main_concept",
            "guardrail": "Do not meta-analyze OS, five-year metastasis, and RFS as one endpoint.",
        },
        {
            "claim_id": "C5",
            "claim": "M11 and M24 remain secondary or limited programs.",
            "primary_evidence": (
                f"M11 WGCNA Zsummary={numbers['M11']['zsummary']:.2f}, "
                f"blind q={numbers['M11']['blind_q']:.3f}; "
                f"M24 WGCNA Zsummary={numbers['M24']['zsummary']:.2f}, "
                f"blind q={numbers['M24']['blind_q']:.3f}."
            ),
            "status": "secondary",
            "guardrail": "Small-module WGCNA support does not override weak direct/blind replication.",
        },
    ]

    result = pd.DataFrame(rows)

    for module in PRIMARY_MODULES:
        result.loc[result["claim"].str.contains(module, regex=False), "locked_class_reference"] = (
            interp.loc[module, "final_multidimensional_transport_class"]
        )

    return result


def abstract_text(numbers: dict[str, Any]) -> str:
    m34 = numbers["M34"]
    m40 = numbers["M40"]

    return f"""Osteosarcoma shares clinical and molecular features across dogs and humans, but the extent to which prognostic transcriptional programs retain both molecular identity and clinical meaning across species remains unclear. We developed an outcome-blind cross-species framework in which canine RNA-sequencing programs were frozen before human evaluation and were subsequently assessed across orthogonal representation, outcome, assay, latent-factor, and single-cell evidence layers. Four primary canine programs were transferred using strict one-to-one ortholog mapping without human refitting.

M34 showed the strongest translational evidence. In GSE21257, the frozen M34 score discriminated five-year metastasis with an AUC of {m34['gse_auc']:.3f} (95% CI {m34['gse_auc_low']:.3f}-{m34['gse_auc_high']:.3f}; p={fmt_p(m34['gse_p'])}) and remained the only significant test after project-wide correction across 12 frozen human primary analyses (q={m34['gse_q12']:.4f}). TARGET-OS provided nominal directionally concordant support (HR per SD {m34['target_hr']:.2f}, 95% CI {m34['target_ci_low']:.2f}-{m34['target_ci_high']:.2f}; p={m34['target_p']:.3f}), whereas GSE39055 showed discordant outcome transport in an assay-limited setting. Independent canine validation in GSE239948 supported strong M34 and M40 molecular preservation. M34 showed direct edge/loading concordance of {m34['edge']:.3f}/{m34['loading']:.3f}, WGCNA Zsummary={m34['zsummary']:.2f}, and significant blind recovery of an M34-related core (q={m34['blind_q']:.4f}). M40 showed even stronger molecular reproducibility (edge/loading {m40['edge']:.3f}/{m40['loading']:.3f}; WGCNA Zsummary={m40['zsummary']:.2f}; blind F1={m40['blind_f1']:.3f}, q={m40['blind_q']:.4f}) despite unstable prognostic transport.

These findings distinguish preservation of molecular representation from preservation of prognostic association. Cross-species osteosarcoma programs can remain highly reproducible as transcriptional architectures while their clinical effects vary across endpoints and assay contexts, arguing for explicit separation of biological representation transport from biomarker transport in comparative oncology."""


def introduction_text() -> str:
    return """Osteosarcoma is an aggressive primary bone malignancy with substantial biological heterogeneity and limited improvement in survival for patients with metastatic or recurrent disease. Spontaneous canine osteosarcoma recapitulates several clinical and molecular features of the human disease and therefore provides a valuable comparative model for identifying conserved tumor programs. However, cross-species studies often move directly from molecular similarity to biomarker claims, even though preservation of a transcriptional program and preservation of its clinical association are logically distinct questions.

Most molecular signatures are evaluated within a single cohort or species and are vulnerable to feature re-selection, score reorientation, assay-specific preprocessing, and endpoint-specific optimization. These issues are amplified in cross-species studies, where differences in orthology, platform, tumor composition, and clinical annotation can create apparent non-replication even when an underlying biological program remains present. Conversely, a statistically preserved co-expression structure does not imply that its prognostic association must remain invariant across populations or endpoints.

We therefore designed a leakage-controlled comparative framework in which canine osteosarcoma transcriptional programs were defined and frozen before human outcome analysis. Rather than treating transfer as a single binary property, we separated evidence for molecular representation, prognostic transport, assay robustness, latent recurrence, independent canine replication, and cellular localization. Human overall survival, five-year metastasis, and recurrence-free survival were retained as distinct clinical endpoints and were not pooled into a single meta-analytic effect.

Our central hypothesis was that conserved osteosarcoma programs would show heterogeneous levels of transport across these evidence dimensions. We further tested whether the most reproducible molecular architectures necessarily displayed the most reproducible prognostic effects. This design identified M34 as the principal translational program and M40 as a particularly strong example of a reproducible cycling-related architecture whose clinical association remained context-sensitive."""


def methods_text() -> str:
    return """### Study design and leakage control

The analysis was organized as a sequential evidence-locking workflow. Canine discovery, module membership, transfer eligibility, score direction, and primary human scoring rules were frozen before human outcome evaluation. Human outcomes were not used to alter ortholog selection, module genes, score orientation, or evidence tiers. Later assay-quality and representation analyses were treated as sensitivity or orthogonal evidence layers and were not allowed to replace the frozen primary analyses.

### Canine discovery cohort and frozen transcriptional programs

The discovery cohort comprised 186 primary canine osteosarcoma tumors from GSE238110 (DOG²). [VERIFY: add exact RNA-sequencing processing, normalization, and original module-discovery details from scripts 1–17.] Candidate programs were evaluated for prognostic association with disease-free interval and overall survival and were subsequently audited for dependence on generic proliferation. Four primary programs were locked for human transfer: M34, M11, M24, and M40. M34, M11, and M24 were treated as primary non-proliferation programs, whereas M40 was retained as a proliferation-dominant axis with a reproducible residual component.

### Ortholog mapping and frozen human scoring

Canine genes were mapped to strict one-to-one human orthologs using the frozen ortholog-QC table. Human primary scores were computed as signed means of standardized expression values using the canine risk direction fixed before human outcome analysis. No human outcome was used to select genes, optimize weights, or reverse the score direction. The primary strict human coverage was 7 genes for M11, 7 for M24, 162 for M34, and 111 for M40 before cohort-specific platform losses.

### Human validation cohorts and endpoints

Three independent human osteosarcoma settings were used. TARGET-OS provided an overall-survival endpoint; GSE21257 provided five-year metastatic status; and GSE39055 provided recurrence-free survival from diagnostic biopsy samples. [VERIFY: insert exact inclusion/exclusion counts and endpoint definitions from the locked cohort-preparation scripts.] Cox proportional-hazards models were used for time-to-event endpoints and fixed-score discrimination was summarized using concordance. Five-year metastasis in GSE21257 was evaluated using zero-shot ROC-AUC and average precision. The three endpoints were analyzed separately and were not combined into one effect estimate.

### Project-wide multiplicity control

The 12 primary human tests comprised four frozen modules across the three predefined human settings. Benjamini-Hochberg correction was applied across all 12 primary tests to define project-wide false-discovery-rate support. Cohort-specific or assay-sensitivity analyses did not replace this project-wide primary family.

### Proliferation adjustment and M40 decomposition

Generic proliferation was quantified independently of each tested module. Module scores were residualized against disjoint proliferation representations using outcome-blind procedures, including train-only repeated cross-validation in the canine cohort where appropriate. M40 was therefore interpreted as a proliferation-dominant program with a reproducible deviation component rather than as a clean non-proliferation module.

### Human structural preservation

Cross-species representation was evaluated using direct gene-gene edge concordance, loading concordance, and non-overlapping split-half reliability. Gene-label permutation was used to assess whether observed concordance exceeded that expected from matched genes. Conservative structural-preservation classes were defined independently of human outcomes. [VERIFY: add exact permutation counts and classification thresholds from scripts 27–28.]

### Assay-quality audit in GSE39055

Because GSE39055 used an FFPE DASL platform, detection P-values and alternative outcome-blind probe-selection rules were audited. Locked strict scores remained primary. Detection-aware rules were treated as sensitivity analyses and could not be selected based on outcome performance. This analysis distinguished stable biological discordance from assay-rule-sensitive direction changes.

### Multi-group latent-factor analysis

Outcome-blind multi-group latent-factor models were fitted across the human expression datasets. A variable-only analysis was used to test whether M34- and M40-related latent representations emerged without forcing frozen-program genes into the feature set. Matched random panels excluded genes from all four primary frozen programs, and max-over-factors null correction was used to account for latent-factor search.

### Single-cell biological localization

Canine single-cell RNA-sequencing data from GSE252470 were analyzed at the biological-dog level rather than the technical-library level. Six biological dogs were retained after accounting for duplicated technical libraries. Frozen program components were localized across osteoblast-lineage, immune, osteoclast, and cycling compartments. Exact sign-flip inference was performed across biological dogs; with six dogs, p=0.03125 represents the minimum attainable two-sided exact value.

### Independent canine representation validation

GSE239948 was used as an external canine expression-validation cohort. Exact normalized sample identifiers were compared against DOG² and expression-fingerprint screening was used to detect potential exact or near-duplicate profiles. Fingerprinting was treated as a screening procedure rather than proof of donor independence.

Direct external preservation was quantified using edge concordance, loading concordance, score reliability, and variance-matched random-panel controls. Blind de novo rediscovery was then performed using GSE239948 expression without access to frozen module membership. Frozen programs were matched to the discovered modules only after clustering was complete, and the empirical null repeated the same best-of-many-module search using variance-matched random panels. Finally, standard WGCNA modulePreservation was run with DOG² as reference and GSE239948 as test using 200 permutations.

### Exploratory pathological-response analysis

Baseline GSE39055 frozen scores were tested against subsequent pathological necrosis as an exploratory response-association analysis. Continuous necrosis and the fixed 90% good-response threshold were evaluated. This analysis did not estimate chemotherapy benefit because no untreated comparator was available and did not alter any frozen evidence tier.

### Statistical software and reproducibility

All primary processing and statistical analyses were implemented in Python, with the WGCNA modulePreservation benchmark run in R. [VERIFY: insert exact Python, R, pandas, scipy, lifelines, scikit-learn, statsmodels, WGCNA, and other package versions from the reproducibility environment.] All analysis scripts were executed without outcome-dependent command-line tuning, and locked manifests with SHA-256 hashes were retained for the final evidence layers."""


def results_text(numbers: dict[str, Any]) -> str:
    m34 = numbers["M34"]
    m11 = numbers["M11"]
    m24 = numbers["M24"]
    m40 = numbers["M40"]

    return f"""### Frozen canine programs retained distinct prognostic and proliferation-related properties

The canine discovery workflow prioritized four frozen programs for cross-species evaluation. M34, M11, and M24 were retained as primary non-proliferation programs, whereas M40 was treated separately as a proliferation-dominant axis with a reproducible residual component. Cross-fitted residualization analyses in DOG² showed that removal of generic proliferation did not eliminate the prognostic signal of several modules, while M40 remained the clearest example of a strong cycling/proliferation program with additional reproducible structure.

### Human outcome transport was strongest for M34 but was not universal

Frozen human scores were evaluated without refitting across three clinically distinct osteosarcoma settings (Figures 2–4). In TARGET-OS, M34 was nominally associated with overall survival in the frozen canine risk direction (HR per SD {m34['target_hr']:.2f}, 95% CI {m34['target_ci_low']:.2f}-{m34['target_ci_high']:.2f}; p={m34['target_p']:.3f}; project-wide q={m34['target_q12']:.3f}). M11 and M40 showed HRs above 1 but were not statistically supported, whereas M24 was directionally discordant.

In GSE21257, M34 showed the strongest zero-shot discrimination of five-year metastasis (AUC {m34['gse_auc']:.3f}, 95% CI {m34['gse_auc_low']:.3f}-{m34['gse_auc_high']:.3f}; p={fmt_p(m34['gse_p'])}; project-wide q={m34['gse_q12']:.4f}). This was the only one of the 12 frozen primary human tests that remained significant after project-wide false-discovery-rate correction. M11 and M24 showed nominal discrimination before project-wide correction (AUC {m11['gse_auc']:.3f} and {m24['gse_auc']:.3f}, respectively), while M40 showed a weaker trend (AUC {m40['gse_auc']:.3f}).

GSE39055 did not reproduce the frozen outcome direction for M34, M24, or the locked M40 score. M34 had an RFS HR of {m34['rfs_hr']:.2f} (95% CI {m34['rfs_ci_low']:.2f}-{m34['rfs_ci_high']:.2f}; p={m34['rfs_p']:.3f}), while M40 had an HR of {m40['rfs_hr']:.2f} (95% CI {m40['rfs_ci_low']:.2f}-{m40['rfs_ci_high']:.2f}; p={m40['rfs_p']:.3f}). Detection-aware reanalysis showed that M40 direction was assay-rule sensitive, whereas the M34 discordance was more stable across reasonable assay-processing rules. These results motivated explicit separation of molecular representation from prognostic transport rather than post-hoc score reorientation.

### Cross-species molecular representation and clinical transport were separable

Conservative human structural analyses showed that M34 and M40 retained the clearest molecular representation across multiple settings, but the clinical meaning of these representations differed by endpoint and platform (Figure 8). In particular, M40 could remain strongly represented at the network level while lacking stable prognostic transport. This pattern argues against treating structural preservation as equivalent to biomarker replication.

### Single-cell localization clarified the biological meaning of M34 and M40

Single-cell analysis across six biological dogs localized the M34 program primarily to an immune-related axis. Most detected M34 genes carried negative canine risk loadings, and the frozen signed score was higher in osteoblast-lineage cells than in immune cells, supporting interpretation of M34 as an immune-depletion or immune-exclusion risk program rather than a simple immune-abundance signature. All six dogs showed the same direction, yielding the minimum attainable two-sided exact sign-flip p-value at this sample size (p=0.03125).

M40 localized strongly to cycling states. Its positive-loading component was elevated in cycling osteoblast-lineage cells and also captured broader cycling activity across other proliferative compartments, supporting interpretation as a highly reproducible cell-cycle architecture rather than an osteoblast-specific marker.

### Independent canine validation strongly replicated M34 and M40 molecular architecture

GSE239948 provided an orthogonal canine representation-validation setting (Figures 5–7). Exact normalized identifier overlap with DOG² was zero, and no expression-fingerprint pair crossed the pre-specified near-duplicate thresholds. Direct preservation was strong for M34 (edge rho={m34['edge']:.3f}; loading rho={m34['loading']:.3f}) and M40 (edge rho={m40['edge']:.3f}; loading rho={m40['loading']:.3f}), but not for M11 or M24.

Standard WGCNA modulePreservation independently confirmed the same hierarchy. M34 showed Zsummary={m34['zsummary']:.2f}, with strong density and connectivity preservation, while M40 showed Zsummary={m40['zsummary']:.2f}. The six-gene M11 and M24 mappings yielded only moderate, module-size-sensitive WGCNA support and were not upgraded to strong preservation because the direct and blind analyses were not concordant.

Blind de novo analysis further strengthened the external canine evidence. Frozen memberships were withheld until after GSE239948 module discovery. M40 showed substantial recovery of the naturally available frozen genes in one independently discovered module (best-match F1={m40['blind_f1']:.3f}; BH q={m40['blind_q']:.4f}). M34 showed a smaller but statistically specific overlap (best-match F1={m34['blind_f1']:.3f}; BH q={m34['blind_q']:.4f}), supporting recovery of an M34-related co-expression core rather than exact reconstruction of the entire frozen module.

### M40 molecular reproducibility did not translate into a pathological-response biomarker

The exploratory GSE39055 pathological-necrosis analysis did not support M40 as a response marker. The locked score showed only a weak positive association with continuous necrosis and near-null discrimination of the fixed 90% good-response threshold. Detection-aware score variants remained directionally similar for continuous necrosis but did not produce nominal statistical support. The result therefore reinforced, rather than weakened, the distinction between reproducible molecular architecture and reproducible clinical association.

### Final evidence hierarchy

The locked multidimensional synthesis identified M34 as the principal translational program, combining the strongest human outcome evidence with replicated canine and cross-species representation. M40 provided the strongest architecture-reproducibility result but retained assay- and context-sensitive clinical transport. M11 showed the most directionally consistent but weakly supported outcome pattern with uncertain structural replication, whereas M24 remained limited and endpoint-specific (Figure 9)."""


def discussion_text(numbers: dict[str, Any]) -> str:
    m34 = numbers["M34"]
    m40 = numbers["M40"]

    return f"""This study separates two questions that are often conflated in cross-species biomarker research: whether a molecular program remains biologically represented in a new setting and whether the same program retains a stable association with clinical outcome. The resulting evidence hierarchy was not a simple ranking of prognostic signatures. Instead, M34 emerged as the strongest translational program, whereas M40 became the strongest demonstration that molecular reproducibility and clinical transportability can diverge.

M34 combined the only project-wide FDR-supported frozen human outcome result with nominal directional support in TARGET-OS and strong independent canine molecular replication. Its GSE21257 metastasis AUC of {m34['gse_auc']:.3f} and project-wide q={m34['gse_q12']:.4f} were accompanied by direct DOG²-to-GSE239948 edge/loading concordance, WGCNA preservation, and statistically significant blind recovery of an M34-related co-expression core. Single-cell localization further suggested that the canine risk direction reflects depletion of an immune-associated component rather than simple enrichment of immune cells. Together, these layers support M34 as a conserved tumor-microenvironment-related program while also showing that its prognostic effect is not universal across platforms and endpoints.

M40 provides an even clearer test of the central conceptual model. Its independent canine structural replication was exceptionally strong, including direct edge rho={m40['edge']:.3f}, loading rho={m40['loading']:.3f}, WGCNA Zsummary={m40['zsummary']:.2f}, and significant blind de novo rediscovery. Yet the same program lacked stable prognostic transport and did not show convincing association with pathological necrosis. The simplest interpretation is therefore not that M40 failed to reproduce molecularly, but that the biological meaning of a reproducible cycling architecture is conditioned by assay, cohort composition, endpoint, and clinical context.

The study also highlights the importance of small-module caution. M11 and M24 contained only six shared genes in the external canine WGCNA benchmark. Moderate WGCNA Zsummary values in such modules were not treated as sufficient evidence of preservation because direct loading/edge concordance and blind rediscovery were weak. This illustrates why no single preservation metric should determine cross-study transferability.

Several design choices reduce common sources of optimism. Module membership, ortholog mapping, score direction, and primary human scoring were frozen before outcome analysis. Human endpoints were kept separate rather than pooled. Project-wide multiplicity was controlled across all 12 primary human tests. Assay-aware GSE39055 analyses were used to diagnose instability but were not allowed to replace the frozen primary score. Independent canine validation used both targeted preservation and a blind de novo procedure whose null repeated the same best-of-many-module search. These safeguards make the negative and heterogeneous findings scientifically informative rather than failures to optimize.

The study has limitations. The human cohorts were modest in size and used different clinical endpoints and assay platforms. GSE39055 was particularly constrained by FFPE DASL measurement quality. The single-cell localization analysis included only six biological dogs, making p=0.03125 the minimum attainable two-sided exact sign-flip value. GSE239948 provided expression-based representation validation rather than clinical outcome validation, and expression fingerprinting cannot prove donor independence when identifier systems differ. Finally, the current analysis is predominantly transcriptomic; chromatin, spatial, methylation, and copy-number layers remain natural targets for future multimodal extension.

In conclusion, comparative osteosarcoma analysis revealed that highly reproducible cross-species molecular programs need not carry invariant clinical associations. M34 provides the strongest evidence for a translational immune-related risk program, while M40 demonstrates that a deeply conserved cycling architecture can remain clinically context-sensitive. Explicitly separating molecular representation transport from prognostic transport may improve the design and interpretation of cross-species biomarker studies beyond osteosarcoma."""


def build_manuscript(
    numbers: dict[str, Any],
    figure_captions: str,
) -> str:
    title = (
        "Cross-species osteosarcoma transcriptional programs show conserved "
        "molecular architecture but heterogeneous prognostic transport"
    )

    return f"""# {title}

## Authors

[First Author Name]^1, [Professor / Senior Author Name]^2

^1 [Affiliation 1 — VERIFY]  
^2 [Affiliation 2 — VERIFY]

**Corresponding author:** [VERIFY corresponding author, email, postal address]

**Target journal:** {TARGET_JOURNAL}

---

## Abstract

{abstract_text(numbers)}

## Keywords

osteosarcoma; comparative oncology; canine cancer; cross-species transcriptomics; prognostic transport; representation preservation; tumor microenvironment; proliferation; bioinformatics

---

## Introduction

{introduction_text()}

---

## Materials and Methods

{methods_text()}

---

## Results

{results_text(numbers)}

---

## Discussion

{discussion_text(numbers)}

---

## Conclusions

Cross-species molecular programs should not be classified as successfully translated solely because they preserve co-expression structure or solely because they show a favorable effect estimate in one clinical cohort. The present analysis supports M34 as the principal translational osteosarcoma program and M40 as a strongly conserved molecular architecture with context-sensitive clinical transport. These findings motivate transportability frameworks that evaluate molecular representation and clinical association as distinct evidence dimensions.

---

## Data availability

All datasets analyzed in this study were obtained from public repositories. [VERIFY: insert GEO, TARGET/GDC, and any other accession links used in the final manuscript.] Processed derivative tables required to reproduce the locked analyses will be deposited in [GitHub/Zenodo — VERIFY] to the extent permitted by source-data terms.

## Code availability

All analysis scripts used to generate the frozen evidence hierarchy, project-wide multiplicity analysis, representation-preservation analyses, single-cell localization, independent canine replication, and manuscript figures will be released at [GitHub repository — VERIFY]. Final locked analysis manifests include SHA-256 hashes of key inputs and outputs.

## Author contributions

**[First Author Name]:** Conceptualization, Data curation, Formal analysis, Investigation, Methodology, Software, Validation, Visualization, Writing – original draft, Writing – review & editing.

**[Professor / Senior Author Name]:** [VERIFY after substantive contribution: Conceptualization, Methodology, Supervision, Validation, Writing – review & editing.]

Do not finalize senior-author authorship until the author has reviewed the manuscript, made a substantive intellectual contribution, approved the final version, and agreed to accountability for the work.

## Funding

[VERIFY funding statement.]

## Conflict of interest

The authors declare [VERIFY].

## Acknowledgements

[VERIFY whether any contributors should be acknowledged.]

---

# Figure legends

{figure_captions}

---

# Main manuscript claim guardrails

1. Only GSE21257-M34 is project-wide FDR supported among the 12 frozen human primary tests.
2. TARGET-OS M34 is nominal directional support, not project-wide FDR support.
3. GSE39055 discordance must remain visible and must not be removed by selecting favorable assay rules.
4. Human OS, five-year metastasis, and RFS must not be meta-analyzed as one outcome.
5. M34 blind rediscovery is partial/core recovery rather than exact full-module reconstruction.
6. M40 is highly reproducible molecularly but is not a stable prognostic or chemotherapy-response biomarker.
7. M11 and M24 small-module WGCNA signals do not override weak direct/blind external replication.
8. GSE239948 is a representation-validation cohort, not a clinical outcome-validation cohort.
9. Single-cell exact p=0.03125 is the minimum attainable two-sided value for six biological dogs.
10. No post-lock outcome-driven feature selection, score reversal, threshold search, or evidence-tier reclassification.
"""


def build_authorship_checklist() -> str:
    return f"""Paper 4 authorship and submission checklist
=========================================

Target journal
--------------
{TARGET_JOURNAL}

Recommended immediate authorship plan
-------------------------------------
1. Keep the first-author position for the person who performed the project, analysis,
   manuscript drafting, and submission preparation.
2. Keep the professor as a prospective second/last author only if a substantive contribution
   is completed before submission.
3. Do not submit with a person's name as an author without their explicit approval.
4. Do not add an author only to improve prestige, affiliation, or APC eligibility.
5. Before submission, each author should approve the final manuscript and author-contribution statement.

Suggested senior-author contribution package
--------------------------------------------
- Review and refine the central conceptual framing.
- Critically review Methods and Discussion.
- Advise on bioinformatics / AI positioning.
- Review the interpretation of representation-versus-outcome transport.
- Approve final claims and limitations.
- Contribute to revision strategy after peer review.

Corresponding author decision
-----------------------------
[VERIFY]
The corresponding author should be the person who will manage submission, editor communication,
reviewer responses, and post-publication accountability. Do not select the corresponding author
only to obtain publication-fee coverage.

Submission-readiness checklist
------------------------------
[ ] Author names and affiliations finalized.
[ ] All authors approve the final version.
[ ] CRediT contributions finalized.
[ ] Funding statement verified.
[ ] Conflict-of-interest statement verified.
[ ] Data-availability statement verified.
[ ] Code repository public or archived.
[ ] Accession numbers verified.
[ ] Software/package versions inserted.
[ ] Figure order compressed to the final journal-ready set.
[ ] Supplementary Methods and Tables assembled.
[ ] References added and checked.
[ ] Cover letter drafted.
[ ] Journal-specific word limits and figure limits checked immediately before submission.
"""


def create_manifest(
    input_paths: list[Path],
    output_paths: list[Path],
) -> None:
    payload = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "target_journal": TARGET_JOURNAL,
        "analysis_type": "manuscript_generation_from_locked_outputs",
        "new_statistical_tests": False,
        "outcome_model_fitting": False,
        "feature_selection": False,
        "module_redefinition": False,
        "inputs": {},
        "outputs": {},
        "guardrails": [
            "The manuscript draft uses only locked numerical results.",
            "No new statistical analysis is performed.",
            "All unverified metadata are marked with VERIFY placeholders.",
            "Only GSE21257-M34 is described as project-wide FDR supported.",
            "M34 blind rediscovery is described as partial/core recovery.",
            "M40 is not described as a stable prognostic or chemotherapy-response biomarker.",
            "Authorship placeholders are not evidence of final authorship.",
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
    print("Generate Paper 4 manuscript draft from locked evidence")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Target journal: {TARGET_JOURNAL}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Design:")
    print("  Read only locked outputs from scripts 51 and 52.")
    print("  Perform no new statistical tests.")
    print("  Generate a manuscript draft with locked numerical results.")
    print("  Mark unverified metadata and software details with VERIFY placeholders.")
    print("  Generate a claim matrix and authorship/submission checklist.")
    print("")

    final_lock = read_required_json(FINAL_LOCK_MANIFEST_FILE)
    asset_manifest = read_required_json(ASSET_MANIFEST_FILE)
    verify_lock_manifests(final_lock, asset_manifest)

    main_results = read_required_csv(MAIN_RESULTS_FILE)
    human_results = read_required_csv(HUMAN_RESULTS_FILE)
    canine_results = read_required_csv(CANINE_TRIANGULATION_FILE)
    decoupling = read_required_csv(DECOUPLING_FILE)
    interpretation = read_required_csv(FINAL_INTERPRETATION_FILE)
    figure_captions = read_required_text(FIGURE_CAPTIONS_FILE)
    results_outline = read_required_text(RESULTS_OUTLINE_FILE)

    numbers = extract_locked_numbers(
        human=human_results,
        canine=canine_results,
    )
    validate_locked_story(numbers)

    manuscript = build_manuscript(
        numbers=numbers,
        figure_captions=figure_captions,
    )
    titles = build_title_options()
    highlights = build_highlights()
    claim_matrix = build_claim_matrix(
        numbers=numbers,
        interpretation=interpretation,
    )
    authorship_checklist = build_authorship_checklist()

    OUTPUT_MANUSCRIPT.write_text(manuscript, encoding="utf-8")
    OUTPUT_TITLE_OPTIONS.write_text(titles, encoding="utf-8")
    OUTPUT_HIGHLIGHTS.write_text(highlights, encoding="utf-8")
    claim_matrix.to_csv(OUTPUT_CLAIM_MATRIX, index=False)
    OUTPUT_AUTHOR_CHECKLIST.write_text(authorship_checklist, encoding="utf-8")

    print("=" * 80)
    print("Locked manuscript evidence check")
    print("=" * 80)
    check_rows = []
    for module in PRIMARY_MODULES:
        value = numbers[module]
        check_rows.append(
            {
                "module": module,
                "TARGET_HR": value["target_hr"],
                "TARGET_q12": value["target_q12"],
                "GSE21257_AUC": value["gse_auc"],
                "GSE21257_q12": value["gse_q12"],
                "GSE39055_RFS_HR": value["rfs_hr"],
                "GSE39055_q12": value["rfs_q12"],
                "external_edge_rho": value["edge"],
                "external_loading_rho": value["loading"],
                "WGCNA_Zsummary": value["zsummary"],
                "blind_q": value["blind_q"],
            }
        )
    print(pd.DataFrame(check_rows).to_string(index=False))

    print("")
    print("=" * 80)
    print("Manuscript claim matrix")
    print("=" * 80)
    print(
        claim_matrix[
            ["claim_id", "claim", "status", "guardrail"]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Draft guardrails")
    print("=" * 80)
    print("Only GSE21257-M34 is project-wide FDR supported.")
    print("M34 blind rediscovery is partial/core recovery.")
    print("M40 is not a stable prognostic or chemotherapy-response biomarker.")
    print("Human OS, metastasis, and RFS remain separate endpoints.")
    print("All unverified manuscript metadata remain marked as VERIFY.")

    output_paths = [
        OUTPUT_MANUSCRIPT,
        OUTPUT_TITLE_OPTIONS,
        OUTPUT_HIGHLIGHTS,
        OUTPUT_CLAIM_MATRIX,
        OUTPUT_AUTHOR_CHECKLIST,
    ]

    input_paths = [
        MAIN_RESULTS_FILE,
        HUMAN_RESULTS_FILE,
        CANINE_TRIANGULATION_FILE,
        DECOUPLING_FILE,
        FINAL_INTERPRETATION_FILE,
        FIGURE_CAPTIONS_FILE,
        RESULTS_OUTLINE_FILE,
        ASSET_MANIFEST_FILE,
        FINAL_LOCK_MANIFEST_FILE,
    ]

    create_manifest(input_paths, output_paths)

    print("")
    print("Saved:")
    for path in output_paths + [OUTPUT_MANIFEST]:
        print(path)
    print("Done.")


if __name__ == "__main__":
    main()
