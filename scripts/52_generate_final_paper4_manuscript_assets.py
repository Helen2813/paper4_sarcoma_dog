from __future__ import annotations

from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_VERSION = "52-generate-final-paper4-manuscript-assets-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "paper4_final_locked"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]

TARGET_FILE = RESULTS_DIR / "TARGET_OS_primary_frozen_program_validation.csv"
GSE21257_FILE = RESULTS_DIR / "GSE21257_metastasis_primary_frozen_program_validation.csv"
GSE39055_FILE = RESULTS_DIR / "GSE39055_RFS_primary_frozen_program_validation.csv"

PROJECTWIDE_MULTIPLICITY_FILE = RESULTS_DIR / "paper4_projectwide_primary_multiplicity.csv"
TYPOLOGY_FILE = RESULTS_DIR / "paper4_representation_outcome_decoupling_typology.csv"
ASSAY_AWARE_SUMMARY_FILE = RESULTS_DIR / "paper4_assay_aware_locked_module_evidence_summary.csv"

TRIANGULATION_FILE = RESULTS_DIR / "paper4_locked_external_canine_triangulation.csv"
FINAL_MASTER_FILE = RESULTS_DIR / "paper4_locked_multidimensional_transport_evidence_final.csv"
FINAL_INTERPRETATION_FILE = RESULTS_DIR / "paper4_locked_final_module_interpretation.csv"
FINAL_LOCK_MANIFEST_FILE = RESULTS_DIR / "paper4_final_analysis_lock_manifest.json"

CONSERVATIVE_STRUCTURE_FILE = RESULTS_DIR / "cross_cohort_conservative_preservation_classification.csv"

OUTPUT_MAIN_TABLE = RESULTS_DIR / "Paper4_final_main_results_table.csv"
OUTPUT_MAIN_TABLE_TEX = RESULTS_DIR / "Paper4_final_main_results_table.tex"
OUTPUT_HUMAN_TABLE = RESULTS_DIR / "Paper4_final_human_primary_outcomes_table.csv"
OUTPUT_CANINE_TABLE = RESULTS_DIR / "Paper4_final_external_canine_triangulation_table.csv"
OUTPUT_DECOUPLING_TABLE = RESULTS_DIR / "Paper4_final_representation_outcome_typology_table.csv"
OUTPUT_CAPTIONS = RESULTS_DIR / "Paper4_final_figure_captions.txt"
OUTPUT_RESULTS_OUTLINE = RESULTS_DIR / "Paper4_final_results_section_outline.txt"
OUTPUT_README = RESULTS_DIR / "Paper4_final_manuscript_assets_README.txt"
OUTPUT_MANIFEST = RESULTS_DIR / "Paper4_final_manuscript_assets_manifest.json"

FIGURE_FILES = {
    "study_design": FIGURES_DIR / "Figure1_study_design",
    "target": FIGURES_DIR / "Figure2_TARGET_OS_primary_effects",
    "gse21257": FIGURES_DIR / "Figure3_GSE21257_primary_auc",
    "gse39055": FIGURES_DIR / "Figure4_GSE39055_primary_effects",
    "direct_canine": FIGURES_DIR / "Figure5_external_canine_direct_preservation",
    "wgcna": FIGURES_DIR / "Figure6_external_canine_WGCNA_preservation",
    "blind": FIGURES_DIR / "Figure7_external_canine_blind_rediscovery",
    "decoupling": FIGURES_DIR / "Figure8_representation_outcome_decoupling",
    "evidence_matrix": FIGURES_DIR / "Figure9_final_evidence_matrix",
}


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


def format_num(value: Any, digits: int = 3) -> str:
    value = safe_float(value)
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def format_p(value: Any) -> str:
    value = safe_float(value)
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=400, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def ordered(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result["module_label"] = pd.Categorical(
        result["module_label"],
        categories=PRIMARY_MODULES,
        ordered=True,
    )
    return result.sort_values("module_label").reset_index(drop=True)


def verify_final_lock(manifest: dict[str, Any]) -> None:
    expected = "51-final-lock-external-canine-triangulation-v1"
    observed = str(manifest.get("script_version", ""))
    if observed != expected:
        raise RuntimeError(
            f"Unexpected final lock version: {observed}. Expected: {expected}"
        )
    if bool(manifest.get("new_statistical_tests", False)):
        raise RuntimeError("Final lock manifest unexpectedly reports new statistical tests.")
    if bool(manifest.get("feature_selection", False)):
        raise RuntimeError("Final lock manifest unexpectedly reports feature selection.")
    if bool(manifest.get("outcome_model_fitting", False)):
        raise RuntimeError("Final lock manifest unexpectedly reports outcome fitting.")


def projectwide_map(multiplicity: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    require_columns(
        multiplicity,
        [
            "cohort",
            "module_label",
            "primary_p",
            "projectwide_q_12",
            "projectwide_fdr_supported",
        ],
        "Project-wide multiplicity table",
    )
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in multiplicity.iterrows():
        result[(str(row["cohort"]), str(row["module_label"]))] = row.to_dict()
    return result


def build_human_primary_table(
    target: pd.DataFrame,
    gse21257: pd.DataFrame,
    gse39055: pd.DataFrame,
    multiplicity: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        target,
        [
            "module_label",
            "score_hr_per_sd",
            "score_ci_low",
            "score_ci_high",
            "primary_p",
            "fixed_score_c_index",
        ],
        "TARGET-OS table",
    )
    require_columns(
        gse21257,
        [
            "module_label",
            "auc",
            "auc_ci_low",
            "auc_ci_high",
            "primary_p",
            "average_precision",
        ],
        "GSE21257 table",
    )
    require_columns(
        gse39055,
        [
            "module_label",
            "hr_per_sd",
            "ci_low",
            "ci_high",
            "primary_p",
            "fixed_score_c_index",
        ],
        "GSE39055 table",
    )

    target_i = target.set_index("module_label")
    gse_i = gse21257.set_index("module_label")
    rfs_i = gse39055.set_index("module_label")
    qmap = projectwide_map(multiplicity)

    rows = []
    for module in PRIMARY_MODULES:
        target_mult = qmap[("TARGET_OS", module)]
        gse_mult = qmap[("GSE21257", module)]
        rfs_mult = qmap[("GSE39055", module)]

        rows.append(
            {
                "module_label": module,
                "TARGET_OS_HR_per_SD": target_i.loc[module, "score_hr_per_sd"],
                "TARGET_OS_CI_low": target_i.loc[module, "score_ci_low"],
                "TARGET_OS_CI_high": target_i.loc[module, "score_ci_high"],
                "TARGET_OS_primary_p": target_i.loc[module, "primary_p"],
                "TARGET_OS_projectwide_q12": target_mult["projectwide_q_12"],
                "TARGET_OS_fixed_direction_C_index": target_i.loc[
                    module, "fixed_score_c_index"
                ],
                "GSE21257_AUC": gse_i.loc[module, "auc"],
                "GSE21257_AUC_CI_low": gse_i.loc[module, "auc_ci_low"],
                "GSE21257_AUC_CI_high": gse_i.loc[module, "auc_ci_high"],
                "GSE21257_primary_p": gse_i.loc[module, "primary_p"],
                "GSE21257_projectwide_q12": gse_mult["projectwide_q_12"],
                "GSE21257_average_precision": gse_i.loc[module, "average_precision"],
                "GSE39055_RFS_HR_per_SD": rfs_i.loc[module, "hr_per_sd"],
                "GSE39055_RFS_CI_low": rfs_i.loc[module, "ci_low"],
                "GSE39055_RFS_CI_high": rfs_i.loc[module, "ci_high"],
                "GSE39055_RFS_primary_p": rfs_i.loc[module, "primary_p"],
                "GSE39055_RFS_projectwide_q12": rfs_mult["projectwide_q_12"],
                "GSE39055_fixed_direction_C_index": rfs_i.loc[
                    module, "fixed_score_c_index"
                ],
                "n_projectwide_fdr_supported_settings": int(
                    bool(target_mult["projectwide_fdr_supported"])
                    + bool(gse_mult["projectwide_fdr_supported"])
                    + bool(rfs_mult["projectwide_fdr_supported"])
                ),
            }
        )

    return pd.DataFrame(rows)


def build_main_results_table(
    human: pd.DataFrame,
    triangulation: pd.DataFrame,
    interpretation: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        triangulation,
        [
            "module_label",
            "edge_spearman",
            "loading_spearman",
            "zsummary_pres",
            "best_match_f1",
            "empirical_max_match_q_bh_4",
            "external_canine_triangulation_class",
        ],
        "External canine triangulation",
    )
    require_columns(
        interpretation,
        [
            "module_label",
            "final_multidimensional_transport_class",
            "final_locked_interpretation",
        ],
        "Final interpretation",
    )

    result = human.merge(
        triangulation[
            [
                "module_label",
                "edge_spearman",
                "loading_spearman",
                "zsummary_pres",
                "zdensity_pres",
                "zconnectivity_pres",
                "median_rank_pres",
                "best_match_f1",
                "empirical_max_match_q_bh_4",
                "external_canine_triangulation_class",
            ]
        ],
        on="module_label",
        how="left",
    )
    result = result.merge(
        interpretation[
            [
                "module_label",
                "final_multidimensional_transport_class",
                "final_locked_interpretation",
            ]
        ],
        on="module_label",
        how="left",
    )
    return ordered(result)


def manuscript_table_view(main: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in main.iterrows():
        module = str(row["module_label"])
        rows.append(
            {
                "Module": module,
                "TARGET-OS overall survival": (
                    f"HR {format_num(row['TARGET_OS_HR_per_SD'], 2)} "
                    f"({format_num(row['TARGET_OS_CI_low'], 2)}–"
                    f"{format_num(row['TARGET_OS_CI_high'], 2)}), "
                    f"p={format_p(row['TARGET_OS_primary_p'])}, "
                    f"q12={format_p(row['TARGET_OS_projectwide_q12'])}"
                ),
                "GSE21257 metastasis": (
                    f"AUC {format_num(row['GSE21257_AUC'], 3)} "
                    f"({format_num(row['GSE21257_AUC_CI_low'], 3)}–"
                    f"{format_num(row['GSE21257_AUC_CI_high'], 3)}), "
                    f"p={format_p(row['GSE21257_primary_p'])}, "
                    f"q12={format_p(row['GSE21257_projectwide_q12'])}"
                ),
                "GSE39055 recurrence-free survival": (
                    f"HR {format_num(row['GSE39055_RFS_HR_per_SD'], 2)} "
                    f"({format_num(row['GSE39055_RFS_CI_low'], 2)}–"
                    f"{format_num(row['GSE39055_RFS_CI_high'], 2)}), "
                    f"p={format_p(row['GSE39055_RFS_primary_p'])}, "
                    f"q12={format_p(row['GSE39055_RFS_projectwide_q12'])}"
                ),
                "Independent canine representation": (
                    f"edge ρ={format_num(row['edge_spearman'], 3)}; "
                    f"loading ρ={format_num(row['loading_spearman'], 3)}; "
                    f"WGCNA Z={format_num(row['zsummary_pres'], 2)}; "
                    f"blind F1={format_num(row['best_match_f1'], 3)}, "
                    f"q={format_p(row['empirical_max_match_q_bh_4'])}"
                ),
                "Locked interpretation": row["final_multidimensional_transport_class"],
            }
        )
    return pd.DataFrame(rows)


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def write_latex_table(table: pd.DataFrame) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Locked cross-species evidence for the four primary canine transcriptional programs. Human endpoints are displayed separately and are not pooled.}",
        r"\begin{tabular}{p{0.07\textwidth} p{0.18\textwidth} p{0.18\textwidth} p{0.18\textwidth} p{0.20\textwidth}}",
        r"\hline",
        r"Module & TARGET-OS OS & GSE21257 metastasis & GSE39055 RFS & Independent canine representation \\",
        r"\hline",
    ]
    for _, row in table.iterrows():
        cells = [
            latex_escape(row["Module"]),
            latex_escape(row["TARGET-OS overall survival"]),
            latex_escape(row["GSE21257 metastasis"]),
            latex_escape(row["GSE39055 recurrence-free survival"]),
            latex_escape(row["Independent canine representation"]),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    OUTPUT_MAIN_TABLE_TEX.write_text("\n".join(lines), encoding="utf-8")


def figure_study_design() -> list[Path]:
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.axis("off")

    boxes = [
        (
            0.08,
            0.55,
            "Canine DOG² discovery\n186 tumors\nFrozen module membership,\nweights and risk direction",
        ),
        (
            0.31,
            0.55,
            "Outcome-blind representation\nStrict ortholog transfer\nProliferation controls\nLatent-factor analyses",
        ),
        (
            0.54,
            0.72,
            "Human outcome settings\nTARGET-OS: overall survival\nGSE21257: 5-y metastasis\nGSE39055: RFS",
        ),
        (
            0.54,
            0.34,
            "Independent canine replication\nGSE239948: 43 tumors\nDirect preservation\nBlind de novo rediscovery\nWGCNA benchmark",
        ),
        (
            0.80,
            0.55,
            "Locked multidimensional evidence\nRepresentation preservation\nOutcome transport\nAssay sensitivity\nBiological localization",
        ),
    ]

    for x, y, text in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10.5,
            transform=ax.transAxes,
            bbox={"boxstyle": "round,pad=0.5"},
        )

    arrows = [
        ((0.17, 0.55), (0.22, 0.55)),
        ((0.40, 0.55), (0.46, 0.68)),
        ((0.40, 0.55), (0.46, 0.40)),
        ((0.63, 0.72), (0.71, 0.60)),
        ((0.63, 0.34), (0.71, 0.50)),
    ]
    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "lw": 1.5},
        )

    ax.set_title(
        "Locked analysis design: separating molecular representation from clinical-outcome transport",
        fontsize=13,
        pad=18,
    )
    return save_figure(fig, FIGURE_FILES["study_design"])


def forest_hr(
    table: pd.DataFrame,
    effect_col: str,
    low_col: str,
    high_col: str,
    title: str,
    xlabel: str,
    stem: Path,
) -> list[Path]:
    part = ordered(table)
    y = np.arange(len(part))[::-1]
    effect = pd.to_numeric(part[effect_col], errors="coerce").to_numpy(float)
    low = pd.to_numeric(part[low_col], errors="coerce").to_numpy(float)
    high = pd.to_numeric(part[high_col], errors="coerce").to_numpy(float)

    xerr = np.vstack([effect - low, high - effect])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.errorbar(
        effect,
        y,
        xerr=xerr,
        fmt="o",
        capsize=4,
        linewidth=1.4,
    )
    ax.axvline(1.0, linestyle="--", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(part["module_label"].astype(str))
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.20)
    fig.tight_layout()
    return save_figure(fig, stem)


def figure_gse21257_auc(gse: pd.DataFrame) -> list[Path]:
    part = ordered(gse)
    y = np.arange(len(part))[::-1]
    auc = pd.to_numeric(part["auc"], errors="coerce").to_numpy(float)
    low = pd.to_numeric(part["auc_ci_low"], errors="coerce").to_numpy(float)
    high = pd.to_numeric(part["auc_ci_high"], errors="coerce").to_numpy(float)
    xerr = np.vstack([auc - low, high - auc])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.errorbar(
        auc,
        y,
        xerr=xerr,
        fmt="o",
        capsize=4,
        linewidth=1.4,
    )
    ax.axvline(0.5, linestyle="--", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(part["module_label"].astype(str))
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel("ROC-AUC for metastasis within five years")
    ax.set_title("GSE21257: frozen zero-shot metastasis discrimination")
    ax.grid(axis="x", alpha=0.20)
    fig.tight_layout()
    return save_figure(fig, FIGURE_FILES["gse21257"])


def figure_direct_canine(tri: pd.DataFrame) -> list[Path]:
    part = ordered(tri)
    y = np.arange(len(part))[::-1]

    fig, ax = plt.subplots(figsize=(7.5, 4.9))
    ax.scatter(
        pd.to_numeric(part["edge_spearman"], errors="coerce"),
        y + 0.10,
        marker="o",
        s=65,
        label="Edge Spearman",
    )
    ax.scatter(
        pd.to_numeric(part["loading_spearman"], errors="coerce"),
        y - 0.10,
        marker="s",
        s=60,
        label="Loading Spearman",
    )
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlim(-0.35, 1.02)
    ax.set_yticks(y)
    ax.set_yticklabels(part["module_label"].astype(str))
    ax.set_xlabel("DOG²–GSE239948 concordance")
    ax.set_title("Independent canine direct representation preservation")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.20)
    fig.tight_layout()
    return save_figure(fig, FIGURE_FILES["direct_canine"])


def figure_wgcna(tri: pd.DataFrame) -> list[Path]:
    part = ordered(tri)
    y = np.arange(len(part))[::-1]
    values = pd.to_numeric(part["zsummary_pres"], errors="coerce").to_numpy(float)

    fig, ax = plt.subplots(figsize=(7.5, 4.9))
    ax.barh(y, values)
    ax.axvline(2.0, linestyle="--", linewidth=1.0)
    ax.axvline(10.0, linestyle="--", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(part["module_label"].astype(str))
    ax.set_xlabel("WGCNA Zsummary preservation statistic")
    ax.set_title("Standard WGCNA modulePreservation benchmark")
    ax.grid(axis="x", alpha=0.20)

    for yi, value in zip(y, values):
        if np.isfinite(value):
            ax.text(value, yi, f"  {value:.1f}", va="center", fontsize=9)

    fig.tight_layout()
    return save_figure(fig, FIGURE_FILES["wgcna"])


def figure_blind_rediscovery(tri: pd.DataFrame) -> list[Path]:
    part = ordered(tri)
    y = np.arange(len(part))[::-1]
    f1 = pd.to_numeric(part["best_match_f1"], errors="coerce").to_numpy(float)
    q95 = pd.to_numeric(part["random_max_f1_q95"], errors="coerce").to_numpy(float)

    fig, ax = plt.subplots(figsize=(7.5, 4.9))
    ax.scatter(
        f1,
        y + 0.10,
        marker="o",
        s=65,
        label="Observed best-match F1",
    )
    ax.scatter(
        q95,
        y - 0.10,
        marker="x",
        s=65,
        label="95th percentile random best-match F1",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(part["module_label"].astype(str))
    ax.set_xlabel("Best-match F1")
    ax.set_title("Outcome-blind de novo rediscovery in GSE239948")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.20)
    fig.tight_layout()
    return save_figure(fig, FIGURE_FILES["blind"])


def structure_position(value: str) -> float:
    mapping = {
        "preserved": 3.0,
        "partial": 2.0,
        "limited": 1.0,
        "not_preserved": 0.0,
        "unknown": 0.0,
    }
    return mapping.get(str(value), 0.0)


def outcome_position(row: pd.Series) -> float:
    assay = str(row.get("assay_stability_status", ""))
    if assay in {
        "assay_rule_sensitive_direction",
        "not_detection_filter_estimable",
    }:
        return 0.5
    return 1.0 if bool(row["outcome_direction_concordant"]) else 0.0


def figure_decoupling(typology: pd.DataFrame) -> list[Path]:
    require_columns(
        typology,
        [
            "cohort",
            "module_label",
            "structure_class_simplified",
            "outcome_direction_concordant",
            "assay_stability_status",
        ],
        "Decoupling typology",
    )

    cohort_markers = {
        "TARGET_OS": "o",
        "GSE21257": "s",
        "GSE39055": "^",
    }
    offsets = {
        "M34": -0.18,
        "M11": -0.06,
        "M24": 0.06,
        "M40": 0.18,
    }

    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    for cohort, marker in cohort_markers.items():
        part = typology[typology["cohort"].eq(cohort)].copy()
        xs = []
        ys = []
        for _, row in part.iterrows():
            xs.append(
                structure_position(row["structure_class_simplified"])
                + offsets.get(str(row["module_label"]), 0.0)
            )
            ys.append(outcome_position(row))
        ax.scatter(xs, ys, marker=marker, s=85, label=cohort)

        for x, y, (_, row) in zip(xs, ys, part.iterrows()):
            ax.text(
                x,
                y + 0.045,
                str(row["module_label"]),
                ha="center",
                va="bottom",
                fontsize=8.8,
            )

    ax.axhline(0.5, linestyle=":", linewidth=1.0)
    ax.set_xlim(-0.45, 3.45)
    ax.set_ylim(-0.15, 1.18)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(
        [
            "No clear\npreservation",
            "Limited",
            "Partial",
            "Preserved",
        ]
    )
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(
        [
            "Discordant\ndirection",
            "Assay-sensitive /\nnot estimable",
            "Concordant\ndirection",
        ]
    )
    ax.set_xlabel("Conservative molecular-representation evidence")
    ax.set_ylabel("Frozen prognostic-direction transport")
    ax.set_title("Representation–outcome decoupling across human osteosarcoma settings")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return save_figure(fig, FIGURE_FILES["decoupling"])


def evidence_code(
    module: str,
    cohort: str,
    multiplicity: pd.DataFrame,
    typology: pd.DataFrame,
) -> str:
    mult = multiplicity[
        multiplicity["cohort"].eq(cohort)
        & multiplicity["module_label"].eq(module)
    ].iloc[0]
    typ = typology[
        typology["cohort"].eq(cohort)
        & typology["module_label"].eq(module)
    ].iloc[0]

    if bool(mult["projectwide_fdr_supported"]):
        outcome = "PW-FDR"
    elif bool(mult["nominal_supported"]):
        outcome = "Nominal"
    else:
        outcome = "Direction" if bool(mult["direction_concordant"]) else "Discordant"

    assay = str(typ["assay_stability_status"])
    if assay == "assay_rule_sensitive_direction":
        outcome = "Assay-sensitive"
    elif assay == "not_detection_filter_estimable":
        outcome = "Not estimable"

    structure = str(typ["structure_class_simplified"])
    structure_short = {
        "preserved": "Preserved",
        "partial": "Partial",
        "limited": "Limited",
        "not_preserved": "No clear",
        "unknown": "Unknown",
    }.get(structure, structure)

    return f"{structure_short}\n{outcome}"


def figure_evidence_matrix(
    multiplicity: pd.DataFrame,
    typology: pd.DataFrame,
    tri: pd.DataFrame,
    interpretation: pd.DataFrame,
) -> list[Path]:
    tri_i = tri.set_index("module_label")
    interpretation_i = interpretation.set_index("module_label")

    columns = [
        "TARGET-OS",
        "GSE21257",
        "GSE39055",
        "GSE239948 direct",
        "GSE239948 WGCNA",
        "GSE239948 blind",
        "Final class",
    ]

    cell_text = []
    for module in PRIMARY_MODULES:
        direct_class = str(
            tri_i.loc[module, "external_canine_representation_class"]
        )
        direct_label = (
            "Strong"
            if direct_class == "strong_external_canine_representation_preservation"
            else "No clear"
        )

        zsummary = safe_float(tri_i.loc[module, "zsummary_pres"])
        if np.isfinite(zsummary) and zsummary >= 10:
            wgcna_label = f"Strong\nZ={zsummary:.1f}"
        elif np.isfinite(zsummary) and zsummary >= 2:
            wgcna_label = f"Moderate\nZ={zsummary:.1f}"
        else:
            wgcna_label = f"Limited\nZ={zsummary:.1f}" if np.isfinite(zsummary) else "NA"

        blind_q = safe_float(tri_i.loc[module, "empirical_max_match_q_bh_4"])
        blind_f1 = safe_float(tri_i.loc[module, "best_match_f1"])
        blind_label = (
            f"Supported\nF1={blind_f1:.2f}\nq={blind_q:.3f}"
            if np.isfinite(blind_q) and blind_q < 0.05
            else f"No clear\nF1={blind_f1:.2f}"
        )

        final_class = str(
            interpretation_i.loc[module, "final_multidimensional_transport_class"]
        )
        final_short = {
            "replicated_canine_and_cross_species_representation_with_endpoint_and_platform_dependent_outcome_transport":
                "Replicated representation\nheterogeneous outcome",
            "triangulated_canine_and_cross_species_cycling_representation_with_assay_sensitive_outcome_transport":
                "Strong architecture\nassay-sensitive outcome",
            "directional_outcome_concordance_with_small_module_representation_uncertainty":
                "Directional consistency\nrepresentation uncertain",
            "limited_representation_and_endpoint_specific_outcome_evidence":
                "Limited / endpoint-specific",
        }.get(final_class, final_class)

        cell_text.append(
            [
                evidence_code(module, "TARGET_OS", multiplicity, typology),
                evidence_code(module, "GSE21257", multiplicity, typology),
                evidence_code(module, "GSE39055", multiplicity, typology),
                direct_label,
                wgcna_label,
                blind_label,
                final_short,
            ]
        )

    fig, ax = plt.subplots(figsize=(15.5, 5.3))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=PRIMARY_MODULES,
        colLabels=columns,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 2.2)
    ax.set_title(
        "Locked multidimensional evidence: molecular representation and clinical-outcome transport",
        fontsize=12.5,
        pad=18,
    )
    fig.tight_layout()
    return save_figure(fig, FIGURE_FILES["evidence_matrix"])


def write_captions(
    target: pd.DataFrame,
    gse21257: pd.DataFrame,
    gse39055: pd.DataFrame,
    tri: pd.DataFrame,
) -> None:
    target_i = target.set_index("module_label")
    gse_i = gse21257.set_index("module_label")
    rfs_i = gse39055.set_index("module_label")
    tri_i = tri.set_index("module_label")

    text = f"""Figure 1. Locked study design.
Canine DOG² discovery was frozen before human outcome analysis. Human validation, independent canine representation validation, assay-quality diagnostics, latent-factor analyses, and biological localization were treated as distinct evidence layers. No human result was used to revise module membership, score direction, or validation tier.

Figure 2. TARGET-OS overall-survival associations.
Hazard ratios per 1-SD increase in the frozen strict one-to-one signed-mean scores are shown with 95% confidence intervals. M34 showed a nominal directionally concordant association with overall survival (HR={target_i.loc['M34', 'score_hr_per_sd']:.2f}, 95% CI {target_i.loc['M34', 'score_ci_low']:.2f}-{target_i.loc['M34', 'score_ci_high']:.2f}, p={target_i.loc['M34', 'primary_p']:.3f}). Human endpoints were not pooled.

Figure 3. GSE21257 five-year metastasis discrimination.
Frozen zero-shot scores were evaluated without human refitting. M34 showed the strongest discrimination (AUC={gse_i.loc['M34', 'auc']:.3f}, 95% CI {gse_i.loc['M34', 'auc_ci_low']:.3f}-{gse_i.loc['M34', 'auc_ci_high']:.3f}). This was the only frozen primary human outcome test that remained significant after project-wide Benjamini-Hochberg correction across all 12 primary tests.

Figure 4. GSE39055 recurrence-free-survival associations.
Locked strict scores are displayed with the original canine risk direction. M34, M24, and M40 did not show concordant locked RFS directions; interpretation of the FFPE cohort is constrained by assay-quality sensitivity. Alternative detection-aware rules are sensitivity analyses and do not replace the locked primary models.

Figure 5. Independent canine direct representation preservation.
DOG²-to-GSE239948 edge and PC1-loading Spearman correlations are shown for the four primary frozen programs. M34 and M40 displayed strong concordance, whereas the six-gene M11 and M24 external mappings did not show clear direct preservation.

Figure 6. Standard WGCNA preservation benchmark.
WGCNA modulePreservation Zsummary statistics in GSE239948 are shown using DOG² as the reference network. M34 (Zsummary={tri_i.loc['M34', 'zsummary_pres']:.2f}) and M40 (Zsummary={tri_i.loc['M40', 'zsummary_pres']:.2f}) showed strong preservation. M11 and M24 are interpreted cautiously because only six shared genes were available and Zsummary is module-size dependent.

Figure 7. Blind de novo rediscovery in GSE239948.
Frozen memberships were withheld until after outcome-blind de novo clustering. M34 and M40 exceeded the best-of-many-module variance-matched random-panel null (BH q={tri_i.loc['M34', 'empirical_max_match_q_bh_4']:.3f} and {tri_i.loc['M40', 'empirical_max_match_q_bh_4']:.3f}, respectively). M34 is interpreted as statistically supported partial core recovery rather than exact reconstruction of the full frozen module.

Figure 8. Representation-outcome decoupling.
Each human module-cohort pair is positioned according to conservative molecular-representation evidence and preservation of the frozen canine prognostic direction. GSE39055 assay-sensitive or non-estimable directions are displayed separately rather than treated as definitive biological reversals. The figure emphasizes that molecular representation and prognostic transport are separable quantities.

Figure 9. Final locked evidence matrix.
Human outcome support, conservative structural preservation, and three independent canine representation analyses are summarized without pooling distinct clinical endpoints. M34 represents the strongest translational program, whereas M40 provides the strongest molecular-architecture reproducibility with assay- and context-sensitive clinical transport.
"""
    OUTPUT_CAPTIONS.write_text(text, encoding="utf-8")


def write_results_outline(
    main: pd.DataFrame,
    multiplicity: pd.DataFrame,
) -> None:
    m34 = main.set_index("module_label").loc["M34"]
    m40 = main.set_index("module_label").loc["M40"]
    pw = multiplicity[multiplicity["projectwide_fdr_supported"]].copy()

    lines = [
        "Paper 4 Results section outline",
        "==============================",
        "",
        "1. Frozen canine programs capture prognostic biology beyond generic proliferation",
        "- Introduce the frozen hierarchy and repeated cross-fitted canine evidence.",
        "- Keep M40 separate as a proliferation-dominant deviation axis.",
        "",
        "2. Human outcome transport is heterogeneous across endpoints and platforms",
        (
            f"- M34: TARGET-OS HR {m34['TARGET_OS_HR_per_SD']:.2f} "
            f"({m34['TARGET_OS_CI_low']:.2f}-{m34['TARGET_OS_CI_high']:.2f}), "
            f"GSE21257 AUC {m34['GSE21257_AUC']:.3f}, "
            f"GSE39055 RFS HR {m34['GSE39055_RFS_HR_per_SD']:.2f}."
        ),
        f"- Project-wide FDR-supported human primary tests: {pw.shape[0]}.",
        "- State explicitly that OS, five-year metastasis, and RFS are not meta-analyzed.",
        "",
        "3. Molecular representation and prognostic transport are separable",
        "- Use the conservative human preservation analysis and the decoupling typology.",
        "- Do not post-hoc reverse frozen score directions.",
        "",
        "4. Independent canine validation confirms M34 and M40 as reproducible molecular architectures",
        (
            f"- M34: edge rho {m34['edge_spearman']:.3f}, "
            f"loading rho {m34['loading_spearman']:.3f}, "
            f"WGCNA Zsummary {m34['zsummary_pres']:.2f}, "
            f"blind F1 {m34['best_match_f1']:.3f}."
        ),
        (
            f"- M40: edge rho {m40['edge_spearman']:.3f}, "
            f"loading rho {m40['loading_spearman']:.3f}, "
            f"WGCNA Zsummary {m40['zsummary_pres']:.2f}, "
            f"blind F1 {m40['best_match_f1']:.3f}."
        ),
        "- Describe M34 blind rediscovery as recovery of a stable related core, not full-boundary reconstruction.",
        "",
        "5. Biological convergence clarifies program meaning",
        "- M34: immune-exclusion risk interpretation supported by TME and single-cell analyses.",
        "- M40: cycling/proliferation architecture is highly reproducible, but clinical meaning is assay/context sensitive.",
        "- Retain the negative pathological-necrosis analysis as an exploratory limitation.",
        "",
        "6. Final evidence hierarchy",
        "- M34: principal translational program.",
        "- M40: strongest architecture-reproducibility result.",
        "- M11: directional consistency with small-module representation uncertainty.",
        "- M24: limited and endpoint-specific transport.",
        "",
        "Writing restriction",
        "-------------------",
        "Do not add new outcome-driven tuning after the final lock.",
    ]
    OUTPUT_RESULTS_OUTLINE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme() -> None:
    text = f"""Paper 4 final manuscript assets
Script version: {SCRIPT_VERSION}

Purpose
-------
Generate manuscript-ready figures, tables, captions, and a Results-section outline
from locked outputs only.

No new analyses
---------------
This script performs no:
- feature selection,
- outcome model fitting,
- score orientation,
- clustering,
- network fitting,
- multiplicity testing,
- module redefinition.

Source hierarchy
----------------
- Scripts 23 and 26: locked human primary outcome analyses.
- Script 33: project-wide BH across all 12 frozen primary human tests and assay-aware typology.
- Script 28: conservative human structure-preservation classes.
- Script 51: final external canine direct/blind/WGCNA triangulation and module interpretations.

Figure design
-------------
Human endpoints are plotted separately because HRs and AUCs are not interchangeable.
The representation-outcome figure is categorical and does not create a synthetic pooled effect.
The evidence matrix is descriptive and uses locked classifications only.

Primary manuscript emphasis
---------------------------
M34 is the principal translational program.
M40 is the strongest molecular-architecture reproducibility result.
M11 is a secondary directional-consistency finding.
M24 is a limited or endpoint-specific transfer result.

Final restriction
-----------------
After this asset-generation stage, further work should focus on manuscript writing,
figure layout, supplementary organization, and reproducibility packaging rather than
new outcome-driven discovery.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def create_manifest(input_paths: list[Path], output_paths: list[Path]) -> None:
    payload = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "analysis_type": "locked_asset_generation_only",
        "new_statistical_tests": False,
        "outcome_model_fitting": False,
        "feature_selection": False,
        "module_redefinition": False,
        "inputs": {},
        "outputs": {},
        "guardrails": [
            "Human endpoints remain separate and are not pooled.",
            "Only script 33 project-wide multiplicity is displayed.",
            "M34 blind rediscovery is described as partial/core recovery.",
            "M40 outcome transport remains assay/context sensitive.",
            "M11 and M24 small-module WGCNA support does not override negative direct/blind evidence.",
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
    print("Generate final Paper 4 manuscript figures and tables")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Figures directory: {FIGURES_DIR}")
    print("")
    print("Design:")
    print("  Read locked evidence only.")
    print("  Generate endpoint-specific human outcome figures.")
    print("  Generate independent canine preservation triangulation figures.")
    print("  Generate representation-outcome decoupling and final evidence figures.")
    print("  Perform no new statistical testing.")
    print("")

    manifest = read_required_json(FINAL_LOCK_MANIFEST_FILE)
    verify_final_lock(manifest)

    target = read_required_csv(TARGET_FILE)
    gse21257 = read_required_csv(GSE21257_FILE)
    gse39055 = read_required_csv(GSE39055_FILE)
    multiplicity = read_required_csv(PROJECTWIDE_MULTIPLICITY_FILE)
    typology = read_required_csv(TYPOLOGY_FILE)
    assay_summary = read_required_csv(ASSAY_AWARE_SUMMARY_FILE)
    triangulation = read_required_csv(TRIANGULATION_FILE)
    final_master = read_required_csv(FINAL_MASTER_FILE)
    interpretation = read_required_csv(FINAL_INTERPRETATION_FILE)
    conservative_structure = read_required_csv(CONSERVATIVE_STRUCTURE_FILE)

    human_table = build_human_primary_table(
        target=target,
        gse21257=gse21257,
        gse39055=gse39055,
        multiplicity=multiplicity,
    )
    main_table = build_main_results_table(
        human=human_table,
        triangulation=triangulation,
        interpretation=interpretation,
    )
    manuscript_table = manuscript_table_view(main_table)

    human_table.to_csv(OUTPUT_HUMAN_TABLE, index=False)
    ordered(triangulation).to_csv(OUTPUT_CANINE_TABLE, index=False)
    typology.to_csv(OUTPUT_DECOUPLING_TABLE, index=False)
    manuscript_table.to_csv(OUTPUT_MAIN_TABLE, index=False)
    write_latex_table(manuscript_table)

    generated_figures: list[Path] = []
    generated_figures.extend(figure_study_design())
    generated_figures.extend(
        forest_hr(
            target,
            effect_col="score_hr_per_sd",
            low_col="score_ci_low",
            high_col="score_ci_high",
            title="TARGET-OS: frozen program association with overall survival",
            xlabel="Hazard ratio per 1-SD frozen score",
            stem=FIGURE_FILES["target"],
        )
    )
    generated_figures.extend(figure_gse21257_auc(gse21257))
    generated_figures.extend(
        forest_hr(
            gse39055,
            effect_col="hr_per_sd",
            low_col="ci_low",
            high_col="ci_high",
            title="GSE39055: frozen program association with recurrence-free survival",
            xlabel="Hazard ratio per 1-SD frozen score",
            stem=FIGURE_FILES["gse39055"],
        )
    )
    generated_figures.extend(figure_direct_canine(triangulation))
    generated_figures.extend(figure_wgcna(triangulation))
    generated_figures.extend(figure_blind_rediscovery(triangulation))
    generated_figures.extend(figure_decoupling(typology))
    generated_figures.extend(
        figure_evidence_matrix(
            multiplicity=multiplicity,
            typology=typology,
            tri=triangulation,
            interpretation=interpretation,
        )
    )

    write_captions(
        target=target,
        gse21257=gse21257,
        gse39055=gse39055,
        tri=triangulation,
    )
    write_results_outline(
        main=main_table,
        multiplicity=multiplicity,
    )
    write_readme()

    print("=" * 80)
    print("Final main manuscript table")
    print("=" * 80)
    print(manuscript_table.to_string(index=False))

    print("")
    print("=" * 80)
    print("Project-wide human multiplicity check")
    print("=" * 80)
    print(
        multiplicity[
            [
                "cohort",
                "module_label",
                "primary_p",
                "projectwide_q_12",
                "projectwide_fdr_supported",
            ]
        ].sort_values(
            ["projectwide_q_12", "primary_p"]
        ).to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("External canine triangulation for figures")
    print("=" * 80)
    print(
        ordered(triangulation)[
            [
                "module_label",
                "edge_spearman",
                "loading_spearman",
                "zsummary_pres",
                "best_match_f1",
                "empirical_max_match_q_bh_4",
                "external_canine_triangulation_class",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Final interpretation guardrails")
    print("=" * 80)
    print("Only GSE21257 M34 is project-wide FDR supported among the 12 human primary tests.")
    print("Human OS, metastasis, and RFS are displayed separately and are not pooled.")
    print("M34 blind rediscovery is partial/core recovery, not exact full-module reconstruction.")
    print("M40 is strongly reproducible molecularly but has assay/context-sensitive outcome transport.")
    print("No additional outcome-driven tuning should follow this figure-generation stage.")

    output_paths = [
        OUTPUT_MAIN_TABLE,
        OUTPUT_MAIN_TABLE_TEX,
        OUTPUT_HUMAN_TABLE,
        OUTPUT_CANINE_TABLE,
        OUTPUT_DECOUPLING_TABLE,
        OUTPUT_CAPTIONS,
        OUTPUT_RESULTS_OUTLINE,
        OUTPUT_README,
        *generated_figures,
    ]

    input_paths = [
        TARGET_FILE,
        GSE21257_FILE,
        GSE39055_FILE,
        PROJECTWIDE_MULTIPLICITY_FILE,
        TYPOLOGY_FILE,
        ASSAY_AWARE_SUMMARY_FILE,
        TRIANGULATION_FILE,
        FINAL_MASTER_FILE,
        FINAL_INTERPRETATION_FILE,
        FINAL_LOCK_MANIFEST_FILE,
        CONSERVATIVE_STRUCTURE_FILE,
    ]

    create_manifest(input_paths, output_paths)

    print("")
    print("Saved:")
    for path in output_paths + [OUTPUT_MANIFEST]:
        print(path)
    print("Done.")


if __name__ == "__main__":
    main()
