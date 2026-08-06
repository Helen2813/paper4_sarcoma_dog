from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_VERSION = "30-paper4-locked-figures-tables-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "paper4_locked"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTCOME_FILE = RESULTS_DIR / "paper4_locked_human_outcome_evidence.csv"
STRUCTURE_FILE = RESULTS_DIR / "paper4_locked_structure_evidence.csv"
SUMMARY_FILE = RESULTS_DIR / "paper4_locked_module_evidence_summary.csv"
INTERPRETATION_FILE = RESULTS_DIR / "paper4_locked_module_interpretation.csv"

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]
COHORT_ORDER = ["TARGET_OS", "GSE21257", "GSE39055"]

OUTPUT_MAIN_TABLE = RESULTS_DIR / "Paper4_locked_main_results_table.csv"
OUTPUT_MAIN_TABLE_TEX = RESULTS_DIR / "Paper4_locked_main_results_table.tex"
OUTPUT_STRUCTURE_TABLE = RESULTS_DIR / "Paper4_locked_structure_table.csv"
OUTPUT_CAPTIONS = RESULTS_DIR / "Paper4_locked_figure_captions.txt"
OUTPUT_MANIFEST = RESULTS_DIR / "Paper4_locked_figure_table_manifest.json"
OUTPUT_README = RESULTS_DIR / "Paper4_locked_figure_table_README.txt"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    print(f"Loaded: {path}")
    return pd.read_csv(path)


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    png = FIGURES_DIR / f"{stem}.png"
    pdf = FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def module_order_index(values: pd.Series) -> pd.Series:
    order = {module: index for index, module in enumerate(PRIMARY_MODULES)}
    return values.map(order)


def forest_plot(
    data: pd.DataFrame,
    cohort: str,
    title: str,
    output_stem: str,
) -> list[Path]:
    part = data[data["cohort"].eq(cohort)].copy()
    part["module_order"] = module_order_index(part["module_label"])
    part = part.sort_values("module_order", ascending=False)

    effects = pd.to_numeric(part["effect"], errors="coerce").values
    low = pd.to_numeric(part["ci_low"], errors="coerce").values
    high = pd.to_numeric(part["ci_high"], errors="coerce").values
    y = np.arange(part.shape[0])

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.errorbar(
        effects,
        y,
        xerr=np.vstack([effects - low, high - effects]),
        fmt="o",
        capsize=4,
    )
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(part["module_label"].tolist())
    ax.set_xlabel(part["effect_type"].iloc[0].replace("_", " "))
    ax.set_title(title)

    finite_values = np.concatenate([low[np.isfinite(low)], high[np.isfinite(high)]])
    if finite_values.size:
        minimum = max(0.05, float(np.min(finite_values)) * 0.80)
        maximum = float(np.max(finite_values)) * 1.20
        ax.set_xlim(minimum, maximum)
        if minimum > 0:
            ax.set_xscale("log")

    for index, row in enumerate(part.itertuples(index=False)):
        text = f"{row.outcome_support_class}"
        ax.text(
            1.01,
            index,
            text,
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    return save_figure(fig, output_stem)


def discrimination_plot(data: pd.DataFrame) -> list[Path]:
    part = data.copy()
    part["module_order"] = module_order_index(part["module_label"])
    part["cohort_order"] = part["cohort"].map(
        {cohort: index for index, cohort in enumerate(COHORT_ORDER)}
    )
    part = part.sort_values(["module_order", "cohort_order"])

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x_base = np.arange(len(PRIMARY_MODULES), dtype=float)
    offsets = {
        "TARGET_OS": -0.22,
        "GSE21257": 0.0,
        "GSE39055": 0.22,
    }

    for cohort in COHORT_ORDER:
        cohort_part = (
            part[part["cohort"].eq(cohort)]
            .set_index("module_label")
            .reindex(PRIMARY_MODULES)
        )
        ax.plot(
            x_base + offsets[cohort],
            cohort_part["discrimination"].values,
            marker="o",
            linestyle="none",
            label=cohort,
        )

    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_xticks(x_base)
    ax.set_xticklabels(PRIMARY_MODULES)
    ax.set_ylabel("Fixed-direction C-index or AUC")
    ax.set_xlabel("Frozen canine program")
    ax.set_title("Human discrimination with frozen canine score direction")
    ax.legend(frameon=False)
    ax.set_ylim(0.30, 0.90)
    fig.tight_layout()
    return save_figure(fig, "Paper4_fixed_direction_discrimination")


def heatmap(
    structure: pd.DataFrame,
    value_col: str,
    title: str,
    output_stem: str,
    value_format: str = ".2f",
) -> list[Path]:
    matrix = (
        structure.pivot(
            index="module_label",
            columns="cohort",
            values=value_col,
        )
        .reindex(index=PRIMARY_MODULES, columns=COHORT_ORDER)
    )

    fig, ax = plt.subplots(figsize=(6.7, 4.6))
    image = ax.imshow(matrix.values, aspect="auto")
    ax.set_xticks(np.arange(len(COHORT_ORDER)))
    ax.set_xticklabels(COHORT_ORDER)
    ax.set_yticks(np.arange(len(PRIMARY_MODULES)))
    ax.set_yticklabels(PRIMARY_MODULES)
    ax.set_title(title)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix.iloc[row, col]
            label = "NA" if not np.isfinite(value) else format(value, value_format)
            ax.text(col, row, label, ha="center", va="center", fontsize=9)

    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    return save_figure(fig, output_stem)


def evidence_matrix_figure(
    outcome: pd.DataFrame,
    structure: pd.DataFrame,
) -> list[Path]:
    outcome_map = {
        (row.module_label, row.cohort): row.outcome_support_class
        for row in outcome.itertuples(index=False)
    }
    structure_map = {
        (row.module_label, row.cohort): row.conservative_preservation_class
        for row in structure.itertuples(index=False)
    }

    structure_short = {
        "strong_cross_cohort_representation_preservation": "Strong structure",
        "partial_cross_cohort_representation_preservation": "Partial structure",
        "limited_cross_cohort_representation_evidence": "Limited structure",
        "no_clear_cross_cohort_representation_preservation": "No clear structure",
    }
    outcome_short = {
        "global_fdr_directional_support": "Global-FDR support",
        "endpoint_fdr_directional_support": "Endpoint-FDR support",
        "nominal_directional_support": "Nominal support",
        "directional_without_nominal_support": "Direction only",
        "direction_discordant": "Opposite direction",
        "not_estimable": "Not estimable",
    }

    cell_text = []
    for module in PRIMARY_MODULES:
        row = []
        for cohort in COHORT_ORDER:
            structure_label = structure_short.get(
                structure_map.get((module, cohort), ""),
                "Unknown structure",
            )
            outcome_label = outcome_short.get(
                outcome_map.get((module, cohort), ""),
                "Unknown outcome",
            )
            row.append(f"{structure_label}\n{outcome_label}")
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=PRIMARY_MODULES,
        colLabels=COHORT_ORDER,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 2.3)
    ax.set_title(
        "Locked cross-species evidence matrix\n"
        "Representation preservation and outcome transfer are shown separately",
        pad=18,
    )
    fig.tight_layout()
    return save_figure(fig, "Paper4_locked_evidence_matrix")


def build_main_table(
    outcome: pd.DataFrame,
    structure: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    summary_index = summary.set_index("module_label")

    for module in PRIMARY_MODULES:
        row: dict[str, Any] = {
            "module_label": module,
            "locked_evidence_grade": summary_index.loc[
                module, "locked_evidence_grade"
            ],
            "locked_interpretation": summary_index.loc[
                module, "locked_interpretation"
            ],
        }

        for cohort in COHORT_ORDER:
            outcome_row = outcome[
                outcome["module_label"].eq(module)
                & outcome["cohort"].eq(cohort)
            ].iloc[0]
            structure_row = structure[
                structure["module_label"].eq(module)
                & structure["cohort"].eq(cohort)
            ].iloc[0]

            prefix = cohort.lower()
            row[f"{prefix}_endpoint"] = outcome_row["endpoint"]
            row[f"{prefix}_effect"] = outcome_row["effect"]
            row[f"{prefix}_ci_low"] = outcome_row["ci_low"]
            row[f"{prefix}_ci_high"] = outcome_row["ci_high"]
            row[f"{prefix}_primary_p"] = outcome_row["primary_p"]
            row[f"{prefix}_endpoint_q"] = outcome_row["endpoint_q"]
            row[f"{prefix}_discrimination"] = outcome_row["discrimination"]
            row[f"{prefix}_outcome_class"] = outcome_row[
                "outcome_support_class"
            ]
            row[f"{prefix}_structure_class"] = structure_row[
                "conservative_preservation_class"
            ]
            row[f"{prefix}_edge_spearman"] = structure_row["edge_spearman"]
            row[f"{prefix}_loading_spearman"] = structure_row[
                "loading_spearman"
            ]
            row[f"{prefix}_split_half_median"] = structure_row[
                "split_half_median"
            ]

        rows.append(row)

    return pd.DataFrame(rows)


def latex_escape(text: Any) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    result = str(text)
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


def write_latex_table(main_table: pd.DataFrame) -> None:
    grade_short = {
        "multi_setting_transfer_with_third_setting_heterogeneity":
            "Transfer in two settings; third-setting heterogeneity",
        "directionally_consistent_but_weakly_supported":
            "Directionally consistent; weak support",
        "limited_or_inconsistent_cross_species_transfer":
            "Limited or inconsistent transfer",
        "structure_preserved_but_outcome_heterogeneous":
            "Structure preserved; outcome heterogeneous",
    }

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Locked cross-species evidence hierarchy for the four frozen primary canine programs.}",
        r"\label{tab:locked_cross_species_evidence}",
        r"\small",
        r"\begin{tabular}{lllll}",
        r"\toprule",
        r"Program & TARGET-OS & GSE21257 & GSE39055 & Locked interpretation \\",
        r"\midrule",
    ]

    for _, row in main_table.iterrows():
        target = (
            f"{row['target_os_outcome_class']}; "
            f"{row['target_os_structure_class']}"
        )
        gse21257 = (
            f"{row['gse21257_outcome_class']}; "
            f"{row['gse21257_structure_class']}"
        )
        gse39055 = (
            f"{row['gse39055_outcome_class']}; "
            f"{row['gse39055_structure_class']}"
        )
        grade = grade_short.get(
            row["locked_evidence_grade"],
            row["locked_evidence_grade"],
        )
        lines.append(
            f"{latex_escape(row['module_label'])} & "
            f"{latex_escape(target)} & "
            f"{latex_escape(gse21257)} & "
            f"{latex_escape(gse39055)} & "
            f"{latex_escape(grade)} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    OUTPUT_MAIN_TABLE_TEX.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_captions() -> None:
    captions = """Suggested figure captions

Figure: TARGET-OS frozen-program effects
Hazard ratios per one-standard-deviation increase in each frozen strict canine-to-human program score for overall survival in TARGET-OS. The canine risk orientation was preserved without human outcome-based score reversal. Horizontal intervals show 95% confidence intervals. The vertical line marks a hazard ratio of one.

Figure: GSE21257 frozen-program effects
Odds ratios per one-standard-deviation increase in each frozen strict program score for metastasis within five years in GSE21257. The canine risk orientation was preserved. Horizontal intervals show bootstrap 95% confidence intervals. The vertical line marks an odds ratio of one.

Figure: GSE39055 frozen-program effects
Hazard ratios per one-standard-deviation increase in each frozen strict program score for recurrence-free survival in GSE39055. The primary analysis excluded one nonpositive follow-up time; the prespecified one-day replacement sensitivity produced materially unchanged estimates. The vertical line marks a hazard ratio of one.

Figure: Fixed-direction human discrimination
Discrimination of the frozen canine-oriented program scores in three human osteosarcoma settings. TARGET-OS and GSE39055 are summarized by fixed-direction Harrell C-index, whereas GSE21257 is summarized by AUC for five-year metastasis. The horizontal line marks chance-level discrimination. These endpoint-specific metrics are displayed for triangulation and are not pooled.

Figure: Conservative representation preservation
Canine-to-human preservation of within-module correlation structure, PC1-loading structure, and non-overlapping split-half coherence. Values are shown separately because transcriptional representation preservation and outcome association are distinct scientific quantities.

Figure: Locked evidence matrix
Final locked evidence hierarchy for the four primary frozen canine programs. Each cell reports the conservative representation-preservation class from script 28 and the endpoint-specific outcome-support class. Overall survival, five-year metastasis, and recurrence-free survival were not combined in a formal meta-analysis.
"""
    OUTPUT_CAPTIONS.write_text(captions, encoding="utf-8")


def write_readme() -> None:
    text = f"""Paper 4 locked figures and tables
Script version: {SCRIPT_VERSION}

This script performs no new feature selection, model fitting, score orientation,
or hypothesis testing. It reads only the locked evidence files from script 29
and the conservative representation-preservation results from script 28.

Generated material
------------------
- Three endpoint-specific forest plots
- One fixed-direction discrimination plot
- Three structure-preservation heatmaps
- One locked evidence matrix
- Main results CSV and LaTeX table
- Suggested figure captions

Interpretation restriction
--------------------------
TARGET-OS overall survival, GSE21257 five-year metastasis, and GSE39055
recurrence-free survival are displayed separately and are not pooled.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("Generate Paper 4 locked figures and tables")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Figures directory: {FIGURES_DIR}")
    print("")

    outcome = read_required_csv(OUTCOME_FILE)
    structure = read_required_csv(STRUCTURE_FILE)
    summary = read_required_csv(SUMMARY_FILE)
    interpretation = read_required_csv(INTERPRETATION_FILE)

    generated_files: list[Path] = []

    generated_files.extend(
        forest_plot(
            outcome,
            cohort="TARGET_OS",
            title="TARGET-OS: frozen program association with overall survival",
            output_stem="Paper4_TARGET_OS_forest",
        )
    )
    generated_files.extend(
        forest_plot(
            outcome,
            cohort="GSE21257",
            title="GSE21257: frozen program association with metastasis within five years",
            output_stem="Paper4_GSE21257_forest",
        )
    )
    generated_files.extend(
        forest_plot(
            outcome,
            cohort="GSE39055",
            title="GSE39055: frozen program association with recurrence-free survival",
            output_stem="Paper4_GSE39055_forest",
        )
    )
    generated_files.extend(discrimination_plot(outcome))

    generated_files.extend(
        heatmap(
            structure,
            value_col="edge_spearman",
            title="Canine-human preservation of within-module correlation edges",
            output_stem="Paper4_structure_edge_spearman",
        )
    )
    generated_files.extend(
        heatmap(
            structure,
            value_col="loading_spearman",
            title="Canine-human preservation of PC1 loadings",
            output_stem="Paper4_structure_loading_spearman",
        )
    )
    generated_files.extend(
        heatmap(
            structure,
            value_col="split_half_median",
            title="Non-overlapping split-half coherence in human cohorts",
            output_stem="Paper4_structure_split_half",
        )
    )
    generated_files.extend(evidence_matrix_figure(outcome, structure))

    main_table = build_main_table(outcome, structure, summary)
    main_table.to_csv(OUTPUT_MAIN_TABLE, index=False)

    structure[
        [
            "cohort",
            "module_label",
            "n_genes_shared",
            "edge_spearman",
            "edge_permutation_q_positive",
            "loading_spearman",
            "loading_permutation_q_positive",
            "split_half_median",
            "split_half_q05",
            "conservative_preservation_class",
        ]
    ].to_csv(OUTPUT_STRUCTURE_TABLE, index=False)

    write_latex_table(main_table)
    write_captions()
    write_readme()

    generated_files.extend(
        [
            OUTPUT_MAIN_TABLE,
            OUTPUT_MAIN_TABLE_TEX,
            OUTPUT_STRUCTURE_TABLE,
            OUTPUT_CAPTIONS,
            OUTPUT_README,
        ]
    )

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "source_files": [
            str(OUTCOME_FILE),
            str(STRUCTURE_FILE),
            str(SUMMARY_FILE),
            str(INTERPRETATION_FILE),
        ],
        "generated_files": [str(path) for path in generated_files],
        "interpretation_guardrail": (
            "No formal pooling across overall survival, five-year metastasis, "
            "and recurrence-free survival."
        ),
    }
    OUTPUT_MANIFEST.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("")
    print("=" * 80)
    print("Generated Paper 4 assets")
    print("=" * 80)
    for path in generated_files + [OUTPUT_MANIFEST]:
        print(path)

    print("")
    print("=" * 80)
    print("Locked main results table")
    print("=" * 80)
    print(
        main_table[
            [
                "module_label",
                "target_os_outcome_class",
                "gse21257_outcome_class",
                "gse39055_outcome_class",
                "locked_evidence_grade",
            ]
        ].to_string(index=False)
    )

    print("")
    print("Done.")


if __name__ == "__main__":
    main()
