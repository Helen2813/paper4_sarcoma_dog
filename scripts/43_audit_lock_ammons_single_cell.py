from __future__ import annotations

from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_VERSION = "43-audit-lock-ammons-single-cell-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "single_cell" / "Ammons_GSE252470"
PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed" / "single_cell" / "Ammons_GSE252470"
)
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

METADATA_FILE = RAW_DIR / "meta.tsv"
SCORE_COVERAGE_FILE = RESULTS_DIR / "Ammons_scRNA_score_gene_coverage.csv"
TARGETED_FILE = RESULTS_DIR / "Ammons_scRNA_targeted_localization_tests.csv"
CELLTYPE_SUMMARY_FILE = RESULTS_DIR / "Ammons_scRNA_celltype_score_summary.csv"
RANK_STABILITY_FILE = RESULTS_DIR / "Ammons_scRNA_celltype_rank_stability.csv"
LOCALIZATION_MANIFEST_FILE = RESULTS_DIR / "Ammons_scRNA_localization_manifest.json"
MULTIDIMENSIONAL_FILE = (
    RESULTS_DIR / "paper4_locked_multidimensional_transport_evidence.csv"
)

REPLICATE_COLUMN = "orig.ident"
CELL_ID_COLUMN = "barcode"
LOW_CARDINALITY_MAX = 25

OUTPUT_REPLICATE_COUNTS = RESULTS_DIR / "Ammons_scRNA_replicate_cell_counts.csv"
OUTPUT_METADATA_AUDIT = RESULTS_DIR / "Ammons_scRNA_low_cardinality_metadata_audit.csv"
OUTPUT_RELATIONSHIPS = RESULTS_DIR / "Ammons_scRNA_replicate_metadata_relationships.csv"
OUTPUT_MAPPING_TEMPLATE = (
    PROCESSED_DIR / "Ammons_orig_ident_to_dog_mapping_template.csv"
)
OUTPUT_TARGETED_AUDIT = RESULTS_DIR / "Ammons_scRNA_targeted_sign_consistency_audit.csv"
OUTPUT_CLEAN_CELLTYPE = (
    RESULTS_DIR / "Ammons_scRNA_celltype_score_summary_estimable_only.csv"
)
OUTPUT_CLEAN_STABILITY = (
    RESULTS_DIR / "Ammons_scRNA_celltype_rank_stability_estimable_only.csv"
)
OUTPUT_LOCKED_SINGLE_CELL = (
    RESULTS_DIR / "paper4_locked_single_cell_biological_localization.csv"
)
OUTPUT_UPDATED_MASTER = (
    RESULTS_DIR
    / "paper4_locked_multidimensional_transport_evidence_with_single_cell.csv"
)
OUTPUT_SENTENCES = RESULTS_DIR / "paper4_locked_single_cell_results_sentences.txt"
OUTPUT_README = RESULTS_DIR / "Ammons_scRNA_replicate_audit_README.txt"
OUTPUT_MANIFEST = RESULTS_DIR / "Ammons_scRNA_replicate_audit_manifest.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_required_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    print(f"Loaded: {path}")
    return pd.read_csv(path, **kwargs)


def read_required_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def low_cardinality_audit(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in metadata.columns:
        if column == CELL_ID_COLUMN:
            continue
        series = metadata[column].astype(str)
        unique_values = series.dropna().drop_duplicates()
        n_unique = int(unique_values.shape[0])
        if n_unique > LOW_CARDINALITY_MAX:
            continue
        rows.append(
            {
                "column": str(column),
                "n_unique": n_unique,
                "n_nonmissing": int(metadata[column].notna().sum()),
                "example_values": ";".join(
                    unique_values.astype(str).head(25)
                ),
                "possible_biological_unit": any(
                    token in str(column).lower()
                    for token in [
                        "dog",
                        "patient",
                        "donor",
                        "animal",
                        "sample",
                        "orig.ident",
                    ]
                ),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "column",
                "n_unique",
                "n_nonmissing",
                "example_values",
                "possible_biological_unit",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["possible_biological_unit", "n_unique", "column"],
        ascending=[False, True, True],
    )


def relationship_audit(
    metadata: pd.DataFrame,
    metadata_audit: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    n_replicates = int(metadata[REPLICATE_COLUMN].astype(str).nunique())

    for column in metadata_audit["column"].tolist():
        if column == REPLICATE_COLUMN:
            continue

        pair = metadata[[REPLICATE_COLUMN, column]].dropna().astype(str).drop_duplicates()
        if pair.empty:
            continue

        candidate_per_replicate = pair.groupby(REPLICATE_COLUMN)[column].nunique()
        replicate_per_candidate = pair.groupby(column)[REPLICATE_COLUMN].nunique()

        max_candidate_per_replicate = int(candidate_per_replicate.max())
        max_replicate_per_candidate = int(replicate_per_candidate.max())
        n_candidate_values = int(pair[column].nunique())

        if (
            max_candidate_per_replicate == 1
            and max_replicate_per_candidate == 1
        ):
            relationship = "one_to_one"
        elif (
            max_candidate_per_replicate == 1
            and max_replicate_per_candidate > 1
        ):
            relationship = "orig_ident_nested_within_candidate"
        elif (
            max_candidate_per_replicate > 1
            and max_replicate_per_candidate == 1
        ):
            relationship = "candidate_nested_within_orig_ident"
        else:
            relationship = "many_to_many"

        rows.append(
            {
                "candidate_column": column,
                "n_orig_ident": n_replicates,
                "n_candidate_values": n_candidate_values,
                "max_candidate_values_per_orig_ident": max_candidate_per_replicate,
                "max_orig_ident_per_candidate_value": max_replicate_per_candidate,
                "relationship": relationship,
                "possible_six_dog_mapping": bool(
                    n_candidate_values == 6
                    and relationship == "orig_ident_nested_within_candidate"
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "candidate_column",
                "n_orig_ident",
                "n_candidate_values",
                "max_candidate_values_per_orig_ident",
                "max_orig_ident_per_candidate_value",
                "relationship",
                "possible_six_dog_mapping",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        ["possible_six_dog_mapping", "relationship", "n_candidate_values"],
        ascending=[False, True, True],
    )


def create_mapping_template(
    metadata: pd.DataFrame,
    relationships: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str | None]:
    candidates = relationships[
        relationships["possible_six_dog_mapping"].astype(bool)
    ] if not relationships.empty else pd.DataFrame()

    mapping = (
        metadata[[REPLICATE_COLUMN]]
        .drop_duplicates()
        .sort_values(REPLICATE_COLUMN)
        .reset_index(drop=True)
    )

    if not candidates.empty:
        candidate_column = str(candidates.iloc[0]["candidate_column"])
        candidate_map = (
            metadata[[REPLICATE_COLUMN, candidate_column]]
            .dropna()
            .astype(str)
            .drop_duplicates()
        )
        mapping = mapping.merge(
            candidate_map,
            on=REPLICATE_COLUMN,
            how="left",
        )
        mapping["proposed_dog_id"] = mapping[candidate_column]
        return (
            mapping,
            "candidate_six_dog_mapping_found",
            candidate_column,
        )

    mapping["proposed_dog_id"] = ""
    mapping["review_note"] = (
        "Fill only if orig.ident represents libraries/samples "
        "rather than independent dogs."
    )
    return mapping, "orig_ident_unverified_as_dog", None


def parse_differences(value: Any) -> np.ndarray:
    text = str(value).strip()
    if text in {"", "nan", "None"}:
        return np.asarray([], dtype=float)
    return np.asarray(
        [float(item) for item in text.split(";") if str(item).strip()],
        dtype=float,
    )


def targeted_sign_audit(targeted: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in targeted.itertuples(index=False):
        differences = parse_differences(
            getattr(row, "dog_level_differences", "")
        )
        n = int(differences.size)
        n_positive = int(np.sum(differences > 0))
        n_negative = int(np.sum(differences < 0))
        n_zero = int(np.sum(differences == 0))

        all_same_nonzero_sign = bool(
            n > 0
            and n_zero == 0
            and (n_positive == n or n_negative == n)
        )
        minimum_two_sided_sign_flip_p = (
            2.0 / (2.0**n) if n > 0 else np.nan
        )

        reported_p = float(row.exact_sign_flip_p)
        rows.append(
            {
                "contrast_name": row.contrast_name,
                "n_replicate_units": n,
                "n_positive_differences": n_positive,
                "n_negative_differences": n_negative,
                "n_zero_differences": n_zero,
                "all_same_nonzero_sign": all_same_nonzero_sign,
                "reported_exact_sign_flip_p": reported_p,
                "minimum_attainable_two_sided_p": minimum_two_sided_sign_flip_p,
                "reported_p_equals_minimum": bool(
                    np.isfinite(reported_p)
                    and np.isfinite(minimum_two_sided_sign_flip_p)
                    and np.isclose(
                        reported_p,
                        minimum_two_sided_sign_flip_p,
                    )
                ),
                "mean_paired_difference": row.mean_paired_difference,
                "median_paired_difference": row.median_paired_difference,
                "bootstrap_ci_low": row.bootstrap_ci_low,
                "bootstrap_ci_high": row.bootstrap_ci_high,
                "primary_bh_q": row.primary_bh_q,
            }
        )
    return pd.DataFrame(rows)


def estimable_score_columns(coverage: pd.DataFrame) -> set[str]:
    columns: set[str] = set()
    for row in coverage.itertuples(index=False):
        module = str(row.module_label)
        if bool(row.signed_score_estimable):
            columns.add(f"{module}_signed_risk_score")
        if bool(row.positive_component_estimable):
            columns.add(f"{module}_positive_component_expression")
        if bool(row.negative_component_estimable):
            columns.add(f"{module}_negative_component_expression")
    return columns


def clean_nonestimable_outputs(
    celltype_summary: pd.DataFrame,
    rank_stability: pd.DataFrame,
    coverage: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    allowed = estimable_score_columns(coverage)
    return (
        celltype_summary[
            celltype_summary["score_column"].isin(allowed)
        ].copy(),
        rank_stability[
            rank_stability["score_column"].isin(allowed)
        ].copy(),
    )


def get_targeted_row(
    targeted: pd.DataFrame,
    prefix: str,
) -> pd.Series:
    part = targeted[
        targeted["contrast_name"].astype(str).str.startswith(prefix)
    ]
    if part.empty:
        raise ValueError(f"Targeted contrast not found: {prefix}")
    return part.iloc[0]


def build_locked_single_cell(
    targeted: pd.DataFrame,
    replicate_status: str,
    candidate_column: str | None,
) -> pd.DataFrame:
    m34_signed = get_targeted_row(targeted, "M34 signed risk:")
    m34_negative = get_targeted_row(
        targeted,
        "M34 negative-loading expression:",
    )
    m40_signed = get_targeted_row(targeted, "M40 signed risk:")
    m40_positive = get_targeted_row(
        targeted,
        "M40 positive-loading expression:",
    )

    common_guardrail = (
        "Primary inference is paired at orig.ident level. "
        f"Replicate audit status: {replicate_status}."
    )
    if candidate_column:
        common_guardrail += (
            f" Candidate biological-unit column: {candidate_column}."
        )

    return pd.DataFrame(
        [
            {
                "module_label": "M34",
                "single_cell_localization_class": (
                    "immune_negative_component_with_"
                    "osteoblast_high_signed_risk"
                ),
                "primary_contrast_1": m34_signed["contrast_name"],
                "primary_effect_1": m34_signed["mean_paired_difference"],
                "primary_q_1": m34_signed["primary_bh_q"],
                "primary_contrast_2": m34_negative["contrast_name"],
                "primary_effect_2": m34_negative["mean_paired_difference"],
                "primary_q_2": m34_negative["primary_bh_q"],
                "locked_single_cell_interpretation": (
                    "M34 is primarily an inverse immune-lineage "
                    "expression program: its negative-loading component "
                    "is enriched in IFN-TAM, TAM, DC, and TIM populations, "
                    "whereas the signed risk-oriented score is higher in "
                    "osteoblast-lineage compartments. Because only two "
                    "positive-loading genes are detected, this supports "
                    "an immune-depletion/exclusion interpretation rather "
                    "than a balanced tumor-versus-immune program."
                ),
                "replicate_guardrail": common_guardrail,
            },
            {
                "module_label": "M40",
                "single_cell_localization_class": (
                    "pan_cycling_program_with_"
                    "cycling_osteoblast_enrichment"
                ),
                "primary_contrast_1": m40_signed["contrast_name"],
                "primary_effect_1": m40_signed["mean_paired_difference"],
                "primary_q_1": m40_signed["primary_bh_q"],
                "primary_contrast_2": m40_positive["contrast_name"],
                "primary_effect_2": m40_positive["mean_paired_difference"],
                "primary_q_2": m40_positive["primary_bh_q"],
                "locked_single_cell_interpretation": (
                    "M40 localizes to cycling states across multiple "
                    "lineages, with the highest scores in cycling "
                    "osteoclast, cycling T-cell, and cycling osteoblast "
                    "populations. The paired cycling-versus-non-cycling "
                    "osteoblast contrast confirms tumor-lineage "
                    "proliferation localization, but the program should "
                    "be described as a broad cycling/proliferation axis "
                    "rather than osteoblast-specific."
                ),
                "replicate_guardrail": common_guardrail,
            },
            {
                "module_label": "M11",
                "single_cell_localization_class": (
                    "secondary_positive_component_only"
                ),
                "primary_contrast_1": "",
                "primary_effect_1": np.nan,
                "primary_q_1": np.nan,
                "primary_contrast_2": "",
                "primary_effect_2": np.nan,
                "primary_q_2": np.nan,
                "locked_single_cell_interpretation": (
                    "M11 has six detected positive-loading genes and no "
                    "detected negative-loading genes. Its cell-type "
                    "localization is secondary and does not alter the "
                    "locked cross-species evidence grade."
                ),
                "replicate_guardrail": common_guardrail,
            },
            {
                "module_label": "M24",
                "single_cell_localization_class": (
                    "secondary_positive_component_only"
                ),
                "primary_contrast_1": "",
                "primary_effect_1": np.nan,
                "primary_q_1": np.nan,
                "primary_contrast_2": "",
                "primary_effect_2": np.nan,
                "primary_q_2": np.nan,
                "locked_single_cell_interpretation": (
                    "M24 has six detected positive-loading genes and no "
                    "detected negative-loading genes. Its cell-type "
                    "localization is secondary and cannot compensate for "
                    "limited cross-species representation and outcome "
                    "evidence."
                ),
                "replicate_guardrail": common_guardrail,
            },
        ]
    )


def write_sentences(
    locked_single_cell: pd.DataFrame,
    replicate_status: str,
) -> None:
    indexed = locked_single_cell.set_index("module_label")
    lines = [
        "Locked single-cell localization results",
        "======================================",
        "",
        "M34",
        "---",
        indexed.loc["M34", "locked_single_cell_interpretation"],
        "",
        "M40",
        "---",
        indexed.loc["M40", "locked_single_cell_interpretation"],
        "",
        "Statistical interpretation",
        "--------------------------",
        (
            "All four targeted paired contrasts reached the exact "
            "two-sided sign-flip floor for eight orig.ident units "
            "(P=0.0078125), indicating the same contrast direction "
            "in every analyzed unit. These P values quantify "
            "directional consistency, not large-sample precision."
        ),
        "",
        "Replicate-unit guardrail",
        "------------------------",
        (
            f"Replicate audit status: {replicate_status}. "
            "Until orig.ident is verified as one independent dog per "
            "level, manuscript wording should use 'sample/dog-level "
            "pseudobulk' rather than unequivocally 'dog-level'."
        ),
    ]
    OUTPUT_SENTENCES.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_readme(
    replicate_status: str,
    candidate_column: str | None,
    n_orig_ident: int,
) -> None:
    text = f"""Ammons single-cell replicate-unit audit
Script version: {SCRIPT_VERSION}

Purpose
-------
1. Audit whether orig.ident can safely be treated as an independent dog.
2. Identify any metadata column that collapses the {n_orig_ident} orig.ident
   levels to six biological dogs.
3. Verify the sign consistency underlying the exact paired tests.
4. Remove non-estimable M11/M24 negative-component rankings.
5. Lock manuscript-ready M34 and M40 biological localization statements.

Current replicate status
------------------------
{replicate_status}

Candidate six-dog mapping column
--------------------------------
{candidate_column}

Important statistical point
---------------------------
For eight paired units, the minimum attainable two-sided exact sign-flip
P value is 2 / 2^8 = 0.0078125. All four targeted contrasts reached this
floor because every paired difference had the same sign. This is strong
directional consistency, but the four tests are biologically related and
should not be presented as four independent replications.

Biological interpretation
-------------------------
- M34: negative-loading genes localize to IFN-TAM/TAM/DC/TIM populations,
  while the signed risk score is highest in osteoblast-lineage populations.
  Because 153 of 155 detected genes have negative loadings, the score is
  principally an inverse immune-lineage axis.
- M40: highest in cycling osteoclast, cycling T-cell, and cycling osteoblast
  populations. The program is a broad cycling/proliferation axis with clear
  cycling-osteoblast enrichment, not an osteoblast-specific program.
- M11/M24 negative-component rankings from script 42 were non-estimable
  artifacts of all-NaN components and are excluded from cleaned outputs.

No clinical outcome is loaded or reanalyzed.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("Audit and lock Ammons single-cell localization evidence")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")

    metadata = read_required_csv(
        METADATA_FILE,
        sep="\t",
        dtype=str,
        low_memory=False,
    )
    coverage = read_required_csv(SCORE_COVERAGE_FILE)
    targeted = read_required_csv(TARGETED_FILE)
    celltype_summary = read_required_csv(CELLTYPE_SUMMARY_FILE)
    rank_stability = read_required_csv(RANK_STABILITY_FILE)
    localization_manifest = read_required_json(
        LOCALIZATION_MANIFEST_FILE
    )

    if bool(localization_manifest.get("outcome_loaded", True)):
        raise RuntimeError(
            "Localization manifest does not confirm outcome_loaded=false."
        )
    if REPLICATE_COLUMN not in metadata.columns:
        raise ValueError(f"Replicate column not found: {REPLICATE_COLUMN}")

    replicate_counts = (
        metadata[REPLICATE_COLUMN]
        .astype(str)
        .value_counts()
        .rename_axis(REPLICATE_COLUMN)
        .reset_index(name="n_cells")
        .sort_values(REPLICATE_COLUMN)
    )
    replicate_counts["fraction_cells"] = (
        replicate_counts["n_cells"] / metadata.shape[0]
    )

    metadata_audit = low_cardinality_audit(metadata)
    relationships = relationship_audit(metadata, metadata_audit)
    mapping_template, replicate_status, candidate_column = (
        create_mapping_template(metadata, relationships)
    )

    targeted_audit = targeted_sign_audit(targeted)
    clean_summary, clean_stability = clean_nonestimable_outputs(
        celltype_summary=celltype_summary,
        rank_stability=rank_stability,
        coverage=coverage,
    )
    locked_single_cell = build_locked_single_cell(
        targeted=targeted,
        replicate_status=replicate_status,
        candidate_column=candidate_column,
    )

    replicate_counts.to_csv(OUTPUT_REPLICATE_COUNTS, index=False)
    metadata_audit.to_csv(OUTPUT_METADATA_AUDIT, index=False)
    relationships.to_csv(OUTPUT_RELATIONSHIPS, index=False)
    mapping_template.to_csv(OUTPUT_MAPPING_TEMPLATE, index=False)
    targeted_audit.to_csv(OUTPUT_TARGETED_AUDIT, index=False)
    clean_summary.to_csv(OUTPUT_CLEAN_CELLTYPE, index=False)
    clean_stability.to_csv(OUTPUT_CLEAN_STABILITY, index=False)
    locked_single_cell.to_csv(OUTPUT_LOCKED_SINGLE_CELL, index=False)

    if MULTIDIMENSIONAL_FILE.exists():
        multidimensional = pd.read_csv(MULTIDIMENSIONAL_FILE)
        updated_master = multidimensional.merge(
            locked_single_cell[
                [
                    "module_label",
                    "single_cell_localization_class",
                    "locked_single_cell_interpretation",
                    "replicate_guardrail",
                ]
            ],
            on="module_label",
            how="left",
        )
        updated_master.to_csv(OUTPUT_UPDATED_MASTER, index=False)
    else:
        updated_master = pd.DataFrame()

    write_sentences(
        locked_single_cell=locked_single_cell,
        replicate_status=replicate_status,
    )
    write_readme(
        replicate_status=replicate_status,
        candidate_column=candidate_column,
        n_orig_ident=replicate_counts.shape[0],
    )

    input_paths = [
        METADATA_FILE,
        SCORE_COVERAGE_FILE,
        TARGETED_FILE,
        CELLTYPE_SUMMARY_FILE,
        RANK_STABILITY_FILE,
        LOCALIZATION_MANIFEST_FILE,
    ]
    if MULTIDIMENSIONAL_FILE.exists():
        input_paths.append(MULTIDIMENSIONAL_FILE)

    output_paths = [
        OUTPUT_REPLICATE_COUNTS,
        OUTPUT_METADATA_AUDIT,
        OUTPUT_RELATIONSHIPS,
        OUTPUT_MAPPING_TEMPLATE,
        OUTPUT_TARGETED_AUDIT,
        OUTPUT_CLEAN_CELLTYPE,
        OUTPUT_CLEAN_STABILITY,
        OUTPUT_LOCKED_SINGLE_CELL,
        OUTPUT_SENTENCES,
        OUTPUT_README,
    ]
    if not updated_master.empty:
        output_paths.append(OUTPUT_UPDATED_MASTER)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "outcome_loaded": False,
        "replicate_column": REPLICATE_COLUMN,
        "n_orig_ident": int(replicate_counts.shape[0]),
        "replicate_status": replicate_status,
        "candidate_six_dog_mapping_column": candidate_column,
        "guardrails": [
            "No clinical endpoint or outcome was loaded.",
            "orig.ident is not called a dog until metadata audit verifies it.",
            "All four exact sign-flip P values are at the n=8 two-sided floor.",
            "M11 and M24 non-estimable negative-component ranks are removed.",
            "Single-cell localization is biological annotation, not outcome validation.",
        ],
        "inputs": {
            str(path): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in input_paths
        },
        "outputs": {
            str(path): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in output_paths
            if path.exists()
        },
    }
    OUTPUT_MANIFEST.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("Replicate-unit audit")
    print("=" * 80)
    print(f"orig.ident levels: {replicate_counts.shape[0]}")
    print(f"Status: {replicate_status}")
    print(f"Candidate six-dog column: {candidate_column}")
    print("")
    print(replicate_counts.to_string(index=False))

    print("")
    print("=" * 80)
    print("Low-cardinality metadata relationships")
    print("=" * 80)
    print(relationships.to_string(index=False))

    print("")
    print("=" * 80)
    print("Targeted sign-consistency audit")
    print("=" * 80)
    print(targeted_audit.to_string(index=False))

    print("")
    print("=" * 80)
    print("Locked single-cell localization")
    print("=" * 80)
    print(
        locked_single_cell[
            [
                "module_label",
                "single_cell_localization_class",
                "primary_effect_1",
                "primary_q_1",
                "primary_effect_2",
                "primary_q_2",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print(
        "Use sample/dog-level wording until orig.ident is "
        "verified as an independent dog identifier."
    )
    print(
        "All four exact P values are the n=8 two-sided "
        "minimum, indicating identical sign across units."
    )
    print(
        "M34 is principally an inverse immune-lineage axis, "
        "not a balanced positive-versus-negative program."
    )
    print(
        "M40 is a pan-cycling program with clear cycling-"
        "osteoblast enrichment, not osteoblast-specific."
    )
    print(
        "M11/M24 negative-component rankings from script 42 "
        "were non-estimable and have been removed."
    )

    print("")
    print("Saved:")
    for path in [
        OUTPUT_REPLICATE_COUNTS,
        OUTPUT_METADATA_AUDIT,
        OUTPUT_RELATIONSHIPS,
        OUTPUT_MAPPING_TEMPLATE,
        OUTPUT_TARGETED_AUDIT,
        OUTPUT_CLEAN_CELLTYPE,
        OUTPUT_CLEAN_STABILITY,
        OUTPUT_LOCKED_SINGLE_CELL,
        OUTPUT_UPDATED_MASTER,
        OUTPUT_SENTENCES,
        OUTPUT_README,
        OUTPUT_MANIFEST,
    ]:
        if path.exists():
            print(path)
    print("Done.")


if __name__ == "__main__":
    main()
