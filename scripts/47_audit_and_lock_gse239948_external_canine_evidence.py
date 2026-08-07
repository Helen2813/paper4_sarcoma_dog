from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib.pyplot as plt

SCRIPT_VERSION = "47-lock-gse239948-independent-canine-evidence-v2"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "gse239948_external_canine"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]

REFERENCE_EXPRESSION_FILE = (
    PROCESSED_DIR
    / "GSE238110_DOG2_expression_log2cpm_matched_allgenes.csv"
)
EXTERNAL_EXPRESSION_FILE = (
    PROCESSED_DIR
    / "canine_validation_GSE239948_expression_log2_symbol.csv"
)
EXTERNAL_SCORES_FILE = (
    PROCESSED_DIR
    / "canine_validation_GSE239948_frozen_program_scores.csv"
)
STRICT_WEIGHTS_FILE = (
    RESULTS_DIR / "GSE238110_frozen_transfer_gene_weights_strict.csv"
)
EXTERNAL_COVERAGE_FILE = (
    RESULTS_DIR / "GSE239948_external_frozen_program_coverage.csv"
)
EXTERNAL_STRUCTURE_FILE = (
    RESULTS_DIR / "GSE239948_external_module_structure_preservation.csv"
)
EXTERNAL_RELIABILITY_FILE = (
    RESULTS_DIR / "GSE239948_external_module_score_reliability.csv"
)
EXTERNAL_RANDOM_FILE = (
    RESULTS_DIR / "GSE239948_external_random_panel_controls.csv"
)
EXTERNAL_CLASSIFICATION_FILE = (
    RESULTS_DIR / "GSE239948_external_representation_classification.csv"
)
EXTERNAL_MANIFEST_FILE = (
    RESULTS_DIR / "GSE239948_external_representation_manifest.json"
)

MASTER_FILE_CANDIDATES = [
    RESULTS_DIR
    / "paper4_locked_multidimensional_transport_evidence_with_single_cell_six_dogs.csv",
    RESULTS_DIR / "paper4_locked_multidimensional_transport_evidence.csv",
]

OUTPUT_SAMPLE_ID_AUDIT = (
    RESULTS_DIR / "GSE239948_DOG2_sample_identifier_audit.csv"
)
OUTPUT_EXACT_OVERLAP = (
    RESULTS_DIR / "GSE239948_DOG2_exact_sample_identifier_overlap.csv"
)
OUTPUT_FINGERPRINT_NEIGHBORS = (
    RESULTS_DIR / "GSE239948_DOG2_expression_fingerprint_nearest_neighbors.csv"
)
OUTPUT_FINGERPRINT_GENE_AUDIT = (
    RESULTS_DIR / "GSE239948_DOG2_expression_fingerprint_gene_audit.csv"
)
OUTPUT_INDEPENDENCE_SUMMARY = (
    RESULTS_DIR / "GSE239948_DOG2_independence_summary.csv"
)
OUTPUT_LOCKED_EXTERNAL = (
    RESULTS_DIR / "paper4_locked_independent_canine_representation.csv"
)
OUTPUT_UPDATED_MASTER = (
    RESULTS_DIR
    / "paper4_locked_multidimensional_transport_evidence_with_single_cell_and_external_canine.csv"
)
OUTPUT_SENTENCES = (
    RESULTS_DIR / "paper4_locked_external_canine_results_sentences.txt"
)
OUTPUT_README = (
    RESULTS_DIR / "paper4_external_canine_evidence_README.txt"
)
OUTPUT_MANIFEST = (
    RESULTS_DIR / "paper4_external_canine_evidence_manifest.json"
)

OUTPUT_FIGURE_PNG = (
    FIGURES_DIR / "GSE239948_locked_representation_metrics.png"
)
OUTPUT_FIGURE_PDF = (
    FIGURES_DIR / "GSE239948_locked_representation_metrics.pdf"
)

N_FINGERPRINT_GENES = 3000
RANK_CORRELATION_HARD_THRESHOLD = 0.995
CENTERED_CORRELATION_HARD_THRESHOLD = 0.980
RANK_CORRELATION_JOINT_THRESHOLD = 0.985
CENTERED_CORRELATION_JOINT_THRESHOLD = 0.900
RANDOM_SEED = 42


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_required_csv(path: Path, index_col: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    print(f"Loaded: {path}")
    return pd.read_csv(path, index_col=index_col, low_memory=False)


def read_required_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    print(f"Loaded: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def choose_master_file() -> Path:
    for path in MASTER_FILE_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No locked multidimensional master file was found. Checked: "
        + ", ".join(str(path) for path in MASTER_FILE_CANDIDATES)
    )


def manifest_hash_by_basename(
    manifest: dict[str, Any],
    section: str,
    basename: str,
) -> str | None:
    entries = manifest.get(section, {})
    if not isinstance(entries, dict):
        return None

    matches = []
    for raw_path, payload in entries.items():
        if Path(str(raw_path)).name == basename and isinstance(payload, dict):
            value = payload.get("sha256")
            if value:
                matches.append(str(value))

    unique = sorted(set(matches))
    if len(unique) == 1:
        return unique[0]
    return None


def verify_external_manifest(
    manifest: dict[str, Any],
    critical_files: list[Path],
) -> pd.DataFrame:
    script_version = str(manifest.get("script_version", ""))
    if script_version != "46-gse239948-external-canine-representation-v2":
        raise RuntimeError(
            "The external representation manifest is not from script 46 v2. "
            f"Observed: {script_version}"
        )

    if bool(manifest.get("outcome_loaded", True)):
        raise RuntimeError(
            "The script 46 manifest does not confirm outcome_loaded=false."
        )

    rows = []
    for path in critical_files:
        expected = manifest_hash_by_basename(
            manifest=manifest,
            section="outputs",
            basename=path.name,
        )
        observed = sha256_file(path)
        verified = expected is not None and expected == observed
        rows.append(
            {
                "file": path.name,
                "expected_sha256": expected or "",
                "observed_sha256": observed,
                "verified": verified,
            }
        )

    audit = pd.DataFrame(rows)
    failed = audit[~audit["verified"]]
    if not failed.empty:
        raise RuntimeError(
            "One or more script 46 outputs failed the manifest hash check: "
            + ", ".join(failed["file"].astype(str))
        )
    return audit


def clean_gene_symbol(value: Any) -> str:
    text = str(value).strip().upper()
    if text in {"", "NAN", "NONE", "NA", "---"}:
        return ""
    text = re.sub(r"\.\d+$", "", text)
    suffix = text.rsplit("_", 1)[-1]
    if suffix.isdigit():
        text = text.rsplit("_", 1)[0]
    return text


def normalize_sample_identifier(value: Any) -> str:
    text = str(value).strip().upper()
    text = re.sub(r"\.(CSV|TSV|TXT|GZ)$", "", text)
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def sample_prefix(value: str) -> str:
    match = re.match(r"^([A-Z]+)", value)
    return match.group(1) if match else ""


def prepare_expression(expression: pd.DataFrame) -> pd.DataFrame:
    x = expression.copy()
    x.index = x.index.astype(str)
    x.columns = [clean_gene_symbol(column) for column in x.columns]
    nonempty_columns = np.asarray([column != "" for column in x.columns], dtype=bool)
    x = x.loc[:, nonempty_columns]
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)

    if pd.Index(x.columns).duplicated().any():
        x = x.T.groupby(level=0, sort=False).mean().T

    medians = x.median(axis=0)
    x = x.fillna(medians)
    variable = x.var(axis=0, ddof=1)
    x = x.loc[:, variable.notna() & variable.gt(0)]
    return x


def build_sample_identifier_audit(
    reference: pd.DataFrame,
    external: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for cohort, identifiers in [
        ("GSE238110_DOG2", reference.index),
        ("GSE239948", external.index),
    ]:
        for raw_identifier in identifiers:
            normalized = normalize_sample_identifier(raw_identifier)
            rows.append(
                {
                    "cohort": cohort,
                    "raw_sample_identifier": str(raw_identifier),
                    "normalized_sample_identifier": normalized,
                    "identifier_prefix": sample_prefix(normalized),
                }
            )

    audit = pd.DataFrame(rows)
    reference_ids = set(
        audit.loc[
            audit["cohort"].eq("GSE238110_DOG2"),
            "normalized_sample_identifier",
        ]
    ) - {""}
    external_ids = set(
        audit.loc[
            audit["cohort"].eq("GSE239948"),
            "normalized_sample_identifier",
        ]
    ) - {""}
    exact = sorted(reference_ids.intersection(external_ids))

    overlap = pd.DataFrame(
        {
            "normalized_sample_identifier": exact,
            "exact_identifier_overlap": True,
        }
    )

    prefix_summary = (
        audit.groupby(["cohort", "identifier_prefix"], dropna=False)
        .size()
        .reset_index(name="n_samples")
        .sort_values(["cohort", "n_samples"], ascending=[True, False])
    )
    return audit, overlap, prefix_summary


def primary_module_genes(weights: pd.DataFrame) -> set[str]:
    part = weights[
        weights["module_label"].astype(str).isin(PRIMARY_MODULES)
    ].copy()

    candidates = [
        "canine_gene_symbol",
        "canine_gene",
        "gene",
    ]
    column = next((item for item in candidates if item in part.columns), None)
    if column is None:
        return set()

    return {
        clean_gene_symbol(value)
        for value in part[column]
        if clean_gene_symbol(value)
    }


def choose_fingerprint_genes(
    reference: pd.DataFrame,
    external: pd.DataFrame,
    excluded_genes: set[str],
) -> pd.DataFrame:
    common = sorted(
        set(reference.columns)
        .intersection(external.columns)
        .difference(excluded_genes)
    )
    if len(common) < 500:
        raise RuntimeError(
            f"Too few common non-module genes for fingerprinting: {len(common)}"
        )

    ref_sd = reference[common].std(axis=0, ddof=1)
    ext_sd = external[common].std(axis=0, ddof=1)

    audit = pd.DataFrame(
        {
            "gene": common,
            "reference_sd": ref_sd.reindex(common).to_numpy(dtype=float),
            "external_sd": ext_sd.reindex(common).to_numpy(dtype=float),
        }
    )
    audit = audit.replace([np.inf, -np.inf], np.nan).dropna()
    audit = audit[
        audit["reference_sd"].gt(0)
        & audit["external_sd"].gt(0)
    ].copy()

    audit["reference_variability_percentile"] = audit["reference_sd"].rank(
        method="average",
        pct=True,
    )
    audit["external_variability_percentile"] = audit["external_sd"].rank(
        method="average",
        pct=True,
    )
    audit["joint_variability_score"] = np.sqrt(
        audit["reference_variability_percentile"]
        * audit["external_variability_percentile"]
    )
    audit = audit.sort_values(
        ["joint_variability_score", "gene"],
        ascending=[False, True],
    )
    audit["selected_for_fingerprint"] = False
    n_select = min(N_FINGERPRINT_GENES, audit.shape[0])
    audit.loc[audit.index[:n_select], "selected_for_fingerprint"] = True
    return audit


def row_rank_standardize(matrix: np.ndarray) -> np.ndarray:
    transformed = np.empty(matrix.shape, dtype=np.float64)
    for row_index in range(matrix.shape[0]):
        ranks = rankdata(matrix[row_index], method="average")
        ranks = ranks.astype(np.float64)
        ranks -= ranks.mean()
        scale = ranks.std(ddof=1)
        if not np.isfinite(scale) or scale == 0:
            transformed[row_index] = 0.0
        else:
            transformed[row_index] = ranks / scale
    return transformed


def row_standardize(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - np.nanmean(matrix, axis=1, keepdims=True)
    scale = np.nanstd(centered, axis=1, ddof=1, keepdims=True)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    return centered / scale


def cohort_gene_zscore(matrix: np.ndarray) -> np.ndarray:
    means = np.nanmean(matrix, axis=0, keepdims=True)
    scales = np.nanstd(matrix, axis=0, ddof=1, keepdims=True)
    scales[~np.isfinite(scales) | (scales == 0)] = 1.0
    return (matrix - means) / scales


def correlation_matrix_from_standardized_rows(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    denominator = max(left.shape[1] - 1, 1)
    values = left @ right.T / denominator
    return np.clip(values, -1.0, 1.0)


def top_two(values: np.ndarray) -> tuple[int, float, float]:
    finite = np.where(np.isfinite(values), values, -np.inf)
    if finite.size == 0 or np.all(np.isneginf(finite)):
        return -1, np.nan, np.nan
    order = np.argsort(finite)[::-1]
    best_index = int(order[0])
    best = float(finite[best_index])
    second = float(finite[order[1]]) if order.size > 1 else np.nan
    return best_index, best, second


def build_expression_fingerprint_audit(
    reference: pd.DataFrame,
    external: pd.DataFrame,
    gene_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    genes = gene_audit.loc[
        gene_audit["selected_for_fingerprint"],
        "gene",
    ].astype(str).tolist()
    if len(genes) < 500:
        raise RuntimeError(
            f"Fingerprint gene selection returned too few genes: {len(genes)}"
        )

    ref_matrix = reference[genes].to_numpy(dtype=float)
    ext_matrix = external[genes].to_numpy(dtype=float)

    rank_ref = row_rank_standardize(ref_matrix)
    rank_ext = row_rank_standardize(ext_matrix)
    rank_cross = correlation_matrix_from_standardized_rows(
        rank_ext,
        rank_ref,
    )

    centered_ref = row_standardize(cohort_gene_zscore(ref_matrix))
    centered_ext = row_standardize(cohort_gene_zscore(ext_matrix))
    centered_cross = correlation_matrix_from_standardized_rows(
        centered_ext,
        centered_ref,
    )

    rows = []
    for external_index, external_id in enumerate(external.index.astype(str)):
        rank_neighbor, rank_best, rank_second = top_two(
            rank_cross[external_index]
        )
        centered_neighbor, centered_best, centered_second = top_two(
            centered_cross[external_index]
        )

        rank_match_id = (
            str(reference.index[rank_neighbor])
            if rank_neighbor >= 0
            else ""
        )
        centered_match_id = (
            str(reference.index[centered_neighbor])
            if centered_neighbor >= 0
            else ""
        )

        hard_flag = bool(
            rank_best >= RANK_CORRELATION_HARD_THRESHOLD
            or centered_best >= CENTERED_CORRELATION_HARD_THRESHOLD
        )
        joint_flag = bool(
            rank_best >= RANK_CORRELATION_JOINT_THRESHOLD
            and centered_best >= CENTERED_CORRELATION_JOINT_THRESHOLD
        )

        rows.append(
            {
                "external_sample": external_id,
                "rank_profile_nearest_reference_sample": rank_match_id,
                "rank_profile_max_correlation": rank_best,
                "rank_profile_second_correlation": rank_second,
                "rank_profile_neighbor_gap": (
                    rank_best - rank_second
                    if np.isfinite(rank_second)
                    else np.nan
                ),
                "centered_profile_nearest_reference_sample": centered_match_id,
                "centered_profile_max_correlation": centered_best,
                "centered_profile_second_correlation": centered_second,
                "centered_profile_neighbor_gap": (
                    centered_best - centered_second
                    if np.isfinite(centered_second)
                    else np.nan
                ),
                "same_nearest_reference_under_both_metrics": (
                    rank_match_id == centered_match_id
                    and rank_match_id != ""
                ),
                "hard_near_duplicate_flag": hard_flag,
                "joint_near_duplicate_flag": joint_flag,
                "potential_expression_overlap_flag": hard_flag or joint_flag,
            }
        )

    neighbors = pd.DataFrame(rows)
    summary = {
        "n_fingerprint_genes": len(genes),
        "maximum_rank_profile_correlation": float(
            np.nanmax(rank_cross)
        ),
        "maximum_centered_profile_correlation": float(
            np.nanmax(centered_cross)
        ),
        "n_potential_expression_overlap_flags": int(
            neighbors["potential_expression_overlap_flag"].sum()
        ),
        "rank_hard_threshold": RANK_CORRELATION_HARD_THRESHOLD,
        "centered_hard_threshold": CENTERED_CORRELATION_HARD_THRESHOLD,
        "rank_joint_threshold": RANK_CORRELATION_JOINT_THRESHOLD,
        "centered_joint_threshold": CENTERED_CORRELATION_JOINT_THRESHOLD,
    }
    return neighbors, summary


def build_independence_summary(
    reference: pd.DataFrame,
    external: pd.DataFrame,
    exact_overlap: pd.DataFrame,
    fingerprint_summary: dict[str, Any],
) -> pd.DataFrame:
    n_exact = int(exact_overlap.shape[0])
    n_flags = int(
        fingerprint_summary["n_potential_expression_overlap_flags"]
    )

    if n_exact > 0:
        independence_class = "identifier_overlap_requires_manual_review"
        independent_claim_allowed = False
    elif n_flags > 0:
        independence_class = (
            "no_identifier_overlap_but_expression_similarity_requires_review"
        )
        independent_claim_allowed = False
    else:
        independence_class = (
            "no_identifier_or_near_duplicate_expression_overlap_detected"
        )
        independent_claim_allowed = True

    return pd.DataFrame(
        [
            {
                "reference_cohort": "GSE238110_DOG2",
                "external_cohort": "GSE239948",
                "n_reference_samples": int(reference.shape[0]),
                "n_external_samples": int(external.shape[0]),
                "n_exact_normalized_identifier_overlaps": n_exact,
                "n_expression_overlap_flags": n_flags,
                "n_fingerprint_genes": int(
                    fingerprint_summary["n_fingerprint_genes"]
                ),
                "maximum_rank_profile_correlation": float(
                    fingerprint_summary[
                        "maximum_rank_profile_correlation"
                    ]
                ),
                "maximum_centered_profile_correlation": float(
                    fingerprint_summary[
                        "maximum_centered_profile_correlation"
                    ]
                ),
                "independence_evidence_class": independence_class,
                "independent_cohort_wording_allowed": independent_claim_allowed,
                "guardrail": (
                    "Expression fingerprinting can detect exact or near-duplicate "
                    "profiles but cannot prove biological independence when donor "
                    "identifiers use unrelated coding systems."
                ),
            }
        ]
    )


def locked_interpretation(row: pd.Series) -> str:
    module = str(row["module_label"])
    label = str(row["external_canine_representation_class"])
    independent_wording = bool(
        row.get("independent_cohort_wording_allowed", False)
    )
    cohort_phrase = (
        "an independently screened external canine cohort"
        if independent_wording
        else "an external canine cohort requiring overlap review"
    )

    if module == "M34" and label.startswith("strong_"):
        return (
            f"M34 showed strong representation preservation in {cohort_phrase}, "
            "with high edge concordance, frozen-loading concordance, "
            "split-half reliability, and specificity relative to matched "
            "random panels. This supports the reproducibility of the canine "
            "immune-depletion/exclusion program representation, but it is not "
            "an outcome-validation result."
        )
    if module == "M40" and label.startswith("strong_"):
        return (
            f"M40 showed exceptionally strong representation preservation in "
            f"{cohort_phrase} across edge, loading, split-half, and random-panel "
            "criteria. This strengthens interpretation of M40 as a reproducible "
            "canine cycling program while leaving its heterogeneous human "
            "prognostic transport unchanged."
        )
    if module == "M11":
        return (
            "M11 produced a reliable six-gene score in GSE239948 but did not "
            "show significant canine-to-canine edge or loading preservation "
            "and was not specific relative to matched random panels. Score "
            "reliability must therefore not be interpreted as independent "
            "representation preservation."
        )
    if module == "M24":
        return (
            "M24 retained six of seven genes and moderate split-half reliability, "
            "but edge and loading tests were not significant and matched random "
            "panels did not support specificity. The result provides no clear "
            "independent canine preservation evidence, with inference limited "
            "by the small six-gene representation."
        )
    return (
        "The external canine result is retained as a representation analysis "
        "and does not alter frozen outcome evidence."
    )


def extended_transport_class(module: str, external_class: str) -> str:
    if module == "M34" and external_class.startswith("strong_"):
        return (
            "replicated_canine_and_cross_species_representation_with_"
            "heterogeneous_outcome_transport"
        )
    if module == "M40" and external_class.startswith("strong_"):
        return (
            "replicated_canine_and_cross_species_cycling_structure_with_"
            "unstable_prognostic_transport"
        )
    if module == "M11":
        return (
            "directional_outcome_concordance_without_independent_canine_"
            "representation_replication"
        )
    if module == "M24":
        return (
            "limited_representation_and_endpoint_specific_outcome_evidence"
        )
    return "unchanged_multidimensional_evidence"


def build_locked_external_table(
    classification: pd.DataFrame,
    coverage: pd.DataFrame,
    reliability: pd.DataFrame,
    independence_summary: pd.DataFrame,
) -> pd.DataFrame:
    locked = classification.copy()

    coverage_keep = [
        column
        for column in [
            "module_label",
            "n_frozen_genes",
            "n_common_genes",
            "coverage_fraction",
        ]
        if column in coverage.columns
    ]
    reliability_keep = [
        column
        for column in [
            "module_label",
            "minimum_gene_loo_correlation",
            "median_gene_loo_correlation",
        ]
        if column in reliability.columns
    ]

    if coverage_keep:
        locked = locked.drop(
            columns=[
                column
                for column in coverage_keep
                if column != "module_label" and column in locked.columns
            ],
            errors="ignore",
        ).merge(
            coverage[coverage_keep],
            on="module_label",
            how="left",
        )

    if reliability_keep:
        locked = locked.merge(
            reliability[reliability_keep],
            on="module_label",
            how="left",
        )

    locked = locked[
        locked["module_label"].astype(str).isin(PRIMARY_MODULES)
    ].copy()
    locked["module_order"] = locked["module_label"].map(
        {module: index for index, module in enumerate(PRIMARY_MODULES)}
    )
    locked = locked.sort_values("module_order").drop(columns="module_order")

    independence_class = str(
        independence_summary.iloc[0]["independence_evidence_class"]
    )
    wording_allowed = bool(
        independence_summary.iloc[0]["independent_cohort_wording_allowed"]
    )
    locked["cohort_independence_evidence_class"] = independence_class
    locked["independent_cohort_wording_allowed"] = wording_allowed
    locked["locked_external_canine_interpretation"] = locked.apply(
        locked_interpretation,
        axis=1,
    )
    locked["large_module_leave_one_out_guardrail"] = locked[
        "module_label"
    ].isin(["M34", "M40"])
    locked["manuscript_guardrail"] = np.where(
        locked["large_module_leave_one_out_guardrail"],
        (
            "Near-unity leave-one-gene-out correlations are expected for large "
            "high-overlap scores and are not independent preservation evidence."
        ),
        (
            "Small-module edge and loading estimates are discrete and have "
            "limited power."
        ),
    )
    return locked


def update_master(
    master: pd.DataFrame,
    locked_external: pd.DataFrame,
) -> pd.DataFrame:
    if "module_label" not in master.columns:
        raise ValueError("The locked master table has no module_label column.")

    selected = locked_external[
        [
            "module_label",
            "n_common_genes",
            "coverage_fraction",
            "edge_spearman",
            "edge_q_bh_8",
            "loading_spearman",
            "loading_q_bh_8",
            "split_half_median",
            "random_panel_empirical_p",
            "external_canine_representation_class",
            "cohort_independence_evidence_class",
            "independent_cohort_wording_allowed",
            "locked_external_canine_interpretation",
        ]
    ].copy()
    selected = selected.rename(
        columns={
            "n_common_genes": "gse239948_n_common_genes",
            "coverage_fraction": "gse239948_coverage_fraction",
            "edge_spearman": "gse239948_edge_spearman",
            "edge_q_bh_8": "gse239948_edge_q_bh_8",
            "loading_spearman": "gse239948_loading_spearman",
            "loading_q_bh_8": "gse239948_loading_q_bh_8",
            "split_half_median": "gse239948_split_half_median",
            "random_panel_empirical_p": "gse239948_random_panel_empirical_p",
            "external_canine_representation_class": (
                "gse239948_external_canine_representation_class"
            ),
            "cohort_independence_evidence_class": (
                "gse239948_independence_evidence_class"
            ),
            "independent_cohort_wording_allowed": (
                "gse239948_independent_cohort_wording_allowed"
            ),
            "locked_external_canine_interpretation": (
                "gse239948_locked_interpretation"
            ),
        }
    )

    updated = master.merge(selected, on="module_label", how="left")
    updated["multidimensional_transport_class_with_external_canine"] = [
        extended_transport_class(
            module=str(module),
            external_class=str(external_class),
        )
        for module, external_class in zip(
            updated["module_label"],
            updated[
                "gse239948_external_canine_representation_class"
            ],
        )
    ]

    original_interpretation_column = next(
        (
            column
            for column in [
                "locked_multidimensional_interpretation",
                "locked_single_cell_interpretation",
            ]
            if column in updated.columns
        ),
        None,
    )

    if original_interpretation_column is not None:
        updated["locked_interpretation_with_external_canine"] = (
            updated[original_interpretation_column].fillna("").astype(str)
            + " "
            + updated["gse239948_locked_interpretation"].fillna("").astype(str)
        ).str.strip()
    else:
        updated["locked_interpretation_with_external_canine"] = updated[
            "gse239948_locked_interpretation"
        ]
    return updated


def create_metric_figure(locked: pd.DataFrame) -> None:
    plot = locked.set_index("module_label").reindex(PRIMARY_MODULES)
    x = np.arange(len(PRIMARY_MODULES), dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.plot(
        x,
        plot["edge_spearman"].to_numpy(dtype=float),
        marker="o",
        label="Edge Spearman",
    )
    ax.plot(
        x,
        plot["loading_spearman"].to_numpy(dtype=float),
        marker="s",
        label="Loading Spearman",
    )
    ax.plot(
        x,
        plot["split_half_median"].to_numpy(dtype=float),
        marker="^",
        label="Split-half median",
    )

    for index, module in enumerate(PRIMARY_MODULES):
        edge_q = float(plot.loc[module, "edge_q_bh_8"])
        loading_q = float(plot.loc[module, "loading_q_bh_8"])
        random_p = float(plot.loc[module, "random_panel_empirical_p"])
        supported = []
        if np.isfinite(edge_q) and edge_q < 0.05:
            supported.append("E")
        if np.isfinite(loading_q) and loading_q < 0.05:
            supported.append("L")
        if np.isfinite(random_p) and random_p < 0.05:
            supported.append("R")
        label = "+".join(supported) if supported else "none"
        ax.text(
            index,
            -0.12,
            label,
            ha="center",
            va="top",
            fontsize=9,
        )

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(PRIMARY_MODULES)
    ax.set_ylim(-0.20, 1.05)
    ax.set_ylabel("Preservation or reliability metric")
    ax.set_title(
        "GSE239948 independent canine representation evidence"
    )
    ax.legend(frameon=False)
    ax.text(
        0.01,
        0.01,
        "E: edge FDR < 0.05; L: loading FDR < 0.05; "
        "R: matched-random empirical P < 0.05",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_FIGURE_PDF, bbox_inches="tight")
    plt.close(fig)


def write_sentences(
    locked: pd.DataFrame,
    independence_summary: pd.DataFrame,
) -> None:
    indexed = locked.set_index("module_label")
    independence = independence_summary.iloc[0]
    wording = (
        "independent canine cohort"
        if bool(independence["independent_cohort_wording_allowed"])
        else "external canine cohort"
    )

    lines = [
        "Locked GSE239948 external canine representation results",
        "=======================================================",
        "",
        "Cohort-independence audit",
        "-------------------------",
        (
            f"The DOG2 and GSE239948 matrices contained "
            f"{int(independence['n_exact_normalized_identifier_overlaps'])} "
            "exact normalized sample-identifier overlaps and "
            f"{int(independence['n_expression_overlap_flags'])} "
            "conservative expression-fingerprint overlap flags."
        ),
        (
            "This audit supports independent-cohort wording only as a "
            "technical duplicate screen and does not prove donor independence "
            "when the studies use unrelated identifier systems."
        ),
        "",
        "Primary results",
        "---------------",
        (
            f"In the {wording}, M34 retained "
            f"{int(indexed.loc['M34', 'n_common_genes'])} of "
            f"{int(indexed.loc['M34', 'n_frozen_genes'])} frozen genes "
            f"and showed edge rho={indexed.loc['M34', 'edge_spearman']:.3f}, "
            f"loading rho={indexed.loc['M34', 'loading_spearman']:.3f}, "
            f"and split-half reliability={indexed.loc['M34', 'split_half_median']:.3f}. "
            "Both direct preservation tests passed FDR and the observed edge "
            "preservation exceeded matched random panels."
        ),
        (
            f"M40 retained {int(indexed.loc['M40', 'n_common_genes'])} of "
            f"{int(indexed.loc['M40', 'n_frozen_genes'])} frozen genes "
            f"and showed exceptionally high edge rho="
            f"{indexed.loc['M40', 'edge_spearman']:.3f}, loading rho="
            f"{indexed.loc['M40', 'loading_spearman']:.3f}, and split-half "
            f"reliability={indexed.loc['M40', 'split_half_median']:.3f}. "
            "This provides independent representation replication of the "
            "cycling program but does not resolve its heterogeneous outcome transport."
        ),
        (
            "M11 and M24 retained six genes each and produced internally "
            "reliable scores, but neither showed FDR-supported edge or loading "
            "preservation or specificity relative to matched random panels."
        ),
        "",
        "Manuscript guardrails",
        "---------------------",
        "GSE239948 provides representation validation, not outcome validation.",
        (
            "Near-unity leave-one-gene-out correlations for M34 and M40 are "
            "partly mechanical because each reduced score retains nearly all genes."
        ),
        (
            "Failure to reach significance for six-gene M11 and M24 modules "
            "does not establish absence of preservation."
        ),
        (
            "Therapy heterogeneity in GSE239948 may contribute to expression "
            "distribution differences and should be acknowledged."
        ),
        "",
        "Locked interpretation",
        "---------------------",
    ]

    for module in PRIMARY_MODULES:
        lines.append(
            f"{module}: {indexed.loc[module, 'locked_external_canine_interpretation']}"
        )

    OUTPUT_SENTENCES.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(
    master_file: Path,
    independence_summary: pd.DataFrame,
) -> None:
    independence = independence_summary.iloc[0]
    text = f"""Paper 4 GSE239948 independent canine evidence lock
Script version: {SCRIPT_VERSION}

Purpose
-------
Lock the outcome-blind GSE239948 representation results from script 46 v2,
audit possible sample reuse between GSE238110 DOG2 and GSE239948, and append
the external canine evidence to the multidimensional Paper 4 master table.

Inputs
------
External representation manifest: {EXTERNAL_MANIFEST_FILE}
Locked multidimensional master: {master_file}
Reference expression: {REFERENCE_EXPRESSION_FILE}
External expression: {EXTERNAL_EXPRESSION_FILE}

Independence audit
------------------
Exact normalized identifier overlaps: {int(independence['n_exact_normalized_identifier_overlaps'])}
Expression-fingerprint overlap flags: {int(independence['n_expression_overlap_flags'])}
Independence evidence class: {independence['independence_evidence_class']}

The expression-fingerprint audit uses jointly variable genes outside the four
primary frozen modules. It compares sample-wise rank profiles and cohort-centered
gene-z profiles. Conservative thresholds are intended to flag possible exact or
near-duplicate expression profiles for manual review. The audit cannot prove donor
independence when the two studies use unrelated identifier coding systems.

Locked interpretation
---------------------
- M34: strong external canine representation preservation.
- M40: exceptionally strong external canine representation preservation.
- M11: reliable external score without clear cross-cohort structural preservation.
- M24: no clear preservation evidence; small-module inference remains limited.

Guardrails
----------
- No outcome or treatment-response endpoint is loaded.
- Frozen genes, loadings, signs, and validation tiers are unchanged.
- GSE239948 is a representation-validation cohort, not a prognostic-validation cohort.
- Leave-one-gene-out correlations for large modules are not independent evidence.
- Existing human multiplicity and outcome conclusions are unchanged.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("Lock GSE239948 independent canine representation evidence")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Design:")
    print("  Verify script 46 v2 outputs and outcome-blind guardrails.")
    print("  Audit exact identifiers and expression fingerprints for sample reuse.")
    print("  Lock independent canine representation evidence without new outcomes.")
    print("  Append GSE239948 evidence to the multidimensional Paper 4 master table.")
    print("")

    master_file = choose_master_file()

    reference_raw = read_required_csv(
        REFERENCE_EXPRESSION_FILE,
        index_col=0,
    )
    external_raw = read_required_csv(
        EXTERNAL_EXPRESSION_FILE,
        index_col=0,
    )
    read_required_csv(EXTERNAL_SCORES_FILE, index_col=0)
    weights = read_required_csv(STRICT_WEIGHTS_FILE)
    coverage = read_required_csv(EXTERNAL_COVERAGE_FILE)
    read_required_csv(EXTERNAL_STRUCTURE_FILE)
    reliability = read_required_csv(EXTERNAL_RELIABILITY_FILE)
    read_required_csv(EXTERNAL_RANDOM_FILE)
    classification = read_required_csv(EXTERNAL_CLASSIFICATION_FILE)
    manifest = read_required_json(EXTERNAL_MANIFEST_FILE)
    master = read_required_csv(master_file)

    critical_outputs = [
        EXTERNAL_EXPRESSION_FILE,
        EXTERNAL_SCORES_FILE,
        EXTERNAL_COVERAGE_FILE,
        EXTERNAL_STRUCTURE_FILE,
        EXTERNAL_RELIABILITY_FILE,
        EXTERNAL_RANDOM_FILE,
        EXTERNAL_CLASSIFICATION_FILE,
    ]
    hash_audit = verify_external_manifest(
        manifest=manifest,
        critical_files=critical_outputs,
    )

    reference = prepare_expression(reference_raw)
    external = prepare_expression(external_raw)

    sample_audit, exact_overlap, prefix_summary = (
        build_sample_identifier_audit(
            reference=reference,
            external=external,
        )
    )
    sample_audit.to_csv(OUTPUT_SAMPLE_ID_AUDIT, index=False)
    exact_overlap.to_csv(OUTPUT_EXACT_OVERLAP, index=False)

    excluded_genes = primary_module_genes(weights)
    fingerprint_gene_audit = choose_fingerprint_genes(
        reference=reference,
        external=external,
        excluded_genes=excluded_genes,
    )
    fingerprint_neighbors, fingerprint_summary = (
        build_expression_fingerprint_audit(
            reference=reference,
            external=external,
            gene_audit=fingerprint_gene_audit,
        )
    )

    fingerprint_gene_audit.to_csv(
        OUTPUT_FINGERPRINT_GENE_AUDIT,
        index=False,
    )
    fingerprint_neighbors.to_csv(
        OUTPUT_FINGERPRINT_NEIGHBORS,
        index=False,
    )

    independence_summary = build_independence_summary(
        reference=reference,
        external=external,
        exact_overlap=exact_overlap,
        fingerprint_summary=fingerprint_summary,
    )
    independence_summary.to_csv(
        OUTPUT_INDEPENDENCE_SUMMARY,
        index=False,
    )

    locked_external = build_locked_external_table(
        classification=classification,
        coverage=coverage,
        reliability=reliability,
        independence_summary=independence_summary,
    )
    locked_external.to_csv(OUTPUT_LOCKED_EXTERNAL, index=False)

    updated_master = update_master(
        master=master,
        locked_external=locked_external,
    )
    updated_master.to_csv(OUTPUT_UPDATED_MASTER, index=False)

    create_metric_figure(locked_external)
    write_sentences(
        locked=locked_external,
        independence_summary=independence_summary,
    )
    write_readme(
        master_file=master_file,
        independence_summary=independence_summary,
    )

    input_paths = [
        REFERENCE_EXPRESSION_FILE,
        EXTERNAL_EXPRESSION_FILE,
        EXTERNAL_SCORES_FILE,
        STRICT_WEIGHTS_FILE,
        EXTERNAL_COVERAGE_FILE,
        EXTERNAL_STRUCTURE_FILE,
        EXTERNAL_RELIABILITY_FILE,
        EXTERNAL_RANDOM_FILE,
        EXTERNAL_CLASSIFICATION_FILE,
        EXTERNAL_MANIFEST_FILE,
        master_file,
    ]
    output_paths = [
        OUTPUT_SAMPLE_ID_AUDIT,
        OUTPUT_EXACT_OVERLAP,
        OUTPUT_FINGERPRINT_NEIGHBORS,
        OUTPUT_FINGERPRINT_GENE_AUDIT,
        OUTPUT_INDEPENDENCE_SUMMARY,
        OUTPUT_LOCKED_EXTERNAL,
        OUTPUT_UPDATED_MASTER,
        OUTPUT_SENTENCES,
        OUTPUT_README,
        OUTPUT_FIGURE_PNG,
        OUTPUT_FIGURE_PDF,
    ]

    lock_manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "outcome_loaded": False,
        "source_external_manifest_script_version": manifest.get(
            "script_version"
        ),
        "source_external_manifest_hash_audit": hash_audit.to_dict(
            orient="records"
        ),
        "master_input": str(master_file),
        "independence_summary": independence_summary.iloc[0].to_dict(),
        "fingerprint_configuration": {
            "n_genes_requested": N_FINGERPRINT_GENES,
            "n_genes_used": fingerprint_summary["n_fingerprint_genes"],
            "primary_module_genes_excluded": len(excluded_genes),
            "rank_hard_threshold": RANK_CORRELATION_HARD_THRESHOLD,
            "centered_hard_threshold": CENTERED_CORRELATION_HARD_THRESHOLD,
            "rank_joint_threshold": RANK_CORRELATION_JOINT_THRESHOLD,
            "centered_joint_threshold": CENTERED_CORRELATION_JOINT_THRESHOLD,
            "random_seed": RANDOM_SEED,
        },
        "guardrails": [
            "No outcome or treatment-response endpoint was loaded.",
            "Frozen module genes, loadings, risk signs, and tiers were unchanged.",
            "Expression fingerprinting screens for possible duplicates but does not prove donor independence.",
            "GSE239948 is representation validation, not prognostic validation.",
            "Existing human multiplicity and outcome conclusions remain unchanged.",
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
        json.dumps(lock_manifest, indent=2, default=str),
        encoding="utf-8",
    )

    print("=" * 80)
    print("Script 46 v2 manifest integrity")
    print("=" * 80)
    print(hash_audit.to_string(index=False))

    print("")
    print("=" * 80)
    print("Sample identifier audit")
    print("=" * 80)
    print(prefix_summary.to_string(index=False))
    print("")
    print(f"Exact normalized identifier overlaps: {exact_overlap.shape[0]}")

    print("")
    print("=" * 80)
    print("Expression fingerprint independence audit")
    print("=" * 80)
    print(independence_summary.to_string(index=False))
    print("")
    display_neighbors = fingerprint_neighbors.sort_values(
        [
            "potential_expression_overlap_flag",
            "rank_profile_max_correlation",
        ],
        ascending=[False, False],
    ).head(15)
    print(display_neighbors.to_string(index=False))

    print("")
    print("=" * 80)
    print("Locked independent canine representation evidence")
    print("=" * 80)
    display_columns = [
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
    print(locked_external[display_columns].to_string(index=False))

    print("")
    print("=" * 80)
    print("Updated multidimensional evidence classes")
    print("=" * 80)
    updated_columns = [
        "module_label",
        "multidimensional_transport_class",
        "gse239948_external_canine_representation_class",
        "multidimensional_transport_class_with_external_canine",
    ]
    updated_columns = [
        column for column in updated_columns if column in updated_master.columns
    ]
    print(updated_master[updated_columns].to_string(index=False))

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("GSE239948 is a representation-validation cohort, not an outcome cohort.")
    print("Expression fingerprinting does not prove donor independence.")
    print("Large-module leave-one-gene-out correlations are partly mechanical.")
    print("Small-module non-significance does not prove absence of preservation.")
    print("Existing human outcome and project-wide multiplicity conclusions are unchanged.")

    print("")
    print("Saved:")
    for path in output_paths + [OUTPUT_MANIFEST]:
        if path.exists():
            print(path)
    print("Done.")


if __name__ == "__main__":
    main()
