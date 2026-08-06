from __future__ import annotations

from itertools import combinations
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_VERSION = "36-align-frozen-modules-to-mofa-v2"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "multigroup_mofa_alignment"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_INDEX_FILE = RESULTS_DIR / "multigroup_mofa_model_index.csv"
FACTOR_ACTIVITY_FILE = RESULTS_DIR / "multigroup_mofa_factor_activity.csv"
MODEL_SUMMARY_FILE = RESULTS_DIR / "multigroup_mofa_model_summary.csv"
MOFA_MANIFEST_FILE = RESULTS_DIR / "multigroup_mofa_unsupervised_freeze_manifest.json"
STRICT_WEIGHTS_FILE = (
    RESULTS_DIR / "GSE238110_frozen_transfer_gene_weights_strict.csv"
)
MSFA_CONFIG_FILE = RESULTS_DIR / "multistudy_factor_model_config.json"

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]
PRIMARY_ACTIVITY_THRESHOLD_PERCENT = 1.0
N_MATCHED_RANDOM_SETS = 2000
MIN_MODULE_GENES = 3
N_VARIABILITY_BINS = 10
RANDOM_SEED = 42

KEY_SUBSPACES = [
    "all_retained",
    "shared_two_or_more",
    "ubiquitous_all_groups",
    "partially_shared",
    "shared_non_ffpe_only",
    "gse39055_shared",
    "gse39055_specific",
]

OUTPUT_FACTOR_ALIGNMENT = (
    RESULTS_DIR / "multigroup_mofa_frozen_module_factor_alignment.csv"
)
OUTPUT_SUBSPACE_CAPTURE = (
    RESULTS_DIR / "multigroup_mofa_frozen_module_subspace_capture.csv"
)
OUTPUT_CAPTURE_RANDOM = (
    RESULTS_DIR / "multigroup_mofa_module_capture_random_controls.csv"
)
OUTPUT_RANK_SUMMARY = (
    RESULTS_DIR / "multigroup_mofa_module_capture_rank_summary.csv"
)
OUTPUT_CROSS_SET = (
    RESULTS_DIR / "multigroup_mofa_module_cross_set_sensitivity.csv"
)
OUTPUT_PCA_CAPTURE = (
    RESULTS_DIR / "multigroup_mofa_stacked_pca_module_capture.csv"
)
OUTPUT_MOFA_VS_PCA = (
    RESULTS_DIR / "multigroup_mofa_module_capture_vs_pca.csv"
)
OUTPUT_FACTOR_PATTERNS = (
    RESULTS_DIR / "multigroup_mofa_factor_activity_patterns.csv"
)
OUTPUT_INTERPRETATION = (
    RESULTS_DIR / "multigroup_mofa_module_representation_interpretation.csv"
)
OUTPUT_README = RESULTS_DIR / "multigroup_mofa_alignment_README.txt"
OUTPUT_MANIFEST = RESULTS_DIR / "multigroup_mofa_alignment_manifest.json"


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


def orthonormal_basis(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        return np.empty((values.shape[0], 0), dtype=float)

    q, r = np.linalg.qr(values)
    diagonal = np.abs(np.diag(r))
    if diagonal.size == 0:
        return np.empty((values.shape[0], 0), dtype=float)

    tolerance = max(values.shape) * np.finfo(float).eps * diagonal.max()
    rank = int(np.sum(diagonal > tolerance))
    return q[:, :rank]


def subspace_capture(vector: np.ndarray, loadings: np.ndarray) -> float:
    v = np.asarray(vector, dtype=float)
    denominator = float(v @ v)
    if denominator <= 0 or not np.isfinite(denominator):
        return np.nan

    basis = orthonormal_basis(loadings)
    if basis.shape[1] == 0:
        return np.nan

    projected = basis.T @ v
    return float(np.clip((projected @ projected) / denominator, 0.0, 1.0))


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0 or not np.isfinite(denominator):
        return np.nan
    return float((a @ b) / denominator)


def parse_active_groups(value: Any) -> list[str]:
    text = str(value).strip()
    if text in {"", "nan", "None"}:
        return []
    return [item for item in text.split(";") if item]


def read_gene_metadata(config: dict[str, Any], analysis_set: str) -> pd.DataFrame:
    path = Path(config["analysis_sets"][analysis_set]["gene_metadata"])
    metadata = read_required_csv(path)
    metadata["human_gene_symbol"] = (
        metadata["human_gene_symbol"].astype(str).str.upper()
    )
    return metadata.drop_duplicates("human_gene_symbol", keep="first")


def variability_metric_name(analysis_set: str) -> str:
    if analysis_set == "three_cohort_no_ffpe":
        return "median_three_cohort_variability_percentile"
    return "median_four_cohort_variability_percentile"


def add_variability_bins(
    metadata: pd.DataFrame,
    analysis_set: str,
) -> pd.DataFrame:
    result = metadata.copy()
    metric = variability_metric_name(analysis_set)

    if metric not in result.columns:
        raise ValueError(
            f"Missing variability metric {metric} for {analysis_set}."
        )

    values = pd.to_numeric(result[metric], errors="coerce")
    result["variability_bin"] = pd.qcut(
        values.rank(method="average"),
        q=min(N_VARIABILITY_BINS, int(values.notna().sum())),
        labels=False,
        duplicates="drop",
    )
    return result


def align_metadata_to_genes(
    metadata: pd.DataFrame,
    genes: list[str],
) -> pd.DataFrame:
    result = metadata.copy()

    if "human_gene_symbol" not in result.columns:
        if result.index.name == "human_gene_symbol":
            result["human_gene_symbol"] = result.index.astype(str)
        else:
            raise KeyError(
                "human_gene_symbol is missing from gene metadata."
            )

    result["human_gene_symbol"] = (
        result["human_gene_symbol"].astype(str).str.upper()
    )
    result = result.drop_duplicates(
        "human_gene_symbol",
        keep="first",
    )

    # Preserve human_gene_symbol as an explicit column after reindexing.
    result = (
        result.set_index("human_gene_symbol", drop=False)
        .reindex(genes)
    )
    result["human_gene_symbol"] = pd.Index(
        genes,
        dtype="object",
    )
    return result.reset_index(drop=True)


def frozen_module_vector(
    weights: pd.DataFrame,
    module: str,
    genes: list[str],
) -> tuple[np.ndarray, int, list[str]]:
    part = weights[weights["module_label"].astype(str).eq(module)].copy()
    part["human_gene_symbol"] = (
        part["human_gene_symbol"].astype(str).str.upper()
    )
    part = part.drop_duplicates("human_gene_symbol", keep="first")
    loading_map = (
        part.set_index("human_gene_symbol")["risk_oriented_loading"]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_dict()
    )

    vector = np.asarray(
        [float(loading_map.get(gene, 0.0)) for gene in genes],
        dtype=float,
    )
    present = [gene for gene in genes if gene in loading_map]
    return vector, len(present), present


def factor_subspaces(
    weights: pd.DataFrame,
    activity: pd.DataFrame,
    groups: list[str],
) -> dict[str, list[str]]:
    retained = [str(column) for column in weights.columns]
    activity = activity[
        activity["activity_threshold_percent"].eq(
            PRIMARY_ACTIVITY_THRESHOLD_PERCENT
        )
    ].copy()
    activity["factor"] = activity["factor"].astype(str)
    activity = activity[activity["factor"].isin(retained)]

    active_map = {
        row.factor: parse_active_groups(row.active_groups)
        for row in activity.itertuples(index=False)
    }
    n_groups = len(groups)

    result = {
        "all_retained": retained,
        "shared_two_or_more": [
            factor
            for factor in retained
            if len(active_map.get(factor, [])) >= 2
        ],
        "ubiquitous_all_groups": [
            factor
            for factor in retained
            if len(active_map.get(factor, [])) == n_groups
        ],
        "partially_shared": [
            factor
            for factor in retained
            if 2 <= len(active_map.get(factor, [])) < n_groups
        ],
        "shared_non_ffpe_only": [
            factor
            for factor in retained
            if (
                len(active_map.get(factor, [])) >= 2
                and "GSE39055" not in active_map.get(factor, [])
            )
        ],
        "gse39055_shared": [
            factor
            for factor in retained
            if (
                "GSE39055" in active_map.get(factor, [])
                and len(active_map.get(factor, [])) >= 2
            )
        ],
        "gse39055_specific": [
            factor
            for factor in retained
            if active_map.get(factor, []) == ["GSE39055"]
        ],
    }
    return result


def matched_random_vectors(
    module_vector: np.ndarray,
    genes: list[str],
    metadata: pd.DataFrame,
    n_random: int,
    seed: int,
) -> np.ndarray:
    nonzero_indices = np.where(module_vector != 0)[0]
    n_module_genes = len(nonzero_indices)
    if n_module_genes < MIN_MODULE_GENES:
        return np.empty((0, len(genes)), dtype=float)

    metadata_aligned = align_metadata_to_genes(
        metadata=metadata,
        genes=genes,
    )
    metadata_index = metadata_aligned.set_index(
        "human_gene_symbol",
        drop=False,
    )
    bins = metadata_index["variability_bin"].to_numpy()
    module_bins = bins[nonzero_indices]
    loading_values = module_vector[nonzero_indices].copy()

    candidate_indices = np.arange(len(genes))
    module_index_set = set(nonzero_indices.tolist())
    rng = np.random.default_rng(seed)
    random_vectors = np.zeros((n_random, len(genes)), dtype=float)

    for repeat in range(n_random):
        chosen: list[int] = []
        used = set(module_index_set)

        for target_bin in module_bins:
            if np.isfinite(target_bin):
                exact_pool = [
                    index
                    for index in candidate_indices
                    if (
                        index not in used
                        and np.isfinite(bins[index])
                        and bins[index] == target_bin
                    )
                ]
            else:
                exact_pool = []

            if not exact_pool:
                valid_pool = [
                    index
                    for index in candidate_indices
                    if index not in used and np.isfinite(bins[index])
                ]
                if valid_pool and np.isfinite(target_bin):
                    distances = np.asarray(
                        [abs(float(bins[index]) - float(target_bin)) for index in valid_pool]
                    )
                    minimum_distance = float(distances.min())
                    exact_pool = [
                        index
                        for index, distance in zip(valid_pool, distances)
                        if distance == minimum_distance
                    ]
                else:
                    exact_pool = [
                        index
                        for index in candidate_indices
                        if index not in used
                    ]

            if not exact_pool:
                chosen = []
                break

            selected = int(rng.choice(exact_pool))
            chosen.append(selected)
            used.add(selected)

        if len(chosen) != n_module_genes:
            continue

        permuted_loadings = rng.permutation(loading_values)
        random_vectors[repeat, chosen] = permuted_loadings

    valid = np.linalg.norm(random_vectors, axis=1) > 0
    return random_vectors[valid]


def capture_random_control(
    observed_vector: np.ndarray,
    loadings: np.ndarray,
    random_vectors: np.ndarray,
) -> dict[str, float]:
    observed = subspace_capture(observed_vector, loadings)
    if random_vectors.shape[0] == 0 or not np.isfinite(observed):
        return {
            "observed_capture": observed,
            "n_random_valid": 0,
            "random_mean": np.nan,
            "random_q95": np.nan,
            "observed_percentile": np.nan,
            "empirical_p_greater_equal": np.nan,
        }

    basis = orthonormal_basis(loadings)
    denominators = np.sum(random_vectors**2, axis=1)
    projected = random_vectors @ basis
    captures = np.sum(projected**2, axis=1) / denominators
    captures = captures[np.isfinite(captures)]

    if captures.size == 0:
        return {
            "observed_capture": observed,
            "n_random_valid": 0,
            "random_mean": np.nan,
            "random_q95": np.nan,
            "observed_percentile": np.nan,
            "empirical_p_greater_equal": np.nan,
        }

    empirical_p = (
        1.0 + np.sum(captures >= observed)
    ) / (captures.size + 1.0)

    return {
        "observed_capture": observed,
        "n_random_valid": int(captures.size),
        "random_mean": float(captures.mean()),
        "random_q95": float(np.quantile(captures, 0.95)),
        "observed_percentile": float(np.mean(captures <= observed)),
        "empirical_p_greater_equal": float(empirical_p),
    }


def pca_capture(
    pca_loadings_path: Path,
    module_vector: np.ndarray,
) -> tuple[float, int]:
    if not pca_loadings_path.exists():
        return np.nan, 0

    loadings = pd.read_csv(pca_loadings_path, index_col=0)
    matrix = loadings.to_numpy(dtype=float)
    return subspace_capture(module_vector, matrix), matrix.shape[1]


def build_interpretation(
    rank_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for module in PRIMARY_MODULES:
        part = rank_summary[
            rank_summary["module_label"].eq(module)
        ].copy()

        def median_capture(set_name: str, subspace: str) -> float:
            values = part[
                part["analysis_set"].eq(set_name)
                & part["subspace_name"].eq(subspace)
            ]["median_capture"]
            return float(values.iloc[0]) if not values.empty else np.nan

        core_ubiquitous = median_capture(
            "four_cohort_core_plus_frozen",
            "ubiquitous_all_groups",
        )
        core_non_ffpe = median_capture(
            "four_cohort_core_plus_frozen",
            "shared_non_ffpe_only",
        )
        detection_ubiquitous = median_capture(
            "four_cohort_detection_aware",
            "ubiquitous_all_groups",
        )
        no_ffpe_ubiquitous = median_capture(
            "three_cohort_no_ffpe",
            "ubiquitous_all_groups",
        )

        if (
            np.isfinite(core_ubiquitous)
            and core_ubiquitous >= 0.50
            and np.isfinite(no_ffpe_ubiquitous)
            and no_ffpe_ubiquitous >= 0.50
        ):
            interpretation_class = "strong_shared_latent_representation"
        elif (
            np.isfinite(no_ffpe_ubiquitous)
            and no_ffpe_ubiquitous >= 0.50
            and (
                not np.isfinite(core_ubiquitous)
                or core_ubiquitous < 0.50
            )
        ):
            interpretation_class = (
                "shared_non_ffpe_representation_attenuated_by_gse39055"
            )
        elif (
            np.isfinite(core_non_ffpe)
            and core_non_ffpe >= 0.50
        ):
            interpretation_class = "partially_shared_non_ffpe_representation"
        else:
            interpretation_class = "limited_shared_latent_capture"

        rows.append(
            {
                "module_label": module,
                "core_ubiquitous_median_capture": core_ubiquitous,
                "core_non_ffpe_partial_median_capture": core_non_ffpe,
                "detection_aware_ubiquitous_median_capture": detection_ubiquitous,
                "no_ffpe_ubiquitous_median_capture": no_ffpe_ubiquitous,
                "mofa_representation_class": interpretation_class,
                "interpretation_note": (
                    "This classification summarizes factor-subspace alignment "
                    "only and does not use or modify outcome results."
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_rank_capture_heatmap(
    rank_summary: pd.DataFrame,
    analysis_set: str,
    subspace_name: str,
) -> list[Path]:
    part = rank_summary[
        rank_summary["analysis_set"].eq(analysis_set)
        & rank_summary["subspace_name"].eq(subspace_name)
    ].copy()

    if part.empty:
        return []

    matrix = (
        part.pivot(
            index="module_label",
            columns="subspace_name",
            values="median_capture",
        )
        .reindex(index=PRIMARY_MODULES)
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    image = ax.imshow(matrix.values, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([subspace_name], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(PRIMARY_MODULES)))
    ax.set_yticklabels(PRIMARY_MODULES)
    ax.set_title(
        f"{analysis_set}\nMedian frozen-module capture across MOFA ranks"
    )

    for row in range(matrix.shape[0]):
        value = matrix.iloc[row, 0]
        label = "NA" if not np.isfinite(value) else f"{value:.2f}"
        ax.text(0, row, label, ha="center", va="center")

    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()

    safe_set = analysis_set.replace("/", "_")
    stem = f"MOFA_{safe_set}_{subspace_name}_capture"
    png = FIGURES_DIR / f"{stem}.png"
    pdf = FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def write_readme() -> None:
    text = f"""Frozen-module alignment to outcome-blind multi-group MOFA
Script version: {SCRIPT_VERSION}

Purpose
-------
Project the four frozen primary canine program-loading vectors onto the
outcome-blind MOFA weight subspaces frozen by script 35.

Primary metrics
---------------
1. Signed and absolute cosine similarity with individual factors.
2. Rotation-invariant squared projection of each frozen module vector onto:
   - all retained factors
   - factors active in at least two cohorts
   - factors active in every cohort
   - partially shared factors
   - shared factors excluding GSE39055
   - shared factors including GSE39055
   - GSE39055-specific factors
3. Rank sensitivity across 8, 12, and 16 initial factors.
4. Sensitivity to Detection-P-aware feature filtering and exclusion of GSE39055.
5. Comparison with the corresponding stacked-PCA subspace.

Matched random controls
-----------------------
For each module, random gene sets are matched approximately on the outcome-blind
cross-study variability percentile. The frozen loading values are permuted across
the matched genes. These controls assess representation specificity within the
fitted feature space.

Important limitation
--------------------
The core feature sets were enriched with available frozen primary-module genes.
Therefore, this analysis is a targeted latent-representation audit, not an
independent rediscovery of the modules. Matched random controls reduce but do
not eliminate this design dependence. A later variable-only factor sensitivity
can test independent recurrence without forced module inclusion.

Guardrails
----------
No clinical endpoint or outcome label is loaded.
The MOFA models and frozen module weights are read after their hashes were fixed.
No factor rank is selected using an outcome.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("Align frozen canine programs to multi-group MOFA subspaces")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Design:")
    print("  Read only frozen MOFA weights/activity and frozen canine loadings.")
    print("  Compute factor cosine similarity and rotation-invariant subspace capture.")
    print("  Use variability-matched random gene-set controls.")
    print("  Compare ranks, assay-aware sets, no-FFPE sensitivity, and stacked PCA.")
    print("  Load no clinical outcomes.")
    print("")

    model_index = read_required_csv(MODEL_INDEX_FILE)
    activity_all = read_required_csv(FACTOR_ACTIVITY_FILE)
    model_summary = read_required_csv(MODEL_SUMMARY_FILE)
    strict_weights = read_required_csv(STRICT_WEIGHTS_FILE)

    if not MOFA_MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Missing MOFA freeze manifest: {MOFA_MANIFEST_FILE}"
        )
    if not MSFA_CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Missing factor config: {MSFA_CONFIG_FILE}"
        )

    mofa_manifest = json.loads(
        MOFA_MANIFEST_FILE.read_text(encoding="utf-8")
    )
    config = json.loads(
        MSFA_CONFIG_FILE.read_text(encoding="utf-8")
    )

    if bool(mofa_manifest.get("outcome_loaded", True)):
        raise RuntimeError(
            "The MOFA manifest does not confirm outcome_loaded=false."
        )

    strict_weights["human_gene_symbol"] = (
        strict_weights["human_gene_symbol"].astype(str).str.upper()
    )

    factor_alignment_rows: list[dict[str, Any]] = []
    capture_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    pca_rows: list[dict[str, Any]] = []
    factor_pattern_rows: list[dict[str, Any]] = []
    generated_paths: list[Path] = []

    for model_row in model_index.itertuples(index=False):
        analysis_set = str(model_row.analysis_set)
        initial_factors = int(model_row.initial_factors)
        weights_path = Path(model_row.weights_path)
        activity_path = Path(model_row.activity_path)

        weights = read_required_csv(weights_path, index_col=0)
        weights.index = weights.index.astype(str).str.upper()
        weights.columns = weights.columns.astype(str)

        activity = read_required_csv(activity_path)
        primary_activity = activity[
            activity["activity_threshold_percent"].eq(
                PRIMARY_ACTIVITY_THRESHOLD_PERCENT
            )
        ].copy()
        primary_activity["factor"] = primary_activity["factor"].astype(str)

        groups = list(config["analysis_sets"][analysis_set]["cohorts"])
        metadata = add_variability_bins(
            read_gene_metadata(config, analysis_set),
            analysis_set,
        )
        genes = weights.index.tolist()
        metadata = align_metadata_to_genes(
            metadata=metadata,
            genes=genes,
        )

        subspaces = factor_subspaces(
            weights=weights,
            activity=activity,
            groups=groups,
        )

        for row in primary_activity.itertuples(index=False):
            factor_pattern_rows.append(
                {
                    "analysis_set": analysis_set,
                    "initial_factors": initial_factors,
                    "factor": str(row.factor),
                    "active_group_count": int(row.active_group_count),
                    "active_groups": str(row.active_groups),
                    "activity_class": str(row.activity_class),
                    "minimum_group_r2": float(row.minimum_group_r2),
                    "maximum_group_r2": float(row.maximum_group_r2),
                    "mean_group_r2": float(row.mean_group_r2),
                }
            )

        pca_path = (
            weights_path.parent
            / f"stacked_pca_k{initial_factors}_loadings.csv"
        )

        for module_index, module in enumerate(PRIMARY_MODULES):
            vector, n_module_genes, present_genes = frozen_module_vector(
                strict_weights,
                module,
                genes,
            )

            if n_module_genes < MIN_MODULE_GENES:
                continue

            for factor in weights.columns:
                factor_vector = weights[factor].to_numpy(dtype=float)
                cosine = cosine_similarity(vector, factor_vector)
                factor_row = primary_activity[
                    primary_activity["factor"].eq(str(factor))
                ]

                factor_alignment_rows.append(
                    {
                        "analysis_set": analysis_set,
                        "initial_factors": initial_factors,
                        "module_label": module,
                        "n_module_genes": n_module_genes,
                        "factor": str(factor),
                        "signed_cosine": cosine,
                        "absolute_cosine": abs(cosine)
                        if np.isfinite(cosine)
                        else np.nan,
                        "squared_cosine": cosine**2
                        if np.isfinite(cosine)
                        else np.nan,
                        "active_group_count": (
                            int(factor_row.iloc[0]["active_group_count"])
                            if not factor_row.empty
                            else np.nan
                        ),
                        "active_groups": (
                            str(factor_row.iloc[0]["active_groups"])
                            if not factor_row.empty
                            else ""
                        ),
                        "activity_class": (
                            str(factor_row.iloc[0]["activity_class"])
                            if not factor_row.empty
                            else ""
                        ),
                    }
                )

            random_vectors = matched_random_vectors(
                module_vector=vector,
                genes=genes,
                metadata=metadata,
                n_random=N_MATCHED_RANDOM_SETS,
                seed=(
                    RANDOM_SEED
                    + initial_factors * 1000
                    + module_index * 100
                    + sum(ord(char) for char in analysis_set)
                ),
            )

            for subspace_name in KEY_SUBSPACES:
                factors = [
                    factor
                    for factor in subspaces.get(subspace_name, [])
                    if factor in weights.columns
                ]
                if not factors:
                    continue

                loading_matrix = weights[factors].to_numpy(dtype=float)
                control = capture_random_control(
                    observed_vector=vector,
                    loadings=loading_matrix,
                    random_vectors=random_vectors,
                )

                capture_rows.append(
                    {
                        "analysis_set": analysis_set,
                        "initial_factors": initial_factors,
                        "module_label": module,
                        "n_module_genes": n_module_genes,
                        "module_coverage_fraction": (
                            n_module_genes
                            / int(
                                strict_weights[
                                    strict_weights["module_label"].eq(module)
                                ]["human_gene_symbol"]
                                .drop_duplicates()
                                .shape[0]
                            )
                        ),
                        "subspace_name": subspace_name,
                        "n_subspace_factors": len(factors),
                        "subspace_factors": ";".join(factors),
                        **control,
                    }
                )

                random_rows.append(
                    {
                        "analysis_set": analysis_set,
                        "initial_factors": initial_factors,
                        "module_label": module,
                        "subspace_name": subspace_name,
                        "n_random_valid": control["n_random_valid"],
                        "random_mean_capture": control["random_mean"],
                        "random_q95_capture": control["random_q95"],
                        "observed_capture": control["observed_capture"],
                        "observed_percentile": control["observed_percentile"],
                        "empirical_p_greater_equal": control[
                            "empirical_p_greater_equal"
                        ],
                    }
                )

            pca_value, pca_components = pca_capture(
                pca_loadings_path=pca_path,
                module_vector=vector,
            )
            pca_rows.append(
                {
                    "analysis_set": analysis_set,
                    "initial_factors": initial_factors,
                    "module_label": module,
                    "n_module_genes": n_module_genes,
                    "pca_components": pca_components,
                    "stacked_pca_subspace_capture": pca_value,
                }
            )

    factor_alignment = pd.DataFrame(factor_alignment_rows)
    capture = pd.DataFrame(capture_rows)
    random_controls = pd.DataFrame(random_rows)
    pca_capture_table = pd.DataFrame(pca_rows)
    factor_patterns = pd.DataFrame(factor_pattern_rows)

    capture["empirical_q_within_model_subspace"] = np.nan
    for keys, index in capture.groupby(
        ["analysis_set", "initial_factors", "subspace_name"]
    ).groups.items():
        capture.loc[
            index,
            "empirical_q_within_model_subspace",
        ] = bh_adjust(
            capture.loc[index, "empirical_p_greater_equal"]
        )
    capture["empirical_q_global"] = bh_adjust(
        capture["empirical_p_greater_equal"]
    )

    random_controls = random_controls.merge(
        capture[
            [
                "analysis_set",
                "initial_factors",
                "module_label",
                "subspace_name",
                "empirical_q_within_model_subspace",
                "empirical_q_global",
            ]
        ],
        on=[
            "analysis_set",
            "initial_factors",
            "module_label",
            "subspace_name",
        ],
        how="left",
    )

    rank_summary = (
        capture.groupby(
            ["analysis_set", "module_label", "subspace_name"],
            as_index=False,
        )
        .agg(
            n_ranks=("initial_factors", "nunique"),
            median_capture=("observed_capture", "median"),
            minimum_capture=("observed_capture", "min"),
            maximum_capture=("observed_capture", "max"),
            median_random_percentile=("observed_percentile", "median"),
            maximum_empirical_q_global=("empirical_q_global", "max"),
            fraction_ranks_global_q_lt_0_05=(
                "empirical_q_global",
                lambda values: float(
                    (pd.to_numeric(values, errors="coerce") < 0.05).mean()
                ),
            ),
            median_module_coverage_fraction=(
                "module_coverage_fraction",
                "median",
            ),
        )
    )

    cross_set_rows = []
    for initial_factors in sorted(capture["initial_factors"].unique()):
        for module in PRIMARY_MODULES:
            for subspace_name in KEY_SUBSPACES:
                part = capture[
                    capture["initial_factors"].eq(initial_factors)
                    & capture["module_label"].eq(module)
                    & capture["subspace_name"].eq(subspace_name)
                ].copy()

                if part.shape[0] < 2:
                    continue

                indexed = part.set_index("analysis_set")["observed_capture"]
                for set_a, set_b in combinations(indexed.index.tolist(), 2):
                    cross_set_rows.append(
                        {
                            "initial_factors": initial_factors,
                            "module_label": module,
                            "subspace_name": subspace_name,
                            "analysis_set_a": set_a,
                            "analysis_set_b": set_b,
                            "capture_a": float(indexed.loc[set_a]),
                            "capture_b": float(indexed.loc[set_b]),
                            "capture_difference_a_minus_b": float(
                                indexed.loc[set_a] - indexed.loc[set_b]
                            ),
                        }
                    )

    cross_set = pd.DataFrame(cross_set_rows)

    mofa_all = capture[
        capture["subspace_name"].eq("shared_two_or_more")
    ][
        [
            "analysis_set",
            "initial_factors",
            "module_label",
            "observed_capture",
        ]
    ].rename(
        columns={"observed_capture": "mofa_shared_subspace_capture"}
    )
    mofa_vs_pca = mofa_all.merge(
        pca_capture_table,
        on=["analysis_set", "initial_factors", "module_label"],
        how="outer",
    )
    mofa_vs_pca["mofa_minus_pca_capture"] = (
        mofa_vs_pca["mofa_shared_subspace_capture"]
        - mofa_vs_pca["stacked_pca_subspace_capture"]
    )

    interpretation = build_interpretation(rank_summary)

    factor_alignment.to_csv(OUTPUT_FACTOR_ALIGNMENT, index=False)
    capture.to_csv(OUTPUT_SUBSPACE_CAPTURE, index=False)
    random_controls.to_csv(OUTPUT_CAPTURE_RANDOM, index=False)
    rank_summary.to_csv(OUTPUT_RANK_SUMMARY, index=False)
    cross_set.to_csv(OUTPUT_CROSS_SET, index=False)
    pca_capture_table.to_csv(OUTPUT_PCA_CAPTURE, index=False)
    mofa_vs_pca.to_csv(OUTPUT_MOFA_VS_PCA, index=False)
    factor_patterns.to_csv(OUTPUT_FACTOR_PATTERNS, index=False)
    interpretation.to_csv(OUTPUT_INTERPRETATION, index=False)

    figure_paths = []
    for analysis_set in [
        "four_cohort_core_plus_frozen",
        "four_cohort_detection_aware",
        "three_cohort_no_ffpe",
    ]:
        figure_paths.extend(
            plot_rank_capture_heatmap(
                rank_summary=rank_summary,
                analysis_set=analysis_set,
                subspace_name="ubiquitous_all_groups",
            )
        )
        if analysis_set != "three_cohort_no_ffpe":
            figure_paths.extend(
                plot_rank_capture_heatmap(
                    rank_summary=rank_summary,
                    analysis_set=analysis_set,
                    subspace_name="shared_non_ffpe_only",
                )
            )

    write_readme()

    input_paths = [
        MODEL_INDEX_FILE,
        FACTOR_ACTIVITY_FILE,
        MODEL_SUMMARY_FILE,
        MOFA_MANIFEST_FILE,
        STRICT_WEIGHTS_FILE,
        MSFA_CONFIG_FILE,
    ]
    output_paths = [
        OUTPUT_FACTOR_ALIGNMENT,
        OUTPUT_SUBSPACE_CAPTURE,
        OUTPUT_CAPTURE_RANDOM,
        OUTPUT_RANK_SUMMARY,
        OUTPUT_CROSS_SET,
        OUTPUT_PCA_CAPTURE,
        OUTPUT_MOFA_VS_PCA,
        OUTPUT_FACTOR_PATTERNS,
        OUTPUT_INTERPRETATION,
        OUTPUT_README,
        *figure_paths,
    ]

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "outcome_loaded": False,
        "mofa_manifest_sha256": sha256_file(MOFA_MANIFEST_FILE),
        "matched_random_sets_per_test": N_MATCHED_RANDOM_SETS,
        "primary_activity_threshold_percent": (
            PRIMARY_ACTIVITY_THRESHOLD_PERCENT
        ),
        "guardrails": [
            "No clinical endpoint or outcome label was loaded.",
            "MOFA models were read only after the unsupervised freeze manifest existed.",
            "Frozen module weights and risk orientation were not changed.",
            "No rank was selected using an outcome.",
            "The feature space is frozen-program enriched; this is a targeted representation audit, not independent rediscovery.",
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

    print("")
    print("=" * 80)
    print("MOFA factor activity patterns")
    print("=" * 80)
    pattern_summary = (
        factor_patterns.groupby(
            ["analysis_set", "initial_factors", "activity_class"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n_factors"})
    )
    print(pattern_summary.to_string(index=False))

    print("")
    print("=" * 80)
    print("Top frozen-module factor alignments")
    print("=" * 80)
    top_factor = (
        factor_alignment.sort_values(
            [
                "analysis_set",
                "initial_factors",
                "module_label",
                "absolute_cosine",
            ],
            ascending=[True, True, True, False],
        )
        .groupby(
            ["analysis_set", "initial_factors", "module_label"],
            as_index=False,
        )
        .head(1)
    )
    print(
        top_factor[
            [
                "analysis_set",
                "initial_factors",
                "module_label",
                "n_module_genes",
                "factor",
                "signed_cosine",
                "absolute_cosine",
                "active_groups",
                "activity_class",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Frozen-module MOFA subspace capture")
    print("=" * 80)
    key_capture = capture[
        capture["subspace_name"].isin(
            [
                "ubiquitous_all_groups",
                "shared_non_ffpe_only",
                "gse39055_shared",
            ]
        )
    ].copy()
    print(
        key_capture[
            [
                "analysis_set",
                "initial_factors",
                "module_label",
                "subspace_name",
                "n_subspace_factors",
                "module_coverage_fraction",
                "observed_capture",
                "observed_percentile",
                "empirical_q_within_model_subspace",
                "empirical_q_global",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Rank-stable frozen-module capture summary")
    print("=" * 80)
    print(
        rank_summary[
            rank_summary["subspace_name"].isin(
                [
                    "shared_two_or_more",
                    "ubiquitous_all_groups",
                    "shared_non_ffpe_only",
                ]
            )
        ][
            [
                "analysis_set",
                "module_label",
                "subspace_name",
                "n_ranks",
                "median_capture",
                "minimum_capture",
                "maximum_capture",
                "median_random_percentile",
                "fraction_ranks_global_q_lt_0_05",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("MOFA versus stacked-PCA module capture")
    print("=" * 80)
    print(
        mofa_vs_pca[
            [
                "analysis_set",
                "initial_factors",
                "module_label",
                "mofa_shared_subspace_capture",
                "stacked_pca_subspace_capture",
                "mofa_minus_pca_capture",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Outcome-blind module representation interpretation")
    print("=" * 80)
    print(interpretation.to_string(index=False))

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("No clinical outcome was loaded.")
    print("Subspace capture is rotation invariant and summarized across all predefined ranks.")
    print("Matched random controls are variability matched within each fitted feature space.")
    print("The current factor panels were enriched with frozen module genes and are not independent module rediscovery.")
    print("Detection-aware M24 capture is not interpreted when fewer than three genes are represented.")

    print("")
    print("Saved:")
    for path in [
        OUTPUT_FACTOR_ALIGNMENT,
        OUTPUT_SUBSPACE_CAPTURE,
        OUTPUT_CAPTURE_RANDOM,
        OUTPUT_RANK_SUMMARY,
        OUTPUT_CROSS_SET,
        OUTPUT_PCA_CAPTURE,
        OUTPUT_MOFA_VS_PCA,
        OUTPUT_FACTOR_PATTERNS,
        OUTPUT_INTERPRETATION,
        OUTPUT_README,
        OUTPUT_MANIFEST,
    ]:
        print(path)
    print(f"Figures directory: {FIGURES_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
