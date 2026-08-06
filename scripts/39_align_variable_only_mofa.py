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

SCRIPT_VERSION = "39-align-variable-only-mofa-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "multigroup_mofa_variable_only_alignment"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_INDEX_FILE = (
    RESULTS_DIR / "multigroup_mofa_variable_only_model_index.csv"
)
FACTOR_ACTIVITY_FILE = (
    RESULTS_DIR / "multigroup_mofa_variable_only_factor_activity.csv"
)
MODEL_SUMMARY_FILE = (
    RESULTS_DIR / "multigroup_mofa_variable_only_model_summary.csv"
)
MOFA_MANIFEST_FILE = (
    RESULTS_DIR
    / "multigroup_mofa_variable_only_unsupervised_freeze_manifest.json"
)
STRICT_WEIGHTS_FILE = (
    RESULTS_DIR / "GSE238110_frozen_transfer_gene_weights_strict.csv"
)
VARIABLE_ONLY_CONFIG_FILE = (
    RESULTS_DIR / "multistudy_factor_variable_only_model_config.json"
)
VARIABLE_ONLY_COVERAGE_FILE = (
    RESULTS_DIR / "multistudy_factor_variable_only_module_coverage.csv"
)
TARGETED_RANK_SUMMARY_FILE = (
    RESULTS_DIR / "multigroup_mofa_module_capture_rank_summary.csv"
)

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]
PRIMARY_ACTIVITY_THRESHOLD_PERCENT = 1.0

N_MATCHED_RANDOM_SETS = 2000
N_VARIABILITY_BINS = 10
MIN_INTERPRETABLE_GENES = 5
MIN_INTERPRETABLE_COVERAGE = 0.20
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
    RESULTS_DIR
    / "multigroup_mofa_variable_only_frozen_module_factor_alignment.csv"
)
OUTPUT_FACTOR_MAX_RANDOM = (
    RESULTS_DIR
    / "multigroup_mofa_variable_only_factor_max_alignment_random_controls.csv"
)
OUTPUT_SUBSPACE_CAPTURE = (
    RESULTS_DIR
    / "multigroup_mofa_variable_only_frozen_module_subspace_capture.csv"
)
OUTPUT_SUBSPACE_RANDOM = (
    RESULTS_DIR
    / "multigroup_mofa_variable_only_subspace_random_controls.csv"
)
OUTPUT_RANK_SUMMARY = (
    RESULTS_DIR
    / "multigroup_mofa_variable_only_module_capture_rank_summary.csv"
)
OUTPUT_FACTOR_SUMMARY = (
    RESULTS_DIR
    / "multigroup_mofa_variable_only_module_factor_rank_summary.csv"
)
OUTPUT_CROSS_SET = (
    RESULTS_DIR
    / "multigroup_mofa_variable_only_module_cross_set_sensitivity.csv"
)
OUTPUT_TARGETED_COMPARISON = (
    RESULTS_DIR
    / "multigroup_mofa_targeted_vs_variable_only_capture.csv"
)
OUTPUT_INTERPRETATION = (
    RESULTS_DIR
    / "multigroup_mofa_variable_only_representation_interpretation.csv"
)
OUTPUT_README = (
    RESULTS_DIR / "multigroup_mofa_variable_only_alignment_README.txt"
)
OUTPUT_MANIFEST = (
    RESULTS_DIR / "multigroup_mofa_variable_only_alignment_manifest.json"
)


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


def read_gene_metadata(
    config: dict[str, Any],
    analysis_set: str,
) -> pd.DataFrame:
    path = Path(config["analysis_sets"][analysis_set]["gene_metadata"])
    metadata = read_required_csv(path)
    metadata["human_gene_symbol"] = (
        metadata["human_gene_symbol"].astype(str).str.upper()
    )
    return metadata.drop_duplicates("human_gene_symbol", keep="first")


def variability_metric_name(analysis_set: str) -> str:
    if analysis_set.startswith("three_cohort_no_ffpe"):
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
    n_valid = int(values.notna().sum())
    q = min(N_VARIABILITY_BINS, max(1, n_valid))

    result["variability_bin"] = pd.qcut(
        values.rank(method="average"),
        q=q,
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
    result = result.drop_duplicates("human_gene_symbol", keep="first")

    result = (
        result.set_index("human_gene_symbol", drop=False)
        .reindex(genes)
    )
    result["human_gene_symbol"] = pd.Index(genes, dtype="object")
    return result.reset_index(drop=True)


def frozen_module_vector(
    weights: pd.DataFrame,
    module: str,
    genes: list[str],
) -> tuple[np.ndarray, int, int, list[str]]:
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
    return vector, len(present), part.shape[0], present


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

    return {
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


def matched_random_vectors(
    module_vector: np.ndarray,
    genes: list[str],
    metadata: pd.DataFrame,
    excluded_genes: set[str],
    n_random: int,
    seed: int,
) -> np.ndarray:
    module_indices = np.where(module_vector != 0)[0]
    n_module_genes = len(module_indices)

    if n_module_genes < MIN_INTERPRETABLE_GENES:
        return np.empty((0, len(genes)), dtype=np.float32)

    metadata_aligned = align_metadata_to_genes(metadata, genes)
    bins = metadata_aligned["variability_bin"].to_numpy()

    module_bins = bins[module_indices]
    loading_values = module_vector[module_indices].copy()

    candidate_indices = np.asarray(
        [
            index
            for index, gene in enumerate(genes)
            if gene not in excluded_genes
        ],
        dtype=int,
    )

    rng = np.random.default_rng(seed)
    random_vectors = np.zeros(
        (n_random, len(genes)),
        dtype=np.float32,
    )

    for repeat in range(n_random):
        chosen: list[int] = []
        used: set[int] = set()

        for target_bin in module_bins:
            exact_pool = np.asarray(
                [
                    index
                    for index in candidate_indices
                    if (
                        index not in used
                        and np.isfinite(bins[index])
                        and np.isfinite(target_bin)
                        and bins[index] == target_bin
                    )
                ],
                dtype=int,
            )

            if exact_pool.size == 0:
                available = np.asarray(
                    [
                        index
                        for index in candidate_indices
                        if index not in used and np.isfinite(bins[index])
                    ],
                    dtype=int,
                )

                if available.size == 0:
                    chosen = []
                    break

                if np.isfinite(target_bin):
                    distances = np.abs(
                        bins[available].astype(float) - float(target_bin)
                    )
                    exact_pool = available[
                        distances == np.nanmin(distances)
                    ]
                else:
                    exact_pool = available

            selected = int(rng.choice(exact_pool))
            chosen.append(selected)
            used.add(selected)

        if len(chosen) != n_module_genes:
            continue

        random_vectors[
            repeat,
            chosen,
        ] = rng.permutation(loading_values).astype(np.float32)

    valid = np.linalg.norm(random_vectors, axis=1) > 0
    return random_vectors[valid]


def max_factor_alignment_random_control(
    observed_vector: np.ndarray,
    factor_loadings: np.ndarray,
    random_vectors: np.ndarray,
) -> dict[str, float]:
    loading_norms = np.linalg.norm(factor_loadings, axis=0)
    loading_norms[loading_norms == 0] = np.nan

    observed_norm = np.linalg.norm(observed_vector)
    observed_cosines = (
        observed_vector @ factor_loadings
    ) / (observed_norm * loading_norms)
    observed_max = float(np.nanmax(np.abs(observed_cosines)))

    if random_vectors.shape[0] == 0:
        return {
            "observed_max_absolute_cosine": observed_max,
            "n_random_valid": 0,
            "random_mean_max_absolute_cosine": np.nan,
            "random_q95_max_absolute_cosine": np.nan,
            "observed_percentile": np.nan,
            "empirical_p_greater_equal": np.nan,
        }

    random_norms = np.linalg.norm(random_vectors, axis=1)
    cosine_matrix = (
        random_vectors @ factor_loadings
    ) / (
        random_norms[:, None] * loading_norms[None, :]
    )
    random_max = np.nanmax(np.abs(cosine_matrix), axis=1)
    random_max = random_max[np.isfinite(random_max)]

    empirical_p = (
        1.0 + np.sum(random_max >= observed_max)
    ) / (random_max.size + 1.0)

    return {
        "observed_max_absolute_cosine": observed_max,
        "n_random_valid": int(random_max.size),
        "random_mean_max_absolute_cosine": float(random_max.mean()),
        "random_q95_max_absolute_cosine": float(
            np.quantile(random_max, 0.95)
        ),
        "observed_percentile": float(np.mean(random_max <= observed_max)),
        "empirical_p_greater_equal": float(empirical_p),
    }


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
            "random_mean_capture": np.nan,
            "random_q95_capture": np.nan,
            "observed_percentile": np.nan,
            "empirical_p_greater_equal": np.nan,
        }

    basis = orthonormal_basis(loadings)
    denominators = np.sum(random_vectors**2, axis=1)
    projected = random_vectors @ basis
    captures = np.sum(projected**2, axis=1) / denominators
    captures = captures[np.isfinite(captures)]

    empirical_p = (
        1.0 + np.sum(captures >= observed)
    ) / (captures.size + 1.0)

    return {
        "observed_capture": observed,
        "n_random_valid": int(captures.size),
        "random_mean_capture": float(captures.mean()),
        "random_q95_capture": float(np.quantile(captures, 0.95)),
        "observed_percentile": float(np.mean(captures <= observed)),
        "empirical_p_greater_equal": float(empirical_p),
    }


def lookup_summary(
    table: pd.DataFrame,
    analysis_set: str,
    module: str,
    subspace: str,
    column: str,
) -> float:
    part = table[
        table["analysis_set"].eq(analysis_set)
        & table["module_label"].eq(module)
        & table["subspace_name"].eq(subspace)
    ]

    if part.empty:
        return np.nan

    return float(part.iloc[0][column])


def build_interpretation(
    rank_summary: pd.DataFrame,
    factor_summary: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    coverage_index = coverage.set_index(
        ["analysis_set", "module_label"]
    )

    core_set = "four_cohort_variable_only_1500"
    detection_set = (
        "four_cohort_detection_aware_variable_only_700"
    )
    no_ffpe_set = "three_cohort_no_ffpe_variable_only_1500"

    rows = []

    for module in PRIMARY_MODULES:
        core_coverage = (
            float(
                coverage_index.loc[
                    (core_set, module),
                    "natural_coverage_fraction",
                ]
            )
            if (core_set, module) in coverage_index.index
            else np.nan
        )
        detection_coverage = (
            float(
                coverage_index.loc[
                    (detection_set, module),
                    "natural_coverage_fraction",
                ]
            )
            if (detection_set, module) in coverage_index.index
            else np.nan
        )
        no_ffpe_coverage = (
            float(
                coverage_index.loc[
                    (no_ffpe_set, module),
                    "natural_coverage_fraction",
                ]
            )
            if (no_ffpe_set, module) in coverage_index.index
            else np.nan
        )

        core_ubiquitous = lookup_summary(
            rank_summary,
            core_set,
            module,
            "ubiquitous_all_groups",
            "median_capture",
        )
        core_ubiquitous_q_fraction = lookup_summary(
            rank_summary,
            core_set,
            module,
            "ubiquitous_all_groups",
            "fraction_ranks_global_q_lt_0_05",
        )
        core_non_ffpe = lookup_summary(
            rank_summary,
            core_set,
            module,
            "shared_non_ffpe_only",
            "median_capture",
        )
        core_non_ffpe_q_fraction = lookup_summary(
            rank_summary,
            core_set,
            module,
            "shared_non_ffpe_only",
            "fraction_ranks_global_q_lt_0_05",
        )
        detection_ubiquitous = lookup_summary(
            rank_summary,
            detection_set,
            module,
            "ubiquitous_all_groups",
            "median_capture",
        )
        no_ffpe_ubiquitous = lookup_summary(
            rank_summary,
            no_ffpe_set,
            module,
            "ubiquitous_all_groups",
            "median_capture",
        )

        factor_part = factor_summary[
            factor_summary["module_label"].eq(module)
        ]
        median_max_cosine = (
            float(
                factor_part[
                    factor_part["analysis_set"].eq(core_set)
                ]["median_max_absolute_cosine"].iloc[0]
            )
            if not factor_part[
                factor_part["analysis_set"].eq(core_set)
            ].empty
            else np.nan
        )

        interpretable_core = (
            np.isfinite(core_coverage)
            and core_coverage >= MIN_INTERPRETABLE_COVERAGE
            and coverage_index.loc[
                (core_set, module),
                "n_naturally_selected_module_genes",
            ] >= MIN_INTERPRETABLE_GENES
        )

        if not interpretable_core:
            representation_class = "not_interpretable_low_natural_coverage"
        elif (
            core_coverage >= 0.50
            and np.isfinite(core_ubiquitous)
            and core_ubiquitous >= 0.30
            and core_ubiquitous_q_fraction == 1.0
            and np.isfinite(no_ffpe_ubiquitous)
            and no_ffpe_ubiquitous >= 0.30
        ):
            representation_class = (
                "independent_ubiquitous_latent_recurrence"
            )
        elif (
            core_coverage >= 0.40
            and np.isfinite(core_non_ffpe)
            and core_non_ffpe >= 0.30
            and core_non_ffpe_q_fraction >= (2 / 3)
            and np.isfinite(no_ffpe_ubiquitous)
            and no_ffpe_ubiquitous >= 0.30
            and (
                not np.isfinite(core_ubiquitous)
                or core_ubiquitous < 0.20
            )
        ):
            if (
                np.isfinite(detection_ubiquitous)
                and detection_ubiquitous >= 0.20
            ):
                representation_class = (
                    "independent_non_ffpe_recurrence_with_"
                    "detection_aware_restoration"
                )
            else:
                representation_class = (
                    "independent_non_ffpe_latent_recurrence"
                )
        else:
            representation_class = (
                "limited_or_no_independent_latent_recurrence"
            )

        rows.append(
            {
                "module_label": module,
                "core_natural_coverage_fraction": core_coverage,
                "detection_aware_natural_coverage_fraction": detection_coverage,
                "no_ffpe_natural_coverage_fraction": no_ffpe_coverage,
                "core_ubiquitous_median_capture": core_ubiquitous,
                "core_non_ffpe_median_capture": core_non_ffpe,
                "detection_aware_ubiquitous_median_capture": (
                    detection_ubiquitous
                ),
                "no_ffpe_ubiquitous_median_capture": no_ffpe_ubiquitous,
                "core_median_max_absolute_factor_cosine": median_max_cosine,
                "variable_only_representation_class": representation_class,
                "interpretation_note": (
                    "This classification uses no clinical outcome and "
                    "summarizes only naturally selected frozen genes."
                ),
            }
        )

    return pd.DataFrame(rows)


def targeted_comparison(
    variable_rank_summary: pd.DataFrame,
) -> pd.DataFrame:
    if not TARGETED_RANK_SUMMARY_FILE.exists():
        return pd.DataFrame()

    targeted = pd.read_csv(TARGETED_RANK_SUMMARY_FILE)

    mapping = {
        "four_cohort_variable_only_1500": (
            "four_cohort_core_plus_frozen"
        ),
        "four_cohort_detection_aware_variable_only_700": (
            "four_cohort_detection_aware"
        ),
        "three_cohort_no_ffpe_variable_only_1500": (
            "three_cohort_no_ffpe"
        ),
    }

    rows = []
    for variable_set, targeted_set in mapping.items():
        variable_part = variable_rank_summary[
            variable_rank_summary["analysis_set"].eq(variable_set)
        ]

        for row in variable_part.itertuples(index=False):
            targeted_part = targeted[
                targeted["analysis_set"].eq(targeted_set)
                & targeted["module_label"].eq(row.module_label)
                & targeted["subspace_name"].eq(row.subspace_name)
            ]

            if targeted_part.empty:
                continue

            targeted_capture = float(
                targeted_part.iloc[0]["median_capture"]
            )
            variable_capture = float(row.median_capture)

            rows.append(
                {
                    "variable_only_analysis_set": variable_set,
                    "targeted_analysis_set": targeted_set,
                    "module_label": row.module_label,
                    "subspace_name": row.subspace_name,
                    "variable_only_median_capture": variable_capture,
                    "targeted_median_capture": targeted_capture,
                    "variable_minus_targeted_capture": (
                        variable_capture - targeted_capture
                    ),
                    "variable_to_targeted_capture_ratio": (
                        variable_capture / targeted_capture
                        if targeted_capture > 0
                        else np.nan
                    ),
                    "comparison_guardrail": (
                        "Descriptive only because feature spaces and natural "
                        "module coverage differ."
                    ),
                }
            )

    return pd.DataFrame(rows)


def plot_summary(
    interpretation: pd.DataFrame,
) -> list[Path]:
    columns = [
        "core_ubiquitous_median_capture",
        "core_non_ffpe_median_capture",
        "detection_aware_ubiquitous_median_capture",
        "no_ffpe_ubiquitous_median_capture",
    ]
    labels = [
        "Core ubiquitous",
        "Core non-FFPE",
        "Detection-aware ubiquitous",
        "No-FFPE ubiquitous",
    ]

    matrix = (
        interpretation.set_index("module_label")
        .reindex(PRIMARY_MODULES)[columns]
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    image = ax.imshow(
        matrix.to_numpy(dtype=float),
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(PRIMARY_MODULES)))
    ax.set_yticklabels(PRIMARY_MODULES)
    ax.set_title(
        "Variable-only MOFA: frozen-module latent-subspace capture"
    )

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix.iloc[row_index, column_index]
            label = "NA" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()

    png_path = FIGURES_DIR / "variable_only_mofa_module_capture.png"
    pdf_path = FIGURES_DIR / "variable_only_mofa_module_capture.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return [png_path, pdf_path]


def write_readme() -> None:
    text = f"""Variable-only frozen-module alignment to multi-group MOFA
Script version: {SCRIPT_VERSION}

Purpose
-------
Test whether frozen canine programs recur in an outcome-blind factor space in
which frozen-module membership was not used for feature selection.

The factor models were frozen by script 38 before this alignment was performed.

Primary tests
-------------
- Maximum absolute cosine with any retained factor, assessed against the
  maximum-over-factors null from variability-matched random gene sets.
- Rotation-invariant capture by shared, ubiquitous, non-FFPE-shared, and
  GSE39055-associated factor subspaces.
- Stability across all prespecified initial factor ranks.
- Sensitivity to GSE39055 exclusion and Detection-P-aware feature filtering.

Random controls
---------------
Random panels are matched approximately on cross-study variability. All genes
belonging to any primary frozen module are excluded from the random candidate
pool. Frozen loading magnitudes and signs are permuted across matched genes.

Interpretability rule
---------------------
A module is interpreted only when at least {MIN_INTERPRETABLE_GENES} naturally
selected genes and at least {MIN_INTERPRETABLE_COVERAGE:.0%} of the frozen module
are present. This intentionally excludes low-coverage M24 analyses.

Important distinction
---------------------
This is stronger than the targeted MOFA audit because no frozen genes were
forced into the factor-analysis feature space. Alignment is nevertheless a
post-fit test, not a new independent patient cohort or outcome validation.

No clinical endpoint or outcome label is loaded.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("Align frozen programs to variable-only multi-group MOFA")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Design:")
    print("  Use MOFA models fitted without forced frozen-module genes.")
    print("  Align naturally selected frozen genes after the unsupervised freeze.")
    print("  Correct factor-selection bias using maximum-over-factor random nulls.")
    print("  Test rotation-invariant shared-subspace capture.")
    print("  Load no clinical outcome.")
    print("")

    model_index = read_required_csv(MODEL_INDEX_FILE)
    activity_all = read_required_csv(FACTOR_ACTIVITY_FILE)
    model_summary = read_required_csv(MODEL_SUMMARY_FILE)
    strict_weights = read_required_csv(STRICT_WEIGHTS_FILE)
    coverage = read_required_csv(VARIABLE_ONLY_COVERAGE_FILE)

    if not MOFA_MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Missing variable-only MOFA manifest: {MOFA_MANIFEST_FILE}"
        )
    if not VARIABLE_ONLY_CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Missing variable-only config: {VARIABLE_ONLY_CONFIG_FILE}"
        )

    mofa_manifest = json.loads(
        MOFA_MANIFEST_FILE.read_text(encoding="utf-8")
    )
    config = json.loads(
        VARIABLE_ONLY_CONFIG_FILE.read_text(encoding="utf-8")
    )

    if bool(mofa_manifest.get("outcome_loaded", True)):
        raise RuntimeError(
            "Variable-only MOFA manifest does not confirm outcome_loaded=false."
        )
    if bool(
        mofa_manifest.get("selection_used_frozen_membership", True)
    ):
        raise RuntimeError(
            "Variable-only MOFA manifest does not confirm "
            "selection_used_frozen_membership=false."
        )
    if bool(config.get("selection_used_frozen_membership", True)):
        raise RuntimeError(
            "Variable-only config does not confirm "
            "selection_used_frozen_membership=false."
        )

    strict_weights["human_gene_symbol"] = (
        strict_weights["human_gene_symbol"].astype(str).str.upper()
    )
    all_primary_frozen_genes = set(
        strict_weights[
            strict_weights["module_label"].isin(PRIMARY_MODULES)
        ]["human_gene_symbol"]
    )

    factor_alignment_rows: list[dict[str, Any]] = []
    factor_max_rows: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []

    for model_row in model_index.itertuples(index=False):
        analysis_set = str(model_row.analysis_set)
        initial_factors = int(model_row.initial_factors)

        weights = read_required_csv(
            Path(model_row.weights_path),
            index_col=0,
        )
        weights.index = weights.index.astype(str).str.upper()
        weights.columns = weights.columns.astype(str)

        activity = read_required_csv(Path(model_row.activity_path))
        activity["factor"] = activity["factor"].astype(str)

        groups = list(config["analysis_sets"][analysis_set]["cohorts"])
        metadata = add_variability_bins(
            read_gene_metadata(config, analysis_set),
            analysis_set,
        )
        genes = weights.index.tolist()
        metadata = align_metadata_to_genes(metadata, genes)

        subspaces = factor_subspaces(
            weights=weights,
            activity=activity,
            groups=groups,
        )

        primary_activity = activity[
            activity["activity_threshold_percent"].eq(
                PRIMARY_ACTIVITY_THRESHOLD_PERCENT
            )
        ].copy()

        for module_index, module in enumerate(PRIMARY_MODULES):
            (
                vector,
                n_selected_module_genes,
                n_frozen_genes,
                selected_module_genes,
            ) = frozen_module_vector(
                strict_weights,
                module,
                genes,
            )

            coverage_fraction = (
                n_selected_module_genes / n_frozen_genes
                if n_frozen_genes
                else np.nan
            )
            interpretable = (
                n_selected_module_genes >= MIN_INTERPRETABLE_GENES
                and coverage_fraction >= MIN_INTERPRETABLE_COVERAGE
            )

            for factor in weights.columns:
                factor_vector = weights[factor].to_numpy(dtype=float)
                cosine = cosine_similarity(vector, factor_vector)
                activity_row = primary_activity[
                    primary_activity["factor"].eq(str(factor))
                ]

                factor_alignment_rows.append(
                    {
                        "analysis_set": analysis_set,
                        "initial_factors": initial_factors,
                        "module_label": module,
                        "n_frozen_genes": n_frozen_genes,
                        "n_naturally_selected_module_genes": (
                            n_selected_module_genes
                        ),
                        "natural_coverage_fraction": coverage_fraction,
                        "interpretable": interpretable,
                        "factor": str(factor),
                        "signed_cosine": cosine,
                        "absolute_cosine": (
                            abs(cosine)
                            if np.isfinite(cosine)
                            else np.nan
                        ),
                        "squared_cosine": (
                            cosine**2
                            if np.isfinite(cosine)
                            else np.nan
                        ),
                        "active_group_count": (
                            int(
                                activity_row.iloc[0][
                                    "active_group_count"
                                ]
                            )
                            if not activity_row.empty
                            else np.nan
                        ),
                        "active_groups": (
                            str(activity_row.iloc[0]["active_groups"])
                            if not activity_row.empty
                            else ""
                        ),
                        "activity_class": (
                            str(activity_row.iloc[0]["activity_class"])
                            if not activity_row.empty
                            else ""
                        ),
                    }
                )

            if not interpretable:
                continue

            random_vectors = matched_random_vectors(
                module_vector=vector,
                genes=genes,
                metadata=metadata,
                excluded_genes=all_primary_frozen_genes,
                n_random=N_MATCHED_RANDOM_SETS,
                seed=(
                    RANDOM_SEED
                    + initial_factors * 1000
                    + module_index * 100
                    + sum(ord(char) for char in analysis_set)
                ),
            )

            factor_matrix = weights.to_numpy(dtype=float)
            factor_control = max_factor_alignment_random_control(
                observed_vector=vector,
                factor_loadings=factor_matrix,
                random_vectors=random_vectors,
            )
            factor_max_rows.append(
                {
                    "analysis_set": analysis_set,
                    "initial_factors": initial_factors,
                    "module_label": module,
                    "n_frozen_genes": n_frozen_genes,
                    "n_naturally_selected_module_genes": (
                        n_selected_module_genes
                    ),
                    "natural_coverage_fraction": coverage_fraction,
                    **factor_control,
                }
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

                subspace_rows.append(
                    {
                        "analysis_set": analysis_set,
                        "initial_factors": initial_factors,
                        "module_label": module,
                        "n_frozen_genes": n_frozen_genes,
                        "n_naturally_selected_module_genes": (
                            n_selected_module_genes
                        ),
                        "natural_coverage_fraction": coverage_fraction,
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
                        "random_mean_capture": control[
                            "random_mean_capture"
                        ],
                        "random_q95_capture": control[
                            "random_q95_capture"
                        ],
                        "observed_capture": control["observed_capture"],
                        "observed_percentile": control[
                            "observed_percentile"
                        ],
                        "empirical_p_greater_equal": control[
                            "empirical_p_greater_equal"
                        ],
                    }
                )

    factor_alignment = pd.DataFrame(factor_alignment_rows)
    factor_max = pd.DataFrame(factor_max_rows)
    subspace_capture = pd.DataFrame(subspace_rows)
    subspace_random = pd.DataFrame(random_rows)

    factor_max["empirical_q_global"] = bh_adjust(
        factor_max["empirical_p_greater_equal"]
    )
    factor_max["empirical_q_within_analysis_set"] = np.nan

    for analysis_set, index in factor_max.groupby(
        "analysis_set"
    ).groups.items():
        factor_max.loc[
            index,
            "empirical_q_within_analysis_set",
        ] = bh_adjust(
            factor_max.loc[index, "empirical_p_greater_equal"]
        )

    subspace_capture["empirical_q_global"] = bh_adjust(
        subspace_capture["empirical_p_greater_equal"]
    )
    subspace_capture["empirical_q_within_subspace"] = np.nan

    for keys, index in subspace_capture.groupby(
        ["analysis_set", "subspace_name"]
    ).groups.items():
        subspace_capture.loc[
            index,
            "empirical_q_within_subspace",
        ] = bh_adjust(
            subspace_capture.loc[
                index,
                "empirical_p_greater_equal",
            ]
        )

    subspace_random = subspace_random.merge(
        subspace_capture[
            [
                "analysis_set",
                "initial_factors",
                "module_label",
                "subspace_name",
                "empirical_q_global",
                "empirical_q_within_subspace",
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
        subspace_capture.groupby(
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
                    (
                        pd.to_numeric(values, errors="coerce") < 0.05
                    ).mean()
                ),
            ),
            median_n_naturally_selected_module_genes=(
                "n_naturally_selected_module_genes",
                "median",
            ),
            median_natural_coverage_fraction=(
                "natural_coverage_fraction",
                "median",
            ),
        )
    )

    factor_summary = (
        factor_max.groupby(
            ["analysis_set", "module_label"],
            as_index=False,
        )
        .agg(
            n_ranks=("initial_factors", "nunique"),
            median_max_absolute_cosine=(
                "observed_max_absolute_cosine",
                "median",
            ),
            minimum_max_absolute_cosine=(
                "observed_max_absolute_cosine",
                "min",
            ),
            maximum_max_absolute_cosine=(
                "observed_max_absolute_cosine",
                "max",
            ),
            median_random_percentile=("observed_percentile", "median"),
            fraction_ranks_global_q_lt_0_05=(
                "empirical_q_global",
                lambda values: float(
                    (
                        pd.to_numeric(values, errors="coerce") < 0.05
                    ).mean()
                ),
            ),
        )
    )

    cross_set_rows = []
    for initial_factors in sorted(
        subspace_capture["initial_factors"].unique()
    ):
        for module in PRIMARY_MODULES:
            for subspace_name in KEY_SUBSPACES:
                part = subspace_capture[
                    subspace_capture["initial_factors"].eq(
                        initial_factors
                    )
                    & subspace_capture["module_label"].eq(module)
                    & subspace_capture["subspace_name"].eq(
                        subspace_name
                    )
                ]

                if part.shape[0] < 2:
                    continue

                indexed = part.set_index("analysis_set")[
                    "observed_capture"
                ]

                for set_a, set_b in combinations(
                    indexed.index.tolist(),
                    2,
                ):
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
    targeted_vs_variable = targeted_comparison(rank_summary)
    interpretation = build_interpretation(
        rank_summary=rank_summary,
        factor_summary=factor_summary,
        coverage=coverage,
    )

    factor_alignment.to_csv(OUTPUT_FACTOR_ALIGNMENT, index=False)
    factor_max.to_csv(OUTPUT_FACTOR_MAX_RANDOM, index=False)
    subspace_capture.to_csv(OUTPUT_SUBSPACE_CAPTURE, index=False)
    subspace_random.to_csv(OUTPUT_SUBSPACE_RANDOM, index=False)
    rank_summary.to_csv(OUTPUT_RANK_SUMMARY, index=False)
    factor_summary.to_csv(OUTPUT_FACTOR_SUMMARY, index=False)
    cross_set.to_csv(OUTPUT_CROSS_SET, index=False)
    targeted_vs_variable.to_csv(
        OUTPUT_TARGETED_COMPARISON,
        index=False,
    )
    interpretation.to_csv(OUTPUT_INTERPRETATION, index=False)

    figure_paths = plot_summary(interpretation)
    write_readme()

    input_paths = [
        MODEL_INDEX_FILE,
        FACTOR_ACTIVITY_FILE,
        MODEL_SUMMARY_FILE,
        MOFA_MANIFEST_FILE,
        STRICT_WEIGHTS_FILE,
        VARIABLE_ONLY_CONFIG_FILE,
        VARIABLE_ONLY_COVERAGE_FILE,
    ]
    if TARGETED_RANK_SUMMARY_FILE.exists():
        input_paths.append(TARGETED_RANK_SUMMARY_FILE)

    output_paths = [
        OUTPUT_FACTOR_ALIGNMENT,
        OUTPUT_FACTOR_MAX_RANDOM,
        OUTPUT_SUBSPACE_CAPTURE,
        OUTPUT_SUBSPACE_RANDOM,
        OUTPUT_RANK_SUMMARY,
        OUTPUT_FACTOR_SUMMARY,
        OUTPUT_CROSS_SET,
        OUTPUT_TARGETED_COMPARISON,
        OUTPUT_INTERPRETATION,
        OUTPUT_README,
        *figure_paths,
    ]

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "outcome_loaded": False,
        "selection_used_frozen_membership": False,
        "mofa_manifest_sha256": sha256_file(MOFA_MANIFEST_FILE),
        "matched_random_sets_per_test": N_MATCHED_RANDOM_SETS,
        "minimum_interpretable_genes": MIN_INTERPRETABLE_GENES,
        "minimum_interpretable_coverage": (
            MIN_INTERPRETABLE_COVERAGE
        ),
        "guardrails": [
            "No clinical endpoint or outcome label was loaded.",
            "Variable-only MOFA models were frozen before alignment.",
            "Frozen module membership was not used for feature selection.",
            "All primary frozen-module genes were excluded from the random candidate pool.",
            "Maximum-over-factor nulls correct selection of the best-aligned factor.",
            "Low natural-coverage modules are not interpreted.",
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
    print("Top naturally recurring factor alignments")
    print("=" * 80)
    top_factor = (
        factor_alignment[
            factor_alignment["interpretable"].astype(bool)
        ]
        .sort_values(
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
                "n_naturally_selected_module_genes",
                "natural_coverage_fraction",
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
    print("Maximum-factor alignment against matched random panels")
    print("=" * 80)
    print(
        factor_max[
            [
                "analysis_set",
                "initial_factors",
                "module_label",
                "natural_coverage_fraction",
                "observed_max_absolute_cosine",
                "random_q95_max_absolute_cosine",
                "observed_percentile",
                "empirical_p_greater_equal",
                "empirical_q_global",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Variable-only frozen-module subspace capture")
    print("=" * 80)
    key_capture = subspace_capture[
        subspace_capture["subspace_name"].isin(
            [
                "ubiquitous_all_groups",
                "shared_non_ffpe_only",
                "gse39055_shared",
            ]
        )
    ]
    print(
        key_capture[
            [
                "analysis_set",
                "initial_factors",
                "module_label",
                "subspace_name",
                "n_subspace_factors",
                "natural_coverage_fraction",
                "observed_capture",
                "observed_percentile",
                "empirical_q_within_subspace",
                "empirical_q_global",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Rank-stable variable-only module recurrence")
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
                "median_natural_coverage_fraction",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Independent latent-recurrence interpretation")
    print("=" * 80)
    print(interpretation.to_string(index=False))

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("No clinical outcome was loaded.")
    print("Frozen module membership was not used for feature selection.")
    print("Random panels exclude all primary frozen-module genes.")
    print("Best-factor significance uses a maximum-over-factors null.")
    print("M24 is not interpreted when fewer than five genes are naturally selected.")
    print("The targeted versus variable-only capture comparison is descriptive because feature spaces differ.")

    print("")
    print("Saved:")
    for path in [
        OUTPUT_FACTOR_ALIGNMENT,
        OUTPUT_FACTOR_MAX_RANDOM,
        OUTPUT_SUBSPACE_CAPTURE,
        OUTPUT_SUBSPACE_RANDOM,
        OUTPUT_RANK_SUMMARY,
        OUTPUT_FACTOR_SUMMARY,
        OUTPUT_CROSS_SET,
        OUTPUT_TARGETED_COMPARISON,
        OUTPUT_INTERPRETATION,
        OUTPUT_README,
        OUTPUT_MANIFEST,
    ]:
        print(path)
    print(f"Figures directory: {FIGURES_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
