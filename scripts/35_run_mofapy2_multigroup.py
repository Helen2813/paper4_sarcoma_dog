from __future__ import annotations

from itertools import combinations
from pathlib import Path
import hashlib
import importlib
import json
import platform
import sys
from datetime import datetime, timezone
from typing import Any

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

SCRIPT_VERSION = "35-mofapy2-multigroup-factor-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
FIT_ROOT = RESULTS_DIR / "multigroup_mofa_fits"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIT_ROOT.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = RESULTS_DIR / "multistudy_factor_model_config.json"
INPUT_MANIFEST_FILE = RESULTS_DIR / "multistudy_factor_input_manifest.json"

# Multi-group MOFA uses one RNA view and one group per cohort.
# ARD on factors lets factors be active in all, some, or one cohort.
INITIAL_FACTOR_GRIDS = [8, 12, 16]
SEED_BY_GRID = {8: 1103, 12: 2207, 16: 3301}

MAX_ITER = 1000
CONVERGENCE_MODE = "medium"
DROP_R2_PERCENT = 0.1
USE_FLOAT32 = True

# MOFA/mofax report R2 in percentage points.
ACTIVITY_THRESHOLDS_PERCENT = [0.5, 1.0, 2.0]
PRIMARY_ACTIVITY_THRESHOLD_PERCENT = 1.0

OUTPUT_MODEL_INDEX = RESULTS_DIR / "multigroup_mofa_model_index.csv"
OUTPUT_FACTOR_ACTIVITY = RESULTS_DIR / "multigroup_mofa_factor_activity.csv"
OUTPUT_MODEL_SUMMARY = RESULTS_DIR / "multigroup_mofa_model_summary.csv"
OUTPUT_R2 = RESULTS_DIR / "multigroup_mofa_variance_explained.csv"
OUTPUT_RANK_SENSITIVITY = RESULTS_DIR / "multigroup_mofa_rank_sensitivity.csv"
OUTPUT_CROSS_SET_SENSITIVITY = RESULTS_DIR / "multigroup_mofa_cross_set_sensitivity.csv"
OUTPUT_PCA_BASELINE = RESULTS_DIR / "multigroup_mofa_stacked_pca_baseline.csv"
OUTPUT_MOFA_VS_PCA = RESULTS_DIR / "multigroup_mofa_vs_pca.csv"
OUTPUT_README = RESULTS_DIR / "multigroup_mofa_README.txt"
OUTPUT_MANIFEST = RESULTS_DIR / "multigroup_mofa_unsupervised_freeze_manifest.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def require_python_packages() -> tuple[Any, Any]:
    missing = []
    for package_name in ["mofapy2", "mofax"]:
        if importlib.util.find_spec(package_name) is None:
            missing.append(package_name)

    if missing:
        command = (
            f'"{sys.executable}" -m pip install '
            '"mofapy2==0.7.3" mofax'
        )
        raise ImportError(
            "Missing Python package(s): "
            + ", ".join(missing)
            + "\nInstall them inside the active venv with:\n"
            + command
        )

    from mofapy2.run.entry_point import entry_point
    import mofax as mfx

    return entry_point, mfx


def read_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Matrix file not found: {path}")

    matrix = pd.read_csv(path, index_col=0)
    matrix = matrix.apply(pd.to_numeric, errors="coerce")

    if matrix.isna().any().any():
        missing_count = int(matrix.isna().sum().sum())
        raise RuntimeError(
            f"Non-numeric or missing matrix values in {path}: "
            f"{missing_count}"
        )
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise RuntimeError(f"Non-finite values in matrix: {path}")

    return matrix


def ensure_aligned_features(
    matrices: dict[str, pd.DataFrame],
) -> list[str]:
    first_cohort = next(iter(matrices))
    reference = matrices[first_cohort].columns.tolist()

    for cohort, matrix in matrices.items():
        if matrix.columns.tolist() != reference:
            raise RuntimeError(
                f"Gene columns are not identically aligned for {cohort}."
            )
    return reference


def normalize_r2_table(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    normalized = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in result.columns
    }

    required = {}
    for canonical, candidates in {
        "factor": ["factor"],
        "view": ["view"],
        "group": ["group"],
        "r2": ["r2", "variance_explained"],
    }.items():
        source = next(
            (
                normalized[candidate]
                for candidate in candidates
                if candidate in normalized
            ),
            None,
        )
        if source is None:
            raise ValueError(
                f"Could not find {canonical} in MOFA R2 table. "
                f"Columns: {list(result.columns)}"
            )
        required[source] = canonical

    result = result.rename(columns=required)
    result = result[["factor", "view", "group", "r2"]].copy()
    result["factor"] = result["factor"].astype(str)
    result["view"] = result["view"].astype(str)
    result["group"] = result["group"].astype(str)
    result["r2"] = pd.to_numeric(result["r2"], errors="coerce")
    return result


def orthonormal_basis(loadings: np.ndarray) -> np.ndarray:
    matrix = np.asarray(loadings, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return np.empty((matrix.shape[0], 0), dtype=float)

    q, r = np.linalg.qr(matrix)
    diagonal = np.abs(np.diag(r))
    if diagonal.size == 0:
        return np.empty((matrix.shape[0], 0), dtype=float)

    tolerance = max(matrix.shape) * np.finfo(float).eps * diagonal.max()
    rank = int(np.sum(diagonal > tolerance))
    return q[:, :rank]


def subspace_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    q_first = orthonormal_basis(first)
    q_second = orthonormal_basis(second)

    if q_first.shape[1] == 0 or q_second.shape[1] == 0:
        return np.nan

    singular_values = np.linalg.svd(
        q_first.T @ q_second,
        compute_uv=False,
    )
    return float(np.mean(np.clip(singular_values, 0.0, 1.0) ** 2))


def rv_coefficient(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)

    numerator = float(np.sum(first * second))
    denominator = float(
        np.sqrt(np.sum(first * first) * np.sum(second * second))
    )
    if denominator <= 0 or not np.isfinite(denominator):
        return np.nan
    return numerator / denominator


def read_training_elbo(model_path: Path) -> dict[str, float]:
    result = {
        "elbo_final": np.nan,
        "elbo_max": np.nan,
        "n_elbo_records": 0,
    }

    try:
        with h5py.File(model_path, "r") as handle:
            candidates = [
                "training_stats/elbo",
                "training_stats/ELBO",
                "training_stats/lower_bound",
            ]
            values = None
            for candidate in candidates:
                if candidate in handle:
                    values = np.asarray(handle[candidate][...], dtype=float)
                    break

            if values is not None:
                values = values[np.isfinite(values)]
                if values.size:
                    result["elbo_final"] = float(values[-1])
                    result["elbo_max"] = float(np.max(values))
                    result["n_elbo_records"] = int(values.size)
    except Exception:
        pass

    return result


def classify_factor_activity(
    r2: pd.DataFrame,
    groups: list[str],
    threshold: float,
) -> pd.DataFrame:
    pivot = (
        r2.pivot_table(
            index="factor",
            columns="group",
            values="r2",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(columns=groups, fill_value=0.0)
    )

    rows = []
    n_groups = len(groups)

    for factor, row in pivot.iterrows():
        active = [
            group
            for group in groups
            if float(row[group]) >= threshold
        ]
        active_count = len(active)

        if active_count == n_groups:
            activity_class = "ubiquitous_shared"
        elif active_count >= 2:
            activity_class = "partially_shared"
        elif active_count == 1:
            activity_class = "group_specific"
        else:
            activity_class = "inactive"

        values = np.asarray(
            [float(row[group]) for group in groups],
            dtype=float,
        )

        rows.append(
            {
                "factor": str(factor),
                "activity_threshold_percent": threshold,
                "active_group_count": active_count,
                "active_groups": ";".join(active),
                "activity_class": activity_class,
                "minimum_group_r2": float(values.min()),
                "maximum_group_r2": float(values.max()),
                "mean_group_r2": float(values.mean()),
                "median_group_r2": float(np.median(values)),
                "r2_range": float(values.max() - values.min()),
                **{
                    f"r2_{group}": float(row[group])
                    for group in groups
                },
            }
        )

    return pd.DataFrame(rows)


def stacked_pca_baseline(
    matrices: dict[str, pd.DataFrame],
    n_components: int,
    analysis_set: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stacked = pd.concat(matrices.values(), axis=0)
    actual_components = min(
        n_components,
        stacked.shape[0] - 1,
        stacked.shape[1],
    )

    model = PCA(
        n_components=actual_components,
        svd_solver="full",
        random_state=SEED_BY_GRID.get(n_components, 42),
    )
    model.fit(stacked.to_numpy(dtype=float))

    diagnostic_rows = []
    for cohort, matrix in matrices.items():
        values = matrix.to_numpy(dtype=float)
        scores = model.transform(values)
        reconstruction = model.inverse_transform(scores)

        denominator = float(np.sum(values**2))
        r2 = (
            1.0 - float(np.sum((values - reconstruction) ** 2)) / denominator
            if denominator > 0
            else np.nan
        )

        diagnostic_rows.append(
            {
                "analysis_set": analysis_set,
                "initial_factors": n_components,
                "cohort": cohort,
                "pca_components": actual_components,
                "stacked_pca_r2_percent": 100.0 * r2,
            }
        )

    loadings = pd.DataFrame(
        model.components_.T,
        index=stacked.columns,
        columns=[
            f"PC{index + 1}"
            for index in range(actual_components)
        ],
    )
    return pd.DataFrame(diagnostic_rows), loadings


def train_one_model(
    entry_point_class: Any,
    mfx: Any,
    analysis_set: str,
    matrices: dict[str, pd.DataFrame],
    n_factors: int,
    output_dir: Path,
) -> dict[str, Any]:
    groups = list(matrices.keys())
    genes = ensure_aligned_features(matrices)

    nested_data = [
        [
            matrices[group].to_numpy(
                dtype=np.float32 if USE_FLOAT32 else np.float64
            )
            for group in groups
        ]
    ]
    sample_names = [
        [
            f"{group}::{sample}"
            for sample in matrices[group].index.astype(str)
        ]
        for group in groups
    ]

    model_path = output_dir / f"mofa_k{n_factors}.hdf5"
    weights_path = output_dir / f"mofa_k{n_factors}_weights.csv"
    factors_path = output_dir / f"mofa_k{n_factors}_factors.csv"
    r2_path = output_dir / f"mofa_k{n_factors}_r2.csv"
    activity_path = output_dir / f"mofa_k{n_factors}_factor_activity.csv"

    ent = entry_point_class()
    ent.set_data_options(
        scale_views=False,
        scale_groups=False,
        center_groups=True,
        use_float32=USE_FLOAT32,
    )
    ent.set_data_matrix(
        nested_data,
        likelihoods=["gaussian"],
        views_names=["RNA"],
        groups_names=groups,
        samples_names=sample_names,
        features_names=[genes],
    )
    ent.set_model_options(
        factors=n_factors,
        spikeslab_factors=False,
        spikeslab_weights=True,
        ard_factors=True,
        ard_weights=True,
    )
    ent.set_train_options(
        iter=MAX_ITER,
        convergence_mode=CONVERGENCE_MODE,
        dropR2=DROP_R2_PERCENT / 100.0,
        verbose=False,
        quiet=True,
        seed=SEED_BY_GRID[n_factors],
        gpu_mode=False,
        outfile=str(model_path),
        save_interrupted=True,
    )
    ent.build()
    ent.run()
    ent.save(
        outfile=str(model_path),
        save_data=True,
        save_parameters=False,
    )

    model = mfx.mofa_model(str(model_path))
    try:
        weights = model.get_weights(
            views="RNA",
            df=True,
            scale=False,
            absolute_values=False,
        )
        factors = model.get_factors(
            df=True,
            concatenate_groups=True,
            scale=False,
            absolute_values=False,
        )
        samples = model.get_samples()
        r2 = normalize_r2_table(model.get_r2())

        if not isinstance(weights, pd.DataFrame):
            weights = pd.DataFrame(weights, index=genes)
        if not isinstance(factors, pd.DataFrame):
            factors = pd.DataFrame(factors)

        weights.index = weights.index.astype(str)
        weights.to_csv(weights_path)

        factors.index = factors.index.astype(str)
        if isinstance(samples, pd.DataFrame) and not samples.empty:
            samples_copy = samples.copy()
            sample_column = next(
                (
                    column
                    for column in ["sample", "cell"]
                    if column in samples_copy.columns
                ),
                None,
            )
            if sample_column is not None:
                samples_copy = samples_copy.set_index(sample_column)
                factors = factors.join(
                    samples_copy[["group"]],
                    how="left",
                )
        factors.to_csv(factors_path)
        r2.to_csv(r2_path, index=False)

        activity_tables = []
        for threshold in ACTIVITY_THRESHOLDS_PERCENT:
            activity = classify_factor_activity(
                r2=r2,
                groups=groups,
                threshold=threshold,
            )
            activity.insert(0, "initial_factors", n_factors)
            activity.insert(0, "analysis_set", analysis_set)
            activity_tables.append(activity)

        activity_all = pd.concat(
            activity_tables,
            ignore_index=True,
        )
        activity_all.to_csv(activity_path, index=False)

        primary_activity = activity_all[
            activity_all["activity_threshold_percent"].eq(
                PRIMARY_ACTIVITY_THRESHOLD_PERCENT
            )
        ].copy()

        group_total_r2 = (
            r2.groupby("group", as_index=False)["r2"]
            .sum()
            .rename(columns={"r2": "summed_factor_r2_percent"})
        )

        summary = {
            "analysis_set": analysis_set,
            "initial_factors": n_factors,
            "seed": SEED_BY_GRID[n_factors],
            "n_samples_total": int(
                sum(matrix.shape[0] for matrix in matrices.values())
            ),
            "n_genes": len(genes),
            "n_groups": len(groups),
            "n_retained_factors": int(weights.shape[1]),
            "n_ubiquitous_shared": int(
                primary_activity["activity_class"]
                .eq("ubiquitous_shared")
                .sum()
            ),
            "n_partially_shared": int(
                primary_activity["activity_class"]
                .eq("partially_shared")
                .sum()
            ),
            "n_group_specific": int(
                primary_activity["activity_class"]
                .eq("group_specific")
                .sum()
            ),
            "n_inactive": int(
                primary_activity["activity_class"]
                .eq("inactive")
                .sum()
            ),
            "mean_total_r2_percent": float(
                group_total_r2["summed_factor_r2_percent"].mean()
            ),
            "minimum_total_r2_percent": float(
                group_total_r2["summed_factor_r2_percent"].min()
            ),
            "maximum_total_r2_percent": float(
                group_total_r2["summed_factor_r2_percent"].max()
            ),
            "model_path": str(model_path),
            "weights_path": str(weights_path),
            "factors_path": str(factors_path),
            "r2_path": str(r2_path),
            "activity_path": str(activity_path),
            **read_training_elbo(model_path),
        }

        return {
            "summary": summary,
            "weights": weights,
            "r2": r2,
            "activity": activity_all,
            "group_total_r2": group_total_r2,
            "paths": [
                model_path,
                weights_path,
                factors_path,
                r2_path,
                activity_path,
            ],
        }
    finally:
        model.close()


def shared_factor_columns(
    activity: pd.DataFrame,
    minimum_active_groups: int,
) -> list[str]:
    primary = activity[
        activity["activity_threshold_percent"].eq(
            PRIMARY_ACTIVITY_THRESHOLD_PERCENT
        )
    ]
    return (
        primary[
            primary["active_group_count"].ge(minimum_active_groups)
        ]["factor"]
        .astype(str)
        .tolist()
    )


def compare_models_within_set(
    fit_records: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = []

    for first, second in combinations(fit_records, 2):
        first_weights = first["weights"]
        second_weights = second["weights"]
        common_genes = first_weights.index.intersection(
            second_weights.index
        )

        for label, minimum_active_groups in [
            ("shared_two_or_more_groups", 2),
            (
                "ubiquitous_all_groups",
                int(first["summary"]["n_groups"]),
            ),
        ]:
            first_factors = [
                factor
                for factor in shared_factor_columns(
                    first["activity"],
                    minimum_active_groups,
                )
                if factor in first_weights.columns
            ]
            second_factors = [
                factor
                for factor in shared_factor_columns(
                    second["activity"],
                    minimum_active_groups,
                )
                if factor in second_weights.columns
            ]

            similarity = np.nan
            covariance_rv = np.nan
            if first_factors and second_factors:
                first_matrix = first_weights.loc[
                    common_genes,
                    first_factors,
                ].to_numpy(dtype=float)
                second_matrix = second_weights.loc[
                    common_genes,
                    second_factors,
                ].to_numpy(dtype=float)

                similarity = subspace_similarity(
                    first_matrix,
                    second_matrix,
                )
                covariance_rv = rv_coefficient(
                    first_matrix @ first_matrix.T,
                    second_matrix @ second_matrix.T,
                )

            rows.append(
                {
                    "analysis_set": first["summary"]["analysis_set"],
                    "comparison_type": label,
                    "initial_factors_a": first["summary"]["initial_factors"],
                    "initial_factors_b": second["summary"]["initial_factors"],
                    "n_common_genes": len(common_genes),
                    "n_factors_a": len(first_factors),
                    "n_factors_b": len(second_factors),
                    "subspace_similarity": similarity,
                    "shared_covariance_rv": covariance_rv,
                }
            )

    return pd.DataFrame(rows)


def compare_models_across_sets(
    all_records: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = []

    by_k: dict[int, list[dict[str, Any]]] = {}
    for record in all_records:
        by_k.setdefault(
            int(record["summary"]["initial_factors"]),
            [],
        ).append(record)

    for n_factors, records in by_k.items():
        for first, second in combinations(records, 2):
            first_weights = first["weights"]
            second_weights = second["weights"]
            common_genes = first_weights.index.intersection(
                second_weights.index
            )

            first_shared = [
                factor
                for factor in shared_factor_columns(
                    first["activity"],
                    2,
                )
                if factor in first_weights.columns
            ]
            second_shared = [
                factor
                for factor in shared_factor_columns(
                    second["activity"],
                    2,
                )
                if factor in second_weights.columns
            ]

            similarity = np.nan
            covariance_rv = np.nan
            if first_shared and second_shared:
                first_matrix = first_weights.loc[
                    common_genes,
                    first_shared,
                ].to_numpy(dtype=float)
                second_matrix = second_weights.loc[
                    common_genes,
                    second_shared,
                ].to_numpy(dtype=float)

                similarity = subspace_similarity(
                    first_matrix,
                    second_matrix,
                )
                covariance_rv = rv_coefficient(
                    first_matrix @ first_matrix.T,
                    second_matrix @ second_matrix.T,
                )

            rows.append(
                {
                    "initial_factors": n_factors,
                    "analysis_set_a": first["summary"]["analysis_set"],
                    "analysis_set_b": second["summary"]["analysis_set"],
                    "n_common_genes": len(common_genes),
                    "n_shared_factors_a": len(first_shared),
                    "n_shared_factors_b": len(second_shared),
                    "shared_subspace_similarity": similarity,
                    "shared_covariance_rv": covariance_rv,
                }
            )

    return pd.DataFrame(rows)


def write_readme() -> None:
    text = f"""Python-native multi-group MOFA analysis
Script version: {SCRIPT_VERSION}

Method
------
This script uses the official Python packages mofapy2 and mofax.

It is not an exact implementation of the De Vito multi-study factor analysis
or the R-only VIMSFA package. Instead, it uses MOFA2 multi-group inference with
one RNA view and one group per cohort.

In multi-group MOFA, feature weights are common, while group-wise factor
activity is learned through factor scores and group-specific ARD. Factors can
therefore be active in all groups, in a subset of groups, or in one group only.

Input
-----
Outcome-blind rank-Gaussian matrices created by script 34.

Analysis sets
-------------
- four_cohort_core_plus_frozen
- four_cohort_detection_aware
- three_cohort_no_ffpe

Initial factor grids
--------------------
{INITIAL_FACTOR_GRIDS}

Factor activity
---------------
The primary descriptive threshold is at least
{PRIMARY_ACTIVITY_THRESHOLD_PERCENT:.1f}% variance explained within a group.
Sensitivity thresholds are {ACTIVITY_THRESHOLDS_PERCENT}%.

Model-selection guardrail
-------------------------
No rank is selected using an outcome. All ranks and analysis sets are retained.
Standard variational MOFA uses PCA-based initialization; one fixed seed is
recorded for each rank. Rank sensitivity is evaluated through rotation-invariant
weight-subspace comparisons.

Interpretation
--------------
Call this analysis "multi-group MOFA2" or "unsupervised multi-group factor
analysis" in the manuscript. Do not call it VIMSFA or exact MSFA.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("Python-native multi-group MOFA factor analysis")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Design:")
    print("  Use the official Python mofapy2 multi-group implementation.")
    print("  Treat each cohort as a group and RNA as one shared view.")
    print("  Infer factors that are ubiquitous, partially shared, or group-specific.")
    print("  Retain all predefined factor ranks without outcome-based selection.")
    print("  Compare with stacked PCA and test rank/set sensitivity.")
    print("")

    entry_point_class, mfx = require_python_packages()

    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Missing config: {CONFIG_FILE}\n"
            "Run scripts/34_prepare_multistudy_factor_inputs.py first."
        )

    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    analysis_sets = config["analysis_sets"]

    all_records: list[dict[str, Any]] = []
    model_index_rows = []
    model_summary_rows = []
    r2_tables = []
    activity_tables = []
    pca_tables = []
    generated_paths: list[Path] = []

    for analysis_set, set_config in analysis_sets.items():
        print("")
        print("=" * 80)
        print(f"Analysis set: {analysis_set}")
        print("=" * 80)

        groups = list(set_config["cohorts"])
        matrices = {
            group: read_matrix(Path(set_config["matrices"][group]))
            for group in groups
        }
        genes = ensure_aligned_features(matrices)

        output_dir = FIT_ROOT / analysis_set
        output_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"Cohorts: {', '.join(groups)} | "
            f"Genes: {len(genes)} | "
            f"Samples: "
            + ", ".join(
                f"{group}={matrices[group].shape[0]}"
                for group in groups
            )
        )

        set_records = []

        for n_factors in INITIAL_FACTOR_GRIDS:
            print(
                f"  Fitting MOFA: initial factors={n_factors}, "
                f"seed={SEED_BY_GRID[n_factors]}"
            )
            record = train_one_model(
                entry_point_class=entry_point_class,
                mfx=mfx,
                analysis_set=analysis_set,
                matrices=matrices,
                n_factors=n_factors,
                output_dir=output_dir,
            )
            set_records.append(record)
            all_records.append(record)
            model_summary_rows.append(record["summary"])
            generated_paths.extend(record["paths"])

            model_index_rows.append(
                {
                    "analysis_set": analysis_set,
                    "initial_factors": n_factors,
                    "model_path": record["summary"]["model_path"],
                    "weights_path": record["summary"]["weights_path"],
                    "factors_path": record["summary"]["factors_path"],
                    "r2_path": record["summary"]["r2_path"],
                    "activity_path": record["summary"]["activity_path"],
                }
            )

            r2 = record["r2"].copy()
            r2.insert(0, "initial_factors", n_factors)
            r2.insert(0, "analysis_set", analysis_set)
            r2_tables.append(r2)

            activity_tables.append(record["activity"])

            pca_diagnostics, pca_loadings = stacked_pca_baseline(
                matrices=matrices,
                n_components=n_factors,
                analysis_set=analysis_set,
            )
            pca_tables.append(pca_diagnostics)

            pca_path = output_dir / f"stacked_pca_k{n_factors}_loadings.csv"
            pca_loadings.to_csv(pca_path)
            generated_paths.append(pca_path)

        rank_sensitivity_set = compare_models_within_set(set_records)
        if not rank_sensitivity_set.empty:
            rank_path = output_dir / "rank_sensitivity.csv"
            rank_sensitivity_set.to_csv(rank_path, index=False)
            generated_paths.append(rank_path)

    model_index = pd.DataFrame(model_index_rows)
    model_summary = pd.DataFrame(model_summary_rows)
    r2_all = pd.concat(r2_tables, ignore_index=True)
    activity_all = pd.concat(activity_tables, ignore_index=True)
    pca_all = pd.concat(pca_tables, ignore_index=True)

    rank_sensitivity = pd.concat(
        [
            compare_models_within_set(
                [
                    record
                    for record in all_records
                    if record["summary"]["analysis_set"] == analysis_set
                ]
            )
            for analysis_set in analysis_sets
        ],
        ignore_index=True,
    )
    cross_set_sensitivity = compare_models_across_sets(all_records)

    mofa_total = (
        r2_all.groupby(
            ["analysis_set", "initial_factors", "group"],
            as_index=False,
        )["r2"]
        .sum()
        .rename(
            columns={
                "group": "cohort",
                "r2": "mofa_summed_factor_r2_percent",
            }
        )
    )
    mofa_vs_pca = mofa_total.merge(
        pca_all,
        on=["analysis_set", "initial_factors", "cohort"],
        how="outer",
    )
    mofa_vs_pca["mofa_minus_pca_r2_percent"] = (
        mofa_vs_pca["mofa_summed_factor_r2_percent"]
        - mofa_vs_pca["stacked_pca_r2_percent"]
    )

    model_index.to_csv(OUTPUT_MODEL_INDEX, index=False)
    model_summary.to_csv(OUTPUT_MODEL_SUMMARY, index=False)
    r2_all.to_csv(OUTPUT_R2, index=False)
    activity_all.to_csv(OUTPUT_FACTOR_ACTIVITY, index=False)
    rank_sensitivity.to_csv(OUTPUT_RANK_SENSITIVITY, index=False)
    cross_set_sensitivity.to_csv(
        OUTPUT_CROSS_SET_SENSITIVITY,
        index=False,
    )
    pca_all.to_csv(OUTPUT_PCA_BASELINE, index=False)
    mofa_vs_pca.to_csv(OUTPUT_MOFA_VS_PCA, index=False)

    write_readme()

    output_paths = [
        OUTPUT_MODEL_INDEX,
        OUTPUT_MODEL_SUMMARY,
        OUTPUT_R2,
        OUTPUT_FACTOR_ACTIVITY,
        OUTPUT_RANK_SENSITIVITY,
        OUTPUT_CROSS_SET_SENSITIVITY,
        OUTPUT_PCA_BASELINE,
        OUTPUT_MOFA_VS_PCA,
        OUTPUT_README,
        *generated_paths,
    ]

    package_versions = {}
    for package_name in ["mofapy2", "mofax", "numpy", "pandas", "sklearn"]:
        try:
            module = importlib.import_module(package_name)
            package_versions[package_name] = getattr(
                module,
                "__version__",
                "unknown",
            )
        except Exception:
            package_versions[package_name] = "unknown"

    input_paths = [
        Path(path)
        for set_config in analysis_sets.values()
        for path in set_config["matrices"].values()
    ]
    input_paths.append(CONFIG_FILE)
    if INPUT_MANIFEST_FILE.exists():
        input_paths.append(INPUT_MANIFEST_FILE)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "method": "MOFA2 multi-group, one RNA view",
        "exact_de_vito_msfa": False,
        "outcome_loaded": False,
        "initial_factor_grids": INITIAL_FACTOR_GRIDS,
        "seed_by_grid": SEED_BY_GRID,
        "activity_thresholds_percent": ACTIVITY_THRESHOLDS_PERCENT,
        "primary_activity_threshold_percent": (
            PRIMARY_ACTIVITY_THRESHOLD_PERCENT
        ),
        "training": {
            "max_iter": MAX_ITER,
            "convergence_mode": CONVERGENCE_MODE,
            "drop_r2_percent": DROP_R2_PERCENT,
            "use_float32": USE_FLOAT32,
        },
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions,
        "guardrails": [
            "No clinical endpoint or outcome label was loaded.",
            "All predefined ranks and analysis sets were retained.",
            "No rank was selected using downstream outcome association.",
            "The method must be described as multi-group MOFA2, not exact MSFA or VIMSFA.",
            "Sharedness is defined by group-specific factor activity, using variance explained.",
        ],
        "inputs": {
            str(path): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(set(input_paths))
            if path.exists()
        },
        "outputs": {
            str(path): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(set(output_paths))
            if path.exists()
        },
    }
    OUTPUT_MANIFEST.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("")
    print("=" * 80)
    print("Multi-group MOFA model summary")
    print("=" * 80)
    print(
        model_summary[
            [
                "analysis_set",
                "initial_factors",
                "n_retained_factors",
                "n_ubiquitous_shared",
                "n_partially_shared",
                "n_group_specific",
                "n_inactive",
                "mean_total_r2_percent",
                "minimum_total_r2_percent",
                "maximum_total_r2_percent",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Factor activity summary")
    print("=" * 80)
    primary_activity = activity_all[
        activity_all["activity_threshold_percent"].eq(
            PRIMARY_ACTIVITY_THRESHOLD_PERCENT
        )
    ]
    activity_summary = (
        primary_activity.groupby(
            ["analysis_set", "initial_factors", "activity_class"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n_factors"})
    )
    print(activity_summary.to_string(index=False))

    print("")
    print("=" * 80)
    print("Rank sensitivity")
    print("=" * 80)
    print(rank_sensitivity.to_string(index=False))

    print("")
    print("=" * 80)
    print("Cross-set sensitivity")
    print("=" * 80)
    print(cross_set_sensitivity.to_string(index=False))

    print("")
    print("=" * 80)
    print("MOFA versus stacked PCA")
    print("=" * 80)
    print(
        mofa_vs_pca[
            [
                "analysis_set",
                "initial_factors",
                "cohort",
                "mofa_summed_factor_r2_percent",
                "stacked_pca_r2_percent",
                "mofa_minus_pca_r2_percent",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("No outcome or clinical endpoint was loaded.")
    print("This is official Python multi-group MOFA2, not exact VIMSFA/MSFA.")
    print("Factors are labelled shared or specific from group-wise R2 activity.")
    print("All ranks are retained; no preferred rank is selected from outcomes.")
    print("The next script may align frozen module vectors with factor-weight subspaces only after this manifest is frozen.")

    print("")
    print("Saved:")
    for path in [
        OUTPUT_MODEL_INDEX,
        OUTPUT_MODEL_SUMMARY,
        OUTPUT_R2,
        OUTPUT_FACTOR_ACTIVITY,
        OUTPUT_RANK_SENSITIVITY,
        OUTPUT_CROSS_SET_SENSITIVITY,
        OUTPUT_PCA_BASELINE,
        OUTPUT_MOFA_VS_PCA,
        OUTPUT_README,
        OUTPUT_MANIFEST,
    ]:
        print(path)
    print(f"Model directory: {FIT_ROOT}")
    print("Done.")


if __name__ == "__main__":
    main()
