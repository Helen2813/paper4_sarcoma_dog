from __future__ import annotations

from itertools import combinations
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_VERSION = "28-conservative-module-preservation-audit-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
HUMAN_DIR = PROCESSED_DIR / "human_validation"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STRICT_WEIGHTS_FILE = (
    RESULTS_DIR / "GSE238110_frozen_transfer_gene_weights_strict.csv"
)
CANINE_EXPRESSION_FILE = (
    PROCESSED_DIR / "GSE238110_DOG2_expression_log2cpm_matched_allgenes.csv"
)
TARGET_EXPRESSION_FILE = HUMAN_DIR / "TARGET_OS_expression_log2_gene_symbol.csv"
GSE21257_EXPRESSION_FILE = HUMAN_DIR / "GSE21257_expression_gene_symbol.csv"
GSE39055_EXPRESSION_FILE = HUMAN_DIR / "GSE39055_expression_gene_symbol.csv"

SCRIPT27_PRESERVATION_FILE = (
    RESULTS_DIR / "cross_cohort_module_representation_preservation.csv"
)
SCRIPT27_SYNTHESIS_FILE = (
    RESULTS_DIR / "cross_cohort_module_preservation_outcome_synthesis.csv"
)

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]
HUMAN_COHORTS = ["TARGET_OS", "GSE21257", "GSE39055"]

N_EDGE_PERMUTATIONS = 5000
N_LOADING_PERMUTATIONS = 5000
N_SPLIT_REPEATS = 1000
RANDOM_SEED = 42

OUTPUT_EDGE = RESULTS_DIR / "cross_cohort_edge_preservation_audit.csv"
OUTPUT_SPLIT_HALF = RESULTS_DIR / "cross_cohort_disjoint_split_half_reliability.csv"
OUTPUT_CONSERVATIVE = (
    RESULTS_DIR / "cross_cohort_conservative_preservation_classification.csv"
)
OUTPUT_MANIFEST = (
    RESULTS_DIR / "cross_cohort_conservative_preservation_manifest.json"
)
OUTPUT_README = (
    RESULTS_DIR / "cross_cohort_conservative_preservation_README.txt"
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


def zscore_matrix(expression: pd.DataFrame) -> pd.DataFrame:
    x = expression.apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(axis=0))
    std = x.std(axis=0).replace(0, np.nan)
    z = (x - x.mean(axis=0)) / std
    return z.loc[:, z.notna().all(axis=0)]


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    if np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return np.nan
    return float(stats.spearmanr(a[mask], b[mask]).statistic)


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    if np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[indices]


def orient_pc1_to_frozen_score(
    z: pd.DataFrame,
    frozen_signs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    values = z.values.astype(float)
    _, singular_values, vt = np.linalg.svd(values, full_matrices=False)
    loading = vt[0].copy()
    score = values @ loading
    frozen_score = (values * frozen_signs).mean(axis=1)

    corr = safe_pearson(score, frozen_score)
    if np.isfinite(corr) and corr < 0:
        loading = -loading
        score = -score
        corr = -corr

    denominator = float(np.sum(singular_values ** 2))
    variance_explained = (
        float(singular_values[0] ** 2 / denominator)
        if denominator > 0
        else np.nan
    )
    return loading, score, variance_explained


def align_module_matrices(
    canine_expression: pd.DataFrame,
    human_expression: pd.DataFrame,
    weights: pd.DataFrame,
    module: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    part = weights[weights["module_label"].eq(module)].copy()
    part["canine_gene"] = part["canine_gene"].astype(str)
    part["human_gene_symbol"] = (
        part["human_gene_symbol"].astype(str).str.upper()
    )
    part = part.drop_duplicates("human_gene_symbol", keep="first")
    part = part.drop_duplicates("canine_gene", keep="first")

    canine_columns = {str(column): column for column in canine_expression.columns}

    human = human_expression.copy()
    human.columns = human.columns.astype(str).str.upper()
    human = human.loc[:, ~human.columns.duplicated()].copy()

    available = part[
        part["canine_gene"].isin(canine_columns)
        & part["human_gene_symbol"].isin(human.columns)
    ].copy()

    if available.shape[0] < 3:
        return (
            pd.DataFrame(index=canine_expression.index),
            pd.DataFrame(index=human.index),
            available,
        )

    canine = canine_expression[
        [canine_columns[gene] for gene in available["canine_gene"]]
    ].copy()
    canine.columns = available["human_gene_symbol"].tolist()

    human_matrix = human[
        available["human_gene_symbol"].tolist()
    ].copy()

    return canine, human_matrix, available


def correlation_edge_audit(
    canine_z: pd.DataFrame,
    human_z: pd.DataFrame,
    seed: int,
) -> dict[str, float]:
    canine_corr = np.corrcoef(canine_z.values, rowvar=False)
    human_corr = np.corrcoef(human_z.values, rowvar=False)

    canine_edges = upper_triangle(canine_corr)
    human_edges = upper_triangle(human_corr)

    observed_spearman = safe_spearman(canine_edges, human_edges)
    observed_pearson = safe_pearson(canine_edges, human_edges)

    valid = (
        np.isfinite(canine_edges)
        & np.isfinite(human_edges)
        & (canine_edges != 0)
        & (human_edges != 0)
    )
    sign_concordance = (
        float(
            np.mean(
                np.sign(canine_edges[valid])
                == np.sign(human_edges[valid])
            )
        )
        if valid.sum() > 0
        else np.nan
    )

    rng = np.random.default_rng(seed)
    null = np.empty(N_EDGE_PERMUTATIONS, dtype=float)
    n_genes = human_corr.shape[0]

    for index in range(N_EDGE_PERMUTATIONS):
        permutation = rng.permutation(n_genes)
        permuted_human = human_corr[np.ix_(permutation, permutation)]
        null[index] = safe_spearman(
            canine_edges,
            upper_triangle(permuted_human),
        )

    empirical_p_positive = (
        (1.0 + np.sum(null >= observed_spearman))
        / (N_EDGE_PERMUTATIONS + 1.0)
        if np.isfinite(observed_spearman)
        else np.nan
    )
    empirical_p_two_sided = (
        (
            1.0
            + np.sum(np.abs(null) >= abs(observed_spearman))
        )
        / (N_EDGE_PERMUTATIONS + 1.0)
        if np.isfinite(observed_spearman)
        else np.nan
    )

    return {
        "edge_spearman": observed_spearman,
        "edge_pearson": observed_pearson,
        "edge_sign_concordance": sign_concordance,
        "edge_permutation_p_positive": empirical_p_positive,
        "edge_permutation_p_two_sided": empirical_p_two_sided,
        "edge_null_mean": float(np.nanmean(null)),
        "edge_null_q95": float(np.nanquantile(null, 0.95)),
    }


def loading_audit(
    human_z: pd.DataFrame,
    frozen_loadings: np.ndarray,
    seed: int,
) -> dict[str, float]:
    frozen_signs = np.sign(frozen_loadings)
    frozen_signs[frozen_signs == 0] = 1.0

    human_loading, _, variance_explained = orient_pc1_to_frozen_score(
        human_z,
        frozen_signs,
    )

    observed_spearman = safe_spearman(
        human_loading,
        frozen_loadings,
    )
    observed_sign_concordance = float(
        np.mean(
            np.sign(human_loading)
            == np.sign(frozen_loadings)
        )
    )

    rng = np.random.default_rng(seed)
    null = np.empty(N_LOADING_PERMUTATIONS, dtype=float)

    for index in range(N_LOADING_PERMUTATIONS):
        permuted = rng.permutation(human_loading)
        null[index] = safe_spearman(
            permuted,
            frozen_loadings,
        )

    empirical_p_positive = (
        (1.0 + np.sum(null >= observed_spearman))
        / (N_LOADING_PERMUTATIONS + 1.0)
        if np.isfinite(observed_spearman)
        else np.nan
    )
    empirical_p_two_sided = (
        (
            1.0
            + np.sum(np.abs(null) >= abs(observed_spearman))
        )
        / (N_LOADING_PERMUTATIONS + 1.0)
        if np.isfinite(observed_spearman)
        else np.nan
    )

    return {
        "human_pc1_variance_explained": variance_explained,
        "loading_spearman": observed_spearman,
        "loading_sign_concordance": observed_sign_concordance,
        "loading_permutation_p_positive": empirical_p_positive,
        "loading_permutation_p_two_sided": empirical_p_two_sided,
        "loading_null_mean": float(np.nanmean(null)),
        "loading_null_q95": float(np.nanquantile(null, 0.95)),
    }


def unique_small_module_splits(n_genes: int) -> list[tuple[np.ndarray, np.ndarray]]:
    first_size = n_genes // 2
    all_indices = np.arange(n_genes)
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for first_tuple in combinations(range(n_genes), first_size):
        first = np.asarray(first_tuple, dtype=int)
        second = np.setdiff1d(all_indices, first)
        splits.append((first, second))

    return splits


def random_large_module_splits(
    n_genes: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    first_size = n_genes // 2
    all_indices = np.arange(n_genes)
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for _ in range(N_SPLIT_REPEATS):
        first = np.sort(
            rng.choice(
                all_indices,
                size=first_size,
                replace=False,
            )
        )
        second = np.setdiff1d(all_indices, first)
        splits.append((first, second))

    return splits


def disjoint_split_half_reliability(
    human_z: pd.DataFrame,
    frozen_loadings: np.ndarray,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    signs = np.sign(frozen_loadings)
    signs[signs == 0] = 1.0
    signed_values = human_z.values.astype(float) * signs

    n_genes = human_z.shape[1]
    if n_genes <= 12:
        splits = unique_small_module_splits(n_genes)
        method = "all_balanced_disjoint_splits"
    else:
        splits = random_large_module_splits(n_genes, seed)
        method = "random_balanced_disjoint_splits"

    rows: list[dict[str, Any]] = []

    for iteration, (first, second) in enumerate(splits, start=1):
        first_score = signed_values[:, first].mean(axis=1)
        second_score = signed_values[:, second].mean(axis=1)
        correlation = safe_pearson(first_score, second_score)

        rows.append(
            {
                "iteration": iteration,
                "method": method,
                "n_genes_first_half": len(first),
                "n_genes_second_half": len(second),
                "disjoint_half_score_correlation": correlation,
            }
        )

    table = pd.DataFrame(rows)
    values = pd.to_numeric(
        table["disjoint_half_score_correlation"],
        errors="coerce",
    ).dropna()

    summary = {
        "split_half_method": method,
        "n_split_half_valid": int(values.shape[0]),
        "split_half_mean": float(values.mean()),
        "split_half_median": float(values.median()),
        "split_half_q05": float(values.quantile(0.05)),
        "split_half_q95": float(values.quantile(0.95)),
        "split_half_fraction_positive": float((values > 0).mean()),
        "split_half_fraction_above_0_30": float((values > 0.30).mean()),
    }
    return summary, table


def conservative_classification(row: pd.Series) -> str:
    edge_q = row.get("edge_permutation_q_positive", np.nan)
    loading_q = row.get("loading_permutation_q_positive", np.nan)
    split_median = row.get("split_half_median", np.nan)
    split_q05 = row.get("split_half_q05", np.nan)
    edge_rho = row.get("edge_spearman", np.nan)
    loading_rho = row.get("loading_spearman", np.nan)

    edge_strong = (
        np.isfinite(edge_q)
        and edge_q < 0.05
        and np.isfinite(edge_rho)
        and edge_rho > 0
    )
    loading_strong = (
        np.isfinite(loading_q)
        and loading_q < 0.05
        and np.isfinite(loading_rho)
        and loading_rho > 0
    )
    internally_coherent = (
        np.isfinite(split_median)
        and split_median > 0.20
        and np.isfinite(split_q05)
        and split_q05 > -0.10
    )

    if edge_strong and loading_strong and internally_coherent:
        return "strong_cross_cohort_representation_preservation"

    evidence_count = int(edge_strong) + int(loading_strong) + int(
        internally_coherent
    )
    if evidence_count >= 2:
        return "partial_cross_cohort_representation_preservation"
    if evidence_count == 1:
        return "limited_cross_cohort_representation_evidence"
    return "no_clear_cross_cohort_representation_preservation"


def main() -> None:
    print("=" * 80)
    print("Conservative cross-cohort module-preservation audit")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Purpose:")
    print("  Replace mechanically inflated preservation metrics with direct tests.")
    print("  Compare canine and human within-module correlation matrices.")
    print("  Test PC1-loading concordance by gene-label permutation.")
    print("  Estimate reliability from non-overlapping gene halves.")
    print("  Preserve all frozen genes, weights, directions, and outcome results.")
    print("")

    weights = read_required_csv(STRICT_WEIGHTS_FILE)
    canine_expression = read_required_csv(
        CANINE_EXPRESSION_FILE,
        index_col=0,
    )
    human_expression = {
        "TARGET_OS": read_required_csv(
            TARGET_EXPRESSION_FILE,
            index_col=0,
        ),
        "GSE21257": read_required_csv(
            GSE21257_EXPRESSION_FILE,
            index_col=0,
        ),
        "GSE39055": read_required_csv(
            GSE39055_EXPRESSION_FILE,
            index_col=0,
        ),
    }

    old_preservation = (
        read_required_csv(SCRIPT27_PRESERVATION_FILE)
        if SCRIPT27_PRESERVATION_FILE.exists()
        else pd.DataFrame()
    )
    old_synthesis = (
        read_required_csv(SCRIPT27_SYNTHESIS_FILE)
        if SCRIPT27_SYNTHESIS_FILE.exists()
        else pd.DataFrame()
    )

    summary_rows: list[dict[str, Any]] = []
    split_tables: list[pd.DataFrame] = []

    job = 0
    total_jobs = len(HUMAN_COHORTS) * len(PRIMARY_MODULES)

    for cohort_index, cohort in enumerate(HUMAN_COHORTS):
        for module_index, module in enumerate(PRIMARY_MODULES):
            job += 1
            print(f"  Job {job}/{total_jobs}: {cohort} {module}")

            canine_matrix, human_matrix, mapping = align_module_matrices(
                canine_expression=canine_expression,
                human_expression=human_expression[cohort],
                weights=weights,
                module=module,
            )

            if mapping.shape[0] < 3:
                summary_rows.append(
                    {
                        "cohort": cohort,
                        "module_label": module,
                        "n_genes_shared": mapping.shape[0],
                        "error": "fewer_than_three_shared_genes",
                    }
                )
                continue

            canine_z = zscore_matrix(canine_matrix)
            human_z = zscore_matrix(human_matrix)

            shared = canine_z.columns.intersection(human_z.columns)
            canine_z = canine_z[shared]
            human_z = human_z[shared]
            mapping_indexed = (
                mapping.set_index("human_gene_symbol")
                .loc[shared]
            )
            frozen_loadings = pd.to_numeric(
                mapping_indexed["risk_oriented_loading"],
                errors="coerce",
            ).fillna(0.0).values

            seed = (
                RANDOM_SEED
                + cohort_index * 100000
                + module_index * 1000
            )

            edge = correlation_edge_audit(
                canine_z,
                human_z,
                seed=seed,
            )
            loading = loading_audit(
                human_z,
                frozen_loadings,
                seed=seed + 100,
            )
            split_summary, split_table = disjoint_split_half_reliability(
                human_z,
                frozen_loadings,
                seed=seed + 200,
            )

            split_table.insert(0, "module_label", module)
            split_table.insert(0, "cohort", cohort)
            split_tables.append(split_table)

            summary_rows.append(
                {
                    "cohort": cohort,
                    "module_label": module,
                    "n_canine_samples": canine_z.shape[0],
                    "n_human_samples": human_z.shape[0],
                    "n_genes_shared": len(shared),
                    **edge,
                    **loading,
                    **split_summary,
                    "error": "",
                }
            )

    audit = pd.DataFrame(summary_rows)
    audit["edge_permutation_q_positive"] = bh_adjust(
        audit["edge_permutation_p_positive"]
    )
    audit["loading_permutation_q_positive"] = bh_adjust(
        audit["loading_permutation_p_positive"]
    )
    audit["conservative_preservation_class"] = audit.apply(
        conservative_classification,
        axis=1,
    )

    if not old_synthesis.empty:
        old_columns = [
            "cohort",
            "module_label",
            "structural_support_class",
            "outcome_direction_concordant",
            "joint_preservation_outcome_class",
        ]
        old_columns = [
            column for column in old_columns if column in old_synthesis.columns
        ]
        audit = audit.merge(
            old_synthesis[old_columns],
            on=["cohort", "module_label"],
            how="left",
        )

    audit.to_csv(OUTPUT_EDGE, index=False)

    split_output = (
        pd.concat(split_tables, ignore_index=True)
        if split_tables
        else pd.DataFrame()
    )
    split_output.to_csv(OUTPUT_SPLIT_HALF, index=False)

    conservative_columns = [
        "cohort",
        "module_label",
        "n_genes_shared",
        "edge_spearman",
        "edge_permutation_p_positive",
        "edge_permutation_q_positive",
        "loading_spearman",
        "loading_permutation_p_positive",
        "loading_permutation_q_positive",
        "split_half_median",
        "split_half_q05",
        "split_half_fraction_positive",
        "conservative_preservation_class",
        "structural_support_class",
        "outcome_direction_concordant",
        "joint_preservation_outcome_class",
    ]
    conservative_columns = [
        column for column in conservative_columns if column in audit.columns
    ]
    conservative = audit[conservative_columns].copy()
    conservative.to_csv(OUTPUT_CONSERVATIVE, index=False)

    readme = f"""Conservative module-preservation audit
Script version: {SCRIPT_VERSION}

Why this audit was added
------------------------
The script 27 signed-mean/weighted-score correlation and overlapping-subset
reliability metrics can be high partly because the compared scores share most
or all genes. They are useful diagnostics but should not, by themselves,
define cross-cohort structural preservation.

Primary preservation evidence in this audit
-------------------------------------------
1. Spearman preservation of the full within-module gene-correlation matrix
   between canine DOG2 and each human cohort.
2. Concordance of human PC1 loadings with frozen canine risk-oriented loadings.
3. Correlation between scores formed from non-overlapping gene halves.

Permutation tests
-----------------
Gene-label permutations are used for correlation-matrix and loading
concordance. BH correction is applied across the 12 human cohort-module
comparisons for each test family.

Interpretation
--------------
Outcome association and representation preservation remain separate.
Frozen score direction is never reversed after viewing a human outcome.
"""
    OUTPUT_README.write_text(readme, encoding="utf-8")

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "frozen_weights_sha256": sha256_file(STRICT_WEIGHTS_FILE),
        "edge_permutations": N_EDGE_PERMUTATIONS,
        "loading_permutations": N_LOADING_PERMUTATIONS,
        "large_module_split_repeats": N_SPLIT_REPEATS,
        "files": {},
    }
    for path in [
        OUTPUT_EDGE,
        OUTPUT_SPLIT_HALF,
        OUTPUT_CONSERVATIVE,
        OUTPUT_README,
    ]:
        if path.exists():
            manifest["files"][path.name] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }

    OUTPUT_MANIFEST.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("")
    print("=" * 80)
    print("Conservative preservation classification")
    print("=" * 80)
    print(conservative.to_string(index=False))

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("Script 27 metrics remain descriptive and are not deleted.")
    print("High correlation between scores sharing the same genes is not treated as independent preservation evidence.")
    print("Non-overlapping split halves replace overlapping gene-subset reliability for the conservative classification.")
    print("Outcome-direction discordance is never repaired by flipping a frozen score.")
    print("Small seven-gene modules require cautious interpretation because edge and loading estimates are discrete and unstable.")

    print("")
    print("Saved:")
    for path in [
        OUTPUT_EDGE,
        OUTPUT_SPLIT_HALF,
        OUTPUT_CONSERVATIVE,
        OUTPUT_README,
        OUTPUT_MANIFEST,
    ]:
        print(path)
    print("Done.")


if __name__ == "__main__":
    main()
