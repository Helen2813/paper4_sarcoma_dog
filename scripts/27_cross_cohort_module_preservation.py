from __future__ import annotations

from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_VERSION = "27-cross-cohort-module-preservation-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
HUMAN_DIR = PROCESSED_DIR / "human_validation"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STRICT_WEIGHTS_FILE = RESULTS_DIR / "GSE238110_frozen_transfer_gene_weights_strict.csv"
FREEZE_JSON_FILE = RESULTS_DIR / "GSE238110_frozen_transfer_program_freeze.json"

CANINE_EXPRESSION_FILE = PROCESSED_DIR / "GSE238110_DOG2_expression_log2cpm_matched_allgenes.csv"
TARGET_EXPRESSION_FILE = HUMAN_DIR / "TARGET_OS_expression_log2_gene_symbol.csv"
GSE21257_EXPRESSION_FILE = HUMAN_DIR / "GSE21257_expression_gene_symbol.csv"
GSE39055_EXPRESSION_FILE = HUMAN_DIR / "GSE39055_expression_gene_symbol.csv"

TARGET_PRIMARY_FILE = RESULTS_DIR / "TARGET_OS_primary_frozen_program_validation.csv"
GSE21257_ROBUST_FILE = RESULTS_DIR / "GSE21257_primary_robust_logistic_effects.csv"
GSE39055_PRIMARY_FILE = RESULTS_DIR / "GSE39055_RFS_primary_frozen_program_validation.csv"

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]
COHORT_ORDER = ["CANINE_DOG2", "TARGET_OS", "GSE21257", "GSE39055"]

N_RANDOM_SETS = 500
N_GENE_SUBSET_REPEATS = 500
GENE_SUBSET_FRACTION = 0.80
RANDOM_SEED = 42

OUTPUT_PRESERVATION = RESULTS_DIR / "cross_cohort_module_representation_preservation.csv"
OUTPUT_GENE_LOADINGS = RESULTS_DIR / "cross_cohort_module_gene_loading_concordance.csv"
OUTPUT_SUBSET_RELIABILITY = RESULTS_DIR / "cross_cohort_module_gene_subset_reliability.csv"
OUTPUT_RANDOM_SUMMARY = RESULTS_DIR / "cross_cohort_module_structure_random_controls.csv"
OUTPUT_RANDOM_DISTRIBUTION = RESULTS_DIR / "cross_cohort_module_structure_random_distribution.csv"
OUTPUT_OUTCOME_SYNTHESIS = RESULTS_DIR / "cross_cohort_module_preservation_outcome_synthesis.csv"
OUTPUT_MANIFEST = RESULTS_DIR / "cross_cohort_module_preservation_manifest.json"
OUTPUT_README = RESULTS_DIR / "cross_cohort_module_preservation_README.txt"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_required_csv(path: Path, index_col: int | str | None = None) -> pd.DataFrame:
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


def safe_corr(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    frame = pd.DataFrame({"a": np.asarray(a, dtype=float), "b": np.asarray(b, dtype=float)})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if frame.shape[0] < 3 or frame["a"].std() == 0 or frame["b"].std() == 0:
        return np.nan
    return float(frame["a"].corr(frame["b"]))


def off_diagonal_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.asarray([], dtype=float)
    return matrix[~np.eye(matrix.shape[0], dtype=bool)]


def pc1_metrics(z: pd.DataFrame, frozen_signs: pd.Series, frozen_loadings: pd.Series) -> tuple[dict[str, float], pd.DataFrame]:
    genes = list(z.columns)
    values = z.values.astype(float)
    signs = frozen_signs.reindex(genes).fillna(1.0).values.astype(float)
    signs[signs == 0] = 1.0
    signed_values = values * signs
    signed_mean = signed_values.mean(axis=1)

    _, singular_values, vt = np.linalg.svd(values, full_matrices=False)
    loading = vt[0].copy()
    score = values @ loading
    if safe_corr(score, signed_mean) < 0:
        score = -score
        loading = -loading

    denominator = float(np.sum(singular_values ** 2))
    pc1_variance = float(singular_values[0] ** 2 / denominator) if denominator > 0 else np.nan

    frozen_loading = frozen_loadings.reindex(genes).fillna(0.0).values.astype(float)
    frozen_loading_sign = np.sign(frozen_loading)
    frozen_loading_sign[frozen_loading_sign == 0] = 1.0
    cohort_loading_sign = np.sign(loading)
    cohort_loading_sign[cohort_loading_sign == 0] = 1.0

    sign_concordance = float(np.mean(cohort_loading_sign == frozen_loading_sign))
    loading_spearman = float(stats.spearmanr(loading, frozen_loading).statistic)

    signed_corr = np.corrcoef(signed_values, rowvar=False)
    signed_offdiag = off_diagonal_values(signed_corr)
    raw_corr = np.corrcoef(values, rowvar=False)
    raw_offdiag = off_diagonal_values(raw_corr)

    weighted_score = values @ frozen_loading
    metrics = {
        "pc1_variance_explained": pc1_variance,
        "frozen_loading_sign_concordance": sign_concordance,
        "frozen_loading_spearman": loading_spearman,
        "mean_signed_pairwise_correlation": float(np.mean(signed_offdiag)) if signed_offdiag.size else np.nan,
        "median_signed_pairwise_correlation": float(np.median(signed_offdiag)) if signed_offdiag.size else np.nan,
        "mean_absolute_pairwise_correlation": float(np.mean(np.abs(raw_offdiag))) if raw_offdiag.size else np.nan,
        "signed_mean_vs_canine_weighted_correlation": safe_corr(signed_mean, weighted_score),
        "signed_mean_vs_human_pc1_correlation": safe_corr(signed_mean, score),
    }

    gene_table = pd.DataFrame({
        "gene": genes,
        "frozen_risk_loading": frozen_loading,
        "frozen_risk_sign": frozen_loading_sign,
        "cohort_pc1_loading_aligned": loading,
        "cohort_pc1_sign": cohort_loading_sign,
        "loading_sign_concordant": cohort_loading_sign == frozen_loading_sign,
    })
    return metrics, gene_table


def subset_score_reliability(z: pd.DataFrame, frozen_signs: pd.Series, seed: int) -> tuple[dict[str, float], pd.DataFrame]:
    genes = list(z.columns)
    signs = np.sign(frozen_signs.reindex(genes).fillna(1.0)).replace(0, 1)
    full_score = z.mul(signs, axis=1).mean(axis=1)
    rows: list[dict[str, Any]] = []

    if len(genes) <= 10:
        subsets = [[gene for gene in genes if gene != removed] for removed in genes]
        method = "leave_one_gene_out"
    else:
        rng = np.random.default_rng(seed)
        subset_size = max(3, int(np.ceil(len(genes) * GENE_SUBSET_FRACTION)))
        subsets = [rng.choice(genes, size=subset_size, replace=False).tolist() for _ in range(N_GENE_SUBSET_REPEATS)]
        method = "random_gene_subsets"

    for iteration, subset in enumerate(subsets, start=1):
        subset_score = z[subset].mul(signs.reindex(subset), axis=1).mean(axis=1)
        rows.append({
            "iteration": iteration,
            "method": method,
            "n_genes_total": len(genes),
            "n_genes_subset": len(subset),
            "score_correlation_with_full": safe_corr(full_score, subset_score),
            "genes": ";".join(subset),
        })

    table = pd.DataFrame(rows)
    correlations = pd.to_numeric(table["score_correlation_with_full"], errors="coerce").dropna()
    summary = {
        "subset_method": method,
        "n_subset_iterations": table.shape[0],
        "subset_reliability_mean": float(correlations.mean()),
        "subset_reliability_median": float(correlations.median()),
        "subset_reliability_q05": float(correlations.quantile(0.05)),
        "subset_reliability_min": float(correlations.min()),
    }
    return summary, table


def expression_bins(expression: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame({"mean": expression.mean(axis=0), "sd": expression.std(axis=0)})
    table = table.replace([np.inf, -np.inf], np.nan).dropna()
    table = table[table["sd"] > 0].copy()
    table["mean_bin"] = pd.qcut(table["mean"], q=10, labels=False, duplicates="drop")
    table["sd_bin"] = pd.qcut(table["sd"], q=10, labels=False, duplicates="drop")
    table["bin_key"] = table["mean_bin"].astype(str) + "_" + table["sd_bin"].astype(str)
    return table


def draw_expression_matched_set(target_genes: list[str], stats_table: pd.DataFrame, excluded: set[str], rng: np.random.Generator) -> list[str]:
    targets = [gene for gene in target_genes if gene in stats_table.index]
    if not targets:
        return []
    target_bins = stats_table.loc[targets, "bin_key"].value_counts()
    selected: list[str] = []
    used = set(excluded)
    for bin_key, count in target_bins.items():
        pool = [gene for gene in stats_table.index[stats_table["bin_key"].eq(bin_key)] if gene not in used]
        take = min(int(count), len(pool))
        if take:
            chosen = rng.choice(pool, size=take, replace=False).tolist()
            selected.extend(chosen)
            used.update(chosen)
    needed = len(targets) - len(selected)
    if needed > 0:
        fallback = [gene for gene in stats_table.index if gene not in used]
        if len(fallback) < needed:
            return []
        selected.extend(rng.choice(fallback, size=needed, replace=False).tolist())
    return selected


def structure_metrics_fast(z: pd.DataFrame, signs: np.ndarray) -> tuple[float, float]:
    values = z.values.astype(float)
    signed = values * signs
    corr = np.corrcoef(signed, rowvar=False)
    offdiag = off_diagonal_values(corr)
    mean_signed_corr = float(np.mean(offdiag)) if offdiag.size else np.nan
    singular_values = np.linalg.svd(values, full_matrices=False, compute_uv=False)
    denominator = float(np.sum(singular_values ** 2))
    pc1_variance = float(singular_values[0] ** 2 / denominator) if denominator > 0 else np.nan
    return mean_signed_corr, pc1_variance


def random_structure_controls(expression: pd.DataFrame, z_all: pd.DataFrame, target_genes: list[str], frozen_signs: pd.Series, observed_signed_corr: float, observed_pc1_variance: float, seed: int) -> tuple[dict[str, float], pd.DataFrame]:
    stats_table = expression_bins(expression)
    rng = np.random.default_rng(seed)
    target_signs = np.sign(frozen_signs.reindex(target_genes).fillna(1.0).values)
    target_signs[target_signs == 0] = 1.0
    rows: list[dict[str, Any]] = []

    for repeat in range(1, N_RANDOM_SETS + 1):
        genes = draw_expression_matched_set(target_genes, stats_table, set(target_genes), rng)
        if len(genes) != len(target_genes):
            continue
        random_signs = rng.permutation(target_signs)
        signed_corr, pc1_variance = structure_metrics_fast(z_all[genes], random_signs)
        rows.append({
            "repeat": repeat,
            "mean_signed_pairwise_correlation": signed_corr,
            "pc1_variance_explained": pc1_variance,
        })

    table = pd.DataFrame(rows)
    if table.empty:
        return {
            "n_random_valid": 0,
            "signed_corr_random_percentile": np.nan,
            "signed_corr_empirical_p": np.nan,
            "pc1_variance_random_percentile": np.nan,
            "pc1_variance_empirical_p": np.nan,
        }, table

    signed_values = table["mean_signed_pairwise_correlation"].values
    pc1_values = table["pc1_variance_explained"].values
    summary = {
        "n_random_valid": table.shape[0],
        "signed_corr_random_mean": float(np.mean(signed_values)),
        "signed_corr_random_q95": float(np.quantile(signed_values, 0.95)),
        "signed_corr_random_percentile": float(np.mean(signed_values <= observed_signed_corr)),
        "signed_corr_empirical_p": float((1.0 + np.sum(signed_values >= observed_signed_corr)) / (len(signed_values) + 1.0)),
        "pc1_variance_random_mean": float(np.mean(pc1_values)),
        "pc1_variance_random_q95": float(np.quantile(pc1_values, 0.95)),
        "pc1_variance_random_percentile": float(np.mean(pc1_values <= observed_pc1_variance)),
        "pc1_variance_empirical_p": float((1.0 + np.sum(pc1_values >= observed_pc1_variance)) / (len(pc1_values) + 1.0)),
    }
    return summary, table


def canine_module_matrix(expression: pd.DataFrame, weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    part = weights.copy()
    part["canine_gene"] = part["canine_gene"].astype(str)
    part = part.drop_duplicates("canine_gene", keep="first")
    genes = [gene for gene in part["canine_gene"] if gene in expression.columns]
    if len(genes) < 3:
        return pd.DataFrame(index=expression.index), pd.Series(dtype=float), pd.Series(dtype=float)
    indexed = part.set_index("canine_gene").loc[genes]
    loadings = pd.to_numeric(indexed["risk_oriented_loading"], errors="coerce")
    signs = np.sign(loadings).replace(0, 1)
    return expression[genes].copy(), signs, loadings


def human_module_matrix(expression: pd.DataFrame, weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    part = weights.copy()
    part["human_gene_symbol"] = part["human_gene_symbol"].astype(str).str.upper()
    part = part.drop_duplicates("human_gene_symbol", keep="first")
    expression = expression.copy()
    expression.columns = expression.columns.astype(str).str.upper()
    expression = expression.loc[:, ~expression.columns.duplicated()].copy()
    genes = [gene for gene in part["human_gene_symbol"] if gene in expression.columns]
    if len(genes) < 3:
        return pd.DataFrame(index=expression.index), pd.Series(dtype=float), pd.Series(dtype=float)
    indexed = part.set_index("human_gene_symbol").loc[genes]
    loadings = pd.to_numeric(indexed["risk_oriented_loading"], errors="coerce")
    signs = np.sign(loadings).replace(0, 1)
    return expression[genes].copy(), signs, loadings


def analyze_cohort_module(cohort: str, expression: pd.DataFrame, weights: pd.DataFrame, module: str, seed: int) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    part = weights[weights["module_label"].eq(module)].copy()
    if cohort == "CANINE_DOG2":
        matrix, signs, loadings = canine_module_matrix(expression, part)
    else:
        matrix, signs, loadings = human_module_matrix(expression, part)

    if matrix.shape[1] < 3:
        return ({"cohort": cohort, "module_label": module, "n_samples": matrix.shape[0], "n_genes_available": matrix.shape[1], "error": "fewer_than_three_genes"}, pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame())

    z = zscore_matrix(matrix)
    genes = list(z.columns)
    signs = signs.reindex(genes)
    loadings = loadings.reindex(genes)

    metrics, gene_table = pc1_metrics(z, signs, loadings)
    reliability, subset_table = subset_score_reliability(z, signs, seed)

    expression_all = expression.copy()
    if cohort != "CANINE_DOG2":
        expression_all.columns = expression_all.columns.astype(str).str.upper()
        expression_all = expression_all.loc[:, ~expression_all.columns.duplicated()]
    expression_all = expression_all.apply(pd.to_numeric, errors="coerce")
    expression_all = expression_all.fillna(expression_all.median(axis=0))
    expression_all = expression_all.loc[:, expression_all.std(axis=0) > 0]
    z_all = zscore_matrix(expression_all)

    random_summary, random_table = random_structure_controls(
        expression_all,
        z_all,
        genes,
        signs,
        metrics["mean_signed_pairwise_correlation"],
        metrics["pc1_variance_explained"],
        seed + 100000,
    )

    summary = {"cohort": cohort, "module_label": module, "n_samples": z.shape[0], "n_genes_available": z.shape[1], **metrics, **reliability, **random_summary, "error": ""}
    gene_table.insert(0, "module_label", module)
    gene_table.insert(0, "cohort", cohort)
    subset_table.insert(0, "module_label", module)
    subset_table.insert(0, "cohort", cohort)
    random_table.insert(0, "module_label", module)
    random_table.insert(0, "cohort", cohort)
    return summary, gene_table, subset_table, random_summary, random_table


def load_outcome_synthesis() -> pd.DataFrame:
    target = read_required_csv(TARGET_PRIMARY_FILE)
    gse = read_required_csv(GSE21257_ROBUST_FILE)
    gse39055 = read_required_csv(GSE39055_PRIMARY_FILE)

    target_use = target[["module_label", "score_hr_per_sd", "primary_p", "q_within_endpoint", "fixed_score_c_index"]].rename(columns={
        "score_hr_per_sd": "target_os_hr_per_sd",
        "primary_p": "target_os_p",
        "q_within_endpoint": "target_os_q",
        "fixed_score_c_index": "target_os_fixed_c_index",
    })
    gse_use = gse[["module_label", "auc", "or_per_sd", "permutation_auc_q_bh"]].rename(columns={
        "auc": "gse21257_metastasis_auc",
        "or_per_sd": "gse21257_metastasis_or_per_sd",
        "permutation_auc_q_bh": "gse21257_permutation_q",
    })
    rfs_use = gse39055[["module_label", "hr_per_sd", "primary_p", "q_within_gse39055", "fixed_score_c_index"]].rename(columns={
        "hr_per_sd": "gse39055_rfs_hr_per_sd",
        "primary_p": "gse39055_rfs_p",
        "q_within_gse39055": "gse39055_rfs_q",
        "fixed_score_c_index": "gse39055_rfs_fixed_c_index",
    })

    result = target_use.merge(gse_use, on="module_label", how="outer").merge(rfs_use, on="module_label", how="outer")
    result["target_direction_concordant"] = result["target_os_hr_per_sd"] > 1
    result["gse21257_direction_concordant"] = result["gse21257_metastasis_or_per_sd"] > 1
    result["gse39055_direction_concordant"] = result["gse39055_rfs_hr_per_sd"] > 1
    result["n_human_settings_direction_concordant"] = result[["target_direction_concordant", "gse21257_direction_concordant", "gse39055_direction_concordant"]].sum(axis=1)
    return result


def classify_structure(row: pd.Series) -> str:
    sign_concordance = row.get("frozen_loading_sign_concordance", np.nan)
    reliability = row.get("subset_reliability_median", np.nan)
    signed_q = row.get("signed_corr_empirical_q", np.nan)
    pc1_q = row.get("pc1_variance_empirical_q", np.nan)

    if (
        np.isfinite(sign_concordance)
        and sign_concordance >= 0.60
        and np.isfinite(reliability)
        and reliability >= 0.90
        and ((np.isfinite(signed_q) and signed_q < 0.10) or (np.isfinite(pc1_q) and pc1_q < 0.10))
    ):
        return "strong_structural_preservation"
    if np.isfinite(sign_concordance) and sign_concordance >= 0.55 and np.isfinite(reliability) and reliability >= 0.80:
        return "moderate_structural_preservation"
    return "weak_structural_preservation"


def build_preservation_outcome_synthesis(preservation: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    human = preservation[preservation["cohort"].ne("CANINE_DOG2")].copy()
    human["structural_support_class"] = human.apply(classify_structure, axis=1)
    synthesis = human.merge(outcomes, on="module_label", how="left")

    def outcome_direction(row: pd.Series) -> bool:
        if row["cohort"] == "TARGET_OS":
            return bool(row.get("target_direction_concordant", False))
        if row["cohort"] == "GSE21257":
            return bool(row.get("gse21257_direction_concordant", False))
        if row["cohort"] == "GSE39055":
            return bool(row.get("gse39055_direction_concordant", False))
        return False

    synthesis["outcome_direction_concordant"] = synthesis.apply(outcome_direction, axis=1)

    def classify_joint(row: pd.Series) -> str:
        structural = row["structural_support_class"]
        outcome = bool(row["outcome_direction_concordant"])
        if structural == "strong_structural_preservation" and outcome:
            return "preserved_representation_and_concordant_outcome"
        if structural == "strong_structural_preservation" and not outcome:
            return "preserved_representation_but_outcome_heterogeneity"
        if structural == "moderate_structural_preservation" and outcome:
            return "moderate_representation_with_concordant_direction"
        if not outcome:
            return "weak_or_moderate_representation_with_outcome_discordance"
        return "directionally_concordant_with_weak_structural_support"

    synthesis["joint_preservation_outcome_class"] = synthesis.apply(classify_joint, axis=1)
    return synthesis


def write_readme() -> None:
    text = f"""Cross-cohort frozen-module representation preservation audit
Script version: {SCRIPT_VERSION}

Purpose
-------
This analysis distinguishes structural transport of a frozen module from
transport of its outcome association. It evaluates canine DOG2, TARGET-OS,
GSE21257, and GSE39055 without changing frozen genes, weights, directions,
or validation tiers.

Structural metrics
------------------
- PC1 variance explained
- concordance of cohort PC1 loadings with frozen canine risk loadings
- mean signed pairwise gene correlation
- correlation among signed-mean, canine-weighted, and human-PC1 scores
- leave-one-gene-out or repeated gene-subset score reliability
- expression-matched random-module controls

Interpretation
--------------
A module can preserve its co-expression representation while its prognostic
association changes across endpoints or cohorts. Such a result indicates
outcome heterogeneity rather than proof that the module definition failed.
Conversely, weak structural preservation can implicate platform or
representation instability.

Random controls
---------------
{N_RANDOM_SETS} expression-matched random gene sets are used per
cohort-module pair. Their p-values are descriptive specificity controls.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def create_manifest(outputs: list[Path]) -> None:
    payload = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "frozen_weights_sha256": sha256_file(STRICT_WEIGHTS_FILE),
        "freeze_json_sha256": sha256_file(FREEZE_JSON_FILE) if FREEZE_JSON_FILE.exists() else None,
        "primary_modules": PRIMARY_MODULES,
        "cohorts": COHORT_ORDER,
        "random_sets_per_cohort_module": N_RANDOM_SETS,
        "gene_subset_repeats": N_GENE_SUBSET_REPEATS,
        "files": {},
    }
    for path in outputs:
        if path.exists():
            payload["files"][path.name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    OUTPUT_MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("Cross-cohort frozen-module representation preservation audit")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Results directory: {RESULTS_DIR}")
    print("")
    print("Design:")
    print("  Separate module-structure preservation from outcome-association transfer.")
    print("  Compare canine DOG2 with TARGET-OS, GSE21257, and GSE39055.")
    print("  Preserve frozen strict genes, canine loadings, risk signs, and tiers.")
    print("  Use expression-matched random modules and gene-subset reliability checks.")
    print("")

    weights = read_required_csv(STRICT_WEIGHTS_FILE)
    canine_expression = read_required_csv(CANINE_EXPRESSION_FILE, index_col=0)
    target_expression = read_required_csv(TARGET_EXPRESSION_FILE, index_col=0)
    gse21257_expression = read_required_csv(GSE21257_EXPRESSION_FILE, index_col=0)
    gse39055_expression = read_required_csv(GSE39055_EXPRESSION_FILE, index_col=0)

    cohort_expression = {
        "CANINE_DOG2": canine_expression,
        "TARGET_OS": target_expression,
        "GSE21257": gse21257_expression,
        "GSE39055": gse39055_expression,
    }

    preservation_rows: list[dict[str, Any]] = []
    gene_tables: list[pd.DataFrame] = []
    subset_tables: list[pd.DataFrame] = []
    random_summaries: list[dict[str, Any]] = []
    random_tables: list[pd.DataFrame] = []

    total_jobs = len(COHORT_ORDER) * len(PRIMARY_MODULES)
    job_index = 0

    for cohort_index, cohort in enumerate(COHORT_ORDER):
        expression = cohort_expression[cohort]
        print("")
        print("=" * 80)
        print(f"Cohort: {cohort}")
        print("=" * 80)
        print(f"Expression matrix: {expression.shape}")

        for module_index, module in enumerate(PRIMARY_MODULES):
            job_index += 1
            print(f"  Job {job_index}/{total_jobs}: {cohort} {module}")
            seed = RANDOM_SEED + cohort_index * 1000000 + module_index * 10000
            summary, gene_table, subset_table, random_summary, random_table = analyze_cohort_module(
                cohort=cohort,
                expression=expression,
                weights=weights,
                module=module,
                seed=seed,
            )
            preservation_rows.append(summary)
            if not gene_table.empty:
                gene_tables.append(gene_table)
            if not subset_table.empty:
                subset_tables.append(subset_table)
            if random_summary:
                random_summaries.append({"cohort": cohort, "module_label": module, **random_summary})
            if not random_table.empty:
                random_tables.append(random_table)

    preservation = pd.DataFrame(preservation_rows)
    if not preservation.empty:
        preservation["signed_corr_empirical_q"] = bh_adjust(preservation["signed_corr_empirical_p"])
        preservation["pc1_variance_empirical_q"] = bh_adjust(preservation["pc1_variance_empirical_p"])

    gene_loading_table = pd.concat(gene_tables, ignore_index=True) if gene_tables else pd.DataFrame()
    subset_reliability_table = pd.concat(subset_tables, ignore_index=True) if subset_tables else pd.DataFrame()
    random_summary_table = pd.DataFrame(random_summaries)
    random_distribution_table = pd.concat(random_tables, ignore_index=True) if random_tables else pd.DataFrame()

    outcomes = load_outcome_synthesis()
    synthesis = build_preservation_outcome_synthesis(preservation, outcomes)

    preservation.to_csv(OUTPUT_PRESERVATION, index=False)
    gene_loading_table.to_csv(OUTPUT_GENE_LOADINGS, index=False)
    subset_reliability_table.to_csv(OUTPUT_SUBSET_RELIABILITY, index=False)
    random_summary_table.to_csv(OUTPUT_RANDOM_SUMMARY, index=False)
    random_distribution_table.to_csv(OUTPUT_RANDOM_DISTRIBUTION, index=False)
    synthesis.to_csv(OUTPUT_OUTCOME_SYNTHESIS, index=False)

    write_readme()
    create_manifest([
        OUTPUT_PRESERVATION,
        OUTPUT_GENE_LOADINGS,
        OUTPUT_SUBSET_RELIABILITY,
        OUTPUT_RANDOM_SUMMARY,
        OUTPUT_RANDOM_DISTRIBUTION,
        OUTPUT_OUTCOME_SYNTHESIS,
        OUTPUT_README,
    ])

    print("")
    print("=" * 80)
    print("Module representation preservation summary")
    print("=" * 80)
    display_cols = [
        "cohort", "module_label", "n_samples", "n_genes_available",
        "pc1_variance_explained", "frozen_loading_sign_concordance",
        "frozen_loading_spearman", "mean_signed_pairwise_correlation",
        "signed_mean_vs_canine_weighted_correlation",
        "signed_mean_vs_human_pc1_correlation",
        "subset_reliability_median", "signed_corr_random_percentile",
        "pc1_variance_random_percentile",
    ]
    print(preservation[[c for c in display_cols if c in preservation.columns]].to_string(index=False))

    print("")
    print("=" * 80)
    print("Human preservation and outcome synthesis")
    print("=" * 80)
    synthesis_cols = [
        "cohort", "module_label", "structural_support_class",
        "outcome_direction_concordant", "joint_preservation_outcome_class",
        "frozen_loading_sign_concordance", "frozen_loading_spearman",
        "subset_reliability_median", "n_human_settings_direction_concordant",
    ]
    print(synthesis[[c for c in synthesis_cols if c in synthesis.columns]].sort_values(["module_label", "cohort"]).to_string(index=False))

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("This audit does not alter frozen module membership, weights, risk direction, or validation tier.")
    print("Structure preservation and outcome association are separate scientific quantities.")
    print("A preserved representation with a discordant outcome supports endpoint or cohort heterogeneity, not post hoc score reversal.")
    print("Human PC1 loadings are aligned to the frozen signed-mean score without using outcomes.")
    print("Expression-matched random-module p-values are descriptive specificity controls.")

    print("")
    print("Saved:")
    for path in [
        OUTPUT_PRESERVATION,
        OUTPUT_GENE_LOADINGS,
        OUTPUT_SUBSET_RELIABILITY,
        OUTPUT_RANDOM_SUMMARY,
        OUTPUT_RANDOM_DISTRIBUTION,
        OUTPUT_OUTCOME_SYNTHESIS,
        OUTPUT_README,
        OUTPUT_MANIFEST,
    ]:
        print(path)
    print("Done.")


if __name__ == "__main__":
    main()
