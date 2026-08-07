from pathlib import Path
import hashlib
import json
import math
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import t as student_t

warnings.filterwarnings("ignore")

SCRIPT_VERSION = "49-gse239948-blind-de-novo-rediscovery-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXPRESSION_FILE = (
    PROCESSED_DIR / "canine_validation_GSE239948_expression_log2_symbol.csv"
)
FROZEN_WEIGHTS_FILE = (
    RESULTS_DIR / "GSE238110_frozen_transfer_gene_weights_strict.csv"
)
SCRIPT47_LOCK_FILE = (
    RESULTS_DIR / "paper4_locked_independent_canine_representation.csv"
)
SCRIPT47_MANIFEST_FILE = (
    RESULTS_DIR / "paper4_external_canine_evidence_manifest.json"
)

OUTPUT_GENE_UNIVERSE = (
    RESULTS_DIR / "GSE239948_blind_discovery_gene_universe.csv"
)
OUTPUT_MODULE_MEMBERSHIP = (
    RESULTS_DIR / "GSE239948_blind_discovered_module_membership.csv"
)
OUTPUT_MODULE_SUMMARY = (
    RESULTS_DIR / "GSE239948_blind_discovered_module_summary.csv"
)
OUTPUT_SUBSAMPLE_STABILITY = (
    RESULTS_DIR / "GSE239948_blind_discovered_module_subsample_stability.csv"
)
OUTPUT_REDISCOVERY = (
    RESULTS_DIR / "GSE239948_blind_frozen_program_rediscovery.csv"
)
OUTPUT_RANDOM_CONTROLS = (
    RESULTS_DIR / "GSE239948_blind_frozen_program_random_controls.csv"
)
OUTPUT_README = (
    RESULTS_DIR / "GSE239948_blind_de_novo_rediscovery_README.txt"
)
OUTPUT_MANIFEST = (
    RESULTS_DIR / "GSE239948_blind_de_novo_rediscovery_manifest.json"
)

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]

DISCOVERY_TOP_VARIABLE_GENES = 3000
MIN_DISCOVERED_MODULE_SIZE = 5
CORRELATION_GRAPH_NOMINAL_P = 0.01
SUBSAMPLE_REPEATS = 30
SUBSAMPLE_FRACTION = 0.80
RANDOM_PANELS = 2000
VARIANCE_BINS = 10
RANDOM_SEED = 20260806


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_gene_symbol(value) -> str:
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE"}:
        return ""
    tail = text.rsplit("_", 1)[-1]
    if tail.isdigit():
        text = text.rsplit("_", 1)[0]
    return text


def read_required_csv(path: Path, index_col=None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    print(f"Loaded: {path}")
    return pd.read_csv(path, index_col=index_col)


def prepare_expression(raw: pd.DataFrame) -> pd.DataFrame:
    x = raw.copy()
    x.index = x.index.astype(str)
    x.columns = [clean_gene_symbol(c) for c in x.columns]
    keep = np.asarray([bool(c) for c in x.columns], dtype=bool)
    x = x.loc[:, keep]
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)

    if x.columns.duplicated().any():
        x = x.T.groupby(level=0).median().T

    medians = x.median(axis=0)
    x = x.fillna(medians)
    variable = x.var(axis=0, ddof=1)
    keep_cols = variable.index[(variable > 0) & np.isfinite(variable)]
    return x.loc[:, keep_cols].copy()


def select_discovery_universe(expression: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    variances = expression.var(axis=0, ddof=1).sort_values(ascending=False)
    n_keep = min(DISCOVERY_TOP_VARIABLE_GENES, variances.shape[0])
    selected = variances.iloc[:n_keep].index.tolist()
    x = expression[selected].copy()

    ranks = variances.rank(method="average", pct=True)
    audit = pd.DataFrame(
        {
            "gene_symbol": variances.index,
            "variance": variances.values,
            "variance_percentile": ranks.reindex(variances.index).values,
            "selected_for_blind_discovery": variances.index.isin(selected),
        }
    )
    return x, audit


def rank_transform(expression: pd.DataFrame) -> np.ndarray:
    ranked = expression.rank(axis=0, method="average")
    values = ranked.to_numpy(dtype=float)
    values -= values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, ddof=1, keepdims=True)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    values /= std
    return values


def correlation_threshold(n_samples: int) -> float:
    if n_samples < 5:
        return 0.99
    df = n_samples - 2
    tcrit = float(student_t.ppf(1.0 - CORRELATION_GRAPH_NOMINAL_P, df=df))
    return float(tcrit / math.sqrt(tcrit * tcrit + df))


def spearman_gene_correlation(expression: pd.DataFrame) -> np.ndarray:
    ranked = rank_transform(expression)
    corr = np.corrcoef(ranked, rowvar=False)
    corr = np.asarray(corr, dtype=float)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def cluster_expression(expression: pd.DataFrame) -> tuple[pd.Series, float]:
    if expression.shape[1] < MIN_DISCOVERED_MODULE_SIZE:
        return pd.Series(0, index=expression.columns, dtype=int), np.nan

    corr = spearman_gene_correlation(expression)
    distance = 1.0 - corr
    distance = np.clip(distance, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method="average", optimal_ordering=False)

    r_cut = correlation_threshold(expression.shape[0])
    distance_cut = 1.0 - r_cut
    raw_labels = fcluster(tree, t=distance_cut, criterion="distance")
    labels = pd.Series(raw_labels, index=expression.columns, dtype=int)

    counts = labels.value_counts()
    valid = counts[counts >= MIN_DISCOVERED_MODULE_SIZE].index
    labels = labels.where(labels.isin(valid), 0)

    nonzero = sorted(v for v in labels.unique() if v != 0)
    remap = {old: new for new, old in enumerate(nonzero, start=1)}
    labels = labels.map(lambda value: remap.get(value, 0)).astype(int)
    return labels, r_cut


def summarize_modules(expression: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    rows = []
    for module_id in sorted(v for v in labels.unique() if v != 0):
        genes = labels.index[labels.eq(module_id)].tolist()
        part = expression[genes]
        corr = spearman_gene_correlation(part)
        upper = corr[np.triu_indices(len(genes), k=1)]
        rows.append(
            {
                "discovered_module_id": int(module_id),
                "n_genes": len(genes),
                "mean_pairwise_spearman": float(np.nanmean(upper)) if upper.size else np.nan,
                "median_pairwise_spearman": float(np.nanmedian(upper)) if upper.size else np.nan,
                "genes": ";".join(genes),
            }
        )
    return pd.DataFrame(rows)


def module_sets_from_labels(labels: pd.Series) -> dict[int, set[str]]:
    result = {}
    for module_id in sorted(v for v in labels.unique() if v != 0):
        result[int(module_id)] = set(labels.index[labels.eq(module_id)])
    return result


def jaccard(a: set[str], b: set[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else np.nan


def f1_overlap(a: set[str], b: set[str]) -> float:
    denom = len(a) + len(b)
    return 2.0 * len(a & b) / denom if denom else np.nan


def best_set_match(query: set[str], candidates: dict[int, set[str]]) -> dict:
    best = {
        "module_id": np.nan,
        "overlap": 0,
        "jaccard": 0.0,
        "f1": 0.0,
        "query_recall": 0.0,
        "module_precision": 0.0,
        "module_size": 0,
    }
    if not query or not candidates:
        return best

    for module_id, genes in candidates.items():
        overlap = len(query & genes)
        score = f1_overlap(query, genes)
        if (
            score > best["f1"]
            or (np.isclose(score, best["f1"]) and overlap > best["overlap"])
        ):
            best = {
                "module_id": int(module_id),
                "overlap": int(overlap),
                "jaccard": float(jaccard(query, genes)),
                "f1": float(score),
                "query_recall": float(overlap / len(query)) if query else np.nan,
                "module_precision": float(overlap / len(genes)) if genes else np.nan,
                "module_size": int(len(genes)),
            }
    return best


def subsample_stability(
    expression: pd.DataFrame,
    full_labels: pd.Series,
    rng: np.random.Generator,
) -> pd.DataFrame:
    full_modules = module_sets_from_labels(full_labels)
    records = []
    n_samples = expression.shape[0]
    subset_n = max(10, int(round(n_samples * SUBSAMPLE_FRACTION)))

    for repeat in range(1, SUBSAMPLE_REPEATS + 1):
        chosen = rng.choice(n_samples, size=subset_n, replace=False)
        subset = expression.iloc[np.sort(chosen)]
        labels, r_cut = cluster_expression(subset)
        repeat_modules = module_sets_from_labels(labels)

        for module_id, genes in full_modules.items():
            match = best_set_match(genes, repeat_modules)
            records.append(
                {
                    "repeat": repeat,
                    "full_discovered_module_id": module_id,
                    "subsample_n": subset_n,
                    "subsample_r_cut": r_cut,
                    "best_subsample_module_id": match["module_id"],
                    "best_jaccard": match["jaccard"],
                    "best_f1": match["f1"],
                    "overlap_genes": match["overlap"],
                }
            )

    raw = pd.DataFrame(records)
    if raw.empty:
        return raw

    summary = (
        raw.groupby("full_discovered_module_id", as_index=False)
        .agg(
            n_repeats=("repeat", "count"),
            median_best_jaccard=("best_jaccard", "median"),
            q05_best_jaccard=("best_jaccard", lambda x: float(np.quantile(x, 0.05))),
            median_best_f1=("best_f1", "median"),
            fraction_jaccard_ge_0_50=("best_jaccard", lambda x: float(np.mean(np.asarray(x) >= 0.50))),
            fraction_f1_ge_0_50=("best_f1", lambda x: float(np.mean(np.asarray(x) >= 0.50))),
        )
    )
    return summary


def find_canine_gene_column(weights: pd.DataFrame) -> str:
    for column in ["canine_gene_symbol", "canine_gene", "gene"]:
        if column in weights.columns:
            return column
    raise ValueError("No canine gene-symbol column found in frozen weights.")


def frozen_gene_sets(weights: pd.DataFrame) -> dict[str, set[str]]:
    gene_col = find_canine_gene_column(weights)
    output = {}
    for module in PRIMARY_MODULES:
        part = weights[weights["module_label"].astype(str).eq(module)]
        output[module] = {
            clean_gene_symbol(value)
            for value in part[gene_col]
            if clean_gene_symbol(value)
        }
    return output


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvalues, errors="coerce")
    q = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.notna() & np.isfinite(p)
    if valid.sum() == 0:
        return q
    values = p[valid].to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty(n, dtype=float)
    restored[order] = adjusted
    q.loc[p[valid].index] = restored
    return q


def make_variance_bins(universe_audit: pd.DataFrame) -> pd.Series:
    part = universe_audit[universe_audit["selected_for_blind_discovery"]].copy()
    values = part.set_index("gene_symbol")["variance"]
    ranks = values.rank(method="first", pct=True)
    bins = np.minimum((ranks * VARIANCE_BINS).astype(int), VARIANCE_BINS - 1)
    bins = bins.clip(lower=0)
    return bins.astype(int)


def matched_random_panel(
    target_genes: set[str],
    variance_bins: pd.Series,
    forbidden: set[str],
    rng: np.random.Generator,
) -> set[str]:
    target = [g for g in target_genes if g in variance_bins.index]
    if not target:
        return set()

    pools = {}
    for bin_id in sorted(variance_bins.unique()):
        genes = set(variance_bins.index[variance_bins.eq(bin_id)]) - forbidden
        pools[int(bin_id)] = sorted(genes)

    selected = []
    used = set()
    for gene in target:
        bin_id = int(variance_bins.loc[gene])
        pool = [g for g in pools.get(bin_id, []) if g not in used]
        if not pool:
            pool = [g for g in variance_bins.index if g not in forbidden and g not in used]
        if not pool:
            break
        chosen = str(rng.choice(pool))
        selected.append(chosen)
        used.add(chosen)
    return set(selected)


def rediscovery_analysis(
    discovery_genes: set[str],
    discovered_modules: dict[int, set[str]],
    stability: pd.DataFrame,
    weights: pd.DataFrame,
    universe_audit: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frozen = frozen_gene_sets(weights)
    all_primary_frozen = set().union(*frozen.values())
    variance_bins = make_variance_bins(universe_audit)

    stability_index = (
        stability.set_index("full_discovered_module_id")
        if not stability.empty
        else pd.DataFrame()
    )

    result_rows = []
    null_rows = []

    for module in PRIMARY_MODULES:
        frozen_all = frozen[module]
        frozen_in_universe = frozen_all & discovery_genes
        observed = best_set_match(frozen_in_universe, discovered_modules)

        null_f1 = []
        null_jaccard = []
        for iteration in range(1, RANDOM_PANELS + 1):
            panel = matched_random_panel(
                frozen_in_universe,
                variance_bins,
                forbidden=all_primary_frozen,
                rng=rng,
            )
            match = best_set_match(panel, discovered_modules)
            null_f1.append(match["f1"])
            null_jaccard.append(match["jaccard"])
            null_rows.append(
                {
                    "module_label": module,
                    "iteration": iteration,
                    "random_panel_size": len(panel),
                    "maximum_match_f1": match["f1"],
                    "maximum_match_jaccard": match["jaccard"],
                }
            )

        null_f1_arr = np.asarray(null_f1, dtype=float)
        null_j_arr = np.asarray(null_jaccard, dtype=float)
        empirical_p = (
            1.0 + np.sum(null_f1_arr >= observed["f1"])
        ) / (1.0 + len(null_f1_arr))

        best_id = observed["module_id"]
        stability_median = np.nan
        stability_q05 = np.nan
        stability_fraction = np.nan
        if (
            np.isfinite(best_id)
            and not stability.empty
            and int(best_id) in stability_index.index
        ):
            row = stability_index.loc[int(best_id)]
            stability_median = float(row["median_best_jaccard"])
            stability_q05 = float(row["q05_best_jaccard"])
            stability_fraction = float(row["fraction_jaccard_ge_0_50"])

        result_rows.append(
            {
                "module_label": module,
                "n_frozen_canine_genes": len(frozen_all),
                "n_frozen_genes_in_blind_discovery_universe": len(frozen_in_universe),
                "discovery_universe_coverage_fraction": (
                    len(frozen_in_universe) / len(frozen_all) if frozen_all else np.nan
                ),
                "best_discovered_module_id": best_id,
                "best_discovered_module_size": observed["module_size"],
                "overlap_genes": observed["overlap"],
                "frozen_gene_recall_within_discovery_universe": observed["query_recall"],
                "discovered_module_precision": observed["module_precision"],
                "best_match_jaccard": observed["jaccard"],
                "best_match_f1": observed["f1"],
                "best_module_subsample_stability_median_jaccard": stability_median,
                "best_module_subsample_stability_q05_jaccard": stability_q05,
                "best_module_fraction_subsamples_jaccard_ge_0_50": stability_fraction,
                "random_max_f1_mean": float(np.nanmean(null_f1_arr)),
                "random_max_f1_q95": float(np.nanquantile(null_f1_arr, 0.95)),
                "random_max_jaccard_q95": float(np.nanquantile(null_j_arr, 0.95)),
                "empirical_max_match_p": float(empirical_p),
            }
        )

    results = pd.DataFrame(result_rows)
    results["empirical_max_match_q_bh_4"] = bh_adjust(results["empirical_max_match_p"])

    classes = []
    for _, row in results.iterrows():
        n_in = int(row["n_frozen_genes_in_blind_discovery_universe"])
        q = row["empirical_max_match_q_bh_4"]
        recall = row["frozen_gene_recall_within_discovery_universe"]
        stability_median = row["best_module_subsample_stability_median_jaccard"]

        if n_in < 3:
            label = "insufficient_blind_discovery_coverage"
        elif (
            np.isfinite(q)
            and q < 0.05
            and recall >= 0.30
            and np.isfinite(stability_median)
            and stability_median >= 0.50
        ):
            label = "strong_blind_independent_rediscovery"
        elif np.isfinite(q) and q < 0.10 and recall >= 0.20:
            label = "partial_blind_independent_rediscovery"
        else:
            label = "no_clear_blind_independent_rediscovery"
        classes.append(label)

    results["blind_rediscovery_class"] = classes
    return results, pd.DataFrame(null_rows)


def create_readme(r_cut: float) -> None:
    text = f"""GSE239948 blind de novo rediscovery audit
=========================================

Script version
--------------
{SCRIPT_VERSION}

Purpose
-------
Test whether de novo co-expression modules formed in GSE239948 recover frozen canine programs without using frozen membership during module discovery.

Discovery procedure
-------------------
- Only GSE239948 expression is used during feature selection and clustering.
- The top {DISCOVERY_TOP_VARIABLE_GENES} genes by GSE239948 variance define the discovery universe.
- Gene-gene Spearman correlation is clustered by average linkage.
- The full-cohort correlation cut corresponds to a one-sided nominal positive-correlation P value of {CORRELATION_GRAPH_NOMINAL_P}; the observed full-cohort r cut was {r_cut:.4f}.
- Clusters smaller than {MIN_DISCOVERED_MODULE_SIZE} genes are not treated as discovered modules.
- Stability is evaluated across {SUBSAMPLE_REPEATS} outcome-blind {int(SUBSAMPLE_FRACTION * 100)}% sample subsamples.

Frozen-program matching
-----------------------
Frozen M34, M11, M24, and M40 gene sets are loaded only after de novo modules have been constructed.
Each frozen program is compared with every discovered module, and the best F1 overlap is retained.
Expression-variance-matched random panels also take their maximum match across all discovered modules, so the empirical null includes the best-of-many-module search.
BH correction is applied across the four frozen primary programs.

Interpretation guardrails
-------------------------
- This is a representation rediscovery analysis, not an outcome validation analysis.
- Variable-gene filtering is outcome-blind but may reduce coverage of frozen programs.
- A significant best-match result supports independent rediscovery of a related co-expression structure; it does not imply identical module boundaries.
- The cohort has 43 samples, so module boundaries and small-module results require caution.
- No human or canine outcome is loaded or used.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def create_manifest(input_paths: list[Path], output_paths: list[Path], r_cut: float) -> None:
    payload = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "outcome_loaded": False,
        "discovery_top_variable_genes": DISCOVERY_TOP_VARIABLE_GENES,
        "minimum_discovered_module_size": MIN_DISCOVERED_MODULE_SIZE,
        "correlation_graph_nominal_p": CORRELATION_GRAPH_NOMINAL_P,
        "full_cohort_r_cut": r_cut,
        "subsample_repeats": SUBSAMPLE_REPEATS,
        "subsample_fraction": SUBSAMPLE_FRACTION,
        "random_panels": RANDOM_PANELS,
        "random_seed": RANDOM_SEED,
        "guardrails": [
            "Frozen module membership is not used during discovery clustering.",
            "Frozen programs are loaded only after de novo modules are constructed.",
            "Random nulls include maximum matching across all discovered modules.",
            "No outcome data are loaded.",
        ],
        "inputs": {},
        "outputs": {},
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

    OUTPUT_MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("GSE239948 blind de novo consensus rediscovery of canine programs")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Design:")
    print("  Build de novo modules using GSE239948 expression only.")
    print("  Do not load frozen module membership until discovery is complete.")
    print("  Audit module stability across outcome-blind sample subsamples.")
    print("  Test best frozen-program matches against variance-matched random panels.")
    print("  Correct the null for searching across all discovered modules.")
    print("")

    raw = read_required_csv(EXPRESSION_FILE, index_col=0)
    expression = prepare_expression(raw)
    discovery_expression, universe_audit = select_discovery_universe(expression)

    print("Blind discovery data:")
    print(f"  Samples: {discovery_expression.shape[0]}")
    print(f"  Expression genes available: {expression.shape[1]}")
    print(f"  Genes in blind discovery universe: {discovery_expression.shape[1]}")

    labels, r_cut = cluster_expression(discovery_expression)
    discovered_modules = module_sets_from_labels(labels)
    module_summary = summarize_modules(discovery_expression, labels)

    membership = pd.DataFrame(
        {
            "gene_symbol": labels.index,
            "discovered_module_id": labels.values,
        }
    )
    membership = membership.merge(
        universe_audit[["gene_symbol", "variance", "variance_percentile"]],
        on="gene_symbol",
        how="left",
    )

    print("")
    print("=" * 80)
    print("Blind de novo module discovery")
    print("=" * 80)
    print(f"Positive-correlation cut: r >= {r_cut:.4f}")
    print(f"Discovered modules >= {MIN_DISCOVERED_MODULE_SIZE} genes: {len(discovered_modules)}")
    if not module_summary.empty:
        display = module_summary.sort_values("n_genes", ascending=False).head(30)
        print(display[[
            "discovered_module_id",
            "n_genes",
            "mean_pairwise_spearman",
            "median_pairwise_spearman",
        ]].to_string(index=False))

    rng = np.random.default_rng(RANDOM_SEED)
    stability = subsample_stability(discovery_expression, labels, rng)

    print("")
    print("=" * 80)
    print("Subsample module stability")
    print("=" * 80)
    if stability.empty:
        print("No stable discovered modules were available for subsample analysis.")
    else:
        display = stability.sort_values(
            ["median_best_jaccard", "median_best_f1"],
            ascending=False,
        ).head(30)
        print(display.to_string(index=False))

    # Frozen membership is intentionally loaded only after blind discovery is complete.
    weights = read_required_csv(FROZEN_WEIGHTS_FILE)
    if SCRIPT47_LOCK_FILE.exists():
        read_required_csv(SCRIPT47_LOCK_FILE)
    if SCRIPT47_MANIFEST_FILE.exists():
        print(f"Loaded: {SCRIPT47_MANIFEST_FILE}")
        manifest47 = json.loads(SCRIPT47_MANIFEST_FILE.read_text(encoding="utf-8"))
        if manifest47.get("script_version") != "47-lock-gse239948-independent-canine-evidence-v2":
            raise RuntimeError(
                "Script 47 lock is not from the expected v2 independence audit."
            )

    rediscovery, random_controls = rediscovery_analysis(
        discovery_genes=set(discovery_expression.columns),
        discovered_modules=discovered_modules,
        stability=stability,
        weights=weights,
        universe_audit=universe_audit,
        rng=rng,
    )

    print("")
    print("=" * 80)
    print("Blind frozen-program rediscovery")
    print("=" * 80)
    columns = [
        "module_label",
        "n_frozen_canine_genes",
        "n_frozen_genes_in_blind_discovery_universe",
        "discovery_universe_coverage_fraction",
        "best_discovered_module_id",
        "best_discovered_module_size",
        "overlap_genes",
        "frozen_gene_recall_within_discovery_universe",
        "discovered_module_precision",
        "best_match_jaccard",
        "best_match_f1",
        "best_module_subsample_stability_median_jaccard",
        "random_max_f1_q95",
        "empirical_max_match_p",
        "empirical_max_match_q_bh_4",
        "blind_rediscovery_class",
    ]
    print(rediscovery[columns].to_string(index=False))

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("Frozen genes are not used to construct the de novo modules.")
    print("The random-panel null repeats the best-of-many discovered-module search.")
    print("Variable-gene filtering is outcome-blind and natural frozen-gene coverage is reported.")
    print("A recovered module need not have identical boundaries to the frozen canine program.")
    print("No outcome data are loaded or tested.")

    universe_audit.to_csv(OUTPUT_GENE_UNIVERSE, index=False)
    membership.to_csv(OUTPUT_MODULE_MEMBERSHIP, index=False)
    module_summary.to_csv(OUTPUT_MODULE_SUMMARY, index=False)
    stability.to_csv(OUTPUT_SUBSAMPLE_STABILITY, index=False)
    rediscovery.to_csv(OUTPUT_REDISCOVERY, index=False)
    random_controls.to_csv(OUTPUT_RANDOM_CONTROLS, index=False)
    create_readme(r_cut)

    output_paths = [
        OUTPUT_GENE_UNIVERSE,
        OUTPUT_MODULE_MEMBERSHIP,
        OUTPUT_MODULE_SUMMARY,
        OUTPUT_SUBSAMPLE_STABILITY,
        OUTPUT_REDISCOVERY,
        OUTPUT_RANDOM_CONTROLS,
        OUTPUT_README,
    ]
    create_manifest(
        input_paths=[
            EXPRESSION_FILE,
            FROZEN_WEIGHTS_FILE,
            SCRIPT47_LOCK_FILE,
            SCRIPT47_MANIFEST_FILE,
        ],
        output_paths=output_paths,
        r_cut=r_cut,
    )

    print("")
    print("Saved:")
    for path in output_paths + [OUTPUT_MANIFEST]:
        print(path)
    print("Done.")


if __name__ == "__main__":
    main()
