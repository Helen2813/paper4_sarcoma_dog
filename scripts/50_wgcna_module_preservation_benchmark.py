from pathlib import Path
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

SCRIPT_VERSION = "50-wgcna-module-preservation-benchmark-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_EXPRESSION_FILE = (
    PROCESSED_DIR / "GSE238110_DOG2_expression_log2cpm_matched_allgenes.csv"
)
TEST_EXPRESSION_FILE = (
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
SCRIPT49_REDISCOVERY_FILE = (
    RESULTS_DIR / "GSE239948_blind_frozen_program_rediscovery.csv"
)
SCRIPT49_MANIFEST_FILE = (
    RESULTS_DIR / "GSE239948_blind_de_novo_rediscovery_manifest.json"
)

OUTPUT_INPUT_UNIVERSE = (
    RESULTS_DIR / "GSE239948_WGCNA_module_preservation_input_universe.csv"
)
OUTPUT_WGCNA_RESULTS = (
    RESULTS_DIR / "GSE239948_WGCNA_module_preservation_signed.csv"
)
OUTPUT_METHOD_CONCORDANCE = (
    RESULTS_DIR / "paper4_GSE239948_preservation_method_concordance.csv"
)
OUTPUT_README = (
    RESULTS_DIR / "GSE239948_WGCNA_module_preservation_README.txt"
)
OUTPUT_MANIFEST = (
    RESULTS_DIR / "GSE239948_WGCNA_module_preservation_manifest.json"
)
OUTPUT_R_LOG = (
    RESULTS_DIR / "GSE239948_WGCNA_module_preservation_R.log"
)

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]
MODULE_COLORS = {
    "M34": "turquoise",
    "M11": "blue",
    "M24": "brown",
    "M40": "yellow",
}
COLOR_TO_MODULE = {value: key for key, value in MODULE_COLORS.items()}

BACKGROUND_GENES = 3000
N_PERMUTATIONS = 200
RANDOM_SEED = 20260806
NETWORK_TYPE = "signed"


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
    variances = x.var(axis=0, ddof=1)
    keep_cols = variances.index[(variances > 0) & np.isfinite(variances)]
    return x.loc[:, keep_cols].copy()


def find_canine_gene_column(weights: pd.DataFrame) -> str:
    for column in ["canine_gene_symbol", "canine_gene", "gene"]:
        if column in weights.columns:
            return column
    raise ValueError("No canine gene-symbol column found in frozen weights.")


def frozen_gene_sets(weights: pd.DataFrame) -> dict[str, set[str]]:
    if "module_label" not in weights.columns:
        raise ValueError("Frozen weights do not contain module_label.")

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


def variance_percentiles(expression: pd.DataFrame) -> pd.Series:
    values = expression.var(axis=0, ddof=1)
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    return values.rank(method="average", pct=True)


def build_benchmark_universe(
    reference: pd.DataFrame,
    test: pd.DataFrame,
    frozen: dict[str, set[str]],
) -> tuple[list[str], pd.DataFrame]:
    common = reference.columns.intersection(test.columns)
    if len(common) < 100:
        raise RuntimeError("Too few common genes for WGCNA preservation benchmark.")

    ref_common = reference.loc[:, common]
    test_common = test.loc[:, common]

    ref_pct = variance_percentiles(ref_common).reindex(common)
    test_pct = variance_percentiles(test_common).reindex(common)
    combined_pct = pd.concat(
        [ref_pct.rename("reference_variance_percentile"),
         test_pct.rename("test_variance_percentile")],
        axis=1,
    )
    combined_pct["combined_variance_percentile"] = combined_pct.mean(axis=1)

    all_frozen = set().union(*frozen.values())
    shared_frozen = sorted(all_frozen & set(common))

    background_candidates = combined_pct.loc[
        ~combined_pct.index.isin(all_frozen)
    ].sort_values("combined_variance_percentile", ascending=False)

    n_background = min(BACKGROUND_GENES, background_candidates.shape[0])
    background = background_candidates.head(n_background).index.tolist()
    universe = sorted(set(shared_frozen) | set(background))

    module_lookup = {}
    for module, genes in frozen.items():
        for gene in genes:
            module_lookup[gene] = module

    audit = combined_pct.reindex(universe).copy()
    audit.index.name = "gene_symbol"
    audit = audit.reset_index()
    audit["frozen_module_label"] = audit["gene_symbol"].map(module_lookup).fillna("")
    audit["wgcna_color"] = audit["frozen_module_label"].map(MODULE_COLORS).fillna("grey")
    audit["is_frozen_primary_gene"] = audit["frozen_module_label"].ne("")
    audit["is_background_gene"] = ~audit["is_frozen_primary_gene"]
    return universe, audit


def verify_manifest_version(path: Path, expected_version: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    observed = payload.get("script_version", "")
    if observed != expected_version:
        raise RuntimeError(
            f"Unexpected manifest version for {path.name}: {observed}. "
            f"Expected: {expected_version}"
        )
    return payload


def find_rscript() -> str | None:
    direct = shutil.which("Rscript")
    if direct:
        return direct

    candidates = []
    if os.name == "nt":
        candidates.extend(
            glob.glob(r"C:\Program Files\R\R-*\bin\Rscript.exe")
        )
        candidates.extend(
            glob.glob(r"C:\Program Files\R\R-*\bin\x64\Rscript.exe")
        )
    else:
        candidates.extend(["/usr/bin/Rscript", "/usr/local/bin/Rscript"])

    existing = [path for path in candidates if Path(path).exists()]
    if not existing:
        return None
    return sorted(existing)[-1]


def r_quote(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "\\'")


def build_r_script(
    reference_csv: Path,
    test_csv: Path,
    color_csv: Path,
    output_csv: Path,
) -> str:
    return f"""
suppressPackageStartupMessages({{
  if (!requireNamespace("WGCNA", quietly = TRUE)) {{
    stop("R package WGCNA is not installed.")
  }}
  library(WGCNA)
}})
options(stringsAsFactors = FALSE)

read_expr <- function(path) {{
  dat <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  rownames(dat) <- dat$sample_id
  dat$sample_id <- NULL
  dat <- as.data.frame(lapply(dat, as.numeric), check.names = FALSE)
  dat
}}

reference_data <- read_expr('{r_quote(reference_csv)}')
test_data <- read_expr('{r_quote(test_csv)}')
colors_df <- read.csv('{r_quote(color_csv)}', check.names = FALSE, stringsAsFactors = FALSE)

if (!identical(colnames(reference_data), colnames(test_data))) {{
  stop("Reference and test gene columns are not identical.")
}}

color_map <- setNames(colors_df$wgcna_color, colors_df$gene_symbol)
module_colors <- unname(color_map[colnames(reference_data)])
if (any(is.na(module_colors))) {{
  stop("Missing WGCNA color assignments for benchmark genes.")
}}

multi_data <- list(
  DOG2 = list(data = reference_data),
  GSE239948 = list(data = test_data)
)
multi_color <- list(DOG2 = module_colors)

set.seed({RANDOM_SEED})
mp <- modulePreservation(
  multiData = multi_data,
  multiColor = multi_color,
  dataIsExpr = TRUE,
  networkType = "{NETWORK_TYPE}",
  corFnc = "cor",
  corOptions = "use = 'p'",
  referenceNetworks = 1,
  testNetworks = 2,
  nPermutations = {N_PERMUTATIONS},
  includekMEallInSummary = FALSE,
  restrictSummaryForGeneralNetworks = TRUE,
  calculateQvalue = FALSE,
  randomSeed = {RANDOM_SEED},
  maxGoldModuleSize = 500,
  maxModuleSize = 1000,
  quickCor = 0,
  calculateCor.kIMall = FALSE,
  calculateClusterCoeff = FALSE,
  useInterpolation = FALSE,
  checkData = TRUE,
  parallelCalculation = FALSE,
  verbose = 2
)

observed <- mp$preservation$observed[[1]][[2]]
zstats <- mp$preservation$Z[[1]][[2]]

common_rows <- intersect(rownames(observed), rownames(zstats))
observed <- observed[common_rows, , drop = FALSE]
zstats <- zstats[common_rows, , drop = FALSE]

pick_col <- function(df, name) {{
  if (name %in% colnames(df)) return(df[[name]])
  rep(NA_real_, nrow(df))
}}

out <- data.frame(
  wgcna_color = common_rows,
  module_size = pick_col(observed, "moduleSize"),
  median_rank_pres = pick_col(observed, "medianRank.pres"),
  zsummary_pres = pick_col(zstats, "Zsummary.pres"),
  zdensity_pres = pick_col(zstats, "Zdensity.pres"),
  zconnectivity_pres = pick_col(zstats, "Zconnectivity.pres"),
  cor_kim = pick_col(observed, "cor.kIM"),
  cor_kme = pick_col(observed, "cor.kME"),
  mean_cor = pick_col(observed, "meanCor"),
  stringsAsFactors = FALSE
)

write.csv(out, '{r_quote(output_csv)}', row.names = FALSE)
"""


def run_wgcna(rscript: str, r_file: Path) -> None:
    command = [rscript, str(r_file)]
    print("")
    print("Running WGCNA modulePreservation in R:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    print("")

    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PROJECT_ROOT),
    )
    OUTPUT_R_LOG.write_text(process.stdout, encoding="utf-8")
    print(process.stdout)

    if process.returncode != 0:
        if "WGCNA is not installed" in process.stdout:
            print("")
            print("WGCNA is not installed in R.")
            print("Install it once with:")
            print(
                f'"{rscript}" -e "install.packages(\'WGCNA\', '
                'repos=\'https://cloud.r-project.org\', dependencies=TRUE)"'
            )
        raise RuntimeError(
            f"R WGCNA benchmark failed with exit code {process.returncode}. "
            f"See: {OUTPUT_R_LOG}"
        )


def classify_zsummary(value) -> str:
    if not np.isfinite(value):
        return "not_estimable"
    if value >= 10:
        return "strong_wgcna_preservation"
    if value >= 2:
        return "moderate_wgcna_preservation"
    return "no_clear_wgcna_preservation"


def parse_wgcna_results(raw: pd.DataFrame) -> pd.DataFrame:
    result = raw.copy()
    result["module_label"] = result["wgcna_color"].map(COLOR_TO_MODULE)
    result = result[result["module_label"].notna()].copy()

    numeric_cols = [
        "module_size",
        "median_rank_pres",
        "zsummary_pres",
        "zdensity_pres",
        "zconnectivity_pres",
        "cor_kim",
        "cor_kme",
        "mean_cor",
    ]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    result["wgcna_preservation_class"] = result["zsummary_pres"].apply(classify_zsummary)
    result["small_module_guardrail"] = result["module_size"].apply(
        lambda n: "small_module_size_sensitive" if np.isfinite(n) and n < 10 else ""
    )

    order = {module: idx for idx, module in enumerate(PRIMARY_MODULES)}
    result["_order"] = result["module_label"].map(order)
    result = result.sort_values("_order").drop(columns="_order")
    return result


def create_method_concordance(
    wgcna: pd.DataFrame,
    script47: pd.DataFrame,
    script49: pd.DataFrame,
) -> pd.DataFrame:
    left_cols = [
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
    left_cols = [c for c in left_cols if c in script47.columns]

    blind_cols = [
        "module_label",
        "n_frozen_genes_in_blind_discovery_universe",
        "discovery_universe_coverage_fraction",
        "best_discovered_module_id",
        "overlap_genes",
        "frozen_gene_recall_within_discovery_universe",
        "discovered_module_precision",
        "best_match_f1",
        "best_module_subsample_stability_median_jaccard",
        "empirical_max_match_q_bh_4",
        "blind_rediscovery_class",
    ]
    blind_cols = [c for c in blind_cols if c in script49.columns]

    merged = wgcna.merge(script47[left_cols], on="module_label", how="left")
    merged = merged.merge(script49[blind_cols], on="module_label", how="left")

    synthesis = []
    for _, row in merged.iterrows():
        module = str(row["module_label"])
        z = pd.to_numeric(pd.Series([row.get("zsummary_pres")]), errors="coerce").iloc[0]
        custom = str(row.get("external_canine_representation_class", ""))
        blind = str(row.get("blind_rediscovery_class", ""))

        if (
            np.isfinite(z)
            and z >= 10
            and custom == "strong_external_canine_representation_preservation"
            and blind == "strong_blind_independent_rediscovery"
        ):
            label = "concordant_standard_custom_and_blind_support"
        elif (
            np.isfinite(z)
            and z >= 2
            and custom == "strong_external_canine_representation_preservation"
        ):
            label = "concordant_standard_and_custom_preservation_support"
        elif custom == "strong_external_canine_representation_preservation":
            label = "custom_preservation_support_without_strong_wgcna_benchmark"
        elif np.isfinite(z) and z >= 2:
            label = "wgcna_support_without_custom_primary_class"
        else:
            label = "no_concordant_preservation_support"

        if module in {"M11", "M24"}:
            label += "__small_module_caution"
        synthesis.append(label)

    merged["preservation_method_concordance_class"] = synthesis
    return merged


def create_readme(universe_n: int, rscript: str) -> None:
    text = f"""GSE239948 WGCNA module-preservation benchmark
================================================

Script version
--------------
{SCRIPT_VERSION}

Purpose
-------
Run the standard WGCNA modulePreservation() framework as a reviewer-facing benchmark for the frozen canine programs in DOG2 versus GSE239948.

Design
------
- Reference network: DOG2.
- Test network: GSE239948.
- Frozen transferable modules: M34, M11, M24, M40.
- Network type: {NETWORK_TYPE}.
- Permutations: {N_PERMUTATIONS}.
- Benchmark gene universe: {universe_n} shared genes consisting of all shared frozen primary genes plus {BACKGROUND_GENES} outcome-blind high-variance background genes when available.
- No outcome data are loaded.

Interpretation
--------------
WGCNA Zsummary.pres is interpreted using the conventional descriptive thresholds:
- Zsummary.pres < 2: no clear preservation evidence.
- 2 <= Zsummary.pres < 10: moderate preservation evidence.
- Zsummary.pres >= 10: strong preservation evidence.

MedianRank.pres is relative: lower ranks indicate stronger preservation compared with other reference modules and should not be interpreted using a universal cutoff.

Small-module guardrail
----------------------
M11 and M24 contain fewer than 10 shared frozen genes. WGCNA Z statistics are strongly module-size dependent, so low Zsummary values for these modules cannot establish absence of preservation.

Relationship to existing analyses
---------------------------------
This benchmark does not replace the custom ortholog-constrained preservation audit or the blind de novo rediscovery analysis. It provides an independent standard-framework sensitivity analysis based on the WGCNA modulePreservation() implementation.

R executable
------------
{rscript}
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def create_manifest(input_paths: list[Path], output_paths: list[Path], rscript: str) -> None:
    payload = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "outcome_loaded": False,
        "network_type": NETWORK_TYPE,
        "n_permutations": N_PERMUTATIONS,
        "random_seed": RANDOM_SEED,
        "background_genes_requested": BACKGROUND_GENES,
        "rscript": rscript,
        "inputs": {},
        "outputs": {},
        "guardrails": [
            "No outcome data are loaded.",
            "Frozen module definitions are not modified.",
            "WGCNA is used as a benchmark rather than as a replacement for the custom preservation analysis.",
            "Small-module Z statistics are interpreted cautiously because of module-size dependence.",
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

    OUTPUT_MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("WGCNA module-preservation benchmark for GSE239948")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Design:")
    print("  Use DOG2 as the reference expression network.")
    print("  Use GSE239948 as the independent canine test network.")
    print("  Benchmark frozen M34, M11, M24, and M40 with WGCNA modulePreservation().")
    print("  Load no outcome data and modify no frozen definitions.")
    print("")

    verify_manifest_version(
        SCRIPT47_MANIFEST_FILE,
        "47-lock-gse239948-independent-canine-evidence-v2",
    )
    verify_manifest_version(
        SCRIPT49_MANIFEST_FILE,
        "49-gse239948-blind-de-novo-rediscovery-v2",
    )

    reference_raw = read_required_csv(REFERENCE_EXPRESSION_FILE, index_col=0)
    test_raw = read_required_csv(TEST_EXPRESSION_FILE, index_col=0)
    weights = read_required_csv(FROZEN_WEIGHTS_FILE)
    script47 = read_required_csv(SCRIPT47_LOCK_FILE)
    script49 = read_required_csv(SCRIPT49_REDISCOVERY_FILE)

    reference = prepare_expression(reference_raw)
    test = prepare_expression(test_raw)
    frozen = frozen_gene_sets(weights)

    universe, universe_audit = build_benchmark_universe(reference, test, frozen)
    reference_benchmark = reference.loc[:, universe].copy()
    test_benchmark = test.loc[:, universe].copy()

    print("Benchmark data:")
    print(f"  DOG2 samples: {reference_benchmark.shape[0]}")
    print(f"  GSE239948 samples: {test_benchmark.shape[0]}")
    print(f"  Shared benchmark genes: {len(universe)}")
    print("")
    print("Frozen shared gene counts:")
    for module in PRIMARY_MODULES:
        count = int((universe_audit["frozen_module_label"] == module).sum())
        total = len(frozen[module])
        print(f"  {module}: {count}/{total}")

    rscript = find_rscript()
    if rscript is None:
        print("")
        print("Rscript was not found.")
        print("Install R for Windows and ensure Rscript.exe is available in PATH.")
        print("After installing R, install WGCNA once with:")
        print(
            'Rscript -e "install.packages(\'WGCNA\', '
            'repos=\'https://cloud.r-project.org\', dependencies=TRUE)"'
        )
        raise RuntimeError("Rscript is required for the standard WGCNA benchmark.")

    with tempfile.TemporaryDirectory(prefix="paper4_wgcna_") as tmp_dir_text:
        tmp_dir = Path(tmp_dir_text)
        reference_csv = tmp_dir / "reference_expression.csv"
        test_csv = tmp_dir / "test_expression.csv"
        colors_csv = tmp_dir / "module_colors.csv"
        raw_output_csv = tmp_dir / "wgcna_raw_results.csv"
        r_file = tmp_dir / "run_wgcna_module_preservation.R"

        reference_out = reference_benchmark.copy()
        reference_out.insert(0, "sample_id", reference_out.index.astype(str))
        reference_out.to_csv(reference_csv, index=False)

        test_out = test_benchmark.copy()
        test_out.insert(0, "sample_id", test_out.index.astype(str))
        test_out.to_csv(test_csv, index=False)

        universe_audit[["gene_symbol", "wgcna_color"]].to_csv(colors_csv, index=False)
        r_file.write_text(
            build_r_script(reference_csv, test_csv, colors_csv, raw_output_csv),
            encoding="utf-8",
        )

        run_wgcna(rscript, r_file)
        if not raw_output_csv.exists():
            raise RuntimeError("R completed without creating the expected WGCNA result file.")
        raw_wgcna = pd.read_csv(raw_output_csv)

    wgcna = parse_wgcna_results(raw_wgcna)
    concordance = create_method_concordance(wgcna, script47, script49)

    print("")
    print("=" * 80)
    print("Standard WGCNA preservation benchmark")
    print("=" * 80)
    display_cols = [
        "module_label",
        "module_size",
        "zsummary_pres",
        "zdensity_pres",
        "zconnectivity_pres",
        "median_rank_pres",
        "wgcna_preservation_class",
        "small_module_guardrail",
    ]
    print(wgcna[display_cols].to_string(index=False))

    print("")
    print("=" * 80)
    print("Cross-method preservation synthesis")
    print("=" * 80)
    synthesis_cols = [
        "module_label",
        "zsummary_pres",
        "wgcna_preservation_class",
        "edge_spearman",
        "loading_spearman",
        "external_canine_representation_class",
        "best_match_f1",
        "empirical_max_match_q_bh_4",
        "blind_rediscovery_class",
        "preservation_method_concordance_class",
    ]
    synthesis_cols = [c for c in synthesis_cols if c in concordance.columns]
    print(concordance[synthesis_cols].to_string(index=False))

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("WGCNA modulePreservation is a standard-framework benchmark, not a replacement for scripts 46-49.")
    print("Zsummary is module-size dependent; low values for M11 and M24 cannot prove non-preservation.")
    print("MedianRank is relative and should not be interpreted using a universal absolute cutoff.")
    print("No outcome data are loaded and no frozen module definition is altered.")

    universe_audit.to_csv(OUTPUT_INPUT_UNIVERSE, index=False)
    wgcna.to_csv(OUTPUT_WGCNA_RESULTS, index=False)
    concordance.to_csv(OUTPUT_METHOD_CONCORDANCE, index=False)
    create_readme(len(universe), rscript)

    output_paths = [
        OUTPUT_INPUT_UNIVERSE,
        OUTPUT_WGCNA_RESULTS,
        OUTPUT_METHOD_CONCORDANCE,
        OUTPUT_README,
        OUTPUT_R_LOG,
    ]
    create_manifest(
        input_paths=[
            REFERENCE_EXPRESSION_FILE,
            TEST_EXPRESSION_FILE,
            FROZEN_WEIGHTS_FILE,
            SCRIPT47_LOCK_FILE,
            SCRIPT47_MANIFEST_FILE,
            SCRIPT49_REDISCOVERY_FILE,
            SCRIPT49_MANIFEST_FILE,
        ],
        output_paths=output_paths,
        rscript=rscript,
    )

    print("")
    print("Saved:")
    for path in output_paths + [OUTPUT_MANIFEST]:
        print(path)
    print("Done.")


if __name__ == "__main__":
    main()
