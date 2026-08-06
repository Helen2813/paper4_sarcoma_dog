from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

SCRIPT_VERSION = "25-gse39055-preparation-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"

GSE_ACCESSION = "GSE39055"
GSE_RAW_DIR = DATA_RAW_DIR / "human_GSE39055"
HUMAN_PROCESSED_DIR = PROCESSED_DIR / "human_validation"

for directory in [GSE_RAW_DIR, HUMAN_PROCESSED_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

FREEZE_JSON_FILE = RESULTS_DIR / "GSE238110_frozen_transfer_program_freeze.json"
FROZEN_MANIFEST_FILE = RESULTS_DIR / "GSE238110_frozen_canine_transfer_program_manifest.csv"
STRICT_WEIGHTS_FILE = RESULTS_DIR / "GSE238110_frozen_transfer_gene_weights_strict.csv"
BROAD_WEIGHTS_FILE = RESULTS_DIR / "GSE238110_frozen_transfer_gene_weights_broad.csv"
SCORING_SPEC_FILE = RESULTS_DIR / "GSE238110_frozen_transfer_scoring_specification.csv"
PROLIFERATION_MAPPING_FILE = RESULTS_DIR / "frozen_strict_human_proliferation_mapping.csv"

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]
SECONDARY_MODULES = ["M28", "M38", "M25", "M17"]

MIN_SCORE_GENES = 3
MIN_SCORE_FRACTION = 0.50
MIN_PROLIFERATION_GENES = 20
MIN_PROLIFERATION_FRACTION = 0.30
RANDOM_SEED = 42

PROLIFERATION_ANCHOR_SYMBOLS = [
    "PCNA", "MKI67", "TOP2A", "BIRC5", "UBE2C", "UBE2S", "AURKA",
    "AURKB", "CDC20", "CDC6", "CDK1", "CCNA2", "CCNB1", "CCNB2",
    "MCM2", "MCM4", "MCM5", "MCM10", "TYMS", "RRM2", "TK1", "PLK1",
    "PLK4", "CENPA", "CENPE", "CENPF", "CENPK", "CENPV", "KIF11",
    "KIF15", "KIF18B", "KIF23", "MELK", "MYBL2", "BUB1", "BUB1B",
    "DLGAP5", "SPAG5", "STMN1",
]

OUTPUT_EXPRESSION = HUMAN_PROCESSED_DIR / "GSE39055_expression_gene_symbol.csv"
OUTPUT_CLINICAL = HUMAN_PROCESSED_DIR / "GSE39055_clinical_standardized.csv"
OUTPUT_SCORES = HUMAN_PROCESSED_DIR / "GSE39055_frozen_transfer_scores.csv"
OUTPUT_COVERAGE = RESULTS_DIR / "GSE39055_frozen_transfer_score_coverage.csv"
OUTPUT_PROBE_MAP = RESULTS_DIR / "GSE39055_probe_to_gene_symbol_selected.csv"
OUTPUT_PHENOTYPE_RAW = RESULTS_DIR / "GSE39055_GEO_phenotype_raw.csv"
OUTPUT_PREPARATION_SUMMARY = RESULTS_DIR / "GSE39055_preparation_summary.csv"
OUTPUT_PREPARATION_MANIFEST = RESULTS_DIR / "GSE39055_preparation_manifest.json"
OUTPUT_README = RESULTS_DIR / "GSE39055_preparation_README.txt"


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


def verify_frozen_inputs() -> dict[str, Any]:
    if not FREEZE_JSON_FILE.exists():
        raise FileNotFoundError(
            f"Frozen-program manifest is missing: {FREEZE_JSON_FILE}"
        )

    freeze = json.loads(FREEZE_JSON_FILE.read_text(encoding="utf-8"))
    recorded = freeze.get("files", {})
    required = [
        FROZEN_MANIFEST_FILE,
        STRICT_WEIGHTS_FILE,
        BROAD_WEIGHTS_FILE,
        SCORING_SPEC_FILE,
    ]

    print("")
    print("Frozen input integrity check:")
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Frozen input is missing: {path}")
        expected = recorded.get(path.name, {}).get("sha256")
        observed = sha256_file(path)
        if expected and observed != expected:
            raise RuntimeError(
                f"Frozen input hash mismatch for {path.name}. "
                "Do not continue after modifying a frozen file."
            )
        print(f"  {path.name}: {'verified' if expected else 'present_without_recorded_hash'}")

    strict = pd.read_csv(STRICT_WEIGHTS_FILE)
    primary_counts = (
        strict[strict["module_label"].isin(PRIMARY_MODULES)]
        .groupby("module_label")["human_gene_symbol"]
        .nunique()
    )
    insufficient = [
        module for module in PRIMARY_MODULES if primary_counts.get(module, 0) < 3
    ]
    if insufficient:
        raise RuntimeError(
            "Primary modules have insufficient strict genes: "
            + ", ".join(insufficient)
        )

    print("  Primary strict gene counts:")
    print(primary_counts.to_string())
    return freeze


def import_geoparse():
    try:
        import GEOparse  # type: ignore
        return GEOparse
    except ImportError as exc:
        raise ImportError(
            "GEOparse is required. Install it inside the project venv with: "
            "python -m pip install GEOparse"
        ) from exc


def detect_platform_columns(platform: pd.DataFrame) -> tuple[str, str]:
    probe_candidates = ["ID", "ID_REF", "ProbeID", "PROBE_ID", "SPOT_ID"]
    probe_col = next(
        (column for column in probe_candidates if column in platform.columns),
        platform.columns[0],
    )

    exact_symbol_candidates = [
        "Symbol", "SYMBOL", "Gene Symbol", "GENE_SYMBOL", "Gene symbol",
        "ILMN_Gene", "gene_assignment",
    ]
    symbol_col = next(
        (column for column in exact_symbol_candidates if column in platform.columns),
        None,
    )
    if symbol_col is None:
        candidates = [
            column for column in platform.columns
            if "symbol" in str(column).lower()
        ]
        symbol_col = candidates[0] if candidates else None
    if symbol_col is None:
        raise ValueError(
            "Could not detect a gene-symbol column in GPL14951. "
            f"Columns: {list(platform.columns)}"
        )
    return str(probe_col), str(symbol_col)


def normalize_unambiguous_gene_symbol(value: Any) -> str:
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NA", "---", "NONE", "NULL"}:
        return ""
    parts = [
        item.strip()
        for item in re.split(r"\s*///\s*|\s*//\s*|\s*;\s*|\s*,\s*", text)
        if item.strip() and item.strip() not in {"---", "NA"}
    ]
    unique = list(dict.fromkeys(parts))
    if len(unique) != 1:
        return ""
    symbol = unique[0]
    if " " in symbol:
        return ""
    return symbol


def metadata_as_text(metadata: dict[str, Any]) -> str:
    pieces: list[str] = []
    for key, value in metadata.items():
        if isinstance(value, list):
            for item in value:
                pieces.append(f"{key}: {item}")
        else:
            pieces.append(f"{key}: {value}")
    return " | ".join(pieces)


def parse_characteristics(metadata: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    values = metadata.get("characteristics_ch1", [])
    if not isinstance(values, list):
        values = [values]
    for item in values:
        text = str(item).strip()
        if ":" not in text:
            continue
        key, value = text.split(":", 1)
        result[key.strip().lower()] = value.strip()
    return result


def numeric_from_text(value: Any) -> float:
    match = re.search(r"-?[0-9]+(?:\.[0-9]+)?", str(value))
    return float(match.group(0)) if match else np.nan


def binary_yes_no(value: Any) -> float:
    text = str(value).strip().lower()
    if text in {"y", "yes", "1", "true", "positive"}:
        return 1.0
    if text in {"n", "no", "0", "false", "negative"}:
        return 0.0
    return np.nan


def parse_percent_necrosis(value: Any) -> tuple[float, float]:
    text = str(value).strip().lower()
    number = numeric_from_text(text)
    if not np.isfinite(number):
        return np.nan, np.nan
    good_response = 1.0 if number >= 90 or text.startswith(">90") else 0.0
    return number, good_response


def prepare_gse39055() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    GEOparse = import_geoparse()

    print("")
    print("Downloading/loading GSE39055 from NCBI GEO:")
    gse = GEOparse.get_GEO(
        geo=GSE_ACCESSION,
        destdir=str(GSE_RAW_DIR),
        silent=False,
    )

    expression_probe_by_sample = gse.pivot_samples("VALUE")
    if expression_probe_by_sample.empty:
        raise RuntimeError("GSE39055 expression matrix is empty.")
    expression_probe_by_sample.index = expression_probe_by_sample.index.astype(str)

    phenotype = gse.phenotype_data.copy()
    phenotype.index = phenotype.index.astype(str)
    phenotype.to_csv(OUTPUT_PHENOTYPE_RAW)

    if not gse.gpls:
        raise RuntimeError("GSE39055 platform annotation was not loaded.")
    gpl_name = sorted(gse.gpls.keys())[0]
    platform = gse.gpls[gpl_name].table.copy()
    probe_col, symbol_col = detect_platform_columns(platform)

    annotation = platform[[probe_col, symbol_col]].copy()
    annotation[probe_col] = annotation[probe_col].astype(str)
    annotation["gene_symbol"] = annotation[symbol_col].map(
        normalize_unambiguous_gene_symbol
    )
    annotation = annotation[annotation["gene_symbol"].ne("")].copy()
    annotation = annotation.drop_duplicates(probe_col, keep="first")

    common_probes = expression_probe_by_sample.index.intersection(
        annotation[probe_col]
    )
    if len(common_probes) == 0:
        raise RuntimeError("No GSE39055 probes matched GPL14951 annotation.")

    expr = expression_probe_by_sample.loc[common_probes].apply(
        pd.to_numeric, errors="coerce"
    )
    ann = annotation.set_index(probe_col).loc[common_probes]
    probe_variance = expr.var(axis=1)

    probe_map = pd.DataFrame(
        {
            "probe_id": common_probes,
            "gene_symbol": ann["gene_symbol"].values,
            "probe_variance": probe_variance.loc[common_probes].values,
        }
    )
    probe_map = probe_map.sort_values(
        ["gene_symbol", "probe_variance", "probe_id"],
        ascending=[True, False, True],
    )
    selected_map = probe_map.drop_duplicates("gene_symbol", keep="first")
    selected_map.to_csv(OUTPUT_PROBE_MAP, index=False)

    selected_expr = expr.loc[selected_map["probe_id"].tolist()].copy()
    selected_expr.index = selected_map.set_index("probe_id").loc[
        selected_expr.index, "gene_symbol"
    ]
    expression = selected_expr.T
    expression.index.name = "geo_sample_id"
    expression = expression.loc[:, ~expression.columns.duplicated()].copy()
    expression = expression.loc[:, expression.var(axis=0) > 0]

    clinical_rows = []
    for sample_id, gsm in gse.gsms.items():
        characteristics = parse_characteristics(gsm.metadata)
        necrosis_value, good_necrosis = parse_percent_necrosis(
            characteristics.get("percent necrosis", "")
        )
        recurrence_event = binary_yes_no(
            characteristics.get("recurrence", "")
        )
        death_event = binary_yes_no(characteristics.get("death", ""))
        rfs_time = numeric_from_text(
            characteristics.get(
                "time until first recurrence or latest follow-up (months)",
                "",
            )
        )
        age_years = numeric_from_text(characteristics.get("age", ""))

        clinical_rows.append(
            {
                "geo_sample_id": sample_id,
                "title": str(gsm.metadata.get("title", [""])[0]),
                "age_years": age_years,
                "sex": characteristics.get("gender", ""),
                "chemotherapy": characteristics.get("chemotherapy", ""),
                "percent_necrosis_raw": characteristics.get(
                    "percent necrosis", ""
                ),
                "percent_necrosis_numeric": necrosis_value,
                "good_necrosis_response_ge90": good_necrosis,
                "recurrence_event": recurrence_event,
                "death_event_descriptive": death_event,
                "rfs_time_months": rfs_time,
                "tissue": characteristics.get("tissue", ""),
                "biopsy_resection_pair": characteristics.get(
                    "biopsy/resection pair", ""
                ),
                "metadata_text_combined": metadata_as_text(gsm.metadata),
                "endpoint_note": (
                    "RFS time is the GEO field 'time until first recurrence or "
                    "latest follow-up (months)'; event is recurrence Y/N. "
                    "Death status is descriptive because no death time is supplied."
                ),
            }
        )

    clinical = pd.DataFrame(clinical_rows).set_index("geo_sample_id")
    common_samples = expression.index.intersection(clinical.index)
    expression = expression.loc[common_samples].copy()
    clinical = clinical.loc[common_samples].copy()

    return expression, clinical, selected_map


def zscore_columns(expression: pd.DataFrame) -> pd.DataFrame:
    x = expression.apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(axis=0))
    stds = x.std(axis=0).replace(0, np.nan)
    z = (x - x.mean(axis=0)) / stds
    return z.loc[:, z.notna().all(axis=0)]


def zscore_series(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    std = values.std()
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=values.index)
    return (values - values.mean()) / std


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    frame = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if frame.shape[0] < 5 or frame["a"].std() == 0 or frame["b"].std() == 0:
        return np.nan
    return float(frame["a"].corr(frame["b"]))


def human_cohort_pc1(
    z: pd.DataFrame,
    reference: pd.Series | None = None,
) -> pd.Series:
    if z.shape[1] < 2:
        return pd.Series(np.nan, index=z.index)
    pca = PCA(n_components=1, random_state=RANDOM_SEED)
    score = pd.Series(pca.fit_transform(z).ravel(), index=z.index)
    if reference is not None:
        corr = safe_corr(score, reference)
        if np.isfinite(corr) and corr < 0:
            score = -score
    return zscore_series(score)


def compute_module_scores(
    expression: pd.DataFrame,
    strict_weights: pd.DataFrame,
    broad_weights: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expression = expression.copy()
    expression.columns = expression.columns.astype(str).str.upper()
    expression = expression.loc[:, ~expression.columns.duplicated()].copy()

    scores = pd.DataFrame(index=expression.index)
    coverage_rows = []

    for mapping_name, weights in [
        ("strict", strict_weights),
        ("broad", broad_weights),
    ]:
        for module_label, part in weights.groupby("module_label"):
            part = part.copy()
            part["human_gene_symbol"] = (
                part["human_gene_symbol"].astype(str).str.upper()
            )
            part = part.drop_duplicates("human_gene_symbol", keep="first")
            requested = part["human_gene_symbol"].tolist()
            available = [gene for gene in requested if gene in expression.columns]

            n_requested = len(requested)
            n_available = len(available)
            fraction = n_available / n_requested if n_requested else 0.0
            passed = (
                n_available >= MIN_SCORE_GENES
                and fraction >= MIN_SCORE_FRACTION
            )

            coverage_rows.append(
                {
                    "cohort": GSE_ACCESSION,
                    "module_label": module_label,
                    "mapping": mapping_name,
                    "n_frozen_genes": n_requested,
                    "n_available_genes": n_available,
                    "coverage_fraction": fraction,
                    "minimum_rule_passed": passed,
                    "available_genes": ";".join(available),
                    "missing_genes": ";".join(
                        gene for gene in requested if gene not in available
                    ),
                }
            )
            if not passed:
                continue

            z = zscore_columns(expression[available])
            available = list(z.columns)
            if len(available) < MIN_SCORE_GENES:
                continue

            weight_indexed = part.set_index("human_gene_symbol").loc[available]
            raw_loadings = pd.to_numeric(
                weight_indexed["risk_oriented_loading"],
                errors="coerce",
            ).fillna(0.0)
            signs = np.sign(raw_loadings).replace(0, 1)

            signed_mean = z.mul(signs, axis=1).mean(axis=1)
            signed_mean = zscore_series(signed_mean)

            if raw_loadings.abs().sum() > 0:
                normalized = raw_loadings / raw_loadings.abs().sum()
                weighted = z.mul(normalized, axis=1).sum(axis=1)
                weighted = zscore_series(weighted)
            else:
                weighted = pd.Series(np.nan, index=z.index)

            pc1 = human_cohort_pc1(z, reference=signed_mean)
            prefix = f"{module_label}__{mapping_name}"
            scores[f"{prefix}__signed_mean_z"] = signed_mean
            scores[f"{prefix}__canine_pca_weighted_z"] = weighted
            scores[f"{prefix}__human_pc1_z"] = pc1

    tier_map = manifest.set_index("module_label")["validation_tier"].to_dict()
    coverage = pd.DataFrame(coverage_rows)
    coverage["validation_tier"] = coverage["module_label"].map(tier_map)
    scores.insert(0, "cohort", GSE_ACCESSION)
    return scores, coverage


def compute_proliferation_scores(
    expression: pd.DataFrame,
    proliferation_mapping: pd.DataFrame,
    strict_weights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expression = expression.copy()
    expression.columns = expression.columns.astype(str).str.upper()
    expression = expression.loc[:, ~expression.columns.duplicated()].copy()

    frozen_genes = (
        proliferation_mapping["human_gene_symbol"]
        .dropna()
        .astype(str)
        .str.upper()
        .drop_duplicates()
        .tolist()
    )
    available = [gene for gene in frozen_genes if gene in expression.columns]
    fraction = len(available) / len(frozen_genes) if frozen_genes else 0.0
    passed = (
        len(available) >= MIN_PROLIFERATION_GENES
        and fraction >= MIN_PROLIFERATION_FRACTION
    )

    coverage_rows = [
        {
            "cohort": GSE_ACCESSION,
            "score": "strict_human_meta_proliferation_pc1",
            "n_frozen_genes": len(frozen_genes),
            "n_available_genes": len(available),
            "coverage_fraction": fraction,
            "minimum_rule_passed": passed,
        }
    ]

    scores = pd.DataFrame(index=expression.index)
    if not passed:
        return scores, pd.DataFrame(coverage_rows)

    z = zscore_columns(expression[available])
    anchors = [gene for gene in PROLIFERATION_ANCHOR_SYMBOLS if gene in z.columns]
    reference = z[anchors].mean(axis=1) if len(anchors) >= 3 else z.mean(axis=1)
    proliferation_pc1 = human_cohort_pc1(z, reference=reference)
    scores["strict_human_meta_proliferation_pc1_z"] = proliferation_pc1

    m40_genes = (
        strict_weights[strict_weights["module_label"].eq("M40")]
        ["human_gene_symbol"]
        .dropna()
        .astype(str)
        .str.upper()
        .drop_duplicates()
        .tolist()
    )
    disjoint_frozen = [
        gene for gene in frozen_genes if gene not in set(m40_genes)
    ]
    disjoint_available = [
        gene for gene in disjoint_frozen if gene in expression.columns
    ]
    disjoint_fraction = (
        len(disjoint_available) / len(disjoint_frozen)
        if disjoint_frozen else 0.0
    )
    disjoint_passed = (
        len(disjoint_available) >= MIN_PROLIFERATION_GENES
        and disjoint_fraction >= MIN_PROLIFERATION_FRACTION
    )
    coverage_rows.append(
        {
            "cohort": GSE_ACCESSION,
            "score": "M40_disjoint_strict_human_meta_proliferation_pc1",
            "n_frozen_genes": len(disjoint_frozen),
            "n_available_genes": len(disjoint_available),
            "coverage_fraction": disjoint_fraction,
            "minimum_rule_passed": disjoint_passed,
        }
    )

    if disjoint_passed:
        z_disjoint = zscore_columns(expression[disjoint_available])
        anchors_disjoint = [
            gene for gene in PROLIFERATION_ANCHOR_SYMBOLS
            if gene in z_disjoint.columns
        ]
        reference_disjoint = (
            z_disjoint[anchors_disjoint].mean(axis=1)
            if len(anchors_disjoint) >= 3
            else z_disjoint.mean(axis=1)
        )
        scores["M40_disjoint_strict_human_meta_proliferation_pc1_z"] = (
            human_cohort_pc1(z_disjoint, reference=reference_disjoint)
        )

    return scores, pd.DataFrame(coverage_rows)


def residualize_outcome_blind(
    score: pd.Series,
    covariate: pd.Series,
) -> pd.Series:
    frame = pd.concat(
        [score.rename("score"), covariate.rename("covariate")],
        axis=1,
    ).dropna()
    residual = pd.Series(np.nan, index=score.index)
    if frame.shape[0] < 10 or frame["covariate"].std() == 0:
        return residual

    x = np.column_stack(
        [np.ones(frame.shape[0]), frame["covariate"].values]
    )
    beta, _, _, _ = np.linalg.lstsq(
        x,
        frame["score"].values,
        rcond=None,
    )
    residual.loc[frame.index] = frame["score"].values - x @ beta
    return zscore_series(residual)


def merge_score_components(
    module_scores: pd.DataFrame,
    proliferation_scores: pd.DataFrame,
) -> pd.DataFrame:
    cohort = module_scores["cohort"]
    out = module_scores.drop(columns=["cohort"]).join(
        proliferation_scores,
        how="outer",
    )

    proliferation_col = (
        "M40_disjoint_strict_human_meta_proliferation_pc1_z"
    )
    if proliferation_col in out.columns:
        for module_col in [
            "M40__strict__signed_mean_z",
            "M40__strict__canine_pca_weighted_z",
        ]:
            if module_col not in out.columns:
                continue
            residual_col = module_col.replace(
                "_z",
                "__residual_to_disjoint_proliferation_z",
            )
            out[residual_col] = residualize_outcome_blind(
                out[module_col],
                out[proliferation_col],
            )

    out.insert(0, "cohort", cohort.reindex(out.index))
    return out


def create_manifest(
    freeze: dict[str, Any],
    expression: pd.DataFrame,
    clinical: pd.DataFrame,
    scores: pd.DataFrame,
) -> None:
    output_paths = [
        OUTPUT_EXPRESSION,
        OUTPUT_CLINICAL,
        OUTPUT_SCORES,
        OUTPUT_COVERAGE,
        OUTPUT_PROBE_MAP,
        OUTPUT_PHENOTYPE_RAW,
        OUTPUT_PREPARATION_SUMMARY,
        OUTPUT_README,
    ]
    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "frozen_program_freeze_sha256": sha256_file(FREEZE_JSON_FILE),
        "frozen_program_definition": freeze,
        "source": {
            "accession": GSE_ACCESSION,
            "source": "NCBI GEO via GEOparse",
            "platform": "GPL14951",
            "sample_type": "diagnostic osteosarcoma biopsy",
        },
        "cohort_dimensions": {
            "expression": list(expression.shape),
            "clinical": list(clinical.shape),
            "scores": list(scores.shape),
        },
        "outcome_guardrail": (
            "Recurrence and follow-up fields were parsed after expression "
            "harmonization rules were fixed. Outcomes were not used to select "
            "probes, genes, weights, score direction, or validation tier."
        ),
        "files": {},
    }
    for path in output_paths:
        if path.exists():
            manifest["files"][path.name] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }

    OUTPUT_PREPARATION_MANIFEST.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def write_readme() -> None:
    text = f"""GSE39055 third human osteosarcoma cohort preparation
Script version: {SCRIPT_VERSION}

Purpose
-------
1. Download and parse GSE39055 diagnostic-biopsy expression data.
2. Collapse probes to unambiguous gene symbols using highest probe variance.
3. Parse recurrence-free survival metadata without outcome-guided feature processing.
4. Construct the frozen canine-to-human module scores.
5. Defer all outcome association testing to the next script.

Primary future endpoint
-----------------------
Recurrence-free survival:
- time: GEO field 'time until first recurrence or latest follow-up (months)'
- event: recurrence Y/N

Important limitation
--------------------
Death status is available, but a separate time-to-death field is not supplied.
Therefore death is not incorporated into the primary time-to-event definition.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("Prepare GSE39055 third human osteosarcoma cohort")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Human processed directory: {HUMAN_PROCESSED_DIR}")
    print("")
    print("Design:")
    print("  Verify frozen canine transfer assets and hashes.")
    print("  Download GSE39055 expression and phenotype metadata from NCBI GEO.")
    print("  Collapse probes to human gene symbols using outcome-blind variance rules.")
    print("  Parse recurrence-free survival metadata.")
    print("  Construct frozen module scores without fitting any outcome model.")
    print("")

    freeze = verify_frozen_inputs()
    manifest = read_required_csv(FROZEN_MANIFEST_FILE)
    strict_weights = read_required_csv(STRICT_WEIGHTS_FILE)
    broad_weights = read_required_csv(BROAD_WEIGHTS_FILE)
    proliferation_mapping = read_required_csv(PROLIFERATION_MAPPING_FILE)

    strict_weights["human_gene_symbol"] = (
        strict_weights["human_gene_symbol"].astype(str).str.upper()
    )
    broad_weights["human_gene_symbol"] = (
        broad_weights["human_gene_symbol"].astype(str).str.upper()
    )
    proliferation_mapping["human_gene_symbol"] = (
        proliferation_mapping["human_gene_symbol"].astype(str).str.upper()
    )

    expression, clinical, _ = prepare_gse39055()

    module_scores, module_coverage = compute_module_scores(
        expression=expression,
        strict_weights=strict_weights,
        broad_weights=broad_weights,
        manifest=manifest,
    )
    proliferation_scores, proliferation_coverage = compute_proliferation_scores(
        expression=expression,
        proliferation_mapping=proliferation_mapping,
        strict_weights=strict_weights,
    )
    scores = merge_score_components(
        module_scores=module_scores,
        proliferation_scores=proliferation_scores,
    )

    coverage = pd.concat(
        [module_coverage, proliferation_coverage],
        axis=0,
        ignore_index=True,
        sort=False,
    )

    expression.to_csv(OUTPUT_EXPRESSION)
    clinical.to_csv(OUTPUT_CLINICAL)
    scores.to_csv(OUTPUT_SCORES)
    coverage.to_csv(OUTPUT_COVERAGE, index=False)

    rfs_complete = clinical[["rfs_time_months", "recurrence_event"]].dropna()
    summary = pd.DataFrame(
        [
            {
                "cohort": GSE_ACCESSION,
                "n_expression_samples": expression.shape[0],
                "n_expression_genes": expression.shape[1],
                "n_clinical_rows": clinical.shape[0],
                "n_rfs_complete": rfs_complete.shape[0],
                "n_recurrence_events": int(rfs_complete["recurrence_event"].sum()),
                "n_censored_without_recurrence": int(
                    (rfs_complete["recurrence_event"] == 0).sum()
                ),
                "n_frozen_score_columns": (
                    scores.shape[1] - int("cohort" in scores.columns)
                ),
            }
        ]
    )
    summary.to_csv(OUTPUT_PREPARATION_SUMMARY, index=False)

    write_readme()
    create_manifest(
        freeze=freeze,
        expression=expression,
        clinical=clinical,
        scores=scores,
    )

    print("")
    print("=" * 80)
    print("GSE39055 preparation summary")
    print("=" * 80)
    print(summary.to_string(index=False))

    print("")
    print("=" * 80)
    print("Frozen score coverage: GSE39055")
    print("=" * 80)
    display_cols = [
        "module_label",
        "mapping",
        "validation_tier",
        "n_frozen_genes",
        "n_available_genes",
        "coverage_fraction",
        "minimum_rule_passed",
    ]
    display_cols = [
        column for column in display_cols if column in module_coverage.columns
    ]
    print(
        module_coverage[display_cols]
        .sort_values(["validation_tier", "module_label", "mapping"])
        .to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Endpoint preparation audit")
    print("=" * 80)
    print("Recurrence event counts:")
    print(clinical["recurrence_event"].value_counts(dropna=False).to_string())
    print("")
    print("RFS fields:")
    print(
        clinical[["rfs_time_months", "recurrence_event"]]
        .agg(["count", "min", "median", "max"])
        .to_string()
    )
    print("")
    print("Clinical covariate availability:")
    print(
        clinical[
            [
                "age_years",
                "sex",
                "percent_necrosis_numeric",
                "good_necrosis_response_ge90",
                "death_event_descriptive",
            ]
        ]
        .agg(["count"])
        .to_string()
    )

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("No GSE39055 outcome was used to select probes, genes, weights, score direction, or validation tier.")
    print("Probe collapsing used the highest-variance probe per unambiguous gene symbol.")
    print("The primary future endpoint is recurrence-free survival using recurrence Y/N and the provided recurrence/follow-up time.")
    print("Death status is descriptive because no separate time-to-death field is provided.")
    print("Outcome association testing and multiplicity control are deferred to script 26.")

    print("")
    print("Saved:")
    for path in [
        OUTPUT_EXPRESSION,
        OUTPUT_CLINICAL,
        OUTPUT_SCORES,
        OUTPUT_COVERAGE,
        OUTPUT_PROBE_MAP,
        OUTPUT_PHENOTYPE_RAW,
        OUTPUT_PREPARATION_SUMMARY,
        OUTPUT_PREPARATION_MANIFEST,
        OUTPUT_README,
    ]:
        print(path)
    print("Done.")


if __name__ == "__main__":
    main()
