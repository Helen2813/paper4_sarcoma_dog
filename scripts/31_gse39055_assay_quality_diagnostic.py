from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

SCRIPT_VERSION = "31-gse39055-assay-quality-diagnostic-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
HUMAN_DIR = PROCESSED_DIR / "human_validation"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"

GSE_RAW_DIR = DATA_RAW_DIR / "human_GSE39055"
SOFT_FILE = GSE_RAW_DIR / "GSE39055_family.soft.gz"

STRICT_WEIGHTS_FILE = (
    RESULTS_DIR / "GSE238110_frozen_transfer_gene_weights_strict.csv"
)
CLINICAL_FILE = HUMAN_DIR / "GSE39055_clinical_standardized.csv"
LOCKED_SCORES_FILE = HUMAN_DIR / "GSE39055_frozen_transfer_scores.csv"
LOCKED_PROBE_MAP_FILE = (
    RESULTS_DIR / "GSE39055_probe_to_gene_symbol_selected.csv"
)
LOCKED_RFS_FILE = (
    RESULTS_DIR / "GSE39055_RFS_primary_frozen_program_validation.csv"
)

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]
CANONICAL_QC_GENES = ["MKI67", "TOP2A", "BIRC5", "UBE2C", "EZR"]

TIME_COL = "rfs_time_months"
EVENT_COL = "recurrence_event"

DETECTION_THRESHOLD_STRICT = 0.01
DETECTION_THRESHOLD_RELAXED = 0.05
MIN_DETECTED_FRACTION_50 = 0.50
MIN_DETECTED_FRACTION_80 = 0.80

COX_PENALIZER = 0.05
RANDOM_SEED = 42

OUTPUT_PROBE_QC = RESULTS_DIR / "GSE39055_probe_assay_quality.csv"
OUTPUT_SELECTION = (
    RESULTS_DIR / "GSE39055_gene_probe_selection_comparison.csv"
)
OUTPUT_MODULE_COVERAGE = (
    RESULTS_DIR / "GSE39055_detection_aware_module_coverage.csv"
)
OUTPUT_SCORES = (
    HUMAN_DIR / "GSE39055_detection_aware_frozen_scores.csv"
)
OUTPUT_RFS = (
    RESULTS_DIR / "GSE39055_detection_aware_RFS_sensitivity.csv"
)
OUTPUT_CANONICAL = (
    RESULTS_DIR / "GSE39055_canonical_gene_assay_audit.csv"
)
OUTPUT_SAMPLE_QC = (
    RESULTS_DIR / "GSE39055_sample_assay_quality.csv"
)
OUTPUT_ENDPOINT_QC = (
    RESULTS_DIR / "GSE39055_assay_quality_endpoint_diagnostics.csv"
)
OUTPUT_SUMMARY = (
    RESULTS_DIR / "GSE39055_assay_quality_module_summary.csv"
)
OUTPUT_README = (
    RESULTS_DIR / "GSE39055_assay_quality_diagnostic_README.txt"
)
OUTPUT_MANIFEST = (
    RESULTS_DIR / "GSE39055_assay_quality_diagnostic_manifest.json"
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


def import_geoparse():
    try:
        import GEOparse  # type: ignore
        return GEOparse
    except ImportError as exc:
        raise ImportError(
            "GEOparse is required. Install it with: "
            "python -m pip install GEOparse"
        ) from exc


def normalize_column_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def find_table_column(
    columns: list[Any],
    exact_candidates: list[str],
    required_tokens: list[str] | None = None,
) -> str | None:
    normalized = {
        normalize_column_name(column): str(column)
        for column in columns
    }

    for candidate in exact_candidates:
        key = normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]

    if required_tokens:
        normalized_tokens = [
            normalize_column_name(token)
            for token in required_tokens
        ]
        for column in columns:
            key = normalize_column_name(column)
            if all(token in key for token in normalized_tokens):
                return str(column)

    return None


def normalize_unambiguous_gene_symbol(value: Any) -> str:
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NA", "---", "NONE", "NULL"}:
        return ""

    parts = [
        item.strip()
        for item in re.split(
            r"\s*///\s*|\s*//\s*|\s*;\s*|\s*,\s*",
            text,
        )
        if item.strip() and item.strip() not in {"---", "NA"}
    ]
    unique = list(dict.fromkeys(parts))
    if len(unique) != 1:
        return ""

    symbol = unique[0]
    if " " in symbol:
        return ""
    return symbol


def detect_platform_columns(platform: pd.DataFrame) -> tuple[str, str]:
    probe_col = find_table_column(
        list(platform.columns),
        exact_candidates=["ID", "ID_REF", "ProbeID", "PROBE_ID"],
    )
    if probe_col is None:
        probe_col = str(platform.columns[0])

    symbol_col = find_table_column(
        list(platform.columns),
        exact_candidates=[
            "Symbol",
            "SYMBOL",
            "Gene Symbol",
            "GENE_SYMBOL",
            "Gene symbol",
            "ILMN_Gene",
        ],
        required_tokens=["symbol"],
    )
    if symbol_col is None:
        raise ValueError(
            "Could not detect a gene-symbol column in GPL14951. "
            f"Columns: {list(platform.columns)}"
        )

    return probe_col, symbol_col


def load_gse39055():
    GEOparse = import_geoparse()

    if SOFT_FILE.exists():
        print(f"Parsing cached SOFT file: {SOFT_FILE}")
        return GEOparse.get_GEO(
            filepath=str(SOFT_FILE),
            silent=False,
        )

    GSE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Cached SOFT file not found; downloading GSE39055.")
    return GEOparse.get_GEO(
        geo="GSE39055",
        destdir=str(GSE_RAW_DIR),
        silent=False,
    )


def extract_probe_matrices(
    gse,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    value_series: dict[str, pd.Series] = {}
    detection_series: dict[str, pd.Series] = {}
    detected_columns: dict[str, str] = {}

    for sample_id, gsm in gse.gsms.items():
        table = gsm.table.copy()

        id_col = find_table_column(
            list(table.columns),
            exact_candidates=["ID_REF", "ID", "ProbeID"],
        )
        value_col = find_table_column(
            list(table.columns),
            exact_candidates=["VALUE"],
        )
        detection_col = find_table_column(
            list(table.columns),
            exact_candidates=[
                "Detection PVal",
                "Detection P Value",
                "Detection_Pval",
                "Detection",
            ],
            required_tokens=["detection", "p"],
        )

        if value_col is None:
            raise ValueError(
                f"VALUE column not found for {sample_id}. "
                f"Columns: {list(table.columns)}"
            )
        if detection_col is None:
            raise ValueError(
                f"Detection P-value column not found for {sample_id}. "
                f"Columns: {list(table.columns)}"
            )

        if id_col is None:
            probe_ids = table.index.astype(str)
        else:
            probe_ids = table[id_col].astype(str)

        values = pd.Series(
            pd.to_numeric(table[value_col], errors="coerce").values,
            index=probe_ids,
            name=sample_id,
        )
        detection = pd.Series(
            pd.to_numeric(table[detection_col], errors="coerce").values,
            index=probe_ids,
            name=sample_id,
        )

        values = values[~values.index.duplicated(keep="first")]
        detection = detection[~detection.index.duplicated(keep="first")]

        value_series[sample_id] = values
        detection_series[sample_id] = detection
        detected_columns[sample_id] = detection_col

    values_probe_by_sample = pd.concat(
        value_series.values(),
        axis=1,
        join="inner",
    )
    values_probe_by_sample.columns = list(value_series.keys())

    detection_probe_by_sample = pd.concat(
        detection_series.values(),
        axis=1,
        join="inner",
    )
    detection_probe_by_sample.columns = list(detection_series.keys())

    common_probes = values_probe_by_sample.index.intersection(
        detection_probe_by_sample.index
    )
    common_samples = values_probe_by_sample.columns.intersection(
        detection_probe_by_sample.columns
    )

    values_probe_by_sample = values_probe_by_sample.loc[
        common_probes,
        common_samples,
    ]
    detection_probe_by_sample = detection_probe_by_sample.loc[
        common_probes,
        common_samples,
    ]

    return (
        values_probe_by_sample,
        detection_probe_by_sample,
        detected_columns,
    )


def build_annotation(gse) -> pd.DataFrame:
    if not gse.gpls:
        raise RuntimeError("No platform annotation was loaded.")

    gpl_name = sorted(gse.gpls.keys())[0]
    platform = gse.gpls[gpl_name].table.copy()
    probe_col, symbol_col = detect_platform_columns(platform)

    annotation = platform[[probe_col, symbol_col]].copy()
    annotation[probe_col] = annotation[probe_col].astype(str)
    annotation["gene_symbol"] = annotation[symbol_col].map(
        normalize_unambiguous_gene_symbol
    )
    annotation = annotation.rename(columns={probe_col: "probe_id"})
    annotation = annotation.drop_duplicates("probe_id", keep="first")
    return annotation[["probe_id", "gene_symbol"]]


def build_probe_quality(
    values_probe_by_sample: pd.DataFrame,
    detection_probe_by_sample: pd.DataFrame,
    annotation: pd.DataFrame,
) -> pd.DataFrame:
    quality = pd.DataFrame(
        {
            "probe_id": values_probe_by_sample.index.astype(str),
            "expression_mean": values_probe_by_sample.mean(axis=1).values,
            "expression_median": values_probe_by_sample.median(axis=1).values,
            "expression_variance": values_probe_by_sample.var(axis=1).values,
            "expression_sd": values_probe_by_sample.std(axis=1).values,
            "detection_p_median": detection_probe_by_sample.median(axis=1).values,
            "detection_p_mean": detection_probe_by_sample.mean(axis=1).values,
            "detected_fraction_p_lt_0_01": (
                detection_probe_by_sample.lt(
                    DETECTION_THRESHOLD_STRICT
                ).mean(axis=1).values
            ),
            "detected_fraction_p_lt_0_05": (
                detection_probe_by_sample.lt(
                    DETECTION_THRESHOLD_RELAXED
                ).mean(axis=1).values
            ),
            "n_samples": values_probe_by_sample.shape[1],
        }
    )

    quality = quality.merge(annotation, on="probe_id", how="left")
    quality["gene_symbol"] = (
        quality["gene_symbol"].fillna("").astype(str).str.upper()
    )
    quality["unambiguous_gene_symbol"] = quality["gene_symbol"].ne("")
    return quality


def select_probes(
    probe_quality: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = probe_quality[
        probe_quality["unambiguous_gene_symbol"]
    ].copy()

    max_variance = eligible.sort_values(
        [
            "gene_symbol",
            "expression_variance",
            "detected_fraction_p_lt_0_01",
            "detection_p_median",
            "probe_id",
        ],
        ascending=[True, False, False, True, True],
    ).drop_duplicates("gene_symbol", keep="first")

    best_detected = eligible.sort_values(
        [
            "gene_symbol",
            "detected_fraction_p_lt_0_01",
            "detected_fraction_p_lt_0_05",
            "detection_p_median",
            "expression_variance",
            "probe_id",
        ],
        ascending=[True, False, False, True, False, True],
    ).drop_duplicates("gene_symbol", keep="first")

    max_columns = {
        column: f"max_variance_{column}"
        for column in max_variance.columns
        if column != "gene_symbol"
    }
    best_columns = {
        column: f"best_detected_{column}"
        for column in best_detected.columns
        if column != "gene_symbol"
    }

    comparison = max_variance.rename(columns=max_columns).merge(
        best_detected.rename(columns=best_columns),
        on="gene_symbol",
        how="outer",
    )
    comparison["same_selected_probe"] = (
        comparison["max_variance_probe_id"]
        == comparison["best_detected_probe_id"]
    )

    return max_variance, best_detected, comparison


def add_locked_probe_map_comparison(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    if not LOCKED_PROBE_MAP_FILE.exists():
        comparison["locked_probe_id"] = np.nan
        comparison["locked_matches_recomputed_max_variance"] = np.nan
        return comparison

    locked = pd.read_csv(LOCKED_PROBE_MAP_FILE)
    locked["gene_symbol"] = (
        locked["gene_symbol"].astype(str).str.upper()
    )
    locked = locked.rename(columns={"probe_id": "locked_probe_id"})

    merged = comparison.merge(
        locked[["gene_symbol", "locked_probe_id"]],
        on="gene_symbol",
        how="left",
    )
    merged["locked_matches_recomputed_max_variance"] = (
        merged["locked_probe_id"]
        == merged["max_variance_probe_id"]
    )
    return merged


def zscore_columns(expression: pd.DataFrame) -> pd.DataFrame:
    x = expression.apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(axis=0))
    std = x.std(axis=0).replace(0, np.nan)
    z = (x - x.mean(axis=0)) / std
    return z.loc[:, z.notna().all(axis=0)]


def zscore_series(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    std = values.std()
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (values - values.mean()) / std


def selection_to_gene_expression(
    selection: pd.DataFrame,
    values_probe_by_sample: pd.DataFrame,
    min_detected_fraction: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = selection.copy()

    if min_detected_fraction is not None:
        selected = selected[
            selected["detected_fraction_p_lt_0_01"].ge(
                min_detected_fraction
            )
        ].copy()

    selected = selected[
        selected["probe_id"].isin(values_probe_by_sample.index)
    ].copy()
    selected = selected.drop_duplicates("gene_symbol", keep="first")

    expression = values_probe_by_sample.loc[
        selected["probe_id"].tolist()
    ].T
    expression.columns = selected["gene_symbol"].tolist()
    expression = expression.loc[
        :,
        ~expression.columns.duplicated(keep="first"),
    ].copy()

    selected = selected.set_index("gene_symbol").loc[
        expression.columns
    ].reset_index()
    return expression, selected


def fixed_direction_c_index(
    time: pd.Series,
    event: pd.Series,
    risk_score: pd.Series,
) -> float:
    frame = pd.concat(
        [
            pd.to_numeric(time, errors="coerce").rename("time"),
            pd.to_numeric(event, errors="coerce").rename("event"),
            pd.to_numeric(risk_score, errors="coerce").rename("score"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if (
        frame.shape[0] < 10
        or frame["event"].sum() < 3
        or frame["event"].nunique() < 2
    ):
        return np.nan

    return float(
        concordance_index(
            frame["time"].values,
            -frame["score"].values,
            frame["event"].values,
        )
    )


def fit_cox_score(
    frame: pd.DataFrame,
    score_col: str,
) -> dict[str, Any]:
    data = frame[
        [TIME_COL, EVENT_COL, score_col]
    ].replace([np.inf, -np.inf], np.nan).dropna().copy()
    data = data[data[TIME_COL].gt(0)].copy()

    result: dict[str, Any] = {
        "n": data.shape[0],
        "events": (
            int(data[EVENT_COL].sum())
            if data.shape[0]
            else 0
        ),
        "hr_per_sd": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "p": np.nan,
        "fixed_direction_c_index": np.nan,
        "direction_concordant_with_canine": np.nan,
        "error": "",
    }

    if (
        data.shape[0] < 20
        or data[EVENT_COL].sum() < 5
        or data[score_col].std() == 0
    ):
        result["error"] = "insufficient_data"
        return result

    result["fixed_direction_c_index"] = fixed_direction_c_index(
        data[TIME_COL],
        data[EVENT_COL],
        data[score_col],
    )

    model = CoxPHFitter(penalizer=COX_PENALIZER)
    try:
        model.fit(
            data,
            duration_col=TIME_COL,
            event_col=EVENT_COL,
            fit_options={"max_steps": 500},
        )
        summary = model.summary.loc[score_col]
        result["hr_per_sd"] = float(summary["exp(coef)"])
        result["ci_low"] = float(
            summary["exp(coef) lower 95%"]
        )
        result["ci_high"] = float(
            summary["exp(coef) upper 95%"]
        )
        result["p"] = float(summary["p"])
        result["direction_concordant_with_canine"] = bool(
            result["hr_per_sd"] > 1.0
        )
    except Exception as exc:
        result["error"] = str(exc)[:500]

    return result


def compute_module_scores(
    strict_weights: pd.DataFrame,
    values_probe_by_sample: pd.DataFrame,
    selection_strategies: dict[str, tuple[pd.DataFrame, float | None]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.DataFrame(index=values_probe_by_sample.columns)
    coverage_rows: list[dict[str, Any]] = []

    weights = strict_weights.copy()
    weights["human_gene_symbol"] = (
        weights["human_gene_symbol"].astype(str).str.upper()
    )

    for strategy_name, (
        selection,
        min_detected_fraction,
    ) in selection_strategies.items():
        gene_expression, selected = selection_to_gene_expression(
            selection=selection,
            values_probe_by_sample=values_probe_by_sample,
            min_detected_fraction=min_detected_fraction,
        )
        selected_index = selected.set_index("gene_symbol")

        for module in PRIMARY_MODULES:
            module_weights = weights[
                weights["module_label"].eq(module)
            ].copy()
            module_weights = module_weights.drop_duplicates(
                "human_gene_symbol",
                keep="first",
            )
            requested_genes = module_weights[
                "human_gene_symbol"
            ].tolist()
            available_genes = [
                gene
                for gene in requested_genes
                if gene in gene_expression.columns
            ]

            n_requested = len(requested_genes)
            n_available = len(available_genes)
            coverage_fraction = (
                n_available / n_requested
                if n_requested
                else np.nan
            )

            detection_values = (
                selected_index.loc[
                    available_genes,
                    "detected_fraction_p_lt_0_01",
                ]
                if available_genes
                else pd.Series(dtype=float)
            )

            coverage_rows.append(
                {
                    "module_label": module,
                    "strategy": strategy_name,
                    "n_frozen_genes": n_requested,
                    "n_available_genes": n_available,
                    "coverage_fraction": coverage_fraction,
                    "median_gene_detected_fraction_p_lt_0_01": (
                        float(detection_values.median())
                        if not detection_values.empty
                        else np.nan
                    ),
                    "minimum_gene_detected_fraction_p_lt_0_01": (
                        float(detection_values.min())
                        if not detection_values.empty
                        else np.nan
                    ),
                    "available_genes": ";".join(available_genes),
                    "missing_or_filtered_genes": ";".join(
                        gene
                        for gene in requested_genes
                        if gene not in available_genes
                    ),
                }
            )

            if n_available < 3:
                continue

            z = zscore_columns(
                gene_expression[available_genes]
            )
            available_genes = list(z.columns)
            if len(available_genes) < 3:
                continue

            loadings = (
                module_weights
                .set_index("human_gene_symbol")
                .loc[available_genes, "risk_oriented_loading"]
            )
            signs = np.sign(
                pd.to_numeric(loadings, errors="coerce").fillna(0.0)
            )
            signs = signs.replace(0, 1)

            score = z.mul(signs, axis=1).mean(axis=1)
            scores[f"{module}__{strategy_name}"] = zscore_series(
                score
            )

    return scores, pd.DataFrame(coverage_rows)


def score_correlations_with_locked(
    scores: pd.DataFrame,
    locked_scores: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for module in PRIMARY_MODULES:
        locked_col = f"{module}__strict__signed_mean_z"
        if locked_col not in locked_scores.columns:
            continue

        for score_col in [
            column
            for column in scores.columns
            if column.startswith(f"{module}__")
        ]:
            frame = pd.concat(
                [
                    locked_scores[locked_col].rename("locked"),
                    scores[score_col].rename("diagnostic"),
                ],
                axis=1,
            ).dropna()

            correlation = (
                float(frame["locked"].corr(frame["diagnostic"]))
                if (
                    frame.shape[0] >= 5
                    and frame["locked"].std() > 0
                    and frame["diagnostic"].std() > 0
                )
                else np.nan
            )
            rows.append(
                {
                    "module_label": module,
                    "strategy": score_col.replace(
                        f"{module}__",
                        "",
                    ),
                    "score_correlation_with_locked": correlation,
                }
            )

    return pd.DataFrame(rows)


def run_score_rfs_sensitivity(
    clinical: pd.DataFrame,
    scores: pd.DataFrame,
    correlations: pd.DataFrame,
) -> pd.DataFrame:
    frame = clinical.join(scores, how="inner")
    rows: list[dict[str, Any]] = []

    correlation_index = correlations.set_index(
        ["module_label", "strategy"]
    )

    for score_col in scores.columns:
        module, strategy = score_col.split("__", 1)
        fit = fit_cox_score(frame, score_col)

        correlation = np.nan
        if (module, strategy) in correlation_index.index:
            correlation = correlation_index.loc[
                (module, strategy),
                "score_correlation_with_locked",
            ]

        rows.append(
            {
                "module_label": module,
                "strategy": strategy,
                "score_column": score_col,
                "score_correlation_with_locked": correlation,
                **fit,
            }
        )

    result = pd.DataFrame(rows)
    return result.sort_values(
        ["module_label", "strategy"]
    ).reset_index(drop=True)


def build_sample_quality(
    detection_probe_by_sample: pd.DataFrame,
    annotation: pd.DataFrame,
    best_detected: pd.DataFrame,
    strict_weights: pd.DataFrame,
) -> pd.DataFrame:
    sample_qc = pd.DataFrame(
        index=detection_probe_by_sample.columns
    )
    sample_qc["n_all_probes"] = detection_probe_by_sample.shape[0]
    sample_qc["n_detected_p_lt_0_01"] = (
        detection_probe_by_sample.lt(
            DETECTION_THRESHOLD_STRICT
        ).sum(axis=0)
    )
    sample_qc["fraction_detected_p_lt_0_01"] = (
        detection_probe_by_sample.lt(
            DETECTION_THRESHOLD_STRICT
        ).mean(axis=0)
    )
    sample_qc["n_detected_p_lt_0_05"] = (
        detection_probe_by_sample.lt(
            DETECTION_THRESHOLD_RELAXED
        ).sum(axis=0)
    )
    sample_qc["fraction_detected_p_lt_0_05"] = (
        detection_probe_by_sample.lt(
            DETECTION_THRESHOLD_RELAXED
        ).mean(axis=0)
    )
    sample_qc["median_detection_p_all_probes"] = (
        detection_probe_by_sample.median(axis=0)
    )

    weights = strict_weights.copy()
    weights["human_gene_symbol"] = (
        weights["human_gene_symbol"].astype(str).str.upper()
    )
    best_index = best_detected.set_index("gene_symbol")

    for module in PRIMARY_MODULES:
        genes = (
            weights[weights["module_label"].eq(module)]
            ["human_gene_symbol"]
            .drop_duplicates()
            .tolist()
        )
        available = [
            gene for gene in genes if gene in best_index.index
        ]
        probes = best_index.loc[available, "probe_id"].tolist()
        probes = [
            probe
            for probe in probes
            if probe in detection_probe_by_sample.index
        ]
        if not probes:
            continue

        sample_qc[
            f"{module}_best_detected_probe_fraction_p_lt_0_01"
        ] = (
            detection_probe_by_sample.loc[probes]
            .lt(DETECTION_THRESHOLD_STRICT)
            .mean(axis=0)
        )

    sample_qc.index.name = "geo_sample_id"
    return sample_qc


def run_sample_quality_endpoint_diagnostics(
    sample_qc: pd.DataFrame,
    clinical: pd.DataFrame,
    locked_scores: pd.DataFrame,
) -> pd.DataFrame:
    frame = clinical.join(sample_qc, how="inner").join(
        locked_scores.drop(columns=["cohort"], errors="ignore"),
        how="left",
    )

    predictor_cols = [
        column
        for column in sample_qc.columns
        if (
            "fraction_detected" in column
            or column.endswith("fraction_p_lt_0_01")
        )
    ]

    rows: list[dict[str, Any]] = []

    for predictor in predictor_cols:
        frame[f"{predictor}__z"] = zscore_series(
            frame[predictor]
        )
        fit = fit_cox_score(
            frame,
            f"{predictor}__z",
        )

        event_values = frame.loc[
            frame[EVENT_COL].eq(1),
            predictor,
        ].dropna()
        censor_values = frame.loc[
            frame[EVENT_COL].eq(0),
            predictor,
        ].dropna()

        mann_whitney_p = np.nan
        if (
            event_values.shape[0] >= 3
            and censor_values.shape[0] >= 3
        ):
            mann_whitney_p = float(
                stats.mannwhitneyu(
                    event_values,
                    censor_values,
                    alternative="two-sided",
                ).pvalue
            )

        rows.append(
            {
                "diagnostic_type": "sample_quality_vs_endpoint",
                "predictor": predictor,
                "module_label": "",
                "spearman_rho": np.nan,
                "spearman_p": np.nan,
                "event_group_median": (
                    float(event_values.median())
                    if not event_values.empty
                    else np.nan
                ),
                "censored_group_median": (
                    float(censor_values.median())
                    if not censor_values.empty
                    else np.nan
                ),
                "mann_whitney_p": mann_whitney_p,
                **fit,
            }
        )

    for module in PRIMARY_MODULES:
        score_col = f"{module}__strict__signed_mean_z"
        if score_col not in frame.columns:
            continue

        for predictor in predictor_cols:
            pair = frame[[score_col, predictor]].dropna()
            rho = np.nan
            p_value = np.nan
            if (
                pair.shape[0] >= 5
                and pair[score_col].std() > 0
                and pair[predictor].std() > 0
            ):
                result = stats.spearmanr(
                    pair[score_col],
                    pair[predictor],
                )
                rho = float(result.statistic)
                p_value = float(result.pvalue)

            rows.append(
                {
                    "diagnostic_type": "sample_quality_vs_locked_score",
                    "predictor": predictor,
                    "module_label": module,
                    "spearman_rho": rho,
                    "spearman_p": p_value,
                    "event_group_median": np.nan,
                    "censored_group_median": np.nan,
                    "mann_whitney_p": np.nan,
                    "n": pair.shape[0],
                    "events": np.nan,
                    "hr_per_sd": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p": np.nan,
                    "fixed_direction_c_index": np.nan,
                    "direction_concordant_with_canine": np.nan,
                    "error": "",
                }
            )

    return pd.DataFrame(rows)


def build_canonical_panel(
    strict_weights: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "gene_symbol": gene,
            "panel_role": "canonical_assay_direction_check",
        }
        for gene in CANONICAL_QC_GENES
    ]

    m34 = strict_weights[
        strict_weights["module_label"].eq("M34")
    ].copy()
    m34["human_gene_symbol"] = (
        m34["human_gene_symbol"].astype(str).str.upper()
    )
    m34["risk_oriented_loading"] = pd.to_numeric(
        m34["risk_oriented_loading"],
        errors="coerce",
    )

    top_positive = (
        m34.sort_values(
            "risk_oriented_loading",
            ascending=False,
        )
        .head(5)
    )
    top_negative = (
        m34.sort_values(
            "risk_oriented_loading",
            ascending=True,
        )
        .head(5)
    )

    for _, row in top_positive.iterrows():
        rows.append(
            {
                "gene_symbol": row["human_gene_symbol"],
                "panel_role": "M34_top_positive_frozen_loading",
            }
        )
    for _, row in top_negative.iterrows():
        rows.append(
            {
                "gene_symbol": row["human_gene_symbol"],
                "panel_role": "M34_top_negative_frozen_loading",
            }
        )

    panel = pd.DataFrame(rows).drop_duplicates(
        ["gene_symbol", "panel_role"]
    )
    return panel


def run_canonical_gene_audit(
    panel: pd.DataFrame,
    probe_quality: pd.DataFrame,
    values_probe_by_sample: pd.DataFrame,
    clinical: pd.DataFrame,
    selection_comparison: pd.DataFrame,
    strict_weights: pd.DataFrame,
) -> pd.DataFrame:
    comparison = selection_comparison.set_index("gene_symbol")
    weights = strict_weights.copy()
    weights["human_gene_symbol"] = (
        weights["human_gene_symbol"].astype(str).str.upper()
    )
    frozen_loading_map = (
        weights.groupby("human_gene_symbol")[
            "risk_oriented_loading"
        ]
        .first()
        .to_dict()
    )

    frame_base = clinical.copy()
    rows: list[dict[str, Any]] = []

    for _, panel_row in panel.iterrows():
        gene = panel_row["gene_symbol"]
        role = panel_row["panel_role"]

        probes = probe_quality[
            probe_quality["gene_symbol"].eq(gene)
        ].copy()

        for _, probe_row in probes.iterrows():
            probe_id = probe_row["probe_id"]
            if probe_id not in values_probe_by_sample.index:
                continue

            score = zscore_series(
                values_probe_by_sample.loc[probe_id]
            )
            frame = frame_base.join(
                score.rename("probe_expression_z"),
                how="inner",
            )
            fit = fit_cox_score(
                frame,
                "probe_expression_z",
            )

            max_probe = (
                comparison.loc[
                    gene,
                    "max_variance_probe_id",
                ]
                if gene in comparison.index
                else np.nan
            )
            best_probe = (
                comparison.loc[
                    gene,
                    "best_detected_probe_id",
                ]
                if gene in comparison.index
                else np.nan
            )

            rows.append(
                {
                    "gene_symbol": gene,
                    "panel_role": role,
                    "probe_id": probe_id,
                    "selected_by_max_variance": (
                        probe_id == max_probe
                    ),
                    "selected_by_best_detected": (
                        probe_id == best_probe
                    ),
                    "frozen_risk_oriented_loading": (
                        frozen_loading_map.get(gene, np.nan)
                    ),
                    "expression_variance": probe_row[
                        "expression_variance"
                    ],
                    "detected_fraction_p_lt_0_01": probe_row[
                        "detected_fraction_p_lt_0_01"
                    ],
                    "detected_fraction_p_lt_0_05": probe_row[
                        "detected_fraction_p_lt_0_05"
                    ],
                    "detection_p_median": probe_row[
                        "detection_p_median"
                    ],
                    **fit,
                }
            )

    result = pd.DataFrame(rows)

    expected_notes = {
        "MKI67": (
            "Proliferation marker; higher mRNA is a descriptive "
            "aggressiveness sanity check, not a definitive validation control."
        ),
        "TOP2A": (
            "Proliferation marker; higher mRNA is a descriptive "
            "aggressiveness sanity check."
        ),
        "BIRC5": (
            "Cell-cycle/survival marker; higher mRNA is a descriptive "
            "aggressiveness sanity check."
        ),
        "UBE2C": (
            "Cell-cycle marker; higher mRNA is a descriptive "
            "aggressiveness sanity check."
        ),
        "EZR": (
            "Ezrin evidence is largely protein/localization based; "
            "mRNA direction is not a definitive positive control."
        ),
    }
    if not result.empty:
        result["interpretation_note"] = result[
            "gene_symbol"
        ].map(expected_notes).fillna(
            "Frozen M34-loading diagnostic; no independent expected "
            "outcome direction is imposed."
        )

    return result


def build_module_summary(
    coverage: pd.DataFrame,
    rfs: pd.DataFrame,
    locked_rfs: pd.DataFrame,
) -> pd.DataFrame:
    locked = locked_rfs.set_index("module_label")
    rows: list[dict[str, Any]] = []

    for module in PRIMARY_MODULES:
        module_rfs = rfs[
            rfs["module_label"].eq(module)
        ].copy()
        module_coverage = coverage[
            coverage["module_label"].eq(module)
        ].copy()

        locked_hr = (
            locked.loc[module, "hr_per_sd"]
            if module in locked.index
            else np.nan
        )
        locked_direction = (
            bool(locked_hr > 1)
            if np.isfinite(locked_hr)
            else np.nan
        )

        valid_directions = module_rfs[
            "direction_concordant_with_canine"
        ].dropna()
        valid_correlations = module_rfs[
            "score_correlation_with_locked"
        ].dropna()

        rows.append(
            {
                "module_label": module,
                "locked_hr_per_sd": locked_hr,
                "locked_direction_concordant_with_canine": locked_direction,
                "n_diagnostic_score_strategies": module_rfs.shape[0],
                "n_direction_concordant_strategies": int(
                    valid_directions.sum()
                ),
                "fraction_direction_concordant_strategies": (
                    float(valid_directions.mean())
                    if not valid_directions.empty
                    else np.nan
                ),
                "minimum_score_correlation_with_locked": (
                    float(valid_correlations.min())
                    if not valid_correlations.empty
                    else np.nan
                ),
                "median_score_correlation_with_locked": (
                    float(valid_correlations.median())
                    if not valid_correlations.empty
                    else np.nan
                ),
                "best_detected_50_coverage_fraction": (
                    module_coverage.loc[
                        module_coverage["strategy"].eq(
                            "best_detected_p01_ge50"
                        ),
                        "coverage_fraction",
                    ].iloc[0]
                    if (
                        module_coverage["strategy"].eq(
                            "best_detected_p01_ge50"
                        ).any()
                    )
                    else np.nan
                ),
                "best_detected_80_coverage_fraction": (
                    module_coverage.loc[
                        module_coverage["strategy"].eq(
                            "best_detected_p01_ge80"
                        ),
                        "coverage_fraction",
                    ].iloc[0]
                    if (
                        module_coverage["strategy"].eq(
                            "best_detected_p01_ge80"
                        ).any()
                    )
                    else np.nan
                ),
                "diagnostic_interpretation": "",
            }
        )

    summary = pd.DataFrame(rows)

    def interpretation(row: pd.Series) -> str:
        fraction = row[
            "fraction_direction_concordant_strategies"
        ]
        minimum_corr = row[
            "minimum_score_correlation_with_locked"
        ]
        coverage_50 = row[
            "best_detected_50_coverage_fraction"
        ]

        if (
            np.isfinite(fraction)
            and fraction == 1.0
            and np.isfinite(minimum_corr)
            and minimum_corr >= 0.90
            and np.isfinite(coverage_50)
            and coverage_50 >= 0.70
        ):
            return (
                "Direction and score representation are stable across "
                "detection-aware rules."
            )
        if (
            np.isfinite(fraction)
            and fraction == 0.0
            and np.isfinite(minimum_corr)
            and minimum_corr >= 0.90
        ):
            return (
                "Discordant direction persists across detection-aware "
                "rules despite high score concordance."
            )
        return (
            "Assay-rule sensitivity is material or coverage is limited; "
            "interpret GSE39055 cautiously."
        )

    summary["diagnostic_interpretation"] = summary.apply(
        interpretation,
        axis=1,
    )
    return summary


def write_readme() -> None:
    text = f"""GSE39055 assay-quality diagnostic
Script version: {SCRIPT_VERSION}

Purpose
-------
This script audits the FFPE WG-DASL assay layer without changing the frozen
program definitions or the locked primary analyses.

Data used
---------
- GEO sample-level normalized VALUE
- GEO sample-level Detection PVal
- GPL14951 probe-to-gene annotation
- Frozen strict canine-to-human genes and risk-oriented signs

Outcome-blind probe rules
-------------------------
1. Highest-variance probe per unambiguous gene.
2. Best-detected probe per unambiguous gene.
3. Highest-variance probe filtered to Detection PVal < 0.01 in at least 50% of samples.
4. Best-detected probe filtered to Detection PVal < 0.01 in at least 50% of samples.
5. Best-detected probe filtered to Detection PVal < 0.01 in at least 80% of samples.

Interpretation restriction
--------------------------
RFS associations under alternative assay rules are diagnostic sensitivities.
They do not replace script 26, change frozen weights, reverse score direction,
or reopen the locked evidence hierarchy from script 29.

Canonical-gene restriction
--------------------------
MKI67, TOP2A, BIRC5, UBE2C, and EZR are descriptive assay-direction checks.
They are not treated as gold-standard positive controls, and no result is used
to select or orient a module.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def create_manifest(
    input_paths: list[Path],
    output_paths: list[Path],
    detection_columns: dict[str, str],
) -> None:
    payload = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "detection_thresholds": {
            "strict_p": DETECTION_THRESHOLD_STRICT,
            "relaxed_p": DETECTION_THRESHOLD_RELAXED,
            "minimum_fraction_50": MIN_DETECTED_FRACTION_50,
            "minimum_fraction_80": MIN_DETECTED_FRACTION_80,
        },
        "detection_columns_by_sample": detection_columns,
        "guardrails": [
            "No outcome-guided probe selection.",
            "No change to frozen genes, loadings, directions, or tiers.",
            "No replacement of the locked script 26 primary analysis.",
            "Canonical genes are descriptive assay checks only.",
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

    OUTPUT_MANIFEST.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 80)
    print("GSE39055 FFPE DASL assay-quality diagnostic")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Design:")
    print("  Read sample-level GEO Detection PVal values.")
    print("  Compare highest-variance and best-detected probe selection.")
    print("  Reconstruct frozen scores under outcome-blind detection filters.")
    print("  Audit sample quality, canonical genes, and RFS direction stability.")
    print("  Preserve the locked script 26 and script 29 conclusions.")
    print("")

    strict_weights = read_required_csv(STRICT_WEIGHTS_FILE)
    clinical = read_required_csv(CLINICAL_FILE, index_col=0)
    locked_scores = read_required_csv(
        LOCKED_SCORES_FILE,
        index_col=0,
    )
    locked_rfs = read_required_csv(LOCKED_RFS_FILE)

    strict_weights["human_gene_symbol"] = (
        strict_weights["human_gene_symbol"]
        .astype(str)
        .str.upper()
    )

    gse = load_gse39055()
    (
        values_probe_by_sample,
        detection_probe_by_sample,
        detection_columns,
    ) = extract_probe_matrices(gse)
    annotation = build_annotation(gse)

    common_samples = (
        values_probe_by_sample.columns
        .intersection(clinical.index)
        .intersection(locked_scores.index)
    )
    values_probe_by_sample = values_probe_by_sample[
        common_samples
    ]
    detection_probe_by_sample = detection_probe_by_sample[
        common_samples
    ]
    clinical = clinical.loc[common_samples].copy()
    locked_scores = locked_scores.loc[common_samples].copy()

    probe_quality = build_probe_quality(
        values_probe_by_sample=values_probe_by_sample,
        detection_probe_by_sample=detection_probe_by_sample,
        annotation=annotation,
    )
    probe_quality.to_csv(OUTPUT_PROBE_QC, index=False)

    (
        max_variance,
        best_detected,
        selection_comparison,
    ) = select_probes(probe_quality)
    selection_comparison = add_locked_probe_map_comparison(
        selection_comparison
    )
    selection_comparison.to_csv(
        OUTPUT_SELECTION,
        index=False,
    )

    selection_strategies = {
        "max_variance_unfiltered": (
            max_variance,
            None,
        ),
        "max_variance_p01_ge50": (
            max_variance,
            MIN_DETECTED_FRACTION_50,
        ),
        "best_detected_unfiltered": (
            best_detected,
            None,
        ),
        "best_detected_p01_ge50": (
            best_detected,
            MIN_DETECTED_FRACTION_50,
        ),
        "best_detected_p01_ge80": (
            best_detected,
            MIN_DETECTED_FRACTION_80,
        ),
    }

    scores, coverage = compute_module_scores(
        strict_weights=strict_weights,
        values_probe_by_sample=values_probe_by_sample,
        selection_strategies=selection_strategies,
    )
    coverage.to_csv(
        OUTPUT_MODULE_COVERAGE,
        index=False,
    )
    scores.to_csv(OUTPUT_SCORES)

    correlations = score_correlations_with_locked(
        scores=scores,
        locked_scores=locked_scores,
    )
    rfs = run_score_rfs_sensitivity(
        clinical=clinical,
        scores=scores,
        correlations=correlations,
    )
    rfs.to_csv(OUTPUT_RFS, index=False)

    sample_qc = build_sample_quality(
        detection_probe_by_sample=detection_probe_by_sample,
        annotation=annotation,
        best_detected=best_detected,
        strict_weights=strict_weights,
    )
    sample_qc.to_csv(OUTPUT_SAMPLE_QC)

    endpoint_qc = run_sample_quality_endpoint_diagnostics(
        sample_qc=sample_qc,
        clinical=clinical,
        locked_scores=locked_scores,
    )
    endpoint_qc.to_csv(
        OUTPUT_ENDPOINT_QC,
        index=False,
    )

    canonical_panel = build_canonical_panel(
        strict_weights
    )
    canonical = run_canonical_gene_audit(
        panel=canonical_panel,
        probe_quality=probe_quality,
        values_probe_by_sample=values_probe_by_sample,
        clinical=clinical,
        selection_comparison=selection_comparison,
        strict_weights=strict_weights,
    )
    canonical.to_csv(
        OUTPUT_CANONICAL,
        index=False,
    )

    summary = build_module_summary(
        coverage=coverage,
        rfs=rfs,
        locked_rfs=locked_rfs,
    )
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    write_readme()

    input_paths = [
        SOFT_FILE,
        STRICT_WEIGHTS_FILE,
        CLINICAL_FILE,
        LOCKED_SCORES_FILE,
        LOCKED_PROBE_MAP_FILE,
        LOCKED_RFS_FILE,
    ]
    output_paths = [
        OUTPUT_PROBE_QC,
        OUTPUT_SELECTION,
        OUTPUT_MODULE_COVERAGE,
        OUTPUT_SCORES,
        OUTPUT_RFS,
        OUTPUT_CANONICAL,
        OUTPUT_SAMPLE_QC,
        OUTPUT_ENDPOINT_QC,
        OUTPUT_SUMMARY,
        OUTPUT_README,
    ]
    create_manifest(
        input_paths=input_paths,
        output_paths=output_paths,
        detection_columns=detection_columns,
    )

    print("")
    print("=" * 80)
    print("Detection P-value extraction")
    print("=" * 80)
    unique_detection_columns = sorted(
        set(detection_columns.values())
    )
    print(f"Samples parsed: {len(detection_columns)}")
    print(f"Detection columns: {unique_detection_columns}")
    print(
        f"Probe-by-sample matrix: "
        f"{values_probe_by_sample.shape[0]} probes x "
        f"{values_probe_by_sample.shape[1]} samples"
    )

    print("")
    print("=" * 80)
    print("Sample-level assay quality")
    print("=" * 80)
    print(
        sample_qc[
            [
                "n_detected_p_lt_0_01",
                "fraction_detected_p_lt_0_01",
                "n_detected_p_lt_0_05",
                "fraction_detected_p_lt_0_05",
                "median_detection_p_all_probes",
            ]
        ]
        .describe()
        .to_string()
    )

    print("")
    print("=" * 80)
    print("Selected-probe detectability comparison")
    print("=" * 80)
    comparison_summary = pd.DataFrame(
        {
            "metric": [
                "genes_compared",
                "same_probe_fraction",
                "locked_matches_recomputed_max_variance_fraction",
                "median_max_variance_detected_fraction_p01",
                "median_best_detected_detected_fraction_p01",
            ],
            "value": [
                selection_comparison.shape[0],
                selection_comparison[
                    "same_selected_probe"
                ].mean(),
                selection_comparison[
                    "locked_matches_recomputed_max_variance"
                ].dropna().mean(),
                selection_comparison[
                    "max_variance_detected_fraction_p_lt_0_01"
                ].median(),
                selection_comparison[
                    "best_detected_detected_fraction_p_lt_0_01"
                ].median(),
            ],
        }
    )
    print(comparison_summary.to_string(index=False))

    print("")
    print("=" * 80)
    print("Detection-aware frozen-module coverage")
    print("=" * 80)
    print(
        coverage[
            [
                "module_label",
                "strategy",
                "n_frozen_genes",
                "n_available_genes",
                "coverage_fraction",
                "median_gene_detected_fraction_p_lt_0_01",
                "minimum_gene_detected_fraction_p_lt_0_01",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Detection-aware RFS direction sensitivity")
    print("=" * 80)
    print(
        rfs[
            [
                "module_label",
                "strategy",
                "score_correlation_with_locked",
                "n",
                "events",
                "hr_per_sd",
                "ci_low",
                "ci_high",
                "p",
                "fixed_direction_c_index",
                "direction_concordant_with_canine",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Module-level assay diagnostic summary")
    print("=" * 80)
    print(summary.to_string(index=False))

    print("")
    print("=" * 80)
    print("Canonical and M34-loading probe audit")
    print("=" * 80)
    canonical_display = canonical[
        canonical["selected_by_max_variance"]
        | canonical["selected_by_best_detected"]
    ].copy()
    if canonical_display.empty:
        print("No selected canonical probes were available.")
    else:
        print(
            canonical_display[
                [
                    "gene_symbol",
                    "panel_role",
                    "probe_id",
                    "selected_by_max_variance",
                    "selected_by_best_detected",
                    "detected_fraction_p_lt_0_01",
                    "detection_p_median",
                    "hr_per_sd",
                    "p",
                    "fixed_direction_c_index",
                ]
            ].to_string(index=False)
        )

    print("")
    print("=" * 80)
    print("Assay-quality endpoint diagnostics")
    print("=" * 80)
    print(
        endpoint_qc[
            endpoint_qc["diagnostic_type"].eq(
                "sample_quality_vs_endpoint"
            )
        ][
            [
                "predictor",
                "event_group_median",
                "censored_group_median",
                "mann_whitney_p",
                "hr_per_sd",
                "p",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("This script is a diagnostic addendum and does not replace the locked script 26 analysis.")
    print("Probe selection and detection filters are outcome-blind.")
    print("Frozen genes, risk-oriented signs, weights, and validation tiers are unchanged.")
    print("Canonical-gene results are descriptive assay checks, not positive-control validation.")
    print("A stable discordant direction across detection-aware rules supports cohort/platform heterogeneity; an unstable direction indicates assay-rule sensitivity.")

    print("")
    print("Saved:")
    for path in output_paths + [OUTPUT_MANIFEST]:
        print(path)
    print("Done.")


if __name__ == "__main__":
    main()
