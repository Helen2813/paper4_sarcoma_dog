from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

SCRIPT_VERSION = "46-gse239948-external-canine-representation-v2"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
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

ACCESSION = "GSE239948"
RAW_FILENAME = "GSE239948_CCOGC.txt.gz"
DOWNLOAD_URLS = [
    (
        "https://www.ncbi.nlm.nih.gov/geo/download/"
        "?acc=GSE239948&file=GSE239948_CCOGC.txt.gz&format=file"
    ),
    (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE239nnn/"
        "GSE239948/suppl/GSE239948_CCOGC.txt.gz"
    ),
]

REFERENCE_EXPRESSION_FILE = (
    PROCESSED_DIR
    / "GSE238110_DOG2_expression_log2cpm_matched_allgenes.csv"
)
STRICT_WEIGHTS_FILE = (
    RESULTS_DIR / "GSE238110_frozen_transfer_gene_weights_strict.csv"
)
FREEZE_FILE = (
    RESULTS_DIR / "GSE238110_frozen_transfer_program_freeze.json"
)

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]

MIN_MODULE_GENES = 3
N_GENE_LABEL_PERMUTATIONS = 5000
N_RANDOM_PANELS = 1000
N_SPLIT_HALF_REPEATS = 2000
N_VARIABILITY_BINS = 10
RANDOM_SEED = 42

OUTPUT_INPUT_AUDIT = (
    RESULTS_DIR / "GSE239948_external_input_audit.csv"
)
OUTPUT_IDENTIFIER_PAIR_AUDIT = (
    RESULTS_DIR / "GSE239948_external_identifier_pair_audit.csv"
)
OUTPUT_GENE_MAP = (
    RESULTS_DIR / "GSE239948_external_gene_mapping.csv"
)
OUTPUT_EXPRESSION = (
    PROCESSED_DIR / "canine_validation_GSE239948_expression_log2_symbol.csv"
)
OUTPUT_SCORES = (
    PROCESSED_DIR / "canine_validation_GSE239948_frozen_program_scores.csv"
)
OUTPUT_COVERAGE = (
    RESULTS_DIR / "GSE239948_external_frozen_program_coverage.csv"
)
OUTPUT_STRUCTURE = (
    RESULTS_DIR / "GSE239948_external_module_structure_preservation.csv"
)
OUTPUT_RELIABILITY = (
    RESULTS_DIR / "GSE239948_external_module_score_reliability.csv"
)
OUTPUT_GENE_LOO = (
    RESULTS_DIR / "GSE239948_external_module_gene_leave_one_out.csv"
)
OUTPUT_RANDOM = (
    RESULTS_DIR / "GSE239948_external_random_panel_controls.csv"
)
OUTPUT_CLASSIFICATION = (
    RESULTS_DIR / "GSE239948_external_representation_classification.csv"
)
OUTPUT_README = (
    RESULTS_DIR / "GSE239948_external_representation_README.txt"
)
OUTPUT_MANIFEST = (
    RESULTS_DIR / "GSE239948_external_representation_manifest.json"
)

HEATMAP_PNG = (
    FIGURES_DIR / "GSE239948_external_representation_heatmap.png"
)
HEATMAP_PDF = (
    FIGURES_DIR / "GSE239948_external_representation_heatmap.pdf"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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


def clean_symbol(value: Any) -> str:
    text = str(value).strip().upper()

    if text in {"", "NAN", "NONE", "NA"}:
        return ""

    if "|" in text:
        parts = [part.strip() for part in text.split("|") if part.strip()]
        symbol_like = [
            part
            for part in parts
            if not part.startswith("ENSCAFG")
        ]
        text = symbol_like[-1] if symbol_like else parts[-1]

    text = re.sub(r"\.\d+$", "", text)
    return text


def locate_or_download_raw_file() -> Path:
    candidates = [
        RAW_DIR / "canine_validation_GSE239948" / RAW_FILENAME,
        RAW_DIR / "canine_validation_GSE239948" / "GSE239948_CCOGC.txt.gz",
        RAW_DIR / "GSE239948" / RAW_FILENAME,
        RAW_DIR / RAW_FILENAME,
        PROJECT_ROOT / RAW_FILENAME,
    ]

    for path in candidates:
        if path.exists():
            print(f"Using cached GSE239948 file: {path}")
            return path

    recursive = list(PROJECT_ROOT.rglob(RAW_FILENAME))
    if recursive:
        print(f"Using cached GSE239948 file: {recursive[0]}")
        return recursive[0]

    destination = (
        RAW_DIR / "canine_validation_GSE239948" / RAW_FILENAME
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    errors = []
    for url in DOWNLOAD_URLS:
        print(f"Downloading: {url}")
        try:
            urllib.request.urlretrieve(url, destination)
            if destination.exists() and destination.stat().st_size > 0:
                print(f"Saved: {destination}")
                return destination
        except Exception as error:
            errors.append(f"{url}: {error}")

    raise RuntimeError(
        "Could not download GSE239948.\n" + "\n".join(errors)
    )


def read_raw_table(path: Path) -> pd.DataFrame:
    attempts = [
        {"sep": "\t", "compression": "gzip"},
        {"sep": ",", "compression": "gzip"},
        {"sep": None, "engine": "python", "compression": "gzip"},
    ]

    best = None
    for kwargs in attempts:
        try:
            table = pd.read_csv(
                path,
                low_memory=False,
                **kwargs,
            )
            if table.shape[1] >= 10 and table.shape[0] >= 100:
                best = table
                break
        except Exception:
            continue

    if best is None:
        raise RuntimeError(
            "Could not parse the GSE239948 processed matrix."
        )
    return best


def sample_columns(table: pd.DataFrame) -> list[str]:
    columns = [str(column) for column in table.columns]
    matched = [
        column
        for column in columns
        if re.match(r"^CCB\d+", column, flags=re.IGNORECASE)
        or re.match(r"^GSM\d+", column, flags=re.IGNORECASE)
    ]

    if len(matched) >= 30:
        return matched

    numeric_fraction = {}
    for column in columns:
        numeric = pd.to_numeric(table[column], errors="coerce")
        numeric_fraction[column] = float(numeric.notna().mean())

    candidates = [
        column
        for column in columns
        if numeric_fraction[column] >= 0.95
    ]

    if len(candidates) < 30:
        raise RuntimeError(
            "Could not identify at least 30 numeric sample columns."
        )
    return candidates



def normalize_identifier(value: Any, scheme: str) -> str:
    text = str(value).strip().upper()

    if text in {"", "NAN", "NONE", "NA", "---"}:
        return ""

    if scheme == "ensembl":
        match = re.search(r"ENSCAFG\d+", text)
        return match.group(0) if match else ""

    if scheme == "symbol":
        gene_name_match = re.search(
            r"(?:GENE[_ ]?NAME|SYMBOL)\s*[=:]\s*([A-Z0-9_.-]+)",
            text,
        )
        if gene_name_match:
            text = gene_name_match.group(1)

        if "|" in text:
            parts = [
                part.strip()
                for part in text.split("|")
                if part.strip()
            ]
            non_ensembl = [
                part
                for part in parts
                if not re.fullmatch(r"ENSCAFG\d+(?:\.\d+)?", part)
            ]
            text = non_ensembl[-1] if non_ensembl else ""

        if ";" in text:
            parts = [
                part.strip()
                for part in text.split(";")
                if part.strip()
            ]
            symbol_like = [
                part
                for part in parts
                if re.fullmatch(r"[A-Z][A-Z0-9_.-]*", part)
                and not part.startswith("ENSCAFG")
            ]
            if symbol_like:
                text = symbol_like[-1]

        text = re.sub(r"\.\d+$", "", text)
        text = re.sub(r"_\d+$", "", text)
        text = text.strip()

        if text.startswith("ENSCAFG"):
            return ""
        if not re.fullmatch(r"[A-Z][A-Z0-9_.-]*", text):
            return ""
        return text

    if scheme == "raw":
        return re.sub(r"\.\d+$", "", text)

    raise ValueError(f"Unknown identifier scheme: {scheme}")


def frozen_identifier_columns(weights: pd.DataFrame) -> list[str]:
    preferred = [
        "canine_gene_symbol",
        "canine_gene",
        "dog_gene_symbol",
        "dog_gene",
        "canine_ensembl_gene_id",
        "dog_ensembl_gene_id",
        "ensembl_gene_id",
    ]
    columns = []

    for column in preferred:
        if column in weights.columns:
            columns.append(column)

    for column in weights.columns:
        lower = str(column).lower()
        if (
            column not in columns
            and any(token in lower for token in ["canine", "dog"])
            and any(
                token in lower
                for token in ["gene", "symbol", "ensembl", "id"]
            )
        ):
            columns.append(str(column))

    return columns


def identifier_pair_audit(
    table: pd.DataFrame,
    sample_cols: list[str],
    weights: pd.DataFrame,
) -> pd.DataFrame:
    external_columns = [
        str(column)
        for column in table.columns
        if str(column) not in sample_cols
    ]
    frozen_columns = frozen_identifier_columns(weights)

    if not external_columns:
        raise RuntimeError(
            "No non-sample columns were available for gene identifiers."
        )
    if not frozen_columns:
        raise RuntimeError(
            "No canine identifier column was found in the frozen weights."
        )

    primary_weights = weights[
        weights["module_label"].isin(PRIMARY_MODULES)
    ].copy()

    rows = []
    for frozen_column in frozen_columns:
        for scheme in ["symbol", "ensembl", "raw"]:
            frozen_keys = primary_weights[frozen_column].map(
                lambda value: normalize_identifier(value, scheme)
            )
            frozen_keys = frozen_keys[frozen_keys.ne("")]

            if frozen_keys.empty:
                continue

            frozen_set = set(frozen_keys)
            module_key_sets = {
                module: set(
                    primary_weights.loc[
                        primary_weights["module_label"].eq(module),
                        frozen_column,
                    ].map(
                        lambda value: normalize_identifier(value, scheme)
                    )
                )
                - {""}
                for module in PRIMARY_MODULES
            }

            for external_column in external_columns:
                external_keys = table[external_column].map(
                    lambda value: normalize_identifier(value, scheme)
                )
                external_set = set(external_keys) - {""}
                overlap_set = frozen_set.intersection(external_set)

                module_overlaps = {
                    module: len(
                        module_key_sets[module].intersection(external_set)
                    )
                    for module in PRIMARY_MODULES
                }
                modules_ge3 = sum(
                    count >= MIN_MODULE_GENES
                    for count in module_overlaps.values()
                )

                rows.append(
                    {
                        "external_column": external_column,
                        "frozen_identifier_column": frozen_column,
                        "identifier_scheme": scheme,
                        "n_external_unique_ids": len(external_set),
                        "n_frozen_unique_ids": len(frozen_set),
                        "n_unique_overlap": len(overlap_set),
                        "n_primary_modules_with_at_least_3_genes": modules_ge3,
                        **{
                            f"{module}_overlap": module_overlaps[module]
                            for module in PRIMARY_MODULES
                        },
                        "example_overlap": ";".join(
                            sorted(overlap_set)[:20]
                        ),
                    }
                )

    if not rows:
        raise RuntimeError(
            "No usable external/frozen identifier combinations were found."
        )

    audit = pd.DataFrame(rows)
    preference = {
        "canine_gene_symbol": 4,
        "canine_gene": 3,
        "dog_gene_symbol": 2,
        "dog_gene": 1,
    }
    scheme_preference = {
        "symbol": 3,
        "ensembl": 2,
        "raw": 1,
    }
    audit["frozen_column_preference"] = audit[
        "frozen_identifier_column"
    ].map(preference).fillna(0)
    audit["scheme_preference"] = audit[
        "identifier_scheme"
    ].map(scheme_preference).fillna(0)

    return audit.sort_values(
        [
            "n_primary_modules_with_at_least_3_genes",
            "n_unique_overlap",
            "frozen_column_preference",
            "scheme_preference",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def choose_identifier_pair(
    audit: pd.DataFrame,
) -> tuple[str, str, str]:
    top = audit.iloc[0]

    if int(top["n_unique_overlap"]) < MIN_MODULE_GENES:
        raise RuntimeError(
            "No external/frozen identifier pair had at least three "
            "overlapping genes. Review "
            "GSE239948_external_identifier_pair_audit.csv."
        )

    return (
        str(top["external_column"]),
        str(top["frozen_identifier_column"]),
        str(top["identifier_scheme"]),
    )


def collapse_to_identifiers(
    table: pd.DataFrame,
    gene_column: str,
    sample_cols: list[str],
    identifier_scheme: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expression = table[[gene_column, *sample_cols]].copy()
    expression["analysis_gene_id"] = expression[gene_column].map(
        lambda value: normalize_identifier(
            value,
            identifier_scheme,
        )
    )

    for column in sample_cols:
        expression[column] = pd.to_numeric(
            expression[column],
            errors="coerce",
        )

    expression = expression[
        expression["analysis_gene_id"].ne("")
    ].copy()
    expression["row_variance"] = expression[
        sample_cols
    ].var(axis=1)

    expression = (
        expression.sort_values(
            ["analysis_gene_id", "row_variance"],
            ascending=[True, False],
        )
        .drop_duplicates("analysis_gene_id", keep="first")
    )

    gene_map = expression[
        [gene_column, "analysis_gene_id", "row_variance"]
    ].copy()

    matrix = expression.set_index("analysis_gene_id")[
        sample_cols
    ].T
    matrix.index.name = "sample_id"
    matrix = matrix.loc[
        :,
        matrix.notna().any(axis=0),
    ]
    matrix = matrix.fillna(matrix.median(axis=0))

    return matrix, gene_map


def remap_reference_expression(
    reference: pd.DataFrame,
    weights: pd.DataFrame,
    frozen_identifier_column: str,
    identifier_scheme: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "canine_gene" not in weights.columns:
        raise ValueError(
            "Frozen weights are missing the canine_gene column needed "
            "to map the DOG2 reference expression matrix."
        )

    mapping = weights[
        ["canine_gene", frozen_identifier_column]
    ].copy()
    mapping["canine_gene"] = mapping["canine_gene"].astype(str)
    mapping["analysis_gene_id"] = mapping[
        frozen_identifier_column
    ].map(
        lambda value: normalize_identifier(
            value,
            identifier_scheme,
        )
    )
    mapping = mapping[
        mapping["analysis_gene_id"].ne("")
    ].drop_duplicates(
        ["canine_gene", "analysis_gene_id"],
        keep="first",
    )

    exact_map = (
        mapping.drop_duplicates("canine_gene")
        .set_index("canine_gene")["analysis_gene_id"]
        .to_dict()
    )

    rename_map = {}
    mapping_rows = []
    for column in reference.columns.astype(str):
        analysis_id = exact_map.get(column, "")
        if not analysis_id and identifier_scheme == "symbol":
            analysis_id = normalize_identifier(column, "symbol")
        if not analysis_id:
            continue
        rename_map[column] = analysis_id
        mapping_rows.append(
            {
                "reference_expression_column": column,
                "analysis_gene_id": analysis_id,
            }
        )

    remapped = reference.rename(columns=rename_map)
    remapped = remapped.loc[:, list(rename_map.values())]
    remapped = remapped.loc[
        :,
        ~pd.Index(remapped.columns).duplicated(keep="first"),
    ]

    return remapped, pd.DataFrame(mapping_rows)
def choose_transform(expression: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    values = expression.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        raise RuntimeError("The external expression matrix is empty.")

    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    median = float(np.median(finite))
    integer_like_fraction = float(
        np.mean(np.isclose(finite, np.round(finite)))
    )

    if minimum >= 0 and (maximum > 50 or median > 10):
        transformed = np.log2(expression.clip(lower=0) + 1.0)
        transform = "log2_x_plus_1"
    else:
        transformed = expression.copy()
        transform = "as_provided"

    diagnostics = {
        "minimum": minimum,
        "median": median,
        "maximum": maximum,
        "integer_like_fraction": integer_like_fraction,
        "transform": transform,
    }
    return transformed, diagnostics


def zscore_columns(expression: pd.DataFrame) -> pd.DataFrame:
    x = expression.apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(axis=0))
    std = x.std(axis=0).replace(0, np.nan)
    z = (x - x.mean(axis=0)) / std
    return z.loc[:, z.notna().all(axis=0)]


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[indices]


def permutation_gene_label_p(
    reference_values: np.ndarray,
    external_matrix: np.ndarray,
    observed: float,
    seed: int,
) -> float:
    if not np.isfinite(observed):
        return np.nan
    rng = np.random.default_rng(seed)
    count = 0
    n_genes = external_matrix.shape[0]

    for _ in range(N_GENE_LABEL_PERMUTATIONS):
        permutation = rng.permutation(n_genes)
        permuted = external_matrix[
            np.ix_(permutation, permutation)
        ]
        value = stats.spearmanr(
            reference_values,
            upper_triangle(permuted),
        ).statistic

        if abs(value) >= abs(observed):
            count += 1

    return float(
        (count + 1) / (N_GENE_LABEL_PERMUTATIONS + 1)
    )


def loading_permutation_p(
    frozen_loadings: np.ndarray,
    external_loadings: np.ndarray,
    observed: float,
    seed: int,
) -> float:
    if not np.isfinite(observed):
        return np.nan
    rng = np.random.default_rng(seed)
    count = 0

    for _ in range(N_GENE_LABEL_PERMUTATIONS):
        permuted = rng.permutation(external_loadings)
        value = stats.spearmanr(
            frozen_loadings,
            permuted,
        ).statistic

        if abs(value) >= abs(observed):
            count += 1

    return float(
        (count + 1) / (N_GENE_LABEL_PERMUTATIONS + 1)
    )


def frozen_signed_score(
    expression_z: pd.DataFrame,
    genes: list[str],
    loadings: pd.Series,
) -> pd.Series:
    signs = np.sign(loadings.reindex(genes)).replace(0, 1)
    score = expression_z[genes].mul(
        signs,
        axis=1,
    ).mean(axis=1)
    return score


def split_half_reliability(
    expression_z: pd.DataFrame,
    genes: list[str],
    loadings: pd.Series,
    seed: int,
) -> dict[str, float]:
    if len(genes) < 4:
        return {
            "split_half_median": np.nan,
            "split_half_q05": np.nan,
            "split_half_q95": np.nan,
            "split_half_valid_repeats": 0,
        }

    rng = np.random.default_rng(seed)
    correlations = []

    for _ in range(N_SPLIT_HALF_REPEATS):
        shuffled = np.asarray(genes, dtype=object)
        shuffled = rng.permutation(shuffled)
        midpoint = len(shuffled) // 2
        first = shuffled[:midpoint].tolist()
        second = shuffled[midpoint:].tolist()

        first_score = frozen_signed_score(
            expression_z,
            first,
            loadings,
        )
        second_score = frozen_signed_score(
            expression_z,
            second,
            loadings,
        )
        correlation = first_score.corr(second_score)

        if np.isfinite(correlation):
            correlations.append(float(correlation))

    if not correlations:
        return {
            "split_half_median": np.nan,
            "split_half_q05": np.nan,
            "split_half_q95": np.nan,
            "split_half_valid_repeats": 0,
        }

    return {
        "split_half_median": float(np.median(correlations)),
        "split_half_q05": float(
            np.quantile(correlations, 0.05)
        ),
        "split_half_q95": float(
            np.quantile(correlations, 0.95)
        ),
        "split_half_valid_repeats": len(correlations),
    }


def gene_leave_one_out(
    expression_z: pd.DataFrame,
    module: str,
    genes: list[str],
    loadings: pd.Series,
) -> pd.DataFrame:
    full_score = frozen_signed_score(
        expression_z,
        genes,
        loadings,
    )
    rows = []

    if len(genes) <= MIN_MODULE_GENES:
        return pd.DataFrame()

    for gene in genes:
        subset = [item for item in genes if item != gene]
        score = frozen_signed_score(
            expression_z,
            subset,
            loadings,
        )
        rows.append(
            {
                "module_label": module,
                "left_out_gene": gene,
                "n_genes_remaining": len(subset),
                "correlation_with_full_score": float(
                    full_score.corr(score)
                ),
            }
        )

    return pd.DataFrame(rows)


def analyze_module(
    module: str,
    reference_z: pd.DataFrame,
    external_z: pd.DataFrame,
    weights: pd.DataFrame,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.Series]:
    part = weights[
        weights["module_label"].astype(str).eq(module)
    ].copy()
    part["canine_gene_symbol"] = (
        part["canine_gene_symbol"].map(clean_symbol)
    )
    part = part.drop_duplicates(
        "canine_gene_symbol",
        keep="first",
    )
    part = part.set_index("canine_gene_symbol")

    genes = [
        gene
        for gene in part.index
        if gene in reference_z.columns
        and gene in external_z.columns
    ]

    coverage = {
        "module_label": module,
        "n_frozen_genes": int(part.shape[0]),
        "n_common_genes": len(genes),
        "coverage_fraction": (
            len(genes) / part.shape[0]
            if part.shape[0]
            else np.nan
        ),
        "common_genes": ";".join(genes),
        "missing_external_genes": ";".join(
            gene
            for gene in part.index
            if gene not in external_z.columns
        ),
    }

    if len(genes) < MIN_MODULE_GENES:
        result = {
            "module_label": module,
            "n_common_genes": len(genes),
            "edge_spearman": np.nan,
            "edge_permutation_p": np.nan,
            "loading_spearman": np.nan,
            "loading_permutation_p": np.nan,
            "external_pc1_variance_explained": np.nan,
            "pc1_orientation_correlation_with_frozen_score": np.nan,
            "split_half_median": np.nan,
            "split_half_q05": np.nan,
            "split_half_q95": np.nan,
            "split_half_valid_repeats": 0,
            "estimable": False,
            "nonestimable_reason": "fewer_than_three_common_genes",
        }
        return result, coverage, pd.DataFrame(), pd.Series(
            np.nan,
            index=external_z.index,
        )

    reference_correlation = np.corrcoef(
        reference_z[genes].to_numpy(dtype=float),
        rowvar=False,
    )
    external_correlation = np.corrcoef(
        external_z[genes].to_numpy(dtype=float),
        rowvar=False,
    )

    reference_edges = upper_triangle(reference_correlation)
    external_edges = upper_triangle(external_correlation)
    edge_spearman = float(
        stats.spearmanr(
            reference_edges,
            external_edges,
        ).statistic
    )
    edge_p = permutation_gene_label_p(
        reference_edges,
        external_correlation,
        edge_spearman,
        seed,
    )

    frozen_loadings = pd.to_numeric(
        part.loc[genes, "risk_oriented_loading"],
        errors="coerce",
    ).fillna(0.0)

    signed_score = frozen_signed_score(
        external_z,
        genes,
        frozen_loadings,
    )

    pca = PCA(n_components=1, random_state=RANDOM_SEED)
    pc_score = pd.Series(
        pca.fit_transform(
            external_z[genes].to_numpy(dtype=float)
        ).ravel(),
        index=external_z.index,
    )
    pc_loadings = pca.components_[0].copy()

    orientation_correlation = pc_score.corr(signed_score)
    if np.isfinite(orientation_correlation) and orientation_correlation < 0:
        pc_score = -pc_score
        pc_loadings = -pc_loadings
        orientation_correlation = -orientation_correlation

    loading_spearman = float(
        stats.spearmanr(
            frozen_loadings.to_numpy(dtype=float),
            pc_loadings,
        ).statistic
    )
    loading_p = loading_permutation_p(
        frozen_loadings.to_numpy(dtype=float),
        pc_loadings,
        loading_spearman,
        seed + 1,
    )

    reliability = split_half_reliability(
        external_z,
        genes,
        frozen_loadings,
        seed + 2,
    )

    result = {
        "module_label": module,
        "n_common_genes": len(genes),
        "edge_spearman": edge_spearman,
        "edge_permutation_p": edge_p,
        "loading_spearman": loading_spearman,
        "loading_permutation_p": loading_p,
        "external_pc1_variance_explained": float(
            pca.explained_variance_ratio_[0]
        ),
        "pc1_orientation_correlation_with_frozen_score": (
            float(orientation_correlation)
            if np.isfinite(orientation_correlation)
            else np.nan
        ),
        **reliability,
        "estimable": True,
        "nonestimable_reason": "",
    }

    loo = gene_leave_one_out(
        external_z,
        module,
        genes,
        frozen_loadings,
    )

    return result, coverage, loo, signed_score


def variability_bins(
    reference_z: pd.DataFrame,
    external_z: pd.DataFrame,
) -> pd.Series:
    common = reference_z.columns.intersection(
        external_z.columns
    )
    reference_variance = reference_z[common].var(axis=0)
    external_variance = external_z[common].var(axis=0)

    combined_rank = (
        reference_variance.rank(pct=True)
        + external_variance.rank(pct=True)
    ) / 2.0

    return pd.qcut(
        combined_rank.rank(method="average"),
        q=min(N_VARIABILITY_BINS, len(combined_rank)),
        labels=False,
        duplicates="drop",
    )


def random_panel_controls(
    reference_z: pd.DataFrame,
    external_z: pd.DataFrame,
    weights: pd.DataFrame,
    observed: pd.DataFrame,
) -> pd.DataFrame:
    common = reference_z.columns.intersection(
        external_z.columns
    )
    bins = variability_bins(reference_z, external_z)

    frozen_union = set(
        weights.loc[
            weights["module_label"].isin(PRIMARY_MODULES),
            "canine_gene_symbol",
        ].map(clean_symbol)
    )
    candidate_genes = [
        gene for gene in common if gene not in frozen_union
    ]

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for module_index, module in enumerate(PRIMARY_MODULES):
        part = weights[
            weights["module_label"].astype(str).eq(module)
        ].copy()
        part["canine_gene_symbol"] = part[
            "canine_gene_symbol"
        ].map(clean_symbol)
        part = part.drop_duplicates("canine_gene_symbol")
        target_genes = [
            gene
            for gene in part["canine_gene_symbol"]
            if gene in common
        ]

        if len(target_genes) < MIN_MODULE_GENES:
            continue

        target_bins = bins.reindex(target_genes).to_numpy()
        null_values = []

        for repeat in range(N_RANDOM_PANELS):
            selected = []
            used = set()

            for target_bin in target_bins:
                pool = [
                    gene
                    for gene in candidate_genes
                    if gene not in used
                    and bins.get(gene, np.nan) == target_bin
                ]

                if not pool:
                    pool = [
                        gene
                        for gene in candidate_genes
                        if gene not in used
                    ]

                if not pool:
                    selected = []
                    break

                gene = str(rng.choice(pool))
                selected.append(gene)
                used.add(gene)

            if len(selected) != len(target_genes):
                continue

            reference_correlation = np.corrcoef(
                reference_z[selected].to_numpy(dtype=float),
                rowvar=False,
            )
            external_correlation = np.corrcoef(
                external_z[selected].to_numpy(dtype=float),
                rowvar=False,
            )
            edge = stats.spearmanr(
                upper_triangle(reference_correlation),
                upper_triangle(external_correlation),
            ).statistic

            if np.isfinite(edge):
                null_values.append(float(edge))

        observed_edge = float(
            observed.loc[
                observed["module_label"].eq(module),
                "edge_spearman",
            ].iloc[0]
        )

        null_array = np.asarray(null_values, dtype=float)
        empirical_p = (
            float(
                (
                    1
                    + np.sum(
                        np.abs(null_array) >= abs(observed_edge)
                    )
                )
                / (len(null_array) + 1)
            )
            if len(null_array) and np.isfinite(observed_edge)
            else np.nan
        )

        rows.append(
            {
                "module_label": module,
                "n_module_genes": len(target_genes),
                "n_random_panels": len(null_array),
                "observed_edge_spearman": observed_edge,
                "random_edge_median": (
                    float(np.median(null_array))
                    if len(null_array)
                    else np.nan
                ),
                "random_edge_q05": (
                    float(np.quantile(null_array, 0.05))
                    if len(null_array)
                    else np.nan
                ),
                "random_edge_q95": (
                    float(np.quantile(null_array, 0.95))
                    if len(null_array)
                    else np.nan
                ),
                "random_panel_empirical_p": empirical_p,
            }
        )

    return pd.DataFrame(rows)


def classify_results(
    structure: pd.DataFrame,
    random_controls: pd.DataFrame,
) -> pd.DataFrame:
    result = structure.merge(
        random_controls[
            [
                "module_label",
                "random_panel_empirical_p",
            ]
        ],
        on="module_label",
        how="left",
    )

    direct_p = pd.concat(
        [
            result[
                ["module_label", "edge_permutation_p"]
            ].rename(
                columns={"edge_permutation_p": "p"}
            ).assign(test="edge"),
            result[
                ["module_label", "loading_permutation_p"]
            ].rename(
                columns={"loading_permutation_p": "p"}
            ).assign(test="loading"),
        ],
        ignore_index=True,
    )
    direct_p["q_bh_8"] = bh_adjust(direct_p["p"])

    result = result.merge(
        direct_p[direct_p["test"].eq("edge")][
            ["module_label", "q_bh_8"]
        ].rename(columns={"q_bh_8": "edge_q_bh_8"}),
        on="module_label",
        how="left",
    )
    result = result.merge(
        direct_p[direct_p["test"].eq("loading")][
            ["module_label", "q_bh_8"]
        ].rename(
            columns={"q_bh_8": "loading_q_bh_8"}
        ),
        on="module_label",
        how="left",
    )

    classes = []
    for row in result.itertuples(index=False):
        edge_supported = bool(
            np.isfinite(row.edge_q_bh_8)
            and row.edge_q_bh_8 < 0.05
        )
        loading_supported = bool(
            np.isfinite(row.loading_q_bh_8)
            and row.loading_q_bh_8 < 0.05
        )
        random_supported = bool(
            np.isfinite(row.random_panel_empirical_p)
            and row.random_panel_empirical_p < 0.05
        )
        reliable = bool(
            np.isfinite(row.split_half_median)
            and row.split_half_median >= 0.60
        )

        if edge_supported and loading_supported and reliable:
            label = "strong_external_canine_representation_preservation"
        elif (
            (edge_supported or loading_supported)
            and reliable
        ):
            label = "partial_external_canine_representation_preservation"
        elif (
            random_supported
            or edge_supported
            or loading_supported
        ):
            label = "limited_specific_external_canine_signal"
        else:
            label = "no_clear_external_canine_representation_preservation"

        classes.append(label)

    result[
        "external_canine_representation_class"
    ] = classes
    return result


def create_heatmap(classification: pd.DataFrame) -> None:
    matrix = classification.set_index(
        "module_label"
    ).reindex(PRIMARY_MODULES)[
        [
            "edge_spearman",
            "loading_spearman",
            "split_half_median",
        ]
    ]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    image = ax.imshow(
        matrix.to_numpy(dtype=float),
        aspect="auto",
        vmin=-1,
        vmax=1,
    )
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(
        [
            "Correlation-edge\npreservation",
            "Frozen-loading\nconcordance",
            "Split-half\nreliability",
        ]
    )
    ax.set_yticks(np.arange(len(PRIMARY_MODULES)))
    ax.set_yticklabels(PRIMARY_MODULES)
    ax.set_title(
        "Independent canine cohort representation preservation"
    )

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix.iloc[row_index, column_index]
            text = "NA" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(HEATMAP_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(HEATMAP_PDF, bbox_inches="tight")
    plt.close(fig)


def write_readme(
    raw_path: Path,
    transform_diagnostics: dict[str, Any],
) -> None:
    text = f"""GSE239948 independent canine representation validation
Script version: {SCRIPT_VERSION}

Dataset
-------
Accession: GSE239948
Samples: expected 43 fresh-frozen canine osteosarcoma tumors.
Processed source: {raw_path}

Transformation
--------------
{json.dumps(transform_diagnostics, indent=2)}

Purpose
-------
Evaluate whether the frozen DOG2 programs preserve their transcriptional
representation in an independent canine osteosarcoma RNA-seq cohort.

No outcome model is fitted because GSE239948 does not provide a standardized
survival endpoint in the processed matrix.

Primary representation metrics
------------------------------
1. Spearman concordance of within-module gene-gene correlation edges between
   DOG2 and GSE239948.
2. Concordance between frozen DOG2 risk-oriented loadings and GSE239948 PC1
   loadings. The external PC1 sign is oriented using the frozen signed score,
   without outcomes.
3. Repeated non-overlapping split-half reliability of the frozen signed score.
4. Gene leave-one-out stability.
5. Variability-matched random-panel specificity controls.

Multiplicity
------------
BH correction is applied jointly across four edge-preservation and four
loading-concordance tests.

Guardrails
----------
- Frozen genes, loadings, signs, and tiers are unchanged.
- No GSE239948 outcome is used.
- GSE239948 tumors underwent different therapies; therefore treatment-related
  distribution differences are possible.
- Representation preservation is not prognostic validation.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("GSE239948 independent canine representation validation")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Design:")
    print("  Load the independent 43-sample canine osteosarcoma cohort.")
    print("  Match external genes to frozen canine symbols outcome-blind.")
    print("  Test edge preservation, loading concordance, and score reliability.")
    print("  Use gene-label permutations and variability-matched random panels.")
    print("")

    raw_path = locate_or_download_raw_file()

    reference = pd.read_csv(
        REFERENCE_EXPRESSION_FILE,
        index_col=0,
    )
    weights = pd.read_csv(STRICT_WEIGHTS_FILE)
    freeze = (
        json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
        if FREEZE_FILE.exists()
        else {}
    )

    raw_table = read_raw_table(raw_path)
    samples = sample_columns(raw_table)

    identifier_audit = identifier_pair_audit(
        table=raw_table,
        sample_cols=samples,
        weights=weights,
    )
    (
        gene_column,
        frozen_identifier_column,
        identifier_scheme,
    ) = choose_identifier_pair(identifier_audit)

    identifier_audit["selected_identifier_pair"] = (
        identifier_audit["external_column"].eq(gene_column)
        & identifier_audit["frozen_identifier_column"].eq(
            frozen_identifier_column
        )
        & identifier_audit["identifier_scheme"].eq(
            identifier_scheme
        )
    )
    identifier_audit.to_csv(
        OUTPUT_IDENTIFIER_PAIR_AUDIT,
        index=False,
    )

    weights["canine_gene_symbol"] = weights[
        frozen_identifier_column
    ].map(
        lambda value: normalize_identifier(
            value,
            identifier_scheme,
        )
    )

    external_raw, gene_map = collapse_to_identifiers(
        table=raw_table,
        gene_column=gene_column,
        sample_cols=samples,
        identifier_scheme=identifier_scheme,
    )
    external, transform_diagnostics = choose_transform(
        external_raw
    )

    reference, reference_gene_map = remap_reference_expression(
        reference=reference,
        weights=weights,
        frozen_identifier_column=frozen_identifier_column,
        identifier_scheme=identifier_scheme,
    )

    external = external.loc[
        :,
        ~pd.Index(external.columns).duplicated(keep="first"),
    ]

    reference_z = zscore_columns(reference)
    external_z = zscore_columns(external)

    input_audit = identifier_audit.copy()
    input_audit["n_detected_sample_columns"] = len(samples)
    input_audit["external_matrix_rows"] = raw_table.shape[0]
    input_audit["external_matrix_columns"] = raw_table.shape[1]
    input_audit["selected_external_column"] = gene_column
    input_audit["selected_frozen_identifier_column"] = (
        frozen_identifier_column
    )
    input_audit["selected_identifier_scheme"] = identifier_scheme
    input_audit.to_csv(OUTPUT_INPUT_AUDIT, index=False)

    gene_map = gene_map.merge(
        reference_gene_map,
        on="analysis_gene_id",
        how="left",
    )
    gene_map.to_csv(OUTPUT_GENE_MAP, index=False)
    external.to_csv(OUTPUT_EXPRESSION)

    print("")
    print("Selected identifier mapping:")
    print(f"  External column: {gene_column}")
    print(f"  Frozen identifier column: {frozen_identifier_column}")
    print(f"  Identifier scheme: {identifier_scheme}")
    print(
        identifier_audit.head(10)[
            [
                "external_column",
                "frozen_identifier_column",
                "identifier_scheme",
                "n_unique_overlap",
                "n_primary_modules_with_at_least_3_genes",
                "M34_overlap",
                "M11_overlap",
                "M24_overlap",
                "M40_overlap",
                "selected_identifier_pair",
            ]
        ].to_string(index=False)
    )

    structure_rows = []
    coverage_rows = []
    reliability_rows = []
    loo_tables = []
    score_table = pd.DataFrame(index=external.index)

    for module_index, module in enumerate(PRIMARY_MODULES):
        result, coverage, loo, score = analyze_module(
            module=module,
            reference_z=reference_z,
            external_z=external_z,
            weights=weights,
            seed=RANDOM_SEED + module_index * 100,
        )
        structure_rows.append(result)
        coverage_rows.append(coverage)

        reliability_rows.append(
            {
                "module_label": module,
                "n_common_genes": result["n_common_genes"],
                "split_half_median": result[
                    "split_half_median"
                ],
                "split_half_q05": result["split_half_q05"],
                "split_half_q95": result["split_half_q95"],
                "split_half_valid_repeats": result[
                    "split_half_valid_repeats"
                ],
                "minimum_gene_loo_correlation": (
                    float(
                        loo["correlation_with_full_score"].min()
                    )
                    if not loo.empty
                    else np.nan
                ),
                "median_gene_loo_correlation": (
                    float(
                        loo["correlation_with_full_score"].median()
                    )
                    if not loo.empty
                    else np.nan
                ),
            }
        )

        if not loo.empty:
            loo_tables.append(loo)

        score_table[
            f"{module}__strict_signed_mean_z"
        ] = (
            (score - score.mean()) / score.std()
            if score.notna().sum() > 1
            else score
        )

    structure = pd.DataFrame(structure_rows)
    coverage = pd.DataFrame(coverage_rows)
    reliability = pd.DataFrame(reliability_rows)
    gene_loo = (
        pd.concat(loo_tables, ignore_index=True)
        if loo_tables
        else pd.DataFrame()
    )

    random_controls = random_panel_controls(
        reference_z=reference_z,
        external_z=external_z,
        weights=weights,
        observed=structure,
    )
    classification = classify_results(
        structure=structure,
        random_controls=random_controls,
    )

    score_table.to_csv(OUTPUT_SCORES)
    coverage.to_csv(OUTPUT_COVERAGE, index=False)
    structure.to_csv(OUTPUT_STRUCTURE, index=False)
    reliability.to_csv(OUTPUT_RELIABILITY, index=False)
    gene_loo.to_csv(OUTPUT_GENE_LOO, index=False)
    random_controls.to_csv(OUTPUT_RANDOM, index=False)
    classification.to_csv(
        OUTPUT_CLASSIFICATION,
        index=False,
    )

    create_heatmap(classification)
    write_readme(raw_path, transform_diagnostics)

    input_paths = [
        raw_path,
        REFERENCE_EXPRESSION_FILE,
        STRICT_WEIGHTS_FILE,
    ]
    if FREEZE_FILE.exists():
        input_paths.append(FREEZE_FILE)

    output_paths = [
        OUTPUT_INPUT_AUDIT,
        OUTPUT_IDENTIFIER_PAIR_AUDIT,
        OUTPUT_GENE_MAP,
        OUTPUT_EXPRESSION,
        OUTPUT_SCORES,
        OUTPUT_COVERAGE,
        OUTPUT_STRUCTURE,
        OUTPUT_RELIABILITY,
        OUTPUT_GENE_LOO,
        OUTPUT_RANDOM,
        OUTPUT_CLASSIFICATION,
        OUTPUT_README,
        HEATMAP_PNG,
        HEATMAP_PDF,
    ]

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "accession": ACCESSION,
        "n_external_samples": int(external.shape[0]),
        "n_external_genes": int(external.shape[1]),
        "selected_gene_identifier_column": gene_column,
        "selected_frozen_identifier_column": frozen_identifier_column,
        "selected_identifier_scheme": identifier_scheme,
        "transform_diagnostics": transform_diagnostics,
        "frozen_definition": freeze,
        "gene_label_permutations": N_GENE_LABEL_PERMUTATIONS,
        "random_panels": N_RANDOM_PANELS,
        "split_half_repeats": N_SPLIT_HALF_REPEATS,
        "outcome_loaded": False,
        "guardrails": [
            "No external outcome was loaded.",
            "Frozen module genes, loadings, signs, and tiers were unchanged.",
            "External PC1 orientation used only the frozen signed score.",
            "Representation preservation is not prognostic validation.",
            "Therapy heterogeneity may contribute to expression-domain shift.",
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
    print("External matrix preflight")
    print("=" * 80)
    print(
        pd.DataFrame(
            [
                {
                    "raw_rows": raw_table.shape[0],
                    "raw_columns": raw_table.shape[1],
                    "sample_columns": len(samples),
                    "selected_gene_column": gene_column,
                    "selected_frozen_identifier_column": frozen_identifier_column,
                    "identifier_scheme": identifier_scheme,
                    "processed_samples": external.shape[0],
                    "processed_genes": external.shape[1],
                    "transform": transform_diagnostics["transform"],
                }
            ]
        ).to_string(index=False)
    )
    print("")
    print(
        input_audit.head(10).to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Frozen-program external canine coverage")
    print("=" * 80)
    print(
        coverage[
            [
                "module_label",
                "n_frozen_genes",
                "n_common_genes",
                "coverage_fraction",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("External canine representation preservation")
    print("=" * 80)
    print(
        classification[
            [
                "module_label",
                "n_common_genes",
                "edge_spearman",
                "edge_q_bh_8",
                "loading_spearman",
                "loading_q_bh_8",
                "split_half_median",
                "random_panel_empirical_p",
                "external_canine_representation_class",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Frozen-score reliability")
    print("=" * 80)
    print(reliability.to_string(index=False))

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("GSE239948 is an independent canine expression cohort, not an outcome-validation cohort.")
    print("No external survival or treatment-response endpoint was used.")
    print("Strong preservation requires edge, loading, and split-half support.")
    print("Therapy heterogeneity may contribute to external expression differences.")
    print("Human transfer tiers and project-wide multiplicity remain unchanged.")

    print("")
    print("Saved:")
    for path in output_paths + [OUTPUT_MANIFEST]:
        if path.exists():
            print(path)
    print("Done.")


if __name__ == "__main__":
    main()
