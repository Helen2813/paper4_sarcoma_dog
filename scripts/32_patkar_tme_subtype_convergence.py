from __future__ import annotations

from itertools import combinations
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
import matplotlib.pyplot as plt

SCRIPT_VERSION = "32-patkar-tme-convergence-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "patkar_tme_convergence"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

EXPRESSION_FILE = (
    PROCESSED_DIR / "GSE238110_DOG2_expression_log2cpm_matched_allgenes.csv"
)
CLINICAL_FILE = (
    PROCESSED_DIR / "GSE238110_DOG2_clinical_matched_indexed.csv"
)
STRICT_WEIGHTS_FILE = (
    RESULTS_DIR / "GSE238110_frozen_transfer_gene_weights_strict.csv"
)
FREEZE_FILE = (
    RESULTS_DIR / "GSE238110_frozen_transfer_program_freeze.json"
)

PATKAR_FILENAME = "ccr-24-1854_supplementary_table_s10_suppts10.xlsx"
PATKAR_DOWNLOAD_URL = (
    "https://aacr.figshare.com/ndownloader/files/51205537"
)

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]
SUBTYPE_ORDER = ["ID", "IE", "IE-ECM"]
N_PERMUTATIONS = 10000
RANDOM_SEED = 42

OUTPUT_MATCHED = RESULTS_DIR / "Patkar_TME_frozen_program_scores.csv"
OUTPUT_OMNIBUS = RESULTS_DIR / "Patkar_TME_module_omnibus_associations.csv"
OUTPUT_PAIRWISE = RESULTS_DIR / "Patkar_TME_module_pairwise_associations.csv"
OUTPUT_TARGETED = RESULTS_DIR / "Patkar_TME_targeted_convergence_tests.csv"
OUTPUT_MEDIANS = RESULTS_DIR / "Patkar_TME_subtype_score_summary.csv"
OUTPUT_MATCH_AUDIT = RESULTS_DIR / "Patkar_TME_sample_matching_audit.csv"
OUTPUT_MANIFEST = RESULTS_DIR / "Patkar_TME_convergence_manifest.json"
OUTPUT_README = RESULTS_DIR / "Patkar_TME_convergence_README.txt"


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


def normalize_patient_id(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na"}:
        return ""

    try:
        numeric = float(text)
        if np.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    except ValueError:
        pass

    groups = re.findall(r"\d+", text)
    if not groups:
        return ""

    # DOG2 identifiers in Table S10 are numeric patient IDs.
    # Prefer the final numeric token after removing leading zeros.
    return str(int(groups[-1]))


def locate_or_download_patkar_table() -> Path:
    exact_candidates = [
        RAW_DIR / "canine_clinical_DOG2" / PATKAR_FILENAME,
        RAW_DIR / "DOG2" / PATKAR_FILENAME,
        RAW_DIR / PATKAR_FILENAME,
        PROJECT_ROOT / PATKAR_FILENAME,
    ]
    for path in exact_candidates:
        if path.exists():
            print(f"Patkar Table S10 found: {path}")
            return path

    recursive = list(PROJECT_ROOT.rglob(PATKAR_FILENAME))
    if recursive:
        print(f"Patkar Table S10 found: {recursive[0]}")
        return recursive[0]

    destination = RAW_DIR / "canine_clinical_DOG2" / PATKAR_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Patkar Table S10 not found locally; downloading to: {destination}")
    urllib.request.urlretrieve(PATKAR_DOWNLOAD_URL, destination)
    return destination


def load_patkar_table(path: Path) -> pd.DataFrame:
    workbook = pd.ExcelFile(path)
    selected_sheet = None
    selected_table = None

    for sheet_name in workbook.sheet_names:
        table = pd.read_excel(path, sheet_name=sheet_name)
        normalized_columns = {
            str(column).strip().lower(): column
            for column in table.columns
        }
        if (
            "patient id" in normalized_columns
            and "primary_immune_subtype" in table.columns
        ):
            selected_sheet = sheet_name
            selected_table = table
            break

    if selected_table is None:
        for sheet_name in workbook.sheet_names:
            table = pd.read_excel(path, sheet_name=sheet_name)
            if "primary_immune_subtype" in table.columns:
                selected_sheet = sheet_name
                selected_table = table
                break

    if selected_table is None:
        raise ValueError(
            "Could not find a sheet containing primary_immune_subtype. "
            f"Sheets: {workbook.sheet_names}"
        )

    print(f"Patkar sheet selected: {selected_sheet}")
    table = selected_table.copy()

    patient_col = next(
        (
            column
            for column in table.columns
            if str(column).strip().lower() == "patient id"
        ),
        None,
    )
    if patient_col is None:
        raise ValueError(
            f"Patient ID column not found. Columns: {list(table.columns)}"
        )

    table["patient_id_normalized"] = table[patient_col].map(
        normalize_patient_id
    )
    table["primary_immune_subtype"] = (
        table["primary_immune_subtype"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace(
            {
                "IMMUNE DESERT": "ID",
                "IMMUNE-ENRICHED": "IE",
                "IMMUNE ENRICHED": "IE",
                "IMMUNE-ENRICHED DENSE ECM-LIKE": "IE-ECM",
                "IMMUNE ENRICHED DENSE ECM-LIKE": "IE-ECM",
                "IE_ECM": "IE-ECM",
            }
        )
    )
    table = table[
        table["patient_id_normalized"].ne("")
        & table["primary_immune_subtype"].isin(SUBTYPE_ORDER)
    ].copy()

    duplicated = table["patient_id_normalized"].duplicated(keep=False)
    if duplicated.any():
        duplicate_ids = sorted(
            table.loc[duplicated, "patient_id_normalized"].unique()
        )
        raise RuntimeError(
            "Duplicate patient IDs in Patkar Table S10: "
            + ", ".join(duplicate_ids[:20])
        )

    return table


def candidate_id_mapping(
    data: pd.DataFrame,
    target_ids: set[str],
    source_name: str,
) -> pd.DataFrame:
    candidates: list[dict[str, Any]] = []

    index_values = pd.Series(data.index, index=data.index)
    index_normalized = index_values.map(normalize_patient_id)
    candidates.append(
        {
            "source": source_name,
            "candidate": "__index__",
            "n_nonempty": int(index_normalized.ne("").sum()),
            "n_unique": int(index_normalized[index_normalized.ne("")].nunique()),
            "n_overlap": int(index_normalized.isin(target_ids).sum()),
            "overlap_fraction": float(index_normalized.isin(target_ids).mean()),
        }
    )

    for column in data.columns:
        values = data[column].map(normalize_patient_id)
        if values.ne("").sum() == 0:
            continue
        candidates.append(
            {
                "source": source_name,
                "candidate": str(column),
                "n_nonempty": int(values.ne("").sum()),
                "n_unique": int(values[values.ne("")].nunique()),
                "n_overlap": int(values.isin(target_ids).sum()),
                "overlap_fraction": float(values.isin(target_ids).mean()),
            }
        )

    return pd.DataFrame(candidates).sort_values(
        ["n_overlap", "n_unique"],
        ascending=[False, False],
    )


def map_expression_to_patients(
    expression: pd.DataFrame,
    clinical: pd.DataFrame,
    patkar: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_ids = set(patkar["patient_id_normalized"])

    expression_audit = candidate_id_mapping(
        expression,
        target_ids,
        "expression",
    )
    clinical_audit = candidate_id_mapping(
        clinical,
        target_ids,
        "clinical",
    )
    audit = pd.concat(
        [expression_audit, clinical_audit],
        ignore_index=True,
    )

    best_expression = expression_audit.iloc[0]
    best_clinical = clinical_audit.iloc[0]

    if best_expression["n_overlap"] >= 0.8 * expression.shape[0]:
        candidate = best_expression["candidate"]
        if candidate == "__index__":
            patient_ids = pd.Series(
                expression.index,
                index=expression.index,
            ).map(normalize_patient_id)
        else:
            raise RuntimeError(
                "Expression identifiers appear to be stored in an expression "
                "column, which is unexpected for a gene-expression matrix."
            )
    else:
        candidate = best_clinical["candidate"]
        if candidate == "__index__":
            clinical_patient_ids = pd.Series(
                clinical.index,
                index=clinical.index,
            ).map(normalize_patient_id)
        else:
            clinical_patient_ids = clinical[candidate].map(
                normalize_patient_id
            )

        if not expression.index.equals(clinical.index):
            shared = expression.index.intersection(clinical.index)
            if shared.shape[0] < 0.8 * expression.shape[0]:
                raise RuntimeError(
                    "Expression and clinical indexes do not align sufficiently "
                    "for fallback patient-ID mapping."
                )
            expression = expression.loc[shared].copy()
            clinical_patient_ids = clinical_patient_ids.loc[shared]

        patient_ids = clinical_patient_ids.reindex(expression.index)

    mapping = pd.DataFrame(
        {
            "expression_sample_id": expression.index.astype(str),
            "patient_id_normalized": patient_ids.values,
        },
        index=expression.index,
    )
    mapping = mapping[
        mapping["patient_id_normalized"].isin(target_ids)
    ].copy()

    duplicated = mapping["patient_id_normalized"].duplicated(keep=False)
    if duplicated.any():
        duplicated_ids = sorted(
            mapping.loc[duplicated, "patient_id_normalized"].unique()
        )
        raise RuntimeError(
            "Multiple expression samples map to the same Patkar patient ID: "
            + ", ".join(duplicated_ids[:20])
        )

    return mapping, audit


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


def compute_frozen_canine_scores(
    expression: pd.DataFrame,
    weights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.DataFrame(index=expression.index)
    coverage_rows: list[dict[str, Any]] = []

    for module in PRIMARY_MODULES:
        part = weights[
            weights["module_label"].astype(str).eq(module)
        ].copy()
        part["canine_gene"] = part["canine_gene"].astype(str)
        part = part.drop_duplicates("canine_gene", keep="first")

        requested = part["canine_gene"].tolist()
        available = [
            gene for gene in requested if gene in expression.columns
        ]

        coverage_rows.append(
            {
                "module_label": module,
                "n_frozen_strict_canine_genes": len(requested),
                "n_available_canine_genes": len(available),
                "coverage_fraction": (
                    len(available) / len(requested)
                    if requested
                    else np.nan
                ),
                "available_genes": ";".join(available),
                "missing_genes": ";".join(
                    gene for gene in requested if gene not in available
                ),
            }
        )

        if len(available) < 3:
            continue

        z = zscore_columns(expression[available])
        available = list(z.columns)
        indexed = part.set_index("canine_gene").loc[available]
        loadings = pd.to_numeric(
            indexed["risk_oriented_loading"],
            errors="coerce",
        ).fillna(0.0)

        signs = np.sign(loadings).replace(0, 1)
        signed_mean = z.mul(signs, axis=1).mean(axis=1)

        if loadings.abs().sum() > 0:
            normalized_weights = loadings / loadings.abs().sum()
            weighted = z.mul(normalized_weights, axis=1).sum(axis=1)
        else:
            weighted = pd.Series(np.nan, index=z.index)

        scores[f"{module}__strict_signed_mean_z"] = zscore_series(
            signed_mean
        )
        scores[f"{module}__strict_canine_weighted_z"] = zscore_series(
            weighted
        )

    return scores, pd.DataFrame(coverage_rows)


def kruskal_epsilon_squared(
    groups: list[np.ndarray],
    h_statistic: float,
) -> float:
    n = int(sum(len(group) for group in groups))
    k = len(groups)
    if n <= k:
        return np.nan
    return float(max(0.0, (h_statistic - k + 1) / (n - k)))


def permutation_kruskal_p(
    values: np.ndarray,
    labels: np.ndarray,
    observed_h: float,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    unique_labels = np.unique(labels)
    count = 0

    for _ in range(N_PERMUTATIONS):
        permuted = rng.permutation(labels)
        groups = [
            values[permuted == label]
            for label in unique_labels
        ]
        h = stats.kruskal(*groups).statistic
        if h >= observed_h:
            count += 1

    return float((count + 1) / (N_PERMUTATIONS + 1))


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return np.nan
    u = stats.mannwhitneyu(
        x,
        y,
        alternative="two-sided",
    ).statistic
    return float((2.0 * u) / (len(x) * len(y)) - 1.0)


def omnibus_tests(
    matched: pd.DataFrame,
    score_suffix: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for module_index, module in enumerate(PRIMARY_MODULES):
        score_col = f"{module}{score_suffix}"
        part = matched[
            ["primary_immune_subtype", score_col]
        ].dropna()

        groups = [
            part.loc[
                part["primary_immune_subtype"].eq(subtype),
                score_col,
            ].values
            for subtype in SUBTYPE_ORDER
        ]
        if any(len(group) < 2 for group in groups):
            continue

        result = stats.kruskal(*groups)
        values = part[score_col].values
        labels = part["primary_immune_subtype"].values

        rows.append(
            {
                "module_label": module,
                "score_variant": score_suffix.lstrip("__"),
                "n": part.shape[0],
                "n_ID": len(groups[0]),
                "n_IE": len(groups[1]),
                "n_IE_ECM": len(groups[2]),
                "kruskal_h": float(result.statistic),
                "kruskal_p": float(result.pvalue),
                "permutation_kruskal_p": permutation_kruskal_p(
                    values,
                    labels,
                    float(result.statistic),
                    RANDOM_SEED + module_index * 1000,
                ),
                "epsilon_squared": kruskal_epsilon_squared(
                    groups,
                    float(result.statistic),
                ),
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["kruskal_q_bh"] = bh_adjust(result["kruskal_p"])
        result["permutation_kruskal_q_bh"] = bh_adjust(
            result["permutation_kruskal_p"]
        )
    return result


def pairwise_tests(
    matched: pd.DataFrame,
    score_suffix: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for module in PRIMARY_MODULES:
        score_col = f"{module}{score_suffix}"

        for group_1, group_2 in combinations(SUBTYPE_ORDER, 2):
            x = matched.loc[
                matched["primary_immune_subtype"].eq(group_1),
                score_col,
            ].dropna().values
            y = matched.loc[
                matched["primary_immune_subtype"].eq(group_2),
                score_col,
            ].dropna().values

            if len(x) < 2 or len(y) < 2:
                continue

            test = stats.mannwhitneyu(
                x,
                y,
                alternative="two-sided",
            )
            rows.append(
                {
                    "module_label": module,
                    "score_variant": score_suffix.lstrip("__"),
                    "group_1": group_1,
                    "group_2": group_2,
                    "n_group_1": len(x),
                    "n_group_2": len(y),
                    "median_group_1": float(np.median(x)),
                    "median_group_2": float(np.median(y)),
                    "median_difference_group_1_minus_group_2": float(
                        np.median(x) - np.median(y)
                    ),
                    "cliffs_delta_group_1_vs_group_2": cliffs_delta(x, y),
                    "mann_whitney_p": float(test.pvalue),
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["mann_whitney_q_global_12"] = bh_adjust(
            result["mann_whitney_p"]
        )
        result["mann_whitney_q_within_module"] = np.nan
        for module in PRIMARY_MODULES:
            mask = result["module_label"].eq(module)
            result.loc[
                mask,
                "mann_whitney_q_within_module",
            ] = bh_adjust(
                result.loc[mask, "mann_whitney_p"]
            )
    return result


def targeted_convergence_tests(
    matched: pd.DataFrame,
    score_suffix: str,
) -> pd.DataFrame:
    hypotheses = [
        {
            "hypothesis": "M34_immune_enriched_any_vs_immune_desert",
            "module_label": "M34",
            "group_1_label": "IE_or_IE-ECM",
            "group_2_label": "ID",
            "group_1_subtypes": {"IE", "IE-ECM"},
            "group_2_subtypes": {"ID"},
            "rationale": (
                "Tests whether the frozen immune/myeloid-like M34 program "
                "tracks the independently derived immune-enriched TME axis."
            ),
        },
        {
            "hypothesis": "M11_IE_ECM_vs_IE",
            "module_label": "M11",
            "group_1_label": "IE-ECM",
            "group_2_label": "IE",
            "group_1_subtypes": {"IE-ECM"},
            "group_2_subtypes": {"IE"},
            "rationale": (
                "Tests whether the frozen angiogenesis/ECM-remodeling-like "
                "M11 program distinguishes the dense-ECM immune-enriched subtype."
            ),
        },
    ]

    rows: list[dict[str, Any]] = []

    for hypothesis in hypotheses:
        module = hypothesis["module_label"]
        score_col = f"{module}{score_suffix}"

        x = matched.loc[
            matched["primary_immune_subtype"].isin(
                hypothesis["group_1_subtypes"]
            ),
            score_col,
        ].dropna().values
        y = matched.loc[
            matched["primary_immune_subtype"].isin(
                hypothesis["group_2_subtypes"]
            ),
            score_col,
        ].dropna().values

        test = stats.mannwhitneyu(
            x,
            y,
            alternative="two-sided",
        )

        rows.append(
            {
                "hypothesis": hypothesis["hypothesis"],
                "module_label": module,
                "score_variant": score_suffix.lstrip("__"),
                "group_1": hypothesis["group_1_label"],
                "group_2": hypothesis["group_2_label"],
                "n_group_1": len(x),
                "n_group_2": len(y),
                "median_group_1": float(np.median(x)),
                "median_group_2": float(np.median(y)),
                "median_difference_group_1_minus_group_2": float(
                    np.median(x) - np.median(y)
                ),
                "cliffs_delta_group_1_vs_group_2": cliffs_delta(x, y),
                "mann_whitney_p": float(test.pvalue),
                "rationale": hypothesis["rationale"],
            }
        )

    result = pd.DataFrame(rows)
    result["targeted_q_bh_2"] = bh_adjust(
        result["mann_whitney_p"]
    )
    return result


def score_summary(
    matched: pd.DataFrame,
    score_suffix: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for module in PRIMARY_MODULES:
        score_col = f"{module}{score_suffix}"
        for subtype in SUBTYPE_ORDER:
            values = matched.loc[
                matched["primary_immune_subtype"].eq(subtype),
                score_col,
            ].dropna()

            rows.append(
                {
                    "module_label": module,
                    "score_variant": score_suffix.lstrip("__"),
                    "primary_immune_subtype": subtype,
                    "n": values.shape[0],
                    "mean": float(values.mean()),
                    "sd": float(values.std()),
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )

    return pd.DataFrame(rows)


def plot_module_boxplots(
    matched: pd.DataFrame,
    score_suffix: str,
) -> list[Path]:
    paths: list[Path] = []
    rng = np.random.default_rng(RANDOM_SEED)

    for module in PRIMARY_MODULES:
        score_col = f"{module}{score_suffix}"
        groups = [
            matched.loc[
                matched["primary_immune_subtype"].eq(subtype),
                score_col,
            ].dropna().values
            for subtype in SUBTYPE_ORDER
        ]

        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        ax.boxplot(
            groups,
            labels=SUBTYPE_ORDER,
            showfliers=False,
        )

        for position, values in enumerate(groups, start=1):
            jitter = rng.normal(0, 0.045, size=len(values))
            ax.scatter(
                np.full(len(values), position) + jitter,
                values,
                alpha=0.55,
                s=18,
            )

        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xlabel("Patkar primary TME subtype")
        ax.set_ylabel("Frozen strict canine risk-oriented score (z)")
        ax.set_title(
            f"{module}: frozen program score across Patkar TME subtypes"
        )
        fig.tight_layout()

        png = FIGURES_DIR / f"Patkar_TME_{module}_boxplot.png"
        pdf = FIGURES_DIR / f"Patkar_TME_{module}_boxplot.pdf"
        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
        paths.extend([png, pdf])

    return paths


def plot_median_heatmap(summary: pd.DataFrame) -> list[Path]:
    matrix = (
        summary.pivot(
            index="module_label",
            columns="primary_immune_subtype",
            values="median",
        )
        .reindex(index=PRIMARY_MODULES, columns=SUBTYPE_ORDER)
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    image = ax.imshow(matrix.values, aspect="auto")
    ax.set_xticks(np.arange(len(SUBTYPE_ORDER)))
    ax.set_xticklabels(SUBTYPE_ORDER)
    ax.set_yticks(np.arange(len(PRIMARY_MODULES)))
    ax.set_yticklabels(PRIMARY_MODULES)
    ax.set_xlabel("Patkar primary TME subtype")
    ax.set_title("Median frozen risk-oriented program scores")

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            label = "NA" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=9,
            )

    fig.colorbar(image, ax=ax, shrink=0.82)
    fig.tight_layout()

    png = FIGURES_DIR / "Patkar_TME_module_median_heatmap.png"
    pdf = FIGURES_DIR / "Patkar_TME_module_median_heatmap.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def write_readme() -> None:
    text = f"""Patkar TME-subtype convergence analysis
Script version: {SCRIPT_VERSION}

Purpose
-------
This analysis tests whether frozen canine risk-oriented transcriptional
programs align with the independently developed deconvolution-based TME
subtypes reported by Patkar et al.

Important limitation
--------------------
The frozen programs and Patkar subtypes are evaluated in overlapping DOG2
samples and both use the same bulk transcriptomic data. Therefore, this is
not an independent external validation. It is an orthogonal-method biological
convergence/annotation analysis.

Primary targeted convergence questions
--------------------------------------
1. Does M34 differ between immune-enriched tumors (IE or IE-ECM) and
   immune-desert tumors (ID)?
2. Does M11 differ between IE-ECM and IE tumors?

Additional analyses
-------------------
- Kruskal-Wallis tests across ID, IE, and IE-ECM for M34, M11, M24, and M40.
- Pairwise Mann-Whitney tests with multiplicity correction.
- Strict signed-mean score is primary.
- Frozen canine PCA-weighted score is a sensitivity variant.

Guardrails
----------
No subtype label is used to select genes, alter weights, orient scores, revise
validation tiers, or change the locked human outcome evidence hierarchy.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def create_manifest(
    input_paths: list[Path],
    output_paths: list[Path],
    figure_paths: list[Path],
    table_s10_path: Path,
) -> None:
    payload = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "patkar_table_s10_path": str(table_s10_path),
        "patkar_table_s10_sha256": sha256_file(table_s10_path),
        "primary_modules": PRIMARY_MODULES,
        "subtype_order": SUBTYPE_ORDER,
        "permutations_per_omnibus_test": N_PERMUTATIONS,
        "interpretation": (
            "Orthogonal-method convergence within overlapping DOG2 samples; "
            "not independent external validation."
        ),
        "inputs": {},
        "outputs": {},
    }

    for path in input_paths:
        if path.exists():
            payload["inputs"][path.name] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }

    for path in output_paths + figure_paths:
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
    print("Patkar TME-subtype convergence of frozen canine programs")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Design:")
    print("  Match Patkar Table S10 subtype assignments to DOG2 samples.")
    print("  Reconstruct frozen strict canine risk-oriented program scores.")
    print("  Test module-score differences across ID, IE, and IE-ECM subtypes.")
    print("  Treat the result as same-cohort orthogonal-method convergence, not external validation.")
    print("")

    expression = read_required_csv(
        EXPRESSION_FILE,
        index_col=0,
    )
    clinical = read_required_csv(
        CLINICAL_FILE,
        index_col=0,
    )
    weights = read_required_csv(
        STRICT_WEIGHTS_FILE,
    )

    patkar_path = locate_or_download_patkar_table()
    patkar = load_patkar_table(patkar_path)

    sample_mapping, matching_audit = map_expression_to_patients(
        expression=expression,
        clinical=clinical,
        patkar=patkar,
    )
    matching_audit.to_csv(OUTPUT_MATCH_AUDIT, index=False)

    mapped_expression = expression.loc[
        sample_mapping.index
    ].copy()

    scores, coverage = compute_frozen_canine_scores(
        expression=mapped_expression,
        weights=weights,
    )

    matched = (
        sample_mapping
        .join(scores, how="inner")
        .reset_index(drop=True)
        .merge(
            patkar,
            on="patient_id_normalized",
            how="inner",
            validate="one_to_one",
        )
    )

    matched.to_csv(OUTPUT_MATCHED, index=False)

    signed_suffix = "__strict_signed_mean_z"
    weighted_suffix = "__strict_canine_weighted_z"

    omnibus_signed = omnibus_tests(matched, signed_suffix)
    omnibus_weighted = omnibus_tests(matched, weighted_suffix)
    omnibus = pd.concat(
        [omnibus_signed, omnibus_weighted],
        ignore_index=True,
    )
    omnibus.to_csv(OUTPUT_OMNIBUS, index=False)

    pairwise_signed = pairwise_tests(matched, signed_suffix)
    pairwise_weighted = pairwise_tests(matched, weighted_suffix)
    pairwise = pd.concat(
        [pairwise_signed, pairwise_weighted],
        ignore_index=True,
    )
    pairwise.to_csv(OUTPUT_PAIRWISE, index=False)

    targeted_signed = targeted_convergence_tests(
        matched,
        signed_suffix,
    )
    targeted_weighted = targeted_convergence_tests(
        matched,
        weighted_suffix,
    )
    targeted = pd.concat(
        [targeted_signed, targeted_weighted],
        ignore_index=True,
    )
    targeted.to_csv(OUTPUT_TARGETED, index=False)

    summary_signed = score_summary(matched, signed_suffix)
    summary_weighted = score_summary(matched, weighted_suffix)
    subtype_summary = pd.concat(
        [summary_signed, summary_weighted],
        ignore_index=True,
    )
    subtype_summary.to_csv(OUTPUT_MEDIANS, index=False)

    figure_paths = plot_module_boxplots(
        matched,
        signed_suffix,
    )
    figure_paths.extend(
        plot_median_heatmap(summary_signed)
    )

    write_readme()

    output_paths = [
        OUTPUT_MATCHED,
        OUTPUT_OMNIBUS,
        OUTPUT_PAIRWISE,
        OUTPUT_TARGETED,
        OUTPUT_MEDIANS,
        OUTPUT_MATCH_AUDIT,
        OUTPUT_README,
    ]
    create_manifest(
        input_paths=[
            EXPRESSION_FILE,
            CLINICAL_FILE,
            STRICT_WEIGHTS_FILE,
            FREEZE_FILE,
        ],
        output_paths=output_paths,
        figure_paths=figure_paths,
        table_s10_path=patkar_path,
    )

    print("")
    print("=" * 80)
    print("DOG2–Patkar sample matching")
    print("=" * 80)
    print(f"Expression samples: {expression.shape[0]}")
    print(f"Patkar subtype rows: {patkar.shape[0]}")
    print(f"Matched one-to-one samples: {matched.shape[0]}")
    print("")
    print("Subtype counts:")
    print(
        matched["primary_immune_subtype"]
        .value_counts()
        .reindex(SUBTYPE_ORDER)
        .to_string()
    )

    print("")
    print("=" * 80)
    print("Frozen strict canine score coverage")
    print("=" * 80)
    print(coverage.to_string(index=False))

    print("")
    print("=" * 80)
    print("Patkar TME omnibus associations")
    print("=" * 80)
    print(
        omnibus[
            [
                "module_label",
                "score_variant",
                "n",
                "kruskal_h",
                "kruskal_p",
                "kruskal_q_bh",
                "permutation_kruskal_p",
                "permutation_kruskal_q_bh",
                "epsilon_squared",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Targeted biological convergence tests")
    print("=" * 80)
    print(
        targeted[
            [
                "hypothesis",
                "score_variant",
                "n_group_1",
                "n_group_2",
                "median_group_1",
                "median_group_2",
                "cliffs_delta_group_1_vs_group_2",
                "mann_whitney_p",
                "targeted_q_bh_2",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Subtype score medians: primary signed score")
    print("=" * 80)
    print(
        summary_signed[
            [
                "module_label",
                "primary_immune_subtype",
                "n",
                "median",
                "q25",
                "q75",
            ]
        ].to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("Patkar subtypes and frozen modules use overlapping DOG2 bulk-expression samples.")
    print("This is biological convergence across methods, not independent external validation.")
    print("Subtype labels were not used to select genes, alter weights, orient scores, or revise frozen tiers.")
    print("M34 and M11 targeted contrasts were specified from their frozen biological labels before inspecting subtype-score results.")
    print("M24 and M40 subtype comparisons are exploratory.")

    print("")
    print("Saved:")
    for path in output_paths + figure_paths + [OUTPUT_MANIFEST]:
        print(path)
    print("Done.")


if __name__ == "__main__":
    main()
