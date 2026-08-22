from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05d4a-read-only-paired-ablation-audit-v1-no-cli"
AUDIT_VERSION = "sable-mo-05d4a-paired-confirmatory-ablation-audit-v1"
BOOTSTRAP_REPLICATES = 5_000
BOOTSTRAP_GLOBAL_SEED = 54_041
BOOTSTRAP_ALPHA = 0.05
ZERO_TOLERANCE = 1e-12

EXPECTED_CONFIRMATORY_DATASETS = 360
EXPECTED_SCENARIOS = 36
EXPECTED_REPLICATES_PER_SCENARIO = 10
EXPECTED_VARIANTS = (
    "V0_BALANCED_AE",
    "V1_SHARED_PRIVATE_AE",
    "V2_CONTRASTIVE_SPAE",
    "V3_DROPOUT_SPAE",
    "V4_SABLE_MO",
    "V5_SABLE_VAE",
)
EXPECTED_DATASET_VARIANT_ROWS = EXPECTED_CONFIRMATORY_DATASETS * len(EXPECTED_VARIANTS)
EXPECTED_PROFILE = "ARCH_COMPACT__REG_WEAK"

CONTRASTS = (
    {
        "contrast_id": "C1_SHARED_PRIVATE",
        "higher_variant": "V1_SHARED_PRIVATE_AE",
        "lower_variant": "V0_BALANCED_AE",
        "role": "primary_confirmatory_ablation",
        "component": "shared/private decomposition",
        "headline_metrics": (
            "shared_subspace_recovery",
            "private_subspace_recovery",
            "simulation_utility",
        ),
        "interpretation": (
            "Descriptive paired confirmatory contrast for the shared/private decomposition. "
            "No p-value and no method selection are performed in 05d4a."
        ),
    },
    {
        "contrast_id": "C2_CONTRASTIVE",
        "higher_variant": "V2_CONTRASTIVE_SPAE",
        "lower_variant": "V1_SHARED_PRIVATE_AE",
        "role": "primary_confirmatory_ablation",
        "component": "paired contrastive alignment",
        "headline_metrics": (
            "random_adjusted_cross_view_mrr",
            "same_patient_similarity_margin",
            "simulation_utility",
        ),
        "interpretation": (
            "Descriptive paired confirmatory contrast for cross-view alignment. "
            "The hypothesis-matched endpoint is cross-view retrieval/alignment rather than monotone utility alone."
        ),
    },
    {
        "contrast_id": "C3_DROPOUT_MISSING_VIEW",
        "higher_variant": "V3_DROPOUT_SPAE",
        "lower_variant": "V2_CONTRASTIVE_SPAE",
        "role": "primary_confirmatory_ablation",
        "component": "modality dropout + missing-view reconstruction",
        "headline_metrics": (
            "missing_view_reconstruction_score",
            "heldout_reconstruction_score",
            "simulation_utility",
        ),
        "interpretation": (
            "Descriptive paired confirmatory contrast for missing-view robustness. "
            "The hypothesis-matched endpoint is missing-view reconstruction rather than monotone utility alone."
        ),
    },
    {
        "contrast_id": "S1_VAE_VS_V3",
        "higher_variant": "V5_SABLE_VAE",
        "lower_variant": "V3_DROPOUT_SPAE",
        "role": "secondary_exploratory",
        "component": "variational latent extension versus V3 pre-gate training model",
        "headline_metrics": (
            "simulation_utility",
            "random_adjusted_cross_view_mrr",
            "missing_view_reconstruction_score",
            "heldout_reconstruction_score",
        ),
        "interpretation": (
            "Secondary exploratory paired comparison. V5 was prospectively ineligible for final method freeze; "
            "05d4a cannot promote V5 to the primary method."
        ),
    },
    {
        "contrast_id": "S2_VAE_VS_V4_PRE_GATE",
        "higher_variant": "V5_SABLE_VAE",
        "lower_variant": "V4_SABLE_MO",
        "role": "secondary_exploratory",
        "component": "variational latent extension versus V4 pre-gate training metrics",
        "headline_metrics": (
            "simulation_utility",
            "random_adjusted_cross_view_mrr",
            "missing_view_reconstruction_score",
            "heldout_reconstruction_score",
        ),
        "interpretation": (
            "Secondary exploratory paired comparison. V4 metrics in 05d4 are PRE-GATE training metrics; "
            "the resampling-based V4 stability component is evaluated in 05d5, not by V4-minus-V3 in 05d4a."
        ),
    },
)

AUDIT_METRICS = (
    "simulation_utility",
    "heldout_reconstruction_score",
    "heldout_normalized_reconstruction_error",
    "random_adjusted_cross_view_mrr",
    "cross_view_mrr",
    "cross_view_top1_accuracy",
    "same_patient_similarity_margin",
    "missing_view_reconstruction_score",
    "missing_view_r2_RNA",
    "missing_view_r2_METHYLATION",
    "missing_view_r2_CNV",
    "modality_balance_score",
    "modality_contribution_entropy",
    "shared_subspace_recovery",
    "private_subspace_recovery",
    "technical_subspace_recovery",
    "heldout_latent_effective_rank",
    "heldout_median_shared_variance",
)

PAIR_KEYS = ("scenario_id", "replicate_index", "simulation_seed")
PAIR_METADATA = (
    "scenario_origin",
    "scenario_family",
    "calibration_role",
    "n_samples",
    "n_true_biological_shared_factors",
    "n_true_private_factors_per_view",
    "n_true_technical_shared_factors",
    "factor_level_identifiable",
    "subspace_level_identifiable",
    "primary_truth_target",
    "profile_id",
    "architecture_profile",
    "regularization_profile",
)


SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

SOURCE_CONTRACT_DIR = PAPER5_ROOT / "method_contract" / "05d4"
SOURCE_RESULTS_DIR = (
    PAPER5_ROOT
    / "results"
    / "simulation_benchmark"
    / "05d_neural"
    / "confirmatory_ablation"
)
SOURCE_LOCK = SOURCE_CONTRACT_DIR / "neural_confirmatory_ablation_results_lock.json"
SOURCE_MANIFEST = SOURCE_CONTRACT_DIR / "neural_confirmatory_ablation_results_manifest.json"
SOURCE_DATASET_VARIANT = SOURCE_RESULTS_DIR / "neural_confirmatory_dataset_variant_metrics.csv.gz"

OUTPUT_CONTRACT_DIR = PAPER5_ROOT / "method_contract" / "05d4a"
OUTPUT_RESULTS_DIR = (
    PAPER5_ROOT
    / "results"
    / "simulation_benchmark"
    / "05d_neural"
    / "confirmatory_ablation_audit"
)
OUTPUT_DATASET_DELTAS = OUTPUT_RESULTS_DIR / "05d4a_paired_dataset_deltas.csv.gz"
OUTPUT_OVERALL = OUTPUT_RESULTS_DIR / "05d4a_paired_overall_summary.csv"
OUTPUT_SCENARIO = OUTPUT_RESULTS_DIR / "05d4a_paired_scenario_summary.csv"
OUTPUT_SCENARIO_FAMILY = OUTPUT_RESULTS_DIR / "05d4a_paired_scenario_family_summary.csv"
OUTPUT_HEADLINE = OUTPUT_RESULTS_DIR / "05d4a_headline_contrast_summary.csv"
OUTPUT_CONTRAST_REGISTRY = OUTPUT_CONTRACT_DIR / "05d4a_contrast_registry.csv"
OUTPUT_EXECUTION_RECORD = OUTPUT_CONTRACT_DIR / "05d4a_execution_record.json"
OUTPUT_README = OUTPUT_CONTRACT_DIR / "05d4a_README.txt"
OUTPUT_LOCK = OUTPUT_CONTRACT_DIR / "05d4a_results_lock.json"


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = clean_text(value).lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_json(path: Path) -> dict[str, Any]:
    require_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv_atomic(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    table.to_csv(temporary, index=False, float_format="%.12g", lineterminator="\n")
    os.replace(temporary, path)


def write_csv_gzip_atomic(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    table.to_csv(temporary, index=False, float_format="%.12g", compression="gzip")
    os.replace(temporary, path)


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def deterministic_seed(*tokens: Any) -> int:
    text = "|".join(clean_text(value) for value in tokens)
    digest = hashlib.sha256(f"{BOOTSTRAP_GLOBAL_SEED}|{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32)


# -----------------------------------------------------------------------------
# Frozen 05d4 verification
# -----------------------------------------------------------------------------


def expected_record_for_path(lock: dict[str, Any], path: Path) -> dict[str, Any]:
    relative = relative_path(path)
    record = lock.get("generated_files", {}).get(relative)
    if not isinstance(record, dict):
        raise RuntimeError(f"05d4 lock does not contain a generated-file record for {relative}")
    return record


def verify_record(path: Path, record: dict[str, Any], label: str) -> None:
    require_file(path)
    expected_size = record.get("size_bytes")
    if expected_size is not None and int(path.stat().st_size) != int(expected_size):
        raise RuntimeError(f"{label} size mismatch: {path}")
    expected_hash = clean_text(record.get("sha256"))
    observed_hash = sha256_file(path)
    if not expected_hash or observed_hash != expected_hash:
        raise RuntimeError(f"{label} SHA-256 mismatch: {path}")


def verify_05d4() -> dict[str, Any]:
    require_file(SOURCE_LOCK)
    require_file(SOURCE_MANIFEST)
    require_file(SOURCE_DATASET_VARIANT)

    lock = load_json(SOURCE_LOCK)
    manifest = load_json(SOURCE_MANIFEST)

    if clean_text(lock.get("status")) != "FROZEN_NEURAL_CONFIRMATORY_ABLATION_RESULTS_BEFORE_05D5":
        raise RuntimeError(
            "05d4a requires a fully passing frozen 05d4 result lock. "
            f"Observed status: {clean_text(lock.get('status'))!r}"
        )

    expected_manifest_hash = clean_text(lock.get("result_manifest", {}).get("sha256"))
    if not expected_manifest_hash or sha256_file(SOURCE_MANIFEST) != expected_manifest_hash:
        raise RuntimeError("05d4 result manifest hash differs from the frozen lock.")

    verify_record(
        SOURCE_DATASET_VARIANT,
        expected_record_for_path(lock, SOURCE_DATASET_VARIANT),
        "05d4 dataset-variant metrics",
    )

    state = manifest.get("scientific_state", {})
    forbidden_true = {
        "outcomes_loaded": state.get("outcomes_loaded", False),
        "clinical_labels_loaded": state.get("clinical_labels_loaded", False),
        "TARGET_model_fitting_performed": state.get("TARGET_model_fitting_performed", False),
        "confirmatory_V4_stability_performed": state.get("confirmatory_V4_stability_performed", False),
        "final_gate_calibration_data_used": state.get("final_gate_calibration_data_used", False),
        "representation_method_selected": state.get("representation_method_selected", False),
    }
    violations = [name for name, value in forbidden_true.items() if bool(value)]
    if violations:
        raise RuntimeError("05d4 scientific-state violation: " + ", ".join(violations))

    scope = manifest.get("scope", {})
    expected_scope = {
        "confirmatory_datasets": EXPECTED_CONFIRMATORY_DATASETS,
        "dataset_variant_rows": EXPECTED_DATASET_VARIANT_ROWS,
        "passed_fold_seed_evaluations": 21_600,
        "failed_fold_seed_evaluations": 0,
    }
    for key, expected in expected_scope.items():
        observed = int(scope.get(key, -1))
        if observed != expected:
            raise RuntimeError(f"05d4 scope mismatch for {key}: {observed} vs {expected}")

    profile_id = clean_text(lock.get("profile_id"))
    if profile_id != EXPECTED_PROFILE:
        raise RuntimeError(f"Unexpected 05d4 profile: {profile_id!r}")

    return {
        "lock": lock,
        "manifest": manifest,
        "lock_sha256": sha256_file(SOURCE_LOCK),
        "manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "dataset_variant_sha256": sha256_file(SOURCE_DATASET_VARIANT),
    }


# -----------------------------------------------------------------------------
# Input-table validation and paired deltas
# -----------------------------------------------------------------------------


def load_dataset_table() -> pd.DataFrame:
    table = pd.read_csv(SOURCE_DATASET_VARIANT, low_memory=False)

    if table.shape[0] != EXPECTED_DATASET_VARIANT_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_DATASET_VARIANT_ROWS:,} dataset-variant rows; found {table.shape[0]:,}."
        )

    required = set(PAIR_KEYS) | set(PAIR_METADATA) | {
        "variant_id",
        "all_fold_seed_evaluations_passed",
        "n_fold_seed_evaluations",
        "secondary_exploratory_variant",
        "stability_gate_applied_in_05d4",
    }
    required |= {f"{metric}_mean" for metric in AUDIT_METRICS}
    missing = sorted(required - set(table.columns))
    if missing:
        raise RuntimeError("05d4 dataset table is missing required columns:\n- " + "\n- ".join(missing))

    variants = tuple(sorted(table["variant_id"].astype(str).unique().tolist()))
    if variants != tuple(sorted(EXPECTED_VARIANTS)):
        raise RuntimeError(f"Unexpected V0-V5 variant set: {variants}")

    profile_values = sorted(table["profile_id"].astype(str).unique().tolist())
    if profile_values != [EXPECTED_PROFILE]:
        raise RuntimeError(f"Unexpected profile values in dataset table: {profile_values}")

    if not table["all_fold_seed_evaluations_passed"].map(parse_bool).all():
        raise RuntimeError("At least one 05d4 dataset-variant row contains a failed fold-seed evaluation.")

    if not pd.to_numeric(table["n_fold_seed_evaluations"], errors="raise").astype(int).eq(10).all():
        raise RuntimeError("Every dataset-variant row must contain exactly 10 fold-seed evaluations.")

    key_counts = table.groupby(list(PAIR_KEYS), sort=True, dropna=False)["variant_id"].nunique()
    if key_counts.shape[0] != EXPECTED_CONFIRMATORY_DATASETS:
        raise RuntimeError(f"Expected {EXPECTED_CONFIRMATORY_DATASETS} paired datasets; found {key_counts.shape[0]}.")
    if not key_counts.eq(len(EXPECTED_VARIANTS)).all():
        raise RuntimeError("At least one confirmatory dataset does not contain all six V0-V5 variants.")

    scenario_counts = table[list(PAIR_KEYS)].drop_duplicates().groupby("scenario_id").size()
    if scenario_counts.shape[0] != EXPECTED_SCENARIOS or not scenario_counts.eq(EXPECTED_REPLICATES_PER_SCENARIO).all():
        raise RuntimeError("Scenario pairing does not match 36 scenarios x 10 confirmatory replicates.")

    v4 = table[table["variant_id"].eq("V4_SABLE_MO")]
    if v4["stability_gate_applied_in_05d4"].map(parse_bool).any():
        raise RuntimeError("05d4 unexpectedly applied the V4 stability gate; 05d4a assumptions are invalid.")

    v5 = table[table["variant_id"].eq("V5_SABLE_VAE")]
    if not v5["secondary_exploratory_variant"].map(parse_bool).all():
        raise RuntimeError("V5 is not consistently marked secondary exploratory in frozen 05d4 results.")

    return table


def assert_metadata_pairing(left: pd.DataFrame, right: pd.DataFrame, contrast_id: str) -> None:
    left_meta = left.set_index(list(PAIR_KEYS))[list(PAIR_METADATA)].sort_index()
    right_meta = right.set_index(list(PAIR_KEYS))[list(PAIR_METADATA)].sort_index()
    if not left_meta.index.equals(right_meta.index):
        raise RuntimeError(f"{contrast_id}: higher/lower dataset keys do not match exactly.")

    for column in PAIR_METADATA:
        a = left_meta[column].astype(str)
        b = right_meta[column].astype(str)
        if not a.equals(b):
            mismatches = (a != b)
            first_key = a.index[np.flatnonzero(mismatches.to_numpy())[0]]
            raise RuntimeError(f"{contrast_id}: metadata mismatch for {column} at {first_key}.")


def build_paired_dataset_deltas(table: pd.DataFrame) -> pd.DataFrame:
    output_parts: list[pd.DataFrame] = []

    for spec in CONTRASTS:
        contrast_id = str(spec["contrast_id"])
        higher = str(spec["higher_variant"])
        lower = str(spec["lower_variant"])

        left = table[table["variant_id"].eq(higher)].copy()
        right = table[table["variant_id"].eq(lower)].copy()
        if left.shape[0] != EXPECTED_CONFIRMATORY_DATASETS or right.shape[0] != EXPECTED_CONFIRMATORY_DATASETS:
            raise RuntimeError(f"{contrast_id}: expected 360 rows per variant.")

        assert_metadata_pairing(left, right, contrast_id)

        left = left.sort_values(list(PAIR_KEYS)).reset_index(drop=True)
        right = right.sort_values(list(PAIR_KEYS)).reset_index(drop=True)

        paired = left[list(PAIR_KEYS) + list(PAIR_METADATA)].copy()
        paired["contrast_id"] = contrast_id
        paired["contrast_role"] = spec["role"]
        paired["added_component_or_comparison"] = spec["component"]
        paired["higher_variant"] = higher
        paired["lower_variant"] = lower
        paired["interpretation_guardrail"] = spec["interpretation"]
        paired["V4_stability_gate_applied_in_05d4"] = False

        for metric in AUDIT_METRICS:
            source_column = f"{metric}_mean"
            a = pd.to_numeric(left[source_column], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(right[source_column], errors="coerce").to_numpy(dtype=float)
            paired[f"higher_{metric}"] = a
            paired[f"lower_{metric}"] = b
            paired[f"delta_{metric}"] = a - b

        output_parts.append(paired)

    output = pd.concat(output_parts, ignore_index=True)
    expected_rows = EXPECTED_CONFIRMATORY_DATASETS * len(CONTRASTS)
    if output.shape[0] != expected_rows:
        raise RuntimeError(f"Expected {expected_rows:,} paired delta rows; found {output.shape[0]:,}.")
    return output


# -----------------------------------------------------------------------------
# Deterministic paired bootstrap summaries
# -----------------------------------------------------------------------------


def bootstrap_mean_median_ci(values: np.ndarray, seed: int) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return (float("nan"),) * 4

    rng = np.random.default_rng(seed)
    n = int(values.size)
    mean_stats = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    median_stats = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    chunk_size = 500
    offset = 0
    while offset < BOOTSTRAP_REPLICATES:
        count = min(chunk_size, BOOTSTRAP_REPLICATES - offset)
        indices = rng.integers(0, n, size=(count, n), endpoint=False)
        sampled = values[indices]
        mean_stats[offset : offset + count] = sampled.mean(axis=1)
        median_stats[offset : offset + count] = np.median(sampled, axis=1)
        offset += count

    lower_q = BOOTSTRAP_ALPHA / 2.0
    upper_q = 1.0 - lower_q
    return (
        float(np.quantile(mean_stats, lower_q)),
        float(np.quantile(mean_stats, upper_q)),
        float(np.quantile(median_stats, lower_q)),
        float(np.quantile(median_stats, upper_q)),
    )


def summarize_values(values: Iterable[float], seed_tokens: tuple[Any, ...]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    n = int(array.size)
    if n == 0:
        return {
            "n_paired": 0,
            "delta_mean": None,
            "delta_median": None,
            "delta_sd": None,
            "delta_se": None,
            "delta_q25": None,
            "delta_q75": None,
            "bootstrap_mean_ci95_low": None,
            "bootstrap_mean_ci95_high": None,
            "bootstrap_median_ci95_low": None,
            "bootstrap_median_ci95_high": None,
            "fraction_positive": None,
            "fraction_negative": None,
            "fraction_zero_within_tolerance": None,
        }

    mean_value = float(np.mean(array))
    median_value = float(np.median(array))
    sd_value = float(np.std(array, ddof=1)) if n > 1 else None
    se_value = float(sd_value / math.sqrt(n)) if sd_value is not None else None

    ci_mean_low, ci_mean_high, ci_median_low, ci_median_high = bootstrap_mean_median_ci(
        array,
        deterministic_seed(*seed_tokens),
    )

    return {
        "n_paired": n,
        "delta_mean": mean_value,
        "delta_median": median_value,
        "delta_sd": sd_value,
        "delta_se": se_value,
        "delta_q25": float(np.quantile(array, 0.25)),
        "delta_q75": float(np.quantile(array, 0.75)),
        "bootstrap_mean_ci95_low": ci_mean_low,
        "bootstrap_mean_ci95_high": ci_mean_high,
        "bootstrap_median_ci95_low": ci_median_low,
        "bootstrap_median_ci95_high": ci_median_high,
        "fraction_positive": float(np.mean(array > ZERO_TOLERANCE)),
        "fraction_negative": float(np.mean(array < -ZERO_TOLERANCE)),
        "fraction_zero_within_tolerance": float(np.mean(np.abs(array) <= ZERO_TOLERANCE)),
    }


def contrast_spec(contrast_id: str) -> dict[str, Any]:
    for spec in CONTRASTS:
        if spec["contrast_id"] == contrast_id:
            return spec
    raise KeyError(contrast_id)


def make_summary_rows(
    paired: pd.DataFrame,
    group_columns: list[str],
    summary_level: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if group_columns:
        grouped = paired.groupby(group_columns, sort=True, dropna=False)
    else:
        grouped = [((), paired)]

    for keys, part in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_metadata = dict(zip(group_columns, keys))
        contrast_id = clean_text(part.iloc[0]["contrast_id"])
        spec = contrast_spec(contrast_id)
        headline = set(spec["headline_metrics"])

        for metric in AUDIT_METRICS:
            delta_column = f"delta_{metric}"
            row: dict[str, Any] = {
                "summary_level": summary_level,
                **group_metadata,
                "contrast_id": contrast_id,
                "contrast_role": spec["role"],
                "higher_variant": spec["higher_variant"],
                "lower_variant": spec["lower_variant"],
                "added_component_or_comparison": spec["component"],
                "metric": metric,
                "hypothesis_matched_headline_metric": metric in headline,
                "interpretation_guardrail": spec["interpretation"],
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_type": "paired nonparametric percentile bootstrap over confirmatory datasets",
                "bootstrap_ci": "two-sided 95% percentile interval",
            }
            row.update(
                summarize_values(
                    pd.to_numeric(part[delta_column], errors="coerce").to_numpy(dtype=float),
                    (
                        summary_level,
                        contrast_id,
                        metric,
                        *[group_metadata.get(column, "") for column in group_columns],
                    ),
                )
            )
            rows.append(row)

    return pd.DataFrame(rows)


def build_summaries(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_parts: list[pd.DataFrame] = []
    scenario_parts: list[pd.DataFrame] = []
    family_parts: list[pd.DataFrame] = []

    for spec in CONTRASTS:
        contrast_id = str(spec["contrast_id"])
        part = paired[paired["contrast_id"].eq(contrast_id)].copy()

        overall_parts.append(make_summary_rows(part, [], "overall_360_paired_datasets"))
        scenario_parts.append(
            make_summary_rows(
                part,
                ["scenario_id", "scenario_family", "n_samples"],
                "scenario_10_paired_datasets",
            )
        )
        family_parts.append(
            make_summary_rows(
                part,
                ["scenario_family"],
                "frozen_scenario_family_paired_datasets",
            )
        )

    overall = pd.concat(overall_parts, ignore_index=True)
    scenario = pd.concat(scenario_parts, ignore_index=True)
    family = pd.concat(family_parts, ignore_index=True)

    headline = overall[overall["hypothesis_matched_headline_metric"].map(parse_bool)].copy()
    headline = headline.sort_values(["contrast_id", "metric"]).reset_index(drop=True)

    return overall, scenario, family, headline


# -----------------------------------------------------------------------------
# Output verification / README
# -----------------------------------------------------------------------------


def build_contrast_registry() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in CONTRASTS:
        rows.append(
            {
                "contrast_id": spec["contrast_id"],
                "contrast_role": spec["role"],
                "higher_variant": spec["higher_variant"],
                "lower_variant": spec["lower_variant"],
                "added_component_or_comparison": spec["component"],
                "headline_metrics": "|".join(spec["headline_metrics"]),
                "interpretation_guardrail": spec["interpretation"],
                "method_selection_permitted": False,
                "V4_minus_V3_as_stability_gate_ablation_permitted": False,
            }
        )
    return pd.DataFrame(rows)


def build_readme(source: dict[str, Any], headline: pd.DataFrame) -> str:
    lines = [
        "Paper 5 - 05d4a read-only paired confirmatory ablation audit",
        "=" * 78,
        "",
        f"Script version: {SCRIPT_VERSION}",
        f"Audit version: {AUDIT_VERSION}",
        f"Created UTC: {utc_now_iso()}",
        "",
        "Purpose",
        "-------",
        "05d4a is a derived READ-ONLY audit of the already frozen 05d4 confirmatory ablation results.",
        "It performs no model fitting, no GPU computation, no TARGET access, no outcome access, and no method selection.",
        "",
        "Pairing",
        "-------",
        "All contrasts are computed within the same confirmatory simulation dataset, paired by scenario_id, replicate_index, and simulation_seed.",
        "The primary C1-C3 contrasts therefore use 360 paired dataset-level differences rather than differences between unpaired grand means.",
        "",
        "Uncertainty",
        "-----------",
        f"Two-sided 95% percentile bootstrap intervals use {BOOTSTRAP_REPLICATES:,} deterministic paired bootstrap resamples.",
        "Bootstrap resampling is performed over confirmatory datasets within the relevant reporting group.",
        "These intervals are descriptive confirmatory uncertainty summaries; 05d4a introduces no new hypothesis-testing gate.",
        "",
        "Interpretation guardrails",
        "------------------------",
        "1. C1 tests the descriptive contribution of shared/private decomposition.",
        "2. C2 is interpreted primarily using cross-view retrieval/alignment metrics.",
        "3. C3 is interpreted primarily using missing-view reconstruction/robustness metrics.",
        "4. Monotone improvement in one composite utility is NOT required for every component.",
        "5. V4 training metrics in 05d4 are PRE-GATE. The V4 resampling-based stability component is evaluated in 05d5.",
        "6. V4-minus-V3 must NOT be presented as a stability-gate ablation from 05d4.",
        "7. V5 remains secondary exploratory and cannot be promoted to the final primary method by 05d4a.",
        "8. Frozen scenario_family labels are reported as descriptive strata; no subgroup result changes the frozen method.",
        "",
        "Frozen source provenance",
        "------------------------",
        f"05d4 lock SHA-256: {source['lock_sha256']}",
        f"05d4 result manifest SHA-256: {source['manifest_sha256']}",
        f"05d4 dataset-variant table SHA-256: {source['dataset_variant_sha256']}",
        "",
        "Headline overall paired contrasts",
        "---------------------------------",
    ]

    display_columns = [
        "contrast_id",
        "metric",
        "n_paired",
        "delta_mean",
        "bootstrap_mean_ci95_low",
        "bootstrap_mean_ci95_high",
        "fraction_positive",
    ]
    for _, row in headline[display_columns].iterrows():
        lines.append(
            f"{row['contrast_id']} | {row['metric']} | n={int(row['n_paired'])} | "
            f"mean delta={float(row['delta_mean']):.6g} | "
            f"95% bootstrap CI [{float(row['bootstrap_mean_ci95_low']):.6g}, {float(row['bootstrap_mean_ci95_high']):.6g}] | "
            f"fraction positive={float(row['fraction_positive']):.3f}"
        )

    lines.extend(
        [
            "",
            "Next methodological use",
            "-----------------------",
            "Use these derived paired summaries to present C1-C3 by hypothesis-matched endpoints and to describe V5 without changing its frozen exploratory status.",
            "05d4a itself does not select or freeze the representation method.",
            "",
        ]
    )
    return "\n".join(lines)


def verify_existing_lock() -> None:
    lock = load_json(OUTPUT_LOCK)
    if clean_text(lock.get("status")) != "DERIVED_READ_ONLY_AUDIT_OF_FROZEN_05D4_RESULTS":
        raise RuntimeError("Unexpected existing 05d4a lock status.")

    source_hash = clean_text(lock.get("source_05d4", {}).get("dataset_variant_sha256"))
    require_file(SOURCE_DATASET_VARIANT)
    if not source_hash or sha256_file(SOURCE_DATASET_VARIANT) != source_hash:
        raise RuntimeError("Existing 05d4a audit refers to a different 05d4 dataset-variant source file.")

    failures: list[str] = []
    for relative_name, record in lock.get("generated_files", {}).items():
        path = resolve_project_path(relative_name)
        if not path.is_file():
            failures.append(f"missing: {relative_name}")
            continue
        if int(path.stat().st_size) != int(record.get("size_bytes", -1)):
            failures.append(f"size mismatch: {relative_name}")
            continue
        if sha256_file(path) != clean_text(record.get("sha256")):
            failures.append(f"hash mismatch: {relative_name}")
    if failures:
        raise RuntimeError("Existing 05d4a verification failed:\n- " + "\n- ".join(failures))

    print("=" * 88)
    print("Existing Paper 5 05d4a paired ablation audit: PASS")
    print("=" * 88)
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("No recomputation performed.")
    print("=" * 88)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    print("=" * 88)
    print("Paper 5 - read-only paired audit of neural confirmatory ablation")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Source 05d4 results: {SOURCE_RESULTS_DIR}")
    print(f"Output directory: {OUTPUT_RESULTS_DIR}")
    print("GPU computation in 05d4a: NO")
    print("Model fitting in 05d4a: NO")
    print("TARGET/outcomes/final-gate data accessed: NO")
    print("Method selection in 05d4a: FORBIDDEN")
    print()

    if OUTPUT_LOCK.exists():
        verify_existing_lock()
        return

    source = verify_05d4()
    print("Frozen 05d4 result lock + manifest: PASS")
    print("Frozen 05d4 dataset-variant SHA-256: PASS")
    print("05d4 V4 stability gate applied: NO")
    print("V5 secondary exploratory status: FROZEN")
    print()

    table = load_dataset_table()
    print(f"Dataset-variant rows: {table.shape[0]:,} / {EXPECTED_DATASET_VARIANT_ROWS:,}: PASS")
    print(f"Paired confirmatory datasets: {EXPECTED_CONFIRMATORY_DATASETS}")
    print(f"Scenarios: {EXPECTED_SCENARIOS}")
    print(f"Contrasts: {len(CONTRASTS)} (3 primary ablation + 2 secondary V5 comparisons)")
    print()

    OUTPUT_CONTRACT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    registry = build_contrast_registry()
    write_csv_atomic(registry, OUTPUT_CONTRAST_REGISTRY)

    paired = build_paired_dataset_deltas(table)
    write_csv_gzip_atomic(paired, OUTPUT_DATASET_DELTAS)

    print("Computing deterministic paired bootstrap summaries...")
    overall, scenario, family, headline = build_summaries(paired)

    write_csv_atomic(overall, OUTPUT_OVERALL)
    write_csv_atomic(scenario, OUTPUT_SCENARIO)
    write_csv_atomic(family, OUTPUT_SCENARIO_FAMILY)
    write_csv_atomic(headline, OUTPUT_HEADLINE)

    readme_text = build_readme(source, headline)
    OUTPUT_README.write_text(readme_text, encoding="utf-8")

    output_files = [
        OUTPUT_CONTRAST_REGISTRY,
        OUTPUT_DATASET_DELTAS,
        OUTPUT_OVERALL,
        OUTPUT_SCENARIO,
        OUTPUT_SCENARIO_FAMILY,
        OUTPUT_HEADLINE,
        OUTPUT_README,
    ]

    execution_record = {
        "audit_version": AUDIT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": "DERIVED_READ_ONLY_AUDIT_OF_FROZEN_05D4_RESULTS",
        "scientific_state": {
            "model_fitting_performed": False,
            "GPU_computation_performed": False,
            "TARGET_accessed": False,
            "outcomes_accessed": False,
            "final_gate_data_accessed": False,
            "method_selection_performed": False,
            "V4_stability_gate_interpreted_from_05d4": False,
            "V5_promoted_from_secondary_exploratory": False,
        },
        "source_05d4": {
            "lock_path": relative_path(SOURCE_LOCK),
            "lock_sha256": source["lock_sha256"],
            "result_manifest_path": relative_path(SOURCE_MANIFEST),
            "result_manifest_sha256": source["manifest_sha256"],
            "dataset_variant_path": relative_path(SOURCE_DATASET_VARIANT),
            "dataset_variant_sha256": source["dataset_variant_sha256"],
            "profile_id": EXPECTED_PROFILE,
        },
        "scope": {
            "paired_datasets_per_contrast": EXPECTED_CONFIRMATORY_DATASETS,
            "scenarios": EXPECTED_SCENARIOS,
            "confirmatory_replicates_per_scenario": EXPECTED_REPLICATES_PER_SCENARIO,
            "contrasts": len(CONTRASTS),
            "primary_ablation_contrasts": 3,
            "secondary_exploratory_contrasts": 2,
            "audit_metrics": list(AUDIT_METRICS),
            "dataset_delta_rows": int(paired.shape[0]),
            "overall_summary_rows": int(overall.shape[0]),
            "scenario_summary_rows": int(scenario.shape[0]),
            "scenario_family_summary_rows": int(family.shape[0]),
            "headline_summary_rows": int(headline.shape[0]),
        },
        "uncertainty": {
            "bootstrap_type": "paired nonparametric percentile bootstrap over confirmatory datasets",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "global_seed": BOOTSTRAP_GLOBAL_SEED,
            "confidence_level": 0.95,
            "two_sided": True,
            "selection_or_testing_role": "descriptive confirmatory uncertainty only",
        },
        "guardrails": [
            "05d4a is read-only with respect to frozen 05d4 results.",
            "All contrasts are paired within the same simulation dataset.",
            "C1-C3 are interpreted using hypothesis-matched endpoints, not a required monotone composite utility curve.",
            "V4 metrics from 05d4 are pre-gate; 05d5 evaluates the resampling-based stability component.",
            "V4-minus-V3 is not treated as a stability-gate ablation in 05d4a.",
            "V5 remains secondary exploratory and cannot be selected by 05d4a.",
            "Scenario and frozen scenario-family summaries are descriptive and do not change the method.",
            "No p-value, threshold retuning, architecture retuning, or representation-method selection is performed.",
        ],
        "outputs": {relative_path(path): file_record(path) for path in output_files},
        "next_stage_relationship": (
            "05d4a may run while 05d5 is computing because it reads only the already frozen 05d4 outputs; "
            "it does not read, modify, or depend on 05d5 results."
        ),
    }
    write_json_atomic(execution_record, OUTPUT_EXECUTION_RECORD)

    generated = [*output_files, OUTPUT_EXECUTION_RECORD]
    lock = {
        "audit_version": AUDIT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": "DERIVED_READ_ONLY_AUDIT_OF_FROZEN_05D4_RESULTS",
        "source_05d4": execution_record["source_05d4"],
        "execution_record": file_record(OUTPUT_EXECUTION_RECORD),
        "generated_files": {relative_path(path): file_record(path) for path in generated},
        "rewrite_policy": (
            "05d4a does not rewrite frozen 05d4 results. Existing 05d4a outputs are immutable under this script; "
            "a scientifically justified change requires a new versioned audit stage."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    print()
    print("-" * 88)
    print("05d4a headline paired audit summary")
    print("-" * 88)
    display_columns = [
        "contrast_id",
        "metric",
        "delta_mean",
        "bootstrap_mean_ci95_low",
        "bootstrap_mean_ci95_high",
        "fraction_positive",
    ]
    with pd.option_context("display.max_rows", 50, "display.width", 180):
        print(headline[display_columns].to_string(index=False))

    print()
    print("=" * 88)
    print("Paper 5 05d4a paired confirmatory ablation audit: PASS")
    print("=" * 88)
    print(f"Paired dataset deltas: {relative_path(OUTPUT_DATASET_DELTAS)}")
    print(f"Overall summary: {relative_path(OUTPUT_OVERALL)}")
    print(f"Scenario summary: {relative_path(OUTPUT_SCENARIO)}")
    print(f"Frozen scenario-family summary: {relative_path(OUTPUT_SCENARIO_FAMILY)}")
    print(f"Headline summary: {relative_path(OUTPUT_HEADLINE)}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("05d5 may continue running independently: YES")
    print("=" * 88)


if __name__ == "__main__":
    main()
