from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05b-generate-small-n-multiomics-simulations-v2-no-cli"
GENERATOR_VERSION = "sable-mo-simulation-generator-v2"
CONTRACT_VERSION = "paper5-confirmatory-architecture-simulation-materialization-v2"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# ---------------------------------------------------------------------
# Frozen upstream contracts
# ---------------------------------------------------------------------

CONTRACT_05A_DIR = PAPER5_ROOT / "method_contract" / "05a"
CONTRACT_05A = CONTRACT_05A_DIR / "ai_method_contract.json"
LOCK_05A = CONTRACT_05A_DIR / "ai_method_contract_lock.json"

CONTRACT_05A1_DIR = PAPER5_ROOT / "method_contract" / "05a1"
CONTRACT_05A1 = (
    CONTRACT_05A1_DIR
    / "simulation_and_technical_confound_addendum.json"
)
LOCK_05A1 = (
    CONTRACT_05A1_DIR
    / "simulation_and_technical_confound_addendum_lock.json"
)
COMBINED_SCENARIOS_FILE = (
    CONTRACT_05A1_DIR
    / "combined_architecture_screening_scenarios.csv"
)
COMBINED_REPLICATES_FILE = (
    CONTRACT_05A1_DIR
    / "combined_architecture_screening_replicates.csv"
)
TRUTH_TAXONOMY_FILE = (
    CONTRACT_05A1_DIR
    / "simulation_truth_taxonomy.csv"
)
PILOT_DISCLOSURE_FILE = (
    CONTRACT_05A1_DIR
    / "pre_addendum_05b_pilot_materialization_disclosure.json"
)
FINAL_GATE_SCENARIOS_FILE = (
    CONTRACT_05A1_DIR
    / "final_gate_calibration_scenarios.csv"
)
FINAL_GATE_REPLICATES_FILE = (
    CONTRACT_05A1_DIR
    / "final_gate_calibration_replicates.csv"
)

TECHNICAL_DIR = (
    PAPER5_ROOT
    / "data_manifest"
    / "TARGET_OS_technical_metadata"
)
TECHNICAL_MANIFEST_FILE = (
    TECHNICAL_DIR
    / "TARGET_OS_technical_metadata_preflight_manifest.json"
)
TECHNICAL_LOCK_FILE = (
    TECHNICAL_DIR
    / "TARGET_OS_technical_metadata_preflight_lock.json"
)
TECHNICAL_SOURCE_INVENTORY_FILE = (
    TECHNICAL_DIR
    / "TARGET_OS_technical_metadata_source_inventory.csv"
)

# ---------------------------------------------------------------------
# Outputs: new root; the disclosed v1 pilot is never modified
# ---------------------------------------------------------------------

OUTPUT_CONTRACT_DIR = PAPER5_ROOT / "method_contract" / "05b_v2"
OUTPUT_DATA_ROOT = PAPER5_ROOT / "simulations" / "sable_mo_v2"
ARCHITECTURE_DATA_ROOT = OUTPUT_DATA_ROOT / "architecture_screening"
FINAL_GATE_DATA_ROOT = OUTPUT_DATA_ROOT / "final_gate_calibration"

OUTPUT_GENERATOR_SPEC = (
    OUTPUT_CONTRACT_DIR
    / "simulation_generator_v2_specification.json"
)
OUTPUT_DATA_MANIFEST = (
    OUTPUT_CONTRACT_DIR
    / "architecture_screening_data_manifest.csv"
)
OUTPUT_SCENARIO_SUMMARY = (
    OUTPUT_CONTRACT_DIR
    / "architecture_screening_materialization_summary.csv"
)
OUTPUT_FIXTURE_HASHES = (
    OUTPUT_CONTRACT_DIR
    / "simulation_reference_fixture_hashes.json"
)
OUTPUT_ENVIRONMENT = (
    OUTPUT_CONTRACT_DIR
    / "simulation_environment_snapshot.json"
)
OUTPUT_README = (
    OUTPUT_CONTRACT_DIR
    / "simulation_generation_v2_README.txt"
)
OUTPUT_LOCK = (
    OUTPUT_CONTRACT_DIR
    / "simulation_generation_v2_lock.json"
)
OUTPUT_IN_PROGRESS = (
    OUTPUT_CONTRACT_DIR
    / ".simulation_generation_v2_in_progress.json"
)

# ---------------------------------------------------------------------
# Constants and exact truth conventions
# ---------------------------------------------------------------------

VIEWS = ("RNA", "METHYLATION", "CNV")
VIEW_PREFIX = {
    "RNA": "rna",
    "METHYLATION": "methylation",
    "CNV": "cnv",
}
VIEW_INDEX = {
    "RNA": 0,
    "METHYLATION": 1,
    "CNV": 2,
}

DEFAULT_DTYPE = np.float32
DEFAULT_FEATURE_OFFSET_SD = 0.35
DEFAULT_TECHNICAL_LOADING_FRACTION = 0.10

MISSING_REASON_OBSERVED = 0
MISSING_REASON_BASE_MCAR = 1
MISSING_REASON_FEATURE_STRUCTURED = 2
MISSING_REASON_SAMPLE_QUALITY = 3
MISSING_REASON_PARTIAL_VIEW = 4
MISSING_REASON_BATCH_DEPENDENT = 5

MISSING_REASON_REGISTRY = {
    MISSING_REASON_OBSERVED: "observed",
    MISSING_REASON_BASE_MCAR: "base_mcar",
    MISSING_REASON_FEATURE_STRUCTURED: "feature_structured",
    MISSING_REASON_SAMPLE_QUALITY: "sample_quality",
    MISSING_REASON_PARTIAL_VIEW: "partial_view",
    MISSING_REASON_BATCH_DEPENDENT: "batch_dependent",
}

REQUIRED_SCENARIO_COLUMNS = {
    "scenario_id",
    "description",
    "simulation_stage",
    "n_samples",
    "n_shared_factors",
    "n_private_factors_per_view",
    "n_features_rna",
    "n_features_methylation",
    "n_features_cnv",
    "shared_signal_strength",
    "private_signal_strength",
    "noise_sd_rna",
    "noise_sd_methylation",
    "noise_sd_cnv",
    "missing_fraction_rna",
    "missing_fraction_methylation",
    "missing_fraction_cnv",
    "pair_corruption_fraction",
    "uninformative_modality",
    "block_scale_rna",
    "block_scale_methylation",
    "block_scale_cnv",
    "generator_required_version",
    "planned_replicates",
}

REQUIRED_REPLICATE_COLUMNS = {
    "registry_label",
    "simulation_stage",
    "scenario_id",
    "scenario_order",
    "replicate_index",
    "simulation_seed",
    "stability_resamples_per_dataset",
    "materialization_policy",
}

MODEL_RESULT_PATHS = (
    PAPER5_ROOT / "method_contract" / "05c",
    PAPER5_ROOT / "method_contract" / "05d",
    PAPER5_ROOT / "method_contract" / "05e",
    PAPER5_ROOT / "results" / "representation_benchmark",
    PAPER5_ROOT / "results" / "simulation_benchmark",
    PAPER5_ROOT / "results" / "sable_mo",
)


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    description: str
    scenario_origin: str
    scenario_family: str
    calibration_role: str
    simulation_stage: str

    n_samples: int
    n_shared_factors: int
    n_private_factors_per_view: int

    n_features_rna: int
    n_features_methylation: int
    n_features_cnv: int

    shared_signal_strength: float
    private_signal_strength: float

    noise_sd_rna: float
    noise_sd_methylation: float
    noise_sd_cnv: float

    missing_fraction_rna: float
    missing_fraction_methylation: float
    missing_fraction_cnv: float

    pair_corruption_fraction: float
    uninformative_modality: str

    block_scale_rna: float
    block_scale_methylation: float
    block_scale_cnv: float

    nonlinear_mode: str
    shared_factor_correlation: float
    shared_strength_pattern: str
    shared_support_pattern: str
    shared_support_by_factor: str

    private_cross_view_correlation: float
    correlated_private_mode: str

    shared_loading_fraction: float
    private_loading_fraction: float
    loading_regime: str

    missingness_mode: str
    missingness_target_view: str
    low_quality_sample_fraction: float
    low_quality_feature_fraction: float
    low_quality_missing_fraction: float
    partial_view_sample_fraction: float
    partial_view_missing_fraction: float

    batch_mode: str
    batch_cardinality: int
    batch_prevalence: float
    batch_effect_strength: float
    batch_view_support: str
    batch_signal_correlation: float
    batch_missingness_increment: float

    true_biological_shared_factors: int
    true_technical_shared_factors: int
    factor_level_identifiable: bool
    subspace_level_identifiable: bool
    primary_truth_target: str

    stability_resamples_per_dataset: int
    materialization_policy: str
    generator_required_version: str
    planned_replicates: int

    def n_features(self, view: str) -> int:
        return {
            "RNA": self.n_features_rna,
            "METHYLATION": self.n_features_methylation,
            "CNV": self.n_features_cnv,
        }[view]

    def noise_sd(self, view: str) -> float:
        return {
            "RNA": self.noise_sd_rna,
            "METHYLATION": self.noise_sd_methylation,
            "CNV": self.noise_sd_cnv,
        }[view]

    def missing_fraction(self, view: str) -> float:
        return {
            "RNA": self.missing_fraction_rna,
            "METHYLATION": self.missing_fraction_methylation,
            "CNV": self.missing_fraction_cnv,
        }[view]

    def block_scale(self, view: str) -> float:
        return {
            "RNA": self.block_scale_rna,
            "METHYLATION": self.block_scale_methylation,
            "CNV": self.block_scale_cnv,
        }[view]

    def canonical_dict(self) -> dict[str, Any]:
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__
        }


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na"}:
        return ""
    return text


def normalized_token(value: Any, default: str = "NONE") -> str:
    text = clean_text(value).upper()
    return text if text else default


def parse_int(value: Any, default: int = 0) -> int:
    text = clean_text(value)
    if not text:
        return int(default)
    return int(float(text))


def parse_float(value: Any, default: float = 0.0) -> float:
    text = clean_text(value)
    if not text:
        return float(default)
    return float(text)


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = clean_text(value).lower()
    if not text:
        return bool(default)
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def relative_path(path: Path) -> str:
    try:
        return str(
            path.resolve().relative_to(
                PROJECT_ROOT.resolve()
            )
        )
    except ValueError:
        return str(path.resolve())


def resolve_project_path(value: Any) -> Path:
    path = Path(clean_text(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def package_version(name: str) -> dict[str, Any]:
    try:
        return {
            "installed": True,
            "version": importlib.metadata.version(name),
        }
    except importlib.metadata.PackageNotFoundError:
        return {
            "installed": False,
            "version": None,
        }


def stable_seed(master_seed: int, *parts: Any) -> int:
    digest = hashlib.sha256()
    digest.update(str(master_seed).encode("utf-8"))
    for part in parts:
        digest.update(b"|")
        digest.update(str(part).encode("utf-8"))
    return int(
        int.from_bytes(digest.digest()[:8], "big")
        % (2**32 - 1)
    )


def standardize_columns(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("Expected a two-dimensional matrix.")
    if x.shape[1] == 0:
        return x.astype(DEFAULT_DTYPE)

    mean = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, ddof=1, keepdims=True)
    sd[(~np.isfinite(sd)) | (sd <= 0)] = 1.0
    return ((x - mean) / sd).astype(DEFAULT_DTYPE)


def view_tokens(value: Any) -> set[str]:
    token = normalized_token(value)
    aliases = {
        "ALL3": {"RNA", "METHYLATION", "CNV"},
        "RNA_METH": {"RNA", "METHYLATION"},
        "RNA_METHYLATION": {"RNA", "METHYLATION"},
        "RNA_CNV": {"RNA", "CNV"},
        "METH_CNV": {"METHYLATION", "CNV"},
        "METHYLATION_CNV": {"METHYLATION", "CNV"},
        "RNA": {"RNA"},
        "METH": {"METHYLATION"},
        "METHYLATION": {"METHYLATION"},
        "CNV": {"CNV"},
        "NONE": set(),
        "FROM_BASE_COLUMNS": set(),
    }
    if token not in aliases:
        raise ValueError(
            f"Unsupported view-support token: {token}"
        )
    return aliases[token]


# ---------------------------------------------------------------------
# Frozen-lock verification
# ---------------------------------------------------------------------


def verify_generated_files(
    lock: dict[str, Any],
    label: str,
) -> None:
    failures: list[str] = []
    for relative_name, record in lock.get(
        "generated_files",
        {},
    ).items():
        path = resolve_project_path(relative_name)
        if not path.exists():
            failures.append(
                f"missing: {relative_name}"
            )
            continue
        expected_size = record.get("size_bytes")
        if expected_size is not None:
            observed_size = int(path.stat().st_size)
            if observed_size != int(expected_size):
                failures.append(
                    f"size mismatch: {relative_name}"
                )
                continue
        expected_hash = clean_text(
            record.get("sha256")
        )
        if expected_hash and sha256_file(path) != expected_hash:
            failures.append(
                f"hash mismatch: {relative_name}"
            )

    if failures:
        raise RuntimeError(
            f"{label} generated-file verification failed:\n"
            + "\n".join(
                f"- {item}"
                for item in failures[:100]
            )
        )


def verify_05a() -> dict[str, Any]:
    require_file(CONTRACT_05A)
    require_file(LOCK_05A)

    contract = json.loads(
        CONTRACT_05A.read_text(encoding="utf-8")
    )
    lock = json.loads(
        LOCK_05A.read_text(encoding="utf-8")
    )

    expected = clean_text(lock.get("contract_sha256"))
    observed = sha256_file(CONTRACT_05A)
    if observed != expected:
        raise RuntimeError(
            "The 05a AI method contract does not match its lock."
        )

    verify_generated_files(lock, "05a")

    state = contract.get("scientific_state", {})
    if bool(state.get("outcome_loaded", True)):
        raise RuntimeError(
            "05a does not confirm outcome_loaded=false."
        )
    if bool(state.get("model_fitting_performed", True)):
        raise RuntimeError(
            "05a indicates model fitting before 05b v2."
        )

    return {
        "contract": contract,
        "lock": lock,
        "contract_sha256": observed,
        "lock_sha256": sha256_file(LOCK_05A),
    }


def verify_05a1() -> dict[str, Any]:
    required = [
        CONTRACT_05A1,
        LOCK_05A1,
        COMBINED_SCENARIOS_FILE,
        COMBINED_REPLICATES_FILE,
        TRUTH_TAXONOMY_FILE,
        PILOT_DISCLOSURE_FILE,
        FINAL_GATE_SCENARIOS_FILE,
        FINAL_GATE_REPLICATES_FILE,
    ]
    for path in required:
        require_file(path)

    addendum = json.loads(
        CONTRACT_05A1.read_text(encoding="utf-8")
    )
    lock = json.loads(
        LOCK_05A1.read_text(encoding="utf-8")
    )

    expected = clean_text(lock.get("addendum_sha256"))
    observed = sha256_file(CONTRACT_05A1)
    if observed != expected:
        raise RuntimeError(
            "The 05a1 addendum does not match its frozen lock."
        )

    verify_generated_files(lock, "05a1")

    status = clean_text(addendum.get("status"))
    if not status.startswith("FROZEN_BEFORE_MODEL_FITTING"):
        raise RuntimeError(
            f"Unexpected 05a1 status: {status}"
        )

    state = addendum.get("scientific_state", {})
    forbidden_true = {
        "outcome_loaded": state.get("outcome_loaded", True),
        "clinical_labels_loaded": state.get(
            "clinical_labels_loaded",
            True,
        ),
        "biological_annotations_loaded": state.get(
            "biological_annotations_loaded",
            True,
        ),
        "simulation_model_fitting_performed": state.get(
            "simulation_model_fitting_performed",
            True,
        ),
        "target_model_fitting_performed": state.get(
            "target_model_fitting_performed",
            True,
        ),
    }
    violations = [
        key
        for key, value in forbidden_true.items()
        if bool(value)
    ]
    if violations:
        raise RuntimeError(
            "05a1 scientific-state violation: "
            + ", ".join(violations)
        )

    requirement = addendum.get(
        "generator_requirement",
        {},
    )
    if bool(
        requirement.get(
            "current_05b_v1_full_generation_allowed",
            True,
        )
    ):
        raise RuntimeError(
            "05a1 unexpectedly permits the old 05b v1 generator."
        )
    if clean_text(
        requirement.get("required_version")
    ) != GENERATOR_VERSION:
        raise RuntimeError(
            "05a1 does not require the current v2 generator."
        )

    pilot = addendum.get("pre_addendum_05b_pilot", {})
    if bool(
        pilot.get(
            "eligible_for_confirmatory_analysis",
            True,
        )
    ):
        raise RuntimeError(
            "The disclosed 05b v1 pilot is not excluded."
        )

    return {
        "addendum": addendum,
        "lock": lock,
        "addendum_sha256": observed,
        "lock_sha256": sha256_file(LOCK_05A1),
        "pilot_disclosure_sha256": sha256_file(
            PILOT_DISCLOSURE_FILE
        ),
    }


def verify_04d() -> dict[str, Any]:
    for path in [
        TECHNICAL_MANIFEST_FILE,
        TECHNICAL_LOCK_FILE,
        TECHNICAL_SOURCE_INVENTORY_FILE,
    ]:
        require_file(path)

    manifest = json.loads(
        TECHNICAL_MANIFEST_FILE.read_text(
            encoding="utf-8"
        )
    )
    lock = json.loads(
        TECHNICAL_LOCK_FILE.read_text(
            encoding="utf-8"
        )
    )

    expected = clean_text(lock.get("manifest_sha256"))
    observed = sha256_file(TECHNICAL_MANIFEST_FILE)
    if observed != expected:
        raise RuntimeError(
            "The 04d technical preflight does not match its lock."
        )

    verify_generated_files(lock, "04d")

    if clean_text(lock.get("status")) != (
        "FROZEN_BEFORE_REPRESENTATION_MODEL_FITTING"
    ):
        raise RuntimeError(
            "Unexpected 04d preflight status."
        )

    resolved = manifest.get("resolved_covariates", {})
    if int(resolved.get("n_strict_usable", -1)) != 0:
        raise RuntimeError(
            "04d no longer reports zero strict usable technical variables."
        )
    if int(
        resolved.get(
            "n_potential_cross_view_batch_candidates",
            -1,
        )
    ) != 0:
        raise RuntimeError(
            "04d no longer reports zero cross-view batch candidates."
        )

    source_inventory = pd.read_csv(
        TECHNICAL_SOURCE_INVENTORY_FILE,
        dtype=str,
        low_memory=False,
    ).fillna("")

    inventory_rows = int(source_inventory.shape[0])
    discovered = int(
        manifest.get("source_audit", {}).get(
            "n_permitted_sources_discovered",
            inventory_rows,
        )
    )

    return {
        "manifest": manifest,
        "lock": lock,
        "manifest_sha256": observed,
        "lock_sha256": sha256_file(
            TECHNICAL_LOCK_FILE
        ),
        "source_inventory_sha256": sha256_file(
            TECHNICAL_SOURCE_INVENTORY_FILE
        ),
        "source_inventory_rows": inventory_rows,
        "manifest_discovered_source_count": discovered,
        "source_count_discrepancy": bool(
            inventory_rows != discovered
        ),
        "source_count_interpretation": (
            "Cosmetic source-discovery accounting discrepancy; "
            "the frozen parsed source inventory is authoritative "
            "for downstream audit provenance."
            if inventory_rows != discovered
            else "No source-count discrepancy."
        ),
    }


def find_model_results() -> list[str]:
    found: list[str] = []
    for path in MODEL_RESULT_PATHS:
        if path.is_file() and path.stat().st_size > 0:
            found.append(relative_path(path))
        elif path.is_dir() and any(
            item.is_file()
            for item in path.rglob("*")
        ):
            found.append(relative_path(path))
    return found


# ---------------------------------------------------------------------
# Registry validation and scenario parsing
# ---------------------------------------------------------------------


def scenario_from_row(row: pd.Series) -> Scenario:
    n_shared = parse_int(row.get("n_shared_factors"))
    true_bio = parse_int(
        row.get("true_biological_shared_factors"),
        n_shared,
    )
    if true_bio != n_shared:
        raise RuntimeError(
            f"{row.get('scenario_id')}: n_shared_factors "
            "and true_biological_shared_factors disagree."
        )

    scenario = Scenario(
        scenario_id=clean_text(row.get("scenario_id")),
        description=clean_text(row.get("description")),
        scenario_origin=clean_text(
            row.get("scenario_origin")
        ) or "05a_base",
        scenario_family=clean_text(
            row.get("scenario_family")
        ) or "base_05a_screening",
        calibration_role=clean_text(
            row.get("calibration_role")
        ) or "architecture_screening",
        simulation_stage=clean_text(
            row.get("simulation_stage")
        ) or "architecture_screening",
        n_samples=parse_int(row.get("n_samples")),
        n_shared_factors=n_shared,
        n_private_factors_per_view=parse_int(
            row.get("n_private_factors_per_view")
        ),
        n_features_rna=parse_int(
            row.get("n_features_rna")
        ),
        n_features_methylation=parse_int(
            row.get("n_features_methylation")
        ),
        n_features_cnv=parse_int(
            row.get("n_features_cnv")
        ),
        shared_signal_strength=parse_float(
            row.get("shared_signal_strength"),
            1.0,
        ),
        private_signal_strength=parse_float(
            row.get("private_signal_strength"),
            1.0,
        ),
        noise_sd_rna=parse_float(
            row.get("noise_sd_rna"),
            1.0,
        ),
        noise_sd_methylation=parse_float(
            row.get("noise_sd_methylation"),
            1.0,
        ),
        noise_sd_cnv=parse_float(
            row.get("noise_sd_cnv"),
            1.0,
        ),
        missing_fraction_rna=parse_float(
            row.get("missing_fraction_rna"),
            0.0,
        ),
        missing_fraction_methylation=parse_float(
            row.get("missing_fraction_methylation"),
            0.0,
        ),
        missing_fraction_cnv=parse_float(
            row.get("missing_fraction_cnv"),
            0.0,
        ),
        pair_corruption_fraction=parse_float(
            row.get("pair_corruption_fraction"),
            0.0,
        ),
        uninformative_modality=normalized_token(
            row.get("uninformative_modality")
        ),
        block_scale_rna=parse_float(
            row.get("block_scale_rna"),
            1.0,
        ),
        block_scale_methylation=parse_float(
            row.get("block_scale_methylation"),
            1.0,
        ),
        block_scale_cnv=parse_float(
            row.get("block_scale_cnv"),
            1.0,
        ),
        nonlinear_mode=clean_text(
            row.get("nonlinear_mode")
        ).lower() or "none",
        shared_factor_correlation=parse_float(
            row.get("shared_factor_correlation"),
            0.0,
        ),
        shared_strength_pattern=clean_text(
            row.get("shared_strength_pattern")
        ).lower() or "equal",
        shared_support_pattern=normalized_token(
            row.get("shared_support_pattern"),
            "ALL3",
        ),
        shared_support_by_factor=clean_text(
            row.get("shared_support_by_factor")
        ),
        private_cross_view_correlation=parse_float(
            row.get("private_cross_view_correlation"),
            0.0,
        ),
        correlated_private_mode=clean_text(
            row.get("correlated_private_mode")
        ).lower() or "none",
        shared_loading_fraction=parse_float(
            row.get("shared_loading_fraction"),
            0.10,
        ),
        private_loading_fraction=parse_float(
            row.get("private_loading_fraction"),
            0.10,
        ),
        loading_regime=clean_text(
            row.get("loading_regime")
        ).lower() or "moderate_10pct",
        missingness_mode=clean_text(
            row.get("missingness_mode")
        ).lower() or "none",
        missingness_target_view=normalized_token(
            row.get("missingness_target_view")
        ),
        low_quality_sample_fraction=parse_float(
            row.get("low_quality_sample_fraction"),
            0.0,
        ),
        low_quality_feature_fraction=parse_float(
            row.get("low_quality_feature_fraction"),
            0.0,
        ),
        low_quality_missing_fraction=parse_float(
            row.get("low_quality_missing_fraction"),
            0.0,
        ),
        partial_view_sample_fraction=parse_float(
            row.get("partial_view_sample_fraction"),
            0.0,
        ),
        partial_view_missing_fraction=parse_float(
            row.get("partial_view_missing_fraction"),
            0.0,
        ),
        batch_mode=clean_text(
            row.get("batch_mode")
        ).lower() or "none",
        batch_cardinality=parse_int(
            row.get("batch_cardinality"),
            2,
        ),
        batch_prevalence=parse_float(
            row.get("batch_prevalence"),
            0.5,
        ),
        batch_effect_strength=parse_float(
            row.get("batch_effect_strength"),
            0.0,
        ),
        batch_view_support=normalized_token(
            row.get("batch_view_support")
        ),
        batch_signal_correlation=parse_float(
            row.get("batch_signal_correlation"),
            0.0,
        ),
        batch_missingness_increment=parse_float(
            row.get("batch_missingness_increment"),
            0.0,
        ),
        true_biological_shared_factors=true_bio,
        true_technical_shared_factors=parse_int(
            row.get("true_technical_shared_factors"),
            0,
        ),
        factor_level_identifiable=parse_bool(
            row.get("factor_level_identifiable"),
            n_shared > 0,
        ),
        subspace_level_identifiable=parse_bool(
            row.get("subspace_level_identifiable"),
            n_shared > 0,
        ),
        primary_truth_target=clean_text(
            row.get("primary_truth_target")
        ) or ("factor_and_subspace" if n_shared > 0 else "null_specificity"),
        stability_resamples_per_dataset=parse_int(
            row.get("stability_resamples_per_dataset"),
            20,
        ),
        materialization_policy=clean_text(
            row.get("materialization_policy")
        ) or "persist_npz",
        generator_required_version=clean_text(
            row.get("generator_required_version")
        ) or GENERATOR_VERSION,
        planned_replicates=parse_int(
            row.get("planned_replicates"),
            parse_int(row.get("replicates"), 20),
        ),
    )

    validate_scenario(scenario)
    return scenario


def validate_scenario(scenario: Scenario) -> None:
    if not scenario.scenario_id:
        raise ValueError("Empty scenario_id.")
    if scenario.simulation_stage != "architecture_screening":
        raise RuntimeError(
            f"{scenario.scenario_id}: only architecture_screening "
            "may be materialized by 05b v2."
        )
    if scenario.generator_required_version != GENERATOR_VERSION:
        raise RuntimeError(
            f"{scenario.scenario_id}: unexpected generator version "
            f"{scenario.generator_required_version}."
        )
    if scenario.materialization_policy != "persist_npz":
        raise RuntimeError(
            f"{scenario.scenario_id}: unsupported materialization policy."
        )
    if scenario.n_samples < 4:
        raise ValueError(
            f"{scenario.scenario_id}: n_samples is too small."
        )
    if min(
        scenario.n_features_rna,
        scenario.n_features_methylation,
        scenario.n_features_cnv,
    ) < 1:
        raise ValueError(
            f"{scenario.scenario_id}: feature counts must be positive."
        )
    if scenario.n_shared_factors < 0:
        raise ValueError("Negative shared-factor count.")
    if scenario.n_private_factors_per_view < 0:
        raise ValueError("Negative private-factor count.")
    if not (-0.99 < scenario.shared_factor_correlation < 0.99):
        raise ValueError(
            f"{scenario.scenario_id}: invalid shared correlation."
        )
    for value, label in [
        (scenario.missing_fraction_rna, "missing_fraction_rna"),
        (scenario.missing_fraction_methylation, "missing_fraction_methylation"),
        (scenario.missing_fraction_cnv, "missing_fraction_cnv"),
        (scenario.pair_corruption_fraction, "pair_corruption_fraction"),
        (scenario.low_quality_sample_fraction, "low_quality_sample_fraction"),
        (scenario.low_quality_feature_fraction, "low_quality_feature_fraction"),
        (scenario.low_quality_missing_fraction, "low_quality_missing_fraction"),
        (scenario.partial_view_sample_fraction, "partial_view_sample_fraction"),
        (scenario.partial_view_missing_fraction, "partial_view_missing_fraction"),
        (scenario.batch_prevalence, "batch_prevalence"),
        (scenario.batch_missingness_increment, "batch_missingness_increment"),
    ]:
        if not (0.0 <= value <= 1.0):
            raise ValueError(
                f"{scenario.scenario_id}: {label} outside [0, 1]."
            )
    if scenario.batch_mode != "none" and scenario.batch_cardinality != 2:
        raise RuntimeError(
            f"{scenario.scenario_id}: v2 currently requires binary batch truth."
        )


def load_and_validate_registries(
    addendum: dict[str, Any],
) -> tuple[
    dict[str, Scenario],
    pd.DataFrame,
    pd.DataFrame,
]:
    scenarios_table = pd.read_csv(
        COMBINED_SCENARIOS_FILE,
        dtype=str,
        low_memory=False,
    ).fillna("")
    replicates = pd.read_csv(
        COMBINED_REPLICATES_FILE,
        dtype=str,
        low_memory=False,
    ).fillna("")

    missing_scenario_columns = (
        REQUIRED_SCENARIO_COLUMNS
        - set(scenarios_table.columns)
    )
    if missing_scenario_columns:
        raise RuntimeError(
            "Combined scenario registry is missing columns: "
            + ", ".join(sorted(missing_scenario_columns))
        )

    missing_replicate_columns = (
        REQUIRED_REPLICATE_COLUMNS
        - set(replicates.columns)
    )
    if missing_replicate_columns:
        raise RuntimeError(
            "Combined replicate registry is missing columns: "
            + ", ".join(sorted(missing_replicate_columns))
        )

    if scenarios_table.empty or replicates.empty:
        raise RuntimeError(
            "The combined architecture-screening registries are empty."
        )
    if scenarios_table["scenario_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate scenario IDs exist in the combined registry."
        )

    scenarios = {
        scenario.scenario_id: scenario
        for scenario in (
            scenario_from_row(row)
            for _, row in scenarios_table.iterrows()
        )
    }

    unknown = sorted(
        set(replicates["scenario_id"])
        - set(scenarios)
    )
    if unknown:
        raise RuntimeError(
            "Replicate registry contains unknown scenarios: "
            + ", ".join(unknown)
        )

    if replicates.duplicated(
        ["scenario_id", "replicate_index"]
    ).any():
        raise RuntimeError(
            "Duplicate scenario/replicate entries exist."
        )

    if replicates["simulation_seed"].duplicated().any():
        duplicates = (
            replicates.loc[
                replicates["simulation_seed"].duplicated(),
                "simulation_seed",
            ]
            .head(20)
            .tolist()
        )
        raise RuntimeError(
            f"Duplicate simulation seeds detected: {duplicates}"
        )

    for column in [
        "scenario_order",
        "replicate_index",
        "simulation_seed",
        "stability_resamples_per_dataset",
    ]:
        replicates[column] = pd.to_numeric(
            replicates[column],
            errors="raise",
        ).astype("int64")

    if not replicates["simulation_stage"].eq(
        "architecture_screening"
    ).all():
        raise RuntimeError(
            "Non-architecture rows exist in the v2 materialization registry."
        )
    if not replicates["materialization_policy"].eq(
        "persist_npz"
    ).all():
        raise RuntimeError(
            "Unexpected materialization policy in replicate registry."
        )

    expected_scenarios = int(
        addendum.get("architecture_screening", {}).get(
            "combined_scenarios",
            len(scenarios),
        )
    )
    expected_datasets = int(
        addendum.get("architecture_screening", {}).get(
            "combined_replicates",
            replicates.shape[0],
        )
    )

    if len(scenarios) != expected_scenarios:
        raise RuntimeError(
            f"Scenario count mismatch: {len(scenarios)} vs {expected_scenarios}."
        )
    if replicates.shape[0] != expected_datasets:
        raise RuntimeError(
            f"Dataset count mismatch: {replicates.shape[0]} vs {expected_datasets}."
        )

    observed_counts = (
        replicates.groupby("scenario_id")
        .size()
        .to_dict()
    )
    for scenario_id, scenario in scenarios.items():
        observed = int(observed_counts.get(scenario_id, 0))
        if observed != scenario.planned_replicates:
            raise RuntimeError(
                f"{scenario_id}: replicate count {observed} "
                f"does not match planned {scenario.planned_replicates}."
            )

    truth_taxonomy = pd.read_csv(
        TRUTH_TAXONOMY_FILE,
        dtype=str,
        low_memory=False,
    ).fillna("")
    if truth_taxonomy.empty:
        raise RuntimeError(
            "The simulation truth taxonomy is empty."
        )

    final_gate_replicates = pd.read_csv(
        FINAL_GATE_REPLICATES_FILE,
        dtype=str,
        low_memory=False,
    ).fillna("")
    if final_gate_replicates.empty:
        raise RuntimeError(
            "The frozen final-gate registry is unexpectedly empty."
        )

    return scenarios, replicates, truth_taxonomy


# ---------------------------------------------------------------------
# Simulation truth generation
# ---------------------------------------------------------------------


def support_tokens_for_factors(
    scenario: Scenario,
) -> list[str]:
    if scenario.n_shared_factors == 0:
        return []

    raw = clean_text(
        scenario.shared_support_by_factor
    )
    if raw:
        tokens = [
            normalized_token(item)
            for item in raw.split("|")
            if clean_text(item)
        ]
    else:
        token = (
            scenario.shared_support_pattern
            if scenario.shared_support_pattern != "MIXED"
            else "ALL3"
        )
        tokens = [
            normalized_token(token)
            for _ in range(
                scenario.n_shared_factors
            )
        ]

    if len(tokens) != scenario.n_shared_factors:
        raise RuntimeError(
            f"{scenario.scenario_id}: support-token count "
            f"{len(tokens)} does not match shared factors "
            f"{scenario.n_shared_factors}."
        )

    for token in tokens:
        if len(view_tokens(token)) < 2:
            raise RuntimeError(
                f"{scenario.scenario_id}: biological shared factor "
                f"has unsupported view truth {token}."
            )

    return tokens


def support_mask_from_tokens(
    tokens: Iterable[str],
) -> np.ndarray:
    token_list = list(tokens)
    mask = np.zeros(
        (len(token_list), len(VIEWS)),
        dtype=np.uint8,
    )
    for factor_index, token in enumerate(token_list):
        support = view_tokens(token)
        for view in support:
            mask[factor_index, VIEW_INDEX[view]] = 1
    return mask


def truth_class_from_support(token: str) -> str:
    mapping = {
        "ALL3": "BIO_SHARED_ALL3",
        "RNA_METH": "BIO_SHARED_RNA_METH",
        "RNA_CNV": "BIO_SHARED_RNA_CNV",
        "METH_CNV": "BIO_SHARED_METH_CNV",
    }
    normalized = normalized_token(token)
    if normalized not in mapping:
        raise ValueError(
            f"No biological truth class for support {token}."
        )
    return mapping[normalized]


def generate_correlated_scores(
    rng: np.random.Generator,
    n_samples: int,
    n_factors: int,
    correlation: float,
) -> np.ndarray:
    if n_factors == 0:
        return np.zeros(
            (n_samples, 0),
            dtype=DEFAULT_DTYPE,
        )
    if n_factors == 1:
        return standardize_columns(
            rng.normal(size=(n_samples, 1))
        )

    lower_bound = -1.0 / (n_factors - 1)
    if not (lower_bound < correlation < 1.0):
        raise ValueError(
            "Equicorrelation matrix is not positive definite."
        )

    covariance = np.full(
        (n_factors, n_factors),
        correlation,
        dtype=np.float64,
    )
    np.fill_diagonal(covariance, 1.0)
    cholesky = np.linalg.cholesky(covariance)
    raw = rng.normal(size=(n_samples, n_factors))
    return standardize_columns(raw @ cholesky.T)


def factor_strengths(scenario: Scenario) -> np.ndarray:
    k = scenario.n_shared_factors
    if k == 0:
        return np.zeros((0,), dtype=DEFAULT_DTYPE)

    pattern = scenario.shared_strength_pattern
    if pattern in {"equal", "equal_near_degenerate"}:
        values = np.ones(k, dtype=DEFAULT_DTYPE)
    elif pattern in {"decreasing", "linear_decay"}:
        values = np.linspace(
            1.0,
            0.5,
            k,
            dtype=DEFAULT_DTYPE,
        )
    else:
        raise ValueError(
            f"Unsupported shared-strength pattern: {pattern}"
        )

    return values


def generate_private_scores(
    rng: np.random.Generator,
    scenario: Scenario,
) -> dict[str, np.ndarray]:
    n = scenario.n_samples
    k = scenario.n_private_factors_per_view

    if k == 0:
        return {
            view: np.zeros((n, 0), dtype=DEFAULT_DTYPE)
            for view in VIEWS
        }

    rho = scenario.private_cross_view_correlation
    correlated = (
        scenario.correlated_private_mode
        == "correlated_private_scores"
        and rho > 0
    )

    if correlated:
        common = standardize_columns(
            rng.normal(size=(n, k))
        ).astype(np.float64)
        output: dict[str, np.ndarray] = {}
        for view in VIEWS:
            noise = standardize_columns(
                rng.normal(size=(n, k))
            ).astype(np.float64)
            values = (
                math.sqrt(rho) * common
                + math.sqrt(1.0 - rho) * noise
            )
            output[view] = standardize_columns(values)
        return output

    return {
        view: standardize_columns(
            rng.normal(size=(n, k))
        )
        for view in VIEWS
    }


def generate_sparse_loadings(
    rng: np.random.Generator,
    n_features: int,
    n_factors: int,
    support_fraction: float,
    factor_support: np.ndarray | None = None,
) -> np.ndarray:
    matrix = np.zeros(
        (n_features, n_factors),
        dtype=DEFAULT_DTYPE,
    )
    if n_factors == 0:
        return matrix

    n_support = max(
        1,
        min(
            n_features,
            int(round(n_features * support_fraction)),
        ),
    )

    for factor_index in range(n_factors):
        if (
            factor_support is not None
            and not bool(factor_support[factor_index])
        ):
            continue
        selected = rng.choice(
            n_features,
            size=n_support,
            replace=False,
        )
        values = rng.normal(size=n_support)
        rms = float(np.sqrt(np.mean(values**2)))
        if not np.isfinite(rms) or rms <= 0:
            raise RuntimeError(
                "Invalid simulated loading scale."
            )
        matrix[selected, factor_index] = (
            values / rms
        ).astype(DEFAULT_DTYPE)

    return matrix


def exact_binary_labels(
    rng: np.random.Generator,
    n_samples: int,
    prevalence: float,
) -> np.ndarray:
    n_positive = int(round(n_samples * prevalence))
    n_positive = max(1, min(n_samples - 1, n_positive))
    labels = np.zeros(n_samples, dtype=np.int8)
    selected = rng.choice(
        n_samples,
        size=n_positive,
        replace=False,
    )
    labels[selected] = 1
    return labels


def standardized_binary_score(labels: np.ndarray) -> np.ndarray:
    values = labels.astype(np.float64)
    sd = float(values.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError(
            "Batch labels have no variation."
        )
    return (
        (values - values.mean()) / sd
    ).astype(DEFAULT_DTYPE)


def generate_technical_truth(
    rng: np.random.Generator,
    scenario: Scenario,
    z_bio: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    k = scenario.true_technical_shared_factors
    n = scenario.n_samples

    if k == 0:
        return (
            np.zeros((n, 0), dtype=DEFAULT_DTYPE),
            np.zeros((n, 0), dtype=np.int8),
            np.zeros((0, len(VIEWS)), dtype=np.uint8),
            float("nan"),
        )

    if k != 1:
        raise RuntimeError(
            f"{scenario.scenario_id}: v2 currently supports one "
            "technical factor per batch scenario."
        )

    labels = exact_binary_labels(
        rng,
        n,
        scenario.batch_prevalence,
    )

    if (
        scenario.batch_mode == "signal_correlated"
        and scenario.batch_signal_correlation > 0
        and z_bio.shape[1] > 0
    ):
        rho = scenario.batch_signal_correlation
        independent = standardize_columns(
            rng.normal(size=(n, 1))
        )[:, 0].astype(np.float64)
        base = z_bio[:, 0].astype(np.float64)
        score = (
            rho * base
            + math.sqrt(1.0 - rho**2) * independent
        )
        score = standardize_columns(
            score.reshape(-1, 1)
        )[:, 0]

        # Batch labels remain exact-prevalence categorical truth.
        n_positive = int(labels.sum())
        order = np.argsort(score)
        labels[:] = 0
        labels[order[-n_positive:]] = 1
    else:
        score = standardized_binary_score(labels)

    support = support_mask_from_tokens(
        [scenario.batch_view_support]
    )

    actual_correlation = float("nan")
    if z_bio.shape[1] > 0:
        actual_correlation = float(
            np.corrcoef(
                score.astype(np.float64),
                z_bio[:, 0].astype(np.float64),
            )[0, 1]
        )

    return (
        score.reshape(-1, 1).astype(DEFAULT_DTYPE),
        labels.reshape(-1, 1),
        support,
        actual_correlation,
    )


def apply_shared_nonlinearity(
    component: np.ndarray,
    view: str,
    mode: str,
) -> np.ndarray:
    normalized_mode = clean_text(mode).lower() or "none"
    if normalized_mode == "none":
        return component.astype(DEFAULT_DTYPE, copy=False)

    transform = "identity"
    if normalized_mode == "methylation_tanh":
        transform = (
            "tanh" if view == "METHYLATION" else "identity"
        )
    elif normalized_mode == "view_specific_monotonic":
        transform = {
            "RNA": "identity",
            "METHYLATION": "tanh",
            "CNV": "signed_sqrt",
        }[view]
    elif normalized_mode == "tanh_all":
        transform = "tanh"
    else:
        raise ValueError(
            f"Unsupported nonlinear mode: {mode}"
        )

    if transform == "identity":
        return component.astype(DEFAULT_DTYPE, copy=False)

    x = component.astype(np.float64)
    original_sd = x.std(axis=0, ddof=1)
    original_sd[(~np.isfinite(original_sd)) | (original_sd <= 0)] = 1.0

    if transform == "tanh":
        transformed = np.tanh(x)
    elif transform == "signed_sqrt":
        transformed = np.sign(x) * np.sqrt(np.abs(x))
    else:
        raise AssertionError(transform)

    transformed -= transformed.mean(axis=0, keepdims=True)
    transformed_sd = transformed.std(axis=0, ddof=1)
    transformed_sd[
        (~np.isfinite(transformed_sd))
        | (transformed_sd <= 0)
    ] = 1.0
    transformed = transformed / transformed_sd * original_sd
    return transformed.astype(DEFAULT_DTYPE)


def deranged_row_map(
    rng: np.random.Generator,
    n_samples: int,
    corruption_fraction: float,
) -> np.ndarray:
    row_map = np.arange(n_samples, dtype=np.int32)
    proposed = int(round(n_samples * corruption_fraction))
    if proposed < 2:
        return row_map
    proposed = min(proposed, n_samples)
    selected = rng.choice(
        n_samples,
        size=proposed,
        replace=False,
    )
    selected = selected[
        rng.permutation(proposed)
    ]
    shift = int(rng.integers(1, proposed))
    row_map[selected] = np.roll(selected, shift)
    if np.any(row_map[selected] == selected):
        raise RuntimeError(
            "Failed to create an exact derangement."
        )
    return row_map


# ---------------------------------------------------------------------
# Missingness mechanisms
# ---------------------------------------------------------------------


def choose_exact_mask(
    rng: np.random.Generator,
    eligible: np.ndarray,
    target_count: int,
) -> np.ndarray:
    if eligible.dtype != bool:
        eligible = eligible.astype(bool)
    available = np.flatnonzero(eligible.ravel())
    target = max(0, min(int(target_count), available.size))
    chosen = np.zeros(eligible.size, dtype=bool)
    if target > 0:
        indices = rng.choice(
            available,
            size=target,
            replace=False,
        )
        chosen[indices] = True
    return chosen.reshape(eligible.shape)


def mark_missing(
    reason: np.ndarray,
    selected: np.ndarray,
    reason_code: int,
) -> int:
    valid = selected & (reason == MISSING_REASON_OBSERVED)
    reason[valid] = np.uint8(reason_code)
    return int(valid.sum())


def apply_missingness(
    rng: np.random.Generator,
    scenario: Scenario,
    view: str,
    complete_observed: np.ndarray,
    row_map: np.ndarray,
    batch_labels: np.ndarray,
    low_quality_samples_canonical: np.ndarray,
    partial_samples_canonical: np.ndarray,
    low_quality_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    n, p = complete_observed.shape
    reason = np.zeros((n, p), dtype=np.uint8)
    counts = {
        name: 0
        for name in MISSING_REASON_REGISTRY.values()
        if name != "observed"
    }

    # Base per-view MCAR defined in 05a columns.
    base_fraction = scenario.missing_fraction(view)
    base_target = int(round(n * p * base_fraction))
    selected = choose_exact_mask(
        rng,
        np.ones((n, p), dtype=bool),
        base_target,
    )
    counts["base_mcar"] += mark_missing(
        reason,
        selected,
        MISSING_REASON_BASE_MCAR,
    )

    mode = scenario.missingness_mode
    target_views = view_tokens(
        scenario.missingness_target_view
    )

    if mode in {"none", "mcar_cell"}:
        pass

    elif mode == "feature_structured":
        if view in target_views:
            eligible = np.broadcast_to(
                low_quality_features.reshape(1, -1),
                (n, p),
            ).copy()
            target = int(round(
                eligible.sum()
                * scenario.low_quality_missing_fraction
            ))
            selected = choose_exact_mask(
                rng,
                eligible & (reason == 0),
                target,
            )
            counts["feature_structured"] += mark_missing(
                reason,
                selected,
                MISSING_REASON_FEATURE_STRUCTURED,
            )

    elif mode == "sample_quality":
        if view in target_views:
            observed_low_quality = (
                low_quality_samples_canonical[row_map]
            )
            eligible = np.broadcast_to(
                observed_low_quality.reshape(-1, 1),
                (n, p),
            ).copy()
            target = int(round(
                eligible.sum()
                * scenario.low_quality_missing_fraction
            ))
            selected = choose_exact_mask(
                rng,
                eligible & (reason == 0),
                target,
            )
            counts["sample_quality"] += mark_missing(
                reason,
                selected,
                MISSING_REASON_SAMPLE_QUALITY,
            )

    elif mode == "partial_view":
        if view in target_views:
            observed_partial = (
                partial_samples_canonical[row_map]
            )
            eligible = np.broadcast_to(
                observed_partial.reshape(-1, 1),
                (n, p),
            ).copy()
            target = int(round(
                eligible.sum()
                * scenario.partial_view_missing_fraction
            ))
            selected = choose_exact_mask(
                rng,
                eligible & (reason == 0),
                target,
            )
            counts["partial_view"] += mark_missing(
                reason,
                selected,
                MISSING_REASON_PARTIAL_VIEW,
            )

    elif mode == "batch_dependent":
        if view in target_views:
            if batch_labels.shape[1] == 0:
                raise RuntimeError(
                    f"{scenario.scenario_id}: batch-dependent missingness "
                    "requires batch truth."
                )
            observed_batch = batch_labels[
                row_map,
                0,
            ] == 1
            eligible = np.broadcast_to(
                observed_batch.reshape(-1, 1),
                (n, p),
            ).copy()
            target = int(round(
                eligible.sum()
                * scenario.batch_missingness_increment
            ))
            selected = choose_exact_mask(
                rng,
                eligible & (reason == 0),
                target,
            )
            counts["batch_dependent"] += mark_missing(
                reason,
                selected,
                MISSING_REASON_BATCH_DEPENDENT,
            )

    else:
        raise ValueError(
            f"Unsupported missingness mode: {mode}"
        )

    observed = complete_observed.copy()
    observed[reason > 0] = np.nan
    return observed.astype(DEFAULT_DTYPE), reason, counts


# ---------------------------------------------------------------------
# Complete dataset generation
# ---------------------------------------------------------------------


def generate_dataset(
    scenario: Scenario,
    registry_label: str,
    replicate_index: int,
    simulation_seed: int,
) -> dict[str, np.ndarray]:
    root_rng = np.random.default_rng(simulation_seed)
    n = scenario.n_samples

    support_tokens = support_tokens_for_factors(scenario)
    bio_support_mask = support_mask_from_tokens(
        support_tokens
    )

    z_bio = generate_correlated_scores(
        root_rng,
        n,
        scenario.n_shared_factors,
        scenario.shared_factor_correlation,
    )
    shared_strength = factor_strengths(scenario)

    private_scores = generate_private_scores(
        root_rng,
        scenario,
    )

    (
        z_technical,
        batch_labels,
        technical_support_mask,
        actual_batch_signal_correlation,
    ) = generate_technical_truth(
        root_rng,
        scenario,
        z_bio,
    )

    low_quality_samples = np.zeros(n, dtype=np.uint8)
    if (
        scenario.missingness_mode == "sample_quality"
        and scenario.low_quality_sample_fraction > 0
    ):
        count = max(
            1,
            int(round(
                n * scenario.low_quality_sample_fraction
            )),
        )
        selected = root_rng.choice(
            n,
            size=min(count, n),
            replace=False,
        )
        low_quality_samples[selected] = 1

    partial_samples = np.zeros(n, dtype=np.uint8)
    if (
        scenario.missingness_mode == "partial_view"
        and scenario.partial_view_sample_fraction > 0
    ):
        count = max(
            1,
            int(round(
                n * scenario.partial_view_sample_fraction
            )),
        )
        selected = root_rng.choice(
            n,
            size=min(count, n),
            replace=False,
        )
        partial_samples[selected] = 1

    arrays: dict[str, np.ndarray] = {
        "sample_index": np.arange(n, dtype=np.int32),
        "Z_biological_shared": z_bio,
        "biological_factor_strengths": shared_strength,
        "biological_support_mask": bio_support_mask,
        "biological_factor_ids": np.asarray(
            [
                f"BIO_SHARED_{index + 1:03d}"
                for index in range(
                    scenario.n_shared_factors
                )
            ],
            dtype="U32",
        ),
        "biological_factor_truth_class": np.asarray(
            [
                truth_class_from_support(token)
                for token in support_tokens
            ],
            dtype="U32",
        ),
        "Z_technical": z_technical,
        "batch_labels": batch_labels,
        "technical_support_mask": technical_support_mask,
        "technical_factor_ids": np.asarray(
            [
                f"TECH_BATCH_{index + 1:03d}"
                for index in range(
                    scenario.true_technical_shared_factors
                )
            ],
            dtype="U32",
        ),
        "low_quality_sample_mask": low_quality_samples,
        "partial_view_sample_mask": partial_samples,
        "scenario_id": np.asarray(
            scenario.scenario_id,
            dtype="U128",
        ),
        "scenario_family": np.asarray(
            scenario.scenario_family,
            dtype="U128",
        ),
        "registry_label": np.asarray(
            registry_label,
            dtype="U128",
        ),
        "simulation_stage": np.asarray(
            scenario.simulation_stage,
            dtype="U64",
        ),
        "replicate_index": np.asarray(
            replicate_index,
            dtype=np.int32,
        ),
        "simulation_seed": np.asarray(
            simulation_seed,
            dtype=np.uint32,
        ),
        "generator_version": np.asarray(
            GENERATOR_VERSION,
            dtype="U64",
        ),
        "scenario_sha256": np.asarray(
            sha256_text(
                canonical_json(
                    scenario.canonical_dict()
                )
            ),
            dtype="U64",
        ),
        "factor_level_identifiable": np.asarray(
            scenario.factor_level_identifiable,
            dtype=np.bool_,
        ),
        "subspace_level_identifiable": np.asarray(
            scenario.subspace_level_identifiable,
            dtype=np.bool_,
        ),
        "primary_truth_target": np.asarray(
            scenario.primary_truth_target,
            dtype="U64",
        ),
        "actual_batch_signal_correlation": np.asarray(
            actual_batch_signal_correlation,
            dtype=np.float32,
        ),
    }

    for view in VIEWS:
        prefix = VIEW_PREFIX[view]
        p = scenario.n_features(view)
        view_rng = np.random.default_rng(
            stable_seed(
                simulation_seed,
                scenario.scenario_id,
                view,
            )
        )

        bio_support_for_view = (
            bio_support_mask[:, VIEW_INDEX[view]].astype(bool)
            if scenario.n_shared_factors > 0
            else np.zeros((0,), dtype=bool)
        )
        technical_support_for_view = (
            technical_support_mask[:, VIEW_INDEX[view]].astype(bool)
            if scenario.true_technical_shared_factors > 0
            else np.zeros((0,), dtype=bool)
        )

        w_bio = generate_sparse_loadings(
            view_rng,
            p,
            scenario.n_shared_factors,
            scenario.shared_loading_fraction,
            bio_support_for_view,
        )
        w_private = generate_sparse_loadings(
            view_rng,
            p,
            scenario.n_private_factors_per_view,
            scenario.private_loading_fraction,
        )
        w_technical = generate_sparse_loadings(
            view_rng,
            p,
            scenario.true_technical_shared_factors,
            DEFAULT_TECHNICAL_LOADING_FRACTION,
            technical_support_for_view,
        )

        if view == scenario.uninformative_modality:
            w_bio[:] = 0
            w_private[:] = 0
            w_technical[:] = 0

        if scenario.n_shared_factors > 0:
            weighted_scores = (
                z_bio
                * shared_strength.reshape(1, -1)
            )
            bio_component = weighted_scores @ w_bio.T
            bio_component = apply_shared_nonlinearity(
                bio_component.astype(DEFAULT_DTYPE),
                view,
                scenario.nonlinear_mode,
            )
        else:
            bio_component = np.zeros(
                (n, p),
                dtype=DEFAULT_DTYPE,
            )

        if scenario.n_private_factors_per_view > 0:
            private_component = (
                private_scores[view]
                @ w_private.T
            ).astype(DEFAULT_DTYPE)
        else:
            private_component = np.zeros(
                (n, p),
                dtype=DEFAULT_DTYPE,
            )

        if scenario.true_technical_shared_factors > 0:
            technical_component = (
                z_technical
                @ w_technical.T
            ).astype(DEFAULT_DTYPE)
        else:
            technical_component = np.zeros(
                (n, p),
                dtype=DEFAULT_DTYPE,
            )

        feature_offset = view_rng.normal(
            0.0,
            DEFAULT_FEATURE_OFFSET_SD,
            size=(1, p),
        ).astype(DEFAULT_DTYPE)
        noise = view_rng.normal(
            0.0,
            scenario.noise_sd(view),
            size=(n, p),
        ).astype(DEFAULT_DTYPE)

        canonical_complete = (
            scenario.shared_signal_strength
            * bio_component
            + scenario.private_signal_strength
            * private_component
            + scenario.batch_effect_strength
            * technical_component
            + feature_offset
            + noise
        ).astype(DEFAULT_DTYPE)
        canonical_complete *= np.float32(
            scenario.block_scale(view)
        )

        if view == "RNA":
            row_map = np.arange(n, dtype=np.int32)
        else:
            row_map = deranged_row_map(
                view_rng,
                n,
                scenario.pair_corruption_fraction,
            )

        complete_observed = canonical_complete[
            row_map,
            :,
        ].astype(DEFAULT_DTYPE, copy=False)

        low_quality_features = np.zeros(
            p,
            dtype=np.uint8,
        )
        if (
            scenario.missingness_mode
            == "feature_structured"
            and view
            in view_tokens(
                scenario.missingness_target_view
            )
            and scenario.low_quality_feature_fraction > 0
        ):
            count = max(
                1,
                int(round(
                    p
                    * scenario.low_quality_feature_fraction
                )),
            )
            selected = view_rng.choice(
                p,
                size=min(count, p),
                replace=False,
            )
            low_quality_features[selected] = 1

        observed, missing_reason, missing_counts = (
            apply_missingness(
                view_rng,
                scenario,
                view,
                complete_observed,
                row_map,
                batch_labels,
                low_quality_samples,
                partial_samples,
                low_quality_features.astype(bool),
            )
        )

        arrays[f"X_{prefix}"] = observed
        arrays[f"X_complete_{prefix}"] = (
            complete_observed
        )
        arrays[f"missing_reason_{prefix}"] = (
            missing_reason
        )
        arrays[f"low_quality_feature_mask_{prefix}"] = (
            low_quality_features
        )
        arrays[f"Z_private_{prefix}"] = (
            private_scores[view]
        )
        arrays[f"W_biological_shared_{prefix}"] = w_bio
        arrays[f"W_private_{prefix}"] = w_private
        arrays[f"W_technical_{prefix}"] = w_technical
        arrays[f"row_source_{prefix}"] = row_map
        arrays[f"missing_count_base_mcar_{prefix}"] = (
            np.asarray(
                missing_counts["base_mcar"],
                dtype=np.int64,
            )
        )
        arrays[
            f"missing_count_feature_structured_{prefix}"
        ] = np.asarray(
            missing_counts["feature_structured"],
            dtype=np.int64,
        )
        arrays[f"missing_count_sample_quality_{prefix}"] = (
            np.asarray(
                missing_counts["sample_quality"],
                dtype=np.int64,
            )
        )
        arrays[f"missing_count_partial_view_{prefix}"] = (
            np.asarray(
                missing_counts["partial_view"],
                dtype=np.int64,
            )
        )
        arrays[
            f"missing_count_batch_dependent_{prefix}"
        ] = np.asarray(
            missing_counts["batch_dependent"],
            dtype=np.int64,
        )

    return arrays


# ---------------------------------------------------------------------
# Deterministic NPZ serialization
# ---------------------------------------------------------------------


def deterministic_npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(
        buffer,
        np.asarray(array),
        allow_pickle=False,
        version=(2, 0),
    )
    return buffer.getvalue()


def write_deterministic_npz(
    path: Path,
    arrays: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.stem + ".",
        suffix=".tmp.npz",
        dir=str(path.parent),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)

    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_STORED,
            strict_timestamps=True,
        ) as archive:
            for key in sorted(arrays):
                info = zipfile.ZipInfo(
                    filename=f"{key}.npy",
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                archive.writestr(
                    info,
                    deterministic_npy_bytes(arrays[key]),
                    compress_type=zipfile.ZIP_STORED,
                )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            key: archive[key]
            for key in archive.files
        }


def dataset_filename(
    scenario_id: str,
    replicate_index: int,
    simulation_seed: int,
) -> str:
    return (
        f"{scenario_id}__R{replicate_index:03d}__"
        f"S{simulation_seed:010d}.npz"
    )


def dataset_path(
    scenario: Scenario,
    replicate_index: int,
    simulation_seed: int,
) -> Path:
    return (
        ARCHITECTURE_DATA_ROOT
        / scenario.scenario_family
        / scenario.scenario_id
        / dataset_filename(
            scenario.scenario_id,
            replicate_index,
            simulation_seed,
        )
    )


# ---------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------


def expected_corrupted_rows(
    scenario: Scenario,
    view: str,
) -> int:
    if view == "RNA":
        return 0
    proposed = int(round(
        scenario.n_samples
        * scenario.pair_corruption_fraction
    ))
    return proposed if proposed >= 2 else 0


def validate_dataset_arrays(
    arrays: dict[str, np.ndarray],
    scenario: Scenario,
    registry_label: str,
    replicate_index: int,
    simulation_seed: int,
) -> dict[str, Any]:
    scalar_expectations = {
        "scenario_id": scenario.scenario_id,
        "scenario_family": scenario.scenario_family,
        "registry_label": registry_label,
        "simulation_stage": "architecture_screening",
        "generator_version": GENERATOR_VERSION,
        "primary_truth_target": scenario.primary_truth_target,
    }
    for key, expected in scalar_expectations.items():
        if key not in arrays:
            raise RuntimeError(f"Missing scalar array: {key}")
        observed = str(arrays[key].item())
        if observed != expected:
            raise RuntimeError(
                f"{key} mismatch: {observed!r} vs {expected!r}"
            )

    if int(arrays["replicate_index"].item()) != replicate_index:
        raise RuntimeError("Replicate index mismatch.")
    if int(arrays["simulation_seed"].item()) != simulation_seed:
        raise RuntimeError("Simulation seed mismatch.")

    expected_scenario_hash = sha256_text(
        canonical_json(scenario.canonical_dict())
    )
    if str(arrays["scenario_sha256"].item()) != expected_scenario_hash:
        raise RuntimeError("Scenario-hash mismatch.")

    n = scenario.n_samples
    k_bio = scenario.n_shared_factors
    k_private = scenario.n_private_factors_per_view
    k_tech = scenario.true_technical_shared_factors

    required_global = {
        "sample_index",
        "Z_biological_shared",
        "biological_factor_strengths",
        "biological_support_mask",
        "biological_factor_ids",
        "biological_factor_truth_class",
        "Z_technical",
        "batch_labels",
        "technical_support_mask",
        "technical_factor_ids",
        "low_quality_sample_mask",
        "partial_view_sample_mask",
        "actual_batch_signal_correlation",
    }
    missing_global = required_global - set(arrays)
    if missing_global:
        raise RuntimeError(
            "Missing global arrays: "
            + ", ".join(sorted(missing_global))
        )

    if arrays["sample_index"].shape != (n,):
        raise RuntimeError("Sample-index shape mismatch.")
    if arrays["Z_biological_shared"].shape != (n, k_bio):
        raise RuntimeError("Biological-score shape mismatch.")
    if arrays["biological_support_mask"].shape != (
        k_bio,
        len(VIEWS),
    ):
        raise RuntimeError("Biological support-mask shape mismatch.")
    if arrays["Z_technical"].shape != (n, k_tech):
        raise RuntimeError("Technical-score shape mismatch.")
    if arrays["batch_labels"].shape != (n, k_tech):
        raise RuntimeError("Batch-label shape mismatch.")
    if arrays["technical_support_mask"].shape != (
        k_tech,
        len(VIEWS),
    ):
        raise RuntimeError("Technical support-mask shape mismatch.")

    for key in [
        "Z_biological_shared",
        "Z_technical",
    ]:
        if not np.all(np.isfinite(arrays[key])):
            raise RuntimeError(
                f"Non-finite latent truth in {key}."
            )

    summary: dict[str, Any] = {
        "n_samples": n,
        "n_biological_shared_factors": k_bio,
        "n_private_factors_per_view": k_private,
        "n_technical_shared_factors": k_tech,
        "actual_batch_signal_correlation": float(
            arrays[
                "actual_batch_signal_correlation"
            ].item()
        ),
    }

    for view in VIEWS:
        prefix = VIEW_PREFIX[view]
        p = scenario.n_features(view)
        required = {
            f"X_{prefix}",
            f"X_complete_{prefix}",
            f"missing_reason_{prefix}",
            f"low_quality_feature_mask_{prefix}",
            f"Z_private_{prefix}",
            f"W_biological_shared_{prefix}",
            f"W_private_{prefix}",
            f"W_technical_{prefix}",
            f"row_source_{prefix}",
        }
        missing = required - set(arrays)
        if missing:
            raise RuntimeError(
                f"{view}: missing arrays: "
                + ", ".join(sorted(missing))
            )

        observed = arrays[f"X_{prefix}"]
        complete = arrays[f"X_complete_{prefix}"]
        reason = arrays[f"missing_reason_{prefix}"]
        private = arrays[f"Z_private_{prefix}"]
        w_bio = arrays[f"W_biological_shared_{prefix}"]
        w_private = arrays[f"W_private_{prefix}"]
        w_technical = arrays[f"W_technical_{prefix}"]
        row_source = arrays[f"row_source_{prefix}"]

        if observed.shape != (n, p) or observed.dtype != DEFAULT_DTYPE:
            raise RuntimeError(
                f"{view}: observed matrix shape/dtype mismatch."
            )
        if complete.shape != (n, p) or complete.dtype != DEFAULT_DTYPE:
            raise RuntimeError(
                f"{view}: complete matrix shape/dtype mismatch."
            )
        if not np.all(np.isfinite(complete)):
            raise RuntimeError(
                f"{view}: complete matrix contains non-finite values."
            )
        if reason.shape != (n, p) or reason.dtype != np.uint8:
            raise RuntimeError(
                f"{view}: missing-reason shape/dtype mismatch."
            )
        if private.shape != (n, k_private):
            raise RuntimeError(
                f"{view}: private-score shape mismatch."
            )
        if w_bio.shape != (p, k_bio):
            raise RuntimeError(
                f"{view}: biological-loading shape mismatch."
            )
        if w_private.shape != (p, k_private):
            raise RuntimeError(
                f"{view}: private-loading shape mismatch."
            )
        if w_technical.shape != (p, k_tech):
            raise RuntimeError(
                f"{view}: technical-loading shape mismatch."
            )
        if row_source.shape != (n,):
            raise RuntimeError(
                f"{view}: row-source shape mismatch."
            )
        if set(row_source.astype(int).tolist()) != set(range(n)):
            raise RuntimeError(
                f"{view}: row-source is not a permutation."
            )

        observed_missing = ~np.isfinite(observed)
        reason_missing = reason > 0
        if not np.array_equal(
            observed_missing,
            reason_missing,
        ):
            raise RuntimeError(
                f"{view}: NaN mask and missing-reason mask disagree."
            )
        finite = ~observed_missing
        if finite.any() and not np.array_equal(
            observed[finite],
            complete[finite],
        ):
            raise RuntimeError(
                f"{view}: observed finite values differ from complete truth."
            )

        allowed_reason_codes = set(
            MISSING_REASON_REGISTRY
        )
        observed_codes = set(
            np.unique(reason).astype(int).tolist()
        )
        if not observed_codes.issubset(
            allowed_reason_codes
        ):
            raise RuntimeError(
                f"{view}: unknown missing-reason codes."
            )

        expected_corruption = expected_corrupted_rows(
            scenario,
            view,
        )
        observed_corruption = int(
            np.sum(
                row_source
                != np.arange(n, dtype=np.int32)
            )
        )
        if observed_corruption != expected_corruption:
            raise RuntimeError(
                f"{view}: pair-corruption count mismatch "
                f"{observed_corruption} vs {expected_corruption}."
            )

        summary[f"{prefix}_missing_fraction"] = float(
            observed_missing.mean()
        )
        summary[f"{prefix}_missing_count"] = int(
            observed_missing.sum()
        )
        summary[f"{prefix}_corrupted_rows"] = (
            observed_corruption
        )
        summary[f"{prefix}_mean_complete"] = float(
            complete.mean()
        )
        summary[f"{prefix}_sd_complete"] = float(
            complete.std(ddof=1)
        )
        for code, name in MISSING_REASON_REGISTRY.items():
            if code == 0:
                continue
            summary[
                f"{prefix}_missing_{name}"
            ] = int(np.sum(reason == code))

    return summary


# ---------------------------------------------------------------------
# Materialization and frozen-lock verification
# ---------------------------------------------------------------------


def materialize_one(
    scenario: Scenario,
    registry_label: str,
    replicate_index: int,
    simulation_seed: int,
) -> dict[str, Any]:
    path = dataset_path(
        scenario,
        replicate_index,
        simulation_seed,
    )

    if path.exists():
        arrays = read_npz_arrays(path)
        validation = validate_dataset_arrays(
            arrays,
            scenario,
            registry_label,
            replicate_index,
            simulation_seed,
        )
        status = "verified_existing"
    else:
        arrays = generate_dataset(
            scenario,
            registry_label,
            replicate_index,
            simulation_seed,
        )
        validation = validate_dataset_arrays(
            arrays,
            scenario,
            registry_label,
            replicate_index,
            simulation_seed,
        )
        write_deterministic_npz(path, arrays)
        serialized = read_npz_arrays(path)
        validate_dataset_arrays(
            serialized,
            scenario,
            registry_label,
            replicate_index,
            simulation_seed,
        )
        status = "generated"

    return {
        "registry_label": registry_label,
        "simulation_stage": scenario.simulation_stage,
        "scenario_id": scenario.scenario_id,
        "scenario_origin": scenario.scenario_origin,
        "scenario_family": scenario.scenario_family,
        "calibration_role": scenario.calibration_role,
        "replicate_index": replicate_index,
        "simulation_seed": simulation_seed,
        "n_samples": scenario.n_samples,
        "n_features_rna": scenario.n_features_rna,
        "n_features_methylation": (
            scenario.n_features_methylation
        ),
        "n_features_cnv": scenario.n_features_cnv,
        "n_biological_shared_factors": (
            scenario.n_shared_factors
        ),
        "n_private_factors_per_view": (
            scenario.n_private_factors_per_view
        ),
        "n_technical_shared_factors": (
            scenario.true_technical_shared_factors
        ),
        "shared_factor_correlation": (
            scenario.shared_factor_correlation
        ),
        "shared_support_by_factor": (
            scenario.shared_support_by_factor
        ),
        "loading_regime": scenario.loading_regime,
        "missingness_mode": scenario.missingness_mode,
        "batch_mode": scenario.batch_mode,
        "pair_corruption_fraction": (
            scenario.pair_corruption_fraction
        ),
        "factor_level_identifiable": (
            scenario.factor_level_identifiable
        ),
        "subspace_level_identifiable": (
            scenario.subspace_level_identifiable
        ),
        "primary_truth_target": (
            scenario.primary_truth_target
        ),
        "stability_resamples_per_dataset": (
            scenario.stability_resamples_per_dataset
        ),
        "scenario_sha256": sha256_text(
            canonical_json(
                scenario.canonical_dict()
            )
        ),
        "path": relative_path(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "generation_status": status,
        **validation,
    }


def estimate_dataset_storage_bytes(
    scenario: Scenario,
) -> int:
    """Conservative uncompressed NPZ size estimate for one dataset."""
    n = scenario.n_samples
    k_shared = scenario.n_shared_factors
    k_private = scenario.n_private_factors_per_view
    k_technical = scenario.true_technical_shared_factors

    total = 0

    # Global latent truth and metadata arrays.
    total += n * k_shared * 4
    total += n * k_technical * 4
    total += k_shared * 3
    total += k_technical * 3
    total += n * 4
    total += n * 2

    for view in VIEWS:
        p = scenario.n_features(view)

        # Observed and complete matrices.
        total += 2 * n * p * 4

        # Missing-reason matrix and feature/sample masks.
        total += n * p
        total += p

        # Biological, private and technical loadings.
        total += p * (k_shared + k_private + k_technical) * 4

        # Canonical and observed private scores plus row mapping.
        total += 2 * n * k_private * 4
        total += n * 4

    # NPY/ZIP headers, scalar metadata and a conservative margin.
    return int(total * 1.08 + 128_000)


def estimate_missing_storage(
    scenarios: dict[str, Scenario],
    replicates: pd.DataFrame,
) -> tuple[int, int]:
    estimated_bytes = 0
    missing_datasets = 0

    for _, row in replicates.iterrows():
        scenario = scenarios[clean_text(row["scenario_id"])]
        replicate_index = int(row["replicate_index"])
        simulation_seed = int(row["simulation_seed"])
        path = dataset_path(
            scenario,
            replicate_index,
            simulation_seed,
        )

        if not path.exists():
            missing_datasets += 1
            estimated_bytes += estimate_dataset_storage_bytes(
                scenario
            )

    return estimated_bytes, missing_datasets


def run_reference_fixture() -> dict[str, Any]:
    scenario = Scenario(
        scenario_id="FIXTURE_V2",
        description="Internal deterministic generator fixture.",
        scenario_origin="internal_fixture",
        scenario_family="fixture",
        calibration_role="generator_regression_test",
        simulation_stage="architecture_screening",
        n_samples=12,
        n_shared_factors=2,
        n_private_factors_per_view=1,
        n_features_rna=15,
        n_features_methylation=25,
        n_features_cnv=15,
        shared_signal_strength=1.0,
        private_signal_strength=1.0,
        noise_sd_rna=1.0,
        noise_sd_methylation=1.0,
        noise_sd_cnv=1.0,
        missing_fraction_rna=0.05,
        missing_fraction_methylation=0.10,
        missing_fraction_cnv=0.05,
        pair_corruption_fraction=0.25,
        uninformative_modality="NONE",
        block_scale_rna=1.0,
        block_scale_methylation=2.0,
        block_scale_cnv=1.0,
        nonlinear_mode="none",
        shared_factor_correlation=0.5,
        shared_strength_pattern="equal",
        shared_support_pattern="MIXED",
        shared_support_by_factor="ALL3|RNA_METH",
        private_cross_view_correlation=0.6,
        correlated_private_mode="correlated_private_scores",
        shared_loading_fraction=0.20,
        private_loading_fraction=0.20,
        loading_regime="fixture",
        missingness_mode="batch_dependent",
        missingness_target_view="METHYLATION",
        low_quality_sample_fraction=0.0,
        low_quality_feature_fraction=0.0,
        low_quality_missing_fraction=0.0,
        partial_view_sample_fraction=0.0,
        partial_view_missing_fraction=0.0,
        batch_mode="signal_correlated",
        batch_cardinality=2,
        batch_prevalence=0.5,
        batch_effect_strength=0.75,
        batch_view_support="ALL3",
        batch_signal_correlation=0.5,
        batch_missingness_increment=0.10,
        true_biological_shared_factors=2,
        true_technical_shared_factors=1,
        factor_level_identifiable=True,
        subspace_level_identifiable=True,
        primary_truth_target="factor_and_subspace",
        stability_resamples_per_dataset=5,
        materialization_policy="persist_npz",
        generator_required_version=GENERATOR_VERSION,
        planned_replicates=1,
    )
    validate_scenario(scenario)

    seed = 20260814
    arrays_a = generate_dataset(
        scenario,
        "internal_fixture",
        1,
        seed,
    )
    arrays_b = generate_dataset(
        scenario,
        "internal_fixture",
        1,
        seed,
    )

    if set(arrays_a) != set(arrays_b):
        raise RuntimeError(
            "Reference fixture array-key mismatch."
        )
    for key in arrays_a:
        left = arrays_a[key]
        right = arrays_b[key]
        if left.dtype.kind == "f":
            if not np.array_equal(
                left,
                right,
                equal_nan=True,
            ):
                raise RuntimeError(
                    f"Reference fixture is not deterministic: {key}"
                )
        elif not np.array_equal(left, right):
            raise RuntimeError(
                f"Reference fixture is not deterministic: {key}"
            )

    validate_dataset_arrays(
        arrays_a,
        scenario,
        "internal_fixture",
        1,
        seed,
    )

    OUTPUT_CONTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    path_a = OUTPUT_CONTRACT_DIR / ".fixture_a.npz"
    path_b = OUTPUT_CONTRACT_DIR / ".fixture_b.npz"
    try:
        write_deterministic_npz(path_a, arrays_a)
        write_deterministic_npz(path_b, arrays_b)
        hash_a = sha256_file(path_a)
        hash_b = sha256_file(path_b)
        if hash_a != hash_b:
            raise RuntimeError(
                "Deterministic NPZ serialization self-test failed."
            )
        return {
            "fixture_seed": seed,
            "scenario_sha256": sha256_text(
                canonical_json(
                    scenario.canonical_dict()
                )
            ),
            "npz_sha256": hash_a,
            "npz_size_bytes": int(
                path_a.stat().st_size
            ),
            "array_sha256": {
                key: hashlib.sha256(
                    deterministic_npy_bytes(value)
                ).hexdigest()
                for key, value in sorted(arrays_a.items())
            },
            "status": "PASS",
        }
    finally:
        for path in [path_a, path_b]:
            if path.exists():
                path.unlink()


def verify_existing_lock() -> None:
    require_file(OUTPUT_LOCK)
    require_file(OUTPUT_DATA_MANIFEST)

    lock = json.loads(
        OUTPUT_LOCK.read_text(encoding="utf-8")
    )
    if clean_text(lock.get("status")) != (
        "FROZEN_ARCHITECTURE_SCREENING_INPUTS_BEFORE_MODEL_FITTING"
    ):
        raise RuntimeError(
            "Unexpected 05b v2 lock status."
        )

    verify_generated_files(lock, "05b v2")

    manifest = pd.read_csv(
        OUTPUT_DATA_MANIFEST,
        dtype=str,
        low_memory=False,
    ).fillna("")
    expected_count = int(lock.get("n_datasets", -1))
    if manifest.shape[0] != expected_count:
        raise RuntimeError(
            "05b v2 manifest row-count mismatch."
        )
    if manifest["path"].duplicated().any():
        raise RuntimeError(
            "Duplicate paths in the frozen 05b v2 manifest."
        )

    print(
        "Verifying frozen architecture-screening datasets..."
    )
    failures: list[str] = []
    for index, row in manifest.iterrows():
        path = resolve_project_path(row["path"])
        if not path.exists():
            failures.append(
                f"missing: {relative_path(path)}"
            )
            continue
        expected_size = int(float(row["size_bytes"]))
        if int(path.stat().st_size) != expected_size:
            failures.append(
                f"size mismatch: {relative_path(path)}"
            )
            continue
        if sha256_file(path) != clean_text(row["sha256"]):
            failures.append(
                f"hash mismatch: {relative_path(path)}"
            )

        current = index + 1
        if (
            current == 1
            or current % 50 == 0
            or current == manifest.shape[0]
        ):
            print(
                f" {current:04d}/{manifest.shape[0]:04d} "
                "existence/size/SHA256: PASS"
            )

    if failures:
        raise RuntimeError(
            "Frozen 05b v2 dataset verification failed:\n"
            + "\n".join(
                f"- {item}"
                for item in failures[:100]
            )
        )

    if FINAL_GATE_DATA_ROOT.exists() and any(
        item.is_file()
        for item in FINAL_GATE_DATA_ROOT.rglob("*")
    ):
        raise RuntimeError(
            "Final gate-calibration data were materialized too early."
        )

    print()
    print("Existing confirmatory 05b v2 lock: PASS")
    print(f"Datasets: {manifest.shape[0]}")
    print(
        "Final gate-calibration datasets materialized: NO"
    )


# ---------------------------------------------------------------------
# Specifications and outputs
# ---------------------------------------------------------------------


def build_generator_specification(
    technical: dict[str, Any],
) -> dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": utc_now_iso(),
        "simulation_stage": "architecture_screening",
        "outcomes_generated": False,
        "clinical_labels_generated": False,
        "biological_annotations_loaded": False,
        "models_fitted": False,
        "stability_gate_thresholds_retuned": False,
        "views": list(VIEWS),
        "format": {
            "container": "deterministic NumPy NPZ",
            "compression": "ZIP_STORED",
            "dtype": "float32 for molecular and latent numeric arrays",
            "pickle_allowed": False,
            "fixed_zip_timestamp": "1980-01-01T00:00:00",
            "atomic_replacement": True,
        },
        "generative_equation": (
            "X_view = block_scale_view * "
            "(shared_signal_strength * f_view(Z_bio diag(s) W_bio_view^T) + "
            "private_signal_strength * Z_private_view W_private_view^T + "
            "batch_effect_strength * Z_technical W_technical_view^T + "
            "feature_offset + Gaussian_noise), followed by the frozen row-pairing "
            "corruption and missingness mechanism."
        ),
        "truth_layers": {
            "biological_shared": (
                "Factor-specific ALL3 or pair-view support with exact score, "
                "loading, support-mask and truth-class arrays."
            ),
            "private": (
                "View-specific score and loading arrays; selected null scenarios "
                "may correlate corresponding private scores across views without "
                "creating biological shared truth."
            ),
            "technical": (
                "Known binary-batch label, continuous technical score, exact view "
                "support, loadings and batch/biology correlation."
            ),
            "complete_observed_matrix": (
                "Observed row ordering after any intentional pairing corruption, "
                "before missing-value masking."
            ),
            "missingness": (
                "Per-cell reason codes with separate base MCAR, feature-structured, "
                "sample-quality, partial-view and batch-dependent mechanisms."
            ),
        },
        "factor_identifiability": {
            "factor_level": (
                "Evaluated only where the frozen scenario marks factors identifiable."
            ),
            "subspace_level": (
                "Always preserved for designated rotationally ambiguous scenarios; "
                "principal angles/projection distance are required downstream."
            ),
        },
        "pair_corruption": (
            "RNA remains in canonical order; methylation and CNV receive independent "
            "exact derangements of the frozen row fraction."
        ),
        "materialization_scope": {
            "architecture_screening": "materialized now",
            "final_gate_calibration": (
                "not materialized until architecture selection is complete and frozen"
            ),
            "pre_addendum_v1_pilot": (
                "preserved, disclosed and ineligible for confirmatory results"
            ),
        },
        "technical_metadata_preflight": {
            "strict_usable_variables": 0,
            "cross_view_batch_candidates": 0,
            "source_inventory_rows": technical[
                "source_inventory_rows"
            ],
            "manifest_discovered_source_count": technical[
                "manifest_discovered_source_count"
            ],
            "source_count_discrepancy": technical[
                "source_count_discrepancy"
            ],
            "interpretation": technical[
                "source_count_interpretation"
            ],
            "consequence": (
                "Technical batch truth is formally stress-tested in simulations; "
                "real TARGET factors retain technical-status-unknown unless later "
                "supported by independent evidence."
            ),
        },
        "guardrails": [
            "05a and 05a1 are verified by SHA256 before generation.",
            "The 04d technical preflight is verified by SHA256 before generation.",
            "No representation, survival or biological result is read or fitted.",
            "The disclosed sable_mo_v1 pilot is never modified or reused for confirmatory evaluation.",
            "Only the combined architecture-screening registry is materialized.",
            "The final gate-calibration registry is not materialized by this script.",
            "Every dataset contains exact known biological, private and technical truth.",
            "No stability threshold is tuned or evaluated in this script.",
        ],
    }


def environment_snapshot() -> dict[str, Any]:
    return {
        "created_at_utc": utc_now_iso(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {
            name: package_version(name)
            for name in [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "mofapy2",
                "torch",
            ]
        },
    }


def write_readme(
    n_scenarios: int,
    n_datasets: int,
    total_bytes: int,
    technical: dict[str, Any],
) -> None:
    lines = [
        "Paper 5 confirmatory architecture-screening simulations v2",
        "=" * 76,
        "",
        f"Script version: {SCRIPT_VERSION}",
        f"Generator version: {GENERATOR_VERSION}",
        "",
        "Scientific state",
        "----------------",
        "outcomes generated: false",
        "clinical labels generated: false",
        "representation models fitted: false",
        "stability thresholds retuned: false",
        "",
        "Materialized scope",
        "------------------",
        f"Architecture-screening scenarios: {n_scenarios}",
        f"Architecture-screening datasets: {n_datasets}",
        f"Total bytes: {total_bytes}",
        "Final gate-calibration datasets: not materialized",
        "",
        "Truth retained in every dataset",
        "-------------------------------",
        "- biological shared scores, loadings and view support",
        "- modality-private scores and loadings",
        "- technical batch scores, labels, loadings and view support",
        "- complete matrices before missing-value masking",
        "- intentional patient-pairing row maps",
        "- structured missingness reason codes",
        "- factor-level and subspace-level identifiability flags",
        "",
        "Technical metadata note",
        "-----------------------",
        (
            f"Frozen parsed source-inventory rows: "
            f"{technical['source_inventory_rows']}"
        ),
        (
            f"04d manifest discovery count: "
            f"{technical['manifest_discovered_source_count']}"
        ),
        technical["source_count_interpretation"],
        "",
        "Important",
        "---------",
        (
            "The earlier sable_mo_v1 datasets are a disclosed developmental "
            "pilot and are excluded from all confirmatory comparisons."
        ),
        (
            "The 3,100 final stability-gate calibration datasets remain frozen "
            "as a registry only and are generated after architecture freeze."
        ),
        "",
    ]
    OUTPUT_README.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    print("=" * 88)
    print(
        "Paper 5 - generate confirmatory small-n multi-omics simulations v2"
    )
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Generator version: {GENERATOR_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data root: {OUTPUT_DATA_ROOT}")
    print()

    if OUTPUT_LOCK.exists():
        verify_existing_lock()
        print("=" * 88)
        return

    results_before = find_model_results()
    if results_before:
        raise RuntimeError(
            "05b v2 must run before fitted simulation or TARGET model results. Found:\n"
            + "\n".join(
                f"- {item}"
                for item in results_before
            )
        )

    if FINAL_GATE_DATA_ROOT.exists() and any(
        item.is_file()
        for item in FINAL_GATE_DATA_ROOT.rglob("*")
    ):
        raise RuntimeError(
            "Final gate-calibration files already exist. "
            "They are forbidden before architecture freeze."
        )

    base = verify_05a()
    addendum_bundle = verify_05a1()
    technical = verify_04d()

    print("05a AI method contract: PASS")
    print("05a1 simulation/confound addendum: PASS")
    print("04d technical metadata preflight: PASS")
    print("Fitted model results before 05b v2: NO")
    print("Outcome access: FORBIDDEN")
    print()

    if technical["source_count_discrepancy"]:
        print(
            "Technical metadata source count note: "
            f"inventory={technical['source_inventory_rows']}, "
            f"04d discovery counter={technical['manifest_discovered_source_count']}."
        )
        print(
            "The frozen parsed inventory is used for provenance; "
            "this does not alter the zero-usable-covariate result."
        )
        print()

    scenarios, replicates, truth_taxonomy = (
        load_and_validate_registries(
            addendum_bundle["addendum"]
        )
    )

    print("-" * 88)
    print("Frozen confirmatory materialization scope")
    print("-" * 88)
    print(f"Architecture scenarios: {len(scenarios)}")
    print(f"Architecture datasets: {replicates.shape[0]}")
    print(
        "Final gate-calibration datasets materialized now: 0"
    )
    print(
        "Disclosed sable_mo_v1 pilot eligible for confirmatory use: NO"
    )
    print()

    OUTPUT_CONTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    ARCHITECTURE_DATA_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    in_progress = {
        "script_version": SCRIPT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "started_at_utc": utc_now_iso(),
        "expected_scenarios": len(scenarios),
        "expected_datasets": int(replicates.shape[0]),
        "output_root": relative_path(OUTPUT_DATA_ROOT),
        "model_fitting_performed": False,
        "outcome_access": False,
    }
    write_json(in_progress, OUTPUT_IN_PROGRESS)

    estimated_bytes, missing_datasets = estimate_missing_storage(
        scenarios,
        replicates,
    )
    free_bytes = shutil.disk_usage(
        OUTPUT_DATA_ROOT.parent
    ).free
    required_bytes = int(
        estimated_bytes * 1.25
        + 1024 * 1024 * 1024
    )

    print("Disk preflight:")
    print(f" Datasets not yet present: {missing_datasets}")
    print(f" Estimated additional bytes: {estimated_bytes:,}")
    print(f" Free bytes: {free_bytes:,}")

    if free_bytes < required_bytes:
        raise RuntimeError(
            "Insufficient free disk space for confirmatory v2 "
            "simulation materialization. Required estimate with "
            f"safety headroom: {required_bytes:,} bytes."
        )

    print()

    print("Running deterministic generator reference fixture...")
    fixture = run_reference_fixture()
    write_json(fixture, OUTPUT_FIXTURE_HASHES)
    print("Simulation generator reference fixture: PASS")
    print()

    manifest_rows: list[dict[str, Any]] = []
    total = int(replicates.shape[0])

    for row_number, row in replicates.iterrows():
        scenario_id = clean_text(row["scenario_id"])
        scenario = scenarios[scenario_id]
        registry_label = clean_text(
            row["registry_label"]
        )
        replicate_index = int(row["replicate_index"])
        simulation_seed = int(row["simulation_seed"])

        record = materialize_one(
            scenario,
            registry_label,
            replicate_index,
            simulation_seed,
        )
        manifest_rows.append(record)

        current = row_number + 1
        if (
            current == 1
            or current % 10 == 0
            or current == total
        ):
            print(
                f" {current:04d}/{total:04d} "
                f"{scenario_id} R{replicate_index:03d}: "
                f"{record['generation_status']}"
            )

    manifest = pd.DataFrame(manifest_rows)
    if manifest.shape[0] != total:
        raise RuntimeError(
            "Materialized manifest row-count mismatch."
        )
    if manifest["path"].duplicated().any():
        raise RuntimeError(
            "Duplicate dataset paths were generated."
        )
    if not manifest["generation_status"].isin(
        ["generated", "verified_existing"]
    ).all():
        raise RuntimeError(
            "Unexpected generation status."
        )

    manifest = manifest.sort_values(
        [
            "scenario_id",
            "replicate_index",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    manifest.to_csv(
        OUTPUT_DATA_MANIFEST,
        index=False,
        float_format="%.12g",
    )

    summary = (
        manifest.groupby(
            [
                "scenario_id",
                "scenario_origin",
                "scenario_family",
                "calibration_role",
            ],
            dropna=False,
        )
        .agg(
            n_datasets=("path", "size"),
            n_generated=(
                "generation_status",
                lambda values: int(
                    np.sum(values == "generated")
                ),
            ),
            n_verified_existing=(
                "generation_status",
                lambda values: int(
                    np.sum(values == "verified_existing")
                ),
            ),
            total_size_bytes=("size_bytes", "sum"),
            n_samples=("n_samples", "first"),
            n_biological_shared_factors=(
                "n_biological_shared_factors",
                "first",
            ),
            n_technical_shared_factors=(
                "n_technical_shared_factors",
                "first",
            ),
            mean_rna_missing_fraction=(
                "rna_missing_fraction",
                "mean",
            ),
            mean_methylation_missing_fraction=(
                "methylation_missing_fraction",
                "mean",
            ),
            mean_cnv_missing_fraction=(
                "cnv_missing_fraction",
                "mean",
            ),
        )
        .reset_index()
    )
    summary.to_csv(
        OUTPUT_SCENARIO_SUMMARY,
        index=False,
        float_format="%.12g",
    )

    generator_spec = build_generator_specification(
        technical
    )
    write_json(generator_spec, OUTPUT_GENERATOR_SPEC)
    write_json(environment_snapshot(), OUTPUT_ENVIRONMENT)

    total_bytes = int(manifest["size_bytes"].sum())
    write_readme(
        len(scenarios),
        total,
        total_bytes,
        technical,
    )

    generated_contract_files = [
        OUTPUT_GENERATOR_SPEC,
        OUTPUT_DATA_MANIFEST,
        OUTPUT_SCENARIO_SUMMARY,
        OUTPUT_FIXTURE_HASHES,
        OUTPUT_ENVIRONMENT,
        OUTPUT_README,
    ]

    lock = {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": (
            "FROZEN_ARCHITECTURE_SCREENING_INPUTS_BEFORE_MODEL_FITTING"
        ),
        "scientific_state": {
            "outcomes_generated": False,
            "clinical_labels_generated": False,
            "biological_annotations_loaded": False,
            "simulation_model_fitting_performed": False,
            "target_model_fitting_performed": False,
            "stability_gate_thresholds_retuned": False,
            "architecture_results_viewed": False,
        },
        "scope": {
            "n_architecture_scenarios": len(scenarios),
            "n_datasets": total,
            "total_size_bytes": total_bytes,
            "architecture_data_root": relative_path(
                ARCHITECTURE_DATA_ROOT
            ),
            "final_gate_calibration_materialized": False,
            "final_gate_registry_rows": int(
                pd.read_csv(
                    FINAL_GATE_REPLICATES_FILE,
                    dtype=str,
                ).shape[0]
            ),
            "pre_addendum_v1_pilot_used": False,
        },
        "upstream_locks": {
            "05a_contract": {
                "path": relative_path(CONTRACT_05A),
                "sha256": base["contract_sha256"],
            },
            "05a_lock": {
                "path": relative_path(LOCK_05A),
                "sha256": base["lock_sha256"],
            },
            "05a1_addendum": {
                "path": relative_path(CONTRACT_05A1),
                "sha256": addendum_bundle[
                    "addendum_sha256"
                ],
            },
            "05a1_lock": {
                "path": relative_path(LOCK_05A1),
                "sha256": addendum_bundle[
                    "lock_sha256"
                ],
            },
            "05a1_pilot_disclosure": {
                "path": relative_path(
                    PILOT_DISCLOSURE_FILE
                ),
                "sha256": addendum_bundle[
                    "pilot_disclosure_sha256"
                ],
            },
            "04d_manifest": {
                "path": relative_path(
                    TECHNICAL_MANIFEST_FILE
                ),
                "sha256": technical[
                    "manifest_sha256"
                ],
            },
            "04d_lock": {
                "path": relative_path(
                    TECHNICAL_LOCK_FILE
                ),
                "sha256": technical[
                    "lock_sha256"
                ],
            },
            "04d_source_inventory": {
                "path": relative_path(
                    TECHNICAL_SOURCE_INVENTORY_FILE
                ),
                "sha256": technical[
                    "source_inventory_sha256"
                ],
                "n_rows": technical[
                    "source_inventory_rows"
                ],
                "manifest_discovery_count": technical[
                    "manifest_discovered_source_count"
                ],
                "count_discrepancy_documented": technical[
                    "source_count_discrepancy"
                ],
            },
            "combined_scenarios": {
                "path": relative_path(
                    COMBINED_SCENARIOS_FILE
                ),
                "sha256": sha256_file(
                    COMBINED_SCENARIOS_FILE
                ),
            },
            "combined_replicates": {
                "path": relative_path(
                    COMBINED_REPLICATES_FILE
                ),
                "sha256": sha256_file(
                    COMBINED_REPLICATES_FILE
                ),
            },
            "truth_taxonomy": {
                "path": relative_path(
                    TRUTH_TAXONOMY_FILE
                ),
                "sha256": sha256_file(
                    TRUTH_TAXONOMY_FILE
                ),
                "n_rows": int(truth_taxonomy.shape[0]),
            },
        },
        "data_manifest": {
            "path": relative_path(
                OUTPUT_DATA_MANIFEST
            ),
            "sha256": sha256_file(
                OUTPUT_DATA_MANIFEST
            ),
            "n_rows": int(manifest.shape[0]),
        },
        "generated_files": {
            relative_path(path): {
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
            for path in generated_contract_files
        },
        "rewrite_policy": (
            "This zero-argument script never rewrites a frozen 05b v2 lock. "
            "Any scientifically justified change requires a new versioned "
            "generator and preservation of this complete data manifest."
        ),
        "guardrails": [
            "Only architecture-screening simulations were materialized.",
            "Final stability-gate calibration simulations were not materialized.",
            "The disclosed v1 pilot was not read as confirmatory input.",
            "No model fitting, metric estimation or threshold tuning occurred.",
            "Every dataset is deterministic and hash-locked.",
        ],
        "n_datasets": total,
    }
    write_json(lock, OUTPUT_LOCK)

    if OUTPUT_IN_PROGRESS.exists():
        OUTPUT_IN_PROGRESS.unlink()

    print()
    print("-" * 88)
    print("Confirmatory simulation materialization summary")
    print("-" * 88)
    print(f"Scenarios: {len(scenarios)}")
    print(f"Datasets: {total}")
    print(f"Total size bytes: {total_bytes}")
    print(
        "Final gate-calibration datasets materialized: NO"
    )
    print(
        "Simulation or TARGET models fitted: NO"
    )
    print(
        "Stability thresholds retuned: NO"
    )
    print()
    print("=" * 88)
    print(
        "Small-n multi-omics simulation generation v2: PASS"
    )
    print("=" * 88)
    print(
        f"Manifest: {relative_path(OUTPUT_DATA_MANIFEST)}"
    )
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("=" * 88)


if __name__ == "__main__":
    main()
