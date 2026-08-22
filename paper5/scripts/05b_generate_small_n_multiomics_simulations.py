from __future__ import annotations

import argparse
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
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05b-generate-small-n-multiomics-simulations-v1"
GENERATOR_VERSION = "sable-mo-simulation-generator-v1"
ADDENDUM_VERSION = "stability-gate-operating-characteristics-v1"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

CONTRACT_05A_DIR = PAPER5_ROOT / "method_contract" / "05a"
CONTRACT_05A = CONTRACT_05A_DIR / "ai_method_contract.json"
LOCK_05A = CONTRACT_05A_DIR / "ai_method_contract_lock.json"
CORE_SCENARIOS_FILE = CONTRACT_05A_DIR / "simulation_scenarios.csv"
CORE_REPLICATES_FILE = CONTRACT_05A_DIR / "simulation_replicate_registry.csv"

OUTPUT_CONTRACT_DIR = PAPER5_ROOT / "method_contract" / "05b"
OUTPUT_ARCHIVE_DIR = OUTPUT_CONTRACT_DIR / "archive"
OUTPUT_DATA_ROOT = PAPER5_ROOT / "simulations" / "sable_mo_v1"
CORE_DATA_DIR = OUTPUT_DATA_ROOT / "core"
ADDENDUM_DATA_DIR = OUTPUT_DATA_ROOT / "stability_gate_addendum"
SMOKE_DATA_DIR = OUTPUT_DATA_ROOT / "smoke_test"

OUTPUT_ADDENDUM_SCENARIOS = OUTPUT_CONTRACT_DIR / "stability_gate_calibration_scenarios.csv"
OUTPUT_ADDENDUM_REPLICATES = OUTPUT_CONTRACT_DIR / "stability_gate_calibration_replicates.csv"
OUTPUT_GENERATOR_SPEC = OUTPUT_CONTRACT_DIR / "simulation_generator_specification.json"
OUTPUT_GATE_CONTRACT = OUTPUT_CONTRACT_DIR / "stability_gate_operating_characteristic_contract.json"
OUTPUT_DATA_MANIFEST = OUTPUT_CONTRACT_DIR / "simulation_data_manifest.csv"
OUTPUT_SCENARIO_SUMMARY = OUTPUT_CONTRACT_DIR / "simulation_materialization_summary.csv"
OUTPUT_FIXTURE_HASHES = OUTPUT_CONTRACT_DIR / "simulation_reference_fixture_hashes.json"
OUTPUT_ENVIRONMENT = OUTPUT_CONTRACT_DIR / "simulation_environment_snapshot.json"
OUTPUT_README = OUTPUT_CONTRACT_DIR / "simulation_generation_README.txt"
OUTPUT_LOCK = OUTPUT_CONTRACT_DIR / "simulation_generation_lock.json"
OUTPUT_IN_PROGRESS = OUTPUT_CONTRACT_DIR / ".generation_in_progress.json"

ADDENDUM_MASTER_SEED = 20260813
DEFAULT_REPLICATES_PER_ADDENDUM_SCENARIO = 20
DEFAULT_SHARED_LOADING_FRACTION = 0.10
DEFAULT_PRIVATE_LOADING_FRACTION = 0.10
DEFAULT_FEATURE_OFFSET_SD = 0.35
DEFAULT_DTYPE = np.float32

VIEWS = ("RNA", "METHYLATION", "CNV")
VIEW_TO_PREFIX = {"RNA": "rna", "METHYLATION": "methylation", "CNV": "cnv"}

REQUIRED_CORE_SCENARIO_COLUMNS = {
    "scenario_id", "description", "n_samples", "n_shared_factors",
    "n_private_factors_per_view", "n_features_rna", "n_features_methylation",
    "n_features_cnv", "shared_signal_strength", "private_signal_strength",
    "noise_sd_rna", "noise_sd_methylation", "noise_sd_cnv",
    "missing_fraction_rna", "missing_fraction_methylation", "missing_fraction_cnv",
    "pair_corruption_fraction", "uninformative_modality", "block_scale_rna",
    "block_scale_methylation", "block_scale_cnv", "replicates",
}
REQUIRED_CORE_REPLICATE_COLUMNS = {
    "scenario_id", "scenario_order", "replicate_index", "simulation_seed",
}
DATASET_REQUIRED_ARRAYS = {
    "X_rna", "X_methylation", "X_cnv", "Z_shared",
    "Z_private_rna", "Z_private_methylation", "Z_private_cnv",
    "W_shared_rna", "W_shared_methylation", "W_shared_cnv",
    "W_private_rna", "W_private_methylation", "W_private_cnv",
    "row_source_rna", "row_source_methylation", "row_source_cnv",
    "sample_index", "scenario_id", "replicate_index", "simulation_seed",
    "generator_version",
}


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    description: str
    scenario_family: str
    calibration_role: str
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
    replicates: int

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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na"}:
        return ""
    return text


def normalize_none_token(value: Any) -> str:
    text = str(value).strip().upper()
    if text in {"", "NAN", "NA", "NULL", "NONE"}:
        return "NONE"
    return text


def normalize_nonlinear_mode(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"", "nan", "na", "null", "none"}:
        return "none"
    return text


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_project_path(value: Any) -> Path:
    path = Path(clean_text(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(master_seed: int, *parts: Any) -> int:
    digest = hashlib.sha256()
    digest.update(str(master_seed).encode("utf-8"))
    for part in parts:
        digest.update(b"|")
        digest.update(str(part).encode("utf-8"))
    return int(int.from_bytes(digest.digest()[:8], "big") % (2**32 - 1))


def write_json(value: Any, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def package_version(name: str) -> dict[str, Any]:
    try:
        return {"installed": True, "version": importlib.metadata.version(name)}
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None}


def archive_existing_outputs(reason: str) -> Path:
    OUTPUT_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    target = OUTPUT_ARCHIVE_DIR / utc_stamp()
    target.mkdir(parents=True, exist_ok=False)
    for path in OUTPUT_CONTRACT_DIR.iterdir():
        if path.name == "archive":
            continue
        destination = target / path.name
        if path.is_file():
            shutil.copy2(path, destination)
        elif path.is_dir():
            shutil.copytree(path, destination)
    (target / "rewrite_reason.txt").write_text(reason.strip() + "\n", encoding="utf-8")
    return target


def validate_05a_contract() -> dict[str, Any]:
    for path in [CONTRACT_05A, LOCK_05A, CORE_SCENARIOS_FILE, CORE_REPLICATES_FILE]:
        require_file(path)

    contract = json.loads(CONTRACT_05A.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_05A.read_text(encoding="utf-8"))

    observed_contract_hash = sha256_file(CONTRACT_05A)
    if observed_contract_hash != clean_text(lock.get("contract_sha256")):
        raise RuntimeError("The 05a AI method contract does not match its frozen lock.")

    failures: list[str] = []
    for relative_name, record in lock.get("generated_files", {}).items():
        path = resolve_project_path(relative_name)
        if not path.exists():
            failures.append(f"missing: {relative_name}")
        elif sha256_file(path) != clean_text(record.get("sha256")):
            failures.append(f"hash mismatch: {relative_name}")
    if failures:
        raise RuntimeError(
            "05a generated-file verification failed:\n"
            + "\n".join(f"- {item}" for item in failures)
        )

    state = contract.get("scientific_state", {})
    scope = contract.get("paper_scope", {})
    if bool(state.get("outcome_loaded", True)):
        raise RuntimeError("05a does not confirm outcome_loaded=false.")
    if bool(state.get("model_fitting_performed", True)):
        raise RuntimeError("05a does not confirm model_fitting_performed=false.")
    if not bool(scope.get("outcomes_accessed_only_after_representation_freeze", False)):
        raise RuntimeError("05a does not keep outcomes inaccessible until representation freeze.")

    return {
        "contract": contract,
        "lock": lock,
        "contract_sha256": observed_contract_hash,
        "lock_sha256": sha256_file(LOCK_05A),
    }


def validate_probability(value: Any, name: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], found {number}.")
    return number


def scenario_from_row(
    row: pd.Series,
    scenario_family: str,
    calibration_role: str,
    nonlinear_mode: str,
) -> Scenario:
    scenario = Scenario(
        scenario_id=clean_text(row["scenario_id"]),
        description=clean_text(row["description"]),
        scenario_family=scenario_family,
        calibration_role=calibration_role,
        n_samples=int(row["n_samples"]),
        n_shared_factors=int(row["n_shared_factors"]),
        n_private_factors_per_view=int(row["n_private_factors_per_view"]),
        n_features_rna=int(row["n_features_rna"]),
        n_features_methylation=int(row["n_features_methylation"]),
        n_features_cnv=int(row["n_features_cnv"]),
        shared_signal_strength=float(row["shared_signal_strength"]),
        private_signal_strength=float(row["private_signal_strength"]),
        noise_sd_rna=float(row["noise_sd_rna"]),
        noise_sd_methylation=float(row["noise_sd_methylation"]),
        noise_sd_cnv=float(row["noise_sd_cnv"]),
        missing_fraction_rna=validate_probability(row["missing_fraction_rna"], "missing_fraction_rna"),
        missing_fraction_methylation=validate_probability(
            row["missing_fraction_methylation"], "missing_fraction_methylation"
        ),
        missing_fraction_cnv=validate_probability(row["missing_fraction_cnv"], "missing_fraction_cnv"),
        pair_corruption_fraction=validate_probability(
            row["pair_corruption_fraction"], "pair_corruption_fraction"
        ),
        uninformative_modality=normalize_none_token(row["uninformative_modality"]),
        block_scale_rna=float(row["block_scale_rna"]),
        block_scale_methylation=float(row["block_scale_methylation"]),
        block_scale_cnv=float(row["block_scale_cnv"]),
        nonlinear_mode=normalize_nonlinear_mode(nonlinear_mode),
        replicates=int(row["replicates"]),
    )

    if not scenario.scenario_id:
        raise ValueError("Empty scenario_id.")
    if scenario.n_samples < 10:
        raise ValueError(f"{scenario.scenario_id}: n_samples must be >= 10.")
    if scenario.n_shared_factors < 0 or scenario.n_private_factors_per_view < 0:
        raise ValueError(f"{scenario.scenario_id}: factor counts cannot be negative.")
    if min(scenario.n_features_rna, scenario.n_features_methylation, scenario.n_features_cnv) < 20:
        raise ValueError(f"{scenario.scenario_id}: every view must contain >=20 features.")
    if scenario.shared_signal_strength < 0 or scenario.private_signal_strength < 0:
        raise ValueError(f"{scenario.scenario_id}: signal strengths cannot be negative.")
    if scenario.replicates < 1:
        raise ValueError(f"{scenario.scenario_id}: replicates must be positive.")
    if scenario.uninformative_modality not in {"NONE", *VIEWS}:
        raise ValueError(
            f"{scenario.scenario_id}: invalid uninformative modality "
            f"{scenario.uninformative_modality}."
        )
    if scenario.nonlinear_mode not in {"none", "methylation_tanh", "view_specific_monotonic"}:
        raise ValueError(
            f"{scenario.scenario_id}: invalid nonlinear_mode {scenario.nonlinear_mode}."
        )
    return scenario


def load_core_registries() -> tuple[pd.DataFrame, pd.DataFrame, list[Scenario]]:
    scenarios = pd.read_csv(CORE_SCENARIOS_FILE, low_memory=False)
    replicates = pd.read_csv(CORE_REPLICATES_FILE, low_memory=False)
    missing_s = REQUIRED_CORE_SCENARIO_COLUMNS - set(scenarios.columns)
    missing_r = REQUIRED_CORE_REPLICATE_COLUMNS - set(replicates.columns)
    if missing_s:
        raise ValueError("Core scenarios missing columns: " + ", ".join(sorted(missing_s)))
    if missing_r:
        raise ValueError("Core replicates missing columns: " + ", ".join(sorted(missing_r)))
    if scenarios["scenario_id"].duplicated().any():
        raise RuntimeError("Duplicate core scenario IDs detected.")

    scenario_objects = [
        scenario_from_row(row, "05a_core", "core_method_benchmark", "none")
        for _, row in scenarios.iterrows()
    ]
    expected = {
        (scenario.scenario_id, replicate_index)
        for scenario in scenario_objects
        for replicate_index in range(1, scenario.replicates + 1)
    }
    observed = {
        (clean_text(row["scenario_id"]), int(row["replicate_index"]))
        for _, row in replicates.iterrows()
    }
    if expected != observed:
        raise RuntimeError("05a replicate registry does not match scenario replicate counts.")
    if replicates["simulation_seed"].duplicated().any():
        raise RuntimeError("Duplicate core simulation seeds detected.")
    return scenarios, replicates, scenario_objects


def build_addendum_scenarios() -> pd.DataFrame:
    base = {
        "n_private_factors_per_view": 3,
        "n_features_rna": 500,
        "n_features_methylation": 4_000,
        "n_features_cnv": 500,
        "private_signal_strength": 1.0,
        "noise_sd_rna": 1.0,
        "noise_sd_methylation": 1.0,
        "noise_sd_cnv": 1.0,
        "missing_fraction_rna": 0.0,
        "missing_fraction_methylation": 0.0,
        "missing_fraction_cnv": 0.0,
        "pair_corruption_fraction": 0.0,
        "uninformative_modality": "none",
        "block_scale_rna": 1.0,
        "block_scale_methylation": 1.0,
        "block_scale_cnv": 1.0,
        "replicates": DEFAULT_REPLICATES_PER_ADDENDUM_SCENARIO,
    }

    rows: list[dict[str, Any]] = []
    signal_grid = [
        ("NULL", 0, 0.0, "estimate_null_acceptance"),
        ("S025", 5, 0.25, "estimate_false_negative_rate"),
        ("S075", 5, 0.75, "estimate_false_negative_rate"),
        ("S150", 5, 1.50, "estimate_false_negative_rate"),
    ]
    for n_samples in [50, 75, 100]:
        for label, n_shared, strength, role in signal_grid:
            record = dict(base)
            record.update({
                "scenario_id": f"GATE_N{n_samples}_{label}",
                "description": (
                    "Stability-gate operating-characteristic scenario: "
                    f"n={n_samples}, shared factors={n_shared}, shared signal={strength}."
                ),
                "scenario_family": "stability_gate_calibration",
                "calibration_role": role,
                "n_samples": n_samples,
                "n_shared_factors": n_shared,
                "shared_signal_strength": strength,
                "nonlinear_mode": "none",
            })
            rows.append(record)

    nonlinear_specs = [
        {
            "scenario_id": "NONLINEAR_METH_TANH",
            "description": "Linear RNA/CNV shared structure with monotone nonlinear methylation.",
            "scenario_family": "nonlinear_method_stress",
            "calibration_role": "test_nonlinear_shared_recovery",
            "n_samples": 75,
            "n_shared_factors": 5,
            "shared_signal_strength": 1.0,
            "nonlinear_mode": "methylation_tanh",
        },
        {
            "scenario_id": "NONLINEAR_VIEW_SPECIFIC",
            "description": "Each view expresses the shared component through a different monotone map.",
            "scenario_family": "nonlinear_method_stress",
            "calibration_role": "test_nonlinear_shared_recovery",
            "n_samples": 75,
            "n_shared_factors": 5,
            "shared_signal_strength": 1.0,
            "nonlinear_mode": "view_specific_monotonic",
        },
        {
            "scenario_id": "NONLINEAR_WEAK_SMALL_N",
            "description": (
                "Small-n weak shared signal with view-specific nonlinearity and "
                "ten-percent methylation missingness."
            ),
            "scenario_family": "nonlinear_method_stress",
            "calibration_role": "combined_nonlinear_small_n_stress",
            "n_samples": 50,
            "n_shared_factors": 5,
            "shared_signal_strength": 0.50,
            "nonlinear_mode": "view_specific_monotonic",
            "missing_fraction_methylation": 0.10,
        },
    ]
    for specification in nonlinear_specs:
        record = dict(base)
        record.update(specification)
        rows.append(record)

    table = pd.DataFrame(rows)
    if table["scenario_id"].duplicated().any():
        raise RuntimeError("Duplicate addendum scenario IDs detected.")
    return table


def build_addendum_replicates(scenarios: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario_order, row in scenarios.reset_index(drop=True).iterrows():
        scenario_id = clean_text(row["scenario_id"])
        for replicate_index in range(1, int(row["replicates"]) + 1):
            rows.append({
                "scenario_id": scenario_id,
                "scenario_order": int(scenario_order + 1),
                "replicate_index": replicate_index,
                "simulation_seed": stable_seed(
                    ADDENDUM_MASTER_SEED, scenario_id, replicate_index
                ),
            })
    table = pd.DataFrame(rows)
    if table["simulation_seed"].duplicated().any():
        raise RuntimeError("Duplicate addendum simulation seeds detected.")
    return table


def addendum_scenario_objects(scenarios: pd.DataFrame) -> list[Scenario]:
    return [
        scenario_from_row(
            row,
            clean_text(row["scenario_family"]),
            clean_text(row["calibration_role"]),
            normalize_nonlinear_mode(row["nonlinear_mode"]),
        )
        for _, row in scenarios.iterrows()
    ]

# ---------------------------------------------------------------------
# Generative model
# ---------------------------------------------------------------------


def standardize_score_columns(values: np.ndarray) -> np.ndarray:
    if values.shape[1] == 0:
        return values.astype(DEFAULT_DTYPE, copy=False)
    centered = values - values.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, ddof=1, keepdims=True)
    scale[(~np.isfinite(scale)) | (scale == 0)] = 1.0
    return (centered / scale).astype(DEFAULT_DTYPE, copy=False)


def generate_sparse_loadings(
    rng: np.random.Generator,
    n_features: int,
    n_factors: int,
    support_fraction: float,
) -> np.ndarray:
    loadings = np.zeros((n_features, n_factors), dtype=DEFAULT_DTYPE)
    if n_factors == 0:
        return loadings

    support_size = min(
        n_features,
        max(20, int(round(n_features * support_fraction))),
    )
    for factor_index in range(n_factors):
        support = rng.choice(n_features, size=support_size, replace=False)
        values = rng.normal(0.0, 1.0, size=support_size)
        rms = math.sqrt(float(np.mean(values**2)))
        if not np.isfinite(rms) or rms == 0:
            raise RuntimeError("Degenerate loading draw.")
        loadings[support, factor_index] = (values / rms).astype(DEFAULT_DTYPE)
    return loadings


def apply_shared_nonlinearity(
    shared_component: np.ndarray,
    view: str,
    nonlinear_mode: str,
) -> np.ndarray:
    if nonlinear_mode == "none":
        return shared_component.astype(DEFAULT_DTYPE, copy=False)

    scale = float(np.std(shared_component, ddof=1))
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    normalized = shared_component / scale

    if nonlinear_mode == "methylation_tanh":
        if view == "METHYLATION":
            return np.tanh(normalized).astype(DEFAULT_DTYPE)
        return shared_component.astype(DEFAULT_DTYPE, copy=False)

    if nonlinear_mode == "view_specific_monotonic":
        if view == "RNA":
            return normalized.astype(DEFAULT_DTYPE)
        if view == "METHYLATION":
            return np.tanh(normalized).astype(DEFAULT_DTYPE)
        if view == "CNV":
            return (
                np.sign(normalized)
                * np.sqrt(np.abs(normalized) + 1e-8)
            ).astype(DEFAULT_DTYPE)

    raise ValueError(f"Unsupported nonlinear mode: {nonlinear_mode}")


def deranged_row_map(
    rng: np.random.Generator,
    n_samples: int,
    corruption_fraction: float,
) -> np.ndarray:
    row_map = np.arange(n_samples, dtype=np.int32)
    n_corrupt = int(round(n_samples * corruption_fraction))
    if n_corrupt < 2:
        return row_map

    selected = np.sort(rng.choice(n_samples, size=n_corrupt, replace=False))
    shift = int(rng.integers(1, n_corrupt))
    permuted = np.roll(selected, shift)
    if np.any(permuted == selected):
        raise RuntimeError("Failed to construct a row derangement.")
    row_map[selected] = permuted
    return row_map


def apply_exact_entry_missingness(
    rng: np.random.Generator,
    values: np.ndarray,
    missing_fraction: float,
) -> tuple[np.ndarray, int]:
    output = np.asarray(values, dtype=DEFAULT_DTYPE).copy()
    n_missing = int(round(output.size * missing_fraction))
    if n_missing <= 0:
        return output, 0
    flat_indices = rng.choice(output.size, size=n_missing, replace=False)
    output.reshape(-1)[flat_indices] = np.nan
    return output, n_missing


def generate_dataset(
    scenario: Scenario,
    replicate_index: int,
    simulation_seed: int,
) -> dict[str, np.ndarray]:
    master_rng = np.random.default_rng(simulation_seed)
    n = scenario.n_samples
    k_shared = scenario.n_shared_factors
    k_private = scenario.n_private_factors_per_view

    Z_shared = standardize_score_columns(
        master_rng.normal(size=(n, k_shared)).astype(DEFAULT_DTYPE)
    )

    arrays: dict[str, np.ndarray] = {
        "Z_shared": Z_shared,
        "sample_index": np.arange(n, dtype=np.int32),
        "scenario_id": np.asarray(scenario.scenario_id),
        "replicate_index": np.asarray(replicate_index, dtype=np.int32),
        "simulation_seed": np.asarray(simulation_seed, dtype=np.uint32),
        "generator_version": np.asarray(GENERATOR_VERSION),
    }

    for view_index, view in enumerate(VIEWS):
        prefix = VIEW_TO_PREFIX[view]
        p = scenario.n_features(view)
        view_rng = np.random.default_rng(
            stable_seed(simulation_seed, "view", view_index, view)
        )

        Z_private = standardize_score_columns(
            view_rng.normal(size=(n, k_private)).astype(DEFAULT_DTYPE)
        )
        W_shared = generate_sparse_loadings(
            view_rng,
            n_features=p,
            n_factors=k_shared,
            support_fraction=DEFAULT_SHARED_LOADING_FRACTION,
        )
        W_private = generate_sparse_loadings(
            view_rng,
            n_features=p,
            n_factors=k_private,
            support_fraction=DEFAULT_PRIVATE_LOADING_FRACTION,
        )

        if view == scenario.uninformative_modality:
            W_shared[:] = 0.0
            W_private[:] = 0.0

        if k_shared > 0:
            shared_component = Z_shared @ W_shared.T
            shared_component = apply_shared_nonlinearity(
                shared_component.astype(DEFAULT_DTYPE, copy=False),
                view=view,
                nonlinear_mode=scenario.nonlinear_mode,
            )
        else:
            shared_component = np.zeros((n, p), dtype=DEFAULT_DTYPE)

        if k_private > 0:
            private_component = Z_private @ W_private.T
        else:
            private_component = np.zeros((n, p), dtype=DEFAULT_DTYPE)

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

        complete = (
            scenario.shared_signal_strength * shared_component
            + scenario.private_signal_strength * private_component
            + feature_offset
            + noise
        ).astype(DEFAULT_DTYPE)
        complete *= np.float32(scenario.block_scale(view))

        if view == "RNA":
            row_map = np.arange(n, dtype=np.int32)
        else:
            row_map = deranged_row_map(
                view_rng,
                n_samples=n,
                corruption_fraction=scenario.pair_corruption_fraction,
            )

        observed = complete[row_map, :]
        observed, _ = apply_exact_entry_missingness(
            view_rng,
            observed,
            scenario.missing_fraction(view),
        )

        arrays[f"X_{prefix}"] = observed.astype(DEFAULT_DTYPE, copy=False)
        arrays[f"Z_private_{prefix}"] = Z_private
        arrays[f"W_shared_{prefix}"] = W_shared
        arrays[f"W_private_{prefix}"] = W_private
        arrays[f"row_source_{prefix}"] = row_map

    return arrays


# ---------------------------------------------------------------------
# Deterministic serialization and validation
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
    compressed: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.stem + ".",
        suffix=".tmp.npz",
        dir=str(path.parent),
    )
    os.close(fd)
    temporary = Path(temporary_name)

    compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    compresslevel = 6 if compressed else None

    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=compression,
            compresslevel=compresslevel,
            strict_timestamps=True,
        ) as archive:
            for key in sorted(arrays):
                info = zipfile.ZipInfo(
                    filename=f"{key}.npy",
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                info.compress_type = compression
                info.create_system = 3
                info.external_attr = 0o600 << 16
                archive.writestr(
                    info,
                    deterministic_npy_bytes(arrays[key]),
                    compress_type=compression,
                    compresslevel=compresslevel,
                )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def dataset_filename(
    scenario_id: str,
    replicate_index: int,
    simulation_seed: int,
) -> str:
    return (
        f"{scenario_id}__R{replicate_index:03d}__"
        f"S{simulation_seed:010d}.npz"
    )


def scenario_output_dir(scenario: Scenario) -> Path:
    root = CORE_DATA_DIR if scenario.scenario_family == "05a_core" else ADDENDUM_DATA_DIR
    return root / scenario.scenario_id


def validate_dataset_arrays(
    arrays: dict[str, np.ndarray],
    scenario: Scenario,
    replicate_index: int,
    simulation_seed: int,
) -> dict[str, Any]:
    missing_arrays = DATASET_REQUIRED_ARRAYS - set(arrays)
    if missing_arrays:
        raise RuntimeError(
            f"{scenario.scenario_id}: missing arrays: "
            + ", ".join(sorted(missing_arrays))
        )

    n = scenario.n_samples
    if str(arrays["scenario_id"].item()) != scenario.scenario_id:
        raise RuntimeError("Serialized scenario_id mismatch.")
    if int(arrays["replicate_index"].item()) != replicate_index:
        raise RuntimeError("Serialized replicate_index mismatch.")
    if int(arrays["simulation_seed"].item()) != simulation_seed:
        raise RuntimeError("Serialized simulation_seed mismatch.")
    if str(arrays["generator_version"].item()) != GENERATOR_VERSION:
        raise RuntimeError("Serialized generator_version mismatch.")
    if arrays["Z_shared"].shape != (n, scenario.n_shared_factors):
        raise RuntimeError("Shared-score shape mismatch.")
    if not np.all(np.isfinite(arrays["Z_shared"])):
        raise RuntimeError("Non-finite shared scores detected.")

    summary: dict[str, Any] = {
        "n_samples": n,
        "n_shared_factors": scenario.n_shared_factors,
        "n_private_factors_per_view": scenario.n_private_factors_per_view,
    }

    for view in VIEWS:
        prefix = VIEW_TO_PREFIX[view]
        p = scenario.n_features(view)
        X = arrays[f"X_{prefix}"]
        Z_private = arrays[f"Z_private_{prefix}"]
        W_shared = arrays[f"W_shared_{prefix}"]
        W_private = arrays[f"W_private_{prefix}"]
        row_source = arrays[f"row_source_{prefix}"]

        if X.shape != (n, p) or X.dtype != DEFAULT_DTYPE:
            raise RuntimeError(f"{view}: matrix shape or dtype mismatch.")
        if Z_private.shape != (n, scenario.n_private_factors_per_view):
            raise RuntimeError(f"{view}: private-score shape mismatch.")
        if W_shared.shape != (p, scenario.n_shared_factors):
            raise RuntimeError(f"{view}: shared-loading shape mismatch.")
        if W_private.shape != (p, scenario.n_private_factors_per_view):
            raise RuntimeError(f"{view}: private-loading shape mismatch.")
        if row_source.shape != (n,):
            raise RuntimeError(f"{view}: row-source shape mismatch.")
        if set(row_source.astype(int).tolist()) != set(range(n)):
            raise RuntimeError(f"{view}: row-source is not a permutation.")

        target_missing_count = int(round(X.size * scenario.missing_fraction(view)))
        observed_missing_count = int(np.sum(~np.isfinite(X)))
        if observed_missing_count != target_missing_count:
            raise RuntimeError(
                f"{view}: missing-count mismatch "
                f"{observed_missing_count} vs {target_missing_count}."
            )

        expected_corrupted = 0
        if view != "RNA":
            proposed = int(round(n * scenario.pair_corruption_fraction))
            expected_corrupted = proposed if proposed >= 2 else 0
        observed_corrupted = int(np.sum(row_source != np.arange(n)))
        if observed_corrupted != expected_corrupted:
            raise RuntimeError(
                f"{view}: corrupted-row mismatch "
                f"{observed_corrupted} vs {expected_corrupted}."
            )

        summary[f"{prefix}_missing_fraction"] = observed_missing_count / X.size
        summary[f"{prefix}_corrupted_rows"] = observed_corrupted
        summary[f"{prefix}_finite_fraction"] = float(np.mean(np.isfinite(X)))

    return summary


def materialize_dataset(
    scenario: Scenario,
    replicate_index: int,
    simulation_seed: int,
    compressed: bool,
    force: bool,
    output_override: Path | None = None,
) -> dict[str, Any]:
    output_path = output_override or (
        scenario_output_dir(scenario)
        / dataset_filename(scenario.scenario_id, replicate_index, simulation_seed)
    )

    if output_path.exists() and not force:
        arrays = read_npz_arrays(output_path)
        validation = validate_dataset_arrays(
            arrays, scenario, replicate_index, simulation_seed
        )
        status = "verified_existing"
    else:
        arrays = generate_dataset(scenario, replicate_index, simulation_seed)
        validation = validate_dataset_arrays(
            arrays, scenario, replicate_index, simulation_seed
        )
        write_deterministic_npz(output_path, arrays, compressed=compressed)
        serialized = read_npz_arrays(output_path)
        validate_dataset_arrays(
            serialized, scenario, replicate_index, simulation_seed
        )
        status = "generated"

    return {
        "scenario_id": scenario.scenario_id,
        "scenario_family": scenario.scenario_family,
        "calibration_role": scenario.calibration_role,
        "replicate_index": replicate_index,
        "simulation_seed": simulation_seed,
        "n_samples": scenario.n_samples,
        "n_shared_factors": scenario.n_shared_factors,
        "n_private_factors_per_view": scenario.n_private_factors_per_view,
        "shared_signal_strength": scenario.shared_signal_strength,
        "private_signal_strength": scenario.private_signal_strength,
        "nonlinear_mode": scenario.nonlinear_mode,
        "pair_corruption_fraction": scenario.pair_corruption_fraction,
        "uninformative_modality": scenario.uninformative_modality,
        "path": relative_path(output_path),
        "size_bytes": int(output_path.stat().st_size),
        "sha256": sha256_file(output_path),
        "generation_status": status,
        **validation,
    }

# ---------------------------------------------------------------------
# Prospective addendum contracts
# ---------------------------------------------------------------------


def build_generator_specification() -> dict[str, Any]:
    return {
        "generator_version": GENERATOR_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "outcomes_generated": False,
        "clinical_covariates_generated": False,
        "views": list(VIEWS),
        "latent_model": {
            "shared_scores": (
                "Common standardized Gaussian Z_shared in canonical sample order."
            ),
            "private_scores": (
                "Independent standardized Gaussian score matrix per view."
            ),
            "shared_loadings": (
                "View-specific sparse loadings with 10% support per factor and "
                "unit root-mean-square loading on the support."
            ),
            "private_loadings": (
                "View-specific sparse loadings generated independently from shared loadings."
            ),
            "matrix_equation": (
                "block_scale * (shared_strength * f_view(Z_shared W_shared^T) + "
                "private_strength * Z_private W_private^T + feature_offset + noise)"
            ),
            "feature_offset_sd": DEFAULT_FEATURE_OFFSET_SD,
        },
        "nonlinear_modes": {
            "none": "Identity shared mapping in every view.",
            "methylation_tanh": (
                "Tanh shared mapping in methylation; identity in RNA and CNV."
            ),
            "view_specific_monotonic": (
                "Standardized linear RNA, tanh methylation, signed-square-root CNV."
            ),
        },
        "missingness": {
            "mechanism": "entry-level MCAR with an exact prespecified count",
            "applied_after_pair_corruption": True,
            "represented_as": "NaN in observed matrices",
        },
        "pair_corruption": {
            "RNA_anchor": True,
            "non_anchor_views": ["METHYLATION", "CNV"],
            "mechanism": (
                "Independent deterministic derangement of the prespecified row fraction "
                "in each non-anchor view."
            ),
            "row_source_arrays_saved": True,
        },
        "serialization": {
            "format": "deterministic NumPy NPZ",
            "numeric_dtype": "float32",
            "fixed_ZIP_entry_timestamp": "1980-01-01T00:00:00",
            "sorted_array_names": True,
            "allow_pickle_required": False,
            "atomic_write": True,
            "sha256_each_dataset": True,
        },
        "guardrails": [
            "No outcome or clinical variable is generated.",
            "All seeds are deterministic and registry-derived.",
            "True shared and private scores are retained for simulation-only evaluation.",
            "Null-shared scenarios contain zero true shared factors.",
            "Patient-pair corruption is explicitly recorded by row-source maps.",
            "TARGET-OS outcomes and biological annotations are never accessed.",
        ],
    }


def build_gate_contract(upstream_contract: dict[str, Any]) -> dict[str, Any]:
    thresholds = upstream_contract.get("stability_gate", {}).get("thresholds", {})
    return {
        "contract_version": ADDENDUM_VERSION,
        "created_at_utc": utc_now_iso(),
        "relationship_to_05a": (
            "Prospective simulation addendum created after 05a was frozen and before "
            "any simulation-model or TARGET-OS benchmark results were fitted or viewed."
        ),
        "fixed_05a_stability_thresholds": thresholds,
        "threshold_retuning_from_this_addendum": False,
        "true_factor_match_threshold": {
            "absolute_aligned_factor_correlation": 0.50,
            "purpose": (
                "Classify whether a retained estimated factor corresponds to a known "
                "true shared factor."
            ),
        },
        "operating_characteristics": {
            "true_positive_rate": (
                "True shared factors matched by at least one retained factor divided by "
                "the number of true shared factors."
            ),
            "false_negative_rate": "1 - true_positive_rate.",
            "false_discovery_proportion": (
                "Retained factors unmatched to any true shared factor divided by the "
                "number retained; zero when none are retained."
            ),
            "null_acceptance_probability": (
                "Fraction of n_shared_factors=0 replicates retaining one or more shared factors."
            ),
            "familywise_true_factor_recovery": (
                "Fraction of non-null replicates recovering every true shared factor."
            ),
        },
        "prespecified_reporting": [
            "Operating characteristics stratified by n and shared-signal strength.",
            "Wilson 95% confidence intervals for null acceptance and recovery rates.",
            "False-negative and false-discovery summaries without threshold retuning.",
            "Primary, low-sensitivity, and high-sensitivity 05a gate profiles reported.",
        ],
        "hard_interpretation_rules": [
            (
                "Null acceptance probability >0.10 indicates insufficient specificity "
                "of the primary gate."
            ),
            (
                "Recovery of <80% of true factors in the n=75, shared-signal=1.0 "
                "core scenario indicates inadequate sensitivity."
            ),
            (
                "Gate thresholds may not change after TARGET-OS, biological, or outcome results."
            ),
            (
                "An inadequate gate is reported as a failed method component rather than "
                "silently redefined."
            ),
        ],
        "outcome_access": False,
        "biological_annotation_access": False,
    }


def environment_snapshot() -> dict[str, Any]:
    return {
        "created_at_utc": utc_now_iso(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "torch": package_version("torch"),
        "mofapy2": package_version("mofapy2"),
    }


def verify_locked_generation() -> None:
    require_file(OUTPUT_LOCK)
    lock = json.loads(OUTPUT_LOCK.read_text(encoding="utf-8"))
    failures: list[str] = []

    for relative_name, record in lock.get("contract_files", {}).items():
        path = resolve_project_path(relative_name)
        if not path.exists():
            failures.append(f"missing contract file: {relative_name}")
        elif sha256_file(path) != clean_text(record.get("sha256")):
            failures.append(f"contract hash mismatch: {relative_name}")

    manifest_path = resolve_project_path(lock.get("data_manifest", ""))
    if not manifest_path.exists():
        failures.append("simulation data manifest missing")
    else:
        manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
        for _, row in manifest.iterrows():
            path = resolve_project_path(row["path"])
            if not path.exists():
                failures.append(f"missing dataset: {row['path']}")
            elif sha256_file(path) != clean_text(row["sha256"]):
                failures.append(f"dataset hash mismatch: {row['path']}")

    if failures:
        raise RuntimeError(
            "Simulation lock verification failed:\n"
            + "\n".join(f"- {item}" for item in failures[:50])
        )

    print("=" * 88)
    print("Existing simulation generation lock: PASS")
    print("=" * 88)
    print(f"Datasets: {lock.get('n_datasets')}")
    print(f"Manifest: {lock.get('data_manifest')}")
    print("No files were rewritten.")
    print("=" * 88)


# ---------------------------------------------------------------------
# Run planning
# ---------------------------------------------------------------------


def estimate_dataset_bytes(scenario: Scenario) -> int:
    n = scenario.n_samples
    ks = scenario.n_shared_factors
    kp = scenario.n_private_factors_per_view
    total_float_values = n * ks
    total_int_values = n
    for view in VIEWS:
        p = scenario.n_features(view)
        total_float_values += n * p
        total_float_values += n * kp
        total_float_values += p * ks
        total_float_values += p * kp
        total_int_values += n
    return int(total_float_values * 4 + total_int_values * 4 + 8192)


def build_run_registry(
    core_scenarios: list[Scenario],
    core_replicates: pd.DataFrame,
    addendum_scenarios: list[Scenario],
    addendum_replicates: pd.DataFrame,
    scope: str,
) -> pd.DataFrame:
    scenario_map = {
        scenario.scenario_id: scenario
        for scenario in [*core_scenarios, *addendum_scenarios]
    }
    tables: list[pd.DataFrame] = []
    if scope in {"core", "all"}:
        part = core_replicates.copy()
        part["scenario_family"] = "05a_core"
        tables.append(part)
    if scope in {"addendum", "all"}:
        part = addendum_replicates.copy()
        part["scenario_family"] = part["scenario_id"].map(
            {s.scenario_id: s.scenario_family for s in addendum_scenarios}
        )
        tables.append(part)
    if not tables:
        raise ValueError(f"Unsupported scope: {scope}")

    registry = pd.concat(tables, ignore_index=True)
    registry["estimated_uncompressed_bytes"] = registry["scenario_id"].map(
        lambda scenario_id: estimate_dataset_bytes(scenario_map[scenario_id])
    )
    registry["dataset_path"] = registry.apply(
        lambda row: relative_path(
            scenario_output_dir(scenario_map[row["scenario_id"]])
            / dataset_filename(
                row["scenario_id"],
                int(row["replicate_index"]),
                int(row["simulation_seed"]),
            )
        ),
        axis=1,
    )
    return registry


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------


def run_self_test() -> None:
    row = pd.Series({
        "scenario_id": "SELF_TEST",
        "description": "Small deterministic generator self-test.",
        "n_samples": 24,
        "n_shared_factors": 3,
        "n_private_factors_per_view": 2,
        "n_features_rna": 60,
        "n_features_methylation": 120,
        "n_features_cnv": 60,
        "shared_signal_strength": 1.0,
        "private_signal_strength": 1.0,
        "noise_sd_rna": 1.0,
        "noise_sd_methylation": 1.0,
        "noise_sd_cnv": 1.0,
        "missing_fraction_rna": 0.0,
        "missing_fraction_methylation": 0.10,
        "missing_fraction_cnv": 0.0,
        "pair_corruption_fraction": 0.25,
        "uninformative_modality": "none",
        "block_scale_rna": 1.0,
        "block_scale_methylation": 2.0,
        "block_scale_cnv": 1.0,
        "replicates": 1,
    })
    scenario = scenario_from_row(
        row,
        "self_test",
        "self_test",
        "view_specific_monotonic",
    )
    seed = 1234567
    arrays_a = generate_dataset(scenario, 1, seed)
    arrays_b = generate_dataset(scenario, 1, seed)
    validate_dataset_arrays(arrays_a, scenario, 1, seed)
    validate_dataset_arrays(arrays_b, scenario, 1, seed)

    for key in arrays_a:
        left = arrays_a[key]
        right = arrays_b[key]
        if np.issubdtype(left.dtype, np.floating):
            if not np.array_equal(left, right, equal_nan=True):
                raise RuntimeError(f"Generator determinism failed for {key}.")
        elif not np.array_equal(left, right):
            raise RuntimeError(f"Generator determinism failed for {key}.")

    with tempfile.TemporaryDirectory() as directory:
        path_a = Path(directory) / "a.npz"
        path_b = Path(directory) / "b.npz"
        write_deterministic_npz(path_a, arrays_a, compressed=False)
        write_deterministic_npz(path_b, arrays_b, compressed=False)
        if sha256_file(path_a) != sha256_file(path_b):
            raise RuntimeError("Deterministic serialization self-test failed.")
        validate_dataset_arrays(read_npz_arrays(path_a), scenario, 1, seed)

    print("Simulation generator self-test: PASS")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic small-n multi-omics simulations and freeze "
            "the stability-gate operating-characteristic addendum."
        )
    )
    parser.add_argument(
        "--scope",
        choices=["core", "addendum", "all"],
        default="all",
        help="Dataset families to materialize. Default: all.",
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Use deterministic deflated NPZ files. Slower; random data compress weakly.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate contracts and print the plan without writing data.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing completed simulation-generation lock.",
    )
    parser.add_argument(
        "--force-regenerate-data",
        action="store_true",
        help="Regenerate dataset files even when valid files already exist.",
    )
    parser.add_argument(
        "--rewrite-contract",
        action="store_true",
        help="Archive and intentionally replace existing 05b contract outputs.",
    )
    parser.add_argument(
        "--amendment-reason",
        type=str,
        default="",
        help="Required with --rewrite-contract.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Generate one core and one addendum fixture without freezing 05b outputs.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run an in-memory deterministic generator/serialization test and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.self_test:
        run_self_test()
        return
    if args.verify_only:
        verify_locked_generation()
        return
    if OUTPUT_LOCK.exists() and not args.rewrite_contract and not args.smoke_test:
        verify_locked_generation()
        return
    if args.rewrite_contract and not clean_text(args.amendment_reason):
        raise ValueError("--amendment-reason is required with --rewrite-contract.")

    print("=" * 88)
    print("Paper 5 - generate small-n multi-omics simulations")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Generator version: {GENERATOR_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Requested scope: {args.scope}")
    print(f"Compressed NPZ: {bool(args.compressed)}")
    print()

    upstream = validate_05a_contract()
    print("05a AI method contract verification: PASS")
    print(f"05a contract SHA256: {upstream['contract_sha256']}")

    core_scenario_table, core_replicates, core_scenarios = load_core_registries()
    addendum_scenario_table = build_addendum_scenarios()
    addendum_replicates = build_addendum_replicates(addendum_scenario_table)
    addendum_scenarios = addendum_scenario_objects(addendum_scenario_table)

    core_seed_set = set(core_replicates["simulation_seed"].astype(int))
    addendum_seed_set = set(addendum_replicates["simulation_seed"].astype(int))
    if core_seed_set & addendum_seed_set:
        raise RuntimeError("Core and addendum simulation seeds overlap.")
    if set(core_scenario_table["scenario_id"]) & set(addendum_scenario_table["scenario_id"]):
        raise RuntimeError("Core and addendum scenario IDs overlap.")

    run_registry = build_run_registry(
        core_scenarios,
        core_replicates,
        addendum_scenarios,
        addendum_replicates,
        scope=args.scope,
    )

    if args.smoke_test:
        first_core = run_registry[run_registry["scenario_family"].eq("05a_core")].head(1)
        first_addendum = run_registry[~run_registry["scenario_family"].eq("05a_core")].head(1)
        run_registry = pd.concat([first_core, first_addendum], ignore_index=True)

    estimated_bytes = int(run_registry["estimated_uncompressed_bytes"].sum())
    print()
    print("Simulation plan:")
    print(f" Core scenarios: {len(core_scenarios)}")
    print(f" Addendum scenarios: {len(addendum_scenarios)}")
    print(f" Datasets in requested run: {run_registry.shape[0]}")
    print(
        " Estimated uncompressed numeric payload: "
        f"{estimated_bytes / (1024**3):.2f} GiB"
    )
    print()
    print(
        run_registry.groupby("scenario_family")
        .size()
        .rename("n_datasets")
        .reset_index()
        .to_string(index=False)
    )

    if args.dry_run:
        print()
        print("Dry run complete. No data or contracts were written.")
        return

    if args.rewrite_contract and OUTPUT_CONTRACT_DIR.exists():
        archived = archive_existing_outputs(args.amendment_reason)
        print(f"Archived prior 05b outputs: {archived}")

    OUTPUT_CONTRACT_DIR.mkdir(parents=True, exist_ok=True)
    CORE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ADDENDUM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not args.smoke_test:
        write_json(
            {
                "started_at_utc": utc_now_iso(),
                "scope": args.scope,
                "n_planned_datasets": int(run_registry.shape[0]),
            },
            OUTPUT_IN_PROGRESS,
        )
    else:
        print()
        print("Smoke-test mode: no frozen 05b contract will be written.")

    scenario_map = {
        scenario.scenario_id: scenario
        for scenario in [*core_scenarios, *addendum_scenarios]
    }

    manifest_rows: list[dict[str, Any]] = []
    total = run_registry.shape[0]

    for row_index, row in run_registry.iterrows():
        scenario = scenario_map[clean_text(row["scenario_id"])]
        replicate_index = int(row["replicate_index"])
        simulation_seed = int(row["simulation_seed"])

        output_override = None
        if args.smoke_test:
            output_override = (
                SMOKE_DATA_DIR
                / scenario.scenario_id
                / dataset_filename(scenario.scenario_id, replicate_index, simulation_seed)
            )

        record = materialize_dataset(
            scenario,
            replicate_index=replicate_index,
            simulation_seed=simulation_seed,
            compressed=bool(args.compressed),
            force=bool(args.force_regenerate_data or args.smoke_test),
            output_override=output_override,
        )
        if args.smoke_test:
            record["generation_status"] = "smoke_generated"
        manifest_rows.append(record)

        current = row_index + 1
        if current == 1 or current % 10 == 0 or current == total:
            print(
                f" {current:04d}/{total:04d} "
                f"{scenario.scenario_id} R{replicate_index:03d} "
                f"{record['generation_status']}"
            )

    manifest = pd.DataFrame(manifest_rows)

    if args.smoke_test:
        print()
        print("Simulation smoke test: PASS")
        print(manifest[["scenario_id", "path", "sha256"]].to_string(index=False))
        return

    addendum_scenario_table.to_csv(OUTPUT_ADDENDUM_SCENARIOS, index=False)
    addendum_replicates.to_csv(OUTPUT_ADDENDUM_REPLICATES, index=False)
    write_json(build_generator_specification(), OUTPUT_GENERATOR_SPEC)
    write_json(build_gate_contract(upstream["contract"]), OUTPUT_GATE_CONTRACT)
    write_json(environment_snapshot(), OUTPUT_ENVIRONMENT)
    manifest.to_csv(OUTPUT_DATA_MANIFEST, index=False)

    scenario_summary = (
        manifest.groupby(["scenario_family", "scenario_id"], dropna=False)
        .agg(
            n_datasets=("path", "size"),
            total_size_bytes=("size_bytes", "sum"),
            mean_rna_missing_fraction=("rna_missing_fraction", "mean"),
            mean_methylation_missing_fraction=("methylation_missing_fraction", "mean"),
            mean_cnv_missing_fraction=("cnv_missing_fraction", "mean"),
            mean_methylation_corrupted_rows=("methylation_corrupted_rows", "mean"),
            mean_cnv_corrupted_rows=("cnv_corrupted_rows", "mean"),
        )
        .reset_index()
    )
    scenario_summary.to_csv(OUTPUT_SCENARIO_SUMMARY, index=False)

    fixture_rows = []
    for scenario_id, part in manifest.groupby("scenario_id"):
        fixture = part.sort_values("replicate_index").iloc[0]
        fixture_rows.append({
            "scenario_id": scenario_id,
            "replicate_index": int(fixture["replicate_index"]),
            "simulation_seed": int(fixture["simulation_seed"]),
            "path": fixture["path"],
            "sha256": fixture["sha256"],
        })
    write_json(
        {
            "generator_version": GENERATOR_VERSION,
            "created_at_utc": utc_now_iso(),
            "reference_rule": "Lowest materialized replicate_index per scenario.",
            "fixtures": sorted(fixture_rows, key=lambda item: item["scenario_id"]),
        },
        OUTPUT_FIXTURE_HASHES,
    )

    contract_paths = [
        OUTPUT_ADDENDUM_SCENARIOS,
        OUTPUT_ADDENDUM_REPLICATES,
        OUTPUT_GENERATOR_SPEC,
        OUTPUT_GATE_CONTRACT,
        OUTPUT_DATA_MANIFEST,
        OUTPUT_SCENARIO_SUMMARY,
        OUTPUT_FIXTURE_HASHES,
        OUTPUT_ENVIRONMENT,
    ]

    readme_lines = [
        "Paper 5 small-n multi-omics simulation generation",
        "=" * 72,
        "",
        f"Script version: {SCRIPT_VERSION}",
        f"Generator version: {GENERATOR_VERSION}",
        f"Created UTC: {utc_now_iso()}",
        "",
        "Prospective status",
        "------------------",
        "The frozen 05a AI method contract was verified before generation.",
        (
            "The stability-gate operating-characteristic addendum was created "
            "before any simulation-model results were fitted or inspected."
        ),
        "No outcome or biological annotation was generated or accessed.",
        "",
        "Materialized datasets",
        "---------------------",
        f"Scope: {args.scope}",
        f"Datasets: {manifest.shape[0]}",
        f"Total bytes: {int(manifest['size_bytes'].sum())}",
        "",
        "Stability-gate characterization",
        "--------------------------------",
        (
            "Null-shared and graded-signal scenarios prospectively characterize "
            "null acceptance, false-negative rate, true-factor recovery, and "
            "false shared-factor discovery under fixed 05a thresholds."
        ),
        "The fixed gate is evaluated rather than retuned from these results.",
        "",
        "Nonlinear method stress tests",
        "-----------------------------",
        (
            "Three scenarios test monotone nonlinear expression of the same shared "
            "latent scores across modalities."
        ),
        "",
        "Outputs",
        "-------",
    ]
    readme_lines.extend(relative_path(path) for path in contract_paths)
    readme_lines.append("")
    OUTPUT_README.write_text("\n".join(readme_lines), encoding="utf-8")
    contract_paths.append(OUTPUT_README)

    contract_file_records = {
        relative_path(path): {
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for path in contract_paths
    }

    lock = {
        "lock_version": "paper5-simulation-generation-lock-v1",
        "created_at_utc": utc_now_iso(),
        "script_version": SCRIPT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "scope": args.scope,
        "compressed": bool(args.compressed),
        "upstream_05a_contract": {
            "path": relative_path(CONTRACT_05A),
            "sha256": upstream["contract_sha256"],
        },
        "upstream_05a_lock": {
            "path": relative_path(LOCK_05A),
            "sha256": upstream["lock_sha256"],
        },
        "n_datasets": int(manifest.shape[0]),
        "n_core_datasets": int(manifest["scenario_family"].eq("05a_core").sum()),
        "n_addendum_datasets": int((~manifest["scenario_family"].eq("05a_core")).sum()),
        "total_size_bytes": int(manifest["size_bytes"].sum()),
        "data_manifest": relative_path(OUTPUT_DATA_MANIFEST),
        "data_manifest_sha256": sha256_file(OUTPUT_DATA_MANIFEST),
        "contract_files": contract_file_records,
        "outcome_access": False,
        "model_fitting_performed": False,
        "stability_gate_thresholds_retuned": False,
        "silent_regeneration_forbidden": True,
    }
    write_json(lock, OUTPUT_LOCK)

    if OUTPUT_IN_PROGRESS.exists():
        OUTPUT_IN_PROGRESS.unlink()

    print()
    print("-" * 88)
    print("Simulation materialization summary")
    print("-" * 88)
    print(
        scenario_summary[
            ["scenario_family", "scenario_id", "n_datasets", "total_size_bytes"]
        ].to_string(index=False)
    )
    print()
    print("=" * 88)
    print("Small-n multi-omics simulation generation: PASS")
    print("=" * 88)
    print(f"Datasets: {manifest.shape[0]}")
    print(f"Total size: {int(manifest['size_bytes'].sum()) / (1024**3):.2f} GiB")
    print(f"Manifest: {relative_path(OUTPUT_DATA_MANIFEST)}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("Outcomes generated: NO")
    print("Simulation model results fitted: NO")
    print("Stability-gate thresholds retuned: NO")
    print("=" * 88)


if __name__ == "__main__":
    main()
