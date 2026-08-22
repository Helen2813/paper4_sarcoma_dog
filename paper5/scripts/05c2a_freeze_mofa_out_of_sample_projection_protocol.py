from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = (
    "05c2a-freeze-mofa-out-of-sample-projection-protocol-v1-no-cli"
)
CONTRACT_VERSION = "paper5-mofa-oos-projection-protocol-v1"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# ---------------------------------------------------------------------
# Frozen upstream artifacts
# ---------------------------------------------------------------------

CONTRACT_05A_DIR = PAPER5_ROOT / "method_contract" / "05a"
CONTRACT_05A = CONTRACT_05A_DIR / "ai_method_contract.json"
LOCK_05A = CONTRACT_05A_DIR / "ai_method_contract_lock.json"
MODEL_REGISTRY = CONTRACT_05A_DIR / "model_family_registry.csv"
BASELINE_GRID = CONTRACT_05A_DIR / "baseline_hyperparameter_grid.csv"
METRIC_REGISTRY = CONTRACT_05A_DIR / "metric_registry.csv"
FAILURE_GATES = CONTRACT_05A_DIR / "failure_gate_registry.csv"
SELECTION_POLICY = CONTRACT_05A_DIR / "model_selection_policy.json"

CONTRACT_05A1_DIR = PAPER5_ROOT / "method_contract" / "05a1"
CONTRACT_05A1 = (
    CONTRACT_05A1_DIR
    / "simulation_and_technical_confound_addendum.json"
)
LOCK_05A1 = (
    CONTRACT_05A1_DIR
    / "simulation_and_technical_confound_addendum_lock.json"
)

TECHNICAL_DIR = (
    PAPER5_ROOT
    / "data_manifest"
    / "TARGET_OS_technical_metadata"
)
TECHNICAL_MANIFEST = (
    TECHNICAL_DIR
    / "TARGET_OS_technical_metadata_preflight_manifest.json"
)
TECHNICAL_LOCK = (
    TECHNICAL_DIR
    / "TARGET_OS_technical_metadata_preflight_lock.json"
)

CONTRACT_05B_DIR = PAPER5_ROOT / "method_contract" / "05b_v2"
SIMULATION_LOCK = (
    CONTRACT_05B_DIR
    / "simulation_generation_v2_lock.json"
)
SIMULATION_MANIFEST = (
    CONTRACT_05B_DIR
    / "architecture_screening_data_manifest.csv"
)

CONTRACT_05C1_DIR = PAPER5_ROOT / "method_contract" / "05c1"
LINEAR_RESULT_MANIFEST = (
    CONTRACT_05C1_DIR
    / "linear_baseline_results_manifest.json"
)
LINEAR_LOCK = (
    CONTRACT_05C1_DIR
    / "linear_simulation_baseline_lock.json"
)

LINEAR_RESULTS_DIR = (
    PAPER5_ROOT
    / "results"
    / "simulation_benchmark"
    / "05c1_linear_baselines"
)
LINEAR_SPLIT_REGISTRY = (
    LINEAR_RESULTS_DIR
    / "linear_simulation_split_registry.csv.gz"
)

FINAL_GATE_DATA_ROOT = (
    PAPER5_ROOT
    / "simulations"
    / "sable_mo_v2"
    / "final_gate_calibration"
)

# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05c2a"
OUTPUT_PROTOCOL = (
    OUTPUT_DIR
    / "mofa_out_of_sample_projection_protocol.json"
)
OUTPUT_SEEDS = (
    OUTPUT_DIR
    / "mofa_simulation_fit_seed_registry.csv"
)
OUTPUT_FIXTURE = (
    OUTPUT_DIR
    / "mofa_gaussian_projection_reference_fixture.json"
)
OUTPUT_ENVIRONMENT = (
    OUTPUT_DIR
    / "mofa_projection_environment_snapshot.json"
)
OUTPUT_README = (
    OUTPUT_DIR
    / "mofa_out_of_sample_projection_README.txt"
)
OUTPUT_LOCK = (
    OUTPUT_DIR
    / "mofa_out_of_sample_projection_lock.json"
)

FUTURE_RESULT_PATHS = (
    PAPER5_ROOT / "method_contract" / "05c2b",
    PAPER5_ROOT / "method_contract" / "05d",
    PAPER5_ROOT / "method_contract" / "05e",
    PAPER5_ROOT / "results" / "simulation_benchmark" / "05c2_mofa",
    PAPER5_ROOT / "results" / "simulation_benchmark" / "05d_sable_mo",
    PAPER5_ROOT / "results" / "representation_benchmark",
)

# ---------------------------------------------------------------------
# Frozen MOFA implementation choices
# ---------------------------------------------------------------------

VIEWS = ("RNA", "METHYLATION", "CNV")
VIEW_PREFIX = {
    "RNA": "rna",
    "METHYLATION": "methylation",
    "CNV": "cnv",
}

MOFA_MODEL_IDS = (
    "MOFA_K5",
    "MOFA_K10",
    "MOFA_K15",
    "MOFA_K20",
)
MOFA_RANKS = (5, 10, 15, 20)

PRIMARY_FEATURE_BUDGETS = {
    "RNA": 5_000,
    "METHYLATION": 10_000,
    "CNV": 5_000,
}

MAX_FEATURE_MISSING_FRACTION = 0.20
MIN_FINITE_TRAINING_VALUES = 3
MIN_PRESTANDARDIZATION_VARIANCE = 1e-10
MIN_TRAINING_STD = 1e-8

MOFA_SIMULATION_MASTER_SEED = 20260815

MOFA_MODEL_OPTIONS = {
    "spikeslab_factors": False,
    "spikeslab_weights": True,
    "ard_factors": False,
    "ard_weights": True,
}

MOFA_DATA_OPTIONS = {
    "scale_views": False,
    "scale_groups": False,
    "center_groups": False,
    "use_float32": True,
}

MOFA_TRAIN_OPTIONS = {
    "iter": 1000,
    "convergence_mode": "fast",
    "dropR2": None,
    "nostop": False,
    "quiet": True,
    "gpu_mode": False,
    "weight_views": False,
}

REQUIRED_MOFAPY2_PARAMETERS = {
    "set_data_options": {
        "scale_views",
        "scale_groups",
        "center_groups",
        "use_float32",
    },
    "set_model_options": {
        "factors",
        "spikeslab_factors",
        "spikeslab_weights",
        "ard_factors",
        "ard_weights",
    },
    "set_train_options": {
        "iter",
        "convergence_mode",
        "dropR2",
        "quiet",
        "seed",
        "gpu_mode",
        "weight_views",
    },
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {
        "",
        "nan",
        "none",
        "null",
        "na",
    }:
        return ""
    return text


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return clean_text(value).lower() in {
        "true",
        "1",
        "yes",
    }


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
    if not path.exists() or not path.is_file():
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


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def stable_seed(*parts: Any) -> int:
    payload = "|".join(
        str(part)
        for part in parts
    )
    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).digest()
    return int.from_bytes(
        digest[:4],
        "little",
    )


def write_json_atomic(
    value: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.stem + ".",
        suffix=".tmp.json",
        dir=str(path.parent),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)

    try:
        temporary.write_text(
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        os.replace(
            temporary,
            path,
        )
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json(path: Path) -> dict[str, Any]:
    require_file(path)
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def is_nonempty_path(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.stat().st_size > 0
    return any(
        item.is_file()
        for item in path.rglob("*")
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


def verify_generated_files(
    lock: dict[str, Any],
    label: str,
) -> None:
    failures: list[str] = []

    for relative_name, record in lock.get(
        "generated_files",
        {},
    ).items():
        path = resolve_project_path(
            relative_name
        )

        if not path.exists():
            failures.append(
                f"missing: {relative_name}"
            )
            continue

        expected_size = record.get(
            "size_bytes"
        )
        if expected_size is not None:
            if int(path.stat().st_size) != int(
                expected_size
            ):
                failures.append(
                    f"size mismatch: {relative_name}"
                )
                continue

        expected_hash = clean_text(
            record.get("sha256")
        )
        if (
            expected_hash
            and sha256_file(path) != expected_hash
        ):
            failures.append(
                f"SHA256 mismatch: {relative_name}"
            )

    if failures:
        raise RuntimeError(
            f"{label} generated-file verification failed:\n"
            + "\n".join(
                f"- {item}"
                for item in failures[:100]
            )
        )


# ---------------------------------------------------------------------
# Upstream verification
# ---------------------------------------------------------------------

def verify_05a() -> dict[str, Any]:
    for path in [
        CONTRACT_05A,
        LOCK_05A,
        MODEL_REGISTRY,
        BASELINE_GRID,
        METRIC_REGISTRY,
        FAILURE_GATES,
        SELECTION_POLICY,
    ]:
        require_file(path)

    contract = load_json(CONTRACT_05A)
    lock = load_json(LOCK_05A)

    observed_contract_hash = sha256_file(
        CONTRACT_05A
    )
    expected_contract_hash = clean_text(
        lock.get("contract_sha256")
    )

    if observed_contract_hash != expected_contract_hash:
        raise RuntimeError(
            "The 05a AI method contract does not match its frozen lock."
        )

    verify_generated_files(
        lock,
        "05a",
    )

    state = contract.get(
        "scientific_state",
        {},
    )
    if bool(
        state.get(
            "outcome_loaded",
            True,
        )
    ):
        raise RuntimeError(
            "05a does not confirm outcome_loaded=false."
        )

    model_table = pd.read_csv(
        MODEL_REGISTRY,
        dtype=str,
    ).fillna("")

    observed_mofa_ids = set(
        model_table.loc[
            model_table["model_id"].str.startswith(
                "MOFA_K"
            ),
            "model_id",
        ].astype(str)
    )

    if observed_mofa_ids != set(
        MOFA_MODEL_IDS
    ):
        raise RuntimeError(
            "The frozen 05a MOFA model IDs differ from "
            f"{list(MOFA_MODEL_IDS)}."
        )

    baseline = pd.read_csv(
        BASELINE_GRID,
        dtype=str,
    ).fillna("")

    mofa = baseline[
        baseline["model_family"].eq(
            "MOFA_plus"
        )
    ].copy()

    observed_ranks = sorted(
        pd.to_numeric(
            mofa["latent_rank"],
            errors="raise",
        ).astype(int).tolist()
    )

    if observed_ranks != list(MOFA_RANKS):
        raise RuntimeError(
            f"Expected frozen MOFA ranks {list(MOFA_RANKS)}, "
            f"found {observed_ranks}."
        )

    if not mofa[
        "feature_budget_role"
    ].eq("primary").all():
        raise RuntimeError(
            "05a does not freeze the primary feature budget for all MOFA models."
        )

    if not mofa[
        "native_missingness"
    ].map(parse_bool).all():
        raise RuntimeError(
            "05a does not freeze native missing-value handling for all MOFA models."
        )

    expected_likelihoods = (
        "gaussian;gaussian;gaussian"
    )
    if not mofa[
        "likelihoods"
    ].eq(expected_likelihoods).all():
        raise RuntimeError(
            "05a MOFA likelihoods are not the frozen three-view Gaussian specification."
        )

    preprocessing = contract.get(
        "preprocessing",
        {},
    )
    if clean_text(
        preprocessing.get(
            "MOFA_missingness"
        )
    ) != "native probabilistic handling without imputation":
        raise RuntimeError(
            "05a MOFA missingness rule has changed."
        )

    return {
        "contract": contract,
        "lock": lock,
        "contract_sha256": observed_contract_hash,
    }


def verify_05a1() -> dict[str, Any]:
    require_file(CONTRACT_05A1)
    require_file(LOCK_05A1)

    addendum = load_json(CONTRACT_05A1)
    lock = load_json(LOCK_05A1)

    expected_hash = clean_text(
        lock.get("addendum_sha256")
    )
    observed_hash = sha256_file(
        CONTRACT_05A1
    )

    if observed_hash != expected_hash:
        raise RuntimeError(
            "The 05a1 addendum does not match its frozen lock."
        )

    verify_generated_files(
        lock,
        "05a1",
    )

    state = addendum.get(
        "scientific_state",
        {},
    )
    forbidden = [
        "outcome_loaded",
        "simulation_model_fitting_performed",
        "target_model_fitting_performed",
    ]
    failures = [
        key
        for key in forbidden
        if bool(state.get(key, False))
    ]

    if failures:
        raise RuntimeError(
            "05a1 scientific state is incompatible with 05c2a: "
            + ", ".join(failures)
        )

    return {
        "addendum": addendum,
        "lock": lock,
        "addendum_sha256": observed_hash,
    }


def verify_04d() -> dict[str, Any]:
    require_file(TECHNICAL_MANIFEST)
    require_file(TECHNICAL_LOCK)

    manifest = load_json(
        TECHNICAL_MANIFEST
    )
    lock = load_json(
        TECHNICAL_LOCK
    )

    expected_hash = clean_text(
        lock.get("manifest_sha256")
    )
    observed_hash = sha256_file(
        TECHNICAL_MANIFEST
    )

    if observed_hash != expected_hash:
        raise RuntimeError(
            "The 04d technical metadata manifest does not match its lock."
        )

    state = manifest.get(
        "scientific_state",
        {},
    )
    if bool(
        state.get(
            "representation_model_fitting_performed",
            False,
        )
    ):
        raise RuntimeError(
            "04d reports representation-model fitting."
        )

    return {
        "manifest": manifest,
        "lock": lock,
        "manifest_sha256": observed_hash,
    }


def verify_05b() -> dict[str, Any]:
    require_file(SIMULATION_LOCK)
    require_file(SIMULATION_MANIFEST)

    lock = load_json(
        SIMULATION_LOCK
    )

    if clean_text(
        lock.get("status")
    ) != (
        "FROZEN_ARCHITECTURE_SCREENING_INPUTS_BEFORE_MODEL_FITTING"
    ):
        raise RuntimeError(
            "Unexpected 05b v2 lock status."
        )

    verify_generated_files(
        lock,
        "05b v2",
    )

    locked_manifest = lock.get(
        "data_manifest",
        {},
    )
    expected_hash = clean_text(
        locked_manifest.get("sha256")
    )
    observed_hash = sha256_file(
        SIMULATION_MANIFEST
    )

    if observed_hash != expected_hash:
        raise RuntimeError(
            "The 05b v2 simulation manifest does not match its lock."
        )

    table = pd.read_csv(
        SIMULATION_MANIFEST,
        dtype=str,
        low_memory=False,
    ).fillna("")

    expected_rows = int(
        lock.get(
            "scope",
            {},
        ).get(
            "n_datasets",
            0,
        )
    )

    if table.shape[0] != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} frozen simulation datasets; "
            f"found {table.shape[0]}."
        )

    if table.shape[0] != 720:
        raise RuntimeError(
            "The confirmatory architecture-screening bank is not the frozen 720-dataset set."
        )

    if (
        FINAL_GATE_DATA_ROOT.exists()
        and any(
            item.is_file()
            for item in FINAL_GATE_DATA_ROOT.rglob("*")
        )
    ):
        raise RuntimeError(
            "Final gate-calibration data were materialized before architecture freeze."
        )

    return {
        "lock": lock,
        "manifest": table,
        "manifest_sha256": observed_hash,
    }


def verify_05c1() -> dict[str, Any]:
    for path in [
        LINEAR_RESULT_MANIFEST,
        LINEAR_LOCK,
        LINEAR_SPLIT_REGISTRY,
    ]:
        require_file(path)

    result_manifest = load_json(
        LINEAR_RESULT_MANIFEST
    )
    lock = load_json(
        LINEAR_LOCK
    )

    if clean_text(
        lock.get("status")
    ) != (
        "FROZEN_LINEAR_BASELINE_RESULTS_BEFORE_MOFA_AND_NEURAL_FITTING"
    ):
        raise RuntimeError(
            "Unexpected 05c1 lock status."
        )

    expected_result_hash = clean_text(
        lock.get(
            "result_manifest",
            {},
        ).get(
            "sha256"
        )
    )
    observed_result_hash = sha256_file(
        LINEAR_RESULT_MANIFEST
    )

    if observed_result_hash != expected_result_hash:
        raise RuntimeError(
            "The 05c1 results manifest does not match its frozen lock."
        )

    verify_generated_files(
        lock,
        "05c1",
    )

    state = result_manifest.get(
        "scientific_state",
        {},
    )
    forbidden_true = [
        "outcomes_loaded",
        "clinical_labels_loaded",
        "biological_annotations_loaded",
        "TARGET_model_fitting_performed",
        "MOFA_model_fitting_performed",
        "neural_model_fitting_performed",
        "architecture_selected",
        "stability_gate_thresholds_retuned",
        "final_gate_calibration_data_used",
    ]

    failures = [
        key
        for key in forbidden_true
        if bool(state.get(key, False))
    ]

    if failures:
        raise RuntimeError(
            "05c1 scientific state is incompatible with prospective MOFA fitting: "
            + ", ".join(failures)
        )

    split_hash = sha256_file(
        LINEAR_SPLIT_REGISTRY
    )

    split_table = pd.read_csv(
        LINEAR_SPLIT_REGISTRY,
        dtype=str,
        low_memory=False,
    ).fillna("")

    required = {
        "scenario_id",
        "replicate_index",
        "simulation_seed",
        "fold_index",
        "training_indices",
        "test_indices",
    }
    missing = required - set(
        split_table.columns
    )
    if missing:
        raise RuntimeError(
            "05c1 split registry is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    if split_table.shape[0] != 720 * 5:
        raise RuntimeError(
            "The 05c1 split registry does not contain exactly 720 x 5 held-out folds."
        )

    key_counts = (
        split_table.groupby(
            [
                "scenario_id",
                "replicate_index",
                "simulation_seed",
            ],
            sort=False,
        )
        .size()
    )
    if (
        key_counts.shape[0] != 720
        or not key_counts.eq(5).all()
    ):
        raise RuntimeError(
            "The 05c1 split registry does not contain five folds for every simulation dataset."
        )

    return {
        "result_manifest": result_manifest,
        "lock": lock,
        "result_manifest_sha256": observed_result_hash,
        "split_sha256": split_hash,
        "split_table": split_table,
    }


def validate_future_state() -> None:
    found = [
        relative_path(path)
        for path in FUTURE_RESULT_PATHS
        if is_nonempty_path(path)
    ]

    if found:
        raise RuntimeError(
            "05c2a must be frozen before MOFA/neural/selection result files exist. Found:\n- "
            + "\n- ".join(found)
        )


# ---------------------------------------------------------------------
# Installed mofapy2 API audit
# ---------------------------------------------------------------------

def inspect_mofapy2_api() -> dict[str, Any]:
    package = package_version(
        "mofapy2"
    )
    if not package["installed"]:
        raise RuntimeError(
            "mofapy2 is not installed in the active environment."
        )

    try:
        from mofapy2.run.entry_point import entry_point
    except Exception as exc:
        raise RuntimeError(
            "mofapy2 is installed but its entry_point API could not be imported."
        ) from exc

    method_names = [
        "set_data_options",
        "set_model_options",
        "set_train_options",
    ]

    signatures: dict[str, str] = {}
    parameters: dict[str, list[str]] = {}

    for method_name in method_names:
        method = getattr(
            entry_point,
            method_name,
            None,
        )
        if method is None:
            raise RuntimeError(
                f"Installed mofapy2 has no entry_point.{method_name}."
            )

        signature = inspect.signature(
            method
        )
        signatures[
            method_name
        ] = str(signature)
        parameters[
            method_name
        ] = list(
            signature.parameters
        )

        required = REQUIRED_MOFAPY2_PARAMETERS[
            method_name
        ]
        missing = required - set(
            signature.parameters
        )
        if missing:
            raise RuntimeError(
                f"Installed mofapy2 {method_name} is missing required parameters: "
                + ", ".join(
                    sorted(missing)
                )
            )

    return {
        "package": package,
        "entry_point_signatures": signatures,
        "entry_point_parameters": parameters,
        "required_api_parameters_verified": True,
    }


def environment_snapshot(
    mofapy2_api: dict[str, Any],
) -> dict[str, Any]:
    packages = {
        name: package_version(name)
        for name in [
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
            "mofapy2",
        ]
    }

    return {
        "created_at_utc": utc_now_iso(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": packages,
        "mofapy2_api": mofapy2_api,
        "mofa_dependencies_ready": all(
            packages[name]["installed"]
            for name in [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "mofapy2",
            ]
        ),
    }


# ---------------------------------------------------------------------
# Deterministic Gaussian posterior projection
# ---------------------------------------------------------------------

def gaussian_posterior_project(
    observed_by_view: dict[str, np.ndarray],
    weights_by_view: dict[str, np.ndarray],
    precision_by_view: dict[str, np.ndarray],
    views_used: tuple[str, ...],
) -> np.ndarray:
    first_view = views_used[0]
    n_samples = int(
        observed_by_view[
            first_view
        ].shape[0]
    )
    k = int(
        weights_by_view[
            first_view
        ].shape[1]
    )

    output = np.zeros(
        (n_samples, k),
        dtype=np.float64,
    )
    identity = np.eye(
        k,
        dtype=np.float64,
    )

    for sample_index in range(
        n_samples
    ):
        precision_matrix = identity.copy()
        information_vector = np.zeros(
            k,
            dtype=np.float64,
        )

        for view in views_used:
            x = np.asarray(
                observed_by_view[
                    view
                ][sample_index, :],
                dtype=np.float64,
            )
            w = np.asarray(
                weights_by_view[
                    view
                ],
                dtype=np.float64,
            )
            tau = np.asarray(
                precision_by_view[
                    view
                ],
                dtype=np.float64,
            ).reshape(-1)

            if (
                x.shape[0] != w.shape[0]
                or x.shape[0] != tau.shape[0]
            ):
                raise ValueError(
                    f"{view}: projection dimensionality mismatch."
                )

            observed = (
                np.isfinite(x)
                & np.isfinite(tau)
                & (tau > 0)
                & np.all(
                    np.isfinite(w),
                    axis=1,
                )
            )

            if not observed.any():
                continue

            x_obs = x[
                observed
            ]
            w_obs = w[
                observed,
                :,
            ]
            tau_obs = tau[
                observed
            ]

            weighted_w = (
                tau_obs[:, None]
                * w_obs
            )
            precision_matrix += (
                w_obs.T
                @ weighted_w
            )
            information_vector += (
                w_obs.T
                @ (
                    tau_obs
                    * x_obs
                )
            )

        try:
            output[
                sample_index,
                :,
            ] = np.linalg.solve(
                precision_matrix,
                information_vector,
            )
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "Gaussian posterior projection linear solve failed."
            ) from exc

    return output


def run_projection_reference_fixture() -> dict[str, Any]:
    rng = np.random.default_rng(
        20260815
    )
    n_samples = 7
    k = 4
    feature_counts = {
        "RNA": 9,
        "METHYLATION": 13,
        "CNV": 8,
    }

    z_true = rng.normal(
        size=(
            n_samples,
            k,
        )
    )

    weights: dict[
        str,
        np.ndarray,
    ] = {}
    precision: dict[
        str,
        np.ndarray,
    ] = {}
    observed: dict[
        str,
        np.ndarray,
    ] = {}

    for view in VIEWS:
        d = feature_counts[
            view
        ]
        w = rng.normal(
            scale=0.55,
            size=(
                d,
                k,
            ),
        )
        tau = rng.uniform(
            0.8,
            2.5,
            size=d,
        )
        noise = rng.normal(
            scale=(
                1.0
                / np.sqrt(tau)
            ),
            size=(
                n_samples,
                d,
            ),
        )

        x = (
            z_true @ w.T
            + noise
        )

        weights[
            view
        ] = w
        precision[
            view
        ] = tau
        observed[
            view
        ] = x

    observed[
        "METHYLATION"
    ][1, [0, 3, 7]] = np.nan
    observed[
        "RNA"
    ][4, [2]] = np.nan
    observed[
        "CNV"
    ][5, [1, 5]] = np.nan

    all_view_scores = (
        gaussian_posterior_project(
            observed,
            weights,
            precision,
            VIEWS,
        )
    )

    if all_view_scores.shape != (
        n_samples,
        k,
    ):
        raise RuntimeError(
            "Projection fixture returned an unexpected latent shape."
        )

    if not np.all(
        np.isfinite(
            all_view_scores
        )
    ):
        raise RuntimeError(
            "Projection fixture returned non-finite latent values."
        )

    # Verify the posterior normal equations for one held-out sample.
    sample_index = 1
    a = np.eye(
        k,
        dtype=np.float64,
    )
    b = np.zeros(
        k,
        dtype=np.float64,
    )

    for view in VIEWS:
        x = observed[
            view
        ][sample_index, :]
        w = weights[
            view
        ]
        tau = precision[
            view
        ]
        valid = np.isfinite(
            x
        )

        w_obs = w[
            valid,
            :,
        ]
        tau_obs = tau[
            valid
        ]
        x_obs = x[
            valid
        ]

        a += (
            w_obs.T
            @ (
                tau_obs[:, None]
                * w_obs
            )
        )
        b += (
            w_obs.T
            @ (
                tau_obs
                * x_obs
            )
        )

    residual = (
        a
        @ all_view_scores[
            sample_index,
            :,
        ]
        - b
    )
    normal_equation_residual = float(
        np.linalg.norm(
            residual
        )
    )

    if normal_equation_residual > 1e-9:
        raise RuntimeError(
            "Projection fixture failed its posterior normal-equation check."
        )

    single_view_scores = {
        view: gaussian_posterior_project(
            observed,
            weights,
            precision,
            (view,),
        )
        for view in VIEWS
    }

    omitted_view_mse: dict[
        str,
        float,
    ] = {}
    for omitted_view in VIEWS:
        used = tuple(
            view
            for view in VIEWS
            if view != omitted_view
        )
        z_from_remaining = (
            gaussian_posterior_project(
                observed,
                weights,
                precision,
                used,
            )
        )
        reconstructed = (
            z_from_remaining
            @ weights[
                omitted_view
            ].T
        )
        complete_mask = np.isfinite(
            observed[
                omitted_view
            ]
        )
        difference = (
            reconstructed[
                complete_mask
            ]
            - observed[
                omitted_view
            ][
                complete_mask
            ]
        )
        omitted_view_mse[
            omitted_view
        ] = float(
            np.mean(
                difference ** 2
            )
        )

    # A raw W multiplication is intentionally not the frozen projection.
    concatenated_x = np.concatenate(
        [
            np.nan_to_num(
                observed[
                    view
                ],
                nan=0.0,
            )
            for view in VIEWS
        ],
        axis=1,
    )
    concatenated_w = np.concatenate(
        [
            weights[
                view
            ]
            for view in VIEWS
        ],
        axis=0,
    )
    naive_scores = (
        concatenated_x
        @ concatenated_w
    )
    naive_difference = float(
        np.max(
            np.abs(
                naive_scores
                - all_view_scores
            )
        )
    )

    if naive_difference <= 1e-6:
        raise RuntimeError(
            "Projection fixture unexpectedly made raw X@W equivalent to the frozen posterior projection."
        )

    return {
        "fixture_id": (
            "05c2a-gaussian-oos-projection-v1"
        ),
        "seed": 20260815,
        "n_samples": n_samples,
        "latent_rank": k,
        "feature_counts": feature_counts,
        "normal_equation_residual_norm": (
            normal_equation_residual
        ),
        "all_view_latent_frobenius_norm": float(
            np.linalg.norm(
                all_view_scores
            )
        ),
        "single_view_latent_frobenius_norm": {
            view: float(
                np.linalg.norm(
                    single_view_scores[
                        view
                    ]
                )
            )
            for view in VIEWS
        },
        "missing_view_reconstruction_mse": (
            omitted_view_mse
        ),
        "max_abs_difference_raw_XW_vs_posterior": (
            naive_difference
        ),
        "raw_XW_is_frozen_projection": False,
        "status": "PASS",
    }


# ---------------------------------------------------------------------
# Deterministic MOFA fit seeds
# ---------------------------------------------------------------------

def build_fit_seed_registry(
    simulation_manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, Any]
    ] = []

    manifest = simulation_manifest.copy()
    manifest = manifest.sort_values(
        [
            "scenario_id",
            "replicate_index",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    for _, row in manifest.iterrows():
        scenario_id = clean_text(
            row["scenario_id"]
        )
        replicate_index = int(
            float(
                row[
                    "replicate_index"
                ]
            )
        )
        simulation_seed = int(
            float(
                row[
                    "simulation_seed"
                ]
            )
        )

        for rank in MOFA_RANKS:
            model_id = (
                f"MOFA_K{rank}"
            )

            for fold_index in range(
                1,
                6,
            ):
                rows.append(
                    {
                        "fit_stage": (
                            "heldout_cv"
                        ),
                        "scenario_id": (
                            scenario_id
                        ),
                        "replicate_index": (
                            replicate_index
                        ),
                        "simulation_seed": (
                            simulation_seed
                        ),
                        "model_id": model_id,
                        "latent_rank": rank,
                        "fold_index": (
                            fold_index
                        ),
                        "mofa_seed": stable_seed(
                            MOFA_SIMULATION_MASTER_SEED,
                            "heldout_cv",
                            scenario_id,
                            replicate_index,
                            simulation_seed,
                            rank,
                            fold_index,
                        ),
                    }
                )

            rows.append(
                {
                    "fit_stage": (
                        "full_data_truth_recovery"
                    ),
                    "scenario_id": (
                        scenario_id
                    ),
                    "replicate_index": (
                        replicate_index
                    ),
                    "simulation_seed": (
                        simulation_seed
                    ),
                    "model_id": model_id,
                    "latent_rank": rank,
                    "fold_index": 0,
                    "mofa_seed": stable_seed(
                        MOFA_SIMULATION_MASTER_SEED,
                        "full_data_truth_recovery",
                        scenario_id,
                        replicate_index,
                        simulation_seed,
                        rank,
                    ),
                }
            )

    table = pd.DataFrame(
        rows
    )

    expected = (
        720
        * len(
            MOFA_RANKS
        )
        * 6
    )

    if table.shape[0] != expected:
        raise RuntimeError(
            f"Expected {expected} deterministic MOFA fit seeds, "
            f"found {table.shape[0]}."
        )

    if table[
        [
            "fit_stage",
            "scenario_id",
            "replicate_index",
            "model_id",
            "fold_index",
        ]
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate deterministic MOFA fit keys were generated."
        )

    return table


# ---------------------------------------------------------------------
# Protocol construction
# ---------------------------------------------------------------------

def build_protocol(
    upstream: dict[str, Any],
    environment: dict[str, Any],
    seed_registry_hash: str,
) -> dict[str, Any]:
    return {
        "contract_version": (
            CONTRACT_VERSION
        ),
        "script_version": (
            SCRIPT_VERSION
        ),
        "created_at_utc": (
            utc_now_iso()
        ),
        "scientific_state": {
            "outcomes_loaded": False,
            "clinical_labels_loaded": False,
            "biological_annotations_loaded": False,
            "TARGET_model_fitting_performed": False,
            "MOFA_model_fitting_performed": False,
            "neural_model_fitting_performed": False,
            "architecture_selected": False,
            "stability_gate_thresholds_retuned": False,
            "final_gate_calibration_data_used": False,
        },
        "purpose": (
            "Freeze a leakage-safe out-of-sample projection rule for "
            "Gaussian MOFA+ before the first MOFA architecture-screening fit."
        ),
        "reason_for_separate_protocol": (
            "The frozen Paper 5 benchmark requires held-out reconstruction, "
            "same-patient cross-view retrieval, and missing-view reconstruction. "
            "A raw matrix multiplication of held-out data by fitted MOFA weights "
            "is not assumed to reproduce posterior factor expectations. "
            "The benchmark therefore freezes a model-based Gaussian posterior "
            "projection rule before any MOFA result is generated."
        ),
        "scope": {
            "simulation_stage": (
                "architecture_screening"
            ),
            "n_simulation_datasets": 720,
            "model_ids": list(
                MOFA_MODEL_IDS
            ),
            "latent_ranks": list(
                MOFA_RANKS
            ),
            "heldout_folds_per_dataset": 5,
            "full_data_truth_fit_per_dataset_and_rank": True,
            "planned_mofa_fits": (
                720
                * len(
                    MOFA_RANKS
                )
                * 6
            ),
            "architecture_selection_performed_here": False,
        },
        "paired_comparison_with_05c1": {
            "reuse_exact_05c1_split_registry": True,
            "split_registry": relative_path(
                LINEAR_SPLIT_REGISTRY
            ),
            "split_registry_sha256": upstream[
                "05c1"
            ][
                "split_sha256"
            ],
            "new_random_cv_splits_allowed": False,
            "reason": (
                "Use identical held-out samples for paired linear-versus-MOFA comparisons."
            ),
        },
        "fold_local_preprocessing": {
            "maximum_feature_missing_fraction": (
                MAX_FEATURE_MISSING_FRACTION
            ),
            "minimum_finite_training_values": (
                MIN_FINITE_TRAINING_VALUES
            ),
            "minimum_prestandardization_variance": (
                MIN_PRESTANDARDIZATION_VARIANCE
            ),
            "minimum_training_standard_deviation": (
                MIN_TRAINING_STD
            ),
            "feature_budgets": (
                PRIMARY_FEATURE_BUDGETS
            ),
            "feature_budget_rule": (
                "Within each training fold, retain eligible features and, "
                "only if their count exceeds the frozen primary budget, "
                "rank by training-fold pre-standardization variance descending "
                "with original feature index as deterministic tie-breaker."
            ),
            "post_zscore_variance_ranking_allowed": False,
            "imputation": (
                "none; missing values remain NaN for native MOFA Gaussian handling"
            ),
            "centering": (
                "training-fold finite-value feature mean"
            ),
            "scaling": (
                "training-fold finite-value sample standard deviation"
            ),
            "heldout_data_used_for_preprocessing": False,
            "complete_simulation_truth_used_for_preprocessing": False,
            "mofapy2_internal_centering": False,
            "mofapy2_internal_view_scaling": False,
            "mofapy2_internal_group_scaling": False,
        },
        "mofapy2_configuration": {
            "likelihoods": {
                "RNA": "gaussian",
                "METHYLATION": "gaussian",
                "CNV": "gaussian",
            },
            "single_group": True,
            "data_options": (
                MOFA_DATA_OPTIONS
            ),
            "model_options": (
                MOFA_MODEL_OPTIONS
            ),
            "train_options": (
                MOFA_TRAIN_OPTIONS
            ),
            "factor_dropping_during_training": False,
            "fixed_rank_model_ids_preserved": True,
            "deterministic_seed_registry": relative_path(
                OUTPUT_SEEDS
            ),
            "deterministic_seed_registry_sha256": (
                seed_registry_hash
            ),
            "mofapy2_version": environment[
                "packages"
            ][
                "mofapy2"
            ][
                "version"
            ],
        },
        "out_of_sample_projection": {
            "name": (
                "fixed-parameter Gaussian posterior mean projection"
            ),
            "native_mofapy2_transform_claimed": False,
            "raw_X_times_W_projection_allowed": False,
            "prior": (
                "z ~ N(0, I_K)"
            ),
            "per_sample_equations": {
                "A": (
                    "I_K + sum_m W_m,obs^T diag(tau_m,obs) W_m,obs"
                ),
                "b": (
                    "sum_m W_m,obs^T diag(tau_m,obs) x_m,obs"
                ),
                "z_hat": (
                    "solve(A, b)"
                ),
            },
            "frozen_parameter_estimates": (
                "Use fitted posterior means of MOFA weights W and "
                "expected Gaussian noise precisions tau from the training fit. "
                "No held-out sample may update W, tau, sparsity parameters, "
                "ARD parameters, or preprocessing statistics."
            ),
            "missing_values": (
                "For each held-out sample and view, only finite observed features "
                "contribute to A and b."
            ),
            "all_view_projection": (
                "Use RNA, methylation, and CNV observed entries to infer z_hat."
            ),
            "single_view_projection_for_retrieval": (
                "Infer z_hat independently from one view at a time using the same "
                "frozen W/tau. Because all projections share the fitted factor "
                "coordinate system, no test-set alignment or refitting is allowed."
            ),
            "missing_view_projection": (
                "Exclude the omitted view from A and b, infer z_hat from the remaining "
                "view or views, then reconstruct the omitted standardized view as "
                "z_hat @ W_omitted^T."
            ),
            "reconstruction": (
                "x_hat_m = z_hat @ W_m^T in the fold-local standardized coordinate system"
            ),
            "test_sample_refitting": False,
            "test_sample_variational_updates": False,
        },
        "heldout_metrics": {
            "heldout_reconstruction": (
                "Same normalized reconstruction-error definition used in 05c1, "
                "evaluated against complete known simulation truth."
            ),
            "cross_view_retrieval": (
                "Same true row_source pairing and random-adjusted MRR definitions "
                "used in 05c1, using one-view posterior projections."
            ),
            "missing_view_reconstruction": (
                "Same omitted-view R2 and clipped reconstruction-score definitions "
                "used in 05c1."
            ),
            "modality_balance": (
                "Normalized entropy of per-view MOFA loading energy for each factor."
            ),
            "latent_effective_rank": (
                "Effective rank of held-out all-view posterior factor scores."
            ),
        },
        "full_data_simulation_truth_evaluation": {
            "allowed": True,
            "role": (
                "truth recovery only; never substituted for held-out reconstruction"
            ),
            "shared_subspace_recovery": True,
            "factor_matching": True,
            "private_subspace_recovery": (
                "Evaluate view-specific signal capture using fitted factor scores/loadings "
                "and the frozen truth-metric definitions; preserve descriptive status where "
                "MOFA has no explicit private latent block."
            ),
            "technical_factor_contamination": True,
            "view_support_recovery": True,
            "rotationally_ambiguous_factor_metrics_primary": False,
            "subspace_metrics_primary_under_rotational_ambiguity": True,
        },
        "seed_policy": {
            "master_seed": (
                MOFA_SIMULATION_MASTER_SEED
            ),
            "seed_rule": (
                "SHA256-derived deterministic seed from master seed, fit stage, "
                "scenario ID, replicate index, simulation seed, latent rank, and "
                "fold index where applicable."
            ),
            "one_seed_per_simulation_fit": True,
            "target_seed_stability_evaluation_unchanged": True,
        },
        "failure_policy": {
            "all_frozen_MOFA_ranks_reported": True,
            "nonconverged_or_nonfinite_fit_retained_in_audit": True,
            "silent_omission_forbidden": True,
            "architecture_selected_here": False,
        },
        "prohibited_operations": [
            "Using the disclosed sable_mo_v1 pilot for confirmatory MOFA results.",
            "Accessing final gate-calibration simulations.",
            "Changing MOFA ranks after viewing results.",
            "Using held-out samples to rank, impute, center, or scale features.",
            "Replacing native MOFA missingness with global imputation.",
            "Computing held-out factors as raw X @ W.",
            "Refitting or variationally updating the MOFA model on held-out samples.",
            "Using survival, metastasis, clinical labels, or biological program labels.",
            "Selecting an architecture in 05c2a.",
        ],
        "reference_fixture": relative_path(
            OUTPUT_FIXTURE
        ),
        "upstream_hashes": {
            relative_path(
                CONTRACT_05A
            ): sha256_file(
                CONTRACT_05A
            ),
            relative_path(
                LOCK_05A
            ): sha256_file(
                LOCK_05A
            ),
            relative_path(
                CONTRACT_05A1
            ): sha256_file(
                CONTRACT_05A1
            ),
            relative_path(
                LOCK_05A1
            ): sha256_file(
                LOCK_05A1
            ),
            relative_path(
                TECHNICAL_MANIFEST
            ): sha256_file(
                TECHNICAL_MANIFEST
            ),
            relative_path(
                TECHNICAL_LOCK
            ): sha256_file(
                TECHNICAL_LOCK
            ),
            relative_path(
                SIMULATION_LOCK
            ): sha256_file(
                SIMULATION_LOCK
            ),
            relative_path(
                SIMULATION_MANIFEST
            ): sha256_file(
                SIMULATION_MANIFEST
            ),
            relative_path(
                LINEAR_RESULT_MANIFEST
            ): sha256_file(
                LINEAR_RESULT_MANIFEST
            ),
            relative_path(
                LINEAR_LOCK
            ): sha256_file(
                LINEAR_LOCK
            ),
            relative_path(
                LINEAR_SPLIT_REGISTRY
            ): upstream[
                "05c1"
            ][
                "split_sha256"
            ],
        },
        "next_required_stage": (
            "05c2b_fit_mofa_simulation_baselines.py"
        ),
    }


# ---------------------------------------------------------------------
# Existing-lock verification
# ---------------------------------------------------------------------

def verify_existing_lock() -> None:
    lock = load_json(
        OUTPUT_LOCK
    )

    if clean_text(
        lock.get("script_version")
    ) != SCRIPT_VERSION:
        raise RuntimeError(
            "Existing 05c2a script-version mismatch."
        )

    failures: list[
        str
    ] = []

    for relative_name, record in lock.get(
        "generated_files",
        {},
    ).items():
        path = resolve_project_path(
            relative_name
        )

        if not path.exists():
            failures.append(
                f"missing: {relative_name}"
            )
            continue

        if int(
            path.stat().st_size
        ) != int(
            record[
                "size_bytes"
            ]
        ):
            failures.append(
                f"size mismatch: {relative_name}"
            )
            continue

        if sha256_file(
            path
        ) != clean_text(
            record[
                "sha256"
            ]
        ):
            failures.append(
                f"SHA256 mismatch: {relative_name}"
            )

    if failures:
        raise RuntimeError(
            "Existing 05c2a lock verification failed:\n"
            + "\n".join(
                f"- {failure}"
                for failure in failures
            )
        )

    print("=" * 88)
    print(
        "Existing Paper 5 MOFA out-of-sample projection protocol: PASS"
    )
    print("=" * 88)
    print(
        f"Protocol: {relative_path(OUTPUT_PROTOCOL)}"
    )
    print(
        f"Lock: {relative_path(OUTPUT_LOCK)}"
    )
    print(
        "No files were rewritten."
    )
    print("=" * 88)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 88)
    print(
        "Paper 5 - freeze MOFA out-of-sample projection protocol"
    )
    print("=" * 88)
    print(
        f"Script version: {SCRIPT_VERSION}"
    )
    print(
        f"Project root: {PROJECT_ROOT}"
    )
    print(
        f"Output directory: {OUTPUT_DIR}"
    )
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT_LOCK.exists():
        verify_existing_lock()
        return

    validate_future_state()

    base = verify_05a()
    addendum = verify_05a1()
    technical = verify_04d()
    simulations = verify_05b()
    linear = verify_05c1()

    print(
        "05a AI method contract: PASS"
    )
    print(
        "05a1 simulation/confound addendum: PASS"
    )
    print(
        "04d technical metadata preflight: PASS"
    )
    print(
        "05b v2 confirmatory simulation bank: PASS"
    )
    print(
        "05c1 frozen linear baseline results: PASS"
    )
    print(
        "Fitted MOFA results before 05c2a: NO"
    )
    print(
        "Neural model results before 05c2a: NO"
    )
    print(
        "Final gate-calibration data accessed: NO"
    )
    print(
        "Outcome access: FORBIDDEN"
    )
    print()

    mofapy2_api = inspect_mofapy2_api()
    environment = environment_snapshot(
        mofapy2_api
    )

    if not environment[
        "mofa_dependencies_ready"
    ]:
        raise RuntimeError(
            "The active environment is not ready for MOFA."
        )

    print(
        "Installed mofapy2 API compatibility: PASS"
    )
    print(
        "mofapy2 version: "
        f"{environment['packages']['mofapy2']['version']}"
    )

    fit_seeds = build_fit_seed_registry(
        simulations[
            "manifest"
        ]
    )
    fit_seeds.to_csv(
        OUTPUT_SEEDS,
        index=False,
        lineterminator="\n",
    )
    seed_hash = sha256_file(
        OUTPUT_SEEDS
    )

    print(
        "Deterministic MOFA fit seeds frozen: "
        f"{fit_seeds.shape[0]:,}"
    )
    print(
        "Exact 05c1 held-out split registry reused: YES"
    )
    print()

    print(
        "Running deterministic Gaussian projection reference fixture..."
    )
    fixture = run_projection_reference_fixture()
    write_json_atomic(
        fixture,
        OUTPUT_FIXTURE,
    )
    print(
        "Gaussian out-of-sample projection fixture: PASS"
    )
    print(
        "Raw X @ W treated as MOFA held-out projection: NO"
    )
    print()

    write_json_atomic(
        environment,
        OUTPUT_ENVIRONMENT,
    )

    upstream = {
        "05a": base,
        "05a1": addendum,
        "04d": technical,
        "05b": simulations,
        "05c1": linear,
    }

    protocol = build_protocol(
        upstream,
        environment,
        seed_hash,
    )
    write_json_atomic(
        protocol,
        OUTPUT_PROTOCOL,
    )

    readme_lines = [
        "Paper 5 MOFA out-of-sample projection protocol",
        "=" * 72,
        "",
        f"Script version: {SCRIPT_VERSION}",
        f"Created UTC: {protocol['created_at_utc']}",
        "",
        "Status",
        "------",
        "Frozen before the first MOFA simulation fit.",
        "No architecture was selected.",
        "No outcome, clinical label, or biological program label was accessed.",
        "",
        "Why this stage exists",
        "---------------------",
        (
            "The Paper 5 benchmark requires genuinely held-out reconstruction, "
            "cross-view retrieval, and missing-view reconstruction."
        ),
        (
            "Held-out MOFA factor scores are therefore inferred with a fixed-parameter "
            "Gaussian posterior mean under the training-fitted MOFA model."
        ),
        (
            "Raw multiplication of held-out data by fitted MOFA weights is explicitly "
            "forbidden as the benchmark projection rule."
        ),
        "",
        "Frozen MOFA models",
        "------------------",
        "MOFA_K5",
        "MOFA_K10",
        "MOFA_K15",
        "MOFA_K20",
        "",
        "Preprocessing",
        "-------------",
        "Training-fold eligibility and pre-z-score variance ranking only.",
        "Primary budgets: RNA 5000; methylation 10000; CNV 5000.",
        "No imputation for MOFA; missing observations remain NaN.",
        "Training-fold finite-value centering and scaling are applied externally.",
        "mofapy2 internal centering/scaling is disabled.",
        "",
        "Held-out projection",
        "-------------------",
        "A = I + sum W_obs^T diag(tau_obs) W_obs",
        "b = sum W_obs^T diag(tau_obs) x_obs",
        "z_hat = solve(A, b)",
        "",
        "Paired benchmark rule",
        "---------------------",
        (
            "05c2b must reuse the exact 05c1 five-fold split registry so that "
            "linear-versus-MOFA comparisons use identical held-out samples."
        ),
        "",
        "Next stage",
        "----------",
        "05c2b_fit_mofa_simulation_baselines.py",
        "",
    ]

    OUTPUT_README.write_text(
        "\n".join(
            readme_lines
        ),
        encoding="utf-8",
    )

    generated_paths = [
        OUTPUT_PROTOCOL,
        OUTPUT_SEEDS,
        OUTPUT_FIXTURE,
        OUTPUT_ENVIRONMENT,
        OUTPUT_README,
    ]

    generated_files = {
        relative_path(path): {
            "sha256": sha256_file(
                path
            ),
            "size_bytes": int(
                path.stat().st_size
            ),
        }
        for path in generated_paths
    }

    lock = {
        "contract_version": (
            CONTRACT_VERSION
        ),
        "script_version": (
            SCRIPT_VERSION
        ),
        "created_at_utc": (
            utc_now_iso()
        ),
        "status": (
            "FROZEN_MOFA_OOS_PROJECTION_PROTOCOL_BEFORE_MOFA_FITTING"
        ),
        "scientific_state": (
            protocol[
                "scientific_state"
            ]
        ),
        "protocol": {
            "path": relative_path(
                OUTPUT_PROTOCOL
            ),
            "sha256": sha256_file(
                OUTPUT_PROTOCOL
            ),
        },
        "fit_seed_registry": {
            "path": relative_path(
                OUTPUT_SEEDS
            ),
            "sha256": seed_hash,
            "n_rows": int(
                fit_seeds.shape[0]
            ),
        },
        "paired_split_registry": {
            "path": relative_path(
                LINEAR_SPLIT_REGISTRY
            ),
            "sha256": linear[
                "split_sha256"
            ],
        },
        "generated_files": (
            generated_files
        ),
        "rewrite_policy": (
            "This zero-argument stage never rewrites a frozen protocol. "
            "Any scientifically justified change requires a new versioned stage."
        ),
    }

    write_json_atomic(
        lock,
        OUTPUT_LOCK,
    )

    print("-" * 88)
    print(
        "Frozen MOFA implementation"
    )
    print("-" * 88)
    print(
        f"Model ranks: {list(MOFA_RANKS)}"
    )
    print(
        "Likelihoods: Gaussian / Gaussian / Gaussian"
    )
    print(
        "Native missing-value handling: YES"
    )
    print(
        "Training-fold external centering/scaling: YES"
    )
    print(
        "mofapy2 internal centering/scaling: NO"
    )
    print(
        "ARD weights: YES"
    )
    print(
        "Spike-and-slab weights: YES"
    )
    print(
        "Factor dropping during training: NO"
    )
    print(
        "Exact 05c1 held-out splits reused by 05c2b: YES"
    )
    print(
        "Out-of-sample rule: fixed-parameter Gaussian posterior mean"
    )
    print(
        "Raw X @ W projection: FORBIDDEN"
    )
    print()

    print("=" * 88)
    print(
        "Paper 5 MOFA out-of-sample projection protocol: FROZEN"
    )
    print("=" * 88)
    print(
        f"Protocol: {relative_path(OUTPUT_PROTOCOL)}"
    )
    print(
        f"Seed registry: {relative_path(OUTPUT_SEEDS)}"
    )
    print(
        f"Lock: {relative_path(OUTPUT_LOCK)}"
    )
    print(
        "MOFA simulation model fitting performed: NO"
    )
    print(
        "Architecture selected: NO"
    )
    print(
        "Final gate-calibration data accessed: NO"
    )
    print(
        "Next stage: 05c2b_fit_mofa_simulation_baselines.py"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
