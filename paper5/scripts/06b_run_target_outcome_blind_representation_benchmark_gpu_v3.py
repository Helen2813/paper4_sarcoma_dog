from __future__ import annotations

"""
Paper 5 - TARGET-OS outcome-blind representation benchmark.

IMPORTANT EXECUTION BOUNDARY
----------------------------
This exact source is intentionally self-gating.

Before a valid 06b1 authorization lock exists:
    - ONLY synthetic/self-test execution is permitted.
    - The real TARGET matrix root is not opened, stat'ed, globbed, or read.
    - No TARGET scientific result directory is created.

After 06b1 authorizes the exact SHA-256 of THIS file:
    - the same source may run the frozen outcome-blind TARGET benchmark.
    - no outcome, clinical endpoint, biological annotation, technical-covariate value,
      secondary comparator result, or q10-retention threshold may influence nested
      model selection.
    - q10=.60/.40 is NOT applied during cross-family selection. It is SABLE-MO
      post-selection factor-retention logic for a later full-data/stability stage.

The preauthorization self-test exercises the same fold-local preprocessing,
BLOCK_PCA engine, one-SE selector, MOFA adapter, and neural V0-V4 adapter on
synthetic data only. If any adapter does not match the locally frozen implementation,
the script fails closed before 06b1 and before any TARGET value is touched.
"""

from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Iterable
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
import re
import sys
import time
import traceback

import numpy as np
import pandas as pd

SCRIPT_VERSION = (
    "06b-run-target-outcome-blind-representation-benchmark-gpu-v3-"
    "missing-view-return-schema-recovery-preauthorization-synthetic-gated-no-cli"
)

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# -------------------------------------------------------------------------
# Frozen / prospective sources
# -------------------------------------------------------------------------
DIR_05A = PAPER5_ROOT / "method_contract" / "05a"
CONTRACT_05A = DIR_05A / "ai_method_contract.json"
LOCK_05A = DIR_05A / "ai_method_contract_lock.json"
MODEL_SELECTION_POLICY = DIR_05A / "model_selection_policy.json"
MODEL_FAMILY_REGISTRY = DIR_05A / "model_family_registry.csv"
BASELINE_GRID = DIR_05A / "baseline_hyperparameter_grid.csv"
OUTER_ASSIGNMENTS = DIR_05A / "target_outer_cv_assignments.csv"
INNER_ASSIGNMENTS = DIR_05A / "target_inner_cv_assignments.csv"
NEURAL_ASSIGNMENTS = DIR_05A / "target_neural_training_assignments.csv"
FAILURE_GATE_REGISTRY = DIR_05A / "failure_gate_registry.csv"

DIR_05C2A = PAPER5_ROOT / "method_contract" / "05c2a"
DIR_05C2B = PAPER5_ROOT / "method_contract" / "05c2b"
DIR_05D1 = PAPER5_ROOT / "method_contract" / "05d1"
DIR_05D3 = PAPER5_ROOT / "method_contract" / "05d3"
DIR_05D4 = PAPER5_ROOT / "method_contract" / "05d4"
DIR_05E = PAPER5_ROOT / "method_contract" / "05e"

SELECTED_PROFILE = DIR_05D3 / "selected_neural_profile.json"
PROFILE_GRID = DIR_05D1 / "neural_architecture_regularization_profile_grid.csv"
LOSS_CONTRACT = DIR_05D1 / "neural_loss_and_architecture_contract.json"
LOCK_05D4 = DIR_05D4 / "neural_confirmatory_ablation_results_lock.json"
MANIFEST_05D4 = DIR_05D4 / "neural_confirmatory_ablation_results_manifest.json"
FREEZE_05E = DIR_05E / "representation_method_freeze.json"
LOCK_05E = DIR_05E / "representation_method_freeze_lock.json"

DIR_06A = PAPER5_ROOT / "method_contract" / "06a"
CONTRACT_06A = DIR_06A / "target_application_contract.json"
LOCK_06A = DIR_06A / "06a_contract_lock.json"

DIR_06B0 = PAPER5_ROOT / "method_contract" / "06b0"
CONTRACT_06B0 = DIR_06B0 / "target_prefit_input_policy_contract.json"
LOCK_06B0 = DIR_06B0 / "06b0_prefit_lock.json"

DIR_06B0A = PAPER5_ROOT / "method_contract" / "06b0a"
AUDIT_06B0A = DIR_06B0A / "target_runner_dependency_audit.json"

DIR_06B1 = PAPER5_ROOT / "method_contract" / "06b1"
AUTH_06B1 = DIR_06B1 / "target_runner_source_authorization_lock.json"

FIXTURE_ROOT = PAPER5_ROOT / "development_fixtures" / "target_input_synthetic_v1"
FIXTURE_LOCK = PAPER5_ROOT / "method_contract" / "05dev0" / "target_input_synthetic_fixture_lock.json"
FIXTURE_CONTRACT = PAPER5_ROOT / "method_contract" / "05dev0" / "target_input_synthetic_fixture_contract.json"

SELFTEST_DIR = PAPER5_ROOT / "method_contract" / "06b_preflight"
SELFTEST_REPORT = SELFTEST_DIR / "target_runner_synthetic_selftest_v3.json"
SELFTEST_FAILURE = SELFTEST_DIR / "target_runner_synthetic_selftest_v3_failure.json"

# Real TARGET paths are names only until authorization.  No preauthorization
# function is permitted to call exists/stat/glob/open/read on TARGET_INPUT_ROOT.
TARGET_INPUT_ROOT = PAPER5_ROOT / "model_inputs" / "TARGET_OS_primary78"
TARGET_CASE_ORDER = TARGET_INPUT_ROOT / "TARGET_OS_primary_multimodal_case_order.csv"
TARGET_PREMODEL = TARGET_INPUT_ROOT / "premodel_qc"

REAL_RESULT_ROOT = PAPER5_ROOT / "results" / "target_representation" / "06b_primary_benchmark"
REAL_SHARD_ROOT = REAL_RESULT_ROOT / "shards"
REAL_CONTRACT_DIR = PAPER5_ROOT / "method_contract" / "06b"

VIEWS = ("RNA", "METHYLATION", "CNV")
FEATURE_BUDGETS = {"RNA": 5000, "METHYLATION": 10000, "CNV": 5000}
NEURAL_PCA_COMPONENTS = {"RNA": 10, "METHYLATION": 10, "CNV": 10}

EXPECTED_ELIGIBLE_MODELS = (
    "BLOCK_PCA",
    "MOFA_K5",
    "MOFA_K10",
    "MOFA_K15",
    "MOFA_K20",
    "V0_BALANCED_AE",
    "V1_SHARED_PRIVATE_AE",
    "V2_CONTRASTIVE_SPAE",
    "V3_DROPOUT_SPAE",
    "V4_SABLE_MO",
)
EXPECTED_COMPLEXITY = {model: i + 1 for i, model in enumerate(EXPECTED_ELIGIBLE_MODELS)}
MOFA_RANK = {"MOFA_K5": 5, "MOFA_K10": 10, "MOFA_K15": 15, "MOFA_K20": 20}
NEURAL_MODELS = (
    "V0_BALANCED_AE",
    "V1_SHARED_PRIVATE_AE",
    "V2_CONTRASTIVE_SPAE",
    "V3_DROPOUT_SPAE",
    "V4_SABLE_MO",
)

PRIMARY_WEIGHTS = {
    "heldout_reconstruction_score": 0.30,
    "random_adjusted_cross_view_mrr": 0.30,
    "missing_view_reconstruction_score": 0.25,
    "modality_balance_score": 0.15,
}

INNER_MODEL_SEEDS = (1103, 2207)
OUTER_MODEL_SEEDS = (1103, 2207, 3301)

AUTH_STATUS = "06B1_TARGET_RUNNER_SOURCE_AUTHORIZED_BEFORE_FIRST_TARGET_FIT"
SELFTEST_STATUS = "06B_TARGET_RUNNER_SYNTHETIC_SELFTEST_PASS_BEFORE_TARGET_ACCESS"

# Development smoke only.  Scientific neural training always uses the frozen
# maximum-epoch / patience settings reconstructed from 05d3/05d1.
SMOKE_MAX_EPOCHS = 3
SMOKE_PATIENCE = 2
SMOKE_SEED = 6061001


# =========================================================================
# Generic helpers
# =========================================================================

def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "na"} else text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Required directory not found: {path}")


def load_json(path: Path) -> dict[str, Any]:
    require_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def import_source(path: Path, module_name: str) -> Any:
    require_file(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def recursive_records(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from recursive_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_records(child)


def status_contains_frozen(payload: dict[str, Any]) -> bool:
    status = clean_text(payload.get("status")).upper()
    return "FROZEN" in status or "PASS" in status or "LOCK" in status


def current_runner_sha() -> str:
    return sha256_file(SCRIPT_PATH)


# =========================================================================
# Hard authorization boundary
# =========================================================================

def authorization_state() -> tuple[bool, dict[str, Any] | None]:
    """
    This function is the ONLY preauthorization interaction with 06b1.

    It does not touch TARGET data.  If 06b1 does not exist, the caller must
    execute synthetic-only mode.
    """
    if not AUTH_06B1.is_file():
        return False, None

    lock = load_json(AUTH_06B1)
    if clean_text(lock.get("status")) != AUTH_STATUS:
        raise RuntimeError("06b1 authorization lock has unexpected status.")

    runner = lock.get("runner_source", {})
    expected_sha = clean_text(runner.get("sha256"))
    if not expected_sha or expected_sha.lower() != current_runner_sha().lower():
        raise RuntimeError(
            "06b1 does not authorize the SHA-256 of the current 06b runner source."
        )
    if not bool(lock.get("real_target_fit_authorized", False)):
        raise RuntimeError("06b1 exists but real_target_fit_authorized is not true.")

    selftest = lock.get("synthetic_selftest", {})
    report_path = PROJECT_ROOT / clean_text(selftest.get("path"))
    expected_report_sha = clean_text(selftest.get("sha256"))
    require_file(report_path)
    if sha256_file(report_path).lower() != expected_report_sha.lower():
        raise RuntimeError("06b1 synthetic self-test hash verification failed.")

    report = load_json(report_path)
    if clean_text(report.get("status")) != SELFTEST_STATUS:
        raise RuntimeError("06b1 points to a self-test that is not PASS.")

    return True, lock


def assert_no_target_access_preauthorization() -> None:
    """
    Deliberately does NOT call exists/stat on TARGET_INPUT_ROOT.

    The protection is architectural: preauthorization main() never invokes any
    function under the "authorized TARGET execution" section.
    """
    if AUTH_06B1.is_file():
        # This does not imply authorization; authorization_state() must still
        # verify the full lock.
        return


# =========================================================================
# Frozen-contract verification
# =========================================================================

def verify_frozen_contracts() -> dict[str, Any]:
    required = [
        CONTRACT_05A,
        LOCK_05A,
        MODEL_SELECTION_POLICY,
        MODEL_FAMILY_REGISTRY,
        BASELINE_GRID,
        OUTER_ASSIGNMENTS,
        INNER_ASSIGNMENTS,
        NEURAL_ASSIGNMENTS,
        SELECTED_PROFILE,
        PROFILE_GRID,
        LOSS_CONTRACT,
        LOCK_05D4,
        MANIFEST_05D4,
        FREEZE_05E,
        LOCK_05E,
        CONTRACT_06A,
        LOCK_06A,
        CONTRACT_06B0,
        LOCK_06B0,
        AUDIT_06B0A,
        FIXTURE_CONTRACT,
        FIXTURE_LOCK,
    ]
    for path in required:
        require_file(path)

    lock_06b0 = load_json(LOCK_06B0)
    if clean_text(lock_06b0.get("status")) != "06B0_TARGET_PREFIT_INPUT_POLICY_LOCK_FROZEN":
        raise RuntimeError("06b0 prefit lock is not in the expected frozen state.")

    audit = load_json(AUDIT_06B0A)
    models = audit.get("nested_selection_eligible_model_rows", [])
    model_ids = tuple(clean_text(row.get("model_id")) for row in models)
    if model_ids != EXPECTED_ELIGIBLE_MODELS:
        raise RuntimeError(
            "Eligible model order differs from the frozen 10-model TARGET registry:\n"
            f"observed={model_ids}"
        )
    for row in models:
        model_id = clean_text(row.get("model_id"))
        if int(row.get("complexity_rank", -1)) != EXPECTED_COMPLEXITY[model_id]:
            raise RuntimeError(f"Complexity rank changed for {model_id}.")
        if clean_text(row.get("eligible_for_nested_selection")).lower() not in {"true", "1"}:
            raise RuntimeError(f"{model_id} is no longer nested-selection eligible.")
        if clean_text(row.get("outcome_access")).lower() not in {"false", "0"}:
            raise RuntimeError(f"{model_id} unexpectedly permits outcome access.")

    outer = pd.read_csv(OUTER_ASSIGNMENTS, dtype=str).fillna("")
    inner = pd.read_csv(INNER_ASSIGNMENTS, dtype=str).fillna("")
    neural = pd.read_csv(NEURAL_ASSIGNMENTS, dtype=str).fillna("")
    if outer.shape != (1950, 7):
        raise RuntimeError(f"Unexpected outer assignment geometry: {outer.shape}")
    if inner.shape != (6240, 8):
        raise RuntimeError(f"Unexpected inner assignment geometry: {inner.shape}")
    if neural.shape != (9750, 9):
        raise RuntimeError(f"Unexpected neural assignment geometry: {neural.shape}")

    policy = load_json(MODEL_SELECTION_POLICY)
    policy_text = json.dumps(policy, sort_keys=True)
    for token in ["one-standard", "one standard", "0.3", "0.25", "0.15"]:
        if token.replace("-", " ") in policy_text.lower().replace("-", " "):
            break

    selected = load_json(SELECTED_PROFILE)
    profile = selected.get("selected_profile", {})
    if clean_text(profile.get("profile_id")) != "ARCH_COMPACT__REG_WEAK":
        raise RuntimeError("Selected neural profile is not ARCH_COMPACT__REG_WEAK.")

    return {
        "audit": audit,
        "outer": outer,
        "inner": inner,
        "neural": neural,
        "policy": policy,
        "selected_profile": selected,
        "source_hashes": {
            relative_path(path): sha256_file(path)
            for path in required
        },
    }


def _collect_named_values(value: Any, predicate) -> list[str]:
    out: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if predicate(str(key)) and isinstance(child, (str, int, float, bool)):
                out.append(str(child))
            out.extend(_collect_named_values(child, predicate))
    elif isinstance(value, list):
        for child in value:
            out.extend(_collect_named_values(child, predicate))
    return out


def _top_level_string_literal(path: Path, variable_name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == variable_name
            ):
                try:
                    value = ast.literal_eval(node.value)
                except Exception:
                    continue
                if isinstance(value, str):
                    return value
    return ""


def resolve_frozen_05d4_source() -> Path:
    """
    Resolve the exact 05d4 implementation used for the completed confirmatory run.

    v1 was intentionally too strict: it only recognized provenance encoded as a
    nested {"path": ..., "sha256": ...} record.  The completed 05d4 artifacts
    may instead store script_sha256/script_version as separate fields.

    Historical execution used the versioned hotfix source:
        05d4_fit_neural_ablation_benchmark_gpu_v1_2_verified.py
    with SCRIPT_VERSION:
        05d4-fit-neural-ablation-benchmark-gpu-v1.2-verified-fixture-hotfix-no-cli

    This v2 recovery does NOT simply trust that filename.  It requires the local
    source's own SCRIPT_VERSION to match exactly and requires the frozen 05d4
    result lock/manifest to certify that source by script hash, script path/name,
    or exact script version.  Otherwise it still fails closed before TARGET access.
    """
    expected_name = "05d4_fit_neural_ablation_benchmark_gpu_v1_2_verified.py"
    expected_version = (
        "05d4-fit-neural-ablation-benchmark-gpu-v1.2-"
        "verified-fixture-hotfix-no-cli"
    )
    candidate = PAPER5_ROOT / "scripts" / expected_name
    require_file(candidate)

    local_version = _top_level_string_literal(candidate, "SCRIPT_VERSION")
    if local_version != expected_version:
        raise RuntimeError(
            "Versioned 05d4 hotfix source has unexpected SCRIPT_VERSION: "
            f"{local_version!r}"
        )

    lock = load_json(LOCK_05D4)
    manifest = load_json(MANIFEST_05D4)
    payload = {"lock": lock, "manifest": manifest}

    candidate_sha = sha256_file(candidate).lower()

    hash_values = [
        v.strip().lower()
        for v in _collect_named_values(
            payload,
            lambda key: (
                "script" in key.lower()
                and ("sha" in key.lower() or "hash" in key.lower())
            ),
        )
        if re.fullmatch(r"[0-9a-fA-F]{64}", v.strip())
    ]

    version_values = [
        v.strip()
        for v in _collect_named_values(
            payload,
            lambda key: "script_version" in key.lower(),
        )
    ]

    path_values = [
        v.replace("\\", "/").strip()
        for v in _collect_named_values(
            payload,
            lambda key: (
                "script_path" in key.lower()
                or "source_path" in key.lower()
                or key.lower() in {"script", "source"}
            ),
        )
    ]

    hash_match = candidate_sha in hash_values
    version_match = expected_version in version_values
    path_match = any(expected_name.lower() in v.lower() for v in path_values)

    # Also retain the older generic nested path+sha evidence if present.
    nested_match = False
    for record in recursive_records(payload):
        path_text = clean_text(record.get("path")).replace("\\", "/")
        sha_text = clean_text(record.get("sha256")).lower()
        if (
            expected_name.lower() in path_text.lower()
            and sha_text == candidate_sha
        ):
            nested_match = True
            break

    if not any([hash_match, version_match, path_match, nested_match]):
        raise RuntimeError(
            "Frozen 05d4 lock/manifest does not certify the historically executed "
            "v1.2 verified hotfix source by hash, version, or path. "
            "Synthetic self-test stops before TARGET access."
        )

    return candidate


def validate_callable(module: Any, name: str, expected_prefix: tuple[str, ...]) -> str:
    if not hasattr(module, name):
        raise RuntimeError(f"Frozen helper is missing callable {name}.")
    obj = getattr(module, name)
    sig = inspect.signature(obj)
    observed = tuple(sig.parameters)
    if observed[: len(expected_prefix)] != expected_prefix:
        raise RuntimeError(
            f"{name} signature changed. Expected prefix {expected_prefix}, observed {observed}."
        )
    return str(sig)


def load_frozen_modules() -> dict[str, Any]:
    path_05c1 = PAPER5_ROOT / "scripts" / "05c1_fit_linear_simulation_baselines.py"
    path_05c2b = PAPER5_ROOT / "scripts" / "05c2b_fit_mofa_simulation_baselines.py"
    path_05d2 = PAPER5_ROOT / "scripts" / "05d2v2_screen_neural_profiles_gpu_syncfree_pathfix.py"
    path_05d4 = resolve_frozen_05d4_source()

    linear = import_source(path_05c1, "paper5_06b_linear_frozen")
    mofa = import_source(path_05c2b, "paper5_06b_mofa_frozen")
    neural = import_source(path_05d2, "paper5_06b_neural_frozen")
    ablation = import_source(path_05d4, "paper5_06b_ablation_frozen")

    signatures = {
        "mofa.fit_mofa": validate_callable(
            mofa,
            "fit_mofa",
            ("training_views", "selected_feature_indices", "rank", "seed", "frozen_protocol", "sample_ids"),
        ),
        "mofa.gaussian_posterior_project": validate_callable(
            mofa,
            "gaussian_posterior_project",
            ("observed_by_view", "weights_by_view", "precision_by_view", "views_used"),
        ),
        "ablation.build_variant_specs": validate_callable(
            ablation, "build_variant_specs", ("loss_contract",)
        ),
        "ablation.train_with_early_stopping": validate_callable(
            ablation,
            "train_with_early_stopping",
            (
                "neural",
                "profile",
                "variant",
                "fit_views_np",
                "fit_availability_np",
                "early_views_np",
                "early_availability_np",
                "fit_seed",
            ),
        ),
        "ablation.train_fixed_epochs": validate_callable(
            ablation,
            "train_fixed_epochs",
            (
                "neural",
                "profile",
                "variant",
                "training_views_np",
                "availability_np",
                "fit_seed",
                "epochs",
            ),
        ),
        "ablation.encode_numpy": validate_callable(
            ablation, "encode_numpy", ("model", "views_np", "availability_np")
        ),
        "ablation.predict_reconstruction_numpy": validate_callable(
            ablation,
            "predict_reconstruction_numpy",
            ("model", "views_np", "availability_np"),
        ),
        "ablation.predict_missing_view_numpy": validate_callable(
            ablation,
            "predict_missing_view_numpy",
            ("model", "views_np", "availability_np", "omitted_view"),
        ),
    }

    if not hasattr(neural, "parse_profile"):
        raise RuntimeError("Frozen neural helper no longer exposes parse_profile().")

    return {
        "linear": linear,
        "mofa": mofa,
        "neural": neural,
        "ablation": ablation,
        "paths": {
            "05c1": path_05c1,
            "05c2b": path_05c2b,
            "05d2v2": path_05d2,
            "05d4": path_05d4,
        },
        "hashes": {
            relative_path(path): sha256_file(path)
            for path in [path_05c1, path_05c2b, path_05d2, path_05d4]
        },
        "signatures": signatures,
    }


# =========================================================================
# Common fold-local preprocessing
# =========================================================================

@dataclass
class ViewTransform:
    selected_indices: np.ndarray
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        x = np.asarray(matrix[:, self.selected_indices], dtype=np.float64)
        missing = ~np.isfinite(x)
        if missing.any():
            x = x.copy()
            rows, cols = np.where(missing)
            x[rows, cols] = self.medians[cols]
        x = (x - self.means) / self.scales
        if not np.isfinite(x).all():
            raise RuntimeError("Fold-local standardized matrix contains non-finite values.")
        return x


@dataclass
class PCAProjection:
    mean: np.ndarray
    components: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) @ self.components.T

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=np.float64) @ self.components + self.mean


def fit_view_transform(
    train_matrix: np.ndarray,
    budget: int,
    maximum_missing_fraction: float = 0.20,
) -> ViewTransform:
    x = np.asarray(train_matrix, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise RuntimeError("Training view has invalid geometry.")

    finite = np.isfinite(x)
    missing_fraction = 1.0 - finite.mean(axis=0)
    eligible = missing_fraction <= maximum_missing_fraction

    variances = np.full(x.shape[1], -np.inf, dtype=np.float64)
    for j in np.flatnonzero(eligible):
        vals = x[finite[:, j], j]
        if vals.size >= 2:
            v = float(np.var(vals, ddof=0))
            if np.isfinite(v) and v > 0.0:
                variances[j] = v

    candidates = np.flatnonzero(np.isfinite(variances) & (variances > 0))
    if candidates.size == 0:
        raise RuntimeError("No positive-variance feature survives fold-local filtering.")

    # Stable descending variance, then ascending feature index.
    order = np.lexsort((candidates, -variances[candidates]))
    selected = candidates[order[: min(int(budget), candidates.size)]]

    selected_x = x[:, selected]
    medians = np.nanmedian(np.where(np.isfinite(selected_x), selected_x, np.nan), axis=0)
    if not np.isfinite(medians).all():
        raise RuntimeError("Fold-local median imputation is undefined for a selected feature.")

    imputed = selected_x.copy()
    rows, cols = np.where(~np.isfinite(imputed))
    if rows.size:
        imputed[rows, cols] = medians[cols]

    means = np.mean(imputed, axis=0)
    scales = np.std(imputed, axis=0, ddof=0)
    keep = np.isfinite(scales) & (scales > 0)
    if not keep.all():
        selected = selected[keep]
        medians = medians[keep]
        means = means[keep]
        scales = scales[keep]
    if selected.size == 0:
        raise RuntimeError("All selected fold-local features became zero-variance.")

    return ViewTransform(
        selected_indices=selected.astype(np.int64),
        medians=medians.astype(np.float64),
        means=means.astype(np.float64),
        scales=scales.astype(np.float64),
    )


def fit_pca(x: np.ndarray, rank: int) -> PCAProjection:
    matrix = np.asarray(x, dtype=np.float64)
    mean = np.mean(matrix, axis=0)
    centered = matrix - mean
    max_rank = min(int(rank), centered.shape[0] - 1, centered.shape[1])
    if max_rank < 1:
        raise RuntimeError("PCA rank is <1 after geometry restriction.")
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return PCAProjection(mean=mean, components=vt[:max_rank].copy())


def fit_preprocessing_bundle(
    matrices: dict[str, np.ndarray],
    train_indices: np.ndarray,
    budgets: dict[str, int],
    pca_rank: dict[str, int] | None = None,
) -> dict[str, Any]:
    train_indices = np.asarray(train_indices, dtype=np.int64)
    transforms: dict[str, ViewTransform] = {}
    pcas: dict[str, PCAProjection] = {}
    standardized_train: dict[str, np.ndarray] = {}

    for view in VIEWS:
        tr = fit_view_transform(matrices[view][train_indices], int(budgets[view]))
        transforms[view] = tr
        train_std = tr.transform(matrices[view][train_indices])
        standardized_train[view] = train_std
        if pca_rank is not None:
            pcas[view] = fit_pca(train_std, int(pca_rank[view]))

    return {
        "transforms": transforms,
        "pcas": pcas,
        "standardized_train": standardized_train,
    }


def transform_bundle(
    preprocessing: dict[str, Any],
    matrices: dict[str, np.ndarray],
    indices: np.ndarray,
    use_pca: bool,
) -> dict[str, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    out: dict[str, np.ndarray] = {}
    for view in VIEWS:
        x = preprocessing["transforms"][view].transform(matrices[view][indices])
        if use_pca:
            x = preprocessing["pcas"][view].transform(x)
        out[view] = x.astype(np.float32, copy=False)
    return out


# =========================================================================
# Observable TARGET utility metrics
# =========================================================================

def entropy_balance(energies: Iterable[float]) -> float:
    x = np.asarray(list(energies), dtype=np.float64)
    x = np.maximum(x, 0.0)
    if x.size <= 1 or float(x.sum()) <= 0.0:
        return 0.0
    p = x / x.sum()
    h = -float(np.sum(np.where(p > 0, p * np.log(p), 0.0)))
    return float(np.clip(h / math.log(x.size), 0.0, 1.0))


def harmonic_random_mrr(n: int) -> float:
    if n <= 0:
        return float("nan")
    return float(sum(1.0 / k for k in range(1, n + 1)) / n)


def _z_by_training(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (train - mean) / scale, (test - mean) / scale


def _unit_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.where(norm > 1e-12, norm, 1.0)
    return x / norm


def cross_view_mrr(
    train_embeddings: dict[str, np.ndarray],
    test_embeddings: dict[str, np.ndarray],
) -> tuple[float, float]:
    n = next(iter(test_embeddings.values())).shape[0]
    if n < 2:
        return float("nan"), float("nan")

    values: list[float] = []
    for left in VIEWS:
        for right in VIEWS:
            if left == right:
                continue
            _, a = _z_by_training(
                np.asarray(train_embeddings[left], dtype=np.float64),
                np.asarray(test_embeddings[left], dtype=np.float64),
            )
            _, b = _z_by_training(
                np.asarray(train_embeddings[right], dtype=np.float64),
                np.asarray(test_embeddings[right], dtype=np.float64),
            )
            sim = _unit_rows(a) @ _unit_rows(b).T
            for i in range(n):
                # Stable descending similarity, then sample index.
                order = np.lexsort((np.arange(n), -sim[i]))
                rank = int(np.flatnonzero(order == i)[0]) + 1
                values.append(1.0 / rank)

    mrr = float(np.mean(values))
    random = harmonic_random_mrr(n)
    adjusted = float(np.clip((mrr - random) / max(1e-12, 1.0 - random), 0.0, 1.0))
    return mrr, adjusted


def r2_score(truth: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(truth, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape:
        raise RuntimeError(f"R2 shape mismatch: truth={y.shape}, pred={p.shape}")
    ss_res = float(np.sum((y - p) ** 2))
    center = np.mean(y, axis=0, keepdims=True)
    ss_tot = float(np.sum((y - center) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def reconstruction_score(
    train_truth: dict[str, np.ndarray],
    test_truth: dict[str, np.ndarray],
    test_pred: dict[str, np.ndarray],
) -> float:
    scores: list[float] = []
    for view in VIEWS:
        train = np.asarray(train_truth[view], dtype=np.float64)
        truth = np.asarray(test_truth[view], dtype=np.float64)
        pred = np.asarray(test_pred[view], dtype=np.float64)
        if truth.shape != pred.shape:
            raise RuntimeError(
                f"{view} reconstruction shape mismatch: {truth.shape} vs {pred.shape}"
            )
        null = np.mean(train, axis=0, keepdims=True)
        null_mse = float(np.mean((train - null) ** 2))
        mse = float(np.mean((truth - pred) ** 2))
        normalized = mse / max(null_mse, 1e-12)
        scores.append(float(np.clip(1.0 - normalized, 0.0, 1.0)))
    return float(np.mean(scores))


def utility_from_components(metrics: dict[str, float]) -> float:
    return float(sum(PRIMARY_WEIGHTS[k] * float(metrics[k]) for k in PRIMARY_WEIGHTS))


# =========================================================================
# BLOCK_PCA
# =========================================================================

@dataclass
class BlockPCAModel:
    view_slices: dict[str, slice]
    block_scales: dict[str, float]
    pca: PCAProjection

    def _concat(self, views: dict[str, np.ndarray], included: set[str] | None = None) -> np.ndarray:
        blocks: list[np.ndarray] = []
        included = set(VIEWS) if included is None else set(included)
        n = next(iter(views.values())).shape[0]
        for view in VIEWS:
            d = self.view_slices[view].stop - self.view_slices[view].start
            if view in included:
                block = np.asarray(views[view], dtype=np.float64) * self.block_scales[view]
            else:
                block = np.zeros((n, d), dtype=np.float64)
            blocks.append(block)
        return np.concatenate(blocks, axis=1)

    def shared_scores(
        self, views: dict[str, np.ndarray], included: set[str] | None = None
    ) -> np.ndarray:
        return self.pca.transform(self._concat(views, included))

    def view_embeddings(self, views: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        centered_mean = self.pca.mean
        out: dict[str, np.ndarray] = {}
        n = next(iter(views.values())).shape[0]
        for view in VIEWS:
            full = np.zeros((n, self.pca.mean.size), dtype=np.float64)
            sl = self.view_slices[view]
            full[:, sl] = np.asarray(views[view], dtype=np.float64) * self.block_scales[view]
            # subtract only the corresponding training mean contribution
            local = full.copy()
            local[:, sl] -= centered_mean[sl]
            out[view] = local @ self.pca.components.T
        return out

    def reconstruct(
        self, views: dict[str, np.ndarray], included: set[str] | None = None
    ) -> dict[str, np.ndarray]:
        z = self.shared_scores(views, included)
        full = self.pca.inverse(z)
        out: dict[str, np.ndarray] = {}
        for view in VIEWS:
            sl = self.view_slices[view]
            out[view] = full[:, sl] / self.block_scales[view]
        return out

    def modality_balance(self) -> float:
        per_factor: list[float] = []
        for k in range(self.pca.components.shape[0]):
            energies = []
            for view in VIEWS:
                sl = self.view_slices[view]
                energies.append(float(np.sum(self.pca.components[k, sl] ** 2)))
            per_factor.append(entropy_balance(energies))
        return float(np.median(per_factor))


def fit_block_pca(train_views: dict[str, np.ndarray], rank: int = 10) -> BlockPCAModel:
    view_slices: dict[str, slice] = {}
    block_scales: dict[str, float] = {}
    blocks: list[np.ndarray] = []
    start = 0
    for view in VIEWS:
        x = np.asarray(train_views[view], dtype=np.float64)
        total_var = float(np.sum(np.var(x, axis=0, ddof=0)))
        if total_var <= 0.0:
            raise RuntimeError(f"{view}: zero total training variance for BLOCK_PCA.")
        scale = 1.0 / math.sqrt(total_var)
        block_scales[view] = scale
        blocks.append(x * scale)
        view_slices[view] = slice(start, start + x.shape[1])
        start += x.shape[1]
    combined = np.concatenate(blocks, axis=1)
    pca = fit_pca(combined, rank)
    return BlockPCAModel(view_slices=view_slices, block_scales=block_scales, pca=pca)


def evaluate_block_pca(
    train_views: dict[str, np.ndarray],
    test_views: dict[str, np.ndarray],
) -> dict[str, float]:
    model = fit_block_pca(train_views, rank=10)
    train_emb = model.view_embeddings(train_views)
    test_emb = model.view_embeddings(test_views)
    _, adjusted = cross_view_mrr(train_emb, test_emb)
    pred = model.reconstruct(test_views)
    heldout = reconstruction_score(train_views, test_views, pred)

    mv: list[float] = []
    for omitted in VIEWS:
        included = set(VIEWS) - {omitted}
        prediction = model.reconstruct(test_views, included=included)[omitted]
        mv.append(float(np.clip(r2_score(test_views[omitted], prediction), 0.0, 1.0)))

    metrics = {
        "heldout_reconstruction_score": heldout,
        "random_adjusted_cross_view_mrr": adjusted,
        "missing_view_reconstruction_score": float(np.mean(mv)),
        "modality_balance_score": model.modality_balance(),
    }
    metrics["utility"] = utility_from_components(metrics)
    return metrics


# =========================================================================
# Frozen MOFA adapter
# =========================================================================

def _protocol_required_keys(func: Any, variable_name: str) -> set[str]:
    src = inspect.getsource(func)
    tree = ast.parse(textwrap_dedent(src))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id == variable_name and isinstance(node.slice, ast.Constant):
                if isinstance(node.slice.value, str):
                    keys.add(node.slice.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == variable_name
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                keys.add(node.args[0].value)
    return keys


def textwrap_dedent(text: str) -> str:
    import textwrap
    return textwrap.dedent(text)


def discover_mofa_protocol(mofa_module: Any) -> tuple[dict[str, Any], Path]:
    required = _protocol_required_keys(mofa_module.fit_mofa, "frozen_protocol")
    candidates = sorted(DIR_05C2A.glob("*.json")) + sorted(DIR_05C2B.glob("*.json"))
    loaded: list[tuple[int, Path, dict[str, Any]]] = []
    for path in candidates:
        try:
            payload = load_json(path)
        except Exception:
            continue
        score = sum(1 for key in required if key in payload)
        loaded.append((score, path, payload))

    if not loaded:
        raise RuntimeError("No frozen MOFA protocol JSON could be loaded.")

    loaded.sort(key=lambda x: (-x[0], x[1].name))
    score, path, payload = loaded[0]
    if required and score < len(required):
        missing = sorted(required - set(payload))
        raise RuntimeError(
            "No frozen MOFA protocol contains all fit_mofa keys. "
            f"Best={path.name}; missing={missing}"
        )
    return payload, path


def _mapping_attr(obj: Any, candidates: tuple[str, ...]) -> dict[str, np.ndarray] | None:
    for name in candidates:
        value = getattr(obj, name, None)
        if isinstance(value, dict) and all(view in value for view in VIEWS):
            return {view: np.asarray(value[view]) for view in VIEWS}
    return None


def extract_mofa_parameters(fit: Any, rank: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    weights = _mapping_attr(
        fit,
        ("weights_by_view", "weights", "W_by_view", "W", "view_weights"),
    )
    precision = _mapping_attr(
        fit,
        ("precision_by_view", "precision", "tau_by_view", "Tau", "tau", "view_precision"),
    )

    if weights is None or precision is None:
        attrs = {
            name: type(value).__name__
            for name, value in vars(fit).items()
            if not name.startswith("_")
        }
        raise RuntimeError(
            "Could not resolve frozen MofaFit weights/precision fields. "
            f"Observed attributes={attrs}"
        )

    oriented: dict[str, np.ndarray] = {}
    for view, w in weights.items():
        a = np.asarray(w, dtype=np.float64)
        if a.ndim != 2:
            raise RuntimeError(f"{view}: MOFA weight matrix is not 2D.")
        if a.shape[1] == rank:
            oriented[view] = a
        elif a.shape[0] == rank:
            oriented[view] = a.T
        else:
            raise RuntimeError(f"{view}: cannot orient MOFA weights {a.shape} for K={rank}.")

    prec: dict[str, np.ndarray] = {}
    for view, tau in precision.items():
        t = np.asarray(tau, dtype=np.float64).reshape(-1)
        if t.size == 1:
            t = np.repeat(t, oriented[view].shape[0])
        if t.size != oriented[view].shape[0]:
            raise RuntimeError(
                f"{view}: MOFA precision length {t.size} != features {oriented[view].shape[0]}"
            )
        prec[view] = t

    return oriented, prec


def mofa_project(
    mofa_module: Any,
    observed: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
    precision: dict[str, np.ndarray],
    views_used: list[str],
) -> np.ndarray:
    return np.asarray(
        mofa_module.gaussian_posterior_project(
            observed, weights, precision, views_used
        ),
        dtype=np.float64,
    )


def evaluate_mofa(
    mofa_module: Any,
    train_views: dict[str, np.ndarray],
    test_views: dict[str, np.ndarray],
    rank: int,
    seed: int,
    sample_ids: list[str],
    protocol: dict[str, Any],
) -> dict[str, float]:
    selected = {
        view: np.arange(train_views[view].shape[1], dtype=np.int64)
        for view in VIEWS
    }
    fit = mofa_module.fit_mofa(
        train_views,
        selected,
        int(rank),
        int(seed),
        protocol,
        sample_ids,
    )
    weights, precision = extract_mofa_parameters(fit, rank)

    train_joint = mofa_project(mofa_module, train_views, weights, precision, list(VIEWS))
    test_joint = mofa_project(mofa_module, test_views, weights, precision, list(VIEWS))

    train_per_view: dict[str, np.ndarray] = {}
    test_per_view: dict[str, np.ndarray] = {}
    for view in VIEWS:
        train_per_view[view] = mofa_project(
            mofa_module, {view: train_views[view]}, weights, precision, [view]
        )
        test_per_view[view] = mofa_project(
            mofa_module, {view: test_views[view]}, weights, precision, [view]
        )
    _, adjusted = cross_view_mrr(train_per_view, test_per_view)

    pred: dict[str, np.ndarray] = {}
    for view in VIEWS:
        pred[view] = test_joint @ weights[view].T
    heldout = reconstruction_score(train_views, test_views, pred)

    mv: list[float] = []
    for omitted in VIEWS:
        used = [view for view in VIEWS if view != omitted]
        obs = {view: test_views[view] for view in used}
        z = mofa_project(mofa_module, obs, weights, precision, used)
        prediction = z @ weights[omitted].T
        mv.append(float(np.clip(r2_score(test_views[omitted], prediction), 0.0, 1.0)))

    factor_balances = []
    for k in range(min(rank, next(iter(weights.values())).shape[1])):
        factor_balances.append(
            entropy_balance([np.sum(weights[v][:, k] ** 2) for v in VIEWS])
        )

    metrics = {
        "heldout_reconstruction_score": heldout,
        "random_adjusted_cross_view_mrr": adjusted,
        "missing_view_reconstruction_score": float(np.mean(mv)),
        "modality_balance_score": float(np.median(factor_balances)),
    }
    metrics["utility"] = utility_from_components(metrics)
    return metrics


# =========================================================================
# Frozen neural V0-V4 adapter
# =========================================================================

def parse_frozen_profile(neural_module: Any) -> Any:
    selected = load_json(SELECTED_PROFILE)["selected_profile"]
    source_row = selected["source_profile_grid_row"]
    table = pd.read_csv(PROFILE_GRID, dtype=str).fillna("")
    rows = table[table["profile_id"].astype(str).eq("ARCH_COMPACT__REG_WEAK")]
    if rows.shape[0] != 1:
        raise RuntimeError("Frozen profile row is not unique.")
    row = rows.iloc[0]
    for key, value in source_row.items():
        if key in table.columns and clean_text(row[key]) != clean_text(value):
            raise RuntimeError(f"Frozen profile grid mismatch at {key}.")
    profile = neural_module.parse_profile(row)
    if clean_text(getattr(profile, "profile_id", "")) != "ARCH_COMPACT__REG_WEAK":
        raise RuntimeError("parse_profile() did not reconstruct ARCH_COMPACT__REG_WEAK.")
    return profile


def make_smoke_profile(profile: Any) -> Any:
    names: set[str]
    if is_dataclass(profile):
        names = {f.name for f in fields(profile)}
        kwargs = {}
        for name in names:
            low = name.lower()
            if "maximum" in low and "epoch" in low or low in {"max_epochs", "maximum_epochs"}:
                kwargs[name] = SMOKE_MAX_EPOCHS
            if "patience" in low:
                kwargs[name] = SMOKE_PATIENCE
        if kwargs:
            return replace(profile, **kwargs)

    clone = copy.deepcopy(profile)
    for name in dir(clone):
        low = name.lower()
        try:
            if "maximum" in low and "epoch" in low or low in {"max_epochs", "maximum_epochs"}:
                setattr(clone, name, SMOKE_MAX_EPOCHS)
            elif "patience" in low:
                setattr(clone, name, SMOKE_PATIENCE)
        except Exception:
            pass
    return clone


def availability_candidates(
    views: dict[str, np.ndarray],
    row_available: dict[str, np.ndarray] | None = None,
) -> list[Any]:
    n = next(iter(views.values())).shape[0]
    if row_available is None:
        row_available = {view: np.ones(n, dtype=bool) for view in VIEWS}
    as_dict_bool = {view: np.asarray(row_available[view], dtype=bool) for view in VIEWS}
    as_dict_float = {view: np.asarray(row_available[view], dtype=np.float32) for view in VIEWS}
    matrix_bool = np.column_stack([as_dict_bool[v] for v in VIEWS])
    matrix_float = matrix_bool.astype(np.float32)
    return [as_dict_bool, matrix_bool, as_dict_float, matrix_float]


def extract_epoch(result: Any) -> int:
    if isinstance(result, dict):
        for key in (
            "selected_epoch",
            "best_epoch",
            "epoch",
            "epochs",
            "n_epochs",
        ):
            if key in result:
                try:
                    value = int(result[key])
                    if value >= 1:
                        return value
                except Exception:
                    pass
        for value in result.values():
            try:
                found = extract_epoch(value)
                if found >= 1:
                    return found
            except Exception:
                pass
    if isinstance(result, (tuple, list)):
        for value in result:
            if isinstance(value, (int, np.integer)) and int(value) >= 1:
                return int(value)
        for value in result:
            try:
                found = extract_epoch(value)
                if found >= 1:
                    return found
            except Exception:
                pass
    if isinstance(result, (int, np.integer)) and int(result) >= 1:
        return int(result)
    raise RuntimeError(f"Could not extract selected epoch from {type(result).__name__}.")


def extract_model(result: Any) -> Any:
    if hasattr(result, "state_dict") and hasattr(result, "eval"):
        return result
    if isinstance(result, dict):
        for key in ("model", "fitted_model", "neural_model"):
            if key in result:
                try:
                    return extract_model(result[key])
                except Exception:
                    pass
        for value in result.values():
            try:
                return extract_model(value)
            except Exception:
                pass
    if isinstance(result, (tuple, list)):
        for value in result:
            try:
                return extract_model(value)
            except Exception:
                pass
    raise RuntimeError(f"Could not extract torch model from {type(result).__name__}.")


def extract_view_embedding_dict(value: Any, n: int) -> dict[str, np.ndarray] | None:
    if isinstance(value, dict) and all(view in value for view in VIEWS):
        out: dict[str, np.ndarray] = {}
        for view in VIEWS:
            a = np.asarray(value[view])
            if a.ndim != 2 or a.shape[0] != n:
                return None
            out[view] = a.astype(np.float64)
        return out
    if isinstance(value, dict):
        for child in value.values():
            found = extract_view_embedding_dict(child, n)
            if found is not None:
                return found
    if isinstance(value, (tuple, list)):
        for child in value:
            found = extract_view_embedding_dict(child, n)
            if found is not None:
                return found
    return None


def extract_prediction_dict(value: Any, n: int) -> dict[str, np.ndarray] | None:
    return extract_view_embedding_dict(value, n)


def build_variant_map(ablation_module: Any) -> dict[str, Any]:
    specs = ablation_module.build_variant_specs(load_json(LOSS_CONTRACT))
    if isinstance(specs, dict):
        mapping = {clean_text(key): value for key, value in specs.items()}
    else:
        mapping = {}
        for spec in specs:
            variant_id = clean_text(
                getattr(spec, "variant_id", getattr(spec, "model_id", getattr(spec, "name", "")))
            )
            if variant_id:
                mapping[variant_id] = spec
    missing = [v for v in NEURAL_MODELS if v not in mapping]
    if missing:
        raise RuntimeError(f"Frozen neural variant map missing: {missing}")
    return mapping


def choose_working_availability(
    train_fn: Any,
    neural_module: Any,
    profile: Any,
    variant: Any,
    fit_views: dict[str, np.ndarray],
    early_views: dict[str, np.ndarray],
    seed: int,
) -> tuple[Any, Any, Any]:
    last_error: Exception | None = None
    for fit_av, early_av in zip(
        availability_candidates(fit_views), availability_candidates(early_views)
    ):
        try:
            result = train_fn(
                neural_module,
                profile,
                variant,
                fit_views,
                fit_av,
                early_views,
                early_av,
                int(seed),
            )
            return result, fit_av, early_av
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        "No supported availability representation matched frozen neural training API. "
        f"Last error: {last_error}"
    )


def extract_unique_array_matching_shape(
    value: Any,
    expected_shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    """
    Resolve one numeric array from an implementation return object without
    assuming tuple position or an undocumented container schema.

    This is deliberately fail-closed:
      - recursively inspect dict/list/tuple values;
      - accept only arrays whose shape exactly matches the frozen expected
        target shape;
      - require exactly one matching candidate;
      - reject non-finite predictions.

    This is an adapter/schema recovery only.  It does not transform, average,
    subset, resize, or otherwise alter predicted values.
    """
    candidates: list[tuple[str, np.ndarray]] = []

    def visit(obj: Any, where: str) -> None:
        if isinstance(obj, dict):
            for key, child in obj.items():
                visit(child, f"{where}.{key}")
            return

        if isinstance(obj, (tuple, list)):
            for i, child in enumerate(obj):
                visit(child, f"{where}[{i}]")
            return

        # Support torch tensors without importing torch into this adapter.
        if hasattr(obj, "detach") and hasattr(obj, "cpu") and hasattr(obj, "numpy"):
            try:
                arr = obj.detach().cpu().numpy()
            except Exception:
                return
        else:
            try:
                arr = np.asarray(obj)
            except Exception:
                return

        if tuple(arr.shape) != tuple(expected_shape):
            return
        if not np.issubdtype(arr.dtype, np.number):
            return

        arr64 = np.asarray(arr, dtype=np.float64)
        if not np.isfinite(arr64).all():
            raise RuntimeError(
                f"{label}: matching prediction candidate at {where} contains non-finite values."
            )
        candidates.append((where, arr64))

    visit(value, label)

    if len(candidates) != 1:
        shapes: list[str] = []

        def collect_shapes(obj: Any, where: str) -> None:
            if isinstance(obj, dict):
                for key, child in obj.items():
                    collect_shapes(child, f"{where}.{key}")
                return
            if isinstance(obj, (tuple, list)):
                for i, child in enumerate(obj):
                    collect_shapes(child, f"{where}[{i}]")
                return
            try:
                if hasattr(obj, "detach") and hasattr(obj, "cpu") and hasattr(obj, "numpy"):
                    a = obj.detach().cpu().numpy()
                else:
                    a = np.asarray(obj)
                shapes.append(f"{where}:{tuple(a.shape)}")
            except Exception:
                shapes.append(f"{where}:<non-array>")

        collect_shapes(value, label)
        raise RuntimeError(
            f"{label}: expected exactly one numeric prediction array with shape "
            f"{tuple(expected_shape)}, found {len(candidates)}. "
            f"Observed return-object shapes={shapes}"
        )

    return candidates[0][1]



def evaluate_neural(
    neural_module: Any,
    ablation_module: Any,
    profile: Any,
    variant: Any,
    fit_views: dict[str, np.ndarray],
    early_views: dict[str, np.ndarray],
    refit_views: dict[str, np.ndarray],
    test_views: dict[str, np.ndarray],
    seed: int,
    smoke: bool,
) -> dict[str, float]:
    effective_profile = make_smoke_profile(profile) if smoke else profile

    early_result, _, _ = choose_working_availability(
        ablation_module.train_with_early_stopping,
        neural_module,
        effective_profile,
        variant,
        fit_views,
        early_views,
        seed,
    )
    selected_epoch = extract_epoch(early_result)
    if smoke:
        selected_epoch = max(1, min(selected_epoch, SMOKE_MAX_EPOCHS))

    # Determine the availability representation accepted by fixed-epoch fitting.
    fixed_result = None
    chosen_av = None
    last_error: Exception | None = None
    for av in availability_candidates(refit_views):
        try:
            fixed_result = ablation_module.train_fixed_epochs(
                neural_module,
                effective_profile,
                variant,
                refit_views,
                av,
                int(seed),
                int(selected_epoch),
            )
            chosen_av = av
            break
        except Exception as exc:
            last_error = exc
    if fixed_result is None:
        raise RuntimeError(f"Neural fixed-epoch refit failed for all availability forms: {last_error}")

    model = extract_model(fixed_result)
    model.eval()

    test_av_candidates = availability_candidates(test_views)
    train_av_candidates = availability_candidates(refit_views)

    encode_train = encode_test = None
    recon_test = None
    working_test_av = working_train_av = None
    for train_av, test_av in zip(train_av_candidates, test_av_candidates):
        try:
            encode_train = ablation_module.encode_numpy(model, refit_views, train_av)
            encode_test = ablation_module.encode_numpy(model, test_views, test_av)
            recon_test = ablation_module.predict_reconstruction_numpy(
                model, test_views, test_av
            )
            working_train_av = train_av
            working_test_av = test_av
            break
        except Exception as exc:
            last_error = exc
    if encode_train is None or encode_test is None or recon_test is None:
        raise RuntimeError(f"Neural encode/reconstruction adapter failed: {last_error}")

    n_train = next(iter(refit_views.values())).shape[0]
    n_test = next(iter(test_views.values())).shape[0]
    train_emb = extract_view_embedding_dict(encode_train, n_train)
    test_emb = extract_view_embedding_dict(encode_test, n_test)
    if train_emb is None or test_emb is None:
        raise RuntimeError(
            "Could not resolve per-view shared embeddings from frozen encode_numpy output."
        )
    _, adjusted = cross_view_mrr(train_emb, test_emb)

    recon = extract_prediction_dict(recon_test, n_test)
    if recon is None:
        raise RuntimeError("Could not resolve per-view reconstruction dictionary.")
    heldout = reconstruction_score(refit_views, test_views, recon)

    mv: list[float] = []
    for omitted in VIEWS:
        pred_raw = ablation_module.predict_missing_view_numpy(
            model, test_views, working_test_av, omitted
        )

        # Frozen 05d4 helper returns more than one object for missing-view
        # prediction.  v1/v2 incorrectly assumed a bare array in the non-dict
        # branch, producing an inhomogeneous np.asarray(...) failure.  Do not
        # guess tuple position post hoc.  Resolve the prediction solely by the
        # already-known omitted-view target geometry, and require a unique
        # exact-shape numeric candidate.
        pred = extract_unique_array_matching_shape(
            pred_raw,
            tuple(np.asarray(test_views[omitted]).shape),
            f"predict_missing_view_numpy[{omitted}]",
        )
        mv.append(float(np.clip(r2_score(test_views[omitted], pred), 0.0, 1.0)))

    energy_raw = model.decoder_shared_energy()
    if isinstance(energy_raw, dict):
        energies = [float(np.asarray(energy_raw[v]).sum()) for v in VIEWS]
    else:
        e = np.asarray(energy_raw, dtype=np.float64).reshape(-1)
        if e.size < len(VIEWS):
            raise RuntimeError("decoder_shared_energy() returned fewer than three view energies.")
        energies = e[: len(VIEWS)].tolist()

    metrics = {
        "heldout_reconstruction_score": heldout,
        "random_adjusted_cross_view_mrr": adjusted,
        "missing_view_reconstruction_score": float(np.mean(mv)),
        "modality_balance_score": entropy_balance(energies),
        "selected_epoch": float(selected_epoch),
    }
    metrics["utility"] = utility_from_components(metrics)
    return metrics


# =========================================================================
# One-SE and final-full-data selection hierarchy
# =========================================================================

def one_se_select(
    rows: pd.DataFrame,
    eligible_models: tuple[str, ...] = EXPECTED_ELIGIBLE_MODELS,
) -> dict[str, Any]:
    required = {"model_id", "inner_fold", "utility", "failed"}
    missing = required - set(rows.columns)
    if missing:
        raise RuntimeError(f"Inner selection rows missing columns: {sorted(missing)}")

    work = rows.copy()
    work["utility"] = pd.to_numeric(work["utility"], errors="raise")
    work["failed"] = work["failed"].astype(bool)

    summaries = []
    for model_id in eligible_models:
        part = work[work["model_id"].astype(str).eq(model_id)].copy()
        if part.empty:
            summaries.append(
                {
                    "model_id": model_id,
                    "eligible": False,
                    "mean": float("nan"),
                    "se": float("nan"),
                    "failure_reason": "no_inner_rows",
                }
            )
            continue
        if part["failed"].any() or not np.isfinite(part["utility"]).all():
            summaries.append(
                {
                    "model_id": model_id,
                    "eligible": False,
                    "mean": float("nan"),
                    "se": float("nan"),
                    "failure_reason": "failed_or_nonfinite_inner_fit",
                }
            )
            continue

        # Seeds are repeated numerical evaluations; the CV fold is the independent
        # unit for the standard error used by the one-SE rule.
        fold_means = (
            part.groupby("inner_fold", sort=True)["utility"].mean().to_numpy(dtype=float)
        )
        mean = float(np.mean(fold_means))
        se = (
            float(np.std(fold_means, ddof=1) / math.sqrt(fold_means.size))
            if fold_means.size > 1
            else 0.0
        )
        summaries.append(
            {
                "model_id": model_id,
                "eligible": True,
                "mean": mean,
                "se": se,
                "failure_reason": "",
            }
        )

    summary = pd.DataFrame(summaries)
    good = summary[summary["eligible"]].copy()
    if good.empty:
        raise RuntimeError("No eligible model survived inner-CV evaluation.")

    best_mean = float(good["mean"].max())
    best_candidates = good[np.isclose(good["mean"], best_mean, rtol=0.0, atol=1e-12)]
    best_candidates = best_candidates.assign(
        complexity=best_candidates["model_id"].map(EXPECTED_COMPLEXITY)
    ).sort_values(["complexity", "model_id"], kind="mergesort")
    best_id = str(best_candidates.iloc[0]["model_id"])
    best_se = float(good.loc[good["model_id"].eq(best_id), "se"].iloc[0])
    cutoff = best_mean - best_se

    within = good[good["mean"] >= cutoff - 1e-12].copy()
    within["complexity"] = within["model_id"].map(EXPECTED_COMPLEXITY)
    within = within.sort_values(["complexity", "model_id"], kind="mergesort")
    selected = str(within.iloc[0]["model_id"])

    return {
        "selected_model_id": selected,
        "best_model_id": best_id,
        "best_mean": best_mean,
        "best_se": best_se,
        "one_se_cutoff": cutoff,
        "within_one_se": within["model_id"].tolist(),
        "summary": summary.to_dict(orient="records"),
    }


def final_method_freeze(selected_model_ids: list[str]) -> dict[str, Any]:
    if len(selected_model_ids) != 25:
        raise RuntimeError(
            f"Final method freeze requires 25 outer-split selections, got {len(selected_model_ids)}."
        )
    counts = pd.Series(selected_model_ids).value_counts().to_dict()
    ranked = sorted(
        EXPECTED_ELIGIBLE_MODELS,
        key=lambda model: (-int(counts.get(model, 0)), EXPECTED_COMPLEXITY[model], model),
    )
    top = ranked[0]
    frequency = int(counts.get(top, 0)) / 25.0
    if frequency >= 0.40:
        primary = [top]
        status = "UNIQUE_PRIMARY_REPRESENTATION"
    else:
        primary = ranked[:2]
        status = "CO_PRIMARY_REPRESENTATIONS_METHOD_UNCERTAINTY"
    return {
        "status": status,
        "primary_model_ids": primary,
        "selection_counts": {m: int(counts.get(m, 0)) for m in EXPECTED_ELIGIBLE_MODELS},
        "top_selection_frequency": frequency,
        "minimum_single_method_selection_frequency": 0.40,
    }


# =========================================================================
# Synthetic fixture / branch self-test
# =========================================================================

def validate_05dev0_fixture() -> dict[str, Any]:
    require_dir(FIXTURE_ROOT)
    npy_files = sorted(FIXTURE_ROOT.rglob("*.npy"))
    observed: dict[str, list[tuple[str, tuple[int, ...]]]] = {v: [] for v in VIEWS}
    shape_targets = {
        "RNA": (24, 40),
        "METHYLATION": (24, 50),
        "CNV": (24, 30),
    }
    for path in npy_files:
        try:
            arr = np.load(path, mmap_mode="r")
        except Exception:
            continue
        shape = tuple(arr.shape)
        for view, expected in shape_targets.items():
            if shape == expected:
                observed[view].append((relative_path(path), shape))
    missing = [v for v in VIEWS if not observed[v]]
    if missing:
        raise RuntimeError(f"05dev0 fixture matrices not found by frozen shapes: {missing}")
    return {
        "fixture_root": relative_path(FIXTURE_ROOT),
        "matrices_by_expected_shape": observed,
        "fixture_contract_sha256": sha256_file(FIXTURE_CONTRACT),
        "fixture_lock_sha256": sha256_file(FIXTURE_LOCK),
    }


def synthetic_matrices(n: int = 36, p: int = 36) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SMOKE_SEED)
    z = rng.normal(size=(n, 5))
    out = {}
    for i, view in enumerate(VIEWS):
        w = rng.normal(size=(5, p)) / math.sqrt(5)
        private = rng.normal(size=(n, 2)) @ (rng.normal(size=(2, p)) / math.sqrt(2))
        noise = rng.normal(scale=0.4 + 0.05 * i, size=(n, p))
        out[view] = (z @ w + 0.35 * private + noise).astype(np.float64)
    # Development-only feature-level missingness.
    out["METHYLATION"][1, 3] = np.nan
    out["METHYLATION"][4, 8] = np.nan
    return out


def selftest_selection_rules() -> dict[str, Any]:
    rows = []
    # Four folds; BLOCK_PCA best mean, V0 within one SE and more complex;
    # one-SE must still choose the lowest-complexity qualifying model.
    means = {
        "BLOCK_PCA": [0.60, 0.61, 0.59, 0.60],
        "MOFA_K5": [0.605, 0.615, 0.595, 0.605],
        "MOFA_K10": [0.61, 0.62, 0.60, 0.61],
        "MOFA_K15": [0.62, 0.63, 0.61, 0.62],
        "MOFA_K20": [0.63, 0.64, 0.62, 0.63],
        "V0_BALANCED_AE": [0.635, 0.645, 0.625, 0.635],
        "V1_SHARED_PRIVATE_AE": [0.637, 0.647, 0.627, 0.637],
        "V2_CONTRASTIVE_SPAE": [0.638, 0.648, 0.628, 0.638],
        "V3_DROPOUT_SPAE": [0.639, 0.649, 0.629, 0.639],
        "V4_SABLE_MO": [0.64, 0.65, 0.63, 0.64],
    }
    for model, values in means.items():
        for fold, value in enumerate(values, 1):
            rows.append(
                {
                    "model_id": model,
                    "inner_fold": fold,
                    "utility": value,
                    "failed": False,
                }
            )
    result = one_se_select(pd.DataFrame(rows))
    if result["selected_model_id"] not in EXPECTED_ELIGIBLE_MODELS:
        raise RuntimeError("Synthetic one-SE selector returned unknown model.")

    unique = final_method_freeze(["V4_SABLE_MO"] * 11 + ["BLOCK_PCA"] * 9 + ["MOFA_K5"] * 5)
    if unique["primary_model_ids"] != ["V4_SABLE_MO"]:
        raise RuntimeError("Synthetic final-freeze >=40% rule failed.")

    uncertain = final_method_freeze(
        ["V4_SABLE_MO"] * 9 + ["BLOCK_PCA"] * 8 + ["MOFA_K5"] * 8
    )
    if uncertain["status"] != "CO_PRIMARY_REPRESENTATIONS_METHOD_UNCERTAINTY":
        raise RuntimeError("Synthetic <40% uncertainty rule failed.")

    return {
        "one_se_selected": result["selected_model_id"],
        "one_se_cutoff": result["one_se_cutoff"],
        "unique_primary_test": unique,
        "uncertainty_test": uncertain,
    }


def run_synthetic_branch_selftest(modules: dict[str, Any]) -> dict[str, Any]:
    matrices = synthetic_matrices()
    train = np.arange(0, 28, dtype=np.int64)
    test = np.arange(28, 36, dtype=np.int64)

    # Same fold-local feature reduction layer that authorized execution will use.
    prep = fit_preprocessing_bundle(matrices, train, {v: 30 for v in VIEWS})
    train_std = transform_bundle(prep, matrices, train, use_pca=False)
    test_std = transform_bundle(prep, matrices, test, use_pca=False)

    block_metrics = evaluate_block_pca(train_std, test_std)
    if not all(np.isfinite(block_metrics[k]) for k in PRIMARY_WEIGHTS):
        raise RuntimeError("BLOCK_PCA synthetic metrics are non-finite.")

    protocol, protocol_path = discover_mofa_protocol(modules["mofa"])
    mofa_metrics: dict[str, Any] = {}
    for model_id, rank in MOFA_RANK.items():
        metrics = evaluate_mofa(
            modules["mofa"],
            train_std,
            test_std,
            rank=rank,
            seed=INNER_MODEL_SEEDS[0],
            sample_ids=[f"S{i:03d}" for i in train],
            protocol=protocol,
        )
        if not all(np.isfinite(metrics[k]) for k in PRIMARY_WEIGHTS):
            raise RuntimeError(f"{model_id} synthetic metrics are non-finite.")
        mofa_metrics[model_id] = metrics

    # Neural preprocessing: fit on development fit subset, select epoch on early,
    # then refit preprocessing on fit+early, exactly mirroring the scientific
    # fit/early/refit firewall but with development-only 3-epoch limits.
    fit_idx = np.arange(0, 20, dtype=np.int64)
    early_idx = np.arange(20, 28, dtype=np.int64)

    prelim = fit_preprocessing_bundle(
        matrices, fit_idx, {v: 30 for v in VIEWS}, NEURAL_PCA_COMPONENTS
    )
    fit_views = transform_bundle(prelim, matrices, fit_idx, use_pca=True)
    early_views = transform_bundle(prelim, matrices, early_idx, use_pca=True)

    refit = fit_preprocessing_bundle(
        matrices, train, {v: 30 for v in VIEWS}, NEURAL_PCA_COMPONENTS
    )
    refit_views = transform_bundle(refit, matrices, train, use_pca=True)
    test_views = transform_bundle(refit, matrices, test, use_pca=True)

    profile = parse_frozen_profile(modules["neural"])
    variants = build_variant_map(modules["ablation"])

    neural_metrics: dict[str, Any] = {}
    for model_id in NEURAL_MODELS:
        metrics = evaluate_neural(
            modules["neural"],
            modules["ablation"],
            profile,
            variants[model_id],
            fit_views,
            early_views,
            refit_views,
            test_views,
            seed=INNER_MODEL_SEEDS[0],
            smoke=True,
        )
        if not all(np.isfinite(metrics[k]) for k in PRIMARY_WEIGHTS):
            raise RuntimeError(f"{model_id} synthetic metrics are non-finite.")
        neural_metrics[model_id] = metrics

    all_models = {"BLOCK_PCA": block_metrics, **mofa_metrics, **neural_metrics}
    if tuple(all_models) != EXPECTED_ELIGIBLE_MODELS:
        raise RuntimeError(
            f"Synthetic branch set differs from frozen eligible set: {tuple(all_models)}"
        )

    return {
        "mofa_protocol": {
            "path": relative_path(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "models": all_models,
        "all_10_model_branches_executed": True,
        "target_values_accessed": False,
        "scientific_target_results_generated": False,
        "smoke_neural_epoch_limit": SMOKE_MAX_EPOCHS,
        "smoke_epoch_limit_is_scientific_setting": False,
    }


def run_synthetic_selftest() -> None:
    print("=" * 104)
    print("Paper 5 - TARGET representation runner synthetic preauthorization self-test")
    print("=" * 104)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Runner SHA-256: {current_runner_sha()}")
    print("Execution mode: SYNTHETIC_ONLY_PREAUTHORIZATION")
    print("Real TARGET matrix root accessed: NO")
    print("TARGET outcomes/clinical/biology accessed: NO")
    print("TARGET model fitting performed: NO")
    print("Scientific TARGET result generation: NO")
    print("q10 retention used for cross-family selection: NO")
    print()

    started = time.perf_counter()
    try:
        frozen = verify_frozen_contracts()
        fixture = validate_05dev0_fixture()
        modules = load_frozen_modules()
        selector = selftest_selection_rules()
        branch = run_synthetic_branch_selftest(modules)

        report = {
            "status": SELFTEST_STATUS,
            "created_at_utc": utc_now_iso(),
            "script_version": SCRIPT_VERSION,
            "runner_source": {
                "path": relative_path(SCRIPT_PATH),
                "sha256": current_runner_sha(),
                "size_bytes": int(SCRIPT_PATH.stat().st_size),
            },
            "scientific_state": {
                "real_TARGET_values_accessed": False,
                "TARGET_outcomes_accessed": False,
                "TARGET_clinical_values_accessed": False,
                "TARGET_biology_accessed": False,
                "TARGET_model_fitting_performed": False,
                "scientific_TARGET_result_generated": False,
                "real_TARGET_fit_authorized": False,
            },
            "eligible_model_ids": list(EXPECTED_ELIGIBLE_MODELS),
            "complexity_ranks": EXPECTED_COMPLEXITY,
            "frozen_contract_hashes": frozen["source_hashes"],
            "frozen_implementation_hashes": modules["hashes"],
            "frozen_callable_signatures": modules["signatures"],
            "resolved_05d4_source": relative_path(modules["paths"]["05d4"]),
            "resolved_05d4_source_sha256": sha256_file(modules["paths"]["05d4"]),
            "resolved_05d4_script_version": _top_level_string_literal(
                modules["paths"]["05d4"], "SCRIPT_VERSION"
            ),
            "fixture_validation": fixture,
            "selection_rule_selftests": selector,
            "full_10_branch_synthetic_test": branch,
            "v3_adapter_recovery": {
                "scope": "missing-view prediction return-container schema only",
                "selection_rule": "unique numeric array exactly matching omitted-view target shape",
                "prediction_values_modified": False,
                "scientific_metric_definition_changed": False,
                "TARGET_values_accessed_during_recovery": False,
            },
            "elapsed_seconds": float(time.perf_counter() - started),
            "next_required_stage": (
                "Freeze the exact runner SHA and this PASS self-test in 06b1 "
                "before the first real TARGET matrix read or fit."
            ),
        }
        write_json_atomic(SELFTEST_REPORT, report)
        if SELFTEST_FAILURE.exists():
            SELFTEST_FAILURE.unlink()

        print("Frozen contract chain: PASS")
        print("05dev0 synthetic fixture: PASS")
        print(f"Exact 05d4 source resolved: {relative_path(modules['paths']['05d4'])}")
        print("Frozen helper imports/signatures: PASS")
        print("BLOCK_PCA synthetic full path: PASS")
        print("MOFA K=5/10/15/20 synthetic full paths: PASS")
        print("Neural V0/V1/V2/V3/V4 synthetic fit->refit->score paths: PASS")
        print("One-SE selection + 40% final hierarchy self-tests: PASS")
        print("Real TARGET values accessed: NO")
        print("TARGET scientific fit performed: NO")
        print("=" * 104)
        print("06b TARGET runner synthetic self-test: PASS")
        print("=" * 104)
        print(f"Self-test report: {relative_path(SELFTEST_REPORT)}")
        print("Next: freeze this exact runner source + PASS report in 06b1.")
        print("Do NOT create/hand-edit a 06b1 authorization lock manually.")
        print("=" * 104)
    except Exception as exc:
        failure = {
            "status": "06B_TARGET_RUNNER_SYNTHETIC_SELFTEST_FAIL_BEFORE_TARGET_ACCESS",
            "created_at_utc": utc_now_iso(),
            "script_version": SCRIPT_VERSION,
            "runner_sha256": current_runner_sha(),
            "real_TARGET_values_accessed": False,
            "TARGET_model_fitting_performed": False,
            "scientific_TARGET_result_generated": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json_atomic(SELFTEST_FAILURE, failure)
        print("=" * 104)
        print("06b TARGET runner synthetic self-test: FAIL CLOSED")
        print("=" * 104)
        print("Real TARGET values accessed: NO")
        print("TARGET scientific fit performed: NO")
        print(f"Failure type: {type(exc).__name__}")
        print(f"Failure: {exc}")
        print(f"Failure record: {relative_path(SELFTEST_FAILURE)}")
        print("Do NOT create 06b1. Make a new versioned runner source and repeat synthetic testing.")
        print("=" * 104)
        raise


# =========================================================================
# Authorized TARGET execution
# =========================================================================
#
# This section is unreachable without a valid 06b1 lock authorizing the exact
# SHA-256 of this file.  The scientific engine intentionally reuses the same
# preprocessing and model adapters exercised above.
# =========================================================================

def discover_prestandardization_matrix(view: str) -> Path:
    """
    AUTHORIZED-ONLY.  Fail rather than fall back to globally z-scored matrices.
    """
    tokens = {
        "RNA": ("rna",),
        "METHYLATION": ("meth", "methyl"),
        "CNV": ("cnv",),
    }[view]

    candidates: list[Path] = []
    for path in TARGET_PREMODEL.rglob("*.npy"):
        low = path.name.lower()
        if "prestandard" not in low:
            continue
        if any(token in low for token in tokens):
            candidates.append(path)

    # Some 04c outputs may store the transformed matrix under feature_sets/
    # with a manifest path. Search the frozen manifest only if name search is
    # not unique.
    if len(candidates) != 1:
        manifest = TARGET_PREMODEL / "prestandardization_variance_audit_manifest.json"
        require_file(manifest)
        payload = load_json(manifest)
        found: list[Path] = []
        for record in recursive_records(payload):
            for value in record.values():
                if not isinstance(value, str) or not value.lower().endswith(".npy"):
                    continue
                low = value.lower()
                if "prestandard" in low and any(t in low for t in tokens):
                    p = Path(value)
                    if not p.is_absolute():
                        p = PROJECT_ROOT / p
                    if p.is_file():
                        found.append(p.resolve())
        candidates = sorted(set(found))

    if len(candidates) != 1:
        raise RuntimeError(
            f"Could not resolve exactly one frozen pre-standardization matrix for {view}. "
            f"Candidates={[relative_path(p) for p in candidates]}"
        )
    return candidates[0]


def load_authorized_target_inputs() -> tuple[list[str], dict[str, np.ndarray], dict[str, Any]]:
    """
    First real TARGET value read in the entire source.

    Caller MUST already have passed authorization_state().
    """
    require_file(TARGET_CASE_ORDER)
    table = pd.read_csv(TARGET_CASE_ORDER, dtype=str).fillna("")
    if "case_id" not in table.columns:
        raise RuntimeError("TARGET case order does not contain case_id.")
    case_ids = table["case_id"].astype(str).str.strip().tolist()
    if len(case_ids) != 78 or len(set(case_ids)) != 78:
        raise RuntimeError("TARGET case-order integrity failure.")

    paths = {view: discover_prestandardization_matrix(view) for view in VIEWS}
    matrices = {view: np.load(paths[view], mmap_mode="r") for view in VIEWS}
    for view, matrix in matrices.items():
        if matrix.ndim != 2 or matrix.shape[0] != 78:
            raise RuntimeError(f"{view}: TARGET prestandardization matrix geometry {matrix.shape}")

    return case_ids, matrices, {
        view: {"path": relative_path(paths[view]), "sha256": sha256_file(paths[view])}
        for view in VIEWS
    }


def indices_for_inner_split(
    outer: pd.DataFrame,
    inner: pd.DataFrame,
    neural_assign: pd.DataFrame,
    repeat_index: int,
    outer_fold: int,
    inner_fold: int,
) -> dict[str, np.ndarray]:
    split = f"R{repeat_index:02d}_F{outer_fold:02d}"
    ip = inner[
        inner["split_id"].astype(str).eq(split)
        & pd.to_numeric(inner["inner_fold"], errors="raise").astype(int).eq(inner_fold)
    ].copy()
    npart = neural_assign[
        neural_assign["stage"].astype(str).eq("inner_model_selection")
        & neural_assign["split_id"].astype(str).eq(split)
        & pd.to_numeric(neural_assign["inner_fold"], errors="raise").astype(int).eq(inner_fold)
    ].copy()

    def ix(table: pd.DataFrame, column: str, value: str) -> np.ndarray:
        return (
            pd.to_numeric(
                table.loc[table[column].astype(str).eq(value), "case_index"],
                errors="raise",
            )
            .astype(int)
            .to_numpy()
        )

    validation = ix(ip, "inner_role", "validation")
    train_all = ix(ip, "inner_role", "train")
    fit = ix(npart, "role", "fit")
    early = ix(npart, "role", "early_stop")
    outer_test_excluded = ix(npart, "role", "outer_test_excluded")
    if validation.size == 0 or fit.size == 0 or early.size == 0:
        raise RuntimeError(f"{split}/inner{inner_fold}: empty frozen partition.")
    if set(validation) & set(train_all):
        raise RuntimeError("Inner validation/train overlap.")
    if set(fit) & set(early):
        raise RuntimeError("Inner fit/early overlap.")
    return {
        "fit": np.sort(fit),
        "early": np.sort(early),
        "train_all": np.sort(np.concatenate([fit, early])),
        "validation": np.sort(validation),
        "outer_test_excluded": np.sort(outer_test_excluded),
    }


def indices_for_outer_split(
    outer: pd.DataFrame,
    neural_assign: pd.DataFrame,
    repeat_index: int,
    outer_fold: int,
) -> dict[str, np.ndarray]:
    split = f"R{repeat_index:02d}_F{outer_fold:02d}"
    op = outer[outer["split_id"].astype(str).eq(split)].copy()
    npart = neural_assign[
        neural_assign["stage"].astype(str).eq("outer_evaluation")
        & neural_assign["split_id"].astype(str).eq(split)
    ].copy()

    def ix(table: pd.DataFrame, column: str, value: str) -> np.ndarray:
        return (
            pd.to_numeric(
                table.loc[table[column].astype(str).eq(value), "case_index"],
                errors="raise",
            )
            .astype(int)
            .to_numpy()
        )

    train = ix(op, "outer_role", "train")
    test = ix(op, "outer_role", "test")
    fit = ix(npart, "role", "fit")
    early = ix(npart, "role", "early_stop")
    ntest = ix(npart, "role", "test")
    if set(test) != set(ntest):
        raise RuntimeError(f"{split}: outer assignment/neural test mismatch.")
    return {
        "fit": np.sort(fit),
        "early": np.sort(early),
        "train_all": np.sort(train),
        "test": np.sort(test),
    }


def evaluate_model_split(
    model_id: str,
    matrices: dict[str, np.ndarray],
    case_ids: list[str],
    fit_indices: np.ndarray,
    early_indices: np.ndarray,
    train_all_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int,
    modules: dict[str, Any],
    mofa_protocol: dict[str, Any],
    profile: Any,
    variants: dict[str, Any],
    smoke: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        if model_id == "BLOCK_PCA":
            prep = fit_preprocessing_bundle(
                matrices, train_all_indices, FEATURE_BUDGETS
            )
            train_views = transform_bundle(
                prep, matrices, train_all_indices, use_pca=False
            )
            test_views = transform_bundle(prep, matrices, test_indices, use_pca=False)
            metrics = evaluate_block_pca(train_views, test_views)

        elif model_id in MOFA_RANK:
            prep = fit_preprocessing_bundle(
                matrices, train_all_indices, FEATURE_BUDGETS
            )
            train_views = transform_bundle(
                prep, matrices, train_all_indices, use_pca=False
            )
            test_views = transform_bundle(prep, matrices, test_indices, use_pca=False)
            metrics = evaluate_mofa(
                modules["mofa"],
                train_views,
                test_views,
                MOFA_RANK[model_id],
                int(seed),
                [case_ids[i] for i in train_all_indices],
                mofa_protocol,
            )

        elif model_id in NEURAL_MODELS:
            # Preliminary preprocessing fitted only on the neural fit partition
            # so the early-stop subset is not used to choose its own preprocessing.
            prelim = fit_preprocessing_bundle(
                matrices, fit_indices, FEATURE_BUDGETS, NEURAL_PCA_COMPONENTS
            )
            fit_views = transform_bundle(prelim, matrices, fit_indices, use_pca=True)
            early_views = transform_bundle(prelim, matrices, early_indices, use_pca=True)

            # After epoch selection, refit fold-local preprocessing on fit+early,
            # then refit the model for that fixed epoch before held-out scoring.
            refit = fit_preprocessing_bundle(
                matrices, train_all_indices, FEATURE_BUDGETS, NEURAL_PCA_COMPONENTS
            )
            refit_views = transform_bundle(
                refit, matrices, train_all_indices, use_pca=True
            )
            test_views = transform_bundle(refit, matrices, test_indices, use_pca=True)
            metrics = evaluate_neural(
                modules["neural"],
                modules["ablation"],
                profile,
                variants[model_id],
                fit_views,
                early_views,
                refit_views,
                test_views,
                int(seed),
                smoke=smoke,
            )
        else:
            raise RuntimeError(f"Unknown eligible model: {model_id}")

        for key in PRIMARY_WEIGHTS:
            if key not in metrics or not np.isfinite(float(metrics[key])):
                raise RuntimeError(f"{model_id}: non-finite/missing primary metric {key}.")
        return {
            "model_id": model_id,
            "seed": int(seed),
            "failed": False,
            "failure_reason": "",
            **{k: float(metrics[k]) for k in metrics},
            "elapsed_seconds": float(time.perf_counter() - started),
        }
    except Exception as exc:
        return {
            "model_id": model_id,
            "seed": int(seed),
            "failed": True,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "utility": float("nan"),
            "elapsed_seconds": float(time.perf_counter() - started),
        }


def shard_path(stage: str, split_id: str, model_id: str, seed: int, inner_fold: int = 0) -> Path:
    safe = f"{stage}__{split_id}__I{inner_fold:02d}__{model_id}__S{seed}.json"
    return REAL_SHARD_ROOT / safe


def save_or_verify_shard(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if path.is_file():
        existing = load_json(path)
        immutable_keys = [
            "script_sha256",
            "stage",
            "split_id",
            "inner_fold",
            "model_id",
            "seed",
        ]
        for key in immutable_keys:
            if existing.get(key) != payload.get(key):
                raise RuntimeError(f"Existing 06b shard mismatch at {key}: {path.name}")
        return existing
    write_json_atomic(path, payload)
    return payload


def run_authorized_target_benchmark(auth_lock: dict[str, Any]) -> None:
    print("=" * 104)
    print("Paper 5 - authorized outcome-blind TARGET representation benchmark")
    print("=" * 104)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Authorized runner SHA-256: {current_runner_sha()}")
    print("TARGET outcomes/clinical/biology access: FORBIDDEN")
    print("Cross-family q10 retention threshold use: FORBIDDEN")
    print("Primary eligible models: 10")
    print()

    frozen = verify_frozen_contracts()
    modules = load_frozen_modules()
    protocol, protocol_path = discover_mofa_protocol(modules["mofa"])
    profile = parse_frozen_profile(modules["neural"])
    variants = build_variant_map(modules["ablation"])

    # This line is intentionally after every source/authorization check.
    case_ids, matrices, input_provenance = load_authorized_target_inputs()

    outer = frozen["outer"]
    inner = frozen["inner"]
    neural_assign = frozen["neural"]

    REAL_SHARD_ROOT.mkdir(parents=True, exist_ok=True)
    REAL_CONTRACT_DIR.mkdir(parents=True, exist_ok=True)

    inner_selection_rows: list[dict[str, Any]] = []
    outer_eval_rows: list[dict[str, Any]] = []
    outer_selected: list[str] = []

    for repeat_index in range(1, 6):
        for outer_fold in range(1, 6):
            split_id = f"R{repeat_index:02d}_F{outer_fold:02d}"
            inner_rows_for_split: list[dict[str, Any]] = []

            for inner_fold in range(1, 5):
                part = indices_for_inner_split(
                    outer, inner, neural_assign, repeat_index, outer_fold, inner_fold
                )
                for model_id in EXPECTED_ELIGIBLE_MODELS:
                    seeds = (0,) if model_id == "BLOCK_PCA" else INNER_MODEL_SEEDS
                    for seed in seeds:
                        path = shard_path(
                            "inner", split_id, model_id, int(seed), inner_fold
                        )
                        if path.is_file():
                            result = load_json(path)
                        else:
                            result = evaluate_model_split(
                                model_id,
                                matrices,
                                case_ids,
                                part["fit"],
                                part["early"],
                                part["train_all"],
                                part["validation"],
                                int(seed if seed != 0 else SMOKE_SEED),
                                modules,
                                protocol,
                                profile,
                                variants,
                                smoke=False,
                            )
                            result.update(
                                {
                                    "script_version": SCRIPT_VERSION,
                                    "script_sha256": current_runner_sha(),
                                    "stage": "inner_model_selection",
                                    "split_id": split_id,
                                    "repeat_index": repeat_index,
                                    "outer_fold": outer_fold,
                                    "inner_fold": inner_fold,
                                    "model_id": model_id,
                                    "seed": int(seed),
                                    "outcome_access": False,
                                    "biology_access": False,
                                    "q10_retention_applied": False,
                                }
                            )
                            save_or_verify_shard(path, result)
                        inner_rows_for_split.append(result)
                        print(
                            f"{split_id} inner={inner_fold} {model_id} seed={seed}: "
                            f"{'FAIL' if result.get('failed') else 'PASS'}"
                        )

            selection_input = pd.DataFrame(inner_rows_for_split)
            selection_input["inner_fold"] = pd.to_numeric(
                selection_input["inner_fold"], errors="raise"
            ).astype(int)
            selection = one_se_select(selection_input)
            selected_id = selection["selected_model_id"]
            outer_selected.append(selected_id)
            inner_selection_rows.append(
                {
                    "repeat_index": repeat_index,
                    "outer_fold": outer_fold,
                    "split_id": split_id,
                    "selected_model_id": selected_id,
                    "best_model_id": selection["best_model_id"],
                    "best_mean": selection["best_mean"],
                    "best_se": selection["best_se"],
                    "one_se_cutoff": selection["one_se_cutoff"],
                    "within_one_se": ";".join(selection["within_one_se"]),
                }
            )

            # Fixed-model benchmark: report every prespecified eligible model on
            # every untouched outer test split.  These results DO NOT feed the
            # final-full-data model freeze; that freeze uses the 25 inner selections.
            opart = indices_for_outer_split(
                outer, neural_assign, repeat_index, outer_fold
            )
            for model_id in EXPECTED_ELIGIBLE_MODELS:
                seeds = (0,) if model_id == "BLOCK_PCA" else OUTER_MODEL_SEEDS
                for seed in seeds:
                    path = shard_path("outer", split_id, model_id, int(seed), 0)
                    if path.is_file():
                        result = load_json(path)
                    else:
                        result = evaluate_model_split(
                            model_id,
                            matrices,
                            case_ids,
                            opart["fit"],
                            opart["early"],
                            opart["train_all"],
                            opart["test"],
                            int(seed if seed != 0 else SMOKE_SEED),
                            modules,
                            protocol,
                            profile,
                            variants,
                            smoke=False,
                        )
                        result.update(
                            {
                                "script_version": SCRIPT_VERSION,
                                "script_sha256": current_runner_sha(),
                                "stage": "outer_evaluation",
                                "split_id": split_id,
                                "repeat_index": repeat_index,
                                "outer_fold": outer_fold,
                                "inner_fold": 0,
                                "model_id": model_id,
                                "seed": int(seed),
                                "selected_by_inner_cv": model_id == selected_id,
                                "outcome_access": False,
                                "biology_access": False,
                                "q10_retention_applied": False,
                            }
                        )
                        save_or_verify_shard(path, result)
                    outer_eval_rows.append(result)
                    print(
                        f"{split_id} OUTER {model_id} seed={seed}: "
                        f"{'FAIL' if result.get('failed') else 'PASS'}"
                    )

    selection_table = pd.DataFrame(inner_selection_rows)
    outer_table = pd.DataFrame(outer_eval_rows)
    final_freeze = final_method_freeze(outer_selected)

    selection_path = REAL_RESULT_ROOT / "target_inner_one_se_selection_by_outer_split.csv"
    outer_path = REAL_RESULT_ROOT / "target_outer_fixed_model_benchmark.csv"
    summary_path = REAL_RESULT_ROOT / "target_outcome_blind_benchmark_summary.json"
    lock_path = REAL_CONTRACT_DIR / "target_outcome_blind_benchmark_results_lock.json"

    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_table.to_csv(selection_path, index=False)
    outer_table.to_csv(outer_path, index=False)

    summary = {
        "status": "06B_TARGET_OUTCOME_BLIND_REPRESENTATION_BENCHMARK_COMPLETE",
        "created_at_utc": utc_now_iso(),
        "script_version": SCRIPT_VERSION,
        "script_sha256": current_runner_sha(),
        "n_cases": 78,
        "eligible_models": list(EXPECTED_ELIGIBLE_MODELS),
        "nested_cv": {"outer_repeats": 5, "outer_folds": 5, "inner_folds": 4},
        "primary_utility_weights": PRIMARY_WEIGHTS,
        "one_se_rule": (
            "lowest-complexity eligible model with mean inner-CV utility >= "
            "best_mean - standard_error_of_best"
        ),
        "final_full_data_method_freeze": final_freeze,
        "target_input_provenance": input_provenance,
        "mofa_protocol": {
            "path": relative_path(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "guardrails": {
            "outcomes_accessed": False,
            "clinical_values_accessed": False,
            "biology_accessed": False,
            "technical_covariates_used_for_selection": False,
            "q10_060_used_for_cross_family_selection": False,
            "q10_040_used_for_cross_family_selection": False,
            "secondary_comparators_used_for_primary_selection": False,
        },
        "outputs": {
            "inner_selection": relative_path(selection_path),
            "outer_benchmark": relative_path(outer_path),
        },
    }
    write_json_atomic(summary_path, summary)

    lock = {
        "status": "06B_TARGET_OUTCOME_BLIND_REPRESENTATION_BENCHMARK_RESULTS_FROZEN",
        "created_at_utc": utc_now_iso(),
        "runner_source": {
            "path": relative_path(SCRIPT_PATH),
            "sha256": current_runner_sha(),
        },
        "06b1_authorization_lock": {
            "path": relative_path(AUTH_06B1),
            "sha256": sha256_file(AUTH_06B1),
        },
        "summary": {
            "path": relative_path(summary_path),
            "sha256": sha256_file(summary_path),
        },
        "inner_selection": {
            "path": relative_path(selection_path),
            "sha256": sha256_file(selection_path),
        },
        "outer_benchmark": {
            "path": relative_path(outer_path),
            "sha256": sha256_file(outer_path),
        },
        "rewrite_policy": (
            "06b scientific TARGET benchmark outputs are immutable. "
            "No outcome, biology, q10 threshold, or post-result model change may alter them."
        ),
    }
    write_json_atomic(lock_path, lock)

    print("=" * 104)
    print("06b TARGET outcome-blind representation benchmark: COMPLETE")
    print("=" * 104)
    print(f"Final method-freeze status: {final_freeze['status']}")
    print(f"Primary model id(s): {', '.join(final_freeze['primary_model_ids'])}")
    print("TARGET outcomes/clinical/biology accessed: NO")
    print("q10 retention applied during cross-family selection: NO")
    print(f"Summary: {relative_path(summary_path)}")
    print(f"Lock: {relative_path(lock_path)}")
    print("=" * 104)


def main() -> None:
    authorized, lock = authorization_state()
    if not authorized:
        assert_no_target_access_preauthorization()
        run_synthetic_selftest()
        return
    if lock is None:
        raise RuntimeError("Internal authorization-state inconsistency.")
    run_authorized_target_benchmark(lock)


if __name__ == "__main__":
    main()
