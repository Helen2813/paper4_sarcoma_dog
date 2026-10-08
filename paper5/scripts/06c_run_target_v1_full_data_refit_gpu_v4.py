#!/usr/bin/env python3
"""
Paper 5 - TARGET V1 all-78 full-data refit runner v3.

This version deliberately removes the runtime/profile inference that caused v1/v2
synthetic-only failures. It reuses the exact already-authorized 06b v4 public helper
path demonstrated by the frozen source:

    modules = v4.load_frozen_modules()
    profile = v4.parse_frozen_profile(modules["neural"])
    variants = v4.build_variant_map(modules["ablation"])
    modules["ablation"].train_fixed_epochs(
        modules["neural"], profile, variants["V1_SHARED_PRIVATE_AE"], ...
    )
    encoded = modules["ablation"].encode_numpy(...)
    patient_shared = encoded["patient_shared"]

Two-stage gate:
- before a matching 06c2_v4 authorization exists: synthetic-only self-test;
- after authorization of this exact SHA: real all-78 outcome-blind fit.

No clinical/outcome/biology file is referenced by this source.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


SCRIPT_VERSION = (
    "06c-run-target-v1-full-data-refit-gpu-v3-"
    "exact-authorized-v4-profile-variant-path-two-stage-no-cli"
)

EXPECTED = {
    "06c_contract_sha256": "8ab8516e30964f220d0f1204ca892e29ca337691e06cbcbf29315e1cb6aec2e5",
    "06c_lock_sha256": "1c3b4fb36706dc6afd0208fda23b1dc6d51c2953b4ccb9541099466aea13a374",
    "06c_freeze_script_sha256": "537b4856c8b24fd9aa22ea4f490423c4f418cf602f8f5cf6cb6e836cc0760e8d",
    "06b_v4_sha256": "4f69680a2d26d6c8e4bd5b337a5e94e6cdc36d149ea6a4b9af547b3855a93eab",
    "primary_model_id": "V1_SHARED_PRIVATE_AE",
    "profile_id": "ARCH_COMPACT__REG_WEAK",
    "n_cases": 78,
    "primary_seed": 1103,
    "reproducibility_seeds": [2207, 3301],
    "fixed_epochs": 400,
    "feature_budgets": {"RNA": 5000, "METHYLATION": 10000, "CNV": 5000},
    "pca_components": {"RNA": 10, "METHYLATION": 10, "CNV": 10},
    "shared_latent_dim": 5,
    "views": ("RNA", "METHYLATION", "CNV"),
}


# --------------------------------------------------------------------------------------
# Generic utilities
# --------------------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    return path


def require_hash(label: str, path: Path, expected: str) -> str:
    observed = sha256_file(path)
    if observed.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA mismatch.\n"
            f"Expected: {expected}\nObserved: {observed}\nPath: {path}"
        )
    return observed


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((json_safe(v) for v in value), key=lambda x: repr(x))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")


def import_module_from_path(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def relative_to_paper5(path: Path, paper5: Path) -> str:
    return path.resolve().relative_to(paper5.resolve()).as_posix()


def availability_description(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": [str(k) for k in value.keys()],
            "values": {
                str(k): (
                    {"shape": list(v.shape), "dtype": str(v.dtype)}
                    if isinstance(v, np.ndarray)
                    else {"type": type(v).__name__}
                )
                for k, v in value.items()
            },
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "items": [
                (
                    {"shape": list(v.shape), "dtype": str(v.dtype)}
                    if isinstance(v, np.ndarray)
                    else {"type": type(v).__name__}
                )
                for v in value
            ],
        }
    return {"type": type(value).__name__, "repr": repr(value)[:400]}


# --------------------------------------------------------------------------------------
# Exact frozen-chain verification
# --------------------------------------------------------------------------------------

def verify_frozen_chain(paper5: Path, script_path: Path) -> dict[str, Any]:
    contract_path = (
        paper5 / "method_contract" / "06c" / "target_v1_full_data_refit_contract.json"
    )
    lock_path = (
        paper5 / "method_contract" / "06c" / "06c_full_data_refit_contract_lock.json"
    )
    freeze_script = (
        paper5 / "scripts" / "06c_freeze_target_v1_full_data_refit_contract_v1.py"
    )
    v4_path = (
        paper5
        / "scripts"
        / "06b_run_target_outcome_blind_representation_benchmark_gpu_v4.py"
    )

    require_hash("06c contract", contract_path, EXPECTED["06c_contract_sha256"])
    require_hash("06c lock", lock_path, EXPECTED["06c_lock_sha256"])
    require_hash(
        "06c freeze script",
        freeze_script,
        EXPECTED["06c_freeze_script_sha256"],
    )
    require_hash("authorized 06b v4", v4_path, EXPECTED["06b_v4_sha256"])

    contract = load_json(contract_path)
    lock = load_json(lock_path)

    if contract.get("status") != "FROZEN_BEFORE_TARGET_FULL_DATA_REFIT":
        raise RuntimeError("06c contract status mismatch.")
    if lock.get("status") != "06C_TARGET_V1_FULL_DATA_REFIT_CONTRACT_FROZEN":
        raise RuntimeError("06c lock status mismatch.")
    if lock.get("full_data_fit_performed") is not False:
        raise RuntimeError("06c no longer proves a pre-full-data-fit state.")

    full = contract.get("full_data_refit", {})
    training = full.get("training", {})
    if full.get("model_id") != EXPECTED["primary_model_id"]:
        raise RuntimeError("Frozen model mismatch.")
    if full.get("profile_id") != EXPECTED["profile_id"]:
        raise RuntimeError("Frozen profile mismatch.")
    if int(full.get("n_cases")) != EXPECTED["n_cases"]:
        raise RuntimeError("Frozen n_cases mismatch.")
    if {k: int(v) for k, v in full.get("feature_budget", {}).items()} != EXPECTED["feature_budgets"]:
        raise RuntimeError("Frozen feature-budget mismatch.")
    if int(training.get("primary_seed")) != EXPECTED["primary_seed"]:
        raise RuntimeError("Frozen primary seed mismatch.")
    if [int(x) for x in training.get("reproducibility_seeds", [])] != EXPECTED["reproducibility_seeds"]:
        raise RuntimeError("Frozen reproducibility seeds mismatch.")
    if int(training.get("fixed_epochs")) != EXPECTED["fixed_epochs"]:
        raise RuntimeError("Frozen fixed-epoch mismatch.")
    if training.get("early_stopping_on_full_78") is not False:
        raise RuntimeError("Frozen early-stopping firewall mismatch.")
    if training.get("post_fit_seed_selection") is not False:
        raise RuntimeError("Frozen seed-selection firewall mismatch.")
    if training.get("post_fit_epoch_selection") is not False:
        raise RuntimeError("Frozen epoch-selection firewall mismatch.")

    boundary = contract.get("sable_retention_boundary", {})
    if boundary.get("q10_060_applied_to_v1") is not False:
        raise RuntimeError("q10=.60 must not be applied to V1.")
    if boundary.get("q10_040_applied_to_v1") is not False:
        raise RuntimeError("q10=.40 must not be applied to V1.")
    if boundary.get("sable_mo_stability_gate_applied_to_v1") is not False:
        raise RuntimeError("SABLE-MO stability gate must not be applied to V1.")

    return {
        "contract_path": contract_path,
        "lock_path": lock_path,
        "v4_path": v4_path,
        "contract": contract,
        "lock": lock,
        "script_sha256": sha256_file(script_path),
    }


# --------------------------------------------------------------------------------------
# Exact authorized-v4 runtime path
# --------------------------------------------------------------------------------------

def load_exact_runtime(state: dict[str, Any]) -> dict[str, Any]:
    v4 = import_module_from_path("paper5_06b_v4_for_06c_v3", state["v4_path"])

    required_v4 = [
        "load_frozen_modules",
        "parse_frozen_profile",
        "build_variant_map",
        "fit_preprocessing_bundle",
        "transform_bundle",
        "availability_candidates",
        "synthetic_matrices",
        "load_authorized_target_inputs",
    ]
    for name in required_v4:
        if not callable(getattr(v4, name, None)):
            raise RuntimeError(f"Authorized 06b v4 helper missing/not callable: {name}")

    modules = v4.load_frozen_modules()
    if not isinstance(modules, dict):
        raise RuntimeError("Authorized v4 load_frozen_modules() did not return a dict.")

    required_module_keys = {"neural", "ablation"}
    if not required_module_keys.issubset(modules):
        raise RuntimeError(
            f"Authorized v4 frozen modules missing keys: {sorted(required_module_keys - set(modules))}"
        )

    neural = modules["neural"]
    ablation = modules["ablation"]

    for name in ("train_fixed_epochs", "encode_numpy"):
        if not callable(getattr(ablation, name, None)):
            raise RuntimeError(f"Frozen ablation module missing helper: {name}")
    if not callable(getattr(neural, "parse_profile", None)):
        raise RuntimeError("Frozen neural helper missing parse_profile().")

    # This is the exact profile path used by both frozen v4 callers shown in the source audit.
    profile = v4.parse_frozen_profile(neural)
    if getattr(profile, "profile_id", None) != EXPECTED["profile_id"]:
        raise RuntimeError(
            f"Parsed frozen profile mismatch: {getattr(profile, 'profile_id', None)!r}"
        )
    if int(getattr(profile, "maximum_epochs")) != EXPECTED["fixed_epochs"]:
        raise RuntimeError("Parsed frozen profile maximum_epochs mismatch.")
    if int(getattr(profile, "shared_dim")) != EXPECTED["shared_latent_dim"]:
        raise RuntimeError("Parsed frozen profile shared_dim mismatch.")

    # This is the exact variant path used by frozen v4 callers.
    variants = v4.build_variant_map(ablation)
    if EXPECTED["primary_model_id"] not in variants:
        raise RuntimeError(
            f"Frozen variant map lacks {EXPECTED['primary_model_id']}."
        )
    variant = variants[EXPECTED["primary_model_id"]]
    if getattr(variant, "variant_id", None) != EXPECTED["primary_model_id"]:
        raise RuntimeError("Frozen V1 variant identity mismatch.")

    # Exact v4 constants must still agree with the already-frozen 06c contract.
    feature_budgets = {
        k: int(v) for k, v in getattr(v4, "FEATURE_BUDGETS").items()
    }
    pca_components = {
        k: int(v) for k, v in getattr(v4, "NEURAL_PCA_COMPONENTS").items()
    }
    if feature_budgets != EXPECTED["feature_budgets"]:
        raise RuntimeError(
            f"Authorized v4 FEATURE_BUDGETS changed: {feature_budgets}"
        )
    if pca_components != EXPECTED["pca_components"]:
        raise RuntimeError(
            f"Authorized v4 NEURAL_PCA_COMPONENTS changed: {pca_components}"
        )

    return {
        "v4": v4,
        "modules": modules,
        "neural": neural,
        "ablation": ablation,
        "profile": profile,
        "variants": variants,
        "variant": variant,
        "feature_budgets": feature_budgets,
        "pca_components": pca_components,
    }


def fit_preprocess_exact(
    runtime: dict[str, Any],
    matrices: dict[str, np.ndarray],
    indices: np.ndarray,
) -> tuple[Any, dict[str, np.ndarray]]:
    v4 = runtime["v4"]
    prep = v4.fit_preprocessing_bundle(
        matrices,
        indices,
        runtime["feature_budgets"],
        runtime["pca_components"],
    )
    views_raw = v4.transform_bundle(prep, matrices, indices, use_pca=True)
    views = {k: np.asarray(v, dtype=np.float32) for k, v in views_raw.items()}

    if set(views) != set(EXPECTED["views"]):
        raise RuntimeError(f"Transformed view set mismatch: {sorted(views)}")
    for view in EXPECTED["views"]:
        arr = views[view]
        expected_shape = (len(indices), EXPECTED["pca_components"][view])
        if arr.shape != expected_shape:
            raise RuntimeError(
                f"{view} transformed geometry mismatch: "
                f"expected={expected_shape}, observed={arr.shape}"
            )
        if not np.isfinite(arr).all():
            raise RuntimeError(f"{view} contains non-finite transformed values.")

    return prep, views


def train_v1_fixed_first_success(
    runtime: dict[str, Any],
    views: dict[str, np.ndarray],
    *,
    seed: int,
    epochs: int,
) -> tuple[Any, dict[str, Any], np.ndarray, dict[str, Any]]:
    """
    Reproduce the exact adapter semantics already used by authorized 06b v4.

    Frozen v4 does NOT require a unique working availability representation.
    It iterates availability_candidates(...) in their frozen order and BREAKS on
    the first train_fixed_epochs success.  Encoding is then resolved in a
    separate first-success loop.

    That exact first-success behavior is reused here; no candidate index is
    selected using fit quality or downstream results.
    """
    v4 = runtime["v4"]
    ablation = runtime["ablation"]
    neural = runtime["neural"]
    profile = runtime["profile"]
    variant = runtime["variant"]

    fit_model = None
    fit_audit = None
    fit_availability = None
    fit_availability_index = None
    fit_failures: list[dict[str, Any]] = []

    for idx, availability in enumerate(v4.availability_candidates(views)):
        try:
            # Exact 05d4 return schema proven from source:
            #     return model, audit
            model, audit = ablation.train_fixed_epochs(
                neural,
                profile,
                variant,
                views,
                availability,
                int(seed),
                int(epochs),
            )
            fit_model = model
            fit_audit = audit
            fit_availability = availability
            fit_availability_index = int(idx)
            break  # exact authorized-v4 first-success semantics
        except Exception as exc:
            fit_failures.append(
                {
                    "candidate_index": int(idx),
                    "availability": availability_description(availability),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    if fit_model is None or fit_audit is None or fit_availability_index is None:
        raise RuntimeError(
            "V1 fixed-epoch refit failed for every frozen availability candidate. "
            + json.dumps(json_safe(fit_failures), indent=2)
        )

    encoded = None
    encode_availability = None
    encode_availability_index = None
    encode_failures: list[dict[str, Any]] = []

    # Authorized v4 performs a distinct first-success availability loop for
    # encode/reconstruction after fixed-epoch fitting.  Reproduce that boundary
    # rather than assuming the fitting candidate must also be the encoding candidate.
    for idx, availability in enumerate(v4.availability_candidates(views)):
        try:
            candidate = ablation.encode_numpy(
                fit_model,
                views,
                availability,
            )
            if not isinstance(candidate, dict):
                raise RuntimeError(
                    f"encode_numpy returned {type(candidate).__name__}, expected dict."
                )
            if "patient_shared" not in candidate:
                raise RuntimeError("encode_numpy output lacks patient_shared.")

            shared = np.asarray(
                candidate["patient_shared"],
                dtype=np.float64,
            )
            expected_shape = (
                next(iter(views.values())).shape[0],
                EXPECTED["shared_latent_dim"],
            )
            if shared.shape != expected_shape:
                raise RuntimeError(
                    f"patient_shared geometry mismatch: "
                    f"expected={expected_shape}, observed={shared.shape}"
                )
            if not np.isfinite(shared).all():
                raise RuntimeError("patient_shared contains non-finite values.")

            encoded = candidate
            encode_availability = availability
            encode_availability_index = int(idx)
            patient_shared = shared
            break  # exact authorized-v4 first-success semantics
        except Exception as exc:
            encode_failures.append(
                {
                    "candidate_index": int(idx),
                    "availability": availability_description(availability),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    if encoded is None or encode_availability_index is None:
        raise RuntimeError(
            "V1 encode_numpy failed for every frozen availability candidate. "
            + json.dumps(json_safe(encode_failures), indent=2)
        )

    adapter_audit = {
        "fit_availability_candidate_index": fit_availability_index,
        "fit_availability": availability_description(fit_availability),
        "fit_candidate_failures_before_success": fit_failures,
        "encode_availability_candidate_index": encode_availability_index,
        "encode_availability": availability_description(encode_availability),
        "encode_candidate_failures_before_success": encode_failures,
        "fit_audit": fit_audit,
        "patient_shared_shape": list(patient_shared.shape),
        "encode_keys": sorted(str(k) for k in encoded.keys()),
        "selection_semantics": (
            "Exact authorized 06b v4 adapter semantics: iterate frozen "
            "availability_candidates in order and break on first successful "
            "train_fixed_epochs; resolve encode_numpy in a separate ordered "
            "first-success loop."
        ),
    }
    return fit_model, fit_audit, patient_shared, adapter_audit



# --------------------------------------------------------------------------------------
# Synthetic-only preauthorization stage
# --------------------------------------------------------------------------------------

def synthetic_selftest(
    state: dict[str, Any],
    runtime: dict[str, Any],
    paper5: Path,
) -> None:
    print("=" * 104)
    print("Paper 5 - TARGET V1 all-78 runner v4 synthetic preauthorization self-test")
    print("=" * 104)
    print("Script version:", SCRIPT_VERSION)
    print("Runner SHA-256:", state["script_sha256"])
    print("Execution mode: SYNTHETIC_ONLY_PREAUTHORIZATION")
    print("Real TARGET molecular matrix load: FORBIDDEN")
    print("TARGET outcomes/clinical/biology access: FORBIDDEN")

    preflight_dir = paper5 / "method_contract" / "06c_preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    report_path = (
        preflight_dir / "target_v1_full_data_runner_synthetic_selftest_v4.json"
    )

    if report_path.exists():
        existing = load_json(report_path)
        if (
            existing.get("status")
            == "06C_V4_TARGET_V1_FULL_DATA_RUNNER_SYNTHETIC_SELFTEST_PASS"
            and str(existing.get("runner_sha256", "")).lower()
            == state["script_sha256"].lower()
        ):
            print("Existing exact-SHA synthetic PASS report: VERIFIED")
            print("Real TARGET molecular values accessed: NO")
            print("TARGET scientific full-data fit performed: NO")
            print("=" * 104)
            print("06c v4 TARGET V1 full-data runner synthetic self-test: PASS")
            print("=" * 104)
            print("Self-test report:", relative_to_paper5(report_path, paper5))
            return
        raise RuntimeError(
            "A v4 preflight report already exists but is not an exact PASS "
            f"for this source: {report_path}"
        )

    matrices_raw = runtime["v4"].synthetic_matrices()
    matrices = {k: np.asarray(v) for k, v in matrices_raw.items()}
    n = next(iter(matrices.values())).shape[0]
    if any(arr.shape[0] != n for arr in matrices.values()):
        raise RuntimeError(
            "Authorized v4 synthetic matrices have inconsistent row counts."
        )

    indices = np.arange(n, dtype=np.int64)
    _, views = fit_preprocess_exact(runtime, matrices, indices)

    model, fit_audit, shared, adapter_audit = train_v1_fixed_first_success(
        runtime,
        views,
        seed=EXPECTED["primary_seed"],
        epochs=2,  # compatibility-only smoke run; not a scientific epoch choice
    )

    # The user-observed v3 failure demonstrated the exact candidate ordering:
    # dict(bool) fails, ndarray(bool) succeeds, then later candidates are never
    # reached by authorized-v4 first-success semantics.  We do not require
    # uniqueness; we verify the first-success boundary itself.
    if int(adapter_audit["fit_availability_candidate_index"]) != 1:
        raise RuntimeError(
            "Authorized-v4 first-success fitting boundary changed. "
            f"Expected candidate index 1 from the frozen candidate ordering, "
            f"observed {adapter_audit['fit_availability_candidate_index']}."
        )
    if int(adapter_audit["encode_availability_candidate_index"]) != 1:
        raise RuntimeError(
            "Authorized-v4 first-success encoding boundary changed. "
            f"Expected candidate index 1, "
            f"observed {adapter_audit['encode_availability_candidate_index']}."
        )

    try:
        import torch  # type: ignore
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    report = {
        "status": "06C_V4_TARGET_V1_FULL_DATA_RUNNER_SYNTHETIC_SELFTEST_PASS",
        "script_version": SCRIPT_VERSION,
        "runner_sha256": state["script_sha256"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "real_target_molecular_values_read": False,
        "target_outcomes_clinical_biology_read": False,
        "target_scientific_fit_performed": False,
        "06c_identity": {
            "contract_sha256": EXPECTED["06c_contract_sha256"],
            "lock_sha256": EXPECTED["06c_lock_sha256"],
        },
        "authorized_v4_identity": {
            "sha256": EXPECTED["06b_v4_sha256"],
        },
        "exact_reuse_path": {
            "modules": {
                "neural": "modules['neural']",
                "ablation": "modules['ablation']",
            },
            "profile": "v4.parse_frozen_profile(modules['neural'])",
            "variant": (
                "v4.build_variant_map(modules['ablation'])"
                "['V1_SHARED_PRIVATE_AE']"
            ),
            "preprocessing": (
                "v4.fit_preprocessing_bundle(..., FEATURE_BUDGETS, "
                "NEURAL_PCA_COMPONENTS)"
            ),
            "fixed_fit": (
                "modules['ablation'].train_fixed_epochs("
                "modules['neural'], profile, variant, ...)"
            ),
            "encoding": (
                "modules['ablation'].encode_numpy(...)["
                "'patient_shared']"
            ),
            "availability_semantics": (
                "Exact authorized 06b v4 semantics: iterate "
                "availability_candidates in frozen order and stop at the first "
                "successful adapter; fitting and encoding use separate loops."
            ),
        },
        "synthetic": {
            "n_samples": n,
            "model_input_shapes": {
                k: list(v.shape) for k, v in views.items()
            },
            "fit_audit": fit_audit,
            "adapter_audit": adapter_audit,
            "patient_shared_shape": list(shared.shape),
            "smoke_epochs": 2,
            "smoke_epochs_are_scientific_setting": False,
        },
        "historical_runner_failures": {
            "v1": {
                "sha256": "06842dc803d17c3ea3789262655429481c081232ed8828b6e0d2cce56cb2703f",
                "boundary": "synthetic-only before real TARGET molecular load",
                "cause": (
                    "incorrect uniqueness assumption: both modules['neural'] "
                    "and modules['ablation'] expose train_fixed_epochs+encode_numpy"
                ),
                "scientific_consequence": "NONE",
            },
            "v2": {
                "sha256": "8901ea9dd3870ade3910934fba6106ae4b0fa4b8a6d2183e0cea9dd802962f9c",
                "boundary": "synthetic-only before real TARGET molecular load",
                "cause": (
                    "profile-shape inference was unnecessary; frozen v4 already "
                    "defines parse_frozen_profile(modules['neural'])"
                ),
                "scientific_consequence": "NONE",
            },
            "v3": {
                "sha256": "9d27534767434b1c8e41f46731ea8cd48c16fbdf1813ea3d85cf6a0cdfef1777",
                "boundary": "synthetic-only before real TARGET molecular load",
                "cause": (
                    "incorrectly required a unique working availability "
                    "representation even though authorized 06b v4 explicitly "
                    "uses ordered first-success semantics"
                ),
                "scientific_consequence": "NONE",
            },
        },
        "real_scientific_contract": {
            "model_id": EXPECTED["primary_model_id"],
            "profile_id": EXPECTED["profile_id"],
            "n_cases": EXPECTED["n_cases"],
            "feature_budgets": EXPECTED["feature_budgets"],
            "pca_components": EXPECTED["pca_components"],
            "primary_seed": EXPECTED["primary_seed"],
            "reproducibility_only_seeds": EXPECTED["reproducibility_seeds"],
            "fixed_epochs": EXPECTED["fixed_epochs"],
            "post_fit_seed_selection": False,
            "q10_transfer_to_v1": False,
        },
    }
    write_json(report_path, report)

    print("Exact 06c contract/lock identity: PASS")
    print("Exact authorized 06b v4 source identity: PASS")
    print("Exact frozen profile path: PASS")
    print("Exact frozen V1 variant path: PASS")
    print("Exact v4 full-cohort preprocessing path on synthetic data: PASS")
    print("Exact authorized-v4 first-success availability semantics: PASS")
    print("Exact train_fixed_epochs -> encode_numpy['patient_shared'] path: PASS")
    print(
        "Fit availability candidate index:",
        adapter_audit["fit_availability_candidate_index"],
    )
    print(
        "Encode availability candidate index:",
        adapter_audit["encode_availability_candidate_index"],
    )
    print("Synthetic patient_shared shape:", tuple(shared.shape))
    print("Real TARGET molecular values accessed: NO")
    print("TARGET scientific full-data fit performed: NO")
    print("=" * 104)
    print("06c v4 TARGET V1 full-data runner synthetic self-test: PASS")
    print("=" * 104)
    print("Self-test report:", relative_to_paper5(report_path, paper5))
    print("Next: freeze this exact source SHA in a separate 06c2_v4 authorization.")
    print("=" * 104)


# --------------------------------------------------------------------------------------
# Authorization gate
# --------------------------------------------------------------------------------------

def load_authorization(
    paper5: Path,
    runner_sha256: str,
) -> dict[str, Any] | None:
    auth_path = (
        paper5
        / "method_contract"
        / "06c2_v4"
        / "target_v1_full_data_runner_v4_source_authorization_lock.json"
    )
    if not auth_path.exists():
        return None

    auth = load_json(auth_path)
    if auth.get("status") != "06C2_V4_TARGET_V1_FULL_DATA_RUNNER_SOURCE_AUTHORIZED":
        raise RuntimeError("06c2_v4 authorization status mismatch.")
    if (
        str(auth.get("runner_source", {}).get("sha256", "")).lower()
        != runner_sha256.lower()
    ):
        raise RuntimeError("06c2_v3 authorizes a different runner SHA.")

    if auth.get("target_molecular_values_read") is not False:
        raise RuntimeError("06c2_v4 molecular firewall mismatch.")
    if auth.get("target_model_fit") is not False:
        raise RuntimeError("06c2_v4 fit firewall mismatch.")

    return auth


# --------------------------------------------------------------------------------------
# Real all-78 execution (reachable only after exact-source authorization)
# --------------------------------------------------------------------------------------

def extract_case_ids(raw: Any) -> list[str]:
    if isinstance(raw, np.ndarray):
        values = raw.tolist()
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        raise RuntimeError(
            f"Unsupported case-id container: {type(raw).__name__}"
        )
    return [str(x) for x in values]


def save_model_state_cpu(model: Any, path: Path) -> None:
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError("torch is required to save model state.") from exc

    if not hasattr(model, "state_dict"):
        raise RuntimeError("Trained model lacks state_dict().")
    cpu_state = {
        str(k): v.detach().cpu()
        for k, v in model.state_dict().items()
    }
    torch.save(cpu_state, path)


def write_case_order(path: Path, case_ids: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_index", "case_id"])
        for i, cid in enumerate(case_ids):
            w.writerow([i, cid])


def write_representation_csv(
    path: Path,
    case_ids: list[str],
    shared: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["case_id"]
            + [f"shared_{i+1}" for i in range(shared.shape[1])]
        )
        for cid, row in zip(case_ids, shared):
            w.writerow(
                [cid] + [format(float(x), ".12g") for x in row]
            )


def run_real(
    state: dict[str, Any],
    runtime: dict[str, Any],
    paper5: Path,
    authorization: dict[str, Any],
) -> None:
    print("=" * 104)
    print("Paper 5 - AUTHORIZED outcome-blind TARGET V1 all-78 full-data refit")
    print("=" * 104)
    print("Script version:", SCRIPT_VERSION)
    print("Authorized runner SHA-256:", state["script_sha256"])
    print("Model:", EXPECTED["primary_model_id"])
    print("Profile:", EXPECTED["profile_id"])
    print("Primary seed:", EXPECTED["primary_seed"])
    print("Reproducibility-only seeds:", EXPECTED["reproducibility_seeds"])
    print("Fixed epochs:", EXPECTED["fixed_epochs"])
    print("TARGET outcomes/clinical/biology access: FORBIDDEN")
    print("q10=.60/.40 transfer to V1: FORBIDDEN")

    preflight_path = (
        paper5
        / "method_contract"
        / "06c_preflight"
        / "target_v1_full_data_runner_synthetic_selftest_v4.json"
    )
    preflight = load_json(preflight_path)
    if (
        preflight.get("status")
        != "06C_V4_TARGET_V1_FULL_DATA_RUNNER_SYNTHETIC_SELFTEST_PASS"
    ):
        raise RuntimeError("Exact v4 synthetic preauthorization PASS is missing.")

    result_dir = (
        paper5
        / "results"
        / "target_representation"
        / "06c_full_data_v1"
    )
    result_lock_dir = paper5 / "method_contract" / "06c3"

    if result_dir.exists() and any(result_dir.iterdir()):
        raise RuntimeError(
            "Scientific 06c full-data result directory already exists and is non-empty. "
            "Do not overwrite or silently rerun."
        )
    if result_lock_dir.exists() and any(result_lock_dir.iterdir()):
        raise RuntimeError(
            "06c3 results-lock directory already exists and is non-empty."
        )

    # Real TARGET molecular values are first loaded here, after exact-source authorization.
    load_inputs = runtime["v4"].load_authorized_target_inputs
    case_ids_raw, matrices_raw, input_provenance = load_inputs()

    case_ids = extract_case_ids(case_ids_raw)
    if len(case_ids) != EXPECTED["n_cases"]:
        raise RuntimeError(
            f"Expected {EXPECTED['n_cases']} cases, observed {len(case_ids)}."
        )
    if len(set(case_ids)) != EXPECTED["n_cases"]:
        raise RuntimeError("TARGET case IDs are not unique.")

    matrices = {
        k: np.asarray(v)
        for k, v in matrices_raw.items()
    }
    if set(matrices) != set(EXPECTED["views"]):
        raise RuntimeError(
            f"TARGET modality set mismatch: {sorted(matrices)}"
        )
    for view in EXPECTED["views"]:
        if matrices[view].shape[0] != EXPECTED["n_cases"]:
            raise RuntimeError(
                f"{view} row geometry mismatch: {matrices[view].shape}"
            )

    all_indices = np.arange(EXPECTED["n_cases"], dtype=np.int64)
    _, views = fit_preprocess_exact(
        runtime,
        matrices,
        all_indices,
    )

    # Only after all real input/preprocessing checks pass do we create scientific outputs.
    result_dir.mkdir(parents=True, exist_ok=False)
    result_lock_dir.mkdir(parents=True, exist_ok=False)

    case_order_path = result_dir / "target_v1_full_data_case_order.csv"
    write_case_order(case_order_path, case_ids)

    model_input_files = {}
    for view in EXPECTED["views"]:
        p = result_dir / f"target_v1_full_data_model_input_{view}.npy"
        np.save(p, views[view].astype(np.float32, copy=False))
        model_input_files[view] = {
            "path": p.name,
            "sha256": sha256_file(p),
            "size_bytes": p.stat().st_size,
            "shape": list(views[view].shape),
            "dtype": str(views[view].dtype),
        }

    seed_records = []
    started_total = time.time()
    all_seeds = [
        EXPECTED["primary_seed"],
        *EXPECTED["reproducibility_seeds"],
    ]

    for seed in all_seeds:
        role = (
            "PRIMARY"
            if seed == EXPECTED["primary_seed"]
            else "REPRODUCIBILITY_ONLY"
        )
        print(
            f"V1 all-78 seed={seed} role={role}: "
            f"FIT fixed {EXPECTED['fixed_epochs']} epochs"
        )
        started = time.time()

        model, fit_audit, shared, adapter_audit = train_v1_fixed_first_success(
            runtime,
            views,
            seed=seed,
            epochs=EXPECTED["fixed_epochs"],
        )

        npy_path = (
            result_dir
            / f"target_v1_patient_shared_seed{seed}.npy"
        )
        csv_path = (
            result_dir
            / f"target_v1_patient_shared_seed{seed}.csv"
        )
        state_path = (
            result_dir
            / f"target_v1_model_state_seed{seed}.pt"
        )
        audit_path = (
            result_dir
            / f"target_v1_fit_audit_seed{seed}.json"
        )

        np.save(
            npy_path,
            shared.astype(np.float64, copy=False),
        )
        write_representation_csv(
            csv_path,
            case_ids,
            shared,
        )
        save_model_state_cpu(model, state_path)
        write_json(
            audit_path,
            {
                "seed": seed,
                "role": role,
                "epochs": EXPECTED["fixed_epochs"],
                "fit_audit": fit_audit,
                "adapter_audit": adapter_audit,
            },
        )

        seed_records.append(
            {
                "seed": seed,
                "role": role,
                "epochs": EXPECTED["fixed_epochs"],
                "elapsed_seconds": time.time() - started,
                "patient_shared_shape": list(shared.shape),
                "representation_npy": {
                    "path": npy_path.name,
                    "sha256": sha256_file(npy_path),
                    "size_bytes": npy_path.stat().st_size,
                },
                "representation_csv": {
                    "path": csv_path.name,
                    "sha256": sha256_file(csv_path),
                    "size_bytes": csv_path.stat().st_size,
                },
                "model_state": {
                    "path": state_path.name,
                    "sha256": sha256_file(state_path),
                    "size_bytes": state_path.stat().st_size,
                },
                "fit_audit": {
                    "path": audit_path.name,
                    "sha256": sha256_file(audit_path),
                    "size_bytes": audit_path.stat().st_size,
                },
            }
        )

        try:
            import torch  # type: ignore
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        print(f"V1 all-78 seed={seed}: PASS")

    summary = {
        "status": "06C_TARGET_V1_FULL_DATA_REFIT_COMPLETE",
        "script_version": SCRIPT_VERSION,
        "runner_sha256": state["script_sha256"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds_total": time.time() - started_total,
        "model_id": EXPECTED["primary_model_id"],
        "profile_id": EXPECTED["profile_id"],
        "n_cases": EXPECTED["n_cases"],
        "feature_budgets": EXPECTED["feature_budgets"],
        "pca_components": EXPECTED["pca_components"],
        "fixed_epochs": EXPECTED["fixed_epochs"],
        "primary_seed": EXPECTED["primary_seed"],
        "reproducibility_only_seeds": EXPECTED["reproducibility_seeds"],
        "availability_semantics": "authorized_06b_v4_ordered_first_success",
        "post_fit_seed_selection_performed": False,
        "seed_ensemble_or_averaging_performed": False,
        "q10_060_applied_to_v1": False,
        "q10_040_applied_to_v1": False,
        "sable_mo_stability_gate_applied_to_v1": False,
        "outcomes_accessed": False,
        "clinical_values_accessed": False,
        "biological_annotations_accessed": False,
        "technical_covariates_used_for_fit": False,
        "case_order": {
            "path": case_order_path.name,
            "sha256": sha256_file(case_order_path),
            "size_bytes": case_order_path.stat().st_size,
        },
        "model_inputs": model_input_files,
        "input_provenance": input_provenance,
        "seed_outputs": seed_records,
        "primary_representation_rule": (
            "Use seed 1103 patient_shared representation. "
            "Seeds 2207/3301 are reproducibility-only and may not replace it."
        ),
        "interpretation_boundary": (
            "V1_SHARED_PRIVATE_AE is the frozen TARGET primary representation. "
            "SABLE-MO q10=.60/.40 and SABLE-MO stability retention are not "
            "transferred to V1."
        ),
    }
    summary_path = (
        result_dir
        / "target_v1_full_data_refit_summary.json"
    )
    write_json(summary_path, summary)

    result_files = {}
    for p in sorted(result_dir.iterdir()):
        if p.is_file():
            result_files[p.name] = {
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
            }

    auth_path = (
        paper5
        / "method_contract"
        / "06c2_v4"
        / "target_v1_full_data_runner_v4_source_authorization_lock.json"
    )
    results_lock = {
        "status": "06C3_TARGET_V1_FULL_DATA_REFIT_RESULTS_FROZEN",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_source": {
            "path": relative_to_paper5(Path(__file__).resolve(), paper5),
            "sha256": state["script_sha256"],
        },
        "source_authorization_lock": {
            "path": relative_to_paper5(auth_path, paper5),
            "sha256": sha256_file(auth_path),
        },
        "06c_contract_sha256": EXPECTED["06c_contract_sha256"],
        "06c_lock_sha256": EXPECTED["06c_lock_sha256"],
        "model_id": EXPECTED["primary_model_id"],
        "profile_id": EXPECTED["profile_id"],
        "primary_seed": EXPECTED["primary_seed"],
        "reproducibility_only_seeds": EXPECTED["reproducibility_seeds"],
        "fixed_epochs": EXPECTED["fixed_epochs"],
        "post_fit_seed_selection_performed": False,
        "q10_applied_to_v1": False,
        "outcomes_clinical_biology_accessed": False,
        "result_files": result_files,
    }
    results_lock_path = (
        result_lock_dir
        / "target_v1_full_data_refit_results_lock.json"
    )
    write_json(results_lock_path, results_lock)

    print("=" * 104)
    print("06c TARGET V1 all-78 full-data refit: COMPLETE")
    print("=" * 104)
    print("Primary representation seed: 1103")
    print("Reproducibility-only seeds: 2207, 3301")
    print("Post-fit seed selection: NO")
    print("q10=.60/.40 applied to V1: NO")
    print("TARGET outcomes/clinical/biology accessed: NO")
    print("Summary:", relative_to_paper5(summary_path, paper5))
    print("Results lock:", relative_to_paper5(results_lock_path, paper5))
    print("=" * 104)


def main() -> None:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    paper5 = project_root / "paper5"

    state = verify_frozen_chain(
        paper5,
        script_path,
    )
    authorization = load_authorization(
        paper5,
        state["script_sha256"],
    )

    # The frozen v4 source and helper modules may be imported in either mode.
    # v4.load_frozen_modules() imports code only; it does not load real TARGET
    # matrices. Real values can be reached only through run_real() ->
    # load_authorized_target_inputs(), and run_real is unreachable without 06c2_v3.
    runtime = load_exact_runtime(state)

    if authorization is None:
        synthetic_selftest(
            state,
            runtime,
            paper5,
        )
        return

    run_real(
        state,
        runtime,
        paper5,
        authorization,
    )


if __name__ == "__main__":
    main()
