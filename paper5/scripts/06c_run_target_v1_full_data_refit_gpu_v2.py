#!/usr/bin/env python3
"""
Paper 5 - TARGET V1 all-78 full-data execution runner.

Two-stage behavior:
1) Before 06c2 source authorization exists, this exact source can run ONLY a
   synthetic/runtime compatibility self-test. It does not load real TARGET matrices.
2) After 06c2 authorizes this exact source SHA, rerunning the unchanged file performs
   the frozen all-78 outcome-blind V1 refit defined by 06c.

Scientific choices are NOT made here:
- model = V1_SHARED_PRIVATE_AE
- profile = ARCH_COMPACT__REG_WEAK
- primary seed = 1103
- reproducibility-only seeds = 2207, 3301
- fixed epochs = 400
- primary feature budgets = RNA 5000 / METHYLATION 10000 / CNV 5000
- no SABLE-MO q10/stability rule is transferred to V1
"""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import numpy as np


SCRIPT_VERSION = (
    "06c-run-target-v1-full-data-refit-gpu-v2-runtime-module-role-resolution-"
    "reuse-authorized-06b-v4-and-frozen-05d4-two-stage-no-cli"
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
    "views": ["RNA", "METHYLATION", "CNV"],
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
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA256 mismatch.\nExpected: {expected}\nObserved: {actual}\nPath: {path}"
        )
    return actual


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((json_safe(v) for v in value), key=lambda x: repr(x))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def import_module_from_path(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def flatten_objects(value: Any, prefix: str = "root") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = [(prefix, value)]
    if isinstance(value, dict):
        for k, v in value.items():
            out.extend(flatten_objects(v, f"{prefix}[{k!r}]"))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            out.extend(flatten_objects(v, f"{prefix}[{i}]"))
    return out


def recursive_numeric_arrays(value: Any, prefix: str = "root") -> list[tuple[str, np.ndarray]]:
    found: list[tuple[str, np.ndarray]] = []
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            found.append((prefix, value))
        return found
    try:
        import torch  # type: ignore
        if isinstance(value, torch.Tensor):
            found.append((prefix, value.detach().cpu().numpy()))
            return found
    except Exception:
        pass
    if isinstance(value, dict):
        for k, v in value.items():
            found.extend(recursive_numeric_arrays(v, f"{prefix}[{k!r}]"))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            found.extend(recursive_numeric_arrays(v, f"{prefix}[{i}]"))
    return found


def extract_unique_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    matches = []
    for path, arr in recursive_numeric_arrays(value):
        if tuple(arr.shape) == tuple(shape):
            matches.append((path, np.asarray(arr, dtype=np.float32)))
    if len(matches) != 1:
        desc = [(p, tuple(a.shape)) for p, a in matches]
        raise RuntimeError(
            f"{label}: expected exactly one numeric array of shape {shape}; "
            f"found {len(matches)} matches: {desc}"
        )
    return matches[0][1]


def relative_to_paper5(path: Path, paper5: Path) -> str:
    return path.resolve().relative_to(paper5.resolve()).as_posix()


# --------------------------------------------------------------------------------------
# Frozen identity verification
# --------------------------------------------------------------------------------------

def verify_frozen_chain(paper5: Path, script_path: Path) -> dict[str, Any]:
    contract_path = paper5 / "method_contract" / "06c" / "target_v1_full_data_refit_contract.json"
    lock_path = paper5 / "method_contract" / "06c" / "06c_full_data_refit_contract_lock.json"
    freeze_script = paper5 / "scripts" / "06c_freeze_target_v1_full_data_refit_contract_v1.py"
    v4_path = paper5 / "scripts" / "06b_run_target_outcome_blind_representation_benchmark_gpu_v4.py"
    d4_path = paper5 / "scripts" / "05d4_fit_neural_ablation_benchmark_gpu_v1_2_verified.py"

    require_hash("06c contract", contract_path, EXPECTED["06c_contract_sha256"])
    require_hash("06c lock", lock_path, EXPECTED["06c_lock_sha256"])
    require_hash("06c freeze script", freeze_script, EXPECTED["06c_freeze_script_sha256"])
    require_hash("06b v4 runner", v4_path, EXPECTED["06b_v4_sha256"])
    require_file(d4_path)

    contract = load_json(contract_path)
    lock = load_json(lock_path)

    if contract.get("status") != "FROZEN_BEFORE_TARGET_FULL_DATA_REFIT":
        raise RuntimeError("06c contract status mismatch.")
    if lock.get("status") != "06C_TARGET_V1_FULL_DATA_REFIT_CONTRACT_FROZEN":
        raise RuntimeError("06c lock status mismatch.")
    if lock.get("full_data_fit_performed") is not False:
        raise RuntimeError("06c no longer proves pre-full-data-fit state.")

    full = contract.get("full_data_refit", {})
    training = full.get("training", {})
    if full.get("model_id") != EXPECTED["primary_model_id"]:
        raise RuntimeError("06c primary model mismatch.")
    if full.get("profile_id") != EXPECTED["profile_id"]:
        raise RuntimeError("06c profile mismatch.")
    if int(full.get("n_cases")) != EXPECTED["n_cases"]:
        raise RuntimeError("06c n_cases mismatch.")
    if {k: int(v) for k, v in full.get("feature_budget", {}).items()} != EXPECTED["feature_budgets"]:
        raise RuntimeError("06c feature-budget mismatch.")
    if int(training.get("primary_seed")) != EXPECTED["primary_seed"]:
        raise RuntimeError("06c primary seed mismatch.")
    if [int(x) for x in training.get("reproducibility_seeds", [])] != EXPECTED["reproducibility_seeds"]:
        raise RuntimeError("06c reproducibility seed mismatch.")
    if int(training.get("fixed_epochs")) != EXPECTED["fixed_epochs"]:
        raise RuntimeError("06c fixed epoch mismatch.")
    if training.get("early_stopping_on_full_78") is not False:
        raise RuntimeError("06c early-stopping firewall mismatch.")
    if training.get("post_fit_seed_selection") is not False:
        raise RuntimeError("06c post-fit seed-selection firewall mismatch.")
    if training.get("post_fit_epoch_selection") is not False:
        raise RuntimeError("06c post-fit epoch-selection firewall mismatch.")

    boundary = contract.get("sable_retention_boundary", {})
    if boundary.get("q10_060_applied_to_v1") is not False or boundary.get("q10_040_applied_to_v1") is not False:
        raise RuntimeError("06c q10-transfer firewall mismatch.")

    return {
        "contract_path": contract_path,
        "lock_path": lock_path,
        "v4_path": v4_path,
        "d4_path": d4_path,
        "contract": contract,
        "lock": lock,
        "script_sha256": sha256_file(script_path),
        "v4_sha256": sha256_file(v4_path),
        "d4_sha256": sha256_file(d4_path),
    }


# --------------------------------------------------------------------------------------
# Static resolver for the already-authorized 06b v4 preprocessing path
# --------------------------------------------------------------------------------------

def parse_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))


def direct_called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                names.add(f.id)
            elif isinstance(f, ast.Attribute):
                names.add(f.attr)
    return names


def string_constants(node: ast.AST) -> set[str]:
    return {
        n.value
        for n in ast.walk(node)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }


def resolve_preprocessing_function(v4_module: ModuleType, v4_path: Path) -> tuple[str, Any]:
    tree = parse_tree(v4_path)
    matches = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls = direct_called_names(node)
        strings = string_constants(node)
        if (
            "fit_view_transform" in calls
            and "transforms" in strings
            and "pcas" in strings
        ):
            matches.append(node.name)

    # Some Python source may build the return dict without literal string constants being
    # visible exactly as expected. Fall back to source/name semantics but still require unique.
    if len(matches) != 1:
        candidates = []
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = direct_called_names(node)
            if "fit_view_transform" in calls:
                candidates.append(node.name)
        matches = candidates

    if len(matches) != 1:
        raise RuntimeError(
            f"Could not uniquely resolve v4 preprocessing-fit function. Candidates={matches}"
        )
    name = matches[0]
    fn = getattr(v4_module, name, None)
    if not callable(fn):
        raise RuntimeError(f"Resolved v4 preprocessing function is not callable: {name}")
    return name, fn


def call_with_mapped_signature(
    fn: Any,
    *,
    matrices: dict[str, np.ndarray],
    indices: np.ndarray,
    budgets: dict[str, int],
    pca_components: dict[str, int],
    model_id: str,
    use_pca: bool,
) -> Any:
    sig = inspect.signature(fn)
    kwargs: dict[str, Any] = {}

    matrix_names = {"matrices", "arrays", "views", "data", "input_matrices"}
    index_names = {
        "train_indices", "training_indices", "indices", "fit_indices",
        "training_idx", "train_idx"
    }
    budget_names = {"budgets", "feature_budgets", "budget"}
    pca_names = {"pca_components", "neural_pca_components", "components_by_view"}
    use_pca_names = {"use_pca", "fit_pca", "include_pca", "with_pca"}
    model_names = {"model_id", "variant", "model"}
    family_names = {"family", "model_family"}

    for name, p in sig.parameters.items():
        lname = name.lower()
        if lname in matrix_names:
            kwargs[name] = matrices
        elif lname in index_names:
            kwargs[name] = indices
        elif lname in budget_names:
            kwargs[name] = budgets
        elif lname in pca_names:
            kwargs[name] = pca_components
        elif lname in use_pca_names:
            kwargs[name] = use_pca
        elif lname in model_names:
            kwargs[name] = model_id
        elif lname in family_names:
            kwargs[name] = "SABLE_MO_ablation"
        elif p.default is not inspect._empty:
            continue
        else:
            raise RuntimeError(
                f"Unmapped required parameter {name!r} for resolved preprocessing function "
                f"{fn.__name__}{sig}"
            )
    return fn(**kwargs)


def resolve_transform_bundle(v4_module: ModuleType) -> Any:
    fn = getattr(v4_module, "transform_bundle", None)
    if not callable(fn):
        raise RuntimeError("Authorized 06b v4 no longer exposes transform_bundle.")
    return fn


# --------------------------------------------------------------------------------------
# Frozen neural helper resolution
# --------------------------------------------------------------------------------------

def resolve_runtime_modules(v4_module: ModuleType) -> tuple[Any, Any, Any]:
    """
    Resolve the two distinct runtime roles used by authorized 06b v4 without guessing.

    Historical v1 incorrectly assumed that exactly one object returned by
    load_frozen_modules() would expose train_fixed_epochs+encode_numpy. On the real
    frozen stack both modules['neural'] and modules['ablation'] expose that API.

    v2 instead proves the roles from the already-authorized v4 source itself:
    1) inspect evaluate_neural(...);
    2) identify the parameter used as the receiver of .train_fixed_epochs(...);
    3) identify the parameter passed as its first positional runtime argument;
    4) inspect actual authorized v4 call sites of evaluate_neural(...);
    5) resolve those two arguments back to modules[...] objects;
    6) require all resolvable call sites to agree exactly.
    """
    load = getattr(v4_module, "load_frozen_modules", None)
    if not callable(load):
        raise RuntimeError("Authorized 06b v4 no longer exposes load_frozen_modules().")
    modules = load()

    eval_fn = getattr(v4_module, "evaluate_neural", None)
    if not callable(eval_fn):
        raise RuntimeError("Authorized 06b v4 no longer exposes evaluate_neural.")

    eval_source = inspect.getsource(eval_fn)
    eval_tree = ast.parse(eval_source)
    train_calls = []
    for n in ast.walk(eval_tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Attribute) and f.attr == "train_fixed_epochs":
            train_calls.append(n)
        elif isinstance(f, ast.Name) and f.id == "train_fixed_epochs":
            train_calls.append(n)

    if len(train_calls) != 1:
        raise RuntimeError(
            "Could not prove a unique train_fixed_epochs call inside authorized evaluate_neural. "
            f"Observed calls={len(train_calls)}"
        )

    train_call = train_calls[0]
    receiver_param = None
    if isinstance(train_call.func, ast.Attribute):
        base = train_call.func.value
        if not isinstance(base, ast.Name):
            raise RuntimeError(
                "Authorized evaluate_neural uses a non-simple receiver for train_fixed_epochs: "
                + ast.dump(base, include_attributes=False)
            )
        receiver_param = base.id

    if not train_call.args or not isinstance(train_call.args[0], ast.Name):
        raise RuntimeError(
            "Could not prove the first runtime argument passed to train_fixed_epochs "
            "inside authorized evaluate_neural."
        )
    first_runtime_param = train_call.args[0].id

    eval_sig = inspect.signature(eval_fn)
    eval_params = list(eval_sig.parameters)
    for p in [first_runtime_param] + ([receiver_param] if receiver_param else []):
        if p not in eval_params:
            raise RuntimeError(
                f"Resolved evaluate_neural role parameter {p!r} is not in signature {eval_sig}."
            )

    # Verify encode_numpy uses the same helper-receiver role when called as an attribute.
    encode_receivers = set()
    for n in ast.walk(eval_tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Attribute) and f.attr == "encode_numpy":
            if isinstance(f.value, ast.Name):
                encode_receivers.add(f.value.id)
            else:
                raise RuntimeError(
                    "Authorized evaluate_neural uses a non-simple receiver for encode_numpy."
                )
    if encode_receivers and encode_receivers != {receiver_param}:
        raise RuntimeError(
            "train_fixed_epochs and encode_numpy helper receiver roles disagree inside "
            f"authorized evaluate_neural: train={receiver_param!r}, encode={sorted(encode_receivers)!r}"
        )

    def eval_simple(expr: ast.AST) -> Any:
        if isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Name) and expr.value.id == "modules":
            sl = expr.slice
            if isinstance(sl, ast.Constant):
                return modules[sl.value]
            raise RuntimeError("Unsupported modules[...] subscript in authorized v4 caller.")
        if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name) and expr.value.id == "modules":
            return getattr(modules, expr.attr)
        raise RuntimeError(
            "Authorized v4 evaluate_neural caller argument cannot be resolved without "
            "executing arbitrary code: " + ast.dump(expr, include_attributes=False)
        )

    def argument_for_parameter(call: ast.Call, parameter_name: str) -> ast.AST | None:
        # Explicit keyword wins.
        for kw in call.keywords:
            if kw.arg == parameter_name:
                return kw.value
        pos = eval_params.index(parameter_name)
        if pos < len(call.args):
            return call.args[pos]
        return None

    v4_tree = parse_tree(Path(v4_module.__file__).resolve())
    resolved_pairs = []
    unresolved_call_dumps = []

    for node in v4_tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for n in ast.walk(node):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            is_eval = (
                isinstance(f, ast.Name) and f.id == "evaluate_neural"
            ) or (
                isinstance(f, ast.Attribute) and f.attr == "evaluate_neural"
            )
            if not is_eval:
                continue

            runtime_expr = argument_for_parameter(n, first_runtime_param)
            receiver_expr = (
                argument_for_parameter(n, receiver_param)
                if receiver_param is not None
                else None
            )
            if runtime_expr is None or (receiver_param is not None and receiver_expr is None):
                unresolved_call_dumps.append(ast.dump(n, include_attributes=False))
                continue

            try:
                runtime_obj = eval_simple(runtime_expr)
                if receiver_param is not None:
                    helper_obj = eval_simple(receiver_expr)  # type: ignore[arg-type]
                else:
                    # Direct function call form: resolve by exact function identity among modules.
                    helper_candidates = []
                    for path, obj in flatten_objects(modules):
                        fn = getattr(obj, "train_fixed_epochs", None)
                        if callable(fn):
                            helper_candidates.append((path, obj))
                    uniq = []
                    seen = set()
                    for path, obj in helper_candidates:
                        if id(obj) not in seen:
                            uniq.append((path, obj))
                            seen.add(id(obj))
                    if len(uniq) != 1:
                        raise RuntimeError(
                            "Direct train_fixed_epochs form could not be mapped uniquely to a frozen module. "
                            f"Candidates={[p for p, _ in uniq]}"
                        )
                    helper_obj = uniq[0][1]
                resolved_pairs.append((runtime_obj, helper_obj))
            except Exception:
                unresolved_call_dumps.append(ast.dump(n, include_attributes=False))

    if not resolved_pairs:
        raise RuntimeError(
            "No authorized v4 evaluate_neural call site could be resolved to frozen module roles. "
            f"Unresolved call count={len(unresolved_call_dumps)}"
        )

    # Require all resolved v4 call sites to agree by object identity.
    role_ids = {(id(runtime_obj), id(helper_obj)) for runtime_obj, helper_obj in resolved_pairs}
    if len(role_ids) != 1:
        raise RuntimeError(
            "Authorized v4 call sites disagree on neural-runtime vs helper-module roles."
        )

    neural_runtime, ablation_module = resolved_pairs[0]

    for name in ("train_fixed_epochs", "encode_numpy"):
        if not callable(getattr(ablation_module, name, None)):
            raise RuntimeError(
                f"Resolved helper module does not expose required frozen helper {name}."
            )

    return modules, neural_runtime, ablation_module


def choose_profile_object(ablation_module: Any, selected_profile: dict[str, Any]) -> Any:
    params = selected_profile.get("parameters", {})
    fn = getattr(ablation_module, "train_fixed_epochs")
    source = inspect.getsource(fn)
    tree = ast.parse(source)

    direct_keys = set()
    nested_parameters = False
    for n in ast.walk(tree):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "profile":
            sl = n.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                direct_keys.add(sl.value)
                if sl.value == "parameters":
                    nested_parameters = True

    known_parameter_keys = set(params)
    if direct_keys & known_parameter_keys:
        return params
    if nested_parameters:
        return selected_profile

    # construct_model is also allowed to reveal expected profile shape.
    construct = getattr(ablation_module, "construct_model", None)
    if callable(construct):
        try:
            src = inspect.getsource(construct)
            t = ast.parse(src)
            keys = {
                n.slice.value
                for n in ast.walk(t)
                if isinstance(n, ast.Subscript)
                and isinstance(n.value, ast.Name)
                and n.value.id == "profile"
                and isinstance(n.slice, ast.Constant)
                and isinstance(n.slice.value, str)
            }
            if keys & known_parameter_keys:
                return params
            if "parameters" in keys:
                return selected_profile
        except Exception:
            pass

    raise RuntimeError(
        "Could not prove whether frozen 05d4 expects selected_profile or selected_profile['parameters']."
    )


def fit_and_extract_shared(
    ablation_module: Any,
    neural_runtime: Any,
    profile_obj: Any,
    views: dict[str, np.ndarray],
    availability: np.ndarray,
    seed: int,
    epochs: int,
    expected_shared_dim: int,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    train_fn = getattr(ablation_module, "train_fixed_epochs")
    encode_fn = getattr(ablation_module, "encode_numpy")

    raw = train_fn(
        neural_runtime,
        profile_obj,
        EXPECTED["primary_model_id"],
        views,
        availability,
        int(seed),
        int(epochs),
    )

    # Resolve the trained model schema by requiring exactly one candidate object that
    # encode_numpy can accept and that yields exactly one shared array of expected shape.
    candidates = flatten_objects(raw)
    successful = []
    errors = []
    n = next(iter(views.values())).shape[0]
    target_shape = (n, expected_shared_dim)

    seen = set()
    for path, obj in candidates:
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, (dict, list, tuple, str, bytes, int, float, bool, np.ndarray)):
            continue
        try:
            encoded = encode_fn(obj, views, availability)
            shared = extract_unique_array(encoded, target_shape, f"encode_numpy candidate {path}")
            successful.append((path, obj, encoded, shared))
        except Exception as exc:
            errors.append((path, type(exc).__name__, str(exc)[:240]))

    if len(successful) != 1:
        raise RuntimeError(
            f"Expected exactly one trained-model object producing shared representation {target_shape}; "
            f"found {len(successful)}. Successful paths={[x[0] for x in successful]}; "
            f"sample errors={errors[:10]}"
        )

    path, model, encoded, shared = successful[0]
    audit = {
        "trained_model_container_path": path,
        "train_return_type": type(raw).__name__,
        "encoded_return_type": type(encoded).__name__,
        "shared_shape": list(shared.shape),
    }
    return model, shared, audit


# --------------------------------------------------------------------------------------
# Synthetic-only preauthorization
# --------------------------------------------------------------------------------------

def synthetic_selftest(state: dict[str, Any], paper5: Path, script_path: Path) -> None:
    print("=" * 104)
    print("Paper 5 - TARGET V1 all-78 runner synthetic preauthorization self-test")
    print("=" * 104)
    print("Script version:", SCRIPT_VERSION)
    print("Runner SHA-256:", state["script_sha256"])
    print("Execution mode: SYNTHETIC_ONLY_PREAUTHORIZATION")
    print("Real TARGET molecular matrix load: FORBIDDEN")
    print("TARGET outcomes/clinical/biology access: FORBIDDEN")

    v4 = import_module_from_path("paper5_06b_v4_for_06c", state["v4_path"])
    prep_name, prep_fn = resolve_preprocessing_function(v4, state["v4_path"])
    transform_bundle = resolve_transform_bundle(v4)
    modules, neural_runtime, ablation = resolve_runtime_modules(v4)

    contract = state["contract"]
    selected_profile = {
        "profile_id": contract["full_data_refit"]["profile_id"],
        "parameters": contract["full_data_refit"]["profile_parameters"],
    }
    profile_obj = choose_profile_object(ablation, selected_profile)

    rng = np.random.default_rng(619031)
    n = 24
    synth = {
        "RNA": rng.normal(size=(n, 40)).astype(np.float32),
        "METHYLATION": rng.normal(size=(n, 50)).astype(np.float32),
        "CNV": rng.normal(size=(n, 30)).astype(np.float32),
    }
    # Include deterministic finite missing entries to exercise median-imputation semantics.
    synth["RNA"][1, 2] = np.nan
    synth["METHYLATION"][3, 4] = np.nan
    synth["CNV"][5, 6] = np.nan

    idx = np.arange(n, dtype=np.int64)
    budgets = {"RNA": 20, "METHYLATION": 24, "CNV": 18}
    pcs = {"RNA": 5, "METHYLATION": 5, "CNV": 5}

    prep = call_with_mapped_signature(
        prep_fn,
        matrices=synth,
        indices=idx,
        budgets=budgets,
        pca_components=pcs,
        model_id=EXPECTED["primary_model_id"],
        use_pca=True,
    )
    views = transform_bundle(prep, synth, idx, use_pca=True)
    if set(views) != set(EXPECTED["views"]):
        raise RuntimeError(f"Synthetic transform bundle views mismatch: {views.keys()}")
    for view in EXPECTED["views"]:
        arr = np.asarray(views[view])
        if arr.shape[0] != n or not np.isfinite(arr).all():
            raise RuntimeError(f"Synthetic transformed view invalid: {view} {arr.shape}")

    # Use whatever PCA dimensionality the authorized v4 path actually produced.
    dims = {v: int(np.asarray(views[v]).shape[1]) for v in EXPECTED["views"]}
    availability = np.ones((n, len(EXPECTED["views"])), dtype=bool)

    # Only 2 epochs: compatibility test, not scientific estimation.
    model, shared, neural_audit = fit_and_extract_shared(
        ablation,
        neural_runtime,
        profile_obj,
        {k: np.asarray(v, dtype=np.float32) for k, v in views.items()},
        availability,
        seed=EXPECTED["primary_seed"],
        epochs=2,
        expected_shared_dim=EXPECTED["shared_latent_dim"],
    )
    if shared.shape != (n, EXPECTED["shared_latent_dim"]):
        raise RuntimeError("Synthetic shared representation geometry mismatch.")
    if not np.isfinite(shared).all():
        raise RuntimeError("Synthetic shared representation contains non-finite values.")

    report = {
        "status": "06C_TARGET_V1_FULL_DATA_RUNNER_SYNTHETIC_SELFTEST_PASS",
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
            "preprocessing_function": prep_name,
            "transform_bundle": "transform_bundle",
        },
        "synthetic_preprocessing": {
            "n": n,
            "budgets": budgets,
            "requested_pca_components": pcs,
            "observed_output_dims": dims,
        },
        "synthetic_neural": neural_audit,
        "historical_v1_boundary": {
            "runner_sha256": "06842dc803d17c3ea3789262655429481c081232ed8828b6e0d2cce56cb2703f",
            "failure_stage": "synthetic runtime module-role resolution before any real TARGET matrix load",
            "failure": "Both modules['neural'] and modules['ablation'] exposed train_fixed_epochs+encode_numpy; v1 required false uniqueness.",
            "real_target_molecular_values_read": False,
            "target_scientific_fit_performed": False,
            "scientific_consequence": "NONE"
        },
        "scientific_contract": {
            "model_id": EXPECTED["primary_model_id"],
            "primary_seed": EXPECTED["primary_seed"],
            "fixed_epochs_real_run": EXPECTED["fixed_epochs"],
            "q10_transfer_to_v1": False,
        },
    }

    preflight_dir = paper5 / "method_contract" / "06c_preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    report_path = preflight_dir / "target_v1_full_data_runner_synthetic_selftest_v2.json"
    write_json(report_path, report)

    print("Exact 06c contract/lock identity: PASS")
    print("Exact authorized 06b v4 source identity: PASS")
    print("v4 full-cohort preprocessing path on synthetic data: PASS")
    print("Frozen V1 train_fixed_epochs -> encode_numpy path: PASS")
    print("Synthetic shared representation shape:", tuple(shared.shape))
    print("Real TARGET molecular values accessed: NO")
    print("TARGET scientific full-data fit performed: NO")
    print("=" * 104)
    print("06c TARGET V1 full-data runner synthetic self-test: PASS")
    print("=" * 104)
    print("Self-test report:", relative_to_paper5(report_path, paper5))
    print("Next: authorize this exact runner source in 06c2_v2, then rerun the exact same source unchanged.")
    print("=" * 104)


# --------------------------------------------------------------------------------------
# Real authorized all-78 execution
# --------------------------------------------------------------------------------------

def load_authorization(paper5: Path, runner_sha: str) -> dict[str, Any] | None:
    auth_path = (
        paper5
        / "method_contract"
        / "06c2_v2"
        / "target_v1_full_data_runner_v2_source_authorization_lock.json"
    )
    if not auth_path.exists():
        return None
    auth = load_json(auth_path)
    if auth.get("status") != "06C2_V2_TARGET_V1_FULL_DATA_RUNNER_SOURCE_AUTHORIZED":
        raise RuntimeError("06c2 authorization status mismatch.")
    observed = str(auth.get("runner_source", {}).get("sha256", "")).lower()
    if observed != runner_sha.lower():
        raise RuntimeError(
            f"06c2 authorizes a different runner source. Authorized={observed}; current={runner_sha}"
        )
    return auth


def extract_case_ids(case_ids_raw: Any) -> list[str]:
    if isinstance(case_ids_raw, np.ndarray):
        vals = case_ids_raw.tolist()
    elif isinstance(case_ids_raw, (list, tuple)):
        vals = list(case_ids_raw)
    else:
        raise RuntimeError(f"Unsupported case-id container type: {type(case_ids_raw).__name__}")
    return [str(x) for x in vals]


def save_preprocessing_numeric_snapshot(prep: Any, out_path: Path) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    manifest: dict[str, Any] = {}

    def collect(prefix: str, obj: Any, depth: int = 0) -> None:
        if depth > 4:
            return
        if isinstance(obj, np.ndarray):
            key = prefix.replace(".", "__").replace("[", "_").replace("]", "").replace("'", "")
            arrays[key] = obj
            manifest[prefix] = {"type": "ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype)}
            return
        if isinstance(obj, np.generic):
            manifest[prefix] = obj.item()
            return
        if obj is None or isinstance(obj, (str, int, float, bool)):
            manifest[prefix] = obj
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                collect(f"{prefix}.{k}", v, depth + 1)
            return
        if isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                collect(f"{prefix}[{i}]", v, depth + 1)
            return
        if hasattr(obj, "__dict__"):
            manifest[prefix] = {"type": type(obj).__name__}
            for k, v in vars(obj).items():
                collect(f"{prefix}.{k}", v, depth + 1)
            return
        manifest[prefix] = {"type": type(obj).__name__, "repr": repr(obj)[:400]}

    collect("preprocessing", prep)
    np.savez_compressed(out_path, **arrays)
    return manifest


def save_model_state(model: Any, path: Path) -> None:
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError("torch is required to save the frozen neural model state.") from exc

    if not hasattr(model, "state_dict"):
        raise RuntimeError(f"Resolved trained model has no state_dict(): {type(model).__name__}")
    torch.save(model.state_dict(), path)


def write_representation_csv(path: Path, case_ids: list[str], shared: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id"] + [f"shared_{i+1}" for i in range(shared.shape[1])])
        for cid, row in zip(case_ids, shared):
            w.writerow([cid] + [format(float(x), ".10g") for x in row])


def run_real(state: dict[str, Any], paper5: Path, auth: dict[str, Any]) -> None:
    print("=" * 104)
    print("Paper 5 - authorized outcome-blind TARGET V1 all-78 full-data refit")
    print("=" * 104)
    print("Script version:", SCRIPT_VERSION)
    print("Authorized runner SHA-256:", state["script_sha256"])
    print("Frozen model:", EXPECTED["primary_model_id"])
    print("Primary seed:", EXPECTED["primary_seed"])
    print("Reproducibility-only seeds:", EXPECTED["reproducibility_seeds"])
    print("Fixed epochs:", EXPECTED["fixed_epochs"])
    print("TARGET outcomes/clinical/biology access: FORBIDDEN")
    print("SABLE-MO q10 transfer to V1: FORBIDDEN")

    result_dir = paper5 / "results" / "target_representation" / "06c_full_data_v1"
    lock_dir = paper5 / "method_contract" / "06c3"
    if result_dir.exists() and any(result_dir.iterdir()):
        raise RuntimeError(
            f"Scientific 06c full-data result directory already exists and is non-empty: {result_dir}\n"
            "Do not overwrite or silently rerun a completed/partial all-78 fit."
        )
    if lock_dir.exists() and any(lock_dir.iterdir()):
        raise RuntimeError(
            f"06c3 result-lock directory already exists and is non-empty: {lock_dir}"
        )
    result_dir.mkdir(parents=True, exist_ok=False)
    lock_dir.mkdir(parents=True, exist_ok=False)

    v4 = import_module_from_path("paper5_06b_v4_for_06c_real", state["v4_path"])
    prep_name, prep_fn = resolve_preprocessing_function(v4, state["v4_path"])
    transform_bundle = resolve_transform_bundle(v4)
    modules, neural_runtime, ablation = resolve_runtime_modules(v4)

    load_inputs = getattr(v4, "load_authorized_target_inputs", None)
    if not callable(load_inputs):
        raise RuntimeError("Authorized 06b v4 no longer exposes load_authorized_target_inputs().")

    t0 = time.time()
    case_ids_raw, matrices_raw, input_provenance = load_inputs()
    case_ids = extract_case_ids(case_ids_raw)
    if len(case_ids) != EXPECTED["n_cases"] or len(set(case_ids)) != EXPECTED["n_cases"]:
        raise RuntimeError(
            f"TARGET case identity mismatch: n={len(case_ids)}, unique={len(set(case_ids))}"
        )

    matrices = {k: np.asarray(v) for k, v in matrices_raw.items()}
    if set(matrices) != set(EXPECTED["views"]):
        raise RuntimeError(f"TARGET matrix views mismatch: {matrices.keys()}")
    for view in EXPECTED["views"]:
        if matrices[view].shape[0] != EXPECTED["n_cases"]:
            raise RuntimeError(f"{view} row count mismatch: {matrices[view].shape}")

    all_idx = np.arange(EXPECTED["n_cases"], dtype=np.int64)
    prep = call_with_mapped_signature(
        prep_fn,
        matrices=matrices,
        indices=all_idx,
        budgets=EXPECTED["feature_budgets"],
        pca_components=EXPECTED["pca_components"],
        model_id=EXPECTED["primary_model_id"],
        use_pca=True,
    )
    views = transform_bundle(prep, matrices, all_idx, use_pca=True)
    views = {k: np.asarray(v, dtype=np.float32) for k, v in views.items()}

    for view in EXPECTED["views"]:
        arr = views[view]
        expected_dim = EXPECTED["pca_components"][view]
        if arr.shape != (EXPECTED["n_cases"], expected_dim):
            raise RuntimeError(
                f"{view}: full-data V1 PCA-space geometry mismatch. "
                f"Expected {(EXPECTED['n_cases'], expected_dim)}, observed {arr.shape}"
            )
        if not np.isfinite(arr).all():
            raise RuntimeError(f"{view}: non-finite values remain after authorized full-data preprocessing.")

    selected_profile = {
        "profile_id": state["contract"]["full_data_refit"]["profile_id"],
        "parameters": state["contract"]["full_data_refit"]["profile_parameters"],
    }
    profile_obj = choose_profile_object(ablation, selected_profile)
    availability = np.ones((EXPECTED["n_cases"], len(EXPECTED["views"])), dtype=bool)

    # Freeze preprocessing once and reuse identically for all seeds.
    prep_npz = result_dir / "target_v1_full_data_preprocessing_numeric_snapshot.npz"
    prep_manifest = save_preprocessing_numeric_snapshot(prep, prep_npz)
    prep_manifest_path = result_dir / "target_v1_full_data_preprocessing_manifest.json"
    write_json(
        prep_manifest_path,
        {
            "preprocessing_function": prep_name,
            "feature_budgets": EXPECTED["feature_budgets"],
            "pca_components": EXPECTED["pca_components"],
            "numeric_snapshot": prep_npz.name,
            "numeric_snapshot_sha256": sha256_file(prep_npz),
            "objects": prep_manifest,
        },
    )

    seed_records = []
    seeds = [EXPECTED["primary_seed"]] + EXPECTED["reproducibility_seeds"]
    for seed in seeds:
        role = "PRIMARY" if seed == EXPECTED["primary_seed"] else "REPRODUCIBILITY_ONLY"
        print(f"V1 full-data seed={seed} role={role}: FIT {EXPECTED['fixed_epochs']} epochs")
        s0 = time.time()

        model, shared, audit = fit_and_extract_shared(
            ablation,
            neural_runtime,
            profile_obj,
            views,
            availability,
            seed=seed,
            epochs=EXPECTED["fixed_epochs"],
            expected_shared_dim=EXPECTED["shared_latent_dim"],
        )
        if not np.isfinite(shared).all():
            raise RuntimeError(f"seed {seed}: shared representation contains non-finite values.")

        npy_path = result_dir / f"target_v1_shared_representation_seed{seed}.npy"
        csv_path = result_dir / f"target_v1_shared_representation_seed{seed}.csv"
        model_path = result_dir / f"target_v1_model_state_seed{seed}.pt"

        np.save(npy_path, shared.astype(np.float32, copy=False))
        write_representation_csv(csv_path, case_ids, shared)
        save_model_state(model, model_path)

        rec = {
            "seed": seed,
            "role": role,
            "epochs": EXPECTED["fixed_epochs"],
            "elapsed_seconds": time.time() - s0,
            "shared_shape": list(shared.shape),
            "adapter_audit": audit,
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
                "path": model_path.name,
                "sha256": sha256_file(model_path),
                "size_bytes": model_path.stat().st_size,
            },
        }
        seed_records.append(rec)
        print(f"V1 full-data seed={seed}: PASS")

    # Never select among seeds. Primary remains 1103 by contract.
    summary = {
        "status": "06C_TARGET_V1_FULL_DATA_REFIT_COMPLETE",
        "script_version": SCRIPT_VERSION,
        "runner_sha256": state["script_sha256"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds_total": time.time() - t0,
        "model_id": EXPECTED["primary_model_id"],
        "profile_id": EXPECTED["profile_id"],
        "n_cases": EXPECTED["n_cases"],
        "feature_budgets": EXPECTED["feature_budgets"],
        "pca_components": EXPECTED["pca_components"],
        "fixed_epochs": EXPECTED["fixed_epochs"],
        "primary_seed": EXPECTED["primary_seed"],
        "reproducibility_seeds": EXPECTED["reproducibility_seeds"],
        "post_fit_seed_selection_performed": False,
        "seed_ensemble_or_averaging_performed": False,
        "q10_060_applied_to_v1": False,
        "q10_040_applied_to_v1": False,
        "sable_mo_stability_gate_applied_to_v1": False,
        "outcomes_accessed": False,
        "clinical_values_accessed": False,
        "biological_annotations_accessed": False,
        "technical_covariates_used_for_fit": False,
        "preprocessing_function_from_authorized_06b_v4": prep_name,
        "input_provenance": input_provenance,
        "preprocessing_manifest": {
            "path": prep_manifest_path.name,
            "sha256": sha256_file(prep_manifest_path),
        },
        "preprocessing_numeric_snapshot": {
            "path": prep_npz.name,
            "sha256": sha256_file(prep_npz),
        },
        "seed_outputs": seed_records,
        "interpretation_boundary": (
            "The frozen TARGET primary representation is V1_SHARED_PRIVATE_AE. "
            "The q10=.60/.40 and SABLE-MO stability-retention rules are not transferred to V1."
        ),
    }
    summary_path = result_dir / "target_v1_full_data_refit_summary.json"
    write_json(summary_path, summary)

    result_files = {}
    for p in sorted(result_dir.iterdir()):
        if p.is_file():
            result_files[p.name] = {
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
            }

    lock = {
        "status": "06C3_TARGET_V1_FULL_DATA_REFIT_RESULTS_FROZEN",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_source": {
            "path": relative_to_paper5(Path(__file__).resolve(), paper5),
            "sha256": state["script_sha256"],
        },
        "source_authorization_lock": {
            "path": "method_contract/06c2_v2/target_v1_full_data_runner_v2_source_authorization_lock.json",
            "sha256": sha256_file(
                paper5 / "method_contract" / "06c2_v2" / "target_v1_full_data_runner_v2_source_authorization_lock.json"
            ),
        },
        "06c_contract_sha256": EXPECTED["06c_contract_sha256"],
        "06c_lock_sha256": EXPECTED["06c_lock_sha256"],
        "model_id": EXPECTED["primary_model_id"],
        "primary_seed": EXPECTED["primary_seed"],
        "fixed_epochs": EXPECTED["fixed_epochs"],
        "post_fit_seed_selection_performed": False,
        "q10_applied_to_v1": False,
        "outcomes_clinical_biology_accessed": False,
        "result_files": result_files,
    }
    result_lock_path = lock_dir / "target_v1_full_data_refit_results_lock.json"
    write_json(result_lock_path, lock)

    print("=" * 104)
    print("06c TARGET V1 all-78 full-data refit: COMPLETE")
    print("=" * 104)
    print("Primary representation seed: 1103")
    print("Reproducibility-only seeds: 2207, 3301")
    print("Post-fit seed selection: NO")
    print("q10=.60/.40 applied to V1: NO")
    print("TARGET outcomes/clinical/biology accessed: NO")
    print("Summary:", relative_to_paper5(summary_path, paper5))
    print("Results lock:", relative_to_paper5(result_lock_path, paper5))
    print("=" * 104)


def main() -> None:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    paper5 = project_root / "paper5"

    state = verify_frozen_chain(paper5, script_path)
    auth = load_authorization(paper5, state["script_sha256"])

    if auth is None:
        synthetic_selftest(state, paper5, script_path)
        return

    run_real(state, paper5, auth)


if __name__ == "__main__":
    main()
