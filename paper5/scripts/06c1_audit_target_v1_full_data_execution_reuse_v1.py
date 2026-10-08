#!/usr/bin/env python3
"""
Paper 5 - read-only audit for TARGET V1 all-78 execution reuse.

This stage DOES NOT:
- load TARGET molecular matrices;
- read TARGET outcomes/clinical/biology;
- fit any model;
- encode any TARGET sample;
- modify 06b or 06c.

It verifies the frozen 06c identity and statically inspects the already-authorized
06b v4 runner plus the frozen 05d4 neural implementation so the subsequent
full-data execution runner can reuse the exact numerical path instead of
inventing a second implementation.
"""

from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "06c1-audit-target-v1-full-data-execution-reuse-v1-no-cli"

EXPECTED = {
    "06c_contract_sha256": "8ab8516e30964f220d0f1204ca892e29ca337691e06cbcbf29315e1cb6aec2e5",
    "06c_lock_sha256": "1c3b4fb36706dc6afd0208fda23b1dc6d51c2953b4ccb9541099466aea13a374",
    "06c_script_sha256": "537b4856c8b24fd9aa22ea4f490423c4f418cf602f8f5cf6cb6e836cc0760e8d",
    "06b_v4_sha256": "4f69680a2d26d6c8e4bd5b337a5e94e6cdc36d149ea6a4b9af547b3855a93eab",
    "primary_model_id": "V1_SHARED_PRIVATE_AE",
    "primary_seed": 1103,
    "fixed_epochs": 400,
    "profile_id": "ARCH_COMPACT__REG_WEAK",
}


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
            f"{label} SHA256 mismatch.\n"
            f"Expected: {expected}\n"
            f"Observed: {actual}\n"
            f"Path: {path}"
        )
    return actual


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = []
    pos = list(node.args.posonlyargs) + list(node.args.args)
    defaults = [None] * (len(pos) - len(node.args.defaults)) + list(node.args.defaults)
    for a, d in zip(pos, defaults):
        s = a.arg
        if d is not None:
            try:
                s += "=" + ast.unparse(d)
            except Exception:
                s += "=..."
        args.append(s)
    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)
    if node.args.kwonlyargs:
        if not node.args.vararg:
            args.append("*")
        for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults):
            s = a.arg
            if d is not None:
                try:
                    s += "=" + ast.unparse(d)
                except Exception:
                    s += "=..."
            args.append(s)
    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)
    return f"{node.name}({', '.join(args)})"


def call_name(node: ast.Call) -> str | None:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        parts = [f.attr]
        cur = f.value
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def inspect_source(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text, filename=str(path))

    functions = []
    classes = []
    imports = []
    assignments = {}

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            calls = sorted(
                {
                    c
                    for n in ast.walk(node)
                    if isinstance(n, ast.Call)
                    for c in [call_name(n)]
                    if c
                }
            )
            literal_strings = sorted(
                {
                    n.value
                    for n in ast.walk(node)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                }
            )
            functions.append(
                {
                    "name": node.name,
                    "signature": signature(node),
                    "lineno": node.lineno,
                    "end_lineno": getattr(node, "end_lineno", None),
                    "direct_calls": calls,
                    "string_literals": literal_strings,
                }
            )
        elif isinstance(node, ast.ClassDef):
            methods = []
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(
                        {
                            "name": sub.name,
                            "signature": signature(sub),
                            "lineno": sub.lineno,
                            "end_lineno": getattr(sub, "end_lineno", None),
                        }
                    )
            classes.append(
                {
                    "name": node.name,
                    "lineno": node.lineno,
                    "end_lineno": getattr(node, "end_lineno", None),
                    "methods": methods,
                }
            )
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            try:
                imports.append(ast.unparse(node))
            except Exception:
                imports.append(type(node).__name__)
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if names:
                try:
                    val = ast.literal_eval(node.value)
                except Exception:
                    val = None
                for n in names:
                    assignments[n] = val
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            try:
                val = ast.literal_eval(node.value) if node.value is not None else None
            except Exception:
                val = None
            assignments[node.target.id] = val

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "imports": imports,
        "functions": functions,
        "classes": classes,
        "literal_top_level_assignments": assignments,
    }


def candidate_functions(source: dict[str, Any]) -> list[dict[str, Any]]:
    keywords = (
        "load",
        "target",
        "input",
        "transform",
        "preprocess",
        "feature",
        "rank",
        "pca",
        "neural",
        "fit",
        "train",
        "epoch",
        "encode",
        "representation",
        "model",
    )
    must_calls = {
        "train_fixed_epochs",
        "encode_numpy",
        "construct_model",
    }
    out = []
    for f in source["functions"]:
        lname = f["name"].lower()
        strings = " ".join(f["string_literals"])
        calls = set(f["direct_calls"])
        if (
            any(k in lname for k in keywords)
            or EXPECTED["primary_model_id"] in strings
            or any(c.split(".")[-1] in must_calls for c in calls)
        ):
            out.append(f)
    return out


def find_function(source: dict[str, Any], name: str) -> dict[str, Any] | None:
    for f in source["functions"]:
        if f["name"] == name:
            return f
    return None


def main() -> None:
    print("=" * 104)
    print("Paper 5 - read-only audit for TARGET V1 all-78 execution reuse")
    print("=" * 104)
    print("Script version:", SCRIPT_VERSION)
    print("TARGET molecular values read: NO")
    print("TARGET outcomes/clinical/biology read: NO")
    print("TARGET model fit: NO")
    print("Scientific representation generated: NO")
    print("06b/06c rewritten: NO")

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    paper5 = project_root / "paper5"

    contract_path = paper5 / "method_contract" / "06c" / "target_v1_full_data_refit_contract.json"
    lock_path = paper5 / "method_contract" / "06c" / "06c_full_data_refit_contract_lock.json"
    freeze_script_path = paper5 / "scripts" / "06c_freeze_target_v1_full_data_refit_contract_v1.py"
    v4_path = paper5 / "scripts" / "06b_run_target_outcome_blind_representation_benchmark_gpu_v4.py"
    source05d4_path = paper5 / "scripts" / "05d4_fit_neural_ablation_benchmark_gpu_v1_2_verified.py"

    for p in [contract_path, lock_path, freeze_script_path, v4_path, source05d4_path]:
        require_file(p)

    hashes = {
        "06c_contract": require_hash("06c contract", contract_path, EXPECTED["06c_contract_sha256"]),
        "06c_lock": require_hash("06c lock", lock_path, EXPECTED["06c_lock_sha256"]),
        "06c_freeze_script": require_hash(
            "06c freeze script", freeze_script_path, EXPECTED["06c_script_sha256"]
        ),
        "06b_v4_runner": require_hash("06b v4 runner", v4_path, EXPECTED["06b_v4_sha256"]),
        "05d4_verified_source": sha256_file(source05d4_path),
    }

    contract = load_json(contract_path)
    lock = load_json(lock_path)

    if contract.get("status") != "FROZEN_BEFORE_TARGET_FULL_DATA_REFIT":
        raise RuntimeError(f"Unexpected 06c contract status: {contract.get('status')!r}")
    if lock.get("status") != "06C_TARGET_V1_FULL_DATA_REFIT_CONTRACT_FROZEN":
        raise RuntimeError(f"Unexpected 06c lock status: {lock.get('status')!r}")
    if lock.get("full_data_fit_performed") is not False:
        raise RuntimeError("06c lock does not prove pre-full-data-fit state.")

    full = contract.get("full_data_refit", {})
    train = full.get("training", {})
    if full.get("model_id") != EXPECTED["primary_model_id"]:
        raise RuntimeError("06c primary model mismatch.")
    if full.get("profile_id") != EXPECTED["profile_id"]:
        raise RuntimeError("06c profile mismatch.")
    if int(train.get("primary_seed")) != EXPECTED["primary_seed"]:
        raise RuntimeError("06c primary seed mismatch.")
    if int(train.get("fixed_epochs")) != EXPECTED["fixed_epochs"]:
        raise RuntimeError("06c fixed epoch mismatch.")
    if train.get("early_stopping_on_full_78") is not False:
        raise RuntimeError("06c early-stopping firewall mismatch.")
    if train.get("post_fit_seed_selection") is not False:
        raise RuntimeError("06c seed-selection firewall mismatch.")
    if train.get("post_fit_epoch_selection") is not False:
        raise RuntimeError("06c epoch-selection firewall mismatch.")

    v4 = inspect_source(v4_path)
    d4 = inspect_source(source05d4_path)

    v4_candidates = candidate_functions(v4)
    d4_candidates = candidate_functions(d4)

    required_d4 = {}
    for name in ["construct_model", "train_fixed_epochs", "encode_numpy"]:
        f = find_function(d4, name)
        if f is None:
            raise RuntimeError(f"Required frozen 05d4 helper not found: {name}")
        required_d4[name] = f

    # Identify v4 functions that directly call the frozen 05d4 helpers.
    callers = {}
    for helper in required_d4:
        matches = []
        for f in v4["functions"]:
            if any(c.split(".")[-1] == helper for c in f["direct_calls"]):
                matches.append(
                    {
                        "name": f["name"],
                        "signature": f["signature"],
                        "lineno": f["lineno"],
                        "end_lineno": f["end_lineno"],
                        "direct_calls": f["direct_calls"],
                    }
                )
        callers[helper] = matches

    # Also locate exact V1 references in v4.
    v1_references = []
    for f in v4["functions"]:
        if EXPECTED["primary_model_id"] in f["string_literals"]:
            v1_references.append(
                {
                    "name": f["name"],
                    "signature": f["signature"],
                    "lineno": f["lineno"],
                    "end_lineno": f["end_lineno"],
                }
            )

    out_dir = paper5 / "method_contract" / "06c1"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(
            f"06c1 output directory already exists and is non-empty: {out_dir}\n"
            "Do not overwrite a prior audit."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "06C1_TARGET_V1_EXECUTION_REUSE_AUDIT_COMPLETE",
        "scientific_state": {
            "target_molecular_values_read": False,
            "target_outcomes_clinical_biology_read": False,
            "target_model_fit": False,
            "scientific_representation_generated": False,
            "06b_rewritten": False,
            "06c_rewritten": False,
        },
        "06c_identity": {
            "contract_sha256": hashes["06c_contract"],
            "lock_sha256": hashes["06c_lock"],
            "freeze_script_sha256": hashes["06c_freeze_script"],
            "model_id": full.get("model_id"),
            "profile_id": full.get("profile_id"),
            "primary_seed": train.get("primary_seed"),
            "reproducibility_seeds": train.get("reproducibility_seeds"),
            "fixed_epochs": train.get("fixed_epochs"),
            "feature_budget": full.get("feature_budget"),
            "n_cases": full.get("n_cases"),
        },
        "source_identity": {
            "06b_v4_runner": {
                "path": "scripts/06b_run_target_outcome_blind_representation_benchmark_gpu_v4.py",
                "sha256": hashes["06b_v4_runner"],
            },
            "05d4_verified_neural_source": {
                "path": "scripts/05d4_fit_neural_ablation_benchmark_gpu_v1_2_verified.py",
                "sha256": hashes["05d4_verified_source"],
            },
        },
        "required_frozen_05d4_helpers": required_d4,
        "06b_v4_callers_of_required_05d4_helpers": callers,
        "06b_v4_v1_literal_references": v1_references,
        "06b_v4_candidate_execution_functions": v4_candidates,
        "05d4_candidate_neural_functions": d4_candidates,
        "06b_v4_top_level_literal_assignments": v4["literal_top_level_assignments"],
        "05d4_top_level_literal_assignments": d4["literal_top_level_assignments"],
        "next_required_stage": (
            "Build a new versioned all-78 V1 execution runner that consumes the exact "
            "06c lock and reuses the already-authorized v4/05d4 numerical path. "
            "Do not fit TARGET in 06c1."
        ),
    }

    report_path = out_dir / "target_v1_full_data_execution_reuse_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("06c exact contract SHA: PASS")
    print("06c exact lock SHA: PASS")
    print("06c freeze script SHA: PASS")
    print("06b v4 runner SHA: PASS")
    print("Primary model:", full.get("model_id"))
    print("Primary seed:", train.get("primary_seed"))
    print("Fixed epochs:", train.get("fixed_epochs"))
    print("Frozen 05d4 helpers:")
    for name, f in required_d4.items():
        print(f"  {f['signature']} @ lines {f['lineno']}-{f['end_lineno']}")
    print("06b v4 callers of frozen helpers:")
    for helper, matches in callers.items():
        print(f"  {helper}:")
        if not matches:
            print("    NONE")
        else:
            for m in matches:
                print(f"    {m['signature']} @ lines {m['lineno']}-{m['end_lineno']}")
    print("06b v4 explicit V1 references:")
    if not v1_references:
        print("  NONE")
    else:
        for m in v1_references:
            print(f"  {m['signature']} @ lines {m['lineno']}-{m['end_lineno']}")
    print("TARGET molecular values read: NO")
    print("TARGET model fit: NO")
    print("=" * 104)
    print("06c1 TARGET V1 execution reuse audit: COMPLETE")
    print("=" * 104)
    print("Report: method_contract/06c1/target_v1_full_data_execution_reuse_audit.json")
    print("=" * 104)


if __name__ == "__main__":
    main()
