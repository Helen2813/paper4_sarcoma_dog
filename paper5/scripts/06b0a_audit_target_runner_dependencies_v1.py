#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_VERSION = "06b0a-audit-target-runner-dependencies-v1-no-cli"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

MC05A = PAPER5_ROOT / "method_contract" / "05a"
MC05D3 = PAPER5_ROOT / "method_contract" / "05d3"
MC06B0 = PAPER5_ROOT / "method_contract" / "06b0"

OUT = PAPER5_ROOT / "method_contract" / "06b0a"
REPORT = OUT / "target_runner_dependency_audit.json"
README = OUT / "06b0a_README.txt"

REQUIRED_06B0_STATUS = "06B0_TARGET_PREFIT_INPUT_POLICY_LOCK_FROZEN"

REGISTRY_FILES = [
    MC05A / "model_family_registry.csv",
    MC05A / "baseline_hyperparameter_grid.csv",
    MC05A / "sable_mo_ablation_registry.csv",
    MC05A / "sable_mo_architecture_profiles.csv",
    MC05A / "sable_mo_regularization_profiles.csv",
    MC05A / "target_cv_split_summary.csv",
]

ASSIGNMENT_FILES = [
    MC05A / "target_outer_cv_assignments.csv",
    MC05A / "target_inner_cv_assignments.csv",
    MC05A / "target_neural_training_assignments.csv",
]

CANDIDATE_SCRIPT_GLOBS = [
    "05c1*.py",
    "05c2*.py",
    "05d2*.py",
    "05d3*.py",
    "05d4*.py",
    "05d5*.py",
    "05f1*.py",
]

KEYWORDS = (
    "fit", "train", "model", "score", "metric", "preprocess", "standard",
    "feature", "pca", "gcca", "mofa", "neural", "sable", "encode",
    "decode", "project", "align", "stability", "reconstruct", "missing",
    "utility", "fold", "epoch", "profile", "build",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def summarize_csv(path: Path, include_rows: bool = True) -> dict[str, Any]:
    fields, rows = read_csv(path)
    result: dict[str, Any] = {
        "path": rel(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "columns": fields,
        "n_rows": len(rows),
    }
    if include_rows:
        result["rows"] = rows
    return result


def unique_counts(rows: list[dict[str, str]], columns: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in columns:
        if not rows or col not in rows[0]:
            continue
        values = [str(r.get(col, "")).strip() for r in rows]
        uniq = sorted(set(values))
        out[col] = {
            "n_unique": len(uniq),
            "values": uniq[:100],
            "truncated": len(uniq) > 100,
        }
    return out


def function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = []
    pos = list(node.args.posonlyargs) + list(node.args.args)
    defaults = [None] * (len(pos) - len(node.args.defaults)) + list(node.args.defaults)
    for arg, default in zip(pos, defaults):
        text = arg.arg
        if default is not None:
            try:
                text += "=" + ast.unparse(default)
            except Exception:
                text += "=..."
        args.append(text)
    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)
    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        text = arg.arg
        if default is not None:
            try:
                text += "=" + ast.unparse(default)
            except Exception:
                text += "=..."
        args.append(text)
    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)
    return f"{node.name}({', '.join(args)})"


def inspect_python(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))

    defs = []
    classes = []
    constants: dict[str, Any] = {}
    imports = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = function_signature(node)
            if any(k in node.name.lower() for k in KEYWORDS):
                defs.append({
                    "name": node.name,
                    "signature": sig,
                    "lineno": node.lineno,
                })
        elif isinstance(node, ast.ClassDef):
            if any(k in node.name.lower() for k in KEYWORDS):
                methods = []
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if child.name == "__init__" or any(k in child.name.lower() for k in KEYWORDS):
                            methods.append({
                                "name": child.name,
                                "signature": function_signature(child),
                                "lineno": child.lineno,
                            })
                classes.append({
                    "name": node.name,
                    "lineno": node.lineno,
                    "methods": methods,
                })
        elif isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if any(x in name.upper() for x in (
                    "VARIANT", "PROFILE", "MODEL", "SEED", "EPOCH", "LATENT",
                    "SHARED", "PRIVATE", "DEVICE", "STABILITY", "SUBSAMPLE",
                )):
                    try:
                        value = ast.literal_eval(node.value)
                        if isinstance(value, (str, int, float, bool, list, tuple, dict, type(None))):
                            constants[name] = value
                    except Exception:
                        pass
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.append((node.module or "") + ":" + ",".join(a.name for a in node.names))

    return {
        "path": rel(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "relevant_top_level_functions": defs,
        "relevant_classes": classes,
        "selected_literal_constants": constants,
        "imports": imports,
    }


def main() -> None:
    print("=" * 104)
    print("Paper 5 - read-only TARGET runner dependency audit")
    print("=" * 104)
    print(f"Script version: {SCRIPT_VERSION}")
    print("TARGET molecular values read: NO")
    print("TARGET outcomes/clinical read: NO")
    print("TARGET model fit: NO")
    print("Scientific result generation: NO")
    print()

    lock_path = MC06B0 / "06b0_prefit_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"Missing 06b0 lock: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != REQUIRED_06B0_STATUS:
        raise RuntimeError(f"Unexpected 06b0 status: {lock.get('status')}")
    if lock.get("target_fit_performed") is not False:
        raise RuntimeError("06b0 says a TARGET fit has already been performed.")
    if lock.get("runner_source_authorized") is not False:
        raise RuntimeError("06b0 unexpectedly authorizes a runner already.")

    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"06b0a output directory already exists and is non-empty: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    registries = {}
    for path in REGISTRY_FILES:
        registries[path.name] = summarize_csv(path, include_rows=True)

    assignments = {}
    for path in ASSIGNMENT_FILES:
        fields, rows = read_csv(path)
        assignments[path.name] = {
            "path": rel(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "columns": fields,
            "n_rows": len(rows),
            "unique_counts": unique_counts(
                rows,
                [
                    "case_id", "sample_id", "outer_repeat", "outer_fold", "inner_fold",
                    "role", "split_role", "training_role", "model_seed", "seed",
                    "neural_seed", "early_stop_seed",
                ],
            ),
            "first_3_rows_for_schema_only": rows[:3],
        }

    scripts = []
    seen = set()
    script_dir = PAPER5_ROOT / "scripts"
    for pattern in CANDIDATE_SCRIPT_GLOBS:
        for path in sorted(script_dir.glob(pattern)):
            if path.name in seen:
                continue
            seen.add(path.name)
            try:
                scripts.append(inspect_python(path))
            except SyntaxError as exc:
                scripts.append({
                    "path": rel(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "parse_error": str(exc),
                })

    selected_profile = {}
    selected_profile_path = MC05D3 / "selected_neural_profile.json"
    if selected_profile_path.is_file():
        selected_profile = json.loads(selected_profile_path.read_text(encoding="utf-8"))

    model_rows = registries["model_family_registry.csv"]["rows"]
    eligible = [
        r for r in model_rows
        if str(r.get("eligible_for_nested_selection", "")).strip().lower() in {"true", "1", "yes"}
    ]

    report = {
        "status": "READ_ONLY_TARGET_RUNNER_DEPENDENCY_AUDIT_COMPLETE",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "06b0_lock": {
            "path": rel(lock_path),
            "sha256": sha256_file(lock_path),
            "status": lock["status"],
        },
        "registries": registries,
        "nested_selection_eligible_model_rows": eligible,
        "assignments": assignments,
        "selected_neural_profile": selected_profile,
        "candidate_existing_implementation_sources": scripts,
        "guardrails": {
            "target_molecular_values_read": False,
            "target_outcomes_read": False,
            "target_clinical_values_read": False,
            "target_model_fit_performed": False,
            "scientific_results_generated": False,
            "purpose": (
                "Resolve exact frozen registry rows, CV assignment schemas, and reusable implementation "
                "entry points before writing the real TARGET runner. No numerical method is selected here."
            ),
        },
        "next_required_stage": (
            "Write one new versioned TARGET runner using only frozen registry/policy choices and existing "
            "implementation paths where possible; test it only on 05dev0 synthetic fixtures; then source-hash "
            "lock the exact runner in 06b1 before first real TARGET fit."
        ),
    }

    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    README.write_text(
        "Paper 5 06b0a - read-only TARGET runner dependency audit\n"
        "========================================================\n\n"
        "No TARGET molecular values, outcomes, clinical values, or biological annotations are read.\n"
        "No model is fit. This stage only records frozen registry rows, CV-assignment schemas, and\n"
        "top-level reusable implementation entry points from existing Paper 5 source files.\n\n"
        "The audit exists to avoid inventing a new numerical implementation when the frozen simulation\n"
        "pipeline already contains reusable preprocessing/model/scoring code.\n",
        encoding="utf-8",
    )

    print("06b0a dependency audit: COMPLETE")
    print(f"Nested-selection eligible model rows: {len(eligible)}")
    for row in eligible:
        model_id = row.get("model_id", "<no model_id>")
        family = row.get("family", row.get("model_family", ""))
        rank = row.get("complexity_rank", "")
        print(f"  - {model_id} | family={family} | complexity_rank={rank}")

    print()
    for name, item in assignments.items():
        print(f"{name}: rows={item['n_rows']} columns={len(item['columns'])}")
        print("  columns:", ", ".join(item["columns"]))

    print()
    print("Candidate implementation sources:")
    for item in scripts:
        print(f"  {item['path']} | functions={len(item.get('relevant_top_level_functions', []))} "
              f"| classes={len(item.get('relevant_classes', []))}")

    print()
    print(f"Report: {rel(REPORT)}")
    print("TARGET fit performed: NO")
    print("Next: use this audit to build the exact synthetic-tested runner, then freeze its source in 06b1.")


if __name__ == "__main__":
    main()
