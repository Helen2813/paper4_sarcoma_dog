#!/usr/bin/env python
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "05y0d-extract-private-score-generator-contract-v1-no-cli"
STATUS = "READ_ONLY_GENERATOR_SOURCE_EXTRACTION_NO_DIAGNOSTIC_RESULT_COMPUTED"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

SCENARIO_REGISTRY = (
    PAPER5_ROOT / "method_contract" / "05a1" / "final_gate_calibration_scenarios.csv"
)
LOCK_05Y0C = (
    PAPER5_ROOT / "method_contract" / "05y0c" / "05y0c_preflight_lock.json"
)

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05y0d"
OUTPUT_REPORT = OUTPUT_DIR / "private_score_generator_contract.json"
OUTPUT_SOURCE = OUTPUT_DIR / "private_score_generator_source.txt"
OUTPUT_README = OUTPUT_DIR / "05y0d_README.txt"
OUTPUT_LOCK = OUTPUT_DIR / "05y0d_preflight_lock.json"

G19 = "G19_CORRELATED_PRIVATE_NULL_N62"

SEARCH_PATTERNS = (
    "05b*.py",
    "05f0*.py",
    "05f1*.py",
)

TARGET_STRINGS = (
    "correlated_private_scores",
    "private_cross_view_correlation",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".txt", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def g19_row() -> dict[str, str]:
    require_file(SCENARIO_REGISTRY)
    with SCENARIO_REGISTRY.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = [
            row for row in csv.DictReader(fh)
            if str(row.get("scenario_id", "")).strip() == G19
        ]
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one {G19} row; found {len(rows)}.")
    return rows[0]


def source_files() -> list[Path]:
    found: dict[str, Path] = {}
    scripts_dir = PAPER5_ROOT / "scripts"
    for pattern in SEARCH_PATTERNS:
        for path in scripts_dir.glob(pattern):
            if path.is_file():
                found[str(path.resolve()).lower()] = path
    return sorted(found.values(), key=lambda p: p.name.lower())


def source_segment(lines: list[str], start: int, end: int) -> str:
    return "\n".join(
        f"{i+1:05d}: {lines[i]}"
        for i in range(start, min(end, len(lines)))
    )


def function_contains_targets(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    source_text: str,
) -> bool:
    segment = ast.get_source_segment(source_text, node) or ""
    low = segment.lower()
    return all(target.lower() in low for target in TARGET_STRINGS)


def find_generator_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for path in source_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if function_contains_targets(node, text):
                    candidates.append(
                        {
                            "path": path,
                            "function_name": node.name,
                            "lineno": int(node.lineno),
                            "end_lineno": int(getattr(node, "end_lineno", node.lineno)),
                            "source_text": text,
                        }
                    )
    return candidates


def call_sites(
    path: Path,
    function_name: str,
    text: str,
) -> list[dict[str, Any]]:
    tree = ast.parse(text)
    lines = text.splitlines()
    rows = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        called = None
        if isinstance(node.func, ast.Name):
            called = node.func.id
        elif isinstance(node.func, ast.Attribute):
            called = node.func.attr

        if called != function_name:
            continue

        line = int(getattr(node, "lineno", 1))
        start = max(0, line - 12)
        end = min(len(lines), line + 18)
        rows.append(
            {
                "line": line,
                "context": source_segment(lines, start, end),
            }
        )
    return rows


def return_expressions(
    text: str,
    function_name: str,
) -> list[str]:
    tree = ast.parse(text)
    target = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            target = node
            break
    if target is None:
        return []

    out = []
    for node in ast.walk(target):
        if isinstance(node, ast.Return):
            segment = ast.get_source_segment(text, node.value) if node.value is not None else None
            out.append(segment or "<bare return>")
    return out


def main() -> None:
    print("=" * 96)
    print("Paper 5 - blind extraction of exact correlated-private score generator")
    print("=" * 96)
    print(f"Script version: {SCRIPT_VERSION}")
    print("Failure labels inspected: NO")
    print("70-vs-130 comparison computed: NO")
    print("Threshold frontier computed: NO")
    print("Model fitting: NO")
    print("GPU computation: NO")
    print()

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(
            f"05y0d output directory already exists and is non-empty: {OUTPUT_DIR}. "
            "Do not overwrite a versioned preflight."
        )

    require_file(LOCK_05Y0C)
    lock_05y0c = load_json(LOCK_05Y0C)
    if str(lock_05y0c.get("status", "")).strip() != (
        "READ_ONLY_SEMANTIC_SOURCE_PREFLIGHT_NO_DIAGNOSTIC_RESULT_COMPUTED"
    ):
        raise RuntimeError("Unexpected 05y0c lock status.")

    scenario = g19_row()
    nominal_metadata = {
        key: value
        for key, value in scenario.items()
        if any(
            token in str(key).lower()
            for token in (
                "scenario_id",
                "scenario_family",
                "n_samples",
                "private",
                "rho",
                "corr",
                "seed",
            )
        )
    }

    candidates = find_generator_candidates()
    if not candidates:
        raise RuntimeError(
            "No function containing both correlated_private_scores and "
            "private_cross_view_correlation was found."
        )

    # Prefer an 05b generator over wrappers/callers, then shortest candidate function.
    candidates.sort(
        key=lambda row: (
            0 if row["path"].name.lower().startswith("05b") else 1,
            row["end_lineno"] - row["lineno"],
            row["path"].name.lower(),
        )
    )
    selected = candidates[0]
    selected_path: Path = selected["path"]
    function_name = str(selected["function_name"])
    text = str(selected["source_text"])
    lines = text.splitlines()

    start = max(0, int(selected["lineno"]) - 8)
    end = min(len(lines), int(selected["end_lineno"]) + 8)
    function_source = source_segment(lines, start, end)

    calls = call_sites(selected_path, function_name, text)
    returns = return_expressions(text, function_name)

    # Also capture RNG initialization and simulator call order from the same source file.
    rng_contexts: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        low = line.lower()
        if (
            "default_rng" in low
            or "simulation_seed" in low
            or "generate_" + "private" in low
            or function_name.lower() in low
        ):
            if function_name.lower() in low or "default_rng" in low:
                s = max(0, i - 12)
                e = min(len(lines), i + 24)
                rng_contexts.append(
                    {
                        "line": i + 1,
                        "context": source_segment(lines, s, e),
                    }
                )

    # Deduplicate overlapping textual contexts.
    unique_contexts = []
    seen = set()
    for row in rng_contexts:
        key = row["context"]
        if key not in seen:
            seen.add(key)
            unique_contexts.append(row)

    source_blocks = [
        f"SOURCE: {relative_path(selected_path)}",
        f"SHA256: {sha256_file(selected_path)}",
        f"FUNCTION: {function_name}",
        f"LINES: {selected['lineno']}-{selected['end_lineno']}",
        "",
        "=== EXACT FUNCTION SOURCE ===",
        function_source,
        "",
        "=== RETURN EXPRESSIONS ===",
        *[f"- {expr}" for expr in returns],
        "",
        "=== CALL SITES ===",
    ]
    for row in calls:
        source_blocks.extend(
            [
                f"\n--- call near line {row['line']} ---",
                row["context"],
            ]
        )

    source_blocks.append("\n=== RNG / CALL-ORDER CONTEXTS ===")
    for row in unique_contexts:
        source_blocks.extend(
            [
                f"\n--- context near line {row['line']} ---",
                row["context"],
            ]
        )

    source_text = "\n".join(source_blocks) + "\n"

    report = {
        "script_version": SCRIPT_VERSION,
        "status": STATUS,
        "scientific_result_computed": False,
        "failure_labels_inspected": False,
        "threshold_frontier_computed": False,
        "g19_nominal_private_correlation_metadata": nominal_metadata,
        "generator_candidate_count": len(candidates),
        "selected_generator": {
            "path": relative_path(selected_path),
            "sha256": sha256_file(selected_path),
            "function_name": function_name,
            "start_line": int(selected["lineno"]),
            "end_line": int(selected["end_lineno"]),
            "return_expressions": returns,
            "call_site_count_in_same_file": len(calls),
        },
        "all_generator_candidates": [
            {
                "path": relative_path(row["path"]),
                "function_name": row["function_name"],
                "start_line": row["lineno"],
                "end_line": row["end_lineno"],
                "sha256": sha256_file(row["path"]),
            }
            for row in candidates
        ],
        "next_step_rule": (
            "Use the exact extracted generator function and its frozen simulation seed "
            "path to define a dataset-level realized private cross-view correlation. "
            "Freeze the scalar formula before inspecting any 70-vs-130 association. "
            "If the generator returns per-view private score matrices, the preferred "
            "candidate is the mean same-factor Pearson correlation across all ordered "
            "or unordered view pairs, with the exact choice fixed before computation."
        ),
        "sources": {
            "scenario_registry": file_record(SCENARIO_REGISTRY),
            "05y0c_lock": file_record(LOCK_05Y0C),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_text_atomic(source_text, OUTPUT_SOURCE)
    write_json_atomic(report, OUTPUT_REPORT)

    readme = f"""Paper 5 05y0d — private-score generator source extraction
============================================================

No scientific diagnostic result was computed.

Selected frozen generator source:
  {relative_path(selected_path)}

Function:
  {function_name}

The exact function body, return expressions, call sites, and RNG/call-order
contexts are exported without reading failure labels.

Next:
  freeze the realized-rho scalar definition and deterministic replay rule before
  any 70-vs-130 comparison.
"""
    write_text_atomic(readme, OUTPUT_README)

    lock = {
        "script_version": SCRIPT_VERSION,
        "script_path": relative_path(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "status": STATUS,
        "scientific_result_computed": False,
        "generated_files": {
            relative_path(path): file_record(path)
            for path in [OUTPUT_REPORT, OUTPUT_SOURCE, OUTPUT_README]
        },
        "rewrite_policy": (
            "05y0d is source extraction only. Do not add diagnostic association "
            "calculations after viewing labels."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    print("05y0d generator-source extraction: COMPLETE")
    print(f"G19 nominal metadata: {nominal_metadata}")
    print(f"Selected source: {relative_path(selected_path)}")
    print(f"Generator function: {function_name}")
    print(f"Function lines: {selected['lineno']}-{selected['end_lineno']}")
    print(f"Return expressions: {len(returns)}")
    print(f"Call sites in same file: {len(calls)}")
    print(f"Source export: {relative_path(OUTPUT_SOURCE)}")
    print(f"Report:        {relative_path(OUTPUT_REPORT)}")
    print(f"Lock:          {relative_path(OUTPUT_LOCK)}")
    print("Scientific diagnostic result computed: NO")


if __name__ == "__main__":
    main()
