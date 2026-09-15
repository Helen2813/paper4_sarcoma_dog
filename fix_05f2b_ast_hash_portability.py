from __future__ import annotations

import ast
import codecs
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


PATCH_VERSION = "paper5-fix-05f2b-ast-hash-portability-v1-no-cli"

TARGET_NAME = "05f2b_freeze_final_gate_aggregation_poststart_preunblinding.py"
PREEXISTING_05F3_NAME = "05f3_aggregate_final_gate_results_cpu.py"
AGGREGATION_05F3B_NAME = (
    "05f3b_aggregate_final_gate_results_cpu_preunblinding_locked.py"
)

OLD_SCRIPT_VERSION = (
    "05f2b-freeze-final-gate-aggregation-poststart-preunblinding-v1-no-cli"
)
NEW_SCRIPT_VERSION = (
    "05f2b-freeze-final-gate-aggregation-poststart-preunblinding-"
    "v2-runtime-portable-ast-equivalence-no-cli"
)

SCIENTIFIC_CORE_FUNCTIONS = (
    "wilson_interval",
    "percentile_bootstrap_mean",
    "fit_logistic_sample_size_trend",
    "extract_sealed_results",
    "with_numeric_scenario_fields",
    "add_dataset_false_acceptance_event",
    "pooled_wilson_record",
    "biological_recovery_count",
    "technical_opportunity_counts",
    "compute_formal_results",
    "build_scenario_summary",
    "build_operating_metric_summary",
    "select_failure_policy",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_project_root() -> Path:
    starts = [Path.cwd(), Path(__file__).resolve().parent]
    checked: set[Path] = set()

    for start in starts:
        for candidate in (start, *start.parents):
            resolved = candidate.resolve()
            if resolved in checked:
                continue
            checked.add(resolved)
            target = resolved / "paper5" / "scripts" / TARGET_NAME
            if target.is_file():
                return resolved

    raise FileNotFoundError(
        "Could not locate the project root containing "
        f"paper5/scripts/{TARGET_NAME}. Run this patcher from the "
        "paper4_sarcoma_dog project root."
    )


def literal_constants(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    output: dict[str, Any] = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            output[target.id] = ast.literal_eval(node.value)
        except Exception:
            continue

    return output


def scientific_core_ast_hash(path: Path) -> str:
    """Reproduce the current 05f2b runtime AST fingerprint exactly."""
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    hashes: dict[str, str] = {}

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in SCIENTIFIC_CORE_FUNCTIONS:
            hashes[node.name] = hashlib.sha256(
                ast.dump(
                    node,
                    annotate_fields=True,
                    include_attributes=False,
                ).encode("utf-8")
            ).hexdigest()

    missing = set(SCIENTIFIC_CORE_FUNCTIONS) - set(hashes)
    if missing:
        raise RuntimeError(
            f"{path.name} lacks scientific-core functions: "
            + ", ".join(sorted(missing))
        )

    return hashlib.sha256(
        json.dumps(
            hashes,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def encode_preserving_format(
    normalized_source: str,
    *,
    newline: str,
    had_utf8_bom: bool,
) -> bytes:
    rendered = normalized_source.replace("\n", newline)
    encoded = rendered.encode("utf-8")
    return codecs.BOM_UTF8 + encoded if had_utf8_bom else encoded


def main() -> None:
    print("=" * 88)
    print("Paper 5 - repair 05f2b cross-Python AST-hash portability")
    print("=" * 88)
    print(f"Patch version: {PATCH_VERSION}")
    print(f"Python: {sys.version.splitlines()[0]}")
    print("Scientific shard access: NO")
    print("Scientific metric calculation: NO")
    print("PASS/FAIL evaluation: NO")
    print("05f3/05f3b modification: NO")
    print()

    project_root = locate_project_root()
    scripts_dir = project_root / "paper5" / "scripts"
    target = scripts_dir / TARGET_NAME
    old_05f3 = scripts_dir / PREEXISTING_05F3_NAME
    new_05f3b = scripts_dir / AGGREGATION_05F3B_NAME

    for path in (target, old_05f3, new_05f3b):
        if not path.is_file():
            raise FileNotFoundError(path)

    constants = literal_constants(target)
    required_constant_names = (
        "EXPECTED_PREEXISTING_05F3_SHA256",
        "EXPECTED_05F3_SHA256",
        "EXPECTED_SCIENTIFIC_CORE_AST_SHA256",
    )
    missing_constants = [
        name for name in required_constant_names if name not in constants
    ]
    if missing_constants:
        raise RuntimeError(
            "05f2b lacks expected audit constants: "
            + ", ".join(missing_constants)
        )

    expected_old_sha = str(constants["EXPECTED_PREEXISTING_05F3_SHA256"])
    expected_new_sha = str(constants["EXPECTED_05F3_SHA256"])
    legacy_expected_ast = str(constants["EXPECTED_SCIENTIFIC_CORE_AST_SHA256"])

    observed_old_sha = sha256_file(old_05f3)
    observed_new_sha = sha256_file(new_05f3b)

    print(f"Project root: {project_root}")
    print(f"05f2b target: {target}")
    print()
    print("Exact source locks")
    print("-" * 88)
    print(f"Pre-existing 05f3 expected SHA256: {expected_old_sha}")
    print(f"Pre-existing 05f3 observed SHA256: {observed_old_sha}")
    print(f"05f3b expected SHA256:             {expected_new_sha}")
    print(f"05f3b observed SHA256:             {observed_new_sha}")

    if observed_old_sha != expected_old_sha:
        raise RuntimeError(
            "Pre-existing 05f3 exact source SHA256 does not match. "
            "Refusing to patch anything."
        )
    if observed_new_sha != expected_new_sha:
        raise RuntimeError(
            "05f3b exact source SHA256 does not match. "
            "Refusing to patch anything."
        )

    old_core_hash = scientific_core_ast_hash(old_05f3)
    new_core_hash = scientific_core_ast_hash(new_05f3b)

    print()
    print("Runtime scientific-core AST audit")
    print("-" * 88)
    print(f"Legacy parser-specific expected hash: {legacy_expected_ast}")
    print(f"Pre-existing 05f3 runtime AST hash:   {old_core_hash}")
    print(f"05f3b runtime AST hash:                {new_core_hash}")
    print(
        "Runtime 05f3/05f3b scientific-core equality: "
        f"{'PASS' if old_core_hash == new_core_hash else 'FAIL'}"
    )

    if old_core_hash != new_core_hash:
        raise RuntimeError(
            "The two scientific cores are not AST-identical in the current "
            "interpreter. This is not the portability-only failure; refusing "
            "to patch."
        )

    raw = target.read_bytes()
    had_utf8_bom = raw.startswith(codecs.BOM_UTF8)
    newline = "\r\n" if b"\r\n" in raw else "\n"
    source = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")

    old_gate = """    old_core_hash = scientific_core_ast_hash(PREEXISTING_05F3_SCRIPT)
    new_core_hash = scientific_core_ast_hash(AGGREGATION_SCRIPT)
    if old_core_hash != EXPECTED_SCIENTIFIC_CORE_AST_SHA256:
        raise RuntimeError("Pre-existing 05f3 scientific-core AST hash changed.")
    if new_core_hash != EXPECTED_SCIENTIFIC_CORE_AST_SHA256:
        raise RuntimeError(
            "05f3b scientific aggregation core is not AST-identical to pre-existing 05f3."
        )
"""

    new_gate = """    # ast.dump output is tied to the running Python AST schema, so a digest
    # generated under another Python minor version is not a portable source
    # identity check. The exact SHA256 locks above pin both source files.
    # Compare their scientific cores directly after parsing both in this same
    # interpreter; this preserves the intended equivalence guard without a
    # parser-version-dependent false failure.
    old_core_hash = scientific_core_ast_hash(PREEXISTING_05F3_SCRIPT)
    new_core_hash = scientific_core_ast_hash(AGGREGATION_SCRIPT)
    if old_core_hash != new_core_hash:
        raise RuntimeError(
            "05f3b scientific aggregation core is not AST-identical to pre-existing 05f3."
        )
"""

    already_patched = (
        "if old_core_hash != new_core_hash:" in source
        and "if old_core_hash != EXPECTED_SCIENTIFIC_CORE_AST_SHA256:" not in source
    )
    if already_patched:
        print()
        print("05f2b is already patched. No file was rewritten.")
        print("=" * 88)
        return

    gate_count = source.count(old_gate)
    if gate_count != 1:
        raise RuntimeError(
            "Could not identify exactly one expected legacy AST-gate block "
            f"(found {gate_count}). Refusing to edit the source."
        )

    old_version_literal = f'SCRIPT_VERSION = "{OLD_SCRIPT_VERSION}"'
    new_version_literal = f'SCRIPT_VERSION = "{NEW_SCRIPT_VERSION}"'
    version_count = source.count(old_version_literal)
    if version_count != 1:
        raise RuntimeError(
            "Could not identify exactly one expected 05f2b SCRIPT_VERSION "
            f"literal (found {version_count}). Refusing to edit the source."
        )

    patched = source.replace(old_gate, new_gate, 1)
    patched = patched.replace(old_version_literal, new_version_literal, 1)

    # Syntax-check the exact text before touching the original.
    compile(patched, str(target), "exec")

    backup = target.with_name(
        target.name + ".before_runtime_portable_ast_fix.bak"
    )
    if backup.exists():
        raise FileExistsError(
            f"Backup already exists: {backup}\n"
            "No source file was changed. Inspect or rename that backup first."
        )

    old_target_sha = sha256_bytes(raw)
    shutil.copy2(target, backup)

    encoded = encode_preserving_format(
        patched,
        newline=newline,
        had_utf8_bom=had_utf8_bom,
    )
    temporary = target.with_name(target.name + ".tmp_ast_portability_fix")
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()

    # Re-open and compile the installed file.
    installed = target.read_text(encoding="utf-8-sig")
    compile(installed, str(target), "exec")
    new_target_sha = sha256_file(target)

    print()
    print("Patch result")
    print("-" * 88)
    print(f"Backup: {backup}")
    print(f"05f2b old SHA256: {old_target_sha}")
    print(f"05f2b new SHA256: {new_target_sha}")
    print(f"05f2b new script version: {NEW_SCRIPT_VERSION}")
    print("Modified scientific aggregation code: NO")
    print("Modified thresholds/selectors/denominators: NO")
    print("Modified 05f3 or 05f3b: NO")
    print()
    print("Next command:")
    print(
        "python paper5\\scripts\\"
        "05f2b_freeze_final_gate_aggregation_poststart_preunblinding.py"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
