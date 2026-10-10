#!/usr/bin/env python3
"""Read-only Paper 5 iClusterPlus tree-hash diagnostic, Python 3.10+.

Run from the project root. Prints to stdout only. Reads exactly the two
specified Python source files, two frozen 05cmp0c JSON metadata files, and
files inside the specified isolated iClusterPlus package directory.
Does not import or execute project code, invoke R, fit models, alter files,
create authorization, or read simulation/result directories.

No replacement tree-hash algorithm is introduced: the diagnostic prints
existing implementations for review and inventories individual files.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tokenize

EXPECTED_RUNNER_SHA256 = (
    "371f4475ab639607496f198a3420fc47f1c7fa6f063d1e33574260c3987e41f5"
)
RUNNER_REL = Path(
    "paper5/scripts/05cmp3_run_secondary_comparator_confirmatory_benchmark_v2.py"
)
SMOKE_REL = Path("paper5/scripts/05cmp0c_smoke_test_icluster2b_cpu_v4.py")
METADATA_RELS = (
    Path("paper5/method_contract/05cmp0c/icluster2b_cpu_smoke_environment.json"),
    Path("paper5/method_contract/05cmp0c/icluster2b_cpu_smoke_lock.json"),
)
PACKAGE_REL = Path("paper5/comparator_envs/r_4_6_bioc_3_23/iClusterPlus")
MAX_TEXT_BYTES = 4 * 1024 * 1024


def heading(label: str) -> None:
    print("\n" + "=" * 100)
    print(label)
    print("=" * 100)


def reject_link(path: Path) -> None:
    info = path.lstat()  # Missing or unreadable paths are errors, never skipped.
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & reparse:
        raise RuntimeError(f"Refusing symlink/reparse-point path: {path}")


def checked_path(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unexpected relative path: {relative}")
    path = root
    for part in relative.parts:
        path = path / part
        reject_link(path)
    return path


def stable_file(path: Path, collect: bool = False) -> tuple[str, int, bytes]:
    reject_link(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"Not a regular file: {path}")
    if collect and before.st_size > MAX_TEXT_BYTES:
        raise RuntimeError(f"Unexpectedly large source/metadata file: {path}")
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            if collect and size > MAX_TEXT_BYTES:
                raise RuntimeError(f"Text size limit exceeded: {path}")
            digest.update(block)
            if collect:
                chunks.append(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed while being read: {path}")
    if size != after.st_size:
        raise RuntimeError(f"Read size mismatch: {path}")
    return digest.hexdigest(), size, b"".join(chunks)


def source_excerpt(relative: Path, raw: bytes) -> None:
    encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
    source = raw.decode(encoding)
    tree = ast.parse(source, filename=str(relative))  # Parse only; never execute.
    lines = source.splitlines()
    funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    selected = [
        n for n in funcs
        if n.name == "verify_r_library"
        or "tree" in n.name.lower()
        or "sha256" in n.name.lower()
        or n.name in {"file_hash", "hash_file", "clean_text", "resolve_project_path"}
    ]
    print("TOP-LEVEL FUNCTION INDEX:")
    for node in funcs:
        print(f"  {node.lineno:04d}-{node.end_lineno:04d} {node.name}")
    print("\nSELECTED DEFINITIONS (source text, not executed):")
    for node in selected:
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        print(f"\n--- {node.name}, lines {start}-{node.end_lineno} ---")
        for i in range(start - 1, node.end_lineno):
            print(f"{i + 1:04d}: {lines[i]}")
    if not any("tree" in n.name.lower() for n in selected):
        print("NOTE: No tree-named function found; inspect the function index and verifier.")
    print("\nRELATED TOP-LEVEL ASSIGNMENTS (not evaluated):")
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [n.id for target in targets for n in ast.walk(target) if isinstance(n, ast.Name)]
        if any(name == "SCRIPT_VERSION" or any(
            token in name.upper() for token in ("R_LIBRARY", "RSCRIPT", "ICLUSTER")
        ) for name in names):
            for i in range(node.lineno - 1, node.end_lineno):
                print(f"{i + 1:04d}: {lines[i]}")


def package_files(package: Path) -> list[Path]:
    def walk_error(error: OSError) -> None:
        raise error
    found: list[Path] = []
    for base, directories, filenames in os.walk(package, followlinks=False, onerror=walk_error):
        for name in directories:
            reject_link(Path(base) / name)
        for name in filenames:
            path = Path(base) / name
            reject_link(path)
            if not path.is_file():
                raise RuntimeError(f"Unexpected non-file package entry: {path}")
            found.append(path)
    return sorted(found, key=lambda p: p.relative_to(package).as_posix())


def diagnose(root: Path) -> None:
    root = root.resolve(strict=True)
    heading("Paper 5 - iClusterPlus tree-hash diagnostic v1")
    print(f"Project root: {root}")
    print(f"Python: {sys.version.split()[0]}")
    print("Mode: SOURCE_AND_PACKAGE_INVENTORY_ONLY")
    print("Project code/R execution: NO; fits: NO; authorization: NOT CREATED")
    print("Simulation/result directories: NOT ACCESSED")
    print("Output: stdout only; no project files are written")

    runner = checked_path(root, RUNNER_REL)
    runner_sha, _, runner_raw = stable_file(runner, collect=True)
    print(f"Runner SHA-256: {runner_sha}")
    if runner_sha != EXPECTED_RUNNER_SHA256:
        raise RuntimeError("Runner differs from the v2 SHA in the supplied traceback. Stop for review.")

    records: list[tuple[Path, str]] = []
    for relative in (RUNNER_REL, SMOKE_REL):
        path = checked_path(root, relative)
        digest, size, raw = stable_file(path, collect=True)
        if relative == RUNNER_REL and raw != runner_raw:
            raise RuntimeError("Runner changed during diagnosis.")
        records.append((path, digest))
        heading(f"SOURCE: {relative.as_posix()}")
        print(f"SHA-256: {digest}; bytes: {size}")
        source_excerpt(relative, raw)

    for relative in METADATA_RELS:
        path = checked_path(root, relative)
        digest, size, raw = stable_file(path, collect=True)
        value = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(value, dict):
            raise RuntimeError(f"Expected a JSON object: {relative}")
        records.append((path, digest))
        heading(f"FROZEN METADATA: {relative.as_posix()}")
        print(f"SHA-256: {digest}; bytes: {size}")
        print(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True))

    package = checked_path(root, PACKAGE_REL)
    if not package.is_dir():
        raise RuntimeError(f"Package directory is missing: {package}")
    paths = package_files(package)
    if not paths:
        raise RuntimeError("Package directory is empty.")
    heading("CURRENT ISOLATED PACKAGE - INDIVIDUAL FILE INVENTORY")
    print(f"Directory: {package}")
    print("relative_path\tbytes\tfile_sha256")
    total = 0
    for path in paths:
        digest, size, _ = stable_file(path)
        total += size
        records.append((path, digest))
        print(f"{path.relative_to(package).as_posix()}\t{size}\t{digest}")
    print(f"Observed file count (this inventory): {len(paths)}")
    print(f"Observed total bytes (this inventory): {total}")
    print("No replacement aggregate tree hash computed; no acceptance decision made.")
    print("Equal file counts/sizes alone do NOT establish identical package contents.")

    # Detect ordinary concurrent edits during collection; this is not an OS snapshot.
    if package_files(package) != paths:
        raise RuntimeError("Package file list changed during collection.")
    for path, earlier_hash in records:
        later_hash, _, _ = stable_file(path)
        if later_hash != earlier_hash:
            raise RuntimeError(f"File changed during collection: {path}")
    heading("DIAGNOSTIC COLLECTION COMPLETE - NOT BENCHMARK AUTHORIZATION")
    print("No content change detected in re-read source/metadata/package files.")
    print("Project code/R execution: NO; simulation/results access: NO; project writes: NO")
    print("Next step: review the original hash implementations and frozen metadata.")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    try:
        if len(sys.argv) != 1:
            raise RuntimeError("No command-line arguments. Run from the project root.")
        diagnose(Path.cwd())
    except (OSError, RuntimeError, ValueError, SyntaxError, UnicodeError) as exc:
        print(f"\nDIAGNOSTIC STOPPED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
