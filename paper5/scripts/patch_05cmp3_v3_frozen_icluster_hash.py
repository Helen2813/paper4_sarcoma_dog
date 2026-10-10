#!/usr/bin/env python3
"""Create Paper 5 runner v3 without changing v1/v2, frozen records, or packages.

Run without arguments from the Windows project root, in move_1_5_0:
    python -I -B ./paper5/scripts/patch_05cmp3_v3_frozen_icluster_hash.py

Reads five pinned source/metadata files and the isolated iClusterPlus package.
Reproduces BOTH hashes before creating v3. It never imports/executes project
code, starts R, reads simulation/results directories, or creates authorization.
Only a new v3 source file is written. Existing identical v3 is left untouched;
existing different v3 is never overwritten.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path, PureWindowsPath
import stat
import sys
import tokenize

BASE = "05cmp3_run_secondary_comparator_confirmatory_benchmark"
V1_REL = Path("paper5/scripts") / (BASE + "_v1.py")
V2_REL = Path("paper5/scripts") / (BASE + "_v2.py")
V3_REL = Path("paper5/scripts") / (BASE + "_v3.py")
SMOKE_REL = Path("paper5/scripts/05cmp0c_smoke_test_icluster2b_cpu_v4.py")
ENV_REL = Path("paper5/method_contract/05cmp0c/icluster2b_cpu_smoke_environment.json")
LOCK_REL = Path("paper5/method_contract/05cmp0c/icluster2b_cpu_smoke_lock.json")
PACKAGE_REL = Path("paper5/comparator_envs/r_4_6_bioc_3_23/iClusterPlus")
PINS = {
    V1_REL: "9f752ee29ba5248fdc718709917aae06b27693a3fafbdab19f85b325f829c1be",
    V2_REL: "371f4475ab639607496f198a3420fc47f1c7fa6f063d1e33574260c3987e41f5",
    SMOKE_REL: "f97154a0ebe11c57c4a6e6a3cb381c3f7147913580aea651b0d3b698516d3a52",
    ENV_REL: "c2bc881d056ada211929626dac80fd07c345c66b4e36d41ed77ab9c2ac251b76",
    LOCK_REL: "c3c2b08c443d973e72fae00e67f5a0577886cbfc25c4935a8c4edb0a5b3cb75f",
}
FROZEN_TREE_SHA = "cad0c995150e778026fb227ac533a974d77106e431c65a3802aeda4254f6fd70"
V2_TREE_SHA = "b2916d0dd122b71b04a849e0adcc1465df768d5b91204f910da7f098a3f65cd1"
EXPECTED_COUNT = 43
EXPECTED_BYTES = 23258345
OLD_VERSION = (
    "05cmp3-run-secondary-comparator-confirmatory-benchmark-v2-"
    "sysmodules-loader-fix-two-stage-source-authorized-resumable-no-cli"
)
NEW_VERSION = (
    "05cmp3-run-secondary-comparator-confirmatory-benchmark-v3-"
    "frozen-icluster-tree-hash-sysmodules-loader-fix-"
    "two-stage-source-authorized-resumable-no-cli"
)
NEW_HELPER = "icluster_tree_sha256_frozen_05cmp0c"
NEW_HELPER_SOURCE = r'''def icluster_tree_sha256_frozen_05cmp0c(root: Path) -> tuple[str, int, int]:
    # Reproduce the exact 05cmp0c v4 format, including native Windows Path order.
    # Do not substitute sorted POSIX strings, omit size, or use raw digest bytes.
    # This helper is used ONLY for the frozen iClusterPlus package verification.
    if not root.is_dir():
        raise FileNotFoundError(f"Directory missing: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = int(path.stat().st_size)
        file_hash = sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        total_bytes += size
    return digest.hexdigest(), len(files), total_bytes
'''


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reject_link(path: Path) -> None:
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & 0x400):
        raise RuntimeError(f"Refusing symlink/reparse point: {path}")


def checked_path(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Invalid relative path: {relative}")
    path = root
    for part in relative.parts:
        path = path / part
        reject_link(path)
    return path


def read_stable(path: Path) -> bytes:
    reject_link(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"Not a regular file: {path}")
    data = path.read_bytes()
    after = path.stat()
    if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
            or len(data) != after.st_size):
        raise RuntimeError(f"File changed during read: {path}")
    return data


def source_text(raw: bytes) -> tuple[str, str]:
    encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
    text = raw.decode(encoding)
    if text.encode(encoding) != raw:
        raise RuntimeError("Source encoding does not round-trip.")
    return text, encoding


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(found) != 1 or found[0].decorator_list:
        raise RuntimeError(f"Expected one undecorated function: {name}")
    return found[0]


def assignment(tree: ast.Module, name: str) -> ast.expr:
    found = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            found.append(node.value)
    if len(found) != 1 or found[0] is None:
        raise RuntimeError(f"Expected one assigned value: {name}")
    return found[0]


def check_expected_hash(tree: ast.Module) -> None:
    value = assignment(tree, "EXPECTED")
    if not isinstance(value, ast.Dict):
        raise RuntimeError("EXPECTED is not a literal dictionary; stop for source review.")
    entries = [v for k, v in zip(value.keys, value.values)
               if isinstance(k, ast.Constant) and k.value == "icluster_tree_sha256"]
    if len(entries) != 1 or ast.literal_eval(entries[0]) != FROZEN_TREE_SHA:
        raise RuntimeError("Runner's expected iClusterPlus hash differs from the frozen hash.")


def tree_call(verifier: ast.FunctionDef, name: str) -> ast.Call:
    found = [n for n in ast.walk(verifier) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == name]
    if len(found) != 1:
        raise RuntimeError(f"Expected one {name} call inside verify_r_library.")
    call = found[0]
    if (len(call.args) != 1 or call.keywords
            or not isinstance(call.args[0], ast.Name)
            or call.args[0].id != "package_root"):
        raise RuntimeError("Unexpected package-tree call arguments.")
    return call


def build_patch(original: bytes, smoke_raw: bytes) -> bytes:
    """Edit only the version, one call target, and add the dedicated helper."""
    text, encoding = source_text(original)
    tree = ast.parse(text)
    old_version = assignment(tree, "SCRIPT_VERSION")
    if ast.literal_eval(old_version) != OLD_VERSION:
        raise RuntimeError("Unexpected v2 SCRIPT_VERSION.")
    check_expected_hash(tree)
    if any((isinstance(n, ast.Name) and n.id == NEW_HELPER)
           or (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
               and n.name == NEW_HELPER) for n in ast.walk(tree)):
        raise RuntimeError("New helper name already exists in v2.")
    verifier = function(tree, "verify_r_library")
    call = tree_call(verifier, "tree_sha256")
    frozen_tree = function(ast.parse(source_text(smoke_raw)[0]), "tree_sha256")
    new_helper = function(ast.parse(NEW_HELPER_SOURCE), NEW_HELPER)
    # The first statement only validates the directory. ALL hash-building
    # statements must be AST-identical to the pinned 05cmp0c v4 implementation.
    if (ast.dump(ast.Module(body=frozen_tree.body[1:], type_ignores=[]))
            != ast.dump(ast.Module(body=new_helper.body[1:], type_ignores=[]))):
        raise RuntimeError("New hash-building code differs from pinned 05cmp0c v4.")
    required_first = ast.parse("require_dir(root)").body[0]
    if ast.dump(frozen_tree.body[0]) != ast.dump(required_first):
        raise RuntimeError("Unexpected frozen directory validation.")

    utf8 = text.encode("utf-8")
    lines = text.splitlines(keepends=True)
    def offset(line: int, column: int) -> int:
        return sum(len(s.encode("utf-8")) for s in lines[:line - 1]) + column
    def span(node: ast.AST) -> tuple[int, int]:
        return (offset(node.lineno, node.col_offset),
                offset(node.end_lineno, node.end_col_offset))
    newline = "\r\n" if "\r\n" in text else "\n"
    insert_at = offset(verifier.lineno, 0)
    helper_bytes = (NEW_HELPER_SOURCE.replace("\n", newline) + newline * 2).encode("utf-8")
    a, b = span(old_version)
    c, d = span(call.func)
    edits = [(a, b, repr(NEW_VERSION).encode("utf-8")),
             (c, d, NEW_HELPER.encode("ascii")),
             (insert_at, insert_at, helper_bytes)]
    updated = utf8
    for start, end, replacement in sorted(edits, reverse=True):
        updated = updated[:start] + replacement + updated[end:]
    result = updated.decode("utf-8").encode(encoding)
    compile(result, BASE + "_v3.py", "exec", dont_inherit=True)

    after = ast.parse(source_text(result)[0])
    if ast.literal_eval(assignment(after, "SCRIPT_VERSION")) != NEW_VERSION:
        raise RuntimeError("New version verification failed.")
    if ast.dump(function(after, NEW_HELPER)) != ast.dump(new_helper):
        raise RuntimeError("Inserted helper validation failed.")
    # Restore the three allowed edits in the AST and require exact identity.
    normalized = copy.deepcopy(after)
    normalized.body = [n for n in normalized.body
                       if not (isinstance(n, ast.FunctionDef) and n.name == NEW_HELPER)]
    restored_call = tree_call(function(normalized, "verify_r_library"), NEW_HELPER)
    restored_call.func.id = "tree_sha256"
    for node in normalized.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "SCRIPT_VERSION" for t in targets):
                node.value = copy.deepcopy(old_version)
    if ast.dump(normalized) != ast.dump(tree):
        raise RuntimeError("Unexpected AST changes outside the authorized patch scope.")
    return result


def package_inventory(package: Path) -> list[tuple[str, int, str]]:
    if not package.is_dir():
        raise RuntimeError(f"Package directory missing: {package}")
    reject_link(package)
    records = []
    def walk_error(error: OSError) -> None:
        raise error
    for base, directories, filenames in os.walk(package, followlinks=False, onerror=walk_error):
        for name in directories:
            reject_link(Path(base) / name)
        for name in filenames:
            path = Path(base) / name
            data = read_stable(path)
            records.append((path.relative_to(package).as_posix(), len(data), sha(data)))
    return sorted(records)


def inventory_hashes(records: list[tuple[str, int, str]]) -> tuple[str, str]:
    # PureWindowsPath reproduces the historical Windows ordering without
    # running any code from the original project source files.
    frozen = hashlib.sha256()
    for relative, size, digest in sorted(records, key=lambda r: PureWindowsPath(r[0])):
        frozen.update(relative.encode("utf-8"))
        frozen.update(b"\0")
        frozen.update(str(size).encode("ascii"))
        frozen.update(b"\0")
        frozen.update(digest.encode("ascii"))
        frozen.update(b"\n")
    v2 = hashlib.sha256()
    for relative, _, digest in sorted(records, key=lambda r: r[0]):
        v2.update(relative.encode("utf-8"))
        v2.update(b"\0")
        v2.update(bytes.fromhex(digest))
        v2.update(b"\0")
    return frozen.hexdigest(), v2.hexdigest()


def apply_patch(root: Path) -> None:
    root = root.resolve(strict=True)
    snapshots = {}
    for relative, expected in PINS.items():
        raw = read_stable(checked_path(root, relative))
        if sha(raw) != expected:
            raise RuntimeError(f"Pinned SHA mismatch: {relative}. No new source written.")
        snapshots[relative] = raw
    environment = json.loads(snapshots[ENV_REL].decode("utf-8-sig"))
    lock = json.loads(snapshots[LOCK_REL].decode("utf-8-sig"))
    r = environment["R"]
    if (r["iclusterplus_tree_sha256"] != FROZEN_TREE_SHA
            or r["iclusterplus_tree_file_count"] != EXPECTED_COUNT
            or r["iclusterplus_tree_total_bytes"] != EXPECTED_BYTES
            or r["iclusterplus_version"] != "1.47.3"):
        raise RuntimeError("Frozen environment fields differ from the supplied diagnostic.")
    if (lock["environment"]["sha256"] != sha(snapshots[ENV_REL])
            or lock["script_source"]["sha256"] != sha(snapshots[SMOKE_REL])):
        raise RuntimeError("Frozen smoke lock/source/environment linkage mismatch.")
    package = checked_path(root, PACKAGE_REL)
    records = package_inventory(package)
    frozen_hash, v2_hash = inventory_hashes(records)
    print("Pinned source/metadata hashes: PASS")
    print("Current package file count:", len(records))
    print("Current package total bytes:", sum(row[1] for row in records))
    print("Frozen 05cmp0c Windows-format SHA-256:", frozen_hash)
    print("Original v2-format SHA-256:", v2_hash)
    if (len(records) != EXPECTED_COUNT or sum(row[1] for row in records) != EXPECTED_BYTES
            or frozen_hash != FROZEN_TREE_SHA or v2_hash != V2_TREE_SHA):
        raise RuntimeError("Package no longer reproduces BOTH reported hashes. No v3 written.")
    print("Frozen package match and v2 mismatch reproduction: PASS")
    result = build_patch(snapshots[V2_REL], snapshots[SMOKE_REL])
    print("New helper hash body equals frozen 05cmp0c v4: PASS")
    print("AST scope verification: PASS")

    # Repeat reads before any write; this detects ordinary concurrent changes,
    # not an OS-level atomic snapshot. Scientific execution rechecks the package.
    if package_inventory(package) != records:
        raise RuntimeError("Package changed during patch preparation. No v3 written.")
    for relative, raw in snapshots.items():
        if read_stable(checked_path(root, relative)) != raw:
            raise RuntimeError(f"Input changed during patch preparation: {relative}")
    checked_path(root, V3_REL.parent)
    destination = root / V3_REL
    created = False
    try:
        handle = destination.open("xb")
    except FileExistsError:
        if read_stable(destination) != result:
            raise RuntimeError("A DIFFERENT v3 already exists; it was NOT overwritten.")
    else:
        # Exclusive creation never overwrites v1, v2, or any existing v3.
        try:
            with handle:
                handle.write(result)
                handle.flush()
                os.fsync(handle.fileno())
            created = True
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
    if read_stable(destination) != result:
        raise RuntimeError("v3 post-write verification failed. Do NOT run the benchmark.")
    for relative, raw in snapshots.items():
        if read_stable(checked_path(root, relative)) != raw:
            raise RuntimeError(f"Post-write input check failed: {relative}. Do NOT run v3.")
    print("PATCH: PASS")
    print("v3 source:", "CREATED" if created else "ALREADY_PRESENT_IDENTICAL (not rewritten)")
    print("v1/v2/frozen source and metadata unchanged: PASS")
    print("v3 syntax compilation: PASS")
    print("v3 SCRIPT_VERSION:", NEW_VERSION)
    print("v3 SHA-256:", sha(result))
    print("Changes: dedicated frozen iClusterPlus hash helper; one verifier call; SCRIPT_VERSION")
    print("Original generic tree_sha256 and all other runner logic: UNCHANGED")
    print("Project code/R execution: NO; simulation/results access: NO")
    print("Scientific source authorization: NOT CREATED")


def main() -> int:
    try:
        if len(sys.argv) != 1:
            raise RuntimeError("No arguments. Run from the project root.")
        if os.name != "nt":
            raise RuntimeError("Run this patch in the existing Windows project, not Linux/macOS.")
        apply_patch(Path.cwd())
    except Exception as exc:
        print(f"PATCH STOPPED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
