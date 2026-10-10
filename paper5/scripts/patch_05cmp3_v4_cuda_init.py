#!/usr/bin/env python3
"""Create Paper 5 comparator runner v4 from the exact diagnosed v3.

Run from the project root, without arguments:
    python -I -B ./paper5/scripts/patch_05cmp3_v4_cuda_init.py

Only two source edits are permitted:
  * insert torch.cuda.init() immediately before seed_everything(seed) in fit_move;
  * change SCRIPT_VERSION from v3 to v4.

The entire v3 SHA-256 is pinned. Source bytes and the complete AST are checked.
The existing source is never imported/executed. This patch does not import
PyTorch, launch R or MOVE, inspect datasets/results, or create authorization.
Only the new v4 source may be written. A different existing v4 is never replaced.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import io
import os
from pathlib import Path
import stat
import sys
import tokenize

BASE = "05cmp3_run_secondary_comparator_confirmatory_benchmark"
SOURCE_REL = Path("paper5/scripts") / (BASE + "_v3.py")
DEST_REL = Path("paper5/scripts") / (BASE + "_v4.py")
SOURCE_SHA256 = "41bee9179af62cc56104dbdfa4dfa47ec8d82110d5bd90078bb06c3bfacd21d8"
OLD_VERSION = (
    "05cmp3-run-secondary-comparator-confirmatory-benchmark-v3-"
    "frozen-icluster-tree-hash-sysmodules-loader-fix-"
    "two-stage-source-authorized-resumable-no-cli"
)
NEW_VERSION = (
    "05cmp3-run-secondary-comparator-confirmatory-benchmark-v4-"
    "explicit-cuda-init-frozen-icluster-tree-hash-sysmodules-loader-fix-"
    "two-stage-source-authorized-resumable-no-cli"
)
INSERT_TEXT = (
    "    # Initialize CUDA before seeding, cache cleanup, and allocator statistics.\n"
    "    torch.cuda.init()\n"
)
EXPECTED_PREFIX = '''VAE = api["VAE"]
training_loop = api["training_loop"]
make_dataloader = api["make_dataloader"]
seed_everything(seed)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(0)
'''


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_text(raw: bytes) -> tuple[str, str]:
    encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
    text = raw.decode(encoding)
    if text.encode(encoding) != raw:
        raise RuntimeError("Source encoding does not round-trip exactly.")
    return text, encoding


def top_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1 or matches[0].decorator_list:
        raise RuntimeError(f"Expected one undecorated top-level function: {name}")
    return matches[0]


def version_assignment(tree: ast.Module) -> ast.Assign:
    matches = [
        n for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "SCRIPT_VERSION" for t in n.targets)
    ]
    if len(matches) != 1 or len(matches[0].targets) != 1:
        raise RuntimeError("Expected one simple SCRIPT_VERSION assignment.")
    return matches[0]


def verify_ast_scope(before: ast.Module, after: ast.Module) -> None:
    """Undo only the allowed AST edits, then compare the whole source AST."""
    if ast.literal_eval(version_assignment(after).value) != NEW_VERSION:
        raise RuntimeError("New SCRIPT_VERSION validation failed.")
    normalized = copy.deepcopy(after)
    fit = top_function(normalized, "fit_move")
    expected_init = ast.parse("torch.cuda.init()").body[0]
    if len(fit.body) < 7 or ast.dump(fit.body[3]) != ast.dump(expected_init):
        raise RuntimeError("CUDA init is not in the required position before seeding.")
    del fit.body[3]
    version_assignment(normalized).value = copy.deepcopy(version_assignment(before).value)
    if ast.dump(normalized) != ast.dump(before):
        raise RuntimeError("Unexpected AST change outside fit_move init and SCRIPT_VERSION.")


def edit_bytes(original: bytes, edits: list[tuple[int, int, bytes]]) -> bytes:
    """Apply non-overlapping byte edits, and verify exact inverse reconstruction."""
    ordered = sorted(edits)
    cursor = 0
    parts: list[bytes] = []
    inverse: list[tuple[int, int, bytes]] = []
    shift = 0
    for start, end, replacement in ordered:
        if not (cursor <= start <= end <= len(original)):
            raise RuntimeError("Invalid or overlapping source edits.")
        parts.extend((original[cursor:start], replacement))
        new_start = start + shift
        inverse.append((new_start, new_start + len(replacement), original[start:end]))
        shift += len(replacement) - (end - start)
        cursor = end
    parts.append(original[cursor:])
    updated = b"".join(parts)
    restored = updated
    for start, end, previous in reversed(inverse):
        restored = restored[:start] + previous + restored[end:]
    if restored != original:
        raise RuntimeError("Byte-level inverse verification failed.")
    return updated


def build_patch(raw: bytes) -> bytes:
    """Transform source text only. No project code is imported or executed."""
    text, encoding = source_text(raw)
    compile(raw, str(SOURCE_REL), "exec", dont_inherit=True)
    before = ast.parse(text, filename=str(SOURCE_REL))
    version = version_assignment(before).value
    if not isinstance(version, ast.Constant) or version.value != OLD_VERSION:
        raise RuntimeError("Unexpected v3 SCRIPT_VERSION.")
    fit = top_function(before, "fit_move")
    required = ast.parse(EXPECTED_PREFIX).body
    if len(fit.body) < len(required) or any(
        ast.dump(a) != ast.dump(b) for a, b in zip(fit.body, required)
    ):
        raise RuntimeError("fit_move prefix differs from the supplied diagnostic.")

    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line.encode("utf-8")))
    seed = fit.body[3]
    seed_line = lines[seed.lineno - 1]
    if seed.col_offset != 4 or seed_line.rstrip("\r\n") != "    seed_everything(seed)":
        raise RuntimeError("Unexpected seed statement indentation or line contents.")
    if seed_line.endswith("\r\n"):
        newline = "\r\n"
    elif seed_line.endswith("\n"):
        newline = "\n"
    else:
        raise RuntimeError("Seed line has no supported line ending.")
    assert version.end_lineno is not None and version.end_col_offset is not None
    a = offsets[version.lineno - 1] + version.col_offset
    b = offsets[version.end_lineno - 1] + version.end_col_offset
    insert_at = offsets[seed.lineno - 1]
    edits = [
        (a, b, repr(NEW_VERSION).encode("utf-8")),
        (insert_at, insert_at, INSERT_TEXT.replace("\n", newline).encode("utf-8")),
    ]
    updated_utf8 = edit_bytes(text.encode("utf-8"), edits)
    result = updated_utf8.decode("utf-8").encode(encoding)
    compile(result, str(DEST_REL), "exec", dont_inherit=True)
    after = ast.parse(source_text(result)[0], filename=str(DEST_REL))
    verify_ast_scope(before, after)
    return result


def reject_link(path: Path) -> None:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise RuntimeError(f"Refusing a symlink/reparse point: {path}")


def checked_path(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Invalid project-relative path: {relative}")
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
    if (
        (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
        or len(data) != after.st_size
    ):
        raise RuntimeError(f"File changed during read: {path}")
    return data


def apply_patch(root: Path) -> None:
    root = root.resolve(strict=True)
    source = checked_path(root, SOURCE_REL)
    original = read_stable(source)
    observed = sha256(original)
    if observed != SOURCE_SHA256:
        raise RuntimeError(
            f"v3 SHA-256 mismatch. Expected {SOURCE_SHA256}; observed {observed}. "
            "No v4 written."
        )
    print("Pinned v3 source SHA-256: PASS")
    result = build_patch(original)
    print("Byte-level patch scope verification: PASS")
    print("Complete AST scope verification: PASS")
    print("v4 syntax compilation: PASS")
    if read_stable(source) != original:
        raise RuntimeError("v3 changed during patch preparation. No v4 written.")

    checked_path(root, DEST_REL.parent)
    destination = root / DEST_REL
    created = False
    try:
        handle = destination.open("xb")
    except FileExistsError:
        if read_stable(destination) != result:
            raise RuntimeError("A DIFFERENT v4 already exists. It was NOT overwritten.")
    else:
        try:
            with handle:
                handle.write(result)
                handle.flush()
                os.fsync(handle.fileno())
            created = True
        except BaseException:
            # Only the new file owned by this exclusive creation is cleaned up.
            destination.unlink(missing_ok=True)
            raise
    if read_stable(destination) != result:
        raise RuntimeError("v4 post-write verification failed. Do NOT run it.")
    if read_stable(source) != original:
        raise RuntimeError("v3 changed during final verification. Do NOT run v4.")

    print("PATCH: PASS")
    print("v4 source:", "CREATED" if created else "ALREADY_PRESENT_IDENTICAL (not rewritten)")
    print("v3 unchanged: PASS")
    print("v4 SCRIPT_VERSION:", NEW_VERSION)
    print("v4 SHA-256:", sha256(result))
    print("Changes: torch.cuda.init() before fit_move seeding; SCRIPT_VERSION")
    print("Earlier fixes, training settings and authorization logic: UNCHANGED")
    print("Runner/MOVE/R/CUDA execution by patch: NO")
    print("Data/results/package/contract access by patch: NO")
    print("Scientific source authorization: NOT CREATED")


def main() -> int:
    try:
        if len(sys.argv) != 1:
            raise RuntimeError("No arguments are accepted. Run from the project root.")
        apply_patch(Path.cwd())
    except Exception as exc:
        print(f"PATCH STOPPED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
