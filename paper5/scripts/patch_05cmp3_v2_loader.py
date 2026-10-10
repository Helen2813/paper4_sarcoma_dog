"""Apply a source-only loader fix to the copied 05cmp3 v2 runner.

Run without arguments from the paper4_sarcoma_dog project root.
Does not import or execute either runner, any helper, or any scientific data.
Only an unmodified copy of the exact reported v1 source is accepted as input.
"""
from __future__ import annotations

import ast
import hashlib
import io
import os
from pathlib import Path
import re
import tempfile
import tokenize

EXPECTED_SHA256 = "9f752ee29ba5248fdc718709917aae06b27693a3fafbdab19f85b325f829c1be"
OLD_VERSION = "05cmp3-run-secondary-comparator-confirmatory-benchmark-v1-two-stage-source-authorized-resumable-no-cli"
NEW_VERSION = "05cmp3-run-secondary-comparator-confirmatory-benchmark-v2-sysmodules-loader-fix-two-stage-source-authorized-resumable-no-cli"
BASENAME = "05cmp3_run_secondary_comparator_confirmatory_benchmark"

OLD_BLOCK = (
    "    module = importlib.util.module_from_spec(spec)",
    "    spec.loader.exec_module(module)",
)
NEW_BLOCK = (
    "    module = importlib.util.module_from_spec(spec)",
    "    import sys as _paper5_loader_sys",
    "",
    "    _paper5_loader_missing = object()",
    "    _paper5_loader_previous = _paper5_loader_sys.modules.get(",
    "        spec.name, _paper5_loader_missing",
    "    )",
    "    _paper5_loader_sys.modules[spec.name] = module",
    "    try:",
    "        spec.loader.exec_module(module)",
    "    except BaseException:",
    "        if _paper5_loader_previous is _paper5_loader_missing:",
    "            _paper5_loader_sys.modules.pop(spec.name, None)",
    "        else:",
    "            _paper5_loader_sys.modules[spec.name] = _paper5_loader_previous",
    "        raise",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def version_value(tree: ast.Module) -> ast.expr:
    found = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == "SCRIPT_VERSION" for t in targets):
            if node.value is not None:
                found.append(node.value)
    if len(found) != 1:
        raise RuntimeError("Expected exactly one top-level SCRIPT_VERSION definition.")
    return found[0]


def make_patch(original: bytes) -> bytes:
    """Transform source in memory; refuse unexpected source layouts."""
    encoding, _ = tokenize.detect_encoding(io.BytesIO(original).readline)
    source = original.decode(encoding)
    if source.encode(encoding) != original:
        raise RuntimeError("Source encoding does not round-trip exactly.")
    tree = ast.parse(source)
    functions = [n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name == "import_linear_helper"]
    if len(functions) != 1:
        raise RuntimeError("Expected exactly one import_linear_helper function.")
    helper = functions[0]
    if any(isinstance(n, ast.Name) and n.id.startswith("_paper5_loader_")
           for n in ast.walk(helper)):
        raise RuntimeError("Loader-fix local names already exist; refusing to overwrite.")

    # Locate precisely the two executable lines shown in the reported traceback.
    matches = []
    for newline in ("\n", "\r\n"):
        old = newline.join(OLD_BLOCK) + newline
        for m in re.finditer(re.escape(old), source):
            line = source.count("\n", 0, m.start()) + 1
            if m.start() and source[m.start() - 1] != "\n":
                continue
            matches.append((m.start(), m.end(), newline, line))
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one matching two-line loader block.")
    start, end, newline, line = matches[0]
    if not (helper.lineno < line and line + 1 <= helper.end_lineno):
        raise RuntimeError("Matching loader block is outside import_linear_helper.")
    loader_only = source[:start] + newline.join(NEW_BLOCK) + newline + source[end:]

    # Change only SCRIPT_VERSION's expression, including split-string definitions.
    value = version_value(ast.parse(loader_only))
    if ast.literal_eval(value) != OLD_VERSION:
        raise RuntimeError("SCRIPT_VERSION differs from the reported v1 version.")
    lines = loader_only.splitlines(keepends=True)
    utf8 = loader_only.encode("utf-8")
    start = sum(len(s.encode("utf-8")) for s in lines[:value.lineno - 1]) + value.col_offset
    end = sum(len(s.encode("utf-8")) for s in lines[:value.end_lineno - 1]) + value.end_col_offset
    updated = (utf8[:start] + repr(NEW_VERSION).encode("utf-8") + utf8[end:]).decode("utf-8")
    result = updated.encode(encoding)
    compile(result, BASENAME + "_v2.py", "exec", dont_inherit=True)
    if ast.literal_eval(version_value(ast.parse(updated))) != NEW_VERSION:
        raise RuntimeError("Updated version validation failed.")
    return result


def main() -> None:
    scripts = Path.cwd() / "paper5" / "scripts"
    v1 = scripts / (BASENAME + "_v1.py")
    v2 = scripts / (BASENAME + "_v2.py")
    if not v1.is_file() or not v2.is_file():
        raise RuntimeError("Run from the project root; both existing v1 and copied v2 are required.")
    if v1.is_symlink() or v2.is_symlink() or os.path.samefile(v1, v2):
        raise RuntimeError("v1 and v2 must be separate regular source files, not links.")
    original = v1.read_bytes()
    if sha256(original) != EXPECTED_SHA256:
        raise RuntimeError("v1 SHA-256 mismatch. No source was changed.")
    if v2.read_bytes() != original:
        raise RuntimeError("v2 is not the unchanged v1 copy. No source was changed.")
    result = make_patch(original)

    # Finish all checks before replacing v2. Never write v1 or frozen helpers.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=scripts, prefix=".05cmp3_patch_", suffix=".tmp", delete=False) as f:
            temp_path = Path(f.name)
            f.write(result)
            f.flush()
            os.fsync(f.fileno())
        if v1.read_bytes() != original or v2.read_bytes() != original:
            raise RuntimeError("Source changed during patch preparation; refusing replacement.")
        os.replace(temp_path, v2)
        temp_path = None
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    if v1.read_bytes() != original or v2.read_bytes() != result:
        raise RuntimeError("Post-write verification failed. Do not execute the runner.")
    print("PATCH: PASS")
    print("Changes: import_linear_helper loader registration/cleanup; SCRIPT_VERSION")
    print("v1 unchanged: PASS")
    print("v2 syntax compilation: PASS")
    print("Runner/helper execution: NO")
    print("Scientific data access: NO")
    print("v2 SCRIPT_VERSION:", NEW_VERSION)
    print("v2 SHA-256:", sha256(result))
    print("Scientific source authorization: NOT CREATED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit("STOP: " + str(exc)) from exc
