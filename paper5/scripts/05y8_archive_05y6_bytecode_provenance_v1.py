#!/usr/bin/env python3
from __future__ import annotations

import dis
import hashlib
import importlib.util
import io
import json
import marshal
import struct
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_VERSION = "05y8-archive-05y6-bytecode-provenance-v1-no-cli"
EXPECTED_PYC_SHA256 = "bad3ee4833918b578c76daa013ee8e92de72f945c0f2b976db7dc75873ce39e8"
EXPECTED_SOURCE_NAME = "05y6_aggregate_prospective_q10_validation.py"

ROOT = Path(__file__).resolve().parents[1]
PYC = ROOT / "scripts" / "__pycache__" / "05y6_aggregate_prospective_q10_validation.cpython-310.pyc"
EXPECTED_SOURCE = ROOT / "scripts" / EXPECTED_SOURCE_NAME

OUT = ROOT / "method_contract" / "05y8"
HEADER_JSON = OUT / "05y6_pyc_header.json"
DIS_TXT = OUT / "05y6_disassembly_readonly.txt"
CODE_JSON = OUT / "05y6_code_object_inventory.json"
README = OUT / "05y8_README.txt"
LOCK = OUT / "05y8_bytecode_archive_lock.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def code_inventory(code: types.CodeType, qualname: str = "<module>") -> list[dict]:
    rows = [{
        "qualname": qualname,
        "co_name": code.co_name,
        "co_filename": code.co_filename,
        "co_firstlineno": code.co_firstlineno,
        "co_argcount": code.co_argcount,
        "co_posonlyargcount": getattr(code, "co_posonlyargcount", None),
        "co_kwonlyargcount": code.co_kwonlyargcount,
        "co_nlocals": code.co_nlocals,
        "co_stacksize": code.co_stacksize,
        "co_flags": code.co_flags,
        "co_names": list(code.co_names),
        "co_varnames": list(code.co_varnames),
        "co_freevars": list(code.co_freevars),
        "co_cellvars": list(code.co_cellvars),
        "bytecode_sha256": hashlib.sha256(code.co_code).hexdigest(),
    }]
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            rows.extend(code_inventory(const, f"{qualname}.{const.co_name}"))
    return rows


def main() -> None:
    print("=" * 100)
    print("Paper 5 - archive frozen 05y6 bytecode provenance")
    print("=" * 100)
    print(f"Script version: {SCRIPT_VERSION}")
    print("Frozen 05y6 execution: NOT PERFORMED")
    print("Scientific shards/results: NOT READ")
    print()

    if not PYC.is_file():
        raise FileNotFoundError(f"Frozen 05y6 pyc missing: {PYC}")

    pyc_sha = sha256_file(PYC)
    if pyc_sha.lower() != EXPECTED_PYC_SHA256:
        raise RuntimeError(f"05y6 pyc SHA256 mismatch: {pyc_sha}")

    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"05y8 output directory already exists and is non-empty: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    raw = PYC.read_bytes()
    if len(raw) < 16:
        raise RuntimeError("PYC is too short to contain a valid 16-byte header.")

    magic = raw[:4]
    flags = struct.unpack("<I", raw[4:8])[0]
    hash_based = bool(flags & 0x01)
    check_source = bool(flags & 0x02) if hash_based else None

    header = {
        "archive_role": "READ_ONLY_PROVENANCE_ARCHIVE_OF_PREEXISTING_05Y6_BYTECODE",
        "script_version": SCRIPT_VERSION,
        "pyc_path": str(PYC.relative_to(ROOT)).replace("\\", "/"),
        "pyc_sha256": pyc_sha,
        "pyc_size_bytes": PYC.stat().st_size,
        "python_runtime_used_for_archive": sys.version,
        "runtime_magic_number_hex": importlib.util.MAGIC_NUMBER.hex(),
        "pyc_magic_number_hex": magic.hex(),
        "magic_matches_current_runtime": magic == importlib.util.MAGIC_NUMBER,
        "flags_uint32": flags,
        "hash_based_pyc": hash_based,
        "expected_source_path": str(EXPECTED_SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "expected_source_exists_at_archive_time": EXPECTED_SOURCE.is_file(),
    }

    if EXPECTED_SOURCE.is_file():
        header["recovered_source_sha256"] = sha256_file(EXPECTED_SOURCE)
        header["recovered_source_size_bytes"] = EXPECTED_SOURCE.stat().st_size
        header["recovered_source_mtime_utc"] = datetime.fromtimestamp(
            EXPECTED_SOURCE.stat().st_mtime, tz=timezone.utc
        ).isoformat()

    if hash_based:
        header["check_source_flag"] = check_source
        header["source_hash_8bytes_hex"] = raw[8:16].hex()
    else:
        source_mtime = struct.unpack("<I", raw[8:12])[0]
        source_size = struct.unpack("<I", raw[12:16])[0]
        header["source_mtime_unix_from_pyc_header"] = source_mtime
        header["source_mtime_utc_from_pyc_header"] = datetime.fromtimestamp(source_mtime, tz=timezone.utc).isoformat()
        header["source_size_bytes_from_pyc_header"] = source_size

    with PYC.open("rb") as f:
        f.read(16)
        code = marshal.load(f)

    if not isinstance(code, types.CodeType):
        raise RuntimeError("PYC payload did not decode to a Python code object.")

    header["embedded_co_filename"] = code.co_filename
    HEADER_JSON.write_text(json.dumps(header, indent=2) + "\n", encoding="utf-8")

    inventory = {
        "status": "READ_ONLY_CODE_OBJECT_INVENTORY",
        "pyc_sha256": pyc_sha,
        "note": "Derived from pre-existing frozen bytecode; not source-code recovery and not executed.",
        "code_objects": code_inventory(code),
    }
    CODE_JSON.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")

    buf = io.StringIO()
    buf.write("READ-ONLY DISASSEMBLY OF FROZEN 05Y6 BYTECODE\n")
    buf.write("================================================\n\n")
    buf.write(f"PYC SHA256: {pyc_sha}\n")
    buf.write(f"Embedded source filename: {code.co_filename}\n")
    buf.write("This disassembly was produced without executing the code object.\n")
    buf.write("It is not a claim of exact source-code reconstruction.\n\n")
    dis.dis(code, file=buf, depth=None)
    DIS_TXT.write_text(buf.getvalue(), encoding="utf-8")

    README.write_text(
        "Paper 5 05y8 - frozen 05y6 bytecode provenance archive\n"
        "=======================================================\n\n"
        "Purpose: preserve read-only evidence for the pre-existing 05y6 aggregator whose .py source is absent.\n"
        "No 05y6 code is executed by this stage. No 05y5 shard or scientific result is read or recomputed.\n\n"
        "Artifacts:\n"
        "  05y6_pyc_header.json              parsed CPython .pyc header and embedded filename\n"
        "  05y6_disassembly_readonly.txt     disassembly of the frozen code object, never executed here\n"
        "  05y6_code_object_inventory.json   code-object metadata and bytecode hashes\n\n"
        "Important: disassembly/inventory are provenance aids, not exact source-code reconstruction.\n",
        encoding="utf-8",
    )

    generated = {}
    for path in [HEADER_JSON, DIS_TXT, CODE_JSON, README]:
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        generated[rel] = {"path": rel, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}

    self_path = Path(__file__).resolve()
    lock = {
        "status": "05Y8_05Y6_BYTECODE_PROVENANCE_ARCHIVED",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_script": {
            "path": str(self_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(self_path),
            "size_bytes": self_path.stat().st_size,
        },
        "frozen_05y6_pyc": {
            "path": str(PYC.relative_to(ROOT)).replace("\\", "/"),
            "sha256": pyc_sha,
            "size_bytes": PYC.stat().st_size,
        },
        "expected_source_exists_at_archive_time": EXPECTED_SOURCE.is_file(),
        "recovered_source_if_present": (
            {
                "path": str(EXPECTED_SOURCE.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(EXPECTED_SOURCE),
                "size_bytes": EXPECTED_SOURCE.stat().st_size,
            }
            if EXPECTED_SOURCE.is_file()
            else None
        ),
        "scientific_endpoint_recomputation": False,
        "frozen_05y6_execution": False,
        "generated_files": generated,
        "interpretation": (
            "Documents the exact pre-existing bytecode and its CPython header/code-object structure. "
            "Does not replace the missing original .py source and does not claim exact source reconstruction."
        ),
    }
    LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    print("05y8 bytecode provenance archive: COMPLETE")
    print(f"PYC SHA256: {pyc_sha}")
    print(f"Hash-based PYC: {hash_based}")
    if not hash_based:
        print(f"Header source mtime UTC: {header['source_mtime_utc_from_pyc_header']}")
        print(f"Header source size: {header['source_size_bytes_from_pyc_header']}")
    print(f"Expected source currently exists: {EXPECTED_SOURCE.is_file()}")
    print(f"Output: {OUT.relative_to(ROOT)}")
    print(f"Lock:   {LOCK.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
