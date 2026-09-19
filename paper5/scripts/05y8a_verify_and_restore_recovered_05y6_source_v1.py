#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import marshal
import os
import shutil
import struct
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_VERSION = "05y8a-verify-and-restore-recovered-05y6-source-v1-no-cli"

EXPECTED_PYC_SHA256 = "bad3ee4833918b578c76daa013ee8e92de72f945c0f2b976db7dc75873ce39e8"
EXPECTED_SOURCE_BASENAME = "05y6_aggregate_prospective_q10_validation.py"

ROOT = Path(__file__).resolve().parents[1]
PYC = ROOT / "scripts" / "__pycache__" / "05y6_aggregate_prospective_q10_validation.cpython-310.pyc"
RESTORED_SOURCE = ROOT / "scripts" / EXPECTED_SOURCE_BASENAME

# Candidate discovered by the pre-archive source search.
TELEGRAM_CANDIDATE = (
    Path.home()
    / "Downloads"
    / "Telegram Desktop"
    / EXPECTED_SOURCE_BASENAME
)

OUT = ROOT / "method_contract" / "05y8a"
ARCHIVE_DIR = OUT / "recovered_source"
ARCHIVED_SOURCE = ARCHIVE_DIR / EXPECTED_SOURCE_BASENAME
RESULT = OUT / "05y6_recovered_source_verification.json"
README = OUT / "05y8a_README.txt"
LOCK = OUT / "05y8a_recovered_source_lock.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def code_object_inventory(code: types.CodeType, qualname: str = "<module>") -> list[dict]:
    rows = [{
        "qualname": qualname,
        "co_name": code.co_name,
        "co_filename": code.co_filename,
        "co_firstlineno": code.co_firstlineno,
        "co_argcount": code.co_argcount,
        "co_kwonlyargcount": code.co_kwonlyargcount,
        "co_nlocals": code.co_nlocals,
        "co_stacksize": code.co_stacksize,
        "co_flags": code.co_flags,
        "co_code_sha256": sha256_bytes(code.co_code),
    }]
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            rows.extend(code_object_inventory(const, f"{qualname}.{const.co_name}"))
    return rows


def compare_inventory(a: types.CodeType, b: types.CodeType) -> bool:
    return code_object_inventory(a) == code_object_inventory(b)


def main() -> None:
    print("=" * 100)
    print("Paper 5 - verify and restore recovered 05y6 source")
    print("=" * 100)
    print(f"Script version: {SCRIPT_VERSION}")
    print("05y6 execution: NO")
    print("Scientific shards/results read: NO")
    print("Source restoration occurs only after exact compiled-code verification.")
    print()

    if not PYC.is_file():
        raise FileNotFoundError(f"Frozen 05y6 PYC missing: {PYC}")
    if sha256_file(PYC).lower() != EXPECTED_PYC_SHA256:
        raise RuntimeError("Frozen 05y6 PYC SHA256 mismatch.")
    if not TELEGRAM_CANDIDATE.is_file():
        raise FileNotFoundError(f"Recovered source candidate missing: {TELEGRAM_CANDIDATE}")

    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"05y8a output directory already exists and is non-empty: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    pyc_raw = PYC.read_bytes()
    if len(pyc_raw) < 16:
        raise RuntimeError("PYC too short.")

    flags = struct.unpack("<I", pyc_raw[4:8])[0]
    if flags & 0x01:
        raise RuntimeError("Expected timestamp-based PYC, but hash-based PYC was found.")

    header_source_mtime = struct.unpack("<I", pyc_raw[8:12])[0]
    header_source_size = struct.unpack("<I", pyc_raw[12:16])[0]

    with PYC.open("rb") as f:
        f.read(16)
        frozen_code = marshal.load(f)

    if not isinstance(frozen_code, types.CodeType):
        raise RuntimeError("Frozen PYC payload is not a code object.")

    embedded_filename = frozen_code.co_filename

    candidate_bytes = TELEGRAM_CANDIDATE.read_bytes()
    candidate_stat = TELEGRAM_CANDIDATE.stat()
    candidate_size = len(candidate_bytes)
    candidate_mtime_int = int(candidate_stat.st_mtime)

    size_matches_header = candidate_size == header_source_size
    mtime_matches_header = candidate_mtime_int == header_source_mtime

    compile_attempts = []
    exact_match_optimize = None
    compiled_match_marshaled = False
    compiled_match_inventory = False
    frozen_marshaled_sha256 = sha256_bytes(marshal.dumps(frozen_code))

    for optimize in (0, 1, 2):
        compiled = compile(
            candidate_bytes,
            embedded_filename,
            "exec",
            dont_inherit=True,
            optimize=optimize,
        )
        compiled_marshaled = marshal.dumps(compiled)
        marshaled_equal = compiled_marshaled == marshal.dumps(frozen_code)
        inventory_equal = compare_inventory(compiled, frozen_code)

        compile_attempts.append({
            "optimize": optimize,
            "marshal_payload_sha256": sha256_bytes(compiled_marshaled),
            "exact_marshaled_code_object_match": marshaled_equal,
            "recursive_code_inventory_match": inventory_equal,
        })

        if marshaled_equal and exact_match_optimize is None:
            exact_match_optimize = optimize
            compiled_match_marshaled = True
            compiled_match_inventory = inventory_equal

    verified = (
        size_matches_header
        and mtime_matches_header
        and compiled_match_marshaled
        and compiled_match_inventory
        and exact_match_optimize == 0
    )

    verification = {
        "status": "EXACT_RECOVERED_SOURCE_MATCH" if verified else "RECOVERED_SOURCE_NOT_VERIFIED",
        "script_version": SCRIPT_VERSION,
        "candidate": {
            "path": str(TELEGRAM_CANDIDATE),
            "sha256": sha256_bytes(candidate_bytes),
            "size_bytes": candidate_size,
            "mtime_unix": candidate_mtime_int,
            "mtime_utc": datetime.fromtimestamp(
                candidate_stat.st_mtime, tz=timezone.utc
            ).isoformat(),
            "creation_time_note": (
                "Python stdlib does not provide portable Windows creation-time provenance here; "
                "the pre-archive PowerShell search separately recorded candidate CreationTime."
            ),
        },
        "frozen_pyc": {
            "path": str(PYC.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(PYC),
            "size_bytes": PYC.stat().st_size,
            "header_source_mtime_unix": header_source_mtime,
            "header_source_mtime_utc": datetime.fromtimestamp(
                header_source_mtime, tz=timezone.utc
            ).isoformat(),
            "header_source_size_bytes": header_source_size,
            "embedded_co_filename": embedded_filename,
            "marshal_payload_sha256": frozen_marshaled_sha256,
        },
        "checks": {
            "candidate_size_matches_pyc_header": size_matches_header,
            "candidate_mtime_matches_pyc_header": mtime_matches_header,
            "exact_marshaled_compiled_code_match": compiled_match_marshaled,
            "recursive_code_inventory_match": compiled_match_inventory,
            "matching_optimize_level": exact_match_optimize,
            "verified_for_restoration": verified,
        },
        "compile_attempts": compile_attempts,
        "interpretation": (
            "Verification requires matching PYC-header source size and mtime plus an exact marshaled "
            "CPython code-object match when the candidate is compiled under the embedded source filename. "
            "Because the original PYC is timestamp-based rather than source-hash-based, this establishes "
            "exact compiled-code equivalence and matching header provenance; it does not mathematically prove "
            "that comments/whitespace bytes were identical to a now-missing original source file."
        ),
    }

    RESULT.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")

    if not verified:
        raise RuntimeError(
            "Recovered candidate did not satisfy all exact verification checks. "
            f"See {RESULT}."
        )

    # Archive candidate bytes exactly as found.
    ARCHIVED_SOURCE.write_bytes(candidate_bytes)
    if sha256_file(ARCHIVED_SOURCE) != sha256_bytes(candidate_bytes):
        raise RuntimeError("Archived recovered-source SHA256 mismatch.")

    # Restore only because the expected source path is currently absent.
    if RESTORED_SOURCE.exists():
        raise RuntimeError(
            f"Expected source path unexpectedly already exists: {RESTORED_SOURCE}. "
            "Refusing to overwrite."
        )

    shutil.copy2(TELEGRAM_CANDIDATE, RESTORED_SOURCE)

    restored_sha = sha256_file(RESTORED_SOURCE)
    candidate_sha = sha256_bytes(candidate_bytes)
    if restored_sha != candidate_sha:
        raise RuntimeError("Restored source SHA256 differs from verified candidate.")

    README.write_text(
        "Paper 5 05y8a - recovered 05y6 source verification and restoration\n"
        "===================================================================\n\n"
        "A source file discovered outside the active project was checked against the pre-existing frozen 05y6 .pyc.\n"
        "Restoration was permitted only after all of the following held:\n"
        "  1. frozen .pyc SHA256 matched the previously locked value;\n"
        "  2. candidate source byte size matched the timestamp-based .pyc header;\n"
        "  3. candidate source mtime matched the timestamp-based .pyc header;\n"
        "  4. compiling the candidate with CPython 3.10 under the embedded source filename at optimize=0\n"
        "     produced an exact marshaled code-object match to the frozen .pyc;\n"
        "  5. recursive code-object metadata/bytecode inventory also matched.\n\n"
        "The recovered source was archived byte-for-byte under method_contract/05y8a/recovered_source and copied\n"
        "to its expected scripts/ path without executing it. Scientific shards and results were not read.\n\n"
        "Caveat: the historical .pyc used timestamp/size invalidation rather than a cryptographic source hash.\n"
        "Accordingly, this proves exact compiled-code equivalence plus matching header provenance, not cryptographic\n"
        "identity of comments/whitespace with a missing original file.\n",
        encoding="utf-8",
    )

    self_path = Path(__file__).resolve()
    lock = {
        "status": "05Y8A_RECOVERED_05Y6_SOURCE_VERIFIED_AND_RESTORED",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_result": {
            "path": str(RESULT.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(RESULT),
            "size_bytes": RESULT.stat().st_size,
        },
        "verified_candidate": {
            "original_external_path": str(TELEGRAM_CANDIDATE),
            "sha256": candidate_sha,
            "size_bytes": candidate_size,
            "mtime_unix": candidate_mtime_int,
        },
        "archived_recovered_source": {
            "path": str(ARCHIVED_SOURCE.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(ARCHIVED_SOURCE),
            "size_bytes": ARCHIVED_SOURCE.stat().st_size,
        },
        "restored_source": {
            "path": str(RESTORED_SOURCE.relative_to(ROOT)).replace("\\", "/"),
            "sha256": restored_sha,
            "size_bytes": RESTORED_SOURCE.stat().st_size,
        },
        "frozen_pyc": {
            "path": str(PYC.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(PYC),
            "size_bytes": PYC.stat().st_size,
        },
        "verification_checks": verification["checks"],
        "05y6_executed": False,
        "scientific_shards_read": False,
        "scientific_results_read": False,
        "source_overwrite_performed": False,
        "archive_script": {
            "path": str(self_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(self_path),
            "size_bytes": self_path.stat().st_size,
        },
    }
    LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    print("05y8a recovered source verification: PASS")
    print(f"Candidate SHA256:              {candidate_sha}")
    print(f"PYC-header size match:         {size_matches_header}")
    print(f"PYC-header mtime match:        {mtime_matches_header}")
    print(f"Exact marshaled code match:    {compiled_match_marshaled}")
    print(f"Recursive code inventory match:{compiled_match_inventory}")
    print(f"Matching optimize level:       {exact_match_optimize}")
    print(f"Archived source:               {ARCHIVED_SOURCE.relative_to(ROOT)}")
    print(f"Restored source:               {RESTORED_SOURCE.relative_to(ROOT)}")
    print("05y6 executed:                  NO")
    print(f"Lock:                          {LOCK.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
