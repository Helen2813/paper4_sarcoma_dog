#!/usr/bin/env python3
from __future__ import annotations

import dis
import hashlib
import json
import marshal
import shutil
import struct
import types
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_VERSION = "05y8c-accept-and-restore-recovered-05y6-source-v1-no-cli"

EXPECTED_PYC_SHA256 = "bad3ee4833918b578c76daa013ee8e92de72f945c0f2b976db7dc75873ce39e8"
EXPECTED_CANDIDATE_SHA256 = "9a790077a7e7369839573924db1bbc95693c8210a4936f65a6d249c7434850be"
EXPECTED_SOURCE_SIZE = 24971
EXPECTED_SOURCE_MTIME_UNIX = 1789519956
SOURCE_NAME = "05y6_aggregate_prospective_q10_validation.py"

ROOT = Path(__file__).resolve().parents[1]
PYC = ROOT / "scripts" / "__pycache__" / "05y6_aggregate_prospective_q10_validation.cpython-310.pyc"
CANDIDATE = Path.home() / "Downloads" / "Telegram Desktop" / SOURCE_NAME
RESTORED_SOURCE = ROOT / "scripts" / SOURCE_NAME

Y8B_REPORT = ROOT / "method_contract" / "05y8b" / "05y6_recovered_source_code_object_diff.json"
Y8B_LOCK = ROOT / "method_contract" / "05y8b" / "05y8b_code_object_diff_lock.json"

OUT = ROOT / "method_contract" / "05y8c"
ARCHIVE_DIR = OUT / "recovered_source"
ARCHIVED_SOURCE = ARCHIVE_DIR / SOURCE_NAME
RESULT = OUT / "05y6_recovered_source_acceptance.json"
README = OUT / "05y8c_README.txt"
LOCK = OUT / "05y8c_recovered_source_lock.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def norm_const(x):
    if isinstance(x, types.CodeType):
        return ("code", x.co_name, x.co_firstlineno)
    if isinstance(x, tuple):
        return ("tuple", tuple(norm_const(v) for v in x))
    if isinstance(x, frozenset):
        return ("frozenset", tuple(sorted(repr(v) for v in x)))
    if isinstance(x, bytes):
        return ("bytes", x.hex())
    return (type(x).__name__, repr(x))


def resolved_instructions(code: types.CodeType):
    rows = []
    for ins in dis.get_instructions(code):
        if isinstance(ins.argval, types.CodeType):
            argval = ("code", ins.argval.co_name, ins.argval.co_firstlineno)
        else:
            argval = (type(ins.argval).__name__, repr(ins.argval))
        rows.append((ins.offset, ins.opname, ins.arg, argval))
    return rows


def compare_code_exact_structure(a: types.CodeType, b: types.CodeType, where="<module>") -> list[str]:
    diffs = []

    scalar_fields = [
        "co_argcount",
        "co_posonlyargcount",
        "co_kwonlyargcount",
        "co_nlocals",
        "co_stacksize",
        "co_flags",
        "co_code",
        "co_names",
        "co_varnames",
        "co_filename",
        "co_name",
        "co_firstlineno",
        "co_lnotab",
        "co_freevars",
        "co_cellvars",
    ]
    for field in scalar_fields:
        if getattr(a, field) != getattr(b, field):
            diffs.append(f"{where}: {field} differs")

    if len(a.co_consts) != len(b.co_consts):
        diffs.append(f"{where}: co_consts length differs")
        return diffs

    for i, (ca, cb) in enumerate(zip(a.co_consts, b.co_consts)):
        if isinstance(ca, types.CodeType) or isinstance(cb, types.CodeType):
            if not (isinstance(ca, types.CodeType) and isinstance(cb, types.CodeType)):
                diffs.append(f"{where}: const#{i} code/non-code type differs")
            else:
                diffs.extend(compare_code_exact_structure(ca, cb, f"{where}.{ca.co_name}[const#{i}]"))
        elif norm_const(ca) != norm_const(cb):
            diffs.append(f"{where}: const#{i} differs")

    if resolved_instructions(a) != resolved_instructions(b):
        diffs.append(f"{where}: resolved instruction stream differs")

    return diffs


def main() -> None:
    print("=" * 100)
    print("Paper 5 - accept and restore recovered 05y6 source after structural verification")
    print("=" * 100)
    print(f"Script version: {SCRIPT_VERSION}")
    print("05y6 execution: NO")
    print("Scientific shards/results read: NO")
    print()

    for path in [PYC, CANDIDATE, Y8B_REPORT, Y8B_LOCK]:
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")

    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"05y8c output directory already exists and is non-empty: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    pyc_sha = sha256_file(PYC)
    candidate_sha = sha256_file(CANDIDATE)

    if pyc_sha.lower() != EXPECTED_PYC_SHA256:
        raise RuntimeError(f"Frozen PYC SHA256 mismatch: {pyc_sha}")
    if candidate_sha.lower() != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError(f"Recovered candidate SHA256 mismatch: {candidate_sha}")

    raw = PYC.read_bytes()
    if len(raw) < 16:
        raise RuntimeError("PYC too short.")
    flags = struct.unpack("<I", raw[4:8])[0]
    if flags & 1:
        raise RuntimeError("Expected timestamp-based PYC.")

    header_mtime = struct.unpack("<I", raw[8:12])[0]
    header_size = struct.unpack("<I", raw[12:16])[0]

    if header_size != EXPECTED_SOURCE_SIZE:
        raise RuntimeError(f"Unexpected PYC source size: {header_size}")
    if header_mtime != EXPECTED_SOURCE_MTIME_UNIX:
        raise RuntimeError(f"Unexpected PYC source mtime: {header_mtime}")

    if CANDIDATE.stat().st_size != header_size:
        raise RuntimeError("Candidate size does not match PYC header.")
    if int(CANDIDATE.stat().st_mtime) != header_mtime:
        raise RuntimeError("Candidate mtime does not match PYC header.")

    y8b = json.loads(Y8B_REPORT.read_text(encoding="utf-8"))
    summary = y8b.get("summary", {})
    required_y8b = {
        "diff_entry_count": 0,
        "all_raw_opcode_bytes_equal_recursive": True,
        "no_noncode_constant_differences_recursive": True,
        "no_resolved_instruction_differences_recursive": True,
    }
    for key, expected in required_y8b.items():
        if summary.get(key) != expected:
            raise RuntimeError(f"05y8b verification failed for {key}: {summary.get(key)!r}")

    with PYC.open("rb") as f:
        f.read(16)
        frozen_code = marshal.load(f)
    if not isinstance(frozen_code, types.CodeType):
        raise RuntimeError("Frozen PYC payload is not a code object.")

    candidate_bytes = CANDIDATE.read_bytes()
    candidate_code = compile(
        candidate_bytes,
        frozen_code.co_filename,
        "exec",
        dont_inherit=True,
        optimize=0,
    )

    diffs = compare_code_exact_structure(frozen_code, candidate_code)
    if diffs:
        raise RuntimeError(
            "Independent structural re-check failed:\n" + "\n".join(diffs[:50])
        )

    # Archive exact recovered bytes.
    ARCHIVED_SOURCE.write_bytes(candidate_bytes)
    if sha256_file(ARCHIVED_SOURCE) != candidate_sha:
        raise RuntimeError("Archived source SHA256 mismatch.")

    # Restore only if the expected active-project source is absent.
    if RESTORED_SOURCE.exists():
        raise RuntimeError(
            f"Expected source path already exists; refusing to overwrite: {RESTORED_SOURCE}"
        )
    shutil.copy2(CANDIDATE, RESTORED_SOURCE)

    if sha256_file(RESTORED_SOURCE) != candidate_sha:
        raise RuntimeError("Restored source SHA256 mismatch.")
    if RESTORED_SOURCE.stat().st_size != header_size:
        raise RuntimeError("Restored source size mismatch.")

    result = {
        "status": "RECOVERED_05Y6_SOURCE_ACCEPTED_AS_EXECUTABLE_EQUIVALENT_AND_RESTORED",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence": {
            "frozen_pyc_sha256": pyc_sha,
            "candidate_source_sha256": candidate_sha,
            "candidate_size_matches_pyc_header": True,
            "candidate_mtime_matches_pyc_header": True,
            "05y8b_diff_entry_count": 0,
            "raw_opcode_bytes_equal_recursive": True,
            "noncode_constants_equal_recursive": True,
            "resolved_instruction_stream_equal_recursive": True,
            "independent_structural_recheck_diff_count": len(diffs),
        },
        "source_identity_statement": (
            "The recovered source is accepted as executable-semantics equivalent to the frozen 05y6 PYC and "
            "matches the timestamp-based PYC header in source size and mtime. Because the historical PYC does not "
            "contain a cryptographic source hash, byte-identical identity of comments/whitespace with a hypothetical "
            "missing original cannot be mathematically proven. The recovered candidate itself is archived and restored "
            "byte-for-byte under its recorded SHA256."
        ),
        "archived_source": {
            "path": str(ARCHIVED_SOURCE.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(ARCHIVED_SOURCE),
            "size_bytes": ARCHIVED_SOURCE.stat().st_size,
        },
        "restored_source": {
            "path": str(RESTORED_SOURCE.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(RESTORED_SOURCE),
            "size_bytes": RESTORED_SOURCE.stat().st_size,
            "last_write_time_utc": datetime.fromtimestamp(
                RESTORED_SOURCE.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
        },
        "frozen_pyc": {
            "path": str(PYC.relative_to(ROOT)).replace("\\", "/"),
            "sha256": pyc_sha,
            "size_bytes": PYC.stat().st_size,
            "header_source_size_bytes": header_size,
            "header_source_mtime_unix": header_mtime,
            "embedded_source_filename": frozen_code.co_filename,
        },
        "05y6_executed": False,
        "scientific_shards_read": False,
        "scientific_results_read": False,
    }
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    README.write_text(
        "Paper 5 05y8c - recovered 05y6 source acceptance/restoration\n"
        "============================================================\n\n"
        "The recovered Telegram/Recycle-Bin source candidate had the same SHA256 in both discovered copies.\n"
        "Its size and mtime match the timestamp-based frozen .pyc header exactly.\n"
        "05y8b found zero recursive code-object differences: raw opcode bytes, non-code constants,\n"
        "names/variables, line metadata, and resolved instruction streams all matched.\n\n"
        "The candidate is therefore accepted as executable-semantics equivalent to the frozen 05y6 bytecode,\n"
        "archived byte-for-byte, and restored to scripts/05y6_aggregate_prospective_q10_validation.py.\n\n"
        "Caveat: timestamp-based .pyc files do not contain a cryptographic source hash, so exact historical\n"
        "identity of comments/whitespace cannot be proven. This does not affect executable semantics.\n\n"
        "05y6 was NOT executed here. Scientific shards/results were NOT read.\n",
        encoding="utf-8",
    )

    self_path = Path(__file__).resolve()
    lock = {
        "status": "05Y8C_RECOVERED_05Y6_SOURCE_ACCEPTED_AND_RESTORED",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": {
            "path": str(RESULT.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(RESULT),
            "size_bytes": RESULT.stat().st_size,
        },
        "readme": {
            "path": str(README.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(README),
            "size_bytes": README.stat().st_size,
        },
        "archived_source": result["archived_source"],
        "restored_source": result["restored_source"],
        "frozen_pyc": result["frozen_pyc"],
        "05y8b_report": {
            "path": str(Y8B_REPORT.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(Y8B_REPORT),
            "size_bytes": Y8B_REPORT.stat().st_size,
        },
        "archive_script": {
            "path": str(self_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(self_path),
            "size_bytes": self_path.stat().st_size,
        },
        "05y6_executed": False,
        "scientific_shards_read": False,
        "scientific_results_read": False,
        "source_claim": "EXECUTABLE_EQUIVALENCE_PLUS_MATCHING_TIMESTAMP_BASED_HEADER_PROVENANCE",
    }
    LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    print("05y8c recovered source acceptance/restoration: PASS")
    print(f"Recovered source SHA256:       {candidate_sha}")
    print(f"PYC header size/mtime match:   True / True")
    print(f"05y8b structural diffs:        0")
    print(f"Independent structural diffs:  0")
    print(f"Archived source:               {ARCHIVED_SOURCE.relative_to(ROOT)}")
    print(f"Restored source:               {RESTORED_SOURCE.relative_to(ROOT)}")
    print("05y6 executed:                  NO")
    print(f"Lock:                          {LOCK.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
