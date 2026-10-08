#!/usr/bin/env python3
from __future__ import annotations

import dis
import hashlib
import json
import marshal
import struct
import types
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_VERSION = "05y8b-compare-recovered-05y6-source-to-frozen-bytecode-v1-no-cli"
EXPECTED_PYC_SHA256 = "bad3ee4833918b578c76daa013ee8e92de72f945c0f2b976db7dc75873ce39e8"
SOURCE_NAME = "05y6_aggregate_prospective_q10_validation.py"

ROOT = Path(__file__).resolve().parents[1]
PYC = ROOT / "scripts" / "__pycache__" / "05y6_aggregate_prospective_q10_validation.cpython-310.pyc"
CANDIDATE = Path.home() / "Downloads" / "Telegram Desktop" / SOURCE_NAME

OUT = ROOT / "method_contract" / "05y8b"
REPORT = OUT / "05y6_recovered_source_code_object_diff.json"
TEXT = OUT / "05y6_recovered_source_code_object_diff.txt"
LOCK = OUT / "05y8b_code_object_diff_lock.json"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def jsonable_const(x):
    if isinstance(x, types.CodeType):
        return {
            "type": "code",
            "co_name": x.co_name,
            "co_firstlineno": x.co_firstlineno,
        }
    if isinstance(x, bytes):
        return {"type": "bytes", "hex": x.hex()}
    if isinstance(x, tuple):
        return {"type": "tuple", "items": [jsonable_const(v) for v in x]}
    if isinstance(x, frozenset):
        items = [jsonable_const(v) for v in x]
        return {"type": "frozenset", "items_repr_sorted": sorted(repr(v) for v in items)}
    return {
        "type": type(x).__name__,
        "repr": repr(x),
    }


def const_equal(a, b) -> bool:
    if isinstance(a, types.CodeType) or isinstance(b, types.CodeType):
        return isinstance(a, types.CodeType) and isinstance(b, types.CodeType)
    if type(a) is not type(b):
        return False
    if isinstance(a, tuple):
        return len(a) == len(b) and all(const_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, frozenset):
        return a == b
    try:
        return a == b
    except Exception:
        return repr(a) == repr(b)


def norm_argval(v):
    if isinstance(v, types.CodeType):
        return {
            "kind": "code",
            "name": v.co_name,
            "firstlineno": v.co_firstlineno,
        }
    if isinstance(v, bytes):
        return {"kind": "bytes", "hex": v.hex()}
    if isinstance(v, (str, int, float, complex, bool, type(None))):
        return {"kind": type(v).__name__, "repr": repr(v)}
    if isinstance(v, tuple):
        return {"kind": "tuple", "repr": repr(v)}
    return {"kind": type(v).__name__, "repr": repr(v)}


def instruction_signature(code: types.CodeType):
    rows = []
    for ins in dis.get_instructions(code):
        rows.append({
            "offset": ins.offset,
            "opname": ins.opname,
            "arg": ins.arg,
            "argval": norm_argval(ins.argval),
        })
    return rows


def field_value(code: types.CodeType, field: str):
    v = getattr(code, field)
    if isinstance(v, bytes):
        return {
            "len": len(v),
            "sha256": sha256_bytes(v),
            "hex": v.hex() if len(v) <= 256 else None,
        }
    if isinstance(v, tuple):
        return list(v)
    return v


def compare_code(frozen: types.CodeType, candidate: types.CodeType, qualname="<module>"):
    diffs = []
    child_pairs = []

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
        a = getattr(frozen, field)
        b = getattr(candidate, field)
        if a != b:
            diffs.append({
                "qualname": qualname,
                "category": "code_field",
                "field": field,
                "frozen": field_value(frozen, field),
                "candidate": field_value(candidate, field),
            })

    if len(frozen.co_consts) != len(candidate.co_consts):
        diffs.append({
            "qualname": qualname,
            "category": "const_count",
            "frozen": len(frozen.co_consts),
            "candidate": len(candidate.co_consts),
        })

    for idx, (a, b) in enumerate(zip(frozen.co_consts, candidate.co_consts)):
        if isinstance(a, types.CodeType) or isinstance(b, types.CodeType):
            if not (isinstance(a, types.CodeType) and isinstance(b, types.CodeType)):
                diffs.append({
                    "qualname": qualname,
                    "category": "const_type",
                    "const_index": idx,
                    "frozen": jsonable_const(a),
                    "candidate": jsonable_const(b),
                })
            else:
                child_pairs.append((a, b, f"{qualname}.{a.co_name}[const#{idx}]"))
        elif not const_equal(a, b):
            diffs.append({
                "qualname": qualname,
                "category": "const_value",
                "const_index": idx,
                "frozen": jsonable_const(a),
                "candidate": jsonable_const(b),
            })

    sig_a = instruction_signature(frozen)
    sig_b = instruction_signature(candidate)
    if sig_a != sig_b:
        # Keep only differing instruction rows, aligned by position.
        n = max(len(sig_a), len(sig_b))
        inst_diffs = []
        for i in range(n):
            ra = sig_a[i] if i < len(sig_a) else None
            rb = sig_b[i] if i < len(sig_b) else None
            if ra != rb:
                inst_diffs.append({
                    "instruction_index": i,
                    "frozen": ra,
                    "candidate": rb,
                })
        diffs.append({
            "qualname": qualname,
            "category": "resolved_instruction_signature",
            "difference_count": len(inst_diffs),
            "differences": inst_diffs[:200],
            "truncated": len(inst_diffs) > 200,
        })

    for a, b, child_name in child_pairs:
        diffs.extend(compare_code(a, b, child_name))

    return diffs


def main():
    print("=" * 100)
    print("Paper 5 - compare recovered 05y6 source to frozen bytecode")
    print("=" * 100)
    print(f"Script version: {SCRIPT_VERSION}")
    print("05y6 execution: NO")
    print("Scientific shards/results read: NO")
    print()

    if not PYC.is_file():
        raise FileNotFoundError(PYC)
    if sha256_file(PYC).lower() != EXPECTED_PYC_SHA256:
        raise RuntimeError("Frozen 05y6 PYC SHA256 mismatch.")
    if not CANDIDATE.is_file():
        raise FileNotFoundError(CANDIDATE)

    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"05y8b output directory already exists and is non-empty: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    raw = PYC.read_bytes()
    flags = struct.unpack("<I", raw[4:8])[0]
    if flags & 1:
        raise RuntimeError("Unexpected hash-based PYC.")

    header_mtime = struct.unpack("<I", raw[8:12])[0]
    header_size = struct.unpack("<I", raw[12:16])[0]

    with PYC.open("rb") as f:
        f.read(16)
        frozen = marshal.load(f)
    if not isinstance(frozen, types.CodeType):
        raise RuntimeError("Frozen payload is not a code object.")

    candidate_bytes = CANDIDATE.read_bytes()
    candidate = compile(
        candidate_bytes,
        frozen.co_filename,
        "exec",
        dont_inherit=True,
        optimize=0,
    )

    diffs = compare_code(frozen, candidate)

    category_counts = {}
    for d in diffs:
        category_counts[d["category"]] = category_counts.get(d["category"], 0) + 1

    only_debug_metadata = bool(diffs) and all(
        d["category"] == "code_field" and d.get("field") in {"co_lnotab", "co_firstlineno"}
        for d in diffs
    )

    no_resolved_instruction_differences = not any(
        d["category"] == "resolved_instruction_signature" for d in diffs
    )
    no_constant_differences = not any(
        d["category"] in {"const_value", "const_type", "const_count"} for d in diffs
    )
    all_raw_opcode_bytes_equal = not any(
        d["category"] == "code_field" and d.get("field") == "co_code" for d in diffs
    )

    report = {
        "status": "RECOVERED_SOURCE_VS_FROZEN_PYC_STRUCTURAL_DIFF_COMPLETE",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_pyc": {
            "path": str(PYC.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(PYC),
            "header_source_size_bytes": header_size,
            "header_source_mtime_unix": header_mtime,
            "embedded_co_filename": frozen.co_filename,
        },
        "candidate": {
            "path": str(CANDIDATE),
            "sha256": sha256_file(CANDIDATE),
            "size_bytes": CANDIDATE.stat().st_size,
            "mtime_unix": int(CANDIDATE.stat().st_mtime),
        },
        "summary": {
            "diff_entry_count": len(diffs),
            "category_counts": category_counts,
            "all_raw_opcode_bytes_equal_recursive": all_raw_opcode_bytes_equal,
            "no_noncode_constant_differences_recursive": no_constant_differences,
            "no_resolved_instruction_differences_recursive": no_resolved_instruction_differences,
            "differences_only_debug_line_metadata": only_debug_metadata,
        },
        "differences": diffs,
        "interpretation_rule": (
            "If non-code constants or resolved instruction signatures differ, do not treat the candidate as "
            "exact executable semantics. If raw opcode bytes, constants, names/variables, and resolved instruction "
            "signatures all match and differences are limited to line/debug metadata, the candidate is executable-"
            "semantics-equivalent but still not cryptographically proven byte-identical source."
        ),
    }

    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "05Y8B RECOVERED SOURCE / FROZEN PYC DIFF",
        "=" * 80,
        f"Frozen PYC SHA256: {sha256_file(PYC)}",
        f"Candidate SHA256: {sha256_file(CANDIDATE)}",
        f"Candidate size / header size: {CANDIDATE.stat().st_size} / {header_size}",
        f"Candidate mtime / header mtime: {int(CANDIDATE.stat().st_mtime)} / {header_mtime}",
        "",
        f"Diff entries: {len(diffs)}",
        f"Category counts: {category_counts}",
        f"All raw opcode bytes equal recursively: {all_raw_opcode_bytes_equal}",
        f"No non-code constant differences recursively: {no_constant_differences}",
        f"No resolved instruction differences recursively: {no_resolved_instruction_differences}",
        f"Differences only debug line metadata: {only_debug_metadata}",
        "",
        "Detailed differences:",
    ]
    for d in diffs:
        lines.append(json.dumps(d, ensure_ascii=False))
    TEXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    self_path = Path(__file__).resolve()
    lock = {
        "status": "05Y8B_RECOVERED_SOURCE_CODE_OBJECT_DIFF_FROZEN",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "report": {
            "path": str(REPORT.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(REPORT),
            "size_bytes": REPORT.stat().st_size,
        },
        "text_report": {
            "path": str(TEXT.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(TEXT),
            "size_bytes": TEXT.stat().st_size,
        },
        "archive_script": {
            "path": str(self_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(self_path),
            "size_bytes": self_path.stat().st_size,
        },
        "05y6_executed": False,
        "scientific_shards_read": False,
        "scientific_results_read": False,
    }
    LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    print("05y8b structural comparison: COMPLETE")
    print(f"Diff entries:                         {len(diffs)}")
    print(f"Category counts:                      {category_counts}")
    print(f"Raw opcode bytes equal recursively:   {all_raw_opcode_bytes_equal}")
    print(f"No non-code constant differences:     {no_constant_differences}")
    print(f"No resolved instruction differences:  {no_resolved_instruction_differences}")
    print(f"Only debug line metadata differs:      {only_debug_metadata}")
    print(f"Report: {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
