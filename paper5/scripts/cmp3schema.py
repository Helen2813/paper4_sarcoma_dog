#!/usr/bin/env python3
r"""Paper 5: collect evidence for the authorized v5 NPZ-schema failure.

This is a diagnostic, NOT a runner, patch, authorizer or result evaluator.
Only the FIRST failed dataset is opened. Its bytes are hashed, then NPY
headers are inspected. No array payload is interpreted, returned, or copied
into the output. No NumPy/PyTorch/MOVE/R or project source is imported.
The sole write is a NEW timestamped ZIP under paper5, containing source and
contract snapshots, header metadata and a result-directory metadata inventory.
Existing sources, data, preflight, authorization and results remain untouched.

Run from the project after saving this file to paper5/scripts/cmp3schema.py:
    python -I -B .\paper5\scripts\cmp3schema.py
"""
from __future__ import annotations
import ast
import csv
import hashlib
import io
import json
import os
import re
import stat
import struct
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PureWindowsPath
from typing import Any, BinaryIO

VERSION = "05cmp3-v5-first-failure-schema-evidence-v1-no-fit-no-amendment"
INPUT_SPECS = {'runner': ('paper5/scripts/05cmp3_run_secondary_comparator_confirmatory_benchmark_v5.py',
            '802c131dfda75effb4e60dd38ebf05512fa5da6c1b3eec97c0c499b3c4e869f5'),
 'authorizer': ('paper5/scripts/05cmp3a_freeze_secondary_comparator_runner_v5_source_authorization_v1.py',
                '526bf30c101dea38cb26d07c9acd625892febebcd697fb466be0c4c1fe653bfa'),
 'authorization': ('paper5/method_contract/05cmp3a/secondary_comparator_runner_source_authorization_lock.json',
                   '5dbb075b3467fa865cb2b0570a79c2419e952a2c8c001eb04287d8ade5fba319'),
 'preflight': ('paper5/method_contract/05cmp3_preflight/secondary_comparator_runner_synthetic_preflight_v5.json',
               '7ec4e687151a4af67b718a1ee13e521109a0071be91de2e751e9d0bb02b96b94'),
 'linear_helper': ('paper5/scripts/05c1_fit_linear_simulation_baselines.py',
                   'c2c91b4ad274f78ccc369e6fca765102285cf3d9c8191f3c33f4a450b7426367'),
 'cmp2_contract': ('paper5/method_contract/05cmp2/secondary_comparator_confirmatory_benchmark_contract.json',
                   '8e0a4d95a1f554af8e9c5188ca683d32c336c6bd5b797b3a91426adbe4ee9fc2'),
 'cmp2_lock': ('paper5/method_contract/05cmp2/secondary_comparator_confirmatory_benchmark_contract_lock.json',
               'a55d3e3498d9cc6a4acaaab64e5098007b093071b60999c82cf8e124a0301b5b'),
 'cmp2_registry': ('paper5/method_contract/05cmp2/secondary_comparator_confirmatory_dataset_registry.csv',
                   '4b72b94c4df49dbbbd1a90d035855e628a4fa50eede0451dfe796227bda53314'),
 'cmp1_contract': ('paper5/method_contract/05cmp1/secondary_comparator_implementation_contract.json',
                   '418259cedd36cbd1913a4ff1ae9874a3decfddbb7a75e9b3bca2e9623f0aca95'),
 'cmp1_lock': ('paper5/method_contract/05cmp1/secondary_comparator_implementation_contract_lock.json',
               '37e0db502a88f7beab52f1457cb0c5367091314faa6b86c7796810d883f31648'),
 'cmp1_impl': ('paper5/method_contract/05cmp1/secondary_comparator_implementation_registry.csv',
               '85dbc68289bae71b9cf4f8a01af34b09d50750fde97f0b57ef3b6c57ebf299b0'),
 'cmp1_adapter': ('paper5/method_contract/05cmp1/secondary_comparator_scientific_adapter_registry.csv',
                  'f30c73d19627d357073786b75f7b8141a757becf9d2413491518948cc099ea48'),
 'cmp1_seed': ('paper5/method_contract/05cmp1/secondary_comparator_seed_registry.csv',
               '2c70c516ad07496a71fe513139b357b92f46fd746bea5258f1fb237028c4e87c'),
 'cmp1_metric': ('paper5/method_contract/05cmp1/secondary_comparator_metric_eligibility_registry.csv',
                 '8802d8d4bba20cfd4acc3263b12f0a119bcef4e4107164e6f944d0a01d9bef46'),
 'manifest05b': ('paper5/method_contract/05b_v2/architecture_screening_data_manifest.csv',
                 'f3980e88aff408b65cbfce96aca50bad4fc08ecece2bc42bbc04f365c35d9c87')}
GENERATOR_REL = "paper5/scripts/05b_generate_small_n_multiomics_simulations_v2.py"
ARCHIVED_GENERATOR_SHA = "b4adb1fcb32b65a998ee70e1b72d52e13490194c7ad8e5ddd802b95bf05a28db"
DATASET_ID = "A01_N62_STANDARD__R11__S1956851480"
RESULT_SCOPES = (
    "paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1",
    "paper5/method_contract/05cmp4",
)
V5_REQUIRED = {"X_rna", "X_methylation", "X_cnv", "Z_shared", "sample_index"}
MAX_SOURCE_BYTES = 32 * 1024 * 1024
MAX_HEADER_BYTES = 64 * 1024
MAX_MEMBERS = 512
MAX_RESULT_ENTRIES = 10000

class EvidenceError(RuntimeError):
    pass

def need(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def project_path(root: Path, relative: str) -> Path:
    normalized = str(relative).replace("\\", "/")
    rel = Path(normalized)
    need(normalized != "" and not rel.is_absolute()
         and not PureWindowsPath(normalized).drive
         and ".." not in rel.parts,
         "Non-project-relative path rejected: " + normalized)
    current = root
    for part in rel.parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        need(not stat.S_ISLNK(info.st_mode) and not (
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)),
            "Symlink/reparse point rejected: " + str(current))
    need(current.resolve().is_relative_to(root.resolve()), "Path escapes project")
    return current

def snapshot(root: Path, relative: str, expected: str | None = None
             ) -> tuple[bytes, dict[str, Any]]:
    path = project_path(root, relative)
    need(path.is_file(), "Required file missing: " + str(path))
    need(path.stat().st_size <= MAX_SOURCE_BYTES, "Unexpectedly large source/metadata file: " + relative)
    with path.open("rb") as handle:
        raw = handle.read(MAX_SOURCE_BYTES + 1)
    need(len(raw) <= MAX_SOURCE_BYTES, "Source/metadata file exceeded size limit")
    observed = digest(raw)
    if expected is not None:
        need(observed == expected,
             "SHA mismatch: " + relative + "\nobserved=" + observed + "\nexpected=" + expected)
    return raw, {"path": relative, "sha256": observed, "size_bytes": len(raw)}

def strict_json(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in items:
            need(k not in out, "Duplicate JSON key in " + label + ": " + k)
            out[k] = v
        return out
    def reject(value: str) -> None:
        raise EvidenceError("Nonfinite JSON constant: " + value)
    obj = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=pairs, parse_constant=reject)
    need(isinstance(obj, dict), "Expected a JSON object: " + label)
    return obj

def as_int(value: str, label: str) -> int:
    try:
        num = Decimal(value.strip())
    except (InvalidOperation, AttributeError) as exc:
        raise EvidenceError("Invalid integer metadata: " + label) from exc
    need(num.is_finite() and num == num.to_integral_value(), "Noninteger metadata: " + label)
    return int(num)

def first_row(raw: bytes) -> tuple[dict[str, str], int]:
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    fields = reader.fieldnames or []
    required = {"comparator_dataset_id", "scenario_id", "replicate_index", "simulation_seed",
                "path", "size_bytes", "sha256", "n_samples"}
    need(required.issubset(fields), "Required registry columns missing")
    need(len(fields) == len(set(fields)), "Duplicate CSV column")
    rows = list(reader)
    need(len(rows) == 360, "Expected exactly 360 registry rows")
    by_scenario: dict[str, list[int]] = defaultdict(list)
    ids: set[str] = set()
    for row in rows:
        need(None not in row and all(v is not None for v in row.values()), "Malformed registry row")
        need(row["comparator_dataset_id"] not in ids, "Duplicate dataset id")
        ids.add(row["comparator_dataset_id"])
        by_scenario[row["scenario_id"]].append(as_int(row["replicate_index"], "replicate_index"))
    need(len(by_scenario) == 36 and all(sorted(v) == list(range(11, 21))
         for v in by_scenario.values()), "Unexpected scenario/replicate scope")
    row = sorted(rows, key=lambda r: (r["scenario_id"], as_int(r["replicate_index"], "replicate"),
                                      as_int(r["simulation_seed"], "seed")))[0]
    need(row["comparator_dataset_id"] == DATASET_ID, "First dataset differs from reported traceback")
    return row, len(rows)

def exact_read(handle: BinaryIO, n: int) -> bytes:
    raw = handle.read(n)
    need(len(raw) == n, "Truncated NPY header")
    return raw

def npy_header(handle: BinaryIO) -> dict[str, Any]:
    """Parse ONLY the NPY header; do not request array-payload bytes.

    ZIP decompression may buffer ahead internally. No payload is interpreted.
    Format definition: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
    """
    need(exact_read(handle, 6) == b"\x93NUMPY", "Bad NPY magic")
    version = tuple(exact_read(handle, 2))
    need(version in {(1, 0), (2, 0), (3, 0)}, "Unsupported NPY format: " + repr(version))
    length_size = 2 if version == (1, 0) else 4
    header_len = struct.unpack("<H" if length_size == 2 else "<I",
                               exact_read(handle, length_size))[0]
    need(0 < header_len <= MAX_HEADER_BYTES, "NPY header exceeds diagnostic limit")
    text = exact_read(handle, header_len).decode("utf-8" if version == (3, 0) else "latin1")
    expression = ast.parse(text.strip(), mode="eval")
    need(isinstance(expression.body, ast.Dict), "NPY header is not a dictionary literal")
    keys = [ast.literal_eval(k) for k in expression.body.keys]
    need(len(keys) == 3 and set(keys) == {"descr", "shape", "fortran_order"},
         "NPY header keys are missing, duplicated or unexpected")
    hdr = ast.literal_eval(expression)
    shape = hdr["shape"]
    need(isinstance(shape, tuple) and all(type(i) is int and i >= 0 for i in shape), "Invalid NPY shape")
    need(type(hdr["fortran_order"]) is bool, "Invalid NPY fortran_order")
    need(isinstance(hdr["descr"], (str, list)), "Invalid NPY dtype descriptor")
    return {"format_version": list(version), "shape": list(shape), "descr": hdr["descr"],
            "fortran_order": hdr["fortran_order"], "header_bytes": 8 + length_size + header_len}

def header_inventory(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with zipfile.ZipFile(path, "r") as archive:
        members = archive.infolist()
        need(0 < len(members) <= MAX_MEMBERS, "Unexpected NPZ member count")
        names = [m.filename for m in members]
        need(len(names) == len(set(names)), "Duplicate NPZ members")
        for member in sorted(members, key=lambda m: m.filename):
            need(not member.is_dir() and re.fullmatch(r"[A-Za-z0-9_.-]+\.npy", member.filename) is not None,
                 "Unexpected NPZ member name: " + member.filename)
            need(not (member.flag_bits & 1), "Encrypted NPZ member")
            with archive.open(member, "r") as handle:
                hdr = npy_header(handle)
            need(hdr["header_bytes"] <= member.file_size, "NPY header larger than archive member")
            output.append({"key": member.filename[:-4], "member": member.filename,
                           "uncompressed_bytes": member.file_size, "crc32": f"{member.CRC:08x}", **hdr})
    return output

def result_inventory(root: Path) -> list[dict[str, Any]]:
    # No result-file open/read/hash calls. Inventory alone is not proof that
    # results were never created or read during any earlier process.
    rows: list[dict[str, Any]] = []
    for rel in RESULT_SCOPES:
        scope = project_path(root, rel)
        if not scope.exists():
            rows.append({"path": rel, "kind": "absent"})
            continue
        need(scope.is_dir(), "Result scope is not a directory: " + rel)
        rows.append({"path": rel, "kind": "directory"})
        def onerror(error: OSError) -> None:
            raise error
        for directory, dirs, files in os.walk(scope, followlinks=False, onerror=onerror):
            for name in sorted(dirs):
                p = Path(directory) / name
                project_path(root, p.relative_to(root).as_posix())
            for name in sorted(files):
                p = Path(directory) / name
                relative = p.relative_to(root).as_posix()
                p = project_path(root, relative)
                info = p.stat()
                rows.append({"path": relative, "kind": "file", "size_bytes": info.st_size,
                             "mtime_ns": info.st_mtime_ns, "content_read": False})
                need(len(rows) <= MAX_RESULT_ENTRIES, "Result inventory exceeds diagnostic limit")
    return sorted(rows, key=lambda r: r["path"])

def build_evidence(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    copies: dict[str, bytes] = {}
    records: dict[str, dict[str, Any]] = {}
    for label, (rel, expected) in INPUT_SPECS.items():
        raw, record = snapshot(root, rel, expected)
        copies[rel] = raw
        records[label] = record
    auth = strict_json(copies[INPUT_SPECS["authorization"][0]], "authorization")
    preflight = strict_json(copies[INPUT_SPECS["preflight"][0]], "preflight")
    need(auth.get("status") == "05CMP3A_SECONDARY_COMPARATOR_RUNNER_SOURCE_AUTHORIZED", "Unexpected authorization status")
    need(preflight.get("status") == "05CMP3_SECONDARY_COMPARATOR_RUNNER_SYNTHETIC_PREAUTH_PASS", "Unexpected preflight status")
    for auth_key, label in (("runner_source", "runner"), ("synthetic_preflight", "preflight"),
                            ("cmp2_contract", "cmp2_contract"), ("linear_helper_source", "linear_helper")):
        need(auth.get(auth_key, {}).get("sha256") == records[label]["sha256"], "Authorization binding mismatch: " + auth_key)
    need(preflight.get("runner_sha256") == records["runner"]["sha256"], "Preflight runner mismatch")
    need(preflight.get("linear_helper_source", {}).get("sha256") == records["linear_helper"]["sha256"], "Preflight helper mismatch")
    print("Pinned v5 / authorization / preflight / helper / upstream: PASS", flush=True)

    warnings: list[str] = []
    generator = project_path(root, GENERATOR_REL)
    if generator.is_file():
        raw, rec = snapshot(root, GENERATOR_REL)
        rec["matches_archived_generator_sha"] = rec["sha256"] == ARCHIVED_GENERATOR_SHA
        copies[GENERATOR_REL] = raw
        records["generator"] = rec
        if not rec["matches_archived_generator_sha"]:
            warnings.append("Generator hash differs from archived description; review its frozen identity before patching.")
    else:
        warnings.append("Generator source is missing at the documented path; not replaced or guessed.")

    # Sources are parsed only. The complete bytes are included for review;
    # never import/exec them or run a generator self-test on the user's project.
    indexes: dict[str, Any] = {}
    for label in ("runner", "linear_helper", "generator"):
        if label not in records:
            continue
        rel = records[label]["path"]
        tree = ast.parse(copies[rel].decode("utf-8-sig"), filename=rel, feature_version=(3, 10))
        indexes[label] = [{"name": n.name, "start_line": n.lineno, "end_line": n.end_lineno}
                          for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]

    before_results = result_inventory(root)
    row, count = first_row(copies[INPUT_SPECS["cmp2_registry"][0]])
    dataset = project_path(root, row["path"])
    simulation_scope = project_path(root, "paper5/simulations")
    need(dataset.is_relative_to(simulation_scope) and dataset.suffix.lower() == ".npz",
         "Frozen row does not point to an NPZ within paper5/simulations")
    need(dataset.is_file(), "First dataset missing: " + str(dataset))
    size = as_int(row["size_bytes"], "size_bytes")
    need(dataset.stat().st_size == size, "First NPZ size mismatch")
    expected_npz_sha = row["sha256"]
    need(re.fullmatch(r"[0-9a-f]{64}", expected_npz_sha) is not None, "Invalid registry NPZ hash")
    need(sha256_file(dataset) == expected_npz_sha, "First NPZ SHA mismatch")
    print("First failed NPZ size/SHA against frozen registry: PASS", flush=True)
    headers = header_inventory(dataset)
    keys = {h["key"] for h in headers}
    missing = sorted(V5_REQUIRED - keys)
    print("Dataset: " + DATASET_ID, flush=True)
    print("NPZ header inventory (names / shapes / dtype only):", flush=True)
    for item in headers:
        print("  " + item["key"] + "  shape=" + repr(item["shape"]) + "  dtype=" + repr(item["descr"]))
    print("Missing keys required by v5: " + repr(missing), flush=True)
    print("Z_biological_shared key present: " + str("Z_biological_shared" in keys), flush=True)
    if not missing:
        warnings.append("The current header inventory does not reproduce the reported missing-key failure.")
    if any(r["kind"] == "file" for r in before_results):
        warnings.append("Result files exist. Their contents were NOT read. Do not delete them or reauthorize blindly.")

    # Recheck all existing inputs BEFORE publishing any evidence ZIP.
    for label, rec in records.items():
        snapshot(root, rec["path"], rec["sha256"])
    need(dataset.stat().st_size == size and sha256_file(dataset) == expected_npz_sha,
         "First NPZ changed during collection")
    need(result_inventory(root) == before_results, "Result directory changed during collection; stop concurrent runner")

    report: dict[str, Any] = {
        "diagnostic_version": VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version, "python_executable": sys.executable,
        "project_root": str(root), "registry_count": count,
        "first_failed_dataset": {"registry_row": row, "size_bytes": size,
                                 "sha256": expected_npz_sha, "header_inventory": headers,
                                 "v5_missing_required_keys": missing,
                                 "Z_biological_shared_present": "Z_biological_shared" in keys},
        "snapshots": records, "source_definition_indexes": indexes,
        "result_directory_metadata": before_results, "warnings": warnings,
        "access": {
            "scientific_npz_files_opened": 1, "scientific_npz_bytes_hashed": True,
            "array_payload_values_decoded_or_analyzed": False, "array_payload_in_output_zip": False,
            "npy_headers_decoded": True,
            "other_simulation_npz_opened": False, "results_content_read": False,
            "TARGET_outcomes_clinical_biology_accessed": False,
            "project_sources_executed_or_imported": False,
            "model_fits_or_metrics_computed": False, "R_CUDA_MOVE_executed": False,
            "existing_project_files_modified": False,
            "authorization_created_modified_or_revoked": False,
            "sole_write": "new evidence ZIP outside results and method_contract",
        },
        "finding_scope": "metadata/schema evidence only; no repair, no scientific PASS, no renewed authorization",
    }
    return copies, report

def publish(root: Path, copies: dict[str, bytes], report: dict[str, Any]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = project_path(root, "paper5/cmp3_schema_review_" + stamp + ".zip")
    with zipfile.ZipFile(out, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in sorted(copies):
            archive.writestr("inputs/" + relative, copies[relative])
        archive.writestr("schema_report.json", json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        archive.writestr("README.txt", (
            "Metadata/source evidence for the FIRST v5 NPZ-loader failure.\n"
            "No NPZ/NPY array payloads and no result-file contents are included.\n"
            "Original files are unchanged. This ZIP is NOT authorization.\n"
            "Input SHA-256 values and access limits are in schema_report.json.\n"
            "Preserve the existing v5 source, preflight, authorization and any results.\n"))
    with zipfile.ZipFile(out, "r") as archive:
        need(archive.testzip() is None, "New evidence ZIP failed its own CRC check")
        for rel, data in copies.items():
            need(archive.read("inputs/" + rel) == data, "New evidence ZIP snapshot mismatch")
    return out

def main() -> int:
    print("Paper 5 - first-failure NPZ schema evidence; NOT a runner or authorizer", flush=True)
    print("Version: " + VERSION, flush=True)
    script = Path(__file__).resolve()
    need(script.parent.name == "scripts" and script.parent.parent.name == "paper5",
         "Save cmp3schema.py in your project's paper5/scripts directory")
    root = script.parents[2]
    copies, report = build_evidence(root)
    # Include the executed diagnostic as evidence, but never execute its snapshots.
    copies[script.relative_to(root).as_posix()] = script.read_bytes()
    out = publish(root, copies, report)
    print("Input recheck: PASS", flush=True)
    print("Existing sources/data/preflight/authorization/results: UNCHANGED", flush=True)
    print("Fits/metrics: NOT RUN; array values: NOT DECODED", flush=True)
    print("Evidence ZIP: " + str(out), flush=True)
    print("Evidence ZIP SHA-256: " + sha256_file(out), flush=True)
    for warning in report["warnings"]:
        print("REVIEW NOTE: " + warning, flush=True)
    print("COLLECTION COMPLETE - NOT BENCHMARK AUTHORIZATION", flush=True)
    print("Upload the Evidence ZIP. Do not run or edit v5.", flush=True)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EvidenceError, OSError, ValueError, SyntaxError, zipfile.BadZipFile, RuntimeError) as exc:
        print("DIAGNOSTIC STOPPED: " + type(exc).__name__ + ": " + str(exc), file=sys.stderr, flush=True)
        print("Do not edit frozen hashes, data or locks. Benchmark was not started.", file=sys.stderr, flush=True)
        raise SystemExit(1)
