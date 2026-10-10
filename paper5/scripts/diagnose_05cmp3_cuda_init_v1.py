#!/usr/bin/env python3
"""Inspect the pinned v3 source and compare cold/initialized CUDA stats resets.

Run from the project root with the existing move_1_5_0 interpreter:
    python -I -B ./paper5/scripts/diagnose_05cmp3_cuda_init_v1.py

No runner, MOVE model, R code, simulation data, or results are executed/read.
Two separate child processes import installed PyTorch. The initialized probe
runs only a tiny constant-valued CUDA tensor calculation, not a model fit.
No project files are created or modified by this script.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

RUNNER_NAME = "05cmp3_run_secondary_comparator_confirmatory_benchmark_v3.py"
EXPECTED_SHA256 = "41bee9179af62cc56104dbdfa4dfa47ec8d82110d5bd90078bb06c3bfacd21d8"
SELECTED_FUNCTIONS = (
    "import_move_api", "seed_everything", "fit_move", "run_synthetic_move_test",
)
MARKER = "PAPER5_CUDA_PROBE_JSON="

# Only this fixed, standalone probe is executed. The runner source is never exec'd.
PROBE_CODE = r'''
import importlib.metadata
import inspect
import json
import os
import sys
import traceback

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

mode = sys.argv[1]
report = {"mode": mode, "python_executable": sys.executable,
          "python_version": sys.version, "stages": {}}
report["environment"] = {
    key: os.environ.get(key) for key in (
        "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "PYTORCH_CUDA_ALLOC_CONF",
        "PYTORCH_ALLOC_CONF", "PYTORCH_NVML_BASED_CUDA_CHECK",
        "PYTORCH_NO_CUDA_MEMORY_CACHING", "CUDA_MODULE_LOADING",
    )
}

def stage(name, function):
    try:
        value = function()
    except Exception as exc:
        report["stages"][name] = {
            "ok": False, "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
        }
        return False
    report["stages"][name] = {"ok": True, "value": value}
    return True

try:
    import torch
    report["torch_version"] = str(torch.__version__)
    report["torch_file"] = str(torch.__file__)
    report["torch_cuda_build"] = torch.version.cuda
    try:
        report["move_dl_distribution_version"] = importlib.metadata.version("move-dl")
    except importlib.metadata.PackageNotFoundError:
        report["move_dl_distribution_version"] = "NOT FOUND"

    # Do not get GPU properties/name, select a device, allocate a tensor, or
    # synchronize before the cold test: those calls may initialize CUDA.
    report["initialized_after_import"] = bool(torch.cuda.is_initialized())
    report["cuda_available"] = bool(torch.cuda.is_available())
    report["device_count"] = int(torch.cuda.device_count())
    report["initialized_after_availability_checks"] = bool(torch.cuda.is_initialized())

    if mode == "cold":
        report["initialized_before_reset"] = bool(torch.cuda.is_initialized())
        stage("reset_without_explicit_init", lambda: torch.cuda.reset_peak_memory_stats(0))
        report["initialized_after_reset"] = bool(torch.cuda.is_initialized())
        # Inspect the actual installed implementation, AFTER the cold experiment.
        try:
            report["installed_reset_source"] = inspect.getsource(torch.cuda.reset_peak_memory_stats)
        except (OSError, TypeError) as exc:
            report["installed_reset_source_error"] = repr(exc)
    elif mode == "initialized":
        if stage("explicit_init", torch.cuda.init):
            report["initialized_after_init"] = bool(torch.cuda.is_initialized())
            # Crucially, reset immediately after init, before set_device or tensors.
            if stage("reset_after_init", lambda: torch.cuda.reset_peak_memory_stats(0)):
                def kernel_probe():
                    torch.cuda.set_device(0)
                    x = torch.ones(8, dtype=torch.float32, device="cuda:0")
                    x.add_(1.0)
                    value = float(x.sum().item())
                    torch.cuda.synchronize(0)
                    if x.device.type != "cuda" or x.device.index != 0:
                        raise RuntimeError("Unexpected tensor device: " + str(x.device))
                    if value != 16.0:
                        raise RuntimeError("Constant-valued CUDA probe returned " + repr(value))
                    result = {
                        "tensor_device": str(x.device), "sum": value,
                        "gpu_name": torch.cuda.get_device_name(0),
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                    }
                    del x
                    torch.cuda.synchronize(0)
                    # Exercise a later reset too, after work on the same device.
                    torch.cuda.reset_peak_memory_stats(0)
                    result["second_reset"] = "PASS"
                    return result
                stage("constant_tensor_kernel", kernel_probe)
    else:
        raise ValueError("Unknown probe mode: " + repr(mode))
except Exception as exc:
    report["fatal_error"] = {
        "type": type(exc).__name__, "message": str(exc),
        "traceback": traceback.format_exc(),
    }

print("PAPER5_CUDA_PROBE_JSON=" + json.dumps(report, ensure_ascii=True, sort_keys=True))
'''


def show_source(raw: bytes) -> None:
    text = raw.decode("utf-8-sig")
    tree = ast.parse(text, filename=RUNNER_NAME)
    lines = text.splitlines()
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    missing = set(SELECTED_FUNCTIONS) - set(functions)
    if missing:
        raise RuntimeError("Expected source functions missing: " + repr(sorted(missing)))
    for name in SELECTED_FUNCTIONS:
        node = functions[name]
        print(f"\n--- SOURCE ONLY: {name}, lines {node.lineno}-{node.end_lineno} ---")
        for i in range(node.lineno - 1, node.end_lineno):
            print(f"{i + 1:04d}: {lines[i]}")
    print("\n--- CUDA CALL INDEX: complete runner, not executed ---")
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if "cuda" in name.lower():
                calls.append((node.lineno, ast.unparse(node)))
    for lineno, expression in sorted(calls):
        print(f"{lineno:04d}: {expression}")


def run_probe(mode: str) -> dict:
    command = [sys.executable, "-I", "-B", "-c", PROBE_CODE, mode]
    completed = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=90, check=False,
    )
    records = []
    for line in completed.stdout.splitlines():
        if line.startswith(MARKER):
            records.append(json.loads(line[len(MARKER):]))
        elif line.strip():
            print("CHILD STDOUT: " + line)
    if completed.stderr.strip():
        print("CHILD STDERR:\n" + completed.stderr.rstrip())
    if completed.returncode != 0 or len(records) != 1:
        raise RuntimeError(
            f"{mode} probe did not return one report: "
            f"exit_code={completed.returncode}; reports={len(records)}"
        )
    report = records[0]
    print(json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True))
    return report


def stage_passed(report: dict, name: str) -> bool:
    return report.get("stages", {}).get(name, {}).get("ok") is True


def classify(cold: dict, initialized: dict) -> str:
    reset = cold.get("stages", {}).get("reset_without_explicit_init", {})
    reproduced = (
        "fatal_error" not in cold and "fatal_error" not in initialized
        and cold.get("torch_version") == initialized.get("torch_version")
        and cold.get("torch_file") == initialized.get("torch_file")
        and cold.get("cuda_available") is True
        and cold.get("device_count", 0) >= 1
        and cold.get("initialized_before_reset") is False
        and reset.get("ok") is False
        and reset.get("error_type") == "RuntimeError"
        and "invalid device argument" in reset.get("error", "").lower()
        and initialized.get("initialized_after_availability_checks") is False
        and initialized.get("initialized_after_init") is True
        and stage_passed(initialized, "explicit_init")
        and stage_passed(initialized, "reset_after_init")
        and stage_passed(initialized, "constant_tensor_kernel")
    )
    if reproduced:
        return "STANDALONE_LAZY_INIT_FAILURE_REPRODUCED_AND_EXPLICIT_INIT_SUCCEEDED"
    return "NOT_ESTABLISHED_REVIEW_SOURCE_AND_PROBE_REPORTS"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    project = Path.cwd().resolve()
    runner = project / "paper5" / "scripts" / RUNNER_NAME
    raw = runner.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"Pinned v3 SHA mismatch: {digest}")
    expected_env = (project / "paper5" / "comparator_envs" / "move_1_5_0").resolve()
    if Path(sys.prefix).resolve() != expected_env:
        raise RuntimeError(
            f"Wrong interpreter environment: {sys.prefix}; expected: {expected_env}"
        )
    print("Paper 5 - pinned v3 CUDA initialization diagnostic")
    print("Runner SHA-256: " + digest)
    print("Parent interpreter: " + sys.executable)
    print("Source: read as text only; no runner/MOVE/R execution")
    print("GPU: two standalone probes; one tiny constant-valued tensor test")
    print("No simulation/outcome/result access; no source/contract/package modification")
    show_source(raw)
    reports = {}
    for mode in ("cold", "initialized"):
        print("\n=== FRESH PROCESS: " + mode.upper() + " ===", flush=True)
        reports[mode] = run_probe(mode)
    if runner.read_bytes() != raw:
        raise RuntimeError("Runner changed during diagnostic; review before proceeding")
    print("\nDIAGNOSTIC FINDING: " + classify(reports["cold"], reports["initialized"]))
    print("This finding concerns the standalone probe, not execution of the runner.")
    print("Runner source unchanged: YES")
    print("BENCHMARK AUTHORIZATION: NOT CREATED")
    print("DIAGNOSTIC COLLECTION COMPLETE - DO NOT LAUNCH THE BENCHMARK FROM THIS RESULT")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("DIAGNOSTIC STOPPED", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
