from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05cmp0c-smoke-test-icluster2b-cpu-v1-no-cli"
CONTRACT_VERSION = "sable-mo-icluster2b-cpu-synthetic-smoke-v1"
STATUS = "PASS_SYNTHETIC_ICLUSTER2B_CPU_SMOKE_BEFORE_05CMP1"

EXPECTED_R_VERSION_PREFIX = "R version 4.6.1"
EXPECTED_BIOCONDUCTOR_VERSION = "3.23"
EXPECTED_ICLUSTERPLUS_VERSION = "1.48.0"
EXPECTED_SYNTHETIC_SAMPLES = 24
VIEW_KEYS = ("rna", "methylation", "cnv")
EXPECTED_FEATURES = {"rna": 40, "methylation": 50, "cnv": 30}

# Development-only smoke configuration. These values exercise the current
# iCluster2b API and are NOT scientific comparator hyperparameters. 05cmp1 must
# freeze the scientific K/lambda/tuning/preprocessing policy independently.
SMOKE_SEED = 1103
SMOKE_K = 3
SMOKE_LAMBDA = 10.0
SMOKE_EM_ITER = 10
SMOKE_EPS = 1e-4
SMOKE_EPS2 = 1e-6
SMOKE_METHOD = "lasso"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# -----------------------------------------------------------------------------
# Frozen prospective comparator anchors
# -----------------------------------------------------------------------------

DIR_05CMP0 = PAPER5_ROOT / "method_contract" / "05cmp0"
LOCK_05CMP0 = DIR_05CMP0 / "secondary_comparator_selection_contract_lock.json"
CONTRACT_05CMP0 = DIR_05CMP0 / "secondary_comparator_selection_contract.json"

DIR_05CMP0A = PAPER5_ROOT / "method_contract" / "05cmp0a"
LOCK_05CMP0A = DIR_05CMP0A / "secondary_comparator_continuous_data_amendment_lock.json"
AMENDMENT_05CMP0A = DIR_05CMP0A / "secondary_comparator_continuous_data_amendment.json"
EFFECTIVE_SELECTED_05CMP0A = DIR_05CMP0A / "effective_secondary_comparator_selected_registry.csv"

DIR_05CMP0B = PAPER5_ROOT / "method_contract" / "05cmp0b"
LOCK_05CMP0B = DIR_05CMP0B / "move_gpu_smoke_lock.json"
RESULT_05CMP0B = DIR_05CMP0B / "move_gpu_smoke_result.json"

DIR_05DEV0 = PAPER5_ROOT / "method_contract" / "05dev0"
LOCK_05DEV0 = DIR_05DEV0 / "target_input_synthetic_fixture_lock.json"
CONTRACT_05DEV0 = DIR_05DEV0 / "target_input_synthetic_fixture_contract.json"

FIXTURE_ROOT = PAPER5_ROOT / "development_fixtures" / "target_input_synthetic_v1"
SAMPLE_MANIFEST = FIXTURE_ROOT / "sample_manifest.csv"
AVAILABILITY = FIXTURE_ROOT / "view_availability.csv"
RNA_NPY = FIXTURE_ROOT / "rna.npy"
METH_NPY = FIXTURE_ROOT / "methylation.npy"
CNV_NPY = FIXTURE_ROOT / "cnv.npy"

# Isolated R library established prospectively for this comparator branch.
R_LIBRARY = PAPER5_ROOT / "comparator_envs" / "r_4_6_bioc_3_23"

DIR_05CMP1 = PAPER5_ROOT / "method_contract" / "05cmp1"
SECONDARY_RESULTS = PAPER5_ROOT / "results" / "secondary_comparators"
DIR_05E = PAPER5_ROOT / "method_contract" / "05e"

# -----------------------------------------------------------------------------
# Outputs: implementation smoke only. No latent/coefficient arrays are persisted.
# -----------------------------------------------------------------------------

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05cmp0c"
OUTPUT_ENVIRONMENT = OUTPUT_DIR / "icluster2b_cpu_smoke_environment.json"
OUTPUT_RESULT = OUTPUT_DIR / "icluster2b_cpu_smoke_result.json"
OUTPUT_R_LOG_1 = OUTPUT_DIR / "icluster2b_cpu_smoke_r_run1.log"
OUTPUT_R_LOG_2 = OUTPUT_DIR / "icluster2b_cpu_smoke_r_run2.log"
OUTPUT_README = OUTPUT_DIR / "05cmp0c_README.txt"
OUTPUT_LOCK = OUTPUT_DIR / "icluster2b_cpu_smoke_lock.json"


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "na"} else text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_array(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(b"|")
    digest.update(str(tuple(values.shape)).encode("utf-8"))
    digest.update(b"|")
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_project_path(value: Any) -> Path:
    path = Path(clean_text(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Required directory not found: {path}")


def load_json(path: Path) -> dict[str, Any]:
    require_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def file_record(path: Path) -> dict[str, Any]:
    require_file(path)
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def verify_generated_files(lock: dict[str, Any], label: str) -> None:
    failures: list[str] = []
    for relative_name, record in lock.get("generated_files", {}).items():
        candidate = resolve_project_path(relative_name)
        if not candidate.is_file():
            failures.append(f"missing: {relative_name}")
            continue
        expected_size = record.get("size_bytes")
        if expected_size is not None and int(candidate.stat().st_size) != int(expected_size):
            failures.append(f"size mismatch: {relative_name}")
            continue
        expected_hash = clean_text(record.get("sha256"))
        if expected_hash and sha256_file(candidate) != expected_hash:
            failures.append(f"SHA256 mismatch: {relative_name}")
    if failures:
        raise RuntimeError(
            f"{label} generated-file verification failed:\n- " + "\n- ".join(failures[:100])
        )


def any_files(path: Path) -> bool:
    if path.is_file():
        return True
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def parse_bool_series(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    unknown = sorted(set(values) - set(mapping))
    if unknown:
        raise RuntimeError(f"Unparseable boolean values: {unknown[:10]}")
    return values.map(mapping).astype(bool)


def tree_sha256(root: Path) -> tuple[str, int, int]:
    require_dir(root)
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


# -----------------------------------------------------------------------------
# Existing PASS verification
# -----------------------------------------------------------------------------

def verify_existing_smoke() -> None:
    require_file(OUTPUT_LOCK)
    require_file(OUTPUT_RESULT)
    require_file(OUTPUT_ENVIRONMENT)

    lock = load_json(OUTPUT_LOCK)
    result = load_json(OUTPUT_RESULT)

    if clean_text(lock.get("status")) != STATUS:
        raise RuntimeError("Unexpected existing 05cmp0c lock status.")
    if clean_text(result.get("status")) != STATUS:
        raise RuntimeError("Unexpected existing 05cmp0c result status.")
    if clean_text(lock.get("script_version")) != SCRIPT_VERSION:
        raise RuntimeError("Existing 05cmp0c script version mismatch.")

    verify_generated_files(lock, "05cmp0c")

    script_record = lock.get("script_source", {})
    expected_script_hash = clean_text(script_record.get("sha256"))
    if expected_script_hash and sha256_file(SCRIPT_PATH) != expected_script_hash:
        raise RuntimeError("Existing 05cmp0c script-source hash mismatch.")

    for relative_name, expected_hash in lock.get("upstream_hashes", {}).items():
        candidate = resolve_project_path(relative_name)
        require_file(candidate)
        if sha256_file(candidate) != clean_text(expected_hash):
            raise RuntimeError(f"05cmp0c upstream hash mismatch: {relative_name}")

    print("=" * 88)
    print("Existing Paper 5 iCluster2b CPU synthetic smoke test: VERIFIED PASS")
    print("=" * 88)
    print(f"Result: {relative_path(OUTPUT_RESULT)}")
    print(f"Environment: {relative_path(OUTPUT_ENVIRONMENT)}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("No files were rewritten.")
    print("=" * 88)


# -----------------------------------------------------------------------------
# Frozen upstream verification
# -----------------------------------------------------------------------------

def verify_upstream() -> dict[str, Any]:
    required = [
        LOCK_05CMP0,
        CONTRACT_05CMP0,
        LOCK_05CMP0A,
        AMENDMENT_05CMP0A,
        EFFECTIVE_SELECTED_05CMP0A,
        LOCK_05CMP0B,
        RESULT_05CMP0B,
        LOCK_05DEV0,
        CONTRACT_05DEV0,
        SAMPLE_MANIFEST,
        AVAILABILITY,
        RNA_NPY,
        METH_NPY,
        CNV_NPY,
    ]
    for path in required:
        require_file(path)

    require_dir(R_LIBRARY)

    if any_files(DIR_05CMP1):
        raise RuntimeError(
            "05cmp1 already contains files. This synthetic implementation smoke belongs before 05cmp1."
        )
    if any_files(SECONDARY_RESULTS):
        raise RuntimeError(
            "Secondary-comparator scientific results already exist; refusing a pre-05cmp1 smoke run."
        )

    lock0 = load_json(LOCK_05CMP0)
    contract0 = load_json(CONTRACT_05CMP0)
    expected0 = "FROZEN_SECONDARY_COMPARATOR_SELECTION_BEFORE_05E_AND_TARGET_RESULTS"
    if clean_text(lock0.get("status")) != expected0 or clean_text(contract0.get("status")) != expected0:
        raise RuntimeError("05cmp0 is not in the expected frozen state.")
    if sha256_file(CONTRACT_05CMP0) != clean_text(lock0.get("contract_sha256")):
        raise RuntimeError("05cmp0 contract hash mismatch.")
    verify_generated_files(lock0, "05cmp0")

    lock0a = load_json(LOCK_05CMP0A)
    amendment0a = load_json(AMENDMENT_05CMP0A)
    expected0a = "FROZEN_COMPARATOR_METHOD_CORRECTION_BEFORE_SCIENTIFIC_RUNS"
    if clean_text(lock0a.get("status")) != expected0a or clean_text(amendment0a.get("status")) != expected0a:
        raise RuntimeError("05cmp0a is not in the expected frozen state.")
    if sha256_file(AMENDMENT_05CMP0A) != clean_text(lock0a.get("amendment_sha256")):
        raise RuntimeError("05cmp0a amendment hash mismatch.")
    verify_generated_files(lock0a, "05cmp0a")

    selected = pd.read_csv(EFFECTIVE_SELECTED_05CMP0A, dtype=str).fillna("")
    classical = selected[selected["comparator_role"].eq("required_external_classical")]
    if classical.shape[0] != 1 or clean_text(classical.iloc[0]["method_name"]) != "iCluster2b":
        raise RuntimeError(
            "Effective comparator registry does not contain iCluster2b as the unique required classical comparator."
        )
    if clean_text(classical.iloc[0]["synthetic_smoke_test_allowed_before_05e"]).lower() not in {
        "true",
        "1",
        "yes",
    }:
        raise RuntimeError("05cmp0/05cmp0a no longer permit the iCluster2b synthetic smoke test.")
    if clean_text(classical.iloc[0]["scientific_run_allowed_before_05e"]).lower() in {
        "true",
        "1",
        "yes",
    }:
        raise RuntimeError("Unexpected permission for a pre-05e scientific iCluster2b run.")

    lock0b = load_json(LOCK_05CMP0B)
    result0b = load_json(RESULT_05CMP0B)
    expected0b = "PASS_SYNTHETIC_MOVE_GPU_SMOKE_BEFORE_05CMP1"
    if clean_text(lock0b.get("status")) != expected0b or clean_text(result0b.get("status")) != expected0b:
        raise RuntimeError("05cmp0b MOVE smoke is not in the expected PASS state.")
    verify_generated_files(lock0b, "05cmp0b")

    lockdev = load_json(LOCK_05DEV0)
    contractdev = load_json(CONTRACT_05DEV0)
    expecteddev = "FROZEN_SYNTHETIC_TARGET_INPUT_FIXTURES_BEFORE_TARGET_CODE"
    if clean_text(lockdev.get("status")) != expecteddev or clean_text(contractdev.get("status")) != expecteddev:
        raise RuntimeError("05dev0 is not in the expected frozen state.")
    if sha256_file(CONTRACT_05DEV0) != clean_text(lockdev.get("contract_sha256")):
        raise RuntimeError("05dev0 contract hash mismatch.")
    verify_generated_files(lockdev, "05dev0")

    science = contractdev.get("scientific_state", {})
    forbidden_true = [
        "real_TARGET_values_accessed",
        "TARGET_outcomes_accessed",
        "biological_annotations_accessed",
        "final_representation_method_selected_by_fixture",
        "latent_output_schema_frozen_by_fixture",
        "scientific_TARGET_splits_defined_by_fixture",
        "scientific_TARGET_seeds_defined_by_fixture",
    ]
    active = [key for key in forbidden_true if bool(science.get(key, False))]
    if active:
        raise RuntimeError("05dev0 scientific boundary violated: " + ", ".join(active))

    return {
        "selected_registry": selected,
        "05e_state": "FILES_PRESENT" if any_files(DIR_05E) else "NO_FILES_DETECTED",
        "upstream_hashes": {
            relative_path(path): sha256_file(path)
            for path in required
            if path.suffix.lower() in {".json", ".csv", ".npy"}
        },
    }


# -----------------------------------------------------------------------------
# R/iClusterPlus environment discovery
# -----------------------------------------------------------------------------

def find_rscript() -> Path:
    explicit = clean_text(os.environ.get("PAPER5_RSCRIPT"))
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))

    on_path = shutil.which("Rscript")
    if on_path:
        candidates.append(Path(on_path))

    if os.name == "nt":
        candidates.extend(
            [
                Path(r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe"),
                Path(r"C:\Program Files\R\R-4.6.1\bin\x64\Rscript.exe"),
            ]
        )

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()

    raise RuntimeError(
        "Rscript was not found. Expected R 4.6.1. Set PAPER5_RSCRIPT or install/locate "
        r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe."
    )


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


def probe_r_environment(rscript: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env["R_LIBS_USER"] = str(R_LIBRARY.resolve())
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )

    expression = r'''
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages(library(BiocManager))
suppressPackageStartupMessages(library(iClusterPlus))
cat("R_VERSION=", R.version.string, "\n", sep="")
cat("BIOC_VERSION=", as.character(BiocManager::version()), "\n", sep="")
cat("ICLUSTERPLUS_VERSION=", as.character(packageVersion("iClusterPlus")), "\n", sep="")
cat("ICLUSTERPLUS_PATH=", normalizePath(find.package("iClusterPlus"), winslash="/", mustWork=TRUE), "\n", sep="")
cat("R_LIBS_USER=", normalizePath(Sys.getenv("R_LIBS_USER"), winslash="/", mustWork=TRUE), "\n", sep="")
cat("ICLUSTER2B_EXISTS=", exists("iCluster2b", where=asNamespace("iClusterPlus"), inherits=FALSE), "\n", sep="")
cat("TUNE_ICLUSTER2B_EXISTS=", exists("tune.iCluster2b", where=asNamespace("iClusterPlus"), inherits=FALSE), "\n", sep="")
cat("ICLUSTER2B_FORMALS=", paste(names(formals(iClusterPlus::iCluster2b)), collapse=","), "\n", sep="")
cat("TUNE_ICLUSTER2B_FORMALS=", paste(names(formals(iClusterPlus::tune.iCluster2b)), collapse=","), "\n", sep="")
cat("LIBPATHS=", paste(normalizePath(.libPaths(), winslash="/", mustWork=FALSE), collapse="|"), "\n", sep="")
'''
    completed = run_command([str(rscript), "-e", expression], env=env)
    if completed.returncode != 0:
        raise RuntimeError(
            "R/iClusterPlus environment probe failed.\nSTDOUT:\n"
            + completed.stdout
            + "\nSTDERR:\n"
            + completed.stderr
        )

    parsed: dict[str, str] = {}
    for raw_line in completed.stdout.splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        parsed[key.strip()] = value.strip()

    required_keys = {
        "R_VERSION",
        "BIOC_VERSION",
        "ICLUSTERPLUS_VERSION",
        "ICLUSTERPLUS_PATH",
        "R_LIBS_USER",
        "ICLUSTER2B_EXISTS",
        "TUNE_ICLUSTER2B_EXISTS",
        "ICLUSTER2B_FORMALS",
        "TUNE_ICLUSTER2B_FORMALS",
        "LIBPATHS",
    }
    missing = sorted(required_keys - set(parsed))
    if missing:
        raise RuntimeError("R environment probe did not return expected fields: " + ", ".join(missing))

    if not parsed["R_VERSION"].startswith(EXPECTED_R_VERSION_PREFIX):
        raise RuntimeError(
            f"R version mismatch: {parsed['R_VERSION']}; expected prefix {EXPECTED_R_VERSION_PREFIX!r}."
        )
    if parsed["BIOC_VERSION"] != EXPECTED_BIOCONDUCTOR_VERSION:
        raise RuntimeError(
            f"Bioconductor version mismatch: {parsed['BIOC_VERSION']} != {EXPECTED_BIOCONDUCTOR_VERSION}."
        )
    if parsed["ICLUSTERPLUS_VERSION"] != EXPECTED_ICLUSTERPLUS_VERSION:
        raise RuntimeError(
            f"iClusterPlus version mismatch: {parsed['ICLUSTERPLUS_VERSION']} != "
            f"{EXPECTED_ICLUSTERPLUS_VERSION}. Update the isolated Bioconductor 3.23 library before this smoke test."
        )
    if parsed["ICLUSTER2B_EXISTS"].lower() != "true":
        raise RuntimeError("iCluster2b is not available in the active iClusterPlus namespace.")
    if parsed["TUNE_ICLUSTER2B_EXISTS"].lower() != "true":
        raise RuntimeError("tune.iCluster2b is not available in the active iClusterPlus namespace.")

    required_i2b_formals = {"xList", "K", "lambda", "method", "chr", "EM.iter", "eps", "eps2"}
    observed_i2b_formals = set(parsed["ICLUSTER2B_FORMALS"].split(","))
    missing_i2b_formals = sorted(required_i2b_formals - observed_i2b_formals)
    if missing_i2b_formals:
        raise RuntimeError("iCluster2b API mismatch; missing formals: " + ", ".join(missing_i2b_formals))

    required_tune_formals = {
        "xList",
        "K",
        "method",
        "min.lambda",
        "max.lambda",
        "lambda.iter",
        "EM.iter",
        "min.shrinkage.rate",
        "eps",
        "eps2",
    }
    observed_tune_formals = set(parsed["TUNE_ICLUSTER2B_FORMALS"].split(","))
    missing_tune_formals = sorted(required_tune_formals - observed_tune_formals)
    if missing_tune_formals:
        raise RuntimeError(
            "tune.iCluster2b API mismatch; missing formals: " + ", ".join(missing_tune_formals)
        )

    package_path = Path(parsed["ICLUSTERPLUS_PATH"])
    require_dir(package_path)
    expected_library = R_LIBRARY.resolve()
    try:
        package_path.resolve().relative_to(expected_library)
    except ValueError as exc:
        raise RuntimeError(
            "iClusterPlus was not loaded from the isolated Paper 5 R library. "
            f"Loaded: {package_path}; expected under: {expected_library}."
        ) from exc

    package_tree_hash, package_file_count, package_total_bytes = tree_sha256(package_path)

    return {
        "rscript_path": str(rscript),
        "rscript_sha256": sha256_file(rscript),
        "rscript_size_bytes": int(rscript.stat().st_size),
        "r_version": parsed["R_VERSION"],
        "bioconductor_version": parsed["BIOC_VERSION"],
        "iclusterplus_version": parsed["ICLUSTERPLUS_VERSION"],
        "iclusterplus_path": str(package_path),
        "iclusterplus_tree_sha256": package_tree_hash,
        "iclusterplus_tree_file_count": package_file_count,
        "iclusterplus_tree_total_bytes": package_total_bytes,
        "r_libs_user": parsed["R_LIBS_USER"],
        "lib_paths": parsed["LIBPATHS"].split("|") if parsed["LIBPATHS"] else [],
        "icluster2b_formals": parsed["ICLUSTER2B_FORMALS"].split(","),
        "tune_icluster2b_formals": parsed["TUNE_ICLUSTER2B_FORMALS"].split(","),
        "thread_environment": {
            key: env[key]
            for key in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ]
        },
        "probe_stdout": completed.stdout,
        "probe_stderr": completed.stderr,
    }


# -----------------------------------------------------------------------------
# 05dev0 adapter validation
# -----------------------------------------------------------------------------

def load_and_validate_fixture() -> dict[str, Any]:
    sample_manifest = pd.read_csv(SAMPLE_MANIFEST, dtype=str).fillna("")
    availability = pd.read_csv(AVAILABILITY, dtype=str).fillna("")

    if sample_manifest.shape[0] != EXPECTED_SYNTHETIC_SAMPLES:
        raise RuntimeError("05dev0 sample-manifest count changed.")
    if availability.shape[0] != EXPECTED_SYNTHETIC_SAMPLES:
        raise RuntimeError("05dev0 availability-row count changed.")

    expected_ids = sample_manifest["synthetic_sample_id"].astype(str).tolist()
    observed_ids = availability["synthetic_sample_id"].astype(str).tolist()
    if observed_ids != expected_ids:
        raise RuntimeError("05dev0 availability/sample-manifest order mismatch.")
    if len(set(expected_ids)) != EXPECTED_SYNTHETIC_SAMPLES:
        raise RuntimeError("05dev0 contains duplicate synthetic sample IDs.")

    availability_bool = pd.DataFrame(
        {
            "rna": parse_bool_series(availability["rna_available"]),
            "methylation": parse_bool_series(availability["methylation_available"]),
            "cnv": parse_bool_series(availability["cnv_available"]),
        }
    )
    if (availability_bool.sum(axis=1) < 1).any():
        raise RuntimeError("05dev0 valid fixture contains an all-views-missing sample.")

    raw_matrices = {
        "rna": np.load(RNA_NPY, allow_pickle=False),
        "methylation": np.load(METH_NPY, allow_pickle=False),
        "cnv": np.load(CNV_NPY, allow_pickle=False),
    }

    audit: dict[str, Any] = {}
    encoded: dict[str, np.ndarray] = {}
    for key in VIEW_KEYS:
        matrix = np.asarray(raw_matrices[key], dtype=np.float64)
        expected_shape = (EXPECTED_SYNTHETIC_SAMPLES, EXPECTED_FEATURES[key])
        if matrix.shape != expected_shape:
            raise RuntimeError(f"05dev0 {key} matrix shape changed: {matrix.shape} != {expected_shape}")

        available = availability_bool[key].to_numpy(dtype=bool)
        all_nan = np.isnan(matrix).all(axis=1)
        if np.any(all_nan[available]):
            raise RuntimeError(f"05dev0 {key}: available row is all NaN.")
        if np.any(~all_nan[~available]):
            raise RuntimeError(f"05dev0 {key}: unavailable row is not all NaN.")

        feature_level_nan_available = int(np.isnan(matrix[available, :]).sum())
        whole_view_missing_rows = int((~available).sum())

        # Development-only complete-matrix adapter. iCluster2b expects ordinary
        # numeric sample-by-feature matrices and does not expose a missingness mask
        # in this API. 05dev0 is already standardized synthetic input, so NaN->0 is
        # used ONLY to exercise the implementation. 05cmp1 must independently freeze
        # the scientific missing-view/preprocessing policy before any real comparator run.
        adapted = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
        if not np.isfinite(adapted).all():
            raise RuntimeError(f"iCluster2b development adapter produced non-finite {key} input.")
        if np.any(adapted[~available, :] != 0.0):
            raise RuntimeError(f"iCluster2b adapter did not zero-code unavailable {key} rows.")

        encoded[key] = adapted
        audit[key] = {
            "shape": list(matrix.shape),
            "dtype_before_adapter": str(matrix.dtype),
            "dtype_after_adapter": str(adapted.dtype),
            "whole_view_missing_rows": whole_view_missing_rows,
            "feature_level_nan_count_in_available_rows": feature_level_nan_available,
            "adapter": "NaN_to_zero_only_on_already_standardized_05dev0_fixture",
            "encoded_finite": True,
            "encoded_sha256": sha256_array(adapted),
        }

    return {
        "sample_ids": expected_ids,
        "availability": availability_bool,
        "encoded": encoded,
        "audit": audit,
    }


# -----------------------------------------------------------------------------
# Embedded R implementation smoke runner
# -----------------------------------------------------------------------------

R_SMOKE_SCRIPT = r'''
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 10) {
  stop("Expected 10 command-line arguments, received ", length(args))
}

input_dir <- args[[1]]
output_dir <- args[[2]]
K <- as.integer(args[[3]])
lambda_value <- as.numeric(args[[4]])
EM_iter <- as.integer(args[[5]])
eps <- as.numeric(args[[6]])
eps2 <- as.numeric(args[[7]])
seed <- as.integer(args[[8]])
expected_pkg <- args[[9]]
expected_bioc <- args[[10]]

dir.create(output_dir, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages(library(BiocManager))
suppressPackageStartupMessages(library(iClusterPlus))

if (as.character(packageVersion("iClusterPlus")) != expected_pkg) {
  stop("iClusterPlus version mismatch inside smoke runner")
}
if (as.character(BiocManager::version()) != expected_bioc) {
  stop("Bioconductor version mismatch inside smoke runner")
}
if (!exists("iCluster2b", where=asNamespace("iClusterPlus"), inherits=FALSE)) {
  stop("iCluster2b is unavailable")
}
if (!exists("tune.iCluster2b", where=asNamespace("iClusterPlus"), inherits=FALSE)) {
  stop("tune.iCluster2b is unavailable")
}

required_i2b <- c("xList", "K", "lambda", "method", "chr", "EM.iter", "eps", "eps2")
missing_i2b <- setdiff(required_i2b, names(formals(iClusterPlus::iCluster2b)))
if (length(missing_i2b) > 0) {
  stop("iCluster2b API mismatch: ", paste(missing_i2b, collapse=","))
}
required_tune <- c("xList", "K", "method", "min.lambda", "max.lambda", "lambda.iter", "EM.iter", "min.shrinkage.rate", "eps", "eps2")
missing_tune <- setdiff(required_tune, names(formals(iClusterPlus::tune.iCluster2b)))
if (length(missing_tune) > 0) {
  stop("tune.iCluster2b API mismatch: ", paste(missing_tune, collapse=","))
}

read_matrix <- function(name) {
  path <- file.path(input_dir, paste0(name, ".csv"))
  x <- as.matrix(read.csv(path, header=FALSE, check.names=FALSE))
  storage.mode(x) <- "double"
  if (!all(is.finite(x))) {
    stop("Non-finite values in adapted input: ", name)
  }
  x
}

xList <- list(
  read_matrix("rna"),
  read_matrix("methylation"),
  read_matrix("cnv")
)

expected_rows <- 24L
expected_cols <- c(40L, 50L, 30L)
for (i in seq_along(xList)) {
  if (nrow(xList[[i]]) != expected_rows || ncol(xList[[i]]) != expected_cols[[i]]) {
    stop("Adapted input dimension mismatch for view ", i)
  }
}

set.seed(seed)
lambda <- as.list(rep(lambda_value, length(xList)))
method <- rep("lasso", length(xList))
started <- proc.time()[["elapsed"]]
fit <- iClusterPlus::iCluster2b(
  xList=xList,
  K=K,
  lambda=lambda,
  method=method,
  chr=NULL,
  EM.iter=EM_iter,
  eps=eps,
  eps2=eps2
)
elapsed <- proc.time()[["elapsed"]] - started

if (is.null(fit$meanZ)) stop("iCluster2b result has no meanZ")
meanZ <- as.matrix(fit$meanZ)
storage.mode(meanZ) <- "double"
if (nrow(meanZ) != expected_rows) stop("meanZ row count mismatch")
if (ncol(meanZ) != K) stop("meanZ latent dimension mismatch")
if (!all(is.finite(meanZ))) stop("meanZ contains non-finite values")

if (is.null(fit$beta) || length(fit$beta) != length(xList)) {
  stop("iCluster2b beta list length mismatch")
}
for (i in seq_along(fit$beta)) {
  beta_i <- as.matrix(fit$beta[[i]])
  storage.mode(beta_i) <- "double"
  if (nrow(beta_i) != expected_cols[[i]]) {
    stop("beta row count mismatch for view ", i)
  }
  if (ncol(beta_i) != K) {
    stop("beta factor count mismatch for view ", i)
  }
  if (!all(is.finite(beta_i))) {
    stop("beta contains non-finite values for view ", i)
  }
  write.table(beta_i, file=file.path(output_dir, paste0("beta_", i, ".csv")), sep=",", row.names=FALSE, col.names=FALSE, quote=FALSE)
}

iter_value <- if (is.null(fit$iter)) NA_real_ else as.numeric(fit$iter[[1]])
if (length(iter_value) != 1 || !is.finite(iter_value) || iter_value < 0) {
  stop("Unexpected iCluster2b iteration diagnostic")
}

write.table(meanZ, file=file.path(output_dir, "meanZ.csv"), sep=",", row.names=FALSE, col.names=FALSE, quote=FALSE)

metadata <- data.frame(
  key=c(
    "r_version", "bioconductor_version", "iclusterplus_version", "iclusterplus_path",
    "K", "lambda_value", "method", "EM_iter", "eps", "eps2", "seed",
    "meanZ_rows", "meanZ_cols", "beta_list_length", "iter", "elapsed_seconds"
  ),
  value=c(
    R.version.string,
    as.character(BiocManager::version()),
    as.character(packageVersion("iClusterPlus")),
    normalizePath(find.package("iClusterPlus"), winslash="/", mustWork=TRUE),
    as.character(K), as.character(lambda_value), "lasso", as.character(EM_iter),
    format(eps, scientific=TRUE), format(eps2, scientific=TRUE), as.character(seed),
    as.character(nrow(meanZ)), as.character(ncol(meanZ)), as.character(length(fit$beta)),
    as.character(iter_value), as.character(elapsed)
  ),
  stringsAsFactors=FALSE
)
write.table(metadata, file=file.path(output_dir, "metadata.tsv"), sep="\t", row.names=FALSE, col.names=TRUE, quote=FALSE)

session_lines <- capture.output(sessionInfo())
writeLines(session_lines, con=file.path(output_dir, "sessionInfo.txt"), useBytes=TRUE)
cat("ICLUSTER2B_SMOKE_R_PASS\n")
'''


def write_adapter_inputs(fixture: dict[str, Any], input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    for key in VIEW_KEYS:
        path = input_dir / f"{key}.csv"
        np.savetxt(path, fixture["encoded"][key], delimiter=",", fmt="%.17g")


def read_key_value_tsv(path: Path) -> dict[str, str]:
    require_file(path)
    table = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    if list(table.columns) != ["key", "value"]:
        raise RuntimeError(f"Unexpected metadata schema: {path}")
    if table["key"].duplicated().any():
        raise RuntimeError(f"Duplicate metadata keys: {path}")
    return dict(zip(table["key"].astype(str), table["value"].astype(str), strict=True))


def run_icluster_once(
    *,
    rscript: Path,
    r_script_path: Path,
    input_dir: Path,
    run_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["R_LIBS_USER"] = str(R_LIBRARY.resolve())
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )

    command = [
        str(rscript),
        str(r_script_path),
        str(input_dir),
        str(run_dir),
        str(SMOKE_K),
        repr(SMOKE_LAMBDA),
        str(SMOKE_EM_ITER),
        repr(SMOKE_EPS),
        repr(SMOKE_EPS2),
        str(SMOKE_SEED),
        EXPECTED_ICLUSTERPLUS_VERSION,
        EXPECTED_BIOCONDUCTOR_VERSION,
    ]
    completed = run_command(command, env=env)
    combined_log = (
        "COMMAND:\n"
        + subprocess.list2cmdline(command)
        + "\n\nSTDOUT:\n"
        + completed.stdout
        + "\nSTDERR:\n"
        + completed.stderr
        + f"\nRETURN_CODE:\n{completed.returncode}\n"
    )
    write_text_atomic(combined_log, log_path)

    if completed.returncode != 0:
        raise RuntimeError(
            "iCluster2b R smoke process failed. Fallback is NOT activated automatically. "
            f"Inspect {relative_path(log_path)}.\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    if "ICLUSTER2B_SMOKE_R_PASS" not in completed.stdout:
        raise RuntimeError("iCluster2b R runner returned zero but did not emit its PASS sentinel.")

    mean_z_path = run_dir / "meanZ.csv"
    metadata_path = run_dir / "metadata.tsv"
    session_path = run_dir / "sessionInfo.txt"
    require_file(mean_z_path)
    require_file(metadata_path)
    require_file(session_path)

    latent = np.loadtxt(mean_z_path, delimiter=",", ndmin=2)
    if latent.shape != (EXPECTED_SYNTHETIC_SAMPLES, SMOKE_K):
        raise RuntimeError(f"Unexpected iCluster2b latent shape: {latent.shape}")
    if not np.isfinite(latent).all():
        raise RuntimeError("iCluster2b latent output contains non-finite values.")

    beta_shapes: list[list[int]] = []
    beta_hashes: list[str] = []
    beta_finite: list[bool] = []
    for index, key in enumerate(VIEW_KEYS, start=1):
        beta_path = run_dir / f"beta_{index}.csv"
        require_file(beta_path)
        beta = np.loadtxt(beta_path, delimiter=",", ndmin=2)
        expected_shape = (EXPECTED_FEATURES[key], SMOKE_K)
        if beta.shape != expected_shape:
            raise RuntimeError(f"Unexpected iCluster2b beta shape for {key}: {beta.shape} != {expected_shape}")
        finite = bool(np.isfinite(beta).all())
        if not finite:
            raise RuntimeError(f"iCluster2b beta contains non-finite values for {key}.")
        beta_shapes.append(list(beta.shape))
        beta_hashes.append(sha256_array(beta))
        beta_finite.append(finite)

    metadata = read_key_value_tsv(metadata_path)
    if metadata.get("iclusterplus_version") != EXPECTED_ICLUSTERPLUS_VERSION:
        raise RuntimeError("Run metadata reports an unexpected iClusterPlus version.")
    if metadata.get("bioconductor_version") != EXPECTED_BIOCONDUCTOR_VERSION:
        raise RuntimeError("Run metadata reports an unexpected Bioconductor version.")

    # Latent/beta arrays stay in a TemporaryDirectory and are intentionally not
    # copied into method_contract or results.
    return {
        "latent": latent,
        "latent_shape": list(latent.shape),
        "latent_all_finite": True,
        "latent_sha256": sha256_array(latent),
        "beta_shapes": beta_shapes,
        "beta_all_finite": all(beta_finite),
        "beta_sha256_by_view": dict(zip(VIEW_KEYS, beta_hashes, strict=True)),
        "reported_iterations": float(metadata["iter"]),
        "elapsed_seconds": float(metadata["elapsed_seconds"]),
        "r_version": metadata["r_version"],
        "bioconductor_version": metadata["bioconductor_version"],
        "iclusterplus_version": metadata["iclusterplus_version"],
        "session_info_sha256": sha256_file(session_path),
        "r_stdout_pass_sentinel": True,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    print("=" * 88)
    print("Paper 5 - iCluster2b CPU synthetic implementation smoke test")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Synthetic fixture root: {FIXTURE_ROOT}")
    print("Real TARGET values accessed: NO")
    print("TARGET outcomes/biology accessed: NO")
    print("Scientific comparator performance generated: NO")
    print("05e primary-method influence: FORBIDDEN")
    print("GPU computation in 05cmp0c: NO")
    print("Automatic iClusterBayes2 fallback: FORBIDDEN")
    print()

    if OUTPUT_LOCK.exists():
        verify_existing_smoke()
        return

    upstream = verify_upstream()
    print("05cmp0 frozen comparator selection: PASS")
    print("05cmp0a continuous-data correction: PASS")
    print("05cmp0b MOVE GPU synthetic smoke: PASS")
    print("05dev0 frozen synthetic fixture: PASS")
    print(f"05e state: {upstream['05e_state']}")
    print("05cmp1 files present: NO")
    print("Secondary comparator scientific results present: NO")
    print()

    rscript = find_rscript()
    r_environment = probe_r_environment(rscript)

    environment = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "R": r_environment,
        "expected": {
            "R_version_prefix": EXPECTED_R_VERSION_PREFIX,
            "Bioconductor_version": EXPECTED_BIOCONDUCTOR_VERSION,
            "iClusterPlus_version": EXPECTED_ICLUSTERPLUS_VERSION,
            "isolated_R_library": relative_path(R_LIBRARY),
        },
        "embedded_R_smoke_script_sha256": sha256_text(R_SMOKE_SCRIPT),
        "GPU_used": False,
    }
    write_json_atomic(environment, OUTPUT_ENVIRONMENT)

    print(f"R: {r_environment['r_version']}")
    print(f"Bioconductor: {r_environment['bioconductor_version']}")
    print(f"iClusterPlus: {r_environment['iclusterplus_version']}")
    print(f"iClusterPlus path: {r_environment['iclusterplus_path']}")
    print("iCluster2b API: PASS")
    print("tune.iCluster2b API: PASS")
    print("Isolated Paper 5 R library: PASS")
    print()

    fixture = load_and_validate_fixture()
    print("05dev0 -> iCluster2b development adapter validation: PASS")
    print(f"Synthetic samples: {len(fixture['sample_ids'])}")
    print(f"Continuous shapes: {[EXPECTED_FEATURES[key] for key in VIEW_KEYS]}")
    print("Missing-value adapter: NaN -> 0 on already-standardized 05dev0 only")
    print("Scientific preprocessing/K/lambda/tuning policy frozen here: NO")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="paper5_05cmp0c_") as temp_name:
        temp_root = Path(temp_name)
        input_dir = temp_root / "input"
        run1_dir = temp_root / "run1"
        run2_dir = temp_root / "run2"
        r_script_path = temp_root / "05cmp0c_runner.R"

        write_adapter_inputs(fixture, input_dir)
        r_script_path.write_text(R_SMOKE_SCRIPT, encoding="utf-8", newline="\n")

        print("Running iCluster2b CPU smoke fit #1...")
        first = run_icluster_once(
            rscript=rscript,
            r_script_path=r_script_path,
            input_dir=input_dir,
            run_dir=run1_dir,
            log_path=OUTPUT_R_LOG_1,
        )
        print("iCluster2b CPU smoke fit #1: PASS")

        print("Running same-seed independent R-process repeatability diagnostic...")
        second = run_icluster_once(
            rscript=rscript,
            r_script_path=r_script_path,
            input_dir=input_dir,
            run_dir=run2_dir,
            log_path=OUTPUT_R_LOG_2,
        )
        print("iCluster2b CPU smoke fit #2: PASS")

        latent_a = first.pop("latent")
        latent_b = second.pop("latent")

    max_abs_difference = float(np.max(np.abs(latent_a - latent_b)))
    exact_match = bool(np.array_equal(latent_a, latent_b))
    close_1e10 = bool(np.allclose(latent_a, latent_b, rtol=1e-10, atol=1e-10))

    beta_hash_match = bool(first["beta_sha256_by_view"] == second["beta_sha256_by_view"])
    repeatability = {
        "same_seed": SMOKE_SEED,
        "independent_R_processes": True,
        "exact_latent_match": exact_match,
        "allclose_rtol_1e-10_atol_1e-10": close_1e10,
        "maximum_absolute_latent_difference": max_abs_difference,
        "run1_latent_sha256": first["latent_sha256"],
        "run2_latent_sha256": second["latent_sha256"],
        "beta_hashes_exact_match": beta_hash_match,
        "gate_consequence": (
            "NONE_IN_05CMP0C; this is an implementation-repeatability diagnostic only. "
            "Scientific seed/tuning/stability rules must be frozen prospectively in 05cmp1."
        ),
    }

    result = {
        "script_version": SCRIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": STATUS,
        "comparator": {
            "method": "iCluster2b",
            "package": "iClusterPlus",
            "package_version": r_environment["iclusterplus_version"],
            "Bioconductor_version": r_environment["bioconductor_version"],
            "role": "required_external_classical_secondary_comparator",
            "same_package_fallback": "iClusterBayes2",
            "fallback_activated": False,
        },
        "scientific_boundaries": {
            "synthetic_05dev0_only": True,
            "real_TARGET_values_accessed": False,
            "TARGET_outcomes_accessed": False,
            "biological_annotations_accessed": False,
            "scientific_comparator_result_generated": False,
            "may_influence_05e": False,
            "may_influence_final_gate": False,
            "scientific_hyperparameters_frozen_here": False,
            "scientific_preprocessing_frozen_here": False,
            "scientific_missing_view_policy_frozen_here": False,
            "scientific_tuning_rule_frozen_here": False,
            "latent_or_beta_values_persisted": False,
        },
        "development_smoke_configuration": {
            "seed": SMOKE_SEED,
            "K": SMOKE_K,
            "lambda_per_view": [SMOKE_LAMBDA] * len(VIEW_KEYS),
            "method_per_view": [SMOKE_METHOD] * len(VIEW_KEYS),
            "EM_iter": SMOKE_EM_ITER,
            "eps": SMOKE_EPS,
            "eps2": SMOKE_EPS2,
            "tune_iCluster2b_executed": False,
            "purpose": (
                "Implementation smoke only. These values exercise the documented iCluster2b continuous-data path "
                "and must not be reused as scientific K/lambda/tuning settings without a prospective 05cmp1 freeze."
            ),
        },
        "fixture_adapter_audit": fixture["audit"],
        "run_1": first,
        "run_2": second,
        "same_seed_repeatability_diagnostic": repeatability,
        "pass_checks": {
            "R_version_exact_expected_prefix": r_environment["r_version"].startswith(EXPECTED_R_VERSION_PREFIX),
            "Bioconductor_exactly_3_23": r_environment["bioconductor_version"] == EXPECTED_BIOCONDUCTOR_VERSION,
            "iClusterPlus_exactly_1_48_0": r_environment["iclusterplus_version"] == EXPECTED_ICLUSTERPLUS_VERSION,
            "iCluster2b_API_available": True,
            "tune_iCluster2b_API_available": True,
            "loaded_from_isolated_Paper5_R_library": True,
            "latent_rows_equal_24": first["latent_shape"][0] == EXPECTED_SYNTHETIC_SAMPLES,
            "latent_columns_equal_development_K": first["latent_shape"][1] == SMOKE_K,
            "latent_finite": bool(first["latent_all_finite"] and second["latent_all_finite"]),
            "beta_shapes_match_three_views": first["beta_shapes"]
            == [[EXPECTED_FEATURES[key], SMOKE_K] for key in VIEW_KEYS],
            "beta_finite": bool(first["beta_all_finite"] and second["beta_all_finite"]),
            "whole_view_missingness_adapter_exercised": any(
                fixture["audit"][key]["whole_view_missing_rows"] > 0 for key in VIEW_KEYS
            ),
            "feature_level_missingness_adapter_exercised": (
                fixture["audit"]["methylation"]["feature_level_nan_count_in_available_rows"] > 0
            ),
            "automatic_fallback_not_activated": True,
        },
        "upstream_hashes": upstream["upstream_hashes"],
        "environment": {
            "path": relative_path(OUTPUT_ENVIRONMENT),
            "sha256": sha256_file(OUTPUT_ENVIRONMENT),
        },
        "next_stage": (
            "Create 05cmp1 to freeze exact iCluster2b/MOVE implementations, scientific adapters, "
            "hyperparameter/tuning rules, seeds and comparable output metrics before any scientific "
            "secondary-comparator run. Secondary comparators remain unable to influence 05e."
        ),
    }

    if not all(bool(value) for value in result["pass_checks"].values()):
        failed = [name for name, value in result["pass_checks"].items() if not bool(value)]
        raise RuntimeError("iCluster2b CPU smoke hard checks failed: " + ", ".join(failed))

    write_json_atomic(result, OUTPUT_RESULT)

    readme = f"""Paper 5 - iCluster2b CPU synthetic implementation smoke test
==============================================================================
Script version: {SCRIPT_VERSION}
Status: {STATUS}

PURPOSE
-------
Verify that the frozen required external classical comparator, iCluster2b, can
be loaded from the isolated R/Bioconductor environment and fit to ONLY the
frozen 05dev0 synthetic three-view fixture.

EXPECTED IMPLEMENTATION
-----------------------
R: {EXPECTED_R_VERSION_PREFIX}*
Bioconductor: {EXPECTED_BIOCONDUCTOR_VERSION}
iClusterPlus: {EXPECTED_ICLUSTERPLUS_VERSION}
method: iCluster2b
same-package fallback: iClusterBayes2 (NOT activated by a PASS smoke test)

WHAT THIS PROVES
----------------
- the isolated Paper 5 R library is used;
- iCluster2b and tune.iCluster2b expose the expected Bioconductor 3.23 API;
- the 24-row RNA/methylation/CNV synthetic fixture is accepted after a
  development-only NaN-to-zero complete-matrix adapter;
- iCluster2b returns finite 24 x {SMOKE_K} meanZ and finite coefficient matrices
  with the expected feature-by-factor dimensions;
- a second independent R process with the same development seed provides a
  repeatability diagnostic for 05cmp1 design.

WHAT THIS DOES NOT DO
---------------------
- no real TARGET value is read;
- no outcome, biology or clinical label is read;
- no scientific comparator performance estimate is generated;
- tune.iCluster2b is NOT used to select a scientific lambda here;
- no scientific K, lambda, tuning, missing-view, preprocessing or stability
  policy is frozen here;
- no meanZ or beta numeric array is persisted;
- iCluster2b cannot influence 05e or the final-gate decision;
- iClusterBayes2 fallback is not activated automatically by this script.

DEVELOPMENT-ONLY SMOKE CONFIGURATION
------------------------------------
seed: {SMOKE_SEED}
K: {SMOKE_K}
lambda per view: {[SMOKE_LAMBDA] * len(VIEW_KEYS)}
method per view: {[SMOKE_METHOD] * len(VIEW_KEYS)}
EM.iter: {SMOKE_EM_ITER}
eps: {SMOKE_EPS}
eps2: {SMOKE_EPS2}

NEXT
----
Create 05cmp1 and prospectively freeze exact comparator implementations and
scientific adapter/tuning rules before any scientific secondary-comparator run.
"""
    write_text_atomic(readme, OUTPUT_README)

    generated = [
        OUTPUT_ENVIRONMENT,
        OUTPUT_RESULT,
        OUTPUT_R_LOG_1,
        OUTPUT_R_LOG_2,
        OUTPUT_README,
    ]
    lock = {
        "script_version": SCRIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": STATUS,
        "result": file_record(OUTPUT_RESULT),
        "environment": file_record(OUTPUT_ENVIRONMENT),
        "generated_files": {
            relative_path(path): file_record(path)
            for path in generated
        },
        "upstream_hashes": upstream["upstream_hashes"],
        "script_source": {
            "path": relative_path(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "embedded_R_smoke_script_sha256": sha256_text(R_SMOKE_SCRIPT),
        "rewrite_policy": (
            "A successful 05cmp0c smoke record is immutable. If the R/Bioconductor/iClusterPlus "
            "installation or implementation changes, create a new versioned smoke/amendment rather "
            "than silently rewriting this PASS."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    print()
    print("-" * 88)
    print("iCluster2b CPU synthetic smoke summary")
    print("-" * 88)
    print(f"R: {r_environment['r_version']}")
    print(f"Bioconductor: {r_environment['bioconductor_version']}")
    print(f"iClusterPlus: {r_environment['iclusterplus_version']}")
    print(f"Synthetic rows projected: {first['latent_shape'][0]}")
    print(f"Latent dimensions (development-only): {first['latent_shape'][1]}")
    print(f"Run 1 elapsed: {first['elapsed_seconds']:.3f}s")
    print(f"Run 2 elapsed: {second['elapsed_seconds']:.3f}s")
    print(f"Same-seed exact latent repeat: {'YES' if exact_match else 'NO'}")
    print(f"Same-seed latent allclose 1e-10: {'YES' if close_1e10 else 'NO'}")
    print(f"Same-seed max |delta|: {max_abs_difference:.8g}")
    print(f"Same-seed beta hash repeat: {'YES' if beta_hash_match else 'NO'}")
    print("Scientific performance interpretation: FORBIDDEN")
    print("iClusterBayes2 fallback activated: NO")
    print()
    print("=" * 88)
    print("Paper 5 iCluster2b CPU synthetic implementation smoke: PASS")
    print("=" * 88)
    print(f"Result: {relative_path(OUTPUT_RESULT)}")
    print(f"Environment: {relative_path(OUTPUT_ENVIRONMENT)}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("Next: 05cmp1 exact comparator implementation freeze.")
    print("=" * 88)


if __name__ == "__main__":
    main()
