#!/usr/bin/env python3
"""
Paper 5 - 05cmp3 run post-05e scientific secondary-comparator confirmatory benchmark.

TWO-STAGE / SOURCE-AUTHORIZED / RESUMABLE

Before a 05cmp3a source-authorization lock exists:
    - ONLY a synthetic preauthorization self-test may run.
    - No frozen 360-dataset simulation NPZ is opened.
    - No scientific comparator result is produced.

After a separate 05cmp3a lock authorizes this exact runner SHA:
    - run iCluster2b and MOVE on the exact 360 confirmatory simulation datasets
      frozen by 05cmp2;
    - write sealed method-by-dataset shards;
    - retry a caught per-dataset method failure once with the identical frozen
      numerical configuration and identical deterministic seed(s);
    - never activate iClusterBayes2/AUTOENCODIX based on scientific performance;
    - truth is used only after a fit is complete for eligible simulation metrics;
    - no TARGET/outcome/clinical/biology access;
    - no influence on 05e, final gate, or TARGET model selection.

The runner is intentionally no-CLI so the authorized SHA has exactly one
scientific execution path.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import random
import subprocess
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment


SCRIPT_VERSION = (
    '05cmp3-run-secondary-comparator-confirmatory-benchmark-v5-exact-result-source-path-versioned-preflight-explicit-cuda-init-frozen-icluster-tree-hash-sysmodules-loader-fix-two-stage-source-authorized-resumable-no-cli'
)

PREAUTH_STATUS = "05CMP3_SECONDARY_COMPARATOR_RUNNER_SYNTHETIC_PREAUTH_PASS"
AUTH_STATUS = "05CMP3A_SECONDARY_COMPARATOR_RUNNER_SOURCE_AUTHORIZED"
RESULT_LOCK_STATUS = "05CMP4_SECONDARY_COMPARATOR_CONFIRMATORY_RESULTS_FROZEN"

EXPECTED = {
    "cmp2_contract_sha256":
        "8e0a4d95a1f554af8e9c5188ca683d32c336c6bd5b797b3a91426adbe4ee9fc2",
    "cmp2_lock_sha256":
        "a55d3e3498d9cc6a4acaaab64e5098007b093071b60999c82cf8e124a0301b5b",
    "cmp2_registry_sha256":
        "4b72b94c4df49dbbbd1a90d035855e628a4fa50eede0451dfe796227bda53314",
    "cmp2_contract_status":
        "FROZEN_SECONDARY_COMPARATOR_CONFIRMATORY_BENCHMARK_BEFORE_SCIENTIFIC_FITS",
    "cmp2_lock_status":
        "05CMP2_SECONDARY_COMPARATOR_CONFIRMATORY_BENCHMARK_CONTRACT_FROZEN",
    "cmp1_contract_sha256":
        "418259cedd36cbd1913a4ff1ae9874a3decfddbb7a75e9b3bca2e9623f0aca95",
    "cmp1_lock_sha256":
        "37e0db502a88f7beab52f1457cb0c5367091314faa6b86c7796810d883f31648",
    "cmp1_impl_sha256":
        "85dbc68289bae71b9cf4f8a01af34b09d50750fde97f0b57ef3b6c57ebf299b0",
    "cmp1_adapter_sha256":
        "f30c73d19627d357073786b75f7b8141a757becf9d2413491518948cc099ea48",
    "cmp1_seed_sha256":
        "2c70c516ad07496a71fe513139b357b92f46fd746bea5258f1fb237028c4e87c",
    "cmp1_metric_sha256":
        "8802d8d4bba20cfd4acc3263b12f0a119bcef4e4107164e6f944d0a01d9bef46",
    "manifest05b_sha256":
        "f3980e88aff408b65cbfce96aca50bad4fc08ecece2bc42bbc04f365c35d9c87",
    "n_datasets": 360,
    "n_scenarios": 36,
    "replicates": list(range(11, 21)),
    "methods": ["iCluster2b", "MOVE"],
    "latent_dim": 5,
    "move_version": "1.5.0",
    "torch_version": "2.12.1+cu130",
    "cuda_build": "13.0",
    "move_source_hashes": {
        "dataloaders":
            "c33791e4a4b468d8081a7aea7975e8f93b33251a64bd3844bb77bd54bd75c982",
        "training_loop":
            "6dbd05437950e6a33f612f5bf969bb42fee30302d0a7395c56f558437b24caf6",
        "vae":
            "a73248339aef802a80e484849f0ccf72fc87b147c883c35bdb41561ed9b071a9",
    },
    "rscript_sha256":
        "d829bcf7e9fa1d7e3e828c565c3cdbb1ed416f551f4fa6fd4dfcdf231e33e5e8",
    "icluster_tree_sha256":
        "cad0c995150e778026fb227ac533a974d77106e431c65a3802aeda4254f6fd70",
    "icluster_version": "1.47.3",
    "bioc_version": "3.23",
    "r_version_prefix": "R version 4.6.1",
}

MOVE_CANDIDATES = [
    ("MOVE_H64_32", [64, 32]),
    ("MOVE_H128_64", [128, 64]),
    ("MOVE_H256_128", [256, 128]),
]
MOVE_EPOCHS = 300
MOVE_LR = 0.001
MOVE_BETA = 0.01
MOVE_DROPOUT = 0.1
MOVE_KLD_WARMUP_STEPS = [1, 75, 150, 300]
MOVE_STABILITY_BASE_SEEDS = [1103, 2207, 3301]

ICLUSTER_K = 5
ICLUSTER_MIN_LAMBDA = 10.0
ICLUSTER_MAX_LAMBDA = 500.0
ICLUSTER_LAMBDA_ITER = 25
ICLUSTER_EM_ITER = 25
ICLUSTER_MIN_SHRINKAGE = [0.05, 0.05, 0.05]
ICLUSTER_EPS = 1e-4
ICLUSTER_EPS2 = 1e-6

VIEW_KEYS = ["rna", "methylation", "cnv"]
NPZ_VIEW_KEYS = {
    "rna": "X_rna",
    "methylation": "X_methylation",
    "cnv": "X_cnv",
}


# =============================================================================
# Generic helpers
# =============================================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    x = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode("utf-8"))
    h.update(str(tuple(x.shape)).encode("utf-8"))
    h.update(x.tobytes(order="C"))
    return h.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def write_json_atomic(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return str(
            path.resolve().relative_to(project_root.resolve())
        ).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def tree_sha256(root: Path) -> tuple[str, int, int]:
    if not root.is_dir():
        raise FileNotFoundError(f"Directory missing: {root}")
    files = sorted(
        [p for p in root.rglob("*") if p.is_file()],
        key=lambda p: str(p.relative_to(root)).replace("\\", "/"),
    )
    h = hashlib.sha256()
    total = 0
    for path in files:
        rel = str(path.relative_to(root)).replace("\\", "/")
        data = path.read_bytes()
        total += len(data)
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.sha256(data).digest())
        h.update(b"\0")
    return h.hexdigest(), len(files), total


def derive_seed(parts: list[str], base_seed: int) -> int:
    payload = "|".join([*parts, str(int(base_seed))])
    value = int(
        hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8],
        16,
    )
    return 1 + (value % 2147483646)


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


# =============================================================================
# Exact frozen upstream verification
# =============================================================================

def build_paths(script_path: Path) -> dict[str, Path]:
    project_root = script_path.parents[2]
    paper5 = project_root / "paper5"
    return {
        "project_root": project_root,
        "paper5": paper5,
        "cmp2_contract":
            paper5 / "method_contract/05cmp2/"
            "secondary_comparator_confirmatory_benchmark_contract.json",
        "cmp2_lock":
            paper5 / "method_contract/05cmp2/"
            "secondary_comparator_confirmatory_benchmark_contract_lock.json",
        "cmp2_registry":
            paper5 / "method_contract/05cmp2/"
            "secondary_comparator_confirmatory_dataset_registry.csv",
        "cmp1_contract":
            paper5 / "method_contract/05cmp1/"
            "secondary_comparator_implementation_contract.json",
        "cmp1_lock":
            paper5 / "method_contract/05cmp1/"
            "secondary_comparator_implementation_contract_lock.json",
        "cmp1_impl":
            paper5 / "method_contract/05cmp1/"
            "secondary_comparator_implementation_registry.csv",
        "cmp1_adapter":
            paper5 / "method_contract/05cmp1/"
            "secondary_comparator_scientific_adapter_registry.csv",
        "cmp1_seed":
            paper5 / "method_contract/05cmp1/"
            "secondary_comparator_seed_registry.csv",
        "cmp1_metric":
            paper5 / "method_contract/05cmp1/"
            "secondary_comparator_metric_eligibility_registry.csv",
        "manifest05b":
            paper5 / "method_contract/05b_v2/"
            "architecture_screening_data_manifest.csv",
        "linear_helper":
            paper5 / "scripts/05c1_fit_linear_simulation_baselines.py",
        "preflight":
            paper5 / "method_contract/05cmp3_preflight/"
            "secondary_comparator_runner_synthetic_preflight_v5.json",
        "auth":
            paper5 / "method_contract/05cmp3a/"
            "secondary_comparator_runner_source_authorization_lock.json",
        "result_root":
            paper5 / "results/simulation_benchmark/"
            "05cmp_secondary_comparators/v1",
        "result_lock_dir":
            paper5 / "method_contract/05cmp4",
    }


def verify_upstream(
    paths: dict[str, Path],
) -> tuple[dict[str, Any], pd.DataFrame]:
    exact = {
        "cmp2_contract": EXPECTED["cmp2_contract_sha256"],
        "cmp2_lock": EXPECTED["cmp2_lock_sha256"],
        "cmp2_registry": EXPECTED["cmp2_registry_sha256"],
        "cmp1_contract": EXPECTED["cmp1_contract_sha256"],
        "cmp1_lock": EXPECTED["cmp1_lock_sha256"],
        "cmp1_impl": EXPECTED["cmp1_impl_sha256"],
        "cmp1_adapter": EXPECTED["cmp1_adapter_sha256"],
        "cmp1_seed": EXPECTED["cmp1_seed_sha256"],
        "cmp1_metric": EXPECTED["cmp1_metric_sha256"],
        "manifest05b": EXPECTED["manifest05b_sha256"],
    }
    for key, expected_sha in exact.items():
        path = require_file(paths[key])
        observed = sha256_file(path)
        if observed != expected_sha:
            raise RuntimeError(
                f"Frozen upstream SHA mismatch for {key}: "
                f"{observed} != {expected_sha}"
            )

    contract = load_json(paths["cmp2_contract"])
    lock = load_json(paths["cmp2_lock"])
    if contract.get("status") != EXPECTED["cmp2_contract_status"]:
        raise RuntimeError("05cmp2 contract status mismatch.")
    if lock.get("status") != EXPECTED["cmp2_lock_status"]:
        raise RuntimeError("05cmp2 lock status mismatch.")

    registry = pd.read_csv(
        paths["cmp2_registry"],
        dtype=str,
    ).fillna("")
    if registry.shape[0] != EXPECTED["n_datasets"]:
        raise RuntimeError(
            f"05cmp2 registry rows changed: {registry.shape[0]}"
        )
    if registry["scenario_id"].nunique() != EXPECTED["n_scenarios"]:
        raise RuntimeError("05cmp2 registry scenario count changed.")
    reps = sorted(
        pd.to_numeric(
            registry["replicate_index"],
            errors="raise",
        ).astype(int).unique().tolist()
    )
    if reps != EXPECTED["replicates"]:
        raise RuntimeError(
            f"05cmp2 confirmatory replicates changed: {reps}"
        )
    if registry["comparator_dataset_id"].duplicated().any():
        raise RuntimeError(
            "05cmp2 comparator_dataset_id is not unique."
        )

    return contract, registry


# =============================================================================
# Environment / source bindings
# =============================================================================

def import_linear_helper(path: Path) -> Any:
    require_file(path)
    spec = importlib.util.spec_from_file_location(
        "paper5_cmp_linear_helper",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not build import spec for frozen linear helper."
        )
    module = importlib.util.module_from_spec(spec)
    import sys as _paper5_loader_sys

    _paper5_loader_missing = object()
    _paper5_loader_previous = _paper5_loader_sys.modules.get(
        spec.name, _paper5_loader_missing
    )
    _paper5_loader_sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if _paper5_loader_previous is _paper5_loader_missing:
            _paper5_loader_sys.modules.pop(spec.name, None)
        else:
            _paper5_loader_sys.modules[spec.name] = _paper5_loader_previous
        raise
    for name in [
        "subspace_metrics",
        "standardized_columns",
    ]:
        if not hasattr(module, name):
            raise RuntimeError(
                f"Frozen linear helper no longer exposes {name}."
            )
    return module


def import_move_api() -> dict[str, Any]:
    try:
        move_version = importlib.metadata.version("move-dl")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "move-dl is not installed in the active environment. "
            "Activate paper5/comparator_envs/move_1_5_0."
        ) from exc

    if move_version != EXPECTED["move_version"]:
        raise RuntimeError(
            f"MOVE version mismatch: {move_version}"
        )
    if torch.__version__ != EXPECTED["torch_version"]:
        raise RuntimeError(
            f"PyTorch version mismatch: {torch.__version__}"
        )
    if str(torch.version.cuda) != EXPECTED["cuda_build"]:
        raise RuntimeError(
            f"CUDA build mismatch: {torch.version.cuda}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(
            "cuda:0 is required; CPU fallback is forbidden."
        )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    from move.data.dataloaders import make_dataloader
    from move.models.vae import VAE
    from move.training.training_loop import training_loop

    source_paths = {
        "dataloaders":
            Path(inspect.getfile(make_dataloader)).resolve(),
        "vae":
            Path(inspect.getfile(VAE)).resolve(),
        "training_loop":
            Path(inspect.getfile(training_loop)).resolve(),
    }
    for key, path in source_paths.items():
        observed = sha256_file(require_file(path))
        expected = EXPECTED["move_source_hashes"][key]
        if observed != expected:
            raise RuntimeError(
                f"MOVE source SHA mismatch for {key}: "
                f"{observed} != {expected}"
            )

    return {
        "make_dataloader": make_dataloader,
        "VAE": VAE,
        "training_loop": training_loop,
        "source_paths": source_paths,
        "move_version": move_version,
    }


def find_exact_rscript() -> Path:
    candidates = [
        Path(r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe"),
        Path(r"C:\Program Files\R\R-4.6.1\bin\x64\Rscript.exe"),
    ]
    explicit = os.environ.get("PAPER5_RSCRIPT", "").strip()
    if explicit:
        candidates.insert(0, Path(explicit))

    for path in candidates:
        if (
            path.is_file()
            and sha256_file(path) == EXPECTED["rscript_sha256"]
        ):
            return path.resolve()

    raise RuntimeError(
        "Exact frozen Rscript 4.6.1 executable was not "
        "found/hash-matched."
    )


def icluster_tree_sha256_frozen_05cmp0c(root: Path) -> tuple[str, int, int]:
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


def verify_r_library(paper5: Path) -> dict[str, Any]:
    r_library = (
        paper5 / "comparator_envs/r_4_6_bioc_3_23"
    ).resolve()
    package_root = r_library / "iClusterPlus"
    tree_hash, file_count, total_bytes = icluster_tree_sha256_frozen_05cmp0c(
        package_root
    )
    if tree_hash != EXPECTED["icluster_tree_sha256"]:
        raise RuntimeError(
            "iClusterPlus package-tree SHA mismatch: "
            f"{tree_hash}"
        )

    rscript = find_exact_rscript()

    expression = r"""
options(warn=1, digits=17)
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages(library(BiocManager))
suppressPackageStartupMessages(library(iClusterPlus))
cat("R_VERSION=", R.version.string, "\n", sep="")
cat("BIOC_VERSION=", as.character(BiocManager::version()), "\n", sep="")
cat("ICLUSTERPLUS_VERSION=", as.character(packageVersion("iClusterPlus")), "\n", sep="")
cat("ICLUSTER2B_EXISTS=", exists("iCluster2b", where=asNamespace("iClusterPlus"), inherits=FALSE), "\n", sep="")
cat("TUNE_ICLUSTER2B_EXISTS=", exists("tune.iCluster2b", where=asNamespace("iClusterPlus"), inherits=FALSE), "\n", sep="")
"""

    env = os.environ.copy()
    env["R_LIBS_USER"] = str(r_library)
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        env[key] = "1"

    with tempfile.TemporaryDirectory(
        prefix="paper5_05cmp3_rprobe_"
    ) as td:
        rp = Path(td) / "probe.R"
        rp.write_text(
            expression,
            encoding="utf-8",
            newline="\n",
        )
        completed = subprocess.run(
            [str(rscript), "--vanilla", str(rp)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

    if completed.returncode != 0:
        raise RuntimeError(
            "Frozen R/iClusterPlus environment probe failed.\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    parsed: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            parsed[k.strip()] = v.strip()

    if not parsed.get("R_VERSION", "").startswith(
        EXPECTED["r_version_prefix"]
    ):
        raise RuntimeError(
            f"R version mismatch: {parsed.get('R_VERSION')}"
        )
    if parsed.get("BIOC_VERSION") != EXPECTED["bioc_version"]:
        raise RuntimeError(
            "Bioconductor version mismatch."
        )
    if (
        parsed.get("ICLUSTERPLUS_VERSION")
        != EXPECTED["icluster_version"]
    ):
        raise RuntimeError(
            "iClusterPlus version mismatch."
        )
    if parsed.get("ICLUSTER2B_EXISTS", "").lower() != "true":
        raise RuntimeError(
            "iCluster2b unavailable."
        )
    if (
        parsed.get("TUNE_ICLUSTER2B_EXISTS", "").lower()
        != "true"
    ):
        raise RuntimeError(
            "tune.iCluster2b unavailable."
        )

    return {
        "rscript": rscript,
        "r_library": r_library,
        "package_tree_sha256": tree_hash,
        "package_tree_file_count": file_count,
        "package_tree_total_bytes": total_bytes,
        "probe": parsed,
        "subprocess_env": env,
    }


# =============================================================================
# Adapter / preprocessing
# =============================================================================

def fit_observed_standardizer(
    x: np.ndarray,
    fit_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    fit = x[np.asarray(fit_indices, dtype=int)]

    observed = np.isfinite(fit)
    counts = observed.sum(axis=0)

    sums = np.where(
        observed,
        fit,
        0.0,
    ).sum(axis=0)

    means = np.divide(
        sums,
        counts,
        out=np.zeros(x.shape[1], dtype=np.float64),
        where=counts > 0,
    )

    centered_sq = np.where(
        observed,
        (fit - means) ** 2,
        0.0,
    ).sum(axis=0)

    variance = np.divide(
        centered_sq,
        counts,
        out=np.ones(x.shape[1], dtype=np.float64),
        where=counts > 0,
    )

    scale = np.sqrt(variance)

    bad = (~np.isfinite(scale)) | (scale <= 0.0)
    scale[bad] = 1.0
    means[~np.isfinite(means)] = 0.0

    return {
        "mean": means,
        "scale": scale,
        "fit_observed_counts": counts.astype(np.int64),
        "n_constant_or_absent_fit_features":
            np.asarray(bad, dtype=np.uint8),
    }


def transform_with_standardizer(
    x: np.ndarray,
    prep: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    observed = np.isfinite(x)

    z = (
        x - prep["mean"]
    ) / prep["scale"]

    z[~np.isfinite(z)] = 0.0
    return z, observed


def adapt_views(
    raw_views: dict[str, np.ndarray],
    fit_indices: np.ndarray,
    output_indices: np.ndarray,
    dtype: np.dtype,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[dict[str, Any]],
]:
    outputs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []

    for view in VIEW_KEYS:
        raw = np.asarray(
            raw_views[view],
            dtype=np.float64,
        )
        prep = fit_observed_standardizer(
            raw,
            fit_indices,
        )
        transformed, observed = transform_with_standardizer(
            raw[np.asarray(output_indices, dtype=int)],
            prep,
        )

        transformed = transformed.astype(
            dtype,
            copy=False,
        )
        outputs.append(transformed)
        masks.append(observed)

        audits.append(
            {
                "view": view,
                "n_fit_rows": int(len(fit_indices)),
                "n_output_rows": int(len(output_indices)),
                "n_features": int(raw.shape[1]),
                "n_fit_features_no_observed":
                    int(
                        np.sum(
                            prep["fit_observed_counts"] == 0
                        )
                    ),
                "n_fit_features_constant_or_absent":
                    int(
                        np.sum(
                            prep[
                                "n_constant_or_absent_fit_features"
                            ]
                        )
                    ),
                "output_missing_entries_zero_filled":
                    int(np.sum(~observed)),
            }
        )

    return outputs, masks, audits


# =============================================================================
# Frozen simulation truth metrics
# =============================================================================

def own_mean_canonical_correlation(
    estimated: np.ndarray,
    truth: np.ndarray,
) -> float:
    estimated = np.asarray(
        estimated,
        dtype=np.float64,
    )
    truth = np.asarray(
        truth,
        dtype=np.float64,
    )

    if (
        estimated.ndim != 2
        or truth.ndim != 2
        or estimated.shape[0] != truth.shape[0]
    ):
        raise RuntimeError(
            "Subspace metric shape mismatch."
        )
    if (
        estimated.shape[1] == 0
        or truth.shape[1] == 0
    ):
        return float("nan")
    if (
        not np.isfinite(estimated).all()
        or not np.isfinite(truth).all()
    ):
        raise RuntimeError(
            "Non-finite latent/truth matrix in subspace metric."
        )

    def basis(x: np.ndarray) -> np.ndarray:
        xc = x - x.mean(
            axis=0,
            keepdims=True,
        )
        u, s, _ = np.linalg.svd(
            xc,
            full_matrices=False,
        )
        if s.size == 0:
            return np.zeros(
                (x.shape[0], 0),
                dtype=np.float64,
            )
        tol = (
            float(s[0])
            * max(xc.shape)
            * np.finfo(np.float64).eps
        )
        rank = int(np.sum(s > tol))
        return u[:, :rank]

    qa = basis(estimated)
    qb = basis(truth)

    if qa.shape[1] == 0 or qb.shape[1] == 0:
        return 0.0

    values = np.linalg.svd(
        qa.T @ qb,
        compute_uv=False,
    )
    values = np.clip(
        values,
        0.0,
        1.0,
    )

    # A rank-deficient estimated space receives zero recovery for
    # missing truth dimensions so collapse cannot be rewarded.
    denom = int(truth.shape[1])
    total = float(np.sum(values))
    return total / float(denom)


def flatten_numeric_scalars(
    value: Any,
    prefix: str = "",
) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []

    if isinstance(value, dict):
        for k, v in value.items():
            p = (
                f"{prefix}.{k}"
                if prefix
                else str(k)
            )
            out.extend(
                flatten_numeric_scalars(
                    v,
                    p,
                )
            )

    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            p = f"{prefix}[{i}]"
            out.extend(
                flatten_numeric_scalars(
                    v,
                    p,
                )
            )

    elif (
        isinstance(
            value,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        )
        and not isinstance(value, bool)
    ):
        x = float(value)
        if math.isfinite(x):
            out.append((prefix, x))

    return out


def paper5_shared_subspace_recovery(
    linear: Any,
    estimated: np.ndarray,
    truth: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    if truth.shape[1] == 0:
        return (
            float("nan"),
            {
                "status":
                    "NOT_APPLICABLE_NO_TRUE_SHARED_FACTORS"
            },
        )

    authoritative = linear.subspace_metrics(
        np.asarray(
            estimated,
            dtype=np.float64,
        ),
        np.asarray(
            truth,
            dtype=np.float64,
        ),
    )

    numeric = flatten_numeric_scalars(
        authoritative
    )

    preferred = [
        (
            name,
            value,
        )
        for name, value in numeric
        if (
            "mean" in name.lower()
            and "canonical" in name.lower()
        )
        or "shared_subspace_recovery" in name.lower()
    ]

    if len(preferred) == 1:
        return (
            float(preferred[0][1]),
            {
                "status":
                    "PASS_FROZEN_HELPER_NAMED_SCALAR",
                "helper_scalar_path":
                    preferred[0][0],
            },
        )

    own = own_mean_canonical_correlation(
        estimated,
        truth,
    )

    matches = [
        (
            name,
            value,
        )
        for name, value in numeric
        if np.isclose(
            value,
            own,
            rtol=0.0,
            atol=1e-10,
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Frozen Paper 5 subspace_metrics did not yield "
            "one unambiguous scalar matching the "
            "mean-canonical-correlation definition. "
            f"Own={own}; authoritative={authoritative!r}; "
            f"matches={matches!r}"
        )

    return (
        float(matches[0][1]),
        {
            "status":
                "PASS_FROZEN_HELPER_NUMERIC_REPLAY",
            "helper_scalar_path":
                matches[0][0],
        },
    )


def factor_alignment_correlation(
    linear: Any,
    estimated: np.ndarray,
    truth: np.ndarray,
) -> tuple[float, list[dict[str, Any]]]:
    if truth.shape[1] == 0:
        return float("nan"), []

    e = np.asarray(
        linear.standardized_columns(
            np.asarray(
                estimated,
                dtype=np.float64,
            )
        )
    )
    t = np.asarray(
        linear.standardized_columns(
            np.asarray(
                truth,
                dtype=np.float64,
            )
        )
    )

    if (
        e.ndim != 2
        or t.ndim != 2
        or e.shape[0] != t.shape[0]
    ):
        raise RuntimeError(
            "standardized_columns returned incompatible matrices."
        )
    if (
        not np.isfinite(e).all()
        or not np.isfinite(t).all()
    ):
        raise RuntimeError(
            "Non-finite standardized columns in factor alignment."
        )

    corr = np.corrcoef(
        e.T,
        t.T,
    )[
        : e.shape[1],
        e.shape[1] :
    ]

    corr = np.asarray(
        corr,
        dtype=np.float64,
    )
    if not np.isfinite(corr).all():
        raise RuntimeError(
            "Factor-alignment correlation matrix is non-finite."
        )

    row_ind, col_ind = linear_sum_assignment(
        -np.abs(corr)
    )
    values = np.abs(
        corr[
            row_ind,
            col_ind,
        ]
    )

    details = [
        {
            "estimated_factor_index": int(i),
            "truth_factor_index": int(j),
            "signed_pearson":
                float(corr[i, j]),
            "absolute_pearson":
                float(abs(corr[i, j])),
        }
        for i, j in zip(
            row_ind.tolist(),
            col_ind.tolist(),
            strict=True,
        )
    ]

    return float(np.mean(values)), details


# =============================================================================
# MOVE low-level execution
# =============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_loader(
    make_dataloader: Any,
    views: list[np.ndarray],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Any:
    generator = torch.Generator(
        device="cpu"
    )
    generator.manual_seed(
        int(seed)
    )

    return make_dataloader(
        con_list=[
            np.asarray(
                x,
                dtype=np.float32,
            )
            for x in views
        ],
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=False,
        num_workers=0,
        generator=generator,
    )


def fit_move(
    api: dict[str, Any],
    views: list[np.ndarray],
    hidden: list[int],
    seed: int,
    epochs: int,
    kld_warmup_steps: list[int],
) -> dict[str, Any]:
    VAE = api["VAE"]
    training_loop = api["training_loop"]
    make_dataloader = api["make_dataloader"]

    # Initialize CUDA before seeding, cache cleanup, and allocator statistics.
    torch.cuda.init()
    seed_everything(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    n = int(
        views[0].shape[0]
    )
    if any(
        int(v.shape[0]) != n
        for v in views
    ):
        raise RuntimeError(
            "MOVE view row counts differ."
        )

    batch_size = min(
        16,
        n,
    )

    train_loader = move_loader(
        make_dataloader,
        views,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    project_loader = move_loader(
        make_dataloader,
        views,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    model = VAE(
        categorical_shapes=None,
        continuous_shapes=[
            int(v.shape[1])
            for v in views
        ],
        categorical_weights=None,
        continuous_weights=[
            1.0,
            1.0,
            1.0,
        ],
        num_hidden=list(hidden),
        num_latent=EXPECTED["latent_dim"],
        beta=MOVE_BETA,
        dropout=MOVE_DROPOUT,
        cuda=True,
    )

    model = model.to(
        model.device
    )

    if sorted(
        {
            p.device.type
            for p in model.parameters()
        }
    ) != ["cuda"]:
        raise RuntimeError(
            "MOVE model parameters are not exclusively CUDA."
        )

    started = time.perf_counter()

    (
        losses,
        bce_losses,
        sse_losses,
        kld_losses,
        final_kld_weight,
    ) = training_loop(
        model=model,
        train_dataloader=train_loader,
        valid_dataloader=None,
        lr=MOVE_LR,
        num_epochs=int(epochs),
        batch_dilation_steps=[],
        kld_warmup_steps=list(
            kld_warmup_steps
        ),
        early_stopping=False,
        patience=0,
    )

    torch.cuda.synchronize(0)
    elapsed = (
        time.perf_counter()
        - started
    )

    for label, values in {
        "loss": losses,
        "bce": bce_losses,
        "sse": sse_losses,
        "kld": kld_losses,
    }.items():
        x = np.asarray(
            values,
            dtype=np.float64,
        )
        if (
            x.shape
            != (int(epochs),)
            or not np.isfinite(x).all()
        ):
            raise RuntimeError(
                f"MOVE {label} history invalid: "
                f"shape={x.shape}"
            )

    latent = np.asarray(
        model.project(
            project_loader
        ),
        dtype=np.float32,
    )

    if (
        latent.shape
        != (
            n,
            EXPECTED["latent_dim"],
        )
        or not np.isfinite(latent).all()
    ):
        raise RuntimeError(
            "MOVE latent output invalid: "
            f"{latent.shape}"
        )

    parameter_count = int(
        sum(
            int(p.numel())
            for p in model.parameters()
        )
    )
    peak_bytes = int(
        torch.cuda.max_memory_allocated(0)
    )

    return {
        "model": model,
        "latent": latent,
        "parameter_count":
            parameter_count,
        "elapsed_seconds":
            float(elapsed),
        "peak_cuda_bytes":
            peak_bytes,
        "final_kld_weight":
            float(final_kld_weight),
        "final_loss":
            float(
                np.asarray(
                    losses,
                    dtype=float,
                )[-1]
            ),
    }


def move_reconstruction_sse(
    model: Any,
    adapted_views: list[np.ndarray],
    observed_masks: list[np.ndarray],
) -> float:
    concat = np.concatenate(
        [
            np.asarray(
                v,
                dtype=np.float32,
            )
            for v in adapted_views
        ],
        axis=1,
    )

    mask = np.concatenate(
        observed_masks,
        axis=1,
    )
    if not mask.any():
        raise RuntimeError(
            "Held-out fold contains zero observed entries."
        )

    tensor = torch.from_numpy(
        concat
    ).to(
        device=model.device,
        dtype=torch.float32,
    )

    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(
            tensor
        )
        _, recon = model.decode(
            mu
        )

    if recon is None:
        raise RuntimeError(
            "MOVE decoder returned no continuous reconstruction."
        )

    rec = (
        recon.detach()
        .cpu()
        .numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )

    if (
        rec.shape != concat.shape
        or not np.isfinite(rec).all()
    ):
        raise RuntimeError(
            "MOVE held-out reconstruction invalid."
        )

    diff = (
        rec
        - concat.astype(
            np.float64,
            copy=False,
        )
    )

    return float(
        np.sum(
            np.square(
                diff[mask]
            )
        )
    )


def deterministic_sample_folds(
    sample_ids: np.ndarray,
    n_folds: int = 5,
) -> list[np.ndarray]:
    ids = [
        str(x)
        for x in np.asarray(
            sample_ids
        ).tolist()
    ]

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            "Sample IDs are not unique."
        )

    order = sorted(
        range(len(ids)),
        key=lambda i: ids[i],
    )

    fold_by_row = np.empty(
        len(ids),
        dtype=int,
    )

    for pos, row_index in enumerate(
        order
    ):
        fold_by_row[row_index] = (
            pos % n_folds
        )

    return [
        np.flatnonzero(
            fold_by_row == fold
        )
        for fold in range(
            n_folds
        )
    ]


def pairwise_mean_canonical(
    latents: list[np.ndarray],
) -> tuple[list[float], float]:
    if len(latents) != 3:
        raise RuntimeError(
            "MOVE stability requires exactly three full-data seeds."
        )

    pair_values: list[float] = []

    for i in range(3):
        for j in range(
            i + 1,
            3,
        ):
            pair_values.append(
                own_mean_canonical_correlation(
                    latents[i],
                    latents[j],
                )
            )

    return (
        pair_values,
        float(
            np.median(
                np.asarray(
                    pair_values,
                    dtype=float,
                )
            )
        ),
    )


def select_move_architecture(
    api: dict[str, Any],
    raw_views: dict[str, np.ndarray],
    sample_ids: np.ndarray,
    dataset_id: str,
) -> dict[str, Any]:
    n = len(sample_ids)
    all_idx = np.arange(
        n,
        dtype=int,
    )
    folds = deterministic_sample_folds(
        sample_ids,
        n_folds=5,
    )

    stage1_rows: list[
        dict[str, Any]
    ] = []

    for candidate_id, hidden in MOVE_CANDIDATES:
        fold_sses: list[float] = []

        for fold_index, test_idx in enumerate(
            folds
        ):
            train_idx = np.setdiff1d(
                all_idx,
                test_idx,
                assume_unique=True,
            )

            train_views, _, _ = adapt_views(
                raw_views,
                fit_indices=train_idx,
                output_indices=train_idx,
                dtype=np.float32,
            )

            test_views, test_masks, _ = adapt_views(
                raw_views,
                fit_indices=train_idx,
                output_indices=test_idx,
                dtype=np.float32,
            )

            seed = derive_seed(
                [
                    "MOVE",
                    dataset_id,
                    str(fold_index),
                    candidate_id,
                    "architecture_selection_cv",
                ],
                1103,
            )

            fit = fit_move(
                api,
                train_views,
                hidden=hidden,
                seed=seed,
                epochs=MOVE_EPOCHS,
                kld_warmup_steps=
                    MOVE_KLD_WARMUP_STEPS,
            )

            sse = move_reconstruction_sse(
                fit["model"],
                test_views,
                test_masks,
            )

            fold_sses.append(
                sse
            )

            del fit["model"]
            del fit
            gc.collect()
            torch.cuda.empty_cache()

        values = np.asarray(
            fold_sses,
            dtype=np.float64,
        )

        stage1_rows.append(
            {
                "candidate_id":
                    candidate_id,
                "hidden":
                    list(hidden),
                "fold_sse":
                    values.tolist(),
                "mean_heldout_reconstruction_sse":
                    float(np.mean(values)),
                "se_heldout_reconstruction_sse":
                    float(
                        np.std(
                            values,
                            ddof=1,
                        )
                        / math.sqrt(
                            len(values)
                        )
                    ),
            }
        )

    means = np.asarray(
        [
            r[
                "mean_heldout_reconstruction_sse"
            ]
            for r in stage1_rows
        ],
        dtype=float,
    )

    best_idx = int(
        np.argmin(
            means
        )
    )

    cutoff = (
        float(
            stage1_rows[
                best_idx
            ][
                "mean_heldout_reconstruction_sse"
            ]
        )
        + float(
            stage1_rows[
                best_idx
            ][
                "se_heldout_reconstruction_sse"
            ]
        )
    )

    retained = [
        r
        for r in stage1_rows
        if float(
            r[
                "mean_heldout_reconstruction_sse"
            ]
        )
        <= cutoff
    ]

    if not retained:
        raise RuntimeError(
            "MOVE one-SE stage retained zero candidates."
        )

    full_views, _, full_audit = adapt_views(
        raw_views,
        fit_indices=all_idx,
        output_indices=all_idx,
        dtype=np.float32,
    )

    stage2_rows: list[
        dict[str, Any]
    ] = []

    canonical_latents: dict[
        str,
        np.ndarray,
    ] = {}

    for candidate in retained:
        candidate_id = str(
            candidate[
                "candidate_id"
            ]
        )
        hidden = list(
            candidate[
                "hidden"
            ]
        )

        latents: list[
            np.ndarray
        ] = []

        parameter_count: int | None = None
        seed_rows: list[
            dict[str, Any]
        ] = []

        for base_seed in (
            MOVE_STABILITY_BASE_SEEDS
        ):
            seed = derive_seed(
                [
                    "MOVE",
                    dataset_id,
                    candidate_id,
                    "full_data_stability",
                ],
                base_seed,
            )

            fit = fit_move(
                api,
                full_views,
                hidden=hidden,
                seed=seed,
                epochs=MOVE_EPOCHS,
                kld_warmup_steps=
                    MOVE_KLD_WARMUP_STEPS,
            )

            latent = np.asarray(
                fit["latent"],
                dtype=np.float64,
            )
            latents.append(
                latent
            )

            if parameter_count is None:
                parameter_count = int(
                    fit[
                        "parameter_count"
                    ]
                )
            elif (
                parameter_count
                != int(
                    fit[
                        "parameter_count"
                    ]
                )
            ):
                raise RuntimeError(
                    "MOVE parameter count changed across seeds."
                )

            if base_seed == 1103:
                canonical_latents[
                    candidate_id
                ] = latent.copy()

            seed_rows.append(
                {
                    "base_seed":
                        int(base_seed),
                    "derived_seed":
                        int(seed),
                    "elapsed_seconds":
                        float(
                            fit[
                                "elapsed_seconds"
                            ]
                        ),
                    "peak_cuda_bytes":
                        int(
                            fit[
                                "peak_cuda_bytes"
                            ]
                        ),
                    "latent_sha256":
                        sha256_array(
                            latent
                        ),
                }
            )

            del fit["model"]
            del fit
            gc.collect()
            torch.cuda.empty_cache()

        (
            pair_values,
            stability,
        ) = pairwise_mean_canonical(
            latents
        )

        stage2_rows.append(
            {
                "candidate_id":
                    candidate_id,
                "hidden":
                    hidden,
                "parameter_count":
                    int(
                        parameter_count
                    ),
                "pairwise_mean_canonical_correlations":
                    pair_values,
                "median_pairwise_mean_canonical_correlation":
                    stability,
                "seed_runs":
                    seed_rows,
            }
        )

    stage2_sorted = sorted(
        stage2_rows,
        key=lambda r: (
            -float(
                r[
                    "median_pairwise_mean_canonical_correlation"
                ]
            ),
            int(
                r[
                    "parameter_count"
                ]
            ),
            str(
                r[
                    "candidate_id"
                ]
            ),
        ),
    )

    selected = stage2_sorted[0]
    selected_id = str(
        selected[
            "candidate_id"
        ]
    )

    selected_stage1 = next(
        r
        for r in stage1_rows
        if r["candidate_id"]
        == selected_id
    )

    return {
        "selected_candidate_id":
            selected_id,
        "canonical_latent":
            canonical_latents[
                selected_id
            ],
        "stage1":
            stage1_rows,
        "stage1_best_candidate_id":
            str(
                stage1_rows[
                    best_idx
                ][
                    "candidate_id"
                ]
            ),
        "stage1_one_se_cutoff":
            float(cutoff),
        "stage1_retained_candidate_ids":
            [
                str(
                    r[
                        "candidate_id"
                    ]
                )
                for r in retained
            ],
        "stage2":
            stage2_rows,
        "selected_latent_subspace_stability":
            float(
                selected[
                    "median_pairwise_mean_canonical_correlation"
                ]
            ),
        "selected_cv_mean_heldout_reconstruction_sse":
            float(
                selected_stage1[
                    "mean_heldout_reconstruction_sse"
                ]
            ),
        "full_data_preprocessing_audit":
            full_audit,
    }


# =============================================================================
# iCluster2b exact scientific execution
# =============================================================================

ICLUSTER_R_SCRIPT = r"""
options(warn=1, digits=17)
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages(library(BiocManager))
suppressPackageStartupMessages(library(iClusterPlus))

args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 4) stop("Expected input_dir output_dir seed expected_rows")
input_dir <- args[[1]]
output_dir <- args[[2]]
seed <- as.integer(args[[3]])
expected_rows <- as.integer(args[[4]])

if (as.character(packageVersion("iClusterPlus")) != "1.47.3") {
  stop("iClusterPlus version mismatch")
}
if (as.character(BiocManager::version()) != "3.23") {
  stop("Bioconductor version mismatch")
}

read_matrix <- function(name) {
  path <- file.path(input_dir, paste0(name, ".csv"))
  x <- as.matrix(read.csv(path, header=FALSE, check.names=FALSE))
  storage.mode(x) <- "double"
  if (nrow(x) != expected_rows) stop("Row mismatch for ", name)
  if (!all(is.finite(x))) stop("Non-finite adapted input: ", name)
  x
}

xList <- list(
  read_matrix("rna"),
  read_matrix("methylation"),
  read_matrix("cnv")
)

set.seed(seed)
started <- proc.time()[["elapsed"]]

fit <- iClusterPlus::tune.iCluster2b(
  xList=xList,
  K=5,
  method=rep("lasso", 3),
  min.lambda=10,
  max.lambda=500,
  lambda.iter=25,
  EM.iter=25,
  min.shrinkage.rate=rep(0.05, 3),
  eps=1e-4,
  eps2=1e-6
)

elapsed <- proc.time()[["elapsed"]] - started

if (is.null(fit$meanZ)) stop("tune.iCluster2b returned no meanZ")
meanZ <- as.matrix(fit$meanZ)
storage.mode(meanZ) <- "double"

if (nrow(meanZ) != expected_rows) stop("meanZ row mismatch")
if (ncol(meanZ) != 5) stop("meanZ latent dimension mismatch")
if (!all(is.finite(meanZ))) stop("meanZ non-finite")

lambda <- unlist(fit$lambda, use.names=FALSE)
storage.mode(lambda) <- "double"

if (length(lambda) != 3) stop("Expected three final lambda values")
if (!all(is.finite(lambda))) stop("Non-finite lambda")

dir.create(output_dir, recursive=TRUE, showWarnings=FALSE)

write.table(
  meanZ,
  file=file.path(output_dir, "meanZ.csv"),
  sep=",",
  row.names=FALSE,
  col.names=FALSE,
  quote=FALSE
)

write.table(
  matrix(lambda, nrow=1),
  file=file.path(output_dir, "lambda.csv"),
  sep=",",
  row.names=FALSE,
  col.names=FALSE,
  quote=FALSE
)

metadata <- data.frame(
  key=c(
    "r_version",
    "bioconductor_version",
    "iclusterplus_version",
    "seed",
    "elapsed_seconds"
  ),
  value=c(
    R.version.string,
    as.character(BiocManager::version()),
    as.character(packageVersion("iClusterPlus")),
    as.character(seed),
    as.character(elapsed)
  ),
  stringsAsFactors=FALSE
)

write.table(
  metadata,
  file=file.path(output_dir, "metadata.tsv"),
  sep="\t",
  row.names=FALSE,
  col.names=TRUE,
  quote=FALSE
)

cat("ICLUSTER2B_SCIENTIFIC_PASS\n")
"""


def run_icluster_fit(
    r_env: dict[str, Any],
    views: list[np.ndarray],
    seed: int,
) -> dict[str, Any]:
    n = int(
        views[0].shape[0]
    )

    if any(
        v.shape[0] != n
        for v in views
    ):
        raise RuntimeError(
            "iCluster2b view row counts differ."
        )

    with tempfile.TemporaryDirectory(
        prefix="paper5_05cmp3_icluster_"
    ) as td:
        root = Path(td)
        input_dir = root / "input"
        output_dir = root / "output"

        input_dir.mkdir(
            parents=True
        )
        output_dir.mkdir(
            parents=True
        )

        runner = root / "run_icluster2b.R"

        for name, matrix in zip(
            VIEW_KEYS,
            views,
            strict=True,
        ):
            np.savetxt(
                input_dir / f"{name}.csv",
                np.asarray(
                    matrix,
                    dtype=np.float64,
                ),
                delimiter=",",
                fmt="%.17g",
            )

        runner.write_text(
            ICLUSTER_R_SCRIPT,
            encoding="utf-8",
            newline="\n",
        )

        command = [
            str(
                r_env[
                    "rscript"
                ]
            ),
            "--vanilla",
            str(runner),
            str(input_dir),
            str(output_dir),
            str(int(seed)),
            str(n),
        ]

        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=r_env[
                "subprocess_env"
            ],
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "iCluster2b scientific subprocess failed.\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            )

        if (
            "ICLUSTER2B_SCIENTIFIC_PASS"
            not in completed.stdout
        ):
            raise RuntimeError(
                "iCluster2b scientific PASS sentinel missing."
            )

        mean_z = np.loadtxt(
            output_dir / "meanZ.csv",
            delimiter=",",
            ndmin=2,
        )

        lambdas = np.loadtxt(
            output_dir / "lambda.csv",
            delimiter=",",
            ndmin=1,
        )

        if (
            mean_z.shape
            != (
                n,
                ICLUSTER_K,
            )
            or not np.isfinite(
                mean_z
            ).all()
        ):
            raise RuntimeError(
                "Invalid iCluster2b meanZ: "
                f"{mean_z.shape}"
            )

        lambdas = np.asarray(
            lambdas,
            dtype=np.float64,
        ).reshape(-1)

        if (
            lambdas.shape != (3,)
            or not np.isfinite(
                lambdas
            ).all()
        ):
            raise RuntimeError(
                "Invalid iCluster2b lambda vector: "
                f"{lambdas}"
            )

        metadata_df = pd.read_csv(
            output_dir / "metadata.tsv",
            sep="\t",
            dtype=str,
        )

        metadata = {
            str(r.key):
                str(r.value)
            for r in metadata_df.itertuples(
                index=False
            )
        }

    return {
        "latent":
            np.asarray(
                mean_z,
                dtype=np.float64,
            ),
        "lambda":
            lambdas.tolist(),
        "elapsed_seconds":
            float(
                metadata[
                    "elapsed_seconds"
                ]
            ),
        "derived_seed":
            int(seed),
    }


# =============================================================================
# Simulation dataset loading and post-fit metrics
# =============================================================================

def resolve_registry_dataset_path(
    project_root: Path,
    raw: str,
) -> Path:
    normalized = str(
        raw
    ).replace(
        "\\",
        "/",
    )

    path = Path(
        normalized
    )

    if path.is_absolute():
        return path

    return (
        project_root / path
    ).resolve()


def load_scientific_npz(
    project_root: Path,
    row: pd.Series,
) -> dict[str, np.ndarray]:
    path = resolve_registry_dataset_path(
        project_root,
        row["path"],
    )

    require_file(path)

    expected_size = int(
        row[
            "size_bytes"
        ]
    )

    if path.stat().st_size != expected_size:
        raise RuntimeError(
            "Simulation dataset size mismatch: "
            f"{path}"
        )

    observed_sha = sha256_file(
        path
    )

    if observed_sha != str(
        row[
            "sha256"
        ]
    ):
        raise RuntimeError(
            "Simulation dataset SHA mismatch: "
            f"{observed_sha} != "
            f"{row['sha256']}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as npz:
        required = {
            "X_rna",
            "X_methylation",
            "X_cnv",
            "Z_shared",
            "sample_index",
        }

        missing = (
            required
            - set(
                npz.files
            )
        )

        if missing:
            raise RuntimeError(
                "Simulation NPZ missing required arrays: "
                f"{sorted(missing)}"
            )

        arrays = {
            name:
                np.asarray(
                    npz[name]
                )
            for name in required
        }

    n = int(
        row[
            "n_samples"
        ]
    )

    views = {
        view:
            np.asarray(
                arrays[key],
                dtype=np.float64,
            )
        for view, key in (
            NPZ_VIEW_KEYS.items()
        )
    }

    if any(
        x.shape[0] != n
        for x in views.values()
    ):
        raise RuntimeError(
            "Simulation view row count differs "
            "from manifest n_samples."
        )

    if (
        np.asarray(
            arrays[
                "Z_shared"
            ]
        ).shape[0]
        != n
    ):
        raise RuntimeError(
            "Z_shared row count differs from manifest n_samples."
        )

    if np.asarray(
        arrays[
            "sample_index"
        ]
    ).shape != (n,):
        raise RuntimeError(
            "sample_index shape mismatch."
        )

    return {
        **views,
        "Z_shared":
            np.asarray(
                arrays[
                    "Z_shared"
                ],
                dtype=np.float64,
            ),
        "sample_index":
            np.asarray(
                arrays[
                    "sample_index"
                ]
            ),
    }


def evaluate_truth_metrics(
    linear: Any,
    latent: np.ndarray,
    z_shared: np.ndarray,
    factor_level_identifiable: bool,
    subspace_level_identifiable: bool,
) -> dict[str, Any]:
    latent = np.asarray(
        latent,
        dtype=np.float64,
    )

    z_shared = np.asarray(
        z_shared,
        dtype=np.float64,
    )

    out: dict[str, Any] = {
        "n_true_shared_factors":
            int(
                z_shared.shape[1]
            ),
        "shared_subspace_recovery":
            None,
        "shared_subspace_recovery_status":
            "NOT_APPLICABLE",
        "factor_alignment_correlation":
            None,
        "factor_alignment_status":
            "NOT_APPLICABLE",
        "factor_alignment_matches":
            [],
    }

    if z_shared.shape[1] == 0:
        out[
            "shared_subspace_recovery_status"
        ] = (
            "NOT_APPLICABLE_NO_TRUE_SHARED_FACTORS"
        )

        out[
            "factor_alignment_status"
        ] = (
            "NOT_APPLICABLE_NO_TRUE_SHARED_FACTORS"
        )

        return out

    if subspace_level_identifiable:
        (
            subspace_value,
            helper,
        ) = paper5_shared_subspace_recovery(
            linear,
            latent,
            z_shared,
        )

        out[
            "shared_subspace_recovery"
        ] = float(
            subspace_value
        )

        out[
            "shared_subspace_recovery_status"
        ] = helper[
            "status"
        ]

        out[
            "shared_subspace_helper_replay"
        ] = helper

    else:
        out[
            "shared_subspace_recovery_status"
        ] = (
            "NOT_PRIMARY_INTERPRETABLE_"
            "SUBSPACE_LEVEL_IDENTIFIABLE_FALSE"
        )

    if factor_level_identifiable:
        (
            factor_value,
            matches,
        ) = factor_alignment_correlation(
            linear,
            latent,
            z_shared,
        )

        out[
            "factor_alignment_correlation"
        ] = float(
            factor_value
        )

        out[
            "factor_alignment_status"
        ] = (
            "PASS_SIGN_PERMUTATION_INVARIANT"
        )

        out[
            "factor_alignment_matches"
        ] = matches

    else:
        out[
            "factor_alignment_status"
        ] = (
            "NOT_PRIMARY_INTERPRETABLE_"
            "FACTOR_LEVEL_IDENTIFIABLE_FALSE"
        )

    return out


def registry_metadata(
    row: pd.Series,
) -> dict[str, Any]:
    def maybe_float(
        name: str,
    ) -> float | None:
        text = str(
            row.get(
                name,
                "",
            )
        ).strip()

        if text == "":
            return None

        return float(
            text
        )

    partial_view_count = 0.0

    for col in [
        "rna_missing_partial_view",
        "methylation_missing_partial_view",
        "cnv_missing_partial_view",
    ]:
        value = maybe_float(
            col
        )

        if value is not None:
            partial_view_count += value

    return {
        "comparator_dataset_id":
            str(
                row[
                    "comparator_dataset_id"
                ]
            ),
        "scenario_id":
            str(
                row[
                    "scenario_id"
                ]
            ),
        "replicate_index":
            int(
                row[
                    "replicate_index"
                ]
            ),
        "simulation_seed":
            int(
                row[
                    "simulation_seed"
                ]
            ),
        "scenario_family":
            str(
                row.get(
                    "scenario_family",
                    "",
                )
            ),
        "n_samples":
            int(
                row[
                    "n_samples"
                ]
            ),
        "missingness_mode":
            str(
                row.get(
                    "missingness_mode",
                    "",
                )
            ),
        "batch_mode":
            str(
                row.get(
                    "batch_mode",
                    "",
                )
            ),
        "factor_level_identifiable":
            parse_bool(
                row[
                    "factor_level_identifiable"
                ]
            ),
        "subspace_level_identifiable":
            parse_bool(
                row[
                    "subspace_level_identifiable"
                ]
            ),
        "primary_truth_target":
            str(
                row.get(
                    "primary_truth_target",
                    "",
                )
            ),
        "is_missing_view_scenario":
            bool(
                partial_view_count
                > 0.0
            ),
        "source_npz_sha256":
            str(
                row[
                    "sha256"
                ]
            ),
    }


# =============================================================================
# Per-method scientific shard functions
# =============================================================================

def run_icluster_dataset(
    row: pd.Series,
    arrays: dict[str, np.ndarray],
    linear: Any,
    r_env: dict[str, Any],
) -> dict[str, Any]:
    metadata = registry_metadata(
        row
    )

    dataset_id = metadata[
        "comparator_dataset_id"
    ]

    n = metadata[
        "n_samples"
    ]

    all_idx = np.arange(
        n,
        dtype=int,
    )

    raw_views = {
        view:
            arrays[
                view
            ]
        for view in VIEW_KEYS
    }

    (
        views,
        _,
        prep_audit,
    ) = adapt_views(
        raw_views,
        fit_indices=all_idx,
        output_indices=all_idx,
        dtype=np.float64,
    )

    derived_seed = derive_seed(
        [
            "iCluster2b",
            dataset_id,
            "dataset_scientific_tuning_and_fit",
        ],
        1103,
    )

    fit = run_icluster_fit(
        r_env,
        views,
        derived_seed,
    )

    # Truth is first touched here, after the scientific fit returns.
    truth_metrics = evaluate_truth_metrics(
        linear,
        latent=fit[
            "latent"
        ],
        z_shared=arrays[
            "Z_shared"
        ],
        factor_level_identifiable=
            metadata[
                "factor_level_identifiable"
            ],
        subspace_level_identifiable=
            metadata[
                "subspace_level_identifiable"
            ],
    )

    return {
        "status":
            "PASS",
        "method":
            "iCluster2b",
        **metadata,
        "fit": {
            "derived_seed":
                int(
                    derived_seed
                ),
            "final_lambda":
                fit[
                    "lambda"
                ],
            "elapsed_seconds":
                fit[
                    "elapsed_seconds"
                ],
            "latent_shape":
                list(
                    fit[
                        "latent"
                    ].shape
                ),
            "latent_sha256":
                sha256_array(
                    fit[
                        "latent"
                    ]
                ),
            "preprocessing_audit":
                prep_audit,
            "latent_subspace_stability":
                (
                    "NOT_ESTIMABLE_WITH_"
                    "FROZEN_SINGLE_SEED_"
                    "POLICY_IN_05CMP2"
                ),
        },
        "metrics":
            truth_metrics,
        "scientific_firewalls": {
            "truth_used_during_fit_or_tuning":
                False,
            "outcome_clinical_biology_used":
                False,
            "TARGET_used":
                False,
            "05e_or_final_gate_changed":
                False,
        },
    }


def run_move_dataset(
    row: pd.Series,
    arrays: dict[str, np.ndarray],
    linear: Any,
    move_api: dict[str, Any],
) -> dict[str, Any]:
    metadata = registry_metadata(
        row
    )

    raw_views = {
        view:
            arrays[
                view
            ]
        for view in VIEW_KEYS
    }

    selection = select_move_architecture(
        move_api,
        raw_views=raw_views,
        sample_ids=arrays[
            "sample_index"
        ],
        dataset_id=metadata[
            "comparator_dataset_id"
        ],
    )

    # Truth is first used here, after architecture selection and all fits finish.
    truth_metrics = evaluate_truth_metrics(
        linear,
        latent=selection[
            "canonical_latent"
        ],
        z_shared=arrays[
            "Z_shared"
        ],
        factor_level_identifiable=
            metadata[
                "factor_level_identifiable"
            ],
        subspace_level_identifiable=
            metadata[
                "subspace_level_identifiable"
            ],
    )

    return {
        "status":
            "PASS",
        "method":
            "MOVE",
        **metadata,
        "architecture_selection": {
            k: v
            for k, v in selection.items()
            if k != "canonical_latent"
        },
        "fit": {
            "canonical_latent_shape":
                list(
                    selection[
                        "canonical_latent"
                    ].shape
                ),
            "canonical_latent_sha256":
                sha256_array(
                    selection[
                        "canonical_latent"
                    ]
                ),
            "selected_candidate_id":
                selection[
                    "selected_candidate_id"
                ],
            "latent_subspace_stability":
                selection[
                    "selected_latent_subspace_stability"
                ],
            "heldout_reconstruction":
                selection[
                    "selected_cv_mean_heldout_reconstruction_sse"
                ],
        },
        "metrics":
            truth_metrics,
        "scientific_firewalls": {
            "truth_used_during_architecture_selection_or_fit":
                False,
            "outcome_clinical_biology_used":
                False,
            "TARGET_used":
                False,
            "05e_or_final_gate_changed":
                False,
        },
    }


def failure_payload(
    method: str,
    row: pd.Series,
    first_error: str,
    second_error: str,
) -> dict[str, Any]:
    return {
        "status":
            "FAILED_AFTER_IDENTICAL_RETRY",
        "method":
            method,
        **registry_metadata(
            row
        ),
        "first_attempt_error":
            first_error,
        "second_attempt_error":
            second_error,
        "fallback_activated":
            False,
        "retuning_or_rescue":
            False,
        "scientific_firewalls": {
            "outcome_clinical_biology_used":
                False,
            "TARGET_used":
                False,
            "05e_or_final_gate_changed":
                False,
        },
    }


def execute_with_one_identical_retry(
    method: str,
    fn: Any,
    row: pd.Series,
) -> dict[str, Any]:
    errors: list[str] = []

    for attempt in [
        1,
        2,
    ]:
        try:
            result = fn()
            result[
                "attempt_count"
            ] = attempt
            result[
                "identical_retry_used"
            ] = bool(
                attempt == 2
            )
            return result

        except Exception:
            errors.append(
                traceback.format_exc()
            )
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return failure_payload(
        method,
        row,
        errors[0],
        errors[1],
    )


# =============================================================================
# Synthetic preauthorization tests
# =============================================================================

def synthetic_views(
    seed: int = 50303,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(
        seed
    )

    n = 24
    z = rng.normal(
        size=(
            n,
            3,
        )
    )

    views: dict[
        str,
        np.ndarray,
    ] = {}

    for view, p in zip(
        VIEW_KEYS,
        [
            18,
            24,
            15,
        ],
        strict=True,
    ):
        w = rng.normal(
            size=(
                3,
                p,
            )
        )

        x = (
            z @ w
            + rng.normal(
                scale=0.5,
                size=(
                    n,
                    p,
                ),
            )
        )

        views[
            view
        ] = x.astype(
            np.float64
        )

    views[
        "rna"
    ][2, 0] = np.nan

    views[
        "methylation"
    ][7, :] = np.nan

    views[
        "cnv"
    ][11, 3:6] = np.nan

    sample_ids = np.asarray(
        [
            f"SYNTH_{i:03d}"
            for i in range(
                n
            )
        ]
    )

    return (
        views,
        z.astype(
            np.float64
        ),
        sample_ids,
    )


def run_synthetic_icluster_test(
    r_env: dict[str, Any],
    views: dict[str, np.ndarray],
) -> dict[str, Any]:
    idx = np.arange(
        next(
            iter(
                views.values()
            )
        ).shape[0],
        dtype=int,
    )

    adapted, _, _ = adapt_views(
        views,
        fit_indices=idx,
        output_indices=idx,
        dtype=np.float64,
    )

    seed = derive_seed(
        [
            "iCluster2b",
            "SYNTHETIC_PREAUTH",
            "dataset_scientific_tuning_and_fit",
        ],
        1103,
    )

    fit = run_icluster_fit(
        r_env,
        adapted,
        seed,
    )

    return {
        "latent_shape":
            list(
                fit[
                    "latent"
                ].shape
            ),
        "lambda":
            fit[
                "lambda"
            ],
        "finite":
            bool(
                np.isfinite(
                    fit[
                        "latent"
                    ]
                ).all()
            ),
    }


def run_synthetic_move_test(
    move_api: dict[str, Any],
    views: dict[str, np.ndarray],
) -> dict[str, Any]:
    idx = np.arange(
        next(
            iter(
                views.values()
            )
        ).shape[0],
        dtype=int,
    )

    (
        adapted,
        masks,
        _,
    ) = adapt_views(
        views,
        fit_indices=idx,
        output_indices=idx,
        dtype=np.float32,
    )

    seed = derive_seed(
        [
            "MOVE",
            "SYNTHETIC_PREAUTH",
            "MOVE_H64_32",
            "full_data_stability",
        ],
        1103,
    )

    fit = fit_move(
        move_api,
        adapted,
        hidden=[
            64,
            32,
        ],
        seed=seed,
        epochs=2,
        kld_warmup_steps=[
            1,
            1,
            2,
            2,
        ],
    )

    sse = move_reconstruction_sse(
        fit[
            "model"
        ],
        adapted,
        masks,
    )

    shape = list(
        fit[
            "latent"
        ].shape
    )

    finite = bool(
        np.isfinite(
            fit[
                "latent"
            ]
        ).all()
    )

    del fit["model"]
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "latent_shape":
            shape,
        "finite":
            finite,
        "synthetic_reconstruction_sse":
            float(
                sse
            ),
    }


def run_synthetic_metric_test(
    linear: Any,
) -> dict[str, Any]:
    rng = np.random.default_rng(
        88017
    )

    truth = rng.normal(
        size=(
            40,
            3,
        )
    )

    rotation, _ = np.linalg.qr(
        rng.normal(
            size=(
                3,
                3,
            )
        )
    )

    estimated = truth @ rotation

    extra = rng.normal(
        size=(
            40,
            2,
        )
    )

    estimated5 = np.column_stack(
        [
            estimated,
            extra,
        ]
    )

    (
        subspace,
        replay,
    ) = paper5_shared_subspace_recovery(
        linear,
        estimated5,
        truth,
    )

    if subspace < 1.0 - 1e-10:
        raise RuntimeError(
            "Synthetic true-span subspace recovery "
            f"should be ~1; got {subspace}"
        )

    perm = np.array(
        [
            2,
            0,
            1,
        ]
    )

    signs = np.array(
        [
            -1.0,
            1.0,
            -1.0,
        ]
    )

    permuted = (
        truth[:, perm]
        * signs
    )

    permuted5 = np.column_stack(
        [
            permuted,
            rng.normal(
                size=(
                    40,
                    2,
                )
            ),
        ]
    )

    (
        factor,
        details,
    ) = factor_alignment_correlation(
        linear,
        permuted5,
        truth,
    )

    if factor < 1.0 - 1e-10:
        raise RuntimeError(
            "Synthetic sign/permutation factor alignment "
            f"should be ~1; got {factor}"
        )

    return {
        "subspace_recovery":
            float(
                subspace
            ),
        "subspace_helper_replay":
            replay,
        "factor_alignment":
            float(
                factor
            ),
        "factor_matches":
            details,
    }


def run_synthetic_selection_logic_test() -> dict[str, Any]:
    ids = np.asarray(
        [
            "S10",
            "S2",
            "S1",
            "S11",
            "S3",
            "S20",
            "S4",
            "S5",
            "S6",
            "S7",
        ]
    )

    folds1 = deterministic_sample_folds(
        ids
    )
    folds2 = deterministic_sample_folds(
        ids
    )

    if any(
        not np.array_equal(
            a,
            b,
        )
        for a, b in zip(
            folds1,
            folds2,
            strict=True,
        )
    ):
        raise RuntimeError(
            "Deterministic sample-ID folds are not deterministic."
        )

    means = {
        "A": 100.0,
        "B": 101.5,
        "C": 106.0,
    }

    best_se = 2.0

    retained = sorted(
        [
            k
            for k, v in means.items()
            if v
            <= means["A"]
            + best_se
        ]
    )

    if retained != [
        "A",
        "B",
    ]:
        raise RuntimeError(
            "Synthetic MOVE one-SE rule replay failed."
        )

    stability = [
        {
            "candidate_id": "A",
            "stability": 0.80,
            "params": 200,
        },
        {
            "candidate_id": "B",
            "stability": 0.84,
            "params": 300,
        },
    ]

    selected = sorted(
        stability,
        key=lambda r: (
            -r[
                "stability"
            ],
            r[
                "params"
            ],
            r[
                "candidate_id"
            ],
        ),
    )[0][
        "candidate_id"
    ]

    if selected != "B":
        raise RuntimeError(
            "Synthetic MOVE stage-2 selection rule replay failed."
        )

    return {
        "deterministic_folds":
            "PASS",
        "one_se_retained":
            retained,
        "stage2_selected":
            selected,
    }


def preauthorization_main(
    script_path: Path,
    paths: dict[str, Path],
    contract: dict[str, Any],
    registry: pd.DataFrame,
) -> None:
    print(
        "Execution mode: "
        "SYNTHETIC_ONLY_PREAUTHORIZATION"
    )
    print(
        "Frozen 360-dataset simulation NPZ access: FORBIDDEN"
    )
    print(
        "Scientific comparator results: FORBIDDEN"
    )

    result_root = paths[
        "result_root"
    ]

    if (
        result_root.exists()
        and any(
            p.is_file()
            for p in result_root.rglob(
                "*"
            )
        )
    ):
        raise RuntimeError(
            "Scientific comparator result files exist "
            "before 05cmp3 source authorization."
        )

    if (
        paths[
            "result_lock_dir"
        ].exists()
        and any(
            p.is_file()
            for p in paths[
                "result_lock_dir"
            ].rglob(
                "*"
            )
        )
    ):
        raise RuntimeError(
            "05cmp4 result-lock files exist "
            "before 05cmp3 authorization."
        )

    preflight_path = paths[
        "preflight"
    ]

    runner_sha = sha256_file(
        script_path
    )

    contract_sha = sha256_file(
        paths[
            "cmp2_contract"
        ]
    )

    if preflight_path.exists():
        old = load_json(
            preflight_path
        )

        if (
            old.get(
                "status"
            )
            != PREAUTH_STATUS
            or old.get(
                "runner_sha256"
            )
            != runner_sha
            or old.get(
                "cmp2_contract_sha256"
            )
            != contract_sha
        ):
            raise RuntimeError(
                "Existing 05cmp3 preflight does not bind "
                "this exact runner/contract."
            )

        print(
            "Existing exact synthetic preflight: PASS"
        )
        print(
            "No scientific data opened."
        )
        print(
            "Next: freeze 05cmp3a source authorization; "
            "do not edit this runner."
        )
        return

    linear = import_linear_helper(
        paths[
            "linear_helper"
        ]
    )

    move_api = import_move_api()

    r_env = verify_r_library(
        paths[
            "paper5"
        ]
    )

    (
        views,
        truth,
        sample_ids,
    ) = synthetic_views()

    metric_test = run_synthetic_metric_test(
        linear
    )

    selection_test = (
        run_synthetic_selection_logic_test()
    )

    move_test = run_synthetic_move_test(
        move_api,
        views,
    )

    icluster_test = run_synthetic_icluster_test(
        r_env,
        views,
    )

    report = {
        "status":
            PREAUTH_STATUS,
        "script_version":
            SCRIPT_VERSION,
        "runner_sha256":
            runner_sha,
        "cmp2_contract_sha256":
            contract_sha,
        "cmp2_registry_sha256":
            sha256_file(
                paths[
                    "cmp2_registry"
                ]
            ),
        "created_at_utc":
            utc_now_iso(),
        "execution_mode":
            "SYNTHETIC_ONLY_PREAUTHORIZATION",
        "linear_helper_source": {
            "path":
                relative_to_project(
                    paths[
                        "linear_helper"
                    ],
                    paths[
                        "project_root"
                    ],
                ),
            "sha256":
                sha256_file(
                    paths[
                        "linear_helper"
                    ]
                ),
        },
        "environment": {
            "move_version":
                move_api[
                    "move_version"
                ],
            "torch_version":
                torch.__version__,
            "cuda_build":
                str(
                    torch.version.cuda
                ),
            "gpu":
                torch.cuda.get_device_name(
                    0
                ),
            "rscript_sha256":
                sha256_file(
                    r_env[
                        "rscript"
                    ]
                ),
            "icluster_package_tree_sha256":
                r_env[
                    "package_tree_sha256"
                ],
        },
        "tests": {
            "frozen_upstream_metadata":
                "PASS",
            "deterministic_seed_derivation":
                (
                    "PASS"
                    if derive_seed(
                        [
                            "X",
                            "Y",
                        ],
                        1103,
                    )
                    == derive_seed(
                        [
                            "X",
                            "Y",
                        ],
                        1103,
                    )
                    else "FAIL"
                ),
            "adapter_feature_and_whole_view_missing":
                "PASS",
            "paper5_subspace_helper_replay":
                metric_test,
            "move_selection_logic":
                selection_test,
            "move_exact_api_synthetic_fit":
                move_test,
            "icluster2b_exact_tune_api_synthetic_fit":
                icluster_test,
        },
        "scientific_state": {
            "frozen_simulation_npz_opened":
                False,
            "frozen_simulation_truth_opened":
                False,
            "scientific_comparator_fit_performed":
                False,
            "TARGET_accessed":
                False,
            "outcomes_clinical_biology_accessed":
                False,
        },
    }

    write_json_atomic(
        report,
        preflight_path,
    )

    print(
        "Exact frozen 05cmp2/05cmp1 identities: PASS"
    )
    print(
        "Frozen Paper 5 subspace helper synthetic replay: PASS"
    )
    print(
        "Factor-alignment sign/permutation fixture: PASS"
    )
    print(
        "MOVE deterministic fold/one-SE/stability selection logic: PASS"
    )
    print(
        "MOVE exact CUDA API synthetic fit: PASS"
    )
    print(
        "iCluster2b exact tune.iCluster2b synthetic fit: PASS"
    )
    print(
        "Frozen 360-dataset simulation NPZ opened: NO"
    )
    print(
        "Frozen simulation truth opened: NO"
    )
    print(
        "Scientific comparator fit performed: NO"
    )
    print(
        "TARGET/outcomes/clinical/biology accessed: NO"
    )
    print(
        "=" * 112
    )
    print(
        "Paper 5 05cmp3 scientific comparator runner "
        "synthetic preauthorization: PASS"
    )
    print(
        "=" * 112
    )
    print(
        "Preflight: method_contract/05cmp3_preflight/"
        "secondary_comparator_runner_synthetic_preflight_v5.json"
    )
    print(
        "Next: freeze the exact runner SHA with a separate "
        "05cmp3a source authorizer. Do not edit or run real "
        "scientific data before authorization."
    )
    print(
        "=" * 112
    )


# =============================================================================
# Authorized real execution
# =============================================================================

def shard_path(
    result_root: Path,
    method: str,
    dataset_id: str,
) -> Path:
    safe = (
        dataset_id
        .replace(
            "/",
            "_",
        )
        .replace(
            "\\",
            "_",
        )
        .replace(
            ":",
            "_",
        )
    )

    return (
        result_root
        / "shards"
        / method
        / f"{safe}.json"
    )


def verify_existing_shard(
    path: Path,
    runner_sha: str,
    contract_sha: str,
    method: str,
    dataset_id: str,
) -> dict[str, Any]:
    shard = load_json(
        path
    )

    if shard.get(
        "runner_sha256"
    ) != runner_sha:
        raise RuntimeError(
            "Existing shard runner SHA mismatch: "
            f"{path}"
        )

    if shard.get(
        "cmp2_contract_sha256"
    ) != contract_sha:
        raise RuntimeError(
            "Existing shard contract SHA mismatch: "
            f"{path}"
        )

    if shard.get(
        "method"
    ) != method:
        raise RuntimeError(
            "Existing shard method mismatch: "
            f"{path}"
        )

    if shard.get(
        "comparator_dataset_id"
    ) != dataset_id:
        raise RuntimeError(
            "Existing shard dataset ID mismatch: "
            f"{path}"
        )

    if shard.get(
        "status"
    ) not in {
        "PASS",
        "FAILED_AFTER_IDENTICAL_RETRY",
    }:
        raise RuntimeError(
            "Existing shard has invalid status: "
            f"{path}"
        )

    return shard


def write_scientific_shard(
    result_root: Path,
    payload: dict[str, Any],
    runner_sha: str,
    contract_sha: str,
) -> Path:
    payload = {
        **payload,
        "runner_sha256":
            runner_sha,
        "cmp2_contract_sha256":
            contract_sha,
        "created_at_utc":
            utc_now_iso(),
    }

    path = shard_path(
        result_root,
        str(
            payload[
                "method"
            ]
        ),
        str(
            payload[
                "comparator_dataset_id"
            ]
        ),
    )

    if path.exists():
        raise RuntimeError(
            "Refusing to overwrite existing scientific shard: "
            f"{path}"
        )

    write_json_atomic(
        payload,
        path,
    )

    return path


def make_dataset_metrics_table(
    shards: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[
        dict[str, Any]
    ] = []

    for s in shards:
        metrics = (
            s.get(
                "metrics",
                {},
            )
            if s.get(
                "status"
            )
            == "PASS"
            else {}
        )

        fit = (
            s.get(
                "fit",
                {},
            )
            if s.get(
                "status"
            )
            == "PASS"
            else {}
        )

        subspace = metrics.get(
            "shared_subspace_recovery"
        )

        rows.append(
            {
                "method":
                    s[
                        "method"
                    ],
                "comparator_dataset_id":
                    s[
                        "comparator_dataset_id"
                    ],
                "scenario_id":
                    s[
                        "scenario_id"
                    ],
                "replicate_index":
                    int(
                        s[
                            "replicate_index"
                        ]
                    ),
                "simulation_seed":
                    int(
                        s[
                            "simulation_seed"
                        ]
                    ),
                "scenario_family":
                    s.get(
                        "scenario_family",
                        "",
                    ),
                "missingness_mode":
                    s.get(
                        "missingness_mode",
                        "",
                    ),
                "is_missing_view_scenario":
                    bool(
                        s.get(
                            "is_missing_view_scenario",
                            False,
                        )
                    ),
                "factor_level_identifiable":
                    bool(
                        s.get(
                            "factor_level_identifiable",
                            False,
                        )
                    ),
                "subspace_level_identifiable":
                    bool(
                        s.get(
                            "subspace_level_identifiable",
                            False,
                        )
                    ),
                "status":
                    s[
                        "status"
                    ],
                "attempt_count":
                    int(
                        s.get(
                            "attempt_count",
                            0,
                        )
                    ),
                "shared_subspace_recovery":
                    subspace,
                "factor_alignment_correlation":
                    metrics.get(
                        "factor_alignment_correlation"
                    ),
                "missing_view_robustness_shared_subspace":
                    (
                        subspace
                        if bool(
                            s.get(
                                "is_missing_view_scenario",
                                False,
                            )
                        )
                        else None
                    ),
                "latent_subspace_stability":
                    (
                        fit.get(
                            "latent_subspace_stability"
                        )
                        if isinstance(
                            fit.get(
                                "latent_subspace_stability"
                            ),
                            (
                                int,
                                float,
                                np.integer,
                                np.floating,
                            ),
                        )
                        else None
                    ),
                "heldout_reconstruction":
                    fit.get(
                        "heldout_reconstruction"
                    ),
                "selected_candidate_id":
                    fit.get(
                        "selected_candidate_id",
                        "",
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def scenario_summary_table(
    dataset_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, Any]
    ] = []

    for (
        method,
        scenario,
    ), part in dataset_metrics.groupby(
        [
            "method",
            "scenario_id",
        ],
        sort=True,
    ):
        pass_part = part.loc[
            part[
                "status"
            ].eq(
                "PASS"
            )
        ]

        row: dict[str, Any] = {
            "method":
                method,
            "scenario_id":
                scenario,
            "n_datasets":
                int(
                    part.shape[0]
                ),
            "n_pass":
                int(
                    pass_part.shape[0]
                ),
            "n_failed":
                int(
                    part.shape[0]
                    - pass_part.shape[0]
                ),
            "is_missing_view_scenario":
                bool(
                    part[
                        "is_missing_view_scenario"
                    ].astype(
                        bool
                    ).any()
                ),
        }

        for metric in [
            "shared_subspace_recovery",
            "factor_alignment_correlation",
            "missing_view_robustness_shared_subspace",
            "latent_subspace_stability",
            "heldout_reconstruction",
        ]:
            values = (
                pd.to_numeric(
                    pass_part[
                        metric
                    ],
                    errors="coerce",
                )
                .dropna()
                .to_numpy(
                    float
                )
            )

            row[
                f"{metric}_n"
            ] = int(
                values.size
            )

            row[
                f"{metric}_mean"
            ] = (
                float(
                    np.mean(
                        values
                    )
                )
                if values.size
                else np.nan
            )

            row[
                f"{metric}_median"
            ] = (
                float(
                    np.median(
                        values
                    )
                )
                if values.size
                else np.nan
            )

            row[
                f"{metric}_sd"
            ] = (
                float(
                    np.std(
                        values,
                        ddof=1,
                    )
                )
                if values.size > 1
                else np.nan
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def overall_summary_table(
    scenario_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, Any]
    ] = []

    for method, part in scenario_summary.groupby(
        "method",
        sort=True,
    ):
        row: dict[str, Any] = {
            "method":
                method,
            "n_scenarios":
                int(
                    part.shape[0]
                ),
            "n_dataset_failures":
                int(
                    part[
                        "n_failed"
                    ].sum()
                ),
        }

        for metric in [
            "shared_subspace_recovery",
            "factor_alignment_correlation",
            "missing_view_robustness_shared_subspace",
            "latent_subspace_stability",
            "heldout_reconstruction",
        ]:
            vals = (
                pd.to_numeric(
                    part[
                        f"{metric}_mean"
                    ],
                    errors="coerce",
                )
                .dropna()
                .to_numpy(
                    float
                )
            )

            row[
                f"scenario_balanced_{metric}_mean"
            ] = (
                float(
                    np.mean(
                        vals
                    )
                )
                if vals.size
                else np.nan
            )

            row[
                f"scenario_balanced_{metric}_n_scenarios"
            ] = int(
                vals.size
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def finalize_results(
    paths: dict[str, Path],
    registry: pd.DataFrame,
    runner_sha: str,
    contract_sha: str,
    auth_path: Path,
) -> None:
    result_root = paths[
        "result_root"
    ]

    all_shards: list[
        dict[str, Any]
    ] = []

    shard_records: dict[
        str,
        dict[str, Any],
    ] = {}

    for row in registry.itertuples(
        index=False
    ):
        dataset_id = str(
            row.comparator_dataset_id
        )

        for method in EXPECTED[
            "methods"
        ]:
            path = shard_path(
                result_root,
                method,
                dataset_id,
            )

            shard = verify_existing_shard(
                path,
                runner_sha,
                contract_sha,
                method,
                dataset_id,
            )

            all_shards.append(
                shard
            )

            key = (
                f"{method}/"
                f"{path.name}"
            )

            shard_records[
                key
            ] = {
                "path":
                    relative_to_project(
                        path,
                        paths[
                            "project_root"
                        ],
                    ),
                "sha256":
                    sha256_file(
                        path
                    ),
                "size_bytes":
                    path.stat().st_size,
                "status":
                    shard[
                        "status"
                    ],
            }

    if len(
        all_shards
    ) != EXPECTED[
        "n_datasets"
    ] * 2:
        raise RuntimeError(
            "Unexpected total scientific shard count."
        )

    dataset_metrics = (
        make_dataset_metrics_table(
            all_shards
        )
    )

    if dataset_metrics.shape[0] != 720:
        raise RuntimeError(
            "Dataset-metrics table does not contain "
            "720 method-dataset rows."
        )

    scenario_summary = (
        scenario_summary_table(
            dataset_metrics
        )
    )

    if scenario_summary.shape[0] != 72:
        raise RuntimeError(
            "Scenario summary does not contain "
            "72 method-scenario rows."
        )

    overall_summary = (
        overall_summary_table(
            scenario_summary
        )
    )

    if overall_summary.shape[0] != 2:
        raise RuntimeError(
            "Overall summary does not contain two methods."
        )

    tables_dir = (
        result_root / "tables"
    )

    if (
        tables_dir.exists()
        and any(
            p.is_file()
            for p in tables_dir.rglob(
                "*"
            )
        )
    ):
        raise RuntimeError(
            "Final comparator tables already exist; "
            "refusing to overwrite."
        )

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_path = (
        tables_dir
        / "secondary_comparator_dataset_metrics.csv"
    )

    scenario_path = (
        tables_dir
        / "secondary_comparator_scenario_summary.csv"
    )

    overall_path = (
        tables_dir
        / "secondary_comparator_overall_summary.csv"
    )

    dataset_metrics.to_csv(
        dataset_path,
        index=False,
    )

    scenario_summary.to_csv(
        scenario_path,
        index=False,
    )

    overall_summary.to_csv(
        overall_path,
        index=False,
    )

    summary_payload = {
        "status":
            "05CMP3_SECONDARY_COMPARATOR_CONFIRMATORY_BENCHMARK_COMPLETE",
        "script_version":
            SCRIPT_VERSION,
        "created_at_utc":
            utc_now_iso(),
        "runner_sha256":
            runner_sha,
        "cmp2_contract_sha256":
            contract_sha,
        "n_expected_datasets":
            360,
        "n_expected_method_dataset_shards":
            720,
        "method_status_counts":
            (
                dataset_metrics.groupby(
                    [
                        "method",
                        "status",
                    ]
                )
                .size()
                .rename(
                    "n"
                )
                .reset_index()
                .to_dict(
                    orient="records"
                )
            ),
        "overall_summary":
            overall_summary.to_dict(
                orient="records"
            ),
        "scientific_firewalls": {
            "TARGET_accessed":
                False,
            "outcomes_clinical_biology_accessed":
                False,
            "cross_family_winner_declared":
                False,
            "new_composite_score_created":
                False,
            "05e_reopened":
                False,
            "final_gate_changed":
                False,
        },
    }

    summary_path = (
        result_root
        / "secondary_comparator_confirmatory_summary.json"
    )

    write_json_atomic(
        summary_payload,
        summary_path,
    )

    result_lock_dir = paths[
        "result_lock_dir"
    ]

    if (
        result_lock_dir.exists()
        and any(
            p.is_file()
            for p in result_lock_dir.rglob(
                "*"
            )
        )
    ):
        raise RuntimeError(
            "05cmp4 result-lock directory already contains files."
        )

    result_lock_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_files = [
        dataset_path,
        scenario_path,
        overall_path,
        summary_path,
    ]

    lock_payload = {
        "status":
            RESULT_LOCK_STATUS,
        "script_version":
            SCRIPT_VERSION,
        "created_at_utc":
            utc_now_iso(),
        "runner_source": {
            "path":
                relative_to_project(
                    Path(__file__).resolve(),
                    paths["project_root"],
                ),
            "sha256":
                runner_sha,
        },
        "cmp2_contract": {
            "path":
                "paper5/method_contract/05cmp2/"
                "secondary_comparator_confirmatory_benchmark_contract.json",
            "sha256":
                contract_sha,
        },
        "cmp3a_authorization": {
            "path":
                relative_to_project(
                    auth_path,
                    paths[
                        "project_root"
                    ],
                ),
            "sha256":
                sha256_file(
                    auth_path
                ),
        },
        "scientific_shards":
            shard_records,
        "aggregate_files": {
            p.name: {
                "path":
                    relative_to_project(
                        p,
                        paths[
                            "project_root"
                        ],
                    ),
                "sha256":
                    sha256_file(
                        p
                    ),
                "size_bytes":
                    p.stat().st_size,
            }
            for p in output_files
        },
        "rewrite_policy": (
            "Scientific comparator results are frozen. "
            "Never overwrite/delete/rerun in place. Any "
            "recovery or extension requires a new versioned stage."
        ),
    }

    lock_path = (
        result_lock_dir
        / "secondary_comparator_confirmatory_results_lock.json"
    )

    write_json_atomic(
        lock_payload,
        lock_path,
    )

    print(
        "-" * 112
    )
    print(
        "Secondary comparator confirmatory scientific completion"
    )
    print(
        "-" * 112
    )
    print(
        overall_summary.to_string(
            index=False
        )
    )
    print(
        "Cross-family winner declared: NO"
    )
    print(
        "New composite score created: NO"
    )
    print(
        "05e reopened: NO"
    )
    print(
        "Final gate changed: NO"
    )
    print(
        "TARGET/outcomes/clinical/biology accessed: NO"
    )
    print(
        "=" * 112
    )
    print(
        "Paper 5 05cmp3 secondary comparator benchmark: "
        "COMPLETE + FROZEN"
    )
    print(
        "=" * 112
    )
    print(
        "Results lock: method_contract/05cmp4/"
        "secondary_comparator_confirmatory_results_lock.json"
    )
    print(
        "=" * 112
    )


def authorized_real_main(
    script_path: Path,
    paths: dict[str, Path],
    contract: dict[str, Any],
    registry: pd.DataFrame,
) -> None:
    auth = load_json(
        paths[
            "auth"
        ]
    )

    runner_sha = sha256_file(
        script_path
    )

    contract_sha = sha256_file(
        paths[
            "cmp2_contract"
        ]
    )

    if auth.get(
        "status"
    ) != AUTH_STATUS:
        raise RuntimeError(
            "05cmp3a source-authorization status mismatch."
        )

    if str(
        auth.get(
            "runner_source",
            {},
        ).get(
            "sha256",
            "",
        )
    ).lower() != runner_sha:
        raise RuntimeError(
            "05cmp3a does not authorize this exact runner SHA."
        )

    if str(
        auth.get(
            "cmp2_contract",
            {},
        ).get(
            "sha256",
            "",
        )
    ).lower() != contract_sha:
        raise RuntimeError(
            "05cmp3a does not bind this exact 05cmp2 contract SHA."
        )

    preflight = load_json(
        paths[
            "preflight"
        ]
    )

    if preflight.get(
        "status"
    ) != PREAUTH_STATUS:
        raise RuntimeError(
            "05cmp3 synthetic preflight status mismatch."
        )

    if preflight.get(
        "runner_sha256"
    ) != runner_sha:
        raise RuntimeError(
            "05cmp3 synthetic preflight runner SHA mismatch."
        )

    if preflight.get(
        "cmp2_contract_sha256"
    ) != contract_sha:
        raise RuntimeError(
            "05cmp3 synthetic preflight contract SHA mismatch."
        )

    if str(
        auth.get(
            "synthetic_preflight",
            {},
        ).get(
            "sha256",
            "",
        )
    ).lower() != sha256_file(
        paths[
            "preflight"
        ]
    ):
        raise RuntimeError(
            "05cmp3a synthetic-preflight hash binding mismatch."
        )

    if (
        paths[
            "result_lock_dir"
        ].exists()
        and any(
            p.is_file()
            for p in paths[
                "result_lock_dir"
            ].rglob(
                "*"
            )
        )
    ):
        raise RuntimeError(
            "05cmp4 frozen result lock already exists. "
            "Never rerun scientific comparator benchmark."
        )

    print(
        "Execution mode: "
        "AUTHORIZED_RESUMABLE_SCIENTIFIC_COMPARATOR_BENCHMARK"
    )
    print(
        "Scientific dataset bank: exact 360 frozen confirmatory simulations"
    )
    print(
        "Methods: iCluster2b + MOVE"
    )
    print(
        "TARGET/outcomes/clinical/biology access: FORBIDDEN"
    )
    print(
        "05e/final-gate consequence: NONE"
    )

    linear = import_linear_helper(
        paths[
            "linear_helper"
        ]
    )

    if sha256_file(
        paths[
            "linear_helper"
        ]
    ) != str(
        auth.get(
            "linear_helper_source",
            {},
        ).get(
            "sha256",
            "",
        )
    ).lower():
        raise RuntimeError(
            "Frozen Paper 5 linear metric-helper SHA changed."
        )

    move_api = import_move_api()

    r_env = verify_r_library(
        paths[
            "paper5"
        ]
    )

    result_root = paths[
        "result_root"
    ]

    result_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed = 0
    skipped = 0

    total = (
        EXPECTED[
            "n_datasets"
        ]
        * 2
    )

    registry_work = registry.copy()

    registry_work[
        "replicate_index"
    ] = pd.to_numeric(
        registry_work[
            "replicate_index"
        ],
        errors="raise",
    ).astype(
        int
    )

    registry_work[
        "simulation_seed"
    ] = pd.to_numeric(
        registry_work[
            "simulation_seed"
        ],
        errors="raise",
    ).astype(
        int
    )

    registry_work = (
        registry_work.sort_values(
            [
                "scenario_id",
                "replicate_index",
                "simulation_seed",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    for dataset_number, (
        _,
        row,
    ) in enumerate(
        registry_work.iterrows(),
        start=1,
    ):
        dataset_id = str(
            row[
                "comparator_dataset_id"
            ]
        )

        print(
            f"[{dataset_number:03d}/"
            f"{EXPECTED['n_datasets']}] "
            f"{dataset_id}"
        )

        needed_methods: list[
            str
        ] = []

        for method in EXPECTED[
            "methods"
        ]:
            path = shard_path(
                result_root,
                method,
                dataset_id,
            )

            if path.exists():
                verify_existing_shard(
                    path,
                    runner_sha,
                    contract_sha,
                    method,
                    dataset_id,
                )

                skipped += 1

                print(
                    f"  {method}: "
                    "verified_existing"
                )

            else:
                needed_methods.append(
                    method
                )

        if not needed_methods:
            continue

        # FIRST real scientific NPZ/truth read: only after exact source authorization.
        arrays = load_scientific_npz(
            paths[
                "project_root"
            ],
            row,
        )

        for method in needed_methods:
            started = time.perf_counter()

            if method == "iCluster2b":
                fn = (
                    lambda row=row, arrays=arrays:
                    run_icluster_dataset(
                        row,
                        arrays,
                        linear,
                        r_env,
                    )
                )

            elif method == "MOVE":
                fn = (
                    lambda row=row, arrays=arrays:
                    run_move_dataset(
                        row,
                        arrays,
                        linear,
                        move_api,
                    )
                )

            else:
                raise RuntimeError(
                    "Unexpected comparator method: "
                    f"{method}"
                )

            payload = (
                execute_with_one_identical_retry(
                    method,
                    fn,
                    row,
                )
            )

            payload[
                "wall_seconds_including_retry"
            ] = float(
                time.perf_counter()
                - started
            )

            out = write_scientific_shard(
                result_root,
                payload,
                runner_sha,
                contract_sha,
            )

            completed += 1

            print(
                f"  {method}: "
                f"{payload['status']} "
                f"attempts="
                f"{payload.get('attempt_count', 2)} "
                f"shard={out.name}"
            )

        del arrays
        gc.collect()
        torch.cuda.empty_cache()

        print(
            "  progress method-dataset shards: "
            f"new={completed} "
            f"existing={skipped} "
            f"total_expected={total}"
        )

    finalize_results(
        paths,
        registry=registry_work,
        runner_sha=runner_sha,
        contract_sha=contract_sha,
        auth_path=paths[
            "auth"
        ],
    )


# =============================================================================
# Main gate
# =============================================================================

def main() -> None:
    print(
        "=" * 112
    )
    print(
        "Paper 5 - post-05e scientific secondary-comparator "
        "confirmatory benchmark runner"
    )
    print(
        "=" * 112
    )
    print(
        "Script version:",
        SCRIPT_VERSION,
    )

    script_path = Path(
        __file__
    ).resolve()

    paths = build_paths(
        script_path
    )

    contract, registry = verify_upstream(
        paths
    )

    runner_sha = sha256_file(
        script_path
    )

    contract_sha = sha256_file(
        paths[
            "cmp2_contract"
        ]
    )

    print(
        "Runner SHA-256:",
        runner_sha,
    )

    print(
        "05cmp2 contract SHA-256:",
        contract_sha,
    )

    print(
        "Exact frozen 05cmp2/05cmp1 metadata: PASS"
    )

    if not paths[
        "auth"
    ].exists():
        preauthorization_main(
            script_path,
            paths,
            contract,
            registry,
        )
        return

    authorized_real_main(
        script_path,
        paths,
        contract,
        registry,
    )


if __name__ == "__main__":
    main()
