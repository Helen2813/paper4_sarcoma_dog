from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "05cmp1-freeze-secondary-comparator-implementations-v1-no-cli"
CONTRACT_VERSION = "sable-mo-secondary-comparator-implementation-contract-v1"
STATUS = "FROZEN_SECONDARY_COMPARATOR_IMPLEMENTATIONS_BEFORE_05E_AND_SCIENTIFIC_RUNS"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# -----------------------------------------------------------------------------
# Upstream comparator-selection / smoke anchors
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
ENV_05CMP0B = DIR_05CMP0B / "move_gpu_smoke_environment.json"

DIR_05CMP0C = PAPER5_ROOT / "method_contract" / "05cmp0c"
LOCK_05CMP0C = DIR_05CMP0C / "icluster2b_cpu_smoke_lock.json"
RESULT_05CMP0C = DIR_05CMP0C / "icluster2b_cpu_smoke_result.json"
ENV_05CMP0C = DIR_05CMP0C / "icluster2b_cpu_smoke_environment.json"

DIR_05DEV0 = PAPER5_ROOT / "method_contract" / "05dev0"
LOCK_05DEV0 = DIR_05DEV0 / "target_input_synthetic_fixture_lock.json"
CONTRACT_05DEV0 = DIR_05DEV0 / "target_input_synthetic_fixture_contract.json"

DIR_05E = PAPER5_ROOT / "method_contract" / "05e"
SECONDARY_RESULTS = PAPER5_ROOT / "results" / "secondary_comparators"

# -----------------------------------------------------------------------------
# 05cmp1 outputs
# -----------------------------------------------------------------------------

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05cmp1"
OUTPUT_CONTRACT = OUTPUT_DIR / "secondary_comparator_implementation_contract.json"
OUTPUT_IMPLEMENTATION_REGISTRY = OUTPUT_DIR / "secondary_comparator_implementation_registry.csv"
OUTPUT_ADAPTER_REGISTRY = OUTPUT_DIR / "secondary_comparator_scientific_adapter_registry.csv"
OUTPUT_SEED_REGISTRY = OUTPUT_DIR / "secondary_comparator_seed_registry.csv"
OUTPUT_METRIC_REGISTRY = OUTPUT_DIR / "secondary_comparator_metric_eligibility_registry.csv"
OUTPUT_FAILURE_POLICY = OUTPUT_DIR / "secondary_comparator_failure_policy.json"
OUTPUT_README = OUTPUT_DIR / "05cmp1_README.txt"
OUTPUT_LOCK = OUTPUT_DIR / "secondary_comparator_implementation_contract_lock.json"

VIEW_ORDER = ("RNA", "METHYLATION", "CNV")
LATENT_DIM = 5

# iCluster2b: current documented continuous-data tuning path.
ICLUSTER_POLICY = {
    "method": "iCluster2b",
    "package": "iClusterPlus",
    "package_version": "1.47.3",
    "official_Bioconductor_source_release": "1.48.0",
    "Bioconductor_version": "3.23",
    "R_version_prefix": "R version 4.6.1",
    "platform_binary": "official Bioconductor 3.23 Windows x86_64 binary for R 4.6",
    "latent_K": LATENT_DIM,
    "tuning_function": "tune.iCluster2b",
    "regularization": "lasso",
    "min_lambda": 10.0,
    "max_lambda": 500.0,
    "lambda_iter": 25,
    "EM_iter": 25,
    "min_shrinkage_rate_per_view": [0.05, 0.05, 0.05],
    "eps": 1e-4,
    "eps2": 1e-6,
    "chr": None,
    "thread_policy": "single-threaded process for numerical reproducibility",
    "scientific_fit_rule": (
        "Run one prospectively seeded tune.iCluster2b call per analysis dataset with K=5 and the "
        "fixed documented lasso search settings above; use the lambda values returned by that call. "
        "No truth, outcome, biological label, or comparator-performance metric may enter tuning."
    ),
}

# MOVE: outcome-blind architecture selection.  The authors' public pipeline explicitly
# separates reconstruction-focused architecture search from stability-focused selection;
# this contract freezes a compact small-n implementation of that principle before results.
MOVE_POLICY = {
    "method": "MOVE",
    "distribution": "move-dl",
    "version": "1.5.0",
    "required_torch": "2.12.1+cu130",
    "required_CUDA_build": "13.0",
    "required_runtime": "cuda:0",
    "dtype": "float32",
    "AMP": False,
    "TF32": False,
    "torch_compile": False,
    "latent_dim": LATENT_DIM,
    "candidate_hidden_architectures": [
        {"candidate_id": "MOVE_H64_32", "hidden": [64, 32]},
        {"candidate_id": "MOVE_H128_64", "hidden": [128, 64]},
        {"candidate_id": "MOVE_H256_128", "hidden": [256, 128]},
    ],
    "beta": 0.01,
    "dropout": 0.10,
    "learning_rate": 1e-3,
    "epochs": 300,
    "kld_warmup_steps": [1, 75, 150, 300],
    "batch_size_rule": "min(16, n_fit_samples), with no batch smaller than the available n_fit_samples",
    "continuous_weights": [1.0, 1.0, 1.0],
    "optimizer_and_loss": "MOVE 1.5.0 public VAE/training_loop implementation without local numerical modification",
    "architecture_selection": {
        "stage_1": (
            "Five deterministic sample-ID folds; for each candidate, fit one model per fold with the frozen "
            "architecture-selection seed and rank by mean held-out reconstruction SSE computed only over observed "
            "standardized entries. Retain candidates within one standard error of the best mean reconstruction SSE."
        ),
        "stage_2": (
            "For each stage-1-retained candidate, fit the full analysis dataset with the three frozen stability seeds. "
            "Compute median pairwise latent-subspace canonical correlation after orthogonal alignment. Select the "
            "highest-stability candidate; exact ties prefer fewer hidden parameters, then lexical candidate_id."
        ),
        "truth_outcome_use": "FORBIDDEN",
        "single_canonical_scientific_fit": (
            "After architecture selection, the seed tagged canonical_fit is the sole latent representation used for "
            "factor-level / sample-level scientific comparator outputs. Other seeds are stability diagnostics only."
        ),
    },
}

COMMON_ADAPTER_POLICY = {
    "view_order": list(VIEW_ORDER),
    "latent_dimension": LATENT_DIM,
    "feature_selection": (
        "Use exactly the frozen outcome-blind Paper 5 feature-budget / preprocessing inputs supplied to the primary "
        "benchmark for that dataset or split. The comparator wrapper may not perform outcome-guided feature selection."
    ),
    "standardization": (
        "Within any fit/training split, estimate per-feature center and scale from observed entries in that split only. "
        "Apply those parameters to held-out rows. Zero-variance features follow the already-frozen Paper 5 preprocessing "
        "policy and are not rescued comparator-specifically."
    ),
    "missing_feature_values": (
        "After standardization, encode remaining missing feature values as 0.0, i.e. the training-set standardized mean."
    ),
    "whole_view_missingness": (
        "Encode an unavailable view as an all-zero standardized row for that view. Do not add comparator-specific "
        "missingness indicator features."
    ),
    "sample_pairing": "Preserve the frozen sample-ID order and cross-view pairing exactly; no re-pairing is permitted.",
    "information_boundary": (
        "Outcomes, clinical labels, biological annotations, simulation truth, and final-gate labels are forbidden from "
        "preprocessing, tuning, architecture selection, fitting, or failure handling."
    ),
}


# -----------------------------------------------------------------------------
# Helpers
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


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def any_files(path: Path) -> bool:
    return path.exists() and any(candidate.is_file() for candidate in path.rglob("*"))


def load_json(path: Path) -> dict[str, Any]:
    require_file(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp, path)


def write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(tmp, path)


def write_csv_atomic(rows: list[dict[str, Any]], fieldnames: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp, path)


def verify_generated_files(lock: dict[str, Any], label: str) -> None:
    records = lock.get("generated_files", {})
    if not isinstance(records, dict) or not records:
        raise RuntimeError(f"{label} lock has no generated_files registry.")
    failures: list[str] = []
    for rel_name, record in records.items():
        candidate = PROJECT_ROOT / Path(str(rel_name))
        if not candidate.is_file():
            failures.append(f"missing: {rel_name}")
            continue
        expected_hash = clean_text(record.get("sha256"))
        if expected_hash and sha256_file(candidate) != expected_hash:
            failures.append(f"hash mismatch: {rel_name}")
    if failures:
        raise RuntimeError(f"{label} frozen-file verification failed:\n" + "\n".join(f"- {x}" for x in failures))


def verify_existing_lock() -> None:
    require_file(OUTPUT_LOCK)
    lock = load_json(OUTPUT_LOCK)
    if clean_text(lock.get("status")) != STATUS:
        raise RuntimeError("Existing 05cmp1 lock has unexpected status.")
    if clean_text(lock.get("script_version")) != SCRIPT_VERSION:
        raise RuntimeError("Existing 05cmp1 lock was produced by a different script version.")
    verify_generated_files(lock, "05cmp1")
    print("=" * 88)
    print("Existing Paper 5 05cmp1 comparator implementation contract: VERIFIED")
    print("=" * 88)
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("No files were rewritten.")
    print("=" * 88)


def read_selected_registry() -> list[dict[str, str]]:
    require_file(EFFECTIVE_SELECTED_05CMP0A)
    with EFFECTIVE_SELECTED_05CMP0A.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


# -----------------------------------------------------------------------------
# Upstream verification
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
        ENV_05CMP0B,
        LOCK_05CMP0C,
        RESULT_05CMP0C,
        ENV_05CMP0C,
        LOCK_05DEV0,
        CONTRACT_05DEV0,
    ]
    for path in required:
        require_file(path)

    if any_files(SECONDARY_RESULTS):
        raise RuntimeError(
            "Secondary-comparator scientific results already exist. 05cmp1 must be frozen before any scientific comparator run."
        )
    if any_files(DIR_05E):
        raise RuntimeError(
            "05e files already exist. This 05cmp1 version is intentionally frozen before 05e so secondary comparators cannot influence primary-method selection."
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
    amend0a = load_json(AMENDMENT_05CMP0A)
    expected0a = "FROZEN_COMPARATOR_METHOD_CORRECTION_BEFORE_SCIENTIFIC_RUNS"
    if clean_text(lock0a.get("status")) != expected0a or clean_text(amend0a.get("status")) != expected0a:
        raise RuntimeError("05cmp0a is not in the expected frozen state.")
    if sha256_file(AMENDMENT_05CMP0A) != clean_text(lock0a.get("amendment_sha256")):
        raise RuntimeError("05cmp0a amendment hash mismatch.")
    verify_generated_files(lock0a, "05cmp0a")

    selected = read_selected_registry()
    classical = [r for r in selected if clean_text(r.get("comparator_role")) == "required_external_classical"]
    deep = [r for r in selected if clean_text(r.get("comparator_role")) == "required_external_deep"]
    if len(classical) != 1 or clean_text(classical[0].get("method_name")) != "iCluster2b":
        raise RuntimeError("05cmp0a effective classical comparator is not uniquely iCluster2b.")
    if len(deep) != 1 or clean_text(deep[0].get("method_name")) != "MOVE":
        raise RuntimeError("05cmp0a effective deep comparator is not uniquely MOVE.")

    lock0b = load_json(LOCK_05CMP0B)
    result0b = load_json(RESULT_05CMP0B)
    env0b = load_json(ENV_05CMP0B)
    expected0b = "PASS_SYNTHETIC_MOVE_GPU_SMOKE_BEFORE_05CMP1"
    if clean_text(lock0b.get("status")) != expected0b or clean_text(result0b.get("status")) != expected0b:
        raise RuntimeError("05cmp0b MOVE smoke is not in the expected PASS state.")
    verify_generated_files(lock0b, "05cmp0b")
    if not all(bool(v) for v in result0b.get("pass_checks", {}).values()):
        raise RuntimeError("05cmp0b contains a failed hard smoke check.")

    lock0c = load_json(LOCK_05CMP0C)
    result0c = load_json(RESULT_05CMP0C)
    env0c = load_json(ENV_05CMP0C)
    expected0c = "PASS_SYNTHETIC_ICLUSTER2B_CPU_SMOKE_BEFORE_05CMP1"
    if clean_text(lock0c.get("status")) != expected0c or clean_text(result0c.get("status")) != expected0c:
        raise RuntimeError("05cmp0c iCluster2b smoke is not in the expected PASS state.")
    verify_generated_files(lock0c, "05cmp0c")
    if not all(bool(v) for v in result0c.get("pass_checks", {}).values()):
        raise RuntimeError("05cmp0c contains a failed hard smoke check.")

    lockdev = load_json(LOCK_05DEV0)
    contractdev = load_json(CONTRACT_05DEV0)
    expecteddev = "FROZEN_SYNTHETIC_TARGET_INPUT_FIXTURES_BEFORE_TARGET_CODE"
    if clean_text(lockdev.get("status")) != expecteddev or clean_text(contractdev.get("status")) != expecteddev:
        raise RuntimeError("05dev0 is not in the expected frozen state.")
    if sha256_file(CONTRACT_05DEV0) != clean_text(lockdev.get("contract_sha256")):
        raise RuntimeError("05dev0 contract hash mismatch.")
    verify_generated_files(lockdev, "05dev0")

    # Exact runtime/API evidence from successful smoke tests.
    if clean_text(result0b.get("comparator", {}).get("version")) != MOVE_POLICY["version"]:
        raise RuntimeError("MOVE version differs from the 05cmp1 prospective implementation policy.")
    if clean_text(env0b.get("torch_version")) != MOVE_POLICY["required_torch"]:
        raise RuntimeError("MOVE smoke PyTorch version differs from the 05cmp1 prospective runtime policy.")
    if clean_text(env0b.get("torch_cuda_build")) != MOVE_POLICY["required_CUDA_build"]:
        raise RuntimeError("MOVE smoke CUDA build differs from the 05cmp1 prospective runtime policy.")

    r_env = env0c.get("R", {})
    if clean_text(r_env.get("iclusterplus_version")) != ICLUSTER_POLICY["package_version"]:
        raise RuntimeError("iClusterPlus Windows binary version differs from the 05cmp1 prospective policy.")
    if clean_text(r_env.get("bioconductor_version")) != ICLUSTER_POLICY["Bioconductor_version"]:
        raise RuntimeError("Bioconductor version differs from the 05cmp1 prospective policy.")
    if not clean_text(r_env.get("r_version")).startswith(ICLUSTER_POLICY["R_version_prefix"]):
        raise RuntimeError("R version differs from the 05cmp1 prospective policy.")

    return {
        "move_result": result0b,
        "move_environment": env0b,
        "icluster_result": result0c,
        "icluster_environment": env0c,
        "upstream_hashes": {
            relative_path(path): sha256_file(path)
            for path in required
        },
    }


# -----------------------------------------------------------------------------
# Frozen registries
# -----------------------------------------------------------------------------

def implementation_rows(upstream: dict[str, Any]) -> list[dict[str, Any]]:
    move_env = upstream["move_environment"]
    i_env = upstream["icluster_environment"].get("R", {})
    move_sources = move_env.get("move_source_files", {})
    move_source_hashes = ";".join(
        f"{name}:{clean_text(record.get('sha256'))}"
        for name, record in sorted(move_sources.items())
    )
    return [
        {
            "comparator_role": "required_external_classical",
            "method_name": "iCluster2b",
            "implementation": "iClusterPlus::tune.iCluster2b -> iCluster2b",
            "package_or_distribution": "iClusterPlus",
            "version": ICLUSTER_POLICY["package_version"],
            "runtime": "R 4.6.1 / Bioconductor 3.23 / Windows x86_64",
            "device": "CPU",
            "numeric_dtype": "R double",
            "latent_dimension": LATENT_DIM,
            "implementation_hash_evidence": clean_text(i_env.get("iclusterplus_tree_sha256")),
            "fallback": "iClusterBayes2",
            "fallback_state_after_smoke": "LOCKED_INACTIVE",
            "scientific_run_before_05e": "FORBIDDEN",
            "may_influence_05e": "NO",
            "may_influence_final_gate": "NO",
        },
        {
            "comparator_role": "required_external_deep",
            "method_name": "MOVE",
            "implementation": "move.models.VAE + move.training.training_loop",
            "package_or_distribution": "move-dl",
            "version": MOVE_POLICY["version"],
            "runtime": f"PyTorch {MOVE_POLICY['required_torch']} / CUDA {MOVE_POLICY['required_CUDA_build']}",
            "device": "cuda:0 REQUIRED",
            "numeric_dtype": "float32",
            "latent_dimension": LATENT_DIM,
            "implementation_hash_evidence": move_source_hashes,
            "fallback": "AUTOENCODIX varix",
            "fallback_state_after_smoke": "LOCKED_INACTIVE",
            "scientific_run_before_05e": "FORBIDDEN",
            "may_influence_05e": "NO",
            "may_influence_final_gate": "NO",
        },
    ]


def adapter_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, dtype in [("iCluster2b", "float64/R double"), ("MOVE", "float32")]:
        rows.append(
            {
                "method_name": method,
                "view_order": "RNA|METHYLATION|CNV",
                "input_orientation": "samples_x_features_per_view",
                "latent_dimension": LATENT_DIM,
                "input_dtype": dtype,
                "feature_selection": "same frozen outcome-blind Paper5 feature-budget/preprocessing inputs",
                "standardization": "fit-split observed-entry center/scale only",
                "feature_missing_after_standardization": "0.0",
                "whole_view_missing": "all-zero standardized row",
                "missingness_indicator_added": "NO",
                "sample_repair_or_repairing": "FORBIDDEN",
                "outcome_or_truth_access": "FORBIDDEN",
            }
        )
    return rows


def seed_rows() -> list[dict[str, Any]]:
    return [
        {
            "method_name": "iCluster2b",
            "purpose": "dataset_scientific_tuning_and_fit",
            "seed_label": "icluster_fit",
            "base_seed": 1103,
            "derivation": "SHA256(method|dataset_id|purpose|base_seed) -> integer in [1, 2147483646]",
            "selection_role": "single deterministic seed; no seed selection by performance",
        },
        {
            "method_name": "MOVE",
            "purpose": "architecture_selection_cv",
            "seed_label": "architecture_selection",
            "base_seed": 1103,
            "derivation": "SHA256(method|dataset_id|fold_id|candidate_id|purpose|base_seed) -> integer in [1, 2147483646]",
            "selection_role": "reconstruction-only architecture screening",
        },
        {
            "method_name": "MOVE",
            "purpose": "full_data_stability",
            "seed_label": "canonical_fit",
            "base_seed": 1103,
            "derivation": "SHA256(method|dataset_id|candidate_id|purpose|base_seed) -> integer in [1, 2147483646]",
            "selection_role": "a-priori canonical scientific latent representation",
        },
        {
            "method_name": "MOVE",
            "purpose": "full_data_stability",
            "seed_label": "stability_rep_2",
            "base_seed": 2207,
            "derivation": "SHA256(method|dataset_id|candidate_id|purpose|base_seed) -> integer in [1, 2147483646]",
            "selection_role": "stability diagnostic only",
        },
        {
            "method_name": "MOVE",
            "purpose": "full_data_stability",
            "seed_label": "stability_rep_3",
            "base_seed": 3301,
            "derivation": "SHA256(method|dataset_id|candidate_id|purpose|base_seed) -> integer in [1, 2147483646]",
            "selection_role": "stability diagnostic only",
        },
    ]


def metric_rows() -> list[dict[str, Any]]:
    metrics = [
        ("shared_subspace_recovery", "YES", "YES", "simulation truth used only after fit for evaluation"),
        ("factor_alignment_correlation", "YES", "YES", "simulation truth used only after fit; sign/permutation invariant"),
        ("scenario_robustness_summary", "YES", "YES", "same frozen scenario groups; descriptive secondary comparator"),
        ("missing_view_robustness", "YES", "YES", "evaluate frozen missing-view scenarios under the fixed zero adapter"),
        ("latent_subspace_stability", "YES", "YES", "outcome-blind resampling/seed stability; method-specific implementation"),
        ("heldout_reconstruction", "NO", "YES", "MOVE-native; iCluster2b not promoted as directly comparable reconstruction endpoint"),
        ("cross_view_retrieval_MRR", "NO", "NO", "not a native apples-to-apples endpoint for these integrated comparator APIs"),
        ("private_subspace_recovery", "NO", "NO", "comparators do not define SABLE-MO view-private latent factors"),
        ("SABLE_stability_gate_PASS_FAIL", "NO", "NO", "SABLE-specific gate is not retrofitted onto external methods"),
        ("primary_05e_selection_score", "NO", "NO", "secondary comparators are ineligible to influence 05e"),
        ("final_gate_decision", "NO", "NO", "secondary comparators are ineligible to alter the frozen final gate"),
    ]
    return [
        {
            "metric": metric,
            "iCluster2b_eligible": icluster,
            "MOVE_eligible": move,
            "interpretation": interpretation,
            "selection_consequence": "NONE_FOR_05E_OR_FINAL_GATE",
        }
        for metric, icluster, move, interpretation in metrics
    ]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    print("=" * 88)
    print("Paper 5 - freeze exact secondary comparator implementations")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("GPU computation in 05cmp1: NO")
    print("Model fitting in 05cmp1: NO")
    print("Scientific comparator result access: NO")
    print("TARGET/outcome access: NO")
    print()

    if OUTPUT_LOCK.exists():
        verify_existing_lock()
        return

    upstream = verify_upstream()
    print("05cmp0 frozen comparator selection: PASS")
    print("05cmp0a continuous-data correction: PASS")
    print("05cmp0b MOVE GPU synthetic smoke: PASS")
    print("05cmp0c iCluster2b CPU synthetic smoke: PASS")
    print("05dev0 synthetic fixture anchor: PASS")
    print("05e files present: NO")
    print("Secondary comparator scientific results present: NO")
    print()

    implementation = implementation_rows(upstream)
    adapters = adapter_rows()
    seeds = seed_rows()
    metrics = metric_rows()

    failure_policy = {
        "status": STATUS,
        "pre_scientific_environment_mismatch": "STOP; do not silently substitute package/runtime versions.",
        "per_dataset_fit_failure": (
            "Repeat once with the identical frozen numerical configuration and identical seed solely to distinguish "
            "transient execution failure from deterministic method failure. If it fails again, record the dataset-method "
            "fit as failed. Do not retune, change architecture, alter lambda bounds, drop hard samples, or switch fallback."
        ),
        "nonfinite_output": "Record method-dataset failure; no numerical clipping/rescue unless a new prospective amendment is frozen.",
        "iClusterBayes2_fallback": (
            "Not activated: iCluster2b passed the frozen synthetic implementation smoke. The same-package fallback may not "
            "be activated later because of scientific performance or dataset-specific failure."
        ),
        "AUTOENCODIX_varix_fallback": (
            "Not activated: MOVE passed the frozen synthetic GPU implementation smoke. The deep fallback may not be "
            "activated later because of scientific performance or dataset-specific failure."
        ),
        "failure_reporting": "All failures remain in denominator/manifest and are reported by method and scenario.",
        "05e_consequence": "NONE",
        "final_gate_consequence": "NONE",
    }

    contract = {
        "script_version": SCRIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": STATUS,
        "scientific_state": {
            "05cmp0b_MOVE_synthetic_smoke_passed": True,
            "05cmp0c_iCluster2b_synthetic_smoke_passed": True,
            "scientific_secondary_comparator_results_accessed": False,
            "real_TARGET_values_accessed": False,
            "TARGET_outcomes_accessed": False,
            "biological_annotations_accessed": False,
            "05e_results_accessed": False,
            "primary_method_selected_here": False,
            "final_gate_changed_here": False,
        },
        "primary_rule": (
            "Secondary external comparators are frozen as reporting/benchmark methods only. They are categorically "
            "ineligible to influence 05e primary-method selection, the frozen final-gate calibration, or TARGET outcome access."
        ),
        "common_scientific_adapter_policy": COMMON_ADAPTER_POLICY,
        "iCluster2b_scientific_policy": ICLUSTER_POLICY,
        "MOVE_scientific_policy": MOVE_POLICY,
        "seed_derivation_rule": (
            "Where a derived seed is required, concatenate the literal fields with '|' in the order stated in the seed "
            "registry, compute SHA256 UTF-8, take the first 8 hexadecimal characters as an unsigned integer, then map "
            "to 1 + value mod 2147483646. No seed is selected by fit quality or truth/outcome metrics."
        ),
        "metric_scope": (
            "Only metrics explicitly marked eligible in the frozen metric registry may be used for cross-method claims. "
            "Method-specific endpoints remain descriptive and are not inserted into the 05e primary selection score."
        ),
        "runtime_reproducibility": {
            "MOVE": {
                "required_move_version": MOVE_POLICY["version"],
                "required_torch": MOVE_POLICY["required_torch"],
                "required_CUDA_build": MOVE_POLICY["required_CUDA_build"],
                "cuda_required": True,
                "AMP": False,
                "TF32": False,
                "torch_compile": False,
                "synthetic_smoke_source_hashes": upstream["move_environment"].get("move_source_files", {}),
            },
            "iCluster2b": {
                "R": upstream["icluster_environment"].get("R", {}),
                "package_tree_hash": upstream["icluster_environment"].get("R", {}).get("iclusterplus_tree_sha256"),
                "single_thread": True,
            },
        },
        "upstream_hashes": upstream["upstream_hashes"],
        "next_stage": "05e_compare_ablate_and_freeze_method.py",
        "post_05e_rule": (
            "Only after a frozen successful 05e primary-method decision may scientific secondary-comparator runs begin. "
            "Their outputs remain secondary and cannot reopen 05e or the final-gate contract."
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(contract, OUTPUT_CONTRACT)
    write_csv_atomic(implementation, list(implementation[0].keys()), OUTPUT_IMPLEMENTATION_REGISTRY)
    write_csv_atomic(adapters, list(adapters[0].keys()), OUTPUT_ADAPTER_REGISTRY)
    write_csv_atomic(seeds, list(seeds[0].keys()), OUTPUT_SEED_REGISTRY)
    write_csv_atomic(metrics, list(metrics[0].keys()), OUTPUT_METRIC_REGISTRY)
    write_json_atomic(failure_policy, OUTPUT_FAILURE_POLICY)

    readme = f"""Paper 5 - 05cmp1 secondary comparator implementation freeze
================================================================================
Script version: {SCRIPT_VERSION}
Status: {STATUS}

Frozen required external comparators
------------------------------------
Classical: iCluster2b (iClusterPlus 1.47.3 official Windows binary; R 4.6.1; Bioconductor 3.23)
Deep: MOVE (move-dl 1.5.0; PyTorch 2.12.1+cu130; CUDA 13.0)
Fallbacks: iClusterBayes2 and AUTOENCODIX varix remain locked inactive because both required methods passed synthetic smoke tests.

Common scientific representation dimension
-------------------------------------------
Integrated/shared latent dimension: {LATENT_DIM}

Scientific boundaries
---------------------
- No scientific comparator results were read or generated here.
- No TARGET values, outcomes, biology, or final-gate data were accessed.
- Secondary comparators cannot influence 05e or the final gate.
- Scientific comparator execution remains forbidden until 05e is frozen.

Exact iCluster2b policy
-----------------------
K={LATENT_DIM}; lasso; tune.iCluster2b; min.lambda=10; max.lambda=500; lambda.iter=25;
EM.iter=25; min.shrinkage.rate=0.05 per view; eps=1e-4; eps2=1e-6.
No performance-based fallback or post-hoc retuning.

Exact MOVE policy
-----------------
Latent={LATENT_DIM}; beta=0.01; dropout=0.10; lr=1e-3; epochs=300; equal continuous weights.
Candidate hidden architectures: [64,32], [128,64], [256,128].
Architecture selection is outcome-blind: held-out reconstruction first, then latent-subspace stability,
with deterministic simplicity tie-breaking. Canonical scientific seed is fixed prospectively.

Missingness / preprocessing
---------------------------
Use the same frozen outcome-blind Paper 5 inputs. Standardization is fit-split only using observed entries.
After standardization, feature-level missing values and whole missing views are zero-coded. No extra missingness
indicator features are added by either comparator.

Next
----
05e_compare_ablate_and_freeze_method.py
"""
    write_text_atomic(readme, OUTPUT_README)

    generated = [
        OUTPUT_CONTRACT,
        OUTPUT_IMPLEMENTATION_REGISTRY,
        OUTPUT_ADAPTER_REGISTRY,
        OUTPUT_SEED_REGISTRY,
        OUTPUT_METRIC_REGISTRY,
        OUTPUT_FAILURE_POLICY,
        OUTPUT_README,
    ]

    lock = {
        "script_version": SCRIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": STATUS,
        "contract": file_record(OUTPUT_CONTRACT),
        "script_source": file_record(SCRIPT_PATH),
        "generated_files": {
            relative_path(path): file_record(path)
            for path in generated
        },
        "upstream_hashes": upstream["upstream_hashes"],
        "rewrite_policy": (
            "05cmp1 is immutable after creation. Any scientifically justified comparator implementation, runtime, "
            "adapter, seed, tuning, metric, or failure-policy change requires a new versioned amendment."
        ),
    }
    write_json_atomic(lock, OUTPUT_LOCK)

    print("-" * 88)
    print("Frozen secondary comparator implementation")
    print("-" * 88)
    print("Classical comparator: iCluster2b")
    print("  R: 4.6.1 | Bioconductor: 3.23 | iClusterPlus: 1.47.3 official Windows binary")
    print("  Latent K: 5 | tuning: tune.iCluster2b | lasso")
    print("Deep comparator: MOVE")
    print("  move-dl: 1.5.0 | PyTorch: 2.12.1+cu130 | CUDA: 13.0")
    print("  Latent: 5 | GPU required | outcome-blind reconstruction -> stability architecture rule")
    print("Fallbacks activated: NO")
    print("Scientific comparator results used: NO")
    print("05e influence: FORBIDDEN")
    print()
    print("=" * 88)
    print("Paper 5 05cmp1 secondary comparator implementation contract: FROZEN")
    print("=" * 88)
    print(f"Contract: {relative_path(OUTPUT_CONTRACT)}")
    print(f"Implementation registry: {relative_path(OUTPUT_IMPLEMENTATION_REGISTRY)}")
    print(f"Adapter registry: {relative_path(OUTPUT_ADAPTER_REGISTRY)}")
    print(f"Seed registry: {relative_path(OUTPUT_SEED_REGISTRY)}")
    print(f"Metric registry: {relative_path(OUTPUT_METRIC_REGISTRY)}")
    print(f"Failure policy: {relative_path(OUTPUT_FAILURE_POLICY)}")
    print(f"Lock: {relative_path(OUTPUT_LOCK)}")
    print("Next: 05e_compare_ablate_and_freeze_method.py")
    print("=" * 88)


if __name__ == "__main__":
    main()
