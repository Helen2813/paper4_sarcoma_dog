from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05dev0-freeze-target-input-synthetic-fixtures-v1-no-cli"
FIXTURE_VERSION = "paper5-target-input-synthetic-fixture-v1"
STATUS = "FROZEN_SYNTHETIC_TARGET_INPUT_FIXTURES_BEFORE_TARGET_CODE"

SEED = 20260821

# These are deliberately tiny DEVELOPMENT dimensions.
# They are not scientific feature budgets and do not define the final representation.
N_SAMPLES = 24
N_RNA = 40
N_METH = 50
N_CNV = 30

GENERATOR_SHARED_LATENT_DIM = 4
GENERATOR_PRIVATE_LATENT_DIM = 2

# Development-only fold geometry. This does NOT replace or amend any scientific
# TARGET nested-CV registry that may later be frozen/used.
N_DEVELOPMENT_FOLDS = 5
DEVELOPMENT_SEEDS = (1103, 2207)

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# Upstream prospective anchor. 05dev0 intentionally does NOT depend on 05d5 or 05e.
DIR_05A2 = PAPER5_ROOT / "method_contract" / "05a2"
LOCK_05A2 = DIR_05A2 / "inference_final_gate_implementation_amendment_lock.json"
AMENDMENT_05A2 = DIR_05A2 / "inference_final_gate_implementation_amendment.json"

# Output locations.
CONTRACT_DIR = PAPER5_ROOT / "method_contract" / "05dev0"
FIXTURE_ROOT = PAPER5_ROOT / "development_fixtures" / "target_input_synthetic_v1"

MANIFEST_CSV = FIXTURE_ROOT / "sample_manifest.csv"
AVAILABILITY_CSV = FIXTURE_ROOT / "view_availability.csv"
RNA_NPY = FIXTURE_ROOT / "rna.npy"
METH_NPY = FIXTURE_ROOT / "methylation.npy"
CNV_NPY = FIXTURE_ROOT / "cnv.npy"
RNA_FEATURES_CSV = FIXTURE_ROOT / "rna_features.csv"
METH_FEATURES_CSV = FIXTURE_ROOT / "methylation_features.csv"
CNV_FEATURES_CSV = FIXTURE_ROOT / "cnv_features.csv"
CORRECT_PAIRING_CSV = FIXTURE_ROOT / "pairing_map_correct.csv"
CORRUPTED_PAIRING_CSV = FIXTURE_ROOT / "pairing_map_corrupted.csv"
DEV_FOLDS_CSV = FIXTURE_ROOT / "development_folds.csv"
DEV_SEEDS_CSV = FIXTURE_ROOT / "development_seed_registry.csv"
EXPECTED_SCHEMA_JSON = FIXTURE_ROOT / "expected_input_schema.json"
TRUTH_JSON = FIXTURE_ROOT / "synthetic_generator_truth.json"

MALFORMED_DIR = FIXTURE_ROOT / "malformed"
MALFORMED_DUPLICATE_IDS = MALFORMED_DIR / "duplicate_sample_ids.csv"
MALFORMED_ALL_VIEWS_MISSING = MALFORMED_DIR / "all_views_missing_sample.csv"
MALFORMED_MASK_ROW_MISMATCH = MALFORMED_DIR / "mask_matrix_inconsistency.json"
MALFORMED_PAIRING_DUPLICATE = MALFORMED_DIR / "pairing_duplicate_target.csv"

CONTRACT_JSON = CONTRACT_DIR / "target_input_synthetic_fixture_contract.json"
README_TXT = CONTRACT_DIR / "05dev0_README.txt"
LOCK_JSON = CONTRACT_DIR / "target_input_synthetic_fixture_lock.json"


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na"}:
        return ""
    return text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_project_path(value: Any) -> Path:
    path = Path(clean_text(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_json(path: Path) -> dict[str, Any]:
    require_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def write_csv_atomic(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    table.to_csv(
        temp,
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    os.replace(temp, path)


def write_npy_atomic(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}.npy")
    np.save(temp, array, allow_pickle=False)
    os.replace(temp, path)


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def verify_generated_files(lock: dict[str, Any], label: str) -> None:
    failures: list[str] = []
    for relative_name, record in lock.get("generated_files", {}).items():
        path = resolve_project_path(relative_name)
        if not path.is_file():
            failures.append(f"missing: {relative_name}")
            continue

        expected_size = record.get("size_bytes")
        if expected_size is not None and int(path.stat().st_size) != int(expected_size):
            failures.append(f"size mismatch: {relative_name}")
            continue

        expected_hash = clean_text(record.get("sha256"))
        if expected_hash and sha256_file(path) != expected_hash:
            failures.append(f"SHA-256 mismatch: {relative_name}")

    if failures:
        raise RuntimeError(
            f"{label} generated-file verification failed:\n- "
            + "\n- ".join(failures[:100])
        )


# -----------------------------------------------------------------------------
# Upstream verification
# -----------------------------------------------------------------------------

def verify_05a2() -> dict[str, Any]:
    require_file(LOCK_05A2)
    require_file(AMENDMENT_05A2)

    lock = load_json(LOCK_05A2)
    amendment = load_json(AMENDMENT_05A2)

    expected_status = "FROZEN_BEFORE_FINAL_GATE_MATERIALIZATION"
    if clean_text(lock.get("status")) != expected_status:
        raise RuntimeError("05a2 lock is not frozen in the expected prospective state.")
    if clean_text(amendment.get("status")) != expected_status:
        raise RuntimeError("05a2 amendment is not frozen in the expected prospective state.")

    expected_amendment_hash = clean_text(lock.get("amendment_sha256"))
    observed_amendment_hash = sha256_file(AMENDMENT_05A2)
    if expected_amendment_hash != observed_amendment_hash:
        raise RuntimeError("05a2 amendment hash does not match the 05a2 lock.")

    verify_generated_files(lock, "05a2")

    return {
        "amendment_sha256": observed_amendment_hash,
        "lock_sha256": sha256_file(LOCK_05A2),
    }


# -----------------------------------------------------------------------------
# Synthetic fixture generation
# -----------------------------------------------------------------------------

def standardize_columns(matrix: np.ndarray) -> np.ndarray:
    mean = np.nanmean(matrix, axis=0, keepdims=True)
    std = np.nanstd(matrix, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((matrix - mean) / std).astype(np.float32)


def build_latent_matrices(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    shared = rng.normal(0.0, 1.0, size=(N_SAMPLES, GENERATOR_SHARED_LATENT_DIM))
    private = {
        "rna": rng.normal(0.0, 1.0, size=(N_SAMPLES, GENERATOR_PRIVATE_LATENT_DIM)),
        "methylation": rng.normal(
            0.0, 1.0, size=(N_SAMPLES, GENERATOR_PRIVATE_LATENT_DIM)
        ),
        "cnv": rng.normal(0.0, 1.0, size=(N_SAMPLES, GENERATOR_PRIVATE_LATENT_DIM)),
    }
    return shared, private


def synthesize_view(
    rng: np.random.Generator,
    shared: np.ndarray,
    private: np.ndarray,
    n_features: int,
    noise_sd: float,
) -> np.ndarray:
    shared_loadings = rng.normal(
        0.0,
        0.65,
        size=(GENERATOR_SHARED_LATENT_DIM, n_features),
    )
    private_loadings = rng.normal(
        0.0,
        0.35,
        size=(GENERATOR_PRIVATE_LATENT_DIM, n_features),
    )
    noise = rng.normal(0.0, noise_sd, size=(N_SAMPLES, n_features))
    matrix = shared @ shared_loadings + private @ private_loadings + noise
    return standardize_columns(matrix)


def build_availability() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index in range(N_SAMPLES):
        rna = True
        meth = True
        cnv = True
        missingness_case = "complete"

        if index in {8, 9, 10, 11}:
            meth = False
            missingness_case = "methylation_missing"
        elif index in {12, 13, 14}:
            cnv = False
            missingness_case = "cnv_missing"
        elif index in {15, 16}:
            rna = False
            missingness_case = "rna_missing"
        elif index in {17, 18}:
            meth = False
            cnv = False
            missingness_case = "rna_only"
        elif index in {19, 20}:
            rna = False
            cnv = False
            missingness_case = "methylation_only"
        elif index in {21}:
            rna = False
            meth = False
            missingness_case = "cnv_only"
        elif index in {22}:
            rna = False
            missingness_case = "methylation_cnv_only"
        elif index in {23}:
            meth = False
            missingness_case = "rna_cnv_only"

        if not (rna or meth or cnv):
            raise RuntimeError("Synthetic generator created an all-views-missing valid sample.")

        rows.append(
            {
                "synthetic_sample_id": f"SYN_{index + 1:03d}",
                "sample_index": index,
                "rna_available": rna,
                "methylation_available": meth,
                "cnv_available": cnv,
                "n_available_views": int(rna) + int(meth) + int(cnv),
                "missingness_case": missingness_case,
            }
        )

    return pd.DataFrame(rows)


def apply_view_missingness(
    rna: np.ndarray,
    meth: np.ndarray,
    cnv: np.ndarray,
    availability: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_rna = rna.copy()
    out_meth = meth.copy()
    out_cnv = cnv.copy()

    for row in availability.itertuples(index=False):
        index = int(row.sample_index)
        if not bool(row.rna_available):
            out_rna[index, :] = np.nan
        if not bool(row.methylation_available):
            out_meth[index, :] = np.nan
        if not bool(row.cnv_available):
            out_cnv[index, :] = np.nan

    # Small feature-level missingness only inside available methylation rows.
    # This exists to test feature-level NA handling without touching a real dataset.
    feature_missing_pairs = [(0, 1), (2, 7), (3, 13), (5, 21), (6, 32)]
    for sample_index, feature_index in feature_missing_pairs:
        if bool(availability.loc[sample_index, "methylation_available"]):
            out_meth[sample_index, feature_index] = np.nan

    return out_rna, out_meth, out_cnv


def build_feature_table(prefix: str, n_features: int, view: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_index": np.arange(n_features, dtype=int),
            "feature_id": [f"SYN_{prefix}_{i + 1:04d}" for i in range(n_features)],
            "view": view,
            "synthetic_only": True,
        }
    )


def build_pairing_maps(
    availability: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sample_ids = availability["synthetic_sample_id"].astype(str).tolist()

    correct = pd.DataFrame(
        {
            "anchor_sample_id": sample_ids,
            "rna_sample_id": sample_ids,
            "methylation_sample_id": sample_ids,
            "cnv_sample_id": sample_ids,
            "pairing_status": "correct",
        }
    )

    corrupted = correct.copy()
    # Rotate methylation identities across six samples.
    methylation_corruption_indices = [1, 2, 3, 4, 5, 6]
    rotated = methylation_corruption_indices[1:] + methylation_corruption_indices[:1]
    for source_index, target_index in zip(
        methylation_corruption_indices,
        rotated,
        strict=True,
    ):
        corrupted.loc[source_index, "methylation_sample_id"] = sample_ids[target_index]

    # Swap CNV identities across four samples.
    cnv_swaps = [(8, 9), (10, 11)]
    for left, right in cnv_swaps:
        left_value = corrupted.loc[left, "cnv_sample_id"]
        right_value = corrupted.loc[right, "cnv_sample_id"]
        corrupted.loc[left, "cnv_sample_id"] = right_value
        corrupted.loc[right, "cnv_sample_id"] = left_value

    mismatch_mask = (
        corrupted["rna_sample_id"].ne(corrupted["anchor_sample_id"])
        | corrupted["methylation_sample_id"].ne(corrupted["anchor_sample_id"])
        | corrupted["cnv_sample_id"].ne(corrupted["anchor_sample_id"])
    )
    corrupted.loc[mismatch_mask, "pairing_status"] = "intentionally_corrupted"

    truth = {
        "pairing_corruption": {
            "methylation_rotated_anchor_indices": methylation_corruption_indices,
            "cnv_swapped_anchor_index_pairs": cnv_swaps,
            "n_corrupted_anchor_rows": int(mismatch_mask.sum()),
        }
    }
    return correct, corrupted, truth


def build_development_folds() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for repeat_id in (1, 2):
        # Deliberately deterministic and simple; development-only.
        offset = repeat_id - 1
        for sample_index in range(N_SAMPLES):
            fold = ((sample_index + offset) % N_DEVELOPMENT_FOLDS) + 1
            rows.append(
                {
                    "synthetic_sample_id": f"SYN_{sample_index + 1:03d}",
                    "development_repeat": repeat_id,
                    "development_fold": fold,
                    "scientific_split": False,
                    "notes": (
                        "Synthetic development split only; must never replace a frozen TARGET scientific split registry."
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_development_seeds() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "seed_role": "development_fixture_only",
                "seed": int(seed),
                "scientific_fit_seed": False,
                "notes": "For deterministic fixture/unit tests only.",
            }
            for seed in DEVELOPMENT_SEEDS
        ]
    )


def build_malformed_cases(
    availability: pd.DataFrame,
    correct_pairing: pd.DataFrame,
) -> dict[str, Any]:
    duplicate_ids = availability[["synthetic_sample_id", "sample_index"]].copy()
    duplicate_ids.loc[1, "synthetic_sample_id"] = duplicate_ids.loc[0, "synthetic_sample_id"]

    all_missing = availability.copy()
    all_missing.loc[0, ["rna_available", "methylation_available", "cnv_available"]] = False
    all_missing.loc[0, "n_available_views"] = 0
    all_missing.loc[0, "missingness_case"] = "INTENTIONALLY_INVALID_ALL_VIEWS_MISSING"

    pairing_duplicate = correct_pairing.copy()
    pairing_duplicate.loc[1, "methylation_sample_id"] = pairing_duplicate.loc[
        0, "methylation_sample_id"
    ]
    pairing_duplicate.loc[1, "pairing_status"] = "INTENTIONALLY_INVALID_DUPLICATE_TARGET"

    write_csv_atomic(duplicate_ids, MALFORMED_DUPLICATE_IDS)
    write_csv_atomic(all_missing, MALFORMED_ALL_VIEWS_MISSING)
    write_csv_atomic(pairing_duplicate, MALFORMED_PAIRING_DUPLICATE)

    mask_mismatch = {
        "case_id": "MASK_MATRIX_INCONSISTENCY",
        "intentionally_invalid": True,
        "synthetic_sample_id": "SYN_001",
        "declared_rna_available": True,
        "matrix_condition": "all RNA values are NaN",
        "expected_validator_behavior": "FAIL",
    }
    write_json_atomic(mask_mismatch, MALFORMED_MASK_ROW_MISMATCH)

    return {
        "duplicate_sample_id_fixture": relative_path(MALFORMED_DUPLICATE_IDS),
        "all_views_missing_fixture": relative_path(MALFORMED_ALL_VIEWS_MISSING),
        "mask_matrix_inconsistency_fixture": relative_path(MALFORMED_MASK_ROW_MISMATCH),
        "pairing_duplicate_target_fixture": relative_path(MALFORMED_PAIRING_DUPLICATE),
    }


def validate_fixture(
    availability: pd.DataFrame,
    rna: np.ndarray,
    meth: np.ndarray,
    cnv: np.ndarray,
    correct_pairing: pd.DataFrame,
    corrupted_pairing: pd.DataFrame,
) -> dict[str, Any]:
    expected_shapes = {
        "rna": (N_SAMPLES, N_RNA),
        "methylation": (N_SAMPLES, N_METH),
        "cnv": (N_SAMPLES, N_CNV),
    }
    observed_shapes = {
        "rna": tuple(rna.shape),
        "methylation": tuple(meth.shape),
        "cnv": tuple(cnv.shape),
    }
    if observed_shapes != expected_shapes:
        raise RuntimeError(
            f"Synthetic fixture shape mismatch: {observed_shapes} != {expected_shapes}"
        )

    if availability["synthetic_sample_id"].duplicated().any():
        raise RuntimeError("Valid fixture contains duplicate synthetic sample IDs.")
    if (availability["n_available_views"] < 1).any():
        raise RuntimeError("Valid fixture contains a sample with no available views.")

    matrices = {
        "rna": rna,
        "methylation": meth,
        "cnv": cnv,
    }
    availability_columns = {
        "rna": "rna_available",
        "methylation": "methylation_available",
        "cnv": "cnv_available",
    }

    for view, matrix in matrices.items():
        column = availability_columns[view]
        for row in availability.itertuples(index=False):
            index = int(row.sample_index)
            available = bool(getattr(row, column))
            all_nan = bool(np.isnan(matrix[index, :]).all())
            if available and all_nan:
                raise RuntimeError(
                    f"Valid fixture inconsistency: {view} available but row {index} is all NaN."
                )
            if not available and not all_nan:
                raise RuntimeError(
                    f"Valid fixture inconsistency: {view} unavailable but row {index} is not all NaN."
                )

    sample_ids = set(availability["synthetic_sample_id"].astype(str))
    for column in ["anchor_sample_id", "rna_sample_id", "methylation_sample_id", "cnv_sample_id"]:
        if not set(correct_pairing[column].astype(str)).issubset(sample_ids):
            raise RuntimeError(f"Correct pairing contains unknown IDs in {column}.")
        if not set(corrupted_pairing[column].astype(str)).issubset(sample_ids):
            raise RuntimeError(f"Corrupted pairing contains unknown IDs in {column}.")

    if not (
        correct_pairing["anchor_sample_id"].eq(correct_pairing["rna_sample_id"]).all()
        and correct_pairing["anchor_sample_id"].eq(correct_pairing["methylation_sample_id"]).all()
        and correct_pairing["anchor_sample_id"].eq(correct_pairing["cnv_sample_id"]).all()
    ):
        raise RuntimeError("Correct pairing map is not identity aligned.")

    corrupted_rows = int(
        corrupted_pairing["pairing_status"].eq("intentionally_corrupted").sum()
    )
    if corrupted_rows <= 0:
        raise RuntimeError("Corrupted pairing fixture contains no corrupted rows.")

    feature_level_meth_missing = 0
    for row in availability.itertuples(index=False):
        if bool(row.methylation_available):
            feature_level_meth_missing += int(np.isnan(meth[int(row.sample_index), :]).sum())

    return {
        "valid_fixture_pass": True,
        "n_samples": N_SAMPLES,
        "matrix_shapes": {
            key: list(value)
            for key, value in observed_shapes.items()
        },
        "n_missing_view_rows": {
            "rna": int((~availability["rna_available"]).sum()),
            "methylation": int((~availability["methylation_available"]).sum()),
            "cnv": int((~availability["cnv_available"]).sum()),
        },
        "methylation_feature_level_nan_count_in_available_rows": feature_level_meth_missing,
        "corrupted_pairing_anchor_rows": corrupted_rows,
    }


# -----------------------------------------------------------------------------
# Existing freeze verification
# -----------------------------------------------------------------------------

def verify_existing_freeze() -> None:
    require_file(LOCK_JSON)
    require_file(CONTRACT_JSON)

    lock = load_json(LOCK_JSON)
    contract = load_json(CONTRACT_JSON)

    if clean_text(lock.get("status")) != STATUS:
        raise RuntimeError("Unexpected existing 05dev0 lock status.")
    if clean_text(contract.get("status")) != STATUS:
        raise RuntimeError("Unexpected existing 05dev0 contract status.")

    expected_contract_hash = clean_text(lock.get("contract_sha256"))
    if sha256_file(CONTRACT_JSON) != expected_contract_hash:
        raise RuntimeError("Existing 05dev0 contract hash mismatch.")

    verify_generated_files(lock, "05dev0")

    print("=" * 88)
    print("Existing Paper 5 synthetic TARGET input fixture freeze: PASS")
    print("=" * 88)
    print(f"Contract: {relative_path(CONTRACT_JSON)}")
    print(f"Fixture root: {relative_path(FIXTURE_ROOT)}")
    print(f"Lock: {relative_path(LOCK_JSON)}")
    print("No files were rewritten.")
    print("=" * 88)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    print("=" * 88)
    print("Paper 5 - freeze synthetic TARGET input/development fixtures")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Fixture root: {FIXTURE_ROOT}")
    print("GPU computation in 05dev0: NO")
    print("Model fitting in 05dev0: NO")
    print("TARGET data access in 05dev0: NO")
    print("TARGET outcomes access in 05dev0: NO")
    print("Final representation method assumed by fixture: NO")
    print()

    if LOCK_JSON.exists():
        verify_existing_freeze()
        return

    upstream = verify_05a2()
    print("05a2 prospective clarification lock: PASS")
    print("Synthetic fixture generator depends on real TARGET values: NO")
    print()

    rng = np.random.default_rng(SEED)
    shared, private = build_latent_matrices(rng)

    rna = synthesize_view(
        rng,
        shared,
        private["rna"],
        N_RNA,
        noise_sd=0.55,
    )
    meth = synthesize_view(
        rng,
        shared,
        private["methylation"],
        N_METH,
        noise_sd=0.70,
    )
    cnv = synthesize_view(
        rng,
        shared,
        private["cnv"],
        N_CNV,
        noise_sd=0.60,
    )

    availability = build_availability()
    rna, meth, cnv = apply_view_missingness(rna, meth, cnv, availability)

    sample_manifest = availability[
        [
            "synthetic_sample_id",
            "sample_index",
            "missingness_case",
        ]
    ].copy()
    sample_manifest["synthetic_only"] = True
    sample_manifest["real_TARGET_identifier"] = False

    correct_pairing, corrupted_pairing, pairing_truth = build_pairing_maps(availability)
    development_folds = build_development_folds()
    development_seeds = build_development_seeds()

    rna_features = build_feature_table("RNA", N_RNA, "rna")
    meth_features = build_feature_table("METH", N_METH, "methylation")
    cnv_features = build_feature_table("CNV", N_CNV, "cnv")

    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    MALFORMED_DIR.mkdir(parents=True, exist_ok=True)
    CONTRACT_DIR.mkdir(parents=True, exist_ok=True)

    write_csv_atomic(sample_manifest, MANIFEST_CSV)
    write_csv_atomic(availability, AVAILABILITY_CSV)
    write_npy_atomic(rna, RNA_NPY)
    write_npy_atomic(meth, METH_NPY)
    write_npy_atomic(cnv, CNV_NPY)
    write_csv_atomic(rna_features, RNA_FEATURES_CSV)
    write_csv_atomic(meth_features, METH_FEATURES_CSV)
    write_csv_atomic(cnv_features, CNV_FEATURES_CSV)
    write_csv_atomic(correct_pairing, CORRECT_PAIRING_CSV)
    write_csv_atomic(corrupted_pairing, CORRUPTED_PAIRING_CSV)
    write_csv_atomic(development_folds, DEV_FOLDS_CSV)
    write_csv_atomic(development_seeds, DEV_SEEDS_CSV)

    malformed = build_malformed_cases(availability, correct_pairing)

    validation = validate_fixture(
        availability,
        rna,
        meth,
        cnv,
        correct_pairing,
        corrupted_pairing,
    )

    expected_schema = {
        "fixture_version": FIXTURE_VERSION,
        "synthetic_only": True,
        "purpose": (
            "Development/unit testing of future TARGET input alignment, missing-view handling, "
            "pairing checks, split plumbing, and output-schema plumbing without reading any real TARGET value."
        ),
        "sample_axis": {
            "n_samples": N_SAMPLES,
            "id_column": "synthetic_sample_id",
            "row_order_contract": "all three aligned matrices follow sample_manifest.csv sample_index order",
        },
        "views": {
            "rna": {
                "path": relative_path(RNA_NPY),
                "shape": [N_SAMPLES, N_RNA],
                "dtype": "float32",
                "unavailable_view_encoding": "all NaN row",
            },
            "methylation": {
                "path": relative_path(METH_NPY),
                "shape": [N_SAMPLES, N_METH],
                "dtype": "float32",
                "unavailable_view_encoding": "all NaN row",
                "feature_level_missingness_in_available_rows": True,
            },
            "cnv": {
                "path": relative_path(CNV_NPY),
                "shape": [N_SAMPLES, N_CNV],
                "dtype": "float32",
                "unavailable_view_encoding": "all NaN row",
            },
        },
        "availability": {
            "path": relative_path(AVAILABILITY_CSV),
            "columns": ["rna_available", "methylation_available", "cnv_available"],
            "valid_sample_requires_at_least_one_available_view": True,
        },
        "pairing": {
            "correct_map": relative_path(CORRECT_PAIRING_CSV),
            "corrupted_map": relative_path(CORRUPTED_PAIRING_CSV),
        },
        "development_only_split_registry": {
            "path": relative_path(DEV_FOLDS_CSV),
            "scientific_split": False,
        },
        "development_only_seed_registry": {
            "path": relative_path(DEV_SEEDS_CSV),
            "scientific_fit_seed": False,
        },
        "explicitly_not_frozen_here": [
            "final representation method",
            "latent dimension",
            "selected-method output schema",
            "TARGET scientific nested-CV splits",
            "TARGET scientific fit seeds",
            "feature budgets",
            "clinical/outcome columns",
            "biology/program-score columns",
            "final-gate thresholds",
        ],
    }
    write_json_atomic(expected_schema, EXPECTED_SCHEMA_JSON)

    generator_truth = {
        "fixture_version": FIXTURE_VERSION,
        "synthetic_only": True,
        "rng_seed": SEED,
        "generator_shared_latent_dim": GENERATOR_SHARED_LATENT_DIM,
        "generator_private_latent_dim_per_view": GENERATOR_PRIVATE_LATENT_DIM,
        "generator_latents_are_test_truth_not_scientific_representation": True,
        "pairing_truth": pairing_truth,
        "validation": validation,
        "malformed_cases": malformed,
    }
    write_json_atomic(generator_truth, TRUTH_JSON)

    generated_fixture_files = [
        MANIFEST_CSV,
        AVAILABILITY_CSV,
        RNA_NPY,
        METH_NPY,
        CNV_NPY,
        RNA_FEATURES_CSV,
        METH_FEATURES_CSV,
        CNV_FEATURES_CSV,
        CORRECT_PAIRING_CSV,
        CORRUPTED_PAIRING_CSV,
        DEV_FOLDS_CSV,
        DEV_SEEDS_CSV,
        EXPECTED_SCHEMA_JSON,
        TRUTH_JSON,
        MALFORMED_DUPLICATE_IDS,
        MALFORMED_ALL_VIEWS_MISSING,
        MALFORMED_MASK_ROW_MISMATCH,
        MALFORMED_PAIRING_DUPLICATE,
    ]

    contract = {
        "script_version": SCRIPT_VERSION,
        "fixture_version": FIXTURE_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": STATUS,
        "purpose": (
            "Freeze synthetic, method-agnostic TARGET INPUT fixtures before downstream TARGET code is written, "
            "so loaders/validators/pairing/split plumbing can be developed without touching real TARGET values."
        ),
        "prospective_anchor": {
            "05a2_amendment_sha256": upstream["amendment_sha256"],
            "05a2_lock_sha256": upstream["lock_sha256"],
        },
        "scientific_state": {
            "real_TARGET_values_accessed": False,
            "TARGET_outcomes_accessed": False,
            "biological_annotations_accessed": False,
            "final_representation_method_selected_by_fixture": False,
            "latent_output_schema_frozen_by_fixture": False,
            "scientific_TARGET_splits_defined_by_fixture": False,
            "scientific_TARGET_seeds_defined_by_fixture": False,
        },
        "development_boundaries": {
            "allowed_uses": [
                "input-loader development",
                "shape/schema validation",
                "view-availability validation",
                "pairing validation",
                "missing-view code-path testing",
                "feature-level missing-value handling",
                "development-only split/seed plumbing",
                "negative/failure-path tests",
            ],
            "forbidden_interpretations": [
                "scientific performance estimate",
                "TARGET performance proxy",
                "representation-method selection",
                "latent-dimension selection",
                "threshold selection",
                "biology or outcome analysis",
            ],
        },
        "fixture_files": {
            relative_path(path): file_record(path)
            for path in generated_fixture_files
        },
        "generator_constants": {
            "rng_seed": SEED,
            "n_samples": N_SAMPLES,
            "n_rna": N_RNA,
            "n_methylation": N_METH,
            "n_cnv": N_CNV,
            "generator_shared_latent_dim": GENERATOR_SHARED_LATENT_DIM,
            "generator_private_latent_dim": GENERATOR_PRIVATE_LATENT_DIM,
        },
        "post_05e_rule": (
            "After 05e selects/freezes the primary representation method, create a separate versioned synthetic "
            "selected-method output fixture if needed. Do not retroactively rewrite 05dev0."
        ),
    }
    write_json_atomic(contract, CONTRACT_JSON)

    readme = f"""Paper 5 - 05dev0 synthetic TARGET input fixture freeze
==============================================================================
Script version: {SCRIPT_VERSION}
Fixture version: {FIXTURE_VERSION}
Status: {STATUS}

WHY THIS EXISTS
---------------
This package is created BEFORE downstream TARGET code so that future loaders,
validators, pairing checks, missing-view logic, and split/seed plumbing can be
written and tested without reading any real TARGET value.

WHY THIS IS NOT TOO EARLY
-------------------------
05dev0 freezes only the INPUT/DEVELOPMENT interface. It does NOT freeze the
final representation method or latent-output schema. 05e has not yet selected
the primary representation method, so any selected-method output fixture must
be created later as a new versioned artifact.

REAL DATA ACCESS
----------------
Real TARGET values accessed: NO
TARGET outcomes accessed: NO
Biological annotations accessed: NO

SCIENTIFIC CHOICES NOT MADE HERE
--------------------------------
- final representation method
- latent dimension
- selected-method output schema
- scientific TARGET nested-CV splits
- scientific TARGET fit seeds
- feature budgets
- outcomes
- biology/program scores

VALID FIXTURES
--------------
{relative_path(MANIFEST_CSV)}
{relative_path(AVAILABILITY_CSV)}
{relative_path(RNA_NPY)}
{relative_path(METH_NPY)}
{relative_path(CNV_NPY)}
{relative_path(CORRECT_PAIRING_CSV)}
{relative_path(CORRUPTED_PAIRING_CSV)}
{relative_path(DEV_FOLDS_CSV)}
{relative_path(DEV_SEEDS_CSV)}
{relative_path(EXPECTED_SCHEMA_JSON)}

MALFORMED NEGATIVE-TEST FIXTURES
--------------------------------
{relative_path(MALFORMED_DUPLICATE_IDS)}
{relative_path(MALFORMED_ALL_VIEWS_MISSING)}
{relative_path(MALFORMED_MASK_ROW_MISMATCH)}
{relative_path(MALFORMED_PAIRING_DUPLICATE)}

RULE AFTER 05e
--------------
If the selected representation method needs method-specific expected outputs,
create a NEW versioned fixture artifact after 05e. Never rewrite 05dev0.
"""
    write_text_atomic(readme, README_TXT)

    generated_all = generated_fixture_files + [CONTRACT_JSON, README_TXT]
    lock = {
        "script_version": SCRIPT_VERSION,
        "fixture_version": FIXTURE_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": STATUS,
        "contract_path": relative_path(CONTRACT_JSON),
        "contract_sha256": sha256_file(CONTRACT_JSON),
        "generated_files": {
            relative_path(path): file_record(path)
            for path in generated_all
        },
        "rewrite_policy": (
            "05dev0 is immutable once frozen. Any change requires a new versioned fixture stage. "
            "Method-specific output fixtures belong after 05e and must not rewrite 05dev0."
        ),
    }
    write_json_atomic(lock, LOCK_JSON)

    print("-" * 88)
    print("Synthetic TARGET input fixture summary")
    print("-" * 88)
    print(f"Synthetic samples: {N_SAMPLES}")
    print(f"RNA shape: {rna.shape}")
    print(f"Methylation shape: {meth.shape}")
    print(f"CNV shape: {cnv.shape}")
    print(
        "Missing-view rows: "
        f"RNA={validation['n_missing_view_rows']['rna']}, "
        f"METH={validation['n_missing_view_rows']['methylation']}, "
        f"CNV={validation['n_missing_view_rows']['cnv']}"
    )
    print(
        "Available-row methylation feature-level NaNs: "
        f"{validation['methylation_feature_level_nan_count_in_available_rows']}"
    )
    print(
        "Intentionally corrupted pairing anchor rows: "
        f"{validation['corrupted_pairing_anchor_rows']}"
    )
    print("Malformed negative-test fixtures: 4")
    print("Final representation method frozen here: NO")
    print("Scientific TARGET splits/seeds frozen here: NO")
    print()

    print("=" * 88)
    print("Paper 5 synthetic TARGET input fixtures: FROZEN")
    print("=" * 88)
    print(f"Fixture root: {relative_path(FIXTURE_ROOT)}")
    print(f"Contract: {relative_path(CONTRACT_JSON)}")
    print(f"Lock: {relative_path(LOCK_JSON)}")
    print("05d5 may continue running independently: YES")
    print("Next scientific stage remains 05d5 -> 05e; 05dev0 does not alter that sequence.")
    print("=" * 88)


if __name__ == "__main__":
    main()
