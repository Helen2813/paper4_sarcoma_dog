#!/usr/bin/env python3
"""
Paper 5 - 06e1 V1 biological alignment + matched-random controls runner v1.

Two-stage, no-CLI runner.

Before source authorization:
    - reads ONLY frozen Paper-5 contract/provenance JSON files;
    - runs a fully synthetic mathematical/schema self-test;
    - MUST NOT hash, parse, or read any real biological input;
    - MUST NOT read the numeric TARGET V1 representation array.

After a separate source-authorization lock exists for this exact source SHA:
    - verifies every frozen biological input SHA-256 BEFORE parsing any row/value;
    - loads only the frozen 78-case V1 representation and outcome-blind TARGET
      expression / frozen Paper-4 program definitions;
    - computes the four prespecified M34/M40/M11/M24 program scores;
    - computes all 4 x 5 signed Spearman correlations;
    - uses max absolute Spearman across the same five frozen V1 coordinates as
      the program-level observed statistic;
    - evaluates 1,000 variability-matched random programs per frozen program,
      with the null also maximizing across the same five coordinates;
    - applies the +1 empirical probability and BH correction across four programs;
    - writes immutable scientific outputs and a 06e2 result lock.

This runner NEVER:
    - reads clinical or outcome values;
    - selects/refits the representation;
    - applies SABLE-MO q10/stability-retention thresholds to V1;
    - calls a V1 coordinate "retained" or "stable";
    - changes the frozen technical status UNKNOWN.

No command-line arguments.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = (
    "06e1-run-v1-biological-alignment-random-controls-v2-"
    "exact-risk-oriented-weight-schema-two-stage-source-authorized-no-cli"
)

PREAUTH_STATUS = "06E1_V2_BIOLOGY_RUNNER_SYNTHETIC_SELFTEST_PASS"
AUTH_STATUS = "06E1A_V2_TARGET_V1_BIOLOGY_RUNNER_SOURCE_AUTHORIZED"
RESULT_STATUS = "06E2_V2_V1_BIOLOGICAL_ALIGNMENT_RANDOM_CONTROLS_FROZEN"

EXPECTED = {
    "model_id": "V1_SHARED_PRIVATE_AE",
    "profile_id": "ARCH_COMPACT__REG_WEAK",
    "primary_seed": 1103,
    "n_cases": 78,
    "shared_dim": 5,
    "programs": ["M34", "M40", "M11", "M24"],
    "strict_gene_counts": {
        "M34": 162,
        "M40": 111,
        "M11": 7,
        "M24": 7,
    },
    "representation_filename": "target_v1_patient_shared_seed1103.npy",
    "representation_sha256":
        "c05bf08017ad8544e1e4ce0c8f58f086060dd7a6a0a68ed7f8d5a6723768a95f",
    "technical_status":
        "UNKNOWN_NO_STRICT_USABLE_TECHNICAL_COVARIATES",
    "random_panels_per_program": 1000,
    "random_seed": 42,
    "variability_bins": 10,
    "primary_score_variant": "strict_signed_mean",
    "primary_alignment_metric":
        "max_absolute_spearman_across_5_frozen_v1_shared_coordinates",
    "minimum_program_coverage_fraction": 1.0,
    "multiplicity": "BH",
    "multiplicity_alpha": 0.05,
}

TARGET_CASE_PATTERN = re.compile(
    r"TARGET-\d{2}-[A-Z0-9]+",
    flags=re.IGNORECASE,
)


REAL_STRICT_WEIGHT_COLUMNS = [
    "module_label",
    "canine_gene",
    "canine_gene_symbol",
    "human_gene_symbol",
    "ortholog_qc_status",
    "raw_pca_loading",
    "risk_oriented_loading",
    "absolute_risk_loading",
    "is_strict_mapping",
    "is_broad_mapping",
    "human_symbol_duplicate_count",
    "normalized_abs_sum_weight",
]

V1_FAILURE_LOCK_STATUS = (
    "06E1R0_V1_FIRST_REAL_BIOLOGY_ATTEMPT_SCHEMA_FAILURE_FROZEN"
)
V1_FAILED_RUNNER_SHA256 = (
    "51eface403ec7b41a6436289590f0cd4159ad4b6e167800052de8eb2f11bd180"
)
RISK_ORIENTATION_COLUMN = "risk_oriented_loading"




# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def relative_to_paper5(path: Path, paper5: Path) -> str:
    return path.resolve().relative_to(paper5.resolve()).as_posix()


def own_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def normalize_gene_symbol(value: Any) -> str:
    return str(value).strip().upper()


def normalize_target_case_id(value: Any) -> str:
    text = str(value).strip().upper()
    match = TARGET_CASE_PATTERN.search(text)
    if match is None:
        raise ValueError(f"Could not resolve TARGET case id from {value!r}")
    return match.group(0).upper()


def safe_float_array(values: Any, *, label: str) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if not np.isfinite(x).all():
        raise RuntimeError(f"{label} contains non-finite values.")
    return x


# ---------------------------------------------------------------------
# Frozen-contract verification
# ---------------------------------------------------------------------

def verify_06e0_chain(
    paper5: Path,
    *,
    touch_biological_files: bool,
) -> dict[str, Any]:
    """
    Verify 06e0 + upstream Paper-5 provenance.

    In synthetic/preauthorization mode touch_biological_files MUST be False.
    The function then reads only Paper-5 method-contract metadata and does not
    stat/hash/open any Paper-4 biological artifact.
    """
    contract_path = (
        paper5 / "method_contract" / "06e0"
        / "v1_biological_alignment_random_control_contract.json"
    )
    lock_path = (
        paper5 / "method_contract" / "06e0"
        / "06e0_v1_biological_alignment_random_control_contract_lock.json"
    )
    results_lock_path = (
        paper5 / "method_contract" / "06c3"
        / "target_v1_full_data_refit_results_lock.json"
    )
    scope_lock_path = (
        paper5 / "method_contract" / "06d0"
        / "06d0_v1_postfit_downstream_scope_audit_lock.json"
    )
    technical_lock_path = (
        paper5 / "method_contract" / "06d1_v2"
        / "06d1_v2_v1_postfreeze_technical_status_audit_lock.json"
    )
    v1_failure_lock_path = (
        paper5 / "method_contract" / "06e1r0"
        / "v1_first_real_biology_attempt_schema_failure_boundary_lock.json"
    )

    contract = load_json(contract_path)
    lock = load_json(lock_path)
    results_lock = load_json(results_lock_path)
    scope_lock = load_json(scope_lock_path)
    technical_lock = load_json(technical_lock_path)
    v1_failure_lock = load_json(v1_failure_lock_path)

    if lock.get("status") != (
        "06E0_V1_BIOLOGICAL_ALIGNMENT_RANDOM_CONTROL_CONTRACT_FROZEN"
    ):
        raise RuntimeError("06e0 lock is not frozen.")

    locked_contract = lock.get("contract", {})
    if str(locked_contract.get("sha256", "")).lower() != sha256_file(contract_path):
        raise RuntimeError("06e0 lock does not match the current 06e0 contract SHA.")

    if contract.get("status") != (
        "FROZEN_BEFORE_V1_BIOLOGICAL_ALIGNMENT_AND_RANDOM_CONTROLS"
    ):
        raise RuntimeError("06e0 contract status mismatch.")

    rep = contract.get("frozen_representation", {})
    if rep.get("model_id") != EXPECTED["model_id"]:
        raise RuntimeError("06e0 model mismatch.")
    if rep.get("profile_id") != EXPECTED["profile_id"]:
        raise RuntimeError("06e0 profile mismatch.")
    if int(rep.get("n_cases")) != EXPECTED["n_cases"]:
        raise RuntimeError("06e0 case count mismatch.")
    if int(rep.get("primary_seed")) != EXPECTED["primary_seed"]:
        raise RuntimeError("06e0 primary seed mismatch.")
    if int(rep.get("shared_dim")) != EXPECTED["shared_dim"]:
        raise RuntimeError("06e0 shared dimension mismatch.")
    if str(rep.get("representation_sha256", "")).lower() != (
        EXPECTED["representation_sha256"]
    ):
        raise RuntimeError("06e0 representation SHA mismatch.")
    if rep.get("coordinate_stability_qualification") != "NOT_ESTABLISHED_FOR_V1":
        raise RuntimeError("06e0 V1 stability qualification boundary changed.")
    if rep.get("sable_mo_q10_or_retention_transfer") != "FORBIDDEN":
        raise RuntimeError("06e0 no longer forbids SABLE q10/retention transfer.")

    tech = contract.get("technical_boundary", {})
    if tech.get("status") != EXPECTED["technical_status"]:
        raise RuntimeError("06e0 technical-status boundary mismatch.")
    if tech.get("technical_unknown_means_nontechnical") is not False:
        raise RuntimeError("06e0 incorrectly treats technical UNKNOWN as clean.")
    if tech.get("technical_unknown_means_biological") is not False:
        raise RuntimeError("06e0 incorrectly treats technical UNKNOWN as biological.")

    programs = contract.get("primary_biological_programs", {})
    if programs.get("program_ids") != EXPECTED["programs"]:
        raise RuntimeError("06e0 program order/set mismatch.")
    if programs.get("strict_gene_counts") != EXPECTED["strict_gene_counts"]:
        raise RuntimeError("06e0 strict gene counts mismatch.")
    if programs.get("additional_programs_may_be_added_after_results") is not False:
        raise RuntimeError("06e0 allows post-result biological program expansion.")

    score = contract.get("program_score", {})
    if score.get("primary_variant") != EXPECTED["primary_score_variant"]:
        raise RuntimeError("06e0 primary score variant mismatch.")
    if float(score.get("minimum_program_coverage_fraction")) != 1.0:
        raise RuntimeError("06e0 no longer requires complete strict-gene coverage.")

    alignment = contract.get("primary_alignment", {})
    if alignment.get("primary_program_statistic") != (
        EXPECTED["primary_alignment_metric"]
    ):
        raise RuntimeError("06e0 primary alignment statistic mismatch.")
    if int(alignment.get("n_coordinates")) != EXPECTED["shared_dim"]:
        raise RuntimeError("06e0 alignment coordinate count mismatch.")
    if int(alignment.get("n_programs")) != len(EXPECTED["programs"]):
        raise RuntimeError("06e0 alignment program count mismatch.")

    random_contract = contract.get("matched_random_program_controls", {})
    if random_contract.get("enabled") is not True:
        raise RuntimeError("06e0 matched random controls unexpectedly disabled.")
    if int(random_contract.get("panels_per_program")) != (
        EXPECTED["random_panels_per_program"]
    ):
        raise RuntimeError("06e0 random-panel count mismatch.")
    if int(random_contract.get("random_seed")) != EXPECTED["random_seed"]:
        raise RuntimeError("06e0 random seed mismatch.")
    if int(random_contract.get("variability_bins")) != EXPECTED["variability_bins"]:
        raise RuntimeError("06e0 variability-bin count mismatch.")
    if "maximizes over five coordinates" not in str(
        random_contract.get("selection_correction", "")
    ):
        raise RuntimeError("06e0 null no longer mirrors the five-coordinate search.")

    multiplicity = contract.get("multiplicity", {})
    if multiplicity.get("procedure") != "Benjamini-Hochberg":
        raise RuntimeError("06e0 multiplicity procedure mismatch.")
    if float(multiplicity.get("alpha")) != EXPECTED["multiplicity_alpha"]:
        raise RuntimeError("06e0 multiplicity alpha mismatch.")

    firewall = contract.get("interpretation_firewall", {})
    required_false = [
        "biology_may_change_selected_model",
        "biology_may_change_primary_seed",
        "biology_may_change_fixed_epochs",
        "biology_may_apply_SABLE_q10_or_stability_gate_to_V1",
        "biology_may_rescue_or_reject_V1_coordinates",
        "strong_alignment_may_be_called_stable_biological_factor",
    ]
    for key in required_false:
        if firewall.get(key) is not False:
            raise RuntimeError(f"06e0 interpretation firewall changed: {key}")
    if firewall.get("clinical_or_outcome_use") != "FORBIDDEN":
        raise RuntimeError("06e0 clinical/outcome firewall changed.")

    # Replay upstream Paper-5 lock identities recorded by 06e0.
    source_locks = contract.get("source_locks", {})
    for label, path, obj in [
        ("06c3", results_lock_path, results_lock),
        ("06d0", scope_lock_path, scope_lock),
        ("06d1_v2", technical_lock_path, technical_lock),
    ]:
        expected_record = source_locks.get(label, {})
        if str(expected_record.get("sha256", "")).lower() != sha256_file(path):
            raise RuntimeError(f"06e0 recorded {label} SHA no longer matches.")

    if results_lock.get("status") != (
        "06C3_TARGET_V1_FULL_DATA_REFIT_RESULTS_FROZEN"
    ):
        raise RuntimeError("06c3 result lock is not frozen.")
    if scope_lock.get("status") != (
        "06D0_V1_POSTFIT_DOWNSTREAM_SCOPE_AUDIT_FROZEN"
    ):
        raise RuntimeError("06d0 scope lock is not frozen.")
    if technical_lock.get("status") != (
        "06D1_V2_V1_POSTFREEZE_TECHNICAL_STATUS_AUDIT_FROZEN"
    ):
        raise RuntimeError("06d1_v2 technical lock is not frozen.")

    if v1_failure_lock.get("status") != V1_FAILURE_LOCK_STATUS:
        raise RuntimeError("06e1r0 v1 failure-boundary lock is not frozen.")
    failed_runner = v1_failure_lock.get("failed_runner", {})
    if str(failed_runner.get("sha256", "")).lower() != V1_FAILED_RUNNER_SHA256:
        raise RuntimeError("06e1r0 failed-runner SHA mismatch.")
    observed_failure = v1_failure_lock.get("observed_failure", {})
    if observed_failure.get("observed_weight_columns") != REAL_STRICT_WEIGHT_COLUMNS:
        raise RuntimeError("06e1r0 observed strict-weight schema changed.")
    if observed_failure.get("repair_semantics") != (
        "primary strict_signed_mean must use sign(risk_oriented_loading)"
    ):
        raise RuntimeError("06e1r0 repair semantics changed.")

    biological_inputs = contract.get(
        "biological_input_identities_for_future_execution",
        {},
    )
    if set(biological_inputs) != {
        "paper4_frozen_program_manifest",
        "paper4_frozen_strict_gene_weights",
        "paper4_frozen_scoring_specification",
        "paper4_target_expression",
    }:
        raise RuntimeError("06e0 biological input identity set changed.")

    if touch_biological_files:
        # This is permitted only after source authorization and must happen
        # before parsing any biological file.
        project_root = paper5.parent
        for label, record in biological_inputs.items():
            path = project_root / str(record.get("path"))
            require_file(path)
            expected_sha = str(
                record.get("expected_sha256_for_future_execution", "")
            ).lower()
            observed_sha = sha256_file(path)
            if observed_sha != expected_sha:
                raise RuntimeError(
                    f"Frozen biological input SHA mismatch for {label}. "
                    f"Expected={expected_sha} Observed={observed_sha}"
                )
            if int(record.get("size_bytes")) != path.stat().st_size:
                raise RuntimeError(
                    f"Frozen biological input size mismatch for {label}."
                )

    return {
        "contract_path": contract_path,
        "contract_sha256": sha256_file(contract_path),
        "lock_path": lock_path,
        "lock_sha256": sha256_file(lock_path),
        "contract": contract,
        "lock": lock,
        "results_lock_path": results_lock_path,
        "results_lock": results_lock,
        "scope_lock_path": scope_lock_path,
        "technical_lock_path": technical_lock_path,
        "v1_failure_lock_path": v1_failure_lock_path,
        "v1_failure_lock_sha256": sha256_file(v1_failure_lock_path),
        "biological_inputs": biological_inputs,
    }


# ---------------------------------------------------------------------
# Source authorization gate
# ---------------------------------------------------------------------

def load_authorization(
    paper5: Path,
    runner_sha256: str,
) -> dict[str, Any] | None:
    path = (
        paper5 / "method_contract" / "06e1a_v2"
        / "target_v1_biology_runner_v2_source_authorization_lock.json"
    )
    if not path.is_file():
        return None

    lock = load_json(path)
    if lock.get("status") != AUTH_STATUS:
        raise RuntimeError("06e1a_v2 authorization status mismatch.")

    source = lock.get("runner_source", {})
    if str(source.get("sha256", "")).lower() != runner_sha256.lower():
        raise RuntimeError("06e1a_v2 authorization belongs to a different runner SHA.")
    if source.get("script_version") != SCRIPT_VERSION:
        raise RuntimeError("06e1a_v2 authorization SCRIPT_VERSION mismatch.")

    preflight = lock.get("synthetic_selftest", {})
    if preflight.get("status") != PREAUTH_STATUS:
        raise RuntimeError("06e1a_v2 authorization lacks the exact synthetic PASS.")

    return lock


# ---------------------------------------------------------------------
# Math / schema primitives
# ---------------------------------------------------------------------

def average_ranks(x: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(x, dtype=np.float64))
    return series.rank(method="average", na_option="keep").to_numpy(
        dtype=np.float64
    )


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = safe_float_array(x, label="Spearman x")
    y = safe_float_array(y, label="Spearman y")
    if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise RuntimeError("Spearman inputs must be equal-length vectors.")
    if x.shape[0] < 3:
        raise RuntimeError("Spearman requires at least 3 observations.")

    rx = average_ranks(x)
    ry = average_ranks(y)
    sx = np.std(rx, ddof=1)
    sy = np.std(ry, ddof=1)
    if not np.isfinite(sx) or sx <= 0.0:
        raise RuntimeError("Spearman x ranks are constant/invalid.")
    if not np.isfinite(sy) or sy <= 0.0:
        raise RuntimeError("Spearman y ranks are constant/invalid.")

    rho = float(np.corrcoef(rx, ry)[0, 1])
    if not np.isfinite(rho):
        raise RuntimeError("Spearman correlation is non-finite.")
    return rho


def standardize_ddof1(x: np.ndarray, *, label: str) -> np.ndarray:
    x = safe_float_array(x, label=label)
    mean = np.mean(x, axis=0)
    sd = np.std(x, axis=0, ddof=1)
    if np.any(~np.isfinite(sd)) or np.any(sd <= 0.0):
        raise RuntimeError(f"{label}: zero/non-finite ddof1 scale.")
    z = (x - mean) / sd
    if not np.isfinite(z).all():
        raise RuntimeError(f"{label}: standardized values are non-finite.")
    return z


def compute_signed_mean_program_score(
    expression: pd.DataFrame,
    ordered_genes: list[str],
    ordered_signs: np.ndarray,
) -> np.ndarray:
    """
    Frozen Paper-4 primary strict signed-mean score:
      1. require every frozen strict gene;
      2. forbid +/-inf;
      3. median-impute missing expression within the exact analysis cohort;
      4. gene-wise z-standardize with ddof=1;
      5. multiply by sign(risk_oriented_loading);
      6. mean across genes;
      7. z-standardize the module score with ddof=1.
    """
    if len(ordered_genes) != len(ordered_signs):
        raise RuntimeError("Program genes/signs length mismatch.")
    if len(set(ordered_genes)) != len(ordered_genes):
        raise RuntimeError("Program contains duplicated strict human genes.")

    missing = [g for g in ordered_genes if g not in expression.columns]
    if missing:
        raise RuntimeError(
            "Primary program lacks complete strict-gene coverage: "
            + ",".join(missing[:20])
        )

    x = expression.loc[:, ordered_genes].to_numpy(dtype=np.float64)
    if np.isinf(x).any():
        raise RuntimeError("Program expression contains infinite values.")

    if np.isnan(x).any():
        with np.errstate(invalid="ignore"):
            medians = np.nanmedian(x, axis=0)
        if not np.isfinite(medians).all():
            raise RuntimeError(
                "Program expression median imputation is undefined "
                "for at least one strict gene."
            )
        rows, cols = np.where(np.isnan(x))
        x = x.copy()
        x[rows, cols] = medians[cols]

    z = standardize_ddof1(
        x,
        label="program gene expression",
    )
    signs = np.asarray(ordered_signs, dtype=np.float64)
    if not np.isfinite(signs).all():
        raise RuntimeError("Program signs are non-finite.")
    if not np.all(np.isin(signs, [-1.0, 1.0])):
        raise RuntimeError("Program signs must be exactly -1 or +1.")

    raw = np.mean(
        z * signs.reshape(1, -1),
        axis=1,
    )
    score = standardize_ddof1(
        raw.reshape(-1, 1),
        label="program signed-mean score",
    )[:, 0]
    return score


def resolve_weight_schema(
    weights: pd.DataFrame,
) -> dict[str, str]:
    """
    Accept only the exact 12-column frozen Paper-4 strict-weight schema recorded
    by 06e1r0. Primary strict_signed_mean orientation is
    sign(risk_oriented_loading).
    """
    columns = list(weights.columns)
    if columns != REAL_STRICT_WEIGHT_COLUMNS:
        raise RuntimeError(
            "Frozen strict-weight schema mismatch. "
            f"Expected={REAL_STRICT_WEIGHT_COLUMNS} Observed={columns}"
        )
    return {
        "module": "module_label",
        "gene": "human_gene_symbol",
        "risk_orientation": RISK_ORIENTATION_COLUMN,
    }


def extract_program_definitions(
    weights: pd.DataFrame,
    schema: dict[str, str],
) -> dict[str, dict[str, Any]]:
    module_col = schema["module"]
    gene_col = schema["gene"]
    risk_col = schema["risk_orientation"]

    output: dict[str, dict[str, Any]] = {}

    for program in EXPECTED["programs"]:
        part = weights.loc[
            weights[module_col].astype(str).eq(program)
        ].copy()

        expected_n = EXPECTED["strict_gene_counts"][program]
        if part.shape[0] != expected_n:
            raise RuntimeError(
                f"{program}: strict weight-row count changed. "
                f"Expected={expected_n} Observed={part.shape[0]}"
            )

        genes = [
            normalize_gene_symbol(x)
            for x in part[gene_col].tolist()
        ]
        if any(not g for g in genes):
            raise RuntimeError(f"{program}: empty human gene symbol.")
        if len(set(genes)) != len(genes):
            raise RuntimeError(f"{program}: duplicate human gene symbol.")

        risk_loading = pd.to_numeric(
            part[risk_col],
            errors="coerce",
        ).to_numpy(dtype=np.float64)

        if not np.isfinite(risk_loading).all():
            raise RuntimeError(
                f"{program}: non-finite risk_oriented_loading."
            )
        signs = np.sign(risk_loading)
        if np.any(signs == 0.0):
            raise RuntimeError(
                f"{program}: zero risk_oriented_loading cannot define "
                "the frozen strict signed-mean orientation."
            )

        strict_values = (
            part["is_strict_mapping"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        if not strict_values.isin({"true", "1"}).all():
            raise RuntimeError(
                f"{program}: non-strict row found in frozen strict-weight table."
            )

        output[program] = {
            "genes": genes,
            "signs": signs.astype(np.float64),
            "risk_oriented_loading":
                risk_loading.astype(np.float64),
        }

    return output


def deterministic_variability_bins(
    variances: dict[str, float],
    n_bins: int,
) -> dict[str, int]:
    if n_bins < 2:
        raise RuntimeError("At least two variability bins are required.")
    items = []
    for gene, var in variances.items():
        if not np.isfinite(var) or var <= 0.0:
            continue
        items.append((float(var), str(gene)))

    if len(items) < n_bins:
        raise RuntimeError("Too few eligible genes for requested variability bins.")

    # Stable scientific definition:
    # primary key variance ascending, deterministic tie-break gene symbol ascending.
    items.sort(key=lambda t: (t[0], t[1]))
    n = len(items)

    bins: dict[str, int] = {}
    for rank, (_, gene) in enumerate(items):
        b = min(n_bins - 1, (rank * n_bins) // n)
        bins[gene] = int(b)

    return bins


def sample_matched_panel(
    *,
    anchor_genes: list[str],
    anchor_signs: np.ndarray,
    bins: dict[str, int],
    candidate_genes: list[str],
    rng: np.random.Generator,
    n_bins: int,
) -> tuple[list[str], np.ndarray, list[dict[str, Any]]]:
    if len(anchor_genes) != len(anchor_signs):
        raise RuntimeError("Matched-panel anchor gene/sign mismatch.")

    candidate_by_bin: dict[int, list[str]] = {
        b: [] for b in range(n_bins)
    }
    for gene in candidate_genes:
        if gene in bins:
            candidate_by_bin[bins[gene]].append(gene)

    # Deterministic input order before RNG choice.
    for b in candidate_by_bin:
        candidate_by_bin[b] = sorted(candidate_by_bin[b])

    used: set[str] = set()
    chosen: list[str] = []
    audit: list[dict[str, Any]] = []

    for position, (anchor_gene, sign) in enumerate(
        zip(anchor_genes, anchor_signs),
        start=1,
    ):
        if anchor_gene not in bins:
            raise RuntimeError(
                f"Anchor gene is not eligible for variance matching: {anchor_gene}"
            )
        target_bin = bins[anchor_gene]

        selected_gene: str | None = None
        selected_bin: int | None = None
        fallback_distance: int | None = None

        for distance in range(0, n_bins):
            bins_to_try = []
            left = target_bin - distance
            right = target_bin + distance
            if left >= 0:
                bins_to_try.append(left)
            if right < n_bins and right != left:
                bins_to_try.append(right)

            pool = sorted(
                gene
                for b in bins_to_try
                for gene in candidate_by_bin[b]
                if gene not in used
            )
            if pool:
                index = int(rng.integers(0, len(pool)))
                selected_gene = pool[index]
                selected_bin = bins[selected_gene]
                fallback_distance = distance
                break

        if selected_gene is None:
            raise RuntimeError(
                "Unable to construct an exact-size matched random panel."
            )

        used.add(selected_gene)
        chosen.append(selected_gene)
        audit.append(
            {
                "position": position,
                "anchor_gene": anchor_gene,
                "anchor_bin": target_bin,
                "random_gene": selected_gene,
                "random_bin": selected_bin,
                "fallback_distance": fallback_distance,
                "sign": float(sign),
            }
        )

    if len(chosen) != len(anchor_genes):
        raise RuntimeError("Matched random panel size mismatch.")
    if len(set(chosen)) != len(chosen):
        raise RuntimeError("Matched random panel sampled a duplicate gene.")

    return chosen, np.asarray(anchor_signs, dtype=np.float64), audit


def coordinate_correlations(
    score: np.ndarray,
    latent: np.ndarray,
) -> tuple[list[float], float, int]:
    score = safe_float_array(score, label="program score")
    latent = safe_float_array(latent, label="V1 latent representation")

    if latent.ndim != 2 or latent.shape[1] != EXPECTED["shared_dim"]:
        raise RuntimeError("V1 latent representation must have five coordinates.")
    if latent.shape[0] != score.shape[0]:
        raise RuntimeError("Program score / latent sample count mismatch.")

    correlations = [
        spearman_rho(score, latent[:, j])
        for j in range(latent.shape[1])
    ]
    abs_values = np.abs(np.asarray(correlations, dtype=np.float64))
    best_index = int(np.argmax(abs_values))
    return correlations, float(abs_values[best_index]), best_index


def empirical_plus_one_p(
    observed: float,
    null_values: np.ndarray,
) -> tuple[float, int]:
    null_values = safe_float_array(
        null_values,
        label="matched-random null values",
    )
    if not np.isfinite(observed):
        raise RuntimeError("Observed program statistic is non-finite.")
    count = int(np.sum(null_values >= observed))
    p = (1.0 + count) / (1.0 + null_values.size)
    return float(p), count


def benjamini_hochberg(
    p_values: list[float],
) -> list[float]:
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise RuntimeError("BH requires a non-empty p-value vector.")
    if np.any(~np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise RuntimeError("BH received invalid p-values.")

    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    m = p.size
    q_ranked = np.empty(m, dtype=np.float64)

    running = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        candidate = ranked[i] * m / rank
        running = min(running, candidate)
        q_ranked[i] = min(1.0, running)

    q = np.empty(m, dtype=np.float64)
    q[order] = q_ranked
    return [float(x) for x in q]


# ---------------------------------------------------------------------
# Synthetic preauthorization self-test
# ---------------------------------------------------------------------

def run_math_selftest() -> dict[str, Any]:
    rng = np.random.default_rng(20261007)

    n = 36
    genes = [f"G{i:03d}" for i in range(120)]
    expression = pd.DataFrame(
        rng.normal(size=(n, len(genes))),
        columns=genes,
        index=[f"TARGET-40-SYN{i:03d}" for i in range(n)],
    )

    synthetic_programs = {
        "M34": genes[0:12],
        "M40": genes[12:24],
        "M11": genes[24:31],
        "M24": genes[31:38],
    }
    synthetic_signs = {
        key: np.where(
            np.arange(len(value)) % 2 == 0,
            1.0,
            -1.0,
        )
        for key, value in synthetic_programs.items()
    }

    # Build synthetic V1 latent coordinates with one strong positive and one
    # strong negative association to prove sign-invariant best-coordinate logic.
    latent = rng.normal(size=(n, 5))
    score_m34 = compute_signed_mean_program_score(
        expression,
        synthetic_programs["M34"],
        synthetic_signs["M34"],
    )
    score_m40 = compute_signed_mean_program_score(
        expression,
        synthetic_programs["M40"],
        synthetic_signs["M40"],
    )
    latent[:, 1] = score_m34 + rng.normal(scale=0.05, size=n)
    latent[:, 3] = -score_m40 + rng.normal(scale=0.05, size=n)

    # Exact real 12-column strict-weight schema fixture captured by 06e1r0.
    weights_fixture = pd.DataFrame(
        {
            "module_label": ["M34", "M34"],
            "canine_gene": ["DOG_A", "DOG_B"],
            "canine_gene_symbol": ["A", "B"],
            "human_gene_symbol": ["A", "B"],
            "ortholog_qc_status": [
                "strict_symbol_concordant_one_to_one",
                "strict_symbol_concordant_one_to_one",
            ],
            "raw_pca_loading": [0.2, 0.3],
            "risk_oriented_loading": [0.2, -0.3],
            "absolute_risk_loading": [0.2, 0.3],
            "is_strict_mapping": [True, True],
            "is_broad_mapping": [True, True],
            "human_symbol_duplicate_count": [1, 1],
            "normalized_abs_sum_weight": [0.4, 0.6],
        },
        columns=REAL_STRICT_WEIGHT_COLUMNS,
    )
    schema = resolve_weight_schema(weights_fixture)
    if schema != {
        "module": "module_label",
        "gene": "human_gene_symbol",
        "risk_orientation": "risk_oriented_loading",
    }:
        raise RuntimeError(
            "Synthetic exact real weight-schema resolution failed."
        )

    synthetic_risk_signs = np.sign(
        weights_fixture[
            "risk_oriented_loading"
        ].to_numpy(dtype=np.float64)
    )
    if synthetic_risk_signs.tolist() != [1.0, -1.0]:
        raise RuntimeError(
            "Synthetic sign(risk_oriented_loading) test failed."
        )

    missing_fixture = pd.DataFrame(
        {
            "X": [1.0, 2.0, np.nan, 4.0, 5.0],
            "Y": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
    missing_score = compute_signed_mean_program_score(
        missing_fixture,
        ["X", "Y"],
        np.asarray([1.0, -1.0]),
    )
    if (
        missing_score.shape != (5,)
        or not np.isfinite(missing_score).all()
    ):
        raise RuntimeError(
            "Synthetic Paper-4 median-imputation scoring test failed."
        )

    # Case normalization fixture.
    if normalize_target_case_id("prefix_TARGET-40-ABC123-01A_suffix") != (
        "TARGET-40-ABC123"
    ):
        raise RuntimeError("Synthetic TARGET case normalization failed.")

    variances = {
        gene: float(np.var(expression[gene].to_numpy(dtype=float), ddof=1))
        for gene in genes
    }
    bins = deterministic_variability_bins(variances, n_bins=10)

    exclude = set().union(*synthetic_programs.values())
    candidates = sorted(set(genes) - exclude)

    smoke_panels = 40
    all_program_rows = []
    fallback_total = 0

    for program in ["M34", "M40", "M11", "M24"]:
        observed_score = compute_signed_mean_program_score(
            expression,
            synthetic_programs[program],
            synthetic_signs[program],
        )
        correlations, observed, best = coordinate_correlations(
            observed_score,
            latent,
        )

        null_stats = []
        for panel_index in range(smoke_panels):
            panel_genes, panel_signs, audit = sample_matched_panel(
                anchor_genes=synthetic_programs[program],
                anchor_signs=synthetic_signs[program],
                bins=bins,
                candidate_genes=candidates,
                rng=rng,
                n_bins=10,
            )
            fallback_total += sum(
                int(row["fallback_distance"] > 0)
                for row in audit
            )
            random_score = compute_signed_mean_program_score(
                expression,
                panel_genes,
                panel_signs,
            )
            _, null_stat, _ = coordinate_correlations(
                random_score,
                latent,
            )
            null_stats.append(null_stat)

        p, exceed = empirical_plus_one_p(
            observed,
            np.asarray(null_stats),
        )

        all_program_rows.append(
            {
                "program": program,
                "observed": observed,
                "best_coordinate_index": best,
                "correlations": correlations,
                "smoke_null_n": len(null_stats),
                "smoke_empirical_p": p,
                "smoke_null_exceedances": exceed,
            }
        )

    # Expected strong synthetic placement for the deliberately embedded signals.
    m34 = next(x for x in all_program_rows if x["program"] == "M34")
    m40 = next(x for x in all_program_rows if x["program"] == "M40")
    if m34["best_coordinate_index"] != 1 or m34["observed"] < 0.90:
        raise RuntimeError("Synthetic M34 positive-coordinate recovery failed.")
    if m40["best_coordinate_index"] != 3 or m40["observed"] < 0.90:
        raise RuntimeError("Synthetic M40 sign-invariant recovery failed.")

    # BH known-shape test.
    q = benjamini_hochberg([0.001, 0.02, 0.04, 0.9])
    if len(q) != 4 or not all(0.0 <= x <= 1.0 for x in q):
        raise RuntimeError("Synthetic BH test failed.")

    # Explicit fallback-path fixture: bin 0 has no candidate, nearest bin 1 does.
    fallback_bins = {"ANCHOR": 0, "C1": 1, "C2": 1, "C3": 2}
    _, _, fallback_audit = sample_matched_panel(
        anchor_genes=["ANCHOR"],
        anchor_signs=np.asarray([1.0]),
        bins=fallback_bins,
        candidate_genes=["C1", "C2", "C3"],
        rng=np.random.default_rng(123),
        n_bins=3,
    )
    if fallback_audit[0]["fallback_distance"] != 1:
        raise RuntimeError("Synthetic nearest-bin fallback test failed.")

    return {
        "status": "PASS",
        "n_samples": n,
        "n_genes": len(genes),
        "n_coordinates": 5,
        "n_programs": 4,
        "smoke_random_panels_per_program": smoke_panels,
        "smoke_random_panel_count_is_scientific_setting": False,
        "program_results": all_program_rows,
        "fallback_draws_exercised": fallback_total,
        "explicit_nearest_bin_fallback_test": "PASS",
        "exact_real_weight_schema_resolution": schema,
        "risk_oriented_loading_sign_test": "PASS",
        "median_imputation_scoring_test": "PASS",
        "target_case_normalization": "PASS",
        "bh_test": "PASS",
        "sign_invariant_best_coordinate_test": "PASS",
    }


def synthetic_selftest(
    paper5: Path,
    runner_sha256: str,
) -> None:
    print("=" * 104)
    print(
        "Paper 5 - 06e1 V1 biology runner v2 synthetic "
        "preauthorization self-test"
    )
    print("=" * 104)
    print("Script version:", SCRIPT_VERSION)
    print("Runner SHA-256:", runner_sha256)
    print("Execution mode: SYNTHETIC_ONLY_PREAUTHORIZATION")
    print("Real biological input hashing/parsing: FORBIDDEN")
    print("V1 numeric representation array read: FORBIDDEN")
    print("TARGET clinical/outcomes access: FORBIDDEN")

    state = verify_06e0_chain(
        paper5,
        touch_biological_files=False,
    )
    math_audit = run_math_selftest()

    preflight_dir = paper5 / "method_contract" / "06e1_preflight_v2"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    report_path = (
        preflight_dir
        / "v1_biological_alignment_runner_synthetic_selftest_v2.json"
    )

    if report_path.exists():
        existing = load_json(report_path)
        if (
            existing.get("status") == PREAUTH_STATUS
            and str(existing.get("runner_sha256", "")).lower()
            == runner_sha256.lower()
            and existing.get("script_version") == SCRIPT_VERSION
        ):
            print("Existing exact synthetic self-test report: PASS")
            print("Real biological inputs accessed by this run: NO")
            print("V1 numeric representation array read by this run: NO")
            print("=" * 104)
            print("06e1 V1 biology runner v2 synthetic self-test: PASS")
            print("=" * 104)
            print(
                "Self-test report:",
                relative_to_paper5(report_path, paper5),
            )
            print(
                "Next: freeze this exact source SHA in a separate "
                "06e1a_v1 source authorization."
            )
            print("=" * 104)
            return

        raise RuntimeError(
            "A 06e1 v1 preflight report already exists but does not belong "
            "to this exact runner source. Do not overwrite it; use a new "
            "versioned runner."
        )

    payload = {
        "status": PREAUTH_STATUS,
        "script_version": SCRIPT_VERSION,
        "runner_sha256": runner_sha256,
        "created_at_utc": utc_now_iso(),
        "execution_mode": "SYNTHETIC_ONLY_PREAUTHORIZATION",
        "06e0_identity": {
            "contract_path":
                relative_to_paper5(state["contract_path"], paper5),
            "contract_sha256": state["contract_sha256"],
            "lock_path":
                relative_to_paper5(state["lock_path"], paper5),
            "lock_sha256": state["lock_sha256"],
        },
        "v1_failure_boundary_identity": {
            "path":
                relative_to_paper5(
                    state["v1_failure_lock_path"],
                    paper5,
                ),
            "sha256":
                state["v1_failure_lock_sha256"],
            "failed_runner_sha256":
                V1_FAILED_RUNNER_SHA256,
            "observed_weight_columns":
                REAL_STRICT_WEIGHT_COLUMNS,
            "repair_semantics":
                "primary strict_signed_mean must use "
                "sign(risk_oriented_loading)",
        },
        "synthetic_math_and_schema_audit": math_audit,
        "scientific_state": {
            "real_biological_input_hashed": False,
            "real_biological_input_parsed": False,
            "v1_numeric_representation_array_read": False,
            "target_clinical_or_outcomes_read": False,
            "target_model_fit_or_refit": False,
            "biological_alignment_computed_on_real_data": False,
            "random_controls_computed_on_real_data": False,
        },
        "real_execution_boundary": {
            "authorization_required": True,
            "authorization_status": AUTH_STATUS,
            "authorization_namespace": "method_contract/06e1a_v2",
            "real_runner_must_verify_all_frozen_biological_hashes_before_parsing":
                True,
        },
    }
    write_json(report_path, payload)

    print("Exact 06e0 contract/lock identity: PASS")
    print("Synthetic strict signed-mean score path: PASS")
    print("Synthetic 4 x 5 Spearman path: PASS")
    print("Synthetic sign-invariant max-|rho| selection: PASS")
    print("Synthetic deterministic variance-bin path: PASS")
    print("Synthetic same-bin/nearest-bin random-panel path: PASS")
    print("Synthetic +1 empirical probability path: PASS")
    print("Synthetic BH multiplicity path: PASS")
    print("Synthetic exact real 12-column weight schema: PASS")
    print("Synthetic sign(risk_oriented_loading): PASS")
    print("Synthetic median-imputation score semantics: PASS")
    print("Synthetic TARGET case-id resolver: PASS")
    print("Real biological inputs accessed: NO")
    print("V1 numeric representation array read: NO")
    print("TARGET clinical/outcomes accessed: NO")
    print("=" * 104)
    print("06e1 V1 biology runner v2 synthetic self-test: PASS")
    print("=" * 104)
    print(
        "Self-test report:",
        relative_to_paper5(report_path, paper5),
    )
    print(
        "Next: freeze this exact source SHA in a separate "
        "06e1a_v1 source authorization."
    )
    print("=" * 104)


# ---------------------------------------------------------------------
# Real authorized data loading
# ---------------------------------------------------------------------

def load_case_order(
    paper5: Path,
    results_lock: dict[str, Any],
) -> list[str]:
    filename = "target_v1_full_data_case_order.csv"
    record = results_lock.get("result_files", {}).get(filename)
    if not isinstance(record, dict):
        raise RuntimeError("06c3 lock lacks frozen case-order file.")

    path = (
        paper5 / "results" / "target_representation"
        / "06c_full_data_v1" / filename
    )
    require_file(path)
    if sha256_file(path) != str(record.get("sha256", "")).lower():
        raise RuntimeError("Frozen V1 case-order SHA mismatch.")

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "case_id" not in rows[0]:
        raise RuntimeError("Frozen V1 case-order schema mismatch.")

    case_ids = [
        normalize_target_case_id(row["case_id"])
        for row in rows
    ]
    if len(case_ids) != EXPECTED["n_cases"]:
        raise RuntimeError("Frozen V1 case-order row count mismatch.")
    if len(set(case_ids)) != len(case_ids):
        raise RuntimeError("Frozen V1 case-order contains duplicate case ids.")
    return case_ids


def load_primary_representation(
    paper5: Path,
    results_lock: dict[str, Any],
) -> np.ndarray:
    filename = EXPECTED["representation_filename"]
    record = results_lock.get("result_files", {}).get(filename)
    if not isinstance(record, dict):
        raise RuntimeError("06c3 lock lacks primary V1 representation record.")

    path = (
        paper5 / "results" / "target_representation"
        / "06c_full_data_v1" / filename
    )
    require_file(path)

    observed = sha256_file(path)
    if observed != EXPECTED["representation_sha256"]:
        raise RuntimeError("Primary V1 representation SHA changed.")
    if observed != str(record.get("sha256", "")).lower():
        raise RuntimeError("06c3 lock representation SHA mismatch.")

    z = np.load(path, allow_pickle=False)
    z = safe_float_array(z, label="frozen V1 primary representation")
    if z.shape != (EXPECTED["n_cases"], EXPECTED["shared_dim"]):
        raise RuntimeError(
            f"Frozen V1 representation geometry changed: {z.shape}"
        )
    return z


def load_target_expression_78(
    expression_path: Path,
    case_order: list[str],
) -> pd.DataFrame:
    expression = pd.read_csv(
        expression_path,
        index_col=0,
        low_memory=False,
    )

    if expression.shape[0] < EXPECTED["n_cases"]:
        raise RuntimeError("Paper-4 TARGET expression has fewer than 78 rows.")

    normalized_index = [
        normalize_target_case_id(x)
        for x in expression.index
    ]
    if len(set(normalized_index)) != len(normalized_index):
        duplicates = pd.Series(normalized_index).value_counts()
        duplicates = duplicates[duplicates > 1].index.tolist()
        raise RuntimeError(
            "Paper-4 TARGET expression has duplicate normalized case ids: "
            + ",".join(duplicates[:20])
        )

    expression = expression.copy()
    expression.index = normalized_index
    missing_cases = [
        case for case in case_order
        if case not in expression.index
    ]
    if missing_cases:
        raise RuntimeError(
            "Paper-4 TARGET expression is missing frozen Paper-5 cases: "
            + ",".join(missing_cases[:20])
        )

    expression = expression.loc[case_order].copy()

    normalized_columns = [
        normalize_gene_symbol(c)
        for c in expression.columns
    ]
    if len(set(normalized_columns)) != len(normalized_columns):
        duplicates = pd.Series(normalized_columns).value_counts()
        duplicates = duplicates[duplicates > 1].index.tolist()
        raise RuntimeError(
            "Paper-4 TARGET expression has duplicated normalized gene symbols: "
            + ",".join(duplicates[:20])
        )
    expression.columns = normalized_columns

    # Numeric conversion is the first scientific-value parsing of this file.
    expression = expression.apply(
        pd.to_numeric,
        errors="raise",
    )
    x = expression.to_numpy(dtype=np.float64)
    if np.isinf(x).any():
        raise RuntimeError(
            "78-case TARGET expression contains infinite values."
        )

    return expression


def build_real_variability_state(
    expression: pd.DataFrame,
    program_defs: dict[str, dict[str, Any]],
) -> tuple[dict[str, float], dict[str, int], list[str]]:
    values = expression.to_numpy(dtype=np.float64)
    variances_vec = np.var(values, axis=0, ddof=1)

    variances = {
        gene: float(var)
        for gene, var in zip(expression.columns, variances_vec)
        if np.isfinite(var) and var > 0.0
    }

    all_anchor_genes = set()
    for program in EXPECTED["programs"]:
        all_anchor_genes.update(program_defs[program]["genes"])

    missing_variance = sorted(all_anchor_genes - set(variances))
    if missing_variance:
        raise RuntimeError(
            "Frozen strict program gene has zero/non-finite 78-case variance: "
            + ",".join(missing_variance[:20])
        )

    bins = deterministic_variability_bins(
        variances,
        n_bins=EXPECTED["variability_bins"],
    )
    candidates = sorted(set(variances) - all_anchor_genes)
    if len(candidates) < max(EXPECTED["strict_gene_counts"].values()):
        raise RuntimeError("Matched-random candidate pool is unexpectedly small.")

    return variances, bins, candidates


# ---------------------------------------------------------------------
# Real authorized scientific execution
# ---------------------------------------------------------------------

def run_real(
    paper5: Path,
    runner_sha256: str,
    authorization: dict[str, Any],
) -> None:
    print("=" * 104)
    print(
        "Paper 5 - AUTHORIZED outcome-blind V1 biological alignment "
        "+ matched random controls"
    )
    print("=" * 104)
    print("Script version:", SCRIPT_VERSION)
    print("Authorized runner SHA-256:", runner_sha256)
    print("Model:", EXPECTED["model_id"])
    print("Primary seed:", EXPECTED["primary_seed"])
    print("Programs:", EXPECTED["programs"])
    print("Random panels/program:", EXPECTED["random_panels_per_program"])
    print("Clinical/outcome access: FORBIDDEN")
    print("SABLE-MO q10/stability transfer to V1: FORBIDDEN")

    # First: verify all exact biological hashes BEFORE parsing any scientific row.
    state = verify_06e0_chain(
        paper5,
        touch_biological_files=True,
    )
    print("All frozen biological input SHA-256 identities: PASS")

    result_dir = (
        paper5 / "results" / "target_representation"
        / "06e1_biological_alignment_v2"
    )
    result_lock_dir = paper5 / "method_contract" / "06e2_v2"

    if result_dir.exists() and any(result_dir.iterdir()):
        raise RuntimeError(
            "Scientific 06e1 result directory already exists and is non-empty. "
            "Do not rerun or delete prior scientific outputs."
        )
    if result_lock_dir.exists() and any(result_lock_dir.iterdir()):
        raise RuntimeError(
            "06e2 result-lock directory already exists and is non-empty."
        )

    project_root = paper5.parent
    bio = state["biological_inputs"]

    weights_path = project_root / bio[
        "paper4_frozen_strict_gene_weights"
    ]["path"]
    expression_path = project_root / bio[
        "paper4_target_expression"
    ]["path"]

    # The other two frozen Paper-4 biological artifacts were already SHA-verified.
    # Their rows/values are not needed for the prespecified numeric calculation.
    # Avoid opening unnecessary scientific content.
    case_order = load_case_order(
        paper5,
        state["results_lock"],
    )

    # Validate the exact frozen real schema before reading the numeric V1 array.
    weights = pd.read_csv(
        weights_path,
        low_memory=False,
    )
    schema = resolve_weight_schema(weights)
    program_defs = extract_program_definitions(
        weights,
        schema,
    )
    print("Exact frozen 12-column strict-weight schema: PASS")
    print(
        "Primary score orientation = sign(risk_oriented_loading): PASS"
    )

    latent = load_primary_representation(
        paper5,
        state["results_lock"],
    )

    expression = load_target_expression_78(
        expression_path,
        case_order,
    )

    # Enforce exact 100% coverage for every primary program.
    for program in EXPECTED["programs"]:
        missing = [
            g for g in program_defs[program]["genes"]
            if g not in expression.columns
        ]
        if missing:
            raise RuntimeError(
                f"{program}: strict 78-case TARGET coverage is below 100%: "
                + ",".join(missing[:20])
            )

    variances, bins, candidate_genes = build_real_variability_state(
        expression,
        program_defs,
    )

    # Compute observed program scores and all 20 signed correlations.
    observed_scores: dict[str, np.ndarray] = {}
    observed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for program in EXPECTED["programs"]:
        definition = program_defs[program]
        score = compute_signed_mean_program_score(
            expression,
            definition["genes"],
            definition["signs"],
        )
        observed_scores[program] = score

        correlations, observed_stat, best_index = coordinate_correlations(
            score,
            latent,
        )

        for j, rho in enumerate(correlations, start=1):
            observed_rows.append(
                {
                    "program_id": program,
                    "coordinate_id": f"V1_SHARED_{j:02d}",
                    "spearman_rho": rho,
                    "absolute_spearman_rho": abs(rho),
                    "is_best_absolute_coordinate":
                        (j - 1) == best_index,
                    "technical_status":
                        EXPECTED["technical_status"],
                    "coordinate_stability_qualification":
                        "NOT_ESTABLISHED_FOR_V1",
                }
            )

        summary_rows.append(
            {
                "program_id": program,
                "n_strict_genes":
                    len(definition["genes"]),
                "coverage_fraction": 1.0,
                "observed_max_abs_spearman":
                    observed_stat,
                "best_coordinate_id":
                    f"V1_SHARED_{best_index + 1:02d}",
                "best_signed_spearman":
                    correlations[best_index],
                "technical_status":
                    EXPECTED["technical_status"],
            }
        )

    # Generate exact matched random programs with one shared RNG stream.
    rng = np.random.default_rng(EXPECTED["random_seed"])
    random_draw_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []

    for program in EXPECTED["programs"]:
        definition = program_defs[program]
        observed_stat = next(
            float(row["observed_max_abs_spearman"])
            for row in summary_rows
            if row["program_id"] == program
        )

        null_values = []

        for panel_index in range(
            1,
            EXPECTED["random_panels_per_program"] + 1,
        ):
            panel_genes, panel_signs, panel_audit = sample_matched_panel(
                anchor_genes=definition["genes"],
                anchor_signs=definition["signs"],
                bins=bins,
                candidate_genes=candidate_genes,
                rng=rng,
                n_bins=EXPECTED["variability_bins"],
            )

            random_score = compute_signed_mean_program_score(
                expression,
                panel_genes,
                panel_signs,
            )
            _, null_stat, best_index = coordinate_correlations(
                random_score,
                latent,
            )
            null_values.append(null_stat)

            fallback_draws = sum(
                int(row["fallback_distance"] > 0)
                for row in panel_audit
            )
            random_draw_rows.append(
                {
                    "program_id": program,
                    "panel_index": panel_index,
                    "null_max_abs_spearman": null_stat,
                    "best_coordinate_id":
                        f"V1_SHARED_{best_index + 1:02d}",
                    "fallback_draws": fallback_draws,
                }
            )

            for row in panel_audit:
                membership_rows.append(
                    {
                        "program_id": program,
                        "panel_index": panel_index,
                        **row,
                    }
                )

        null_array = np.asarray(null_values, dtype=np.float64)
        p, exceedances = empirical_plus_one_p(
            observed_stat,
            null_array,
        )

        row = next(
            x for x in summary_rows
            if x["program_id"] == program
        )
        row.update(
            {
                "random_panels": int(null_array.size),
                "null_median": float(np.median(null_array)),
                "null_q05": float(np.quantile(null_array, 0.05)),
                "null_q95": float(np.quantile(null_array, 0.95)),
                "null_exceedances": exceedances,
                "empirical_p_plus1": p,
                "fallback_draws_total": int(
                    sum(
                        x["fallback_draws"]
                        for x in random_draw_rows
                        if x["program_id"] == program
                    )
                ),
            }
        )

    p_values = [
        float(
            next(
                row["empirical_p_plus1"]
                for row in summary_rows
                if row["program_id"] == program
            )
        )
        for program in EXPECTED["programs"]
    ]
    q_values = benjamini_hochberg(p_values)

    for program, q in zip(EXPECTED["programs"], q_values):
        row = next(
            x for x in summary_rows
            if x["program_id"] == program
        )
        row["bh_q_4"] = q
        row["matched_random_support_q_le_0_05"] = (
            q <= EXPECTED["multiplicity_alpha"]
        )
        row["interpretation"] = (
            "program_aligned_v1_coordinate_under_technical_status_unknown"
            if q <= EXPECTED["multiplicity_alpha"]
            else
            "alignment_not_specific_against_prespecified_matched_random_controls"
        )
        row["stable_biological_factor_claim_allowed"] = False

    # Create scientific directories only after all computation has succeeded.
    result_dir.mkdir(parents=True, exist_ok=False)
    result_lock_dir.mkdir(parents=True, exist_ok=False)

    observed_path = result_dir / "v1_program_coordinate_correlations.csv"
    program_summary_path = result_dir / "v1_program_alignment_summary.csv"
    random_draws_path = result_dir / "v1_matched_random_control_draws.csv"
    memberships_path = (
        result_dir / "v1_matched_random_panel_memberships.csv.gz"
    )
    score_path = result_dir / "v1_frozen_program_scores_78.csv"
    summary_path = result_dir / "v1_biological_alignment_summary.json"

    pd.DataFrame(observed_rows).to_csv(
        observed_path,
        index=False,
    )
    pd.DataFrame(summary_rows).to_csv(
        program_summary_path,
        index=False,
    )
    pd.DataFrame(random_draw_rows).to_csv(
        random_draws_path,
        index=False,
    )
    pd.DataFrame(membership_rows).to_csv(
        memberships_path,
        index=False,
        compression="gzip",
    )

    score_table = pd.DataFrame(
        {
            "case_id": case_order,
            **{
                f"{program}__strict_signed_mean_z":
                    observed_scores[program]
                for program in EXPECTED["programs"]
            },
        }
    )
    score_table.to_csv(
        score_path,
        index=False,
    )

    scientific_summary = {
        "status":
            "06E1_V1_BIOLOGICAL_ALIGNMENT_RANDOM_CONTROLS_COMPLETE",
        "script_version": SCRIPT_VERSION,
        "runner_sha256": runner_sha256,
        "created_at_utc": utc_now_iso(),
        "model_id": EXPECTED["model_id"],
        "profile_id": EXPECTED["profile_id"],
        "primary_seed": EXPECTED["primary_seed"],
        "n_cases": EXPECTED["n_cases"],
        "shared_dim": EXPECTED["shared_dim"],
        "programs": EXPECTED["programs"],
        "v1_schema_failure_recovery": {
            "failed_v1_runner_sha256":
                V1_FAILED_RUNNER_SHA256,
            "failure_boundary_lock_sha256":
                state["v1_failure_lock_sha256"],
            "repair_only":
                "exact strict-weight schema adapter",
            "exact_weight_columns":
                REAL_STRICT_WEIGHT_COLUMNS,
            "primary_sign_source":
                "sign(risk_oriented_loading)",
            "scientific_alignment_or_null_rule_changed":
                False,
        },
        "technical_status":
            EXPECTED["technical_status"],
        "coordinate_stability_qualification":
            "NOT_ESTABLISHED_FOR_V1",
        "primary_alignment_metric":
            EXPECTED["primary_alignment_metric"],
        "matched_random_controls": {
            "panels_per_program":
                EXPECTED["random_panels_per_program"],
            "random_seed":
                EXPECTED["random_seed"],
            "variability_bins":
                EXPECTED["variability_bins"],
            "null_maximizes_over_same_five_coordinates":
                True,
            "empirical_probability":
                "plus_one",
            "multiplicity":
                "Benjamini-Hochberg across four programs",
        },
        "program_results": summary_rows,
        "firewalls": {
            "target_clinical_or_outcomes_read": False,
            "model_fit_or_refit": False,
            "representation_selection_changed": False,
            "primary_seed_changed": False,
            "SABLE_q10_or_stability_retention_applied_to_V1":
                False,
            "V1_coordinate_called_retained_or_stable": False,
            "technical_UNKNOWN_reinterpreted_as_clean": False,
        },
        "authorization": {
            "status": authorization.get("status"),
            "authorization_lock_sha256": sha256_file(
                paper5 / "method_contract" / "06e1a_v2"
                / "target_v1_biology_runner_v2_source_authorization_lock.json"
            ),
        },
        "06e0": {
            "contract_sha256": state["contract_sha256"],
            "lock_sha256": state["lock_sha256"],
        },
    }
    write_json(summary_path, scientific_summary)

    output_files = [
        observed_path,
        program_summary_path,
        random_draws_path,
        memberships_path,
        score_path,
        summary_path,
    ]
    result_files = {
        path.name: {
            "path": relative_to_paper5(path, paper5),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in output_files
    }

    result_lock = {
        "status": RESULT_STATUS,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "runner_source": {
            "path":
                "scripts/06e1_run_v1_biological_alignment_random_controls_v2.py",
            "sha256": runner_sha256,
            "script_version": SCRIPT_VERSION,
        },
        "selected_model": EXPECTED["model_id"],
        "primary_seed": EXPECTED["primary_seed"],
        "technical_status":
            EXPECTED["technical_status"],
        "programs": EXPECTED["programs"],
        "random_panels_per_program":
            EXPECTED["random_panels_per_program"],
        "result_files": result_files,
        "firewalls": scientific_summary["firewalls"],
        "rewrite_policy":
            "Frozen scientific result. Never overwrite/delete/rerun in place. "
            "Any scientifically necessary recovery requires a new versioned stage "
            "that preserves this lock and all current result files.",
    }

    result_lock_path = (
        result_lock_dir
        / "v1_biological_alignment_random_controls_results_lock_v2.json"
    )
    write_json(result_lock_path, result_lock)

    print("-" * 104)
    print("Frozen biological alignment results")
    print("-" * 104)
    for row in summary_rows:
        print(
            f"{row['program_id']}: "
            f"max|rho|={row['observed_max_abs_spearman']:.6f} "
            f"best={row['best_coordinate_id']} "
            f"p={row['empirical_p_plus1']:.6f} "
            f"q={row['bh_q_4']:.6f} "
            f"fallbacks={row['fallback_draws_total']}"
        )

    print("TARGET clinical/outcomes accessed: NO")
    print("TARGET model fit/refit: NO")
    print("SABLE-MO q10/stability applied to V1: NO")
    print("Technical UNKNOWN reinterpreted as clean: NO")
    print("=" * 104)
    print(
        "Paper 5 06e1 v2 V1 biological alignment + matched random controls: COMPLETE"
    )
    print("=" * 104)
    print("Summary:", relative_to_paper5(summary_path, paper5))
    print("Results lock:", relative_to_paper5(result_lock_path, paper5))
    print("=" * 104)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    paper5 = project_root / "paper5"
    runner_sha256 = own_sha256()

    authorization = load_authorization(
        paper5,
        runner_sha256,
    )

    if authorization is None:
        synthetic_selftest(
            paper5,
            runner_sha256,
        )
        return

    run_real(
        paper5,
        runner_sha256,
        authorization,
    )


if __name__ == "__main__":
    main()
