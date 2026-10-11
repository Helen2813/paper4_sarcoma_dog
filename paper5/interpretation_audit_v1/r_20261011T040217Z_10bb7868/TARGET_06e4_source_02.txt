#!/usr/bin/env python3
"""
Paper 5 - 06e4a source authorization for the exact 06e4 V1 seed/subspace
biological-sensitivity runner.

This stage is metadata/source-only. It MUST NOT open real latent arrays, frozen
program-score rows, TARGET molecular data, outcomes, or clinical variables.

It authorizes exactly one runner SHA after:
- 06e3 sensitivity contract is already frozen;
- the exact runner has passed its synthetic-only preauthorization self-test;
- no 06e4 scientific sensitivity result or 06e5 result lock exists;
- static source ordering confirms authorization/result-collision guards precede
  the first real score/latent read.

No command-line arguments.
"""

from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = (
    "06e4a-freeze-v1-seed-subspace-runner-source-authorization-v1-no-cli"
)

AUTH_STATUS = "06E4A_V1_SEED_SUBSPACE_RUNNER_SOURCE_AUTHORIZED"

EXPECTED = {
    "runner_filename":
        "06e4_run_v1_seed_subspace_biology_sensitivity_v1.py",
    "runner_sha256":
        "97a8af048e38c16000f27a078ea976af1c894a21850ac2924a05e3b8c16cee86",
    "runner_script_version":
        "06e4-run-v1-seed-subspace-biology-sensitivity-v1-"
        "two-stage-source-authorized-readonly-no-cli",
    "contract_sha256":
        "cbac52ec634e566ccee070ec0116577388a23d4ceefd881fbe761b55a39e40a6",
    "contract_status":
        "FROZEN_BEFORE_V1_SEED_SUBSPACE_BIOLOGY_SENSITIVITY",
    "contract_lock_status":
        "06E3_V1_SEED_SUBSPACE_BIOLOGY_SENSITIVITY_CONTRACT_FROZEN",
    "preflight_status":
        "06E4_V1_SEED_SUBSPACE_RUNNER_SYNTHETIC_SELFTEST_PASS",
    "preflight_execution_mode":
        "SYNTHETIC_ONLY_PREAUTHORIZATION",
    "primary_seed": 1103,
    "reproducibility_seeds": [2207, 3301],
    "programs": ["M34", "M40", "M11", "M24"],
}


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


def read_assigned_string(tree: ast.AST, name: str) -> str:
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    return value.value
                # SCRIPT_VERSION is built by implicit string concatenation in parens;
                # ast still represents it as Constant.
    raise RuntimeError(f"Could not resolve assigned string {name!r} from runner AST.")


def require_no_files(path: Path, label: str) -> None:
    if not path.exists():
        return
    files = [p for p in path.rglob("*") if p.is_file()]
    if files:
        raise RuntimeError(
            f"{label} already contains files; refusing source authorization: "
            + ", ".join(str(p) for p in files[:20])
        )


def main() -> None:
    print("=" * 108)
    print("Paper 5 - freeze exact 06e4 seed/subspace runner source authorization")
    print("=" * 108)
    print("Script version:", SCRIPT_VERSION)
    print("Real V1 latent arrays opened/hashed in 06e4a: NO")
    print("Frozen biological score rows opened/hashed in 06e4a: NO")
    print("TARGET molecular values read in 06e4a: NO")
    print("TARGET outcomes/clinical read in 06e4a: NO")
    print("Model fit/refit in 06e4a: NO")
    print("Scientific seed/subspace results read in 06e4a: NO")

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    paper5 = project_root / "paper5"

    runner_path = paper5 / "scripts" / EXPECTED["runner_filename"]
    contract_path = (
        paper5 / "method_contract" / "06e3"
        / "v1_seed_subspace_biology_sensitivity_contract.json"
    )
    contract_lock_path = (
        paper5 / "method_contract" / "06e3"
        / "06e3_v1_seed_subspace_biology_sensitivity_contract_lock.json"
    )
    preflight_path = (
        paper5 / "method_contract" / "06e4_preflight"
        / "v1_seed_subspace_runner_synthetic_selftest_v1.json"
    )
    auth_dir = paper5 / "method_contract" / "06e4a"
    auth_lock_path = (
        auth_dir
        / "v1_seed_subspace_sensitivity_runner_source_authorization_lock.json"
    )
    scientific_result_dir = (
        paper5 / "results" / "target_representation"
        / "06e4_seed_subspace_sensitivity_v1"
    )
    result_lock_dir = paper5 / "method_contract" / "06e5"

    for path in [runner_path, contract_path, contract_lock_path, preflight_path]:
        require_file(path)

    if auth_dir.exists():
        raise RuntimeError(
            "06e4a authorization directory already exists. Do not rerun or "
            "overwrite authorization history; audit the existing directory instead."
        )

    require_no_files(
        scientific_result_dir,
        "06e4 scientific sensitivity result directory",
    )
    require_no_files(
        result_lock_dir,
        "06e5 result-lock directory",
    )

    # ------------------------------------------------------------------
    # Exact runner identity.
    # ------------------------------------------------------------------
    runner_sha = sha256_file(runner_path)
    if runner_sha != EXPECTED["runner_sha256"]:
        raise RuntimeError(
            f"Exact 06e4 runner SHA mismatch: {runner_sha} "
            f"!= {EXPECTED['runner_sha256']}"
        )

    runner_text = runner_path.read_text(encoding="utf-8")
    runner_tree = ast.parse(runner_text)
    observed_version = read_assigned_string(runner_tree, "SCRIPT_VERSION")
    if observed_version != EXPECTED["runner_script_version"]:
        raise RuntimeError(
            "06e4 runner SCRIPT_VERSION mismatch: "
            f"{observed_version!r}"
        )

    # ------------------------------------------------------------------
    # Exact 06e3 contract identity / lock binding.
    # ------------------------------------------------------------------
    contract_sha = sha256_file(contract_path)
    if contract_sha != EXPECTED["contract_sha256"]:
        raise RuntimeError(
            f"06e3 contract SHA mismatch: {contract_sha} "
            f"!= {EXPECTED['contract_sha256']}"
        )
    contract = load_json(contract_path)
    contract_lock = load_json(contract_lock_path)

    if contract.get("status") != EXPECTED["contract_status"]:
        raise RuntimeError("06e3 contract status mismatch.")
    if contract_lock.get("status") != EXPECTED["contract_lock_status"]:
        raise RuntimeError("06e3 contract-lock status mismatch.")
    if str(
        contract_lock.get("contract", {}).get("sha256", "")
    ).lower() != contract_sha:
        raise RuntimeError("06e3 contract lock does not bind exact contract SHA.")

    if int(
        contract.get("primary_result_boundary", {}).get("primary_seed", -1)
    ) != EXPECTED["primary_seed"]:
        raise RuntimeError("06e3 primary-seed boundary mismatch.")
    if contract.get("seed_roles", {}).get("2207") != (
        "MANDATORY_REPRODUCIBILITY_ONLY"
    ):
        raise RuntimeError("06e3 seed 2207 role changed.")
    if contract.get("seed_roles", {}).get("3301") != (
        "MANDATORY_REPRODUCIBILITY_ONLY"
    ):
        raise RuntimeError("06e3 seed 3301 role changed.")
    if contract.get("program_seed_sensitivity", {}).get("programs") != (
        EXPECTED["programs"]
    ):
        raise RuntimeError("06e3 program list/order changed.")
    if int(
        contract.get("program_seed_sensitivity", {}).get("n_cells", -1)
    ) != 12:
        raise RuntimeError("06e3 no longer requires all 12 program x seed cells.")

    # ------------------------------------------------------------------
    # Exact synthetic preflight identity/content.
    # ------------------------------------------------------------------
    preflight_sha = sha256_file(preflight_path)
    preflight = load_json(preflight_path)

    if preflight.get("status") != EXPECTED["preflight_status"]:
        raise RuntimeError("06e4 synthetic preflight status mismatch.")
    if preflight.get("script_version") != EXPECTED["runner_script_version"]:
        raise RuntimeError("06e4 synthetic preflight SCRIPT_VERSION mismatch.")
    if str(preflight.get("runner_sha256", "")).lower() != runner_sha:
        raise RuntimeError("06e4 synthetic preflight runner SHA mismatch.")
    if str(preflight.get("contract_sha256", "")).lower() != contract_sha:
        raise RuntimeError("06e4 synthetic preflight contract SHA mismatch.")
    if preflight.get("execution_mode") != EXPECTED["preflight_execution_mode"]:
        raise RuntimeError("06e4 preflight was not synthetic-only.")
    for key in [
        "real_numeric_latents_opened",
        "real_biological_score_rows_opened",
        "target_outcomes_or_clinical_read",
        "model_fit_or_refit",
    ]:
        if preflight.get(key) is not False:
            raise RuntimeError(f"06e4 preflight scientific firewall failed: {key}")

    tests = preflight.get("tests", {})
    required_pass_tests = [
        "canonical_correlation_sign_permutation_invariance",
        "canonical_correlation_rotation_invariance",
        "procrustes_sign_permutation_recovery",
        "procrustes_rotation_recovery",
        "hungarian_coordinate_matching",
        "max_abs_spearman_sign_permutation_invariance",
        "max_abs_spearman_rotation_sensitivity_fixture",
    ]
    for name in required_pass_tests:
        if tests.get(name) != "PASS":
            raise RuntimeError(f"06e4 synthetic test did not PASS: {name}")
    if int(tests.get("synthetic_program_coordinate_rows", -1)) != 20:
        raise RuntimeError("Synthetic program-coordinate geometry mismatch.")
    if int(tests.get("synthetic_program_summary_rows", -1)) != 4:
        raise RuntimeError("Synthetic program-summary geometry mismatch.")

    # ------------------------------------------------------------------
    # Static ordering audit of the exact runner source.
    # ------------------------------------------------------------------
    markers = {
        "auth_gate": "if not auth_path.exists():",
        "authorized_load": "auth = load_json(auth_path)",
        "result_guard": "if result_dir.exists() and any(result_dir.iterdir()):",
        "result_lock_guard":
            "if result_lock_dir.exists() and any(result_lock_dir.iterdir()):",
        "real_score_read": "scores = pd.read_csv(scores_path)",
        "real_latent_read": "np.load(rep_paths[seed], allow_pickle=False)",
        "primary_replay":
            'print("Primary seed-1103 4x5 biology replay: PASS [20/20]")',
        "all_12_guard": "raw_summary.shape[0] != 12",
        "subspace_pairs":
            "seed_pairs = [(1103, 2207), (1103, 3301), (2207, 3301)]",
        "no_new_pq_flag": '"new_seed_specific_p_or_q": False',
        "no_subspace_biology_flag":
            '"rotation_invariant_program_to_subspace_statistic_computed": False',
    }
    positions: dict[str, int] = {}
    for key, marker in markers.items():
        pos = runner_text.find(marker)
        if pos < 0:
            raise RuntimeError(
                f"Required 06e4 runner source marker missing: {key}"
            )
        positions[key] = pos

    first_real_read = min(
        positions["real_score_read"],
        positions["real_latent_read"],
    )
    for guard_key in [
        "auth_gate",
        "authorized_load",
        "result_guard",
        "result_lock_guard",
    ]:
        if positions[guard_key] >= first_real_read:
            raise RuntimeError(
                f"Runner static-order firewall failed: {guard_key} "
                "does not precede first real scientific read."
            )

    if runner_text.count("np.load(rep_paths[seed], allow_pickle=False)") != 1:
        raise RuntimeError(
            "Unexpected number of real latent np.load sites in exact runner."
        )
    if runner_text.count("scores = pd.read_csv(scores_path)") != 1:
        raise RuntimeError(
            "Unexpected number of real biological-score read sites in exact runner."
        )

    # No training backend should exist in this read-only runner.
    forbidden_training_markers = [
        "optimizer.step(",
        "loss.backward(",
        "train_fixed_epochs(",
        "torch.optim",
        ".fit(",
    ]
    for marker in forbidden_training_markers:
        if marker in runner_text:
            raise RuntimeError(
                f"Read-only 06e4 runner unexpectedly contains training marker: {marker}"
            )

    static_audit = {
        "authorization_gate_before_first_real_read": "PASS",
        "authorization_lock_loaded_before_first_real_read": "PASS",
        "scientific_result_collision_guard_before_first_real_read": "PASS",
        "result_lock_collision_guard_before_first_real_read": "PASS",
        "single_real_latent_load_site": "PASS",
        "single_real_program_score_read_site": "PASS",
        "primary_seed_1103_replay_required": "PASS",
        "all_12_raw_program_seed_cells_required": "PASS",
        "three_pair_seed_subspace_diagnostics_required": "PASS",
        "new_seed_specific_p_or_q_false": "PASS",
        "rotation_invariant_program_to_subspace_statistic_false": "PASS",
        "training_backend_absent": "PASS",
    }

    # ------------------------------------------------------------------
    # Freeze authorization only after all checks pass.
    # ------------------------------------------------------------------
    auth_dir.mkdir(parents=True, exist_ok=False)

    lock = {
        "status": AUTH_STATUS,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "scientific_state": {
            "numeric_v1_latent_arrays_opened_or_hashed": False,
            "biological_program_score_rows_opened_or_hashed": False,
            "target_molecular_values_read": False,
            "target_outcomes_or_clinical_read": False,
            "seed_subspace_result_computed": False,
            "program_seed_sensitivity_computed": False,
            "model_fit_or_refit": False,
        },
        "runner_source": {
            "path":
                "scripts/"
                "06e4_run_v1_seed_subspace_biology_sensitivity_v1.py",
            "sha256": runner_sha,
            "script_version": observed_version,
            "size_bytes": runner_path.stat().st_size,
        },
        "06e3_contract": {
            "path":
                "method_contract/06e3/"
                "v1_seed_subspace_biology_sensitivity_contract.json",
            "sha256": contract_sha,
            "status": contract.get("status"),
        },
        "06e3_contract_lock": {
            "path":
                "method_contract/06e3/"
                "06e3_v1_seed_subspace_biology_sensitivity_contract_lock.json",
            "sha256": sha256_file(contract_lock_path),
            "status": contract_lock.get("status"),
        },
        "synthetic_preflight": {
            "path":
                "method_contract/06e4_preflight/"
                "v1_seed_subspace_runner_synthetic_selftest_v1.json",
            "sha256": preflight_sha,
            "status": preflight.get("status"),
            "execution_mode": preflight.get("execution_mode"),
        },
        "authorized_scientific_execution": {
            "analysis_role":
                "POST_RESULT_SENSITIVITY_USING_PROSPECTIVELY_FROZEN_"
                "REPRODUCIBILITY_SEEDS",
            "primary_seed": 1103,
            "reproducibility_only_seeds": [2207, 3301],
            "programs": EXPECTED["programs"],
            "report_all_12_program_seed_cells": True,
            "pairwise_seed_pairs": [
                [1103, 2207],
                [1103, 3301],
                [2207, 3301],
            ],
            "primary_seed_1103_replay_required": True,
            "new_seed_specific_p_or_q": "FORBIDDEN",
            "matched_random_panel_recomputation": "FORBIDDEN",
            "rotation_invariant_program_to_subspace_biology_statistic":
                "FORBIDDEN",
            "seed_reselection_or_replacement": "FORBIDDEN",
            "seed_averaging_or_ensemble": "FORBIDDEN",
            "SABLE_q10_or_stability_transfer_to_V1": "FORBIDDEN",
            "target_outcome_or_clinical_access": "FORBIDDEN",
            "model_fit_or_refit": "FORBIDDEN",
        },
        "static_runner_audit": static_audit,
        "authorization_rule": (
            "Only the exact 06e4 runner SHA recorded here is authorized to "
            "consume the exact frozen 06e3 contract. Any source change requires "
            "a new versioned synthetic preflight and new source authorization "
            "before any real sensitivity computation."
        ),
        "rewrite_policy": (
            "06e4a is immutable after creation. Never overwrite or rerun this "
            "authorization in place."
        ),
    }
    auth_lock_path.write_text(
        json.dumps(lock, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Exact 06e4 runner SHA: PASS")
    print("Exact 06e4 runner SCRIPT_VERSION: PASS")
    print("Exact frozen 06e3 contract SHA: PASS")
    print("06e3 contract-lock binding: PASS")
    print("Exact synthetic preflight identity: PASS")
    print("Synthetic preflight scientific firewalls: PASS")
    print("Synthetic canonical/subspace tests: PASS")
    print("Synthetic Procrustes/Hungarian tests: PASS")
    print("Synthetic max-|rho| invariance/sensitivity tests: PASS")
    print("No pre-existing 06e4 scientific results: PASS")
    print("No pre-existing 06e5 result lock: PASS")
    print("Authorization gate before first real scientific read: PASS")
    print("Result collision guards before first real scientific read: PASS")
    print("Primary seed-1103 replay requirement: PASS")
    print("All 12 program x seed cells requirement: PASS")
    print("Three pairwise seed-subspace comparisons requirement: PASS")
    print("Training backend absent: PASS")
    print("New seed-specific p/q values: FORBIDDEN")
    print("Rotation-invariant program-to-subspace biology statistic: FORBIDDEN")
    print("Real V1 latent arrays opened/hashed in 06e4a: NO")
    print("Frozen biological score rows opened/hashed in 06e4a: NO")
    print("TARGET outcomes/clinical read in 06e4a: NO")
    print("Model fit/refit in 06e4a: NO")
    print("=" * 108)
    print("Paper 5 06e4a exact seed/subspace runner source authorization: FROZEN")
    print("=" * 108)
    print("Authorized runner SHA:", runner_sha)
    print("Synthetic preflight SHA:", preflight_sha)
    print(
        "Authorization lock: method_contract/06e4a/"
        "v1_seed_subspace_sensitivity_runner_source_authorization_lock.json"
    )
    print(
        "Next: run the exact same 06e4 runner unchanged ONCE. "
        "Do not rerun this authorizer."
    )
    print("=" * 108)


if __name__ == "__main__":
    main()
