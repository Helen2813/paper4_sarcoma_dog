#!/usr/bin/env python3
"""
Paper 5 - 06e1r0 freeze the first real 06e1 v1 schema-failure boundary.

Purpose
-------
Preserve the exact provenance boundary of the first authorized real biological
attempt before any repair is made.

Observed console boundary:
- exact authorized 06e1 v1 runner entered the real branch;
- all frozen biological input SHA-256 identities passed;
- execution then failed in resolve_weight_schema() because the frozen Paper-4
  strict-weight table uses risk_oriented_loading rather than any sign/loading
  column anticipated by v1;
- no biological alignment or random-control statistic was computed;
- the v1 runner creates scientific output directories only after all observed
  and random-control computations succeed, so this failure occurred before
  scientific output creation.

This script is provenance-only. It DOES NOT open/hash/parse Paper-4 biological
files, does not read the numeric V1 representation, and does not compute any
scientific result.

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
    "06e1r0-freeze-v1-biology-runner-v1-schema-failure-boundary-v1-no-cli"
)

EXPECTED = {
    "runner_sha256":
        "51eface403ec7b41a6436289590f0cd4159ad4b6e167800052de8eb2f11bd180",
    "runner_script_version":
        "06e1-run-v1-biological-alignment-random-controls-v1-"
        "two-stage-source-authorized-no-cli",
    "authorization_status":
        "06E1A_V1_TARGET_V1_BIOLOGY_RUNNER_SOURCE_AUTHORIZED",
    "preauth_status":
        "06E1_V1_BIOLOGY_RUNNER_SYNTHETIC_SELFTEST_PASS",
    "failure_type": "RuntimeError",
    "failure_message_prefix":
        "Could not resolve frozen strict-weight sign or loading column.",
    "observed_weight_columns": [
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
    ],
    "repair_semantics":
        "primary strict_signed_mean must use sign(risk_oriented_loading)",
}


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


def function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    node = next(
        (
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == name
        ),
        None,
    )
    if node is None:
        raise RuntimeError(f"Runner function missing: {name}")
    return node


def main() -> None:
    print("=" * 104)
    print(
        "Paper 5 - freeze first authorized 06e1 v1 real-run "
        "schema-failure boundary"
    )
    print("=" * 104)
    print("Script version:", SCRIPT_VERSION)
    print("Paper-4 biological files opened/hashed here: NO")
    print("V1 numeric representation read here: NO")
    print("TARGET clinical/outcomes read here: NO")
    print("Scientific result computed here: NO")

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    paper5 = project_root / "paper5"

    runner = (
        paper5 / "scripts"
        / "06e1_run_v1_biological_alignment_random_controls_v1.py"
    )
    auth_path = (
        paper5 / "method_contract" / "06e1a_v1"
        / "target_v1_biology_runner_v1_source_authorization_lock.json"
    )
    preflight_path = (
        paper5 / "method_contract" / "06e1_preflight"
        / "v1_biological_alignment_runner_synthetic_selftest_v1.json"
    )

    for path in [runner, auth_path, preflight_path]:
        require_file(path)

    runner_sha = sha256_file(runner)
    if runner_sha != EXPECTED["runner_sha256"]:
        raise RuntimeError(
            "Historical 06e1 v1 runner SHA changed; refusing to freeze "
            "a failure boundary against a modified source."
        )

    auth = load_json(auth_path)
    if auth.get("status") != EXPECTED["authorization_status"]:
        raise RuntimeError("Historical 06e1a_v1 authorization status mismatch.")
    if str(
        auth.get("runner_source", {}).get("sha256", "")
    ).lower() != EXPECTED["runner_sha256"]:
        raise RuntimeError(
            "Historical 06e1a_v1 authorization belongs to another runner."
        )

    preflight = load_json(preflight_path)
    if preflight.get("status") != EXPECTED["preauth_status"]:
        raise RuntimeError("Historical 06e1 v1 synthetic preflight mismatch.")
    if str(preflight.get("runner_sha256", "")).lower() != (
        EXPECTED["runner_sha256"]
    ):
        raise RuntimeError("Historical preflight runner SHA mismatch.")

    source = runner.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source, filename=str(runner))
    real = function_node(tree, "run_real")
    real_source = ast.get_source_segment(source, real) or ""

    # Prove the source-order boundary that follows from the observed traceback.
    ordered_markers = [
        "touch_biological_files=True",
        "case_order = load_case_order(",
        "latent = load_primary_representation(",
        "weights = pd.read_csv(",
        "schema = resolve_weight_schema(weights)",
        "program_defs = extract_program_definitions(",
        "expression = load_target_expression_78(",
        "result_dir.mkdir(parents=True, exist_ok=False)",
    ]
    positions = {}
    for marker in ordered_markers:
        pos = real_source.find(marker)
        if pos < 0:
            raise RuntimeError(
                f"Historical v1 source marker missing: {marker}"
            )
        positions[marker] = pos

    if positions[ordered_markers[0]] >= positions[ordered_markers[1]]:
        raise RuntimeError("Historical hash-before-read source ordering changed.")
    if not (
        positions[ordered_markers[1]]
        < positions[ordered_markers[2]]
        < positions[ordered_markers[3]]
        < positions[ordered_markers[4]]
        < positions[ordered_markers[5]]
        < positions[ordered_markers[6]]
        < positions[ordered_markers[7]]
    ):
        raise RuntimeError(
            "Historical v1 real-branch source ordering no longer matches "
            "the observed failure boundary."
        )

    scientific_result_dir = (
        paper5 / "results" / "target_representation"
        / "06e1_biological_alignment_v1"
    )
    result_lock_dir = paper5 / "method_contract" / "06e2"

    result_files_present = (
        scientific_result_dir.exists()
        and any(scientific_result_dir.iterdir())
    )
    result_lock_files_present = (
        result_lock_dir.exists()
        and any(result_lock_dir.iterdir())
    )

    if result_files_present:
        raise RuntimeError(
            "06e1 v1 scientific result files exist. Stop and audit them; "
            "this zero-output failure-boundary lock is not applicable."
        )
    if result_lock_files_present:
        raise RuntimeError(
            "06e2 result-lock files exist. Stop and audit them."
        )

    out_dir = paper5 / "method_contract" / "06e1r0"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(
            f"06e1r0 directory already exists and is non-empty: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "status":
            "06E1R0_V1_FIRST_REAL_BIOLOGY_ATTEMPT_SCHEMA_FAILURE_FROZEN",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),

        "failed_runner": {
            "path":
                "scripts/06e1_run_v1_biological_alignment_random_controls_v1.py",
            "sha256": EXPECTED["runner_sha256"],
            "script_version": EXPECTED["runner_script_version"],
            "authorization_status": EXPECTED["authorization_status"],
            "synthetic_preflight_status": EXPECTED["preauth_status"],
        },

        "observed_failure": {
            "failure_type": EXPECTED["failure_type"],
            "failure_message_prefix":
                EXPECTED["failure_message_prefix"],
            "function": "resolve_weight_schema",
            "observed_weight_columns":
                EXPECTED["observed_weight_columns"],
            "root_cause":
                "v1 schema resolver did not recognize the frozen Paper-4 "
                "risk_oriented_loading column as the source of the signed "
                "strict human score orientation.",
            "scientific_rule_changed_by_repair": False,
            "repair_semantics":
                EXPECTED["repair_semantics"],
        },

        "boundary_inferred_from_observed_traceback_and_exact_source_order": {
            "all_frozen_biological_input_hashes_verified_before_failure":
                True,
            "frozen_case_order_loaded_before_failure": True,
            "v1_numeric_representation_loaded_before_failure": True,
            "strict_weight_csv_parsed_before_failure": True,
            "program_definitions_constructed": False,
            "target_expression_values_loaded": False,
            "program_scores_computed": False,
            "program_coordinate_correlations_computed": False,
            "matched_random_panels_generated": False,
            "empirical_probabilities_computed": False,
            "bh_q_values_computed": False,
            "scientific_result_directory_created": False,
            "06e2_result_lock_created": False,
            "clinical_or_outcome_values_read": False,
            "model_fit_or_refit": False,
        },

        "filesystem_replay": {
            "06e1_v1_scientific_result_directory_nonempty": False,
            "06e2_result_lock_directory_nonempty": False,
        },

        "recovery_policy": {
            "rerun_failed_v1": "FORBIDDEN",
            "delete_or_rewrite_failed_v1_history": "FORBIDDEN",
            "required_repair":
                "new versioned 06e1 v2 runner with exact real strict-weight "
                "schema support and a new synthetic/schema preflight",
            "required_before_next_real_biology_run": [
                "06e1 v2 synthetic preflight PASS",
                "separate exact-source authorization for 06e1 v2",
            ],
            "no_postresult_scientific_adaptation":
                "The repair is restricted to the schema adapter exposed by "
                "the pre-statistic failure. No biological alignment, null, "
                "threshold, program set, seed, multiplicity, or interpretation "
                "rule may change.",
        },

        "scientific_state_of_this_recovery_freeze": {
            "paper4_biological_files_opened_or_hashed": False,
            "v1_numeric_representation_read": False,
            "target_expression_read": False,
            "target_clinical_or_outcomes_read": False,
            "scientific_result_computed": False,
        },
    }

    lock_path = (
        out_dir
        / "v1_first_real_biology_attempt_schema_failure_boundary_lock.json"
    )
    lock_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Exact historical 06e1 v1 runner SHA: PASS")
    print("Exact historical 06e1a_v1 authorization: PASS")
    print("Exact historical synthetic preflight: PASS")
    print("Historical source-order failure boundary: PASS")
    print("Existing 06e1 v1 scientific result files: NO")
    print("Existing 06e2 result lock: NO")
    print("Program definitions constructed before failure: NO")
    print("TARGET expression values loaded before failure: NO")
    print("Program scores/correlations/random controls computed: NO")
    print("V1 representation had been loaded by failed v1 attempt: YES")
    print("Strict-weight CSV had been parsed by failed v1 attempt: YES")
    print("Clinical/outcome values read by failed v1 attempt: NO")
    print("Failed v1 rerun: FORBIDDEN")
    print("=" * 104)
    print(
        "Paper 5 06e1r0 first-real-run schema-failure boundary: FROZEN"
    )
    print("=" * 104)
    print(
        "Lock: method_contract/06e1r0/"
        "v1_first_real_biology_attempt_schema_failure_boundary_lock.json"
    )
    print(
        "Next: create a new v2 runner; do not modify or rerun the v1 runner."
    )
    print("=" * 104)


if __name__ == "__main__":
    main()
