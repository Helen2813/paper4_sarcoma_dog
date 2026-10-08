#!/usr/bin/env python3
"""
Paper 5 - 06e1a_v3 freeze exact 06e1 v3 biology-runner source authorization.

Authorization-only stage.

This authorizes only the exact SHA-256-locked 06e1 v3 runner after its
synthetic-only preauthorization PASS. It reads only Paper-5 method-contract /
provenance metadata and the runner source itself.

It DOES NOT:
- open/hash/parse Paper-4 biological inputs;
- read TARGET expression values;
- read the numeric V1 representation array;
- read TARGET clinical variables or outcomes;
- compute biological alignment;
- generate matched random controls;
- fit/refit/select any model.

It refuses authorization if v3 scientific outputs already exist.

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
    "06e1a-v3-freeze-v1-biology-runner-source-authorization-v1-no-cli"
)

EXPECTED = {
    "runner_sha256": "77fbe2d681ddaa906db71c93a654c5f6ddac1fed10c5ad96fcd211db48f78d26",
    "runner_script_version": '06e1-run-v1-biological-alignment-random-controls-v3-exact-risk-oriented-weight-schema-provenance-text-fix-two-stage-source-authorized-no-cli',
    "preauth_status": '06E1_V3_BIOLOGY_RUNNER_SYNTHETIC_SELFTEST_PASS',
    "authorization_status": '06E1A_V3_TARGET_V1_BIOLOGY_RUNNER_SOURCE_AUTHORIZED',
    "result_status": '06E2_V3_V1_BIOLOGICAL_ALIGNMENT_RANDOM_CONTROLS_FROZEN',
    "model_id": 'V1_SHARED_PRIVATE_AE',
    "profile_id": 'ARCH_COMPACT__REG_WEAK',
    "primary_seed": 1103,
    "n_cases": 78,
    "shared_dim": 5,
    "programs": ['M34', 'M40', 'M11', 'M24'],
    "strict_gene_counts": {'M34': 162, 'M40': 111, 'M11': 7, 'M24': 7},
    "representation_filename": 'target_v1_patient_shared_seed1103.npy',
    "representation_sha256": 'c05bf08017ad8544e1e4ce0c8f58f086060dd7a6a0a68ed7f8d5a6723768a95f',
    "technical_status": 'UNKNOWN_NO_STRICT_USABLE_TECHNICAL_COVARIATES',
    "random_panels_per_program": 1000,
    "random_seed": 42,
    "variability_bins": 10,
    "primary_score_variant": 'strict_signed_mean',
    "primary_alignment_metric": 'max_absolute_spearman_across_5_frozen_v1_shared_coordinates',
    "minimum_program_coverage_fraction":
        1.0,
    "multiplicity": 'BH',
    "multiplicity_alpha": 0.05,
    "real_strict_weight_columns": ['module_label', 'canine_gene', 'canine_gene_symbol', 'human_gene_symbol', 'ortholog_qc_status', 'raw_pca_loading', 'risk_oriented_loading', 'absolute_risk_loading', 'is_strict_mapping', 'is_broad_mapping', 'human_symbol_duplicate_count', 'normalized_abs_sum_weight'],
    "risk_orientation_column": 'risk_oriented_loading',
    "v1_failure_lock_status": '06E1R0_V1_FIRST_REAL_BIOLOGY_ATTEMPT_SCHEMA_FAILURE_FROZEN',
    "v1_failed_runner_sha256": '51eface403ec7b41a6436289590f0cd4159ad4b6e167800052de8eb2f11bd180',
    "v2_superseded_runner_sha256": 'b2cb9b9fa67412b373f02d8593df5b79e156b21bf5da742075c8605c858fc5a3',
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


def literal_assignment(path: Path, name: str) -> Any:
    tree = ast.parse(
        path.read_text(encoding="utf-8", errors="replace"),
        filename=str(path),
    )
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name
            for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"{name} assignment not found in {path}")


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


def static_runner_audit(runner: Path) -> dict[str, Any]:
    source = runner.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source, filename=str(runner))

    observed_version = literal_assignment(runner, "SCRIPT_VERSION")
    observed_preauth = literal_assignment(runner, "PREAUTH_STATUS")
    observed_auth = literal_assignment(runner, "AUTH_STATUS")
    observed_result = literal_assignment(runner, "RESULT_STATUS")
    observed_expected = literal_assignment(runner, "EXPECTED")
    observed_columns = literal_assignment(
        runner, "REAL_STRICT_WEIGHT_COLUMNS"
    )
    observed_risk_col = literal_assignment(
        runner, "RISK_ORIENTATION_COLUMN"
    )
    observed_v1_fail_status = literal_assignment(
        runner, "V1_FAILURE_LOCK_STATUS"
    )
    observed_v1_fail_sha = literal_assignment(
        runner, "V1_FAILED_RUNNER_SHA256"
    )
    observed_v2_sha = literal_assignment(
        runner, "V2_SYNTHETIC_ONLY_SUPERSEDED_RUNNER_SHA256"
    )

    if observed_version != EXPECTED["runner_script_version"]:
        raise RuntimeError("Runner SCRIPT_VERSION mismatch.")
    if observed_preauth != EXPECTED["preauth_status"]:
        raise RuntimeError("Runner PREAUTH_STATUS mismatch.")
    if observed_auth != EXPECTED["authorization_status"]:
        raise RuntimeError("Runner AUTH_STATUS mismatch.")
    if observed_result != EXPECTED["result_status"]:
        raise RuntimeError("Runner RESULT_STATUS mismatch.")

    for key in [
        "model_id",
        "profile_id",
        "primary_seed",
        "n_cases",
        "shared_dim",
        "programs",
        "strict_gene_counts",
        "representation_filename",
        "representation_sha256",
        "technical_status",
        "random_panels_per_program",
        "random_seed",
        "variability_bins",
        "primary_score_variant",
        "primary_alignment_metric",
        "minimum_program_coverage_fraction",
        "multiplicity",
        "multiplicity_alpha",
    ]:
        if observed_expected.get(key) != EXPECTED[key]:
            raise RuntimeError(
                f"Runner EXPECTED[{key!r}] mismatch: "
                f"{observed_expected.get(key)!r} != {EXPECTED[key]!r}"
            )

    if observed_columns != EXPECTED["real_strict_weight_columns"]:
        raise RuntimeError("Runner exact real strict-weight schema mismatch.")
    if observed_risk_col != EXPECTED["risk_orientation_column"]:
        raise RuntimeError("Runner risk-orientation column mismatch.")
    if observed_v1_fail_status != EXPECTED["v1_failure_lock_status"]:
        raise RuntimeError("Runner v1 failure-lock status mismatch.")
    if observed_v1_fail_sha != EXPECTED["v1_failed_runner_sha256"]:
        raise RuntimeError("Runner v1 failed-source SHA mismatch.")
    if observed_v2_sha != EXPECTED["v2_superseded_runner_sha256"]:
        raise RuntimeError("Runner v2 superseded-source SHA mismatch.")

    main_node = function_node(tree, "main")
    main_text = ast.unparse(main_node)
    for fragment in [
        "authorization = load_authorization",
        "if authorization is None:",
        "synthetic_selftest",
        "return",
        "run_real",
    ]:
        if fragment not in main_text:
            raise RuntimeError(
                f"Runner main authorization gate changed: missing {fragment}"
            )

    load_auth = function_node(tree, "load_authorization")
    load_auth_source = ast.get_source_segment(source, load_auth) or ""
    for fragment in [
        '"06e1a_v3"',
        '"target_v1_biology_runner_v3_source_authorization_lock.json"',
        "AUTH_STATUS",
        "PREAUTH_STATUS",
    ]:
        if fragment not in load_auth_source:
            raise RuntimeError(
                f"Runner authorization contract changed: missing {fragment}"
            )

    verify_chain = function_node(tree, "verify_06e0_chain")
    verify_source = ast.get_source_segment(source, verify_chain) or ""
    for fragment in [
        '"06e1r0"',
        "V1_FAILURE_LOCK_STATUS",
        "V1_FAILED_RUNNER_SHA256",
        "REAL_STRICT_WEIGHT_COLUMNS",
        "sign(risk_oriented_loading)",
    ]:
        if fragment not in verify_source:
            raise RuntimeError(
                f"Runner v1-failure provenance guard changed: missing {fragment}"
            )

    synthetic = function_node(tree, "synthetic_selftest")
    synthetic_source = ast.get_source_segment(source, synthetic) or ""
    if "touch_biological_files=False" not in synthetic_source:
        raise RuntimeError(
            "Synthetic branch no longer explicitly forbids biological-file access."
        )

    resolver = function_node(tree, "resolve_weight_schema")
    resolver_source = ast.get_source_segment(source, resolver) or ""
    for fragment in [
        "REAL_STRICT_WEIGHT_COLUMNS",
        '"module_label"',
        '"human_gene_symbol"',
        "RISK_ORIENTATION_COLUMN",
    ]:
        if fragment not in resolver_source:
            raise RuntimeError(
                f"Exact weight-schema resolver changed: missing {fragment}"
            )

    extractor = function_node(tree, "extract_program_definitions")
    extractor_source = ast.get_source_segment(source, extractor) or ""
    for fragment in [
        "risk_oriented_loading",
        "np.sign",
        '"is_strict_mapping"',
        'isin({"true", "1"})',
    ]:
        if fragment not in extractor_source:
            raise RuntimeError(
                f"Risk-oriented program extraction changed: missing {fragment}"
            )

    score_fn = function_node(tree, "compute_signed_mean_program_score")
    score_source = ast.get_source_segment(source, score_fn) or ""
    for fragment in [
        "np.nanmedian",
        "np.isnan",
        "standardize_ddof1",
        "np.mean",
    ]:
        if fragment not in score_source:
            raise RuntimeError(
                f"Primary strict signed-mean scoring changed: missing {fragment}"
            )

    real = function_node(tree, "run_real")
    real_source = ast.get_source_segment(source, real) or ""

    hash_pos = real_source.find("touch_biological_files=True")
    weights_pos = real_source.find("weights = pd.read_csv(")
    schema_pos = real_source.find("schema = resolve_weight_schema(weights)")
    latent_pos = real_source.find("latent = load_primary_representation(")
    expression_pos = real_source.find("expression = load_target_expression_78(")

    if min(hash_pos, weights_pos, schema_pos, latent_pos, expression_pos) < 0:
        raise RuntimeError("Could not resolve real scientific read ordering.")
    if not (
        hash_pos
        < weights_pos
        < schema_pos
        < latent_pos
        < expression_pos
    ):
        raise RuntimeError(
            "Real v3 scientific read order changed: "
            "hash -> weights -> exact schema -> V1 -> expression is required."
        )

    for fragment in [
        '"06e1_biological_alignment_v3"',
        '"06e2_v3"',
        "Do not rerun or delete prior scientific outputs.",
    ]:
        if fragment not in real_source:
            raise RuntimeError(
                f"Runner v3 output guard changed: missing {fragment}"
            )

    prohibited = [
        "TARGET_OS_clinical_standardized.csv",
        "os_time_days",
        "os_event",
        "overall_survival",
        "metastasis_event",
    ]
    present = [x for x in prohibited if x in source]
    if present:
        raise RuntimeError(
            "Runner contains prohibited clinical/outcome terms: "
            f"{present}"
        )

    coordinate_fn = function_node(tree, "coordinate_correlations")
    coordinate_source = ast.get_source_segment(source, coordinate_fn) or ""
    if "np.argmax(abs_values)" not in coordinate_source:
        raise RuntimeError(
            "Coordinate statistic no longer uses max absolute correlation."
        )
    if real_source.count("coordinate_correlations(") < 2:
        raise RuntimeError(
            "Observed and random controls no longer use the same "
            "coordinate-correlation statistic."
        )

    empirical_fn = function_node(tree, "empirical_plus_one_p")
    empirical_source = ast.get_source_segment(source, empirical_fn) or ""
    if "(1.0 + count) / (1.0 + null_values.size)" not in empirical_source:
        raise RuntimeError("+1 empirical probability implementation changed.")

    if "benjamini_hochberg(p_values)" not in real_source:
        raise RuntimeError("BH correction call missing from real branch.")

    sample_fn = function_node(tree, "sample_matched_panel")
    sample_source = ast.get_source_segment(source, sample_fn) or ""
    for fragment in [
        "for distance in range(0, n_bins):",
        "gene not in used",
        "used.add(selected_gene)",
        "fallback_distance",
    ]:
        if fragment not in sample_source:
            raise RuntimeError(
                f"Matched-random sampler changed: missing {fragment}"
            )

    bh_pos = real_source.find("q_values = benjamini_hochberg(p_values)")
    mkdir_pos = real_source.find(
        "result_dir.mkdir(parents=True, exist_ok=False)"
    )
    if bh_pos < 0 or mkdir_pos < 0 or mkdir_pos <= bh_pos:
        raise RuntimeError(
            "Scientific result directory is created before all statistics complete."
        )

    return {
        "script_version": observed_version,
        "authorization_gate": "PASS",
        "synthetic_biological_access": "FORBIDDEN",
        "v1_failure_provenance_guard": "PASS",
        "exact_real_strict_weight_schema": "PASS",
        "risk_orientation_sign_rule": "PASS",
        "median_imputation_scoring_rule": "PASS",
        "real_biological_hash_verification_before_parse": "PASS",
        "schema_validation_before_v1_numeric_read": "PASS",
        "clinical_outcome_firewall": "PASS",
        "observed_and_null_use_same_coordinate_statistic": "PASS",
        "matched_random_without_replacement_and_nearest_bin_fallback": "PASS",
        "plus_one_empirical_probability": "PASS",
        "bh_across_four_programs": "PASS",
        "scientific_output_creation_after_computation": "PASS",
    }


def main() -> None:
    print("=" * 104)
    print(
        "Paper 5 - 06e1a_v3 freeze exact V1 biology-runner "
        "source authorization"
    )
    print("=" * 104)
    print("Script version:", SCRIPT_VERSION)
    print("Paper-4 biological artifacts opened/hashed in 06e1a_v3: NO")
    print("TARGET biological values read in 06e1a_v3: NO")
    print("V1 numeric representation read in 06e1a_v3: NO")
    print("TARGET clinical/outcomes read in 06e1a_v3: NO")
    print("Biological alignment computed in 06e1a_v3: NO")
    print("Matched random controls computed in 06e1a_v3: NO")

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    paper5 = project_root / "paper5"

    runner = (
        paper5 / "scripts"
        / "06e1_run_v1_biological_alignment_random_controls_v3.py"
    )
    preflight_path = (
        paper5 / "method_contract" / "06e1_preflight_v3"
        / "v1_biological_alignment_runner_synthetic_selftest_v3.json"
    )
    contract_path = (
        paper5 / "method_contract" / "06e0"
        / "v1_biological_alignment_random_control_contract.json"
    )
    contract_lock_path = (
        paper5 / "method_contract" / "06e0"
        / "06e0_v1_biological_alignment_random_control_contract_lock.json"
    )
    v1_failure_lock_path = (
        paper5 / "method_contract" / "06e1r0"
        / "v1_first_real_biology_attempt_schema_failure_boundary_lock.json"
    )

    for path in [
        runner,
        preflight_path,
        contract_path,
        contract_lock_path,
        v1_failure_lock_path,
    ]:
        require_file(path)

    observed_runner_sha = sha256_file(runner)
    if observed_runner_sha != EXPECTED["runner_sha256"]:
        raise RuntimeError(
            "Exact 06e1 v3 runner SHA mismatch. "
            f"Expected={EXPECTED['runner_sha256']} "
            f"Observed={observed_runner_sha}"
        )

    static_audit = static_runner_audit(runner)

    preflight = load_json(preflight_path)
    if preflight.get("status") != EXPECTED["preauth_status"]:
        raise RuntimeError("06e1 v3 synthetic preflight status mismatch.")
    if preflight.get("script_version") != EXPECTED["runner_script_version"]:
        raise RuntimeError("06e1 v3 preflight SCRIPT_VERSION mismatch.")
    if str(preflight.get("runner_sha256", "")).lower() != (
        EXPECTED["runner_sha256"]
    ):
        raise RuntimeError(
            "06e1 v3 synthetic preflight belongs to a different runner SHA."
        )
    if preflight.get("execution_mode") != "SYNTHETIC_ONLY_PREAUTHORIZATION":
        raise RuntimeError("06e1 v3 preflight execution-mode mismatch.")

    scientific_state = preflight.get("scientific_state")
    if not isinstance(scientific_state, dict):
        raise RuntimeError("06e1 v3 preflight scientific_state missing.")
    for key in [
        "real_biological_input_hashed",
        "real_biological_input_parsed",
        "v1_numeric_representation_array_read",
        "target_clinical_or_outcomes_read",
        "target_model_fit_or_refit",
        "biological_alignment_computed_on_real_data",
        "random_controls_computed_on_real_data",
    ]:
        if scientific_state.get(key) is not False:
            raise RuntimeError(
                f"06e1 v3 preflight scientific firewall failed: {key}"
            )

    boundary = preflight.get("real_execution_boundary", {})
    if boundary.get("authorization_required") is not True:
        raise RuntimeError("06e1 v3 preflight no longer requires authorization.")
    if boundary.get("authorization_status") != EXPECTED["authorization_status"]:
        raise RuntimeError("06e1 v3 preflight authorization status mismatch.")
    if boundary.get("authorization_namespace") != "method_contract/06e1a_v3":
        raise RuntimeError("06e1 v3 preflight authorization namespace mismatch.")
    if boundary.get(
        "real_runner_must_verify_all_frozen_biological_hashes_before_parsing"
    ) is not True:
        raise RuntimeError(
            "06e1 v3 preflight hash-before-parse boundary is missing."
        )

    math_audit = preflight.get("synthetic_math_and_schema_audit", {})
    if math_audit.get("status") != "PASS":
        raise RuntimeError("06e1 v3 synthetic math/schema audit is not PASS.")
    if int(math_audit.get("n_coordinates")) != EXPECTED["shared_dim"]:
        raise RuntimeError("06e1 v3 synthetic coordinate count mismatch.")
    if int(math_audit.get("n_programs")) != len(EXPECTED["programs"]):
        raise RuntimeError("06e1 v3 synthetic program count mismatch.")
    if math_audit.get(
        "smoke_random_panel_count_is_scientific_setting"
    ) is not False:
        raise RuntimeError(
            "Synthetic smoke panel count is incorrectly marked scientific."
        )

    required_math_passes = [
        "explicit_nearest_bin_fallback_test",
        "target_case_normalization",
        "bh_test",
        "sign_invariant_best_coordinate_test",
        "risk_oriented_loading_sign_test",
        "median_imputation_scoring_test",
    ]
    for key in required_math_passes:
        if math_audit.get(key) != "PASS":
            raise RuntimeError(
                f"06e1 v3 synthetic math/schema subtest failed: {key}"
            )

    if math_audit.get("exact_real_weight_schema_resolution") != {
        "module": "module_label",
        "gene": "human_gene_symbol",
        "risk_orientation": "risk_oriented_loading",
    }:
        raise RuntimeError(
            "06e1 v3 exact-real-schema synthetic resolver mismatch."
        )

    identity = preflight.get("06e0_identity", {})
    if str(identity.get("contract_sha256", "")).lower() != (
        sha256_file(contract_path)
    ):
        raise RuntimeError(
            "06e1 v3 preflight 06e0 contract identity no longer matches."
        )
    if str(identity.get("lock_sha256", "")).lower() != (
        sha256_file(contract_lock_path)
    ):
        raise RuntimeError(
            "06e1 v3 preflight 06e0 lock identity no longer matches."
        )

    v1_failure_lock = load_json(v1_failure_lock_path)
    if v1_failure_lock.get("status") != EXPECTED["v1_failure_lock_status"]:
        raise RuntimeError("06e1r0 failure-boundary status mismatch.")
    if str(
        v1_failure_lock.get("failed_runner", {}).get("sha256", "")
    ).lower() != EXPECTED["v1_failed_runner_sha256"]:
        raise RuntimeError("06e1r0 failed v1 runner SHA mismatch.")

    failure_identity = preflight.get("v1_failure_boundary_identity", {})
    if str(failure_identity.get("sha256", "")).lower() != (
        sha256_file(v1_failure_lock_path)
    ):
        raise RuntimeError(
            "06e1 v3 preflight failure-boundary lock identity mismatch."
        )
    if failure_identity.get("observed_weight_columns") != (
        EXPECTED["real_strict_weight_columns"]
    ):
        raise RuntimeError(
            "06e1 v3 preflight observed real strict-weight schema mismatch."
        )
    if failure_identity.get("repair_semantics") != (
        "primary strict_signed_mean must use sign(risk_oriented_loading)"
    ):
        raise RuntimeError(
            "06e1 v3 preflight repair-semantics mismatch."
        )

    superseded = preflight.get("superseded_v2_source", {})
    if superseded.get("runner_sha256") != EXPECTED["v2_superseded_runner_sha256"]:
        raise RuntimeError("06e1 v3 superseded-v2 SHA mismatch.")
    if superseded.get("scientific_execution_authorized") is not False:
        raise RuntimeError("Superseded v2 was unexpectedly marked authorized.")
    if superseded.get("real_biology_run_performed") is not False:
        raise RuntimeError("Superseded v2 was unexpectedly marked as real-run.")

    contract = load_json(contract_path)
    contract_lock = load_json(contract_lock_path)
    if contract.get("status") != (
        "FROZEN_BEFORE_V1_BIOLOGICAL_ALIGNMENT_AND_RANDOM_CONTROLS"
    ):
        raise RuntimeError("06e0 contract status mismatch.")
    if contract_lock.get("status") != (
        "06E0_V1_BIOLOGICAL_ALIGNMENT_RANDOM_CONTROL_CONTRACT_FROZEN"
    ):
        raise RuntimeError("06e0 lock status mismatch.")
    if str(
        contract_lock.get("contract", {}).get("sha256", "")
    ).lower() != sha256_file(contract_path):
        raise RuntimeError("06e0 lock does not seal current contract.")

    rep = contract.get("frozen_representation", {})
    if rep.get("model_id") != EXPECTED["model_id"]:
        raise RuntimeError("06e0 model mismatch.")
    if rep.get("profile_id") != EXPECTED["profile_id"]:
        raise RuntimeError("06e0 profile mismatch.")
    if int(rep.get("primary_seed")) != EXPECTED["primary_seed"]:
        raise RuntimeError("06e0 primary seed mismatch.")
    if int(rep.get("n_cases")) != EXPECTED["n_cases"]:
        raise RuntimeError("06e0 n_cases mismatch.")
    if int(rep.get("shared_dim")) != EXPECTED["shared_dim"]:
        raise RuntimeError("06e0 shared_dim mismatch.")
    if str(rep.get("representation_sha256", "")).lower() != (
        EXPECTED["representation_sha256"]
    ):
        raise RuntimeError("06e0 representation SHA mismatch.")
    if rep.get("coordinate_stability_qualification") != "NOT_ESTABLISHED_FOR_V1":
        raise RuntimeError("06e0 V1 stability boundary changed.")
    if rep.get("sable_mo_q10_or_retention_transfer") != "FORBIDDEN":
        raise RuntimeError("06e0 SABLE-transfer boundary changed.")

    programs = contract.get("primary_biological_programs", {})
    if programs.get("program_ids") != EXPECTED["programs"]:
        raise RuntimeError("06e0 program set/order mismatch.")
    if programs.get("strict_gene_counts") != EXPECTED["strict_gene_counts"]:
        raise RuntimeError("06e0 strict gene counts mismatch.")

    random_contract = contract.get("matched_random_program_controls", {})
    if int(random_contract.get("panels_per_program")) != (
        EXPECTED["random_panels_per_program"]
    ):
        raise RuntimeError("06e0 random-panel count mismatch.")
    if int(random_contract.get("random_seed")) != EXPECTED["random_seed"]:
        raise RuntimeError("06e0 random seed mismatch.")
    if int(random_contract.get("variability_bins")) != EXPECTED["variability_bins"]:
        raise RuntimeError("06e0 variability-bin count mismatch.")

    # Refuse post-hoc v3 authorization.
    scientific_result_dir = (
        paper5 / "results" / "target_representation"
        / "06e1_biological_alignment_v3"
    )
    result_lock_dir = paper5 / "method_contract" / "06e2_v3"

    if scientific_result_dir.exists() and any(scientific_result_dir.iterdir()):
        raise RuntimeError(
            "06e1 v3 scientific results already exist; refusing post-hoc authorization."
        )
    if result_lock_dir.exists() and any(result_lock_dir.iterdir()):
        raise RuntimeError(
            "06e2_v3 result-lock artifacts already exist; refusing post-hoc authorization."
        )

    out_dir = paper5 / "method_contract" / "06e1a_v3"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(
            f"06e1a_v3 directory already exists and is non-empty: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    lock_payload = {
        "status": EXPECTED["authorization_status"],
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),

        "scientific_state": {
            "paper4_biological_artifacts_opened_or_hashed": False,
            "target_biological_values_read": False,
            "v1_numeric_representation_read": False,
            "target_molecular_matrices_read": False,
            "target_clinical_or_outcomes_read": False,
            "biological_alignment_computed": False,
            "matched_random_controls_computed": False,
            "model_fit_or_refit": False,
        },

        "runner_source": {
            "path":
                "scripts/06e1_run_v1_biological_alignment_random_controls_v3.py",
            "sha256": EXPECTED["runner_sha256"],
            "script_version": EXPECTED["runner_script_version"],
            "size_bytes": runner.stat().st_size,
        },

        "synthetic_selftest": {
            "path":
                "method_contract/06e1_preflight_v3/"
                "v1_biological_alignment_runner_synthetic_selftest_v3.json",
            "sha256": sha256_file(preflight_path),
            "status": preflight.get("status"),
            "script_version": preflight.get("script_version"),
            "runner_sha256": preflight.get("runner_sha256"),
            "execution_mode": preflight.get("execution_mode"),
            "n_coordinates": math_audit.get("n_coordinates"),
            "n_programs": math_audit.get("n_programs"),
            "exact_real_weight_schema_resolution":
                math_audit.get("exact_real_weight_schema_resolution"),
            "risk_oriented_loading_sign_test":
                math_audit.get("risk_oriented_loading_sign_test"),
            "median_imputation_scoring_test":
                math_audit.get("median_imputation_scoring_test"),
            "explicit_nearest_bin_fallback_test":
                math_audit.get("explicit_nearest_bin_fallback_test"),
            "target_case_normalization":
                math_audit.get("target_case_normalization"),
            "bh_test": math_audit.get("bh_test"),
            "sign_invariant_best_coordinate_test":
                math_audit.get("sign_invariant_best_coordinate_test"),
        },

        "06e0_identity": {
            "contract_sha256": sha256_file(contract_path),
            "lock_sha256": sha256_file(contract_lock_path),
        },

        "v1_failure_boundary_identity": {
            "path":
                "method_contract/06e1r0/"
                "v1_first_real_biology_attempt_schema_failure_boundary_lock.json",
            "sha256": sha256_file(v1_failure_lock_path),
            "status": v1_failure_lock.get("status"),
            "failed_runner_sha256":
                EXPECTED["v1_failed_runner_sha256"],
        },

        "superseded_v2_source": {
            "runner_sha256":
                EXPECTED["v2_superseded_runner_sha256"],
            "scientific_execution_authorized": False,
            "real_biology_run_performed": False,
        },

        "static_runner_audit": static_audit,

        "authorized_scientific_execution": {
            "model_id": EXPECTED["model_id"],
            "profile_id": EXPECTED["profile_id"],
            "primary_seed": EXPECTED["primary_seed"],
            "n_cases": EXPECTED["n_cases"],
            "shared_dim": EXPECTED["shared_dim"],
            "programs": EXPECTED["programs"],
            "strict_gene_counts": EXPECTED["strict_gene_counts"],
            "representation_sha256":
                EXPECTED["representation_sha256"],
            "technical_status":
                EXPECTED["technical_status"],
            "program_score":
                EXPECTED["primary_score_variant"],
            "exact_strict_weight_columns":
                EXPECTED["real_strict_weight_columns"],
            "primary_sign_source":
                "sign(risk_oriented_loading)",
            "primary_alignment_metric":
                EXPECTED["primary_alignment_metric"],
            "random_panels_per_program":
                EXPECTED["random_panels_per_program"],
            "random_seed": EXPECTED["random_seed"],
            "variability_bins": EXPECTED["variability_bins"],
            "minimum_program_coverage_fraction":
                EXPECTED["minimum_program_coverage_fraction"],
            "multiplicity": EXPECTED["multiplicity"],
            "multiplicity_alpha": EXPECTED["multiplicity_alpha"],
            "clinical_or_outcome_access": "FORBIDDEN",
            "representation_refit_or_reselection": "FORBIDDEN",
            "SABLE_q10_or_stability_transfer_to_V1": "FORBIDDEN",
            "technical_UNKNOWN_reinterpretation_as_clean": "FORBIDDEN",
        },

        "authorization_rule": (
            "Only this exact 06e1 v3 runner SHA is authorized. Any source "
            "change requires a new versioned synthetic preflight and a new "
            "source authorization before real biological computation."
        ),
    }

    lock_path = (
        out_dir
        / "target_v1_biology_runner_v3_source_authorization_lock.json"
    )
    lock_path.write_text(
        json.dumps(lock_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Exact 06e1 v3 runner SHA-256: PASS")
    print("Exact v3 SCRIPT_VERSION/constants: PASS")
    print("Exact v3 synthetic preflight identity: PASS")
    print("Synthetic scientific-access firewalls: PASS")
    print("Exact real 12-column strict-weight schema self-test: PASS")
    print("sign(risk_oriented_loading) self-test: PASS")
    print("Median-imputation score-semantics self-test: PASS")
    print("Exact 06e0 contract/lock identity: PASS")
    print("Exact 06e1r0 failed-v1 provenance identity: PASS")
    print("Superseded v2 real scientific execution: NO")
    print("Runner static real-path authorization audit: PASS")
    print("Biological hash-before-parse boundary: PASS")
    print("Exact weight schema before V1 numeric array: PASS")
    print("Observed/null identical five-coordinate statistic: PASS")
    print("Matched-random same-bin/nearest-bin + without-replacement path: PASS")
    print("+1 empirical probability + BH family: PASS")
    print("Existing v3 scientific results before authorization: NO")
    print("Existing 06e2_v3 result lock before authorization: NO")
    print("Paper-4 biological artifacts opened/hashed in 06e1a_v3: NO")
    print("TARGET clinical/outcomes read in 06e1a_v3: NO")
    print("Biological alignment computed in 06e1a_v3: NO")
    print("=" * 104)
    print(
        "Paper 5 06e1a_v3 exact biology-runner source authorization: FROZEN"
    )
    print("=" * 104)
    print("Authorized runner SHA:", EXPECTED["runner_sha256"])
    print(
        "Authorization lock: method_contract/06e1a_v3/"
        "target_v1_biology_runner_v3_source_authorization_lock.json"
    )
    print("Next: rerun the exact same 06e1 v3 runner unchanged.")
    print("=" * 104)


if __name__ == "__main__":
    main()
