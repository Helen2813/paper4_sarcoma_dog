#!/usr/bin/env python3
"""
Paper 5 - 06e1a_v1 freeze exact 06e1 biology-runner source authorization.

Authorization-only stage.

This script authorizes only the exact SHA-256-locked 06e1 v1 runner after its
synthetic preauthorization PASS. It reads only Paper-5 method-contract/provenance
metadata and the runner source itself.

It DOES NOT:
- open/hash/parse any Paper-4 biological input;
- read the numeric V1 representation array;
- read TARGET molecular matrices;
- read clinical variables or outcomes;
- compute a real biological alignment;
- generate matched random controls;
- fit/refit any model.

It refuses post-hoc source authorization if any 06e1 scientific result or 06e2
result-lock artifact already exists.

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
    "06e1a-v1-freeze-v1-biology-runner-source-authorization-v1-no-cli"
)

EXPECTED = {
    "runner_sha256": "51eface403ec7b41a6436289590f0cd4159ad4b6e167800052de8eb2f11bd180",
    "runner_script_version": '06e1-run-v1-biological-alignment-random-controls-v1-two-stage-source-authorized-no-cli',
    "preauth_status": '06E1_V1_BIOLOGY_RUNNER_SYNTHETIC_SELFTEST_PASS',
    "authorization_status": '06E1A_V1_TARGET_V1_BIOLOGY_RUNNER_SOURCE_AUTHORIZED',
    "result_status": '06E2_V1_BIOLOGICAL_ALIGNMENT_RANDOM_CONTROLS_FROZEN',
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
    "random_panels_per_program":
        1000,
    "random_seed": 42,
    "variability_bins": 10,
    "primary_score_variant": 'strict_signed_mean',
    "primary_alignment_metric":
        'max_absolute_spearman_across_5_frozen_v1_shared_coordinates',
    "minimum_program_coverage_fraction":
        1.0,
    "multiplicity": 'BH',
    "multiplicity_alpha": 0.05,
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
    if observed_version != EXPECTED["runner_script_version"]:
        raise RuntimeError("Runner SCRIPT_VERSION mismatch.")

    observed_preauth = literal_assignment(runner, "PREAUTH_STATUS")
    observed_auth = literal_assignment(runner, "AUTH_STATUS")
    observed_result = literal_assignment(runner, "RESULT_STATUS")
    observed_expected = literal_assignment(runner, "EXPECTED")

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
        '"06e1a_v1"',
        '"target_v1_biology_runner_v1_source_authorization_lock.json"',
        "AUTH_STATUS",
        "PREAUTH_STATUS",
    ]:
        if fragment not in load_auth_source:
            raise RuntimeError(
                f"Runner authorization contract changed: missing {fragment}"
            )

    synthetic = function_node(tree, "synthetic_selftest")
    synthetic_source = ast.get_source_segment(source, synthetic) or ""
    if "touch_biological_files=False" not in synthetic_source:
        raise RuntimeError(
            "Synthetic branch no longer explicitly forbids biological-file access."
        )

    real = function_node(tree, "run_real")
    real_source = ast.get_source_segment(source, real) or ""
    hash_gate_pos = real_source.find("touch_biological_files=True")
    if hash_gate_pos < 0:
        raise RuntimeError(
            "Real branch no longer verifies frozen biological hashes."
        )

    real_parse_markers = [
        "load_case_order(",
        "load_primary_representation(",
        "pd.read_csv(",
        "load_target_expression_78(",
    ]
    real_parse_positions = [
        real_source.find(marker)
        for marker in real_parse_markers
        if real_source.find(marker) >= 0
    ]
    if not real_parse_positions:
        raise RuntimeError(
            "Could not identify real scientific read boundary."
        )
    if hash_gate_pos >= min(real_parse_positions):
        raise RuntimeError(
            "Real scientific parsing occurs before biological SHA verification."
        )

    # Exact post-hoc authorization/result guards.
    for fragment in [
        '"06e1_biological_alignment_v1"',
        '"06e2"',
        "Do not rerun or delete prior scientific outputs.",
    ]:
        if fragment not in real_source:
            raise RuntimeError(
                f"Runner real-output guard changed: missing {fragment}"
            )

    # Outcome/clinical firewall.
    prohibited_path_or_endpoint_strings = [
        "TARGET_OS_clinical_standardized.csv",
        "os_time_days",
        "os_event",
        "overall_survival",
        "metastasis_event",
    ]
    present = [
        x for x in prohibited_path_or_endpoint_strings
        if x in source
    ]
    if present:
        raise RuntimeError(
            "Runner contains prohibited clinical/outcome path or endpoint strings: "
            f"{present}"
        )

    # Observed statistic and null must use the same five-coordinate maximization.
    coordinate_fn = function_node(tree, "coordinate_correlations")
    coordinate_source = ast.get_source_segment(source, coordinate_fn) or ""
    if "np.argmax(abs_values)" not in coordinate_source:
        raise RuntimeError(
            "Coordinate statistic no longer selects maximum absolute correlation."
        )

    observed_calls = real_source.count("coordinate_correlations(")
    if observed_calls < 2:
        raise RuntimeError(
            "Real branch no longer applies coordinate_correlations to both "
            "observed and random-program scores."
        )

    # Exact +1 probability and BH family.
    empirical_fn = function_node(tree, "empirical_plus_one_p")
    empirical_source = ast.get_source_segment(source, empirical_fn) or ""
    if "(1.0 + count) / (1.0 + null_values.size)" not in empirical_source:
        raise RuntimeError("+1 empirical probability implementation changed.")

    if "benjamini_hochberg(p_values)" not in real_source:
        raise RuntimeError("BH correction call is missing from the real branch.")

    # Random sampling semantics.
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

    # Scientific output creation must occur only after all program/random
    # computations have completed.
    result_mkdir_pos = real_source.find(
        "result_dir.mkdir(parents=True, exist_ok=False)"
    )
    bh_pos = real_source.find("q_values = benjamini_hochberg(p_values)")
    if result_mkdir_pos < 0 or bh_pos < 0 or result_mkdir_pos <= bh_pos:
        raise RuntimeError(
            "Scientific output directory is created before real computation completes."
        )

    return {
        "script_version": observed_version,
        "authorization_gate": "PASS",
        "synthetic_biological_access": "FORBIDDEN",
        "real_biological_hash_verification_before_parse": "PASS",
        "posthoc_result_guard": "PASS",
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
        "Paper 5 - 06e1a_v1 freeze exact V1 biology-runner "
        "source authorization"
    )
    print("=" * 104)
    print("Script version:", SCRIPT_VERSION)
    print("Paper-4 biological artifacts opened/hashed in 06e1a_v1: NO")
    print("TARGET biological values read in 06e1a_v1: NO")
    print("V1 numeric representation read in 06e1a_v1: NO")
    print("TARGET clinical/outcomes read in 06e1a_v1: NO")
    print("Biological alignment computed in 06e1a_v1: NO")
    print("Matched random controls computed in 06e1a_v1: NO")

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    paper5 = project_root / "paper5"

    runner = (
        paper5 / "scripts"
        / "06e1_run_v1_biological_alignment_random_controls_v1.py"
    )
    preflight_path = (
        paper5 / "method_contract" / "06e1_preflight"
        / "v1_biological_alignment_runner_synthetic_selftest_v1.json"
    )
    contract_path = (
        paper5 / "method_contract" / "06e0"
        / "v1_biological_alignment_random_control_contract.json"
    )
    contract_lock_path = (
        paper5 / "method_contract" / "06e0"
        / "06e0_v1_biological_alignment_random_control_contract_lock.json"
    )

    for path in [
        runner,
        preflight_path,
        contract_path,
        contract_lock_path,
    ]:
        require_file(path)

    observed_runner_sha = sha256_file(runner)
    if observed_runner_sha != EXPECTED["runner_sha256"]:
        raise RuntimeError(
            "Exact 06e1 runner SHA mismatch. "
            f"Expected={EXPECTED['runner_sha256']} "
            f"Observed={observed_runner_sha}"
        )

    static_audit = static_runner_audit(runner)

    preflight = load_json(preflight_path)
    if preflight.get("status") != EXPECTED["preauth_status"]:
        raise RuntimeError("06e1 synthetic preflight status mismatch.")
    if preflight.get("script_version") != EXPECTED["runner_script_version"]:
        raise RuntimeError("06e1 synthetic preflight SCRIPT_VERSION mismatch.")
    if str(preflight.get("runner_sha256", "")).lower() != (
        EXPECTED["runner_sha256"]
    ):
        raise RuntimeError(
            "06e1 synthetic preflight belongs to a different runner SHA."
        )
    if preflight.get("execution_mode") != "SYNTHETIC_ONLY_PREAUTHORIZATION":
        raise RuntimeError("06e1 preflight execution-mode mismatch.")

    scientific_state = preflight.get("scientific_state")
    if not isinstance(scientific_state, dict):
        raise RuntimeError("06e1 preflight scientific_state missing.")

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
                f"06e1 preflight scientific firewall failed: {key}"
            )

    boundary = preflight.get("real_execution_boundary", {})
    if boundary.get("authorization_required") is not True:
        raise RuntimeError("06e1 preflight no longer requires authorization.")
    if boundary.get("authorization_status") != EXPECTED["authorization_status"]:
        raise RuntimeError("06e1 preflight authorization status mismatch.")
    if boundary.get("authorization_namespace") != "method_contract/06e1a_v1":
        raise RuntimeError("06e1 preflight authorization namespace mismatch.")
    if boundary.get(
        "real_runner_must_verify_all_frozen_biological_hashes_before_parsing"
    ) is not True:
        raise RuntimeError(
            "06e1 preflight hash-before-parse boundary is missing."
        )

    math_audit = preflight.get("synthetic_math_and_schema_audit", {})
    if math_audit.get("status") != "PASS":
        raise RuntimeError("06e1 synthetic math/schema audit is not PASS.")
    if int(math_audit.get("n_coordinates")) != EXPECTED["shared_dim"]:
        raise RuntimeError("06e1 synthetic coordinate count mismatch.")
    if int(math_audit.get("n_programs")) != len(EXPECTED["programs"]):
        raise RuntimeError("06e1 synthetic program count mismatch.")
    if math_audit.get(
        "smoke_random_panel_count_is_scientific_setting"
    ) is not False:
        raise RuntimeError(
            "Synthetic smoke panel count is incorrectly marked scientific."
        )
    for key in [
        "explicit_nearest_bin_fallback_test",
        "target_case_normalization",
        "bh_test",
        "sign_invariant_best_coordinate_test",
    ]:
        if math_audit.get(key) != "PASS":
            raise RuntimeError(
                f"06e1 synthetic math/schema subtest failed: {key}"
            )

    # The preflight must certify the exact current 06e0 metadata identities.
    identity = preflight.get("06e0_identity", {})
    if str(identity.get("contract_sha256", "")).lower() != (
        sha256_file(contract_path)
    ):
        raise RuntimeError(
            "06e1 preflight 06e0 contract identity no longer matches."
        )
    if str(identity.get("lock_sha256", "")).lower() != (
        sha256_file(contract_lock_path)
    ):
        raise RuntimeError(
            "06e1 preflight 06e0 lock identity no longer matches."
        )

    contract = load_json(contract_path)
    contract_lock = load_json(contract_lock_path)
    if contract.get("status") != (
        "FROZEN_BEFORE_V1_BIOLOGICAL_ALIGNMENT_AND_RANDOM_CONTROLS"
    ):
        raise RuntimeError("06e0 contract status mismatch.")
    if contract_lock.get("status") != (
        "06E0_V1_BIOLOGICAL_ALIGNMENT_RANDOM_CONTROL_CONTRACT_FROZEN"
    ):
        raise RuntimeError("06e0 contract lock status mismatch.")
    if str(
        contract_lock.get("contract", {}).get("sha256", "")
    ).lower() != sha256_file(contract_path):
        raise RuntimeError("06e0 lock does not seal the current contract.")

    # Verify only metadata identities inside the 06e0 contract.
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

    # Refuse post-hoc authorization.
    scientific_result_dir = (
        paper5 / "results" / "target_representation"
        / "06e1_biological_alignment_v1"
    )
    result_lock_dir = paper5 / "method_contract" / "06e2"

    if scientific_result_dir.exists() and any(scientific_result_dir.iterdir()):
        raise RuntimeError(
            "06e1 scientific results already exist; refusing post-hoc source authorization."
        )
    if result_lock_dir.exists() and any(result_lock_dir.iterdir()):
        raise RuntimeError(
            "06e2 scientific result lock already exists; refusing post-hoc source authorization."
        )

    out_dir = paper5 / "method_contract" / "06e1a_v1"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(
            f"06e1a_v1 authorization directory already exists and is non-empty: {out_dir}"
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
                "scripts/06e1_run_v1_biological_alignment_random_controls_v1.py",
            "sha256": EXPECTED["runner_sha256"],
            "script_version": EXPECTED["runner_script_version"],
            "size_bytes": runner.stat().st_size,
        },

        "synthetic_selftest": {
            "path":
                "method_contract/06e1_preflight/"
                "v1_biological_alignment_runner_synthetic_selftest_v1.json",
            "sha256": sha256_file(preflight_path),
            "status": preflight.get("status"),
            "script_version": preflight.get("script_version"),
            "runner_sha256": preflight.get("runner_sha256"),
            "execution_mode": preflight.get("execution_mode"),
            "n_coordinates": math_audit.get("n_coordinates"),
            "n_programs": math_audit.get("n_programs"),
            "explicit_nearest_bin_fallback_test":
                math_audit.get("explicit_nearest_bin_fallback_test"),
            "target_case_normalization":
                math_audit.get("target_case_normalization"),
            "bh_test": math_audit.get("bh_test"),
            "sign_invariant_best_coordinate_test":
                math_audit.get("sign_invariant_best_coordinate_test"),
        },

        "06e0_identity": {
            "contract_path":
                "method_contract/06e0/"
                "v1_biological_alignment_random_control_contract.json",
            "contract_sha256": sha256_file(contract_path),
            "lock_path":
                "method_contract/06e0/"
                "06e0_v1_biological_alignment_random_control_contract_lock.json",
            "lock_sha256": sha256_file(contract_lock_path),
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
            "primary_alignment_metric":
                EXPECTED["primary_alignment_metric"],
            "random_panels_per_program":
                EXPECTED["random_panels_per_program"],
            "random_seed":
                EXPECTED["random_seed"],
            "variability_bins":
                EXPECTED["variability_bins"],
            "minimum_program_coverage_fraction":
                EXPECTED["minimum_program_coverage_fraction"],
            "multiplicity":
                EXPECTED["multiplicity"],
            "multiplicity_alpha":
                EXPECTED["multiplicity_alpha"],
            "clinical_or_outcome_access": "FORBIDDEN",
            "representation_refit_or_reselection": "FORBIDDEN",
            "SABLE_q10_or_stability_transfer_to_V1": "FORBIDDEN",
            "technical_UNKNOWN_reinterpretation_as_clean": "FORBIDDEN",
        },

        "authorization_rule": (
            "Only this exact 06e1 v1 runner SHA is authorized. Any source "
            "change requires a new versioned synthetic preflight and separate "
            "source authorization before real biological computation."
        ),
    }

    lock_path = (
        out_dir
        / "target_v1_biology_runner_v1_source_authorization_lock.json"
    )
    lock_path.write_text(
        json.dumps(lock_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Exact 06e1 runner SHA-256: PASS")
    print("Exact 06e1 runner SCRIPT_VERSION/constants: PASS")
    print("Exact 06e1 synthetic preflight identity: PASS")
    print("Synthetic scientific-access firewalls: PASS")
    print("Synthetic mathematical/schema subtests: PASS")
    print("Exact 06e0 contract/lock identity: PASS")
    print("Runner authorization gate/static real-path audit: PASS")
    print("Biological hash-before-parse boundary: PASS")
    print("Observed/null identical five-coordinate statistic: PASS")
    print("Matched-random same-bin/nearest-bin + without-replacement path: PASS")
    print("+1 empirical probability + BH family: PASS")
    print("Existing 06e1 scientific results before authorization: NO")
    print("Existing 06e2 result lock before authorization: NO")
    print("Paper-4 biological artifacts opened/hashed in 06e1a_v1: NO")
    print("TARGET clinical/outcomes read in 06e1a_v1: NO")
    print("Biological alignment computed in 06e1a_v1: NO")
    print("=" * 104)
    print(
        "Paper 5 06e1a_v1 exact biology-runner source authorization: FROZEN"
    )
    print("=" * 104)
    print("Authorized runner SHA:", EXPECTED["runner_sha256"])
    print(
        "Authorization lock: "
        "method_contract/06e1a_v1/"
        "target_v1_biology_runner_v1_source_authorization_lock.json"
    )
    print("Next: rerun the exact same 06e1 v1 runner unchanged.")
    print("=" * 104)


if __name__ == "__main__":
    main()
