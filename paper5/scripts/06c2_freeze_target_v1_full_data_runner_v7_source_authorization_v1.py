#!/usr/bin/env python3
"""
Paper 5 - freeze exact TARGET V1 all-78 runner v7 source authorization.

Authorization-only stage.

This script:
- verifies the exact v7 runner source SHA and SCRIPT_VERSION;
- verifies the exact frozen 06c contract/lock and authorized 06b v4 identity;
- verifies the exact v7 synthetic-only PASS report;
- replays static source checks for the real-input call contract and authorization gate;
- verifies that the earlier authorized v5 failure happened before the v4 loader body;
- refuses post-hoc authorization if scientific 06c full-data outputs already exist;
- writes only the 06c2_v7 source-authorization lock.

It does NOT load TARGET molecular matrices, read TARGET clinical/outcome/biology
values, fit any TARGET model, or read scientific TARGET full-data results.
"""

from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_VERSION = (
    "06c2-freeze-target-v1-full-data-runner-v7-source-authorization-v1-no-cli"
)

EXPECTED = {
    "runner_sha256": "006f718de8b0a1f768f8b4b7c35e4951719970f4d91ef43fa75541c74c8dc3bb",
    "runner_script_version": '06c-run-target-v1-full-data-refit-gpu-v7-exact-authorized-v4-profile-variant-path-two-stage-no-cli',
    "06c_contract_sha256": "8ab8516e30964f220d0f1204ca892e29ca337691e06cbcbf29315e1cb6aec2e5",
    "06c_lock_sha256": "1c3b4fb36706dc6afd0208fda23b1dc6d51c2953b4ccb9541099466aea13a374",
    "06b_v4_sha256": "4f69680a2d26d6c8e4bd5b337a5e94e6cdc36d149ea6a4b9af547b3855a93eab",
    "selftest_status": "06C_V7_TARGET_V1_FULL_DATA_RUNNER_SYNTHETIC_SELFTEST_PASS",
    "model_id": "V1_SHARED_PRIVATE_AE",
    "profile_id": "ARCH_COMPACT__REG_WEAK",
    "n_cases": 78,
    "primary_seed": 1103,
    "reproducibility_seeds": [2207, 3301],
    "fixed_epochs": 400,
    "feature_budgets": {'RNA': 5000, 'METHYLATION': 10000, 'CNV': 5000},
    "pca_components": {'RNA': 10, 'METHYLATION': 10, 'CNV': 10},
    "patient_shared_dim": 5,
    "fit_availability_candidate_index": 1,
    "encode_availability_candidate_index": 1,
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


def require_hash(label: str, path: Path, expected: str) -> str:
    observed = sha256_file(path)
    if observed.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA256 mismatch. "
            f"Expected={expected} Observed={observed} Path={path}"
        )
    return observed


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def literal_assignment(path: Path, variable_name: str) -> Any:
    tree = ast.parse(
        path.read_text(encoding="utf-8", errors="replace"),
        filename=str(path),
    )
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == variable_name
            for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError(
        f"{variable_name} assignment not found in {path}"
    )


def static_runner_audit(runner: Path) -> dict[str, Any]:
    source = runner.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source, filename=str(runner))

    # 1. Real TARGET loader use must be confined to run_real().
    real_loader_attribute_functions: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if any(
            isinstance(sub, ast.Attribute)
            and sub.attr == "load_authorized_target_inputs"
            for sub in ast.walk(node)
        ):
            real_loader_attribute_functions.append(node.name)

    if set(real_loader_attribute_functions) != {
        "verify_authorized_real_input_loader_contract",
        "run_real",
    }:
        raise RuntimeError(
            "Unexpected functions reference load_authorized_target_inputs: "
            f"{real_loader_attribute_functions}"
        )

    # 2. The actual loader invocation in run_real must be exactly
    #    load_inputs(runtime["modules"]).
    run_real = next(
        (
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "run_real"
        ),
        None,
    )
    if run_real is None:
        raise RuntimeError("Runner run_real() not found.")

    loader_calls: list[str] = []
    for node in ast.walk(run_real):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "load_inputs"
        ):
            loader_calls.append(ast.unparse(node))

    if loader_calls != ["load_inputs(runtime['modules'])"]:
        raise RuntimeError(
            f"Runner real loader call changed: {loader_calls}"
        )

    # 3. main() must route to synthetic and return when authorization is absent.
    main_node = next(
        (
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "main"
        ),
        None,
    )
    if main_node is None:
        raise RuntimeError("Runner main() not found.")
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

    # 4. load_authorization must expect this exact v7 namespace/status.
    load_auth = next(
        (
            n for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "load_authorization"
        ),
        None,
    )
    if load_auth is None:
        raise RuntimeError("Runner load_authorization() not found.")
    load_auth_text = ast.get_source_segment(source, load_auth) or ""
    for fragment in [
        '"06c2_v7"',
        '"target_v1_full_data_runner_v7_source_authorization_lock.json"',
        '"06C2_V7_TARGET_V1_FULL_DATA_RUNNER_SOURCE_AUTHORIZED"',
    ]:
        if fragment not in load_auth_text:
            raise RuntimeError(
                f"Runner authorization namespace changed: missing {fragment}"
            )

    # 5. Exact frozen implementation reuse path.
    for fragment in [
        'profile = v4.parse_frozen_profile(neural)',
        'variants = v4.build_variant_map(ablation)',
        'model, audit = ablation.train_fixed_epochs(',
        'candidate = ablation.encode_numpy(',
        'candidate["patient_shared"]',
        'load_authorized_target_inputs(modules)',
    ]:
        if fragment not in source:
            raise RuntimeError(
                f"Runner frozen reuse path changed: missing {fragment}"
            )

    if source.count(
        "break  # exact authorized-v4 first-success semantics"
    ) < 2:
        raise RuntimeError(
            "Separate ordered first-success fitting/encoding loops missing."
        )

    return {
        "real_loader_attribute_functions": real_loader_attribute_functions,
        "real_loader_call": loader_calls[0],
        "main_authorization_gate": "PASS",
        "authorization_namespace": "06c2_v7",
        "authorization_status": (
            "06C2_V7_TARGET_V1_FULL_DATA_RUNNER_SOURCE_AUTHORIZED"
        ),
        "exact_frozen_reuse_path": "PASS",
        "ordered_first_success_fit_and_encode": "PASS",
    }


def main() -> None:
    print("=" * 104)
    print(
        "Paper 5 - freeze exact TARGET V1 all-78 runner v7 "
        "source authorization"
    )
    print("=" * 104)
    print("Script version:", SCRIPT_VERSION)
    print("TARGET molecular values read in 06c2_v7: NO")
    print("TARGET outcomes/clinical/biology read in 06c2_v7: NO")
    print("TARGET model fit in 06c2_v7: NO")
    print("Scientific TARGET full-data results read in 06c2_v7: NO")

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    paper5 = project_root / "paper5"

    runner = (
        paper5 / "scripts"
        / "06c_run_target_v1_full_data_refit_gpu_v7.py"
    )
    contract = (
        paper5 / "method_contract" / "06c"
        / "target_v1_full_data_refit_contract.json"
    )
    lock06c = (
        paper5 / "method_contract" / "06c"
        / "06c_full_data_refit_contract_lock.json"
    )
    authorized_v4 = (
        paper5 / "scripts"
        / "06b_run_target_outcome_blind_representation_benchmark_gpu_v4.py"
    )
    selftest = (
        paper5 / "method_contract" / "06c_preflight"
        / "target_v1_full_data_runner_synthetic_selftest_v7.json"
    )
    v5_auth_path = (
        paper5 / "method_contract" / "06c2_v5"
        / "target_v1_full_data_runner_v5_source_authorization_lock.json"
    )

    for path in [
        runner,
        contract,
        lock06c,
        authorized_v4,
        selftest,
        v5_auth_path,
    ]:
        require_file(path)

    # Exact source and upstream identities.
    require_hash(
        "v7 runner",
        runner,
        EXPECTED["runner_sha256"],
    )
    require_hash(
        "06c contract",
        contract,
        EXPECTED["06c_contract_sha256"],
    )
    require_hash(
        "06c lock",
        lock06c,
        EXPECTED["06c_lock_sha256"],
    )
    require_hash(
        "authorized 06b v4",
        authorized_v4,
        EXPECTED["06b_v4_sha256"],
    )

    observed_runner_version = literal_assignment(
        runner,
        "SCRIPT_VERSION",
    )
    if observed_runner_version != EXPECTED["runner_script_version"]:
        raise RuntimeError(
            "v7 runner SCRIPT_VERSION mismatch. "
            f"Expected={EXPECTED['runner_script_version']} "
            f"Observed={observed_runner_version}"
        )

    runner_expected = literal_assignment(runner, "EXPECTED")
    for key in [
        "06c_contract_sha256",
        "06c_lock_sha256",
        "06b_v4_sha256",
        "primary_model_id",
        "profile_id",
        "n_cases",
        "primary_seed",
        "reproducibility_seeds",
        "fixed_epochs",
        "feature_budgets",
        "pca_components",
        "shared_latent_dim",
    ]:
        expected_value = {
            "primary_model_id": EXPECTED["model_id"],
            "shared_latent_dim": EXPECTED["patient_shared_dim"],
        }.get(key, EXPECTED.get(key))
        if runner_expected.get(key) != expected_value:
            raise RuntimeError(
                f"Runner EXPECTED[{key!r}] mismatch: "
                f"{runner_expected.get(key)!r} != {expected_value!r}"
            )

    frozen_contract = load_json(contract)
    frozen_lock = load_json(lock06c)
    st = load_json(selftest)
    v5_auth = load_json(v5_auth_path)

    # Frozen 06c scientific contract.
    if frozen_contract.get("status") != (
        "FROZEN_BEFORE_TARGET_FULL_DATA_REFIT"
    ):
        raise RuntimeError("06c contract status mismatch.")
    if frozen_lock.get("status") != (
        "06C_TARGET_V1_FULL_DATA_REFIT_CONTRACT_FROZEN"
    ):
        raise RuntimeError("06c lock status mismatch.")
    if frozen_lock.get("full_data_fit_performed") is not False:
        raise RuntimeError(
            "06c lock no longer proves pre-full-data-fit state."
        )
    if frozen_lock.get("q10_applied_to_v1") is not False:
        raise RuntimeError("06c q10 boundary changed.")

    full = frozen_contract.get("full_data_refit", {})
    training = full.get("training", {})
    if full.get("model_id") != EXPECTED["model_id"]:
        raise RuntimeError("06c model mismatch.")
    if full.get("profile_id") != EXPECTED["profile_id"]:
        raise RuntimeError("06c profile mismatch.")
    if int(full.get("n_cases")) != EXPECTED["n_cases"]:
        raise RuntimeError("06c n_cases mismatch.")
    if {
        k: int(v)
        for k, v in full.get("feature_budget", {}).items()
    } != EXPECTED["feature_budgets"]:
        raise RuntimeError("06c feature budgets mismatch.")
    if int(training.get("primary_seed")) != EXPECTED["primary_seed"]:
        raise RuntimeError("06c primary seed mismatch.")
    if [
        int(x)
        for x in training.get("reproducibility_seeds", [])
    ] != EXPECTED["reproducibility_seeds"]:
        raise RuntimeError("06c reproducibility seeds mismatch.")
    if int(training.get("fixed_epochs")) != EXPECTED["fixed_epochs"]:
        raise RuntimeError("06c fixed epochs mismatch.")
    if training.get("early_stopping_on_full_78") is not False:
        raise RuntimeError("06c full-data early-stopping firewall changed.")
    if training.get("post_fit_seed_selection") is not False:
        raise RuntimeError("06c post-fit seed-selection firewall changed.")
    if training.get("post_fit_epoch_selection") is not False:
        raise RuntimeError("06c post-fit epoch-selection firewall changed.")

    # Exact v7 synthetic PASS.
    if st.get("status") != EXPECTED["selftest_status"]:
        raise RuntimeError(
            f"v7 synthetic self-test status mismatch: {st.get('status')!r}"
        )
    if st.get("script_version") != EXPECTED["runner_script_version"]:
        raise RuntimeError("v7 synthetic self-test SCRIPT_VERSION mismatch.")
    if (
        str(st.get("runner_sha256", "")).lower()
        != EXPECTED["runner_sha256"]
    ):
        raise RuntimeError(
            "v7 synthetic self-test belongs to a different runner SHA."
        )
    if st.get("real_target_molecular_values_read") is not False:
        raise RuntimeError(
            "v7 synthetic self-test real TARGET molecular firewall mismatch."
        )
    if st.get("target_outcomes_clinical_biology_read") is not False:
        raise RuntimeError(
            "v7 synthetic clinical/outcome/biology firewall mismatch."
        )
    if st.get("target_scientific_fit_performed") is not False:
        raise RuntimeError(
            "v7 synthetic self-test unexpectedly performed TARGET fitting."
        )

    synthetic = st.get("synthetic", {})
    adapter = synthetic.get("adapter_audit", {})
    if int(
        adapter.get("fit_availability_candidate_index")
    ) != EXPECTED["fit_availability_candidate_index"]:
        raise RuntimeError(
            "v7 synthetic first-success fit candidate changed."
        )
    if int(
        adapter.get("encode_availability_candidate_index")
    ) != EXPECTED["encode_availability_candidate_index"]:
        raise RuntimeError(
            "v7 synthetic first-success encode candidate changed."
        )
    if list(
        synthetic.get("patient_shared_shape", [])
    ) != [36, EXPECTED["patient_shared_dim"]]:
        raise RuntimeError(
            "v7 synthetic patient_shared geometry mismatch: "
            f"{synthetic.get('patient_shared_shape')!r}"
        )
    if synthetic.get("smoke_epochs_are_scientific_setting") is not False:
        raise RuntimeError(
            "v7 synthetic smoke epochs incorrectly marked scientific."
        )

    # v7 self-test must include the cryptographic/static v5 boundary proof.
    v5_boundary = st.get(
        "authorized_v5_failure_boundary",
        {},
    )
    if (
        v5_boundary.get(
            "failure_occurs_before_loader_body_execution"
        )
        is not True
    ):
        raise RuntimeError(
            "v7 self-test lacks verified v5 pre-loader failure boundary."
        )
    if (
        v5_boundary.get(
            "scientific_result_directory_nonempty"
        )
        is not False
    ):
        raise RuntimeError(
            "v7 self-test reports scientific result directory after v5 failure."
        )
    if (
        v5_boundary.get(
            "06c3_result_lock_directory_nonempty"
        )
        is not False
    ):
        raise RuntimeError(
            "v7 self-test reports a 06c3 lock after v5 failure."
        )

    # Historical v5 authorization must still be intact.
    if v5_auth.get("status") != (
        "06C2_V5_TARGET_V1_FULL_DATA_RUNNER_SOURCE_AUTHORIZED"
    ):
        raise RuntimeError("Historical 06c2_v5 authorization status mismatch.")
    if (
        str(v5_auth.get("runner_source", {}).get("sha256", "")).lower()
        != "a2d87f4d7bd8ad7ca6493361ca2405944e8de2b4f3f3117215f3c54dd3d8c22c"
    ):
        raise RuntimeError("Historical 06c2_v5 runner SHA mismatch.")

    static_audit = static_runner_audit(runner)

    # Refuse post-hoc authorization if scientific outputs already exist.
    result_dir = (
        paper5 / "results" / "target_representation"
        / "06c_full_data_v1"
    )
    result_lock_dir = paper5 / "method_contract" / "06c3"
    if result_dir.exists() and any(result_dir.iterdir()):
        raise RuntimeError(
            "Scientific 06c full-data results already exist; "
            "refusing post-hoc v7 source authorization."
        )
    if result_lock_dir.exists() and any(result_lock_dir.iterdir()):
        raise RuntimeError(
            "06c3 result lock already exists; "
            "refusing post-hoc v7 source authorization."
        )

    out_dir = paper5 / "method_contract" / "06c2_v7"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(
            f"06c2_v7 output directory already exists and is non-empty: "
            f"{out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    lock_payload = {
        "status": (
            "06C2_V7_TARGET_V1_FULL_DATA_RUNNER_SOURCE_AUTHORIZED"
        ),
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_molecular_values_read": False,
        "target_outcomes_clinical_biology_read": False,
        "target_model_fit": False,
        "scientific_target_full_data_results_read": False,
        "runner_source": {
            "path": (
                "scripts/"
                "06c_run_target_v1_full_data_refit_gpu_v7.py"
            ),
            "sha256": EXPECTED["runner_sha256"],
            "script_version": EXPECTED["runner_script_version"],
            "size_bytes": runner.stat().st_size,
        },
        "synthetic_selftest": {
            "path": (
                "method_contract/06c_preflight/"
                "target_v1_full_data_runner_synthetic_selftest_v7.json"
            ),
            "sha256": sha256_file(selftest),
            "status": st.get("status"),
            "script_version": st.get("script_version"),
            "fit_availability_candidate_index": adapter.get(
                "fit_availability_candidate_index"
            ),
            "encode_availability_candidate_index": adapter.get(
                "encode_availability_candidate_index"
            ),
            "patient_shared_shape": synthetic.get(
                "patient_shared_shape"
            ),
            "real_target_molecular_values_read": st.get(
                "real_target_molecular_values_read"
            ),
            "target_scientific_fit_performed": st.get(
                "target_scientific_fit_performed"
            ),
        },
        "v5_authorized_attempt_boundary": {
            "source_authorization_lock_path": (
                "method_contract/06c2_v5/"
                "target_v1_full_data_runner_v5_source_authorization_lock.json"
            ),
            "source_authorization_lock_sha256": sha256_file(
                v5_auth_path
            ),
            "failure_occurs_before_loader_body_execution": True,
            "real_target_molecular_values_read": False,
            "target_scientific_fit_performed": False,
            "scientific_result_directory_nonempty_after_failure": False,
            "06c3_result_lock_directory_nonempty_after_failure": False,
        },
        "frozen_upstream": {
            "06c_contract_sha256": EXPECTED["06c_contract_sha256"],
            "06c_lock_sha256": EXPECTED["06c_lock_sha256"],
            "authorized_06b_v4_sha256": EXPECTED["06b_v4_sha256"],
        },
        "static_source_audit": static_audit,
        "authorized_scientific_execution": {
            "model_id": EXPECTED["model_id"],
            "profile_id": EXPECTED["profile_id"],
            "n_cases": EXPECTED["n_cases"],
            "feature_budgets": EXPECTED["feature_budgets"],
            "pca_components": EXPECTED["pca_components"],
            "primary_seed": EXPECTED["primary_seed"],
            "reproducibility_only_seeds": (
                EXPECTED["reproducibility_seeds"]
            ),
            "fixed_epochs": EXPECTED["fixed_epochs"],
            "real_input_loader_call": (
                "load_authorized_target_inputs(modules)"
            ),
            "availability_semantics": (
                "authorized_06b_v4_ordered_first_success"
            ),
            "post_fit_seed_selection": False,
            "post_fit_epoch_selection": False,
            "seed_ensemble_or_averaging": False,
            "q10_transfer_to_v1": False,
        },
        "historical_pre_authorization_boundaries": {
            "v1_sha256": (
                "06842dc803d17c3ea3789262655429481c081232ed8828b6e0d2cce56cb2703f"
            ),
            "v2_sha256": (
                "8901ea9dd3870ade3910934fba6106ae4b0fa4b8a6d2183e0cea9dd802962f9c"
            ),
            "v3_sha256": (
                "9d27534767434b1c8e41f46731ea8cd48c16fbdf1813ea3d85cf6a0cdfef1777"
            ),
            "v4_sha256": (
                "75297d8b4c92083b020d2396cad3c83ea236bd905f6e62f826dec14138925980"
            ),
            "v5_sha256": (
                "a2d87f4d7bd8ad7ca6493361ca2405944e8de2b4f3f3117215f3c54dd3d8c22c"
            ),
            "v6_sha256": (
                "6eda2be1ee0abe243fec386f193c9b4afba52f0f5d1e50d9a3f046a00a1491de"
            ),
            "v6_note": (
                "Synthetic PASS but deliberately not source-authorized "
                "because final pre-authorization review found stale "
                "diagnostic version labels. Scientific/runtime logic was "
                "unchanged."
            ),
        },
        "authorization_rule": (
            "Only this exact v7 runner SHA is authorized. Any source change, "
            "including comments or version labels, requires a new synthetic "
            "PASS and a new source-authorization stage before real TARGET "
            "execution."
        ),
    }

    lock_path = (
        out_dir
        / "target_v1_full_data_runner_v7_source_authorization_lock.json"
    )
    lock_path.write_text(
        json.dumps(lock_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Exact v7 runner SHA-256: PASS")
    print("Exact v7 runner SCRIPT_VERSION: PASS")
    print("Exact v7 runner EXPECTED scientific constants: PASS")
    print("Exact frozen 06c contract/lock identity: PASS")
    print("Exact authorized 06b v4 source identity: PASS")
    print("Exact v7 synthetic self-test identity: PASS")
    print("v7 synthetic first-success fit candidate index: 1")
    print("v7 synthetic first-success encode candidate index: 1")
    print("v7 synthetic patient_shared shape: (36, 5)")
    print("v5 pre-loader failure boundary: VERIFIED")
    print("v7 real-input call contract: load_authorized_target_inputs(modules)")
    print("v7 static authorization/real-loader firewall: PASS")
    print("Existing scientific full-data results before v7 authorization: NO")
    print("TARGET molecular values read in 06c2_v7: NO")
    print("TARGET model fit in 06c2_v7: NO")
    print("=" * 104)
    print(
        "Paper 5 06c2_v7 TARGET V1 all-78 runner "
        "source authorization: FROZEN"
    )
    print("=" * 104)
    print("Authorized runner SHA:", EXPECTED["runner_sha256"])
    print(
        "Authorization lock: "
        "method_contract/06c2_v7/"
        "target_v1_full_data_runner_v7_source_authorization_lock.json"
    )
    print("Next: rerun the exact same v7 runner unchanged.")
    print("=" * 104)


if __name__ == "__main__":
    main()
