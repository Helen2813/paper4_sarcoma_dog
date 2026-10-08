#!/usr/bin/env python3
"""
Paper 5 - freeze exact TARGET V1 all-78 runner v5 source authorization.

Authorization-only stage. It reads frozen contracts/source/self-test metadata.
It does NOT load TARGET molecular matrices, read TARGET clinical/outcome/biology
data, fit any TARGET model, or read scientific TARGET full-data results.
"""

from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_VERSION = (
    "06c2-freeze-target-v1-full-data-runner-v5-source-authorization-v1-no-cli"
)

EXPECTED = {
    "runner_sha256": "a2d87f4d7bd8ad7ca6493361ca2405944e8de2b4f3f3117215f3c54dd3d8c22c",
    "runner_script_version": (
        "06c-run-target-v1-full-data-refit-gpu-v5-"
        "exact-authorized-v4-profile-variant-path-two-stage-no-cli"
    ),
    "06c_contract_sha256": "8ab8516e30964f220d0f1204ca892e29ca337691e06cbcbf29315e1cb6aec2e5",
    "06c_lock_sha256": "1c3b4fb36706dc6afd0208fda23b1dc6d51c2953b4ccb9541099466aea13a374",
    "06b_v4_sha256": "4f69680a2d26d6c8e4bd5b337a5e94e6cdc36d149ea6a4b9af547b3855a93eab",
    "selftest_status": "06C_V5_TARGET_V1_FULL_DATA_RUNNER_SYNTHETIC_SELFTEST_PASS",
    "model_id": "V1_SHARED_PRIVATE_AE",
    "profile_id": "ARCH_COMPACT__REG_WEAK",
    "n_cases": 78,
    "primary_seed": 1103,
    "reproducibility_seeds": [2207, 3301],
    "fixed_epochs": 400,
    "patient_shared_dim": 5,
    "fit_availability_candidate_index": 1,
    "encode_availability_candidate_index": 1,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
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
            f"{label} SHA256 mismatch. Expected={expected} Observed={observed} Path={path}"
        )
    return observed


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def literal_assignment(path: Path, variable_name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == variable_name for t in node.targets):
                return ast.literal_eval(node.value)
    raise RuntimeError(f"{variable_name} assignment not found in {path}")


def static_runner_firewall_audit(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source, filename=str(path))

    functions_with_real_loader: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if any(
                isinstance(sub, ast.Attribute)
                and sub.attr == "load_authorized_target_inputs"
                for sub in ast.walk(node)
            ):
                functions_with_real_loader.append(node.name)

    if set(functions_with_real_loader) != {"run_real"}:
        raise RuntimeError(
            f"Real TARGET loader firewall changed: {functions_with_real_loader}"
        )

    main_node = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"),
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
            raise RuntimeError(f"Runner main() authorization gate changed: missing {fragment}")

    for fragment in [
        'profile = v4.parse_frozen_profile(neural)',
        'variants = v4.build_variant_map(ablation)',
        'model, audit = ablation.train_fixed_epochs(',
        'candidate = ablation.encode_numpy(',
        'candidate["patient_shared"]',
        'break  # exact authorized-v4 first-success semantics',
    ]:
        if fragment not in source:
            raise RuntimeError(f"Frozen reuse path changed: missing {fragment}")

    if source.count("break  # exact authorized-v4 first-success semantics") < 2:
        raise RuntimeError("Separate first-success fitting/encoding loops are missing.")

    return {
        "functions_with_real_target_loader": functions_with_real_loader,
        "main_authorization_gate": "PASS",
        "exact_frozen_reuse_path": "PASS",
        "separate_first_success_fit_and_encode": "PASS",
    }


def main() -> None:
    print("=" * 104)
    print("Paper 5 - freeze exact TARGET V1 all-78 runner v5 source authorization")
    print("=" * 104)
    print("Script version:", SCRIPT_VERSION)
    print("TARGET molecular values read in 06c2_v5: NO")
    print("TARGET outcomes/clinical/biology read in 06c2_v5: NO")
    print("TARGET model fit in 06c2_v5: NO")
    print("Scientific TARGET full-data results read in 06c2_v5: NO")

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    paper5 = project_root / "paper5"

    runner = paper5 / "scripts" / "06c_run_target_v1_full_data_refit_gpu_v5.py"
    contract = paper5 / "method_contract" / "06c" / "target_v1_full_data_refit_contract.json"
    lock06c = paper5 / "method_contract" / "06c" / "06c_full_data_refit_contract_lock.json"
    v4 = paper5 / "scripts" / "06b_run_target_outcome_blind_representation_benchmark_gpu_v4.py"
    selftest = paper5 / "method_contract" / "06c_preflight" / "target_v1_full_data_runner_synthetic_selftest_v5.json"

    for path in [runner, contract, lock06c, v4, selftest]:
        require_file(path)

    require_hash("v5 runner", runner, EXPECTED["runner_sha256"])
    require_hash("06c contract", contract, EXPECTED["06c_contract_sha256"])
    require_hash("06c lock", lock06c, EXPECTED["06c_lock_sha256"])
    require_hash("authorized 06b v4", v4, EXPECTED["06b_v4_sha256"])

    observed_runner_version = literal_assignment(runner, "SCRIPT_VERSION")
    if observed_runner_version != EXPECTED["runner_script_version"]:
        raise RuntimeError(
            f"Runner SCRIPT_VERSION mismatch. Expected={EXPECTED['runner_script_version']} "
            f"Observed={observed_runner_version}"
        )

    frozen_contract = load_json(contract)
    frozen_lock = load_json(lock06c)
    st = load_json(selftest)

    if frozen_contract.get("status") != "FROZEN_BEFORE_TARGET_FULL_DATA_REFIT":
        raise RuntimeError("06c contract status mismatch.")
    if frozen_lock.get("status") != "06C_TARGET_V1_FULL_DATA_REFIT_CONTRACT_FROZEN":
        raise RuntimeError("06c lock status mismatch.")
    if frozen_lock.get("full_data_fit_performed") is not False:
        raise RuntimeError("06c lock no longer proves pre-full-data-fit state.")
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
    if int(training.get("primary_seed")) != EXPECTED["primary_seed"]:
        raise RuntimeError("06c primary seed mismatch.")
    if [int(x) for x in training.get("reproducibility_seeds", [])] != EXPECTED["reproducibility_seeds"]:
        raise RuntimeError("06c reproducibility seeds mismatch.")
    if int(training.get("fixed_epochs")) != EXPECTED["fixed_epochs"]:
        raise RuntimeError("06c fixed epochs mismatch.")
    if training.get("post_fit_seed_selection") is not False:
        raise RuntimeError("06c post-fit seed-selection firewall changed.")
    if training.get("post_fit_epoch_selection") is not False:
        raise RuntimeError("06c post-fit epoch-selection firewall changed.")

    if st.get("status") != EXPECTED["selftest_status"]:
        raise RuntimeError(f"v5 self-test status mismatch: {st.get('status')!r}")
    if str(st.get("runner_sha256", "")).lower() != EXPECTED["runner_sha256"]:
        raise RuntimeError("v5 self-test belongs to a different runner SHA.")
    if st.get("real_target_molecular_values_read") is not False:
        raise RuntimeError("v5 self-test real TARGET firewall mismatch.")
    if st.get("target_outcomes_clinical_biology_read") is not False:
        raise RuntimeError("v5 self-test clinical/outcome/biology firewall mismatch.")
    if st.get("target_scientific_fit_performed") is not False:
        raise RuntimeError("v5 self-test unexpectedly fit scientific TARGET data.")

    synthetic = st.get("synthetic", {})
    adapter = synthetic.get("adapter_audit", {})
    if int(adapter.get("fit_availability_candidate_index")) != EXPECTED["fit_availability_candidate_index"]:
        raise RuntimeError("v5 fit first-success candidate changed.")
    if int(adapter.get("encode_availability_candidate_index")) != EXPECTED["encode_availability_candidate_index"]:
        raise RuntimeError("v5 encode first-success candidate changed.")
    if list(synthetic.get("patient_shared_shape", [])) != [36, EXPECTED["patient_shared_dim"]]:
        raise RuntimeError(f"v5 patient_shared geometry changed: {synthetic.get('patient_shared_shape')!r}")

    static_audit = static_runner_firewall_audit(runner)

    result_dir = paper5 / "results" / "target_representation" / "06c_full_data_v1"
    result_lock_dir = paper5 / "method_contract" / "06c3"
    if result_dir.exists() and any(result_dir.iterdir()):
        raise RuntimeError("Scientific 06c results already exist; refusing post-hoc authorization.")
    if result_lock_dir.exists() and any(result_lock_dir.iterdir()):
        raise RuntimeError("06c3 results locks already exist; refusing post-hoc authorization.")

    out_dir = paper5 / "method_contract" / "06c2_v5"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"06c2_v5 output directory already exists and is non-empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    lock_payload = {
        "status": "06C2_V5_TARGET_V1_FULL_DATA_RUNNER_SOURCE_AUTHORIZED",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_molecular_values_read": False,
        "target_outcomes_clinical_biology_read": False,
        "target_model_fit": False,
        "scientific_target_full_data_results_read": False,
        "runner_source": {
            "path": "scripts/06c_run_target_v1_full_data_refit_gpu_v5.py",
            "sha256": EXPECTED["runner_sha256"],
            "script_version": EXPECTED["runner_script_version"],
            "size_bytes": runner.stat().st_size,
        },
        "synthetic_selftest": {
            "path": "method_contract/06c_preflight/target_v1_full_data_runner_synthetic_selftest_v5.json",
            "sha256": sha256_file(selftest),
            "status": st.get("status"),
            "fit_availability_candidate_index": adapter.get("fit_availability_candidate_index"),
            "encode_availability_candidate_index": adapter.get("encode_availability_candidate_index"),
            "patient_shared_shape": synthetic.get("patient_shared_shape"),
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
            "primary_seed": EXPECTED["primary_seed"],
            "reproducibility_only_seeds": EXPECTED["reproducibility_seeds"],
            "fixed_epochs": EXPECTED["fixed_epochs"],
            "availability_semantics": "authorized_06b_v4_ordered_first_success",
            "post_fit_seed_selection": False,
            "post_fit_epoch_selection": False,
            "seed_ensemble_or_averaging": False,
            "q10_transfer_to_v1": False,
        },
        "historical_pre_authorization_boundaries": {
            "v1_sha256": "06842dc803d17c3ea3789262655429481c081232ed8828b6e0d2cce56cb2703f",
            "v2_sha256": "8901ea9dd3870ade3910934fba6106ae4b0fa4b8a6d2183e0cea9dd802962f9c",
            "v3_sha256": "9d27534767434b1c8e41f46731ea8cd48c16fbdf1813ea3d85cf6a0cdfef1777",
            "v4_sha256": "75297d8b4c92083b020d2396cad3c83ea236bd905f6e62f826dec14138925980",
            "all_v1_to_v4_real_target_molecular_values_read": False,
            "all_v1_to_v4_target_scientific_fit_performed": False,
            "v4_note": (
                "Synthetic PASS but not authorized because its internal SCRIPT_VERSION "
                "still carried the v3 identity label."
            ),
        },
        "authorization_rule": (
            "Only this exact runner SHA is authorized. Any source change requires "
            "a new synthetic PASS and new source authorization before real TARGET execution."
        ),
    }

    lock_path = out_dir / "target_v1_full_data_runner_v5_source_authorization_lock.json"
    lock_path.write_text(json.dumps(lock_payload, indent=2) + "\n", encoding="utf-8")

    print("Exact v5 runner SHA-256: PASS")
    print("Exact v5 runner SCRIPT_VERSION: PASS")
    print("Exact frozen 06c contract/lock identity: PASS")
    print("Exact authorized 06b v4 source identity: PASS")
    print("Exact v5 synthetic self-test identity: PASS")
    print("Synthetic first-success fit candidate index: 1")
    print("Synthetic first-success encode candidate index: 1")
    print("Synthetic patient_shared shape: (36, 5)")
    print("Runner static real-TARGET firewall: PASS")
    print("Existing scientific full-data results before authorization: NO")
    print("TARGET molecular values read in 06c2_v5: NO")
    print("TARGET model fit in 06c2_v5: NO")
    print("=" * 104)
    print("Paper 5 06c2_v5 TARGET V1 all-78 runner source authorization: FROZEN")
    print("=" * 104)
    print("Authorized runner SHA:", EXPECTED["runner_sha256"])
    print(
        "Authorization lock: method_contract/06c2_v5/"
        "target_v1_full_data_runner_v5_source_authorization_lock.json"
    )
    print("Next: rerun the exact same v5 runner unchanged.")
    print("=" * 104)


if __name__ == "__main__":
    main()
