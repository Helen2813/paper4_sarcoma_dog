#!/usr/bin/env python3
"""
Paper 5 - authorize exact 06c TARGET V1 all-78 runner source after synthetic self-test.

This stage reads no TARGET molecular values and performs no model fit.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_VERSION = "06c2-freeze-target-v1-full-data-runner-source-authorization-v1-no-cli"
EXPECTED_RUNNER_SHA256 = "06842dc803d17c3ea3789262655429481c081232ed8828b6e0d2cce56cb2703f"
EXPECTED_06C_CONTRACT_SHA256 = "8ab8516e30964f220d0f1204ca892e29ca337691e06cbcbf29315e1cb6aec2e5"
EXPECTED_06C_LOCK_SHA256 = "1c3b4fb36706dc6afd0208fda23b1dc6d51c2953b4ccb9541099466aea13a374"
EXPECTED_V4_SHA256 = "4f69680a2d26d6c8e4bd5b337a5e94e6cdc36d149ea6a4b9af547b3855a93eab"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    print("=" * 104)
    print("Paper 5 - authorize exact TARGET V1 all-78 runner source")
    print("=" * 104)
    print("Script version:", SCRIPT_VERSION)
    print("TARGET molecular values read in 06c2: NO")
    print("TARGET outcomes/clinical/biology read in 06c2: NO")
    print("TARGET model fit in 06c2: NO")
    print("Scientific TARGET results read in 06c2: NO")

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    paper5 = project_root / "paper5"

    runner = paper5 / "scripts" / "06c_run_target_v1_full_data_refit_gpu_v1.py"
    contract = paper5 / "method_contract" / "06c" / "target_v1_full_data_refit_contract.json"
    lock06c = paper5 / "method_contract" / "06c" / "06c_full_data_refit_contract_lock.json"
    v4 = paper5 / "scripts" / "06b_run_target_outcome_blind_representation_benchmark_gpu_v4.py"
    selftest = paper5 / "method_contract" / "06c_preflight" / "target_v1_full_data_runner_synthetic_selftest_v1.json"

    for p in [runner, contract, lock06c, v4, selftest]:
        if not p.is_file():
            raise FileNotFoundError(f"Required file missing: {p}")

    if sha256_file(runner) != EXPECTED_RUNNER_SHA256:
        raise RuntimeError("Exact 06c runner SHA mismatch.")
    if sha256_file(contract) != EXPECTED_06C_CONTRACT_SHA256:
        raise RuntimeError("06c contract SHA mismatch.")
    if sha256_file(lock06c) != EXPECTED_06C_LOCK_SHA256:
        raise RuntimeError("06c lock SHA mismatch.")
    if sha256_file(v4) != EXPECTED_V4_SHA256:
        raise RuntimeError("Authorized 06b v4 SHA mismatch.")

    st = load_json(selftest)
    if st.get("status") != "06C_TARGET_V1_FULL_DATA_RUNNER_SYNTHETIC_SELFTEST_PASS":
        raise RuntimeError("06c runner synthetic self-test status is not PASS.")
    if str(st.get("runner_sha256", "")).lower() != EXPECTED_RUNNER_SHA256:
        raise RuntimeError("Synthetic self-test belongs to a different runner SHA.")
    if st.get("real_target_molecular_values_read") is not False:
        raise RuntimeError("Synthetic self-test does not preserve real TARGET molecular firewall.")
    if st.get("target_scientific_fit_performed") is not False:
        raise RuntimeError("Synthetic self-test unexpectedly performed a TARGET scientific fit.")

    result_dir = paper5 / "results" / "target_representation" / "06c_full_data_v1"
    result_lock_dir = paper5 / "method_contract" / "06c3"
    if result_dir.exists() and any(result_dir.iterdir()):
        raise RuntimeError("Scientific 06c full-data result directory already exists and is non-empty.")
    if result_lock_dir.exists() and any(result_lock_dir.iterdir()):
        raise RuntimeError("06c3 result-lock directory already exists and is non-empty.")

    out_dir = paper5 / "method_contract" / "06c2"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"06c2 output directory already exists and is non-empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "status": "06C2_TARGET_V1_FULL_DATA_RUNNER_SOURCE_AUTHORIZED",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_molecular_values_read": False,
        "target_outcomes_clinical_biology_read": False,
        "target_model_fit": False,
        "scientific_target_results_read": False,
        "runner_source": {
            "path": "scripts/06c_run_target_v1_full_data_refit_gpu_v1.py",
            "sha256": EXPECTED_RUNNER_SHA256,
            "size_bytes": runner.stat().st_size
        },
        "synthetic_selftest": {
            "path": "method_contract/06c_preflight/target_v1_full_data_runner_synthetic_selftest_v1.json",
            "sha256": sha256_file(selftest),
            "status": st.get("status")
        },
        "06c_contract": {
            "sha256": EXPECTED_06C_CONTRACT_SHA256
        },
        "06c_lock": {
            "sha256": EXPECTED_06C_LOCK_SHA256
        },
        "authorized_06b_v4": {
            "sha256": EXPECTED_V4_SHA256
        },
        "authorized_scientific_execution": {
            "model_id": "V1_SHARED_PRIVATE_AE",
            "primary_seed": 1103,
            "reproducibility_only_seeds": [2207, 3301],
            "fixed_epochs": 400,
            "n_cases": 78,
            "q10_transfer_to_v1": False,
            "post_fit_seed_selection": False
        }
    }

    out = out_dir / "target_v1_full_data_runner_source_authorization_lock.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("Exact runner SHA-256: PASS")
    print("Exact frozen 06c identity: PASS")
    print("Authorized 06b v4 source identity: PASS")
    print("Synthetic runtime compatibility self-test: PASS")
    print("Real TARGET molecular values read in 06c2: NO")
    print("TARGET model fit in 06c2: NO")
    print("=" * 104)
    print("Paper 5 06c2 TARGET V1 all-78 runner source authorization: FROZEN")
    print("=" * 104)
    print("Authorized runner SHA:", EXPECTED_RUNNER_SHA256)
    print("Authorization lock: method_contract/06c2/target_v1_full_data_runner_source_authorization_lock.json")
    print("Next: rerun the exact same 06c runner unchanged.")
    print("=" * 104)


if __name__ == "__main__":
    main()
