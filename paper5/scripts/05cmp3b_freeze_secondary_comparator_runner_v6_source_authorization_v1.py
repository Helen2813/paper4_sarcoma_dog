#!/usr/bin/env python3
"""Paper 5: freeze exact 05cmp3 v6 source authorization; standard library only.

No CLI switches, runner imports, model fits, R/CUDA subprocesses or NPZ reads.
Only enumerates the two comparator result scopes to require zero files.
The first failed NPZ container was opened previously; this authorizer opens no NPZ.
The failed v5 source, successful preflight and authorization remain immutable history.
A NEW successful v6 preflight is mandatory; this program never creates one.
Publishes one complete JSON lock with an exclusive same-directory hard link.
An existing lock is never overwritten, amended, removed or silently accepted.
Run from the reviewed paper5/scripts filename. Do not run concurrently with
another authorizer/runner or change any inputs during this operation.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
import os
import stat
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable

SCRIPT_VERSION = "05cmp3b-freeze-secondary-comparator-runner-v6-source-authorization-v1-no-cli"
AUTHOR_FILENAME = "05cmp3b_freeze_secondary_comparator_runner_v6_source_authorization_v1.py"
RUNNER_REL = "paper5/scripts/05cmp3_run_secondary_comparator_confirmatory_benchmark_v6.py"
RUNNER_SHA = '26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a'
RUNNER_VERSION = '05cmp3-run-secondary-comparator-confirmatory-benchmark-v6-v2-npz-schema-postfit-biological-anchor-truth-serialized-preflight-source-amendment-no-cli'
PREAUTH_STATUS = "05CMP3_SECONDARY_COMPARATOR_RUNNER_SYNTHETIC_PREAUTH_PASS"
AUTH_STATUS = "05CMP3B_SECONDARY_COMPARATOR_RUNNER_V6_SOURCE_AUTHORIZED"
PREFLIGHT_REL = "paper5/method_contract/05cmp3_preflight/secondary_comparator_runner_synthetic_preflight_v6.json"
AUTH_REL = "paper5/method_contract/05cmp3b/secondary_comparator_runner_v6_source_authorization_lock.json"
OLD_RUNNER_REL = "paper5/scripts/05cmp3_run_secondary_comparator_confirmatory_benchmark_v5.py"
OLD_RUNNER_SHA = '802c131dfda75effb4e60dd38ebf05512fa5da6c1b3eec97c0c499b3c4e869f5'
OLD_PREFLIGHT_REL = 'paper5/method_contract/05cmp3_preflight/secondary_comparator_runner_synthetic_preflight_v5.json'
OLD_PREFLIGHT_SHA = '7ec4e687151a4af67b718a1ee13e521109a0071be91de2e751e9d0bb02b96b94'
HELPER_REL = "paper5/scripts/05c1_fit_linear_simulation_baselines.py"
HELPER_SHA = "c2c91b4ad274f78ccc369e6fca765102285cf3d9c8191f3c33f4a450b7426367"

OLD_AUTH_REL = "paper5/method_contract/05cmp3a/secondary_comparator_runner_source_authorization_lock.json"
OLD_AUTH_SHA = "5dbb075b3467fa865cb2b0570a79c2419e952a2c8c001eb04287d8ade5fba319"
OLD_AUTHOR_REL = "paper5/scripts/05cmp3a_freeze_secondary_comparator_runner_v5_source_authorization_v1.py"
OLD_AUTHOR_SHA = "526bf30c101dea38cb26d07c9acd625892febebcd697fb466be0c4c1fe653bfa"
GENERATOR_REL = "paper5/scripts/05b_generate_small_n_multiomics_simulations_v2.py"
GENERATOR_SHA = "b4adb1fcb32b65a998ee70e1b72d52e13490194c7ad8e5ddd802b95bf05a28db"
REVIEWED_EVIDENCE_ZIP_SHA = "0752c651b93a47505d3887f67630a421c31d279606472e3f018e7322ca5d9e1d"
REVIEWED_HISTORY = {
    OLD_RUNNER_REL: OLD_RUNNER_SHA, OLD_PREFLIGHT_REL: OLD_PREFLIGHT_SHA,
    OLD_AUTH_REL: OLD_AUTH_SHA, OLD_AUTHOR_REL: OLD_AUTHOR_SHA,
    GENERATOR_REL: GENERATOR_SHA, HELPER_REL: HELPER_SHA,
}
POSITIVE_SCHEMA_CASES = [
    "all3", "partial_RNA_METH", "partial_RNA_CNV", "partial_METH_CNV",
    "mixed_support_and_pair_corruption", "rotational_subspace_only",
    "technical_only_biological_null",
]
NEGATIVE_SCHEMA_CASES = [
    "legacy_Z_shared_only_rejected", "dual_truth_keys_rejected",
    "missing_biological_truth_rejected", "sample_index_reordering_rejected",
    "view_shape_mismatch_rejected", "RNA_anchor_repair_forbidden",
    "biological_column_count_mismatch_rejected", "size_mismatch_rejected", "SHA_mismatch_rejected",
]

RESULT_SCOPES = (
    "paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1",
    "paper5/method_contract/05cmp4",
)
SCIENTIFIC_STATE = {
    "frozen_simulation_npz_opened": False,
    "frozen_simulation_truth_opened": False,
    "scientific_comparator_fit_performed": False,
    "TARGET_accessed": False,
    "outcomes_clinical_biology_accessed": False,
}

FROZEN_INPUTS = {'cmp2_contract': ('paper5/method_contract/05cmp2/secondary_comparator_confirmatory_benchmark_contract.json',
                   '8e0a4d95a1f554af8e9c5188ca683d32c336c6bd5b797b3a91426adbe4ee9fc2'),
 'cmp2_lock': ('paper5/method_contract/05cmp2/secondary_comparator_confirmatory_benchmark_contract_lock.json',
               'a55d3e3498d9cc6a4acaaab64e5098007b093071b60999c82cf8e124a0301b5b'),
 'cmp2_registry': ('paper5/method_contract/05cmp2/secondary_comparator_confirmatory_dataset_registry.csv',
                   '4b72b94c4df49dbbbd1a90d035855e628a4fa50eede0451dfe796227bda53314'),
 'cmp1_contract': ('paper5/method_contract/05cmp1/secondary_comparator_implementation_contract.json',
                   '418259cedd36cbd1913a4ff1ae9874a3decfddbb7a75e9b3bca2e9623f0aca95'),
 'cmp1_lock': ('paper5/method_contract/05cmp1/secondary_comparator_implementation_contract_lock.json',
               '37e0db502a88f7beab52f1457cb0c5367091314faa6b86c7796810d883f31648'),
 'cmp1_impl': ('paper5/method_contract/05cmp1/secondary_comparator_implementation_registry.csv',
               '85dbc68289bae71b9cf4f8a01af34b09d50750fde97f0b57ef3b6c57ebf299b0'),
 'cmp1_adapter': ('paper5/method_contract/05cmp1/secondary_comparator_scientific_adapter_registry.csv',
                  'f30c73d19627d357073786b75f7b8141a757becf9d2413491518948cc099ea48'),
 'cmp1_seed': ('paper5/method_contract/05cmp1/secondary_comparator_seed_registry.csv',
               '2c70c516ad07496a71fe513139b357b92f46fd746bea5258f1fb237028c4e87c'),
 'cmp1_metric': ('paper5/method_contract/05cmp1/secondary_comparator_metric_eligibility_registry.csv',
                 '8802d8d4bba20cfd4acc3263b12f0a119bcef4e4107164e6f944d0a01d9bef46'),
 'manifest05b': ('paper5/method_contract/05b_v2/architecture_screening_data_manifest.csv',
                 'f3980e88aff408b65cbfce96aca50bad4fc08ecece2bc42bbc04f365c35d9c87')}

ENVIRONMENT_EXPECTED = {'move_version': '1.5.0',
 'torch_version': '2.12.1+cu130',
 'cuda_build': '13.0',
 'rscript_sha256': 'd829bcf7e9fa1d7e3e828c565c3cdbb1ed416f551f4fa6fd4dfcdf231e33e5e8',
 'icluster_package_tree_sha256': 'cad0c995150e778026fb227ac533a974d77106e431c65a3802aeda4254f6fd70'}


class AuthorizationError(RuntimeError):
    """Fail-closed authorizer error; no fallback or input repair is permitted."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorizationError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_json(data: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in items:
            require(key not in out, f"Duplicate JSON key in {label}: {key}")
            out[key] = value
        return out

    def reject_constant(value: str) -> None:
        raise AuthorizationError(f"Non-finite JSON constant in {label}: {value}")

    value = json.loads(data.decode("utf-8-sig"), object_pairs_hook=pairs,
                       parse_constant=reject_constant)
    require(isinstance(value, dict), f"Expected JSON object: {label}")
    return value


def project_path(root: Path, relative: str) -> Path:
    rel = Path(relative)
    require(not rel.is_absolute() and ".." not in rel.parts,
            f"Non-project-relative path rejected: {relative}")
    path = root / rel
    # Static, reviewed paths only. Refuse links/reparse points rather than
    # following them to scientific data or a different project location.
    current = root
    for part in rel.parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        linked = stat.S_ISLNK(info.st_mode) or bool(
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        )
        require(not linked, f"Symlink/reparse path rejected: {current}")
    require(path.resolve().is_relative_to(root.resolve()),
            f"Path escapes the project: {path}")
    return path


def read_snapshot(root: Path, relative: str, expected: str | None = None
                  ) -> tuple[bytes, dict[str, Any]]:
    path = project_path(root, relative)
    require(path.is_file(), f"Required file missing: {path}")
    data = path.read_bytes()
    digest = sha256_bytes(data)
    if expected is not None:
        require(digest == expected,
                f"SHA-256 mismatch for {relative}: {digest} != {expected}. "
                "Do not edit hashes or rewrite the input to bypass this check.")
    return data, {"path": relative, "sha256": digest, "size_bytes": len(data)}


def assert_empty_results(root: Path) -> None:
    # Directory metadata only: never read, hash, import or parse a result file.
    for relative in RESULT_SCOPES:
        path = project_path(root, relative)
        if not path.exists():
            continue
        require(path.is_dir(), f"Unexpected non-directory result scope: {path}")

        def walk_error(error: OSError) -> None:
            raise AuthorizationError(f"Cannot inspect result directory: {error}")

        for directory, dirs, files in os.walk(path, followlinks=False, onerror=walk_error):
            require(not files, f"Pre-authorization result files already exist in {directory}. "
                    "Do not delete them or create a replacement authorization.")
            for name in dirs:
                project_path(root, (Path(directory) / name).relative_to(root).as_posix())


def assert_no_authorization(root: Path) -> Path:
    path = project_path(root, AUTH_REL)
    require(not path.exists(), f"Authorization already exists: {path}. "
            "It was not changed. Do not rerun the authorizer or remove the lock.")
    return path


def top_assignments(tree: ast.Module) -> dict[str, ast.expr]:
    return {node.targets[0].id: node.value for node in tree.body
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)}


def validate_runner(data: bytes) -> None:
    compile(data, RUNNER_REL, "exec", dont_inherit=True)  # Compilation only.
    tree = ast.parse(data.decode("utf-8-sig"), feature_version=(3, 10))
    assignments = top_assignments(tree)
    for key, expected in (("SCRIPT_VERSION", RUNNER_VERSION),
                          ("PREAUTH_STATUS", PREAUTH_STATUS),
                          ("AUTH_STATUS", AUTH_STATUS)):
        require(key in assignments and ast.literal_eval(assignments[key]) == expected,
                f"Runner constant mismatch: {key}")
    require(ast.literal_eval(assignments["RUNNER_FILENAME"]) == Path(RUNNER_REL).name,
            "Runner filename binding mismatch.")
    require(ast.literal_eval(assignments["AMENDMENT_REVIEWED_INPUTS"]) == REVIEWED_HISTORY,
            "Runner reviewed-history bindings mismatch.")
    node = assignments.get("EXPECTED")
    require(isinstance(node, ast.Dict), "Runner EXPECTED is not a literal dictionary.")
    fields = {ast.literal_eval(key): value for key, value in zip(node.keys, node.values)}
    for key, (_, digest) in FROZEN_INPUTS.items():
        require(ast.literal_eval(fields[key + "_sha256"]) == digest,
                f"Authorizer/runner frozen binding mismatch: {key}")
    for key, value in (("n_datasets", 360), ("n_scenarios", 36),
                       ("methods", ["iCluster2b", "MOVE"]), ("latent_dim", 5)):
        require(ast.literal_eval(fields[key]) == value, f"Runner scope mismatch: {key}")


def integer(text: str, label: str) -> int:
    try:
        value = Decimal(text.strip())
    except (InvalidOperation, AttributeError) as exc:
        raise AuthorizationError(f"Invalid integer metadata: {label}={text!r}") from exc
    require(value.is_finite() and value == value.to_integral_value(),
            f"Non-integer metadata: {label}={text!r}")
    return int(value)


def validate_registry(data: bytes) -> dict[str, Any]:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    fields = reader.fieldnames or []
    needed = {"comparator_dataset_id", "scenario_id", "replicate_index", "simulation_seed"}
    require(needed.issubset(fields), f"Registry is missing columns: {sorted(needed-set(fields))}")
    require(len(fields) == len(set(fields)), "Duplicate registry column names.")
    rows = list(reader)
    require(len(rows) == 360, f"Registry row count: {len(rows)} != 360")
    ids: set[str] = set()
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        require(None not in row and all(value is not None for value in row.values()),
                "Malformed CSV row in frozen registry.")
        dataset_id = row["comparator_dataset_id"]
        scenario = row["scenario_id"]
        require(bool(dataset_id.strip()) and dataset_id not in ids,
                "Empty or duplicate comparator_dataset_id.")
        require(bool(scenario.strip()), "Empty scenario_id.")
        ids.add(dataset_id)
        grouped[scenario].append(integer(row["replicate_index"], "replicate_index"))
        integer(row["simulation_seed"], "simulation_seed")
    require(len(grouped) == 36, f"Registry scenario count: {len(grouped)} != 36")
    required_replicates = list(range(11, 21))
    for scenario, replicates in grouped.items():
        require(sorted(replicates) == required_replicates,
                f"Incomplete/duplicate confirmatory replicates for {scenario}.")
    return {"n_datasets": 360, "n_scenarios": 36,
            "replicate_indices": required_replicates, "methods": ["iCluster2b", "MOVE"],
            "expected_method_dataset_shards": 720,
            "simulation_npz_paths_followed": False}


def mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"Expected object: {label}")
    return value


def finite_number(value: Any, label: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value),
            f"Non-finite or nonnumeric value: {label}")
    return float(value)


def validate_preflight(report: dict[str, Any]) -> None:
    for key, expected in (
        ("status", PREAUTH_STATUS), ("execution_mode", "SYNTHETIC_ONLY_PREAUTHORIZATION"),
        ("script_version", RUNNER_VERSION), ("runner_sha256", RUNNER_SHA),
        ("cmp2_contract_sha256", FROZEN_INPUTS["cmp2_contract"][1]),
        ("cmp2_registry_sha256", FROZEN_INPUTS["cmp2_registry"][1]),
    ):
        require(report.get(key) == expected, f"v6 preflight mismatch: {key}")
    helper = mapping(report.get("linear_helper_source"), "linear_helper_source")
    require(helper.get("path") == HELPER_REL and helper.get("sha256") == HELPER_SHA,
            "v6 preflight does not bind the reviewed linear helper.")
    state = mapping(report.get("scientific_state"), "scientific_state")
    for key in SCIENTIFIC_STATE:
        require(state.get(key) is False, f"Preflight scientific firewall is not false: {key}")
    environment = mapping(report.get("environment"), "environment")
    for key, value in ENVIRONMENT_EXPECTED.items():
        require(environment.get(key) == value, f"Preflight environment mismatch: {key}")
    require(isinstance(environment.get("gpu"), str) and bool(environment["gpu"].strip()),
            "Preflight GPU evidence is missing.")
    timestamp = datetime.fromisoformat(str(report.get("created_at_utc", "")).replace("Z", "+00:00"))
    require(timestamp.tzinfo is not None, "Preflight timestamp lacks a timezone.")
    tests = mapping(report.get("tests"), "tests")
    for key in ("adapter_feature_and_whole_view_missing", "deterministic_seed_derivation",
                "frozen_upstream_metadata"):
        require(tests.get(key) == "PASS", f"Synthetic test not PASS: {key}")
    for key in ("move_exact_api_synthetic_fit", "icluster2b_exact_tune_api_synthetic_fit"):
        fit = mapping(tests.get(key), key)
        require(fit.get("finite") is True and fit.get("latent_shape") == [24, 5],
                f"Synthetic fit is not a finite 24-by-5 result: {key}")
    move = tests["move_exact_api_synthetic_fit"]
    require(finite_number(move.get("synthetic_reconstruction_sse"), "synthetic SSE") >= 0,
            "Negative synthetic SSE.")
    lambdas = tests["icluster2b_exact_tune_api_synthetic_fit"].get("lambda")
    require(isinstance(lambdas, list) and len(lambdas) == 3, "Synthetic lambda shape mismatch.")
    for value in lambdas:
        require(10 <= finite_number(value, "synthetic lambda") <= 500,
                "Synthetic lambda outside the unchanged frozen interval.")
    selection = mapping(tests.get("move_selection_logic"), "move_selection_logic")
    require(selection.get("deterministic_folds") == "PASS"
            and selection.get("one_se_retained") == ["A", "B"]
            and selection.get("stage2_selected") == "B", "MOVE selection fixture failed.")
    metric = mapping(tests.get("paper5_subspace_helper_replay"), "metric replay")
    for key in ("factor_alignment", "subspace_recovery"):
        require(abs(finite_number(metric.get(key), key) - 1.0) <= 1e-10,
                f"Exact synthetic metric fixture failed: {key}")
    replay = mapping(metric.get("subspace_helper_replay"), "subspace_helper_replay")
    require(replay.get("status") == "PASS_FROZEN_HELPER_NAMED_SCALAR"
            and replay.get("helper_scalar_path") == "mean_canonical_correlation",
            "Frozen named metric-helper replay mismatch.")


    schema = mapping(tests.get("serialized_v2_npz_schema_and_truth"), "serialized schema test")
    require(schema.get("status") == "PASS_SERIALIZED_V2_NPZ_LOADERS_AND_TRUTH_SEMANTICS",
            "Actual serialized v2 NPZ loaders have not passed.")
    require(schema.get("positive_cases") == POSITIVE_SCHEMA_CASES
            and schema.get("negative_cases") == NEGATIVE_SCHEMA_CASES,
            "Serialized schema positive/negative coverage mismatch.")
    for key in ("observed_input_payloads_only", "truth_payload_read_only_in_explicit_truth_stage"):
        require(schema.get(key) is True, f"Schema read firewall failed: {key}")
    for key in ("technical_and_private_truth_payloads_decoded",
                "biological_columns_filtered_by_view_support", "frozen_scientific_NPZ_opened"):
        require(schema.get(key) is False, f"Schema firewall failed: {key}")
    require(schema.get("RNA_anchor_helper") == "05c1.integrated_anchor_truth(arrays, 'biological')",
            "Integrated biological anchor helper mismatch.")
    require(schema.get("tracked_payload_reads") == 49, "Payload tracing coverage mismatch.")
    amendment = mapping(report.get("schema_amendment"), "schema_amendment")
    require(amendment.get("historical_authorized_v5_preserved") is True
            and amendment.get("current_invocation_scientific_npz_values_decoded") is False
            and amendment.get("truth_key") == "Z_biological_shared"
            and amendment.get("truth_stage") == "after each method fit/architecture selection completes",
            "Schema-amendment state mismatch.")
    history = mapping(amendment.get("history_bindings"), "history_bindings")
    require(set(history) == set(REVIEWED_HISTORY), "Incomplete preflight history bindings.")
    for relative, digest in REVIEWED_HISTORY.items():
        require(mapping(history[relative], relative).get("sha256") == digest,
                f"Preflight history hash mismatch: {relative}")


def atomic_publish(path: Path, content: bytes, recheck: Callable[[], None]) -> None:
    """Publish complete bytes without overwrite. No weaker fallback is used."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".05cmp3b_pending_", suffix=".tmp", dir=path.parent)
    pending = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        require(pending.read_bytes() == content, "Pending authorization verification failed.")
        recheck()
        # Same directory/volume. A pre-existing destination makes this fail.
        # On filesystems without hard-link support we stop, not overwrite.
        os.link(pending, path)
    finally:
        try:
            pending.unlink()
        except FileNotFoundError:
            pass


def authorize(project_root: Path, authorizer_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    authorizer_path = authorizer_path.resolve()
    expected_self = root / "paper5" / "scripts" / AUTHOR_FILENAME
    require(authorizer_path == expected_self, "Place the authorizer under paper5/scripts "
            f"with the exact filename {AUTHOR_FILENAME}")
    auth_path = assert_no_authorization(root)
    assert_empty_results(root)
    snapshots: dict[str, dict[str, Any]] = {}

    def capture(relative: str, expected: str | None = None) -> bytes:
        data, record = read_snapshot(root, relative, expected)
        snapshots[relative] = record
        return data

    capture(authorizer_path.relative_to(root).as_posix())
    runner = capture(RUNNER_REL, RUNNER_SHA)
    validate_runner(runner)
    print("Exact v6 runner SHA/version and source bindings: PASS")
    capture(OLD_RUNNER_REL, OLD_RUNNER_SHA)
    capture(OLD_PREFLIGHT_REL, OLD_PREFLIGHT_SHA)
    old_auth = strict_json(capture(OLD_AUTH_REL, OLD_AUTH_SHA), "historical v5 authorization")
    require(old_auth.get("runner_source", {}).get("sha256") == OLD_RUNNER_SHA,
            "Historical authorization runner binding mismatch.")
    capture(OLD_AUTHOR_REL, OLD_AUTHOR_SHA)
    capture(GENERATOR_REL, GENERATOR_SHA)
    print("Historical v5 source/preflight/authorization and generator: PASS")
    blobs = {key: capture(relative, digest) for key, (relative, digest) in FROZEN_INPUTS.items()}
    contract = strict_json(blobs["cmp2_contract"], "05cmp2 contract")
    contract_lock = strict_json(blobs["cmp2_lock"], "05cmp2 lock")
    require(contract.get("status") ==
            "FROZEN_SECONDARY_COMPARATOR_CONFIRMATORY_BENCHMARK_BEFORE_SCIENTIFIC_FITS",
            "Frozen 05cmp2 contract status mismatch.")
    require(contract_lock.get("status") ==
            "05CMP2_SECONDARY_COMPARATOR_CONFIRMATORY_BENCHMARK_CONTRACT_FROZEN",
            "Frozen 05cmp2 lock status mismatch.")
    scope = validate_registry(blobs["cmp2_registry"])
    print("Ten exact frozen upstream files and 360-row registry: PASS")
    capture(HELPER_REL, HELPER_SHA)
    print("Frozen linear metric-helper SHA: PASS")
    preflight_bytes = capture(PREFLIGHT_REL)
    preflight = strict_json(preflight_bytes, "NEW v6 preflight")
    validate_preflight(preflight)
    print("NEW v6 synthetic preflight and all scientific firewalls: PASS")

    def record(relative: str) -> dict[str, Any]:
        return dict(snapshots[relative])

    payload: dict[str, Any] = {
        "status": AUTH_STATUS,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_source": {**record(RUNNER_REL), "script_version": RUNNER_VERSION},
        "cmp2_contract": record(FROZEN_INPUTS["cmp2_contract"][0]),
        "synthetic_preflight": record(PREFLIGHT_REL),
        "linear_helper_source": record(HELPER_REL),
        "authorizer_source": record(authorizer_path.relative_to(root).as_posix()),
        "frozen_upstream_files": {key: record(relative)
                                  for key, (relative, _) in FROZEN_INPUTS.items()},
        "source_history": {
            "v5_source": record(OLD_RUNNER_REL),
            "v5_original_successful_preflight": record(OLD_PREFLIGHT_REL),
            "v5_authorization": record(OLD_AUTH_REL),
            "v5_authorizer": record(OLD_AUTHOR_REL),
            "generator_source": record(GENERATOR_REL),
            "reviewed_evidence_zip_sha256": REVIEWED_EVIDENCE_ZIP_SHA,
            "v6_changes": [
                "strict generator-v2 NPZ observed-input loader; no legacy aliases",
                "biological truth decoded only after fit/selection via frozen integrated_anchor_truth",
                "synthetic serialized NPZ integration regression and guard cases",
                "separate v6 preflight and additive 05cmp3b authorization gate",
                "exact retained v5/generator/helper source bindings",
                "final result lock identifies current cmp3b_authorization explicitly",
            ],
            "fitting_selection_and_metric_formula_code_changed": False,
            "input_validation_and_truth_loading_code_changed": True,
            "historical_preflight_relabelled_as_v6": False,
            "v5_is_preserved_as_failed_schema_execution_not_a_scientific_method_failure": True,
        },
        "execution_scope": scope,
        "environment_evidence_from_v6_preflight": preflight["environment"],
        "preauthorization_scientific_state": dict(SCIENTIFIC_STATE),
        "historical_access_disclosure": {
            "source": "reviewed schema evidence and v5 source/traceback; not new NPZ access here",
            "first_dataset_id": "A01_N62_STANDARD__R11__S1956851480",
            "previous_authorized_v5_npz_bytes_hashed_and_container_opened": True,
            "previous_v5_array_values_decoded_before_missing_key_error": False,
            "followup_diagnostic_read_npy_headers": True,
            "followup_diagnostic_decoded_array_values": False,
            "scientific_fit_started_in_failed_v5_attempt": False,
        },
        "authorizer_actions": {
            "runner_or_helper_imported": False, "R_or_CUDA_started": False,
            "model_fit_performed": False, "simulation_npz_read": False,
            "scientific_result_contents_read": False,
            "result_directory_metadata_checked": True,
            "existing_sources_preflights_contracts_modified": False,
        },
        "scope_limits": {
            "TARGET_outcomes_clinical_biology_access_authorized": False,
            "05e_final_gate_or_TARGET_selection_changes_authorized": False,
            "fitting_selection_or_metric_formula_changes_authorized": False,
            "data_interface_and_truth_access_order_correction_authorized": True,
            "fallback_authorized": False,
            "benchmark_started_by_authorizer": False,
            "full_scientific_success_guaranteed": False,
        },
        "rewrite_policy": "Do not overwrite/delete/rebind this lock. Changes require a separate "
                          "versioned review and successful matching preflight before scientific use.",
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    require(strict_json(encoded, "authorization payload") == payload,
            "Authorization serialization round-trip failed.")

    def recheck() -> None:
        assert_no_authorization(root)
        assert_empty_results(root)
        for relative, previous in snapshots.items():
            _, current = read_snapshot(root, relative, previous["sha256"])
            require(current == previous, f"Input changed before publication: {relative}")

    recheck()
    atomic_publish(auth_path, encoded, recheck)
    require(auth_path.read_bytes() == encoded,
            "Authorization post-write verification failed. Preserve the lock and report the error.")
    print("Final read-only input recheck before publication: PASS")
    print("Paper 5 05cmp3b v6 source authorization: FROZEN")
    print(f"Authorization: {AUTH_REL}")
    print(f"Authorization SHA-256: {sha256_bytes(encoded)}")
    print(f"Runner SHA-256: {RUNNER_SHA}")
    print(f"Synthetic preflight SHA-256: {snapshots[PREFLIGHT_REL]['sha256']}")
    print("Scientific benchmark started: NO")
    print("STOP HERE. A subsequent v6 runner invocation will enter scientific mode.")
    return payload


def main() -> int:
    print("=" * 104)
    print("Paper 5 - 05cmp3b exact v6 source authorizer (no scientific execution)")
    print(f"Script version: {SCRIPT_VERSION}")
    print("=" * 104)
    try:
        require(len(sys.argv) == 1, "No command-line arguments are accepted.")
        require(sys.version_info >= (3, 10), "Python 3.10 or newer is required.")
        path = Path(__file__).resolve()
        require(path.name == AUTHOR_FILENAME and path.parent.name == "scripts"
                and path.parent.parent.name == "paper5",
                "Incorrect authorizer filename/location; expected paper5/scripts/" + AUTHOR_FILENAME)
        authorize(path.parents[2], path)
    except (AuthorizationError, OSError, ValueError, TypeError, KeyError, SyntaxError) as exc:
        print(f"AUTHORIZATION STOPPED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Benchmark was not started. Do not edit hashes, delete evidence or bypass the error.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
