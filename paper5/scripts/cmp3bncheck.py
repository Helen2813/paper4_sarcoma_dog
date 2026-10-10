#!/usr/bin/env python3
"""Paper 5: v6 singleton-batch diagnosis and candidate validation, NOT authorization.

Reads the pinned v6 source/metadata and existing shard provenance/errors.
Imports v6 under a non-main name, then runs ONLY fresh artificial MOVE inputs.
The candidate loader is substituted in memory inside this process, never on disk.
All scientific loading/fitting entry points and R subprocess entry points are blocked.
The only persistent output is a new evidence ZIP. No installed packages are modified.
"""
from __future__ import annotations

import ast
import gc
import hashlib
import importlib.util
import json
import os
import stat
import sys
import traceback
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

VERSION = "05cmp3-v6-singleton-batch-probe-v1-synthetic-only-not-authorizer"
FILENAME = "cmp3bncheck.py"
RUNNER_REL = "paper5/scripts/05cmp3_run_secondary_comparator_confirmatory_benchmark_v6.py"
RUNNER_SHA = "26c2a2c7efc84e52969ffc2c1f2380406b4a42b00e756fbccd798bc0c433b44a"
AUTH_REL = "paper5/method_contract/05cmp3b/secondary_comparator_runner_v6_source_authorization_lock.json"
AUTH_SHA = "cca95c069ae27b1c8adf7ad17d577ba38e80f3f5df778fe60068c67e3062675e"
PREFLIGHT_REL = "paper5/method_contract/05cmp3_preflight/secondary_comparator_runner_synthetic_preflight_v6.json"
PREFLIGHT_SHA = "e213d756bfcbef5ee01e46e72cec361ca39b3656674b8cba32afd5b821e49179"
RESULT_REL = "paper5/results/simulation_benchmark/05cmp_secondary_comparators/v1"
CONTRACT_SHA = "8e0a4d95a1f554af8e9c5188ca683d32c336c6bd5b797b3a91426adbe4ee9fc2"
POLICY = {
    "status": "PROPOSED_IMPLEMENTATION_AMENDMENT_NOT_AUTHORIZED",
    "base_batch_size": "min(16, n_fit_samples)",
    "trigger": "training only; final batch has exactly one sample",
    "action": "move the final index of the preceding full batch to the front of the singleton batch",
    "example_n49": [[16, 16, 16, 1], [16, 16, 15, 2]],
    "all_samples_once_per_epoch": True,
    "same_flattened_shuffle_order": True,
    "same_batch_count_and_optimizer_step_count": True,
    "training_BatchNorm_retained": True,
    "training_batch_boundaries_changed": True,
    "numerical_training_equivalence_claimed": False,
    "result_based_batch_size_search": False,
    "projection_loader_changed": False,
    "old_scientific_records_or_authorization_overwritten": False,
}


class ProbeError(RuntimeError):
    pass


def require(ok: bool, text: str) -> None:
    if not ok:
        raise ProbeError(text)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    rel = Path(relative)
    require(not rel.is_absolute() and ".." not in rel.parts, "Unsafe relative path")
    target = root / rel
    current = root
    for part in rel.parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(info.st_mode) and not bool(
            getattr(info, "st_file_attributes", 0) & 0x400),
            f"Symlink/reparse path rejected: {current}")
    require(target.resolve().is_relative_to(root.resolve()), "Path escapes project")
    return target


def json_object(data: bytes, label: str) -> dict[str, Any]:
    def pairs(items):
        obj = {}
        for key, value in items:
            require(key not in obj, f"Duplicate JSON key in {label}: {key}")
            obj[key] = value
        return obj
    obj = json.loads(data.decode("utf-8-sig"), object_pairs_hook=pairs)
    require(isinstance(obj, dict), f"Expected JSON object: {label}")
    return obj


def record_file(root: Path, relative: str, expected: str | None = None):
    path = safe_path(root, relative)
    require(path.is_file(), f"Missing file: {path}")
    data = path.read_bytes()
    digest = sha(data)
    require(expected is None or digest == expected,
            f"SHA mismatch: {relative}: {digest}; do not change the pin or file")
    return data, {"path": relative, "size_bytes": len(data), "sha256": digest}


def classify_error(text: Any) -> str:
    if not isinstance(text, str):
        return "MISSING_ERROR_TEXT"
    if ("Expected more than 1 value per channel when training" in text
            and ("torch.Size([1," in text or "got input size [1," in text)
            and "select_move_architecture" in text and "fit_move" in text
            and "batch_norm" in text):
        return "SINGLETON_BATCHNORM_DURING_MOVE_SELECTION"
    return "OTHER_ERROR_REQUIRES_REVIEW"


def inventory_results(root: Path, valid_ids: set[str]):
    """Parse only for provenance/status/error classification, not metric comparison.

    Whole JSON bytes are read/parsed; numeric metrics may be present in those
    objects but are never exported, compared, ranked or used to choose a fix.
    """
    base = safe_path(root, RESULT_REL)
    records = []
    errors = {}
    extras = []
    require(base.is_dir(), f"Expected existing historical result directory: {base}")
    def walk_error(exc):
        raise ProbeError(f"Cannot inventory historical files: {exc}")
    for directory, dirs, files in os.walk(base, followlinks=False, onerror=walk_error):
        for name in dirs:
            safe_path(root, (Path(directory) / name).relative_to(root).as_posix())
        for name in sorted(files):
            path = Path(directory) / name
            rel = path.relative_to(root).as_posix()
            data, item = record_file(root, rel)
            sub = path.relative_to(base).parts
            if (len(sub) != 3 or sub[0] != "shards" or
                    sub[1] not in ("MOVE", "iCluster2b") or path.suffix != ".json"):
                extras.append(item)
                continue
            obj = json_object(data, rel)
            method, dataset = sub[1], path.stem
            require(dataset in valid_ids, f"Unknown historical dataset: {dataset}")
            require(obj.get("method") == method and
                    obj.get("comparator_dataset_id") == dataset,
                    f"Historical identity mismatch: {rel}")
            require(obj.get("runner_sha256") == RUNNER_SHA and
                    obj.get("cmp2_contract_sha256") == CONTRACT_SHA,
                    f"Historical source/contract mismatch: {rel}")
            require(obj.get("status") in ("PASS", "FAILED_AFTER_IDENTICAL_RETRY"),
                    f"Unexpected historical status: {rel}")
            item.update({"method": method, "dataset_id": dataset,
                         "status": obj["status"], "attempt_count": obj.get("attempt_count"),
                         "runner_sha256": obj["runner_sha256"]})
            if obj["status"] == "FAILED_AFTER_IDENTICAL_RETRY":
                first, second = obj.get("first_attempt_error"), obj.get("second_attempt_error")
                item["first_error_class"] = classify_error(first)
                item["second_error_class"] = classify_error(second)
                errors[f"errors/{method}/{dataset}.json"] = (
                    json.dumps({"source_record": item,
                                "first_attempt_error": first, "second_attempt_error": second},
                               indent=2, sort_keys=True) + "\n").encode()
            records.append(item)
    records.sort(key=lambda x: x["path"])
    extras.sort(key=lambda x: x["path"])
    require(records, "No historical shards found; do not run a benchmark to create them")
    counts = Counter((r["method"], r["status"]) for r in records)
    summary = [{"method": m, "status": s, "count": n}
               for (m, s), n in sorted(counts.items())]
    return {"records": records, "other_files": extras, "status_counts": summary}, errors


class RebalancedTailBatchSampler:
    """Keep the existing random index sequence; fix only a terminal singleton.

    Used ONLY as a candidate in this probe. It is not an authorized fit policy.
    The source sampler and its generator are reused, with no additional draw.
    """
    def __init__(self, original: Any, n: int, batch_size: int):
        self.original = original
        self.n = int(n)
        self.batch_size = int(batch_size)
        require(self.n >= 2, "BatchNorm training requires at least two fit rows")
        require(self.batch_size == min(16, self.n), "Unexpected base batch size")

    def __len__(self) -> int:
        return len(self.original)

    def __iter__(self):
        batches = [list(batch) for batch in self.original]
        require(batches, "Empty sampler")
        before = [i for batch in batches for i in batch]
        require(sorted(before) == list(range(self.n)), "Sampler must visit each row once")
        require(len(batches) == len(self), "Sampler batch count mismatch")
        if len(batches[-1]) == 1:
            require(len(batches) >= 2 and len(batches[-2]) >= 3,
                    "Cannot rebalance tail under the fixed batch cap")
            moved = batches[-2].pop()
            batches[-1].insert(0, moved)
        after = [i for batch in batches for i in batch]
        require(after == before, "Rebalancing changed flattened sample order")
        require(all(2 <= len(batch) <= self.batch_size for batch in batches),
                "Candidate produced an invalid training batch")
        yield from batches


def candidate_loader_factory(original_loader: Callable, torch: Any):
    def candidate(make_dataloader, views, batch_size, shuffle, seed):
        loader = original_loader(make_dataloader, views, batch_size, shuffle, seed)
        if not shuffle:
            return loader
        n = len(loader.dataset)
        require(n >= 2, "Cannot train BatchNorm on fewer than two fit rows")
        require(batch_size == min(16, n), "Base batch size differs from frozen rule")
        if n % batch_size != 1:
            return loader  # Unaffected training sizes use the original loader.
        require(loader.num_workers == 0 and not loader.drop_last,
                "Unexpected loader settings")
        return torch.utils.data.DataLoader(
            dataset=loader.dataset,
            batch_sampler=RebalancedTailBatchSampler(loader.batch_sampler, n, batch_size),
            num_workers=0,
            collate_fn=loader.collate_fn,
            pin_memory=loader.pin_memory,
            generator=loader.generator,
        )
    return candidate


def loader_order(loader):
    ids, sizes = [], []
    for cat, con in loader:
        ids.extend(int(x) for x in con[:, 0].tolist())
        sizes.append(int(con.shape[0]))
    return ids, sizes


def exercise_orders(old_loader, candidate, api, np, torch):
    sizes = list(range(2, 101)) + [113, 129, 145, 161]
    cases = []
    for n in sizes:
        views = [np.arange(n, dtype=np.float32).reshape(n, 1),
                 np.ones((n, 2), dtype=np.float32), np.ones((n, 2), dtype=np.float32)]
        for seed in (1701, 2701):
            old = old_loader(api["make_dataloader"], views, min(16, n), True, seed)
            new = candidate(api["make_dataloader"], views, min(16, n), True, seed)
            for epoch in range(2):
                a, old_sizes = loader_order(old)
                b, new_sizes = loader_order(new)
                require(a == b and sorted(b) == list(range(n)), "Row loss/duplication/order drift")
                require(len(old_sizes) == len(new_sizes), "Optimizer step count changed")
                require(all(2 <= size <= min(16, n) for size in new_sizes), "Singleton remains")
                require(torch.equal(old.generator.get_state(), new.generator.get_state()),
                        "Loader generator state changed")
                if n % min(16, n) != 1:
                    require(old_sizes == new_sizes, "Unaffected batch geometry changed")
            proj = candidate(api["make_dataloader"], views, min(16, n), False, seed)
            ids, proj_sizes = loader_order(proj)
            require(ids == list(range(n)), "Projection order changed")
            cases.append({"n": n, "seed": seed, "epochs": 2,
                          "original_sizes": old_sizes, "candidate_sizes": new_sizes,
                          "projection_sizes": proj_sizes})
    return {"status": "PASS", "case_count": len(cases), "cases": cases}


def import_pinned_runner(path: Path):
    name = "_paper5_v6_synthetic_batch_probe"
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Cannot build import spec")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def deny_scientific(*args, **kwargs):
    raise ProbeError("Scientific/R entry point is forbidden in this synthetic probe")


def run_synthetic_probe(runner, api):
    np, torch = runner.np, runner.torch
    old_loader, old_epochs, old_warmup = runner.move_loader, runner.MOVE_EPOCHS, runner.MOVE_KLD_WARMUP_STEPS
    candidate = candidate_loader_factory(old_loader, torch)
    order_report = exercise_orders(old_loader, candidate, api, np, torch)
    print("Candidate order/coverage/step-count checks: PASS", flush=True)
    rng = np.random.default_rng(740091)  # New artificial fixture, not a frozen dataset seed.
    raw = {key: rng.normal(size=(62, p)) for key, p in
           zip(runner.VIEW_KEYS, [500, 4000, 500])}
    ids = np.arange(62, dtype=np.int32)
    folds = runner.deterministic_sample_folds(ids, 5)
    train = np.setdiff1d(ids, folds[0])
    require(len(train) == 49, "Expected n62 first-fold train size 49")
    views, _, _ = runner.adapt_views(raw, train, train, np.float32)
    old_error = None
    try:
        fit = runner.fit_move(api, views, [64, 32], 740093, 2, [1, 1, 2, 2])
        del fit
    except ValueError as exc:
        require("Expected more than 1 value per channel when training" in str(exc)
                and "[1, 64]" in str(exc), "Unexpected baseline ValueError")
        old_error = traceback.format_exc()
    require(old_error is not None, "Original v6 singleton failure did NOT reproduce; stop review")
    gc.collect()
    torch.cuda.empty_cache()
    print("Original v6 / artificial first fold / 49 rows: singleton BatchNorm failure reproduced", flush=True)

    fit_calls = []
    original_fit = runner.fit_move
    def traced_fit(api_arg, fitted_views, hidden, seed, epochs, kld_warmup_steps):
        require(int(epochs) == 2, "Only two-epoch artificial fits are allowed in this probe")
        value = original_fit(api_arg, fitted_views, hidden, seed, epochs, kld_warmup_steps)
        fit_calls.append({"n_fit": len(fitted_views[0]), "hidden": list(hidden),
                          "epochs": epochs, "seed": seed,
                          "latent_shape": list(value["latent"].shape),
                          "finite": bool(np.isfinite(value["latent"]).all())})
        return value
    try:
        runner.move_loader = candidate
        runner.fit_move = traced_fit
        runner.MOVE_EPOCHS = 2
        runner.MOVE_KLD_WARMUP_STEPS = [1, 1, 2, 2]
        selected = runner.select_move_architecture(
            api, raw, ids, "SYNTHETIC_ONLY_BN_TAIL_N62_740091")
        require(len(selected["stage1"]) == 3 and all(
            len(row["fold_sse"]) == 5 for row in selected["stage1"]), "Incomplete 3x5 CV test")
        require(sum(c["n_fit"] in (49, 50) for c in fit_calls) == 15,
                "Expected fifteen actual CV fits")
        latent = np.asarray(selected["canonical_latent"])
        require(latent.shape == (62, 5) and np.isfinite(latent).all(), "Invalid canonical latent")
        require(all(c["finite"] for c in fit_calls), "Non-finite artificial fit")
        require(all(len(row["seed_runs"]) == 3 for row in selected["stage2"]),
                "Incomplete stability-seed test")
        result = {"status": "PASS_CANDIDATE_ACTUAL_MOVE_SYNTHETIC_SELECTION",
                  "fixture": {"n": 62, "features": [500, 4000, 500], "fresh_seed": 740091,
                              "fold_test_sizes": [len(f) for f in folds],
                              "fold_train_sizes": [62-len(f) for f in folds]},
                  "baseline_error": old_error, "batch_order_tests": order_report,
                  "completed_actual_synthetic_fit_count": len(fit_calls), "fit_calls": fit_calls,
                  "canonical_latent_shape": list(latent.shape),
                  "selected_on_artificial_data_only": selected["selected_candidate_id"],
                  "scientific_metric_values_evaluated": False,
                  "synthetic_epochs": 2, "frozen_scientific_epochs_unchanged": old_epochs}
    finally:
        runner.move_loader, runner.fit_move = old_loader, original_fit
        runner.MOVE_EPOCHS, runner.MOVE_KLD_WARMUP_STEPS = old_epochs, old_warmup
        gc.collect()
        torch.cuda.empty_cache()
    print("Candidate / all 3 architectures x 5 folds + full-data stability / actual CUDA fits: PASS", flush=True)
    return result


def publish_zip(root: Path, entries: dict[str, bytes], recheck: Callable[[], None]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = safe_path(root, f"paper5/cmp3_bn_review_{stamp}.zip")
    recheck()
    # Exclusive creation: a pre-existing file is never replaced.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(entries.items()):
            archive.writestr(name, data)
    with zipfile.ZipFile(output, "r") as archive:
        require(archive.testzip() is None, "Evidence ZIP CRC failure")
        require(set(archive.namelist()) == set(entries), "Evidence ZIP entry mismatch")
        require(all(archive.read(name) == data for name, data in entries.items()),
                "Evidence ZIP content mismatch")
    recheck()
    return output


def main() -> int:
    sys.dont_write_bytecode = True
    print("Paper 5 - v6 singleton-batch probe; NOT a benchmark or authorizer", flush=True)
    print("Version:", VERSION, flush=True)
    report: dict[str, Any] = {"version": VERSION, "policy_proposal": POLICY}
    try:
        require(len(sys.argv) == 1, "No command-line arguments accepted")
        script = Path(__file__).resolve()
        require(script.name == FILENAME and script.parent.name == "scripts"
                and script.parent.parent.name == "paper5", "Place cmp3bncheck.py under paper5/scripts")
        root = script.parents[2]
        require(Path(sys.prefix).resolve() == (root / "paper5/comparator_envs/move_1_5_0").resolve(),
                "Use the existing move_1_5_0 environment")
        snapshots = {}
        entries = {}
        def capture(rel, expected=None):
            data, record = record_file(root, rel, expected)
            snapshots[rel] = record
            entries["inputs/" + rel] = data
            return data
        source = capture(RUNNER_REL, RUNNER_SHA)
        capture(script.relative_to(root).as_posix())
        auth = json_object(capture(AUTH_REL, AUTH_SHA), AUTH_REL)
        preflight = json_object(capture(PREFLIGHT_REL, PREFLIGHT_SHA), PREFLIGHT_REL)
        require(auth.get("runner_source", {}).get("sha256") == RUNNER_SHA and
                auth.get("synthetic_preflight", {}).get("sha256") == PREFLIGHT_SHA,
                "Old authorization bindings differ")
        require(preflight.get("runner_sha256") == RUNNER_SHA, "Old preflight binding differs")
        compile(source, RUNNER_REL, "exec", dont_inherit=True)
        ast.parse(source.decode("utf-8-sig"), feature_version=(3, 10))
        runner = import_pinned_runner(safe_path(root, RUNNER_REL))
        for name in ("main", "authorized_real_main", "preauthorization_main", "load_scientific_npz",
                     "checked_simulation_path", "load_scientific_truth", "run_move_dataset",
                     "run_icluster_dataset", "run_icluster_fit", "find_exact_rscript",
                     "verify_r_library", "evaluate_truth_metrics", "write_scientific_shard",
                     "finalize_results", "write_json_atomic"):
            setattr(runner, name, deny_scientific)
        paths = runner.build_paths(safe_path(root, RUNNER_REL))
        runner.verify_schema_amendment_inputs(paths)
        contract, registry = runner.verify_upstream(paths)
        require(contract["MOVE"]["fixed_training"]["batch_size_rule"] == "min(16, n_fit_samples)",
                "Frozen batch-size rule differs")
        for key in ("cmp2_contract", "cmp2_lock", "cmp2_registry"):
            capture(paths[key].relative_to(root).as_posix())
        for rel, digest in runner.AMENDMENT_REVIEWED_INPUTS.items():
            _, record = record_file(root, rel, digest)
            snapshots[rel] = record
        for key, expected_key in (("cmp1_contract", "cmp1_contract_sha256"),
              ("cmp1_lock", "cmp1_lock_sha256"), ("cmp1_impl", "cmp1_impl_sha256"),
              ("cmp1_adapter", "cmp1_adapter_sha256"), ("cmp1_seed", "cmp1_seed_sha256"),
              ("cmp1_metric", "cmp1_metric_sha256"), ("manifest05b", "manifest05b_sha256")):
            rel = paths[key].relative_to(root).as_posix()
            _, record = record_file(root, rel, runner.EXPECTED[expected_key])
            snapshots[rel] = record
        valid_ids = set(registry["comparator_dataset_id"].astype(str))
        historical, errors = inventory_results(root, valid_ids)
        entries.update(errors)
        report["historical_shards"] = historical
        for row in historical["status_counts"]:
            print(f"Historical {row['method']} / {row['status']}: {row['count']}", flush=True)
        api = runner.import_move_api()  # Pinned installed sources; CUDA required, no R.
        for name, path in api["source_paths"].items():
            rel = path.relative_to(root).as_posix()
            capture(rel, runner.EXPECTED["move_source_hashes"][name])
        report["environment"] = {"python": sys.version, "python_executable": sys.executable,
             "torch": runner.torch.__version__, "cuda": str(runner.torch.version.cuda),
             "move": api["move_version"]}
        print("Pinned v6/history/upstream and installed MOVE/CUDA source checks: PASS", flush=True)
        unexpected = [r for r in historical["records"] if r["method"] == "MOVE" and
                      (r["status"] != "FAILED_AFTER_IDENTICAL_RETRY" or
                       r.get("first_error_class") != "SINGLETON_BATCHNORM_DURING_MOVE_SELECTION" or
                       r.get("second_error_class") != "SINGLETON_BATCHNORM_DURING_MOVE_SELECTION")]
        report["unexpected_MOVE_records"] = unexpected
        if unexpected or historical["other_files"]:
            report["status"] = "HISTORY_REQUIRES_REVIEW_NO_SYNTHETIC_FITS_STARTED"
            print("Unexpected historical record/files: collecting evidence only; no fitting", flush=True)
        else:
            try:
                report["synthetic_validation"] = run_synthetic_probe(runner, api)
                report["status"] = "SYNTHETIC_CANDIDATE_PASS_NOT_AUTHORIZED"
            except Exception:
                report["status"] = "SYNTHETIC_CANDIDATE_FAILED_NOT_AUTHORIZED"
                report["probe_error"] = traceback.format_exc()
                print(report["probe_error"], flush=True)
        report["source_snapshots"] = snapshots
        report["access_disclosure"] = {
            "frozen_npz_or_truth_opened": False, "R_called": False,
            "scientific_fit_performed": False, "authorization_created": False,
            "old_shard_JSON_bytes_read_and_parsed": True,
            "old_numeric_metrics_compared_or_exported": False,
            "old_shard_hash_status_and_errors_inspected": True,
            "installed_MOVE_source_bytes_copied_for_review": True,
            "runner_imported_under_non_main_name": True,
            "candidate_loader_substituted_in_memory_for_synthetic_probe_only": True,
            "existing_sources_results_locks_modified": False}
        def recheck():
            for rel, before in snapshots.items():
                _, after = record_file(root, rel, before["sha256"])
                require(after == before, f"Input changed: {rel}")
            now, _ = inventory_results(root, valid_ids)
            require(now == historical, "Historical result set changed; keep runner stopped")
        recheck()
        entries["report.json"] = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
        entries["README.txt"] = ("Synthetic candidate evidence only. No new scientific authorization.\n"
            "Batch composition changes require a separate disclosed amendment.\n"
            "Historical results remain in place; successful metric values are not included.\n").encode()
        output = publish_zip(root, entries, recheck)
        print("Status:", report["status"], flush=True)
        print("Sources / metadata / historical shards recheck: PASS", flush=True)
        print("Frozen NPZ files were not opened or modified by this probe", flush=True)
        print("Evidence ZIP:", output, flush=True)
        print("Evidence ZIP SHA-256:", sha(output.read_bytes()), flush=True)
        print("STOP HERE. Upload this ZIP; do not run or edit v6.", flush=True)
        return 0 if report["status"] == "SYNTHETIC_CANDIDATE_PASS_NOT_AUTHORIZED" else 2
    except Exception:
        print("PROBE STOPPED:\n" + traceback.format_exc(), file=sys.stderr)
        print("No benchmark authorization was created. Preserve existing files.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
