#!/usr/bin/env python
from __future__ import annotations

import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
import tempfile
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_VERSION = "05y2-run-correlated-private-failure-diagnostics-fast-cpu-v1-no-cli"
STATUS = "POSTUNBLINDING_EXPLORATORY_FAILURE_DIAGNOSTICS_COMPLETE"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

DIR_05Y1 = PAPER5_ROOT / "method_contract" / "05y1"
CONTRACT_05Y1 = DIR_05Y1 / "correlated_private_failure_diagnostic_contract.json"
LOCK_05Y1 = DIR_05Y1 / "05y1_contract_lock.json"

GENERATOR = PAPER5_ROOT / "scripts" / "05b_generate_small_n_multiomics_simulations_v2.py"
SCENARIO_REGISTRY = PAPER5_ROOT / "method_contract" / "05a1" / "final_gate_calibration_scenarios.csv"
PROGRESS_05F2 = PAPER5_ROOT / "method_contract" / "05f2" / "full_run_progress_manifest.csv"

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05y2"
OUT_DATASET = OUTPUT_DIR / "g19_realized_rho_dataset_diagnostic.csv"
OUT_QUINTILES = OUTPUT_DIR / "g19_realized_rho_quintile_failure_rates.csv"
OUT_FACTORS = OUTPUT_DIR / "factor_space_classifier_summary.csv"
OUT_FRONTIER = OUTPUT_DIR / "retention_threshold_frontier.csv"
OUT_RESULT = OUTPUT_DIR / "correlated_private_failure_diagnostic_result.json"
OUT_README = OUTPUT_DIR / "05y2_README.txt"
OUT_LOCK = OUTPUT_DIR / "05y2_results_lock.json"

EXPECTED_CONTRACT_STATUS = (
    "FROZEN_POSTUNBLINDING_EXPLORATORY_DIAGNOSTIC_BEFORE_RESULT_COMPUTATION"
)

G19 = "G19_CORRELATED_PRIVATE_NULL_N62"
STANDARD = "G07_IND_N62_S1P00"

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260915
N_FOLDS = 5
FOLD_SEED = 20260915

FEATURES = (
    "median_aligned_correlation",
    "q10_aligned_correlation",
    "inclusion_frequency",
)

REQUIRED_FACTOR_FIELDS = {
    *FEATURES,
    "reference_factor_variance",
    "n_supported_modalities",
    "passes_stability_gate",
    "biological_truth_match",
    "technical_truth_match",
    "false_biological_acceptance",
}

SEED_CANDIDATES = (
    "simulation_seed",
    "dataset_seed",
    "replicate_seed",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (1, 1.0):
        return True
    if value in (0, 0.0):
        return False
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes"}:
        return True
    if text in {"false", "f", "0", "no"}:
        return False
    raise RuntimeError(f"Cannot parse boolean value: {value!r}")


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".txt", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_csv_atomic(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".csv", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def verify_contract() -> dict[str, Any]:
    for path in [CONTRACT_05Y1, LOCK_05Y1, GENERATOR, SCENARIO_REGISTRY, PROGRESS_05F2]:
        require_file(path)

    lock = load_json(LOCK_05Y1)
    if str(lock.get("status", "")).strip() != EXPECTED_CONTRACT_STATUS:
        raise RuntimeError("05y1 lock status mismatch.")

    record = lock.get("contract", {})
    if sha256_file(CONTRACT_05Y1).lower() != str(record.get("sha256", "")).lower():
        raise RuntimeError("05y1 contract hash mismatch.")

    contract = load_json(CONTRACT_05Y1)
    if str(contract.get("status", "")).strip() != EXPECTED_CONTRACT_STATUS:
        raise RuntimeError("05y1 contract status mismatch.")

    expected_gen = contract["implementation_integrity"]["generator_source"]["sha256"]
    if sha256_file(GENERATOR).lower() != str(expected_gen).lower():
        raise RuntimeError("Frozen generator source hash mismatch.")

    return contract


def resolve_project_path(raw: str) -> Path:
    p = Path(str(raw))
    return p if p.is_absolute() else PROJECT_ROOT / p


def scenario_rows() -> dict[str, dict[str, str]]:
    with SCENARIO_REGISTRY.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = {
            str(row["scenario_id"]).strip(): row
            for row in csv.DictReader(fh)
        }
    return rows


def select_manifest() -> dict[str, list[dict[str, str]]]:
    selected = {G19: [], STANDARD: []}
    with PROGRESS_05F2.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sid = str(row["scenario_id"]).strip()
            if sid in selected:
                selected[sid].append(row)
    if len(selected[G19]) != 200 or len(selected[STANDARD]) != 100:
        raise RuntimeError(
            f"Frozen scenario counts changed: G19={len(selected[G19])}, "
            f"STANDARD={len(selected[STANDARD])}."
        )
    return selected


def read_one(row: dict[str, str]) -> dict[str, Any]:
    path = resolve_project_path(row["shard_path"])
    require_file(path)
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    expected = str(row["shard_sha256"]).strip()
    if actual.lower() != expected.lower():
        raise RuntimeError(f"Shard hash mismatch: {path}")

    shard = json.loads(data)
    payload = shard.get("scientific_payload")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Missing scientific_payload: {path}")

    dataset = payload.get("dataset_row")
    factors = payload.get("factor_rows")
    if not isinstance(dataset, dict) or not isinstance(factors, list):
        raise RuntimeError(f"Unexpected scientific_payload schema: {path}")

    if str(dataset.get("dataset_key", "")).strip() != str(row["dataset_key"]).strip():
        raise RuntimeError(f"dataset_key mismatch: {path}")
    if str(dataset.get("scenario_id", "")).strip() != str(row["scenario_id"]).strip():
        raise RuntimeError(f"scenario_id mismatch: {path}")

    for factor in factors:
        if not isinstance(factor, dict):
            raise RuntimeError(f"Non-object factor row: {path}")
        missing = REQUIRED_FACTOR_FIELDS - set(factor)
        if missing:
            raise RuntimeError(
                f"Factor row missing fields in {path}: {sorted(missing)}"
            )

    manifest_seed = None
    for key in SEED_CANDIDATES:
        raw = str(row.get(key, "")).strip()
        if raw:
            manifest_seed = int(float(raw))
            break

    dataset_seed = None
    dataset_seed_key = None
    for key in SEED_CANDIDATES:
        if key in dataset and str(dataset[key]).strip() != "":
            dataset_seed = int(float(dataset[key]))
            dataset_seed_key = key
            break

    if manifest_seed is None and dataset_seed is None:
        raise RuntimeError(
            f"No frozen simulation seed found in manifest or dataset row for {path}."
        )
    if manifest_seed is not None and dataset_seed is not None and manifest_seed != dataset_seed:
        raise RuntimeError(f"Manifest/dataset simulation seed mismatch: {path}")

    seed = manifest_seed if manifest_seed is not None else dataset_seed
    seed_source = "manifest" if manifest_seed is not None else f"dataset_row:{dataset_seed_key}"

    return {
        "dataset_key": str(row["dataset_key"]).strip(),
        "scenario_id": str(row["scenario_id"]).strip(),
        "simulation_seed": int(seed),
        "seed_source": seed_source,
        "dataset": dataset,
        "factors": factors,
    }


def import_generator_module():
    spec = importlib.util.spec_from_file_location("paper5_05b_frozen_generator", GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import frozen 05b generator.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def replay_realized_rho(
    module: Any,
    scenario_row: dict[str, str],
    simulation_seed: int,
) -> dict[str, Any]:
    n = int(float(scenario_row["n_samples"]))
    k_private = int(float(scenario_row["n_private_factors_per_view"]))
    k_shared = int(float(scenario_row["n_shared_factors"]))
    shared_corr = float(scenario_row["shared_factor_correlation"])
    rho = float(scenario_row["private_cross_view_correlation"])
    mode = str(scenario_row["correlated_private_mode"]).strip()

    root_rng = np.random.default_rng(int(simulation_seed))

    # Exact frozen pre-private RNG call order.
    module.generate_correlated_scores(
        root_rng,
        n,
        k_shared,
        shared_corr,
    )

    scenario_min = SimpleNamespace(
        n_samples=n,
        n_private_factors_per_view=k_private,
        private_cross_view_correlation=rho,
        correlated_private_mode=mode,
    )

    private = module.generate_private_scores(root_rng, scenario_min)

    views = list(module.VIEWS)
    if len(views) != 3:
        raise RuntimeError(f"Expected exactly 3 frozen views; found {views}.")
    if k_private != 2:
        raise RuntimeError(f"Expected G19 private factor count 2; found {k_private}.")

    pairwise = []
    for factor_index in range(k_private):
        for a, b in itertools.combinations(views, 2):
            x = np.asarray(private[a][:, factor_index], dtype=float)
            y = np.asarray(private[b][:, factor_index], dtype=float)
            corr = float(np.corrcoef(x, y)[0, 1])
            if not math.isfinite(corr):
                raise RuntimeError("Non-finite replayed private correlation.")
            pairwise.append(corr)

    arr = np.asarray(pairwise, dtype=float)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "minimum": float(arr.min()),
        "maximum": float(arr.max()),
        "sd": float(arr.std(ddof=0)),
        "pairwise": pairwise,
    }


def wilson(k: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def stratified_bootstrap_auc(
    y: np.ndarray,
    score: np.ndarray,
    reps: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    values = np.empty(reps, dtype=float)
    for i in range(reps):
        idx = np.concatenate(
            [
                rng.choice(pos, size=pos.size, replace=True),
                rng.choice(neg, size=neg.size, replace=True),
            ]
        )
        values[i] = roc_auc_score(y[idx], score[idx])
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)


def predictor_only_quintile_labels(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    cuts = np.quantile(values, [0.2, 0.4, 0.6, 0.8], method="linear")
    labels = np.searchsorted(cuts, values, side="right") + 1
    return labels.astype(int), [float(x) for x in cuts]


def stable_hash_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def balanced_group_folds(
    groups: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    seed: int,
) -> dict[str, int]:
    unique = sorted(set(str(g) for g in groups))
    stats = []
    for group in unique:
        mask = groups == group
        pos = int(np.sum(y[mask] == 1))
        neg = int(np.sum(y[mask] == 0))
        key = stable_hash_int(f"{seed}|{group}")
        stats.append((group, pos, neg, pos + neg, key))

    # Difficult / large groups first, deterministic hash breaks ties.
    stats.sort(key=lambda x: (-max(x[1], x[2]), -x[3], x[4]))

    fold_pos = [0] * n_folds
    fold_neg = [0] * n_folds
    fold_n = [0] * n_folds
    assignment: dict[str, int] = {}

    for group, pos, neg, total, key in stats:
        choices = []
        for fold in range(n_folds):
            new_pos = fold_pos[fold] + pos
            new_neg = fold_neg[fold] + neg
            new_total = fold_n[fold] + total
            # lexicographic balancing criterion
            choices.append(
                (
                    max(new_pos, new_neg),
                    abs(new_pos - new_neg),
                    new_total,
                    fold_n[fold],
                    fold,
                )
            )
        best = min(range(n_folds), key=lambda f: choices[f])
        assignment[group] = best
        fold_pos[best] += pos
        fold_neg[best] += neg
        fold_n[best] += total

    return assignment


def grouped_oof_auc(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    assignment = balanced_group_folds(groups, y, n_folds, seed)
    pred = np.full(y.shape[0], np.nan, dtype=float)

    for fold in range(n_folds):
        test = np.asarray([assignment[str(g)] == fold for g in groups], dtype=bool)
        train = ~test
        if len(np.unique(y[test])) < 2:
            raise RuntimeError(f"Grouped fold {fold} lacks one class in test.")
        if len(np.unique(y[train])) < 2:
            raise RuntimeError(f"Grouped fold {fold} lacks one class in train.")

        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logit",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        solver="lbfgs",
                        class_weight="balanced",
                        max_iter=10000,
                    ),
                ),
            ]
        )
        model.fit(X[train], y[train])
        pred[test] = model.predict_proba(X[test])[:, 1]

    if not np.all(np.isfinite(pred)):
        raise RuntimeError("Non-finite grouped OOF predictions.")
    return float(roc_auc_score(y, pred)), pred


def cluster_bootstrap_oof_auc(
    y: np.ndarray,
    score: np.ndarray,
    groups: np.ndarray,
    reps: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique_groups = np.asarray(sorted(set(str(g) for g in groups)), dtype=object)
    group_to_idx = {
        g: np.flatnonzero(groups == g)
        for g in unique_groups
    }

    values = []
    attempts = 0
    max_attempts = reps * 3
    while len(values) < reps and attempts < max_attempts:
        attempts += 1
        sampled = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sampled])
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(float(roc_auc_score(y[idx], score[idx])))

    if len(values) < reps:
        raise RuntimeError(
            f"Only {len(values)} valid cluster bootstrap replicates out of requested {reps}."
        )
    lo, hi = np.quantile(np.asarray(values), [0.025, 0.975])
    return float(lo), float(hi)


def classifier_dataset(
    records: list[dict[str, Any]],
    contrast: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_rows = []
    y_rows = []
    groups = []

    for rec in records:
        sid = rec["scenario_id"]
        for factor in rec["factors"]:
            retained = parse_bool(factor["passes_stability_gate"])
            bio = parse_bool(factor["biological_truth_match"])
            tech = parse_bool(factor["technical_truth_match"])
            false = parse_bool(factor["false_biological_acceptance"])

            include = False
            label = None

            if contrast == "target":
                if sid == G19 and retained and false:
                    include, label = True, 1
                elif sid == STANDARD and retained and bio and not tech:
                    include, label = True, 0

            elif contrast == "negative_anchor":
                if sid == G19 and not retained:
                    include, label = True, 1
                elif sid == STANDARD and not retained:
                    include, label = True, 0

            elif contrast == "positive_anchor":
                if sid == G19 and retained and false:
                    include, label = True, 1
                elif sid == G19 and not retained:
                    include, label = True, 0

            else:
                raise ValueError(contrast)

            if include:
                X_rows.append([float(factor[key]) for key in FEATURES])
                y_rows.append(int(label))
                groups.append(str(rec["dataset_key"]))

    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_rows, dtype=int)
    g = np.asarray(groups, dtype=object)
    if X.shape[0] == 0 or len(np.unique(y)) != 2:
        raise RuntimeError(f"Classifier contrast {contrast} is not estimable.")
    return X, y, g


def factor_classifier_summary(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for offset, contrast in enumerate(("target", "negative_anchor", "positive_anchor")):
        X, y, groups = classifier_dataset(records, contrast)
        auc, pred = grouped_oof_auc(X, y, groups, N_FOLDS, FOLD_SEED)
        lo, hi = cluster_bootstrap_oof_auc(
            y,
            pred,
            groups,
            BOOTSTRAP_REPLICATES,
            BOOTSTRAP_SEED + 100 + offset,
        )
        rows.append(
            {
                "contrast": contrast,
                "n_factor_rows": int(y.size),
                "n_positive_factor_rows": int(y.sum()),
                "n_negative_factor_rows": int((1 - y).sum()),
                "n_dataset_groups": int(len(set(groups.tolist()))),
                "oof_auc": auc,
                "ci95_low": lo,
                "ci95_high": hi,
            }
        )
    return rows


def grid_values() -> np.ndarray:
    return np.round(np.arange(0.0, 1.0000001, 0.05), 10)


def frontier(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    g19 = [r for r in records if r["scenario_id"] == G19]
    standard = [r for r in records if r["scenario_id"] == STANDARD]

    values = grid_values()
    rows: list[dict[str, Any]] = []

    for m in values:
        for q in values:
            for inc in values:
                g_events = 0
                for rec in g19:
                    event = False
                    for f in rec["factors"]:
                        keep = (
                            float(f["median_aligned_correlation"]) >= m
                            and float(f["q10_aligned_correlation"]) >= q
                            and float(f["inclusion_frequency"]) >= inc
                            and float(f["reference_factor_variance"]) >= 0.0001
                            and int(f["n_supported_modalities"]) >= 2
                        )
                        false = (
                            not parse_bool(f["technical_truth_match"])
                            and not parse_bool(f["biological_truth_match"])
                        )
                        if keep and false:
                            event = True
                            break
                    if event:
                        g_events += 1

                standard_recovered = 0
                for rec in standard:
                    for f in rec["factors"]:
                        keep = (
                            float(f["median_aligned_correlation"]) >= m
                            and float(f["q10_aligned_correlation"]) >= q
                            and float(f["inclusion_frequency"]) >= inc
                            and float(f["reference_factor_variance"]) >= 0.0001
                            and int(f["n_supported_modalities"]) >= 2
                        )
                        if keep and parse_bool(f["biological_truth_match"]):
                            standard_recovered += 1

                if standard_recovered > 400:
                    raise RuntimeError(
                        f"STANDARD recovered count exceeds frozen denominator: {standard_recovered}."
                    )

                rows.append(
                    {
                        "median_threshold": float(m),
                        "q10_threshold": float(q),
                        "inclusion_threshold": float(inc),
                        "g19_false_event_numerator": int(g_events),
                        "g19_false_event_denominator": 200,
                        "g19_false_event_rate": g_events / 200.0,
                        "standard_recovery_numerator": int(standard_recovered),
                        "standard_recovery_denominator": 400,
                        "standard_recovery_rate": standard_recovered / 400.0,
                        "is_pareto_efficient": False,
                        "dominates_frozen_point": False,
                    }
                )

    frozen = [
        r for r in rows
        if abs(r["median_threshold"] - 0.70) < 1e-12
        and abs(r["q10_threshold"] - 0.40) < 1e-12
        and abs(r["inclusion_threshold"] - 0.80) < 1e-12
    ]
    if len(frozen) != 1:
        raise RuntimeError("Frozen threshold point missing from frontier grid.")
    frozen = frozen[0]

    if (
        frozen["g19_false_event_numerator"] != 70
        or frozen["standard_recovery_numerator"] != 371
    ):
        raise RuntimeError(
            "Frozen frontier point does not reproduce 70/200 G19 and 371/400 STANDARD: "
            f"{frozen['g19_false_event_numerator']}/200, "
            f"{frozen['standard_recovery_numerator']}/400."
        )

    # Pareto efficiency in (min G19 rate, max STANDARD recovery).
    best_recovery_at_or_below = -1.0
    for row in sorted(
        rows,
        key=lambda r: (
            r["g19_false_event_rate"],
            -r["standard_recovery_rate"],
        ),
    ):
        if row["standard_recovery_rate"] > best_recovery_at_or_below + 1e-15:
            row["is_pareto_efficient"] = True
            best_recovery_at_or_below = row["standard_recovery_rate"]

    frozen_g = frozen["g19_false_event_rate"]
    frozen_s = frozen["standard_recovery_rate"]
    n_dominating = 0
    for row in rows:
        dominates = (
            row["g19_false_event_rate"] <= frozen_g + 1e-15
            and row["standard_recovery_rate"] >= frozen_s - 1e-15
            and (
                row["g19_false_event_rate"] < frozen_g - 1e-15
                or row["standard_recovery_rate"] > frozen_s + 1e-15
            )
        )
        row["dominates_frozen_point"] = bool(dominates)
        if dominates:
            n_dominating += 1

    frozen["is_pareto_efficient"] = not any(
        r["dominates_frozen_point"] for r in rows
    )

    meta = {
        "grid_points": len(rows),
        "frozen_point": dict(frozen),
        "n_grid_points_dominating_frozen": n_dominating,
        "frozen_point_pareto_efficient": bool(
            frozen["is_pareto_efficient"]
        ),
    }
    return rows, meta


def main() -> None:
    print("=" * 96)
    print("Paper 5 - FAST correlated-private failure diagnostics")
    print("=" * 96)
    print(f"Script version: {SCRIPT_VERSION}")
    print("Role: POST-UNBLINDING EXPLORATORY CHARACTERIZATION")
    print("SABLE-MO model fitting: NO")
    print("GPU computation: NO")
    print("Formal 05f3e modification: NO")
    print()

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(
            f"05y2 output directory already exists and is non-empty: {OUTPUT_DIR}. "
            "Do not overwrite exploratory results."
        )

    contract = verify_contract()
    scenarios = scenario_rows()
    manifest = select_manifest()

    rows = manifest[G19] + manifest[STANDARD]
    workers = min(32, max(4, 2 * (os.cpu_count() or 4)))
    print(f"Parallel shard read/hash workers: {workers}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(read_one, rows))

    print(f"Verified/read sealed datasets: {len(records)}/300")

    generator = import_generator_module()
    g19_row = scenarios[G19]

    dataset_rows = []
    y = []
    rho_mean = []

    g19_records = [r for r in records if r["scenario_id"] == G19]
    for rec in g19_records:
        realized = replay_realized_rho(
            generator,
            g19_row,
            rec["simulation_seed"],
        )
        event = any(
            parse_bool(f["false_biological_acceptance"])
            for f in rec["factors"]
        )
        dataset_rows.append(
            {
                "dataset_key": rec["dataset_key"],
                "simulation_seed": rec["simulation_seed"],
                "seed_source": rec["seed_source"],
                "false_acceptance_event": int(event),
                "realized_private_corr_mean": realized["mean"],
                "realized_private_corr_median": realized["median"],
                "realized_private_corr_min": realized["minimum"],
                "realized_private_corr_max": realized["maximum"],
                "realized_private_corr_sd": realized["sd"],
                "pairwise_correlations_json": json.dumps(realized["pairwise"]),
            }
        )
        y.append(int(event))
        rho_mean.append(float(realized["mean"]))

    y_arr = np.asarray(y, dtype=int)
    rho_arr = np.asarray(rho_mean, dtype=float)

    if int(y_arr.sum()) != 70 or y_arr.size != 200:
        raise RuntimeError(
            f"Frozen G19 event integrity failed: {int(y_arr.sum())}/{y_arr.size}."
        )

    auc = float(roc_auc_score(y_arr, rho_arr))
    auc_lo, auc_hi = stratified_bootstrap_auc(
        y_arr,
        rho_arr,
        BOOTSTRAP_REPLICATES,
        BOOTSTRAP_SEED,
    )

    quintile_labels, cuts = predictor_only_quintile_labels(rho_arr)
    quintile_rows = []
    for q in range(1, 6):
        mask = quintile_labels == q
        n = int(mask.sum())
        k = int(y_arr[mask].sum())
        lo, hi = wilson(k, n)
        quintile_rows.append(
            {
                "quintile": q,
                "n": n,
                "events": k,
                "event_rate": k / n,
                "wilson95_low": lo,
                "wilson95_high": hi,
                "rho_min": float(np.min(rho_arr[mask])),
                "rho_max": float(np.max(rho_arr[mask])),
            }
        )

    q5_minus_q1 = (
        quintile_rows[-1]["event_rate"] - quintile_rows[0]["event_rate"]
    )
    monotone_non_decreasing = all(
        quintile_rows[i]["event_rate"] <= quintile_rows[i + 1]["event_rate"] + 1e-15
        for i in range(4)
    )

    print(
        f"Primary realized-rho AUC: {auc:.4f} "
        f"[95% bootstrap CI {auc_lo:.4f}, {auc_hi:.4f}]"
    )

    classifier_rows = factor_classifier_summary(records)
    lookup = {r["contrast"]: r for r in classifier_rows}
    target_auc = lookup["target"]["oof_auc"]
    neg_auc = lookup["negative_anchor"]["oof_auc"]
    pos_auc = lookup["positive_anchor"]["oof_auc"]
    distance_neg = abs(target_auc - neg_auc)
    distance_pos = abs(target_auc - pos_auc)

    print(
        "Factor-space OOF AUCs: "
        f"target={target_auc:.4f}, negative_anchor={neg_auc:.4f}, "
        f"positive_anchor={pos_auc:.4f}"
    )

    frontier_rows, frontier_meta = frontier(records)
    print(
        "Threshold frontier: "
        f"{frontier_meta['grid_points']} grid points; "
        f"dominating frozen={frontier_meta['n_grid_points_dominating_frozen']}; "
        f"frozen Pareto={frontier_meta['frozen_point_pareto_efficient']}"
    )

    result = {
        "script_version": SCRIPT_VERSION,
        "status": STATUS,
        "analysis_role": "POSTUNBLINDING_EXPLORATORY_FAILURE_CHARACTERIZATION",
        "formal_final_gate_consequence": "NONE",
        "SABLE_MO_refit": False,
        "GPU_computation": False,
        "primary_realized_rho_diagnostic": {
            "n_datasets": 200,
            "false_acceptance_events": int(y_arr.sum()),
            "realized_rho_mean_across_datasets": float(rho_arr.mean()),
            "realized_rho_sd_across_datasets": float(rho_arr.std(ddof=1)),
            "auc_higher_rho_predicts_failure": auc,
            "auc_ci95_low": auc_lo,
            "auc_ci95_high": auc_hi,
            "predictor_only_quintile_cutpoints": cuts,
            "q5_minus_q1_failure_rate": float(q5_minus_q1),
            "quintile_failure_rates_monotone_non_decreasing": bool(
                monotone_non_decreasing
            ),
        },
        "factor_space_calibration": {
            "classifier_rows": classifier_rows,
            "absolute_distance_target_to_negative": float(distance_neg),
            "absolute_distance_target_to_positive": float(distance_pos),
            "target_closer_to_negative_than_positive": bool(
                distance_neg < distance_pos
            ),
        },
        "threshold_frontier": frontier_meta,
        "guardrails": [
            "No new threshold is selected or adopted from the exploratory frontier.",
            "No formal D01/D02/D03 decision is changed.",
            "No universal identifiability theorem is claimed.",
            "Any prospective discriminator or threshold modification requires fresh-seed validation.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(dataset_rows, OUT_DATASET)
    write_csv_atomic(quintile_rows, OUT_QUINTILES)
    write_csv_atomic(classifier_rows, OUT_FACTORS)
    write_csv_atomic(frontier_rows, OUT_FRONTIER)
    write_json_atomic(result, OUT_RESULT)

    readme_lines = [
        "Paper 5 05y2 — correlated-private failure diagnostics",
        "=" * 62,
        "",
        "POST-UNBLINDING EXPLORATORY CHARACTERIZATION",
        "Formal 05f3e consequence: NONE",
        "",
        f"G19 failures: {int(y_arr.sum())}/200",
        (
            f"Realized-rho AUC: {auc:.4f} "
            f"[95% bootstrap CI {auc_lo:.4f}, {auc_hi:.4f}]"
        ),
        f"Q5-Q1 failure-rate difference: {q5_minus_q1:.4f}",
        f"Quintile rates monotone non-decreasing: {monotone_non_decreasing}",
        "",
        (
            f"Factor-space target OOF AUC: {target_auc:.4f}; "
            f"negative anchor: {neg_auc:.4f}; positive anchor: {pos_auc:.4f}"
        ),
        (
            f"Target closer to negative anchor than positive: "
            f"{distance_neg < distance_pos}"
        ),
        "",
        (
            f"Threshold grid points: {frontier_meta['grid_points']}; "
            f"points dominating frozen: "
            f"{frontier_meta['n_grid_points_dominating_frozen']}; "
            f"frozen Pareto-efficient: "
            f"{frontier_meta['frozen_point_pareto_efficient']}"
        ),
        "",
        "No threshold selected from this frontier.",
        "Any method change requires fresh-seed prospective validation.",
        "",
    ]
    write_text_atomic("\n".join(readme_lines), OUT_README)

    generated = [
        OUT_DATASET,
        OUT_QUINTILES,
        OUT_FACTORS,
        OUT_FRONTIER,
        OUT_RESULT,
        OUT_README,
    ]
    lock = {
        "script_version": SCRIPT_VERSION,
        "script_path": relative_path(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "status": STATUS,
        "05y1_contract": file_record(CONTRACT_05Y1),
        "05y1_lock": file_record(LOCK_05Y1),
        "formal_final_gate_consequence": "NONE",
        "generated_files": {
            relative_path(path): file_record(path)
            for path in generated
        },
        "rewrite_policy": (
            "05y2 exploratory diagnostic results are immutable. New discriminators, "
            "thresholds, or additional outcome-driven analyses require a new versioned stage."
        ),
    }
    write_json_atomic(lock, OUT_LOCK)

    print()
    print("05y2 exploratory diagnostics: COMPLETE")
    print(f"Dataset diagnostic: {relative_path(OUT_DATASET)}")
    print(f"Quintile curve:      {relative_path(OUT_QUINTILES)}")
    print(f"Factor classifiers:  {relative_path(OUT_FACTORS)}")
    print(f"Threshold frontier:  {relative_path(OUT_FRONTIER)}")
    print(f"Result:              {relative_path(OUT_RESULT)}")
    print(f"Lock:                {relative_path(OUT_LOCK)}")
    print("Formal 05f3e gate modified: NO")


if __name__ == "__main__":
    main()
