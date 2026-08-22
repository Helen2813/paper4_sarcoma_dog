from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_VERSION = "05d3-freeze-selected-neural-profile-v1-no-cli"
CONTRACT_VERSION = "paper5-selected-neural-profile-freeze-v1"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

EXPECTED_PROFILES = 9
EXPECTED_DEVELOPMENT_DATASETS = 360
EXPECTED_SHARDS = 3_240
EXPECTED_FIT_AUDIT_ROWS = 74_520
EXPECTED_PREPROCESSING_AUDIT_ROWS = 223_560
EXPECTED_STABILITY_FACTOR_ROWS = 32_400
EXPECTED_05D2_SCRIPT_VERSION = (
    "05d2v2-screen-neural-profiles-gpu-syncfree-v2-no-cli"
)
EXPECTED_05D2_LOCK_STATUS = (
    "FROZEN_NEURAL_PROFILE_SCREEN_RESULTS_BEFORE_CONFIRMATORY_ABLATION"
)
EXPECTED_SELECTION_STATUS = "PROFILE_SELECTED_BY_FROZEN_ONE_SE_RULE"
FREEZE_STATUS = (
    "FROZEN_OUTCOME_BLIND_NEURAL_PROFILE_BEFORE_CONFIRMATORY_ABLATION"
)

EXPECTED_PROFILE_REUSE = {
    "V0_BALANCED_AE",
    "V1_SHARED_PRIVATE_AE",
    "V2_CONTRASTIVE_SPAE",
    "V3_DROPOUT_SPAE",
    "V5_SABLE_VAE",
}

# ---------------------------------------------------------------------
# Frozen upstream artifacts
# ---------------------------------------------------------------------

CONTRACT_05A = (
    PAPER5_ROOT
    / "method_contract"
    / "05a"
    / "ai_method_contract.json"
)

LOCK_05D1 = (
    PAPER5_ROOT
    / "method_contract"
    / "05d1"
    / "neural_simulation_protocol_lock.json"
)

PROFILE_GRID = (
    PAPER5_ROOT
    / "method_contract"
    / "05d1"
    / "neural_architecture_regularization_profile_grid.csv"
)

SELECTION_CONTRACT_05D1 = (
    PAPER5_ROOT
    / "method_contract"
    / "05d1"
    / "neural_profile_selection_contract.json"
)

LOSS_CONTRACT_05D1 = (
    PAPER5_ROOT
    / "method_contract"
    / "05d1"
    / "neural_loss_and_architecture_contract.json"
)

CONFIRMATORY_FIT_SEEDS_05D1 = (
    PAPER5_ROOT
    / "method_contract"
    / "05d1"
    / "neural_confirmatory_ablation_fit_seed_registry.csv"
)

CONFIRMATORY_STABILITY_05D1 = (
    PAPER5_ROOT
    / "method_contract"
    / "05d1"
    / "neural_confirmatory_stability_resample_registry.csv"
)

LOCK_05D2 = (
    PAPER5_ROOT
    / "method_contract"
    / "05d2_v2"
    / "neural_profile_screen_results_lock.json"
)

RESULT_MANIFEST_05D2 = (
    PAPER5_ROOT
    / "method_contract"
    / "05d2_v2"
    / "neural_profile_screen_results_manifest.json"
)

SELECTION_RESULT_05D2 = (
    PAPER5_ROOT
    / "results"
    / "simulation_benchmark"
    / "05d_neural"
    / "profile_screen_v2"
    / "neural_profile_selection_result.json"
)

SELECTION_TABLE_05D2 = (
    PAPER5_ROOT
    / "results"
    / "simulation_benchmark"
    / "05d_neural"
    / "profile_screen_v2"
    / "neural_profile_selection_table.csv"
)

OVERALL_SUMMARY_05D2 = (
    PAPER5_ROOT
    / "results"
    / "simulation_benchmark"
    / "05d_neural"
    / "profile_screen_v2"
    / "neural_profile_overall_summary.csv"
)

SHARD_MANIFEST_05D2 = (
    PAPER5_ROOT
    / "method_contract"
    / "05d2_v2"
    / "neural_profile_screen_shard_manifest.csv"
)

FINAL_GATE_DATA_ROOT = (
    PAPER5_ROOT
    / "simulations"
    / "sable_mo_v2"
    / "final_gate_calibration"
)

FUTURE_RESULT_PATHS = (
    PAPER5_ROOT / "method_contract" / "05d4",
    PAPER5_ROOT / "method_contract" / "05d5",
    PAPER5_ROOT / "method_contract" / "05e",
    PAPER5_ROOT
    / "results"
    / "simulation_benchmark"
    / "05d_neural"
    / "confirmatory",
    PAPER5_ROOT / "results" / "representation_benchmark",
)

# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05d3"

OUTPUT_CONTRACT = (
    OUTPUT_DIR
    / "selected_neural_profile.json"
)

OUTPUT_PROFILE_CSV = (
    OUTPUT_DIR
    / "selected_neural_profile.csv"
)

OUTPUT_README = (
    OUTPUT_DIR
    / "selected_neural_profile_README.txt"
)

OUTPUT_LOCK = (
    OUTPUT_DIR
    / "selected_neural_profile_lock.json"
)

OUTPUT_FILES = (
    OUTPUT_CONTRACT,
    OUTPUT_PROFILE_CSV,
    OUTPUT_README,
)


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def clean_text(
    value: Any,
) -> str:
    if value is None:
        return ""

    text = str(
        value
    ).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "null",
        "na",
    }:
        return ""

    return text


def parse_bool(
    value: Any,
) -> bool:
    if isinstance(
        value,
        bool,
    ):
        return value

    return clean_text(
        value
    ).lower() in {
        "true",
        "1",
        "yes",
    }


def relative_path(
    path: Path,
) -> str:
    try:
        return str(
            path.resolve().relative_to(
                PROJECT_ROOT.resolve()
            )
        )
    except ValueError:
        return str(
            path.resolve()
        )


def resolve_project_path(
    value: Any,
) -> Path:
    path = Path(
        clean_text(
            value
        )
    )

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def require_file(
    path: Path,
) -> None:
    if (
        not path.exists()
        or not path.is_file()
    ):
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for block in iter(
            lambda: handle.read(
                4 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def load_json(
    path: Path,
) -> dict[str, Any]:
    require_file(
        path
    )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_text_atomic(
    text: str,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tmp_",
        suffix=path.suffix or ".tmp",
        dir=str(
            path.parent
        ),
    )

    os.close(
        descriptor
    )

    temporary = Path(
        temporary_name
    )

    try:
        temporary.write_text(
            text,
            encoding="utf-8",
        )

        os.replace(
            temporary,
            path,
        )

    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(
    value: Any,
    path: Path,
) -> None:
    write_text_atomic(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        path,
    )


def write_csv_atomic(
    table: pd.DataFrame,
    path: Path,
) -> None:
    text = table.to_csv(
        index=False,
        lineterminator="\n",
    )

    write_text_atomic(
        text,
        path,
    )


def is_nonempty_path(
    path: Path,
) -> bool:
    if not path.exists():
        return False

    if path.is_file():
        return path.stat().st_size > 0

    return any(
        item.is_file()
        for item in path.rglob(
            "*"
        )
    )


def file_record(
    path: Path,
) -> dict[str, Any]:
    require_file(
        path
    )

    return {
        "path": relative_path(
            path
        ),
        "sha256": sha256_file(
            path
        ),
        "size_bytes": int(
            path.stat().st_size
        ),
    }


def verify_file_record(
    record: dict[str, Any],
    label: str,
) -> Path:
    path_text = clean_text(
        record.get(
            "path"
        )
    )

    if not path_text:
        raise RuntimeError(
            f"{label}: path missing."
        )

    path = resolve_project_path(
        path_text
    )

    require_file(
        path
    )

    expected_size = record.get(
        "size_bytes"
    )

    if (
        expected_size is not None
        and int(
            path.stat().st_size
        )
        != int(
            expected_size
        )
    ):
        raise RuntimeError(
            f"{label}: size mismatch for {path_text}."
        )

    expected_hash = clean_text(
        record.get(
            "sha256"
        )
    )

    if (
        expected_hash
        and sha256_file(
            path
        )
        != expected_hash
    ):
        raise RuntimeError(
            f"{label}: SHA256 mismatch for {path_text}."
        )

    return path


def verify_generated_files(
    lock: dict[str, Any],
    label: str,
) -> None:
    failures: list[str] = []

    generated = lock.get(
        "generated_files",
        {},
    )

    if not generated:
        raise RuntimeError(
            f"{label}: generated_files registry is empty."
        )

    for relative_name, record in generated.items():
        candidate = resolve_project_path(
            relative_name
        )

        if not candidate.exists():
            failures.append(
                f"missing: {relative_name}"
            )
            continue

        expected_size = record.get(
            "size_bytes"
        )

        if (
            expected_size is not None
            and int(
                candidate.stat().st_size
            )
            != int(
                expected_size
            )
        ):
            failures.append(
                f"size mismatch: {relative_name}"
            )
            continue

        expected_hash = clean_text(
            record.get(
                "sha256"
            )
        )

        if (
            expected_hash
            and sha256_file(
                candidate
            )
            != expected_hash
        ):
            failures.append(
                f"SHA256 mismatch: {relative_name}"
            )

    if failures:
        raise RuntimeError(
            f"{label} generated-file verification failed:\n"
            + "\n".join(
                f"- {failure}"
                for failure in failures
            )
        )


def numeric_value(
    value: Any,
    label: str,
) -> float:
    try:
        output = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise RuntimeError(
            f"{label}: expected numeric value, got {value!r}."
        ) from exc

    if not math.isfinite(
        output
    ):
        raise RuntimeError(
            f"{label}: value is non-finite."
        )

    return output


def int_value(
    value: Any,
    label: str,
) -> int:
    numeric = numeric_value(
        value,
        label,
    )

    rounded = int(
        round(
            numeric
        )
    )

    if abs(
        numeric
        - rounded
    ) > 1e-9:
        raise RuntimeError(
            f"{label}: expected integer-valued number, got {numeric}."
        )

    return rounded


def assert_close(
    observed: Any,
    expected: float,
    label: str,
    tolerance: float = 1e-12,
) -> None:
    value = numeric_value(
        observed,
        label,
    )

    if not math.isclose(
        value,
        float(
            expected
        ),
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise RuntimeError(
            f"{label}: {value} != {expected}."
        )


# ---------------------------------------------------------------------
# Prospective state guards
# ---------------------------------------------------------------------


def validate_future_state() -> None:
    future_found = [
        relative_path(
            candidate
        )
        for candidate in FUTURE_RESULT_PATHS
        if is_nonempty_path(
            candidate
        )
    ]

    if future_found:
        raise RuntimeError(
            "05d3 must be frozen before 05d4/05d5/05e or confirmatory outputs. "
            "Found:\n- "
            + "\n- ".join(
                future_found
            )
        )

    if (
        FINAL_GATE_DATA_ROOT.exists()
        and any(
            item.is_file()
            for item in FINAL_GATE_DATA_ROOT.rglob(
                "*"
            )
        )
    ):
        raise RuntimeError(
            "Final gate-calibration simulations were materialized before the "
            "selected neural profile was frozen."
        )


def validate_no_partial_05d3_outputs() -> None:
    if OUTPUT_LOCK.exists():
        return

    existing = [
        relative_path(
            path
        )
        for path in OUTPUT_FILES
        if path.exists()
    ]

    if existing:
        raise RuntimeError(
            "Partial 05d3 outputs exist without a freeze lock. "
            "Do not overwrite them silently. Review/remove the partial files first:\n- "
            + "\n- ".join(
                existing
            )
        )


# ---------------------------------------------------------------------
# 05d1 verification
# ---------------------------------------------------------------------


def verify_05d1() -> dict[str, Any]:
    required = [
        CONTRACT_05A,
        LOCK_05D1,
        PROFILE_GRID,
        SELECTION_CONTRACT_05D1,
        LOSS_CONTRACT_05D1,
        CONFIRMATORY_FIT_SEEDS_05D1,
        CONFIRMATORY_STABILITY_05D1,
    ]

    for path in required:
        require_file(
            path
        )

    lock = load_json(
        LOCK_05D1
    )

    if clean_text(
        lock.get(
            "status"
        )
    ) != "FROZEN_NEURAL_SIMULATION_PROTOCOL_BEFORE_NEURAL_MODEL_FITTING":
        raise RuntimeError(
            "Unexpected 05d1 lock status."
        )

    verify_generated_files(
        lock,
        "05d1",
    )

    profile_grid = pd.read_csv(
        PROFILE_GRID,
        dtype=str,
    ).fillna(
        ""
    )

    if profile_grid.shape[
        0
    ] != EXPECTED_PROFILES:
        raise RuntimeError(
            f"05d1 profile grid must contain {EXPECTED_PROFILES} profiles."
        )

    if profile_grid[
        "profile_id"
    ].duplicated().any():
        raise RuntimeError(
            "05d1 profile grid contains duplicate profile IDs."
        )

    selection_contract = load_json(
        SELECTION_CONTRACT_05D1
    )

    if int_value(
        selection_contract.get(
            "development_dataset_count"
        ),
        "05d1 development_dataset_count",
    ) != EXPECTED_DEVELOPMENT_DATASETS:
        raise RuntimeError(
            "05d1 development dataset count is not 360."
        )

    if clean_text(
        selection_contract.get(
            "profile_selected_for"
        )
    ) != "V4_SABLE_MO":
        raise RuntimeError(
            "05d1 selection contract no longer selects the profile for V4_SABLE_MO."
        )

    reused = set(
        selection_contract.get(
            "profile_reused_by",
            [],
        )
    )

    if reused != EXPECTED_PROFILE_REUSE:
        raise RuntimeError(
            "05d1 profile-reuse set changed."
        )

    return {
        "lock": lock,
        "profile_grid": profile_grid,
        "selection_contract": selection_contract,
    }


# ---------------------------------------------------------------------
# 05d2 v2 frozen-result verification
# ---------------------------------------------------------------------


def verify_05d2_shards(
    lock: dict[str, Any],
) -> pd.DataFrame:
    shard_record = lock.get(
        "shard_manifest",
        {},
    )

    shard_path = verify_file_record(
        shard_record,
        "05d2 shard manifest",
    )

    if shard_path.resolve() != SHARD_MANIFEST_05D2.resolve():
        raise RuntimeError(
            "05d2 lock points to an unexpected shard manifest path."
        )

    shard_manifest = pd.read_csv(
        shard_path,
        dtype=str,
    ).fillna(
        ""
    )

    if shard_manifest.shape[
        0
    ] != EXPECTED_SHARDS:
        raise RuntimeError(
            "05d2 shard manifest does not contain 3,240 rows."
        )

    required_columns = {
        "shard_path",
        "sha256",
        "size_bytes",
    }

    missing = required_columns - set(
        shard_manifest.columns
    )

    if missing:
        raise RuntimeError(
            "05d2 shard manifest missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    failures: list[str] = []

    for _, row in shard_manifest.iterrows():
        candidate = resolve_project_path(
            row[
                "shard_path"
            ]
        )

        if (
            not candidate.exists()
            or not candidate.is_file()
        ):
            failures.append(
                f"missing shard: {row['shard_path']}"
            )
            break

        if int(
            candidate.stat().st_size
        ) != int_value(
            row[
                "size_bytes"
            ],
            "05d2 shard size_bytes",
        ):
            failures.append(
                f"shard size mismatch: {row['shard_path']}"
            )
            break

        if sha256_file(
            candidate
        ) != clean_text(
            row[
                "sha256"
            ]
        ):
            failures.append(
                f"shard SHA256 mismatch: {row['shard_path']}"
            )
            break

    if failures:
        raise RuntimeError(
            "05d2 shard verification failed:\n"
            + "\n".join(
                f"- {failure}"
                for failure in failures
            )
        )

    return shard_manifest


def verify_05d2() -> dict[str, Any]:
    required = [
        LOCK_05D2,
        RESULT_MANIFEST_05D2,
        SELECTION_RESULT_05D2,
        SELECTION_TABLE_05D2,
        OVERALL_SUMMARY_05D2,
        SHARD_MANIFEST_05D2,
    ]

    for path in required:
        require_file(
            path
        )

    lock = load_json(
        LOCK_05D2
    )

    if clean_text(
        lock.get(
            "script_version"
        )
    ) != EXPECTED_05D2_SCRIPT_VERSION:
        raise RuntimeError(
            "05d2 v2 lock script-version mismatch."
        )

    if clean_text(
        lock.get(
            "status"
        )
    ) != EXPECTED_05D2_LOCK_STATUS:
        raise RuntimeError(
            "05d2 v2 is not frozen as a successful profile screen."
        )

    verify_generated_files(
        lock,
        "05d2 v2",
    )

    result_manifest_path = verify_file_record(
        lock.get(
            "result_manifest",
            {},
        ),
        "05d2 result manifest",
    )

    if result_manifest_path.resolve() != RESULT_MANIFEST_05D2.resolve():
        raise RuntimeError(
            "05d2 lock points to an unexpected result manifest."
        )

    selection_record = lock.get(
        "selection_result",
        {},
    )

    selection_result_path = verify_file_record(
        selection_record,
        "05d2 selection result",
    )

    if selection_result_path.resolve() != SELECTION_RESULT_05D2.resolve():
        raise RuntimeError(
            "05d2 lock points to an unexpected selection result."
        )

    result_manifest = load_json(
        RESULT_MANIFEST_05D2
    )

    selection_result = load_json(
        SELECTION_RESULT_05D2
    )

    if clean_text(
        result_manifest.get(
            "script_version"
        )
    ) != EXPECTED_05D2_SCRIPT_VERSION:
        raise RuntimeError(
            "05d2 result-manifest script-version mismatch."
        )

    scope = result_manifest.get(
        "scope",
        {},
    )

    expected_scope = {
        "candidate_profiles": EXPECTED_PROFILES,
        "development_datasets": EXPECTED_DEVELOPMENT_DATASETS,
        "dataset_profile_shards": EXPECTED_SHARDS,
        "passed_shards": EXPECTED_SHARDS,
        "failed_shards": 0,
        "fit_audit_rows": EXPECTED_FIT_AUDIT_ROWS,
        "preprocessing_audit_rows": EXPECTED_PREPROCESSING_AUDIT_ROWS,
        "stability_factor_rows": EXPECTED_STABILITY_FACTOR_ROWS,
    }

    for key, expected in expected_scope.items():
        if int_value(
            scope.get(
                key
            ),
            f"05d2 scope.{key}",
        ) != expected:
            raise RuntimeError(
                f"05d2 scope mismatch for {key}."
            )

    scientific_state = result_manifest.get(
        "scientific_state",
        {},
    )

    required_false = [
        "outcomes_loaded",
        "clinical_labels_loaded",
        "biological_annotations_used_for_selection",
        "TARGET_model_fitting_performed",
        "representation_method_selected",
        "final_gate_calibration_data_used",
    ]

    active = [
        key
        for key in required_false
        if parse_bool(
            scientific_state.get(
                key,
                False,
            )
        )
    ]

    if active:
        raise RuntimeError(
            "05d2 scientific-state guardrail violated: "
            + ", ".join(
                active
            )
        )

    if not parse_bool(
        scientific_state.get(
            "neural_profile_screening_performed",
            False,
        )
    ):
        raise RuntimeError(
            "05d2 result manifest does not mark neural profile screening as performed."
        )

    if not parse_bool(
        scientific_state.get(
            "neural_profile_selected",
            False,
        )
    ):
        raise RuntimeError(
            "05d2 result manifest does not mark a neural profile as selected."
        )

    if clean_text(
        selection_result.get(
            "status"
        )
    ) != EXPECTED_SELECTION_STATUS:
        raise RuntimeError(
            "05d2 selection did not end with the frozen one-SE rule."
        )

    for key in [
        "outcome_access",
        "TARGET_access",
        "confirmatory_replicates_11_to_20_accessed",
        "final_gate_calibration_accessed",
        "representation_method_selected",
    ]:
        if parse_bool(
            selection_result.get(
                key,
                False,
            )
        ):
            raise RuntimeError(
                f"05d2 selection-result guardrail violated: {key}=True."
            )

    if clean_text(
        selection_result.get(
            "profile_selected_for"
        )
    ) != "V4_SABLE_MO":
        raise RuntimeError(
            "05d2 selection result is not for V4_SABLE_MO."
        )

    reused = set(
        selection_result.get(
            "profile_reused_by_confirmatory_variants",
            [],
        )
    )

    if reused != EXPECTED_PROFILE_REUSE:
        raise RuntimeError(
            "05d2 confirmatory profile-reuse set changed."
        )

    selected_ids = {
        clean_text(
            selection_result.get(
                "selected_profile_id"
            )
        ),
        clean_text(
            selection_record.get(
                "selected_profile_id"
            )
        ),
        clean_text(
            scope.get(
                "selected_profile_id"
            )
        ),
    }

    if "" in selected_ids or len(
        selected_ids
    ) != 1:
        raise RuntimeError(
            "05d2 selected-profile ID is missing or inconsistent across frozen artifacts."
        )

    shard_manifest = verify_05d2_shards(
        lock
    )

    return {
        "lock": lock,
        "result_manifest": result_manifest,
        "selection_result": selection_result,
        "shard_manifest": shard_manifest,
    }


# ---------------------------------------------------------------------
# Independent replay of the frozen one-SE rule
# ---------------------------------------------------------------------


def replay_one_se_selection(
    selection_result: dict[str, Any],
) -> dict[str, Any]:
    table = pd.read_csv(
        SELECTION_TABLE_05D2,
        dtype=str,
    ).fillna(
        ""
    )

    required_columns = {
        "profile_id",
        "architecture_profile",
        "regularization_profile",
        "combined_complexity_rank",
        "architecture_complexity_rank",
        "regularization_complexity_rank",
        "overall_mean_simulation_utility",
        "scenario_mean_utility_se",
        "mean_stability_gated_false_biological_discovery_fraction",
        "n_failed_shards",
        "n_missing_utility",
        "n_latent_collapse",
        "hard_gate_eligible",
        "within_one_se_of_best",
        "selected_profile",
    }

    missing = required_columns - set(
        table.columns
    )

    if missing:
        raise RuntimeError(
            "05d2 selection table missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    if table.shape[
        0
    ] != EXPECTED_PROFILES:
        raise RuntimeError(
            "05d2 selection table does not contain 9 profiles."
        )

    if table[
        "profile_id"
    ].duplicated().any():
        raise RuntimeError(
            "05d2 selection table contains duplicate profile IDs."
        )

    working = table.copy()

    numeric_columns = [
        "combined_complexity_rank",
        "architecture_complexity_rank",
        "regularization_complexity_rank",
        "overall_mean_simulation_utility",
        "scenario_mean_utility_se",
        "mean_stability_gated_false_biological_discovery_fraction",
        "n_failed_shards",
        "n_missing_utility",
        "n_latent_collapse",
    ]

    for column in numeric_columns:
        working[
            column
        ] = pd.to_numeric(
            working[
                column
            ],
            errors="raise",
        )

    eligible = working[
        working[
            "hard_gate_eligible"
        ].map(
            parse_bool
        )
    ].copy()

    if eligible.empty:
        raise RuntimeError(
            "05d2 selection table contains no hard-gate-eligible profile."
        )

    best_row = eligible.sort_values(
        [
            "overall_mean_simulation_utility",
            "combined_complexity_rank",
            "architecture_complexity_rank",
            "regularization_complexity_rank",
            "profile_id",
        ],
        ascending=[
            False,
            True,
            True,
            True,
            True,
        ],
        kind="mergesort",
    ).iloc[
        0
    ]

    best_mean = float(
        best_row[
            "overall_mean_simulation_utility"
        ]
    )

    best_se = float(
        best_row[
            "scenario_mean_utility_se"
        ]
    )

    one_se_threshold = (
        best_mean
        - best_se
    )

    within = eligible[
        eligible[
            "overall_mean_simulation_utility"
        ]
        >= one_se_threshold
        - 1e-12
    ].copy()

    within = within.sort_values(
        [
            "combined_complexity_rank",
            "architecture_complexity_rank",
            "regularization_complexity_rank",
            "profile_id",
        ],
        ascending=[
            True,
            True,
            True,
            True,
        ],
        kind="mergesort",
    )

    if within.empty:
        raise RuntimeError(
            "Independent one-SE replay produced an empty within-one-SE set."
        )

    selected_row = within.iloc[
        0
    ]

    selected_profile_id = clean_text(
        selected_row[
            "profile_id"
        ]
    )

    marked_within = set(
        working.loc[
            working[
                "within_one_se_of_best"
            ].map(
                parse_bool
            ),
            "profile_id",
        ].map(
            clean_text
        )
    )

    replayed_within = set(
        within[
            "profile_id"
        ].map(
            clean_text
        )
    )

    if marked_within != replayed_within:
        raise RuntimeError(
            "05d2 within-one-SE markings do not match an independent replay."
        )

    marked_selected = working.loc[
        working[
            "selected_profile"
        ].map(
            parse_bool
        ),
        "profile_id",
    ].map(
        clean_text
    ).tolist()

    if marked_selected != [
        selected_profile_id
    ]:
        raise RuntimeError(
            "05d2 selected_profile marking does not match independent replay."
        )

    frozen_selected = clean_text(
        selection_result.get(
            "selected_profile_id"
        )
    )

    if frozen_selected != selected_profile_id:
        raise RuntimeError(
            "05d2 selection-result ID does not match independent one-SE replay."
        )

    if int_value(
        selected_row[
            "n_failed_shards"
        ],
        "selected n_failed_shards",
    ) != 0:
        raise RuntimeError(
            "Selected profile has failed shards."
        )

    if int_value(
        selected_row[
            "n_missing_utility"
        ],
        "selected n_missing_utility",
    ) != 0:
        raise RuntimeError(
            "Selected profile has missing utility values."
        )

    if int_value(
        selected_row[
            "n_latent_collapse"
        ],
        "selected n_latent_collapse",
    ) != 0:
        raise RuntimeError(
            "Selected profile has latent collapse events."
        )

    false_discovery = numeric_value(
        selected_row[
            "mean_stability_gated_false_biological_discovery_fraction"
        ],
        "selected false-discovery fraction",
    )

    if false_discovery > 0.10 + 1e-12:
        raise RuntimeError(
            "Selected profile fails the frozen <=0.10 false-discovery hard gate."
        )

    if clean_text(
        selection_result.get(
            "best_profile_id"
        )
    ) != clean_text(
        best_row[
            "profile_id"
        ]
    ):
        raise RuntimeError(
            "05d2 best_profile_id does not match independent replay."
        )

    assert_close(
        selection_result.get(
            "best_mean_simulation_utility"
        ),
        best_mean,
        "best_mean_simulation_utility",
    )

    assert_close(
        selection_result.get(
            "best_standard_error"
        ),
        best_se,
        "best_standard_error",
    )

    assert_close(
        selection_result.get(
            "one_se_threshold"
        ),
        one_se_threshold,
        "one_se_threshold",
    )

    assert_close(
        selection_result.get(
            "selected_mean_simulation_utility"
        ),
        float(
            selected_row[
                "overall_mean_simulation_utility"
            ]
        ),
        "selected_mean_simulation_utility",
    )

    if int_value(
        selection_result.get(
            "n_hard_gate_eligible_profiles"
        ),
        "n_hard_gate_eligible_profiles",
    ) != int(
        eligible.shape[
            0
        ]
    ):
        raise RuntimeError(
            "05d2 hard-gate-eligible profile count does not match replay."
        )

    if int_value(
        selection_result.get(
            "n_within_one_se_profiles"
        ),
        "n_within_one_se_profiles",
    ) != int(
        within.shape[
            0
        ]
    ):
        raise RuntimeError(
            "05d2 within-one-SE profile count does not match replay."
        )

    if int_value(
        selection_result.get(
            "selected_combined_complexity_rank"
        ),
        "selected_combined_complexity_rank",
    ) != int_value(
        selected_row[
            "combined_complexity_rank"
        ],
        "selected combined complexity rank",
    ):
        raise RuntimeError(
            "Selected combined complexity rank mismatch."
        )

    if clean_text(
        selection_result.get(
            "selected_architecture_profile"
        )
    ) != clean_text(
        selected_row[
            "architecture_profile"
        ]
    ):
        raise RuntimeError(
            "Selected architecture profile mismatch."
        )

    if clean_text(
        selection_result.get(
            "selected_regularization_profile"
        )
    ) != clean_text(
        selected_row[
            "regularization_profile"
        ]
    ):
        raise RuntimeError(
            "Selected regularization profile mismatch."
        )

    return {
        "selection_table": table,
        "selected_profile_id": selected_profile_id,
        "selected_row": selected_row,
        "best_profile_id": clean_text(
            best_row[
                "profile_id"
            ]
        ),
        "best_mean": best_mean,
        "best_se": best_se,
        "one_se_threshold": one_se_threshold,
        "n_hard_gate_eligible_profiles": int(
            eligible.shape[
                0
            ]
        ),
        "n_within_one_se_profiles": int(
            within.shape[
                0
            ]
        ),
        "within_one_se_profile_ids": sorted(
            replayed_within
        ),
        "selected_false_discovery_fraction": false_discovery,
    }


# ---------------------------------------------------------------------
# Freeze the exact selected 05d1 parameter row
# ---------------------------------------------------------------------


def selected_profile_row(
    profile_grid: pd.DataFrame,
    selected_profile_id: str,
) -> pd.Series:
    matches = profile_grid[
        profile_grid[
            "profile_id"
        ].eq(
            selected_profile_id
        )
    ]

    if matches.shape[
        0
    ] != 1:
        raise RuntimeError(
            "Selected profile ID is not unique in the frozen 05d1 profile grid."
        )

    return matches.iloc[
        0
    ].copy()


def profile_parameter_contract(
    row: pd.Series,
) -> dict[str, Any]:
    required = {
        "profile_id",
        "architecture_profile",
        "regularization_profile",
        "architecture_complexity_rank",
        "regularization_complexity_rank",
        "combined_complexity_rank",
        "arch_rna_pca_components",
        "arch_methylation_pca_components",
        "arch_cnv_pca_components",
        "arch_shared_latent_dim",
        "arch_private_latent_dim_per_view",
        "arch_encoder_hidden_dim",
        "arch_decoder_hidden_dim",
        "arch_maximum_epochs",
        "arch_early_stopping_patience",
        "arch_early_stopping_min_delta",
        "reg_reconstruction_weight",
        "reg_shared_alignment_weight",
        "reg_contrastive_weight",
        "reg_shared_private_orthogonality_weight",
        "reg_private_separation_weight",
        "reg_cross_reconstruction_weight",
        "reg_modality_dropout_probability",
        "reg_contrastive_temperature",
        "reg_learning_rate",
        "reg_weight_decay",
    }

    missing = required - set(
        row.index
    )

    if missing:
        raise RuntimeError(
            "Frozen 05d1 profile row missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    encoder_hidden = int_value(
        row[
            "arch_encoder_hidden_dim"
        ],
        "arch_encoder_hidden_dim",
    )

    decoder_hidden = int_value(
        row[
            "arch_decoder_hidden_dim"
        ],
        "arch_decoder_hidden_dim",
    )

    if encoder_hidden != decoder_hidden:
        raise RuntimeError(
            "Selected profile has unequal encoder/decoder hidden dimensions."
        )

    source_row = {
        str(
            column
        ): clean_text(
            row[
                column
            ]
        )
        for column in row.index
    }

    return {
        "profile_id": clean_text(
            row[
                "profile_id"
            ]
        ),
        "architecture_profile": clean_text(
            row[
                "architecture_profile"
            ]
        ),
        "regularization_profile": clean_text(
            row[
                "regularization_profile"
            ]
        ),
        "complexity": {
            "architecture_complexity_rank": int_value(
                row[
                    "architecture_complexity_rank"
                ],
                "architecture_complexity_rank",
            ),
            "regularization_complexity_rank": int_value(
                row[
                    "regularization_complexity_rank"
                ],
                "regularization_complexity_rank",
            ),
            "combined_complexity_rank": int_value(
                row[
                    "combined_complexity_rank"
                ],
                "combined_complexity_rank",
            ),
        },
        "parameters": {
            "pca_components": {
                "RNA": int_value(
                    row[
                        "arch_rna_pca_components"
                    ],
                    "arch_rna_pca_components",
                ),
                "METHYLATION": int_value(
                    row[
                        "arch_methylation_pca_components"
                    ],
                    "arch_methylation_pca_components",
                ),
                "CNV": int_value(
                    row[
                        "arch_cnv_pca_components"
                    ],
                    "arch_cnv_pca_components",
                ),
            },
            "shared_latent_dim": int_value(
                row[
                    "arch_shared_latent_dim"
                ],
                "arch_shared_latent_dim",
            ),
            "private_latent_dim_per_view": int_value(
                row[
                    "arch_private_latent_dim_per_view"
                ],
                "arch_private_latent_dim_per_view",
            ),
            "encoder_hidden_dim": encoder_hidden,
            "decoder_hidden_dim": decoder_hidden,
            "maximum_epochs": int_value(
                row[
                    "arch_maximum_epochs"
                ],
                "arch_maximum_epochs",
            ),
            "early_stopping_patience": int_value(
                row[
                    "arch_early_stopping_patience"
                ],
                "arch_early_stopping_patience",
            ),
            "early_stopping_min_delta": numeric_value(
                row[
                    "arch_early_stopping_min_delta"
                ],
                "arch_early_stopping_min_delta",
            ),
            "reconstruction_weight": numeric_value(
                row[
                    "reg_reconstruction_weight"
                ],
                "reg_reconstruction_weight",
            ),
            "shared_alignment_weight": numeric_value(
                row[
                    "reg_shared_alignment_weight"
                ],
                "reg_shared_alignment_weight",
            ),
            "contrastive_weight": numeric_value(
                row[
                    "reg_contrastive_weight"
                ],
                "reg_contrastive_weight",
            ),
            "shared_private_orthogonality_weight": numeric_value(
                row[
                    "reg_shared_private_orthogonality_weight"
                ],
                "reg_shared_private_orthogonality_weight",
            ),
            "private_separation_weight": numeric_value(
                row[
                    "reg_private_separation_weight"
                ],
                "reg_private_separation_weight",
            ),
            "cross_reconstruction_weight": numeric_value(
                row[
                    "reg_cross_reconstruction_weight"
                ],
                "reg_cross_reconstruction_weight",
            ),
            "modality_dropout_probability": numeric_value(
                row[
                    "reg_modality_dropout_probability"
                ],
                "reg_modality_dropout_probability",
            ),
            "contrastive_temperature": numeric_value(
                row[
                    "reg_contrastive_temperature"
                ],
                "reg_contrastive_temperature",
            ),
            "learning_rate": numeric_value(
                row[
                    "reg_learning_rate"
                ],
                "reg_learning_rate",
            ),
            "weight_decay": numeric_value(
                row[
                    "reg_weight_decay"
                ],
                "reg_weight_decay",
            ),
        },
        "source_profile_grid_row": source_row,
    }


# ---------------------------------------------------------------------
# Freeze contract and lock
# ---------------------------------------------------------------------


def build_contract(
    profile: dict[str, Any],
    replay: dict[str, Any],
    selection_result: dict[str, Any],
) -> dict[str, Any]:
    source_files = [
        CONTRACT_05A,
        LOCK_05D1,
        PROFILE_GRID,
        SELECTION_CONTRACT_05D1,
        LOSS_CONTRACT_05D1,
        CONFIRMATORY_FIT_SEEDS_05D1,
        CONFIRMATORY_STABILITY_05D1,
        LOCK_05D2,
        RESULT_MANIFEST_05D2,
        SELECTION_RESULT_05D2,
        SELECTION_TABLE_05D2,
        OVERALL_SUMMARY_05D2,
        SHARD_MANIFEST_05D2,
    ]

    return {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": FREEZE_STATUS,
        "selected_profile": profile,
        "selection_evidence": {
            "selection_status": EXPECTED_SELECTION_STATUS,
            "selection_rule": (
                "Among hard-gate-eligible profiles, identify the highest mean "
                "simulation utility. Profiles with mean utility >= best mean - "
                "SE(best) form the one-SE set. Select the lowest combined "
                "complexity rank; ties use architecture complexity, regularization "
                "complexity, then profile_id lexicographically."
            ),
            "independent_rule_replay": "PASS",
            "best_profile_id": replay[
                "best_profile_id"
            ],
            "best_mean_simulation_utility": replay[
                "best_mean"
            ],
            "best_standard_error": replay[
                "best_se"
            ],
            "one_se_threshold": replay[
                "one_se_threshold"
            ],
            "selected_mean_simulation_utility": float(
                replay[
                    "selected_row"
                ][
                    "overall_mean_simulation_utility"
                ]
            ),
            "selected_mean_stability_gated_false_biological_discovery_fraction": replay[
                "selected_false_discovery_fraction"
            ],
            "n_hard_gate_eligible_profiles": replay[
                "n_hard_gate_eligible_profiles"
            ],
            "n_within_one_se_profiles": replay[
                "n_within_one_se_profiles"
            ],
            "within_one_se_profile_ids": replay[
                "within_one_se_profile_ids"
            ],
            "selected_profile_id": replay[
                "selected_profile_id"
            ],
            "frozen_05d2_selection_order": selection_result.get(
                "selection_order",
                [],
            ),
        },
        "profile_application": {
            "selected_for": "V4_SABLE_MO",
            "reused_without_profile_retuning_by": sorted(
                EXPECTED_PROFILE_REUSE
            ),
            "confirmatory_profile_retuning_allowed": False,
            "TARGET_may_change_profile": False,
            "outcomes_may_change_profile": False,
            "confirmatory_replicates_11_to_20_may_change_profile": False,
            "final_gate_calibration_may_change_profile": False,
        },
        "scientific_state": {
            "outcomes_loaded": False,
            "clinical_labels_loaded": False,
            "biological_annotations_used_for_profile_selection": False,
            "TARGET_model_fitting_performed": False,
            "development_neural_profile_screening_performed": True,
            "development_neural_profile_selected": True,
            "selected_neural_profile_frozen": True,
            "confirmatory_replicates_11_to_20_accessed": False,
            "confirmatory_neural_ablation_performed": False,
            "final_gate_calibration_data_used": False,
            "representation_method_selected": False,
        },
        "development_screen_scope": {
            "candidate_profiles": EXPECTED_PROFILES,
            "development_datasets": EXPECTED_DEVELOPMENT_DATASETS,
            "dataset_profile_shards": EXPECTED_SHARDS,
            "passed_shards": EXPECTED_SHARDS,
            "failed_shards": 0,
            "fit_audit_rows": EXPECTED_FIT_AUDIT_ROWS,
            "preprocessing_audit_rows": EXPECTED_PREPROCESSING_AUDIT_ROWS,
            "stability_factor_rows": EXPECTED_STABILITY_FACTOR_ROWS,
        },
        "provenance": {
            "source_files": {
                relative_path(
                    path
                ): {
                    "sha256": sha256_file(
                        path
                    ),
                    "size_bytes": int(
                        path.stat().st_size
                    ),
                }
                for path in source_files
            },
            "05d1_profile_grid": file_record(
                PROFILE_GRID
            ),
            "05d1_selection_contract": file_record(
                SELECTION_CONTRACT_05D1
            ),
            "05d2_v2_results_lock": file_record(
                LOCK_05D2
            ),
            "05d2_v2_results_manifest": file_record(
                RESULT_MANIFEST_05D2
            ),
            "05d2_v2_selection_result": file_record(
                SELECTION_RESULT_05D2
            ),
            "05d2_v2_selection_table": file_record(
                SELECTION_TABLE_05D2
            ),
        },
        "guardrails": [
            "05d3 performs no neural fitting and no GPU computation.",
            "The selected profile is read only from the frozen 05d2 v2 result and independently replayed from its frozen selection table.",
            "All 3,240 05d2 v2 shard files are reverified against the frozen shard manifest before profile freeze.",
            "Only development simulation replicates 1-10 informed profile selection.",
            "Confirmatory simulation replicates 11-20 remain untouched at 05d3.",
            "TARGET matrices, outcomes, and clinical labels remain untouched at 05d3.",
            "Final gate-calibration simulations must not exist before this freeze.",
            "The selected architecture and regularization parameters are copied exactly from the frozen 05d1 profile grid.",
            "The frozen profile must be reused by confirmatory neural variants without profile retuning.",
            "This neural-profile freeze does not select the final representation method.",
        ],
        "next_required_stage": (
            "05d4_fit_neural_ablation_benchmark_gpu.py"
        ),
    }


def build_readme(
    contract: dict[str, Any],
) -> str:
    profile = contract[
        "selected_profile"
    ]

    evidence = contract[
        "selection_evidence"
    ]

    parameters = profile[
        "parameters"
    ]

    lines = [
        "Paper 5 selected neural profile freeze",
        "=" * 72,
        "",
        f"Script version: {SCRIPT_VERSION}",
        f"Freeze status: {FREEZE_STATUS}",
        "",
        "Purpose",
        "-------",
        "Freeze the outcome-blind simulation-selected neural architecture and",
        "regularization profile before any confirmatory neural ablation is run.",
        "No model is trained in this stage.",
        "",
        "Selected profile",
        "----------------",
        f"Profile ID: {profile['profile_id']}",
        f"Architecture: {profile['architecture_profile']}",
        f"Regularization: {profile['regularization_profile']}",
        f"Shared latent dimension: {parameters['shared_latent_dim']}",
        f"Private latent dimension per view: {parameters['private_latent_dim_per_view']}",
        f"Hidden dimension: {parameters['encoder_hidden_dim']}",
        "PCA components: "
        f"RNA={parameters['pca_components']['RNA']}, "
        f"METHYLATION={parameters['pca_components']['METHYLATION']}, "
        f"CNV={parameters['pca_components']['CNV']}",
        "",
        "Selection evidence",
        "------------------",
        f"Status: {evidence['selection_status']}",
        f"Independent one-SE replay: {evidence['independent_rule_replay']}",
        f"Best hard-gate-eligible profile: {evidence['best_profile_id']}",
        f"Best mean utility: {evidence['best_mean_simulation_utility']:.12g}",
        f"Best-profile SE: {evidence['best_standard_error']:.12g}",
        f"One-SE threshold: {evidence['one_se_threshold']:.12g}",
        f"Selected mean utility: {evidence['selected_mean_simulation_utility']:.12g}",
        "Selected mean stability-gated false biological discovery fraction: "
        f"{evidence['selected_mean_stability_gated_false_biological_discovery_fraction']:.12g}",
        "",
        "Guardrails",
        "----------",
        "TARGET accessed: NO",
        "Outcomes/clinical labels accessed: NO",
        "Confirmatory replicates 11-20 accessed: NO",
        "Final gate-calibration data accessed: NO",
        "Representation method selected: NO",
        "Neural fitting performed by 05d3: NO",
        "",
        "Confirmatory use",
        "---------------",
        "Use this exact frozen profile for V4 SABLE-MO and reuse it without",
        "profile retuning for V0, V1, V2, V3, and V5.",
        "",
        "Next stage",
        "----------",
        "05d4_fit_neural_ablation_benchmark_gpu.py",
        "",
    ]

    return "\n".join(
        lines
    )


def source_records_for_lock() -> dict[str, dict[str, Any]]:
    source_paths = [
        CONTRACT_05A,
        LOCK_05D1,
        PROFILE_GRID,
        SELECTION_CONTRACT_05D1,
        LOSS_CONTRACT_05D1,
        CONFIRMATORY_FIT_SEEDS_05D1,
        CONFIRMATORY_STABILITY_05D1,
        LOCK_05D2,
        RESULT_MANIFEST_05D2,
        SELECTION_RESULT_05D2,
        SELECTION_TABLE_05D2,
        OVERALL_SUMMARY_05D2,
        SHARD_MANIFEST_05D2,
    ]

    return {
        relative_path(
            path
        ): {
            "sha256": sha256_file(
                path
            ),
            "size_bytes": int(
                path.stat().st_size
            ),
        }
        for path in source_paths
    }


def build_lock(
    selected_profile_id: str,
) -> dict[str, Any]:
    generated = {
        relative_path(
            path
        ): {
            "sha256": sha256_file(
                path
            ),
            "size_bytes": int(
                path.stat().st_size
            ),
        }
        for path in OUTPUT_FILES
    }

    return {
        "contract_version": CONTRACT_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": FREEZE_STATUS,
        "selected_profile_id": selected_profile_id,
        "contract": file_record(
            OUTPUT_CONTRACT
        ),
        "selected_profile_table": file_record(
            OUTPUT_PROFILE_CSV
        ),
        "generated_files": generated,
        "frozen_source_files": source_records_for_lock(),
        "scientific_state": {
            "selected_neural_profile_frozen": True,
            "TARGET_model_fitting_performed": False,
            "confirmatory_neural_ablation_performed": False,
            "final_gate_calibration_data_used": False,
            "representation_method_selected": False,
        },
        "rewrite_policy": (
            "This zero-argument stage never rewrites a frozen 05d3 profile. "
            "Any change to the selected profile or its source evidence requires "
            "a new versioned prospective branch before confirmatory neural fitting."
        ),
        "next_required_stage": (
            "05d4_fit_neural_ablation_benchmark_gpu.py"
        ),
    }


# ---------------------------------------------------------------------
# Existing frozen 05d3 verification
# ---------------------------------------------------------------------


def verify_existing_lock(
    announce: bool = True,
) -> None:
    lock = load_json(
        OUTPUT_LOCK
    )

    failures: list[str] = []

    if clean_text(
        lock.get(
            "script_version"
        )
    ) != SCRIPT_VERSION:
        failures.append(
            "script version mismatch"
        )

    if clean_text(
        lock.get(
            "status"
        )
    ) != FREEZE_STATUS:
        failures.append(
            "freeze status mismatch"
        )

    for relative_name, record in lock.get(
        "generated_files",
        {},
    ).items():
        candidate = resolve_project_path(
            relative_name
        )

        if (
            not candidate.exists()
            or not candidate.is_file()
        ):
            failures.append(
                f"missing generated file: {relative_name}"
            )
            continue

        if int(
            candidate.stat().st_size
        ) != int(
            record[
                "size_bytes"
            ]
        ):
            failures.append(
                f"generated-file size mismatch: {relative_name}"
            )
            continue

        if sha256_file(
            candidate
        ) != clean_text(
            record[
                "sha256"
            ]
        ):
            failures.append(
                f"generated-file SHA256 mismatch: {relative_name}"
            )

    for relative_name, record in lock.get(
        "frozen_source_files",
        {},
    ).items():
        candidate = resolve_project_path(
            relative_name
        )

        if (
            not candidate.exists()
            or not candidate.is_file()
        ):
            failures.append(
                f"missing frozen source: {relative_name}"
            )
            continue

        if int(
            candidate.stat().st_size
        ) != int(
            record[
                "size_bytes"
            ]
        ):
            failures.append(
                f"frozen-source size mismatch: {relative_name}"
            )
            continue

        if sha256_file(
            candidate
        ) != clean_text(
            record[
                "sha256"
            ]
        ):
            failures.append(
                f"frozen-source SHA256 mismatch: {relative_name}"
            )

    if not failures:
        try:
            contract = load_json(
                OUTPUT_CONTRACT
            )

            if clean_text(
                contract.get(
                    "status"
                )
            ) != FREEZE_STATUS:
                failures.append(
                    "contract freeze status mismatch"
                )

            contract_profile = clean_text(
                contract.get(
                    "selected_profile",
                    {},
                ).get(
                    "profile_id"
                )
            )

            if contract_profile != clean_text(
                lock.get(
                    "selected_profile_id"
                )
            ):
                failures.append(
                    "selected profile mismatch between lock and contract"
                )

        except Exception as exc:
            failures.append(
                f"could not verify frozen contract: {type(exc).__name__}: {exc}"
            )

    if failures:
        raise RuntimeError(
            "Existing 05d3 frozen-profile verification failed:\n"
            + "\n".join(
                f"- {failure}"
                for failure in failures
            )
        )

    contract = load_json(
        OUTPUT_CONTRACT
    )

    selected = clean_text(
        contract[
            "selected_profile"
        ][
            "profile_id"
        ]
    )

    if announce:
        print(
            "=" * 88
        )

        print(
            "Existing Paper 5 selected neural profile freeze: PASS"
        )

        print(
            "=" * 88
        )

        print(
            f"Selected profile: {selected}"
        )

        print(
            f"Lock: {relative_path(OUTPUT_LOCK)}"
        )

        print(
            "No files were rewritten."
        )

        print(
            "=" * 88
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    print(
        "=" * 88
    )

    print(
        "Paper 5 - freeze selected neural profile"
    )

    print(
        "=" * 88
    )

    print(
        f"Script version: {SCRIPT_VERSION}"
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    print()

    if OUTPUT_LOCK.exists():
        verify_existing_lock(
            announce=True
        )
        return

    validate_future_state()

    validate_no_partial_05d3_outputs()

    upstream_05d1 = verify_05d1()

    print(
        "05d1 frozen neural simulation protocol: PASS"
    )

    upstream_05d2 = verify_05d2()

    print(
        "05d2 v2 frozen neural profile screen: PASS"
    )

    print(
        "05d2 v2 all 3,240 shard hashes: PASS"
    )

    replay = replay_one_se_selection(
        upstream_05d2[
            "selection_result"
        ]
    )

    print(
        "Frozen hard-gate + one-SE selection replay: PASS"
    )

    selected_id = replay[
        "selected_profile_id"
    ]

    frozen_row = selected_profile_row(
        upstream_05d1[
            "profile_grid"
        ],
        selected_id,
    )

    profile = profile_parameter_contract(
        frozen_row
    )

    if clean_text(
        profile[
            "profile_id"
        ]
    ) != selected_id:
        raise RuntimeError(
            "Selected 05d1 profile row does not match replayed 05d2 selection."
        )

    print(
        "Exact selected 05d1 architecture/regularization row: PASS"
    )

    print(
        "TARGET accessed: NO"
    )

    print(
        "Confirmatory replicates 11-20 accessed: NO"
    )

    print(
        "Final gate-calibration data accessed: NO"
    )

    print(
        "Representation method selected: NO"
    )

    print(
        "Neural fitting in 05d3: NO"
    )

    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    contract = build_contract(
        profile=profile,
        replay=replay,
        selection_result=(
            upstream_05d2[
                "selection_result"
            ]
        ),
    )

    write_json_atomic(
        contract,
        OUTPUT_CONTRACT,
    )

    selected_table = pd.DataFrame(
        [
            {
                column: frozen_row[
                    column
                ]
                for column in upstream_05d1[
                    "profile_grid"
                ].columns
            }
        ]
    )

    write_csv_atomic(
        selected_table,
        OUTPUT_PROFILE_CSV,
    )

    write_text_atomic(
        build_readme(
            contract
        ),
        OUTPUT_README,
    )

    lock = build_lock(
        selected_id
    )

    write_json_atomic(
        lock,
        OUTPUT_LOCK,
    )

    # Re-read and verify the completed freeze before reporting success.
    verify_existing_lock(
        announce=False
    )

    print()

    print(
        "-" * 88
    )

    print(
        "Frozen selected neural profile"
    )

    print(
        "-" * 88
    )

    print(
        f"Profile ID: {selected_id}"
    )

    print(
        "Architecture: "
        f"{profile['architecture_profile']}"
    )

    print(
        "Regularization: "
        f"{profile['regularization_profile']}"
    )

    print(
        "Mean simulation utility: "
        f"{replay['selected_row']['overall_mean_simulation_utility']}"
    )

    print(
        "Mean stability-gated false biological discovery fraction: "
        f"{replay['selected_false_discovery_fraction']:.12g}"
    )

    print()

    print(
        "=" * 88
    )

    print(
        "Paper 5 selected neural profile: FROZEN"
    )

    print(
        "=" * 88
    )

    print(
        f"Contract: {relative_path(OUTPUT_CONTRACT)}"
    )

    print(
        f"Profile table: {relative_path(OUTPUT_PROFILE_CSV)}"
    )

    print(
        f"Lock: {relative_path(OUTPUT_LOCK)}"
    )

    print(
        "Next stage: 05d4_fit_neural_ablation_benchmark_gpu.py"
    )

    print(
        "=" * 88
    )


if __name__ == "__main__":
    main()
