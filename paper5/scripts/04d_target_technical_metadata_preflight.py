from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_VERSION = "04d-target-technical-metadata-preflight-v1"
CONTRACT_VERSION = "paper5-target-technical-metadata-preflight-execution-v1"


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

TARGET_ROOT = (
    PAPER5_ROOT
    / "human"
    / "TARGET_OS"
)

MODEL_INPUT_DIR = (
    PAPER5_ROOT
    / "model_inputs"
    / "TARGET_OS_primary78"
)

CASE_ORDER_FILE = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_multimodal_case_order.csv"
)

SELECTED_FILE_MAP = (
    MODEL_INPUT_DIR
    / "TARGET_OS_primary_multimodal_file_map.csv"
)

GDC_FILE_INVENTORY = (
    PAPER5_ROOT
    / "data_manifest"
    / "TARGET_OS_gdc_file_inventory.csv"
)

HARMONIZED_LOCK = (
    MODEL_INPUT_DIR
    / "harmonized"
    / "harmonized_input_lock.json"
)

ADDENDUM_DIR = (
    PAPER5_ROOT
    / "method_contract"
    / "05a1"
)

ADDENDUM_FILE = (
    ADDENDUM_DIR
    / "simulation_and_technical_confound_addendum.json"
)

ADDENDUM_LOCK = (
    ADDENDUM_DIR
    / "simulation_and_technical_confound_addendum_lock.json"
)

FROZEN_TECHNICAL_CONTRACT = (
    ADDENDUM_DIR
    / "target_technical_metadata_preflight_contract.json"
)

FROZEN_COVARIATE_REGISTRY = (
    ADDENDUM_DIR
    / "target_technical_covariate_registry.csv"
)

OUTPUT_DIR = (
    PAPER5_ROOT
    / "data_manifest"
    / "TARGET_OS_technical_metadata"
)

OUTPUT_SOURCE_INVENTORY = (
    OUTPUT_DIR
    / "TARGET_OS_technical_metadata_source_inventory.csv"
)

OUTPUT_KEY_INVENTORY = (
    OUTPUT_DIR
    / "TARGET_OS_technical_metadata_key_inventory.csv"
)

OUTPUT_VALUE_PROVENANCE = (
    OUTPUT_DIR
    / "TARGET_OS_technical_metadata_value_provenance.csv"
)

OUTPUT_LINKAGE_AUDIT = (
    OUTPUT_DIR
    / "TARGET_OS_technical_metadata_linkage_audit.csv"
)

OUTPUT_SELECTED_FILE_METADATA = (
    OUTPUT_DIR
    / "TARGET_OS_selected_file_technical_metadata.csv"
)

OUTPUT_CONFLICTS = (
    OUTPUT_DIR
    / "TARGET_OS_technical_metadata_conflicts.csv"
)

OUTPUT_CASE_MATRIX = (
    OUTPUT_DIR
    / "TARGET_OS_technical_covariate_case_matrix.csv"
)

OUTPUT_COVARIATE_SUMMARY = (
    OUTPUT_DIR
    / "TARGET_OS_technical_covariate_summary.csv"
)

OUTPUT_LEVEL_COUNTS = (
    OUTPUT_DIR
    / "TARGET_OS_technical_covariate_level_counts.csv"
)

OUTPUT_CROSS_VIEW = (
    OUTPUT_DIR
    / "TARGET_OS_technical_covariate_cross_view_concordance.csv"
)

OUTPUT_RESOLVED_REGISTRY = (
    OUTPUT_DIR
    / "TARGET_OS_resolved_technical_covariate_registry.csv"
)

OUTPUT_MANIFEST = (
    OUTPUT_DIR
    / "TARGET_OS_technical_metadata_preflight_manifest.json"
)

OUTPUT_README = (
    OUTPUT_DIR
    / "TARGET_OS_technical_metadata_preflight_README.txt"
)

OUTPUT_LOCK = (
    OUTPUT_DIR
    / "TARGET_OS_technical_metadata_preflight_lock.json"
)


EXPECTED_CASES = 78
EXPECTED_MODALITIES = (
    "RNA",
    "METHYLATION",
    "CNV",
)

MODALITY_MAP = {
    "rna": "RNA",
    "methylation": "METHYLATION",
    "cnv_gene": "CNV",
    "cnv": "CNV",
}

VIEW_ORDER = {
    "RNA": 0,
    "METHYLATION": 1,
    "CNV": 2,
}


# ---------------------------------------------------------------------
# Frozen eligibility defaults
# ---------------------------------------------------------------------

DEFAULT_MAXIMUM_MISSING_FRACTION = 0.30
DEFAULT_MAXIMUM_UNIQUE_FRACTION = 0.50
DEFAULT_MINIMUM_LEVEL_SIZE = 5
DEFAULT_MAXIMUM_CATEGORICAL_LEVELS = 10

MISSING_VALUE_TOKENS = {
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "not available",
    "not applicable",
    "not reported",
    "unknown",
    "unavailable",
    "--",
}

SOURCE_PRIORITY = {
    "metadata_json": 1,
    "gdc_file_inventory": 2,
    "gdc_sample_sheet": 2,
    "files_table": 3,
    "biospecimen_table": 3,
    "selected_file_map": 4,
}

TARGET_CASE_PATTERN = re.compile(
    r"TARGET-\d{2}-[A-Z0-9]+",
    flags=re.IGNORECASE,
)

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    flags=re.IGNORECASE,
)

DATE_PATTERN = re.compile(
    r"^(\d{4})-(\d{2})(?:-(\d{2}))?",
)


# ---------------------------------------------------------------------
# Prospective field classes
# ---------------------------------------------------------------------

FORBIDDEN_CLINICAL_KEY_PATTERN = re.compile(
    r"survival|death|deceased|event|metast|recurr|response|therapy|treatment|"
    r"follow[_ ]?up|vital[_ ]?status|days[_ ]?to|age|sex|gender|race|ethnicity|"
    r"stage|grade|histolog|diagnos|prognos|progression|disease[_ ]?free|"
    r"overall[_ ]?survival|outcome|cause[_ ]?of[_ ]?death",
    flags=re.IGNORECASE,
)

BIOLOGICAL_CONTEXT_KEY_PATTERN = re.compile(
    r"tumou?r[_ ]?descriptor|primary[_ ]?site|disease[_ ]?type|project[_ ]?id|"
    r"organism|species|morphology|tumou?r[_ ]?code",
    flags=re.IGNORECASE,
)

LINKAGE_KEY_PATTERN = re.compile(
    r"(^|[._])(?:parsed_)?(?:file|case|sample|aliquot|portion|analyte|slide|submitter)?_?ids?$|"
    r"uuid|file[_ ]?name|filename|submitter[_ ]?ids?|barcode",
    flags=re.IGNORECASE,
)

TECHNICAL_FIELD_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "submission_or_creation_time_bin",
        re.compile(
            r"created[_ ]?datetime|updated[_ ]?datetime|submission[_ ]?date|"
            r"created[_ ]?date|updated[_ ]?date",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "array_plate_or_chip",
        re.compile(
            r"plate|chip|sentrix|array[_ ]?id|batch[_ ]?id|processing[_ ]?batch|"
            r"array[_ ]?barcode|slide[_ ]?barcode|chip[_ ]?id",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "sequencing_or_processing_center",
        re.compile(
            r"sequencing[_ ]?center|processing[_ ]?center|tissue[_ ]?source[_ ]?site|"
            r"(^|[._])center([._]|$)",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "platform",
        re.compile(
            r"platform|instrument[_ ]?model|array[_ ]?platform",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "workflow_type",
        re.compile(
            r"workflow[_ ]?type|analysis[_ ]?workflow|pipeline|(^|[._])workflow([._]|$)",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "experimental_strategy",
        re.compile(
            r"experimental[_ ]?strategy|assay[_ ]?type|data[_ ]?type|"
            r"library[_ ]?strategy",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "preservation_method",
        re.compile(
            r"preservation[_ ]?method|freezing[_ ]?method|(^|[._])preservation([._]|$)",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "analyte_type",
        re.compile(
            r"analyte[_ ]?type|molecule[_ ]?type|(^|[._])analyte([._]|$)",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "specimen_or_portion_type",
        re.compile(
            r"specimen[_ ]?type|portion[_ ]?type|sample[_ ]?type|tissue[_ ]?type",
            flags=re.IGNORECASE,
        ),
    ),
]

KEY_PREFERENCE_RULES: dict[str, list[re.Pattern[str]]] = {
    "submission_or_creation_time_bin": [
        re.compile(r"created[_ ]?datetime|created[_ ]?date", re.IGNORECASE),
        re.compile(r"submission[_ ]?date", re.IGNORECASE),
        re.compile(r"updated[_ ]?datetime|updated[_ ]?date", re.IGNORECASE),
    ],
    "sequencing_or_processing_center": [
        re.compile(r"sequencing[_ ]?center|processing[_ ]?center", re.IGNORECASE),
        re.compile(r"center[._]name|(^|[._])center$", re.IGNORECASE),
        re.compile(r"tissue[_ ]?source[_ ]?site", re.IGNORECASE),
        re.compile(r"center[._]code", re.IGNORECASE),
    ],
    "platform": [
        re.compile(r"array[_ ]?platform", re.IGNORECASE),
        re.compile(r"(^|[._])platform$", re.IGNORECASE),
        re.compile(r"instrument[_ ]?model", re.IGNORECASE),
    ],
    "workflow_type": [
        re.compile(r"workflow[_ ]?type", re.IGNORECASE),
        re.compile(r"analysis[_ ]?workflow", re.IGNORECASE),
        re.compile(r"pipeline", re.IGNORECASE),
    ],
    "experimental_strategy": [
        re.compile(r"experimental[_ ]?strategy", re.IGNORECASE),
        re.compile(r"assay[_ ]?type", re.IGNORECASE),
        re.compile(r"library[_ ]?strategy", re.IGNORECASE),
        re.compile(r"data[_ ]?type", re.IGNORECASE),
    ],
    "array_plate_or_chip": [
        re.compile(r"sentrix|array[_ ]?barcode|chip[_ ]?id", re.IGNORECASE),
        re.compile(r"array[_ ]?id|array[_ ]?plate", re.IGNORECASE),
        re.compile(r"(^|[._])plate([._]|$)", re.IGNORECASE),
        re.compile(r"batch[_ ]?id|processing[_ ]?batch", re.IGNORECASE),
        re.compile(r"slide[_ ]?barcode", re.IGNORECASE),
    ],
    "preservation_method": [
        re.compile(r"preservation[_ ]?method", re.IGNORECASE),
        re.compile(r"freezing[_ ]?method", re.IGNORECASE),
        re.compile(r"(^|[._])preservation([._]|$)", re.IGNORECASE),
    ],
    "analyte_type": [
        re.compile(r"analyte[_ ]?type", re.IGNORECASE),
        re.compile(r"molecule[_ ]?type", re.IGNORECASE),
        re.compile(r"(^|[._])analyte([._]|$)", re.IGNORECASE),
    ],
    "specimen_or_portion_type": [
        re.compile(r"specimen[_ ]?type", re.IGNORECASE),
        re.compile(r"portion[_ ]?type", re.IGNORECASE),
        re.compile(r"tissue[_ ]?type", re.IGNORECASE),
        re.compile(r"sample[_ ]?type", re.IGNORECASE),
    ],
}

MODEL_RESULT_MARKERS = [
    PAPER5_ROOT / "method_contract" / "05c",
    PAPER5_ROOT / "method_contract" / "05d",
    PAPER5_ROOT / "method_contract" / "05e",
    PAPER5_ROOT / "results" / "representation_benchmark",
    PAPER5_ROOT / "results" / "sable_mo",
    PAPER5_ROOT / "models" / "sable_mo",
]


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in MISSING_VALUE_TOKENS:
        return ""

    return text


def normalize_name(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    ).strip("_")
    return text


def normalize_key_path(parts: Iterable[Any]) -> str:
    normalized = [
        normalize_name(part)
        for part in parts
        if normalize_name(part)
    ]
    return ".".join(normalized)


def normalize_categorical_value(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    text = text.strip(
        "\"'"
    )

    if text.lower() in MISSING_VALUE_TOKENS:
        return ""

    return text


def comparison_value(value: Any) -> str:
    text = normalize_categorical_value(value)
    return text.casefold()


def relative_path(path: Path) -> str:
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


def resolve_project_path(value: Any) -> Path:
    path = Path(
        clean_text(value)
    )
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                4 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def hash_strings(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()

    for value in values:
        digest.update(
            str(value).encode("utf-8")
        )
        digest.update(b"\n")

    return digest.hexdigest()


def load_json(path: Path) -> Any:
    require_file(path)
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_json_atomic(
    payload: Any,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


def write_csv_atomic(
    table: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name + ".tmp"
    )

    table.to_csv(
        temporary,
        index=False,
        float_format="%.12g",
    )

    temporary.replace(path)


def is_nonempty_path(path: Path) -> bool:
    if not path.exists():
        return False

    if path.is_file():
        return path.stat().st_size > 0

    return any(
        child.is_file()
        for child in path.rglob("*")
    )


def extract_target_case_ids(value: Any) -> list[str]:
    text = clean_text(value).upper()
    if not text:
        return []

    output: list[str] = []

    for match in TARGET_CASE_PATTERN.findall(text):
        case_id = match.upper()
        if case_id not in output:
            output.append(case_id)

    return output


def is_uuid_like(value: Any) -> bool:
    return bool(
        UUID_PATTERN.fullmatch(
            clean_text(value)
        )
    )


def derive_calendar_quarter(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""

    match = DATE_PATTERN.match(text)
    if match is None:
        return ""

    year = int(match.group(1))
    month = int(match.group(2))

    if month < 1 or month > 12:
        return ""

    quarter = (
        (month - 1) // 3
    ) + 1

    return f"{year}-Q{quarter}"


def classify_key_path(
    key_path: str,
) -> tuple[str, str]:
    normalized = normalize_name(
        key_path.replace(".", "_")
    )

    if FORBIDDEN_CLINICAL_KEY_PATTERN.search(
        normalized
    ):
        return (
            "forbidden_clinical_or_outcome",
            "",
        )

    if BIOLOGICAL_CONTEXT_KEY_PATTERN.search(
        normalized
    ):
        return (
            "excluded_biological_context",
            "",
        )

    for covariate_id, pattern in TECHNICAL_FIELD_RULES:
        if pattern.search(
            normalized
        ):
            return (
                "technical_candidate",
                covariate_id,
            )

    if LINKAGE_KEY_PATTERN.search(
        normalized
    ):
        return (
            "linkage_identifier",
            "",
        )

    return (
        "ignored_non_technical",
        "",
    )


def key_preference_score(
    covariate_id: str,
    key_path: str,
) -> int:
    patterns = KEY_PREFERENCE_RULES.get(
        covariate_id,
        [],
    )

    normalized = normalize_name(
        key_path.replace(".", "_")
    )

    for index, pattern in enumerate(patterns):
        if pattern.search(normalized):
            return index

    return len(patterns) + 10


def expand_candidate_values(
    covariate_id: str,
    value: Any,
) -> list[str]:
    text = normalize_categorical_value(value)
    if not text:
        return []

    if covariate_id == "submission_or_creation_time_bin":
        quarter = derive_calendar_quarter(text)
        return [quarter] if quarter else []

    if (
        covariate_id
        not in {
            "array_plate_or_chip",
        }
        and (
            is_uuid_like(text)
            or bool(
                TARGET_CASE_PATTERN.search(
                    text
                )
            )
        )
    ):
        return []

    split_allowed = covariate_id in {
        "experimental_strategy",
        "preservation_method",
        "analyte_type",
        "specimen_or_portion_type",
    }

    if split_allowed and (
        ";" in text
        or ", " in text
    ):
        parts = [
            normalize_categorical_value(part)
            for part in re.split(
                r"\s*;\s*|\s*,\s*",
                text,
            )
        ]

        parts = [
            part
            for part in parts
            if part
        ]

        if 1 <= len(parts) <= 8:
            return list(
                dict.fromkeys(parts)
            )

    return [text]


# ---------------------------------------------------------------------
# Existing-lock verification
# ---------------------------------------------------------------------


def verify_existing_output_lock() -> bool:
    if not OUTPUT_LOCK.exists():
        return False

    lock = load_json(
        OUTPUT_LOCK
    )

    generated_files = lock.get(
        "generated_files",
        {},
    )

    if not generated_files:
        raise RuntimeError(
            "The existing 04d lock contains no generated-file registry."
        )

    print(
        "An existing frozen TARGET technical-metadata preflight was detected."
    )
    print(
        "Verifying locked artifacts without rewriting them..."
    )

    for index, (
        path_text,
        expected,
    ) in enumerate(
        sorted(
            generated_files.items()
        ),
        start=1,
    ):
        path = resolve_project_path(
            path_text
        )
        require_file(path)

        observed_size = int(
            path.stat().st_size
        )
        expected_size = int(
            expected["size_bytes"]
        )

        if observed_size != expected_size:
            raise RuntimeError(
                f"Locked 04d artifact size mismatch: {path}"
            )

        observed_hash = sha256_file(path)
        expected_hash = clean_text(
            expected["sha256"]
        )

        if observed_hash != expected_hash:
            raise RuntimeError(
                f"Locked 04d artifact hash mismatch: {path}"
            )

        if (
            index == 1
            or index % 5 == 0
            or index == len(generated_files)
        ):
            print(
                f" {index:02d}/{len(generated_files):02d} locked artifacts: PASS"
            )

    upstream_inputs = lock.get(
        "upstream_inputs",
        {},
    )

    for label, record in upstream_inputs.items():
        path = resolve_project_path(
            record["path"]
        )
        require_file(path)

        if sha256_file(path) != clean_text(
            record["sha256"]
        ):
            raise RuntimeError(
                f"Upstream input changed after the 04d freeze: {label}"
            )

    print()
    print("=" * 88)
    print(
        "Existing TARGET technical-metadata preflight: PASS"
    )
    print("=" * 88)
    print(
        f"Lock: {relative_path(OUTPUT_LOCK)}"
    )
    return True


# ---------------------------------------------------------------------
# Upstream contract validation
# ---------------------------------------------------------------------


def validate_no_representation_results() -> None:
    found = [
        relative_path(path)
        for path in MODEL_RESULT_MARKERS
        if is_nonempty_path(path)
    ]

    if found:
        raise RuntimeError(
            "04d must be frozen before representation-model results are generated. "
            "Found:\n- "
            + "\n- ".join(found)
        )


def verify_addendum_generated_file(
    addendum_lock: dict[str, Any],
    path: Path,
) -> str:
    generated = addendum_lock.get(
        "generated_files",
        {},
    )

    key = relative_path(path)

    if key not in generated:
        raise RuntimeError(
            f"05a1 lock does not register required file: {key}"
        )

    expected = generated[key]
    require_file(path)

    observed_size = int(
        path.stat().st_size
    )

    if observed_size != int(
        expected["size_bytes"]
    ):
        raise RuntimeError(
            f"05a1 generated-file size mismatch: {path}"
        )

    observed_hash = sha256_file(path)

    if observed_hash != clean_text(
        expected["sha256"]
    ):
        raise RuntimeError(
            f"05a1 generated-file hash mismatch: {path}"
        )

    return observed_hash


def validate_upstream_contracts() -> dict[str, Any]:
    for path in [
        CASE_ORDER_FILE,
        SELECTED_FILE_MAP,
        GDC_FILE_INVENTORY,
        HARMONIZED_LOCK,
        ADDENDUM_FILE,
        ADDENDUM_LOCK,
        FROZEN_TECHNICAL_CONTRACT,
        FROZEN_COVARIATE_REGISTRY,
    ]:
        require_file(path)

    validate_no_representation_results()

    addendum = load_json(
        ADDENDUM_FILE
    )
    addendum_lock = load_json(
        ADDENDUM_LOCK
    )
    technical_contract = load_json(
        FROZEN_TECHNICAL_CONTRACT
    )

    scientific_state = addendum.get(
        "scientific_state",
        {},
    )

    forbidden_true_states = {
        "outcome_loaded": scientific_state.get(
            "outcome_loaded",
            True,
        ),
        "clinical_labels_loaded": scientific_state.get(
            "clinical_labels_loaded",
            True,
        ),
        "biological_annotations_loaded": scientific_state.get(
            "biological_annotations_loaded",
            True,
        ),
        "simulation_model_fitting_performed": scientific_state.get(
            "simulation_model_fitting_performed",
            True,
        ),
        "target_model_fitting_performed": scientific_state.get(
            "target_model_fitting_performed",
            True,
        ),
    }

    failed_states = [
        key
        for key, value in forbidden_true_states.items()
        if bool(value)
    ]

    if failed_states:
        raise RuntimeError(
            "05a1 scientific state is incompatible with a prospective 04d preflight: "
            + ", ".join(failed_states)
        )

    if clean_text(
        technical_contract.get(
            "status"
        )
    ) != "FROZEN_BEFORE_MODEL_FITTING":
        raise RuntimeError(
            "The frozen TARGET technical-metadata contract has an unexpected status."
        )

    technical_contract_hash = verify_addendum_generated_file(
        addendum_lock,
        FROZEN_TECHNICAL_CONTRACT,
    )

    registry_hash = verify_addendum_generated_file(
        addendum_lock,
        FROZEN_COVARIATE_REGISTRY,
    )

    if sha256_file(ADDENDUM_FILE) != clean_text(
        addendum_lock.get(
            "addendum_sha256"
        )
    ):
        raise RuntimeError(
            "05a1 addendum hash does not match its lock."
        )

    harmonized_lock = load_json(
        HARMONIZED_LOCK
    )

    if bool(
        harmonized_lock.get(
            "outcome_loaded",
            True,
        )
    ):
        raise RuntimeError(
            "The upstream harmonized lock does not confirm outcome_loaded=false."
        )

    return {
        "addendum": addendum,
        "addendum_lock": addendum_lock,
        "technical_contract": technical_contract,
        "technical_contract_hash": technical_contract_hash,
        "covariate_registry_hash": registry_hash,
        "harmonized_lock": harmonized_lock,
    }


# ---------------------------------------------------------------------
# Selected molecular files
# ---------------------------------------------------------------------


def load_case_order() -> list[str]:
    table = pd.read_csv(
        CASE_ORDER_FILE,
        dtype=str,
    ).fillna("")

    if "case_id" not in table.columns:
        raise ValueError(
            "The locked case-order file does not contain case_id."
        )

    case_ids = (
        table["case_id"]
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )

    if len(case_ids) != EXPECTED_CASES:
        raise RuntimeError(
            f"Expected {EXPECTED_CASES} locked cases; found {len(case_ids)}."
        )

    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError(
            "Duplicate case IDs were found in the locked case order."
        )

    return case_ids


def load_selected_files(
    case_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    file_map = pd.read_csv(
        SELECTED_FILE_MAP,
        dtype=str,
        low_memory=False,
    ).fillna("")

    required = {
        "case_id",
        "modality",
        "file_name",
        "local_path",
    }

    missing = required - set(file_map.columns)
    if missing:
        raise ValueError(
            "Selected file map is missing columns: "
            + ", ".join(sorted(missing))
        )

    file_map["case_id"] = (
        file_map["case_id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    file_map["modality_original"] = (
        file_map["modality"]
        .astype(str)
    )

    file_map["modality"] = (
        file_map["modality"]
        .astype(str)
        .str.lower()
        .map(MODALITY_MAP)
    )

    if file_map["modality"].isna().any():
        values = sorted(
            set(
                file_map.loc[
                    file_map["modality"].isna(),
                    "modality_original",
                ].astype(str)
            )
        )
        raise RuntimeError(
            f"Unsupported modalities in the selected file map: {values}"
        )

    if set(file_map["case_id"]) != set(case_ids):
        raise RuntimeError(
            "Selected molecular file map does not contain exactly the locked cases."
        )

    expected_rows = (
        len(case_ids)
        * len(EXPECTED_MODALITIES)
    )

    if file_map.shape[0] != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} selected case-modality rows; "
            f"found {file_map.shape[0]}."
        )

    if file_map.duplicated(
        ["case_id", "modality"]
    ).any():
        raise RuntimeError(
            "More than one selected file exists for at least one case-modality pair."
        )

    inventory = pd.read_csv(
        GDC_FILE_INVENTORY,
        dtype=str,
        low_memory=False,
    ).fillna("")

    if "file_name" not in inventory.columns:
        raise ValueError(
            "GDC file inventory does not contain file_name."
        )

    inventory_unique = (
        inventory
        .sort_values("file_name")
        .drop_duplicates(
            "file_name",
            keep="first",
        )
    )

    join_columns = [
        column
        for column in [
            "file_name",
            "file_id",
            "data_category",
            "data_type",
            "project_id",
            "case_id",
            "sample_id",
            "tissue_type",
            "tumor_descriptor",
            "specimen_type",
            "preservation_method",
            "parsed_case_ids",
            "parsed_sample_ids",
            "source_sample_sheet",
            "downloaded",
        ]
        if column in inventory_unique.columns
    ]

    inventory_for_join = inventory_unique[
        join_columns
    ].copy()

    rename_map = {
        column: f"inventory_{column}"
        for column in join_columns
        if column != "file_name"
    }

    inventory_for_join = inventory_for_join.rename(
        columns=rename_map
    )

    selected = file_map.merge(
        inventory_for_join,
        how="left",
        on="file_name",
        validate="one_to_one",
    )

    if "inventory_file_id" not in selected.columns:
        selected["inventory_file_id"] = ""

    selected["file_id"] = (
        selected["inventory_file_id"]
        .astype(str)
        .str.strip()
    )

    case_index = {
        case_id: index
        for index, case_id in enumerate(case_ids)
    }

    selected["case_order_index"] = (
        selected["case_id"]
        .map(case_index)
        .astype(int)
    )

    selected["view_order_index"] = (
        selected["modality"]
        .map(VIEW_ORDER)
        .astype(int)
    )

    selected = selected.sort_values(
        [
            "case_order_index",
            "view_order_index",
        ]
    ).reset_index(drop=True)

    selected["selected_file_key"] = (
        selected["case_id"]
        + "::"
        + selected["modality"]
    )

    return selected, inventory


# ---------------------------------------------------------------------
# Metadata source discovery
# ---------------------------------------------------------------------


def discover_metadata_sources() -> list[tuple[Path, str]]:
    sources: dict[Path, str] = {
        GDC_FILE_INVENTORY.resolve(): (
            "gdc_file_inventory"
        ),
        SELECTED_FILE_MAP.resolve(): (
            "selected_file_map"
        ),
    }

    if TARGET_ROOT.exists():
        for path in TARGET_ROOT.rglob("*"):
            if not path.is_file():
                continue

            lower = path.name.lower()
            suffix = path.suffix.lower()

            if (
                lower.startswith("gdc_sample_sheet")
                and suffix == ".tsv"
            ):
                sources[
                    path.resolve()
                ] = "gdc_sample_sheet"

            elif (
                lower.startswith("files-table")
                and suffix == ".tsv"
            ):
                sources[
                    path.resolve()
                ] = "files_table"

            elif (
                "metadata" in lower
                and suffix == ".json"
            ):
                sources[
                    path.resolve()
                ] = "metadata_json"

            elif (
                "biospecimen" in lower
                and suffix in {
                    ".tsv",
                    ".csv",
                }
            ):
                sources[
                    path.resolve()
                ] = "biospecimen_table"

    return sorted(
        [
            (path, source_type)
            for path, source_type in sources.items()
        ],
        key=lambda item: (
            SOURCE_PRIORITY.get(
                item[1],
                99,
            ),
            str(item[0]),
        ),
    )


# ---------------------------------------------------------------------
# Metadata extraction state
# ---------------------------------------------------------------------


class ExtractionState:
    def __init__(
        self,
        selected: pd.DataFrame,
    ) -> None:
        self.selected = selected

        self.by_file_name: dict[str, dict[str, Any]] = {
            str(row["file_name"]): row.to_dict()
            for _, row in selected.iterrows()
        }

        self.by_file_id: dict[str, dict[str, Any]] = {
            str(row["file_id"]): row.to_dict()
            for _, row in selected.iterrows()
            if clean_text(row["file_id"])
        }

        self.provenance_rows: list[
            dict[str, Any]
        ] = []

        self.linkage_rows: list[
            dict[str, Any]
        ] = []

        self.source_rows: list[
            dict[str, Any]
        ] = []

        self.key_counts: dict[
            tuple[str, str, str, str, str],
            dict[str, int],
        ] = defaultdict(
            lambda: {
                "n_occurrences": 0,
                "n_nonempty_values": 0,
                "n_selected_records": 0,
            }
        )

    def match_selected(
        self,
        file_name: Any,
        file_id: Any,
    ) -> dict[str, Any] | None:
        name = clean_text(file_name)
        identifier = clean_text(file_id)

        if name and name in self.by_file_name:
            return self.by_file_name[name]

        if identifier and identifier in self.by_file_id:
            return self.by_file_id[identifier]

        return None

    def register_key(
        self,
        source_file: Path,
        source_type: str,
        key_path: str,
        classification: str,
        covariate_id: str,
        nonempty: bool,
    ) -> None:
        key = (
            relative_path(source_file),
            source_type,
            key_path,
            classification,
            covariate_id,
        )

        record = self.key_counts[key]
        record["n_occurrences"] += 1
        record["n_selected_records"] += 1
        if nonempty:
            record["n_nonempty_values"] += 1

    def add_linkage(
        self,
        selected_record: dict[str, Any],
        source_file: Path,
        source_type: str,
        key_path: str,
        value: Any,
    ) -> None:
        case_ids = extract_target_case_ids(
            value
        )

        for case_id in case_ids:
            self.linkage_rows.append(
                {
                    "case_id": (
                        selected_record["case_id"]
                    ),
                    "modality": (
                        selected_record["modality"]
                    ),
                    "file_name": (
                        selected_record["file_name"]
                    ),
                    "file_id": (
                        selected_record["file_id"]
                    ),
                    "source_type": source_type,
                    "source_file": (
                        relative_path(source_file)
                    ),
                    "source_key_path": key_path,
                    "observed_case_id": case_id,
                    "case_id_matches_selected": (
                        case_id
                        == selected_record["case_id"]
                    ),
                }
            )

    def add_candidate(
        self,
        selected_record: dict[str, Any],
        source_file: Path,
        source_type: str,
        key_path: str,
        covariate_id: str,
        value: Any,
    ) -> None:
        values = expand_candidate_values(
            covariate_id,
            value,
        )

        for normalized_value in values:
            self.provenance_rows.append(
                {
                    "case_id": (
                        selected_record["case_id"]
                    ),
                    "modality": (
                        selected_record["modality"]
                    ),
                    "file_name": (
                        selected_record["file_name"]
                    ),
                    "file_id": (
                        selected_record["file_id"]
                    ),
                    "covariate_id": covariate_id,
                    "value": normalized_value,
                    "value_comparison": (
                        comparison_value(
                            normalized_value
                        )
                    ),
                    "source_type": source_type,
                    "source_priority": (
                        SOURCE_PRIORITY.get(
                            source_type,
                            99,
                        )
                    ),
                    "source_file": (
                        relative_path(source_file)
                    ),
                    "source_key_path": key_path,
                    "key_preference_score": (
                        key_preference_score(
                            covariate_id,
                            key_path,
                        )
                    ),
                    "date_was_binned": (
                        covariate_id
                        == "submission_or_creation_time_bin"
                    ),
                }
            )

    def process_leaf(
        self,
        selected_record: dict[str, Any],
        source_file: Path,
        source_type: str,
        key_path: str,
        value: Any,
    ) -> None:
        classification, covariate_id = (
            classify_key_path(
                key_path
            )
        )

        text = clean_text(value)

        self.register_key(
            source_file=source_file,
            source_type=source_type,
            key_path=key_path,
            classification=classification,
            covariate_id=covariate_id,
            nonempty=bool(text),
        )

        if classification == "linkage_identifier":
            self.add_linkage(
                selected_record=selected_record,
                source_file=source_file,
                source_type=source_type,
                key_path=key_path,
                value=value,
            )

        elif classification == "technical_candidate":
            self.add_candidate(
                selected_record=selected_record,
                source_file=source_file,
                source_type=source_type,
                key_path=key_path,
                covariate_id=covariate_id,
                value=value,
            )


# ---------------------------------------------------------------------
# Table parsing
# ---------------------------------------------------------------------


def detect_file_name_column(
    table: pd.DataFrame,
) -> str | None:
    candidates = [
        "file_name",
        "filename",
        "name",
    ]

    for candidate in candidates:
        if candidate in table.columns:
            return candidate

    return None


def detect_file_id_column(
    table: pd.DataFrame,
) -> str | None:
    candidates = [
        "file_id",
        "id",
    ]

    for candidate in candidates:
        if candidate in table.columns:
            return candidate

    return None


def parse_table_source(
    path: Path,
    source_type: str,
    state: ExtractionState,
) -> None:
    separator = (
        "\t"
        if path.suffix.lower() == ".tsv"
        else ","
    )

    source_record = {
        "source_file": relative_path(path),
        "source_type": source_type,
        "size_bytes": int(
            path.stat().st_size
        ),
        "sha256": sha256_file(path),
        "parse_status": "",
        "n_records_total": 0,
        "n_selected_file_records": 0,
        "n_technical_values_extracted": 0,
        "error": "",
    }

    provenance_before = len(
        state.provenance_rows
    )

    try:
        table = pd.read_csv(
            path,
            sep=separator,
            dtype=str,
            low_memory=False,
        ).fillna("")

        original_columns = list(
            table.columns
        )

        table.columns = [
            normalize_name(column)
            for column in table.columns
        ]

        source_record[
            "n_records_total"
        ] = int(table.shape[0])

        file_name_column = detect_file_name_column(
            table
        )
        file_id_column = detect_file_id_column(
            table
        )

        if file_name_column is None and file_id_column is None:
            source_record[
                "parse_status"
            ] = "SKIPPED_NO_FILE_LINK_FIELD"
            state.source_rows.append(
                source_record
            )
            return

        selected_count = 0

        for row_index, row in table.iterrows():
            file_name = (
                row.get(
                    file_name_column,
                    "",
                )
                if file_name_column
                else ""
            )

            file_id = (
                row.get(
                    file_id_column,
                    "",
                )
                if file_id_column
                else ""
            )

            selected_record = state.match_selected(
                file_name=file_name,
                file_id=file_id,
            )

            if selected_record is None:
                continue

            selected_count += 1

            for column in table.columns:
                value = row.get(
                    column,
                    "",
                )

                state.process_leaf(
                    selected_record=selected_record,
                    source_file=path,
                    source_type=source_type,
                    key_path=column,
                    value=value,
                )

        source_record[
            "n_selected_file_records"
        ] = selected_count
        source_record[
            "n_technical_values_extracted"
        ] = (
            len(state.provenance_rows)
            - provenance_before
        )
        source_record[
            "parse_status"
        ] = "PASS"
        source_record[
            "original_columns"
        ] = ";".join(
            str(column)
            for column in original_columns
        )

    except Exception as exc:
        source_record[
            "parse_status"
        ] = "FAILED"
        source_record[
            "error"
        ] = (
            f"{type(exc).__name__}: {exc}"
        )

    state.source_rows.append(
        source_record
    )


# ---------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------


def direct_dict_value(
    record: dict[str, Any],
    candidates: Iterable[str],
) -> Any:
    normalized_map = {
        normalize_name(key): value
        for key, value in record.items()
    }

    for candidate in candidates:
        if candidate in normalized_map:
            return normalized_map[candidate]

    return ""


def iter_selected_json_records(
    obj: Any,
    state: ExtractionState,
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    seen_objects: set[int] = set()

    def walk(value: Any) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
        if isinstance(value, dict):
            object_id = id(value)

            file_name = direct_dict_value(
                value,
                [
                    "file_name",
                    "filename",
                ],
            )

            file_id = direct_dict_value(
                value,
                [
                    "file_id",
                ],
            )

            selected_record = state.match_selected(
                file_name=file_name,
                file_id=file_id,
            )

            if (
                selected_record is not None
                and object_id not in seen_objects
            ):
                seen_objects.add(object_id)
                yield value, selected_record

            for child in value.values():
                yield from walk(child)

        elif isinstance(value, list):
            for child in value:
                yield from walk(child)

    yield from walk(obj)


def traverse_selected_json_record(
    value: Any,
    path_parts: tuple[str, ...],
    selected_record: dict[str, Any],
    source_file: Path,
    state: ExtractionState,
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_name = normalize_name(key)
            child_path = (
                *path_parts,
                key_name,
            )
            key_path = normalize_key_path(
                child_path
            )

            classification, covariate_id = (
                classify_key_path(
                    key_path
                )
            )

            if classification in {
                "forbidden_clinical_or_outcome",
                "excluded_biological_context",
            }:
                state.register_key(
                    source_file=source_file,
                    source_type="metadata_json",
                    key_path=key_path,
                    classification=classification,
                    covariate_id=covariate_id,
                    nonempty=False,
                )
                continue

            traverse_selected_json_record(
                value=child,
                path_parts=child_path,
                selected_record=selected_record,
                source_file=source_file,
                state=state,
            )

    elif isinstance(value, list):
        for child in value:
            traverse_selected_json_record(
                value=child,
                path_parts=path_parts,
                selected_record=selected_record,
                source_file=source_file,
                state=state,
            )

    else:
        key_path = normalize_key_path(
            path_parts
        )

        state.process_leaf(
            selected_record=selected_record,
            source_file=source_file,
            source_type="metadata_json",
            key_path=key_path,
            value=value,
        )


def parse_json_source(
    path: Path,
    state: ExtractionState,
) -> None:
    source_record = {
        "source_file": relative_path(path),
        "source_type": "metadata_json",
        "size_bytes": int(
            path.stat().st_size
        ),
        "sha256": sha256_file(path),
        "parse_status": "",
        "n_records_total": np.nan,
        "n_selected_file_records": 0,
        "n_technical_values_extracted": 0,
        "error": "",
    }

    provenance_before = len(
        state.provenance_rows
    )

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        selected_records = list(
            iter_selected_json_records(
                payload,
                state,
            )
        )

        for record, selected_record in selected_records:
            traverse_selected_json_record(
                value=record,
                path_parts=(),
                selected_record=selected_record,
                source_file=path,
                state=state,
            )

        source_record[
            "n_selected_file_records"
        ] = len(selected_records)
        source_record[
            "n_technical_values_extracted"
        ] = (
            len(state.provenance_rows)
            - provenance_before
        )
        source_record[
            "parse_status"
        ] = "PASS"

    except Exception as exc:
        source_record[
            "parse_status"
        ] = "FAILED"
        source_record[
            "error"
        ] = (
            f"{type(exc).__name__}: {exc}"
        )

    state.source_rows.append(
        source_record
    )


# ---------------------------------------------------------------------
# Base selected-file map extraction
# ---------------------------------------------------------------------


def extract_selected_file_map_fields(
    selected: pd.DataFrame,
    state: ExtractionState,
) -> None:
    path = SELECTED_FILE_MAP

    for _, row in selected.iterrows():
        selected_record = row.to_dict()

        for column in [
            "tissue_type",
            "parsed_sample_ids",
            "file_name",
            "file_id",
        ]:
            if column not in row.index:
                continue

            state.process_leaf(
                selected_record=selected_record,
                source_file=path,
                source_type="selected_file_map",
                key_path=column,
                value=row[column],
            )

        for column in [
            "inventory_tissue_type",
            "inventory_specimen_type",
            "inventory_preservation_method",
            "inventory_data_type",
            "inventory_parsed_case_ids",
            "inventory_parsed_sample_ids",
            "inventory_case_id",
            "inventory_sample_id",
        ]:
            if column not in row.index:
                continue

            state.process_leaf(
                selected_record=selected_record,
                source_file=GDC_FILE_INVENTORY,
                source_type="gdc_file_inventory",
                key_path=column.replace(
                    "inventory_",
                    "",
                    1,
                ),
                value=row[column],
            )


# ---------------------------------------------------------------------
# Canonical metadata resolution
# ---------------------------------------------------------------------


def build_key_inventory(
    state: ExtractionState,
) -> pd.DataFrame:
    rows = []

    for (
        source_file,
        source_type,
        key_path,
        classification,
        covariate_id,
    ), counts in sorted(
        state.key_counts.items()
    ):
        rows.append(
            {
                "source_file": source_file,
                "source_type": source_type,
                "key_path": key_path,
                "classification": classification,
                "covariate_id": covariate_id,
                **counts,
            }
        )

    return pd.DataFrame(rows)


def resolve_canonical_values(
    selected: pd.DataFrame,
    provenance: pd.DataFrame,
    covariate_ids: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    canonical_rows = []
    conflict_rows = []

    if provenance.empty:
        provenance = pd.DataFrame(
            columns=[
                "case_id",
                "modality",
                "file_name",
                "covariate_id",
                "value",
                "value_comparison",
                "source_type",
                "source_priority",
                "source_file",
                "source_key_path",
                "key_preference_score",
            ]
        )

    for _, selected_row in selected.iterrows():
        record = {
            "case_order_index": int(
                selected_row[
                    "case_order_index"
                ]
            ),
            "case_id": selected_row["case_id"],
            "modality": selected_row["modality"],
            "file_name": selected_row["file_name"],
            "file_id": selected_row["file_id"],
            "local_path": selected_row["local_path"],
        }

        file_part = provenance[
            provenance["file_name"].eq(
                selected_row["file_name"]
            )
        ]

        observed_covariates = 0
        conflicted_covariates = 0
        source_files = set()

        for covariate_id in covariate_ids:
            part = file_part[
                file_part["covariate_id"].eq(
                    covariate_id
                )
            ].copy()

            value_column = covariate_id
            conflict_column = (
                f"{covariate_id}__conflict"
            )
            source_column = (
                f"{covariate_id}__source"
            )
            key_path_column = (
                f"{covariate_id}__key_path"
            )

            if part.empty:
                record[value_column] = ""
                record[conflict_column] = False
                record[source_column] = ""
                record[key_path_column] = ""
                continue

            minimum_source_priority = int(
                part["source_priority"].min()
            )

            part = part[
                part["source_priority"].eq(
                    minimum_source_priority
                )
            ].copy()

            minimum_key_priority = int(
                part[
                    "key_preference_score"
                ].min()
            )

            part = part[
                part[
                    "key_preference_score"
                ].eq(
                    minimum_key_priority
                )
            ].copy()

            value_map: dict[str, str] = {}

            for _, value_row in part.iterrows():
                comparison = clean_text(
                    value_row[
                        "value_comparison"
                    ]
                )
                display = clean_text(
                    value_row["value"]
                )

                if comparison and comparison not in value_map:
                    value_map[comparison] = display

                source_files.add(
                    clean_text(
                        value_row[
                            "source_file"
                        ]
                    )
                )

            values = sorted(
                value_map.values(),
                key=lambda value: value.casefold(),
            )

            if not values:
                record[value_column] = ""
                record[conflict_column] = False
                record[source_column] = ""
                record[key_path_column] = ""
                continue

            observed_covariates += 1
            conflict = len(values) > 1

            if conflict:
                conflicted_covariates += 1

            canonical_value = (
                " || ".join(values)
            )

            record[value_column] = canonical_value
            record[conflict_column] = conflict
            record[source_column] = (
                ";".join(
                    sorted(
                        set(
                            part[
                                "source_type"
                            ].astype(str)
                        )
                    )
                )
            )
            record[key_path_column] = (
                ";".join(
                    sorted(
                        set(
                            part[
                                "source_key_path"
                            ].astype(str)
                        )
                    )
                )
            )

            if conflict:
                conflict_rows.append(
                    {
                        "case_id": (
                            selected_row["case_id"]
                        ),
                        "modality": (
                            selected_row["modality"]
                        ),
                        "file_name": (
                            selected_row["file_name"]
                        ),
                        "covariate_id": covariate_id,
                        "canonical_value": (
                            canonical_value
                        ),
                        "n_distinct_values": len(values),
                        "source_priority_used": (
                            minimum_source_priority
                        ),
                        "key_preference_score_used": (
                            minimum_key_priority
                        ),
                        "source_files": (
                            ";".join(
                                sorted(
                                    set(
                                        part[
                                            "source_file"
                                        ].astype(str)
                                    )
                                )
                            )
                        ),
                    }
                )

        record[
            "n_technical_covariates_observed"
        ] = observed_covariates
        record[
            "n_conflicted_covariates"
        ] = conflicted_covariates
        record[
            "n_metadata_source_files"
        ] = len(
            {
                value
                for value in source_files
                if value
            }
        )

        canonical_rows.append(record)

    canonical = pd.DataFrame(
        canonical_rows
    )

    conflicts = pd.DataFrame(
        conflict_rows,
        columns=[
            "case_id",
            "modality",
            "file_name",
            "covariate_id",
            "canonical_value",
            "n_distinct_values",
            "source_priority_used",
            "key_preference_score_used",
            "source_files",
        ],
    )

    return canonical, conflicts


def build_linkage_audit(
    selected: pd.DataFrame,
    linkage_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    linkage = pd.DataFrame(
        linkage_rows,
        columns=[
            "case_id",
            "modality",
            "file_name",
            "file_id",
            "source_type",
            "source_file",
            "source_key_path",
            "observed_case_id",
            "case_id_matches_selected",
        ],
    )

    rows = []

    for _, selected_row in selected.iterrows():
        if linkage.empty:
            part = linkage
        else:
            part = linkage[
                linkage["file_name"].eq(
                    selected_row["file_name"]
                )
            ]

        observed = sorted(
            set(
                part[
                    "observed_case_id"
                ].astype(str)
            )
        ) if not part.empty else []

        mismatched = [
            case_id
            for case_id in observed
            if case_id != selected_row["case_id"]
        ]

        if mismatched:
            status = "FAIL_MISMATCH"
        elif observed:
            status = "PASS_MATCHED"
        else:
            status = "NO_CASE_ID_EXPOSED"

        rows.append(
            {
                "case_order_index": int(
                    selected_row[
                        "case_order_index"
                    ]
                ),
                "case_id": selected_row["case_id"],
                "modality": selected_row["modality"],
                "file_name": selected_row["file_name"],
                "file_id": selected_row["file_id"],
                "n_linkage_records": int(
                    part.shape[0]
                ),
                "observed_case_ids": (
                    ";".join(observed)
                ),
                "linkage_status": status,
            }
        )

    audit = pd.DataFrame(rows)

    failed = audit[
        audit["linkage_status"].eq(
            "FAIL_MISMATCH"
        )
    ]

    if not failed.empty:
        examples = failed[
            [
                "case_id",
                "modality",
                "file_name",
                "observed_case_ids",
            ]
        ].head(20)

        raise RuntimeError(
            "Technical metadata linkage mismatch detected:\n"
            + examples.to_string(index=False)
        )

    return audit


# ---------------------------------------------------------------------
# Case matrix and covariate eligibility
# ---------------------------------------------------------------------


def build_case_matrix(
    case_ids: list[str],
    canonical: pd.DataFrame,
    covariate_ids: list[str],
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "case_order_index": np.arange(
                len(case_ids),
                dtype=int,
            ),
            "case_id": case_ids,
        }
    )

    for modality in EXPECTED_MODALITIES:
        part = canonical[
            canonical["modality"].eq(
                modality
            )
        ].copy()

        part = (
            part
            .set_index("case_id")
            .loc[case_ids]
        )

        prefix = modality.lower()

        for covariate_id in covariate_ids:
            output[
                f"{prefix}__{covariate_id}"
            ] = (
                part[covariate_id]
                .astype(str)
                .tolist()
            )

    return output


def load_covariate_registry() -> pd.DataFrame:
    registry = pd.read_csv(
        FROZEN_COVARIATE_REGISTRY,
        dtype=str,
    ).fillna("")

    required = {
        "covariate_id",
        "data_type",
        "primary_status",
        "allowed_for_model_selection",
        "allowed_for_factor_flagging",
        "notes",
    }

    missing = required - set(registry.columns)

    if missing:
        raise ValueError(
            "Frozen technical covariate registry is missing: "
            + ", ".join(sorted(missing))
        )

    return registry


def load_eligibility_thresholds(
    technical_contract: dict[str, Any],
) -> dict[str, float | int]:
    rules = technical_contract.get(
        "field_usability_rules",
        {},
    )

    return {
        "maximum_missing_fraction": float(
            rules.get(
                "maximum_missing_fraction",
                DEFAULT_MAXIMUM_MISSING_FRACTION,
            )
        ),
        "maximum_unique_fraction": float(
            rules.get(
                "maximum_unique_fraction",
                DEFAULT_MAXIMUM_UNIQUE_FRACTION,
            )
        ),
        "minimum_level_size": int(
            rules.get(
                "minimum_level_size",
                DEFAULT_MINIMUM_LEVEL_SIZE,
            )
        ),
        "maximum_categorical_levels": int(
            rules.get(
                "maximum_categorical_levels",
                DEFAULT_MAXIMUM_CATEGORICAL_LEVELS,
            )
        ),
    }


def summarize_covariates(
    case_ids: list[str],
    canonical: pd.DataFrame,
    registry: pd.DataFrame,
    thresholds: dict[str, float | int],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    registry_map = (
        registry
        .set_index("covariate_id")
        .to_dict(orient="index")
    )

    summary_rows = []
    level_rows = []
    resolved_rows = []

    candidate_ids = [
        covariate_id
        for covariate_id, record in registry_map.items()
        if clean_text(
            record[
                "allowed_for_factor_flagging"
            ]
        ).lower() == "true"
    ]

    for modality in EXPECTED_MODALITIES:
        part = canonical[
            canonical["modality"].eq(
                modality
            )
        ].copy()

        part = (
            part
            .set_index("case_id")
            .loc[case_ids]
            .reset_index()
        )

        for covariate_id in candidate_ids:
            registry_record = registry_map[
                covariate_id
            ]

            values = (
                part[covariate_id]
                .astype(str)
                .map(normalize_categorical_value)
            )

            observed = values[values.ne("")]
            n_total = len(case_ids)
            n_observed = int(observed.shape[0])
            missing_fraction = (
                1.0
                - (
                    n_observed
                    / n_total
                )
            )

            counts = observed.value_counts(
                dropna=False
            )
            n_unique = int(counts.shape[0])
            unique_fraction = (
                n_unique / n_observed
                if n_observed > 0
                else np.nan
            )
            minimum_level_size = (
                int(counts.min())
                if n_unique > 0
                else 0
            )
            maximum_level_fraction = (
                float(counts.max() / n_observed)
                if n_observed > 0
                else np.nan
            )

            conflict_column = (
                f"{covariate_id}__conflict"
            )
            n_conflicted_cases = int(
                part[conflict_column]
                .astype(bool)
                .sum()
            )

            allowed_flagging = (
                clean_text(
                    registry_record[
                        "allowed_for_factor_flagging"
                    ]
                ).lower()
                == "true"
            )

            source_key_column = (
                f"{covariate_id}__key_path"
            )
            source_key_values = (
                part[source_key_column]
                .astype(str)
                if source_key_column in part.columns
                else pd.Series(
                    ["" for _ in range(part.shape[0])],
                    index=part.index,
                    dtype=str,
                )
            )
            tissue_source_site_used = bool(
                source_key_values
                .str.contains(
                    "tissue_source_site",
                    case=False,
                    regex=False,
                )
                .any()
            )

            ambiguous = (
                covariate_id
                == "specimen_or_portion_type"
                or (
                    covariate_id
                    == "sequencing_or_processing_center"
                    and tissue_source_site_used
                )
                or "ambiguous" in clean_text(
                    registry_record[
                        "primary_status"
                    ]
                ).lower()
            )

            if not allowed_flagging:
                eligibility = "EXCLUDED_BY_FROZEN_REGISTRY"
                reason = "Frozen registry forbids factor flagging."

            elif n_observed == 0:
                eligibility = "NO_DATA"
                reason = "No selected case has an observed value."

            elif n_unique == 1:
                eligibility = "CONSTANT_DESCRIPTIVE_ONLY"
                reason = "The field is constant within this modality."

            elif n_conflicted_cases > 0:
                eligibility = "EXCLUDED_METADATA_CONFLICT"
                reason = (
                    "At least one selected case has conflicting values "
                    "at the highest-priority metadata source."
                )

            elif missing_fraction > float(
                thresholds[
                    "maximum_missing_fraction"
                ]
            ):
                eligibility = "EXCLUDED_MISSINGNESS"
                reason = "Missingness exceeds the frozen threshold."

            elif n_unique > int(
                thresholds[
                    "maximum_categorical_levels"
                ]
            ):
                eligibility = "EXCLUDED_TOO_MANY_LEVELS"
                reason = "Number of categorical levels exceeds the frozen threshold."

            elif (
                np.isfinite(unique_fraction)
                and unique_fraction > float(
                    thresholds[
                        "maximum_unique_fraction"
                    ]
                )
            ):
                eligibility = "EXCLUDED_IDENTIFIER_LIKE_CARDINALITY"
                reason = "Unique-value fraction exceeds the frozen threshold."

            elif minimum_level_size < int(
                thresholds[
                    "minimum_level_size"
                ]
            ):
                eligibility = "EXCLUDED_SPARSE_LEVEL"
                reason = "At least one level is smaller than the frozen minimum."

            elif ambiguous:
                eligibility = "ELIGIBLE_AMBIGUOUS_SPECIMEN_ASSOCIATION"
                reason = (
                    "Usable for flagging, but an association cannot be assumed "
                    "purely technical because the field may contain biological information."
                )

            else:
                eligibility = "ELIGIBLE_FOR_POSTFREEZE_FACTOR_FLAGGING"
                reason = "Passes all frozen usability rules."

            if n_unique == 2:
                future_test = (
                    "standardized_mean_difference_plus_10000_label_permutations"
                )
            elif 3 <= n_unique <= int(
                thresholds[
                    "maximum_categorical_levels"
                ]
            ):
                future_test = (
                    "eta_squared_plus_10000_label_permutations"
                )
            else:
                future_test = "not_testable"

            ordered_values = [
                values.iloc[index]
                for index in range(values.shape[0])
            ]

            summary_rows.append(
                {
                    "modality": modality,
                    "covariate_id": covariate_id,
                    "data_type": registry_record["data_type"],
                    "frozen_primary_status": (
                        registry_record[
                            "primary_status"
                        ]
                    ),
                    "n_cases": n_total,
                    "n_observed": n_observed,
                    "missing_fraction": missing_fraction,
                    "n_unique_levels": n_unique,
                    "unique_fraction": unique_fraction,
                    "minimum_level_size": minimum_level_size,
                    "maximum_level_fraction": maximum_level_fraction,
                    "n_conflicted_cases": n_conflicted_cases,
                    "eligibility_status": eligibility,
                    "eligibility_reason": reason,
                    "future_association_test": future_test,
                    "multiplicity_family": (
                        "technical_confound"
                    ),
                    "allowed_for_model_selection": False,
                    "allowed_for_primary_biological_claim": False,
                    "ordered_case_value_hash": (
                        hash_strings(
                            ordered_values
                        )
                    ),
                }
            )

            for value, count in counts.items():
                level_rows.append(
                    {
                        "modality": modality,
                        "covariate_id": covariate_id,
                        "value": value,
                        "n_cases": int(count),
                        "fraction_of_observed": (
                            float(count / n_observed)
                            if n_observed > 0
                            else np.nan
                        ),
                    }
                )

            resolved_rows.append(
                {
                    "resolved_variable_id": (
                        f"{modality.lower()}__{covariate_id}"
                    ),
                    "modality": modality,
                    "covariate_id": covariate_id,
                    "eligibility_status": eligibility,
                    "future_association_test": future_test,
                    "multiplicity_family": (
                        "technical_confound"
                    ),
                    "technical_flag_status_if_associated": (
                        "stable_ambiguous_specimen_association"
                        if ambiguous
                        else "stable_technical_flag"
                    ),
                    "candidate_column_in_case_matrix": (
                        f"{modality.lower()}__{covariate_id}"
                    ),
                    "allowed_for_model_selection": False,
                    "allowed_for_factor_flagging": (
                        eligibility
                        in {
                            "ELIGIBLE_FOR_POSTFREEZE_FACTOR_FLAGGING",
                            "ELIGIBLE_AMBIGUOUS_SPECIMEN_ASSOCIATION",
                        }
                    ),
                    "notes": registry_record["notes"],
                }
            )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(level_rows),
        pd.DataFrame(resolved_rows),
    )


def build_cross_view_concordance(
    case_ids: list[str],
    canonical: pd.DataFrame,
    covariate_ids: list[str],
    covariate_summary: pd.DataFrame,
) -> pd.DataFrame:
    summary_lookup = {
        (
            row["modality"],
            row["covariate_id"],
        ): row["eligibility_status"]
        for _, row in covariate_summary.iterrows()
    }

    rows = []

    indexed = {
        modality: (
            canonical[
                canonical["modality"].eq(
                    modality
                )
            ]
            .set_index("case_id")
            .loc[case_ids]
        )
        for modality in EXPECTED_MODALITIES
    }

    for covariate_id in covariate_ids:
        patterns = []
        at_least_two = 0
        all_three = 0
        equal_observed = 0
        equal_all_three = 0

        for case_id in case_ids:
            values = {
                modality: normalize_categorical_value(
                    indexed[modality].loc[
                        case_id,
                        covariate_id,
                    ]
                )
                for modality in EXPECTED_MODALITIES
            }

            observed = {
                modality: value
                for modality, value in values.items()
                if value
            }

            if observed:
                patterns.append(
                    "|".join(
                        f"{modality}={values[modality] or '<missing>'}"
                        for modality in EXPECTED_MODALITIES
                    )
                )

            if len(observed) >= 2:
                at_least_two += 1
                if len(
                    {
                        value.casefold()
                        for value in observed.values()
                    }
                ) == 1:
                    equal_observed += 1

            if len(observed) == 3:
                all_three += 1
                if len(
                    {
                        value.casefold()
                        for value in observed.values()
                    }
                ) == 1:
                    equal_all_three += 1

        eligible_views = [
            modality
            for modality in EXPECTED_MODALITIES
            if summary_lookup.get(
                (
                    modality,
                    covariate_id,
                ),
                "",
            )
            in {
                "ELIGIBLE_FOR_POSTFREEZE_FACTOR_FLAGGING",
                "ELIGIBLE_AMBIGUOUS_SPECIMEN_ASSOCIATION",
            }
        ]

        fraction_equal_observed = (
            equal_observed / at_least_two
            if at_least_two > 0
            else np.nan
        )

        potential_shared_batch = bool(
            len(eligible_views) >= 2
            and at_least_two >= 10
            and np.isfinite(
                fraction_equal_observed
            )
            and fraction_equal_observed >= 0.80
            and len(set(patterns)) >= 2
        )

        rows.append(
            {
                "covariate_id": covariate_id,
                "n_cases_with_any_view_observed": len(patterns),
                "n_cases_with_at_least_two_views": at_least_two,
                "n_cases_with_all_three_views": all_three,
                "n_equal_across_observed_views": equal_observed,
                "fraction_equal_across_observed_views": (
                    fraction_equal_observed
                ),
                "n_equal_across_all_three_views": equal_all_three,
                "fraction_equal_across_all_three_views": (
                    equal_all_three / all_three
                    if all_three > 0
                    else np.nan
                ),
                "n_unique_cross_view_patterns": len(
                    set(patterns)
                ),
                "eligible_views": ";".join(
                    eligible_views
                ),
                "potential_cross_view_batch_candidate": (
                    potential_shared_batch
                ),
                "interpretation": (
                    "Potential repeated cross-view technical label; test only after representation freeze."
                    if potential_shared_batch
                    else "No prespecified cross-view batch-candidate rule was met."
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    print("=" * 88)
    print(
        "Paper 5 - TARGET-OS technical metadata preflight"
    )
    print("=" * 88)
    print(
        f"Script version: {SCRIPT_VERSION}"
    )
    print(
        f"Project root: {PROJECT_ROOT}"
    )
    print(
        f"TARGET root: {TARGET_ROOT}"
    )
    print(
        f"Output directory: {OUTPUT_DIR}"
    )
    print()

    if verify_existing_output_lock():
        return

    upstream = validate_upstream_contracts()

    print(
        "05a1 simulation and technical-confound addendum: PASS"
    )
    print(
        "Representation-model results before technical preflight: NO"
    )
    print(
        "Clinical and outcome fields permitted: NO"
    )
    print(
        "Raw timestamps retained as covariates: NO"
    )
    print()

    case_ids = load_case_order()
    selected, inventory = load_selected_files(
        case_ids
    )

    harmonized_case_hash = clean_text(
        upstream[
            "harmonized_lock"
        ].get(
            "case_order",
            {},
        ).get(
            "case_hash",
            "",
        )
    )

    observed_case_hash = hash_strings(
        case_ids
    )

    if observed_case_hash != harmonized_case_hash:
        raise RuntimeError(
            "Current TARGET case order does not match the 04b harmonized lock."
        )

    print(
        f"Locked cases: {len(case_ids)}"
    )
    print(
        f"Selected molecular files: {selected.shape[0]}"
    )
    print(
        selected.groupby(
            "modality"
        )["case_id"]
        .nunique()
        .to_string()
    )

    registry = load_covariate_registry()
    technical_contract = upstream[
        "technical_contract"
    ]
    thresholds = load_eligibility_thresholds(
        technical_contract
    )

    covariate_ids = [
        covariate_id
        for covariate_id in registry[
            "covariate_id"
        ].astype(str)
        if covariate_id
        not in {
            "file_case_sample_aliquot_uuid",
            "clinical_or_outcome_field",
        }
    ]

    state = ExtractionState(
        selected
    )

    extract_selected_file_map_fields(
        selected,
        state,
    )

    sources = discover_metadata_sources()

    print()
    print("-" * 88)
    print(
        "Metadata source audit"
    )
    print("-" * 88)
    print(
        f"Discovered permitted metadata sources: {len(sources)}"
    )

    for index, (path, source_type) in enumerate(
        sources,
        start=1,
    ):
        if path.resolve() in {
            SELECTED_FILE_MAP.resolve(),
        }:
            continue

        if source_type == "metadata_json":
            parse_json_source(
                path,
                state,
            )
        else:
            parse_table_source(
                path,
                source_type,
                state,
            )

        latest = state.source_rows[-1]
        print(
            f" {index:02d}/{len(sources):02d} "
            f"{source_type}: {latest['parse_status']} | "
            f"selected records={latest['n_selected_file_records']} | "
            f"values={latest['n_technical_values_extracted']} | "
            f"{relative_path(path)}"
        )

    source_inventory = pd.DataFrame(
        state.source_rows
    )

    failed_sources = (
        source_inventory[
            source_inventory[
                "parse_status"
            ].eq("FAILED")
        ]
        if not source_inventory.empty
        else pd.DataFrame()
    )

    if not failed_sources.empty:
        print()
        print(
            "Warning: some optional metadata sources could not be parsed."
        )
        print(
            failed_sources[
                [
                    "source_file",
                    "source_type",
                    "error",
                ]
            ].to_string(index=False)
        )

    provenance = pd.DataFrame(
        state.provenance_rows,
        columns=[
            "case_id",
            "modality",
            "file_name",
            "file_id",
            "covariate_id",
            "value",
            "value_comparison",
            "source_type",
            "source_priority",
            "source_file",
            "source_key_path",
            "key_preference_score",
            "date_was_binned",
        ],
    )

    if not provenance.empty:
        provenance = (
            provenance
            .drop_duplicates()
            .sort_values(
                [
                    "case_id",
                    "modality",
                    "covariate_id",
                    "source_priority",
                    "key_preference_score",
                    "value_comparison",
                ]
            )
            .reset_index(drop=True)
        )

    key_inventory = build_key_inventory(
        state
    )

    linkage_audit = build_linkage_audit(
        selected,
        state.linkage_rows,
    )

    canonical, conflicts = resolve_canonical_values(
        selected=selected,
        provenance=provenance,
        covariate_ids=covariate_ids,
    )

    canonical = canonical.merge(
        linkage_audit[
            [
                "case_id",
                "modality",
                "linkage_status",
                "observed_case_ids",
            ]
        ],
        how="left",
        on=[
            "case_id",
            "modality",
        ],
        validate="one_to_one",
    )

    case_matrix = build_case_matrix(
        case_ids=case_ids,
        canonical=canonical,
        covariate_ids=covariate_ids,
    )

    (
        covariate_summary,
        level_counts,
        resolved_registry,
    ) = summarize_covariates(
        case_ids=case_ids,
        canonical=canonical,
        registry=registry,
        thresholds=thresholds,
    )

    cross_view = build_cross_view_concordance(
        case_ids=case_ids,
        canonical=canonical,
        covariate_ids=covariate_ids,
        covariate_summary=covariate_summary,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv_atomic(
        source_inventory,
        OUTPUT_SOURCE_INVENTORY,
    )
    write_csv_atomic(
        key_inventory,
        OUTPUT_KEY_INVENTORY,
    )
    write_csv_atomic(
        provenance,
        OUTPUT_VALUE_PROVENANCE,
    )
    write_csv_atomic(
        linkage_audit,
        OUTPUT_LINKAGE_AUDIT,
    )
    write_csv_atomic(
        canonical,
        OUTPUT_SELECTED_FILE_METADATA,
    )
    write_csv_atomic(
        conflicts,
        OUTPUT_CONFLICTS,
    )
    write_csv_atomic(
        case_matrix,
        OUTPUT_CASE_MATRIX,
    )
    write_csv_atomic(
        covariate_summary,
        OUTPUT_COVARIATE_SUMMARY,
    )
    write_csv_atomic(
        level_counts,
        OUTPUT_LEVEL_COUNTS,
    )
    write_csv_atomic(
        cross_view,
        OUTPUT_CROSS_VIEW,
    )
    write_csv_atomic(
        resolved_registry,
        OUTPUT_RESOLVED_REGISTRY,
    )

    usable_statuses = {
        "ELIGIBLE_FOR_POSTFREEZE_FACTOR_FLAGGING",
        "ELIGIBLE_AMBIGUOUS_SPECIMEN_ASSOCIATION",
    }

    usable = covariate_summary[
        covariate_summary[
            "eligibility_status"
        ].isin(usable_statuses)
    ]

    strict_usable = covariate_summary[
        covariate_summary[
            "eligibility_status"
        ].eq(
            "ELIGIBLE_FOR_POSTFREEZE_FACTOR_FLAGGING"
        )
    ]

    ambiguous_usable = covariate_summary[
        covariate_summary[
            "eligibility_status"
        ].eq(
            "ELIGIBLE_AMBIGUOUS_SPECIMEN_ASSOCIATION"
        )
    ]

    potential_cross_view = cross_view[
        cross_view[
            "potential_cross_view_batch_candidate"
        ].eq(True)
    ]

    print()
    print("-" * 88)
    print(
        "Resolved technical covariate availability"
    )
    print("-" * 88)

    display_columns = [
        "modality",
        "covariate_id",
        "n_observed",
        "n_unique_levels",
        "missing_fraction",
        "minimum_level_size",
        "eligibility_status",
    ]

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_colwidth",
        52,
        "display.width",
        200,
    ):
        print(
            covariate_summary[
                display_columns
            ].to_string(index=False)
        )

    print()
    print(
        f"Strict usable technical variables: {strict_usable.shape[0]}"
    )
    print(
        f"Ambiguous specimen/site variables: {ambiguous_usable.shape[0]}"
    )
    print(
        f"Potential cross-view batch candidates: {potential_cross_view.shape[0]}"
    )
    print(
        f"Metadata conflicts: {conflicts.shape[0]}"
    )
    print(
        "Outcome or clinical values retained: NO"
    )
    print(
        "Raw unique timestamps retained: NO"
    )
    print(
        "Identifiers eligible as technical covariates: NO"
    )

    input_paths = [
        CASE_ORDER_FILE,
        SELECTED_FILE_MAP,
        GDC_FILE_INVENTORY,
        HARMONIZED_LOCK,
        ADDENDUM_FILE,
        ADDENDUM_LOCK,
        FROZEN_TECHNICAL_CONTRACT,
        FROZEN_COVARIATE_REGISTRY,
        *[
            path
            for path, _ in sources
        ],
    ]

    unique_input_paths = sorted(
        set(
            path.resolve()
            for path in input_paths
            if path.exists()
        ),
        key=lambda path: str(path),
    )

    upstream_inputs = {
        label: {
            "path": relative_path(path),
            "sha256": sha256_file(path),
            "size_bytes": int(
                path.stat().st_size
            ),
        }
        for label, path in {
            "case_order": CASE_ORDER_FILE,
            "selected_file_map": SELECTED_FILE_MAP,
            "gdc_file_inventory": GDC_FILE_INVENTORY,
            "harmonized_lock": HARMONIZED_LOCK,
            "05a1_addendum": ADDENDUM_FILE,
            "05a1_lock": ADDENDUM_LOCK,
            "frozen_technical_contract": FROZEN_TECHNICAL_CONTRACT,
            "frozen_covariate_registry": FROZEN_COVARIATE_REGISTRY,
        }.items()
    }

    manifest = {
        "script_version": SCRIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": "FROZEN_BEFORE_REPRESENTATION_MODEL_FITTING",
        "project_root": str(PROJECT_ROOT),
        "scientific_state": {
            "outcome_loaded": False,
            "clinical_values_retained": False,
            "biological_labels_used": False,
            "representation_model_fitting_performed": False,
            "technical_covariates_used_for_model_selection": False,
            "raw_dates_retained": False,
            "identifiers_used_as_covariates": False,
        },
        "case_cohort": {
            "n_cases": len(case_ids),
            "case_order_hash": observed_case_hash,
            "n_selected_files": int(
                selected.shape[0]
            ),
            "selected_files_by_modality": {
                modality: int(
                    selected[
                        selected["modality"].eq(
                            modality
                        )
                    ]["case_id"].nunique()
                )
                for modality in EXPECTED_MODALITIES
            },
        },
        "source_audit": {
            "n_permitted_sources_discovered": len(sources),
            "n_sources_parsed": int(
                source_inventory[
                    "parse_status"
                ].eq("PASS").sum()
            ) if not source_inventory.empty else 0,
            "n_sources_failed": int(
                source_inventory[
                    "parse_status"
                ].eq("FAILED").sum()
            ) if not source_inventory.empty else 0,
            "n_technical_value_records": int(
                provenance.shape[0]
            ),
            "n_metadata_key_records": int(
                key_inventory.shape[0]
            ),
        },
        "eligibility_rules": thresholds,
        "resolved_covariates": {
            "n_view_specific_variables": int(
                covariate_summary.shape[0]
            ),
            "n_strict_usable": int(
                strict_usable.shape[0]
            ),
            "n_ambiguous_usable": int(
                ambiguous_usable.shape[0]
            ),
            "n_potential_cross_view_batch_candidates": int(
                potential_cross_view.shape[0]
            ),
            "n_metadata_conflicts": int(
                conflicts.shape[0]
            ),
            "technical_status_if_no_strict_usable_fields": (
                "stable_technical_status_unknown"
            ),
        },
        "future_factor_audit": {
            "execution_timing": (
                "after representation and stability freeze, before primary biological interpretation"
            ),
            "multiplicity_family": (
                "technical_confound"
            ),
            "model_selection_use": False,
            "automatic_batch_residualization": False,
            "factor_status_taxonomy_source": (
                relative_path(
                    FROZEN_TECHNICAL_CONTRACT
                )
            ),
        },
        "guardrails": [
            "Only selected molecular file metadata were eligible for extraction.",
            "Clinical and outcome branches were skipped before values were inspected.",
            "Identifiers were used only for linkage and integrity checks.",
            "Raw dates were discarded after calendar-quarter derivation.",
            "High-cardinality or sparse fields were excluded using frozen rules.",
            "Technical covariates cannot select the representation model family.",
            "A stable factor is not assumed biological solely because it is reproducible.",
            "No automatic batch correction was performed.",
        ],
        "metadata_input_files": [
            {
                "path": relative_path(path),
                "sha256": sha256_file(path),
                "size_bytes": int(
                    path.stat().st_size
                ),
            }
            for path in unique_input_paths
        ],
        "upstream_inputs": upstream_inputs,
    }

    generated_paths = [
        OUTPUT_SOURCE_INVENTORY,
        OUTPUT_KEY_INVENTORY,
        OUTPUT_VALUE_PROVENANCE,
        OUTPUT_LINKAGE_AUDIT,
        OUTPUT_SELECTED_FILE_METADATA,
        OUTPUT_CONFLICTS,
        OUTPUT_CASE_MATRIX,
        OUTPUT_COVARIATE_SUMMARY,
        OUTPUT_LEVEL_COUNTS,
        OUTPUT_CROSS_VIEW,
        OUTPUT_RESOLVED_REGISTRY,
    ]

    manifest[
        "outputs"
    ] = {
        path.stem: relative_path(path)
        for path in generated_paths
    }

    write_json_atomic(
        manifest,
        OUTPUT_MANIFEST,
    )

    generated_paths.append(
        OUTPUT_MANIFEST
    )

    readme_lines = [
        "TARGET-OS technical metadata preflight",
        "=" * 72,
        "",
        f"Script version: {SCRIPT_VERSION}",
        f"Created UTC: {manifest['created_at_utc']}",
        "",
        "Purpose",
        "-------",
        (
            "Inventory sample-level technical covariates for the 78 locked "
            "TARGET-OS RNA/methylation/CNV cases before representation-model fitting."
        ),
        "",
        "Scientific state",
        "----------------",
        "outcome_loaded: false",
        "clinical values retained: false",
        "representation-model fitting: false",
        "technical covariates used for model selection: false",
        "raw timestamps retained: false",
        "identifiers used as covariates: false",
        "",
        "Summary",
        "-------",
        f"Locked cases: {len(case_ids)}",
        f"Selected molecular files: {selected.shape[0]}",
        f"Strict usable technical variables: {strict_usable.shape[0]}",
        f"Ambiguous usable variables: {ambiguous_usable.shape[0]}",
        f"Potential cross-view batch candidates: {potential_cross_view.shape[0]}",
        f"Metadata conflicts: {conflicts.shape[0]}",
        "",
        "Interpretation",
        "--------------",
        (
            "Usable technical covariates are reserved for a separate post-freeze "
            "technical-confound association family. They cannot be used to choose "
            "the representation model."
        ),
        (
            "If no strict usable covariate is available, a stable factor receives "
            "technical status unknown rather than being assumed biological."
        ),
        "",
        "Outputs",
        "-------",
        *[
            relative_path(path)
            for path in generated_paths
        ],
        relative_path(OUTPUT_README),
        relative_path(OUTPUT_LOCK),
        "",
    ]

    OUTPUT_README.write_text(
        "\n".join(readme_lines),
        encoding="utf-8",
    )

    generated_paths.append(
        OUTPUT_README
    )

    generated_files = {
        relative_path(path): {
            "sha256": sha256_file(path),
            "size_bytes": int(
                path.stat().st_size
            ),
        }
        for path in sorted(
            generated_paths,
            key=lambda path: str(path),
        )
    }

    lock = {
        "script_version": SCRIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": (
            "FROZEN_BEFORE_REPRESENTATION_MODEL_FITTING"
        ),
        "manifest_path": relative_path(
            OUTPUT_MANIFEST
        ),
        "manifest_sha256": sha256_file(
            OUTPUT_MANIFEST
        ),
        "upstream_inputs": upstream_inputs,
        "generated_files": generated_files,
        "rewrite_policy": (
            "This zero-argument script never rewrites a frozen 04d preflight. "
            "Any future scientifically justified change requires a new versioned script "
            "and preservation of this lock."
        ),
    }

    write_json_atomic(
        lock,
        OUTPUT_LOCK,
    )

    print()
    print("=" * 88)
    print(
        "TARGET-OS technical metadata preflight: PASS"
    )
    print("=" * 88)
    print(
        f"Strict usable technical variables: {strict_usable.shape[0]}"
    )
    print(
        f"Ambiguous usable variables: {ambiguous_usable.shape[0]}"
    )
    print(
        f"Potential cross-view batch candidates: {potential_cross_view.shape[0]}"
    )
    print(
        "Technical labels used for model selection: NO"
    )
    print(
        "Representation-model fitting performed: NO"
    )
    print(
        f"Manifest: {relative_path(OUTPUT_MANIFEST)}"
    )
    print(
        f"Lock: {relative_path(OUTPUT_LOCK)}"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
