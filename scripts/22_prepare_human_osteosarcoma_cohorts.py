from __future__ import annotations

from pathlib import Path
import gzip
import hashlib
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.decomposition import PCA

SCRIPT_VERSION = "22-human-cohort-preparation-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"

TARGET_RAW_DIR = DATA_RAW_DIR / "human_TARGET_OS_GDC"
GSE21257_RAW_DIR = DATA_RAW_DIR / "human_GSE21257"
HUMAN_PROCESSED_DIR = PROCESSED_DIR / "human_validation"

for directory in [
    TARGET_RAW_DIR,
    GSE21257_RAW_DIR,
    HUMAN_PROCESSED_DIR,
    RESULTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Frozen canine inputs created by script 21.
FREEZE_JSON_FILE = RESULTS_DIR / "GSE238110_frozen_transfer_program_freeze.json"
FROZEN_MANIFEST_FILE = RESULTS_DIR / "GSE238110_frozen_canine_transfer_program_manifest.csv"
STRICT_WEIGHTS_FILE = RESULTS_DIR / "GSE238110_frozen_transfer_gene_weights_strict.csv"
BROAD_WEIGHTS_FILE = RESULTS_DIR / "GSE238110_frozen_transfer_gene_weights_broad.csv"
SCORING_SPEC_FILE = RESULTS_DIR / "GSE238110_frozen_transfer_scoring_specification.csv"
ORTHOLOG_QC_FILE = RESULTS_DIR / "GSE238110_RNA_master_candidate_evidence_table_with_ortholog_qc.csv"
PROLIFERATION_GENE_FILE = RESULTS_DIR / "GSE238110_meta_proliferation_gene_set.csv"

# Public data sources.
GDC_API_BASE = "https://api.gdc.cancer.gov"
GDC_PROJECT_ID = "TARGET-OS"
GSE_ACCESSION = "GSE21257"

REQUEST_TIMEOUT_SEC = 180
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
RANDOM_SEED = 42

PRIMARY_MODULES = ["M34", "M11", "M24", "M40"]
SECONDARY_MODULES = ["M28", "M38", "M25", "M17"]

MIN_SCORE_GENES = 3
MIN_SCORE_FRACTION = 0.50
MIN_PROLIFERATION_GENES = 20
MIN_PROLIFERATION_FRACTION = 0.30

PROLIFERATION_ANCHOR_SYMBOLS = [
    "PCNA",
    "MKI67",
    "TOP2A",
    "BIRC5",
    "UBE2C",
    "UBE2S",
    "AURKA",
    "AURKB",
    "CDC20",
    "CDC6",
    "CDK1",
    "CCNA2",
    "CCNB1",
    "CCNB2",
    "MCM2",
    "MCM4",
    "MCM5",
    "MCM10",
    "TYMS",
    "RRM2",
    "TK1",
    "PLK1",
    "PLK4",
    "CENPA",
    "CENPE",
    "CENPF",
    "CENPK",
    "CENPV",
    "KIF11",
    "KIF15",
    "KIF18B",
    "KIF23",
    "MELK",
    "MYBL2",
    "BUB1",
    "BUB1B",
    "DLGAP5",
    "SPAG5",
    "STMN1",
]

OUTPUT_TARGET_EXPRESSION = HUMAN_PROCESSED_DIR / "TARGET_OS_expression_log2_gene_symbol.csv"
OUTPUT_TARGET_CLINICAL = HUMAN_PROCESSED_DIR / "TARGET_OS_clinical_standardized.csv"
OUTPUT_TARGET_SAMPLE_MAP = HUMAN_PROCESSED_DIR / "TARGET_OS_expression_sample_map.csv"
OUTPUT_TARGET_SCORES = HUMAN_PROCESSED_DIR / "TARGET_OS_frozen_transfer_scores.csv"
OUTPUT_TARGET_COVERAGE = RESULTS_DIR / "TARGET_OS_frozen_transfer_score_coverage.csv"
OUTPUT_TARGET_GDC_FILE_MANIFEST = RESULTS_DIR / "TARGET_OS_GDC_expression_file_manifest.csv"
OUTPUT_TARGET_GDC_CASES_RAW = TARGET_RAW_DIR / "TARGET_OS_GDC_cases_raw.json"

OUTPUT_GSE_EXPRESSION = HUMAN_PROCESSED_DIR / "GSE21257_expression_gene_symbol.csv"
OUTPUT_GSE_CLINICAL = HUMAN_PROCESSED_DIR / "GSE21257_clinical_standardized.csv"
OUTPUT_GSE_PROBE_MAP = RESULTS_DIR / "GSE21257_probe_to_gene_symbol_selected.csv"
OUTPUT_GSE_SCORES = HUMAN_PROCESSED_DIR / "GSE21257_frozen_transfer_scores.csv"
OUTPUT_GSE_COVERAGE = RESULTS_DIR / "GSE21257_frozen_transfer_score_coverage.csv"
OUTPUT_GSE_PHENOTYPE_RAW = RESULTS_DIR / "GSE21257_GEO_phenotype_raw.csv"

OUTPUT_PROLIFERATION_MAPPING = RESULTS_DIR / "frozen_strict_human_proliferation_mapping.csv"
OUTPUT_PREPARATION_SUMMARY = RESULTS_DIR / "human_validation_cohort_preparation_summary.csv"
OUTPUT_PREPARATION_MANIFEST = RESULTS_DIR / "human_validation_cohort_preparation_manifest.json"
OUTPUT_README = RESULTS_DIR / "human_validation_cohort_preparation_README.txt"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_required_csv(path: Path, index_col: int | str | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    print(f"Loaded: {path}")
    return pd.read_csv(path, index_col=index_col)


def verify_frozen_inputs() -> dict[str, Any]:
    if not FREEZE_JSON_FILE.exists():
        raise FileNotFoundError(
            f"Frozen-program manifest is missing: {FREEZE_JSON_FILE}. "
            "Run the corrected script 21 first."
        )

    freeze = json.loads(FREEZE_JSON_FILE.read_text(encoding="utf-8"))
    files = freeze.get("files", {})
    required = [
        FROZEN_MANIFEST_FILE,
        STRICT_WEIGHTS_FILE,
        BROAD_WEIGHTS_FILE,
        SCORING_SPEC_FILE,
    ]

    print("")
    print("Frozen input integrity check:")
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Frozen input is missing: {path}")
        expected = files.get(path.name, {}).get("sha256")
        observed = sha256_file(path)
        if expected and expected != observed:
            raise RuntimeError(
                f"Frozen input hash mismatch for {path.name}. "
                "Do not continue after modifying a frozen file."
            )
        status = "verified" if expected else "hash_not_recorded_but_file_present"
        print(f"  {path.name}: {status}")

    strict = pd.read_csv(STRICT_WEIGHTS_FILE)
    if strict.empty:
        raise RuntimeError("The strict frozen weight table is empty.")

    primary_counts = (
        strict[strict["module_label"].isin(PRIMARY_MODULES)]
        .groupby("module_label")["human_gene_symbol"]
        .nunique()
    )
    missing_primary = [module for module in PRIMARY_MODULES if primary_counts.get(module, 0) < 3]
    if missing_primary:
        raise RuntimeError(
            "Primary frozen modules have insufficient strict genes: "
            + ", ".join(missing_primary)
        )

    print("  Primary strict gene counts:")
    print(primary_counts.to_string())
    return freeze


def request_json_post(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{GDC_API_BASE}/{endpoint.lstrip('/')}"
    response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SEC)
    response.raise_for_status()
    return response.json()


def download_url(url: str, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT_SEC) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if chunk:
                    handle.write(chunk)
    temporary.replace(destination)
    return destination


def nested_first(value: Any, default: Any = "") -> Any:
    if isinstance(value, list):
        return value[0] if value else default
    return value if value is not None else default


def first_nonempty(values: list[Any], default: Any = "") -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "not reported", "--"}:
            return value
    return default


def numeric_or_nan(value: Any) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (list, tuple)):
        values = [numeric_or_nan(item) for item in value]
        values = [item for item in values if np.isfinite(item)]
        return float(max(values)) if values else np.nan
    try:
        number = float(value)
        return number if np.isfinite(number) else np.nan
    except Exception:
        return np.nan


def extract_gdc_file_metadata(hit: dict[str, Any]) -> dict[str, Any]:
    cases = hit.get("cases", []) or []
    case = cases[0] if cases else {}
    samples = case.get("samples", []) or []
    sample = samples[0] if samples else {}
    sample_types = sorted(
        {
            str(item.get("sample_type", "")).strip()
            for item in samples
            if str(item.get("sample_type", "")).strip()
        }
    )
    sample_submitters = sorted(
        {
            str(item.get("submitter_id", "")).strip()
            for item in samples
            if str(item.get("submitter_id", "")).strip()
        }
    )
    return {
        "file_id": hit.get("file_id", ""),
        "file_name": hit.get("file_name", ""),
        "md5sum": hit.get("md5sum", ""),
        "file_size": hit.get("file_size", np.nan),
        "created_datetime": hit.get("created_datetime", ""),
        "updated_datetime": hit.get("updated_datetime", ""),
        "case_id": case.get("case_id", ""),
        "case_submitter_id": case.get("submitter_id", ""),
        "sample_id": sample.get("sample_id", ""),
        "sample_submitter_id": ";".join(sample_submitters),
        "sample_types": ";".join(sample_types),
        "is_primary_tumor": any(item.lower() == "primary tumor" for item in sample_types),
    }


def query_target_os_expression_files() -> pd.DataFrame:
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": [GDC_PROJECT_ID],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "data_category",
                    "value": ["Transcriptome Profiling"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "data_type",
                    "value": ["Gene Expression Quantification"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "analysis.workflow_type",
                    "value": ["STAR - Counts"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "access",
                    "value": ["open"],
                },
            },
        ],
    }
    payload = {
        "filters": filters,
        "format": "JSON",
        "fields": ",".join(
            [
                "file_id",
                "file_name",
                "md5sum",
                "file_size",
                "created_datetime",
                "updated_datetime",
                "cases.case_id",
                "cases.submitter_id",
                "cases.samples.sample_id",
                "cases.samples.submitter_id",
                "cases.samples.sample_type",
            ]
        ),
        "expand": "cases,cases.samples",
        "size": 2000,
    }
    response = request_json_post("files", payload)
    hits = response.get("data", {}).get("hits", [])
    if not hits:
        raise RuntimeError(
            "The GDC API returned no open TARGET-OS STAR-Counts files. "
            "Check the current GDC workflow names before changing the analysis design."
        )

    manifest = pd.DataFrame([extract_gdc_file_metadata(hit) for hit in hits])
    manifest["updated_datetime_sort"] = pd.to_datetime(
        manifest["updated_datetime"], errors="coerce", utc=True
    )
    manifest = manifest.sort_values(
        ["case_submitter_id", "is_primary_tumor", "updated_datetime_sort", "file_id"],
        ascending=[True, False, False, True],
    )
    manifest = manifest.drop_duplicates("case_submitter_id", keep="first")
    manifest = manifest.drop(columns=["updated_datetime_sort"])
    if manifest["case_submitter_id"].eq("").any():
        raise RuntimeError("At least one GDC expression file lacks a case submitter ID.")
    manifest.to_csv(OUTPUT_TARGET_GDC_FILE_MANIFEST, index=False)
    return manifest


def verify_md5(path: Path, expected: str) -> bool:
    if not expected:
        return True
    md5 = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest().lower() == expected.lower()


def download_target_expression_files(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    print("")
    print("Downloading cached/open TARGET-OS expression files:")
    for index, row in manifest.reset_index(drop=True).iterrows():
        file_id = str(row["file_id"])
        file_name = str(row["file_name"])
        destination = TARGET_RAW_DIR / file_id / file_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or destination.stat().st_size == 0:
            print(f"  {index + 1}/{manifest.shape[0]}: {file_name}")
            download_url(f"{GDC_API_BASE}/data/{file_id}", destination)
        if not verify_md5(destination, str(row.get("md5sum", ""))):
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"MD5 verification failed for {file_name}")
        out = row.to_dict()
        out["local_path"] = str(destination)
        rows.append(out)
    return pd.DataFrame(rows)


def read_gdc_star_counts(path: Path) -> tuple[pd.Series, str]:
    table = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    if table.empty:
        raise ValueError(f"Empty GDC expression file: {path}")

    symbol_col = next(
        (column for column in ["gene_name", "gene_symbol", "symbol"] if column in table.columns),
        None,
    )
    if symbol_col is None:
        raise ValueError(f"No gene symbol column found in {path.name}: {list(table.columns)}")

    if "tpm_unstranded" in table.columns:
        value_col = "tpm_unstranded"
        value_type = "TPM"
    elif "unstranded" in table.columns:
        value_col = "unstranded"
        value_type = "counts"
    else:
        numeric_candidates = [
            column
            for column in table.columns
            if column not in {"gene_id", symbol_col, "gene_type"}
            and pd.api.types.is_numeric_dtype(table[column])
        ]
        if not numeric_candidates:
            raise ValueError(f"No usable expression column found in {path.name}")
        value_col = numeric_candidates[0]
        value_type = "counts"

    use = table[[symbol_col, value_col]].copy()
    use[symbol_col] = use[symbol_col].astype(str).str.strip().str.upper()
    use[value_col] = pd.to_numeric(use[value_col], errors="coerce")
    use = use[
        use[symbol_col].ne("")
        & ~use[symbol_col].str.startswith("N_")
        & use[value_col].notna()
    ]
    series = use.groupby(symbol_col, sort=False)[value_col].sum()
    series.name = path.stem
    return series, value_type


def build_target_expression(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    series_list: list[pd.Series] = []
    sample_rows = []
    value_types = set()

    print("")
    print("Parsing TARGET-OS gene-expression files:")
    for index, row in manifest.reset_index(drop=True).iterrows():
        path = Path(str(row["local_path"]))
        series, value_type = read_gdc_star_counts(path)
        case_id = str(row["case_submitter_id"])
        series.name = case_id
        series_list.append(series)
        value_types.add(value_type)
        sample_rows.append(
            {
                "case_submitter_id": case_id,
                "case_id": row.get("case_id", ""),
                "sample_submitter_id": row.get("sample_submitter_id", ""),
                "sample_types": row.get("sample_types", ""),
                "file_id": row.get("file_id", ""),
                "file_name": row.get("file_name", ""),
                "value_type": value_type,
            }
        )
        if (index + 1) % 20 == 0 or index + 1 == manifest.shape[0]:
            print(f"  Parsed {index + 1}/{manifest.shape[0]}")

    if len(value_types) != 1:
        raise RuntimeError(f"Mixed GDC expression value types were detected: {sorted(value_types)}")

    matrix = pd.concat(series_list, axis=1).T
    matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    value_type = next(iter(value_types))
    if value_type == "TPM":
        expression = np.log2(matrix + 1.0)
    else:
        library_size = matrix.sum(axis=1).replace(0, np.nan)
        cpm = matrix.div(library_size, axis=0) * 1_000_000.0
        expression = np.log2(cpm.fillna(0.0) + 1.0)

    expression.index.name = "case_submitter_id"
    expression = expression.loc[:, expression.var(axis=0) > 0]
    sample_map = pd.DataFrame(sample_rows).drop_duplicates("case_submitter_id")
    return expression, sample_map


def query_target_cases() -> list[dict[str, Any]]:
    filters = {
        "op": "in",
        "content": {
            "field": "project.project_id",
            "value": [GDC_PROJECT_ID],
        },
    }
    payload = {
        "filters": filters,
        "format": "JSON",
        "fields": "case_id,submitter_id,primary_site,disease_type",
        "expand": "demographic,diagnoses,follow_ups",
        "size": 2000,
    }
    response = request_json_post("cases", payload)
    hits = response.get("data", {}).get("hits", [])
    if not hits:
        raise RuntimeError("The GDC API returned no TARGET-OS cases.")
    OUTPUT_TARGET_GDC_CASES_RAW.write_text(json.dumps(hits, indent=2), encoding="utf-8")
    return hits


def standardize_target_clinical(cases: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for case in cases:
        demographic = case.get("demographic") or {}
        if isinstance(demographic, list):
            demographic = demographic[0] if demographic else {}
        diagnoses = case.get("diagnoses") or []
        follow_ups = case.get("follow_ups") or []

        vital_values = [demographic.get("vital_status")]
        vital_values.extend(item.get("vital_status") for item in follow_ups if isinstance(item, dict))
        vital_status = str(first_nonempty(vital_values, default="Unknown"))
        os_event = 1.0 if vital_status.lower() == "dead" else 0.0 if vital_status.lower() == "alive" else np.nan

        days_to_death = numeric_or_nan(demographic.get("days_to_death"))
        follow_up_times = [
            numeric_or_nan(item.get("days_to_follow_up"))
            for item in follow_ups
            if isinstance(item, dict)
        ]
        diagnosis_follow_up_times = []
        diagnosis_age_values = []
        diagnosis_metastasis_values = []
        for diagnosis in diagnoses:
            if not isinstance(diagnosis, dict):
                continue
            diagnosis_follow_up_times.extend(
                [
                    numeric_or_nan(diagnosis.get("days_to_last_follow_up")),
                    numeric_or_nan(diagnosis.get("days_to_last_known_disease_status")),
                ]
            )
            diagnosis_age_values.append(numeric_or_nan(diagnosis.get("age_at_diagnosis")))
            diagnosis_metastasis_values.extend(
                [
                    diagnosis.get("metastasis_at_diagnosis"),
                    diagnosis.get("ajcc_clinical_m"),
                    diagnosis.get("ajcc_pathologic_m"),
                ]
            )

        candidate_follow_up = [
            value
            for value in follow_up_times + diagnosis_follow_up_times
            if np.isfinite(value)
        ]
        days_to_last_follow_up = max(candidate_follow_up) if candidate_follow_up else np.nan
        if os_event == 1 and np.isfinite(days_to_death):
            os_time = days_to_death
        else:
            os_time = days_to_last_follow_up

        age_at_diagnosis_days = next(
            (value for value in diagnosis_age_values if np.isfinite(value)),
            np.nan,
        )
        age_at_diagnosis_years = age_at_diagnosis_days / 365.25 if np.isfinite(age_at_diagnosis_days) else np.nan

        metastasis_text = ";".join(
            sorted(
                {
                    str(value).strip()
                    for value in diagnosis_metastasis_values
                    if value is not None and str(value).strip()
                }
            )
        )

        rows.append(
            {
                "case_submitter_id": case.get("submitter_id", ""),
                "case_id": case.get("case_id", ""),
                "primary_site": case.get("primary_site", ""),
                "disease_type": case.get("disease_type", ""),
                "sex": demographic.get("gender", ""),
                "race": demographic.get("race", ""),
                "ethnicity": demographic.get("ethnicity", ""),
                "vital_status": vital_status,
                "os_time_days": os_time,
                "os_event": os_event,
                "days_to_death": days_to_death,
                "days_to_last_follow_up": days_to_last_follow_up,
                "age_at_diagnosis_days": age_at_diagnosis_days,
                "age_at_diagnosis_years": age_at_diagnosis_years,
                "metastasis_fields_raw": metastasis_text,
                "endpoint_preparation_note": "OS prepared from GDC vital status and death/follow-up fields; no outcome model fitted in script 22",
            }
        )

    clinical = pd.DataFrame(rows)
    clinical = clinical[clinical["case_submitter_id"].astype(str).str.len() > 0].copy()
    clinical = clinical.drop_duplicates("case_submitter_id", keep="first")
    clinical = clinical.set_index("case_submitter_id")
    return clinical


def import_geoparse():
    try:
        import GEOparse  # type: ignore

        return GEOparse
    except ImportError as exc:
        raise ImportError(
            "GEOparse is required for GSE21257 preparation. Install it inside the project venv with: "
            "python -m pip install GEOparse"
        ) from exc


def detect_platform_columns(platform: pd.DataFrame) -> tuple[str, str]:
    probe_candidates = ["ID", "ID_REF", "ProbeID", "PROBE_ID", "SPOT_ID"]
    probe_col = next((column for column in probe_candidates if column in platform.columns), None)
    if probe_col is None:
        probe_col = platform.columns[0]

    exact_symbol_candidates = [
        "Gene Symbol",
        "GENE_SYMBOL",
        "Gene symbol",
        "Symbol",
        "SYMBOL",
        "gene_assignment",
    ]
    symbol_col = next((column for column in exact_symbol_candidates if column in platform.columns), None)
    if symbol_col is None:
        candidates = [
            column
            for column in platform.columns
            if "symbol" in str(column).lower()
        ]
        symbol_col = candidates[0] if candidates else None
    if symbol_col is None:
        raise ValueError(
            "Could not detect a gene-symbol column in the GSE21257 platform annotation. "
            f"Columns: {list(platform.columns)}"
        )
    return str(probe_col), str(symbol_col)


def normalize_unambiguous_gene_symbol(value: Any) -> str:
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NA", "---", "NONE", "NULL"}:
        return ""
    parts = [
        item.strip()
        for item in re.split(r"\s*///\s*|\s*//\s*|\s*;\s*|\s*,\s*", text)
        if item.strip() and item.strip() not in {"---", "NA"}
    ]
    unique = list(dict.fromkeys(parts))
    if len(unique) != 1:
        return ""
    symbol = unique[0]
    if " " in symbol:
        return ""
    return symbol


def classify_gse21257_metastasis(text: str) -> float:
    value = text.lower()
    negative_patterns = [
        r"did not develop metast",
        r"without metast",
        r"non[- ]?metast",
        r"no metast",
        r"metastasis\s*[:=]\s*(no|negative|0)",
    ]
    positive_patterns = [
        r"developed metast",
        r"with metast",
        r"metastases? present",
        r"metastases within 5",
        r"metastatic",
        r"metastasis\s*[:=]\s*(yes|positive|1)",
        r"\bmetastas(?:is|es)\b",
    ]
    if any(re.search(pattern, value) for pattern in negative_patterns):
        return 0.0
    if any(re.search(pattern, value) for pattern in positive_patterns):
        return 1.0
    return np.nan


def parse_gse21257_survival(text: str) -> tuple[float, float]:
    value = text.lower()
    if "deceased" in value or "died" in value:
        event = 1.0
    elif "alive" in value:
        event = 0.0
    else:
        event = np.nan

    patterns = [
        r"(?:deceased|died)\s+(?:after|at)\s+([0-9]+(?:\.[0-9]+)?)\s+months",
        r"alive\s+(?:after|at)\s+([0-9]+(?:\.[0-9]+)?)\s+months",
        r"(?:after|at)\s+([0-9]+(?:\.[0-9]+)?)\s+months",
    ]
    time_months = np.nan
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            time_months = float(match.group(1))
            break
    return time_months, event


def parse_gse21257_age_months(text: str) -> float:
    match = re.search(r"age\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*months", text.lower())
    return float(match.group(1)) if match else np.nan


def extract_gse_characteristic(text: str, label: str) -> str:
    pattern = rf"{re.escape(label.lower())}\s*:\s*([^|]+)"
    match = re.search(pattern, text.lower())
    if not match:
        return ""
    return match.group(1).strip()


def prepare_gse21257() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    GEOparse = import_geoparse()
    print("")
    print("Downloading/loading GSE21257 from NCBI GEO:")
    gse = GEOparse.get_GEO(geo=GSE_ACCESSION, destdir=str(GSE21257_RAW_DIR), silent=False)

    expression_probe_by_sample = gse.pivot_samples("VALUE")
    if expression_probe_by_sample.empty:
        raise RuntimeError("GSE21257 expression matrix is empty.")
    expression_probe_by_sample.index = expression_probe_by_sample.index.astype(str)

    phenotype = gse.phenotype_data.copy()
    phenotype.index = phenotype.index.astype(str)
    phenotype.to_csv(OUTPUT_GSE_PHENOTYPE_RAW)

    if not gse.gpls:
        raise RuntimeError("GSE21257 platform annotation was not loaded by GEOparse.")
    gpl_name = sorted(gse.gpls.keys())[0]
    platform = gse.gpls[gpl_name].table.copy()
    probe_col, symbol_col = detect_platform_columns(platform)

    annotation = platform[[probe_col, symbol_col]].copy()
    annotation[probe_col] = annotation[probe_col].astype(str)
    annotation["gene_symbol"] = annotation[symbol_col].map(normalize_unambiguous_gene_symbol)
    annotation = annotation[annotation["gene_symbol"].ne("")].copy()
    annotation = annotation.drop_duplicates(probe_col, keep="first")

    common_probes = expression_probe_by_sample.index.intersection(annotation[probe_col])
    if len(common_probes) == 0:
        raise RuntimeError("No GSE21257 probes matched the platform annotation.")

    expr = expression_probe_by_sample.loc[common_probes].apply(pd.to_numeric, errors="coerce")
    ann = annotation.set_index(probe_col).loc[common_probes]
    probe_variance = expr.var(axis=1)
    probe_map = pd.DataFrame(
        {
            "probe_id": common_probes,
            "gene_symbol": ann["gene_symbol"].values,
            "probe_variance": probe_variance.loc[common_probes].values,
        }
    )
    probe_map = probe_map.sort_values(
        ["gene_symbol", "probe_variance", "probe_id"],
        ascending=[True, False, True],
    )
    selected_map = probe_map.drop_duplicates("gene_symbol", keep="first")
    selected_map.to_csv(OUTPUT_GSE_PROBE_MAP, index=False)

    selected_expr = expr.loc[selected_map["probe_id"].tolist()].copy()
    selected_expr.index = selected_map.set_index("probe_id").loc[selected_expr.index, "gene_symbol"]
    expression = selected_expr.T
    expression.index.name = "geo_sample_id"
    expression = expression.loc[:, ~expression.columns.duplicated()].copy()
    expression = expression.loc[:, expression.var(axis=0) > 0]

    phenotype_text = phenotype.astype(str).agg(" | ".join, axis=1)
    clinical = phenotype.copy()
    clinical["metadata_text_combined"] = phenotype_text
    clinical["metastasis_within_5y"] = phenotype_text.map(classify_gse21257_metastasis)
    survival_parsed = phenotype_text.map(parse_gse21257_survival)
    clinical["os_time_months"] = survival_parsed.map(lambda value: value[0])
    clinical["os_time_days_approx"] = clinical["os_time_months"] * 30.4375
    clinical["os_event"] = survival_parsed.map(lambda value: value[1])
    clinical["age_months"] = phenotype_text.map(parse_gse21257_age_months)
    clinical["age_years"] = clinical["age_months"] / 12.0
    clinical["group_parsed"] = phenotype_text.map(lambda value: extract_gse_characteristic(value, "group"))
    clinical["status_parsed"] = phenotype_text.map(lambda value: extract_gse_characteristic(value, "status"))
    clinical["metastasis_endpoint_note"] = (
        "Pre-chemotherapy biopsy group parsed from GEO sample metadata; "
        "no classifier or association test fitted in script 22"
    )
    clinical["survival_endpoint_note"] = (
        "OS time/event parsed from GEO status text for sensitivity analysis; "
        "no survival model fitted in script 22"
    )
    clinical.index.name = "geo_sample_id"

    common_samples = expression.index.intersection(clinical.index)
    expression = expression.loc[common_samples].copy()
    clinical = clinical.loc[common_samples].copy()

    counts = clinical["metastasis_within_5y"].value_counts(dropna=False)
    print("GSE21257 metastasis label counts:")
    print(counts.to_string())
    return expression, clinical, selected_map


def zscore_columns(expression: pd.DataFrame) -> pd.DataFrame:
    x = expression.apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(axis=0))
    stds = x.std(axis=0).replace(0, np.nan)
    z = (x - x.mean(axis=0)) / stds
    return z.loc[:, z.notna().all(axis=0)]


def zscore_series(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    std = values.std()
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=values.index)
    return (values - values.mean()) / std


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    frame = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if frame.shape[0] < 5 or frame["a"].std() == 0 or frame["b"].std() == 0:
        return np.nan
    return float(frame["a"].corr(frame["b"]))


def human_cohort_pc1(z: pd.DataFrame, reference: pd.Series | None = None) -> pd.Series:
    if z.shape[1] < 2:
        return pd.Series(np.nan, index=z.index)
    pca = PCA(n_components=1, random_state=RANDOM_SEED)
    score = pd.Series(pca.fit_transform(z).ravel(), index=z.index)
    if reference is not None:
        corr = safe_corr(score, reference)
        if np.isfinite(corr) and corr < 0:
            score = -score
    return zscore_series(score)


def compute_module_scores(
    expression: pd.DataFrame,
    strict_weights: pd.DataFrame,
    broad_weights: pd.DataFrame,
    manifest: pd.DataFrame,
    cohort_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expression = expression.copy()
    expression.columns = expression.columns.astype(str).str.upper()
    expression = expression.loc[:, ~expression.columns.duplicated()].copy()

    scores = pd.DataFrame(index=expression.index)
    coverage_rows = []

    for mapping_name, weights in [("strict", strict_weights), ("broad", broad_weights)]:
        for module_label, part in weights.groupby("module_label"):
            part = part.copy()
            part["human_gene_symbol"] = part["human_gene_symbol"].astype(str).str.upper()
            part = part.drop_duplicates("human_gene_symbol", keep="first")
            requested = part["human_gene_symbol"].tolist()
            available = [gene for gene in requested if gene in expression.columns]
            n_requested = len(requested)
            n_available = len(available)
            fraction = n_available / n_requested if n_requested else 0.0
            passed = n_available >= MIN_SCORE_GENES and fraction >= MIN_SCORE_FRACTION

            coverage_rows.append(
                {
                    "cohort": cohort_name,
                    "module_label": module_label,
                    "mapping": mapping_name,
                    "n_frozen_genes": n_requested,
                    "n_available_genes": n_available,
                    "coverage_fraction": fraction,
                    "minimum_rule_passed": passed,
                    "available_genes": ";".join(available),
                    "missing_genes": ";".join([gene for gene in requested if gene not in available]),
                }
            )
            if not passed:
                continue

            z = zscore_columns(expression[available])
            available = list(z.columns)
            if len(available) < MIN_SCORE_GENES:
                continue
            weight_indexed = part.set_index("human_gene_symbol").loc[available]
            signs = np.sign(
                pd.to_numeric(weight_indexed["risk_oriented_loading"], errors="coerce")
            ).replace(0, 1)
            signed_mean = z.mul(signs, axis=1).mean(axis=1)
            signed_mean = zscore_series(signed_mean)

            raw_weights = pd.to_numeric(
                weight_indexed["risk_oriented_loading"], errors="coerce"
            ).fillna(0.0)
            if raw_weights.abs().sum() > 0:
                normalized_weights = raw_weights / raw_weights.abs().sum()
                weighted = z.mul(normalized_weights, axis=1).sum(axis=1)
                weighted = zscore_series(weighted)
            else:
                weighted = pd.Series(np.nan, index=z.index)

            pc1 = human_cohort_pc1(z, reference=signed_mean)
            prefix = f"{module_label}__{mapping_name}"
            scores[f"{prefix}__signed_mean_z"] = signed_mean
            scores[f"{prefix}__canine_pca_weighted_z"] = weighted
            scores[f"{prefix}__human_pc1_z"] = pc1

    coverage = pd.DataFrame(coverage_rows)
    tier_map = manifest.set_index("module_label")["validation_tier"].to_dict()
    scores.insert(0, "cohort", cohort_name)
    coverage["validation_tier"] = coverage["module_label"].map(tier_map)
    return scores, coverage


def detect_ortholog_columns(table: pd.DataFrame) -> tuple[str, str, str]:
    dog_col = next(
        (column for column in ["gene", "gene_symbol_clean", "canine_gene_symbol"] if column in table.columns),
        None,
    )
    human_col = next(
        (column for column in ["human_gene_symbol", "human_symbol"] if column in table.columns),
        None,
    )
    status_col = "ortholog_qc_status" if "ortholog_qc_status" in table.columns else None
    if dog_col is None or human_col is None or status_col is None:
        raise ValueError("Ortholog QC table lacks required dog, human, or QC-status columns.")
    return dog_col, human_col, status_col


def clean_canine_symbol(value: Any) -> str:
    text = str(value)
    tail = text.rsplit("_", 1)[-1]
    if tail.isdigit():
        text = text.rsplit("_", 1)[0]
    return text.strip().upper()


def build_strict_human_proliferation_mapping(
    proliferation_table: pd.DataFrame,
    ortholog_qc: pd.DataFrame,
) -> pd.DataFrame:
    proliferation_gene_col = next(
        (
            column
            for column in ["gene", "expression_column", "gene_symbol", "canine_gene"]
            if column in proliferation_table.columns
        ),
        None,
    )
    if proliferation_gene_col is None:
        raise ValueError(
            f"Could not detect the proliferation gene column: {list(proliferation_table.columns)}"
        )

    dog_col, human_col, status_col = detect_ortholog_columns(ortholog_qc)
    mapping = ortholog_qc.copy()
    mapping["dog_symbol_key"] = mapping[dog_col].map(clean_canine_symbol)
    mapping[human_col] = mapping[human_col].astype(str).str.strip().str.upper()
    mapping = mapping[
        mapping[status_col].eq("strict_symbol_concordant_one_to_one")
        & mapping[human_col].ne("")
    ].copy()

    genes = proliferation_table[[proliferation_gene_col]].copy()
    genes["canine_proliferation_gene"] = genes[proliferation_gene_col].astype(str)
    genes["dog_symbol_key"] = genes[proliferation_gene_col].map(clean_canine_symbol)
    merged = genes.merge(
        mapping[["dog_symbol_key", human_col, status_col]],
        on="dog_symbol_key",
        how="left",
    )
    merged = merged.rename(columns={human_col: "human_gene_symbol"})
    merged = merged[merged["human_gene_symbol"].notna()].copy()
    merged["human_gene_symbol"] = merged["human_gene_symbol"].astype(str).str.upper()
    merged = merged.drop_duplicates("human_gene_symbol", keep="first")
    merged.to_csv(OUTPUT_PROLIFERATION_MAPPING, index=False)
    return merged


def compute_proliferation_scores(
    expression: pd.DataFrame,
    proliferation_mapping: pd.DataFrame,
    strict_weights: pd.DataFrame,
    cohort_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expression = expression.copy()
    expression.columns = expression.columns.astype(str).str.upper()
    expression = expression.loc[:, ~expression.columns.duplicated()].copy()

    frozen_genes = proliferation_mapping["human_gene_symbol"].dropna().astype(str).str.upper().drop_duplicates().tolist()
    available = [gene for gene in frozen_genes if gene in expression.columns]
    fraction = len(available) / len(frozen_genes) if frozen_genes else 0.0
    passed = len(available) >= MIN_PROLIFERATION_GENES and fraction >= MIN_PROLIFERATION_FRACTION

    coverage_rows = [
        {
            "cohort": cohort_name,
            "score": "strict_human_meta_proliferation_pc1",
            "n_frozen_genes": len(frozen_genes),
            "n_available_genes": len(available),
            "coverage_fraction": fraction,
            "minimum_rule_passed": passed,
            "available_genes": ";".join(available),
            "missing_genes": ";".join([gene for gene in frozen_genes if gene not in available]),
        }
    ]

    scores = pd.DataFrame(index=expression.index)
    if not passed:
        return scores, pd.DataFrame(coverage_rows)

    z = zscore_columns(expression[available])
    anchor_genes = [gene for gene in PROLIFERATION_ANCHOR_SYMBOLS if gene in z.columns]
    reference = z[anchor_genes].mean(axis=1) if len(anchor_genes) >= 3 else z.mean(axis=1)
    proliferation_pc1 = human_cohort_pc1(z, reference=reference)
    scores["strict_human_meta_proliferation_pc1_z"] = proliferation_pc1

    m40_genes = (
        strict_weights[strict_weights["module_label"].eq("M40")]["human_gene_symbol"]
        .dropna()
        .astype(str)
        .str.upper()
        .drop_duplicates()
        .tolist()
    )
    disjoint_frozen = [gene for gene in frozen_genes if gene not in set(m40_genes)]
    disjoint_available = [gene for gene in disjoint_frozen if gene in expression.columns]
    disjoint_fraction = len(disjoint_available) / len(disjoint_frozen) if disjoint_frozen else 0.0
    disjoint_passed = (
        len(disjoint_available) >= MIN_PROLIFERATION_GENES
        and disjoint_fraction >= MIN_PROLIFERATION_FRACTION
    )
    coverage_rows.append(
        {
            "cohort": cohort_name,
            "score": "M40_disjoint_strict_human_meta_proliferation_pc1",
            "n_frozen_genes": len(disjoint_frozen),
            "n_available_genes": len(disjoint_available),
            "coverage_fraction": disjoint_fraction,
            "minimum_rule_passed": disjoint_passed,
            "available_genes": ";".join(disjoint_available),
            "missing_genes": ";".join([gene for gene in disjoint_frozen if gene not in disjoint_available]),
        }
    )
    if disjoint_passed:
        z_disjoint = zscore_columns(expression[disjoint_available])
        anchor_disjoint = [gene for gene in PROLIFERATION_ANCHOR_SYMBOLS if gene in z_disjoint.columns]
        reference_disjoint = (
            z_disjoint[anchor_disjoint].mean(axis=1)
            if len(anchor_disjoint) >= 3
            else z_disjoint.mean(axis=1)
        )
        disjoint_pc1 = human_cohort_pc1(z_disjoint, reference=reference_disjoint)
        scores["M40_disjoint_strict_human_meta_proliferation_pc1_z"] = disjoint_pc1

    return scores, pd.DataFrame(coverage_rows)


def residualize_outcome_blind(score: pd.Series, covariate: pd.Series) -> pd.Series:
    frame = pd.concat([score.rename("score"), covariate.rename("covariate")], axis=1).dropna()
    residual = pd.Series(np.nan, index=score.index)
    if frame.shape[0] < 10 or frame["covariate"].std() == 0:
        return residual
    x = np.column_stack([np.ones(frame.shape[0]), frame["covariate"].values])
    beta, _, _, _ = np.linalg.lstsq(x, frame["score"].values, rcond=None)
    residual.loc[frame.index] = frame["score"].values - x @ beta
    return zscore_series(residual)


def add_m40_residual_scores(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    proliferation_col = "M40_disjoint_strict_human_meta_proliferation_pc1_z"
    if proliferation_col not in out.columns:
        return out
    for module_score_col in [
        "M40__strict__signed_mean_z",
        "M40__strict__canine_pca_weighted_z",
    ]:
        if module_score_col not in out.columns:
            continue
        residual_col = module_score_col.replace(
            "_z", "__residual_to_disjoint_proliferation_z"
        )
        out[residual_col] = residualize_outcome_blind(
            out[module_score_col],
            out[proliferation_col],
        )
    return out


def merge_score_components(
    module_scores: pd.DataFrame,
    proliferation_scores: pd.DataFrame,
) -> pd.DataFrame:
    cohort = module_scores["cohort"] if "cohort" in module_scores.columns else None
    module_numeric = module_scores.drop(columns=["cohort"], errors="ignore")
    out = module_numeric.join(proliferation_scores, how="outer")
    out = add_m40_residual_scores(out)
    if cohort is not None:
        out.insert(0, "cohort", cohort.reindex(out.index))
    return out


def write_readme() -> None:
    text = f"""Human osteosarcoma cohort preparation\n\nScript version: {SCRIPT_VERSION}\n\nThis script prepares TARGET-OS and GSE21257 without fitting any outcome model.\nFrozen canine module membership, score direction, PCA weights, and validation tiers are not changed.\n\nPrimary external scores:\n- strict one-to-one signed mean z-score for M34, M11, M24, and M40\n- TARGET-OS: overall-survival metadata prepared from public GDC clinical fields\n- GSE21257: metastasis-within-five-years label parsed from GEO metadata\n\nSecondary/sensitivity scores:\n- strict canine PCA-weighted score\n- broad mapped score\n- human-cohort PC1 oriented without outcomes\n- M40 residual to a disjoint strict human proliferation PC1\n\nNo survival or metastasis association is tested in this script.\n"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def create_preparation_manifest(
    freeze: dict[str, Any],
    target_expression: pd.DataFrame,
    target_clinical: pd.DataFrame,
    target_scores: pd.DataFrame,
    gse_expression: pd.DataFrame,
    gse_clinical: pd.DataFrame,
    gse_scores: pd.DataFrame,
) -> None:
    output_paths = [
        OUTPUT_TARGET_EXPRESSION,
        OUTPUT_TARGET_CLINICAL,
        OUTPUT_TARGET_SAMPLE_MAP,
        OUTPUT_TARGET_SCORES,
        OUTPUT_TARGET_COVERAGE,
        OUTPUT_TARGET_GDC_FILE_MANIFEST,
        OUTPUT_GSE_EXPRESSION,
        OUTPUT_GSE_CLINICAL,
        OUTPUT_GSE_PROBE_MAP,
        OUTPUT_GSE_SCORES,
        OUTPUT_GSE_COVERAGE,
        OUTPUT_PROLIFERATION_MAPPING,
        OUTPUT_PREPARATION_SUMMARY,
        OUTPUT_README,
    ]
    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "frozen_program_freeze_sha256": sha256_file(FREEZE_JSON_FILE),
        "frozen_program_definition": freeze,
        "sources": {
            "TARGET_OS": {
                "project": GDC_PROJECT_ID,
                "api_base": GDC_API_BASE,
                "data_category": "Transcriptome Profiling",
                "data_type": "Gene Expression Quantification",
                "workflow_type": "STAR - Counts",
                "access": "open",
            },
            "GSE21257": {
                "accession": GSE_ACCESSION,
                "source": "NCBI GEO via GEOparse",
            },
        },
        "cohort_dimensions": {
            "TARGET_OS_expression": list(target_expression.shape),
            "TARGET_OS_clinical": list(target_clinical.shape),
            "TARGET_OS_scores": list(target_scores.shape),
            "GSE21257_expression": list(gse_expression.shape),
            "GSE21257_clinical": list(gse_clinical.shape),
            "GSE21257_scores": list(gse_scores.shape),
        },
        "outcome_guardrail": "Outcomes were prepared but were not used to construct or revise any molecular score.",
        "files": {},
    }
    for path in output_paths:
        if path.exists():
            manifest["files"][path.name] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    OUTPUT_PREPARATION_MANIFEST.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def print_coverage_summary(coverage: pd.DataFrame, cohort: str) -> None:
    print("")
    print("=" * 80)
    print(f"Frozen score coverage: {cohort}")
    print("=" * 80)
    display = coverage.copy()
    keep = [
        "module_label",
        "mapping",
        "validation_tier",
        "n_frozen_genes",
        "n_available_genes",
        "coverage_fraction",
        "minimum_rule_passed",
    ]
    keep = [column for column in keep if column in display.columns]
    if keep:
        print(
            display[keep]
            .sort_values([column for column in ["validation_tier", "module_label", "mapping"] if column in keep])
            .to_string(index=False)
        )


def main() -> None:
    print("=" * 80)
    print("Prepare human osteosarcoma validation cohorts")
    print("=" * 80)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Human processed directory: {HUMAN_PROCESSED_DIR}")
    print("")
    print("Design:")
    print("  Verify frozen canine transfer assets and hashes.")
    print("  Acquire public TARGET-OS RNA-seq and clinical metadata from the GDC API.")
    print("  Acquire GSE21257 expression and phenotype metadata from NCBI GEO.")
    print("  Harmonize expression to human gene symbols using outcome-blind rules.")
    print("  Construct frozen module scores without fitting any outcome model.")
    print("")

    freeze = verify_frozen_inputs()
    manifest = read_required_csv(FROZEN_MANIFEST_FILE)
    strict_weights = read_required_csv(STRICT_WEIGHTS_FILE)
    broad_weights = read_required_csv(BROAD_WEIGHTS_FILE)
    ortholog_qc = read_required_csv(ORTHOLOG_QC_FILE)
    proliferation_table = read_required_csv(PROLIFERATION_GENE_FILE)

    strict_weights["human_gene_symbol"] = strict_weights["human_gene_symbol"].astype(str).str.upper()
    broad_weights["human_gene_symbol"] = broad_weights["human_gene_symbol"].astype(str).str.upper()

    proliferation_mapping = build_strict_human_proliferation_mapping(
        proliferation_table,
        ortholog_qc,
    )
    print(f"Strict human proliferation genes: {proliferation_mapping.shape[0]}")

    target_file_manifest = query_target_os_expression_files()
    print("")
    print("TARGET-OS GDC expression manifest:")
    print(f"  Cases/files selected: {target_file_manifest.shape[0]}")
    print(
        target_file_manifest["sample_types"]
        .replace("", "<missing>")
        .value_counts()
        .head(10)
        .to_string()
    )
    target_downloaded = download_target_expression_files(target_file_manifest)
    target_expression, target_sample_map = build_target_expression(target_downloaded)
    target_cases = query_target_cases()
    target_clinical = standardize_target_clinical(target_cases)

    target_common = target_expression.index.intersection(target_clinical.index)
    target_expression = target_expression.loc[target_common].copy()
    target_clinical = target_clinical.loc[target_common].copy()
    target_sample_map = target_sample_map[
        target_sample_map["case_submitter_id"].isin(target_common)
    ].copy()

    target_module_scores, target_coverage = compute_module_scores(
        target_expression,
        strict_weights,
        broad_weights,
        manifest,
        "TARGET_OS",
    )
    target_proliferation_scores, target_proliferation_coverage = compute_proliferation_scores(
        target_expression,
        proliferation_mapping,
        strict_weights,
        "TARGET_OS",
    )
    target_scores = merge_score_components(
        target_module_scores,
        target_proliferation_scores,
    )
    target_coverage_all = pd.concat(
        [target_coverage, target_proliferation_coverage],
        axis=0,
        ignore_index=True,
        sort=False,
    )

    target_expression.to_csv(OUTPUT_TARGET_EXPRESSION)
    target_clinical.to_csv(OUTPUT_TARGET_CLINICAL)
    target_sample_map.to_csv(OUTPUT_TARGET_SAMPLE_MAP, index=False)
    target_scores.to_csv(OUTPUT_TARGET_SCORES)
    target_coverage_all.to_csv(OUTPUT_TARGET_COVERAGE, index=False)

    gse_expression, gse_clinical, _ = prepare_gse21257()
    gse_module_scores, gse_coverage = compute_module_scores(
        gse_expression,
        strict_weights,
        broad_weights,
        manifest,
        GSE_ACCESSION,
    )
    gse_proliferation_scores, gse_proliferation_coverage = compute_proliferation_scores(
        gse_expression,
        proliferation_mapping,
        strict_weights,
        GSE_ACCESSION,
    )
    gse_scores = merge_score_components(
        gse_module_scores,
        gse_proliferation_scores,
    )
    gse_coverage_all = pd.concat(
        [gse_coverage, gse_proliferation_coverage],
        axis=0,
        ignore_index=True,
        sort=False,
    )

    gse_expression.to_csv(OUTPUT_GSE_EXPRESSION)
    gse_clinical.to_csv(OUTPUT_GSE_CLINICAL)
    gse_scores.to_csv(OUTPUT_GSE_SCORES)
    gse_coverage_all.to_csv(OUTPUT_GSE_COVERAGE, index=False)

    summary = pd.DataFrame(
        [
            {
                "cohort": "TARGET_OS",
                "n_expression_samples": target_expression.shape[0],
                "n_expression_genes": target_expression.shape[1],
                "n_clinical_rows": target_clinical.shape[0],
                "n_os_complete": int(
                    target_clinical[["os_time_days", "os_event"]].dropna().shape[0]
                ),
                "n_metastasis_labels": np.nan,
                "n_frozen_score_columns": target_scores.shape[1] - int("cohort" in target_scores.columns),
            },
            {
                "cohort": GSE_ACCESSION,
                "n_expression_samples": gse_expression.shape[0],
                "n_expression_genes": gse_expression.shape[1],
                "n_clinical_rows": gse_clinical.shape[0],
                "n_os_complete": int(
                    gse_clinical[["os_time_months", "os_event"]].dropna().shape[0]
                ),
                "n_metastasis_labels": int(gse_clinical["metastasis_within_5y"].notna().sum()),
                "n_frozen_score_columns": gse_scores.shape[1] - int("cohort" in gse_scores.columns),
            },
        ]
    )
    summary.to_csv(OUTPUT_PREPARATION_SUMMARY, index=False)
    write_readme()
    create_preparation_manifest(
        freeze=freeze,
        target_expression=target_expression,
        target_clinical=target_clinical,
        target_scores=target_scores,
        gse_expression=gse_expression,
        gse_clinical=gse_clinical,
        gse_scores=gse_scores,
    )

    print("")
    print("=" * 80)
    print("Human cohort preparation summary")
    print("=" * 80)
    print(summary.to_string(index=False))
    print_coverage_summary(target_coverage, "TARGET_OS")
    print_coverage_summary(gse_coverage, GSE_ACCESSION)

    print("")
    print("=" * 80)
    print("Endpoint preparation audit")
    print("=" * 80)
    print("TARGET-OS OS fields:")
    print(
        target_clinical[["os_time_days", "os_event", "vital_status"]]
        .agg(["count"])
        .to_string()
    )
    print("")
    print("GSE21257 metastasis labels:")
    print(gse_clinical["metastasis_within_5y"].value_counts(dropna=False).to_string())
    print("")
    print("GSE21257 OS fields:")
    print(
        gse_clinical[["os_time_months", "os_event"]]
        .agg(["count"])
        .to_string()
    )

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("No human outcome was used to select genes, orient scores, tune weights, or revise validation tiers.")
    print("GSE21257 probe collapsing used the highest-variance probe per unambiguous gene symbol without outcome labels.")
    print("TARGET-OS expression used open GDC STAR-Counts files and was harmonized at the gene-symbol level.")
    print("M40 residual scores are mechanistic sensitivity scores based on a disjoint human proliferation PC1.")
    print("Outcome association, multiplicity control, and external performance estimation are deferred to script 23.")

    print("")
    print("Saved:")
    for path in [
        OUTPUT_TARGET_EXPRESSION,
        OUTPUT_TARGET_CLINICAL,
        OUTPUT_TARGET_SAMPLE_MAP,
        OUTPUT_TARGET_SCORES,
        OUTPUT_TARGET_COVERAGE,
        OUTPUT_GSE_EXPRESSION,
        OUTPUT_GSE_CLINICAL,
        OUTPUT_GSE_SCORES,
        OUTPUT_GSE_COVERAGE,
        OUTPUT_PROLIFERATION_MAPPING,
        OUTPUT_PREPARATION_SUMMARY,
        OUTPUT_PREPARATION_MANIFEST,
        OUTPUT_README,
    ]:
        print(path)
    print("Done.")


if __name__ == "__main__":
    main()
