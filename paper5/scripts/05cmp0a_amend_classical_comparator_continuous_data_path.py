from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_VERSION = "05cmp0a-amend-classical-comparator-continuous-data-path-v1-no-cli"
AMENDMENT_VERSION = "sable-mo-secondary-comparator-continuous-data-amendment-v1"
STATUS = "FROZEN_COMPARATOR_METHOD_CORRECTION_BEFORE_SCIENTIFIC_RUNS"
LITERATURE_CHECK_DATE = "2026-08-21"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

DIR_05CMP0 = PAPER5_ROOT / "method_contract" / "05cmp0"
LOCK_05CMP0 = DIR_05CMP0 / "secondary_comparator_selection_contract_lock.json"
CONTRACT_05CMP0 = DIR_05CMP0 / "secondary_comparator_selection_contract.json"
SELECTED_05CMP0 = DIR_05CMP0 / "secondary_comparator_selected_registry.csv"
CANDIDATES_05CMP0 = DIR_05CMP0 / "secondary_comparator_candidate_registry.csv"

DIR_05A2 = PAPER5_ROOT / "method_contract" / "05a2"
LOCK_05A2 = DIR_05A2 / "inference_final_gate_implementation_amendment_lock.json"

DIR_05DEV0 = PAPER5_ROOT / "method_contract" / "05dev0"
LOCK_05DEV0 = DIR_05DEV0 / "target_input_synthetic_fixture_lock.json"

DIR_05D5 = PAPER5_ROOT / "method_contract" / "05d5"
LOCK_05D5 = DIR_05D5 / "neural_confirmatory_stability_results_lock.json"
IN_PROGRESS_05D5 = DIR_05D5 / ".neural_confirmatory_stability_in_progress.json"

FORBIDDEN_RESULT_ROOTS = (
    PAPER5_ROOT / "method_contract" / "05cmp1",
    PAPER5_ROOT / "method_contract" / "05e",
    PAPER5_ROOT / "results" / "secondary_comparators",
    PAPER5_ROOT / "results" / "target_representation",
    PAPER5_ROOT / "results" / "biological_alignment",
    PAPER5_ROOT / "results" / "outcome_analysis",
    PAPER5_ROOT / "results" / "outcome_transport",
)

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05cmp0a"
AMENDMENT_JSON = OUTPUT_DIR / "secondary_comparator_continuous_data_amendment.json"
EFFECTIVE_SELECTED_CSV = OUTPUT_DIR / "effective_secondary_comparator_selected_registry.csv"
LITERATURE_NOTE_JSON = OUTPUT_DIR / "continuous_data_method_choice_evidence.json"
README_TXT = OUTPUT_DIR / "05cmp0a_README.txt"
LOCK_JSON = OUTPUT_DIR / "secondary_comparator_continuous_data_amendment_lock.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na"}:
        return ""
    return text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_project_path(value: Any) -> Path:
    path = Path(clean_text(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_json(path: Path) -> dict[str, Any]:
    require_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv_atomic(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    table.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def verify_generated_files(lock: dict[str, Any], label: str) -> None:
    failures: list[str] = []
    for relative_name, record in lock.get("generated_files", {}).items():
        path = resolve_project_path(relative_name)
        if not path.is_file():
            failures.append(f"missing: {relative_name}")
            continue
        if int(path.stat().st_size) != int(record.get("size_bytes", -1)):
            failures.append(f"size mismatch: {relative_name}")
            continue
        if sha256_file(path) != clean_text(record.get("sha256")):
            failures.append(f"hash mismatch: {relative_name}")
    if failures:
        raise RuntimeError(
            f"{label} generated-file verification failed:\n- "
            + "\n- ".join(failures[:100])
        )


def any_files(path: Path) -> bool:
    return path.is_file() or (path.is_dir() and any(item.is_file() for item in path.rglob("*")))


def verify_upstream() -> dict[str, Any]:
    for path in [
        LOCK_05CMP0,
        CONTRACT_05CMP0,
        SELECTED_05CMP0,
        CANDIDATES_05CMP0,
        LOCK_05A2,
        LOCK_05DEV0,
    ]:
        require_file(path)

    cmp0_lock = load_json(LOCK_05CMP0)
    cmp0_contract = load_json(CONTRACT_05CMP0)

    expected_cmp0_status = "FROZEN_SECONDARY_COMPARATOR_SELECTION_BEFORE_05E_AND_TARGET_RESULTS"
    if clean_text(cmp0_lock.get("status")) != expected_cmp0_status:
        raise RuntimeError("05cmp0 lock is not in the expected frozen state.")
    if clean_text(cmp0_contract.get("status")) != expected_cmp0_status:
        raise RuntimeError("05cmp0 contract is not in the expected frozen state.")
    if sha256_file(CONTRACT_05CMP0) != clean_text(cmp0_lock.get("contract_sha256")):
        raise RuntimeError("05cmp0 contract hash does not match its lock.")
    verify_generated_files(cmp0_lock, "05cmp0")

    selected = pd.read_csv(SELECTED_05CMP0, dtype=str).fillna("")
    required_classical = selected[
        selected["comparator_role"].eq("required_external_classical")
    ]
    if required_classical.shape[0] != 1:
        raise RuntimeError("05cmp0 does not contain exactly one required classical comparator.")
    if clean_text(required_classical.iloc[0]["method_name"]) != "iClusterBayes2":
        raise RuntimeError("05cmp0 classical comparator is no longer iClusterBayes2.")

    if any_files(PAPER5_ROOT / "results" / "secondary_comparators"):
        raise RuntimeError("Secondary comparator scientific result files already exist.")

    forbidden = [relative_path(root) for root in FORBIDDEN_RESULT_ROOTS if any_files(root)]
    if forbidden:
        raise RuntimeError(
            "Comparator amendment is no longer prospective because future result files exist:\n- "
            + "\n- ".join(forbidden)
        )

    if LOCK_05D5.is_file():
        d5_state = "COMPLETE"
    elif IN_PROGRESS_05D5.is_file():
        d5_state = "IN_PROGRESS"
    else:
        d5_state = "NOT_STARTED_OR_MARKER_NOT_DETECTED"

    return {
        "05cmp0_lock_sha256": sha256_file(LOCK_05CMP0),
        "05cmp0_contract_sha256": sha256_file(CONTRACT_05CMP0),
        "05cmp0_selected_registry_sha256": sha256_file(SELECTED_05CMP0),
        "05cmp0_candidate_registry_sha256": sha256_file(CANDIDATES_05CMP0),
        "05a2_lock_sha256": sha256_file(LOCK_05A2),
        "05dev0_lock_sha256": sha256_file(LOCK_05DEV0),
        "05d5_state": d5_state,
        "selected_registry": selected,
    }


def build_effective_selected(base: pd.DataFrame) -> pd.DataFrame:
    table = base.copy()
    table["base_method_name"] = table["method_name"]
    table["amended_by"] = ""
    table["amendment_reason"] = ""

    classical = table["comparator_role"].eq("required_external_classical")
    if classical.sum() != 1:
        raise RuntimeError("Expected exactly one required_external_classical row.")

    table.loc[classical, "candidate_id"] = "CMP_ICLUSTER2B"
    table.loc[classical, "method_name"] = "iCluster2b"
    table.loc[classical, "method_family"] = "Regularized joint latent-variable integration for continuous multi-omics"
    table.loc[classical, "publication_year"] = "2013"
    table.loc[classical, "publication"] = "Shen, Wang & Mo, Annals of Applied Statistics 7(1):269-294"
    table.loc[classical, "doi"] = ""
    table.loc[classical, "implementation"] = "Bioconductor iClusterPlus >= 3.23; iCluster2b / tune.iCluster2b"
    table.loc[classical, "whole_view_missingness_native"] = "not established as a required native capability"
    table.loc[classical, "small_n_evidence"] = (
        "Official package examples and iCluster family cancer workflows; no TARGET-like performance guarantee is inferred."
    )
    table.loc[classical, "selection_reason"] = (
        "Current Bioconductor iClusterPlus documentation explicitly recommends iCluster2b and tune.iCluster2b "
        "for continuous multi-omics data. Paper 5 RNA, methylation, and CNV representations are treated as "
        "continuous inputs, so this is the package-recommended continuous-data path."
    )
    table.loc[classical, "source_url_1"] = (
        "https://bioconductor.posit.co/packages/3.24/bioc/vignettes/iClusterPlus/inst/doc/iManual.pdf"
    )
    table.loc[classical, "source_url_2"] = (
        "https://bioconductor.posit.co/packages/3.23/bioc/manuals/iClusterPlus/man/iClusterPlus.pdf"
    )
    table.loc[classical, "amended_by"] = "05cmp0a"
    table.loc[classical, "amendment_reason"] = (
        "Prospective technical-method correction after checking current official documentation: "
        "iCluster2b is recommended for continuous multi-omics; iClusterBayes2 is recommended for mixed data types."
    )

    # iClusterBayes2 remains a predeclared same-package fallback, not a co-primary comparator.
    fallback_row = {
        column: ""
        for column in table.columns
    }
    fallback_row.update(
        {
            "candidate_id": "CMP_ICLUSTERBAYES2_FALLBACK",
            "method_name": "iClusterBayes2",
            "method_family": "Bayesian joint latent-variable integration",
            "publication_year": "2018",
            "publication": "Mo et al., Biostatistics 19(1):71-86",
            "doi": "10.1093/biostatistics/kxx017",
            "implementation": "Bioconductor iClusterPlus >= 3.23; iClusterBayes2",
            "public_code": "True",
            "unsupervised": "True",
            "requires_outcome_or_class_label": "False",
            "supports_joint_latent_representation": "True",
            "supports_continuous_multiomics": "True",
            "whole_view_missingness_native": "not established",
            "bulk_multiomics_relevance": "HIGH",
            "status": "PREDECLARED_SAME_PACKAGE_FALLBACK_ONLY",
            "rank": "1b",
            "selection_reason": "",
            "exclusion_or_fallback_reason": (
                "Use only if iCluster2b/tune.iCluster2b fails the predeclared implementation smoke test. "
                "It remains mathematically capable of Gaussian continuous inputs but is not the package-recommended "
                "first path for all-continuous multi-omics."
            ),
            "source_url_1": "https://bioconductor.posit.co/packages/3.23/bioc/html/iClusterPlus.html",
            "source_url_2": (
                "https://bioconductor.posit.co/packages/3.24/bioc/vignettes/iClusterPlus/inst/doc/iManual.pdf"
            ),
            "comparator_role": "predeclared_classical_same_package_fallback_only",
            "may_influence_05e_primary_method_freeze": "False",
            "may_influence_final_gate": "False",
            "may_access_outcomes": "False",
            "scientific_run_allowed_before_05e": "False",
            "synthetic_smoke_test_allowed_before_05e": "True",
            "exact_implementation_freeze_required_before_scientific_run": "True",
            "base_method_name": "iClusterBayes2",
            "amended_by": "05cmp0a",
            "amendment_reason": (
                "Retained only as same-package fallback after continuous-data recommendation correction."
            ),
        }
    )

    table = pd.concat([table, pd.DataFrame([fallback_row])], ignore_index=True)
    return table


def build_literature_note() -> dict[str, Any]:
    return {
        "evidence_version": "paper5-icluster-continuous-data-method-choice-v1",
        "checked_on": LITERATURE_CHECK_DATE,
        "official_source": (
            "Bioconductor iClusterPlus manual/tutorial current for the 2026 release cycle."
        ),
        "official_recommendation": (
            "Starting from Bioconductor 3.23, updated functions include iCluster2b, tune.iCluster2b, "
            "iClusterPlus2, and iClusterBayes2. The official manual recommends iCluster2b/tune.iCluster2b "
            "for continuous multi-omics and iClusterPlus2/iClusterBayes2 for mixed data types."
        ),
        "paper5_input_type_interpretation": (
            "The secondary comparator branch treats the Paper 5 RNA, methylation, and CNV analysis matrices "
            "as continuous-valued multi-omics inputs."
        ),
        "consequence": (
            "Use iCluster2b as the required classical external comparator. Keep iClusterBayes2 only as a "
            "same-package fallback if the recommended continuous-data path fails the predeclared smoke test."
        ),
        "scientific_results_used": False,
        "TARGET_results_used": False,
        "05d5_results_used": False,
        "source_urls": [
            "https://bioconductor.posit.co/packages/3.24/bioc/vignettes/iClusterPlus/inst/doc/iManual.pdf",
            "https://bioconductor.posit.co/packages/3.23/bioc/manuals/iClusterPlus/man/iClusterPlus.pdf",
            "https://bioconductor.posit.co/packages/3.23/bioc/html/iClusterPlus.html",
        ],
    }


def build_amendment(upstream: dict[str, Any]) -> dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "amendment_version": AMENDMENT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": STATUS,
        "amends": "05cmp0 classical comparator method choice only",
        "base_classical_comparator": "iClusterBayes2",
        "effective_classical_comparator": "iCluster2b",
        "same_package_fallback": "iClusterBayes2",
        "deep_comparator_unchanged": "MOVE",
        "deep_fallback_unchanged": "AUTOENCODIX varix",
        "reason": (
            "Official current iClusterPlus documentation recommends iCluster2b/tune.iCluster2b for continuous "
            "multi-omics data and iClusterBayes2 for mixed data types. Paper 5 uses continuous-valued RNA, "
            "methylation, and CNV analysis matrices. This correction is frozen before comparator installation "
            "smoke tests and before any scientific comparator result."
        ),
        "prospective_state": {
            "05d5_state": upstream["05d5_state"],
            "05d5_results_used": False,
            "05e_results_exist": False,
            "secondary_comparator_scientific_results_exist": False,
            "TARGET_representation_results_exist": False,
            "TARGET_outcomes_accessed": False,
        },
        "upstream_hashes": {
            "05cmp0_lock": upstream["05cmp0_lock_sha256"],
            "05cmp0_contract": upstream["05cmp0_contract_sha256"],
            "05cmp0_selected_registry": upstream["05cmp0_selected_registry_sha256"],
            "05cmp0_candidate_registry": upstream["05cmp0_candidate_registry_sha256"],
            "05a2_lock": upstream["05a2_lock_sha256"],
            "05dev0_lock": upstream["05dev0_lock_sha256"],
        },
        "unchanged_05cmp0_rules": [
            "secondary-only role",
            "no influence on 05e primary method freeze",
            "no influence on final-gate thresholds",
            "synthetic smoke tests only before 05e",
            "exact implementation must be frozen in 05cmp1 before scientific runs",
            "no outcome or biological labels for comparator configuration",
            "MOVE remains required deep comparator",
            "AUTOENCODIX varix remains MOVE-only fallback",
        ],
        "next_comparator_stage": (
            "Use 05cmp0 + 05cmp0a together as the effective comparator selection layer. "
            "Proceed to installation/synthetic smoke testing, then freeze exact implementations in 05cmp1."
        ),
    }


def verify_existing() -> None:
    require_file(LOCK_JSON)
    require_file(AMENDMENT_JSON)

    lock = load_json(LOCK_JSON)
    amendment = load_json(AMENDMENT_JSON)
    if clean_text(lock.get("status")) != STATUS:
        raise RuntimeError("Unexpected existing 05cmp0a lock status.")
    if clean_text(amendment.get("status")) != STATUS:
        raise RuntimeError("Unexpected existing 05cmp0a amendment status.")
    if sha256_file(AMENDMENT_JSON) != clean_text(lock.get("amendment_sha256")):
        raise RuntimeError("05cmp0a amendment hash mismatch.")
    verify_generated_files(lock, "05cmp0a")

    print("=" * 88)
    print("Existing Paper 5 05cmp0a comparator amendment: PASS")
    print("=" * 88)
    print(f"Amendment: {relative_path(AMENDMENT_JSON)}")
    print(f"Lock: {relative_path(LOCK_JSON)}")
    print("No files were rewritten.")
    print("=" * 88)


def main() -> None:
    print("=" * 88)
    print("Paper 5 - amend classical secondary comparator for continuous multi-omics")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("GPU computation in 05cmp0a: NO")
    print("Model fitting in 05cmp0a: NO")
    print("Scientific comparator result access: NO")
    print("TARGET/outcome access: NO")
    print()

    if LOCK_JSON.exists():
        verify_existing()
        return

    upstream = verify_upstream()
    print("05cmp0 frozen comparator-selection contract: PASS")
    print("05a2 + 05dev0 prospective anchors: PASS")
    print(f"05d5 state: {upstream['05d5_state']}")
    print("05e/TARGET/comparator scientific result roots present: NO")
    print()

    effective = build_effective_selected(upstream["selected_registry"])
    evidence = build_literature_note()
    amendment = build_amendment(upstream)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(effective, EFFECTIVE_SELECTED_CSV)
    write_json_atomic(evidence, LITERATURE_NOTE_JSON)
    write_json_atomic(amendment, AMENDMENT_JSON)

    readme = f"""Paper 5 - 05cmp0a classical comparator continuous-data amendment
==============================================================================
Script version: {SCRIPT_VERSION}
Status: {STATUS}
Literature check date: {LITERATURE_CHECK_DATE}

CORRECTION
----------
05cmp0 selected iClusterBayes2 as the required classical external comparator.

Current official iClusterPlus documentation recommends:
- iCluster2b / tune.iCluster2b for continuous multi-omics;
- iClusterPlus2 / iClusterBayes2 for mixed data types.

Paper 5 RNA, methylation and CNV analysis matrices are continuous-valued.
Therefore the effective required classical comparator is now iCluster2b.

FALLBACK
--------
iClusterBayes2 is retained only as a same-package fallback if iCluster2b fails
the frozen synthetic installation/smoke-test policy.

UNCHANGED
---------
MOVE remains the required deep external comparator.
AUTOENCODIX varix remains the MOVE-only deep fallback.
Secondary comparators remain ineligible to influence 05e or final-gate rules.

PROSPECTIVE STATUS
------------------
Scientific comparator results viewed: NO
TARGET comparator results viewed: NO
TARGET outcomes accessed: NO
05d5 results used to make this correction: NO

NEXT
----
Install/smoke-test iCluster2b and MOVE on 05dev0 synthetic fixtures only.
Then freeze exact versions and adapters in 05cmp1 before scientific runs.
"""
    write_text_atomic(readme, README_TXT)

    generated = [
        EFFECTIVE_SELECTED_CSV,
        LITERATURE_NOTE_JSON,
        AMENDMENT_JSON,
        README_TXT,
    ]

    lock = {
        "script_version": SCRIPT_VERSION,
        "amendment_version": AMENDMENT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": STATUS,
        "amendment_path": relative_path(AMENDMENT_JSON),
        "amendment_sha256": sha256_file(AMENDMENT_JSON),
        "base_05cmp0_lock_sha256": upstream["05cmp0_lock_sha256"],
        "generated_files": {
            relative_path(path): file_record(path)
            for path in generated
        },
        "rewrite_policy": (
            "05cmp0 remains immutable. 05cmp0a is the prospective correction layer. "
            "Any later method replacement requires a new versioned amendment."
        ),
    }
    write_json_atomic(lock, LOCK_JSON)

    print("-" * 88)
    print("Effective secondary comparator correction")
    print("-" * 88)
    print("05cmp0 classical comparator: iClusterBayes2")
    print("05cmp0a effective classical comparator: iCluster2b")
    print("Same-package classical fallback: iClusterBayes2")
    print("Deep comparator: MOVE (UNCHANGED)")
    print("Deep fallback: AUTOENCODIX varix (UNCHANGED)")
    print("Scientific results used for correction: NO")
    print()

    print("=" * 88)
    print("Paper 5 05cmp0a comparator correction: FROZEN")
    print("=" * 88)
    print(f"Amendment: {relative_path(AMENDMENT_JSON)}")
    print(f"Effective selected registry: {relative_path(EFFECTIVE_SELECTED_CSV)}")
    print(f"Evidence note: {relative_path(LITERATURE_NOTE_JSON)}")
    print(f"Lock: {relative_path(LOCK_JSON)}")
    print("05d5 may continue independently: YES")
    print("Next: synthetic installation/smoke tests, then 05cmp1.")
    print("=" * 88)


if __name__ == "__main__":
    main()
