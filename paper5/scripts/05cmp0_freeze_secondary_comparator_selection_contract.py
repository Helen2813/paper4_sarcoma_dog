from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_VERSION = "05cmp0-freeze-secondary-comparator-selection-contract-v1-no-cli"
CONTRACT_VERSION = "sable-mo-secondary-comparator-selection-contract-v1"
STATUS = "FROZEN_SECONDARY_COMPARATOR_SELECTION_BEFORE_05E_AND_TARGET_RESULTS"

LITERATURE_SEARCH_DATE = "2026-08-21"
LITERATURE_SEARCH_CUTOFF = "2026-08-21"

SCRIPT_PATH = Path(__file__).resolve()
PAPER5_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# -----------------------------------------------------------------------------
# Required upstream prospective anchors
# -----------------------------------------------------------------------------

DIR_05A2 = PAPER5_ROOT / "method_contract" / "05a2"
LOCK_05A2 = DIR_05A2 / "inference_final_gate_implementation_amendment_lock.json"
AMENDMENT_05A2 = DIR_05A2 / "inference_final_gate_implementation_amendment.json"

DIR_05DEV0 = PAPER5_ROOT / "method_contract" / "05dev0"
LOCK_05DEV0 = DIR_05DEV0 / "target_input_synthetic_fixture_lock.json"
CONTRACT_05DEV0 = DIR_05DEV0 / "target_input_synthetic_fixture_contract.json"

DIR_05D5 = PAPER5_ROOT / "method_contract" / "05d5"
LOCK_05D5 = DIR_05D5 / "neural_confirmatory_stability_results_lock.json"
IN_PROGRESS_05D5 = DIR_05D5 / ".neural_confirmatory_stability_in_progress.json"

# Scientific comparator results must not exist before this contract is frozen.
FORBIDDEN_RESULT_ROOTS = (
    PAPER5_ROOT / "method_contract" / "05e",
    PAPER5_ROOT / "method_contract" / "05cmp1",
    PAPER5_ROOT / "results" / "secondary_comparators",
    PAPER5_ROOT / "results" / "target_representation",
    PAPER5_ROOT / "results" / "biological_alignment",
    PAPER5_ROOT / "results" / "outcome_analysis",
    PAPER5_ROOT / "results" / "outcome_transport",
)

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

OUTPUT_DIR = PAPER5_ROOT / "method_contract" / "05cmp0"
CANDIDATE_REGISTRY = OUTPUT_DIR / "secondary_comparator_candidate_registry.csv"
SELECTED_REGISTRY = OUTPUT_DIR / "secondary_comparator_selected_registry.csv"
SEARCH_MANIFEST = OUTPUT_DIR / "secondary_comparator_literature_search_manifest.json"
SELECTION_CONTRACT = OUTPUT_DIR / "secondary_comparator_selection_contract.json"
FAILURE_POLICY = OUTPUT_DIR / "secondary_comparator_failure_policy.json"
BENCHMARK_POLICY = OUTPUT_DIR / "secondary_comparator_benchmark_policy.json"
README = OUTPUT_DIR / "05cmp0_README.txt"
LOCK = OUTPUT_DIR / "secondary_comparator_selection_contract_lock.json"


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

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
    table.to_csv(
        temporary,
        index=False,
        lineterminator="\n",
    )
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


# -----------------------------------------------------------------------------
# Prospective-state verification
# -----------------------------------------------------------------------------

def verify_upstream() -> dict[str, Any]:
    for path in [LOCK_05A2, AMENDMENT_05A2, LOCK_05DEV0, CONTRACT_05DEV0]:
        require_file(path)

    lock_05a2 = load_json(LOCK_05A2)
    amendment_05a2 = load_json(AMENDMENT_05A2)
    if clean_text(lock_05a2.get("status")) != "FROZEN_BEFORE_FINAL_GATE_MATERIALIZATION":
        raise RuntimeError("05a2 is not frozen in the expected state.")
    if clean_text(amendment_05a2.get("status")) != "FROZEN_BEFORE_FINAL_GATE_MATERIALIZATION":
        raise RuntimeError("05a2 amendment status is unexpected.")
    if sha256_file(AMENDMENT_05A2) != clean_text(lock_05a2.get("amendment_sha256")):
        raise RuntimeError("05a2 amendment hash does not match its lock.")
    verify_generated_files(lock_05a2, "05a2")

    lock_05dev0 = load_json(LOCK_05DEV0)
    contract_05dev0 = load_json(CONTRACT_05DEV0)
    expected_fixture_status = "FROZEN_SYNTHETIC_TARGET_INPUT_FIXTURES_BEFORE_TARGET_CODE"
    if clean_text(lock_05dev0.get("status")) != expected_fixture_status:
        raise RuntimeError("05dev0 lock status is unexpected.")
    if clean_text(contract_05dev0.get("status")) != expected_fixture_status:
        raise RuntimeError("05dev0 contract status is unexpected.")
    if sha256_file(CONTRACT_05DEV0) != clean_text(lock_05dev0.get("contract_sha256")):
        raise RuntimeError("05dev0 contract hash does not match its lock.")
    verify_generated_files(lock_05dev0, "05dev0")

    forbidden: list[str] = []
    for root in FORBIDDEN_RESULT_ROOTS:
        if any_files(root):
            forbidden.append(relative_path(root))
    if forbidden:
        raise RuntimeError(
            "Secondary-comparator selection is no longer prospective because future result files exist:\n- "
            + "\n- ".join(forbidden)
        )

    if LOCK_05D5.is_file():
        d5_state = "COMPLETE"
    elif IN_PROGRESS_05D5.is_file():
        d5_state = "IN_PROGRESS"
    else:
        d5_state = "NOT_STARTED_OR_MARKER_NOT_DETECTED"

    return {
        "05a2_amendment_sha256": sha256_file(AMENDMENT_05A2),
        "05a2_lock_sha256": sha256_file(LOCK_05A2),
        "05dev0_contract_sha256": sha256_file(CONTRACT_05DEV0),
        "05dev0_lock_sha256": sha256_file(LOCK_05DEV0),
        "05d5_state": d5_state,
    }


# -----------------------------------------------------------------------------
# Literature audit frozen on 2026-08-21
# -----------------------------------------------------------------------------

def build_candidate_registry() -> pd.DataFrame:
    rows = [
        {
            "candidate_id": "CMP_ICLUSTERBAYES2",
            "method_name": "iClusterBayes2",
            "method_family": "Bayesian joint latent-variable integration",
            "publication_year": 2018,
            "publication": "Mo et al., Biostatistics 19(1):71-86",
            "doi": "10.1093/biostatistics/kxx017",
            "implementation": "Bioconductor iClusterPlus >= 3.23; iClusterBayes2",
            "public_code": True,
            "unsupervised": True,
            "requires_outcome_or_class_label": False,
            "supports_joint_latent_representation": True,
            "supports_continuous_multiomics": True,
            "whole_view_missingness_native": "not established as a required native capability",
            "bulk_multiomics_relevance": "HIGH",
            "small_n_evidence": (
                "Official 2026 iClusterBayes2 GBM tutorial uses 84 primary samples and produces a latent-factor matrix."
            ),
            "status": "SELECTED_PRIMARY_SECONDARY_COMPARATOR",
            "rank": 1,
            "selection_reason": (
                "Established published non-neural Bayesian latent-variable comparator, directly relevant to cancer "
                "multi-omics and sample size close to TARGET."
            ),
            "exclusion_or_fallback_reason": "",
            "source_url_1": "https://pubmed.ncbi.nlm.nih.gov/28541380/",
            "source_url_2": (
                "https://bioconductor.posit.co/packages/3.23/bioc/vignettes/"
                "iClusterPlus/inst/doc/Tutorial_iClusterBayes2_GBM.html"
            ),
        },
        {
            "candidate_id": "CMP_MOVE",
            "method_name": "MOVE",
            "method_family": "Multi-Omics Variational autoEncoder",
            "publication_year": 2023,
            "publication": "Allesoe et al., Nature Biotechnology",
            "doi": "10.1038/s41587-022-01520-x",
            "implementation": "move-dl 1.5.0; Python >=3.9; PyTorch",
            "public_code": True,
            "unsupervised": True,
            "requires_outcome_or_class_label": False,
            "supports_joint_latent_representation": True,
            "supports_continuous_multiomics": True,
            "whole_view_missingness_native": "YES / published framework reports resistance to substantial missingness",
            "bulk_multiomics_relevance": "HIGH",
            "small_n_evidence": (
                "Published application n=789; no claim is made here that small-n performance is established."
            ),
            "status": "SELECTED_PRIMARY_SECONDARY_COMPARATOR",
            "rank": 2,
            "selection_reason": (
                "Named, peer-reviewed, generative multi-omics VAE with public PyTorch implementation, "
                "continuous-data support, latent-space output, and documented reconstruction/stability workflow."
            ),
            "exclusion_or_fallback_reason": "",
            "source_url_1": "https://www.nature.com/articles/s41587-022-01520-x",
            "source_url_2": "https://pypi.org/project/move-dl/",
        },
        {
            "candidate_id": "CMP_AUTOENCODIX_VARIX",
            "method_name": "AUTOENCODIX varix",
            "method_family": "General biological variational autoencoder framework",
            "publication_year": 2026,
            "publication": "Joas et al., Nature Computational Science 6:96-108",
            "doi": "10.1038/s43588-025-00916-4",
            "implementation": "AUTOENCODIX public PyTorch repository; varix architecture",
            "public_code": True,
            "unsupervised": True,
            "requires_outcome_or_class_label": False,
            "supports_joint_latent_representation": True,
            "supports_continuous_multiomics": True,
            "whole_view_missingness_native": "not required for fallback eligibility",
            "bulk_multiomics_relevance": "HIGH",
            "small_n_evidence": (
                "Framework explicitly addresses high-dimensional biological representation learning; "
                "no TARGET-like n=78 guarantee is inferred."
            ),
            "status": "FALLBACK_FOR_MOVE_ONLY",
            "rank": 3,
            "selection_reason": (
                "Recent peer-reviewed open-source PyTorch framework with a variational architecture and TCGA/multimodal examples."
            ),
            "exclusion_or_fallback_reason": (
                "Not first choice because the standalone Python-package transition is still evolving; "
                "used only if MOVE fails the predeclared implementation smoke-test policy."
            ),
            "source_url_1": "https://www.nature.com/articles/s43588-025-00916-4",
            "source_url_2": "https://github.com/jan-forest/autoencodix",
        },
        {
            "candidate_id": "CMP_MAUI",
            "method_name": "maui",
            "method_family": "Multi-omics autoencoder / Bayesian latent-factor neural model",
            "publication_year": 2019,
            "publication": "Ronen et al., Life Science Alliance 2(6):e201900517",
            "doi": "10.26508/lsa.201900517",
            "implementation": "maui-tools; historical Python 3.6 / TensorFlow 1.1 stack",
            "public_code": True,
            "unsupervised": True,
            "requires_outcome_or_class_label": False,
            "supports_joint_latent_representation": True,
            "supports_continuous_multiomics": True,
            "whole_view_missingness_native": "not established",
            "bulk_multiomics_relevance": "HIGH",
            "small_n_evidence": "No TARGET-like small-n guarantee inferred.",
            "status": "NOT_SELECTED_TECHNICAL_FRAGILITY",
            "rank": 4,
            "selection_reason": "",
            "exclusion_or_fallback_reason": (
                "Conceptually eligible, but official repository documents an obsolete TensorFlow 1.1 / Python 3.6 stack; "
                "not preferred when maintained modern alternatives are available."
            ),
            "source_url_1": "https://github.com/BIMSBbioinfo/maui",
            "source_url_2": "https://maui.readthedocs.io/en/latest/",
        },
        {
            "candidate_id": "CMP_SCMaui",
            "method_name": "scMaui",
            "method_family": "Single-cell multimodal variational PoE autoencoder",
            "publication_year": 2024,
            "publication": "BMC Bioinformatics",
            "doi": "10.1186/s12859-024-05880-w",
            "implementation": "public implementation described by authors",
            "public_code": True,
            "unsupervised": True,
            "requires_outcome_or_class_label": False,
            "supports_joint_latent_representation": True,
            "supports_continuous_multiomics": True,
            "whole_view_missingness_native": "YES",
            "bulk_multiomics_relevance": "LOW",
            "small_n_evidence": "Single-cell method; not treated as a bulk small-n comparator.",
            "status": "EXCLUDED_SCOPE_MISMATCH",
            "rank": 5,
            "selection_reason": "",
            "exclusion_or_fallback_reason": (
                "Designed specifically for single-cell multi-omics; excluded from the bulk patient-level primary secondary comparison."
            ),
            "source_url_1": "https://link.springer.com/article/10.1186/s12859-024-05880-w",
            "source_url_2": "",
        },
        {
            "candidate_id": "CMP_MODVAE",
            "method_name": "Multi-Omics Deconfounding VAE / XVAE family",
            "method_family": "Cancer multi-omics VAE with deconfounding variants",
            "publication_year": 2024,
            "publication": "Li et al., Briefings in Bioinformatics 25(6):bbae512",
            "doi": "10.1093/bib/bbae512",
            "implementation": "public Python repository",
            "public_code": True,
            "unsupervised": "PARTLY / vanilla XVAE is unsupervised; principal deconfounding workflows add confounder/label machinery",
            "requires_outcome_or_class_label": "principal wrappers include label/confounder inputs",
            "supports_joint_latent_representation": True,
            "supports_continuous_multiomics": True,
            "whole_view_missingness_native": "not established",
            "bulk_multiomics_relevance": "MEDIUM",
            "small_n_evidence": "Published TCGA examples are much larger than TARGET.",
            "status": "NOT_SELECTED_OBJECTIVE_MISMATCH",
            "rank": 6,
            "selection_reason": "",
            "exclusion_or_fallback_reason": (
                "Useful literature context, but the published contribution centers on deconfounding/subtyping and "
                "two-view workflows rather than a clean generic three-view outcome-blind comparator."
            ),
            "source_url_1": "https://academic.oup.com/bib/article/25/6/bbae512/7824239",
            "source_url_2": "https://github.com/ZuqiLi/Multi-Omics-Deconfounding-VAE",
        },
        {
            "candidate_id": "CMP_MOSA",
            "method_name": "MOSA",
            "method_family": "Cancer cell-line multi-omic synthetic augmentation VAE",
            "publication_year": 2024,
            "publication": "Cai et al., Nature Communications",
            "doi": "10.1038/s41467-024-54771-4",
            "implementation": "public PyTorch code in PhenPred repository",
            "public_code": True,
            "unsupervised": True,
            "requires_outcome_or_class_label": False,
            "supports_joint_latent_representation": True,
            "supports_continuous_multiomics": True,
            "whole_view_missingness_native": "YES / whole-view dropout and synthetic completion are core features",
            "bulk_multiomics_relevance": "MEDIUM",
            "small_n_evidence": "Developed on 1,523 cancer cell lines, not a TARGET-like patient cohort.",
            "status": "NOT_SELECTED_CONTEXT_MISMATCH",
            "rank": 7,
            "selection_reason": "",
            "exclusion_or_fallback_reason": (
                "Strong method but bespoke to DepMap cell lines and conditional cancer-cell-line covariates; "
                "less clean as a patient-level comparator than MOVE."
            ),
            "source_url_1": "https://www.nature.com/articles/s41467-024-54771-4",
            "source_url_2": "https://github.com/QuantitativeBiology/PhenPred",
        },
        {
            "candidate_id": "CMP_DIABLO",
            "method_name": "DIABLO",
            "method_family": "Supervised multi-omics integration",
            "publication_year": "",
            "publication": "mixOmics supervised integration method",
            "doi": "",
            "implementation": "mixOmics",
            "public_code": True,
            "unsupervised": False,
            "requires_outcome_or_class_label": True,
            "supports_joint_latent_representation": True,
            "supports_continuous_multiomics": True,
            "whole_view_missingness_native": "",
            "bulk_multiomics_relevance": "HIGH",
            "small_n_evidence": "",
            "status": "EXCLUDED_SUPERVISED",
            "rank": 8,
            "selection_reason": "",
            "exclusion_or_fallback_reason": (
                "Excluded from the outcome-blind representation benchmark because it requires supervised class/outcome information."
            ),
            "source_url_1": "",
            "source_url_2": "",
        },
        {
            "candidate_id": "CMP_MOFA_ALREADY_PRIMARY",
            "method_name": "MOFA / MOFA+",
            "method_family": "Unsupervised multi-view factor analysis",
            "publication_year": "",
            "publication": "Already in frozen Paper 5 primary baseline registry",
            "doi": "",
            "implementation": "already frozen upstream",
            "public_code": True,
            "unsupervised": True,
            "requires_outcome_or_class_label": False,
            "supports_joint_latent_representation": True,
            "supports_continuous_multiomics": True,
            "whole_view_missingness_native": "YES",
            "bulk_multiomics_relevance": "HIGH",
            "small_n_evidence": "",
            "status": "NOT_SECONDARY_ALREADY_PRIMARY_BASELINE",
            "rank": 9,
            "selection_reason": "",
            "exclusion_or_fallback_reason": (
                "Not duplicated in the secondary comparator branch because multiple MOFA K values are already frozen primary baselines."
            ),
            "source_url_1": "",
            "source_url_2": "",
        },
    ]
    return pd.DataFrame(rows)


def build_selected_registry(candidates: pd.DataFrame) -> pd.DataFrame:
    selected = candidates[
        candidates["status"].isin(
            ["SELECTED_PRIMARY_SECONDARY_COMPARATOR", "FALLBACK_FOR_MOVE_ONLY"]
        )
    ].copy()

    selected["comparator_role"] = selected["candidate_id"].map(
        {
            "CMP_ICLUSTERBAYES2": "required_external_classical",
            "CMP_MOVE": "required_external_deep",
            "CMP_AUTOENCODIX_VARIX": "predeclared_deep_fallback_only",
        }
    )
    selected["may_influence_05e_primary_method_freeze"] = False
    selected["may_influence_final_gate"] = False
    selected["may_access_outcomes"] = False
    selected["scientific_run_allowed_before_05e"] = False
    selected["synthetic_smoke_test_allowed_before_05e"] = True
    selected["exact_implementation_freeze_required_before_scientific_run"] = True

    return selected


def build_search_manifest() -> dict[str, Any]:
    return {
        "manifest_version": "paper5-secondary-comparator-literature-search-v1",
        "search_date": LITERATURE_SEARCH_DATE,
        "search_cutoff": LITERATURE_SEARCH_CUTOFF,
        "purpose": (
            "Select a small, defensible secondary comparator set before 05e and before any TARGET representation result."
        ),
        "eligibility_criteria": [
            "peer-reviewed published method or framework",
            "unsupervised for the benchmarked representation-learning step",
            "does not require survival, metastasis, treatment response, or class labels",
            "produces a patient/sample-level joint latent representation",
            "applicable to bulk multi-omics with continuous modalities",
            "public implementation or official package available",
            "reasonable path to RNA + methylation + CNV-like inputs",
            "can be evaluated without altering the frozen primary SABLE-MO selection hierarchy",
        ],
        "preference_criteria": [
            "published cancer or biomedical multi-omics use",
            "implementation currently installable or actively documented",
            "modern maintained Python/PyTorch or official Bioconductor implementation",
            "small-n evidence or at least no hard large-n requirement",
            "methodological diversity relative to existing MOFA/GCCA/PCA primary baselines",
        ],
        "exclusion_rules": [
            "supervised method requiring outcome or class labels",
            "single-cell-specific method without a justified bulk-patient formulation",
            "method already present in the frozen primary baseline registry",
            "implementation so obsolete that a maintained eligible alternative is available",
            "bespoke context that requires additional biological/clinical covariates not permitted in the outcome-blind benchmark",
        ],
        "search_terms_recorded": [
            "iClusterBayes multi-omics latent variable",
            "multi-omics variational autoencoder public code",
            "MOVE multi-omics VAE",
            "AUTOENCODIX variational autoencoder multi-omics",
            "maui multi-omics autoencoder",
            "multi-omics deconfounding VAE cancer",
            "MOSA unsupervised multi-omics VAE cancer",
        ],
        "selected": [
            "iClusterBayes2",
            "MOVE",
        ],
        "predeclared_fallback": [
            "AUTOENCODIX varix for MOVE only",
        ],
    }


def build_failure_policy() -> dict[str, Any]:
    return {
        "policy_version": "paper5-secondary-comparator-implementation-failure-policy-v1",
        "status": "FROZEN_BEFORE_COMPARATOR_INSTALLATION_AND_SCIENTIFIC_RUN",
        "general_rule": (
            "Comparator availability is determined by a predeclared smoke-test process on 05dev0 synthetic fixtures only. "
            "No real TARGET data, confirmatory comparator result, biology, or outcome may be used to choose a fallback."
        ),
        "implementation_freeze_stage": (
            "A later 05cmp1 stage must record exact package/repository version, environment, commit or release, "
            "input adapter, hyperparameter-selection rule, random seeds, supported metrics, and numerical implementation hash "
            "before the first scientific comparator run."
        ),
        "required_smoke_test": [
            "create an isolated environment using the authors' official installation path",
            "run an official tutorial or minimal author-provided example if available",
            "run the Paper 5 05dev0 synthetic valid fixture through the input adapter",
            "verify that latent output has one row per synthetic sample and finite values for supported cases",
            "verify deterministic/provenance recording under the frozen comparator seed policy to be defined in 05cmp1",
            "do not inspect real TARGET values during smoke testing",
        ],
        "unavailable_definition": (
            "A comparator is implementation-unavailable if both (1) the official published/repository installation path and "
            "(2) one dependency-pinned compatibility environment fail to complete the reference/synthetic smoke test without "
            "altering the published mathematical objective, loss, likelihood, or latent model."
        ),
        "policies": {
            "iClusterBayes2": {
                "preferred": "Bioconductor iClusterPlus >=3.23, function iClusterBayes2",
                "same_family_fallback": (
                    "legacy iClusterBayes from the official iClusterPlus package only if iClusterBayes2 is unavailable "
                    "under the predeclared smoke-test rule"
                ),
                "cross_family_fallback": "NONE",
                "if_unavailable": (
                    "Report iClusterBayes family as implementation-unavailable; do not substitute a different classical method post hoc."
                ),
            },
            "MOVE": {
                "preferred": "official move-dl release, target 1.5.0 unless a later official release is frozen in 05cmp1 before scientific runs",
                "fallback_1": "AUTOENCODIX varix",
                "if_preferred_unavailable": (
                    "Activate AUTOENCODIX varix only after MOVE is formally marked unavailable by the frozen smoke-test rule."
                ),
                "if_fallback_unavailable": (
                    "Report the external deep comparator as implementation-unavailable; do not search for a new favorable VAE after results."
                ),
            },
        },
        "forbidden": [
            "choosing a fallback based on scientific performance",
            "changing the primary SABLE-MO method or 05e decision",
            "using TARGET outcomes or biological annotations to configure a comparator",
            "modifying published comparator mathematics to make it run",
            "replacing a failed comparator with a new method not named in 05cmp0 without a new prospective versioned amendment",
        ],
    }


def build_benchmark_policy() -> dict[str, Any]:
    return {
        "policy_version": "paper5-secondary-comparator-benchmark-policy-v1",
        "status": "FROZEN_BEFORE_05E_AND_COMPARATOR_SCIENTIFIC_RUN",
        "role": (
            "Secondary external benchmarking only. These methods do not participate in primary architecture/profile selection, "
            "05e primary method freeze, or final-gate threshold validation."
        ),
        "timing": {
            "synthetic_smoke_tests": "allowed before 05e using 05dev0 only",
            "scientific_comparator_results": "forbidden before 05e primary representation-method freeze",
            "TARGET_comparator_results": "forbidden before 05e and 05cmp1 implementation freeze",
            "outcomes": "forbidden for comparator tuning and representation evaluation",
        },
        "reporting_principles": [
            "Do not create a forced single composite winner score across methods with different native capabilities.",
            "Use common representation-level endpoints wherever mathematically comparable.",
            "Report capability-specific endpoints only for methods that natively support them.",
            "Do not add custom decoders, imputation modules, or supervised heads to a comparator solely to make an endpoint available.",
            "Separate classical Bayesian and deep VAE comparator results from the frozen primary baseline/ablation table.",
            "Clearly label all comparator analyses as secondary post-freeze benchmarking.",
        ],
        "common_core_endpoints_to_freeze_in_05cmp1": [
            "latent representation dimensionality",
            "representation stability under the same declared secondary resampling protocol where feasible",
            "simulation shared-subspace recovery where a joint latent space is available",
            "technical/batch association diagnostics where applicable",
            "runtime and convergence/failure rate",
        ],
        "capability_specific_examples_to_freeze_in_05cmp1": [
            "held-out reconstruction only if the published implementation natively exposes reconstruction",
            "missing-view reconstruction only if the method natively supports missing-view inference",
            "cross-view retrieval only if separate-view or cross-modal embeddings are natively defined",
        ],
        "scientific_scope_finalization": (
            "05cmp1, frozen before any comparator scientific result, must define exact secondary simulation/TARGET datasets, "
            "metrics, preprocessing adapters, feature-budget policy, split policy, and hyperparameter-selection rule after "
            "both selected comparator implementations have been smoke-tested on 05dev0."
        ),
        "outcome_boundary": (
            "Secondary comparators remain outcome-blind. They cannot be configured or ranked using survival, metastasis, "
            "treatment, or biological-program labels."
        ),
    }


def build_selection_contract(
    upstream: dict[str, Any],
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": STATUS,
        "literature_search_date": LITERATURE_SEARCH_DATE,
        "literature_search_cutoff": LITERATURE_SEARCH_CUTOFF,
        "prospective_state": {
            "05d5_state_at_contract": upstream["05d5_state"],
            "05d5_results_used_for_comparator_selection": False,
            "05e_primary_method_frozen": False,
            "TARGET_representation_results_viewed": False,
            "TARGET_outcomes_viewed": False,
            "secondary_comparator_scientific_results_exist": False,
        },
        "upstream_hashes": {
            "05a2_amendment": upstream["05a2_amendment_sha256"],
            "05a2_lock": upstream["05a2_lock_sha256"],
            "05dev0_contract": upstream["05dev0_contract_sha256"],
            "05dev0_lock": upstream["05dev0_lock_sha256"],
        },
        "selection": {
            "required_external_classical": "iClusterBayes2",
            "required_external_deep": "MOVE",
            "predeclared_deep_fallback": "AUTOENCODIX varix",
            "number_required_if_implementable": 2,
        },
        "why_two_methods_only": (
            "The frozen primary benchmark already includes PCA-by-view, BLOCK_PCA, multiple MOFA configurations, ridge GCCA, "
            "and neural V0-V5. The secondary branch adds one established Bayesian latent-variable comparator and one published "
            "external VAE rather than creating a post-hoc method zoo."
        ),
        "selected_candidate_ids": selected["candidate_id"].astype(str).tolist(),
        "primary_secondary_candidate_ids": candidates[
            candidates["status"].eq("SELECTED_PRIMARY_SECONDARY_COMPARATOR")
        ]["candidate_id"].astype(str).tolist(),
        "fallback_candidate_ids": candidates[
            candidates["status"].eq("FALLBACK_FOR_MOVE_ONLY")
        ]["candidate_id"].astype(str).tolist(),
        "guardrails": [
            "Comparator selection is frozen before 05e and before any TARGET representation result.",
            "Comparator methods cannot alter the primary 05e decision.",
            "Comparator methods cannot alter final-gate thresholds or PASS/FAIL.",
            "No comparator scientific result may be generated before 05cmp1 freezes exact implementations.",
            "Only 05dev0 synthetic fixtures may be used for pre-05e comparator smoke tests.",
            "Any future comparator replacement requires a new versioned prospective amendment.",
        ],
        "next_comparator_stage": (
            "05cmp1_freeze_secondary_comparator_implementations.py after synthetic installation/smoke tests; "
            "05cmp1 must be frozen before any comparator scientific run."
        ),
    }


def build_readme(
    upstream: dict[str, Any],
    candidates: pd.DataFrame,
) -> str:
    selected = candidates[
        candidates["status"].eq("SELECTED_PRIMARY_SECONDARY_COMPARATOR")
    ]["method_name"].tolist()
    fallback = candidates[
        candidates["status"].eq("FALLBACK_FOR_MOVE_ONLY")
    ]["method_name"].tolist()

    return f"""Paper 5 - 05cmp0 secondary comparator selection contract
==============================================================================
Script version: {SCRIPT_VERSION}
Contract version: {CONTRACT_VERSION}
Status: {STATUS}
Literature search date: {LITERATURE_SEARCH_DATE}

WHY THIS CONTRACT EXISTS
------------------------
The primary Paper 5 benchmark was frozen before these external comparators and
already contains PCA-by-view, BLOCK_PCA, multiple MOFA configurations, ridge
GCCA, and V0-V5. 05cmp0 adds a small secondary post-freeze benchmark to address
the methods-paper question: does the frozen SABLE-MO result remain informative
next to established published external multi-omics methods?

SELECTED METHODS
----------------
Required classical external comparator: {selected[0]}
Required deep external comparator: {selected[1]}
Predeclared deep fallback: {fallback[0]}

WHY THESE TWO
-------------
iClusterBayes2:
- established Bayesian latent-variable multi-omics method;
- current official Bioconductor implementation;
- 2026 official tutorial demonstrates an 84-sample GBM multi-omics example,
  close to the TARGET sample-size regime.

MOVE:
- peer-reviewed Multi-Omics Variational autoEncoder;
- Nature Biotechnology publication;
- public PyTorch/PyPI implementation;
- supports continuous multi-modal inputs and latent-space analysis;
- published workflow explicitly uses reconstruction and stability for
  unsupervised model development.

FALLBACK
--------
AUTOENCODIX varix is frozen only as the MOVE fallback. It cannot replace MOVE
because of a favorable result; it activates only if MOVE fails the predefined
implementation smoke-test policy.

IMPORTANT TIMING
----------------
05d5 state when 05cmp0 was frozen: {upstream['05d5_state']}
05d5 result used to select comparators: NO
05e result available: NO
TARGET representation result available: NO
TARGET outcome accessed: NO

ALLOWED BEFORE 05E
------------------
Installation work and smoke tests on 05dev0 synthetic fixtures only.

FORBIDDEN BEFORE 05E / 05CMP1
-----------------------------
- scientific comparator simulation results;
- scientific comparator TARGET results;
- comparator-driven change to SABLE-MO;
- outcome/biology-guided comparator tuning.

NEXT COMPARATOR STAGE
---------------------
After installation/smoke tests, create 05cmp1 to freeze exact versions,
environments, adapters, hyperparameter-selection rules, seeds, and comparable
metrics BEFORE the first scientific comparator run.
"""


# -----------------------------------------------------------------------------
# Existing lock verification
# -----------------------------------------------------------------------------

def verify_existing() -> None:
    require_file(LOCK)
    require_file(SELECTION_CONTRACT)

    lock = load_json(LOCK)
    contract = load_json(SELECTION_CONTRACT)

    if clean_text(lock.get("status")) != STATUS:
        raise RuntimeError("Unexpected existing 05cmp0 lock status.")
    if clean_text(contract.get("status")) != STATUS:
        raise RuntimeError("Unexpected existing 05cmp0 contract status.")
    if sha256_file(SELECTION_CONTRACT) != clean_text(lock.get("contract_sha256")):
        raise RuntimeError("Existing 05cmp0 contract hash mismatch.")

    verify_generated_files(lock, "05cmp0")

    print("=" * 88)
    print("Existing Paper 5 secondary comparator selection contract: PASS")
    print("=" * 88)
    print(f"Contract: {relative_path(SELECTION_CONTRACT)}")
    print(f"Lock: {relative_path(LOCK)}")
    print("No files were rewritten.")
    print("=" * 88)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    print("=" * 88)
    print("Paper 5 - freeze secondary published comparator selection")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("GPU computation in 05cmp0: NO")
    print("Model fitting in 05cmp0: NO")
    print("TARGET data access in 05cmp0: NO")
    print("TARGET outcome access in 05cmp0: NO")
    print("Comparator scientific result access in 05cmp0: NO")
    print()

    if LOCK.exists():
        verify_existing()
        return

    upstream = verify_upstream()
    print("05a2 inference/final-gate clarification: PASS")
    print("05dev0 synthetic input fixture freeze: PASS")
    print(f"05d5 state: {upstream['05d5_state']}")
    print("05d5 results used for comparator choice: NO")
    print("05e/TARGET/comparator result roots present: NO")
    print()

    candidates = build_candidate_registry()
    selected = build_selected_registry(candidates)
    search_manifest = build_search_manifest()
    failure_policy = build_failure_policy()
    benchmark_policy = build_benchmark_policy()
    selection_contract = build_selection_contract(upstream, candidates, selected)

    # Internal deterministic contract checks.
    required_primary = candidates[
        candidates["status"].eq("SELECTED_PRIMARY_SECONDARY_COMPARATOR")
    ]
    if required_primary["candidate_id"].tolist() != [
        "CMP_ICLUSTERBAYES2",
        "CMP_MOVE",
    ]:
        raise RuntimeError("Primary secondary-comparator order is not iClusterBayes2 -> MOVE.")

    fallback = candidates[candidates["status"].eq("FALLBACK_FOR_MOVE_ONLY")]
    if fallback["candidate_id"].tolist() != ["CMP_AUTOENCODIX_VARIX"]:
        raise RuntimeError("Deep fallback is not exactly AUTOENCODIX varix.")

    if candidates[candidates["status"].eq("EXCLUDED_SUPERVISED")].shape[0] < 1:
        raise RuntimeError("Supervised-method exclusion is not represented in the candidate audit.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(candidates, CANDIDATE_REGISTRY)
    write_csv_atomic(selected, SELECTED_REGISTRY)
    write_json_atomic(search_manifest, SEARCH_MANIFEST)
    write_json_atomic(selection_contract, SELECTION_CONTRACT)
    write_json_atomic(failure_policy, FAILURE_POLICY)
    write_json_atomic(benchmark_policy, BENCHMARK_POLICY)
    write_text_atomic(build_readme(upstream, candidates), README)

    generated = [
        CANDIDATE_REGISTRY,
        SELECTED_REGISTRY,
        SEARCH_MANIFEST,
        SELECTION_CONTRACT,
        FAILURE_POLICY,
        BENCHMARK_POLICY,
        README,
    ]

    lock = {
        "script_version": SCRIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": STATUS,
        "contract_path": relative_path(SELECTION_CONTRACT),
        "contract_sha256": sha256_file(SELECTION_CONTRACT),
        "literature_search_date": LITERATURE_SEARCH_DATE,
        "upstream_hashes": selection_contract["upstream_hashes"],
        "generated_files": {
            relative_path(path): file_record(path)
            for path in generated
        },
        "rewrite_policy": (
            "05cmp0 is immutable. Comparator replacements or eligibility changes require a new "
            "versioned prospective amendment. Exact implementation versions/hyperparameters belong "
            "to 05cmp1 and must be frozen before scientific comparator runs."
        ),
    }
    write_json_atomic(lock, LOCK)

    print("-" * 88)
    print("Secondary comparator selection")
    print("-" * 88)
    print("Required external classical comparator: iClusterBayes2")
    print("Required external deep comparator: MOVE")
    print("Predeclared deep fallback only: AUTOENCODIX varix")
    print("DIABLO/supervised comparators in primary outcome-blind benchmark: EXCLUDED")
    print("MOFA duplicated as secondary comparator: NO (already frozen primary baseline)")
    print()
    print("Scientific comparator runs before 05e: FORBIDDEN")
    print("Synthetic 05dev0 smoke tests before 05e: ALLOWED")
    print("Comparator influence on 05e primary freeze: FORBIDDEN")
    print()

    print("=" * 88)
    print("Paper 5 secondary comparator selection contract: FROZEN")
    print("=" * 88)
    print(f"Candidate registry: {relative_path(CANDIDATE_REGISTRY)}")
    print(f"Selected registry: {relative_path(SELECTED_REGISTRY)}")
    print(f"Search manifest: {relative_path(SEARCH_MANIFEST)}")
    print(f"Selection contract: {relative_path(SELECTION_CONTRACT)}")
    print(f"Failure policy: {relative_path(FAILURE_POLICY)}")
    print(f"Benchmark policy: {relative_path(BENCHMARK_POLICY)}")
    print(f"Lock: {relative_path(LOCK)}")
    print("05d5 may continue independently: YES")
    print("Next comparator stage: 05cmp1 after synthetic implementation smoke tests.")
    print("=" * 88)


if __name__ == "__main__":
    main()
