from pathlib import Path
import io
import time
import urllib.parse
import urllib.request
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MASTER_FILE = "GSE238110_RNA_master_candidate_evidence_table.csv"
MODULE_MEMBERSHIP_FILE = "GSE238110_RNA_module_gene_membership.csv"

ORTHOLOG_CACHE_FILE = EXTERNAL_DIR / "ensembl_dog_human_orthologs_biomart.tsv"

OUTPUT_MASTER_ORTHOLOGS = RESULTS_DIR / "GSE238110_RNA_master_candidate_evidence_table_with_orthologs.csv"
OUTPUT_PRIORITY_TRANSFERABLE = RESULTS_DIR / "GSE238110_RNA_priority_transferable_ortholog_candidates.csv"
OUTPUT_MODULE_ORTHOLOG_SUMMARY = RESULTS_DIR / "GSE238110_RNA_module_ortholog_mappability_summary.csv"
OUTPUT_UNMAPPED = RESULTS_DIR / "GSE238110_RNA_candidate_genes_without_human_ortholog_mapping.csv"

BIOMART_URLS = [
    "https://www.ensembl.org/biomart/martservice",
    "http://www.ensembl.org/biomart/martservice",
]

DOG_DATASET = "clfamiliaris_gene_ensembl"

ATTRIBUTE_SETS = [
    [
        "ensembl_gene_id",
        "external_gene_name",
        "gene_biotype",
        "hsapiens_homolog_ensembl_gene",
        "hsapiens_homolog_associated_gene_name",
        "hsapiens_homolog_orthology_type",
        "hsapiens_homolog_orthology_confidence",
        "hsapiens_homolog_perc_id",
        "hsapiens_homolog_perc_id_r1",
    ],
    [
        "ensembl_gene_id",
        "external_gene_name",
        "hsapiens_homolog_ensembl_gene",
        "hsapiens_homolog_associated_gene_name",
        "hsapiens_homolog_orthology_type",
        "hsapiens_homolog_orthology_confidence",
    ],
    [
        "ensembl_gene_id",
        "external_gene_name",
        "hsapiens_homolog_ensembl_gene",
        "hsapiens_homolog_associated_gene_name",
        "hsapiens_homolog_orthology_type",
    ],
]


def clean_gene_symbol(gene):
    gene = str(gene)
    tail = gene.rsplit("_", 1)[-1]
    return gene.rsplit("_", 1)[0] if tail.isdigit() else gene


def build_biomart_xml(attributes):
    attribute_xml = "\n".join(
        [f'    <Attribute name="{attr}" />' for attr in attributes]
    )

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="1" count="" datasetConfigVersion="0.6">
  <Dataset name="{DOG_DATASET}" interface="default">
{attribute_xml}
  </Dataset>
</Query>
"""
    return xml


def query_biomart(url, attributes):
    xml = build_biomart_xml(attributes)
    encoded = urllib.parse.urlencode({"query": xml}).encode("utf-8")

    request = urllib.request.Request(
        url=url,
        data=encoded,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "paper4_sarcoma_dog_ortholog_mapping",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        text = response.read().decode("utf-8", errors="replace")

    if "Query ERROR" in text or "MartException" in text or len(text.strip()) == 0:
        raise RuntimeError(text[:1000])

    df = pd.read_csv(
        io.StringIO(text),
        sep="\t",
        header=None,
        names=attributes,
        dtype=str,
    )

    return df


def download_orthologs_from_ensembl():
    print("Attempting to download dog-human orthologs from Ensembl BioMart.")
    print(f"Dataset: {DOG_DATASET}")

    errors = []

    for url in BIOMART_URLS:
        for attributes in ATTRIBUTE_SETS:
            print("")
            print(f"Trying BioMart URL: {url}")
            print(f"Trying attributes: {len(attributes)}")

            try:
                df = query_biomart(url, attributes)

                if df.empty:
                    raise RuntimeError("BioMart returned an empty table.")

                required = {
                    "ensembl_gene_id",
                    "external_gene_name",
                    "hsapiens_homolog_ensembl_gene",
                    "hsapiens_homolog_associated_gene_name",
                    "hsapiens_homolog_orthology_type",
                }

                missing = required - set(df.columns)

                if missing:
                    raise RuntimeError(f"Missing required columns: {missing}")

                df.to_csv(ORTHOLOG_CACHE_FILE, sep="\t", index=False)

                print("")
                print("BioMart download succeeded.")
                print(f"Rows downloaded: {df.shape[0]}")
                print(f"Saved cache: {ORTHOLOG_CACHE_FILE}")

                return df

            except Exception as e:
                message = str(e).replace("\n", " ")[:500]
                errors.append({
                    "url": url,
                    "n_attributes": len(attributes),
                    "error": message,
                })
                print(f"Failed: {message}")
                time.sleep(2)

    print("")
    print("All BioMart attempts failed.")
    print("Errors:")
    for err in errors:
        print(err)

    print("")
    print("Manual fallback:")
    print("  1. Open Ensembl BioMart in a browser.")
    print("  2. Dataset: Canis lupus familiaris genes.")
    print("  3. Export these columns:")
    print("     ensembl_gene_id, external_gene_name, human homolog gene ID,")
    print("     human homolog gene name, orthology type, orthology confidence.")
    print(f"  4. Save as: {ORTHOLOG_CACHE_FILE}")

    raise SystemExit(1)


def load_or_download_orthologs():
    if ORTHOLOG_CACHE_FILE.exists():
        print(f"Using cached ortholog table: {ORTHOLOG_CACHE_FILE}")
        df = pd.read_csv(ORTHOLOG_CACHE_FILE, sep="\t", dtype=str)
        print(f"Cached ortholog rows: {df.shape[0]}")
        return df

    return download_orthologs_from_ensembl()


def normalize_ortholog_table(orthologs):
    df = orthologs.copy()

    for col in [
        "ensembl_gene_id",
        "external_gene_name",
        "gene_biotype",
        "hsapiens_homolog_ensembl_gene",
        "hsapiens_homolog_associated_gene_name",
        "hsapiens_homolog_orthology_type",
        "hsapiens_homolog_orthology_confidence",
        "hsapiens_homolog_perc_id",
        "hsapiens_homolog_perc_id_r1",
    ]:
        if col not in df.columns:
            df[col] = ""

    string_cols = [
        "ensembl_gene_id",
        "external_gene_name",
        "gene_biotype",
        "hsapiens_homolog_ensembl_gene",
        "hsapiens_homolog_associated_gene_name",
        "hsapiens_homolog_orthology_type",
        "hsapiens_homolog_orthology_confidence",
    ]

    for col in string_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["external_gene_name_upper"] = df["external_gene_name"].str.upper()

    df["hsapiens_homolog_perc_id"] = pd.to_numeric(
        df["hsapiens_homolog_perc_id"],
        errors="coerce",
    )

    df["hsapiens_homolog_perc_id_r1"] = pd.to_numeric(
        df["hsapiens_homolog_perc_id_r1"],
        errors="coerce",
    )

    confidence_raw = df["hsapiens_homolog_orthology_confidence"].replace("", np.nan)
    df["hsapiens_homolog_orthology_confidence_numeric"] = pd.to_numeric(
        confidence_raw,
        errors="coerce",
    )

    df["has_human_homolog"] = (
        df["hsapiens_homolog_ensembl_gene"].ne("") |
        df["hsapiens_homolog_associated_gene_name"].ne("")
    )

    df["is_one_to_one_ortholog"] = (
        df["hsapiens_homolog_orthology_type"].str.lower().eq("ortholog_one2one")
    )

    return df


def select_best_ortholog_per_symbol(orthologs):
    df = orthologs.copy()

    df = df[df["external_gene_name"].ne("")].copy()

    if df.empty:
        return pd.DataFrame()

    df["rank_one_to_one"] = np.where(df["is_one_to_one_ortholog"], 0, 1)
    df["rank_has_human"] = np.where(df["has_human_homolog"], 0, 1)

    df["rank_confidence"] = -df["hsapiens_homolog_orthology_confidence_numeric"].fillna(-1)
    df["rank_perc_id"] = -df["hsapiens_homolog_perc_id"].fillna(-1)
    df["rank_perc_id_r1"] = -df["hsapiens_homolog_perc_id_r1"].fillna(-1)

    df = df.sort_values([
        "external_gene_name_upper",
        "rank_has_human",
        "rank_one_to_one",
        "rank_confidence",
        "rank_perc_id",
        "rank_perc_id_r1",
    ])

    best = df.drop_duplicates("external_gene_name_upper", keep="first").copy()

    keep_cols = [
        "external_gene_name_upper",
        "ensembl_gene_id",
        "external_gene_name",
        "gene_biotype",
        "hsapiens_homolog_ensembl_gene",
        "hsapiens_homolog_associated_gene_name",
        "hsapiens_homolog_orthology_type",
        "hsapiens_homolog_orthology_confidence",
        "hsapiens_homolog_orthology_confidence_numeric",
        "hsapiens_homolog_perc_id",
        "hsapiens_homolog_perc_id_r1",
        "has_human_homolog",
        "is_one_to_one_ortholog",
    ]

    best = best[keep_cols].copy()

    rename = {
        "ensembl_gene_id": "dog_ensembl_gene_id",
        "external_gene_name": "dog_external_gene_name",
        "gene_biotype": "dog_gene_biotype",
        "hsapiens_homolog_ensembl_gene": "human_ensembl_gene_id",
        "hsapiens_homolog_associated_gene_name": "human_gene_symbol",
        "hsapiens_homolog_orthology_type": "dog_human_orthology_type",
        "hsapiens_homolog_orthology_confidence": "dog_human_orthology_confidence",
        "hsapiens_homolog_orthology_confidence_numeric": "dog_human_orthology_confidence_numeric",
        "hsapiens_homolog_perc_id": "dog_human_perc_id",
        "hsapiens_homolog_perc_id_r1": "human_dog_perc_id",
    }

    best = best.rename(columns=rename)

    return best


def annotate_master(master, best_orthologs):
    out = master.copy()

    out["gene_symbol_clean"] = out["gene"].map(clean_gene_symbol)
    out["gene_symbol_clean_upper"] = out["gene_symbol_clean"].astype(str).str.upper()

    out = out.merge(
        best_orthologs,
        left_on="gene_symbol_clean_upper",
        right_on="external_gene_name_upper",
        how="left",
    )

    out["has_human_homolog"] = out["has_human_homolog"].fillna(False).astype(bool)
    out["is_one_to_one_ortholog"] = out["is_one_to_one_ortholog"].fillna(False).astype(bool)

    out["ortholog_mapping_status"] = np.select(
        [
            out["is_one_to_one_ortholog"],
            out["has_human_homolog"],
            out["dog_external_gene_name"].notna(),
        ],
        [
            "one_to_one_human_ortholog",
            "non_one_to_one_human_homolog",
            "dog_symbol_found_no_human_homolog",
        ],
        default="dog_symbol_not_found_in_ensembl_mapping",
    )

    return out


def summarize_master(master):
    print("")
    print("=" * 80)
    print("Ortholog mapping summary")
    print("=" * 80)

    print("All RNA evidence genes:")
    print(master["ortholog_mapping_status"].value_counts().to_string())

    print("")
    print("By RNA evidence tier:")
    tier_summary = (
        master
        .groupby("rna_evidence_tier", dropna=False)
        .agg(
            n_genes=("gene", "count"),
            n_one_to_one=("is_one_to_one_ortholog", "sum"),
            n_any_human_homolog=("has_human_homolog", "sum"),
        )
        .reset_index()
    )

    tier_summary["fraction_one_to_one"] = tier_summary["n_one_to_one"] / tier_summary["n_genes"]
    tier_summary["fraction_any_human_homolog"] = tier_summary["n_any_human_homolog"] / tier_summary["n_genes"]

    print(tier_summary.to_string(index=False))

    print("")
    print("Top transferable candidates:")
    top = master[
        master["is_one_to_one_ortholog"] &
        master["rna_evidence_tier"].isin(["high_rna_evidence", "medium_rna_evidence"])
    ].copy()

    top = top.sort_values(
        ["rna_evidence_priority_score", "nested_any_max_selection_frequency", "dfi_univ_q"],
        ascending=[False, False, True],
        na_position="last",
    )

    cols = [
        "gene",
        "gene_symbol_clean",
        "human_gene_symbol",
        "human_ensembl_gene_id",
        "rna_evidence_tier",
        "rna_evidence_priority_score",
        "dfi_univ_q",
        "os_univ_q",
        "dfi_os_same_direction",
        "nested_any_max_selection_frequency",
        "dog_human_orthology_type",
        "dog_human_orthology_confidence",
        "dog_human_perc_id",
        "human_dog_perc_id",
    ]

    cols = [c for c in cols if c in top.columns]

    if top.empty:
        print("No high/medium RNA evidence genes had one-to-one human orthologs.")
    else:
        print(top[cols].head(50).to_string(index=False))


def annotate_modules(module_membership, master_with_orthologs):
    if module_membership.empty:
        return pd.DataFrame()

    keep_cols = [
        "gene",
        "gene_symbol_clean",
        "human_gene_symbol",
        "human_ensembl_gene_id",
        "has_human_homolog",
        "is_one_to_one_ortholog",
        "ortholog_mapping_status",
        "rna_evidence_tier",
        "rna_evidence_priority_score",
    ]

    keep_cols = [c for c in keep_cols if c in master_with_orthologs.columns]

    annotated = module_membership.merge(
        master_with_orthologs[keep_cols],
        on="gene",
        how="left",
        suffixes=("", "_master"),
    )

    group_cols = ["analysis", "module_label"]

    if "endpoint" in annotated.columns:
        group_cols.append("endpoint")

    if "fold" in annotated.columns:
        group_cols.append("fold")

    summary_rows = []

    for keys, part in annotated.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))

        human_symbols = (
            part.loc[part["is_one_to_one_ortholog"].fillna(False), "human_gene_symbol"]
            .dropna()
            .astype(str)
            .replace("", np.nan)
            .dropna()
            .drop_duplicates()
            .tolist()
        )

        high_medium = part[
            part["rna_evidence_tier"].isin(["high_rna_evidence", "medium_rna_evidence"])
        ].copy()

        row.update({
            "n_module_genes": part.shape[0],
            "n_human_homolog": int(part["has_human_homolog"].fillna(False).sum()),
            "n_one_to_one_ortholog": int(part["is_one_to_one_ortholog"].fillna(False).sum()),
            "fraction_one_to_one": float(part["is_one_to_one_ortholog"].fillna(False).mean()),
            "n_high_or_medium_rna_evidence": high_medium.shape[0],
            "human_one_to_one_symbols": ";".join(human_symbols[:200]),
        })

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    if not summary.empty:
        summary = summary.sort_values(
            ["analysis", "endpoint", "fraction_one_to_one", "n_high_or_medium_rna_evidence"],
            ascending=[True, True, False, False],
            na_position="last",
        )

    return summary


def main():
    print("=" * 80)
    print("Dog-to-human ortholog mapping for RNA candidates and modules")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"External data directory: {EXTERNAL_DIR}")
    print("")

    master_path = RESULTS_DIR / MASTER_FILE
    module_path = RESULTS_DIR / MODULE_MEMBERSHIP_FILE

    if not master_path.exists():
        raise FileNotFoundError(f"Missing master evidence table: {master_path}")

    master = pd.read_csv(master_path)
    print(f"Loaded master RNA evidence table: {master.shape}")

    if module_path.exists():
        module_membership = pd.read_csv(module_path)
        print(f"Loaded module membership table: {module_membership.shape}")
    else:
        module_membership = pd.DataFrame()
        print("Module membership table was not found. Module ortholog summary will be skipped.")

    orthologs = load_or_download_orthologs()
    orthologs = normalize_ortholog_table(orthologs)

    best_orthologs = select_best_ortholog_per_symbol(orthologs)

    print("")
    print("Best ortholog map:")
    print(f"  Dog symbols mapped: {best_orthologs.shape[0]}")
    print(f"  One-to-one orthologs: {int(best_orthologs['is_one_to_one_ortholog'].sum())}")
    print(f"  Any human homolog: {int(best_orthologs['has_human_homolog'].sum())}")

    master_annotated = annotate_master(master, best_orthologs)

    priority_transferable = master_annotated[
        master_annotated["is_one_to_one_ortholog"] &
        master_annotated["rna_evidence_tier"].isin(["high_rna_evidence", "medium_rna_evidence"])
    ].copy()

    priority_transferable = priority_transferable.sort_values(
        ["rna_evidence_priority_score", "nested_any_max_selection_frequency", "dfi_univ_q"],
        ascending=[False, False, True],
        na_position="last",
    )

    unmapped = master_annotated[
        ~master_annotated["has_human_homolog"]
    ].copy()

    module_summary = annotate_modules(module_membership, master_annotated)

    master_annotated.to_csv(OUTPUT_MASTER_ORTHOLOGS, index=False)
    priority_transferable.to_csv(OUTPUT_PRIORITY_TRANSFERABLE, index=False)
    unmapped.to_csv(OUTPUT_UNMAPPED, index=False)

    if not module_summary.empty:
        module_summary.to_csv(OUTPUT_MODULE_ORTHOLOG_SUMMARY, index=False)

    summarize_master(master_annotated)

    print("")
    print("=" * 80)
    print("Module ortholog mappability summary")
    print("=" * 80)

    if module_summary.empty:
        print("No module summary was computed.")
    else:
        cols = [
            "analysis",
            "endpoint",
            "fold",
            "module_label",
            "n_module_genes",
            "n_one_to_one_ortholog",
            "fraction_one_to_one",
            "n_high_or_medium_rna_evidence",
            "human_one_to_one_symbols",
        ]
        cols = [c for c in cols if c in module_summary.columns]
        print(module_summary[cols].head(50).to_string(index=False))

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("One-to-one ortholog genes are the strongest candidates for dog-to-human RNA transfer.")
    print("Non-one-to-one homologs may still be biologically relevant but should be treated cautiously.")
    print("LOC genes and unmapped genes should not be used as primary cross-species biomarkers unless independently resolved.")
    print("Next step: human RNA validation using one-to-one ortholog-mappable candidates/modules.")

    print("")
    print("Saved:")
    print(OUTPUT_MASTER_ORTHOLOGS)
    print(OUTPUT_PRIORITY_TRANSFERABLE)
    print(OUTPUT_MODULE_ORTHOLOG_SUMMARY)
    print(OUTPUT_UNMAPPED)
    print("Done.")


if __name__ == "__main__":
    main()
