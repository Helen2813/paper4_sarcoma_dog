from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MASTER_ORTHOLOG_FILE = "GSE238110_RNA_master_candidate_evidence_table_with_orthologs.csv"
MODULE_MEMBERSHIP_FILE = "GSE238110_RNA_module_gene_membership.csv"
FULL_MODULE_ASSOC_FILE = "GSE238110_RNA_full_cohort_module_associations.csv"
MODULE_MAPPABILITY_FILE = "GSE238110_RNA_module_ortholog_mappability_summary.csv"

OUTPUT_MASTER_QC = "GSE238110_RNA_master_candidate_evidence_table_with_ortholog_qc.csv"
OUTPUT_STRICT_CANDIDATES = "GSE238110_RNA_strict_transferable_candidates.csv"
OUTPUT_BROAD_CANDIDATES = "GSE238110_RNA_broad_transferable_candidates_for_sensitivity.csv"
OUTPUT_MODULE_QC = "GSE238110_RNA_module_ortholog_qc_summary.csv"
OUTPUT_FULL_MODULE_PRIORITY = "GSE238110_RNA_full_cohort_transferable_module_priority.csv"

OUTPUT_STRICT_HUMAN_SYMBOLS = "GSE238110_strict_transferable_human_symbols.txt"
OUTPUT_STRICT_DOG_GENES = "GSE238110_strict_transferable_dog_genes.txt"


def read_csv_if_exists(filename):
    path = RESULTS_DIR / filename

    if not path.exists():
        print(f"Missing file: {path}")
        return pd.DataFrame()

    print(f"Loaded: {path}")
    return pd.read_csv(path)


def clean_gene_symbol(gene):
    gene = str(gene)
    tail = gene.rsplit("_", 1)[-1]
    return gene.rsplit("_", 1)[0] if tail.isdigit() else gene


def is_problematic_symbol(symbol):
    symbol = str(symbol).upper()

    prefixes = [
        "LOC",
        "ENS",
        "CFA",
    ]

    if symbol == "" or symbol == "NAN":
        return True

    if any(symbol.startswith(prefix) for prefix in prefixes):
        return True

    return False


def add_ortholog_qc(master):
    df = master.copy()

    if "gene_symbol_clean" not in df.columns:
        df["gene_symbol_clean"] = df["gene"].map(clean_gene_symbol)

    df["dog_symbol_upper"] = df["gene_symbol_clean"].fillna("").astype(str).str.upper()
    df["human_symbol_upper"] = df["human_gene_symbol"].fillna("").astype(str).str.upper()

    df["dog_human_orthology_confidence_numeric"] = pd.to_numeric(
        df.get("dog_human_orthology_confidence", np.nan),
        errors="coerce",
    )

    df["has_human_gene_symbol"] = df["human_gene_symbol"].fillna("").astype(str).str.strip().ne("")
    df["human_symbol_same_as_dog_symbol"] = (
        df["dog_symbol_upper"].eq(df["human_symbol_upper"]) &
        df["has_human_gene_symbol"]
    )

    df["dog_symbol_problematic"] = df["dog_symbol_upper"].map(is_problematic_symbol)
    df["human_symbol_problematic"] = df["human_symbol_upper"].map(is_problematic_symbol)

    df["ortholog_confidence_high"] = df["dog_human_orthology_confidence_numeric"].eq(1)

    if "is_one_to_one_ortholog" not in df.columns:
        df["is_one_to_one_ortholog"] = False

    if "has_human_homolog" not in df.columns:
        df["has_human_homolog"] = False

    df["is_one_to_one_ortholog"] = df["is_one_to_one_ortholog"].fillna(False).astype(bool)
    df["has_human_homolog"] = df["has_human_homolog"].fillna(False).astype(bool)

    df["broad_transferable_ortholog"] = (
        df["is_one_to_one_ortholog"] &
        df["has_human_gene_symbol"] &
        ~df["dog_symbol_problematic"] &
        ~df["human_symbol_problematic"]
    )

    df["strict_transferable_ortholog"] = (
        df["broad_transferable_ortholog"] &
        df["ortholog_confidence_high"]
    )

    df["strict_symbol_concordant_transferable"] = (
        df["strict_transferable_ortholog"] &
        df["human_symbol_same_as_dog_symbol"]
    )

    df["needs_manual_ortholog_review"] = (
        df["broad_transferable_ortholog"] &
        (
            ~df["ortholog_confidence_high"] |
            ~df["human_symbol_same_as_dog_symbol"]
        )
    )

    df["ortholog_qc_status"] = np.select(
        [
            df["strict_symbol_concordant_transferable"],
            df["strict_transferable_ortholog"] & ~df["human_symbol_same_as_dog_symbol"],
            df["broad_transferable_ortholog"] & ~df["ortholog_confidence_high"],
            df["is_one_to_one_ortholog"] & df["dog_symbol_problematic"],
            df["has_human_homolog"],
        ],
        [
            "strict_symbol_concordant_one_to_one",
            "strict_one_to_one_symbol_mismatch_review",
            "one_to_one_low_confidence_review",
            "one_to_one_problematic_dog_symbol_review",
            "non_strict_human_homolog_review",
        ],
        default="not_transferable_or_unmapped",
    )

    df["primary_human_validation_gene"] = np.where(
        df["strict_symbol_concordant_transferable"],
        df["human_gene_symbol"],
        "",
    )

    df["sensitivity_human_validation_gene"] = np.where(
        df["broad_transferable_ortholog"],
        df["human_gene_symbol"],
        "",
    )

    return df


def make_candidate_tables(master_qc):
    high_medium = master_qc[
        master_qc["rna_evidence_tier"].isin(["high_rna_evidence", "medium_rna_evidence"])
    ].copy()

    strict = high_medium[
        high_medium["strict_symbol_concordant_transferable"]
    ].copy()

    broad = high_medium[
        high_medium["broad_transferable_ortholog"]
    ].copy()

    sort_cols = [
        "rna_evidence_priority_score",
        "nested_any_max_selection_frequency",
        "dfi_univ_q",
    ]

    strict = strict.sort_values(sort_cols, ascending=[False, False, True], na_position="last")
    broad = broad.sort_values(sort_cols, ascending=[False, False, True], na_position="last")

    return strict, broad


def module_qc_from_membership(module_membership, master_qc):
    if module_membership.empty:
        return pd.DataFrame()

    module = module_membership.copy()

    if "gene_symbol_clean" not in module.columns:
        module["gene_symbol_clean"] = module["gene"].map(clean_gene_symbol)

    keep_cols = [
        "gene",
        "gene_symbol_clean",
        "human_gene_symbol",
        "strict_symbol_concordant_transferable",
        "strict_transferable_ortholog",
        "broad_transferable_ortholog",
        "needs_manual_ortholog_review",
        "ortholog_qc_status",
        "rna_evidence_tier",
        "rna_evidence_priority_score",
    ]

    keep_cols = [c for c in keep_cols if c in master_qc.columns]

    annotated = module.merge(
        master_qc[keep_cols],
        on="gene",
        how="left",
        suffixes=("", "_master"),
    )

    group_cols = ["analysis", "module_label"]

    if "endpoint" in annotated.columns:
        group_cols.append("endpoint")

    if "fold" in annotated.columns:
        group_cols.append("fold")

    rows = []

    for keys, part in annotated.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))

        n = part.shape[0]

        strict_symbol = part["strict_symbol_concordant_transferable"].fillna(False)
        strict_any = part["strict_transferable_ortholog"].fillna(False)
        broad = part["broad_transferable_ortholog"].fillna(False)
        review = part["needs_manual_ortholog_review"].fillna(False)

        high_medium = part[
            part["rna_evidence_tier"].isin(["high_rna_evidence", "medium_rna_evidence"])
        ].copy()

        strict_human_symbols = (
            part.loc[strict_symbol, "human_gene_symbol"]
            .dropna()
            .astype(str)
            .replace("", np.nan)
            .dropna()
            .drop_duplicates()
            .tolist()
        )

        broad_human_symbols = (
            part.loc[broad, "human_gene_symbol"]
            .dropna()
            .astype(str)
            .replace("", np.nan)
            .dropna()
            .drop_duplicates()
            .tolist()
        )

        row.update({
            "n_module_genes": n,
            "n_strict_symbol_concordant": int(strict_symbol.sum()),
            "n_strict_one_to_one": int(strict_any.sum()),
            "n_broad_transferable": int(broad.sum()),
            "n_manual_review": int(review.sum()),
            "fraction_strict_symbol_concordant": float(strict_symbol.mean()) if n else np.nan,
            "fraction_strict_one_to_one": float(strict_any.mean()) if n else np.nan,
            "fraction_broad_transferable": float(broad.mean()) if n else np.nan,
            "n_high_or_medium_rna_evidence": high_medium.shape[0],
            "strict_human_symbols": ";".join(strict_human_symbols),
            "broad_human_symbols": ";".join(broad_human_symbols[:300]),
        })

        rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out["module_transfer_qc_tier"] = np.select(
        [
            (out["fraction_strict_symbol_concordant"] >= 0.75) &
            (out["n_high_or_medium_rna_evidence"] >= 1),
            (out["fraction_strict_symbol_concordant"] >= 0.60) &
            (out["n_high_or_medium_rna_evidence"] >= 1),
            (out["fraction_broad_transferable"] >= 0.75),
            (out["fraction_broad_transferable"] >= 0.60),
        ],
        [
            "high_primary_transfer_readiness",
            "medium_primary_transfer_readiness",
            "high_sensitivity_transfer_readiness",
            "medium_sensitivity_transfer_readiness",
        ],
        default="low_transfer_readiness",
    )

    sort_cols = [
        "analysis",
        "endpoint",
        "module_transfer_qc_tier",
        "fraction_strict_symbol_concordant",
        "n_high_or_medium_rna_evidence",
    ]

    sort_cols = [c for c in sort_cols if c in out.columns]

    return out.sort_values(
        sort_cols,
        ascending=[True, True, True, False, False][:len(sort_cols)],
        na_position="last",
    )


def build_full_module_priority(module_qc, full_assoc):
    if module_qc.empty:
        return pd.DataFrame()

    full_qc = module_qc[module_qc["analysis"].eq("full_cohort")].copy()

    if full_qc.empty:
        return pd.DataFrame()

    if full_assoc.empty:
        return full_qc

    assoc = full_assoc.copy()

    assoc["p"] = pd.to_numeric(assoc["p"], errors="coerce")
    assoc["c_index"] = pd.to_numeric(assoc["c_index"], errors="coerce")

    pivot_rows = []

    for module_label, part in assoc.groupby("module_label"):
        row = {"module_label": module_label}

        for endpoint in ["DFI", "OS"]:
            ep = part[part["endpoint"].eq(endpoint)].copy()

            if ep.empty:
                continue

            ep = ep.sort_values("p", na_position="last").iloc[0]

            row[f"{endpoint.lower()}_full_module_p"] = ep.get("p", np.nan)
            row[f"{endpoint.lower()}_full_module_c_index"] = ep.get("c_index", np.nan)
            row[f"{endpoint.lower()}_full_module_gene_symbols"] = ep.get("gene_symbols", "")

        pivot_rows.append(row)

    assoc_pivot = pd.DataFrame(pivot_rows)

    out = full_qc.merge(assoc_pivot, on="module_label", how="left")

    out["dfi_full_module_p"] = pd.to_numeric(out.get("dfi_full_module_p", np.nan), errors="coerce")
    out["os_full_module_p"] = pd.to_numeric(out.get("os_full_module_p", np.nan), errors="coerce")

    out["transfer_priority_score"] = 0.0
    out["transfer_priority_score"] += (out["module_transfer_qc_tier"].eq("high_primary_transfer_readiness")).astype(float) * 4
    out["transfer_priority_score"] += (out["module_transfer_qc_tier"].eq("medium_primary_transfer_readiness")).astype(float) * 3
    out["transfer_priority_score"] += (out["n_high_or_medium_rna_evidence"] >= 1).astype(float) * 2
    out["transfer_priority_score"] += (out["dfi_full_module_p"] < 0.001).astype(float) * 2
    out["transfer_priority_score"] += (out["os_full_module_p"] < 0.01).astype(float) * 1
    out["transfer_priority_score"] += (out["fraction_strict_symbol_concordant"] >= 0.80).astype(float) * 1

    out = out.sort_values(
        ["transfer_priority_score", "fraction_strict_symbol_concordant", "n_high_or_medium_rna_evidence"],
        ascending=[False, False, False],
        na_position="last",
    )

    return out


def print_candidate_summary(master_qc, strict, broad, module_qc, full_module_priority):
    print("")
    print("=" * 80)
    print("Ortholog QC summary")
    print("=" * 80)

    print("All RNA evidence genes:")
    print(master_qc["ortholog_qc_status"].value_counts().to_string())

    print("")
    print("By RNA evidence tier:")
    tier = (
        master_qc
        .groupby("rna_evidence_tier", dropna=False)
        .agg(
            n_genes=("gene", "count"),
            n_strict_symbol_concordant=("strict_symbol_concordant_transferable", "sum"),
            n_strict_one_to_one=("strict_transferable_ortholog", "sum"),
            n_broad_transferable=("broad_transferable_ortholog", "sum"),
            n_manual_review=("needs_manual_ortholog_review", "sum"),
        )
        .reset_index()
    )

    tier["fraction_strict_symbol_concordant"] = tier["n_strict_symbol_concordant"] / tier["n_genes"]
    tier["fraction_broad_transferable"] = tier["n_broad_transferable"] / tier["n_genes"]

    print(tier.to_string(index=False))

    print("")
    print("=" * 80)
    print("Strict high/medium transferable candidates")
    print("=" * 80)

    display_cols = [
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
        "ortholog_qc_status",
    ]

    display_cols = [c for c in display_cols if c in strict.columns]

    if strict.empty:
        print("No strict high/medium transferable candidates were found.")
    else:
        print(strict[display_cols].head(80).to_string(index=False))

    print("")
    print("=" * 80)
    print("Broad high/medium transferable candidates needing review")
    print("=" * 80)

    review = broad[broad["needs_manual_ortholog_review"]].copy()

    review_cols = [
        "gene",
        "gene_symbol_clean",
        "human_gene_symbol",
        "rna_evidence_tier",
        "rna_evidence_priority_score",
        "dog_human_orthology_confidence",
        "human_symbol_same_as_dog_symbol",
        "ortholog_qc_status",
    ]

    review_cols = [c for c in review_cols if c in review.columns]

    if review.empty:
        print("No broad candidates requiring manual review.")
    else:
        print(review[review_cols].head(80).to_string(index=False))

    print("")
    print("=" * 80)
    print("Full-cohort transferable module priority")
    print("=" * 80)

    if full_module_priority.empty:
        print("No full-cohort module priority table was generated.")
    else:
        module_cols = [
            "module_label",
            "module_transfer_qc_tier",
            "transfer_priority_score",
            "n_module_genes",
            "fraction_strict_symbol_concordant",
            "fraction_broad_transferable",
            "n_high_or_medium_rna_evidence",
            "dfi_full_module_p",
            "dfi_full_module_c_index",
            "os_full_module_p",
            "os_full_module_c_index",
            "strict_human_symbols",
        ]

        module_cols = [c for c in module_cols if c in full_module_priority.columns]
        print(full_module_priority[module_cols].head(30).to_string(index=False))


def main():
    print("=" * 80)
    print("Ortholog QC and transferable set construction")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Results directory: {RESULTS_DIR}")
    print("")

    master = read_csv_if_exists(MASTER_ORTHOLOG_FILE)
    module_membership = read_csv_if_exists(MODULE_MEMBERSHIP_FILE)
    full_assoc = read_csv_if_exists(FULL_MODULE_ASSOC_FILE)
    existing_module_map = read_csv_if_exists(MODULE_MAPPABILITY_FILE)

    if master.empty:
        raise FileNotFoundError("Master ortholog table is missing.")

    master_qc = add_ortholog_qc(master)
    strict, broad = make_candidate_tables(master_qc)

    module_qc = module_qc_from_membership(module_membership, master_qc)
    full_module_priority = build_full_module_priority(module_qc, full_assoc)

    master_qc_path = RESULTS_DIR / OUTPUT_MASTER_QC
    strict_path = RESULTS_DIR / OUTPUT_STRICT_CANDIDATES
    broad_path = RESULTS_DIR / OUTPUT_BROAD_CANDIDATES
    module_qc_path = RESULTS_DIR / OUTPUT_MODULE_QC
    full_module_priority_path = RESULTS_DIR / OUTPUT_FULL_MODULE_PRIORITY

    strict_human_symbols_path = RESULTS_DIR / OUTPUT_STRICT_HUMAN_SYMBOLS
    strict_dog_genes_path = RESULTS_DIR / OUTPUT_STRICT_DOG_GENES

    master_qc.to_csv(master_qc_path, index=False)
    strict.to_csv(strict_path, index=False)
    broad.to_csv(broad_path, index=False)

    if not module_qc.empty:
        module_qc.to_csv(module_qc_path, index=False)

    if not full_module_priority.empty:
        full_module_priority.to_csv(full_module_priority_path, index=False)

    strict_human_symbols = (
        strict["human_gene_symbol"]
        .dropna()
        .astype(str)
        .replace("", np.nan)
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    strict_dog_genes = (
        strict["gene"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    with open(strict_human_symbols_path, "w", encoding="utf-8") as f:
        f.write("\n".join(strict_human_symbols))

    with open(strict_dog_genes_path, "w", encoding="utf-8") as f:
        f.write("\n".join(strict_dog_genes))

    print_candidate_summary(master_qc, strict, broad, module_qc, full_module_priority)

    print("")
    print("=" * 80)
    print("Interpretation guardrails")
    print("=" * 80)
    print("Use strict symbol-concordant one-to-one orthologs for primary human validation.")
    print("Use broader one-to-one mappings as sensitivity analyses after manual review.")
    print("Symbol-discordant mappings can be real orthology relationships but should not be treated as clean primary biomarkers without manual checking.")
    print("Large modules with high mappability are better suited for cross-species program transfer than unmapped LOC-heavy gene panels.")

    print("")
    print("Saved:")
    print(master_qc_path)
    print(strict_path)
    print(broad_path)
    print(module_qc_path)
    print(full_module_priority_path)
    print(strict_human_symbols_path)
    print(strict_dog_genes_path)
    print("Done.")


if __name__ == "__main__":
    main()
