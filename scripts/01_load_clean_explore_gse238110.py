"""
Step 01 - Load, clean and explore GSE238110 (canine osteosarcoma RNA-seq).

Input:  raw count matrix downloaded from GEO (GSE238110_RawCountFile_combined.csv.gz)
Output: cleaned expression matrix + QC report saved to data/processed/

Run this first, before any feature selection or modeling. The goal here is
only to understand what the file actually contains and produce a clean
matrix that later scripts can rely on.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# PATHS - auto-detected, no editing needed
# ---------------------------------------------------------------------------
# Assumes this script lives in: <PROJECT_ROOT>/scripts/01_load_clean_explore_gse238110.py
# So the project root is just one level up from this file's folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "canine_discovery_GSE238110" / "GSE238110_RawCountFile_combined.csv"
OUT_MATRIX = PROJECT_ROOT / "data" / "processed" / "GSE238110_counts_clean.csv"
OUT_QC_REPORT = PROJECT_ROOT / "data" / "processed" / "GSE238110_qc_report.txt"

# make sure the output folder exists
(PROJECT_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------
# pandas reads .csv.gz directly, no need to unzip manually
df = pd.read_csv(RAW_FILE, compression="infer")

# GEO count matrices are usually genes x samples with the gene ID as the
# first column. We standardize this to a proper index.
first_col = df.columns[0]
df = df.set_index(first_col)
df.index.name = "gene_id"

n_genes_raw, n_samples_raw = df.shape

# ---------------------------------------------------------------------------
# BASIC QC CHECKS
# ---------------------------------------------------------------------------
report_lines = []
report_lines.append(f"Raw matrix shape: {n_genes_raw} genes x {n_samples_raw} samples")

# 1. Check for duplicate gene IDs
n_duplicate_genes = df.index.duplicated().sum()
report_lines.append(f"Duplicate gene IDs: {n_duplicate_genes}")

# 2. Check for duplicate sample columns (e.g. accidental re-download merge)
n_duplicate_samples = df.columns.duplicated().sum()
report_lines.append(f"Duplicate sample columns: {n_duplicate_samples}")

# 3. Check for missing values
n_missing = df.isna().sum().sum()
report_lines.append(f"Missing values (NaN): {n_missing}")

# 4. Check for non-numeric columns that shouldn't be there
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
report_lines.append(f"Non-numeric columns found: {non_numeric_cols}")

# 5. Check value range - raw counts should be non-negative integers
min_val = df.select_dtypes(include=[np.number]).min().min()
max_val = df.select_dtypes(include=[np.number]).max().max()
report_lines.append(f"Value range: min={min_val}, max={max_val}")

# 6. Library size per sample (total counts) - flags samples with very low depth
library_sizes = df.sum(axis=0)
report_lines.append(f"Library size range: min={library_sizes.min():.0f}, "
                     f"median={library_sizes.median():.0f}, max={library_sizes.max():.0f}")

low_depth_threshold = library_sizes.median() * 0.1
low_depth_samples = library_sizes[library_sizes < low_depth_threshold].index.tolist()
report_lines.append(f"Samples with library size < 10% of median ({len(low_depth_samples)}): "
                     f"{low_depth_samples}")

# 7. Genes with zero counts across all samples (uninformative, safe to drop)
zero_genes = (df.sum(axis=1) == 0).sum()
report_lines.append(f"Genes with zero counts in all samples: {zero_genes}")

# 8. Genes expressed in very few samples (common low-expression filter)
min_samples_expressed = 0.2 * n_samples_raw  # expressed in at least 20% of samples
genes_expressed = (df > 0).sum(axis=1)
low_expression_genes = (genes_expressed < min_samples_expressed).sum()
report_lines.append(f"Genes expressed in <20% of samples: {low_expression_genes}")

# ---------------------------------------------------------------------------
# CLEANING (conservative - only removes clearly uninformative rows)
# ---------------------------------------------------------------------------
df_clean = df.copy()

# drop duplicate gene rows, keep first occurrence
if n_duplicate_genes > 0:
    df_clean = df_clean[~df_clean.index.duplicated(keep="first")]

# drop genes with zero counts everywhere
df_clean = df_clean.loc[df_clean.sum(axis=1) > 0]

n_genes_clean, n_samples_clean = df_clean.shape
report_lines.append(f"Clean matrix shape: {n_genes_clean} genes x {n_samples_clean} samples")
report_lines.append("Note: no variance filtering or normalization applied yet - "
                     "that happens in the next script, after clinical data is merged "
                     "and sample IDs are matched.")

# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------
df_clean.to_csv(OUT_MATRIX)

with open(OUT_QC_REPORT, "w") as f:
    f.write("\n".join(report_lines))

print("Done.")
print(f"Clean matrix saved to: {OUT_MATRIX}")
print(f"QC report saved to: {OUT_QC_REPORT}")