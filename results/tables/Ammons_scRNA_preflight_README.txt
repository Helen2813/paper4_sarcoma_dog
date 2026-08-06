Ammons canine osteosarcoma single-cell atlas preflight
Script version: 41-download-preflight-ammons-single-cell-v2

Source
------
UCSC Cell Browser collection: canine-os-atlas
Resolved leaf dataset: canine-os-atlas/all
Resolved label: All Cells
Publication dataset accession: GSE252470

Why v2 was needed
-----------------
The public URL points to a Cell Browser collection. Collection directories
contain child datasets but do not themselves contain meta.tsv or an expression
matrix. The script now resolves the collection hierarchy through dataset.json
and selects the downloadable full leaf dataset before downloading files.

Detected metadata
-----------------
Cell-ID column: barcode
Candidate cell-type columns:
[
  "celltype.l1",
  "celltype.l2",
  "celltype.l3"
]
Candidate dog/sample columns:
[
  "orig.ident"
]

Purpose
-------
This preflight does not calculate module scores. It verifies:
- Python-readable expression and metadata files,
- cell identifier consistency,
- available cell-type and dog/sample annotations,
- frozen M34/M11/M24/M40 gene coverage,
- positive and negative loading coverage.

No clinical outcome is loaded.
