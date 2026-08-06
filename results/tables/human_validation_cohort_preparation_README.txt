Human osteosarcoma cohort preparation

Script version: 22-human-cohort-preparation-v1

This script prepares TARGET-OS and GSE21257 without fitting any outcome model.
Frozen canine module membership, score direction, PCA weights, and validation tiers are not changed.

Primary external scores:
- strict one-to-one signed mean z-score for M34, M11, M24, and M40
- TARGET-OS: overall-survival metadata prepared from public GDC clinical fields
- GSE21257: metastasis-within-five-years label parsed from GEO metadata

Secondary/sensitivity scores:
- strict canine PCA-weighted score
- broad mapped score
- human-cohort PC1 oriented without outcomes
- M40 residual to a disjoint strict human proliferation PC1

No survival or metastasis association is tested in this script.
