GSE239948 independent canine representation validation
Script version: 46-gse239948-external-canine-representation-v2

Dataset
-------
Accession: GSE239948
Samples: expected 43 fresh-frozen canine osteosarcoma tumors.
Processed source: C:\Users\olegk\Desktop\paper4_sarcoma_dog\data\raw\canine_validation_GSE239948\GSE239948_CCOGC.txt.gz

Transformation
--------------
{
  "minimum": 3.0,
  "median": 194.395348837209,
  "maximum": 956034.744186047,
  "integer_like_fraction": 0.1651926994945313,
  "transform": "log2_x_plus_1"
}

Purpose
-------
Evaluate whether the frozen DOG2 programs preserve their transcriptional
representation in an independent canine osteosarcoma RNA-seq cohort.

No outcome model is fitted because GSE239948 does not provide a standardized
survival endpoint in the processed matrix.

Primary representation metrics
------------------------------
1. Spearman concordance of within-module gene-gene correlation edges between
   DOG2 and GSE239948.
2. Concordance between frozen DOG2 risk-oriented loadings and GSE239948 PC1
   loadings. The external PC1 sign is oriented using the frozen signed score,
   without outcomes.
3. Repeated non-overlapping split-half reliability of the frozen signed score.
4. Gene leave-one-out stability.
5. Variability-matched random-panel specificity controls.

Multiplicity
------------
BH correction is applied jointly across four edge-preservation and four
loading-concordance tests.

Guardrails
----------
- Frozen genes, loadings, signs, and tiers are unchanged.
- No GSE239948 outcome is used.
- GSE239948 tumors underwent different therapies; therefore treatment-related
  distribution differences are possible.
- Representation preservation is not prognostic validation.
