GSE239948 WGCNA module-preservation benchmark
================================================

Script version
--------------
50-wgcna-module-preservation-benchmark-v1

Purpose
-------
Run the standard WGCNA modulePreservation() framework as a reviewer-facing benchmark for the frozen canine programs in DOG2 versus GSE239948.

Design
------
- Reference network: DOG2.
- Test network: GSE239948.
- Frozen transferable modules: M34, M11, M24, M40.
- Network type: signed.
- Permutations: 200.
- Benchmark gene universe: 3272 shared genes consisting of all shared frozen primary genes plus 3000 outcome-blind high-variance background genes when available.
- No outcome data are loaded.

Interpretation
--------------
WGCNA Zsummary.pres is interpreted using the conventional descriptive thresholds:
- Zsummary.pres < 2: no clear preservation evidence.
- 2 <= Zsummary.pres < 10: moderate preservation evidence.
- Zsummary.pres >= 10: strong preservation evidence.

MedianRank.pres is relative: lower ranks indicate stronger preservation compared with other reference modules and should not be interpreted using a universal cutoff.

Small-module guardrail
----------------------
M11 and M24 contain fewer than 10 shared frozen genes. WGCNA Z statistics are strongly module-size dependent, so low Zsummary values for these modules cannot establish absence of preservation.

Relationship to existing analyses
---------------------------------
This benchmark does not replace the custom ortholog-constrained preservation audit or the blind de novo rediscovery analysis. It provides an independent standard-framework sensitivity analysis based on the WGCNA modulePreservation() implementation.

R executable
------------
C:\Program Files\R\R-4.5.2\bin\x64\Rscript.exe
