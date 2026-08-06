GSE39055 third human osteosarcoma cohort preparation
Script version: 25-gse39055-preparation-v1

Purpose
-------
1. Download and parse GSE39055 diagnostic-biopsy expression data.
2. Collapse probes to unambiguous gene symbols using highest probe variance.
3. Parse recurrence-free survival metadata without outcome-guided feature processing.
4. Construct the frozen canine-to-human module scores.
5. Defer all outcome association testing to the next script.

Primary future endpoint
-----------------------
Recurrence-free survival:
- time: GEO field 'time until first recurrence or latest follow-up (months)'
- event: recurrence Y/N

Important limitation
--------------------
Death status is available, but a separate time-to-death field is not supplied.
Therefore death is not incorporated into the primary time-to-event definition.
