Paper 5 05y8 - frozen 05y6 bytecode provenance archive
=======================================================

Purpose: preserve read-only evidence for the pre-existing 05y6 aggregator whose .py source is absent.
No 05y6 code is executed by this stage. No 05y5 shard or scientific result is read or recomputed.

Artifacts:
  05y6_pyc_header.json              parsed CPython .pyc header and embedded filename
  05y6_disassembly_readonly.txt     disassembly of the frozen code object, never executed here
  05y6_code_object_inventory.json   code-object metadata and bytecode hashes

Important: disassembly/inventory are provenance aids, not exact source-code reconstruction.
