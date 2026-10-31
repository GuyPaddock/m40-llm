# Compact Export Evidence

`CompactExportEvidence::load` binds a proposed layer mask to a completed
evaluation. It requires the frozen calibration ranking, paired quality JSONL,
exact corpus file and requested removal count. It does not load a GPU model
or create output files.

The loader verifies source/mask/hash lineage, both split populations, original
prompts and rubrics, scores recomputed from recorded output, loss aggregates
and calibration token arrays. The applied mask's calibration loss must match
its winning frozen trial. First/last layers remain protected by the ranking
contract. Truncated reports, duplicate/missing items, mixed model transforms
and leaked tracked storage fail validation.

This checks consistency, not signed provenance or proof that the measurements
were honestly collected. A quality-evaluated mask is not quality-approved.
The returned summaries preserve regressions, loss and dense-failed inconclusive
rows. They never refit the mask or silently replace a failed output's rubric.

The source writer must still verify the actual GGUF's identity and dimensions,
preserve metadata and tensor bytes, and protect existing files. Numerical
parity with the masked source is a separate inference test. Export must never
be presented as restoring the unchanged model's quality.

CPU tests use synthetic evaluation records, retain deliberately bad outcomes,
and reject twelve incomplete/altered/mismatched claim variants. They do not
establish real model quality or GPU correctness.
