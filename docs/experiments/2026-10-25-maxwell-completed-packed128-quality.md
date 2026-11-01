# FP32/128 Quality Against the Corrected Control

Fresh release `dc7b234` requalifies the explicit packed FP32/128 FFN
candidate after the sequential prefix ownership repair. This is full-model
quality evidence, not a new HTTP performance result or runtime default.

Executable SHA-256:
`83cecd04d01ded613c1e4fc97df6065570ab0b80cc1c38a414f5e8c3ddb7fc01`.
The manifest records `prefix-decode-completed-v1`, original Qwen2.5-3B Q8_0
weights, prepared down/vocabulary, dense context 1024, native sequential
prefill and the frozen M02 environment. The quality corpus and both splits
are unchanged. No other GPU job, build, test or MemPalace embedding overlaps
this run.

## Results

- Both production greedy browser runs generate 512 actual tokens, stop at
  length, and preserve all IDs/output with no divergence. The corrected
  continuation begins `Sure, I'll walk you through...`, not the historical
  unsafe sequential continuation.
- All twelve frozen task outputs match. Seven reference-valid tasks have
  zero candidate regressions; five reference-failed tasks remain outside
  compression/engine quality conclusions. No prompt or rubric is tuned.
- Seven forced-reference-prefix full-logit probes retain top-10 overlap 10.
  Maximum sampled logit difference is 0.011219; the largest sampled mean
  difference is 0.000666. This arithmetic is not bit-identical despite
  preserving the tested actual generations.
- Calibration PPL is 31.556846 reference / 31.554371 candidate over 102
  teacher-forced tokens; held-out is 39.612157 / 39.611313 over 101 tokens.
  Passage logits use the labeled full-logit path, not a claim that the
  GPU-greedy head has identical internal arithmetic.
- Each browser run reports 250,369 kernel launches, five allocations/frees
  and 1155 stream synchronizations. The count includes the new 64 prefix
  completion boundaries. No kernel, copy or allocation was added by them.
  Both retain 1,192,935,424 prepared weight bytes and 37,748,736 dense KV
  bytes, with no full FP32 cache. Tracked allocations return from zero to zero.

The exact quality gate passes. The earlier +8.133% HTTP result belongs to
the old unsafe control and is not reused as corrected-runtime timing.
Collect fresh controlled HTTP pairs after completing the independent packed
prefill quality checks. The full objective and both original speed targets
remain incomplete; no runtime default changes.

## Evidence

- [Full report, actual IDs/output, logits, tasks and loss](2026-10-25-maxwell-completed-packed128-quality.jsonl)
- [Run log](2026-10-25-maxwell-completed-packed128-quality.log.gz)
- [Checksums](2026-10-25-maxwell-completed-packed128-quality.sha256)
- [Ownership repair and matched real-model attribution](2026-10-25-maxwell-prefill-drift-completed.md)

Next regenerate native-packed and dequant-SGEMM quality reports against this
same corrected control. Do not modify old reports to add the new contract
or claim packed full-continuation parity from a first-token match alone.
