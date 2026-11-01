# Complete-generation FFN observations collected

The fixed six-prompt capture completed on the M40 from clean release source
`d6f6ae9898db16b31956a3a07a6511de4b7ace95`. Every observer-off/on pair has
identical generated IDs, output and stop reason. These are unchanged dense
trajectories for offline predictability analysis, not sparse-generation
quality or performance results.

| Calibration task | Actual generated tokens | Consumed generated-input rows per layer | Stop |
| --- | ---: | ---: | --- |
| cal-chat-reminder | 21 | 20 | stop |
| cal-fact-depot | 11 | 10 | stop |
| cal-multi-delivery | 34 | 33 | stop |
| cal-config-retry | 2 | 1 | stop |
| cal-arithmetic-invoice | 3 | 2 | stop |
| cal-long-photo | 57 | 56 | stop |

All 128 actual generated tokens are retained in the report. Layers
0/16/17/34/35 each contain 122 contiguous consumed-token observations,
610 layer rows total. The six prompt-terminal states are outside this
observer and count as dense bootstrap in the study. Final emitted tokens
are not consumed after stopping. No capped prefix substitutes for a full
generation. Short tasks remain visible, including those with insufficient
history for particular causal comparisons.

The arithmetic and long-photo generations are shorter than the older M02
capture. This report uses their current actual counts; it does not assert
cross-revision output identity. The required same-executable observer pairs
match. No task is excluded or extended to obtain a favorable prediction rate.

FP32 observation payload is 31,856,640 bytes, with 250,675,200 bytes reserved
under the 256 MiB budget. Compact JSON is 94,559,097 bytes. Tracked device
allocation bytes are zero before and after the model owner; the final M40
snapshot shows zero MiB. Capture and serialization are untimed diagnostics.

The input artifact is retained outside Git at:

```text
/mnt/array-fastest/home/guyep/playground/2025-12-20-m40-llm/m40-llm/local-artifacts/qualified-references/selector-followup/ffn-inputs-01.json
SHA-256 3c8bc4f9f82db217c0dc38e690e0fafc4a46926ff7574b162946da7c435dd4cc
```

The [paired report](2026-10-31-selector-capture.jsonl.gz) records model,
corpus, execution and executable hashes. The [checksum manifest](2026-10-31-selector-capture.sha256)
covers that report, the large artifact, build/run/audit logs and
[the exact capture command](2026-10-31-selector-capture.sh). Reproduction
requires fresh artifact paths; the command refuses overwrites. A separate
JSON audit checks paired IDs/counts, release source identity, full row counts,
vector dimensions and payload accounting. The causal analyzer must also
validate task positions and source weights before accepting these observations.

Next: implement and test the frozen causal comparisons, then analyze this
capture without modifying its protocol or tuning against its results.
