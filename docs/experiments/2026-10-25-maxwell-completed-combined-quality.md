# Combined FP32/128 and SGEMM Quality

After both [independent HTTP gains](2026-10-25-maxwell-completed-http.md)
qualify, the combined typed configuration passes fresh corrected-control
quality. Timing and Ollama-relative qualification remain separate gates.

Report revision `39de257`, quality executable SHA-256
`83cecd04d01ded613c1e4fc97df6065570ab0b80cc1c38a414f5e8c3ddb7fc01`.
Original Qwen2.5-3B Q8_0 weights, prepared down/vocabulary, dense context
1024, the frozen M02 environment and unchanged corpus/rubrics are retained.
The reference uses native FFN and repaired sequential prefill. The candidate
uses packed FP32/128 for one-row FFN and dequant SGEMM for the 64-token
prefix, with the final prompt token on the normal path. No activation
quantization, pruning, approximate attention, graph or scheduler participates.

- Both actual browser continuations have 512 tokens, natural length stops
  and identical IDs/output, with no divergence.
- All twelve task outputs match. Seven reference-valid tasks have zero
  regressions; five reference-failed tasks remain inconclusive.
- Seven forced-reference-prefix logit probes retain top-10 overlap 10.
  Maximum sampled difference is 0.011406; this is not bitwise equality of
  intermediate arithmetic.
- Prepared down/vocabulary stay at 1,192,935,424 bytes/model and dense KV
  at 37,748,736. The candidate records a 90,177,536-byte temporary weight
  capacity, 252 conversions/SGEMMs, no quantized-activation workspace and
  no full FP32 weight cache. Tracked allocations return from zero to zero.

The single untimed browser diagnostic reports 1591.32 / 504.01 ms prompt,
12728.65 / 11813.27 ms decode and 14320.01 / 12317.33 ms total for
control/candidate. These are not paired HTTP results or a speedup claim.
Packed-prefill quality deliberately does not reuse sequential passage loss
as evidence that the packed path was exercised.

The GPU starts empty after the isolated Ollama server exits. Inference is
serial, with no build/test or MemPalace overlap. Runtime source and both
executables are unchanged from the preceding validated checkpoint.

## Evidence and Next Gate

- [Raw generations, task scores, logits and dispatch](2026-10-25-maxwell-completed-combined-quality.jsonl)
- [Run output](2026-10-25-maxwell-completed-combined-quality.log.gz)
- [Checksums](2026-10-25-maxwell-completed-combined-quality.sha256)

Run controlled combined/native HTTP pairs, then a fresh selected-runtime
Ollama comparison. Neither original target is proved, and no runtime default
is changed by this quality checkpoint.
