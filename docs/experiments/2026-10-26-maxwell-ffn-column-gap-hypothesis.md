# Fixed FFN Payload Column Gap Diagnostic

## Question

Actual-plane checksum reads retain about 94% of qualified gate/up latency.
The thread-owned packet realizes fewer load instructions without a useful
complete-FFN gain. Stop changing load widths. Test the independent placement
question: both split payloads have a 2048-byte column stride in Qwen gate/up.
Does changing only that stride alter read cost enough to justify an inference
experiment? This does not establish undocumented cache/DRAM partition mapping.

## One Fixed Diagnostic

- Add exactly 128 unused bytes after each payload column. Keep both original
  scale planes and every original signed payload byte. Preserve 128-byte
  column alignment; no quantization, sparse weights or changed inference.
- Use a single new checksum kernel with dynamic payload pitch for both the
  zero-gap control and 128-byte-gap case. Keep original four-value tasks,
  128-thread ownership, u32 wrapping checksum, shared publication/reduction,
  load width and decode stream. Only pitch and payload pointers differ.
  Verify against original-block CPU sums and the existing checksum kernel.
- Retain original prepared gate/up, the additional gapped payload pair,
  source identities, guards and unchanged down/vocabulary in both arms.
  Original scales are borrowed from their live owners, never duplicated or
  reinterpreted. Track extra payload, padding, host setup and device bytes.
  Keep combined per-case preparation below 128 MiB and use RAII cleanup.
- Extend `q8_execution_bench --ffn-weight-read` with an exclusive optional
  `--ffn-weight-read-gap`. No runtime selector or inference output is added.
  Original diagnostic behavior remains unchanged when the flag is absent.
- Use frozen M02 actual layers 0/17/35, genuine rows 1/4, five alternating
  pairs, ten iterations. Each pair measures existing checksum, zero-gap
  pitched checksum, 128-byte-gap checksum, real gate/up, real down and real
  complete FFN serially. Warm real FFN for 1000 GPU ms and each diagnostic
  for 100 ms using existing finite budgets. Include enqueue counters and
  GPU process/clock boundaries; keep all other workloads out of timing.

## Validation And Decision

Test byte/inverse construction, arbitrary scale/payload bits, column guards,
odd outputs, legal dynamic K, distinct rows, exact CPU checksum equality,
same-kernel zero-gap equality, original-kernel equality, immutable planes,
ordered producers, invalid pitch/spans/aliasing, scratch reuse and cleanup.
Keep CUDA on/off, regression, non-CUDA warning-as-error, Clippy and formatting
checks. Inspect actual sm_52 resources/loads and unchanged original checksum,
qualified gate/up and prepared down/head instruction words.

Require at least 5% one-row read-cost improvement and positive paired lower
bounds at all three layers merely to justify considering a later inference
consumer. A read surrogate is not DRAM bandwidth, a rigorous latency lower
bound, complete-FFN speedup or HTTP qualification. Otherwise close this fixed
gap without a gap-size, geometry, cache or register rescue. No closed packet,
joint layout or virtual/persistent mapping is combined with it.

Even a positive diagnostic requires a separate inference implementation,
unchanged complete-FFN >=5% gate, actual-512/frozen quality/logit checks and
original HTTP gates. Both 53.06 HTTP tok/s and 1.30x fresh Ollama remain
required. Do not change project defaults or treat approximate computation
as satisfying the exact-output goal.

## Atomic Delivery

1. `docs(research): bound FFN column gap study`: this contract before code.
2. `feat(bench): probe FFN payload column gaps`: checksum-only API, owned
   bounded padding, shared-runner integration and behavior tests; depends on 1.
3. `docs(perf): record FFN column gap costs`: clean release evidence, compiler
   checks, raw/audited costs and decision; depends on tested 2. Preserve code
   before any later removal. No runtime integration is authorized here.
