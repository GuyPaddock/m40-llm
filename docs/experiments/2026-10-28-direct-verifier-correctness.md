# Fully Direct Four-Row Correctness Checkpoint

Clean implementation `83c59a831542617bf613b70060fb76f7fc171cc0`, release
executable SHA256
`7e922a3e4e95613254b6520d6bbf38c33880ee12d03b33c7fad6b3d83dae7de4`.
The [36-record report](2026-10-28-direct-verifier-correctness.jsonl)
passes the [approved contract](2026-10-28-direct-verifier-plan.md) before
primary timing. The same target schedule also passed the
[preliminary run](2026-10-28-direct-verifier-preliminary-correctness.md)
before cold payload auditing was expanded. These are separate reports, not
pooled trials.

## Observable Results

- All 512 actual greedy IDs match the frozen browser continuation, count and
  length stop. All 511 aligned vectors compare every one of 151,936 logits.
  Maximum and mean differences are zero; bitwise equality is diagnostic.
  The required tolerance remains `1e-3 + 2e-5*abs(reference)`.
- Every valid dense KV word matches serial decoding bit-identically, across
  36 layers and absolute positions 0 through 575. The 10,616,832 FP16 words
  hash to `63dd40de8aae9f6b8af712d91a6142d4cd3f17b549b8f50d42196aa924e70a36`.
- Mismatches at positions 0/1/2/3 and forced partial commits at 0/1/2/3 retain
  exact prefixes, clear rejected suffixes before correction, and produce the
  correct ordinary next token. Full acceptance is exercised throughout the
  browser continuation. Dropped attempts, unverified commits, invalid widths
  and IDs, aliases, stop caps and insufficient capacity are checked.
- All twelve frozen task outputs, actual IDs/counts and stop reasons match
  both serial and direct execution. Short task tails use ordinary serial
  consumers; the 512-token browser trace exercises 127 complete four-row
  batches and three serial tails.
- All 36 joint payload hashes match the prior joint-only ownership evidence.
  All 37 actual prepared down/head hashes match before and after this run.
  The 109 logical source identities, 37 physical split owners and 36 joint
  identities remain unchanged. Uploaded GGUF sources remain resident.
- Target work makes zero cuBLAS calls and zero device allocations/frees.
  It launches 4,572 shared-read gate/up kernels, 108 ordinary compound tail
  kernels, 18,396 qualified QKV calls, 18,396 qualified output calls and
  18,907 original prepared down/vocabulary calls. Session scratch is
  1,521,188 bytes, including an additional 831,520-byte four-row workspace.
  Tracked device bytes after final model teardown are zero.

The target D2H count in this correctness report includes full-logit diagnostic
readback. It is not a performance estimate. Prompt preparation retains the
qualified packed SGEMM path; target verification never invokes it.

## Checks

The [independent record audit](2026-10-28-direct-verifier-audit.pl) compares
frozen source/ownership identities and observable results, with twelve
corrupted-report rejection checks. Its
[result](2026-10-28-direct-verifier-correctness-audit.json) is reproducible:

```bash
perl docs/experiments/2026-10-28-direct-verifier-audit.pl \
  docs/experiments/2026-10-28-direct-verifier-correctness.jsonl --self-test
```

CUDA attention grid 11/11, compound 3/3, prepared consumers 2/2, fixed four-row
operator 2/2, forward fixtures 23/23 and transaction tests 3/3 pass serially
with hardware access. Transaction 3/3 and operator 2/2 also pass without
cuBLAS. Warning-as-error non-CUDA tests pass, 219 tests total. CLI/admission
tests 2/2, CUDA/server all-target Clippy and formatting pass. Logs are retained
with the final measurement archive.

All 137 original compiled CUDA entries and the existing four-row operator
remain instruction-identical to the preceding operator checkpoint. Its
[compiled audit](2026-10-28-direct-verifier-compiled-audit.json) is unchanged;
there is no new kernel. R4 remains 36 registers, zero stack/local memory and
4,608 dynamic shared bytes at 128 threads. These are static resources, not
measured occupancy or DRAM traffic.

## Admission And Limits

Timing is admitted only for this executable and frozen environment/model.
The direct session explicitly requires the validated Qwen2.5-3B dimensions,
joint-only compound ownership, packed-warp attention projections, original
prepared down/vocabulary, exclusive dense KV and four rows. Unsupported
configurations fail instead of invoking another verifier.

This checkpoint establishes correctness for the frozen continuations and
transactions, not a real drafter or an HTTP performance improvement. Runtime
defaults, the immutable 50.910426 HTTP tok/s reference and existing acceptance
rules are unchanged. Next: paired complete resident-generation measurements,
rejection costs and a measured feasibility assessment, then stop for review.
