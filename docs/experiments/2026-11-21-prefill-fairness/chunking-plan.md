# Bounded dense prefill during active decode

Begin only if the five-pair baseline reports `PASS` with matched work
in its instrumentation-off series. The hypothesis is that returning to
the scheduler after 16 prompt positions bounds the active request's
longest inter-token gap without appreciably changing throughput or KV.
The independent diagnostic replay observed 344–351 s gaps while one
new 2,060-token prompt ran to its first sample. No candidate has been
timed. [Baseline diagnostic pairs](diagnostic/README.md).

Use the same Phase 4 dense scheduler, Qwen Q8 executable configuration,
four slots, context and request budgets as the baseline. Use a single
experimental switch with 16 positions per mixed tick. Run at most one
dense starting or partial prefill chunk in that tick. Defer the other
prefills in their existing queue order and continue the tick's active
decode. Use the existing per-session `processed_len()` and call
`logits_for_ids` on the absolute ID prefix through the chunk boundary.
Discard intermediate logits and sample once after the complete prompt.
Unmixed ticks can process a pending request to completion. For a partial
request the sequential path resumes at the cursor. Only fresh prompts
are eligible for whole-prefix packed prefill.

Start with behavior checks for one-token, exact-width and partial final
chunks, cancellation after a completed chunk and slot reuse. Include a
second pending request arriving during partial prefill and two ready
decodes. Use sequence-local lengths and compare valid dense KV and
generated IDs against the unchanged sequential arithmetic. Preserve the
existing compressed and streaming routing. Require all four short
clients in the ordinary dense cohort to return their original results.

Alternate five fresh baseline/candidate pairs in the mixed workload.
Record actual arrival during decode, all token clocks, makespan, per-request
TTFT, median/P95/maximum ITL, total sampled throughput and sampled process
peak. Trace each request through completion. In at least four valid
pairs require an 80% reduction in the short request's maximum ITL.
Require a positive paired confidence bound for that reduction, the same
generated IDs and stops, and median group makespan and medium request
TTFT each no more than 5% higher than baseline. Peak
resident model and KV owners should match. Record any allocator drift.
Measure a matched timing-off companion and quantify tracing cost.

Retain every trial. An output or cache failure stops immediately. If
the latency and throughput trade fails, keep the observed baseline stall
and close this policy. The study covers the measured opt-in scheduler
and workload. It supplies no default-serving or compressed-KV policy.

## Replay

Source `scripts/dev-env.sh` in this worktree. Build the release server
with `M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 cargo build --release
--locked --features cuda,server --bin m40-llm`. Build each library suite
with `cargo test --release --locked --features cuda,server --lib --no-run
--message-format=json`. Use NVCC both times, separate Cargo targets for
cuBLAS on and off, and retain the outputs as `test-artifacts.jsonl` and
`no-cublas-artifacts.jsonl`. Record the clean checkout's full commit in
`source.txt` and the completion time in `complete.txt`.

Under the shared M40 lock, run `bash
docs/experiments/2026-11-21-prefill-fairness/check-chunking.sh BUILD_DIRECTORY`.
Both selected tests must run and pass in both binaries before launching
`bash docs/experiments/2026-11-21-prefill-fairness/run-chunking.sh
NEW_OUTPUT_DIRECTORY BUILD_DIRECTORY`, also with the same lock held.
The comparison changes only `M40LLM_RESEARCH_DENSE_PREFILL_CHUNKS` between
arms on one executable. Fresh server processes warm with the short prompt
before each measurement. The calibration fixes the delay for both arms.
Recompute the decision from the captured raw files with `python3
docs/experiments/2026-11-21-prefill-fairness/score-chunking.py OUTPUT_DIRECTORY`.
The interval uses 10,000 bootstrap resamples of the valid paired
maximum-gap reductions with seed zero.

Run `bash docs/experiments/2026-11-21-prefill-fairness/replay-chunking.sh
OUTPUT_DIRECTORY RESEARCH_FAIRNESS_REPORT` on the finished archive.
It recalculates the diagnostic clocks from raw client responses, verifies
work equality across tracing modes, observes the recorded server flags
and reports per-arm NVML process peaks from startup through completion.
