# Buffered HTTP at one, two and four clients

Freeze before M40 measurements. Use the project's serial-versus-batch
scheduler comparison, the exact 65-ID Qwen Q8_0 short prompt, native Q8,
capacity 1024, seed 0, greedy top-1 and 64 sampled tokens or natural EOS.
Submit identical buffered requests simultaneously. Compare the shipped
generation mutex with batch decode and batch prefill enabled together,
four logical slots. Measure the extra resident cost of all four slots,
including in the single-client case. Run separate studies with the
shipped direct FP16-K/Q4-V cache and explicit dense KV. Each comparison
keeps its cache configuration fixed.

Use the D3 server with diagnostic timing disabled for primary client
wall measurements. For each pair alternate the order of the two server
configurations. Warm each with the same concurrency as its timed request.
Retain individual request responses, start/finish times, generated counts
and stop reasons. Count aggregate throughput from total sampled tokens
divided by the interval from the earliest submission to the last response.
Also retain each client's wall time, TTFT and inter-token gaps from a
separate run with token clocks and the scheduler trace enabled. Report
actual batch widths from that trace; the requested slot count alone
cannot establish that requests ran together. Use existing HTTP telemetry
and report current GPU processes, executable/model hashes, peak device
memory and host conditions.

For each concurrency require the primary outputs, counts and stops to
match their own diagnostic run. Compare the actual diagnostic token IDs
slot by slot across arms. Rank equivalent-work gains only if those checks
pass and multi-client trace confirms active batching. Record a speed or
quality difference separately when outputs diverge. Follow the shared
five-pair screen, dispersion extension and twenty-pair confirmation rule.
Stop if the mode is unsupported and record the actual server error. An
unsupported control cannot support a speed claim. This is a baseline of
the shipped server paths; a scheduler change requires separate D3 evidence.

## Reproduction

Use the primary D3 release server built from 29d1c32 (timers disabled).
Build 18b76c3 separately with
`M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 cargo build --locked --release --features cuda,server --bin m40-llm`
for the diagnostic server. It emits a row count after each successful
multirow target forward. Only the diagnostic uses that new server. Save
it to the adjacent raw/serving-concurrency-build-01/diagnostic-server.
Both executables are pinned in [binaries.sha256](binaries.sha256).
Build the CPU report with `CARGO_TARGET_DIR=target/cpu cargo build --locked --release --no-default-features --example research_fairness_report`.

Run `bash docs/experiments/2026-11-21-serving-concurrency/run.sh CACHE CLIENTS NEW_DIRECTORY`
under the common exclusive lock, once per choice of default/dense and
1/2/4 clients. All runs retain their primary client records, instrumented
companion, peak-memory timeline and independent arm timings, including
output-divergent runs. Use eligibility.json before interpreting paired
throughput. The two-client CPU [accounting check](check-accounting.sh)
reconstructed a 40 ms makespan and 100 aggregate tokens/s, checked TTFT
and P95 gaps, identified a divergent output and rejected invalid counts.
