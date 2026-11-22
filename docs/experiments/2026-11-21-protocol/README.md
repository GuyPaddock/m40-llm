# November 21 measurement protocol

Freeze before candidate timing. Technical starting point is Phase 4 48c6333. Main e1403e0 is a detached clean comparison checkout. Docs-only commits can change the checkout identifier; record runtime blobs, build source and executable hashes separately. A1 inventory d9af4f4 traces both trees. The complete assignment lives in the parent workspace's m40-llm-research-plan-2026-11-21.md.

## Inputs, workload and endpoints

Use the Qwen Q8_0 blob in local-models.sha256 for every primary comparison. The other four checksummed GGUFs are distinct cohorts. Use TinyLlama F16 for the first compatible secondary-model check; compare its own control with the same weights. Use same-architecture Qwen Q4_0 only after verifying tensor/backend dispatch. Do not pool models. November briefs were not found in the project tree or project-memory search. The supplied primary sources are identified by head/base in upstream-prs.tsv; columns are number, source head, base, state, merge time, URL, title. The two papers are arXiv:2511.04477v1 (SpQt) and arXiv:2604.18556v3 (GSQ). The OpenCL, MoE, BF16 and NVFP4 results motivate a control or a dataflow question, not a Maxwell speed estimate.

Reuse examples/common/qwen_workload.rs and the existing compare_ollama protocol with its raw ChatML byte string, 65 prompt IDs, greedy top_k=1, max_tokens=512, context 1024, natural EOS and HTTP client wall boundary. Validate actual inputs/outputs and stops. Report differing main BOS behavior and its count separately if observed. Hash the binary, model and build flags. Use the pinned checksum-verified official Ollama 0.40.11 with isolated loopback port, no prompt cache, one worker and no other inference during oracle execution. A5 compares main's normal serving path, Phase 4 defaults, explicit dense/native, prepared family controls and the joint/portfolio research arms through existing entry points. HTTP/default and explicit-dense measurements require separate labels.

For the medium and long performance workloads, retain the same system text and final browser question, inserting numbered irrelevant archive lines before the question, to first reach 2048 or 8192 tokenizer IDs. Use the same frozen string for every arm, record its bytes/IDs and actual length; reserve 128 generated tokens for these runs. Keep the historical 512 limit for short. Use one warmup per arm and five alternating pairs, with KV reset and both arms resident only if the harness permits. For supported serving cohorts run 1, 2 and 4 buffered concurrent clients against a single server; also collect separately instrumented streaming results. Keep input length, generation counts, batch shape, prompt format and EOS behavior visible. Main and Phase 4 may expose different endpoint metrics. Measure actual TTFT and token spacing in an instrumented companion when streaming chunks do not delimit individual generated IDs.

## Diagnostics and decisions

Before each timed cohort and each benchmark invocation capture GPU UUID, clocks, temperature, memory, active PIDs, source SHA/dirty state, release flags, executable/model hashes, dispatch and exact deterministic input manifest. Check GPU availability throughout the series, isolate the CUDA 12.x sm_52 build and serialize every GPU run. Separate cold wall time, load/repack cost, steady HTTP, resident generation, GPU events, CPU submission/synchronization and tracked/live device memory. Record NVML peak sampling interval and attribution limits. Use stage probes and paired instrumentation off/on when needed. Unavailable service counters remain unknown. Keep completed raw rows and failed attempts.

Use docs/controlled_benchmark.md five alternating pairs; MAD/median over 5% extends once to ten. Use 10,000 paired bootstrap samples, seed zero, and separate twenty fresh confirmation pairs at the existing borderline criterion. General promotion requires at least 5% E2E, greater than twice the larger variation, positive lower paired bound. Memory-only promotion requires at least 10% lower whole-process peak, no lost dense passes and at most 3% wall slowdown. Retain historical specialized gates. For C1 require at least 5% lower complete target verification cost on a supported multi-row shape against the fastest correct existing control, positive lower bound and no correctness failure; label it shape-specific until live requests qualify. Rollback and numerical checks precede timing. Graph correctness can carry independently from speed. A missing or noisy primary measurement is inconclusive.

B2 approximations: freeze 24 new prompts × two contexts in addition to the original grid. Evaluate dense reference outcomes before candidate data, preserve all failures. Report aligned teacher-forced logit errors/KL/top-k, free-run divergence, answer/repetition and degeneration separately. Match staging/direct selected block IDs exactly. The direct mixed full-history and top64 arms coincide on grids with at most 64 old blocks; document the equality and include a capacity with more old blocks for distinct selection. Top8 is diagnostic.

Each topic must commit a hypothesis, scope, strongest control and stopping criteria before its own timing. Prioritize B1/B2/B3 and C1 with C2 parity. Screen C3–C5 and D1–D4 first. Log actual commands/raw outputs, never count hardware skips. Decisions use PASS/FAIL/INCONCLUSIVE at the scope measured. Keep closed predictors, ShiftAdd, generic fusion, unsupported rotation, lossless KV, Soft-MTP, DualPace and generic adaptive control closed unless the plan's documented reopening criteria are met. Produce one scoped decision row per component before any integration.

## Source and measurement integrity

The source inventory uses full Git hashes and a separate report SHA-256. Independent audits must replay raw rows, actual counts, stops, memory and quality metrics, and try malformed/mismatched records where appropriate. Store large replay inputs under durable local artifact storage and link hashes. Preserve all historical raw evidence. Do not modify runtime defaults. Final integrations require a separate authorization after the rebuild-from-main plan.

### Sampling token latency

For a diagnostic companion set M40LLM_RESEARCH_TOKEN_TIMINGS=1 and request include_metrics on the buffered generation endpoint. token_elapsed_ms contains one request-relative monotonic wall offset per sampled ID, including sampled stop tokens. Its first value is generation-admission TTFT; adjacent differences are the inter-token observations. Compute per-request median/P95 and report the pooled distribution separately. Client TTFT includes transport/queueing and still requires client observations. The ordinary buffered timing arm omits this setting. Compare it with the instrumented arm before interpreting diagnostics.

### November 21 host activity amendment

The operator reported a concurrent Gentoo system upgrade after the first native/split run. No time-aligned host series was captured for that run. Its paired interval measures within-cohort variability. It cannot estimate bias from persistent CPU contention, I/O, NUMA effects or host-side CUDA submission delays. Preserve the original measurements. Keep within-device correctness findings separate from latency qualification. Do not suspend the upgrade.

For remaining timed cohorts launch `bash docs/experiments/2026-11-21-protocol/host-load.sh "$$" > HOST.log &` from the host-visible benchmark shell. Capture CPU availability, /proc/loadavg, CPU counters and pressure, I/O and memory pressure, and counts of relevant compiler/package processes. Stop only the sampler owned by the cohort at teardown. Compare timestamps with each arm and GPU utilization/clocks/temperature. Keep all rows. Repeat borderline, load-correlated or host-sensitive promotion decisions in a quieter cohort. Preserve independent statistics rather than pooling unlike host conditions. Unknown background load is an explicit limitation for CPU, HTTP, cold-load and fairness conclusions.

### Frozen baseline requests

The checked-in performance-contexts.json contains the exact request bodies, full token IDs and SHA-256 of each UTF-8 prompt. Two independent invocations of the CPU tokenizer produced identical manifests. Short has 65 IDs and matches the retained independent Ollama token oracle. Medium first crosses 2,048 at 2,060 IDs after 105 archive lines. Long first crosses 8,192 at 8,197 IDs after 428 lines. The 1,024/2,304/8,448 capacities include the respective 512/128/128 output allowance. Medium and long use the runtime tokenizer with an exact text round trip. They have no independent external-tokenizer qualification yet.

Replay with cargo run --release --locked --example research_context_inputs -- --model MODEL --oracle-report docs/experiments/2026-11-21-native-split/report.jsonl --output NEW.json and compare against the checked-in manifest. Context capacity is separate from the prompt length. Preserve natural stop behavior and actual output counts for every measurement.

For an independent vocabulary check, run
`bash check-context-ids.sh LLAMA_TOKENIZE QWEN_Q8_GGUF NEW_DIRECTORY`
from this directory under the benchmark lock. The external llama.cpp tool
reads the same GGUF vocabulary and merges. Use raw bytes, preserve the
trailing newline, parse ChatML special tokens and omit automatic BOS.
Require every token ID to match in order in all three prompts. The result
directory retains the prompts, emitted IDs, binary and library hashes, and
per-case decisions. This check exercises tokenization only.

For later cohorts the process-name counts also include the kernel scheduling class and nice value. The initial KV run recorded counts by name only. A host process-tree observation during that run found the top-level emerge and current compiler workers under ordinary scheduling at nice 0, in separate process groups. The benchmark did not adjust their scheduling. Inspect each cohort's actual samples before classifying its load conditions.

For cohorts with timestamped before/after snapshots and the scheduling-class
sampler, run `python3 request-host-load.py RAW_DIRECTORY REQUEST_LABEL ...`
from this directory. Each group count is the number of samples in which
that process name, nice level, scheduling class and state appeared.
Pressure values are the maximum sampled ten-second averages over each
request. They include other host activity. The maximum unsampled interval
includes both request boundaries. An empty sample set gives no evidence
about process state. Even a sampled stopped emerge process cannot establish
what every descendant did between observations.
