# Controlled Qwen / Ollama Comparison

`compare_ollama` measures client-observed buffered HTTP time with resident
weights, a fresh prompt KV state, and the same Qwen2.5-3B Q8_0 GGUF bytes.
It does not change the runtime's general compressed-KV default. This short
workload explicitly uses full dense FP16 KV in both engines.

The [2026-10-24 baseline checkpoint](experiments/2026-10-24-qwen-baseline-checkpoint.md)
combines the repeated HTTP results, stage profile, counter availability, and
bounded optimization decisions. It distinguishes measured timings from
bandwidth/Amdahl estimates.

## Ollama Setup

Use an isolated server, not an existing server with unrelated traffic. The
2026-10-24 checkpoint uses the official prebuilt Linux x86-64
[Ollama v0.40.11 release](https://github.com/ollama/ollama/releases/tag/v0.40.11).
The downloaded `ollama-linux-amd64.tar.zst` SHA-256 is
`7bedab90dd122661be78798e9f0edf37fccf0cccf2699d2c58ea1a3afdc04b6d`.
The system's older Ollama installation is unchanged.

This release runs on the M40 through its CUDA 12 backend and bundled `sm_52`
PTX. Its CUDA 13 backend rejects this device. Allow for driver JIT and model
warmup during cold startup; a 60-second combined discovery/load probe timed
out, while a later 240-second-budget probe completed and generated `4`.
That probe is compatibility evidence, not a throughput measurement.

```bash
OLLAMA_HOST=127.0.0.1:18435 \
OLLAMA_MODELS="$HOME/.ollama/models" \
OLLAMA_DEBUG=1 \
OLLAMA_NUM_PARALLEL=1 \
OLLAMA_MAX_LOADED_MODELS=1 \
OLLAMA_KEEP_ALIVE=30m \
OLLAMA_NO_CLOUD=1 \
OLLAMA_NOPRUNE=1 \
LLAMA_ARG_CACHE_RAM=0 \
/tmp/m40-llm-tools/ollama-0.40.11/bin/ollama serve \
  > /tmp/m40-ollama-controlled.log 2>&1
```

Run that command in its own terminal and record its PID. Disabling the RAM
prompt cache prevents restoration of older measured prompts after the cache
replacement request. The tool also verifies the actual cache count and actual
prompt evaluation in runner diagnostics before accepting a trial. Neither
`prompt_eval_count` alone nor a nominally unrelated replacement prompt proves
that the full prompt was processed.

## Run the Comparison

Build and execute through the project environment with hardware access:

```bash
source scripts/dev-env.sh
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo build --release --features cuda,server --example compare_ollama

M40LLM_ENABLE_NVCC=1 \
M40LLM_ENABLE_CUBLAS=1 \
M40LLM_Q8_DECODE_STREAM=decode \
M40LLM_Q8_DECODE_SPLIT_QBLOCK=4 \
M40LLM_Q8_DECODE_COL_TILE=4 \
M40LLM_Q8_MLP_GATE_UP_SPLIT_QBLOCK=4 \
M40LLM_FUSED_Q8_QKV=1 \
M40LLM_FUSED_Q8_MLP_SWIGLU=1 \
M40LLM_CUDA_GREEDY_ARGMAX=1 \
M40LLM_Q8_LM_HEAD_ARGMAX=1 \
M40LLM_Q8_LM_HEAD_ARGMAX_SHARED_A=1 \
M40LLM_PROJECTION_BACKEND=large-model \
M40LLM_MATERIALIZE_F32_WEIGHTS=0 \
M40LLM_DECODE_GRAPH=0 \
target/release/examples/compare_ollama \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --ollama-pid OLLAMA_SERVER_PID \
  --ollama-log /tmp/m40-ollama-controlled.log \
  --output /tmp/m40-controlled-comparison.jsonl
```

The output file must not already exist. Requested hardware/model failures are
errors, not successful skips. The tool starts the normal m40-llm HTTP router on
an ephemeral loopback port and runs one request at a time. Both models may be
resident, but no inference work runs concurrently. Unknown compute processes
on the selected M40 invalidate the run.

The prompt includes the same explicit Qwen system message and ChatML assistant
prefix in both engines' raw-prompt interfaces. Before timing, actual Ollama
prompt token IDs must match the local tokenizer. Greedy `top_k=1` is argmax in
both engines; m40-llm omits temperature/top-p overrides to retain its existing
GPU greedy path. No sampling or natural stop is overridden to fill the budget.

Ollama 0.40.11 omits the historical `context` field for raw requests. The tool
uses its current native runner's `/tokenize` endpoint as the tokenizer oracle,
with special-token parsing enabled and BOS insertion disabled. Untimed
`/completion` calls request actual generated IDs with `return_tokens=true`;
their output must match the measured Ollama output. Timed requests still use
the public Ollama `/api/generate` route.

The initial matched setup exposed an extra-BOS bug in m40-llm's Raw/Auto
handling of preformatted Qwen ChatML: generation consumed 66 IDs while the
oracle and Ollama consumed 65. Commit `2f4d7ca` fixes this independently of the
benchmark. The mismatched attempt was rejected and does not support a speed
claim. Older 44-token results used a different rendered prompt and remain
historical, not matched measurements.

## Records and Decisions

The JSONL contains model and executable hashes, source revision/worktree state,
tensor shapes/dtypes, environment/build details, GPU conditions, cold-load
timing, tokenizer-oracle evidence, and individual warm HTTP trials. Every
trial records actual generated counts, stop reason, output, wall time,
prefill/decode timing, and cache evidence. HTTP timing ends after reading the
complete body, before parsing JSON. Internal generation timings are not a
substitute for this client wall time.

The initial screen contains five contemporaneous pairs in alternating order.
Variation is MAD divided by median throughput. Variation over 5% extends the
screen to ten pairs once; continuing instability blocks a speed conclusion.
All observations remain in the report.

An optimization must exceed `max(3%, 2 * larger timing variation)` and have a
positive lower bound on its 95% paired-bootstrap gain interval, using 10,000
resamples and seed zero. A point estimate within one percentage point of that
threshold, or above threshold with an interval crossing zero, schedules
exactly twenty fresh confirmation pairs. Screen and confirmation summaries
remain separate. Confirmation is not repeatedly extended until significant.

Only 512-generated-token, length-limited rows qualify for the fixed-workload
claim. Natural EOS and mismatched counts remain visible but stop the paired
comparison as invalid. Untimed companion generations save actual token-ID
traces and must reproduce the measured output. These traces support the
later history-only n-gram acceptance study.

Both unchanged-model targets still apply: at least 53.06 HTTP E2E generated
tokens/s and at least 1.30x the controlled Ollama median. Reaching neither is
not a reason to change the workload or report decode-only TPS as E2E.

Run the protocol tests without a GPU:

```bash
source scripts/dev-env.sh
cargo test --no-default-features --example compare_ollama
```

Long benchmark records belong in linked experiment artifacts. Dated summaries
and interpretations belong in [perf_baselines.md](perf_baselines.md).

## Check Stage-Profiler Perturbation

After building the same release example, keep the backend flags above and add:
Reuse every backend assignment from the full command; the abbreviated invocation
below omits those assignments, not their effect.

```bash
M40LLM_STAGE_PROFILE_TOKENS=0,255,511 \
M40LLM_STAGE_PROFILE_REPORT=/tmp/m40-stages.jsonl \
target/release/examples/compare_ollama \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --ollama-pid OLLAMA_SERVER_PID \
  --ollama-log /tmp/m40-ollama-controlled.log \
  --compare-stage-profile \
  --output /tmp/m40-profile-comparison.jsonl
```

Keep `M40LLM_STAGE_PROFILE` unset initially. This mode alternates five pairs
of ordinary and profiled **m40-llm** HTTP requests; it does not time Ollama
inference. Ollama is used only for the same model/tokenizer/oracle verification
before timing. Each engine invocation completes before the next begins. No
GPU tests or competing work should run during measurement.

The comparison checks equal output for every pair and actual generated-ID
equality in untimed companions. The HTTP report identifies profiling on/off.
The separate stage report contains one profiler session for the enabled warmup,
then the five enabled measured requests, then the enabled untimed companion.
Use measured sessions for attribution rather than mixing in warmup or traces.
This is an instrumentation-perturbation check, not an optimization-promotion
decision or a new Ollama speed comparison. See the
[stage timing caveats](diagnostics.md#launch-sync-and-timing-logs) before adding
stage or host-wait durations together.
