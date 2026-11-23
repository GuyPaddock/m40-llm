# Complete-request cost of thresholded packed Q4

**Frozen hypothesis.** The operator reduction on layers 0, 17 and 35
can reduce resident short-request latency enough to clear the November
5% throughput gate after the native dense Q4 path has warmed. The
expanded quality run must pass before any GPU timing.

Use the existing Q4 secondary baseline's raw Qwen chat request and
input IDs. Its manifest SHA-256 is
`a3d3b51c9ce58eb893a74d0efece48f49a112e0a290ce457ddbcdf00530288c1`.
Keep the pinned Qwen2.5-3B Q4_0/Q6_K checkpoint, all 65 prompt IDs,
dense FP16 KV at capacity 1,024, greedy seed zero and the 32-token
cap with natural stops. Dense uses the native down matrices. Packed
uses the admitted three layers and the exact expanded-quality cutoffs.
The harness installs the research owner through the existing loader
API, then invokes the normal `generate_text` path with sequential
prefill. Time prefill and decode together after a full workload warmup.
The time excludes cold load and HTTP transport. Record cold loading,
owner preparation, requested allocations and sampled process VRAM
separately. The script starts a fresh release CUDA process per arm.

Capture model and executable checksums, exact source, flags, initial
GPU state and five-second host observations. The release image must
contain `sm_52`. Execute one pair at a time under the shared lock.
Alternate first-arm order across five pairs. Verify the tokenizer,
accepted input, generated IDs, output length, stop reason, prepared
layer use and tracked teardown. A different greedy output makes timing
comparison **INCONCLUSIVE** and ends the series at that pair. Keep
both observations. Use the common paired bootstrap and its bounded
extension and confirmation rules. A result below the gate ends this
configuration without changing the cutoff or adding layers. Even a
passing resident result remains specific to this model and workload.

From this worktree, source `scripts/dev-env.sh`, then build a release
`model_quality` example with `M40LLM_ENABLE_NVCC=1`,
`M40LLM_ENABLE_CUBLAS=1`, `--locked` and `--features cuda`. Use the
existing consolidation `research_http_pairs` binary for the statistics.
The adapter's `http_seconds` field holds resident request seconds.
Run with the finished expanded-quality directory and its replay.

```sh
flock -x ../raw/m40-benchmark.lock bash docs/experiments/2026-11-23-sparse-q4-request/run.sh \
    QWEN_Q4_GGUF MODEL_QUALITY RESEARCH_HTTP_PAIRS FINISHED_QUALITY QUALITY_REPLAY NEW_RAW_DIR
```
