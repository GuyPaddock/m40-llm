# Offline N-Gram Feasibility

C11A tests prompt lookup before implementing speculative verification. It does
not enable a drafter in generation or add a KV rollback/verifier API. The
selected C11 native-Q8/split8/256 configuration and runtime defaults are unchanged.

The [2026-10-25 measured acceptance checkpoint](experiments/2026-10-25-ngram-feasibility.md)
fails the formal gate: the best optimistic browser throughput gain is 4.949%,
not the required 10%. C12-C17 are postponed. This is a modeled cost screen,
not a measured GPU speedup or universal limit on speculation.

The concept is related to [llama.cpp prompt lookup](https://github.com/ggml-org/llama.cpp/blob/master/examples/lookup/README.md):
find a matching token suffix and propose its following tokens. Implementation
here independently follows the approved plan; no upstream code is copied.

## Lookup and Replay

`decode::ngram::propose` accepts only committed history, a remaining token
budget, and stop IDs. It searches suffix lengths 3-8, chooses the longest
usable match, and breaks ties by the most recent occurrence with a known
continuation. Returned tokens borrow history. No repetition is extrapolated
beyond known tokens, even when requesting sixteen candidates. Stop tokens
and the remaining output budget bound every draft.

The CPU study evaluates candidate limits 2/4/8/16 in two ways:

- Position-level opportunities rebuild history at every actual reference token.
- Visited-round replay advances by accepted prefix plus correction or bonus,
  without visiting the intermediate positions already committed by that round.

The reference future is available to scoring, never to lookup. Rejected draft
tails never enter history. Full acceptance at the output limit has no bonus;
accepted/corrective EOS stops without extra tokens. Post-EOS traces are rejected.
The tool verifies that replay reconstructs the complete reference continuation.

JSONL contains offered/accepted length histograms, eligibility, zero/partial/full
rejection counts and rates, no-match steps, CPU drafting nanoseconds, and every
visited round's source, offered/accepted/committed tokens and completion kind.
Single-run CPU lookup timings include timer overhead; they are not a controlled
GPU or runtime speculation benchmark.

## Costs and Gate

The browser input must be the selected release dense-KV C11 report, with stable
measured output, actual prompt/generated IDs, verified fresh KV, and five or
more valid screen trials. The tool rejects early EOS, mismatched counts,
prepared configurations, changed prompt IDs, and inconsistent output. Source,
model, configuration and analysis-executable hashes remain in the new report.

The cost model uses the browser's measured median HTTP and decode time, with
one ordinary target step calibrated as decode_ms / (generated_tokens - 1).
HTTP minus decode is kept fixed, retaining prefill and the timing residual.
Do not interpret that residual as a measured CPU stage: independent medians
and phase boundaries do not support that attribution.

For each simulated round, ordinary work costs one target-step unit and a draft
round costs 1.0/1.5/2.0/4.0 units. One unit is credited for the first prediction
already included in prefill; remaining modeled units are floored at zero.
Drafting, commit, rollback, transfers, and extra correction-processing costs
are zero in this optimistic screen. These are assumptions, **not measured
verification speed or a universal performance upper bound**. A real verifier
would still need the later C15 cost/parity gate.

Proceed to verifier feasibility only if a browser candidate estimates at least
10% HTTP E2E throughput improvement under the optimistic 1.0x model, with the
fixed regression checkpoint also present. Otherwise postpone C12-C17, record
C18, and proceed to C19. Repetitive synthetic data and secondary prompts cannot
override a failed browser gate.

## Capture and Run

Regression prompts are frozen before acceptance analysis: the existing Qwen
arithmetic/OK/config smokes plus one binary-search explanation and one Java
palindrome request. The latter two have 128-token limits and are not repetitive
synthetic justifications. Capture runs the canonical raw ChatML prompt twice,
requires identical actual IDs/output, and records natural stop reasons. Short
quality-smoke failures are errors. These are untimed in-process traces; no
HTTP speed conclusions come from their generation metrics.

Load the selected flags from the already committed C11 artifact:

```bash
source scripts/dev-env.sh
while IFS=$'\t' read -r key value; do
  export "$key=$value"
done < <(jq -r 'select(.kind == "manifest") | .environment |
  to_entries[] | [.key, .value] | @tsv' \
  docs/experiments/2026-10-25-qwen-exact-checkpoint.jsonl)

cargo build --offline --release --features cuda --example ngram_acceptance
target/release/examples/ngram_acceptance capture \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --output /tmp/ngram-regressions.jsonl
```

Request hardware access and ensure no other inference runs during capture.
Extra runtime overrides are rejected, not silently mixed with the selected
configuration. No model search, download, Ollama server, or network is needed.

The study itself needs no GPU:

```bash
cargo run --offline --release --no-default-features \
  --example ngram_acceptance -- study \
  --input docs/experiments/2026-10-25-qwen-exact-checkpoint.jsonl \
  --regression-traces /tmp/ngram-regressions.jsonl \
  --output /tmp/ngram-study.jsonl
```

Reports refuse overwrite. Omitting regression traces permits an exploratory
browser-only run but cannot authorize verifier work. Non-CUDA capture fails
clearly instead of reporting a successful skip. Debug study runs are rejected
because CPU drafting timings must use release builds.

CPU validation commands, through the project environment:

```bash
cargo test --no-default-features --test ngram_lookup
cargo test --no-default-features --example ngram_acceptance
```
