# Frozen KV quality inputs

The CPU-only preparer implements the [protocol](2026-10-31-kv-quality-plan.md).
No GPU model or inference was used. It parsed the pinned GGUF metadata, checked
model/corpus SHA-256 identities, and generated the canonical prompt token IDs.
The original corpus, task rubrics and output bounds remain unchanged.

The [manifest](2026-10-31-kv-quality-inputs.json) has SHA-256
`e2e7ce56f0223bf929705dd49c33610ac01a0992837bb9c376bcea6dfc62b5de`.
Its preparation-source fingerprint matches `examples/kv_quality/fixture.rs`.

| Held-out task | Short prompt tokens | Long prompt tokens | Long task end token | Maximum consumed length, long |
|---|---:|---:|---:|---:|
| chat-tour | 87 | 2063 | 100 | 2110 |
| fact-observatory | 69 | 2064 | 82 | 2095 |
| multi-workshop | 110 | 2048 | 123 | 2111 |
| config-cache | 74 | 2050 | 87 | 2065 |
| arithmetic-crates | 58 | 2053 | 71 | 2068 |
| long-garden | 71 | 2066 | 84 | 2225 |

Task ends are measured from decoded token prefixes that cover the closing task
tag. They all precede the long prompt's recent-window boundary. Long cases use
100-103 whole archive notes. Short cases consume at most 230 tokens, entirely
inside the recent window. Even the longest allowed generation has at most
38 old blocks, so top64 covers every old block. At long prompt completion,
top8 cannot cover all old blocks. These are bounds, not measured model behavior.

Three behavioral CPU tests pass with warnings denied: task/rubric preservation,
manifest tamper rejection, and capacity/recent-window guard failures. Non-CUDA
example Clippy, formatting and whitespace checks pass. Raw preparation and
validation logs use the `2026-10-31-kv-quality-preparation` prefix.

Reproduce to a new destination after sourcing the project environment:

```bash
cargo run --release --no-default-features --example kv_quality -- prepare \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --output /tmp/new-kv-quality-inputs.json
```

The next commit will add the runner. No task scores, generated sequences,
quality conclusion or runtime promotion exists at this checkpoint.
