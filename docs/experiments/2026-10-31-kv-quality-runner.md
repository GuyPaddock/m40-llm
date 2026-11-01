# KV quality runner validation

The separate `kv_quality run` example evaluates the committed input manifest
with the three cache arms in the [frozen plan](2026-10-31-kv-quality-plan.md).
It rejects changed model, corpus, inputs, preparation identity, runtime settings,
or uncommitted source. It checks actual attention launches, packed-prefill
cuBLAS calls, every layer's consumed length, and compressed old-block counts.
JSONL rows are flushed as completed; an explicit final row distinguishes a
complete grid from partial evidence. Existing evaluation APIs and runtime
defaults are unchanged.

All checks below completed before real-model inference:

- Warning-free non-CUDA full suite: 220 tests passed.
- Non-CUDA example: seven behavioral tests passed.
- Release CUDA example: all ten tests passed, including the ignored serial M40
  smoke. The synthetic two-layer model used 1,057 prompt IDs, packed prefill,
  three continuation IDs and both cache types. Top64/top8 with the same two
  old blocks gave identical logits. Changing the final target could not alter
  any prediction. Short inputs stayed within the existing forward tolerance;
  all tracked allocations were released.
- CUDA/server all-targets Clippy with warnings denied, Rust formatting, shell
  syntax, and Git whitespace checks passed.
- Frozen input SHA-256 remains
  `e2e7ce56f0223bf929705dd49c33610ac01a0992837bb9c376bcea6dfc62b5de`.

Raw logs use the `2026-10-31-kv-quality-runner-` prefix in this directory.
The initial Clippy failure is retained separately; its API type/method errors
were corrected in the first runner implementation before hardware validation.
The synthetic test validates the prefix pipeline, not real-model quality or
the complete generation wrapper. The real-model grid must satisfy those gates.

Run serially on an idle M40, with a new ignored or external output directory:

```bash
bash scripts/run-kv-quality.sh MODEL_PATH NEW_OUTPUT_DIRECTORY
```

The script records build, source, executable, device and process identities.
It freezes process-local environment settings and refuses a busy M40. Failed
runs retain their directory; use a distinct directory only after diagnosing a
terminal failure. Never restart a live run after a polling timeout.
