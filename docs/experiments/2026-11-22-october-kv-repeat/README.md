# Repeating the October KV quality grid on Maxwell

**PASS for reproducing the 12 October cases.** The November 22 run
matched the original on every deterministic field in all 86 records.
That includes 1,149 sampled tokens across 36 generations, the 24
dense/candidate comparisons and 104 matched-prefix vocabulary observations.
The comparison checks answer text, IDs, stop reasons, numerical
summaries, block counts, per-layer lengths, dispatch and allocation
records. Tracked device storage returned to zero.

| Context | Historical blocks | Dense passes | Passes retained by top-64 / top-8 | Complete ID traces matching dense, top-64 / top-8 |
|---|---:|---:|---:|---:|
| Short | 0 | 4/6 | 4/4 and 4/4 | 6/6 and 6/6 |
| Long | 33–37 | 3/6 | 3/3 and 3/3 | 5/6 and 5/6 |

All 33–37 long-context blocks fit within the top-64 budget. Both mixed
arms diverged from dense on the long garden continuation. Maximum
absolute matched-prefix logit error was 1.9711 for top-64 and 9.9841 for
top-8. Maximum KL was 0.03864 and 0.18875, respectively. Five cases
failed with dense KV and supply no evidence of answer preservation.
The separate November suite found three lost dense passes at top-8.
These runs qualify neither a default change nor an inference speedup.
[Broader generation review](../2026-11-21-kv-quality-result/continuations.md).

The original report is at `research/1031-kv-quality` commit
`9da2390a6e574a4185712d27ddf893143d94adc5`. The original executable
reported source `421df83e085212b78ca0d2e57b91a20ebb879e1e`.
The repeat ran from clean `e100c549e9f40654df3f8fb587f939ac69712ae7`
with release sm_52 binary SHA-256
`02b0eec975481a75c0b29217968bcd50a85bbed17892d9c76c77ca746fe631ea`.
The measured Qwen2.5-3B Q8_0 GGUF and frozen tokenized inputs have
identical hashes across runs. CUDA 12.8.93, cuBLAS on, native head128
and the full remaining environment are in the archived preflight.
The [comparison](comparison.json) binds the two source identities and
the repeat binary. It excludes clocks and elapsed hash, preparation and
generation times from the equality check. The [quality summary](quality.json)
retains the numerical observations by arm and context.

The [raw capture](measurements.tar.gz) includes preflight, input hash,
hardware and host records, stdout/stderr log, completion snapshot and full
JSONL. Its SHA-256 is
`89ab256039760765989ab27c4770fe8494a9bb22f4f64499cbe8ae344974f1d7`;
the original repeat JSONL SHA-256 is
`8abf89f83254f1d688e2f3b397397f4936231795c3af6b00fa8f1216a812931e`.
The October runner retains per-prefix numerical summaries. Full FP32
array recomputation belongs to the distinct expanded November corpus.

Rebuild the comparison from this checkout with
`bash docs/experiments/2026-11-22-october-kv-repeat/replay.sh NEW_DIRECTORY`.
It also executes the original October audit on the original inputs.
To repeat inference, source `scripts/dev-env.sh` and use
`bash docs/experiments/2026-11-21-kv-runs/run.sh october MODEL NEW_DIRECTORY`
under exclusive M40 ownership with the frozen input file at the path
recorded by the runner.
