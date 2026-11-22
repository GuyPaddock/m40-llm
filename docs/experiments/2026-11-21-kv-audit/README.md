# Reading the November KV quality results

The full FP32 [replay](replay.py) checks aligned target IDs, finite values,
individual ranks, top-10 IDs, KL and logit errors against the raw arrays.
It requires the study completion marker and the entire 48-case grid.
Run it on the original JSONL beside its .logits directory with the frozen
tokenized input file as its second argument. Its counts exclude
dense-failing answers from claims about accuracy preservation.

The complete broad-grid package has 48 case shards. A second replay,
after verifying transport hashes and extracting the original files,
reproduced all 522 vocabulary vectors and 348 aligned observations.
The committed manifest pins the durable package and each file size.
[Full archive and reproduction](https://github.com/GuyPaddock/m40-llm/blob/d6cabf1cd0e0bd97df030eb47fbf96f613bb0722/docs/experiments/2026-11-21-kv-quality-result/full-replay/README.md).

## Dense reference audit

We read the 48 completed dense answers from research/1121-quantized-kv
c484536. The [frozen task set](../../../data/research_quality/2026-11-21-kv-tasks.json)
has SHA-256 f16c1104b0488d24b7e577c767811b931e6251b9603df52edb1fc4274621a5e2.
The retained dense first-pass.jsonl.gz has SHA-256
127962f165b357d0dcf650bb5d421067fc1dfd9c1e7851163d7b87e0bbf3edb2.
The frozen scoring rule gives 19/24 short and 10/24 long. For the four
structured answers, the project accepts Markdown-fenced JSON. A caller
which feeds the entire response to a JSON parser sees a different result.

| Dense comparison | Short | Long |
|---|---:|---:|
| Frozen scorer passes | 19/24 | 10/24 |
| Frozen scorer passes and any requested JSON parses as the expected complete object | 15/24 | 10/24 |
| Structured rows with all the correct fields | 4/4 | 4/4 |
| Structured rows usable directly by a JSON client | 0/4 | 4/4 |

All four short multi-fact records wrap the expected object in a code
fence. Each long response is a bare object. The official paired denominator
is 29. The supplementary client-format denominator is 25. Preserve both
denominators for every arm, including any case where compression happens
to correct a dense failure. This audit does not broaden those limited
tasks into a general judgment of reasoning or instruction following.

The other dense failures have several causes. In long ledger, platform,
rewrite and date-format responses, the answer is right but contains extra
words. Long SQL gives the correct full query when one keyword was requested.
Long loop uses exponentiation when the prompt explicitly requires
multiplication. The repeated list-order error on omit, both wrong Python
slices, and the box arithmetic failures reflect wrong answers. Long drawer,
cart ordering and shift time are wrong, and the long ticket-price
response reaches its cap mid-sentence. Keep them in the failed reference
set. Coding rows cover these four bounded prompts only.

### Review of all eight dense longer continuations

| Prompt | Short | Long |
|---|---|---|
| Lost notebook | 86 words. Advises against reading personal details, directs the visitor to reception and asks for the time found. | 82 words. Covers the same actions; recording the time after handover is less direct. |
| Potluck | 35 words. Welcomes guests and conveys the three ideas but falls short of 40 words and uses “labeled” and “clean up.” | 55 words. Includes all three ideas, again missing the expressly required words “labels” and “cleanup.” |
| Trail | 90 words. Covers locating the trailhead, recognizing a marker and retracing the route. | 81 words. Explains using a map, identifying markers and returning to a known marker or landmark. |
| Meeting | 48 words. Directly addresses the agenda, note-taking and deadlines. | 125 words. Gives three concrete instructions and a closing sentence. |

Word counts use whitespace. All eight generations stopped naturally. None
had a repeated suffix under the recorded token check. Only meeting-long
has a nonzero repeated four-gram fraction (0.0133). The six frozen
long-form passes communicate the requested actions on this checkpoint.
The two potluck failures stay excluded under the explicit vocabulary rule.

Use the same [whole-response check](raw-json.jq) for all candidate arms.
This extra diagnostic was introduced after the dense responses were
available. The predeclared scoring and regression rule remain authoritative.
The [machine-readable result](dense-format.json) records all dense cases.
To reproduce from the consolidation checkout:

```sh
study=docs/experiments/2026-11-21-kv-audit
gzip -dc ../quantized-kv/docs/experiments/2026-11-21-kv-dense/first-pass.jsonl.gz |
  jq -n --slurpfile rows /dev/stdin \
    --slurpfile tasks data/research_quality/2026-11-21-kv-tasks.json \
    --argjson arms '["dense"]' -f "$study/raw-json.jq"
```

For the final report pass its uncompressed JSONL to --slurpfile rows and
use the arm array `["dense","quantized_all","quantized_top8"]`. The check
rejects missing or duplicate task/context rows within any selected arm.
Review the corresponding compressed continuations for meaning and
degeneration, including new repetition absent from the dense answer.

## Preserving all vocabulary scores

The [archive tool](pack.sh) requires the final completion marker before
opening a destination. It runs the independent numerical replay, retains
all generation and host records, and stores each case's complete FP32
vectors in its own reproducibly ordered tar.zst. Every original file and
every transport file has a SHA-256 entry. The [restore tool](unpack.sh)
checks both manifests. A round trip on the completed ledger-short case
restored its nine vocabulary arrays and five report/provenance files
byte-for-byte. The incomplete broad grid was rejected before packaging.

```sh
bash "$study/pack.sh" ORIGINAL_RUN FROZEN_INPUTS NEW_BUNDLE
bash "$study/unpack.sh" NEW_BUNDLE NEW_REPLAY_DIRECTORY
python3 "$study/replay.py" NEW_REPLAY_DIRECTORY/report.jsonl NEW_REPLAY_DIRECTORY/inputs.json
```

### Independent generation and dispatch replay

The [generation audit](generation-audit.jq) checks every free-running
answer against the frozen scoring rules, reconstructs repetition from
emitted IDs, checks stop reasons, verifies the 36 physical KV lengths,
and counts the expected attention launches and 252 packed-prefill cuBLAS
calls in each generation. It reuses the [October independent functions](october_quality.jq)
from research/1031-kv-quality at 9da2390. The complete report path also
requires one completion per case, the final token total and released
tracked storage. These checks supplement the independent raw-logit replay.

The [dense replay](dense-generations.json) independently confirms all 48
answers, including the 29 passing rows. Altering a cache length, output
text, dispatch count, repetition length or stop reason invalidated the
dense replay. An attempt to submit the dense-only grid for full
qualification also failed. The full-grid archive runs the same checks on
all 144 outputs. The frozen concepts scorer enforces a minimum paragraph
count. A manual review must still assess audience, meaning and strict
instructions such as a request for exactly one paragraph.

## Independent operator timing replay

The [pair replay](operator-pairs.jq) recalculates the four throughput
medians in each screen, extension and confirmation from the individual
GPU-event and wall intervals. It checks alternating arm order, counts,
absence of allocations and host transfers inside timing, and the
coverage label on the dense-only control. Run it alongside the original
36-case grid and FP64 validation audit. The retained October report
reproduced [2,340 pairs](october-operator-pairs.json). Reversing the
order recorded for pair zero caused a rejection. An unfinished November
report also failed admission.

```sh
audit=docs/experiments/2026-11-21-kv-audit
out=$(mktemp -d)
git show e23d162:docs/experiments/2026-10-31-quantized-kv-timing-summary.jq > "$out/grid.jq"
jq -es -f "$out/grid.jq" COMPLETED_REPORT.jsonl > "$out/grid.json"
jq -es -f "$audit/operator-pairs.jq" COMPLETED_REPORT.jsonl
```

## Aligned logits beyond 64 historical blocks

`history.py` checks the four boundary cases using the same independent
FP32 reader and CPU probability calculations as the November quality audit.
It demands all sixteen generations, all sixteen dense-trajectory replays,
every expected comparison and file, the original model and prompts, and
zero tracked device bytes after model teardown. It checks the actual sequence
length in all 36 layer slots, recomputes old block counts and selection budgets,
and scores each emitted answer against its frozen exact-text or typed JSON
rubric. Byte equality is checked over the entire vocabulary at every sampled
position. The 2K full-mixed/top-64 control must also give identical free text
and generated IDs. The script emits per-case results only after the complete
cohort passes these checks.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 docs/experiments/2026-11-21-kv-audit/history.py \
    ../raw/kv-history-boundary-01/report.jsonl ../raw/kv-history-inputs.json
```

At capacity 4,096, [twelve native constructor records](history-crate-constructors.txt) across the first two
complete cases agree with the allocation formula for the three mixed arms.
The native total is 143,931,024 bytes with 32-token blocks and 138,615,696
with 64-token blocks. The 5,315,328-byte difference comes entirely from
the number of block summaries, counts and accumulators. The recent ring,
old K, packed old V and V scales are identical. Top-8 and top-64 both
allocate the same 143,931,024 bytes at width 32. The dense K/V payload
is 150,994,944 bytes. The Rust tracked-device counter was 4,472,663,048
bytes in all four arms. It cannot resolve these native allocation
differences or establish whole-process peak savings. The complete replay
also matches every constructor to a specific free-generation or
dense-trajectory record before accepting its cache accounting.

Use the boundary packager after the hardware runner exits. It requires
the final allocation check and the post-run device snapshot, verifies all
raw checksums after extraction, then repeats the numerical replay from
the archive. It publishes into a fresh directory after both replays match.

```bash
bash docs/experiments/2026-11-21-kv-audit/pack-history.sh \
    ../raw/kv-history-boundary-01 ../raw/kv-history-inputs.json ../raw/kv-boundary-package-01
```

On the 3,203-token relay task, all four arms generated 17 tokens. The
dense, full mixed and top-64 IDs matched throughout. Top-8 produced the
same port number, 23, and changed the Boolean from true to false. The
only differing ID was at zero-based generated position 13 (830 versus
895). The frozen prefix observations are at positions 0, 1, 8 and 16.
They support comparisons along the dense trajectory, but the logit margin
at position 13 was unmeasured. This lost dense-passing answer fails the
top-8 preservation gate. [Generation records](history-relay-generations.json).
