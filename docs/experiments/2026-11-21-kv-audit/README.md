# Reading the November KV quality results

The full FP32 [replay](replay.py) checks aligned target IDs, finite values,
individual ranks, top-10 IDs, KL and logit errors against the raw arrays.
It requires the study completion marker and the entire 48-case grid.
Run it on the original JSONL beside its .logits directory with the frozen
tokenized input file as its second argument. Its counts exclude
dense-failing answers from claims about accuracy preservation.

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
