# R0 CPU recovery and R1 readiness

## Decision

**R1 is blocked. Stop for review.** There is no selected training manifest
and no authorization to run calibration, teacher generation or training.
The accepted V2 protocol remains a plan, not measured short-context evidence.

The bounded grounded-QA search recovered two historical document matches,
not enough provenance to establish the unchanged quotas. A separate necessary
gate now fails conclusively: only **18 of 4,451 Dolly prose/open-QA prompts**
have 65-447 runtime tokens. That stratum requires 256 training, 48 calibration
and 96 held-out prompts. Even perfect provenance and zero duplicate exclusions
cannot turn 18 prompts into 400. Do not pad prompts, append answers, substitute
categories, reuse rows across splits or transfer quotas.

The [machine-readable assessment](2026-10-30-drafter-r0-audit.json) and
[independent census checks](2026-10-30-drafter-r0-audit.jq) record this stop.
Original failed admission, specifications and V2 artifacts are unchanged:
all 27 original and 16 V2 checksum entries still verify.

## CPU scope and evidence

Implementation commits are `fdf8515` for source recovery/runtime tokenization
and `1da88f8` for span, template and exclusion audits. These tools do not load
a CUDA context or execute source reference programs.

All 15,975 original source records were tokenized with the original Qwen
tokenizer and production QwenChat formatter, without BOS or a system message.
Each formatted prompt roundtrips exactly. MBPP uses its unchanged statement;
Dolly uses instruction plus `\n\nContext:\n` and nonempty context. The response
is never input. Comparison normalization is separate from emitted prompt text.
The no-default-features example checks the complete model SHA256 before use.
Its build lives in `target/drafter-data-cpu`, separate from measured binaries.

Raw records are under
`local-artifacts/2026-10-30-drafter-recovery/`. The
[checksum manifest](2026-10-30-drafter-r0.sha256) binds source cards/payloads,
original and V2 specifications, source/token ledgers, all new tools, recovery
responses and final reports. The complete local Dolly candidate ledger has
15,011 rows; the MBPP component ledger covers all 964 official task IDs.
These are **unselected quarantine ledgers**, not a realized training manifest.

## Dolly document recovery

The source is the unchanged
[pinned Dolly revision](https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a).
Its four fields contain no upstream document identifier. A context hash
identifies a passage, not a document; titles alone do not establish identity.
Source/license provenance stays CC-BY-SA-3.0 with Databricks and applicable
Wikipedia attribution. This checkpoint does not grant redistribution or
commercial clearance for the target or derived features.

The bounded probe froze 16 length-eligible grounded-QA rows by seeded hash.
Each queried the first ten normalized passage words, considered at most two
search results, and sought a revision no later than April 12, 2023. Admission
to the recovered-map candidate list required the entire normalized passage,
at least 32 words, to occur contiguously in exactly one fetched candidate.
HTML is parsed as data; scripts, styles and citation markers are excluded.

| Dolly source line | Reconstructed document | Historical revision | Passage match |
| --- | --- | --- | --- |
| 14914 | `enwiki:63030231`, COVID-19 | [1148746377](https://en.wikipedia.org/w/index.php?oldid=1148746377), 2023-04-08 | 192/192 words |
| 6620 | `enwiki:15825505`, Allen Eager | [1144379811](https://en.wikipedia.org/w/index.php?oldid=1144379811), 2023-03-13 | 78/78 words |

Confidence is `full_passage_in_one_historical_candidate`. This supports
reconstructed article identity, not proof of the exact revision used by the
Dolly editor or uniqueness across all historical Wikipedia pages. Both
`is_editor_source_attested` and `is_training_admitted` remain false. No title
guess or partial passage match became a document ID. Exact context hashes
were checked against the rest of Dolly; neither match adds another row.

The network-enabled run made 22 requests and retained 1,846,627 response bytes.
Two rows had no search hit; twelve encountered HTTP 429; fourteen remain
unresolved. Rate limiting is an acquisition failure, not evidence that a
document does not exist. The earlier sandbox run made 16 failed DNS attempts
and received no bytes. Both attempts and their errors are preserved. No
further retrieval or sample expansion followed this bounded run. A future
approved retrieval run should stop on rate limiting, rather than continue
the remaining fixed queries as this probe did.

All response URLs, hashes, timestamps, candidate revision IDs and unresolved
source-line IDs are in the retained provenance report. The search is not an
exhaustive document map.

## Necessary capacity gates

| Dolly stratum | Original rows | Below 65 tokens | 65-447 tokens | Above 447 tokens | Required total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prose/open QA | 4,451 | 4,433 | 18 | 0 | 400 |
| Grounded QA | 1,773 | 48 | 1,525 | 200 | 400 |
| Extraction/classification | 3,642 | 2,002 | 1,369 | 271 | 400 |

These lengths include the prescribed ChatML wrapper. The 18 eligible prose
rows have source-line IDs 224, 264, 2054, 2712, 2929, 4304, 4864, 5673, 8238,
8483, 9363, 9386, 9654, 9909, 10390, 12802, 12835 and 14238. Maximum prose
prompt length is 186. This upper-bound failure precedes grouping or exclusions.

Grounded QA has 1,443 distinct exact context hashes among 1,525 length-eligible
rows. Those are **not** 1,443 independent documents. Only two document groups
have recovered evidence; neither has completed all other admission gates.
The other 1,771 grounded rows retain unresolved-document quarantine. The
current verified pool cannot satisfy 400 rows or the held-out minimum of 16
independent groups. The corpus-wide grounded quota remains **unestablished**,
not proven intrinsically impossible by a 16-row sample. Split assignment is
not attempted before document/template grouping.

Extraction/classification retains 1,506 unresolved context-bearing rows.
The five other Dolly categories are not substitute quota capacity. No split,
quota, category, input length or provenance requirement changed.

## MBPP isolation work

All upstream partitions remain fixed: train 601-974, calibration 511-600 and
held-out 11-510. Grouping considers every source row before exclusions.

| Check | Completed result | Limitation |
| --- | --- | --- |
| Normalized 32-token shared spans | All 964 statements compared; zero matching pairs | Does not establish semantic isolation |
| Alpha-normalized AST | 668 simple-scope programs processed | 296 complex-scope programs need review; no reference program executed |
| Literal benchmark exclusion | 265,771 texts from 1,574 retained files; no automatic source match | No semantic or dynamic-template clearance |
| Manual template review | Eight conservative families recorded | Partial review only; 397 remaining candidate pairs, plus nonlexical paraphrases |
| Manual benchmark exclusion | MBPP 674 and 932 overlap stable string deduplication benchmark | Cross-language equivalence was missed by literal comparisons |

The [manual decisions](2026-10-30-mbpp-review.json) state their scope and
reasons. They join maximum-frequency, abundant-number, regex, conversion and
other substantive templates without changing upstream splits. All connected
components touching multiple partitions or an excluded benchmark task are
quarantined. A superficial text match is not automatically a paraphrase:
the tuple-quantifier and several deduplication variants remain unresolved.

| MBPP partition | Remaining row upper bound | Independent-group upper bound | Required rows |
| --- | ---: | ---: | ---: |
| Train | 317 | 310 | 256 |
| Calibration | 73 | 72 | 48 |
| Held-out | 436 | 420 | 96 |

The graph has 857 components; 55 quarantine 138 rows. The
[content-free component report](2026-10-30-mbpp-isolation.json) records every
task ID, edge and quarantine reason. These remaining counts include unresolved
reviews, so they are not admitted independent-group capacity. No task has a
complete manual semantic clearance. Further review stopped at the decisive
Dolly quota failure instead of claiming the outstanding MBPP audits complete.

The exclusion inventory covers tracked `data`, `src`, `tests`, `examples`,
`benches`, experiment docs and performance history, plus retained local
textual artifacts. It includes JSON outputs, raw prompt literals and archive
members without extracting or executing them. It excludes the source-dataset
working directories to avoid self-matching. Source hashes and witness IDs are
retained. All selected artifacts parse, and no referenced `local-artifacts`
text file in that scope is missing. That is not proof of global closure:
dynamic templates, prompt literals in other script paths, external/deleted
artifacts and semantic paraphrases still need review.

An initial merge of reusable token rows included texts no longer in the final
archive-expanded inventory. The coverage check correctly rejected it. The
unfiltered artifact is retained; final tokens are filtered by exact normalized
text SHA256 IDs and cover all 265,771 final inputs exactly once. No threshold
was changed to pass that check.

## Validation and reproducibility

The 26 CPU unit tests pass, including inert archive handling, ambiguity and
partial-passage rejection, token-span boundaries, duplicate exclusion
propagation and incomplete tokenizer coverage. The CPU Rust example passes
warning-as-error Clippy; repository formatting and diff whitespace checks pass.
The independent census audit validates source/token cardinality, unique IDs,
per-row token counts, complete component coverage and zero training admission.

```bash
source scripts/dev-env.sh
local-artifacts/2026-10-29-drafter-data/venv/bin/python \
  -m unittest discover -s scripts/drafter_data -v
CARGO_TARGET_DIR=target/drafter-data-cpu RUSTFLAGS='-D warnings' \
  cargo clippy --no-default-features --locked --example drafter_audit_tokens -- -D warnings
sha256sum -c docs/experiments/2026-10-29-drafter-data.sha256
sha256sum -c docs/experiments/2026-10-29-drafter-v2.sha256
sha256sum -c docs/experiments/2026-10-30-drafter-r0.sha256
```

CPU audit regeneration commands are in the
[reproduction script](2026-10-30-drafter-r0-reproduce.sh). It performs no source
downloads and does not regenerate teacher features or training data.

## Review choices

1. Keep V2 unchanged and close dataset admission as blocked. This is the current
   decision; it requires no GPU work or further source collection.
2. Separately review whether a new Dolly natural-short-prompt amendment is
   warranted. It would need its own version, capacity/isolation audit and
   appropriate cost coverage. This checkpoint does not authorize or implement it.
3. If the overall dataset has a viable reviewed specification, obtain an
   authoritative Dolly document map or authorize a bounded, cached historical
   passage-recovery study with rate-limit handling. Finish complex-scope and
   semantic MBPP/benchmark review before selection. More grounded-QA recovery
   alone cannot fix the prose quota failure.

No new GPU hours were consumed. Usage remains 0.005923 of 72 aggregate hours;
parent phase ceilings remain 4/20/40/8, pilot caps 2/2/1, with at most 0.25 hour
of a future reviewed calibration inside the one-hour pilot evaluation cap.
Rank256, three guess positions, the four-epoch schedule, exact-output gates,
joint-only ownership, qualified binaries and runtime defaults are unchanged.
No GPU switching, teacher generation, training, learned evaluation or
speculative integration occurred.
