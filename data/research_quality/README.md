# Research Quality Corpus V1

Frozen C19 inputs for separately labeled model-modification research. Both
splits contain chat, factual retrieval, multi-fact extraction, code/config,
arithmetic, and long-form tasks, plus two fixed language-modeling passages.
Inputs are original synthetic text, not a downloaded benchmark or browser
benchmark answers. The paired task families use distinct facts and content;
this is a bounded diagnostic suite, not a general capability benchmark.

Frozen `v1.json` SHA-256:
`2a1270959ef105a3ec19e0be0ab888048fb619880fd4f014ce216acfb8710c38`.

`calibration` may guide pruning/activation/layer choices. `held_out` is for
reporting quality changes, not adapting thresholds to a failing prompt. Record
the whole-file and selected-split hashes with every run. Do not edit V1 after
observing results; new data or rubric changes require a separately identified
version and preserved prior results.

Scoring is deterministic and task-specific:

- Exact answers reject extra prose and substring matches.
- JSON facts compare the expected scalar values with exactly the expected
  keys. Field order, string case, whitespace, and optional fences do not matter;
  numeric strings are not integer answers.
- Long-form rubrics require named concept groups, minimum words, and paragraph
  counts. Aliases are word-bounded. These are surface checks, not a semantic
  judge or proof of factual correctness.

Report partial fact/concept credit and format validity alongside pass/fail.
Fluency alone never passes. Dense-failed tasks remain capability/fixture evidence
and are inconclusive for pruning-quality regressions; retain their raw output.
Do not rewrite a failing dense prompt to make a variant look safe.

The loader rejects duplicate IDs/content across both splits, missing task
families, empty/invalid rules, and reserved browser prompt/answer material.
Those checks catch direct contamination, not paraphrases; provenance and
manual review remain necessary. Tests exercise data loading and observed
scoring behavior, not assertions against source or documentation text.

Teacher-forced evaluation will score next-token likelihood on the fixed
passages, with explicit tokenization/BOS/context rules and token-weighted
aggregation. This is not a WikiText perplexity claim. Quality, model/logit drift,
resident memory, and clearly scoped E2E timings must remain separate from the
unchanged Qwen HTTP throughput target.
