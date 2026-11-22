# Qwen2.5-3B Q8_0 generation quality with mixed KV

All 144 free-running generations completed on the Tesla M40. Each of the
48 prompts has one dense FP16 answer and two FP16-K/Q4-V answers.
The 24 long prompts exercised 33–38 historical blocks. Top-64 could use
all of them. Top-8 could use eight. The short prompts never evicted any
positions from the recent window. The model, prompt IDs, generation
limits and greedy settings were fixed across arms.

| Frozen answer rubric | Dense | Mixed top-64 | Mixed top-8 |
|---|---:|---:|---:|
| Short answers passing | 19/24 | 19/24 | 19/24 |
| Long answers passing | 10/24 | 10/24 | 8/24 |
| Dense-passing short answers retained | — | 19/19 | 19/19 |
| Dense-passing long answers retained | — | 10/10 | 7/10 |

**Top-8 fails the no-loss gate.** It changes the relay's correct Boolean
to false, puts Iris before Hazel, and uses only the plural “markers” when
the trail prompt explicitly requires “marker”. The trail directions still
communicate the requested route and safe return. The first two losses
change factual content. Top-8 also correctly emits only “Indigo” where
dense adds “Final label:”. That dense-failing case supplies no evidence
of preservation. Nineteen dense failures across the suite stay outside the
preservation denominator. Under a stricter whole-response JSON parse, all
three arms lose the same four short passes that used Markdown fences.
[Whole-response scoring](answer-format.json).

**Top-64 passes this answer-preservation gate on this checkpoint and block
range.** It kept all 29 dense-passing answers. It does not establish how
selection behaves when history contains more than 64 blocks. The separate
history-boundary study addresses that case.

## Reading all eight long-form prompts in all three arms

The four short generations in each mixed arm matched dense token for token.
The lost-notebook, walking-trail and meeting answers communicate the
requested actions. The potluck welcome uses “labeled” and “clean up” and
misses the frozen vocabulary and word-count requirements at short context.

| Long context | Dense | Mixed top-64 | Mixed top-8 |
|---|---|---|---|
| Lost notebook | One paragraph addresses the finder about privacy, reception and recording the time. | One paragraph gives the finder the same three instructions. | The message tells the owner “We found your lost notebook” and then asks them to bring it to reception. It appends “Privacy, reception, time.” as a separate paragraph. The concept scorer passes it. |
| Potluck | Gives a welcome, identifies the shared table and asks people to clean up. Misses the required exact words. | Substantively similar welcome, with the same vocabulary failure. | Substantively similar welcome, with the same vocabulary failure. |
| Walking trail | Explains map, markers and how to return, including the singular “marker”. | Explains the map, a marker, landmarks and retracing the route. | Gives coherent directions with the plural “markers”. Fails the requested singular word. |
| Volunteer meeting | Gives agenda, note-taking and deadline instructions, then finishes the summary at a natural stop. | Gives all three instructions and finishes naturally. | Gives all three instructions, then reaches the 160-token cap halfway through the closing sentence. The frozen scorer passes it. |

The meeting outputs contained 154, 159 and 160 sampled IDs, respectively.
Across all 24 long prompts, one dense generation, one top-64 generation
and three top-8 generations hit their cap. The largest repeated-four-gram
fraction in the top-8 arm was 0.1034 on the dense-failing stock task.
At short context every repeated-four-gram fraction was zero. These are
descriptive checks on a limited suite. The audience reversal and unfinished
summary are reasons to retain the text alongside the automated verdicts.

The [generation rows](generations.jsonl.gz) retain the text, sampled IDs,
scores, stopping reasons, allocation, repetition and dispatch observations.
The original runner was built from `c49e2083f85bda2c32b6fcd0d69f9c2d732ea7c5`.
The [frozen inputs](../2026-11-21-kv-inputs/README.md) identify the model,
context, tokenizer and all task rubrics. This review covers free generation.
Aligned prefix vectors, independent numerical replay and cache residency
have separate evidence.
