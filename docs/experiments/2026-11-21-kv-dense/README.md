# Dense reference on the 48 new quality cases

The M40 dense FP16-KV reference passed 19 of 24 short cases and 10 of 24 long cases. Of the 48 comparisons per candidate arm, 29 can support a conclusion about retaining a passing answer. The 19 reference failures remain inconclusive for preservation. At long context all four multi-fact cases passed, one each passed in code/config, chat and retrieval, three long-form cases passed their surface rubrics, and none of the four arithmetic cases passed. The family file identifies the individual failures. Keep complete outputs for semantic review.

first-pass.jsonl.gz contains the original run from its manifest through dense_grid_complete inclusive. Its 50 records comprise one manifest, 48 dense generations and the completion marker. The source is c49e208; the model hash, input hash, executable, prompt IDs, conditions, output tokens, dispatch and cache checks are embedded in the raw report. This checkpoint establishes dense outcomes only. It includes neither quantized quality verdicts nor timing qualification. The host sampler recorded Gentoo build activity during this run.

## Two observed top-8 regressions in completed comparisons

On the M40, both cases below stopped naturally in all three arms. In
each case dense FP16 and mixed top-64 produced identical sampled IDs.
Top-8 lost a correct answer. These are counterexamples to the frozen
no-loss promotion gate, irrespective of the remaining case outcomes.

| Case | Prompt IDs | Dense and mixed top-64 | Mixed top-8 |
|---|---:|---|---|
| Relay | 2,063 | Correct JSON values: port 23, passed true | Kept port 23 but changed passed to false |
| Ordering | 2,052 | Fern, Hazel, Iris | Fern, Iris, Hazel |

The [six original rows](top8-counterexamples.jsonl.gz) retain the prompts'
case IDs, answers, sampled IDs, scores and runtime diagnostics. These
cases each fit within top-64's available history. The same no-loss rule
has already failed for top-8 on this checkpoint. The full suite will
establish the denominator, other regressions, aligned-logit errors and
free-running continuation behavior. These two examples supply no bound
on failure rates for other workloads.
