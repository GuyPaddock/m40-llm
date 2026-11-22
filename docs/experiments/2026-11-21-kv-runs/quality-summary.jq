# Run on the original broad-grid JSONL with jq -s after independent audit.
def median: sort | length as $n | (.[(($n-1)/2|floor)] + .[($n/2|floor)]) / 2;
def p95: sort | .[((length * 0.95 | ceil) - 1)];

if last.kind != "kv_quality_complete" or last.cases != 48 then error("incomplete grid") else . end |
map(select(.kind=="kv_quality_comparison")) |
if length!=96 or (map([.case,.arm])|unique|length)!=96 then error("incomplete comparisons") else . end |
group_by([.context,.arm]) |
map(. as $cases |
    [$cases[] | .case as $id | .prefix_observations[] | .+{case:$id}] as $observations |
    {context:.[0].context,arm:.[0].arm,cases:length,
     preserved:map(select(.quality_conclusion=="preserved")|.case),
     lost:map(select(.quality_conclusion=="regression")|.case),
     dense_failed:map(select(.quality_conclusion=="inconclusive")|.case),
     identical_ids:map(select(.identical_generated_ids))|length,
     divergent:map(select(.identical_generated_ids==false)|{case,first_generated_difference}),
     old_block_range:(map(.cache.old_blocks) | [min,max]),
     observations:($observations|length),
     greedy_agreements:([$observations[]|select(.logits.reference.top_indices[0]==.logits.candidate.top_indices[0])]|length),
     kl_median:($observations|map(.logits.kl_reference_candidate)|median),
     kl_p95:($observations|map(.logits.kl_reference_candidate)|p95),
     worst_kl:($observations|max_by(.logits.kl_reference_candidate)|{case,step,kl:.logits.kl_reference_candidate}),
     per_vector_mean_abs_error_median:($observations|map(.logits.mean_abs_logit_diff)|median),
     largest_abs_error:($observations|max_by(.logits.max_abs_logit_diff)|{case,step,error:.logits.max_abs_logit_diff}),
     top10_overlap_min:($observations|map(.logits.top10_overlap)|min),
     top10_overlap_median:($observations|map(.logits.top10_overlap)|median)})
