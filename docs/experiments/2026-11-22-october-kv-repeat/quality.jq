def nearest_rank($p): sort | .[(length*$p|ceil)-1];
def summary: {median:nearest_rank(0.5),p95:nearest_rank(0.95),maximum:max};
last as $end |
if $end.kind!="kv_quality_complete" or $end.cases!=12 or $end.comparison_rows!=24 or $end.generation_rows!=36
then error("incomplete October quality grid") else . end |
map(select(.kind=="kv_quality_comparison")) |
group_by([.context,.arm]) | map(. as $cases |
  [.[].prefix_observations[].logits] as $observations |
  {context:first.context,arm:first.arm,cases:length,
   dense_passes:(map(select(.dense_score.passed))|length),
   preserved_dense_passes:(map(select(.dense_score.passed and .candidate_score.passed))|length),
   dense_failures:(map(select(.dense_score.passed|not))|length),
   identical_generated_ids:(map(select(.identical_generated_ids))|length),
   observed_old_blocks:(map(.cache.old_blocks)|[min,max]),prefix_observations:($observations|length),
   kl:($observations|map(.kl_reference_candidate)|summary),
   max_abs_logit_error:($observations|map(.max_abs_logit_diff)|max),
   different_traces:[$cases[]|select(.identical_generated_ids|not)|.case]})
