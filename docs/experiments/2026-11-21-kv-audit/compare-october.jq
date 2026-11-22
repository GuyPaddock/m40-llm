# Compare deterministic observations across the original and repeated M40 grid.
def check($v;$why): if $v then . else error($why) end;
def key: [.kind,.case,.arm];
def stable:
  if .kind=="kv_quality_manifest" then
    del(.executable_sha256,.source_revision,.prepared_weights.model_hash_elapsed_ms,
        .prepared_weights.preparation_elapsed_ms)
  elif .kind=="kv_quality_generation" then
    del(.metrics.prefill_elapsed_ms,.metrics.decode_elapsed_ms,.metrics.total_elapsed_ms)
  elif .kind=="kv_quality_complete" then del(.elapsed_wall_seconds)
  else . end;
def counts: group_by(.kind)|map({kind:first.kind,count:length});
. as $new |
check(($old|length)==86 and ($new|length)==86 and $old[0].source_clean and $new[0].source_clean;
  "wrong grid size or dirty source") |
check($new[0].source_revision==$revision and $new[0].executable_sha256==$executable and
  ($old|map(key)|sort)==($new|map(key)|sort) and ($old|map(key)|unique|length)==86 and
  $old[-1].kind=="kv_quality_complete" and $new[-1].kind=="kv_quality_complete";
  "source binding, row keys, or completion differs") |
($old|sort_by(key)) as $a | ($new|sort_by(key)) as $b |
[range(0;86) as $i | ($a[$i]|stable) as $x | ($b[$i]|stable) as $y |
  select($x!=$y) | {key:($x|key),fields:([($x+$y)|keys[]|select($x[.]!=$y[.])])}] as $diff |
{historical_revision:$old[0].source_revision,repeat_revision:$revision,
 model_sha256:$new[0].model_sha256,inputs_sha256:$new[0].inputs_sha256,
 generated_tokens:$new[-1].generated_tokens,checks:($new|counts),
 matched_prefix_observations:([$new[]|select(.kind=="kv_quality_comparison")|.prefix_observations|length]|add),
 different_records:$diff,all_deterministic_fields_equal:($diff|length==0)}
