def required($condition; $message): if $condition then . else error($message) end;
def only($rows; $kind):
  [$rows[] | select(.kind == $kind)] | required(length == 1; $kind) | first;
def median: sort | .[length/2|floor];
def fresh_load($rows; $label):
  only($rows; "joint_only_load") |
  required(.model.ownership == $label and .model.joint.pair_count == 36; "wrong load") |
  {elapsed_ms, host_high_water, tracked_device_delta, counters,
   prepared_bytes: .model.joint.combined_prepared_bytes,
   preparation_ms: .model.joint.preparation_elapsed_ms};
[$retain, $joint, $quality, $http] |
map(only(.; "joint_only_manifest") | {model_sha256, executable_sha256, revision, worktree, build_profile, frozen_environment}) |
required(unique|length==1; "manifest disagreement") | first |
required(.worktree=="" and .build_profile=="release" and .revision=="c4807e1ff3a49603a37fb1a20fd274eb775f107e"; "wrong source") as $manifest |
fresh_load($retain;"retain-split") as $a | fresh_load($joint;"joint-only") as $b |
only($quality;"joint_only_generation") as $gen |
only($quality;"joint_only_ownership") as $owners |
only($quality;"joint_only_quality_complete") as $qc |
[$quality[]|select(.kind=="joint_only_task")] as $tasks |
[$quality[]|select(.kind=="joint_only_logits")] as $logits |
only($http;"joint_only_http_binding") as $binding |
only($http;"joint_only_http_summary") as $summary |
only($http;"joint_only_http_complete") as $complete |
[$http[]|select(.kind=="joint_only_http_trial")] as $trials |
required($binding.quality_sha256==$quality_sha and $gen.output==$gen.candidate_output and
  ($gen.generated_ids|length)==512 and ($gen.prompt_ids|length)==65; "canonical generation mismatch") |
required(($tasks|length)==12 and ($tasks|all(.has_identical_ids)) and $qc.is_gate_passed and
  $qc.dense_valid==([$tasks[]|select(.score.passed)]|length) and $qc.device_bytes_after==0; "quality accounting") |
required(($logits|length)==7 and ($logits|map(.step)|unique|length)==7 and
  ($logits|all(.vocabulary==151936 and .max_abs_diff==0 and .candidate_sha256==.control_sha256)); "logit disagreement") |
required(($owners.candidate_payload_hashes|length)==36 and $owners.candidate_payload_hashes==$owners.control_payload_hashes and
  $owners.resident_reduction_bytes==($a.tracked_device_delta-$b.tracked_device_delta) and
  $owners.resident_reduction_bytes==($a.prepared_bytes-$b.prepared_bytes); "weight ownership") |
required(($trials|map([.pair,.arm])|sort)==[range(5) as $p|range(2) as $a|[$p,$a]] and
  ($trials|all(.row.output==$gen.output and .row.metrics.generated_tokens==512 and
    .row.metrics.stop_reason=="length" and .row.metrics.prompt_tokens==65 and
    (.row.http_tok_per_sec - (512000/.row.http_ms)|fabs)<1e-9)); "trials or outputs invalid") |
[$trials|group_by(.arm)[]|map(.row.http_tok_per_sec)|median] as $rates |
required(($rates[1]/$rates[0]-1-$summary.comparison.gain|fabs)<1e-12 and
  $complete.device_bytes_after==0 and $complete.is_storage_gate_passed; "endpoint comparison") |
{manifest:$manifest,retain:$a,joint:$b,tracked_saved:$owners.resident_reduction_bytes,
 skipped_allocations:$owners.skipped_device_allocations,load_peak_mib:$peaks,
 dense_passing:$qc.dense_valid,dense_inconclusive:$qc.inconclusive,
 checked_full_logits_steps:($logits|map(.step)),http_rates:$rates,comparison:$summary.comparison}
