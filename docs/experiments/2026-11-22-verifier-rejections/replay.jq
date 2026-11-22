def require($condition; $why): if $condition then . else error($why) end;
def median: sort | length as $n | (.[(($n-1)/2|floor)] + .[($n/2|floor)])/2;
def stats: {median:median,minimum:min,maximum:max};
. as $records | first as $manifest | last as $end |
require($manifest.kind=="direct_verifier_manifest" and $manifest.rows==4 and
  $end.kind=="direct_verifier_complete" and $end.device_bytes_after==0; "incomplete verifier capture") |
($manifest.is_down_comparison // false) as $new |
(if $new then ["multi_row","serial","shared_weights"] else ["serial"] end) as $arms |
map(select((.kind=="down_target_trial") or (.kind=="direct_verifier_trial" and .is_direct)) | .result.generated_token_ids) |
require(length>=5 and (unique|length)==1 and first!=null and (first|length)==512; "missing canonical target outputs") |
first as $ids | $records |
map(select(.kind==(if $new then "down_target_rejection" else "direct_verifier_rejection" end)) |
    {arm:(.down_projection // "serial"),round:(.round // .trial),result}) |
require((map([.arm,.result.accepted_prefix,.round])|sort)==
    [$arms[] as $arm | range(4) as $p | range(5) as $r | [$arm,$p,$r]];
    "incomplete accepted-prefix and schedule grid") |
require(all(.[]; .result as $r | $r.useful_progress==$r.accepted_prefix+1 and
    $r.has_exact_correction and $r.next_token==$ids[$r.accepted_prefix+1] and
    $r.counters.cublas_calls==0 and
    ([$r.elapsed_ms,$r.target_ms,$r.correction_ms,$r.rollback_ms]|all(type=="number" and .>0)) and
    $r.elapsed_ms+0.00001>=($r.target_ms+$r.correction_ms+$r.rollback_ms)); "incorrect rejection or timing") |
{revision:$manifest.revision, model_sha256:$manifest.model_sha256, is_three_schedule_study:$new,
 groups:(group_by([.arm,.result.accepted_prefix])|map({
    down_projection:first.arm, accepted_prefix:first.result.accepted_prefix,
    useful_progress:first.result.useful_progress,
    complete_ms:(map(.result.elapsed_ms)|stats), target_ms:(map(.result.target_ms)|median),
    rollback_ms:(map(.result.rollback_ms)|median), correction_ms:(map(.result.correction_ms)|median),
    ms_per_committed_token:(map(.result.elapsed_ms/.result.useful_progress)|median)
  }))}
