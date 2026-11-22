# Replay complete target work per request. The existing rejection audit prices rejected prefixes.
def check($ok;$why): if $ok then . else error($why) end;
def median: sort | length as $n | (.[(($n-1)/2|floor)] + .[($n/2|floor)])/2;
def arm_order($i): ["serial","multi_row","shared_weights"] as $arms |
  [range(3) | $arms[(.+($i/2|floor))%3]] | if $i%2==1 then reverse else . end;
def phase_keys($name;$begin;$end):
  [range($begin;$end) as $i | arm_order($i) | to_entries[] | [$name,$i,.key,.value]];
def trial($ids):
  . as $row | .result as $r | (.down_projection // "serial") as $arm |
  $r.target_dispatch as $d | $r.target_counters as $c |
  check($r.is_direct and $r.prompt_tokens==65 and $r.generated_token_ids==$ids and
    $r.stop_reason=="length" and $r.generated_tokens==512 and $r.target_input_rows==511 and
    $r.committed_len==576 and $r.layer_lengths==[range(36)|576];"generated IDs or valid KV lengths") |
  check(($r.batch_ms|length)==127 and ($r.serial_step_ms|length)==3 and
    all(($r.batch_ms+$r.serial_step_ms)[];type=="number" and .>0);"incomplete four-row transactions or tails") |
  ($r.batch_ms|add) as $batch | ($r.serial_step_ms|add) as $tail |
  check(($r.decode_ms-$batch-$tail-$r.decode_other_ms|fabs)<1e-6 and $r.decode_other_ms>=0 and
    $r.total_ms+1e-6>=($r.session_setup_ms+$r.prefill_ms+$r.decode_ms+$r.session_teardown_ms) and
    ($r.resident_total_tps-512000/$r.total_ms|fabs)<1e-8 and
    ($row.batches_per_second==null or ($row.batches_per_second-127000/$batch|fabs)<1e-8);"timer accounting") |
  check($r.tracked_after_teardown==$r.tracked_before and
    $r.tracked_after_setup-$r.tracked_before==$r.session_scratch_bytes and
    $r.tracked_after_setup==$r.tracked_after_prefill and $r.tracked_after_prefill==$r.tracked_after_decode and
    $c.device_allocs==0 and $c.device_frees==0 and $c.cublas_calls==0 and
    ($r.total_counters.device_allocs==$r.total_counters.device_frees);"target allocations or ownership") |
  check($d.ffn_q8_joint_four_rows_128==127*36 and $d.ffn_q8_joint_f32x4_shared_scales_128==3*36 and
    $d.qkv_f32xq8_0_gguf_packed_warp==511*36 and $d.gemm_f32xq8_0_gguf_coltile4_packed_warp==511*36 and
    $d.gemm_f32xq8_separate_f32==511+36*(if $arm=="serial" then 511 elif $arm=="multi_row" then 130 else 3 end) and
    ($d.q8_small_rows//0)==(if $arm=="shared_weights" then 127*36 else 0 end);"wrong target dispatch") |
  {arm:$arm,phase:$row.phase,round:($row.round//$row.trial),order:$row.order,
    mean_batch_ms:($batch/127),tail_ms:$tail,decode_ms:$r.decode_ms,resident_ms:$r.total_ms,
    device_event_decode_through_teardown_ms:$r.gpu_decode_through_teardown_ms,session_scratch_bytes:$r.session_scratch_bytes,
    tracked_live_bytes:$r.tracked_after_decode,counters:$c,down_launches:($d.gemm_f32xq8_separate_f32-511+($d.q8_small_rows//0)),
    began:$r.started_unix_us,ended:$r.completed_unix_us};
first as $m | last as $end | ($m.is_down_comparison//false) as $new |
check($m.kind=="direct_verifier_manifest" and $m.model_sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
    and $m.quality_sha256=="adf09f0077f7ec24e7ca61f1a846f8b4e97db4b631fa82c86a6af0201b5e4f24"
    and $m.rows==4 and $m.worktree=="" and $end.kind=="direct_verifier_complete" and
    $m.correctness_only==false and $end.device_bytes_after==0 and $end.is_complete and
    $m.prepared_payload_hashes==$end.prepared_payload_hashes and $m.joint_payload_hashes==$end.joint_payload_hashes;"incomplete measurement") |
[$quality[]|select(.kind=="q8_runtime_browser_generation" and .label=="candidate")|.generation.generated_token_ids] as $ref |
check(($ref|length)==1 and ($ref[0]|length)==512;"wrong canonical output") |
(if $new then ["multi_row","serial","shared_weights"] else ["serial"] end) as $arms |
[$correct[]|select(.kind=="direct_verifier_manifest")] as $cm |
[$correct[]|select(.kind=="direct_verifier_browser_correctness")] as $cb |
[$correct[]|select(.kind=="direct_verifier_complete")] as $ce |
check(($correct|length)==36*($arms|length) and ($cm|map(.down_projection//"serial")|sort)==$arms and
    ($cb|length)==($arms|length) and ($ce|length)==($arms|length) and
    all($cm[];.correctness_only and .executable_sha256==$m.executable_sha256 and .model_sha256==$m.model_sha256 and
        .quality_sha256==$m.quality_sha256 and .joint_payload_hashes==$m.joint_payload_hashes and .prepared_payload_hashes==$m.prepared_payload_hashes) and
    all($ce[];.is_correctness_passed and .is_complete and .device_bytes_after==0) and
    all($cb[];.actual_token_ids==$ref[0] and .has_exact_valid_kv and
      (.valid_kv_sha256|test("^[0-9a-f]{64}$")) and (.full_logit_comparisons|length)==511 and
      all(.full_logit_comparisons|to_entries[];.value.input_row==.key and .value.actual_next==$ref[0][.key+1] and
        .value.expected_next==.value.actual_next and .value.comparison.values==151936 and .value.comparison.has_tolerance_passed)) and
    ($cb|map(.valid_kv_sha256)|unique|length)==1;"missing matched correctness or KV parity") |
check(($new|not) or (($cb|map([.full_logit_comparisons[].comparison.actual_fp32le_sha256])|unique) as $digests |
    ($digests|length)==1 and all($digests[0][];type=="string" and test("^[0-9a-f]{64}$")));"full logit-row digests differ") |
map(select(if $new then .kind=="down_target_trial" else .kind=="direct_verifier_trial" and .is_direct end)|trial($ref[0])) as $rows |
(if $new then
   (if any($rows[];.phase=="extension") then phase_keys("extension";5;10) else [] end) as $ext |
   (if any($rows[];.phase=="confirmation") then phase_keys("confirmation";0;20) else [] end) as $confirm |
   ($rows|map([.phase,.round,.order,.arm]))==(phase_keys("screen";0;5)+$ext+$confirm)
 else ($rows|map([.phase,.round,.arm]))==[range(5)|["screen",.,"serial"]] end) as $ordered |
check($ordered and all(range(0;($rows|length));. as $i|$rows[$i].ended>$rows[$i].began and
    ($i==0 or $rows[$i].began>=$rows[$i-1].ended));"missing, overlapping or misordered trials") |
{revision:$m.revision,executable_sha256:$m.executable_sha256,model_sha256:$m.model_sha256,
 groups:($rows|group_by([.arm,.phase])|map({arm:first.arm,phase:first.phase,requests:length,
   four_row_transactions:(length*127),median_request_mean_batch_ms:(map(.mean_batch_ms)|median),
   median_three_tail_ms:(map(.tail_ms)|median),median_resident_ms:(map(.resident_ms)|median),
   down_launches_per_request:(map(.down_launches)|unique),target_counters_per_request:(map(.counters)|unique),
   session_scratch_bytes:(map(.session_scratch_bytes)|unique),tracked_live_bytes:(map(.tracked_live_bytes)|unique)})),
 trials:$rows}
