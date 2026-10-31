# jq -n --slurpfile correctness ...actual-logits.jsonl
#       --slurpfile operators ...actual-operators.jsonl
#       --slurpfile ffn ...compound-packet-operators.jsonl
#       --slurpfile frozen ...compound-runtime-quality.jsonl -f this-file
def require($ok; $why): if $ok then . else error($why) end;
def median: sort | .[length/2|floor];
def validations: map(select(.kind=="portfolio_actual_logits_validation"));
def completed: map(select(.kind=="portfolio_actual_logits_complete"));
($frozen|map(select(.kind=="q8_runtime_browser_generation" and .label=="candidate"))[0].generation.generated_token_ids) as $expected |
require(($expected|length)==512; "frozen IDs missing") |
reduce [$correctness,$operators][] as $records (.;
  ($records|validations) as $v |
  ($records|completed) as $c |
  require(($v|map(.step))==[0,1,8,32,128,256,511]; "sample coverage") |
  require(($c|length)==1 and $c[0].is_gate_passed and
    $c[0].actual_generated_ids==$expected and $c[0].device_bytes_after==0;
    "incomplete/changed greedy continuation or leaked allocation") |
  require(all($v[]; .result.full_vocabulary_bits_equal and
    .result.control_and_partitioned_argmax_equal and .result.vocabulary==151936 and
    .result.scratch_bytes==384 and .result.expected_token==$expected[.step]); "actual-logit failure")
) |
require(($correctness|validations|map(.result.logits_sha256))==
  ($operators|validations|map(.result.logits_sha256)); "logits changed between runs") |
($ffn|map(select(.kind=="prepared_projection_screen" and .family=="down" and .rows==1))) as $down |
require(($down|map(.layer_id))==[0,17,35] and
  all($down[]; .complete_ffn.bootstrap_gain_ci95[0]>0 and
    .gate_up_kernel=="joint-f32x4-shared-scales-128"); "applicable FFN screen failed") |
require(any($ffn[]; .kind=="prepared_projection_manifest" and .is_compound_control and .worktree==""); "wrong FFN control") |
require(any($ffn[]; .kind=="prepared_projection_complete" and .is_complete and
  .cases==8 and .device_bytes_after==0); "FFN teardown incomplete") |
($operators|map(select(.kind=="portfolio_operator_summary"))) as $summary |
($operators|map(select(.kind=="portfolio_operator_pair"))) as $pairs |
require(($summary|length)==14 and ($pairs|length)==70; "operator timing coverage") |
require(all($summary[]; .pairs==5 and .bootstrap_gain_ci95[0]>0 and
  (.needs_variation_extension|not)); "operator uncertainty gate") |
require(all($pairs[].measurements[]; .iterations==100 and .gpu_ms>0 and
  .enqueue_counters.device_allocs==0 and .enqueue_counters.device_frees==0 and
  .enqueue_counters.h2d_copies==0 and .enqueue_counters.d2h_copies==0 and
  .enqueue_counters.stream_syncs==0 and .enqueue_counters.cublas_calls==0);
  "timed allocation/copy/synchronization") |
($summary|map({step,operator,
  control_ms:(1000/.control_operators_per_sec.median),
  candidate_ms:(1000/.candidate_operators_per_sec.median),
  throughput_gain,bootstrap_gain_ci95})) as $timings |
($down|map(.complete_ffn.native_median_ms-.complete_ffn.candidate_median_ms)|median) as $down_saved |
($timings|map(select(.operator=="vocabulary")|.control_ms-.candidate_ms)|median) as $head_saved |
($timings|map(select(.operator=="argmax")|.control_ms-.candidate_ms)|median) as $argmax_saved |
(512/50.910426*1000) as $request |
(512*(36*$down_saved+$head_saved+$argmax_saved)) as $saved |
{
  correctness_passed:true, operator_screen_passed:true,
  sampled_full_vocabulary_vectors:7,actual_generated_ids:512,
  applicable_rows:1,
  four_row_limitation:"layer35 complete FFN interval crosses zero; no four-row admission",
  ffn:($down|map({layer_id,complete_ffn,isolated_projection})), actual_operators:$timings,
  conditional_absolute_cost_transfer:{
    assumptions:"three sampled layer savings represent36 layers; seven head/argmax inputs represent512 tokens; costs transfer independently with no runtime interaction",
    source:"contemporary absolute component latency differences, not historical percentages",
    down_ms:(512*36*$down_saved),vocabulary_ms:(512*$head_saved),argmax_ms:(512*$argmax_saved),
    total_ms:$saved,request_reference_ms:$request,required_for_three_percent_ms:($request-$request/1.03),
    conditional_gain:($request/($request-$saved)-1),
    interpretation:"admission scope only, not measured HTTP or a joint confidence bound"
  },
  required_full_portfolio_extra_device_bytes:1192935424,
  measured_http_gain:null,runtime_qualification_complete:false,
  next:"fixed five-arm runtime correctness then paired HTTP with fresh confirmation if borderline"
}
