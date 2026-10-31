# jq -s --arg source REV --arg candidate VARIANT --arg report PATH --arg pid PID -f THIS INPUT
# Audit bounded packed128-control FFN follow-ups, never whole-model qualification.
def require($condition; $message): if $condition then . else error($message) end;
def median: sort | if length%2==1 then .[length/2|floor] else (.[length/2-1]+.[length/2])/2 end;
map(select(.kind=="ffn_ablation_manifest")) as $manifest |
map(select(.kind=="ffn_ablation_pair")) as $pairs |
map(select(.kind=="ffn_ablation_validation")) as $checks |
map(select(.kind=="ffn_ablation_screen")) as $screens |
map(select(.kind=="ffn_ablation_complete")) as $complete |
"packed-warp128" as $control |
require(($manifest|length)==1 and $manifest[0].revision==$source and
  $manifest[0].worktree==("?? "+$report) and $manifest[0].build_profile=="release" and
  $manifest[0].trials==5 and $manifest[0].iterations==10 and
  $manifest[0].fp32_variants==[$candidate] and $manifest[0].is_q81_numerical_track==false and
  $manifest[0].memory=={original_weights:3279519744,prepared_weights:1192935424,resident_kv:0,tracked_device_bytes:4472455168};
  "wrong source, clean-code scope, configuration or memory") |
require(($pairs|length)==30 and ($checks|length)==18 and ($screens|length)==6 and
  [$screens[]|[.layer_id,.rows]]==[[0,1],[0,4],[17,1],[17,4],[35,1],[35,4]] and
  ($complete|length)==1 and $complete[0].is_complete and
  $complete[0].device_bytes_before==0 and $complete[0].device_bytes_after==0;
  "incomplete coverage or allocation cleanup") |
require(all($checks[]; .checks.guards_and_immutable_inputs==true and
  (if .variant==$candidate then .checks.intermediate_capture_matches_normal_kernel==true else true end));
  "invalid numerical/guard checks") |
require(([$checks[]|[.layer_id,.rows]]|unique)==[$screens[]|[.layer_id,.rows]] and
  all($checks|group_by(.layer_id,.rows)[]; (map(.variant)|sort)==[null,$control,$candidate]);
  "missing native, timing-control or candidate validation") |
require(all($pairs|group_by(.layer_id,.rows)[]; map(.trial)==[0,1,2,3,4]);"wrong trial coverage") |
require(all($pairs[]; .control_variant==$control and .variant==$candidate and
  .activation_quantization_ms==0 and .is_numerical_track==false and
  .quantization_scratch_capacity_bytes==0 and .threads==128 and
  [.measurements[].variant]==(if .trial%2==0 then [$control,$candidate] else [$candidate,$control] end));
  "wrong candidate, quantization or alternating arm order") |
require(all($pairs[]; . as $pair | all(.measurements[];
  [.stages[].stage]==(if $pair.trial%2==0 then ["gate_up_swiglu","isolated_down","complete"] else ["complete","isolated_down","gate_up_swiglu"] end) and
  all(.stages[]; . as $stage | .enqueue_counters as $counts |
    .iterations==10 and .gpu_ms>0 and .completed_unix_us>.started_unix_us and
    all(["cublas_calls","d2h_bytes","d2h_copies","device_allocs","device_frees","h2d_bytes","h2d_copies","stream_syncs"][]; $counts[.]==0) and
    $counts.launches==(if $stage.stage=="complete" then 20 else 10 end) and
    $counts.stream_waits==(if $pair.rows==1 then 0 elif $stage.stage=="complete" then 20 else 10 end))));
  "wrong stage order or timed enqueue effects") |
require(all($screens[]; . as $screen |
  [$pairs[]|select(.layer_id==$screen.layer_id and .rows==$screen.rows)|
    [.measurements[]|select(.variant==$control)|.stages[]|select(.stage=="complete")|.gpu_ms] +
    [.measurements[]|select(.variant==$candidate)|.stages[]|select(.stage=="complete")|.gpu_ms]] as $actual |
  .result.pairs==$actual and .result.control_median_ms==($actual|map(.[0])|median) and
  .result.candidate_median_ms==($actual|map(.[1])|median) and
  .result.throughput_gain==(.result.control_median_ms/.result.candidate_median_ms-1) and
  .result.passes_micro_screen==(.result.throughput_gain>=0.05 and .result.bootstrap_gain_ci95[0]>0));
  "screen does not match observed complete-FFN pairs") |
[$pairs[]|.gpu_before,.gpu_after] as $gpu |
require(all($gpu[]; . as $sample | (.processes|split(", ")) as $process |
  ($sample.device|contains(", P0,")) and ($sample.device|contains("1113 MHz, 3004 MHz")) and
  $process[0]==$pid and $process[1]=="target/release/examples/q8_execution_bench" and
  ($process[2]|test("^[0-9]+ MiB$")));
  "uncontrolled clocks or another GPU process") |
[$pairs[].measurements[].stages[]]|sort_by(.started_unix_us) as $intervals |
require(all(range(1; $intervals|length); $intervals[.].started_unix_us >= $intervals[.-1].completed_unix_us);
  "overlapping primary GPU intervals") |
{is_complete:true,source:$source,candidate:$candidate,primary_pairs:30,primary_intervals:($intervals|length),
 gpu_boundary_samples:($gpu|length),validation_records:18,tracked_bytes_after_release:0,
 request_pattern_hypothesis:$manifest[0].request_pattern_hypothesis,
 screens:($screens|map({layer_id,rows,control_variant,variant,result})),
 stages:($pairs|group_by(.layer_id,.rows)|map(. as $shape |
   {layer_id:.[0].layer_id,rows:.[0].rows,arms:([$shape[].measurements[]]|group_by(.variant)|map(
     {variant:.[0].variant,stages:([.[].stages[]]|group_by(.stage)|map({stage:.[0].stage,median_gpu_ms:map(.gpu_ms)|median}))}))})),
 candidate_errors:($checks|map(select(.variant==$candidate))|
   {gate_max_abs:map(.checks.gate_vs_cpu.max_abs_diff)|max,up_max_abs:map(.checks.up_vs_cpu.max_abs_diff)|max,
    swiglu_max_abs:map(.checks.swiglu_vs_cpu.max_abs_diff)|max,
    complete_down_max_abs:map(.checks.complete_down_vs_cpu_on_actual_swiglu.max_abs_diff)|max,
    hidden_vs_native_max_abs:map(.hidden_vs_native.max_abs_diff)|max,output_vs_native_max_abs:map(.complete_output_vs_native.max_abs_diff)|max})}
