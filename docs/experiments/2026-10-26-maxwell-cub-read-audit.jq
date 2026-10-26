# Independently reconcile captured inputs, work, read-only results and timings.
def require($ok;$why): if $ok then . else error($why) end;
def median: sort | .[length/2|floor];
def key: [.layer_id,.rows];
def one($kind): map(select(.kind==$kind)) | require(length==1;"record multiplicity") | .[0];
. as $records
| one("ffn_read_manifest") as $m
| $records|one("ffn_read_complete") as $done
| $records|map(select(.kind=="ffn_read_pair")) as $pairs
| $records|map(select(.kind=="ffn_read_validation")) as $checks
| $records|map(select(.kind=="ffn_read_case_complete")) as $finished
| $records|map(select(.kind=="ffn_read_warmup")) as $warmups
| $records|map(select(.kind=="ffn_ablation_preparation")) as $preparation
| $records|map(select(.kind=="ffn_read_cub_screen")) as $screens
| [[0,1],[0,4],[17,1],[17,4],[35,1],[35,4]] as $shapes
| require($m.revision==$source and $m.executable_sha256==$elf and $m.worktree==""
    and $m.build_profile=="release" and $m.schema_version==1
    and $m.model_sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
    and $m.fixture_sha256=="eb51828ef778189567ae01fdd910659025c92ae6f5b2d76c989c314e650d0687"
    and $m.capture_report_sha256=="d9e55963e5872bbc48cfb3a1f7ac217429974d0a48237aabaaab207f5fc95801"
    and $m.execution_configuration==$capture[0].execution_configuration
    and ($m.prepared_weights|del(.preparation_elapsed_ms,.model_hash_elapsed_ms))
      ==($capture[0].prepared_weights|del(.preparation_elapsed_ms,.model_hash_elapsed_ms))
    and ($m.prepared_weights.tensors|length)==37
    and $m.memory=={original_weights:3279519744,prepared_weights:1192935424,resident_kv:0,tracked_device_bytes:4472455168}
    and $m.is_weight_read_diagnostic and ($m.is_fp32_ablation|not) and ($m.is_q81_numerical_track|not)
    and $m.cub_read_contract=={compiled_cub_version:200700,has_measured_dram_traffic:false,
      has_same_modular_column_outputs:true,library:"bundled CUB public segmented reduction",
      one_row_read_gain_gate_percent:10,runtime_integration_allowed:false}
    and $m.weight_read_column_gap_bytes==null and $m.column_order_contract==null
    and $m.rows==[1,4] and $m.trials==5 and $m.iterations==10;"clean frozen source/control")
| require(($records|length)==62 and ($pairs|length)==30
    and all([$checks,$finished,$warmups,$preparation,$screens][];length==6 and (map(key)|sort)==$shapes)
    and $done.is_complete and $done.cases==6 and $done.device_bytes_before==0
    and $done.device_bytes_after==0;"coverage/cleanup")
| require(all($checks[];. as $check|
    .checks.has_matching_cpu_checksums and .checks.has_intact_output_guards
    and .checks.checksum_count==.rows*11008*2 and (.checks.checksum_sha256|test("^[a-f0-9]{64}$"))
    and .checks.checksum_storage_bytes==.rows*11008*8+64
    and .has_verified_cub_checksums and .cub_scratch_bytes==1
    and .tracked_device_bytes==(if .rows==1 then 4520651265 else 4521517569 end)
    and .real_scratch_bytes==(if .rows==1 then 201152 else 803264 end)
    and .column_gap_preparation==null and .has_verified_dispatch_owner_tags==false
    and .preparation.layout=="SeparateScalesValues" and .preparation.additional_prepared_bytes==47906816
    and .real_ffn_checks.guards_and_immutable_inputs
    and .real_ffn_checks.intermediate_capture_matches_normal_kernel
    and .real_ffn_checks.variant=="split-f32x4-128" and .real_ffn_checks.quantization==null
    and all([.real_ffn_checks.gate_vs_cpu,.real_ffn_checks.up_vs_cpu,.real_ffn_checks.swiglu_vs_cpu,
      .real_ffn_checks.complete_down_vs_cpu_on_actual_swiglu,.real_ffn_checks.isolated_down_vs_cpu_on_frozen_swiglu][];
      .max_abs_diff>=0 and .max_abs_diff<0.001 and .mean_abs_diff>=0 and .mean_abs_diff<=.max_abs_diff)
    and ([$finished[]|select(key==($check|key))]|.[0]|.checks==$check.checks
      and .has_intact_planes_and_inputs and .runtime_integration_allowed==false));"checksums/guards/real control")
| require(all($warmups[];.real_ffn.minimum_gpu_ms==1000 and .real_ffn.elapsed_gpu_ms>=1000
    and .real_ffn.iterations_per_batch==32 and .real_ffn.maximum_batches==256 and .real_ffn.batches<=256
    and all([.read_checksum,.cub_read_warmup][];.minimum_gpu_ms==100 and .gpu_ms>=100
      and .iterations_per_batch==256 and .maximum_batches==256 and .batches<=256)
    and .pitched_read_warmups==[] and .ordered_read_warmups==[]);"bounded separate warmups")
| require(([$pairs[]|key]|unique|sort)==$shapes
    and all($pairs|group_by(key)[];map(.trial)==[0,1,2,3,4])
    and all($pairs[];. as $pair|
      .tracked_device_bytes==(if .rows==1 then 4520651265 else 4521517569 end)
      and [.measurements[].stage]==(if .trial%2==0 then
        ["weight_read_checksum","cub_segmented_read_checksum","gate_up_swiglu","isolated_down","complete"]
        else ["complete","isolated_down","gate_up_swiglu","cub_segmented_read_checksum","weight_read_checksum"] end)
      and all(.measurements[];.enqueue_counters as $c|
        .iterations==10 and .gpu_ms>0 and .completed_unix_us>.started_unix_us
        and all(["device_allocs","device_frees","h2d_copies","h2d_bytes","d2h_copies","d2h_bytes","cublas_calls","stream_syncs"][];$c[.]==0)
        and $c.launches==(if .stage=="complete" then 20 else 10 end)
        and $c.stream_waits==(if .operation!=null or $pair.rows==1 then 0 elif .stage=="complete" then 20 else 10 end)
        and (if .operation!=null then
          (.stage=="cub_segmented_read_checksum") as $cub|
          .operation==(if $cub then "ffn_q8_cub_checksum" else "ffn_q8_weight_checksum_128" end)
          and .blocks_per_row==11008 and .threads==(if $cub then 256 else 128 end)
          and .shared_bytes==(if $cub then 0 else 1024 end) and .cub_scratch_bytes==1
          and .logical_unique_weight_bytes_per_call==47906816*$pair.rows
          and .has_inference_output==false and .has_measured_dram_traffic==false
        else .control_dispatch=={operation:"ffn_q8_split_f32x4_128",launches:(if .stage=="isolated_down" then 0 else 10 end)} end)));"actual diagnostic/control enqueue work")
| require(all($screens[];. as $screen|
    [$pairs[]|select(key==($screen|key))|
      [.measurements[]|select(.stage=="weight_read_checksum")|.gpu_ms]+
      [.measurements[]|select(.stage=="cub_segmented_read_checksum")|.gpu_ms]] as $raw|
    .result.pairs==$raw and .result.control_median_ms==($raw|map(.[0])|median)
    and .result.candidate_median_ms==($raw|map(.[1])|median)
    and .result.throughput_gain==(.result.control_median_ms/.result.candidate_median_ms-1)
    and .result.bootstrap_resamples==10000 and .result.bootstrap_seed==0
    and (.result.bootstrap_gain_ci95|length)==2 and .result.bootstrap_gain_ci95[0]<=.result.bootstrap_gain_ci95[1]
    and .result.bootstrap_gain_ci95[0]>=((($raw|map(.[0])|min)/($raw|map(.[1])|max))-1)
    and .result.bootstrap_gain_ci95[1]<=((($raw|map(.[0])|max)/($raw|map(.[1])|min))-1)
    and .result.passes_read_screen==(.result.throughput_gain>=0.10 and .result.bootstrap_gain_ci95[0]>0)
    and .result.passes_micro_screen==null and .one_row_read_gain_gate_percent==10
    and .runtime_integration_allowed==false);"raw pairs, medians, bounds and read-only gate")
| [$pairs[]|.gpu_before,.gpu_after] as $gpu
| require(($gpu|length)==60 and all($gpu[];
    (.device|contains("GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852, Tesla M40 24GB"))
    and (.device|contains(", P0,")) and (.device|contains("1113 MHz, 3004 MHz"))
    and (.processes|test("^"+$pid+", target/release/examples/q8_execution_bench, [0-9]+ MiB$")));"exclusive stable GPU")
| ([$pairs[].measurements[]]|sort_by(.started_unix_us)) as $intervals
| require(($intervals|length)==150 and all(range(1;150);
    $intervals[.].started_unix_us>=$intervals[.-1].completed_unix_us);"serial primary intervals")
| {passes_audit:true,source:$m.revision,executable_sha256:$m.executable_sha256,
    pairs:30,serial_primary_intervals:150,gpu_boundaries:60,tracked_device_bytes_after:0,
    runtime_integration_allowed:false,screens:$screens,
    cases:($pairs|group_by(key)|map({layer_id:.[0].layer_id,rows:.[0].rows,
      median_gpu_ms:([.[].measurements[]]|group_by(.stage)|map({key:.[0].stage,value:map(.gpu_ms)|median})|from_entries)})),
    interpretation:"source-equivalent integer read reference, not inference or measured DRAM cost"}
