# jq -s --slurpfile capture ...ffn-capture.jsonl -f THIS SCREEN
def require($ok;$why): if $ok then . else error($why) end;
def median: sort|.[length/2|floor];
def key: [.family,.layer_id,.rows];
def one($kind): map(select(.kind==$kind))|require(length==1;"record multiplicity")|.[0];
def expected_n: if .family=="down" then 2048 else 151936 end;
def expected_k: if .family=="down" then 11008 else 2048 end;
def expected_threads: if .family=="down" and .rows==1 then 256 else 128 end;
def expected_resident: if .family=="down" then (if .rows==1 then 4520571392 else 4521198080 end)
  else (if .rows==1 then 4473679040 else 4477350080 end) end;
. as $records
| one("prepared_read_manifest") as $m
| $records|one("prepared_read_complete") as $done
| $records|map(select(.kind=="prepared_read_pair")) as $pairs
| $records|map(select(.kind=="prepared_read_validation")) as $checks
| $records|map(select(.kind=="prepared_read_case_complete")) as $finished
| $records|map(select(.kind=="prepared_read_warmup")) as $warmups
| [["down",0,1],["down",0,4],["down",17,1],["down",17,4],["down",35,1],["down",35,4],["vocabulary",null,1],["vocabulary",null,4]] as $shapes
| require($m.schema_version==2 and $m.revision=="375c48898cf20517c88ef50f2d7525e1a91fde8e"
    and $m.executable_sha256=="fab4704d43f5b91923d90a9b7b67b03b075d0cd80ce988442c7272682aa0c1df"
    and $m.worktree=="" and $m.build_profile=="release"
    and $m.model_sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
    and $m.fixture_sha256=="eb51828ef778189567ae01fdd910659025c92ae6f5b2d76c989c314e650d0687"
    and $m.capture_report_sha256=="d9e55963e5872bbc48cfb3a1f7ac217429974d0a48237aabaaab207f5fc95801"
    and $m.execution_configuration==$capture[0].execution_configuration
    and ($m.prepared_weights|del(.preparation_elapsed_ms,.model_hash_elapsed_ms))
      ==($capture[0].prepared_weights|del(.preparation_elapsed_ms,.model_hash_elapsed_ms))
    and ($m.prepared_weights.tensors|length)==37 and $m.original_weight_bytes==3279519744 and $m.tracked_device_bytes==4472455168
    and $m.rows==[1,4] and $m.trials==5 and $m.iterations==10 and $m.is_weight_read_diagnostic
    and $m.candidate_kernel=="projection_q8_weight_checksum" and ($m.runtime_defaults_changed|not) and ($m.is_e2e_claim|not)
    and $m.cuda_flags==["-O3","sm_52","compute_52","no fast math"];"frozen control/source")
| require(($records|length)==66 and ($pairs|length)==40
    and all([$checks,$finished,$warmups][];length==8 and (map(key)|sort)==$shapes)
    and $done.is_complete and $done.cases==8 and $done.device_bytes_before==0 and $done.device_bytes_after==0;"coverage/cleanup")
| require(all($checks[];. as $check|
    .checks.has_matching_cpu_checksums and .checks.has_intact_output_guards and .checks.scale_reads_per_block==4
    and .checks.checksum_count==.rows*expected_n and (.checks.checksum_sha256|test("^[a-f0-9]{64}$"))
    and .checks.checksum_storage_bytes==(.rows*expected_n*4+64) and .resident_bytes==expected_resident
    and .configuration.columns==4 and .configuration.values_per_task==8
    and .configuration.k==expected_k and .configuration.n==expected_n and .configuration.rows==.rows
    and .configuration.threads==expected_threads and .configuration.shared_bytes==expected_threads*16
    and .configuration.dynamic_barriers_per_block==(if expected_threads==256 then 9 else 8 end)
    and .configuration.planes.resident_bytes==(if .family=="down" then 23953408 else 330612736 end)
    and (.plane_sha256|length)==2 and all(.plane_sha256[];test("^[a-f0-9]{64}$"))
    and (if .family=="down" then .real_ffn_checks.variant=="split-f32x4-128" and .real_ffn_checks.quantization==null
      and .real_ffn_checks.guards_and_immutable_inputs and .real_ffn_checks.intermediate_capture_matches_normal_kernel
      and .gate_up_preparation.additional_prepared_bytes==47906816
      and .real_scratch_bytes==(if .rows==1 then 201152 else 803264 end)
      and all([.real_ffn_checks.gate_vs_cpu,.real_ffn_checks.up_vs_cpu,.real_ffn_checks.swiglu_vs_cpu,
        .real_ffn_checks.complete_down_vs_cpu_on_actual_swiglu,.real_ffn_checks.isolated_down_vs_cpu_on_frozen_swiglu][];
        .max_abs_diff>=0 and .max_abs_diff<0.001 and .mean_abs_diff>=0 and .mean_abs_diff<=.max_abs_diff)
    else .cpu_sample_max_abs_diff>=0 and .cpu_sample_max_abs_diff<0.001 and .real_scratch_bytes==(if .rows==1 then 616064 else 2463872 end) end)
    and (([$finished[]|select(key==($check|key))]|.[0]) as $end|
      $end.checks==$check.checks and $end.has_intact_planes_and_inputs and ($end.runtime_integration_allowed|not)
      and (if $check.family=="vocabulary" then $end.has_matching_real_control_bits else true end)));"CPU checksum/real control/ownership")
| require(all($warmups[];.read_checksum.minimum_gpu_ms==100 and .read_checksum.gpu_ms>=100
    and .read_checksum.maximum_batches==256 and .read_checksum.batches<=256
    and (if .family=="down" then .real_ffn.minimum_gpu_ms==1000 and .real_ffn.elapsed_gpu_ms>=1000 and .real_ffn.maximum_batches==256
      else .real_projection.minimum_gpu_ms==1000 and .real_projection.gpu_ms>=1000 and .real_projection.maximum_batches==256 end));"bounded warmup")
| require(([$pairs[]|key]|unique|sort)==$shapes and all($pairs|group_by(key)[];map(.trial)==[0,1,2,3,4])
    and all($pairs[];. as $pair|
      .resident_bytes==expected_resident and [.measurements[].stage]==
        ((if .family=="down" then ["weight_read_checksum","gate_up_swiglu","isolated_down","complete"] else ["weight_read_checksum","vocabulary"] end)
          |if $pair.trial%2==0 then . else reverse end)
      and all(.measurements[];.enqueue_counters as $c|
        .iterations==10 and .gpu_ms>0 and .completed_unix_us>.started_unix_us
        and all(["device_allocs","device_frees","h2d_copies","h2d_bytes","d2h_copies","d2h_bytes","cublas_calls","stream_syncs"][];$c[.]==0)
        and $c.launches==(if .stage=="complete" then 20 else 10 end)
        and $c.stream_waits==(if .stage=="weight_read_checksum" or $pair.family=="vocabulary" or $pair.rows==1 then 0 elif .stage=="complete" then 20 else 10 end)
        and (if .stage=="weight_read_checksum" or .stage=="vocabulary" then .geometry.threads==($pair|expected_threads)
          and .geometry.columns==4 and .geometry.values_per_task==8 and .geometry.shared_bytes==16*($pair|expected_threads)
          and .geometry.blocks_per_row==(($pair|expected_n)/4)
          and .logical_unique_weight_bytes_per_call==$pair.rows*(if $pair.family=="down" then 23953408 else 330612736 end)
          and (.has_measured_dram_traffic|not) and .has_inference_output==(.stage=="vocabulary")
          and .operation==(if .stage=="vocabulary" then "gemm_f32xq8_separate_f32" else "projection_q8_weight_checksum" end)
        else .control_dispatch=={projection_operation:"gemm_f32xq8_separate_f32",projection_launches:(if .stage=="gate_up_swiglu" then 0 else 10 end),
            gate_up_operation:"ffn_q8_split_f32x4_128",gate_up_launches:(if .stage=="isolated_down" then 0 else 10 end)} end)));"actual work/dispatch/side effects")
| [$pairs[]|.gpu_before,.gpu_after] as $gpu
| require(all($gpu[];(.device|contains("GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852, Tesla M40 24GB"))
    and (.device|contains(", P0,")) and (.device|contains("1113 MHz, 3004 MHz"))
    and (.processes|test("^3022210, target/release/examples/q8_execution_bench, [0-9]+ MiB$")));"GPU isolation/clocks")
| ([$pairs[].measurements[]]|sort_by(.started_unix_us)) as $intervals
| require(($intervals|length)==140 and all(range(1;140);$intervals[.].started_unix_us>=$intervals[.-1].completed_unix_us);"serial primary intervals")
| {passes_audit:true,source:$m.revision,executable_sha256:$m.executable_sha256,pairs:40,gpu_boundaries:80,serial_primary_intervals:140,
    validation_records:8,tracked_device_bytes_after:0,runtime_integration_allowed:false,
    cases:($pairs|group_by(key)|map(. as $shape|
      ([$shape[].measurements[]]|group_by(.stage)|map({key:.[0].stage,value:(map(.gpu_ms)|median)})|from_entries) as $times|
      (if .[0].family=="down" then $times.isolated_down else $times.vocabulary end) as $real|
      {family:.[0].family,layer_id:.[0].layer_id,rows:.[0].rows,median_gpu_ms:$times,
        checksum_fraction_of_real_projection:($times.weight_read_checksum/$real),real_projection_minus_checksum_ms:($real-$times.weight_read_checksum),
        stages:([$shape[].measurements[]]|group_by(.stage)|map({stage:.[0].stage,min_gpu_ms:map(.gpu_ms)|min,max_gpu_ms:map(.gpu_ms)|max}))})),
    interpretation:"checksum-only original-mapping diagnostic, not measured DRAM, a rigorous lower bound, removable inference cost or E2E gain"}
