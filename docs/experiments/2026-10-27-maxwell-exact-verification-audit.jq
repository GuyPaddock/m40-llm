# jq -s --arg pid PID --slurpfile gold QUALIFIED_TRACE -f THIS SCREEN
def require($ok;$why):if $ok then . else error($why) end;
def median:sort|.[length/2|floor];
def mad_fraction: . as $v|median as $m|$v|map((.-$m)|fabs)|median/$m;
. as $all
| map(select(.kind=="verification_manifest"))|require(length==1;"manifest count")|.[0] as $m
| $all|map(select(.kind=="verification_trial")) as $trials
| $all|map(select(.kind=="verification_preflight")) as $preflight
| $all|map(select(.kind=="verification_rejection_cost")) as $reject
| $all|map(select(.kind=="verification_summary")) as $summaries
| $all|map(select(.kind=="verification_complete")) as $done
| ($gold[]|select(.kind=="q8_runtime_manifest")) as $g
| ($gold[]|select(.kind=="q8_runtime_browser_generation" and .label=="candidate")) as $browser
| require($m.revision=="885cc98acdede4bee913d94ee4592331e78c5bd5" and $m.worktree=="" and $m.build_profile=="release"
    and $m.executable_sha256=="6ef35663ed85ae60d621d1c43be7493343065b9a15ad1fda2d4d91f48ad0376d"
    and $m.model_sha256==$g.candidate.sha256
    and $m.trace_sha256=="8b499787179f484aeb32e37caf837208e0994da4995918e18e34ba922d3d949e"
    and $m.runtime_config==$g.candidate_config and $m.environment==$g.environment
    and ($m.prepared_weights|del(.preparation_elapsed_ms,.model_hash_elapsed_ms))==($g.prepared_candidate|del(.preparation_elapsed_ms,.model_hash_elapsed_ms))
    and ($m.prepared_weights.tensors|length)==109 and $m.prepared_weights.total_weight_payload_bytes==6197100544
    and $m.prompt_token_ids==$browser.prompt_token_ids and ($m.prompt_token_ids|length)==65
    and $m.reference_token_ids==$browser.generation.generated_token_ids and ($m.reference_token_ids|length)==512
    and $m.widths==[1,2,4,8] and $m.trials==5;"frozen source/configuration/cache/trace")
| require(($all|length)==99 and ($trials|length)==20 and ($preflight|length)==4
    and ([$preflight[].result.width]|sort)==[1,2,4,8]
    and ($reject|length)==70 and ($summaries|length)==3
    and ([$summaries[].width]|sort)==[2,4,8]
    and ($done|length)==1 and $done[0].is_complete and ($done[0].is_preflight_only|not)
    and $done[0].tracked_device_bytes_before==0 and $done[0].tracked_device_bytes_after==0;"coverage/cleanup")
| require(all($trials|group_by(.trial)[];map(.width)|sort==[1,2,4,8])
    and all($trials|group_by(.width)[];map(.trial)==[0,1,2,3,4])
    and all([$preflight[],$trials[]][];.result as $r|
      $r.has_exact_output and $r.generated_token_ids==$m.reference_token_ids and $r.generated_tokens==512
      and $r.stop_reason=="length" and $r.failure==null and $r.first_divergence_step==null
      and $r.prompt_tokens==65 and $r.target_input_rows==511 and $r.committed_len==576
      and $r.layer_lengths==[range(36)|576]
      and $r.verification_batches==(if $r.width==1 then 0 else 511/$r.width|floor end)
      and $r.serial_steps==(if $r.width==1 then 511 else 511%$r.width end)
      and $r.session_scratch_bytes==689668 and $r.resident_kv_bytes==37748736
      and $r.observed_tracked_device_high_water_bytes==({"1":6198064652,"2":6288450068,"4":6288865828,"8":6289697348}[$r.width|tostring])
      and $r.prefill_ms>0 and $r.decode_ms>0 and $r.total_ms>=$r.prefill_ms+$r.decode_ms
      and $r.resident_total_tps==512000/$r.total_ms
      and $r.counters.cublas_calls==252*(1+$r.verification_batches)
      and (if .kind=="verification_preflight" and $r.width==1 then
        $r.counters.device_allocs==43 and $r.counters.device_frees==26
        else $r.counters.device_allocs==$r.counters.device_frees end)
      and $r.started_unix_us<$r.completed_unix_us);"exact outputs/real row work/lengths/memory")
| require(all($reject|group_by(.result.width,.result.accepted_prefix)[];map(.trial)==[0,1,2,3,4])
    and all($reject[];.result as $r|($r.width==2 or $r.width==4 or $r.width==8)
      and $r.accepted_prefix>=0 and $r.accepted_prefix<$r.width and $r.has_exact_correction
      and $r.emitted_tokens==$r.accepted_prefix+1 and $r.prefix_tokens==65 and $r.elapsed_ms>0)
    and all([2,4,8][];. as $k|([$reject[]|select(.result.width==$k)|.result.accepted_prefix]|unique)==[range($k)]);"measured correction coverage")
| require(all($summaries[];. as $s|
    [$trials[]|select(.width==1)|.result] as $control|
    [$trials[]|select(.width==$s.width)|.result] as $candidate|
    $s.pairs==[range(5)|{control_tps:$control[.].resident_total_tps,candidate_tps:$candidate[.].resident_total_tps}]
    and $s.control.median==($control|map(.resident_total_tps)|median)
    and $s.candidate.median==($candidate|map(.resident_total_tps)|median)
    and $s.control.mad_over_median==($control|map(.resident_total_tps)|mad_fraction)
    and $s.candidate.mad_over_median==($candidate|map(.resident_total_tps)|mad_fraction)
    and $s.gain==($s.candidate.median/$s.control.median-1)
    and $s.bootstrap_gain_ci95[0]<=$s.bootstrap_gain_ci95[1] and $s.bootstrap_gain_ci95[1]<0
    and $s.bootstrap_gain_ci95[0]>=((($candidate|map(.resident_total_tps)|min)/($control|map(.resident_total_tps)|max))-1)
    and $s.bootstrap_gain_ci95[1]<=((($candidate|map(.resident_total_tps)|max)/($control|map(.resident_total_tps)|min))-1)
    and $s.promotion_threshold==0.03 and ($s.passes_resident_screen|not)
    and ($s.needs_fresh_confirmation|not) and ($s.needs_variation_extension|not)
    and ($s.is_http_claim|not) and ($s.has_real_drafter|not)
    and $s.serial_mean_step_ms==($control|map(.decode_ms/511)|median)
    and $s.mean_full_verification_ms==($candidate|map(.mean_verification_batch_ms)|median)
    and $s.rejection_cycle_ms_by_accepted_prefix==([range($s.width)|. as $a|
      [$reject[]|select(.result.width==$s.width and .result.accepted_prefix==$a)|.result.elapsed_ms]|median]+[$s.mean_full_verification_ms])
    and all($s.acceptance_curves[];. as $curve|
      .per_token_acceptance_probability as $p|
      ([range($s.width+1)|. as $a|pow($p;$a)*(if $a==$s.width then 1 else 1-$p end)*$s.rejection_cycle_ms_by_accepted_prefix[$a]]|add) as $cost|
      ([range($s.width+1)|. as $a|pow($p;$a)*(if $a==$s.width then 1 else 1-$p end)*(if $a==$s.width then $s.width else $a+1 end)]|add) as $tokens|
      (.target_ms_per_cycle-$cost|fabs)<1e-9 and (.expected_tokens_per_cycle-$tokens|fabs)<1e-9
      and (.max_drafter_ms_to_break_even-($s.serial_mean_step_ms*$tokens-$cost)|fabs)<1e-9
      and (.max_drafter_ms_for_three_percent_gain-($s.serial_mean_step_ms*$tokens/1.03-$cost)|fabs)<1e-9
      and .max_drafter_ms_to_break_even<0));"raw summaries/negative ideal budgets/measured cost curves")
| [$trials[]|.gpu_before,.gpu_after]+[$reject[]|.gpu_before,.gpu_after] as $gpu
| require(all($gpu[];(.device|contains("GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852, Tesla M40 24GB"))
    and (.device|contains(", P0,")) and (.device|contains("1113 MHz, 3004 MHz"))
    and (.processes|test("^"+$pid+", target/release/examples/q8_execution_bench, [0-9]+ MiB$")));"sampled GPU isolation/clocks")
| ([$trials[].result]|sort_by(.started_unix_us)) as $intervals
| require(all(range(1;20);$intervals[.].started_unix_us>=$intervals[.-1].completed_unix_us);"serial primary intervals")
| {passes_audit:true,primary_generations:20,preflight_generations:4,generated_ids_per_run:512,
    correction_cycles:70,gpu_boundaries:($gpu|length),tracked_device_bytes_after:0,
    source:$m.revision,executable_sha256:$m.executable_sha256,summaries:$summaries,
    medians:($trials|group_by(.width)|map({width:.[0].width,prefill_ms:map(.result.prefill_ms)|median,
      decode_ms:map(.result.decode_ms)|median,total_ms:map(.result.total_ms)|median,
      resident_tps:map(.result.resident_total_tps)|median})),
    pursue_eagle:false,is_http_claim:false,changes_runtime_defaults:false}
