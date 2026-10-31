# jq -s --slurpfile capture FROZEN_EVENTS -f THIS REPORT
def require($ok;$why): if $ok then . else error($why) end;
def median: sort|.[length/2|floor];
. as $records
| map(select(.kind=="disabled_diagnostics_manifest"))
| require(length==1;"manifest multiplicity")|.[0] as $m
| $records|map(select(.kind=="disabled_diagnostics_pair")) as $pairs
| $records|map(select(.kind=="disabled_diagnostics_complete"))
| require(length==1;"completion multiplicity")|.[0] as $done
| ["attn_norm","qkv_project","rope_q","kv_append_rope_k","attention","out_project",
    "attn_residual","ffn_norm","mlp_gate_up_swiglu","mlp_down","mlp_residual"] as $stages
| [$capture[]|select(.session_id==2 and .generated_step==255)] as $samples
| require(($records|length)==7 and ($pairs|map(.trial))==[0,1,2,3,4]
    and $m.schema_version==1 and $m.build_profile=="release" and $m.worktree==""
    and $m.source=="a592debdfd46f974f84acace7256451b3adc0ed6"
    and $m.executable_sha256=="dd18d1521904cacf4c166f018feeb4ca4126a487477a044c202dec1007265e06"
    and $m.trace_sha256=="4eeec0020ed346341cfd816cc3ef28ae1e98ae88f2b7a6f24a378b75927b126c"
    and $m.cuda_visible_devices=="-1" and $m.logs_disabled==true
    and ($m.affinity|endswith("affinity list: 0"))
    and ($m.cpu|contains("Xeon(R) CPU E5-2697A v4"))
    and $m.counters_before=={} and $m.device_bytes_before==0
    and $m.layers==36 and $m.stages_per_layer==$stages and $m.operations_per_nominal_token==396
    and $m.session_id==2 and $m.generated_step==255 and $m.absolute_position==319
    and $m.nominal_tokens_per_replay==512 and $m.iterations_per_arm==8 and $m.pairs==5
    and $m.expected_checksum==(396*397/2*512*8) and $m.qualified_http_ms==10417.843301;
    "clean CPU-only identity and fixed work")
| require(($samples|length)==1 and $samples[0].absolute_position==319
    and ([$samples[0].stages[]|select(.layer_id!=null)|[.layer_id,.stage]])
      ==[range(0;36) as $layer|$stages[]|[$layer,.]];
    "independent frozen stage coverage")
| require(all($pairs[];. as $pair|(.arms|length)==2
    and [.arms[].has_diagnostics]==(if .trial%2==0 then [false,true] else [true,false] end)
    and all(.arms[];.checksum==$m.expected_checksum and .elapsed_ns>0
      and ((.elapsed_ms_per_replay-.elapsed_ns/1000000/8)|fabs)<1e-8
      and .completed_unix_us>.started_unix_us
      and .elapsed_ns<=(.completed_unix_us-.started_unix_us)*1000+2000));
    "paired traversal, timing totals and checksums")
| [$pairs[].arms[]] as $arms
| require(all(range(1;10);$arms[.].started_unix_us>=$arms[.-1].completed_unix_us)
    and $done.pairs==5 and $done.counters_unchanged==true
    and $done.counters_after==$m.counters_before and $done.device_bytes_after==0;
    "serial work and cleanup")
| [$arms[]|select(.has_diagnostics)|.elapsed_ms_per_replay] as $wrapper
| [$arms[]|select(.has_diagnostics|not)|.elapsed_ms_per_replay] as $base
| [$pairs[]|([.arms[]|select(.has_diagnostics)|.elapsed_ms_per_replay][0]
    -[.arms[]|select(.has_diagnostics|not)|.elapsed_ms_per_replay][0])] as $excess
| {passes_audit:true,source:$m.source,executable_sha256:$m.executable_sha256,
    pairs:5,iterations_per_arm:8,nominal_tokens_per_replay:512,operations_per_nominal_token:396,
    wrapper_ms_per_replay:($wrapper|median),wrapper_min_ms:($wrapper|min),wrapper_max_ms:($wrapper|max),
    traversal_ms_per_replay:($base|median),paired_excess_ms_per_replay:($excess|median),
    wrapper_us_per_nominal_token:(($wrapper|median)*1000/512),
    observed_cost_fraction_of_qualified_http:(($wrapper|median)/$m.qualified_http_ms),
    maximum_cost_fraction_of_qualified_http:(($wrapper|max)/$m.qualified_http_ms),
    warrants_standalone_runtime_cost_study:(($wrapper|median)/$m.qualified_http_ms>=0.03),
    counters_unchanged:true,device_bytes_after:0,changes_runtime_defaults:false,
    is_e2e_speedup_claim:false,is_inference_latency_bound:false,
    scope:$m.scope}
