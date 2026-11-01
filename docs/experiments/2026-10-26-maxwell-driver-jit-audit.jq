def require($ok;$why): if $ok then . else error($why) end;
def one($kind): map(select(.kind==$kind))|require(length==1;"record multiplicity")|.[0];
def launches: map(select(.kind=="cuda_launch"));
def modules: map(select(.kind=="cuda_module"));
def dispatch: launches|map({domain,api,symbol,grid,block,shared_bytes,launch_status});
def trace_ok($force;$count): . as $trace
  | one("cuda_trace_start") as $start
  | $trace|one("cuda_trace_environment") as $env
  | $trace|one("cuda_trace_complete") as $done
  | $trace|require($start.is_timing_measurement==false and $start.captures_modules==true
      and $start.module_record_limit==64 and $start.module_byte_limit==268435456
      and $env=={kind:"cuda_trace_environment",cuda_force_ptx_jit:$force,cuda_disable_ptx_jit:null,cuda_cache_disable:"1"}
      and $done.launch_records==$count and (launches|length)==$count
      and $done.dropped_launches==0 and $done.failed_launches==0 and $done.unknown_symbols==0
      and $done.unsubscribe_status==0 and $done.module_records==(modules|length)
      and $done.module_bytes==([modules[].cubin_bytes]|add//0)
      and all(launches[];.launch_status==0)
      and all(modules[];.cubin_bytes>0 and .cubin_bytes<=67108864 and (.path|length)>0);
      "trace completeness/settings/status");
. as $e
| $e.identity as $i
| require($i.observation_revision=="b266a7329468470957f596a955f3b8de8a21a6bd"
    and $i.observer_source_sha=="f6d738c4dd65ff39e407ca3b4b7150181d64d9d3d47e8460b49dc4f64efafa43"
    and $i.observer_binary_sha=="cc155e8fdda002984c2354eac831a0d97be2d699b48244289072efe32a3bda26"
    and $i.with_cublas_executable_sha=="88e0cd0221bfb739b927f1fabf653dd889d250564bb0b7efa1f3d46c1627ac47"
    and $i.without_cublas_executable_sha=="7696b754bf3cb9ec70833e6ca265f3dea0f220899a3464402eb138edfbb38189"
    and $i.project_words=={native_sha256:"7a954317700bf84cdc40440471d00b2dfb2b5b126cfb562e88c5566b8449970a",
      forced_sha256:"7a954317700bf84cdc40440471d00b2dfb2b5b126cfb562e88c5566b8449970a",
      cublas_enabled_sha256:"7a954317700bf84cdc40440471d00b2dfb2b5b126cfb562e88c5566b8449970a",word_count:100400,functions:135}
    and $i.resources.native_sha256=="2ea1b13638fb0d47973fd7a402da72d7d15bcc756e795779ea376b0dd8487eb6"
    and $i.resources.forced_sha256==$i.resources.native_sha256
    and $i.is_timing_measurement==false and $i.changes_runtime_defaults==false;
    "clean source and compiled witnesses")
| ($e.native|trace_ok("0";1430)) as $native
| ($e.forced|trace_ok("1";1430)) as $forced
| ($e.enabled_native|trace_ok("0";1430)) as $enabled
| ($e.enabled_forced|trace_ok("1";0)) as $failed
| require(($native|dispatch)==($forced|dispatch) and ($native|dispatch)==($enabled|dispatch)
    and ($native|modules|length)==1 and ($forced|modules|length)==1
    and ($enabled|modules|length)==2 and ($failed|modules|length)==0
    and $i.native_module=={sha256:"546639021f37d31d436cb363be529665aba971739ed199de0218572d579468c9",bytes:1100416}
    and $i.forced_module=={sha256:"475fa3218d0e7114483cef69004cdbbf56e5c321fafb4d437fd8651d71c814ac",bytes:1100960}
    and ($native|modules|.[0].cubin_bytes)==$i.native_module.bytes
    and ($forced|modules|.[0].cubin_bytes)==$i.forced_module.bytes
    and ([$forced|launches[]|select(.domain=="driver" and (.symbol|contains("19gate_up_split_f32x4EP")))]|length)==36;
    "same completed-test dispatch, distinct module containers")
| require($e.probe_native=={force_ptx_jit:"0",cache_disable:"1",cuda_select_status:0,
      cuda_initialize_status:0,cublas_create_status:0,cublas_destroy_status:0}
    and $e.probe_forced=={force_ptx_jit:"1",cache_disable:"1",cuda_select_status:0,
      cuda_initialize_status:0,cublas_create_status:3,cublas_destroy_status:-1};
    "explicit cuBLAS compatibility failure, not passing FFN coverage")
| require($e.smoke_output=={actual:96,expected:96,is_pass:true}
    and ($e.smoke|one("cuda_trace_start").captures_modules)==true
    and ($e.smoke|one("cuda_trace_environment")).cuda_force_ptx_jit=="1"
    and ($e.smoke|one("cuda_trace_complete"))=={kind:"cuda_trace_complete",launch_records:5,
      dropped_launches:0,failed_launches:1,unknown_symbols:0,unsubscribe_status:0,module_records:1,module_bytes:2184}
    and ($e.image_error|map(select(.kind=="cuda_module_error"))|length)==1
    and ($e.image_error|map(select(.kind=="cuda_trace_complete"))|length)==0
    and ($e.setting_error|map(select(.kind=="cuda_module_error"))|length)==1;
    "observer GPU output and explicit capture failures")
| {passes_audit:true,observation_revision:$i.observation_revision,
    complete_launch_records_per_successful_ffn_run:1430,driver_launches_per_successful_run:715,
    qualified_gate_launches_per_run:36,project_disassembled_entries:135,project_encoded_words:100400,
    has_identical_native_and_jit_project_words:true,has_identical_resource_reports:true,
    has_forced_jit_cublas_ffn_coverage:false,cublas_probe_forced_status:3,
    should_run_new_kernel_timing:false,should_enable_global_forced_jit:false,
    is_performance_claim:false,changes_runtime_defaults:false,
    scope:"untimed loaded-code observation and primitive tests, not full-model quality or HTTP"}
