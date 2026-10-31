# Behavioral fixture only. Reuse real tensor/input identities, replacing every
# service time with explicit synthetic values. Never use this for feasibility.
. as $original |
(.costs | map(select(.arm == "q8"))) as $cases |
(.costs | map(select(.arm == "q8" and .batch == 1))) as $weights |
{cpu:([
  {kind:"cpu_projection_manifest",schema_version:1,revision:"synthetic-test-fixture",is_validation_only:false,
   fixture_scope:"SYNTHETIC 10000ns AVX2/20000ns scalar costs; not hardware measurements",
   model_sha256:$original.source_manifest.model_sha256,input_sha256:$original.source_manifest.fixture_sha256,
   capture_report_sha256:$original.source_manifest.capture_report_sha256,iterations:3,
   placement:{threads:1,cpu:15,numa_node:0}},
  ($weights[] | {kind:"cpu_projection_setup",layer,family,source,q8_payload_bytes}),
  ($cases[] as $case |
    {layer:$case.layer,family:$case.family,batch:$case.batch,columns:$case.source.shape[0],rows:$case.source.shape[1],
     input_sha256:$case.input_sha256,row_positions:$case.row_positions} as $key |
    {kind:"cpu_projection_validation",case:$key,
     avx2:{elements:($case.batch*$case.source.shape[1]),max_tolerance_ratio:0},
     scalar:{elements:($case.batch*$case.source.shape[1]),max_tolerance_ratio:0}},
    (range(0;5) as $pair | {kind:"cpu_projection_pair",case:$key,pair:$pair,is_confirmation:false,
      iterations:3,avx2_first:($pair%2==1),scalar_ms:0.02,avx2_ms:0.01}),
    {kind:"cpu_projection_comparison",case:$key,is_confirmation:false,timing_error:null,comparison:{pairs:5}}),
  {kind:"cpu_projection_complete",is_complete:true,cases:36,generated_token_count:0}
]),
transfers:([
  {kind:"cooperative_transfer_manifest",schema_version:1,revision:"synthetic-test-fixture",is_validation_only:false,
   fixture_scope:"SYNTHETIC 1000ns H2D/500ns D2H costs; not hardware measurements",
   placement:{threads:1,cpu:15,numa_node:0},iterations:20,
   gpu_state:"GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852, synthetic fixture",has_overlap_measurement:false},
  (([([2048,11008][]) as $width | ([1,2,4,8][]) as $batch |
      {name:("activation_"+($width|tostring)+"_batch_"+($batch|tostring)),bytes:($width*$batch*4)}] +
    [{name:"fp16_kv_token_one_layer",bytes:1024},{name:"fp16_kv_token_36_layers",bytes:36864},
     {name:"fp16_kv_1k_one_layer",bytes:1048576},{name:"fp16_kv_8k_one_layer",bytes:8388608},
     {name:"one_original_q8_projection",bytes:23953408}])[] as $case |
    {kind:"cooperative_transfer_validation",case:$case,is_bit_exact:true,guards_unchanged:true},
    (range(0;5) as $pair | {kind:"cooperative_transfer_pair",case:$case,comparison:"pageable_h2d_vs_d2h",
      pair:$pair,order:(if $pair%2==0 then [0,1] else [1,0] end),observations:[
        {arm:"h2d",iterations:20,logical_transfer_bytes_per_operation:$case.bytes,wall_ns_per_operation:1000},
        {arm:"d2h",iterations:20,logical_transfer_bytes_per_operation:$case.bytes,wall_ns_per_operation:500}]}),
    {kind:"cooperative_transfer_costs",case:$case,comparison:"pageable_h2d_vs_d2h",timing_error:null,
      paired_operation_rate_comparison:{pairs:5}},
    {kind:"cooperative_transfer_case_complete",case:$case,tracked_device_bytes_before:0,tracked_device_bytes_after:0}),
  {kind:"cooperative_transfer_complete",is_complete:true,transfer_cases:13,generated_token_count:0}
]),
gpu:($original | .cost_scope = "SYNTHETIC native/prepared 100ns/50ns; real identities only, not measurement data" |
  .costs |= map(. as $cost | .raw_observations |= map(.wall_ms = (if $cost.arm == "q8" then 0.0001 else 0.00005 end)))),
capacity:{schema_version:1,fixture_scope:"SYNTHETIC capacity admission fixture",
  gpu:{uuid:"GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852",capacity_bytes:24159191040},
  nodes:[{numa_node:0,capacity_bytes:64000000000},{numa_node:1,capacity_bytes:64000000000}]}}
