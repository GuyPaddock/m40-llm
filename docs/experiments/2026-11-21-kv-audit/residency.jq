# Reconstruct backing sizes independently of the allocator's printed totals.
.[0] as $base | $base.capacity as $n | ($base.arm=="dense") as $dense |
if map(.kind)!=["residency_baseline","cache_allocated","cache_populated","cache_populated","cache_released"] or
   $base.model_sha256!="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" or
   $base.executable_sha256!="02b0eec975481a75c0b29217968bcd50a85bbed17892d9c76c77ca746fe631ea" or
   $base.physical_layer_slots!=36 or $base.logical_sequences!=1 or
   [$n,$base.arm]!=[$capacity,$arm] or [.[1].occupied,.[2].occupied_per_layer,.[3].occupied_per_layer]!=[0,64,$n] or
   any(.[];.memory.cuda_used_bytes+.memory.cuda_free_bytes!=.memory.cuda_total_bytes or .memory.nvml_process_mib<=0)
then error("wrong geometry, source or lifecycle") else . end |
($n*36*2*128*4) as $dense_bytes |
([1,(($n-1024)/32|ceil)]|max) as $blocks |
{recent_fp16_bytes:(if $dense then 0 else 36*2*128*1024*4 end),
 old_k_fp16_bytes:(if $dense then 0 else $n*36*2*128*2 end),
 q4_v_codes_bytes:(if $dense then 0 else $n*36*2*128/2 end),
 q4_v_scale_bytes:(if $dense then 0 else $n*36*2*4 end),
 summaries_counts_and_seq_map_bytes:(if $dense then 0 else 36*($blocks*(2*128*12+4)+4) end),
 dense_kv_payload_bytes:(if $dense then $dense_bytes else 0 end)} as $components |
($components|[.[]]|add) as $reported |
(if $dense then 144 else 0 end) as $map |
if .[1].payloads!=($components+{seq_map_bytes_excluded_from_reported_total:$map,
    known_cuda_malloc_requests_bytes:($reported+$map),cache_reported_bytes:$reported,dense_equivalent_bytes:$dense_bytes}) then
    error("backing totals differ") else . end |
{capacity:$n,arm:$arm,source:$base.revision,payloads:.[1].payloads,allocation_wall_ms:.[1].allocation_wall_ms,
 cuda_used_bytes:map(.memory.cuda_used_bytes),rust_tracked_bytes:map(.memory.rust_tracked_bytes),
 nvml_boundary_mib:map(.memory.nvml_process_mib),sampled_process:$samples,
 cuda_allocation_delta_bytes:(.[1].memory.cuda_used_bytes-$base.memory.cuda_used_bytes),
 cuda_post_release_delta_bytes:(.[4].memory.cuda_used_bytes-$base.memory.cuda_used_bytes)}
