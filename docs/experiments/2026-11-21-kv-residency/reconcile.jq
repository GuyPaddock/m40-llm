# Read the two constructor records as raw lines, with matched quality rows supplied.
def fields: [splits(" ")|select(contains("="))|split("=")|{key:.[0],value:(.[1]|tonumber? // .)}] | from_entries;
map(fields) as $runs |
if ($runs|map(.top_blocks)|sort)!=[8,64] or ($rows|map(.arm)|sort)!=["dense","quantized_all","quantized_top8"] then error("missing arms") else . end |
$manifest[0] as $m | $rows[0].cache.physical_lengths|length as $layers |
($rows|map(.cache.tracked_device_bytes)|unique) as $tracked |
if ($tracked|length)!=1 or $layers!=36 or $m.capacity!=3072 or $m.recent_window!=1024 or $m.block_size!=32 then error("unexpected cohort geometry") else . end |
($m.capacity-$m.recent_window) as $old |
($layers*2*128) as $elements |
{recent_fp16_bytes:($elements*$m.recent_window*4),old_k_fp16_bytes:($elements*$m.capacity*2),
 q4_old_v_payload_bytes:($elements*$m.capacity/2),q4_old_v_scale_bytes:($layers*2*$m.capacity*4),
 summary_index_bytes:($layers*(($old/$m.block_size|ceil)*(2*128*12+4)+4))} as $components |
($components|[.[]]|add) as $mixed | ($elements*$m.capacity*4) as $dense |
if any($runs[]; . as $r | any($components|to_entries[]; $r[.key]!=.value) or
    .actual_allocated_bytes!=$mixed or .dense_equivalent_bytes!=$dense) or
    any($rows[]; .cache.allocated_bytes!=(if .arm=="dense" then $dense else $mixed end))
then error("constructor or generation disagrees with reconstructed storage") else . end |
($m.recent_window*$layers*(2*128*2+2*128/2+2*4)) as $trim |
{layers:$layers,components:$components,dense_payload:$dense,mixed_reported:$mixed,
 excess_over_reported_dense:($mixed-$dense),dense_unreported_seq_map:($layers*4),
 capacity_unused_by_streaming_eviction:$m.recent_window,old_sidecar_tail_bytes:$trim,
 measured_rust_tracked_bytes:$tracked[0],tail_fraction_of_tracked:($trim/$tracked[0]),
 ideal_mixed_requested_bytes_after_tail_elimination:($mixed-$trim)}
