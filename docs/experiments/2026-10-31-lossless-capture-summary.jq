def totals:
  {streams:length,original_bytes:map(.stats.original_bytes)|add,
   encoded_bytes:map(.stats.encoded_bytes)|add,
   metadata_bytes:map(.stats.metadata_bytes)|add,
   raw_blocks:map(.stats.raw_blocks)|add,
   packed_blocks_by_width:[range(0;5) as $w|map(.stats.packed_blocks_by_width[$w])|add],
   exceptional_words:map(.exceptional_words)|add,
   net_saved_bytes:map(.net_saved_bytes)|add} |
  . + {net_saved_fraction:(.net_saved_bytes/.original_bytes)};
if (map(select(.kind=="lossless_kv_capture_complete" and .is_complete))|length)!=1
then error("incomplete capture") else . end |
map(select(.kind=="lossless_kv_compression")) as $rows |
if ($rows|length)!=384 or ($rows|any(.has_bit_exact_reconstruction!=true))
then error("incomplete or failed reconstruction grid") else . end |
{
 manifest:map(select(.kind=="lossless_kv_capture_manifest"))[0],
 counts:(group_by(.kind)|map({key:.[0].kind,value:length})|from_entries),
 footprints:[([1024,2048,4096,8192][]) as $context|([1,2,4,8][]) as $batch|
   {context_tokens:$context,batch:$batch,totals:($rows|map(select(.context_tokens==$context and .case.sequence<$batch))|totals)}],
 by_role:($rows|map(select(.context_tokens==8192))|group_by(.case.role)|map({role:.[0].case.role,totals:totals})),
 by_layer:($rows|map(select(.context_tokens==8192))|group_by(.case.layer)|map({layer:.[0].case.layer,totals:totals})),
 streams:map(select(.kind=="lossless_kv_stream")),
 measurements:$rows,
 sequences:map(select(.kind=="lossless_kv_sequence")),
 complete:map(select(.kind=="lossless_kv_capture_complete"))[0]
}
