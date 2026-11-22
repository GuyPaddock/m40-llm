def nearest_rank($p): sort | if length==0 then null else .[(length*$p|ceil)-1] end;
$trace[0].records[0] as $r | $stream[0].records[0] as $s |
$r.response.metrics as $m | $m.token_elapsed_ms as $samples |
if any([$trace[0], $stream[0]][]; .success!=true or .errors!=[] or (.records|length)!=1 or .records[0].status!=200)
   or $s.output!=$r.response.output or ($s.arrival_ms|type)!="array"
   or $s.arrival_ms!=($s.arrival_ms|sort) or any($s.arrival_ms[]; .<$s.submitted_ms or .>$s.done_ms)
then error("invalid diagnostic responses") else . end |
if $m and (($samples|type)!="array" or ($samples|length)!=($ids[0]|length) or
   $m.generated_tokens!=($samples|length) or $samples[0]!=$m.prefill_elapsed_ms or
   $samples!=($samples|sort) or (($samples[-1]-$samples[0]-$m.decode_elapsed_ms)|fabs)>1e-6)
then error("invalid sampled token trace") else . end |
[range(1;($samples|length)) as $i | $samples[$i]-$samples[$i-1]] as $gaps |
($cli | capture("^wall_s=(?<wall_s>[0-9.]+) user_s=(?<user_s>[0-9.]+) sys_s=(?<sys_s>[0-9.]+) peak_rss_kib=(?<peak_rss_kib>[0-9]+)\n?$")
 | with_entries(.value |= tonumber)) as $time |
{arm:$arm,cli:$time,diagnostic_http_ms:$r.wall_ms,generation_first_sample_ms:$samples[0],
 token_intervals:($gaps|length),itl_median_ms:($gaps|nearest_rank(0.50)),itl_p95_ms:($gaps|nearest_rank(0.95)),
 stream_http_ms:$s.wall_ms,stream_records:($s.arrival_ms|length),
 stream_first_nonterminal_record_ms:(if ($s.arrival_ms|length)>0 then $s.arrival_ms[0]-$s.submitted_ms else null end)}
