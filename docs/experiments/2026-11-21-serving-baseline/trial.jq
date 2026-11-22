.records[0] as $r | $trace[0].records[0].response as $ref |
$prompt[0] as $p | $ids[0] as $seq | ($seq|length) as $n |
(if ($p.stop_ids | index($seq[-1])) == null then "length" else "stop" end) as $reason |
{prompt_tokens:($p.ids|length),generated_tokens:$n,stop_reason:$reason} as $counts |
($r.response.metrics // $counts) as $m |
if .success != true or .errors != [] or (.records|length)!=1 or $r.status!=200 or
   $n==0 or $r.response.output!=$ref.output or ($r.response.output|type)!="string" or
   $m.prompt_tokens!=$counts.prompt_tokens or $m.generated_tokens!=$n or $m.stop_reason!=$reason or
   $m.token_elapsed_ms!=null or $r.wall_ms<=0 or (($r.done_ms-$r.submitted_ms-$r.wall_ms)|fabs)>1e-6
then error("invalid timed response: " + $label) else . end |
if $r.response.metrics and ($m.prefill_elapsed_ms<=0 or $m.decode_elapsed_ms<0 or
   ($n>1 and $m.decode_elapsed_ms==0) or $m.total_elapsed_ms+1e-6<($m.prefill_elapsed_ms+$m.decode_elapsed_ms))
then error("invalid generation timing: " + $label) else . end |
{label:$label,arm:$arm,http_seconds:($r.wall_ms/1000),metrics:$m,
 metric_source:(if $r.response.metrics then "server" else "companion IDs" end),
 http_tokens_s:(1000*$n/$r.wall_ms),
 prompt_tokens_s:(if $r.response.metrics then 1000*$p.length/$m.prefill_elapsed_ms else null end),
 decode_tokens_s:(if $m.decode_elapsed_ms>0 then 1000*($n-1)/$m.decode_elapsed_ms else null end),
 server_cpu_s:$cpu,process_peak_mib:$peak,launch_to_health_s:($launch_ns/1e9)}
