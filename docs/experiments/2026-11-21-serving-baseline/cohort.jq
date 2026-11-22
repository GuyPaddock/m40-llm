def median: map(select(.!=null)) | sort | length as $n |
    if $n==0 then null else (.[$n/2|floor]+.[($n-1)/2|floor])/2 end;
def expected($phase;$n):
    [range(0;$n) as $i | ($i%2,1-$i%2) | "\($phase)-\($i)-\(.)"];
def arms($rows): [range(2) as $arm | [$rows[]|select(.arm==$arm)] as $r |
    {arm:$arm,token_accounting:($r[0].metrics|{prompt_tokens,generated_tokens,stop_reason}),
     metric_source:$r[0].metric_source,
     median_http_s:([$r[].http_seconds]|median),median_http_tokens_s:([$r[].http_tokens_s]|median),
     median_admission_to_first_sample_ms:([$r[].metrics.prefill_elapsed_ms]|median),
     median_prompt_tokens_s:([$r[].prompt_tokens_s]|median),median_decode_tokens_s:([$r[].decode_tokens_s]|median),
     median_server_cpu_s:([$r[].server_cpu_s]|median),max_sampled_process_mib:([$r[].process_peak_mib]|max),
     median_launch_to_health_s:([$r[].launch_to_health_s]|median)}];
def phase($rows;$gate):
    arms($rows) as $arms | ($rows|length/2) as $pairs |
    if $gate.equivalent_work!=false and
       ($gate.pairs!=$pairs or
        (($gate.control_tps.median-$arms[0].median_http_tokens_s)|fabs)>1e-8 or
        (($gate.candidate_tps.median-$arms[1].median_http_tokens_s)|fabs)>1e-8)
    then error("paired gate does not match raw throughput")
    else {pairs:$pairs,arms:$arms,gate:$gate} end;
. as $gates | ($a==$b) as $same |
(if $gates[0].next=="extend" then 10 else 5 end) as $n |
(if any($gates[];.next=="confirm") then 20 else 0 end) as $fresh |
if ($gates|length)!=(1+(if $n==10 then 1 else 0 end)+(if $fresh==20 then 1 else 0 end)) or
   (($gates[0].equivalent_work!=false)!=$same) or
   ([$trials[].label]!=(expected("pair";$n)+expected("confirm";$fresh))) or
   ([$trials[]|{label,arm,http_seconds,metrics,metric_source}]!=$ledger) or
   ($diagnostics|map(.arm))!=[0,1]
then error("incomplete phases, ID parity or request ledger disagreement") else . end |
{same_greedy_ids:$same,screen:phase($trials[:10];$gates[0]),
 extended:(if $n==10 then phase($trials[:20];$gates[1]) else null end),
 confirmation:(if $fresh==20 then phase($trials[($n*2):];$gates[-1]) else null end),
 diagnostics:$diagnostics,trials:$trials}
