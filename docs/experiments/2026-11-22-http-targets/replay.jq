# --slurpfile official and --slurpfile portfolio refer to original M40 request logs.
def require($condition; $reason): if $condition then . else error($reason) end;
def median: sort | length as $n | if ($n%2)==1 then .[($n/2|floor)] else (.[($n/2)-1]+.[$n/2])/2 end;
def close($a;$b): (($a-$b)|fabs)<1e-7;
def rate($ms): 512000/$ms;
def gaps($rate;$targets): [$targets[] | . as $target |
    {target_tok_s:$target,required_throughput_gain:($target/$rate-1),
     required_request_saving_ms:(512000/$rate-512000/$target)}];

($official|map(select(.kind=="manifest"))|first) as $m |
($portfolio|map(select(.kind=="joint_only_manifest"))|first) as $p |
require($m.revision=="63240c52b64e06870922c8b6c0cbae8b0fe2e6f4"
    and $p.revision=="1b1890ad7f2dac99b10536dccdf2a01ae5d6d623"
    and $m.model_sha256==$p.model_sha256
    and $m.model_sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
    and $m.gpu_uuid=="GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852"
    and $m.ollama_version.version=="0.40.11" and $p.reference_id=="compound-runtime-63240c5-v1"
    and $m.prompt_token_ids_sha256=="395153b63b714fd6b481f4cc148988649d1220f4c788915ded1e6c3caf3c6ad2"; "wrong historical configuration") |
($official|map(select(.kind=="trial" and .phase=="screen"))) as $o |
require(($o|map([.pair,.runtime]))==[range(0;5) as $i |
    (if $i%2==0 then ["ollama","m40-llm"] else ["m40-llm","ollama"] end)[] | [$i,.]]
    and all($o[]; (.metrics // .) as $r | $r.prompt_tokens==65 and $r.generated_tokens==512 and $r.stop_reason=="length"
        and .is_fixed_workload and .cache.reused_prompt_tokens==0 and .cache.evaluated_prompt_tokens==65
        and close(.e2e_tps;rate(.client_elapsed_ms))); "invalid October 27 pair, cache or denominator") |
($o|map(select(.runtime=="m40-llm")|rate(.client_elapsed_ms))|median) as $compound |
($o|map(select(.runtime=="ollama")|rate(.client_elapsed_ms))|median) as $ollama |
($portfolio|map(select(.kind=="portfolio_http_trial" and .candidate=="all" and .phase=="confirmation"))) as $rows |
require(($rows|map([.pair,.arm]))==[range(0;20) as $i | (if $i%2==0 then [0,1] else [1,0] end)[]|[$i,.]]
    and all($rows[].row; .metrics.generated_tokens==512 and .metrics.prompt_tokens==65 and .metrics.stop_reason=="length"
        and close(.http_tok_per_sec;rate(.http_ms))) and ($rows|map(.row.output)|unique|length)==1; "invalid October 29 confirmation") |
($rows|map(select(.arm==0)|rate(.row.http_ms))|median) as $control |
($rows|map(select(.arm==1)|rate(.row.http_ms))|median) as $candidate |
[53.06,1.30*$ollama] as $targets |
{model_sha256:$m.model_sha256, prompt_tokens:65,generated_tokens:512,
 october_27:{pairs:5,compound_tok_s:$compound,official_tok_s:$ollama,relative_gain:($compound/$ollama-1),
     output_hashes:($o|group_by(.runtime)|map({runtime:.[0].runtime,sha256:(map(.output_sha256)|unique)})),targets:gaps($compound;$targets)},
 october_29:{confirmation_pairs:20,control_tok_s:$control,portfolio_tok_s:$candidate,
     same_cohort_gain:($candidate/$control-1),relative_to_historical_official:($candidate/$ollama-1),targets:gaps($candidate;$targets)}}
