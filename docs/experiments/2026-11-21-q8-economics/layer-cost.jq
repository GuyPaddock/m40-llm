# Input consists only of measured events from one profiled HTTP cohort.
def median:
    sort as $a | ($a|length) as $n |
    if $n==0 then error("median of no samples")
    else ($a[(($n-1)/2|floor)] + $a[($n/2|floor)]) / 2 end;
def nonnegative: type=="number" and isfinite and .>=0;
def layer_stages:
    ["attn_norm","qkv_project","rope_q","kv_append_rope_k","attention","out_project",
     "attn_residual","ffn_norm","mlp_gate_up_swiglu","mlp_down","mlp_residual"] | sort;
def token_stages: ["embedding","output_norm","lm_head","greedy_argmax"] | sort;
def valid_sample:
    .absolute_position==64+.generated_step and
    (.stages|length)==400 and
    ([.stages[] | select(.layer_id!=null) | .layer_id] | unique)==[range(36)] and
    ([.stages[] | select(.layer_id==null) | .stage]|sort)==token_stages and
    ([.stages[] | select(.layer_id!=null)] | group_by(.layer_id) |
        all(.[]; (map(.stage)|sort)==layer_stages)) and
    all(.stages[]; .stream=="decode" and (.gpu_elapsed_ms|nonnegative) and
        (.cpu_call_elapsed_ms|nonnegative) and (.counters.launches|nonnegative)) and
    all(.host_timings[]; (.elapsed_ms|nonnegative));
if length<15 or ([.[].process_id]|unique|length)!=1 or
    (group_by(.session_id)|any(.[]; (map(.generated_step)|sort)!=[0,255,511])) or
    any(.[]; valid_sample|not) then error("incomplete session, layer or stage coverage") else . end |
group_by(.generated_step) | map(
    . as $rows |
    {step:.[0].generated_step, measured_requests:length,
     gpu_interval_union_ms:(map(.gpu_stage_union_ms)|median),
     token_wall_ms:(map(.token_wall_ms)|median),
     host_operation_ms:([$rows[]|.host_timings|group_by(.operation)[]|
        {name:.[0].operation,ms:(map(.elapsed_ms)|add)}] | group_by(.name) |
        map({name:.[0].name,requests_with_operation:length,median_ms:(map(.ms)|median)})),
     stages:([$rows[]|.stages|group_by(.stage)[]|
        {name:.[0].stage,gpu:(map(.gpu_elapsed_ms)|add),
         cpu:(map(.cpu_call_elapsed_ms)|add),launches:(map(.counters.launches)|add)}] |
        group_by(.name) | map({name:.[0].name,gpu_ms:(map(.gpu)|median),
            cpu_call_ms:(map(.cpu)|median),launches:(map(.launches)|median)})),
     layers:([$rows[]|.stages[]|select(.layer_id!=null)]|group_by([.layer_id,.stage])|
        map({layer:.[0].layer_id,stage:.[0].stage,
             gpu_ms:(map(.gpu_elapsed_ms)|median),cpu_call_ms:(map(.cpu_call_elapsed_ms)|median)}))}
)
