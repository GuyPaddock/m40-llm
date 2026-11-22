#!/usr/bin/env bash
# run.sh qwen-{short,medium,long,q4-short}|tiny-{default,dense} NEW_OUTPUT_DIR
set -euo pipefail
source "$(dirname "$0")/server.sh"
cohort=$1
tiny_hash=76bd2d476450b35a43d294470b091e21d81624a4f3d70d2542c07bd8fbefad93
qwen_hash=4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d
ids_args=()
if [[ "$cohort" == tiny-default || "$cohort" == tiny-dense ]]; then
    start_study "$2" "$HOME/.cache/m40-llm/models/TinyLlama-1.1B-Chat-v1.0.f16.gguf" "$tiny_hash"
    admission="$workspace/main-baseline/docs/experiments/2026-11-21-main-baseline/models.jsonl.gz"
    gzip -dc "$admission" | jq -cs --arg h "$tiny_hash" 'map(select(.sha256==$h and .mapped_all and .mapped_head))|if length==1 then .[0] else error("TinyLlama admission") end' > "$out/model.json"
    jq '.inputs[0] | {prompt,prompt_format:"raw",seed:0,top_k:1,max_tokens:128,stream:false,include_metrics:true}' "$out/model.json" > "$out/request.json"
    prompt_count=$(jq '.inputs[0].length' "$out/model.json")
    stop_ids=$(jq -c '.inputs[0].stop_ids' "$out/model.json")
    context=$(jq .context "$out/model.json")
    layers=22
else
    shape=${cohort#qwen-}
    case "$shape" in
        short|medium|long) start_study "$2" "$HOME/.ollama/models/blobs/sha256-$qwen_hash" "$qwen_hash" ;;
        q4-short)
            shape=short
            ids_args+=(--check-layers)
            start_study "$2" "$HOME/.cache/m40-llm/models/qwen2.5-3b-instruct-q4_0.gguf" 670d82d0fbee6289b661eb323612c65bd2e97ec8c65cdbd9131aa44d30e9816a
            ;;
        *) exit 2 ;;
    esac
    jq -e --arg shape "$shape" '.cases[]|select(.name==$shape)' "$manifest" > "$out/case.json"
    if [[ "$cohort" == qwen-q4-short ]]; then
        jq '.request | .max_tokens=32' "$out/case.json" > "$out/request.json"
    else
        jq '.request' "$out/case.json" > "$out/request.json"
    fi
    prompt_count=$(jq .prompt_token_count "$out/case.json")
    context=$(jq .context "$out/case.json")
    stop_ids=$(jq -sc 'map(select(.kind=="manifest"))[0].m40_stop_token_ids' "$consolidation/docs/experiments/2026-11-21-native-split/report.jsonl")
    layers=36
fi
ids_tool="$consolidation/target/release/examples/research_serving_ids"
"$ids_tool" "${ids_args[@]}" "$model" "$out/request.json" > "$out/phase4-prompt-ids.json"
sha256sum "$ids_tool" "$out/request.json" >> "$out/preflight.txt"
if [[ "$cohort" == tiny-* ]]; then
    jq -e --slurpfile phase4 "$out/phase4-prompt-ids.json" '.inputs[0].ids == $phase4[0].ids and .inputs[0].stop_ids == $phase4[0].stop_ids' "$out/model.json"
else
    jq -e --slurpfile phase4 "$out/phase4-prompt-ids.json" '.prompt_ids == $phase4[0].ids' "$out/case.json"
fi
if [[ "$cohort" == qwen-q4-short ]]; then
    jq -e '.mapping | .layers==36 and .width==2048 and .head.width==.width and .head.name=="output.weight" and .head.dtype=="Q6K" and .head.tied==false' "$out/phase4-prompt-ids.json"
    stop_ids=$(jq -c '.stop_ids' "$out/phase4-prompt-ids.json")
fi
jq '.stream=true|del(.include_metrics)' "$out/request.json" > "$out/stream-request.json"
configure() {
    local arm=$1
    binary="$phase4_bin"
    binary_args=(--max-context-tokens "$context")
    if [[ "$cohort" == tiny-* ]]; then
        if [[ "$arm" == 0 ]]; then binary="$main_bin"; binary_args=(); fi
        if [[ "$cohort" == tiny-dense ]]; then binary_args+=(--kv-compress-mode off); fi
    elif [[ "$arm" == 1 ]]; then
        binary_args+=(--kv-compress-mode off)
    fi
}
# Untimed separate CLI ID trace and streamed HTTP timing companion for each arm.
for arm in 0 1; do
    configure "$arm"
    clear_flags
    export M40LLM_DECODE_LOG=1 M40LLM_RESEARCH_TOKEN_TIMINGS=1
    prompt=$(jq -r '.prompt' "$out/request.json"; printf x)
    prompt=${prompt%x}
    prompt=${prompt%$'\n'}
    maximum=$(jq .max_tokens "$out/request.json")
    check_gpu_owners
    gpu_snapshot > "$out/cli-$arm.before"
    /usr/bin/time -f 'wall_s=%e user_s=%U sys_s=%S peak_rss_kib=%M' -o "$out/cli-$arm.time" \
        "$binary" generate "$model" "$prompt" --prompt-format raw --max-tokens "$maximum" --top-k 1 --seed 0 --require-sm52 "${binary_args[@]}" > "$out/cli-$arm.stdout" 2> "$out/cli-$arm.stderr"
    gpu_snapshot > "$out/cli-$arm.after"
    start_server "$binary" "trace-$arm" "${binary_args[@]}"
    call_clients "calibrate-$arm" 0 "$out/request.json"
    rg -q "mapped $layers standard layers" "$out/trace-$arm.log"
    rg -q "(full-layer forward|packed prefix prefill) enabled layers=$layers" "$out/trace-$arm.log"
    rg 'sampled token id=[0-9]+' "$out/trace-$arm.log" | sed -E 's/.*sampled token id=([0-9]+).*/\1/' | jq -s '.' > "$out/ids-$arm.json"
    rg 'sampled token id=[0-9]+' "$out/cli-$arm.stderr" | sed -E 's/.*sampled token id=([0-9]+).*/\1/' | jq -s '.' > "$out/cli-ids-$arm.json"
    cmp "$out/ids-$arm.json" "$out/cli-ids-$arm.json"
    jq -e --argjson cap "$maximum" 'length>0 and length<=$cap' "$out/ids-$arm.json" >/dev/null
    stop_server
    clear_flags
    start_server "$binary" "stream-$arm" "${binary_args[@]}"
    call_clients "stream-warm-$arm" 0 "$out/stream-request.json"
    call_clients "stream-measured-$arm" 0 "$out/stream-request.json"
    stop_server
done
signature_equal=true
if ! cmp -s "$out/ids-0.json" "$out/ids-1.json"; then signature_equal=false; fi
# One server resident at a time. Both arms get a full warmup per fresh process.
series() {
    local phase=$1 begin=$2 end=$3 trial arm label
    for ((trial=begin;trial<end;trial++)); do
        for arm in "$((trial%2))" "$((1-trial%2))"; do
            label="$phase-$trial-$arm"
            configure "$arm"
            clear_flags
            start_server "$binary" "server-$label" "${binary_args[@]}"
            call_clients "warm-$label" 0 "$out/request.json"
            call_clients "$label" 0 "$out/request.json"
            stop_server
            jq -ce --arg label "$label" --argjson arm "$arm" --argjson prompt "$prompt_count" --argjson stop "$stop_ids" \
                --slurpfile ids "$out/ids-$arm.json" --slurpfile ref "$out/calibrate-$arm.json" \
                '.records[0] as $r | $ids[0] as $seq | ($seq|length) as $count |
                if $r.response.output != $ref[0].records[0].response.output then error("timed output differs from token trace") else . end |
                (if ($stop|index($seq[-1]))==null then "length" else "stop" end) as $reason |
                ($r.response.metrics // {prompt_tokens:$prompt,generated_tokens:$count,stop_reason:$reason}) as $m |
                if $m.prompt_tokens!=$prompt or $m.generated_tokens!=$count or $m.stop_reason!=$reason then error("token accounting differs") else
                {label:$label,arm:$arm,http_seconds:($r.wall_ms/1000),metrics:$m,metric_source:(if $r.response.metrics then "server" else "companion IDs" end)} end' \
                "$out/$label.json" >> "$out/timings.jsonl"
        done
    done
}
series pair 0 5
if [[ "$signature_equal" == true ]]; then
    "$stats" "$out/timings.jsonl" > "$out/screen.json"
    decision=$(jq -r .next "$out/screen.json")
    if [[ "$decision" == extend ]]; then series pair 5 10; "$stats" "$out/timings.jsonl" > "$out/extended.json"; decision=$(jq -r .next "$out/extended.json"); fi
    if [[ "$decision" == confirm ]]; then series confirm 0 20; "$stats" --confirmation "$out/timings.jsonl" > "$out/confirmation.json"; fi
else
    jq -n '{equivalent_work:false,reason:"different greedy IDs; independent arm measurements only"}' > "$out/screen.json"
fi
gpu_snapshot > "$out/complete.txt"
