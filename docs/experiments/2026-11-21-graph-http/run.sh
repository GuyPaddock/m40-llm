#!/usr/bin/env bash
# Run from the graph worktree. Arguments: model, cohort qwen|tiny, new output dir.
set -euo pipefail
model=$(realpath "$1")
cohort=$2
out=$(realpath -m "$3")
mkdir "$out"
bin=$(realpath target/cublas-1/release/m40-llm)
stats=$(realpath ../consolidation/target/release/examples/research_http_pairs)
consolidation=$(realpath ../consolidation/docs/experiments)
manifest="$consolidation/2026-11-21-protocol/performance-contexts.json"
ports=(18461 18462)
servers=()
monitors=()
cleanup() {
    for pid in "${servers[@]}" "${monitors[@]}"; do
        if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
    done
}
trap cleanup EXIT
[[ -z $(git status --porcelain) ]]
case "$cohort" in
  qwen)
    expected=4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d
    jq '.cases[] | select(.name=="short") | .request' "$manifest" > "$out/request.json"
    ;;
  tiny|tiny-decode-stream)
    expected=76bd2d476450b35a43d294470b091e21d81624a4f3d70d2542c07bd8fbefad93
    jq -n --arg prompt $'<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nExplain how a web browser loads a page, from typing the URL to displaying the result.</s>\n<|assistant|>\n' \
      '{prompt:$prompt,prompt_format:"raw",max_tokens:128,seed:0,top_k:1,stream:false,include_metrics:true}' > "$out/request.json"
    ;;
  *) exit 2 ;;
esac
for name in $(compgen -e | rg '^M40LLM_' || true); do
    if [[ "$name" != M40LLM_SYSROOT ]]; then unset "$name"; fi
done
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
export M40LLM_ENABLE_CUBLAS=1 M40LLM_ENABLE_NVCC=1 M40LLM_ATTN_HEAD128_KERNEL=native
export M40LLM_PROJECTION_BACKEND=large-model M40LLM_MATERIALIZE_F32_WEIGHTS=0
export M40LLM_Q8_EXECUTION_WEIGHTS=off M40LLM_Q8_FFN_KERNEL=native M40LLM_Q8_PREFILL_BACKEND=native
export M40LLM_Q8_DECODE_STREAM=decode M40LLM_Q8_DECODE_COL_TILE=4 M40LLM_Q8_DECODE_SPLIT_QBLOCK=4
export M40LLM_FUSED_Q8_MLP_SWIGLU=1 M40LLM_FUSED_Q8_QKV=1 M40LLM_Q8_MLP_GATE_UP_SPLIT_QBLOCK=4
export M40LLM_Q8_LM_HEAD_ARGMAX=1 M40LLM_Q8_LM_HEAD_ARGMAX_SHARED_A=1 M40LLM_CUDA_GREEDY_ARGMAX=1
export M40LLM_PREFILL_CHUNK_SIZE=128
if [[ "$cohort" == tiny-decode-stream ]]; then
    export M40LLM_F16_DECODE_KERNEL=1 M40LLM_F16_DECODE_STREAM=decode
fi
gpu() {
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" "$@"
}
owners() {
    gpu --query-compute-apps=pid --format=csv,noheader,nounits
}
snapshot() {
    date --iso-8601=ns
    gpu --query-gpu=uuid,name,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv
    gpu --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
}
assert_owners() {
    local pid
    local active
    active=$(owners)
    while read -r pid; do
        if [[ -n "$pid" && " ${servers[*]} " != *" $pid "* ]]; then
            echo "Unrelated GPU process $pid" >&2
            return 1
        fi
    done <<< "$active"
}
assert_owners
digest=$(sha256sum "$model")
[[ ${digest%% *} == "$expected" ]]
{
    date --iso-8601=ns
    git rev-parse HEAD
    git status --porcelain
    echo "$digest"
    sha256sum "$bin" "$stats" "$manifest" "$out/request.json"
    cuobjdump -lelf "$bin"
    nvcc --version
    printf 'process_clock_ticks_per_second=%s\n' "$(getconf CLK_TCK)"
    env | rg '^(CUDA_|M40LLM_)' | sort
    snapshot
} > "$out/preflight.txt"
bash "$consolidation/2026-11-21-protocol/host-load.sh" "$$" > "$out/host-load.log" &
monitors+=("$!")
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,uuid,memory.used,temperature.gpu,clocks.sm,utilization.gpu --format=csv --loop-ms=500 > "$out/gpu.csv" &
monitors+=("$!")
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=timestamp,pid,used_gpu_memory --format=csv --loop-ms=500 > "$out/processes.csv" &
monitors+=("$!")
start_server() {
    local graph=$1 diag=$2 label=$3 log_ids=${4:-0} port=${ports[$1]}
    if curl -fsS --max-time 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
        echo "Port $port already in use" >&2; return 1
    fi
    M40LLM_DECODE_LOG=$log_ids M40LLM_DECODE_GRAPH=$graph M40LLM_RESEARCH_GRAPH_SETUP=$diag M40LLM_DECODE_GRAPH_DIAG_SYNC=$diag \
      "$bin" run "$model" --addr "127.0.0.1:$port" --require-sm52 --kv-compress-mode off --max-context-tokens 1024 > "$out/$label.log" 2>&1 &
    servers[$graph]=$!
    for ((attempt=0; attempt<240; attempt++)); do
        kill -0 "${servers[$graph]}" || return 1
        if curl -fsS --max-time 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then return 0; fi
        sleep 1
    done
    return 1
}
stop_server() {
    local graph=$1
    kill "${servers[$graph]}"
    wait "${servers[$graph]}" || true
    servers[$graph]=""
}
request() {
    local graph=$1 label=$2 body=$3 port=${ports[$1]}
    assert_owners
    snapshot > "$out/$label.before"
    local wall
    cat "/proc/${servers[$graph]}/stat" > "$out/$label.cpu-before"
    wall=$(curl -fsS --max-time 900 -H 'Content-Type: application/json' --data-binary "@$body" \
           --output "$out/$label.json" --write-out '%{time_total}' "http://127.0.0.1:$port/generate")
    cat "/proc/${servers[$graph]}/stat" > "$out/$label.cpu-after"
    jq -ce --arg wall "$wall" --arg label "$label" --argjson graph "$graph" \
      'select(.metrics.generated_tokens>0) | {label:$label,arm:$graph,http_seconds:($wall|tonumber),metrics}' \
      "$out/$label.json" >> "$out/timings.jsonl"
    snapshot > "$out/$label.after"
}
signature='[.output,.metrics.prompt_tokens,.metrics.generated_tokens,.metrics.stop_reason]'
for graph in 0 1; do
    start_server "$graph" 0 "trace-$graph" 1
    request "$graph" "trace-response-$graph" "$out/request.json"
    rg 'sampled token id=[0-9]+' "$out/trace-$graph.log" | sed -E 's/.*sampled token id=([0-9]+).*/\1/' | jq -s '.' > "$out/ids-$graph.json"
    jq -e --slurpfile response "$out/trace-response-$graph.json" 'length>0 and length==$response[0].metrics.generated_tokens' "$out/ids-$graph.json" >/dev/null
    rg '^(VmRSS|VmHWM|Threads):' "/proc/${servers[$graph]}/status" > "$out/trace-$graph.host-memory"
    stop_server "$graph"
done
cmp "$out/ids-0.json" "$out/ids-1.json"
rg -q 'captured full-token decode CUDA graph' "$out/trace-1.log"
if rg -q 'disabling full-token decode CUDA graph' "$out/trace-1.log"; then exit 1; fi
start_server 0 0 ordinary
request 0 warm-0 "$out/request.json"
if [[ "$cohort" == qwen ]]; then
    jq -e '.metrics.prompt_tokens == 65 and .metrics.generated_tokens == 512 and .metrics.stop_reason == "length"' "$out/warm-0.json"
fi
start_server 1 0 graph
request 1 warm-1 "$out/request.json"
rg -q 'captured full-token decode CUDA graph' "$out/graph.log"
series() {
    local phase=$1 start=$2 end=$3
    local trial graph
    for ((trial=start; trial<end; trial++)); do
        for graph in "$((trial % 2))" "$((1 - trial % 2))"; do
            request "$graph" "$phase-$trial-$graph" "$out/request.json"
        done
    done
}
series pair 0 5
"$stats" "$out/timings.jsonl" > "$out/screen-5.json"
decision=$(jq -r '.next' "$out/screen-5.json")
if [[ "$decision" == extend ]]; then
    series pair 5 10
    "$stats" "$out/timings.jsonl" > "$out/screen-10.json"
    decision=$(jq -r '.next' "$out/screen-10.json")
fi
if [[ "$decision" == confirm ]]; then
    series confirm 0 20
    "$stats" "$out/timings.jsonl" --confirmation > "$out/confirmation.json"
fi
if rg -i 'disabling full-token decode CUDA graph' "$out/graph.log"; then exit 1; fi
shopt -s nullglob
for file in "$out"/{trace-response,warm,pair,confirm}-*.json; do
    diff -q <(jq -c "$signature" "$file") <(jq -c "$signature" "$out/warm-0.json")
done
for graph in 0 1; do
    stop_server "$graph"
done
start_server 1 1 diagnostic
jq '.max_tokens=32' "$out/request.json" > "$out/diagnostic-request.json"
request 1 diagnostic-response "$out/diagnostic-request.json"
rg -q 'graph-setup.*query_status=0' "$out/diagnostic.log"
rg -q 'graph-replay.*launch_cpu_us' "$out/diagnostic.log"
cleanup
servers=()
monitors=()
snapshot > "$out/after.txt"
