#!/usr/bin/env bash
# Shared by the A5 and D3 protocols. Caller holds the exclusive M40 lock.
set -euo pipefail
serving_study=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
consolidation=$(realpath "$serving_study/../../..")
workspace=$(realpath "$consolidation/..")
manifest="$consolidation/docs/experiments/2026-11-21-protocol/performance-contexts.json"
client="$consolidation/target/release/examples/research_http_load"
stats="$consolidation/target/release/examples/research_http_pairs"
phase4_bin="$consolidation/target/release/m40-llm"
main_bin="$workspace/main-reference/target/release/m40-llm"
server_pid=""
monitor_pids=()
port=18514
url="http://127.0.0.1:$port"
gpu=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
clear_flags() {
    local key
    for key in $(compgen -e | rg '^M40LLM_' || true); do unset "$key"; done
    export M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 CUDA_VISIBLE_DEVICES="$gpu"
}
gpu_snapshot() {
    date --iso-8601=ns
    nvidia-smi -i "$gpu" --query-gpu=uuid,name,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv
    nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
}
check_gpu_owners() {
    local owner active
    active=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)
    while read -r owner; do
        [[ -z "$owner" || "$owner" == "$server_pid" ]] || { echo "Unexpected GPU process $owner" >&2; return 1; }
    done <<< "$active"
}
stop_server() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
        server_pid=""
    fi
}
stop_study() {
    stop_server
    for pid in "${monitor_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
}
start_study() {
    out=$(realpath -m "$1")
    model=$(realpath "$2")
    local expected=$3
    mkdir "$out"
    clear_flags
    [[ -z $(git -C "$consolidation" status --porcelain) ]]
    [[ -z $(git -C "$workspace/main-reference" status --porcelain) ]]
    check_gpu_owners
    if curl -fsS --max-time 1 "$url/health" >/dev/null 2>&1; then echo "Port already in use" >&2; return 1; fi
    {
        date --iso-8601=ns
        git -C "$consolidation" rev-parse HEAD
        git -C "$workspace/main-reference" rev-parse HEAD
        printf '%s  %s\n' "$expected" "$model" \
          a6462eb0b64326c47eb465a9bed257700794438f0a32b01296f2f454a6554036 "$phase4_bin" \
          68c13529ab6182de91810181964d5b74b6ed93636f0a149e6bf7347a6fa4519c "$main_bin" | sha256sum -c -
        sha256sum "$model" "$phase4_bin" "$main_bin" "$client" "$stats" "$manifest"
        cuobjdump -lelf "$phase4_bin"
        cuobjdump -lelf "$main_bin"
        nvcc --version
        printf 'process_clock_ticks_per_second=%s\n' "$(getconf CLK_TCK)"
        gpu_snapshot
    } > "$out/preflight.txt"
    trap stop_study EXIT
    bash "$consolidation/docs/experiments/2026-11-21-protocol/host-load.sh" "$$" > "$out/host-load.log" &
    monitor_pids+=("$!")
    nvidia-smi -i "$gpu" --query-gpu=timestamp,uuid,memory.used,temperature.gpu,clocks.sm,utilization.gpu --format=csv --loop-ms=250 > "$out/gpu.csv" &
    monitor_pids+=("$!")
    nvidia-smi -i "$gpu" --query-compute-apps=timestamp,pid,used_gpu_memory --format=csv --loop-ms=250 > "$out/processes.csv" &
    monitor_pids+=("$!")
}
start_server() {
    local binary=$1 label=$2 attempt
    shift 2
    [[ -z "$server_pid" ]]
    check_gpu_owners
    {
        date --iso-8601=ns
        printf '%q ' "$binary" run "$model" --addr "127.0.0.1:$port" --require-sm52 "$@"
        echo
        env | rg '^(M40LLM_|CUDA_)' |sort
        gpu_snapshot
    } > "$out/$label.start"
    "$binary" run "$model" --addr "127.0.0.1:$port" --require-sm52 "$@" > "$out/$label.log" 2>&1 &
    server_pid=$!
    echo "$server_pid" > "$out/$label.pid"
    for ((attempt=0;attempt<480;attempt++)); do
        if ! kill -0 "$server_pid"; then tail -12 "$out/$label.log" >&2; return 1; fi
        if curl -fsS --max-time 1 "$url/health" >/dev/null 2>&1; then
            gpu_snapshot > "$out/$label.ready"
            return
        fi
        sleep 0.25
    done
    echo "Health timeout for $label" >&2
    return 1
}
call_clients() {
    local label=$1 offsets=$2
    shift 2
    check_gpu_owners
    gpu_snapshot > "$out/$label.before"
    cat "/proc/$server_pid/stat" > "$out/$label.cpu-before"
    "$client" --server "$url" --requests "$@" --offsets-ms "$offsets" --output "$out/$label.json"
    cat "/proc/$server_pid/stat" > "$out/$label.cpu-after"
    gpu_snapshot > "$out/$label.after"
    check_gpu_owners
}
