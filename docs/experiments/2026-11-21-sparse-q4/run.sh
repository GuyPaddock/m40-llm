#!/usr/bin/env bash
# From this worktree with the serial M40 benchmark lock held.
set -euo pipefail
out=$(realpath -m "$1")
study=docs/experiments/2026-11-21-sparse-q4
inputs=../raw/sparse-q4-inputs-01
bin=target/release/examples/sparse_q4_probe
stats=target/release/examples/sparse_q4_screen
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
mkdir "$out"
[[ -z $(git status --porcelain) ]]
gpu() { nvidia-smi -i "$CUDA_VISIBLE_DEVICES" "$@"; }
snapshot() {
    date --iso-8601=ns
    gpu --query-gpu=uuid,name,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv
    gpu --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
}
idle() {
    local owners
    owners=$(gpu --query-compute-apps=pid --format=csv,noheader,nounits)
    if [[ -n "$owners" ]]; then echo "GPU busy: $owners" >&2; return 1; fi
}
{
    git rev-parse HEAD
    git status --porcelain
    sha256sum "$bin" "$stats" "$study/build.sh"
    sha256sum -c "$study/input.sha256"
    sha256sum -c "$study/planes.sha256"
    nvcc --version
    cuobjdump -lelf "$bin"
    env | rg '^(M40LLM_|CUDA_)' | sort
    snapshot
} > "$out/preflight.txt"
idle
pids=()
cleanup() {
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
bash ../consolidation/docs/experiments/2026-11-21-protocol/host-load.sh "$$" > "$out/host-load.log" &
pids+=("$!")
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,uuid,memory.used,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu --format=csv --loop-ms=500 > "$out/gpu.csv" &
pids+=("$!")
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=timestamp,pid,used_gpu_memory --format=csv --loop-ms=500 > "$out/processes.csv" &
pids+=("$!")
run() {
    local label=$1 rounds=$2 start=$3
    idle
    snapshot > "$out/$label.before"
    "$bin" "$inputs/planes" "$rounds" "$start" > "$out/$label.jsonl" 2> "$out/$label.stderr"
    snapshot > "$out/$label.after"
    idle
}
run primary 5 0
"$stats" "$out/primary.jsonl" > "$out/screen.json"
if [[ $(jq -r .next "$out/screen.json") == extend ]]; then
    run extension 5 5
    "$stats" "$out/primary.jsonl" "$out/extension.jsonl" > "$out/extended.json"
fi
decision="$out/screen.json"
if [[ -f "$out/extended.json" ]]; then decision="$out/extended.json"; fi
if [[ $(jq -r .next "$decision") == confirm ]]; then
    run confirmation 20 0
    "$stats" --confirmation "$out/confirmation.jsonl" > "$out/confirmed.json"
fi
