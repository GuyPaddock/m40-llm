#!/usr/bin/env bash
# Run from graph-node-cost under the serial M40 lock.
set -euo pipefail
out=$(realpath -m "${1:?new output directory}")
mkdir "$out"
study=$(realpath docs/experiments/2026-11-21-graph-node-cost)
bin=target/release/examples/graph_node_cost
for key in $(compgen -e | rg '^M40LLM_' || true); do unset "$key"; done
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
export M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 M40LLM_RESEARCH_GRAPH_SETUP=1
export M40LLM_ATTN_HEAD128_KERNEL=native M40LLM_CACHE_EXPERIMENT=off
snapshot() {
    date --iso-8601=ns
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,name,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
    [[ -z $(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits) ]]
}
{
    git rev-parse HEAD
    [[ -z $(git status --porcelain) ]]
    sha256sum -c "$study/binary.sha256"
    cuobjdump -lelf "$bin"
    nvcc --version
    env | rg '^(M40LLM_|CUDA_)' | sort
    snapshot
} > "$out/preflight.txt"
bash ../consolidation/docs/experiments/2026-11-21-protocol/host-load.sh "$$" > "$out/host-load.log" &
host=$!
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,pstate,temperature.gpu,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv --loop-ms=250 > "$out/gpu.csv" &
device=$!
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=timestamp,pid,used_gpu_memory --format=csv --loop-ms=250 > "$out/process.csv" &
processes=$!
trap 'kill "$host" "$device" "$processes" 2>/dev/null || true; wait "$host" "$device" "$processes" 2>/dev/null || true' EXIT
"$bin" "$out/report.jsonl" > "$out/run.log" 2>&1
snapshot > "$out/after.txt"
python3 "$study/analyze.py" "$out/report.jsonl" "$out/run.log" > "$out/summary.json"
