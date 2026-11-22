#!/usr/bin/env bash
# Invoke with the shared M40 flock held, after capture.sh has completed.
set -euo pipefail
source scripts/dev-env.sh
capture=$(realpath "$1")
out=$(realpath -m "$2")
study=docs/experiments/2026-11-22-sparse-q4-live
model="$HOME/.cache/m40-llm/models/qwen2.5-3b-instruct-q4_0.gguf"
planes=../raw/sparse-q4-inputs-01/planes
exporter=target/capture/release/examples/sparse_q4_rows
bin=target/release/examples/sparse_q4_live
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
export PYTHONDONTWRITEBYTECODE=1
mkdir "$out"
[[ -z $(git status --porcelain) ]]
snapshot() {
    date --iso-8601=ns
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,name,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv
    local owners
    owners=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader)
    echo "$owners"
    [[ -z "$owners" ]]
}
{
    git rev-parse HEAD
    git status --porcelain
    sha256sum "$bin" "$exporter" "$model" "$capture"/{held-out.json,held-out.jsonl,cutoffs.json}
    sha256sum -c "$capture/cutoffs.sha256"
    sha256sum -c docs/experiments/2026-11-21-sparse-q4/planes.sha256
    nvcc --version
    cuobjdump -lelf "$bin"
    env | rg '^(M40LLM_|CUDA_)' | sort
    snapshot
} > "$out/preflight.txt"
"$exporter" "$capture/held-out.json" "$capture/held-out.jsonl" "$capture/cutoffs.json" "$planes" "$out/rows"
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
for layer in 0 17 35; do
    read -r count median upper <<< "$(jq -er --argjson id "$layer" '.layers[] | select(.layer==$id) | [(.rows|length), .cutoffs[1], .cutoffs[2]] | @tsv' "$out/rows/manifest.json")"
    snapshot > "$out/layer-$layer.before"
    "$bin" "$planes" "$out/rows" "$layer" "$count" "$median" "$upper" > "$out/layer-$layer.jsonl" 2> "$out/layer-$layer.stderr"
    snapshot > "$out/layer-$layer.after"
done
python3 "$study/screen.py" "$out/rows/manifest.json" "$out" > "$out/screen.json"
