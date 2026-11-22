#!/usr/bin/env bash
# Run from the clean, built quality worktree with the shared M40 lock held.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "usage: bash run-quality.sh MODEL NEW_RAW_DIR MODEL_QUALITY" >&2
    exit 2
fi
source scripts/dev-env.sh
model=$(realpath "$1")
out=$(realpath -m "$2")
bin=$(realpath "$3")
study=docs/experiments/2026-11-22-sparse-q4-live
cutoffs=$study/cutoffs.json
[[ -z $(git status --porcelain) && -x "$bin" ]]
mkdir "$out"
for key in $(compgen -e | rg '^M40LLM_' || true); do
    if [[ "$key" != M40LLM_SYSROOT ]]; then unset "$key"; fi
done
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
export M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1
export M40LLM_PROJECTION_BACKEND=large-model M40LLM_MATERIALIZE_F32_WEIGHTS=0
export M40LLM_DECODE_GRAPH=0 M40LLM_Q8_EXECUTION_WEIGHTS=off M40LLM_Q8_DECODE_STREAM=decode
export M40LLM_PREFILL_CHUNK_SIZE=0 M40LLM_CUDA_GREEDY_ARGMAX=1
snapshot() {
    date --iso-8601=ns
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv
    local owners
    owners=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader)
    [[ -z "$owners" ]]
}
{
    git rev-parse HEAD
    git status --porcelain
    sha256sum "$bin" "$model" "$cutoffs" data/research_quality/v1.json
    nvcc --version
    cuobjdump -lelf "$bin"
    env | rg '^(M40LLM_|CUDA_)' | sort
    snapshot
} > "$out/preflight.txt"
rg -q 'sm_52.cubin' "$out/preflight.txt"
bash ../consolidation/docs/experiments/2026-11-21-protocol/host-load.sh "$$" > "$out/host-load.log" &
host=$!
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=timestamp,pid,used_gpu_memory --format=csv --loop-ms=500 > "$out/processes.csv" &
monitor=$!
trap 'kill "$host" "$monitor" 2>/dev/null || true; wait "$host" "$monitor" 2>/dev/null || true' EXIT
for layout in packed zigzag; do
    for mode in full thresholded; do
        label="$layout-$mode"
        args=(--model "$model" --reference-model "$model" --variant-id "$label" --split held-out
              --sparse-q4-layout "$layout" --sparse-q4-cutoffs "$cutoffs" --output "$out/$label.jsonl")
        if [[ "$mode" == thresholded ]]; then args+=(--sparse-q4-thresholded); fi
        snapshot > "$out/$label.before"
        "$bin" "${args[@]}" > "$out/$label.log" 2>&1 &
        pid=$!
        printf '%s\n' "$pid" > "$out/$label.pid"
        wait "$pid"
        jq -es 'last | .kind == "sparse_q4_quality_complete" and .device_bytes_after == 0' "$out/$label.jsonl"
        snapshot > "$out/$label.after"
    done
done
snapshot > "$out/complete.txt"
