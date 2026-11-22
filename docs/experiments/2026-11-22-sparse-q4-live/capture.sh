#!/usr/bin/env bash
# Run from this worktree with the shared M40 lock held.
set -euo pipefail
out=$(realpath -m "$1")
model="$HOME/.cache/m40-llm/models/qwen2.5-3b-instruct-q4_0.gguf"
bin=target/capture/release/examples/model_quality
fit=target/capture/release/examples/sparse_q4_calibrate
mkdir "$out"
[[ -z $(git status --porcelain) ]]
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
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,name,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv
    local owners
    owners=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader)
    echo "$owners"
    [[ -z "$owners" ]]
}
{
    git rev-parse HEAD
    git status --porcelain
    sha256sum "$bin" "$fit" "$model" data/research_quality/v1.json
    nvcc --version
    cuobjdump -lelf "$bin"
    env | rg '^(M40LLM_|CUDA_)' | sort
    snapshot
} > "$out/preflight.txt"
bash ../consolidation/docs/experiments/2026-11-21-protocol/host-load.sh "$$" > "$out/host-load.log" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true' EXIT
capture() {
    local split=$1
    snapshot > "$out/$split.before"
    "$bin" --model "$model" --reference-model "$model" --variant-id sparse-q4-native \
        --split "$split" --sparse-q4-capture --capture-ffn-inputs "$out/$split.json" \
        --output "$out/$split.jsonl" --context-tokens 1024 > "$out/$split.log" 2>&1
    snapshot > "$out/$split.after"
}
capture calibration
"$fit" "$out/calibration.json" "$out/calibration.jsonl" "$out/cutoffs.json"
sha256sum "$out/cutoffs.json" > "$out/cutoffs.sha256"
capture held-out
sha256sum -c "$out/cutoffs.sha256"
