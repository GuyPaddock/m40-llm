#!/usr/bin/env bash
set -euo pipefail
out=$(realpath -m "$1")
model="$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
inputs=../raw/kv-history-inputs.json
bin=target/hardware/release/examples/kv_quality
mkdir "$out"
for key in $(compgen -e | rg '^M40LLM_' || true); do
    if [[ "$key" != M40LLM_SYSROOT ]]; then unset "$key"; fi
done
while read -r setting; do export "$setting"; done < <(sed -nE 's/^ +\("(M40LLM_[A-Z0-9_]+)", "([^"]+)"\),$/\1=\2/p' examples/kv_quality/config.rs)
export M40LLM_KV_SELECTED_BLOCK_ORDER=chronological
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
snapshot() {
    date --iso-8601=ns
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv
    local owners
    owners=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader)
    echo "$owners"
    [[ -z "$owners" ]]
}
[[ -z $(git status --porcelain) ]]
{
    git rev-parse HEAD
    git status --porcelain
    sha256sum "$bin" "$model" "$inputs"
    nvcc --version
    cuobjdump -lelf "$bin"
    env | rg '^(CUDA_|M40LLM_)' | sort
    snapshot
} > "$out/preflight.txt"
bash ../consolidation/docs/experiments/2026-11-21-protocol/host-load.sh "$$" > "$out/host-load.log" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true' EXIT
"$bin" history "$model" ../raw/november-quality-inputs.json "$inputs" "$out/report.jsonl" > "$out/run.log" 2>&1
snapshot > "$out/after.txt"
