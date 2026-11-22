#!/usr/bin/env bash
# From the consolidation worktree, with the exclusive research GPU lock held.
set -euo pipefail
out=$(realpath -m "$1")
mkdir "$out"
study=$(realpath docs/experiments/2026-11-21-native-split)
reference=$(realpath ../../m40-llm)
bin=$(realpath target/release/examples/compare_ollama)
ollama=$(realpath ../tools/ollama-0.40.11/bin/ollama)
model="$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
gpu=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
[[ $(git -C "$reference" rev-parse HEAD) == 48c63332462e009964e891be1cabaf139e3b50f8 ]]
[[ -z $(git -C "$reference" status --porcelain) ]]
for name in $(compgen -e | rg '^M40LLM_' || true); do unset "$name"; done
while IFS= read -r line; do export "$line"; done < <(rg '^M40LLM_[A-Z0-9_]+=' "$study/preflight.txt")
snapshot() {
    date --iso-8601=ns
    nvidia-smi -i "$gpu" --query-gpu=uuid,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv
    local owners
    owners=$(nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader)
    echo "$owners"
    [[ -z "$owners" ]]
}
{
    snapshot
    git rev-parse HEAD
    git -C "$reference" rev-parse HEAD
    printf '%s  %s\n' cfe6d2cf6e56e2aa0c23b571d39db965fe5d4ee732ecfc68acdb7e47e50b1eea "$bin" \
        b0a4becfaf872e3ad5d2e9051244e0dfe55409f03d6e2cd9bd9c34400e201b4c "$ollama" \
        4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d "$model" | sha256sum -c -
    nvcc --version
    cuobjdump -lelf "$bin"
    env | rg '^(M40LLM_|CUDA_)' | sort
} > "$out/preflight.txt"
host=$(realpath docs/experiments/2026-11-21-protocol/host-load.sh)
bash "$host" "$$" > "$out/host-load.log" &
monitor=$!
oracle=""
cleanup() {
    if [[ -n "$oracle" ]]; then kill "$oracle" 2>/dev/null || true; wait "$oracle" 2>/dev/null || true; fi
    kill "$monitor" 2>/dev/null || true
    wait "$monitor" 2>/dev/null || true
}
trap cleanup EXIT
if curl -fsS --max-time 1 http://127.0.0.1:18435/api/version >/dev/null 2>&1; then echo "Oracle port already occupied" >&2; exit 1; fi
OLLAMA_HOST=127.0.0.1:18435 OLLAMA_MODELS="$HOME/.ollama/models" OLLAMA_DEBUG=1 \
    OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=30m \
    OLLAMA_NO_CLOUD=1 OLLAMA_NOPRUNE=1 LLAMA_ARG_CACHE_RAM=0 "$ollama" serve > "$out/oracle.log" 2>&1 &
oracle=$!
for ((try=0;try<120;try++)); do
    kill -0 "$oracle"
    if curl -fsS --max-time 1 http://127.0.0.1:18435/api/version > "$out/oracle-version.json" 2>/dev/null; then break; fi
    sleep 1
done
jq -e '.version == "0.40.11"' "$out/oracle-version.json"
cd "$reference"
"$bin" --model "$model" --ollama-pid "$oracle" --ollama-log "$out/oracle.log" --compare-q8-execution split \
    --q8-families down,vocabulary --output "$out/report.jsonl" > "$out/run.log" 2>&1
jq -s -f "$study/audit.jq" "$out/report.jsonl" > "$out/audit.json"
