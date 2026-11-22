#!/usr/bin/env bash
# From the clean copy worktree with scripts/dev-env.sh sourced.
set -euo pipefail
if [[ $# -ne 4 ]]; then
    echo "usage: bash run.sh MODEL NEW_RAW_DIR DRAFT_FEASIBILITY SHARED_GPU_LOCK" >&2
    exit 2
fi
model=$(realpath "$1")
out=$(realpath -m "$2")
bin=$(realpath "$3")
lock=$(realpath "$4")
reference=docs/experiments/2026-10-27-compound-reference.json
heldout=docs/experiments/2026-10-28-draft-heldout.jsonl
[[ -f "$lock" && -x "$bin" && -z $(git status --porcelain) ]]
mkdir "$out"
mapfile -t runtime_env < <(jq -er '.configuration.environment | to_entries[] | .key+"="+.value' "$reference")
for variable in ${!M40LLM_@} CUDA_VISIBLE_DEVICES; do
    unset "$variable"
done
export "${runtime_env[@]}"
exec 9<>"$lock"
flock -x 9
snapshot() {
    date --iso-8601=ns
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
}
idle() {
    local owners
    owners=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)
    [[ -z "$owners" ]]
}
idle
{
    git rev-parse HEAD
    git status --porcelain
    sha256sum "$bin" "$model" "$heldout" "$reference"
    cuobjdump -lelf "$bin"
    nvcc --version
    rustc --version
    env | rg '^(CUDA_VISIBLE_DEVICES|M40LLM_[A-Z0-9_]+)=' | sort
    snapshot
} > "$out/preflight.txt"
rg -q 'sm_52.cubin' "$out/preflight.txt"
bash ../consolidation/docs/experiments/2026-11-21-protocol/host-load.sh "$$" > "$out/host-load.log" &
host_sampler=$!
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=timestamp,pid,process_name,used_gpu_memory --format=csv -lms 250 > "$out/processes.csv" &
gpu_sampler=$!
trap 'kill "$host_sampler" "$gpu_sampler" 2>/dev/null || true; wait "$host_sampler" "$gpu_sampler" 2>/dev/null || true' EXIT
live_args=(live --compare-target-schedules --model "$model" --heldout "$heldout")
"$bin" "${live_args[@]}" --output "$out/correctness.jsonl" > "$out/correctness.log" 2>&1
idle
snapshot > "$out/before-timing.txt"
"$bin" "${live_args[@]}" --correctness-report "$out/correctness.jsonl" \
    --output "$out/timing.jsonl" > "$out/timing.log" 2>&1
idle
snapshot > "$out/after.txt"
jq -es '([.[] | select(.kind == "copy_schedule_gate")] | length) == 2 and last.does_release_all_device_bytes == true' "$out/timing.jsonl"
date --iso-8601=ns > "$out/complete.txt"
