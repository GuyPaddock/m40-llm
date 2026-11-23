#!/usr/bin/env bash
# Invoked with the shared M40 lock held, from this source checkout.
set -euo pipefail
if [[ $# -ne 6 ]]; then
    echo 'usage: run.sh MODEL MODEL_QUALITY RESEARCH_HTTP_PAIRS FINISHED_QUALITY QUALITY_REPLAY NEW_RAW_DIR' >&2
    exit 2
fi
source scripts/dev-env.sh
model=$(realpath "$1")
binary=$(realpath "$2")
stats=$(realpath "$3")
quality=$(realpath "$4")
quality_replay=$(realpath "$5")
out=$(realpath -m "$6")
cutoffs=docs/experiments/2026-11-22-sparse-q4-live/cutoffs.json
manifest=../consolidation/docs/experiments/2026-11-21-protocol/performance-contexts.json
[[ -z $(git status --porcelain) && -s "$quality/complete.txt" && -x "$binary" && -x "$stats" ]]
jq -es 'last | .kind == "sparse_q4_extended_complete" and .summary.cases == 48 and
    .summary.disposition == "PASS" and .device_bytes_after == 0' "$quality/packed-thresholded.jsonl" >/dev/null
jq -e '.disposition == "PASS" and .cases == 48' "$quality_replay" >/dev/null
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
    sha256sum "$model" "$binary" "$stats" "$manifest" "$cutoffs" "$quality_replay" "$quality/packed-thresholded.jsonl"
    nvcc --version
    cuobjdump -lelf "$binary"
    env | rg '^(M40LLM_|CUDA_)' | sort
    snapshot
} > "$out/preflight.txt"
rg -q 'sm_52.cubin' "$out/preflight.txt"
bash ../consolidation/docs/experiments/2026-11-21-protocol/host-load.sh "$$" > "$out/host-load.log" &
host=$!
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=timestamp,pid,used_gpu_memory --format=csv --loop-ms=250 > "$out/processes.csv" &
monitor=$!
trap 'kill "$host" "$monitor" 2>/dev/null || true; wait "$host" "$monitor" 2>/dev/null || true' EXIT

sample() {
    local label=$1 arm=$2 variant=dense pid
    if ((arm == 1)); then variant=packed-thresholded; fi
    snapshot > "$out/$label.before"
    "$binary" --model "$model" --reference-model "$model" --split held-out \
        --context-tokens 1024 --variant-id "$variant" --sparse-q4-layout packed \
        --sparse-q4-thresholded --sparse-q4-cutoffs "$cutoffs" \
        --sparse-q4-request-input "$manifest" --output "$out/$label.jsonl" \
        > "$out/$label.log" 2>&1 &
    pid=$!
    printf '%s\n' "$pid" > "$out/$label.pid"
    wait "$pid"
    jq -es --arg variant "$variant" 'length == 2 and
        .[0].kind == "sparse_q4_request" and .[0].variant == $variant and
        .[1].kind == "sparse_q4_request_complete" and .[1].tracked_device_bytes == 0' "$out/$label.jsonl" >/dev/null
    jq -cS 'select(.kind == "sparse_q4_request") | .generation |
        {generated_token_ids, stop_reason:.metrics.stop_reason}' "$out/$label.jsonl" > "$out/$label.work.json"
    jq -c --arg label "$label" --argjson arm "$arm" 'select(.kind == "sparse_q4_request") |
        {label:$label,arm:$arm,http_seconds:(.generation.resident_generation_wall_ms/1000), metrics:.generation.metrics}' \
        "$out/$label.jsonl" >> "$out/timings.jsonl"
    snapshot > "$out/$label.after"
}

series() {
    local phase=$1 first=$2 limit=$3 trial arm
    for ((trial=first; trial<limit; trial++)); do
        for arm in "$((trial%2))" "$((1-trial%2))"; do
            sample "$phase-$trial-$arm" "$arm"
        done
        for arm in 0 1; do
            if ! cmp -s "$out/pair-0-0.work.json" "$out/$phase-$trial-$arm.work.json"; then
                printf '%s\n' "$phase-$trial-$arm" > "$out/nonmatching-work.txt"
                snapshot > "$out/complete.txt"
                exit 0
            fi
        done
    done
}

series pair 0 5
"$stats" "$out/timings.jsonl" > "$out/screen.json"
decision=$(jq -r .next "$out/screen.json")
if [[ "$decision" == extend ]]; then
    series pair 5 10
    "$stats" "$out/timings.jsonl" > "$out/extended.json"
    decision=$(jq -r .next "$out/extended.json")
fi
if [[ "$decision" == confirm ]]; then
    series confirm 0 20
    "$stats" --confirmation "$out/timings.jsonl" > "$out/confirmation.json"
fi
snapshot > "$out/complete.txt"
