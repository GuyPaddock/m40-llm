#!/usr/bin/env bash
# From the D3 worktree; take the exclusive hardware lock before entry.
set -euo pipefail
topic=$(pwd)
source ../consolidation/docs/experiments/2026-11-21-serving-baseline/server.sh
qwen=4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d
start_study "$1" "$HOME/.ollama/models/blobs/sha256-$qwen" "$qwen"
binary="$topic/target/release/m40-llm"
[[ -z $(git status --porcelain) ]]
{
    git rev-parse HEAD
    sha256sum "$binary"
    cuobjdump -lelf "$binary"
} >> "$out/preflight.txt"
jq '.cases[] | select(.name=="short") | .request | .max_tokens=128' "$manifest" > "$out/short.json"
jq '.cases[] | select(.name=="medium") | .request | .max_tokens=32' "$manifest" > "$out/medium.json"
short="$out/short.json"
medium="$out/medium.json"
schedule() {
    clear_flags
    export M40LLM_SERVER_BATCH_DECODE=1 M40LLM_SERVER_BATCH_PREFILL=1 M40LLM_SERVER_BATCH_DECODE_SLOTS=4
    if [[ "$1" == diagnostic ]]; then export M40LLM_RESEARCH_TOKEN_TIMINGS=1; fi
    start_server "$binary" "$1-server" --kv-compress-mode off --max-context-tokens 2304
    call_clients "$1-warm-short" 0 "$short"
    call_clients "$1-warm-medium" 0 "$medium"
}
schedule diagnostic
call_clients calibration 0 "$short"
delay=$(jq -er '.records[0].response.metrics.token_elapsed_ms[3]|select(.>0) | ceil+25' "$out/calibration.json")
echo "$delay" > "$out/stagger-ms.txt"
series() {
    local phase=$1 i arm
    for i in 0 1 2 3 4; do
        for arm in "$((i%2))" "$((1-i%2))"; do
            if [[ "$arm" == 0 ]]; then
                call_clients "$phase-$i-control" "0,$delay" "$short" "$short"
            else
                call_clients "$phase-$i-treatment" "0,$delay" "$short" "$medium"
            fi
        done
    done
}
for phase in diagnostic primary; do
    if [[ "$phase" == primary ]]; then schedule primary; fi
    call_clients "$phase-single-short" 0 "$short"
    call_clients "$phase-single-medium" 0 "$medium"
    call_clients "$phase-four" 0,0,0,0 "$short" "$short" "$short" "$short"
    series "$phase"
    stop_server
done
gpu_snapshot > "$out/complete.txt"
"$topic/target/cpu/release/examples/research_fairness_report" "$out" > "$out/summary.json"
