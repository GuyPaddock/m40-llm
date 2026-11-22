#!/usr/bin/env bash
# From this research worktree, with the serial M40 lock held.
set -euo pipefail
cache=${1:?default or dense}
count=${2:?number of concurrent requests}
case "$cache" in dense|default) ;; *) exit 2 ;; esac
case "$count" in 1|2|4) ;; *) exit 2 ;; esac
topic=$(pwd)
study="$topic/docs/experiments/2026-11-21-serving-concurrency"
source ../consolidation/docs/experiments/2026-11-21-serving-baseline/server.sh
model_hash=4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d
start_study "$3" "$HOME/.ollama/models/blobs/sha256-$model_hash" "$model_hash"
primary="$topic/target/release/m40-llm"
diagnostic="$workspace/raw/serving-concurrency-build-01/diagnostic-server"
report="$topic/target/cpu/release/examples/research_fairness_report"
ids="$consolidation/target/release/examples/research_serving_ids"
[[ -z $(git status --porcelain) ]]
{
    git rev-parse HEAD
    (cd "$workspace" && sha256sum -c "$study/binaries.sha256")
    sha256sum "$report" "$ids"
    cuobjdump -lelf "$diagnostic"
    cuobjdump -lelf "$primary"
} >> "$out/preflight.txt"
jq '.cases[]|select(.name=="short")|.request|.max_tokens=64' "$manifest" > "$out/request.json"
"$ids" "$model" "$out/request.json" > "$out/prompt-ids.json"
jq -e --slurpfile ids "$out/prompt-ids.json" '.cases[]|select(.name=="short")|.prompt_ids==$ids[0].ids and (.prompt_ids|length)==65' "$manifest" >/dev/null
requests=()
offsets=()
for ((slot=0; slot<count; slot++)); do requests+=("$out/request.json"); offsets+=(0); done
delays=$(IFS=,; echo "${offsets[*]}")
args=(--max-context-tokens 1024)
if [[ "$cache" == dense ]]; then args+=(--kv-compress-mode off); fi
boot() {
    local arm=$1 label=$2 server=$3
    clear_flags
    export M40LLM_SERVER_BATCH_DECODE="$arm" M40LLM_SERVER_BATCH_PREFILL="$arm" M40LLM_SERVER_BATCH_DECODE_SLOTS=4
    if [[ "$server" == "$diagnostic" ]]; then
        export M40LLM_RESEARCH_TOKEN_TIMINGS=1 M40LLM_SERVER_BATCH_LOG=1 M40LLM_SERVER_BATCH_PREFILL_LOG=1
    fi
    start_server "$server" "$label" "${args[@]}"
    call_clients "warm-$label" "$delays" "${requests[@]}"
}
for arm in 0 1; do
    label="trace-$arm"
    boot "$arm" "$label" "$diagnostic"
    before=$(wc -c < "$out/$label.log")
    call_clients "$label" "$delays" "${requests[@]}"
    stop_server
    tail -c "+$((before + 1))" "$out/$label.log" > "$out/$label.measured.log"
    "$report" "$out" --client-run "$label" --diagnostic > "$out/$label.summary.json"
    jq -ce '[.records[]|[.slot,.response.metrics.sampled_ids]]' "$out/$label.json" > "$out/$label.ids"
done
equivalent=true
cmp -s "$out/trace-0.ids" "$out/trace-1.ids" || equivalent=false
completed_width=$(sed -nE 's/.*completed batched decode rows=([0-9]+).*/\1/p' "$out/trace-1.measured.log" | sort -nr | sed -n '1p')
completed_width=${completed_width:-0}
# At one client, the scheduler intentionally takes the single-sequence path.
if ((count>1 && completed_width<count)); then equivalent=false; fi
jq -n --argjson n "$count" --argjson w "$completed_width" --argjson ok "$equivalent" \
    '{clients:$n,max_completed_decode_width:$w,diagnostic_gate:$ok}' > "$out/admission.json"
series() {
    local phase=$1 begin=$2 end=$3 trial arm label
    for ((trial=begin;trial<end;trial++)); do
        for arm in "$((trial%2))" "$((1-trial%2))"; do
            label="$phase-$trial-$arm"
            boot "$arm" "server-$label" "$primary"
            call_clients "$label" "$delays" "${requests[@]}"
            stop_server
            "$report" "$out" --client-run "$label" > "$out/$label.summary.json"
            jq -ce --arg label "$label" --argjson arm "$arm" --argjson count "$count" \
                --slurpfile diagnostic "$out/trace-$arm.json" --slurpfile summary "$out/$label.summary.json" \
                -f "$study/observe.jq" "$out/$label.json" >> "$out/timings.jsonl"
        done
    done
}
series pair 0 5
matches_companions() {
    jq -es 'all(.[]; .matches_companion)' "$out/timings.jsonl" > /dev/null
}
if [[ "$equivalent" == true ]] && matches_companions; then
    "$stats" "$out/timings.jsonl" > "$out/screen.json"
    decision=$(jq -r .next "$out/screen.json")
    if [[ "$decision" == extend ]]; then
        series pair 5 10
        if matches_companions; then
            "$stats" "$out/timings.jsonl" > "$out/extended.json"
            decision=$(jq -r .next "$out/extended.json")
        fi
    fi
    if [[ "$decision" == confirm ]]; then
        series confirm 0 20
        if matches_companions; then "$stats" --confirmation "$out/timings.jsonl" > "$out/confirmation.json"; fi
    fi
fi
jq -es --argjson ok "$equivalent" '{all_outputs_match:all(.[];.matches_companion),
    equivalent_work:($ok and all(.[];.matches_companion))}' "$out/timings.jsonl" > "$out/eligibility.json"
gpu_snapshot > "$out/complete.txt"
