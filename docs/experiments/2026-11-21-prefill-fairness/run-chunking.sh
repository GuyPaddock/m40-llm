#!/usr/bin/env bash
# Run from the prefill-chunking worktree with the shared M40 lock held.
set -euo pipefail
topic=$(pwd)
study=docs/experiments/2026-11-21-prefill-fairness
source "../consolidation/docs/experiments/2026-11-21-serving-baseline/server.sh"
build=$(realpath "${2:?checked build directory}")
binary="$topic/target/release/m40-llm"
report="$workspace/prefill-fairness/target/cpu/release/examples/research_fairness_report"
qwen=4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d
test -s "$build/complete.txt"
test -s "$build/cublas-on.pass"
test -s "$build/cublas-off.pass"
sha256sum -c "$build/server.sha256"
git diff --quiet "$(cat "$build/source.txt")" HEAD -- src cuda examples Cargo.toml Cargo.lock build.rs
test -z "$(git status --porcelain)"

start_study "${1:?fresh output directory}" "$HOME/.ollama/models/blobs/sha256-$qwen" "$qwen"
{
    git rev-parse HEAD
    cat "$build/source.txt"
    sha256sum "$binary" "$report"
    cuobjdump -lelf "$binary" | rg 'sm_52'
} >> "$out/preflight.txt"
jq '.cases[] | select(.name=="short") | .request | .max_tokens=128' "$manifest" > "$out/short.json"
jq '.cases[] | select(.name=="medium") | .request | .max_tokens=32' "$manifest" > "$out/medium.json"
short="$out/short.json"
medium="$out/medium.json"

schedule() {
    local label=$1 phase=$2 variant=$3
    clear_flags
    export M40LLM_SERVER_BATCH_DECODE=1 M40LLM_SERVER_BATCH_PREFILL=1 M40LLM_SERVER_BATCH_DECODE_SLOTS=4
    if [[ "$phase" == diagnostic ]]; then export M40LLM_RESEARCH_TOKEN_TIMINGS=1; fi
    if [[ "$variant" == candidate ]]; then export M40LLM_RESEARCH_DENSE_PREFILL_CHUNKS=1; fi
    start_server "$binary" "$label" --kv-compress-mode off --max-context-tokens 2304
    call_clients "$label-warm" 0 "$short"
}

schedule calibration diagnostic baseline
call_clients calibration 0 "$short"
delay=$(jq -er '.records[0].response.metrics.token_elapsed_ms[3]|select(.>0) | ceil+25' "$out/calibration.json")
echo "$delay" > "$out/stagger-ms.txt"
stop_server

for variant in baseline candidate; do
    schedule "four-$variant-server" diagnostic "$variant"
    call_clients "four-$variant" "0,0,0,0" "$short" "$short" "$short" "$short"
    stop_server
done
jq -es '
    map(.records | map(.response.metrics | [.prompt_tokens,.generated_tokens,.stop_reason,.sampled_ids])) |
    (.[0]|length)==4 and .[0]==.[1] and all(.[0][]; .[0:2]==[65,128])
' "$out"/four-{baseline,candidate}.json > /dev/null

for phase in diagnostic primary; do
    for i in 0 1 2 3 4; do
        arms=(baseline candidate)
        if ((i % 2 == 1)); then arms=(candidate baseline); fi
        for variant in "${arms[@]}"; do
            label="$phase-$i-$variant"
            schedule "$label-server" "$phase" "$variant"
            call_clients "$label" "0,$delay" "$short" "$medium"
            stop_server
            if [[ "$phase" == diagnostic ]]; then
                "$report" "$out" --client-run "$label" --diagnostic > "$out/$label.clocks.json"
            else
                jq -es -f "$study/responses.jq" "$out"/{diagnostic,primary}-"$i-$variant.json" > "$out/$label.replay.json"
                jq -e '.work_matches' "$out/$label.replay.json" > /dev/null
            fi
        done
        if [[ "$phase" == diagnostic ]]; then
            jq -es '
              map(.records | map(.response.metrics |
                  [.prompt_tokens,.generated_tokens,.stop_reason,.sampled_ids])) |
              .[0] == .[1] and (.[0]|map(.[0:2])) == [[65,128],[2060,32]]
              ' "$out"/diagnostic-"$i"-{baseline,candidate}.json > /dev/null
        fi
    done
done
gpu_snapshot > "$out/complete.txt"
python3 "$study/score-chunking.py" "$out" > "$out/summary.json"
