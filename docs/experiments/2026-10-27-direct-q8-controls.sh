#!/usr/bin/env bash
# Exercise audit outcomes; never mutate a measured report.
set -euo pipefail
raw=${1:?diagnostic JSONL path}
model=docs/experiments/2026-10-27-direct-q8-model.pl
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
perl "$model" --self-test > "$temporary/math.json"
perl "$model" "$raw" --audit-only > "$temporary/valid.json"

reject() {
    local label=$1 mutation=$2
    jq -c "$mutation" "$raw" > "$temporary/corrupt.jsonl"
    if perl "$model" "$temporary/corrupt.jsonl" --audit-only > "$temporary/result" 2>&1; then
        printf 'unexpected audit success: %s\n' "$label" >&2
        exit 1
    fi
    jq -nc --arg label "$label" '{control:$label,is_rejected:true}'
}

reject incomplete 'select(.kind!="direct_q8_cost_complete")'
reject wrong-model 'if .kind=="direct_q8_cost_manifest" then .model_sha256="wrong" else . end'
reject changed-owner 'if .kind=="direct_q8_cost_ownership" then .ownership="retain-split" else . end'
reject wrong-bytes 'if .kind=="direct_q8_cost_ownership" then .split.resident_bytes-=1 else . end'
reject changed-output 'if .kind=="direct_q8_cost_trial" then .row.predictions[0]+=1 else . end'
reject changed-prefix 'if .kind=="direct_q8_cost_trial" then .row.after_prefix_sha256="wrong" else . end'
reject wrong-length 'if .kind=="direct_q8_cost_trial" then .row.lengths[0]+=1 else . end'
reject free-bonus 'if .kind=="direct_q8_cost_trial" then .row.useful_progress+=1 else . end'
reject missing-pair 'select(.kind!="direct_q8_cost_trial" or .pair!=0 or .row.width!=2 or .row.is_profiled==true)'
reject uncounted-gemm 'if .kind=="direct_q8_cost_trial" then .row.begin_counters.cublas_calls=0 else . end'
reject altered-clocks 'if .kind=="direct_q8_cost_trial" then .gpu_before.device|=sub("1113 MHz";"822 MHz") else . end'
reject competing-process 'if .kind=="direct_q8_cost_trial" then .gpu_after.processes+="\nother GPU user" else . end'
reject missing-span 'if .kind=="direct_q8_cost_trial" and .row.is_profiled then .row.gpu_profile.stages|=.[1:] else . end'
reject changed-observer 'if .kind=="direct_q8_cost_observer" then .latency_overhead=0.1 else . end'
reject uncleared-suffix 'if .kind=="direct_q8_cost_rejection" then .row.has_cleared_suffix=false else . end'
reject wrong-correction 'if .kind=="direct_q8_cost_rejection" then .row.correction_token+=1 else . end'
reject negative-rollback 'if .kind=="direct_q8_cost_rejection" then .row.commit_ms= -1 else . end'
reject teardown-leak 'if .kind=="direct_q8_cost_complete" then .device_bytes_after=1 else . end'
