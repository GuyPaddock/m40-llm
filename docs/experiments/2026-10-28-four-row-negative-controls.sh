#!/usr/bin/env bash
# Corrupt copies only; measured evidence is never edited.
set -euo pipefail
raw=${1:?screen JSONL path}
audit=docs/experiments/2026-10-28-four-row-audit.pl
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
perl "$audit" "$raw" > "$temporary/positive.json"

# The audit must reject changed identity, behavior, work and pairing evidence.
reject() {
    local label=$1 mutation=$2
    jq -c "$mutation" "$raw" > "$temporary/corrupt.jsonl"
    if perl "$audit" "$temporary/corrupt.jsonl" > "$temporary/result" 2>&1; then
        printf 'unexpected audit success: %s\n' "$label" >&2
        exit 1
    fi
    jq -nc --arg label "$label" '{control:$label,is_rejected:true}'
}

reject incomplete 'select(.kind!="r4_complete")'
reject wrong-model 'if .kind=="r4_manifest" then .model_sha256="wrong" else . end'
reject changed-fixture 'if .kind=="r4_manifest" then .fixture_sha256="wrong" else . end'
reject changed-memory 'if .kind=="r4_preparation" then .tracked_resident+=1 else . end'
reject wrong-down 'if .kind=="r4_trial" then .measurements[0].stages[0].dispatch.gemm_f32xq8_separate_f32=999 else . end'
reject extra-sync 'if .kind=="r4_trial" then .measurements[0].stages[0].enqueue_counters.stream_syncs+=1 else . end'
reject changed-output 'if .kind=="r4_validation" then .has_exact_bits=false else . end'
reject changed-pair 'if .kind=="r4_summary" then .versus_serial.pairs[0][0]+=1 else . end'
reject overlap 'if .kind=="r4_trial" then .measurements[0].stages[0].started_unix_us=0 else . end'
reject altered-clocks 'if .kind=="r4_trial" then .gpu_before.device|=sub("1113 MHz";"822 MHz") else . end'
