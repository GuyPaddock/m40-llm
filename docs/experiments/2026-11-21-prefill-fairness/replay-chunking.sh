#!/usr/bin/env bash
# Replay a completed prefill-chunking cohort from its original client records.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo 'usage: bash replay-chunking.sh FINISHED_RAW_DIR RESEARCH_FAIRNESS_REPORT' >&2
    exit 2
fi
raw=$(realpath "$1")
report=$(realpath "$2")
study=$(cd "$(dirname "$0")" && pwd)
[[ -s "$raw/complete.txt" ]]
sha=$(sha256sum "$report" | cut -d ' ' -f 1)
awk -v sha="$sha" '$1==sha && $2 ~ /\/research_fairness_report$/ { found=1 } END { exit !found }' "$raw/preflight.txt"
python3 "$study/score-chunking.py" "$raw" | diff -u "$raw/summary.json" -

jq -es '
    map(.records | map(.response.metrics | [.prompt_tokens,.generated_tokens,.stop_reason,.sampled_ids])) |
    length==2 and (.[0]|length)==4 and .[0]==.[1] and
    all(.[0][]; .[0:2]==[65,128] and (.[3]|length)==128)
' "$raw"/four-{baseline,candidate}.json > /dev/null

server_peak() {
    local request=$1 prefix="$1-server" pid from through
    pid=$(<"$raw/$prefix.pid")
    [[ "$pid" =~ ^[1-9][0-9]*$ ]]
    from=$(date -d "$(sed -n '1p' "$raw/$prefix.start")" '+%Y/%m/%d %T.%3N')
    through=$(date -d "$(sed -n '1p' "$raw/$request.after")" '+%Y/%m/%d %T.%3N')
    awk -F, -v pid="$pid" -v from="$from" -v through="$through" '
        $2+0==pid && $1 >= from && $1 <= through && $3 ~ /[0-9]+ MiB/ {
            value=$3+0; if (!samples || value>peak) peak=value; samples++
        } END { if (!samples) exit 1; print peak }
    ' "$raw/processes.csv"
}

records=$(mktemp)
trap 'rm -f -- "$records"' EXIT
for i in 0 1 2 3 4; do
    jq -es '
      map(.records | map(.response.metrics | [.prompt_tokens,.generated_tokens,.stop_reason,.sampled_ids])) |
      length==2 and .[0]==.[1] and (.[0]|map(.[0:2]))==[[65,128],[2060,32]]
    ' "$raw"/diagnostic-"$i"-{baseline,candidate}.json > /dev/null
    for arm in baseline candidate; do
        key="$i-$arm"
        "$report" "$raw" --client-run "diagnostic-$key" --diagnostic |
            diff -u "$raw/diagnostic-$key.clocks.json" -
        jq -es -f "$study/responses.jq" "$raw"/{diagnostic,primary}-"$key.json" |
            diff -u "$raw/primary-$key.replay.json" -
        for phase in diagnostic primary; do
            label="$phase-$key"
            flags="$raw/$label-server.start"
            rg -Fq -- '--kv-compress-mode off --max-context-tokens 2304' "$flags"
            rg -xq 'M40LLM_SERVER_BATCH_DECODE_SLOTS=4' "$flags"
            rg -xq 'M40LLM_SERVER_BATCH_DECODE=1' "$flags"
            rg -xq 'M40LLM_SERVER_BATCH_PREFILL=1' "$flags"
            if [[ "$arm" == candidate ]]; then
                rg -xq 'M40LLM_RESEARCH_DENSE_PREFILL_CHUNKS=1' "$flags"
            else
                if rg -q '^M40LLM_RESEARCH_DENSE_PREFILL_CHUNKS=' "$flags"; then exit 1; fi
            fi
            if [[ "$phase" == diagnostic ]]; then
                rg -xq 'M40LLM_RESEARCH_TOKEN_TIMINGS=1' "$flags"
            else
                if rg -q '^M40LLM_RESEARCH_TOKEN_TIMINGS=' "$flags"; then exit 1; fi
            fi
            jq -nc --arg arm "$arm" --arg phase "$phase" --argjson pair "$i" \
                --argjson peak "$(server_peak "$label")" \
                '{pair:$pair,arm:$arm,phase:$phase,server_peak_mib:$peak}' >> "$records"
        done
    done
done
jq -s --slurpfile gate "$raw/summary.json" \
    '{four_client_exact_ids:512,variants:.,latency_decision:$gate[0]}' "$records"
