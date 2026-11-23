#!/usr/bin/env bash
# Replay the completed five-pair cohort without a GPU.
set -euo pipefail
raw=$(realpath "${1:?finished raw directory}")
stats=$(realpath "${2:?research_http_pairs binary}")
[[ -s "$raw/complete.txt" && ! -e "$raw/nonmatching-work.txt" ]]
stats_sha=$(sha256sum "$stats" | cut -d ' ' -f 1)
awk -v sha="$stats_sha" '$1==sha && $2 ~ /\/research_http_pairs$/ {found=1} END {exit !found}' "$raw/preflight.txt"
temporary=$(mktemp)
trap 'rm -f "$temporary"' EXIT
for ((pair=0; pair<5; pair++)); do
    for arm in "$((pair%2))" "$((1-pair%2))"; do
        label="pair-$pair-$arm"
        variant=dense
        if ((arm == 1)); then variant=packed-thresholded; fi
        source="$raw/$label.jsonl"
        jq -es --arg variant "$variant" 'length==2 and .[0].kind=="sparse_q4_request" and
            .[0].variant==$variant and .[0].generation.metrics.prompt_tokens==65 and
            .[0].generation.metrics.generated_tokens==32 and .[0].generation.metrics.stop_reason=="length" and
            .[1].kind=="sparse_q4_request_complete" and .[1].tracked_device_bytes==0' "$source" >/dev/null
        jq -cS 'select(.kind=="sparse_q4_request") | .generation |
            {generated_token_ids,stop_reason:.metrics.stop_reason}' "$source" |
            cmp "$raw/pair-0-0.work.json" -
        jq -c --arg label "$label" --argjson arm "$arm" 'select(.kind=="sparse_q4_request") |
            {label:$label,arm:$arm,http_seconds:(.generation.resident_generation_wall_ms/1000), metrics:.generation.metrics}' \
            "$source" >> "$temporary"
    done
done
cmp "$raw/timings.jsonl" "$temporary"
"$stats" "$temporary" | cmp "$raw/screen.json" -
echo 'PASS: ten matching requests, exact outputs, allocation release and paired statistics'
