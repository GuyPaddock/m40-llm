#!/usr/bin/env bash
# Relate fresh-process NVML samples to the two independent load intervals.
set -euo pipefail
export LC_ALL=C TZ=America/New_York
raw=$(realpath "${1:?raw directory}")
here=$(cd "$(dirname "$0")" && pwd)
peaks=()
for name in load-retain load-joint; do
    start=$(date -d "$(head -1 "$raw/$name.before")" '+%Y/%m/%d %T.%3N')
    stop=$(date -d "$(head -1 "$raw/$name.after")" '+%Y/%m/%d %T.%3N')
    peaks+=("$(awk -F ', *' -v lo="$start" -v hi="$stop" '
        $1>=lo && $1<=hi {
            if($2 !~ /^[0-9]+$/ || (pid && pid!=$2) || $3 !~ /^[0-9]+ MiB$/) bad=1
            pid=$2; if($3+0>peak) peak=$3+0
        }
        END{if(!peak||bad)exit 1; print peak}' "$raw/processes.csv")")
done
jq -en --slurpfile retain "$raw/load-retain.jsonl" --slurpfile joint "$raw/load-joint.jsonl" \
    --slurpfile quality "$raw/quality.jsonl" --slurpfile http "$raw/http.jsonl" \
    --arg quality_sha "$(sha256sum "$raw/quality.jsonl" | cut -d' ' -f1)" \
    --argjson peaks "[${peaks[0]},${peaks[1]}]" -f "$here/ownership.jq"
