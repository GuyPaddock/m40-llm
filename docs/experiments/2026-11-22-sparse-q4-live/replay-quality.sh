#!/usr/bin/env bash
# Reconstruct each arm's frozen stop decision and process memory from raw data.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "usage: bash replay-quality.sh FINISHED_RAW_DIR CORPUS DENSE_HELDOUT_CAPTURE_JSONL" >&2
    exit 2
fi
raw=$(realpath "$1")
corpus=$(realpath "$2")
dense=$(realpath "$3")
study=$(cd "$(dirname "$0")" && pwd)
[[ -s "$raw/complete.txt" ]]
for label in packed-full packed-thresholded zigzag-full zigzag-thresholded; do
    pid=$(cat "$raw/$label.pid")
    peak=$(awk -F, -v pid="$pid" '$2+0==pid {v=$3+0;if(v>max)max=v} END{if(!max)exit 1;print max}' "$raw/processes.csv")
    jq --arg label "$label" --argjson peak "$peak" --slurpfile corpus "$corpus" \
       --slurpfile dense "$dense" -s -f "$study/replay-quality.jq" "$raw/$label.jsonl"
done | jq -s '.'
