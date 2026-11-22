#!/usr/bin/env bash
# Read all nine single-process arms. Times in the NVML CSV are local wall time.
set -euo pipefail
export TZ=America/New_York LC_ALL=C
raw=$(realpath "${1:?completed residency directory}")
audit=$(cd "$(dirname "$0")" && pwd)
test -s "$raw/after.txt"
for capacity in 1024 3072 8448; do
    for label in dense top64 top8; do
        name="$raw/$capacity-$label"
        start=$(date -d "$(head -1 "$name.before")" '+%Y/%m/%d %T.%3N')
        end=$(date -d "$(head -1 "$name.after")" '+%Y/%m/%d %T.%3N')
        samples=$(awk -F ', *' -v lo="$start" -v hi="$end" '
            $1>=lo && $1<=hi {
                if ($2 !~ /^[0-9]+$/ || (pid && $2!=pid) || $3 !~ /^[0-9]+ MiB$/) bad=1
                pid=$2; used=$3+0; if (used>peak) peak=used; ++count
            }
            END {if(bad || !peak || !count) exit 1
                 printf "{\"pid\":%d,\"observations\":%d,\"peak_mib\":%d}",pid,count,peak}' "$raw/processes.csv")
        case "$label" in
            dense) arm=dense ;;
            top64) arm=quantized_all ;;
            top8) arm=quantized_top8 ;;
        esac
        jq -cse --arg arm "$arm" --argjson capacity "$capacity" --argjson samples "$samples" \
            -f "$audit/residency.jq" "$name.jsonl"
    done
done | jq -es 'if length==9 and (map(.sampled_process.pid)|unique|length)==9 then . else error("missing independent process") end'
