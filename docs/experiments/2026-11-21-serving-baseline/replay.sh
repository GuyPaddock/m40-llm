#!/usr/bin/env bash
# Reconstruct single-request results from a completed run.sh cohort.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cd "${1:?completed raw directory}"
[[ -s complete.txt ]]
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
clock=$(sed -n 's/^process_clock_ticks_per_second=//p' preflight.txt)
for arm in 0 1; do
    cmp "cli-ids-$arm.json" "ids-$arm.json"
done
# Parse /proc/PID/stat after the closing comm delimiter. The first remaining
# field is state (3), making user/system time 12/13 and process start 20.
cpu_sample() {
    awk '{pid=$1; sub(/^.*\) /,""); print pid,$20,$12+$13}' "$1"
}
process_peak() {
    awk -F, -v pid="$1" '$2+0==pid && $3 ~ /[0-9]+ MiB/ {
        m=$3+0; if(!samples || m>peak) peak=m; samples++
    } END {print samples ? peak : "null"}' processes.csv
}
while IFS= read -r label; do
    [[ $label =~ ^(pair|confirm)-[0-9]+-[01]$ ]]
    arm=${label##*-}
    pid=$(<"server-$label.pid")
    [[ $pid =~ ^[1-9][0-9]*$ ]]
    launch=$(date -d "$(head -1 "server-$label.start")" +%s%N)
    health=$(date -d "$(head -1 "server-$label.ready")" +%s%N)
    (( health >= launch ))
    cpu=null
    if [[ -e "$label.cpu-before" || -e "$label.cpu-after" || -n "$clock" ]]; then
        [[ $clock =~ ^[1-9][0-9]*$ ]]
        read -r first start a < <(cpu_sample "$label.cpu-before")
        read -r last end b < <(cpu_sample "$label.cpu-after")
        [[ $first == "$pid" && $last == "$pid" && $start == "$end" && $b -ge $a ]]
        cpu=$(jq -n --argjson delta "$((b-a))" --argjson hz "$clock" '$delta/$hz')
    fi
    jq -ec --arg label "$label" --argjson arm "$arm" \
        --argjson cpu "$cpu" --argjson peak "$(process_peak "$pid")" --argjson launch_ns "$((health-launch))" \
        --slurpfile trace "calibrate-$arm.json" --slurpfile ids "ids-$arm.json" \
        --slurpfile prompt phase4-prompt-ids.json -f "$here/trial.jq" "$label.json"
done < <(jq -r .label timings.jsonl) > "$scratch/trials"
for arm in 0 1; do
    jq -enc --argjson arm "$arm" --slurpfile trace "calibrate-$arm.json" \
        --slurpfile ids "ids-$arm.json" --slurpfile stream "stream-measured-$arm.json" \
        --rawfile cli "cli-$arm.time" -f "$here/diagnostic.jq"
done > "$scratch/diagnostics"
# Retain screen, extension and fresh confirmation as separate populations.
gates=(screen.json)
for gate in extended confirmation; do
    if [[ -e "$gate.json" ]]; then gates+=("$gate.json"); fi
done
jq -es --slurpfile trials "$scratch/trials" --slurpfile ledger timings.jsonl \
    --slurpfile diagnostics "$scratch/diagnostics" --slurpfile a ids-0.json \
    --slurpfile b ids-1.json -f "$here/cohort.jq" "${gates[@]}"
