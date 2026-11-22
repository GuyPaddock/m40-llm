#!/usr/bin/env bash
# Confirm matched instrumented requests and per-process sample attribution.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: bash server-peak-replay.sh FINISHED_RAW_DIR compound|native-split" >&2
    exit 2
fi
raw=$(realpath "$1")
case "$2" in
    compound) arms=(retain-split joint-only) ;;
    native-split) arms=(native split) ;;
    *) exit 2 ;;
esac
[[ -s "$raw/complete.txt" ]]
previous_end=""
for arm in "${arms[@]}"; do
    pid=$(cat "$raw/$arm.pid")
    [[ "$pid" =~ ^[1-9][0-9]*$ ]]
    started=$(head -1 "$raw/$arm.start")
    ready=$(head -1 "$raw/$arm.ready")
    ended=$(head -1 "$raw/$arm.end")
    [[ "$previous_end" < "$started" && "$started" < "$ready" && "$ready" < "$ended" ]]
    previous_end=$ended
    peak=$(awk -F, -v pid="$pid" '
        $2+0 == pid && $3 ~ /MiB$/ { m=$3+0; if(m>max)max=m; count++ }
        END { if(!count || !max)exit 1; printf "%d",max }
    ' "$raw/processes.csv")
    rg -o 'sampled token id=[0-9]+' "$raw/$arm.log" | cut -d= -f2 |
        jq -es --slurpfile ids "$raw/expected.json" '. == ($ids[0]+$ids[0]) and length == 1024' >/dev/null
    if [[ "$2" == compound ]]; then
        rg -q 'packed prefix prefill enabled layers=36' "$raw/$arm.log"
        rg -q '^M40LLM_Q8_PREFILL_BACKEND=dequant-sgemm$' "$raw/$arm.start"
        if [[ "$arm" == joint-only ]]; then owner=JointOnly; else owner=RetainSplit; fi
        rg -q "ownership=$owner" "$raw/$arm.log"
    else
        rg -q 'full-layer forward enabled layers=36' "$raw/$arm.log"
        if rg -q 'packed prefix prefill enabled' "$raw/$arm.log"; then exit 1; fi
    fi
    setup_ns=$(($(date -d "$ready" +%s%N) - $(date -d "$started" +%s%N)))
    jq -n --arg arm "$arm" --argjson pid "$pid" --argjson peak "$peak" \
        --argjson setup "$setup_ns" --slurpfile cold "$raw/$arm-cold.json" --slurpfile warm "$raw/$arm-warm.json" '
        [$cold[0],$warm[0]] as $requests
        | if all($requests[]; .success and (.records|length) == 1
             and .records[0].status == 200 and .records[0].response.metrics.prompt_tokens == 65
             and .records[0].response.metrics.generated_tokens == 512
             and .records[0].response.metrics.stop_reason == "length")
             and $requests[0].records[0].response.output == $requests[1].records[0].response.output
          then {arm:$arm, pid:$pid, sampled_peak_mib:$peak, launch_to_health_ms:($setup / 1000000),
                diagnostic_wall_ms:($requests | map(.records[0].wall_ms)), matched_generated_ids_per_request:512}
          else error("wrong diagnostic request signature") end
    '
done | jq -s '. as $arms | {arms:$arms, candidate_minus_control_peak_mib:($arms[1].sampled_peak_mib-$arms[0].sampled_peak_mib)}'
