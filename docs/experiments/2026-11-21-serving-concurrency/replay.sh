#!/usr/bin/env bash
# Verify completed buffered concurrency data from its original client records.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "usage: bash replay.sh FINISHED_RAW_DIR RESEARCH_FAIRNESS_REPORT RESEARCH_HTTP_PAIRS" >&2
    exit 2
fi
raw=$(realpath "$1")
report=$(realpath "$2")
stats=$(realpath "$3")
study=$(cd "$(dirname "$0")" && pwd)
[[ -s "$raw/complete.txt" ]]
for tool in "$report" "$stats"; do
    sha=$(sha256sum "$tool" | cut -d ' ' -f 1)
    awk -v sha="$sha" -v executable="$(basename "$tool")" '
        $1==sha && $2 ~ "/" executable "$" { found=1 }
        END { exit !found }
    ' "$raw/preflight.txt"
done
count=$(jq -er '.clients' "$raw/admission.json")
width=$(sed -nE 's/.*completed batched decode rows=([0-9]+).*/\1/p' "$raw/trace-1.measured.log" | sort -nr | sed -n '1p')
width=${width:-0}
for arm in 0 1; do
    "$report" "$raw" --client-run "trace-$arm" --diagnostic | diff -u "$raw/trace-$arm.summary.json" -
    jq -ce '[.records[] | [.slot,.response.metrics.sampled_ids]]' "$raw/trace-$arm.json" | diff -u "$raw/trace-$arm.ids" -
done
diagnostic_ok=true
if ! cmp -s "$raw/trace-0.ids" "$raw/trace-1.ids" || (( count > 1 && width < count )); then
    diagnostic_ok=false
fi
jq -e --argjson width "$width" --argjson valid "$diagnostic_ok" \
    '.max_completed_decode_width == $width and .diagnostic_gate == $valid' "$raw/admission.json" > /dev/null

work=$(mktemp -d)
trap 'rm -r -- "$work"' EXIT
jq -r '.label' "$raw/timings.jsonl" | while IFS= read -r label; do
    [[ "$label" =~ ^(pair|confirm)-[0-9]+-[01]$ ]]
    arm=${label##*-}
    "$report" "$raw" --client-run "$label" > "$work/report.json"
    diff -u "$raw/$label.summary.json" "$work/report.json"
    jq -ce --arg label "$label" --argjson arm "$arm" --argjson count "$count" \
        --slurpfile diagnostic "$raw/trace-$arm.json" --slurpfile summary "$work/report.json" \
        -f "$study/observe.jq" "$raw/$label.json"
done > "$work/timings.jsonl"
diff -u "$raw/timings.jsonl" "$work/timings.jsonl"
jq -es --argjson ok "$diagnostic_ok" '{all_outputs_match:all(.[];.matches_companion),
    equivalent_work:($ok and all(.[];.matches_companion))}' "$work/timings.jsonl" |
    diff -u "$raw/eligibility.json" -

if jq -e .equivalent_work "$raw/eligibility.json" > /dev/null; then
    check_screen() {
        "$stats" <(jq -c --argjson cutoff "$1" \
            'select(.label | test("^pair-[0-9]+-[01]$")) |
            select((.label | split("-")[1] | tonumber) < $cutoff)' "$work/timings.jsonl") |
            diff -u "$raw/$2.json" -
    }
    check_screen 5 screen
    if [[ -e "$raw/extended.json" ]]; then check_screen 10 extended; fi
    if [[ -e "$raw/confirmation.json" ]]; then
        "$stats" --confirmation "$work/timings.jsonl" | diff -u "$raw/confirmation.json" -
    fi
fi
jq -n --slurpfile admission "$raw/admission.json" --slurpfile eligibility "$raw/eligibility.json" \
    --slurpfile trials "$work/timings.jsonl" --slurpfile trace0 "$raw/trace-0.summary.json" \
    --slurpfile trace1 "$raw/trace-1.summary.json" \
    '{admission:$admission[0], eligibility:$eligibility[0], verified_timed_trials:($trials|length),
      traces:[$trace0[0],$trace1[0]]}'
