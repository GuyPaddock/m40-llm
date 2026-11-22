#!/usr/bin/env bash
# Reconstruct portfolio request gates and resource evidence independently.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: bash portfolio-replay.sh RAW_DIRECTORY RESEARCH_HTTP_PAIRS" >&2
    exit 2
fi
raw=$(realpath "$1")
stats=$(realpath "$2")
study=$(cd "$(dirname "$0")" && pwd)
[[ -s "$raw/portfolio-http.after" ]]
work=$(mktemp -d)
trap 'rm -r -- "$work"' EXIT
digest=$(sha256sum "$raw/portfolio-quality.jsonl" | cut -d ' ' -f 1)
jq -n --arg digest "$digest" --slurpfile quality "$raw/portfolio-quality.jsonl" \
    --slurpfile timing "$raw/portfolio-http.jsonl" -f "$study/portfolio-replay.jq" > "$work/summary.json"

jq -c '.summaries[]' "$work/summary.json" | while IFS= read -r summary; do
    candidate=$(jq -r .candidate <<< "$summary")
    phase=$(jq -r .phase <<< "$summary")
    count=$(jq -r .comparison.pairs <<< "$summary")
    jq -c --arg candidate "$candidate" --arg phase "$phase" --argjson count "$count" '
        select(.kind == "portfolio_http_trial" and .candidate == $candidate and .phase == $phase and .pair < $count)
        | {label: "pair-\(.pair)-\(.arm)", arm, http_seconds: (.row.http_ms / 1000), metrics: .row.metrics}
        | if $phase == "confirmation" then .label |= sub("^pair-"; "confirm-") else . end
    ' "$raw/portfolio-http.jsonl" > "$work/pairs.jsonl"
    options=()
    if [[ "$phase" == confirmation ]]; then options+=(--confirmation); fi
    "$stats" "$work/pairs.jsonl" "${options[@]}" > "$work/stats.json"
    jq -en --argjson recorded "$summary" --slurpfile replay "$work/stats.json" '
        def close($a; $b): (($a - $b) | fabs) < 1e-10;
        def distribution($a; $b):
            all(["min","max","median","mad_over_median"][]; . as $k | close($a[$k]; $b[$k]));
        $recorded.comparison as $a | $replay[0] as $b |
        $a.pairs == $b.pairs and $a.bootstrap_seed == $b.bootstrap_seed and
        $a.bootstrap_resamples == $b.bootstrap_resamples and close($a.gain; $b.gain) and
        all(range(0; 2); . as $i | close($a.bootstrap_gain_ci95[$i]; $b.bootstrap_gain_ci95[$i])) and
        distribution($a.control; $b.control_tps) and distribution($a.candidate; $b.candidate_tps)
    ' > /dev/null
done
cat "$work/summary.json"
