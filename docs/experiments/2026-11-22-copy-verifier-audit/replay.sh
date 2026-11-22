#!/usr/bin/env bash
# Recompute copy-verifier statistics with the pinned controlled benchmark binary.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: bash replay.sh FINISHED_RAW_DIR RESEARCH_HTTP_PAIRS" >&2
    exit 2
fi
raw=$(realpath "$1")
stats=$(realpath "$2")
study=$(cd "$(dirname "$0")" && pwd)
[[ -s "$raw/complete.txt" ]]
work=$(mktemp -d)
trap 'rm -r -- "$work"' EXIT
jq -n --slurpfile correctness "$raw/correctness.jsonl" --slurpfile timing "$raw/timing.jsonl" \
    -f "$study/validate.jq" > "$work/admitted.json"

jq -c '.comparisons[]' "$work/admitted.json" | while IFS= read -r comparison; do
    id=$(jq -r .id <<< "$comparison")
    phase=$(jq -r .phase <<< "$comparison")
    count=$(jq -r .shared_vs_ordinary.pairs <<< "$comparison")
    prompt=$(jq -r --arg id "$id" '.workloads[] | select(.id == $id) | .prompt_tokens' "$work/admitted.json")
    for pair in '0 1 existing_vs_ordinary' '0 2 shared_vs_ordinary' '1 2 shared_vs_multi_row'; do
        read -r control candidate field <<< "$pair"
        # The scorer only needs a matching positive work count and elapsed seconds.
        # Its transport field names are inherited from the HTTP harness.
        jq -c --arg id "$id" --arg phase "$phase" --argjson count "$count" --argjson prompt "$prompt" \
            --argjson control "$control" --argjson candidate "$candidate" '
            select(.kind == "copy_schedule_trial" and .id == $id and .phase == $phase
                and .round < $count and (.arm == $control or .arm == $candidate)) |
            (if .arm == $control then 0 else 1 end) as $arm |
            {label: "pair-\(.round)-\($arm)", arm: $arm,
                http_seconds: (.result.total_ms / 1000),
                metrics: {generated_tokens: (.result.generated_token_ids | length),
                    prompt_tokens: $prompt, stop_reason: .result.stop_reason}}
            | if $phase == "confirmation" then .label |= sub("^pair-"; "confirm-") else . end
        ' "$raw/timing.jsonl" > "$work/pairs.jsonl"
        options=()
        if [[ "$phase" == confirmation ]]; then options+=(--confirmation); fi
        "$stats" "$work/pairs.jsonl" "${options[@]}" > "$work/scores.json"
        jq -en --arg key "$field" --argjson comparison "$comparison" --slurpfile replay "$work/scores.json" '
            def close($a; $b): (($a - $b) | fabs) < 1e-10;
            def distribution($a; $b):
                all(["min", "max", "median", "mad_over_median"][]; . as $k | close($a[$k]; $b[$k]));
            $comparison[$key] as $actual | $replay[0] as $score |
            $actual.pairs == $score.pairs and
            $actual.bootstrap_resamples == 10000 and $actual.bootstrap_seed == 0 and
            close($actual.gain; $score.gain) and
            all(range(0; 2); . as $i | close($actual.bootstrap_gain_ci95[$i]; $score.bootstrap_gain_ci95[$i])) and
            distribution($actual.control; $score.control_tps) and distribution($actual.candidate; $score.candidate_tps)
        ' >/dev/null
    done
done
jq 'del(.comparisons)' "$work/admitted.json"
