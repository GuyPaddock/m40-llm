#!/usr/bin/env bash
set -euo pipefail
cd /tmp/m40-research-1030/dualpace
scenario_pricing=/tmp/dualpace-pricing-20261031-03
scenario_destination=target/dualpace-simulations-20261031-01
[[ -z $(git status --porcelain) ]]
mkdir "$scenario_destination"
gzip -dc docs/experiments/2026-10-31-dualpace-gpu-costs.json.gz > "$scenario_destination/gpu-costs.json"
sha256sum target/release/examples/cooperative_decode > "$scenario_destination/executable.sha256"
for scenario_node in 0 1; do
    bash docs/experiments/2026-10-31-dualpace-build-scenarios.sh \
        "$scenario_pricing/cpu-node${scenario_node}.jsonl" \
        "$scenario_pricing/transfers-node${scenario_node}.jsonl" \
        "$scenario_destination/gpu-costs.json" \
        docs/experiments/2026-10-31-dualpace-capacity.json \
        "$scenario_destination/node${scenario_node}"
    mkdir "$scenario_destination/node${scenario_node}/results" "$scenario_destination/node${scenario_node}/summaries"
    while IFS= read -r scenario_name; do
        target/release/examples/cooperative_decode \
            --trace "$scenario_destination/node${scenario_node}/${scenario_name}.json" \
            --output "$scenario_destination/node${scenario_node}/results/${scenario_name}.jsonl"
        jq -s -f docs/experiments/2026-10-31-dualpace-summarize.jq \
            "$scenario_destination/node${scenario_node}/results/${scenario_name}.jsonl" \
            > "$scenario_destination/node${scenario_node}/summaries/${scenario_name}.json"
        printf '%s\n' "$scenario_name"
    done < <(jq -r '.scenarios[].name' "$scenario_destination/node${scenario_node}/scenarios.json")
done
jq -s 'if length != 108 or any(.is_simulation != true or .is_serving_measurement != false or (.modes|length)!=3) then error("incomplete simulation summaries") else . end' \
    "$scenario_destination"/node*/summaries/*.json > "$scenario_destination/complete-summaries.json"
printf '%s\n' 'All 108 sourced simulations and 324 mode summaries complete.'
