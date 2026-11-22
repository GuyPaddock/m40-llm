#!/usr/bin/env bash
# Run from the repository with access to the pinned historical and audit commits.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${1:?fresh replay directory}
mkdir "$out"
tar -xzf "$here/measurements.tar.gz" -C "$out"
grid=docs/experiments/2026-10-31-quantized-kv-timing-summary
git show "e23d1621bc3b8bbb5c73a531201e6ae0850fa819:$grid.jq" > "$out/grid.jq"
git show "e23d1621bc3b8bbb5c73a531201e6ae0850fa819:$grid.json" > "$out/october.json"
git show ac985bf:docs/experiments/2026-11-21-kv-audit/operator-pairs.jq > "$out/pairs.jq"
rows="$out/kv-operators-01/report.jsonl"
jq -es -f "$out/grid.jq" "$rows" > "$out/grid.json"
jq -es -f "$out/pairs.jq" "$rows" > "$out/pairs.json"
jq -e --slurpfile oct "$out/october.json" -f "$here/compare.jq" "$out/grid.json" > "$out/comparison.json"
cmp "$out/comparison.json" "$here/comparison.json"
cmp "$out/pairs.json" "$here/pairs.json"
