#!/usr/bin/env bash
# Reject incomplete setup/cleanup records and mislabeled profiling evidence.
set -euo pipefail
prefix=docs/experiments/2026-10-26-maxwell-pc-sampling
[[ $# == 0 || ( $# == 1 && $1 == --negative-controls ) ]]
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
jq -s . "$prefix-capability.jsonl" > "$temporary/input.json"
jq -f "$prefix-audit.jq" "$temporary/input.json" > "$temporary/result.json"
if [[ $# == 0 ]]; then
  jq . "$temporary/result.json"
  exit 0
fi
mutations=(
  '.[0].sampling_period="LOW"'
  '.[0].effective_uid=0'
  'map(select(.api!="cuCtxDestroy"))'
  'map(if .api=="cuptiActivityConfigurePCSampling" then .status=35 else . end)'
  'map(if .api=="cuCtxDestroy" then .status=1 else . end)'
  'map(if .kind=="device" then .major=6 else . end)'
  '.[-1].has_setup_support=true'
  '.[-1].has_kernel_measurements=true'
  '.[-1].samples=0'
  '.[-1].exit_code=0'
)
for index in "${!mutations[@]}"; do
  jq "${mutations[index]}" "$temporary/input.json" > "$temporary/mutated.json"
  if jq -f "$prefix-audit.jq" "$temporary/mutated.json" > /dev/null 2> "$temporary/error.log"; then
    printf 'Corruption control %s was incorrectly accepted\n' "$index" >&2
    exit 1
  fi
  printf 'Corruption control %s rejected\n' "$index"
done
