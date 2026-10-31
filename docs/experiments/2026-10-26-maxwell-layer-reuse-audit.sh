#!/usr/bin/env bash
# Reject unequal work, changed provenance and invented order-sensitivity claims.
set -euo pipefail
prefix=docs/experiments/2026-10-26-maxwell-layer-reuse
[[ $# == 0 || ( $# == 1 && $1 == --negative-controls ) ]]
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
jq -s --slurpfile capture docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  -f "$prefix-audit.jq" "$prefix-screen.jsonl" > "$temporary/result.json"
if [[ $# == 0 ]]; then
  jq . "$temporary/result.json"
  exit 0
fi
mutations=(
  'map(select(.kind!="ffn_reuse_pair" or .trial!=0))'
  'map(if .kind=="ffn_reuse_manifest" then .rows=[4] else . end)'
  'map(if .kind=="ffn_reuse_manifest" then .layer_reuse_contract.has_inter_layer_hidden_dependency=true else . end)'
  'map(if .kind=="ffn_reuse_manifest" then .prepared_weights.resident_bytes=0 else . end)'
  'map(if .kind=="ffn_reuse_pair" then .has_matching_output_bits=false else . end)'
  'map(if .kind=="ffn_reuse_pair" then .measurements[0].iterations_per_layer=9 else . end)'
  'map(if .kind=="ffn_reuse_pair" then .measurements[0].gpu_ms_per_layer*=2 else . end)'
  'map(if .kind=="ffn_reuse_pair" then .measurements[0].enqueue_counters.device_allocs=1 else . end)'
  'map(if .kind=="ffn_reuse_pair" then .measurements[0].order="wrong" else . end)'
  'map(if .kind=="ffn_reuse_case_complete" then .results[0].result.has_material_order_sensitivity=true else . end)'
  'map(if .kind=="ffn_reuse_pair" then .gpu_before.processes+="; competitor" else . end)'
  'map(if .kind=="ffn_reuse_complete" then .device_bytes_after=1 else . end)'
)
jq -s . "$prefix-screen.jsonl" > "$temporary/input.json"
for index in "${!mutations[@]}"; do
  jq "${mutations[index]}" "$temporary/input.json" > "$temporary/mutated.json"
  if jq --slurpfile capture docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
    -f "$prefix-audit.jq" "$temporary/mutated.json" > /dev/null 2> "$temporary/error.log"; then
    printf 'Corruption control %s was incorrectly accepted\n' "$index" >&2
    exit 1
  fi
  printf 'Corruption control %s rejected\n' "$index"
done
