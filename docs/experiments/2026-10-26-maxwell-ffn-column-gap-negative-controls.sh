#!/usr/bin/env bash
# Run from the repository root. Every corrupted report must fail the audit.
set -euo pipefail
prefix=docs/experiments/2026-10-26-maxwell-ffn-column-gap
capture=docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
mutations=(
  'map(select(.kind!="ffn_read_pair" or .trial!=0))'
  'map(if .kind=="ffn_read_validation" then .column_gap_preparation.padding_bytes=0 else . end)'
  'map(if .kind=="ffn_read_pair" then .measurements|=map(if .stage=="pitched_read_column_gap" then .column_gap_bytes=0 else . end) else . end)'
  'map(if .kind=="ffn_read_validation" then .column_gap_preparation.keys[0].source.tensor_name="wrong" else . end)'
  'map(if .kind=="ffn_read_validation" then .checks.has_matching_cpu_checksums=false else . end)'
  'map(if .kind=="ffn_read_pair" then .measurements[0].enqueue_counters.device_allocs=1 else . end)'
  'map(if .kind=="ffn_read_gap_screen" then .result.throughput_gain=1 else . end)'
  'map(if .kind=="ffn_read_manifest" then .prepared_weights.tensors[0].key.source.tensor_name="wrong" else . end)'
  'map(if .kind=="ffn_read_complete" then .device_bytes_after=16 else . end)'
  'map(if .kind=="ffn_read_pair" then .gpu_before.processes+="\n1, other-model, 100 MiB" else . end)'
)
for index in "${!mutations[@]}"; do
  jq -s "${mutations[index]}|.[]" "$prefix-screen.jsonl" > "$temporary/mutated.jsonl"
  if jq -s --arg source 2e96ce79a7287bd07a675ce419dcf8644dd0e17d \
      --arg elf df5b2756625d3a270539091fed416f0a77d3e54a3636ea0709b0824d55f9b72e \
      --arg pid 3161242 --slurpfile capture "$capture" \
      -f "$prefix-audit.jq" "$temporary/mutated.jsonl" \
      > "$temporary/result.json" 2> "$temporary/error.log"; then
    printf 'Corruption control %s was incorrectly accepted\n' "$index" >&2
    exit 1
  fi
  printf 'Corruption control %s rejected\n' "$index"
done
