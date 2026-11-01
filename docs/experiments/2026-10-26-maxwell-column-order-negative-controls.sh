#!/usr/bin/env bash
# A valid screen passes; damaged controls, owner tags and timing must fail.
set -euo pipefail
prefix=docs/experiments/2026-10-26-maxwell-column-order
capture=docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
audit=(jq -s --arg source 034be328bbfbb766c58ee82e1128a3c2cf545218
  --arg elf 74331e5dfa4fe5572dac27a99b80acb0cca4e2f0ca83ede2a0b99e156a365363
  --arg pid 3357843 --slurpfile capture "$capture" -f "$prefix-audit.jq")
"${audit[@]}" "$prefix-screen.jsonl" > "$temporary/valid.json"
mutations=(
  'map(select(.kind!="ffn_read_pair" or .trial!=0))'
  'map(if .kind=="ffn_read_validation" then .has_verified_dispatch_owner_tags=false else . end)'
  'map(if .kind=="ffn_read_pair" then .measurements|=map(if .stage=="ordered_read_transpose32" then .has_cta_index_tag=false else . end) else . end)'
  'map(if .kind=="ffn_read_pair" then .measurements|=map(if .stage=="ordered_read_sequential" then .operation="ffn_q8_weight_checksum_128" else . end) else . end)'
  'map(if .kind=="ffn_read_pair" then .measurements[0].enqueue_counters.device_allocs=1 else . end)'
  'map(if .kind=="ffn_read_manifest" then .prepared_weights.tensors[0].key.source.tensor_name="wrong" else . end)'
  'map(if .kind=="ffn_read_validation" then .checks.has_matching_cpu_checksums=false else . end)'
  'map(if .kind=="ffn_read_manifest" then .worktree="dirty" else . end)'
  'map(if .kind=="ffn_read_order_screen" then .comparisons[0].result.passes_micro_screen=true else . end)'
  'map(if .kind=="ffn_read_pair" then .measurements[1].started_unix_us=0 else . end)'
  'map(if .kind=="ffn_read_complete" then .device_bytes_after=16 else . end)'
  'map(if .kind=="ffn_read_pair" then .gpu_before.processes+="\n1, other-model, 100 MiB" else . end)'
)
for index in "${!mutations[@]}"; do
  jq -s "${mutations[index]}|.[]" "$prefix-screen.jsonl" > "$temporary/mutated.jsonl"
  if "${audit[@]}" "$temporary/mutated.jsonl" > "$temporary/result.json" 2> "$temporary/error.log"; then
    printf 'Corruption %s incorrectly accepted\n' "$index" >&2
    exit 1
  fi
  printf 'Corruption %s rejected\n' "$index"
done
