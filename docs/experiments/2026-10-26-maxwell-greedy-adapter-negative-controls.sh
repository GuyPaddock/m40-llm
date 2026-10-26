#!/usr/bin/env bash
# Valid evidence audits; each damaged provenance, timing or output record fails.
set -euo pipefail
prefix=docs/experiments/2026-10-26-maxwell-greedy-adapter
trace=docs/experiments/2026-10-25-maxwell-packed-projection-runtime-quality.jsonl
http=docs/experiments/2026-10-25-maxwell-packed-projection-ollama-http.jsonl
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
audit=(jq -s --slurpfile trace "$trace" --slurpfile http "$http" -f "$prefix-audit.jq")
"${audit[@]}" "$prefix-cost.jsonl" > "$temporary/valid.json"
mutations=(
  'map(select(.kind!="greedy_adapter_pair" or .trial!=0))'
  'map(if .kind=="greedy_adapter_manifest" then .worktree="dirty" else . end)'
  'map(if .kind=="greedy_adapter_manifest" then .cuda_visible_devices="0" else . end)'
  'map(if .kind=="greedy_adapter_manifest" then .generated_ids[0]=0 else . end)'
  'map(if .kind=="greedy_adapter_manifest" then .synthetic_vector_requested_bytes_per_trace=0 else . end)'
  'map(if .kind=="greedy_adapter_pair" then .measurements[0].checksum=0 else . end)'
  'map(if .kind=="greedy_adapter_pair" then .measurements[0].per_trace_ms=1 else . end)'
  'map(if .kind=="greedy_adapter_pair" then .measurements[1].started_unix_us=0 else . end)'
  'map(if .kind=="greedy_adapter_complete" then .tracked_device_bytes=1 else . end)'
)
for index in "${!mutations[@]}"; do
  jq -s "${mutations[index]}|.[]" "$prefix-cost.jsonl" > "$temporary/mutated.jsonl"
  if "${audit[@]}" "$temporary/mutated.jsonl" > "$temporary/result.json" 2> "$temporary/error.log"; then
    printf 'Corruption %s incorrectly accepted\n' "$index" >&2
    exit 1
  fi
  printf 'Corruption %s rejected\n' "$index"
done
