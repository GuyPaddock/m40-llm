#!/usr/bin/env bash
# Verify corrupt CPU reports cannot support the cost conclusion.
set -euo pipefail
prefix=docs/experiments/2026-10-26-maxwell-disabled-diagnostics
capture=docs/experiments/2026-10-26-maxwell-packed-stage-events.jsonl
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
mutations=(
  'map(select(.kind!="disabled_diagnostics_pair" or .trial!=0))'
  'map(if .kind=="disabled_diagnostics_manifest" then .worktree="dirty" else . end)'
  'map(if .kind=="disabled_diagnostics_manifest" then .cuda_visible_devices="0" else . end)'
  'map(if .kind=="disabled_diagnostics_manifest" then .operations_per_nominal_token=1 else . end)'
  'map(if .kind=="disabled_diagnostics_pair" then .arms[0].checksum=0 else . end)'
  'map(if .kind=="disabled_diagnostics_pair" then .arms[0].elapsed_ms_per_replay=1 else . end)'
  'map(if .kind=="disabled_diagnostics_complete" then .device_bytes_after=16 else . end)'
  'map(if .kind=="disabled_diagnostics_complete" then .counters_after={wrong:{launches:1}} else . end)'
  'map(if .kind=="disabled_diagnostics_pair" then .arms[1].started_unix_us=.arms[0].started_unix_us else . end)'
  'map(if .kind=="disabled_diagnostics_manifest" then .affinity="affinity list: 0-71" else . end)'
)
for index in "${!mutations[@]}"; do
  jq -s "${mutations[index]}|.[]" "$prefix-cost.jsonl" > "$temporary/mutated.jsonl"
  if jq -s --slurpfile capture "$capture" -f "$prefix-audit.jq" \
      "$temporary/mutated.jsonl" > "$temporary/result.json" 2> "$temporary/error.log"; then
    printf 'Corruption control %s was incorrectly accepted\n' "$index" >&2
    exit 1
  fi
  printf 'Corruption control %s rejected\n' "$index"
done
