#!/usr/bin/env bash
set -euo pipefail
raw=docs/experiments/2026-10-29-compact-draft-cost.jsonl
audit=docs/experiments/2026-10-29-compact-draft-audit.jq
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
jq -s -f "$audit" "$raw" > /dev/null
# Reject changed observable work, sample coverage, accounting and claims.
for mutation in \
  'map(select(.kind != "compact_cost_trial" or .trial != 2))' \
  'map(if .kind == "compact_cost_trial" then .counters.device_allocs=1 else . end)' \
  'map(if .kind == "compact_cost_trial" then .median_ms=0 else . end)' \
  'map(if .kind == "compact_cost_decision" then .tracked_device_bytes_after=1 else . end)' \
  'map(if .kind == "compact_cost_decision" then .is_http_measured=true else . end)' \
  'map(if .kind == "compact_cost_setup" then .validation.is_passed=false else . end)'; do
  jq -sc "$mutation | .[]" "$raw" > "$temporary/corrupt.jsonl"
  if jq -s -f "$audit" "$temporary/corrupt.jsonl" > /dev/null 2>&1; then
    printf 'Unexpected audit acceptance: %s\n' "$mutation" >&2
    exit 1
  fi
done
printf '{"is_passed":true,"uncorrupted_acceptances":1,"corruptions_rejected":6}\n'
