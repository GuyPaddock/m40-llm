#!/usr/bin/env bash
# Read retained CPU evidence only. No downloads, model execution or GPU work.
set -euo pipefail
cd "$(dirname "$0")/../.."
root=local-artifacts/2026-10-30-drafter-recovery
original=local-artifacts/2026-10-29-drafter-data
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT

"$original/venv/bin/python" scripts/drafter_data/mbpp_remaining.py \
  --tokens "$root/tokens.jsonl" --exclusions "$root/exclusions-final" \
  --review docs/experiments/2026-10-30-mbpp-review.json \
  --output "$temporary/mbpp.json"
cmp "$temporary/mbpp.json" "$root/mbpp-final.json"

jq -n \
  --slurpfile source "$original/sources/dolly/databricks-dolly-15k.jsonl" \
  --slurpfile tokens "$root/tokens.jsonl" \
  --slurpfile dolly "$root/dolly-candidates.json" \
  --slurpfile mbpp "$temporary/mbpp.json" \
  --slurpfile provenance "$root/provenance-network/report.json" \
  --slurpfile inventory "$root/exclusions-final/inventory.json" \
  --slurpfile identity docs/experiments/2026-10-29-drafter-r1-decision.json \
  -f docs/experiments/2026-10-30-drafter-r0-audit.jq > "$temporary/readiness.json"
cmp "$temporary/readiness.json" docs/experiments/2026-10-30-drafter-r0-audit.json
printf 'CPU reports reproduce exactly; R1 remains blocked.\n'
