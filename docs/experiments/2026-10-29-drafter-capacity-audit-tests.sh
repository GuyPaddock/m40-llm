#!/usr/bin/env bash
# Accept retained source/ledger agreement; reject fabricated admission or counts.
set -euo pipefail
artifacts="${1:-local-artifacts/2026-10-29-drafter-data}"
report=docs/experiments/2026-10-29-drafter-capacity.json

run_audit() {
  jq -e \
    --slurpfile spec docs/experiments/2026-10-29-drafter-data-spec.json \
    --slurpfile train "$artifacts/sources/mbpp/train.json" \
    --slurpfile validation "$artifacts/sources/mbpp/validation.json" \
    --slurpfile test "$artifacts/sources/mbpp/test.json" \
    --slurpfile ledger "$artifacts/row-ledger.json" \
    -f docs/experiments/2026-10-29-drafter-capacity-audit.jq "$@"
}

run_audit "$report" >/dev/null
for mutation in \
  '.has_source_capacity = true' \
  '.is_r1_admitted = true' \
  '.partitions[0].row_counts[0].prompt_tokens += 1' \
  '.partitions[0].length_eligible_upper_bound = 256' \
  '.partitions[1].required_quota = 0'; do
  if jq "$mutation" "$report" | run_audit >/dev/null 2>&1; then
    printf 'Audit incorrectly accepted mutation: %s\n' "$mutation" >&2
    exit 1
  fi
done
printf '{"original_accepted":true,"corruptions_rejected":5}\n'
