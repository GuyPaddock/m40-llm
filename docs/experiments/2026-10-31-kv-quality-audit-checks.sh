#!/usr/bin/env bash
# Completed evidence must pass; corrupted identities, scores, paths and counts
# must fail the same complete-grid audit. No model inference is executed here.
set -euo pipefail
records=docs/experiments/2026-10-31-kv-quality-results.jsonl
inputs=docs/experiments/2026-10-31-kv-quality-inputs.json
audit=docs/experiments/2026-10-31-kv-quality-audit.jq
temporary=$(mktemp -d /tmp/m40-kv-audit-checks.XXXXXX)
trap 'rm -rf "$temporary"' EXIT
check_report() {
    jq -s -e --slurpfile inputs "$inputs" -f "$audit" "$1"
}
check_report "$records" > /dev/null
names=(completion identity arm count cache score target dispatch overlap control repetition configuration)
mutations=(
  '.[0:-1]'
  'map(if .kind=="kv_quality_manifest" then .source_revision="wrong" else . end)'
  'map(if .kind=="kv_quality_generation" and .case=="held-chat-tour-short" and .arm=="quantized_all" then .arm="dense" else . end)'
  'map(if .kind=="kv_quality_generation" then .metrics.generated_tokens+=1 else . end)'
  'map(if .kind=="kv_quality_generation" then .cache.physical_lengths[0]+=1 else . end)'
  'map(if .kind=="kv_quality_generation" then .score.passed=(.score.passed|not) else . end)'
  'map(if .kind=="kv_quality_comparison" then .prefix_observations[0].target+=1 else . end)'
  'map(if .kind=="kv_quality_generation" then .dispatch.totals.cublas_calls=0 else . end)'
  'map(if .kind=="kv_quality_comparison" then .prefix_observations[0].logits.top10_overlap=11 else . end)'
  'map(if .kind=="kv_quality_comparison" then .no_old_control=(.no_old_control|not) else . end)'
  'map(if .kind=="kv_quality_generation" then .repetition.suffix_pattern_copies=999 else . end)'
  'map(if .kind=="kv_quality_manifest" then .quantized_top8_top_blocks=64 else . end)'
)
for index in "${!mutations[@]}"; do
    jq -sc "${mutations[index]} | .[]" "$records" > "$temporary/changed.jsonl"
    if check_report "$temporary/changed.jsonl" > /dev/null 2> "$temporary/error"; then
        echo "corruption accepted: ${names[index]}" >&2
        exit 1
    fi
    printf '%s: ' "${names[index]}" >&2
    cat "$temporary/error" >&2
done
jq -n --argjson count "${#mutations[@]}" \
    '{positive_passed:true,rejected_corruptions:$count,is_complete_grid_check:true}'
