#!/usr/bin/env bash
# Both valid reports must audit; every damaged contract must be rejected.
set -euo pipefail
prefix=docs/experiments/2026-10-26-maxwell-q81-corrected
prior=docs/experiments/2026-10-25-maxwell-completed-packed128-quality.jsonl
capture=docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
mutations=(
  'map(select(.kind!="q8_runtime_loss_summary"))'
  'map(if .kind=="q8_runtime_manifest" then .prefill_execution_contract="unsafe-old" else . end)'
  'map(if .kind=="q8_runtime_browser_generation" and .label=="control" then .generation.generated_token_ids[0]=0 else . end)'
  'map(if .kind=="q8_runtime_browser_generation" and .label=="candidate" then .dispatch.operations.quantize_q8_1.launches=0 else . end)'
  'map(if .kind=="q8_runtime_browser_comparison" then .first_generated_divergence_step_zero_based=42 else . end)'
  'map(if .kind=="q8_runtime_manifest" then .prepared_candidate.tensors[0].key.source.tensor_name="wrong" else . end)'
  'map(if .kind=="q8_runtime_prefix_logits" then .expected_actual_reference_token_id=0 else . end)'
  'map(if .kind=="q8_runtime_passage" then .comparison.tokens[0].target_id=0 else . end)'
  'map(if .kind=="q8_runtime_loss_summary" then .candidate.perplexity=1 else . end)'
  'map(if .kind=="q8_runtime_summary" then .tracked_device_bytes_after=1 else . end)'
  'map(if .kind=="q8_runtime_summary" then .passes_exact_quality_gate=true else . end)'
  'map(if .kind=="model_quality_task" then .reference_score.passed=true else . end)'
)
for geometry in 128 256; do
  runtime_log=$(gzip -dc "$prefix-$geometry.log.gz")
  audit=(jq -s --arg geometry "$geometry" --slurpfile prior "$prior"
    --slurpfile capture "$capture" --arg log "$runtime_log"
    -f "$prefix-audit.jq")
  "${audit[@]}" "$prefix-$geometry.jsonl" > "$temporary/valid.json"
  for index in "${!mutations[@]}"; do
    jq -s "${mutations[index]}|.[]" "$prefix-$geometry.jsonl" > "$temporary/mutated.jsonl"
    if "${audit[@]}" "$temporary/mutated.jsonl" > "$temporary/result.json" 2> "$temporary/error.log"; then
      printf 'Geometry %s corruption %s incorrectly accepted\n' "$geometry" "$index" >&2
      exit 1
    fi
    printf 'Geometry %s corruption %s rejected\n' "$geometry" "$index"
  done
done
