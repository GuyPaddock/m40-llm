#!/usr/bin/env bash
set -euo pipefail
cd /mnt/array-fastest/home/guyep/playground/2025-12-20-m40-llm/m40-llm
out=/tmp/m40-compound-runtime
revision=63240c52b64e06870922c8b6c0cbae8b0fe2e6f4
quality_audit() {
  jq -s --arg revision "$revision" \
    --slurpfile prior docs/experiments/2026-10-25-maxwell-packed-projection-runtime-quality.jsonl \
    -f "$out/quality-audit.jq" "$1"
}
http_audit() {
  jq -s --arg mode "$2" --arg revision "$revision" \
    --slurpfile quality "$out/quality.jsonl" -f "$out/http-audit.jq" "$1"
}
quality_audit "$out/quality.jsonl" > /dev/null
for mutation in ids logits source bytes dispatch; do
  case "$mutation" in
    ids) expression='if .kind=="q8_runtime_browser_generation" and .label=="candidate" then .generation.generated_token_ids[1]+=1 else . end' ;;
    logits) expression='if .kind=="q8_runtime_prefix_logits" then .max_abs_logit_diff=0.1 else . end' ;;
    source) expression='if .kind=="q8_runtime_manifest" then .joint_candidate.pairs[0].gate.tensor_name="wrong" else . end' ;;
    bytes) expression='if .kind=="q8_runtime_manifest" then .joint_candidate.resident_bytes=0 else . end' ;;
    dispatch) expression='if .kind=="q8_runtime_browser_generation" and .label=="candidate" then .dispatch.operations.ffn_q8_joint_f32x4_shared_scales_128.launches=0 else . end' ;;
  esac
  jq -c "$expression" "$out/quality.jsonl" > "$out/negative-quality.jsonl"
  if quality_audit "$out/negative-quality.jsonl" > /dev/null 2>&1; then
    printf 'ERROR accepted quality %s\n' "$mutation" >&2
    exit 1
  fi
  printf 'Rejected quality %s\n' "$mutation"
done
for mode in incremental ollama; do
  file="$out/http.jsonl"
  if [[ "$mode" == ollama ]]; then file="$out/ollama-http.jsonl"; fi
  http_audit "$file" "$mode" > /dev/null
  for mutation in output bytes dispatch pair clocks cache threshold interval; do
    case "$mutation" in
      output) expression='if .kind=="trial" and .runtime=="m40-llm" then .output="wrong" else . end' ;;
      bytes) expression='if .kind=="trial" and .runtime=="m40-llm" then .joint_ffn_pairs.resident_bytes=0 else . end' ;;
      dispatch) expression='if .kind=="trial" and .runtime=="m40-llm" then .launch_deltas.ffn_q8_joint_f32x4_shared_scales_128=0 else . end' ;;
      pair) expression='if .kind=="trial" then .pair=0 else . end' ;;
      clocks) expression='if .kind=="trial" then .gpu_after.gpus_csv="GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852, Tesla M40 24GB, 580.159.04, P0, 78, 80, 999, 3004, 17000, 90" else . end' ;;
      cache) expression='if .kind=="trial" then .cache.reused_prompt_tokens=1 else . end' ;;
      threshold) expression='if .comparison then .comparison.promotion_threshold=0.02 else . end' ;;
      interval) expression='if .comparison then .comparison.bootstrap_gain_ci95[0]=-1 else . end' ;;
    esac
    jq -c "$expression" "$file" > "$out/negative-http.jsonl"
    if http_audit "$out/negative-http.jsonl" "$mode" > /dev/null 2>&1; then
      printf 'ERROR accepted %s %s\n' "$mode" "$mutation" >&2
      exit 1
    fi
    printf 'Rejected %s %s\n' "$mode" "$mutation"
  done
done
