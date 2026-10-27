#!/usr/bin/env bash
# Corrupted benchmark observations must not pass the frozen attribution audit.
set -euo pipefail
cd /mnt/array-fastest/home/guyep/playground/2025-12-20-m40-llm/m40-llm
out=/tmp/m40-compound-profile
audit() {
  jq -s --slurpfile quality docs/experiments/2026-10-27-maxwell-compound-runtime-quality.jsonl \
    --slurpfile events "$1" -f "$out/audit.jq" "$2"
}
audit "$out/events.jsonl" "$out/http.jsonl" > /dev/null
for mutation in dispatch bytes cache ids clocks pairs summary; do
  case "$mutation" in
    dispatch) expression='if .kind=="trial" then .launch_deltas.ffn_q8_joint_f32x4_shared_scales_128=0 else . end' ;;
    bytes) expression='if .kind=="trial" then .joint_ffn_pairs.resident_bytes=0 else . end' ;;
    cache) expression='if .kind=="trial" then .cache.reused_prompt_tokens=1 else . end' ;;
    ids) expression='if .kind=="token_trace" then .generated_token_ids[1]+=1 else . end' ;;
    clocks) expression='if .kind=="trial" then .gpu_after.gpus_csv="GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852, Tesla M40 24GB, 580.159.04, P0, 78, 80, 999, 3004, 11000, 90" else . end' ;;
    pairs) expression='select(.kind!="trial" or .phase!="profile-on" or .pair!=0)' ;;
    summary) expression='if .kind=="profile_comparison_summary" then .profile_on.median*=0.95 else . end' ;;
  esac
  jq -c "$expression" "$out/http.jsonl" > "$out/negative-http.jsonl"
  if audit "$out/events.jsonl" "$out/negative-http.jsonl" > /dev/null 2>&1; then
    printf 'ERROR accepted HTTP %s\n' "$mutation" >&2
    exit 1
  fi
  printf 'Rejected HTTP %s\n' "$mutation"
done
for mutation in spans layer position pool stream timing; do
  case "$mutation" in
    spans) expression='.stages|=.[1:]' ;;
    layer) expression='.stages|=map(if .stage=="attention" and .layer_id==0 then .layer_id=1 else . end)' ;;
    position) expression='.absolute_position+=1' ;;
    pool) expression='.event_pool_reused=false' ;;
    stream) expression='.stages[0].stream="prefill"' ;;
    timing) expression='.stages[0].gpu_elapsed_ms=-1' ;;
  esac
  jq -c "$expression" "$out/events.jsonl" > "$out/negative-events.jsonl"
  if audit "$out/negative-events.jsonl" "$out/http.jsonl" > /dev/null 2>&1; then
    printf 'ERROR accepted events %s\n' "$mutation" >&2
    exit 1
  fi
  printf 'Rejected events %s\n' "$mutation"
done
