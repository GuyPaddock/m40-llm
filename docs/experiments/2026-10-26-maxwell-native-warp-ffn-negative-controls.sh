#!/usr/bin/env bash
# Run from the repository root. Every corrupted report must fail its audit.
set -euo pipefail
prefix=docs/experiments/2026-10-26-maxwell-native-warp-ffn
capture=docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
mutations=(
  'map(select(.kind!="ffn_ablation_pair" or .trial!=0))'
  'map(if .kind=="ffn_ablation_pair" then .measurements[0].stages[0].enqueue_counters.device_allocs=1 else . end)'
  'map(if .kind=="ffn_ablation_pair" then .measurements|=map(if .variant=="split-f32x4-native-warp-128" then .stages[0].native_warp_dispatch.partials_per_output=128 else . end) else . end)'
  'map(if .kind=="ffn_ablation_validation" then .drift_vs_timing_control.swiglu.max_abs_diff=1 else . end)'
  'map(if .kind=="ffn_ablation_validation" then .checks.intermediate_capture_matches_normal_kernel=false else . end)'
  'map(if .kind=="ffn_ablation_manifest" then .prepared_weights.tensors[0].key.source.tensor_name="wrong" else . end)'
  'map(if .kind=="ffn_ablation_screen" then .result.control_median_ms=1 else . end)'
  'map(if .kind=="ffn_ablation_pair" then .tracked_device_bytes=0 else . end)'
  'map(if .kind=="ffn_ablation_complete" then .device_bytes_after=16 else . end)'
  'map(if .kind=="ffn_ablation_pair" then .gpu_before.processes+="\n1, other-model, 100 MiB" else . end)'
)
for index in "${!mutations[@]}"; do
  jq -s "${mutations[index]}|.[]" "$prefix-screen.jsonl" > "$temporary/mutated.jsonl"
  if jq -s --arg source dd78a1a2f6b37f7c78f9b727bef40c099d85b87e \
      --arg elf c769e6507f368e5b1b8117a417bd71d3e1086fbf16c73978b9e4410a4baf68ec \
      --arg pid 3192070 --slurpfile capture "$capture" \
      -f "$prefix-audit.jq" "$temporary/mutated.jsonl" \
      > "$temporary/result.json" 2> "$temporary/error.log"; then
    printf 'Corruption control %s was incorrectly accepted\n' "$index" >&2
    exit 1
  fi
  printf 'Corruption control %s rejected\n' "$index"
done
