#!/usr/bin/env bash
# Run from the repository root. Every corrupted report must fail its audit.
set -euo pipefail
prefix=docs/experiments/2026-10-26-maxwell-role-parallel-ffn
capture=docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
mutations=(
  'map(select(.kind!="ffn_ablation_pair" or .trial!=0))'
  'map(if .kind=="ffn_ablation_pair" then .measurements[0].stages[0].enqueue_counters.device_allocs=1 else . end)'
  'map(if .kind=="ffn_ablation_pair" then .measurements|=map(if .variant=="split-f32x4-role-parallel-256" then .stages[0].role_parallel_dispatch.partials_per_role=32 else . end) else . end)'
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
  if jq -s --arg source 0d99dc8748cfe0cd36ef2435ab264a620313a141 \
      --arg elf 04347df8f326233e03b172b56d4b3759e1c2373ab3bdb23458ef2658ed37e2ec \
      --arg pid 3261330 --slurpfile capture "$capture" \
      -f "$prefix-audit.jq" "$temporary/mutated.jsonl" \
      > "$temporary/result.json" 2> "$temporary/error.log"; then
    printf 'Corruption control %s was incorrectly accepted\n' "$index" >&2
    exit 1
  fi
  printf 'Corruption control %s rejected\n' "$index"
done
