#!/usr/bin/env bash
# A valid four-arm report must pass; each corrupted report must fail.
set -euo pipefail
prefix=docs/experiments/2026-10-27-maxwell-compound-ffn
phase=${1:-confirm}
raw="$prefix-$phase.jsonl"
source=$(head -n1 "$raw" | jq -r .revision)
elf=$(head -n1 "$raw" | jq -r .executable_sha256)
trials=$(head -n1 "$raw" | jq -r .trials)
pid=$(jq -r 'select(.kind=="ffn_compound_trial")|.gpu_before.processes|split(",")[0]' "$raw" | head -n1)
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
audit() {
  jq -s --arg source "$source" --arg elf "$elf" --arg pid "$pid" --argjson trials "$trials" \
    --slurpfile capture docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
    -f "$prefix-audit.jq" "$1"
}
audit "$raw" > "$temporary/valid.json"
mutations=(
  'map(select(.kind!="ffn_compound_trial" or .trial!=0))'
  'map(if .kind=="ffn_compound_trial" then .measurements[0].stages[0].enqueue_counters.device_allocs=1 else . end)'
  'map(if .kind=="ffn_compound_validation" then .has_control_bits=false else . end)'
  'map(if .kind=="ffn_compound_validation" then .checks.intermediate_capture_matches_normal_kernel=false else . end)'
  'map(if .kind=="ffn_compound_manifest" then .prepared_weights.tensors[0].key.source.tensor_name="wrong" else . end)'
  'map(if .kind=="ffn_compound_trial" then .order=[0,1,2,3] else . end)'
  'map(if .kind=="ffn_compound_trial" then .tracked_device_bytes=0 else . end)'
  'map(if .kind=="ffn_ablation_complete" then .device_bytes_after=16 else . end)'
  'map(if .kind=="ffn_compound_trial" then .gpu_before.processes+="\n1, other-model, 100 MiB" else . end)'
  'map(if .kind=="ffn_compound_summary" then .compound_vs_joint.native_median_ms=1 else . end)'
  'map(if .kind=="ffn_compound_summary" then .compound_vs_split.passes_micro_screen=false else . end)'
  'map(if .kind=="ffn_compound_trial" then .measurements|=map(if .variant=="joint-f32x4-shared-scales-128" then .stages[0].shared_scales_dispatch.block_barriers=1 else . end) else . end)'
)
for index in "${!mutations[@]}"; do
  jq -s "${mutations[index]}|.[]" "$raw" > "$temporary/mutated.jsonl"
  if audit "$temporary/mutated.jsonl" > "$temporary/result.json" 2> "$temporary/error.log"; then
    printf 'Corruption control %s incorrectly accepted\n' "$index" >&2
    exit 1
  fi
  printf 'Corruption control %s rejected\n' "$index"
done
