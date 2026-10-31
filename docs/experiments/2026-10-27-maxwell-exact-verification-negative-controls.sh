#!/usr/bin/env bash
set -euo pipefail
prefix=docs/experiments/2026-10-27-maxwell-exact-verification
raw="$prefix-screen.jsonl"
pid=$(jq -r 'select(.kind=="verification_trial")|.gpu_before.processes|split(",")[0]' "$raw"|head -n1)
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
audit() {
  jq -s --arg pid "$pid" --slurpfile gold docs/experiments/2026-10-25-maxwell-packed-projection-runtime-quality.jsonl \
    -f "$prefix-audit.jq" "$1"
}
audit "$raw" > "$temporary/valid.json"
mutations=(
  'map(select(.kind!="verification_trial" or .trial!=0))'
  'map(if .kind=="verification_trial" then .result.generated_token_ids[255]+=1 else . end)'
  'map(if .kind=="verification_trial" then .result.target_input_rows=512 else . end)'
  'map(if .kind=="verification_trial" then .result.layer_lengths[0]=577 else . end)'
  'map(if .kind=="verification_trial" then .result.counters.cublas_calls=0 else . end)'
  'map(if .kind=="verification_summary" then .acceptance_curves[-1].max_drafter_ms_to_break_even=1 else . end)'
  'map(if .kind=="verification_summary" then .rejection_cycle_ms_by_accepted_prefix[0]=0 else . end)'
  'map(if .kind=="verification_rejection_cost" then .result.has_exact_correction=false else . end)'
  'map(if .kind=="verification_trial" then .gpu_before.processes+="\n1, other-model, 100 MiB" else . end)'
  'map(if .kind=="verification_complete" then .tracked_device_bytes_after=16 else . end)'
  'map(if .kind=="verification_manifest" then .prepared_weights.tensors[0].key.source.tensor_name="wrong" else . end)'
  'map(if .kind=="verification_summary" then .gain=1 else . end)'
)
for index in "${!mutations[@]}"; do
  jq -s "${mutations[index]}|.[]" "$raw" > "$temporary/mutated.jsonl"
  if audit "$temporary/mutated.jsonl" > "$temporary/result.json" 2> "$temporary/error.log"; then
    printf 'Corruption control %s incorrectly accepted\n' "$index" >&2
    exit 1
  fi
  printf 'Corruption control %s rejected\n' "$index"
done
