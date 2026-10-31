#!/usr/bin/env bash
# Run from the repository root. Every corruption must fail the size audit.
set -euo pipefail
prefix=docs/experiments/2026-10-26-maxwell-ffn-entropy
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
mutations=(
  'map(select(.kind!="ffn_payload_entropy_tensor" or .family!="up" or .layer_id!=0))'
  'map(if .kind=="ffn_payload_entropy_manifest" then .model_sha256="wrong" else . end)'
  'map(if .kind=="ffn_payload_entropy_tensor" then .statistics.payload_histogram[0]+=1 else . end)'
  'map(if .kind=="ffn_payload_entropy_tensor" then .statistics.huffman_code_lengths[0]+=1 else . end)'
  'map(if .kind=="ffn_payload_entropy_tensor" then .statistics.scale_bytes=0 else . end)'
  'map(if .kind=="ffn_payload_entropy_tensor" then .statistics.indexed_size_models[0].index_bytes=0 else . end)'
  'map(if .kind=="ffn_payload_entropy_tensor" then .statistics.indexed_size_models[0].aligned_huffman_payload_bytes=16 else . end)'
  'map(if .kind=="ffn_payload_entropy_tensor" then .statistics.indexed_size_models[0].stored_byte_saving=0.5 else . end)'
  'map(if .kind=="ffn_payload_entropy_tensor" then .statistics.has_decoder=true else . end)'
  'map(if .kind=="ffn_payload_entropy_complete" then .size_model_eligible_at_all_tensors=[true,true,true] else . end)'
)
for index in "${!mutations[@]}"; do
  jq -s "${mutations[index]}|.[]" "$prefix-study.jsonl" > "$temporary/mutated.jsonl"
  if jq -s --slurpfile fixture target/research-fixtures/m02-ffn-inputs.json \
      --slurpfile capture docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
      -f "$prefix-audit.jq" "$temporary/mutated.jsonl" \
      > "$temporary/result.json" 2> "$temporary/error.log"; then
    printf 'Corruption control %s was incorrectly accepted\n' "$index" >&2
    exit 1
  fi
  printf 'Corruption control %s rejected\n' "$index"
done
