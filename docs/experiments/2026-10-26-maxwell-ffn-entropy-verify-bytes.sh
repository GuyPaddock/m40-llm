#!/usr/bin/env bash
# Independent byte/count audit. Explicit model path; no CUDA or inference.
set -euo pipefail
model=${1:?provide the original Q8 GGUF path}
report=${2:-docs/experiments/2026-10-26-maxwell-ffn-entropy-study.jsonl}
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
expected=$(head -n 1 "$report" | jq -r '.model_sha256')
test "$(sha256sum "$model" | cut -d' ' -f1)" = "$expected"
jq -r 'select(.kind=="ffn_payload_entropy_tensor")|[.source.name,.absolute_file_offset,.statistics.source_bytes,.source.source_sha256]|@tsv' \
  "$report" > "$temporary/tensors.tsv"
test "$(wc -l < "$temporary/tensors.tsv")" -eq 6
while IFS=$'\t' read -r name offset bytes digest; do
  dd if="$model" of="$temporary/source.q8" bs=1M iflag=skip_bytes,count_bytes \
    skip="$offset" count="$bytes" status=none
  test "$(sha256sum "$temporary/source.q8" | cut -d' ' -f1)" = "$digest"
  jq -r --arg name "$name" 'select(.kind=="ffn_payload_entropy_tensor" and .source.name==$name)|.statistics.huffman_code_lengths[]' \
    "$report" > "$temporary/lengths"
  od -An -v -tu1 "$temporary/source.q8" | awk '
    BEGIN {size[1]=32; size[2]=256; size[3]=1024}
    FNR==NR {code[FNR-1]=$1;next}
    {
      for(i=1;i<=NF;i++) {
        if(position%34>=2) {
          histogram[$i]++; total_bits+=code[$i]
          for(j=1;j<=3;j++) {
            bits[j]+=code[$i]; used[j]++
            if(used[j]==size[j]) {
              aligned[j]+=int((bits[j]+127)/128)*16; chunks[j]++
              bits[j]=0; used[j]=0
            }
          }
        }
        position++
      }
    }
    END {
      for(j=1;j<=3;j++) if(used[j]>0) {
        aligned[j]+=int((bits[j]+127)/128)*16; chunks[j]++
      }
      printf "{\"histogram\":["
      for(i=0;i<256;i++) printf "%s%.0f",(i ? "," : ""),histogram[i]+0
      printf "],\"huffman_payload_bits\":%.0f,\"aligned_payload_bytes\":[",total_bits
      for(j=1;j<=3;j++) printf "%s%.0f",(j>1 ? "," : ""),aligned[j]
      printf "],\"chunk_counts\":["
      for(j=1;j<=3;j++) printf "%s%.0f",(j>1 ? "," : ""),chunks[j]
      print "]}"
    }' "$temporary/lengths" - > "$temporary/counts.json"
  jq -e --arg name "$name" --slurpfile independent "$temporary/counts.json" '
    select(.kind=="ffn_payload_entropy_tensor" and .source.name==$name)|
    (.statistics.payload_histogram==$independent[0].histogram and
     .statistics.huffman_payload_bits==$independent[0].huffman_payload_bits and
     [.statistics.indexed_size_models[].aligned_huffman_payload_bytes]==$independent[0].aligned_payload_bytes and
     [.statistics.indexed_size_models[].chunks]==$independent[0].chunk_counts)
  ' "$report" > "$temporary/accepted"
  printf '{"tensor":"%s","source_sha256":"%s","independent_counts_match":true}\n' "$name" "$digest"
done < "$temporary/tensors.tsv"
