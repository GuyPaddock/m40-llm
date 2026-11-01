#!/usr/bin/env bash
set -euo pipefail
cd /tmp/m40-selector-followup
while IFS= read -r capture_key; do
    unset "$capture_key"
done < <(compgen -e | rg '^(M40LLM_|CUDA_)')
source scripts/dev-env.sh
while IFS=$'\t' read -r capture_key capture_value; do
    export "$capture_key=$capture_value"
done < <(jq -r 'select(.kind=="ffn_capture_manifest") | .execution_configuration.environment | to_entries[] | [.key,.value] | @tsv' docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl)
[[ -z $(git status --porcelain) ]]
cargo build --release --features cuda --locked --example model_quality > /tmp/m40-selector-capture-build.log 2>&1
[[ -z $(git status --porcelain) ]]
capture_gpu_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z ${capture_gpu_processes//[[:space:]]/} ]]
artifact_directory=/mnt/array-fastest/home/guyep/playground/2025-12-20-m40-llm/m40-llm/local-artifacts/qualified-references/selector-followup
mkdir -p "$artifact_directory"
original_model="$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
[[ ! -e "$artifact_directory/capture-01.jsonl" && ! -e "$artifact_directory/ffn-inputs-01.json" ]]
target/release/examples/model_quality --model "$original_model" --reference-model "$original_model" --variant-id selector-complete-history --split calibration --context-tokens 1024 --capture-ffn-inputs "$artifact_directory/ffn-inputs-01.json" --capture-ffn-complete-generations --output "$artifact_directory/capture-01.jsonl" > /tmp/m40-selector-capture-run.log 2>&1
jq -c 'select(.kind=="ffn_capture_task") | {id,generated:.captured.metrics.generated_tokens,stop:.captured.metrics.stop_reason,captured_rows_per_layer,has_identical_generation}' "$artifact_directory/capture-01.jsonl"
jq -c 'select(.kind=="ffn_capture_complete")' "$artifact_directory/capture-01.jsonl"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,memory.used --format=csv,noheader
