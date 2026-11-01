#!/usr/bin/env bash
# Run the frozen quality grid once, preserving partial output on failure.
set -euo pipefail

if [[ $# != 2 ]]; then
    echo "usage: $0 MODEL_PATH NEW_OUTPUT_DIRECTORY" >&2
    exit 2
fi
model_path=$(realpath "$1")
output_directory=$(realpath -m "$2")
cd "$(dirname "$0")/.."
if [[ -e "$output_directory" ]]; then
    echo "refusing to overwrite $output_directory" >&2
    exit 2
fi

# Freeze process-local settings without changing the caller's environment.
while IFS= read -r setting; do
    unset "$setting"
done < <(compgen -v M40LLM_)
source scripts/dev-env.sh
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
export M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1
export M40LLM_PROJECTION_BACKEND=large-model M40LLM_MATERIALIZE_F32_WEIGHTS=0
export M40LLM_Q8_EXECUTION_WEIGHTS=off M40LLM_DECODE_GRAPH=0
export M40LLM_Q8_FFN_KERNEL=native M40LLM_Q8_PREFILL_BACKEND=dequant-sgemm
export M40LLM_ATTN_HEAD128_KERNEL=native M40LLM_Q8_DECODE_STREAM=decode
export M40LLM_FUSED_Q8_QKV=1 M40LLM_FUSED_Q8_MLP_SWIGLU=1
export M40LLM_Q8_DECODE_COL_TILE=4 M40LLM_Q8_DECODE_SPLIT_QBLOCK=4
export M40LLM_Q8_MLP_GATE_UP_SPLIT_QBLOCK=4
export M40LLM_CUDA_GREEDY_ARGMAX=1 M40LLM_Q8_LM_HEAD_ARGMAX=1
export M40LLM_Q8_LM_HEAD_ARGMAX_SHARED_A=1

mkdir -p "$output_directory"
cargo build --release --features cuda --locked --example kv_quality \
    > "$output_directory/build.log" 2>&1
git rev-parse HEAD > "$output_directory/source.txt"
git status --porcelain > "$output_directory/source-status.txt"
if [[ -s "$output_directory/source-status.txt" ]]; then
    echo "commit source before inference" >&2
    exit 2
fi
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
    --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv \
    > "$output_directory/gpu-before.csv"
if [[ $(wc -l < "$output_directory/gpu-before.csv") != 1 ]]; then
    echo "M40 has a compute process; refusing concurrent inference" >&2
    exit 2
fi
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" > "$output_directory/device.txt"
sha256sum target/release/examples/kv_quality > "$output_directory/executable.sha256"
target/release/examples/kv_quality run --model "$model_path" \
    --inputs docs/experiments/2026-10-31-kv-quality-inputs.json \
    --output "$output_directory/results.jsonl" > "$output_directory/run.log" 2>&1
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
    --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv \
    > "$output_directory/gpu-after.csv"
