#!/usr/bin/env bash
set -euo pipefail
cd /tmp/m40-research-1030/quantized-kv
quantized_timing_dir=/tmp/quantized-kv-timing-20261031-01
mkdir "$quantized_timing_dir"
[[ -z $(git status --porcelain) ]]
git rev-parse HEAD > "$quantized_timing_dir/revision.txt"
while IFS= read -r quantized_timing_env; do
    unset "$quantized_timing_env"
done < <(compgen -e | rg '^(M40LLM_|CUDA_|OMP_NUM_THREADS$)')
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/quantized-kv/target
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
export M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1
quantized_timing_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z ${quantized_timing_processes//[[:space:]]/} ]]
jq -s -f docs/experiments/2026-10-31-quantized-kv-release-admission.jq /tmp/quantized-kv-release-20261031-03/correctness.jsonl > "$quantized_timing_dir/admission.json"
cargo build --release --features cuda --locked --example kv_attention_bench -j 4 > "$quantized_timing_dir/build.log" 2>&1
sha256sum target/release/examples/kv_attention_bench > "$quantized_timing_dir/executable.sha256"
target/release/examples/kv_attention_bench \
    --capture-report /tmp/m40-research-1030/lossless-kv/target/lossless-kv-capture-01.jsonl \
    --kv-inputs /mnt/array-fastest/home/guyep/playground/2025-12-20-m40-llm/m40-llm/local-artifacts/qualified-references/research-1030-lossless/kv-inputs-01.bin \
    --benchmark --iterations 10 --output "$quantized_timing_dir/timings.jsonl" \
    > "$quantized_timing_dir/timings.log" 2>&1
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,memory.used,pstate,temperature.gpu --format=csv,noheader
quantized_timing_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z ${quantized_timing_processes//[[:space:]]/} ]]
printf '%s\n' 'Quantized KV paired timing grid complete.'
