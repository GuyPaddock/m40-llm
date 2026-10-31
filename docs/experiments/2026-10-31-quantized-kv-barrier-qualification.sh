#!/usr/bin/env bash
set -euo pipefail
cd /tmp/m40-research-1030/quantized-kv
while IFS= read -r barrier_env; do
    unset "$barrier_env"
done < <(compgen -e | rg '^(M40LLM_|CUDA_|OMP_NUM_THREADS$)')
source scripts/dev-env.sh
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
export M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1
export KV_CAPTURE_REPORT=/tmp/m40-research-1030/lossless-kv/target/lossless-kv-capture-01.jsonl
export KV_INPUTS=/mnt/array-fastest/home/guyep/playground/2025-12-20-m40-llm/m40-llm/local-artifacts/qualified-references/research-1030-lossless/kv-inputs-01.bin
barrier_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z ${barrier_processes//[[:space:]]/} ]]
sha256sum -c /tmp/quantized-kv-barrier-source.sha256
cargo test --release --features cuda --locked --test attention_parity_cuda_grid --test attention_batched_varlen --test cuda_transfer_order --test forward_with_layer_smoke --test forward_minimal --test forward_parity_toy --test mlp_wrappers --test proj_wrappers --test weight_allocation_accounting -- --include-ignored --test-threads=1 --nocapture > /tmp/quantized-kv-barrier-affected.log 2>&1
M40LLM_ENABLE_CUBLAS=0 cargo test --release --features cuda --locked --test quantized_kv_staging --example kv_attention_bench -- --include-ignored --test-threads=1 --nocapture > /tmp/quantized-kv-barrier-off.log 2>&1
cargo clippy --features cuda,server --all-targets -- -D warnings > /tmp/quantized-kv-barrier-clippy.log 2>&1
cargo fmt --all -- --check
sha256sum -c /tmp/quantized-kv-barrier-source.sha256
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,memory.used,pstate,temperature.gpu --format=csv,noheader
barrier_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z ${barrier_processes//[[:space:]]/} ]]
