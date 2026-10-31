#!/usr/bin/env bash
set -euo pipefail
cd /tmp/m40-research-1030/lossless-kv
while IFS= read -r rounding_env_name; do
    unset "$rounding_env_name"
done < <(compgen -e | rg '^(M40LLM_|CUDA_|OMP_NUM_THREADS$)')
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/lossless-kv/target
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
export M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1
export LOSSLESS_CAPTURE_REPORT=target/lossless-kv-capture-01.jsonl LOSSLESS_KV_INPUTS=target/lossless-kv-inputs-01.bin
rounding_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z ${rounding_processes//[[:space:]]/} ]]
cargo fmt --all -- --check
cargo test --release --features cuda --locked --test attention_parity_cuda_grid --test cuda_transfer_order --test proj_wrappers --test mlp_wrappers --test forward_with_layer_smoke --test forward_minimal --test forward_parity_toy --test weight_allocation_accounting -- --test-threads=1 --nocapture > /tmp/lossless-rounding-affected.log 2>&1
M40LLM_ENABLE_CUBLAS=0 cargo test --release --features cuda --locked --test lossless_kv_cuda --example lossless_attention_bench -- --include-ignored --test-threads=1 --nocapture > /tmp/lossless-rounding-off.log 2>&1
cargo clippy --features cuda,server --all-targets -- -D warnings > /tmp/lossless-rounding-clippy.log 2>&1
RUSTFLAGS='-D warnings' cargo test --no-default-features --locked --example lossless_attention_bench > /tmp/lossless-rounding-cpu.log 2>&1
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,memory.used,pstate,temperature.gpu --format=csv,noheader
rounding_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z ${rounding_processes//[[:space:]]/} ]]
