#!/usr/bin/env bash
# Invoke only after the lossless timing job is terminal and M40 is idle.
set -euo pipefail
cd /tmp/m40-research-1030/quantized-kv
quantized_qualification_dir=/tmp/quantized-kv-qualification-20261031-01
mkdir "$quantized_qualification_dir"
while IFS= read -r quantized_qualification_env; do
    unset "$quantized_qualification_env"
done < <(compgen -e | rg '^(M40LLM_|CUDA_|OMP_NUM_THREADS$)')
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/quantized-kv/target
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
export M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1
quantized_qualification_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z ${quantized_qualification_processes//[[:space:]]/} ]]
sha256sum cuda/kernels.cu src/cuda.rs tests/quantized_kv_staging.rs examples/kv_attention_bench/{main,pipeline,benchmark,validation}.rs > "$quantized_qualification_dir/source.sha256"
cargo fmt --all -- --check
cargo test --release --features cuda --locked --test quantized_kv_staging --example kv_attention_bench -- --include-ignored --test-threads=1 --nocapture > "$quantized_qualification_dir/initial.log" 2>&1
cargo test --release --features cuda --locked --test attention_parity_cuda_grid --test attention_batched_varlen --test cuda_transfer_order --test forward_with_layer_smoke --test forward_minimal --test forward_parity_toy --test mlp_wrappers --test proj_wrappers --test weight_allocation_accounting -- --include-ignored --test-threads=1 --nocapture > "$quantized_qualification_dir/affected.log" 2>&1
M40LLM_ENABLE_CUBLAS=0 cargo test --release --features cuda --locked --test quantized_kv_staging --example kv_attention_bench --test mlp_wrappers --test proj_wrappers --test forward_minimal --test forward_parity_toy -- --include-ignored --test-threads=1 --nocapture > "$quantized_qualification_dir/cublas-off.log" 2>&1
sha256sum -c "$quantized_qualification_dir/source.sha256"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,memory.used,pstate,temperature.gpu --format=csv,noheader
quantized_qualification_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z ${quantized_qualification_processes//[[:space:]]/} ]]
