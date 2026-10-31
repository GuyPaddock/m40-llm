#!/usr/bin/env bash
set -euo pipefail
cd /tmp/m40-research-1030/lossless-kv
regression_dir=/tmp/lossless-regressions-20261031-01
mkdir "$regression_dir"
while IFS= read -r regression_env_name; do
    unset "$regression_env_name"
done < <(compgen -e | rg '^(M40LLM_|CUDA_|OMP_NUM_THREADS$)')
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/lossless-kv/target
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
export M40LLM_ENABLE_NVCC=1

require_idle_m40() {
    local regression_processes
    regression_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits)
    [[ -z ${regression_processes//[[:space:]]/} ]] || { printf 'GPU busy: %s\n' "$regression_processes" >&2; return 1; }
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,memory.used,pstate,temperature.gpu --format=csv,noheader
}

require_idle_m40
sha256sum --check /tmp/lossless-qualification-20261031-01/source.sha256
export M40LLM_ENABLE_CUBLAS=1
cargo test --features cuda --locked \
    --test attention_parity_cuda_grid --test attention_batched_varlen \
    --test head128_attention_parallel --test cuda_transfer_order \
    --test weight_allocation_accounting --test proj_wrappers --test mlp_wrappers \
    --test forward_with_layer_smoke --test forward_parity_toy --test forward_minimal \
    --test gemm_fallback --test gemm_mixed \
    -- --include-ignored --test-threads=1 --nocapture > "$regression_dir/cublas-on.log" 2>&1
printf '%s\n' 'cuBLAS-enabled regression command completed.'
require_idle_m40
export M40LLM_ENABLE_CUBLAS=0
cargo test --features cuda --locked \
    --test lossless_kv_cuda --test proj_wrappers --test mlp_wrappers \
    --test forward_parity_toy --test forward_minimal --test gemm_fallback --test gemm_mixed \
    --example lossless_attention_bench \
    -- --include-ignored --test-threads=1 --nocapture > "$regression_dir/cublas-off.log" 2>&1
printf '%s\n' 'cuBLAS-disabled regression command completed.'
require_idle_m40
sha256sum --check /tmp/lossless-qualification-20261031-01/source.sha256
printf '%s\n' 'Serial regression commands complete; inspect logs for skipped execution.'
