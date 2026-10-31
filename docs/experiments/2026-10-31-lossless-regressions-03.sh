#!/usr/bin/env bash
# Complete the remaining cuBLAS-independent consumer and wrapper coverage.
set -euo pipefail
cd /tmp/m40-research-1030/lossless-kv
while IFS= read -r regression_env_name; do
    unset "$regression_env_name"
done < <(compgen -e | rg '^(M40LLM_|CUDA_|OMP_NUM_THREADS$)')
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/lossless-kv/target
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
export M40LLM_ENABLE_NVCC=1
export M40LLM_ENABLE_CUBLAS=0
regression_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z ${regression_processes//[[:space:]]/} ]]
cargo test --features cuda --locked \
    --test lossless_kv_cuda --test proj_wrappers --test mlp_wrappers \
    --example lossless_attention_bench \
    -- --include-ignored --test-threads=1 --nocapture > /tmp/lossless-regressions-20261031-03.log 2>&1
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,memory.used,pstate,temperature.gpu --format=csv,noheader
regression_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z ${regression_processes//[[:space:]]/} ]]
sha256sum --check /tmp/lossless-qualification-20261031-01/source.sha256
printf '%s\n' 'Remaining cuBLAS-disabled coverage completed.'
