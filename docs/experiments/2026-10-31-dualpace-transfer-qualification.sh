#!/usr/bin/env bash
# Invoke only after the quantized-KV timing process exits and M40 is idle.
set -euo pipefail
cd /tmp/m40-research-1030/dualpace
while IFS= read -r transfer_qualification_env; do
    unset "$transfer_qualification_env"
done < <(compgen -e | rg '^(M40LLM_|CUDA_|OMP_NUM_THREADS$)')
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/dualpace/target
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
export M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1
transfer_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z ${transfer_processes//[[:space:]]/} ]]
sha256sum examples/cooperative_transfer_probe/{main,gpu,probe}.rs > /tmp/dualpace-transfer-qualification-20261031-01-source.sha256
cargo fmt --all -- --check
cargo test --features cuda --locked --example cooperative_transfer_probe -- --include-ignored --test-threads=1 --nocapture > /tmp/dualpace-transfer-qualification-20261031-01.log 2>&1
sha256sum -c /tmp/dualpace-transfer-qualification-20261031-01-source.sha256
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,memory.used,pstate,temperature.gpu --format=csv,noheader
transfer_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z ${transfer_processes//[[:space:]]/} ]]
