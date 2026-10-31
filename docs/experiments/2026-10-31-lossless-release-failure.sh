#!/usr/bin/env bash
# Run real-input release correctness before admitting any timed comparison.
set -euo pipefail
cd /tmp/m40-research-1030/lossless-kv
release_dir=/tmp/lossless-release-20261031-01
mkdir "$release_dir"
[[ -z $(git status --porcelain) ]]
git rev-parse HEAD > "$release_dir/revision.txt"
while IFS= read -r release_env_name; do
    unset "$release_env_name"
done < <(compgen -e | rg '^(M40LLM_|CUDA_|OMP_NUM_THREADS$)')
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/lossless-kv/target
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
export M40LLM_ENABLE_NVCC=1
export M40LLM_ENABLE_CUBLAS=1
release_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z ${release_processes//[[:space:]]/} ]]
cargo build --release --features cuda --locked --example lossless_attention_bench -j 4 \
    > "$release_dir/build.log" 2>&1
sha256sum target/release/examples/lossless_attention_bench > "$release_dir/executable.sha256"
target/release/examples/lossless_attention_bench \
    --capture-report target/lossless-kv-capture-01.jsonl \
    --kv-inputs target/lossless-kv-inputs-01.bin \
    --benchmark --correctness-only --output "$release_dir/correctness.jsonl" \
    > "$release_dir/correctness.log" 2>&1
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,memory.used,pstate,temperature.gpu --format=csv,noheader
release_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z ${release_processes//[[:space:]]/} ]]
printf '%s\n' 'Release correctness complete. Review full-grid admission before timings.'
