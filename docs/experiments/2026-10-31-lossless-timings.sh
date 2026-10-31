#!/usr/bin/env bash
set -euo pipefail
cd /tmp/m40-research-1030/lossless-kv
lossless_timing_dir=/tmp/lossless-timing-20261031-01
mkdir "$lossless_timing_dir"
[[ -z $(git status --porcelain) ]]
git rev-parse HEAD > "$lossless_timing_dir/revision.txt"
while IFS= read -r lossless_timing_env; do
    unset "$lossless_timing_env"
done < <(compgen -e | rg '^(M40LLM_|CUDA_|OMP_NUM_THREADS$)')
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/lossless-kv/target
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
export M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1
lossless_timing_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z ${lossless_timing_processes//[[:space:]]/} ]]
jq -s -f docs/experiments/2026-10-31-lossless-release-admission.jq /tmp/lossless-release-20261031-02/correctness.jsonl > "$lossless_timing_dir/admission.json"
cargo build --release --features cuda --locked --example lossless_attention_bench -j 4 > "$lossless_timing_dir/build.log" 2>&1
sha256sum target/release/examples/lossless_attention_bench > "$lossless_timing_dir/executable.sha256"
target/release/examples/lossless_attention_bench \
    --capture-report target/lossless-kv-capture-01.jsonl \
    --kv-inputs target/lossless-kv-inputs-01.bin \
    --benchmark --iterations 10 --output "$lossless_timing_dir/timings.jsonl" \
    > "$lossless_timing_dir/timings.log" 2>&1
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,memory.used,pstate,temperature.gpu --format=csv,noheader
lossless_timing_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z ${lossless_timing_processes//[[:space:]]/} ]]
printf '%s\n' 'Lossless paired timing grid complete.'
