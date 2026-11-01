#!/usr/bin/env bash
# One frozen transfer-only follow-up after preserving the short-interval cohort.
set -euo pipefail
cd /tmp/m40-research-1030/dualpace
pricing_dir=/tmp/dualpace-pricing-20261031-03
mkdir "$pricing_dir"
[[ -z $(git status --porcelain) ]]
git rev-parse HEAD > "$pricing_dir/revision.txt"
cp /tmp/dualpace-pricing-20261031-02/cpu-node0.jsonl "$pricing_dir/cpu-node0.jsonl"
cp /tmp/dualpace-pricing-20261031-02/cpu-node1.jsonl "$pricing_dir/cpu-node1.jsonl"
sha256sum "$pricing_dir"/cpu-node*.jsonl > "$pricing_dir/retained-cpu.sha256"
while IFS= read -r pricing_env; do
    unset "$pricing_env"
done < <(compgen -e | rg '^(M40LLM_|CUDA_|OMP_NUM_THREADS$)')
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/dualpace/target
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
export M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1
cargo build --release --features cuda --locked --example cooperative_transfer_probe -j 4 > "$pricing_dir/transfer-build.log" 2>&1
sha256sum target/release/examples/cooperative_transfer_probe > "$pricing_dir/transfer-executable.sha256"
for pricing_node in 0 1; do
    if [[ "$pricing_node" == 0 ]]; then pricing_cpu=15; else pricing_cpu=30; fi
    pricing_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits)
    [[ -z ${pricing_processes//[[:space:]]/} ]]
    ps -eLo pid,tid,psr,pcpu,comm --sort=-pcpu > "$pricing_dir/node${pricing_node}-load.txt"
    LC_ALL=C mpstat -P "$pricing_cpu,$((pricing_cpu + 32))" 1 3 > "$pricing_dir/node${pricing_node}-mpstat.txt"
    awk '$1 == "Average:" && $2 ~ /^[0-9]+$/ && $NF < 90 { bad = 1 } END { exit bad }' "$pricing_dir/node${pricing_node}-mpstat.txt"
    numactl --physcpubind="$pricing_cpu" --membind="$pricing_node" target/release/examples/cooperative_transfer_probe \
        --cpu "$pricing_cpu" --numa-node "$pricing_node" --validate-only \
        --output "$pricing_dir/transfers-node${pricing_node}-validation.jsonl" > "$pricing_dir/transfers-node${pricing_node}-validation.log" 2>&1
    numactl --physcpubind="$pricing_cpu" --membind="$pricing_node" target/release/examples/cooperative_transfer_probe \
        --cpu "$pricing_cpu" --numa-node "$pricing_node" --iterations 100 \
        --output "$pricing_dir/transfers-node${pricing_node}.jsonl" > "$pricing_dir/transfers-node${pricing_node}.log" 2>&1
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,memory.used,pstate,temperature.gpu --format=csv,noheader
    pricing_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits)
    [[ -z ${pricing_processes//[[:space:]]/} ]]
    printf 'Follow-up transfer node %s complete\n' "$pricing_node"
done
[[ -z $(git status --porcelain) ]]
printf '%s\n' 'Bounded transfer follow-up complete.'
