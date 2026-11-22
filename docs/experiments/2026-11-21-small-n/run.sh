#!/usr/bin/env bash
# From a clean small-N worktree, with scripts/dev-env.sh sourced.
set -euo pipefail
model=$(realpath "$1")
out=$(realpath -m "$2")
mkdir "$out"
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
export M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1
bin=target/release/examples/q8_execution_bench
guards=(target/release/deps/q8_separate_projection-*)
guard_binary=""
for candidate in "${guards[@]}"; do
    if [[ -x "$candidate" && -f "$candidate" ]]; then
        [[ -z "$guard_binary" ]]
        guard_binary=$candidate
    fi
done
[[ -n "$guard_binary" && -z $(git status --porcelain) ]]
snapshot() {
    date --iso-8601=ns
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
}
idle() {
    local owners
    owners=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)
    [[ -z "$owners" ]]
}
idle
{
    git rev-parse HEAD
    git status --porcelain
    sha256sum "$bin" "$guard_binary" "$model"
    cuobjdump -lelf "$bin"
    nvcc --version
    rustc --version
    env | rg '^(CUDA_|M40LLM_)' | sort
    snapshot
} > "$out/preflight.txt"
cuobjdump -res-usage "$bin" | rg -A1 'q8_small_rows_kernel|q8_separate_projection_kernel' > "$out/resources.txt"
bash ../consolidation/docs/experiments/2026-11-21-protocol/host-load.sh "$$" > "$out/host-load.log" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true' EXIT
"$guard_binary" test_small_row_weight_reuse_preserves_serial_bits_and_guards --exact --include-ignored --test-threads=1 --nocapture > "$out/guard.log" 2>&1
rg -q '1 passed; 0 failed' "$out/guard.log"
idle
snapshot > "$out/before-timing.txt"
"$bin" --small-rows --model "$model" --output "$out/report.jsonl" --layout separate \
    --tensors blk.0.ffn_down.weight,blk.17.ffn_down.weight,blk.35.ffn_down.weight \
    --rows 1,2,3,4,8,16 --trials 5 --iterations 10 > "$out/bench.log" 2>&1
jq -es '([.[]|select(.kind=="small_rows_timing")]|length)==540 and ([.[]|select(.kind=="small_rows_parity")]|length)==108' "$out/report.jsonl"
idle
snapshot > "$out/after.txt"
