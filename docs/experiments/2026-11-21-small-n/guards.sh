#!/usr/bin/env bash
# Run under the common M40 lock from the small-N worktree.
set -euo pipefail
out=$(realpath -m "${1:?new result directory}")
off=$(realpath "${2:?release cuBLAS-disabled test binary}")
on=$(realpath "${3:?release cuBLAS-enabled test binary}")
mkdir "$out"
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
[[ -z $(git status --porcelain) ]]
snapshot() {
    date --iso-8601=ns
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv
    local active
    active=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader)
    echo "$active"
    [[ -z "$active" ]]
}
{
    git rev-parse HEAD
    git status --porcelain
    rustc --version
    nvcc --version
    sha256sum "$off" "$on" tests/q8_separate_projection.rs
    env | rg '^(CUDA_|M40LLM_)' | sort
    snapshot
} > "$out/preflight.txt"
for mode in 0 1; do
    binary="$off"
    if [[ "$mode" == 1 ]]; then binary="$on"; fi
    readelf -d "$binary" > "$out/link-$mode.txt"
    if rg -q 'NEEDED.*libcublas\.so' "$out/link-$mode.txt"; then linked=1; else linked=0; fi
    [[ "$linked" == "$mode" ]]
    cuobjdump -lelf "$binary" | rg 'sm_52' > "$out/image-$mode.txt"
    snapshot > "$out/before-$mode.txt"
    M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS="$mode" "$binary" --include-ignored --test-threads=1 --nocapture > "$out/suite-$mode.log" 2>&1
    rg -q 'test result: ok. 3 passed; 0 failed; 0 ignored;' "$out/suite-$mode.log"
    snapshot > "$out/after-$mode.txt"
done
date --iso-8601=ns > "$out/complete.txt"
