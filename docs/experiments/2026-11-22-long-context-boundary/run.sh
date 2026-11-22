#!/usr/bin/env bash
# Run in this worktree while holding the shared M40 lock.
set -euo pipefail
source scripts/dev-env.sh
out=$(realpath -m "${1:?fresh boundary measurement directory}")
[[ -z $(git status --porcelain) ]]
mkdir "$out"
export CUDA_VISIBLE_DEVICES='' M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 CARGO_BUILD_JOBS=3
export CARGO_TARGET_DIR="$PWD/target/boundary"
nice -n 19 ionice -c 3 cargo test --release --locked --features cuda --test head128_attention_parallel \
    --no-run --message-format=json > "$out/artifacts.jsonl" 2> "$out/build.log"
exe=$(jq -ser '[.[]|select(.reason=="compiler-artifact" and .target.name=="head128_attention_parallel"
    and .profile.test and .executable!=null)|.executable] |
    if length==1 then first else error("expected one test executable") end' "$out/artifacts.jsonl")
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
snapshot() {
    date --iso-8601=ns
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv
    local other
    other=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader)
    [[ -z "$other" ]]
}
{
    git rev-parse HEAD
    git status --porcelain
    sha256sum "$exe"
    nvcc --version
    cuobjdump -lelf "$exe"
    env | rg '^(M40LLM_|CUDA_)' | sort
    snapshot
} > "$out/preflight.txt"
rg -q 'sm_52' "$out/preflight.txt"
bash ../consolidation/docs/experiments/2026-11-21-protocol/host-load.sh "$$" > "$out/host-load.log" &
host=$!
trap 'kill "$host" 2>/dev/null || true; wait "$host" 2>/dev/null || true' EXIT
status=0
timeout --signal=TERM 900s "$exe" benchmark_dense_head128_context_boundary --exact \
    --include-ignored --nocapture --test-threads=1 > "$out/test.log" 2>&1 || status=$?
printf '%d\n' "$status" > "$out/test.exit"
snapshot > "$out/after.txt"
(( status == 0 ))
rg 'test result: ok. 1 passed; 0 failed; 0 ignored' "$out/test.log"
rg 'head128 fast-warp-reduce kernel' "$out/test.log"
rg 'generic fallback kernel' "$out/test.log"
sed -nE 's/^[^{]*(\{.*\})$/\1/p' "$out/test.log" > "$out/observations.jsonl"
jq -es 'map(select(.kind=="boundary_trial")|[.pair,.cache_length]) ==
    [[0,8192],[0,8193],[1,8193],[1,8192],[2,8192],[2,8193]]' "$out/observations.jsonl" >/dev/null
date --iso-8601=ns > "$out/complete.txt"
