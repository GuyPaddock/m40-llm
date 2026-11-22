#!/usr/bin/env bash
# Run one independent B2 phase in a new raw directory, with the dev environment sourced.
set -euo pipefail
phase=$1
model=$(realpath "$2")
out=$(realpath -m "$3")
mkdir "$out"
for key in $(compgen -e | rg '^M40LLM_' || true); do
    if [[ "$key" != M40LLM_SYSROOT ]]; then unset "$key"; fi
done
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
export M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1
if [[ "$phase" == operators ]]; then
    bin=target/release/examples/kv_attention_bench
else
    bin=target/residency/release/examples/kv_quality
    while read -r setting; do export "$setting"; done < <(sed -nE 's/^ +\("(M40LLM_[A-Z0-9_]+)", "([^"]+)"\),$/\1=\2/p' examples/kv_quality/config.rs)
fi
snapshot() {
    date --iso-8601=ns
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv
    local owners
    owners=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)
    [[ -z "$owners" ]]
}
[[ -z $(git status --porcelain) ]]
{
    git rev-parse HEAD
    git status --porcelain
    sha256sum "$bin" "$model"
    cuobjdump -lelf "$bin"
    nvcc --version
    env | rg '^(CUDA_|M40LLM_)' | sort
    snapshot
} > "$out/preflight.txt"
bash ../consolidation/docs/experiments/2026-11-21-protocol/host-load.sh "$$" > "$out/host-load.log" &
sampler=$!
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=timestamp,pid,used_gpu_memory --format=csv --loop-ms=500 > "$out/processes.csv" &
meter=$!
trap 'kill "$sampler" "$meter" 2>/dev/null || true; wait "$sampler" "$meter" 2>/dev/null || true' EXIT
case "$phase" in
  residency)
    for capacity in 1024 3072 8448; do
        for arm in dense top64 top8; do
            snapshot > "$out/$capacity-$arm.before"
            "$bin" residency --arm "$arm" --capacity "$capacity" --model "$model" --output "$out/$capacity-$arm.jsonl" > "$out/$capacity-$arm.log" 2>&1
            snapshot > "$out/$capacity-$arm.after"
        done
    done ;;
  operators)
    input=../../m40-llm/local-artifacts/qualified-references/research-1030-lossless/kv-inputs-01.bin
    captured=../raw/october-lossless-capture.jsonl
    sha256sum "$captured" "$input" > "$out/inputs.sha256"
    "$bin" --capture-report "$captured" --kv-inputs "$input" --benchmark --iterations 10 --output "$out/report.jsonl" > "$out/run.log" 2>&1 ;;
  october)
    sha256sum ../raw/october-quality-inputs.json > "$out/inputs.sha256"
    "$bin" run --model "$model" --inputs ../raw/october-quality-inputs.json --output "$out/report.jsonl" > "$out/run.log" 2>&1 ;;
  *) exit 2 ;;
esac
snapshot > "$out/after.txt"
