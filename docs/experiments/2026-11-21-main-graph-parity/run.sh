#!/usr/bin/env bash
# Run from the main-graph-parity worktree under the M40 lock.
set -euo pipefail
artifacts=$(realpath "${1:?compiled binaries}")
out=$(realpath -m "${2:?fresh destination}")
model=$(realpath "${3:?pinned TinyLlama F16 GGUF}")
mkdir "$out"
study=$(realpath docs/experiments/2026-11-21-main-graph-parity)
for variable in $(compgen -e | rg '^M40LLM_' || true); do unset "$variable"; done
export M40LLM_ENABLE_NVCC=1 M40LLM_FORCE_M40=1 M40LLM_RESEARCH_TINY_MODEL="$model"
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
snapshot() {
    date --iso-8601=ns
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv
    local owners
    owners=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader)
    echo "$owners"
    [[ -z "$owners" ]]
}
{
    git rev-parse HEAD
    git status --porcelain
    [[ -z $(git status --porcelain) ]]
    printf '%s  %s\n' 76bd2d476450b35a43d294470b091e21d81624a4f3d70d2542c07bd8fbefad93 "$model" | sha256sum -c -
    (cd "$artifacts" && sha256sum -c "$study/binaries.sha256")
    env | rg '^(M40LLM_|CUDA_)'
    snapshot
} > "$out/preflight.txt"
run_test() {
    local label=$1 exe=$2 name=$3 rc=0
    snapshot > "$out/$label.before"
    "$artifacts/$exe" --exact "$name" --include-ignored --test-threads=1 --nocapture > "$out/$label.log" 2>&1 || rc=$?
    rg -q '^running 1 test$' "$out/$label.log"
    printf '%s\t%d\n' "$label" "$rc" >> "$out/exits.tsv"
    snapshot > "$out/$label.after"
}
for build in unfixed-on fixed-on fixed-off; do
    if [[ "$build" == fixed-off ]]; then export M40LLM_ENABLE_CUBLAS=0; else export M40LLM_ENABLE_CUBLAS=1; fi
    run_test "$build-short" "$build-grid" test_graph_attention_short_dispatch
    run_test "$build-boundary" "$build-grid" test_graph_attention_dispatch_boundary
    run_test "$build-tiny" "$build-session" decode_session::main_graph_tests::tinyllama_graph_preserves_logits_and_valid_kv
done
