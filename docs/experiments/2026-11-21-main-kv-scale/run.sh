#!/usr/bin/env bash
# Invoke under the common exclusive M40 lock.
set -euo pipefail
artifacts=$(realpath "${1:?compiled test directory}")
out=$(realpath -m "${2:?fresh output directory}")
study=$(realpath docs/experiments/2026-11-21-main-kv-scale)
gpu=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
mkdir "$out"
for variable in $(compgen -e | rg '^M40LLM_' || true); do unset "$variable"; done
export M40LLM_FORCE_M40=1 CUDA_VISIBLE_DEVICES="$gpu"
snapshot() {
    date --iso-8601=ns
    nvidia-smi -i "$gpu" --query-gpu=uuid,driver_version,clocks.sm,clocks.mem,pstate,temperature.gpu,memory.used,utilization.gpu --format=csv
    local owners
    owners=$(nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader)
    echo "$owners"
    [[ -z "$owners" ]]
}
{
    git rev-parse HEAD
    git status --porcelain
    test -z "$(git status --porcelain)"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES M40LLM_FORCE_M40=$M40LLM_FORCE_M40"
    (cd "$artifacts" && sha256sum -c "$study/binaries.sha256")
    snapshot
} > "$out/preflight.txt"
for suite in unfixed-on q8_kv_scale_reuse-on attention_parity_cuda_grid-on q8_kv_scale_reuse-off attention_parity_cuda_grid-off; do
    snapshot > "$out/$suite.before"
    rc=0
    "$artifacts/$suite" --include-ignored --nocapture --test-threads=1 > "$out/$suite.log" 2>&1 || rc=$?
    printf '%s\t%d\n' "$suite" "$rc" >> "$out/exits.tsv"
    snapshot > "$out/$suite.after"
done
