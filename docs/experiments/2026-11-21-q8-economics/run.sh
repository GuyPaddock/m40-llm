#!/usr/bin/env bash
# Source scripts/dev-env.sh, then use the clean Q8 worktree and a new output path.
set -euo pipefail
model=$(realpath "$1")
out=$(realpath -m "$2")
section=${3:-all}
case "$section" in all|portfolio) ;; *) exit 2 ;; esac
mkdir "$out"
ref=docs/experiments/2026-10-27-compound-reference.json
bin=target/release/examples/joint_only
for key in $(compgen -e | rg '^M40LLM_' || true); do unset "$key"; done
while IFS=$'\t' read -r key value; do
    export "$key=$value"
done < <(jq -r '.configuration.environment|to_entries[]|[.key,.value]|@tsv' "$ref")
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
    sha256sum "$bin" "$model" "$ref"
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
run_phase() {
    local name=$1 phase=$2
    shift 2
    snapshot > "$out/$name.before"
    "$bin" --model "$model" --phase "$phase" --output "$out/$name.jsonl" "$@" > "$out/$name.log" 2>&1
    snapshot > "$out/$name.after"
}
profile_status=0
if [[ "$section" == all ]]; then
    run_phase load-retain load --ownership retain-split
    run_phase load-joint load --ownership joint-only
    run_phase quality quality
    run_phase http http --quality-report "$out/quality.jsonl"
    # Profiling has its own perturbation gate. Keep the portfolio's independent
    # quality prerequisite and end-to-end timings reachable after a failure.
    (set -e; run_phase dataflow dataflow --ownership joint-only --quality-report "$out/quality.jsonl" --profile-report "$out/stages.jsonl") &
    phase_pid=$!
    wait "$phase_pid" || profile_status=$?
    printf '%d\n' "$profile_status" > "$out/dataflow.exit"
fi
run_phase portfolio-quality portfolio-quality
run_phase portfolio-http portfolio-http --quality-report "$out/portfolio-quality.jsonl"
exit "$profile_status"
