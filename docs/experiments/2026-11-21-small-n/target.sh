#!/usr/bin/env bash
# From this research worktree, with scripts/dev-env.sh sourced and the M40 lock held.
set -euo pipefail
out=$(realpath -m "${1:?fresh output directory}")
model=$(realpath "${2:?frozen Qwen Q8_0 file}")
bin=$(realpath "${3:?release executable from the bound build}")
build=$(realpath "${4:?build manifest with source SHA then binary SHA}")
reference=docs/experiments/2026-10-27-compound-reference.json
source_sha=$(sed -n '1p' "$build")
binary_sha=$(sed -n '2p' "$build")
[[ "$source_sha" =~ ^[a-f0-9]{40}$ && "$binary_sha" =~ ^[a-f0-9]{64}$ ]]
git diff --quiet "$source_sha" HEAD -- src cuda examples build.rs Cargo.toml Cargo.lock
[[ -z $(git status --porcelain) ]]
mkdir "$out"
mkdir "$out/correctness"
for key in $(compgen -e | rg '^M40LLM_' || true); do unset "$key"; done
while IFS= read -r setting; do export "$setting"; done < <(
    jq -r '.configuration.environment|to_entries[]|"\(.key)=\(.value)"' "$reference")
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
    cat "$build"
    printf '%s  %s\n' "$binary_sha" "$bin" "$(jq -r '.model.sha256' "$reference")" "$model" | sha256sum -c -
    sha256sum "$reference" "$model" "$bin"
    nvcc --version
    cuobjdump -lelf "$bin" | rg 'sm_52'
    env | rg '^(M40LLM_|CUDA_)' | sort
    snapshot
} > "$out/preflight.txt"
bash ../consolidation/docs/experiments/2026-11-21-protocol/host-load.sh "$$" > "$out/host-load.log" &
sampler=$!
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=timestamp,pid,used_gpu_memory --format=csv --loop-ms=250 > "$out/processes.csv" &
meter=$!
trap 'kill "$sampler" "$meter" 2>/dev/null || true; wait "$sampler" "$meter" 2>/dev/null || true' EXIT
args=(--model "$model" --direct-four-row-verifier --rows 4 --trials 5 --iterations 10)
for arm in serial multi-row shared-weights; do
    snapshot > "$out/$arm.before"
    "$bin" "${args[@]}" --direct-verifier-correctness-only --verifier-down "$arm" \
        --output "$out/correctness/$arm.jsonl" > "$out/$arm.log" 2>&1
    jq -es 'last.kind=="direct_verifier_complete" and last.is_correctness_passed' "$out/correctness/$arm.jsonl" >/dev/null
    snapshot > "$out/$arm.after"
done
snapshot > "$out/measurement.before"
"$bin" "${args[@]}" --compare-verifier-down "$out/correctness" --output "$out/measurement.jsonl" > "$out/measurement.log" 2>&1
jq -es 'last.kind=="direct_verifier_complete" and last.device_bytes_after==0 and
    (map(select(.kind=="down_target_gate"))|length)==1 and (map(select(.kind=="down_target_rejection"))|length)==60' "$out/measurement.jsonl" >/dev/null
snapshot > "$out/complete.txt"
