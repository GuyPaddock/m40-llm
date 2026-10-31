#!/usr/bin/env bash
# Qualify pending lossless consumers only after the existing capture retires.
set -euo pipefail
qualification_dir=/tmp/lossless-qualification-20261031-01
cd /tmp/m40-research-1030/lossless-kv
capture_pid=318282
capture_report=target/lossless-kv-capture-01.jsonl
capture_artifact=target/lossless-kv-inputs-01.bin
m40_uuid=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
printf '%s\n' 'Waiting for existing capture PID 318282; no GPU work starts during capture.'
while kill -0 "$capture_pid" 2>/dev/null; do
    sleep 30
done
printf '%s\n' 'Capture process retired; checking complete evidence and GPU occupancy.'
sha256sum --check "$qualification_dir/source.sha256"
jq -s -f "$qualification_dir/capture-summary.jq" "$capture_report" > "$qualification_dir/capture-summary.json"
jq -e '
    .counts.lossless_kv_stream == 96 and .counts.lossless_kv_compression == 384
    and .counts.lossless_kv_readback == 24 and .counts.lossless_kv_sequence == 8
    and .complete.artifact_bytes == 201326592
    and .complete.tracked_device_bytes_before == .complete.tracked_device_bytes_after
' "$qualification_dir/capture-summary.json" > /dev/null
expected_capture_digest=$(jq -er '.complete.artifact_sha256' "$qualification_dir/capture-summary.json")
printf '%s  %s\n' "$expected_capture_digest" "$capture_artifact" | sha256sum --check
[[ $(stat -c %s "$capture_artifact") == 201326592 ]]

require_idle_m40() {
    local active_processes
    active_processes=$(nvidia-smi -i "$m40_uuid" --query-compute-apps=pid --format=csv,noheader,nounits)
    if [[ -n ${active_processes//[[:space:]]/} ]]; then
        printf 'GPU busy; refusing qualification: %s\n' "$active_processes" >&2
        return 1
    fi
    nvidia-smi -i "$m40_uuid" --query-gpu=uuid,memory.used,pstate,temperature.gpu --format=csv,noheader
}

require_idle_m40
while IFS= read -r qualification_env_name; do
    unset "$qualification_env_name"
done < <(compgen -e | rg '^(M40LLM_|CUDA_|OMP_NUM_THREADS$)')
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/lossless-kv/target
export CUDA_VISIBLE_DEVICES="$m40_uuid"
export M40LLM_ENABLE_CUBLAS=1
export M40LLM_ENABLE_NVCC=1

# The CPU reader checks every stream identity and full-file digest before
# reconstructing all twelve real layer/context fixtures.
cargo run --release --no-default-features --locked --example lossless_attention_bench -- \
    --capture-report "$capture_report" --kv-inputs "$capture_artifact" \
    --output "$qualification_dir/fixture-inspection.jsonl" \
    > "$qualification_dir/fixture-inspection.log" 2>&1
printf '%s\n' 'Complete capture and CPU fixture inspection passed.'
require_idle_m40
sha256sum --check "$qualification_dir/source.sha256"
cargo test --features cuda --locked --test lossless_kv_cuda -- \
    --include-ignored --test-threads=1 --nocapture \
    > "$qualification_dir/kernel-tests.log" 2>&1
printf '%s\n' 'Lossless kernel tests completed successfully.'
require_idle_m40
cargo test --features cuda --locked --example lossless_attention_bench -- \
    --include-ignored --test-threads=1 --nocapture \
    > "$qualification_dir/runner-tests.log" 2>&1
printf '%s\n' 'Lossless runner tests completed successfully.'
require_idle_m40
printf '%s\n' 'Qualification queue complete. Review logs before commits and real-input timing.'
