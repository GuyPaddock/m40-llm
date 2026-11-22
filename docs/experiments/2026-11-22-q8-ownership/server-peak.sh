#!/usr/bin/env bash
# Run under the shared M40 lock with a fresh output directory.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../2026-11-21-serving-baseline/server.sh"
ref="$consolidation/docs/experiments/2026-10-27-compound-reference.json"
quality="$consolidation/docs/experiments/2026-10-27-maxwell-compound-runtime-quality.jsonl"
model_sha=$(jq -er '.model.sha256' "$ref")
start_study "${1:?new result directory}" "$HOME/.ollama/models/blobs/sha256-$model_sha" "$model_sha"
printf '%s  %s\n' a6462eb0b64326c47eb465a9bed257700794438f0a32b01296f2f454a6554036 "$phase4_bin" | sha256sum -c -
jq -e '.cases[] | select(.name=="short") | .request' "$manifest" > "$out/request.json"
jq -e --slurpfile req "$out/request.json" '.configuration.browser_prompt==$req[0].prompt and $req[0].max_tokens==512' "$ref" >/dev/null
jq -sce 'map(select(.kind=="q8_runtime_browser_generation" and .label=="candidate")) | first.generation.generated_token_ids|select(length==512)' "$quality" > "$out/expected.json"
while IFS=$'\t' read -r key value; do export "$key=$value"; done < <(jq -r '.configuration.environment|to_entries[]|[.key,.value]|@tsv' "$ref")
export M40LLM_Q8_EXECUTION_WEIGHTS=split M40LLM_Q8_EXECUTION_FAMILIES=gate-up,down,vocabulary
export M40LLM_Q8_EXECUTION_BUDGET_MB=6144 M40LLM_Q8_ATTN_PROJECTIONS=packed-warp-256
export M40LLM_Q8_FFN_KERNEL=joint-shared-scales-128 M40LLM_DECODE_LOG=1
for arm in retain-split joint-only; do
    export M40LLM_Q8_FFN_OWNERSHIP="$arm"
    start_server "$phase4_bin" "$arm" --kv-compress-mode off --max-context-tokens 1024
    call_clients "$arm-cold" 0 "$out/request.json"
    call_clients "$arm-warm" 0 "$out/request.json"
    if [[ "$arm" == retain-split ]]; then owner=RetainSplit; else owner=JointOnly; fi
    rg -q "ownership=$owner" "$out/$arm.log"
    rg -q 'packed prefix prefill enabled layers=36' "$out/$arm.log"
    rg -o 'sampled token id=[0-9]+' "$out/$arm.log" | cut -d= -f2 |
        jq -es --slurpfile reference "$out/expected.json" '.==($reference[0]+$reference[0])' >/dev/null
    gpu_snapshot > "$out/$arm.end"
    stop_server
    check_gpu_owners
done
gpu_snapshot > "$out/complete.txt"
