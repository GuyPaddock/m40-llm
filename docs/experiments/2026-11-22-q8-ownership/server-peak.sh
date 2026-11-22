#!/usr/bin/env bash
# Run under the shared M40 lock with a fresh output directory.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../2026-11-21-serving-baseline/server.sh"
cohort=${2:-compound}
case "$cohort" in
    compound)
        ref="$consolidation/docs/experiments/2026-10-27-compound-reference.json"
        quality="$consolidation/docs/experiments/2026-10-27-maxwell-compound-runtime-quality.jsonl"
        model_sha=$(jq -er '.model.sha256' "$ref")
        arms=(retain-split joint-only)
        ;;
    native-split)
        report="$consolidation/docs/experiments/2026-11-21-native-split/report.jsonl"
        model_sha=$(jq -esr 'first.model_sha256' "$report")
        arms=(native split)
        ;;
    *) exit 2 ;;
esac
start_study "${1:?new result directory}" "$HOME/.ollama/models/blobs/sha256-$model_sha" "$model_sha"
jq -e '.cases[] | select(.name=="short") | .request' "$manifest" > "$out/request.json"
if [[ "$cohort" == compound ]]; then
    jq -e --slurpfile req "$out/request.json" '.configuration.browser_prompt==$req[0].prompt and $req[0].max_tokens==512' "$ref" >/dev/null
    jq -sce 'map(select(.kind=="q8_runtime_browser_generation" and .label=="candidate")) | first.generation.generated_token_ids|select(length==512)' "$quality" > "$out/expected.json"
    # The October example supplied this backend through the model constructor.
    jq '.configuration.environment + {M40LLM_Q8_PREFILL_BACKEND:"dequant-sgemm"}' "$ref" > "$out/environment.json"
else
    jq -es --slurpfile req "$out/request.json" 'first as $m | $m.prompt==$req[0].prompt and
      $req[0].max_tokens==512 and $m.context_tokens==1024 and $m.kv_mode=="off" and
      $m.candidate_q8_config.families==["down","vocabulary"] and $m.actual_q8_attention_projection_backend=="native"' "$report" >/dev/null
    jq -esc '[.[]|select(.kind=="token_trace" and .prefill_mode=="sequential")|.generated_token_ids] |
      select(length==2 and .[0]==.[1] and (.[0]|length)==512) | first' "$report" > "$out/expected.json"
    jq -s 'first.environment' "$report" > "$out/environment.json"
fi
while IFS=$'\t' read -r key value; do export "$key=$value"; done < <(jq -r 'to_entries[]|[.key,.value]|@tsv' "$out/environment.json")
export M40LLM_Q8_EXECUTION_BUDGET_MB=6144 M40LLM_DECODE_LOG=1
for arm in "${arms[@]}"; do
    case "$arm" in
        retain-split|joint-only)
            export M40LLM_Q8_EXECUTION_WEIGHTS=split M40LLM_Q8_EXECUTION_FAMILIES=gate-up,down,vocabulary
            export M40LLM_Q8_ATTN_PROJECTIONS=packed-warp-256 M40LLM_Q8_FFN_KERNEL=joint-shared-scales-128
            export M40LLM_Q8_FFN_OWNERSHIP="$arm"
            ;;
        native) export M40LLM_Q8_EXECUTION_WEIGHTS=off ;;
        split) export M40LLM_Q8_EXECUTION_WEIGHTS=split M40LLM_Q8_EXECUTION_FAMILIES=down,vocabulary ;;
    esac
    start_server "$phase4_bin" "$arm" --kv-compress-mode off --max-context-tokens 1024
    call_clients "$arm-cold" 0 "$out/request.json"
    call_clients "$arm-warm" 0 "$out/request.json"
    if [[ "$cohort" == compound ]]; then
        if [[ "$arm" == retain-split ]]; then owner=RetainSplit; else owner=JointOnly; fi
        rg -q "ownership=$owner" "$out/$arm.log"
        rg -q 'packed prefix prefill enabled layers=36' "$out/$arm.log"
    else
        rg -q 'full-layer forward enabled layers=36' "$out/$arm.log"
        if rg -q 'packed prefix prefill enabled' "$out/$arm.log"; then exit 1; fi
        if [[ "$arm" == split ]]; then
            rg -Fq 'mode=Split families=[Down, Vocabulary] tensors=37 layouts=37 prepared_bytes=1192935424' "$out/$arm.log"
        elif rg -Fq '[q8-execution] mode=Split' "$out/$arm.log"; then
            exit 1
        fi
    fi
    rg -o 'sampled token id=[0-9]+' "$out/$arm.log" | cut -d= -f2 |
        jq -es --slurpfile reference "$out/expected.json" '.==($reference[0]+$reference[0])' >/dev/null
    rg '^(VmRSS|VmHWM|Threads):' "/proc/$server_pid/status" > "$out/$arm.host-memory"
    gpu_snapshot > "$out/$arm.end"
    stop_server
    check_gpu_owners
done
gpu_snapshot > "$out/complete.txt"
