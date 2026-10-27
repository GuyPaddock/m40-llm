#!/usr/bin/env bash
# Validate completed GPU spans, exact clocks and actual request boundaries.
set -euo pipefail
[[ $# == 0 || ( $# == 1 && $1 == --negative-controls ) ]]
prefix=docs/experiments/2026-10-26-maxwell-kernel-activity
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
[[ ! -s "$prefix-final-gpu.csv" ]]
[[ $(wc -l < "$prefix-final-processes.txt") == 1 ]]
gzip -dc "$prefix-server.log.gz" | rg -F 'cached n_tokens = 0, memory_seq_rm [0, end)' > "$temporary/cache.txt"
[[ $(wc -l < "$temporary/cache.txt") == 1 ]]
gzip -dc "$prefix-server.log.gz" | rg 'prompt eval time =.* / +65 tokens' > "$temporary/prompt.txt"
[[ $(wc -l < "$temporary/prompt.txt") == 1 ]]
rg -F /tmp/m40-llm-tools/ollama-0.40.11/lib/ollama/cuda_v12/libggml-cuda.so "$prefix-backend-maps.txt" > /dev/null
[[ $(wc -l < "$prefix-processes.csv") == 1 ]]
rg '^3636932, /tmp/m40-llm-tools/ollama-0.40.11/lib/ollama/llama-server, 3463 MiB$' "$prefix-processes.csv" > /dev/null
[[ $(gzip -dc "$prefix-runner-trace.jsonl.gz" | sha256sum | cut -d' ' -f1) == 039e4a2a4812d794c3bd645e69c2cc11efd479bbbea9d2bc1dfd7cdc5a659793 ]]
gzip -dc "$prefix-runner-trace.jsonl.gz" > "$temporary/raw.jsonl"
perl scripts/normalize_cuda_activity.pl "$temporary/raw.jsonl" > "$temporary/normalized.jsonl"
jq -n --slurpfile trace "$temporary/normalized.jsonl" \
  --slurpfile identity "$prefix-identities.json" --slurpfile request "$prefix-request.json" \
  --slurpfile response "$prefix-response.json" --slurpfile tokenizer "$prefix-tokenizer.json" \
  --slurpfile pin docs/experiments/2026-10-26-maxwell-ollama-jit-identities.json \
  --slurpfile frozen docs/experiments/2026-10-25-maxwell-packed-projection-ollama-http.jsonl \
  '{trace:$trace,identity:$identity[0],request:$request[0],response:$response[0],
    tokenizer:$tokenizer[0],pin:$pin[0],frozen:$frozen[0]}' > "$temporary/input.json"
jq -f "$prefix-audit.jq" "$temporary/input.json" > "$temporary/result.json"
if [[ $# == 0 ]]; then
  jq . "$temporary/result.json"
  exit 0
fi
mutations=(
  '.trace|=map(select(.kind!="cuda_activity_complete"))'
  '.trace|=map(if .kind=="cuda_activity_complete" then .dropped_records=1 else . end)'
  '.trace|=map(if .kind=="cuda_kernel_activity" and .correlation_id==19030 then .has_valid_timestamps=false else . end)'
  '.trace|=map(if .kind=="cuda_kernel_activity" and .correlation_id==19030 then .duration_ns+=1 else . end)'
  '.trace|=map(if .kind=="cuda_kernel_activity" and .correlation_id==19030 then .correlation_id=1 else . end)'
  '.trace|=map(select(.kind!="cuda_launch" or .domain!="runtime" or .correlation_id!=19030))'
  '.trace|=map(if .kind=="cuda_kernel_activity" and .correlation_id==19030 then .grid=[1,1,1] else . end)'
  '.response.prompt_eval_cached_count=1'
  '.response.eval_count=512'
  '.tokenizer.tokens[0]=0'
  '.identity.backend_sha256="wrong"'
  '.trace|=map(if .kind=="cuda_trace_environment" then .cuda_force_ptx_jit="1" else . end)'
)
for index in "${!mutations[@]}"; do
  jq "${mutations[index]}" "$temporary/input.json" > "$temporary/mutated.json"
  if jq -f "$prefix-audit.jq" "$temporary/mutated.json" > /dev/null 2> "$temporary/error.log"; then
    printf 'Corruption control %s was incorrectly accepted\n' "$index" >&2
    exit 1
  fi
  printf 'Corruption control %s rejected\n' "$index"
done
