#!/usr/bin/env bash
# Validate actual-runner artifacts and reject incomplete or mislabeled evidence.
set -euo pipefail
prefix=docs/experiments/2026-10-26-maxwell-ollama-jit
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
[[ $# == 0 || ( $# == 1 && $1 == --negative-controls ) ]]
[[ $(gzip -dc "$prefix-mmvq1.sass.gz" | rg -o '0x[0-9a-f]{16}' | sha256sum | cut -d' ' -f1) == 9c27a7b45565707b1f4094d1d3c022bce84c238c54a9e3faf2b5f9830742f5cb ]]
[[ $(gzip -dc "$prefix-mmvq1.sass.gz" | rg -o '0x[0-9a-f]{16}' | wc -l) == 480 ]]
[[ $(gzip -dc "$prefix-quantizer.sass.gz" | rg -o '0x[0-9a-f]{16}' | sha256sum | cut -d' ' -f1) == 972c76d2c6c57d489d1d3bf9f71131795b12b7af22632efb3794bb8a8b3430f1 ]]
[[ $(rg -o '0x[0-9a-f]{16}' docs/experiments/2026-10-25-maxwell-q8-ollama-q8-offline-sass.txt | sha256sum | cut -d' ' -f1) == 9c27a7b45565707b1f4094d1d3c022bce84c238c54a9e3faf2b5f9830742f5cb ]]
[[ $(sha256sum "$prefix-resources.txt" | cut -d' ' -f1) == 700c054672c55481b44c3dfbe805fa379ee73486899b70bf442c0d8909382557 ]]
[[ $(wc -l < "$prefix-final-gpu.csv") == 1 && $(wc -l < "$prefix-final-processes.txt") == 1 ]]
gzip -dc "$prefix-server.log.gz" | rg -F 'cached n_tokens = 0, memory_seq_rm [0, end)' > "$temporary/cache.txt"
[[ $(wc -l < "$temporary/cache.txt") == 1 ]]
gzip -dc "$prefix-server.log.gz" | rg 'prompt eval time =.* / +65 tokens' > "$temporary/prompt.txt"
[[ $(wc -l < "$temporary/prompt.txt") == 1 ]]
rg -F /tmp/m40-llm-tools/ollama-0.40.11/lib/ollama/cuda_v12/libggml-cuda.so "$prefix-backend-maps.txt" > /dev/null
jq -n --slurpfile identity "$prefix-identities.json" \
  --slurpfile trace <(gzip -dc "$prefix-runner-trace.jsonl.gz") \
  --slurpfile modules "$prefix-modules.json" --slurpfile request "$prefix-request.json" \
  --slurpfile response "$prefix-response.json" --slurpfile tokenizer "$prefix-tokenize-response.json" \
  --slurpfile original docs/experiments/2026-10-25-maxwell-q8-dispatch-audit.json \
  --slurpfile frozen docs/experiments/2026-10-25-maxwell-packed-projection-ollama-http.jsonl \
  '{identity:$identity[0],trace:$trace,modules:$modules[0],request:$request[0],response:$response[0],
    tokenizer:$tokenizer[0],original:$original[0],frozen:$frozen[0]}' > "$temporary/input.json"
jq -f "$prefix-audit.jq" "$temporary/input.json" > "$temporary/result.json"
if [[ $# == 0 ]]; then
  jq . "$temporary/result.json"
  exit 0
fi
mutations=(
  '.trace|=map(select(.kind!="cuda_trace_complete"))'
  '.trace|=map(if .kind=="cuda_trace_environment" then .cuda_force_ptx_jit="1" else . end)'
  '.trace|=map(if .kind=="cuda_trace_complete" then .dropped_launches=1 else . end)'
  '.trace|=map(select(.kind!="cuda_launch" or .domain!="runtime" or .correlation_id!=1))'
  '.trace|=map(if .kind=="cuda_module" then .cubin_bytes=1 else . end)'
  '.response.eval_count=512'
  '.response.prompt_eval_cached_count=1'
  '.tokenizer.tokens[0]=0'
  '.identity.loaded_mmvq_words_sha256="wrong"'
  '.identity.backend_sha256="wrong"'
)
for index in "${!mutations[@]}"; do
  jq "${mutations[index]}" "$temporary/input.json" > "$temporary/mutated.json"
  if jq -f "$prefix-audit.jq" "$temporary/mutated.json" > /dev/null 2> "$temporary/error.log"; then
    printf 'Corruption control %s was incorrectly accepted\n' "$index" >&2
    exit 1
  fi
  printf 'Corruption control %s rejected\n' "$index"
done
