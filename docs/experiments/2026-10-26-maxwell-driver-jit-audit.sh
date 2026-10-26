#!/usr/bin/env bash
# Verify trace behavior and retained compiled-code witnesses, not source text.
set -euo pipefail
prefix=docs/experiments/2026-10-26-maxwell-driver-jit
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
[[ $# == 0 || ( $# == 1 && $1 == --negative-controls ) ]]
[[ $(gzip -dc "$prefix-project.words.gz" | sha256sum | cut -d' ' -f1) == 7a954317700bf84cdc40440471d00b2dfb2b5b126cfb562e88c5566b8449970a ]]
[[ $(gzip -dc "$prefix-project.words.gz" | wc -l) == 100400 ]]
[[ $(gzip -dc "$prefix-gate.sass.gz" | rg -o '0x[0-9a-f]{16}' | sha256sum | cut -d' ' -f1) == 31cd38f6ca0d4a0bebd03206f484bf8045834d8af592a645cdb168cfaf02e321 ]]
[[ $(gzip -dc "$prefix-down.sass.gz" | rg -o '0x[0-9a-f]{16}' | sha256sum | cut -d' ' -f1) == fafef036cf36582ad7c230dd9b2502575a53356ada1396aaeac8e9baf0872397 ]]
[[ $(gzip -dc "$prefix-resources.txt.gz" | sha256sum | cut -d' ' -f1) == 2ea1b13638fb0d47973fd7a402da72d7d15bcc756e795779ea376b0dd8487eb6 ]]
for mode in native no-cublas-native no-cublas-forced; do
  gzip -dc "$prefix-$mode.log.gz" | rg -q '^test result: ok. 2 passed; 0 failed; 0 ignored;'
done
gzip -dc "$prefix-forced.log.gz" | rg -q '^test result: FAILED. 0 passed; 2 failed;'
jq -n --slurpfile identity "$prefix-identities.json" \
  --slurpfile native <(gzip -dc "$prefix-no-cublas-native-trace.jsonl.gz") \
  --slurpfile forced <(gzip -dc "$prefix-no-cublas-forced-trace.jsonl.gz") \
  --slurpfile enabled_native <(gzip -dc "$prefix-native-trace.jsonl.gz") \
  --slurpfile enabled_forced <(gzip -dc "$prefix-forced-trace.jsonl.gz") \
  --slurpfile probe_native "$prefix-cublas-native.json" --slurpfile probe_forced "$prefix-cublas-forced.json" \
  --slurpfile smoke <(gzip -dc "$prefix-smoke-provenance-trace.jsonl.gz") \
  --slurpfile smoke_output "$prefix-smoke-provenance-output.json" \
  --slurpfile image_error <(gzip -dc "$prefix-image-write-error-trace.jsonl.gz") \
  --slurpfile setting_error <(gzip -dc "$prefix-invalid-setting-trace.jsonl.gz") \
  '{identity:$identity[0],native:$native,forced:$forced,enabled_native:$enabled_native,enabled_forced:$enabled_forced,
    probe_native:$probe_native[0],probe_forced:$probe_forced[0],smoke:$smoke,smoke_output:$smoke_output[0],
    image_error:$image_error,setting_error:$setting_error}' > "$temporary/input.json"
jq -f "$prefix-audit.jq" "$temporary/input.json" > "$temporary/result.json"
if [[ $# == 0 ]]; then
  jq . "$temporary/result.json"
  exit 0
fi
mutations=(
  '.forced|=map(select(.kind!="cuda_trace_complete"))'
  '.forced|=map(if .kind=="cuda_trace_environment" then .cuda_force_ptx_jit="0" else . end)'
  '.native|=map(if .kind=="cuda_module" then .cubin_bytes=1 else . end)'
  '.forced|=map(if .kind=="cuda_launch" then .grid=[24,1,1] else . end)'
  '.forced|=map(if .kind=="cuda_launch" then .launch_status=1 else . end)'
  '.probe_forced.cublas_create_status=0'
  '.identity.project_words.forced_sha256="wrong"'
  '.smoke_output.actual=95'
  '.image_error=[]'
  '.identity.resources.forced_sha256="wrong"'
)
for index in "${!mutations[@]}"; do
  jq "${mutations[index]}" "$temporary/input.json" > "$temporary/mutated.json"
  if jq -f "$prefix-audit.jq" "$temporary/mutated.json" > /dev/null 2> "$temporary/error.log"; then
    printf 'Corruption control %s was incorrectly accepted\n' "$index" >&2
    exit 1
  fi
  printf 'Corruption control %s rejected\n' "$index"
done
