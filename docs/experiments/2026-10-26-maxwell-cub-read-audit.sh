#!/usr/bin/env bash
# Audit recorded work and reject incomplete, corrupted or overstated evidence.
set -euo pipefail
prefix=docs/experiments/2026-10-26-maxwell-cub-read
[[ $# == 0 || ( $# == 1 && $1 == --negative-controls ) ]]
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
jq -e '.original_functions==135 and .original_words==100400 and .current_functions==137
  and .has_identical_original_words and .changed_functions==[] and (.new_functions|length)==2
  and .original_control_words_sha256=="7a954317700bf84cdc40440471d00b2dfb2b5b126cfb562e88c5566b8449970a"' \
  "$prefix-code-comparison.json" > /dev/null
[[ $(gzip -dc "$prefix-candidate.sass.gz" | rg -o '0x[0-9a-f]{16}' | sha256sum | cut -d' ' -f1) == f4f076d1f6bacd28c0fdc4aee8ec70c65e5ecb3c8e0abc7c42bff49dc8f71218 ]]
gzip -dc "$prefix-launch-trace.jsonl.gz" | jq -e -s '
  def require($ok): if $ok then . else error("incomplete or wrong dispatch") end;
  . as $records | map(select(.kind=="cuda_launch")) as $launches
  | require(($records|length)==51 and ($launches|length)==48)
  | require((map(select(.kind=="cuda_trace_complete")))==[{
      kind:"cuda_trace_complete",launch_records:48,dropped_launches:0,failed_launches:0,
      unknown_symbols:0,unsubscribe_status:0,module_records:0,module_bytes:0}])
  | require(all($launches[];.launch_status==0 and .block==[256,1,1] and .shared_bytes==0
      and (.symbol|contains("CUB_200700_520_NS27DeviceSegmentedReduceKernel"))))
  | require(($launches|group_by(.correlation_id)|length)==24
      and all($launches|group_by(.correlation_id)[];length==2
        and (map(.domain)|sort)==["driver","runtime"] and .[0].symbol==.[1].symbol
        and .[0].grid==.[1].grid and .[0].block==.[1].block))
  | true' > /dev/null
jq -s --arg source dfb64513f9a48e56db32c5838e492537ceaf5637 \
  --arg elf 5e6f775b1f993225d1e7a4b68fe667a60453bd0af01aebdcbdd12fc02a557fcd \
  --arg pid 3559220 \
  --slurpfile capture docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  -f "$prefix-audit.jq" "$prefix-screen.jsonl" > "$temporary/result.json"
if [[ $# == 0 ]]; then
  jq . "$temporary/result.json"
  exit 0
fi
mutations=(
  'map(select(.kind!="ffn_read_pair" or .trial!=0 or .layer_id!=0 or .rows!=1))'
  'map(if .kind=="ffn_read_manifest" then .cub_read_contract.compiled_cub_version=200800 else . end)'
  'map(if .kind=="ffn_read_manifest" then .memory.original_weights=0 else . end)'
  'map(if .kind=="ffn_read_validation" then .has_verified_cub_checksums=false else . end)'
  'map(if .kind=="ffn_read_validation" then .checks.has_intact_output_guards=false else . end)'
  'map(if .kind=="ffn_read_validation" then .cub_scratch_bytes=0 else . end)'
  'map(if .kind=="ffn_read_pair" then .measurements[0].enqueue_counters.device_allocs=1 else . end)'
  'map(if .kind=="ffn_read_pair" then .measurements[0].operation="wrong" else . end)'
  'map(if .kind=="ffn_read_pair" then .measurements[0].has_measured_dram_traffic=true else . end)'
  'map(if .kind=="ffn_read_cub_screen" then .result.passes_read_screen=true else . end)'
  'map(if .kind=="ffn_read_pair" then .gpu_before.processes+="; competitor" else . end)'
  'map(if .kind=="ffn_read_complete" then .device_bytes_after=1 else . end)'
)
jq -s . "$prefix-screen.jsonl" > "$temporary/input.json"
for index in "${!mutations[@]}"; do
  jq "${mutations[index]}" "$temporary/input.json" > "$temporary/mutated.json"
  if jq --arg source dfb64513f9a48e56db32c5838e492537ceaf5637 \
    --arg elf 5e6f775b1f993225d1e7a4b68fe667a60453bd0af01aebdcbdd12fc02a557fcd \
    --arg pid 3559220 --slurpfile capture docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
    -f "$prefix-audit.jq" "$temporary/mutated.json" > /dev/null 2> "$temporary/error.log"; then
    printf 'Corruption control %s was incorrectly accepted\n' "$index" >&2
    exit 1
  fi
  printf 'Corruption control %s rejected\n' "$index"
done
