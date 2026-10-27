#!/usr/bin/env bash
# Recompute every tensor and reject altered coordinates, counts or byte budgets.
set -euo pipefail
[[ $# == 0 || ( $# == 1 && $1 == --negative-controls ) ]]
ulimit -v 524288
prefix=docs/experiments/2026-10-26-maxwell-channel-entropy
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
tar -xzf "$prefix-data.tar.gz" -C "$temporary"
directory="$temporary/m40-channel-entropy-confirm"
[[ $(cut -d' ' -f1 "$directory/model.sha256") == 4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d ]]
[[ $(sha256sum "$temporary/m40-channel-entropy.pl"|cut -d' ' -f1) == 93f519f29258d372f3c63cb40574069c5fc18edba75a77b0b098ea92297144e5 ]]
[[ $(sha256sum "$temporary/m40-channel-entropy-run.sh"|cut -d' ' -f1) == 4621c1c141b2d1ccb9e59e2b108e6d9a0406f68af94c41ea2f7e1392585ce488 ]]
[[ $(sha256sum "$temporary/m40-channel-recount.awk"|cut -d' ' -f1) == 0ae56c0e5fd660634b9bfd6947dad984ad242f4eeed4b4c157b2151e2e702f7a ]]
jq -e '.is_complete and .tensors==6 and (.has_created_cuda_context|not) and (.has_encoder|not)
  and (.has_decoder|not) and (.is_performance_measurement|not)' "$directory/completion.json" > /dev/null

# Validate observations against original source records, without inference.
run_audit() {
  jq --slurpfile original docs/experiments/2026-10-26-maxwell-ffn-entropy-study.jsonl \
    -f "$prefix-audit.jq" "$1"
}

while IFS= read -r source; do
  layer=$(jq -r .layer_id <<< "$source")
  family=$(jq -r .family <<< "$source")
  case_prefix="$directory/layer-$layer-$family"
  gzip -dc "$case_prefix-statistics.json.gz" > "$temporary/statistics.json"
  jq -n --argjson source "$source" --slurpfile raw "$temporary/statistics.json" \
    --slurpfile independent "$case_prefix-independent.jsonl" \
    --arg revision "$(< "$directory/source.txt")" \
    --arg actual_sha "$(cut -d' ' -f1 "$case_prefix-source.sha256")" \
    --arg model_sha "$(cut -d' ' -f1 "$directory/model.sha256")" \
    --argjson rss "$(awk '/Maximum resident set size/{print $NF}' "$case_prefix-resources.log")" \
    --argjson exit_code "$(awk '/Exit status/{print $NF}' "$case_prefix-resources.log")" \
    '{layer_id:$source.layer_id,family:$source.family,source:$source.source,
      absolute_file_offset:$source.absolute_file_offset,source_revision:$revision,
      actual_tensor_sha256:$actual_sha,actual_model_sha256:$model_sha,
      raw:$raw[0],independent:$independent,peak_rss_kib:$rss,resource_exit_code:$exit_code}' > "$temporary/input.json"
  if [[ $layer == 0 && $family == gate ]]; then cp "$temporary/input.json" "$temporary/control.json"; fi
  run_audit "$temporary/input.json" >> "$temporary/cases.jsonl"
done < "$directory/frozen-sources.jsonl"
jq -s 'if length!=6 or (map([.layer_id,.family])|sort)!=[[0,"gate"],[0,"up"],[17,"gate"],[17,"up"],[35,"gate"],[35,"up"]]
  then error("incomplete tensor coverage") else
  {passes_audit:true,tensors:6,input_channels_per_tensor:2048,cases:.,
   has_candidate_for_format_design:all(.[];.has_size_feasibility),has_encoder:false,has_decoder:false,
   has_measured_speedup:false,changes_defaults:false} end' "$temporary/cases.jsonl" > "$temporary/result.json"
if [[ $# == 0 ]]; then cat "$temporary/result.json"; exit 0; fi
mutations=(
  '.actual_model_sha256="wrong"'
  '.actual_tensor_sha256="wrong"'
  '.raw.statistics.scale_bytes=0'
  '.raw.statistics.code_length_table_bytes=0'
  '.raw.statistics.index_bytes=0'
  '.raw.statistics.channels[0].histogram[0]+=1'
  '.raw.statistics.channels[0] as $a|.raw.statistics.channels[1] as $b|
    .raw.statistics.channels[0].histogram=$b.histogram|.raw.statistics.channels[0].entropy_bits=$b.entropy_bits|
    .raw.statistics.channels[1].histogram=$a.histogram|.raw.statistics.channels[1].entropy_bits=$a.entropy_bits'
  '.raw.statistics.conditional_entropy_bits=0'
  '.raw.statistics.optimistic_stored_byte_saving=0.5'
  '.raw.statistics.has_decoder=true'
  '.raw.loaded_cuda_libraries=["libcuda.so"]'
  '.peak_rss_kib=524288'
)
for index in "${!mutations[@]}"; do
  jq -c "${mutations[index]}" "$temporary/control.json" > "$temporary/mutated.json"
  if run_audit "$temporary/mutated.json" > /dev/null 2> "$temporary/error.log"; then
    printf 'Corruption control %s was incorrectly accepted\n' "$index" >&2
    exit 1
  fi
  printf 'Corruption control %s rejected\n' "$index"
done
