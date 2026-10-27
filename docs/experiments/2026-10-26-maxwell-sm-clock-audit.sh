#!/usr/bin/env bash
# Reconcile actual work, process-level timing units and original GPU policy.
set -euo pipefail
[[ $# == 0 || ( $# == 1 && $1 == --negative-controls ) ]]
prefix=docs/experiments/2026-10-26-maxwell-sm-clock
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
tar -xzf "$prefix-data.tar.gz" -C "$temporary"
directory="$temporary/m40-sm-clock-screen"
while IFS= read -r meta; do
  pair=$(jq -r .pair <<< "$meta")
  state=$(jq -r .state <<< "$meta")
  jq -cs --argjson meta "$meta" '{meta:$meta,records:.}' "$directory/pair-$pair-$state.jsonl"
done < "$directory/processes.jsonl" > "$temporary/runs.jsonl"
jq -n --slurpfile runs "$temporary/runs.jsonl" --slurpfile completion "$directory/completion.json" \
  --rawfile before "$directory/before-policy.csv" --rawfile after "$directory/final-policy.csv" \
  --rawfile independent_after "$directory/independent-final-policy.csv" \
  --rawfile boost_before "$directory/before-boost.csv" --rawfile boost_after "$directory/final-boost.csv" \
  --rawfile before_processes "$directory/before-processes.csv" --rawfile after_processes "$directory/final-processes.csv" \
  --rawfile independent_after_processes "$directory/independent-final-processes.csv" \
  '{runs:$runs,completion:$completion[0],policy:{before:$before,after:$after,independent_after:$independent_after,
    boost_before:$boost_before,boost_after:$boost_after,before_processes:$before_processes,
    after_processes:$after_processes,independent_after_processes:$independent_after_processes}|map_values(gsub(", +";",")|gsub("\n$";""))}' > "$temporary/input.json"

# The validator reads observations, never source or documentation assertions.
run_audit() {
  jq --slurpfile capture docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
    -f "$prefix-audit.jq" "$1"
}
run_audit "$temporary/input.json" > "$temporary/result.json"
if [[ $# == 0 ]]; then cat "$temporary/result.json"; exit 0; fi
mutations=(
  '.runs|=.[1:]'
  '.runs[0].meta.state="low"'
  '(.runs[0].records[]|select(.kind=="ffn_read_pair")|.gpu_after.device)|=gsub("3004 MHz";"405 MHz")'
  '(.runs[0].records[]|select(.kind=="ffn_read_pair")|.gpu_before.device)|=gsub("1113 MHz|1114 MHz";"324 MHz")'
  '(.runs[0].records[]|select(.kind=="ffn_read_validation")|.checks.has_matching_cpu_checksums)=false'
  '(.runs[0].records[]|select(.kind=="ffn_read_pair")|.measurements[]|select(.stage=="weight_read_checksum")|.has_measured_dram_traffic)=true'
  '(.runs[0].records[]|select(.kind=="ffn_read_pair")|.measurements[].enqueue_counters.device_allocs)=1'
  '(.runs[0].records[]|select(.kind=="ffn_read_warmup")|.real_ffn.elapsed_gpu_ms)=0'
  '(.runs[0].records[]|select(.kind=="ffn_read_manifest")|.prepared_weights.resident_bytes)=0'
  '(.runs[0].records[]|select(.kind=="ffn_read_pair")|.gpu_after.processes)+="\n999, unrelated, 1 MiB"'
  '.policy.boost_after="Off,Off"'
  '.completion.is_original_policy_restored=false'
  '(.runs[0].records[]|select(.kind=="ffn_read_pair")|.measurements[].started_unix_us)=0'
  '(.runs[0].records[]|select(.kind=="ffn_read_complete")|.device_bytes_after)=1'
)
for index in "${!mutations[@]}"; do
  jq "${mutations[index]}" "$temporary/input.json" > "$temporary/mutated.json"
  if run_audit "$temporary/mutated.json" > /dev/null 2> "$temporary/error.log"; then
    printf 'Corruption control %s was incorrectly accepted\n' "$index" >&2
    exit 1
  fi
  printf 'Corruption control %s rejected\n' "$index"
done
