#!/usr/bin/env bash
# Reject corrupted work, identity, coverage and calibration observations.
set -euo pipefail
[[ $# == 0 || ( $# == 1 && $1 == --negative-controls ) ]]
prefix=docs/experiments/2026-10-26-maxwell-prefill-nongemm
quality=docs/experiments/2026-10-25-maxwell-packed-projection-runtime-quality.jsonl
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
gzip -dc "$prefix-events.jsonl.gz" > "$temporary/events.jsonl"

# Evaluate recorded behavior with the immutable completed-quality reference.
run_audit() {
  jq -s --slurpfile quality "$quality" --slurpfile events "$2" \
    -f "$prefix-audit.jq" "$1"
}

run_audit "$prefix-http.jsonl" "$temporary/events.jsonl" > "$temporary/result.json"
if [[ $# == 0 ]]; then cat "$temporary/result.json"; exit 0; fi
jq -n --slurpfile rows "$prefix-http.jsonl" --slurpfile events "$temporary/events.jsonl" \
  '{rows:$rows,events:$events}' > "$temporary/input.json"
mutations=(
  '(.rows[]|select(.kind=="trial" and .phase=="profile-on")|.enqueue_counters.stream_syncs)+=1'
  '(.rows[]|select(.kind=="token_trace")|.generated_token_ids[0])=0'
  '(.rows[]|select(.kind=="manifest")|.control_q8_execution.prepared_bytes)=0'
  '(.rows[]|select(.kind=="trial" and .phase=="profile-off")|.metrics.generated_tokens)=511'
  '(.events[0].stages)|=map(select(.stage!="prefill_attention"))'
  '(.events[0].stages[]|select(.stage=="prefill_attention")|.stream)="decode"'
  '(.events[0].stages[]|select(.stage=="prefill_rope_kv")|.layer_id)=35'
  '(.events[0].stages[]|select(.stage=="prefill_ffn_norm")|.counters.device_allocs)=1'
  '.events[0].event_pool_capacity=592'
  '.events[0].gpu_stage_union_ms=0'
  '(.rows[]|select(.kind=="trial")|.gpu_after.compute_processes_csv)+="\nGPU-extra, 999, unrelated, 1"'
  '(.rows[]|select(.kind=="profile_comparison_summary")|.is_optimization_claim)=true'
)
for index in "${!mutations[@]}"; do
  jq "${mutations[index]}" "$temporary/input.json" > "$temporary/mutated.json"
  jq -c '.rows[]' "$temporary/mutated.json" > "$temporary/rows.jsonl"
  jq -c '.events[]' "$temporary/mutated.json" > "$temporary/mutated-events.jsonl"
  if run_audit "$temporary/rows.jsonl" "$temporary/mutated-events.jsonl" > /dev/null 2> "$temporary/error.log"; then
    printf 'Corruption control %s was incorrectly accepted\n' "$index" >&2
    exit 1
  fi
  printf 'Corruption control %s rejected\n' "$index"
done
