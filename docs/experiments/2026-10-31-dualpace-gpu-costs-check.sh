#!/usr/bin/env bash
# Exercise cost admission on retained measurement records and corrupted variants.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
cost_source=docs/experiments/2026-10-31-dualpace-gpu-source.jsonl.gz
cost_audit=docs/experiments/2026-10-31-dualpace-gpu-costs.jq
cost_tmp=$(mktemp -d /tmp/dualpace-gpu-cost-check.XXXXXX)
trap 'rm -rf "$cost_tmp"' EXIT
printf '%s  %s\n' e2d107a2457ba4089e0a5e5049fcefad98a68e15d0a99ae767f43a81111ee610 "$cost_source" | sha256sum --check
gzip -dc "$cost_source" > "$cost_tmp/source.jsonl"
jq -s -f "$cost_audit" "$cost_tmp/source.jsonl" > "$cost_tmp/accepted.json"

# Known native layer-0 down timings retain their measured units and median.
# Both encodings and all 36 cases must survive successful admission.
jq -e '
  (.costs | length) == 72 and .is_full_decode_profile == false
  and .is_new_hardware_measurement == false
  and (.costs | all(.wall_ns_per_projection.samples == 5 and .iterations_per_trial == 100))
  and ([.costs[] | select(.layer == 0 and .family == "down" and .batch == 1 and .arm == "q8")][0]
    | (.wall_ns_per_projection.median - 177493.88 | fabs) < 0.000001
      and .q8_payload_bytes == 23953408)
' "$cost_tmp/accepted.json" > /dev/null
printf '%s\n' 'PASS: complete source, 72 series, known wall median and units'

# Existing CPU validation must refer to exactly the same tensor and input bytes.
# This is a data join, not a claim that CPU and GPU arithmetic are bit-identical.
sha256sum --check docs/experiments/2026-10-31-dualpace-cpu-validation.sha256
gzip -dc docs/experiments/2026-10-31-dualpace-cpu-validation.jsonl.gz > "$cost_tmp/cpu.jsonl"
jq -e --slurpfile cpu "$cost_tmp/cpu.jsonl" '
  . as $gpu |
  ($cpu | map(select(.kind == "cpu_projection_manifest"))) as $manifest |
  ($cpu | map(select(.kind == "cpu_projection_validation"))) as $rows |
  ($cpu | map(select(.kind == "cpu_projection_setup"))) as $weights |
  ($manifest | length) == 1 and ($rows | length) == 36 and ($weights | length) == 9
  and $manifest[0].model_sha256 == .source_manifest.model_sha256
  and $manifest[0].input_sha256 == .source_manifest.fixture_sha256
  and $manifest[0].capture_report_sha256 == .source_manifest.capture_report_sha256
  and ($rows | all(. as $row |
    ($gpu.costs | map(select(.layer == $row.case.layer and .family == $row.case.family and .batch == $row.case.batch))) as $matches |
    ($matches | length) == 2 and ($matches | all(
      .input_sha256 == $row.case.input_sha256 and .row_positions == $row.case.row_positions
      and .source.shape == [$row.case.columns,$row.case.rows]))))
  and ($weights | all(. as $weight |
    ($gpu.costs | map(select(.layer == $weight.layer and .family == $weight.family))) as $matches |
    ($matches | length) == 8 and ($matches | all(.source == $weight.source and .q8_payload_bytes == $weight.q8_payload_bytes))))
' "$cost_tmp/accepted.json" > /dev/null
printf '%s\n' 'PASS: all 36 CPU input cases and nine original weight identities match both GPU paths'

expect_rejection() {
  local description=$1
  local mutation=$2
  jq -c "$mutation" "$cost_tmp/source.jsonl" > "$cost_tmp/changed.jsonl"
  if jq -s -f "$cost_audit" "$cost_tmp/changed.jsonl" > "$cost_tmp/rejected.json" 2> "$cost_tmp/error"; then
    printf 'FAIL: accepted %s\n' "$description" >&2
    exit 1
  fi
  if [[ -s "$cost_tmp/rejected.json" ]]; then
    printf 'FAIL: emitted partial cost library for %s\n' "$description" >&2
    exit 1
  fi
  printf 'PASS: reject %s without output\n' "$description"
}

# Incomplete captures must not generate a usable cost library.
expect_rejection 'missing completion' 'select(.kind != "shiftadd_complete")'
# Model identity is mandatory before joining CPU and GPU measurements.
expect_rejection 'changed model identity' 'if .kind == "shiftadd_manifest" then .model_sha256 = "wrong" else . end'
# Failed arithmetic/ownership evidence invalidates an otherwise complete run.
expect_rejection 'failed output guards' 'if .kind == "shiftadd_validation" then .checks.guards = "failed" else . end'
expect_rejection 'failed arithmetic' 'if .kind == "shiftadd_validation" then .checks.checks |= map(if .arm == "q8" then .arithmetic.maximum_bound_fraction = 2 else . end) else . end'
# A repeated trial index cannot masquerade as five independent timing pairs.
expect_rejection 'duplicate pair index' 'if .kind == "shiftadd_pair" and .case.control == "q8" and .pair == 4 then .pair = 3 else . end'
# Invalid time and dispatch data must not be converted into service costs.
expect_rejection 'zero duration' 'if .kind == "shiftadd_pair" then .times |= map(if .arm == "q8" then .wall_ms = 0 else . end) else . end'
expect_rejection 'wrong launch count' 'if .kind == "shiftadd_pair" then .times |= map(if .arm == "q8" then .launches = 0 else . end) else . end'
expect_rejection 'missing batch input rows' 'if .kind == "shiftadd_validation" and .phase == "before" then .row_positions = [] else . end'
