#!/usr/bin/env bash
# From a checkout containing these local Git objects, replay a retained GPU run.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=$(realpath -m "${1:?fresh analysis directory}")
mkdir "$out"
tar -xzf "$here/measurements.tar.gz" -C "$out"
raw="$out/kv-october-01"
original=9da2390a6e574a4185712d27ddf893143d94adc5
base=docs/experiments/2026-10-31-kv-quality
git show "$original:$base-results.jsonl" > "$out/original.jsonl"
git show "$original:$base-inputs.json" > "$out/inputs.json"
git show "$original:$base-audit.jq" > "$out/original-audit.jq"
git show c0393f03a7b8adf82ee16017c3121856fa5d3a55:docs/experiments/2026-11-21-kv-audit/compare-october.jq > "$out/compare.jq"
jq -es --slurpfile inputs "$out/inputs.json" -f "$out/original-audit.jq" "$out/original.jsonl" > "$out/original-audit.json"
revision=$(head -1 "$raw/preflight.txt")
exe=$(awk '$2=="target/residency/release/examples/kv_quality" {print $1}' "$raw/preflight.txt")
model=$(awk '$2 ~ /\/blobs\/sha256-/ {print $1}' "$raw/preflight.txt")
inputs=$(cut -d' ' -f1 "$raw/inputs.sha256")
[[ $revision =~ ^[a-f0-9]{40}$ && $exe =~ ^[a-f0-9]{64}$ && $model =~ ^[a-f0-9]{64}$ ]]
rg -q 'sm_52.cubin' "$raw/preflight.txt"
printf '%s  %s\n' "$inputs" "$out/inputs.json" | sha256sum -c - >/dev/null
jq -es --arg model "$model" --arg inputs "$inputs" 'first.model_sha256==$model and first.inputs_sha256==$inputs and
  last.initial_tracked_device_bytes==0 and last.final_tracked_device_bytes==0 and last.is_performance_measurement==false' "$raw/report.jsonl" >/dev/null
jq -es --slurpfile old "$out/original.jsonl" --arg revision "$revision" --arg executable "$exe" \
  -f "$out/compare.jq" "$raw/report.jsonl" > "$out/comparison.json"
jq -es -f "$here/quality.jq" "$raw/report.jsonl" > "$out/quality.json"
cmp "$out/comparison.json" "$here/comparison.json"
cmp "$out/quality.json" "$here/quality.json"
