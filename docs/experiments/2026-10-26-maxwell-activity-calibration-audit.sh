#!/usr/bin/env bash
# Reconcile resident timings, actual output and intentionally failed calibration.
set -euo pipefail
[[ $# == 0 || ( $# == 1 && $1 == --negative-controls ) ]]
prefix=docs/experiments/2026-10-26-maxwell-activity-calibration
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
tar -xzf "$prefix-data.tar.gz" -C "$temporary"
m40="$temporary/m40-activity-calibration-screen"
native="$temporary/m40-activity-calibration-native"
[[ ! -s "$native/final-processes.csv" && ! -s "$m40/native-0-off-backend-maps.txt" ]]
rg -F 'no usable GPU found' "$m40/native-0-off.log" > /dev/null
[[ ! -e "$m40/native-0-off-0.json" ]]
for engine in m40 native; do
  directory=$m40
  if [[ $engine == native ]]; then directory=$native; fi
  for pair in 0 1 2 3 4; do
    for mode in off on; do
      case_prefix="$directory/$engine-$pair-$mode"
      if [[ $engine == m40 ]]; then
        head -n1 "$case_prefix.jsonl" | jq -r .pid > "$temporary/pid"
        jq -s '{manifest:.[0],generations:map(select(.kind=="activity_generation")),complete:.[-1]}' \
          "$case_prefix.jsonl" > "$temporary/case.json"
      else
        cp "$case_prefix.pid" "$temporary/pid"
        [[ $(cut -d' ' -f1 "$case_prefix-executable.sha256") == 234b05b2138264f8fb263c3205e85f4c290e8afe5067e280a4f6f90cdac5696b ]]
        rg -F /tmp/m40-llm-tools/ollama-0.40.11/lib/ollama/cuda_v12/libggml-cuda.so "$case_prefix-backend-maps.txt" > /dev/null
        rg -F 'offloaded 37/37 layers to GPU' "$case_prefix.log" > /dev/null
        rg -F 'CUDA0 KV buffer size =    36.00 MiB' "$case_prefix.log" > /dev/null
        jq -s --slurpfile request "$native/request.json" '{generations:.,request:$request[0]}' \
          "$case_prefix-0.json" "$case_prefix-1.json" "$case_prefix-2.json" \
          "$case_prefix-3.json" "$case_prefix-4.json" > "$temporary/case.json"
        for round in 0 1 2 3 4; do
          for boundary in before after; do
            [[ $(wc -l < "$case_prefix-$round-$boundary-processes.csv") == 1 ]]
            [[ $(cut -d, -f1 "$case_prefix-$round-$boundary-processes.csv") == "$(cat "$temporary/pid")" ]]
            rg -F 'P0, 1113 MHz, 3004 MHz' "$case_prefix-$round-$boundary-gpu.csv" > /dev/null
          done
        done
      fi
      pid=$(cat "$temporary/pid")
      printf 'null\n' > "$temporary/trace.json"
      if [[ $mode == on ]]; then
        trace="$case_prefix-trace.$pid.jsonl"
        gzip -dc "$prefix-$engine-$pair-trace.jsonl.gz" > "$trace"
        head -n1 "$trace" > "$temporary/start.json"
        tail -n2 "$trace" > "$temporary/completion.jsonl"
        [[ $(rg -c '"kind":"cuda_activity_error"' "$trace" || true) == '' ]]
        jq -n --slurpfile start "$temporary/start.json" --slurpfile end "$temporary/completion.jsonl" \
          --argjson actual "$(rg -c '"kind":"cuda_kernel_activity"' "$trace")" \
          --arg sha256 "$(sha256sum "$trace" | cut -d' ' -f1)" \
          '{start:$start[0],activity:$end[0],callbacks:$end[1],actual_records:$actual,sha256:$sha256}' > "$temporary/trace.json"
      else
        if compgen -G "$case_prefix-trace.*.jsonl" > /dev/null; then exit 1; fi
      fi
      jq -nc --arg engine "$engine" --arg mode "$mode" --argjson pair "$pair" --argjson pid "$pid" \
        --slurpfile data "$temporary/case.json" --slurpfile trace "$temporary/trace.json" \
        '{engine:$engine,pair:$pair,mode:$mode,pid:$pid,data:$data[0],trace:$trace[0]}' >> "$temporary/cases.jsonl"
    done
  done
done
jq -n --slurpfile cases "$temporary/cases.jsonl" --slurpfile original "$m40/manifest.json" \
  --slurpfile resumed "$native/manifest.json" \
  --slurpfile frozen docs/experiments/2026-10-25-maxwell-packed-projection-runtime-quality.jsonl \
  '{cases:$cases,original:$original[0],resumed:$resumed[0],frozen:$frozen}' > "$temporary/input.json"
jq -f "$prefix-audit.jq" "$temporary/input.json" > "$temporary/result.json"
if [[ $# == 0 ]]; then cat "$temporary/result.json"; exit 0; fi
mutations=(
  '.cases|=.[1:]'
  '.cases[0].data.generations[4].generated_ids[0]=0'
  '.cases[0].data.complete.tracked_after=1'
  '.cases[1].trace.activity.dropped_records=0'
  '.cases[1].trace.actual_records=99999'
  '.cases[1].trace.start.captures_launch_callbacks=true'
  '.cases[10].data.generations[4].timings.cache_n=1'
  '.cases[10].data.generations[4].tokens[0]=0'
  '.cases[10].data.generations[4].timings.predicted_n=512'
  '.cases[11].trace.activity.dropped_records=1'
  '.original.observer="wrong"'
  '.cases[0].data.generations[4].operations.ffn_q8_split_f32x4_128=0'
)
for index in "${!mutations[@]}"; do
  jq "${mutations[index]}" "$temporary/input.json" > "$temporary/mutated.json"
  if jq -f "$prefix-audit.jq" "$temporary/mutated.json" > /dev/null 2> "$temporary/error.log"; then
    printf 'Corruption control %s was incorrectly accepted\n' "$index" >&2; exit 1
  fi
  printf 'Corruption control %s rejected\n' "$index"
done
