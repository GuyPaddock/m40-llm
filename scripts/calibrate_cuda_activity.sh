#!/usr/bin/env bash
# Five serial resident short-request off/on pairs; never HTTP qualification.
set -euo pipefail
if [[ $# != 1 || $1 != /* || -e $1 ]]; then
  printf 'Usage: %s /absolute/new-result-directory\n' "$0" >&2
  exit 1
fi
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
directory=$1
mkdir "$directory"
model=${M40LLM_QWEN_THROUGHPUT_MODEL:?explicit model path required}
probe=${M40LLM_ACTIVITY_TEST_BINARY:?explicit release throughput test binary required}
observer=${M40LLM_ACTIVITY_OBSERVER:?explicit tested observer library required}
runner=${M40LLM_ACTIVITY_NATIVE_RUNNER:?explicit official native runner required}
reference=docs/experiments/2026-10-25-maxwell-packed-projection-runtime-quality.jsonl
[[ -x $probe && -x $runner && -f $observer ]]
[[ $(git status --short) == '' ]]
[[ $(sha256sum "$model" | cut -d' ' -f1) == 4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d ]]
[[ $(sha256sum "$runner" | cut -d' ' -f1) == 234b05b2138264f8fb263c3205e85f4c290e8afe5067e280a4f6f90cdac5696b ]]
while IFS=$'\t' read -r key value; do export "$key=$value"; done < <(
  head -n1 "$reference" | jq -r '.environment|to_entries[]|[.key,.value]|@tsv')
export M40LLM_Q8_EXECUTION_WEIGHTS=split M40LLM_Q8_EXECUTION_FAMILIES=gate-up,down,vocabulary
export M40LLM_Q8_FFN_KERNEL=split-f32x4-128 M40LLM_Q8_PREFILL_BACKEND=dequant-sgemm
export M40LLM_Q8_ATTN_PROJECTIONS=packed-warp-256
export M40LLM_QWEN_THROUGHPUT_MODEL="$model" M40LLM_QWEN_ACTIVITY_REFERENCE="$reference"
: "${CUDA_VISIBLE_DEVICES:?frozen GPU UUID required}"
jq -n --arg revision "$(git rev-parse HEAD)" --arg observer "$(sha256sum "$observer")" \
  --arg probe "$(sha256sum "$probe")" --arg runner "$(sha256sum "$runner")" \
  --arg model "$model" --arg gpu "$CUDA_VISIBLE_DEVICES" \
  '{revision:$revision,observer:$observer,probe:$probe,runner:$runner,model:$model,gpu:$gpu,
    pairs:5,warmups:4,generated_tokens:16,scope:"resident/native short requests, not public HTTP qualification"}' > "$directory/manifest.json"
head -n1 "$reference" | jq '{prompt:.browser_prompt_ids,stream:false,cache_prompt:false,
  return_tokens:true,n_predict:16,temperature:0,top_k:1,top_p:1,min_p:0,
  repeat_penalty:1,frequency_penalty:0,presence_penalty:0,seed:0}' > "$directory/request.json"
runner_pid=
cleanup() {
  if [[ -n $runner_pid ]]; then
    kill -INT "$runner_pid" 2>/dev/null || true
    wait "$runner_pid" || true
  fi
}
trap cleanup EXIT
check_idle() {
  local active
  active=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid,process_name --format=csv,noheader)
  if [[ -n $active ]]; then printf 'GPU is not idle: %s\n' "$active" >&2; return 1; fi
}
snapshot() {
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,pstate,clocks.sm,clocks.mem --format=csv,noheader > "$1-gpu.csv"
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader > "$1-processes.csv"
}
for engine in m40 native; do
  for pair in 0 1 2 3 4; do
    order=(off on)
    if (( pair%2 )); then order=(on off); fi
    for mode in "${order[@]}"; do
      check_idle
      prefix="$directory/$engine-$pair-$mode"
      environment=(env -u LD_PRELOAD -u M40LLM_CUDA_TRACE_PREFIX -u M40LLM_CUDA_TRACE_ACTIVITY -u M40LLM_CUDA_TRACE_CALLBACKS)
      if [[ $mode == on ]]; then
        environment+=(LD_PRELOAD="$observer" M40LLM_CUDA_TRACE_PREFIX="$prefix-trace"
          M40LLM_CUDA_TRACE_ACTIVITY=1 M40LLM_CUDA_TRACE_CALLBACKS=0 M40LLM_CUDA_TRACE_MODULES=0)
      fi
      if [[ $engine == m40 ]]; then
        "${environment[@]}" M40LLM_QWEN_THROUGHPUT_REPORT="$prefix.jsonl" \
          "$probe" --exact q8_activity::test_qualified_runtime_activity_probe \
          --ignored --nocapture --test-threads=1 > "$prefix.log" 2>&1
      else
        "${environment[@]}" LD_LIBRARY_PATH="$(dirname "$runner")/cuda_v12:$(dirname "$runner")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
          "$runner" --model "$model" --port 18436 --host 127.0.0.1 --no-webui --offline \
          -c 1024 -np 1 -b 512 -ub 512 -ngl 99 --flash-attn off --cache-ram 0 \
          --no-jinja --chat-template chatml --context-shift --keep 4 \
          --log-verbosity 4 --no-log-prefix --no-log-timestamps > "$prefix.log" 2>&1 &
        runner_pid=$!
        ready=0
        for attempt in {1..150}; do
          if curl --silent --fail --max-time 1 http://127.0.0.1:18436/health > /dev/null; then ready=1; break; fi
          kill -0 "$runner_pid"
          sleep 1
        done
        [[ $ready == 1 ]]
        printf '%s\n' "$runner_pid" > "$prefix.pid"
        sha256sum "/proc/$runner_pid/exe" > "$prefix-executable.sha256"
        rg 'cuda_v12/libggml-cuda' "/proc/$runner_pid/maps" > "$prefix-backend-maps.txt"
        for round in 0 1 2 3 4; do
          snapshot "$prefix-$round-before"
          curl --silent --show-error --fail --max-time 60 http://127.0.0.1:18436/completion \
            -H 'Content-Type: application/json' --data-binary @"$directory/request.json" \
            --output "$prefix-$round.json" --write-out '%{time_total}\n' > "$prefix-$round-wall.txt"
          jq -e '.timings.cache_n==0 and .timings.prompt_n==65 and .timings.predicted_n==16
            and (.tokens|length)==16 and .stop_type=="limit" and .generation_settings.ignore_eos==false' \
            "$prefix-$round.json" > /dev/null
          snapshot "$prefix-$round-after"
        done
        kill -INT "$runner_pid"
        wait "$runner_pid"
        runner_pid=
      fi
      check_idle
      printf '%s pair=%s mode=%s complete\n' "$engine" "$pair" "$mode"
    done
  done
done
snapshot "$directory/final"
printf 'All ten process pairs completed without overlapping inference.\n'
