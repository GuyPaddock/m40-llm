#!/usr/bin/env bash
# Benchmark buffered /generate with and without the queued batched decode scheduler.
#
# Usage:
#   source scripts/dev-env.sh
#   M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
#     scripts/bench_server_batch_decode.sh
#
# The server runtime now defaults to compressed KV. Dense batched prefill/decode
# is the path this script measures, so it passes --kv-compress-mode off unless
# SERVER_EXTRA_ARGS overrides the full argument list behavior for diagnostics.
set -euo pipefail

MODEL=${MODEL:-/mnt/array-fastest/home/guyep/.cache/m40-llm/models/TinyLlama-1.1B-Chat-v1.0.f16.gguf}
PORT_BASE=${PORT_BASE:-52240}
TRIALS=${TRIALS:-3}
MAX_TOKENS=${MAX_TOKENS:-2}
MAX_CONTEXT_TOKENS=${MAX_CONTEXT_TOKENS:-}
SLOTS=${M40LLM_SERVER_BATCH_DECODE_SLOTS:-8}
BATCH_DECODE_MODES=${BATCH_DECODE_MODES:-"0 1"}
PREFILL_MODES=${PREFILL_MODES:-"0"}
CASES=${CASES:-"batch1_hello batch2_same batch4_mixed batch4_skewed"}
STAGGER_MS=${STAGGER_MS:-0}
LOG_DIR=${LOG_DIR:-/tmp/m40llm_batch_decode_bench_$(date +%Y%m%d_%H%M%S)}
CARGO=${CARGO:-cargo}
CARGO_RUN_ARGS=${CARGO_RUN_ARGS:-}
CURL=${CURL:-curl}
SERVER_EXTRA_ARGS=${SERVER_EXTRA_ARGS:---kv-compress-mode off}

mkdir -p "$LOG_DIR"
RESULTS_TSV="${LOG_DIR}/results.tsv"
exec > >(tee "$RESULTS_TSV")

SERVER_PID=""

cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 50); do
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
  fi
}
trap cleanup EXIT INT TERM

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

post_generate() {
  local addr="$1"
  local prompt="$2"
  local out_file="$3"
  local meta_file="$4"
  local payload
  payload=$(printf '{"prompt":"%s","max_tokens":%s,"temperature":1.0,"top_k":1}' \
    "$(json_escape "$prompt")" "$MAX_TOKENS")
  "$CURL" -sS \
    -w 'http_code=%{http_code}\ntime_total=%{time_total}\n' \
    -H 'Content-Type: application/json' \
    -d "$payload" \
    "http://${addr}/generate" \
    -o "$out_file" >"$meta_file"
}

sleep_ms() {
  local ms="$1"
  if [ "$ms" -le 0 ]; then
    return 0
  fi
  awk -v ms="$ms" 'BEGIN { printf "%.3f\n", ms / 1000.0 }' | xargs sleep
}

wait_for_server() {
  local addr="$1"
  local log_file="$2"
  for _ in $(seq 1 240); do
    if "$CURL" -sf "http://${addr}/health" >/dev/null; then
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server exited before becoming healthy; log follows:" >&2
      tail -n 120 "$log_file" >&2 || true
      return 1
    fi
    sleep 0.5
  done
  echo "server did not become healthy at ${addr}; log follows:" >&2
  tail -n 120 "$log_file" >&2 || true
  return 1
}

run_case() {
  local batch_decode="$1"
  local batch_prefill="$2"
  local case_name="$3"
  local trial="$4"
  shift 4
  local prompts=("$@")
  local port=$((PORT_BASE + batch_decode * 100 + batch_prefill * 200 + trial))
  local addr="127.0.0.1:${port}"
  local case_dir="${LOG_DIR}/decode${batch_decode}_prefill${batch_prefill}_${case_name}_trial${trial}"
  local server_log="${case_dir}/server.log"
  mkdir -p "$case_dir"

  cleanup
  SERVER_PID=""
  local context_args=()
  if [ -n "$MAX_CONTEXT_TOKENS" ]; then
    context_args=(--max-context-tokens "$MAX_CONTEXT_TOKENS")
  fi

  M40LLM_DECODE_GRAPH=0 \
  M40LLM_SERVER_BATCH_DECODE="$batch_decode" \
  M40LLM_SERVER_BATCH_PREFILL="$batch_prefill" \
  M40LLM_SERVER_BATCH_DECODE_SLOTS="$SLOTS" \
  M40LLM_SERVER_BATCH_LOG="${M40LLM_SERVER_BATCH_LOG:-1}" \
  M40LLM_ENABLE_NVCC="${M40LLM_ENABLE_NVCC:-1}" \
  M40LLM_ENABLE_CUBLAS="${M40LLM_ENABLE_CUBLAS:-1}" \
    "$CARGO" run $CARGO_RUN_ARGS --features cuda,server -- run "$MODEL" \
      --addr "$addr" --require-sm52 "${context_args[@]}" \
      $SERVER_EXTRA_ARGS >"$server_log" 2>&1 &
  SERVER_PID=$!
  wait_for_server "$addr" "$server_log"

  post_generate "$addr" "Hello" "${case_dir}/warmup.json" "${case_dir}/warmup.meta"

  local start_ns end_ns wall_ms
  start_ns=$(date +%s%N)
  local pids=()
  for i in "${!prompts[@]}"; do
    local launch_offset_ms=$((i * STAGGER_MS))
    (
      printf 'launch_offset_ms=%s\n' "$launch_offset_ms" >"${case_dir}/response_${i}.launch"
      sleep_ms "$launch_offset_ms"
      post_generate "$addr" "${prompts[$i]}" \
        "${case_dir}/response_${i}.json" \
        "${case_dir}/response_${i}.meta"
    ) &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  end_ns=$(date +%s%N)
  wall_ms=$(((end_ns - start_ns) / 1000000))

  local ok_count=0
  local time_sum
  time_sum=0
  for i in "${!prompts[@]}"; do
    local http_code time_total output
    http_code=$(awk -F= '/^http_code=/{print $2}' "${case_dir}/response_${i}.meta")
    time_total=$(awk -F= '/^time_total=/{print $2}' "${case_dir}/response_${i}.meta")
    launch_offset_ms=$(awk -F= '/^launch_offset_ms=/{print $2}' "${case_dir}/response_${i}.launch")
    output=$(tr '\n' ' ' <"${case_dir}/response_${i}.json" | sed 's/[[:space:]]\+/ /g')
    if [ "$http_code" = "200" ]; then
      ok_count=$((ok_count + 1))
    else
      failed=1
    fi
    time_sum=$(awk -v a="$time_sum" -v b="$time_total" 'BEGIN { printf "%.6f", a + b }')
    printf 'detail\tbatch=%s\tprefill=%s\tcase=%s\ttrial=%s\trequest=%s\tlaunch_offset_ms=%s\thttp=%s\ttime_total_s=%s\tprompt=%q\tresponse=%s\n' \
      "$batch_decode" "$batch_prefill" "$case_name" "$trial" "$i" "$launch_offset_ms" "$http_code" "$time_total" "${prompts[$i]}" "$output"
  done

  local avg_latency_s tokens_per_s
  avg_latency_s=$(awk -v sum="$time_sum" -v n="${#prompts[@]}" 'BEGIN { printf "%.6f", sum / n }')
  tokens_per_s=$(awk -v n="${#prompts[@]}" -v mt="$MAX_TOKENS" -v ms="$wall_ms" \
    'BEGIN { if (ms > 0) printf "%.3f", (n * mt) / (ms / 1000.0); else printf "inf" }')

  printf 'summary\tbatch=%s\tprefill=%s\tcase=%s\ttrial=%s\trequests=%s\tok=%s\tfailed=%s\twall_ms=%s\tavg_latency_s=%s\ttokens_per_s=%s\tlog=%s\n' \
    "$batch_decode" "$batch_prefill" "$case_name" "$trial" "${#prompts[@]}" "$ok_count" "$failed" "$wall_ms" "$avg_latency_s" "$tokens_per_s" "$case_dir"

  cleanup
  SERVER_PID=""
}

main() {
  echo "log_dir=${LOG_DIR}"
  echo "model=${MODEL}"
  echo "trials=${TRIALS} max_tokens=${MAX_TOKENS} max_context_tokens=${MAX_CONTEXT_TOKENS:-model} slots=${SLOTS} batch_decode_modes=${BATCH_DECODE_MODES} prefill_modes=${PREFILL_MODES} cases=${CASES} stagger_ms=${STAGGER_MS}"
  echo "cargo_run_args=${CARGO_RUN_ARGS}"
  echo "server_extra_args=${SERVER_EXTRA_ARGS}"

  for batch_decode in $BATCH_DECODE_MODES; do
    for batch_prefill in $PREFILL_MODES; do
      if [ "$batch_decode" = "0" ] && [ "$batch_prefill" = "1" ]; then
        continue
      fi

      for trial in $(seq 1 "$TRIALS"); do
        for case_name in $CASES; do
          case "$case_name" in
            batch1_hello)
              run_case "$batch_decode" "$batch_prefill" "$case_name" "$trial" \
                "Hello"
              ;;
            batch2_same)
              run_case "$batch_decode" "$batch_prefill" "$case_name" "$trial" \
                "Hello" \
                "Hello"
              ;;
            batch4_mixed)
              run_case "$batch_decode" "$batch_prefill" "$case_name" "$trial" \
                "Hello" \
                "Please write a small Java program to check for stock quotes: " \
                "Summarize CUDA streams in one sentence." \
                "Name one benefit of batching LLM decode requests."
              ;;
            batch4_repeat)
              run_case "$batch_decode" "$batch_prefill" "$case_name" "$trial" \
                "Repeat the word BLUE over and over, separated by spaces. Continue until stopped." \
                "Repeat the word GREEN over and over, separated by spaces. Continue until stopped." \
                "Repeat the word RED over and over, separated by spaces. Continue until stopped." \
                "Repeat the word YELLOW over and over, separated by spaces. Continue until stopped."
              ;;
            batch4_skewed)
              run_case "$batch_decode" "$batch_prefill" "$case_name" "$trial" \
                "Hi" \
                "Please write a small Java program to check for stock quotes: " \
                "Explain why the Tesla M40 has no Tensor Cores and what that means for GEMM optimization." \
                "Give a concise checklist for validating a CUDA kernel on sm_52."
              ;;
            staggered_mixed)
              run_case "$batch_decode" "$batch_prefill" "$case_name" "$trial" \
                "Hello" \
                "Explain CUDA stream scheduling and why overlap matters for LLM serving on Maxwell GPUs." \
                "Summarize CUDA streams in one sentence." \
                "Name one benefit of batching LLM decode requests."
              ;;
            staggered_skewed)
              run_case "$batch_decode" "$batch_prefill" "$case_name" "$trial" \
                "Hi" \
                "Explain why the Tesla M40 has no Tensor Cores and outline a careful FP16-storage FP32-compute inference strategy." \
                "Give a concise checklist for validating a CUDA kernel on sm_52." \
                "Please write a small Java program to check for stock quotes: "
              ;;
            *)
              echo "unknown benchmark case '${case_name}'" >&2
              exit 2
              ;;
          esac
        done
      done
    done
  done
}

main "$@"
