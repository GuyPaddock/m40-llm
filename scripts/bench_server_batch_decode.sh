#!/usr/bin/env bash
# Benchmark buffered /generate with and without the queued batched decode scheduler.
#
# Usage:
#   source scripts/dev-env.sh
#   M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
#     scripts/bench_server_batch_decode.sh
set -euo pipefail

MODEL=${MODEL:-/mnt/array-fastest/home/guyep/.cache/m40-llm/models/TinyLlama-1.1B-Chat-v1.0.f16.gguf}
PORT_BASE=${PORT_BASE:-52240}
TRIALS=${TRIALS:-3}
MAX_TOKENS=${MAX_TOKENS:-2}
SLOTS=${M40LLM_SERVER_BATCH_DECODE_SLOTS:-8}
LOG_DIR=${LOG_DIR:-/tmp/m40llm_batch_decode_bench_$(date +%Y%m%d_%H%M%S)}
CARGO=${CARGO:-cargo}
CURL=${CURL:-curl}

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
  local case_name="$2"
  local trial="$3"
  shift 3
  local prompts=("$@")
  local port=$((PORT_BASE + batch_decode * 100 + trial))
  local addr="127.0.0.1:${port}"
  local case_dir="${LOG_DIR}/batch${batch_decode}_${case_name}_trial${trial}"
  local server_log="${case_dir}/server.log"
  mkdir -p "$case_dir"

  cleanup
  SERVER_PID=""

  M40LLM_DECODE_GRAPH=0 \
  M40LLM_SERVER_BATCH_DECODE="$batch_decode" \
  M40LLM_SERVER_BATCH_DECODE_SLOTS="$SLOTS" \
  M40LLM_ENABLE_NVCC="${M40LLM_ENABLE_NVCC:-1}" \
  M40LLM_ENABLE_CUBLAS="${M40LLM_ENABLE_CUBLAS:-1}" \
    "$CARGO" run --features cuda,server -- run "$MODEL" \
      --addr "$addr" --require-sm52 >"$server_log" 2>&1 &
  SERVER_PID=$!
  wait_for_server "$addr" "$server_log"

  post_generate "$addr" "Hello" "${case_dir}/warmup.json" "${case_dir}/warmup.meta"

  local start_ns end_ns wall_ms
  start_ns=$(date +%s%N)
  local pids=()
  for i in "${!prompts[@]}"; do
    post_generate "$addr" "${prompts[$i]}" \
      "${case_dir}/response_${i}.json" \
      "${case_dir}/response_${i}.meta" &
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
    output=$(tr '\n' ' ' <"${case_dir}/response_${i}.json" | sed 's/[[:space:]]\+/ /g')
    if [ "$http_code" = "200" ]; then
      ok_count=$((ok_count + 1))
    else
      failed=1
    fi
    time_sum=$(awk -v a="$time_sum" -v b="$time_total" 'BEGIN { printf "%.6f", a + b }')
    printf 'detail\tbatch=%s\tcase=%s\ttrial=%s\trequest=%s\thttp=%s\ttime_total_s=%s\tprompt=%q\tresponse=%s\n' \
      "$batch_decode" "$case_name" "$trial" "$i" "$http_code" "$time_total" "${prompts[$i]}" "$output"
  done

  local avg_latency_s tokens_per_s
  avg_latency_s=$(awk -v sum="$time_sum" -v n="${#prompts[@]}" 'BEGIN { printf "%.6f", sum / n }')
  tokens_per_s=$(awk -v n="${#prompts[@]}" -v mt="$MAX_TOKENS" -v ms="$wall_ms" \
    'BEGIN { if (ms > 0) printf "%.3f", (n * mt) / (ms / 1000.0); else printf "inf" }')

  printf 'summary\tbatch=%s\tcase=%s\ttrial=%s\trequests=%s\tok=%s\tfailed=%s\twall_ms=%s\tavg_latency_s=%s\ttokens_per_s=%s\tlog=%s\n' \
    "$batch_decode" "$case_name" "$trial" "${#prompts[@]}" "$ok_count" "$failed" "$wall_ms" "$avg_latency_s" "$tokens_per_s" "$case_dir"

  cleanup
  SERVER_PID=""
}

main() {
  echo "log_dir=${LOG_DIR}"
  echo "model=${MODEL}"
  echo "trials=${TRIALS} max_tokens=${MAX_TOKENS} slots=${SLOTS}"

  for batch_decode in 0 1; do
    for trial in $(seq 1 "$TRIALS"); do
      run_case "$batch_decode" "batch1_hello" "$trial" \
        "Hello"
      run_case "$batch_decode" "batch2_same" "$trial" \
        "Hello" \
        "Hello"
      run_case "$batch_decode" "batch4_mixed" "$trial" \
        "Hello" \
        "Please write a small Java program to check for stock quotes: " \
        "Summarize CUDA streams in one sentence." \
        "Name one benefit of batching LLM decode requests."
      run_case "$batch_decode" "batch4_skewed" "$trial" \
        "Hi" \
        "Please write a small Java program to check for stock quotes: " \
        "Explain why the Tesla M40 has no Tensor Cores and what that means for GEMM optimization." \
        "Give a concise checklist for validating a CUDA kernel on sm_52."
    done
  done
}

main "$@"
