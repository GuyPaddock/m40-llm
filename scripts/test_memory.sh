#!/usr/bin/env bash
# scripts/test_memory.sh
# Background-run the m40-llm server, probe /generate at varying lengths, sample nvidia-smi, and cleanup.
set -u

PORT=${PORT:-55180}
ADDR=0.0.0.0:${PORT}
MODEL_NAME=${MODEL_NAME:-local}
MODEL_PATH=${MODEL_PATH:-./model.gguf}
INTERVAL=${INTERVAL:-1}
DURATION=${DURATION:-60}
MAX_TOKENS_LIST=${MAX_TOKENS_LIST:-"1 8 32 128 256"}
CURL=${CURL:-curl}
NVIDIA_SMI=${NVIDIA_SMI:-nvidia-smi}
# Auto-align GPU_INDEX to DEVICE_ID by default
DEVICE_ID=${DEVICE_ID:-0}
GPU_INDEX=${GPU_INDEX:-$DEVICE_ID}
BIN=./target/release/m40-llm
LOG_DIR=./scripts
SERVER_LOG=${LOG_DIR}/server_memtest.log
SMI_LOG=${LOG_DIR}/nvidia_smi_gpu${GPU_INDEX}_${PORT}.log
PID_FILE=${LOG_DIR}/server_${PORT}.pid
CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
# Respect --device-id by default; set to 1 to force M40 auto-select
M40LLM_FORCE_M40=${M40LLM_FORCE_M40:-0}
M40LLM_ALLOC_BT=${M40LLM_ALLOC_BT:-0}

cleanup() {
  if [ -n "${SMI_PID:-}" ] && kill -0 $SMI_PID 2>/dev/null; then
    kill $SMI_PID 2>/dev/null || true
    wait $SMI_PID 2>/dev/null || true
  fi
  if [ -n "${SRV_PID:-}" ] && kill -0 $SRV_PID 2>/dev/null; then
    kill $SRV_PID 2>/dev/null || true
    for _ in $(seq 1 20); do
      if ! kill -0 $SRV_PID 2>/dev/null; then break; fi
      sleep 0.2
    done
  fi
  rm -f "$PID_FILE"
}
trap cleanup EXIT INT TERM

mkdir -p "$LOG_DIR"

# Build release if needed
if [ ! -x "$BIN" ]; then
  echo "[build] cargo build --release --features server,cuda,gguf_ext"
  cargo build --release --features server,cuda,gguf_ext || exit 1
fi

# If MODEL_NAME 'local' is used, symlink MODEL_PATH into cache so CLI finds it
CACHE_DIR="$HOME/.local/share/m40-llm/models/${MODEL_NAME}"
if [ ! -f "$CACHE_DIR/model.gguf" ]; then
  mkdir -p "$CACHE_DIR"
  if [ -f "$MODEL_PATH" ]; then
    ln -sf "$(realpath "$MODEL_PATH")" "$CACHE_DIR/model.gguf"
  else
    echo "Model not found at MODEL_PATH=$MODEL_PATH" >&2
    exit 1
  fi
fi

# Start server in background
set -m
CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING M40LLM_FORCE_M40=$M40LLM_FORCE_M40 M40LLM_ALLOC_BT=$M40LLM_ALLOC_BT \
$BIN run "$MODEL_NAME" --addr "$ADDR" --device-id "$DEVICE_ID" >"$SERVER_LOG" 2>&1 &
SRV_PID=$!
echo $SRV_PID > "$PID_FILE"
echo "[server] pid=$SRV_PID log=$SERVER_LOG"

# Wait for ready (GET /health)
for _ in $(seq 1 60); do
  if $CURL -sf "http://127.0.0.1:${PORT}/health" -o /dev/null ; then
    break
  fi
  sleep 0.5
  if ! kill -0 $SRV_PID 2>/dev/null; then
    echo "[server] exited prematurely" >&2
    tail -n +1 "$SERVER_LOG" >&2
    exit 1
  fi
done

# nvidia-smi sampler
(
  for _ in $(seq 1 $DURATION); do
    ts=$(date +%s)
    $NVIDIA_SMI --query-gpu=timestamp,name,index,uuid,memory.total,memory.used,memory.free --format=csv -i ${GPU_INDEX} | sed "s/^/$ts,/" >> "$SMI_LOG" 2>/dev/null || true
    sleep $INTERVAL
  done
) &
SMI_PID=$!

echo "[smi] pid=$SMI_PID log=$SMI_LOG"

# Issue test requests
for mt in $MAX_TOKENS_LIST; do
  echo "[test] max_tokens=$mt"
  $CURL -s -X POST "http://127.0.0.1:${PORT}/generate" \
    -H 'Content-Type: application/json' \
    --data-binary @<(printf '{"prompt":"Hello","max_tokens":%s}' "$mt") \
    | tee -a "$LOG_DIR/curl_${PORT}.log" >/dev/null
  echo
  sleep 1
  if ! kill -0 $SRV_PID 2>/dev/null; then
    echo "[server] died during tests" >&2
    break
  fi
done

# Cleanup
if kill -0 $SMI_PID 2>/dev/null; then
  kill $SMI_PID 2>/dev/null || true
  wait $SMI_PID 2>/dev/null || true
fi

if kill -0 $SRV_PID 2>/dev/null; then
  kill $SRV_PID 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! kill -0 $SRV_PID 2>/dev/null; then break; fi
    sleep 0.2
  done
fi

rm -f "$PID_FILE"

echo "[done] Logs: $SERVER_LOG $SMI_LOG $LOG_DIR/curl_${PORT}.log"
