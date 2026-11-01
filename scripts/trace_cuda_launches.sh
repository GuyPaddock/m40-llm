#!/usr/bin/env bash
# Untimed SDK-only launch observation; no change to the inference build.
set -euo pipefail

if [[ $# -lt 3 || "$2" != -- ]]; then
  printf 'Usage: %s /absolute/new-trace-prefix -- command [args...]\n' "$0" >&2
  exit 1
fi
prefix="$1"
shift 2
if [[ "$prefix" != /* ]]; then
  printf '%s\n' 'Trace prefix must be absolute.' >&2
  exit 1
fi
cuda_root="${CUDA_HOME:-/opt/cuda}"
if [[ -z "${CUPTI_ROOT:-}" || ! -f "$CUPTI_ROOT/include/cupti.h" || ! -f "$cuda_root/include/cuda.h" ]]; then
  printf '%s\n' 'Set CUPTI_ROOT and CUDA_HOME to CUDA 12.x SDK directories.' >&2
  exit 2
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="$root/target/research-tools"
mkdir -p "$output"
library="$output/cuda-launch-trace.so"
"${CXX:-g++}" -std=c++17 -O2 -Wall -Wextra -Werror -fPIC -shared -pthread \
  -I"$cuda_root/include" -I"$CUPTI_ROOT/include" \
  "$root/examples/cuda_launch_trace.cpp" \
  -L"$CUPTI_ROOT/lib" -Wl,-rpath,"$CUPTI_ROOT/lib" -lcupti \
  -o "$library"
exec env M40LLM_CUDA_TRACE_PREFIX="$prefix" \
  LD_PRELOAD="$library${LD_PRELOAD:+:$LD_PRELOAD}" "$@"
