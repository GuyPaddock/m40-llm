#!/usr/bin/env bash
# Optional SDK-only probe. No CUPTI dependency is added to Cargo or inference.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/dev-env.sh"

if [[ -z "${CUPTI_ROOT:-}" || ! -f "$CUPTI_ROOT/include/cupti.h" || ! -d "$CUPTI_ROOT/lib" ]]; then
  printf '%s\n' 'Set CUPTI_ROOT to an extracted CUDA 12.x CUPTI SDK with include/ and lib/.' >&2
  exit 2
fi
cuda_root="${CUDA_HOME:-/opt/cuda}"
if [[ ! -f "$cuda_root/include/cuda.h" ]]; then
  printf '%s\n' 'Set CUDA_HOME to a CUDA 12.x toolkit with cuda.h.' >&2
  exit 2
fi
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/m40-cupti-probe.XXXXXXXX")"
trap 'rm -rf -- "$build_dir"' EXIT

"${CXX:-g++}" -std=c++17 -O2 -Wall -Wextra -Werror \
  -I"$cuda_root/include" -I"$CUPTI_ROOT/include" \
  "$script_dir/../examples/maxwell_counters.cpp" \
  -L"$CUPTI_ROOT/lib" -L"$cuda_root/lib64/stubs" \
  -Wl,-rpath,"$CUPTI_ROOT/lib" -lcupti -lcuda \
  -o "$build_dir/maxwell_counters"

"$build_dir/maxwell_counters" "$@"
