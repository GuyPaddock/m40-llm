#!/usr/bin/env bash
# Run from the worktree. Mirror the Maxwell options in build.rs.
set -eu
source scripts/dev-env.sh
mkdir -p target/release/examples
nvcc -O3 -std=c++17 -gencode=arch=compute_52,code=sm_52 -allow-unsupported-compiler \
    -I"$M40LLM_SYSROOT/usr/include" -Xcompiler=-U_FORTIFY_SOURCE \
    -cudart=shared -Xlinker -rpath -Xlinker /opt/cuda/lib64 \
    examples/sparse_q4_probe.cu -o target/release/examples/sparse_q4_probe
