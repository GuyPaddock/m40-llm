#!/usr/bin/env bash
# Run from the worktree. Mirror the Maxwell options in build.rs.
set -eu
source scripts/dev-env.sh
probe=${1:-sparse_q4_probe}
case "$probe" in
    sparse_q4_probe|sparse_q4_live) ;;
    *) echo "unknown sparse Q4 probe: $probe" >&2; exit 2 ;;
esac
mkdir -p target/release/examples
nvcc -O3 -std=c++17 -gencode=arch=compute_52,code=sm_52 -allow-unsupported-compiler \
    -I"$M40LLM_SYSROOT/usr/include" -Xcompiler=-U_FORTIFY_SOURCE \
    -cudart=shared -Xlinker -rpath -Xlinker /opt/cuda/lib64 \
    "examples/$probe.cu" -o "target/release/examples/$probe"
