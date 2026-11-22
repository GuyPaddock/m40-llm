#!/usr/bin/env bash
# Run from a research worktree with no in-progress edits.
set -euo pipefail
out=$(realpath -m "${1:?fresh results directory}")
export CARGO_TARGET_DIR=$(realpath -m "${2:?shared compatibility target directory}")
mkdir "$out"
source scripts/dev-env.sh
[[ -z $(git status --porcelain) ]]
export CARGO_BUILD_JOBS=3 CUDA_VISIBLE_DEVICES="" CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0
{
    date --iso-8601=ns
    git rev-parse HEAD
    git branch --show-current
    git status --porcelain
    rustc --version --verbose
    cargo --version
    nvcc --version
    env | rg '^(CARGO_(BUILD_JOBS|TARGET_DIR|PROFILE_)|CUDA_VISIBLE_DEVICES)'
} > "$out/preflight.txt"
check() {
    local check=$1 rc=0
    shift
    printf '%q ' "$@" > "$out/$check.command"
    echo >> "$out/$check.command"
    nice -n 19 "$@" > "$out/$check.log" 2>&1 || rc=$?
    printf '%s\t%d\n' "$check" "$rc" >> "$out/status.tsv"
}
check format cargo fmt --all -- --check
check noncuda env M40LLM_ENABLE_NVCC=0 M40LLM_ENABLE_CUBLAS=0 RUSTFLAGS="-D warnings" cargo test --no-default-features --locked -- --test-threads=1
check clippy env M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 cargo clippy --locked --features cuda,server --all-targets -- -D warnings
date --iso-8601=ns > "$out/complete.txt"
awk -F '\t' '$2!=0 {failed=1} END {exit failed}' "$out/status.tsv"
