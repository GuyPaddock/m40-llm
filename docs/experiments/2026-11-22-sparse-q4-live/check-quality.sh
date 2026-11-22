#!/usr/bin/env bash
# Build and check the four-arm quality variant while holding the M40 lock.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: bash check-quality.sh MODEL NEW_BUILD_REPORT_DIR" >&2
    exit 2
fi
source scripts/dev-env.sh
model=$(realpath "$1")
out=$(realpath -m "$2")
[[ -z $(git status --porcelain) ]]
mkdir "$out"
git rev-parse HEAD > "$out/source.txt"
export CUDA_VISIBLE_DEVICES=-1 M40LLM_ENABLE_NVCC=1
for mode in 1 0; do
    target="target/quality-cublas-$mode"
    export M40LLM_ENABLE_CUBLAS="$mode"
    nice -n 19 cargo test --release --locked --features cuda,server --lib --no-run \
        --target-dir "$target" --message-format=json > "$out/build-$mode.jsonl" 2> "$out/build-$mode.log"
    exe=$(jq -ser '[.[]|select(.reason=="compiler-artifact" and .target.name=="m40_llm"
        and .profile.test and .executable!=null)|.executable] |
        if length==1 then first else error("expected one library test executable") end' "$out/build-$mode.jsonl")
    sha256sum "$exe" > "$out/test-$mode.sha256"
    cuobjdump -lelf "$exe" | rg 'sm_52' > "$out/test-$mode.elf"
    CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852 \
        "$exe" cuda::q4_channels::repacked_tests:: --include-ignored --test-threads=1 --nocapture \
        > "$out/test-$mode.log" 2>&1
    rg -q '1 passed; 0 failed; 0 ignored' "$out/test-$mode.log"
done
export M40LLM_ENABLE_CUBLAS=1
target=target/quality-cublas-1
nice -n 19 cargo build --release --locked --features cuda,server --target-dir "$target" \
    --example model_quality --example sparse_q4_inputs > "$out/examples.log" 2>&1
sha256sum "$target/release/examples/model_quality" "$target/release/examples/sparse_q4_inputs" \
    > "$out/examples.sha256"
cuobjdump -lelf "$target/release/examples/model_quality" | rg 'sm_52' > "$out/model-quality.elf"
nice -n 19 "$target/release/examples/sparse_q4_inputs" "$model" "$out/planes" > "$out/planes.jsonl"
for layer in 0 17 35; do
    for plane in native codes scales zigzag; do
        cmp "$out/planes/$layer/$plane" "../raw/sparse-q4-inputs-01/planes/$layer/$plane"
    done
done
date --iso-8601=ns > "$out/complete.txt"
