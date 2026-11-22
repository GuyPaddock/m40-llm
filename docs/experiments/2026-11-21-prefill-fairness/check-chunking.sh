#!/usr/bin/env bash
# Execute the two pinned release test binaries under the common M40 lock.
set -euo pipefail
build=$(realpath "${1:?checked build directory}")
test -s "$build/complete.txt"
[[ -z $(git status --porcelain) ]]
git diff --quiet "$(cat "$build/source.txt")" HEAD -- src cuda examples Cargo.toml Cargo.lock build.rs
source "../consolidation/docs/experiments/2026-11-21-serving-baseline/server.sh"
clear_flags
check_gpu_owners
sha256sum target/release/m40-llm > "$build/server.sha256"
cuobjdump -lelf target/release/m40-llm | rg 'sm_52' > "$build/server.elf"
nvcc --version > "$build/nvcc.txt"
for mode in on off; do
    if [[ "$mode" == on ]]; then
        artifacts="$build/test-artifacts.jsonl"
        export M40LLM_ENABLE_CUBLAS=1
    else
        artifacts="$build/no-cublas-artifacts.jsonl"
        export M40LLM_ENABLE_CUBLAS=0
    fi
    exe=$(jq -ser '
        [.[] | select(.reason=="compiler-artifact" and .target.name=="m40_llm" and
            .profile.test and .executable!=null) | .executable] |
        if length==1 then .[0] else error("expected one library test binary") end
    ' "$artifacts")
    check_gpu_owners
    gpu_snapshot > "$build/cublas-$mode.before"
    sha256sum "$exe" > "$build/cublas-$mode.sha256"
    cuobjdump -lelf "$exe" | rg 'sm_52' > "$build/cublas-$mode.elf"
    "$exe" 'server::prefill_chunk_tests::' --include-ignored --nocapture --test-threads=1 > "$build/cublas-$mode.log" 2>&1
    rg -q '2 passed; 0 failed; 0 ignored' "$build/cublas-$mode.log"
    check_gpu_owners
    gpu_snapshot > "$build/cublas-$mode.after"
    date --iso-8601=ns > "$build/cublas-$mode.pass"
done
