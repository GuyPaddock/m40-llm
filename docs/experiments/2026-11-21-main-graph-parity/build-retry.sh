#!/usr/bin/env bash
# Run with scripts/dev-env.sh sourced, from the main-graph-parity worktree.
set -euo pipefail
out=$(realpath -m "${1:?new artifact directory}")
fixed=$(pwd)
control=$(realpath ../main-graph-control)
test -z "$(git status --porcelain)"
test -z "$(git -C "$control" status --porcelain)"
for path in tests/attention_graph_dispatch.rs src/decode_session/main_graph_tests.rs; do
    cmp "$fixed/$path" "$control/$path"
done
mkdir "$out"
export CUDA_VISIBLE_DEVICES=-1 M40LLM_ENABLE_NVCC=1 CARGO_BUILD_JOBS=3

build() (
    local checkout=$1 label=$2 blas=$3
    cd "$checkout"
    export M40LLM_ENABLE_CUBLAS="$blas"
    rev=$(git rev-parse HEAD)
    printf '%s\t%s\t%s\n' "$label" "$rev" "$blas" >> "$out/sources.tsv"
    cargo test --release --locked --features cuda --lib --test attention_graph_dispatch \
        --no-run --target-dir "target/retry-$label" --message-format=json \
        > "$out/$label.jsonl" 2> "$out/$label.log"
    for entry in grid session; do
        if [[ "$entry" == grid ]]; then target=attention_graph_dispatch; else target=m40_llm; fi
        exe=$(jq -ser --arg target "$target" '
            [.[] | select(.reason=="compiler-artifact" and .target.name==$target and
                .profile.test and .executable!=null) | .executable] |
            if length==1 then .[0] else error("missing release test artifact") end
        ' "$out/$label.jsonl")
        cp "$exe" "$out/$label-$entry"
    done
)

build "$control" unfixed-on 1
build "$fixed" fixed-on 1
build "$fixed" fixed-off 0
(cd "$out" && sha256sum {unfixed-on,fixed-on,fixed-off}-{grid,session} > binaries.sha256)
date --iso-8601=ns > "$out/complete.txt"
