#!/usr/bin/env bash
set -euo pipefail
out=$(realpath -m "$1")
manifest=$(realpath "$2")
models=$(realpath "$3")
mkdir "$out"
bin=target/release/examples/main_model_admission
{
    git rev-parse HEAD
    git status --porcelain
    sha256sum "$bin" "$manifest" "$models"
} > "$out/provenance.txt"
while read -r digest model; do
    args=()
    if [[ "$model" == *TinyLlama* ]]; then args+=(--tiny); fi
    printf '%s  %s\n' "$digest" "$model" | sha256sum -c - >> "$out/weights.log"
    "$bin" "$model" "$manifest" "${args[@]}" > "$out/$digest.json" 2>"$out/$digest.log"
done < "$models"
