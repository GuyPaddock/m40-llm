#!/usr/bin/env bash
# Package the complete November broad grid into one replayable archive per case.
set -euo pipefail
raw=$(realpath "${1:?completed kv_quality output directory}")
inputs=$(realpath "${2:?frozen November tokenized manifest}")
out=$(realpath -m "${3:?new destination outside the worktree}")
audit=$(cd "$(dirname "$0")" && pwd)
corpus="$audit/../../../data/research_quality/2026-11-21-kv-tasks.json"
jq -es 'last|.kind=="kv_quality_complete" and .cases==48 and .generation_rows==144' "$raw/report.jsonl" >/dev/null
mkdir "$out"
mkdir "$out/vectors"
python3 "$audit/replay.py" "$raw/report.jsonl" "$inputs" > "$out/independent-replay.json"
jq -s -L "$audit" --argjson dense_only false --slurpfile inputs "$inputs" \
    -f "$audit/generation-audit.jq" "$raw/report.jsonl" > "$out/generation-audit.json"
jq -n --slurpfile rows "$raw/report.jsonl" --slurpfile tasks "$corpus" \
    --argjson arms '["dense","quantized_all","quantized_top8"]' -f "$audit/raw-json.jq" > "$out/answer-format.json"
cp "$raw/preflight.txt" "$out/"
gzip -nc "$inputs" > "$out/inputs.json.gz"
for item in report.jsonl run.log host-load.log; do gzip -nc "$raw/$item" > "$out/$item.gz"; done
(cd "$raw" && sha256sum report.jsonl run.log host-load.log preflight.txt report.logits/*.f32le) > "$out/raw.sha256"
(cd "$(dirname "$inputs")" && sha256sum "$(basename "$inputs")") | awk '{print $1 "  inputs.json"}' >> "$out/raw.sha256"
mapfile -t cases < <(jq -r '.cases[].id' "$inputs")
for id in "${cases[@]}"; do
    [[ "$id" =~ ^nov-[a-z]+-(short|long)$ ]]
    # Store original little-endian FP32 bytes, including the dense reference.
    (cd "$raw/report.logits" && tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -cf - "$id"-*.f32le) |
        zstd -q -T1 -1 -o "$out/vectors/$id.tar.zst"
done
(cd "$out" && sha256sum *.gz *.json *.txt raw.sha256 vectors/*.tar.zst) > "$out/bundle.sha256"
