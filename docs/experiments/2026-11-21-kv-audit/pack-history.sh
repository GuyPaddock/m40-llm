#!/usr/bin/env bash
# Archive original M40 boundary readbacks only after independent replay and clean exit evidence.
set -euo pipefail
raw=$(realpath "${1:?completed boundary run}")
inputs=$(realpath "${2:?frozen boundary inputs}")
out=$(realpath -m "${3:?new evidence directory}")
audit=$(cd "$(dirname "$0")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
jq -es 'last|.kind=="complete" and .cases==4 and .generations==16 and .initial_bytes==0 and .final_bytes==0' "$raw/report.jsonl" >/dev/null
test -s "$raw/after.txt"
test ! -e "$out"
stage=$(mktemp -d "$(dirname "$out")/.kv-boundary-package-XXXXXXXX")
trap 'rm -r -- "$stage"' EXIT
package="$stage/package"
verify="$stage/verify"
mkdir "$package" "$verify"
cp "$inputs" "$stage/inputs.json"
python3 "$audit/history.py" "$raw/report.jsonl" "$stage/inputs.json" > "$package/replay.json"
(cd "$raw" && sha256sum preflight.txt after.txt host-load.log run.log report.jsonl report.logits/*.f32le) > "$package/raw.sha256"
(cd "$stage" && sha256sum inputs.json) >> "$package/raw.sha256"
tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -cf - \
    -C "$raw" preflight.txt after.txt host-load.log run.log report.jsonl report.logits \
    -C "$stage" inputs.json | gzip -n > "$package/measurements.tar.gz"
tar -xzf "$package/measurements.tar.gz" -C "$verify"
(cd "$verify" && sha256sum --quiet -c "$package/raw.sha256")
python3 "$audit/history.py" "$verify/report.jsonl" "$verify/inputs.json" > "$stage/extracted-replay.json"
cmp "$package/replay.json" "$stage/extracted-replay.json"
(cd "$package" && sha256sum measurements.tar.gz raw.sha256 replay.json > package.sha256)
mv -Tn "$package" "$out"
[[ ! -e "$package" ]]
