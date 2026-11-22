#!/usr/bin/env bash
# Audit a finished single-request cohort, then publish a replayable archive.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "usage: bash archive-cohort.sh FINISHED_RAW_DIR NEW_PACKAGE_DIR RESEARCH_HTTP_PAIRS" >&2
    exit 2
fi
study=$(cd "$(dirname "$0")" && pwd)
raw=$(realpath "$1")
out=$(realpath -m "$2")
stats=$(realpath "$3")
[[ -s "$raw/complete.txt" && ! -e "$out" ]]
if [[ "$out/" == "$raw/"* ]]; then
    echo "Package must be outside the raw directory." >&2
    exit 2
fi

stage=$(mktemp -d "$(dirname "$out")/.serving-package.XXXXXX")
bash "$study/replay.sh" "$raw" > "$stage/replay.json"
bash "$study/verify-pair-statistics.sh" "$raw" "$stats"
if [[ -e "$raw/dispatch-replay.json" ]]; then
    python3 "$study/kv-trace.py" "$raw" > "$stage/dispatch.json"
    diff -u "$raw/dispatch-replay.json" "$stage/dispatch.json"
fi

tar -C "$raw" --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -cf - . |
    gzip -1n > "$stage/measurements.tar.gz"
mkdir "$stage/replay"
tar -xzf "$stage/measurements.tar.gz" -C "$stage/replay"
bash "$study/replay.sh" "$stage/replay" | diff -u "$stage/replay.json" -
bash "$study/verify-pair-statistics.sh" "$stage/replay" "$stats"
if [[ -e "$stage/dispatch.json" ]]; then
    python3 "$study/kv-trace.py" "$stage/replay" | diff -u "$stage/dispatch.json" -
fi
rm -r -- "$stage/replay"
(cd "$stage" && sha256sum measurements.tar.gz > archive.sha256)
[[ ! -e "$out" ]]
mv -T -- "$stage" "$out"
