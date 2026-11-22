#!/usr/bin/env bash
# Check transport hashes, recover original files and verify their source bytes.
set -euo pipefail
bundle=$(realpath "${1:?archived November KV package}")
dest=$(realpath -m "${2:?fresh replay directory}")
(cd "$bundle" && sha256sum --quiet -c bundle.sha256)
mkdir "$dest"
mkdir "$dest/report.logits"
cp "$bundle/preflight.txt" "$dest/"
for file in inputs.json report.jsonl run.log host-load.log; do
    gzip -dc "$bundle/$file.gz" > "$dest/$file"
done
for archive in "$bundle"/vectors/*.tar.zst; do
    zstd -qdc "$archive" | tar -xf - -C "$dest/report.logits" --no-same-owner
done
(cd "$dest" && sha256sum --quiet -c "$bundle/raw.sha256")
