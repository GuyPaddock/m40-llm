#!/usr/bin/env bash
# Preserve one completed source-specific compatibility run; print its TSV index row.
set -euo pipefail
run=$(realpath "${1:?completed run directory}")
out=$(realpath -m "${2:?archive directory}")
sha=$(sed -n '2p' "$run/preflight.txt")
branch=$(sed -n '3p' "$run/preflight.txt")
tree=${branch#research/1121-}
target=$(sed -n 's/^CARGO_TARGET_DIR=//p' "$run/preflight.txt")
[[ "$sha" =~ ^[0-9a-f]{40}$ && "$tree" =~ ^[a-z0-9-]+$ && "${target##*/}" == "$sha" ]]
git merge-base --is-ancestor "$sha" "$branch"
test -s "$run/complete.txt"
diff -u <(printf 'format\t0\nnoncuda\t0\nclippy\t0\n') "$run/status.tsv"
rg -F "($target/debug/deps/" "$run/noncuda.log" >/dev/null
for step in noncuda clippy; do
    rg "m40-llm v[0-9.]+ \(.*/$tree\)" "$run/$step.log" >/dev/null
done
counts=$(awk '/^test result: ok\./ {passed+=$4; failed+=$6; ignored+=$8; ++suites}
    END {if (!suites || !passed || failed) exit 1; printf "%d\t%d\t%d",passed,failed,ignored}' "$run/noncuda.log")
mkdir -p "$out"
archive="$out/$tree-$sha.tar.gz"
[[ ! -e "$archive" ]]
(cd "$run" && tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -cf - .) | gzip -n > "$archive"
for record in preflight.txt status.tsv format.log noncuda.log clippy.log; do
    tar -xOf "$archive" "./$record" | cmp - "$run/$record"
done
printf '%s\t%s\t%s\t%s\t%s\n' "$branch" "$sha" "$counts" "${archive##*/}" "$(sha256sum "$archive"|cut -d' ' -f1)"
