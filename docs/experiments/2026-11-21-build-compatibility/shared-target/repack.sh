#!/usr/bin/env bash
set -euo pipefail
raw=$(realpath "${1:?original raw directory}")
out=$(realpath -m "${2:?new evidence directory}")
mkdir "$out"
printf 'run\tbranch\tsource\tpass\tfail\tignored\tarchive_sha256\n' > "$out/index.tsv"
for run in graph-dispatch quantized-kv q8-projections verifier-correctness sparse-q4-layout active-row-kv prefill-fairness kv-history-boundary consolidation graph-node-cost consolidation-02; do
    name=${run%-02}
    sequence=01
    if [[ "$run" == *-02 ]]; then sequence=02; fi
    d="$raw/$name-build-checks-$sequence"
    test -s "$d/complete.txt"
    diff -u <(printf 'format\t0\nnoncuda\t0\nclippy\t0\n') "$d/status.tsv"
    source_sha=$(sed -n '2p' "$d/preflight.txt")
    branch=$(sed -n '3p' "$d/preflight.txt")
    [[ "$source_sha" =~ ^[[:xdigit:]]{40}$ && "$branch" == "research/1121-$name" ]]
    counts=$(awk '/^test result: ok\./ {s+=$4; f+=$6; i+=$8; rows++} END{if(!rows||!s||f)exit 1;printf "%d\t%d\t%d",s,f,i}' "$d/noncuda.log")
    (cd "$d" && tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -cf - .) | gzip -n > "$out/$run.tar.gz"
    digest=$(sha256sum "$out/$run.tar.gz" | cut -f1 -d' ')
    printf '%s\t%s\t%s\t%s\t%s\n' "$run" "$branch" "$source_sha" "$counts" "$digest" >> "$out/index.tsv"
    gzip -t "$out/$run.tar.gz"
    tar -xOf "$out/$run.tar.gz" ./noncuda.log | cmp - "$d/noncuda.log"
done
