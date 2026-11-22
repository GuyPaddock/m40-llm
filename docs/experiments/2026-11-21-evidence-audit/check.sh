#!/usr/bin/env bash
# No hardware access. Replay the archived result transcripts and source diff.
set -euo pipefail
cd "$(dirname "$0")"
while IFS=$'\t' read -r study source path archive bytes expected; do
    git cat-file -e "$source^{commit}"
    actual=$(gzip -dc "$archive" | sha256sum | cut -d' ' -f1)
    [[ "$actual" == "$expected" && $(gzip -dc "$archive" | wc -c) -eq "$bytes" ]]
done < <(tail -n +2 files.tsv)
for mode in 0 1; do
    file="graph-fixed/dispatch-$mode.log.gz"
    for observation in '8193 comparisons=84' '129 comparisons=144'; do
        gzip -dc "$file" | rg "capacity=$observation unequal_elements=0 guards_inputs_cleanup=pass" >/dev/null
    done
    gzip -dc "$file" | rg 'test result: ok. 3 passed; 0 failed; 0 ignored;' >/dev/null
    file="graph-fixed/session-$mode.log.gz"
    for path in 'None' 'Some((Capture, 0))' 'Some((Launch, 0))' 'Some((Launch, 3))'; do
        gzip -dc "$file" | rg -F "failure=$path compared_tokens=32 maximum_logit_error=0 cache_differences=0 maximum_cache_error=0 tracked_cleanup=pass" >/dev/null
    done
    gzip -dc "$file" | rg 'test result: ok. 5 passed; 0 failed; 0 ignored;' >/dev/null
    file="verifier/exact-$mode.log.gz"
    gzip -dc "$file" | rg 'test result: ok. 5 passed; 0 failed; 0 ignored;' >/dev/null
    gzip -dc "$file" | awk '/^near-tied Q8 head margin=/ {
        split($4,m,"="); if (!(m[2]+0>8e-9 && m[2]+0<1.2e-8 && $6=="id=1")) exit 1; ++n
    } END {if(n!=4) exit 1}'
done
for file in graph-unfixed/test_native*.log.gz; do
    mismatches=20
    if [[ "$file" == *graph_sequences.log.gz ]]; then mismatches=83; fi
    gzip -dc "$file" | rg "compared_tokens=32 .*cache_differences=$mismatches .*tracked_cleanup=pass" >/dev/null
    gzip -dc "$file" | rg 'test result: FAILED. 0 passed; 1 failed;' >/dev/null
done
repo=$(git rev-parse --show-toplevel)
git -C "$repo" diff a21f192 fc487e8 -- tests/exact_verification.rs | cmp - <(gzip -dc verifier/test-changes.patch.gz)
echo 'PASS: 20 transport payloads, 2 fixed graph builds, 4 failing controls, 2 verifier builds, and the measured source diff'
