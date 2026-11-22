#!/usr/bin/env bash
# Recover each recorded timing decision with the pinned statistics executable.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: bash verify-pair-statistics.sh COMPLETED_COHORT RESEARCH_HTTP_PAIRS" >&2
    exit 2
fi
raw=$(realpath "$1")
stats=$(realpath "$2")
[[ -s "$raw/complete.txt" ]]
identity=$(sha256sum "$stats" | cut -d ' ' -f 1)
awk -v identity="$identity" '
    $1==identity && $2~/\/research_http_pairs$/ { found=1 }
    END { exit !found }
' "$raw/preflight.txt"

if jq -e '.equivalent_work==false' "$raw/screen.json" > /dev/null; then
    [[ ! -e "$raw/extended.json" && ! -e "$raw/confirmation.json" ]]
    echo "No paired speed gate for unequal generated IDs."
    exit 0
fi

verify_pairs() {
    local count=$1 expected=$2
    "$stats" <(
        jq -c --argjson count "$count" '
            select(.label|test("^pair-[0-9]+-[01]$")) |
            select((.label|split("-")[1]|tonumber)<$count)
        ' "$raw/timings.jsonl"
    ) | diff -u "$expected" -
}

verify_pairs 5 "$raw/screen.json"
if [[ -e "$raw/extended.json" ]]; then
    verify_pairs 10 "$raw/extended.json"
fi
if [[ -e "$raw/confirmation.json" ]]; then
    "$stats" --confirmation "$raw/timings.jsonl" | diff -u "$raw/confirmation.json" -
fi
