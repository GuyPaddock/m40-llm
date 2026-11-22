#!/usr/bin/env bash
# Capture a low-frequency host timeline in the benchmark's own host namespace.
# Supply the parent benchmark shell PID. Run at the same cadence for both arms.
set -eu
benchmark_pid=$1
while kill -0 "$benchmark_pid" 2>/dev/null; do
    date --iso-8601=ns
    cat /proc/loadavg
    awk '/^cpu / {print; exit}' /proc/stat
    for resource in cpu io memory; do
        if [[ -r "/proc/pressure/$resource" ]]; then
            sed "s/^/$resource /" "/proc/pressure/$resource"
        fi
    done
    ps -eo comm= | awk '/^(emerge|ebuild|cc1|cc1plus|rustc|ld|ld.lld|ninja|make)$/ {n[$1]++} END {for (name in n) print name,n[name]}'
    sleep 5
done
