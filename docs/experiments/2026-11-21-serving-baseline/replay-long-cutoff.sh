#!/usr/bin/env bash
# Recheck the completed default diagnostics and the bounded dense attempt.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo 'usage: bash replay-long-cutoff.sh RAW_DIR CUTOFF_DIR QUEUE_EXITS' >&2
    exit 2
fi
raw=$(realpath "$1")
cutoff=$(realpath "$2")
study=$(cd "$(dirname "$0")" && pwd)
rg -xq 'qwen-long 143' "$3"
[[ ! -e "$raw/complete.txt" && ! -e "$raw/timings.jsonl" ]]
rg -xq 'Command terminated by signal 15' "$raw/cli-1.time"
rg -xq 'request_limit_seconds=3600' "$cutoff/watcher.txt"
jq -e --slurpfile ids "$raw/phase4-prompt-ids.json" '
    .prompt_token_count==8197 and .context==8448 and .prompt_ids==$ids[0].ids
' "$raw/case.json" >/dev/null
cmp "$raw/cli-ids-0.json" "$raw/ids-0.json"
jq -es '
    map(.records[0]) as $records |
    $records[0].response.metrics as $m |
    ($records|length)==3 and all($records[]; .status==200) and
    $m.prompt_tokens==8197 and $m.generated_tokens==128 and $m.stop_reason=="length" and
    all($records[1:][]; (.arrival_ms|length)==128 and .output==$records[0].response.output)
' "$raw"/{calibrate-0,stream-warm-0,stream-measured-0}.json >/dev/null
for stream in stderr stdout; do
    expected=$(awk -v name="cli-1.$stream" '$2~("/" name "$") {print $1}' "$cutoff/cutoff.txt")
    [[ "$expected" == "$(sha256sum "$raw/cli-1.$stream" | cut -d ' ' -f 1)" ]]
done
sampled=$(sed -n 's/^sampled_ids=//p' "$cutoff/cutoff.txt")
elapsed=$(sed -n 's/^elapsed_ns=//p' "$cutoff/cutoff.txt")
pid=$(sed -n 's/^own_process_pid=//p' "$cutoff/cutoff.txt")
[[ "$pid" == "$(sed -n 's/^pid=//p' "$cutoff/watcher.txt")" ]]
[[ "$sampled" == "$(awk '/sampled token id=/{++n} END {print n+0}' "$raw/cli-1.stderr")" ]]
(( elapsed >= 3600 * 1000000000 && sampled < 128 ))
peak_default=$(awk -F, -v a="$(cat "$raw/trace-0.pid")" -v b="$(cat "$raw/stream-0.pid")" '
    ($2+0==a || $2+0==b) && $3 ~ / MiB/ { value=$3+0; if(value>peak) peak=value }
    END { if(!peak)exit 1; print peak }
' "$raw/processes.csv")
peak_dense=$(awk -F, -v pid="$pid" '
    $2+0==pid && $3 ~ / MiB/ { value=$3+0; if(value>peak) peak=value }
    END { if(!peak)exit 1; print peak }
' "$raw/processes.csv")
diagnostic=$(jq -enc --argjson arm 0 --slurpfile trace "$raw/calibrate-0.json" \
    --slurpfile ids "$raw/ids-0.json" --slurpfile stream "$raw/stream-measured-0.json" \
    --rawfile cli "$raw/cli-0.time" -f "$study/diagnostic.jq")
jq -n --argjson diagnostic "$diagnostic" --argjson sampled "$sampled" --argjson elapsed "$elapsed" \
    --argjson default_peak "$peak_default" --argjson dense_peak "$peak_dense" '
    {default_diagnostics:$diagnostic,default_max_sampled_server_peak_mib:$default_peak,
     dense_diagnostic:{sampled_ids:$sampled,stop_snapshot_elapsed_s:($elapsed/1000000000),
                       sampled_process_peak_mib:$dense_peak,process_status:143},
     completed_paired_trials:0}'
