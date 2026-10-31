#!/usr/bin/env bash
# Check reported owner latencies against hand calculations and preserve rejects.
set -euo pipefail
[[ $# == 1 ]] || { printf 'usage: %s SIMULATOR_EXECUTABLE\n' "$0" >&2; exit 2; }
summary_simulator=$(realpath "$1")
summary_scripts=$(cd "$(dirname "$0")" && pwd)
summary_tmp=$(mktemp -d /tmp/dualpace-summary-check.XXXXXX)
trap 'rm -rf -- "$summary_tmp"' EXIT

# A retained one-token fixture distinguishes declared ownership in GPU-only mode.
gzip -dc "$summary_scripts/2026-10-31-dualpace-omitted-work-result.jsonl.gz" > "$summary_tmp/one.jsonl"
jq -s -f "$summary_scripts/2026-10-31-dualpace-summarize.jq" "$summary_tmp/one.jsonl" > "$summary_tmp/one-summary.json"
jq -e '.is_simulation and (.modes|map(.elapsed_ns)) == [40,20,34]
  and .local_assistance[0].saving_before_queueing_ns == 6
  and (.modes[]|select(.mode=="gpu_only")|.by_declared_owner[]|select(.owner=="cpu")|.first_decode_latency.p50_ns) == 40
  and (.modes[]|select(.mode=="assisted")|.by_declared_owner[]|select(.owner=="cpu")|.first_decode_latency.p50_ns) == 14
  and (.modes[]|select(.mode=="assisted")|.queueing[]|select(.resource=="gpu")|.total_wait_ns) == 14
  and (.modes|all(.by_declared_owner|all(.inter_token_latency==null)))' "$summary_tmp/one-summary.json" > /dev/null
printf '%s\n' 'PASS: declared ownership, six-ns local saving, queueing and absent ITL samples'

# Two tokens and a delayed CPU arrival require within-request ITL, excluding TTFT.
jq '.requests |= map(.token_profiles=["token","token"] | if .id=="cpu" then .ready_ns=5 else . end)' \
  "$summary_scripts/2026-10-31-dualpace-omitted-synthetic.json" > "$summary_tmp/two.json"
"$summary_simulator" --trace "$summary_tmp/two.json" --output "$summary_tmp/two.jsonl"
jq -s -f "$summary_scripts/2026-10-31-dualpace-summarize.jq" "$summary_tmp/two.jsonl" > "$summary_tmp/two-summary.json"
jq -e '.modes[]|select(.mode=="independent")|
  .elapsed_ns==45 and .simulated_tokens==4
  and (.by_declared_owner|all(.first_decode_latency.p50_ns==20
    and .inter_token_latency=={samples:1,p50_ns:20,p95_ns:20,p99_ns:20}))' \
  "$summary_tmp/two-summary.json" > /dev/null
printf '%s\n' 'PASS: delayed arrivals and per-owner ITL use recorded token completions'

# Rejected GPU-only capacity must remain visible; no ratio may use its absent time.
jq '.memory.gpu_capacity_bytes=400' "$summary_tmp/two.json" > "$summary_tmp/rejected.json"
"$summary_simulator" --trace "$summary_tmp/rejected.json" --output "$summary_tmp/rejected.jsonl"
jq -s -f "$summary_scripts/2026-10-31-dualpace-summarize.jq" "$summary_tmp/rejected.jsonl" > "$summary_tmp/rejected-summary.json"
jq -e '(.modes[]|select(.mode=="gpu_only")|.is_admitted==false and (.error|length)>0)
  and (.modes|all(.throughput_ratio_to_gpu_only==null))' "$summary_tmp/rejected-summary.json" > /dev/null
printf '%s\n' 'PASS: capacity rejection retained and absent baseline ratio stays null'

# Seven different ITLs exercise nearest-rank tails and exclusion of a long TTFT.
jq '.profiles.token as $stage |
  .profiles = ([1000,10,20,30,40,50,60,70]|to_entries|map(. as $cost |
    {key:("token_"+(.key|tostring)),value:($stage|.[1].cpu.duration_ns=$cost.value)})|from_entries)' \
  "$summary_tmp/two.json" > "$summary_tmp/varied-base.json"
jq '.requests=[.requests[]|select(.id=="cpu")|.ready_ns=0|
  .token_profiles=[range(0;8)|"token_"+(tostring)]]' \
  "$summary_tmp/varied-base.json" > "$summary_tmp/varied.json"
"$summary_simulator" --trace "$summary_tmp/varied.json" --output "$summary_tmp/varied.jsonl"
jq -s -f "$summary_scripts/2026-10-31-dualpace-summarize.jq" "$summary_tmp/varied.jsonl" > "$summary_tmp/varied-summary.json"
jq -e '.modes[]|select(.mode=="independent")|.by_declared_owner[0]|
  .first_decode_latency.p50_ns==1000
  and .inter_token_latency=={samples:7,p50_ns:40,p95_ns:70,p99_ns:70}' \
  "$summary_tmp/varied-summary.json" > /dev/null
printf '%s\n' 'PASS: seven varied ITLs retain nearest-rank median and tails, excluding 1000-ns TTFT'

# Malformed reports must not become apparently complete decision evidence.
for summary_mutation in '.[0:4]' '.[2]=.[1]' '.[4].admitted_scenarios=2' \
    '.[1].result.requests[0].completion_ns=[]'; do
  jq -s "$summary_mutation | .[]" "$summary_tmp/one.jsonl" > "$summary_tmp/bad.jsonl"
  if jq -s -f "$summary_scripts/2026-10-31-dualpace-summarize.jq" "$summary_tmp/bad.jsonl" \
      > "$summary_tmp/bad-summary.json" 2> "$summary_tmp/bad.log"; then
    printf 'FAIL: accepted invalid report\n' >&2
    exit 1
  fi
  [[ ! -s "$summary_tmp/bad-summary.json" ]]
done
printf '%s\n' 'PASS: incomplete, duplicate-mode, inconsistent admission and missing-completion reports reject'
