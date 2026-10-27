#!/usr/bin/env bash
# Exercise output, timestamp/launch correlation and explicit observer opt-in.
# Requires an idle sm_52 GPU and the configured CUDA/CUPTI SDKs.
set -euo pipefail
if [[ $# != 1 || $1 != /* || -e $1 ]]; then
  printf 'Usage: %s /absolute/new-result-directory\n' "$0" >&2
  exit 1
fi
directory=$1
mkdir "$directory"
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fixture="$root/target/research-tools/cuda-launch-trace-smoke"
[[ -x $fixture ]]
# Protect nanosecond subtraction above the exact-integer range of JSON doubles.
printf '%s\n' \
  '{"kind":"cuda_kernel_activity","start_ns":1793061468773531615,"end_ns":1793061468773531632}' \
  '{"kind":"cuda_kernel_activity","start_ns":1793061468773531600,"end_ns":1793061468773531605}' \
  '{"kind":"cuda_trace_complete","launch_records":2}' \
  | perl "$root/scripts/normalize_cuda_activity.pl" > "$directory/normalized.jsonl"
jq -se 'length==3 and .[0].duration_ns==17 and .[0].relative_start_ns==0
  and .[0].relative_end_ns==17 and .[0].start_ns=="1793061468773531615"
  and .[1].duration_ns==5 and .[1].relative_start_ns== -15
  and .[2]=={kind:"cuda_trace_complete",launch_records:2}' "$directory/normalized.jsonl" > /dev/null
for timestamps in '0,1' '17,17' '19,18' '9223372036854775806,9223372036854775808'; do
  if printf '{"kind":"cuda_kernel_activity","start_ns":%s,"end_ns":%s}\n' \
      "${timestamps%,*}" "${timestamps#*,}" \
      | perl "$root/scripts/normalize_cuda_activity.pl" > /dev/null 2> "$directory/rejected-clock.log"; then
    printf 'Invalid clock range accepted: %s\n' "$timestamps" >&2
    exit 1
  fi
done
for mode in unset zero enabled; do
  environment=(env -u LD_PRELOAD -u M40LLM_CUDA_TRACE_ACTIVITY M40LLM_CUDA_TRACE_MODULES=0)
  if [[ $mode == zero ]]; then environment+=(M40LLM_CUDA_TRACE_ACTIVITY=0); fi
  if [[ $mode == enabled ]]; then environment+=(M40LLM_CUDA_TRACE_ACTIVITY=1); fi
  "${environment[@]}" bash "$root/scripts/trace_cuda_launches.sh" "$directory/$mode" -- "$fixture" > "$directory/$mode-output.json"
  jq -e '.actual==96 and .expected==96 and .is_pass' "$directory/$mode-output.json" > /dev/null
  jq -se --arg mode "$mode" '
    . as $records |
    map(select(.kind=="cuda_trace_start")) as $start |
    $records|map(select(.kind=="cuda_trace_complete")) as $done |
    $records|map(select(.kind=="cuda_kernel_activity")) as $gpu |
    $records|map(select(.kind=="cuda_activity_complete")) as $activity_done |
    $records|map(select(.kind=="cuda_launch" and .domain=="driver" and .launch_status==0)) as $launches |
    ($start|length)==1 and ($done|length)==1
    and $done[0].launch_records==5 and $done[0].failed_launches==1
    and $done[0].dropped_launches==0 and $done[0].unknown_symbols==0 and $done[0].unsubscribe_status==0
    and ($launches|length)==2
    and if $mode=="enabled" then
      $start[0].captures_kernel_activity and ($gpu|length)==2 and ($activity_done|length)==1
      and $activity_done[0].kernel_records==2 and $activity_done[0].dropped_records==0
      and $activity_done[0].invalid_timestamps==0
      and all($gpu[];. as $kernel | .has_valid_timestamps and .start_ns>0 and .end_ns>.start_ns
        and .symbol=="m40_trace_increment"
        and any($launches[];.context_id==$kernel.context_id and .correlation_id==$kernel.correlation_id
          and .symbol==$kernel.symbol and .grid==$kernel.grid and .block==$kernel.block
          and .shared_bytes==$kernel.dynamic_shared_bytes))
      and ($gpu|map(.correlation_id)|unique|length)==2
    else ($start[0].captures_kernel_activity|not) and ($gpu|length)==0 and ($activity_done|length)==0 end
  ' "$directory/$mode".*.jsonl > /dev/null
done
set +e
env -u LD_PRELOAD M40LLM_CUDA_TRACE_MODULES=0 M40LLM_CUDA_TRACE_ACTIVITY=invalid \
  bash "$root/scripts/trace_cuda_launches.sh" "$directory/invalid" -- /usr/bin/true \
  > "$directory/invalid.log" 2>&1
status=$?
set -e
[[ $status == 2 ]]
jq -se 'length==1 and .[0].kind=="cuda_module_error" and
  .[0].message=="M40LLM_CUDA_TRACE_ACTIVITY must be 0 or 1"' "$directory/invalid".*.jsonl > /dev/null
printf 'CUDA activity output/correlation and unset/zero/invalid controls pass.\n'
