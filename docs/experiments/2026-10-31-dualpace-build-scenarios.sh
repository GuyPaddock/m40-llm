#!/usr/bin/env bash
# Build portable trace inputs from complete component reports, never overwrite.
set -euo pipefail
if [[ $# != 5 ]]; then
    printf 'usage: %s CPU.jsonl TRANSFERS.jsonl GPU_COSTS.json CAPACITY.json NEW_DIRECTORY\n' "$0" >&2
    exit 2
fi
scenario_source_dir=$(cd "$(dirname "$0")" && pwd)
scenario_output=$5
[[ ! -e "$scenario_output" ]] || { printf 'output already exists\n' >&2; exit 1; }
scenario_tmp=$(mktemp -d "${scenario_output}.tmp.XXXXXX")
trap 'rm -rf -- "$scenario_tmp"' EXIT
mkdir "$scenario_tmp/sources"
cp -- "$1" "$scenario_tmp/sources/cpu.jsonl"
cp -- "$2" "$scenario_tmp/sources/transfers.jsonl"
cp -- "$3" "$scenario_tmp/sources/gpu-costs.json"
cp -- "$4" "$scenario_tmp/sources/capacity.json"
cp -- "$scenario_source_dir/2026-10-31-dualpace-scenarios.jq" "$scenario_tmp/sources/2026-10-31-dualpace-scenarios.jq"
cp -- "$0" "$scenario_tmp/sources/2026-10-31-dualpace-build-scenarios.sh"
scenario_cpu_digest=$(sha256sum "$scenario_tmp/sources/cpu.jsonl" | cut -d ' ' -f 1)
scenario_transfer_digest=$(sha256sum "$scenario_tmp/sources/transfers.jsonl" | cut -d ' ' -f 1)
scenario_gpu_digest=$(sha256sum "$scenario_tmp/sources/gpu-costs.json" | cut -d ' ' -f 1)
scenario_capacity_digest=$(sha256sum "$scenario_tmp/sources/capacity.json" | cut -d ' ' -f 1)
jq -n --slurpfile cpu "$scenario_tmp/sources/cpu.jsonl" \
    --slurpfile transfers "$scenario_tmp/sources/transfers.jsonl" \
    --slurpfile gpu "$scenario_tmp/sources/gpu-costs.json" \
    --slurpfile capacity "$scenario_tmp/sources/capacity.json" \
    --arg cpu_sha "$scenario_cpu_digest" --arg transfer_sha "$scenario_transfer_digest" \
    --arg gpu_sha "$scenario_gpu_digest" --arg capacity_sha "$scenario_capacity_digest" '
    if ($gpu|length) != 1 or ($capacity|length) != 1 then error("expected one GPU/capacity object") else . end |
    {cpu:$cpu,transfers:$transfers,gpu:$gpu[0],capacity:$capacity[0],evidence:{
      cpu_costs:{kind:"artifact",path:"sources/cpu.jsonl",sha256:$cpu_sha,
        description:"Single-core AVX2 projection trial wall medians, rounded up to integer ns; sampled layers extrapolated explicitly"},
      transfers:{kind:"artifact",path:"sources/transfers.jsonl",sha256:$transfer_sha,
        description:"Completed pageable copy trial wall medians; synchronization included; serialized against GPU work"},
      gpu_costs:{kind:"artifact",path:"sources/gpu-costs.json",sha256:$gpu_sha,
        description:"Retained native/prepared Q8 projection wall medians; 100-operation intervals amortize completion; not single-assist qualification"},
      capacity:{kind:"artifact",path:"sources/capacity.json",sha256:$capacity_sha,
        description:"Read-only NUMA/GPU capacity inventory; fixed workspace budget is an assumption"}}}' |
    jq -f "$scenario_tmp/sources/2026-10-31-dualpace-scenarios.jq" > "$scenario_tmp/scenarios.json"
jq -e '.is_simulation == true and .is_complete_decode_measured == false and (.scenarios|length) == 54' \
    "$scenario_tmp/scenarios.json" > /dev/null
while IFS= read -r scenario_name; do
    [[ "$scenario_name" =~ ^node[01]-(q8|prepared_q8)-cpu[014]-gpu[014]-(balanced|gpu_loaded|staggered)$ ]]
    jq --arg name "$scenario_name" '.scenarios[] | select(.name == $name) | .trace' \
        "$scenario_tmp/scenarios.json" > "$scenario_tmp/$scenario_name.json"
done < <(jq -r '.scenarios[].name' "$scenario_tmp/scenarios.json")
(
    cd "$scenario_tmp"
    sha256sum sources/* ./*.json > artifacts.sha256
)
# -n prevents replacing a destination created since initial admission. Check
# that the move occurred because GNU mv can return success when -n skips it.
mv -T -n -- "$scenario_tmp" "$scenario_output"
[[ ! -d "$scenario_tmp" ]] || { printf 'output appeared during generation\n' >&2; exit 1; }
trap - EXIT
printf 'Created 54 sourced scenario inputs in %s\n' "$scenario_output"
