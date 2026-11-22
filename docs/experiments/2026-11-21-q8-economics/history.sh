#!/usr/bin/env bash
# From the q8-projections worktree. Reuse the five frozen historical HTTP cohorts.
set -euo pipefail
study=$(cd "$(dirname "$0")" && pwd)
root=$(realpath "$study/..")
out=$(realpath -m "${1:?fresh output directory}")
mkdir "$out"
for prefix in 2026-10-25-maxwell-{selected,split,projection}-stage 2026-10-26-maxwell-packed-stage 2026-10-27-maxwell-compound-stage; do
    events="$root/$prefix-events.jsonl"
    http="$root/$prefix-http.jsonl"
    jq -se '
        [.[]|select(.kind=="trial" and (.phase=="profile-on" or .phase=="profile-off"))] as $trials |
        [.[]|select(.kind=="profile_token_comparison")] as $ids |
        (.[0].model_sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d") and
        (.[0].gpu_uuid=="GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852") and
        ($ids|length)==1 and $ids[0].actual_ids_equal and $ids[0].matches_completed_quality and $ids[0].actual_generated_tokens==512 and
        ($trials|length)==10 and ($trials|group_by(.pair)|length)==5 and
        ($trials|group_by(.pair)|all(.[]; (map(.stage_profile_enabled)|sort)==[false,true])) and
        all($trials[]; .matches_completed_quality_output and .metrics.prompt_tokens==65 and
            .metrics.generated_tokens==512 and .metrics.stop_reason=="length")' "$http" >/dev/null
    jq -se 'length==21 and (map(.session_id)|unique)==[1,2,3,4,5,6,7] and
        (group_by(.session_id)|all(.[]; map(.generated_step)==[0,255,511]))' "$events" >/dev/null
    # Session 1 is warmup; session 7 is a separate ID replay.
    jq -c 'select(.session_id>=2 and .session_id<=6)' "$events" |
        jq -sc -f "$study/layer-cost.jq" | gzip -n > "$out/$prefix.json.gz"
    jq -sc --arg file "$prefix" --arg events "$(sha256sum "$events"|cut -d" " -f1)" --arg http "$(sha256sum "$http"|cut -d" " -f1)" '
        def med: sort|.[2];
        [.[]|select(.kind=="trial" and .phase=="profile-on")|.e2e_tps] as $on |
        [.[]|select(.kind=="trial" and .phase=="profile-off")|.e2e_tps] as $off |
        {prefix:$file,revision:.[0].revision,exe:.[0].benchmark_executable_sha256,
         worktree:.[0].worktree,config:.[0].primary_runtime_config,events_sha256:$events,http_sha256:$http,
         profile_on_tps:($on|med),profile_off_tps:($off|med)}' "$http" >> "$out/cohorts.jsonl"
done
