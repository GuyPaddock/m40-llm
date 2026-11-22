#!/usr/bin/env bash
# Recover complete-target statistics from paired target trials.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: bash replay-target-gate.sh MEASUREMENT_JSONL RESEARCH_HTTP_PAIRS" >&2
    exit 2
fi
raw=$(realpath "$1")
stats=$(realpath "$2")
jq -es 'last | .kind=="direct_verifier_complete" and .is_complete and .device_bytes_after==0' "$raw" > /dev/null

statistics() {
    local phase=$1 control=$2
    shift 2
    if [[ "$phase" == confirmation ]]; then set -- --confirmation; fi
    "$stats" "$@" <(
        jq -c --arg phase "$phase" --arg control "$control" '
          select(.kind=="down_target_trial" and (.phase==$phase or
              ($phase=="extended" and (.phase=="screen" or .phase=="extension")))) |
          select(.down_projection==$control or .down_projection=="shared_weights") |
          (if .down_projection==$control then 0 else 1 end) as $arm |
          (if $phase=="confirmation" then "confirm" else "pair" end) as $label |
          {label:"\($label)-\(.round)-\($arm)",arm:$arm,http_seconds:(1/.batches_per_second),
           metrics:{prompt_tokens:65,generated_tokens:1,stop_reason:"length"}}
        ' "$raw"
    )
}

{
    while IFS=$'\t' read -r phase control; do
        recovered=$(statistics "$phase" "$control")
        jq -ces --arg phase "$phase" --arg control "$control" --argjson a "$recovered" '
          def near($x;$y): (($x-$y)|fabs)<1e-12;
          [ .[] | select(.kind=="down_target_summary" and .phase==$phase and .control==$control)] |
          if length!=1 then error("missing or repeated target summary") else .[0] end |
          .rate_comparison as $reported |
          ($a.gain/(1+$a.gain)) as $saving |
          ([$a.control_tps.mad_over_median,$a.candidate_tps.mad_over_median]|max) as $variation |
          ([0.05,2*$variation]|max) as $threshold |
          if $a.pairs!=$reported.pairs or
             (near($a.gain;$reported.gain) and near($saving;.median_target_latency_saving_fraction) and
              near($threshold;.required_latency_saving_fraction) and
              all(range(2); . as $i | near($a.bootstrap_gain_ci95[$i];$reported.bootstrap_gain_ci95[$i])) | not)
          then error("target statistics differ from the measured trials") else
            {phase:$phase,control:$control,pairs:$a.pairs,median_latency_saving:$saving,
             threshold:$threshold,unstable:($variation>0.05),
             latency_saving_ci95:[$a.bootstrap_gain_ci95[]|./(1+.)],
             passed:($variation<=0.05 and $saving>=$threshold and $a.bootstrap_gain_ci95[0]>0)}
          end' "$raw"
    done < <(jq -r 'select(.kind=="down_target_summary") | [.phase,.control] | @tsv' "$raw")
} | jq -es --slurpfile raw "$raw" '
    [ $raw[] | select(.kind=="down_target_gate") ] as $g |
    last.phase as $phase |
    map(select(.phase==$phase)) as $final |
    if ($g|length)!=1 or ($final|map(.control)|sort)!=["multi_row","serial"] or
       ($final|all(.passed))!=$g[0].passes_complete_target_gate or
       ($final|any(.unstable))!=$g[0].unstable
    then error("complete-target decision differs")
    else {final_phase:$phase,gate:$g[0],summaries:.} end'
