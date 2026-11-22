#!/usr/bin/env bash
# Two clients with overlapping work have a 40 ms makespan and 100 total tokens/s.
set -euo pipefail
out=$(realpath -m "${1:?fresh output directory}")
mkdir "$out"
study=$(realpath "$(dirname "$0")")
report=$(realpath target/cpu/release/examples/research_fairness_report)
jq -n '{success:true,client_origin_unix_ns:"1000000000",
  records:[range(0;2) as $i |{slot:$i,status:200,submitted_ms:($i*4),done_ms:(30+$i*10),
    response:{output:"test",metrics:{prompt_tokens:65,generated_tokens:2,stop_reason:"length",
    prefill_elapsed_ms:4,decode_elapsed_ms:8,total_elapsed_ms:12,
    sample_origin_unix_ns:(1000000000+$i*5000000),token_elapsed_ms:[4,12],sampled_ids:[12,34]}}}]}' > "$out/trace.json"
jq 'del(.records[].response.metrics|.sample_origin_unix_ns,.token_elapsed_ms,.sampled_ids)' "$out/trace.json" > "$out/primary.json"
"$report" "$out" --client-run trace --diagnostic > "$out/trace.summary.json"
"$report" "$out" --client-run primary > "$out/primary.summary.json"
jq -e '.wall_ms==40 and .aggregate_tps==100 and .requests[1].client_to_first_sample_ms==5 and .requests[0].itl_p95_ms==8' "$out/trace.summary.json"
audit() {
    jq -ce --arg label pair-0-0 --argjson arm 0 --argjson count 2 --slurpfile diagnostic "$out/trace.json" \
       --slurpfile summary "$out/primary.summary.json" -f "$study/observe.jq" "$1"
}
audit "$out/primary.json" | jq -e '.matches_companion and .http_seconds==0.04 and .metrics.generated_tokens==4 and .metrics.prompt_tokens==130'
jq '.records[1].response.output="changed"' "$out/primary.json" > "$out/diverged.json"
audit "$out/diverged.json" | jq -e '.matches_companion==false'
jq '.records[0].response.metrics.generated_tokens=0' "$out/primary.json" > "$out/invalid.json"
if audit "$out/invalid.json"; then echo 'invalid token count accepted' >&2; exit 1; fi
echo 'two-client makespan, diagnostic TTFT/ITL, divergence and invalid counts checked'
