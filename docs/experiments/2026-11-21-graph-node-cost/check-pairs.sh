#!/usr/bin/env bash
# A common 2 us allocation effect and additional 3 us replay effect must separate.
set -euo pipefail
out=$(realpath -m "${1:?fresh test directory}")
mkdir "$out"
study=$(realpath "$(dirname "$0")")
for spec in "128 65" "7 65" "128 2048"; do
    read -r head length <<< "$spec"
    for cap in 8192 8193; do
        nodes=$((1 + (head==128 && cap==8193)))
        echo "capture head=$head length=$length capacity=$cap"
        echo "[graph-setup] capture_us=12.300 instantiate_us=21.000 nodes=$nodes query_status=0"
    done
done > "$out/setup.txt"
jq -nc '([[128,65],[7,65],[128,2048]][] as [$h,$n] |
   [8192,8193][] as $c |
   {kind:"verified",head_dim:$h,live_length:$n,capacity:$c,exact_outputs:($h*16)},
   (range(0;10) as $i | [false,true][] as $g |
    (10+(if $c==8193 then 2 else 0 end)+(if $c==8193 and $g and $h==128 then 3 else 0 end)) as $v |
    {kind:"timing",head_dim:$h,live_length:$n,capacity:$c,graph:$g,round:$i,
    timing:{iterations:100,wall_us:$v,event_span_us:$v,enqueue_us:$v}})),{kind:"complete",timings:120}' > "$out/fixture.jsonl"
python3 "$study/analyze.py" "$out/fixture.jsonl" "$out/setup.txt" > "$out/summary.json"
jq -e '(.captures|length)==6 and (.paired_measurements|length)==9 and
 all(.paired_measurements[]; .ordinary_capacity_effect.median_us==2 and
 .graph_minus_ordinary_effect.paired_ci95_us==(if .head==7 then [0,0] else [3,3] end))' "$out/summary.json"
sed 's/nodes=2/nodes=1/g' "$out/setup.txt" > "$out/wrong-nodes.txt"
if python3 "$study/analyze.py" "$out/fixture.jsonl" "$out/wrong-nodes.txt" > /dev/null 2> "$out/expected-rejection.log"; then exit 1; fi
rg -q 'unexpected graph structure' "$out/expected-rejection.log"
