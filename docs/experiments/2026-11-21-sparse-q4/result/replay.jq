def check($ok;$why): if $ok then . else error($why) end;
def median: sort | length as $n | (.[(($n-1)/2|floor)]+.[$n/2|floor])/2;
["dense_native","dense_zigzag","native_oracle","packed_oracle","zigzag_oracle","native_charged","packed_charged","zigzag_charged"] as $arms |
[0,17,35] as $layers | [0,1376,2752,5504,11008] as $counts |
map(select(.kind=="check")) as $checks | map(select(.kind=="trial")) as $timing |
check(last.kind=="complete" and (map(select(.kind=="complete"))|length)==1 and
    ($checks|map([.layer,.count,.arm])|sort)==([$layers[] as $l|$counts[] as $n|$arms[]|[$l,$n,.]]|sort) and
    all($checks[];.max_error>=0 and .max_layout_delta>=0);"missing or duplicate numerical checks") |
check(($timing|map([.layer,.count,.round,.order,.arm]))==
    [$layers[] as $l|$counts[1:][] as $n|range(-1;5) as $r|range(8) as $i|[$l,$n,$r,$i,$arms[($i+$r+1)%8]]] and
    all($timing[];.repeats==20 and .wall_us>0 and .submit_us>0 and .event_us>0);"incorrect rotation, durations or trial coverage") |
map(select(.kind=="residency" or .kind=="released")) as $memory |
check(($memory|map([.kind,.layer]))==[$layers[] as $l|("residency","released")|[.,$l]];"missing allocation observations") |
[$timing[]|select(.round>=0)] | group_by([.layer,.count]) |
map(first.layer as $l | first.count as $n |
  (group_by(.arm)|map({key:first.arm,value:{wall:map(.wall_us)|median,event:map(.event_us)|median,submit:map(.submit_us)|median}})|from_entries) as $cost |
  {layer:$l,active:$n,us:$cost,
   packed_reduction_vs_dense_native:(1-$cost.packed_charged.wall/$cost.dense_native.wall),
   zigzag_reduction_vs_dense_native:(1-$cost.zigzag_charged.wall/$cost.dense_native.wall),
   zigzag_reduction_vs_dense_zigzag:(1-$cost.zigzag_charged.wall/$cost.dense_zigzag.wall)}) as $groups |
{full_vector_checks:($checks|length),maximum_ref_error:($checks|map(.max_error)|max),
 maximum_regrouping_delta:($checks|map(.max_layout_delta)|max),memory:$memory,case_groups:$groups}
