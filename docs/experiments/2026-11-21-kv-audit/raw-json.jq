# The task says "only JSON". Parse the complete returned text and compare it
# with the frozen expected object, including its keys and scalar value types.
($tasks[0]|INDEX(.id)) as $index |
[$arms[] as $arm | $index[] as $task | ["short","long"][] |
 [$arm,($task.id+"-"+.)]] | sort as $expected |
$rows | map(select(.kind=="kv_quality_generation" and (.arm as $a | $arms|index($a))!=null)) as $gens |
if (([$gens[]|[.arm,.case]]|sort)!=$expected) then error("incomplete or duplicated case set") else
 [$gens[] as $g | ($g.case|sub("-(short|long)$";"")) as $id | $index[$id] as $task |
  ($task.scoring.kind=="json_object") as $json |
  (if $json then (try (($g.output|fromjson)==$task.scoring.expected) catch false) else null end) as $raw |
  {case:$g.case,context:$g.context,arm:$g.arm,family:$task.family,frozen_pass:$g.score.passed,
   raw_json_matches:$raw,passes_with_raw_json:($g.score.passed and ($raw!=false))}] as $results |
 {cohorts:($results|group_by([.arm,.context])|map({arm:.[0].arm,context:.[0].context,
    total:length,frozen_passing:(map(select(.frozen_pass))|length),
    passing_with_raw_json:(map(select(.passes_with_raw_json))|length),
    accepted_only_after_normalization:map(select(.frozen_pass and .raw_json_matches==false)|.case)})),
  cases:$results}
end
