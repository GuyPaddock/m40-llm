# Inputs in order: diagnostic single-short, single-medium, control, treatment.
def check($ok; $message): if $ok then . else error($message) end;
def signature: .response | [.output, .metrics.stop_reason, .metrics.sampled_ids];
def samples($epoch):
  .response.metrics as $m |
  (($m.sample_origin_unix_ns - $epoch) / 1e6) as $origin |
  {origin_ms:$origin, times:[$m.token_elapsed_ms[] | . + $origin]} |
  check((.times|length)==$m.generated_tokens and (.times|length)>1 and
        (.times|sort)==.times and ($m.sampled_ids|length)==$m.generated_tokens;
        "invalid server sample count or order");
def observation($short; $other):
  check(.success and (.records|length)==2 and [.records[].slot]==[0,1]; "expected two completed clients") |
  .records as $r | (.client_origin_unix_ns|tonumber) as $epoch |
  ($r[0]|samples($epoch)) as $a | ($r[1]|samples($epoch)) as $b |
  $r[1].submitted_ms as $arrival |
  check($a.origin_ms+1 >= $r[0].submitted_ms and $b.origin_ms+1 >= $arrival and
        $a.times[-1] <= $r[0].done_ms+1 and $b.times[-1] <= $r[1].done_ms+1;
        "server/client clock inversion") |
  [range(1; $a.times|length) as $i |
    [$a.times[$i-1], $a.times[$i]] as $gap |
    select($gap[0] < $b.times[0] and $gap[1] > $b.origin_ms) |
    {from_output:($i-1), to_output:$i, gap_ms:($gap[1]-$gap[0]),
     overlapping_ms:(([$gap[1],$b.times[0]]|min)-([$gap[0],$b.origin_ms]|max))}] as $gaps |
  {matches_isolated:[($r[0]|signature)==($short|signature), ($r[1]|signature)==($other|signature)],
   request_wall_ms:[$r[].wall_ms], second_arrival_ms:$arrival,
   second_admission_delay_ms:($b.origin_ms-$arrival),
   short_samples_before_arrival:([$a.times[]|select(.<$arrival)]|length),
   arrived_during_decode:($arrival>$a.times[0] and $arrival<$a.times[-1]),
   largest_overlapping_gap:($gaps|max_by(.gap_ms))};
check(length==4 and all(.[]; .success) and
      (.[0].records|length)==1 and (.[1].records|length)==1 and
      ([.[0,1].records[0].response.metrics.prompt_tokens]==[65,2060]) and
      ([.[2,3].records[].response.metrics.prompt_tokens]==[65,65,65,2060]) and
      (.[2].offsets_ms==.[3].offsets_ms); "expected matched study inputs") |
.[0].records[0] as $short | .[1].records[0] as $medium |
{control:(.[2]|observation($short;$short)), treatment:(.[3]|observation($short;$medium))}
