# Reconstruct every reported median from the warm alternating pairs,
# including five/ten screens and separate twenty-pair confirmation.
def demand($condition; $why): if $condition then . else error($why) end;
def median: sort | length as $n | (.[(($n-1)/2|floor)]+.[($n/2|floor)])/2;
def rate($pairs; $key; $arm): [$pairs[]|1000/.[$key][$arm]]|median;
def agrees($a; $b): ($a-$b|fabs)<1e-8;
def check_group:
  map(select(.kind=="kv_attention_pair")) as $pairs |
  map(select(.kind=="kv_attention_comparison")) as $summaries |
  demand(($summaries|length)>0 and ($pairs|length)==$summaries[-1].wall_comparison.pairs;
         "incomplete pair group") |
  demand(($pairs|map(.pair))==[range($pairs|length)] and
    ($pairs|all(.iterations>0 and .order==(if .pair%2==0 then [0,1] else [1,0] end) and
      (.wall_ms_per_operation|length==2 and all(.>0)) and
      (.gpu_ms_per_operation|length==2 and all(.>0)) and
      (.wrapper_counters|length==2 and all(.device_allocs==0 and .device_frees==0 and .d2h_copies==0 and .h2d_copies==0))));
    "pair order, timing, or ownership violation") |
  [$summaries[] | . as $s | $pairs[0:.wall_comparison.pairs] as $p |
    demand((($p|length)==5 or ($p|length)==10 or ($p|length)==20) and
      agrees(rate($p;"wall_ms_per_operation";0); $s.wall_comparison.control.median) and
      agrees(rate($p;"wall_ms_per_operation";1); $s.wall_comparison.candidate.median) and
      agrees(rate($p;"gpu_ms_per_operation";0); $s.gpu_control.median) and
      agrees(rate($p;"gpu_ms_per_operation";1); $s.gpu_candidate.median) and
      ($s.coverage_matches==($s.comparison!="dense_context_only")) and $s.is_model_speedup==false;
      "reported median or coverage disagrees with raw pairs") | true] |
  {pairs:($pairs|length), summaries:length};

demand(last.kind=="kv_attention_benchmark_complete" and last.cases==36; "incomplete hardware report") |
map(select(.kind=="kv_attention_pair" or .kind=="kv_attention_comparison")) |
group_by([.case,.comparison,.is_confirmation]) | map(check_group) |
{screen_and_confirmation_groups:length,reported_summaries:(map(.summaries)|add),reconstructed_pairs:(map(.pairs)|add)}
