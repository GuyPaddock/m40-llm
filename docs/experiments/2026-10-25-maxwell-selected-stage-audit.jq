def median: sort | if length%2==1 then .[length/2|floor] else (.[length/2-1]+.[length/2])/2 end;
def require($condition;$message): if $condition then . else error($message) end;
sort_by(.session_id,.generated_step)|group_by(.session_id) as $sessions |
require(($sessions|length)==7 and all($sessions[]; map(.generated_step)==[0,255,511]);"wrong profile session coverage") |
($sessions[1:-1]|add) as $primary |
require(all($primary[];.event_pool_capacity>0 and (.stages|length)>0 and
  all(.stages[];.gpu_elapsed_ms>=0));"invalid stage timings") |
{profile_sessions:7,primary_sessions:5,primary_samples:($primary|length),
 mapping:"first session is warmup, last is actual-ID replay; middle five follow the serial profile-on requests",
 by_step:($primary|group_by(.generated_step)|map(. as $samples |
  {generated_step:.[0].generated_step,samples:length,
   median_stage_sum_ms:map(.gpu_stage_sum_ms)|median,
   median_token_wall_ms:map(.token_wall_ms)|median,
   stages:([.[]|.gpu_stage_sum_ms as $total|.stages|group_by(.stage)|map({stage:.[0].stage,
     gpu_ms:(map(.gpu_elapsed_ms)|add),share:((map(.gpu_elapsed_ms)|add)/$total)})|.[]]|
     group_by(.stage)|map({stage:.[0].stage,median_ms:map(.gpu_ms)|median,median_share:map(.share)|median})|sort_by(.median_ms)|reverse)}))}
