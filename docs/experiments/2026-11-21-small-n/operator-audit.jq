# Per-call service is unknown. GPU events and host completion each have an independent comparison.
def arms: ["serial","existing_multi_row","shared","native","sgemm","dequant_sgemm"];
def widths: [1,2,3,4,8,16];
def tensors: [0,17,35]|map("blk.\(.).ffn_down.weight");
def positive: type=="number" and isfinite and .>0;
def median: sort|if length==5 then .[2] else error("five matched trials required") end;
def variability: median as $m | map((.-$m)|abs)|median / $m;
def summary: {median_ms:median,mad_over_median:variability,min_ms:min,max_ms:max};
def results($times;$metric):
    [arms[] as $arm | [$times[]|select(.arm==$arm)|.timing[$metric]] | summary + {arm:$arm}] as $summaries |
    [$times[]|select(.arm=="shared")|.timing[$metric]] as $shared |
    {interval:$metric,arms:$summaries,fastest_control:($summaries|map(select(.arm!="shared"))|min_by(.median_ms)),
     shared_savings:[arms[]|select(.!="shared") as $arm |
        [$times[]|select(.arm==$arm)|.timing[$metric]] as $base |
        [range(5) as $i|$base[$i]-$shared[$i]] as $delta |
        {control:$arm,paired_ms:$delta,median_ms:($delta|median),
         paired_bootstrap_median_ci95_ms:[($delta|min),($delta|max)]}]};
def valid_order:
    . as $row | (if .trial%2==0 then .order else 5-.order end) as $slot |
    .arm == arms[(($slot+($row.trial/2|floor))%6)];
def analyze($p;$t;$weight;$n):
    [$p[]|select(.tensor==$weight and .rows==$n)] as $checks |
    [$t[]|select(.tensor==$weight and .rows==$n)] as $samples |
    if ($checks|map(.arm)|sort)!=(arms|sort) or
        any($checks[]; .outputs!=(2048*$n) or .shared_temporary_bytes!=0 or
            .sgemm_temporary_bytes!=90177536 or (.max_abs|type)!="number" or
            (.max_abs|isfinite|not) or .max_abs<0 or
            ((.arm=="shared" or .arm=="serial" or .arm=="existing_multi_row") and (.different_bits!=0 or .max_abs!=0)))
    then error("parity grid invalid at \($weight), width \($n)") else . end |
    if ($samples|length)!=30 or ($samples|group_by(.trial)|map(.[0].trial))!=[range(5)] or
        ($samples|group_by(.trial)|any(.[]; map(.order)!=[range(6)] or (map(.arm)|unique)!=(arms|sort))) or
        any($samples[]; (valid_order|not) or .iterations!=10 or
            (.timing.gpu_ms|positive|not) or (.timing.wall_ms|positive|not) or
            .timing.completed_unix_us<=.timing.started_unix_us)
    then error("invalid timing grid at \($weight), width \($n)") else . end |
    {tensor:$weight,rows:$n,numerical:[$checks[]|{arm,outputs,different_bits,max_abs}],
     costs:[results(($samples|sort_by(.trial,.order));"gpu_ms"),results(($samples|sort_by(.trial,.order));"wall_ms")]};
.[0] as $m |
[.[]|select(.kind=="preparation")] as $prep |
[.[]|select(.kind=="small_rows_parity")] as $parity |
[.[]|select(.kind=="small_rows_timing")] as $timing |
if $m.kind!="manifest" or $m.small_rows!=true or $m.build_profile!="release" or $m.worktree!="" or
    $m.model_sha256!=$budget[0].model_sha256 or $m.rows!=widths or $m.trials!=5 or $m.requested_iterations!=10 or
    ($m.cuda_flags|index("sm_52"))==null or ($parity|length)!=108 or ($timing|length)!=540 or
    ($prep|map(.tensor.name)|sort)!=(tensors|sort) or
    any($timing[]; (.gpu_before_round.device|startswith("GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852")|not))
then error("wrong manifest or incomplete hardware evidence") else . end |
[tensors[] as $w|widths[] as $n|analyze($parity;$timing;$w;$n)] as $cases |
{source:$m.revision,model:$m.model_sha256,executable:$m.executable_sha256,preparation:$prep,cases:$cases,
 four_row_feasibility:[["gpu_ms","wall_ms"][] as $metric |
   [$cases[]|select(.rows==4)|.costs[]|select(.interval==$metric)|.shared_savings[]|select(.control=="serial")] as $s |
   {interval:$metric,per_layer_savings_ms:($s|map(.median_ms)),
    extrapolated_target_saving_ms:(36*($s|map(.median_ms)|add)/3),
    largest_observed_pair_scaled_to_36_ms:(36*($s|map(.paired_ms|max)|max)),
    required_complete_target_saving_ms:$budget[0].five_percent_target_latency_saving_ms}],
 scope:"Three sampled down weights; whole-target savings are estimates until measured on the complete verifier."}
