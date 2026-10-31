# jq -s --slurpfile quality QUALITY --arg quality_sha256 HASH -f THIS HTTP
def require($condition;$message): if $condition then . else error($message) end;
def one($kind): map(select(.kind==$kind))|require(length==1;"record multiplicity: \($kind)")|.[0];
def median: sort | length as $n | if $n%2==0 then (.[($n/2)-1]+.[$n/2])/2 else .[($n/2|floor)] end;
def close($a;$b): (($a-$b)|fabs)<1e-9;
def gpu_valid:
  (.clocks|split(",")|map(gsub("^ +| +$";""))) as $c
  | ($c==["GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852","P0","1113","3004"])
    and (.processes|split("\n")|length)==1
    and (.processes|test("^GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852, [0-9]+, .+, [0-9]+$"));
. as $rows | one("joint_only_manifest") as $m
| $rows | one("joint_only_http_binding") as $b
| $rows | one("joint_only_http_complete") as $done
| ($quality|one("joint_only_manifest")) as $qm
| ($quality|one("joint_only_ownership")) as $qo
| ($quality|one("joint_only_generation")) as $g
| ($quality|one("joint_only_quality_complete")) as $qd
| $rows
| require(($m|del(.gpu))==($qm|del(.gpu)) and $qd.is_gate_passed and $qd.device_bytes_after==0
    and $b.quality_sha256==$quality_sha256;"quality/source/executable binding")
| require($b.control.ownership=="retain-split" and $b.candidate.ownership=="joint-only"
    and $b.control.logical_sources==$qo.control.logical_sources and $b.candidate.logical_sources==$qo.candidate.logical_sources
    and $b.control.resident_split.tensors==$qo.control.resident_split.tensors
    and $b.candidate.resident_split.tensors==$qo.candidate.resident_split.tensors
    and $b.control.joint.pairs==$qo.control.joint.pairs and $b.candidate.joint.pairs==$qo.candidate.joint.pairs
    and $b.control.uploaded_gguf_bytes==3279519744 and $b.candidate.uploaded_gguf_bytes==3279519744
    and $b.control.joint.resident_bytes==1724645376 and $b.candidate.joint.resident_bytes==1724645376
    and $b.control.resident_split.resident_bytes==$qo.control.resident_split.resident_bytes
    and $b.candidate.resident_split.resident_bytes==$qo.candidate.resident_split.resident_bytes
    and $b.control.joint.combined_prepared_bytes==$qo.control.joint.combined_prepared_bytes
    and $b.candidate.joint.combined_prepared_bytes==$qo.candidate.joint.combined_prepared_bytes
    and $b.control.resident_split.resident_bytes-$b.candidate.resident_split.resident_bytes==1724645376;
    "HTTP model ownership differs from quality")
| ($rows|map(select(.kind=="joint_only_http_trial"))) as $trials
| ($rows|map(select(.kind=="joint_only_http_summary"))) as $summaries
| require(($summaries|length)>=1 and all($trials[];
    .row.output==$g.output and .row.metrics.prompt_tokens==65 and .row.metrics.generated_tokens==512
    and .row.metrics.stop_reason=="length" and .row.dispatch==$g.control_dispatch
    and .row.http_ms>0 and close(.row.http_tok_per_sec;512000/.row.http_ms)
    and (.gpu_before|gpu_valid) and (.gpu_after|gpu_valid)
    and .gpu_before.processes==.gpu_after.processes);"fixed work/dispatch/GPU/timing boundary")
| require(all($summaries[];
    . as $s | ([$trials[]|select(.phase==$s.phase and .pair<$s.comparison.pairs)]) as $t
    | ([$t[]|select(.arm==0)|.row.http_tok_per_sec]|median) as $a
    | ([$t[]|select(.arm==1)|.row.http_tok_per_sec]|median) as $b
    | ($s.comparison) as $c
    | ($t|length)==2*$c.pairs
      and ($t|map([.pair,.arm])|unique|length)==2*$c.pairs
      and ($t|map(.pair)|unique)==[range($c.pairs)]
      and all(range($c.pairs);. as $p|([$t[]|select(.pair==$p)|.arm])==(if $p%2==0 then [0,1] else [1,0] end))
      and close($a;$c.control.median) and close($b;$c.candidate.median) and close($b/$a-1;$c.gain)
      and $c.bootstrap_resamples==10000 and $c.bootstrap_seed==0
      and $c.bootstrap_gain_ci95[0]<=$c.bootstrap_gain_ci95[1]
      and close($c.promotion_threshold;([0.03,2*$c.control.mad_over_median,2*$c.candidate.mad_over_median]|max)));
    "paired statistics or alternating schedule")
| $summaries[-1] as $last | $last.comparison as $c
| require(($last.phase=="screen" and ($c.pairs==5 or $c.pairs==10)
      and ($summaries|length)==(if $c.pairs==5 then 1 else 2 end)
      and ($c.pairs!=10 or $summaries[0].comparison.needs_variation_extension)
      and ($trials|length)==2*$c.pairs
      and ($c.needs_fresh_confirmation|not) and ($last.reverse_comparison.needs_fresh_confirmation|not))
    or ($last.phase=="confirmation" and $c.pairs==20
      and ($summaries[-2].comparison.needs_fresh_confirmation or $summaries[-2].reverse_comparison.needs_fresh_confirmation));
    "five/ten/fresh-twenty protocol")
| require(($c.needs_variation_extension|not) and $c.gain > -$c.promotion_threshold
    and $c.bootstrap_gain_ci95[0] > -$c.promotion_threshold
    and $done.is_storage_gate_passed and $done.device_bytes_after==0 and $done.requires_speedup==false;
    "storage non-regression/cleanup")
| {is_http_audit_passed:true,revision:$m.revision,executable_sha256:$m.executable_sha256,
    pairs:$c.pairs,phase:$last.phase,control_http_tok_per_sec:$c.control.median,
    candidate_http_tok_per_sec:$c.candidate.median,gain:$c.gain,paired_ci95:$c.bootstrap_gain_ci95,
    meaningful_threshold:$c.promotion_threshold,requires_speedup:false,device_bytes_after:0}
