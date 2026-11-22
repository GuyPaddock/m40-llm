# Compare like operator groups, keeping throughput and reduction in latency distinct.
def require($v;$m): if $v then . else error($m) end;
def key: [.layer,.length,.head_dim,.backing,.batch,.comparison];
def span: {minimum:min,maximum:max};
def pct: 100 * (1 - .candidate_ms / .control_ms);
def scope: {groups:length, passes:map(select(.passes==true and .timing_error==null))|length,
  unresolved:map(select(.timing_error!=null))|length,
  wall_latency_reduction_percent:(map(pct)|span),
  event_latency_reduction_percent:(map(100*(1-.gpu_candidate_ms/.gpu_control_ms))|span),
  median_control_ms_range:(map(.control_ms)|span),median_candidate_ms_range:(map(.candidate_ms)|span)};

$oct[0] as $old | . as $new |
require($old.artifact_sha256==$new.artifact_sha256 and
  ($old.rows|map(key)|sort)==($new.rows|map(key)|sort) and (.rows|length)==384;
  "unmatched October/November input grid") |
[.rows[]|select(.comparison=="staged_vs_direct")] as $direct |
[.rows[]|select(.comparison=="serial_vs_batched_direct")] as $batch |
{input_sha256:.artifact_sha256,final_timing_errors:.timing_errors,
 matched_staging:($direct|scope),formats:($direct|group_by([.backing,.head_dim])|
 map({backing:first.backing,head_dim:first.head_dim,summary:scope})),
 head128_one_request:($direct|map(select(.head_dim==128 and .batch==1))|group_by(.length)|
 map({context:first.length,summary:scope})),
 batched_direct:($batch|group_by(.batch)|map({batch:first.batch,summary:scope})),
 coverage_changing_groups:([.rows[]|select(.comparison=="dense_context_only")]|length),
 october_matched_staging:($old.rows|map(select(.comparison=="staged_vs_direct"))|scope)}
