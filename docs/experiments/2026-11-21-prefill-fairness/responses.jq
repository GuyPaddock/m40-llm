# Inputs in order: sampled diagnostic and tracing-off HTTP client records.
def require($condition; $description):
  if $condition then . else error($description) end;
def trace_ok($on):
  .metrics as $m |
  if $on then
    ($m.sampled_ids|length)==$m.generated_tokens and
    ($m.token_elapsed_ms|length)==$m.generated_tokens and
    ($m.sample_origin_unix_ns|type)=="number"
  else
    ($m|has("sampled_ids") or has("token_elapsed_ms") or has("sample_origin_unix_ns"))|not
  end;
def signature:
  [.output, .metrics.prompt_tokens, .metrics.generated_tokens, .metrics.stop_reason];
def arm($on):
  require(.success and (.errors|length)==0 and (.records|length)>0; "failed HTTP client") |
  .records |
  require([.[].slot]==[range(length)]; "unexpected HTTP slots") |
  require(all(.[]; .status==200 and .done_ms>=.headers_ms and .headers_ms>=.submitted_ms and
      .response.metrics.generated_tokens>0 and
      ((.done_ms-.submitted_ms-.wall_ms)|fabs)<0.001 and (.response|trace_ok($on)));
    "invalid HTTP timing, counts or trace mode") |
  {signatures:map(.response|signature),
   completion_spread_ms:((map(.done_ms)|max)-(map(.done_ms)|min)),
   makespan_ms:((map(.done_ms)|max)-(map(.submitted_ms)|min)),
   generated_tokens:(map(.response.metrics.generated_tokens)|add),
   request_wall_ms:map(.wall_ms)} |
  .aggregate_tps=(1000*.generated_tokens/.makespan_ms);
require(length==2 and .[0].offsets_ms==.[1].offsets_ms; "unmatched arrival offsets") |
{diagnostic:(.[0]|arm(true)), primary:(.[1]|arm(false))} |
.work_matches=(.diagnostic.signatures==.primary.signatures) |
del(.diagnostic.signatures, .primary.signatures)
