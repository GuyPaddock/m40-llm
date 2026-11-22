# Read one instrumentation-off run, its own diagnostic trace and summary.
def signature: [.slot,.response.output,.response.metrics.prompt_tokens,
  .response.metrics.generated_tokens,.response.metrics.stop_reason];
.records as $rows | $diagnostic[0].records as $traces |
if .success != true or $diagnostic[0].success != true or
   ($rows|length)!=$count or ($traces|length)!=$count or
   ($summary[0].requests|length)!=$count or
   (all($rows[]; .status==200 and .response.metrics.prompt_tokens==65 and
        .response.metrics.generated_tokens>0 and .response.metrics.generated_tokens<=64 and
        .response.metrics.token_elapsed_ms==null and .response.metrics.sampled_ids==null)|not) or
   (all($traces[]; .response.metrics as $m |
        ($m.sampled_ids|type)=="array" and ($m.sampled_ids|length)==$m.generated_tokens)|not)
then error("incomplete or invalid client token accounting")
else
  {label:$label,arm:$arm,matches_companion:([$rows[]|signature] == [$traces[]|signature]),
   http_seconds:($summary[0].wall_ms/1000),
   metrics:{prompt_tokens:([$rows[].response.metrics.prompt_tokens]|add),
     generated_tokens:([$rows[].response.metrics.generated_tokens]|add),
     stop_reason:([$rows[].response.metrics.stop_reason]|join(","))}}
end
