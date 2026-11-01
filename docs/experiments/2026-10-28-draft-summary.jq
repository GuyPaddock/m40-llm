def med:
  sort | if length % 2 == 1 then .[length / 2 | floor]
  else (.[length / 2 - 1] + .[length / 2]) / 2 end;
def dist:
  . as $v | ($v | med) as $m |
  {min:min, median:$m, max:max,
   mad_over_median:(if $m > 0 then (map((. - $m) | fabs) | med) / $m else 0 end)};
{
  scope:"joint causal acceptance and release CPU costs; projected target/HTTP scenarios, not measured HTTP",
  cases:map(select(.kind == "causal_draft_case") | {
    id, split, class, source, prompt_tokens, generated_tokens, stop_reason,
    visited_stats, position_stats,
    cpu_ms:(.all_in_cpu_ns | map(. / 1000000) | dist),
    construction_ms:(.trials | map(.construction_ns / 1000000) | dist),
    logical_payload_bytes:(.trials | map(.logical_payload_bytes) | max),
    scenarios:[.projections.scenarios[] | {
      extra_per_attempt_ms, resident_gain, decode_only_gain,
      http_tps:(.model_samples | map(.projected_http_tok_per_sec) |
        map(select(. != null)) | if length > 0 then dist else null end),
      http_gain:(.model_samples | map(.projected_http_gain) |
        map(select(. != null)) | if length > 0 then dist else null end),
      max_cpu_ms_to_break_even:(.model_samples | map(.max_all_in_cpu_ms_to_break_even) | dist),
      extrapolated_attempts:(.model_samples | map(.extrapolated_verifier_attempts) | max),
      extrapolated_serial_rows:(.model_samples | map(.extrapolated_serial_rows) | max)
    }]
  })
}
