def require($ok; $message): if $ok then . else error($message) end;
def median: sort | .[2];
map(select(.kind == "packed_prefill_gpu_profile" and
           (.session_id >= 2 and .session_id <= 6))) | sort_by(.session_id)
| require(map(.session_id) == [2,3,4,5,6]; "missing primary prefix")
| map(
    . as $sample
    | require(.prefix_tokens == 64; "different prefix")
    | [.stages[] | select(.layer_id == 35)] as $layer
    | [.stages[] | select(.stage == "prefill_output_copy")] as $copies
    | [$layer[] | select(.stage == "prefill_attention")] as $attention
    | require(($layer|length) == 22 and ($copies|length) == 1 and
              ($attention|length) == 1; "incomplete terminal-layer profile")
    | ($copies[0].gpu_end_ms - ($layer | map(.gpu_start_ms) | min)) as $window
    | ($copies[0].gpu_end_ms - $attention[0].gpu_start_ms) as $tail
    | require(0 < $tail and $tail < $window and $window < .prefill_wall_ms;
              "invalid event ordering")
    | {session_id, prefix_wall_ms:.prefill_wall_ms,
       full_window_ms:$window, post_append_window_ms:$tail,
       pct_prefix: (100*$window/.prefill_wall_ms),
       pct_request: (100*$window/$audit[0].off_http_ms)}
  )
| {rows:., p50_full_window_ms:(map(.full_window_ms)|median),
   p50_post_append_ms:(map(.post_append_window_ms)|median),
   p50_pct_prefix:(map(.pct_prefix)|median),
   p50_pct_request:(map(.pct_request)|median),
   off_request_ms:$audit[0].off_http_ms}
