def records($kind): map(select(.kind == $kind));
{
  manifest: records("ternary_screen_manifest")[0],
  complete: records("ternary_screen_complete")[0],
  sample_count: (records("ternary_screen_sample") | length),
  correctness_count: (records("ternary_screen_correctness") | length),
  peak_tracked_device_bytes: (map(.tracked_device_bytes // 0) | max),
  maximum_plane_error: (records("ternary_screen_correctness") | map(.checks.planes[].max_abs_diff) | max),
  observations: records("ternary_screen_observation"),
  layer_summary: (records("ternary_screen_observation") | group_by(.case.layer) | map({
    layer: .[0].case.layer,
    median_gpu_us_range: (map(.median_gpu_us) | [min,max]),
    maximum_throughput_variation: (map(.throughput_mad_over_median) | max),
    all_stable: (map(.is_stable) | all),
    extensions: (map(select(.samples > 5)) | length),
    fits_10_percent: (map(select(.fits_10_percent_point_budget)) | length),
    fits_20_percent: (map(select(.fits_20_percent_point_budget)) | length)
  })),
  host_sample_ranges: (records("ternary_screen_sample") | {
    enqueue_ms: (map(.timing.host_enqueue_ms) | [min,max]),
    enqueue_and_completion_ms: (map(.timing.host_enqueue_and_completion_ms) | [min,max])
  }),
  layer_encodings: records("ternary_screen_layer"),
  layer_completions: records("ternary_screen_layer_complete")
}
