def trim: gsub("^ +| +$"; "");

map(select(.kind == "timing")) as $timings |
[$timings[] | .baseline, .candidate] | sort_by(.started_unix_us) as $intervals |
[$timings[] | .gpu_before, .gpu_after] as $gpu |
{
  timing_pairs: ($timings | length),
  measured_intervals: ($intervals | length),
  overlapping_intervals: ([range(1; $intervals | length) |
    select($intervals[. - 1].completed_unix_us > $intervals[.].started_unix_us)] | length),
  missing_timestamps: ([$intervals[] |
    select(.started_unix_us == null or .completed_unix_us == null)] | length),
  reversed_or_empty_intervals: ([$intervals[] |
    select(.completed_unix_us <= .started_unix_us)] | length),
  gpu_process_counts: ($gpu | map(.processes | split("\n") |
    map(select(length > 0)) | length) | unique),
  gpu_uuids: ($gpu | map(.device | split(",")[0] | trim) | unique),
  pstates: ($gpu | map(.device | split(",")[3] | trim) | unique),
  sm_clocks: ($gpu | map(.device | split(",")[5] | trim) | unique),
  memory_clocks: ($gpu | map(.device | split(",")[6] | trim) | unique),
  temperature_range_c: ($gpu | map(.device | split(",")[4] | trim | tonumber) | [min,max]),
  tiled_launch_configurations: ([$timings[] |
    select(.execution_layout | type == "object") |
    .config + {access: (.config.access // "vector")}] | unique | length),
  max_abs_diff: ($timings | map(.errors.max_abs_diff) | max),
  mean_abs_diff: ($timings | map(.errors.mean_abs_diff) | max),
  cpu_sample_max_abs_diff: ($timings | map(.errors.cpu_sample_max_abs_diff) | max)
}
