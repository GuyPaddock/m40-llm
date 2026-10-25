def median:
  sort | length as $n |
  if $n % 2 == 1 then .[($n / 2 | floor)]
  else (.[($n / 2 - 1)] + .[($n / 2)]) / 2 end;

def variation:
  median as $m | map((. - $m) | fabs) | median / $m;

. as $records |
{
  manifest: ($records | map(select(.kind == "manifest")) | first),
  preparation: ($records | map(select(.kind == "preparation"))),
  results: ($records | map(select(.kind == "timing")) |
    group_by([.tensor, .rows, .config.columns, .config.threads]) |
    map({
      tensor: .[0].tensor,
      shape: .[0].shape,
      rows: .[0].rows,
      config: .[0].config,
      pairs: length,
      baseline_gpu_ms: (map(.baseline.gpu_ms) | median),
      candidate_gpu_ms: (map(.candidate.gpu_ms) | median),
      baseline_mad_ratio: (map(.baseline.gpu_ms) | variation),
      candidate_mad_ratio: (map(.candidate.gpu_ms) | variation),
      speedup: ((map(.baseline.gpu_ms) | median) / (map(.candidate.gpu_ms) | median)),
      paired_speedup_median: (map(.baseline.gpu_ms / .candidate.gpu_ms) | median),
      max_abs_diff: (map(.errors.max_abs_diff) | max),
      mean_abs_diff: (map(.errors.mean_abs_diff) | max),
      cpu_sample_max_abs_diff: (map(.errors.cpu_sample_max_abs_diff) | max)
    }))
}
