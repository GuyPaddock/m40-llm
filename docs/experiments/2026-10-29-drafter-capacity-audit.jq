# Recompute the negative gate from retained source records and token-ID ledger.
# This verifies capacity evidence, not training-data provenance or R1 readiness.
def require($condition; $message):
  if $condition then . else error($message) end;

. as $report
| $spec[0] as $s
| [$train[0], $validation[0], $test[0]] as $sources
| ["train", "calibration", "held_out"] as $splits
| ($s.selection.strata[] | select(.name == "mbpp_code")) as $quotas
| require($report.schema_version == 1; "schema")
| require($report.kind == "mbpp_source_capacity_necessary_gate"; "kind")
| require($report.model_sha256 == $s.target.sha256; "target identity")
| require($report.has_source_capacity == false; "negative decision changed")
| require($report.is_r1_admitted == false; "unauthorized R1 admission")
| require($report.is_realized_training_manifest == false; "census is not a split manifest")
| require($report.additional_gpu_hours == 0; "unexpected GPU use")
| require(($report.partitions | length) == 3; "partition count")
| require(($ledger[0] | length) == ($sources | map(.source_rows) | add); "ledger count")
| [range(0;3) as $i
    | $splits[$i] as $split
    | $sources[$i] as $source
    | $report.partitions[$i] as $part
    | [$ledger[0][] | select(.split == $split)] as $rows
    | $s.sources[1].source_task_id_ranges[$split] as $range
    | require($part.split == $split; "split order")
    | require(($source.rows | map(.task_id) | sort) == [range($range[0]; $range[1]+1)]; "official task IDs")
    | require(($source.rows | length) == $source.source_rows; "source count")
    | require(($rows | map(.task_id)) == ($source.rows | map(.task_id)); "ledger IDs")
    | require(($part.row_counts | map(.task_id)) == ($rows | map(.task_id)); "report IDs")
    | require($part.parquet_sha256 == $source.source_sha256; "Parquet identity")
    | [range(0; $rows | length) as $j
        | $rows[$j] as $row
        | $source.rows[$j].text as $text
        | ($row.prompt_token_ids | length) as $n
        | require($row.source_text == $text; "source text changed")
        | require($row.prompt == ("<|im_start|>user\n" + ($text | gsub("^\\s+|\\s+$"; "")) + "<|im_end|>\n<|im_start|>assistant\n"); "prompt changed")
        | require(all($row.prompt_token_ids[]; type == "number" and floor == . and . >= 0 and . < 151936); "vocabulary range")
        | require(([$row.prompt_token_ids[] | select(. == 151644)] | length) == 2; "start markers")
        | require(([$row.prompt_token_ids[] | select(. == 151645)] | length) == 1; "end marker")
        | require($part.row_counts[$j].prompt_tokens == $n; "reported token length")
        | any($s.selection.prompt_token_buckets_inclusive[]; .[0] <= $n and $n <= .[1]) as $eligible
        | (if $eligible then "length_eligible_before_other_filters" else "outside_frozen_length_buckets" end) as $status
        | require($row.status == $status and $part.row_counts[$j].status == $status; "row classification")
        | {task_id:$row.task_id, length:$n, eligible:$eligible}
      ] as $counts
    | ([$counts[] | select(.eligible)] | length) as $eligible
    | require($eligible == $part.length_eligible_upper_bound; "capacity upper bound")
    | require($part.source_rows == ($counts | length); "reported source count")
    | require($part.minimum_prompt_tokens == ($counts | map(.length) | min); "minimum")
    | require($part.maximum_prompt_tokens == ($counts | map(.length) | max); "maximum")
    | require($part.bucket_counts == [$s.selection.prompt_token_buckets_inclusive[] as $bucket
        | {minimum:$bucket[0],maximum:$bucket[1],
           count:([$counts[] | select($bucket[0] <= .length and .length <= $bucket[1])] | length)}]; "bucket counts")
    | require($part.required_quota == $quotas[$split]; "quota changed")
    | require($part.has_sufficient_upper_bound == ($eligible >= $quotas[$split]); "capacity flag")
    | require($eligible < $quotas[$split]; "negative prerequisite not proven")
    | {split:$split, source_rows:($counts|length), eligible:$eligible, quota:$quotas[$split]}
  ] as $checked
| {status:"verified_negative_capacity", partitions:$checked,
   checked_prompt_rows:($checked|map(.source_rows)|add), is_r1_admitted:false}
