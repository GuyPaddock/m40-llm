def require($condition; $message): if $condition then . else error($message) end;
. as $records
| [ .[] | select(.kind == "cpu_projection_validation") ] as $cases
| [ .[] | select(.kind == "cpu_projection_pair") ] as $pairs
| require(length == 48; "unexpected record count")
| require(.[0].kind == "cpu_projection_manifest" and .[0].is_validation_only == true;
          "not a numerical-validation manifest")
| require(.[0].revision | startswith("bf8dc96"); "unexpected source revision")
| require(.[0].placement.cpu == 15 and .[0].placement.numa_node == 0
          and .[0].placement.threads == 1; "unexpected placement")
| require(.[-1].kind == "cpu_projection_complete" and .[-1].is_complete == true
          and .[-1].cases == 36 and .[-1].generated_token_count == 0; "incomplete run")
| require(($cases | length) == 36 and ($pairs | length) == 0; "case/timing count differs")
| require(([$cases[].case | [.layer, .family, .batch]] | sort) ==
          ([ [0,17,35][] as $layer | ["gate","up","down"][] as $family |
             [1,2,4,8][] as $batch | [$layer,$family,$batch] ] | sort); "grid differs")
| require(all($cases[]; .scalar.elements == .case.rows * .case.batch and
              .avx2.elements == .scalar.elements and
              .scalar.max_tolerance_ratio <= 1 and .avx2.max_tolerance_ratio <= 1);
          "invalid numerical result")
| {
    source_revision: $records[0].revision,
    executable_sha256: $records[0].executable_sha256,
    cases: ($cases | length),
    output_elements_per_implementation: ([$cases[].scalar.elements] | add),
    scalar_max_absolute_error: ([$cases[].scalar.max_absolute_error] | max),
    avx2_max_absolute_error: ([$cases[].avx2.max_absolute_error] | max),
    scalar_max_tolerance_ratio: ([$cases[].scalar.max_tolerance_ratio] | max),
    avx2_max_tolerance_ratio: ([$cases[].avx2.max_tolerance_ratio] | max),
    timed_pairs: ($pairs | length),
    generated_token_count: 0,
    is_complete: true,
    is_performance_measured: false
  }
