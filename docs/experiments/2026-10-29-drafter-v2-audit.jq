. as $all |
(.mbpp.rows | sort_by(.task_id)) as $rows |
(.mbpp.components | map(. as $g | {key: .group_id, value: $g}) | from_entries) as $groups |
($rows | group_by(.group_id) | map({
  group_id: .[0].group_id,
  task_ids: (map(.task_id) | sort),
  source_splits: (map(.source_split) | unique)
}) | map(. + {is_cross_partition: (.source_splits | length > 1)})) as $derived_groups |
(["train", "calibration", "held_out"] | map(. as $s | {
  key: $s,
  value: ([$rows[] | select(.source_split == $s and (.quarantine_reasons | length) == 0)] | length)
}) | from_entries) as $bounds |
([.mbpp.edges[] | . as $edge |
  [$rows[] | select(.task_id == $edge.left or .task_id == $edge.right) | .group_id] |
  unique | length == 1
] | all) as $edges_consistent |
{
  kind: "independent_v2_candidate_consistency_audit",
  is_complete_training_admission: false,
  mbpp_rows: ($rows | length),
  dolly_rows: (.dolly.rows | length),
  mbpp_ids_match: ([$rows[].task_id] == [range(11;975)]),
  dolly_lines_match: ([.dolly.rows[].source_line] == [range(1;15012)]),
  group_memberships_match: (($derived_groups | sort_by(.group_id)) == (.mbpp.components | sort_by(.group_id))),
  no_selected_rows: ([$rows[].is_selected, .dolly.rows[].is_selected] | all(. == false)),
  all_cross_partition_groups_quarantined: ([$rows[] |
    select($groups[.group_id].is_cross_partition) |
    (.quarantine_reasons | index("cross_partition_component") != null)
  ] | all),
  edge_components_consistent: $edges_consistent,
  capacities_recomputed: $bounds,
  capacities_match: ($bounds == $report[0].mbpp_residual_row_upper_bounds),
  source_partitions_preserved: ([$rows[] |
    if .task_id >= 601 then .source_split == "train"
    elif .task_id >= 511 then .source_split == "calibration"
    else .source_split == "held_out" end
  ] | all),
  grounded_contexts_quarantined: ([.dolly.rows[] |
    select(.stratum == "dolly_grounded_qa") |
    .quarantine_reasons | index("unresolved_upstream_document") != null
  ] | all)
} |
if (.mbpp_ids_match and .dolly_lines_match and .group_memberships_match and .no_selected_rows and
    .all_cross_partition_groups_quarantined and .edge_components_consistent and
    .capacities_match and .source_partitions_preserved and
    .grounded_contexts_quarantined and .mbpp_rows == 964 and .dolly_rows == 15011)
then . else error("candidate audit mismatch") end
