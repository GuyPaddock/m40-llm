# Independently count necessary eligibility gates from the pinned source and
# runtime tokenizer ledger. This is not a training selection or R1 admission.
def require($condition; $message):
  if $condition then . else error($message) end;

require(($source | length) == 15011; "Dolly source count differs")
| require(($tokens | length) == 15975; "source tokenizer coverage differs")
| require(($tokens | map(.id) | unique | length) == 15975; "duplicate token IDs")
| require(all($tokens[]; .prompt_tokens == (.prompt_ids | length)); "token count differs")
| ($tokens | map({key:.id,value:.prompt_tokens}) | from_entries) as $counts
| require(all(range(1;15012); $counts["dolly:"+tostring] != null); "missing Dolly token row")
| require(all(range(11;975); $counts["mbpp:"+tostring] != null); "missing MBPP token row")
| [range(0;15011) as $index | $source[$index] + {
    source_line:($index+1), prompt_tokens:$counts["dolly:"+(($index+1)|tostring)]
  }] as $rows
| [
    {name:"dolly_prose_open_qa",categories:["creative_writing","open_qa"]},
    {name:"dolly_grounded_qa",categories:["closed_qa"]},
    {name:"dolly_extraction_classification",categories:["information_extraction","classification"]}
  ] as $strata
| [ $strata[] as $stratum
    | [$rows[] | select(.category as $category | $stratum.categories | index($category))] as $members
    | {stratum:$stratum.name, source_rows:($members|length),
       below65:([$members[]|select(.prompt_tokens<65)]|length),
       length_eligible:([$members[]|select(.prompt_tokens>=65 and .prompt_tokens<=447)]|length),
       above447:([$members[]|select(.prompt_tokens>447)]|length),
       required_rows:400, quota:{train:256,calibration:48,held_out:96}}
  ] as $capacity
| require(($dolly[0].rows|length)==15011; "Dolly quarantine ledger incomplete")
| require(all($dolly[0].rows[]; .prompt_tokens==$counts["dolly:"+(.source_line|tostring)] and .is_training_admitted==false and .is_selected==false); "Dolly ledger drift or admission")
| require(($mbpp[0].components|map(.task_ids[])|sort)==[range(11;975)]; "MBPP component coverage differs")
| require(($mbpp[0].benchmark_parse_gaps|length)==0; "unresolved literal parse gaps")
| require(($provenance[0].rows|length)==16; "bounded recovery sample differs")
| {
  kind:"r0_cpu_readiness_assessment", status:"blocked_stop_for_review",
  is_r1_admitted:false, has_realized_training_manifest:false,
  additional_gpu_hours:0, selected_training_rows:0,
  dolly_capacity:$capacity,
  hard_length_quota_failures:[$capacity[]|select(.length_eligible<.required_rows)|.stratum],
  provenance:{
    attempted_records:16,
    network_requests:($provenance[0].requests|length),
    received_bytes:$provenance[0].received_bytes,
    rate_limited_requests:([$provenance[0].requests[]|select((.error//"")|contains("429"))]|length),
    unresolved_records:[$provenance[0].rows[]|select(.canonical_document==null)|.source_line],
    recovered:[$provenance[0].rows[]|select(.canonical_document!=null)],
    grounded_unresolved_total:([$dolly[0].rows[]|select(.stratum=="dolly_grounded_qa" and .canonical_upstream_document==null)]|length),
    recovered_independent_document_count:([$dolly[0].rows[]|select(.stratum=="dolly_grounded_qa" and .canonical_upstream_document!=null)|.canonical_upstream_document]|unique|length),
    admitted_document_groups:0,
    has_established_grounded_qa_quota_feasibility:false
  },
  mbpp:{
    capacity_upper_bounds:$mbpp[0].remaining_capacity,
    shared_span_pair_count:$mbpp[0].shared_span_pair_count,
    alpha_normalized_tasks:$mbpp[0].alpha_normalized_tasks,
    complex_scope_review_pending:$mbpp[0].complex_scope_review_pending,
    paraphrase_pairs_pending:($mbpp[0].paraphrase_review_queue|length),
    automatic_benchmark_overlap_tasks:($mbpp[0].benchmark_overlap|keys|length),
    manual_benchmark_overlap_tasks:[674,932],
    quarantined_components:([$mbpp[0].components[]|select(.quarantine_reasons|length>0)]|length),
    quarantined_rows:([$mbpp[0].components[]|select(.quarantine_reasons|length>0)|.task_ids[]]|length),
    is_complete_manual_template_audit:$mbpp[0].is_complete_manual_template_audit
  },
  benchmark_inventory:{
    files:($inventory[0].files|length),unique_texts:$inventory[0].unique_texts,
    parse_gaps:[$inventory[0].files[]|select(.errors|length>0)],
    missing_local_references:$inventory[0].missing_local_references,
    has_complete_semantic_exclusion_clearance:false
  },
  stop_reasons:["Dolly prose/open-QA length capacity below unchanged quota",
                "Dolly grounded-QA upstream-document capacity unestablished",
                "MBPP complex-scope and semantic isolation reviews incomplete"],
  source_payloads:$identity[0].source_payloads
}
