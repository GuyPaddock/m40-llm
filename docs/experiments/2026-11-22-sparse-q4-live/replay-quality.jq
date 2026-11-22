def check($condition; $label): if $condition then . else error($label) end;

def repeating($ids):
    any(range(0; ($ids|length) - 15); . as $start
        | $ids[$start:$start+16] | unique | length == 1)
    or any(range(2; 9); . as $width
        | any(range(0; ($ids|length) - 4*$width + 1); . as $start
            | ($ids[$start:$start+$width]) as $pattern
            | ($pattern | unique | length) > 1
            and $ids[$start:$start+4*$width] == ($pattern+$pattern+$pattern+$pattern)));

def lost_task:
    (.reference_score.passed and (.candidate_score.passed | not))
    or .has_new_repetition;

.[0] as $manifest | .[-1] as $end
| check($manifest.kind == "sparse_q4_quality_manifest" and $manifest.variant == $label
    and $manifest.is_thresholded == ($label | endswith("-thresholded"))
    and ($label | startswith($manifest.layout))
    and $end.kind == "sparse_q4_quality_complete" and $end.device_bytes_after == 0;
    "incomplete or mismatched variant")
| check($end.candidate_layout.layers == [0,17,35]
    and ($end.candidate_layout.projection_calls | min) > 0
    and ($end.candidate_layout.projection_calls | unique | length) == 1
    and $end.candidate_memory.activation_sparsity.extra_weight_bytes == 38043648
    and $end.candidate_memory.activation_sparsity.selection_workspace_bytes == 350212;
    "missing projection execution or unexpected layout allocation")
| map(select(.kind == "model_quality_task")) as $tasks
| map(select(.kind == "sparse_q4_quality_passage")) as $passages
| check(($tasks | map(.id)) == ($corpus[0].held_out.tasks | map(.id)[:$end.tasks_run])
    and ($passages | map(.id)) == ($corpus[0].held_out.passages | map(.id)[:$end.passages_run]);
    "missing, duplicated or reordered held-out work")
| check(all($tasks[]; . as $case
    | ($dense[] | select(.kind == "ffn_capture_task" and .id == $case.id)) as $original
    | .prompt_token_ids == $original.prompt_token_ids
      and .reference.generated_token_ids == $original.baseline.generated_token_ids
      and .reference.output == $original.baseline.output
      and .reference_score.passed == $original.score.passed
      and .has_new_repetition == (repeating(.candidate.generated_token_ids)
        and (repeating(.reference.generated_token_ids) | not)));
    "dense baseline or repetition classification differs")
| check($end.dense_passes == ($tasks | map(select(.reference_score.passed)) | length)
    and $end.dense_failures == ($tasks | map(select(.reference_score.passed | not)) | length)
    and all($tasks[:-1][]; lost_task | not)
    and all($passages[:-1][]; .perplexity_ratio <= 1.03)
    and all($passages[]; .comparison.reference.perplexity as $a | .comparison.candidate.perplexity as $b
        | if $a == null or $b == null then .perplexity_ratio == null
          else ((.perplexity_ratio - $b / $a) | fabs) < 1e-12 end);
    "stopping or loss accounting differs")
| check(
    if $end.failure == null then
        $end.tasks_run == ($corpus[0].held_out.tasks | length)
        and $end.passages_run == ($corpus[0].held_out.passages | length)
        and all($tasks[]; lost_task | not) and all($passages[]; .perplexity_ratio <= 1.03)
        and $end.disposition == (if $end.dense_passes > 0 then "PASS" else "INCONCLUSIVE" end)
    elif $end.failure.reason == "new_repetition" or $end.failure.reason == "lost_dense_pass" then
        $end.disposition == "FAIL" and $end.passages_run == 0
        and ($tasks | last | lost_task) and ($tasks | last | .id) == $end.failure.id
    elif $end.failure.reason == "perplexity" then
        $end.disposition == "FAIL" and ($passages | last | .id) == $end.failure.id
        and ($passages | last | (.perplexity_ratio == null or .perplexity_ratio > 1.03))
    else $end.failure.reason == "nonfinite_logits" or $end.failure.reason == "unusable_dense_perplexity"
    end; "wrong final quality disposition")
| {variant:$label, disposition:$end.disposition, tasks:$end.tasks_run, passages:$end.passages_run,
    dense_passes:$end.dense_passes, dense_failures:$end.dense_failures,
    stop:$end.failure, sampled_two_model_peak_mib:$peak,
    candidate_memory:$end.candidate_memory.activation_sparsity, layout:$end.candidate_layout}
