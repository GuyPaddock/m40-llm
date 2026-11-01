# Independent serialized accounting; does not rerun attention or reconstruct logits.
def require($condition; $message):
  if $condition then . else error($message) end;
def attention($arm):
  if $arm == "dense" then "attention_last_token_f32_gqa"
  else "attention_last_token_f32_gqa_block_select_exact_fp16_k_q4_v_old_direct" end;
def check_cache($case; $arm; $count):
  (($case.prompt_ids|length) + $count - 1) as $consumed |
  ([$consumed - 1024, 0]|max) as $old |
  (($old / 32)|ceil) as $blocks |
  require(.consumed_tokens == $consumed and .old_tokens == $old and .old_blocks == $blocks;
          "cache length/block arithmetic") |
  require((.physical_lengths|length) == 36 and all(.physical_lengths[]; . == $consumed);
          "layer cache lengths") |
  require(.is_compressed == ($arm != "dense") and .all_old_blocks_fit_top64 and $blocks <= 64;
          "cache semantics/coverage") |
  require(.configured_old_block_budget == (if $arm == "quantized_top8" then ([$blocks,8]|min) else $blocks end);
          "selection budget") |
  require(if $case.context == "short" then $old == 0 else $blocks > 8 end; "context control");
def check_dispatch($arm; $count):
  require(.operations[attention($arm)].launches == 36 * $count; "attention dispatch count") |
  require(.totals.cublas_calls == 252; "packed prefill projection count");
def trim_text: gsub("^\\s+|\\s+$"; "");
def normalize: gsub("\\s+"; " ") | trim_text | ascii_downcase;
def word_text: [scan("[[:alnum:]]+")|ascii_downcase] | join(" ") | " " + . + " ";
def rescore($rule):
  . as $output |
  (if $rule.kind=="exact" then
     {format_valid:true, checks:[{id:"answer",has_match:any($rule.answers[]; normalize==($output|normalize))}]}
   elif $rule.kind=="json_object" then
     ($output|trim_text|if startswith("```") and endswith("```") and contains("\n")
       then sub("^```[^\n]*\n";"")|sub("```$";"")|trim_text else . end |
       try fromjson catch null) as $actual |
     {format_valid:(($actual|type)=="object" and ($actual|length)==($rule.expected|length)),
      checks:[$rule.expected|to_entries[]|. as $entry|
        {id:.key,has_match:(if ($actual|type)!="object" then false
          elif (.value|type)=="string" and ($actual[.key]|type)=="string"
          then (.value|normalize)==($actual[.key]|normalize)
          else .value==$actual[.key] end)}]}
   elif $rule.kind=="concepts" then
     ($output|word_text) as $words |
     {format_valid:(([$output|scan("\\S+")]|length)>=$rule.min_words and
        ($output|gsub("\r\n";"\n")|trim_text|split("\n\n")|map(select((trim_text|length)>0))|length)>=$rule.min_paragraphs),
      checks:[$rule.facts[]|{id,has_match:any(.aliases[]; (word_text) as $phrase | $words|contains($phrase))}]}
   else error("unknown scoring rule") end) as $scored |
  {format_valid:$scored.format_valid,
   matched:[$scored.checks[]|select(.has_match)|.id],missing:[$scored.checks[]|select(.has_match|not)|.id]} |
  . + {passed:(.format_valid and (.missing|length)==0),score:((.matched|length)/($scored.checks|length))};
def repetition:
  (if .[-1]==151645 or .[-1]==151643 then .[:-1] else . end) as $ids |
  ($ids|length) as $length |
  # Materialize slices: this host's jq compares shared array slices incorrectly.
  [range(0;[$length-3,0]|max)|$ids[.:.+4]|map(.)] as $grams |
  ([{width:0,copies:0,span:0}] + [range(1;[16,($length/2|floor)]|min+1) as $width |
    [range(1;($length/$width|floor)+1) as $copy |
     ($ids[$length-$copy*$width:$length-($copy-1)*$width]|map(.))==($ids[$length-$width:]|map(.))] |
    (index(false)//length) as $copies | select($copies>=2) |
    {width:$width,copies:$copies,span:($width*$copies)}] | sort_by([-.span,.width])|.[0]) as $suffix |
  {suffix_pattern_width:$suffix.width,suffix_pattern_copies:$suffix.copies,analyzed_tokens:$length,
   repeated_fourgram_fraction:(if ($grams|length)==0 then 0 else 1-(($grams|unique|length)/($grams|length)) end)};

. as $rows |
[$rows[]|select(.kind=="kv_quality_manifest")] as $headers |
[$rows[]|select(.kind=="kv_quality_complete")] as $ends |
require(($headers|length)==1 and ($ends|length)==1 and $rows[-1].kind=="kv_quality_complete";
        "missing/duplicate completion or header") |
$headers[0] as $header | $ends[0] as $end |
require($header.inputs == $inputs[0] and $header.source_clean and
        $header.source_revision == "421df83e085212b78ca0d2e57b91a20ebb879e1e"; "source/input identity") |
require($header.executable_sha256=="2cd0408f1019df978a45a791a3a750fbf5bad1a6a135fba743104b10f3aafea0" and
        $header.model_sha256==$inputs[0].model_sha256 and $header.corpus_sha256==$inputs[0].corpus_sha256 and
        $header.inputs_sha256=="e2e7ce56f0223bf929705dd49c33610ac01a0992837bb9c376bcea6dfc62b5de";
        "artifact identity") |
require($header.capacity==3072 and $header.recent_window==1024 and $header.block_size==32 and
        $header.quantized_all_top_blocks==64 and $header.quantized_top8_top_blocks==8 and
        $header.quantized_backing=="fp16-k-q4-v" and $header.quantized_attention=="fp16-k-q4-v-direct" and
        $header.environment.M40LLM_DECODE_GRAPH=="0" and $header.environment.M40LLM_ATTN_HEAD128_KERNEL=="native" and
        $header.is_performance_measurement==false; "configuration scope") |
[$rows[]|select(.kind=="kv_quality_generation")] as $generations |
[$rows[]|select(.kind=="kv_quality_comparison")] as $comparisons |
[$rows[]|select(.kind=="kv_quality_prefix_reference")] as $references |
[$rows[]|select(.kind=="kv_quality_case_complete")] as $cases |
require(($rows|length)==86 and ($generations|length)==36 and ($comparisons|length)==24 and
        ($references|length)==12 and ($cases|length)==12; "record coverage") |
require($end.cases==12 and $end.generation_rows==36 and $end.comparison_rows==24 and
        $end.generated_tokens==([$generations[].generated_ids|length]|add) and
        $end.initial_tracked_device_bytes==0 and $end.final_tracked_device_bytes==0;
        "completion accounting/ownership") |
[
  $inputs[0].cases[] as $case |
  [$generations[]|select(.case==$case.id)] as $arms |
  [$comparisons[]|select(.case==$case.id)] as $paired |
  [$references[]|select(.case==$case.id)] as $ref |
  require(($arms|map(.arm)|sort)==["dense","quantized_all","quantized_top8"] and
          ($paired|map(.arm)|sort)==["quantized_all","quantized_top8"] and ($ref|length)==1;
          "case arm uniqueness") |
  require(([$cases[]|select(.case==$case.id and .generation_arms==3 and .comparison_arms==2)]|length)==1;
          "case completion") |
  $arms[0] as $dense | ($dense.generated_ids|length) as $dense_count |
  ([0,1,8,32,64,127,$dense_count-1]|map(select(.<$dense_count))|unique) as $steps |
  require($ref[0].steps==$steps and $ref[0].targets==[$steps[]|$dense.generated_ids[.]];
          "reference observation positions") |
  ($ref[0].cache|check_cache($case; "dense"; $dense_count)) as $checked_reference_cache |
  ($ref[0].dispatch|check_dispatch("dense"; $dense_count)) as $checked_reference_dispatch |
  [ $arms[] |
    . as $arm | (.generated_ids|length) as $count |
    require($count>0 and $count<=$case.task.max_tokens and .metrics.generated_tokens==$count and
            .metrics.prompt_tokens==($case.prompt_ids|length) and .context==$case.context;
            "generation counts/context") |
    require(.metrics.stop_reason=="stop" or .metrics.stop_reason=="length"; "unknown stop") |
    require(if .metrics.stop_reason=="length" then $count==$case.task.max_tokens
            else (.generated_ids[-1]==151645 or .generated_ids[-1]==151643) end; "stop accounting") |
    require(all(.generated_ids[:-1][]; .!=151645 and .!=151643); "tokens after stop") |
    require(.score==(.output|rescore($case.task.scoring)); "independent task scoring") |
    (.generated_ids|repetition) as $repeated |
    require(.repetition.analyzed_tokens==$repeated.analyzed_tokens and
            .repetition.suffix_pattern_width==$repeated.suffix_pattern_width and
            .repetition.suffix_pattern_copies==$repeated.suffix_pattern_copies and
            ((.repetition.repeated_fourgram_fraction-$repeated.repeated_fourgram_fraction)|fabs)<1e-12;
            "independent repetition diagnostic") |
    (.cache|check_cache($case; $arm.arm; $count)) as $checked_cache |
    (.dispatch|check_dispatch($arm.arm; $count)) as $checked_dispatch |
    {arm, count:$count, stop:.metrics.stop_reason, passed:.score.passed, output, repetition}
  ] as $outputs |
  [ $paired[] |
    . as $pair | ($arms[]|select(.arm==$pair.arm)) as $candidate |
    ([($dense.generated_ids|length),($candidate.generated_ids|length)]|min) as $common_length |
    ([range(0;$common_length)|select($dense.generated_ids[.]!=$candidate.generated_ids[.])]|.[0]) as $difference |
    require(.first_generated_difference==($difference//(if ($dense.generated_ids|length)!=($candidate.generated_ids|length)
        then $common_length else null end)); "first output difference") |
    require(.dense_score==$dense.score and .candidate_score==$candidate.score and
            .quality_conclusion==(if $dense.score.passed|not then "inconclusive"
                                 elif $candidate.score.passed then "preserved" else "regression" end);
            "quality classification") |
    require(.identical_generated_ids==($dense.generated_ids==$candidate.generated_ids) and
            .identical_output==($dense.output==$candidate.output) and
            .no_old_control==($case.context=="short"); "output equality/control") |
    require((.prefix_observations|map(.step))==$steps; "paired positions") |
    require(all(.prefix_observations[];
      .prefix_tokens==($case.prompt_ids|length)+.step and .target==$dense.generated_ids[.step] and
      .logits.kl_reference_candidate>=0 and .logits.max_abs_logit_diff>=.logits.mean_abs_logit_diff and
      .logits.top10_overlap==([.logits.reference.top_indices[] as $id |
                             select(.logits.candidate.top_indices|index($id))]|length));
      "prefix causality/logit accounting") |
    (.cache|check_cache($case; $pair.arm; $dense_count)) as $checked_cache |
    (.dispatch|check_dispatch($pair.arm; $dense_count)) as $checked_dispatch |
    {arm, quality_conclusion, identical_generated_ids, first_generated_difference,
     observations:(.prefix_observations|length),
     max_logit_difference:(.prefix_observations|map(.logits.max_abs_logit_diff)|max),
     max_kl:(.prefix_observations|map(.logits.kl_reference_candidate)|max),
     min_top10_overlap:(.prefix_observations|map(.logits.top10_overlap)|min)}
  ] as $checks |
  {case:$case.id, context:$case.context, outputs:$outputs, comparisons:$checks}
] as $summary |
{audit:"serialized accounting, task scores, repetition, output equality and recorded top10 overlap; logits not independently recomputed",
 source:$header.source_revision, cases:12, generated_tokens:$end.generated_tokens,
 comparisons:24, results:$summary}
