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

null |
 require((" 47\n"|rescore({kind:"exact",answers:["47"]})).passed;"exact positive") |
 require(("total 47"|rescore({kind:"exact",answers:["47"]})).passed==false;"exact negative") |
 require(("```json\n{\"day\":\"SATURDAY\"}\n```"|rescore({kind:"json_object",expected:{day:"Saturday"}})).passed;"JSON fence") |
 require(("{\"day\":\"Saturday\",\"extra\":1}"|rescore({kind:"json_object",expected:{day:"Saturday"}})).passed==false;"JSON extras") |
 require(("Mulching soil saves water.\n\nUse a rain barrel."|rescore({kind:"concepts",min_words:8,min_paragraphs:2,facts:[{id:"cover",aliases:["mulching"]},{id:"collection",aliases:["rain barrel"]}]})).passed;"concept positive") |
 require(("Mulching soil. Rain barrels."|rescore({kind:"concepts",min_words:8,min_paragraphs:2,facts:[{id:"cover",aliases:["mulching"]}]})).passed==false;"concept formatting") |
 require(([1,2,3,4,5]|repetition).suffix_pattern_width==0;"distinct suffix") |
 ([1,2,1,2,1,2,1,2,151645]|repetition) as $repeat |
 require($repeat.suffix_pattern_width==2 and $repeat.suffix_pattern_copies==4 and $repeat.repeated_fourgram_fraction==0.6;"repeated suffix and grams") |
 {behavioral_checks:8,passed:true,scope:"audit functions only; complete-grid admission remains separate"}
