include "october_quality";
# Re-score actual free-running text and verify the cache path used to produce each token.
. as $rows | $inputs[0].cases as $cases |
require(($cases|length)==48 and ($cases|map(.id)|unique|length)==48; "frozen case set") |
.[0] as $header |
require($header.kind=="kv_quality_manifest" and $header.source_clean and
        $header.source_revision=="c49e2083f85bda2c32b6fcd0d69f9c2d732ea7c5" and
        $header.executable_sha256=="85df424712a9da22afd7f9518078d48cdc4172d8fe3b14eaa665a4651fbce525" and
        $header.model_sha256=="4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" and
        $header.inputs==$inputs[0] and $header.block_size==32 and $header.recent_window==1024 and
        $header.capacity==3072 and $header.quantized_top8_top_blocks==8 and $header.quantized_all_top_blocks==64;
        "November source, inputs or allocation configuration") |
(if $dense_only then ["dense"] else ["dense","quantized_all","quantized_top8"] end) as $arms |
[.[]|select(.kind=="kv_quality_generation" and (.arm as $a | $arms|index($a))!=null)] as $gens |
require(([$gens[]|[.case,.arm]]|sort)==([$cases[].id as $id|$arms[]|[$id,.]]|sort); "generation identities") |
require(if $dense_only then
    ($rows|map(select(.kind=="dense_grid_complete" and .cases==48))|length)==1
else
    $rows[-1].kind=="kv_quality_complete" and $rows[-1].cases==48 and
    $rows[-1].generation_rows==144 and $rows[-1].comparison_rows==96 and
    $rows[-1].initial_tracked_device_bytes==0 and $rows[-1].final_tracked_device_bytes==0 and
    $rows[-1].generated_tokens==([$gens[].generated_ids|length]|add) and
    ([$rows[]|select(.kind=="kv_quality_case_complete" and .comparison_arms==2 and .generation_arms==3)|.case]|sort)==($cases|map(.id)|sort)
end; "completion or teardown") |
($cases|INDEX(.id)) as $by_id |
[$gens[] | . as $row | $by_id[.case] as $case | (.generated_ids|length) as $count |
 require(.context==$case.context and .metrics.prompt_tokens==($case.prompt_ids|length) and
         $count>0 and $count<=$case.task.max_tokens and .metrics.generated_tokens==$count and
         (if (.generated_ids[-1]==151645 or .generated_ids[-1]==151643) then .metrics.stop_reason=="stop"
          else .metrics.stop_reason=="length" and $count==$case.task.max_tokens end) and
         all(.generated_ids[:-1][]; .!=151645 and .!=151643); "token accounting or stop") |
 require(.score==(.output|rescore($case.task.scoring)); "independent output score") |
 (.generated_ids|repetition) as $rep |
 require(.repetition.suffix_pattern_width==$rep.suffix_pattern_width and
         .repetition.suffix_pattern_copies==$rep.suffix_pattern_copies and .repetition.analyzed_tokens==$rep.analyzed_tokens and
         ((.repetition.repeated_fourgram_fraction-$rep.repeated_fourgram_fraction)|abs)<1e-12; "repetition") |
 (.cache|check_cache($case;$row.arm;$count)) as $cache |
 (.dispatch|check_dispatch($row.arm;$count)) as $dispatch |
 {case,arm,context,family:$case.task.family,tokens:$count,stop:.metrics.stop_reason,passed:.score.passed,
  repeated_fourgram_fraction:$rep.repeated_fourgram_fraction,suffix_pattern_copies:$rep.suffix_pattern_copies}] as $results |
{cases:48,generation_rows:($results|length),dense_only:$dense_only,by_family:($results|group_by([.context,.arm,.family])|map({
  context:.[0].context,arm:.[0].arm,family:.[0].family,total:length,passed:(map(select(.passed))|length),
  natural_stops:(map(select(.stop=="stop"))|length),tokens:(map(.tokens)|add)})),results:$results}
