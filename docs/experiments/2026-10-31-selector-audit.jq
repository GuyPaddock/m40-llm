# Stream the full row trace. Independently recount set overlap, dense/sparse
# totals and the summary gate; check that causal selections reference prior
# recorded dense labels. This does not independently recompute down norms.
def require(condition; message): if condition then . else error(message) end;
def absolute: if . < 0 then -. else . end;
def causal(method): method == "first_token" or (method|startswith("refresh_")) or (method|contains("sketch_"));
reduce inputs as $r (
  {rows:0,summaries:0,manifests:0,completions:0,causal_sparse_rows:0,groups:{},labels:{}};
  if $r.kind == "selector_manifest" then
    .manifests += 1
    | require($r.inputs_sha256 == "3c8bc4f9f82db217c0dc38e690e0fafc4a46926ff7574b162946da7c435dd4cc"
        and $r.uses_cuda == false and $r.is_performance_measurement == false; "wrong manifest")
  elif $r.kind == "selector_row" then
    .rows += 1
    | ([$r.layer,$r.retained_percent,$r.task]|tojson) as $case
    | ([$r.layer,$r.retained_percent,$r.task,$r.row]|tojson) as $row_key
    | (if $r.retained_percent == 10 then 1101 elif $r.retained_percent == 20 then 2202 else 0 end) as $count
    | require($count > 0 and ($r.selected_channels|length)==$count
        and all($r.selected_channels[]; .>=0 and .<11008 and floor==.)
        and all(range(1;$count); $r.selected_channels[.] > $r.selected_channels[.-1]); "invalid selection")
    | if $r.method == "current_oracle" then
        if .case != $case then .labels={} | .case=$case else . end
        | .oracle_key=$row_key
        | .oracle=($r.selected_channels|map({key:tostring,value:true})|from_entries)
        | if $r.row % 8 == 0 then .labels[$r.row|tostring]=$r.selected_channels else . end
      else . end
    | require(.oracle_key==$row_key; "missing matching oracle")
    | .oracle as $oracle
    | (($r.selected_channels|map(select($oracle[tostring]==true))|length)/$count) as $recall
    | require((($recall-$r.metrics.recall)|absolute)<1e-12; "recall disagrees with selected indices")
    | ($recall>=0.8 and $r.metrics.weighted_oracle_fraction!=null and $r.metrics.weighted_oracle_fraction>=0.95) as $passes
    | require($passes==$r.metrics.passes_row_screen; "row gate disagrees")
    | if causal($r.method) then
        (if $r.method=="first_token" then null else ($r.method|split("_")[-1]|tonumber) end) as $interval
        | require($r.is_dense_refresh==(if $interval==null then $r.row==0 else $r.row%$interval==0 end); "wrong refresh schedule")
        | if $r.is_dense_refresh then
            require($r.source_rows==[$r.row]; "dense row has wrong label identity")
          else
            .causal_sparse_rows+=1
            | require(($r.source_rows|length)>0 and all($r.source_rows[]; .<$r.row); "future or missing causal labels")
            | if $r.method|startswith("refresh_frequency") then
                require($r.source_rows==[range(0;($r.row-($r.row%$interval)+1);$interval)]; "frequency references wrong history")
              else
                require(($r.source_rows|length)==1 and $r.selected_channels==.labels[$r.source_rows[0]|tostring]; "selection differs from referenced dense labels")
              end
          end
      else . end
    | reduce [$r.task,"all_tasks"][] as $scope (. ;
        ([$r.layer,$r.retained_percent,$r.method,$scope]|tojson) as $key
        | .groups[$key] //= {rows:0,dense:0,sparse:0,passing:0,recall_sum:0}
        | .groups[$key].rows+=1
        | if $r.is_dense_refresh then .groups[$key].dense+=1
          else .groups[$key].sparse+=1 | .groups[$key].recall_sum+=$recall
            | if $passes then .groups[$key].passing+=1 else . end
          end)
  elif $r.kind == "selector_summary" then
    .summaries+=1
    | ([$r.layer,$r.retained_percent,$r.method,$r.scope]|tojson) as $key
    | .groups[$key] as $g
    | require($g!=null and $g.rows==$r.summary.observed_rows and $g.dense==$r.summary.dense_refresh_rows
        and $g.sparse==$r.summary.eligible_sparse_rows and $g.passing==$r.summary.passing_sparse_rows
        and (($g.recall_sum-$r.summary.recall_sum)|absolute)<1e-9; "summary differs from row recount")
    | require($r.passes_fixed_screen==($g.sparse>0 and $g.passing*10 >= $g.sparse*9); "summary gate differs")
  elif $r.kind == "selector_complete" then
    .completions+=1
    | require($r.is_complete==true and all($r.decisions[]; .passes_all_applicable_layers==all(.layer_results[]; .[1])); "incomplete decision")
  else error("unknown analysis record") end
)
| require(.rows==8296 and .summaries==476 and .causal_sparse_rows==5382 and .manifests==1 and .completions==1; "incomplete trace counts")
| {is_valid:true,rows,summaries,causal_sparse_rows,manifests,completions,
   checks:["sorted unique in-range indices","recomputed top-k recall","dense refresh schedule","prior dense label identity","summary recount","fixed screen gate"],
   limitation:"down-column norms and weighted contribution arithmetic are covered by analyzer behavioral tests, not independently recomputed here"}
