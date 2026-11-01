# jq -s --slurpfile frozen ORIGINAL_COMPOUND_QUALITY -f THIS NEW_OWNERSHIP_REPORT
# New logical-source versus physical-residency contract, not the old audit.
def require($condition;$message): if $condition then . else error($message) end;
def one($kind): map(select(.kind==$kind))|require(length==1;"record multiplicity: \($kind)")|.[0];
def by_name: sort_by(.tensor_name);
. as $rows | one("joint_only_manifest") as $m
| $rows | one("joint_only_ownership") as $o
| $rows | one("joint_only_generation") as $g
| $rows | one("joint_only_quality_complete") as $done
| $o.control as $a | $o.candidate as $b
| $frozen[0] as $f
| ([$a.logical_sources[] | select(.tensor_name|endswith("ffn_gate.weight")) | . as $gate
    | {gate:$gate,up:($a.logical_sources | map(select(.tensor_name==($gate.tensor_name|sub("ffn_gate";"ffn_up")))) | .[0])}]
    | sort_by(.gate.tensor_name)) as $expected_pairs
| ($frozen|map(select(.kind=="q8_runtime_browser_generation" and .label=="candidate"))|.[0]) as $prior
| $rows
| require($m.schema_version==1 and $m.reference_id=="compound-runtime-63240c5-v1"
    and $m.worktree=="" and $m.build_profile=="release" and ($m.revision|test("^[a-f0-9]{40}$"))
    and ($m.executable_sha256|test("^[a-f0-9]{64}$")) and $m.model_sha256==$f.candidate.sha256
    and $m.frozen_environment==$f.environment;"source/model/config binding")
| require($a.ownership=="retain-split" and $b.ownership=="joint-only"
    and ($a.logical_sources|length)==109 and $a.logical_sources==$b.logical_sources
    and ($a.logical_sources|by_name)==([$f.prepared_candidate.tensors[].key.source]|by_name)
    and ($a.resident_split.tensors|length)==109 and ($b.resident_split.tensors|length)==37;
    "logical coverage is not physical split residency")
| require($a.resident_split.tensors==$f.prepared_candidate.tensors
    and $b.resident_split.tensors==($a.resident_split.tensors|map(select(.key.source.tensor_name|test("ffn_down.weight$|^token_embd.weight$"))))
    and $a.uploaded_gguf_bytes==3279519744 and $b.uploaded_gguf_bytes==3279519744
    and $a.resident_split.resident_bytes==2917580800 and $b.resident_split.resident_bytes==1192935424
    and $o.resident_reduction_bytes==1724645376 and $o.skipped_device_allocations==144
    and ([$b.resident_split.tensors[].resident_bytes]|add)==1192935424
    and all($b.resident_split.tensors[];.key.source.tensor_name|test("ffn_down.weight$|^token_embd.weight$"));
    "resident allocation or skipped allocations")
| require(($o.control_payload_hashes|length)==36 and $o.control_payload_hashes==$o.candidate_payload_hashes
    and all($o.control_payload_hashes[];.[1]|test("^[a-f0-9]{64}$"))
    and ([$o.control_payload_hashes[][0]]|sort_by(.gate.tensor_name))==($a.joint.pairs|sort_by(.gate.tensor_name))
    and $a.joint.pairs==$b.joint.pairs and $a.joint.resident_bytes==1724645376 and $b.joint.resident_bytes==1724645376
    and ($expected_pairs|length)==36 and all($expected_pairs[];.up!=null)
    and ($a.joint.pairs|sort_by(.gate.tensor_name))==$expected_pairs
    and $a.joint.pair_count==36 and $b.joint.pair_count==36
    and $a.joint.combined_prepared_bytes==4642226176 and $b.joint.combined_prepared_bytes==2917580800
    and $b.joint.budget_bytes==6442450944 and $b.joint.retained_host_pair_bytes==0
    and $b.joint.peak_temporary_host_pair_bytes==47906816;"joint identity/payload/budget")
| require(($g.generated_ids|length)==512 and $g.generated_ids==$prior.generation.generated_token_ids
    and $g.prompt_ids==$prior.prompt_token_ids and ($g.prompt_ids|length)==65
    and $g.output==$prior.generation.output and $g.candidate_output==$g.output
    and all([$g.control_metrics,$g.candidate_metrics][];.generated_tokens==512 and .prompt_tokens==65 and .stop_reason=="length")
    and $g.control_dispatch==$g.candidate_dispatch and $g.control_dispatch.totals==$prior.dispatch.totals
    and $g.control_dispatch.operations==$prior.dispatch.operations
    and $g.control_prefill_mode=="packed-prefix-dequant-sgemm" and $g.candidate_prefill_mode==$g.control_prefill_mode
    and $g.control_prefix_bytes==90177536 and $g.candidate_prefix_bytes==90177536;"actual512 IDs/dispatch/prefill")
| ($rows|map(select(.kind=="joint_only_logits"))) as $logits
| require(($logits|map(.step))==[0,1,8,32,128,256,511]
    and all($logits[];.vocabulary==151936 and .control_sha256==.candidate_sha256
        and (.control_sha256|test("^[a-f0-9]{64}$")) and .max_abs_diff==0 and .mean_abs_diff==0 and .top10_overlap==10);
    "sampled full-vocabulary bits")
| ($rows|map(select(.kind=="joint_only_task"))) as $tasks
| require(($tasks|length)==12 and ([$tasks[].id]|unique|length)==12
    and all($tasks[];. as $task|.has_identical_ids and .output==.candidate_output
        and ([$frozen[]|select(.kind=="model_quality_task" and .id==$task.id)
             | .candidate.output==$task.output and .candidate_score==$task.score])==[true])
    and $done.is_gate_passed and $done.tasks==12 and $done.dense_valid==7 and $done.inconclusive==5
    and $done.device_bytes_after==0;"frozen tasks/teardown")
| {is_quality_audit_passed:true,schema_version:1,revision:$m.revision,logical_sources:109,
    control_split_owners:109,candidate_split_owners:37,joint_pairs:36,skipped_device_allocations:144,
    resident_reduction_bytes:1724645376,actual512_ids_identical:true,sampled_logit_bits_identical:true,
    frozen_tasks:12,dense_valid:7,inconclusive:5,device_bytes_after:0}
