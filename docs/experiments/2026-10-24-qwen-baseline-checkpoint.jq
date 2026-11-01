# Reconcile recorded observations, not source/document text. See the companion
# report for the inputs, extrapolation assumptions, and reproduction command.
def median: sort | .[length / 2 | floor];
def require($condition; $message):
  if $condition then . else error($message) end;
def family:
  if . == "token_embd.weight" then "tied_vocabulary"
  elif test("ffn_(gate|up)\\.weight$") then "gate_up"
  elif test("ffn_down\\.weight$") then "down"
  elif test("attn_(q|k|v)\\.weight$") then "qkv"
  elif test("attn_output\\.weight$") then "attention_output"
  elif test("(norm\\.weight|attn_(q|k|v)\\.bias)$") then "norms_biases"
  else error("unclassified tensor") end;

[$control[] | select(.kind == "manifest")] as $manifests |
[$control[] | select(.kind == "summary")] as $summaries |
[$control[] | select(.kind == "tokenizer_oracle")] as $oracles |
[$perturbation[] | select(.kind == "profile_comparison_summary")] as $profile_summaries |
require(($manifests|length)==1 and ($summaries|length)==1 and
        ($oracles|length)==1 and ($profile_summaries|length)==1 and
        ($hardware|length)==1; "missing or duplicate checkpoint inputs") |
$manifests[0] as $manifest |
$summaries[0].comparison as $comparison |
[$control[] | select(.kind == "trial" and .phase == "screen")] as $trials |
[$trials[] | select(.runtime == "m40-llm")] as $m40 |
[$trials[] | select(.runtime == "ollama")] as $ollama |
[$control[] | select(.kind == "token_trace")] as $traces |
require(($trials | length) == 10 and ($m40 | length) == 5 and
        ($ollama | length) == 5; "expected five pairs") |
require(all($trials[];
  .is_fixed_workload and (.metrics.prompt_tokens // .prompt_tokens) == 65 and
  (.metrics.generated_tokens // .generated_tokens) == 512 and
  (.metrics.stop_reason // .stop_reason) == "length" and
  .cache.reused_prompt_tokens == 0 and .cache.evaluated_prompt_tokens == 65 and
  ((.e2e_tps - 512000 / .client_elapsed_ms) | fabs) < 1e-9);
  "invalid workload, timing, or cache evidence") |
require(($traces | length) == 2 and all($traces[];
  .matches_measured_output and (.generated_token_ids | length) == 512 and
  .prompt_token_ids == $manifest.prompt_token_ids);
  "missing actual matching token traces") |
require(all($traces[]; . as $trace |
  all($trials[] | select(.runtime == $trace.runtime); .output == $trace.output));
  "trace output differs from measured continuation") |
require($oracles[0].is_equal and
  $oracles[0].prompt_token_ids == $manifest.prompt_token_ids;
  "tokenizer oracle mismatch") |
require(($manifest.prompt_token_ids | length) == 65 and
        $manifest.prompt_token_ids[0] != $manifest.replacement_token_ids[0];
        "replacement prompt shares a prefix") |
require([$m40[].e2e_tps] | median == $comparison.candidate.median;
        "m40 median mismatch") |
require([$ollama[].e2e_tps] | median == $comparison.control.median;
        "Ollama median mismatch") |

($manifest.tensor_manifest | map(. + {
  family: (.name | family),
  elements: (.shape | reduce .[] as $d (1; . * $d))
}) | map(. + {bytes: (.elements *
  (if .dtype == "Q8_0" then 34 / 32 elif .dtype == "F32" then 4
   else error("unhandled tensor dtype") end))})) as $tensors |
require(all($tensors[] | select(.dtype == "Q8_0"); .elements % 32 == 0);
        "partial Q8 blocks require explicit accounting") |
([$tensors[] | select(.dtype == "Q8_0") | .bytes] | add) as $q8_bytes |
([$events[] | select(.session_id >= 2 and .session_id <= 6)]) as $measured |
require(($measured | length) == 15 and all($measured[];
  .absolute_position == 64 + .generated_step and
  .counters.device_allocs == 0 and .counters.device_frees == 0 and
  .counters.stream_syncs == 1 and .counters.stream_waits == 0);
  "unexpected sampled sessions, positions, or scheduling counters") |
($measured | group_by(.generated_step) | map({
  step: .[0].generated_step,
  absolute_position: .[0].absolute_position,
  samples: length,
  wall_ms: ([.[].token_wall_ms] | median),
  gpu_union_ms: ([.[].gpu_stage_union_ms] | median),
  stages: (map(.stages | group_by(.stage) | map({
    stage: .[0].stage, ms: ([.[].gpu_elapsed_ms] | add)
  })) | flatten | group_by(.stage) | map({
    key: .[0].stage, value: ([.[].ms] | median)
  }) | from_entries)
})) as $samples |
require([$samples[].step] == [0,255,511] and all($samples[]; .samples == 5);
        "expected five early/middle/late samples") |
require(($counters | length) == 2 and all($counters[];
  .collection_status == "unavailable" and .event_group_status.code == 27 and
  .domain_status.code == 27 and .kernel_measurements == null);
  "counter availability differs from this checkpoint") |
($samples[1].stages) as $middle |
([$m40[].client_elapsed_ms] | median) as $wall |
(65 + 512 - 1) as $forwards |
(($samples[2].stages.attention - $samples[0].stages.attention) /
 ($samples[2].absolute_position - $samples[0].absolute_position)) as $slope |
($samples[0].stages.attention - $slope * 65) as $intercept |
([
  {stage:"attention", estimated_workload_ms:
    ($forwards * ($intercept + $slope * ($forwards + 1) / 2))},
  {stage:"gate_up", estimated_workload_ms:($forwards * $middle.mlp_gate_up_swiglu)},
  {stage:"down", estimated_workload_ms:($forwards * $middle.mlp_down)},
  {stage:"combined_ffn", estimated_workload_ms:
    ($forwards * ($middle.mlp_gate_up_swiglu + $middle.mlp_down))},
  {stage:"qkv", estimated_workload_ms:($forwards * $middle.qkv_project)},
  {stage:"attention_output", estimated_workload_ms:($forwards * $middle.out_project)},
  {stage:"vocabulary_argmax", estimated_workload_ms:(512 * $middle.lm_head_argmax)}
] | map(. + {
  estimated_e2e_fraction:(.estimated_workload_ms / $wall),
  idealized_e2e_speedup_if_removed:($wall / ($wall - .estimated_workload_ms))
})) as $limits |
{
  kind:"controlled_baseline_checkpoint", model_sha256:$manifest.model_sha256,
  prompt_sha256:$manifest.prompt_sha256,
  prompt_token_ids_sha256:$manifest.prompt_token_ids_sha256,
  source_files:["2026-10-24-qwen-ollama-controlled.jsonl",
    "2026-10-24-qwen-profile-comparison.jsonl",
    "2026-10-24-qwen-stage-events.jsonl.gz",
    "2026-10-24-maxwell-counter-preflight.jsonl"],
  hardware:$hardware[0], comparison:$comparison,
  original_target_tps:53.06, refreshed_target_tps:(1.3*$comparison.control.median),
  required_e2e_speedup:(1.3*$comparison.control.median/$comparison.candidate.median),
  weight_inventory:{
    parameters:([$tensors[].elements]|add),
    payload_bytes:([$tensors[].bytes]|add), q8_matrix_bytes:$q8_bytes,
    ffn_parameter_fraction:
      (([$tensors[]|select(.family=="gate_up" or .family=="down")|.elements]|add)/
       ([$tensors[].elements]|add)),
    families:($tensors|group_by(.family)|map({family:.[0].family,tensors:length,
      parameters:([.[].elements]|add),bytes:([.[].bytes]|add)}))
  },
  estimated_once_per_token_matrix_GB_s:
    ([$comparison.candidate.median,$comparison.control.median,53.06,
       1.3*$comparison.control.median]|map({tps:.,GB_s:(.*$q8_bytes/1e9)})),
  samples:$samples,
  estimated_amdahl_limits:$limits,
  extrapolation:{forward_calls:$forwards,vocabulary_calls:512,
    projection_cost:"middle-sample constant across all sequential forwards",
    attention_cost:"linear fit from early and late samples, including prompt prefix",
    is_measured_workload_stage_total:false,is_measured_dram_traffic:false},
  perturbation:$profile_summaries[0],
  counter_availability:($counters|map({effective_uid,collection_status,
    domain_status,event_group_status,kernel_measurements}))
}
