def require($test; $message): if $test then . else error($message) end;
def rows($kind): [.[] | select(.kind == $kind)];
. as $all
| ["control","down","vocabulary","argmax","all"] as $arms
| require(length > 0 and .[0].kind == "joint_only_manifest"; "manifest missing")
| require(.[-1].kind == "portfolio_quality_complete" and .[-1].is_gate_passed and .[-1].device_bytes_after == 0; "quality incomplete")
| require((rows("portfolio_quality_arm_complete") | map(.arm) | sort) == ($arms | sort); "arm coverage")
| require((rows("portfolio_quality_arm_complete") | all(.is_gate_passed and .tasks == 12 and .dense_valid == 7 and .inconclusive == 5 and .device_bytes_after == 0)); "arm gate failed")
| require((rows("portfolio_load") | length) == 5; "load coverage")
| require((rows("portfolio_load") | all(.packets.resident_bytes == 1192935424 and .packets.combined_prepared_bytes == 4110516224 and (.packets.tensors | length) == 37 and .packet_setup_counters.device_allocs == 74 and .packet_setup_counters.device_frees == 0)); "resource mismatch")
| require((rows("portfolio_logits") | length) == 35; "logit coverage")
| require((rows("portfolio_logits") | all(.is_bitwise_equal and .vocabulary == 151936 and .sha256 == .original_sha256)); "logits mismatch")
| require((rows("portfolio_task") | length) == 60; "task coverage")
| require((rows("portfolio_task") | all(.is_equal)); "task mismatch")
| require((rows("portfolio_kv") | length) == 5; "KV coverage")
| require((rows("portfolio_kv") | all(.is_bitwise_equal and .valid_words == 10616832 and .length == 576)); "KV mismatch")
| require((rows("portfolio_kv") | map(.sha256) | unique | length) == 1; "KV hashes differ")
| ($frozen[] | select(.kind == "q8_runtime_browser_generation" and .label == "candidate")) as $reference
| require((rows("portfolio_generation") | length) == 5; "generation coverage")
| require((rows("portfolio_generation") | all(.generated_ids == $reference.generation.generated_token_ids and .prompt_ids == $reference.prompt_token_ids and .output == $reference.generation.output and .metrics.generated_tokens == 512 and .metrics.stop_reason == "length" and .prefill_mode == "packed-prefix-dequant-sgemm" and .prefix_bytes == 90177536)); "actual workload differs")
| {is_passed:true, revision:.[0].revision, executable_sha256:.[0].executable_sha256,
   model_sha256:.[0].model_sha256, arms:$arms, full_logit_vectors:35, vocabulary:151936,
   valid_kv_words_per_arm:10616832, frozen_tasks_per_arm:12, dense_valid_per_arm:7,
   loads:(rows("portfolio_load") | map({arm, original_load_ms, packet_preparation_ms:.packets.preparation_ms,
     packet_bytes:.packets.resident_bytes, combined_prepared_bytes:.packets.combined_prepared_bytes,
     temporary_packet_host_payload_bytes:.packets.peak_temporary_host_bytes,
     tracked_device_delta, process_lifetime_host_high_water}))}
