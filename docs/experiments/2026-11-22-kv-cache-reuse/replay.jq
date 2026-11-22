def check($condition; $message):
    if $condition then . else error($message) end;

def analyze($arm; $recorded):
    $recorded
    | check(length == 13 and .[0].kind == "residency_baseline"
        and .[0].arm == $arm and .[0].capacity == 3072 and .[0].cycles == 3
        and .[0].physical_layer_slots == 36 and .[0].logical_sequences == 1;
        "unexpected cohort")
    | .[0] as $base
    | check(all(.[]; .memory.rust_tracked_bytes == $base.memory.rust_tracked_bytes
        and .memory.cuda_used_bytes + .memory.cuda_free_bytes == .memory.cuda_total_bytes);
        "incompatible memory accounting")
    | ([range(1; 4) as $cycle
        | [ .[] | select(.cycle == $cycle) ]
        | check(map(.kind) == ["cache_allocated", "cache_populated", "cache_populated", "cache_released"]
            and .[0].occupied == 0 and [.[1:3][].occupied_per_layer] == [64, 3072];
            "incomplete cache lifetime")
        | {
            cycle: $cycle,
            cache_requested_bytes: .[0].payloads.known_cuda_malloc_requests_bytes,
            full_cuda_bytes: .[2].memory.cuda_used_bytes,
            full_process_mib: .[2].memory.nvml_process_mib,
            released_cuda_bytes: .[3].memory.cuda_used_bytes,
            released_process_mib: .[3].memory.nvml_process_mib
        }]) as $cycles
    | {
        arm: $arm,
        revision: $base.revision,
        executable_sha256: $base.executable_sha256,
        model_sha256: $base.model_sha256,
        environment: $base.environment,
        base: $base.memory,
        cycles: $cycles,
        first_release_cuda_delta_bytes: ($cycles[0].released_cuda_bytes - $base.memory.cuda_used_bytes)
    }
    | [.cycles[].released_cuda_bytes] as $ends
    | .disposition = (
        if ($ends | unique | length) == 1 then "PASS"
        elif $ends[2] > $ends[1] and $ends[1] > $ends[0] then "FAIL"
        else "INCONCLUSIVE" end
    );

analyze("dense"; $dense) as $control
| analyze("quantized_all"; $top64) as $candidate
| check(all(["revision", "executable_sha256", "model_sha256", "environment", "base"][];
    . as $key | $control[$key] == $candidate[$key]); "unmatched cache lifetime controls")
| {dense: $control, top64: $candidate}
