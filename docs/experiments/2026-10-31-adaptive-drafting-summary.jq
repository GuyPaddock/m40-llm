# Compact all cases without changing the frozen policy or score model.
[.[] | select(.kind == "adaptive_case") |
  {id, policy, scale,
   decisions: .visited_stats.decisions,
   attempts: .visited_stats.eligible,
   coverage: .visited_stats.eligible_coverage,
   acceptance: .visited_stats.conditional_acceptance,
   outcome_histogram: .visited_stats.rejected_guess_or_full_histogram,
   cpu_ms: ([.trials[].replay.cpu_elapsed_ns / 1000000] | {min:min,max:max}),
   reasons: ([.trials[0].decisions[].decision.reason] | group_by(.) | map({reason:.[0],count:length})),
   scenarios: [.projections.scenarios[] |
     {extra_per_attempt_ms,resident_gain,decode_only_gain,
      extrapolated_serial_rows: ([.model_samples[].extrapolated_serial_rows]|max),
      extrapolated_verifier_attempts: ([.model_samples[].extrapolated_verifier_attempts]|max)}]}] as $cases |
{case_count:($cases|length),cases:$cases,
 sensitivity:($cases|group_by(.scale)|map({scale:.[0].scale,
    policies:(group_by(.policy)|map({policy:.[0].policy,
      attempts:(map(.attempts)|add),cpu_ms_max:(map(.cpu_ms.max)|max),
      reasons:([.[].reasons[]]|group_by(.reason)|map({reason:.[0].reason,count:(map(.count)|add)})),
      qualified_resident_gains:(map(select(.scenarios[0].resident_gain.min != null and .scenarios[0].resident_gain.min >= 0.03))|map(.id))}))})) }
