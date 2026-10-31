use super::{
    data::{hash_f32, make_logits, DISTRIBUTIONS},
    probe::{NormSource, ReductionProbe},
    summarize_scope,
};
use anyhow::Result;
use half::f16;
use m40_llm::{
    cuda::{CudaContext, DeviceBuffer},
    profile,
};
use serde_json::json;
#[path = "../../../../tests/common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "../../../../tests/common/env_restore.rs"]
mod env_restore;

/// Labeled distributions are deterministic, distinct across requests, finite,
/// and preserve the prescribed late tie and ascending-reference indices.
#[test]
fn test_logit_distributions_preserve_distinct_reproducible_rows() -> Result<()> {
    for distribution in DISTRIBUTIONS {
        let values = make_logits(151936, 4, distribution)?;
        assert_eq!(
            hash_f32(&values)?,
            hash_f32(&make_logits(151936, 4, distribution)?)?
        );
        assert_eq!(&values[..151936], make_logits(151936, 1, distribution)?);
        for row in values.chunks_exact(151936) {
            assert!(row.iter().all(|v| v.is_finite()));
            let expected = row
                .iter()
                .enumerate()
                .fold(0, |best, (i, v)| if *v > row[best] { i } else { best });
            if distribution == "ascending" {
                assert_eq!(expected, 151935);
            }
            if distribution == "late-tie" {
                assert_eq!(expected, 151936 / 3);
                assert_eq!(row[151936 / 3], row[151934]);
            }
        }
        for pair in values.chunks_exact(151936).collect::<Vec<_>>().windows(2) {
            assert_ne!(pair[0], pair[1]);
        }
    }
    assert!(make_logits(2, 1, "random").is_err());
    assert!(make_logits(32, 2, "random").is_err());
    assert!(make_logits(32, 1, "unknown").is_err());
    Ok(())
}

/// Failed independent stages contribute no optimistic saved time. Incomplete
/// case grids cannot masquerade as an eligible combined runtime candidate.
#[test]
fn test_scope_gate_requires_complete_independently_passing_stages() -> Result<()> {
    let mut screens = Vec::new();
    for stage in ["attn_norm", "ffn_norm", "argmax"] {
        for _ in 0..3 {
            screens.push(json!({"case":{"stage":stage,"rows":1},"result":{"passes_micro_screen":true,"throughput_gain":if stage=="argmax" {9.0} else {1.0}}}));
        }
    }
    let eligible = summarize_scope(&screens)?;
    assert_eq!(eligible["passes_profile_scope_gate"], true);
    assert_eq!(eligible["runtime_integration_allowed"], false);
    screens[0]["result"]["passes_micro_screen"] = json!(false);
    let partial = summarize_scope(&screens)?;
    assert_eq!(partial["passes_profile_scope_gate"], false);
    assert_eq!(partial["stages"][0]["eligible_saved_fraction"], 0.0);
    screens.pop();
    assert!(summarize_scope(&screens).is_err());
    Ok(())
}

/// The shared-runner adapter executes the real control/candidate reductions,
/// verifies independent CPU outputs and guards, dispatches both argmax stages,
/// reuses buffers without enqueue side effects, and releases all allocations.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_reduction_probe_preserves_outputs_dispatch_and_reuse() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _cache = env_restore::EnvRestore::set("M40LLM_CACHE_EXPERIMENT", "0");
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    for dtype in [0, 1] {
        let bytes: Vec<u8> = (0..2048)
            .flat_map(|i| {
                let value = 0.25 + (i % 29) as f32 * 0.03125;
                if dtype == 0 {
                    f16::from_f32(value).to_bits().to_le_bytes().to_vec()
                } else {
                    value.to_le_bytes().to_vec()
                }
            })
            .collect();
        {
            let weights =
                DeviceBuffer::new_tagged(&ctx, bytes.len(), "reduction_probe:test_weights")?;
            unsafe {
                ctx.memcpy_h2d(weights.as_mut_ptr(), bytes.as_ptr().cast(), bytes.len())?;
            }
            let resident = CudaContext::total_device_bytes();
            for rows in [1, 4] {
                {
                    let values = make_logits(2048, rows, "random")?;
                    let probe = unsafe {
                        ReductionProbe::new_norm(
                            &ctx,
                            &values,
                            rows,
                            NormSource {
                                pointer: weights.as_ptr(),
                                bytes: bytes.clone(),
                                dtype,
                                eps: 1e-5,
                            },
                        )?
                    };
                    assert_eq!(CudaContext::total_device_bytes(), resident + probe.bytes());
                    check_probe(&probe, rows, 1)?;
                }
                assert_eq!(CudaContext::total_device_bytes(), resident);
            }
        }
        assert_eq!(CudaContext::total_device_bytes(), initial);
    }
    for distribution in DISTRIBUTIONS {
        for rows in [1, 4] {
            {
                let values = make_logits(151936, rows, distribution)?;
                let probe = ReductionProbe::new_argmax(&ctx, &values, rows, 151936)?;
                assert_eq!(CudaContext::total_device_bytes(), initial + probe.bytes());
                assert_eq!(probe.describe()?["candidate_partial_payload_bytes"], 384);
                check_probe(&probe, rows, rows as u64)?;
            }
            assert_eq!(CudaContext::total_device_bytes(), initial);
        }
    }
    Ok(())
}

/// Validate results before measuring, then alternate complete arms while
/// retaining the same allocation and checking the adapter's dispatch report.
fn check_probe(probe: &ReductionProbe<'_>, rows: usize, control_launches: u64) -> Result<()> {
    let validation = probe.validate()?;
    assert_eq!(validation["control_bits_equal"], true);
    assert_eq!(validation["guards_and_immutable_read_spans"], true);
    assert_eq!(probe.describe()?["rows"], rows);
    let before = profile::snapshot();
    assert!(probe.time(true, 0).is_err());
    assert_eq!(
        profile::counter_delta(&before, &profile::snapshot()),
        profile::counter_delta(&before, &before)
    );
    for is_candidate in [false, true, false, true] {
        let timing = probe.time(is_candidate, 3)?;
        assert_eq!(timing["is_candidate"], is_candidate);
        let launches = probe.operation_counts(is_candidate);
        assert_eq!(
            timing["operations"].as_array().unwrap().len(),
            launches.len()
        );
        for (entry, (name, count)) in timing["operations"]
            .as_array()
            .unwrap()
            .iter()
            .zip(&launches)
        {
            assert_eq!(entry["name"], *name);
            assert_eq!(entry["launches"], count * 3);
        }
        if !is_candidate {
            assert_eq!(timing["enqueue_counters"]["launches"], control_launches * 3);
        }
        for key in [
            "device_allocs",
            "device_frees",
            "h2d_copies",
            "d2h_copies",
            "stream_syncs",
            "stream_waits",
            "cublas_calls",
        ] {
            assert_eq!(timing["enqueue_counters"][key], 0);
        }
        assert!(timing["gpu_ms"].as_f64().unwrap() > 0.0);
    }
    probe.verify()
}
