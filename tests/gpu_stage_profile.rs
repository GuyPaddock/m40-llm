#![cfg(all(feature = "cuda", nvcc))]

use std::{ffi::c_void, path::PathBuf, sync::Mutex};

use anyhow::{ensure, Result};
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer},
    generate::{generate_text, GenerateOptions, PromptFormat},
    gpu_profile::{self, DecodeGpuProfiler, GpuProfileConfig, GpuStageSpan, TokenGpuProfile},
    infer::LoadedModel,
    kv_compression::KvCompressionConfig,
    profile,
};

#[path = "common/tiny_gguf.rs"]
mod tiny_gguf;

static TEST_LOCK: Mutex<()> = Mutex::new(());

struct EnvRestore {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvRestore {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(key);
        std::env::set_var(key, value);
        Self { key, previous }
    }
}

impl Drop for EnvRestore {
    fn drop(&mut self) {
        match &self.previous {
            Some(value) => std::env::set_var(self.key, value),
            None => std::env::remove_var(self.key),
        }
    }
}

fn ctx_m40() -> Result<CudaContext> {
    let ctx = CudaContext::new(-1)?;
    let props = ctx.current_device_props()?;
    ensure!(
        props.major == 5 && props.minor == 2,
        "stage-profile hardware tests require sm_52"
    );
    Ok(ctx)
}

/// Completed event queries must preserve counters and reject foreign contexts.
#[test]
fn test_event_query_does_not_synchronize() -> Result<()> {
    let _lock = TEST_LOCK.lock().unwrap();
    let ctx = ctx_m40()?;
    let start = ctx.create_event()?;
    let stop = ctx.create_event()?;
    let data = DeviceBuffer::new_tagged(&ctx, 16, "event_test")?;
    unsafe {
        ctx.memcpy_h2d(data.as_mut_ptr(), [1.0_f32; 4].as_ptr().cast(), 16)?;
    }
    start.record(CudaStream::Decode)?;
    unsafe {
        ctx.residual_add_f32_async(data.as_ptr(), data.as_ptr(), data.as_mut_ptr(), 4)?;
    }
    stop.record(CudaStream::Decode)?;
    let before_query = profile::snapshot();
    let _pending_or_ready = start.elapsed_ready(&stop)?;
    assert_eq!(profile::snapshot(), before_query);
    ctx.synchronize_stream(CudaStream::Decode)?;
    let before_resolve = profile::snapshot();
    let elapsed = start.elapsed_ready(&stop)?.expect("completed events");
    assert!(elapsed.is_finite() && elapsed >= 0.0);
    assert_eq!(profile::snapshot(), before_resolve);
    let other = ctx_m40()?;
    assert!(start.elapsed_ready(&other.create_event()?).is_err());
    Ok(())
}

/// Reuse event pairs across tokens while preserving an explicit two-stream data
/// dependency; resolving timestamps adds no synchronization or other GPU work.
#[test]
fn test_buffered_profile_preserves_two_stream_result_and_reuses_events() -> Result<()> {
    let _lock = TEST_LOCK.lock().unwrap();
    let ctx = ctx_m40()?;
    let mut profiler = DecodeGpuProfiler::new(
        &ctx,
        1,
        GpuProfileConfig {
            generated_steps: vec![0, 1],
            report_path: None,
        },
    )?;
    let a = DeviceBuffer::new_tagged(&ctx, 16, "stage_test:a")?;
    let b = DeviceBuffer::new_tagged(&ctx, 16, "stage_test:b")?;
    let intermediate = DeviceBuffer::new_tagged(&ctx, 16, "stage_test:intermediate")?;
    let out = DeviceBuffer::new_tagged(&ctx, 16, "stage_test:out")?;
    unsafe {
        ctx.memcpy_h2d(a.as_mut_ptr(), [1.0_f32; 4].as_ptr().cast(), 16)?;
        ctx.memcpy_h2d(b.as_mut_ptr(), [1.0_f32, 0.0, 0.0, 1.0].as_ptr().cast(), 16)?;
    }
    let mut pool_capacity = None;
    for step in 0..2 {
        let scope = profiler.begin_token(step, 10 + step)?;
        gpu_profile::set_layer(Some(0));
        let stage = GpuStageSpan::begin("decode_residual", CudaStream::Decode)?;
        unsafe {
            ctx.residual_add_f32_async(a.as_ptr(), a.as_ptr(), intermediate.as_mut_ptr(), 4)?;
        }
        stage.finish()?;
        ctx.stream_wait_for_stream(
            CudaStream::Prefill,
            CudaStream::Decode,
            "stage_test_dependency",
        )?;
        let stage = GpuStageSpan::begin("prefill_projection", CudaStream::Prefill)?;
        unsafe {
            ctx.gemm_f32xf32_f32_stream_async(
                intermediate.as_ptr(),
                b.as_ptr(),
                out.as_mut_ptr(),
                2,
                2,
                2,
                CudaStream::Prefill,
            )?;
        }
        stage.finish()?;
        ctx.synchronize_stream(CudaStream::Prefill)?;
        let before_resolve = profile::snapshot();
        let report = profiler.finish_token(scope)?;
        assert_eq!(profile::snapshot(), before_resolve);
        assert_eq!(report.stages.len(), 2);
        assert_eq!(report.stages[0].stage, "decode_residual");
        assert_eq!(report.stages[0].stream, "decode");
        assert_eq!(report.stages[1].stage, "prefill_projection");
        assert_eq!(report.stages[1].stream, "prefill");
        assert_eq!(report.counters.stream_syncs, 1);
        assert_eq!(report.counters.stream_waits, 1);
        assert_eq!(report.counters.cublas_calls, 1);
        assert_eq!(report.event_pool_reused, step != 0);
        assert!(report.gpu_stage_union_ms <= report.gpu_stage_sum_ms + 0.001);
        assert!(report
            .host_timings
            .iter()
            .any(|timing| timing.operation == "stream_synchronize"));
        if let Some(capacity) = pool_capacity {
            assert_eq!(report.event_pool_capacity, capacity);
        } else {
            pool_capacity = Some(report.event_pool_capacity);
        }
        let mut values = [0.0_f32; 4];
        unsafe {
            ctx.memcpy_d2h(values.as_mut_ptr().cast::<c_void>(), out.as_ptr(), 16)?;
        }
        assert_eq!(values, [2.0; 4]);
    }
    Ok(())
}

/// Sampled profiling preserves generated IDs and launch/sync/copy/allocation
/// counts; disabled profiling creates no report, and layer labels survive reuse.
/// Graph capture is rejected rather than producing incomplete attribution.
#[test]
fn test_sampled_generation_profile_matches_unprofiled_generation() -> Result<()> {
    let _lock = TEST_LOCK.lock().unwrap();
    let _ctx = ctx_m40()?;
    let path: PathBuf =
        std::env::temp_dir().join(format!("m40-stage-profile-{}.jsonl", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let _env = [
        EnvRestore::set("M40LLM_STAGE_PROFILE", "0"),
        EnvRestore::set("M40LLM_STAGE_PROFILE_REPORT", path.to_str().unwrap()),
        EnvRestore::set("M40LLM_STAGE_PROFILE_TOKENS", "0,1,3"),
        EnvRestore::set("M40LLM_DECODE_GRAPH", "0"),
        EnvRestore::set("M40LLM_SERVER_BATCH_DECODE", "0"),
        EnvRestore::set("M40LLM_CUDA_GREEDY_ARGMAX", "1"),
        EnvRestore::set("M40LLM_DECODE_CUBLAS_STREAM", "prefill"),
        EnvRestore::set("M40LLM_F16_DECODE_KERNEL", "0"),
    ];
    let (gguf, bytes) = tiny_gguf::make_ascii_tiny_gguf(tiny_gguf::TinyGgufConfig {
        vocab: 128,
        d_model: 128,
        head_count: 2,
        block_count: 3,
        context_length: 32,
        ..Default::default()
    });
    let mut model = LoadedModel::from_gguf(gguf, bytes, -1)?;
    model.allocate_kv_cache_for_layers(32)?;
    let options = GenerateOptions {
        prompt: "BC".to_owned(),
        prompt_format: PromptFormat::Raw,
        max_tokens: Some(4),
        top_k: Some(1),
        kv_compression: KvCompressionConfig::dense_reference(),
        ..Default::default()
    };
    generate_text(&model, options.clone())?;
    let before = profile::snapshot();
    let reference = generate_text(&model, options.clone())?;
    let reference_counts = profile::counter_delta(&before, &profile::snapshot());
    assert!(!path.exists());
    let _enabled = EnvRestore::set("M40LLM_STAGE_PROFILE", "1");
    let before = profile::snapshot();
    let measured = generate_text(&model, options.clone())?;
    let measured_counts = profile::counter_delta(&before, &profile::snapshot());
    assert_eq!(measured.token_ids, reference.token_ids);
    assert_eq!(measured.output, "AAAA");
    assert_eq!(measured_counts, reference_counts);
    let reports: Vec<TokenGpuProfile> = std::fs::read_to_string(&path)?
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    assert_eq!(
        reports
            .iter()
            .map(|report| report.generated_step)
            .collect::<Vec<_>>(),
        [0, 1, 3]
    );
    for (index, report) in reports.iter().enumerate() {
        assert_eq!(report.absolute_position, 1 + report.generated_step);
        assert_eq!(report.event_pool_reused, index != 0);
        assert_eq!(report.counters.device_allocs, 0);
        assert_eq!(report.counters.device_frees, 0);
        for layer in 0..3 {
            for stage in ["qkv_project", "attention", "mlp_down"] {
                assert!(report
                    .stages
                    .iter()
                    .any(|timing| timing.layer_id == Some(layer) && timing.stage == stage));
            }
        }
        assert!(report
            .stages
            .iter()
            .any(|timing| timing.stage == "lm_head" && timing.stream == "prefill"));
        assert!(report
            .stages
            .iter()
            .all(|timing| timing.gpu_elapsed_ms.is_finite() && timing.gpu_elapsed_ms >= 0.0));
    }
    {
        let _graph = EnvRestore::set("M40LLM_DECODE_GRAPH", "1");
        let err = generate_text(&model, options).expect_err("graph profiling must be rejected");
        assert!(format!("{err:#}").contains("Graph capture"));
    }
    std::fs::remove_file(path)?;
    Ok(())
}
