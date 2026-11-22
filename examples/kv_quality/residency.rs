use super::{
    config::{validate_environment, CacheArm},
    evidence::{command_text, Report},
    execution::load_weights,
    fixture::MODEL_SHA256,
};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer},
    evaluation::identity::file_sha256,
    kv_compression::ScopedRuntimeConfig,
};
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path, time::Instant};

extern "C" {
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
}

fn snapshot() -> Result<Value> {
    let mut free = 0;
    let mut total = 0;
    // CUDA writes two native size_t values on this process's current device.
    let status = unsafe { cudaMemGetInfo(&mut free, &mut total) };
    ensure!(
        status == 0 && free <= total,
        "cudaMemGetInfo status {status}"
    );
    let owners = command_text(
        "nvidia-smi",
        &[
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
    )?;
    let pid = std::process::id();
    let mut mib = None;
    for line in owners.lines() {
        let (owner, used) = line.split_once(',').context("invalid NVML record")?;
        ensure!(
            owner.trim().parse::<u32>()? == pid,
            "another GPU process is resident"
        );
        mib = Some(used.trim().parse::<usize>()?);
    }
    Ok(
        json!({"cuda_free_bytes":free,"cuda_total_bytes":total,"cuda_used_bytes":total-free,
        "nvml_process_mib":mib.context("missing NVML attribution")?,"rust_tracked_bytes":CudaContext::total_device_bytes()}),
    )
}

pub fn measure(path: &Path, label: &str, capacity: u32, output: &Path) -> Result<()> {
    ensure!(
        !cfg!(debug_assertions) && [1024, 3072, 8448].contains(&capacity),
        "wrong profile or capacity"
    );
    ensure!(
        file_sha256(path)? == MODEL_SHA256
            && command_text("git", &["status", "--porcelain"])?.is_empty(),
        "unqualified input/source"
    );
    let settings: BTreeMap<_, _> = std::env::vars()
        .filter(|(k, _)| k.starts_with("M40LLM_") || k.starts_with("CUDA_"))
        .collect();
    validate_environment(&settings)?;
    let arm = match label {
        "dense" => CacheArm::Dense,
        "top8" => CacheArm::QuantizedTop8,
        "top64" => CacheArm::QuantizedAll,
        _ => anyhow::bail!("unknown cache arm"),
    };
    let mut report = Report::create(output)?;
    let mut model = load_weights(path)?;
    let config = arm.create_config();
    let _scope = ScopedRuntimeConfig::new(config.clone());
    ensure!(
        model.kv_cache.is_none(),
        "starting model unexpectedly owns a cache"
    );
    let slots = model.model_config.block_count;
    let kv_width = (model.model_config.attention_head_count_kv
        * model.model_config.attention_key_length) as usize;
    // Both arms retain the same explicit device inputs before the cache baseline.
    let input = DeviceBuffer::new_tagged(&model.cuda, kv_width * 4, "kv_residency:append_input")?;
    let data: Vec<f32> = (0..kv_width)
        .map(|i| ((i % 19) as f32 - 8.) / 32.)
        .collect();
    unsafe {
        model
            .cuda
            .memcpy_h2d(input.as_mut_ptr(), data.as_ptr().cast(), input.bytes())?;
    }
    model.cuda.synchronize_stream(CudaStream::Prefill)?;
    model.cuda.synchronize_stream(CudaStream::Decode)?;
    let baseline = snapshot()?;
    report.append(json!({"kind":"residency_baseline","revision":command_text("git",&["rev-parse","HEAD"])?,
        "model_sha256":MODEL_SHA256,"executable_sha256":file_sha256(&std::env::current_exe()?)?,
        "arm":arm,"capacity":capacity,"logical_sequences":1,"physical_layer_slots":slots,"environment":settings,
        "device":format!("{:?}",model.cuda.current_device_props()?),"memory":baseline,
        "input_scope":"deterministic synthetic FP32 KV append to each actual model layer slot; fixed prepared model owners; no generation"}))?;
    let start = Instant::now();
    if arm == CacheArm::Dense {
        model.allocate_kv_cache_for_layers(capacity)?;
    } else {
        model.allocate_compressed_kv_cache_for_layers(capacity, &config)?;
    }
    let allocation_ms = start.elapsed().as_secs_f64() * 1000.;
    let cache = model.kv_cache.as_ref().context("cache missing")?;
    let seq_map = slots as usize * size_of::<u32>();
    let map_excluded = if arm == CacheArm::Dense { seq_map } else { 0 };
    let categories = json!({"cache_reported_bytes":cache.actual_bytes(),"dense_equivalent_bytes":cache.dense_equivalent_bytes(),
        "dense_kv_payload_bytes":if arm==CacheArm::Dense {cache.actual_bytes()} else {0},
        "recent_fp16_bytes":cache.recent_fp16_bytes(),"old_k_fp16_bytes":cache.old_k_fp16_bytes(),
        "q4_v_codes_bytes":cache.q4_old_v_payload_bytes(),"q4_v_scale_bytes":cache.q4_old_v_scale_bytes(),
        "summaries_counts_and_seq_map_bytes":cache.summary_index_bytes(),"seq_map_bytes_excluded_from_reported_total":map_excluded,
        "known_cuda_malloc_requests_bytes":cache.actual_bytes()+map_excluded});
    report.append(
        json!({"kind":"cache_allocated","allocation_wall_ms":allocation_ms,"occupied":0,
        "memory":snapshot()?,"payloads":categories}),
    )?;
    let mut filled = 0;
    for end in [64, capacity] {
        for layer in 0..slots {
            let physical = model.kv_cache_physical_slot_for_layer_sequence(layer, 0)?;
            for position in filled..end {
                unsafe {
                    cache.append_token_f32_rope_k_at_async(
                        &model.cuda,
                        physical,
                        input.as_ptr(),
                        input.as_ptr(),
                        position,
                        position,
                        10000.,
                        1.,
                    )?;
                }
            }
        }
        model.cuda.synchronize_stream(CudaStream::Decode)?;
        for layer in 0..slots {
            ensure!(
                cache.debug_sequence_length(
                    &model.cuda,
                    model.kv_cache_physical_slot_for_layer_sequence(layer, 0)?
                )? == end,
                "incomplete populated cache"
            );
        }
        filled = end;
        report.append(
            json!({"kind":"cache_populated","occupied_per_layer":end,"memory":snapshot()?}),
        )?;
    }
    drop(model.kv_cache.take());
    report.append(json!({"kind":"cache_released","memory":snapshot()?}))?;
    Ok(())
}
