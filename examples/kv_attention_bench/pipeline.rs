use super::{
    capture::LayerSample,
    reference::{Backing, Shape},
};
use anyhow::{ensure, Result};
use m40_llm::{
    cuda::{
        CudaContext, CudaEvent, CudaStream, DeviceBuffer, ExactBlockStagingWorkspace,
        ExactOldBacking, KVCache,
    },
    kv_compression::{KvCompressionConfig, KvRepresentativePolicy, ScopedRuntimeConfig},
};
use serde::Serialize;
use serde_json::{json, Value};
use std::{ffi::c_void, time::Instant};

const GUARD_WORDS: usize = 32;
const SENTINEL: u32 = 0x4a123456;

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Arm {
    Staged,
    Direct,
    BatchedDirect,
    Dense,
}

pub struct StagedReadback {
    pub keys: Vec<u16>,
    pub values: Vec<u16>,
    pub positions: Vec<u32>,
    pub counts: Vec<u32>,
}

struct GuardedF32 {
    allocation: DeviceBuffer,
    length: usize,
}

impl GuardedF32 {
    fn new(ctx: &CudaContext, length: usize) -> Result<Self> {
        let result = Self {
            allocation: DeviceBuffer::new_tagged(
                ctx,
                (length + 2 * GUARD_WORDS) * 4,
                "kv_probe:guarded",
            )?,
            length,
        };
        result.reset(ctx)?;
        Ok(result)
    }
    fn pointer(&self) -> *mut c_void {
        // SAFETY: the allocation always includes both 32-word guards.
        unsafe {
            self.allocation
                .as_mut_ptr()
                .cast::<u32>()
                .add(GUARD_WORDS)
                .cast()
        }
    }
    fn reset(&self, ctx: &CudaContext) -> Result<()> {
        let data = vec![SENTINEL; self.length + 2 * GUARD_WORDS];
        unsafe {
            ctx.memcpy_h2d(
                self.allocation.as_mut_ptr(),
                data.as_ptr().cast(),
                data.len() * 4,
            )
        }
    }
    fn read(&self, ctx: &CudaContext) -> Result<Vec<f32>> {
        let mut data = vec![0u32; self.length + 2 * GUARD_WORDS];
        unsafe {
            ctx.memcpy_d2h(
                data.as_mut_ptr().cast(),
                self.allocation.as_ptr(),
                data.len() * 4,
            )?;
        }
        ensure!(
            data[..GUARD_WORDS]
                .iter()
                .chain(&data[GUARD_WORDS + self.length..])
                .all(|v| *v == SENTINEL),
            "attention output/input guard changed"
        );
        Ok(data[GUARD_WORDS..GUARD_WORDS + self.length]
            .iter()
            .map(|v| f32::from_bits(*v))
            .collect())
    }
}

pub struct Case {
    pub ctx: CudaContext,
    pub shape: Shape,
    pub backing: Backing,
    pub sequences: usize,
    pub encoded: KVCache,
    pub dense: KVCache,
    pub staging: Vec<ExactBlockStagingWorkspace>,
    query: GuardedF32,
    output: GuardedF32,
    sequence_ids: DeviceBuffer,
    sequence_lengths: DeviceBuffer,
    pub start: CudaEvent,
    pub stop: CudaEvent,
    expected_queries: Vec<f32>,
    pub setup: Value,
}

impl Case {
    pub fn new(
        ctx: &CudaContext,
        shape: Shape,
        backing: Backing,
        sample: &LayerSample,
    ) -> Result<Self> {
        shape.validate()?;
        ensure!(
            [64, 128].contains(&shape.head_dim),
            "unsupported head width"
        );
        ensure!(
            backing != Backing::Q8 || shape.head_dim == 64,
            "Q8 head128 is unsupported"
        );
        let sequences = sample.keys.len();
        ensure!(
            (1..=8).contains(&sequences)
                && sample.values.len() == sequences
                && sample.queries.len() == sequences,
            "fixture sequence count differs"
        );
        for seq in 0..sequences {
            ensure!(
                sample.keys[seq].len() == shape.length * shape.kv_width()
                    && sample.values[seq].len() == sample.keys[seq].len()
                    && sample.queries[seq].len() == shape.query_width(),
                "fixture shape differs"
            );
            ensure!(
                sample.keys[seq]
                    .iter()
                    .chain(&sample.values[seq])
                    .chain(&sample.queries[seq])
                    .all(|v| v.is_finite()),
                "non-finite fixture"
            );
        }
        let allocation_start = Instant::now();
        let before = CudaContext::total_device_bytes();
        let _dense_config = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
        let encoded = KVCache::new_compressed_with_context(
            ctx,
            shape.length as u32,
            sequences as u32,
            shape.kv_heads as u32,
            shape.head_dim as u32,
            shape.recent as u32,
            shape.block as u32,
            shape.top as u32,
            0,
            KvRepresentativePolicy::Last,
            match backing {
                Backing::Q8 => ExactOldBacking::Q8,
                Backing::Mixed => ExactOldBacking::Fp16KQ4V,
            },
        )?;
        let dense = KVCache::new_with_context(
            ctx,
            shape.length as u32,
            sequences as u32,
            shape.kv_heads as u32,
            shape.head_dim as u32,
        )?;
        let staging = (0..sequences)
            .map(|_| {
                ExactBlockStagingWorkspace::new(
                    ctx,
                    shape.query_heads as u32,
                    shape.head_dim as u32,
                    shape.capacity() as u32,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let query = GuardedF32::new(ctx, sequences * shape.query_width())?;
        let output = GuardedF32::new(ctx, sequences * shape.query_width())?;
        let sequence_ids = DeviceBuffer::new_tagged(ctx, sequences * 4, "kv_probe:sequence_ids")?;
        let sequence_lengths = DeviceBuffer::new_tagged(ctx, sequences * 4, "kv_probe:lengths")?;
        let mut result = Self {
            ctx: ctx.clone(),
            shape,
            backing,
            sequences,
            encoded,
            dense,
            staging,
            query,
            output,
            sequence_ids,
            sequence_lengths,
            start: ctx.create_event()?,
            stop: ctx.create_event()?,
            expected_queries: sample.queries.iter().flatten().copied().collect(),
            setup: Value::Null,
        };
        let allocation_ms = allocation_start.elapsed().as_secs_f64() * 1000.0;
        let query_start = Instant::now();
        result.upload_queries()?;
        let query_ms = query_start.elapsed().as_secs_f64() * 1000.0;
        let mut uploads = 0.0;
        let mut encoded_append_wall = 0.0;
        let mut dense_append_wall = 0.0;
        let mut encoded_append_gpu = 0.0;
        let mut dense_append_gpu = 0.0;
        for seq in 0..sequences {
            let count = shape.length * shape.kv_width();
            let keys = DeviceBuffer::new_tagged(ctx, count * 4, "kv_probe:input_keys")?;
            let values = DeviceBuffer::new_tagged(ctx, count * 4, "kv_probe:input_values")?;
            let began = Instant::now();
            unsafe {
                ctx.memcpy_h2d(
                    keys.as_mut_ptr(),
                    sample.keys[seq].as_ptr().cast(),
                    count * 4,
                )?;
                ctx.memcpy_h2d(
                    values.as_mut_ptr(),
                    sample.values[seq].as_ptr().cast(),
                    count * 4,
                )?;
            }
            uploads += began.elapsed().as_secs_f64() * 1000.0;
            // Explicit completion on either success or error before source
            // buffers leave this scope. Append uses the already-rotated K with
            // zero RoPE frequency, preserving numeric identity.
            let append_result = (|| -> Result<()> {
                for (cache, wall, gpu) in [
                    (
                        &result.encoded,
                        &mut encoded_append_wall,
                        &mut encoded_append_gpu,
                    ),
                    (&result.dense, &mut dense_append_wall, &mut dense_append_gpu),
                ] {
                    let began = Instant::now();
                    result.start.record(CudaStream::Decode)?;
                    for token in 0..shape.length {
                        unsafe {
                            cache.append_token_f32_rope_k_at_async(
                                ctx,
                                seq as u32,
                                keys.as_ptr()
                                    .cast::<f32>()
                                    .add(token * shape.kv_width())
                                    .cast(),
                                values
                                    .as_ptr()
                                    .cast::<f32>()
                                    .add(token * shape.kv_width())
                                    .cast(),
                                token as u32,
                                token as u32,
                                10000.0,
                                0.0,
                            )?;
                        }
                    }
                    result.stop.record(CudaStream::Decode)?;
                    *gpu += f64::from(result.start.elapsed_sync(&result.stop, "kv_probe_append")?);
                    *wall += began.elapsed().as_secs_f64() * 1000.0;
                }
                Ok(())
            })();
            ctx.synchronize_stream(CudaStream::Decode)?;
            append_result?;
            ensure!(
                result.encoded.debug_sequence_length(ctx, seq as u32)? == shape.length as u32
                    && result.dense.debug_sequence_length(ctx, seq as u32)? == shape.length as u32,
                "append length differs"
            );
        }
        let tracked = CudaContext::total_device_bytes() - before;
        result.setup = json!({"allocation_and_guard_initialization_ms":allocation_ms,"query_upload_and_stream_dependency_ms":query_ms,
            "kv_upload_wall_ms":uploads,"encoded_append_wall_ms":encoded_append_wall,"encoded_append_gpu_ms":encoded_append_gpu,
            "dense_append_wall_ms":dense_append_wall,"dense_append_gpu_ms":dense_append_gpu,
            "uploaded_kv_bytes":sequences*shape.length*shape.kv_width()*8,
            "tracked_auxiliary_device_bytes":tracked,"native_cache_payload_bytes":result.encoded.actual_bytes()+result.dense.actual_bytes(),
            "peak_input_transfer_device_bytes":shape.length*shape.kv_width()*8,
            "encoded_cache_bytes":result.encoded.actual_bytes(),"dense_cache_bytes":result.dense.actual_bytes(),
            "q8_payload_bytes":result.encoded.q8_old_backing_bytes(),"q8_scale_bytes":result.encoded.q8_old_backing_scale_bytes(),
            "old_k_fp16_bytes":result.encoded.old_k_fp16_bytes(),"q4_v_payload_bytes":result.encoded.q4_old_v_payload_bytes(),
            "q4_v_scale_bytes":result.encoded.q4_old_v_scale_bytes(),"recent_fp16_bytes":result.encoded.recent_fp16_bytes(),
            "summary_index_bytes":result.encoded.summary_index_bytes(),
            "staging_bytes":result.staging.iter().map(|s|s.bytes()).sum::<usize>(),
            "residency_scope":"both arms retain the same caches, staging and guarded output; native cache payload excludes allocator/context overhead",
            "rope":"frequency zero on already-rotated FP16 keys; no second positional rotation"});
        Ok(result)
    }

    fn upload_queries(&self) -> Result<()> {
        let source = DeviceBuffer::new_tagged(
            &self.ctx,
            self.expected_queries.len() * 4,
            "kv_probe:query_source",
        )?;
        let ids: Vec<u32> = (0..self.sequences as u32).collect();
        let lengths = vec![self.shape.length as u32; self.sequences];
        unsafe {
            self.ctx.memcpy_h2d(
                source.as_mut_ptr(),
                self.expected_queries.as_ptr().cast(),
                self.expected_queries.len() * 4,
            )?;
            self.ctx.memcpy_h2d(
                self.sequence_ids.as_mut_ptr(),
                ids.as_ptr().cast(),
                ids.len() * 4,
            )?;
            self.ctx.memcpy_h2d(
                self.sequence_lengths.as_mut_ptr(),
                lengths.as_ptr().cast(),
                lengths.len() * 4,
            )?;
        }
        let result = (|| -> Result<()> {
            unsafe {
                self.ctx.memcpy_d2d_async(
                    self.query.pointer(),
                    source.as_ptr(),
                    self.expected_queries.len() * 4,
                    CudaStream::Prefill,
                )?;
            }
            self.ctx.stream_wait_for_stream(
                CudaStream::Decode,
                CudaStream::Prefill,
                "kv_probe_query",
            )?;
            self.ctx.synchronize_stream(CudaStream::Decode)
        })();
        self.ctx.synchronize_stream(CudaStream::Prefill)?;
        result
    }

    pub fn enqueue(&self, arm: Arm, batch: usize) -> Result<()> {
        ensure!(
            [1, 2, 4, 8].contains(&batch) && batch <= self.sequences,
            "invalid attention batch"
        );
        let s = self.shape;
        if matches!(arm, Arm::BatchedDirect) {
            ensure!(
                self.backing == Backing::Mixed,
                "batched Q8 direct unsupported"
            );
            unsafe {
                return self.encoded.attention_last_token_f32_gqa_block_select_exact_fp16_k_q4_v_old_direct_batched_async(
                &self.ctx,self.sequence_ids.as_ptr().cast(),self.sequence_lengths.as_ptr().cast(),batch as u32,
                self.query.pointer(),s.query_heads as u32,s.recent as u32,s.block as u32,s.top as u32,self.output.pointer());
            }
        }
        for seq in 0..batch {
            unsafe {
                let query = self
                    .query
                    .pointer()
                    .cast::<f32>()
                    .add(seq * s.query_width())
                    .cast();
                let output = self
                    .output
                    .pointer()
                    .cast::<f32>()
                    .add(seq * s.query_width())
                    .cast();
                match (arm,self.backing) {
                    (Arm::Staged,Backing::Q8)=>self.encoded.attention_last_token_f32_gqa_block_select_exact_staged_q8_old_with_buffers_async(
                        &self.ctx,seq as u32,query,s.query_heads as u32,s.length as u32,s.recent as u32,s.block as u32,s.top as u32,self.staging[seq].ptrs(),output)?,
                    (Arm::Staged,Backing::Mixed)=>self.encoded.attention_last_token_f32_gqa_block_select_exact_staged_fp16_k_q4_v_old_with_buffers_async(
                        &self.ctx,seq as u32,query,s.query_heads as u32,s.length as u32,s.recent as u32,s.block as u32,s.top as u32,self.staging[seq].ptrs(),output)?,
                    (Arm::Direct,Backing::Q8)=>self.encoded.attention_last_token_f32_gqa_block_select_exact_q8_old_direct_async(
                        &self.ctx,seq as u32,query,s.query_heads as u32,s.length as u32,s.recent as u32,s.block as u32,s.top as u32,output)?,
                    (Arm::Direct,Backing::Mixed)=>self.encoded.attention_last_token_f32_gqa_block_select_exact_fp16_k_q4_v_old_direct_async(
                        &self.ctx,seq as u32,query,s.query_heads as u32,s.length as u32,s.recent as u32,s.block as u32,s.top as u32,output)?,
                    (Arm::Dense,_)=>self.dense.attention_last_token_f32_gqa_async(&self.ctx,seq as u32,query,s.query_heads as u32,s.length as u32,output)?,
                    (Arm::BatchedDirect,_)=>unreachable!(),
                }
            }
        }
        Ok(())
    }

    pub fn read_arm(&self, arm: Arm, batch: usize) -> Result<Vec<f32>> {
        self.ctx.synchronize_stream(CudaStream::Decode)?;
        self.output.reset(&self.ctx)?;
        self.enqueue(arm, batch)?;
        let output = self.read_output(batch)?;
        self.verify_queries()?;
        Ok(output)
    }

    pub fn read_output(&self, batch: usize) -> Result<Vec<f32>> {
        ensure!(
            [1, 2, 4, 8].contains(&batch) && batch <= self.sequences,
            "invalid output batch"
        );
        self.ctx.synchronize_stream(CudaStream::Decode)?;
        let output = self.output.read(&self.ctx)?;
        ensure!(
            output[batch * self.shape.query_width()..]
                .iter()
                .all(|v| v.to_bits() == SENTINEL),
            "inactive attention row changed"
        );
        Ok(output[..batch * self.shape.query_width()].to_vec())
    }

    pub fn verify_queries(&self) -> Result<()> {
        let actual = self.query.read(&self.ctx)?;
        ensure!(
            actual
                .iter()
                .zip(&self.expected_queries)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "attention mutated query input"
        );
        Ok(())
    }

    pub fn read_staging(&self, sequence: usize) -> Result<StagedReadback> {
        ensure!(sequence < self.sequences, "invalid staging sequence");
        self.ctx.synchronize_stream(CudaStream::Decode)?;
        let s = self.shape;
        let ptrs = self.staging[sequence].ptrs();
        let mut keys = vec![0u16; s.query_heads * s.capacity() * s.head_dim];
        let mut values = vec![0u16; keys.len()];
        let mut positions = vec![0u32; s.query_heads * s.capacity()];
        let mut counts = vec![0u32; s.query_heads];
        unsafe {
            self.ctx
                .memcpy_d2h(keys.as_mut_ptr().cast(), ptrs.staged_k, keys.len() * 2)?;
            self.ctx
                .memcpy_d2h(values.as_mut_ptr().cast(), ptrs.staged_v, values.len() * 2)?;
            self.ctx.memcpy_d2h(
                positions.as_mut_ptr().cast(),
                ptrs.staged_positions,
                positions.len() * 4,
            )?;
            self.ctx.memcpy_d2h(
                counts.as_mut_ptr().cast(),
                ptrs.staged_counts,
                counts.len() * 4,
            )?;
        }
        Ok(StagedReadback {
            keys,
            values,
            positions,
            counts,
        })
    }
}

impl Drop for Case {
    fn drop(&mut self) {
        let _ = self.ctx.synchronize_stream(CudaStream::Decode);
        let _ = self.ctx.synchronize_stream(CudaStream::Prefill);
    }
}
