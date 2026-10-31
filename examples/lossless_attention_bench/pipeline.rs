use super::{
    capture::{KV_HEADS, QUERY_HEADS},
    fixture::{encode_words, Fixture, DIMENSION},
};
use anyhow::{ensure, Result};
use half::f16;
use m40_llm::{
    cuda::{
        CudaContext, CudaEvent, CudaStream, DeviceBuffer, Head128AttentionConfig,
        Head128AttentionWorkspace, KVCache, LosslessKvAttentionShape, LosslessKvDevice,
    },
    kv_compression::{KvCompressionConfig, ScopedRuntimeConfig},
};
use serde::Serialize;
use serde_json::{json, Value};
use std::{ffi::c_void, time::Instant};

const GUARD: usize = 128;
const SENTINEL: u8 = 0x6d;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Arm {
    Dense,
    Staged,
    Fused,
    SerialFused,
    Native,
}

pub struct Guarded {
    allocation: DeviceBuffer,
    bytes: usize,
}

impl Guarded {
    pub fn new(ctx: &CudaContext, bytes: usize) -> Result<Self> {
        let result = Self {
            allocation: DeviceBuffer::new_tagged(ctx, bytes + GUARD * 2, "lossless_probe:guarded")?,
            bytes,
        };
        result.reset(ctx)?;
        Ok(result)
    }
    pub fn get_pointer(&self) -> *mut c_void {
        unsafe { self.allocation.as_mut_ptr().cast::<u8>().add(GUARD).cast() }
    }
    pub fn reset(&self, ctx: &CudaContext) -> Result<()> {
        let data = vec![SENTINEL; self.bytes + GUARD * 2];
        unsafe {
            ctx.memcpy_h2d(
                self.allocation.as_mut_ptr(),
                data.as_ptr().cast(),
                data.len(),
            )
        }
    }
    pub fn read(&self, ctx: &CudaContext) -> Result<Vec<u8>> {
        let mut data = vec![0; self.bytes + GUARD * 2];
        unsafe {
            ctx.memcpy_d2h(
                data.as_mut_ptr().cast(),
                self.allocation.as_ptr(),
                data.len(),
            )?;
        }
        ensure!(
            data[..GUARD]
                .iter()
                .chain(&data[GUARD + self.bytes..])
                .all(|v| *v == SENTINEL),
            "guard changed"
        );
        Ok(data[GUARD..GUARD + self.bytes].to_vec())
    }
    fn upload(&self, ctx: &CudaContext, data: &[u8]) -> Result<()> {
        ensure!(data.len() == self.bytes, "guarded upload shape differs");
        unsafe { ctx.memcpy_h2d(self.get_pointer(), data.as_ptr().cast(), data.len()) }
    }
}

pub struct Case {
    pub ctx: CudaContext,
    pub fixture: Fixture,
    pub keys: LosslessKvDevice,
    pub values: LosslessKvDevice,
    dense_keys: Guarded,
    dense_values: Guarded,
    staged_keys: Guarded,
    staged_values: Guarded,
    query: Guarded,
    output: Guarded,
    partial: Guarded,
    pub native: KVCache,
    native_workspace: Head128AttentionWorkspace,
    pub last_keys: DeviceBuffer,
    pub last_values: DeviceBuffer,
    pub start: CudaEvent,
    pub stop: CudaEvent,
    pub setup: Value,
}

impl Case {
    pub fn new(ctx: &CudaContext, fixture: Fixture) -> Result<Self> {
        let before = CudaContext::total_device_bytes();
        let began = Instant::now();
        let encoded = fixture.encode()?;
        let encode_ms = began.elapsed().as_secs_f64() * 1000.0;
        let representation = encoded.describe();
        let began = Instant::now();
        let keys = LosslessKvDevice::upload(ctx, &encoded.keys)?;
        let values = LosslessKvDevice::upload(ctx, &encoded.values)?;
        let encoded_upload_ms = began.elapsed().as_secs_f64() * 1000.0;
        let _dense_config = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
        let shape = Self::shape_for(&fixture, fixture.sequences)?;
        let began = Instant::now();
        let mut result = Self {
            ctx: ctx.clone(),
            keys,
            values,
            dense_keys: Guarded::new(ctx, shape.get_kv_bytes())?,
            dense_values: Guarded::new(ctx, shape.get_kv_bytes())?,
            staged_keys: Guarded::new(ctx, shape.get_kv_bytes())?,
            staged_values: Guarded::new(ctx, shape.get_kv_bytes())?,
            query: Guarded::new(ctx, shape.get_query_bytes())?,
            output: Guarded::new(ctx, shape.get_query_bytes())?,
            partial: Guarded::new(ctx, shape.get_partial_bytes())?,
            native: KVCache::new_with_context(
                ctx,
                shape.length,
                shape.batch,
                shape.kv_heads,
                shape.dimension,
            )?,
            native_workspace: Head128AttentionWorkspace::new(ctx, shape.query_heads)?,
            last_keys: DeviceBuffer::new_tagged(
                ctx,
                fixture.sequences * KV_HEADS * DIMENSION * 4,
                "lossless_probe:last_k",
            )?,
            last_values: DeviceBuffer::new_tagged(
                ctx,
                fixture.sequences * KV_HEADS * DIMENSION * 4,
                "lossless_probe:last_v",
            )?,
            start: ctx.create_event()?,
            stop: ctx.create_event()?,
            fixture,
            setup: Value::Null,
        };
        let allocation_ms = began.elapsed().as_secs_f64() * 1000.0;
        let began = Instant::now();
        result
            .dense_keys
            .upload(ctx, &encode_words(&result.fixture.keys))?;
        result
            .dense_values
            .upload(ctx, &encode_words(&result.fixture.values))?;
        let dense_upload_ms = began.elapsed().as_secs_f64() * 1000.0;
        let began = Instant::now();
        let query_bytes: Vec<_> = result
            .fixture
            .queries
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let source =
            DeviceBuffer::new_tagged(ctx, query_bytes.len(), "lossless_probe:query_source")?;
        let uploaded = (|| -> Result<()> {
            unsafe {
                ctx.memcpy_h2d(
                    source.as_mut_ptr(),
                    query_bytes.as_ptr().cast(),
                    query_bytes.len(),
                )?;
                ctx.memcpy_d2d_async(
                    result.query.get_pointer(),
                    source.as_ptr(),
                    query_bytes.len(),
                    CudaStream::Prefill,
                )?;
            }
            ctx.stream_wait_for_stream(
                CudaStream::Decode,
                CudaStream::Prefill,
                "lossless_probe_query_ready",
            )?;
            Ok(())
        })();
        ctx.synchronize_stream(CudaStream::Prefill)?;
        ctx.synchronize_stream(CudaStream::Decode)?;
        uploaded?;
        drop(source);
        let query_upload_ms = began.elapsed().as_secs_f64() * 1000.0;
        let native_setup = result.populate_native()?;
        for (source, target) in [
            (result.fixture.get_last_tokens(true), &result.last_keys),
            (result.fixture.get_last_tokens(false), &result.last_values),
        ] {
            unsafe {
                ctx.memcpy_h2d(
                    target.as_mut_ptr(),
                    source.as_ptr().cast(),
                    source.len() * 4,
                )?;
            }
        }
        result.setup = json!({"encode_validate_and_roundtrip_cpu_ms":encode_ms,
            "encoded_allocation_upload_wall_ms":encoded_upload_ms,"other_allocation_and_guards_ms":allocation_ms,
            "dense_upload_wall_ms":dense_upload_ms,"query_upload_and_dependency_ms":query_upload_ms,
            "native_cache_setup":native_setup,"representation":representation,
            "encoded_device_bytes":result.keys.get_bytes()+result.values.get_bytes(),
            "dense_prototype_bytes":2*shape.get_kv_bytes(),"staging_bytes":2*shape.get_kv_bytes(),
            "partial_bytes":shape.get_partial_bytes(),"native_cache_payload_bytes":2*shape.get_kv_bytes(),
            "native_workspace_bytes":result.native_workspace.bytes(),"query_output_bytes":2*shape.get_query_bytes(),
            "last_token_device_bytes":2*result.last_keys.bytes(),"tracked_device_bytes":CudaContext::total_device_bytes()-before,
            "tracked_scope":"includes guards and auxiliary buffers; native cache storage is separately estimated",
            "retained_host_original_fp16_bytes":2*shape.get_kv_bytes()});
        Ok(result)
    }

    fn shape_for(fixture: &Fixture, batch: usize) -> Result<LosslessKvAttentionShape> {
        ensure!(
            [1, 2, 4, 8].contains(&batch) && batch <= fixture.sequences,
            "invalid active batch"
        );
        let shape = LosslessKvAttentionShape {
            length: fixture.length as u32,
            query_heads: QUERY_HEADS as u32,
            kv_heads: KV_HEADS as u32,
            batch: batch as u32,
            dimension: DIMENSION as u32,
            partitions: fixture.length.min(8) as u32,
        };
        shape.validate()?;
        Ok(shape)
    }
    pub fn get_shape(&self, batch: usize) -> Result<LosslessKvAttentionShape> {
        Self::shape_for(&self.fixture, batch)
    }

    fn populate_native(&self) -> Result<Value> {
        let ctx = &self.ctx;
        let width = KV_HEADS * DIMENSION;
        let mut upload_ms = 0.0;
        let mut append_ms = 0.0;
        let mut append_gpu_ms = 0.0;
        for sequence in 0..self.fixture.sequences {
            let interleave = |streams: &[Vec<u16>]| -> Vec<f32> {
                (0..self.fixture.length)
                    .flat_map(|token| {
                        (0..width).map(move |i| {
                            f16::from_bits(
                                streams[sequence * KV_HEADS + i / DIMENSION]
                                    [token * DIMENSION + i % DIMENSION],
                            )
                            .to_f32()
                        })
                    })
                    .collect()
            };
            let k = interleave(&self.fixture.keys);
            let v = interleave(&self.fixture.values);
            let dk = DeviceBuffer::new_tagged(ctx, k.len() * 4, "lossless_probe:native_k_source")?;
            let dv = DeviceBuffer::new_tagged(ctx, v.len() * 4, "lossless_probe:native_v_source")?;
            let began = Instant::now();
            unsafe {
                ctx.memcpy_h2d(dk.as_mut_ptr(), k.as_ptr().cast(), k.len() * 4)?;
                ctx.memcpy_h2d(dv.as_mut_ptr(), v.as_ptr().cast(), v.len() * 4)?;
            }
            upload_ms += began.elapsed().as_secs_f64() * 1000.0;
            let began = Instant::now();
            self.start.record(CudaStream::Decode)?;
            let launched = (|| -> Result<()> {
                for token in 0..self.fixture.length {
                    unsafe {
                        self.native.append_token_f32_rope_k_at_async(
                            ctx,
                            sequence as u32,
                            dk.as_ptr().cast::<f32>().add(token * width).cast(),
                            dv.as_ptr().cast::<f32>().add(token * width).cast(),
                            token as u32,
                            token as u32,
                            10000.0,
                            0.0,
                        )?;
                    }
                }
                self.stop.record(CudaStream::Decode)?;
                Ok(())
            })();
            ctx.synchronize_stream(CudaStream::Decode)?;
            launched?;
            append_gpu_ms += f64::from(
                self.start
                    .elapsed_sync(&self.stop, "lossless_native_setup")?,
            );
            append_ms += began.elapsed().as_secs_f64() * 1000.0;
            ensure!(
                self.native.debug_sequence_length(ctx, sequence as u32)?
                    == self.fixture.length as u32,
                "native setup length differs"
            );
        }
        Ok(
            json!({"upload_wall_ms":upload_ms,"append_wall_ms":append_ms,"append_gpu_ms":append_gpu_ms,
            "peak_temporary_device_bytes":self.fixture.length*width*8,"rope_frequency":0}),
        )
    }

    pub fn enqueue(&self, arm: Arm, batch: usize) -> Result<()> {
        let shape = self.get_shape(batch)?;
        unsafe {
            match arm {
                Arm::Native => {
                    for sequence in 0..batch {
                        self.native.attention_head128_parallel_async(
                            &self.ctx,
                            sequence as u32,
                            self.query
                                .get_pointer()
                                .cast::<f32>()
                                .add(sequence * QUERY_HEADS * DIMENSION)
                                .cast(),
                            QUERY_HEADS as u32,
                            shape.length,
                            self.output
                                .get_pointer()
                                .cast::<f32>()
                                .add(sequence * QUERY_HEADS * DIMENSION)
                                .cast(),
                            Head128AttentionConfig {
                                partitions: 8,
                                threads: 256,
                            },
                            Some(&self.native_workspace),
                        )?;
                    }
                }
                Arm::Staged => {
                    self.keys
                        .decode_prefix_async(self.staged_keys.get_pointer(), batch * KV_HEADS)?;
                    self.values
                        .decode_prefix_async(self.staged_values.get_pointer(), batch * KV_HEADS)?;
                    self.ctx.lossless_kv_attention_async(
                        self.staged_keys.get_pointer(),
                        self.staged_values.get_pointer(),
                        self.query.get_pointer(),
                        self.partial.get_pointer(),
                        self.output.get_pointer(),
                        shape,
                        false,
                    )?;
                }
                Arm::Dense => self.ctx.lossless_kv_attention_async(
                    self.dense_keys.get_pointer(),
                    self.dense_values.get_pointer(),
                    self.query.get_pointer(),
                    self.partial.get_pointer(),
                    self.output.get_pointer(),
                    shape,
                    false,
                )?,
                Arm::Fused => self.enqueue_encoded(&self.keys, &self.values, batch)?,
                Arm::SerialFused => {
                    for sequence in 0..batch {
                        self.ctx.lossless_kv_attention_async(
                            self.keys
                                .get_directory()
                                .cast::<*const c_void>()
                                .add(sequence * KV_HEADS)
                                .cast(),
                            self.values
                                .get_directory()
                                .cast::<*const c_void>()
                                .add(sequence * KV_HEADS)
                                .cast(),
                            self.query
                                .get_pointer()
                                .cast::<f32>()
                                .add(sequence * QUERY_HEADS * DIMENSION)
                                .cast(),
                            self.partial.get_pointer(),
                            self.output
                                .get_pointer()
                                .cast::<f32>()
                                .add(sequence * QUERY_HEADS * DIMENSION)
                                .cast(),
                            LosslessKvAttentionShape { batch: 1, ..shape },
                            true,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn enqueue_encoded(
        &self,
        keys: &LosslessKvDevice,
        values: &LosslessKvDevice,
        batch: usize,
    ) -> Result<()> {
        ensure!(
            keys.get_stream_count() >= batch * KV_HEADS
                && values.get_stream_count() >= batch * KV_HEADS
                && keys.get_values_per_stream() == self.fixture.length * DIMENSION
                && values.get_values_per_stream() == keys.get_values_per_stream(),
            "replacement encoded shape differs"
        );
        unsafe {
            self.ctx.lossless_kv_attention_async(
                keys.get_directory(),
                values.get_directory(),
                self.query.get_pointer(),
                self.partial.get_pointer(),
                self.output.get_pointer(),
                self.get_shape(batch)?,
                true,
            )
        }
    }

    pub fn reset_output(&self) -> Result<()> {
        self.output.reset(&self.ctx)
    }
    pub fn read_output(&self, batch: usize) -> Result<Vec<u8>> {
        self.ctx.synchronize_stream(CudaStream::Decode)?;
        let count = self.get_shape(batch)?.get_query_bytes();
        let bytes = self.output.read(&self.ctx)?;
        ensure!(
            bytes[count..].iter().all(|v| *v == SENTINEL),
            "inactive output rows changed"
        );
        Ok(bytes[..count].to_vec())
    }
    pub fn read_arm(&self, arm: Arm, batch: usize) -> Result<Vec<u8>> {
        self.reset_output()?;
        self.enqueue(arm, batch)?;
        self.read_output(batch)
    }
    pub fn verify_storage(&self) -> Result<()> {
        self.ctx.synchronize_stream(CudaStream::Decode)?;
        ensure!(
            self.query.read(&self.ctx)?
                == self
                    .fixture
                    .queries
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            "query changed"
        );
        ensure!(
            self.dense_keys.read(&self.ctx)? == encode_words(&self.fixture.keys),
            "original keys changed"
        );
        ensure!(
            self.dense_values.read(&self.ctx)? == encode_words(&self.fixture.values),
            "original values changed"
        );
        self.partial.read(&self.ctx)?;
        Ok(())
    }
    pub fn verify_encoded(
        &self,
        keys: &LosslessKvDevice,
        values: &LosslessKvDevice,
        batch: usize,
    ) -> Result<()> {
        unsafe {
            keys.decode_prefix_async(self.staged_keys.get_pointer(), batch * KV_HEADS)?;
            values.decode_prefix_async(self.staged_values.get_pointer(), batch * KV_HEADS)?;
        }
        self.ctx.synchronize_stream(CudaStream::Decode)?;
        let bytes = self.get_shape(batch)?.get_kv_bytes();
        ensure!(
            self.staged_keys.read(&self.ctx)?[..bytes]
                == encode_words(&self.fixture.keys[..batch * KV_HEADS]),
            "replacement key bits differ"
        );
        ensure!(
            self.staged_values.read(&self.ctx)?[..bytes]
                == encode_words(&self.fixture.values[..batch * KV_HEADS]),
            "replacement value bits differ"
        );
        Ok(())
    }
    pub fn validate(&self) -> Result<Value> {
        let reference = self.fixture.calculate_reference();
        let mut errors = Vec::new();
        for batch in [1, 2, 4, 8]
            .into_iter()
            .filter(|v| *v <= self.fixture.sequences)
        {
            let staged = self.read_arm(Arm::Staged, batch)?;
            let kv_bytes = self.get_shape(batch)?.get_kv_bytes();
            ensure!(
                self.staged_keys.read(&self.ctx)?[..kv_bytes]
                    == encode_words(&self.fixture.keys[..batch * KV_HEADS]),
                "staged K bits differ"
            );
            ensure!(
                self.staged_values.read(&self.ctx)?[..kv_bytes]
                    == encode_words(&self.fixture.values[..batch * KV_HEADS]),
                "staged V bits differ"
            );
            for arm in [
                Arm::Staged,
                Arm::Dense,
                Arm::Fused,
                Arm::SerialFused,
                Arm::Native,
            ] {
                let actual = self.read_arm(arm, batch)?;
                let mut maximum = 0.0f64;
                for (word, expected) in actual.chunks_exact(4).zip(&reference) {
                    let value = f32::from_le_bytes(word.try_into().unwrap());
                    let error = (f64::from(value) - expected).abs();
                    ensure!(
                        value.is_finite() && error <= 1e-3,
                        "FP64 output differs arm={arm:?} batch={batch} error={error}"
                    );
                    maximum = maximum.max(error);
                }
                if !matches!(arm, Arm::Native) {
                    if let Some((index, (word, expected))) = actual
                        .chunks_exact(4)
                        .zip(staged.chunks_exact(4))
                        .enumerate()
                        .find(|(_, (word, expected))| word != expected)
                    {
                        let actual_bits = u32::from_le_bytes(word.try_into().unwrap());
                        let expected_bits = u32::from_le_bytes(expected.try_into().unwrap());
                        anyhow::bail!(
                            "matched prototype output bits differ arm={arm:?} batch={batch} word={index} actual=0x{actual_bits:08x} staged=0x{expected_bits:08x} max_fp64_error={maximum}"
                        );
                    }
                }
                ensure!(
                    self.read_arm(arm, batch)? == actual,
                    "repeated output differs"
                );
                errors.push(json!({"batch":batch,"arm":arm,"max_absolute_fp64_error":maximum}));
            }
        }
        self.verify_storage()?;
        Ok(
            json!({"has_bit_exact_reconstruction":true,"has_matched_prototype_output_bits":true,"fp64_tolerance":1e-3,"errors":errors,
            "native_rope_zero_sign_scope":"native append may change signed zero; lossless decoded words retain source bits"}),
        )
    }
}

impl Drop for Case {
    fn drop(&mut self) {
        let _ = self.ctx.synchronize_stream(CudaStream::Prefill);
        let _ = self.ctx.synchronize_stream(CudaStream::Decode);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::Capture;
    use std::path::PathBuf;

    /// Replay the first failing real capture case against every consumer and
    /// active batch. Missing input artifacts are errors, never skipped passes.
    #[test]
    #[ignore = "requires M40 and LOSSLESS_CAPTURE_REPORT/LOSSLESS_KV_INPUTS; run serially"]
    fn test_real_capture_preserves_matched_output_bits() -> Result<()> {
        let report = PathBuf::from(std::env::var("LOSSLESS_CAPTURE_REPORT")?);
        let artifact = PathBuf::from(std::env::var("LOSSLESS_KV_INPUTS")?);
        let mut capture = Capture::open(&report, &artifact)?;
        crate::benchmark::check_gpu_processes()?;
        let ctx = CudaContext::new_m40()?;
        let baseline = CudaContext::total_device_bytes();
        let sample = capture.read_layer(0, 1024, DIMENSION)?;
        let case = Case::new(&ctx, Fixture::from_sample(&sample, 1024)?)?;
        let result = case.validate()?;
        println!("{result}");
        drop(case);
        ensure!(
            CudaContext::total_device_bytes() == baseline,
            "device storage leaked"
        );
        Ok(())
    }
}
