//! Opt-in research consumers for validated lossless FP16 streams.

use super::{ffi, record_async_kernel, CudaContext, CudaStream, DeviceBuffer};
use crate::lossless_kv::LosslessKvBlocks;
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;

extern "C" {
    fn m40llm_lossless_kv_decode_async(
        ctx: *mut ffi::M40llmCudaContext,
        streams: *const c_void,
        values: u32,
        stream_count: u32,
        output: *mut c_void,
    ) -> i32;
    fn m40llm_lossless_kv_attention_async(
        ctx: *mut ffi::M40llmCudaContext,
        keys: *const c_void,
        values: *const c_void,
        query: *const c_void,
        partial: *mut c_void,
        output: *mut c_void,
        length: u32,
        query_heads: u32,
        kv_heads: u32,
        batch: u32,
        dimension: u32,
        partitions: u32,
        is_packed: u32,
    ) -> i32;
}

/// Equal-length streams, ordered sequence then KV head, with owned device
/// payloads and pointer directory. Upload completes before returning. Drop
/// drains decode before releasing any storage used by queued consumers.
pub struct LosslessKvDevice {
    ctx: CudaContext,
    streams: Vec<DeviceBuffer>,
    directory: DeviceBuffer,
    values: usize,
}

impl LosslessKvDevice {
    pub fn upload(ctx: &CudaContext, encoded: &[LosslessKvBlocks]) -> Result<Self> {
        ensure!(
            (1..=64).contains(&encoded.len()),
            "invalid lossless stream count"
        );
        let values = encoded[0].get_stats().values;
        ensure!(
            values <= (u32::MAX - 255) as usize
                && encoded.iter().all(|v| v.get_stats().values == values),
            "lossless streams differ in length or exceed the decoder limit"
        );
        let mut streams = Vec::with_capacity(encoded.len());
        for stream in encoded {
            let bytes = stream.get_bytes();
            let buffer = DeviceBuffer::new_tagged(ctx, bytes.len(), "lossless:encoded")?;
            unsafe {
                ctx.memcpy_h2d(buffer.as_mut_ptr(), bytes.as_ptr().cast(), bytes.len())?;
            }
            streams.push(buffer);
        }
        let pointers: Vec<_> = streams.iter().map(DeviceBuffer::as_ptr).collect();
        let directory = DeviceBuffer::new_tagged(
            ctx,
            std::mem::size_of_val(pointers.as_slice()),
            "lossless:directory",
        )?;
        unsafe {
            ctx.memcpy_h2d(
                directory.as_mut_ptr(),
                pointers.as_ptr().cast(),
                directory.bytes(),
            )?;
        }
        Ok(Self {
            ctx: ctx.clone(),
            streams,
            directory,
            values,
        })
    }

    pub fn get_values_per_stream(&self) -> usize {
        self.values
    }
    pub fn get_stream_count(&self) -> usize {
        self.streams.len()
    }
    pub fn get_bytes(&self) -> usize {
        self.directory.bytes() + self.streams.iter().map(DeviceBuffer::bytes).sum::<usize>()
    }
    pub fn get_directory(&self) -> *const c_void {
        self.directory.as_ptr()
    }

    /// Reconstruct every original FP16 word on the decode stream.
    ///
    /// # Safety
    /// Output is disjoint live device storage in this context, with at least
    /// stream_count * values_per_stream u16 words. Keep it alive through decode
    /// completion and order other producers/consumers explicitly.
    pub unsafe fn decode_async(&self, output: *mut c_void) -> Result<()> {
        self.decode_prefix_async(output, self.streams.len())
    }

    /// Reconstruct only the first `stream_count` streams for an active batch.
    ///
    /// # Safety
    /// The same ownership and ordering contract as `decode_async` applies,
    /// with output capacity stream_count * values_per_stream u16 words.
    pub unsafe fn decode_prefix_async(
        &self,
        output: *mut c_void,
        stream_count: usize,
    ) -> Result<()> {
        ensure!(
            (1..=self.streams.len()).contains(&stream_count),
            "invalid decode stream prefix"
        );
        validate_span(output, stream_count * self.values * 2, 2)?;
        let _guard = self.ctx.inner.lock.lock().unwrap();
        let rc = m40llm_lossless_kv_decode_async(
            self.ctx.inner.raw.as_ptr(),
            self.directory.as_ptr(),
            self.values as u32,
            stream_count as u32,
            output,
        );
        ensure!(rc == 0, "lossless KV reconstruction failed: {rc}");
        record_async_kernel("lossless_kv_decode");
        Ok(())
    }
}

impl Drop for LosslessKvDevice {
    fn drop(&mut self) {
        let _ = self.ctx.synchronize_stream(CudaStream::Decode);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LosslessKvAttentionShape {
    pub length: u32,
    pub query_heads: u32,
    pub kv_heads: u32,
    pub batch: u32,
    pub dimension: u32,
    pub partitions: u32,
}

impl LosslessKvAttentionShape {
    pub fn validate(self) -> Result<()> {
        ensure!(
            (1..=8192).contains(&self.length)
                && (1..=64).contains(&self.query_heads)
                && self.kv_heads > 0
                && self.query_heads.is_multiple_of(self.kv_heads)
                && (1..=8).contains(&self.batch)
                && self.batch * self.kv_heads <= 64
                && matches!(self.dimension, 64 | 128)
                && (1..=32).contains(&self.partitions)
                && self.partitions <= self.length,
            "invalid lossless attention shape"
        );
        Ok(())
    }
    pub fn get_query_bytes(self) -> usize {
        self.batch as usize * self.query_heads as usize * self.dimension as usize * 4
    }
    pub fn get_kv_bytes(self) -> usize {
        self.batch as usize
            * self.kv_heads as usize
            * self.length as usize
            * self.dimension as usize
            * 2
    }
    pub fn get_partial_bytes(self) -> usize {
        self.batch as usize
            * self.query_heads as usize
            * self.partitions as usize
            * (self.dimension as usize + 2)
            * 4
    }
}

fn validate_span(pointer: *const c_void, bytes: usize, alignment: usize) -> Result<(usize, usize)> {
    let start = pointer as usize;
    ensure!(
        start != 0 && start.is_multiple_of(alignment),
        "null or unaligned lossless span"
    );
    let end = start.checked_add(bytes).context("lossless span overflow")?;
    Ok((start, end))
}

impl CudaContext {
    /// Full-context GQA with matched dense/packed FP32 dot and softmax order.
    /// Dense K/V use [sequence, KV head, token, channel]. Packed K/V are device
    /// pointer directories in [sequence, KV head] order, each stream containing
    /// length * dimension validated codec words. Both variants launch partition
    /// and merge kernels on decode, with no allocation or host synchronization.
    ///
    /// # Safety
    /// All buffers belong to this context, contain the declared shapes, survive
    /// decode completion, and have ordered producers. Packed payloads must be
    /// CPU-validated LosslessKvBlocks with enough values. Query and output are
    /// [batch, query head, dimension] FP32. Partial storage has
    /// shape.get_partial_bytes() bytes. Neither output nor partial may overlap
    /// any input, including payloads behind packed pointer directories. Inputs
    /// must be finite for the numerical attention contract.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn lossless_kv_attention_async(
        &self,
        keys: *const c_void,
        values: *const c_void,
        query: *const c_void,
        partial: *mut c_void,
        output: *mut c_void,
        shape: LosslessKvAttentionShape,
        is_packed: bool,
    ) -> Result<()> {
        shape.validate()?;
        let kv_bytes = if is_packed {
            shape.batch as usize * shape.kv_heads as usize * std::mem::size_of::<*const c_void>()
        } else {
            shape.get_kv_bytes()
        };
        let alignment = if is_packed {
            std::mem::align_of::<*const c_void>()
        } else {
            2
        };
        let inputs = [
            validate_span(keys, kv_bytes, alignment)?,
            validate_span(values, kv_bytes, alignment)?,
            validate_span(query, shape.get_query_bytes(), 4)?,
        ];
        let scratch = validate_span(partial, shape.get_partial_bytes(), 4)?;
        let result = validate_span(output, shape.get_query_bytes(), 4)?;
        for (start, end) in inputs.into_iter().chain([scratch]) {
            ensure!(
                end <= result.0 || result.1 <= start,
                "lossless attention output overlaps storage"
            );
        }
        for (start, end) in inputs {
            ensure!(
                end <= scratch.0 || scratch.1 <= start,
                "lossless attention partial overlaps input"
            );
        }
        let _guard = self.inner.lock.lock().unwrap();
        let rc = m40llm_lossless_kv_attention_async(
            self.inner.raw.as_ptr(),
            keys,
            values,
            query,
            partial,
            output,
            shape.length,
            shape.query_heads,
            shape.kv_heads,
            shape.batch,
            shape.dimension,
            shape.partitions,
            u32::from(is_packed),
        );
        ensure!(rc == 0, "lossless KV attention failed: {rc}");
        record_async_kernel("lossless_kv_attention_partition");
        record_async_kernel("lossless_kv_attention_merge");
        Ok(())
    }
}
