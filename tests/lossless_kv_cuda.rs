#![cfg(all(feature = "cuda", nvcc))]

#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "common/guarded_bytes.rs"]
mod guarded_bytes;

use anyhow::{ensure, Result};
use guarded_bytes::GuardedBytes;
use half::f16;
use m40_llm::{
    cuda::{CudaContext, CudaStream, LosslessKvAttentionShape, LosslessKvDevice},
    lossless_kv::LosslessKvBlocks,
    profile,
};

fn encode_words(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn encode_floats(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn require_m40(ctx: &CudaContext) -> Result<()> {
    let props = ctx.current_device_props()?;
    ensure!(props.major == 5 && props.minor == 2, "test requires sm_52");
    Ok(())
}

/// Every FP16 pattern, all packed exponent widths, short tails and multiple
/// streams reconstruct bit-for-bit. Dropping the encoded owner after enqueue
/// must finish its consumer before payloads or pointer directories are freed.
#[test]
#[ignore = "requires M40 hardware; run serially with --include-ignored"]
fn test_lossless_decode_preserves_all_bits_and_lifetimes() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    require_m40(&ctx)?;
    let baseline = CudaContext::total_device_bytes();
    let mut fixtures = vec![(0..=u16::MAX).collect::<Vec<_>>()];
    for width in 0..=4 {
        for length in [1, 17, 511, 512, 513, 1025] {
            fixtures.push(
                (0..length)
                    .map(|i| {
                        let exponent = 8 + (i % (1 << width));
                        ((exponent as u16) << 10) | (i as u16 & 1023) | ((i as u16 % 2) << 15)
                    })
                    .collect(),
            );
        }
    }
    for words in fixtures {
        let inverted: Vec<_> = words.iter().map(|word| word ^ 0x8000).collect();
        let encoded = [
            LosslessKvBlocks::encode(&words)?,
            LosslessKvBlocks::encode(&inverted)?,
        ];
        let expected = encode_words(&[words, inverted].concat());
        let output = GuardedBytes::new(&ctx, &vec![0x6b; expected.len()], 2)?;
        let device = LosslessKvDevice::upload(&ctx, &encoded)?;
        ensure!(device.get_stream_count() == 2, "stream ownership differs");
        ensure!(
            device.get_values_per_stream() * 4 == expected.len(),
            "stream lengths differ"
        );
        unsafe {
            device.decode_prefix_async(output.pointer(), 1)?;
            assert!(device.decode_prefix_async(output.pointer(), 0).is_err());
            assert!(device.decode_prefix_async(output.pointer(), 3).is_err());
        }
        ctx.synchronize_stream(CudaStream::Decode)?;
        let prefix = output.read(&ctx)?;
        assert_eq!(
            &prefix[..expected.len() / 2],
            &expected[..expected.len() / 2]
        );
        assert!(prefix[expected.len() / 2..]
            .iter()
            .all(|value| *value == 0x6b));
        let before = profile::snapshot();
        unsafe {
            device.decode_async(output.pointer())?;
        }
        let counters = profile::counter_delta(&before, &profile::snapshot());
        ensure!(
            counters.device_allocs == 0 && counters.h2d_copies == 0 && counters.d2h_copies == 0,
            "decode allocated or transferred host data"
        );
        drop(device);
        assert_eq!(output.read(&ctx)?, expected);
    }
    assert_eq!(CudaContext::total_device_bytes(), baseline);
    Ok(())
}

struct AttentionFixture {
    keys: Vec<Vec<u16>>,
    values: Vec<Vec<u16>>,
    query: Vec<f32>,
}

impl AttentionFixture {
    fn build(shape: LosslessKvAttentionShape, generation: usize) -> Self {
        let length = shape.length as usize;
        let dimension = shape.dimension as usize;
        let streams = (shape.batch * shape.kv_heads) as usize;
        let build_streams = |salt: usize| {
            (0..streams)
                .map(|stream| {
                    (0..length * dimension)
                        .map(|i| {
                            let sample = i * 193 + stream * 71 + salt * 23 + generation * 7;
                            if i % 701 == 0 {
                                return if sample.is_multiple_of(2) { 0 } else { 0x8000 };
                            }
                            let exponent = 10 + (sample / 17) % 5;
                            ((exponent as u16) << 10)
                                | (sample as u16 & 1023)
                                | (((sample / 13) as u16 % 2) << 15)
                        })
                        .collect()
                })
                .collect()
        };
        let query = (0..shape.get_query_bytes() / 4)
            .map(|i| ((i * 17 + generation * 3) % 31) as f32 / 64.0 - 0.25)
            .collect();
        Self {
            keys: build_streams(1),
            values: build_streams(2),
            query,
        }
    }

    fn calculate_reference(&self, shape: LosslessKvAttentionShape) -> Vec<f64> {
        let dimension = shape.dimension as usize;
        let mut output = Vec::new();
        for sequence in 0..shape.batch as usize {
            for head in 0..shape.query_heads as usize {
                let stream = sequence * shape.kv_heads as usize
                    + head / (shape.query_heads / shape.kv_heads) as usize;
                let query_start = (sequence * shape.query_heads as usize + head) * dimension;
                let scores: Vec<f64> = self.keys[stream]
                    .chunks_exact(dimension)
                    .map(|row| {
                        row.iter()
                            .enumerate()
                            .map(|(channel, &word)| {
                                f64::from(f16::from_bits(word).to_f32())
                                    * f64::from(self.query[query_start + channel])
                            })
                            .sum::<f64>()
                            / (dimension as f64).sqrt()
                    })
                    .collect();
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<_> = scores.iter().map(|score| (score - maximum).exp()).collect();
                let total: f64 = weights.iter().sum();
                for channel in 0..dimension {
                    output.push(
                        weights
                            .iter()
                            .enumerate()
                            .map(|(token, weight)| {
                                weight
                                    * f64::from(
                                        f16::from_bits(
                                            self.values[stream][token * dimension + channel],
                                        )
                                        .to_f32(),
                                    )
                            })
                            .sum::<f64>()
                            / total,
                    );
                }
            }
        }
        output
    }
}

fn check_attention(
    ctx: &CudaContext,
    shape: LosslessKvAttentionShape,
    generation: usize,
) -> Result<()> {
    let fixture = AttentionFixture::build(shape, generation);
    let reference = fixture.calculate_reference(shape);
    let key_bytes = encode_words(&fixture.keys.concat());
    let value_bytes = encode_words(&fixture.values.concat());
    let keys = GuardedBytes::new(ctx, &key_bytes, 2)?;
    let values = GuardedBytes::new(ctx, &value_bytes, 2)?;
    let key_blocks: Vec<_> = fixture
        .keys
        .iter()
        .map(|words| LosslessKvBlocks::encode(words))
        .collect::<Result<_>>()?;
    let value_blocks: Vec<_> = fixture
        .values
        .iter()
        .map(|words| LosslessKvBlocks::encode(words))
        .collect::<Result<_>>()?;
    let encoded_keys = LosslessKvDevice::upload(ctx, &key_blocks)?;
    let encoded_values = LosslessKvDevice::upload(ctx, &value_blocks)?;
    let staged_keys = GuardedBytes::new(ctx, &vec![0x4d; key_bytes.len()], 2)?;
    let staged_values = GuardedBytes::new(ctx, &vec![0x4d; value_bytes.len()], 2)?;
    let scratch = GuardedBytes::new(ctx, &vec![0x57; shape.get_partial_bytes()], 4)?;
    let output = GuardedBytes::new(ctx, &vec![0x61; shape.get_query_bytes()], 4)?;
    let query_bytes = encode_floats(&fixture.query);
    let source_query = GuardedBytes::new(ctx, &query_bytes, 4)?;
    let query = GuardedBytes::new(ctx, &vec![0x7f; query_bytes.len()], 4)?;
    unsafe {
        ctx.memcpy_d2d_async(
            query.pointer(),
            source_query.pointer(),
            query_bytes.len(),
            CudaStream::Prefill,
        )?;
    }
    ctx.stream_wait_for_stream(
        CudaStream::Decode,
        CudaStream::Prefill,
        "lossless_test_query_ready",
    )?;
    let before = profile::snapshot();
    unsafe {
        encoded_keys.decode_async(staged_keys.pointer())?;
        encoded_values.decode_async(staged_values.pointer())?;
        ctx.lossless_kv_attention_async(
            staged_keys.pointer(),
            staged_values.pointer(),
            query.pointer(),
            scratch.pointer(),
            output.pointer(),
            shape,
            false,
        )?;
    }
    ctx.synchronize_stream(CudaStream::Decode)?;
    let counters = profile::counter_delta(&before, &profile::snapshot());
    ensure!(
        counters.device_allocs == 0
            && counters.device_frees == 0
            && counters.h2d_copies == 0
            && counters.d2h_copies == 0,
        "attention unexpectedly allocated or copied host data"
    );
    assert_eq!(staged_keys.read(ctx)?, key_bytes);
    assert_eq!(staged_values.read(ctx)?, value_bytes);
    let expected = output.read(ctx)?;
    for (index, bytes) in expected.chunks_exact(4).enumerate() {
        let actual = f32::from_le_bytes(bytes.try_into().unwrap());
        ensure!(
            actual.is_finite() && (f64::from(actual) - reference[index]).abs() <= 1e-3,
            "FP64 mismatch shape={shape:?} output={index} actual={actual} reference={}",
            reference[index]
        );
    }
    for _ in 0..3 {
        for is_packed in [false, true] {
            let k = if is_packed {
                encoded_keys.get_directory()
            } else {
                keys.pointer()
            };
            let v = if is_packed {
                encoded_values.get_directory()
            } else {
                values.pointer()
            };
            unsafe {
                ctx.lossless_kv_attention_async(
                    k,
                    v,
                    query.pointer(),
                    scratch.pointer(),
                    output.pointer(),
                    shape,
                    is_packed,
                )?;
            }
            ctx.synchronize_stream(CudaStream::Decode)?;
            assert_eq!(
                output.read(ctx)?,
                expected,
                "dense/staged/fused bits differ {shape:?}"
            );
            scratch.read(ctx)?;
        }
    }
    // Rejected shape and writable aliases must preserve the previous result.
    unsafe {
        assert!(ctx
            .lossless_kv_attention_async(
                keys.pointer(),
                values.pointer(),
                query.pointer(),
                scratch.pointer(),
                output.pointer(),
                LosslessKvAttentionShape {
                    partitions: 0,
                    ..shape
                },
                false
            )
            .is_err());
        assert!(ctx
            .lossless_kv_attention_async(
                keys.pointer(),
                values.pointer(),
                query.pointer(),
                scratch.pointer(),
                query.pointer(),
                shape,
                false
            )
            .is_err());
    }
    assert_eq!(output.read(ctx)?, expected);
    assert_eq!(query.read(ctx)?, query_bytes);
    assert_eq!(keys.read(ctx)?, key_bytes);
    assert_eq!(values.read(ctx)?, value_bytes);
    Ok(())
}

/// Dense, staged and fused consumers use all tokens and match both FP64 and
/// each other. Distinct sequence/head data, repeated allocation lifetimes,
/// irregular tails and an explicit prefill producer protect GQA and ownership.
#[test]
#[ignore = "requires M40 hardware; run serially with --include-ignored"]
fn test_lossless_attention_preserves_outputs_and_ownership() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    require_m40(&ctx)?;
    let baseline = CudaContext::total_device_bytes();
    for generation in 0..2 {
        for dimension in [64, 128] {
            for batch in [1, 2, 4, 8] {
                for length in [1, 5, 19, 257, 1024] {
                    let shape = LosslessKvAttentionShape {
                        length,
                        query_heads: 4,
                        kv_heads: 2,
                        batch,
                        dimension,
                        partitions: length.min(8),
                    };
                    check_attention(&ctx, shape, generation)?;
                    assert_eq!(CudaContext::total_device_bytes(), baseline);
                }
            }
        }
    }
    Ok(())
}
