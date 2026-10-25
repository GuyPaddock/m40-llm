#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;

use anyhow::Result;
use m40_llm::cuda::CudaStream;
use m40_llm::infer::{
    BatchMetadata, BatchSequence, VarlenPrefillPlan, VarlenPrefillTile, VarlenPrefillTileSelection,
};
use std::ffi::c_void;

fn f32s_to_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
        .collect()
}

fn cpu_prefill_gqa_varlen(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    meta: &BatchMetadata,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; meta.total_q_tokens() as usize * q_heads * head_dim];
    let group = q_heads / kv_heads;
    let inv_sqrt = 1.0f32 / (head_dim as f32).sqrt();

    for (seq_idx, seq) in meta.sequences().iter().enumerate() {
        let offsets = meta.offsets()[seq_idx];
        for q_idx in 0..seq.query_len as usize {
            let causal_end = seq.kv_len as usize - seq.query_len as usize + q_idx;
            for qh in 0..q_heads {
                let kvh = qh / group;
                let q_base = ((offsets.q_offset as usize + q_idx) * q_heads + qh) * head_dim;
                let mut scores = Vec::with_capacity(causal_end + 1);
                let mut max_score = f32::NEG_INFINITY;
                for t in 0..=causal_end {
                    let k_base = ((offsets.kv_offset as usize + t) * kv_heads + kvh) * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_base + d] * k[k_base + d];
                    }
                    let score = dot * inv_sqrt;
                    max_score = max_score.max(score);
                    scores.push(score);
                }
                let denom: f32 = scores.iter().map(|score| (*score - max_score).exp()).sum();
                let denom = if denom > 0.0 { denom } else { 1.0 };
                let out_base = ((offsets.q_offset as usize + q_idx) * q_heads + qh) * head_dim;
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for (t, score) in scores.iter().enumerate() {
                        let v_base = ((offsets.kv_offset as usize + t) * kv_heads + kvh) * head_dim;
                        acc += ((*score - max_score).exp() / denom) * v[v_base + d];
                    }
                    out[out_base + d] = acc;
                }
            }
        }
    }

    out
}

#[test]
fn packed_varlen_prefill_gqa_matches_cpu_reference() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    for head_dim in [64usize, 128usize] {
        let q_heads = 4usize;
        let kv_heads = 2usize;
        let meta = BatchMetadata::new(vec![
            BatchSequence {
                seq_len: 3,
                query_len: 3,
                kv_len: 3,
            },
            BatchSequence {
                seq_len: 5,
                query_len: 2,
                kv_len: 5,
            },
            BatchSequence {
                seq_len: 4,
                query_len: 1,
                kv_len: 4,
            },
        ])?;

        let q_len = meta.total_q_tokens() as usize * q_heads * head_dim;
        let kv_len = meta.total_kv_tokens() as usize * kv_heads * head_dim;
        let q: Vec<f32> = (0..q_len)
            .map(|i| ((i * 17 % 251) as f32) * 0.0011 - 0.13)
            .collect();
        let k: Vec<f32> = (0..kv_len)
            .map(|i| ((i * 23 % 263) as f32) * 0.0009 - 0.09)
            .collect();
        let v: Vec<f32> = (0..kv_len)
            .map(|i| ((i * 29 % 269) as f32) * 0.0013 - 0.17)
            .collect();
        let expected = cpu_prefill_gqa_varlen(&q, &k, &v, &meta, q_heads, kv_heads, head_dim);

        let bytes_q = q.len() * std::mem::size_of::<f32>();
        let bytes_kv = k.len() * std::mem::size_of::<f32>();
        let bytes_out = expected.len() * std::mem::size_of::<f32>();
        let plan = VarlenPrefillPlan::new(&ctx, meta.clone(), head_dim as u32)?;
        assert_eq!(plan.batch_size(), meta.sequences().len() as u32);
        assert_eq!(
            plan.tile_selection(),
            VarlenPrefillTileSelection {
                head_dim: head_dim as u32,
                max_query_len: 3,
                max_kv_len: 5,
                tile: if head_dim == 128 {
                    VarlenPrefillTile::CONSERVATIVE_HEAD128
                } else {
                    VarlenPrefillTile::CONSERVATIVE_HEAD64
                },
            }
        );

        let d_q = ctx.device_malloc(bytes_q)?;
        let d_k = ctx.device_malloc(bytes_kv)?;
        let d_v = ctx.device_malloc(bytes_kv)?;
        let d_out = ctx.device_malloc(bytes_out)?;
        let d_out_async = ctx.device_malloc(bytes_out)?;

        unsafe {
            ctx.memcpy_h2d(d_q, f32s_to_bytes(&q).as_ptr() as *const c_void, bytes_q)?;
            ctx.memcpy_h2d(d_k, f32s_to_bytes(&k).as_ptr() as *const c_void, bytes_kv)?;
            ctx.memcpy_h2d(d_v, f32s_to_bytes(&v).as_ptr() as *const c_void, bytes_kv)?;
            plan.dispatch(
                d_q as *const c_void,
                d_k as *const c_void,
                d_v as *const c_void,
                q_heads as u32,
                kv_heads as u32,
                d_out,
            )?;
            plan.dispatch_async(
                d_q as *const c_void,
                d_k as *const c_void,
                d_v as *const c_void,
                q_heads as u32,
                kv_heads as u32,
                d_out_async,
            )?;
            ctx.synchronize_stream(CudaStream::Prefill)?;

            let mut got_bytes = vec![0u8; bytes_out];
            ctx.memcpy_d2h(got_bytes.as_mut_ptr() as *mut c_void, d_out, bytes_out)?;
            let got = bytes_to_f32s(&got_bytes);
            let mut async_bytes = vec![0u8; bytes_out];
            ctx.memcpy_d2h(
                async_bytes.as_mut_ptr() as *mut c_void,
                d_out_async,
                bytes_out,
            )?;
            let async_got = bytes_to_f32s(&async_bytes);
            for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                let diff = (g - e).abs();
                assert!(
                    diff <= 3e-4,
                    "head_dim={head_dim} mismatch at {i}: got {g}, expected {e}"
                );
            }
            for (i, (g, e)) in async_got.iter().zip(got.iter()).enumerate() {
                let diff = (g - e).abs();
                assert!(
                    diff <= 1e-6,
                    "head_dim={head_dim} async mismatch at {i}: got {g}, expected {e}"
                );
            }

            ctx.device_free(d_q)?;
            ctx.device_free(d_k)?;
            ctx.device_free(d_v)?;
            ctx.device_free(d_out)?;
            ctx.device_free(d_out_async)?;
        }
    }

    Ok(())
}

#[test]
fn packed_varlen_prefill_qwen_shape_matches_cpu_reference() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let head_dim = 128usize;
    let q_heads = 16usize;
    let kv_heads = 2usize;
    let meta = BatchMetadata::new(vec![
        BatchSequence {
            seq_len: 19,
            query_len: 19,
            kv_len: 19,
        },
        BatchSequence {
            seq_len: 8,
            query_len: 8,
            kv_len: 8,
        },
        BatchSequence {
            seq_len: 17,
            query_len: 17,
            kv_len: 17,
        },
        BatchSequence {
            seq_len: 16,
            query_len: 16,
            kv_len: 16,
        },
    ])?;

    let q_len = meta.total_q_tokens() as usize * q_heads * head_dim;
    let kv_len = meta.total_kv_tokens() as usize * kv_heads * head_dim;
    let q: Vec<f32> = (0..q_len)
        .map(|i| ((i * 17 % 251) as f32) * 0.19 - 23.0)
        .collect();
    let k: Vec<f32> = (0..kv_len)
        .map(|i| ((i * 23 % 263) as f32) * 0.32 - 41.0)
        .collect();
    let v: Vec<f32> = (0..kv_len)
        .map(|i| ((i * 29 % 269) as f32) * 0.021 - 2.7)
        .collect();
    let expected = cpu_prefill_gqa_varlen(&q, &k, &v, &meta, q_heads, kv_heads, head_dim);

    let bytes_q = q.len() * std::mem::size_of::<f32>();
    let bytes_kv = k.len() * std::mem::size_of::<f32>();
    let bytes_out = expected.len() * std::mem::size_of::<f32>();
    let plan = VarlenPrefillPlan::new(&ctx, meta.clone(), head_dim as u32)?;
    let d_q = ctx.device_malloc(bytes_q)?;
    let d_k = ctx.device_malloc(bytes_kv)?;
    let d_v = ctx.device_malloc(bytes_kv)?;
    let d_out = ctx.device_malloc(bytes_out)?;

    unsafe {
        ctx.memcpy_h2d(d_q, f32s_to_bytes(&q).as_ptr() as *const c_void, bytes_q)?;
        ctx.memcpy_h2d(d_k, f32s_to_bytes(&k).as_ptr() as *const c_void, bytes_kv)?;
        ctx.memcpy_h2d(d_v, f32s_to_bytes(&v).as_ptr() as *const c_void, bytes_kv)?;
        plan.dispatch(
            d_q as *const c_void,
            d_k as *const c_void,
            d_v as *const c_void,
            q_heads as u32,
            kv_heads as u32,
            d_out,
        )?;

        let mut got_bytes = vec![0u8; bytes_out];
        ctx.memcpy_d2h(got_bytes.as_mut_ptr() as *mut c_void, d_out, bytes_out)?;
        let got = bytes_to_f32s(&got_bytes);
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                g.is_finite(),
                "Qwen-shape prefill produced nonfinite at {i}: {g}"
            );
            let diff = (g - e).abs();
            assert!(
                diff <= 2e-3,
                "Qwen-shape prefill mismatch at {i}: got {g}, expected {e}, diff={diff}"
            );
        }

        ctx.device_free(d_q)?;
        ctx.device_free(d_k)?;
        ctx.device_free(d_v)?;
        ctx.device_free(d_out)?;
    }

    Ok(())
}

#[test]
fn varlen_prefill_plan_rejects_unsupported_head_dim() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    let meta = BatchMetadata::new(vec![BatchSequence {
        seq_len: 4,
        query_len: 4,
        kv_len: 4,
    }])?;
    let err = VarlenPrefillPlan::new(&ctx, meta, 32).unwrap_err();
    assert!(err.to_string().contains("head_dim=64 or 128 only"));
    Ok(())
}
