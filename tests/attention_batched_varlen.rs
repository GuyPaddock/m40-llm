#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;

use anyhow::Result;
use m40_llm::cuda::{CudaStream, ExactOldBacking, KVCache};
use m40_llm::decode_batch::{CudaDecodeBatchPlan, DecodeBatchPlan, DecodeRequestState};
use m40_llm::kv_compression::KvRepresentativePolicy;
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

#[test]
fn batched_gqa_attention_matches_individual_varlen_decode() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    for head_dim in [64u32, 128u32] {
        let q_heads = 4u32;
        let kv_heads = 2u32;
        let max_seq_len = 8u32;
        let batch_size = 3u32;
        let seq_lens = [1u32, 3, 5];
        let seq_ids = [0u32, 1, 2];
        let q_dim = (q_heads * head_dim) as usize;
        let kv_dim = (kv_heads * head_dim) as usize;
        let kv = KVCache::new_with_context(&ctx, max_seq_len, batch_size, kv_heads, head_dim)?;

        let bytes_kv = kv_dim * std::mem::size_of::<f32>();
        let d_k = ctx.device_malloc(bytes_kv)?;
        let d_v = ctx.device_malloc(bytes_kv)?;
        for (seq_idx, &seq_len) in seq_lens.iter().enumerate() {
            for t in 0..seq_len as usize {
                let k: Vec<f32> = (0..kv_dim)
                    .map(|i| ((seq_idx * 97 + t * 13 + i) as f32) * 0.0007 - 0.35)
                    .collect();
                let v: Vec<f32> = (0..kv_dim)
                    .map(|i| ((seq_idx * 71 + t * 19 + kv_dim - i) as f32) * 0.0005 - 0.2)
                    .collect();
                unsafe {
                    ctx.memcpy_h2d(d_k, k.as_ptr() as *const c_void, bytes_kv)?;
                    ctx.memcpy_h2d(d_v, v.as_ptr() as *const c_void, bytes_kv)?;
                    kv.append_token_f32(&ctx, seq_idx as u32, d_k as *const c_void, d_v)?;
                }
            }
        }

        let mut q = Vec::with_capacity(batch_size as usize * q_dim);
        for b in 0..batch_size as usize {
            for i in 0..q_dim {
                q.push((b * 31 + i) as f32 * 0.0009 - 0.15);
            }
        }
        let bytes_q = q.len() * std::mem::size_of::<f32>();
        let bytes_out = bytes_q;
        let d_q = ctx.device_malloc(bytes_q)?;
        let d_out_batch = ctx.device_malloc(bytes_out)?;
        let d_out_async = ctx.device_malloc(bytes_out)?;
        let d_out_scheduler = ctx.device_malloc(bytes_out)?;
        let d_out_one = ctx.device_malloc(q_dim * std::mem::size_of::<f32>())?;
        let d_seq_ids = ctx.device_malloc(seq_ids.len() * std::mem::size_of::<u32>())?;
        let d_seq_lens = ctx.device_malloc(seq_lens.len() * std::mem::size_of::<u32>())?;

        unsafe {
            ctx.memcpy_h2d(d_q, f32s_to_bytes(&q).as_ptr() as *const c_void, bytes_q)?;
            ctx.memcpy_h2d(
                d_seq_ids,
                seq_ids.as_ptr() as *const c_void,
                seq_ids.len() * std::mem::size_of::<u32>(),
            )?;
            ctx.memcpy_h2d(
                d_seq_lens,
                seq_lens.as_ptr() as *const c_void,
                seq_lens.len() * std::mem::size_of::<u32>(),
            )?;
            kv.attention_last_token_f32_gqa_batched(
                &ctx,
                d_seq_ids as *const u32,
                d_seq_lens as *const u32,
                batch_size,
                d_q as *const c_void,
                q_heads,
                d_out_batch,
            )?;
            kv.attention_last_token_f32_gqa_batched_async(
                &ctx,
                d_seq_ids as *const u32,
                d_seq_lens as *const u32,
                batch_size,
                d_q as *const c_void,
                q_heads,
                d_out_async,
            )?;
            ctx.synchronize_stream(CudaStream::Decode)?;
        }

        let scheduler_plan = DecodeBatchPlan::from_requests(&[
            DecodeRequestState::active(100, 0, 1),
            DecodeRequestState::completed(101, 9, 7),
            DecodeRequestState::active(102, 1, 3),
            DecodeRequestState::cancelled(103, 10, 4),
            DecodeRequestState::active(104, 2, 5),
        ])?;
        assert_eq!(scheduler_plan.metadata().total_q_tokens(), batch_size);
        assert_eq!(
            scheduler_plan.metadata().total_kv_tokens(),
            seq_lens.iter().sum::<u32>()
        );
        let scheduler_plan = CudaDecodeBatchPlan::new(&ctx, scheduler_plan)?;
        unsafe {
            scheduler_plan.dispatch_attention(
                &ctx,
                &kv,
                d_q as *const c_void,
                q_heads,
                d_out_scheduler,
            )?;
        }

        let mut default_bytes = vec![0u8; bytes_out];
        unsafe {
            ctx.memcpy_d2h(
                default_bytes.as_mut_ptr() as *mut c_void,
                d_out_batch,
                bytes_out,
            )?;
        }
        let default = bytes_to_f32s(&default_bytes);
        let mut async_bytes = vec![0u8; bytes_out];
        unsafe {
            ctx.memcpy_d2h(
                async_bytes.as_mut_ptr() as *mut c_void,
                d_out_async,
                bytes_out,
            )?;
        }
        let async_result = bytes_to_f32s(&async_bytes);
        for (i, (g, e)) in async_result.iter().zip(default.iter()).enumerate() {
            let diff = (g - e).abs();
            assert!(
                diff <= 1e-6,
                "head_dim={head_dim} async mismatch at {i}: got {g}, expected {e}"
            );
        }
        let mut scheduler_bytes = vec![0u8; bytes_out];
        unsafe {
            ctx.memcpy_d2h(
                scheduler_bytes.as_mut_ptr() as *mut c_void,
                d_out_scheduler,
                bytes_out,
            )?;
        }
        let scheduler_result = bytes_to_f32s(&scheduler_bytes);
        for (i, (g, e)) in scheduler_result.iter().zip(default.iter()).enumerate() {
            let diff = (g - e).abs();
            assert!(
                diff <= 1e-6,
                "head_dim={head_dim} scheduler mismatch at {i}: got {g}, expected {e}"
            );
        }

        if head_dim == 64 {
            let d_out_ldg = ctx.device_malloc(bytes_out)?;
            unsafe {
                std::env::set_var("M40LLM_CACHE_EXPERIMENT", "ldg_kv");
                let ldg_result = kv.attention_last_token_f32_gqa_batched(
                    &ctx,
                    d_seq_ids as *const u32,
                    d_seq_lens as *const u32,
                    batch_size,
                    d_q as *const c_void,
                    q_heads,
                    d_out_ldg,
                );
                std::env::remove_var("M40LLM_CACHE_EXPERIMENT");
                ldg_result?;
                let mut ldg_bytes = vec![0u8; bytes_out];
                ctx.memcpy_d2h(ldg_bytes.as_mut_ptr() as *mut c_void, d_out_ldg, bytes_out)?;
                let ldg = bytes_to_f32s(&ldg_bytes);
                for (i, (g, e)) in ldg.iter().zip(default.iter()).enumerate() {
                    let diff = (g - e).abs();
                    assert!(
                        diff <= 1e-6,
                        "head_dim={head_dim} ldg mismatch at {i}: got {g}, expected {e}"
                    );
                }
                ctx.device_free(d_out_ldg)?;
            }
        }

        let mut expected = Vec::with_capacity(q.len());
        for (b, &seq_len) in seq_lens.iter().enumerate() {
            let q_offset = b * q_dim * std::mem::size_of::<f32>();
            unsafe {
                kv.attention_last_token_f32_gqa(
                    &ctx,
                    b as u32,
                    (d_q as usize + q_offset) as *const c_void,
                    q_heads,
                    seq_len,
                    d_out_one,
                )?;
                let mut bytes = vec![0u8; q_dim * std::mem::size_of::<f32>()];
                ctx.memcpy_d2h(bytes.as_mut_ptr() as *mut c_void, d_out_one, bytes.len())?;
                expected.extend(bytes_to_f32s(&bytes));
            }
        }

        for (i, (g, e)) in default.iter().zip(expected.iter()).enumerate() {
            let diff = (g - e).abs();
            assert!(
                diff <= 1e-4,
                "head_dim={head_dim} mismatch at {i}: got {g}, expected {e}"
            );
        }

        unsafe {
            ctx.device_free(d_k)?;
            ctx.device_free(d_v)?;
            ctx.device_free(d_q)?;
            ctx.device_free(d_out_batch)?;
            ctx.device_free(d_out_async)?;
            ctx.device_free(d_out_scheduler)?;
            ctx.device_free(d_out_one)?;
            ctx.device_free(d_seq_ids)?;
            ctx.device_free(d_seq_lens)?;
        }
    }
    Ok(())
}

#[test]
fn batched_fp16_k_q4_v_direct_attention_matches_individual_decode() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    for head_dim in [64u32, 128u32] {
        let q_heads = 4u32;
        let kv_heads = 2u32;
        let max_seq_len = 8u32;
        let batch_size = 2u32;
        let seq_lens = [3u32, 5];
        let seq_ids = [0u32, 1];
        let recent_window = 2u32;
        let block_size = 2u32;
        let top_blocks = 2u32;
        let q_dim = (q_heads * head_dim) as usize;
        let kv_dim = (kv_heads * head_dim) as usize;
        let kv = KVCache::new_compressed_with_context(
            &ctx,
            max_seq_len,
            batch_size,
            kv_heads,
            head_dim,
            recent_window,
            block_size,
            top_blocks,
            0,
            KvRepresentativePolicy::Last,
            ExactOldBacking::Fp16KQ4V,
        )?;

        let bytes_kv = kv_dim * std::mem::size_of::<f32>();
        let d_k = ctx.device_malloc(bytes_kv)?;
        let d_v = ctx.device_malloc(bytes_kv)?;
        for (seq_idx, &seq_len) in seq_lens.iter().enumerate() {
            for t in 0..seq_len as usize {
                let k: Vec<f32> = (0..kv_dim)
                    .map(|i| ((seq_idx * 101 + t * 17 + i) as f32) * 0.0006 - 0.25)
                    .collect();
                let v: Vec<f32> = (0..kv_dim)
                    .map(|i| ((seq_idx * 83 + t * 23 + kv_dim - i) as f32) * 0.0004 - 0.18)
                    .collect();
                unsafe {
                    ctx.memcpy_h2d(d_k, k.as_ptr() as *const c_void, bytes_kv)?;
                    ctx.memcpy_h2d(d_v, v.as_ptr() as *const c_void, bytes_kv)?;
                    kv.append_token_f32_rope_k_at_async(
                        &ctx,
                        seq_idx as u32,
                        d_k as *const c_void,
                        d_v,
                        t as u32,
                        t as u32,
                        10000.0,
                        1.0,
                    )?;
                    ctx.synchronize_stream(CudaStream::Decode)?;
                }
            }
        }

        let mut q = Vec::with_capacity(batch_size as usize * q_dim);
        for b in 0..batch_size as usize {
            for i in 0..q_dim {
                q.push((b * 37 + i) as f32 * 0.0008 - 0.12);
            }
        }
        let bytes_q = q.len() * std::mem::size_of::<f32>();
        let bytes_out = bytes_q;
        let d_q = ctx.device_malloc(bytes_q)?;
        let d_out_batch = ctx.device_malloc(bytes_out)?;
        let d_out_one = ctx.device_malloc(q_dim * std::mem::size_of::<f32>())?;

        let plan = DecodeBatchPlan::from_requests(&[
            DecodeRequestState::active(10, seq_ids[0], seq_lens[0]),
            DecodeRequestState::active(11, seq_ids[1], seq_lens[1]),
        ])?;
        let cuda_plan = CudaDecodeBatchPlan::new(&ctx, plan)?;

        unsafe {
            ctx.memcpy_h2d(d_q, f32s_to_bytes(&q).as_ptr() as *const c_void, bytes_q)?;
            cuda_plan.dispatch_fp16_k_q4_v_direct_attention_async(
                &ctx,
                &kv,
                d_q as *const c_void,
                q_heads,
                recent_window,
                block_size,
                top_blocks,
                d_out_batch,
            )?;
            ctx.synchronize_stream(CudaStream::Decode)?;
        }

        let mut batched_bytes = vec![0u8; bytes_out];
        unsafe {
            ctx.memcpy_d2h(
                batched_bytes.as_mut_ptr() as *mut c_void,
                d_out_batch,
                bytes_out,
            )?;
        }
        let batched = bytes_to_f32s(&batched_bytes);

        let mut expected = Vec::with_capacity(q.len());
        for (b, &seq_len) in seq_lens.iter().enumerate() {
            let q_offset = b * q_dim * std::mem::size_of::<f32>();
            unsafe {
                kv.attention_last_token_f32_gqa_block_select_exact_fp16_k_q4_v_old_direct_async(
                    &ctx,
                    b as u32,
                    (d_q as usize + q_offset) as *const c_void,
                    q_heads,
                    seq_len,
                    recent_window,
                    block_size,
                    top_blocks,
                    d_out_one,
                )?;
                ctx.synchronize_stream(CudaStream::Decode)?;
                let mut bytes = vec![0u8; q_dim * std::mem::size_of::<f32>()];
                ctx.memcpy_d2h(bytes.as_mut_ptr() as *mut c_void, d_out_one, bytes.len())?;
                expected.extend(bytes_to_f32s(&bytes));
            }
        }

        for (i, (got, want)) in batched.iter().zip(expected.iter()).enumerate() {
            let diff = (got - want).abs();
            assert!(
                diff <= 1e-5,
                "head_dim={head_dim} compressed batched mismatch at {i}: got {got}, expected {want}"
            );
        }

        unsafe {
            ctx.device_free(d_k)?;
            ctx.device_free(d_v)?;
            ctx.device_free(d_q)?;
            ctx.device_free(d_out_batch)?;
            ctx.device_free(d_out_one)?;
        }
    }
    Ok(())
}
