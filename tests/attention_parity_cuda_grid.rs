#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;

use anyhow::Result;
use half::f16;
use m40_llm::cuda::{
    ffi_debug_read_kv_token, ExactBlockStagingWorkspace, ExactOldBacking, KVCache,
};
use m40_llm::kv_compression::{
    set_runtime_config, KvCompressMode, KvCompressionConfig, KvRepresentativePolicy,
};
use std::ffi::c_void;

fn cast_f32_to_f16_then_back(vals: &[f32]) -> Vec<f32> {
    vals.iter()
        .map(|&x| half::f16::from_f32(x).to_f32())
        .collect()
}

fn f16_words_to_f32(vals: &[u16]) -> Vec<f32> {
    vals.iter().map(|&x| f16::from_bits(x).to_f32()).collect()
}

fn read_dense_kv_token(
    ctx: &m40_llm::cuda::CudaContext,
    kv: &KVCache,
    token: u32,
    elems_per_token: usize,
) -> (Vec<u16>, Vec<u16>) {
    let mut k_bytes = vec![0u8; elems_per_token * 2];
    let mut v_bytes = vec![0u8; elems_per_token * 2];
    let rc = unsafe {
        ffi_debug_read_kv_token(
            ctx,
            kv,
            0,
            token,
            k_bytes.as_mut_ptr(),
            v_bytes.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0, "ffi_debug_read_kv_token failed for token {token}");
    let k = k_bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();
    let v = v_bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();
    (k, v)
}

#[test]
fn attention_dense_recent_window_matches_reference() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let q_heads = 2u32;
    let kv_heads = 2u32;
    let head_dim = 64u32;
    let seq_len = 12u32;
    let recent_window = 5u32;
    let q_dim = (q_heads * head_dim) as usize;
    let kv_dim = (kv_heads * head_dim) as usize;
    let kv = KVCache::new_with_context(&ctx, seq_len, 1, kv_heads, head_dim)?;

    let mut k_tokens = Vec::new();
    let mut v_tokens = Vec::new();
    for t in 0..seq_len as usize {
        let k: Vec<f32> = (0..kv_dim)
            .map(|i| ((t * 29 + i * 3) as f32) * 0.001 - 0.25)
            .collect();
        let v: Vec<f32> = (0..kv_dim)
            .map(|i| ((t * 11 + kv_dim - i) as f32) * 0.002 - 0.15)
            .collect();
        let k_stored = cast_f32_to_f16_then_back(&k);
        let v_stored = cast_f32_to_f16_then_back(&v);
        let bytes = kv_dim * std::mem::size_of::<f32>();
        let d_k = ctx.device_malloc(bytes)?;
        let d_v = ctx.device_malloc(bytes)?;
        unsafe {
            ctx.memcpy_h2d(d_k, k.as_ptr() as *const c_void, bytes)?;
            ctx.memcpy_h2d(d_v, v.as_ptr() as *const c_void, bytes)?;
            kv.append_token_f32(&ctx, 0, d_k as *const c_void, d_v as *const c_void)?;
            ctx.device_free(d_k)?;
            ctx.device_free(d_v)?;
        }
        k_tokens.push(k_stored);
        v_tokens.push(v_stored);
    }

    let q: Vec<f32> = (0..q_dim).map(|i| (i as f32) * 0.0015 - 0.12).collect();
    let bytes_q = q_dim * std::mem::size_of::<f32>();
    let d_q = ctx.device_malloc(bytes_q)?;
    let d_dense = ctx.device_malloc(bytes_q)?;
    let d_full_window = ctx.device_malloc(bytes_q)?;
    let d_recent = ctx.device_malloc(bytes_q)?;
    unsafe {
        ctx.memcpy_h2d(d_q, q.as_ptr() as *const c_void, bytes_q)?;
        kv.attention_last_token_f32_gqa(&ctx, 0, d_q as *const c_void, q_heads, seq_len, d_dense)?;
        kv.attention_last_token_f32_gqa_dense_recent_async(
            &ctx,
            0,
            d_q as *const c_void,
            q_heads,
            seq_len,
            seq_len,
            d_full_window,
        )?;
        kv.attention_last_token_f32_gqa_dense_recent_async(
            &ctx,
            0,
            d_q as *const c_void,
            q_heads,
            seq_len,
            recent_window,
            d_recent,
        )?;
        ctx.synchronize_stream(m40_llm::cuda::CudaStream::Decode)?;
    }

    let mut dense = vec![0f32; q_dim];
    let mut full_window = vec![0f32; q_dim];
    let mut recent = vec![0f32; q_dim];
    unsafe {
        ctx.memcpy_d2h(
            dense.as_mut_ptr() as *mut c_void,
            d_dense as *const c_void,
            bytes_q,
        )?;
        ctx.memcpy_d2h(
            full_window.as_mut_ptr() as *mut c_void,
            d_full_window as *const c_void,
            bytes_q,
        )?;
        ctx.memcpy_d2h(
            recent.as_mut_ptr() as *mut c_void,
            d_recent as *const c_void,
            bytes_q,
        )?;
        ctx.device_free(d_q)?;
        ctx.device_free(d_dense)?;
        ctx.device_free(d_full_window)?;
        ctx.device_free(d_recent)?;
    }

    let start = (seq_len - recent_window) as usize;
    let recent_ref = cpu_last_token_attention(
        &q,
        &k_tokens[start..],
        &v_tokens[start..],
        q_heads as usize,
        head_dim as usize,
    );

    for (idx, ((&dense_val, &full_window_val), (&recent_val, &ref_val))) in dense
        .iter()
        .zip(&full_window)
        .zip(recent.iter().zip(&recent_ref))
        .enumerate()
    {
        assert!(
            (dense_val - full_window_val).abs() < 1e-3,
            "full-window dense mismatch at {idx}: dense={dense_val} window={full_window_val}"
        );
        assert!(
            (recent_val - ref_val).abs() < 1e-3,
            "recent-window reference mismatch at {idx}: cuda={recent_val} cpu={ref_val}"
        );
    }

    Ok(())
}

#[test]
fn attention_block_select_exact_matches_dense_when_all_old_blocks_selected() -> Result<()> {
    struct EnvGuard {
        key: &'static str,
        previous: Option<String>,
    }
    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var(key).ok();
            unsafe {
                std::env::set_var(key, value);
            }
            Self { key, previous }
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            unsafe {
                match &self.previous {
                    Some(value) => std::env::set_var(self.key, value),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }
    let _q8_env = EnvGuard::set("M40LLM_KV_EXACT_OLD_BACKING", "q8");
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let q_heads = 4u32;
    let kv_heads = 2u32;
    let head_dim = 64u32;
    let seq_len = 16u32;
    let recent_window = 4u32;
    let block_size = 4u32;
    let top_blocks = 3u32;
    let q_dim = (q_heads * head_dim) as usize;
    let kv_dim = (kv_heads * head_dim) as usize;
    let kv = KVCache::new_with_context(&ctx, seq_len, 1, kv_heads, head_dim)?;
    let kv_q8 = KVCache::new_compressed_with_context(
        &ctx,
        seq_len,
        1,
        kv_heads,
        head_dim,
        recent_window,
        block_size,
        top_blocks,
        0,
        KvRepresentativePolicy::Last,
        ExactOldBacking::Q8,
    )?;

    for t in 0..seq_len as usize {
        let k: Vec<f32> = (0..kv_dim)
            .map(|i| ((t * 31 + i * 7) as f32) * 0.002 - 0.4)
            .collect();
        let v: Vec<f32> = (0..kv_dim)
            .map(|i| ((t * 17 + kv_dim - i) as f32) * 0.003 - 0.2)
            .collect();
        let bytes = kv_dim * std::mem::size_of::<f32>();
        let d_k = ctx.device_malloc(bytes)?;
        let d_v = ctx.device_malloc(bytes)?;
        unsafe {
            ctx.memcpy_h2d(d_k, k.as_ptr() as *const c_void, bytes)?;
            ctx.memcpy_h2d(d_v, v.as_ptr() as *const c_void, bytes)?;
            kv.append_token_f32_rope_k_at_async(
                &ctx,
                0,
                d_k as *const c_void,
                d_v as *const c_void,
                t as u32,
                t as u32,
                10000.0,
                0.0,
            )?;
            kv_q8.append_token_f32_rope_k_at_async(
                &ctx,
                0,
                d_k as *const c_void,
                d_v as *const c_void,
                t as u32,
                t as u32,
                10000.0,
                0.0,
            )?;
            ctx.device_free(d_k)?;
            ctx.device_free(d_v)?;
        }
    }

    let q: Vec<f32> = (0..q_dim).map(|i| (i as f32) * 0.001 - 0.1).collect();
    let bytes_q = q_dim * std::mem::size_of::<f32>();
    let d_q = ctx.device_malloc(bytes_q)?;
    let d_dense = ctx.device_malloc(bytes_q)?;
    let d_sparse = ctx.device_malloc(bytes_q)?;
    let d_staged = ctx.device_malloc(bytes_q)?;
    let d_q8_staged = ctx.device_malloc(bytes_q)?;
    let staging_capacity_tokens = recent_window + top_blocks * block_size;
    let staging =
        ExactBlockStagingWorkspace::new(&ctx, q_heads, head_dim, staging_capacity_tokens)?;
    let q8_staging =
        ExactBlockStagingWorkspace::new(&ctx, q_heads, head_dim, staging_capacity_tokens)?;
    unsafe {
        ctx.memcpy_h2d(d_q, q.as_ptr() as *const c_void, bytes_q)?;
        kv.attention_last_token_f32_gqa(&ctx, 0, d_q as *const c_void, q_heads, seq_len, d_dense)?;
        kv.attention_last_token_f32_gqa_block_select_exact_async(
            &ctx,
            0,
            d_q as *const c_void,
            q_heads,
            seq_len,
            recent_window,
            block_size,
            top_blocks,
            d_sparse,
        )?;
        kv.attention_last_token_f32_gqa_block_select_exact_staged_with_buffers_async(
            &ctx,
            0,
            d_q as *const c_void,
            q_heads,
            seq_len,
            recent_window,
            block_size,
            top_blocks,
            staging.ptrs(),
            d_staged,
        )?;
        kv_q8.attention_last_token_f32_gqa_block_select_exact_staged_q8_old_with_buffers_async(
            &ctx,
            0,
            d_q as *const c_void,
            q_heads,
            seq_len,
            recent_window,
            block_size,
            top_blocks,
            q8_staging.ptrs(),
            d_q8_staged,
        )?;
        ctx.synchronize_stream(m40_llm::cuda::CudaStream::Decode)?;
    }

    let mut dense = vec![0f32; q_dim];
    let mut sparse = vec![0f32; q_dim];
    let mut staged = vec![0f32; q_dim];
    let mut q8_staged = vec![0f32; q_dim];
    unsafe {
        ctx.memcpy_d2h(
            dense.as_mut_ptr() as *mut c_void,
            d_dense as *const c_void,
            bytes_q,
        )?;
        ctx.memcpy_d2h(
            sparse.as_mut_ptr() as *mut c_void,
            d_sparse as *const c_void,
            bytes_q,
        )?;
        ctx.memcpy_d2h(
            staged.as_mut_ptr() as *mut c_void,
            d_staged as *const c_void,
            bytes_q,
        )?;
        ctx.memcpy_d2h(
            q8_staged.as_mut_ptr() as *mut c_void,
            d_q8_staged as *const c_void,
            bytes_q,
        )?;
        ctx.device_free(d_q)?;
        ctx.device_free(d_dense)?;
        ctx.device_free(d_sparse)?;
        ctx.device_free(d_staged)?;
        ctx.device_free(d_q8_staged)?;
    }

    for (idx, (((&a, &b), &c), &q8)) in dense
        .iter()
        .zip(&sparse)
        .zip(&staged)
        .zip(&q8_staged)
        .enumerate()
    {
        assert!(
            (a - b).abs() < 1e-3,
            "block-select-exact mismatch at {idx}: dense={a} sparse={b}"
        );
        assert!(
            (b - c).abs() < 1e-3,
            "staged block-select-exact mismatch at {idx}: direct={b} staged={c}"
        );
        assert!(
            (c - q8).abs() < 2e-2,
            "q8 staged block-select-exact mismatch at {idx}: staged={c} q8={q8}"
        );
    }

    set_runtime_config(KvCompressionConfig {
        mode: KvCompressMode::Off,
        ..Default::default()
    });
    Ok(())
}

#[test]
fn attention_block_summary_lossy_is_finite_and_deterministic() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let q_heads = 4u32;
    let kv_heads = 2u32;
    let head_dim = 64u32;
    let seq_len = 20u32;
    let recent_window = 4u32;
    let block_size = 4u32;
    let q_dim = (q_heads * head_dim) as usize;
    let kv_dim = (kv_heads * head_dim) as usize;
    let kv = KVCache::new_with_context(&ctx, seq_len, 1, kv_heads, head_dim)?;

    for t in 0..seq_len as usize {
        let k: Vec<f32> = (0..kv_dim)
            .map(|i| ((t * 13 + i * 5) as f32).sin() * 0.02)
            .collect();
        let v: Vec<f32> = (0..kv_dim)
            .map(|i| ((t * 19 + i * 3) as f32).cos() * 0.03)
            .collect();
        let bytes = kv_dim * std::mem::size_of::<f32>();
        let d_k = ctx.device_malloc(bytes)?;
        let d_v = ctx.device_malloc(bytes)?;
        unsafe {
            ctx.memcpy_h2d(d_k, k.as_ptr() as *const c_void, bytes)?;
            ctx.memcpy_h2d(d_v, v.as_ptr() as *const c_void, bytes)?;
            kv.append_token_f32(&ctx, 0, d_k as *const c_void, d_v as *const c_void)?;
            ctx.device_free(d_k)?;
            ctx.device_free(d_v)?;
        }
    }

    let q: Vec<f32> = (0..q_dim).map(|i| (i as f32).sin() * 0.01).collect();
    let bytes_q = q_dim * std::mem::size_of::<f32>();
    let d_q = ctx.device_malloc(bytes_q)?;
    let d_summary_a = ctx.device_malloc(bytes_q)?;
    let d_summary_b = ctx.device_malloc(bytes_q)?;
    let d_select_lossy = ctx.device_malloc(bytes_q)?;
    unsafe {
        ctx.memcpy_h2d(d_q, q.as_ptr() as *const c_void, bytes_q)?;
        kv.attention_last_token_f32_gqa_block_summary_lossy_async(
            &ctx,
            0,
            d_q as *const c_void,
            q_heads,
            seq_len,
            recent_window,
            block_size,
            0,
            d_summary_a,
        )?;
        kv.attention_last_token_f32_gqa_block_summary_lossy_async(
            &ctx,
            0,
            d_q as *const c_void,
            q_heads,
            seq_len,
            recent_window,
            block_size,
            0,
            d_summary_b,
        )?;
        kv.attention_last_token_f32_gqa_block_summary_lossy_async(
            &ctx,
            0,
            d_q as *const c_void,
            q_heads,
            seq_len,
            recent_window,
            block_size,
            2,
            d_select_lossy,
        )?;
        ctx.synchronize_stream(m40_llm::cuda::CudaStream::Decode)?;
    }

    let mut summary_a = vec![0f32; q_dim];
    let mut summary_b = vec![0f32; q_dim];
    let mut select_lossy = vec![0f32; q_dim];
    unsafe {
        ctx.memcpy_d2h(
            summary_a.as_mut_ptr() as *mut c_void,
            d_summary_a as *const c_void,
            bytes_q,
        )?;
        ctx.memcpy_d2h(
            summary_b.as_mut_ptr() as *mut c_void,
            d_summary_b as *const c_void,
            bytes_q,
        )?;
        ctx.memcpy_d2h(
            select_lossy.as_mut_ptr() as *mut c_void,
            d_select_lossy as *const c_void,
            bytes_q,
        )?;
        ctx.device_free(d_q)?;
        ctx.device_free(d_summary_a)?;
        ctx.device_free(d_summary_b)?;
        ctx.device_free(d_select_lossy)?;
    }

    for idx in 0..q_dim {
        assert!(
            summary_a[idx].is_finite(),
            "summary output {idx} is not finite"
        );
        assert!(
            select_lossy[idx].is_finite(),
            "select-lossy output {idx} is not finite"
        );
        assert!(
            (summary_a[idx] - summary_b[idx]).abs() < 1e-6,
            "summary output not deterministic at {idx}: {} vs {}",
            summary_a[idx],
            summary_b[idx]
        );
    }
    Ok(())
}

#[test]
fn compressed_kv_recent_window_matches_dense_attention() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let q_heads = 4u32;
    let kv_heads = 2u32;
    let head_dim = 64u32;
    let max_seq_len = 32u32;
    let seq_len = 8u32;
    let recent_window = 16u32;
    let block_size = 4u32;
    let q_dim = (q_heads * head_dim) as usize;
    let kv_dim = (kv_heads * head_dim) as usize;
    let dense = KVCache::new_with_context(&ctx, max_seq_len, 1, kv_heads, head_dim)?;
    let compressed = KVCache::new_compressed_with_context(
        &ctx,
        max_seq_len,
        1,
        kv_heads,
        head_dim,
        recent_window,
        block_size,
        2,
        0,
        KvRepresentativePolicy::Last,
        ExactOldBacking::Dense,
    )?;
    assert!(compressed.is_compressed());
    assert!(compressed.actual_bytes() < compressed.dense_equivalent_bytes());

    for t in 0..seq_len as usize {
        let k: Vec<f32> = (0..kv_dim)
            .map(|i| ((t * 11 + i * 3) as f32) * 0.001 - 0.05)
            .collect();
        let v: Vec<f32> = (0..kv_dim)
            .map(|i| ((t * 13 + kv_dim - i) as f32) * 0.0015 - 0.07)
            .collect();
        let bytes = kv_dim * std::mem::size_of::<f32>();
        let d_k = ctx.device_malloc(bytes)?;
        let d_v = ctx.device_malloc(bytes)?;
        unsafe {
            ctx.memcpy_h2d(d_k, k.as_ptr() as *const c_void, bytes)?;
            ctx.memcpy_h2d(d_v, v.as_ptr() as *const c_void, bytes)?;
            dense.append_token_f32_rope_k_at_async(
                &ctx,
                0,
                d_k as *const c_void,
                d_v as *const c_void,
                t as u32,
                t as u32,
                10_000.0,
                1.0,
            )?;
            compressed.append_token_f32_rope_k_at_async(
                &ctx,
                0,
                d_k as *const c_void,
                d_v as *const c_void,
                t as u32,
                t as u32,
                10_000.0,
                1.0,
            )?;
            ctx.device_free(d_k)?;
            ctx.device_free(d_v)?;
        }
    }

    let q: Vec<f32> = (0..q_dim).map(|i| (i as f32) * 0.0007 - 0.02).collect();
    let bytes_q = q_dim * std::mem::size_of::<f32>();
    let d_q = ctx.device_malloc(bytes_q)?;
    let d_dense = ctx.device_malloc(bytes_q)?;
    let d_compressed = ctx.device_malloc(bytes_q)?;
    let d_recent_only = ctx.device_malloc(bytes_q)?;
    unsafe {
        ctx.memcpy_h2d(d_q, q.as_ptr() as *const c_void, bytes_q)?;
        dense.attention_last_token_f32_gqa(
            &ctx,
            0,
            d_q as *const c_void,
            q_heads,
            seq_len,
            d_dense,
        )?;
        compressed.attention_last_token_f32_gqa_block_summary_lossy_async(
            &ctx,
            0,
            d_q as *const c_void,
            q_heads,
            seq_len,
            recent_window,
            block_size,
            0,
            d_compressed,
        )?;
        compressed.attention_last_token_f32_gqa_compressed_recent_only_async(
            &ctx,
            0,
            d_q as *const c_void,
            q_heads,
            seq_len,
            d_recent_only,
        )?;
        ctx.synchronize_stream(m40_llm::cuda::CudaStream::Decode)?;
    }

    let mut dense_out = vec![0f32; q_dim];
    let mut compressed_out = vec![0f32; q_dim];
    let mut recent_only_out = vec![0f32; q_dim];
    unsafe {
        ctx.memcpy_d2h(
            dense_out.as_mut_ptr() as *mut c_void,
            d_dense as *const c_void,
            bytes_q,
        )?;
        ctx.memcpy_d2h(
            compressed_out.as_mut_ptr() as *mut c_void,
            d_compressed as *const c_void,
            bytes_q,
        )?;
        ctx.memcpy_d2h(
            recent_only_out.as_mut_ptr() as *mut c_void,
            d_recent_only as *const c_void,
            bytes_q,
        )?;
        ctx.device_free(d_q)?;
        ctx.device_free(d_dense)?;
        ctx.device_free(d_compressed)?;
        ctx.device_free(d_recent_only)?;
    }

    for (idx, ((&a, &b), &c)) in dense_out
        .iter()
        .zip(&compressed_out)
        .zip(&recent_only_out)
        .enumerate()
    {
        assert!(
            (a - b).abs() < 1e-3,
            "compressed recent mismatch at {idx}: dense={a} compressed={b}"
        );
        assert!(
            (a - c).abs() < 1e-3,
            "recent-only mismatch at {idx}: dense={a} recent_only={c}"
        );
    }
    Ok(())
}

#[test]
fn compressed_kv_recent_ring_matches_dense_window_after_wrap() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let q_heads = 4u32;
    let kv_heads = 2u32;
    let head_dim = 64u32;
    let max_seq_len = 32u32;
    let seq_len = 18u32;
    let recent_window = 7u32;
    let block_size = 4u32;
    let q_dim = (q_heads * head_dim) as usize;
    let kv_dim = (kv_heads * head_dim) as usize;
    let dense = KVCache::new_with_context(&ctx, max_seq_len, 1, kv_heads, head_dim)?;
    let compressed_seq = KVCache::new_compressed_with_context(
        &ctx,
        max_seq_len,
        1,
        kv_heads,
        head_dim,
        recent_window,
        block_size,
        2,
        0,
        KvRepresentativePolicy::Last,
        ExactOldBacking::Dense,
    )?;
    let compressed_built = KVCache::new_compressed_with_context(
        &ctx,
        max_seq_len,
        1,
        kv_heads,
        head_dim,
        recent_window,
        block_size,
        2,
        0,
        KvRepresentativePolicy::Last,
        ExactOldBacking::Dense,
    )?;

    for t in 0..seq_len as usize {
        let k: Vec<f32> = (0..kv_dim)
            .map(|i| ((t * 17 + i * 5) as f32).sin() * 0.08 + (i as f32) * 0.0003)
            .collect();
        let v: Vec<f32> = (0..kv_dim)
            .map(|i| ((t * 19 + i * 7) as f32).cos() * 0.07 - (i as f32) * 0.0002)
            .collect();
        let bytes = kv_dim * std::mem::size_of::<f32>();
        let d_k = ctx.device_malloc(bytes)?;
        let d_v = ctx.device_malloc(bytes)?;
        unsafe {
            ctx.memcpy_h2d(d_k, k.as_ptr() as *const c_void, bytes)?;
            ctx.memcpy_h2d(d_v, v.as_ptr() as *const c_void, bytes)?;
            dense.append_token_f32_rope_k_at_async(
                &ctx,
                0,
                d_k as *const c_void,
                d_v as *const c_void,
                t as u32,
                t as u32,
                10_000.0,
                1.0,
            )?;
            compressed_seq.append_token_f32_rope_k_at_async(
                &ctx,
                0,
                d_k as *const c_void,
                d_v as *const c_void,
                t as u32,
                t as u32,
                10_000.0,
                1.0,
            )?;
            ctx.device_free(d_k)?;
            ctx.device_free(d_v)?;
        }
    }
    compressed_built.build_compressed_from_dense(&ctx, &dense, seq_len)?;

    let seq_snapshot = compressed_seq.debug_compressed_snapshot(&ctx, 0)?;
    let built_snapshot = compressed_built.debug_compressed_snapshot(&ctx, 0)?;
    assert_eq!(seq_snapshot.seq_len, seq_len);
    assert_eq!(built_snapshot.seq_len, seq_len);

    let sampled_positions = [
        seq_len - recent_window,
        seq_len - recent_window + 1,
        seq_len - 2,
        seq_len - 1,
    ];
    for pos in sampled_positions {
        let ring = (pos % recent_window) as usize;
        let base = ring * kv_dim;
        let (dense_k, dense_v) = read_dense_kv_token(&ctx, &dense, pos, kv_dim);
        for i in 0..kv_dim {
            assert_eq!(
                seq_snapshot.recent_k_f16[base + i],
                dense_k[i],
                "sequential compressed K mismatch pos={pos} ring={ring} elem={i}"
            );
            assert_eq!(
                seq_snapshot.recent_v_f16[base + i],
                dense_v[i],
                "sequential compressed V mismatch pos={pos} ring={ring} elem={i}"
            );
            assert_eq!(
                built_snapshot.recent_k_f16[base + i],
                dense_k[i],
                "built compressed K mismatch pos={pos} ring={ring} elem={i}"
            );
            assert_eq!(
                built_snapshot.recent_v_f16[base + i],
                dense_v[i],
                "built compressed V mismatch pos={pos} ring={ring} elem={i}"
            );
        }
    }

    let q: Vec<f32> = (0..q_dim).map(|i| (i as f32) * 0.0009 - 0.04).collect();
    let bytes_q = q_dim * std::mem::size_of::<f32>();
    let d_q = ctx.device_malloc(bytes_q)?;
    let d_dense_window = ctx.device_malloc(bytes_q)?;
    let d_seq_recent = ctx.device_malloc(bytes_q)?;
    let d_built_recent = ctx.device_malloc(bytes_q)?;
    unsafe {
        ctx.memcpy_h2d(d_q, q.as_ptr() as *const c_void, bytes_q)?;
        dense.attention_last_token_f32_gqa_dense_recent_async(
            &ctx,
            0,
            d_q as *const c_void,
            q_heads,
            seq_len,
            recent_window,
            d_dense_window,
        )?;
        compressed_seq.attention_last_token_f32_gqa_compressed_recent_only_async(
            &ctx,
            0,
            d_q as *const c_void,
            q_heads,
            seq_len,
            d_seq_recent,
        )?;
        compressed_built.attention_last_token_f32_gqa_compressed_recent_only_async(
            &ctx,
            0,
            d_q as *const c_void,
            q_heads,
            seq_len,
            d_built_recent,
        )?;
        ctx.synchronize_stream(m40_llm::cuda::CudaStream::Decode)?;
    }

    let mut dense_window = vec![0f32; q_dim];
    let mut seq_recent = vec![0f32; q_dim];
    let mut built_recent = vec![0f32; q_dim];
    unsafe {
        ctx.memcpy_d2h(
            dense_window.as_mut_ptr() as *mut c_void,
            d_dense_window as *const c_void,
            bytes_q,
        )?;
        ctx.memcpy_d2h(
            seq_recent.as_mut_ptr() as *mut c_void,
            d_seq_recent as *const c_void,
            bytes_q,
        )?;
        ctx.memcpy_d2h(
            built_recent.as_mut_ptr() as *mut c_void,
            d_built_recent as *const c_void,
            bytes_q,
        )?;
        ctx.device_free(d_q)?;
        ctx.device_free(d_dense_window)?;
        ctx.device_free(d_seq_recent)?;
        ctx.device_free(d_built_recent)?;
    }

    for (idx, ((&dense_val, &seq_val), &built_val)) in dense_window
        .iter()
        .zip(&seq_recent)
        .zip(&built_recent)
        .enumerate()
    {
        assert!(
            (dense_val - seq_val).abs() < 1e-3,
            "sequential compressed recent attention mismatch at {idx}: dense_window={dense_val} compressed={seq_val}"
        );
        assert!(
            (dense_val - built_val).abs() < 1e-3,
            "built compressed recent attention mismatch at {idx}: dense_window={dense_val} compressed={built_val}"
        );
    }

    let recent_start = (seq_len - recent_window) as usize;
    let mut recent_k = Vec::new();
    let mut recent_v = Vec::new();
    for pos in recent_start..seq_len as usize {
        let (k, v) = read_dense_kv_token(&ctx, &dense, pos as u32, kv_dim);
        recent_k.push(f16_words_to_f32(&k));
        recent_v.push(f16_words_to_f32(&v));
    }
    let cpu_ref = cpu_last_token_attention_gqa(
        &q,
        &recent_k,
        &recent_v,
        q_heads as usize,
        kv_heads as usize,
        head_dim as usize,
    );
    for (idx, (&dense_val, &ref_val)) in dense_window.iter().zip(&cpu_ref).enumerate() {
        assert!(
            (dense_val - ref_val).abs() < 1e-3,
            "dense recent-window CPU reference mismatch at {idx}: cuda={dense_val} cpu={ref_val}"
        );
    }

    Ok(())
}

fn cpu_last_token_attention(
    q: &[f32],             // [num_heads*head_dim]
    k_tokens: &[Vec<f32>], // seq_len entries, each [num_heads*head_dim]
    v_tokens: &[Vec<f32>], // seq_len entries, each [num_heads*head_dim]
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let dim = num_heads * head_dim;
    let inv_sqrt = 1.0f32 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; dim];

    for h in 0..num_heads {
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        // pass 1: max
        let mut max_s = f32::NEG_INFINITY;
        for k_t in k_tokens.iter() {
            let k_base = &k_t[h * head_dim..(h + 1) * head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += qh[d] * k_base[d];
            }
            let s = dot * inv_sqrt;
            if s > max_s {
                max_s = s;
            }
        }
        // pass 2: denom
        let mut denom = 0.0f32;
        let mut scores: Vec<f32> = Vec::with_capacity(k_tokens.len());
        for k_t in k_tokens.iter() {
            let k_base = &k_t[h * head_dim..(h + 1) * head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += qh[d] * k_base[d];
            }
            let s = dot * inv_sqrt;
            let e = (s - max_s).exp();
            scores.push(e);
            denom += e;
        }
        if denom == 0.0 {
            denom = 1.0;
        }
        // pass 3: out
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for (e, v_t) in scores.iter().zip(v_tokens.iter()) {
                let prob = *e / denom;
                let v_base = &v_t[h * head_dim..(h + 1) * head_dim];
                acc += prob * v_base[d];
            }
            out[h * head_dim + d] = acc;
        }
    }
    out
}

fn cpu_last_token_attention_gqa(
    q: &[f32],             // [q_heads*head_dim]
    k_tokens: &[Vec<f32>], // seq_len entries, each [kv_heads*head_dim]
    v_tokens: &[Vec<f32>], // seq_len entries, each [kv_heads*head_dim]
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(q_heads % kv_heads, 0);
    let dim = q_heads * head_dim;
    let inv_sqrt = 1.0f32 / (head_dim as f32).sqrt();
    let group = q_heads / kv_heads;
    let mut out = vec![0.0f32; dim];

    for qh_idx in 0..q_heads {
        let kvh_idx = qh_idx / group;
        let qh = &q[qh_idx * head_dim..(qh_idx + 1) * head_dim];
        let mut max_s = f32::NEG_INFINITY;
        for k_t in k_tokens.iter() {
            let k_base = &k_t[kvh_idx * head_dim..(kvh_idx + 1) * head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += qh[d] * k_base[d];
            }
            max_s = max_s.max(dot * inv_sqrt);
        }

        let mut denom = 0.0f32;
        let mut scores: Vec<f32> = Vec::with_capacity(k_tokens.len());
        for k_t in k_tokens.iter() {
            let k_base = &k_t[kvh_idx * head_dim..(kvh_idx + 1) * head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += qh[d] * k_base[d];
            }
            let e = (dot * inv_sqrt - max_s).exp();
            scores.push(e);
            denom += e;
        }
        if denom == 0.0 {
            denom = 1.0;
        }

        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for (e, v_t) in scores.iter().zip(v_tokens.iter()) {
                let v_base = &v_t[kvh_idx * head_dim..(kvh_idx + 1) * head_dim];
                acc += (*e / denom) * v_base[d];
            }
            out[qh_idx * head_dim + d] = acc;
        }
    }

    out
}

#[test]
fn attention_last_token_cuda_parity_grid() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let head_dims = [1u32, 5, 7, 8, 16, 31, 32];
    let num_heads_list = [1u32, 3, 4, 8];
    let seq_lens = [1u32, 2, 3, 4, 8, 16];

    let max_seq_len = *seq_lens.iter().max().unwrap();

    for &num_heads in &num_heads_list {
        for &head_dim in &head_dims {
            let dim = (num_heads * head_dim) as usize;

            for &seq_len in &seq_lens {
                let kv = KVCache::new_with_context(&ctx, max_seq_len, 1, num_heads, head_dim)?;

                // Build tokens on host with f16->f32 cast to match device storage
                let mut k_tokens_f32: Vec<Vec<f32>> = Vec::new();
                let mut v_tokens_f32: Vec<Vec<f32>> = Vec::new();
                for t in 0..seq_len as usize {
                    let mut k = Vec::with_capacity(dim);
                    let mut v = Vec::with_capacity(dim);
                    for i in 0..dim {
                        k.push(((t * dim + i) as f32) * 0.01 - 1.0);
                        v.push(((t * dim + (dim - 1 - i)) as f32) * 0.02 - 2.0);
                    }
                    let k_cast = cast_f32_to_f16_then_back(&k);
                    let v_cast = cast_f32_to_f16_then_back(&v);
                    k_tokens_f32.push(k_cast.clone());
                    v_tokens_f32.push(v_cast.clone());

                    // Upload this token via append_token_f32
                    let bytes = dim * std::mem::size_of::<f32>();
                    let d_k = ctx.device_malloc(bytes)?;
                    let d_v = ctx.device_malloc(bytes)?;
                    unsafe {
                        ctx.memcpy_h2d(d_k, k_cast.as_ptr() as *const c_void, bytes)?;
                        ctx.memcpy_h2d(d_v, v_cast.as_ptr() as *const c_void, bytes)?;
                        kv.append_token_f32(&ctx, 0, d_k as *const c_void, d_v as *const c_void)?;
                        ctx.device_free(d_k)?;
                        ctx.device_free(d_v)?;
                    }
                }

                // Q and output
                let mut q = Vec::with_capacity(dim);
                for i in 0..dim {
                    q.push((i as f32) * 0.003 - 0.5);
                }
                let bytes_f32 = dim * std::mem::size_of::<f32>();
                let d_q = ctx.device_malloc(bytes_f32)?;
                let d_out = ctx.device_malloc(bytes_f32)?;
                unsafe {
                    ctx.memcpy_h2d(d_q, q.as_ptr() as *const c_void, bytes_f32)?;
                    kv.attention_last_token_f32(&ctx, 0, d_q as *const c_void, seq_len, d_out)?;
                }
                let mut out_gpu = vec![0f32; dim];
                unsafe {
                    ctx.memcpy_d2h(
                        out_gpu.as_mut_ptr() as *mut c_void,
                        d_out as *const c_void,
                        bytes_f32,
                    )?;
                    ctx.device_free(d_q)?;
                    ctx.device_free(d_out)?;
                }

                // CPU ref
                let out_cpu = cpu_last_token_attention(
                    &q,
                    &k_tokens_f32,
                    &v_tokens_f32,
                    num_heads as usize,
                    head_dim as usize,
                );

                for i in 0..dim {
                    let a = out_cpu[i];
                    let b = out_gpu[i];
                    let diff = (a - b).abs();
                    assert!(
                        diff < 1e-3,
                        "mismatch (heads={},dim={},seq_len={},i={}) cpu={} gpu={} diff={}",
                        num_heads,
                        head_dim,
                        seq_len,
                        i,
                        a,
                        b,
                        diff
                    );
                }
            }
        }
    }

    Ok(())
}

#[test]
fn attention_last_token_cuda_gqa_parity_grid() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let head_dims = [1u32, 5, 8, 16, 32];
    let head_pairs = [(2u32, 1u32), (4, 1), (8, 2), (32, 4)];
    let seq_lens = [1u32, 2, 3, 4, 8, 16];
    let max_seq_len = *seq_lens.iter().max().unwrap();

    for &(q_heads, kv_heads) in &head_pairs {
        for &head_dim in &head_dims {
            let q_dim = (q_heads * head_dim) as usize;
            let kv_dim = (kv_heads * head_dim) as usize;

            for &seq_len in &seq_lens {
                let kv = KVCache::new_with_context(&ctx, max_seq_len, 1, kv_heads, head_dim)?;

                let mut k_tokens_f32: Vec<Vec<f32>> = Vec::new();
                let mut v_tokens_f32: Vec<Vec<f32>> = Vec::new();
                for t in 0..seq_len as usize {
                    let mut k = Vec::with_capacity(kv_dim);
                    let mut v = Vec::with_capacity(kv_dim);
                    for i in 0..kv_dim {
                        k.push(((t * kv_dim + i) as f32) * 0.011 - 0.7);
                        v.push(((t * kv_dim + (kv_dim - 1 - i)) as f32) * 0.017 - 1.3);
                    }
                    let k_cast = cast_f32_to_f16_then_back(&k);
                    let v_cast = cast_f32_to_f16_then_back(&v);
                    k_tokens_f32.push(k_cast.clone());
                    v_tokens_f32.push(v_cast.clone());

                    let bytes = kv_dim * std::mem::size_of::<f32>();
                    let d_k = ctx.device_malloc(bytes)?;
                    let d_v = ctx.device_malloc(bytes)?;
                    unsafe {
                        ctx.memcpy_h2d(d_k, k_cast.as_ptr() as *const c_void, bytes)?;
                        ctx.memcpy_h2d(d_v, v_cast.as_ptr() as *const c_void, bytes)?;
                        kv.append_token_f32(&ctx, 0, d_k as *const c_void, d_v as *const c_void)?;
                        ctx.device_free(d_k)?;
                        ctx.device_free(d_v)?;
                    }
                }

                let mut q = Vec::with_capacity(q_dim);
                for i in 0..q_dim {
                    q.push((i as f32) * 0.004 - 0.25);
                }
                let bytes_f32 = q_dim * std::mem::size_of::<f32>();
                let d_q = ctx.device_malloc(bytes_f32)?;
                let d_out = ctx.device_malloc(bytes_f32)?;
                unsafe {
                    ctx.memcpy_h2d(d_q, q.as_ptr() as *const c_void, bytes_f32)?;
                    kv.attention_last_token_f32_gqa(
                        &ctx,
                        0,
                        d_q as *const c_void,
                        q_heads,
                        seq_len,
                        d_out,
                    )?;
                }
                let mut out_gpu = vec![0f32; q_dim];
                unsafe {
                    ctx.memcpy_d2h(
                        out_gpu.as_mut_ptr() as *mut c_void,
                        d_out as *const c_void,
                        bytes_f32,
                    )?;
                    ctx.device_free(d_q)?;
                    ctx.device_free(d_out)?;
                }

                let out_cpu = cpu_last_token_attention_gqa(
                    &q,
                    &k_tokens_f32,
                    &v_tokens_f32,
                    q_heads as usize,
                    kv_heads as usize,
                    head_dim as usize,
                );

                for i in 0..q_dim {
                    let a = out_cpu[i];
                    let b = out_gpu[i];
                    let diff = (a - b).abs();
                    assert!(
                        diff < 1e-3,
                        "GQA mismatch (q_heads={},kv_heads={},dim={},seq_len={},i={}) cpu={} gpu={} diff={}",
                        q_heads,
                        kv_heads,
                        head_dim,
                        seq_len,
                        i,
                        a,
                        b,
                        diff
                    );
                }
            }
        }
    }

    Ok(())
}

#[test]
fn attention_last_token_cuda_gqa_head64_long_parity() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let q_heads = 32u32;
    let kv_heads = 4u32;
    let head_dim = 64u32;
    let seq_lens = [128u32, 512];
    let max_seq_len = *seq_lens.iter().max().unwrap();
    let q_dim = (q_heads * head_dim) as usize;
    let kv_dim = (kv_heads * head_dim) as usize;

    for &seq_len in &seq_lens {
        let kv = KVCache::new_with_context(&ctx, max_seq_len, 1, kv_heads, head_dim)?;
        let mut k_tokens_f32: Vec<Vec<f32>> = Vec::new();
        let mut v_tokens_f32: Vec<Vec<f32>> = Vec::new();

        let bytes = kv_dim * std::mem::size_of::<f32>();
        let d_k = ctx.device_malloc(bytes)?;
        let d_v = ctx.device_malloc(bytes)?;
        for t in 0..seq_len as usize {
            let mut k = Vec::with_capacity(kv_dim);
            let mut v = Vec::with_capacity(kv_dim);
            for i in 0..kv_dim {
                k.push(((t * kv_dim + i) as f32) * 0.0003 - 0.25);
                v.push(((t * kv_dim + (kv_dim - 1 - i)) as f32) * 0.0002 - 0.1);
            }
            let k_cast = cast_f32_to_f16_then_back(&k);
            let v_cast = cast_f32_to_f16_then_back(&v);
            k_tokens_f32.push(k_cast.clone());
            v_tokens_f32.push(v_cast.clone());

            unsafe {
                ctx.memcpy_h2d(d_k, k_cast.as_ptr() as *const c_void, bytes)?;
                ctx.memcpy_h2d(d_v, v_cast.as_ptr() as *const c_void, bytes)?;
                kv.append_token_f32(&ctx, 0, d_k as *const c_void, d_v as *const c_void)?;
            }
        }

        let q: Vec<f32> = (0..q_dim).map(|i| (i as f32) * 0.0004 - 0.2).collect();
        let bytes_q = q_dim * std::mem::size_of::<f32>();
        let d_q = ctx.device_malloc(bytes_q)?;
        let d_out = ctx.device_malloc(bytes_q)?;
        unsafe {
            ctx.memcpy_h2d(d_q, q.as_ptr() as *const c_void, bytes_q)?;
            kv.attention_last_token_f32_gqa(
                &ctx,
                0,
                d_q as *const c_void,
                q_heads,
                seq_len,
                d_out,
            )?;
        }
        let mut out_gpu = vec![0f32; q_dim];
        unsafe {
            ctx.memcpy_d2h(
                out_gpu.as_mut_ptr() as *mut c_void,
                d_out as *const c_void,
                bytes_q,
            )?;
            ctx.device_free(d_q)?;
            ctx.device_free(d_out)?;
            ctx.device_free(d_k)?;
            ctx.device_free(d_v)?;
        }

        let out_cpu = cpu_last_token_attention_gqa(
            &q,
            &k_tokens_f32,
            &v_tokens_f32,
            q_heads as usize,
            kv_heads as usize,
            head_dim as usize,
        );

        for i in 0..q_dim {
            let diff = (out_cpu[i] - out_gpu[i]).abs();
            assert!(
                diff < 1e-3,
                "GQA head64 mismatch (seq_len={}, i={}) cpu={} gpu={} diff={}",
                seq_len,
                i,
                out_cpu[i],
                out_gpu[i],
                diff
            );
        }
    }

    Ok(())
}
