#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;

use anyhow::Result;
use m40_llm::cuda::CudaContext;
use m40_llm::gguf::{GgmlDType, GgufModel, GgufScalar, GgufTensor, GgufValue};
use m40_llm::infer::LoadedModel;
use std::collections::HashMap;
use std::ffi::c_void;

fn make_halves_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        let bits = half::f16::from_f32(v).to_bits();
        out.push((bits & 0xFF) as u8);
        out.push((bits >> 8) as u8);
    }
    out
}

fn make_f32_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn minimal_metadata() -> HashMap<String, GgufValue> {
    use GgufScalar as S;
    let mut m = HashMap::new();
    m.insert(
        "general.architecture".into(),
        GgufValue::Scalar(S::Str("llama".into())),
    );
    m.insert(
        "llama.embedding_length".into(),
        GgufValue::Scalar(S::U32(4)),
    );
    m.insert(
        "llama.attention.head_count".into(),
        GgufValue::Scalar(S::U32(1)),
    );
    m.insert("llama.block_count".into(), GgufValue::Scalar(S::U32(1)));
    m.insert("llama.context_length".into(), GgufValue::Scalar(S::U32(16)));
    m.insert(
        "llama.feed_forward_length".into(),
        GgufValue::Scalar(S::U32(8)),
    );
    m.insert("llama.vocab_size".into(), GgufValue::Scalar(S::U32(32)));
    m
}

#[test]
fn gguf_device_views_cuda_dptr_and_bytes_match() -> Result<()> {
    // Ensure device weights are enabled
    std::env::set_var("M40LLM_ENABLE_NVCC", "1");

    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    // Build an in-memory GGUF model with two tensors at known offsets
    let mut gg = GgufModel::new(0);
    gg.metadata = minimal_metadata();
    let a_shape = vec![2u64, 3u64]; // 2x3 f16 => 12 bytes
    let b_shape = vec![2u64, 2u64]; // 2x2 f32 => 16 bytes
    let a_bytes = make_halves_bytes(&[0.1, -0.2, 0.3, -0.4, 0.5, -0.6]);
    let b_bytes = make_f32_bytes(&[1.25, -2.5, 3.75, -4.5]);

    // Place A at 0, B at offset with padding
    let a_off = 0u64;
    let pad = 32usize; // leave some space after A
    let b_off = (a_off as usize + a_bytes.len() + pad) as u64;

    gg.tensors.push(GgufTensor {
        name: "A".into(),
        dtype: GgmlDType::F16,
        shape: a_shape.clone(),
        offset: a_off,
    });
    gg.tensors.push(GgufTensor {
        name: "B".into(),
        dtype: GgmlDType::F32,
        shape: b_shape.clone(),
        offset: b_off,
    });

    // Compose weights blob with zeros, then write our payloads at the offsets
    let total_len = (b_off as usize) + b_bytes.len();
    let mut weights = vec![0u8; total_len];
    weights[(a_off as usize)..(a_off as usize + a_bytes.len())].copy_from_slice(&a_bytes);
    weights[(b_off as usize)..(b_off as usize + b_bytes.len())].copy_from_slice(&b_bytes);

    // Ensure device weights are enabled
    std::env::set_var("M40LLM_ENABLE_NVCC", "1");
    // Load model
    let lm = LoadedModel::from_gguf(gg, weights.clone(), -1)?;

    // Validate base pointer is non-null and CUDA-valid
    assert!(!lm.d_weights_base.is_null());
    ctx.validate_device_ptr(lm.d_weights_base)
        .expect("valid CUDA base pointer");

    // Check each tensor view's device pointer equals base + byte_offset
    for name in ["A", "B"] {
        let tv = lm.device_tensors.get(name).expect("tensor view present");
        let expect_ptr = (lm.d_weights_base as usize + tv.byte_offset as usize) as *const c_void;

        // Explicit CUDA pointer validation
        ctx.validate_device_ptr(tv.dptr as *const c_void)
            .unwrap_or_else(|e| panic!("invalid CUDA pointer for {}: {}", name, e));
        assert_eq!(
            tv.dptr as *const c_void as usize, expect_ptr as usize,
            "{} dptr mismatch",
            name
        );

        // Copy back first few bytes and compare to source slice
        let check_len = tv.nbytes.min(16);
        if check_len > 0 {
            let mut host = vec![0u8; check_len];
            unsafe {
                ctx.memcpy_d2h(
                    host.as_mut_ptr() as *mut c_void,
                    tv.dptr as *const c_void,
                    check_len,
                )?;
            }
            let src = &weights[(tv.byte_offset as usize)..(tv.byte_offset as usize + check_len)];
            assert_eq!(&host[..], src, "{} bytes mismatch", name);
        }
    }

    Ok(())
}
