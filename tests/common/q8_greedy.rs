use anyhow::{ensure, Context, Result};
use m40_llm::{cuda::DeviceBuffer, infer::LoadedModel, profile};

/// Compare device-only greedy selection with the full-logits reference using
/// reusable scratch, and prove only one token crosses D2H during selection.
pub fn compare_greedy_with_logits(model: &LoadedModel, hidden: &[f32]) -> Result<(u32, Vec<f32>)> {
    let (_, _, d_model, vocab, _) = model.map_lm_head()?;
    ensure!(hidden.len() == d_model, "hidden row size mismatch");
    let d_hidden = DeviceBuffer::new_tagged(&model.cuda, d_model * 4, "greedy:hidden")?;
    let d_logits = DeviceBuffer::new_tagged(&model.cuda, vocab * 4, "greedy:logits")?;
    let d_token = DeviceBuffer::new_tagged(&model.cuda, 4, "greedy:token")?;
    let has_norm = model.map_output_norm()?.is_some();
    let d_norm = if has_norm {
        Some(DeviceBuffer::new_tagged(
            &model.cuda,
            d_model * 4,
            "greedy:norm",
        )?)
    } else {
        None
    };
    let norm = d_norm.as_ref().map(DeviceBuffer::as_mut_ptr);
    unsafe {
        model.cuda.memcpy_h2d(
            d_hidden.as_mut_ptr(),
            hidden.as_ptr().cast(),
            d_hidden.bytes(),
        )?;
    }
    let logits =
        unsafe { model.logits_from_hidden_into(d_hidden.as_ptr(), d_logits.as_mut_ptr(), norm)? };
    ensure!(
        logits.iter().all(|v| v.is_finite()),
        "reference logits must be finite"
    );
    let expected = logits
        .iter()
        .enumerate()
        .fold(None, |best, (index, &value)| match best {
            Some((_, old)) if old >= value => best,
            _ => Some((index as u32, value)),
        })
        .context("empty vocabulary")?
        .0;
    let before = profile::snapshot();
    let actual = unsafe {
        model.greedy_token_from_hidden_into(
            d_hidden.as_ptr(),
            d_logits.as_mut_ptr(),
            norm,
            d_token.as_mut_ptr(),
        )?
    };
    let after = profile::snapshot();
    let delta = profile::counter_delta(&before, &after);
    assert_eq!(
        actual, expected,
        "greedy token differs from full-logits argmax"
    );
    assert_eq!(delta.device_allocs, 0);
    assert_eq!(delta.device_frees, 0);
    assert_eq!(delta.h2d_copies, 0);
    assert_eq!(delta.d2h_copies, 1);
    assert_eq!(
        delta.d2h_bytes, 4,
        "greedy selection must not copy full logits"
    );
    if model.q8_execution_stats().has_prepared_vocabulary {
        assert_eq!(delta.launches, 2 + u64::from(has_norm));
        for op in ["gemm_f32xq8_separate_f32", "argmax_f32"] {
            assert_eq!(
                after.by_op.get(op).map_or(0, |v| v.launches)
                    - before.by_op.get(op).map_or(0, |v| v.launches),
                1,
                "missing {op}"
            );
        }
    }
    Ok((actual, logits))
}
