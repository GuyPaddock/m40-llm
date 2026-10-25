#![cfg(all(feature = "cuda", nvcc))]

use anyhow::{ensure, Result};
use m40_llm::{
    cuda::{CudaContext, DeviceBuffer},
    profile,
};

/// A rejected impossible allocation must return device context without locking
/// twice, preserve live buffers/accounting, and permit subsequent allocations.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_allocation_error_preserves_allocator_and_live_buffers() -> Result<()> {
    let ctx = CudaContext::new(-1)?;
    let props = ctx.current_device_props()?;
    ensure!(
        props.major == 5 && props.minor == 2,
        "allocation test requires sm_52"
    );
    let initial_bytes = CudaContext::total_device_bytes();
    let live = DeviceBuffer::new_tagged(&ctx, 16, "allocation_error:live")?;
    unsafe {
        ctx.memcpy_h2d(live.as_mut_ptr(), [1_u32, 2, 3, 4].as_ptr().cast(), 16)?;
    }
    let before_failure = profile::snapshot();
    let bytes_before_failure = CudaContext::total_device_bytes();
    let error = match ctx.device_malloc_tagged(usize::MAX, "allocation_error:impossible") {
        Err(error) => error,
        Ok(pointer) => {
            unsafe {
                ctx.device_free(pointer)?;
            }
            anyhow::bail!("impossible allocation unexpectedly succeeded");
        }
    };
    let message = error.to_string();
    assert!(
        message.contains(&format!("bytes={}", usize::MAX)),
        "{message}"
    );
    assert!(
        message.contains(&props.name) && message.contains("sm_52"),
        "{message}"
    );
    assert_eq!(CudaContext::total_device_bytes(), bytes_before_failure);
    let delta = profile::counter_delta(&before_failure, &profile::snapshot());
    assert_eq!(delta.device_allocs, 0);
    assert_eq!(delta.device_frees, 0);
    let subsequent = DeviceBuffer::new_tagged(&ctx, 16, "allocation_error:subsequent")?;
    let mut output = [0_u32; 4];
    unsafe {
        ctx.memcpy_d2h(output.as_mut_ptr().cast(), live.as_ptr(), 16)?;
        assert_eq!(output, [1, 2, 3, 4]);
        ctx.memcpy_h2d(subsequent.as_mut_ptr(), output.as_ptr().cast(), 16)?;
        output.fill(0);
        ctx.memcpy_d2h(output.as_mut_ptr().cast(), subsequent.as_ptr(), 16)?;
    }
    assert_eq!(output, [1, 2, 3, 4]);
    drop(live);
    drop(subsequent);
    assert_eq!(CudaContext::total_device_bytes(), initial_bytes);
    Ok(())
}

/// Uploaded weights survive intermediate context clones and are freed/accounted
/// exactly once when the final owner drops, including across successive models.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_uploaded_weights_drop_restores_accounting_after_last_owner() -> Result<()> {
    let initial = CudaContext::total_device_bytes();
    for bytes in [16, 32] {
        let before = profile::snapshot();
        let ctx = CudaContext::new(-1)?;
        let props = ctx.current_device_props()?;
        ensure!(
            props.major == 5 && props.minor == 2,
            "weight cleanup test requires sm_52"
        );
        let expected = vec![7_u8; bytes];
        let pointer = ctx.upload_weights(&expected)?;
        let owner = ctx.clone();
        drop(ctx);
        assert_eq!(CudaContext::total_device_bytes(), initial + bytes);
        let mut actual = vec![0_u8; bytes];
        unsafe {
            owner.memcpy_d2h(actual.as_mut_ptr().cast(), pointer, bytes)?;
        }
        assert_eq!(actual, expected);
        assert_eq!(
            profile::counter_delta(&before, &profile::snapshot()).device_frees,
            0
        );
        drop(owner);
        assert_eq!(CudaContext::total_device_bytes(), initial);
        let delta = profile::counter_delta(&before, &profile::snapshot());
        assert_eq!(delta.device_allocs, 1);
        assert_eq!(delta.device_frees, 1);
    }
    Ok(())
}
