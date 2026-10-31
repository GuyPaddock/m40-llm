#![cfg(any(not(feature = "cuda"), nvcc))]

#[cfg(feature = "cuda")]
#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;

use anyhow::Result;
use m40_llm::cuda::CudaContext;

/// A non-CUDA context has no weight owner or device allocation, even when
/// host data is offered to the stub upload path.
#[cfg(not(feature = "cuda"))]
#[test]
fn test_non_cuda_context_reports_no_uploaded_weight_allocation() -> Result<()> {
    let context = CudaContext::new(-1)?;
    assert_eq!(context.uploaded_weights_bytes()?, 0);
    assert!(context.upload_weights(&[1, 2, 3])?.is_null());
    assert_eq!(context.uploaded_weights_bytes()?, 0);
    Ok(())
}

/// Query the actual successful weight owner, distinguish scratch, and replace
/// rather than accumulate upload sizes. Readback and final cleanup prove the
/// recorded allocation exists and has the claimed contents/lifetime.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_uploaded_weight_allocation_tracks_owned_replacement_and_cleanup() -> Result<()> {
    let _scope = cuda_test_scope::lock_cuda();
    let initial = CudaContext::total_device_bytes();
    {
        let context = CudaContext::new_m40()?;
        assert_eq!(context.uploaded_weights_bytes()?, 0);
        let scratch =
            m40_llm::cuda::DeviceBuffer::new_tagged(&context, 128, "weight-accounting:scratch")?;
        for length in [64usize, 160] {
            let bytes: Vec<u8> = (0..length).map(|index| index as u8).collect();
            let pointer = context.upload_weights(&bytes)?;
            assert!(!pointer.is_null());
            assert_eq!(context.uploaded_weights_bytes()?, length);
            assert_eq!(
                CudaContext::total_device_bytes() - initial,
                length + scratch.bytes()
            );
            let mut actual = vec![0; length];
            unsafe {
                context.memcpy_d2h(actual.as_mut_ptr().cast(), pointer, length)?;
            }
            assert_eq!(actual, bytes);
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}
