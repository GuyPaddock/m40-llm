use anyhow::Result;
use m40_llm::cuda::{CudaContext, DeviceBuffer};
use std::ffi::c_void;

/// Device payload with independently checked prefix/suffix bytes. The offset
/// permits tests to exercise legal two-byte-only FP16 weight alignment.
pub struct GuardedBytes {
    device: DeviceBuffer,
    initial: Vec<u8>,
    offset: usize,
    len: usize,
}

impl GuardedBytes {
    pub fn new(ctx: &CudaContext, data: &[u8], offset: usize) -> Result<Self> {
        let mut initial = vec![0xa5; offset + data.len() + 32];
        initial[offset..offset + data.len()].copy_from_slice(data);
        let device = DeviceBuffer::new_tagged(ctx, initial.len(), "decode_reduction:test")?;
        unsafe {
            ctx.memcpy_h2d(device.as_mut_ptr(), initial.as_ptr().cast(), initial.len())?;
        }
        Ok(Self {
            device,
            initial,
            offset,
            len: data.len(),
        })
    }

    pub fn pointer(&self) -> *mut c_void {
        unsafe {
            self.device
                .as_mut_ptr()
                .cast::<u8>()
                .add(self.offset)
                .cast()
        }
    }

    pub fn read(&self, ctx: &CudaContext) -> Result<Vec<u8>> {
        let mut actual = vec![0; self.initial.len()];
        unsafe {
            ctx.memcpy_d2h(
                actual.as_mut_ptr().cast(),
                self.device.as_ptr(),
                actual.len(),
            )?;
        }
        assert_eq!(&actual[..self.offset], &self.initial[..self.offset]);
        assert_eq!(
            &actual[self.offset + self.len..],
            &self.initial[self.offset + self.len..]
        );
        Ok(actual[self.offset..self.offset + self.len].to_vec())
    }
}
