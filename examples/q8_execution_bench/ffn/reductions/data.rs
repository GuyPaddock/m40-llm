use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{CudaContext, DeviceBuffer},
    evaluation::identity::sha256,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::ffi::c_void;

pub(super) const DISTRIBUTIONS: [&str; 3] = ["random", "ascending", "late-tie"];
pub(super) const LOGIT_SEED: u64 = 0x5245_4455_4354;

/// Synthetic finite logits, never represented as captured model logits.
pub(super) fn make_logits(dim: usize, rows: usize, distribution: &str) -> Result<Vec<f32>> {
    ensure!(
        dim >= 3 && matches!(rows, 1 | 4),
        "invalid synthetic logit shape"
    );
    ensure!(
        DISTRIBUTIONS.contains(&distribution),
        "unknown logit distribution"
    );
    let mut rng = StdRng::seed_from_u64(LOGIT_SEED);
    let mut result = Vec::with_capacity(dim.checked_mul(rows).context("logit size overflow")?);
    for row in 0..rows {
        let start = result.len();
        for index in 0..dim {
            result.push(if distribution == "ascending" {
                index as f32 * 0.125 + row as f32 * 0.03125
            } else {
                rng.gen_range(-20.0f32..20.0)
            });
        }
        if distribution == "late-tie" {
            result[start + dim / 3] = 30.0 + row as f32;
            result[start + dim - 2] = 30.0 + row as f32;
        }
    }
    Ok(result)
}

pub(super) fn hash_f32(values: &[f32]) -> Result<String> {
    sha256(
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

/// U32 payload guards also protect mixed float/index argmax scratch bits.
pub(super) struct GuardedWords {
    device: DeviceBuffer,
    words: usize,
}

impl GuardedWords {
    pub fn new(ctx: &CudaContext, words: usize) -> Result<Self> {
        let count = words.checked_add(16).context("word guard overflow")?;
        let device = DeviceBuffer::new_tagged(
            ctx,
            count.checked_mul(4).context("word byte overflow")?,
            "reduction_probe:words",
        )?;
        let initial = vec![0xa512_3456u32; count];
        unsafe {
            ctx.memcpy_h2d(device.as_mut_ptr(), initial.as_ptr().cast(), device.bytes())?;
        }
        Ok(Self { device, words })
    }

    pub fn pointer(&self) -> *mut c_void {
        unsafe { self.device.as_mut_ptr().cast::<u32>().add(8).cast() }
    }

    pub fn bytes(&self) -> usize {
        self.device.bytes()
    }

    pub fn read(&self, ctx: &CudaContext) -> Result<Vec<u32>> {
        let mut actual = vec![0u32; self.words + 16];
        unsafe {
            ctx.memcpy_d2h(
                actual.as_mut_ptr().cast(),
                self.device.as_ptr(),
                self.device.bytes(),
            )?;
        }
        ensure!(
            actual[..8]
                .iter()
                .chain(&actual[8 + self.words..])
                .all(|&word| word == 0xa512_3456),
            "argmax scratch/output guard overwritten"
        );
        Ok(actual[8..8 + self.words].to_vec())
    }
}
