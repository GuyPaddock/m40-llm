use super::{
    capture::LayerSample,
    pipeline::{Arm, Case},
    reference::{self, Backing},
};
use anyhow::{ensure, Result};
use half::f16;
use m40_llm::{cuda::CudaContext, evaluation::identity::sha256};
use serde_json::{json, Value};

#[derive(Default)]
struct Errors {
    count: usize,
    maximum: f64,
    squared: f64,
}
impl Errors {
    fn observe(&mut self, actual: f64, expected: f64) -> Result<()> {
        ensure!(
            actual.is_finite() && expected.is_finite(),
            "non-finite attention output"
        );
        let error = (actual - expected).abs();
        self.count += 1;
        self.maximum = self.maximum.max(error);
        self.squared += error * error;
        Ok(())
    }
    fn report(&self) -> Value {
        json!({"elements":self.count,"max_absolute_error":self.maximum,
        "rms_error":(self.squared/self.count.max(1) as f64).sqrt()})
    }
    fn require_parity(&self, label: &str) -> Result<()> {
        ensure!(
            self.count > 0 && self.maximum < 1e-3,
            "{label} differs from reference: {}",
            self.maximum
        );
        Ok(())
    }
}

pub fn validate(case: &Case, sample: &LayerSample) -> Result<Value> {
    let s = case.shape;
    let before = CudaContext::total_device_bytes();
    let staged = case.read_arm(Arm::Staged, case.sequences)?;
    let direct = case.read_arm(Arm::Direct, case.sequences)?;
    let dense = case.read_arm(Arm::Dense, case.sequences)?;
    let mut staged_error = Errors::default();
    let mut direct_error = Errors::default();
    let mut dense_error = Errors::default();
    let mut pair_error = Errors::default();
    let mut selection_error = Errors::default();
    let mut quantization_error = Errors::default();
    let mut selected_blocks = Vec::new();
    let mut position_bytes = Vec::new();
    let mut selected_old_counts = Vec::new();
    let mut append_zero_sign_changes = 0usize;
    for seq in 0..case.sequences {
        let appended_keys = reference::zero_frequency_append(&sample.keys[seq])?;
        append_zero_sign_changes += appended_keys
            .iter()
            .zip(&sample.keys[seq])
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let keys = &appended_keys;
        let values = &sample.values[seq];
        let query = &sample.queries[seq];
        let reconstructed = reference::reconstruct(s, case.backing, keys, values)?;
        let readback = case.read_staging(seq)?;
        let (staged_keys, staged_values, positions, counts) = (
            readback.keys,
            readback.values,
            readback.positions,
            readback.counts,
        );
        let mut sequence_blocks = Vec::new();
        let mut sequence_counts = Vec::new();
        for (head, &count) in counts.iter().enumerate() {
            let selected =
                reference::select_positions(s, &reconstructed.selection_keys, query, head)?;
            ensure!(
                count as usize == selected.len(),
                "selected token count differs sequence={seq} head={head}"
            );
            let actual = &positions[head * s.capacity()..head * s.capacity() + selected.len()];
            ensure!(
                actual.iter().zip(&selected).all(|(&a, &b)| a as usize == b),
                "selected positions differ sequence={seq} head={head}"
            );
            for value in actual {
                position_bytes.extend_from_slice(&value.to_le_bytes());
            }
            sequence_blocks.push(
                selected
                    .iter()
                    .filter(|&&t| t < s.old_len() && t.is_multiple_of(s.block))
                    .map(|t| t / s.block)
                    .collect::<Vec<_>>(),
            );
            sequence_counts.push(selected.iter().filter(|&&t| t < s.old_len()).count());
            let kv_head = head / (s.query_heads / s.kv_heads);
            for (index, &token) in selected.iter().enumerate() {
                for channel in 0..s.head_dim {
                    let source = token * s.kv_width() + kv_head * s.head_dim + channel;
                    let target = (head * s.capacity() + index) * s.head_dim + channel;
                    ensure!(staged_keys[target]==f16::from_f32(reconstructed.keys[source]).to_bits(),
                        "staged key differs sequence={seq} head={head} token={token} channel={channel}");
                    ensure!(staged_values[target]==f16::from_f32(reconstructed.values[source]).to_bits(),
                        "staged value differs sequence={seq} head={head} token={token} channel={channel}");
                }
            }
            let expected = reference::attention(
                s,
                &reconstructed.keys,
                &reconstructed.values,
                query,
                head,
                &selected,
            )?;
            let selected_original = reference::attention(s, keys, values, query, head, &selected)?;
            let full = reference::attention(
                s,
                keys,
                values,
                query,
                head,
                &(0..s.length).collect::<Vec<_>>(),
            )?;
            for channel in 0..s.head_dim {
                let index = seq * s.query_width() + head * s.head_dim + channel;
                staged_error.observe(f64::from(staged[index]), expected[channel])?;
                direct_error.observe(f64::from(direct[index]), expected[channel])?;
                dense_error.observe(f64::from(dense[index]), full[channel])?;
                pair_error.observe(f64::from(staged[index]), f64::from(direct[index]))?;
                selection_error.observe(selected_original[channel], full[channel])?;
                quantization_error.observe(expected[channel], selected_original[channel])?;
            }
        }
        selected_blocks.push(sequence_blocks);
        selected_old_counts.push(sequence_counts);
    }
    for (name, error) in [
        ("staged FP64", &staged_error),
        ("direct FP64", &direct_error),
        ("dense FP64", &dense_error),
        ("staged/direct", &pair_error),
    ] {
        error.require_parity(name)?;
    }
    let mut batched_error = Errors::default();
    for batch in [1, 2, 4, 8].into_iter().filter(|&b| b <= case.sequences) {
        for (arm, control) in [
            (Arm::Staged, &staged),
            (Arm::Direct, &direct),
            (Arm::Dense, &dense),
        ] {
            let repeated = case.read_arm(arm, batch)?;
            ensure!(
                repeated
                    .iter()
                    .zip(control)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "repeated serial dispatch changed output"
            );
        }
        if case.backing == Backing::Mixed {
            let batched = case.read_arm(Arm::BatchedDirect, batch)?;
            for (&actual, &expected) in batched.iter().zip(&direct) {
                batched_error.observe(f64::from(actual), f64::from(expected))?;
            }
        }
    }
    if case.backing == Backing::Mixed {
        batched_error.require_parity("batched direct")?;
    }
    ensure!(
        CudaContext::total_device_bytes() == before,
        "attention validation changed tracked allocation"
    );
    Ok(
        json!({"is_valid":true,"selected_blocks":selected_blocks,"selected_old_counts":selected_old_counts,
        "selected_positions_sha256":sha256(&position_bytes)?,"staged_values_bit_exact_after_identity_append":true,
        "identity_append_key_zero_sign_changes":append_zero_sign_changes,
        "staged_vs_fp64":staged_error.report(),"direct_vs_fp64":direct_error.report(),"dense_vs_fp64":dense_error.report(),
        "staged_vs_direct":pair_error.report(),"batched_vs_serial_direct":if case.backing==Backing::Mixed {batched_error.report()} else {Value::Null},
        "selection_only_vs_full_fp16":selection_error.report(),"quantization_only_at_fixed_positions":quantization_error.report(),
        "output_tolerance_absolute":1e-3,"query_and_output_guards_intact":true,
        "inactive_batch_rows_unchanged":true,"repeated_serial_outputs_bit_exact":true,
        "scope":"operator numerical validation, not model logits or generation quality"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::Shape;

    /// Exercise staged/direct/dense and mixed batched consumers on eight
    /// distinct sequences, including short recent-only and wrapped old caches.
    /// All real GPU calls are serial and allocation owners must clean up.
    #[test]
    #[ignore = "requires serial M40 hardware"]
    fn test_pipeline_preserves_attention_and_ownership() -> Result<()> {
        crate::benchmark::validate_environment()?;
        let ctx = CudaContext::new_m40()?;
        let props = ctx.current_device_props()?;
        ensure!(props.major == 5 && props.minor == 2, "requires sm_52");
        let before = CudaContext::total_device_bytes();
        for (head_dim, backing) in [
            (64, Backing::Q8),
            (64, Backing::Mixed),
            (128, Backing::Mixed),
        ] {
            for length in [5, 23] {
                let shape = Shape {
                    length,
                    head_dim,
                    query_heads: 4,
                    kv_heads: 2,
                    recent: 5,
                    block: 4,
                    top: 2,
                };
                let mut sample = LayerSample {
                    keys: Vec::new(),
                    values: Vec::new(),
                    queries: Vec::new(),
                };
                for seq in 0..8 {
                    sample.keys.push(
                        (0..length)
                            .flat_map(|token| {
                                (0..shape.kv_width()).map(move |index| {
                                    f16::from_f32(
                                        0.01 + (token / 4) as f32 * 0.07
                                            + (index % 11) as f32 * 0.0003
                                            + seq as f32 * 0.003,
                                    )
                                    .to_f32()
                                })
                            })
                            .collect(),
                    );
                    sample.values.push(
                        (0..length * shape.kv_width())
                            .map(|i| {
                                f16::from_f32(((i * 71 + seq * 29) % 997) as f32 * 0.0017 - 0.8)
                                    .to_f32()
                            })
                            .collect(),
                    );
                    sample.queries.push(
                        (0..shape.query_width())
                            .map(|i| {
                                if (i / head_dim).is_multiple_of(2) {
                                    0.03 + (i % 7) as f32 * 0.001
                                } else {
                                    -0.04 - (i % 7) as f32 * 0.001
                                }
                            })
                            .collect(),
                    );
                }
                let case = Case::new(&ctx, shape, backing, &sample)?;
                validate(&case, &sample)?;
                drop(case);
                ensure!(
                    CudaContext::total_device_bytes() == before,
                    "case leaked tracked buffers"
                );
            }
        }
        Ok(())
    }
}
