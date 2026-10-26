use super::{ffi, record_async_kernel, CudaContext};
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;

/// The only two geometries in M04's independent numerical experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
#[repr(i32)]
pub enum Q8FfnQ81Launch {
    Threads128 = 128,
    Threads256 = 256,
}

impl Q8FfnQ81Launch {
    pub const ALL: [Self; 2] = [Self::Threads128, Self::Threads256];

    pub fn threads(self) -> i32 {
        self as i32
    }
}

/// M03's frozen set plus bounded follow-up experiments. None selects
/// a runtime policy; ALL deliberately retains the original five-variant set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
#[repr(i32)]
pub enum Q8FfnFp32Variant {
    ScalarShared256 = 0,
    PackedShared256 = 1,
    ScalarWarp256 = 2,
    PackedWarp256 = 3,
    PackedWarp128 = 4,
    PackedWarp128FactoredScales = 5,
    PackedWarp128CoalescedPairs = 6,
}

impl Q8FfnFp32Variant {
    pub const ALL: [Self; 5] = [
        Self::ScalarShared256,
        Self::PackedShared256,
        Self::ScalarWarp256,
        Self::PackedWarp256,
        Self::PackedWarp128,
    ];

    pub fn threads(self) -> u32 {
        if matches!(
            self,
            Self::PackedWarp128
                | Self::PackedWarp128FactoredScales
                | Self::PackedWarp128CoalescedPairs
        ) {
            128
        } else {
            256
        }
    }

    pub fn has_packed_loads(self) -> bool {
        matches!(
            self,
            Self::PackedShared256
                | Self::PackedWarp256
                | Self::PackedWarp128
                | Self::PackedWarp128FactoredScales
                | Self::PackedWarp128CoalescedPairs
        )
    }

    pub fn has_warp_reduction(self) -> bool {
        matches!(
            self,
            Self::ScalarWarp256
                | Self::PackedWarp256
                | Self::PackedWarp128
                | Self::PackedWarp128FactoredScales
                | Self::PackedWarp128CoalescedPairs
        )
    }

    pub fn has_factored_scales(self) -> bool {
        self == Self::PackedWarp128FactoredScales
    }

    pub fn has_coalesced_pairs(self) -> bool {
        self == Self::PackedWarp128CoalescedPairs
    }

    pub fn has_intermediate_capture(self) -> bool {
        self.has_factored_scales() || self.has_coalesced_pairs()
    }

    fn operation_name(self) -> &'static str {
        match self {
            Self::ScalarShared256 => "ffn_q8_f32_scalar_shared_256",
            Self::PackedShared256 => "ffn_q8_f32_packed_shared_256",
            Self::ScalarWarp256 => "ffn_q8_f32_scalar_warp_256",
            Self::PackedWarp256 => "ffn_q8_f32_packed_warp_256",
            Self::PackedWarp128 => "ffn_q8_f32_packed_warp_128",
            Self::PackedWarp128FactoredScales => "ffn_q8_f32_factored_scales_128",
            Self::PackedWarp128CoalescedPairs => "ffn_q8_f32_coalesced_pairs_128",
        }
    }
}

#[derive(Clone, Copy)]
enum SplitFfnKernel {
    RoleParallel,
    NativeWarp,
    EightValues,
    Cohorts,
    PersistentTiles,
    VirtualWarp,
    Dynamic,
    TwoColumns,
    FixedK2048,
    Unfused,
    WeightCg,
    BytePermute,
}

impl CudaContext {
    /// Separate physical gate/up roles with original 128-partial FP32 trees.
    ///
    /// # Safety
    /// Same spans, alignment and stream lifetime as `ffn_q8_split_f32_async`.
    /// No output/capture span may overlap input, weights or another output.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ffn_q8_split_role_parallel_f32_async(
        &self,
        input: *const c_void,
        scales: [*const c_void; 2],
        values: [*const c_void; 2],
        output: *mut c_void,
        rows: i32,
        hidden: i32,
        k: i32,
        intermediates: Option<(*mut c_void, *mut c_void)>,
    ) -> Result<()> {
        self.enqueue_split_f32(
            input,
            scales,
            values,
            output,
            rows,
            hidden,
            k,
            intermediates,
            SplitFfnKernel::RoleParallel,
        )
    }

    /// A native 32-partial FP32 tree per output warp. Summation association
    /// differs from the qualified 128-thread control; weights stay unchanged.
    ///
    /// # Safety
    /// Same spans, alignment and stream lifetime as `ffn_q8_split_f32_async`.
    /// No output/capture span may overlap input, weights or another output.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ffn_q8_split_native_warp_f32_async(
        &self,
        input: *const c_void,
        scales: [*const c_void; 2],
        values: [*const c_void; 2],
        output: *mut c_void,
        rows: i32,
        hidden: i32,
        k: i32,
        intermediates: Option<(*mut c_void, *mut c_void)>,
    ) -> Result<()> {
        self.enqueue_split_f32(
            input,
            scales,
            values,
            output,
            rows,
            hidden,
            k,
            intermediates,
            SplitFfnKernel::NativeWarp,
        )
    }

    /// Enqueue the fixed one-column joint gate/up layout on decode.
    /// No allocation, transfer or synchronization occurs.
    ///
    /// # Safety
    /// All buffers belong to this context and survive decode completion, with
    /// producers ordered before decode. Input holds rows*K f32 at 16-byte
    /// alignment; outputs hold rows*hidden f32 at four-byte alignment. Scales
    /// hold hidden*(K/32) paired original u16 records at four-byte alignment;
    /// values hold hidden*(K/4) paired u32 records at eight-byte alignment.
    /// Every input, weight and output span is disjoint from output spans.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ffn_q8_joint_f32_async(
        &self,
        input: *const c_void,
        scales: *const c_void,
        values: *const c_void,
        output: *mut c_void,
        rows: i32,
        hidden: i32,
        k: i32,
        intermediates: Option<(*mut c_void, *mut c_void)>,
    ) -> Result<()> {
        ensure!(
            matches!(rows, 1 | 4) && hidden > 0 && k > 0 && k % 32 == 0,
            "joint FP32 FFN requires rows 1/4 and complete Q8 blocks"
        );
        ensure!(
            !input.is_null()
                && !output.is_null()
                && input != output.cast_const()
                && (input as usize).is_multiple_of(16)
                && (output as usize).is_multiple_of(4),
            "joint FP32 FFN requires disjoint input/output and 16-byte input alignment"
        );
        ensure!(
            !scales.is_null()
                && !values.is_null()
                && (scales as usize).is_multiple_of(4)
                && (values as usize).is_multiple_of(8),
            "joint FP32 FFN requires aligned paired scale/payload records"
        );
        let (g, u) = intermediates.unwrap_or((std::ptr::null_mut(), std::ptr::null_mut()));
        if intermediates.is_some() {
            ensure!(
                !g.is_null()
                    && !u.is_null()
                    && g != u
                    && g != output
                    && u != output
                    && g.cast_const() != input
                    && u.cast_const() != input
                    && (g as usize).is_multiple_of(4)
                    && (u as usize).is_multiple_of(4),
                "invalid joint FP32 FFN intermediate outputs"
            );
        }
        let _guard = self.inner.lock.lock().unwrap();
        let rc = ffi::m40llm_ffn_q8_joint_f32_async(
            self.inner.raw.as_ptr(),
            input,
            scales,
            values,
            output,
            g,
            u,
            rows,
            hidden,
            k,
        );
        ensure!(rc == 0, "m40llm_ffn_q8_joint_f32_async failed: {rc}");
        record_async_kernel("ffn_q8_joint_f32x4_128");
        Ok(())
    }

    /// Fixed 128-thread/four-value FP32 experiment on existing separate planes.
    /// Optional intermediates use the same kernel; no setup or sync occurs here.
    ///
    /// # Safety
    /// All spans belong to this context and survive decode-stream completion.
    /// Input holds rows*k FP32 values at 16-byte alignment. Each scale plane
    /// holds hidden*(k/32) FP16 values, each four-byte-aligned payload holds
    /// hidden*k signed bytes. Outputs hold rows*hidden FP32 values, disjoint
    /// from inputs/weights and one another. Complete all producer dependencies.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ffn_q8_split_f32_async(
        &self,
        input: *const c_void,
        scales: [*const c_void; 2],
        values: [*const c_void; 2],
        output: *mut c_void,
        rows: i32,
        hidden: i32,
        k: i32,
        intermediates: Option<(*mut c_void, *mut c_void)>,
    ) -> Result<()> {
        self.enqueue_split_f32(
            input,
            scales,
            values,
            output,
            rows,
            hidden,
            k,
            intermediates,
            SplitFfnKernel::Dynamic,
        )
    }

    /// Fixed two-column/128-thread experiment sharing FP32 activation loads.
    /// No additional weight planes, allocation, copies or synchronization.
    ///
    /// # Safety
    /// Same spans, alignment, disjointness and producer/lifetime contract as
    /// `ffn_q8_split_f32_async`, including optional gate/up outputs.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ffn_q8_split_f32_two_columns_async(
        &self,
        input: *const c_void,
        scales: [*const c_void; 2],
        values: [*const c_void; 2],
        output: *mut c_void,
        rows: i32,
        hidden: i32,
        k: i32,
        intermediates: Option<(*mut c_void, *mut c_void)>,
    ) -> Result<()> {
        self.enqueue_split_f32(
            input,
            scales,
            values,
            output,
            rows,
            hidden,
            k,
            intermediates,
            SplitFfnKernel::TwoColumns,
        )
    }

    /// Fixed K=2048 address/loop specialization of the one-column split kernel.
    /// Other widths fail explicitly; this does not select a generation policy.
    ///
    /// # Safety
    /// Same spans, alignment, disjointness and producer/lifetime contract as
    /// `ffn_q8_split_f32_async`, with K exactly 2048.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ffn_q8_split_f32_k2048_async(
        &self,
        input: *const c_void,
        scales: [*const c_void; 2],
        values: [*const c_void; 2],
        output: *mut c_void,
        rows: i32,
        hidden: i32,
        k: i32,
        intermediates: Option<(*mut c_void, *mut c_void)>,
    ) -> Result<()> {
        self.enqueue_split_f32(
            input,
            scales,
            values,
            output,
            rows,
            hidden,
            k,
            intermediates,
            SplitFfnKernel::FixedK2048,
        )
    }

    /// One fixed three-launch decomposition on existing split planes.
    /// No allocation, copy, synchronization or runtime selection is added.
    ///
    /// # Safety
    /// Same spans and producer/lifetime contract as `ffn_q8_split_f32_async`.
    /// Both intermediate buffers are required, hold rows*hidden FP32 values,
    /// and remain disjoint from output/input/weights through decode completion.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ffn_q8_split_unfused_f32_async(
        &self,
        input: *const c_void,
        scales: [*const c_void; 2],
        values: [*const c_void; 2],
        output: *mut c_void,
        rows: i32,
        hidden: i32,
        k: i32,
        intermediates: Option<(*mut c_void, *mut c_void)>,
    ) -> Result<()> {
        self.enqueue_split_f32(
            input,
            scales,
            values,
            output,
            rows,
            hidden,
            k,
            intermediates,
            SplitFfnKernel::Unfused,
        )
    }

    /// Fixed fused split-plane experiment with L2-only weight/scale loads.
    /// Activation loads, FP32 arithmetic, geometry and scratch are unchanged.
    ///
    /// # Safety
    /// The buffer ownership, sizes, alignment, lifetime and stream dependency
    /// requirements of `ffn_q8_split_f32_async` apply without modification.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ffn_q8_split_weight_cg_f32_async(
        &self,
        input: *const c_void,
        scales: [*const c_void; 2],
        values: [*const c_void; 2],
        output: *mut c_void,
        rows: i32,
        hidden: i32,
        k: i32,
        intermediates: Option<(*mut c_void, *mut c_void)>,
    ) -> Result<()> {
        self.enqueue_split_f32(
            input,
            scales,
            values,
            output,
            rows,
            hidden,
            k,
            intermediates,
            SplitFfnKernel::WeightCg,
        )
    }

    /// Fixed fused split-plane experiment with exact byte-permute conversion.
    /// FP32 inputs, scaling/accumulation order, geometry and scratch are fixed.
    ///
    /// # Safety
    /// The buffer ownership, sizes, alignment, lifetime and stream dependency
    /// requirements of `ffn_q8_split_f32_async` apply without modification.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ffn_q8_split_byte_permute_f32_async(
        &self,
        input: *const c_void,
        scales: [*const c_void; 2],
        values: [*const c_void; 2],
        output: *mut c_void,
        rows: i32,
        hidden: i32,
        k: i32,
        intermediates: Option<(*mut c_void, *mut c_void)>,
    ) -> Result<()> {
        self.enqueue_split_f32(
            input,
            scales,
            values,
            output,
            rows,
            hidden,
            k,
            intermediates,
            SplitFfnKernel::BytePermute,
        )
    }

    /// Register-owned virtual partials, with the qualified FP32 tree intact.
    /// This fixed four-output-warp experiment does not select runtime policy.
    ///
    /// # Safety
    /// The ownership, alignment, span and stream requirements of
    /// `ffn_q8_split_f32_async` apply without modification.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ffn_q8_split_virtual_warp_f32_async(
        &self,
        input: *const c_void,
        scales: [*const c_void; 2],
        values: [*const c_void; 2],
        output: *mut c_void,
        rows: i32,
        hidden: i32,
        k: i32,
        intermediates: Option<(*mut c_void, *mut c_void)>,
    ) -> Result<()> {
        self.enqueue_split_f32(
            input,
            scales,
            values,
            output,
            rows,
            hidden,
            k,
            intermediates,
            SplitFfnKernel::VirtualWarp,
        )
    }

    /// Eight-value FP32 tasks on the existing lossless planes, one output/CTA.
    /// Summation ownership changes; no activation quantization occurs.
    ///
    /// # Safety
    /// Same spans and stream lifetime as `ffn_q8_split_f32_async`, but both
    /// packed value pointers require eight-byte rather than four-byte alignment.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ffn_q8_split_f32x8_async(
        &self,
        input: *const c_void,
        scales: [*const c_void; 2],
        values: [*const c_void; 2],
        output: *mut c_void,
        rows: i32,
        hidden: i32,
        k: i32,
        intermediates: Option<(*mut c_void, *mut c_void)>,
    ) -> Result<()> {
        ensure!(
            values.iter().all(|p| (*p as usize).is_multiple_of(8)),
            "eight-value FFN requires eight-byte payload alignment"
        );
        self.enqueue_split_f32(
            input,
            scales,
            values,
            output,
            rows,
            hidden,
            k,
            intermediates,
            SplitFfnKernel::EightValues,
        )
    }

    /// Two independent physical 128-thread output groups in one 256-thread CTA.
    /// Benchmark-only; preserves the original per-output FP32 association.
    ///
    /// # Safety
    /// The ownership, alignment, span and stream requirements of
    /// `ffn_q8_split_f32_async` apply without modification.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ffn_q8_split_cohorts_f32_async(
        &self,
        input: *const c_void,
        scales: [*const c_void; 2],
        values: [*const c_void; 2],
        output: *mut c_void,
        rows: i32,
        hidden: i32,
        k: i32,
        intermediates: Option<(*mut c_void, *mut c_void)>,
    ) -> Result<()> {
        self.enqueue_split_f32(
            input,
            scales,
            values,
            output,
            rows,
            hidden,
            k,
            intermediates,
            SplitFfnKernel::Cohorts,
        )
    }

    /// Reuse at most 192 CTAs per row, with the original per-column FP32 tree.
    /// The mapping is benchmark-only and adds no device-wide synchronization.
    ///
    /// # Safety
    /// The ownership, alignment, span and stream requirements of
    /// `ffn_q8_split_f32_async` apply without modification.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ffn_q8_split_persistent_tiles_f32_async(
        &self,
        input: *const c_void,
        scales: [*const c_void; 2],
        values: [*const c_void; 2],
        output: *mut c_void,
        rows: i32,
        hidden: i32,
        k: i32,
        intermediates: Option<(*mut c_void, *mut c_void)>,
    ) -> Result<()> {
        self.enqueue_split_f32(
            input,
            scales,
            values,
            output,
            rows,
            hidden,
            k,
            intermediates,
            SplitFfnKernel::PersistentTiles,
        )
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn enqueue_split_f32(
        &self,
        input: *const c_void,
        scales: [*const c_void; 2],
        values: [*const c_void; 2],
        output: *mut c_void,
        rows: i32,
        hidden: i32,
        k: i32,
        intermediates: Option<(*mut c_void, *mut c_void)>,
        kernel: SplitFfnKernel,
    ) -> Result<()> {
        ensure!(!matches!(kernel,SplitFfnKernel::FixedK2048) || k==2048,
            "fixed-width split FP32 FFN requires K=2048; use the dynamic split API for other widths");
        ensure!(
            !matches!(kernel, SplitFfnKernel::Unfused) || intermediates.is_some(),
            "unfused split FFN requires caller-owned gate/up scratch"
        );
        ensure!(
            matches!(rows, 1 | 4) && hidden > 0 && k > 0 && k % 32 == 0,
            "split FP32 FFN requires rows 1/4 and complete Q8 blocks"
        );
        ensure!(
            !input.is_null()
                && !output.is_null()
                && input != output.cast_const()
                && (input as usize).is_multiple_of(16)
                && (output as usize).is_multiple_of(4),
            "split FP32 FFN requires disjoint input/output and 16-byte input alignment"
        );
        ensure!(
            scales
                .iter()
                .all(|p| !p.is_null() && (*p as usize).is_multiple_of(2))
                && values
                    .iter()
                    .all(|p| !p.is_null() && (*p as usize).is_multiple_of(4)),
            "split FP32 FFN requires FP16 scales and aligned packed payloads"
        );
        let (g, u) = intermediates.unwrap_or((std::ptr::null_mut(), std::ptr::null_mut()));
        if intermediates.is_some() {
            ensure!(
                !g.is_null()
                    && !u.is_null()
                    && g != u
                    && g != output
                    && u != output
                    && g.cast_const() != input
                    && u.cast_const() != input
                    && (g as usize).is_multiple_of(4)
                    && (u as usize).is_multiple_of(4),
                "invalid split FP32 FFN intermediate outputs"
            );
        }
        if matches!(
            kernel,
            SplitFfnKernel::NativeWarp | SplitFfnKernel::RoleParallel
        ) {
            ensure!(k <= i32::MAX - 31, "native warp K exceeds safe indexing");
            let weights = (hidden as usize)
                .checked_mul(k as usize)
                .context("native warp weight span overflow")?;
            let input_bytes = (rows as usize)
                .checked_mul(k as usize)
                .and_then(|v| v.checked_mul(4))
                .context("native warp input span overflow")?;
            let output_bytes = (rows as usize)
                .checked_mul(hidden as usize)
                .and_then(|v| v.checked_mul(4))
                .context("native warp output span overflow")?;
            let span = |p: *const c_void, bytes: usize| -> Result<_> {
                let start = p as usize;
                Ok((
                    start,
                    start
                        .checked_add(bytes)
                        .context("native warp address overflow")?,
                ))
            };
            let reads = [
                span(input, input_bytes)?,
                span(scales[0], weights / 16)?,
                span(scales[1], weights / 16)?,
                span(values[0], weights)?,
                span(values[1], weights)?,
            ];
            let pointers = [output, g, u];
            let mut writes = [(0, 0); 3];
            for index in 0..if intermediates.is_some() { 3 } else { 1 } {
                let (start, end) = span(pointers[index].cast_const(), output_bytes)?;
                for &(other_start, other_end) in reads.iter().chain(&writes[..index]) {
                    ensure!(
                        end <= other_start || other_end <= start,
                        "native warp output overlaps another span"
                    );
                }
                writes[index] = (start, end);
            }
        }
        let _guard = self.inner.lock.lock().unwrap();
        let enqueue = match kernel {
            SplitFfnKernel::RoleParallel => ffi::m40llm_ffn_q8_split_role_parallel_f32_async,
            SplitFfnKernel::NativeWarp => ffi::m40llm_ffn_q8_split_native_warp_f32_async,
            SplitFfnKernel::EightValues => ffi::m40llm_ffn_q8_split_f32x8_async,
            SplitFfnKernel::Cohorts => ffi::m40llm_ffn_q8_split_cohorts_f32_async,
            SplitFfnKernel::PersistentTiles => ffi::m40llm_ffn_q8_split_persistent_tiles_f32_async,
            SplitFfnKernel::VirtualWarp => ffi::m40llm_ffn_q8_split_virtual_warp_f32_async,
            SplitFfnKernel::Dynamic => ffi::m40llm_ffn_q8_split_f32_async,
            SplitFfnKernel::TwoColumns => ffi::m40llm_ffn_q8_split_f32_two_columns_async,
            SplitFfnKernel::FixedK2048 => ffi::m40llm_ffn_q8_split_f32_k2048_async,
            SplitFfnKernel::Unfused => ffi::m40llm_ffn_q8_split_unfused_f32_async,
            SplitFfnKernel::WeightCg => ffi::m40llm_ffn_q8_split_weight_cg_f32_async,
            SplitFfnKernel::BytePermute => ffi::m40llm_ffn_q8_split_byte_permute_f32_async,
        };
        let rc = enqueue(
            self.inner.raw.as_ptr(),
            input,
            scales.as_ptr(),
            values.as_ptr(),
            output,
            g,
            u,
            rows,
            hidden,
            k,
        );
        ensure!(rc == 0, "m40llm_ffn_q8_split_f32_async failed: {rc}");
        if matches!(kernel, SplitFfnKernel::Unfused) {
            record_async_kernel("ffn_q8_split_unfused_projection");
            record_async_kernel("ffn_q8_split_unfused_projection");
            record_async_kernel("ffn_q8_split_unfused_swiglu");
            return Ok(());
        }
        record_async_kernel(match kernel {
            SplitFfnKernel::RoleParallel => "ffn_q8_split_f32x4_role_parallel_256",
            SplitFfnKernel::NativeWarp => "ffn_q8_split_f32x4_native_warp_128",
            SplitFfnKernel::EightValues => "ffn_q8_split_f32x8_128",
            SplitFfnKernel::Cohorts => "ffn_q8_split_f32x4_cohorts_256",
            SplitFfnKernel::PersistentTiles => "ffn_q8_split_f32x4_persistent_tiles_128",
            SplitFfnKernel::VirtualWarp => "ffn_q8_split_f32x4_virtual_warp_128",
            SplitFfnKernel::Dynamic => "ffn_q8_split_f32x4_128",
            SplitFfnKernel::TwoColumns => "ffn_q8_split_f32x4_cols2_128",
            SplitFfnKernel::FixedK2048 => "ffn_q8_split_f32x4_k2048_128",
            SplitFfnKernel::WeightCg => "ffn_q8_split_f32x4_weight_cg_128",
            SplitFfnKernel::BytePermute => "ffn_q8_split_f32x4_byte_permute_128",
            SplitFfnKernel::Unfused => unreachable!("three launches recorded above"),
        });
        Ok(())
    }

    /// Enqueue packed Q8_0 weights x previously encoded Q8_1 activations,
    /// reusing the same quantized input for gate/up. Optional intermediate
    /// outputs are diagnostic; timed calls omit them.
    ///
    /// # Safety
    /// All spans are disjoint on this context and survive decode completion.
    /// Input holds shape.bytes() encoded bytes at four-byte alignment. Each
    /// two-byte-aligned weight span holds hidden*(columns/32)*34 bytes; output
    /// and any intermediates hold rows*hidden f32 values at four-byte alignment.
    /// Quantization/other-stream producers must precede this decode operation.
    pub unsafe fn ffn_q8_1_async(
        &self,
        input: *const c_void,
        gate: *const c_void,
        up: *const c_void,
        output: *mut c_void,
        intermediates: Option<(*mut c_void, *mut c_void)>,
        hidden: i32,
        shape: crate::infer::q8_activation::Q8ActivationShape,
        launch: Q8FfnQ81Launch,
    ) -> Result<()> {
        ensure!(
            hidden > 0 && shape.columns().is_multiple_of(32),
            "Q8_1 FFN requires positive hidden and complete Q8_0 input blocks"
        );
        ensure!(
            !input.is_null()
                && !gate.is_null()
                && !up.is_null()
                && !output.is_null()
                && input != output.cast_const(),
            "invalid/aliased Q8_1 FFN buffers"
        );
        ensure!(
            (input as usize).is_multiple_of(4)
                && (output as usize).is_multiple_of(4)
                && (gate as usize).is_multiple_of(2)
                && (up as usize).is_multiple_of(2),
            "Q8_1 FFN buffer alignment mismatch"
        );
        let (gate_output, up_output) =
            intermediates.unwrap_or((std::ptr::null_mut(), std::ptr::null_mut()));
        if intermediates.is_some() {
            ensure!(
                !gate_output.is_null()
                    && !up_output.is_null()
                    && gate_output != up_output
                    && gate_output != output
                    && up_output != output
                    && (gate_output as usize).is_multiple_of(4)
                    && (up_output as usize).is_multiple_of(4),
                "invalid Q8_1 diagnostic outputs"
            );
        }
        let _guard = self.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_ffn_q8_1_async(
                self.inner.raw.as_ptr(),
                input,
                gate,
                up,
                output,
                gate_output,
                up_output,
                shape.rows() as i32,
                hidden,
                shape.columns() as i32,
                shape.padded_columns() as i32,
                launch.threads(),
            )
        };
        ensure!(
            rc == 0,
            "m40llm_ffn_q8_1_async failed: {rc}, launch={launch:?}, hidden={hidden}"
        );
        record_async_kernel(match launch {
            Q8FfnQ81Launch::Threads128 => "ffn_q8_1_threads128",
            Q8FfnQ81Launch::Threads256 => "ffn_q8_1_threads256",
        });
        Ok(())
    }

    /// Enqueue the pinned Q8_1 activation contract into reusable caller storage.
    ///
    /// # Safety
    /// Input/output are disjoint, four-byte-aligned buffers on this context,
    /// holding rows*columns f32 values and shape.bytes() output bytes. Input is
    /// finite and each nonzero block has a non-underflowing FP32 amax/127 scale.
    /// Order producers before decode and retain both buffers through completion.
    pub unsafe fn quantize_q8_1_async(
        &self,
        input: *const c_void,
        output: *mut c_void,
        shape: crate::infer::q8_activation::Q8ActivationShape,
    ) -> Result<()> {
        ensure!(
            !input.is_null() && !output.is_null() && input != output.cast_const(),
            "invalid/aliased Q8_1 buffers"
        );
        ensure!(
            (input as usize).is_multiple_of(4) && (output as usize).is_multiple_of(4),
            "Q8_1 buffers require four-byte alignment"
        );
        let _guard = self.inner.lock.lock().unwrap();
        let rc = unsafe {
            ffi::m40llm_quantize_q8_1_async(
                self.inner.raw.as_ptr(),
                input,
                output,
                shape.rows() as i32,
                shape.columns() as i32,
                shape.padded_columns() as i32,
            )
        };
        ensure!(rc == 0, "m40llm_quantize_q8_1_async failed: {rc}");
        record_async_kernel("quantize_q8_1");
        Ok(())
    }

    /// Enqueue one genuine 1/4-row FP32-activation FFN ablation on decode.
    /// No allocation, transfer, environment selection or synchronization occurs.
    ///
    /// # Safety
    /// Buffers belong to this context and outlive decode completion. Input/output
    /// hold rows*k / rows*hidden f32 values and may not alias. Each weight matrix
    /// holds hidden*(k/32)*34 bytes of native Q8_0 at two-byte alignment.
    /// Adjacent-pair input requires eight-byte alignment; other inputs need four.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ffn_q8_f32_ablation_async(
        &self,
        input: *const c_void,
        gate: *const c_void,
        up: *const c_void,
        output: *mut c_void,
        rows: i32,
        hidden: i32,
        k: i32,
        variant: Q8FfnFp32Variant,
    ) -> Result<()> {
        self.enqueue_ffn_fp32(input, gate, up, output, rows, hidden, k, variant, None)
    }

    /// Capture gate/up and SwiGLU from the same follow-up kernel used by
    /// ordinary benchmark enqueue. Timed calls omit these extra output stores.
    ///
    /// # Safety
    /// Same span/stream contract as ffn_q8_f32_ablation_async; both intermediate
    /// spans hold rows*hidden f32 values and are disjoint from every other span.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ffn_q8_f32_capture_async(
        &self,
        input: *const c_void,
        gate: *const c_void,
        up: *const c_void,
        output: *mut c_void,
        gate_output: *mut c_void,
        up_output: *mut c_void,
        rows: i32,
        hidden: i32,
        k: i32,
        variant: Q8FfnFp32Variant,
    ) -> Result<()> {
        ensure!(
            variant.has_intermediate_capture(),
            "FFN variant has no intermediate capture"
        );
        self.enqueue_ffn_fp32(
            input,
            gate,
            up,
            output,
            rows,
            hidden,
            k,
            variant,
            Some((gate_output, up_output)),
        )
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn enqueue_ffn_fp32(
        &self,
        input: *const c_void,
        gate: *const c_void,
        up: *const c_void,
        output: *mut c_void,
        rows: i32,
        hidden: i32,
        k: i32,
        variant: Q8FfnFp32Variant,
        capture: Option<(*mut c_void, *mut c_void)>,
    ) -> Result<()> {
        ensure!(
            !input.is_null()
                && !gate.is_null()
                && !up.is_null()
                && !output.is_null()
                && input != output.cast_const(),
            "invalid/aliased FP32 FFN buffers"
        );
        ensure!(matches!(rows,1|4)&&hidden>0&&k>0&&k%32==0,"FP32 FFN ablations require rows 1/4, positive hidden and complete 32-value Q8_0 blocks");
        ensure!(
            (gate as usize).is_multiple_of(2) && (up as usize).is_multiple_of(2),
            "FP32 FFN Q8_0 weights require two-byte alignment"
        );
        ensure!(
            (input as usize).is_multiple_of(4) && (output as usize).is_multiple_of(4),
            "FP32 FFN input/output require four-byte alignment"
        );
        ensure!(
            !variant.has_coalesced_pairs() || (input as usize).is_multiple_of(8),
            "coalesced FFN input requires eight-byte alignment"
        );
        if let Some((g, u)) = capture {
            ensure!(
                !g.is_null()
                    && !u.is_null()
                    && g != u
                    && g != output
                    && u != output
                    && g.cast_const() != input
                    && u.cast_const() != input
                    && (g as usize).is_multiple_of(4)
                    && (u as usize).is_multiple_of(4),
                "invalid/aliased FFN intermediate outputs"
            );
        }
        let _guard = self.inner.lock.lock().unwrap();
        let rc = unsafe {
            if let Some((g, u)) = capture {
                let capture = if variant.has_coalesced_pairs() {
                    ffi::m40llm_ffn_q8_f32_coalesced_capture_async
                } else {
                    ffi::m40llm_ffn_q8_f32_factored_capture_async
                };
                capture(
                    self.inner.raw.as_ptr(),
                    input,
                    gate,
                    up,
                    output,
                    g,
                    u,
                    rows,
                    hidden,
                    k,
                )
            } else {
                ffi::m40llm_ffn_q8_f32_ablation_async(
                    self.inner.raw.as_ptr(),
                    input,
                    gate,
                    up,
                    output,
                    rows,
                    hidden,
                    k,
                    variant as i32,
                )
            }
        };
        ensure!(rc==0,"m40llm_ffn_q8_f32_ablation_async failed: {rc}, variant={variant:?}, rows={rows}, hidden={hidden}, k={k}");
        record_async_kernel(variant.operation_name());
        Ok(())
    }
}
