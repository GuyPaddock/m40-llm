use anyhow::{ensure, Result};
use m40_llm::{
    cuda::{CudaContext, CudaEvent, CudaStream, DeviceBuffer},
    profile,
};
use serde::Serialize;
use std::{ffi::c_void, time::Instant};

const GUARD: usize = 64;
const SENTINEL: u8 = 0xa5;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Arm {
    H2d,
    D2h,
    SameStream,
    Handoff,
    IdleDecode,
    IdlePrefill,
}

struct GuardedDevice {
    buffer: DeviceBuffer,
    bytes: usize,
}
impl GuardedDevice {
    fn new(ctx: &CudaContext, bytes: usize) -> Result<Self> {
        Ok(Self {
            buffer: DeviceBuffer::new_tagged(ctx, bytes + GUARD * 2, "transfer_probe:guarded")?,
            bytes,
        })
    }
    fn get_pointer(&self) -> *mut c_void {
        unsafe { self.buffer.as_mut_ptr().cast::<u8>().add(GUARD).cast() }
    }
    fn initialize(&self, ctx: &CudaContext, payload: Option<&[u8]>) -> Result<()> {
        let mut bytes = vec![SENTINEL; self.bytes + GUARD * 2];
        if let Some(payload) = payload {
            ensure!(payload.len() == self.bytes, "invalid guarded payload");
            bytes[GUARD..GUARD + self.bytes].copy_from_slice(payload);
        }
        unsafe { ctx.memcpy_h2d(self.buffer.as_mut_ptr(), bytes.as_ptr().cast(), bytes.len()) }
    }
    fn verify(&self, ctx: &CudaContext, expected: Option<&[u8]>) -> Result<()> {
        let mut actual = vec![0; self.bytes + GUARD * 2];
        unsafe {
            ctx.memcpy_d2h(
                actual.as_mut_ptr().cast(),
                self.buffer.as_ptr(),
                actual.len(),
            )?;
        }
        ensure!(
            actual[..GUARD]
                .iter()
                .chain(&actual[GUARD + self.bytes..])
                .all(|v| *v == SENTINEL),
            "device guard changed"
        );
        if let Some(expected) = expected {
            ensure!(
                actual[GUARD..GUARD + self.bytes] == *expected,
                "transferred payload differs"
            );
        } else {
            ensure!(
                actual[GUARD..GUARD + self.bytes]
                    .iter()
                    .all(|v| *v == SENTINEL),
                "unused device payload changed"
            );
        }
        Ok(())
    }
}

pub struct Probe {
    pub ctx: CudaContext,
    pub bytes: usize,
    source: GuardedDevice,
    middle: GuardedDevice,
    destination: GuardedDevice,
    host_source: Vec<u8>,
    host_destination: Vec<u8>,
    start: CudaEvent,
    stop: CudaEvent,
    generation: u32,
}

#[derive(Serialize)]
pub struct Observation {
    pub arm: Arm,
    pub generation: u32,
    pub iterations: usize,
    pub wall_ns_per_operation: f64,
    pub cuda_timeline_ns_per_operation: f64,
    pub prepare_ms: f64,
    pub counters: profile::OpCounters,
    pub explicit_d2d_calls: usize,
    pub logical_transfer_bytes_per_operation: usize,
}

impl Probe {
    pub fn new(ctx: &CudaContext, bytes: usize) -> Result<Self> {
        ensure!(
            (1..=32 * 1024 * 1024).contains(&bytes),
            "transfer size out of bounds"
        );
        Ok(Self {
            ctx: ctx.clone(),
            bytes,
            source: GuardedDevice::new(ctx, bytes)?,
            middle: GuardedDevice::new(ctx, bytes)?,
            destination: GuardedDevice::new(ctx, bytes)?,
            host_source: vec![SENTINEL; bytes + GUARD * 2],
            host_destination: vec![SENTINEL; bytes + GUARD * 2],
            start: ctx.create_event()?,
            stop: ctx.create_event()?,
            generation: 0,
        })
    }

    pub fn get_host_bytes(&self) -> usize {
        self.host_source.len() + self.host_destination.len()
    }

    fn prepare(&mut self) -> Result<()> {
        self.ctx.synchronize_stream(CudaStream::Decode)?;
        self.ctx.synchronize_stream(CudaStream::Prefill)?;
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("fixture generation overflow"))?;
        for (index, value) in self.host_source[GUARD..GUARD + self.bytes]
            .iter_mut()
            .enumerate()
        {
            let mut mixed = (index as u32).wrapping_add(self.generation.wrapping_mul(104729));
            mixed ^= mixed >> 16;
            mixed = mixed.wrapping_mul(0x7feb352d);
            mixed ^= mixed >> 15;
            *value = mixed as u8;
        }
        self.host_destination.fill(SENTINEL);
        self.source.initialize(
            &self.ctx,
            Some(&self.host_source[GUARD..GUARD + self.bytes]),
        )?;
        self.middle.initialize(&self.ctx, None)?;
        self.destination.initialize(&self.ctx, None)?;
        Ok(())
    }

    unsafe fn execute(&mut self, arm: Arm) -> Result<()> {
        match arm {
            Arm::H2d => self.ctx.memcpy_h2d(
                self.middle.get_pointer(),
                self.host_source.as_ptr().add(GUARD).cast(),
                self.bytes,
            )?,
            Arm::D2h => self.ctx.memcpy_d2h(
                self.host_destination.as_mut_ptr().add(GUARD).cast(),
                self.source.get_pointer(),
                self.bytes,
            )?,
            Arm::SameStream | Arm::Handoff => {
                let producer = if matches!(arm, Arm::Handoff) {
                    CudaStream::Prefill
                } else {
                    CudaStream::Decode
                };
                self.ctx.memcpy_d2d_async(
                    self.middle.get_pointer(),
                    self.source.get_pointer(),
                    self.bytes,
                    producer,
                )?;
                if matches!(arm, Arm::Handoff) {
                    self.ctx.stream_wait_for_stream(
                        CudaStream::Decode,
                        CudaStream::Prefill,
                        "transfer_probe_handoff",
                    )?;
                }
                self.ctx.memcpy_d2d_async(
                    self.destination.get_pointer(),
                    self.middle.get_pointer(),
                    self.bytes,
                    CudaStream::Decode,
                )?;
                // Complete every operation before the next producer reuses middle.
                self.ctx.synchronize_stream(CudaStream::Decode)?;
            }
            Arm::IdleDecode => self.ctx.synchronize_stream(CudaStream::Decode)?,
            Arm::IdlePrefill => self.ctx.synchronize_stream(CudaStream::Prefill)?,
        }
        Ok(())
    }

    fn verify(&self, arm: Arm) -> Result<()> {
        let expected = &self.host_source[GUARD..GUARD + self.bytes];
        ensure!(
            self.host_source[..GUARD]
                .iter()
                .chain(&self.host_source[GUARD + self.bytes..])
                .chain(&self.host_destination[..GUARD])
                .chain(&self.host_destination[GUARD + self.bytes..])
                .all(|v| *v == SENTINEL),
            "host guard changed"
        );
        self.source.verify(&self.ctx, Some(expected))?;
        let changes_middle = matches!(arm, Arm::H2d | Arm::SameStream | Arm::Handoff);
        self.middle
            .verify(&self.ctx, changes_middle.then_some(expected))?;
        self.destination.verify(
            &self.ctx,
            matches!(arm, Arm::SameStream | Arm::Handoff).then_some(expected),
        )?;
        if matches!(arm, Arm::D2h) {
            ensure!(
                self.host_destination[GUARD..GUARD + self.bytes] == *expected,
                "host payload differs"
            );
        } else {
            ensure!(
                self.host_destination[GUARD..GUARD + self.bytes]
                    .iter()
                    .all(|v| *v == SENTINEL),
                "unused host payload changed"
            );
        }
        Ok(())
    }

    pub fn measure(&mut self, arm: Arm, iterations: usize) -> Result<Observation> {
        ensure!((1..=1000).contains(&iterations), "iterations out of bounds");
        let preparation = Instant::now();
        self.prepare()?;
        let prepare_ms = preparation.elapsed().as_secs_f64() * 1000.0;
        // The pageable APIs expose no default-stream timing events. This
        // ordered envelope is a device timeline, not DMA active time.
        self.start.record(CudaStream::Decode)?;
        self.ctx.synchronize_stream(CudaStream::Decode)?;
        let live = CudaContext::total_device_bytes();
        let before = profile::snapshot();
        let began = Instant::now();
        for _ in 0..iterations {
            unsafe {
                self.execute(arm)?;
            }
        }
        let wall_ns_per_operation = began.elapsed().as_secs_f64() * 1e9 / iterations as f64;
        let counters = profile::counter_delta(&before, &profile::snapshot());
        self.stop.record(CudaStream::Decode)?;
        let cuda_timeline_ns_per_operation = f64::from(
            self.start
                .elapsed_sync(&self.stop, "transfer_probe_timeline")?,
        ) * 1e6
            / iterations as f64;
        ensure!(
            CudaContext::total_device_bytes() == live
                && counters.device_allocs == 0
                && counters.device_frees == 0,
            "timed transfer allocated or freed device memory"
        );
        match arm {
            Arm::H2d => ensure!(
                counters.h2d_copies == iterations as u64
                    && counters.h2d_bytes == (iterations * self.bytes) as u64
                    && counters.d2h_copies == 0,
                "H2D counters differ"
            ),
            Arm::D2h => ensure!(
                counters.d2h_copies == iterations as u64
                    && counters.d2h_bytes == (iterations * self.bytes) as u64
                    && counters.h2d_copies == 0,
                "D2H counters differ"
            ),
            _ => ensure!(
                counters.h2d_copies == 0 && counters.d2h_copies == 0,
                "unexpected host copy"
            ),
        }
        self.verify(arm)?;
        let copy_multiplier = match arm {
            Arm::H2d | Arm::D2h => 1,
            Arm::SameStream | Arm::Handoff => 2,
            _ => 0,
        };
        Ok(Observation {
            arm,
            generation: self.generation,
            iterations,
            wall_ns_per_operation,
            cuda_timeline_ns_per_operation,
            prepare_ms,
            counters,
            explicit_d2d_calls: if matches!(arm, Arm::SameStream | Arm::Handoff) {
                iterations * 2
            } else {
                0
            },
            logical_transfer_bytes_per_operation: copy_multiplier * self.bytes,
        })
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        let _ = self.ctx.synchronize_stream(CudaStream::Prefill);
        let _ = self.ctx.synchronize_stream(CudaStream::Decode);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distinct generations, odd byte counts and both consumer streams must
    /// preserve transferred payloads and every host/device guard. All allocated
    /// buffers must retire after repeated probe lifetimes.
    #[test]
    #[ignore = "requires idle M40 hardware; run serially with --include-ignored"]
    fn test_completed_transfers_and_handoffs_preserve_bytes() -> Result<()> {
        crate::gpu::check_gpu_processes()?;
        crate::gpu::validate_environment()?;
        let ctx = CudaContext::new_m40()?;
        let props = ctx.current_device_props()?;
        ensure!(props.major == 5 && props.minor == 2, "requires sm_52");
        let baseline = CudaContext::total_device_bytes();
        for _ in 0..2 {
            for bytes in [1, 17, 8192, 44032] {
                let mut probe = Probe::new(&ctx, bytes)?;
                for arm in [
                    Arm::H2d,
                    Arm::D2h,
                    Arm::SameStream,
                    Arm::Handoff,
                    Arm::IdleDecode,
                    Arm::IdlePrefill,
                ] {
                    probe.measure(arm, 3)?;
                }
                drop(probe);
                ensure!(
                    CudaContext::total_device_bytes() == baseline,
                    "transfer probe leaked storage"
                );
            }
        }
        Ok(())
    }
}
