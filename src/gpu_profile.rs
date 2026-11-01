//! Session-owned, sampled CUDA-event spans. No stream waits are added here.

use std::{
    cell::RefCell,
    fs::{File, OpenOptions},
    io::Write,
    marker::PhantomData,
    path::PathBuf,
    rc::Rc,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Instant,
};

use anyhow::{ensure, Context, Result};

use crate::{
    cuda::{CudaContext, CudaEvent, CudaStream},
    profile::{self, OpCounters, ProfileSnapshot},
};

thread_local! {
    // This scope surrounds synchronous Rust enqueue calls, never an await.
    static ACTIVE: RefCell<Option<Arc<Mutex<Collector>>>> = const { RefCell::new(None) };
}
static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct GpuProfileConfig {
    pub generated_steps: Vec<usize>,
    pub report_path: Option<PathBuf>,
    pub should_profile_prefill: bool,
}

impl GpuProfileConfig {
    pub fn from_env() -> Result<Option<Self>> {
        if std::env::var("M40LLM_STAGE_PROFILE").ok().as_deref() != Some("1") {
            return Ok(None);
        }
        let steps =
            std::env::var("M40LLM_STAGE_PROFILE_TOKENS").unwrap_or_else(|_| "0,255,511".to_owned());
        let mut generated_steps: Vec<usize> = steps
            .split(',')
            .map(|step| {
                step.trim()
                    .parse()
                    .context("invalid M40LLM_STAGE_PROFILE_TOKENS index")
            })
            .collect::<Result<_>>()?;
        generated_steps.sort_unstable();
        generated_steps.dedup();
        ensure!(
            !generated_steps.is_empty() && generated_steps.len() <= 32,
            "select between one and 32 stage-profile token indices"
        );
        Ok(Some(Self {
            generated_steps,
            report_path: std::env::var_os("M40LLM_STAGE_PROFILE_REPORT").map(PathBuf::from),
            should_profile_prefill: std::env::var("M40LLM_STAGE_PROFILE_PREFILL")
                .ok()
                .as_deref()
                == Some("1"),
        }))
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GpuStageTiming {
    pub stage: String,
    pub layer_id: Option<u32>,
    pub stream: String,
    pub gpu_start_ms: f32,
    pub gpu_end_ms: f32,
    pub gpu_elapsed_ms: f32,
    /// Host call duration, including any synchronization already inside a wrapper.
    pub cpu_call_elapsed_ms: f64,
    /// Host offset from scope start, not a timestamp on the CUDA event clock.
    /// Historical records have no host offset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_start_ms: Option<f64>,
    pub counters: OpCounters,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HostTiming {
    pub operation: String,
    pub elapsed_ms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_ms: Option<f64>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct TokenGpuProfile {
    pub process_id: u32,
    pub session_id: u64,
    pub generated_step: usize,
    pub absolute_position: usize,
    pub token_wall_ms: f64,
    pub gpu_stage_sum_ms: f64,
    pub gpu_stage_union_ms: f64,
    pub gpu_timeline_span_ms: f64,
    pub wall_minus_gpu_union_ms: f64,
    pub event_pool_capacity: usize,
    pub event_pool_setup_ms: f64,
    pub event_pool_reused: bool,
    pub counters: OpCounters,
    pub stages: Vec<GpuStageTiming>,
    pub host_timings: Vec<HostTiming>,
}

/// A packed-prefix scope is not a generated token. Grouped spans cover its
/// projections and non-GEMM work; stream intervals can include host enqueue gaps.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct PrefillGpuProfile {
    pub kind: String,
    pub process_id: u32,
    pub session_id: u64,
    pub prefix_tokens: usize,
    pub prefill_wall_ms: f64,
    pub gpu_stage_sum_ms: f64,
    pub gpu_stage_union_ms: f64,
    pub gpu_timeline_span_ms: f64,
    pub wall_minus_gpu_union_ms: f64,
    pub event_pool_capacity: usize,
    pub event_pool_setup_ms: f64,
    pub event_pool_reused: bool,
    pub counters: OpCounters,
    pub stages: Vec<GpuStageTiming>,
    pub host_timings: Vec<HostTiming>,
}

struct EventPair {
    start: CudaEvent,
    stop: CudaEvent,
}

struct PendingStage {
    name: &'static str,
    layer_id: Option<u32>,
    stream: CudaStream,
    cpu_call_elapsed_ms: Option<f64>,
    cpu_start_ms: f64,
    counters: OpCounters,
}

struct Collector {
    origin: CudaEvent,
    pool: Vec<EventPair>,
    pending: Vec<PendingStage>,
    host_timings: Vec<HostTiming>,
    layer_id: Option<u32>,
    is_active: bool,
    completed_scopes: usize,
    host_origin: Instant,
}

pub struct DecodeGpuProfiler {
    config: GpuProfileConfig,
    collector: Arc<Mutex<Collector>>,
    session_id: u64,
    prompt_len: Option<usize>,
    setup_ms: f64,
    output: Option<File>,
}

impl DecodeGpuProfiler {
    pub fn from_env(ctx: &CudaContext, layer_count: u32) -> Result<Option<Self>> {
        GpuProfileConfig::from_env()?
            .map(|config| Self::new(ctx, layer_count, config))
            .transpose()
    }

    pub fn new(ctx: &CudaContext, layer_count: u32, config: GpuProfileConfig) -> Result<Self> {
        let capacity = (layer_count as usize)
            .checked_mul(if config.should_profile_prefill {
                24
            } else {
                16
            })
            .and_then(|n| n.checked_add(16))
            .context("stage event pool size overflow")?;
        Self::create(ctx, config, capacity)
    }

    /// Explicit cold allocation for bounded research scopes with more spans
    /// than one ordinary decode token. Existing constructor sizes are unchanged.
    pub fn new_with_event_capacity(
        ctx: &CudaContext,
        config: GpuProfileConfig,
        capacity: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=4096).contains(&capacity),
            "invalid diagnostic event capacity"
        );
        Self::create(ctx, config, capacity)
    }

    fn create(ctx: &CudaContext, config: GpuProfileConfig, capacity: usize) -> Result<Self> {
        let setup_start = Instant::now();
        ensure!(
            !config.generated_steps.is_empty(),
            "stage profile needs token indices"
        );
        let output = config
            .report_path
            .as_ref()
            .map(|path| {
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .with_context(|| format!("open stage-profile report {}", path.display()))
            })
            .transpose()?;
        let origin = ctx.create_event()?;
        let pool: Vec<EventPair> = (0..capacity)
            .map(|_| {
                Ok(EventPair {
                    start: ctx.create_event()?,
                    stop: ctx.create_event()?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            config,
            collector: Arc::new(Mutex::new(Collector {
                origin,
                pool,
                pending: Vec::with_capacity(capacity),
                host_timings: Vec::with_capacity(16),
                layer_id: None,
                is_active: false,
                completed_scopes: 0,
                host_origin: Instant::now(),
            })),
            session_id: NEXT_SESSION.fetch_add(1, Ordering::Relaxed),
            prompt_len: None,
            setup_ms: setup_start.elapsed().as_secs_f64() * 1000.0,
            output,
        })
    }

    pub fn reset_prompt(&mut self) {
        self.prompt_len = None;
    }

    pub fn begin_for_ids(
        &mut self,
        ids_len: usize,
        token_idx: usize,
    ) -> Result<Option<GpuProfileScope>> {
        let prompt_len = *self.prompt_len.get_or_insert(ids_len);
        if token_idx + 1 < prompt_len {
            return Ok(None);
        }
        let step = token_idx + 1 - prompt_len;
        if !self.config.generated_steps.contains(&step) {
            return Ok(None);
        }
        self.begin_token(step, token_idx).map(Some)
    }

    pub fn begin_token(
        &mut self,
        generated_step: usize,
        absolute_position: usize,
    ) -> Result<GpuProfileScope> {
        ensure!(
            ACTIVE.with(|active| active.borrow().is_none()),
            "nested GPU scope profiling is unsupported"
        );
        let mut collector = self.collector.lock().unwrap();
        ensure!(
            !collector.is_active,
            "previous GPU profile scope did not complete"
        );
        collector.pending.clear();
        collector.host_timings.clear();
        collector.layer_id = None;
        collector.origin.record(CudaStream::Decode)?;
        let started_at = Instant::now();
        collector.host_origin = started_at;
        collector.is_active = true;
        drop(collector);
        ACTIVE.with(|active| *active.borrow_mut() = Some(self.collector.clone()));
        Ok(GpuProfileScope {
            collector: self.collector.clone(),
            session_id: self.session_id,
            generated_step,
            absolute_position,
            prefix_tokens: None,
            started_at,
            before: profile::snapshot(),
            setup_ms: self.setup_ms,
            _same_thread: PhantomData,
        })
    }

    pub fn begin_prefill(&mut self, prefix_tokens: usize) -> Result<Option<GpuProfileScope>> {
        if !self.config.should_profile_prefill {
            return Ok(None);
        }
        ensure!(
            prefix_tokens > 0,
            "prefill profile needs a non-empty prefix"
        );
        // Reuse the same event collector before generated step zero. The
        // prefill writer below deliberately emits no generated-token fields.
        let mut scope = self.begin_token(0, prefix_tokens - 1)?;
        scope.prefix_tokens = Some(prefix_tokens);
        Ok(Some(scope))
    }

    /// Resolve only after the packed forward's existing two-stream drain.
    /// Timestamp queries add no stream waits or synchronization.
    pub fn finish_prefill(
        &mut self,
        scope: GpuProfileScope,
        prefix_tokens: usize,
    ) -> Result<PrefillGpuProfile> {
        ensure!(
            Arc::ptr_eq(&self.collector, &scope.collector),
            "prefill profile belongs to a different decode session"
        );
        ensure!(
            prefix_tokens > 0 && scope.prefix_tokens == Some(prefix_tokens),
            "prefill profile prefix length mismatch"
        );
        let collected = scope.finish()?;
        let report = PrefillGpuProfile {
            kind: "packed_prefill_gpu_profile".to_owned(),
            process_id: collected.process_id,
            session_id: collected.session_id,
            prefix_tokens,
            prefill_wall_ms: collected.token_wall_ms,
            gpu_stage_sum_ms: collected.gpu_stage_sum_ms,
            gpu_stage_union_ms: collected.gpu_stage_union_ms,
            gpu_timeline_span_ms: collected.gpu_timeline_span_ms,
            wall_minus_gpu_union_ms: collected.wall_minus_gpu_union_ms,
            event_pool_capacity: collected.event_pool_capacity,
            event_pool_setup_ms: collected.event_pool_setup_ms,
            event_pool_reused: collected.event_pool_reused,
            counters: collected.counters,
            stages: collected.stages,
            host_timings: collected.host_timings,
        };
        self.write_report(&report)?;
        Ok(report)
    }

    pub fn finish_token(&mut self, scope: GpuProfileScope) -> Result<TokenGpuProfile> {
        ensure!(
            scope.prefix_tokens.is_none(),
            "prefill scope must use finish_prefill"
        );
        ensure!(
            Arc::ptr_eq(&self.collector, &scope.collector),
            "token profile belongs to a different decode session"
        );
        let report = scope.finish()?;
        self.write_report(&report)?;
        Ok(report)
    }

    fn write_report(&mut self, report: &impl serde::Serialize) -> Result<()> {
        if let Some(output) = self.output.as_mut() {
            serde_json::to_writer(&mut *output, report)?;
            output.write_all(b"\n")?;
            output.flush()?;
        } else {
            eprintln!("[gpu-stage-profile] {}", serde_json::to_string(report)?);
        }
        Ok(())
    }
}

pub struct GpuProfileScope {
    collector: Arc<Mutex<Collector>>,
    session_id: u64,
    generated_step: usize,
    absolute_position: usize,
    prefix_tokens: Option<usize>,
    started_at: Instant,
    before: ProfileSnapshot,
    setup_ms: f64,
    _same_thread: PhantomData<Rc<()>>,
}

impl GpuProfileScope {
    /// Called only after the existing logits/argmax or packed-forward drain.
    pub fn finish(self) -> Result<TokenGpuProfile> {
        let wall_ms = self.started_at.elapsed().as_secs_f64() * 1000.0;
        let after = profile::snapshot();
        let mut collector = self.collector.lock().unwrap();
        let mut stages = Vec::with_capacity(collector.pending.len());
        for (index, pending) in collector.pending.iter().enumerate() {
            let cpu_ms = pending
                .cpu_call_elapsed_ms
                .context("unfinished GPU profile stage")?;
            let events = &collector.pool[index];
            let gpu_start_ms = collector
                .origin
                .elapsed_ready(&events.start)?
                .context("stage start not ready at token completion")?;
            let gpu_end_ms = collector
                .origin
                .elapsed_ready(&events.stop)?
                .context("stage end not ready at token completion")?;
            let gpu_elapsed_ms = events
                .start
                .elapsed_ready(&events.stop)?
                .context("stage duration not ready at token completion")?;
            stages.push(GpuStageTiming {
                stage: pending.name.to_owned(),
                layer_id: pending.layer_id,
                stream: match pending.stream {
                    CudaStream::Decode => "decode",
                    CudaStream::Prefill => "prefill",
                }
                .to_owned(),
                gpu_start_ms,
                gpu_end_ms,
                gpu_elapsed_ms,
                cpu_call_elapsed_ms: cpu_ms,
                cpu_start_ms: Some(pending.cpu_start_ms),
                counters: pending.counters,
            });
        }
        let intervals: Vec<(f64, f64)> = stages
            .iter()
            .map(|stage| (stage.gpu_start_ms as f64, stage.gpu_end_ms as f64))
            .collect();
        let union_ms = interval_union_ms(&intervals);
        let span_ms = if intervals.is_empty() {
            0.0
        } else {
            intervals
                .iter()
                .map(|(_, end)| *end)
                .fold(f64::NEG_INFINITY, f64::max)
                - intervals
                    .iter()
                    .map(|(start, _)| *start)
                    .fold(f64::INFINITY, f64::min)
        };
        let report = TokenGpuProfile {
            process_id: std::process::id(),
            session_id: self.session_id,
            generated_step: self.generated_step,
            absolute_position: self.absolute_position,
            token_wall_ms: wall_ms,
            gpu_stage_sum_ms: stages.iter().map(|stage| stage.gpu_elapsed_ms as f64).sum(),
            gpu_stage_union_ms: union_ms,
            gpu_timeline_span_ms: span_ms,
            wall_minus_gpu_union_ms: (wall_ms - union_ms).max(0.0),
            event_pool_capacity: collector.pool.len(),
            event_pool_setup_ms: self.setup_ms,
            event_pool_reused: collector.completed_scopes > 0,
            counters: profile::counter_delta(&self.before, &after),
            stages,
            host_timings: collector.host_timings.clone(),
        };
        collector.completed_scopes += 1;
        collector.is_active = false;
        Ok(report)
    }
}

impl Drop for GpuProfileScope {
    fn drop(&mut self) {
        ACTIVE.with(|active| {
            let mut active = active.borrow_mut();
            if active
                .as_ref()
                .is_some_and(|value| Arc::ptr_eq(value, &self.collector))
            {
                *active = None;
            }
        });
    }
}

pub fn set_layer(layer_id: Option<u32>) {
    ACTIVE.with(|active| {
        if let Some(collector) = active.borrow().as_ref() {
            collector.lock().unwrap().layer_id = layer_id;
        }
    });
}

pub(crate) fn is_recording() -> bool {
    ACTIVE.with(|active| active.borrow().is_some())
}

pub struct GpuStageSpan {
    capture: Option<(Arc<Mutex<Collector>>, usize, Instant, ProfileSnapshot)>,
}

impl GpuStageSpan {
    pub fn begin(name: &'static str, stream: CudaStream) -> Result<Self> {
        let collector = ACTIVE.with(|active| active.borrow().clone());
        let Some(shared) = collector else {
            return Ok(Self { capture: None });
        };
        let before = profile::snapshot();
        let mut collector = shared.lock().unwrap();
        let index = collector.pending.len();
        ensure!(
            index < collector.pool.len(),
            "GPU stage event pool exhausted"
        );
        collector.pool[index].start.record(stream)?;
        let layer_id = collector.layer_id;
        let started_at = Instant::now();
        let cpu_start_ms = started_at
            .duration_since(collector.host_origin)
            .as_secs_f64()
            * 1000.0;
        collector.pending.push(PendingStage {
            name,
            layer_id,
            stream,
            cpu_call_elapsed_ms: None,
            cpu_start_ms,
            counters: OpCounters::default(),
        });
        drop(collector);
        Ok(Self {
            capture: Some((shared, index, started_at, before)),
        })
    }

    pub fn finish(self) -> Result<()> {
        let Some((shared, index, started_at, before)) = self.capture else {
            return Ok(());
        };
        let cpu_ms = started_at.elapsed().as_secs_f64() * 1000.0;
        let mut collector = shared.lock().unwrap();
        let stream = collector.pending[index].stream;
        collector.pool[index].stop.record(stream)?;
        let counters = profile::counter_delta(&before, &profile::snapshot());
        collector.pending[index].cpu_call_elapsed_ms = Some(cpu_ms);
        collector.pending[index].counters = counters;
        Ok(())
    }

    pub fn finish_as(self, name: &'static str) -> Result<()> {
        if let Some((shared, index, _, _)) = self.capture.as_ref() {
            shared.lock().unwrap().pending[*index].name = name;
        }
        self.finish()
    }
}

pub fn measure_host<T>(operation: &'static str, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let collector = ACTIVE.with(|active| active.borrow().clone());
    let Some(collector) = collector else {
        return f();
    };
    let started_at = Instant::now();
    let result = f();
    let elapsed_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    let mut collector = collector.lock().unwrap();
    let start_ms = started_at
        .duration_since(collector.host_origin)
        .as_secs_f64()
        * 1000.0;
    collector.host_timings.push(HostTiming {
        operation: operation.to_owned(),
        elapsed_ms,
        start_ms: Some(start_ms),
    });
    result
}

fn interval_union_ms(intervals: &[(f64, f64)]) -> f64 {
    let mut intervals = intervals.to_vec();
    intervals.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut union_ms = 0.0;
    let mut current = None;
    for (start, end) in intervals {
        match current {
            Some((_, ref mut current_end)) if start <= *current_end => {
                *current_end = end.max(*current_end)
            }
            Some((current_start, current_end)) => {
                union_ms += current_end - current_start;
                current = Some((start, end));
            }
            None => current = Some((start, end)),
        }
    }
    if let Some((start, end)) = current {
        union_ms += end - start;
    }
    union_ms
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Account for overlapping producer streams without summing their overlap twice.
    #[test]
    fn test_gpu_interval_union_avoids_double_counting() {
        assert_eq!(
            interval_union_ms(&[(0.0, 4.0), (2.0, 6.0), (8.0, 9.0)]),
            7.0
        );
        assert_eq!(interval_union_ms(&[(3.0, 4.0), (0.0, 10.0)]), 10.0);
        assert_eq!(interval_union_ms(&[]), 0.0);
    }

    /// Historical profiles remain readable without inventing missing host timestamps.
    #[test]
    fn test_historical_host_timing_has_no_offset() {
        let timing: HostTiming = serde_json::from_value(serde_json::json!({
            "operation": "d2h_copy", "elapsed_ms": 0.25
        }))
        .unwrap();
        assert_eq!(timing.start_ms, None);
        assert_eq!(timing.elapsed_ms, 0.25);
        assert!(serde_json::to_value(timing)
            .unwrap()
            .get("start_ms")
            .is_none());
    }

    /// Nested blocking calls overlap enqueue spans and must not be added twice.
    #[test]
    fn test_host_timeline_retains_offsets_and_overlap() {
        let spans: Vec<HostTiming> = serde_json::from_value(serde_json::json!([
            {"operation":"enqueue", "start_ms":1.0, "elapsed_ms":4.0},
            {"operation":"blocking_copy", "start_ms":2.0, "elapsed_ms":2.0},
            {"operation":"next_enqueue", "start_ms":7.0, "elapsed_ms":1.0}
        ]))
        .unwrap();
        let intervals: Vec<_> = spans
            .iter()
            .map(|span| {
                let start = span.start_ms.unwrap();
                (start, start + span.elapsed_ms)
            })
            .collect();
        assert_eq!(interval_union_ms(&intervals), 5.0);
        assert_eq!(serde_json::to_value(&spans).unwrap()[2]["start_ms"], 7.0);
    }

    /// Disabled instrumentation invokes the operation once and preserves its error.
    #[test]
    fn test_unprofiled_host_call_preserves_behavior() {
        let mut calls = 0;
        let result: Result<()> = measure_host("test", || {
            calls += 1;
            anyhow::bail!("operation failed")
        });
        assert_eq!(calls, 1);
        assert_eq!(result.unwrap_err().to_string(), "operation failed");
    }
}
