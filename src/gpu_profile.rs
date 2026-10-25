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
    pub counters: OpCounters,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HostTiming {
    pub operation: String,
    pub elapsed_ms: f64,
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

struct EventPair {
    start: CudaEvent,
    stop: CudaEvent,
}

struct PendingStage {
    name: &'static str,
    layer_id: Option<u32>,
    stream: CudaStream,
    cpu_call_elapsed_ms: Option<f64>,
    counters: OpCounters,
}

struct Collector {
    origin: CudaEvent,
    pool: Vec<EventPair>,
    pending: Vec<PendingStage>,
    host_timings: Vec<HostTiming>,
    layer_id: Option<u32>,
    is_active: bool,
    completed_tokens: usize,
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
        let capacity = (layer_count as usize)
            .checked_mul(16)
            .and_then(|n| n.checked_add(16))
            .context("stage event pool size overflow")?;
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
                completed_tokens: 0,
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
    ) -> Result<Option<ProfileTokenScope>> {
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
    ) -> Result<ProfileTokenScope> {
        ensure!(
            ACTIVE.with(|active| active.borrow().is_none()),
            "nested GPU token profiling is unsupported"
        );
        let mut collector = self.collector.lock().unwrap();
        ensure!(
            !collector.is_active,
            "previous GPU token profile did not complete"
        );
        collector.pending.clear();
        collector.host_timings.clear();
        collector.layer_id = None;
        collector.origin.record(CudaStream::Decode)?;
        collector.is_active = true;
        drop(collector);
        ACTIVE.with(|active| *active.borrow_mut() = Some(self.collector.clone()));
        Ok(ProfileTokenScope {
            collector: self.collector.clone(),
            session_id: self.session_id,
            generated_step,
            absolute_position,
            started_at: Instant::now(),
            before: profile::snapshot(),
            setup_ms: self.setup_ms,
            _same_thread: PhantomData,
        })
    }

    pub fn finish_token(&mut self, scope: ProfileTokenScope) -> Result<TokenGpuProfile> {
        ensure!(
            Arc::ptr_eq(&self.collector, &scope.collector),
            "token profile belongs to a different decode session"
        );
        let report = scope.finish()?;
        if let Some(output) = self.output.as_mut() {
            serde_json::to_writer(&mut *output, &report)?;
            output.write_all(b"\n")?;
            output.flush()?;
        } else {
            eprintln!("[gpu-stage-profile] {}", serde_json::to_string(&report)?);
        }
        Ok(report)
    }
}

pub struct ProfileTokenScope {
    collector: Arc<Mutex<Collector>>,
    session_id: u64,
    generated_step: usize,
    absolute_position: usize,
    started_at: Instant,
    before: ProfileSnapshot,
    setup_ms: f64,
    _same_thread: PhantomData<Rc<()>>,
}

impl ProfileTokenScope {
    /// Called only after the caller's existing logits/argmax completion boundary.
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
            event_pool_reused: collector.completed_tokens > 0,
            counters: profile::counter_delta(&self.before, &after),
            stages,
            host_timings: collector.host_timings.clone(),
        };
        collector.completed_tokens += 1;
        collector.is_active = false;
        Ok(report)
    }
}

impl Drop for ProfileTokenScope {
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
        collector.pending.push(PendingStage {
            name,
            layer_id,
            stream,
            cpu_call_elapsed_ms: None,
            counters: OpCounters::default(),
        });
        drop(collector);
        Ok(Self {
            capture: Some((shared, index, Instant::now(), before)),
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
    collector.lock().unwrap().host_timings.push(HostTiming {
        operation: operation.to_owned(),
        elapsed_ms: started_at.elapsed().as_secs_f64() * 1000.0,
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
}
