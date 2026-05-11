use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkMode {
    ColdLoadTotal,
    ColdFirstTokenWithMaterialization,
    WarmFirstPrompt,
    SteadySecondToken,
    SteadyNTokensAfterMaterialization,
    ServerHttpTotal,
    CliTotal,
}

impl BenchmarkMode {
    pub const ALL: [Self; 7] = [
        Self::ColdLoadTotal,
        Self::ColdFirstTokenWithMaterialization,
        Self::WarmFirstPrompt,
        Self::SteadySecondToken,
        Self::SteadyNTokensAfterMaterialization,
        Self::ServerHttpTotal,
        Self::CliTotal,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ColdLoadTotal => "cold_load_total",
            Self::ColdFirstTokenWithMaterialization => "cold_first_token_with_materialization",
            Self::WarmFirstPrompt => "warm_first_prompt",
            Self::SteadySecondToken => "steady_second_token",
            Self::SteadyNTokensAfterMaterialization => "steady_N_tokens_after_materialization",
            Self::ServerHttpTotal => "server_HTTP_total",
            Self::CliTotal => "CLI_total",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OpCounters {
    pub launches: u64,
    pub cublas_calls: u64,
    pub stream_syncs: u64,
    pub stream_waits: u64,
    pub device_allocs: u64,
    pub device_frees: u64,
    pub h2d_copies: u64,
    pub d2h_copies: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
}

impl OpCounters {
    fn add(&mut self, event: ProfileEvent, bytes: u64) {
        match event {
            ProfileEvent::Launch => self.launches += 1,
            ProfileEvent::CublasCall => self.cublas_calls += 1,
            ProfileEvent::StreamSync => self.stream_syncs += 1,
            ProfileEvent::StreamWait => self.stream_waits += 1,
            ProfileEvent::DeviceAlloc => self.device_allocs += 1,
            ProfileEvent::DeviceFree => self.device_frees += 1,
            ProfileEvent::H2DCopy => {
                self.h2d_copies += 1;
                self.h2d_bytes += bytes;
            }
            ProfileEvent::D2HCopy => {
                self.d2h_copies += 1;
                self.d2h_bytes += bytes;
            }
        }
    }

    fn saturating_sub(self, rhs: Self) -> Self {
        Self {
            launches: self.launches.saturating_sub(rhs.launches),
            cublas_calls: self.cublas_calls.saturating_sub(rhs.cublas_calls),
            stream_syncs: self.stream_syncs.saturating_sub(rhs.stream_syncs),
            stream_waits: self.stream_waits.saturating_sub(rhs.stream_waits),
            device_allocs: self.device_allocs.saturating_sub(rhs.device_allocs),
            device_frees: self.device_frees.saturating_sub(rhs.device_frees),
            h2d_copies: self.h2d_copies.saturating_sub(rhs.h2d_copies),
            d2h_copies: self.d2h_copies.saturating_sub(rhs.d2h_copies),
            h2d_bytes: self.h2d_bytes.saturating_sub(rhs.h2d_bytes),
            d2h_bytes: self.d2h_bytes.saturating_sub(rhs.d2h_bytes),
        }
    }

    fn is_zero(self) -> bool {
        self == Self::default()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileSnapshot {
    pub by_op: BTreeMap<&'static str, OpCounters>,
}

#[derive(Debug, Clone, Copy)]
enum ProfileEvent {
    Launch,
    CublasCall,
    StreamSync,
    StreamWait,
    DeviceAlloc,
    DeviceFree,
    H2DCopy,
    D2HCopy,
}

static COUNTERS: OnceLock<Mutex<BTreeMap<&'static str, OpCounters>>> = OnceLock::new();

fn counters() -> &'static Mutex<BTreeMap<&'static str, OpCounters>> {
    COUNTERS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn launch_log_enabled() -> bool {
    std::env::var("M40LLM_LAUNCH_LOG").ok().as_deref() == Some("1")
}

fn sync_log_enabled() -> bool {
    std::env::var("M40LLM_SYNC_LOG").ok().as_deref() == Some("1")
}

fn copy_log_enabled() -> bool {
    std::env::var("M40LLM_COPY_LOG").ok().as_deref() == Some("1")
}

pub fn enabled() -> bool {
    launch_log_enabled()
        || sync_log_enabled()
        || copy_log_enabled()
        || std::env::var("M40LLM_PROFILE_LOG").ok().as_deref() == Some("1")
}

fn record(op: &'static str, event: ProfileEvent, bytes: u64) {
    if let Ok(mut map) = counters().lock() {
        map.entry(op).or_default().add(event, bytes);
    }

    match event {
        ProfileEvent::Launch | ProfileEvent::CublasCall if launch_log_enabled() => {
            eprintln!("[profile] {event:?}: {op}");
        }
        ProfileEvent::StreamSync if sync_log_enabled() => {
            eprintln!("[profile] stream_sync: {op}");
        }
        ProfileEvent::StreamWait if sync_log_enabled() => {
            eprintln!("[profile] stream_wait: {op}");
        }
        ProfileEvent::H2DCopy | ProfileEvent::D2HCopy if copy_log_enabled() => {
            eprintln!("[profile] {event:?}: {op} bytes={bytes}");
        }
        _ => {}
    }
}

pub fn record_launch(op: &'static str) {
    record(op, ProfileEvent::Launch, 0);
}

pub fn record_cublas_call(op: &'static str) {
    record(op, ProfileEvent::CublasCall, 0);
}

pub fn record_stream_sync(op: &'static str) {
    record(op, ProfileEvent::StreamSync, 0);
}

pub fn record_stream_wait(op: &'static str) {
    record(op, ProfileEvent::StreamWait, 0);
}

pub fn record_device_alloc(op: &'static str, bytes: usize) {
    record(op, ProfileEvent::DeviceAlloc, bytes as u64);
}

pub fn record_device_free(op: &'static str) {
    record(op, ProfileEvent::DeviceFree, 0);
}

pub fn record_h2d_copy(op: &'static str, bytes: usize) {
    record(op, ProfileEvent::H2DCopy, bytes as u64);
}

pub fn record_d2h_copy(op: &'static str, bytes: usize) {
    record(op, ProfileEvent::D2HCopy, bytes as u64);
}

pub fn reset() {
    if let Ok(mut map) = counters().lock() {
        map.clear();
    }
}

pub fn snapshot() -> ProfileSnapshot {
    let by_op = counters().lock().map(|map| map.clone()).unwrap_or_default();
    ProfileSnapshot { by_op }
}

pub fn snapshot_if_enabled() -> Option<ProfileSnapshot> {
    enabled().then(snapshot)
}

pub fn log_delta(label: &str, before: Option<&ProfileSnapshot>, elapsed: std::time::Duration) {
    if !enabled() {
        return;
    }
    let Some(before) = before else {
        return;
    };
    let after = snapshot();
    let elapsed_us = elapsed.as_secs_f64() * 1_000_000.0;
    for (op, after_counts) in after.by_op {
        let before_counts = before.by_op.get(op).copied().unwrap_or_default();
        let delta = after_counts.saturating_sub(before_counts);
        if delta.is_zero() {
            continue;
        }
        eprintln!(
            "[profile] {label}: op={op} launches={} cublas_calls={} syncs={} waits={} allocs={} frees={} h2d={}({} bytes) d2h={}({} bytes) elapsed_us={elapsed_us:.3}",
            delta.launches,
            delta.cublas_calls,
            delta.stream_syncs,
            delta.stream_waits,
            delta.device_allocs,
            delta.device_frees,
            delta.h2d_copies,
            delta.h2d_bytes,
            delta.d2h_copies,
            delta.d2h_bytes
        );
    }
}

pub fn log_summary(label: &str) {
    let snapshot = snapshot();
    eprintln!("[profile] summary {label}");
    for (op, counts) in snapshot.by_op {
        eprintln!(
            "[profile] {op}: launches={} cublas_calls={} syncs={} waits={} allocs={} frees={} h2d={}({} bytes) d2h={}({} bytes)",
            counts.launches,
            counts.cublas_calls,
            counts.stream_syncs,
            counts.stream_waits,
            counts.device_allocs,
            counts.device_frees,
            counts.h2d_copies,
            counts.h2d_bytes,
            counts.d2h_copies,
            counts.d2h_bytes
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_modes_match_plan_names() {
        let names: Vec<_> = BenchmarkMode::ALL
            .iter()
            .map(|mode| mode.as_str())
            .collect();
        assert_eq!(
            names,
            vec![
                "cold_load_total",
                "cold_first_token_with_materialization",
                "warm_first_prompt",
                "steady_second_token",
                "steady_N_tokens_after_materialization",
                "server_HTTP_total",
                "CLI_total",
            ]
        );
    }

    #[test]
    fn counters_accumulate_by_operation() {
        reset();
        record_launch("test.kernel");
        record_stream_sync("test.kernel");
        record_stream_wait("test.kernel");
        record_h2d_copy("test.copy", 17);
        let snapshot = snapshot();
        assert_eq!(snapshot.by_op["test.kernel"].launches, 1);
        assert_eq!(snapshot.by_op["test.kernel"].stream_syncs, 1);
        assert_eq!(snapshot.by_op["test.kernel"].stream_waits, 1);
        assert_eq!(snapshot.by_op["test.copy"].h2d_copies, 1);
        assert_eq!(snapshot.by_op["test.copy"].h2d_bytes, 17);
    }
}
