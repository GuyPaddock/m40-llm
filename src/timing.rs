use std::time::Duration;

pub(crate) fn enabled() -> bool {
    std::env::var("M40LLM_TIMING_LOG").ok().as_deref() == Some("1")
}

pub(crate) fn log(label: impl std::fmt::Display, elapsed: Duration) {
    if enabled() {
        eprintln!("[timing] {label} {:.3} ms", elapsed.as_secs_f64() * 1000.0);
    }
}

macro_rules! timing_log {
    ($elapsed:expr, $($arg:tt)*) => {
        if crate::timing::enabled() {
            crate::timing::log(format_args!($($arg)*), $elapsed);
        }
    };
}

pub(crate) use timing_log;
