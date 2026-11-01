use std::sync::{Mutex, MutexGuard};

static CUDA_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Serialize GPU accounting and process-wide environment changes within one
/// test binary. Separate CUDA test commands must still run serially.
pub fn lock_cuda() -> MutexGuard<'static, ()> {
    CUDA_TEST_LOCK.lock().expect("CUDA test scope poisoned")
}
