//! Completed pageable transfer and producer/consumer handoff costs.
#[cfg(all(feature = "cuda", nvcc))]
mod gpu;
#[cfg(all(feature = "cuda", nvcc))]
mod probe;

#[cfg(all(feature = "cuda", nvcc))]
fn main() -> anyhow::Result<()> {
    gpu::run()
}

#[cfg(not(all(feature = "cuda", nvcc)))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("transfer probe requires --features cuda and NVCC")
}
