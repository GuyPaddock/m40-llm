#[cfg(all(feature = "cuda", nvcc))]
mod measure;

#[cfg(all(feature = "cuda", nvcc))]
fn main() -> anyhow::Result<()> {
    measure::run()
}

#[cfg(not(all(feature = "cuda", nvcc)))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("graph node timing requires a real NVCC-enabled CUDA build")
}
