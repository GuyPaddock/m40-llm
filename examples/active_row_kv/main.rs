//! Hold active compressed attention fixed while changing another slot's history.
#[cfg(all(feature = "cuda", nvcc))]
mod probe;
#[cfg(all(feature = "cuda", nvcc))]
fn main() -> anyhow::Result<()> {
    probe::run()
}
#[cfg(not(all(feature = "cuda", nvcc)))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("active row test requires real Maxwell CUDA")
}
