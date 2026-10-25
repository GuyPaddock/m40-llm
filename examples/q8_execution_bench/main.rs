#[cfg(all(feature = "cuda", nvcc))]
mod run;

#[cfg(all(feature = "cuda", nvcc))]
fn main() -> anyhow::Result<()> {
    run::benchmark()
}

#[cfg(not(all(feature = "cuda", nvcc)))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("q8_execution_bench requires --features cuda and an M40")
}
