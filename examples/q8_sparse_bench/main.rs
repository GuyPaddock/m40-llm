#[cfg(all(feature = "cuda", nvcc))]
mod inputs;
#[cfg(all(feature = "cuda", nvcc))]
mod projection;
#[cfg(all(feature = "cuda", nvcc))]
mod run;

#[cfg(all(feature = "cuda", nvcc))]
fn main() -> anyhow::Result<()> {
    run::benchmark()
}

#[cfg(not(all(feature = "cuda", nvcc)))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("q8_sparse_bench requires --features cuda, NVCC and M40 hardware")
}
