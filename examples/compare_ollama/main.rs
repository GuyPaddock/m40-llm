#[cfg(any(test, all(feature = "cuda", feature = "server", nvcc)))]
mod protocol;
#[cfg(all(feature = "cuda", feature = "server", nvcc))]
#[path = "../common/q8_runtime.rs"]
mod q8_runtime;
#[cfg(all(feature = "cuda", feature = "server", nvcc))]
mod run;

#[cfg(all(feature = "cuda", feature = "server", nvcc))]
#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> anyhow::Result<()> {
    run::compare().await
}

#[cfg(not(all(feature = "cuda", feature = "server", nvcc)))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("compare_ollama requires --release --features cuda,server and an NVCC build")
}
