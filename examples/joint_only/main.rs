//! Independent ownership qualification; never changes the frozen reference.
#[cfg(all(feature = "cuda", feature = "server", nvcc))]
mod compact_draft;
#[cfg(all(feature = "cuda", feature = "server", nvcc))]
mod dataflow;
#[cfg(all(feature = "cuda", feature = "server", nvcc))]
#[path = "../q8_execution_bench/ffn/reductions/data.rs"]
#[allow(dead_code)]
mod guarded_words;
#[cfg(all(feature = "cuda", feature = "server", nvcc))]
mod http;
#[cfg(all(feature = "cuda", feature = "server", nvcc))]
#[path = "../q8_execution_bench/ffn/packet.rs"]
mod packet;
#[cfg(all(feature = "cuda", feature = "server", nvcc))]
mod portfolio;
#[cfg(all(feature = "cuda", feature = "server", nvcc))]
mod portfolio_logits;
#[cfg(all(feature = "cuda", feature = "server", nvcc))]
#[path = "../compare_ollama/protocol.rs"]
#[allow(dead_code)]
mod protocol;
#[cfg(all(feature = "cuda", feature = "server", nvcc))]
mod quality;

#[cfg(all(feature = "cuda", feature = "server", nvcc))]
mod run {
    use anyhow::{ensure, Context, Result};
    use clap::{Parser, ValueEnum};
    use m40_llm::{
        cuda::CudaContext,
        evaluation::identity::file_sha256,
        gguf,
        infer::{
            execution_weights::Q8ExecutionConfig, q8_prefill::Q8PrefillBackend, LoadedModel,
            Q8AttentionProjectionBackend, Q8FfnConfig, Q8FfnOwnership,
        },
        profile,
    };
    use serde_json::{json, Value};
    use std::{
        fs::{self, File, OpenOptions},
        io::Write,
        path::{Path, PathBuf},
        time::Instant,
    };

    #[derive(Clone, Copy, ValueEnum)]
    pub enum Phase {
        Quality,
        Http,
        Load,
        Dataflow,
        PortfolioLogits,
        PortfolioOperators,
        PortfolioQuality,
        PortfolioHttp,
        CompactDraftCost,
    }

    #[derive(Parser)]
    pub struct Args {
        #[arg(long)]
        pub model: PathBuf,
        #[arg(long)]
        pub output: PathBuf,
        #[arg(long, value_enum)]
        pub phase: Phase,
        #[arg(long, default_value = "retain-split")]
        pub ownership: String,
        #[arg(long)]
        pub quality_report: Option<PathBuf>,
        /// New sampled-event output; never append to historical evidence.
        #[arg(long)]
        pub profile_report: Option<PathBuf>,
    }

    pub struct Report(File);
    impl Report {
        pub fn create(path: &Path) -> Result<Self> {
            Ok(Self(
                OpenOptions::new().write(true).create_new(true).open(path)?,
            ))
        }
        pub fn record(&mut self, value: Value) -> Result<()> {
            serde_json::to_writer(&mut self.0, &value)?;
            writeln!(self.0)?;
            self.0.flush()?;
            Ok(())
        }
    }

    pub fn command(program: &str, args: &[&str]) -> Result<String> {
        let result = std::process::Command::new(program).args(args).output()?;
        ensure!(
            result.status.success(),
            "{program} failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        Ok(String::from_utf8(result.stdout)?.trim().to_owned())
    }

    pub fn gpu() -> Result<Value> {
        let device = std::env::var("CUDA_VISIBLE_DEVICES")?;
        let processes = command(
            "nvidia-smi",
            &[
                "-i",
                &device,
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
        )?;
        ensure!(
            processes.lines().all(|s| s
                .split(',')
                .nth(1)
                .and_then(|p| p.trim().parse::<u32>().ok())
                == Some(std::process::id())),
            "competing GPU workload: {processes}"
        );
        Ok(
            json!({"processes":processes,"clocks":command("nvidia-smi", &["-i",&device,"--query-gpu=uuid,pstate,clocks.sm,clocks.mem","--format=csv,noheader,nounits"])?}),
        )
    }

    pub fn load(path: &Path, ownership: Q8FfnOwnership) -> Result<LoadedModel> {
        let gguf = gguf::load_gguf(path)?;
        let mut model = LoadedModel::from_gguf_with_q8_runtime(
            gguf,
            fs::read(path)?,
            -1,
            Q8ExecutionConfig::parse("split", Some("6144"), Some("gate-up,down,vocabulary"))?,
            Q8FfnConfig::joint_shared_scales().with_ownership(ownership)?,
            Q8PrefillBackend::DequantSgemm,
            Q8AttentionProjectionBackend::PackedWarp256,
        )?;
        ensure!(
            model.model_config.block_count == 36
                && model.model_config.embedding_length == 2048
                && model.model_config.attention_key_length == 128,
            "requires frozen Qwen3B architecture"
        );
        model.allocate_kv_cache_for_layers(1024)?;
        Ok(model)
    }

    pub fn identity(model: &LoadedModel) -> Value {
        json!({"ownership":model.q8_ffn.ownership(),"logical_sources":model.q8_execution.logical_sources(),
            "resident_split":model.q8_execution_stats(),"joint":model.q8_ffn_pair_stats(),
            "uploaded_gguf_bytes":model.weights_len,"kv_bytes":model.kv_cache.as_ref().unwrap().actual_bytes(),
            "tracked_device_bytes":CudaContext::total_device_bytes()})
    }

    pub async fn execute() -> Result<()> {
        let args = Args::parse();
        ensure!(!cfg!(debug_assertions), "use a release build");
        let frozen: Value = serde_json::from_slice(&fs::read(
            "docs/experiments/2026-10-27-compound-reference.json",
        )?)?;
        for (key, value) in frozen["configuration"]["environment"]
            .as_object()
            .context("frozen environment")?
        {
            ensure!(
                std::env::var(key).ok().as_deref() == value.as_str(),
                "restore frozen {key}"
            );
        }
        ensure!(
            !std::env::vars()
                .any(|(k, _)| k.starts_with("M40LLM_KV_") || k == "M40LLM_SERVER_BATCH_DECODE"),
            "disable unrelated diagnostics/batching"
        );
        ensure!(
            file_sha256(&args.model)? == frozen["model"]["sha256"],
            "model hash differs from reference"
        );
        for (key, _) in std::env::vars().filter(|(key, _)| key.starts_with("M40LLM_")) {
            ensure!(
                frozen["configuration"]["environment"].get(&key).is_some()
                    || (matches!(args.phase, Phase::Load) && key == "M40LLM_ALLOC_LOG"),
                "unexpected experiment environment setting {key}"
            );
        }
        let worktree = command("git", &["status", "--short"])?;
        ensure!(
            worktree.is_empty(),
            "commit implementation before qualification"
        );
        let mut report = Report::create(&args.output)?;
        report.record(json!({"kind":"joint_only_manifest","schema_version":1,"reference_id":"compound-runtime-63240c5-v1",
            "revision":command("git", &["rev-parse","HEAD"])?,"worktree":worktree,
            "executable_sha256":file_sha256(&std::env::current_exe()?)?,"model_sha256":frozen["model"]["sha256"],
            "build_profile":"release","frozen_environment":frozen["configuration"]["environment"],"gpu":gpu()?}))?;
        match args.phase {
            Phase::Quality => super::quality::execute(&args, &mut report)?,
            Phase::Http => super::http::execute(&args, &mut report).await?,
            Phase::Dataflow => super::dataflow::execute(&args, &mut report).await?,
            Phase::PortfolioLogits => super::portfolio_logits::execute(&args, &mut report)?,
            Phase::PortfolioOperators => super::portfolio_logits::execute(&args, &mut report)?,
            Phase::PortfolioQuality => super::portfolio::quality::execute(&args, &mut report)?,
            Phase::PortfolioHttp => super::portfolio::http::execute(&args, &mut report).await?,
            Phase::CompactDraftCost => super::compact_draft::execute(&args, &mut report)?,
            Phase::Load => {
                let before = profile::snapshot();
                let initial = CudaContext::total_device_bytes();
                let began = Instant::now();
                let model = load(&args.model, Q8FfnOwnership::parse(&args.ownership)?)?;
                let elapsed = began.elapsed().as_secs_f64() * 1000.;
                let status = fs::read_to_string("/proc/self/status")?;
                report.record(json!({"kind":"joint_only_load","model":identity(&model),"elapsed_ms":elapsed,
                    "tracked_device_delta":CudaContext::total_device_bytes()-initial,"counters":profile::counter_delta(&before,&profile::snapshot()),
                    "host_high_water":status.lines().find(|s|s.starts_with("VmHWM:"))}))?;
                drop(model);
                ensure!(
                    CudaContext::total_device_bytes() == initial,
                    "load leaked allocations"
                );
            }
        }
        Ok(())
    }
}

#[cfg(all(feature = "cuda", feature = "server", nvcc))]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    run::execute().await
}

#[cfg(not(all(feature = "cuda", feature = "server", nvcc)))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("joint-only qualification requires CUDA, NVCC and server")
}
