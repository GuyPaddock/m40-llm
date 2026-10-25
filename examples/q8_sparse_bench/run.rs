use std::{
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::PathBuf,
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{ensure, Context, Result};
use clap::Parser;
use m40_llm::{
    cuda::{CudaContext, DeviceBuffer},
    gguf::{load_gguf, GgufTensor},
    infer::{
        execution_weights::{Q8SourceIdentity, Q8SourceView},
        sparse_weights::{DeviceSparseQ8Weight, HostSparseQ8Weight, SparseQ8Launch},
    },
};
use serde_json::{json, Value};

use super::{inputs, projection::Projection};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "blk.0.ffn_gate.weight,blk.0.ffn_up.weight,blk.0.ffn_down.weight"
    )]
    tensors: Vec<String>,
    #[arg(long, value_delimiter = ',', default_value = "1,4,16,64")]
    rows: Vec<usize>,
    #[arg(long, default_value_t = 5)]
    trials: usize,
    #[arg(long, default_value_t = 16)]
    iterations: usize,
}

fn command(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program).args(args).output()?;
    ensure!(
        output.status.success(),
        "{program}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn inspect_gpu() -> Result<Value> {
    let processes = command(
        "nvidia-smi",
        &[
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader",
        ],
    )?;
    for line in processes.lines() {
        let pid: u32 = line
            .split(',')
            .next()
            .context("missing GPU PID")?
            .trim()
            .parse()?;
        ensure!(
            pid == std::process::id(),
            "another GPU process invalidates isolated timing: {line}"
        );
    }
    Ok(
        json!({"processes":processes,"device":command("nvidia-smi", &["--query-gpu=uuid,name,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,memory.used,utilization.gpu", "--format=csv,noheader"])?}),
    )
}

fn write(file: &mut File, record: Value) -> Result<()> {
    serde_json::to_writer(&mut *file, &record)?;
    writeln!(file)?;
    file.flush()?;
    Ok(())
}

fn run_rows(
    projection: &Projection<'_>,
    args: &Args,
    file: &mut File,
    tensor: &str,
    dense: &[u8],
    input: &[f32],
) -> Result<()> {
    let iterations = args.iterations.div_ceil(projection.rows).max(2);
    for columns in [1, 4] {
        for threads in [128, 256] {
            let launch = SparseQ8Launch::new(columns, threads)?;
            let errors = projection.check(launch, dense, input)?;
            for _ in 0..3 {
                projection.enqueue(None)?;
                projection.ctx.synchronize_stream(projection.stream(None))?;
                projection.enqueue(Some(launch))?;
                projection
                    .ctx
                    .synchronize_stream(projection.stream(Some(launch)))?;
            }
            for trial in 0..args.trials {
                let gpu_before = inspect_gpu()?;
                let (baseline, candidate) = if trial % 2 == 0 {
                    (
                        projection.time(None, iterations)?,
                        projection.time(Some(launch), iterations)?,
                    )
                } else {
                    let candidate = projection.time(Some(launch), iterations)?;
                    (projection.time(None, iterations)?, candidate)
                };
                write(
                    file,
                    json!({"kind":"timing","tensor":tensor,"shape":projection.shape,"rows":projection.rows,
                "launch":launch,"trial":trial,"iterations":iterations,"baseline":baseline,"candidate":candidate,
                "errors":errors,"storage":projection.sparse.storage(),"gpu_before":gpu_before,"gpu_after":inspect_gpu()?}),
                )?;
            }
            eprintln!(
                "[sparse-q8] {tensor} rows={} launch={launch:?} parity={errors}",
                projection.rows
            );
        }
    }
    Ok(())
}

fn run_tensor(
    ctx: &CudaContext,
    source_file: &mut File,
    data_offset: u64,
    tensor: &GgufTensor,
    digest: [u8; 32],
    args: &Args,
    file: &mut File,
) -> Result<()> {
    let identity = Q8SourceIdentity::from_gguf_tensor(digest, tensor)?;
    let shape = identity.shape();
    let mut source = vec![0; shape.source_bytes()?];
    source_file.seek(SeekFrom::Start(
        data_offset
            .checked_add(tensor.offset)
            .context("tensor offset overflow")?,
    ))?;
    source_file.read_exact(&mut source)?;
    let began = Instant::now();
    let masks = inputs::make_masks(shape.block_count()?);
    let mask_ms = began.elapsed().as_secs_f64() * 1000.0;
    let began = Instant::now();
    let host =
        HostSparseQ8Weight::pack(Q8SourceView::from_tensor_bytes(&identity, &source)?, &masks)?;
    let pack_ms = began.elapsed().as_secs_f64() * 1000.0;
    let began = Instant::now();
    let dense = host.reconstruct_column_blocks();
    let matched_dense_ms = began.elapsed().as_secs_f64() * 1000.0;
    let began = Instant::now();
    let sparse = DeviceSparseQ8Weight::upload(ctx, &host)?;
    let sparse_upload_ms = began.elapsed().as_secs_f64() * 1000.0;
    let d_dense = DeviceBuffer::new_tagged(ctx, dense.len(), "sparse-bench:dense")?;
    let began = Instant::now();
    unsafe {
        ctx.memcpy_h2d(d_dense.as_mut_ptr(), dense.as_ptr().cast(), dense.len())?;
    }
    let dense_upload_ms = began.elapsed().as_secs_f64() * 1000.0;
    let mask_bytes: Vec<u8> = masks.iter().flat_map(|mask| mask.to_le_bytes()).collect();
    write(
        file,
        json!({"kind":"preparation","tensor":tensor,"source_identity":identity,"storage":host.storage(),
        "source_sha256":inputs::hash_bytes(&source),"masks_sha256":inputs::hash_bytes(&mask_bytes),
        "packed_sha256":inputs::hash_bytes(host.blocks()),"matched_dense_sha256":inputs::hash_bytes(&dense),
        "mask_generation_ms":mask_ms,"pack_ms":pack_ms,"matched_dense_ms":matched_dense_ms,
        "sparse_upload_ms":sparse_upload_ms,"dense_upload_ms":dense_upload_ms,
        "benchmark_weight_device_bytes":host.blocks().len()+dense.len(),
        "measured_dram_bytes":Value::Null,"logical_sparsity":1.0-host.storage().retained_logical_values as f64/host.storage().logical_values as f64}),
    )?;
    drop(host);
    for &rows in &args.rows {
        let input = inputs::make_input(rows, shape.input_dim());
        let projection = Projection {
            ctx,
            sparse: &sparse,
            dense: &d_dense,
            shape,
            rows,
            input: DeviceBuffer::new_tagged(ctx, input.len() * 4, "sparse-bench:input")?,
            output: DeviceBuffer::new_tagged(
                ctx,
                rows * shape.output_dim() * 4,
                "sparse-bench:output",
            )?,
            reference: DeviceBuffer::new_tagged(
                ctx,
                rows * shape.output_dim() * 4,
                "sparse-bench:reference",
            )?,
        };
        unsafe {
            ctx.memcpy_h2d(
                projection.input.as_mut_ptr(),
                input.as_ptr().cast(),
                projection.input.bytes(),
            )?;
        }
        run_rows(&projection, args, file, &tensor.name, &dense, &input)?;
    }
    Ok(())
}

pub fn benchmark() -> Result<()> {
    let args = Args::parse();
    ensure!(!cfg!(debug_assertions), "release build required for timing");
    ensure!(
        args.trials >= 5
            && args.iterations > 0
            && !args.rows.is_empty()
            && args.rows.iter().all(|rows| (1..=64).contains(rows))
            && !args.tensors.is_empty(),
        "require >=5 trials, >0 iterations, rows 1..=64 and explicit tensors"
    );
    let gpu = inspect_gpu()?;
    let sha = command(
        "sha256sum",
        &[args.model.to_str().context("non-UTF8 model path")?],
    )?;
    let digest_text = sha.split_whitespace().next().context("missing SHA256")?;
    ensure!(digest_text.len() == 64, "invalid SHA256");
    let mut digest = [0; 32];
    for (i, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&digest_text[i * 2..i * 2 + 2], 16)?;
    }
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args.output)?;
    let control_flags = [
        ("M40LLM_Q8_DECODE_STREAM", "decode"),
        ("M40LLM_Q8_DECODE_SPLIT_QBLOCK", "4"),
        ("M40LLM_Q8_DECODE_COL_TILE", "4"),
        ("M40LLM_Q8_DECODE_TILED_COLS", "1"),
        ("M40LLM_Q8_DECODE_EVENT_LOG", "0"),
    ];
    for (key, value) in control_flags {
        std::env::set_var(key, value);
    }
    let ctx = CudaContext::new(-1)?;
    let props = ctx.current_device_props()?;
    ensure!(
        (props.major, props.minor) == (5, 2),
        "sparse Q8 benchmark requires M40"
    );
    write(
        &mut file,
        json!({"kind":"manifest","model":args.model,"model_sha256":digest_text,
        "revision":command("git", &["rev-parse","HEAD"])?,"worktree":command("git", &["status","--short"])?,
        "created_unix_s":SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        "executable_sha256":command("sha256sum", &[std::env::current_exe()?.to_str().context("non-UTF8 executable")?])?,
        "build_profile":"release","cuda_flags":["-O3","sm_52","compute_52"],"nvcc":command("nvcc", &["--version"])?,
        "gpu":gpu,"rows":args.rows,"trials":args.trials,"iterations":args.iterations,"control_flags":control_flags,
        "mask_seed":inputs::MASK_SEED,"mask_policy":"uniform deterministic 16-of-32 physical positions, independent of weights/prompts",
        "control":"matched-zero GGUF Q8 dispatch: decode coltile4/split4; multirow existing shared activation/blockloop",
        "timing":"CUDA events and wall time per repeated projection; setup excluded; serial alternating paired order",
        "interpretation":"pruned-matrix kernel screening only, no calibrated model, no E2E or measured DRAM traffic"}),
    )?;
    let model = load_gguf(&args.model)?;
    let mut source = File::open(&args.model)?;
    for name in &args.tensors {
        let tensor = model
            .tensors
            .iter()
            .find(|tensor| tensor.name == *name)
            .with_context(|| format!("missing tensor {name}"))?;
        run_tensor(
            &ctx,
            &mut source,
            model.data_offset,
            tensor,
            digest,
            &args,
            &mut file,
        )?;
    }
    write(&mut file, json!({"kind":"complete","gpu":inspect_gpu()?}))?;
    Ok(())
}
