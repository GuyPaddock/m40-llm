use anyhow::{ensure, Context, Result};
use clap::Parser;
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer, ExactOldBacking, KVCache},
    decode_batch::{CudaDecodeBatchPlan, DecodeBatchPlan, DecodeRequestState},
    evaluation::identity::file_sha256,
    kv_compression::{KvCompressionConfig, KvRepresentativePolicy, ScopedRuntimeConfig},
};
use serde_json::{json, Value};
use std::{fs::OpenOptions, io::Write, path::PathBuf, process::Command, time::Instant};

const CAPACITY: u32 = 4096;
const ACTIVE: u32 = 2048;
const Q_HEADS: u32 = 16;
const HEAD: u32 = 128;
const KV_HEADS: u32 = 2;
const ROW: usize = (Q_HEADS * HEAD) as usize;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    output: PathBuf,
}

fn command(program: &str, args: &[&str]) -> Result<String> {
    let value = Command::new(program).args(args).output()?;
    ensure!(value.status.success(), "{program} failed");
    Ok(String::from_utf8(value.stdout)?.trim().to_owned())
}
fn device_state() -> Result<Value> {
    let pid = std::process::id();
    let owners = command(
        "nvidia-smi",
        &[
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
    )?;
    for row in owners.lines() {
        ensure!(
            row.split(',')
                .next()
                .context("GPU process ID")?
                .trim()
                .parse::<u32>()?
                == pid,
            "other GPU owner: {row}"
        );
    }
    Ok(
        json!({"owners":owners,"gpu":command("nvidia-smi",&["--query-gpu=uuid,driver_version,clocks.sm,clocks.mem,pstate,temperature.gpu,memory.used,utilization.gpu","--format=csv,noheader"])?}),
    )
}
fn upload(ctx: &CudaContext, values: &[f32], tag: &str) -> Result<DeviceBuffer> {
    let device = DeviceBuffer::new_tagged(ctx, std::mem::size_of_val(values), tag)?;
    unsafe {
        ctx.memcpy_h2d(device.as_mut_ptr(), values.as_ptr().cast(), device.bytes())?;
    }
    Ok(device)
}
struct Case {
    cache: KVCache,
    plan: Option<CudaDecodeBatchPlan>,
    second: u32,
    slots: u32,
}
impl Case {
    fn new(ctx: &CudaContext, slots: u32, second: u32, both_active: bool) -> Result<Self> {
        let cache = KVCache::new_compressed_with_context(
            ctx,
            CAPACITY,
            slots,
            KV_HEADS,
            HEAD,
            1024,
            32,
            8,
            0,
            KvRepresentativePolicy::Last,
            ExactOldBacking::Fp16KQ4V,
        )?;
        for (slot, len) in [ACTIVE, second]
            .into_iter()
            .enumerate()
            .take(slots as usize)
        {
            let values: Vec<_> = (0..KV_HEADS * HEAD)
                .map(|i| ((i % 37) as f32 - 18.) * 0.015625 + slot as f32)
                .collect();
            let device = upload(ctx, &values, "active_row:append")?;
            ctx.synchronize_stream(CudaStream::Prefill)?;
            for position in 0..len {
                unsafe {
                    cache.append_token_f32_rope_k_at_async(
                        ctx,
                        slot as u32,
                        device.as_ptr(),
                        device.as_ptr(),
                        position,
                        position,
                        10000.,
                        1.,
                    )?;
                }
            }
            ctx.synchronize_stream(CudaStream::Decode)?;
            ensure!(
                cache.debug_sequence_length(ctx, slot as u32)? == len,
                "slot length mismatch"
            );
        }
        let plan = if both_active {
            Some(CudaDecodeBatchPlan::new(
                ctx,
                DecodeBatchPlan::from_requests(&[
                    DecodeRequestState::active(0, 0, ACTIVE),
                    DecodeRequestState::active(1, 1, second),
                ])?,
            )?)
        } else {
            None
        };
        Ok(Self {
            cache,
            plan,
            second,
            slots,
        })
    }
    fn dispatch(&self, ctx: &CudaContext, q: &DeviceBuffer, out: &DeviceBuffer) -> Result<()> {
        unsafe {
            if let Some(plan) = &self.plan {
                plan.dispatch_fp16_k_q4_v_direct_attention_async(
                    ctx,
                    &self.cache,
                    q.as_ptr(),
                    Q_HEADS,
                    1024,
                    32,
                    8,
                    out.as_mut_ptr(),
                )
            } else {
                self.cache
                    .attention_last_token_f32_gqa_block_select_exact_fp16_k_q4_v_old_direct_async(
                        ctx,
                        0,
                        q.as_ptr(),
                        Q_HEADS,
                        ACTIVE,
                        1024,
                        32,
                        8,
                        out.as_mut_ptr(),
                    )
            }
        }
    }
}
pub fn run() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !cfg!(debug_assertions) && command("git", &["status", "--porcelain"])?.is_empty(),
        "clean release required"
    );
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(args.output)?;
    let mut record = |value: Value| -> Result<()> {
        serde_json::to_writer(&mut file, &value)?;
        writeln!(&mut file)?;
        file.flush()?;
        Ok(())
    };
    let _scope = ScopedRuntimeConfig::new(KvCompressionConfig::default());
    let before = device_state()?;
    let ctx = CudaContext::new_m40()?;
    record(
        json!({"kind":"manifest","revision":command("git",&["rev-parse","HEAD"])?,
        "executable_sha256":file_sha256(&std::env::current_exe()?)?,"before":before,"gpu":device_state()?,
        "cuda_compiler":command("nvcc",&["--version"])?,"build":"release sm_52","model":"synthetic operator; Qwen2.5-3B shape",
        "query":"(i mod 23 - 11)/64","kv":"(i mod 37 - 18)/64 + slot; per-position RoPE",
        "active":ACTIVE,"capacity":CAPACITY,"recent":1024,"block":32,"top":8,"q_heads":Q_HEADS,"kv_heads":KV_HEADS,"head":HEAD,
        "environment":std::env::vars().filter(|(k,_)|k.starts_with("M40LLM_")||k.starts_with("CUDA_")).collect::<std::collections::BTreeMap<_,_>>()}),
    )?;
    let mut cases = Vec::new();
    for (slots, other, active) in [
        (1, 0, false),
        (2, 0, false),
        (2, 512, false),
        (2, 2048, false),
        (2, 4096, false),
        (2, 2048, true),
    ] {
        cases.push(Case::new(&ctx, slots, other, active)?);
    }
    let q = upload(
        &ctx,
        &(0..2 * ROW)
            .map(|i| ((i % 23) as f32 - 11.) / 64.)
            .collect::<Vec<_>>(),
        "active_row:q",
    )?;
    let guard = 8191.25;
    let sentinel = vec![guard; 2 * ROW + 16];
    let out = upload(&ctx, &sentinel, "active_row:output")?;
    let mut reference = None;
    for case in &cases {
        unsafe {
            ctx.memcpy_h2d(out.as_mut_ptr(), sentinel.as_ptr().cast(), out.bytes())?;
        }
        ctx.synchronize_stream(CudaStream::Prefill)?;
        for _ in 0..3 {
            case.dispatch(&ctx, &q, &out)?;
        }
        ctx.synchronize_stream(CudaStream::Decode)?;
        let mut actual = vec![0.0f32; sentinel.len()];
        unsafe {
            ctx.memcpy_d2h(actual.as_mut_ptr().cast(), out.as_ptr(), out.bytes())?;
        }
        let written = if case.plan.is_some() { ROW * 2 } else { ROW };
        ensure!(
            actual[..written]
                .iter()
                .all(|v| v.is_finite() && *v != guard)
                && actual[written..].iter().all(|v| *v == guard),
            "row write extent incorrect"
        );
        let bits: Vec<_> = actual[..ROW].iter().map(|v| v.to_bits()).collect();
        ensure!(
            reference.get_or_insert_with(|| bits.clone()) == &bits,
            "idle history or second active request changed row zero"
        );
        if written == 2 * ROW {
            ensure!(
                actual[ROW..written] != actual[..ROW],
                "positive control did not distinguish rows"
            );
        }
        record(
            json!({"kind":"parity","slots":case.slots,"second_tokens":case.second,"active_rows":written/ROW,"exact_first_row_values":ROW,"cache_bytes":case.cache.actual_bytes()}),
        )?;
    }
    let start = ctx.create_event()?;
    let stop = ctx.create_event()?;
    for trial in 0..5 {
        let state = device_state()?;
        for index in 0..cases.len() {
            let order = if trial % 2 == 0 {
                index
            } else {
                cases.len() - 1 - index
            };
            let case = &cases[(order + trial / 2) % cases.len()];
            let began = Instant::now();
            start.record(CudaStream::Decode)?;
            for _ in 0..16 {
                case.dispatch(&ctx, &q, &out)?;
            }
            stop.record(CudaStream::Decode)?;
            let event_ms = start.elapsed_sync(&stop, "active_rows")? / 16.;
            let wall_ms = began.elapsed().as_secs_f64() * 1000. / 16.;
            record(
                json!({"kind":"timing","trial":trial,"order":index,"slots":case.slots,"second_tokens":case.second,
                "active_rows":if case.plan.is_some(){2}else{1},"event_ms":event_ms,"wall_ms":wall_ms,"repeats":16,"gpu_before_round":state}),
            )?;
        }
    }
    record(json!({"kind":"complete","timing_rows":30,"cases":cases.len()}))
}
