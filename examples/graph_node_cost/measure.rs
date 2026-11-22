use anyhow::{ensure, Context, Result};
use m40_llm::cuda::{CudaContext, CudaGraphExec, CudaStream, DeviceBuffer, KVCache};
use serde_json::json;
use std::{
    fs::File,
    io::{BufWriter, Write},
    time::{Duration, Instant},
};

const QUERY_HEADS: u32 = 16;
const KV_HEADS: u32 = 2;
const STREAM: CudaStream = CudaStream::Decode;

fn upload(ctx: &CudaContext, data: &[f32]) -> Result<DeviceBuffer> {
    let bytes = std::mem::size_of_val(data);
    let device = DeviceBuffer::new_tagged(ctx, bytes, "graph-node:fixture")?;
    unsafe { ctx.memcpy_h2d(device.as_mut_ptr(), data.as_ptr().cast(), bytes)? };
    Ok(device)
}

struct Attention {
    // Destroy the captured executable before releasing any address it references.
    graph: CudaGraphExec,
    kv: KVCache,
    _device_length: DeviceBuffer,
    query: DeviceBuffer,
    output: DeviceBuffer,
    len: u32,
}

impl Attention {
    fn capture(ctx: &CudaContext, capacity: u32, head: u32, len: u32) -> Result<Self> {
        let width = (KV_HEADS * head) as usize;
        let kv = KVCache::new_with_context(ctx, capacity, 1, KV_HEADS, head)?;
        let keys: Vec<_> = (0..len as usize * width)
            .map(|i| ((i * 31 + i / 17) % 521) as f32 / 173. - 1.5)
            .collect();
        let values: Vec<_> = (0..keys.len())
            .map(|i| ((i * 13 + i / 7) % 277) as f32 / 89. - 1.5)
            .collect();
        let key_data = upload(ctx, &keys)?;
        let value_data = upload(ctx, &values)?;
        for row in 0..len as usize {
            unsafe {
                kv.append_token_f32_async(
                    ctx,
                    0,
                    key_data.as_ptr().cast::<f32>().add(row * width).cast(),
                    value_data.as_ptr().cast::<f32>().add(row * width).cast(),
                )?;
            }
        }
        ctx.synchronize_stream(STREAM)?;
        let elements = (QUERY_HEADS * head) as usize;
        let q: Vec<_> = (0..elements)
            .map(|i| ((29 * i) % 257) as f32 / 71. - 1.8)
            .collect();
        let query = upload(ctx, &q)?;
        let output = upload(ctx, &vec![f32::NAN; elements])?;
        let device_length = DeviceBuffer::new_tagged(ctx, 4, "graph-node:length")?;
        unsafe {
            ctx.memcpy_h2d(device_length.as_mut_ptr(), (&len as *const u32).cast(), 4)?;
        }
        eprintln!("capture head={head} length={len} capacity={capacity}");
        let graph = ctx.capture_graph(STREAM, || unsafe {
            kv.attention_last_token_f32_gqa_seq_len_dev_async(
                ctx,
                0,
                query.as_ptr(),
                QUERY_HEADS,
                device_length.as_ptr().cast(),
                output.as_mut_ptr(),
            )
        })?;
        Ok(Self {
            graph,
            kv,
            _device_length: device_length,
            query,
            output,
            len,
        })
    }

    fn enqueue(&self, ctx: &CudaContext, graph: bool) -> Result<()> {
        if graph {
            self.graph.launch(STREAM)
        } else {
            unsafe {
                self.kv.attention_last_token_f32_gqa_async(
                    ctx,
                    0,
                    self.query.as_ptr(),
                    QUERY_HEADS,
                    self.len,
                    self.output.as_mut_ptr(),
                )
            }
        }
    }

    fn checked_output(&self, ctx: &CudaContext, graph: bool, head: u32) -> Result<Vec<u32>> {
        let mut words = vec![f32::NAN.to_bits(); (head * QUERY_HEADS) as usize];
        let bytes = std::mem::size_of_val(words.as_slice());
        unsafe { ctx.memcpy_h2d(self.output.as_mut_ptr(), words.as_ptr().cast(), bytes)? };
        self.enqueue(ctx, graph)?;
        ctx.synchronize_stream(STREAM)?;
        unsafe { ctx.memcpy_d2h(words.as_mut_ptr().cast(), self.output.as_ptr(), bytes)? };
        ensure!(
            words.iter().all(|&w| f32::from_bits(w).is_finite()),
            "incomplete output"
        );
        ensure!(
            self.kv.debug_sequence_length(ctx, 0)? == self.len,
            "KV length changed"
        );
        Ok(words)
    }
}

fn trial(ctx: &CudaContext, arm: &Attention, graph: bool) -> Result<serde_json::Value> {
    const ITERATIONS: usize = 100;
    let start = ctx.create_event()?;
    let stop = ctx.create_event()?;
    let wall_start = Instant::now();
    start.record(STREAM)?;
    let enqueue_start = Instant::now();
    for _ in 0..ITERATIONS {
        arm.enqueue(ctx, graph)?;
    }
    let cpu = enqueue_start.elapsed();
    stop.record(STREAM)?;
    let event = f64::from(start.elapsed_sync(&stop, "graph-node-batch")?);
    let wall = wall_start.elapsed();
    Ok(
        json!({"iterations":ITERATIONS,"enqueue_us":cpu.as_secs_f64()*1e6/ITERATIONS as f64,
        "event_span_us":event*1000./ITERATIONS as f64,
        "wall_us":wall.as_secs_f64()*1e6/ITERATIONS as f64}),
    )
}

fn scenario(ctx: &CudaContext, out: &mut impl Write, head: u32, len: u32) -> Result<()> {
    let arms = [
        Attention::capture(ctx, 8192, head, len)?,
        Attention::capture(ctx, 8193, head, len)?,
    ];
    let expected = arms[0].checked_output(ctx, false, head)?;
    for arm in &arms {
        for graph in [false, true] {
            ensure!(
                arm.checked_output(ctx, graph, head)? == expected,
                "capacity/dispatch output bits differ"
            );
            let warm = Instant::now();
            while warm.elapsed() < Duration::from_secs(1) {
                for _ in 0..16 {
                    arm.enqueue(ctx, graph)?;
                }
                ctx.synchronize_stream(STREAM)?;
            }
        }
        writeln!(
            out,
            "{}",
            json!({"kind":"verified","head_dim":head,"live_length":len,
            "capacity":arm.kv.max_seq_len(),"cache_bytes":arm.kv.actual_bytes(),"exact_outputs":expected.len()})
        )?;
    }
    for round in 0..10 {
        for position in 0..4 {
            let index = ((if round % 2 == 0 {
                position
            } else {
                3 - position
            }) + round / 2)
                % 4;
            let arm = &arms[index / 2];
            let graph = index % 2 == 1;
            writeln!(
                out,
                "{}",
                json!({"kind":"timing","round":round,"order":position,"head_dim":head,
                "live_length":len,"capacity":arm.kv.max_seq_len(),"graph":graph,"timing":trial(ctx,arm,graph)?})
            )?;
        }
    }
    for arm in &arms {
        ensure!(
            arm.checked_output(ctx, true, head)? == expected,
            "repeated replay changed output"
        );
    }
    out.flush()?;
    Ok(())
}

pub(super) fn run() -> Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .context("supply a new output JSONL")?;
    let file = File::create_new(path)?;
    let mut out = BufWriter::new(file);
    let ctx = CudaContext::new_m40()?;
    let props = ctx.current_device_props()?;
    ensure!((props.major, props.minor) == (5, 2), "requires sm_52");
    let initial = CudaContext::total_device_bytes();
    for (head, len) in [(128, 65), (7, 65), (128, 2048)] {
        scenario(&ctx, &mut out, head, len)?;
        ensure!(
            CudaContext::total_device_bytes() == initial,
            "tracked storage after trial"
        );
    }
    writeln!(
        out,
        "{}",
        json!({"kind":"complete","timings":120,"tracked_after":initial})
    )?;
    out.flush()?;
    Ok(())
}
