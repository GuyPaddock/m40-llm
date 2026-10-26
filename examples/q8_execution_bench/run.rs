use std::{
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::PathBuf,
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{ensure, Context, Result};
use clap::{Parser, ValueEnum};
use half::f16;
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer},
    gguf::{load_gguf, GgufTensor},
    infer::execution_weights::{
        DeviceQ8ExecutionView, DeviceQ8ExecutionWeight, HostQ8ExecutionWeight, Q8ExecutionLayout,
        Q8SeparateLaunch, Q8SourceIdentity, Q8SourceView, Q8TiledLaunch,
    },
};
use serde_json::{json, Value};

#[path = "ffn/mod.rs"]
mod ffn;
#[path = "sparse_inputs.rs"]
mod sparse_inputs;

#[derive(Debug, Clone, Copy, ValueEnum, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
enum LayoutArg {
    Separate,
    InputTiled,
}

impl LayoutArg {
    fn layouts(self) -> Vec<Q8ExecutionLayout> {
        match self {
            Self::Separate => vec![Q8ExecutionLayout::SeparateScalesValues],
            Self::InputTiled => [4, 8, 16]
                .map(|columns| Q8ExecutionLayout::InputTiled { columns })
                .into(),
        }
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
#[serde(untagged)]
enum ProjectionLaunch {
    Separate(Q8SeparateLaunch),
    Tiled(Q8TiledLaunch),
}

fn configurations(layout: Q8ExecutionLayout) -> Result<Vec<ProjectionLaunch>> {
    match layout {
        Q8ExecutionLayout::SeparateScalesValues => [1, 4]
            .into_iter()
            .flat_map(|columns| {
                [128, 256].map(move |threads| {
                    Q8SeparateLaunch::new(columns, threads).map(ProjectionLaunch::Separate)
                })
            })
            .collect(),
        Q8ExecutionLayout::InputTiled { columns } => [128, 256]
            .map(|threads| Q8TiledLaunch::new(columns as u32, threads).map(ProjectionLaunch::Tiled))
            .into_iter()
            .collect(),
        Q8ExecutionLayout::ColumnBlocks => {
            anyhow::bail!("column blocks are the control, not a candidate")
        }
    }
}

#[derive(Parser)]
struct Args {
    /// Stage original scales in shared FP32 memory; retain arithmetic order.
    #[arg(long,requires="ffn_inputs",conflicts_with_all=["ffn_f32_role_parallel","ffn_payload_entropy","ffn_f32_native_warp","ffn_f32_task_pairs","prepared_warp_packet","prepared_weight_read","ffn_weight_read","ffn_f32_eight_values","ffn_f32_cohorts","decode_reductions","qkv_f32_split","output_f32_split","ffn_f32_persistent_tiles","ffn_f32_virtual_warp","prepared_f32_packed_activation","prepared_f32_warp","qkv_f32_packed","output_f32_packed","qkv_f32_warp","output_f32_warp","qkv_warp_report","ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes","ffn_f32_split_two_columns","ffn_f32_joint","ffn_f32_fixed_k","ffn_f32_unfused","ffn_f32_weight_cg","ffn_f32_byte_permute"])]
    ffn_f32_shared_scales: bool,
    /// Separate physical gate/up roles, retaining original FP32 reduction trees.
    #[arg(long,requires="ffn_inputs",conflicts_with_all=["ffn_payload_entropy","ffn_f32_native_warp","ffn_f32_task_pairs","prepared_warp_packet","prepared_weight_read","ffn_weight_read","ffn_f32_eight_values","ffn_f32_cohorts","decode_reductions","qkv_f32_split","output_f32_split","ffn_f32_persistent_tiles","ffn_f32_virtual_warp","prepared_f32_packed_activation","prepared_f32_warp","qkv_f32_packed","output_f32_packed","qkv_f32_warp","output_f32_warp","qkv_warp_report","ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes","ffn_f32_split_two_columns","ffn_f32_joint","ffn_f32_fixed_k","ffn_f32_unfused","ffn_f32_weight_cg","ffn_f32_byte_permute"])]
    ffn_f32_role_parallel: bool,
    /// Offline lossless payload size feasibility; never creates a CUDA model.
    #[arg(long,requires="ffn_inputs",conflicts_with_all=["ffn_f32_native_warp","ffn_f32_task_pairs","prepared_warp_packet","prepared_weight_read","ffn_weight_read","ffn_f32_eight_values","ffn_f32_cohorts","decode_reductions","qkv_f32_split","output_f32_split","ffn_f32_persistent_tiles","ffn_f32_virtual_warp","prepared_f32_packed_activation","prepared_f32_warp","qkv_f32_packed","output_f32_packed","qkv_f32_warp","output_f32_warp","qkv_warp_report","ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes","ffn_f32_split_two_columns","ffn_f32_joint","ffn_f32_fixed_k","ffn_f32_unfused","ffn_f32_weight_cg","ffn_f32_byte_permute"])]
    ffn_payload_entropy: bool,
    /// Native 32-partial output warps; FP32 association differs from control.
    #[arg(long,requires="ffn_inputs",conflicts_with_all=["ffn_f32_task_pairs","prepared_warp_packet","prepared_weight_read","ffn_weight_read","ffn_f32_eight_values","ffn_f32_cohorts","decode_reductions","qkv_f32_split","output_f32_split","ffn_f32_persistent_tiles","ffn_f32_virtual_warp","prepared_f32_packed_activation","prepared_f32_warp","qkv_f32_packed","output_f32_packed","qkv_f32_warp","output_f32_warp","qkv_warp_report","ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes","ffn_f32_split_two_columns","ffn_f32_joint","ffn_f32_fixed_k","ffn_f32_unfused","ffn_f32_weight_cg","ffn_f32_byte_permute"])]
    ffn_f32_native_warp: bool,
    /// Fixed logical column visitation, original storage and block geometry.
    #[arg(
        long,
        requires = "ffn_weight_read",
        conflicts_with = "ffn_weight_read_gap"
    )]
    ffn_weight_read_column_order: bool,
    /// Fixed public-library segmented checksum reference; never inference output.
    #[arg(long, requires = "ffn_weight_read", conflicts_with_all = ["ffn_weight_read_gap", "ffn_weight_read_column_order"])]
    ffn_weight_read_cub: bool,
    /// Add one fixed 128-byte payload column-gap read diagnostic.
    #[arg(long, requires = "ffn_weight_read")]
    ffn_weight_read_gap: bool,
    /// Two original per-thread FFN tasks in one payload; scale planes unchanged.
    #[arg(long,requires="ffn_inputs",conflicts_with_all=["prepared_warp_packet","prepared_weight_read","ffn_weight_read","ffn_f32_eight_values","ffn_f32_cohorts","decode_reductions","qkv_f32_split","output_f32_split","ffn_f32_persistent_tiles","ffn_f32_virtual_warp","prepared_f32_packed_activation","prepared_f32_warp","qkv_f32_packed","output_f32_packed","qkv_f32_warp","output_f32_warp","qkv_warp_report","ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes","ffn_f32_split_two_columns","ffn_f32_joint","ffn_f32_fixed_k","ffn_f32_unfused","ffn_f32_weight_cg","ffn_f32_byte_permute"])]
    ffn_f32_task_pairs: bool,
    /// One lossless cross-column warp packet with original arithmetic/tree.
    #[arg(long, requires="ffn_inputs", conflicts_with_all=["prepared_weight_read","ffn_weight_read","ffn_f32_eight_values","ffn_f32_cohorts","decode_reductions","qkv_f32_split","output_f32_split","ffn_f32_persistent_tiles","ffn_f32_virtual_warp","prepared_f32_packed_activation","prepared_f32_warp","qkv_f32_packed","output_f32_packed","qkv_f32_warp","output_f32_warp","qkv_warp_report","ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes","ffn_f32_split_two_columns","ffn_f32_joint","ffn_f32_fixed_k","ffn_f32_unfused","ffn_f32_weight_cg","ffn_f32_byte_permute"])]
    prepared_warp_packet: bool,
    /// Diagnostic read/checksum of the original prepared down/vocabulary mapping.
    #[arg(long, requires="ffn_inputs", conflicts_with_all=["ffn_weight_read","ffn_f32_eight_values","ffn_f32_cohorts","decode_reductions","qkv_f32_split","output_f32_split","ffn_f32_persistent_tiles","ffn_f32_virtual_warp","prepared_f32_packed_activation","prepared_f32_warp","qkv_f32_packed","output_f32_packed","qkv_f32_warp","output_f32_warp","qkv_warp_report","ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes","ffn_f32_split_two_columns","ffn_f32_joint","ffn_f32_fixed_k","ffn_f32_unfused","ffn_f32_weight_cg","ffn_f32_byte_permute"])]
    prepared_weight_read: bool,
    /// Diagnostic read/checksum of actual gate/up planes, not inference output.
    #[arg(long, requires="ffn_inputs", conflicts_with_all=["ffn_f32_eight_values","ffn_f32_cohorts","decode_reductions","qkv_f32_split","output_f32_split","ffn_f32_persistent_tiles","ffn_f32_virtual_warp","prepared_f32_packed_activation","prepared_f32_warp","qkv_f32_packed","output_f32_packed","qkv_f32_warp","output_f32_warp","qkv_warp_report","ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes","ffn_f32_split_two_columns","ffn_f32_joint","ffn_f32_fixed_k","ffn_f32_unfused","ffn_f32_weight_cg","ffn_f32_byte_permute"])]
    ffn_weight_read: bool,
    /// Eight-value payload tasks with FP32 activations and changed sum ownership.
    #[arg(long, requires="ffn_inputs", conflicts_with_all=["ffn_f32_cohorts","decode_reductions","qkv_f32_split","output_f32_split","ffn_f32_persistent_tiles","ffn_f32_virtual_warp","prepared_f32_packed_activation","prepared_f32_warp","qkv_f32_packed","output_f32_packed","qkv_f32_warp","output_f32_warp","qkv_warp_report","ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes","ffn_f32_split_two_columns","ffn_f32_joint","ffn_f32_fixed_k","ffn_f32_unfused","ffn_f32_weight_cg","ffn_f32_byte_permute"])]
    ffn_f32_eight_values: bool,
    /// Two independent physical 128-thread output cohorts per 256-thread CTA.
    #[arg(long, requires="ffn_inputs", conflicts_with_all=["decode_reductions","qkv_f32_split","output_f32_split","ffn_f32_persistent_tiles","ffn_f32_virtual_warp","prepared_f32_packed_activation","prepared_f32_warp","qkv_f32_packed","output_f32_packed","qkv_f32_warp","output_f32_warp","qkv_warp_report","ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes","ffn_f32_split_two_columns","ffn_f32_joint","ffn_f32_fixed_k","ffn_f32_unfused","ffn_f32_weight_cg","ffn_f32_byte_permute"])]
    ffn_f32_cohorts: bool,
    /// Independent exact weighted-norm and two-launch vocabulary argmax screen.
    #[arg(long, requires="ffn_inputs", conflicts_with_all=["qkv_f32_split","output_f32_split","ffn_f32_persistent_tiles","ffn_f32_virtual_warp","prepared_f32_packed_activation","prepared_f32_warp","qkv_f32_packed","output_f32_packed","qkv_f32_warp","output_f32_warp","qkv_warp_report","ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes","ffn_f32_split_two_columns","ffn_f32_joint","ffn_f32_fixed_k","ffn_f32_unfused","ffn_f32_weight_cg","ffn_f32_byte_permute"])]
    decode_reductions: bool,
    /// Existing split QKV planes versus qualified packed raw-GGUF projections.
    #[arg(long, requires="ffn_inputs", conflicts_with_all=["output_f32_split","ffn_f32_persistent_tiles","ffn_f32_virtual_warp","prepared_f32_packed_activation","prepared_f32_warp","qkv_f32_packed","output_f32_packed","qkv_f32_warp","output_f32_warp","ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes","ffn_f32_split_two_columns","ffn_f32_joint","ffn_f32_fixed_k","ffn_f32_unfused","ffn_f32_weight_cg","ffn_f32_byte_permute"])]
    qkv_f32_split: bool,
    /// Existing split output planes versus qualified packed coltile4 projection.
    #[arg(long, requires="ffn_inputs", conflicts_with_all=["qkv_f32_split","ffn_f32_persistent_tiles","ffn_f32_virtual_warp","prepared_f32_packed_activation","prepared_f32_warp","qkv_f32_packed","output_f32_packed","qkv_f32_warp","output_f32_warp","ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes","ffn_f32_split_two_columns","ffn_f32_joint","ffn_f32_fixed_k","ffn_f32_unfused","ffn_f32_weight_cg","ffn_f32_byte_permute"])]
    output_f32_split: bool,
    /// Reuse 192 CTAs per row with the qualified per-column FP32 tree.
    #[arg(long, requires="ffn_inputs", conflicts_with_all=["ffn_f32_virtual_warp","prepared_f32_packed_activation","prepared_f32_warp","qkv_f32_packed","output_f32_packed","qkv_f32_warp","output_f32_warp","ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes","ffn_f32_split_two_columns","ffn_f32_joint","ffn_f32_fixed_k","ffn_f32_unfused","ffn_f32_weight_cg","ffn_f32_byte_permute"])]
    ffn_f32_persistent_tiles: bool,
    /// Four output warps preserving the qualified virtual 128-thread FP32 tree.
    #[arg(long, requires="ffn_inputs", conflicts_with_all=["prepared_f32_packed_activation","prepared_f32_warp","qkv_f32_packed","output_f32_packed","qkv_f32_warp","output_f32_warp","ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes","ffn_f32_split_two_columns","ffn_f32_joint","ffn_f32_fixed_k","ffn_f32_unfused","ffn_f32_weight_cg","ffn_f32_byte_permute"])]
    ffn_f32_virtual_warp: bool,
    /// Fixed vector activation loads, original prepared shared reduction.
    #[arg(long, requires="ffn_inputs", conflicts_with_all=["prepared_f32_warp","qkv_f32_packed","output_f32_packed","qkv_f32_warp","output_f32_warp","ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes","ffn_f32_split_two_columns","ffn_f32_joint","ffn_f32_fixed_k","ffn_f32_unfused","ffn_f32_weight_cg","ffn_f32_byte_permute"])]
    prepared_f32_packed_activation: bool,
    /// Fixed packed loads versus qualified scalar-load QKV warp reduction.
    #[arg(long, requires="ffn_inputs", conflicts_with_all=["output_f32_packed","prepared_f32_warp","qkv_f32_warp","output_f32_warp","ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes","ffn_f32_split_two_columns","ffn_f32_joint","ffn_f32_fixed_k","ffn_f32_unfused","ffn_f32_weight_cg","ffn_f32_byte_permute"])]
    qkv_f32_packed: bool,
    /// Independent packed output loads versus qualified coltile4 warp reduction.
    #[arg(long, requires="ffn_inputs", conflicts_with_all=["qkv_f32_packed","prepared_f32_warp","qkv_f32_warp","output_f32_warp","ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes","ffn_f32_split_two_columns","ffn_f32_joint","ffn_f32_fixed_k","ffn_f32_unfused","ffn_f32_weight_cg","ffn_f32_byte_permute"])]
    output_f32_packed: bool,
    /// One exact signed-byte conversion comparison on the qualified split FFN.
    #[arg(long, requires="ffn_inputs", conflicts_with_all=["prepared_f32_warp","qkv_f32_warp","output_f32_warp","ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes","ffn_f32_split_two_columns","ffn_f32_joint","ffn_f32_fixed_k","ffn_f32_unfused","ffn_f32_weight_cg"])]
    ffn_f32_byte_permute: bool,
    /// One fused split-plane cache-policy comparison with L2-only weight loads.
    #[arg(long, requires="ffn_inputs", conflicts_with_all=["prepared_f32_warp","qkv_f32_warp","output_f32_warp","ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes","ffn_f32_split_two_columns","ffn_f32_joint","ffn_f32_fixed_k","ffn_f32_unfused"])]
    ffn_f32_weight_cg: bool,
    /// One fixed unfused gate/up/SwiGLU strategy on the qualified split planes.
    #[arg(long, requires="ffn_inputs", conflicts_with_all=["prepared_f32_warp","qkv_f32_warp","output_f32_warp","ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes","ffn_f32_split_two_columns","ffn_f32_joint","ffn_f32_fixed_k"])]
    ffn_f32_unfused: bool,
    /// Fixed warp finish for prepared down and vocabulary, with unchanged FFN.
    #[arg(long, requires="ffn_inputs", conflicts_with_all=["qkv_f32_warp","output_f32_warp","ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes","ffn_f32_split_two_columns","ffn_f32_joint","ffn_f32_fixed_k"])]
    prepared_f32_warp: bool,
    /// Complete FFN baseline from frozen paired inputs, not synthetic projections.
    #[arg(long, requires="ffn_capture_report", conflicts_with_all=["sparse_inputs","layout","tensors"])]
    ffn_inputs: Option<PathBuf>,
    #[arg(long, requires = "ffn_inputs")]
    ffn_capture_report: Option<PathBuf>,
    /// Pair the five fixed M03 FP32 kernels against fresh native FFN controls.
    #[arg(long, requires = "ffn_inputs")]
    ffn_f32_ablation: bool,
    /// Pair the two Q8_1 numerical candidates, including activation encoding.
    #[arg(long, requires = "ffn_inputs", conflicts_with = "ffn_f32_ablation")]
    ffn_q81: bool,
    /// One 128-thread scale-factor candidate versus qualified packed FP32/128.
    #[arg(long, requires="ffn_inputs", conflicts_with_all=["ffn_f32_ablation","ffn_q81","ffn_f32_coalesced_pairs"])]
    ffn_f32_factored_scales: bool,
    /// One 128-thread adjacent-pair mapping versus qualified packed FP32/128.
    #[arg(long, requires="ffn_inputs", conflicts_with_all=["ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales"])]
    ffn_f32_coalesced_pairs: bool,
    /// One 128-thread/four-value fused FFN on existing split weight planes.
    #[arg(long, requires="ffn_inputs", conflicts_with_all=["ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs"])]
    ffn_f32_split_planes: bool,
    /// One two-column/128-thread split kernel against qualified one-column split.
    #[arg(long, requires="ffn_inputs", conflicts_with_all=["ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes"])]
    ffn_f32_split_two_columns: bool,
    /// One lossless paired gate/up layout against qualified one-column split.
    #[arg(long, requires="ffn_inputs", conflicts_with_all=["ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes","ffn_f32_split_two_columns"])]
    ffn_f32_joint: bool,
    /// One K=2048 address/loop specialization of the qualified split kernel.
    #[arg(long, requires="ffn_inputs", conflicts_with_all=["ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes","ffn_f32_split_two_columns","ffn_f32_joint"])]
    ffn_f32_fixed_k: bool,
    /// Complete native QKV warp reduction using frozen normalized input vectors.
    #[arg(long, requires="ffn_inputs", conflicts_with_all=["ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes","ffn_f32_split_two_columns","ffn_f32_joint","ffn_f32_fixed_k"])]
    qkv_f32_warp: bool,
    /// Fixed four-column output reduction, with the prespecified QKV peer.
    #[arg(long, requires_all=["ffn_inputs","qkv_warp_report"], conflicts_with_all=["qkv_f32_warp","ffn_f32_ablation","ffn_q81","ffn_f32_factored_scales","ffn_f32_coalesced_pairs","ffn_f32_split_planes","ffn_f32_split_two_columns","ffn_f32_joint","ffn_f32_fixed_k"])]
    output_f32_warp: bool,
    #[arg(long, requires = "output_f32_warp")]
    qkv_warp_report: Option<PathBuf>,
    #[arg(
        long,
        default_value = "data/research_quality/v1.json",
        requires = "ffn_inputs"
    )]
    ffn_corpus: PathBuf,
    /// Screen sparse completed-SwiGLU inputs, not a lossless layout change.
    #[arg(long)]
    sparse_inputs: bool,
    #[arg(long, value_enum, default_value = "separate")]
    layout: LayoutArg,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "blk.0.attn_q.weight,blk.0.attn_k.weight,blk.0.attn_v.weight,blk.0.attn_output.weight,blk.0.ffn_gate.weight,blk.0.ffn_up.weight,blk.0.ffn_down.weight,token_embd.weight"
    )]
    tensors: Vec<String>,
    #[arg(long, value_delimiter = ',', default_value = "1,4")]
    rows: Vec<usize>,
    #[arg(long, default_value_t = 5)]
    trials: usize,
    #[arg(long, default_value_t = 10)]
    iterations: usize,
}

fn command_output(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program).args(args).output()?;
    ensure!(
        output.status.success(),
        "{program}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn inspect_gpu() -> Result<Value> {
    let processes = command_output(
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
    Ok(json!({
        "processes": processes,
        "device": command_output("nvidia-smi", &["--query-gpu=uuid,name,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,memory.used,utilization.gpu", "--format=csv,noheader"])?
    }))
}

fn write_record(file: &mut File, record: Value) -> Result<()> {
    serde_json::to_writer(&mut *file, &record)?;
    writeln!(file)?;
    file.flush()?;
    Ok(())
}

fn read_output(ctx: &CudaContext, buffer: &DeviceBuffer) -> Result<Vec<f32>> {
    let mut output = vec![0.0; buffer.bytes() / 4];
    unsafe {
        ctx.memcpy_d2h(output.as_mut_ptr().cast(), buffer.as_ptr(), buffer.bytes())?;
    }
    Ok(output)
}

struct Projection<'a> {
    ctx: CudaContext,
    original: &'a DeviceQ8ExecutionWeight,
    execution: &'a DeviceQ8ExecutionWeight,
    input: DeviceBuffer,
    output: DeviceBuffer,
    reference: DeviceBuffer,
    rows: usize,
}

impl Projection<'_> {
    unsafe fn enqueue(&self, config: Option<ProjectionLaunch>) -> Result<()> {
        if let Some(config) = config {
            match config {
                ProjectionLaunch::Separate(config) => self.execution.project_separate_async(
                    &self.input,
                    &self.output,
                    self.rows,
                    config,
                ),
                ProjectionLaunch::Tiled(config) => {
                    self.execution
                        .project_tiled_async(&self.input, &self.output, self.rows, config)
                }
            }
        } else {
            let DeviceQ8ExecutionView::ColumnBlocks { key, blocks } = self.original.view() else {
                unreachable!()
            };
            let shape = key.source.shape();
            self.ctx.gemm_f32xq8_0_gguf_f32_async(
                self.input.as_ptr(),
                blocks.as_ptr(),
                self.reference.as_mut_ptr(),
                self.rows as i32,
                shape.output_dim() as i32,
                shape.input_dim() as i32,
            )
        }
    }

    fn stream(&self, config: Option<ProjectionLaunch>) -> CudaStream {
        if config.is_some() || self.rows == 1 {
            CudaStream::Decode
        } else {
            CudaStream::Prefill
        }
    }

    fn time(&self, config: Option<ProjectionLaunch>, iterations: usize) -> Result<Value> {
        let start = self.ctx.create_event()?;
        let stop = self.ctx.create_event()?;
        let started_unix_us = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
        let began = Instant::now();
        start.record(self.stream(config))?;
        for _ in 0..iterations {
            unsafe {
                self.enqueue(config)?;
            }
        }
        stop.record(self.stream(config))?;
        let gpu_ms =
            f64::from(start.elapsed_sync(&stop, "q8_execution_bench")?) / iterations as f64;
        let wall_ms = began.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        let completed_unix_us = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
        Ok(
            json!({"gpu_ms":gpu_ms,"wall_ms":wall_ms,"started_unix_us":started_unix_us,
                "completed_unix_us":completed_unix_us}),
        )
    }

    fn check(&self, blocks: &[u8], input: &[f32], config: ProjectionLaunch) -> Result<Value> {
        unsafe {
            self.enqueue(None)?;
        }
        self.ctx.synchronize_stream(self.stream(None))?;
        unsafe {
            self.enqueue(Some(config))?;
        }
        self.ctx.synchronize_stream(CudaStream::Decode)?;
        let actual = read_output(&self.ctx, &self.output)?;
        let expected = read_output(&self.ctx, &self.reference)?;
        let shape = self.original.key().source.shape();
        let (k, n) = (shape.input_dim(), shape.output_dim());
        let mut maximum = 0.0_f64;
        let mut total = 0.0;
        for (&a, &b) in actual.iter().zip(&expected) {
            ensure!(
                a.is_finite() && b.is_finite(),
                "non-finite projection result"
            );
            let difference = (f64::from(a) - f64::from(b)).abs();
            maximum = maximum.max(difference);
            total += difference;
            ensure!(
                difference < 1e-3 + f64::from(b.abs()) * 2e-5,
                "execution projection drift {difference}"
            );
        }
        let mut cpu_max = 0.0_f64;
        for row in [0, self.rows - 1] {
            for sample in 0..8 {
                let column = sample * (n - 1) / 7;
                let cpu: f64 = (0..k)
                    .map(|index| {
                        let base = (column * k.div_ceil(32) + index / 32) * 34;
                        let scale =
                            f16::from_bits(u16::from_le_bytes([blocks[base], blocks[base + 1]]))
                                .to_f64();
                        f64::from(input[row * k + index])
                            * scale
                            * f64::from(blocks[base + 2 + index % 32] as i8)
                    })
                    .sum();
                let difference = (f64::from(actual[row * n + column]) - cpu).abs();
                cpu_max = cpu_max.max(difference);
                ensure!(
                    difference < 1e-3 + cpu.abs() * 2e-5,
                    "CPU sample drift {difference}"
                );
            }
        }
        Ok(
            json!({"max_abs_diff":maximum,"mean_abs_diff":total/actual.len() as f64,"cpu_sample_max_abs_diff":cpu_max}),
        )
    }
}

fn measure_rows(
    projection: &Projection<'_>,
    file: &mut File,
    args: &Args,
    bytes: &[u8],
    input: &[f32],
    tensor: &GgufTensor,
) -> Result<()> {
    let rows = projection.rows;
    let shape = projection.execution.key().source.shape();
    let layout = projection.execution.key().layout;
    let resident_bytes = projection.execution.plan().resident_bytes();
    let iterations = args.iterations.div_ceil(rows).max(2);
    for config in configurations(layout)? {
        let errors = projection.check(bytes, input, config)?;
        for _ in 0..3 {
            unsafe {
                projection.enqueue(None)?;
            }
            projection.ctx.synchronize_stream(projection.stream(None))?;
            unsafe {
                projection.enqueue(Some(config))?;
            }
            projection.ctx.synchronize_stream(CudaStream::Decode)?;
        }
        for trial in 0..args.trials {
            let gpu_before = inspect_gpu()?;
            let (baseline, candidate) = if trial % 2 == 0 {
                (
                    projection.time(None, iterations)?,
                    projection.time(Some(config), iterations)?,
                )
            } else {
                let candidate = projection.time(Some(config), iterations)?;
                (projection.time(None, iterations)?, candidate)
            };
            write_record(
                file,
                json!({"kind":"timing","tensor":tensor.name,"shape":shape,"rows":rows,
                "execution_layout":layout,"config":config,"trial":trial,"iterations":iterations,
                "baseline":baseline,"candidate":candidate,"errors":errors,
                "resident_execution_bytes":resident_bytes,"source_bytes":bytes.len(),
                "gpu_before":gpu_before,"gpu_after":inspect_gpu()?}),
            )?;
        }
        eprintln!("[q8_execution_bench] tensor={} rows={rows} layout={layout:?} config={config:?} parity={errors}",tensor.name);
    }
    Ok(())
}

fn run_tensor(
    ctx: &CudaContext,
    file: &mut File,
    source_file: &mut File,
    data_offset: u64,
    tensor: &GgufTensor,
    digest: [u8; 32],
    args: &Args,
) -> Result<()> {
    let identity = Q8SourceIdentity::from_gguf_tensor(digest, tensor)?;
    let shape = identity.shape();
    let mut bytes = vec![0; shape.source_bytes()?];
    source_file.seek(SeekFrom::Start(
        data_offset
            .checked_add(tensor.offset)
            .context("file offset overflow")?,
    ))?;
    source_file.read_exact(&mut bytes)?;
    let source = Q8SourceView::from_tensor_bytes(&identity, &bytes)?;
    let start = Instant::now();
    let original = HostQ8ExecutionWeight::copy_column_blocks(source)?;
    let original_copy_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    let original_device = DeviceQ8ExecutionWeight::upload(ctx, &original)?;
    let original_upload_ms = start.elapsed().as_secs_f64() * 1000.0;
    for layout in args.layout.layouts().into_iter().filter(|layout| {
        !args.sparse_inputs || matches!(layout, Q8ExecutionLayout::InputTiled { columns: 8 | 16 })
    }) {
        let start = Instant::now();
        let execution = match layout {
            Q8ExecutionLayout::SeparateScalesValues => {
                HostQ8ExecutionWeight::separate_scales_values(source)?
            }
            Q8ExecutionLayout::InputTiled { columns } => {
                HostQ8ExecutionWeight::tile_inputs(source, columns)?
            }
            Q8ExecutionLayout::ColumnBlocks => unreachable!(),
        };
        let repack_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let execution_device = DeviceQ8ExecutionWeight::upload(ctx, &execution)?;
        let execution_upload_ms = start.elapsed().as_secs_f64() * 1000.0;
        let resident_bytes = execution.plan().resident_bytes();
        write_record(
            file,
            json!({"kind":"preparation","tensor":tensor,"source_bytes":bytes.len(),"execution_layout":layout,
        "resident_execution_bytes":resident_bytes,"extra_resident_bytes":resident_bytes as i64 - bytes.len() as i64,
        "benchmark_weight_device_bytes":resident_bytes+bytes.len(),
        "host_peak_payload_bytes":bytes.len()+original.plan().resident_bytes()+resident_bytes,
        "original_copy_ms":original_copy_ms,"repack_ms":repack_ms,
        "original_upload_ms":original_upload_ms,"execution_upload_ms":execution_upload_ms}),
        )?;
        drop(execution);
        for &rows in &args.rows {
            let input: Vec<f32> = (0..rows * shape.input_dim())
                .map(|i| ((i * 13 % 127) as f32 - 63.0) * 0.0078125)
                .collect();
            let projection = Projection {
                ctx: ctx.clone(),
                original: &original_device,
                execution: &execution_device,
                input: DeviceBuffer::new_tagged(ctx, input.len() * 4, "q8_bench:input")?,
                output: DeviceBuffer::new_tagged(
                    ctx,
                    rows * shape.output_dim() * 4,
                    "q8_bench:output",
                )?,
                reference: DeviceBuffer::new_tagged(
                    ctx,
                    rows * shape.output_dim() * 4,
                    "q8_bench:reference",
                )?,
                rows,
            };
            unsafe {
                ctx.memcpy_h2d(
                    projection.input.as_mut_ptr(),
                    input.as_ptr().cast(),
                    input.len() * 4,
                )?;
            }
            if args.sparse_inputs {
                sparse_inputs::measure(&projection, file, args, &bytes, &input, tensor)?;
            } else {
                measure_rows(&projection, file, args, &bytes, &input, tensor)?;
            }
        }
    }
    Ok(())
}

pub fn benchmark() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.trials >= 5
            && args.iterations > 0
            && !args.rows.is_empty()
            && args.rows.iter().all(|&rows| (1..=64).contains(&rows)),
        "require >=5 trials, >0 iterations and rows in 1..=64"
    );
    if args.ffn_inputs.is_some() {
        return ffn::benchmark(&args);
    }
    if args.sparse_inputs {
        ensure!(
            !cfg!(debug_assertions) && matches!(args.layout, LayoutArg::InputTiled)
                && args.rows == [1] && !args.tensors.is_empty()
                && args.tensors.iter().all(|name| name.ends_with(".ffn_down.weight")),
            "sparse-input screen requires release, --layout input-tiled, --rows 1 and explicit FFN down tensors"
        );
    }
    let mut output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args.output)?;
    let gpu = inspect_gpu()?;
    let sha = command_output(
        "sha256sum",
        &[args.model.to_str().context("non-UTF8 model path")?],
    )?;
    let digest_text = sha.split_whitespace().next().context("missing digest")?;
    ensure!(digest_text.len() == 64, "invalid SHA256");
    let mut digest = [0; 32];
    for (i, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&digest_text[i * 2..i * 2 + 2], 16)?;
    }
    // Fix only the original projection control in this isolated process.
    for (key, value) in [
        ("M40LLM_Q8_DECODE_STREAM", "decode"),
        ("M40LLM_Q8_DECODE_SPLIT_QBLOCK", "4"),
        ("M40LLM_Q8_DECODE_COL_TILE", "4"),
        ("M40LLM_Q8_DECODE_TILED_COLS", "1"),
        ("M40LLM_Q8_DECODE_EVENT_LOG", "0"),
    ] {
        std::env::set_var(key, value);
    }
    let ctx = CudaContext::new(-1)?;
    let props = ctx.current_device_props()?;
    ensure!(
        props.major == 5 && props.minor == 2,
        "Q8 execution benchmark requires M40"
    );
    write_record(
        &mut output,
        json!({"kind":"manifest","model":args.model,"model_sha256":digest_text,
        "revision":command_output("git", &["rev-parse","HEAD"])?,
        "created_unix_s":SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        "executable_sha256":command_output("sha256sum",&[std::env::current_exe()?.to_str().context("non-UTF8 executable path")?])?,
        "worktree":command_output("git", &["status","--short"])?,
        "build_profile":if cfg!(debug_assertions) {"dev"} else {"release"},
        "cuda_flags":["-O3","sm_52","compute_52"],"nvcc":command_output("nvcc", &["--version"])?,
        "gpu":gpu,"trials":args.trials,"rows":args.rows,"requested_iterations":args.iterations,"execution_layout":args.layout,
        "tiled_access":"vector","sparse_inputs":args.sparse_inputs,
        "rustc":command_output("rustc", &["--version"])?,
        "host_compiler":command_output("g++", &["--version"])?,
        "environment":std::env::vars().filter(|(key,_)| key.starts_with("M40LLM_") || key.starts_with("CUDA_")).collect::<std::collections::BTreeMap<_,_>>(),
        "control":"GGUF dispatch: decode split4/coltile4; multirow existing shared-activation/blockloop",
        "timing":"CUDA events per repeated operation, preparation excluded; alternating serial baseline/candidate pairs",
        "interpretation":if args.sparse_inputs {
            "synthetic activation-sparsity prerequisite screen; matched-zero native baseline, selection plus projection and separate stages; not calibrated model quality, E2E inference or measured DRAM traffic"
        } else {"isolated lossless projection experiment, not end-to-end inference or measured DRAM traffic"}}),
    )?;
    let model = load_gguf(&args.model)?;
    let mut source_file = File::open(&args.model)?;
    for name in &args.tensors {
        let tensor = model
            .tensors
            .iter()
            .find(|t| &t.name == name)
            .with_context(|| format!("missing tensor {name}"))?;
        run_tensor(
            &ctx,
            &mut output,
            &mut source_file,
            model.data_offset,
            tensor,
            digest,
            &args,
        )?;
    }
    drop(ctx);
    if args.sparse_inputs {
        write_record(
            &mut output,
            json!({"kind":"sparse_input_benchmark_complete",
            "tensors":args.tensors,"tracked_device_bytes_after_release":CudaContext::total_device_bytes(),
            "release_scope":"tracked device payloads; primary CUDA context may remain until process exit",
            "gpu_after_release":inspect_gpu()?}),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod argument_tests {
    use super::*;

    /// Offline payload analysis requires frozen provenance and cannot select
    /// a CUDA timing experiment or become an implicit inference mode.
    #[test]
    fn test_payload_entropy_requires_exclusive_capture() {
        let base = ["bench", "--model", "model.gguf", "--output", "out.jsonl"];
        let parse = |extra: &[&str]| Args::try_parse_from([base.as_slice(), extra].concat());
        assert!(!parse(&[]).unwrap().ffn_payload_entropy);
        assert!(parse(&["--ffn-payload-entropy"]).is_err());
        let valid = [
            "--ffn-inputs",
            "inputs.json",
            "--ffn-capture-report",
            "capture.jsonl",
            "--ffn-payload-entropy",
        ];
        assert!(parse(&valid).unwrap().ffn_payload_entropy);
        for other in [
            "--ffn-f32-native-warp",
            "--ffn-f32-task-pairs",
            "--ffn-weight-read",
            "--ffn-q81",
        ] {
            assert!(parse(&[valid.as_slice(), &[other]].concat()).is_err());
        }
    }

    /// Shared scales require captured inputs and cannot combine probes.
    #[test]
    fn test_shared_scales_requires_an_exclusive_capture() {
        let base = ["bench", "--model", "model.gguf", "--output", "out.jsonl"];
        let parse = |extra: &[&str]| Args::try_parse_from([base.as_slice(), extra].concat());
        assert!(!parse(&[]).unwrap().ffn_f32_shared_scales);
        assert!(parse(&["--ffn-f32-shared-scales"]).is_err());
        let capture = [
            "--ffn-inputs",
            "inputs.json",
            "--ffn-capture-report",
            "capture.jsonl",
            "--ffn-f32-shared-scales",
        ];
        assert!(parse(&capture).unwrap().ffn_f32_shared_scales);
        for other in [
            "--ffn-f32-role-parallel",
            "--ffn-payload-entropy",
            "--ffn-f32-native-warp",
            "--ffn-weight-read",
            "--ffn-f32-ablation",
            "--ffn-q81",
        ] {
            assert!(parse(&[capture.as_slice(), &[other]].concat()).is_err());
        }
    }

    /// Role-parallel work requires captured inputs and cannot combine probes.
    #[test]
    fn test_role_parallel_requires_an_exclusive_capture() {
        let base = ["bench", "--model", "model.gguf", "--output", "out.jsonl"];
        let parse = |extra: &[&str]| Args::try_parse_from([base.as_slice(), extra].concat());
        assert!(!parse(&[]).unwrap().ffn_f32_role_parallel);
        assert!(parse(&["--ffn-f32-role-parallel"]).is_err());
        let capture = [
            "--ffn-inputs",
            "inputs.json",
            "--ffn-capture-report",
            "capture.jsonl",
            "--ffn-f32-role-parallel",
        ];
        assert!(parse(&capture).unwrap().ffn_f32_role_parallel);
        for other in [
            "--ffn-payload-entropy",
            "--ffn-f32-native-warp",
            "--ffn-f32-cohorts",
            "--ffn-f32-virtual-warp",
            "--ffn-f32-eight-values",
            "--ffn-f32-task-pairs",
            "--ffn-weight-read",
            "--prepared-warp-packet",
            "--ffn-f32-ablation",
            "--ffn-q81",
        ] {
            assert!(parse(&[capture.as_slice(), &[other]].concat()).is_err());
        }
    }

    /// Native warp arithmetic is an explicit captured-input experiment,
    /// never an implicit replacement or combination with a closed mapping.
    #[test]
    fn test_native_warp_requires_an_exclusive_capture() {
        let base = ["bench", "--model", "model.gguf", "--output", "out.jsonl"];
        let parse = |extra: &[&str]| Args::try_parse_from([base.as_slice(), extra].concat());
        assert!(!parse(&[]).unwrap().ffn_f32_native_warp);
        assert!(parse(&["--ffn-f32-native-warp"]).is_err());
        let capture = [
            "--ffn-inputs",
            "inputs.json",
            "--ffn-capture-report",
            "capture.jsonl",
            "--ffn-f32-native-warp",
        ];
        assert!(parse(&capture).unwrap().ffn_f32_native_warp);
        for other in [
            "--ffn-f32-virtual-warp",
            "--ffn-f32-eight-values",
            "--ffn-f32-task-pairs",
            "--ffn-weight-read",
            "--prepared-warp-packet",
            "--ffn-f32-ablation",
            "--ffn-q81",
        ] {
            assert!(parse(&[capture.as_slice(), &[other]].concat()).is_err());
        }
    }

    /// CUB is an explicit read reference, never an implicit or combined experiment.
    #[test]
    fn test_cub_requires_exclusive_read_diagnostics() {
        let base = ["bench", "--model", "model.gguf", "--output", "out.jsonl"];
        let parse = |extra: &[&str]| Args::try_parse_from([base.as_slice(), extra].concat());
        assert!(!parse(&[]).unwrap().ffn_weight_read_cub);
        assert!(parse(&["--ffn-weight-read-cub"]).is_err());
        let valid = [
            "--ffn-inputs",
            "inputs.json",
            "--ffn-capture-report",
            "capture.jsonl",
            "--ffn-weight-read",
            "--ffn-weight-read-cub",
        ];
        assert!(parse(&valid).unwrap().ffn_weight_read_cub);
        for other in [
            "--ffn-weight-read-gap",
            "--ffn-weight-read-column-order",
            "--ffn-q81",
            "--ffn-f32-shared-scales",
        ] {
            assert!(parse(&[valid.as_slice(), &[other]].concat()).is_err());
        }
    }

    /// Column order is a read-only diagnostic, exclusive of physical gaps.
    #[test]
    fn test_column_order_requires_exclusive_read_diagnostics() {
        let base = ["bench", "--model", "model.gguf", "--output", "out.jsonl"];
        let parse = |extra: &[&str]| Args::try_parse_from([base.as_slice(), extra].concat());
        assert!(!parse(&[]).unwrap().ffn_weight_read_column_order);
        assert!(parse(&["--ffn-weight-read-column-order"]).is_err());
        let valid = [
            "--ffn-inputs",
            "inputs.json",
            "--ffn-capture-report",
            "capture.jsonl",
            "--ffn-weight-read",
            "--ffn-weight-read-column-order",
        ];
        assert!(parse(&valid).unwrap().ffn_weight_read_column_order);
        for other in [
            "--ffn-weight-read-gap",
            "--ffn-f32-role-parallel",
            "--ffn-q81",
        ] {
            assert!(parse(&[valid.as_slice(), &[other]].concat()).is_err());
        }
    }

    /// A column-gap diagnostic requires the explicit read experiment and
    /// captured provenance; ordinary inference/other experiments cannot select it.
    #[test]
    fn test_column_gap_requires_the_read_diagnostic() {
        let base = ["bench", "--model", "model.gguf", "--output", "out.jsonl"];
        let parse = |extra: &[&str]| Args::try_parse_from([base.as_slice(), extra].concat());
        assert!(!parse(&[]).unwrap().ffn_weight_read_gap);
        assert!(parse(&["--ffn-weight-read-gap"]).is_err());
        let valid = [
            "--ffn-inputs",
            "inputs.json",
            "--ffn-capture-report",
            "capture.jsonl",
            "--ffn-weight-read",
            "--ffn-weight-read-gap",
        ];
        assert!(parse(&valid).unwrap().ffn_weight_read_gap);
        assert!(parse(&[valid.as_slice(), &["--ffn-f32-task-pairs"]].concat()).is_err());
    }

    /// Original-thread task packets require captured inputs and cannot
    /// silently combine with a different arithmetic or storage experiment.
    #[test]
    fn test_task_pairs_require_an_exclusive_capture() {
        let base = ["bench", "--model", "model.gguf", "--output", "out.jsonl"];
        let capture = [
            "--ffn-inputs",
            "inputs.json",
            "--ffn-capture-report",
            "capture.jsonl",
        ];
        let parse = |extra: &[&str]| Args::try_parse_from([base.as_slice(), extra].concat());
        assert!(!parse(&[]).unwrap().ffn_f32_task_pairs);
        assert!(parse(&["--ffn-f32-task-pairs"]).is_err());
        assert!(
            parse(&[capture.as_slice(), &["--ffn-f32-task-pairs"]].concat())
                .unwrap()
                .ffn_f32_task_pairs
        );
        for other in [
            "--prepared-warp-packet",
            "--prepared-weight-read",
            "--ffn-weight-read",
            "--prepared-f32-warp",
            "--prepared-f32-packed-activation",
            "--ffn-f32-eight-values",
            "--ffn-f32-cohorts",
            "--decode-reductions",
            "--qkv-f32-split",
            "--output-f32-split",
            "--ffn-f32-ablation",
            "--ffn-q81",
        ] {
            assert!(
                parse(&[capture.as_slice(), &["--ffn-f32-task-pairs", other]].concat()).is_err()
            );
        }
    }

    /// Packet storage is an exclusive captured-input experiment and cannot
    /// activate by default or combine with a closed arithmetic/layout probe.
    #[test]
    fn test_warp_packets_require_an_exclusive_capture() {
        let base = ["bench", "--model", "model.gguf", "--output", "out.jsonl"];
        let capture = [
            "--ffn-inputs",
            "inputs.json",
            "--ffn-capture-report",
            "capture.jsonl",
        ];
        let parse = |extra: &[&str]| Args::try_parse_from([base.as_slice(), extra].concat());
        assert!(!parse(&[]).unwrap().prepared_warp_packet);
        assert!(parse(&["--prepared-warp-packet"]).is_err());
        assert!(
            parse(&[capture.as_slice(), &["--prepared-warp-packet"]].concat())
                .unwrap()
                .prepared_warp_packet
        );
        for other in [
            "--prepared-weight-read",
            "--ffn-weight-read",
            "--prepared-f32-warp",
            "--prepared-f32-packed-activation",
            "--ffn-f32-eight-values",
            "--ffn-f32-cohorts",
            "--decode-reductions",
            "--qkv-f32-split",
            "--output-f32-split",
            "--ffn-f32-ablation",
            "--ffn-q81",
        ] {
            assert!(
                parse(&[capture.as_slice(), &["--prepared-warp-packet", other]].concat()).is_err()
            );
        }
    }

    /// Prepared read diagnostics require captured provenance and cannot
    /// combine with arithmetic candidates or another cost diagnostic.
    #[test]
    fn test_prepared_reads_require_an_exclusive_capture() {
        let base = ["bench", "--model", "model.gguf", "--output", "out.jsonl"];
        let capture = [
            "--ffn-inputs",
            "inputs.json",
            "--ffn-capture-report",
            "capture.jsonl",
        ];
        let parse = |extra: &[&str]| Args::try_parse_from([base.as_slice(), extra].concat());
        assert!(!parse(&[]).unwrap().prepared_weight_read);
        assert!(parse(&["--prepared-weight-read"]).is_err());
        assert!(
            parse(&[capture.as_slice(), &["--prepared-weight-read"]].concat())
                .unwrap()
                .prepared_weight_read
        );
        for other in [
            "--ffn-weight-read",
            "--prepared-f32-warp",
            "--prepared-f32-packed-activation",
            "--ffn-f32-eight-values",
            "--ffn-f32-cohorts",
            "--decode-reductions",
            "--qkv-f32-split",
            "--output-f32-split",
            "--ffn-f32-ablation",
            "--ffn-q81",
        ] {
            assert!(
                parse(&[capture.as_slice(), &["--prepared-weight-read", other]].concat()).is_err()
            );
        }
    }

    /// Read/checksum mode requires captured provenance and cannot silently
    /// replace or combine with any inference kernel experiment.
    #[test]
    fn test_weight_read_requires_an_exclusive_capture() {
        let base = ["bench", "--model", "model.gguf", "--output", "out.jsonl"];
        let capture = [
            "--ffn-inputs",
            "inputs.json",
            "--ffn-capture-report",
            "capture.jsonl",
        ];
        let parse = |extra: &[&str]| Args::try_parse_from([base.as_slice(), extra].concat());
        assert!(!parse(&[]).unwrap().ffn_weight_read);
        assert!(parse(&["--ffn-weight-read"]).is_err());
        assert!(
            parse(&[capture.as_slice(), &["--ffn-weight-read"]].concat())
                .unwrap()
                .ffn_weight_read
        );
        for mode in [
            "--ffn-f32-eight-values",
            "--ffn-f32-cohorts",
            "--decode-reductions",
            "--qkv-f32-split",
            "--output-f32-split",
            "--ffn-f32-persistent-tiles",
            "--ffn-f32-virtual-warp",
            "--prepared-f32-packed-activation",
            "--prepared-f32-warp",
            "--qkv-f32-packed",
            "--output-f32-packed",
            "--qkv-f32-warp",
            "--output-f32-warp",
            "--ffn-f32-ablation",
            "--ffn-q81",
            "--ffn-f32-factored-scales",
            "--ffn-f32-coalesced-pairs",
            "--ffn-f32-split-planes",
            "--ffn-f32-split-two-columns",
            "--ffn-f32-joint",
            "--ffn-f32-fixed-k",
            "--ffn-f32-unfused",
            "--ffn-f32-weight-cg",
            "--ffn-f32-byte-permute",
        ] {
            assert!(parse(&[capture.as_slice(), &["--ffn-weight-read", mode]].concat()).is_err());
        }
    }

    /// Split storage has its own captured-input comparison and cannot combine
    /// with an older control, arithmetic experiment or another projection.
    #[test]
    fn test_split_projection_modes_require_exclusive_captured_inputs() {
        let base = ["bench", "--model", "model.gguf", "--output", "new.jsonl"];
        let capture = [
            "--ffn-inputs",
            "inputs.json",
            "--ffn-capture-report",
            "capture.jsonl",
        ];
        let parse = |extra: &[&str]| Args::try_parse_from([base.as_slice(), extra].concat());
        assert!(!parse(&[]).unwrap().qkv_f32_split);
        assert!(!parse(&[]).unwrap().output_f32_split);
        for mode in ["--qkv-f32-split", "--output-f32-split"] {
            assert!(parse(&[mode]).is_err());
            let args = parse(&[capture.as_slice(), &[mode]].concat()).unwrap();
            assert_eq!(args.qkv_f32_split, mode == "--qkv-f32-split");
            assert_eq!(args.output_f32_split, mode == "--output-f32-split");
            for other in [
                "--qkv-f32-split",
                "--output-f32-split",
                "--ffn-f32-persistent-tiles",
                "--ffn-f32-virtual-warp",
                "--prepared-f32-packed-activation",
                "--prepared-f32-warp",
                "--qkv-f32-packed",
                "--output-f32-packed",
                "--qkv-f32-warp",
                "--output-f32-warp",
                "--ffn-f32-ablation",
                "--ffn-q81",
                "--ffn-f32-factored-scales",
                "--ffn-f32-coalesced-pairs",
                "--ffn-f32-split-planes",
                "--ffn-f32-split-two-columns",
                "--ffn-f32-joint",
                "--ffn-f32-fixed-k",
                "--ffn-f32-unfused",
                "--ffn-f32-weight-cg",
                "--ffn-f32-byte-permute",
            ] {
                if other != mode {
                    assert!(parse(&[capture.as_slice(), &[mode, other]].concat()).is_err());
                }
            }
        }
    }

    /// Work ownership is opt-in, requires captured-input provenance, and
    /// cannot accidentally combine with a closed experiment or peer screen.
    /// Physical cohorts keep the same exclusive capture boundary.
    /// Wider FP32 tasks cannot combine with any closed ownership mapping.
    #[test]
    fn test_work_ownership_screens_require_an_exclusive_capture() {
        let base = ["bench", "--model", "model.gguf", "--output", "out.jsonl"];
        let capture = [
            "--ffn-inputs",
            "inputs.json",
            "--ffn-capture-report",
            "capture.jsonl",
        ];
        let parse = |extra: &[&str]| Args::try_parse_from([base.as_slice(), extra].concat());
        assert!(!parse(&[]).unwrap().ffn_f32_virtual_warp);
        assert!(!parse(&[]).unwrap().ffn_f32_persistent_tiles);
        assert!(!parse(&[]).unwrap().ffn_f32_cohorts);
        assert!(!parse(&[]).unwrap().ffn_f32_eight_values);
        for mode in [
            "--ffn-f32-virtual-warp",
            "--ffn-f32-persistent-tiles",
            "--ffn-f32-cohorts",
            "--ffn-f32-eight-values",
        ] {
            assert!(parse(&[mode]).is_err());
            let args = parse(&[capture.as_slice(), &[mode]].concat()).unwrap();
            assert_eq!(args.ffn_f32_virtual_warp, mode == "--ffn-f32-virtual-warp");
            assert_eq!(args.ffn_f32_cohorts, mode == "--ffn-f32-cohorts");
            assert_eq!(args.ffn_f32_eight_values, mode == "--ffn-f32-eight-values");
            assert_eq!(
                args.ffn_f32_persistent_tiles,
                mode == "--ffn-f32-persistent-tiles"
            );
            for other in [
                "--ffn-f32-eight-values",
                "--ffn-f32-cohorts",
                "--decode-reductions",
                "--qkv-f32-split",
                "--output-f32-split",
                "--ffn-f32-virtual-warp",
                "--ffn-f32-persistent-tiles",
                "--prepared-f32-packed-activation",
                "--prepared-f32-warp",
                "--qkv-f32-packed",
                "--output-f32-packed",
                "--qkv-f32-warp",
                "--output-f32-warp",
                "--ffn-f32-ablation",
                "--ffn-q81",
                "--ffn-f32-factored-scales",
                "--ffn-f32-coalesced-pairs",
                "--ffn-f32-split-planes",
                "--ffn-f32-split-two-columns",
                "--ffn-f32-joint",
                "--ffn-f32-fixed-k",
                "--ffn-f32-unfused",
                "--ffn-f32-weight-cg",
                "--ffn-f32-byte-permute",
            ] {
                if other != mode {
                    assert!(parse(&[capture.as_slice(), &[mode, other]].concat()).is_err());
                }
            }
        }
    }

    /// FFN modes require provenance, exclude conflicting arithmetic experiments,
    /// and leave the original projection CLI valid. Two-column split work
    /// cannot silently combine with old controls or another ablation. Output
    /// reduction additionally requires its immutable QKV peer evidence.
    /// Cache-policy, byte-conversion and packed-projection work require
    /// provenance and are exclusive. Packed output is independent of QKV peers.
    /// Prepared activation loads are mutually exclusive with reduction probes.
    /// Decode reductions require captured provenance and exclude all other probes.
    #[test]
    fn test_ffn_arguments_preserve_explicit_modes() {
        let parse = |extra: &[&str]| {
            let mut args = vec![
                "q8_execution_bench",
                "--model",
                "source.gguf",
                "--output",
                "new.jsonl",
            ];
            args.extend_from_slice(extra);
            Args::try_parse_from(args)
        };
        assert!(parse(&[]).unwrap().ffn_inputs.is_none());
        let flags = [
            "--ffn-inputs",
            "inputs.json",
            "--ffn-capture-report",
            "capture.jsonl",
        ];
        assert_eq!(parse(&flags).unwrap().rows, [1, 4]);
        let reductions = "--decode-reductions";
        assert!(parse(&[reductions]).is_err());
        assert!(
            parse(&[flags.as_slice(), &[reductions]].concat())
                .unwrap()
                .decode_reductions
        );
        for other in [
            "--qkv-f32-split",
            "--output-f32-split",
            "--ffn-f32-persistent-tiles",
            "--ffn-f32-virtual-warp",
            "--prepared-f32-packed-activation",
            "--prepared-f32-warp",
            "--qkv-f32-packed",
            "--output-f32-packed",
            "--qkv-f32-warp",
            "--output-f32-warp",
            "--ffn-f32-ablation",
            "--ffn-q81",
            "--ffn-f32-factored-scales",
            "--ffn-f32-coalesced-pairs",
            "--ffn-f32-split-planes",
            "--ffn-f32-split-two-columns",
            "--ffn-f32-joint",
            "--ffn-f32-fixed-k",
            "--ffn-f32-unfused",
            "--ffn-f32-weight-cg",
            "--ffn-f32-byte-permute",
        ] {
            assert!(parse(&[flags.as_slice(), &[reductions, other]].concat()).is_err());
        }
        let activation = "--prepared-f32-packed-activation";
        assert!(
            parse(&[flags.as_slice(), &[activation]].concat())
                .unwrap()
                .prepared_f32_packed_activation
        );
        assert!(parse(&[activation]).is_err());
        for other in [
            "--prepared-f32-warp",
            "--qkv-f32-packed",
            "--output-f32-packed",
            "--qkv-f32-warp",
            "--output-f32-warp",
            "--ffn-f32-ablation",
            "--ffn-q81",
            "--ffn-f32-factored-scales",
            "--ffn-f32-coalesced-pairs",
            "--ffn-f32-split-planes",
            "--ffn-f32-split-two-columns",
            "--ffn-f32-joint",
            "--ffn-f32-fixed-k",
            "--ffn-f32-unfused",
            "--ffn-f32-weight-cg",
            "--ffn-f32-byte-permute",
        ] {
            assert!(parse(&[flags.as_slice(), &[activation, other]].concat()).is_err());
        }
        assert!(!parse(&flags).unwrap().qkv_f32_packed);
        assert!(!parse(&flags).unwrap().output_f32_packed);
        for packed in ["--qkv-f32-packed", "--output-f32-packed"] {
            let args = parse(&[flags.as_slice(), &[packed]].concat()).unwrap();
            assert_eq!(args.qkv_f32_packed, packed == "--qkv-f32-packed");
            assert_eq!(args.output_f32_packed, packed == "--output-f32-packed");
            assert!(parse(&[packed]).is_err());
            for other in [
                "--prepared-f32-warp",
                "--qkv-f32-warp",
                "--output-f32-warp",
                "--ffn-f32-ablation",
                "--ffn-q81",
                "--ffn-f32-factored-scales",
                "--ffn-f32-coalesced-pairs",
                "--ffn-f32-split-planes",
                "--ffn-f32-split-two-columns",
                "--ffn-f32-joint",
                "--ffn-f32-fixed-k",
                "--ffn-f32-unfused",
                "--ffn-f32-weight-cg",
                "--ffn-f32-byte-permute",
            ] {
                assert!(parse(&[flags.as_slice(), &[packed, other]].concat()).is_err());
            }
        }
        assert!(parse(
            &[
                flags.as_slice(),
                &["--qkv-f32-packed", "--output-f32-packed"]
            ]
            .concat()
        )
        .is_err());
        assert!(!parse(&flags).unwrap().ffn_f32_ablation);
        assert!(!parse(&flags).unwrap().ffn_f32_factored_scales);
        assert!(!parse(&flags).unwrap().ffn_f32_coalesced_pairs);
        assert!(!parse(&flags).unwrap().ffn_f32_split_planes);
        assert!(!parse(&flags).unwrap().ffn_f32_split_two_columns);
        assert!(!parse(&flags).unwrap().ffn_f32_joint);
        assert!(!parse(&flags).unwrap().ffn_f32_fixed_k);
        assert!(!parse(&flags).unwrap().qkv_f32_warp);
        assert!(!parse(&flags).unwrap().prepared_f32_warp);
        assert!(!parse(&flags).unwrap().ffn_f32_unfused);
        assert!(!parse(&flags).unwrap().ffn_f32_weight_cg);
        assert!(!parse(&flags).unwrap().ffn_f32_byte_permute);
        assert!(
            parse(&[flags.as_slice(), &["--ffn-f32-byte-permute"]].concat())
                .unwrap()
                .ffn_f32_byte_permute
        );
        assert!(parse(&["--ffn-f32-byte-permute"]).is_err());
        for other in [
            "--prepared-f32-warp",
            "--qkv-f32-warp",
            "--output-f32-warp",
            "--ffn-f32-ablation",
            "--ffn-q81",
            "--ffn-f32-factored-scales",
            "--ffn-f32-coalesced-pairs",
            "--ffn-f32-split-planes",
            "--ffn-f32-split-two-columns",
            "--ffn-f32-joint",
            "--ffn-f32-fixed-k",
            "--ffn-f32-unfused",
            "--ffn-f32-weight-cg",
        ] {
            assert!(
                parse(&[flags.as_slice(), &["--ffn-f32-byte-permute", other]].concat()).is_err()
            );
        }
        assert!(
            parse(&[flags.as_slice(), &["--ffn-f32-weight-cg"]].concat())
                .unwrap()
                .ffn_f32_weight_cg
        );
        assert!(parse(&["--ffn-f32-weight-cg"]).is_err());
        for other in [
            "--prepared-f32-warp",
            "--qkv-f32-warp",
            "--output-f32-warp",
            "--ffn-f32-ablation",
            "--ffn-q81",
            "--ffn-f32-factored-scales",
            "--ffn-f32-coalesced-pairs",
            "--ffn-f32-split-planes",
            "--ffn-f32-split-two-columns",
            "--ffn-f32-joint",
            "--ffn-f32-fixed-k",
            "--ffn-f32-unfused",
        ] {
            assert!(parse(&[flags.as_slice(), &["--ffn-f32-weight-cg", other]].concat()).is_err());
        }
        assert!(
            parse(&[flags.as_slice(), &["--ffn-f32-unfused"]].concat())
                .unwrap()
                .ffn_f32_unfused
        );
        assert!(parse(&["--ffn-f32-unfused"]).is_err());
        for other in [
            "--prepared-f32-warp",
            "--ffn-f32-joint",
            "--ffn-f32-fixed-k",
            "--ffn-q81",
        ] {
            assert!(parse(&[flags.as_slice(), &["--ffn-f32-unfused", other]].concat()).is_err());
        }
        assert!(
            parse(&[flags.as_slice(), &["--prepared-f32-warp"]].concat())
                .unwrap()
                .prepared_f32_warp
        );
        assert!(parse(&["--prepared-f32-warp"]).is_err());
        for other in [
            "--qkv-f32-warp",
            "--ffn-q81",
            "--ffn-f32-split-planes",
            "--ffn-f32-fixed-k",
        ] {
            assert!(parse(&[flags.as_slice(), &["--prepared-f32-warp", other]].concat()).is_err());
        }
        assert!(
            parse(&[flags.as_slice(), &["--qkv-f32-warp"]].concat())
                .unwrap()
                .qkv_f32_warp
        );
        assert!(parse(&["--qkv-f32-warp"]).is_err());
        let output_flags = ["--output-f32-warp", "--qkv-warp-report", "qkv.jsonl"];
        assert!(
            parse(&[flags.as_slice(), &output_flags].concat())
                .unwrap()
                .output_f32_warp
        );
        assert!(parse(&[flags.as_slice(), &["--output-f32-warp"]].concat()).is_err());
        assert!(parse(&[flags.as_slice(), &["--qkv-warp-report", "qkv.jsonl"]].concat()).is_err());
        assert!(parse(&output_flags).is_err());
        assert!(parse(&[flags.as_slice(), &output_flags, &["--qkv-f32-warp"]].concat()).is_err());
        for other in [
            "--ffn-f32-ablation",
            "--ffn-q81",
            "--ffn-f32-factored-scales",
            "--ffn-f32-coalesced-pairs",
            "--ffn-f32-split-planes",
            "--ffn-f32-split-two-columns",
            "--ffn-f32-joint",
            "--ffn-f32-fixed-k",
        ] {
            assert!(parse(&[flags.as_slice(), &["--qkv-f32-warp", other]].concat()).is_err());
            assert!(parse(&[flags.as_slice(), &output_flags, &[other]].concat()).is_err());
        }
        assert!(
            parse(&[flags.as_slice(), &["--ffn-f32-fixed-k"]].concat())
                .unwrap()
                .ffn_f32_fixed_k
        );
        assert!(parse(&["--ffn-f32-fixed-k"]).is_err());
        for other in [
            "--ffn-f32-ablation",
            "--ffn-q81",
            "--ffn-f32-factored-scales",
            "--ffn-f32-coalesced-pairs",
            "--ffn-f32-split-planes",
            "--ffn-f32-split-two-columns",
            "--ffn-f32-joint",
        ] {
            assert!(parse(&[flags.as_slice(), &["--ffn-f32-fixed-k", other]].concat()).is_err());
        }
        assert!(
            parse(&[flags.as_slice(), &["--ffn-f32-joint"]].concat())
                .unwrap()
                .ffn_f32_joint
        );
        assert!(parse(&["--ffn-f32-joint"]).is_err());
        for other in [
            "--ffn-f32-ablation",
            "--ffn-q81",
            "--ffn-f32-factored-scales",
            "--ffn-f32-coalesced-pairs",
            "--ffn-f32-split-planes",
            "--ffn-f32-split-two-columns",
        ] {
            assert!(parse(&[flags.as_slice(), &["--ffn-f32-joint", other]].concat()).is_err());
        }
        assert!(
            parse(&[flags.as_slice(), &["--ffn-f32-split-two-columns"]].concat())
                .unwrap()
                .ffn_f32_split_two_columns
        );
        assert!(parse(&["--ffn-f32-split-two-columns"]).is_err());
        for other in [
            "--ffn-f32-ablation",
            "--ffn-q81",
            "--ffn-f32-factored-scales",
            "--ffn-f32-coalesced-pairs",
            "--ffn-f32-split-planes",
        ] {
            assert!(
                parse(&[flags.as_slice(), &["--ffn-f32-split-two-columns", other]].concat())
                    .is_err()
            );
        }
        assert!(
            parse(&[flags.as_slice(), &["--ffn-f32-split-planes"]].concat())
                .unwrap()
                .ffn_f32_split_planes
        );
        assert!(parse(&["--ffn-f32-split-planes"]).is_err());
        for other in [
            "--ffn-f32-ablation",
            "--ffn-q81",
            "--ffn-f32-factored-scales",
            "--ffn-f32-coalesced-pairs",
        ] {
            assert!(
                parse(&[flags.as_slice(), &["--ffn-f32-split-planes", other]].concat()).is_err()
            );
        }
        assert!(
            parse(&[flags.as_slice(), &["--ffn-f32-coalesced-pairs"]].concat())
                .unwrap()
                .ffn_f32_coalesced_pairs
        );
        assert!(parse(&["--ffn-f32-coalesced-pairs"]).is_err());
        for other in [
            "--ffn-q81",
            "--ffn-f32-ablation",
            "--ffn-f32-factored-scales",
        ] {
            assert!(
                parse(&[flags.as_slice(), &["--ffn-f32-coalesced-pairs", other]].concat()).is_err()
            );
        }
        assert!(
            parse(&[flags.as_slice(), &["--ffn-f32-factored-scales"]].concat())
                .unwrap()
                .ffn_f32_factored_scales
        );
        assert!(parse(&["--ffn-f32-factored-scales"]).is_err());
        for other in ["--ffn-q81", "--ffn-f32-ablation"] {
            assert!(
                parse(&[flags.as_slice(), &["--ffn-f32-factored-scales", other]].concat()).is_err()
            );
        }
        assert!(
            parse(&[flags.as_slice(), &["--ffn-f32-ablation"]].concat())
                .unwrap()
                .ffn_f32_ablation
        );
        assert!(parse(&["--ffn-f32-ablation"]).is_err());
        assert!(parse(&["--ffn-q81"]).is_err());
        assert!(
            parse(&[flags.as_slice(), &["--ffn-q81"]].concat())
                .unwrap()
                .ffn_q81
        );
        assert!(parse(&[flags.as_slice(), &["--ffn-q81", "--ffn-f32-ablation"]].concat()).is_err());
        assert!(parse(&["--ffn-inputs", "inputs.json"]).is_err());
        assert!(parse(&["--ffn-capture-report", "capture.jsonl"]).is_err());
        for extra in [
            vec!["--sparse-inputs"],
            vec!["--layout", "input-tiled"],
            vec!["--tensors", "blk.0.ffn_down.weight"],
        ] {
            assert!(parse(&[flags.as_slice(), extra.as_slice()].concat()).is_err());
        }
        assert!(parse(&["--layout", "input-tiled", "--rows", "1"]).is_ok());
    }
}
