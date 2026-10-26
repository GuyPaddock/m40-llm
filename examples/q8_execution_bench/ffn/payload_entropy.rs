//! Offline source-byte feasibility only; no codec or CUDA consumer is created.
use super::fixture::FfnInputs;
use crate::run::{command_output, write_record, Args};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::CudaContext,
    evaluation::identity::sha256,
    gguf::{load_gguf, GgmlDType},
};
use serde_json::{json, Value};
use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom},
    time::Instant,
};

fn code_lengths(histogram: &[u64; 256]) -> Result<[u8; 256]> {
    let mut children = vec![None; 256];
    let mut heap = BinaryHeap::new();
    for (symbol, &frequency) in histogram.iter().enumerate() {
        if frequency > 0 {
            heap.push(Reverse((frequency, symbol)));
        }
    }
    ensure!(!heap.is_empty(), "empty payload histogram");
    while heap.len() > 1 {
        let Reverse((a, left)) = heap.pop().unwrap();
        let Reverse((b, right)) = heap.pop().unwrap();
        let weight = a.checked_add(b).context("Huffman frequency overflow")?;
        let parent = children.len();
        children.push(Some((left, right)));
        heap.push(Reverse((weight, parent)));
    }
    let root = heap.pop().unwrap().0 .1;
    let mut stack = vec![(root, 0u8)];
    let mut lengths = [0u8; 256];
    while let Some((node, depth)) = stack.pop() {
        if let Some((left, right)) = children[node] {
            let next = depth.checked_add(1).context("Huffman depth overflow")?;
            stack.extend([(left, next), (right, next)]);
        } else {
            lengths[node] = depth.max(1);
        }
    }
    Ok(lengths)
}

fn analyze_blocks(raw: &[u8]) -> Result<Value> {
    ensure!(
        !raw.is_empty() && raw.len().is_multiple_of(34) && raw.len() <= 64 * 1024 * 1024,
        "payload study requires bounded complete Q8_0 blocks"
    );
    let mut histogram = [0u64; 256];
    for block in raw.chunks_exact(34) {
        for &byte in &block[2..] {
            histogram[byte as usize] += 1;
        }
    }
    let symbols = histogram.iter().sum::<u64>();
    let lengths = code_lengths(&histogram)?;
    let entropy = histogram
        .iter()
        .filter(|&&n| n > 0)
        .map(|&n| {
            let p = n as f64 / symbols as f64;
            -p * p.log2()
        })
        .sum::<f64>();
    let bits = histogram
        .iter()
        .zip(lengths)
        .map(|(&n, len)| n * u64::from(len))
        .sum::<u64>();
    let scale_bytes = raw.len() / 34 * 2;
    let mut models = Vec::new();
    for chunk_size in [32usize, 256, 1024] {
        let (mut in_chunk, mut chunk_bits, mut padded_bytes, mut chunks) =
            (0usize, 0u64, 0u64, 0usize);
        for &byte in raw.chunks_exact(34).flat_map(|b| b[2..].iter()) {
            chunk_bits += u64::from(lengths[byte as usize]);
            in_chunk += 1;
            if in_chunk == chunk_size {
                padded_bytes += chunk_bits.div_ceil(128) * 16;
                chunks += 1;
                in_chunk = 0;
                chunk_bits = 0;
            }
        }
        if in_chunk > 0 {
            padded_bytes += chunk_bits.div_ceil(128) * 16;
            chunks += 1;
        }
        let index_bytes = chunks * 4;
        let idealized_stored_bytes = padded_bytes + index_bytes as u64 + scale_bytes as u64 + 256;
        models.push(json!({"chunk_payload_symbols":chunk_size,"chunks":chunks,"aligned_huffman_payload_bytes":padded_bytes,
            "chunk_alignment_bytes":16,"offset_bytes_per_chunk":4,"index_bytes":index_bytes,
            "scale_bytes":scale_bytes,"code_length_table_bytes":256,"idealized_stored_bytes":idealized_stored_bytes,
            "stored_byte_saving":1.0-idealized_stored_bytes as f64/raw.len() as f64,
            "includes_decoder_storage":false,"has_materialized_encoded_stream":false}));
    }
    Ok(
        json!({"source_bytes":raw.len(),"scale_bytes":scale_bytes,"payload_symbols":symbols,
        "payload_histogram":histogram.to_vec(),"zero_payload_symbols":histogram[0],
        "negative_payload_symbols":histogram[128..].iter().sum::<u64>(),
        "zero_order_entropy_bits_per_symbol":entropy,"zero_order_ideal_total_bytes":(entropy*symbols as f64/8.0).ceil() as u64+scale_bytes as u64,
        "huffman_code_lengths":lengths.to_vec(),"huffman_payload_bits":bits,"maximum_code_length":lengths.into_iter().max().unwrap(),
        "indexed_size_models":models,"has_measured_speedup":false,"has_decoder":false} ),
    )
}

pub(super) fn run(args: &Args) -> Result<()> {
    ensure!(
        !cfg!(debug_assertions),
        "payload feasibility requires release"
    );
    let initial = CudaContext::total_device_bytes();
    ensure!(initial == 0, "offline study cannot reuse a live CUDA model");
    let inputs = FfnInputs::load(args)?;
    inputs.validate_environment()?;
    let gguf = load_gguf(&args.model)?;
    let mut source = File::open(&args.model)?;
    let mut output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args.output)?;
    write_record(
        &mut output,
        json!({"kind":"ffn_payload_entropy_manifest","schema_version":1,"build_profile":"release",
        "revision":command_output("git",&["rev-parse","HEAD"])?,"worktree":command_output("git",&["status","--short"])?,
        "model":args.model,"model_sha256":inputs.artifact.plan.model_sha256,"fixture_sha256":inputs.artifact_sha256,
        "capture_report_sha256":inputs.report_sha256,"execution_configuration":inputs.manifest["execution_configuration"],
        "source_layers":[0,17,35],"families":["gate","up"],"has_created_cuda_context":false,
        "tracked_device_bytes":initial,"chunk_sizes":[32,256,1024],"minimum_stored_byte_saving":0.10,
        "interpretation":"offline zero-order byte statistics and indexed Huffman size model; no codec, decoder workspace, GPU timing or inference speed claim"}),
    )?;
    let mut eligibility = [true; 3];
    for layer in &inputs.artifact.layers {
        for (family, tensor_source) in [("gate", &layer.source.gate), ("up", &layer.source.up)] {
            let tensor = gguf
                .tensors
                .iter()
                .find(|t| t.name == tensor_source.name)
                .context("missing entropy source")?;
            ensure!(
                tensor.dtype == GgmlDType::Q8_0
                    && tensor.shape == tensor_source.shape.map(|dimension| dimension as u64)
                    && tensor.offset == tensor_source.byte_offset,
                "entropy GGUF tensor identity differs from frozen capture"
            );
            let len = tensor_source.byte_len()?;
            ensure!(
                len <= 64 * 1024 * 1024,
                "entropy source exceeds bounded tensor read"
            );
            let offset = gguf
                .data_offset
                .checked_add(tensor.offset)
                .context("entropy file offset overflow")?;
            let began = Instant::now();
            source.seek(SeekFrom::Start(offset))?;
            let mut raw = vec![0u8; len];
            source.read_exact(&mut raw)?;
            ensure!(
                sha256(&raw)? == tensor_source.source_sha256,
                "entropy source bytes differ from frozen capture"
            );
            let read_and_hash_ms = began.elapsed().as_secs_f64() * 1000.0;
            let began = Instant::now();
            let statistics = analyze_blocks(&raw)?;
            let analysis_ms = began.elapsed().as_secs_f64() * 1000.0;
            for (index, eligible) in eligibility.iter_mut().enumerate() {
                *eligible &= statistics["indexed_size_models"][index]["stored_byte_saving"]
                    .as_f64()
                    .unwrap()
                    >= 0.10;
            }
            write_record(
                &mut output,
                json!({"kind":"ffn_payload_entropy_tensor","layer_id":layer.source.layer_id,"family":family,
                "source":tensor_source,"absolute_file_offset":offset,"statistics":statistics,"read_and_hash_ms":read_and_hash_ms,
                "analysis_ms":analysis_ms,"raw_tensor_buffer_bytes":len,"has_created_cuda_context":false}),
            )?;
        }
    }
    ensure!(
        CudaContext::total_device_bytes() == initial,
        "offline study allocated device storage"
    );
    write_record(
        &mut output,
        json!({"kind":"ffn_payload_entropy_complete","is_complete":true,"tensors":6,
        "size_model_eligible_at_all_tensors":eligibility,"minimum_stored_byte_saving":0.10,
        "tracked_device_bytes_before":initial,"tracked_device_bytes_after":CudaContext::total_device_bytes(),
        "runtime_integration_allowed":false,"is_e2e_claim":false}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Optimal prefix lengths retain the complete symbol count and reject
    /// empty/overflowing frequency inputs instead of inventing a code cost.
    #[test]
    fn test_huffman_lengths_match_known_costs() -> Result<()> {
        assert!(code_lengths(&[0; 256]).is_err());
        let mut hist = [0; 256];
        hist[0] = 2;
        hist[1] = 1;
        hist[2] = 1;
        let lengths = code_lengths(&hist)?;
        assert_eq!(&lengths[..3], &[1, 2, 2]);
        assert_eq!(
            hist.iter()
                .zip(lengths)
                .map(|(&n, len)| n * u64::from(len))
                .sum::<u64>(),
            6
        );
        assert!(code_lengths(&[u64::MAX; 256]).is_err());
        assert_eq!(code_lengths(&[1; 256])?, [8; 256]);
        Ok(())
    }

    /// Payload-only entropy ignores scale bits while size models charge those
    /// bytes, aligned chunk payload, codebook and random-access offsets.
    #[test]
    fn test_q8_statistics_account_for_all_stored_components() -> Result<()> {
        let mut raw = vec![0xaa, 0xbb];
        raw.extend(0u8..32);
        let stats = analyze_blocks(&raw)?;
        assert_eq!(stats["zero_order_entropy_bits_per_symbol"], 5.0);
        assert_eq!(stats["huffman_payload_bits"], 160);
        assert_eq!(stats["payload_symbols"], 32);
        for model in stats["indexed_size_models"].as_array().unwrap() {
            assert_eq!(model["aligned_huffman_payload_bytes"], 32);
            assert_eq!(model["index_bytes"], 4);
            assert_eq!(model["scale_bytes"], 2);
            assert_eq!(model["idealized_stored_bytes"], 294);
        }
        raw[..2].fill(0);
        assert_eq!(analyze_blocks(&raw)?, stats);
        for len in [0, 1, 33, 35] {
            assert!(analyze_blocks(&vec![0; len]).is_err());
        }
        Ok(())
    }

    /// A one-symbol payload has zero empirical entropy but a one-bit prefix
    /// code; tail chunks still pay alignment and their own offset entry.
    #[test]
    fn test_constant_payload_preserves_tail_chunk_costs() -> Result<()> {
        let stats = analyze_blocks(&vec![0; 34 * 9])?;
        assert_eq!(stats["zero_order_entropy_bits_per_symbol"], 0.0);
        assert_eq!(stats["huffman_payload_bits"], 288);
        assert_eq!(stats["indexed_size_models"][1]["chunks"], 2);
        assert_eq!(
            stats["indexed_size_models"][1]["aligned_huffman_payload_bytes"],
            48
        );
        assert_eq!(stats["indexed_size_models"][1]["index_bytes"], 8);
        Ok(())
    }
}
