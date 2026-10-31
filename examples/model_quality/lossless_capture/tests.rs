use super::*;
use clap::Parser;
use snapshot::{append_token, HeadWords};

/// Fixed token windows are disjoint and complete; insufficient or duplicated
/// input cannot silently turn the batch-eight fixture into copied rows.
#[test]
fn test_disjoint_input_spans() -> Result<()> {
    let ids: Vec<_> = (0..(TOKENS * SEQUENCES + 17) as u32).collect();
    let spans = select_spans(&ids)?;
    assert_eq!(spans.len(), 8);
    assert_eq!(spans.concat(), ids[..TOKENS * SEQUENCES]);
    assert!(select_spans(&ids[..TOKENS * SEQUENCES - 1]).is_err());
    assert!(select_spans(&vec![7; TOKENS * SEQUENCES]).is_err());
    Ok(())
}

/// Head deinterleaving retains token order and every special bit pattern in K
/// and V. Bad shapes fail before any head is changed.
#[test]
fn test_head_order_preserves_bits() -> Result<()> {
    let mut heads = vec![
        HeadWords {
            key: Vec::new(),
            value: Vec::new(),
        },
        HeadWords {
            key: Vec::new(),
            value: Vec::new(),
        },
    ];
    append_token(&mut heads, &[0, 0x8000, 0x7d55, 0xfc00], &[1, 2, 3, 4], 2)?;
    append_token(&mut heads, &[5, 6, 7, 8], &[9, 10, 11, 12], 2)?;
    assert_eq!(heads[0].key, [0, 0x8000, 5, 6]);
    assert_eq!(heads[1].key, [0x7d55, 0xfc00, 7, 8]);
    assert_eq!(heads[0].value, [1, 2, 9, 10]);
    assert_eq!(heads[1].value, [3, 4, 11, 12]);
    assert!(append_token(&mut heads, &[1], &[2], 2).is_err());
    assert!(append_token(&mut [], &[], &[], 2).is_err());
    assert_eq!(heads[0].key, [0, 0x8000, 5, 6]);
    Ok(())
}

/// The persisted stream is exactly the input words in little-endian order;
/// all four prefix reports identify the corresponding stored bytes and size.
#[test]
fn test_stream_artifact_and_prefix_identity() -> Result<()> {
    let data = crate::test_files::TestReportPath::new();
    let output = crate::test_files::TestReportPath::new();
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(data.path())?;
    let mut report = Report::create(output.path())?;
    let words: Vec<_> = (0..TOKENS).map(|v| 0x3c00 | ((v as u16) & 1023)).collect();
    store_stream(&mut file, &mut report, json!({"test":"known"}), &words, 1)?;
    drop(report);
    drop(file);
    let actual = fs::read(data.path())?;
    let expected: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    assert_eq!(actual, expected);
    let rows: Vec<Value> = fs::read_to_string(output.path())?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    assert_eq!(rows.len(), 5);
    assert_eq!(rows[0]["raw_offset"], 0);
    assert_eq!(rows[0]["raw_bytes"], TOKENS * 2);
    assert_eq!(rows[0]["raw_sha256"], records::sha256(&actual)?);
    for (row, count) in rows[1..].iter().zip(CONTEXTS) {
        assert_eq!(row["context_tokens"], count);
        assert_eq!(row["stats"]["values"], count);
        assert_eq!(
            row["source_prefix_sha256"],
            records::sha256(&actual[..count * 2])?
        );
        assert_eq!(
            row["stats"]["encoded_bytes"],
            8 + 8 * (count / 512) + count * 11 / 8
        );
    }
    assert!(Report::create(output.path()).is_err());
    Ok(())
}

/// Capture is an explicit, exclusive mode and cannot overwrite its report or
/// silently accept a short context that misses the frozen long-context scope.
#[test]
fn test_cli_and_capture_request() -> Result<()> {
    let base = [
        "quality",
        "--model",
        "model",
        "--reference-model",
        "model",
        "--variant-id",
        "lossless",
        "--output",
        "/tmp/lossless-test-report-never-created",
        "--capture-lossless-kv",
        "/tmp/lossless-test-data-never-created",
        "--context-tokens",
        "8192",
        "--split",
        "calibration",
    ];
    let mut args = Args::try_parse_from(base)?;
    validate_request(&args)?;
    let mut conflict = base.to_vec();
    conflict.extend(["--skip-layers", "1"]);
    assert!(Args::try_parse_from(conflict).is_err());
    args.context_tokens = 1024;
    assert!(validate_request(&args).is_err());
    args.context_tokens = 8192;
    args.capture_lossless_kv = Some(args.output.clone());
    assert!(validate_request(&args).is_err());
    Ok(())
}

/// Readback uses physical layer addressing and valid lengths, preserves exact
/// device FP16 words, and releases all tracked model/session buffers on M40.
#[test]
#[ignore = "explicit serial M40 lossless KV capture check"]
fn test_real_cache_snapshot_and_cleanup() -> Result<()> {
    use m40_llm::{
        cuda::ffi_debug_read_kv_token,
        infer::{q8_prefill::Q8PrefillBackend, LoadedModel, Q8FfnConfig},
    };
    let _lock = crate::cuda_test_scope::lock_cuda();
    let _env: Vec<_> = [
        ("M40LLM_ENABLE_NVCC", "1"),
        ("M40LLM_PROJECTION_BACKEND", "large-model"),
        ("M40LLM_MATERIALIZE_F32_WEIGHTS", "0"),
        ("M40LLM_Q8_DECODE_STREAM", "decode"),
        ("M40LLM_FUSED_Q8_QKV", "1"),
        ("M40LLM_FUSED_Q8_MLP_SWIGLU", "1"),
        ("M40LLM_DECODE_GRAPH", "0"),
    ]
    .into_iter()
    .map(|(key, value)| crate::env_restore::EnvRestore::set(key, value))
    .collect();
    let _hardware = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    let _dense = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
    {
        let (gguf, bytes) = crate::q8_fixture::build_prefill_fixture(true);
        let mut model = LoadedModel::from_gguf_with_q8_backends(
            gguf,
            bytes,
            -1,
            Q8ExecutionConfig::parse("off", None, None)?,
            Q8FfnConfig::native(),
            Q8PrefillBackend::Native,
        )?;
        model.allocate_kv_cache_for_layers(16)?;
        for ids in [&[65, 66, 67][..], &[68, 69][..]] {
            model.reset_kv_cache()?;
            let mut session = DecodeSession::new_for_sequence_with_kv_config(
                &model,
                0,
                KvCompressionConfig::dense_reference(),
                256,
                true,
                "lossless-test",
                "lossless-test:x",
                "lossless-test:out",
            )?;
            session.prefill_packed_prefix_ids(ids)?;
            session.greedy_token_for_ids(ids)?;
            for layer in [0, 1] {
                let captured = snapshot::read_layer(&model, layer, ids.len())?;
                assert_eq!(captured.len(), 1);
                let repeated = snapshot::read_layer(&model, layer, ids.len())?;
                assert_eq!(captured[0].key, repeated[0].key);
                assert_eq!(captured[0].value, repeated[0].value);
                let cache = model.kv_cache.as_ref().unwrap();
                let slot = model.kv_cache_physical_slot_for_layer_sequence(layer, 0)?;
                let mut key = [0_u16; 128];
                let mut value = [0_u16; 128];
                assert_eq!(
                    unsafe {
                        ffi_debug_read_kv_token(
                            &model.cuda,
                            cache,
                            slot,
                            ids.len() as u32 - 1,
                            key.as_mut_ptr().cast(),
                            value.as_mut_ptr().cast(),
                        )
                    },
                    0
                );
                assert_eq!(&captured[0].key[(ids.len() - 1) * 128..], &key);
                assert_eq!(&captured[0].value[(ids.len() - 1) * 128..], &value);
                for words in [&captured[0].key, &captured[0].value] {
                    assert_eq!(LosslessKvBlocks::encode(words)?.decode()?, *words);
                }
                assert!(snapshot::read_layer(&model, layer, ids.len() + 1).is_err());
            }
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}
