use anyhow::Result;
use m40_llm::{
    gguf::{compact::CompactGgufPlan, GgmlDType, GgufModel, GgufScalar, GgufTensor, GgufValue},
    infer::LayerMask,
};

fn source_model(architecture: &str) -> GgufModel {
    let mut model = GgufModel::new(0);
    model.version = 3;
    model.data_offset = 1024;
    model.metadata.insert(
        "general.architecture".into(),
        GgufValue::Scalar(GgufScalar::Str(architecture.into())),
    );
    model.metadata.insert(
        format!("{architecture}.block_count"),
        GgufValue::Scalar(GgufScalar::U32(5)),
    );
    let mut offset = 0;
    for name in std::iter::once("token_embd.weight".to_owned())
        .chain((0..5).map(|layer| format!("blk.{layer}.attn_q.weight")))
        .chain(std::iter::once("output_norm.weight".to_owned()))
    {
        model.tensors.push(GgufTensor {
            name,
            dtype: GgmlDType::Q8_0,
            shape: vec![32, 2],
            offset,
        });
        offset += 96;
    }
    model
}

/// Compaction renumbers only retained standard blocks, preserves globals and
/// dtype-sized byte ranges, and reports payload savings separately from padding.
#[test]
fn test_compact_plan_maps_original_layers_and_quantized_payloads() -> Result<()> {
    for architecture in ["llama", "qwen2"] {
        let source = source_model(architecture);
        let plan = CompactGgufPlan::build(&source, 1690, &LayerMask::new(5, &[3, 1])?)?;
        assert_eq!(plan.retained_original_layers(), &[0, 2, 4]);
        assert_eq!(plan.tensors().len(), 5);
        assert_eq!(plan.source_weight_payload_bytes(), 7 * 68);
        assert_eq!(plan.retained_weight_payload_bytes(), 5 * 68);
        assert_eq!(plan.destination_data_bytes(), 4 * 96 + 68);
        for (copy, original) in plan.tensors().iter().zip([0, 1, 3, 5, 6]) {
            assert_eq!(copy.source_offset(), 1024 + original * 96);
            assert_eq!(copy.payload_bytes(), 68);
            assert_eq!(copy.destination_offset() % 32, 0);
        }
        assert_eq!(plan.tensors()[2].source_name(), "blk.2.attn_q.weight");
        assert_eq!(plan.tensors()[2].destination_name(), "blk.1.attn_q.weight");
        assert_eq!(plan.tensors()[4].destination_name(), "output_norm.weight");
        assert_eq!(source.tensors[3].name, "blk.2.attn_q.weight");
    }
    Ok(())
}

/// Mixed F32/F16/Q8 payloads and an explicit alignment retain exact byte
/// lengths instead of copying gaps or assuming every tensor is quantized.
#[test]
fn test_compact_plan_counts_mixed_payloads_and_custom_alignment() -> Result<()> {
    let mut source = source_model("qwen2");
    source.metadata.insert(
        "general.alignment".into(),
        GgufValue::Scalar(GgufScalar::U64(64)),
    );
    for (index, tensor) in source.tensors.iter_mut().enumerate() {
        tensor.offset = index as u64 * 256;
        tensor.dtype = [GgmlDType::F32, GgmlDType::F16, GgmlDType::Q8_0][index % 3];
    }
    let plan = CompactGgufPlan::build(&source, 4096, &LayerMask::new(5, &[1, 3])?)?;
    assert_eq!(plan.source_weight_payload_bytes(), 1160);
    assert_eq!(plan.retained_weight_payload_bytes(), 964);
    assert_eq!(
        plan.tensors()
            .iter()
            .map(|tensor| tensor.payload_bytes())
            .collect::<Vec<_>>(),
        vec![256, 128, 256, 68, 256]
    );
    assert!(plan
        .tensors()
        .iter()
        .all(|tensor| tensor.destination_offset().is_multiple_of(64)));
    Ok(())
}

/// Empty/cross-model masks, aliases, truncation and unsupported wire layouts
/// cannot produce an apparently valid plan or mutate the parsed source.
#[test]
fn test_compact_plan_rejects_unsafe_tensor_ranges_and_masks() -> Result<()> {
    let source = source_model("qwen2");
    assert!(CompactGgufPlan::build(&source, 2000, &LayerMask::default()).is_err());
    assert!(CompactGgufPlan::build(&source, 2000, &LayerMask::new(6, &[1])?).is_err());
    assert!(CompactGgufPlan::build(&source, 1667, &LayerMask::new(5, &[1])?).is_err());
    let mutations: [fn(&mut GgufModel); 11] = [
        |model| model.tensors[2].offset = model.tensors[1].offset,
        |model| model.tensors[2].offset += 1,
        |model| model.tensors[2].offset = u64::MAX - 31,
        |model| model.tensors[2].shape = vec![u64::MAX, u64::MAX],
        |model| model.tensors[2].shape[0] = 31,
        |model| model.tensors[2].shape.clear(),
        |model| model.tensors[2].dtype = GgmlDType::Unknown(99),
        |model| model.tensors[2].name = "blk.01.attn_q.weight".into(),
        |model| model.tensors[2].name = "blk.5.attn_q.weight".into(),
        |model| model.tensors[2].name = "blk.1.other.weight".into(),
        |model| model.tensors[2].name = "token_embd.weight".into(),
    ];
    for mutate in mutations {
        let mut source = source_model("qwen2");
        mutate(&mut source);
        assert!(CompactGgufPlan::build(&source, 2000, &LayerMask::new(5, &[1])?).is_err());
    }
    Ok(())
}

/// Heterogeneous/expert, split and already compacted sources need explicit
/// metadata remapping; tokenizer arrays do not depend on transformer layer IDs.
#[test]
fn test_compact_plan_rejects_unsupported_metadata_but_keeps_tokenizer_arrays() -> Result<()> {
    let mask = LayerMask::new(5, &[1])?;
    for key in [
        "qwen2.attention.head_count",
        "qwen2.expert_count",
        "split.no",
        "m40llm.pruning.source_sha256",
    ] {
        let mut source = source_model("qwen2");
        source.metadata.insert(key.into(), GgufValue::Array(vec![]));
        assert!(CompactGgufPlan::build(&source, 2000, &mask).is_err());
    }
    let mut source = source_model("qwen2");
    source
        .metadata
        .insert("tokenizer.ggml.tokens".into(), GgufValue::Array(vec![]));
    CompactGgufPlan::build(&source, 2000, &mask)?;
    source.version = 2;
    assert!(CompactGgufPlan::build(&source, 2000, &mask).is_err());
    assert!(CompactGgufPlan::build(&source_model("mamba"), 2000, &mask).is_err());
    source.version = 3;
    source.metadata.insert(
        "general.type".into(),
        GgufValue::Scalar(GgufScalar::Str("adapter".into())),
    );
    assert!(CompactGgufPlan::build(&source, 2000, &mask).is_err());
    source.metadata.remove("general.type");
    source.metadata.insert(
        "general.alignment".into(),
        GgufValue::Scalar(GgufScalar::U32(3)),
    );
    assert!(CompactGgufPlan::build(&source, 2000, &mask).is_err());
    Ok(())
}
