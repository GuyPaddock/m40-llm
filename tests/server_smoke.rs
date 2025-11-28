#![cfg(feature = "server")]

use anyhow::Result;
use half::f16;
use m40_llm::gguf::{GgmlDType, GgufModel, GgufScalar, GgufTensor, GgufValue};
use m40_llm::infer::LoadedModel;
use m40_llm::server::{app_router, AppState, GenerateRequest, GenerateResponse};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;

fn f16_bytes(v: f32) -> [u8; 2] {
    let h = f16::from_f32(v);
    h.to_bits().to_le_bytes()
}

/// Build a minimal GGUF model/bytes that satisfies server non-CUDA path:
/// - metadata: llama.embedding_length, llama.vocab_size
/// - tensors: tok_embeddings.weight [V,D] F16, output.weight [D,V] F16
/// - plus layer 0 weights required by map_standard_layer (unused, zeros)
fn make_min_gguf(vocab: usize, d_model: usize, hidden: usize) -> (GgufModel, Vec<u8>) {
    let mut metadata = std::collections::HashMap::new();
    metadata.insert(
        "llama.embedding_length".to_string(),
        GgufValue::Scalar(GgufScalar::U32(d_model as u32)),
    );
    metadata.insert(
        "llama.vocab_size".to_string(),
        GgufValue::Scalar(GgufScalar::U32(vocab as u32)),
    );

    let mut tensors: Vec<GgufTensor> = Vec::new();
    let mut weights: Vec<u8> = Vec::new();
    let mut add_tensor =
        |name: &str, dtype: GgmlDType, shape: &[u64], fill: &mut dyn FnMut(&mut Vec<u8>)| {
            let offset = weights.len() as u64;
            fill(&mut weights);
            tensors.push(GgufTensor {
                name: name.to_string(),
                dtype,
                shape: shape.to_vec(),
                offset,
            });
        };

    // tok_embeddings.weight: one-hot rows
    {
        let mut fill = |buf: &mut Vec<u8>| {
            for row in 0..vocab {
                for col in 0..d_model {
                    let v = if row == col { 1.0 } else { 0.0 };
                    buf.extend_from_slice(&f16_bytes(v));
                }
            }
        };
        add_tensor(
            "tok_embeddings.weight",
            GgmlDType::F16,
            &[vocab as u64, d_model as u64],
            &mut fill,
        );
    }

    // output.weight (lm_head): identity D x V
    {
        let mut fill = |buf: &mut Vec<u8>| {
            for r in 0..d_model {
                // rows = D
                for c in 0..vocab {
                    // cols = V
                    let v = if r == c { 1.0 } else { 0.0 };
                    buf.extend_from_slice(&f16_bytes(v));
                }
            }
        };
        add_tensor(
            "output.weight",
            GgmlDType::F16,
            &[d_model as u64, vocab as u64],
            &mut fill,
        );
    }

    // Minimal layer 0 weights for map_standard_layer checks (unused values)
    let zeros_f16 = |n: usize| -> Vec<u8> { (0..n).flat_map(|_| f16_bytes(0.0)).collect() };
    let mut add_zeros = |name: &str, shape: &[u64]| {
        let elems: usize = shape.iter().copied().product::<u64>() as usize;
        let mut fill = |buf: &mut Vec<u8>| buf.extend_from_slice(&zeros_f16(elems));
        add_tensor(name, GgmlDType::F16, shape, &mut fill);
    };
    // attention weights (shapes [D, N]; choose N=D for simplicity)
    add_zeros(
        "layers.0.attention.wq.weight",
        &[d_model as u64, d_model as u64],
    );
    add_zeros(
        "layers.0.attention.wk.weight",
        &[d_model as u64, d_model as u64],
    );
    add_zeros(
        "layers.0.attention.wv.weight",
        &[d_model as u64, d_model as u64],
    );
    add_zeros(
        "layers.0.attention.wo.weight",
        &[d_model as u64, d_model as u64],
    );
    // ffn: w3=gate, w1=up, w2=down
    add_zeros(
        "layers.0.feed_forward.w3.weight",
        &[d_model as u64, hidden as u64],
    );
    add_zeros(
        "layers.0.feed_forward.w1.weight",
        &[d_model as u64, hidden as u64],
    );
    add_zeros(
        "layers.0.feed_forward.w2.weight",
        &[hidden as u64, d_model as u64],
    );

    let gguf = GgufModel {
        version: 1,
        metadata,
        tensors,
        data_offset: 0,
    };
    (gguf, weights)
}

#[tokio::test]
async fn server_generate_smoke() -> Result<()> {
    let vocab = 256usize;
    let d_model = 256usize;
    let hidden = 16usize;
    let (gguf, bytes) = make_min_gguf(vocab, d_model, hidden);

    let mut model = LoadedModel::from_gguf(gguf, bytes, 0)?;
    // KV cache optional for non-CUDA path but harmless to allocate small
    let _ = model.allocate_kv_cache(16, 1);

    let state = Arc::new(AppState { model });
    let router = app_router(state);

    // Bind to an ephemeral local port
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr: SocketAddr = listener.local_addr()?;
    tokio::spawn(async move {
        axum::serve(listener, router.into_make_service())
            .await
            .unwrap();
    });

    // Request small generation
    let client = reqwest::Client::new();
    let url = format!("http://{}/generate", addr);
    let req = GenerateRequest {
        prompt: "A".to_string(),
        max_tokens: Some(3),
    };
    let resp = client.post(&url).json(&req).send().await.unwrap();
    assert!(resp.status().is_success());
    let jr: GenerateResponse = resp.json().await.unwrap();
    // Expect at least prompt length + 1
    assert!(jr.output.len() >= 2);
    Ok(())
}
