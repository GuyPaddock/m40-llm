use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{ffi_debug_read_kv_token, CudaStream},
    infer::LoadedModel,
};

pub(super) struct HeadWords {
    pub key: Vec<u16>,
    pub value: Vec<u16>,
}

pub(super) fn append_token(
    heads: &mut [HeadWords],
    key: &[u16],
    value: &[u16],
    dimension: usize,
) -> Result<()> {
    ensure!(
        !heads.is_empty()
            && dimension > 0
            && key.len() == heads.len() * dimension
            && value.len() == key.len(),
        "invalid head/token shape"
    );
    for (index, head) in heads.iter_mut().enumerate() {
        head.key
            .extend_from_slice(&key[index * dimension..(index + 1) * dimension]);
        head.value
            .extend_from_slice(&value[index * dimension..(index + 1) * dimension]);
    }
    Ok(())
}

pub(super) fn read_layer(model: &LoadedModel, layer: u32, length: usize) -> Result<Vec<HeadWords>> {
    let cache = model.kv_cache.as_ref().context("capture KV absent")?;
    ensure!(
        !cache.is_compressed() && length > 0 && length <= cache.max_seq_len() as usize,
        "capture requires populated dense KV"
    );
    model.cuda.synchronize_stream(CudaStream::Decode)?;
    model.cuda.synchronize_stream(CudaStream::Prefill)?;
    let slot = model.kv_cache_physical_slot_for_layer_sequence(layer, 0)?;
    ensure!(
        cache.debug_sequence_length(&model.cuda, slot)? == length as u32,
        "capture KV valid length mismatch"
    );
    let dimension = cache.head_dim() as usize;
    let count = cache.num_heads() as usize;
    let mut heads: Vec<_> = (0..count)
        .map(|_| HeadWords {
            key: Vec::with_capacity(length * dimension),
            value: Vec::with_capacity(length * dimension),
        })
        .collect();
    let mut key = vec![0_u16; count * dimension];
    let mut value = key.clone();
    for position in 0..length {
        let rc = unsafe {
            ffi_debug_read_kv_token(
                &model.cuda,
                cache,
                slot,
                position as u32,
                key.as_mut_ptr().cast(),
                value.as_mut_ptr().cast(),
            )
        };
        ensure!(rc == 0, "capture KV read failed: {rc}");
        append_token(&mut heads, &key, &value, dimension)?;
    }
    Ok(heads)
}
