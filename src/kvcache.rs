pub struct KVCache {
    inner: Arc<KVCacheInner>,
}

impl KVCache {
    pub fn num_heads(&self) -> u32 {
        self.inner.num_heads
    }
    pub fn head_dim(&self) -> u32 {
        self.inner.head_dim
    }
}

#[derive(Debug)]
struct KVCacheInner {
    // Layout: [seq][token][head][head_dim]
    // - seq in [0, max_batch_size)
    // - token in [0, max_seq_len)
    // - head in [0, num_heads)
    // - head_dim in [0, head_dim)
    // Strides (elements):
    //   elems_per_token = num_heads * head_dim
    //   base(seq, token) = (seq * max_seq_len + token) * elems_per_token
    #[allow(dead_code)]
    //   index(seq, token, head, dim) = base + head * head_dim + dim
    max_seq_len: u32,
    _max_batch_size: u32,
    num_heads: u32,
    head_dim: u32,
    #[cfg(feature = "cuda")]
    raw: NonNull<ffi::M40llmKVCache>,
    #[cfg(not(feature = "cuda"))]
    k: Mutex<Vec<half::f16>>, // length = max_seq_len * max_batch_size * elems_per_token
    #[cfg(not(feature = "cuda"))]
    v: Mutex<Vec<half::f16>>, // same length as k
