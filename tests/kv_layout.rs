use m40_llm::cuda::KVCache;

#[test]
fn test_kv_index_math() {
    let kv = KVCache::new_with_context(&m40_llm::cuda::CudaContext::new(-1).unwrap(), 4, 2, 3, 5)
        .unwrap(); // non-CUDA unit-test math only; context value irrelevant here

    // elems_per_token = 3 * 5 = 15
    assert_eq!(kv.elems_per_token(), 15);

    // base(seq=0, token=0) = 0
    assert_eq!(kv.base_offset_elems(0, 0), 0);

    // base(seq=0, token=1) = 1 * 15 = 15
    assert_eq!(kv.base_offset_elems(0, 1), 15);

    // base(seq=1, token=0) = (1*4 + 0) * 15 = 4 * 15 = 60
    assert_eq!(kv.base_offset_elems(1, 0), 60);

    // index(seq=1, token=2, head=2, dim=4)
    // base = (1*4 + 2) * 15 = 6 * 15 = 90
    // head*head_dim = 2 * 5 = 10
    // dim = 4
    // total = 90 + 10 + 4 = 104
    assert_eq!(kv.index_elems(1, 2, 2, 4), 104);
}
