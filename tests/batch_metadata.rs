use m40_llm::infer::{BatchMetadata, BatchSequence, LengthBucket};

#[test]
fn batch_metadata_builds_prefix_offsets() {
    let meta = BatchMetadata::new(vec![
        BatchSequence {
            seq_len: 7,
            query_len: 7,
            kv_len: 7,
        },
        BatchSequence {
            seq_len: 3,
            query_len: 1,
            kv_len: 9,
        },
        BatchSequence {
            seq_len: 128,
            query_len: 16,
            kv_len: 128,
        },
    ])
    .unwrap();

    assert_eq!(meta.token_offsets(), &[0, 7, 10, 138]);
    assert_eq!(meta.q_offsets(), &[0, 7, 8, 24]);
    assert_eq!(meta.kv_offsets(), &[0, 7, 16, 144]);
    assert_eq!(meta.attention_offsets(), &[0, 49, 58, 2106]);
    assert_eq!(meta.total_tokens(), 138);
    assert_eq!(meta.total_q_tokens(), 24);
    assert_eq!(meta.total_kv_tokens(), 144);
    assert_eq!(meta.total_attention_cells(), 2106);

    assert_eq!(meta.offsets()[1].token_offset, 7);
    assert_eq!(meta.offsets()[1].q_offset, 7);
    assert_eq!(meta.offsets()[1].kv_offset, 7);
    assert_eq!(meta.offsets()[1].attention_offset, 49);
}

#[test]
fn batch_metadata_groups_by_length_bucket() {
    let meta = BatchMetadata::new(vec![
        BatchSequence {
            seq_len: 64,
            query_len: 8,
            kv_len: 64,
        },
        BatchSequence {
            seq_len: 65,
            query_len: 8,
            kv_len: 65,
        },
        BatchSequence {
            seq_len: 512,
            query_len: 32,
            kv_len: 512,
        },
        BatchSequence {
            seq_len: 130,
            query_len: 1,
            kv_len: 130,
        },
    ])
    .unwrap();

    let buckets = meta.buckets();
    assert_eq!(buckets.len(), 4);
    assert_eq!(buckets[0].bucket, LengthBucket::Tokens1To64);
    assert_eq!(buckets[0].sequence_indices, vec![0]);
    assert_eq!(buckets[1].bucket, LengthBucket::Tokens65To128);
    assert_eq!(buckets[1].sequence_indices, vec![1]);
    assert_eq!(buckets[2].bucket, LengthBucket::Tokens129To256);
    assert_eq!(buckets[2].sequence_indices, vec![3]);
    assert_eq!(buckets[3].bucket, LengthBucket::Tokens257To512);
    assert_eq!(buckets[3].sequence_indices, vec![2]);
}

#[test]
fn batch_metadata_rejects_invalid_lengths() {
    let err = BatchMetadata::new(vec![BatchSequence {
        seq_len: 4,
        query_len: 5,
        kv_len: 5,
    }])
    .unwrap_err();
    assert!(err.to_string().contains("query_len 5 exceeds seq_len 4"));

    let err = BatchMetadata::new(vec![BatchSequence {
        seq_len: 4,
        query_len: 4,
        kv_len: 3,
    }])
    .unwrap_err();
    assert!(err.to_string().contains("kv_len 3 is smaller"));
}

#[test]
fn length_bucket_boundaries_are_stable() {
    assert_eq!(LengthBucket::for_len(1), LengthBucket::Tokens1To64);
    assert_eq!(LengthBucket::for_len(64), LengthBucket::Tokens1To64);
    assert_eq!(LengthBucket::for_len(65), LengthBucket::Tokens65To128);
    assert_eq!(LengthBucket::for_len(1024), LengthBucket::Tokens513To1024);
    assert_eq!(LengthBucket::for_len(8193), LengthBucket::Tokens8193Plus);
    assert_eq!(LengthBucket::Tokens257To512.max_len(), Some(512));
    assert_eq!(LengthBucket::Tokens8193Plus.max_len(), None);
}
