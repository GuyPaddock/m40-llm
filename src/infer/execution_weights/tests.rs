use std::{cell::RefCell, collections::HashSet, rc::Rc};

use super::*;

fn source() -> Q8SourceIdentity {
    Q8SourceIdentity::from_column_blocks(
        [7; 32],
        "fixture.weight".into(),
        5,
        Q8MatrixShape::new(35, 3).unwrap(),
    )
    .unwrap()
}

fn host_planes(layout: Q8ExecutionLayout) -> HostQ8ExecutionWeight {
    let source = source();
    let plan = layout.allocation_plan(source.shape()).unwrap();
    let planes = plan
        .planes()
        .iter()
        .enumerate()
        .map(|(plane, spec)| {
            (0..spec.bytes)
                .map(|index| (index.wrapping_mul(17) + plane) as u8)
                .collect()
        })
        .collect();
    HostQ8ExecutionWeight::new(Q8ExecutionKey { source, layout }, planes).unwrap()
}

/// Equal shapes alone cannot alias model/tensor/offset identities or execution
/// layouts; the source identity contains no transient device pointer.
#[test]
fn test_execution_keys_separate_model_tensor_shape_offset_and_layout() {
    let source = source();
    let base = Q8ExecutionKey {
        source: source.clone(),
        layout: Q8ExecutionLayout::ColumnBlocks,
    };
    let mut keys = HashSet::from([base.clone()]);
    for variant in 0..4 {
        let mut distinct = base.clone();
        match variant {
            0 => distinct.source.model_sha256[0] ^= 1,
            1 => distinct.source.tensor_name = "other.weight".into(),
            2 => distinct.source.byte_offset += 1,
            _ => distinct.source.shape = Q8MatrixShape::new(35, 4).unwrap(),
        }
        assert!(keys.insert(distinct));
    }
    for layout in [
        Q8ExecutionLayout::SeparateScalesValues,
        Q8ExecutionLayout::InputTiled { columns: 4 },
        Q8ExecutionLayout::InputTiled { columns: 8 },
    ] {
        assert!(keys.insert(Q8ExecutionKey {
            source: source.clone(),
            layout
        }));
    }
    assert!(!keys.insert(base));
    assert_eq!(keys.len(), 8);
}

/// A source borrows only its bounded tensor region, including every padded
/// column block; copying into an execution owner leaves the source unchanged.
#[test]
fn test_source_view_bounds_and_lossless_column_copy() {
    let identity = source();
    let bytes: Vec<u8> = (0..204).map(|i| (i * 19) as u8).collect();
    let mut blob = vec![0xdd; 5];
    blob.extend_from_slice(&bytes);
    blob.extend_from_slice(&[0xee; 9]);
    let view = Q8SourceView::new(&identity, &blob).unwrap();
    assert_eq!(view.blocks(), bytes);
    let packed = HostQ8ExecutionWeight::copy_column_blocks(view).unwrap();
    assert_eq!(packed.planes(), [bytes]);
    assert_eq!(view.blocks(), &blob[5..209]);
    assert!(Q8SourceView::new(&identity, &blob[..208]).is_err());
    let mut beyond = identity.clone();
    beyond.byte_offset = u64::MAX;
    assert!(Q8SourceView::new(&beyond, &blob).is_err());
}

/// Repacking must preserve all scale bit patterns and signed-value bytes, even
/// zero/negative scales, NaNs, and nonzero inactive K tails. Encoding is not math.
#[test]
fn test_separate_planes_preserve_every_source_byte() {
    let shape = Q8MatrixShape::new(35, 3).unwrap();
    let identity =
        Q8SourceIdentity::from_column_blocks([1; 32], "split".into(), 103, shape).unwrap();
    let scale_bits = [0_u16, 0x8000, 0x3c00, 0xbc00, 0x7c00, 0x7e31];
    let mut bytes = Vec::new();
    for (block, bits) in scale_bits.into_iter().enumerate() {
        bytes.extend_from_slice(&bits.to_le_bytes());
        bytes.extend((0..32).map(|lane| (block * 51 + lane * 7) as u8));
    }
    let view = Q8SourceView::from_tensor_bytes(&identity, &bytes).unwrap();
    let split = HostQ8ExecutionWeight::separate_scales_values(view).unwrap();
    assert_eq!(split.key().source, identity);
    let mut restored = Vec::new();
    for block in 0..6 {
        restored.extend_from_slice(&split.planes()[0][block * 2..block * 2 + 2]);
        restored.extend_from_slice(&split.planes()[1][block * 32..block * 32 + 32]);
    }
    assert_eq!(restored, bytes);
    assert_eq!(split.plan().resident_bytes(), bytes.len());
    assert!(Q8SourceView::from_tensor_bytes(&identity, &bytes[..203]).is_err());
    assert!(Q8SourceView::from_tensor_bytes(&identity, &[bytes, vec![0]].concat()).is_err());
}

/// Launch choices are bounded before enqueue rather than silently rounded or
/// expanded into an unbounded parameter search.
#[test]
fn test_separate_launch_accepts_only_fixed_experiment_configs() {
    for columns in [1, 4] {
        for threads in [128, 256] {
            let launch = Q8SeparateLaunch::new(columns, threads).unwrap();
            assert_eq!(launch.columns(), columns);
            assert_eq!(launch.threads(), threads);
        }
    }
    for (columns, threads) in [(0, 128), (2, 128), (8, 256), (1, 0), (1, 64), (4, 512)] {
        assert!(Q8SeparateLaunch::new(columns, threads).is_err());
    }
}

/// Every supported storage width must reconstruct the original bytes exactly,
/// including scale bit patterns and stored K tails; padded columns/headers are
/// zero rather than uninitialized data interpreted as weights.
#[test]
fn test_input_tiles_preserve_bytes_and_zero_padding() {
    let shape = Q8MatrixShape::new(35, 33).unwrap();
    let identity =
        Q8SourceIdentity::from_column_blocks([2; 32], "tiled".into(), 103, shape).unwrap();
    let mut bytes = Vec::new();
    for block in 0..shape.block_count().unwrap() {
        let bits = [0_u16, 0x8000, 0x3c00, 0xbc00, 0x7c00, 0x7e31][block % 6];
        bytes.extend_from_slice(&bits.to_le_bytes());
        bytes.extend((0..32).map(|lane| (block * 51 + lane * 7) as u8));
    }
    let source = Q8SourceView::from_tensor_bytes(&identity, &bytes).unwrap();
    for columns in [1, 2, 4, 8, 16, 32] {
        let tiled = HostQ8ExecutionWeight::tile_inputs(source, columns).unwrap();
        assert_eq!(tiled.key().source, identity);
        let geometry = tiled.plan().tile().unwrap();
        let tiles = &tiled.planes()[0];
        let mut restored = vec![0; bytes.len()];
        for qb in 0..shape.blocks_per_column() {
            for tile in 0..geometry.output_tiles {
                let base = (qb * geometry.output_tiles + tile) * geometry.record_bytes;
                assert!(tiles[base + columns * 2..base + geometry.values_offset]
                    .iter()
                    .all(|&b| b == 0));
                for local in 0..columns {
                    let col = tile * columns + local;
                    let scale = &tiles[base + local * 2..base + local * 2 + 2];
                    if col < shape.output_dim() {
                        let dest = (col * shape.blocks_per_column() + qb) * 34;
                        restored[dest..dest + 2].copy_from_slice(scale);
                        for lane in 0..32 {
                            restored[dest + 2 + lane] =
                                tiles[base + geometry.values_offset + lane * columns + local];
                        }
                    } else {
                        assert_eq!(scale, [0, 0]);
                        for lane in 0..32 {
                            assert_eq!(
                                tiles[base + geometry.values_offset + lane * columns + local],
                                0
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(restored, bytes, "columns={columns}");
        assert_eq!(tiles.len(), tiled.plan().resident_bytes());
    }
    for columns in [0, 3, 64] {
        assert!(HostQ8ExecutionWeight::tile_inputs(source, columns).is_err());
    }
}

/// The CUDA experiment offers twelve launch shapes, not every storage descriptor
/// width or arbitrary thread count. Reject unsupported choices before enqueue.
#[test]
fn test_tiled_launch_accepts_only_twelve_experiment_configs() {
    for access in [Q8TiledAccess::Vector, Q8TiledAccess::CoalescedLanes] {
        for columns in [4, 8, 16] {
            for threads in [128, 256] {
                let launch = Q8TiledLaunch::new(columns, threads, access).unwrap();
                assert_eq!(launch.columns(), columns);
                assert_eq!(launch.threads(), threads);
                assert_eq!(launch.access(), access);
            }
        }
        for (columns, threads) in [
            (0, 128),
            (1, 128),
            (2, 128),
            (32, 256),
            (4, 0),
            (8, 64),
            (16, 512),
        ] {
            assert!(Q8TiledLaunch::new(columns, threads, access).is_err());
        }
    }
}

/// Real GGUF sources must be rank-two Q8 with complete rows; synthetic ragged
/// rows require the explicit padded-column constructor, never implicit sizing.
#[test]
fn test_gguf_sources_reject_unsupported_shapes_and_dtypes() {
    let valid = GgufTensor {
        name: "gate.weight".into(),
        dtype: GgmlDType::Q8_0,
        shape: vec![64, 7],
        offset: 32,
    };
    let identity = Q8SourceIdentity::from_gguf_tensor([1; 32], &valid).unwrap();
    assert_eq!(identity.shape().source_bytes().unwrap(), 476);
    assert_eq!(identity.byte_offset(), 32);
    assert_eq!(identity.dtype(), GgmlDType::Q8_0);
    for (dtype, shape) in [
        (GgmlDType::F16, vec![64, 7]),
        (GgmlDType::Q8_0, vec![64]),
        (GgmlDType::Q8_0, vec![35, 7]),
        (GgmlDType::Q8_0, vec![0, 7]),
        (GgmlDType::Q8_0, vec![64, 0]),
    ] {
        let tensor = GgufTensor {
            dtype,
            shape,
            ..valid.clone()
        };
        assert!(Q8SourceIdentity::from_gguf_tensor([1; 32], &tensor).is_err());
    }
    assert!(
        Q8SourceIdentity::from_column_blocks([0; 32], " ".into(), 0, identity.shape()).is_err()
    );
}

/// Region sizes account for per-column K tails, padded output tiles, aligned
/// scale headers, and split planes, without dropping stored tail payloads.
#[test]
fn test_layout_plans_account_for_tails_alignment_and_planes() {
    let shape = Q8MatrixShape::new(35, 3).unwrap();
    assert_eq!(shape.block_count().unwrap(), 6);
    let split = Q8ExecutionLayout::SeparateScalesValues
        .allocation_plan(shape)
        .unwrap();
    assert_eq!(
        split.planes(),
        [
            Q8PlaneSpec {
                kind: Q8PlaneKind::Scales,
                bytes: 12
            },
            Q8PlaneSpec {
                kind: Q8PlaneKind::Values,
                bytes: 192
            },
        ]
    );
    assert_eq!(split.resident_bytes(), 204);
    assert_eq!(split.padding_bytes(), 0);
    let tile = Q8ExecutionLayout::InputTiled { columns: 4 }
        .allocation_plan(shape)
        .unwrap();
    assert_eq!(tile.resident_bytes(), 288);
    assert_eq!(tile.padding_bytes(), 84);
    assert_eq!(
        tile.tile(),
        Some(Q8InputTileGeometry {
            columns: 4,
            output_tiles: 1,
            values_offset: 16,
            record_bytes: 144
        })
    );
    let qwen = Q8MatrixShape::new(2048, 11008).unwrap();
    let native = Q8ExecutionLayout::ColumnBlocks
        .allocation_plan(qwen)
        .unwrap();
    let tile8 = Q8ExecutionLayout::InputTiled { columns: 8 }
        .allocation_plan(qwen)
        .unwrap();
    assert_eq!(native.resident_bytes(), 23_953_408);
    assert_eq!(native.resident_bytes(), tile8.resident_bytes());
}

/// Invalid dimensions/layouts, arithmetic overflow, or malformed host planes
/// must fail before a GPU allocator or pointer arithmetic can be reached.
#[test]
fn test_invalid_plans_and_host_regions_are_rejected() {
    for (k, n) in [
        (0, 1),
        (1, 0),
        (usize::MAX, 2),
        (usize::MAX, 1),
        (1, usize::MAX),
    ] {
        assert!(Q8MatrixShape::new(k, n).is_err());
    }
    for columns in [0, 3, 64] {
        assert!(Q8ExecutionLayout::InputTiled { columns }
            .allocation_plan(source().shape())
            .is_err());
    }
    let huge = Q8MatrixShape::new(32, usize::MAX / 34).unwrap();
    assert!(Q8ExecutionLayout::InputTiled { columns: 4 }
        .allocation_plan(huge)
        .is_err());
    let key = Q8ExecutionKey {
        source: source(),
        layout: Q8ExecutionLayout::SeparateScalesValues,
    };
    assert!(HostQ8ExecutionWeight::new(key.clone(), vec![vec![0; 204]]).is_err());
    assert!(HostQ8ExecutionWeight::new(key, vec![vec![0; 11], vec![0; 192]]).is_err());
}

#[derive(Default)]
struct AllocationState {
    attempts: usize,
    successful: usize,
    dropped: usize,
    live_bytes: usize,
    uploads_attempted: usize,
    uploads_completed: usize,
}

struct FakeBuffer {
    state: Rc<RefCell<AllocationState>>,
    contents: RefCell<Vec<u8>>,
}

impl Drop for FakeBuffer {
    fn drop(&mut self) {
        let mut state = self.state.borrow_mut();
        state.live_bytes -= self.contents.borrow().len();
        state.dropped += 1;
    }
}

fn prepare_fake(
    host: &HostQ8ExecutionWeight,
    state: Rc<RefCell<AllocationState>>,
    fail_allocate: Option<usize>,
    fail_upload: Option<usize>,
) -> Result<Vec<FakeBuffer>> {
    let allocation_state = state.clone();
    prepare_buffers(
        host,
        move |spec| {
            let mut counters = allocation_state.borrow_mut();
            counters.attempts += 1;
            ensure!(
                Some(counters.attempts) != fail_allocate,
                "injected allocation failure"
            );
            counters.successful += 1;
            counters.live_bytes += spec.bytes;
            Ok(FakeBuffer {
                state: allocation_state.clone(),
                contents: RefCell::new(vec![0; spec.bytes]),
            })
        },
        move |buffer, data| {
            let mut counters = state.borrow_mut();
            counters.uploads_attempted += 1;
            ensure!(
                Some(counters.uploads_attempted) != fail_upload,
                "injected upload failure"
            );
            buffer.contents.borrow_mut().copy_from_slice(data);
            counters.uploads_completed += 1;
            Ok(())
        },
    )
}

/// Successful setup retains exactly its requested bytes and uploaded data until
/// ownership drops, then frees every plane without a caller cleanup protocol.
#[test]
fn test_prepared_buffers_own_data_and_release_all_planes() {
    let host = host_planes(Q8ExecutionLayout::SeparateScalesValues);
    let state = Rc::new(RefCell::new(AllocationState::default()));
    let buffers = prepare_fake(&host, state.clone(), None, None).unwrap();
    assert_eq!(state.borrow().live_bytes, host.plan().resident_bytes());
    for (buffer, data) in buffers.iter().zip(host.planes()) {
        assert_eq!(&*buffer.contents.borrow(), data);
    }
    drop(buffers);
    let state = state.borrow();
    assert_eq!(state.live_bytes, 0);
    assert_eq!(state.dropped, state.successful);
    assert_eq!(state.uploads_completed, 2);
}

/// Failure at either allocation boundary releases prior ownership; failed
/// allocation attempts are distinct from allocations that actually need free.
#[test]
fn test_partial_allocation_failure_releases_prior_planes() {
    let host = host_planes(Q8ExecutionLayout::SeparateScalesValues);
    for fail_at in [1, 2] {
        let state = Rc::new(RefCell::new(AllocationState::default()));
        assert!(prepare_fake(&host, state.clone(), Some(fail_at), None).is_err());
        let state = state.borrow();
        assert_eq!(state.live_bytes, 0);
        assert_eq!(state.successful, fail_at - 1);
        assert_eq!(state.dropped, state.successful);
        assert_eq!(state.uploads_completed, state.successful);
    }
}

/// Transfer failure frees both the newly allocated uncommitted plane and every
/// earlier uploaded plane, whether the first or second transfer fails.
#[test]
fn test_partial_upload_failure_releases_all_allocated_planes() {
    let host = host_planes(Q8ExecutionLayout::SeparateScalesValues);
    for fail_at in [1, 2] {
        let state = Rc::new(RefCell::new(AllocationState::default()));
        assert!(prepare_fake(&host, state.clone(), None, Some(fail_at)).is_err());
        let state = state.borrow();
        assert_eq!(state.live_bytes, 0);
        assert_eq!(state.successful, fail_at);
        assert_eq!(state.dropped, state.successful);
        assert_eq!(state.uploads_attempted, fail_at);
        assert_eq!(state.uploads_completed, fail_at - 1);
    }
}
