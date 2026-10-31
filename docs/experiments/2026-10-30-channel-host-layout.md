# Channel-selection host representation checkpoint

The nine independent research worktrees are active at
`/tmp/m40-research-1030/<topic>`, with branch names from the source plan.
Each has a committed scoped protocol and a separate target directory.
The original worktree remains at `0eee578`.

## Implemented unit

`infer::q4_channels::Q4ChannelWeights` transposes Q4_0 down-projection
payloads without changing quantization. For a logical [N,K] matrix, codes
occupy [K,ceil(N/2)] bytes, with adjacent outputs in low/high nibbles.
Original FP16 scale bits occupy [K/32,N,2] bytes. K must be a positive
multiple of 32. Odd N adds one zero padding nibble per input channel.
For the planned even N=2048 shape, payload size equals native Q4_0 size.
Dimensions are separate metadata; reported payload bytes exclude Vec headers.

The CPU API exposes upload planes, native-byte reconstruction and indexed
weight decoding. It has no runtime dispatch or device allocations. Tests
check all 65,536 FP16 scale patterns, arbitrary payload round trips, scattered
channels across three scale groups, negative scales, odd-output padding,
invalid dimensions, source byte counts, overflow and out-of-bounds reads.

Full warning-as-error non-CUDA suite: 223 passed, zero failed, 111 suite
summaries. The new representation has three passing tests.
See [raw results](2026-10-30-channel-host-noncuda.log.gz).
Formatting and Clippy for the non-CUDA library and new integration test pass
with warnings denied. See [lint output](2026-10-30-channel-host-clippy.log.gz).

## Inputs and environment

Retained fixture: the original worktree's
`target/research-fixtures/m02-ffn-inputs.json`, SHA-256
`eb51828ef778189567ae01fdd910659025c92ae6f5b2d76c989c314e650d0687`.
Its completion report confirms this hash. It contains 21 rows per layer
for original layers 0/17/35 and dimensions 2048/11008. Later consumers must
validate provenance and distinct rows through the existing fixture reader.
Model SHA-256 is
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.

The M40 reported 23040 MiB total, zero used, and no compute processes at
initial inspection. NVCC is 12.8.93. This inspection is not a reservation;
check the device again before each hardware run.

The first full test build and lint attempt exhausted the 16 GiB /tmp tmpfs.
Its [test log](2026-10-30-channel-host-storage-failure.log.gz) is preserved.
Each worktree's target is now a symlink to an independent directory under
the original project's `target/research-1030-builds/<topic>`. Source worktrees
remain in /tmp. No build outputs are shared across topics. The successful
full-suite rerun used `cargo test --no-default-features --locked -j 4`
with `RUSTFLAGS='-D warnings'` after sourcing `scripts/dev-env.sh`.

## Next

Implement source-bound Q8-to-Q4 fixtures, selected gate/up and down CUDA
consumers, guarded RAII uploads and complete-FFN measurements. Reuse the
existing fixture/provenance and paired-statistics machinery. Compare native
and repacked layouts on identical Q4 bytes and supplied indices before
implementing screening. No speed, model-quality or MoE conclusion follows
from host representation tests.
