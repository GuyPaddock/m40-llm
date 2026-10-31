# Selected-channel Q4 CUDA correctness checkpoint

The oracle execution primitive is implemented and passes M40 numerical and
ownership tests with cuBLAS enabled and disabled. This establishes operator
correctness on synthetic fixtures, not speed or model quality.

## Fixed implementation

The research-only DeviceQ4SelectedFfn owns native gate/up/down, channel-major
down scales/codes, immutable per-row indices, compact hidden outputs and
32 down partials per output. Both timing arms will retain identical owners.
Supported row counts are 1/2/4/8, K/H positive multiples of 32, and a positive
retained count no greater than H. Selections must be sorted, unique and in
bounds within each row. Empty selections are rejected explicitly.

Three kernels execute on the requested stream:

1. A 128-thread block evaluates gate/up for one retained channel and writes
   its SwiGLU value into the compact hidden array.
2. Each 256-thread down block owns adjacent outputs and one of 32 partitions
   of the selected channels. Native and channel-major variants differ only
   in weight addressing; both accumulate in FP32.
3. A final kernel reduces all 32 partials in a fixed order.

Source weights remain Q4_0 throughout. Repacking does not change codes or
scales. Each invocation includes the final reduction and overwrites every
scratch output, including zero-contribution partitions. No warm allocation,
copy, stream wait, synchronization or cuBLAS operation is introduced.
The unsafe enqueue contract requires live disjoint input/output spans,
ordered producers and serialized workspace use across streams.
The checked upload validates shape, source lengths and selection before
device allocation. Each successfully allocated buffer immediately has RAII
ownership; complete teardown returns tracked bytes to the initial count.

## Verification

Tests compare every output to an independent FP64 masked FFN calculation
using native Q4 bytes. Native and packed CUDA variants also match each
other bitwise. The tolerance remains 1e-3 + 2e-5 * abs(reference).

Small fixtures cover K=64,H=96, rows 1/2/4/8 and retained counts 1/17/96.
The real Qwen dimensions K=2048,H=11008 use synthetic weights and row one
at retained counts 1101 and 11008. This includes full retention, but is not
a real-model parity result. Tests cover distinct per-row selections,
both execution streams with other-stream producers, repeated scratch use,
input/output guards, immutable weight/index readback, invalid indices,
source lengths, shapes, pointers and aliases.

| Check | Result | Evidence |
| --- | --- | --- |
| New M40 tests, cuBLAS on | 2 pass, 0 skipped | [log](2026-10-30-q4-selected-hardware.log.gz) |
| New M40 tests, cuBLAS off | 2 pass, 0 skipped | [log](2026-10-30-q4-selected-no-cublas.log.gz) |
| Existing Q4 projection checks | 2 pass | [log](2026-10-30-q4-selected-projection.log.gz) |
| Existing compound forward/ownership | 3 pass, ignored tests explicitly included | [log](2026-10-30-q4-selected-compound-forward.log.gz) |
| Attention parity grid | 11 pass | [log](2026-10-30-q4-selected-attention.log.gz) |
| Non-CUDA, warnings denied | 223 pass | [log](2026-10-30-q4-selected-noncuda.log.gz) |
| CUDA/server all-target Clippy | pass, warnings denied | [log](2026-10-30-q4-selected-clippy.log.gz) |

Release tests use NVCC 12.8.93 and sm_52. The compiler emits its upstream
architecture-deprecation notice. The M40 had no other compute processes
before the new hardware tests. Hardware commands ran serially.

## Legacy fixture failure

The first forward smoke invocation fails during KV allocation because the
legacy head_dim=4 fixture selects the current compressed default. It never
reaches forward execution. An attempted M40LLM_KV_COMPRESS_MODE environment
override has no effect on this test; the test requires the explicit runtime
configuration API. Both failed invocations remain in the
[first log](2026-10-30-q4-selected-forward.log.gz) and
[override log](2026-10-30-q4-selected-forward-dense.log.gz).

The same test fails with the same error on branch research/1030-ffn-fusion
at 35bd531, whose code is unchanged from 0eee578:
[base reproduction](2026-10-30-q4-selected-forward-baseline.log.gz).
The established compound fixtures explicitly select dense KV and pass.
No unrelated fixture or runtime default was changed to hide this failure.

## Next gate

Add source-bound Q8-to-Q4 fixtures and the matched complete-FFN benchmark,
using the frozen 21-row captures and all planned retention/layout patterns.
Record quantization separately from lossless repacking. Include dense Q4,
selection metadata, complete scratch and both resident layouts in accounting.
No screening, runtime integration or speed conclusion is admitted yet.
