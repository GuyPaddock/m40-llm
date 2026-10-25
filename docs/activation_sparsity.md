# SwiGLU Input-Sparsity Research

C23 thresholds the completed SwiGLU vector before the down projection. Gate/up
weights and computation stay unchanged. This changes model semantics and is
not an unchanged-model engine optimization. There is no runtime integration
or default change yet.

## Selection Contract

`infer::activation_sparsity::ActivationThreshold` accepts finite nonnegative
magnitudes. Values with `abs(value) <= threshold` become zero; nonfinite values
are retained so selection cannot hide upstream numerical failures. Retained
values keep their exact FP32 bits and increasing original input indices.
Zero threshold omits zeros only. An empty retained set is valid and projects
to zero; an empty source vector is invalid.

Quantile calibration consumes caller-supplied finite observations only. It
does not discover models or mix calibration and held-out data. Tied magnitudes
may exceed the target sparsity, so reports must measure the retained count.
Per-layer model calibration remains gated on a useful device consumer.

## Bounded Implementation Sequence

1. `feat(infer): define sparse activation selection`: CPU threshold/index
   contract and behavioral tests. Depends on the frozen C19 evaluation split.
2. `perf(cuda): skip Q8 down-projection inputs`: reusable RAII compaction
   workspace, deterministic GPU selection, and a decode-only consumer of the
   retained C10 input-tiled layout. Compare against the same zeroed dense
   inputs and preserve zero-threshold parity. Depends on the selection contract.
3. `test(bench): screen sparse activation projections`: release real-down-tensor
   probe, including selection/indexing time, resident duplicate layout bytes,
   achieved sparsity and requested weight loads. Depends on CUDA parity.
4. Record measurements before any removal. Only proceed to per-layer
   calibration and model-wide opt-in evaluation if the complete selection plus
   projection path reduces measured latency. No browser/held-out tuning.

This uses dense Q8 weights with sparse inputs, not C21's rejected 16-of-32
sparse-weight representation. The initial device screen is single-row decode,
tile widths8/16 and threads128/256, four configurations total. Test requested
activation sparsity 0/25/40/50/60%; do not reopen dense-layout tuning. Actual
DRAM traffic is not implied by a retained-value count. Full latency and later
quality must justify any continuing integration cost.

## Device Probe Contract

`DeviceSparseInputWorkspace` owns reusable K-entry u32 indices, K-entry FP32
values and one u32 count, exactly `8*K+4` requested bytes. A single 256-thread
CTA compacts contiguous input ranges with deterministic prefix sums. The next
kernel reads the device count and only loads weights for retained indices;
there is no host count read, allocation or stream synchronization in enqueue.
Both operations use decode-stream ordering. Diagnostic readback drains decode.

The Q8 consumer shares the retained dense-tiled value loads and FP32 reduction.
It accepts only tile widths8/16, threads128/256, one input row and matching
typed layout/dimensions. Source weights are unchanged. Workspace, weights and
outputs must outlive stream completion, including failure paths. This is a
standalone research API, not a generation flag or a trained threshold artifact.

```bash
source scripts/dev-env.sh
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda --test sparse_activation_cuda \
  --test q8_tiled_projection -- --include-ignored --test-threads=1
```

Hardware checks compare GPU-selected indices/value bits with the CPU contract,
guard output tails, reject invalid calls without launches, and compare every
output against native Q8 with the same zeroed input. Independent FP64 samples
cover real `K=11008,N=2048` down dimensions. No speed or model-quality conclusion
follows from these numerical tests.
