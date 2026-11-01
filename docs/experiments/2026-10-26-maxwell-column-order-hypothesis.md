# Fixed Gate/Up Column Visitation Diagnostic

## Question And Boundary

Original-plane checksum reads retain about 94% of gate/up time. Wider
payload packets, output/role ownership, cache hints and physical column
gaps have failed their bounded screens. Do not retune or combine them.
One different mechanism is logical column visitation: current CTA b reads
column b. Test whether a fixed permutation changes read cost without
changing weights, pitch, load width, arithmetic ownership or block geometry.

This is an address-order experiment, not evidence of partition camping,
SM affinity, CTA scheduling order, cache hit rates or DRAM transactions.
CUDA does not promise a physical block execution order. A positive timing
result would justify investigation, not an explanation of undocumented
hardware or an inference speed claim.

## One Fixed Permutation

- For N divisible by 32, let width=N/32. Sequential maps b to b; transpose
  maps b to `(b % 32) * width + b / 32`. Qwen N=11008 gives width=344.
  Both launch N CTAs per row, 128 threads and 1024 shared bytes, on decode.
- Use one new kernel with a dynamic sequential/transpose selector. Preserve
  original four-value tasks, u32 payload and u16 scale reads, shared
  reduction, owners and allocation spans. No padding, new weight plane,
  gap, cache operator or thread/warp-count change. Unsupported N fails.
- Tag each final modular checksum with physical `blockIdx.x`. CPU validation
  subtracts the independently derived inverse-map owner before comparing
  original GGUF sums. This makes ignored/wrong order selectors observable
  without extra buffers or debug-only stores. Both new arms include tags.
  Preserve the original untagged reader as a second unchanged baseline.
- Extend existing `--ffn-weight-read` with exclusive
  `--ffn-weight-read-column-order`, incompatible with column-gap mode.
  No inference output, generation selector or scheduler change.

## Validation And Screen

Test permutation/inverse coverage, exact tagged and normalized CPU sums,
arbitrary scale bits, wrapping, multiple dynamic K/N, rows1/4, guards,
immutable planes, actual dispatch, async producers, scratch reuse, invalid
dimensions/alignment/alias/address overflow and cleanup. Require CUDA on/off,
affected parity/regressions, non-CUDA warnings, Clippy and formatting.
Inspect actual sm_52 resources and unchanged original reader/gate-up/down
instruction words. Fewer source operations is not a gain claim.

Use the shared M02 layers 0/17/35 and rows1/4, with original prepared
down/vocabulary. Five alternating pairs, ten iterations per stage, bounded
existing warmups. Time original reader, new sequential reader, transpose
reader, real gate/up/SwiGLU, unchanged down and complete real FFN serially.
Record GPU isolation/clocks, dispatch, spans and zero additional weight
storage; checksum outputs are never sent into real down projection.

Advance only if transpose improves **one-row read throughput >=5% with
positive paired lower bounds at all three layers against both baselines**.
A slower new control cannot manufacture success. Otherwise close this fixed
permutation without tile-size, permutation, geometry, layout, cache or
register rescue. These are surrogate costs, not pure DRAM or inference bounds.

Even a passing diagnostic needs a separate inference experiment with the
unchanged >=5% complete-FFN gate, actual512/frozen dense-valid quality and
fresh paired HTTP gates. Keep the full **53.06 HTTP tok/s** and **1.30x fresh
Ollama** requirements. No numerical, sparse, approximate or scheduling
substitution; runtime and KV defaults remain unchanged.

## Atomic Delivery

1. `docs(research): bound column order study`: this contract.
2. `feat(bench): probe fixed FFN column order`: checksum primitive, existing
   runner integration and behavioral tests; depends on 1.
3. `docs(perf): record FFN column order costs`: clean release resources,
   raw/audited results and keep/close decision; depends on tested 2.
4. Only a qualifying diagnostic permits planning a separate inference
   consumer. Preserve code before any later removal.
