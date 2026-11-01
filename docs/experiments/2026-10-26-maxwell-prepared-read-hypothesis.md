# Prepared Projection Read Diagnostic

## Independent Question

Gate/up checksum reads cost about 94% of the real operation on its fixed
one-column/128-thread mapping. That does not establish the cost of prepared
down/vocabulary, whose four-column/eight-value work assignment and shared
reduction differ. Inspect their own original planes before another arithmetic
probe. This is not a rescue of the closed warp-finish or activation-load
experiments and does not combine rejected candidates.

## Fixed Work

Use the existing release `q8_execution_bench` and frozen M02 fixture/report,
original Qwen2.5-3B Q8_0 weights and qualified 37-tensor down/vocabulary
cache. Measure down on layers 0/17/35 and vocabulary using the existing
layer-0 normalized-input surrogates, rows 1/4. No model runtime selector,
KV, quality semantics or default changes.

The checksum-only kernel retains four columns per CTA, eight payload bytes
per task, separate original u16 scales and signed-byte payload planes. Keep
down at 256 threads for one row and 128 for four; vocabulary uses 128.
Retain the original shared reduction tree and its block barriers. Replace
FP32 activations/dequant/FMA with wrapping u32 checksum additions of both
payload words and the scale bits once per task, four times per Q8 block.
Output is one checksum per output column/row, the same byte count as the
real FP32 output. Read every plane bit, including scale bits; verify against
original interleaved GGUF bytes rather than copying the device algorithm.

Do not add geometry, cache, unroll, register-cap or arithmetic alternatives.
Inspect compiled loads, registers, dynamic shared memory, barriers and spills.
Require unchanged encoded control words. Check checksum guards/reuse,
immutable weights/cache, original real-operation CPU bounds, asynchronous
producer ordering, invalid spans/tails and final allocation cleanup.

## Timing And Limits

Use five alternating pairs with ten repeated enqueues per stage. Warm the
real operations for at least 1000 GPU ms and checksum for at least 100 ms
using finite existing budgets. Down cases record separate checksum, real
gate/up, isolated down and complete real FFN intervals. Vocabulary records
checksum and real projection separately. Preserve required stream waits;
no allocation/free, copy or stream synchronization inside enqueue.

Record all primary intervals, source/executable/frozen identities, actual
dispatch, GPU conditions, payload/scratch accounting and cleanup. Do not
overlap GPU workloads, builds/tests or MemPalace with timing. Record genuine
rows 1/4 separately, not repetitions of M=1.

These are read/checksum surrogates, not inference candidates, DRAM metrics,
rigorous lower bounds, proved removable cost or E2E gains. Different
arithmetic/resources can change cache/scheduling behavior. Never feed checksum
output to an inference consumer. Existing >=5% complete-FFN, unchanged actual
512 IDs/frozen quality and repeated HTTP gates remain unchanged. The original
53.06 tok/s and 1.30x fresh Ollama goals remain required.

## Atomic Delivery

1. `docs(research): bound prepared read study`: this contract before code.
2. `feat(bench): measure prepared weight reads`: original-mapping CUDA
   checksum and shared benchmark integration with behavioral tests; depends
   on the contract. No inference configuration or prepared-control rewrite.
3. `docs(perf): record prepared read costs`: clean release measurements,
   raw evidence/audit, status and next evidence-backed action; depends on
   the validated diagnostic. Do not promote from surrogate differences.
