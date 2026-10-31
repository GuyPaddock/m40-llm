# Lossless FP16 consumer and comparison protocol

The CUDA decoder and matched attention consumers are implemented in the
worktree. CUDA/server all-target Clippy and the ignored hardware test's
compile/link check pass. The 225-test non-CUDA suite passes. Hardware numerical,
ownership and timing checks remain pending while the original capture runs.
This checkpoint does not qualify the CUDA implementation.

## Storage evidence and admission

The ongoing capture at `82a118a` has completed three of eight sequences at
this checkpoint. Its partial report has 36 streams, 144 compression records,
nine layer readbacks and three sequence completions. It has no completion
record. The full 96-stream, 384-compression-record grid still requires audit.
Do not restart this capture or run another GPU process alongside it.

One completed real prefix already establishes that storage savings are
possible with the fixed FP16 format. Sequence zero, layer zero, key head zero
at 1K tokens has 262,144 original bytes and 254,536 encoded bytes, including
2,056 metadata bytes. It saves 7,608 bytes and reconstructs exactly. The raw
prefix hash is `fa1359f6dcd601c8713203bc5468d607066c5ae2e9009a7c3521f26f4ec8c8e8`;
an independent read of the binary's first 262,144 bytes matches it. This is
partial compression evidence, not a complete-cache or attention result.
The [retained partial record](2026-10-31-lossless-kv-partial-evidence.json)
contains the source identities and observed counts.
There is no three-percent storage cutoff: any positive net case admits the
conditional comparison. Full-grid reporting must retain expanding cases too.

## Consumer implementation

`cuda/lossless_kv.cuh` reads the CPU-validated serialized format directly.
Byte loads support unaligned packed payloads and avoid reading beyond the
last field. Raw blocks retain their original words. Separate reconstruction
writes a dense FP16 buffer; fused attention reconstructs values in registers.
Each device allocation owns one sequence/head stream. A device pointer
directory orders streams by sequence and KV head; its bytes count as metadata.

The two attention variants share a warp dot-reduction tree, FP32 online
softmax and partition merge. Both consume every token. K/V use head-major
streams; queries and outputs use sequence, query head and channel order.
The bounded implementation supports dimensions 64/128, up to eight distinct
sequences and GQA. The real comparison uses the model's original head128.
Use eight contiguous context partitions for the real grid. No geometry search
or codec-format change follows from this initial experiment.

Partial storage contains a maximum, exponential sum and unnormalized output
vector per partition. Its size depends on head count and partitions, not
context length. The fused path allocates no reconstructed full KV buffer.
The owner finishes decode before freeing encoded payloads or their directory;
callers must retain query, output and partial buffers through completion.
Nothing is connected to runtime inference dispatch or default KV selection.

The hardware tests cover all 65,536 FP16 bit patterns, packed widths zero
through four, raw fallback, short tails, multiple streams and owner destruction
after enqueue. Attention tests cover dimensions 64/128, batches 1/2/4/8,
lengths 1/5/19/257/1024, two allocation lifetimes, distinct sequence/head data,
GQA, guarded buffers and an explicit prefill-to-decode query dependency.
They require identical dense/staged/fused output bits and absolute FP64 output
error at most `1e-3`. Invalid shapes and writable aliases must fail before
changing outputs. Finite inputs govern attention; exceptional FP16 patterns
are covered by the bit-exact decoder test.

## Frozen complete-operation comparison

After the full capture audit and hardware qualification, use layers 0/17/35,
contexts 1K/2K/4K/8K and batches 1/2/4/8. Use the distinct captured sequences
and explicitly labeled deterministic synthetic queries. Retain source, model,
binary, executable and input hashes. A partial capture is not a timing input.

Compare these complete full-context operations with the same resident data:

1. Dense FP16 attention using the matched partition implementation.
2. Compressed KV with separate K/V reconstruction followed by dense attention.
3. Compressed KV consumed inside attention, including partition merge.
4. Existing native dense head128 attention with eight partitions and 256
   threads, as a runtime cost reference. Serial row dispatch is recorded
   separately from the prototype's batched dispatch.

Require bit-exact reconstruction of every source K/V word before timing.
Require FP64 output parity for every arm and bit agreement for the three
matched prototype arms. Native attention has a different reduction order;
use the existing absolute `1e-3` numerical gate. If native cache setup uses
zero-frequency RoPE append, report signed-zero effects of that API separately;
they do not weaken the lossless reconstruction contract.

All timed arms include their actual reconstruction, attention and merge
launches. Reuse persistent allocations and CUDA events. Check timed outputs
outside the measured interval, report CUDA-event and completed host wall
times, and retain allocation/copy counters. Use the established five-pair,
variation-extension and fresh-confirmation protocol. A candidate must beat
both the matched dense control and the native runtime reference before any
integration consideration; beating a newly written control alone is insufficient.
Batch dispatch differences and isolated reconstruction costs remain separate.

Record CPU encoding/validation, uploads, setup allocations and initial device
residency outside attention timing. Also measure the available append path:
extend each retained prefix from length minus one to length using its actual
last captured token, re-encode with the existing CPU codec, and upload the
replacement representation. Include full re-encoding, directory and transfer
costs; this implementation has no incremental GPU compressor. Compare that
cost with a dense one-token append, and report append-plus-attention latency.
Do not assume initial compression is amortized over an unspecified decode run.
An incremental append design remains a prerequisite if replacement cost is
unprofitable; this study does not silently substitute such an implementation.

Report encoded payload, all directory/header bytes, raw fallback, staging,
partition workspace, dense references, upload temporaries and retained source
buffers separately. Distinguish logical bytes and sector estimates from
unavailable DRAM counters. Record final tracked cleanup and external driver
residency. These operator fixtures generate zero model tokens and provide no
task-quality or whole-model throughput result.

## Validation and next steps

```bash
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/lossless-kv/target
M40LLM_ENABLE_CUBLAS=1 cargo clippy --features cuda,server --all-targets -- -D warnings
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda --test lossless_kv_cuda --no-run --locked
RUSTFLAGS="-D warnings" cargo test --no-default-features --locked
# Only after capture completes and M40 compute-process inspection shows idle:
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda --test lossless_kv_cuda \
  -- --include-ignored --test-threads=1
```

Run the affected attention, forward, projection/MLP and transfer/ownership
checks serially, including cuBLAS enabled/disabled parity. Preserve failures
before removing any experiment. The CUDA consumer and tests remain uncommitted
pending those hardware checks. The real-input paired runner, append pricing,
full capture audit and decision report are still required.

The [checksummed validation logs](2026-10-31-lossless-kv-consumer.sha256)
retain the initial missing CUDA constants include, a
missing stream diagnostic label and one Clippy style failure, followed by
successful compile-only and CPU checks.
