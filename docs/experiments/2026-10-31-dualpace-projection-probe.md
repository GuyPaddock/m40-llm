# Single-core Q8 projection cost probe

`cpu_projection_probe` prices one bounded CPU implementation for the
[cooperative scheduling study](2026-10-31-dualpace-audit.md). It consumes the
retained Qwen FFN inputs and original Q8_0 weights. This is an isolated projection
probe, not a complete CPU model backend or an adoption recommendation.

## Frozen scope

- Model SHA-256 `4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
- Input artifact SHA-256 `eb51828ef778189567ae01fdd910659025c92ae6f5b2d76c989c314e650d0687`.
- Capture report SHA-256 `d9e55963e5872bbc48cfb3a1f7ac217429974d0a48237aabaaab207f5fc95801`.
- Gate, up and down projections at layers 0/17/35; batches 1/2/4/8.
- The first eight captured rows per layer, checked pairwise for distinct values.
- One pinned logical CPU and one local bound NUMA node. Planned placements are
  CPU 15/node 0 and CPU 31/node 1, checked for unrelated CPU load before timing.
- One scalar sequential FP32 control and one eight-lane AVX2 FP32 consumer.
  Neither uses a materialized FP32 weight cache or changes the Q8 representation.

The loader pins the completed historical capture by exact artifact/report
hashes, validates its schema and model identity, checks GGUF tensor name, shape,
dtype and byte offset, then hashes each weight payload. It reads one matrix at
a time instead of loading all model weights. Hashing and file I/O are outside
timed intervals. This does not admit new training or evaluation data.

## Numerical and timing contracts

Both consumers decode signed Q8 codes and FP16 block scales into FP32 values.
AVX2 sign-extends eight codes, multiplies by the shared scale and activation,
then accumulates eight FP32 lanes. It uses separate multiply/add operations,
with no requested FMA contraction. The scalar loop accumulates sequentially;
the AVX2 reduction has a different order and is not claimed bit-identical.

An independent FP64 dot product validates every output before timing. The
fixed tolerance is `1e-3 + 2e-5 * abs(reference)`, matching the retained
projection diagnostic. Reports retain maximum absolute error, error/tolerance
ratio and mean squared error. This check does not establish model logit or
long-generation quality.

Each arm warms once. Before every measured operation, an untimed pass reads
one word per cache line in a first-touched 128 MiB buffer. This exceeds either
socket's 40 MiB shared L3, reducing the risk of pricing a whole model from
repeated cache-resident 24 MiB matrix reads. It is an eviction attempt, not a
measured DRAM-transaction guarantee. Both arms pay the same cache policy.

The default is three operations per trial. Wall timing includes one complete
projection, shape checks and input-finiteness checks; output allocation,
eviction and reference validation are excluded. Eviction wall time is reported
separately. Output is reused and consumed outside the timer. Logical matrix
bytes do not establish actual DRAM traffic, especially when several input rows
reuse one output row's weights.

Five alternating scalar/AVX2 pairs reuse `compare_ollama/protocol.rs`. Variation
may extend the same series to ten pairs; a borderline result receives one fresh
twenty-pair confirmation. Unstable cases remain unresolved. Rates mean complete
projections per second. The HTTP target field is removed, and the threshold is
labeled an operator speed gate. No full-model speedup follows from passing it.

The manifest records executable/source identities, CPU affinity, NUMA binding,
host inventory, fixture setup and dispatch. Per-matrix setup records include
payloads and `/proc/self/numa_maps`. Memory binding and single-CPU affinity are
required, not inferred from a command-line label. Worker use is exactly one
thread. Multi-core scaling and bandwidth contention remain unmeasured.

## Commands and current status

```bash
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/dualpace/target
cargo build --release --no-default-features --locked \
  --example cpu_projection_probe -j 4

numactl --physcpubind=15 --membind=0 \
  target/release/examples/cpu_projection_probe \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --ffn-inputs /mnt/array-fastest/home/guyep/playground/2025-12-20-m40-llm/m40-llm/target/research-fixtures/m02-ffn-inputs.json \
  --capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --cpu 15 --numa-node 0 --iterations 3 \
  --output target/dualpace-cpu-node0-new.jsonl
```

Use `--validate-only` for the complete 36-case numerical grid without recording
timings. Repeat the measured command with CPU 31/node 1 and a fresh output.
Real-fixture runs require a clean worktree; timing additionally requires release.
Keep timed hardware experiments serial with the active KV capture and later
GPU transfer measurements. Do not promote a scalar comparison to a comparison
with an optimized multi-core CPU decoder.

Five new behavioral tests pass, including the explicitly invoked AVX2 test;
seven imported statistical-protocol tests also pass. Cases cover signed-code
endpoints, signed and zero scales, distinct batch rows, repeatability, output
guards, malformed payloads, invalid-call output preservation and numerical
admission. Non-CUDA
all-target Clippy passes after removing a duplicate conditional allow attribute;
the initial failed lint log is retained. [Actual Qwen validation](2026-10-31-dualpace-cpu-validation.md)
passes all 36 cases and 1,082,880 outputs per implementation. Timed results are
pending. No GPU code or production dispatch changed.

The release AVX2 function's disassembly contains `vpmovsxbd`, `vcvtdq2ps`,
`vmulps` and `vaddps`, with no `vfmadd` or ZMM operand in that function. This
confirms the generated implementation, not dynamic instruction counts. The
disassembly command uses the function's address and size from `nm -S -C`.
For this build those are `0x83450` and `0x3c1`; executable SHA-256 is
`07f47ffd5009beeb58ed67c91e2a2ade558902dc07334ff3982f10093f88459d`.
The [checksum manifest](2026-10-31-dualpace-projection-probe.sha256) covers the
test, Clippy, build and disassembly artifacts, including the initial lint error.
