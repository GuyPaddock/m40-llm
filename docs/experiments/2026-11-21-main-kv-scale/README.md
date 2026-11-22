# Q8 KV scale synchronization on main

Pinned base e1403e0dbe2037cd8fc2b0af32addf97a4df3488. Both head64
Q8-history conversion kernels read a common K reduction, then overwrite
its shared memory to reduce V. Each warp needs to finish reading the K
maximum first. The two barriers from e930d2109f7940804c2aa17f8f5fede83ae18fe1
apply cleanly in isolation. Test whether the original regression also runs
on main using its existing Rust cache and CUDA interfaces.

Use the same 129-token fixture, two KV heads, four query heads, four
sequences and four reset lifetimes. Compare staged reconstructed values,
direct attention, guard bytes and final allocation count. Reuse the test
and guard helper verbatim if the base APIs allow. Compare the original
and repaired main-based release test binaries on the same M40. Execute the
repair with cuBLAS on and off. Retain each binary hash, build configuration,
CUDA target, device UUID and actual result. A failing negative control
supports the causal comparison. A passing negative control is inconclusive
for reproduction on main. Require every fixed run to pass the original
assertions. This is a correctness test with synthetic KV inputs. It has no
GGUF, token generation rate or performance promotion gate.

The evictions exercise incremental conversion. The bulk conversion uses
the same shared-memory ordering and should also run the existing
dense-to-compressed attention parity grid. Keep that coverage distinct
from the incremental fixture. Stop on mismatched ownership, arithmetic or
guards. No runtime switch is required.

## Source and build audit

The original test and its guard helper both compile unmodified against
main at a898ac2. The fixed runtime at 75ed22f changes two CUDA statements
and their comments. The bulk builder requires a dense cache with a Q8
sidecar; on main only the compressed constructor allocates that sidecar.
The existing parity grid includes a bulk dense-to-compressed check with
dense old backing. It cannot qualify the separate bulk Q8 builder.

Compile the unrepaired test at a898ac2 with
`M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 cargo test --locked --release --features cuda --test q8_kv_scale_reuse --no-run --message-format=json`.
Copy its Cargo compiler-artifact executable as unfixed-on. At 75ed22f use
the same command with `--test attention_parity_cuda_grid` added, once with
cuBLAS enabled and once disabled. Save both executables for each mode.
See [binary hashes](binaries.sha256) and [ELF/build provenance](build-targets.txt).
Run [the serial driver](run.sh) under the exclusive lock with the directory
of those five binaries and a fresh output directory. Inspect every log
for skips and record the exact failed assertion for the negative control.
