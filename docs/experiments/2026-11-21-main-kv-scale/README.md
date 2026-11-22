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
