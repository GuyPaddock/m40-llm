# Prefill chunking build and state checks

**PASS for the two guarded CUDA fixtures in both cuBLAS builds.**
The source was `8133884f04fb5024a3223281a8a0220debe0680d`.
Each actual M40 binary used `sm_52` and reported two passes, zero
failures and zero ignored tests in the selected scope.

- A 33-position prefill advanced through 16 and 32, then completed its
  final position. An exact-width request waited in queue order, then
  reached its one first sample. Two ready decodes made progress through
  the partial prefill. Completed sampled IDs and valid dense FP16 KV
  agreed byte for byte with ordinary sequential execution.
- After cancelling a partially prefilled request, its slot was leased
  to a fresh one-position prompt. Its first result and complete valid KV
  matched the ordinary execution with no stale state. Both fixture runs
  checked allocation ownership and release.

Formatting, 88 non-CUDA library tests, CUDA/server Clippy and the
release server build passed. The [build and guard archive](build-and-guards.tar.gz)
retains the toolchain, source, build logs, test-binary and server hashes,
ELF architectures and exclusive-owner GPU snapshots. Its
[checksum](archive.sha256) covers the transport bytes. The captured test
logs byte-compare to the extracted archive. The
[frozen runner](../chunking-plan.md) supplies the build and hardware
commands. The HTTP experiment evaluates the scheduler's latency and
throughput separately.
