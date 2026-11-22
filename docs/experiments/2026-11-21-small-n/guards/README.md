# Q8 few-row CUDA guards in both builds

**PASS on the M40.** The three selected tests passed with cuBLAS
enabled and disabled, with no skipped invocations. The shared-weight
operator matched serial FP32 bits for the fixture's mixed row groups,
partial columns and zero-scale data. The tests checked input and output
guards, partial scale/code regions, and rejected invalid buffers.

The arithmetic-order control compared partial width 33 at K=2,048
and full width 2,048 at K=11,008. The 256-thread path was exact on both.
The 128-thread path differed in 32 and 1,894 outputs, with maximum
absolute differences of 0.00000191 and 0.00002098. The same observations
occurred in both builds. These controls support the arithmetic-matched
serial and shared comparison for C1. Full target speed and live drafting
are separate qualifications.

The [raw archive](measurements.tar.gz) includes the two release test
hashes, link libraries, both `sm_52` images, test counts, NVIDIA
driver and snapshots before and after each invocation. Verify with
`sha256sum -c archive.sha256`. To repeat, run the original
[guard driver](../guards.sh) with those binaries under the shared
M40 lock. The driver in the recorded revision is cfb0c58.
