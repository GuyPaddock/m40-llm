# Maxwell Counter Preflight

This optional tool probes the legacy CUPTI Event/Metric APIs without adding a
CUPTI dependency to Cargo, `build.rs`, or inference. It enumerates the metric
catalog and tests event-domain enumeration and empty event-group creation in
a fresh CUDA context. It does not launch model kernels or collect counters.
A metric name in the catalog is not evidence that its counter is readable.

## Run

Use an idle M40 with hardware access. Download/extract a CUDA 12.x CUPTI SDK
separately if it is absent from the toolkit; do not replace the system driver
or toolkit for this check. The 2026-10-24 probe used NVIDIA's
[CUPTI 12.8.90 archive](https://developer.download.nvidia.com/compute/cuda/redist/cuda_cupti/linux-x86_64/cuda_cupti-linux-x86_64-12.8.90-archive.tar.xz),
listed in the [CUDA 12.8.1 manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json).
Archive SHA-256:
`7bf5db86cb82f26a6a3cb9e2fa4dc2a131d25885f59fdbc647938929924405db`.
The extracted SDK's license is retained alongside it; no NVIDIA sample code
is copied into this project.

```bash
source scripts/dev-env.sh
export CUPTI_ROOT=/path/to/cuda_cupti-linux-x86_64-12.8.90-archive
export CUDA_HOME=/opt/cuda
bash scripts/probe_maxwell_counters.sh 0 > /tmp/m40-counters-user.jsonl

# If collection is unavailable, distinguish permissions from support.
sudo -n env CUPTI_ROOT="$CUPTI_ROOT" CUDA_HOME="$CUDA_HOME" \
  bash scripts/probe_maxwell_counters.sh 0 > /tmp/m40-counters-root.jsonl
```

The script compiles the standalone C++ preflight with warnings as errors into
a temporary directory and removes the binary afterward. It requires Linux,
`g++` or `CXX`, CUDA headers, and `CUPTI_ROOT/include` and `CUPTI_ROOT/lib`.
No download, installation, inference, or system-path change occurs.

Exit statuses:

- `0`: empty event-group setup succeeded, but counters are **not measured**.
  Targeted kernel measurement/replay must still be implemented and verified
  on that stack before any traffic or occupancy claim.
- `2`: event-group setup is unavailable, or the requested SDK is missing.
  Runtime unavailability emits JSON; missing build dependencies report stderr.
- `1`: invalid input or an unexpected CUDA/compile failure. Invalid device
  arguments produce an error record, not a successful skip.

All metric values and `kernel_measurements` remain `null`. A future collector
must retain the original availability record, report its measured kernels and
replay passes, and run separately from ordinary E2E trials. Do not substitute
model-file size, an occupancy estimate, or a catalog lookup for measured data.

## 2026-10-24 Result

Tesla M40 24GB, `sm_52`, driver 580.159.04, CUDA toolkit 12.8.93, CUPTI
12.8.90/API 26. The driver reports CUDA Driver API version 13000; that does not
mean these tests used a CUDA 13 toolkit. No other GPU compute process was live.

| Operation | User UID 1000 | Root UID 0 |
| --- | --- | --- |
| Metric catalog | 154 entries | 154 entries |
| All six target metric lookups | Success | Success |
| Event-domain enumeration | `27 / CUPTI_ERROR_NOT_SUPPORTED` | Same |
| Event-group creation | `27 / CUPTI_ERROR_NOT_SUPPORTED` | Same |

Targets were `dram_read_bytes`, `dram_write_bytes`, `achieved_occupancy`,
`sm_efficiency`, `gld_efficiency`, and `gst_efficiency`. None was collected.
An unchanged NVIDIA SDK `cupti_query` sample independently enumerated metrics
but failed domain enumeration with the same error. The project probe also
checked actual event-group creation, rather than treating that query failure
alone as proof of collection unavailability.

[Raw project-probe records](experiments/2026-10-24-maxwell-counter-preflight.jsonl)
include both privilege levels, catalog entries, lookup statuses, and explicit
null measurements. Changing profiling privileges did not recover this legacy
path. The precise CUPTI/driver compatibility cause is not isolated; do not
generalize the failure to every Maxwell/toolkit/driver combination.

NVIDIA documents Maxwell support for legacy Event/Metric APIs, deprecation
starting in CUDA 12.8, and Volta-or-newer restrictions for the newer profiling
APIs. Those newer APIs are not an M40 fallback. See the
[CUPTI release notes](https://docs.nvidia.com/cupti/12.9/release-notes/release-notes.html#gpu-support).
Counter unavailability does not block the research plan's C06 checkpoint:
use the buffered CUDA-event profile, and label traffic and occupancy unknown.
