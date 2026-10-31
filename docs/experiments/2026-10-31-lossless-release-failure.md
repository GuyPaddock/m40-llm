# Lossless attention release admission failure

The clean release build at `464ec29` fails the first real-input attention
validation, layer 0 at 1,024 tokens. The runner reports `matched prototype
output bits differ` and exits before timing. This result does not establish
which prototype arm differs; the original diagnostic omitted that detail.
The exact-output gate remains unchanged.

The executable SHA-256 is
`63106286697ae32a5770f44ddb7e8e9aa6d008c3548c04122c17b461ff5274a9`.
The case input SHA-256 is
`d13bc9f6112a53dcfec3e702265190f5789f70657c745a7caf4ed1ef0b166d15`.
The capture manifest, complete input hashes and device identity are retained
in the [three raw records](2026-10-31-lossless-release-failure.jsonl.gz).
There is one manifest, one setup record and one validation failure. There is
no completion record, no timing pair and no generated model token.

The [executed script](2026-10-31-lossless-release-failure.sh) uses the M40,
CUDA and cuBLAS, clears inherited dispatch overrides and requires a clean
worktree. The [build log](2026-10-31-lossless-release-failure-build.log.gz)
records a successful release build; the
[run log](2026-10-31-lossless-release-failure.log) records the failure.
The script exits before its final idle query because admission fails.

The earlier synthetic hardware checks and capture bit reconstruction remain
valid within their recorded scope. They did not cover this real-input
output difference. Preserve consumer `e9343b4` and runner `464ec29`, add
arm/batch/word diagnostics, and identify the cause before changing arithmetic
or admitting timings. Any correction is a separate commit because the failed
experiment must remain reproducible and history rewriting is unauthorized.
