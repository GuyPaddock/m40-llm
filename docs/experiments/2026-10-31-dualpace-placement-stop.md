# Cooperative cost run stops at unavailable CPU placement

The first serial pricing run completes CPU 15/node 0, then exits before
launching the node-1 probe. `numactl` rejects CPU 31. The execution environment
allows CPUs `0-30,32-62`, excluding CPU 31 and its sibling 63; both memory
nodes remain allowed. This is a launch restriction, not a numerical failure.
No node-1 projection or transfer result was produced, and the subsequent
transfer phase did not run.

Node 0 retains all 36 projection cases and 180 alternating pairs at clean
revision `8171d7f`. All numerical checks pass and every final timing series
is stable. No extension or confirmation is needed. Scalar maximum absolute
error is `0.000996087424596226`; AVX2 maximum error is
`0.00014867753466774047`, both within the fixed relative/absolute gate.
The pre-run CPU 15/sibling 47 samples are 100% idle. These completed results
remain valid and will not be rerun just to replace the failed later launch.

The [CPU summary](2026-10-31-dualpace-cpu-node0-summary.json),
[audit program](2026-10-31-dualpace-cpu-timing-summary.jq) and
[raw observations](2026-10-31-dualpace-cpu-node0.jsonl.gz) preserve that result.
The [executed script](2026-10-31-dualpace-pricing-first-attempt.sh), failed
launch log and [affinity evidence](2026-10-31-dualpace-placement-restriction.txt)
remain in the [checksum manifest](2026-10-31-dualpace-placement-stop.sha256).

The next protocol amendment will use allowed CPU 30 on the same node 1,
with sibling 62 checked for unrelated load. The
[replacement load sample](2026-10-31-dualpace-replacement-load.txt) is retained.
This changes the physical core label before any node-1 timing; it preserves
one-thread execution, node-local memory, the source inputs, pair policy and
all numerical gates. Scenario admission must require the replacement CPU
consistently in both its projection and transfer evidence. No cpuset boundary
will be widened and no result will be labeled as CPU 31.
