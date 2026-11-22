# Dense TinyLlama serving across revisions

On the same TinyLlama F16 checkpoint, 46 prompt IDs and 128 matching
greedy output IDs, main delivered 25.993 buffered HTTP tokens/s and the
Phase 4 build delivered 20.944. Five alternating pairs put Phase 4
19.423% below main, with a paired bootstrap interval of −21.609% to
−18.090%. Median request time rose from 4.924 to 6.111 seconds. Both
processes reached 6,200 MiB in the available device-memory samples.
This compares complete revisions on this workload. It does not isolate
the cause of the regression.

The separate cold CLI observations were 11.84 and 13.00 seconds.
Time to the first nonterminal streaming record was 1.334 and 1.726
seconds respectively. The older main endpoint omitted per-token timing.
The replay derives its output count from the companion token IDs.
Both the CLI and HTTP trace returned matching IDs within each revision.

Across the ten measured HTTP requests the archive contains five host
samples on main and seven on Phase 4. Each observed a stopped emerge,
a sleeping emerge and sleeping make processes. The peak sampled
ten-second CPU pressure averages were 0.62% and 2.10% respectively.
The intervals between observations or request boundaries reached 5.07 s.
The Windows VM was running according to the operator; per-VM counters
became available after this cohort. These observations cannot apportion
the cross-revision difference between software and host activity.
Replay from this report directory with
`python3 ../../2026-11-21-protocol/request-host-load.py EXTRACTED_DIRECTORY pair-{0..4}-{0,1}`
in Bash.

[Independent replay](replay.json) · [Raw requests, server logs, clocks and process samples](measurements.tar.gz).
Extract the archive into a fresh directory and run the parent directory's
replay.sh with that directory as its argument.
