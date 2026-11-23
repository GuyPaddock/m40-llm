# Expanded real-model packed-Q4 quality

**PASS for the frozen 48-case quality gate.** The thresholded packed
arm preserved all 26 dense-passing answers on Qwen2.5-3B Q4_0/Q6_K.
It used the same packed down projections and cutoffs at layers 0,
17 and 35. The quality gate authorizes a separate request-cost study.

| Family, four prompts per length | Short dense passes preserved | Long dense passes preserved |
|---|---:|---:|
| Factual retrieval | 3/3 | 1/1 |
| Multi-fact recall | 4/4 | 4/4 |
| Chat and instruction following | 3/3 | 1/1 |
| Code and configuration | 3/3 | 0 dense passes |
| Arithmetic | 2/2 | 0 dense passes |
| Longer-form generation | 2/2 | 3/3 |
| Total | 17/17 | 9/9 |

There were seven short and 15 long dense failures. The candidate
answered one of those cases correctly. It returned the required
`08:45`, while dense reached the output cap while describing the
calculation. Report that case separately from answer preservation.
Long-context code and arithmetic remain inconclusive against this
reference.

Thirty-five of the 48 generated ID sequences matched exactly. Six
dense-passing answers changed. Manual review found equivalent values
in the JSON case. The five longer-form changes retained the requested
actions, length and format for the notebook, trail and meeting tasks.
Both notebook responses at short context advised inspecting the item
while asking readers to protect privacy. No candidate introduced
repetition under the frozen detector.

The independent replay recomputed 122 aligned FP32 comparisons from
244 full vectors and verified the observed tasks, order, generated-ID
divergence and packed execution. Maximum logit error was 6.465, maximum
mean absolute error was 0.755, maximum KL was 0.8643, and the smallest
top-10 overlap was seven. The highest KL came from a failed long-code
reference at offset zero. These diagnostics use native prefixes and
are separate from free-generation scores.

The clean M40 build was `aef2d6278a44d63069636d1eca3749583d8c2868`.
The `sm_52` binary SHA-256 was
`6b41d77cd0cad64b8bc188b78ed7066cb1cf9998a7fa2cd253b7e4ae09586003`.
The candidate used 38,043,648 extra device weight bytes, 350,212 bytes
of selection scratch, and a 12,681,216-byte host repack payload. Its
preparation took 6.498 s. Each selected layer recorded 103,894 calls.
Sampled peak was 4,172 MiB with two owners resident and tracked device
bytes returned to zero. The sequential quality run took 7 h 53 min.
Its generation intervals are quality diagnostics.

The [archive](measurements.tar.gz), [archive checksum](archive.sha256),
[replay result](replay.json) and [vector checksums](vectors.sha256)
retain the original prompt manifest, all scores and generated IDs,
logit summaries, preflight, host snapshots and final disposition.
The 244 FP32 files total 148,289,536 payload bytes. They are retained
in `research-1121/raw/sparse-q4-extended-01/packed-thresholded.logits/`
outside Cargo cleanup, next to the original JSONL. From that raw run
directory, apply `sha256sum -c PATH_TO/vectors.sha256`. Then follow
the [frozen protocol and replay command](../extended.md). With an
extracted measurement archive, place the verified vector directory
next to `packed-thresholded.jsonl` before replay.
