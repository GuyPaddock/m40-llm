# Initial sparse-Q4 quality on three Qwen layers

**PASS for the packed thresholded arm's initial gate. FAIL for the
zigzag thresholded arm.** Both preserved four dense-passing answers.
Zigzag thresholding raised the first held-out passage's perplexity from
32.9521 to 34.3848, or 4.348%, and reached the frozen 3% stop. Keep
both unmasked arms as layout controls. Only the thresholded packed arm
advances to the expanded short- and long-context quality gate.

| Arm | Dense-passing answers preserved | Identical generated traces | Passage perplexity ratios | Initial disposition |
|---|---:|---:|---|---|
| Packed full | 4/4 | 6/6 | 0.999963, 1.000041 | PASS |
| Packed thresholded | 4/4 | 5/6 | 1.003002, 0.996314 | PASS |
| Zigzag full | 4/4 | 6/6 | 1.020175, 0.994901 | PASS |
| Zigzag thresholded | 4/4 | 3/6 | 1.043478, stopped | FAIL |

The unchanged dense Q4_0 reference failed the arithmetic and long-form
cases. Packed thresholding changed the arithmetic continuation. The
zigzag thresholded arm changed both dense-failing traces and the chat
trace. Its changed chat output met the frozen rubric. The largest aligned
logit error across each arm's executed passages was 0.005637, 1.038102,
2.117901 and 4.256675 in table order. This gate has four passing
answers from one model and contains no whole-request speed result.

Each process used the same three down matrices on layers 0, 17 and 35.
Repacked weights added 38,043,648 requested device bytes and the
threshold/select workspace added 350,212. Repacking took 6.399–8.305 s
across the four model loads and used a 12,681,216-byte host payload.
NVML sampled a 4,029 MiB peak with both dense and candidate model
owners resident for every arm. Tracking returned to zero at teardown.
The run recorded 720 calls per selected layer in the three complete
arms and 681 in the stopped arm. This two-model peak is a diagnostic.

The clean build and quality run used `6d5bb7da539363b08536f9fe84cc704a9fe3c96c`.
Both cuBLAS builds executed the `sm_52` projection fixture. Each checked
packed and zigzag output, masked and unmasked dispatch, cutoff ties,
zero activations, guard bytes and release, with one pass and zero skips.
The twelve host weight planes matched the operator inputs byte for byte.
The [plane hashes](planes.sha256) refer to the retained
`research-1121/raw/sparse-q4-quality-build-01/planes` directory. The
[archive](measurements.tar.gz) retains manifests, full generations,
likelihood rows, commands, hashes, device snapshots and both guard logs.
Its transport [checksum](archive.sha256) verifies the archive bytes.

From this worktree run the [replay](../replay-quality.sh) on the extracted
`sparse-q4-quality-01` directory, `data/research_quality/v1.json` and the
previous `research-1121/raw/sparse-q4-live-01/held-out.jsonl` capture.
It checks that the dense generations reproduce the independent capture,
then recomputes task and passage coverage, repetition, the stopping
decision, layout ownership and sampled process peaks. The [receipt](replay.json)
matched before and after archive extraction.
