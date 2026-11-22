# Qwen quality beyond 64 historical blocks

**PASS on the four frozen answers for full mixed history and top-64.
FAIL for top-8.** All four dense references passed. At 3,203 prompt
tokens, top-8 changed the relay Boolean to false while keeping the port
correct. Each arm returned the correct retrieval answer at both prompt
lengths and the correct code result at 3,200 tokens. The prompt set
contains three distinct tasks, including one repeated at two lengths.

The 2,051-token control gave byte-identical full-mixed and top-64 logits
at all three frozen positions. The three longer prompts exceeded 64
old 32-token blocks. Comparing each budget with full mixed history across
all sampled positions, maximum KL was 0.00010917 for top-64 and 0.32391938
for top-8. Minimum top-ten overlap was eight and five respectively.
This quality run supplies no whole-request speed qualification.

### Storage precision and historical selection

Each comparison below covers the same 12 positions on the dense-generated
trajectories. KL uses the reference distribution first. Top-ten overlap
counts shared vocabulary IDs. Each column gives its own worst observation.

| Reference | Candidate | Maximum KL | Minimum top-ten overlap | Greedy mismatches |
|---|---|---:|---:|---:|
| Dense FP16 | Full mixed history | 0.009902 | 8 | 0/12 |
| Dense FP16 | Top-64 | 0.010082 | 7 | 0/12 |
| Dense FP16 | Top-8 | 0.207628 | 6 | 0/12 |
| Full mixed history | Top-64 | 0.000109 | 8 | 0/12 |
| Full mixed history | Top-8 | 0.323919 | 5 | 0/12 |

Dense versus full mixed retains every position and changes the storage
and attention path. Its largest KL occurred on the first output of the
3,200-token code case. Restricting full mixed history to top-8 produced
its largest KL on the first output of the relay task. All arms still
selected the same greedy token there. In free generation the relay answer
first diverged at output index 13, between the frozen observations at 8
and 16. Requiring greedy agreement at sampled positions alone would have
missed the answer regression.

Recompute the table from the retained report:

```bash
gzip -dc report.jsonl.gz | jq -s '
  map(select(.kind=="comparison")) | group_by([.reference,.candidate])[] |
  [.[0].reference, .[0].candidate, length,
   (map(.logits.kl_reference_candidate)|max), (map(.logits.top10_overlap)|min),
   (map(select(.logits.reference.top_indices[0]!=.logits.candidate.top_indices[0]))|length)]'
```

The real M40 run at f1bb7a4201ca1300af08282215d4bbb95620e8a9 completed
16 free generations, 16 dense-trajectory replays and 60 paired comparisons.
All 48 vocabulary readbacks were saved. The tracked device allocation
counter started and ended at zero. The post-run snapshot reported no
compute process and zero MiB in use. The independent replay verified
each of the 48 FP32 readbacks and reconstructed the 60 paired
comparisons again from the extracted archive.

Verify the [package checksums](package.sha256), then extract the
[complete measurement archive](measurements.tar.gz). From the
extracted directory, run `sha256sum -c RESULT/raw.sha256`, substituting
the original result directory for `RESULT`. The
[independent reader](https://github.com/GuyPaddock/m40-llm/blob/d20018ebf9e439a652675b0c04de148dcd782cbc/docs/experiments/2026-11-21-kv-audit/history.py)
reads `report.jsonl inputs.json` from the extracted directory and
reproduces [the complete replay](replay.json). Use the reader's
siblings from the same commit when running it locally.

[Raw report](report.jsonl.gz) · [Build and device identity](preflight.txt) ·
[Post-run snapshot](after.txt) · [Frozen protocol](../README.md)
