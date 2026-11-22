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

The real M40 run at f1bb7a4201ca1300af08282215d4bbb95620e8a9 completed
16 free generations, 16 dense-trajectory replays and 60 paired comparisons.
All 48 vocabulary readbacks were saved. The tracked device allocation
counter started and ended at zero. The post-run snapshot reported no
compute process and zero MiB in use. The complete numerical archive
and its second independent replay are pending.

[Raw report](report.jsonl.gz) · [Build and device identity](preflight.txt) ·
[Post-run snapshot](after.txt) · [Frozen protocol](../README.md)
