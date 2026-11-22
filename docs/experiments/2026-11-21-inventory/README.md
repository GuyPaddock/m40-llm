# October inventory, November 21 inspection

The 13 local research tips matched the live remote heads. The common experimental base is Phase 4 at `48c63332462e009964e891be1cabaf139e3b50f8`. Remote main is `e1403e0dbe2037cd8fc2b0af32addf97a4df3488`, reachable from Phase 4 with 470 additional commits. Local main was five commits behind. All thirteen temporary historical worktrees were registered as prunable. Their reports and source trees remain accessible through Git. The November consolidation branch is a new worktree.

[Branches](branches.tsv) pins each merge base, tip, report path and SHA-256. [Changes](paths.tsv) contains every path changed across each topic's merge base and tip, including tests, build files and raw evidence. Columns give the Git blob on the topic, Phase 4 and main, or `absent`. Different whole-file hashes can include unrelated edits. [Implementation commits](implementation-commits.tsv) lists topic-local changes touching paths outside docs in chronological order. [Original coverage](october-evidence.json) preserves the October audit's fixture counts, comparisons, limitations and available raw hashes. Those numbers are historical measurements. No November speed qualification has been run at this checkpoint.

| Topic | Evidence and dispatch disposition | Dependencies and port decision |
|---|---|---|
| Channel selection | 216 supplied-index cases and 810 numerical checks. Measured predictor cost 508–514 µs against 57–151 µs budgets. Close predictor. Explicit selected-Q4 API on Phase 4 only. | Original host packing and CUDA consumer depend on Q4_0 blocks, device context, immutable indices and scratch. Host component 3df1dd2 precedes device component 9a68003. Preserve for C4 controls. Dynamic production FFN would additionally need session-specific index updates and a single down layout. |
| ShiftAdd | 36 cases beat native Q4 and lost to Q8. Payload saving 4.17% with larger error. No runtime dispatch. | Encoder a6ef091 and sign-add kernel 07e7fd7 require new representation and fixtures. Close both. |
| Gate/up fusion | Existing fused single-row path. Additional multi-row gain at most 0.81% against fused control. | Q4 fixture conversion plus fa09568 CUDA prototype and benchmark. Close port. |
| Activation preparation | Saves 5.24–9.88 µs for fixed consumer. No matching model rotation. | 7ac84f2 requires rotated activations and a matching consumer. Re-evaluate only with compatible weights. |
| Legacy graphs | Four full-forward cases failed exact FP16 cache comparison. | a45cfcf is a negative-control test. Reuse repaired follow-up for B1. |
| Lossless KV | 4.57% storage saving at 8K, attention 7.07–8.51 times slower and append failed. | FP16 block encoder, capture protocol, CUDA decoder e9343b4 and rounding correction fe213e1 are a coupled experiment. Close port. |
| Quantized KV | 144 matched comparisons, 3.53–20.40% operator gain vs staging. No corresponding HTTP gain established. | Main already has the mixed direct runtime. Keep diagnostic staging FFI and examples/kv_attention_bench on research branch. Carry barrier fix separately, advance consumer economics in B2. |
| Soft-MTP | Compatibility/cost paper assessment, no measured learned acceptance or enabled runtime. | Defer pending compatible checkpoint and measured acceptance. No source port. |
| DualPace | 108 inputs, 324 modes, best optimistic simulated gain 0.072%. 18 unresolved handoff diagnostics. | CPU placement and measured transfer harness feed an offline simulator. Close CPU/GPU scheduler. |
| Graph dispatch | 93 differing cache comparisons reduced to zero, 228 exact attention comparisons. Separate opt-in branch, no qualified performance. | b5ea002 fixtures → dbceae1 shared ordinary/device-length CUDA dispatch, Rust enqueue and session preconditions → B1 lifecycle and HTTP measurements. Main already has DecodeSession::forward_with_optional_graph and graph FFI. Compare its kernel geometry and parameter routing before adapting the fix. |
| KV quality | 36 generations, seven dense-passing cases preserved, five dense failures inconclusive. Top8 increased long-prefix differences. | 558203f inputs → 421df83 evaluator → B2 extended evidence. Preserve evaluator as research infrastructure. No new production consumer. |
| Selector follow-up | Zero of 5,382 comparisons passed. Research-only observation hooks and policy replays. | Explicit Q4 components above suffice as operator controls. Exclude d6f6ae9 and f2114c2 predictor machinery. |
| Adaptive drafting | Independent replay of 126 cases/630 trials. Only two copy cases exceeded 3% projected complete-request gain. | 1892432 consumes historical traces and costs. Keep offline. C2 transaction parity and a qualified C1 complete verifier are prerequisites to live D1. |

## Main versus Phase 4

Direct inspection of main's src/kv_compression.rs default and src/infer/kv.rs shows mixed direct attention and top8 already enabled. Keep the explicit dense control and compare coverage separately. No selected-Q4 device owner or direct verification module is present on main. Exporting new research APIs alone would add no inference speed.

Main's cuda/kernels.cu lines 2145 and 2206 both read reduce[0] for K and immediately reuse the same shared array for V. The syncthreads after the write cannot order earlier reads in other warps. Feature e930d21 adds precisely the two missing pre-write barriers. The previous main tip's debug-read synchronization is a distinct change. Proposed port consists of these kernel edits and a behavioral regression adapted from tests/q8_kv_scale_reuse.rs with its guarded-byte helper. Qualify active append hardware and label bulk-builder evidence separately.

Dependencies for B3/C1 run from the model's actual projection storage and activation arithmetic to full FFN, target forward, session verification transaction, and finally request-level timing. The Phase 4 portfolio uses model-local dispatch and 37 packet owners, adds 1,192,935,424 resident bytes, and rejects graph, compressed and batched serving modes. Its 52.55 vs 50.52 HTTP result cannot be applied to those modes. A clean port of any subset requires evidence against the corresponding main workload.

## Replay

For each row of branches.tsv, inspect the document with git show TIP:REPORT and pipe its bytes into sha256sum. Resolve each file in paths.tsv by git rev-parse TIP:PATH. Recompute topic commits with git log --reverse BASE..TIP -- . ':!docs'. Retrieve reports by SHA even if historical worktrees have disappeared. The blob inventory proves object identity only; retained raw runs and their stated hardware boundaries determine each measurement's scope.

## November disposition of each October topic

| October topic | Current decision and evidence |
|---|---|
| Channel selection | Close the original ternary predictor. Preserve explicit supplied-index Q4 execution and host packing for the C4 control. Live thresholding uses a different input signal and must pass its own costs and quality gates. |
| ShiftAdd | Close. Its tested layout lost against both prepared-Q8 controls. |
| Gate/up fusion | Close the additional generic multi-row sweep. Keep the existing single-row fusion. C1 shares prepared **down** weights across verifier rows. |
| Activation preparation | Defer. The retained savings require an admitted checkpoint with the matching rotation. |
| Legacy graphs | Preserve the original failed cache-state case. Use the repaired arithmetic as the comparison baseline. |
| Lossless KV | Close. The measured attention and append penalties outweigh the small storage reduction. |
| Quantized KV | Keep the direct mixed consumer already present on main. Port the independent scale barriers. The matched throughput evidence remains scoped to the operator and current serving cohorts. |
| Soft-MTP | Defer pending a compatible checkpoint and measured learned-draft acceptance. |
| DualPace | Close the independent CPU decode engine under the measured cost model. |
| Graph dispatch follow-up | The Phase 4 repair passed exact state checks. The Qwen graph HTTP comparison missed the speed gate and the extra fallback node missed the pruning gate. The main port still needs the corrected boundary and full-session checks. |
| KV quality follow-up | Reject broader top-8 use. It dropped dense-passing answers in the expanded suite. Top-64 preserved all four dense-passing boundary cases and still needs broader validation at lengths where its budget excludes history. |
| Selector follow-up | Keep all three previous-token, channel-history and sketch policies closed. |
| Adaptive drafting follow-up | Keep the general controller closed. Copy-only drafting depends on the complete target verification result. |

The [candidate matrix](../2026-11-21-integration/candidates.tsv) carries the
exact source hashes, tests and measurement boundaries. The [closure register](../2026-11-21-closed-directions/README.md)
preserves the frozen failure criteria and the specific evidence that
could reopen a closed direction.
