# Bounded Hugging Face source feasibility

## Decision

**Do not acquire a full V3 corpus yet.** CommitPackFT Python and UltraChat
first-user prompts are promising replacements for the code and prose families.
SciRIFF is not admission-ready as packaged: document identities are absent
from the exported schema, substantive templates repeat across splits, and
many sampled prompts exceed the length ceiling. The two inspected Tulu
components do not provide a clean replacement for both remaining families.

The smallest conditional design uses three sources for four families, with
the original **256/48/96 train/calibration/held-out quotas per family**.
The [review-only V3 proposal](2026-10-30-v3-source-proposal.json) records that
design, its blockers and stop conditions. It is not an active specification,
selected manifest, license clearance or acquisition authorization. No V1/V2
requirement was changed to make these samples pass.

## Bounds and pinned sources

Tooling commit `72b756e` enforces a 32 MiB response budget and 100 recorded
requests per source directory. Parquet reads require HTTP 206 and the exact
requested range; a server ignoring ranges cannot trigger a full download.
Remote dataset loaders and source code are never executed.

The retained fetcher made 139 requests, receiving 37,474,534 bytes in total.
SciRIFF used 25,476,250 bytes, below its individual cap. These counts cover
the recorded fetcher, not preliminary metadata web browsing. No weights were
downloaded. Source data and raw responses remain local, outside Git.

| Source | Pinned HF revision | Scope |
| --- | --- | --- |
| [SciRIFF](https://huggingface.co/datasets/allenai/SciRIFF/tree/eddf83df1ab4acf1ead5b81898dd4c7e7450050e) | `eddf83df1ab4acf1ead5b81898dd4c7e7450050e` | `4096`, train/validation/test |
| [CommitPackFT](https://huggingface.co/datasets/bigcode/commitpackft/tree/fc56fe33c030c6daa414c2b112c932b8eed085e6) | `fc56fe33c030c6daa414c2b112c932b8eed085e6` | Python train prefix |
| [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/tree/8049631c405ae6576f93f445c6b8166f76f5505a) | `8049631c405ae6576f93f445c6b8166f76f5505a` | First train_sft shard and test_sft |
| [Tulu 3 mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture/tree/b14afda60f1bbebe55d5d2fa1e4df5042f97f8be) | `b14afda60f1bbebe55d5d2fa1e4df5042f97f8be` | Metadata only; no mixture rows |
| [Tulu FLAN component](https://huggingface.co/datasets/ai2-adapt-dev/flan_v2_converted/tree/20f17e144dfdfe2d20f12f140ace5b95c31577c0) | `20f17e144dfdfe2d20f12f140ace5b95c31577c0` | Bounded train sample |
| [Tulu TableGPT component](https://huggingface.co/datasets/LipengCS/Table-GPT/tree/25754c7a072dffca92e18c56f33832936f53495a) | `25754c7a072dffca92e18c56f33832936f53495a` | ListExtraction train and TableQuestion test prefixes |

The Tulu fallback condition was SciRIFF's demonstrated failure to supply
document identity and split-isolation evidence as packaged. No unrelated
Tulu component, whole mixture or alternative model was acquired.

## Natural token lengths

All 999 inspected inputs used the original Qwen2.5-3B Q8_0 tokenizer from
the locally retained artifact with SHA256
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
The existing no-CUDA tokenizer example verifies that hash and exact prompt
roundtrips. Counts include production QwenChat framing, no system message and
no BOS. These are CPU token counts, not inference or teacher generation.

- SciRIFF uses its complete `input`, without removing task instructions.
- CommitPackFT uses `subject`, a fixed filename separator and `old_contents`.
  Neither `new_contents` nor the longer commit message enters the prompt.
- UltraChat uses only `prompt`, the initial user request. Assistant replies
  and later conversation turns are not concatenated to satisfy length gates.
- FLAN uses `inputs`; TableGPT uses `prompt`. Existing few-shot demonstrations
  remain part of the natural input, but `targets`, `completion` and assistant
  columns are never exported as labels. Their presence inside few-shot inputs
  creates an additional exclusion burden, not permission to use their answers
  as teacher continuations.

| Source/split | Sample | Min | Median | P90 | Max | 1-64 | 65-447 | >447 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SciRIFF train | 126 | 148 | 722 | 1,420 | 3,091 | 0 | 16 | 110 |
| SciRIFF validation | 126 | 222 | 743 | 2,246 | 3,333 | 0 | 21 | 105 |
| SciRIFF test | 128 | 82 | 570.5 | 2,125 | 3,507 | 0 | 51 | 77 |
| CommitPackFT Python train | 128 | 63 | 245.5 | 332 | 398 | 1 | 127 | 0 |
| UltraChat train_sft | 126 | 17 | 97 | 591 | 1,270 | 37 | 72 | 17 |
| UltraChat test_sft | 126 | 12 | 88.5 | 432 | 1,200 | 44 | 70 | 12 |
| FLAN-converted train | 126 | 75 | 443 | 860 | 2,270 | 0 | 64 | 62 |
| TableGPT ListExtraction train | 64 | 130 | 302.5 | 1,379 | 1,667 | 0 | 37 | 27 |
| TableGPT TableQuestion test | 49 | 294 | 714 | 2,264 | 3,429 | 0 | 11 | 38 |

These are **nonrandom feasibility samples**, not population distributions or
quota-yield estimates. Parquet sampling takes evenly spaced rows within first,
middle and last row groups, except SciRIFF test uses eight spaced groups.
This overrepresents some task families because SciRIFF is task-ordered.
CommitPackFT uses the first 128 complete JSONL rows in a 1 MiB prefix.
TableGPT uses at most 64 complete rows in each 256 KiB prefix. No selection
was based on observed target output, acceptance or model quality.

The [primary sample summary](2026-10-30-hf-sample-summary.json) includes all
three original context buckets. [Fallback lengths](2026-10-30-hf-fallback-summary.json)
are separate. No inspected prompt was truncated, padded or rewritten to improve its
eligibility count. A future census must include rows outside the length gate
when forming duplicate and source groups.

## Identities, splits and capacity

### SciRIFF: conditional, not a drop-in source

The pinned `4096` metadata reports 70,521 train, 30,736 validation and 35,875
test rows. Other context configurations are overlapping views, not independent
capacity. Exported fields are `input`, `output`, task metadata and
`_instance_id`. The latter is `{task}:{source split}:{ordinal}`, not a DOI,
PMID or paper ID.

Source inspection is pinned separately at SciRIFF code revision
`e396d00ac844351bc4e166744c79f8fb53020c65`.
[Task construction](https://github.com/allenai/SciRIFF/blob/e396d00ac844351bc4e166744c79f8fb53020c65/sciriff/instructions/task.py)
assigns ordinals before formatting and loads upstream datasets without pinning
their revisions in that call. Replaying today's upstream loader cannot prove
the original ordinal-to-document mapping. Recovery needs original source
revision/order plus exact input reconstruction, or an authoritative mapping.
No source loaders were run here.

Group by canonical paper IDs, connected sets of all cited papers for multi-
document inputs, substantive task/template, and duplicate component. Link the
same paper across QA, extraction and classification before splitting. Quarantine
unresolved documents and cross-split components. Do not treat metadata
`single_source` as an identifier.

Only 16 sampled rows are QA: seven PubMedQA, five QASA and four Qasper. Five,
four and zero respectively fit 65-447 tokens. This does not establish train
or calibration QA capacity. The 88 total length-eligible SciRIFF rows include
non-target summarization tasks and must not be counted as 88 eligible QA/IE
groups. Task-template hints among length-eligible samples number 2/8/12 for
train/validation/test; none is a verified independent-document count.

There are 10,696 sampled shared-32-token pairs within SciRIFF, including 5,788
across official splits. Repeated task instructions are not the exempt ChatML
transport wrapper. The current isolation contract therefore forbids naive
row-wise sampling from those splits. The PubMedQA source template has four
wording variants of the same operation; changing its phrasing is not a new
independent substantive task. Neither a full quota nor the 16-group held-out
minimum is demonstrated. Do not infer a corpus-wide impossibility from this
small sample, but do not acquire the whole source assuming the problem is solved.

### CommitPackFT: strongest code lead

The card lists 56,025 Python samples within 702,062 records across languages.
The inspected pinned loader exposes a single `train` split for each language.
Its download URL uses `main`; this study bypassed the loader and read the
explicit pinned URL instead.

Native keys are commit SHA, old/new file path and comma-separated `repos`.
A stable evidence locator is revision + language + row offset + record hash;
the source identity is repository + commit + path. A commit may touch several
files, and identical commits/files can appear in forks. Form connected
repository/fork/commit/template groups before assigning new research
train/calibration/held-out partitions. Never split adjacent files from the
same commit or cloned repositories across partitions.

The sample has 111 rows whose repository-license labels are on the proposed
review allowlist. All 111 fit 65-447 tokens and span 109 distinct repository-set
strings. That is promising, but **not 109 verified independent groups**:
fork aliases, overlapping repo sets, copied files and substantive templates
still need unioning. Eighteen sampled pairs share 32 tokens. Full usable
capacity after rights, exclusions and grouping is unmeasured. A 400-row code
quota is plausible, not established by scaling the prefix success rate.

### UltraChat: strongest prose/conversation lead

The pinned card reports train_sft 207,865, test_sft 23,110, train_gen 256,032
and test_gen 28,304 rows. Use only SFT splits in the minimal design. Keep all
turns sharing `prompt_id` in one conversation group; link duplicate prompts,
source documents if embedded, and paraphrase/template families. The ID is not
proof of original document identity for requests containing quoted material.

The sample has 72 train and 70 test prompts in 65-447 tokens, with distinct
conversation IDs within each of those subsets. There is one shared-32-token
pair within sampled UltraChat and no sampled cross-split span pair. This is
not global split clearance. Full train_sft supplies proposed train/calibration
groups; test_sft remains held-out only, with cross-split groups quarantined.

The raw scale and sampled lengths make a 400-prompt prose quota plausible
without relaxing the 65-token floor. A full ID/template/exclusion census is
still necessary. Do not concatenate synthetic assistant turns to manufacture
longer contexts or classify this first-turn-only pilot as multi-turn validation.

## License and provenance review

This is a technical admission screen, not legal clearance. The
[ODC-BY terms](https://opendatacommons.org/licenses/by/1-0/index.html) distinguish
database rights from rights in individual contents. A collection label cannot
relicense papers, code, web tables, user text or third-party generated content.
The frozen Qwen target's separately recorded research-use restrictions remain
in force even if every input eventually has a commercially permissive license.

| Source | Verified labels/provenance | Admission and redistribution implications |
| --- | --- | --- |
| SciRIFF | ODC-BY collection; component table includes MIT, Apache, CC-BY, CC0, GPL, CC-BY-NC and unspecified licenses | Retain SciRIFF attribution and each component's authors/license. Verify rights to underlying paper text separately; quarantine unknown/NC/copyleft components in a permissive pilot rather than assuming blanket clearance. |
| CommitPackFT | MIT card; sampled rows include 16 AGPL-3.0 and one EPL-1.0 entry | Card's permissive description does not override row/file rights. Review original-commit LICENSE/NOTICE and file headers. Preserve notices on any redistribution; proposed allowlist labels are only a first filter. Copyleft is not the same as a noncommercial prohibition. |
| UltraChat | MIT card; ChatGPT-generated instructions and responses; H4 filtering/truecasing | Preserve both H4 and original UltraChat attribution/license. Generated prompts are still third-party content; collection MIT alone does not establish all underlying rights or compliance with the generation-time service terms. |
| Tulu 3 | ODC-BY collection; expressly mixed component terms, some noncommercial and third-party outputs | Do not admit the mixture as one license. Component identity and generation provenance must survive every conversion. |
| FLAN component | Card delegates licensing to OpenOrca's reproduced FLAN and original task sources | No unified content permission or per-document provenance established here. Each selected task and embedded demonstration needs review. |
| TableGPT | MIT card; generated/collected tables; card notes prior synthesized-data removal for licensing reasons | Table IDs are useful but not a license. Restore source-table origin/rights and account for few-shot tables, including web/Wikipedia-derived material. |

For SciRIFF, PubMedQA and QASA are grounded-QA leads because their card labels
are MIT, not because their text has been cleared. NLM warns that
[PubMed abstracts can contain copyrighted material](https://www.ncbi.nlm.nih.gov/books/NBK25497/).
Potential IE leads are materials-synthesis events, AnatEM and the NCBI/NLM
annotation components. Their annotation/code licenses do not automatically
cover the journal abstracts they annotate. No candidate component is approved
by this list, and no extra payload for these leads was acquired.

UltraChat's original project says both sides were automatically generated
and warns that benchmark overlap can still occur despite not deliberately
inserting public benchmarks. See its [source repository](https://github.com/thunlp/UltraChat).
Using only the initial prompt avoids importing its answers as supervision;
it does not erase generated-content provenance. Do not distribute raw sampled
content, teacher features, continuations or trained weights on the strength
of a dataset badge. Distribution needs a separate review of applicable input,
collection, generator and target terms.

## Conditional Tulu fallback findings

The [pinned mixture card](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture/blob/b14afda60f1bbebe55d5d2fa1e4df5042f97f8be/README.md)
names FLAN and TableGPT as individual components. Its metadata reports 939,343
rows while the prose says 939,344; neither is used as a verified group count.
The mixture itself was not sampled. Its SciRIFF component is not an independent
replacement for the same upstream source.

**FLAN-converted:** the pinned shard footer reports 89,982 rows. Fields retain
`_task_source`, `_task_name`, `_template_idx`, `_template_type`, `inputs`,
`targets` and messages, but no original example/document ID. All 126 sampled
rows are `fs_opt` few-shot records. Observed tasks include e-SNLI, GSM8K,
Amazon reviews, BoolQ and generated/programming tasks. A row's template index
is not a document key. Few-shot demonstrations can connect otherwise unrelated
records into leakage components. The 64 length-eligible rows do not establish
eligible grounded-QA or extraction counts. No drop-in recommendation follows;
there is no clean benefit over repairing provenance at the original task source.

**TableGPT:** sampled rows contain a JSON `metadata` string with dataset, table,
task, seed and few-shot count. That gives a better grouping lead than the four
fields listed in its card, but seed/task variants of one table remain one
group. Few-shot tables must join that group too. The extraction sample contains
64 distinct target table IDs, 28 few-shot and 36 zero-shot rows; 37 fit the
length gate. Target IDs alone omit the demonstrations' tables.

The natural TableQuestion component is explicitly **test-only**. Its 11
length-eligible sampled rows cannot be moved into pilot training to repair
grounded-QA quotas. The 537-row ListExtraction training component is at most
a reserve IE lead; surviving length, source rights and shared-template checks
is unmeasured. The Tulu mixture uses 5,000 TableGPT prompts in total, but exact
membership of these sampled original components in that mixture was not
verified. No mixture-membership claim or prompt rewriting is needed for this stop.

## Frozen benchmark isolation

The 760 primary inputs were compared with the retained 265,771-text benchmark
inventory from the previous R0 audit: exact normalized text, five-word-shingle
Jaccard at least 0.5 and normalized 32-token spans. There were no literal hits.
That is **not** semantic clearance. The two fallback samples have token-length
evidence only, not a completed benchmark-overlap audit.

Commit messages and code can repeat the frozen deduplication, configuration
lookup or implementation templates. Synthetic prose can paraphrase the browser
explanation and other held-out tasks. SciRIFF/templates and FLAN demonstrations
can leak task support even with different documents. TableGPT shares the broad
lookup/extraction form of existing fixtures. Freeze the existing browser,
twelve quality tasks, held-out drafter/copy tasks and KV fixtures before any
selection. Exclude their prompts, saved outputs and substantive templates
from every partition; do not fit source choice to target acceptance on them.

Also preserve the unresolved script/external-artifact and semantic coverage
gaps documented in the previous exclusion inventory. A zero-match result
does not close those gaps. Source documents and substantive templates must be
grouped across all four families, not audited separately after splitting.

## Minimal V3 recommendation and review gate

| Family | Conditional source selection | Train | Calibration | Held-out |
| --- | --- | ---: | ---: | ---: |
| Prose/conversation | UltraChat first-user SFT prompts | 256 | 48 | 96 |
| Code | CommitPackFT Python, reviewed permissive files | 256 | 48 | 96 |
| Grounded QA | SciRIFF component allowlist only after document/template repair | 256 | 48 | 96 |
| Extraction/classification | SciRIFF component allowlist only after rights/isolation repair | 256 | 48 | 96 |
| Total | Three-source design, not an admitted corpus | 1,024 | 192 | 384 |

Recommend retaining natural **65-447 tokens** for these new sources, no padding
or truncation, 128 generated tokens, 16 held-out independent groups per family,
and 32 held-out prompts in each existing global context bucket. The accepted
MBPP short-context amendment remains immutable but does not automatically
extend short-prompt eligibility to replacement datasets. Report 1-64-token
samples separately; any later short-prompt extension needs explicit review
and the accepted cost-coverage requirements before GPU use.

No usable independent-group capacity is yet verified for any proposed full
corpus. CommitPackFT and UltraChat justify consideration of a future bounded
census. SciRIFF needs a smaller provenance/template feasibility checkpoint
first. Specifically, establish an original-revision document mapping and
enough disjoint substantive support to make its fixed quotas plausible under
the unchanged isolation rules. A component label or more downloaded rows is
not that evidence. TableGPT/FLAN are not automatic replacements if this fails.

If that checkpoint cannot support the four-family quotas, return a blocker
rather than silently broadening licenses, modifying task templates, trimming
contexts, using assistant answers or reallocating family quotas. A two-family
pilot would be a different experiment and would require review, not an
implicit V3 substitution. This report recommends the minimal conditional
design while explicitly declining to recommend a ready four-family selection.

## Artifacts and validation

Raw samples, exact row-group offsets, pinned URLs, response ETags/ranges,
SHA256 receipts, source cards and tokenizer ledgers are retained under
`local-artifacts/2026-10-30-hf-feasibility/`. The
[source pin list](2026-10-30-hf-source-pins.json) and
[checksums](2026-10-30-hf-feasibility.sha256) bind the evidence. Local raw
assistant answers acquired inside column chunks or JSONL records are not
teacher labels and are not published in this repository.

All 31 CPU audit tests pass; formatting and whitespace checks pass. The
tokenizer is the existing no-default-features example built in its separate
CPU target directory. Original admission/specification and performance
artifacts are untouched. No model output, trained acceptance, GPU timings or
teacher features were produced. Future labels and hidden states must come
exclusively from the frozen target runtime after a reviewed manifest.

Rank256, the three-guess architecture, fixed training schedule, exact-output
gates, joint-only ownership and runtime defaults remain unchanged. GPU usage
remains 0.005923/72 hours; parent caps 4/20/40/8 and non-transferable pilot caps
2/2/1 remain intact. No calibration, teacher generation, training, learned
evaluation, GPU switching, speculative integration or bulk acquisition is
authorized by this feasibility report.
