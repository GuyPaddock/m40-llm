# Original-source comparison bounds

This CPU-only comparison follows the rejected V3 proposal at `c1597dd`.
It does not activate a new specification or authorize corpus acquisition.

## Fixed scope

- SciRIFF component chain: original PubMedQA and NCBI Disease. Inspect QASA
  source/split metadata as a secondary check, not another acquired corpus.
- Alternative original datasets: SQuAD for QA, DocRED for extraction.
- Preserve original train/dev/test partitions. The HF PubMedQA train-only
  packaging must not erase its original CV/test split. Unresolved fold
  membership remains unresolved rather than assigned a research partition.
- PubMedQA: one 128 KiB original JSON prefix. NCBI and DocRED: one 64 KiB
  prefix per annotated split. SQuAD: at most 128 sampled rows per split,
  from three spaced row groups, input columns only.
- Reuse the prior HTTP reader's 32 MiB and 100 recorded requests per source
  cap. Do not execute remote loaders, acquire distant-training data or make
  a full corpus census. No model download or GPU work.
- Retain complete natural contexts/questions. Render SciRIFF template zero
  unchanged for the component comparison. DocRED has no natural user question;
  its one explicit extraction instruction is substantive and must be grouped,
  not exempted as transport framing. No gold entity lists, answers or relations
  enter exported prompts or teacher labels.
- QwenChat, original Qwen2.5-3B Q8_0 tokenizer, no system message/BOS,
  65-447 natural tokens. No trimming, padding, question rewriting, or
  eligibility-dependent choice of template.

## Evidence and decision

Report native IDs, all retained source fields, split provenance, full-input
lengths, distinct document counts before and after the length gate, repeated
instructions, shared spans and literal benchmark overlap. Counts before
semantic/template/rights clearance are upper bounds, not admitted groups.
Reuse the frozen exclusion corpus without treating its known coverage gaps
as resolved. Source answers are never teacher labels.

Four-family quotas remain 256/48/96 each, with 16 independent held-out groups
per family. No split merging, quota transfers or license/template exceptions.
If the inspected sources cannot credibly supply these, stop with a concrete
blocker; do not broaden the search to force admission.

## Atomic delivery

1. `feat(research): inspect native source samples`: bounded input adapters
   and CPU behavioral tests, depending on the existing range reader.
2. `docs(research): compare original data sources`: observed evidence,
   checksums, recommendation and research checkpoint after CPU tokenization.

No runtime, rank-256 architecture, 1.538085 ms proposal-cost result, training
schedule or GPU-budget change is part of either commit.
