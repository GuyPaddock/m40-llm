# Original-source selection comparison

## Decision

**No four-family V3 combination is ready for acquisition or R1 admission.**
Stop this bounded comparison without expanding the search or changing the
isolation contract.

Keep CommitPackFT Python and UltraChat first-user SFT as provisional code
and prose leads. Prefer **original SQuAD 1.1 as the next grounded-QA lead**
over the inspected SciRIFF components. SQuAD retains article titles and
human-authored questions, and its sampled lengths are suitable. This is a
source-selection recommendation, not quota, license or semantic clearance.

The extraction/classification slot remains empty. Original NCBI Disease
provides PMIDs, but its unchanged SciRIFF instruction creates a shared
substantive template across splits. Its fixed-template test capacity is also
too small. DocRED provides document titles and suitable lengths, but one
extraction instruction is still one substantive template. Calling it a
transport wrapper would weaken the approved rules.

The [review decision](2026-10-30-original-source-decision.json) preserves
256/48/96 train/calibration/held-out prompts per family, and the 16 independent
held-out-group minimum. No quota moves to the other three families. The
earlier [negative V3 admission](2026-10-30-hf-dataset-feasibility.md) remains
unchanged; this document does not replace its specification artifacts.

## Scope and identities

The [fixed protocol](2026-10-30-source-selection-protocol.md) limited payload
inspection to four sources. QASA received only a pinned source-code check.
No additional mixture, dataset corpus or model was downloaded.

| Source | HF revision | Original evidence |
| --- | --- | --- |
| [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA/tree/9001f2853fb87cab8d220904e0de81ac6973b318) | `9001f2853fb87cab8d220904e0de81ac6973b318` | Original author repository `pubmedqa/pubmedqa`, `1cbae8e92f72f20c8d3747cbb3bf5bc53554d997` |
| [NCBI Disease](https://huggingface.co/datasets/bigbio/ncbi_disease/tree/b96b632b1c1c245b5cb21faaeec8bdac3b67553e) | `b96b632b1c1c245b5cb21faaeec8bdac3b67553e` | Original-data mirror `spyysalo/ncbi-disease`, `5a266c6fc76fb2b5184cfa397ff0a5ecdb133aec` |
| [SQuAD](https://huggingface.co/datasets/rajpurkar/squad/tree/7b6d24c440a36b6815f21b70d25016731768db1f) | `7b6d24c440a36b6815f21b70d25016731768db1f` | Publisher's original `plain_text` data, not an instruction conversion |
| [DocRED](https://huggingface.co/datasets/thunlp/docred/tree/7985b4e0371e6c61a756feb41b7b27becf71c666) | `7985b4e0371e6c61a756feb41b7b27becf71c666` | Publisher metadata/code `thunlp/DocRED`, `64622e608d06e83deda6b6e0e871effb9d1fd74c` |

SciRIFF templates and formatters remain pinned to
`e396d00ac844351bc4e166744c79f8fb53020c65`; its HF source remains
`eddf83df1ab4acf1ead5b81898dd4c7e7450050e`. New original-source identities do
not retroactively prove the historical SciRIFF ordinal-to-paper mapping.

The fetcher retained 67 successful response receipts totaling 1,965,350 bytes,
excluding metadata viewed through the web tool. All source directories stayed
below the existing 32 MiB/100-request caps. PubMedQA used a 128 KiB JSON
prefix; NCBI and DocRED used 64 KiB prefixes per split. SQuAD sampled three
spaced Parquet row groups per split, four input columns only. No complete
source corpus or full population census was acquired.

## Actual fields and split provenance

| Source | Inspected fields and identity | Original partitions and grouping implications |
| --- | --- | --- |
| PubMedQA | Native JSON keys are PMIDs. Records contain `QUESTION`, `CONTEXTS`, `LABELS`, `MESHES`, `YEAR`, `LONG_ANSWER`, `final_decision`, `reasoning_free_pred`, `reasoning_required_pred`. | HF packages PQA-L as one train split, but the author defines 500 test and 500 CV records, with 450 train/50 dev per fold. Retained original test-ID metadata identifies 23 sampled test records. The other 28 remain `cv_unresolved`; no fold membership was invented or merged. |
| NCBI | Original PubTator records contain PMID, title, abstract, mention offsets/types/concepts. The pinned BigBio source schema has `pmid`, `title`, `abstract`, `mentions`; KB schema retains `document_id=pmid`. | Original train/devel/test files are separate. Mirror tree has 593/100/100 document files; one train PMID, 8528200, occurs twice. Group by PMID and duplicate/template component, never file count alone. |
| SQuAD | Parquet schema is `id`, `title`, `context`, `question`, `answers`. Only the first four columns were sampled. `id` identifies a question; `title` identifies the article within the pinned dataset. | HF reports 87,599 train and 10,570 validation questions. A possible future design subdivides original train by article/template into train/calibration, and reserves original validation for held-out. No cross-split moves or merging. |
| DocRED | Native annotated JSON has `title`, `sents`, `vertexSet`, `labels`; test lacks `labels`. Entity mentions have names, sentence/position and type; relation labels carry head/tail indices and evidence. | Publisher card reports 3,053 annotated train, 998 dev, 1,000 test; 101,873 distant-train rows were not acquired or considered. Preserve annotated train/dev/test. Test answer absence is not repaired with source labels; eventual teacher output must come from the frozen target. |

NCBI's mirror explicitly attributes the original NLM files, and its PMIDs
provide a usable identity path. This study did not hash-compare the complete
mirror against today's NLM archives or recover the original SciRIFF build.
The evidence establishes native IDs and a documented source chain, not a
fully audited historical conversion.

SQuAD and DocRED titles are native article keys, not permanent Wikipedia page
IDs or revision IDs. Redirect aliases, copied passages and source revisions
still need review. Keep their title, pinned dataset revision, record locator
and text hash together. A text hash binds evidence; it does not substitute
for an unresolved source identity.

The pinned SciRIFF QASA downloader selects `testset_answerable_1554_v1.1.json`;
its converter writes only `qasa/test.jsonl`. This chain supplies no training
or calibration split. It cannot repair those quotas by relabeling its test
records. No QASA payload was downloaded in this comparison.

## Natural prompt measurements

The existing CPU-only tokenizer checked the original Qwen2.5-3B Q8_0 GGUF
SHA256 `4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`
and exact prompt roundtrips. QwenChat framing uses no system message or BOS.
There are **673 input variants**, not 673 independent documents. The 51
PubMedQA records each have original and SciRIFF views; these are not extra
training examples or extra capacity.

Original PubMedQA and SQuAD inputs use full context plus the original question,
with only `Context:`/`Question:` field headings. PubMedQA retains its section
labels. This is a proposed original-source adapter, not removal of instructions
from existing SciRIFF examples. The separate SciRIFF views retain template
zero and its complete substantive instruction. NCBI title and abstract are
joined as the existing BigBio formatter does. DocRED retains every source
sentence, joins its original tokens with spaces, and adds one fixed entity/
relation extraction request; it has no native user instruction to preserve.
That request is explicitly a substantive dependency, not a waiver.

No answer, gold entity list, relation label, MeSH list or predicted label is
exported into a prompt. No context was trimmed, padded, paraphrased or selected
using answer labels. Incomplete final records at byte-prefix boundaries were
discarded rather than repaired. Source answers are never teacher labels.

| Input/split | n | Min | Median | P90 | Max | 65-447 rows | Distinct documents in range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PubMedQA original, CV unresolved | 28 | 156 | 345 | 461 | 540 | 24 | 24 |
| PubMedQA original, test | 23 | 230 | 344 | 416 | 503 | 21 | 21 |
| PubMedQA SciRIFF, CV unresolved | 28 | 207 | 396 | 512 | 591 | 20 | 20 |
| PubMedQA SciRIFF, test | 23 | 281 | 395 | 467 | 554 | 18 | 18 |
| NCBI SciRIFF, train | 33 | 213 | 449 | 619 | 824 | 16 | 16 |
| NCBI SciRIFF, devel | 37 | 254 | 409 | 535 | 611 | 23 | 23 |
| NCBI SciRIFF, test | 35 | 272 | 398 | 531 | 769 | 21 | 21 |
| SQuAD original, train | 126 | 57 | 190.5 | 305 | 609 | 122 | 9 |
| SQuAD original, validation | 126 | 53 | 185 | 289 | 460 | 122 | 11 |
| DocRED original, annotated train | 71 | 172 | 238 | 344 | 489 | 69 | 69 |
| DocRED original, dev | 69 | 160 | 233 | 355 | 509 | 67 | 67 |
| DocRED original, test | 74 | 169 | 249.5 | 356 | 544 | 71 | 71 |

These are nonrandom samples. SQuAD is article-ordered, so three row groups
inspect many questions but few articles. Prefixes similarly have order bias.
No sample percentage is multiplied by population size to claim admission.
Only SQuAD has sampled 65-127-token inputs here, five train and eleven
validation. The unchanged global short-context bucket quota also needs a
future cross-family census; length eligibility alone does not fill it.

## Independent capacity and repeated instructions

**NCBI:** all 105 sampled inputs share a 32-token span under the unchanged
instruction: 5,460 pairs, including 3,671 across original splits. Its three
wording variants still implement one disease-extraction task. Splitting by
PMID alone would therefore violate substantive-template isolation. Keeping
the component in one partition cannot supply its own train/calibration/test
quotas or 16 independent held-out template/document groups.

There is also a fixed-adapter upper bound independent of sampling bias. The
mirror enumerates 100 test documents, and 14 distinct sampled test documents
already exceed 447 tokens. Even if every uninspected test document fits,
**at most 86** are eligible, below the required 96. This is not a projected
percentage or a complete census. It applies to the retained source and fixed
template-zero adapter, not every conceivable shorter instruction. No alternate
template was chosen to rescue it.

**PubMedQA:** all 51 SciRIFF views share the fixed instruction, producing
1,275 span pairs including 644 between test and unresolved CV. The four
source wording variants are not four independent substantive tasks. Original
natural questions avoid that added shared span: no different-paper original
pair triggered the literal checks. This removes one demonstrated dependency,
but does not certify the authored questions as template-disjoint. Group
paraphrases and PMID links before quota selection. The original 50-record
dev fold has little headroom for 48 calibration prompts; its actual eligible
capacity is unmeasured, and fold membership is unresolved here.

**SQuAD:** 244 length-eligible questions cover 20 article titles, nine train
and eleven validation. Four sampled shared-span pairs all belong to the same
article and original partition. Article grouping is therefore mandatory even
when question IDs differ. There are no observed cross-split title collisions,
but unseen article aliases, paraphrased question patterns, and benchmark
dependencies can only reduce capacity. The eleven sampled held-out articles
do not demonstrate the 16-group requirement. The much larger original source
makes the fixed QA quotas plausible, not verified.

**DocRED:** 207 of 214 sampled documents fit, and no 32-token span pair was
found. That does not establish template independence: the fixed extraction
request is shorter than 32 tokens but identical in meaning for every row.
Its semantic task dependency connects the three splits regardless of literal
span detection. Varying wording, entity pairs or relation IDs cannot be used
to manufacture independent templates. This adapter cannot supply the required
held-out groups under the unchanged contract.

Across all inputs there are 6,790 shared-span pairs. The remaining 51 are
the deliberately duplicated original/SciRIFF PubMedQA views of the same
papers. All input variants were compared against the retained 265,771-text
benchmark inventory with the frozen literal rules; **zero hits** does not
mean semantic clearance. Existing exclusion-inventory coverage gaps remain.
No sampled document overlap appeared between the distinct source corpora,
but complete-source overlap remains unmeasured.

## Rights and provenance

| Source | Verified evidence | Remaining review |
| --- | --- | --- |
| PubMedQA | Author repository has an MIT notice, PMIDs and original split code; questions/context derive from biomedical papers. | Retain author notice/citation. MIT annotations do not establish rights to every journal abstract; no corpus-wide commercial or redistribution clearance is claimed. |
| NCBI | [NLM's public-domain notice](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/disclaimer.html) is present in the pinned original-data mirror, with requested citations. BigBio labels it CC0. | Preserve the government-work notice and attribution. Distinguish NLM annotation work from underlying journal text and assess applicable rights before distributing text, features or a trained artifact. |
| SQuAD | [Original publisher](https://rajpurkar.github.io/SQuAD-explorer/) and pinned HF card state CC-BY-SA 4.0; crowdworker questions over Wikipedia paragraphs. | [CC-BY-SA permits commercial use with conditions](https://creativecommons.org/licenses/by-sa/4.0/); it is not an MIT-style unconditional grant. Retain attribution, license and change notices, trace article provenance, and review share-alike obligations for the actual redistributed artifacts. |
| DocRED | Pinned publisher MIT license names Tsinghua NLP Group; publisher describes Wikipedia text and Wikidata relations. | MIT does not erase Wikipedia text terms. [Wikidata structured data is CC0](https://www.wikidata.org/wiki/Wikidata:Licensing), not a license for the associated article text. Preserve both publisher and upstream attribution and investigate source revisions. |

No source in this comparison is cleared for training or redistribution.
The original target model's separate restrictions also remain unchanged.
These are technical provenance findings, not a legal opinion that input
rights automatically transfer to teacher features or learned weights.

## Concrete review recommendation

| Family | Provisional choice | Fixed train/calibration/held-out quota | Status |
| --- | --- | --- | --- |
| Programming/code | CommitPackFT Python | 256 / 48 / 96 | Prior preferred lead; no new clearance |
| Prose/conversation | UltraChat first-user SFT | 256 / 48 / 96 | Prior preferred lead; no new clearance |
| Grounded QA | Original SQuAD 1.1 | 256 / 48 / 96 | Preferred new lead; document/template census and rights review pending |
| Extraction/classification | None from this comparison | 256 / 48 / 96 | NCBI and DocRED fail fixed-template independence; NCBI also fails the fixed-adapter test bound |

There is no smallest viable four-family source combination to recommend
under these constraints. The smallest useful shortlist has three provisional
sources and an explicit missing fourth slot. Do not fill it by relabeling QA
as extraction, merging original splits, transferring quotas, paraphrasing one
instruction into multiple templates, or reusing source answers as supervision.

Before any broader acquisition, review this blocker. A new extraction lead
would need evidence of independently authored substantive tasks and native
documents, rather than another fixed-label corpus with only more rows. This
study does not authorize that next search or a two/three-family pilot.

## Validation and retained evidence

Tooling commit `a179e6f` passes 39 CPU tests, including complete-record parsing,
PMID mismatch rejection, answer exclusion, source partition retention and
ChatML-control rejection. The CPU tokenizer processed all 673 variants with
no CUDA context. Formatting, JSON, local-link and whitespace checks pass.

[Source pins](2026-10-30-original-source-pins.json),
[sample summary](2026-10-30-original-source-summary.json) and
[checksums](2026-10-30-original-source.sha256) bind the new evidence. Raw
prefixes, response receipts, templates, schema metadata, token ledgers and
per-pair overlap details remain local under
`local-artifacts/2026-10-30-source-selection/`. No sampled source text or
gold answers are published in Git. Prior manifests and failed admissions
are unchanged and their checksums still pass.

Rank-256 architecture, fixed schedule, exact-output contracts, 1.538085 ms
cost-only result, 72-hour aggregate ceiling, phase caps 4/20/40/8 and pilot
caps 2/2/1 are unchanged. GPU usage remains 0.005923 hours. No teacher
generation, training, learned evaluation, calibration, GPU switching,
speculative integration, runtime-default change or performance claim occurred.
