# Drafter Data Admission Stops Before R1

## Decision

The frozen MBPP quotas are impossible under the frozen prompt rules. Stop
before teacher generation, trainer implementation or GPU work. Do not change
the minimum length, add reference solutions/tests to prompts, substitute a
corpus or remove the code stratum to make admission pass.

The user's attached authorization allowed R0 immediately and R1 only if every
admission requirement passed. It also required stopping on a failed prerequisite.
This is that stop, not a request to spend the larger training allocation.
R2 remains unauthorized. No dataset or schedule requirements were revised.

This result does not reject rank256 learned drafting. It rejects this data
specification's combination of MBPP text-only input and a65-token minimum.
The accepted1.538085ms proposal cost remains cost evidence only; learned
acceptance and end-to-end gains remain unmeasured.

## Measured Source Capacity

Implementation `911bc48` uses the existing `prepare_prompt` and `Tokenizer`
with the original Qwen2.5-3B Q8_0 GGUF. It reads only GGUF metadata and source
records; it neither uploads target tensors nor creates a CUDA context. Build
profile is Rust dev, no default features, locked dependencies and warnings
as errors. These are CPU data measurements, not performance benchmarks.

| MBPP partition | Original rows | Prompt tokens, min-max | Eligible65-447 | Frozen quota |
| --- | ---: | ---: | ---: | ---: |
| Train | 374 | 16-58 | 0 | 256 |
| Calibration | 90 | 17-36 | 0 | 48 |
| Held-out | 500 | 16-55 | 0 | 96 |
| Total | 964 | 16-58 | 0 | 400 |

Counts include the actual ChatML wrapper, with no extra BOS/EOS. Input is
MBPP `text` alone. Task IDs are exactly601-974,511-600 and11-510 respectively;
the ten source few-shot examples remain excluded.974 is the maximum task ID,
not the number of audited prompts. Every prompt passed exact encode/decode
roundtrip and vocabulary-range checks with the runtime tokenizer.

These are upper bounds BEFORE document/template grouping and exclusions.
Those operations cannot create additional length-eligible prompts, so further
selection work cannot repair the deficit under the approved specification.
There is no realized training split manifest. The retained row ledger is a
source census with original text, formatted prompts and token IDs, not a
training dataset or approved split.

## Provenance And Limits

The original model hash matches
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
The canonical sorted tokenizer-metadata hash is
`9620a250a9a0b10c3795f2b501e22118c472d755d1e6d62f47f54e6fde04c16d`.
The data specification and plan still match their committed hashes. All three
pinned source/license metadata hashes match the specification.

Retain original payloads, source cards and license text in
`local-artifacts/2026-10-29-drafter-data/sources/`. No model was downloaded.
Source identities and payload hashes are listed in the
[admission decision](2026-10-29-drafter-r1-decision.json).

| Source | Revision | Local audit status |
| --- | --- | --- |
| [Dolly15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a) | bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a | Original JSONL acquired and hashed; selection not run |
| [MBPP full](https://huggingface.co/datasets/google-research-datasets/mbpp/tree/4bb6404fdc6cacfda99d4ac4205087b89d32030c/full) | 4bb6404fdc6cacfda99d4ac4205087b89d32030c | All three approved Parquet partitions acquired, hashed and counted |
| [Qwen license metadata](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE) | aa8e72537993ba99e69dfaafa59ed015b17504d1 | Pinned text integrity checked; no new use or distribution authorized |

The cards identify Databricks/Wikipedia CC BY-SA3.0 and Google Research MBPP
CC BY4.0. Retain source attribution, notices and change records under their
[CC BY-SA3.0](https://creativecommons.org/licenses/by-sa/3.0/legalcode) and
[CC BY4.0](https://creativecommons.org/licenses/by/4.0/legalcode) terms.
Qwen's research restrictions remain in force. This metadata review is not
complete document-provenance or combined artifact-license clearance.

After the capacity failure, the following checks did NOT run: document and
template grouping, source-document resolution, expanded benchmark exclusion
audit, row-level split realization, training toolchain admission, trainer
freeze and causal feature export implementation/validation. None is waived
or represented as passing. The training environment was not installed;
Python3.11.14/PyArrow22.0.0 served only as an isolated CPU Parquet reader.

## GPU Ledger And Unmeasured Results

| Phase | Parent cap, hours | Conditional pilot cap | Additional used |
| --- | ---: | ---: | ---: |
| Cost | 4 | 0 | 0 |
| Teacher and export validation | 20 | 2, including0.25 scout | 0 |
| Training | 40 | 2 | 0 |
| Final evaluation | 8 | 1 | 0 |

Aggregate usage remains0.005923/72 GPU-hours, entirely the earlier cost probe.
No scout, teacher setup, training attempt, GPU validation, cleanup job or device
switch occurred. There is no GPU process to resume. CPU downloads, parsing,
tokenization and tests are not teacher generation.

Teacher throughput/storage measurements, training/calibration loss, held-out
acceptance at guesses1/2/3, full-batch acceptance and eligible coverage are all
unmeasured, represented as null in the decision record. No learned resident or
HTTP projection is calculated. The old hypothetical cost curves remain intact;
this failed corpus cannot supply acceptance estimates for them.

No runtime defaults or numerical/statistical gates changed. The immutable
50.910426 HTTP reference, joint-only configuration and original model remain
intact. All twelve compact-cost checksum entries, including the retained
executable and compiled-control evidence, match. The immutable reference
manifest hash also matches; no GPU benchmark was rerun.

## Verification And Artifacts

- [Raw capacity report](2026-10-29-drafter-capacity.json), including all964 task IDs and token counts.
- [Decision and budget ledger](2026-10-29-drafter-r1-decision.json).
- [Independent source/ledger audit](2026-10-29-drafter-capacity-audit.jq).
- [Audit corruption tests](2026-10-29-drafter-capacity-audit-tests.sh).
- [Recorded audit result](2026-10-29-drafter-capacity-audit.json).
- [Artifact checksums](2026-10-29-drafter-data.sha256).
- [Tool usage and reproduction](../../scripts/drafter_data/README.md).

The local ledger hash is
`c6c16a3e2ed778bb2564959ad4797adb69097d3ec37fa3a34fc2c4e1bab5d721`.
The retained CPU census executable hash is
`11919b2b41ec61b7402216adff03442b062e478ac284b08267fee68927a78140`.
Keep both under `local-artifacts/2026-10-29-drafter-data/`. Do not commit source
prompt text, complete token sequences or third-party dataset payloads as code.

Validation passed3 Python conversion tests,4 Rust capacity tests and7 existing
Qwen tokenizer tests. The independent jq audit recomputes the gate from source
records and the complete token ledger; it accepts the retained report and
rejects five corruptions of capacity, admission, length and quota fields.
Non-CUDA example Clippy with `-D warnings`, formatting and whitespace checks
pass. The census's nonzero exit is the intended negative admission result,
not an unavailable-device skip or a tokenizer crash.

## Next Review

Stop here. A subsequent plan could keep MBPP text-only but admit shorter
natural prompts, which would require explicit coverage and cost-model handling
below the verifier's measured65-token starting position. Alternatively, a new
reviewed code corpus could supply genuinely longer problems. Neither change
is authorized by the present specification. Do not add padding, tests or
answers to MBPP prompts as an unreviewed workaround, and do not reinterpret
this stop as a failed learned drafter or permission to spend the parent budget.
