# Drafter Data Admission Tools

CPU-only necessary-capacity checks for the October29 pilot. A passing
length census does not admit R1. A failed upper bound proves that subsequent
grouping, provenance and exclusion filters cannot meet the frozen quotas.
Stop on that failure; do not pad prompts or revise an immutable specification.
The original census below remains the v1 reproduction path. The separately
approved MBPP short-prompt amendment uses the v2 preparation section below.

The user authorized R0 acquisition and conditionally authorized R1 only after
ALL admission checks pass. Teacher/training/evaluation caps remain2/2/1 GPU
hours, and every device switch requires confirmation. These tools do not
generate teacher features, initialize CUDA, train or admit GPU execution.

## Source Files

Download only the artifacts pinned in
[the data specification](../../docs/experiments/2026-10-29-drafter-data-spec.json).
Keep original payloads and notices under the ignored directory
`local-artifacts/2026-10-29-drafter-data/sources/`. Use MBPP filenames
`train.parquet`, `validation.parquet`, `test.parquet` under `sources/mbpp/`.
The converter binds each JSON output to the original Parquet SHA256 and reads
only `task_id` and `text`. Reference solutions/tests remain inert in Parquet;
they are never executed or used to lengthen prompts.

The reader requires Python3.11 and pinned CPU-only PyArrow22.0.0. Install in
a project-local environment, not the system Python or a training environment:

```bash
UV_CACHE_DIR=local-artifacts/2026-10-29-drafter-data/uv-cache \
  uv venv --python python3.11 --no-python-downloads \
  local-artifacts/2026-10-29-drafter-data/venv
UV_CACHE_DIR=local-artifacts/2026-10-29-drafter-data/uv-cache \
  uv pip install --python local-artifacts/2026-10-29-drafter-data/venv/bin/python \
  --only-binary :all: --no-deps -r scripts/drafter_data/requirements.txt
for split in train validation test; do
  local-artifacts/2026-10-29-drafter-data/venv/bin/python \
    scripts/drafter_data/read_mbpp.py \
    local-artifacts/2026-10-29-drafter-data/sources/mbpp/$split.parquet \
    local-artifacts/2026-10-29-drafter-data/sources/mbpp/$split.json
done
```

Both the converter and census refuse existing output paths. Use fresh paths
for an independent verification; do not overwrite evidence.

## Census And Validation

```bash
source scripts/dev-env.sh
CARGO_TARGET_DIR=target/drafter-data-cpu RUSTFLAGS='-D warnings' \
  cargo run --no-default-features --locked --example drafter_capacity -- \
  --spec docs/experiments/2026-10-29-drafter-data-spec.json \
  --mbpp local-artifacts/2026-10-29-drafter-data/sources/mbpp \
  --output local-artifacts/2026-10-29-drafter-data/capacity.json \
  --row-ledger local-artifacts/2026-10-29-drafter-data/row-ledger.json
```

The census validates the unchanged specification and full GGUF file hash,
official source ID sets, conversion/source hashes, Qwen ChatML IDs, every
prompt roundtrip and vocabulary range. It uses the production formatter and
tokenizer, no alternate tokenizer or character-count estimate. Output contains
per-task token counts, source hashes and quota upper bounds. The local ledger
also contains prompt text and all IDs; keep it with source license notices,
not in the code repository. The census exits nonzero after writing a negative
capacity decision. Even a successful run leaves `is_r1_admitted=false`.

Request test escalation under project policy; these checks need no GPU:

```bash
local-artifacts/2026-10-29-drafter-data/venv/bin/python \
  -m unittest discover -s scripts/drafter_data -v
source scripts/dev-env.sh
CARGO_TARGET_DIR=target/drafter-data-cpu RUSTFLAGS='-D warnings' \
  cargo test --no-default-features --locked \
  --example drafter_capacity --test qwen2_tokenizer_prompt
```

Do not publish a complete realized training manifest, split-isolation pass,
license clearance or teacher authorization based on this necessary check.

## V2 CPU Preparation

The user approved MBPP-only natural short lengths, not GPU work. Use the
parent-linked [v2 specification](../../docs/experiments/2026-10-29-drafter-data-spec-v2.json)
and [frozen calibration protocol](../../docs/experiments/2026-10-29-short-context-calibration-v2.md).
Do not change the original census executable, specification or negative report.

`prepare_v2.py` reuses the hash-bound original runtime-tokenized MBPP ledger
and checks original Parquet payloads, prompt bytes and official partition IDs.
It groups exact normalized text, five-word Jaccard matches and identical parsed
solutions. It never executes reference programs or supplies them as prompts.
This is a conservative subset of the full template audit. All cross-partition
components are quarantined, including links through length-ineligible rows.
Remaining counts are upper bounds before normalized-token spans, manual
paraphrases and the expanded benchmark audit. They are not split admission.

The retained Dolly payload has no reviewed source-document mapping. Context
passages are marked unresolved; a paragraph hash is not a canonical document.
No automatic provenance inference or train/held-out assignment occurs.

```bash
mkdir -p local-artifacts/2026-10-29-drafter-data-v2
local-artifacts/2026-10-29-drafter-data/venv/bin/python \
  scripts/drafter_data/prepare_v2.py \
  --spec docs/experiments/2026-10-29-drafter-data-spec-v2.json \
  --sources local-artifacts/2026-10-29-drafter-data/sources \
  --ledger local-artifacts/2026-10-29-drafter-data/row-ledger.json \
  --decision docs/experiments/2026-10-29-drafter-r1-decision.json \
  --output local-artifacts/2026-10-29-drafter-data-v2/preparation.json \
  --candidates local-artifacts/2026-10-29-drafter-data-v2/candidates.json
```

The nonzero exit explicitly marks an unresolved training-manifest admission.
The candidate file is a realized source-row quarantine ledger, NOT a selected
training split. Report that distinction and every unresolved check. Never
invent document provenance, manual reviewers or exclusion clearances to
produce the requested training manifest. No features or trained artifacts
are produced; GPU use is zero. Run the same unittest discovery command above
under CPU-test escalation to cover the v2 preparation behavior.
