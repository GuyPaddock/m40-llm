"""R0 recovery uses original prompt bytes and separate comparison normalization.

This preparation preserves source IDs and never creates teacher labels. The
runtime tokenizer consumes the emitted JSONL without initializing CUDA.
"""

import argparse
import json
from pathlib import Path

from group_capacity import normalize_text
from prepare_v2 import read_tasks, require_hash


def build_inputs(sources: Path, output: Path) -> None:
    """Write source prompts and comparison text, retaining original payloads.

    Args:
        sources: Original pinned acquisition directory.
        output: New local JSONL destination.
    """
    decision = Path("docs/experiments/2026-10-29-drafter-r1-decision.json")
    ledger = Path("local-artifacts/2026-10-29-drafter-data/row-ledger.json")
    tasks = read_tasks(sources, ledger, decision)
    dolly = sources / "dolly/databricks-dolly-15k.jsonl"
    require_hash(dolly, "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec")
    with output.open("x", encoding="utf-8") as stream:
        for task in tasks:
            stream.write(json.dumps({"id": f"mbpp:{task.task_id}", "text": task.text,
                                     "comparison_text": normalize_text(task.text)}, ensure_ascii=False) + "\n")
        with dolly.open(encoding="utf-8") as source:
            for line, raw in enumerate(source, 1):
                row = json.loads(raw)
                text = row["instruction"] + ("\n\nContext:\n" + row["context"] if row["context"] else "")
                stream.write(json.dumps({"id": f"dolly:{line}", "text": text,
                                         "comparison_text": normalize_text(text)}, ensure_ascii=False) + "\n")


def main() -> None:
    """Prepare explicitly selected local source artifacts on CPU only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_inputs(args.sources, args.output)


if __name__ == "__main__":
    main()
