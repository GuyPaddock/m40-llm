"""MBPP source partitions contain prompts, reference programs and tests.

This CPU-only conversion preserves task IDs and prompt text for the runtime
tokenizer census. Reference programs are not executed or supplied as prompts.
The original Parquet files remain the provenance source for later audits.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow
import pyarrow.parquet


def read_prompt_rows(source: Path) -> list[dict[str, int | str]]:
    """Read inert prompt records without evaluating any dataset code.

    Args:
        source: Original MBPP Parquet partition.

    Returns:
        Task IDs and unmodified prompt strings in original row order.

    Raises:
        ValueError: A required field has the wrong type or an ID is duplicated.
    """
    table = pyarrow.parquet.read_table(source, columns=["task_id", "text"])
    rows: list[dict[str, int | str]] = []
    seen: set[int] = set()
    for value in table.to_pylist():
        task_id = value["task_id"]
        text = value["text"]
        if type(task_id) is not int or not isinstance(text, str):
            raise ValueError("MBPP task_id must be an integer and text a string")
        if task_id in seen:
            raise ValueError(f"duplicate task_id {task_id}")
        seen.add(task_id)
        rows.append({"task_id": task_id, "text": text})
    return rows


def convert_partition(source: Path, destination: Path) -> None:
    """Write an attributable JSON conversion without overwriting prior evidence.

    Args:
        source: Original pinned Parquet file.
        destination: New JSON artifact, outside tracked source directories.

    Raises:
        FileExistsError: The destination already exists.
        ValueError: Source records violate the prompt schema.
    """
    rows = read_prompt_rows(source)
    with source.open("rb") as stream:
        source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    record = {
        "schema_version": 1,
        "source_sha256": source_sha256,
        "source_file": source.name,
        "source_rows": len(rows),
        "pyarrow_version": pyarrow.__version__,
        "rows": rows,
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")


def main() -> None:
    """Convert one explicitly supplied local source partition on CPU only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    convert_partition(arguments.source, arguments.destination)


if __name__ == "__main__":
    main()
