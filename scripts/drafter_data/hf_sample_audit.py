"""Small source samples expose length and isolation risks before acquisition.

Only source prompts enter the original Qwen tokenizer. Dataset answers remain
unused evidence, never teacher labels or hidden-state targets.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from group_capacity import normalize_text
from mbpp_remaining import scan_exclusions
from prepare_v2 import file_hash, write_new
from template_audit import shared_span_edges


def prompt_text(source: str, record: dict[str, object]) -> str:
    """Choose the immutable input side, never the dataset completion.

    Args:
        source: One of the three primary candidate source keys.
        record: Parsed source record.

    Returns:
        Natural single-user input, with a fixed filename separator for code.
    """
    keys = {"sciriff": ["input"], "ultrachat": ["prompt"],
            "commitpack": ["subject", "old_file", "old_contents"]}[source]
    if not all(isinstance(record.get(key), str) for key in keys):
        raise ValueError("source prompt fields must be strings")
    result = str(record[keys[0]])
    if source == "commitpack":
        result += "\n\nFile: " + str(record["old_file"]) + "\n" + str(record["old_contents"])
    if not result.strip() or "<|im_start|>" in result or "<|im_end|>" in result:
        raise ValueError("empty or preformatted source prompt needs quarantine")
    return result


def prepare_samples(root: Path) -> None:
    """Create bounded input-only JSONL and identity metadata for tokenization.

    Args:
        root: Existing pinned sample and metadata directory.
    """
    samples: list[dict[str, object]] = []
    for source, splits in [("sciriff", ["train", "validation", "test"]),
                           ("ultrachat", ["train", "test"]), ("commitpack", ["python"])]:
        api = json.loads((root / f"{source}-api.json").read_text())
        for split in splits:
            if source == "commitpack":
                path = root / "commitpack-python-prefix.bin"
                lines = path.read_bytes().split(b"\n")[:-1][:128]
                rows = [{"record": json.loads(line), "source_line": index + 1} for index, line in enumerate(lines)]
            else:
                path = root / f"{source}-{split}.json"
                rows = json.loads(path.read_text())["rows"]
            if len(rows) > 128:
                raise ValueError("per-split sample bound exceeded")
            for row in rows:
                record = row["record"]
                text = prompt_text(source, record)
                if source == "sciriff":
                    source_id = record["_instance_id"]
                    group = "template:" + source_id.split(":")[0]
                elif source == "ultrachat":
                    source_id = record["prompt_id"]
                    group = "conversation:" + source_id
                else:
                    source_id = record["commit"] + ":" + record["old_file"] + ":" + str(row["source_line"])
                    group = "repositories:" + ",".join(sorted(part.strip().lower() for part in record["repos"].split(",")))
                samples.append({"id": f"{source}:{split}:{source_id}", "text": text,
                                "comparison_text": normalize_text(text), "source": source, "split": split,
                                "revision": api["sha"], "source_id": source_id, "provisional_group": group,
                                "sample_file": str(path), "sample_sha256": file_hash(path),
                                "record_sha256": hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(),
                                "license_label": record.get("license"), "is_training_admitted": False})
    with (root / "sample-inputs.jsonl").open("x") as stream:
        for sample in samples:
            stream.write(json.dumps(sample) + "\n")


def summarize_samples(root: Path, exclusions: Path | None) -> None:
    """Report observed lengths and overlap without population extrapolation.

    Args:
        root: Sample inputs and original-runtime tokenizer output.
        exclusions: Optional frozen literal benchmark corpus from the R0 audit.
    """
    with (root / "sample-inputs.jsonl").open() as stream:
        samples = list(map(json.loads, stream))
    with (root / "sample-tokens.jsonl").open() as stream:
        tokens = {row["id"]: row for row in map(json.loads, stream)}
    if set(tokens) != {row["id"] for row in samples}:
        raise ValueError("sample tokenizer coverage differs")
    summary = []
    for source, split in sorted({(row["source"], row["split"]) for row in samples}):
        subset = [row for row in samples if row["source"] == source and row["split"] == split]
        lengths = sorted(tokens[row["id"]]["prompt_tokens"] for row in subset)
        summary.append({"source": source, "split": split, "rows": len(lengths),
                        "min": min(lengths), "median": statistics.median(lengths),
                        "p90_nearest_rank": lengths[(9 * len(lengths) + 9) // 10 - 1], "max": max(lengths),
                        "buckets": {f"{low}-{high}": sum(low <= n <= high for n in lengths)
                                    for low, high in [(1, 64), (65, 127), (128, 255), (256, 447), (448, 1000000)]},
                        "provisional_groups_1_447": len({row["provisional_group"] for row in subset
                                                        if tokens[row["id"]]["prompt_tokens"] <= 447}),
                        "provisional_groups_65_447": len({row["provisional_group"] for row in subset
                                                         if 65 <= tokens[row["id"]]["prompt_tokens"] <= 447})})
    sequences = {index: tokens[row["id"]]["comparison_ids"] for index, row in enumerate(samples)}
    spans = shared_span_edges(sequences)
    overlap = scan_exclusions(exclusions / "inputs.jsonl", exclusions / "tokens.jsonl",
                              {index: row["text"] for index, row in enumerate(samples)}, sequences) if exclusions else None
    write_new(root / "sample-report.json", {"kind": "bounded_dataset_feasibility_sample",
              "inputs_sha256": file_hash(root / "sample-inputs.jsonl"),
              "tokens_sha256": file_hash(root / "sample-tokens.jsonl"),
              "lengths": summary, "shared_span_edges": spans,
              "sample_ids": [row["id"] for row in samples], "benchmark_literal_overlap": overlap,
              "exclusion_inventory_sha256": file_hash(exclusions / "inventory.json") if exclusions else None,
              "has_full_corpus_census": False, "has_complete_semantic_clearance": False,
              "is_r1_admitted": False, "additional_gpu_hours": 0})


def main() -> None:
    """Prepare or summarize explicitly downloaded small source samples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--exclusions", type=Path)
    args = parser.parse_args()
    if args.summarize:
        summarize_samples(args.root, args.exclusions)
    else:
        prepare_samples(args.root)


if __name__ == "__main__":
    main()
