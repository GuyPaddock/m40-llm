"""R0 audits must account for duplicate components and benchmark overlap together.

This report extends the frozen V2 graph with normalized-AST and runtime-token
span evidence. Manual semantic decisions are separate, attributable input.
Remaining counts are upper bounds, never automatic training admission.
"""

import argparse
import difflib
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path

from group_capacity import find_root, normalize_text, word_shingles
from prepare_v2 import file_hash, read_tasks, require_hash, write_new
from template_audit import normalized_solution, shared_span_edges


def scan_exclusions(text_path: Path, token_path: Path, statements: dict[int, str],
                    sequences: dict[int, list[int]]) -> dict[int, list[dict[str, str]]]:
    """Check every retained benchmark text for the frozen literal overlap rules.

    Args:
        text_path: Deduplicated normalized benchmark text JSONL.
        token_path: Matching runtime-tokenizer output for benchmark text.
        statements: Original MBPP statement per task.
        sequences: Normalized source comparison tokens per task.

    Returns:
        Source tasks and hashed benchmark witnesses; no answer text is copied.
    """
    source_shingles = {task_id: word_shingles(normalize_text(text)) for task_id, text in statements.items()}
    source_text: dict[str, set[int]] = defaultdict(set)
    for task_id, text in statements.items():
        source_text[normalize_text(text)].add(task_id)
    inverted: dict[tuple[str, ...], set[int]] = defaultdict(set)
    for task_id, shingles in source_shingles.items():
        for shingle in shingles:
            inverted[shingle].add(task_id)
    matches: dict[int, list[dict[str, str]]] = defaultdict(list)
    text_ids: set[str] = set()
    with text_path.open() as stream:
        for row in map(json.loads, stream):
            text_ids.add(row["id"])
            text = row["comparison_text"]
            for task_id in source_text.get(text, set()):
                matches[task_id].append({"benchmark_id": row["id"], "reason": "exact_normalized_text"})
            shingles = word_shingles(text)
            candidates: set[int] = set()
            for shingle in shingles:
                candidates.update(inverted.get(shingle, set()))
            for task_id in candidates:
                other = source_shingles[task_id]
                if 2 * len(shingles & other) >= len(shingles | other):
                    matches[task_id].append({"benchmark_id": row["id"], "reason": "five_word_jaccard"})
    spans: dict[tuple[int, ...], set[int]] = defaultdict(set)
    for task_id, tokens in sequences.items():
        for offset in range(len(tokens) - 31):
            spans[tuple(tokens[offset:offset + 32])].add(task_id)
    token_ids: set[str] = set()
    with token_path.open() as stream:
        for row in map(json.loads, stream):
            if row["id"] in token_ids:
                raise ValueError("duplicate benchmark token row")
            token_ids.add(row["id"])
            tokens = row["comparison_ids"]
            hit_tasks: set[int] = set()
            for offset in range(len(tokens) - 31):
                hit_tasks.update(spans.get(tuple(tokens[offset:offset + 32]), set()))
            for task_id in hit_tasks:
                matches[task_id].append({"benchmark_id": row["id"], "reason": "shared_32_runtime_tokens"})
    if text_ids != token_ids:
        raise ValueError("benchmark tokenizer coverage is incomplete")
    return dict(matches)


def run_audit(tokens_path: Path, exclusions: Path, output: Path, review_path: Path | None) -> None:
    """Write source-bound remaining-capacity and unresolved-review evidence.

    Args:
        tokens_path: Runtime-tokenized original source prompts.
        exclusions: Expanded benchmark corpus directory with token evidence.
        output: New report destination.
        review_path: Optional explicit manual semantic decisions.
    """
    sources = Path("local-artifacts/2026-10-29-drafter-data/sources")
    tasks = read_tasks(sources, Path("local-artifacts/2026-10-29-drafter-data/row-ledger.json"),
                       Path("docs/experiments/2026-10-29-drafter-r1-decision.json"))
    prior_path = Path("local-artifacts/2026-10-29-drafter-data-v2/candidates.json")
    require_hash(prior_path, "8aec641405b976e696a1c295d6139daa82cbd3dfe8fbd5f6cb99a3ff641e6020")
    prior = json.loads(prior_path.read_text())["mbpp"]
    with tokens_path.open() as stream:
        sequences = {int(row["id"].split(":")[1]): row["comparison_ids"] for row in map(json.loads, stream)
                     if row["id"].startswith("mbpp:")}
    if set(sequences) != {task.task_id for task in tasks}:
        raise ValueError("source tokenizer coverage differs")
    edges = list(prior["edges"])
    fingerprints: dict[str, list[int]] = defaultdict(list)
    unresolved_ast = []
    for task in tasks:
        digest = normalized_solution(task.code)
        if digest is None:
            unresolved_ast.append(task.task_id)
        else:
            fingerprints[digest].append(task.task_id)
    for members in fingerprints.values():
        for left, right in itertools.combinations(sorted(members), 2):
            edges.append({"left": left, "right": right, "reasons": ["alpha_normalized_simple_scope"]})
    spans = shared_span_edges(sequences)
    edges.extend({"left": left, "right": right, "reasons": ["shared_32_runtime_tokens"]} for left, right in spans)
    review = json.loads(review_path.read_text()) if review_path else {"groups": [], "excluded_task_ids": [], "reviewed_task_ids": []}
    if review_path and review.get("source_identity_record_sha256") != file_hash(Path("docs/experiments/2026-10-29-drafter-r1-decision.json")):
        raise ValueError("manual review source identity differs")
    for group in review["groups"]:
        for left, right in itertools.combinations(sorted(group["task_ids"]), 2):
            edges.append({"left": left, "right": right, "reasons": ["manual_template:" + group["reason"]]})
    parents = {task.task_id: task.task_id for task in tasks}
    for edge in edges:
        roots = sorted([find_root(parents, edge["left"]), find_root(parents, edge["right"])])
        parents[roots[1]] = roots[0]
    components: dict[int, list[int]] = defaultdict(list)
    splits = {task.task_id: task.split for task in tasks}
    texts = {task.task_id: task.text for task in tasks}
    for task in tasks:
        components[find_root(parents, task.task_id)].append(task.task_id)
    overlap = scan_exclusions(exclusions / "inputs.jsonl", exclusions / "tokens.jsonl", texts, sequences)
    excluded = set(overlap) | set(review["excluded_task_ids"])
    queue = []
    for left, right in itertools.combinations(sorted(texts), 2):
        if find_root(parents, left) == find_root(parents, right):
            continue
        a = re.sub(r"^write (?:a |an )?(?:python )?function to ", "", normalize_text(texts[left]))
        b = re.sub(r"^write (?:a |an )?(?:python )?function to ", "", normalize_text(texts[right]))
        ratio = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
        if ratio >= 0.78:
            queue.append({"left": left, "right": right, "ratio": ratio, "left_text": texts[left], "right_text": texts[right]})
    group_rows = []
    counts = {split: {"row_upper_bound": 0, "independent_group_upper_bound": 0} for split in ("train", "calibration", "held_out")}
    for root, members in sorted(components.items()):
        source_splits = sorted({splits[task_id] for task_id in members})
        reasons = []
        if len(source_splits) > 1:
            reasons.append("cross_partition_template")
        if excluded.intersection(members):
            reasons.append("benchmark_overlap")
        if not reasons:
            counts[source_splits[0]]["row_upper_bound"] += len(members)
            counts[source_splits[0]]["independent_group_upper_bound"] += 1
        group_rows.append({"group_id": f"mbpp:{root}", "task_ids": sorted(members), "source_splits": source_splits,
                           "quarantine_reasons": reasons, "is_selected": False})
    inventory = json.loads((exclusions / "inventory.json").read_text())
    write_new(output, {"kind": "remaining_mbpp_cpu_audit", "edges": edges, "components": group_rows,
              "source_tokens_sha256": file_hash(tokens_path), "exclusion_inventory_sha256": file_hash(exclusions / "inventory.json"),
              "manual_review_sha256": file_hash(review_path) if review_path else None,
              "remaining_capacity": counts, "shared_span_pair_count": len(spans),
              "alpha_normalized_tasks": len(tasks) - len(unresolved_ast), "complex_scope_review_pending": unresolved_ast,
              "benchmark_overlap": overlap, "paraphrase_review_queue": queue,
              "unreviewed_task_ids": sorted(set(texts) - set(review["reviewed_task_ids"])),
              "benchmark_parse_gaps": [row for row in inventory["files"] if row["errors"]],
              "missing_benchmark_references": inventory["missing_local_references"],
              "is_complete_manual_template_audit": len(set(texts) - set(review["reviewed_task_ids"])) == 0,
              "is_r1_admitted": False, "additional_gpu_hours": 0})


def main() -> None:
    """Run explicitly supplied local span, template and exclusion checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    run_audit(args.tokens, args.exclusions, args.output, args.review)


if __name__ == "__main__":
    main()
