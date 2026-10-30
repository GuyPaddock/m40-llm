"""Official MBPP partitions can share substantive problem templates.

This CPU-only necessary-capacity audit connects definite duplicate evidence
before counting usable rows. Cross-partition components are quarantined, never
moved. A passing upper bound is not a manual template or benchmark audit.
"""

import ast
import hashlib
import itertools
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    """An inert source record bound to an existing runtime-tokenizer census.

    Attributes:
        task_id: Official MBPP task ID.
        split: Original train, calibration or held_out partition.
        text: Unmodified natural-language problem statement.
        code: Reference program used only for duplicate detection, never run.
        prompt_tokens: Actual formatted runtime-tokenizer length.
    """

    task_id: int
    split: str
    text: str
    code: str
    prompt_tokens: int


def normalize_text(text: str) -> str:
    """Normalize comparison text without altering a source prompt.

    Args:
        text: Original source statement.

    Returns:
        NFKC, lowercase text with collapsed whitespace.
    """
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def solution_fingerprint(code: str) -> str | None:
    """Fingerprint parseable Python without executing or renaming identifiers.

    Only location metadata and source formatting are removed. Keeping names,
    constants and operators makes this a conservative subset of the planned
    normalized-AST template audit, not a substitute for manual review.

    Args:
        code: Untrusted reference program, treated as text.

    Returns:
        AST digest, or None when source cannot be parsed or has an empty body.
    """
    result = None
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        tree = None
    if tree is not None and tree.body:
        payload = ast.dump(tree, include_attributes=False).encode("utf-8")
        result = hashlib.sha256(payload).hexdigest()
    return result


def word_shingles(text: str) -> set[tuple[str, ...]]:
    """Produce the frozen five-word shingles for normalized comparison text.

    Args:
        text: Already normalized problem statement.

    Returns:
        Unique five-word sequences; short statements have no shingles.
    """
    words = text.split()
    return {tuple(words[index:index + 5]) for index in range(len(words) - 4)}


def find_root(parents: dict[int, int], task_id: int) -> int:
    """Resolve and compress a duplicate component's parent chain.

    Args:
        parents: Mutable task-to-parent mapping.
        task_id: Existing source task.

    Returns:
        Stable minimum-ID component root.
    """
    root = task_id
    while parents[root] != root:
        root = parents[root]
    while parents[task_id] != root:
        next_id = parents[task_id]
        parents[task_id] = root
        task_id = next_id
    return root


def audit_groups(tasks: list[Task], buckets: list[list[int]]) -> dict[str, object]:
    """Quarantine whole cross-partition duplicate components before counting.

    Exact normalized text, identical parsed solutions and Jaccard >=0.5
    connect records. Unparseable programs and invalid prompts stay quarantined.
    Remaining counts are upper bounds before the additional frozen filters.

    Args:
        tasks: All official source records, including length-ineligible ones.
        buckets: Approved inclusive prompt-length buckets for MBPP.

    Returns:
        Content-free edge, component, row-disposition and capacity evidence.

    Raises:
        ValueError: Source IDs are duplicated or a split name is invalid.
    """
    ids = {task.task_id for task in tasks}
    if len(ids) != len(tasks):
        raise ValueError("duplicate task IDs")
    if any(task.split not in {"train", "calibration", "held_out"} for task in tasks):
        raise ValueError("unknown source split")
    parents = {task_id: task_id for task_id in ids}
    texts = {task.task_id: normalize_text(task.text) for task in tasks}
    shingles = {task_id: word_shingles(text) for task_id, text in texts.items()}
    solutions = {task.task_id: solution_fingerprint(task.code) for task in tasks}
    edges: list[dict[str, object]] = []
    for left, right in itertools.combinations(sorted(ids), 2):
        reasons: list[str] = []
        if texts[left] and texts[left] == texts[right]:
            reasons.append("exact_normalized_text")
        if solutions[left] is not None and solutions[left] == solutions[right]:
            reasons.append("identical_parsed_solution")
        union = shingles[left] | shingles[right]
        intersection = shingles[left] & shingles[right]
        if union and 2 * len(intersection) >= len(union):
            reasons.append("five_word_jaccard_at_least_half")
        if reasons:
            roots = sorted([find_root(parents, left), find_root(parents, right)])
            parents[roots[1]] = roots[0]
            edges.append({"left": left, "right": right, "reasons": reasons})
    return summarize_groups(tasks, parents, solutions, buckets, edges)


def summarize_groups(
    tasks: list[Task], parents: dict[int, int], solutions: dict[int, str | None],
    buckets: list[list[int]], edges: list[dict[str, object]],
) -> dict[str, object]:
    """Record stable components and conservative residual capacity.

    Args:
        tasks: Original source records.
        parents: Complete duplicate connections.
        solutions: Parsed solution fingerprints or unresolved values.
        buckets: Approved inclusive length ranges.
        edges: Auditable duplicate witnesses without third-party prompt text.

    Returns:
        Row/group records and per-partition upper bounds, never admission.
    """
    groups: dict[int, list[Task]] = {}
    for task in tasks:
        groups.setdefault(find_root(parents, task.task_id), []).append(task)
    rows: list[dict[str, object]] = []
    components: list[dict[str, object]] = []
    counts = {split: 0 for split in ("train", "calibration", "held_out")}
    for root, members in sorted(groups.items()):
        splits = sorted({task.split for task in members})
        has_cross_partition_members = len(splits) > 1
        group_id = f"mbpp:{root}"
        components.append({"group_id": group_id, "task_ids": sorted(task.task_id for task in members),
                           "source_splits": splits, "is_cross_partition": has_cross_partition_members})
        for task in sorted(members, key=lambda item: item.task_id):
            reasons = []
            if has_cross_partition_members:
                reasons.append("cross_partition_component")
            if not task.text.strip() or "<|" in task.text:
                reasons.append("invalid_prompt")
            if not any(low <= task.prompt_tokens <= high for low, high in buckets):
                reasons.append("outside_length_buckets")
            if solutions[task.task_id] is None:
                reasons.append("unresolved_solution_ast")
            if not reasons:
                counts[task.split] += 1
            rows.append({"task_id": task.task_id, "source_split": task.split, "group_id": group_id,
                         "prompt_tokens": task.prompt_tokens,
                         "source_text_sha256": hashlib.sha256(task.text.encode("utf-8")).hexdigest(),
                         "solution_ast_sha256": solutions[task.task_id],
                         "quarantine_reasons": reasons,
                         "is_selected": False,
                         "pending": ["normalized_token_span_audit", "manual_template_review",
                                     "expanded_benchmark_exclusions"]})
    return {"edges": edges, "components": components, "rows": rows,
            "residual_row_upper_bounds": counts,
            "is_complete_template_audit": False, "is_r1_admitted": False}
