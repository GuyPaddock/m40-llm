"""Replay matched sparse Q4 task, prefix, and FP32 observations from a finished arm."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import TypedDict

from replay import check_value, compare, read_logits


INPUT_SHA = "18d0c6f69dbdfa3bf976780b5b20e14d06dfec5f32980418a9d4f308da9f1cc3"
VOCABULARY = 151_936


class Generation(TypedDict):
    """Free-run IDs and text retained by the quality runner."""

    generated_token_ids: list[int]
    output: str


class Case(TypedDict):
    """Fields needed from the pinned November fixture."""

    id: str
    context: str
    prompt_ids: list[int]


class Record(TypedDict, total=False):
    """Shared fields across the three per-case record types."""

    kind: str
    case: str
    role: str
    generation: Generation
    score: dict[str, object]
    has_repetition: bool
    first_divergence: int | None
    new_repetition: bool
    dense_passed: bool
    candidate_passed: bool
    projection_calls_before: list[int]
    projection_calls_after: list[int]
    step: int
    target: int
    prefix_tokens: int
    values: dict[str, object]
    summary: dict[str, object]
    candidate_layout: dict[str, object]
    device_bytes_after: int


def has_repetition(ids: list[int]) -> bool:
    """Return whether the trace triggers either frozen repetition rule.

    Args:
        ids: Generated IDs, including the sampled stop token when present.

    Returns:
        Whether a repeated run or nonconstant period is present.
    """
    return any(len(set(ids[start:start + 16])) == 1 for start in range(len(ids) - 15)) or any(
        len(set(ids[start:start + width])) > 1
        and ids[start:start + width] * 4 == ids[start:start + 4 * width]
        for width in range(2, 9)
        for start in range(len(ids) - 4 * width + 1)
    )


def replay_case(case: Case, rows: list[Record], directory: Path) -> tuple[bool, str | None, int]:
    """Check ordering and recompute matched-prefix metrics for one case.

    Args:
        case: Expected ID, context and input IDs from the original manifest.
        rows: Ordered dense, candidate, comparison and logit rows for this case.
        directory: Files storing each full-vocabulary little-endian FP32 vector.

    Returns:
        Dense answer pass, stop reason if any, and finite comparisons replayed.

    Raises:
        ValueError: The report and raw observations disagree.
    """
    dense, sparse, decision, *vectors = rows
    check_value(["sparse_q4_extended_generation"] * 2 + ["sparse_q4_extended_comparison"],
                [row["kind"] for row in rows[:3]], "case order")
    check_value(["dense", "candidate"], [dense["role"], sparse["role"]], "generation order")
    dense_ids = dense["generation"]["generated_token_ids"]
    candidate_ids = sparse["generation"]["generated_token_ids"]
    divergence = next((i for i, pair in enumerate(itertools.zip_longest(dense_ids, candidate_ids))
                       if pair[0] != pair[1]), None)
    new_repetition = has_repetition(candidate_ids) and not has_repetition(dense_ids)
    check_value(divergence, decision["first_divergence"], "first ID divergence")
    check_value(new_repetition, decision["new_repetition"], "new repetition")
    check_value(dense["score"]["passed"], decision["dense_passed"], "dense score")
    check_value(sparse["score"]["passed"], decision["candidate_passed"], "candidate score")
    deltas = [b - a for a, b in zip(decision["projection_calls_before"],
                                  decision["projection_calls_after"], strict=True)]
    if len(deltas) != 3 or min(deltas) < len(case["prompt_ids"]):
        raise ValueError(f"sparse projection missing in {case['id']}")
    answer_stop = "new_repetition" if new_repetition else None
    if answer_stop is None and decision["dense_passed"] and not decision["candidate_passed"]:
        answer_stop = "lost_dense_pass"
    steps = sorted({0, min(8, len(dense_ids) - 1), len(dense_ids) - 1})
    for index, step in enumerate(steps):
        prefix = directory / f"{case['id']}-{step}"
        try:
            reference = read_logits(prefix.with_name(f"{prefix.name}-dense.f32le"), VOCABULARY)
            candidate = read_logits(prefix.with_name(f"{prefix.name}-candidate.f32le"), VOCABULARY)
        except ValueError as error:
            if "non-finite logit" not in str(error) or len(vectors) != index:
                raise
            reason = "nonfinite_reference" if "-dense.f32le" in str(error) else "nonfinite_candidate"
            return decision["dense_passed"], answer_stop or reason, index
        row = vectors[index]
        check_value("sparse_q4_extended_logits", row["kind"], "observation kind")
        check_value(step, row["step"], "continuation offset")
        check_value(dense_ids[step], row["target"], "dense-history target")
        check_value(len(case["prompt_ids"]) + step, row["prefix_tokens"], "matched prefix length")
        check_value(compare(reference, candidate, dense_ids[step]), row["values"], "FP32 replay")
    check_value(len(steps), len(vectors), "finite observation coverage")
    return decision["dense_passed"], answer_stop, len(steps)


def replay(report: Path, manifest: Path) -> dict[str, object]:
    """Verify a completed arm, including exact stopping point and release.

    Args:
        report: Complete JSONL backed by an adjacent .logits directory.
        manifest: Original measured tokenization manifest.

    Returns:
        Replayed outcome and number of independently checked FP32 comparisons.

    Raises:
        ValueError: Case accounting, raw logits or the gate decision disagree.
    """
    data = manifest.read_bytes()
    check_value(INPUT_SHA, hashlib.sha256(data).hexdigest(), "input identity")
    cases: list[Case] = json.loads(data)["cases"]
    rows: list[Record] = [json.loads(line) for line in report.read_text().splitlines()]
    check_value("sparse_q4_quality_manifest", rows[0]["kind"], "run manifest")
    check_value("sparse_q4_extended_inputs", rows[1]["kind"], "checkpoint input validation")
    check_value("sparse_q4_extended_complete", rows[-1]["kind"], "completion")
    check_value(0, rows[-1]["device_bytes_after"], "tracked device bytes at release")
    check_value([0, 17, 35], rows[-1]["candidate_layout"]["layers"], "selected down layers")
    groups = [(key, list(group)) for key, group in itertools.groupby(rows[2:-1], key=lambda row: row["case"])]
    counts = [[0, 0], [0, 0]]
    observations = 0
    failure = None
    for index, (key, events) in enumerate(groups):
        case = cases[index]
        check_value(case["id"], key, "frozen case order")
        passed, stop, count = replay_case(case, events, report.with_suffix(".logits"))
        counts[not passed][case["context"] == "long"] += 1
        observations += count
        if stop is not None:
            check_value(index + 1, len(groups), "continued past frozen stop")
            failure = [key, stop]
    if failure is None:
        check_value(len(cases), len(groups), "completed case count")
    if failure and failure[1] != "nonfinite_reference":
        disposition = "FAIL"
    elif failure or 0 in counts[0]:
        disposition = "INCONCLUSIVE"
    else:
        disposition = "PASS"
    result = {"disposition": disposition, "failure": failure, "cases": len(groups),
              "dense_passes_short_long": counts[0], "dense_failures_short_long": counts[1]}
    check_value(result, rows[-1]["summary"], "frozen disposition")
    return {**result, "replayed_fp32_comparisons": observations}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.report, args.manifest), indent=2, sort_keys=True))
