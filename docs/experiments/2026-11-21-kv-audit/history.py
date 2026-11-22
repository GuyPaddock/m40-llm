"""Replay the four KV boundary cases from the pinned prompts and FP32 device readbacks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal, TypedDict

from replay import check_value, compare, read_logits

INPUT_HASH = "e793aaae9aa3c6d5d281a67b028ae94ae3306d574957d48200c2f6ee74f781c2"
ARMS = ("dense", "full_mixed", "top64", "top8")
PAIRS = (("dense", "full_mixed"), ("dense", "top64"), ("dense", "top8"),
         ("full_mixed", "top64"), ("full_mixed", "top8"))


class Rubric(TypedDict, total=False):
    kind: Literal["exact", "json_object"]
    answers: list[str]
    expected: dict[str, bool | int]


class Geometry(TypedDict):
    block_size: int
    configured_old_block_budget: int
    old_blocks: int
    consumed_tokens: int
    capacity: int
    allocated_bytes: int
    old_tokens: int
    physical_lengths: list[int]
    is_compressed: bool
    all_old_blocks_fit_top64: bool


def passed(output: str, rubric: Rubric) -> bool:
    """Score the frozen exact-answer and typed JSON rubrics on the emitted text."""
    if rubric["kind"] == "exact":
        text = " ".join(output.split()).lower()
        return text in {" ".join(answer.split()).lower() for answer in rubric["answers"]}
    if rubric["kind"] != "json_object":
        raise ValueError("unknown boundary rubric")
    body = output.strip()
    if body.startswith("```") and body.endswith("```") and "\n" in body:
        body = body.split("\n", 1)[1][:-3].strip()
    try:
        actual = json.loads(body)
    except ValueError:
        return False
    expected = rubric["expected"]
    return (isinstance(actual, dict) and actual.keys() == expected.keys()
            and all(type(actual[k]) is type(v) and actual[k] == v for k, v in expected.items()))


def check_cache(state: Geometry, arm: str, consumed: int, control: bool) -> None:
    """Check selected history coverage and the device-read sequence length in every layer."""
    size = 64 if arm == "full_mixed" else 32
    old = max(0, consumed - 1024)
    blocks = (old + size - 1) // size
    budget = min(8 if arm == "top8" else 64, blocks) if arm != "dense" else blocks
    for field, expected in (("old_blocks", blocks), ("old_tokens", old), ("capacity", 4096),
                            ("consumed_tokens", consumed), ("physical_lengths", [consumed] * 36),
                            ("block_size", size), ("configured_old_block_budget", budget),
                            ("is_compressed", arm != "dense"), ("all_old_blocks_fit_top64", blocks <= 64)):
        check_value(expected, state[field], field)
    check_value(control or arm == "full_mixed", blocks <= 64, f"{arm} boundary regime")


class Task(TypedDict):
    scoring: Rubric


class Case(TypedDict):
    id: str
    task: Task
    prompt_ids: list[int]


class CaseRow(TypedDict, total=False):
    kind: str
    arm: str
    reference: str
    candidate: str
    step: int
    steps: list[int]
    targets: list[int]
    exact: bool
    logits: dict[str, object]
    ids: list[int]
    metrics: dict[str, object]
    cache: Geometry
    output: str
    score: dict[str, object]


def audit_case(case: Case, rows: list[CaseRow], directory: Path) -> tuple[dict[str, object], set[str]]:
    """Check one complete four-arm case against the frozen rubric and dense trajectory."""
    label = case["id"]
    control = label.endswith("-2048")
    filenames: set[str] = set()
    generations = {row["arm"]: row for row in rows if row["kind"] == "generation"}
    prefixes = {row["arm"]: row for row in rows if row["kind"] == "prefix"}
    comparisons = {(r["reference"], r["candidate"], r["step"]): r for r in rows if r["kind"] == "comparison"}
    dense_ids = generations["dense"]["ids"]
    steps = sorted({s for s in (0, 1, 8, 32, 64, 127, len(dense_ids)-1) if 0 <= s < len(dense_ids)})
    check_value(set(ARMS), set(generations), "generation arms")
    check_value(set(ARMS), set(prefixes), "prefix arms")
    check_value({(a,b,s) for a,b in PAIRS for s in steps}, set(comparisons), "comparison coverage")
    check_value(8 + len(comparisons), len(rows), "duplicate case records")
    outcomes = {}
    for arm in ARMS:
        gen, prefix = generations[arm], prefixes[arm]
        ids = gen["ids"]
        prompt_len = len(case["prompt_ids"])
        check_value(prompt_len, gen["metrics"]["prompt_tokens"], "prompt count")
        check_value(len(ids), gen["metrics"]["generated_tokens"], "output count")
        check_cache(gen["cache"], arm, prompt_len + len(ids) - 1, control)
        check_cache(prefix["cache"], arm, prompt_len + len(dense_ids) - 1, control)
        check_value(steps, prefix["steps"], "observation steps")
        check_value([dense_ids[s] for s in steps], prefix["targets"], "dense-history reference targets")
        correct = passed(gen["output"], case["task"]["scoring"])
        check_value(correct, gen["score"]["passed"], "answer score")
        first = next((i for i, (a,b) in enumerate(zip(ids, dense_ids)) if a != b),
                     None if len(ids) == len(dense_ids) else min(len(ids),len(dense_ids)))
        outcomes[arm] = {"passed": correct, "first_token_difference_from_dense": first,
                         "old_blocks": prefix["cache"]["old_blocks"],
                         "block_budget": prefix["cache"]["configured_old_block_budget"],
                         "cache_allocated_bytes": gen["cache"]["allocated_bytes"]}
    if control:
        check_value(generations["full_mixed"]["ids"], generations["top64"]["ids"], "2K generation")
        check_value(generations["full_mixed"]["output"], generations["top64"]["output"], "2K text")
    paired = {f"{a}/{b}": [] for a,b in PAIRS}
    for step in steps:
        vectors = {}
        for arm in ARMS:
            fname = f"{label}-{arm}-{step}.f32le"
            vectors[arm] = read_logits(directory / fname, 151936)
            filenames.add(fname)
        for a,b in PAIRS:
            measurement = comparisons[a,b,step]
            exact = vectors[a].tobytes() == vectors[b].tobytes()
            check_value(exact, measurement["exact"], "full FP32 byte parity")
            if control and (a,b) == ("full_mixed","top64"):
                check_value(True, exact, "2K FP32 equivalence control")
            result = compare(vectors[a], vectors[b], dense_ids[step])
            check_value(result, measurement["logits"], f"{label}/{a}/{b}/{step}")
            paired[f"{a}/{b}"].append({"step":step, "is_exact":exact, "kl":result["kl_reference_candidate"],
                                      "top10_overlap":result["top10_overlap"], "max_abs":result["max_abs_logit_diff"]})
    return {"case":label, "prompt_tokens":len(case["prompt_ids"]), "arms":outcomes, "pairs":paired}, filenames


def replay_history(report: Path, inputs: Path) -> dict[str, object]:
    """Require the completed cohort and reproduce every paired comparison on the dense trajectory."""
    data = inputs.read_bytes()
    check_value(INPUT_HASH, hashlib.sha256(data).hexdigest(), "boundary inputs")
    cases = json.loads(data)
    records = [json.loads(line) for line in report.read_text().splitlines()]
    manifest = records[0]
    for field, expected in (("kind", "manifest"), ("source", "f1bb7a4201ca1300af08282215d4bbb95620e8a9"),
                            ("model", "4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"),
                            ("inputs", INPUT_HASH), ("order", "chronological"), ("capacity", 4096)):
        check_value(expected, manifest[field], field)
    check_value({"kind": "complete", "cases": 4, "generations": 16, "initial_bytes": 0, "final_bytes": 0},
                records[-1], "boundary completion")
    check_value([(c["id"], "dense") for c in cases], [(r["case"], r["arm"]) for r in records[1:5]],
                "reference-first generation")
    directory = report.with_suffix(".logits")
    filenames: set[str] = set()
    rows_used = 2
    summaries = []
    for case in cases:
        rows = [row for row in records if row.get("case") == case["id"]]
        result, used = audit_case(case, rows, directory)
        summaries.append(result)
        filenames.update(used)
        rows_used += len(rows)
    check_value(rows_used, len(records), "unexpected/duplicate report rows")
    check_value(filenames, {p.name for p in directory.iterdir()}, "readback file set")
    return {"source":manifest["source"], "report_sha256":hashlib.sha256(report.read_bytes()).hexdigest(),
            "vectors":len(filenames), "cases":summaries, "tracked_bytes_after":records[-1]["final_bytes"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("inputs", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay_history(args.report, args.inputs), indent=2))
