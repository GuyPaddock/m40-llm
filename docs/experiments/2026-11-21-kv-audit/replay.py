"""Check the November KV report against independent little-endian FP32 readbacks."""

from __future__ import annotations

import argparse
import array
import hashlib
import heapq
import json
import math
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import TypedDict


class Prediction(TypedDict):
    target_logit: float
    target_rank: int
    nll: float
    top_indices: list[int]


class Comparison(TypedDict):
    reference: Prediction
    candidate: Prediction
    max_abs_logit_diff: float
    mean_abs_logit_diff: float
    kl_reference_candidate: float
    top10_overlap: int


def read_logits(path: Path, vocabulary: int) -> array.array[float]:
    """Read a finite vocabulary-sized FP32 vector in token ID order."""
    data = path.read_bytes()
    if len(data) != vocabulary * 4:
        raise ValueError(f"wrong logit size for {path}")
    values = array.array("f")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    if not all(map(math.isfinite, values)):
        raise ValueError(f"non-finite logit in {path}")
    return values


def predict(values: array.array[float], target: int) -> tuple[Prediction, float]:
    """Score the dense-history target using a stable FP64 log partition."""
    if not 0 <= target < len(values):
        raise ValueError("invalid target token ID")
    maximum = max(values)
    log_sum = math.log(math.fsum(math.exp(x - maximum) for x in values))
    top = heapq.nlargest(10, range(len(values)), key=lambda i: (values[i], -i))
    observed = values[target]
    rank = 1 + sum(v > observed or (v == observed and i < target)
                   for i, v in enumerate(values))
    return Prediction(target_logit=observed, target_rank=rank,
                      nll=(maximum - observed) + log_sum, top_indices=top), maximum + log_sum


def compare(dense: array.array[float], candidate: array.array[float], target: int) -> Comparison:
    """Recompute KL from expected raw logit differences and log partitions."""
    if len(dense) != len(candidate):
        raise ValueError("vocabulary changed between arms")
    reference, log_z_dense = predict(dense, target)
    alternate, log_z_candidate = predict(candidate, target)
    weighted = math.fsum(math.exp(a-log_z_dense) * (a-b) for a,b in zip(dense,candidate))
    absolute = [abs(a-b) for a,b in zip(dense,candidate)]
    return Comparison(reference=reference, candidate=alternate,
                      max_abs_logit_diff=max(absolute), mean_abs_logit_diff=math.fsum(absolute)/len(absolute),
                      kl_reference_candidate=max(0., weighted + (log_z_candidate-log_z_dense)),
                      top10_overlap=len(set(reference["top_indices"]) & set(alternate["top_indices"])))


def check_value(expected: object, actual: object, field: str) -> None:
    """Reject changed counts, indices and floating errors beyond FP64 summation noise."""
    if isinstance(expected, dict) and isinstance(actual, dict) and expected.keys() == actual.keys():
        for key, value in expected.items():
            check_value(value, actual[key], f"{field}.{key}")
    elif isinstance(expected, float) and isinstance(actual, (float, int)):
        if field.endswith(".target_logit"):
            if struct.pack("<f", expected) != struct.pack("<f", actual):
                raise ValueError(f"{field}: target FP32 bits differ")
            return
        if not math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-10):
            raise ValueError(f"{field}: {expected} != {actual}")
    elif expected != actual:
        raise ValueError(f"{field}: {expected} != {actual}")


def replay(report: Path, inputs: Path) -> Counter[str]:
    """Check complete case/arm coverage and replay every frozen observation.

    Args:
        report: The original JSONL with its adjacent .logits directory.
        inputs: Exact November JSON used for generation and reference prompts.
    """
    content = inputs.read_bytes()
    pinned = "18d0c6f69dbdfa3bf976780b5b20e14d06dfec5f32980418a9d4f308da9f1cc3"
    check_value(pinned, hashlib.sha256(content).hexdigest(), "input SHA-256")
    cases = {case["id"]: case for case in json.loads(content)["cases"]}
    rows = [json.loads(line) for line in report.read_text().splitlines()]
    check_value("kv_quality_manifest", rows[0]["kind"], "manifest")
    check_value(pinned, rows[0]["inputs_sha256"], "run input SHA-256")
    check_value(json.loads(content), rows[0]["inputs"], "embedded cases")
    check_value("kv_quality_complete", rows[-1]["kind"], "completion")
    check_value(48, len(cases), "case count")
    check_value(48, rows[-1]["cases"], "complete cases")
    generations = {(row["case"], row["arm"]): row for row in rows if row["kind"] == "kv_quality_generation"}
    check_value(144, len(generations), "distinct generations")
    check_value(144, sum(row["kind"] == "kv_quality_generation" for row in rows), "generation rows")
    directory = report.with_suffix(".logits")
    arms = {"quantized_all": "QuantizedAll", "quantized_top8": "QuantizedTop8"}
    pending = {(case, arm) for case in cases for arm in arms}
    file_names: set[str] = set()
    counts: Counter[str] = Counter()
    for row in rows:
        if row["kind"] != "kv_quality_comparison":
            continue
        case, arm = row["case"], row["arm"]
        pending.remove((case,arm))
        dense = generations[case,"dense"]
        candidate = generations[case,arm]
        tokens = dense["generated_ids"]
        other = candidate["generated_ids"]
        check_value(tokens == other, row["identical_generated_ids"], case)
        first = next((i for i,(a,b) in enumerate(zip(tokens,other)) if a != b),
                     None if len(tokens) == len(other) else min(len(tokens),len(other)))
        check_value(first, row["first_generated_difference"], case)
        verdict = ("inconclusive" if not dense["score"]["passed"]
                   else "preserved" if candidate["score"]["passed"] else "regression")
        check_value(verdict, row["quality_conclusion"], case)
        counts[f"{arm}/{row['context']}/{verdict}"] += 1
        steps = sorted({s for s in (0,1,8,32,64,127,len(tokens)-1) if 0 <= s < len(tokens)})
        observations = row["prefix_observations"]
        check_value(steps, [item["step"] for item in observations], case)
        for item in observations:
            step = item["step"]
            check_value(len(cases[case]["prompt_ids"])+step, item["prefix_tokens"], case)
            check_value(tokens[step], item["target"], case)
            ref = directory / f"{case}-Dense-{step}.f32le"
            alt = directory / f"{case}-{arms[arm]}-{step}.f32le"
            file_names.update((ref.name, alt.name))
            metrics = compare(read_logits(ref,151936), read_logits(alt,151936), tokens[step])
            check_value(metrics, item["logits"], f"{case}/{arm}/{step}")
            counts["observations"] += 1
    if pending or file_names != {p.name for p in directory.iterdir()}:
        raise ValueError("missing comparison or unexpected logit artifact")
    counts["verified_vectors"] = len(file_names)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("inputs", type=Path)
    options = parser.parse_args()
    print(json.dumps(replay(options.report, options.inputs), indent=2, sort_keys=True))
