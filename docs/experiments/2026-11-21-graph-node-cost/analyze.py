"""Summarize paired graph timing, correcting for the ordinary capacity effect."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections.abc import Iterable
from pathlib import Path
from statistics import median

SCENARIOS = ((128, 65), (7, 65), (128, 2048))
METRICS = ("event_span_us", "wall_us", "enqueue_us")
ARMS = ((8192, False), (8192, True), (8193, False), (8193, True))


def setup_measurements(lines: Iterable[str]) -> list[dict[str, int | float]]:
    """Read actual node counts and lifecycle intervals from consecutive capture records.

    Args:
        lines: Diagnostic stderr, retaining capture and instantiation order.

    Returns:
        Observations for both capacities and all three head/length choices.

    Raises:
        ValueError: A capture, node count or query status violates the fixed design.
    """
    pending: tuple[int, int, int] | None = None
    seen: set[tuple[int, int, int]] = set()
    result: list[dict[str, int | float]] = []
    for line in lines:
        start = re.fullmatch(r"capture head=(\d+) length=(\d+) capacity=(\d+)", line)
        finish = re.fullmatch(
            r"\[graph-setup\] capture_us=([\d.]+) instantiate_us=([\d.]+) nodes=(\d+) query_status=(\d+)", line
        )
        if start:
            if pending is not None:
                raise ValueError("missing capture result")
            pending = (int(start[1]), int(start[2]), int(start[3]))
        if finish:
            if pending is None or pending in seen:
                raise ValueError("unexpected capture result")
            head, length, capacity = pending
            nodes = int(finish[3])
            if nodes != 1 + int(head == 128 and capacity == 8193) or finish[4] != "0":
                raise ValueError(f"unexpected graph structure at {pending}")
            result.append(dict(head=head, length=length, capacity=capacity, nodes=nodes,
                               capture_us=float(finish[1]), instantiate_us=float(finish[2])))
            seen.add(pending)
            pending = None
    if pending is not None or seen != {(h, n, c) for h, n in SCENARIOS for c in (8192, 8193)}:
        raise ValueError("incomplete graph lifecycle evidence")
    return result


def effect(values: list[float]) -> dict[str, object]:
    """Calculate a timing difference and paired median interval in microseconds.

    Args:
        values: One capacity difference or difference of differences per round.

    Returns:
        Both screen medians and the seeded 10,000-resample interval on ten rounds.
    """
    rng = random.Random(0)
    draws = sorted(median(rng.choices(values, k=len(values))) for _ in range(10_000))
    return dict(median_us=median(values), first_five_us=median(values[:5]),
                second_five_us=median(values[5:]), paired_ci95_us=[draws[249], draws[9749]])


def summarize(report: Path, log: Path) -> dict[str, object]:
    """Check timing coverage and derive within-round graph and ordinary effects.

    Args:
        report: Original release benchmark JSONL.
        log: Corresponding capture and setup output.

    Returns:
        Setup costs, measured distributions and capacity effects for each scenario.

    Raises:
        ValueError: Timing, parity or terminal evidence is missing or invalid.
    """
    setup = setup_measurements(log.read_text().splitlines())
    rows = [json.loads(line) for line in report.read_text().splitlines()]
    if rows[-1].get("kind") != "complete" or rows[-1].get("timings") != 120:
        raise ValueError("missing completion record")
    timings = [r for r in rows if r["kind"] == "timing"]
    checks = [r for r in rows if r["kind"] == "verified"]
    if len(timings) != 120 or len(checks) != 6:
        raise ValueError("incomplete measurements or parity")
    summaries = []
    for head, length in SCENARIOS:
        parity = [r for r in checks if (r["head_dim"], r["live_length"]) == (head, length)]
        if sorted(r["capacity"] for r in parity) != [8192, 8193] or any(
            r["exact_outputs"] != head * 16 for r in parity
        ):
            raise ValueError("missing exact-output control")
        slots = {(r["round"], r["capacity"], r["graph"]): r["timing"] for r in timings
                 if (r["head_dim"], r["live_length"]) == (head, length)}
        if set(slots) != {(i, c, g) for i in range(10) for c, g in ARMS}:
            raise ValueError("missing or duplicate paired arm")
        if not all(t["iterations"] == 100 and all(t[k] > 0 and math.isfinite(t[k]) for k in METRICS)
                   for t in slots.values()):
            raise ValueError("invalid interval")
        for metric in METRICS:
            ordinary, graph = (
                [slots[i, 8193, g][metric] - slots[i, 8192, g][metric] for i in range(10)]
                for g in (False, True)
            )
            summaries.append(dict(head=head, length=length, metric=metric,
                arm_medians_us={f"{c}-{'graph' if g else 'ordinary'}":
                                median(slots[i, c, g][metric] for i in range(10)) for c, g in ARMS},
                ordinary_capacity_effect=effect(ordinary), graph_capacity_effect=effect(graph),
                graph_minus_ordinary_effect=effect([g-o for g, o in zip(graph, ordinary, strict=True)])))
    return dict(captures=setup, paired_measurements=summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.report, args.log), indent=2))
