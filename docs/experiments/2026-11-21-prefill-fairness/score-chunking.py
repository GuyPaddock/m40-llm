"""Apply the frozen dense prefill latency and throughput gates to paired trials."""

import argparse
import json
import random
from pathlib import Path
from statistics import median
from typing import TypedDict


class Timing(TypedDict):
    """Client endpoints and token metrics in milliseconds."""

    max_itl: float
    median_itl: float
    p95_itl: float
    medium_ttft: float
    primary_wall: float
    timing_wall: float
    during_decode: bool


def timing(directory: Path, pair: int, variant: str) -> Timing:
    """Recover the active decode window and the workload's latency endpoints."""
    name = f"{pair}-{variant}"
    diagnostic = json.loads((directory / f"diagnostic-{name}.json").read_text())
    clocks = json.loads((directory / f"diagnostic-{name}.clocks.json").read_text())
    replay = json.loads((directory / f"primary-{name}.replay.json").read_text())
    short, medium = diagnostic["records"]
    short_metrics = short["response"]["metrics"]
    if not replay["work_matches"] or replay["primary"]["generated_tokens"] != 160:
        raise ValueError(f"Workload mismatch in {name}")
    anchor = (short_metrics["sample_origin_unix_ns"] - int(diagnostic["client_origin_unix_ns"])) / 1e6
    samples = short_metrics["token_elapsed_ms"]
    if len(samples) != 128 or not samples[0] <= samples[-1]:
        raise ValueError(f"Invalid short decode window in {name}")
    short_clock, medium_clock = clocks["requests"]
    return Timing(
        max_itl=short_clock["itl_max_ms"],
        median_itl=short_clock["itl_median_ms"],
        p95_itl=short_clock["itl_p95_ms"],
        medium_ttft=medium_clock["client_to_first_sample_ms"],
        primary_wall=replay["primary"]["makespan_ms"],
        timing_wall=replay["diagnostic"]["makespan_ms"],
        during_decode=anchor + samples[0] < medium["submitted_ms"] < anchor + samples[-1],
    )


def paired_interval(reductions: list[float]) -> tuple[float, float]:
    """Bootstrap the median paired reduction with the shared trial count and seed."""
    rng = random.Random(0)
    values = sorted(median(rng.choices(reductions, k=len(reductions))) for _ in range(10_000))
    return values[249], values[9749]


def score(directory: Path) -> dict[str, object]:
    """Check the complete cohort and report each threshold and its evidence."""
    if not (directory / "complete.txt").stat().st_size:
        raise ValueError("Missing completion snapshot")
    trials: list[dict[str, object]] = []
    reductions: list[float] = []
    baseline_wall: list[float] = []
    candidate_wall: list[float] = []
    baseline_ttft: list[float] = []
    candidate_ttft: list[float] = []
    for i in range(5):
        baseline = timing(directory, i, "baseline")
        candidate = timing(directory, i, "candidate")
        valid = baseline["during_decode"] and candidate["during_decode"]
        reduction = 1 - candidate["max_itl"] / baseline["max_itl"]
        trials.append(dict(pair=i, baseline=baseline, candidate=candidate,
                           valid=valid, max_itl_reduction=reduction))
        if valid:
            reductions.append(reduction)
            baseline_wall.append(baseline["primary_wall"])
            candidate_wall.append(candidate["primary_wall"])
            baseline_ttft.append(baseline["medium_ttft"])
            candidate_ttft.append(candidate["medium_ttft"])

    if len(reductions) < 4:
        return dict(decision="INCONCLUSIVE", valid_pairs=len(reductions), trials=trials)
    interval = paired_interval(reductions)
    passing = sum(reduction >= 0.80 for reduction in reductions)
    wall_ratio = median(candidate_wall) / median(baseline_wall)
    ttft_ratio = median(candidate_ttft) / median(baseline_ttft)
    admitted = passing >= 4 and interval[0] > 0 and wall_ratio <= 1.05 and ttft_ratio <= 1.05
    return dict(
        decision="PASS" if admitted else "FAIL",
        valid_pairs=len(reductions),
        pairs_reducing_maximum_itl_by_80_percent=passing,
        bootstrap_max_itl_reduction_ci95=interval,
        primary_median_makespan_ratio=wall_ratio,
        diagnostic_medium_median_ttft_ratio=ttft_ratio,
        trials=trials,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(score(args.directory), indent=2))
