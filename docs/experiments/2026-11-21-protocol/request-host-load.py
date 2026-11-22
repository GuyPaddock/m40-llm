"""Summarize host monitor samples between a benchmark's before/after snapshots."""

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


VM_COUNTER = re.compile(r"^vm pid=(\d+) start_ticks=(\d+) cpu_ticks=(\d+) ticks_per_second=(\d+)$")


def timestamp(text: str) -> datetime:
    """Parse a timestamp from the host sampler or the GPU snapshot."""
    result = datetime.fromisoformat(text.replace(",", "."))
    if result.tzinfo is None:
        raise ValueError(f"Missing UTC offset in {text}")
    return result


def summarize_vm(
    counters: dict[tuple[int, int], list[tuple[int, datetime, int, int]]],
) -> list[dict[str, int | float]]:
    """Convert sampled cumulative process ticks into observed logical CPU use.

    Args:
        counters: Per-lifetime PID/start-tick records containing sample index,
            wall timestamp, cumulative CPU ticks and ticks per second.

    Returns:
        Per-lifetime adjacent-interval counts, covered time and CPU rates.
    """
    results: list[dict[str, int | float]] = []
    for (pid, start_ticks), observed in sorted(counters.items()):
        rates: list[float] = []
        cpu_seconds = 0.0
        wall_seconds = 0.0
        for (first, start, ticks, hz), (last, end, later, later_hz) in zip(observed, observed[1:]):
            if last != first + 1 or hz != later_hz or hz <= 0 or later < ticks:
                continue
            duration = (end - start).total_seconds()
            consumed = (later - ticks) / hz
            cpu_seconds += consumed
            wall_seconds += duration
            rates.append(consumed / duration)
        if rates:
            results.append({"pid": pid, "start_ticks": start_ticks,
                            "intervals": len(rates), "covered_seconds": wall_seconds,
                            "mean_logical_cpus": cpu_seconds / wall_seconds,
                            "peak_interval_logical_cpus": max(rates)})
    return results


def summarize(directory: Path, label: str) -> dict[str, object]:
    """Report observed process states and CPU/IO pressure inside one interval."""
    start, end = (
        timestamp((directory / f"{label}.{suffix}").read_text().splitlines()[0])
        for suffix in ("before", "after")
    )
    if end <= start:
        raise ValueError(f"Invalid snapshot interval for {label}")
    process_states: Counter[str] = Counter()
    maxima: dict[str, float] = {}
    clocks: list[datetime] = []
    vm_counters: dict[tuple[int, int], list[tuple[int, datetime, int, int]]] = {}
    selected = False
    with (directory / "host-load.log").open() as log:
        for line in log:
            if re.match(r"^\d{4}-", line):
                clock = timestamp(line.strip())
                selected = start <= clock <= end
                if selected:
                    if clocks and clock <= clocks[-1]:
                        raise ValueError(f"Host sample clock reversed during {label}")
                    clocks.append(clock)
                continue
            if selected and " count=" in line:
                group, count = line.strip().rsplit(" count=", 1)
                if int(count) > 0:
                    process_states[group] += 1
            if selected and (match := re.match(r"^(cpu|io|memory) some avg10=([\d.]+)", line)):
                resource, value = match.groups()
                maxima[resource] = max(maxima.get(resource, 0.0), float(value))
            if selected and (match := VM_COUNTER.fullmatch(line.strip())):
                pid, start_ticks, cpu_ticks, hz = map(int, match.groups())
                vm_counters.setdefault((pid, start_ticks), []).append(
                    (len(clocks), clocks[-1], cpu_ticks, hz)
                )
    boundaries = [start, *clocks, end]
    return {
        "request": label,
        "sample_count": len(clocks),
        "maximum_unsampled_seconds": max(
            (b - a).total_seconds() for a, b in zip(boundaries, boundaries[1:])
        ),
        "observations_per_process_group": dict(sorted(process_states.items())),
        "peak_some_pressure_avg10_percent": maxima,
        "vm_cpu": summarize_vm(vm_counters),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("labels", nargs="+")
    args = parser.parse_args()
    print(json.dumps([summarize(args.directory, label) for label in args.labels], indent=2))
