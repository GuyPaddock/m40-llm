"""Reproduce paired idle-slot effects from the six M40 attention cohorts."""

import json
import random
import statistics
import sys
from pathlib import Path


def summarize(archive_rows: Path) -> list[dict[str, object]]:
    """Check the trial grid and calculate medians and paired event changes."""
    rows = [json.loads(line) for line in archive_rows.read_text().splitlines()]
    if rows[-1] != {"kind": "complete", "timing_rows": 30, "cases": 6}:
        raise ValueError("Incomplete timing report")
    arms = [(1, 1, 0), (2, 1, 0), (2, 1, 512), (2, 1, 2048),
            (2, 1, 4096), (2, 2, 2048)]
    timings = {}
    for arm in arms:
        matching = [row for row in rows if
                    (row.get("slots"), row.get("active_rows"), row.get("second_tokens")) == arm]
        parity = [row for row in matching if row["kind"] == "parity"]
        if len(parity) != 1 or parity[0]["exact_first_row_values"] != 2048:
            raise ValueError(f"Missing first-row parity for {arm}")
        trials = sorted((row for row in matching if row["kind"] == "timing"),
                        key=lambda row: row["trial"])
        if [row["trial"] for row in trials] != list(range(5)):
            raise ValueError(f"Missing or repeated round for {arm}")
        if any(row["repeats"] != 16 for row in trials):
            raise ValueError("Unexpected count of launches per interval")
        timings[arm] = trials
    control = timings[(2, 1, 0)]
    summaries = []
    for arm, trials in timings.items():
        events = [row["event_ms"] for row in trials]
        median = statistics.median(events)
        changes = [100 * (row["event_ms"] / ref["event_ms"] - 1)
                   for row, ref in zip(trials, control, strict=True)]
        rng = random.Random(0)
        bootstrap = sorted(statistics.median(rng.choices(changes, k=5)) for _ in range(10000))
        summaries.append({
            "slots_active_other_tokens": arm, "event_median_ms": median,
            "event_mad_ms": statistics.median(abs(value - median) for value in events),
            "wall_median_ms": statistics.median(row["wall_ms"] for row in trials),
            "paired_event_change_percent": statistics.median(changes),
            "bootstrap_95_percent": [bootstrap[249], bootstrap[9749]],
        })
    return summaries


if __name__ == "__main__":
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2))
